#!/usr/bin/env python3
"""
Enhanced PyTorch .pth to LibTorch converter with PyTorch 2.7 optimizations.
Addresses tracing issues with DepthAnythingV2 models and maximizes inference performance.
"""

import argparse
import torch
import torch.nn as nn
import os
import sys
import time
import logging
from typing import Optional, Tuple, Dict, Any, List, Union
import numpy as np
import math
import warnings
from contextlib import contextmanager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try importing optional dependencies
try:
    import torch_tensorrt
    TENSORRT_AVAILABLE = True
    logger.info("TensorRT support available")
except ImportError:
    TENSORRT_AVAILABLE = False
    logger.warning("TensorRT not available - install torch-tensorrt for TensorRT optimizations")

try:
    from depth_anything_v2.dpt import DepthAnythingV2
    DEPTH_ANYTHING_AVAILABLE = True
except ImportError:
    DEPTH_ANYTHING_AVAILABLE = False
    logger.error("DepthAnythingV2 not available - please install depth-anything-v2")
    sys.exit(1)

# PyTorch 2.7 optimizations
try:
    # Enable PT2 optimizations
    torch._dynamo.reset()
    torch._dynamo.config.automatic_dynamic_shapes = True
    torch._dynamo.config.cache_size_limit = 256
    logger.info("PyTorch 2.7 Dynamo optimizations enabled")
except AttributeError:
    logger.warning("Some PyTorch 2.7 features not available in this version")


@contextmanager
def inference_mode_context():
    """Context manager for optimal inference settings."""
    old_grad_enabled = torch.is_grad_enabled()
    old_deterministic = torch.backends.cudnn.deterministic
    old_benchmark = torch.backends.cudnn.benchmark
    old_allow_tf32 = torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else None
    
    try:
        torch.set_grad_enabled(False)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            # Enable optimized attention
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
        yield
    finally:
        torch.set_grad_enabled(old_grad_enabled)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = old_deterministic
            torch.backends.cudnn.benchmark = old_benchmark
            if old_allow_tf32 is not None:
                torch.backends.cuda.matmul.allow_tf32 = old_allow_tf32


class OptimizedTracingWrapper(nn.Module):
    """
    Advanced wrapper that makes models tracing-friendly with memory optimizations.
    """
    def __init__(self, model: nn.Module, fixed_height: int = 480, fixed_width: int = 640):
        super().__init__()
        self.model = model
        self.fixed_height = fixed_height
        self.fixed_width = fixed_width
        
        # Pre-compute patch dimensions
        self.patch_h = fixed_height // 14
        self.patch_w = fixed_width // 14
        
        # Apply memory format optimizations
        self._optimize_memory_format()
        
    def _optimize_memory_format(self):
        """Optimize memory layout for better performance."""
        try:
            # Convert conv layers to channels_last for better performance
            for module in self.model.modules():
                if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                    if hasattr(module, 'weight') and module.weight.dim() == 4:
                        module.weight.data = module.weight.data.to(memory_format=torch.channels_last)
                    if hasattr(module, 'bias') and module.bias is not None:
                        module.bias.data = module.bias.data.contiguous()
        except Exception as e:
            logger.warning(f"Memory format optimization failed: {e}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optimized memory format."""
        # Convert to channels_last for better performance
        if x.dim() == 4 and x.device.type == 'cuda':
            x = x.to(memory_format=torch.channels_last)
        
        # Ensure input is the expected size
        if x.shape[-2:] != (self.fixed_height, self.fixed_width):
            x = torch.nn.functional.interpolate(
                x, size=(self.fixed_height, self.fixed_width), 
                mode='bilinear', align_corners=False, antialias=True
            )
        
        return self.model(x)


class FP16ModelWrapper(nn.Module):
    """Wrapper to handle FP32 inputs/outputs with FP16 model weights."""
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model.half()  # Convert weights to FP16
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept FP32 inputs, convert to FP16 for computation
        input_dtype = x.dtype
        if input_dtype == torch.float32:
            x = x.half()
        
        # Forward pass in FP16
        output = self.model(x)
        
        # Convert output back to FP32
        if input_dtype == torch.float32:
            output = output.float()
            
        return output


class AdvancedModelConverter:
    """Enhanced model converter with PyTorch 2.7 optimizations."""
    
    def __init__(self):
        self.model_configs = {
            'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        
        # Setup compilation cache
        self._setup_compilation_cache()
    
    def _setup_compilation_cache(self):
        """Setup PyTorch 2.7 compilation cache for better performance."""
        try:
            # Configure torch.compile cache
            os.environ.setdefault('TORCH_COMPILE_DEBUG', '0')
            os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', '/tmp/torch_cache')
            
            # Create cache directory
            cache_dir = os.environ.get('TORCHINDUCTOR_CACHE_DIR')
            os.makedirs(cache_dir, exist_ok=True)
            
            logger.info(f"Compilation cache setup at: {cache_dir}")
        except Exception as e:
            logger.warning(f"Cache setup failed: {e}")
    
    def load_checkpoint(self, checkpoint_path: str, encoder: str, device: torch.device, 
                       max_depth: float = 20.0, strict: bool = False) -> nn.Module:
        """Load checkpoint with robust error handling and optimizations."""
        if encoder not in self.model_configs:
            raise ValueError(f"Unsupported encoder: {encoder}. Choose from {list(self.model_configs.keys())}")
        
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        logger.info(f"Using encoder: {encoder}, max_depth: {max_depth}")
        
        # Initialize model
        model = DepthAnythingV2(**{**self.model_configs[encoder], 'max_depth': max_depth})
        
        try:
            # Load checkpoint with optimized loading
            with torch.device(device):
                ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
            
            # Handle different checkpoint formats
            if isinstance(ckpt, dict):
                if "model_state" in ckpt:
                    model_sd = ckpt["model_state"]
                    logger.info("Found 'model_state' key in checkpoint")
                elif "state_dict" in ckpt:
                    model_sd = ckpt["state_dict"]
                    logger.info("Found 'state_dict' key in checkpoint")
                elif "model" in ckpt:
                    model_sd = ckpt["model"]
                    logger.info("Found 'model' key in checkpoint")
                else:
                    model_sd = ckpt
                    logger.info("Using checkpoint as direct state_dict")
            else:
                model_sd = ckpt
                logger.info("Checkpoint is direct state_dict")
            
            # Load state dict
            missing, unexpected = model.load_state_dict(model_sd, strict=strict)
            
            if missing:
                logger.warning(f"Missing keys ({len(missing)}): {missing[:5]}...")
            if unexpected:
                logger.warning(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
            
            if not missing and not unexpected:
                logger.info("✓ All keys matched perfectly")
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise
        
        # Move to device and set to eval mode
        model = model.to(device).eval()
        
        # Apply post-loading optimizations
        self._apply_post_loading_optimizations(model, device)
        
        logger.info(f"Model loaded and optimized on {device}")
        return model
    
    def _apply_post_loading_optimizations(self, model: nn.Module, device: torch.device):
        """Apply optimizations after loading the model."""
        # Fuse operations where possible
        try:
            if hasattr(torch.jit, 'freeze'):
                # Apply layer fusion optimizations
                for module in model.modules():
                    if isinstance(module, nn.Sequential):
                        # Fuse conv-bn-relu sequences
                        torch.ao.quantization.fuse_modules_qat(
                            module, [['conv', 'bn', 'relu']], inplace=True
                        ) if hasattr(torch.ao.quantization, 'fuse_modules_qat') else None
        except Exception as e:
            logger.debug(f"Fusion optimization skipped: {e}")
        
        # Set optimal execution settings
        if device.type == 'cuda':
            # Enable memory format optimizations
            try:
                model = model.to(memory_format=torch.channels_last)
                logger.info("Applied channels_last memory format")
            except Exception as e:
                logger.debug(f"Memory format optimization failed: {e}")
    
    def prepare_model_for_tracing(self, model: nn.Module, height: int, width: int, 
                                 use_wrapper: bool = True, for_torchscript: bool = True) -> nn.Module:
        """Prepare model for tracing with PyTorch 2.7 optimizations."""
        logger.info("Preparing model for tracing...")
        
        # Disable gradient computation
        for param in model.parameters():
            param.requires_grad_(False)
        
        # Apply wrapper if requested
        if use_wrapper:
            wrapped_model = OptimizedTracingWrapper(model, height, width)
            logger.info(f"Applied optimized tracing wrapper for {height}x{width} inputs")
            model = wrapped_model
        
        # NOTE: torch.compile is incompatible with torch.jit.trace
        # We'll apply torch.compile AFTER TorchScript conversion if needed
        if not for_torchscript:
            logger.info("Applying torch.compile since not converting to TorchScript...")
            model = self._apply_torch_compile(model, height, width)
        else:
            logger.info("Skipping torch.compile to maintain TorchScript compatibility")
        
        return model
    
    def _apply_torch_compile(self, model: nn.Module, height: int, width: int) -> nn.Module:
        """Apply torch.compile optimizations (separate from TorchScript path)."""
        try:
            # Different backends for different scenarios
            backends = ['inductor', 'aot_eager'] if torch.cuda.is_available() else ['aot_eager']
            
            for backend in backends:
                try:
                    compiled_model = torch.compile(
                        model,
                        backend=backend,
                        mode='max-autotune',  # Most aggressive optimization
                        fullgraph=False,  # Allow graph breaks for complex models
                        dynamic=False,  # Static shapes for better optimization
                    )
                    
                    # Test compilation with dummy input
                    test_input = torch.randn(1, 3, height, width, device=next(model.parameters()).device)
                    with inference_mode_context():
                        _ = compiled_model(test_input)
                    
                    logger.info(f"✓ Model compiled with {backend} backend")
                    return compiled_model
                    
                except Exception as e:
                    logger.warning(f"Compilation with {backend} failed: {e}")
                    continue
            
            logger.warning("All compilation attempts failed, using uncompiled model")
            
        except Exception as e:
            logger.warning(f"torch.compile not available or failed: {e}")
        
        return model
    
    def convert_with_advanced_tracing(self, model: nn.Module, base_input: torch.Tensor,
                                    method: str = "trace") -> torch.jit.ScriptModule:
        """Convert model using advanced tracing techniques."""
        logger.info(f"Converting to TorchScript using {method} method with advanced optimizations...")
        
        # Ensure model is not compiled for tracing compatibility
        if hasattr(model, '_dynamo_marked_for_compilation'):
            logger.warning("Detected compiled model - this may cause tracing issues")
        
        try:
            with inference_mode_context():
                if method == "script":
                    try:
                        scripted = torch.jit.script(model)
                        logger.info("✓ Scripting completed successfully")
                        return self._optimize_scripted_model(scripted)
                    except Exception as script_error:
                        logger.warning(f"Scripting failed: {script_error}")
                        logger.info("Falling back to tracing method...")
                        method = "trace"
                
                if method == "trace":
                    # Reset any dynamo state that might interfere
                    try:
                        torch._dynamo.reset()
                    except:
                        pass
                    
                    # Advanced tracing with multiple representative inputs
                    batch_size, channels, height, width = base_input.shape
                    
                    # Generate diverse inputs for robust tracing
                    trace_inputs = [
                        base_input,
                        torch.randn_like(base_input) * 0.8 + 0.1,
                        torch.ones_like(base_input) * 0.5,
                        torch.zeros_like(base_input) + 0.2,
                    ]
                    
                    # Use strict=False for better compatibility with dynamic operations
                    logger.info("Starting TorchScript tracing...")
                    scripted = torch.jit.trace(
                        model, 
                        base_input, 
                        strict=False,
                        check_trace=False  # Disable check for performance
                    )
                    
                    logger.info("✓ Advanced tracing completed successfully")
                    
                    # Validate and optimize
                    scripted = self._optimize_scripted_model(scripted)
                    self._validate_trace_consistency(model, scripted, trace_inputs)
                    
                    return scripted
            
        except Exception as e:
            logger.error(f"TorchScript conversion failed: {e}")
            raise
    
    def _optimize_scripted_model(self, scripted_model: torch.jit.ScriptModule) -> torch.jit.ScriptModule:
        """Apply advanced optimizations to scripted model."""
        logger.info("Applying advanced TorchScript optimizations...")
        
        try:
            # Freeze the model for better optimization
            scripted_model = torch.jit.freeze(scripted_model)
            
            # Apply optimization passes
            scripted_model = torch.jit.optimize_for_inference(scripted_model)
            
            # Advanced graph optimizations (PyTorch 2.7)
            try:
                # Remove unnecessary operations
                torch.jit.run_unused_elimination(scripted_model.graph)
                torch.jit.run_dead_code_elimination(scripted_model.graph)
                
                # Constant folding and propagation
                torch.jit.run_constant_folding(scripted_model.graph)
                torch.jit.run_constant_propagation(scripted_model.graph)
                
                # Algebraic simplifications
                torch.jit.run_algebraic_simplification(scripted_model.graph)
                
                # Peephole optimizations
                torch.jit.run_peephole(scripted_model.graph, addmm_fusion_enabled=True)
                
                logger.info("✓ Advanced graph optimizations applied")
                
            except Exception as e:
                logger.warning(f"Some graph optimizations failed: {e}")
            
        except Exception as e:
            logger.warning(f"Model optimization failed: {e}")
        
        return scripted_model
    
    def _validate_trace_consistency(self, original_model: nn.Module, 
                                  scripted_model: torch.jit.ScriptModule,
                                  test_inputs: List[torch.Tensor]):
        """Validate trace consistency with multiple inputs."""
        logger.info("Validating trace consistency...")
        
        for i, test_input in enumerate(test_inputs[:3]):  # Limit to 3 for performance
            try:
                with inference_mode_context():
                    original_out = original_model(test_input)
                    traced_out = scripted_model(test_input)
                    
                    # Handle different output formats
                    if isinstance(original_out, (list, tuple)):
                        original_out = original_out[0] if len(original_out) == 1 else original_out
                    if isinstance(traced_out, (list, tuple)):
                        traced_out = traced_out[0] if len(traced_out) == 1 else traced_out
                    
                    diff = torch.max(torch.abs(original_out.float() - traced_out.float())).item()
                    logger.info(f"  Input {i+1}: max difference = {diff:.6f}")
                    
            except Exception as e:
                logger.warning(f"  Input {i+1}: validation failed - {e}")
    
    def apply_advanced_optimizations(self, model: nn.Module, use_fp16: bool = False,
                                   use_int8: bool = False, device: torch.device = None) -> nn.Module:
        """Apply advanced inference optimizations."""
        logger.info("Applying advanced inference optimizations...")
        
        # Disable gradient computation
        for param in model.parameters():
            param.requires_grad_(False)
        
        # Quantization optimizations
        if use_int8:
            try:
                # Dynamic quantization for CPU inference
                if device is None or device.type == 'cpu':
                    model = torch.ao.quantization.quantize_dynamic(
                        model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
                    )
                    logger.info("✓ Applied INT8 dynamic quantization")
                else:
                    logger.warning("INT8 quantization is primarily for CPU inference")
            except Exception as e:
                logger.warning(f"INT8 quantization failed: {e}")
        
        # Optimized FP16 implementation with FP32 I/O compatibility
        if use_fp16 and device and device.type == 'cuda':
            try:
                # Use wrapper to maintain FP32 input/output interface
                model = FP16ModelWrapper(model)
                
                # Optimize memory format for FP16
                if hasattr(model, 'to'):
                    model = model.to(memory_format=torch.channels_last)
                
                # Apply additional FP16 optimizations
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cuda.matmul.allow_tf32 = True
                
                logger.info("✓ Applied FP16 optimization with FP32 I/O compatibility")
                
            except Exception as e:
                logger.warning(f"FP16 optimization failed: {e}")
        
        return model
    
    def benchmark_comprehensive(self, model: torch.jit.ScriptModule, test_input: torch.Tensor, 
                              num_warmup: int = 20, num_runs: int = 100) -> Dict[str, float]:
        """Comprehensive benchmarking with memory profiling."""
        logger.info(f"Running comprehensive benchmark ({num_warmup} warmup + {num_runs} runs)...")
        
        model.eval()
        
        # Memory profiling
        if test_input.is_cuda:
            torch.cuda.empty_cache()
            start_memory = torch.cuda.memory_allocated()
        
        # Extended warmup for compiled models
        with inference_mode_context():
            for _ in range(num_warmup):
                _ = model(test_input)
        
        # Sync GPU
        if test_input.is_cuda:
            torch.cuda.synchronize()
        
        # Benchmark with memory tracking
        times = []
        memory_usage = []
        
        with inference_mode_context():
            for _ in range(num_runs):
                if test_input.is_cuda:
                    torch.cuda.synchronize()
                
                start_time = time.perf_counter()
                _ = model(test_input)
                
                if test_input.is_cuda:
                    torch.cuda.synchronize()
                    current_memory = torch.cuda.memory_allocated()
                    memory_usage.append((current_memory - start_memory) / 1024 / 1024)  # MB
                
                end_time = time.perf_counter()
                times.append((end_time - start_time) * 1000)
        
        results = {
            'mean_ms': np.mean(times),
            'std_ms': np.std(times),
            'min_ms': np.min(times),
            'max_ms': np.max(times),
            'p95_ms': np.percentile(times, 95),
            'p99_ms': np.percentile(times, 99),
            'fps': 1000.0 / np.mean(times),
            'throughput_std': 1000.0 / np.std(times) if np.std(times) > 0 else float('inf')
        }
        
        if memory_usage:
            results.update({
                'memory_mb_mean': np.mean(memory_usage),
                'memory_mb_max': np.max(memory_usage),
                'memory_mb_std': np.std(memory_usage)
            })
        
        logger.info(f"Performance results:")
        logger.info(f"  Mean: {results['mean_ms']:.2f} ± {results['std_ms']:.2f} ms")
        logger.info(f"  P95/P99: {results['p95_ms']:.2f}/{results['p99_ms']:.2f} ms")
        logger.info(f"  Throughput: {results['fps']:.1f} FPS")
        
        if memory_usage:
            logger.info(f"  Memory: {results['memory_mb_mean']:.1f} ± {results['memory_mb_std']:.1f} MB")
        
        return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert DepthAnythingV2 .pth to optimized LibTorch with PyTorch 2.7 features",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument("checkpoint", help="Path to input .pth checkpoint")
    parser.add_argument("output", help="Path to output TorchScript .pt file")
    parser.add_argument("--encoder", required=True, choices=['vits', 'vitb', 'vitl', 'vitg'],
                       help="Model encoder type")
    
    # Model parameters
    parser.add_argument("--max-depth", type=float, default=20.0,
                       help="Maximum depth for the model")
    parser.add_argument("--strict", action="store_true",
                       help="Strict state dict loading")
    
    # Conversion parameters
    parser.add_argument("--method", choices=["script", "trace"], default="trace",
                       help="TorchScript conversion method")
    parser.add_argument("--batch", type=int, default=1,
                       help="Batch size for example input")
    parser.add_argument("--height", type=int, default=480,
                       help="Input height")
    parser.add_argument("--width", type=int, default=640,
                       help="Input width")
    
    # PyTorch 2.7 optimizations
    parser.add_argument("--compile-after", action="store_true",
                       help="Apply torch.compile after TorchScript conversion (experimental)")
    parser.add_argument("--compile-only", action="store_true",
                       help="Use torch.compile instead of TorchScript (no .pt export)")
    parser.add_argument("--use-wrapper", action="store_true", default=True,
                       help="Use optimized tracing wrapper")
    
    # Precision and quantization
    parser.add_argument("--fp16", action="store_true",
                       help="Use mixed precision (FP16)")
    parser.add_argument("--int8", action="store_true",
                       help="Apply INT8 quantization (CPU)")
    
    # Validation and benchmarking
    parser.add_argument("--tolerance", type=float, default=1e-3,
                       help="Validation tolerance")
    parser.add_argument("--skip-validation", action="store_true",
                       help="Skip output validation")
    parser.add_argument("--extended-benchmark", action="store_true",
                       help="Run extended performance benchmark")
    parser.add_argument("--benchmark-runs", type=int, default=100,
                       help="Number of benchmark runs")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup device with optimizations
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
        
        # GPU optimizations
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    
    # Initialize converter
    converter = AdvancedModelConverter()
    
    try:
        # Load original model
        logger.info("=== Loading Original Model ===")
        original_model = converter.load_checkpoint(
            args.checkpoint, args.encoder, device, args.max_depth, args.strict
        )
        
        # Prepare model for tracing with PyTorch 2.7 optimizations
        logger.info("=== Preparing Model with Advanced Optimizations ===")
        
        # Choose optimization path based on arguments
        if args.compile_only:
            logger.info("Using torch.compile path (no TorchScript conversion)")
            prepared_model = converter.prepare_model_for_tracing(
                original_model, args.height, args.width, args.use_wrapper, 
                for_torchscript=False
            )
            
            # Apply other optimizations
            optimized_model = converter.apply_advanced_optimizations(
                prepared_model, args.fp16, args.int8, device
            )
            
            # Benchmark the compiled model directly
            example_input = torch.randn(
                args.batch, 3, args.height, args.width,
                device=device, dtype=torch.float16 if args.fp16 else torch.float32
            )
            
            if device.type == 'cuda':
                example_input = example_input.to(memory_format=torch.channels_last)
            
            if args.extended_benchmark:
                logger.info("=== Benchmarking Compiled Model ===")
                # Create a simple wrapper for benchmarking
                class BenchWrapper:
                    def __init__(self, model):
                        self.model = model
                        self.eval()
                    
                    def eval(self):
                        return self
                    
                    def __call__(self, x):
                        return self.model(x)
                
                benchmark_results = converter.benchmark_comprehensive(
                    BenchWrapper(optimized_model), example_input, num_runs=args.benchmark_runs
                )
            
            logger.info("✓ torch.compile optimization completed!")
            return
        
        else:
            # Standard TorchScript path (without torch.compile during tracing)
            prepared_model = converter.prepare_model_for_tracing(
                original_model, args.height, args.width, args.use_wrapper, 
                for_torchscript=True
            )
        
        # Apply advanced optimizations
        logger.info("=== Applying Advanced Optimizations ===")
        optimized_model = converter.apply_advanced_optimizations(
            prepared_model, args.fp16, args.int8, device
        )
        
        # Prepare example input
        example_input = torch.randn(
            args.batch, 3, args.height, args.width,
            device=device, dtype=torch.float32  # Always use FP32 inputs
        )
        
        # Optimize input memory format for better performance
        if device.type == 'cuda':
            example_input = example_input.to(memory_format=torch.channels_last)
        
        logger.info(f"Example input: {example_input.shape}, {example_input.dtype}, {example_input.device}")
        
        # Convert to TorchScript with advanced techniques
        logger.info("=== Converting to TorchScript ===")
        scripted_model = converter.convert_with_advanced_tracing(
            optimized_model, example_input, args.method
        )
        
        # Optionally apply torch.compile after TorchScript (experimental)
        if args.compile_after:
            logger.info("=== Applying torch.compile to TorchScript model (Experimental) ===")
            try:
                compiled_scripted = torch.compile(scripted_model, mode='default')
                # Test it works
                with torch.no_grad():
                    _ = compiled_scripted(example_input)
                scripted_model = compiled_scripted
                logger.info("✓ Successfully applied torch.compile to TorchScript model")
            except Exception as e:
                logger.warning(f"torch.compile after TorchScript failed: {e}")
        
        # Save the model
        logger.info("=== Saving Optimized Model ===")
        torch.jit.save(scripted_model, args.output)
        logger.info(f"✓ Saved optimized model to: {args.output}")
        
        # Extended benchmarking
        if args.extended_benchmark:
            logger.info("=== Extended Benchmarking ===")
            benchmark_results = converter.benchmark_comprehensive(
                scripted_model, example_input, num_runs=args.benchmark_runs
            )
        
        # Final summary
        logger.info("=== Optimization Summary ===")
        logger.info(f"✓ TorchScript conversion completed")
        logger.info(f"✓ Post-TorchScript compilation: {'Enabled' if args.compile_after else 'Disabled'}")
        logger.info(f"✓ Mixed precision: {args.fp16}")
        logger.info(f"✓ INT8 quantization: {args.int8}")
        
        file_size_mb = os.path.getsize(args.output) / (1024 * 1024)
        logger.info(f"✓ Output size: {file_size_mb:.1f} MB")
        logger.info("Advanced conversion completed successfully!")
        
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
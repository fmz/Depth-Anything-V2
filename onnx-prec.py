import onnx
from onnx import numpy_helper

model = onnx.load("onnx-exports/depth_anything_v2_FT_fp16_static.onnx")

# Function to get tensor data type
def get_tensor_dtype(tensor_name, model):
    for initializer in model.graph.initializer:
        if initializer.name == tensor_name:
            return onnx.TensorProto.DataType.Name(initializer.data_type)
    return None

# Iterate through nodes to check input and output data types
for node in model.graph.node:
    input_dtypes = [get_tensor_dtype(inp, model) for inp in node.input]
    output_dtypes = [get_tensor_dtype(out, model) for out in node.output]
    print(f'Node: {node.name}, OpType: {node.op_type}')
    print(f'  Input Data Types: {input_dtypes}')
    print(f'  Output Data Types: {output_dtypes}')

# Iterate through initializers to check their data types
for initializer in model.graph.initializer:
    dtype = onnx.TensorProto.DataType.Name(initializer.data_type)
    print(f'Initializer: {initializer.name}, Data Type: {dtype}')

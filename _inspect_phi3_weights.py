"""Analyze Phi-3-mini weight structure: how MatMulNBits stores INT4 weights."""
import onnx
from collections import defaultdict

MODEL_PATH = r"C:\Projects\win-acp-java\model\phi3-mini-directml-int4\directml\directml-int4-awq-block-128\model.onnx"

model = onnx.load(MODEL_PATH, load_external_data=False)
graph = model.graph

# Build initializer index
init_map = {}
for init in graph.initializer:
    init_map[init.name] = init

# Analyze first MatMulNBits node in detail
print("=== First MatMulNBits node (layer 0 k_proj) ===")
for node in graph.node:
    if node.op_type == "MatMulNBits":
        print(f"  Op: {node.op_type}")
        print(f"  Domain: {node.domain}")
        print(f"  Inputs ({len(node.input)}):")
        for i, inp_name in enumerate(node.input):
            init = init_map.get(inp_name)
            if init:
                dims = list(init.dims)
                dtype = init.data_type  # 1=float32, 2=uint8, 7=int64, 10=float16
                dtype_names = {1:'float32', 2:'uint8', 3:'int8', 6:'int32', 7:'int64', 10:'float16', 16:'bfloat16'}
                ext = "EXTERNAL" if init.data_location == 1 else "INLINE"
                print(f"    [{i}] {inp_name}: dtype={dtype_names.get(dtype, dtype)}, dims={dims}, loc={ext}")
            else:
                print(f"    [{i}] {inp_name}: (graph input/intermediate)")
        print(f"  Outputs: {list(node.output)}")
        print(f"  Attributes:")
        for attr in node.attribute:
            print(f"    {attr.name} = {attr.i if attr.type == 2 else attr.f if attr.type == 1 else attr.s}")
        break

# Analyze first GroupQueryAttention node
print(f"\n=== First GroupQueryAttention node ===")
for node in graph.node:
    if node.op_type == "GroupQueryAttention":
        print(f"  Inputs ({len(node.input)}):")
        for i, inp_name in enumerate(node.input):
            init = init_map.get(inp_name)
            if init:
                print(f"    [{i}] {inp_name}: dims={list(init.dims)}, dtype={init.data_type}")
            else:
                print(f"    [{i}] {inp_name}: (runtime)")
        print(f"  Outputs: {list(node.output)}")
        print(f"  Attributes:")
        for attr in node.attribute:
            print(f"    {attr.name} = {attr.i if attr.type == 2 else attr.f if attr.type == 1 else attr.s}")
        break

# Analyze RotaryEmbedding
print(f"\n=== First RotaryEmbedding node ===")
for node in graph.node:
    if node.op_type == "RotaryEmbedding":
        print(f"  Inputs ({len(node.input)}):")
        for i, inp_name in enumerate(node.input):
            init = init_map.get(inp_name)
            if init:
                print(f"    [{i}] {inp_name}: dims={list(init.dims)}, dtype={init.data_type}")
            else:
                print(f"    [{i}] {inp_name}: (runtime)")
        print(f"  Attributes:")
        for attr in node.attribute:
            print(f"    {attr.name} = {attr.i if attr.type == 2 else attr.f if attr.type == 1 else attr.s}")
        break

# Summarize all initializer shapes for one transformer layer
print(f"\n=== Layer 0 initializers ===")
layer0_inits = [(name, init) for name, init in init_map.items() if "layers.0" in name or "layer_0" in name or "8714" in name or "8715" in name or "8716" in name or "8753" in name or "8754" in name]
for name, init in sorted(layer0_inits, key=lambda x: x[0])[:20]:
    dtype_names = {1:'float32', 2:'uint8', 3:'int8', 6:'int32', 7:'int64', 10:'float16', 16:'bfloat16'}
    print(f"  {name}: dtype={dtype_names.get(init.data_type, init.data_type)}, dims={list(init.dims)}")

# cos/sin cache
print(f"\n=== cos_cache / sin_cache ===")
for name in ["cos_cache", "sin_cache"]:
    init = init_map.get(name)
    if init:
        dtype_names = {1:'float32', 2:'uint8', 10:'float16'}
        print(f"  {name}: dtype={dtype_names.get(init.data_type, init.data_type)}, dims={list(init.dims)}")

# Embedding
print(f"\n=== Embedding ===")
for name in ["model.embed_tokens.weight"]:
    init = init_map.get(name)
    if init:
        dtype_names = {1:'float32', 2:'uint8', 10:'float16'}
        print(f"  {name}: dtype={dtype_names.get(init.data_type, init.data_type)}, dims={list(init.dims)}")


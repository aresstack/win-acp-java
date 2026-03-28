"""Inspect Phi-3-mini ONNX graph structure: inputs, outputs, op types, node count."""
import onnx
import sys
from collections import Counter

MODEL_PATH = r"C:\Projects\win-acp-java\model\phi3-mini-directml-int4\directml\directml-int4-awq-block-128\model.onnx"

print(f"Loading ONNX model from {MODEL_PATH}...", flush=True)
model = onnx.load(MODEL_PATH, load_external_data=False)
graph = model.graph

print(f"\n=== Model Info ===")
print(f"IR version: {model.ir_version}")
print(f"Opset imports: {[(o.domain or 'ai.onnx', o.version) for o in model.opset_import]}")
print(f"Producer: {model.producer_name} {model.producer_version}")

print(f"\n=== Graph: {graph.name} ===")
print(f"Nodes: {len(graph.node)}")
print(f"Initializers: {len(graph.initializer)}")

print(f"\n=== Inputs ({len(graph.input)}) ===")
for inp in graph.input:
    shape = []
    if inp.type.tensor_type.HasField('shape'):
        for dim in inp.type.tensor_type.shape.dim:
            if dim.HasField('dim_param'):
                shape.append(dim.dim_param)
            elif dim.HasField('dim_value'):
                shape.append(str(dim.dim_value))
            else:
                shape.append('?')
    dtype = inp.type.tensor_type.elem_type
    dtype_names = {1:'float32', 2:'uint8', 3:'int8', 5:'int16', 6:'int32', 7:'int64', 9:'string', 10:'float16', 11:'double', 12:'uint32', 13:'uint64', 16:'bfloat16'}
    print(f"  {inp.name}: {dtype_names.get(dtype, dtype)} [{', '.join(shape)}]")

print(f"\n=== Outputs ({len(graph.output)}) ===")
for out in graph.output:
    shape = []
    if out.type.tensor_type.HasField('shape'):
        for dim in out.type.tensor_type.shape.dim:
            if dim.HasField('dim_param'):
                shape.append(dim.dim_param)
            elif dim.HasField('dim_value'):
                shape.append(str(dim.dim_value))
            else:
                shape.append('?')
    dtype = out.type.tensor_type.elem_type
    print(f"  {out.name}: {dtype_names.get(dtype, dtype)} [{', '.join(shape)}]")

print(f"\n=== Op type histogram ===")
op_counts = Counter(n.op_type for n in graph.node)
for op, count in op_counts.most_common():
    print(f"  {op}: {count}")

# Show first 20 nodes
print(f"\n=== First 20 nodes ===")
for i, node in enumerate(graph.node[:20]):
    ins = ', '.join(node.input[:3])
    if len(node.input) > 3:
        ins += f', ... (+{len(node.input)-3})'
    outs = ', '.join(node.output[:2])
    print(f"  [{i}] {node.op_type}: ({ins}) -> ({outs})")

# Show last 10 nodes (logits head)
print(f"\n=== Last 10 nodes ===")
for i, node in enumerate(graph.node[-10:]):
    idx = len(graph.node) - 10 + i
    ins = ', '.join(node.input[:3])
    if len(node.input) > 3:
        ins += f', ... (+{len(node.input)-3})'
    outs = ', '.join(node.output[:2])
    print(f"  [{idx}] {node.op_type}: ({ins}) -> ({outs})")


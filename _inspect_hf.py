from huggingface_hub import list_repo_tree

files = list_repo_tree(
    "microsoft/Phi-3-mini-4k-instruct-onnx",
    path_in_repo="directml/directml-int4-awq-block-128",
    repo_type="model",
)
for f in files:
    if hasattr(f, "size"):
        print(f"{f.rfilename}  ({f.size:,} bytes)")
    else:
        print(str(f))


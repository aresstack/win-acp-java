from huggingface_hub import hf_hub_download
import os

REPO = "microsoft/Phi-3-mini-4k-instruct-onnx"
SUBDIR = "directml/directml-int4-awq-block-128"
DEST = r"C:\Projects\win-acp-java\model\phi3-mini-directml-int4"

os.makedirs(DEST, exist_ok=True)

# Download small config/tokenizer files first (skip the big .onnx.data for now)
small_files = [
    "config.json",
    "genai_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "model.onnx",          # ~2MB graph definition (no weights)
]

for fname in small_files:
    path = f"{SUBDIR}/{fname}"
    print(f"Downloading {fname}...", flush=True)
    local = hf_hub_download(repo_id=REPO, filename=path, local_dir=DEST)
    print(f"  -> {local}", flush=True)

print("\nDone. Large weight file (model.onnx.data ~2.1GB) not yet downloaded.", flush=True)
print("Download it when ready with:", flush=True)
print(f'  hf_hub_download("{REPO}", "{SUBDIR}/model.onnx.data", local_dir="{DEST}")', flush=True)


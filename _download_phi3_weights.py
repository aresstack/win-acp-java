import time
from huggingface_hub import hf_hub_download

REPO = "microsoft/Phi-3-mini-4k-instruct-onnx"
SUBDIR = "directml/directml-int4-awq-block-128"
DEST = r"C:\Projects\win-acp-java\model\phi3-mini-directml-int4"

MAX_RETRIES = 5
BACKOFF_BASE = 10  # seconds


def download_with_retry():
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"Downloading model.onnx.data (~2.1 GB) [attempt {attempt}/{MAX_RETRIES}]...", flush=True)
            local = hf_hub_download(
                repo_id=REPO,
                filename=f"{SUBDIR}/model.onnx.data",
                local_dir=DEST,
                resume_download=True,
                force_download=False,
            )
            print(f"Done: {local}", flush=True)
            return local
        except Exception as e:
            print(f"Attempt {attempt} failed: {e}", flush=True)
            if attempt < MAX_RETRIES:
                wait = BACKOFF_BASE * (2 ** (attempt - 1))
                print(f"Retrying in {wait}s...", flush=True)
                time.sleep(wait)
            else:
                raise RuntimeError(f"Download failed after {MAX_RETRIES} attempts") from e


if __name__ == "__main__":
    download_with_retry()


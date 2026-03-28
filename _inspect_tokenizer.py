"""Inspect Phi-3 tokenizer.json structure."""
import json

TOKENIZER_PATH = r"C:\Projects\win-acp-java\model\phi3-mini-directml-int4\directml\directml-int4-awq-block-128\tokenizer.json"

with open(TOKENIZER_PATH, "r", encoding="utf-8") as f:
    tok = json.load(f)

print("=== Top-level keys ===")
for k in tok.keys():
    v = tok[k]
    if isinstance(v, dict):
        print(f"  {k}: dict with keys {list(v.keys())[:10]}")
    elif isinstance(v, list):
        print(f"  {k}: list[{len(v)}]")
    elif isinstance(v, str) and len(v) > 100:
        print(f"  {k}: str({len(v)} chars)")
    else:
        print(f"  {k}: {v}")

print(f"\n=== Model type ===")
model = tok.get("model", {})
print(f"  type: {model.get('type')}")
print(f"  vocab size: {len(model.get('vocab', {}))}")
if "merges" in model:
    print(f"  merges: {len(model['merges'])}")

print(f"\n=== Added tokens (first 10) ===")
added = tok.get("added_tokens", [])
print(f"  total: {len(added)}")
for t in added[:10]:
    print(f"  id={t.get('id')}, content={t.get('content')!r}, special={t.get('special')}")

print(f"\n=== Special tokens map ===")
sp_path = r"C:\Projects\win-acp-java\model\phi3-mini-directml-int4\directml\directml-int4-awq-block-128\special_tokens_map.json"
with open(sp_path, "r", encoding="utf-8") as f:
    sp = json.load(f)
for k, v in sp.items():
    print(f"  {k}: {v}")

print(f"\n=== Tokenizer config ===")
tc_path = r"C:\Projects\win-acp-java\model\phi3-mini-directml-int4\directml\directml-int4-awq-block-128\tokenizer_config.json"
with open(tc_path, "r", encoding="utf-8") as f:
    tc = json.load(f)
for k, v in tc.items():
    if k == "chat_template":
        print(f"  {k}: ({len(v)} chars) {v[:200]}...")
    else:
        print(f"  {k}: {v}")


"""
Test NVIDIA NIM API endpoints using Python standard library (urllib.request).
Zero external dependencies.
"""
import json
import urllib.request
import urllib.error

KEY_1 = "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"
KEY_2 = "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"

print("=" * 65)
print("  TESTING NVIDIA NIM API WITH NATIVE PYTHON HTTP CLIENT")
print("=" * 65)

def call_nvidia(api_key, model, prompt):
    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 128,
        "temperature": 0.2
    }
    req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            res_body = response.read().decode('utf-8')
            res_json = json.loads(res_body)
            return True, res_json['choices'][0]['message']['content']
    except urllib.error.HTTPError as e:
        err_msg = e.read().decode('utf-8') if e.fp else str(e)
        return False, f"HTTP {e.code}: {err_msg}"
    except Exception as e:
        return False, str(e)

# 1. Test Key 1 with various models
print("\n[1] Testing Key 1 models...")
models_to_test = [
    "deepseek-ai/deepseek-r1",
    "meta/llama-3.3-70b-instruct",
    "meta/llama-3.1-8b-instruct",
    "nvidia/llama-3.1-nemotron-70b-instruct",
    "mistralai/mistral-7b-instruct-v0.3"
]

working_models_key1 = []
for m in models_to_test:
    print(f"  Testing model: {m}...")
    ok, resp = call_nvidia(KEY_1, m, "Respond with 'ONLINE: OK'")
    if ok:
        print(f"  [SUCCESS] {m} -> {resp.strip()[:80]}")
        working_models_key1.append(m)
    else:
        print(f"  [FAIL] {m} -> {resp[:120]}")

# 2. Test Key 2 models
print("\n[2] Testing Key 2 models...")
working_models_key2 = []
for m in ["meta/llama-3.2-11b-vision-instruct", "meta/llama-3.1-8b-instruct", "nvidia/ising-calibration-1.5-31b"]:
    print(f"  Testing model: {m}...")
    ok, resp = call_nvidia(KEY_2, m, "Respond with 'VISION_ONLINE: OK'")
    if ok:
        print(f"  [SUCCESS] {m} -> {resp.strip()[:80]}")
        working_models_key2.append(m)
    else:
        print(f"  [FAIL] {m} -> {resp[:120]}")

print("\n" + "=" * 65)
print(f"Working Models Key 1: {working_models_key1}")
print(f"Working Models Key 2: {working_models_key2}")
print("=" * 65)

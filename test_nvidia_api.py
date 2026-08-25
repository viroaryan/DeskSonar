"""
Test NVIDIA NIM API connections and model availability with provided keys.
"""
import os
import requests
from openai import OpenAI

KEY_1 = "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"
KEY_2 = "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"

print("=" * 60)
print("  TESTING NVIDIA NIM API KEYS & MODELS")
print("=" * 60)

# Test Key 1 with OpenAI client
print("\n[1] Testing Key 1 with OpenAI client at https://integrate.api.nvidia.com/v1...")
try:
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=KEY_1
    )
    
    # Test models listing or simple chat completion
    # Try deepseek or llama
    models_to_try = [
        "deepseek-ai/deepseek-r1",
        "deepseek-ai/deepseek-v3",
        "meta/llama-3.3-70b-instruct",
        "meta/llama-3.1-8b-instruct",
        "nvidia/llama-3.1-nemotron-70b-instruct"
    ]
    
    success_model = None
    for model_name in models_to_try:
        try:
            print(f"  Attempting model: {model_name}...")
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": "Respond with 'ONLINE: ' followed by your model name and 'READY'"}],
                max_tokens=64,
                temperature=0.1
            )
            print(f"  [SUCCESS] Response from {model_name}: {resp.choices[0].message.content.strip()}")
            success_model = model_name
            break
        except Exception as e:
            print(f"  [INFO] {model_name} failed: {e}")
            
except Exception as e:
    print(f"  [FAIL] Key 1 test error: {e}")

# Test Key 2 with Vision / Multimodal or Chat
print("\n[2] Testing Key 2 with NVIDIA API...")
try:
    client2 = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=KEY_2
    )
    vision_models = [
        "meta/llama-3.2-11b-vision-instruct",
        "nvidia/neva-22b",
        "microsoft/phi-3-vision-128k-instruct",
        "meta/llama-3.1-8b-instruct"
    ]
    for model_name in vision_models:
        try:
            print(f"  Attempting model on Key 2: {model_name}...")
            resp = client2.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": "Respond with 'VISION_ONLINE: READY'"}],
                max_tokens=64,
                temperature=0.1
            )
            print(f"  [SUCCESS] Response from {model_name}: {resp.choices[0].message.content.strip()}")
            break
        except Exception as e:
            print(f"  [INFO] {model_name} failed: {e}")
except Exception as e:
    print(f"  [FAIL] Key 2 test error: {e}")

print("=" * 60)

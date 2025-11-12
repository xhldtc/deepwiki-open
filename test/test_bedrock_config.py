#!/usr/bin/env python3
"""
Quick test to verify Bedrock embedder configuration is loaded correctly.
"""

import os
import sys

# Set the embedder type to bedrock before importing config
os.environ['DEEPWIKI_EMBEDDER_TYPE'] = 'bedrock'

# Import after setting environment variable
from api.config import configs, get_embedder_config, get_embedder_type, is_bedrock_embedder, CLIENT_CLASSES

print("=" * 60)
print("Bedrock Embedder Configuration Test")
print("=" * 60)

print(f"\n1. Environment variable DEEPWIKI_EMBEDDER_TYPE: {os.environ.get('DEEPWIKI_EMBEDDER_TYPE')}")

print(f"\n2. Detected embedder type: {get_embedder_type()}")
print(f"   is_bedrock_embedder(): {is_bedrock_embedder()}")

print(f"\n3. Available embedder configs in configs dict:")
for key in configs.keys():
    if 'embedder' in key:
        print(f"   - {key}")

print(f"\n4. embedder_bedrock in configs: {('embedder_bedrock' in configs)}")
if 'embedder_bedrock' in configs:
    bedrock_config = configs['embedder_bedrock']
    print(f"   - client_class: {bedrock_config.get('client_class')}")
    print(f"   - model_client: {bedrock_config.get('model_client')}")
    print(f"   - model: {bedrock_config.get('model_kwargs', {}).get('model')}")

print(f"\n5. BedrockClient in CLIENT_CLASSES: {('BedrockClient' in CLIENT_CLASSES)}")

print(f"\n6. get_embedder_config() result:")
current_config = get_embedder_config()
if current_config:
    print(f"   - client_class: {current_config.get('client_class')}")
    print(f"   - model_client: {current_config.get('model_client')}")
    print(f"   - model: {current_config.get('model_kwargs', {}).get('model')}")
else:
    print("   ❌ No config returned!")

print("\n" + "=" * 60)

# Test embedder creation
print("\n7. Testing embedder creation...")
try:
    from api.tools.embedder import get_embedder
    embedder = get_embedder(embedder_type='bedrock')
    print(f"   ✅ Bedrock embedder created successfully!")
    print(f"   - Embedder type: {type(embedder)}")
    print(f"   - Model client type: {type(embedder.model_client)}")
except Exception as e:
    print(f"   ❌ Failed to create embedder: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Test completed!")
print("=" * 60)

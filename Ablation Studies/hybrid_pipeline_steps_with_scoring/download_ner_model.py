#!/usr/bin/env python3
"""
Pre-download the biomedical NER model to ensure it's cached locally.
This helps avoid network issues during job execution.
"""

import os
import sys

def download_ner_model():
    """Download and cache the biomedical NER model."""
    print("="*80)
    print("Pre-downloading Biomedical NER Model")
    print("="*80)
    
    try:
        from transformers import pipeline
        import time
        
        model_name = 'd4data/biomedical-ner-all'
        
        print(f"\nDownloading model: {model_name}")
        print("This may take a few minutes on first run...")
        print("Subsequent runs will use the cached version.\n")
        
        # Try to load the model (will download if not cached)
        max_retries = 5
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                print(f"Attempt {attempt + 1}/{max_retries}...")
                ner_pipeline = pipeline(
                    "ner",
                    model=model_name,
                    aggregation_strategy="simple",
                    device=-1  # CPU for download
                )
                print(f"\n✅ Model downloaded and cached successfully!")
                print(f"   Model location: ~/.cache/huggingface/hub/models--d4data--biomedical-ner-all")
                
                # Test it works
                test_text = "Aspirin is used to treat pain."
                results = ner_pipeline(test_text)
                print(f"\n✅ Test successful! Detected {len(results)} entities in test text.")
                
                return True
                
            except Exception as e:
                error_str = str(e).lower()
                if 'network' in error_str or 'resolve' in error_str or 'connection' in error_str:
                    if attempt < max_retries - 1:
                        print(f"⚠️  Network error: {e}")
                        print(f"   Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    else:
                        print(f"\n❌ Failed to download after {max_retries} attempts")
                        print(f"   Error: {e}")
                        print("\n   The model may already be cached. Checking cache...")
                        
                        # Check if model exists in cache
                        cache_path = os.path.expanduser("~/.cache/huggingface/hub/models--d4data--biomedical-ner-all")
                        if os.path.exists(cache_path):
                            print(f"   ✅ Found cached model at: {cache_path}")
                            print("   The script should be able to use the cached version.")
                            return True
                        else:
                            print("   ❌ No cached model found.")
                            print("   You may need to run this script when network is available.")
                            return False
                else:
                    print(f"\n❌ Error: {e}")
                    return False
        
        return False
        
    except ImportError:
        print("❌ Error: transformers not installed.")
        print("   Install with: pip install transformers")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = download_ner_model()
    sys.exit(0 if success else 1)

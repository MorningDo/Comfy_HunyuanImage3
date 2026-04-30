"""
Chain upload for remaining v2 Instruct models.
Runs each upload sequentially with automatic confirmation.
Skips models that are already fully uploaded.
"""
import subprocess
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_SCRIPT = os.path.join(SCRIPT_DIR, "upload_v2_to_hf.py")

# Models to upload (instruct-nf4 is already uploading separately)
MODELS = ["instruct-int8", "instruct-distil-nf4", "instruct-distil-int8"]

def main():
    total = len(MODELS)
    for i, model in enumerate(MODELS, 1):
        print(f"\n{'='*60}")
        print(f"  [{i}/{total}] Starting upload: {model}")
        print(f"{'='*60}\n")
        
        result = subprocess.run(
            [sys.executable, UPLOAD_SCRIPT, "--model", model],
            input="y\n",
            text=True,
        )
        
        if result.returncode != 0:
            print(f"\n*** FAILED: {model} (exit code {result.returncode}) ***")
            print(f"*** Continuing with next model... ***\n")
        else:
            print(f"\n*** DONE: {model} ***\n")
    
    print(f"\n{'='*60}")
    print(f"  All {total} uploads complete!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

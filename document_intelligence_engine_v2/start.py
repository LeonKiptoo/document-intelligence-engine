import sys
import traceback

print("Starting DocIntel...", flush=True)

try:
    print("Testing imports...", flush=True)
    from scripts import api
    print("Imports OK", flush=True)
except Exception as e:
    print(f"IMPORT FAILED: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)

import uvicorn
print("Starting uvicorn on port 10000...", flush=True)
uvicorn.run("scripts.api:app", host="0.0.0.0", port=10000)

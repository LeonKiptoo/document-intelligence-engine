import uvicorn
uvicorn.run("scripts.api:app", host="0.0.0.0", port=10000, log_level="info")
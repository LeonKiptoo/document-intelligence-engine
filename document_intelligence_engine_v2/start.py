import uvicorn
uvicorn.run("scripts.api:app", host="0.0.0.0", port=7860, log_level="info")

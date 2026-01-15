from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
import httpx
import os
import json

app = FastAPI()
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
INFERENCE_MODEL = os.environ.get("INFERENCE_MODEL", "tinyllama")
STREAMING = os.environ.get("OLLAMA_STREAMING", "0") != "0"

@app.get("/")
async def serve_index():
    return FileResponse("index.html")

@app.post("/api/chat")
async def chat(request: Request):
    data = await request.json()
    prompt = data.get("message", "")
    payload = {"model": INFERENCE_MODEL, "prompt": prompt, "stream": STREAMING}
    async with httpx.AsyncClient() as client:
        r = await client.post(OLLAMA_URL, json=payload, timeout=None)
        r.raise_for_status()
        if STREAMING:
            # Stream JSON lines to the frontend as a single event stream
            async def event_stream():
                async for line in r.aiter_lines():
                    line = line.strip()
                    print(line)
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        # Only send the 'response' field
                        yield f"data: {json.dumps({'response': obj.get('response', '')})}\n\n"
                    except Exception:
                        continue
            return StreamingResponse(event_stream(), media_type="text/event-stream")
        else:
            result = r.json()
            return JSONResponse({"response": result.get("response", "")})

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
import httpx
import os
import json

app = FastAPI()
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434") + "/api/generate"
INFERENCE_MODEL = os.environ.get("INFERENCE_MODEL", "tinyllama")
STREAMING = os.environ.get("OLLAMA_STREAMING", "1") != "0"

@app.get("/")
async def serve_index():
    print("serving index.html")
    return FileResponse("index.html")

@app.post("/api/chat")
async def chat(request: Request):
    data = await request.json()
    prompt = data.get("message", "")
    payload = {"model": INFERENCE_MODEL, "prompt": prompt, "stream": STREAMING}

    if STREAMING:
        # Stream JSON lines to the frontend as SSE
        async def event_stream():
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("POST", OLLAMA_URL, json=payload) as r:
                    r.raise_for_status()
                    async for line in r.aiter_lines():
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            # Only send the 'response' field
                            response_text = obj.get('response', '')
                            if response_text:
                                yield f"data: {json.dumps({'response': response_text})}\n\n"
                        except Exception as e:
                            print(f"Error parsing line: {e}")
                            continue
        return StreamingResponse(event_stream(), media_type="text/event-stream")
    else:
        # Non-streaming mode
        async with httpx.AsyncClient(timeout=None) as client:
            r = await client.post(OLLAMA_URL, json=payload)
            r.raise_for_status()
            result = r.json()
            return JSONResponse({"response": result.get("response", "")})

import base64
import httpx
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Enable CORS so your local network devices can talk to the API smoothly
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OLLAMA_URL = "http://localhost:11434/api/chat"

# --- BACKEND API ROUTE ---
@app.post("/identify")
async def identify_plant(file: UploadFile = File(...)):
    contents = await file.read()
    base64_image = base64.b64encode(contents).decode('utf-8')
    
    payload = {
        "model": "qwen2.5vl:3b-lowmem",
        "messages": [
            {
                "role": "user",
                "content": "Identify this plant. Return its common name, botanical name, a fun fact, soil/water/sunlight conditions, and its pollinators.",
                "images": [base64_image]
            }
        ],
        "stream": False
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(OLLAMA_URL, json=payload, timeout=60.0)
        result = response.json()
        return {"response": result['message']['content']}

# --- FRONTEND PWA MOUNT ---
# This mounts the public folder to the root URL (/). 
# Setting html=True automatically redirects requests to index.html
app.mount("/", StaticFiles(directory="public", html=True), name="public")

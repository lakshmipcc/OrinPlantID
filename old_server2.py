import base64
import httpx
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Enable CORS so local network devices can talk to the API smoothly
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
    
    # CHAIN-OF-THOUGHT ENHANCEMENT:
    # We force the model to analyze visual traits FIRST before identifying.
    cot_prompt = (
        "Analyze this plant image step-by-step:\n"
        "1. Visual Observations: Describe the leaf shape, margins, leaf arrangement, stem structure, or flowers/fruit.\n"
        "2. Reasoning: Note distinguishing botanical characteristics.\n"
        "3. Identification & Details: Based on your analysis, state the common name, botanical name, a fun fact, soil/water/sunlight conditions, and its pollinators."
    )

    payload = {
        "model": "qwen2.5vl:3b-lowmem",
        "messages": [
            {
                "role": "user",
                "content": cot_prompt,
                "images": [base64_image]
            }
        ],
        "stream": False
    }
    
    async with httpx.AsyncClient() as client:
        # Timeout set to 90s to give Ollama room for the extra reasoning tokens
        response = await client.post(OLLAMA_URL, json=payload, timeout=90.0)
        result = response.json()
        return {"response": result['message']['content']}

# --- FRONTEND PWA MOUNT ---
app.mount("/", StaticFiles(directory="public", html=True), name="public")

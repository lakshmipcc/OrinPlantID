import asyncio
import base64
import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OLLAMA_URL = "http://localhost:11434/api/chat"

@app.post("/identify")
async def identify_plant(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        base64_image = base64.b64encode(contents).decode('utf-8')

        # PROMPT WITH DELIMITER
        cot_prompt = (
            "You are an expert botanist analyzing a plant sample in the Pacific Northwest (PNW) region.\n"
            "Analyze the image step-by-step:\n"
            "1. Growth Habit & Form: Identify overall plant type.\n"
            "2. Leaf & Stem Features: Describe leaf arrangement, margins, thickness, and surface texture.\n"
            "3. Reproductive/Special Features: Note flowers, berries, or distinct markings.\n"
            "4. Regional & Cultivation Context: Consider PNW native flora and common regional indoor houseplants.\n\n"
            "After your analysis, output the line '---FINAL IDENTIFICATION---' and then provide ONLY:\n"
            "- Common Name (SPECIFICITY RULE: Do NOT use generic terms like 'Succulent'. Give the specific common or genus name)\n"
            "- Botanical (Scientific) Name\n"
            "- Confidence Score: (Estimated probability percentage, e.g., '85% (High Certainty)' or '45% (Low Certainty - Ambiguous traits)')\n"
            "- A fun fact\n"
            "- Soil / Water / Sunlight conditions\n"
            "- Pollinators"
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
            response = await client.post(OLLAMA_URL, json=payload, timeout=90.0)
            result = response.json()
            raw_text = result['message']['content']

            # PYTHON FILTER: Strip out the reasoning trace before sending to the phone app
            if "---FINAL IDENTIFICATION---" in raw_text:
                clean_output = raw_text.split("---FINAL IDENTIFICATION---")[-1].strip()
            else:
                clean_output = raw_text.strip()

            return {"response": clean_output}

    # Catch mobile disconnects / broken pipes gracefully without dumping error logs
    except (asyncio.CancelledError, BrokenPipeError, ConnectionResetError):
        print("Notice: Client disconnected before inference completed.")
        return {"response": "Request cancelled by user."}

    # Catch Ollama timeouts
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Inference engine timed out.")

    # Catch general unexpected errors
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error.")

app.mount("/", StaticFiles(directory="public", html=True), name="public")

import io
from contextlib import asynccontextmanager
from typing import Optional

import torch
import soundfile as sf
import uvicorn
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel


from qwen_tts import Qwen3TTSModel

# Global variable to hold the model instance
model_instance = None


# This ensures the heavy model loads only once when the server starts.
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_instance
    print("Initializing TTS Model...")
    try:

        model_instance = Qwen3TTSModel.from_pretrained(
            "/workspace/models/checkpoint-epoch-2",
            device_map="cuda:0",
            dtype=torch.bfloat16,
            # attn_implementation="flash_attention_2", 
        )
        print("TTS Model loaded successfully.")
    except Exception as e:
        print(f"CRITICAL: Failed to load model: {e}")
    yield
    # Cleanup logic (if any) would go here
    print("Shutting down...")

app = FastAPI(lifespan=lifespan)


class TTSRequest(BaseModel):
    text: str
    language: str = "English"  # Default from example
    speaker: str = "john_e"    # Default from example
    instruct: Optional[str] = None

@app.post("/tts")
async def generate_audio_endpoint(request: TTSRequest):
    global model_instance
    if not model_instance:
        raise HTTPException(status_code=503, detail="Model is not fully initialized")

    try:

        inference_kwargs = {
            "text": request.text,
            "language": request.language,
            "speaker": request.speaker,
        }
        
        if request.instruct:
            inference_kwargs["instruct"] = request.instruct


        # The model returns a tuple (wavs, sr). wavs is a list of arrays.
        wavs, sr = model_instance.generate_custom_voice(**inference_kwargs)


        # Instead of sf.write("file.wav", ...), we write to a BytesIO buffer
        # to return the audio directly over the network.
        buffer = io.BytesIO()
        sf.write(buffer, wavs[0], sr, format='WAV')
        buffer.seek(0)

        return Response(content=buffer.read(), media_type="audio/wav")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=8123)
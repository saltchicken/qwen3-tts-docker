import io
import torch
import soundfile as sf
import uvicorn
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from typing import Optional
from contextlib import asynccontextmanager
from starlette.concurrency import run_in_threadpool

from qwen_tts import Qwen3TTSModel


model_instance = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_instance
    print("Initializing TTS Model...")
    try:

        # Note: If flash_attn is installed in Docker, enabling it here is significantly more efficient
        model_instance = Qwen3TTSModel.from_pretrained(
            "/workspace/models/checkpoint-epoch-2",
            device_map="cuda:0",
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )

        # model_instance.generate_custom_voice(text="init", language="English", speaker="john_e")
        print("TTS Model loaded successfully with Flash Attention 2.")
    except Exception as e:
        print(f"CRITICAL: Failed to load model: {e}")
        # Fallback if Flash Attention fails
        model_instance = Qwen3TTSModel.from_pretrained(
            "/workspace/models/checkpoint-epoch-2",
            device_map="cuda:0",
            dtype=torch.bfloat16,
        )
        print("TTS Model loaded with default attention.")
    yield
    print("Shutting down...")

app = FastAPI(lifespan=lifespan)

class TTSRequest(BaseModel):
    text: str
    language: str = "English"
    speaker: str = "john_e"
    instruct: Optional[str] = None

def _generate_audio(inference_kwargs):
    """‼️ Helper function to isolate the blocking model call"""
    global model_instance
    # The actual inference call
    wavs, sr = model_instance.generate_custom_voice(**inference_kwargs)
    

    buffer = io.BytesIO()
    # Ensure tensor is on CPU before soundfile tries to read it
    audio_data = wavs[0].cpu().numpy() if torch.is_tensor(wavs[0]) else wavs[0]
    sf.write(buffer, audio_data, sr, format='WAV')
    buffer.seek(0)
    return buffer.read()

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


        # Offload the blocking GPU/CPU bound task to a thread pool.
        # This allows FastAPI to remain responsive to other requests while this one computes.
        audio_bytes = await run_in_threadpool(_generate_audio, inference_kwargs)

        return Response(content=audio_bytes, media_type="audio/wav")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":

    # but handle concurrency via threading as shown above.
    uvicorn.run(app, host="0.0.0.0", port=8123, workers=1)

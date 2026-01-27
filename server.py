import io
import re  # ‼️ Added for sentence splitting
import torch
import soundfile as sf
import uvicorn
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse  # ‼️ Added for streaming
from pydantic import BaseModel
from typing import Optional, Generator
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

# ‼️ New Helper: Split text into sentences
def split_sentences(text: str):
    # Splits by punctuation (. ! ?) followed by whitespace, keeping the punctuation
    # This ensures "Hello! How are you?" splits into "Hello!" and "How are you?"
    return re.split(r'(?<=[.!?])\s+', text)

# ‼️ New Generator: Yields audio chunks sentence by sentence
def tts_stream_generator(text: str, language: str, speaker: str, instruct: Optional[str]):
    global model_instance
    if not model_instance:
        # In a generator, we can't easily raise HTTP 503, but we can fail hard or yield error audio
        # Ideally, the global check happens before this, but double checking here is safe
        print("Error: Model not initialized")
        return

    sentences = split_sentences(text)
    print(f"Processing {len(sentences)} sentences for stream...")

    # We use a specific subtype to ensure consistency between the WAV header and subsequent RAW chunks
    # PCM_16 is standard and widely supported
    target_subtype = 'PCM_16'

    for i, sentence in enumerate(sentences):
        if not sentence.strip():
            continue

        try:
            inference_kwargs = {
                "text": sentence,
                "language": language,
                "speaker": speaker,
            }
            if instruct:
                inference_kwargs["instruct"] = instruct

            # Generate audio for this sentence
            # ‼️ This blocks the thread, which is why we use a sync generator with StreamingResponse
            # FastAPI will run this loop in a threadpool automatically.
            wavs, sr = model_instance.generate_custom_voice(**inference_kwargs)

            # Ensure tensor is on CPU
            audio_data = wavs[0].cpu().numpy() if torch.is_tensor(wavs[0]) else wavs[0]

            buffer = io.BytesIO()
            
            if i == 0:
                # ‼️ First chunk: Write a full WAV header + data
                # This establishes the file format (WAV) and Sample Rate for the client
                sf.write(buffer, audio_data, sr, format='WAV', subtype=target_subtype)
            else:
                # ‼️ Subsequent chunks: Write RAW PCM data (no header)
                # We append raw bytes to the stream. Since we stick to PCM_16, 
                # the client plays it seamlessly as part of the initial WAV file.
                sf.write(buffer, audio_data, sr, format='RAW', subtype=target_subtype)

            buffer.seek(0)
            yield buffer.read()
            
        except Exception as e:
            print(f"Error generating sentence '{sentence[:20]}...': {e}")
            # Continue to next sentence even if one fails
            continue

@app.post("/tts")
def generate_audio_endpoint(request: TTSRequest):
    global model_instance
    if not model_instance:
        raise HTTPException(status_code=503, detail="Model is not fully initialized")

    # ‼️ Switched to StreamingResponse
    # We pass the sync generator directly. FastAPI/Starlette executes sync iterators 
    # in a threadpool to prevent blocking the asyncio event loop.
    return StreamingResponse(
        tts_stream_generator(request.text, request.language, request.speaker, request.instruct),
        media_type="audio/wav"
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8123, workers=1)

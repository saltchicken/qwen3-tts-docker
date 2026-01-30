import io
import re
import torch
import soundfile as sf
import uvicorn
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, Generator
from contextlib import asynccontextmanager
from starlette.concurrency import run_in_threadpool

from qwen_tts import Qwen3TTSModel


model_instance = None
voice_clone_prompt_items = None


common_gen_kwargs = dict(
    max_new_tokens=2048,
    do_sample=True,
    top_k=50,
    top_p=1.0,
    temperature=0.9,
    repetition_penalty=1.05,
    subtalker_dosample=True,
    subtalker_top_k=50,
    subtalker_top_p=1.0,
    subtalker_temperature=0.9,
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_instance, voice_clone_prompt_items
    print("Initializing TTS Model...")
    try:
        model_instance = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            device_map="cuda:0",
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        print("TTS Model loaded successfully with Flash Attention 2.")

        print("Initializing Voice Clone Prompts...")
        voice_clone_prompt_items = model_instance.create_voice_clone_prompt( 
            ref_audio = "./refs/glow_ref.wav",
            ref_text  = "The ground was black, still the foliage that grew from it and around it displayed the richest jeweled homes.",
            x_vector_only_mode=False,
        )
        print("Voice Clone Prompts initialized.")

    except Exception as e:
        # TODO: Handle this failure in a better way
        print(f"CRITICAL: Failed to load model: {e}")

    yield
    print("Shutting down...")

app = FastAPI(lifespan=lifespan)

class TTSRequest(BaseModel):
    text: str
    language: str = "English"


def split_sentences(text: str):
    # Splits by punctuation (. ! ?) followed by whitespace, keeping the punctuation
    # This ensures "Hello! How are you?" splits into "Hello!" and "How are you?"
    return re.split(r'(?<=[.!?])\s+', text)


def tts_stream_generator(text: str):
    global model_instance, voice_clone_prompt_items
    
    if not model_instance:
        print("Error: Model not initialized")
        return


    if not voice_clone_prompt_items:
        print("Error: Voice clone prompts not initialized")
        return

    sentences = split_sentences(text)
    print(f"Processing {len(sentences)} sentences for stream (Voice Clone Mode)...")

    # We use a specific subtype to ensure consistency between the WAV header and subsequent RAW chunks
    # PCM_16 is standard and widely supported
    target_subtype = 'PCM_16'

    for i, sentence in enumerate(sentences):
        if not sentence.strip():
            continue

        try:

            # We ignore the 'speaker' argument and use the global voice_clone_prompt_items
            # We also merge the global common_gen_kwargs
            wavs, sr = model_instance.generate_voice_clone(
                text=sentence,
                language="English",
                voice_clone_prompt=voice_clone_prompt_items,
                **common_gen_kwargs
            )

            # Ensure tensor is on CPU
            audio_data = wavs[0].cpu().numpy() if torch.is_tensor(wavs[0]) else wavs[0]

            buffer = io.BytesIO()
            
            if i == 0:

                # This establishes the file format (WAV) and Sample Rate for the client
                sf.write(buffer, audio_data, sr, format='WAV', subtype=target_subtype)
            else:

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


    # We pass the sync generator directly. FastAPI/Starlette executes sync iterators 
    # in a threadpool to prevent blocking the asyncio event loop.
    return StreamingResponse(
        tts_stream_generator(request.text),
        media_type="audio/wav"
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8123, workers=1)

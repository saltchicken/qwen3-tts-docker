import io
import re
import torch
import soundfile as sf
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, Dict
from contextlib import asynccontextmanager
from qwen_tts import Qwen3TTSModel


class TextProcessor:
    @staticmethod
    def split_sentences(text: str):
        """
        Splits text into sentences while preserving punctuation.
        Also ensures the text ends with punctuation to prevent infinite generation loops.
        """
        text = text.strip()
        if not text:
            return []
            

        # Low temp + no punctuation often causes the model to generate silence forever.
        if text[-1] not in ".!?":
            text += "."
            
        # Split by punctuation followed by whitespace
        return re.split(r'(?<=[.!?])\s+', text)

class TTSEngine:
    def __init__(self):
        self.model = None
        self.voice_clone_prompt_items = None
        self.default_gen_kwargs = dict(
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

    def load_model(self):
        print("Initializing TTS Model...")
        try:
            self.model = Qwen3TTSModel.from_pretrained(
                "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                device_map="cuda:0",
                dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            )
            print("TTS Model loaded successfully with Flash Attention 2.")
            
            print("Initializing Voice Clone Prompts...")
            self.voice_clone_prompt_items = self.model.create_voice_clone_prompt( 
                ref_audio = "./refs/glow_ref.wav",
                ref_text  = "The ground was black, still the foliage that grew from it and around it displayed the richest jeweled homes.",
                x_vector_only_mode=False,
            )
            print("Voice Clone Prompts initialized.")
        except Exception as e:
            print(f"CRITICAL: Failed to load model: {e}")
            raise e

    def stream_audio(self, text: str, overrides: Dict = None):
        if not self.model or not self.voice_clone_prompt_items:
            raise RuntimeError("Model not initialized")

        sentences = TextProcessor.split_sentences(text)
        print(f"Processing {len(sentences)} sentences for stream...")

        # Merge defaults with any request-specific overrides
        gen_kwargs = self.default_gen_kwargs.copy()
        if overrides:
            gen_kwargs.update(overrides)
            

        print(f" >> Gen Params: Temp={gen_kwargs.get('temperature')}, TopP={gen_kwargs.get('top_p')}")

        target_subtype = 'PCM_16'

        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue

            try:
                print(f" >> Generating sentence {i+1}/{len(sentences)}: '{sentence[:30]}...'")
                wavs, sr = self.model.generate_voice_clone(
                    text=sentence,
                    language="English",
                    voice_clone_prompt=self.voice_clone_prompt_items,
                    **gen_kwargs
                )

                audio_data = wavs[0].cpu().numpy() if torch.is_tensor(wavs[0]) else wavs[0]

                buffer = io.BytesIO()
                
                if i == 0:
                    sf.write(buffer, audio_data, sr, format='WAV', subtype=target_subtype)
                else:
                    sf.write(buffer, audio_data, sr, format='RAW', subtype=target_subtype)

                buffer.seek(0)
                yield buffer.read()
                
            except Exception as e:
                print(f"Error generating sentence '{sentence[:20]}...': {e}")
                continue

tts_engine = TTSEngine()

@asynccontextmanager
async def lifespan(app: FastAPI):
    tts_engine.load_model()
    yield
    print("Shutting down...")

app = FastAPI(lifespan=lifespan)

class TTSRequest(BaseModel):
    text: str
    language: str = "English"
    temperature: Optional[float] = 0.9
    top_p: Optional[float] = 1.0

@app.post("/tts")
def generate_audio_endpoint(request: TTSRequest):
    if not tts_engine.model:
        raise HTTPException(status_code=503, detail="Model is not fully initialized")

    overrides = {
        "temperature": request.temperature,
        "subtalker_temperature": request.temperature,
        "top_p": request.top_p,
        "subtalker_top_p": request.top_p,
    }

    return StreamingResponse(
        tts_engine.stream_audio(request.text, overrides),
        media_type="audio/wav"
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8123, workers=1)
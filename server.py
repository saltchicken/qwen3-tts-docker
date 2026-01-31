import io
import re
import torch
import soundfile as sf
import uvicorn
import threading # ‼️ Added for thread safety
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, Dict
from contextlib import asynccontextmanager
from qwen_tts import Qwen3TTSModel


class TextProcessor:
    @staticmethod
    def split_sentences(text: str):
        text = text.strip()
        if not text:
            return []
        if text[-1] not in ".!?":
            text += "."
        return re.split(r'(?<=[.!?])\s+', text)

class TTSEngine:
    def __init__(self):
        self.model = None
        self.voice_prompts = {}
        self.default_voice_id = "glow_ref" 
        self.lock = threading.Lock() # ‼️ Added a lock to prevent concurrent GPU access
        
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
            print("TTS Model loaded successfully.")
            

            self._load_voices_from_refs()
            
        except Exception as e:
            print(f"CRITICAL: Failed to load model: {e}")
            raise e

    def _load_voices_from_refs(self):
        """
        Scans ./refs for voices. 
        Prioritizes binary .pt files (fast load).
        Compiles .wav+.txt pairs into .pt files if binary is missing.
        """
        refs_dir = Path("./refs")
        if not refs_dir.exists():
            print(f"Warning: {refs_dir} does not exist.")
            return

        print(f"Scanning {refs_dir} for voices...")
        

        # This is the fastest method and allows single-file distribution
        pt_files = list(refs_dir.glob("*.pt"))
        for pt_path in pt_files:
            voice_id = pt_path.stem
            try:
                print(f" >> Loading cached voice '{voice_id}' from .pt file...")
                # Load directly to the correct device (likely cuda:0 based on model)
                self.voice_prompts[voice_id] = torch.load(pt_path, weights_only=False)
            except Exception as e:
                print(f"Error loading cached voice '{voice_id}': {e}")


        wav_files = list(refs_dir.glob("*.wav"))
        for wav_path in wav_files:
            voice_id = wav_path.stem
            
            # If we already loaded this voice from a .pt file, skip re-processing
            if voice_id in self.voice_prompts:
                continue

            txt_path = wav_path.with_suffix(".txt")
            if not txt_path.exists():
                print(f"⚠️  Skipping '{voice_id}': No matching .txt file found.")
                continue

            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    ref_text = f.read().strip()

                print(f" >> Compiling voice '{voice_id}' from audio...")
                
                # Create the prompt (this is the heavy processing step)
                prompt = self.model.create_voice_clone_prompt(
                    ref_audio=str(wav_path),
                    ref_text=ref_text,
                    x_vector_only_mode=False,
                )
                
                self.voice_prompts[voice_id] = prompt
                

                pt_path = wav_path.with_suffix(".pt")
                torch.save(prompt, pt_path)
                print(f"    (Saved binary cache to {pt_path})")
                
            except Exception as e:
                print(f"Error compiling voice '{voice_id}': {e}")

        # Summary
        loaded_keys = list(self.voice_prompts.keys())
        if loaded_keys:
            if self.default_voice_id not in loaded_keys:
                 self.default_voice_id = loaded_keys[0]
            print(f"✅ Loaded {len(loaded_keys)} voices: {loaded_keys}")
        else:
            print("❌ No valid voices loaded. Ensure you have .pt files OR pairs of .wav and .txt files in /refs.")

    def stream_audio(self, text: str, voice_id: str = None, overrides: Dict = None):
        if not self.model:
            raise RuntimeError("Model not initialized")

        # Select voice
        target_voice = voice_id or self.default_voice_id
        prompt_items = self.voice_prompts.get(target_voice)

        if not prompt_items:
            # Fallback to default if specific voice fails
            print(f"⚠️ Voice '{target_voice}' not found. Falling back to '{self.default_voice_id}'.")
            prompt_items = self.voice_prompts.get(self.default_voice_id)
        
        if not prompt_items:
             raise RuntimeError("No voice prompts loaded. Cannot generate audio.")

        sentences = TextProcessor.split_sentences(text)
        print(f"Processing {len(sentences)} sentences using voice: '{target_voice}'")

        gen_kwargs = self.default_gen_kwargs.copy()
        if overrides:
            gen_kwargs.update(overrides)
            
        print(f" >> Gen Params: Temp={gen_kwargs.get('temperature')}, Voice={target_voice}")

        target_subtype = 'PCM_16'

        # ‼️ Acquire lock to ensure only one request uses the GPU at a time
        with self.lock:
            for i, sentence in enumerate(sentences):
                if not sentence.strip():
                    continue

                try:
                    print(f" >> Generating sentence {i+1}/{len(sentences)}...")
                    wavs, sr = self.model.generate_voice_clone(
                        text=sentence,
                        language="English",
                        voice_clone_prompt=prompt_items,
                        **gen_kwargs
                    )

                    audio_data = wavs[0].cpu().numpy() if torch.is_tensor(wavs[0]) else wavs[0]
                    buffer = io.BytesIO()
                    
                    # ‼️ Server always sends header if it's the start of *this* request's stream
                    # When client sends sentence-by-sentence, every response starts with i=0
                    if i == 0:
                        sf.write(buffer, audio_data, sr, format='WAV', subtype=target_subtype)
                    else:
                        sf.write(buffer, audio_data, sr, format='RAW', subtype=target_subtype)

                    buffer.seek(0)
                    yield buffer.read()
                    
                except Exception as e:
                    print(f"Error generating sentence: {e}")
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
    voice: Optional[str] = None

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
        tts_engine.stream_audio(request.text, request.voice, overrides),
        media_type="audio/wav"
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8123, workers=1)

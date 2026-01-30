import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

prompt_items = model.create_voice_clone_prompt( 
    ref_audio = "./glow_ref.wav",
    ref_text  = "The ground was black, still the foliage that grew from it and around it displayed the richest jeweled homes.",
    x_vector_only_mode=False,
)

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

wavs, sr = model.generate_voice_clone(
    text="Hello there, my name is Cortana. What would you like to ask me about today?",
    language="English",
    voice_clone_prompt=prompt_items,
    **common_gen_kwargs
)

sf.write("output_voice_clone.wav", wavs[0], sr)

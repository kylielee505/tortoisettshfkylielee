import os
import torch
import gradio as gr
import torchaudio
import time
from datetime import datetime
from tortoise.api import TextToSpeech
from tortoise.utils.text import split_and_recombine_text
from tortoise.utils.audio import load_audio, load_voice, load_voices

VOICE_OPTIONS = [
    "angie",
    "deniro",
    "freeman",
    "halle",
    "lj",
    "myself",
    "pat2",
    "snakes",
    "tom",
    "daws",
    "dreams",
    "grace",
    "lescault",
    "weaver",
    "applejack",
    "daniel",
    "emma",
    "geralt",
    "jlaw",
    "mol",
    "pat",
    "rainbow",
    "tim_reynolds",
    "atkins",
    "dotrice",
    "empire",
    "kennard",
    "mouse",
    "william",
    "jane_eyre",
    "random",  # special option for random voice
    "disabled",  # special option for disabled voice
]


def inference(
    text,
    script,
    voice,
    voice_b,
    preset,
    seed,
    split_by_newline,
):
    if text is None or text.strip() == "":
        with open(script.name) as f:
            text = f.read()
        if text.strip() == "":
            raise gr.Error("Please provide either text or script file with content.")

    if split_by_newline == "Yes":
        texts = list(filter(lambda x: x.strip() != "", text.split("\n")))
    else:
        texts = split_and_recombine_text(text)

    voices = [voice]
    if voice_b != "disabled":
        voices.append(voice_b)

    if len(voices) == 1:
        voice_samples, conditioning_latents = load_voice(voice)
    else:
        voice_samples, conditioning_latents = load_voices(voices)

    start_time = time.time()

    all_parts = []
    for j, text in enumerate(texts):
        gen = tts.tts_with_preset(
            text,
            voice_samples=voice_samples,
            conditioning_latents=conditioning_latents,
            preset=preset,
            k=1,
            use_deterministic_seed=seed,
        )

        audio_ = gen.squeeze(0).cpu()
        all_parts.append(audio_)

    full_audio = torch.cat(all_parts, dim=-1)

    # os.makedirs("outputs", exist_ok=True)
    # torchaudio.save(os.path.join("outputs", f"{name}.wav"), full_audio, 24000)

    with open("Tortoise_TTS_Runs_Scripts.log", "a") as f:
        f.write(
            f"{datetime.now()} | Voice: {','.join(voices)} | Text: {text} | Quality: {preset} | Time Taken (s): {time.time()-start_time} | Seed: {seed}\n"
        )

    output_texts = [f"({j+1}) {texts[j]}" for j in range(len(texts))]

    return ((24000, full_audio.squeeze().cpu().numpy()), "\n".join(output_texts))


def main():
    title = "Tortoise TTS 🐢"
    description = "A text-to-speech interface for Tortoise TTS, Strong multi-voice capabilities, Highly realistic prosody and intonation."
    text = gr.Textbox(
        lines=4,
        label="Text (Provide either text, or upload a newline separated text file below):",
    )
    script = gr.File(label="Upload a text file")

    preset = gr.Radio(
        ["ultra_fast", "fast", "standard", "high_quality"],
        value="fast",
        label="Preset mode (determines quality with tradeoff over speed):",
        type="value",
    )
    voice = gr.Dropdown(
        VOICE_OPTIONS, value="angie", label="Select voice:", type="value"
    )
    voice_b = gr.Dropdown(
        VOICE_OPTIONS,
        value="disabled",
        label="(Optional) Select second voice:",
        type="value",
    )
    seed = gr.Number(value=0, precision=0, label="Seed (for reproducibility):")

    split_by_newline = gr.Radio(
        ["Yes", "No"],
        label="Split by newline (If [No], it will automatically try to find relevant splits):",
        type="value",
        value="No",
    )

    output_audio = gr.Audio(label="Combined audio:")
    output_text = gr.Textbox(label="Split texts with indices:", lines=10)

    interface = gr.Interface(
        fn=inference,
        inputs=[
            text,
            script,
            voice,
            voice_b,
            preset,
            seed,
            split_by_newline,
        ],
        title=title,
        description=description,
        outputs=[output_audio, output_text],
    )
    interface.launch()


if __name__ == "__main__":
    tts = TextToSpeech(kv_cache=True, use_deepspeed=True, half=True)

    with open("Tortoise_TTS_Runs_Scripts.log", "a") as f:
        f.write(
            f"\n\n-------------------------Tortoise TTS Scripts Logs, {datetime.now()}-------------------------\n"
        )

    main()
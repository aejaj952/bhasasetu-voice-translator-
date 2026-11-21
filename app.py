"""
Bhasasetu AI Engine - Hindi <-> Santali speech translator (Gradio)

Pipeline
--------
Teacher Mode (Hindi -> Santali):
    Hindi speech --[Whisper ASR]--> Hindi text --[NLLB-200 MT]--> Santali text (Ol Chiki)

Student Mode (Santali -> Hindi):
    Santali speech --[MMS ASR, "sat" adapter]--> Santali text --[NLLB-200 MT]--> Hindi text

Every recognised sentence goes through a real translation model. There is no
phrasebook, no fuzzy matching and no static fallback.

To swap models, edit the CONFIG block below - nothing else needs to change.
"""

import logging
import re
from functools import lru_cache

import gradio as gr
import librosa
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoProcessor,
    AutoTokenizer,
    Wav2Vec2ForCTC,
    pipeline,
)

# --------------------------------------------------------------------------- #
# CONFIG - change checkpoints / language codes here
# --------------------------------------------------------------------------- #

# Hindi ASR. Any Whisper checkpoint works (openai/whisper-medium,
# openai/whisper-large-v3, or a Hindi fine-tune from the Hub).
HINDI_ASR_MODEL = "openai/whisper-small"

# Santali ASR. Whisper has no Santali, so we use Meta MMS, which loads a
# per-language adapter. NOTE: license is CC-BY-NC 4.0 (non-commercial).
# To use a Santali fine-tuned Whisper or wav2vec2 model instead, replace
# `load_santali_asr` / `transcribe_santali` below.
SANTALI_ASR_MODEL = "facebook/mms-1b-all"
SANTALI_ASR_LANG = "sat"  # ISO 639-3 adapter code

# Translation. NLLB-200 translates Hindi <-> Santali directly.
# Upgrade path: "facebook/nllb-200-distilled-1.3B" or "facebook/nllb-200-3.3B".
MT_MODEL = "facebook/nllb-200-distilled-600M"
HINDI_CODE = "hin_Deva"    # Hindi, Devanagari
SANTALI_CODE = "sat_Olck"  # Santali, Ol Chiki script

ASR_SAMPLE_RATE = 16_000
MIN_AUDIO_SECONDS = 0.3
MAX_NEW_TOKENS = 256
NUM_BEAMS = 4

TEACHER_MODE = "Teacher Mode (Hindi -> Santhali)"
STUDENT_MODE = "Student Mode (Santhali -> Hindi)"

# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("bhasasetu")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_FP16 = DEVICE == "cuda"
log.info("Using device: %s", DEVICE)


class UserFacingError(Exception):
    """An error whose message is safe and useful to show in the UI."""


# --------------------------------------------------------------------------- #
# Model loaders (lazy + cached: first request loads, later requests reuse)
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=1)
def load_hindi_asr():
    return pipeline(
        "automatic-speech-recognition",
        model=HINDI_ASR_MODEL,
        device=0 if DEVICE == "cuda" else -1,
    )


@lru_cache(maxsize=1)
def load_santali_asr():
    try:
        processor = AutoProcessor.from_pretrained(
            SANTALI_ASR_MODEL, target_lang=SANTALI_ASR_LANG
        )
        model = Wav2Vec2ForCTC.from_pretrained(
            SANTALI_ASR_MODEL,
            target_lang=SANTALI_ASR_LANG,
            ignore_mismatched_sizes=True,
        )
    except OSError as e:
        raise UserFacingError(
            f"Could not load the '{SANTALI_ASR_LANG}' adapter from "
            f"{SANTALI_ASR_MODEL}. Check the language code / model name in "
            f"CONFIG. ({e})"
        ) from e
    model.to(DEVICE).eval()
    if USE_FP16:
        model.half()
    return processor, model


@lru_cache(maxsize=1)
def load_translator():
    tokenizer = AutoTokenizer.from_pretrained(MT_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(MT_MODEL)
    model.to(DEVICE).eval()
    if USE_FP16:
        model.half()
    return tokenizer, model


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def load_audio(path):
    """Load any audio file as a mono 16 kHz float array; reject empty audio."""
    audio, _ = librosa.load(path, sr=ASR_SAMPLE_RATE, mono=True)
    if len(audio) < MIN_AUDIO_SECONDS * ASR_SAMPLE_RATE:
        raise UserFacingError("Recording is empty or too short. Please record again.")
    if float(abs(audio).max()) < 1e-4:
        raise UserFacingError("Recording is silent. Check your microphone and try again.")
    return audio


def has_ol_chiki(text):
    """True if the text contains Ol Chiki characters (U+1C50-U+1C7F)."""
    return any("\u1c50" <= ch <= "\u1c7f" for ch in text)


def split_sentences(text):
    """Split on sentence-ending punctuation (incl. the Devanagari danda)."""
    parts = re.split(r"(?<=[।.?!])\s+", text.strip())
    return [p for p in parts if p.strip()]


# --------------------------------------------------------------------------- #
# ASR
# --------------------------------------------------------------------------- #

def transcribe_hindi(audio):
    asr = load_hindi_asr()
    result = asr(
        {"raw": audio, "sampling_rate": ASR_SAMPLE_RATE},
        chunk_length_s=30,  # handles recordings longer than Whisper's 30 s window
        generate_kwargs={
            "language": "hi",
            "task": "transcribe",
            "repetition_penalty": 1.3,
            "no_repeat_ngram_size": 3,
        },
    )
    return result["text"].strip()


def transcribe_santali(audio):
    processor, model = load_santali_asr()
    inputs = processor(audio, sampling_rate=ASR_SAMPLE_RATE, return_tensors="pt")
    input_values = inputs["input_values"].to(DEVICE)
    if USE_FP16:
        input_values = input_values.half()
    with torch.no_grad():
        logits = model(input_values).logits
    ids = torch.argmax(logits, dim=-1)[0]
    return processor.decode(ids).strip()


# --------------------------------------------------------------------------- #
# Machine translation
# --------------------------------------------------------------------------- #

def translate_text(text, src_code, tgt_code):
    """Translate a (possibly multi-sentence) string with NLLB-200."""
    tokenizer, model = load_translator()

    tgt_id = tokenizer.convert_tokens_to_ids(tgt_code)
    if tgt_id is None or tgt_id == tokenizer.unk_token_id:
        raise UserFacingError(
            f"Translation model {MT_MODEL} does not know language code '{tgt_code}'."
        )

    tokenizer.src_lang = src_code
    sentences = split_sentences(text) or [text]
    inputs = tokenizer(
        sentences, return_tensors="pt", padding=True, truncation=True, max_length=256
    ).to(DEVICE)

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            forced_bos_token_id=tgt_id,
            num_beams=NUM_BEAMS,
            max_new_tokens=MAX_NEW_TOKENS,
        )
    outputs = tokenizer.batch_decode(generated, skip_special_tokens=True)
    return " ".join(o.strip() for o in outputs)


# --------------------------------------------------------------------------- #
# Gradio handler (generator -> streams status updates to the UI)
# --------------------------------------------------------------------------- #

def translate_speech(mode, audio_path):
    """Yields (status, asr_text, translated_text)."""
    if not audio_path:
        yield "⚠️ No audio received. Click the microphone, record, then stop.", "", ""
        return

    teacher = mode == TEACHER_MODE
    src_name, tgt_name = ("Hindi", "Santali") if teacher else ("Santali", "Hindi")
    src_code, tgt_code = (
        (HINDI_CODE, SANTALI_CODE) if teacher else (SANTALI_CODE, HINDI_CODE)
    )

    stage = "reading audio"
    try:
        yield "⏳ Reading audio...", "", ""
        audio = load_audio(audio_path)

        stage = f"{src_name} speech recognition"
        yield f"⏳ Transcribing {src_name} speech (first run downloads the model)...", "", ""
        recognized = transcribe_hindi(audio) if teacher else transcribe_santali(audio)
        if not recognized:
            yield "⚠️ No speech was recognised. Please speak clearly and try again.", "", ""
            return

        warning = ""
        if not teacher and not has_ol_chiki(recognized):
            # The MT model expects Ol Chiki for Santali input.
            warning = (
                " ⚠️ The Santali transcript is not in Ol Chiki script, so "
                "translation quality may be poor."
            )

        stage = f"{src_name} -> {tgt_name} translation"
        yield f"⏳ Translating {src_name} -> {tgt_name}...", recognized, ""
        translated = translate_text(recognized, src_code, tgt_code)
        if not translated.strip():
            yield "⚠️ The translation model returned an empty result.", recognized, ""
            return

        yield f"✅ Done.{warning}", recognized, translated

    except UserFacingError as e:
        yield f"⚠️ {e}", "", ""
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        yield f"❌ GPU out of memory during {stage}. Try a shorter recording.", "", ""
    except Exception as e:  # noqa: BLE001 - last-resort guard for the UI
        log.exception("Failure during %s", stage)
        yield f"❌ Failed during {stage}: {e}", "", ""


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #

with gr.Blocks(title="Bhasasetu AI Engine") as demo:
    gr.Markdown(
        "# Bhasasetu AI Engine\n"
        "Speech-to-text translation between Hindi and Santali. "
        "Santali text is shown in Ol Chiki script."
    )
    mode = gr.Dropdown(
        [TEACHER_MODE, STUDENT_MODE],
        value=TEACHER_MODE,
        label="Translation Mode",
    )
    audio = gr.Audio(sources=["microphone"], type="filepath", label="Record Voice")
    run = gr.Button("Translate", variant="primary")
    status = gr.Textbox(label="Status", interactive=False)
    asr_out = gr.Textbox(label="ASR Text Output", interactive=False)
    mt_out = gr.Textbox(label="Translated Output", interactive=False)

    run.click(translate_speech, inputs=[mode, audio], outputs=[status, asr_out, mt_out])

if __name__ == "__main__":
    demo.queue().launch(share=True)

# ai_model.py
# Uses the open-source "mistralai/Mistral-7B-Instruct-v0.2" model via Hugging Face
# Runs fully locally — no API key needed.
#
# On first run it will download the model (~14GB). For a lighter alternative,
# swap MODEL_NAME to "TinyLlama/TinyLlama-1.1B-Chat-v1.0" (~2GB).

from transformers import pipeline
import torch

# ── Model selection ──────────────────────────────────────────────────────────
# Full quality (needs ~16GB RAM / GPU VRAM):
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

# Lightweight fallback (needs ~4GB RAM, decent on CPU):
# MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# ─────────────────────────────────────────────────────────────────────────────

# Load once at startup — reused for every request
print(f"[humour.ai] Loading model: {MODEL_NAME} …")
_pipe = pipeline(
    "text-generation",
    model=MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",          # uses GPU if available, else CPU
)
print("[humour.ai] Model ready ✓")


# ── Prompt builder ────────────────────────────────────────────────────────────
MODE_INSTRUCTIONS = {
    "one-liner": (
        "Write a single one-liner where the first half sets up an innocent, "
        "seemingly normal scene — and the second half recontextualizes it with "
        "a twist the audience needs a second to decode."
    ),
    "pun": (
        "Write a pun-based joke where the wordplay is layered — the surface "
        "meaning reads normally, but the double meaning clicks a beat later "
        "like a key turning in a lock."
    ),
    "story": (
        "Write a 3-sentence silly story where the final sentence reframes "
        "everything before it. The reader should finish it, pause, then burst "
        "out laughing as the full picture assembles."
    ),
    "sarcastic": (
        "Write a dry, sarcastic observation that sounds almost sincere — until "
        "the audience catches the irony a moment later and realises the joke "
        "was hiding in plain sight."
    ),
    "knock-knock": (
        "Write a knock-knock joke where the answer to 'who's there?' seems "
        "completely random — until the punchline suddenly makes the connection click."
    ),
    "roast": (
        "Write a gentle self-roast where the first part sounds like a "
        "compliment — then the punchline reveals the self-deprecating twist, "
        "leaving the audience with a delayed 'oh wait…'"
    ),
}

SYSTEM_PROMPT = """You are humour.ai — a comedy AI that specialises in jokes with a DELAYED EUREKA MOMENT.

Your signature style:
- The joke should not be immediately obvious. Plant the setup innocently, then let the punchline recontextualise everything.
- The audience should need 1–2 seconds to "get it" — then feel the satisfying click of the punchline snapping into place.
- Be silly and imaginative — use unexpected metaphors, absurd logic, or surprising connections between unrelated things.
- Be slightly sarcastic in a warm, friendly way — not mean, just knowingly wry.
- Always clean and family-friendly.

CRITICAL: Keep it SHORT (max 3 sentences). Do NOT explain the joke. Trust the audience to find the eureka. No preamble — deliver the joke directly."""


def _build_prompt(situation: str, mode: str) -> str:
    mode_instruction = MODE_INSTRUCTIONS.get(mode, MODE_INSTRUCTIONS["one-liner"])
    user_message = (
        f"{mode_instruction}\n\n"
        f"Make a {mode} joke about: {situation}"
    )
    # Mistral instruct format
    prompt = (
        f"<s>[INST] {SYSTEM_PROMPT}\n\n{user_message} [/INST]"
    )
    return prompt


# ── Main function called by app.py ────────────────────────────────────────────
def generate_joke(situation: str, mode: str) -> dict:
    """
    Returns {"success": True, "joke": "..."} or {"success": False, "error": "..."}
    """
    try:
        prompt = _build_prompt(situation, mode)
        outputs = _pipe(
            prompt,
            max_new_tokens=180,
            do_sample=True,
            temperature=0.85,
            top_p=0.92,
            repetition_penalty=1.15,
            pad_token_id=_pipe.tokenizer.eos_token_id,
        )
        # Strip the input prompt from the output
        full_text = outputs[0]["generated_text"]
        joke = full_text.split("[/INST]")[-1].strip()
        return {"success": True, "joke": joke}

    except Exception as e:
        return {"success": False, "error": str(e)}

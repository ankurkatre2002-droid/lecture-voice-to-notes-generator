"""
Lecture Voice-to-Notes Generator
=====================================
Pipeline:
  Audio Input
      ↓
  Whisper (Speech-to-Text)        ← runs locally, no API key
      ↓
  Raw Transcript
      ↓
  Text Processing (rule-based)    ← no model needed for summary/quiz
      ↓
  Summary + Key Points + Quiz + Flashcards
      ↓
  Gradio UI Display

100% Free — No API key needed!
"""

import gradio as gr
import whisper
import re
import torch

# ── Load Whisper once at startup ──────────────────────────────────────────────
print("Loading Whisper model (tiny)...")
whisper_model = whisper.load_model("tiny")
print("✅ Whisper loaded")


# ── Stage 1: Speech to Text ───────────────────────────────────────────────────
def transcribe_audio(audio_path: str) -> str:
    if audio_path is None:
        raise ValueError("No audio file provided.")
    result = whisper_model.transcribe(audio_path, fp16=False)
    return result["text"].strip()


# ── Stage 2: Clean Transcript ─────────────────────────────────────────────────
def clean_transcript(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(\.\s*){2,}", ". ", text)
    return text.strip()


# ── Stage 3: Summarize (rule-based, no model needed) ─────────────────────────
def summarize_text(text: str) -> str:
    """Extractive summarization — picks most important sentences."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in sentences if len(s.split()) > 5]

    if not sentences:
        return "Not enough content to summarize."

    # Score each sentence by importance keywords
    importance_words = {
        "important", "key", "main", "because", "therefore", "result",
        "shows", "means", "defined", "called", "known", "example",
        "first", "second", "third", "finally", "conclusion", "however",
        "significant", "major", "critical", "essential", "primary"
    }

    def score(s):
        words = s.lower().split()
        keyword_score = sum(1 for w in words if w in importance_words)
        length_score  = min(len(words) / 20, 1.0)  # prefer medium length
        return keyword_score + length_score

    scored    = sorted(enumerate(sentences), key=lambda x: score(x[1]), reverse=True)
    top_idxs  = sorted([i for i, _ in scored[:5]])  # top 5, keep order
    summary   = " ".join(sentences[i] for i in top_idxs)
    return summary


# ── Stage 4: Key Points ───────────────────────────────────────────────────────
def extract_key_points(text: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in sentences if len(s.split()) > 6]

    trigger_words = {
        "is", "are", "means", "defined", "important", "key", "main",
        "because", "therefore", "result", "shows", "called", "known",
        "example", "such as", "first", "second", "third", "finally",
        "however", "significant", "must", "should", "can", "will"
    }

    def score(s):
        return sum(1 for w in s.lower().split() if w in trigger_words)

    top     = sorted(sentences, key=score, reverse=True)[:7]
    ordered = [s for s in sentences if s in top]
    return "\n".join(f"• {s}" for s in ordered) if ordered else "\n".join(f"• {s}" for s in sentences[:5])


# ── Stage 5: Quiz (fill in the blank) ────────────────────────────────────────
def generate_quiz(text: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in sentences if len(s.split()) >= 8]

    quiz_lines = []
    count      = 1

    for sentence in sentences:
        if count > 5:
            break
        words      = sentence.split()
        candidates = [(i, w) for i, w in enumerate(words)
                      if len(w) > 5 and w.isalpha() and i > 1]
        if not candidates:
            continue
        idx, answer  = max(candidates, key=lambda x: len(x[1]))
        blanked      = words.copy()
        blanked[idx] = "_______"
        question     = " ".join(blanked)
        quiz_lines.append(f"Q{count}. {question}")
        quiz_lines.append(f"    ✅ Answer: {answer}\n")
        count += 1

    return "\n".join(quiz_lines) if quiz_lines else "Not enough content to generate a quiz."


# ── Stage 6: Flashcards ───────────────────────────────────────────────────────
def generate_flashcards(text: str) -> str:
    patterns = [
        r"([A-Z][a-z]+(?:\s[a-z]+){0,3})\s+is\s+([^.]{10,80})\.",
        r"([A-Z][a-z]+(?:\s[a-z]+){0,3})\s+means\s+([^.]{10,80})\.",
        r"([A-Z][a-z]+(?:\s[a-z]+){0,3})\s+refers to\s+([^.]{10,80})\.",
    ]
    cards = []
    seen  = set()

    for pattern in patterns:
        for term, definition in re.findall(pattern, text):
            term = term.strip()
            if term.lower() not in seen and len(cards) < 6:
                seen.add(term.lower())
                cards.append(f"📌 {term}\n   → {definition.strip()}")

    if not cards:
        sentences = re.split(r"(?<=[.!?])\s+", text)[:5]
        for i, s in enumerate(sentences, 1):
            if len(s.split()) > 6:
                cards.append(f"📌 Concept {i}\n   → {s.strip()}")

    return "\n\n".join(cards) if cards else "No flashcard patterns found in this lecture."


# ── Full Pipeline ─────────────────────────────────────────────────────────────
def process_lecture(audio_path):
    if audio_path is None:
        return "⚠️ Please upload an audio file.", "", "", "", ""
    try:
        # Step 1 & 2: Transcribe + Clean
        raw        = transcribe_audio(audio_path)
        if not raw:
            return "❌ Could not transcribe. Try a clearer recording.", "", "", "", ""
        transcript = clean_transcript(raw)

        # Steps 3-6: Process
        summary    = summarize_text(transcript)
        key_points = extract_key_points(transcript)
        quiz       = generate_quiz(transcript)
        flashcards = generate_flashcards(transcript)

        return transcript, summary, key_points, quiz, flashcards

    except Exception as e:
        return f"❌ Error: {str(e)}", "", "", "", ""


# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="🎓 Lecture Voice-to-Notes Generator") as demo:

    gr.Markdown("""
    # 🎓 Lecture Voice-to-Notes Generator
    **Upload a lecture audio → instant transcript, summary, key points, quiz & flashcards**
    > 🔒 100% Free · No API Key · Powered by OpenAI Whisper
    """)

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                label="📢 Upload Lecture Audio",
                type="filepath",
                sources=["upload", "microphone"]
            )
            run_btn = gr.Button("🚀 Generate Notes", variant="primary", size="lg")
            gr.Markdown("""
            **Supported:** MP3, WAV, M4A, OGG, FLAC
            **Tips:**
            - Clear audio, minimal background noise
            - English works best
            - Under 5 min processes faster
            """)

    gr.Markdown("---")

    with gr.Tabs():
        with gr.Tab("📝 Transcript"):
            transcript_out = gr.Textbox(label="Full Transcript", lines=12,
                                        placeholder="Transcript will appear here...")
        with gr.Tab("📋 Summary"):
            summary_out    = gr.Textbox(label="Summary",         lines=8,
                                        placeholder="Summary will appear here...")
        with gr.Tab("🔑 Key Points"):
            keypoints_out  = gr.Textbox(label="Key Points",      lines=10,
                                        placeholder="Key points will appear here...")
        with gr.Tab("🧪 Quiz"):
            quiz_out       = gr.Textbox(label="Fill-in-the-Blank Quiz", lines=12,
                                        placeholder="Quiz will appear here...")
        with gr.Tab("🃏 Flashcards"):
            flashcards_out = gr.Textbox(label="Flashcards",      lines=12,
                                        placeholder="Flashcards will appear here...")

    run_btn.click(
        fn=process_lecture,
        inputs=[audio_input],
        outputs=[transcript_out, summary_out, keypoints_out, quiz_out, flashcards_out]
    )

demo.launch(theme=gr.themes.Soft())

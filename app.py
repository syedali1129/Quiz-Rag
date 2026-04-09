import os
import shutil
import tempfile
from pathlib import Path

from flask import Flask, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

from students_models import StudentModel
from tutor_core import (
    explain_concept,
    extract_topics,
    generate_question,
    ingest_document,
)


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "quiz-rag-dev-key")
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024


def build_state() -> dict:
    return {
        "student": StudentModel.load(str(BASE_DIR / "student_model.json")),
        "vectorstore": None,
        "topics": [],
        "current_question": None,
        "current_topic": None,
        "current_difficulty": None,
        "last_feedback": None,
        "hint_shown": False,
        "doc_loaded": False,
        "status_message": None,
        "status_variant": "info",
        "active_explanation": None,
        "uploaded_filename": None,
    }


STATE = build_state()


def student_model_path() -> str:
    return str(BASE_DIR / "student_model.json")


def reset_runtime_state(preserve_progress: bool = False) -> None:
    fresh = build_state()
    if preserve_progress:
        fresh["student"] = STATE["student"]
    STATE.clear()
    STATE.update(fresh)


def set_status(message: str, variant: str = "info") -> None:
    STATE["status_message"] = message
    STATE["status_variant"] = variant


def consume_status() -> tuple[str | None, str]:
    message = STATE.get("status_message")
    variant = STATE.get("status_variant", "info")
    STATE["status_message"] = None
    STATE["status_variant"] = "info"
    return message, variant


def serialize_topic_cards() -> list[dict]:
    cards = []
    for topic in STATE["topics"]:
        topic_state = STATE["student"].topics.get(topic)
        accuracy = int(round((topic_state.accuracy if topic_state else 0.0) * 100))
        cards.append(
            {
                "name": topic,
                "accuracy": accuracy,
                "difficulty": topic_state.current_difficulty if topic_state else "easy",
                "attempts": topic_state.attempts if topic_state else 0,
            }
        )
    return cards


def summary_metrics() -> dict:
    student = STATE["student"]
    answered = student.total_answered
    accuracy = int(round((student.session_correct / answered) * 100)) if answered else 0
    mastered = sum(1 for state in student.topics.values() if state.accuracy >= 0.7 and state.attempts > 0)
    return {
        "answered": answered,
        "accuracy": accuracy,
        "topics": len(STATE["topics"]),
        "mastered": mastered,
    }


@app.route("/", methods=["GET"])
def index():
    message, variant = consume_status()
    return render_template(
        "index.html",
        doc_loaded=STATE["doc_loaded"],
        uploaded_filename=STATE["uploaded_filename"],
        metrics=summary_metrics(),
        topic_cards=serialize_topic_cards(),
        current_question=STATE["current_question"],
        current_topic=STATE["current_topic"],
        current_difficulty=STATE["current_difficulty"],
        hint_shown=STATE["hint_shown"],
        last_feedback=STATE["last_feedback"],
        explanation=STATE["active_explanation"],
        mastery_summary=STATE["student"].get_mastery_summary(),
        status_message=message,
        status_variant=variant,
    )


@app.route("/upload", methods=["POST"])
def upload():
    uploaded = request.files.get("document")
    if not uploaded or not uploaded.filename:
        set_status("Choose a PDF first so I can build the quiz knowledge base.", "warning")
        return redirect(url_for("index"))

    if not uploaded.filename.lower().endswith(".pdf"):
        set_status("Only PDF files are supported right now.", "warning")
        return redirect(url_for("index"))

    temp_dir = tempfile.mkdtemp(dir=UPLOAD_DIR)
    safe_name = secure_filename(uploaded.filename)
    if not safe_name:
        set_status("That filename isn't usable. Rename the PDF and try again.", "warning")
        shutil.rmtree(temp_dir, ignore_errors=True)
        return redirect(url_for("index"))

    temp_path = Path(temp_dir) / safe_name

    try:
        uploaded.save(temp_path)
        vectorstore = ingest_document(str(temp_path))
        topics = extract_topics(vectorstore)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    reset_runtime_state(preserve_progress=True)
    STATE["vectorstore"] = vectorstore
    STATE["topics"] = topics
    STATE["doc_loaded"] = True
    STATE["uploaded_filename"] = safe_name
    set_status(f"Loaded {safe_name} and mapped {len(topics)} key topics.", "success")
    return redirect(url_for("index"))


@app.route("/next-question", methods=["POST"])
def next_question():
    if not STATE["doc_loaded"] or not STATE["vectorstore"] or not STATE["topics"]:
        set_status("Upload a PDF before generating questions.", "warning")
        return redirect(url_for("index"))

    topic = STATE["student"].select_next_topic(STATE["topics"])
    STATE["student"].ensure_topic(topic)
    difficulty = STATE["student"].topics[topic].current_difficulty
    question = generate_question(STATE["vectorstore"], topic, difficulty)

    STATE["current_question"] = question
    STATE["current_topic"] = topic
    STATE["current_difficulty"] = difficulty
    STATE["last_feedback"] = None
    STATE["active_explanation"] = None
    STATE["hint_shown"] = False
    return redirect(url_for("index"))


@app.route("/toggle-hint", methods=["POST"])
def toggle_hint():
    if STATE["current_question"]:
        STATE["hint_shown"] = not STATE["hint_shown"]
    return redirect(url_for("index"))


@app.route("/submit-answer", methods=["POST"])
def submit_answer():
    answer = request.form.get("answer", "").strip()
    question = STATE["current_question"]

    if not question or not STATE["current_topic"]:
        set_status("Generate a question before submitting an answer.", "warning")
        return redirect(url_for("index"))

    if not answer:
        set_status("Write an answer before submitting.", "warning")
        return redirect(url_for("index"))

    from tutor_core import evaluate_answer

    result = evaluate_answer(question["question"], question["answer"], answer)
    is_correct = result.get("correct", False)
    feedback = result.get("feedback", "")

    STATE["student"].record_answer(STATE["current_topic"], is_correct)
    STATE["student"].save(student_model_path())

    explanation = None
    topic_state = STATE["student"].topics[STATE["current_topic"]]
    if not is_correct and topic_state.needs_explanation:
        explanation = explain_concept(
            STATE["vectorstore"],
            STATE["current_topic"],
            question["question"],
            question["answer"],
        )

    STATE["last_feedback"] = {
        "correct": is_correct,
        "feedback": feedback,
        "answer": question["answer"],
        "submitted_answer": answer,
    }
    STATE["active_explanation"] = explanation
    STATE["current_question"] = None
    STATE["hint_shown"] = False
    return redirect(url_for("index"))


@app.route("/reset", methods=["POST"])
def reset():
    model_path = Path(student_model_path())
    if model_path.exists():
        model_path.unlink()
    reset_runtime_state()
    set_status("Progress cleared. Upload a PDF to start a fresh session.", "success")
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)

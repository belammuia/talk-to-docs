import io
import os

import requests
from flask import Flask, render_template, request, send_file
from flask_migrate import Migrate
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sqlalchemy import text as sql_text

from extensions import db
from models import Chunks

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://app:app@localhost:5433/docs",
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.secret_key = os.getenv("SECRET_KEY", "dev-secret")

db.init_app(app)
migrate = Migrate(app, db)

DG_KEY = os.environ.get("DEEPGRAM_API_KEY")
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
embedder = SentenceTransformer("all-MiniLM-L6-v2")

with app.app_context():
    db.session.execute(sql_text("CREATE EXTENSION IF NOT EXISTS vector"))
    db.session.commit()
    db.create_all()


# ---------- helpers ----------

def chunk_text(text, size=120, overlap=20):
    words = text.split()
    step = size - overlap
    chunks = []
    for i in range(0, len(words), step):
        piece = " ".join(words[i:i + size])
        if piece:
            chunks.append(piece)
    return chunks


def answer_the_question(question, chunks):
    context = "\n\n".join(f"[Page {c.page}] {c.text}" for c in chunks)
    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer the question using ONLY the context provided. "
                    "If the answer is not in the context, say: "
                    "\"I don't know based on this document.\" "
                    "Keep it short and conversational, with no bullet points, "
                    "because it will be read aloud."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ],
    )
    return response.choices[0].message.content


# ---------- routes ----------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    file = request.files.get("file")
    if not file or not file.filename.lower().endswith(".pdf"):
        return {"error": "Please choose a PDF file"}, 400

    try:
        reader = PdfReader(io.BytesIO(file.read()))
    except Exception:
        return {"error": "Could not read that PDF"}, 400

    count = 0
    pages_with_text = 0
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        pages_with_text += 1
        for piece in chunk_text(text):
            db.session.add(
                Chunks(
                    text=piece,
                    page=page_number,
                    embedding=embedder.encode(piece).tolist(),
                )
            )
            count += 1

    if count == 0:
        return {"error": "No readable text found (scanned PDF?)"}, 400

    db.session.commit()
    return {"filename": file.filename, "pages": pages_with_text, "chunks": count}


@app.route("/ask", methods=["POST"])
def ask():
    question = request.form.get("question", "").strip()
    if not question:
        return {"error": "Ask a question"}, 400

    q_vector = embedder.encode(question).tolist()
    results = (
        Chunks.query
        .order_by(Chunks.embedding.cosine_distance(q_vector))
        .limit(4)
        .all()
    )
    if not results:
        return {"error": "Upload a PDF first"}, 400

    try:
        answer = answer_the_question(question, results)
    except Exception as e:
        return {"error": f"LLM error: {e}"}, 500

    return {
        "question": question,
        "answer": answer,
        "matches": [{"page": r.page, "text": r.text} for r in results],
    }


@app.route("/tts", methods=["POST"])
def tts():
    text = (request.json or {}).get("text", "")
    if not text:
        return {"error": "No text"}, 400
    try:
        r = requests.post(
            "https://api.deepgram.com/v1/speak?model=aura-2-thalia-en&encoding=mp3",
            headers={"Authorization": f"Token {DG_KEY}", "Content-Type": "application/json"},
            json={"text": text},
            timeout=30,
        )
        r.raise_for_status()
    except Exception as e:
        return {"error": f"TTS error: {e}"}, 500
    return send_file(io.BytesIO(r.content), mimetype="audio/mpeg")


@app.route("/transcribe", methods=["POST"])
def transcribe():
    audio = request.get_data()
    if not audio:
        return {"error": "No audio received"}, 400
    try:
        r = requests.post(
            "https://api.deepgram.com/v1/listen?model=nova-3&smart_format=true",
            headers={"Authorization": f"Token {DG_KEY}", "Content-Type": request.content_type},
            data=audio,
            timeout=20,
        )
        r.raise_for_status()
        text = r.json()["results"]["channels"][0]["alternatives"][0]["transcript"]
    except Exception as e:
        return {"error": f"Transcribe error: {e}"}, 500
    return {"text": text}


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
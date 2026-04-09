# 📚 Quiz RAG — AI-Powered Adaptive Quiz App

A Retrieval-Augmented Generation (RAG) web application that transforms PDF study materials into interactive quizzes. Built with Flask, Groq LLM, and Google Gemini embeddings, it adapts to each student's mastery level in real time.

***

## ✨ Features

- 📄 **PDF Upload** — Upload any study material and instantly generate quiz questions from it
- 🧠 **RAG Pipeline** — Google Gemini embeddings + ChromaDB vector store for accurate context retrieval
- 💡 **Hint System** — Students can request hints without immediately revealing the answer
- 📊 **Topic Radar** — Visual breakdown of performance across topics covered in the PDF
- 📈 **Session Metrics** — Tracks accuracy, questions answered, and time per session
- ✅ **Answer Review** — Detailed feedback after each answer with source context
- 🎯 **Mastery Tracking** — Monitors student progress per topic and adjusts difficulty accordingly
- 👤 **Student Model** — Persistent per-student performance profiles stored in `student_model.json`

***

## 🗂️ Project Structure

```
Quiz/
├── app.py                  # Flask entrypoint — routes and session logic
├── tutor_core.py           # RAG pipeline — retrieval, question generation, answer evaluation
├── students_models.py      # Student mastery model and progress tracking
├── student_model.json      # Persistent student performance data
├── requirements.txt        # Python dependencies
├── templates/
│   └── index.html          # Main frontend UI
├── static/
│   └── styles.css          # App styling
├── uploads/                # Uploaded PDFs (gitignored)
└── chroma_db/              # Local vector store (gitignored)
```

***

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| LLM | Groq (LLaMA / Mixtral) |
| Embeddings | Google Gemini Embeddings |
| Vector Store | ChromaDB (local) |
| Frontend | HTML, CSS (Jinja2 templates) |
| PDF Parsing | LangChain / PyMuPDF |

***

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/syedali1129/Quiz-Rag.git
cd Quiz-Rag
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
GOOGLE_API_KEY=your_google_gemini_api_key_here
```

### 5. Run the App

```bash
python app.py
```

Visit `http://localhost:5000` in your browser.

***

## 📋 Usage

1. **Upload a PDF** — Upload your study material (lecture notes, textbook chapters, etc.)
2. **Start Quiz** — The app chunks and embeds the PDF into ChromaDB
3. **Answer Questions** — RAG retrieves relevant context to generate targeted questions
4. **Use Hints** — Request a hint if you're stuck
5. **Review Answers** — See detailed feedback with source passages
6. **Track Progress** — View your topic radar and mastery scores after each session

***

## ⚙️ Environment Variables

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | API key from [console.groq.com](https://console.groq.com) |
| `GOOGLE_API_KEY` | API key from [Google AI Studio](https://aistudio.google.com) |

***

## 🚧 Known Limitations

- ChromaDB is stored locally — not suitable for serverless deployment without migration to a hosted vector store (e.g. Pinecone, Supabase pgvector)
- Uploaded PDFs are stored locally in `uploads/` — cloud storage (S3, R2) needed for production

***

## 👤 Author

**Syed Ali** — [github.com/syedali1129](https://github.com/syedali1129)

***

## 📄 License

MIT License

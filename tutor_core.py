import json
import os
import shutil
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")

CHROMA_DIR = "/tmp/chroma_db"

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.4)
llm_precise = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0)

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    transport="rest"
)

def ingest_document(file_path: str) -> Chroma:
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    # Improvement for Assignment 6: Improved chunking parameters to preserve context and reduce fragmentation.
    # Increased chunk size and overlap so related concepts aren't split unnaturally.
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=250)
    chunks = splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(
        chunks, embeddings, persist_directory=CHROMA_DIR
    )
    return vectorstore


def load_vectorstore() -> Chroma:
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)


def extract_topics(vectorstore: Chroma, n_topics: int = 6) -> list[str]:
    sample_docs = vectorstore.similarity_search("main concepts and topics", k=10)
    context = "\n\n".join(d.page_content for d in sample_docs)
    prompt = ChatPromptTemplate.from_template("""
You are an expert educator. From the content below extract {n} key learning topics.
Return ONLY a JSON array of short topic strings (2-5 words each), no explanation.

Content:
{context}

Example output: ["Linear Regression", "Gradient Descent", "Overfitting", "Cross Validation"]
""")
    chain = prompt | llm_precise | StrOutputParser()       # ✅ Bug 2 fixed
    result = chain.invoke({"context": context, "n": n_topics})
    try:
        return json.loads(result.strip())
    except Exception:
        return [t.strip().strip('"') for t in result.split(",") if t.strip()]


QUESTION_PROMPT = ChatPromptTemplate.from_template("""   
You are a quiz master. Using ONLY the context below, generate a {difficulty} difficulty
question about "{topic}".

Rules:
- easy: test recall of definitions or basic facts
- medium: test understanding or application
- hard: test analysis, edge cases, or synthesis across concepts

Return a JSON object with exactly these keys:
{{
  "question": "...",
  "answer": "...",
  "hint": "one short hint without giving away the answer",
  "type": "short_answer"
}}

Context:
{context}
""")


def generate_question(vectorstore: Chroma, topic: str, difficulty: str) -> dict:
    docs = vectorstore.similarity_search(topic, k=4)
    context = "\n\n".join(d.page_content for d in docs)
    chain = QUESTION_PROMPT | llm | StrOutputParser()      # ✅ Bug 3 fixed
    raw = chain.invoke({"topic": topic, "difficulty": difficulty, "context": context})
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()  # ✅ Bug 4 fixed
    return json.loads(raw)


EVAL_PROMPT = ChatPromptTemplate.from_template("""
You are a fair but strict grader. The student answered a question.

Question: {question}
Correct Answer: {correct_answer}
Student Answer: {student_answer}

Is the student's answer correct or substantially correct?
Reply with ONLY a JSON object: {{"correct": true/false, "feedback": "brief one-sentence feedback"}}
""")


def evaluate_answer(question: str, correct_answer: str, student_answer: str) -> dict:
    chain = EVAL_PROMPT | llm_precise | StrOutputParser()  # ✅ Bug 5 fixed
    raw = chain.invoke({
        "question": question,
        "correct_answer": correct_answer,
        "student_answer": student_answer
    })
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()  # ✅ Bug 6 fixed
    return json.loads(raw)


EXPLAIN_PROMPT = ChatPromptTemplate.from_template("""
You are a patient, brilliant tutor. The student got a question about "{topic}" wrong.

Question they got wrong: {question}
Correct answer: {correct_answer}

Using the context below, explain the concept clearly in 3 parts:
1. 📖 Core Explanation (2-3 sentences, plain language)
2. 💡 Analogy or Example (make it memorable)
3. 🔁 Key Takeaway (one sentence to remember)

Context from study material:
{context}
""")


def explain_concept(vectorstore: Chroma, topic: str, question: str, correct_answer: str) -> str:
    docs = vectorstore.similarity_search(topic, k=4)
    context = "\n\n".join(d.page_content for d in docs)
    chain = EXPLAIN_PROMPT | llm | StrOutputParser()
    return chain.invoke({
        "topic": topic,
        "question": question,
        "correct_answer": correct_answer,
        "context": context
    })

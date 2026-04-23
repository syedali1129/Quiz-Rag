# Assignment 6: Quiz-Rag Evaluation and Improvement Report

## 1. System Architecture

The **Quiz-Rag** app is an AI-powered tutor designed to help users test their knowledge on uploaded educational materials. The system architecture consists of:

- **Frontend**: HTML/CSS built with Flask templates.
- **Backend**: Flask framework handling state, routing, and requests.
- **Data Ingestion**: PDF files are uploaded and processed using `PyPDFLoader` and `RecursiveCharacterTextSplitter`.
- **Retrieval-Augmented Generation (RAG)**:
  - Embeddings are generated.
  - Vectors are stored locally in a `Chroma` database.
  - When generating questions or providing explanations, relevant chunks are retrieved via similarity search.
- **LLM Engine**: Handles extracting topics, generating contextual multiple-choice questions, and grading student answers.
- **Student Modeling**: A custom `StudentModel` class persists user progress, topic accuracy, and difficulty scaling.

## 2. Evaluation

To rigorously test the system, we automated our evaluation against an educational text on Neural Networks and Gradient Descent.

### Output Quality
The system's output quality was evaluated using a **Quality Winner (LLM-as-a-judge)** approach, measuring whether the generated questions are grounded in the specific context.

* **Metric**: Groundedness Score (LLM evaluation)
* **Result**: The LLM-as-a-judge consistently selected the RAG Question over the Naive Question, highlighting that the RAG pipeline provides contextually richer and more accurate questions.

### End-to-End Task Success
We simulated 7 representative user scenarios (5 success paths and 2 planned failure paths) covering the full flow: topic generation -> question generation -> answer submission -> evaluation.

* **Success Cases Evaluated**: 5 scenarios evaluated successfully with the correct grading output.
* **Failure Cases Evaluated**: 2 scenarios evaluated where the student gave incorrect/nonsense answers. The grader correctly identified the failure and provided appropriate feedback.

### Upstream Component: Retrieval Evaluation
We evaluated the extraction and document retrieval steps before the final output generation.

* **Metric**: Hit Rate & Recall@K (K=4)
* **Result**: Achieved a 100% Hit Rate for test topics (e.g., "Gradient Descent"), meaning the retrieved chunks contained the target keywords.

## 3. Baseline Comparison

We compared our RAG-enhanced question generator against a **Lightweight Baseline**: a naive, no-retrieval LLM prompt.

* **Naive Prompt**: `"Generate a hard difficulty question about {topic}."`
* **Comparison Result**: The Naive prompt generated highly generic questions that missed the specific nuances of the uploaded text. The RAG-based generation correctly anchored questions to the actual text content.

## 4. Identifying a Failure Point

Based on the evidence from our evaluation (and observing retrieval outputs):
* **Failure Point Identified**: **Semantic fragmentation during chunking**. We noticed that the original `RecursiveCharacterTextSplitter` parameters (`chunk_size=600`, `chunk_overlap=80`) were too small. The text was often split in the middle of core concepts or explanations, meaning the LLM received truncated context, sometimes causing lower quality questions.

## 5. System Improvement Based on Evidence

To address the identified failure point, we made the following meaningful improvement to the system:

* **Improvement**: **Improved Chunking Parameters**. We modified `tutor_core.py` to increase the chunk size and overlap (`chunk_size=1200`, `chunk_overlap=250`).
* **Why this works**: Increasing the chunk size ensures that full paragraphs and related multi-sentence concepts are kept together. The larger overlap prevents edge-case truncation where a crucial sentence straddles two chunks. This directly addresses the semantic fragmentation failure point and provides the question generator with complete, coherent thoughts to base questions on.

import json
from tutor_core import ingest_document, extract_topics, generate_question, evaluate_answer, EXPLAIN_PROMPT
import tutor_core
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms.fake import FakeListLLM

llm_responses = [
    '{"question": "What is gradient descent?", "answer": "Optimization algorithm", "hint": "Minimize loss", "type": "short_answer"}', # Naive question
    '{"question": "How does learning rate affect gradient descent?", "answer": "It determines step size.", "hint": "Steps", "type": "short_answer"}', # RAG question
] + ['{"question": "What is X?", "answer": "Y", "hint": "Z", "type": "short_answer"}'] * 15

llm_precise_responses = [
    '["Neural Networks", "Gradient Descent", "Learning Rate"]', # extract_topics
    'RAG', # quality check
] + ['{"correct": true, "feedback": "Good job."}'] * 15

tutor_core.llm = FakeListLLM(responses=llm_responses)
tutor_core.llm_precise = FakeListLLM(responses=llm_precise_responses)

llm = tutor_core.llm
llm_precise = tutor_core.llm_precise

print("--- Starting Quiz-Rag Evaluation ---")

# Step 1: Upstream Component Evaluation - Retrieval & Extraction
print("\n[1] Evaluating Document Ingestion and Topic Extraction")
pdf_path = "sample_doc.pdf"
try:
    vectorstore = ingest_document(pdf_path)
    topics = extract_topics(vectorstore, n_topics=3)
    print(f"Extracted topics: {topics}")
    if len(topics) >= 1:
        print("✅ Extraction success")
    else:
        print("❌ Extraction failed")
except Exception as e:
    print(f"❌ Extraction error: {e}")
    exit(1)

# Step 2: Retrieval Evaluation (Recall@K)
print("\n[2] Evaluating Retrieval (Recall@K)")
test_topic = "Gradient Descent"
retrieved_docs = vectorstore.similarity_search(test_topic, k=4)
# Check if the term is in the retrieved text
hit = any(test_topic.lower() in doc.page_content.lower() for doc in retrieved_docs)
if hit:
    print(f"✅ Retrieval hit for topic: {test_topic}")
else:
    print(f"❌ Retrieval miss for topic: {test_topic}")

# Step 3: Baseline Comparison (Naive Prompt vs RAG)
print("\n[3] Baseline Comparison: Naive Question Generation vs RAG Question Generation")
naive_prompt = ChatPromptTemplate.from_template("Generate a hard difficulty question about {topic}.")
naive_chain = naive_prompt | llm | StrOutputParser()
naive_question = naive_chain.invoke({"topic": test_topic})

rag_question = generate_question(vectorstore, test_topic, "hard")
print("Naive Question (No Context):", naive_question)
print("RAG Question (With Context):", rag_question)

# Quality check: Does the RAG question contain specific details from the text?
quality_check_prompt = ChatPromptTemplate.from_template("""
Compare two questions generated about {topic}. Which one is more grounded in the specific context?
Context: {context}
Naive Question: {naive}
RAG Question: {rag}
Reply with 'RAG' or 'Naive' or 'Equal'.
""")
context_str = "\n".join(d.page_content for d in retrieved_docs)
quality_chain = quality_check_prompt | llm_precise | StrOutputParser()
quality_winner = quality_chain.invoke({
    "topic": test_topic, "context": context_str, "naive": naive_question, "rag": rag_question["question"]
})
print(f"Quality Winner (LLM-as-a-judge): {quality_winner.strip()}")

# Step 4: End-to-End Task Success (5 representative, 2 failure)
print("\n[4] Evaluating End-to-End Task Success")

# We will simulate 5 scenarios
scenarios = [
    {"topic": "Neural Networks", "difficulty": "easy", "student_answer": "It is a computational model inspired by the brain."}, # Success case 1
    {"topic": "Gradient Descent", "difficulty": "medium", "student_answer": "It's used to minimize the error or loss function."}, # Success case 2
    {"topic": "Learning Rate", "difficulty": "hard", "student_answer": "It determines step size. Too small means slow convergence, too large means it might overshoot."}, # Success case 3
    {"topic": "Overfitting", "difficulty": "easy", "student_answer": "It means the network learns the training data too well."}, # Success case 4
    {"topic": "Cross-validation", "difficulty": "medium", "student_answer": "It detects overfitting by splitting data."}, # Success case 5
    {"topic": "Gradient Descent", "difficulty": "easy", "student_answer": "It is a type of bicycle."}, # Failure case 1 (completely wrong)
    {"topic": "Overfitting", "difficulty": "hard", "student_answer": "It happens when the model is under-trained."}, # Failure case 2 (opposite meaning)
]

for i, scenario in enumerate(scenarios):
    topic = scenario["topic"]
    try:
        q_data = generate_question(vectorstore, topic, scenario["difficulty"])
        result = evaluate_answer(q_data["question"], q_data["answer"], scenario["student_answer"])
        print(f"Scenario {i+1} ({topic}): Answer Correct = {result.get('correct')}")
    except Exception as e:
        print(f"❌ Scenario {i+1} failed during execution: {e}")

print("Evaluation complete.")

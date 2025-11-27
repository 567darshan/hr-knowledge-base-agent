# 💼 HR Knowledge Base Agent

An AI **Knowledge Base Agent** that answers employee questions about HR policies  
(leave, benefits, working hours, probation, etc.) using internal HR documents.

This project is built as part of an **AI Agent Internship Challenge**.

---

## 🔍 Problem

HR teams repeatedly get basic questions like:

- "How many paid leaves do I have?"
- "What are my working hours?"
- "What is the probation period?"
- "When do I get health insurance?"

Answering the same questions again and again wastes HR time and delays responses for employees.

---

## 🎯 Solution – HR Knowledge Base Agent

This agent:

1. Reads HR policy text from `data/hr_policies.txt`.
2. Splits the text into smaller chunks.
3. Converts each chunk into vector embeddings using OpenAI.
4. Stores embeddings in a Chroma vector database.
5. For every employee query:
   - Finds the most relevant chunks from the vector database.
   - Sends them + the question to an OpenAI chat model.
   - Generates a clear answer and shows the policy snippets used.

If the policy text does not clearly answer the question, the agent tells the user to contact the HR team.

---

## 🏗️ Architecture Overview

**1. User Interface (Streamlit)**  
Simple web UI where employees can type their HR questions.

**2. Document Layer**  
HR policies stored as plain text file: `data/hr_policies.txt`.

**3. Text Splitting**  
`RecursiveCharacterTextSplitter` splits long policy text into overlapping chunks so they fit in the model context and keep meaning.

**4. Embeddings + Vector Store**  
- `OpenAIEmbeddings` creates high-dimensional vectors for each chunk.  
- `Chroma` stores these vectors and performs semantic similarity search.

**5. Retrieval + LLM Reasoning**  
- For a user question, we retrieve top-k similar chunks using `similarity_search`.  
- We build a prompt that includes:
  - HR policy context
  - User question  
- `ChatOpenAI` generates a final answer **only from the given context**.

**6. Response Layer**  
Streamlit displays:
- The final answer
- The source snippets from the HR document used to answer

---

## 🛠️ Tech Stack

- **Language:** Python 3.11  
- **UI:** Streamlit  
- **AI Framework:** LangChain (core + community packages)  
- **Model Provider:** OpenAI (Chat + Embeddings)  
- **Vector Database:** ChromaDB  
- **Environment Management:** python-dotenv  

---

## 📁 Project Structure

```text
hr-knowledge-base-agent/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .env                 # NOT committed (contains OPENAI_API_KEY)
└── data/
    └── hr_policies.txt

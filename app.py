import os
import streamlit as st
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma

# ---------- 1. Load API key from .env ----------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not found. Please set it in .env.")
    st.stop()

# ---------- 2. Streamlit UI setup ----------
st.set_page_config(
    page_title="HR Knowledge Base Agent",
    page_icon="💼",
)

st.title("💼 HR Knowledge Base Agent")
st.write(
    "Ask questions about HR policies like leave, benefits, working hours, etc. "
    "This agent searches the HR documents and answers using an LLM."
)

# ---------- 3. Build vector store from HR document ----------
@st.cache_resource(show_spinner=True)
def build_vector_store():
    """Read hr_policies.txt, split, embed, and store in ChromaDB."""
    file_path = "data/hr_policies.txt"

    if not os.path.exists(file_path):
        raise FileNotFoundError("HR policies file not found at data/hr_policies.txt")

    # 3.1 Read text
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 3.2 Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )
    docs = splitter.create_documents([text])

    # 3.3 Create embeddings
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

    # 3.4 Store in Chroma vector DB
    vectorstore = Chroma.from_documents(
        docs,
        embedding=embeddings,
        collection_name="hr_policies",
    )
    return vectorstore


@st.cache_resource(show_spinner=True)
def get_llm():
    """Create the ChatOpenAI model."""
    llm = ChatOpenAI(
        model="gpt-4.1-mini",  # or any other chat model available to you
        temperature=0.2,
        api_key=OPENAI_API_KEY,
    )
    return llm


vectorstore = build_vector_store()
llm = get_llm()

# ---------- 4. Sidebar info ----------
with st.sidebar:
    st.header("ℹ️ About this Agent")
    st.markdown(
        """
        **Agent Type:** Knowledge Base Agent  
        **Domain:** HR / People Operations  

        **What it does:**
        - Reads HR policy documents from `data/hr_policies.txt`
        - Builds a vector store using OpenAI embeddings + Chroma
        - Finds the most relevant policy text for your question
        - Uses an LLM to answer in simple language
        """
    )

# ---------- 5. Main Q&A logic WITHOUT langchain.chains ----------
user_question = st.text_input(
    "Ask your HR question here:",
    placeholder="Example: How many paid leaves do I get in a year?",
)

if st.button("Ask") and user_question.strip():
    with st.spinner("Thinking..."):
        # 5.1 Retrieve top 3 relevant chunks from vector store
        docs = vectorstore.similarity_search(user_question, k=3)

        context = "\n\n".join(doc.page_content for doc in docs)

        # 5.2 Build prompt manually
        prompt = f"""
You are an AI HR assistant for a company.

Use ONLY the HR policy context below to answer the question.
If the answer is not clearly in the context, say that the employee
should contact the HR team for confirmation.

HR POLICY CONTEXT:
------------------
{context}

QUESTION:
{user_question}

ANSWER (be clear and concise):
"""

        # 5.3 Ask the LLM
        ai_message = llm.invoke(prompt)
        answer = ai_message.content

    st.subheader("Answer")
    st.write(answer)

    # Show sources
    if docs:
        st.subheader("Source Snippets from HR Policy")
        for i, doc in enumerate(docs, start=1):
            st.markdown(f"**Source {i}:**")
            st.code(doc.page_content[:400] + "...")

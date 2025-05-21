import os
import logging
from dotenv import load_dotenv
from typing import Optional
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# -------------------- Config & Logging Setup --------------------
load_dotenv()  # Load variables from .env file

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DB_FAISS_PATH = "vectorstore/db_faiss"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# -------------------- LLM Setup --------------------
def load_llm() -> ChatGroq:
    if not GROQ_API_KEY:
        logging.error("❌ GROQ_API_KEY is missing. Set it in environment or .env file.")
        raise EnvironmentError("GROQ_API_KEY is not set.")

    return ChatGroq(
        model_name="llama3-70b-8192",  # You can also try: "llama3-70b-8192"
        api_key=GROQ_API_KEY,
        temperature=0.5,
        max_tokens=100
    )

# -------------------- Prompt Template --------------------
def set_custom_prompt(template_str: str) -> PromptTemplate:
    return PromptTemplate(template=template_str, input_variables=["context", "question"])
CUSTOM_PROMPT_TEMPLATE = """<s>[INST]
You are a friendly, human-like AI assistant who responds clearly and briefly.

Use both the previous conversation (chat history) and the given context to answer the question in one short natural-sounding sentence.

Chat History:
{chat_history}

Context:
{context}

Question:
{question}
[/INST]"""


# -------------------- Vector DB Loader --------------------
def load_vectorstore(db_path: str, model_name: str) -> FAISS:
    embedding_model = HuggingFaceEmbeddings(model_name=model_name)
    try:
        db = FAISS.load_local(db_path, embedding_model, allow_dangerous_deserialization=True)
        return db
    except Exception as e:
        logging.error(f"❌ Failed to load FAISS DB: {e}")
        raise

# -------------------- QA Chain with Memory --------------------
def create_qa_chain(llm: ChatGroq, db: FAISS, prompt_template: str, memory) -> ConversationalRetrievalChain:
    prompt = set_custom_prompt(prompt_template)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": 1}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=False,
    )

# -------------------- Main Execution --------------------
def main():
    try:
        llm = load_llm()
        db = load_vectorstore(DB_FAISS_PATH, EMBEDDING_MODEL)
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        qa_chain = create_qa_chain(llm, db, CUSTOM_PROMPT_TEMPLATE, memory)

        while True:
            user_query = input("Enter your query (or type 'exit' to quit): ").strip()
            if user_query.lower() in {"exit", "quit"}:
                logging.info("✅ Exiting the assistant.")
                break

            response = qa_chain.invoke({"question": user_query, "chat_history": memory.chat_memory.messages})
            print(f"\n🔍 Result: {response['answer']}\n")

    except Exception as e:
        logging.error(f"❌ Unexpected error: {e}")

# -------------------- Entry Point --------------------
if __name__ == "__main__":
    main()

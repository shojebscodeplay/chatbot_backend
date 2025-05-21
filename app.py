# filename: app.py

from fastapi import FastAPI, Request
from pydantic import BaseModel
import os, logging
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()
app = FastAPI()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DB_FAISS_PATH = "vectorstore/db_faiss"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

llm = ChatGroq(model_name="llama3-70b-8192", api_key=GROQ_API_KEY, temperature=0.5, max_tokens=100)
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

prompt_template = """<s>[INST]
You are a friendly, human-like AI assistant who responds clearly and briefly.

Use both the previous conversation (chat history) and the given context to answer the question in one short natural-sounding sentence.

Chat History:
{chat_history}

Context:
{context}

Question:
{question}
[/INST]"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=db.as_retriever(search_kwargs={"k": 1}),
    memory=memory,
    combine_docs_chain_kwargs={"prompt": prompt},
    return_source_documents=False,
)

class Query(BaseModel):
    question: str

@app.post("/chat")
async def chat_endpoint(query: Query):
    response = qa_chain.invoke({"question": query.question, "chat_history": memory.chat_memory.messages})
    return {"answer": response['answer']}

import os
import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS

# Set Streamlit page configuration
st.set_page_config(page_title="Mistral Chatbot", page_icon="🤖")

# HuggingFace Setup
HF_TOKEN = os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
DB_FAISS_PATH = "vectorstore/db_faiss"

def load_llm(huggingface_repo_id):
    endpoint = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.7,
        max_new_tokens=100,
        huggingfacehub_api_token=HF_TOKEN
    )
    return ChatHuggingFace(llm=endpoint)

CUSTOM_PROMPT_TEMPLATE = """<s>[INST] 
You are a helpful AI assistant. Based on the following context and chat history, answer the question in one short sentence, not more than one sentence. 
If you don't know the answer, say "I don't know." . If he greets you, say "Hello, how can I help you?". Retain knowledge of the chat history and context to provide a more accurate answer.

Context: {context}

Chat History:
{chat_history}

Question: {question} [/INST]
"""

def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "chat_history", "question"]
    )

@st.cache_resource
def load_retriever():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    return db.as_retriever(search_kwargs={"k": 1})

@st.cache_resource
def get_conversational_chain():
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=load_llm(HUGGINGFACE_REPO_ID),
        retriever=load_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)},
        return_source_documents=False
    )

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("💬 Mistral-Powered QA with FAISS")
st.caption("To load models it may take some time at the beginning.")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input box
if prompt := st.chat_input("Ask your question..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = get_conversational_chain().invoke({"question": prompt})
            response = result["answer"]
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

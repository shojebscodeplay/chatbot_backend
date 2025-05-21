import os
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# Step 1: Setup LLM (Mistral with HuggingFace)
HF_TOKEN = os.environ.get("HF_TOKEN")

if HF_TOKEN is None:
    raise EnvironmentError("❌ HF_TOKEN is missing. Please set it as an environment variable.")
HUGGINGFACE_REPO_ID="mistralai/Mistral-7B-Instruct-v0.3"

def load_llm(huggingface_repo_id):
    endpoint = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.7,
        max_new_tokens=100,
        huggingfacehub_api_token=HF_TOKEN
    )
    chat_model = ChatHuggingFace(llm=endpoint)
    return chat_model

# Step 2: Connect LLM with FAISS and Create chain

CUSTOM_PROMPT_TEMPLATE = """<s>[INST] 
You are a helpful AI assistant. Based only on the context, answer the question in one short sentence, not more than one sentence. 
If you don't know the answer, say "I don't know."

Context: {context}

Question: {question} [/INST]
"""

def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

# Load Database
DB_FAISS_PATH="vectorstore/db_faiss"

embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

#Step 3: Create QA chain
qa_chain=RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k':1}),
    return_source_documents=False,
    chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

#Step 4: Now invoke with a single query
user_query=input("Write Query Here: ")
response=qa_chain.invoke({'query': user_query})
print("RESULT: ", response["result"])
# print("SOURCE DOCUMENTS: ", response["source_documents"])
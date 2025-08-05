# import os

# from langchain_huggingface import HuggingFaceEndpoint
# from langchain_core.prompts import PromptTemplate
# from langchain.chains import RetrievalQA
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS

# ## Uncomment the following files if you're not using pipenv as your virtual environment manager
# #from dotenv import load_dotenv, find_dotenv
# #load_dotenv(find_dotenv())


# # Step 1: Setup LLM (Mistral with HuggingFace)
# HF_TOKEN=os.environ.get("HF_TOKEN")
# HUGGINGFACE_REPO_ID="mistralai/Mistral-7B-Instruct-v0.3"

# def load_llm(huggingface_repo_id):
#     llm=HuggingFaceEndpoint(
#         repo_id=huggingface_repo_id,
#         temperature=0.5,
#         model_kwargs={"token":HF_TOKEN,
#                       "max_length":"512"}
#     )
#     return llm

# # Step 2: Connect LLM with FAISS and Create chain

# CUSTOM_PROMPT_TEMPLATE = """
# Use the pieces of information provided in the context to answer user's question.
# If you dont know the answer, just say that you dont know, dont try to make up an answer. 
# Dont provide anything out of the given context

# Context: {context}
# Question: {question}

# Start the answer directly. No small talk please.
# """

# def set_custom_prompt(custom_prompt_template):
#     prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
#     return prompt

# # Load Database
# DB_FAISS_PATH="vectorstore/db_faiss"
# embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# # Create QA chain
# qa_chain=RetrievalQA.from_chain_type(
#     llm=load_llm(HUGGINGFACE_REPO_ID),
#     chain_type="stuff",
#     retriever=db.as_retriever(search_kwargs={'k':3}),
#     return_source_documents=True,
#     chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
# )

# # Now invoke with a single query
# user_query=input("Write Query Here: ")
# response=qa_chain.invoke({'query': user_query})
# print("RESULT: ", response["result"])
# print("SOURCE DOCUMENTS: ", response["source_documents"])
# import os
# from typing import List
# from langchain_huggingface import HuggingFaceEndpoint
# from langchain_core.prompts import PromptTemplate
# from langchain.chains import RetrievalQA
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain.schema import Document


# # Environment variables
# HF_TOKEN = os.environ.get("HF_TOKEN")
# HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
# DB_FAISS_PATH = "vectorstore/db_faiss"

# if not HF_TOKEN:
#     raise EnvironmentError("HF_TOKEN environment variable not set.")


# def load_llm(repo_id: str, token: str) -> HuggingFaceEndpoint:
#     """Initialize the HuggingFaceEndpoint LLM."""
#     return HuggingFaceEndpoint(
#         repo_id=repo_id,
#         temperature=0.5,
#         model_kwargs={
#             "token": token,
#             "max_length": 512,
#         },
#     )


# def create_prompt_template() -> PromptTemplate:
#     """Create a prompt template to instruct the LLM."""
#     template = """
# Use the pieces of information provided in the context to answer user's question.
# If you don't know the answer, just say that you don't know. Don't try to make up an answer.
# Don't provide anything out of the given context.

# Context: {context}
# Question: {question}

# Start the answer directly. No small talk please.
# """
#     return PromptTemplate(template=template, input_variables=["context", "question"])


# def load_vectorstore(path: str, embedding_model: HuggingFaceEmbeddings) -> FAISS:
#     """Load FAISS vectorstore from the local path."""
#     return FAISS.load_local(path, embedding_model, allow_dangerous_deserialization=True)


# def display_source_documents(docs: List[Document], max_chars: int = 300) -> None:
#     """Print source documents with truncation."""
#     if not docs:
#         print("No source documents returned.")
#         return

#     print("\nSOURCE DOCUMENTS:")
#     for idx, doc in enumerate(docs, 1):
#         content = doc.page_content
#         truncated = content if len(content) <= max_chars else content[:max_chars] + "..."
#         source = doc.metadata.get("source", "Unknown source")
#         doc_type = doc.metadata.get("type", "Unknown type")  # New: show if it's ICD or PDF
#         print(f"\nDocument {idx} (Type: {doc_type}, Source: {source}):\n{truncated}")


# def main() -> None:
#     # Load embedding model and vectorstore
#     embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     vectorstore = load_vectorstore(DB_FAISS_PATH, embedding_model)

#     # Load LLM
#     llm = load_llm(HUGGINGFACE_REPO_ID, HF_TOKEN)

#     # Create prompt template
#     prompt_template = create_prompt_template()

#     # Create RetrievalQA chain
#     qa_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
#         return_source_documents=True,
#         chain_type_kwargs={"prompt": prompt_template},
#     )

#     # Prompt user for query
#     try:
#         user_query = input("Write your query here: ").strip()
#         if not user_query:
#             print("No query entered. Exiting.")
#             return
#     except KeyboardInterrupt:
#         print("\nUser cancelled input. Exiting.")
#         return

#     # Run QA chain
#     try:
#         response = qa_chain.invoke({"query": user_query})
#     except Exception as e:
#         print(f"Error running query: {e}")
#         return

#     # Display answer
#     answer = response.get("result", "No result found.")
#     print(f"\nRESULT:\n{answer}")

#     # Display source documents with type info
#     source_docs = response.get("source_documents", [])
#     display_source_documents(source_docs)


# if __name__ == "__main__":
#     main()
# """
# connect_memory_with_llm.py

# Loads a HuggingFace-hosted LLM and connects it with the FAISS vector store
# to create a RetrievalQA chain for medical Q&A.
# """

# import os
# from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
# from langchain.chains import RetrievalQA
# from langchain_core.prompts import PromptTemplate
# from langchain_community.vectorstores import FAISS

# HF_TOKEN = os.environ.get("HF_TOKEN")
# HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
# DB_FAISS_PATH = "vectorstore/db_faiss"

# def load_llm(repo_id: str, token: str) -> HuggingFaceEndpoint:
#     """Loads the LLM from HuggingFace."""
#     return HuggingFaceEndpoint(
#         repo_id=repo_id,
#         temperature=0.5,
#         model_kwargs={"token": token, "max_length": 512},
#     )

# def create_prompt_template() -> PromptTemplate:
#     """Returns a custom prompt template for focused QA."""
#     template = """
# Use the pieces of information provided in the context to answer the user's question.
# If you don't know the answer, say you don't know. Don't make up an answer.
# Only use the context provided.

# Context: {context}
# Question: {question}

# Start your answer directly. No small talk.
# """
#     return PromptTemplate(template=template, input_variables=["context", "question"])

# def get_qa_chain() -> RetrievalQA:
#     """Creates and returns the RetrievalQA chain."""
#     if not HF_TOKEN:
#         raise EnvironmentError("HF_TOKEN environment variable not set.")

#     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     vectorstore = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
#     llm = load_llm(HUGGINGFACE_REPO_ID, HF_TOKEN)
#     prompt = create_prompt_template()

#     return RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
#         return_source_documents=True,
#         chain_type_kwargs={"prompt": prompt},
#     )

import os
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from create_memory_for_llm import get_embeddings  # import from other module

DB_FAISS_PATH = "vectorstore/db_faiss"
CUSTOM_PROMPT_TEMPLATE = """
Use the context below to answer the user's medical question.
Only answer from the context. If unknown, say "I don't know".

Context: {context}
Question: {question}
Answer:
"""

def load_vectorstore():
    if os.path.exists(DB_FAISS_PATH):
        embeddings = get_embeddings()
        return FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    return None

def save_vectorstore(vectorstore):
    os.makedirs(os.path.dirname(DB_FAISS_PATH), exist_ok=True)
    vectorstore.save_local(DB_FAISS_PATH)

def update_vectorstore_with_docs(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = splitter.split_documents(docs)
    embeddings = get_embeddings()
    vectorstore = load_vectorstore()
    if vectorstore:
        vectorstore.add_documents(texts)
    else:
        vectorstore = FAISS.from_documents(texts, embeddings)
    save_vectorstore(vectorstore)
    return vectorstore

def get_llm(model_choice=None):
    # Model choice param is optional; always return Groq LLaMA 4
    return ChatGroq(
        model_name="meta-llama/llama-4-maverick-17b-128e-instruct",
        temperature=0.0,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

def create_qa_chain(llm, vectorstore):
    from langchain_core.prompts import PromptTemplate
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE, input_variables=["context", "question"])}
    )

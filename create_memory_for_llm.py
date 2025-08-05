
# import os
# import glob
# import pandas as pd
# from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain.schema import Document  # for manual Document creation

# DATA_PATH = "data"
# ICD_FOLDER_PATH = os.path.join(DATA_PATH, "icd")
# DB_FAISS_PATH = "vectorstore/db_faiss"

# def load_pdf_files(data_path):
#     print(f"Loading PDFs from {data_path} ...")
#     loader = DirectoryLoader(
#         data_path,
#         glob="*.pdf",
#         loader_cls=PyPDFLoader
#     )
#     documents = loader.load()
#     print(f"Loaded {len(documents)} PDF pages.")
#     return documents

# def load_icd10_documents(folder_path):
#     print(f"Loading ICD-10 CSV files from {folder_path} ...")
#     csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
#     combined_docs = []
#     for file in csv_files:
#         try:
#             df = pd.read_csv(file)
#             if {"code", "disease"}.issubset(df.columns):
#                 for _, row in df.iterrows():
#                     text = f"ICD-10 Code: {row['code']}\nDisease: {row['disease']}"
#                     if "category" in df.columns:
#                         text += f"\nCategory: {row['category']}"
#                     doc = Document(
#                         page_content=text,
#                         metadata={"source": os.path.basename(file), "code": row["code"]}
#                     )
#                     combined_docs.append(doc)
#             else:
#                 print(f"⚠️ Skipped ICD file without required columns: {file}")
#         except Exception as e:
#             print(f"❌ Error reading {file}: {e}")
#     print(f"Loaded {len(combined_docs)} ICD-10 entries as documents.")
#     return combined_docs

# def create_pdf_chunks(pdf_documents):
#     print(f"Chunking {len(pdf_documents)} PDF documents...")
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#     chunks = text_splitter.split_documents(pdf_documents)
#     print(f"Created {len(chunks)} chunks from PDFs.")
#     return chunks

# def get_embedding_model():
#     print("Loading embedding model...")
#     return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# def main():
#     # Load and chunk PDFs
#     pdf_docs = load_pdf_files(DATA_PATH)
#     pdf_chunks = create_pdf_chunks(pdf_docs)

#     # Load ICD-10 docs (small, so no chunking)
#     icd_docs = load_icd10_documents(ICD_FOLDER_PATH)

#     # Combine all documents
#     all_docs = pdf_chunks + icd_docs
#     print(f"Total documents to index: {len(all_docs)}")

#     # Create embeddings and build FAISS vector store
#     embedding_model = get_embedding_model()
#     vector_db = FAISS.from_documents(all_docs, embedding_model)

#     # Save the vectorstore locally
#     os.makedirs(os.path.dirname(DB_FAISS_PATH), exist_ok=True)
#     vector_db.save_local(DB_FAISS_PATH)
#     print(f"Vector store saved to {DB_FAISS_PATH}")

# if __name__ == "__main__":
#     main()
import os
import tempfile 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from transformers import pipeline
import streamlit as st
MAX_FILE_SIZE_MB = 200

@st.cache_data(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def process_file(uploaded_file):
    suffix = os.path.splitext(uploaded_file.name)[-1].lower()
    if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise ValueError(f"File too large (> {MAX_FILE_SIZE_MB}MB).")
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    try:
        if suffix == ".pdf":
            loader = PyPDFLoader(tmp_path)
        elif suffix == ".txt":
            loader = TextLoader(tmp_path)
        elif suffix == ".docx":
            loader = UnstructuredWordDocumentLoader(tmp_path)
        else:
            raise ValueError("Unsupported file type.")
        return loader.load()
    finally:
        os.remove(tmp_path)

@st.cache_resource(show_spinner=False)
def load_zero_shot_classifier():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def classify_intent_zero_shot(text: str, classifier, labels):
    result = classifier(text, labels)
    top_label = result["labels"][0]
    top_score = result["scores"][0]
    return top_label, top_score

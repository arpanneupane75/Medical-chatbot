# import os
# import streamlit as st

# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.chains import RetrievalQA

# from langchain_community.vectorstores import FAISS
# from langchain_core.prompts import PromptTemplate
# from langchain_huggingface import HuggingFaceEndpoint
# from langchain_groq import ChatGroq



# ## Uncomment the following files if you're not using pipenv as your virtual environment manager
# from dotenv import load_dotenv, find_dotenv
# load_dotenv(find_dotenv())

# from dotenv import load_dotenv
# import os

# load_dotenv()  # Load from .env

# groq_api_key = os.getenv("GROQ_API_KEY")
# if not groq_api_key:
#     raise ValueError("GROQ_API_KEY is not set in the environment variables.")
# DB_FAISS_PATH="vectorstore/db_faiss"
# @st.cache_resource
# def get_vectorstore():
#     embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
#     db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
#     return db


# def set_custom_prompt(custom_prompt_template):
#     prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
#     return prompt


# def load_llm(huggingface_repo_id, HF_TOKEN):
#     llm=HuggingFaceEndpoint(
#         repo_id=huggingface_repo_id,
#         temperature=0.5,
#         model_kwargs={"token":HF_TOKEN,
#                       "max_length":"512"}
#     )
#     return llm


# def main():
#     st.title("Ask Chatbot!")

#     if 'messages' not in st.session_state:
#         st.session_state.messages = []

#     for message in st.session_state.messages:
#         st.chat_message(message['role']).markdown(message['content'])

#     prompt=st.chat_input("Pass your prompt here")

#     if prompt:
#         st.chat_message('user').markdown(prompt)
#         st.session_state.messages.append({'role':'user', 'content': prompt})

#         CUSTOM_PROMPT_TEMPLATE = """
#                 Use the pieces of information provided in the context to answer user's question.
#                 If you dont know the answer, just say that you dont know, dont try to make up an answer. 
#                 Dont provide anything out of the given context

#                 Context: {context}
#                 Question: {question}

#                 Start the answer directly. No small talk please.
#                 """
        
#         #HUGGINGFACE_REPO_ID="mistralai/Mistral-7B-Instruct-v0.3" # PAID
#         #HF_TOKEN=os.environ.get("HF_TOKEN")  

#         #TODO: Create a Groq API key and add it to .env file
        
#         try: 
#             vectorstore=get_vectorstore()
#             if vectorstore is None:
#                 st.error("Failed to load the vector store")

#             qa_chain = RetrievalQA.from_chain_type(
#                 llm=ChatGroq(
#                     model_name="meta-llama/llama-4-maverick-17b-128e-instruct",  # free, fast Groq-hosted model
#                     temperature=0.0,
#                     groq_api_key=os.environ["GROQ_API_KEY"],
#                 ),
#                 chain_type="stuff",
#                 retriever=vectorstore.as_retriever(search_kwargs={'k':3}),
#                 return_source_documents=True,
#                 chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
#             )

#             response=qa_chain.invoke({'query':prompt})

#             result=response["result"]
#             source_documents=response["source_documents"]
#             result_to_show=result+"\nSource Docs:\n"+str(source_documents)
#             #response="Hi, I am MediBot!"
#             st.chat_message('assistant').markdown(result_to_show)
#             st.session_state.messages.append({'role':'assistant', 'content': result_to_show})

#         except Exception as e:
#             st.error(f"Error: {str(e)}")

# if __name__ == "__main__":
#     main()



# Revised MediBot Streamlit Application (UI and Functionality Improved)

# import os
# import tempfile
# import streamlit as st
# from datetime import datetime
# import pyttsx3
# from dotenv import load_dotenv
# from langchain.chains import RetrievalQA
# from langchain_core.prompts import PromptTemplate
# from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
# from langchain_groq import ChatGroq
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import (
#     PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
# )
# from transformers import pipeline
# import wikipedia

# # Load environment variables
# load_dotenv()

# # Constants
# DB_FAISS_PATH = "vectorstore/db_faiss"
# MAX_FILE_SIZE_MB = 200
# MIN_INTENT_CONFIDENCE = 0.4

# # Initialize TTS engine
# tts_engine = pyttsx3.init()

# # Streamlit page config
# st.set_page_config(page_title="MediBot 🤖", page_icon="💊", layout="wide")
# st.sidebar.title("🧠 MediBot Settings")

# # Load zero-shot classifiers
# @st.cache_resource(show_spinner=False)
# def load_zero_shot_classifiers():
#     return (
#         pipeline("zero-shot-classification", model="facebook/bart-large-mnli"),
#         pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli")
#     )

# classifier1, classifier2 = load_zero_shot_classifiers()

# MEDICAL_INTENT_LABELS = [
#     "Symptom Inquiry",
#     "Disease Information",
#     "Treatment Inquiry",
#     "Medication Information",
#     "General Medical Question",
#     "Non-Medical"
# ]

# @st.cache_data(show_spinner=False)
# def get_embeddings():
#     return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# def classify_intent_zero_shot(text: str):
#     results = [clf(text, MEDICAL_INTENT_LABELS) for clf in (classifier1, classifier2)]
#     scores = {}
#     for result in results:
#         for label, score in zip(result["labels"], result["scores"]):
#             scores[label] = scores.get(label, 0) + score
#     for label in scores:
#         scores[label] /= len(results)
#     top_label = max(scores, key=scores.get)
#     return top_label, scores[top_label]

# def speak(text: str):
#     try:
#         tts_engine.say(text[:500])
#         tts_engine.runAndWait()
#     except Exception:
#         pass

# CUSTOM_PROMPT_TEMPLATE = """
# Use the context below to answer the user's medical question.
# Only answer from the context. If unknown, say "I don't know".

# Context: {context}
# Question: {question}
# Answer:
# """

# @st.cache_resource(show_spinner=False)
# def load_vectorstore():
#     if os.path.exists(DB_FAISS_PATH):
#         embeddings = get_embeddings()
#         return FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
#     return None

# def save_vectorstore(vectorstore):
#     os.makedirs(os.path.dirname(DB_FAISS_PATH), exist_ok=True)
#     vectorstore.save_local(DB_FAISS_PATH)

# def process_file(uploaded_file):
#     suffix = os.path.splitext(uploaded_file.name)[-1].lower()
#     if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
#         st.sidebar.error(f"File too large (> {MAX_FILE_SIZE_MB}MB).")
#         return None
#     with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
#         tmp.write(uploaded_file.read())
#         tmp_path = tmp.name
#     try:
#         if suffix == ".pdf":
#             loader = PyPDFLoader(tmp_path)
#         elif suffix == ".txt":
#             loader = TextLoader(tmp_path)
#         elif suffix == ".docx":
#             loader = UnstructuredWordDocumentLoader(tmp_path)
#         else:
#             st.sidebar.error("Unsupported file type.")
#             return None
#         with st.spinner("Loading document..."):
#             return loader.load()
#     except Exception as e:
#         st.sidebar.error(f"Error loading file: {e}")
#         return None
#     finally:
#         os.remove(tmp_path)

# def update_vectorstore_with_docs(docs):
#     splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#     texts = splitter.split_documents(docs)
#     embeddings = get_embeddings()
#     vectorstore = load_vectorstore()
#     if vectorstore:
#         vectorstore.add_documents(texts)
#     else:
#         vectorstore = FAISS.from_documents(texts, embeddings)
#     save_vectorstore(vectorstore)
#     st.sidebar.success("✅ Documents processed and vectorstore updated.")
#     return vectorstore

# def query_wikipedia(question: str) -> str:
#     try:
#         return wikipedia.summary(question, sentences=3)
#     except Exception:
#         return "Sorry, I couldn't find relevant information online."

# def get_llm(model_choice):
#     if model_choice == "Groq - LLaMA 4":
#         return ChatGroq(
#             model_name="meta-llama/llama-4-maverick-17b-128e-instruct",
#             temperature=0.0,
#             groq_api_key=os.getenv("GROQ_API_KEY")
#         )
#     else:
#         return HuggingFaceEndpoint(
#             repo_id="tiiuae/falcon-7b-instruct",
#             temperature=0.5,
#             provider="auto",
            
            
#         )

# def main():
#     if "messages" not in st.session_state:
#         st.session_state.messages = []

#     model_choice = st.sidebar.selectbox("Choose your language model:", ("Groq - LLaMA 4", "HuggingFace - Mistral 7B"))

#     st.sidebar.subheader("📄 Upload Medical Documents")
#     uploaded_files = st.sidebar.file_uploader("Upload PDF, TXT, or DOCX files", type=["pdf", "txt", "docx"], accept_multiple_files=True)

#     vectorstore = load_vectorstore()
#     if uploaded_files:
#         all_docs = []
#         for f in uploaded_files:
#             docs = process_file(f)
#             if docs:
#                 all_docs.extend(docs)
#         if all_docs:
#             vectorstore = update_vectorstore_with_docs(all_docs)

#     if st.sidebar.button("🛉 Clear Chat"):
#         st.session_state.messages = []
#         st.rerun()

#     show_sources = st.sidebar.checkbox("📚 Show Source Documents")
#     enable_tts = st.sidebar.checkbox("🔊 Enable Text-to-Speech", value=True)

#     for msg in st.session_state.messages:
#         with st.chat_message(msg["role"]):
#             st.markdown(msg["content"])

#     prompt = st.chat_input("Ask a medical question...")

#     if prompt:
#         st.chat_message("user").markdown(prompt)
#         st.session_state.messages.append({"role": "user", "content": prompt})

#         intent, confidence = classify_intent_zero_shot(prompt)
#         if confidence < MIN_INTENT_CONFIDENCE or intent == "Non-Medical":
#             st.warning("⚠️ I can't confidently determine the medical relevance. Please rephrase.")
#             return

#         st.info(f"Detected Intent: **{intent}** (Confidence: {confidence:.2f})")

#         llm = get_llm(model_choice)

#         if vectorstore:
#             qa_chain = RetrievalQA.from_chain_type(
#                 llm=llm,
#                 chain_type="stuff",
#                 retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
#                 return_source_documents=True,
#                 chain_type_kwargs={"prompt": PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE, input_variables=["context", "question"])}
#             )
#             with st.spinner("Generating answer from your documents..."):
#                 response = qa_chain.invoke({"query": prompt})
#             answer = response.get("result", "Sorry, no answer found in documents.")
#             source_docs = response.get("source_documents", [])
#         else:
#             with st.spinner("Searching Wikipedia (fallback)..."):
#                 wiki_summary = query_wikipedia(prompt)
#                 formatted_prompt = PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE, input_variables=["context", "question"]).format(
#                     context=wiki_summary,
#                     question=prompt
#                 )
#                 llm_response = llm.invoke(formatted_prompt)
#                 answer = llm_response if isinstance(llm_response, str) else str(llm_response)

#                 source_docs = []

#         with st.chat_message("assistant"):
#             st.markdown(f"✅ **Answer:** {answer}")
#             if enable_tts:
#                 speak(answer)
#             if show_sources and source_docs:
#                 with st.expander("📚 Source Documents"):
#                     for i, doc in enumerate(source_docs):
#                         preview = doc.page_content[:1000].replace("\n", " ").strip()
#                         st.markdown(f"**Doc {i+1}:**\n```\n{preview}\n```")

#         st.session_state.messages.append({"role": "assistant", "content": answer})

# if __name__ == "__main__":
#     main()


# import os
# import tempfile
# import streamlit as st
# from datetime import datetime
# import pyttsx3
# from dotenv import load_dotenv
# from langchain.chains import RetrievalQA
# from langchain_core.prompts import PromptTemplate
# from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
# from langchain_groq import ChatGroq
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import (
#     PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
# )
# from transformers import pipeline
# import wikipedia

# # Load environment variables
# load_dotenv()

# # Constants
# DB_FAISS_PATH = "vectorstore/db_faiss"
# MAX_FILE_SIZE_MB = 200
# MIN_INTENT_CONFIDENCE = 0.4

# # Initialize TTS engine
# tts_engine = pyttsx3.init()

# # Streamlit page config
# st.set_page_config(page_title="MediBot 🤖", page_icon="💊", layout="wide")
# st.sidebar.title("🧠 MediBot Settings")

# # Load zero-shot classifiers
# @st.cache_resource(show_spinner=False)
# def load_zero_shot_classifiers():
#     return (
#         pipeline("zero-shot-classification", model="facebook/bart-large-mnli"),
#         pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli")
#     )

# classifier1, classifier2 = load_zero_shot_classifiers()

# MEDICAL_INTENT_LABELS = [
#     "Symptom Inquiry",
#     "Disease Information",
#     "Treatment Inquiry",
#     "Medication Information",
#     "General Medical Question",
#     "Non-Medical"
# ]

# @st.cache_data(show_spinner=False)
# def get_embeddings():
#     return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# def classify_intent_zero_shot(text: str):
#     results = [clf(text, MEDICAL_INTENT_LABELS) for clf in (classifier1, classifier2)]
#     scores = {}
#     for result in results:
#         for label, score in zip(result["labels"], result["scores"]):
#             scores[label] = scores.get(label, 0) + score
#     for label in scores:
#         scores[label] /= len(results)
#     top_label = max(scores, key=scores.get)
#     return top_label, scores[top_label]

# def speak(text: str):
#     try:
#         tts_engine.say(text[:500])
#         tts_engine.runAndWait()
#     except Exception:
#         pass

# CUSTOM_PROMPT_TEMPLATE = """
# Use the context below to answer the user's medical question.
# Only answer from the context. If unknown, say "I don't know".

# Context: {context}
# Question: {question}
# Answer:
# """

# @st.cache_resource(show_spinner=False)
# def load_vectorstore():
#     if os.path.exists(DB_FAISS_PATH):
#         embeddings = get_embeddings()
#         return FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
#     return None

# def save_vectorstore(vectorstore):
#     os.makedirs(os.path.dirname(DB_FAISS_PATH), exist_ok=True)
#     vectorstore.save_local(DB_FAISS_PATH)

# def process_file(uploaded_file):
#     suffix = os.path.splitext(uploaded_file.name)[-1].lower()
#     if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
#         st.sidebar.error(f"File too large (> {MAX_FILE_SIZE_MB}MB).")
#         return None
#     with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
#         tmp.write(uploaded_file.read())
#         tmp_path = tmp.name
#     try:
#         if suffix == ".pdf":
#             loader = PyPDFLoader(tmp_path)
#         elif suffix == ".txt":
#             loader = TextLoader(tmp_path)
#         elif suffix == ".docx":
#             loader = UnstructuredWordDocumentLoader(tmp_path)
#         else:
#             st.sidebar.error("Unsupported file type.")
#             return None
#         with st.spinner("Loading document..."):
#             return loader.load()
#     except Exception as e:
#         st.sidebar.error(f"Error loading file: {e}")
#         return None
#     finally:
#         os.remove(tmp_path)

# def update_vectorstore_with_docs(docs):
#     splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#     texts = splitter.split_documents(docs)
#     embeddings = get_embeddings()
#     vectorstore = load_vectorstore()
#     if vectorstore:
#         vectorstore.add_documents(texts)
#     else:
#         vectorstore = FAISS.from_documents(texts, embeddings)
#     save_vectorstore(vectorstore)
#     st.sidebar.success("✅ Documents processed and vectorstore updated.")
#     return vectorstore

# def query_wikipedia(question: str) -> str:
#     try:
#         return wikipedia.summary(question, sentences=3)
#     except Exception:
#         return "Sorry, I couldn't find relevant information online."

# def get_llm(model_choice):
#     if model_choice == "Groq - LLaMA 4":
#         return ChatGroq(
#             model_name="meta-llama/llama-4-maverick-17b-128e-instruct",
#             temperature=0.0,
#             groq_api_key=os.getenv("GROQ_API_KEY")
#         )
#     # No else or other model here because Mistral is removed

# def main():
#     if "messages" not in st.session_state:
#         st.session_state.messages = []

#     # Only Groq model option left
#     model_choice = st.sidebar.selectbox("Choose your language model:", ("Groq - LLaMA 4",))

#     st.sidebar.subheader("📄 Upload Medical Documents")
#     uploaded_files = st.sidebar.file_uploader("Upload PDF, TXT, or DOCX files", type=["pdf", "txt", "docx"], accept_multiple_files=True)

#     vectorstore = load_vectorstore()
#     if uploaded_files:
#         all_docs = []
#         for f in uploaded_files:
#             docs = process_file(f)
#             if docs:
#                 all_docs.extend(docs)
#         if all_docs:
#             vectorstore = update_vectorstore_with_docs(all_docs)

#     if st.sidebar.button("🛉 Clear Chat"):
#         st.session_state.messages = []
#         st.rerun()

#     show_sources = st.sidebar.checkbox("📚 Show Source Documents")
#     enable_tts = st.sidebar.checkbox("🔊 Enable Text-to-Speech", value=True)

#     for msg in st.session_state.messages:
#         with st.chat_message(msg["role"]):
#             st.markdown(msg["content"])

#     prompt = st.chat_input("Ask a medical question...")

#     if prompt:
#         st.chat_message("user").markdown(prompt)
#         st.session_state.messages.append({"role": "user", "content": prompt})

#         intent, confidence = classify_intent_zero_shot(prompt)
#         if confidence < MIN_INTENT_CONFIDENCE or intent == "Non-Medical":
#             st.warning("⚠️ I can't confidently determine the medical relevance. Please rephrase.")
#             return

#         st.info(f"Detected Intent: **{intent}** (Confidence: {confidence:.2f})")

#         llm = get_llm(model_choice)

#         if vectorstore:
#             qa_chain = RetrievalQA.from_chain_type(
#                 llm=llm,
#                 chain_type="stuff",
#                 retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
#                 return_source_documents=True,
#                 chain_type_kwargs={"prompt": PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE, input_variables=["context", "question"])}
#             )
#             with st.spinner("Generating answer from your documents..."):
#                 response = qa_chain.invoke({"query": prompt})
#             answer = response.get("result", "Sorry, no answer found in documents.")
#             source_docs = response.get("source_documents", [])
#         else:
#             with st.spinner("Searching Wikipedia (fallback)..."):
#                 wiki_summary = query_wikipedia(prompt)
#                 formatted_prompt = PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE, input_variables=["context", "question"]).format(
#                     context=wiki_summary,
#                     question=prompt
#                 )
#                 llm_response = llm.invoke(formatted_prompt)
#                 answer = llm_response if isinstance(llm_response, str) else str(llm_response)

#                 source_docs = []

#         with st.chat_message("assistant"):
#             st.markdown(f"✅ **Answer:** {answer}")
#             if enable_tts:
#                 speak(answer)
#             if show_sources and source_docs:
#                 with st.expander("📚 Source Documents"):
#                     for i, doc in enumerate(source_docs):
#                         preview = doc.page_content[:1000].replace("\n", " ").strip()
#                         st.markdown(f"**Doc {i+1}:**\n```\n{preview}\n```")

#         st.session_state.messages.append({"role": "assistant", "content": answer})

# if __name__ == "__main__":
#     main()
import os
import streamlit as st
import pyttsx3
import wikipedia
from dotenv import load_dotenv
from create_memory_for_llm import (
    process_file,
    load_zero_shot_classifier,
    classify_intent_zero_shot,
)
from connect_memory_with_llm import (
    load_vectorstore,
    update_vectorstore_with_docs,
    get_llm,
    create_qa_chain,
    save_vectorstore,
)

# Load environment variables
load_dotenv()

# Constants
MIN_INTENT_CONFIDENCE = 0.4

# Initialize TTS engine

try:
    import pyttsx3
    tts_engine = pyttsx3.init()
except ImportError:
    tts_engine = None


# Streamlit page config
st.set_page_config(page_title="MediBot 🤖", page_icon="💊", layout="wide")
st.sidebar.title("🧠 MediBot Settings")

MEDICAL_INTENT_LABELS = [
    "Symptom Inquiry",
    "Disease Information",
    "Treatment Inquiry",
    "Medication Information",
    "General Medical Question",
    "Non-Medical"
]

def speak(text: str):
    try:
        tts_engine.say(text[:500])
        tts_engine.runAndWait()
    except Exception:
        pass

def query_wikipedia(question: str) -> str:
    try:
        return wikipedia.summary(question, sentences=3)
    except Exception:
        return "Sorry, I couldn't find relevant information online."

def main():
    if "messages" not in st.session_state:
        st.session_state.messages = []

    model_choice = st.sidebar.selectbox("Choose your language model:", ("Groq - LLaMA 4",))

    st.sidebar.subheader("📄 Upload Medical Documents")
    uploaded_files = st.sidebar.file_uploader(
        "Upload PDF, TXT, or DOCX files", type=["pdf", "txt", "docx"], accept_multiple_files=True
    )

    vectorstore = load_vectorstore()
    if uploaded_files:
        all_docs = []
        for f in uploaded_files:
            try:
                docs = process_file(f)
                if docs:
                    all_docs.extend(docs)
            except Exception as e:
                st.sidebar.error(str(e))
        if all_docs:
            vectorstore = update_vectorstore_with_docs(all_docs)

    if st.sidebar.button("🛉 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    show_sources = st.sidebar.checkbox("📚 Show Source Documents")
    enable_tts = st.sidebar.checkbox("🔊 Enable Text-to-Speech", value=True)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Ask a medical question...")

    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        classifier = load_zero_shot_classifier()
        intent, confidence = classify_intent_zero_shot(prompt, classifier, MEDICAL_INTENT_LABELS)
        if confidence < MIN_INTENT_CONFIDENCE or intent == "Non-Medical":
            st.warning("⚠️ I can't confidently determine the medical relevance. Please rephrase.")
            return

        st.info(f"Detected Intent: **{intent}** (Confidence: {confidence:.2f})")

        llm = get_llm(model_choice)

        if vectorstore:
            qa_chain = create_qa_chain(llm, vectorstore)
            with st.spinner("Generating answer from your documents..."):
                response = qa_chain.invoke({"query": prompt})
            answer = response.get("result", "Sorry, no answer found in documents.")
            source_docs = response.get("source_documents", [])
        else:
            with st.spinner("Searching Wikipedia (fallback)..."):
                wiki_summary = query_wikipedia(prompt)
                # Create prompt manually since no vectorstore/QA chain here
                from langchain_core.prompts import PromptTemplate
                CUSTOM_PROMPT_TEMPLATE = """
Use the context below to answer the user's medical question.
Only answer from the context. If unknown, say "I don't know".

Context: {context}
Question: {question}
Answer:
"""
                fallback_prompt = PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE, input_variables=["context", "question"]).format(
                    context=wiki_summary,
                    question=prompt
                )
                llm_response = llm.invoke(fallback_prompt)
                answer = llm_response if isinstance(llm_response, str) else str(llm_response)
                source_docs = []

        with st.chat_message("assistant"):
            st.markdown(f"✅ **Answer:** {answer}")
            if enable_tts:
                speak(answer)
            if show_sources and source_docs:
                with st.expander("📚 Source Documents"):
                    for i, doc in enumerate(source_docs):
                        preview = doc.page_content[:1000].replace("\n", " ").strip()
                        st.markdown(f"**Doc {i+1}:**\n```\n{preview}\n```")

        st.session_state.messages.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    main()

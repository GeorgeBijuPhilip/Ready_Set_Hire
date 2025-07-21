import streamlit as st
import os
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq
from dotenv import load_dotenv
from gtts import gTTS
import tempfile

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Groq client
groq_client = Groq(api_key=GROQ_API_KEY)

# Utility to clean markdown-like artifacts
def clean_markdown(text):
    return re.sub(r'[*_#`]+', '', text)

# Resume summarizer
def summarize_resume(resume_text):
    prompt = f"""Create a concise summary of this resume highlighting:
    1. Professional title/role
    2. Years of experience
    3. Core skills/competencies
    4. Education background
    5. Notable achievements

    Resume:
    {resume_text[:3000]}... [truncated]"""
    response = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama3-70b-8192",
        temperature=0.3,
    )
    return clean_markdown(response.choices[0].message.content)

# ATS scoring
def calculate_ats_score(resume_text):
    prompt = f"""Analyze this resume and calculate an ATS score (0-100) considering:
    1. Keyword optimization (20 pts)
    2. Section organization (20 pts)
    3. Experience quality (20 pts)
    4. Education completeness (20 pts)
    5. Readability (20 pts)

    Return ONLY the numerical score and nothing else.

    Resume:
    {resume_text[:3000]}... [truncated]"""
    response = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama3-70b-8192",
        temperature=0,
    )
    try:
        return int(response.choices[0].message.content.strip())
    except:
        return 50

# Resume parsing and vector store
def process_resume(uploaded_path):
    loader = PyPDFLoader(uploaded_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    full_text = "\n".join([doc.page_content for doc in chunks])
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    FAISS.from_documents(chunks, embeddings).save_local("resume_index")
    return summarize_resume(full_text), calculate_ats_score(full_text)

# Placeholder for transcription
def transcribe_audio(file_path):
    return "⚠️ Transcription not available in Streamlit Cloud. Run locally for full features."

# ---------- Streamlit UI ----------
st.set_page_config(page_title="🚀 Ready Set Hire", layout="centered")
st.title("🚀 Ready Set Hire")

tab1, tab2 = st.tabs(["📄 Resume Analysis", "🎤 Mock Interview"])

with tab1:
    st.subheader("📄 Upload Resume (PDF)")
    resume_file = st.file_uploader("Choose a PDF", type="pdf")

    if resume_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(resume_file.read())
            tmp_path = tmp.name

        summary, score = process_resume(tmp_path)
        st.success("✅ Resume processed!")
        st.text_area("📝 Resume Summary", summary, height=200)
        st.metric("📊 ATS Score", f"{score}/100")

with tab2:
    st.subheader("🎤 Record Audio")
    audio_file = st.file_uploader("Upload a WAV/MP3 file for transcription", type=["wav", "mp3"])

    if audio_file:
        st.audio(audio_file)
        st.info(transcribe_audio("dummy_path"))

import gradio as gr
import os
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from elevenlabs.client import ElevenLabs
from gtts import gTTS
import tempfile

# Load environment variables from .env
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# Check API keys
if not GROQ_API_KEY or not ELEVENLABS_API_KEY:
    raise EnvironmentError("Missing API keys. Please create a .env file with GROQ_API_KEY and ELEVENLABS_API_KEY.")

# Initialize clients
groq_client = Groq(api_key=GROQ_API_KEY)
elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
whisper_model = WhisperModel("small", device="cpu", compute_type="int8")

def clean_markdown(text):
    return re.sub(r'[*_#`]+', '', text)

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

def process_resume(file):
    try:
        loader = PyPDFLoader(file.name)
        docs = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        ).split_documents(loader.load())
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        FAISS.from_documents(docs, embeddings).save_local("resume_index")
        full_text = "\n".join([doc.page_content for doc in docs])
        gr.Info("✅ Resume processed successfully!")
        return summarize_resume(full_text), f"ATS Score: {calculate_ats_score(full_text)}/100"
    except Exception as e:
        gr.Warning(f"❌ Error: {e}")
        return f"Error: {e}", "ATS Score: N/A"

def transcribe_audio(audio_path):
    if not audio_path:
        return "No audio recorded"
    segments, _ = whisper_model.transcribe(audio_path)
    return " ".join([segment.text for segment in segments])

def generate_question(resume_text):
    prompt = f"""Generate one general interview question focusing on:
    - Teamwork experiences
    - Challenges overcome
    - Learning experiences
    - Career motivations
    - Problem-solving examples

    Make it conversational and open-ended.

    Resume Excerpt:
    {resume_text[:2000]}... [truncated]"""
    response = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama3-70b-8192",
        temperature=0.7,
    )
    return clean_markdown(response.choices[0].message.content)

def evaluate_response(question, response_text):
    prompt = f"""Evaluate this interview response on:
    1. Clarity (1-5)
    2. Confidence (1-5)
    3. Relevance (1-5)
    4. Suggested improvements

    Question: {question}
    Response: {response_text}"""
    evaluation = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama3-70b-8192",
        temperature=0.2,
    )
    return clean_markdown(evaluation.choices[0].message.content)

def gtts_speak(text):
    try:
        if not text.strip():
            raise ValueError("Empty text")
        tts = gTTS(text, lang="en", tld="com")
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tts.save(tmp.name)
            return tmp.name
    except Exception as e:
        gr.Warning(f"gTTS Error: {e}")
        return None

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("<h1 style='font-size: 3em; text-align: center;'>🚀 Ready Set Hire</h1>")

    with gr.Tab("📄 Resume Analysis"):
        with gr.Row():
            with gr.Column():
                resume_upload = gr.File(label="📄 Upload Resume (PDF)", file_types=[".pdf"])
                process_btn = gr.Button("🔍 Analyze Resume", variant="primary")
            with gr.Column():
                resume_summary = gr.Textbox(label="📝 Resume Summary", lines=10)
                hear_summary_btn = gr.Button("🔊 Hear Summary")
                summary_audio = gr.Audio(visible=True)
                ats_score = gr.Textbox(label="📊 ATS Compatibility Score", interactive=False)
        process_btn.click(fn=process_resume, inputs=resume_upload, outputs=[resume_summary, ats_score])
        hear_summary_btn.click(fn=gtts_speak, inputs=resume_summary, outputs=summary_audio)

    with gr.Tab("🎤 Mock Interview"):
        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(
                    label="🎤 Record Your Response",
                    sources=["microphone"],
                    type="filepath",
                    interactive=True
                )
                transcribe_btn = gr.Button("📝 Transcribe Response")
                question_box = gr.Textbox(label="❓ Current Question")
                generate_btn = gr.Button("🤖 Generate New Question")
                gtts_question_btn = gr.Button("🔊 Hear Question")
                question_audio = gr.Audio(visible=True)
            with gr.Column():
                transcription = gr.Textbox(label="💬 Your Response")
                evaluation = gr.Textbox(label="📝 Feedback", lines=8)
                gtts_feedback_btn = gr.Button("🔊 Hear Feedback")
                feedback_audio = gr.Audio(visible=True)

        transcribe_btn.click(fn=transcribe_audio, inputs=audio_input, outputs=transcription)
        generate_btn.click(fn=generate_question, inputs=resume_summary, outputs=question_box)
        transcription.change(fn=evaluate_response, inputs=[question_box, transcription], outputs=evaluation)
        gtts_question_btn.click(fn=gtts_speak, inputs=question_box, outputs=question_audio)
        gtts_feedback_btn.click(fn=gtts_speak, inputs=evaluation, outputs=feedback_audio)

    gr.Markdown("""
    <div style='text-align:center; margin-top:2em; color:gray'>
      🚀 Built by Cognify.AI
    </div>
    """)

if __name__ == "__main__":
    demo.launch()

# 🚀 Ready Set Hire

Ready Set Hire is an AI-powered resume analyzer and mock interview assistant. It evaluates resumes, calculates ATS compatibility, generates personalized interview questions, and provides instant voice feedback — all within a clean, interactive Gradio UI.

---

## 🛠 Features

* 📄 **Resume Analysis:**

  * Upload PDF resumes
  * Generate summaries
  * Calculate ATS (Applicant Tracking System) score

* 🎤 **Mock Interview:**

  * Generate interview questions
  * Record and transcribe your answers
  * Get AI-powered feedback
  * Listen to the questions and feedback as audio

---

## 🧰 Tech Stack

* Python
* [Gradio](https://gradio.app/) (UI)
* [LangChain](https://www.langchain.com/) (Document Processing)
* [FAISS](https://github.com/facebookresearch/faiss) (Vector Store)
* [HuggingFace Sentence Transformers](https://www.sbert.net/)
* [Groq API](https://console.groq.com/) (LLM Backend)
* [ElevenLabs](https://www.elevenlabs.io/) & gTTS (Text-to-Speech)
* [Faster Whisper](https://github.com/SYSTRAN/faster-whisper) (ASR)

---

## 🚀 Installation

1. **Clone the repo**

```bash
git clone https://github.com/your-username/ready-set-hire.git
cd ready-set-hire
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Set environment variables** Create a `.env` file with the following:

```env
GROQ_API_KEY=your_groq_api_key
ELEVENLABS_API_KEY=your_elevenlabs_api_key
```

4. **Launch the app**

```bash
python app.py
```

---

## 📁 File Structure

```
ready-set-hire/
├── app.py            # Main application code
├── .env              # Environment variables (DO NOT COMMIT)
├── requirements.txt  # Dependencies
└── README.md         # Project description
```

---

## ✅ TODO

* Add session persistence
* Extend ATS scoring criteria
* Add support for more file types

---

## 🛡 Safety

API keys and sensitive info are stored using environment variables in a `.env` file. Be sure to \*\*add \*\*`** to **`.

---

## 🧠 Inspiration

Built for job seekers, recruiters, and AI enthusiasts to improve interview preparedness and resume quality.

---

## 🏷 License

MIT

---

## 👨‍💻 Built by Cognify.AI


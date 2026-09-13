# 🎓 Campus Buddy Pro

> An AI-powered campus assistant for students — combining conversational AI, PDF question answering, hybrid retrieval, website crawling, and live web search in one Streamlit application.

## ✨ Why Campus Buddy Pro?

Campus information is often scattered across PDFs, websites, notices, and the open web. Campus Buddy Pro brings these sources together so students can ask questions in natural language and get useful answers without manually searching through multiple sources.

## 🚀 Features

- 🤖 **AI Chat** — ask questions in a conversational interface
- 📄 **PDF Q&A** — upload documents and ask questions about their contents
- 🔎 **Hybrid Mode** — combine uploaded-document context with internet search
- 🌐 **Web Intelligence** — search the web for current information
- 🕷️ **Website Crawler** — crawl pages from a website and extract readable content
- 🧠 **Semantic Search** — use Hugging Face embeddings with FAISS for vector retrieval
- ⚡ **Groq-powered responses** — fast LLM inference
- 🎯 **Campus-ready information** — predefined answers for common campus questions
- 📱 **Responsive Streamlit UI** — designed for an interactive student experience
- 🛡️ **Environment-based secrets** — API keys stay outside the source code

## 🧩 Modes

| Mode | Best for |
|---|---|
| 💬 AI Mode | General questions and web-backed answers |
| 📄 PDF Mode | Questions grounded in uploaded PDFs |
| 🔀 Hybrid Mode | Combining document context with web information |
| 🌐 Web Crawling Mode | Extracting information from a selected website |

## 🏗️ How It Works

```text
                    ┌──────────────────────┐
                    │   Campus Buddy Pro   │
                    │      Streamlit       │
                    └──────────┬───────────┘
                               │
             ┌─────────────────┼─────────────────┐
             │                 │                 │
             ▼                 ▼                 ▼
        📄 PDF Input      🌐 Web Search      🕷️ Crawler
             │                 │                 │
             ▼                 ▼                 ▼
        Text Extraction   Search Results   Page Extraction
             │                 │                 │
             └────────────┬────┴─────────────────┘
                          ▼
                  🧠 Retrieval Layer
                  Embeddings + FAISS
                          │
                          ▼
                    ⚡ Groq LLM
                          │
                          ▼
                  💬 Student Answer
```

## 🛠️ Tech Stack

### AI & Retrieval
- Python
- LangChain
- Groq
- Hugging Face Embeddings
- FAISS

### Web & Documents
- Streamlit
- Requests
- BeautifulSoup
- PyPDF2
- DuckDuckGo Search

### Configuration
- python-dotenv

## 📁 Project Structure

```text
Chatbot/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── .env.example           # Environment variable template
├── .gitignore             # Local secrets and generated files
└── README.md              # Project documentation
```

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/sharique2123/Chatbot.git
cd Chatbot
```

### 2. Create a virtual environment

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy `.env.example` to `.env` and add your API key:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Never commit real API keys or `.env` files to GitHub.

### 5. Run the application

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in your terminal.

## 💡 Example Use Cases

- Ask about campus facilities, courses, fees, and dining options
- Upload a college brochure or academic PDF and ask questions about it
- Combine a college document with current web information
- Crawl an official college website and extract useful information
- Research a topic using web-backed answers

## 🔐 Security Notes

- Keep API keys in `.env`, not in source code.
- Do not upload secrets, credentials, or private documents to a public repository.
- Treat crawled and AI-generated content as untrusted input.
- Verify important academic, financial, admission, and administrative information with official sources.

## 🧪 Development Checklist

Before publishing a new version, verify:

- [ ] `pip install -r requirements.txt` completes successfully
- [ ] The app starts with `streamlit run app.py`
- [ ] PDF upload and question answering work
- [ ] Web search works
- [ ] Website crawling handles invalid URLs gracefully
- [ ] API keys are loaded from environment variables
- [ ] No secrets are committed

## 🔮 Future Improvements

- 💾 Persistent chat history
- 👤 User authentication
- 🗃️ Database-backed document management
- 📚 Multiple-document collections
- 🔖 Clickable source citations
- 🧾 Conversation export
- ⚡ Retrieval caching
- 🧪 Automated tests
- 🐳 Docker support
- ☁️ Cloud deployment
- 📊 Usage and retrieval analytics

## 📸 Screenshots

Add screenshots or a short demo GIF here after capturing the latest UI. Recommended images:

1. Home / AI chat
2. PDF Q&A
3. Hybrid search
4. Website crawler

## 👨‍💻 Author

**Sharique Azhar**  
GitHub: [@sharique2123](https://github.com/sharique2123)

## 📄 License

See the repository source files for the applicable licensing information.

---

⭐ If Campus Buddy Pro helps you, consider starring the repository!

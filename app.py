"""
Campus Buddy Chatbot - ChatGPT-like with Web Intelligence

Features:
- Ask questions directly (like ChatGPT) - answers from internet
- Upload PDFs for context-aware answers
- Website crawling
- Real-time web search integration
- Multi-mode operation:
  * Pure AI Mode (internet-only, like ChatGPT)
  * PDF Mode (document-based answers)
  * Hybrid Mode (PDFs + Internet)
  * Web Crawling Mode
"""

import os
from pathlib import Path
import re
from datetime import datetime
from urllib.parse import urljoin, urlparse
import time
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

import streamlit as st
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun
from PyPDF2 import PdfReader

# ============================================================================
# 1. WEB SEARCH & INTERNET FUNCTIONS (ENHANCED - ChatGPT MODE!)
# ============================================================================

def perform_comprehensive_web_search(query: str, num_results: int = 5) -> dict:
    """Perform comprehensive web search and extract content."""
    try:
        search = DuckDuckGoSearchRun()
        results = search.run(query)
        
        if not results:
            return {"success": False, "content": "", "sources": []}
        
        return {
            "success": True,
            "content": results,
            "sources": [{"title": query, "url": "Web Search Results"}]
        }
    except Exception as e:
        return {"success": False, "content": str(e), "sources": []}


def extract_web_content(search_results: str) -> list:
    """Extract structured information from web search results."""
    points = []
    sentences = re.split(r'[.!?;]\s+', search_results)
    
    for sentence in sentences:
        sentence = sentence.strip()
        
        if len(sentence) < 20 or len(sentence) > 400:
            continue
        
        if any(pattern in sentence.lower() for pattern in 
               ['click here', 'read more', 'sponsored', 'advertisement', 'cookie']):
            continue
        
        sentence = re.sub(r'\[.*?\]', '', sentence)
        sentence = re.sub(r'\s+', ' ', sentence)
        
        if sentence and len(sentence) > 20:
            points.append(sentence)
    
    return points[:10]


# ============================================================================
# 2. WEB CRAWLING FUNCTIONS
# ============================================================================

def is_valid_url(url: str) -> bool:
    """Validate URL format."""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False


def is_allowed_url(url: str, base_domain: str) -> bool:
    """Check if URL belongs to the same domain."""
    try:
        parsed_url = urlparse(url)
        parsed_base = urlparse(base_domain)
        return parsed_url.netloc == parsed_base.netloc
    except:
        return False


def clean_text(text: str) -> str:
    """Clean extracted text from HTML."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,:;!?()-]', '', text)
    return text.strip()


def extract_text_from_url(url: str, timeout: int = 10) -> tuple:
    """Extract text content from a webpage."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        for script in soup(["script", "style"]):
            script.decompose()
        
        title = soup.title.string if soup.title else "Untitled"
        text = soup.get_text()
        text = clean_text(text)
        
        if not text.strip():
            return None, None
        
        return title, text
        
    except Exception as e:
        return None, f"Error fetching {url}: {str(e)}"


def crawl_website(base_url: str, max_pages: int = 10, max_depth: int = 2) -> dict:
    """Crawl a website and extract text from pages."""
    if not is_valid_url(base_url):
        return {"error": "Invalid URL format"}
    
    crawled_pages = {}
    visited_urls = set()
    to_visit = [(base_url, 0)]
    
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    
    while to_visit and len(crawled_pages) < max_pages:
        current_url, depth = to_visit.pop(0)
        
        if current_url in visited_urls or depth > max_depth:
            continue
        
        visited_urls.add(current_url)
        
        progress = len(crawled_pages) / max_pages
        progress_placeholder.progress(progress)
        status_placeholder.write(f"📄 Crawling ({len(crawled_pages)}/{max_pages}): {current_url[:60]}...")
        
        title, content = extract_text_from_url(current_url)
        
        if content and not isinstance(content, str) or not content.startswith("Error"):
            crawled_pages[current_url] = {
                "title": title,
                "content": content,
                "depth": depth
            }
            
            if depth < max_depth and len(crawled_pages) < max_pages:
                try:
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                    }
                    response = requests.get(current_url, headers=headers, timeout=5)
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    for link in soup.find_all('a', href=True):
                        url = link['href']
                        absolute_url = urljoin(current_url, url)
                        
                        if (is_valid_url(absolute_url) and 
                            is_allowed_url(absolute_url, base_url) and
                            absolute_url not in visited_urls):
                            
                            to_visit.append((absolute_url, depth + 1))
                            
                            if len(crawled_pages) >= max_pages:
                                break
                
                except:
                    pass
        
        time.sleep(0.5)
    
    progress_placeholder.empty()
    status_placeholder.empty()
    
    return crawled_pages


# =============================================================================
# 3. PRE-ANSWERED QUESTIONS DATABASE
# =============================================================================

PRE_ANSWERED_QUESTIONS = {
    "🏛️ What facilities are available in campus?": """
**Campus Facilities Overview:**

**1️⃣ Central Library**
• Large collection of textbooks & reference books
• Digital library with e-journals
• Quiet study zones
• Computer terminals for research

**2️⃣ Smart Classrooms**
• Projectors & smart boards
• Audio-visual teaching system
• Wi-Fi enabled rooms

**3️⃣ Advanced Laboratories**
• Department-specific labs
• Modern equipment for practical learning
• Project & research support

**4️⃣ Hostels (Boys & Girls)**
• Separate secure accommodation
• Furnished rooms
• Mess facility (4 meals daily)

**5️⃣ Sports Grounds & Courts**
• Cricket & football ground
• Basketball & volleyball courts
• Indoor badminton
""",

    "💰 What are the tuition fees?": """
**Fee Structure:**

**B.Tech (Engineering)**
• Annual tuition: ₹2.85 Lakh per year
• Total: ₹11.4 L – ₹11.8 L for 4 years

**Other Programs**
• BBA: ~₹6 L total
• B.Sc: ~₹3.75 L – 5 L total
• B.Pharm: ~₹3.4 L total
""",

    "🎓 What courses are offered?": """
**Engineering Programs:**
✓ B.Tech in Computer Science (CSE)
✓ B.Tech in AI & Machine Learning
✓ B.Tech in Civil Engineering
✓ B.Tech in Electronics & Communication (ECE)
✓ B.Tech in Mechanical Engineering

**Other Programs:**
✓ BBA & B.Com
✓ B.Pharm & Pharm.D
✓ MBA & MCA
""",

    "🍽️ What dining options are available?": """
**Dining Facilities:**

**Central Cafeteria**
• Multiple cuisine options
• North Indian, South Indian, Chinese
• Hygienic dining area

**Hostel Mess**
• 4 meals daily
• Vegetarian & non-vegetarian
• Weekly menu rotation
""",
}

# =============================================================================
# 4. PAGE CONFIGURATION & CSS
# =============================================================================

st.set_page_config(
    page_title="Campus Buddy Pro",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "intro_shown" not in st.session_state:
    st.session_state.intro_shown = False

if not st.session_state.intro_shown:
    st.markdown("""
    <style>
    .intro {
        position: fixed;
        inset: 0;
        background: linear-gradient(135deg, #667eea, #764ba2);
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
        z-index: 999999;
        color: white;
    }

    .loader {
        margin-top: 20px;
        border: 6px solid rgba(255,255,255,0.3);
        border-top: 6px solid white;
        border-radius: 50%;
        width: 60px;
        height: 60px;
        animation: spin 1s linear infinite;
    }

    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    </style>

    <div class="intro">
        <h1>🎓 Campus Buddy Pro</h1>
        <p>Loading AI Assistant...</p>
        <div class="loader"></div>
    </div>
    """, unsafe_allow_html=True)

    time.sleep(2)
    st.session_state.intro_shown = True
    st.rerun()

advanced_css = """
<style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(-45deg, #667eea, #764ba2, #f093fb, #4facfe);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
        color: #fff;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        color: #fff;
    }
    
    h1 {
        color: #fff !important;
        text-shadow: 0 0 20px rgba(102, 126, 234, 0.8), 2px 2px 4px rgba(0,0,0,0.3);
        font-size: 2.8rem !important;
        font-weight: 800 !important;
        letter-spacing: 1px;
        animation: titlePulse 2s ease-in-out infinite;
    }
    
    @keyframes titlePulse {
        0%, 100% { text-shadow: 0 0 20px rgba(102, 126, 234, 0.8), 2px 2px 4px rgba(0,0,0,0.3); }
        50% { text-shadow: 0 0 30px rgba(102, 126, 234, 1), 2px 2px 8px rgba(0,0,0,0.5); }
    }
    
    h2 {
        color: #fff !important;
        border-bottom: 2px solid #667eea !important;
        padding-bottom: 10px !important;
        font-weight: 700 !important;
    }
    
    h3 {
        color: #fff !important;
        font-weight: 600 !important;
    }
    
    p {
        color: rgba(255, 255, 255, 0.9) !important;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: none !important;
        border-radius: 15px !important;
        padding: 12px 30px !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
        transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.05) !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.8) !important;
    }
    
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background-color: rgba(0, 0, 0, 0.3) !important;
        color: white !important;
        border: 2px solid #667eea !important;
        border-radius: 12px !important;
        padding: 12px 15px !important;
    }

    .stTextInput > div > div > input::placeholder,
    .stTextArea > div > div > textarea::placeholder {
        color: rgba(255,255,255,0.6) !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border: 2px solid #667eea !important;
        box-shadow: 0 0 20px rgba(102, 126, 234, 0.5) !important;
    }
    
    .stCheckbox > label {
        color: rgba(255, 255, 255, 0.9) !important;
    }
    
    .stAlert {
        border-radius: 15px !important;
        padding: 1.5rem !important;
        animation: slideIn 0.5s ease-out !important;
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    [data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.1) !important;
        border-radius: 15px !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease !important;
    }
    
    [data-testid="metric-container"]:hover {
        background: rgba(255, 255, 255, 0.15) !important;
        transform: translateY(-5px);
    }
    
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.1) !important;
        color: white !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        transition: all 0.3s ease !important;
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(255, 255, 255, 0.15) !important;
        transform: translateX(5px);
    }
    
    .answer-section {
        background: rgba(255, 255, 255, 0.1) !important;
        border-left: 4px solid #667eea !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        animation: slideIn 0.5s ease-out !important;
    }
    
    .source-section {
        background: rgba(255, 255, 255, 0.08) !important;
        border-radius: 10px !important;
        padding: 1rem !important;
        border: 1px dashed rgba(255, 255, 255, 0.2) !important;
        transition: all 0.3s ease !important;
    }
    
    .source-section:hover {
        background: rgba(255, 255, 255, 0.12) !important;
    }
    
    .web-source-section {
        background: linear-gradient(90deg, rgba(102, 126, 234, 0.15), rgba(118, 75, 162, 0.15)) !important;
        border-left: 4px solid #667eea !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
    }

    .mode-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        margin: 0.5rem 0.25rem;
    }

    .mode-ai {
        background: rgba(76, 175, 80, 0.3);
        border: 1px solid #4CAF50;
        color: #c8e6c9;
    }

    .mode-pdf {
        background: rgba(33, 150, 243, 0.3);
        border: 1px solid #2196F3;
        color: #bbdefb;
    }

    .mode-web {
        background: rgba(255, 152, 0, 0.3);
        border: 1px solid #FF9800;
        color: #ffe0b2;
    }

    .mode-hybrid {
        background: rgba(156, 39, 176, 0.3);
        border: 1px solid #9C27B0;
        color: #e1bee7;
    }
</style>
"""

st.markdown(advanced_css, unsafe_allow_html=True)

# =============================================================================
# 5. ENVIRONMENT & API SETUP
# =============================================================================

def load_and_validate_groq_key():
    """Load and validate Groq API key."""
    env_path = Path(__file__).parent / ".env"
    
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True)
    else:
        load_dotenv(override=True)
    
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    
    if not api_key:
        st.error(
            "❌ **GROQ_API_KEY not found**\n\n"
            "**Setup Instructions:**\n"
            "1. Sign up free at: https://console.groq.com\n"
            "2. Get your API key\n"
            "3. Create `.env` file\n"
            "4. Add: `GROQ_API_KEY=gsk_...`"
        )
        st.stop()
    
    return api_key


@st.cache_resource
def initialize_groq(api_key: str):
    """Initialize Groq LLM and embeddings."""
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        llm = ChatGroq(
            groq_api_key=api_key,
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            max_tokens=2048,
        )
        
        return embeddings, llm
        
    except Exception as e:
        st.error(f"❌ Failed to initialize Groq: {str(e)}")
        st.stop()


# =============================================================================
# 6. PDF PROCESSING
# =============================================================================

def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from PDF."""
    try:
        pdf_reader = PdfReader(pdf_file)
        
        if not pdf_reader.pages:
            raise ValueError("PDF file appears to be empty.")
        
        text = ""
        for page in pdf_reader.pages:
            try:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
            except:
                continue
        
        if not text.strip():
            raise ValueError("No readable text found in PDF.")
        
        return text
        
    except Exception as e:
        raise ValueError(f"Failed to process PDF: {str(e)}")


def split_and_embed_texts(texts_dict: dict, embeddings) -> FAISS:
    """Split and embed texts."""
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        
        all_chunks = []
        
        for source_name, text in texts_dict.items():
            chunks = text_splitter.split_text(text)
            for chunk in chunks:
                all_chunks.append(f"[Source: {source_name}]\n\n{chunk}")
        
        if not all_chunks:
            raise ValueError("No chunks created.")
        
        st.write(f"📊 Created {len(all_chunks)} text chunks from {len(texts_dict)} source(s)")
        vector_store = FAISS.from_texts(all_chunks, embeddings)
        return vector_store
            
    except Exception as e:
        raise ValueError(f"Text processing failed: {str(e)}")


# =============================================================================
# 7. ANSWER GENERATION (ChatGPT-LIKE!)
# =============================================================================

def answer_with_internet_only(llm, user_question: str) -> tuple:
    """
    Answer question using ONLY internet (like ChatGPT).
    No PDFs needed.
    """
    try:
        with st.spinner("🌐 Searching the internet for you..."):
            web_results = perform_comprehensive_web_search(user_question)
        
        if not web_results["success"]:
            return "Unable to find information on the internet.", [], ""
        
        web_content = web_results["content"]
        
        prompt = f"""You are a helpful AI assistant like ChatGPT. Answer the user's question comprehensively using the internet search results provided.

Instructions:
1. Provide a detailed, well-structured answer
2. Use information from the search results
3. Be conversational and helpful
4. If information is incomplete, acknowledge it
5. Format with bullet points where appropriate
6. Provide practical advice when relevant

Internet Search Results:
{web_content}

User Question:
{user_question}

Your Answer:"""
        
        response = llm.invoke(prompt)
        return response.content, [], web_content
            
    except Exception as e:
        return f"Error: {str(e)}", [], ""


def answer_with_pdf_context(vector_store: FAISS, llm, user_question: str, include_internet: bool = True) -> tuple:
    """Answer using PDF context (with optional internet)."""
    try:
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(user_question)
        
        pdf_context = "\n\n".join([doc.page_content for doc in docs])
        
        web_content = ""
        if include_internet:
            with st.spinner("🌐 Enhancing with internet search..."):
                web_results = perform_comprehensive_web_search(user_question)
                if web_results["success"]:
                    web_content = f"\n\nInternet Search Results:\n{web_results['content']}"
        
        prompt = f"""You are a helpful AI assistant answering questions based on uploaded documents and optionally internet information.

Instructions:
1. PRIORITIZE information from uploaded documents
2. Use internet information to supplement
3. Clearly indicate the source of information
4. Be detailed and comprehensive
5. Use formatting with bullet points and headers
6. If information is not in documents, clearly say so

Uploaded Document Context:
{pdf_context}
{web_content}

User Question:
{user_question}

Your Answer (cite sources):"""
        
        response = llm.invoke(prompt)
        return response.content, docs, web_content
            
    except Exception as e:
        return f"Error: {str(e)}", [], ""


def answer_hybrid_mode(vector_store: FAISS, llm, user_question: str) -> tuple:
    """
    Full ChatGPT-like experience: Use PDFs + Internet.
    """
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(user_question)
    
    pdf_context = "\n\n".join([doc.page_content for doc in docs]) if docs else "No documents available."
    
    with st.spinner("🌐 Fetching internet information..."):
        web_results = perform_comprehensive_web_search(user_question)
    
    web_content = web_results["content"] if web_results["success"] else "No internet results found."
    
    prompt = f"""You are an intelligent AI assistant providing comprehensive answers.

Instructions:
1. Combine knowledge from documents AND internet
2. Provide the most complete answer possible
3. Use bullet points and structured formatting
4. Cite sources where applicable
5. Be conversational and helpful
6. Provide practical examples when relevant

Document Context:
{pdf_context}

Internet Search Results:
{web_content}

User Question:
{user_question}

Comprehensive Answer:"""
    
    response = llm.invoke(prompt)
    return response.content, docs, web_content


# =============================================================================
# 8. MAIN APP INITIALIZATION
# =============================================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "question_count" not in st.session_state:
    st.session_state.question_count = 0

if "mode" not in st.session_state:
    st.session_state.mode = "AI_ONLY"  # AI_ONLY, PDF_ONLY, HYBRID, WEB_CRAWL

api_key = load_and_validate_groq_key()

with st.spinner("🚀 Loading AI models..."):
    embeddings, llm = initialize_groq(api_key)

# =============================================================================
# 9. HEADER & METRICS
# =============================================================================

col_header = st.columns([1, 3, 1])

with col_header[1]:
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1>🎓 Campus Buddy Pro</h1>
        <p style="font-size: 1.1rem; color: rgba(255,255,255,0.9); margin: 0.5rem 0;">
            ✨ ChatGPT-like Campus Intelligence ✨
        </p>
        <p style="color: rgba(255,255,255,0.7); font-size: 0.95rem;">
            Ask Anything • PDFs Optional • Internet Powered
        </p>
    </div>
    """, unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("💬 Questions Asked", st.session_state.question_count)

with col2:
    if "vector_store" in st.session_state:
        st.metric("📚 Mode", "HYBRID" if st.session_state.mode == "HYBRID" else "PDF")
    else:
        st.metric("📚 Mode", "AI ONLY")

with col3:
    st.metric("⚡ Speed", "Real-time")

with col4:
    st.metric("🧠 AI", "Groq LLaMA")

st.divider()

# =============================================================================
# 10. SIDEBAR CONTROLS
# =============================================================================

with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h2 style="color: #667eea; margin: 0;">⚙️ Settings</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Mode selection
    st.markdown("### 🎯 Operating Mode")
    mode = st.radio(
        "Select how to use Campus Buddy:",
        options=[
            ("🤖 AI Only (ChatGPT Mode)", "AI_ONLY"),
            ("📄 PDF Only", "PDF_ONLY"),
            ("🔀 Hybrid (PDFs + Internet)", "HYBRID"),
            ("🌐 Web Crawling", "WEB_CRAWL")
        ],
        format_func=lambda x: x[0]
    )
    st.session_state.mode = mode[1]
    
    # Display current mode badge
    if st.session_state.mode == "AI_ONLY":
        st.markdown('<span class="mode-badge mode-ai">🤖 AI Only Mode</span>', unsafe_allow_html=True)
    elif st.session_state.mode == "PDF_ONLY":
        st.markdown('<span class="mode-badge mode-pdf">📄 PDF Mode</span>', unsafe_allow_html=True)
    elif st.session_state.mode == "HYBRID":
        st.markdown('<span class="mode-badge mode-hybrid">🔀 Hybrid Mode</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="mode-badge mode-web">🌐 Web Crawl</span>', unsafe_allow_html=True)
    
    st.divider()
    
    # Clear buttons
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🔄 Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.question_count = 0
            st.rerun()
    
    with col_btn2:
        if st.button("🗑️ Clear All", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    st.divider()
    
    # Help
    with st.expander("❓ How to Use", expanded=False):
        st.markdown("""
        **🤖 AI Only Mode:**
        - Ask any question
        - AI searches internet automatically
        - Like ChatGPT!
        
        **📄 PDF Mode:**
        - Upload campus PDFs
        - Answers from documents only
        
        **🔀 Hybrid Mode:**
        - Upload PDFs
        - AI uses both PDFs + Internet
        - Best for complete answers
        
        **🌐 Web Crawl:**
        - Crawl websites
        - Index all content
        - Ask about crawled sites
        """)
    
    # Pre-answered questions
    with st.expander("⭐ Quick Answers", expanded=True):
        st.markdown("**Click any question:**")
        
        for idx, question in enumerate(PRE_ANSWERED_QUESTIONS.keys()):
            if st.button(question, use_container_width=True, key=f"preanswer_{idx}"):
                st.session_state.selected_pre_answer = question
                st.session_state.show_pre_answered = True
                st.rerun()

# =============================================================================
# 11. DISPLAY PRE-ANSWERED QUESTION
# =============================================================================

if st.session_state.get("show_pre_answered") and "selected_pre_answer" in st.session_state:
    question = st.session_state.selected_pre_answer
    answer = PRE_ANSWERED_QUESTIONS[question]
    
    st.divider()
    
    st.markdown(f"""
    <div class="answer-section">
        <h3 style="margin-top: 0;">❓ Question</h3>
        <h4>{question}</h4>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="answer-section">
        <h3 style="margin-top: 0;">📝 Answer</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style="background: rgba(255,255,255,0.05); padding: 1.5rem; border-radius: 10px; border-left: 4px solid #667eea;">
        {answer}
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    col_feedback1, col_feedback2 = st.columns(2)
    
    with col_feedback1:
        st.markdown("**Helpful?**")
    
    with col_feedback2:
        col_yes, col_no = st.columns(2)
        with col_yes:
            if st.button("👍 Yes", use_container_width=True, key="feedback_yes"):
                st.success("Thank you!", icon="✅")
        with col_no:
            if st.button("👎 No", use_container_width=True, key="feedback_no"):
                st.info("We'll improve!", icon="💡")
    
    if st.button("🔙 Go Back", use_container_width=True):
        st.session_state.show_pre_answered = False
        st.rerun()

else:
    # =============================================================================
    # 12. MAIN INTERFACE - DIFFERENT MODES
    # =============================================================================
    
    if st.session_state.mode == "AI_ONLY":
        # =========== AI ONLY MODE (ChatGPT-like) ===========
        
        st.markdown("""
        ### 🤖 ChatGPT-like AI Mode
        
        Ask any question and I'll search the internet to find you the best answer!
        """)
        
        st.info(
            "💡 **No PDFs needed!** Just ask your questions like you would with ChatGPT. "
            "I'll automatically search the internet for you."
        )
        
        # Question input
        user_question = st.text_area(
            "💬 Ask me anything...",
            placeholder="e.g., What are the best engineering colleges in India? or Tell me about AI and machine learning...",
            height=100,
            key="ai_question"
        )
        
        col_btn1, col_btn2 = st.columns([1, 1])
        
        with col_btn1:
            ask_button = st.button("🚀 Get Answer", use_container_width=True, key="ai_ask_btn")
        
        with col_btn2:
            if st.button("📜 Chat History", use_container_width=True, key="ai_history"):
                if st.session_state.chat_history:
                    with st.expander("📜 Conversation History"):
                        for i, (q, a) in enumerate(st.session_state.chat_history, 1):
                            st.markdown(f"**Q{i}:** {q}")
                            st.markdown(f"**A{i}:** {a[:300]}...")
                            st.divider()
        
        if ask_button and user_question:
            try:
                st.session_state.question_count += 1
                
                with st.spinner("🔍 Searching internet and analyzing..."):
                    answer, docs, web_content = answer_with_internet_only(llm, user_question)
                
                st.session_state.chat_history.append((user_question, answer))
                
                # Display answer
                st.markdown("""
                <div class="answer-section">
                    <h3 style="margin-top: 0;">✨ Answer</h3>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div style="background: rgba(255,255,255,0.05); padding: 1.5rem; border-radius: 10px; border-left: 4px solid #667eea;">
                    {answer}
                </div>
                """, unsafe_allow_html=True)
                
                # Display internet sources
                if web_content:
                    with st.expander("🌐 Internet Sources", expanded=False):
                        st.markdown(f"""
                        <div class="web-source-section">
                            {web_content[:500]}...
                        </div>
                        """, unsafe_allow_html=True)
                
                st.divider()
                col_fb1, col_fb2 = st.columns(2)
                with col_fb1:
                    if st.button("👍 Helpful", use_container_width=True, key="ai_yes"):
                        st.success("Thanks!", icon="✅")
                with col_fb2:
                    if st.button("👎 Not Helpful", use_container_width=True, key="ai_no"):
                        st.info("I'll do better next time!", icon="💡")
                        
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
        
        # Display welcome message
        if not user_question:
            st.markdown("""
            <div style="text-align: center; padding: 3rem 1rem;">
                <h2 style="color: rgba(255,255,255,0.9);">💬 Ask Me Anything!</h2>
                <p style="font-size: 1.1rem; color: rgba(255,255,255,0.8); line-height: 1.8;">
                    🔍 <b>Questions:</b> Ask anything you want to know<br>
                    🌐 <b>Internet:</b> I search the web automatically<br>
                    ⚡ <b>Speed:</b> Get instant AI-powered answers<br>
                </p>
                <hr style="border-color: rgba(255,255,255,0.2); margin: 2rem 0;">
                <p style="color: rgba(255,255,255,0.7);">
                    <i>Similar to ChatGPT • Powered by Groq • Always up-to-date</i>
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    elif st.session_state.mode == "PDF_ONLY":
        # =========== PDF ONLY MODE ===========
        
        st.markdown("### 📄 Upload PDFs & Ask Questions")
        
        col_upload, col_status = st.columns([2, 1])
        
        with col_upload:
            uploaded_files = st.file_uploader(
                "Drag & drop PDFs or click to browse",
                type="pdf",
                accept_multiple_files=True,
                key="pdf_uploader"
            )
        
        with col_status:
            if "vector_store" in st.session_state and st.session_state.mode == "PDF_ONLY":
                st.markdown("""
                <div style="background: rgba(76, 175, 80, 0.2); padding: 1rem; border-radius: 10px; text-align: center; border-left: 4px solid #4CAF50;">
                    <h4 style="margin: 0; color: #fff;">✅ Ready</h4>
                    <p style="margin: 0.5rem 0 0 0; color: rgba(255,255,255,0.8);">PDFs loaded</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background: rgba(255, 193, 7, 0.2); padding: 1rem; border-radius: 10px; text-align: center; border-left: 4px solid #FFC107;">
                    <h4 style="margin: 0; color: #fff;">⏳ Waiting</h4>
                    <p style="margin: 0.5rem 0 0 0; color: rgba(255,255,255,0.8);">Upload PDFs</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Process PDFs
        if uploaded_files and ("vector_store" not in st.session_state or st.session_state.mode != "PDF_ONLY"):
            st.divider()
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                texts_dict = {}
                
                for idx, pdf_file in enumerate(uploaded_files):
                    progress = idx / len(uploaded_files)
                    progress_bar.progress(progress)
                    status_text.write(f"📖 Processing: **{pdf_file.name}**...")
                    
                    try:
                        text = extract_text_from_pdf(pdf_file)
                        texts_dict[pdf_file.name] = text
                    except ValueError as e:
                        st.error(f"❌ Error in {pdf_file.name}: {str(e)}")
                        continue
                
                if not texts_dict:
                    st.error("❌ No valid PDFs processed.")
                    st.stop()
                
                status_text.write("🔗 Creating embeddings...")
                progress_bar.progress(0.8)
                
                vector_store = split_and_embed_texts(texts_dict, embeddings)
                
                st.session_state.vector_store = vector_store
                st.session_state.uploaded_pdfs = list(texts_dict.keys())
                
                progress_bar.progress(1.0)
                
                st.balloons()
                st.success(f"🎉 Successfully loaded **{len(texts_dict)}** PDF(s)!")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
        
        # Q&A Section
        if "vector_store" in st.session_state and st.session_state.mode == "PDF_ONLY":
            st.divider()
            
            st.markdown("### ❓ Ask Your Questions")
            
            user_question = st.text_area(
                "What would you like to know from the PDFs?",
                placeholder="Type your question here...",
                height=80,
                key="pdf_question"
            )
            
            col_btn1, col_btn2 = st.columns([1, 1])
            
            with col_btn1:
                ask_button = st.button("🔍 Search PDFs", use_container_width=True, key="pdf_ask_btn")
            
            with col_btn2:
                if st.button("📜 History", use_container_width=True, key="pdf_history"):
                    if st.session_state.chat_history:
                        with st.expander("Chat History"):
                            for i, (q, a) in enumerate(st.session_state.chat_history, 1):
                                st.markdown(f"**Q{i}:** {q}")
                                st.markdown(f"**A{i}:** {a[:200]}...")
                                st.divider()
            
            if ask_button and user_question:
                try:
                    st.session_state.question_count += 1
                    
                    with st.spinner("🔍 Searching PDFs..."):
                        answer, docs, _ = answer_with_pdf_context(
                            st.session_state.vector_store,
                            llm,
                            user_question,
                            include_internet=False
                        )
                    
                    st.session_state.chat_history.append((user_question, answer))
                    
                    st.markdown("""
                    <div class="answer-section">
                        <h3 style="margin-top: 0;">📝 Answer</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div style="background: rgba(255,255,255,0.05); padding: 1.5rem; border-radius: 10px; border-left: 4px solid #667eea;">
                        {answer}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if docs:
                        with st.expander("📚 Source Documents"):
                            for idx, doc in enumerate(docs, 1):
                                st.markdown(f"**Source {idx}:**")
                                st.markdown(f"""
                                <div class="source-section">
                                    {doc.page_content[:250]}...
                                </div>
                                """, unsafe_allow_html=True)
                    
                    st.divider()
                    col_fb1, col_fb2 = st.columns(2)
                    with col_fb1:
                        if st.button("👍 Helpful", use_container_width=True, key="pdf_yes"):
                            st.success("Thanks!", icon="✅")
                    with col_fb2:
                        if st.button("👎 Not Helpful", use_container_width=True, key="pdf_no"):
                            st.info("I'll improve!", icon="💡")
                            
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    elif st.session_state.mode == "HYBRID":
        # =========== HYBRID MODE (PDFs + Internet) ===========
        
        st.markdown("""
        ### 🔀 Hybrid Mode: PDFs + Internet
        
        Upload PDFs AND ask questions. I'll search both your documents AND the internet for complete answers!
        """)
        
        col_upload, col_status = st.columns([2, 1])
        
        with col_upload:
            uploaded_files = st.file_uploader(
                "Upload PDFs (optional)",
                type="pdf",
                accept_multiple_files=True,
                key="hybrid_pdf_uploader"
            )
        
        with col_status:
            if "vector_store" in st.session_state and st.session_state.mode == "HYBRID":
                st.markdown("""
                <div style="background: rgba(76, 175, 80, 0.2); padding: 1rem; border-radius: 10px; text-align: center; border-left: 4px solid #4CAF50;">
                    <h4 style="margin: 0; color: #fff;">✅ Ready</h4>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("💡 PDFs are optional. You can ask questions without uploading!")
        
        # Process PDFs if uploaded
        if uploaded_files and ("vector_store" not in st.session_state or st.session_state.mode != "HYBRID"):
            st.divider()
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                texts_dict = {}
                
                for idx, pdf_file in enumerate(uploaded_files):
                    progress = idx / len(uploaded_files)
                    progress_bar.progress(progress)
                    status_text.write(f"📖 Processing: **{pdf_file.name}**...")
                    
                    try:
                        text = extract_text_from_pdf(pdf_file)
                        texts_dict[pdf_file.name] = text
                    except ValueError as e:
                        st.error(f"❌ Error in {pdf_file.name}: {str(e)}")
                        continue
                
                if not texts_dict:
                    st.warning("⚠️ No valid PDFs. You can still ask questions using internet!")
                else:
                    status_text.write("🔗 Creating embeddings...")
                    progress_bar.progress(0.8)
                    
                    vector_store = split_and_embed_texts(texts_dict, embeddings)
                    st.session_state.vector_store = vector_store
                    st.session_state.uploaded_pdfs = list(texts_dict.keys())
                
                progress_bar.progress(1.0)
                st.balloons()
                st.success("🎉 Ready to answer your questions!")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
        
        st.divider()
        
        st.markdown("### ❓ Ask Your Questions")
        st.info("💡 I'll search both your PDFs (if uploaded) AND the internet for the best answer!")
        
        user_question = st.text_area(
            "Ask me anything...",
            placeholder="Your question here...",
            height=100,
            key="hybrid_question"
        )
        
        col_btn1, col_btn2 = st.columns([1, 1])
        
        with col_btn1:
            ask_button = st.button("🚀 Search PDFs + Internet", use_container_width=True, key="hybrid_ask_btn")
        
        with col_btn2:
            if st.button("📜 History", use_container_width=True, key="hybrid_history"):
                if st.session_state.chat_history:
                    with st.expander("Chat History"):
                        for i, (q, a) in enumerate(st.session_state.chat_history, 1):
                            st.markdown(f"**Q{i}:** {q}")
                            st.markdown(f"**A{i}:** {a[:200]}...")
                            st.divider()
        
        if ask_button and user_question:
            try:
                st.session_state.question_count += 1
                
                with st.spinner("🔍 Searching PDFs and internet..."):
                    if "vector_store" in st.session_state and st.session_state.mode == "HYBRID":
                        answer, docs, web_content = answer_hybrid_mode(
                            st.session_state.vector_store,
                            llm,
                            user_question
                        )
                    else:
                        answer, docs, web_content = answer_with_internet_only(llm, user_question)
                
                st.session_state.chat_history.append((user_question, answer))
                
                st.markdown("""
                <div class="answer-section">
                    <h3 style="margin-top: 0;">✨ Comprehensive Answer</h3>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div style="background: rgba(255,255,255,0.05); padding: 1.5rem; border-radius: 10px; border-left: 4px solid #667eea;">
                    {answer}
                </div>
                """, unsafe_allow_html=True)
                
                col_sources, col_web = st.columns(2)
                
                if docs:
                    with col_sources:
                        with st.expander("📚 From Your PDFs"):
                            for idx, doc in enumerate(docs, 1):
                                st.markdown(f"**Document {idx}:**")
                                st.markdown(f"""
                                <div class="source-section">
                                    {doc.page_content[:250]}...
                                </div>
                                """, unsafe_allow_html=True)
                
                if web_content:
                    with col_web:
                        with st.expander("🌐 From Internet"):
                            st.markdown(f"""
                            <div class="web-source-section">
                                {web_content[:500]}...
                            </div>
                            """, unsafe_allow_html=True)
                
                st.divider()
                col_fb1, col_fb2 = st.columns(2)
                with col_fb1:
                    if st.button("👍 Helpful", use_container_width=True, key="hybrid_yes"):
                        st.success("Thanks!", icon="✅")
                with col_fb2:
                    if st.button("👎 Not Helpful", use_container_width=True, key="hybrid_no"):
                        st.info("I'll improve!", icon="💡")
                        
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    elif st.session_state.mode == "WEB_CRAWL":
        # =========== WEB CRAWL MODE ===========
        
        st.markdown("### 🌐 Crawl & Index Websites")
        
        website_url = st.text_input(
            "Enter website URL to crawl:",
            placeholder="https://www.example-campus.edu.in",
            key="crawl_url"
        )
        
        col_settings1, col_settings2, col_settings3 = st.columns(3)
        
        with col_settings1:
            max_pages = st.slider("Max Pages", 5, 50, 10)
        
        with col_settings2:
            max_depth = st.slider("Crawl Depth", 1, 3, 2)
        
        with col_settings3:
            st.markdown("")
            crawl_button = st.button("🚀 Start Crawling", use_container_width=True, key="crawl_btn")
        
        if crawl_button:
            if not website_url:
                st.error("❌ Please enter a website URL")
            elif not is_valid_url(website_url):
                st.error("❌ Invalid URL format")
            else:
                st.divider()
                
                with st.spinner("🕷️ Crawling website..."):
                    crawled_data = crawl_website(website_url, max_pages=max_pages, max_depth=max_depth)
                
                if "error" in crawled_data:
                    st.error(f"❌ {crawled_data['error']}")
                elif not crawled_data:
                    st.warning("⚠️ No pages crawled.")
                else:
                    st.success(f"✅ Crawled **{len(crawled_data)}** pages!")
                    
                    texts_dict = {}
                    for url, page_data in crawled_data.items():
                        page_name = f"{urlparse(url).netloc} - {page_data['title']}"
                        texts_dict[page_name] = page_data['content']
                    
                    with st.spinner("🔗 Creating embeddings..."):
                        vector_store = split_and_embed_texts(texts_dict, embeddings)
                    
                    st.session_state.vector_store = vector_store
                    st.session_state.crawled_websites = {website_url: crawled_data}
                    st.session_state.mode = "WEB_CRAWL"
                    
                    st.balloons()
                    st.success("🎉 Website indexed! Now ask questions!")
        
        # Q&A Section
        if "vector_store" in st.session_state and st.session_state.mode == "WEB_CRAWL":
            st.divider()
            
            st.markdown("### ❓ Ask About the Crawled Website")
            
            user_question = st.text_area(
                "Ask questions about the crawled website:",
                placeholder="Type your question...",
                height=100,
                key="crawl_question"
            )
            
            col_btn1, col_btn2 = st.columns([1, 1])
            
            with col_btn1:
                ask_button = st.button("🔍 Search", use_container_width=True, key="crawl_ask_btn")
            
            with col_btn2:
                if st.button("📜 History", use_container_width=True, key="crawl_history"):
                    if st.session_state.chat_history:
                        with st.expander("Chat History"):
                            for i, (q, a) in enumerate(st.session_state.chat_history, 1):
                                st.markdown(f"**Q{i}:** {q}")
                                st.markdown(f"**A{i}:** {a[:200]}...")
                                st.divider()
            
            if ask_button and user_question:
                try:
                    st.session_state.question_count += 1
                    
                    with st.spinner("🔍 Searching crawled content..."):
                        answer, docs, _ = answer_with_pdf_context(
                            st.session_state.vector_store,
                            llm,
                            user_question,
                            include_internet=False
                        )
                    
                    st.session_state.chat_history.append((user_question, answer))
                    
                    st.markdown("""
                    <div class="answer-section">
                        <h3 style="margin-top: 0;">📝 Answer</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div style="background: rgba(255,255,255,0.05); padding: 1.5rem; border-radius: 10px; border-left: 4px solid #667eea;">
                        {answer}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if docs:
                        with st.expander("📄 Source Pages"):
                            for idx, doc in enumerate(docs, 1):
                                st.markdown(f"**Page {idx}:**")
                                st.markdown(f"""
                                <div class="source-section">
                                    {doc.page_content[:250]}...
                                </div>
                                """, unsafe_allow_html=True)
                    
                    st.divider()
                    col_fb1, col_fb2 = st.columns(2)
                    with col_fb1:
                        if st.button("👍 Helpful", use_container_width=True, key="crawl_yes"):
                            st.success("Thanks!", icon="✅")
                    with col_fb2:
                        if st.button("👎 Not Helpful", use_container_width=True, key="crawl_no"):
                            st.info("I'll improve!", icon="💡")
                            
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

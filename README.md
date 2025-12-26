<div align="center">
<img src="https://i.imgur.com/2n3g2yH.png" alt="LegalEase Logo" width="150">
<h1>LegalEase: Your Advanced AI Legal Assistant</h1>
<p>
An intelligent, multi-functional AI assistant designed to streamline legal research and document analysis for students and professionals in India.
</p>

<p>
<img src="https://img.shields.io/badge/Python-3.9+-blue?logo=python" alt="Python Version">
<img src="https://img.shields.io/badge/Framework-Streamlit-red?logo=streamlit" alt="Framework">
<img src="https://img.shields.io/badge/LLM-Llama_3_ (Groq)-green" alt="LLM">
<img src="https://img.shields.io/badge/License-MIT-yellow" alt="License">
</p>
</div>

🚀 Overview
LegalEase is a sophisticated, open-source AI assistant that serves as a comprehensive co-pilot for legal work in India. It addresses a critical need for accessible, fast, and intelligent tools to navigate the complexities of legal research and document analysis. The application is built with a modular architecture that combines multiple AI capabilities into a single, intuitive, WhatsApp-inspired interface.

The system is designed to operate in two distinct but seamlessly integrated modes:

General AI Assistant: By default, LegalEase acts as a conversational AI that can answer general legal questions. It leverages a private, pre-loaded library of key Indian statutes (like the IPC, Labour Act, etc.) and can perform live web searches for up-to-the-minute information, recent case law, and broader legal concepts.

Document Analysis Expert: The application's functionality transforms the moment a user uploads a document (PDF or image). The assistant's context pivots entirely to the uploaded file, turning it into a specialized tool that can provide deep insights. Users can request detailed, structured summaries, ask specific questions about the content, and even find similar case law by triggering a targeted web scraping process.

✨ Key Features
Unified Conversational Interface: A single, elegant chat window that intelligently adapts its behavior based on the context, eliminating the need for users to switch between different modes or tabs.

Hybrid RAG & Web Search: Combines a local, private Retrieval-Augmented Generation (RAG) system for querying a static library of legal documents with the ability to perform live web searches for dynamic, real-time information.

Intelligent Intent Routing: A lightweight "Router of Chains" architecture analyzes each user query to determine the underlying intent (e.g., greeting, library question, web search, off-topic) and directs the request to the appropriate, purpose-built processing chain.

On-the-Fly Document Analysis: Users can upload PDF or image files (JPG, PNG) at any time. The system processes these files, creates a temporary in-memory vector store, and makes the assistant an instant expert on that document.

Advanced OCR Capabilities: Utilizes the Tesseract engine to extract text from scanned documents and images, making non-digital legal documents fully searchable and analyzable.

Targeted Case Law Scraping: When in Document Analysis mode, the assistant can generate a concise summary of the uploaded document and use it as a query for a custom-built Selenium/BeautifulSoup scraper. This tool performs a targeted search on indiankanoon.org to find and return direct hyperlinks to relevant case judgments.

Conversational Memory: Remembers the context of the current conversation (for both general chat and document analysis) to allow for natural, multi-turn follow-up questions.

🛠️ Tech Stack Explained
Component

Technology/Service

Justification

Frontend

Streamlit

Chosen for its ability to rapidly create beautiful, interactive data and chat applications purely in Python, making it ideal for fast-paced development and prototyping.

Backend & Orchestration

Python & LangChain

LangChain provides the essential framework for "chaining" together LLMs with other components like vector stores and tools, enabling the construction of complex, stateful AI applications.

LLMs (Language Models)

Groq API (Llama 3)

The Groq API provides unparalleled inference speed for Llama 3 models. We strategically use the fast 8B model for routing and simple tasks, and the powerful 70B model for complex generation and analysis, optimizing both cost and performance.

Embeddings Model

Hugging Face BAAI/bge-small-en-v1.5

A high-performance, open-source model that runs locally. This choice ensures user data privacy, as sensitive document contents are never sent to an external embedding service.

Vector Store

FAISS

An efficient, local vector library from Facebook AI. It's perfect for creating fast, in-memory indexes for both the persistent legal library and the temporary session-based document stores without requiring heavy database infrastructure.

Web Search (Case Law)

Selenium & BeautifulSoup

This combination provides a powerful, custom solution for targeted web scraping. It allows the application to automate a web browser to navigate indiankanoon.org, a site that may be difficult to query with simple API calls, and extract precise information.

Web Search (General)

Tavily AI

Tavily is a search API specifically designed for LLM agents. It returns clean, pre-processed search results, which is more efficient for general queries than raw scraping.

OCR (Image Text Recognition)

Tesseract (pytesseract)

The industry-standard open-source OCR engine. Its integration allows the application to handle a wider variety of legal documents, including scanned copies and photographs.

⚙️ Setup and Installation
Step 1: Project Structure
Ensure your project folder is set up correctly. Create a sub-folder named LEGAL-DATA and place your library of PDF documents inside it. This library will serve as the internal knowledge base for the General AI Assistant.

LegalEase/
├── app.py
├── requirements.txt
├── .env
└── LEGAL-DATA/
    ├── ipc_act.pdf
    └── CompaniesAct2013.pdf
    └── ...

Step 2: Install System Dependencies (for OCR & Web Scraping)
This project requires Google's Tesseract engine for OCR and Google Chrome for the Selenium web scraper.

Tesseract (OCR):

Windows: Download and install Tesseract from the official UB Mannheim page. Important: During installation, ensure you select the option to add Tesseract to your system's PATH variable.

macOS: brew install tesseract

Linux (Debian/Ubuntu): sudo apt-get install tesseract-ocr

Google Chrome: Ensure you have Google Chrome installed, as the Selenium scraper is configured to use it.

Step 3: Install Python Libraries
Open your terminal, navigate to your main project folder, and run the following command to install all the necessary Python packages from the requirements.txt file:

pip install -r requirements.txt

Step 4: Set Up Your API Keys
In your main project directory, create a new file and name it exactly .env.

Open the .env file with a text editor and add your API keys as shown below, replacing the placeholder text with your actual keys.

# Get your free key from https://console.groq.com/
GROQ_API_KEY="YOUR_GROQ_KEY_HERE"

# Get your free key from https://tavily.com/
TAVILY_API_KEY="YOUR_TAVILY_KEY_HERE"

▶️ How to Run the App
CRITICAL First Step: If a folder named faiss_index_library exists in your project from a previous version, delete it. The application will automatically create a new, correct one on its first run.

Open your terminal and navigate to the root directory of the project.

Run the following command:

streamlit run app.py

The first time you launch the app, it will take a few minutes to build the local vector database from your LEGAL-DATA folder. All subsequent launches will be almost instantaneous.

📖 How to Use LegalEase
General AI Assistant
This is the default mode when the application starts.

You can have a conversation, ask for definitions of legal terms, or inquire about general legal topics (e.g., "tell me about the laws for murder in India").

The assistant will intelligently decide whether to use its internal library of statutes or perform a live web search to provide the best possible answer.

Document Analysis Expert
Upload a Document: Use the file uploader in the sidebar to select one or more PDF or image files (JPG, PNG).

Analyze: Click the "Analyze Document" button. The assistant will process the file(s) and confirm when it's ready. The chat context will now be focused exclusively on this document.

Start Querying: You can now ask questions specifically about the uploaded file:

For a detailed summary: "give me a detailed summary of this case"

To find similar cases: "find similar cases" or "give me reference links"

For specific questions: "who was the appellant in this matter?"

👤 Author
Yash Kamra
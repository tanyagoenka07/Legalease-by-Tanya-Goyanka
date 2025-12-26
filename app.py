import os
import streamlit as st
from dotenv import load_dotenv
import shutil
from PIL import Image
import pytesseract
import time

# LangChain imports
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Web Scraping imports
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="LegalEase",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- STYLES (WhatsApp Inspired) ---
st.markdown("""
<style>
    /* Main App Background */
    .stApp {
        background-color: #E5DDD5; /* WhatsApp background color */
    }
    /* Chat Container */
    .st-emotion-cache-1jicfl2 {
        background-color: #E5DDD5;
    }
    /* Chat Bubbles */
    .st-emotion-cache-1c7y2kd {
        border-radius: 12px;
        padding: 12px 15px;
        margin-bottom: 10px;
        max-width: 85%;
        word-wrap: break-word;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    /* User Bubble */
    div[data-testid="chat-bubble-stream-user"] {
        background-color: #DCF8C6; /* WhatsApp user bubble color */
        align-self: flex-end;
        text-align: right;
    }
    /* Bot Bubble */
    div[data-testid="chat-bubble-stream-assistant"] {
        background-color: #FFFFFF; /* WhatsApp bot bubble color */
        align-self: flex-start;
        text-align: left;
    }
    /* Sidebar */
    .st-emotion-cache-16txtl3 {
        background-color: #F0F2F5;
    }
    h1 {
        color: #128C7E; /* WhatsApp Green */
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# --- CORE FUNCTIONS ---

def process_uploaded_files(uploaded_files):
    """Processes uploaded PDF or image files into text chunks."""
    text = ""
    temp_dir = "temp_uploads"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    for file in uploaded_files:
        file_path = os.path.join(temp_dir, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

        if file.type == "application/pdf":
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            for doc in documents:
                text += doc.page_content
        elif file.type in ["image/jpeg", "image/png"]:
            try:
                img = Image.open(file_path)
                text += pytesseract.image_to_string(img)
            except Exception as e:
                st.error(f"Error processing image {file.name}: {e}")

    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

    if not text:
        return None
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=250)
    return text_splitter.split_text(text)

@st.cache_resource
def setup_library_db():
    """Sets up the vector database from the pre-loaded LEGAL-DATA directory."""
    db_path = "faiss_index_library"
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    if not os.path.exists(db_path):
        with st.sidebar:
            with st.spinner("Building library database for the first time..."):
                if not os.path.exists('LEGAL-DATA'):
                    return None
                loader = DirectoryLoader('LEGAL-DATA/', glob="*.pdf", loader_cls=PyPDFLoader, show_progress=True, use_multithreading=True)
                documents = loader.load()
                if not documents:
                    return None
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                texts = text_splitter.split_documents(documents)
                db = FAISS.from_documents(texts, embeddings)
                db.save_local(db_path)
                st.success("Library database built successfully!")
                return db
    else:
        return FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)

@st.cache_resource
def get_webdriver_service():
    """Initializes and caches the webdriver service."""
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    return Service(ChromeDriverManager().install()), chrome_options

def scrape_indian_kanoon(query):
    """Scrapes Indian Kanoon for a given query and returns links."""
    service, options = get_webdriver_service()
    driver = webdriver.Chrome(service=service, options=options)
    try:
        driver.get("https://indiankanoon.org/")
        time.sleep(2)
        search_box = driver.find_element(By.NAME, "formInput")
        search_box.send_keys(query)
        search_box.send_keys(Keys.RETURN)
        time.sleep(3)
        soup = BeautifulSoup(driver.page_source, "html.parser")
        results = []
        for result in soup.find_all('div', class_='result_title', limit=5):
            title = result.find('a').get_text()
            link = "https://indiankanoon.org" + result.find('a')['href']
            results.append({'title': title, 'link': link})
        return results
    finally:
        driver.quit()

# --- MAIN APP LOGIC ---
load_dotenv()

if not os.getenv("GROQ_API_KEY") or not os.getenv("TAVILY_API_KEY"):
    st.error("API key(s) missing. Please add them to your .env file.")
    st.stop()

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(k=5, memory_key="chat_history", return_messages=True)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "doc_qa_retriever" not in st.session_state:
    st.session_state.doc_qa_retriever = None

# --- UI RENDERING ---

st.title("⚖️ LegalEase")

with st.sidebar:
    st.markdown("<h2>Document Analysis</h2>", unsafe_allow_html=True)
    st.markdown("Upload a document to make the assistant an expert on its content.")
    
    uploaded_files = st.file_uploader(
        "Upload PDF or Image files", 
        type=["pdf", "jpg", "png"], 
        accept_multiple_files=True, 
        key="doc_uploader"
    )
    
    if uploaded_files:
        if st.button("Analyze Document"):
            with st.spinner("Processing your document..."):
                text_chunks = process_uploaded_files(uploaded_files)
                if text_chunks:
                    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
                    session_db = FAISS.from_texts(text_chunks, embeddings)
                    st.session_state.doc_qa_retriever = session_db.as_retriever()
                    st.success("Document analysis complete!")
                    st.session_state.messages.append({"role": "assistant", "content": "I have analyzed your document. You can now ask me for a 'detailed summary', to 'find similar cases', or ask any specific question about its content."})
                else:
                    st.error("Could not extract text from the uploaded files.")
                    st.session_state.doc_qa_retriever = None

    st.markdown("---")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.memory.clear()
        st.session_state.doc_qa_retriever = None
        st.rerun()

library_db = setup_library_db()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input("Ask your legal question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response_text = ""
            try:
                if st.session_state.doc_qa_retriever:
                    # Document Q&A Logic
                    doc_router_prompt = ChatPromptTemplate.from_template(
                        """Given the user's question about an uploaded document, classify their intent: `summarize`, `find_similar_cases`, or `specific_question`.
                        User Question: {question}
                        Classification:"""
                    )
                    routing_llm = ChatGroq(model="llama3-8b-8192", temperature=0)
                    router = doc_router_prompt | routing_llm | StrOutputParser()
                    topic = router.invoke({"question": prompt})

                    if "summarize" in topic.lower():
                        summary_chain = ConversationalRetrievalChain.from_llm(
                            ChatGroq(model="llama3-70b-8192"),
                            retriever=st.session_state.doc_qa_retriever,
                            memory=st.session_state.memory,
                            combine_docs_chain_kwargs={"prompt": ChatPromptTemplate.from_template(
                                "Provide a highly detailed, structured summary of the following document context, including parties, key issues, court findings, and legal principles discussed:\n\n{context}\n\nDetailed Summary:"
                            )}
                        )
                        response = summary_chain.invoke({"question": prompt})
                        response_text = response['answer']
                    elif "find_similar_cases" in topic.lower():
                        with st.spinner("Summarizing document for search..."):
                            summary_chain = ConversationalRetrievalChain.from_llm(
                                ChatGroq(model="llama3-8b-8192"),
                                retriever=st.session_state.doc_qa_retriever,
                                memory=st.session_state.memory
                            )
                            summary = summary_chain.invoke({"question": "Provide a concise one-sentence summary of this document's key legal issue."})['answer']
                        
                        with st.spinner(f"Scraping Indian Kanoon for cases related to: '{summary}'"):
                            search_results = scrape_indian_kanoon(summary)
                        
                        if search_results:
                            response_text = "Here are some links to potentially similar cases I found on Indian Kanoon:\n\n"
                            for result in search_results:
                                response_text += f"- [{result['title']}]({result['link']})\n"
                        else:
                            response_text = "I could not find any similar cases on Indian Kanoon based on the document's summary."
                    else: # specific_question
                        specific_question_chain = ConversationalRetrievalChain.from_llm(
                            ChatGroq(model="llama3-70b-8192"),
                            retriever=st.session_state.doc_qa_retriever,
                            memory=st.session_state.memory
                        )
                        response = specific_question_chain.invoke({"question": prompt})
                        response_text = response['answer']
                else:
                    # General Assistant Logic
                    router_prompt = ChatPromptTemplate.from_template(
                        """Given the user's input, classify it into one of the following categories: `greeting`, `library_question`, `web_search`, `off_topic`
                        Conversation History: {chat_history}
                        User Input: {input}
                        Classification:"""
                    )
                    routing_llm = ChatGroq(model="llama3-8b-8192", temperature=0)
                    router = router_prompt | routing_llm | StrOutputParser()
                    route = router.invoke({
                        "input": prompt,
                        "chat_history": st.session_state.memory.chat_memory.messages
                    })

                    if "greeting" in route.lower():
                        llm = ChatGroq(model_name="llama3-8b-8192")
                        response_text = llm.invoke(f"You are a friendly legal assistant. Respond to this greeting: {prompt}").content
                    elif "library_question" in route.lower() and library_db:
                        rag_chain = ConversationalRetrievalChain.from_llm(
                            ChatGroq(model="llama3-70b-8192"),
                            retriever=library_db.as_retriever(),
                            memory=st.session_state.memory
                        )
                        response = rag_chain.invoke({"question": prompt})
                        response_text = response['answer']
                    elif "web_search" in route.lower():
                        web_search_prompt = ChatPromptTemplate.from_template(
                            """You are a helpful legal research assistant. Answer the user's question based on the provided web search results. Cite your sources using the URLs provided.
                            Search Results: {search_results}
                            Conversation History: {chat_history}
                            Question: {input}
                            Answer:"""
                        )
                        web_search_chain = (
                            RunnablePassthrough.assign(search_results=lambda x: TavilySearchResults(max_results=3).invoke(x["input"]))
                            | web_search_prompt
                            | ChatGroq(model="llama3-70b-8192", temperature=0.5)
                            | StrOutputParser()
                        )
                        response_text = web_search_chain.invoke({
                            "input": prompt,
                            "chat_history": st.session_state.memory.chat_memory.messages
                        })
                    else: # off_topic or library not available
                        llm = ChatGroq(model_name="llama3-8b-8192")
                        response_text = llm.invoke(f"You are LegalEase, a professional legal assistant. The user has asked an off-topic question: '{prompt}'. Politely decline to answer and guide them back to legal topics.").content

                st.session_state.memory.save_context({"input": prompt}, {"output": response_text})
            except Exception as e:
                st.error(f"An error occurred: {e}")
                response_text = "I'm sorry, I encountered an error. Please try again."

            st.write(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})

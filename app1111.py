import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai

# Google API Key for Generative AI
GOOGLE_API_KEY = "AIzaSyB5CKuD8CPYpQ3Yz-kG_MIZ7zjgNFYjJiM"
genai.configure(api_key=GOOGLE_API_KEY)

# Selenium Driver Setup
def setup_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    service = Service(r"C:\Program Files\Google\Chrome\Application\chrome.exe")  # Corrected path
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver

# Function to search Indian Kanoon using Selenium
def selenium_search(query):
    driver = setup_driver()
    try:
        driver.get("https://indiankanoon.org/")
        time.sleep(2)

        search_box = driver.find_element(By.NAME, "formInput")
        search_box.send_keys(query)
        search_box.send_keys(Keys.RETURN)
        time.sleep(3)

        soup = BeautifulSoup(driver.page_source, "html.parser")
        results = []
        for result in soup.find_all('div', class_='result_title'):
            title = result.find('a').get_text()
            link = "https://indiankanoon.org" + result.find('a')['href']
            results.append({'title': title, 'link': link})
        return results
    finally:
        driver.quit()

# The remaining code remains unchanged


# Function to extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into smaller chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunksexe

# Function to create FAISS vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

# Langchain Conversational Chain
def get_conversational_chain():
    prompt_template = """
    You are a legal assistant tasked with extracting specific details from a case document. 
    Answer the question as accurately as possible based on the provided context. 
    If the answer is not available, say, "Answer is not available in the context."
    
    Context: {context}
    Question: {question}
    
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=GOOGLE_API_KEY)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Query PDFs
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question, k=5)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

# Streamlit App
def main():
    st.title("Legal Ease: AI Legal Research Assistant")
    st.markdown("### Empower your legal research with AI and automated search capabilities.")

    pdf_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
    if pdf_files:
        with st.spinner("Processing PDFs..."):
            raw_text = get_pdf_text(pdf_files)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
            st.success("PDFs processed successfully! Ask a question or search for similar cases.")

    option = st.radio("What would you like to do?", ("Ask PDF", "Search Indian Kanoon"))

    question = st.text_input("Enter your query:")

    if option == "Ask PDF":
        if st.button("Get Answer from PDF"):
            if pdf_files and question:
                with st.spinner("Searching your PDFs..."):
                    answer = user_input(question)
                    st.write(f"**Answer:** {answer}")
            else:
                st.error("Please upload a PDF and enter a question.")

    elif option == "Search Indian Kanoon":
        if st.button("Search Indian Kanoon"):
            if question:
                with st.spinner("Searching Indian Kanoon..."):
                    results = selenium_search(question)
                    if results:
                        st.write(f"**Found {len(results)} results:**")
                        for result in results:
                            st.write(f"[{result['title']}]({result['link']})")
                    else:
                        st.error("No results found.")
            else:
                st.error("Please enter a question.")

if __name__ == "__main__":
    main()

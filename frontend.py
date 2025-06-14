import os
import streamlit as st
from google import genai
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Set your API key for Gemini
load_dotenv()
GEMINI_API_KEY = 'AIzaSyC89YzwZQEzKiFCl-8tz8PIxXTUvD8RzQM'

# Initialize the Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)

# ----------- Function to Generate Text Using Gemini ----------- #
def generate_text_from_gemini(query, retrieved_docs):
    """
    Generate a response using Gemini by combining the query and retrieved documents.
    
    Args:
    - query (str): The user's query or question.
    - retrieved_docs (list of str): List of retrieved documents (context) from the database.
    
    Returns:
    - response (str): The generated response based on the query and context.
    """
    
    # Combine the query with the retrieved documents to form a context string
    context = "\n".join(retrieved_docs)  # Join documents into a single context string
    prompt = f"Query: {query}\nContext:\n{context}\nAnswer:"

    try:
        # Generate content using Gemini API
        response = client.models.generate_content(
            model="gemini-2.5-pro-exp-03-25",  # Replace with the appropriate model name if different
            contents=prompt
        )
        
        # Extract the generated text from the response
        generated_text = response.text.strip() if response.text else 'Error: No valid response.'
        return generated_text
    
    except Exception as e:
        print(f"Error generating text: {e}")
        return "Error generating response."

# ----------- Function to Extract Text from PDF Files ----------- #
def get_pdf_text(pdf_docs):
    """Extract text from PDF files"""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""  
    return text

# ----------- Function to Split Text into Chunks ----------- #
def get_text_chunks(text):
    """Split text into chunks for better processing"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=15000, chunk_overlap=100)
    return text_splitter.split_text(text)

# ----------- Function to Load All .txt Files from Outputs/ ----------- #
def load_text_from_file(file_path):
    """Load text content from a file"""
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    return ""

# ----------- Streamlit UI Setup ----------- #
st.set_page_config(page_title="RAG Bot", page_icon="🤖", layout="wide")

# Add custom CSS styling for a cleaner, more modern look
st.markdown("""
    <style>
    .header {
        font-size: 36px;
        font-weight: bold;
        color: #1e88e5;
        text-align: center;
        margin-bottom: 20px;
    }
    .description {
        font-size: 18px;
        color: #616161;
        text-align: center;
        margin-bottom: 40px;
    }
    .file-uploader {
        border: 2px solid #1e88e5;
        padding: 20px;
        border-radius: 10px;
        background-color: #f1f1f1;
    }
    .file-uploader:hover {
        border-color: #0d47a1;
    }
    .response-box {
        background-color: #f9f9f9;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #ddd;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<div class="header">RAG Bot: Ask Questions from PDF Documents</div>', unsafe_allow_html=True)
st.markdown('<div class="description">Upload your PDFs and ask any question related to the documents. The bot will provide you answers based on the content.</div>', unsafe_allow_html=True)

# File uploader for PDFs
pdf_docs = st.file_uploader("Upload Your PDF Files", accept_multiple_files=True, type="pdf", key="pdf_uploader")

# Upload section UI
if pdf_docs:
    with st.spinner("Processing PDFs..."):
        # Extract and process text from PDFs
        extracted_text = get_pdf_text(pdf_docs)
        text_chunks = get_text_chunks(extracted_text)
        
        # Save extracted text to the outputs/ directory
        output_dir = 'outputs/'
        os.makedirs(output_dir, exist_ok=True)
        extracted_file_path = os.path.join(output_dir, "extracted_text.txt")
        with open(extracted_file_path, 'w', encoding='utf-8') as f:
            f.write(extracted_text)
        
        st.success("PDFs processed successfully!")

# Fetch documents for RAG from all extracted_text.txt files in the outputs/ directory
output_dir = 'outputs/'
retrieved_docs = []
for filename in os.listdir(output_dir):
    if filename.endswith("extracted_text.txt"):
        file_path = os.path.join(output_dir, filename)
        retrieved_docs.append(load_text_from_file(file_path))

# User question input
user_query = st.text_input("Ask a question related to the uploaded documents:")

# When the user asks a question
if user_query and retrieved_docs:
    with st.spinner("Generating response..."):
        # Generate response using Gemini
        response = generate_text_from_gemini(user_query, retrieved_docs)
        
        # Display the response in a clean, styled box
        st.markdown('<div class="response-box">', unsafe_allow_html=True)
        st.write("### Answer:")
        st.write(response)
        st.markdown('</div>', unsafe_allow_html=True)

# If no PDFs are uploaded
if not pdf_docs:
    st.warning("Please upload one or more PDF files to get started.")

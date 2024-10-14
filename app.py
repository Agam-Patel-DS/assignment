import streamlit as st
import os
import pymysql
import fitz  # PyMuPDF
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Set up embeddings and LLM configurations
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Define the prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Question: {input}
    Also find the number of occurrences of the searched term, the pages where it occurs in the report, and the count on each page.
    """
)

# Database configuration
def connect_to_database():
    return pymysql.connect(
        host="localhost",
        user="root",  
        password="agam2003",  
        database="document_db"
    )

# Function to save document metadata to the database
def save_document_metadata(name, doc_type, notes, pages, location):
    try:
        connection = connect_to_database()
        cursor = connection.cursor()
        cursor.execute('''
            INSERT INTO documents (name, type, notes, pages, location)
            VALUES (%s, %s, %s, %s, %s)
        ''', (name, doc_type, notes, pages, location))
        connection.commit()
    except Exception as e:
        st.error(f"Failed to save metadata: {e}")
    finally:
        cursor.close()
        connection.close()

# Function to extract text from a multi-page PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as pdf_document:
        for page in pdf_document:
            text += page.get_text() + "\n"  # Extract text from each page
    return text

# Streamlit app layout
st.title("Knowledge-Based Search")
st.write("Demo")

# Additional inputs for document metadata
st.subheader("Document Metadata")
report_type = st.text_input("Type of Report (e.g., Legal, Medical, Insurance)")
report_notes = st.text_area("Additional Notes")
uploaded_file = st.file_uploader("Choose a file", type=["pdf"])
UPLOAD_FOLDER = "files"

if uploaded_file is not None:
    # Save the uploaded file
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Extract text and count pages
    pdf_text = extract_text_from_pdf(file_path)
    pages = pdf_text.count("\n") + 1  # Estimate number of pages based on newlines

    # Save metadata to the database
    save_document_metadata(uploaded_file.name, report_type, report_notes, pages, file_path)

    st.success(f"File '{uploaded_file.name}' has been saved to '{UPLOAD_FOLDER}' and metadata stored.")

# Function to create vector embeddings
def create_vector_embeddings():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        st.session_state.loader = PyPDFDirectoryLoader(UPLOAD_FOLDER)
        st.session_state.docs = st.session_state.loader.load()

        # Verify if documents are loaded correctly
        if not st.session_state.docs:
            st.error("No documents found for embedding. Check document loading.")
            return

        # Split documents into manageable chunks
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)

        # Verify if there are documents to process
        if not st.session_state.final_documents:
            st.error("No document chunks found. Verify document splitting.")
            return

        # Create FAISS vector store from documents
        try:
            st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
            st.success("Vector embeddings created successfully.")
        except Exception as e:
            st.error(f"Error creating vector embeddings: {e}")

# Input for user queries
user_prompt = st.text_input("Enter your query from the research paper!")

# Button to create document embeddings
if st.button("Create Document Embeddings"):
    create_vector_embeddings()

# Handle user query and search
if user_prompt and "vectors" in st.session_state:
    # Set up retrieval and response chain
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retriever_chain = create_retrieval_chain(retriever, document_chain)

    # Measure response time
    start = time.process_time()
    try:
        response = retriever_chain.invoke({"input": user_prompt})
        st.write(f"Response time = {time.process_time() - start} seconds")

        # Display the response
        st.subheader("Answer")
        st.write(response["answer"])

        # Show similar document content
        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response["context"]):
                st.write(f"Document {i+1}:")
                st.write(doc.page_content)
                st.write("--------------------------")
    except Exception as e:
        st.error(f"Error processing the query: {e}")
else:
    if user_prompt:
        st.warning("Please create document embeddings first.")

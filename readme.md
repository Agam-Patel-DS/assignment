
# Knowledge-Based Search System

This is a Streamlit application designed for searching through PDF documents using natural language queries. The application utilizes LangChain, Hugging Face embeddings, and a Groq model to answer questions based on the content of the uploaded PDFs. Metadata about the documents is saved in a MySQL database for easy retrieval.

## Features

- Upload PDF documents.
- Extract text from multi-page PDFs.
- Save document metadata (name, type, notes, pages, location) in a MySQL database.
- Search through the documents using natural language queries.
- Return answers along with relevant context and occurrences of search terms.

## Requirements

- Python 3.7+
- MySQL Server

### Python Libraries

- Streamlit
- PyMuPDF (or pdfplumber)
- pymysql
- langchain
- langchain-community
- langchain-huggingface
- langchain-groq
- dotenv

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Agam-Patel-DS/assignment.git
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up the MySQL database:**
   - Create a MySQL database and table:
   ```sql
   CREATE DATABASE document_db;

   USE document_db;

   CREATE TABLE documents (
       id INT AUTO_INCREMENT PRIMARY KEY,
       name VARCHAR(255) NOT NULL,
       type VARCHAR(50),
       notes TEXT,
       pages INT,
       location VARCHAR(255) NOT NULL
   );
   ```

5. **Create a `.env` file** in the project root with your database credentials:
   ```plaintext
   GROQ_API_KEY=your_groq_api_key
   ```

6. **Replace the database connection details** in `app.py`:
   ```python
   # Replace with your MySQL username and password
   user="your_username"
   password="your_password"
   ```

## Usage

1. **Run the application:**
   ```bash
   streamlit run app.py
   ```

2. **Access the application** in your web browser at `http://localhost:8501`.

3. **Upload a PDF file** and enter the metadata:
   - Type of Report
   - Additional Notes

4. **Create Document Embeddings** and enter a query to search through the uploaded documents.

5. **View the results**, which include answers to your queries along with context and occurrences.

## Acknowledgments

- [LangChain](https://langchain.com/)
- [Hugging Face](https://huggingface.co/)
- [Streamlit](https://streamlit.io/)
- [PyMuPDF](https://pymupdf.readthedocs.io/en/latest/)
- [pymysql](https://pymysql.readthedocs.io/en/latest/)


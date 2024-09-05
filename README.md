# Chat PDF with Gemini 💁

An interactive Streamlit application that allows users to query and interact with PDF files using Google Generative AI (Gemini). The application extracts text from uploaded PDFs, processes the text into searchable chunks, and uses a conversational chain to provide accurate answers based on the content of the documents.

## Features

- **PDF Text Extraction**: Upload PDF files and extract their text content.
- **Text Chunking**: The extracted text is split into manageable chunks for efficient processing.
- **Vector Store Creation**: A FAISS-based vector store is created from the text chunks, enabling fast similarity searches.
- **Conversational Chain**: Utilizes Google Generative AI to answer questions based on the content of the PDFs.

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/chat-pdf-with-gemini.git
    cd chat-pdf-with-gemini
    ```

2. **Create and activate a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Set up environment variables**:
    - Create a `.env` file in the project root directory.
    - Add your Google API key:
    ```plaintext
    GOOGLE_API_KEY=your-google-api-key
    ```

## Usage

1. **Run the Streamlit app**:
    ```bash
    streamlit run app.py
    ```

2. **Upload PDF Files**: Use the sidebar to upload PDF documents.

3. **Ask Questions**: Type your question in the input field and receive answers based on the content of the uploaded PDFs.

## Project Structure

- `app.py`: Main Streamlit application file.
- `get_pdf_text()`: Function to extract text from PDF files.
- `get_text_chunks()`: Function to split extracted text into chunks.
- `get_vector_store()`: Function to create a FAISS vector store from text chunks.
- `get_conversational_chain()`: Function to set up the conversational chain with Google Generative AI.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

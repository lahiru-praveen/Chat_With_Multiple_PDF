# Chat_With_Multiple_PDF
Chat PDF with Gemini 💁 :
This project leverages Streamlit and Google Generative AI to create an interactive chat interface that allows users to ask questions directly from PDF files. It integrates several key components:

PDF Text Extraction: Extracts text from uploaded PDF files using PyPDF2.
Text Chunking: Splits the extracted text into manageable chunks for efficient processing.
Vector Store Creation: Utilizes FAISS to create a vector store of the text chunks, enabling quick similarity searches.
Conversational Chain: Implements a question-answering chain using Gemini's ChatGoogleGenerativeAI model to provide detailed and accurate responses based on the content of the PDFs.
This application is ideal for users who need to interact with and query large PDF documents easily.

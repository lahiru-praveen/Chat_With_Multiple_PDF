import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)


def get_pdf_text(pdf_docs):
    try:
        text = ""
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF files: {e}")
        return ""


def get_text_chunks(text):
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        chunks = text_splitter.split_text(text)
        return chunks
    except Exception as e:
        st.error(f"Error splitting text: {e}")
        return []


def get_vector_store(text_chunks):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
    except Exception as e:
        st.error(f"Error creating vector store: {e}")


def get_conversational_chain():
    try:
        prompt_template = """
        Provide a detailed answer based on the given context. If the answer is not present in the context, respond with "The information is not available in the provided context." Ensure accuracy and completeness in your answer.

        Context:
        {context}

        Question:
        {question}

        Answer:
        """

        model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

        return chain
    except Exception as e:
        st.error(f"Error loading QA chain: {e}")
        return None


def user_input(user_question):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        # Load FAISS index with allow_dangerous_deserialization=True
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain()

        if chain:
            response = chain(
                {"input_documents": docs, "question": user_question},
                return_only_outputs=True
            )

            st.write("### Response:")
            st.write(response["output_text"], unsafe_allow_html=True)
        else:
            st.error("Error: Unable to create a conversational chain.")
    except Exception as e:
        st.error(f"Error processing user input: {e}")


def main():
    st.set_page_config(page_title="Chat with PDF", page_icon=":book:", layout="wide")

    # Custom CSS for styling
    st.markdown("""
        <style>
        .css-18e3th9 {
            background-color: #f0f2f6;
        }
        .css-1v0mbdj {
            padding: 20px;
        }
        .css-1d391kg {
            border-radius: 10px;
            background-color: #ffffff;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        }
        .stButton>button {
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 5px;
            padding: 12px 24px;
            font-size: 16px;
        }
        .stButton>button:hover {
            background-color: #0056b3;
        }
        .stTextInput>div>input {
            border-radius: 5px;
            border: 1px solid #ced4da;
            padding: 12px;
            font-size: 16px;
        }
        .stFileUploader>div>input {
            border-radius: 5px;
            border: 1px solid #ced4da;
            padding: 12px;
        }
        .stApp {
            background-image: linear-gradient(to bottom right, #e0f7fa, #b9fbc0);
        }
        .stMarkdown {
            color: #333;
        }
        .stSpinner {
            color: #007bff;
        }
        </style>
    """, unsafe_allow_html=True)

    st.header("📚 Chat with PDF using Gemini")

    col1, col2 = st.columns([2, 1])

    with col1:
        user_question = st.text_input("Ask a Question Based on the PDF Content:")

    if user_question:
        user_input(user_question)

    with col2:
        st.title("📂 Menu")
        pdf_docs = st.file_uploader("Upload PDF Files Here (multiple allowed):", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                if raw_text:
                    text_chunks = get_text_chunks(raw_text)
                    if text_chunks:
                        get_vector_store(text_chunks)
                        st.success("Processing Complete!")
                    else:
                        st.error("Error: No text chunks were created.")
                else:
                    st.error("Error: No text extracted from PDF.")


if __name__ == "__main__":
    main()

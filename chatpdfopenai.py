import os
import PyPDF2
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

# ----------------------------------------------
# 🔑 Load API Key
# ----------------------------------------------
load_dotenv()
openai_api_key = "YOUR API KEY"
if not openai_api_key:
    raise ValueError("❌ OPENAI_API_KEY is not set. Please check your .env file or set the environment variable manually.")

# ----------------------------------------------
# 📚 PDF Text Extraction
# ----------------------------------------------
def get_pdf_text(pdf_docs):
    """
    Extract text from a list of PDF documents.
    """
    text = ""
    for pdf_file in pdf_docs:
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            for page in pdf_reader.pages:
                text += page.extract_text() or ''
        except Exception as e:
            st.error(f"Failed to read {pdf_file.name}: {e}")
    return text


# ----------------------------------------------
# ✂️ Text Chunking
# ----------------------------------------------
def get_text_chunks(text):
    """
    Split text into manageable chunks for embedding.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_text(text)


# ----------------------------------------------
# 📊 Vector Store (FAISS)
# ----------------------------------------------
def get_vector_store(text_chunks):
    """
    Create a FAISS vector store from text chunks.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=openai_api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


# ----------------------------------------------
# 🤖 Conversational Chain
# ----------------------------------------------
def get_conversational_chain():
    """
    Build a conversational chain for answering questions.
    """
    prompt_template = """
    Answer the question as detailed as possible based on the provided context. 
    If the answer is not available in the context, respond with:
    "The answer is not available in the context."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, openai_api_key=openai_api_key)
    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
    return chain


# ----------------------------------------------
# 🙋 User Interaction
# ----------------------------------------------
def user_input(user_question):
    """
    Handle user question input and return the response.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=openai_api_key)
    vector_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = vector_db.similarity_search(user_question)

    chain = get_conversational_chain()
    response =chain({"input_documents": docs, "question": user_question})
    
    st.write("### 🤖 Reply:")
    st.write(response['output_text'])


# ----------------------------------------------
# 🖥️ Streamlit Interface
# ----------------------------------------------
def main():
    st.set_page_config(page_title="Chat With Multiple PDFs", layout="wide")
    st.header("📄 Chat with Multiple PDFs Using OpenAI")

    # User question interface
    user_question = st.text_input("Ask a question about your PDFs:")
    if user_question:
        user_input(user_question)

    # Sidebar for PDF upload and processing
    with st.sidebar:
        st.title("📂 Menu")
        pdf_docs = st.file_uploader("Upload your PDF files", type=["pdf"], accept_multiple_files=True)
        
        if st.button("🚀 Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing PDFs..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("✅ Processing complete! You can now ask questions.")
            else:
                st.error("⚠️ Please upload at least one PDF file.")


# ----------------------------------------------
# 🚦 Run the App
# ----------------------------------------------
if __name__ == "__main__":
    main()

It is chat with multiple PDF using open AI and Streamlit.
It is useful for more complicated datas to review and for easy retrival and easy to find the answer within a limited time.

step 1 is loading: Loads the OpenAI API key from an .env file using python-dotenv.
Step 2 is extracting: get_pdf_text extracts text from the uploaded PDF files using PyPDF2.
Step 3 is chunking: text is split into smaller chunks using RecursiveCharacterTextSplitter to allow efficient embedding and querying. The chunk size is set to 1000 characters with an overlap of 200 characters
Step 4 is vector store: text chunks are converted into vectors using OpenAIEmbeddings, and the vectors are stored in a FAISS index to support fast similarity search.
Step 5 is conversational chain: custom prompt template and chain are defined to answer questions based on the context provided by the vector store.
Step 6 is user interaction: takes user input for a question, uses FAISS for similarity search to find relevant documents, and the ChatOpenAI model generates an answer.
Step 7 is streamlit UI: provides a simple interface for the user to upload PDF files and ask questions. It also processes the PDFs into chunks and stores them in a vector database.

Note: 
1)Use your OPENAPI KEY
2)Upload upto 200MB per file

code sets up a PDF question-answering system using Streamlit and OpenAI's language model. It allows users to upload PDFs, extract text, split it into chunks, store them in a FAISS vector database, and then answer questions based on the content.
Improvements include better error handling, caching the vector store, refining chunking strategies, improving UI for PDF uploads,
and making the conversational flow smoother for multiple questions

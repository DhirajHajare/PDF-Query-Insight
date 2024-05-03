import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_pdf_text(pdf_docs):
    """
    Extracts text from uploaded PDF files.

    Args:
        pdf_docs (list): A list of uploaded PDF documents.

    Returns:
        str: The extracted text from all PDF files combined.
    """
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    """
    Splits the extracted text into smaller chunks.

    Args:
        text (str): The extracted text from PDF files.

    Returns:
        list: A list of text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    """
    Creates a FAISS vector store using Google Generative AI embeddings.

    Args:
        text_chunks (list): A list of text chunks.

    Returns:
        FAISS: The created FAISS vector store.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store


def get_conversational_chain():
    """
    Defines a question-answering chain using a pre-trained model.

    Returns:
        langchain.chains.question_answering.QuestionAnsweringChain:
            The question-answering chain object.
    """
    prompt_template1 = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not
    available in the context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt = PromptTemplate(template=prompt_template1, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def user_input(user_question):
    """
    Processes user question and provides a response using the question-answering chain.

    Args:
        user_question (str): The user-provided question.

    Returns:
        None
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True,
    )

    st.write("Response:\n\n\n\n", response["output_text"])


def get_gemini_response(input_text):
    """
    Generates key points from the text using a Gemini-Pro model.

    Args:
        input_text (str): The text to be analyzed.

    Returns:
        str: The generated key points.
    """
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(input_text)
    return response.text


def main():
    """
    Main function for the Streamlit application.
    """
    st.set_page_config("Query With PDF")
    st.header("Query With Your Uploaded PDFs")
    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Query With PDFs:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")
        if st.button("Get KeyPoints", type="primary"):
            st.empty()
            input_prompt = """ fetch all  key points in details from given text information ,  do not give extra other information . only give information relevant to text information
            """
            space = "                        "
            raw_text = get_pdf_text(pdf_docs)
            response = get_gemini_response(raw_text + " as text information " + space + input_prompt)
            st.subheader(response)

if __name__ == "__main__":
    main()

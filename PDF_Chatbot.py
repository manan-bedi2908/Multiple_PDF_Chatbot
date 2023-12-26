import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import os

embedding_model_name = os.environ.get('EMBEDDING_MODEL_NAME')

def get_pdf_text(pdf_docs):
    text = ''
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(separator='\n',
                                          chunk_size = 1000,
                                          chunk_overlap = 200,
                                          length_function = len
                                          )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name = embedding_model_name)
    vectorstore = FAISS.from_texts(texts = text_chunks, embedding = embeddings)
    return vectorstore

def main():
    load_dotenv()
    st.set_page_config('Chat with Multiple PDFs', page_icon = ':books:')
    st.header('Chat with Multiple PDF :books:')
    with st.sidebar:
        st.header('Chat with PDF')
        st.title('LLM ChatApp using LangChain')
        st.subheader('Your Documents')
        pdf_docs = st.file_uploader('Upload the PDF Files here and click on Process',
                                    accept_multiple_files = True)
        st.markdown('''
        - [Streamlit](https://streamlit.io/)
        - [LangChain](https://python.langchain.com/)
        - [OpenAI](https://platform.openai.com/docs/models) LLM Model
        ''')

        if st.button('Process'):
            with st.spinner('Processing'):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vector_store(text_chunks)
                st.success('Done')

            

if __name__ == '__main__':
    main()


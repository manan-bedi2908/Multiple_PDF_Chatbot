import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub 
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import os
from htmlCode import css, bot_template, user_template

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

def get_conversation_chain(vectorstore):
    llm = HuggingFaceHub(repo_id = 'google/flan-t5-xxl', 
                         model_kwargs = {'temperature': 0.5,
                                         'max_length': 512})
    memory = ConversationBufferMemory(memory_key = 'chat_history',
                                      return_messages = True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm = llm,
                                                               retriever = vectorstore.as_retriever(),
                                                               memory = memory)
    return conversation_chain

def handle_user_input(user_question):
    response=st.session_state.conversation({'question':user_question})
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2==0:
            #st.write(message)
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            #st.write(message)
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config('Chat with Multiple PDFs', page_icon = ':books:')
    st.write(css, unsafe_allow_html=True)
    st.header('Chat with Multiple PDF :books:')

    if 'conversation' not in st.session_state:
        st.session_state.conversation = None
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = None
    
    user_question = st.text_input("Ask a question from your documents")
    if user_question:
        handle_user_input(user_question)
    
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
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.success('Done')


            

if __name__ == '__main__':
    main()


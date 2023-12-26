import streamlit as st

def main():
    st.set_page_config('Chat with Multiple PDFs', page_icon = ':books:')
    st.header('Chat with Multiple PDF :books:')
    with st.sidebar:
        st.header('Chat with PDF')
        st.title('LLM ChatApp using LangChain')
        st.subheader('Your Documents')
        pdf_docs = st.file_uploader('Upload the PDF Files here and click on Process',
                                    accept_multiple_files = True)
        st.button('Process')
        st.markdown('''
        - [Streamlit](https://streamlit.io/)
        - [LangChain](https://python.langchain.com/)
        - [OpenAI](https://platform.openai.com/docs/models) LLM Model
        ''')

if __name__ == '__main__':
    main()


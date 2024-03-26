import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.write(css, unsafe_allow_html=True)
    st.write(
        """
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
            <div>
                <img src="https://wieck-nissanao-production.s3.amazonaws.com/photos/87dd7b734e373e65761492bd446b20efd0a56737/preview-928x522.jpg" style="width: 150px;">
            </div>
            <div style="text-align: right;">
                <h1>Nissan AI</h1>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if 'num_inputs' not in st.session_state:
        st.session_state['num_inputs'] = 1

    def add_input():
        all_filled = all(st.session_state.get(f"url_{i}", "") != "" for i in range(st.session_state['num_inputs']))
        if not all_filled:
            st.warning("Por favor, rellena todas las URLs existentes antes de agregar una nueva.")
        else:
            st.session_state['num_inputs'] += 1

    if "processed" not in st.session_state:
        st.session_state.processed = False

    with st.sidebar:
        st.subheader("Documentos")
        pdf_docs = st.file_uploader("Cargue sus archivos PDF aquí", accept_multiple_files=True)
        if st.button("Procesar"):
            with st.spinner("Procesando"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)

        st.subheader("Fuentes URL")

        for i in range(st.session_state['num_inputs']):
            st.text_input(f"URL {i+1}", key=f"url_{i}", placeholder="https://www.example.com")

        st.button("➕ Agregar otra URL", on_click=add_input)


    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if st.session_state.processed:
        user_question = st.text_input("Pregúntame sobre tus documentos:")
        if user_question:
            handle_userinput(user_question)
    else:
        st.text_input("Pregúntame sobre tus documentos:", disabled=True)

if __name__ == '__main__':
    main()
import os
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback


def main():
    load_dotenv()
    st.set_page_config(page_title="Langchain pdf")

    # Image
    image_path = os.path.join("assets", "nissan_logo.svg")
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
        """.format(image_path),
        unsafe_allow_html=True
    )

    #upload file
    pdf = st.file_uploader("Subir PDF", type="pdf")

    #extract the next
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        # create embeddings
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        #sow user input
        user_question = st.text_input("Haz una pregunta sobre tu PDF:")
        if user_question:
            docs = knowledge_base.similarity_search(user_question)

            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=user_question)
                print(cb)

            st.write(response)


if __name__ == '__main__':
    main()
import os
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
import langchain
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import tiktoken
import faiss
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
def main():
    load_dotenv()
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF 💭")

    pdf = st.file_uploader("Upload your PDF",type="pdf")
    if pdf is not None :
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        text_splitter = CharacterTextSplitter(
            separator = "\n",
            chunk_size = 1000,
            chunk_overlap = 200,
            length_function = len
        )
        chunks = text_splitter.split_text(text)
        embeddings = OpenAIEmbeddings(openai_api_key="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")
        knowledge_base = FAISS.from_texts(chunks, embeddings) 
        user_input = st.text_input("Ask a question about your PDF:")
        if user_input:
            docs = knowledge_base.similarity_search(user_input)
            llm = OpenAI(openai_api_key="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")
            chains = load_qa_chain(llm,chain_type="stuff")
            response = chains.run(input_documents = docs, question = user_input)
            st.write(response)
if __name__ == '__main__':
    main()
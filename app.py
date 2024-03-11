import streamlit as st
import pickle
import os
import time
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS 
from langchain.memory import ConversationBufferMemory 
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain.document_loaders import UnstructuredURLLoader 
load_dotenv() 
def main():
    
    st.title("InsightInvestor")
    st.write("Your personal Finance-News-Investment Research Analyst 📈")
    st.sidebar.title("Article URLs")

    num_urls = st.sidebar.number_input("Number of URLs", min_value=1, value=3)
    urls = []

    for i in range(num_urls):
        url = st.sidebar.text_input(f"URL {i+1}")
        urls.append(url)

    process_url_clicked = st.sidebar.button("Process URLs")
    file_path = "FAI_Similarity_Search_Space.pkl"

    if process_url_clicked:
        # Load data from URLs
        loader = UnstructuredURLLoader(urls=urls)
        st.text("Data Loading In Progress")
        data = loader.load()

        # Split data
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=1000,
            chunk_overlap=0
        )
        st.text("Text splitting started")
        docs = text_splitter.split_documents(data)

        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(docs, embeddings)
        st.text("Building Embedded Vector Space")
        time.sleep(2)

        with open(file_path, "wb") as f:
            pickle.dump(vectorstore, f)

    query = st.text_input("Question:")

    if query:
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                vectorstore = pickle.load(f)
                llm = ChatOpenAI()  
                chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
                result = chain({"question": query}, return_only_outputs=True)

                st.header("Answer:")
                st.write(result["answer"])

                # Display sources, if available
                sources = result.get("sources", "")
                if sources:
                    st.subheader("Sources:")
                    sources_list = sources.split("\n")
                    for source in sources_list:
                        st.write(source)

if __name__ == '__main__':
    main()

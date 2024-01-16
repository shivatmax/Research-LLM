# import chunk
import os
from unittest import loader
from bardapi import max_token
import streamlit as st
import time
import langchain
from langchain.llms import GooglePalm
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import GooglePalmEmbeddings
from langchain.vectorstores import FAISS
from torch import embedding
import dill

GOOGLE_API_KEY='Your api key'

google_api_key = os.getenv("GOOGLE_API_KEY", GOOGLE_API_KEY)
if google_api_key is None:
    raise ValueError("GOOGLE_API_KEY environment variable is not set")

llm = GooglePalm(google_api_key=google_api_key, temperature=0.8, max_tokens=500)

loader = UnstructuredURLLoader(urls=["https://www.moneycontrol.com/ipo/","https://www.moneycontrol.com/ipo/listed-ipos/"])
data = loader.load()
print(len(data))


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

docs = text_splitter.split_documents(data)
print(len(docs))

embedding = GooglePalmEmbeddings(google_api_key=google_api_key)

vectorindex_palm = FAISS.from_documents(docs, embedding)

#print(vectorindex_palm) #<langchain.vectorstores.faiss.FAISS object at 0x00000255D746A790>

file_path = 'vectorindex_palm.pkl'
# with open(file_path, 'wb') as f:
#     dill.dump(vectorindex_palm, f)

if os.path.exists(file_path):
    with open(file_path, 'rb') as f:
        vectorindex = dill.load(f)

chain = RetrievalQAWithSourcesChain.from_llm(
    retriever=vectorindex.as_retriever(),
    llm=llm,
)

# print(chain)

query ="tell me best upcoming IPO's that i still can apply with there price and news?"

langchain.debug = True

chain({"question":query})


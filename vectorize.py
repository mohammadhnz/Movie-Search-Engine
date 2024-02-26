from langchain.vectorstores.faiss import FAISS
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.utils import DistanceStrategy

from langchain_community.embeddings import HuggingFaceEmbeddings

import dotenv
import os
import pickle

if __name__ == "__main__":
    dotenv.load_dotenv()

    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")

    loader = CSVLoader("datasets/movies.csv", source_column="Title", encoding = 'UTF-8')

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=100,
        length_function=len,
    )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        multi_process=True,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )

    documents = loader.load_and_split(splitter)

    vectorstore = FAISS.from_documents(documents, embeddings, distance_strategy=DistanceStrategy.COSINE)

    with open("datasets/vectorstore.pkl", "wb") as f:
        pickle.dump(vectorstore, f)
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Milvus
from HTMLLoader import html_splitter
import os
from dotenv import load_dotenv

load_dotenv()

def dbgenerator():
    urls = ["https://help.tallysolutions.com/tally-prime/accounting/interest-calculation-tally/"]
    splits = html_splitter(urls)

    model_name = "BAAI/bge-base-en-v1.5"
    encode_kwargs = {"normalize_embeddings": False}
    embedding_function = HuggingFaceBgeEmbeddings(model_name=model_name, encode_kwargs=encode_kwargs)

    # print("hi")
    vectorstore = Milvus.from_documents(splits,
                                        embedding_function,
                                        connection_args={"host": os.environ["MILVUS_IP"], "port": "19530"},
                                        collection_name = "html_embeddings", ## custom collection name 
                                        search_params = {"metric":"IP","offset":0}, ## search params
                                        )
    print("Done")
    return vectorstore

x = dbgenerator()


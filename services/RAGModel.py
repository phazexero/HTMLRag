from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_together import Together
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Milvus
import os
from dotenv import load_dotenv

load_dotenv()

def db_retriever():
    model_name = "BAAI/bge-base-en-v1.5"
    encode_kwargs = {"normalize_embeddings": False}
    embeddings = HuggingFaceBgeEmbeddings(model_name=model_name, encode_kwargs=encode_kwargs)
    vector_db = Milvus(embeddings,
                        connection_args={"host": os.environ["MILVUS_IP"], "port": "19530"},
                        collection_name="html_embeddings",
                        )
    return vector_db

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

prompt = PromptTemplate(input_variables=['context', 'question'],
                        template="You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Keep answers descriptive and mention the process.\nQuestion: {question} \nContext: {context} \nAnswer:")

def rag_call(query):
    vectorstore = db_retriever()
    retriever = vectorstore.as_retriever(search_kwargs={"k":2})
    response = Together(
            model="mistralai/Mistral-7B-Instruct-v0.2",
            together_api_key= os.environ["TOGETHER_API_KEY"],
            temperature=0.3,
            max_tokens=512
        )

    rag_chain = (
        {"context": retriever 
        | format_docs, "question": RunnablePassthrough()}
        | prompt
        | response
        | StrOutputParser())
    
    answer = rag_chain.invoke(query)
    return answer

query = input()
# query = "What are the steps to do interest calculation transaction by transaction?"
ls = rag_call(query)
print(ls)
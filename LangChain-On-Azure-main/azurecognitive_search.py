import os
#from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores.azuresearch import AzureSearch
#from langchain.document_loaders import AzureBlobStorageContainerLoader
from langchain_community.document_loaders import AzureBlobStorageContainerLoader
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()
print('AZURE_COGNITIVE_SEARCH_SERVICE_NAME=',os.environ.get('AZURE_COGNITIVE_SEARCH_SERVICE_NAME'))
print('AZURE_COGNITIVE_SEARCH_API_KEY=',os.environ.get("AZURE_COGNITIVE_SEARCH_API_KEY"))
print('AZURE_CONN_STRING=',os.environ.get("AZURE_CONN_STRING"))

model: str = "text-embedding-ada-002"
print("Before calling vector store")
#vector_store_address: str = f"https://${os.environ.get('AZURE_COGNITIVE_SEARCH_SERVICE_NAME')}.search.windows.net"
vector_store_address: str = "https://cognitivesearch10042024.search.windows.net"

embeddings: OpenAIEmbeddings = OpenAIEmbeddings(deployment=model, chunk_size=1)
index_name: str = "langchain-vector-demo"
vector_store: AzureSearch = AzureSearch(
    azure_search_endpoint=vector_store_address,
    azure_search_key=os.environ.get("AZURE_COGNITIVE_SEARCH_API_KEY"),
    index_name=index_name,
    embedding_function=embeddings.embed_query,
)

loader = AzureBlobStorageContainerLoader(
    conn_str=os.environ.get("AZURE_CONN_STRING"),
    container=os.environ.get("CONTAINER_NAME"),
)
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=150, chunk_overlap=20)
docs = text_splitter.split_documents(documents)
vector_store.add_documents(documents=docs)

print("Data loaded into vectorstore successfully")

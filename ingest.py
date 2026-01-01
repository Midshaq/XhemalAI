import os
from dotenv import load_dotenv
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

#Import the Google GenAI connectors
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

load_dotenv()

# 1. Configure Gemini 2.0 Flash. Change to 3.0 later. 
Settings.llm = GoogleGenAI(
    model="models/gemini-2.0-flash", 
    api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0  # Keep it 0 for accurate whitepaper data extraction
)

#Configure Embeddings (Native Google)
Settings.embed_model = GoogleGenAIEmbedding(
    model_name="models/text-embedding-004",
    api_key=os.getenv("GOOGLE_API_KEY")
)

#Parse PDFs
print("Parsing documents...")
parser = LlamaParse(result_type="markdown")
file_extractor = {".pdf": parser}
documents = SimpleDirectoryReader("./data", file_extractor=file_extractor).load_data()

#Connect to Qdrant Cloud
print("Connecting to Qdrant...")
client = QdrantClient(
    url=os.getenv("QDRANT_URL"), 
    api_key=os.getenv("QDRANT_API_KEY")
)
vector_store = QdrantVectorStore(client=client, collection_name="xhemal_knowledge")
storage_context = StorageContext.from_defaults(vector_store=vector_store)

#Index and Upload
print("Indexing data (putting the fries in the bag)...")
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

print("Success! Xhemal's brain is updated on the Cloud.")

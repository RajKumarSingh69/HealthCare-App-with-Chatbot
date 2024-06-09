import pypdf
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# Extract data from the PDF
def load_pdf(data):
    try:
        loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
        doc = loader.load()
        return doc
    except Exception as e:
        print("Error loading PDF:", e)
        return None

# Convert the corpus into chunks
# Create text chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks 

# Download embedding model
def loading_embedding_model():
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    return model

# Create a custom embedding class
class CustomEmbeddingFunction:
    def __init__(self, text_to_embedding, model):
        self.text_to_embedding = text_to_embedding
        self.model = model

    def embed_documents(self, texts):
        return [self.model.encode(text).tolist() for text in texts]

    def embed_query(self, query):
        return self.model.encode(query).tolist()


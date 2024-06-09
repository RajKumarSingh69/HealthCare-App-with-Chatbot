import chromadb
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from src.helper import load_pdf,text_split,loading_embedding_model
from src.helper import CustomEmbeddingFunction
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

extracted_data=load_pdf("datasets/")
print(len(extracted_data))

#creating text chunks
text_chunks = text_split(extracted_data)
print(len(text_chunks))



#loading the model
model =loading_embedding_model()

# Creating embeddings for each of the text chunks using the model
embedding_vectors = []
for i, t in enumerate(text_chunks):
    embedding = model.encode(t.page_content)
    embedding_vector = {'id': str(i), 'values': embedding.tolist()}

    # Include text content as metadata
    embedding_vector['metadata'] = {'text': t.page_content}

    embedding_vectors.append(embedding_vector)
print(len(embedding_vectors))

# extracing the text and meta data from embeddings
texts = [vector['metadata']['text'] for vector in embedding_vectors]
metadatas = [vector['metadata'] for vector in embedding_vectors]

# Create a mapping from texts to their corresponding embeddings
text_to_embedding = {vector['metadata']['text']: vector['values'] for vector in embedding_vectors}

# Create an instance of the custom embedding class
embedding_function = CustomEmbeddingFunction(text_to_embedding, model)

# Create Chroma vector store instance
db = Chroma()

# Create Chroma vector store from texts
db = Chroma.from_texts(
    texts=texts,
    embedding=embedding_function,
    metadatas=metadatas,
    persist_directory='ChromaDB'
)
print("embeddings created and stored in Vector db")
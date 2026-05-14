import os 
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def load_documents(directory_path="docs"):
    print(f"Loading documents from {directory_path}")
    if not os.path.exists(directory_path):
        raise ValueError(f"Directory {directory_path} does not exist")
    loader = DirectoryLoader(
        path=directory_path,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )

    documents = loader.load()

    if len(documents) == 0:
        raise ValueError(f"No documents found in {directory_path}")

    for i, doc in enumerate(documents[:2]):
        print(f"\nDocument {i+1}:")
        print(f"  Source: {doc.metadata['source']}")
        print(f"  Content length: {len(doc.page_content)} characters")
        print(f"  Content preview: {doc.page_content[:100]}...")
        print(f"  metadata: {doc.metadata}")
        
    return documents

def splitDocs(documents,chunk_size = 200, chunk_overlap = 20):
    """
    Split documents into chunks with overlap
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)

    for i, chunk in enumerate(chunks[:2]):
        print(f"\nChunk {i+1}:")
        print(f"  Source: {chunk.metadata['source']}")
        print(f"  Content length: {len(chunk.page_content)} characters")
        print(f"  Content preview: {chunk.page_content[:100]}...")
        print(f"  metadata: {chunk.metadata}")
    return chunks

def storageVectorDB(chunks,toStoreAt='db/chroma_db'):
    """
    Create and Store chunks in vector database
    """
    print("-----Create and Store chunks in vector database-----")

    embedding_model = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434"
    )

    print(f"Creating vector database at {toStoreAt}")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=toStoreAt,
        collection_metadata={"hnsw:space":"cosine"}
    )

    print("-----Finished creating vector database-----")
    return vector_db

def main():
    print("Hello World")
    documents = load_documents(directory_path='docs')
    chunks = splitDocs(documents)
    vector_db = storageVectorDB(chunks, toStoreAt='db/chroma_db')

if __name__ == '__main__':
    main()


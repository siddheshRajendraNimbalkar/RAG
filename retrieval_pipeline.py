from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from dotenv import load_dotenv

load_dotenv()

embedding_model = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434"
)

db = Chroma(
    persist_directory="db/chroma_db", 
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space":"cosine"}
    )

query = "How much did BMW pay to license the Rolls-Royce name and logo?"
retriever = db.as_retriever(
    Search_Kwargs={
        "k":3,
        "score_threshold": 0.8,
        }
)

retriever_docs = retriever.invoke(query)

print("-------------------")

for i, doc in enumerate(retriever_docs,1):
    print(f"Document {i}:\n{doc.page_content}\n")

context = "\n\n".join([doc.page_content for doc in retriever_docs])

llm = ChatOllama(
    model="llama3.2:1b",
    base_url="http://localhost:11434",
    temperature=0
)

messages = [
    SystemMessage(
        content="""
You are a helpful AI assistant.
Answer the question only from the provided context.
If the answer is not present, say:
'I could not find the answer in the context.'
"""
    ),
    HumanMessage(
        content=f"""
Context:
{context}

Question:
{query}
"""
    )
]

response = llm.invoke(messages)

print("\n===== FINAL ANSWER =====\n")
print(response.content)
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_classic.chains import (
    create_history_aware_retriever,
    create_retrieval_chain
)

from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain
)

# ---------------- EMBEDDING MODEL ---------------- #

embedding_model = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434"
)

# ---------------- VECTOR DB ---------------- #

db = Chroma(
    persist_directory="db/chroma_db",
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 3,
        "score_threshold": 0.7
    }
)

# ---------------- LLM ---------------- #

llm = ChatOllama(
    model="llama3.2:1b",
    base_url="http://localhost:11434",
    temperature=0
)

# ---------------- HISTORY PROMPT ---------------- #

history_prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    (
        "human",
        """
Given the chat history and latest user question,
create a standalone question that can be understood
without previous chat history.
"""
    )
])

# ---------------- HISTORY AWARE RETRIEVER ---------------- #

history_aware_retriever = create_history_aware_retriever(
    llm=llm,
    retriever=retriever,
    prompt=history_prompt
)

# ---------------- QA PROMPT ---------------- #

qa_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a helpful AI assistant.

Answer the question only from the provided context.

If answer is not found in context, say:
'I could not find the answer in the context.'

Context:
{context}
"""
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

# ---------------- DOCUMENT CHAIN ---------------- #

document_chain = create_stuff_documents_chain(
    llm,
    qa_prompt
)

# ---------------- FINAL RAG CHAIN ---------------- #

rag_chain = create_retrieval_chain(
    history_aware_retriever,
    document_chain
)

# ---------------- CHAT HISTORY ---------------- #

chat_history = []

print("\n===== Conversational RAG Started =====")
print("Type 'exit' to stop.\n")

# ---------------- CHAT LOOP ---------------- #

while True:

    user_question = input("You: ")

    if user_question.lower() == "exit":
        chat_history = []
        print("\nGoodbye!")
        break

    response = rag_chain.invoke({
        "chat_history": chat_history,
        "input": user_question
    })

    answer = response["answer"]

    print(f"\nAI: {answer}\n")

    # Store conversation history

    chat_history.extend([
        HumanMessage(content=user_question),
        AIMessage(content=answer)
    ])
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# EMBEDDING MODEL
# ─────────────────────────────────────────────
embedding_model = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434"
)

# ─────────────────────────────────────────────
# VECTOR DB
# ─────────────────────────────────────────────
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

# ─────────────────────────────────────────────
# MAIN RAG LLM  (English answer generation)
# ─────────────────────────────────────────────
rag_llm = ChatOllama(
    model="llama3.2:1b",
    base_url="http://localhost:11434",
    temperature=0
)

# ─────────────────────────────────────────────
# TRANSLATOR LLM 1 — Hindi & Hinglish
# Model : llama3.2:3b
# Pull  : ollama pull llama3.2:3b
# ─────────────────────────────────────────────
translator_llm1 = ChatOllama(
    model="llama3.2:3b",
    base_url="http://localhost:11434",
    temperature=0
)

# ─────────────────────────────────────────────
# TRANSLATOR LLM 2 — Marathi
# Model : gemma2:2b  (Google — strong on Indic scripts)
# Pull  : ollama pull gemma2:2b
# ─────────────────────────────────────────────
translator_llm2 = ChatOllama(
    model="gemma2:2b",
    base_url="http://localhost:11434",
    temperature=0
)

# ─────────────────────────────────────────────
# TRANSLATOR LLM 3 — Japanese
# Model : qwen2.5:3b  (Alibaba — top-tier CJK support)
# Pull  : ollama pull qwen2.5:3b
# ─────────────────────────────────────────────
translator_llm3 = ChatOllama(
    model="qwen2.5:3b",
    base_url="http://localhost:11434",
    temperature=0
)

# ─────────────────────────────────────────────
# TRANSLATION PROMPT TEMPLATES
# ─────────────────────────────────────────────
hindi_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert Hindi translator.
Translate the given English text into fluent, natural Hindi (Devanagari script).
Do not add any explanation — return only the translated text."""
    ),
    ("human", "Translate this to Hindi:\n\n{text}")
])

hinglish_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert in Hinglish — a natural mix of Hindi and English
commonly used in casual Indian conversation.
Translate the given English text into Hinglish.
Write Hindi words in Roman script (not Devanagari).
Keep technical or proper nouns in English.
Do not add any explanation — return only the translated text."""
    ),
    ("human", "Translate this to Hinglish:\n\n{text}")
])

marathi_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert Marathi translator.
Translate the given English text into fluent, natural Marathi (Devanagari script).
Do not add any explanation — return only the translated text."""
    ),
    ("human", "Translate this to Marathi:\n\n{text}")
])

japanese_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert Japanese translator.
Translate the given English text into natural, polite Japanese.
Use a mix of Kanji, Hiragana, and Katakana as appropriate.
Do not add any explanation — return only the translated text."""
    ),
    ("human", "Translate this to Japanese:\n\n{text}")
])

# ─────────────────────────────────────────────
# TRANSLATION CHAINS  (LCEL: prompt | llm | parser)
#
#  LLM 1 (llama3.2:3b)  →  Hindi, Hinglish
#  LLM 2 (gemma2:2b)    →  Marathi
#  LLM 3 (qwen2.5:3b)   →  Japanese
# ─────────────────────────────────────────────
hindi_chain     = hindi_prompt     | translator_llm1 | StrOutputParser()
hinglish_chain  = hinglish_prompt  | translator_llm1 | StrOutputParser()
marathi_chain   = marathi_prompt   | translator_llm2 | StrOutputParser()
japanese_chain  = japanese_prompt  | translator_llm3 | StrOutputParser()

LANGUAGE_CHAINS = {
    "hindi":    hindi_chain,
    "hinglish": hinglish_chain,
    "marathi":  marathi_chain,
    "japanese": japanese_chain,
}

SUPPORTED_LANGUAGES = list(LANGUAGE_CHAINS.keys()) + ["english"]


def translate_answer(english_text: str, language: str) -> str:
    """Route the English answer to the correct translation chain."""
    language = language.strip().lower()
    if language == "english" or language not in LANGUAGE_CHAINS:
        return english_text
    return LANGUAGE_CHAINS[language].invoke({"text": english_text})


# ─────────────────────────────────────────────
# HISTORY-AWARE RETRIEVER  (LCEL)
# ─────────────────────────────────────────────
history_prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    (
        "human",
        """Given the chat history and the latest user question above,
rephrase it as a standalone question that can be understood
without any prior context."""
    )
])

def contextualize_question(input_dict: dict) -> str:
    """
    If there is chat history, ask the LLM to rephrase the question
    as a standalone query. Otherwise return the question as-is.
    """
    chat_history = input_dict.get("chat_history", [])
    question = input_dict["input"]
    if not chat_history:
        return question
    messages = history_prompt.format_messages(
        chat_history=chat_history,
        input=question
    )
    return (rag_llm | StrOutputParser()).invoke(messages)


def retrieve_docs(input_dict: dict) -> list:
    standalone_q = contextualize_question(input_dict)
    return retriever.invoke(standalone_q)


def format_docs(docs: list) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


# ─────────────────────────────────────────────
# QA PROMPT
# ─────────────────────────────────────────────
qa_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful AI assistant.
Answer the question ONLY from the provided context.
If the answer is not found, say: 'I could not find the answer in the context.'

Context:
{context}"""
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

# ─────────────────────────────────────────────
# FULL RAG CHAIN  (LCEL)
# ─────────────────────────────────────────────
rag_chain = (
    RunnablePassthrough.assign(
        context=RunnableLambda(retrieve_docs) | format_docs
    )
    | qa_prompt
    | rag_llm
    | StrOutputParser()
)

# ─────────────────────────────────────────────
# MAIN CHAT LOOP
# ─────────────────────────────────────────────
chat_history = []

print("\n===== Multilingual Conversational RAG =====")
print(f"Supported languages : {', '.join(SUPPORTED_LANGUAGES)}")
print("Type 'exit' to quit.\n")

while True:
    # ── Step 1: question ────────────────────────────
    user_question = input("You: ").strip()
    if user_question.lower() == "exit":
        chat_history = []
        print("\nGoodbye!")
        break

    # ── Step 2: language ────────────────────────────
    lang_input = input(
        f"Language ({'/'.join(SUPPORTED_LANGUAGES)}) [default: english]: "
    ).strip().lower()

    if lang_input == "":
        lang_input = "english"

    if lang_input not in SUPPORTED_LANGUAGES:
        print(f"  ⚠  Unknown language '{lang_input}', defaulting to English.\n")
        lang_input = "english"

    # ── Step 3: RAG → English answer ────────────────
    english_answer: str = rag_chain.invoke({
        "chat_history": chat_history,
        "input": user_question
    })

    # ── Step 4: translate if needed ─────────────────
    if lang_input != "english":
        print(f"\n[English]            {english_answer}")
        final_answer = translate_answer(english_answer, lang_input)
        print(f"[{lang_input.capitalize():<16}] {final_answer}\n")
    else:
        print(f"\nAI: {english_answer}\n")
        final_answer = english_answer

    # ── Step 5: history (always stored in English) ──
    chat_history.extend([
        HumanMessage(content=user_question),
        AIMessage(content=english_answer)
    ])
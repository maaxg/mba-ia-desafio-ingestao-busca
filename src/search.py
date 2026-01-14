import os
import asyncio
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_postgres import PGVector
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""


def get_embeddings():
    """Choose embedding model based on available API keys"""
    openai_key = os.getenv("OPENAI_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")

    if openai_key and openai_key.strip():
        return OpenAIEmbeddings(
            model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        )
    elif google_key and google_key.strip():
        return GoogleGenerativeAIEmbeddings(
            model=os.getenv("GOOGLE_EMBEDDING_MODEL", "models/embedding-001")
        )
    else:
        raise ValueError(
            "No API key found. Set OPENAI_API_KEY or GOOGLE_API_KEY in .env"
        )


def get_llm():
    """Choose LLM model based on available API keys"""
    openai_key = os.getenv("OPENAI_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")

    if openai_key and openai_key.strip():
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-5-nano"),
            temperature=0
        )
    elif google_key and google_key.strip():
        return ChatGoogleGenerativeAI(
            model=os.getenv("GOOGLE_MODEL", "gemini-2.5-flash-lite"),
            temperature=0
        )
    else:
        raise ValueError(
            "No API key found. Set OPENAI_API_KEY or GOOGLE_API_KEY in .env"
        )


def get_vector_store():
    """Initialize PGVector store with embeddings"""
    embeddings = get_embeddings()
    return PGVector(
        embeddings=embeddings,
        collection_name=os.getenv("PGVERCTOR_COLLECTION"),
        connection=os.getenv("DATABASE_URL"),
        use_jsonb=True,
    )


def search_context(question):
    """Search for relevant context from vector store"""
    store = get_vector_store()
    results = store.similarity_search(question, k=10)
    
    context = "\n\n".join([doc.page_content.strip() for doc in results])
    
    return context


def search_prompt(question=None):
    """Create RAG chain for question answering"""
    if question is None:
        # Return chain for interactive use
        llm = get_llm()
        prompt = PromptTemplate(
            input_variables=["contexto", "pergunta"],
            template=PROMPT_TEMPLATE,
        )

        chain = (
            {
                "contexto": lambda x: search_context(x["pergunta"]),
                "pergunta": lambda x: x["pergunta"],
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        return chain
    else:
        # Return answer for specific question
        try:
            # Try to get existing event loop
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            # No event loop in current thread, create new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        llm = get_llm()
        prompt = PromptTemplate(
            input_variables=["contexto", "pergunta"],
            template=PROMPT_TEMPLATE,
        )
        
        # Get context
        context = search_context(question)
        # Format prompt
        formatted_prompt = prompt.format(contexto=context, pergunta=question)
        # Invoke LLM
        response = llm.invoke(formatted_prompt)

        # Extract content from response
        if hasattr(response, 'content'):
            return response.content
        return str(response)

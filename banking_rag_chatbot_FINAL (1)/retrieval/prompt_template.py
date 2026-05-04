"""
retrieval/prompt_template.py
Prompt templates for the banking chatbot.
Uses langchain_core.prompts (compatible with langchain >= 1.0).
"""

from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)

# System prompt
SYSTEM_PROMPT = """You are a knowledgeable and professional banking assistant for Indian customers. \
You have access to up-to-date information from RBI circulars, bank FAQs (SBI, HDFC, ICICI), \
and curated financial data.

STRICT RULES:
1. Answer ONLY based on the provided context below. Do not use outside knowledge.
2. If the answer is not found in the context, respond with:
   "I don't have that information in my current knowledge base. Please check the official \
RBI website (rbi.org.in) or your bank's official portal for the most accurate details."
3. Always be concise, accurate, and helpful.
4. When quoting specific figures (interest rates, fees, limits), mention the source document.
5. Never give personalised financial advice. Always recommend consulting a certified financial advisor.
6. Keep responses under 300 words unless the question requires detail.

CONTEXT:
{context}
"""


def get_chat_prompt() -> ChatPromptTemplate:
    """
    Returns a ChatPromptTemplate with system prompt, history, and user question.
    Compatible with RunnableWithMessageHistory (history injected automatically).
    """
    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{question}"),
    ])


# Simple one-shot prompt for evaluation
SIMPLE_RAG_TEMPLATE = """\
Use the following context to answer the question at the end.
If you cannot find the answer in the context, say:
"I don't have that information in my knowledge base."

Context:
{context}

Question: {question}

Answer:"""


def get_simple_prompt() -> ChatPromptTemplate:
    """Single-turn prompt for stateless RAG evaluation."""
    return ChatPromptTemplate.from_template(SIMPLE_RAG_TEMPLATE)


# Standalone question rewriter (kept for reference, not used in LCEL chain)
CONDENSE_QUESTION_TEMPLATE = """\
Given the following conversation history and a follow-up question, rephrase the \
follow-up question to be a standalone question that can be understood without \
the conversation history.

Chat History:
{chat_history}

Follow-up Question: {question}

Standalone Question:"""


def get_condense_prompt() -> PromptTemplate:
    """Prompt to condense follow-up questions into standalone queries."""
    return PromptTemplate.from_template(CONDENSE_QUESTION_TEMPLATE)


if __name__ == "__main__":
    prompt = get_chat_prompt()
    print("Chat prompt input variables:", prompt.input_variables)
    simple = get_simple_prompt()
    print("Simple prompt input variables:", simple.input_variables)

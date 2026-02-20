# utils/qa_pipeline.py

from typing import Dict, Any, List
from utils.rag import retrieve_relevant_chunks


def build_prompt(
    question: str,
    retrieved_chunks: List[Dict[str, Any]],
    user_language: str
) -> List[Dict[str, str]]:
    """
    Builds strict multilingual SGC auditor prompt.
    """

    context = "\n\n".join(
        [f"[DOC: {item['source']}]\n{item['chunk']}" for item in retrieved_chunks]
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are an official SGC Auditor.\n"
                "Your ONLY authority is the provided CONTEXT.\n\n"
                "CONTEXT (Spanish Official Documents):\n"
                + context
                + "\n\nSTRICT PROTOCOL:\n"
                "1. Answer ONLY using the information contained in the CONTEXT.\n"
                "2. If information is missing, explicitly state it is not available in the SGC manuals.\n"
                f"3. YOU MUST respond strictly in {user_language.upper()}.\n"
                "4. Do NOT switch languages under any circumstances.\n"
                "5. If necessary, translate facts from Spanish context into the required language."
            )
        },
        {"role": "user", "content": question}
    ]

    return messages




def call_llm(messages, llm_client, model_name: str):
    """
    Calls the Groq LLM using chat completion format.
    Returns:
        answer (str),
        usage (dict)
    """

    response = llm_client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.0
    )

    answer = response.choices[0].message.content

    # Extract token usage if available
    usage = getattr(response, "usage", {})

    return answer, usage





def run_qa_pipeline(
    question: str,
    knowledge_base: Dict[str, Any],
    llm_client,
    model_name: str,
    top_k: int = 5
) -> Dict[str, Any]:

    # Step 0: Detect user language (simple heuristic)
    user_language = "Spanish"
    if any(word in question.lower() for word in ["what", "how", "when", "where", "why", "process", "procedure"]):
        user_language = "English"

    # Step 1: Translate for retrieval
    translated_question = translate_to_spanish(question, llm_client, model_name)

    # Step 2: Retrieval using Spanish query
    retrieved_chunks = retrieve_relevant_chunks(
        question=translated_question,
        knowledge_base=knowledge_base,
        top_k=top_k
    )

    if not retrieved_chunks:
        return {
            "answer": "Esta información no está contemplada en los manuales de procedimientos del SGC.",
            "retrieved_chunks": [],
            "metadata": {
                "num_chunks_used": 0,
                "groundedness": 0.0,
                "max_similarity": 0.0,
                "avg_similarity": 0.0
            }
        }

    # --- Groundedness Calculation ---
    similarities = [chunk.get("score", 0.0) for chunk in retrieved_chunks]

    max_similarity = max(similarities)
    avg_similarity = sum(similarities) / len(similarities)

    # More robust groundedness → average of top 3
    top_k_similarities = sorted(similarities, reverse=True)[:3]
    groundedness_raw  = sum(top_k_similarities) / len(top_k_similarities)
    groundedness = min(1.0, groundedness_raw * 1.8)

    # Step 3: Build prompt using ORIGINAL question
    messages = build_prompt(question, retrieved_chunks, user_language)


    # Step 4: LLM generation
    answer, usage = call_llm(messages, llm_client, model_name)
    print("debug: ",max_similarity ,avg_similarity  )
    return {
    "answer": answer,
    "retrieved_chunks": retrieved_chunks,
    "usage": usage,
    "metadata": {
        "num_chunks_used": len(retrieved_chunks),
        "groundedness": groundedness,
        "max_similarity": max_similarity,
        "avg_similarity": avg_similarity
        }
    }




#Translates the user question to Spanish for better retrieval.
def translate_to_spanish(question: str, llm_client, model_name: str) -> str:
    """
    Translates the user question to Spanish for better retrieval.
    """

    messages = [
        {"role": "system", "content": "Translate the user message to Spanish. Return ONLY the translation."},
        {"role": "user", "content": question}
    ]

    response = llm_client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.0
    )

    return response.choices[0].message.content.strip()

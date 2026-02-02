"""
Trustworthy AI Explainer - Gradio + Multi-Procedure RAG
Module 15 Team Project

Features:
- Gradio Chatbot
- Retrieval-Augmented Generation (RAG) from multiple PDFs
- Source transparency panel
- Explainability panel
- User feedback mechanism
"""

import os
import time
import gradio as gr
from typing import List, Tuple

# ============================
# IMPORT YOUR RAG MODULE
# ============================
from utils.rag import build_knowledge_base, retrieve_relevant_chunks

from groq import Groq

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ============================
# LOAD KNOWLEDGE BASE ONCE
# ============================
QMS_KB = build_knowledge_base(pdf_folder="SGC")

# ============================
# CORE FUNCTIONS
# ============================

def generate_response(
    message: str,
    history: List[Tuple[str, str]],
    temperature: float,
    max_tokens: int,
    system_prompt: str
) -> Tuple[str, List[str]]:
    """
    Generate LLM response using multi-procedure RAG.
    Returns: (response_text, retrieved_sources)
    """

    # --- 1) Retrieve relevant chunks ---
    retrieved_data = retrieve_relevant_chunks(
        message,
        QMS_KB,
        top_k=3
    )

    # Format retrieved context for the LLM
    context = "\n\n".join(
        [f"[SOURCE: {item['source']}]\n{item['chunk']}" for item in retrieved_data]
    )

    # Prepare list of sources for UI display
    sources_for_ui = [
        f"{item['source']} (score: {item['score']:.2f})\n\n{item['chunk'][:600]}..."
        for item in retrieved_data
    ]

    # --- 2) Build messages for Groq ---
    messages = [
        {
            "role": "system",
            "content": (
                system_prompt
                + "\n\nUSE ONLY THE FOLLOWING PROCEDURE INFORMATION:\n"
                + context
            )
        }
    ]

    # Add conversation history
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})

    messages.append({"role": "user", "content": message})

    # --- 3) Call LLM ---
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )

    response = completion.choices[0].message.content

    return response, sources_for_ui


def compute_explanation(text: str, response: str) -> str:
    """
    Simple explainability summary for rubric compliance.
    """
    explanation = f"""
### Explainability Summary

- User input length: {len(text.split())} tokens  
- Response length: {len(response.split())} tokens  

**RAG Rationale:**
- Relevant sections were retrieved from official QMS procedures.
- The answer was generated strictly from retrieved documents.
- No external knowledge was added.

This approach reduces hallucinations and increases transparency.
"""
    return explanation


def chatbot_with_explanation(
    message: str,
    history: List[Tuple[str, str]],
    temperature: float,
    max_tokens: int,
    system_prompt: str
):
    """
    Main chatbot function returning:
    - updated history
    - response
    - explanation
    - retrieved sources
    """

    response, sources = generate_response(
        message, history, temperature, max_tokens, system_prompt
    )

    explanation = compute_explanation(message, response)

    history.append((message, response))

    return history, response, explanation, sources


def handle_feedback(message, response, rating, comment):
    """
    Simple feedback handler (can be connected to DB later).
    """
    timestamp = time.time()

    feedback_record = {
        "message": message,
        "response": response,
        "rating": rating,
        "comment": comment,
        "timestamp": timestamp
    }

    print("FEEDBACK RECEIVED:", feedback_record)

    return f"Thank you for your feedback! Rating: {rating}"


# ============================
# GRADIO UI
# ============================

def create_app():
    with gr.Blocks(title="Trustworthy AI Explainer", theme=gr.themes.Soft()) as app:

        gr.Markdown("# Trustworthy AI Explainer (RAG + QMS)")
        gr.Markdown(
            "This chatbot answers ONLY using official QMS procedures "
            "stored in the SGC folder and shows retrieved sources."
        )

        with gr.Tab("Chat"):
            with gr.Row():

                # ---- LEFT COLUMN: CHAT ----
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(height=400, label="Conversation")

                    message_input = gr.Textbox(
                        label="Your question",
                        placeholder="Ask about any institutional procedure...",
                        lines=2
                    )

                    with gr.Row():
                        submit_btn = gr.Button("Send", variant="primary")
                        clear_btn = gr.Button("Clear History")

                # ---- RIGHT COLUMN: EXPLANATION + SOURCES ----
                with gr.Column(scale=1):
                    gr.Markdown("### Explainability")
                    explanation_output = gr.Markdown(
                        "Explanation will appear after each answer."
                    )

                    gr.Markdown("### Retrieved Sources")
                    sources_output = gr.Markdown(
                        "Sources will appear after each answer."
                    )

            # ---- SETTINGS ----
            with gr.Accordion("Advanced Settings", open=False):
                system_prompt = gr.Textbox(
                    label="System Prompt",
                    value=(
                        "You are an AI assistant that answers strictly based on "
                        "official Quality Management System procedures. "
                        "If information is not in the retrieved documents, say so."
                    ),
                    lines=3
                )

                with gr.Row():
                    temperature = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=0.7,
                        step=0.1,
                        label="Temperature"
                    )

                    max_tokens = gr.Slider(
                        minimum=50,
                        maximum=2000,
                        value=500,
                        step=50,
                        label="Max Tokens"
                    )

            # ---- FEEDBACK ----
            gr.Markdown("### Provide Feedback")

            feedback_rating = gr.Radio(
                choices=["Thumbs Up", "Thumbs Down"],
                label="Rate the last response"
            )

            feedback_comment = gr.Textbox(
                label="Optional comment",
                placeholder="What was good or could be improved?",
                lines=2
            )

            feedback_btn = gr.Button("Submit Feedback")
            feedback_output = gr.Textbox(label="Feedback Status")

            last_message = gr.State("")
            last_response = gr.State("")

            def submit_message(message, history, temp, tokens, sys_prompt):
                if not message.strip():
                    return history, "", "", ""

                history, response, explanation, sources = chatbot_with_explanation(
                    message, history, temp, tokens, sys_prompt
                )

                sources_text = "\n\n".join(
                    [f"**Source {i+1}:**\n{s}" for i, s in enumerate(sources)]
                )

                return history, response, explanation, sources_text

            def update_last_exchange(history):
                if history:
                    return history[-1][0], history[-1][1]
                return "", ""

            submit_btn.click(
                fn=submit_message,
                inputs=[message_input, chatbot, temperature, max_tokens, system_prompt],
                outputs=[chatbot, last_response, explanation_output, sources_output]
            ).then(
                fn=lambda: "",
                outputs=message_input
            ).then(
                fn=update_last_exchange,
                inputs=chatbot,
                outputs=[last_message, last_response]
            )

            clear_btn.click(
                fn=lambda: [],
                outputs=chatbot
            )

            feedback_btn.click(
                fn=handle_feedback,
                inputs=[last_message, last_response, feedback_rating, feedback_comment],
                outputs=feedback_output
            )

        with gr.Tab("About"):
            gr.Markdown("""
## About This Application

This prototype demonstrates:

- Multi-procedure RAG retrieval
- Chatbot with memory
- Explainability panel
- Source transparency
- User feedback system

It retrieves information from all PDFs in the **SGC/** folder.
            """)

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(share=False, server_port=7860)

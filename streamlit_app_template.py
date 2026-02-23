"""
RAG-based Academic QA Assistant using SGC documents. - Streamlit Dashboard
Module 15 Team Project Template

This template provides a working Streamlit application with:
- Multi-page dashboard
- LLM chatbot with memory
- Explainability integration
- User feedback system
- State management and caching
- Performance monitoring

Team: Customize this template for your specific use case
"""

from lime import explanation
import streamlit as st
import os
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Optional
from datetime import datetime

from utils.rag import build_knowledge_base, retrieve_relevant_chunks
from utils.qa_pipeline import run_qa_pipeline
from utils.explainability import explain_similarity

from utils.rag import get_embedding_model
embedding_model = get_embedding_model()




# ============================================================================
# PAGE CONFIGURATION  
# ============================================================================

st.set_page_config(
    page_title="Trustworthy RAG – SGC",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_response" not in st.session_state:
    st.session_state.last_response = ""

if "last_question" not in st.session_state:
    st.session_state.last_question = ""

if "feedback_data" not in st.session_state:
    st.session_state.feedback_data = []


if "qms_kb" not in st.session_state:
    st.session_state.qms_kb = build_knowledge_base(pdf_folder="SGC")

if "current_explanation" not in st.session_state:
    st.session_state.current_explanation = {}





#st.title("RAG-based Academic QA Assistant – SGC Documents")

#st.markdown(
#    """
#    **RAG-based Academic QA Assistant using SGC documents.**  
#    This system retrieves relevant fragments from official academic procedures
#    before generating an answer.
#    """
#)

# ============================================================================
# LLM AND EMBEDDING MODEL LOADING
# ============================================================================
import os
from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


class MockLLM:
    """Mock LLM for testing - replace with actual implementation"""
    def generate(self, prompt: str, history: List = None, temperature: float = 0.7) -> str:
        time.sleep(0.5)  # Simulate API latency
        return f"Response to: '{prompt[:50]}...'\n\nThis is a mock response. Integrate your actual LLM here."

# ============================================================================
# CACHING FUNCTIONS
# ============================================================================

@st.cache_resource
def load_llm():
    """
    Load LLM model (cached as resource).
    Use @st.cache_resource for expensive objects like models.
    """
    # TODO: Replace with actual model loading
    # from langchain_openai import ChatOpenAI
    # return ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    
    return MockLLM()

@st.cache_data(ttl=3600)
def load_feedback_data() -> pd.DataFrame:
    """
    Load feedback data (cached for 1 hour).
    Use @st.cache_data for data that can be serialized.
    """
    # TODO: Load from database
    # For demo, create sample data
    if 'feedback_db' in st.session_state:
        return pd.DataFrame(st.session_state.feedback_db)
    
    return pd.DataFrame({
        'timestamp': [],
        'message': [],
        'response': [],
        'rating': [],
        'comment': []
    })

@st.cache_data
def compute_explanation(_input_text: str, _response: str, _cache_buster: float = 0.0) -> dict:

    retrieved = st.session_state.get("last_chunks", [])
    num_chunks = len(retrieved)

    # ======== CONFIDENCE (M16 RUBRIC-ALIGNED) ========
    if num_chunks >= 3:
        confidence = 0.92
    elif num_chunks == 2:
        confidence = 0.80
    elif num_chunks == 1:
        confidence = 0.65
    else:
        confidence = 0.50

    retrieved_docs = list({c.split("\n")[0] for c in retrieved})

    retrieval_methods = [
        "Semantic similarity (embeddings + cosine similarity)",
        "Document-code matching (P-, F-, IT-, D-)",
        "Temporal context boost (primera / first)"
    ]

    return {
        "confidence": confidence,
        "retrieved_documents": retrieved_docs,
        "retrieval_methods": retrieval_methods,
        "num_chunks": num_chunks
    }




# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================


def initialize_session_state():
    """Initialize all session state variables with default values."""
    
    # Chat messages for UI display
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Feedback database (in-memory for demo)
    if 'feedback_db' not in st.session_state:
        st.session_state.feedback_db = []
    
    # User preferences and model constraints
    if 'preferences' not in st.session_state:
        st.session_state.preferences = {
            'temperature': 0.0, # Set to 0.0 for deterministic technical accuracy
            'max_tokens': 500,
            'system_prompt': (
                "You are an AI assistant that helps teachers correctly fill out "
                "Quality Management System (QMS) documents based on official procedures."
            )
        }

    # Internal chat history for LLM context
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Stores the rationale for the latest response
    if 'current_explanation' not in st.session_state:
        st.session_state.current_explanation = None

    # CRITICAL FIX: Initialize last_chunks to prevent AttributeError in Explainability page
    if 'last_chunks' not in st.session_state:
        st.session_state.last_chunks = []
    
    # Application usage metrics
    if 'metrics' not in st.session_state:
        st.session_state.metrics = {
            'total_messages': 0,
            'avg_response_time': 0,
            'total_feedback': 0
        }

    # Knowledge Base initialization
    if "qms_kb" not in st.session_state:
        from utils.rag import build_knowledge_base
        st.session_state.qms_kb = build_knowledge_base(pdf_folder="SGC")
# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def generate_response(message: str, temperature: float = 0.0) -> tuple:

    start_time = time.time()

    result = run_qa_pipeline(
        question=message,
        knowledge_base=st.session_state.qms_kb,
        llm_client=client,
        model_name="llama-3.1-8b-instant",
        top_k=5
    )

    response_time = time.time() - start_time

    response = result["answer"]
    retrieved_chunks = result.get("retrieved_chunks", [])
    metadata = result.get("metadata", {})
    usage = result.get("usage", {})

    groundedness = metadata.get("groundedness", 0.0)
    max_similarity = metadata.get("max_similarity", 0.0)
    avg_similarity = metadata.get("avg_similarity", 0.0)
    retrieval_depth = metadata.get("num_chunks_used", 0)

    # Confidence calculation
    if retrieved_chunks:
        confidence = (
            0.6 * groundedness +
            0.3 * max_similarity +
            0.1 * (retrieval_depth / 5)
        )
    else:
        confidence = 0.0

    explanation = {
        "confidence": confidence,
        "usage": usage,
        "groundedness": groundedness,
        "max_similarity": max_similarity,
        "avg_similarity": avg_similarity,
        "retrieval_depth": retrieval_depth,
        "retrieval_methods": ["Semantic Search"],
        "explanation": (
            f"The system retrieved {retrieval_depth} semantically "
            f"similar document fragments from the official SGC knowledge base. "
            "The answer was generated strictly based on those sources "
            "without introducing external information."
        ),
        "response_time": response_time
    }

    # Save for later rendering
    st.session_state.last_chunks = retrieved_chunks
    st.session_state.last_user_question = message
    if "explanations_history" not in st.session_state:
        st.session_state.explanations_history = []

    if "chunks_history" not in st.session_state:
        st.session_state.chunks_history = []

    st.session_state.explanations_history.append(explanation)
    st.session_state.chunks_history.append(retrieved_chunks)
    print("DEBUG groundedness:", groundedness)
    print("DEBUG max_similarity:", max_similarity)
    print("DEBUG retrieval_depth:", retrieval_depth)
    print("DEBUG final_confidence:", confidence)


    return response, explanation, response_time










def save_feedback(message: str, response: str, rating: str, comment: str):
    """Save user feedback to session state."""
    feedback_entry = {
        'timestamp': datetime.now(),
        'message': message,
        'response': response,
        'rating': rating,
        'comment': comment
    }
    
    st.session_state.feedback_db.append(feedback_entry)
    st.session_state.metrics['total_feedback'] += 1
    
    # Clear cache to reload feedback data
    load_feedback_data.clear()

# ============================================================================
# PAGE: CHAT INTERFACE
# ============================================================================

def page_chat():
    """Main chat interface page with RAG transparency and explainability."""

    st.title("🤖 RAG-based Academic QA Assistant – SGC Documents")
    st.markdown(
        """
        This assistant answers **ONLY based on official Quality Management System (SGC) procedures.**

        How it works:
        1) Retrieves relevant official PDF documents.
        2) Selects the most relevant fragments.
        3) Generates an answer grounded exclusively in those sources.
        4) Displays sources and reasoning in the Explainability panel.

        Important:
        - The system does not invent information.
        - If a procedure is unclear or incomplete, it will state this explicitly.
        """
    )

    # -------------------------------
    # SIDEBAR SETTINGS (LOCAL TO CHAT)
    # -------------------------------
    with st.sidebar:
        st.header("⚙️ Settings")

        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=st.session_state.preferences['temperature'],
            step=0.1,
            help="Higher values make output more random"
        )
        st.session_state.preferences['temperature'] = temperature

        max_tokens = st.slider(
            "Max Tokens",
            min_value=50,
            max_value=2000,
            value=st.session_state.preferences['max_tokens'],
            step=50
        )
        st.session_state.preferences['max_tokens'] = max_tokens

        with st.expander("System Prompt"):
            system_prompt = st.text_area(
                "System Instructions",
                value=st.session_state.preferences['system_prompt'],
                height=120
            )
            st.session_state.preferences['system_prompt'] = system_prompt

        st.divider()

        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.session_state.current_explanation = None
            st.session_state.last_chunks = []
            st.session_state.metrics['total_messages'] = 0
            # Debug info in sidebar
            num_chunks = len(st.session_state.qms_kb['chunks'])
            st.sidebar.write(f"Total fragments in base: {num_chunks}")
            st.rerun()

    # -------------------------------
    # MAIN LAYOUT
    # -------------------------------
    col1, col2 = st.columns([2, 1])

    # ===============================
    # LEFT COLUMN — CHAT
    # ===============================
    with col1:
        st.subheader("Conversation")

        chat_container = st.container(height=300)

        with chat_container:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        if prompt := st.chat_input("Type your message here..."):

            # Save user message to UI state
            st.session_state.messages.append(
                {"role": "user", "content": prompt}
            )

            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)

            with st.spinner("Thinking with RAG..."):
                # Call optimized RAG function
                response, explanation, response_time = generate_response(
                    prompt,
                    temperature
                )
            # Increment total messages FIRST
            st.session_state.metrics['total_messages'] += 1

            total_msgs = st.session_state.metrics['total_messages']
            old_avg = st.session_state.metrics['avg_response_time']

            st.session_state.metrics['avg_response_time'] = (
                old_avg + (response_time - old_avg) / total_msgs
            )

    



            # Save turn in structured history for model memory
            st.session_state.chat_history.append((prompt, response))

            # Save assistant message to UI state
            st.session_state.messages.append(
                {"role": "assistant", "content": response}
            )

            # FLAT STRUCTURE: Save explanation directly to session state
            st.session_state.current_explanation = explanation
            st.session_state.current_explanation["input"] = prompt
            st.session_state.current_explanation["output"] = response
            st.session_state.current_explanation["response_time"] = response_time

            with chat_container:
                with st.chat_message("assistant"):
                    st.markdown(response)

            # Force refresh to update all dashboard components
            st.rerun()

    # ===============================
    # RIGHT COLUMN — EXPLAINABILITY
    # ===============================
    with col2:
        st.subheader("🔍 Explainability & Sources")

        if st.session_state.current_explanation:
            exp = st.session_state.current_explanation

            # --- METRICS ---
            m1, m2 = st.columns(2)
            with m1:
                # Direct access to confidence
                st.metric(
                    "Confidence",
                    f"{exp.get('confidence', 0.0):.2%}"
                )

            with m2:
                st.metric(
                    "Total Messages",
                    st.session_state.metrics['total_messages']
                )

            st.divider()

            # --- EXPLANATION ---
        st.markdown("### 🤔 How the answer was generated")

        # FIX: Removed ["details"] key. Using .get() for safety.
        st.markdown("**📄 Retrieved documents (Top Matches):**")

        docs = st.session_state.get("last_chunks", [])

        if docs:
            for doc in docs:
                source = doc.get("source", "Unknown")
                score = doc.get("score", 0.0)
                st.markdown(f"- {source} (Similarity: {score:.2f})")
        else:
            st.info("No documents retrieved.")



        st.markdown("**⚙️ Retrieval methods used:**")
        exp = st.session_state.get("current_explanation", {}) or {}

        methods = exp.get("retrieval_methods", ["Semantic Search"])
        for method in methods:
            st.markdown(f"- {method}")

        st.markdown("**🧠 Explanation (Rationale):**")
        # Direct access to explanation string
        st.markdown(exp.get("explanation", "No rationale provided."))

        st.divider()

        # --- RETRIEVED SOURCES (RAG TRANSPARENCY) ---
        st.subheader("📄 Retrieved Sources (RAG Transparency)")

        if "last_chunks" in st.session_state and len(st.session_state.last_chunks) > 0:

            for i, chunk in enumerate(st.session_state.last_chunks, 1):

                with st.expander(
                    f"Source {i} — {chunk['source']} ({chunk['score']:.2f})"
                ):
                    st.write(chunk["chunk"])

        else:
            st.info("No relevant sources were retrieved for this query.")

        #st.caption(f"Similarity score: {chunk['score']:.3f}")
        # --- LIME LOCAL EXPLAINABILITY ---
        if (
            "last_chunks" in st.session_state
            and len(st.session_state.last_chunks) > 0
            and "last_user_question" in st.session_state
        ):

            from utils.explainability import explain_similarity
            import pandas as pd
            import plotly.express as px

            st.divider()
            st.subheader("🧠 Local Explainability (LIME)")

            top_doc = st.session_state.last_chunks[0]["chunk"]

            lime_explanation = explain_similarity(
                st.session_state.last_user_question,
                top_doc,
                embedding_model
            )

            if lime_explanation is not None:

                explanation_list = lime_explanation.as_list()

                df = pd.DataFrame(explanation_list, columns=["Term", "Weight"])

                # Sort by absolute contribution
                df = df.reindex(df["Weight"].abs().sort_values().index)

                df["Impact"] = df["Weight"].apply(
                    lambda x: "Increases Similarity" if x > 0 else "Decreases Similarity"
                )

                fig = px.bar(
                    df,
                    x="Weight",
                    y="Term",
                    orientation="h",
                    color="Impact",
                    color_discrete_map={
                        "Increases Similarity": "#2ca02c",
                        "Decreases Similarity": "#d62728"
                    },
                    title="LIME Feature Contribution to Semantic Similarity"
                )

                fig.update_layout(
                    height=400,
                    yaxis_title="",
                    xaxis_title="Contribution Weight"
                )

                st.plotly_chart(fig, use_container_width=True)

                st.caption(
                    "Green terms increase semantic similarity between the question "
                    "and the top retrieved document. Red terms decrease similarity."
                )

            else:
                st.warning("LIME explanation could not be generated.")


            # --- FEEDBACK ---
            st.markdown("### 📝 Provide Feedback")

            rating = st.radio(
                "Rate this response:",
                options=["👍 Helpful", "👎 Not Helpful"],
                key=f"rating_{len(st.session_state.messages)}"
            )

            comment = st.text_area(
                "Optional comment:",
                placeholder="What was good or could be improved?",
                key=f"comment_{len(st.session_state.messages)}"
            )

            if st.button("Submit Feedback", use_container_width=True):
                save_feedback(
                    exp.get("input", ""),
                    exp.get("output", ""),
                    rating,
                    comment
                )
                st.success("✅ Feedback saved!")

        else:
            st.info("Send a message to see explanation and retrieved sources.")






def page_explainability():
    """Detailed explainability analysis page aligned with RAG architecture."""

    st.title("🔍 Explainability Analysis")
    st.markdown("Deep dive into model decisions and behavior patterns.")

    if not st.session_state.messages:
        st.info("No conversations yet. Go to the Chat page to start.")
        return

    # Build conversation pairs
    conversations = []
    for i in range(0, len(st.session_state.messages), 2):
        if i + 1 < len(st.session_state.messages):
            conversations.append({
                "user": st.session_state.messages[i]["content"],
                "assistant": st.session_state.messages[i + 1]["content"]
            })

    if not conversations:
        st.warning("No complete conversations found.")
        return

    selected_idx = st.selectbox(
        "Select conversation to analyze:",
        range(len(conversations)),
        format_func=lambda i: f"Conv {i+1}: {conversations[i]['user'][:50]}..."
    )

    selected_conv = conversations[selected_idx]

    col1, col2 = st.columns([1, 1.2])

    # ===============================
    # LEFT COLUMN — Conversation + Sources
    # ===============================
    with col1:
        st.subheader("💬 Conversation Review")
        st.info(f"**User:**\n{selected_conv['user']}")
        st.success(f"**Assistant:**\n{selected_conv['assistant']}")

        st.divider()
        st.subheader("📄 Retrieved Source Fragments")

        chunks = st.session_state.get("last_chunks", [])

        if chunks:
            for i, chunk in enumerate(chunks, 1):
                source = chunk.get("source", "Unknown")
                score = chunk.get("score", 0.0)
                text_content = chunk.get("chunk", "")

                with st.expander(f"{source} (Similarity: {score:.2f})"):
                    preview = text_content[:800]
                    st.text(preview + "..." if len(text_content) > 800 else preview)
        else:
            st.warning("No context fragments found for this turn.")

    # ===============================
    # RIGHT COLUMN — Metrics
    # ===============================
    with col2:
        st.subheader("🧠 Model Metrics")

        exp = st.session_state.get("current_explanation", None)
        chunks_history = st.session_state.get("chunks_history", [])
        explanations_history = st.session_state.get("explanations_history", [])

        if selected_idx < len(chunks_history):
            chunks = chunks_history[selected_idx]
        else:
            chunks = []

        if selected_idx < len(explanations_history):
            exp = explanations_history[selected_idx]
        else:
            exp = None

        if exp:

            m1, m2 = st.columns(2)
            m3, m4 = st.columns(2)
            m5, m6 = st.columns(2)
            #m7, m8 = st.columns(2)
            with m1:
                st.metric("Confidence", f"{exp.get('confidence', 0.0):.2%}")

            with m2:
                st.metric("Retrieved Chunks", len(chunks))

            with m3:
                st.metric("Retrieval Method", "Semantic Search")

            with m4:
                st.metric("Response Length", len(selected_conv["assistant"]))

            with m5:
                st.metric("Groundedness", f"{exp.get('groundedness', 0.0):.2%}")
                st.metric("Response Time", f"{exp.get('response_time', 0.0):.2}")
            with m6:
                st.metric("Max Similarity", f"{exp.get('max_similarity', 0.0):.2f}")
                st.metric("Average Similarity", f"{exp.get('avg_similarity', 0.0):.2f}")
            if exp.get("groundedness", 0) < 0.4:
                st.warning("Low groundedness: The answer may not be strongly supported by retrieved documents.")

            if exp.get("retrieval_depth", 0) == 0:
                st.error("No supporting documents were retrieved for this answer.")

            st.divider()

            st.markdown("### 🧠 Rationale")
            st.markdown(exp.get("explanation", "No rationale available."))

        else:
            st.info("No explainability metadata available.")

    # ===============================
    # FULL WIDTH — Pipeline Visualization
    # ===============================
    st.divider()
    st.subheader("📊 RAG Pipeline Breakdown")

    if chunks:
        importance = [0.5, 0.3, 0.2]
        labels = ["Retrieved Context", "LLM Generation", "Prompt Conditioning"]
    else:
        importance = [0.2, 0.5, 0.3]
        labels = ["Retrieved Context", "LLM Generation", "Prompt Conditioning"]

    fig = go.Figure(data=[
        go.Bar(x=labels, y=importance)
    ])

    fig.update_layout(
        title="Influence Factors for this Response",
        yaxis_title="Relative Influence",
        height=350
    )

    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# PAGE: FEEDBACK DASHBOARD
# ============================================================================

def page_feedback():
    """User feedback and quality monitoring page."""
    st.title("📊 Feedback Dashboard")
    st.markdown("Monitor user feedback and response quality.")
    
    # Load feedback data
    feedback_df = load_feedback_data()
    
    if feedback_df.empty:
        st.info("No feedback collected yet. Chat with the AI and provide feedback!")
        return
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Feedback", len(feedback_df))
    
    with col2:
        positive = len(feedback_df[feedback_df['rating'].str.contains('👍')])
        st.metric("Positive", positive)
    
    with col3:
        negative = len(feedback_df[feedback_df['rating'].str.contains('👎')])
        st.metric("Negative", negative)
    
    with col4:
        if len(feedback_df) > 0:
            satisfaction = (positive / len(feedback_df)) * 100
            st.metric("Satisfaction", f"{satisfaction:.1f}%")
    
    st.divider()
    
    # Feedback visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Feedback Distribution")
        
        rating_counts = feedback_df['rating'].value_counts()
        fig = px.pie(
            values=rating_counts.values,
            names=rating_counts.index,
            title="Positive vs Negative Feedback"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Feedback Over Time")
        
        feedback_df['date'] = pd.to_datetime(feedback_df['timestamp']).dt.date
        daily_counts = feedback_df.groupby('date').size().reset_index(name='count')
        
        fig = px.line(
            daily_counts,
            x='date',
            y='count',
            title="Feedback Count by Day",
            markers=True
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Recent feedback
    st.subheader("Recent Feedback")
    
    # Display feedback table
    display_df = feedback_df.copy()
    display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
    display_df['message'] = display_df['message'].str[:50] + '...'
    display_df['response'] = display_df['response'].str[:50] + '...'
    
    st.dataframe(
        display_df[['timestamp', 'message', 'response', 'rating', 'comment']],
        use_container_width=True,
        hide_index=True
    )
    
    # Export feedback
    st.divider()
    if st.button("📥 Export Feedback Data"):
        csv = feedback_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"feedback_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

# ============================================================================
# PAGE: MONITORING
# ============================================================================

def page_monitoring():
    """System monitoring and performance metrics page."""
    st.title("📈 System Monitoring")
    st.markdown("Track application performance and usage metrics.")
    
    # Metrics overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Total Messages",
            st.session_state.metrics['total_messages']
        )
    
    with col2:
        st.metric(
            "Avg Response Time",
            f"{st.session_state.metrics['avg_response_time']:.2f}s"
        )
    
    with col3:
        st.metric(
            "Total Feedback",
            st.session_state.metrics['total_feedback']
        )
    
    st.divider()
    
    # Cache status
    st.subheader("Cache Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Model Cache**")
        st.success("✅ Model loaded and cached")
        st.markdown("**Feedback Data Cache**")
        st.info("ℹ️ TTL: 1 hour")
    
    with col2:
        if st.button("🔄 Clear All Caches"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Caches cleared!")
            st.rerun()
    
    st.divider()
    
    # Session state inspection
    with st.expander("🔍 Session State (Debug)"):
        st.json({
            'messages_count': len(st.session_state.messages),
            'feedback_count': len(st.session_state.feedback_db),
            'preferences': st.session_state.preferences,
            'metrics': st.session_state.metrics
        })
    
    st.divider()
    
    # System recommendations
    st.subheader("Optimization Recommendations")
    
    if st.session_state.metrics['avg_response_time'] > 2.0:
        st.warning("⚠️ Average response time is high. Consider implementing streaming responses.")
    else:
        st.success("✅ Response times are within acceptable range.")
    
    if st.session_state.metrics['total_feedback'] < st.session_state.metrics['total_messages'] * 0.3:
        st.info("ℹ️ Feedback rate is low. Consider making feedback more prominent in the UI.")
    else:
        st.success("✅ Good feedback engagement rate.")

# ============================================================================
# PAGE: DOCUMENTATION
# ============================================================================

def page_documentation():
    """Documentation and system architecture overview."""

    st.title("📚 Documentation")

    tab1, tab2, tab3 = st.tabs(["About", "Technical", "Team"])

    # ==========================================================
    # TAB 1 — ABOUT
    # ==========================================================
    with tab1:
        st.markdown("""
        ## About This Application
        
        The **Trustworthy RAG Assistant** is a production-oriented Retrieval-Augmented 
        Generation (RAG) system designed to consult official SGC (Quality Management System) procedures.
        
        The system ensures that responses are strictly grounded in institutional documentation, 
        minimizing hallucinations and maximizing transparency.
        
        ---
        ### Key Features
        
        - **Grounded Conversational AI**
        - **Multilingual Responses**
        - **Explainability Dashboard**
        - **Similarity & Groundedness Metrics**
        - **System Monitoring**
        - **Hybrid Retrieval Strategy**
        
        ---
        ### 🌎 Multilingual Capability
        
        The system supports multilingual interaction:
        
        - User questions are translated to Spanish for optimized retrieval.
        - Retrieval operates on official Spanish SGC documentation.
        - The final response is generated in the SAME language as the user’s original question.
        
        This ensures:
        - Retrieval accuracy (Spanish knowledge base)
        - User-friendly interaction (English or Spanish output)
        - Strict grounding in official documentation
        
        ---
        ### Three Pillars Alignment
        
        This system follows the **Three Pillars of Trustworthy RAG**:
        
        - **Pillar I — Retrieval Quality**  
          Measured using cosine similarity and groundedness scores.
        
        - **Pillar II — Controlled Generation**  
          The LLM is strictly constrained to retrieved document context.
        
        - **Pillar III — Monitoring & Observability**  
          System latency and interaction metrics are tracked in real time.
        
        ---
        ### How to Use
        
        1. Ask a procedural question in English or Spanish.
        2. Review explainability metrics and retrieved fragments.
        3. Monitor system performance via the Monitoring page.
        4. Provide feedback for auditing purposes.
        """)

    # ==========================================================
    # TAB 2 — TECHNICAL
    # ==========================================================
    with tab2:
        st.markdown("""
        ## Technical Documentation
        
        ---
        ### System Architecture
        
        ```
        User Question
            ↓
        Language Detection
            ↓
        Translate (for retrieval optimization)
            ↓
        Semantic Retrieval (Top-K)
            ↓
        Groundedness Calculation
            ↓
        Prompt Construction
            ↓
        LLM Generation (Language Preserved)
            ↓
        Explainability + Monitoring
        ```
        
        ---
        ### Modular Design
        
        The system follows separation of concerns using independent modules.
        
        ```
        app.py
            ├── page_chat()
            ├── page_explainability()
            ├── page_monitoring()
            └── page_documentation()
        
        utils/
            ├── rag.py
            ├── qa_pipeline.py
            └── explainability.py
        ```
        
        ---
        ### utils/rag.py
        
        Handles retrieval logic:
        
        - Embedding comparison
        - Cosine similarity computation
        - Threshold filtering
        - Top-K chunk selection
        
        Returns:
        - Retrieved chunks
        - Similarity scores
        
        ---
        ### utils/qa_pipeline.py
        
        Central orchestration module.
        
        Responsibilities:
        
        - Detect user language
        - Translate question for retrieval
        - Call semantic retrieval
        - Compute:
            - Max Similarity
            - Average Similarity
            - Groundedness (Top-3 average)
        - Build strict auditor prompt
        - Force response language to match user
        - Call Groq LLM
        - Return metadata and usage
        
        ---
        ### utils/explainability.py
        
        Dedicated explainability layer.
        
        Includes:
        
        - `explain_similarity()`  
          Interprets semantic similarity relationships.
        
        - `similarity_predict()`  
          Provides structured similarity reasoning.
        
        Purpose:
        - Improve transparency
        - Separate evaluation logic from retrieval logic
        - Support audit-ready analysis
        
        ---
        ### Explainability Metrics
        
        Each response tracks:
        
        - Groundedness score
        - Max similarity
        - Average similarity
        - Retrieved chunk count
        - Confidence estimation
        - Response time
        
        ---
        ### Monitoring (Production-Oriented)
        
        The Monitoring page tracks:
        
        - Average Response Time (seconds)
        - Total system interactions
        
        This aligns with Pillar III:
        Observability and performance stability.
        
        ---
        ### Technical Stack
        
        - **LLM**: Llama-3.1-8b-instant (Groq API)
        - **Embeddings**: all-MiniLM-L6-v2
        - **Similarity**: Scikit-learn (Cosine Similarity)
        - **Frontend**: Streamlit
        - **Visualization**: Plotly
        - **State Management**: st.session_state
        
        ---
        ### Deterministic Design
        
        - Temperature = 0.0
        - Strict context-bound prompt
        - No external knowledge allowed
        - Transparent source inspection
        """)

    # ==========================================================
    # TAB 3 — TEAM
    # ==========================================================
    with tab3:
        st.markdown("""
        ## Team Information
        
        **Module**: 17 - Production Monitoring in Trustworthy AI  
        **Project**: Trustworthy RAG Assistant
        
        ---
        ### Team Members
        
        - **Daniel Alejandro Hernandez Castro** — Streamlit Architecture & Monitoring Integration
        - **Emer Ignacio Bernal** — Gradio Prototype Development
        - **Iliana Marlen Meza Sánchez** — Explainability Integration
        - **Víctor Daniel Ortiz García** — Deployment & Infrastructure
        
        ### Contact
        
        - **Daniel Hernandez**: daniel.hernandez@uttijuana.edu.mx
        - **Emer Bernal**: emer.ignacio@uttijuana.edu.mx
        - **Iliana Meza**: iliana.meza@tectijuana.edu.mx
        - **Víctor Ortiz**: victordortizg@gmail.com
        
        **Live App**: [Streamlit Cloud](https://qms-rag-dashboard-w5xmdfrwxwxmqeyrnxhvyk.streamlit.app/)            
        ---
        ### Contribution Highlights
        
        - Modular RAG pipeline implementation
        - Hybrid retrieval strategy
        - Multilingual response enforcement
        - Groundedness and similarity scoring
        - Production monitoring dashboard
        - Secure deployment
        
        ---
        © 2026 Trustworthy AI Module Project - 
        """)

          

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main application entry point."""
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar navigation
    with st.sidebar:
        st.title("🤖 Trustworthy AI")
        st.markdown("---")
        
        page = st.radio(
            "Navigation",
            ["💬 Chat", "🔍 Explainability", "📊 Feedback", "📈 Monitoring", "📚 Documentation"],
            label_visibility="collapsed"
        )
        
        st.caption(
        f"Session: {st.session_state.metrics['total_messages']} messages"
        )



    
    # Route to selected page
    if page == "💬 Chat":
        page_chat()
    elif page == "🔍 Explainability":
        page_explainability()
    elif page == "📊 Feedback":
        page_feedback()
    elif page == "📈 Monitoring":
        page_monitoring()
    elif page == "📚 Documentation":
        page_documentation()

if __name__ == "__main__":
    main()


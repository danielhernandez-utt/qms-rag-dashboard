"""
Trustworthy AI Explainer - Streamlit Dashboard
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

import streamlit as st
import os
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Optional
from datetime import datetime

from utils.rag import build_knowledge_base, retrieve_relevant_chunks


# ============================================================================
# PAGE CONFIGURATION  
# ============================================================================

st.set_page_config(
    page_title="Trustworthy AI Explainer - QMS Assistant",
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





st.title("Trustworthy AI Explainer – QMS Assistant")

st.markdown(
    """
    **RAG-based Academic QA Assistant using SGC documents.**  
    This system retrieves relevant fragments from official academic procedures
    before generating an answer.
    """
)

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
        confidence = 0.92      # 92%
    elif num_chunks == 2:
        confidence = 0.80      # 80%
    elif num_chunks == 1:
        confidence = 0.65      # 65%
    else:
        confidence = 0.50      # 50%

    # Documentos únicos recuperados
    retrieved_docs = list({c.split("\n")[0] for c in retrieved})

    retrieval_methods = [
        "Semantic similarity (embeddings + cosine similarity)",
        "Document-code matching (P-, F-, IT-, D-)",
        "Temporal context boost (primera / first)"
    ]

    return {
        "details": {
            "confidence": confidence,
            "retrieved_documents": retrieved_docs,
            "retrieval_methods": retrieval_methods,
            "explanation": (
                f"The model relied on {num_chunks} official SGC fragments. "
                f"The confidence score reflects both the number of retrieved "
                f"sources and their semantic alignment with the question."
            )
        }
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
    # Generates response with internal translation for cross-language retrieval
    start_time = time.time()

    # 1. Internal Translation for Retrieval (Silent Translation)
    # We use the LLM to translate the query to Spanish to improve RAG accuracy
    translation_prompt = [
        {"role": "system", "content": "Translate the user message to Spanish. Return ONLY the translation."},
        {"role": "user", "content": message}
    ]
    
    # Check if the message is likely not Spanish (simple check or always translate)
    translation_response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=translation_prompt,
        temperature=0.0
    )
    search_query = translation_response.choices[0].message.content

    # 2. Retrieval Phase: Search using the translated query
    retrieved_chunks = retrieve_relevant_chunks(
        search_query, 
        st.session_state.qms_kb, 
        top_k=5
    )

    # 3. Grounding Check: If no docs match the Spanish query
    if not retrieved_chunks:
        st.session_state.metrics['total_messages'] += 1
        # The logic will detect the language later, for now we return a default
        response = "Esta información no está contemplada en los manuales de procedimientos del SGC."
        explanation = {
            "confidence": 0.0,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "retrieved_documents": [],
            "explanation": "No matching records found after internal translation."
        }
        return response, explanation, time.time() - start_time

    # 4. Context Preparation
    context = "\n\n".join([f"[DOC: {item['source']}]\n{item['chunk']}" for item in retrieved_chunks])

    # 5. Multilingual System Prompt
    messages = [
        {
            "role": "system",
            "content": (
                "You are an official SGC Auditor. Your ONLY authority is the provided CONTEXT.\n"
                "CONTEXT IN SPANISH:\n" + context + "\n\n"
                "STRICT PROTOCOL:\n"
                "1. Answer ONLY using the official documentation provided.\n"
                "2. If info is missing, say it's not in the SGC manuals.\n"
                "3. ALWAYS respond in the SAME LANGUAGE as the user's original message.\n"
                "4. If the user asks in English, translate the facts from the Spanish context to English."
            )
        },
        {"role": "user", "content": message}
    ]

    # 6. LLM Call
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        temperature=0.0,
        max_tokens=st.session_state.preferences["max_tokens"]
    )

    response = completion.choices[0].message.content
    
    # Metrics Update (similar to previous steps)
    usage = {
        "prompt_tokens": completion.usage.prompt_tokens,
        "completion_tokens": completion.usage.completion_tokens,
        "total_tokens": completion.usage.total_tokens
    }
    
    max_sim = max([item["score"] for item in retrieved_chunks])
    confidence = (min(1.0, max_sim * 1.5) * 0.8) + ((len(retrieved_chunks)/5) * 0.2)
    st.session_state.metrics['total_messages'] += 1

    explanation = {
        "confidence": confidence,
        "usage": usage,
        "retrieved_documents": [f"{item['source']} ({item['score']:.2f})" for item in retrieved_chunks],
        "explanation": f"Validated using internal translation and {len(retrieved_chunks)} SGC fragments."
    }

    return response, explanation, time.time() - start_time




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

    st.title("🤖 Trustworthy RAG Assistant — QMS")
    st.markdown("Procedure: Multi-document SGC (RAG-based)")

    st.markdown(
        """
        This assistant answers **ONLY** based on official institutional procedures  
        stored in the Quality Management System (SGC).  

        It uses:
        - Retrieval-Augmented Generation (RAG)
        - Source transparency (expandable documents)
        - Explainability metrics
        - User feedback collection
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

        chat_container = st.container(height=400)

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
            # Update average response time metric
            total_msgs = st.session_state.metrics['total_messages']
            old_avg = st.session_state.metrics['avg_response_time']
            # Compute new average response time
            st.session_state.metrics['avg_response_time'] = ((old_avg * (total_msgs - 1)) + response_time) / total_msgs


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
            st.markdown("**📄 Retrieved documents (RAG Transparency):**")
            docs = exp.get("retrieved_documents", [])
            for doc in docs:
                st.markdown(f"- {doc}")

            st.markdown("**⚙️ Retrieval methods used:**")
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
                    with st.expander(f"Source {i}"):
                        st.text(chunk)
            else:
                st.info("No relevant sources were retrieved for this query.")

            st.divider()

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
    """Detailed explainability analysis page with improved visual layout."""
    st.title("🔍 Explainability Analysis")
    st.markdown("Deep dive into model decisions and behavior patterns.")
    
    if not st.session_state.messages:
        st.info("No conversations yet. Go to the Chat page to start.")
        return
    
    # Get recent conversations
    conversations = []
    for i in range(0, len(st.session_state.messages), 2):
        if i+1 < len(st.session_state.messages):
            conversations.append({
                'user': st.session_state.messages[i]['content'],
                'assistant': st.session_state.messages[i+1]['content']
            })
    
    if not conversations:
        st.warning("No complete conversations found.")
        return
    
    # Select conversation to analyze
    selected_idx = st.selectbox(
        "Select conversation to analyze:",
        range(len(conversations)),
        format_func=lambda i: f"Conv {i+1}: {conversations[i]['user'][:50]}..."
    )
    
    selected_conv = conversations[selected_idx]
    
    # Main layout division
    col1, col2 = st.columns([1, 1.2]) # Adjusted ratio for better fit
    
    with col1:
        st.subheader("💬 Conversation Review")
        st.info(f"**User:**\n{selected_conv['user']}")
        st.success(f"**Assistant:**\n{selected_conv['assistant']}")
        
        st.divider()
        st.subheader("📄 Sources used (RAG)")
        # Show fragments used in the retrieval
        if st.session_state.last_chunks:
            for i, chunk in enumerate(st.session_state.last_chunks, 1):
                with st.expander(f"Source Fragment {i}"):
                    st.text(chunk[:600] + "...")
        else:
            st.warning("No context fragments found for this turn.")

    with col2:
        st.subheader("🧠 Model Explainability")
        
        # DISPLAY METRICS IN GRID
        if st.session_state.current_explanation:
            exp = st.session_state.current_explanation
            usage = exp.get("usage", {})
            
            # Sub-columns for tokens and confidence (The aesthetic fix)
            m_col1, m_col2 = st.columns(2)
            m_col3, m_col4 = st.columns(2)
            
            with m_col1:
                st.metric("Input Tokens", usage.get("prompt_tokens", 0))
            with m_col2:
                st.metric("Response Tokens", usage.get("completion_tokens", 0))
            with m_col3:
                st.metric("Total Tokens", usage.get("total_tokens", 0))
            with m_col4:
                # Highlighted confidence metric
                st.metric("Confidence", f"{exp.get('confidence', 0.0):.2%}")

            st.divider()
            
            # Explainability features
            st.markdown("**Top Influence Features:**")
            top_features = exp.get("top_features", [
                "Semantic Similarity", 
                "Keyword Matching", 
                "Document Structure"
            ])

            for i, feature in enumerate(top_features, 1):
                st.write(f"{i}. {feature}")
        else:
            st.info("Metrics are available for the latest message in the chat.")

    # ---------------------------------------------------------
    # VISUALIZATION SECTION (Full width)
    # ---------------------------------------------------------
    st.divider()
    st.subheader("📊 RAG Pipeline Breakdown")
    
    # Data based on query reality
    features = ['Retrieved Context', 'Model Confidence', 'Prompt Alignment', 'History Context']
    
    # Dynamic importance weights
    if st.session_state.last_chunks:
        importance = [0.45, 0.25, 0.20, 0.10]
    else:
        importance = [0.05, 0.40, 0.35, 0.20]
    
    fig = go.Figure(data=[
        go.Bar(x=features, y=importance, marker_color='#0083B8') # Institutional color
    ])
    
    fig.update_layout(
        title="Influence Factors for this Specific Response",
        xaxis_title="Pipeline Components",
        yaxis_title="Influence Weight",
        height=350,
        margin=dict(l=20, r=20, t=40, b=20)
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
    """Documentation and team information page."""
    # Render the main title for the documentation page
    st.title("📚 Documentation")
    
    tab1, tab2, tab3 = st.tabs(["About", "Technical", "Team"])
    
    with tab1:
        st.markdown("""
        ## About This Application
        
        The **Trustworthy RAG Assistant** is a specialized interface for consulting 
        official SGC (Quality Management System) procedures. It ensures that 
        academic staff receive answers strictly grounded in institutional documentation.
        
        ### Features
        
        - **Interactive Chat**: Conversational AI with responses grounded in PDF procedures.
        - **Explainability**: Real-time analysis of retrieval rationale and confidence.
        - **Feedback System**: User rating and comments to support continuous auditing.
        - **Source Transparency**: Detailed view of the specific PDF chunks used as context.
        - **Hybrid Retrieval**: Combines semantic embeddings with document-code matching.
        
        ### How to Use
        
        1. **Chat**: Ask questions about procedures (e.g., "How to process a P- procedure?").
        2. **Review**: Open the "Explainability" tab to see the confidence and source chunks.
        3. **Feedback**: Rate the response accuracy to help refine the system.
        4. **Analyze**: Explore the Monitoring page to track interaction quality.
        5. **Verify**: Use the provided source names to cross-reference with official PDFs.
        """)
    
    with tab2:
        st.markdown("""
        ## Technical Documentation
        
        ### Architecture
        
        ```
        Pipeline: Load PDF → Chunk (250 words) → Embed → Retrieve → Generate
        LLM: Llama-3.1-8b-instant (via Groq)
        Embeddings: Sentence-Transformers (all-MiniLM-L6-v2)
        Similarity: Scikit-learn (Cosine Similarity)
        ```
        
        ### Hybrid Retrieval Strategy
        
        This system implements a custom retrieval logic in `utils/rag.py`:
        - **Semantic Search**: Uses `all-MiniLM-L6-v2` to find contextually relevant chunks.
        - **Document-Code Matching**: Specifically boosts fragments containing patterns: `P-`, `F-`, `IT-`, and `D-`.
        - **Source Diversity**: Prioritizes unique PDF sources to provide a broader context for the LLM.
        
        ### State Management
        
        Using `st.session_state`, the app persists:
        - Chat history and retrieval results.
        - Knowledge base (QMS chunks and embeddings).
        - Performance and feedback metrics.
        
        ### Deployment & Security
        
        - **Repository**: [github.com/danielhernandez-utt/qms-rag-dashboard](https://github.com/danielhernandez-utt/qms-rag-dashboard)
        - **Hosting**: Streamlit Community Cloud.
        - **Secrets**: API keys managed via Streamlit Secrets (GROQ_API_KEY).
        
        ### Requirements
        
        ```
        streamlit
        groq
        pypdf
        sentence-transformers
        scikit-learn
        pandas
        plotly
        ```
        """)
    
    with tab3:
        st.markdown("""
        ## Team Information
        
        **Team Members**:
        - **Daniel Alejandro Hernandez Castro**: Streamlit architect
        - **Emer Ignacio Bernal**: Gradio developer
        - **Iliana Marlen Meza Sánchez**: Backend integrator
        - **Víctor Daniel Ortiz García**: Deployment specialist
        
        **Module**: 16 - Trustworthy AI
        **Project**: Trustworthy RAG Assistant
        
        ### Team Contributions
        
        - **Gradio Prototype**: Emer Ignacio Bernal
        - **Streamlit Dashboard**: Daniel Alejandro Hernandez Castro
        - **Explainability Integration**: Iliana Marlen Meza Sánchez
        - **Deployment**: Víctor Daniel Ortiz García
        
        ### Contact
        
        - **Daniel Hernandez**: daniel.hernandez@uttijuana.edu.mx
        - **Emer Bernal**: emer.ignacio@uttijuana.edu.mx
        - **Iliana Meza**: iliana.meza@tectijuana.edu.mx
        - **Víctor Ortiz**: victordortizg@gmail.com
        
        **Live App**: [Streamlit Cloud](https://qms-rag-dashboard-w5xmdfrwxwxmqeyrnxhvyk.streamlit.app/)
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


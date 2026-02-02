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
# PAGE CONFIGURATION (Must be first Streamlit command)
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

st.markdown("""
This dashboard helps teachers correctly fill out Quality Management System (QMS) documents.
Use the navigation menu on the left to interact with the system.
""")
# ============================================================================
# MOCK LLM (Replace with actual LLM)
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
def compute_explanation(_input_text: str, _response: str) -> Dict:
    """
    Compute explainability metrics.
    Prefix with _ to exclude from cache key.
    """
    # TODO: Implement actual SHAP/LIME explanation
    return {
        'input_tokens': len(_input_text.split()),
        'response_tokens': len(_response.split()),
        'confidence': 0.85,
        'top_features': ['feature1', 'feature2', 'feature3'],
        'explanation': 'Mock explanation - implement SHAP/LIME here'
    }

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def initialize_session_state():
    """Initialize all session state variables."""
    
    # Chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Feedback database (in-memory for demo)
    if 'feedback_db' not in st.session_state:
        st.session_state.feedback_db = []
    
    # User preferences
    if 'preferences' not in st.session_state:
        st.session_state.preferences = {
            'temperature': 0.7,
            'max_tokens': 500,
            'system_prompt': (
    "You are an AI assistant that helps teachers correctly fill out Quality Management System (QMS) documents based on official institutional procedures. "
    "Provide clear and concise answers whenever possible.  "
    "Do not invent information.  "
    "If the procedure is unclear, say so explicitly."
)

        }
   # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


    # Current explanation
    if 'current_explanation' not in st.session_state:
        st.session_state.current_explanation = None
    
    # Performance metrics
    if 'metrics' not in st.session_state:
        st.session_state.metrics = {
            'total_messages': 0,
            'avg_response_time': 0,
            'total_feedback': 0
        }

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_response(message: str, temperature: float = 0.7) -> tuple:
    """
    Generate LLM response using RAG and compute explanation.

    Returns:
        Tuple of (response, explanation, response_time)
    """
    start_time = time.time()

    # -----------------------------
    # 1) RETRIEVE SOURCES (RAG)
    # -----------------------------
    retrieved_chunks = retrieve_relevant_chunks(
        message,
        st.session_state.qms_kb,
        top_k=3
    )

    # Save sources for UI transparency
    st.session_state.last_chunks = [
        f"{item['source']} (score: {item['score']:.2f})\n\n{item['chunk']}"
        for item in retrieved_chunks
    ]

    # Build context from retrieved documents
    context = "\n\n".join(
        [f"[SOURCE: {item['source']}]\n{item['chunk']}"
         for item in retrieved_chunks]
    )

    # -----------------------------
    # 2) BUILD PROMPT FOR GROQ
    # -----------------------------
    messages = [
        {
            "role": "system",
            "content": (
                st.session_state.preferences['system_prompt']
                + "\n\n"
                + "USE ONLY THE FOLLOWING PROCEDURE INFORMATION:\n"
                + context
                + "\n\n"
                + "Always reference the procedure used in your answer."
            )
        }
    ]

    # Add conversation history
    for user_msg, assist_msg in st.session_state.chat_history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assist_msg})

    # Add current user message
    messages.append({"role": "user", "content": message})

    # -----------------------------
    # 3) CALL GROQ
    # -----------------------------
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        temperature=st.session_state.preferences["temperature"],
        max_tokens=st.session_state.preferences["max_tokens"]
    )

    response = completion.choices[0].message.content

    # -----------------------------
    # 4) EXPLAINABILITY
    # -----------------------------
    explanation = compute_explanation(message, response)

    response_time = time.time() - start_time

    # -----------------------------
    # 5) UPDATE METRICS
    # -----------------------------
    st.session_state.metrics['total_messages'] += 1
    st.session_state.metrics['avg_response_time'] = (
        (st.session_state.metrics['avg_response_time'] *
         (st.session_state.metrics['total_messages'] - 1) + response_time)
        / st.session_state.metrics['total_messages']
    )

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
    # SIDEBAR SETTINGS
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

            # Save user message
            st.session_state.messages.append(
                {"role": "user", "content": prompt}
            )

            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)

            with st.spinner("Thinking with RAG..."):
                response, explanation, response_time = generate_response(
                    prompt,
                    temperature
                )

            # Save turn in structured history
            st.session_state.chat_history.append((prompt, response))

            # Save assistant message
            st.session_state.messages.append(
                {"role": "assistant", "content": response}
            )

            # Store explanation for right panel
            st.session_state.current_explanation = {
                "input": prompt,
                "output": response,
                "details": explanation,
                "response_time": response_time
            }

            with chat_container:
                with st.chat_message("assistant"):
                    st.markdown(response)

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
                st.metric("Response Time", f"{exp['response_time']:.2f}s")
            with m2:
                st.metric("Total Messages",
                           st.session_state.metrics['total_messages'])

            st.divider()

            # --- EXPLANATION ---
            st.markdown("### 🤔 How the answer was generated")
            st.markdown(exp["details"])

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
                    exp["input"],
                    exp["output"],
                    rating,
                    comment
                )
                st.success("✅ Feedback saved!")

        else:
            st.info("Send a message to see explanation and retrieved sources.")

# ============================================================================
# PAGE: EXPLAINABILITY ANALYSIS
# ============================================================================

def page_explainability():
    """Detailed explainability analysis page."""
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
    
    # Display conversation
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("User Input")
        st.markdown(f"```\n{selected_conv['user']}\n```")
        
        st.subheader("Model Response")
        st.markdown(f"```\n{selected_conv['assistant']}\n```")
        
        st.subheader("📄 Fuentes usadas (RAG)")

        for i, chunk in enumerate(st.session_state.last_chunks, 1):
            st.markdown(f"**Source  {i}:**")
            st.text(chunk[:400] + "...")

    with col2:
        st.subheader("Explainability")
        
        # Compute explanation
        exp = compute_explanation(selected_conv['user'], selected_conv['assistant'])
        
        # Display metrics
        st.metric("Input Tokens", exp['input_tokens'])
        st.metric("Response Tokens", exp['response_tokens'])
        st.metric("Confidence", f"{exp['confidence']:.2%}")
        
        st.divider()
        
        st.markdown("**Top Features:**")
        for i, feature in enumerate(exp['top_features'], 1):
            st.markdown(f"{i}. {feature}")
    
    # Visualization section
    st.divider()
    st.subheader("Feature Importance Visualization")
    
    # Mock feature importance data
    features = ['Input Length', 'Complexity', 'Context Relevance', 'Semantic Match', 'Pattern Recognition']
    importance = [0.25, 0.20, 0.30, 0.15, 0.10]
    
    fig = go.Figure(data=[
        go.Bar(x=features, y=importance, marker_color='lightblue')
    ])
    fig.update_layout(
        title="Feature Importance for Selected Response",
        xaxis_title="Features",
        yaxis_title="Importance Score",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # TODO: Add actual SHAP/LIME visualizations
    st.info("💡 **Team Task**: Implement actual SHAP or LIME visualizations here.")

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
    st.title("📚 Documentation")
    
    tab1, tab2, tab3 = st.tabs(["About", "Technical", "Team"])
    
    with tab1:
        st.markdown("""
        ## About This Application
        
        The Trustworthy AI Explainer Dashboard is a comprehensive interface for interacting
        with LLM applications while maintaining transparency and accountability.
        
        ### Features
        
        - **Interactive Chat**: Conversational AI with memory
        - **Explainability**: Real-time analysis of model decisions
        - **Feedback System**: User rating and comment collection
        - **Monitoring**: Performance metrics and usage analytics
        - **Multi-page Dashboard**: Organized interface for different tasks
        
        ### How to Use
        
        1. **Chat**: Go to the Chat page to interact with the AI
        2. **Review**: Check explanations for each response
        3. **Feedback**: Rate responses to help improve the system
        4. **Analyze**: Use the Explainability page for deep analysis
        5. **Monitor**: Track system performance on the Monitoring page
        """)
    
    with tab2:
        st.markdown("""
        ## Technical Documentation
        
        ### Architecture
        
        ```
        Frontend: Streamlit Multi-page App
        Backend: LLM (OpenAI/Local/Hugging Face)
        Explainability: SHAP/LIME
        State: st.session_state
        Caching: @st.cache_resource, @st.cache_data
        ```
        
        ### State Management
        
        This application uses `st.session_state` to persist:
        - Conversation history
        - User preferences
        - Feedback data
        - Performance metrics
        
        ### Caching Strategy
        
        - `@st.cache_resource`: Model loading (expensive, not serialized)
        - `@st.cache_data`: Data loading (serializable, with TTL)
        
        ### Deployment
        
        ```bash
        # Local
        streamlit run streamlit_app_template.py
        
        # Streamlit Community Cloud
        1. Push to GitHub
        2. Connect repo in Streamlit Cloud
        3. Deploy
        
        # Docker
        docker build -t trustworthy-ai-dashboard .
        docker run -p 8501:8501 trustworthy-ai-dashboard
        ```
        
        ### Environment Variables
        
        ```bash
        OPENAI_API_KEY=your_key_here
        # Add other API keys as needed
        ```
        
        ### Requirements
        
        ```
        streamlit
        pandas
        plotly
        openai  # or your LLM library
        numpy
        ```
        """)
    
    with tab3:
        st.markdown("""
        ## Team Information
        
        **Team Name**: [Your Team Name]
        
        **Team Members**:
        - Member 1: [Role]
        - Member 2: [Role]
        - Member 3: [Role]
        - Member 4: [Role]
        
        **Module**: 15 - App Prototyping with Streamlit
        
        **Project**: Trustworthy AI Explainer Dashboard
        
        ### Team Contributions
        
        - **Gradio Prototype**: [Team Member]
        - **Streamlit Dashboard**: [Team Member]
        - **Explainability Integration**: [Team Member]
        - **Deployment**: [Team Member]
        
        ### Contact
        
        For questions or support, contact: [team@example.com]
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
        
        st.markdown("---")
        st.caption(f"Session: {len(st.session_state.messages)} messages")
    
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


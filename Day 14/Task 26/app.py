import streamlit as st
import os
from dotenv import load_dotenv
from rag_system import RAGSystem

load_dotenv()

st.set_page_config(
    page_title="Ubuntu RAG Bot",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stChatMessage {
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196F3;
    }
    .assistant-message {
        background-color: #f5f5f5;
        border-left: 4px solid #4CAF50;
    }
    .retrieved-docs {
        background-color: #fff3e0;
        border-left: 4px solid #FF9800;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("Ubuntu RAG Bot")
st.markdown("An AI-powered assistant that answers questions about Ubuntu using Retrieval-Augmented Generation")

with st.sidebar:
    st.header("Configuration")
    
    pdf_file = "Ubuntu.pdf"
    if os.path.exists(pdf_file):
        st.success(f"PDF loaded: {pdf_file}")
    else:
        st.error(f"PDF not found: {pdf_file}")
    
    groq_key = os.getenv("GROQ_API_KEY")
    pinecone_key = os.getenv("PINECONE_API")
    
    st.subheader("API Keys Status")
    if groq_key:
        st.success("Groq API Key configured")
    else:
        st.warning("Groq API Key missing - add to .env file")
    
    if pinecone_key:
        st.success("Pinecone API Key configured")
    else:
        st.error("Pinecone API Key missing")
    
    st.divider()
    st.subheader("Settings")
    retrieval_count = st.slider(
        "Number of documents to retrieve:",
        min_value=1,
        max_value=5,
        value=3,
        help="How many relevant documents to use for context"
    )
    
    temperature = st.slider(
        "Response creativity:",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Higher = more creative, Lower = more factual"
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_system" not in st.session_state:
    try:
        if os.path.exists(pdf_file):
            with st.spinner("Initializing RAG system..."):
                st.session_state.rag_system = RAGSystem(pdf_file)
                st.session_state.rag_initialized = True
        else:
            st.session_state.rag_initialized = False
    except Exception as e:
        st.error(f"Error initializing RAG system: {str(e)}")
        st.session_state.rag_initialized = False

if st.session_state.get("rag_initialized", False):
    st.subheader("Conversation")
    
    # Display chat messages
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                if message["role"] == "assistant" and "retrieved_docs" in message:
                    with st.expander("Retrieved Documents"):
                        for i, doc in enumerate(message["retrieved_docs"], 1):
                            st.markdown(f"**Document {i} (Page {doc.get('page', 'N/A')})**")
                            st.markdown(f"*Relevance Score: {doc.get('score', 'N/A'):.2f}*")
                            st.text(doc["content"][:300] + "...")
                            st.divider()
    
    st.divider()
    st.subheader("Ask a question")
    
    col1, col2 = st.columns([1, 0.1])
    with col1:
        user_input = st.text_input(
            "Your question:",
            placeholder="Ask anything about Ubuntu...",
            label_visibility="collapsed"
        )
    
    with col2:
        submit_button = st.button("Send", use_container_width=True)
    
    if submit_button and user_input:
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })
        
        with st.spinner("Thinking..."):
            try:
                result = st.session_state.rag_system.query(user_input)
                
                # Add assistant response
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["response"],
                    "retrieved_docs": result["retrieved_docs"]
                })
                
                st.rerun()
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")
    
    st.divider()
    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

else:
    st.error("RAG system initialization failed. Please check your configuration.")
    st.info("Make sure:")
    st.markdown("- Ubuntu.pdf is in the same directory as the app")
    st.markdown("- GROQ_API_KEY and PINECONE_API are set in .env file")
    st.markdown("- All required packages are installed")

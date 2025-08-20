import streamlit as st
import requests
import json
import time
from datetime import datetime
import config

# Configure Streamlit
st.set_page_config(
    page_title="Smart Study Buddy - Streaming",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'current_response' not in st.session_state:
    st.session_state.current_response = ""
if 'streaming' not in st.session_state:
    st.session_state.streaming = False

def call_api(endpoint, method="GET", data=None):
    """Call FastAPI backend"""
    try:
        url = f"{config.FASTAPI_URL}{endpoint}"
        
        if method == "GET":
            response = requests.get(url, timeout=10)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=30)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
            
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to backend. Start FastAPI server: python fastapi_server.py")
        return None
    except Exception as e:
        st.error(f"Error: {e}")
        return None

def stream_response_simple(question):
    """Simple streaming approach for Streamlit"""
    try:
        url = f"{config.FASTAPI_URL}/ask-stream"
        
        with requests.post(url, json={"question": question}, stream=True, timeout=60) as response:
            if response.status_code != 200:
                return "Error: Could not connect to streaming endpoint"
            
            # Create container for streaming
            response_container = st.empty()
            accumulated_text = ""
            context_info = None
            
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        try:
                            data = json.loads(line[6:])
                            
                            if data['type'] == 'context':
                                context_info = data
                                if data['context_used']:
                                    st.info(f"📚 Using study materials: {data['context'][:100]}...")
                            
                            elif data['type'] == 'chunk':
                                accumulated_text += data['content']
                                # Update display in real-time
                                response_container.markdown(f"**🤖 AI Assistant:** {accumulated_text}")
                                time.sleep(0.05)  # Small delay for smooth streaming effect
                            
                            elif data['type'] == 'done':
                                break
                            
                            elif data['type'] == 'error':
                                return f"Error: {data['message']}"
                        
                        except json.JSONDecodeError:
                            continue
            
            return accumulated_text
            
    except Exception as e:
        return f"Streaming error: {str(e)}"

def main():
    """Main app"""
    
    # Header
    st.title("📚 Smart Study Buddy - Real-time Streaming")
    st.markdown("*Experience real-time AI responses with vector search*")
    
    # Check backend
    health = call_api("/")
    if not health:
        st.error("❌ Backend not available")
        st.code("python fastapi_server.py")
        return
    else:
        st.success("✅ Backend connected")
    
    # Two columns layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("📖 Study Materials")
        
        # Add material form
        with st.expander("➕ Add New Material", expanded=False):
            with st.form("material_form"):
                title = st.text_input("Title*")
                subject = st.text_input("Subject")
                chapter = st.text_input("Chapter")
                content = st.text_area("Content*", height=80)
                
                if st.form_submit_button("Add Material"):
                    if title and content:
                        result = call_api("/materials", "POST", {
                            "title": title,
                            "content": content,
                            "subject": subject,
                            "chapter": chapter
                        })
                        
                        if result and result.get("status") == "success":
                            st.success("✅ Material added!")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("❌ Failed to add")
                    else:
                        st.error("Please fill required fields")
        
        # Show materials
        materials = call_api("/materials")
        if materials:
            st.write(f"**Total Materials:** {len(materials)}")
            for i, material in enumerate(materials[-5:]):  # Last 5
                with st.expander(f"📄 {material['title'][:20]}..."):
                    st.write(f"**Subject:** {material.get('subject', 'N/A')}")
                    st.write(f"**Content:** {material['content'][:100]}...")
        else:
            st.info("No materials yet. Add some to get started!")
        
        # Stats
        stats = call_api("/stats")
        if stats:
            st.markdown("---")
            st.metric("📊 Total Materials", stats['materials']['total_materials'])
            st.metric("🔍 Indexed Documents", stats['vector_store']['total_documents'])
    
    with col2:
        st.header("💬 Chat with AI")
        
        # Username
        if not st.session_state.username:
            username = st.text_input("👤 Enter your name to start:")
            if st.button("Join Chat") and username:
                st.session_state.username = username
                st.rerun()
            return
        else:
            st.success(f"👋 Welcome, **{st.session_state.username}**!")
        
        # Chat messages container
        chat_container = st.container()
        
        with chat_container:
            # Display chat history
            for msg in st.session_state.messages:
                if msg['type'] == 'user':
                    st.markdown(f"**👤 {msg['username']}:** {msg['content']}")
                    st.caption(msg['timestamp'])
                
                elif msg['type'] == 'search':
                    st.info("🔍 **Found relevant materials:**")
                    for result in msg['results'][:2]:
                        st.write(f"• **{result['metadata']['title']}** (Relevance: {result['score']:.1%})")
                        st.write(f"  {result['text'][:120]}...")
                
                elif msg['type'] == 'ai':
                    st.markdown(f"**🤖 AI Assistant:** {msg['content']}")
                    if msg.get('context_used'):
                        st.caption("✅ Based on your study materials")
                    st.caption(msg['timestamp'])
                
                st.markdown("---")
        
        # Question input
        question = st.text_input("💭 Ask a question:", key="question_input", placeholder="e.g., What is machine learning?")
        
        col_send, col_clear = st.columns([1, 1])
        
        with col_send:
            ask_button = st.button("🚀 Ask AI", type="primary", use_container_width=True)
        
        with col_clear:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
        
        # Process question
        if ask_button and question:
            # Add user message
            st.session_state.messages.append({
                'type': 'user',
                'username': st.session_state.username,
                'content': question,
                'timestamp': datetime.now().strftime("%H:%M:%S")
            })
            
            # Search materials
            with st.spinner("🔍 Searching study materials..."):
                search_results = call_api("/search", "POST", {"question": question})
            
            if search_results:
                st.session_state.messages.append({
                    'type': 'search',
                    'results': search_results,
                    'timestamp': datetime.now().strftime("%H:%M:%S")
                })
            
            # Stream AI response
            st.markdown("### 🤖 AI Response (Streaming):")
            
            ai_response = stream_response_simple(question)
            
            if ai_response and not ai_response.startswith("Error"):
                st.session_state.messages.append({
                    'type': 'ai',
                    'content': ai_response,
                    'context_used': bool(search_results),
                    'timestamp': datetime.now().strftime("%H:%M:%S")
                })
                
                st.success("✅ Response complete!")
                time.sleep(1)
                st.rerun()
            else:
                st.error(f"❌ {ai_response}")
        
        # Auto-refresh option
        if st.checkbox("🔄 Auto-refresh (every 10s)", value=False):
            time.sleep(10)
            st.rerun()

if __name__ == "__main__":
    main()
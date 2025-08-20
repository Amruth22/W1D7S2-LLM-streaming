import streamlit as st
import requests
import json
from datetime import datetime
import config

# Configure Streamlit
st.set_page_config(
    page_title="Smart Study Buddy",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'username' not in st.session_state:
    st.session_state.username = ""

def call_api(endpoint, method="GET", data=None):
    """Call FastAPI backend"""
    try:
        url = f"{config.FASTAPI_URL}{endpoint}"
        
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=data)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
            
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to backend server. Make sure FastAPI server is running on port 8080.")
        return None
    except Exception as e:
        st.error(f"Error: {e}")
        return None

def main():
    """Main Streamlit app"""
    
    # Header
    st.title("📚 Smart Study Buddy")
    st.markdown("---")
    
    # Check backend connection
    health = call_api("/")
    if not health:
        st.error("Backend server not available. Please start the FastAPI server first.")
        st.code("python fastapi_server.py")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("👤 User")
        
        # Username
        if not st.session_state.username:
            username = st.text_input("Enter your name:")
            if st.button("Join"):
                if username.strip():
                    st.session_state.username = username.strip()
                    st.rerun()
        else:
            st.success(f"Welcome, {st.session_state.username}!")
            if st.button("Change User"):
                st.session_state.username = ""
                st.rerun()
        
        st.markdown("---")
        
        # Add Material
        st.header("📖 Add Material")
        with st.form("add_material"):
            title = st.text_input("Title*")
            subject = st.text_input("Subject")
            chapter = st.text_input("Chapter")
            content = st.text_area("Content*", height=100)
            
            if st.form_submit_button("Add Material"):
                if title.strip() and content.strip():
                    result = call_api("/materials", "POST", {
                        "title": title.strip(),
                        "content": content.strip(),
                        "subject": subject.strip(),
                        "chapter": chapter.strip()
                    })
                    
                    if result and result.get("status") == "success":
                        st.success("Material added!")
                        st.rerun()
                    else:
                        st.error("Failed to add material")
                else:
                    st.error("Please fill title and content")
        
        st.markdown("---")
        
        # Materials List
        st.header("📚 Materials")
        materials = call_api("/materials")
        
        if materials:
            st.write(f"Total: {len(materials)}")
            for material in materials[-3:]:  # Show last 3
                with st.expander(f"{material['title'][:25]}..."):
                    st.write(f"**Subject:** {material.get('subject', 'N/A')}")
                    st.write(f"**Chapter:** {material.get('chapter', 'N/A')}")
                    st.write(f"**Content:** {material['content'][:100]}...")
        else:
            st.info("No materials yet")
        
        st.markdown("---")
        
        # Stats
        st.header("📊 Stats")
        stats = call_api("/stats")
        if stats:
            st.metric("Materials", stats['materials']['total_materials'])
            st.metric("Indexed", stats['vector_store']['total_documents'])
    
    # Main Chat Area
    if not st.session_state.username:
        st.info("👈 Please enter your name to start")
        return
    
    st.header("💬 Study Chat")
    
    # Display messages
    for message in st.session_state.messages:
        if message['type'] == 'user':
            with st.chat_message("user"):
                st.write(f"**{message['username']}:** {message['content']}")
        
        elif message['type'] == 'search':
            with st.chat_message("assistant"):
                st.info("🔍 Found relevant materials:")
                for result in message['results'][:2]:
                    st.write(f"**{result['metadata']['title']}** (Score: {result['score']:.2f})")
                    st.write(f"{result['text'][:150]}...")
                    st.markdown("---")
        
        elif message['type'] == 'ai':
            with st.chat_message("assistant"):
                st.write(message['content'])
                if message.get('context_used'):
                    st.caption("Response based on study materials")
    
    # Chat input
    question = st.chat_input("Ask a question...")
    
    if question:
        # Add user message
        st.session_state.messages.append({
            'type': 'user',
            'username': st.session_state.username,
            'content': question,
            'timestamp': datetime.now().isoformat()
        })
        
        # Search for materials
        with st.spinner("🔍 Searching materials..."):
            search_results = call_api("/search", "POST", {"question": question})
        
        if search_results:
            st.session_state.messages.append({
                'type': 'search',
                'results': search_results,
                'timestamp': datetime.now().isoformat()
            })
        
        # Get AI response
        with st.spinner("🤖 Getting AI response..."):
            ai_response = call_api("/ask", "POST", {"question": question})
        
        if ai_response:
            st.session_state.messages.append({
                'type': 'ai',
                'content': ai_response['response'],
                'context_used': ai_response.get('context_used', False),
                'timestamp': datetime.now().isoformat()
            })
        
        st.rerun()
    
    # Clear chat
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

if __name__ == "__main__":
    main()
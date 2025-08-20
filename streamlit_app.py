import streamlit as st
import requests
import json
import time
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
            response = requests.get(url, timeout=10)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=30)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
            
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to backend server. Make sure FastAPI server is running on port 8080.")
        return None
    except requests.exceptions.Timeout:
        st.error("Request timeout. Please try again.")
        return None
    except Exception as e:
        st.error(f"Error: {e}")
        return None

def stream_ai_response(question):
    """Stream AI response from backend"""
    try:
        url = f"{config.FASTAPI_URL}/ask-stream"
        
        # Create streaming request
        response = requests.post(
            url, 
            json={"question": question},
            stream=True,
            timeout=60
        )
        
        if response.status_code != 200:
            return None, "Error connecting to streaming endpoint"
        
        # Create placeholder for streaming content
        response_placeholder = st.empty()
        accumulated_response = ""
        context_info = None
        
        # Process streaming response
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    try:
                        data = json.loads(line[6:])  # Remove 'data: ' prefix
                        
                        if data['type'] == 'context':
                            context_info = data
                        
                        elif data['type'] == 'chunk':
                            accumulated_response += data['content']
                            # Update the placeholder with current response
                            response_placeholder.markdown(f"**AI Assistant:** {accumulated_response}")
                        
                        elif data['type'] == 'done':
                            break
                        
                        elif data['type'] == 'error':
                            return None, data['message']
                    
                    except json.JSONDecodeError:
                        continue
        
        return accumulated_response, context_info
        
    except requests.exceptions.Timeout:
        return None, "Streaming timeout. Please try again."
    except Exception as e:
        return None, f"Streaming error: {str(e)}"

def main():
    """Main Streamlit app"""
    
    # Header
    st.title("📚 Smart Study Buddy")
    st.markdown("*Real-time AI streaming with vector search*")
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
                    with st.spinner("Adding material..."):
                        result = call_api("/materials", "POST", {
                            "title": title.strip(),
                            "content": content.strip(),
                            "subject": subject.strip(),
                            "chapter": chapter.strip()
                        })
                    
                    if result and result.get("status") == "success":
                        st.success("Material added!")
                        time.sleep(1)
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
                st.caption(message['timestamp'])
        
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
                    st.caption("✅ Response based on study materials")
                st.caption(message['timestamp'])
    
    # Chat input
    question = st.chat_input("Ask a question...")
    
    if question:
        # Add user message
        st.session_state.messages.append({
            'type': 'user',
            'username': st.session_state.username,
            'content': question,
            'timestamp': datetime.now().strftime("%H:%M:%S")
        })
        
        # Rerun to show user message immediately
        st.rerun()
    
    # Process the last question if it exists and hasn't been processed
    if (st.session_state.messages and 
        st.session_state.messages[-1]['type'] == 'user' and
        len([m for m in st.session_state.messages if m['type'] == 'ai']) < len([m for m in st.session_state.messages if m['type'] == 'user'])):
        
        last_question = st.session_state.messages[-1]['content']
        
        # Search for materials first
        with st.spinner("🔍 Searching materials..."):
            search_results = call_api("/search", "POST", {"question": last_question})
        
        if search_results:
            st.session_state.messages.append({
                'type': 'search',
                'results': search_results,
                'timestamp': datetime.now().strftime("%H:%M:%S")
            })
        
        # Stream AI response
        st.info("🤖 AI is responding...")
        
        ai_response, context_info = stream_ai_response(last_question)
        
        if ai_response:
            st.session_state.messages.append({
                'type': 'ai',
                'content': ai_response,
                'context_used': context_info.get('context_used', False) if context_info else False,
                'timestamp': datetime.now().strftime("%H:%M:%S")
            })
        else:
            st.error(f"Error getting AI response: {context_info}")
        
        # Rerun to show complete conversation
        time.sleep(1)
        st.rerun()
    
    # Control buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Refresh"):
            st.rerun()
    
    with col2:
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()

if __name__ == "__main__":
    main()
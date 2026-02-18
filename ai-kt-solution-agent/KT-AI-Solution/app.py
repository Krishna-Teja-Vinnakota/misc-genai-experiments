"""
Streamlit UI for KT Documentation Generator
"""
import streamlit as st
import os
import zipfile
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

# Import modules
import embeddings
import vector_store
import documentation
import chat
from config import *


def init_session_state():
    """Initialize session state variables"""
    if 'collections' not in st.session_state:
        st.session_state.collections = []
    if 'selected_collection' not in st.session_state:
        st.session_state.selected_collection = None
    if 'indexing_complete' not in st.session_state:
        st.session_state.indexing_complete = False
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    # Documentation generation state
    if 'doc_generation_status' not in st.session_state:
        st.session_state.doc_generation_status = None  # 'complete', 'error'
    if 'generated_docs' not in st.session_state:
        st.session_state.generated_docs = None
    if 'doc_output_path' not in st.session_state:
        st.session_state.doc_output_path = None
    # Indexing state
    if 'indexing_status' not in st.session_state:
        st.session_state.indexing_status = None  # 'complete', 'error'
    if 'indexing_result' not in st.session_state:
        st.session_state.indexing_result = None


def check_env_variables():
    """Check if required environment variables are set"""
    missing = []
    
    if not os.getenv('QDRANT_URL'):
        missing.append('QDRANT_URL')
    if not os.getenv('QDRANT_API_KEY'):
        missing.append('QDRANT_API_KEY')
    
    return missing


def refresh_collections():
    """Refresh the list of collections from Qdrant"""
    try:
        client = vector_store.get_qdrant_client()
        st.session_state.collections = vector_store.list_all_collections(client)
    except Exception as e:
        st.error(f"Error refreshing collections: {e}")
        st.session_state.collections = []


def main():
    st.set_page_config(
        page_title="KT Documentation Generator",
        page_icon="📚",
        layout="wide"
    )
    
    init_session_state()
    
    # Header
    st.title("📚 KT Documentation Generator")
    st.markdown("Generate comprehensive documentation for Python, JavaScript, React, and Node.js codebases using AI")
    
    # Sidebar - Collection Management
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Check environment variables
        missing_vars = check_env_variables()
        if missing_vars:
            st.error(f"❌ Missing: {', '.join(missing_vars)}")
            st.info("Set these in `.env` file")
        else:
            st.success("✅ Environment configured")
        
        st.divider()
        
        # Test connection
        if st.button("🔌 Test Qdrant Connection", use_container_width=True):
            with st.spinner("Testing..."):
                if vector_store.test_connection():
                    st.success("✅ Connected")
                    refresh_collections()
                else:
                    st.error("❌ Connection failed")
        
        st.divider()
        
        # Collections Management
        st.subheader("📦 Collections")
        
        if st.button("🔄 Refresh Collections", use_container_width=True):
            refresh_collections()
            st.success("Refreshed!")
        
        # Show collections
        if st.session_state.collections:
            st.markdown(f"**Total: {len(st.session_state.collections)} collections**")
            
            for coll in st.session_state.collections:
                with st.expander(f"📁 {coll['name']}"):
                    st.metric("Files", coll['vectors_count'])
                    st.text(f"Status: {coll['status']}")
                    
                    # Delete button
                    if st.button(f"🗑️ Delete", key=f"del_{coll['name']}", use_container_width=True):
                        client = vector_store.get_qdrant_client()
                        if vector_store.delete_collection(client, coll['name']):
                            st.success(f"Deleted {coll['name']}")
                            refresh_collections()
                            st.rerun()
        else:
            st.info("No collections yet")
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📥 Index Codebase", "📄 Generate Documentation", "💬 Chat with Codebase"])
    
    # Tab 1: Index Codebase
    with tab1:
        st.header("📥 Index Codebase")
        st.markdown("Scan and index your codebase (Python, JavaScript, React, Node.js) for documentation generation")
        
        # Input method selection
        st.subheader("Choose Input Method")
        input_method = st.radio(
            "Select how to provide your codebase:",
            options=["📁 Directory Path", "📦 ZIP File Upload"],
            horizontal=True
        )
        
        st.divider()
        
        codebase_path = None
        uploaded_file = None
        
        if input_method == "📁 Directory Path":
            codebase_path = st.text_input(
                "Codebase Path",
                placeholder="/path/to/your/project",
                help="Enter the full path to your project directory (Python, JavaScript, React, Node.js)"
            )
        else:  # ZIP File Upload
            uploaded_file = st.file_uploader(
                "Upload ZIP File",
                type=['zip'],
                help="Upload a ZIP file containing your project (Python, JavaScript, React, Node.js)"
            )
            
            if uploaded_file is not None:
                st.success(f"✅ Uploaded: {uploaded_file.name} ({uploaded_file.size / 1024:.2f} KB)")
                # Set a temporary path for processing
                codebase_path = "temp_uploaded_codebase"
        
        if codebase_path and (Path(codebase_path).exists() or uploaded_file is not None):
            # Generate collection name based on input method
            if input_method == "📁 Directory Path":
                suggested_collection = vector_store.get_collection_name_from_path(codebase_path)
            else:  # ZIP File Upload
                base_name = Path(uploaded_file.name).stem if uploaded_file else "uploaded_code"
                suggested_collection = COLLECTION_NAME_PREFIX + base_name.lower().replace(" ", "_").replace("-", "_")
            
            st.info(f"📦 Collection will be: `{suggested_collection}`")
            
            # Check if collection exists
            if missing_vars:
                st.warning("⚠️ Configure environment variables first")
            else:
                try:
                    client = vector_store.get_qdrant_client()
                    coll_exists = vector_store.collection_exists(client, suggested_collection)
                    
                    if coll_exists:
                        st.warning(f"⚠️ Collection `{suggested_collection}` already exists!")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            overwrite = st.checkbox("✅ Overwrite existing collection", value=False)
                        with col2:
                            if overwrite:
                                st.error("⚠️ Existing data will be deleted!")
                    else:
                        overwrite = True  # New collection, proceed
                        
                except Exception as e:
                    st.error(f"Error checking collection: {e}")
                    overwrite = False
        else:
            overwrite = False
        
        # Show indexing success if completed
        if st.session_state.indexing_status == 'complete' and st.session_state.indexing_result:
            result = st.session_state.indexing_result
            st.success(f"✅ Successfully indexed {result['files_count']} files into `{result['collection_name']}`!")
            st.info("👉 Go to 'Generate Documentation' or 'Chat with Codebase' tabs")
            
            if st.button("🔄 Index New Codebase", use_container_width=True):
                st.session_state.indexing_status = None
                st.session_state.indexing_result = None
                st.rerun()
        
        # Index button
        elif st.button("🚀 Start Indexing", type="primary", use_container_width=True, disabled=not codebase_path):
            if not codebase_path:
                st.error("❌ Please enter a codebase path or upload a ZIP file")
            elif input_method == "📁 Directory Path" and not Path(codebase_path).exists():
                st.error("❌ Path does not exist")
            elif input_method == "📦 ZIP File Upload" and uploaded_file is None:
                st.error("❌ Please upload a ZIP file")
            elif missing_vars:
                st.error("❌ Please configure environment variables first")
            else:
                # Start indexing with status tracking
                with st.status("🚀 Indexing codebase...", expanded=True) as status:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    temp_dir = None
                    
                    try:
                        # Handle ZIP file extraction
                        if input_method == "📦 ZIP File Upload" and uploaded_file is not None:
                            status_text.text("📦 Extracting ZIP file...")
                            
                            # Create temporary directory
                            temp_dir = tempfile.mkdtemp(prefix="kt_codebase_")
                            
                            # Save uploaded file
                            zip_path = os.path.join(temp_dir, uploaded_file.name)
                            with open(zip_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            
                            # Extract ZIP
                            extract_dir = os.path.join(temp_dir, "extracted")
                            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                                zip_ref.extractall(extract_dir)
                            
                            # Find the actual codebase root (handle single root folder in ZIP)
                            extracted_items = os.listdir(extract_dir)
                            if len(extracted_items) == 1 and os.path.isdir(os.path.join(extract_dir, extracted_items[0])):
                                codebase_path = os.path.join(extract_dir, extracted_items[0])
                            else:
                                codebase_path = extract_dir
                            
                            st.success(f"✅ ZIP extracted to temporary directory")
                        
                        client = vector_store.get_qdrant_client()
                        
                        # Generate collection name (use uploaded filename for ZIP files)
                        if input_method == "📦 ZIP File Upload" and uploaded_file is not None:
                            # Use the ZIP filename (without extension) for collection name
                            base_name = Path(uploaded_file.name).stem
                            collection_name = COLLECTION_NAME_PREFIX + base_name.lower().replace(" ", "_").replace("-", "_")
                        else:
                            collection_name = vector_store.get_collection_name_from_path(codebase_path)
                        
                        # Delete if overwriting
                        if vector_store.collection_exists(client, collection_name) and 'overwrite' in locals() and overwrite:
                            status_text.text("🗑️ Deleting existing collection...")
                            vector_store.delete_collection(client, collection_name)
                        
                        # Create collection
                        status_text.text("📦 Creating collection...")
                        vector_store.create_collection(client, collection_name)
                        
                        # Index codebase - First scan for total count
                        status_text.text("🔍 Scanning codebase for files...")
                        files_to_process = embeddings.scan_codebase(codebase_path)
                        total_files = len(files_to_process)
                        
                        st.info(f"📊 Found **{total_files} files** to process")
                        
                        status_text.text(f"⚙️ Processing {total_files} files...")
                        
                        # Simple progress callback
                        def progress_callback(current, total, filename):
                            progress = current / total
                            progress_bar.progress(progress)
                            status_text.text(f"📄 Processing [{current}/{total}]: {filename}")
                        
                        processed_files = embeddings.index_codebase(codebase_path, progress_callback)
                        
                        # Store in Qdrant
                        status_text.text("💾 Storing embeddings...")
                        vector_store.store_embeddings(client, collection_name, processed_files)
                    
                        progress_bar.progress(1.0)
                        status_text.empty()
                        
                        # Store results in session state
                        st.session_state.indexing_status = 'complete'
                        st.session_state.indexing_result = {
                            'files_count': len(processed_files),
                            'collection_name': collection_name
                        }
                        st.session_state.indexing_complete = True
                        st.session_state.selected_collection = collection_name
                        
                        # Refresh collections
                        refresh_collections()
                        
                        status.update(label="✅ Indexing completed successfully!", state="complete", expanded=False)
                        st.balloons()
                        st.rerun()
                        
                    except Exception as e:
                        st.session_state.indexing_status = 'error'
                        st.error(f"❌ Error during indexing: {str(e)}")
                        status.update(label="❌ Indexing failed", state="error", expanded=False)
                    
                    finally:
                        # Cleanup temporary directory
                        if temp_dir and os.path.exists(temp_dir):
                            try:
                                status_text.text("🧹 Cleaning up temporary files...")
                                shutil.rmtree(temp_dir)
                                status_text.empty()
                            except Exception as cleanup_error:
                                st.warning(f"⚠️ Could not cleanup temporary files: {cleanup_error}")
    
    # Tab 2: Generate Documentation
    with tab2:
        st.header("📄 Generate Documentation")
        st.markdown("Select collection and documentation types to generate")
        
        # Refresh collections if not loaded
        if not st.session_state.collections:
            refresh_collections()
        
        # Collection selection
        if not st.session_state.collections:
            st.warning("⚠️ No collections found. Please index a codebase first.")
        else:
            collection_names = [c['name'] for c in st.session_state.collections]
            
            selected_collection = st.selectbox(
                "📦 Select Collection",
                options=collection_names,
                index=collection_names.index(st.session_state.selected_collection) if st.session_state.selected_collection in collection_names else 0,
                help="Choose which codebase collection to generate documentation from"
            )
            
            # Show collection info
            selected_coll_info = next((c for c in st.session_state.collections if c['name'] == selected_collection), None)
            if selected_coll_info:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("📊 Total Files", selected_coll_info['vectors_count'])
                with col2:
                    st.metric("📁 Collection", selected_collection.replace(COLLECTION_NAME_PREFIX, ''))
            
            st.divider()
            
            # Select documentation types
            st.subheader("Select Documentation Types")
            
            col1, col2 = st.columns(2)
            
            selected_types = []
            with col1:
                if st.checkbox(DOC_TYPE_LABELS["system_overview"], value=True):
                    selected_types.append("system_overview")
                if st.checkbox(DOC_TYPE_LABELS["api_reference"], value=True):
                    selected_types.append("api_reference")
                if st.checkbox(DOC_TYPE_LABELS["data_models"], value=True):
                    selected_types.append("data_models")
            
            with col2:
                if st.checkbox(DOC_TYPE_LABELS["business_flows"], value=True):
                    selected_types.append("business_flows")
                if st.checkbox(DOC_TYPE_LABELS["integrations"], value=True):
                    selected_types.append("integrations")
                if st.checkbox(DOC_TYPE_LABELS["deployment"], value=True):
                    selected_types.append("deployment")
            
            st.divider()
            
            # Important note about long-running operations
            st.info("⏱️ **Note:** Documentation generation can take 2-5 minutes depending on codebase size.")
            
            # Show existing generated docs if available
            if st.session_state.doc_generation_status == 'complete' and st.session_state.generated_docs:
                st.success(f"✅ Documentation generated successfully!")
                
                # Download button
                if st.session_state.doc_output_path and os.path.exists(st.session_state.doc_output_path):
                    with open(st.session_state.doc_output_path, "rb") as f:
                        st.download_button(
                            label=f"📥 Download {os.path.basename(st.session_state.doc_output_path)}",
                            data=f,
                            file_name=os.path.basename(st.session_state.doc_output_path),
                            mime="application/zip",
                            use_container_width=True,
                            key="download_docs"
                        )
                
                # Show FULL preview (no truncation)
                st.divider()
                st.subheader("📋 Documentation Preview")
                
                for doc_type, content in st.session_state.generated_docs.items():
                    with st.expander(f"📄 {DOC_TYPE_LABELS[doc_type]}", expanded=False):
                        st.markdown(content)
                
                st.divider()
                
                # Clear button to generate new docs
                if st.button("🔄 Generate New Documentation", use_container_width=True):
                    st.session_state.doc_generation_status = None
                    st.session_state.generated_docs = None
                    st.session_state.doc_output_path = None
                    st.rerun()
            
            # Generate button (only show if not already generated)
            elif st.button("🎯 Generate Documentation", type="primary", use_container_width=True, key="generate_docs_btn"):
                if not selected_types:
                    st.error("❌ Please select at least one documentation type")
                else:
                    # Use st.status for better progress tracking
                    with st.status("🚀 Generating documentation...", expanded=True) as status:
                        try:
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            # Simple progress callback
                            def progress_callback(current, total, doc_type):
                                progress = current / total
                                progress_bar.progress(progress)
                                status_text.text(f"📝 Generating {current}/{total}: {DOC_TYPE_LABELS.get(doc_type, doc_type)}")
                            
                            st.write("📝 Analyzing codebase and generating documentation...")
                            docs = documentation.generate_all_documentation(
                                selected_types, 
                                collection_name=selected_collection,
                                progress_callback=progress_callback
                            )
                            
                            # Create ZIP file with timestamp (unique name)
                            st.write("📦 Creating ZIP file...")
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            output_path = f"kt_documentation_{timestamp}.zip"
                            documentation.create_markdown_zip(docs, output_path)
                            
                            progress_bar.progress(1.0)
                            
                            # Store results in session state
                            st.session_state.doc_generation_status = 'complete'
                            st.session_state.generated_docs = docs
                            st.session_state.doc_output_path = output_path
                            
                            status.update(label="✅ Documentation generated successfully!", state="complete", expanded=False)
                            st.balloons()
                            st.rerun()
                            
                        except Exception as e:
                            st.session_state.doc_generation_status = 'error'
                            st.error(f"❌ Error generating documentation: {str(e)}")
                            status.update(label="❌ Generation failed", state="error", expanded=False)
            
    
    # Tab 3: Chat with Codebase
    with tab3:
        st.header("💬 Chat with Codebase")
        st.markdown("Ask questions about your codebase and get AI-powered answers")
        
        # Refresh collections if not loaded
        if not st.session_state.collections:
            refresh_collections()
        
        # Collection selection
        if not st.session_state.collections:
            st.warning("⚠️ No collections found. Please index a codebase first.")
        else:
            collection_names = [c['name'] for c in st.session_state.collections]
            
            chat_collection = st.selectbox(
                "📦 Select Collection to Chat With",
                options=collection_names,
                index=collection_names.index(st.session_state.selected_collection) if st.session_state.selected_collection in collection_names else 0,
                help="Choose which codebase you want to ask questions about",
                key="chat_collection_select"
            )
            
            # Show collection info
            selected_coll_info = next((c for c in st.session_state.collections if c['name'] == chat_collection), None)
            if selected_coll_info:
                st.info(f"📊 Chatting with **{selected_coll_info['vectors_count']} files** from `{chat_collection.replace(COLLECTION_NAME_PREFIX, '')}`")
            
            st.divider()
            
            # Display chat history
            if st.session_state.chat_history:
                for i, message in enumerate(st.session_state.chat_history):
                    # User question
                    with st.chat_message("user"):
                        st.write(message["question"])
                    
                    # AI answer
                    with st.chat_message("assistant"):
                        st.markdown(message["answer"])
                        
                        # Show relevant files
                        if message.get("relevant_files"):
                            with st.expander(f"📁 Relevant Files ({message['file_count']} files)"):
                                for file_ref in message["relevant_files"]:
                                    st.markdown(f"**{file_ref['file_path']}** (score: {file_ref['score']:.3f})")
                                    st.caption(file_ref['summary'])
                                    st.divider()
            
            # Chat input
            user_question = st.chat_input("Ask a question about the codebase...")
            
            if user_question:
                # Add user message to chat
                with st.chat_message("user"):
                    st.write(user_question)
                
                # Generate response with error handling
                with st.chat_message("assistant"):
                    try:
                        with st.spinner("🤔 Analyzing codebase and generating answer..."):
                            response = chat.chat_with_codebase(user_question, chat_collection)
                        
                        # Display answer
                        st.markdown(response["answer"])
                        
                        # Show relevant files
                        if response.get("relevant_files"):
                            with st.expander(f"📁 Relevant Files ({response['file_count']} files)"):
                                for file_ref in response["relevant_files"]:
                                    st.markdown(f"**{file_ref['file_path']}** (score: {file_ref['score']:.3f})")
                                    st.caption(file_ref['summary'])
                                    st.divider()
                        
                        # Save to chat history immediately after successful generation
                        st.session_state.chat_history.append({
                            "question": user_question,
                            "answer": response["answer"],
                            "relevant_files": response.get("relevant_files", []),
                            "file_count": response.get("file_count", 0)
                        })
                        
                    except Exception as e:
                        error_msg = f"❌ Error generating response: {str(e)}"
                        st.error(error_msg)
                        
                        # Save error to chat history
                        st.session_state.chat_history.append({
                            "question": user_question,
                            "answer": error_msg,
                            "relevant_files": [],
                            "file_count": 0
                        })
                
                st.rerun()
            
            # Clear chat history button
            if st.session_state.chat_history:
                st.divider()
                if st.button("🗑️ Clear Chat History"):
                    st.session_state.chat_history = []
                    st.rerun()


if __name__ == "__main__":
    main()

import os
import sys

# 确保 Python 解释器是正确的
print(f"Using Python: {sys.executable}")

# 确保所有必要的库已安装
try:
    import streamlit as st
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_ollama import OllamaEmbeddings, OllamaLLM
    from langchain_community.vectorstores import FAISS
    from langchain.chains import RetrievalQA
    import tempfile
    import time
    import google.generativeai as genai
except ImportError as e:
    print(f"Missing dependency: {e}. Run `pip install streamlit langchain langchain-community langchain-ollama faiss-cpu pypdf`.")
    sys.exit(1)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

persona = '''
You are a helpful assistant that answers questions based on the provided documents.
Answer the question with detailed information from the documents. If the answer is not in the documents, 
say "I don't have enough information to answer this question." Cite specific parts of the documents when possible.
Consider the chat history for context when answering, but prioritize information from the documents.
'''

template = """
{persona}
        
Chat History:
<history>
{chat_history}
</history>

Given the context information and not prior knowledge, answer the following question:
Question: {user_input}
"""

# Set Google API key (replace with your key or use an env variable)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "YOUR_GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# 确保 Ollama 模型存在
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"  # Embedding model for Ollama
OLLAMA_LLM_MODEL = "deepseek-r1:1.5b"  # 修改为 deepseek-r1:1.5b

llm = OllamaLLM(model=OLLAMA_LLM_MODEL, temperature=0.5)
embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL)

st.set_page_config(page_title="Chat with Your PDFs (Ollama)")
st.title("📄💬 Chat with Your PDFs (Deepseek)")

# 文件上传
uploaded_files = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=["pdf"])
if uploaded_files:
    documents = []
    if "vector_store" not in st.session_state:
        with st.spinner("Processing your PDFs..."):
            with tempfile.TemporaryDirectory() as temp_dir:
                for file in uploaded_files:
                    temp_file_path = os.path.join(temp_dir, file.name)
                    with open(temp_file_path, "wb") as f:
                        f.write(file.getbuffer())
                    loader = PyPDFLoader(temp_file_path)
                    documents.extend(loader.load())
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                docs = text_splitter.split_documents(documents)
                st.session_state.vector_store = FAISS.from_documents(docs, embeddings)
        st.success("✅ PDFs uploaded and processed! You can now start chatting.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("Ask a question about your PDFs...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        retriever = st.session_state.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.7}
        )
        chat_history = "\n".join(
            [f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}" for msg in st.session_state.messages[:-1]]
        )
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            return_source_documents=True,
            verbose=True,
            chain_type_kwargs={"verbose": True}
        )
        with st.spinner("Thinking..."):
            response = qa_chain.invoke({
                "query": template.format(
                    persona=persona,
                    user_input=user_input,
                    chat_history=chat_history
                ),
            })
            response_text = response["result"]
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for chunk in response_text.split():
                full_response += chunk + " "
                message_placeholder.markdown(full_response)
                time.sleep(0.05)
        st.session_state.messages.append({"role": "assistant", "content": response_text})
else:
    st.info("Please upload PDF files to begin.)")
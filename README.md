**Overview:**
This is a Streamlit-based Document Q&A Chatbot that allows users to upload PDF documents and ask questions directly based on their content. It leverages LangChain, FAISS for vector storage, Hugging Face Embeddings for semantic text embedding, and Groq's language model to provide accurate answers rooted in document context. Upon uploading a PDF, the file is processed and converted into embeddings using a pre-trained transformer model (sentence-transformers/all-MiniLM-L6-v2), and stored in a FAISS index for efficient retrieval. The chatbot then uses Groq's llama3-8b-8192 model and a customized prompt to answer user questions based solely on document-specific information, ensuring relevant responses. Additionally, the interface displays similar document sections related to the query, providing further context. This tool is ideal for users needing efficient, interactive access to document content without manually searching through large files.

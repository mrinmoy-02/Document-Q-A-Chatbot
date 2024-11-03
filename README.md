This project is a Streamlit-based Document Q&A Chatbot that allows users to upload a PDF document and ask questions directly based on its contents. It uses LangChain, FAISS for vector storage, and Hugging Face Embeddings for semantic vectorization, enabling accurate question-answering based on document context. The chatbot leverages Groq's LLM to provide detailed, context-based responses.

Key Features
PDF Document Upload: Users can upload PDF documents directly through the sidebar.
Automated Document Analysis: Upon document upload, the file is processed and converted into vector embeddings using a pre-trained transformer model.
Vector Storage with FAISS: Extracted text is vectorized and stored in a FAISS index, allowing for efficient retrieval of relevant document segments.
Interactive Q&A Interface: Users can enter questions related to the document, which are then answered by the language model based on the document context.
Document Similarity Display: The chatbot displays similar document sections relevant to the user's query, providing additional context.
 
 

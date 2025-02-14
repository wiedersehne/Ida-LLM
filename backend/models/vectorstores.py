from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.schema import Document
from langchain.storage import InMemoryStore
import pickle
import os

load_dotenv()

## Split the pages / text into chunks
def create_vectorstore(pages, chunk_size, chunk_overlap, embeddings):
    # Create a text splitter with specified chunk size and overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,  # Adjust based on table size
        chunk_overlap=chunk_overlap
    )
    # Split the input pages into smaller document chunks
    documents = text_splitter.split_documents(pages)
    # Create a FAISS vector store from the documents using the embeddings
    vectorstore_faiss = FAISS.from_documents(documents, embeddings)

    docstore = InMemoryStore()
    for doc in pages:
        docstore.mset([(doc.metadata["pdf_filename"], doc.page_content)])
    # Define the filename for saving the vector store
    file_name = f"{os.getenv('INDEX_NAME')}"
    # Define the folder path where to save the vector store
    folder_path = "./backend/index"
    # Save the FAISS vector store to local disk
    vectorstore_faiss.save_local(index_name=file_name, folder_path=folder_path)

    with open(os.getenv("DOCSTORE_SAVE_PATH"), "wb") as f:
        pickle.dump(docstore, f)
    print(f"✅ Docstore saved at `{os.getenv('DOCSTORE_SAVE_PATH')}`")

    return True
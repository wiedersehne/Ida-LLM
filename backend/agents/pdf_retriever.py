from langchain_community.vectorstores import FAISS
from backend.models.llm import get_llm
from backend.models.embeddings import get_embeddings
from langchain.prompts import PromptTemplate
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchRun
from backend.loaders.pdf_loader import get_pages
from dotenv import load_dotenv
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import HumanMessage, AIMessage
from langchain.storage import InMemoryStore
from langchain.retrievers import MultiVectorRetriever
from backend.loaders.pdf_loader import get_pages_by_upload
import pickle
import os
load_dotenv()

class BaseRetriever:
    def __init__(self):
        # Initialize embeddings using the provider specified in environment variables
        self.embeddings = get_embeddings(provider=os.getenv("DEFAULT_EMBEDDING_PROVIDER"))
        # Initialize the language model with the specified provider and temperature
        self.llm = get_llm(provider=os.getenv("DEFAULT_LLM_PROVIDER"), temperature=0)
        # Initialize conversation memory to store chat history
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        # Placeholder for PDF-specific retriever
        self.pdf_retriever = None
        # Initialize in-memory store for metadata
        self.metadata_store = InMemoryStore()
    
    def process_pdf(self, pdf_path):
        # Load pages from the PDF file
        pages = get_pages_by_upload(pdf_path)
        # Initialize a text splitter to split the documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
        # Split the loaded pages into smaller documents
        documents = text_splitter.split_documents(pages)
        # Create a FAISS vector store from the documents using embeddings
        pdf_vectorstore = FAISS.from_documents(documents, self.embeddings)
        # Set the PDF-specific retriever to the created vector store
        self.pdf_retriever = pdf_vectorstore

    def create_retriever(self):
        # Load the FAISS vector store from local storage
        vectorstore = FAISS.load_local(index_name=os.getenv("INDEX_NAME"), 
                                       folder_path=os.getenv("INDEX_DIRECTORY"),
                                       embeddings=self.embeddings,
                                       allow_dangerous_deserialization=True)
        # Load the document store from a pickle file
        with open(os.getenv("DOCSTORE_SAVE_PATH"), "rb") as f:
            docstore = pickle.load(f)
        # Get the keys from the document store
        stored_keys = list(docstore.yield_keys())
        print(f"✅ Docstore loaded with {len(stored_keys)} entries")
        
        # Return a MultiVectorRetriever initialized with the vector store and document store
        return MultiVectorRetriever(vectorstore=vectorstore, docstore=docstore, id_key="pdf_filename")
        
    def retrieval_tool(self, query, pdf_only=False):
        # Check if a PDF-specific retriever is available
        if self.pdf_retriever:
            # Perform similarity search using the PDF-specific retriever
            results = self.pdf_retriever.similarity_search(query, k=3)
        else:
            # Perform similarity search using the general retriever
            results = self.vectorstore.similarity_search(query, k=3)
            # Sort results by similarity score (higher similarity = lower score value)
        
        # Combine the content of the retrieved documents into a single context string
        context = "\n".join([doc.page_content for doc in results])
        
        # Create a prompt template for generating a response
        prompt = PromptTemplate(
            input_variables=["query"],
            template=("You are an assistant for question-answering tasks. "
            "Based on the following context:\n\n"
            "Current Context:\n{context}\n\n"
            "User Query: {query}\n\n"
            "Generate a concise answer. If you do not know the answer, you can search for it."
            ),
        )
        
        # Format the prompt with the context and user query
        formatted_prompt = prompt.format(context=context, query=query)
        
        # Invoke the language model with the formatted prompt to generate a response
        response = self.llm.invoke(formatted_prompt)
        
        # Save the query and response to the conversation memory
        self.memory.save_context({"input": query}, {"output": str(response)})
        
        # Return the generated response
        return response
    
    def run(self, query, pdf_path=None):
        # Ensure memory is properly loaded
        if not hasattr(self, "memory") or self.memory is None:
            self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, input_key="input")

        # Create list of tools for the agent
        tools = [
            Tool(name="Retriever", func=self.retrieval_tool, description="Search for relevant information and generate answer"),
            # WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        ]
        
        # Initialize the agent with tools and memory
        react_agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,  # Conversational memory-enabled agent
            verbose=True,
            memory=self.memory,
        )

        # Ensure both input and chat_history are provided
        response = react_agent.invoke({"input": query, "chat_history": self.memory.load_memory_variables({}).get("chat_history", [])})

        return response["output"]
    
    def run_query(self, query, pdf_path=None):
        # Retrieve context using PDF or general retriever
        if self.pdf_retriever:
            results = self.pdf_retriever.similarity_search_with_score(query, k=3)  # ✅ Use scores for sorting
        else:
            retriever = self.create_retriever()
            print(f"Number of documents in FAISS: {len(retriever.vectorstore.index_to_docstore_id)}")
            results = retriever.vectorstore.similarity_search_with_score(query, k=5)  # ✅ Use correct function

        # ✅ Sort results by similarity score (higher score = better match)
        sorted_results = sorted(results, key=lambda x: x[1], reverse=True)  # ✅ Fix: Sort by high similarity

        # Extract ranked documents
        documents = [doc for doc, score in sorted_results]  # ✅ Extract only documents
        context = "\n\n".join([doc.page_content for doc in documents])

        # Generate citations with scores
        citations = "\n".join([
            f"Source: {doc.metadata.get('pdf_filename', 'Unknown File')}, Page {doc.metadata.get('page_number', 'Unknown Page')} (Score: {score:.4f})"
            for doc, score in sorted_results
        ])

        # Load chat history from the memory storage
        chat_history = self.memory.load_memory_variables({})
        history_str = self.format_history(chat_history.get("chat_history", []))

        # Create a prompt template for generating a response including history
        prompt_template = (
            "You are an assistant for question-answering tasks. "
            "Based on the following context and chat history:\n\n"
            "Chat History:\n{history}\n\n"
            "Current Context:\n{context}\n\n"
            "User Query: {query}\n\n"
            "Generate a concise answer. If you do not know the answer, you can search for it."
        )
        prompt = PromptTemplate(
            input_variables=["history", "context", "query"],
            template=prompt_template
        )

        # Format the prompt with the context, history, and user query
        formatted_prompt = prompt.format(history=history_str, context=context, query=query)

        # Directly invoke the language model with the formatted prompt
        response = self.llm.invoke(formatted_prompt)

        # Save the query and the response to the conversation memory
        self.memory.save_context({"input": query}, {"output": str(response)})

        # Return the generated response along with references
        return f"{response.content}\n\n📖 References:\n{citations}"

    
    def format_history(self, history):
        # Format the chat history, deducing the role based on message type
        formatted_history = []
        for msg in history:
            if isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            else:
                raise ValueError("Unknown message type in history.")
            formatted_history.append(f"{role}: {msg.content}")
        return "\n".join(formatted_history)

if __name__ == "__main__":
    # Initialize the BaseRetriever class
    retriever = BaseRetriever()
    # Example query to test the agent
    query = "What is well guide?"
    # Execute the agent with the query
    response = retriever.run(query)
    # Print the response
    print(response)
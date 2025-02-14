import gradio as gr
from backend.agents.pdf_retriever import BaseRetriever
from langchain.schema import AIMessage, HumanMessage
import os
from dotenv import load_dotenv

load_dotenv(override=True)

# Initialize your retriever instance
retriever = BaseRetriever()

uploaded_pdf = None  # Global variable to store the uploaded file path

def upload_pdf(files):
    global uploaded_pdf
    uploaded_pdf = [file.name for file in files]
    print(uploaded_pdf)
    retriever.process_pdf(uploaded_pdf)  # Process the PDF without triggering retrieval  # Ensure retriever can handle PDF ingestion
    return gr.update(value=True)  # Indicating successful upload


def chat(user_input, history=[]):
    history_langchain_format = []
    for msg in history:
        if msg['role'] == "user":
            history_langchain_format.append(HumanMessage(content=msg['content']))
        elif msg['role'] == "assistant":
            history_langchain_format.append(AIMessage(content=msg['content']))
    history_langchain_format.append(HumanMessage(content=user_input))
    
    # Retrieve response, checking if a PDF has been uploaded
    response = retriever.run_query(user_input, pdf_path=uploaded_pdf if uploaded_pdf else None)
    
    return response

# Create the Gradio chat interface with file upload
demo = gr.Blocks()
with demo:
    gr.Markdown("# Chat with Ida: Upload and Query PDFs")
    with gr.Row():
        upload_button = gr.File(label="Upload PDF", type="filepath", file_count="multiple")
        upload_status = gr.Textbox(label="Upload Status", interactive=False)
        upload_button.upload(upload_pdf, inputs=[upload_button], outputs=[upload_status])
    
    chatbot = gr.ChatInterface(
        fn=chat,
        type="messages",
        title="Chat with Ida",
        description="A chat interface powered by BaseRetriever for answering queries.",
        chatbot=gr.Chatbot(height=600),
        textbox=gr.Textbox(placeholder="Ask me anything about the uploaded PDF or general knowledge", container=False, scale=7),
        theme="ocean",
        examples=["What is WellGuide?", "Summarize the key points in the uploaded PDF", "What is the purpose of the Bottom Hole Assembly (BHA) in drilling?"],
        cache_examples=True,
    )

# Launch the Gradio app
if __name__ == "__main__":
    demo.launch(
        server_name=os.getenv("UVICORN_HOST"), 
        server_port=int(os.getenv("UVICORN_PORT"))
    )

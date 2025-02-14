import os
from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredHTMLLoader
from typing import Any
from pydantic import BaseModel
from unstructured.partition.pdf import partition_pdf
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from backend.models.llm import get_llm
import uuid
import pytesseract
from PIL import Image
from langchain.vectorstores import FAISS
import pdfplumber
from langchain_chroma import Chroma
import pymupdf4llm
from langchain.storage import InMemoryStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
from dotenv import load_dotenv
from paddleocr import PaddleOCR
from pdf2image import convert_from_path
import numpy as np
import camelot

load_dotenv()  # Load environment variables from a .env file

class Element(BaseModel):
    type: str  # Define the type attribute as a string
    text: Any  # Define the text attribute as any type

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang="en")

def extract_tables_from_pdf(pdf_path):
    """Extracts tables from the PDF, summarizes them, and stores structured data for indexing."""
    extracted_tables = []  # Initialize an empty list to store extracted tables

    try:
        # Extract tables using Camelot
        tables = camelot.read_pdf(pdf_path, pages="all", flavor="stream")  # Use "lattice" if grid lines exist
        if len(tables) == 0:
            print("❌ No tables found using Camelot.")
            return extracted_tables  # Return empty list if no tables found

        for i, table in enumerate(tables):
            df = table.df.fillna("")  # Convert to DataFrame & handle empty cells

            print(f"📄 Extracting table from Page {i+1}...")
            print("/////////////////////////////////////////////////")
            print(df)

            # Convert table to Markdown format for structured indexing
            markdown_table = df.to_markdown(index=False)

            # Store summarized table data
            extracted_tables.append(Document(
                page_content=markdown_table,
                metadata={"pdf_filename": os.path.basename(pdf_path), "page_number": i + 1, "source": "Camelot"}
            ))

        print(f"✅ Extracted and summarized {len(tables)} structured tables using Camelot")

    except Exception as e:
        print(f"⚠️ Error extracting tables from PDF: {e}")

    return extracted_tables  # Return the list of extracted tables

def extract_text_from_pdf(pdf_path):
    """Extracts regular text from the PDF with page numbers."""
    loader = PyPDFLoader(pdf_path)  # Initialize PyPDFLoader with the PDF path
    documents = loader.load()  # Load the PDF documents
    extracted_text = []  # Initialize an empty list to store extracted text

    for i, doc in enumerate(documents):
        if doc.page_content != None:
            print("************", doc.page_content)
            extracted_text.append(Document(
                page_content=doc.page_content,
                metadata={"pdf_filename": os.path.basename(pdf_path), "page_number": i + 1}
            ))

    # Convert PDF to images for OCR
    images = convert_from_path(pdf_path)  # Convert PDF pages to images
    for i, image in enumerate(images):
        try:
            # Convert PIL Image to numpy array (Ensure compatibility)
            img_array = np.array(image)

            # Convert grayscale images to RGB
            if len(img_array.shape) == 2:
                img_array = np.stack((img_array,) * 3, axis=-1)  # Convert grayscale to RGB

            # Ensure correct format before OCR
            if isinstance(img_array, np.ndarray):
                ocr_result = ocr.ocr(img_array, cls=True)
                
                # Extract text from OCR result
                ocr_text = "\n".join([" ".join(line[1][0] for line in page) for page in ocr_result if page])

                if ocr_text.strip():  # Store only non-empty OCR text
                    print("*******************************************************")
                    print(ocr_text)
                    extracted_text.append(Document(
                        page_content=ocr_text,
                        metadata={"pdf_filename": os.path.basename(pdf_path), "page_number": i + 1, "source": "OCR"}
                    ))

        except Exception as e:
            print(f"⚠️ Error processing image on page {i+1}: {e}")

    return extracted_text  # Return the list of extracted text

def get_pages(directory_path, structured="semi"):
    """
    Process PDF and HTML files from a directory and extract their contents into pages.
    Returns a list of Page objects containing the extracted content.
    """
    pages = []  # Initialize an empty list to store pages
    # Iterate over all files in the specified directory
    for file in os.listdir(directory_path):
        # Check if the file is a PDF by examining its extension
        if file.endswith('.pdf'):
            pdf_path = os.path.join(directory_path, file)
            if structured == "semi":
                extracted_tables = extract_tables_from_pdf(pdf_path)
                extracted_text = extract_text_from_pdf(pdf_path)
                # extracted_text = []
                print(len(extracted_text), len(extracted_tables))
                # Combine text and tables into documents
                if extracted_text and extracted_tables:
                    pages.extend(extracted_text + extracted_tables)
                elif extracted_text:
                    pages.extend(extracted_text)
                elif extracted_tables:
                    pages.extend(extracted_tables)
            elif structured == "pymupdf":
                with pdfplumber.open(pdf_path) as pdf:
                    for page_num, page in enumerate(pdf.pages):
                        md_text = pymupdf4llm.to_markdown(pdf_path, pages=[page_num])
                        pages.append(Document(page_content=md_text))

    return pages  # Return the list of pages


def get_pages_by_upload(file_names, structured="semi"):

    pages = []  # Initialize an empty list to store pages
    # Iterate over all files in the specified directory
    for file_name in file_names:
        if file_name.endswith('.pdf'):
            if structured == "semi":
                extracted_tables = extract_tables_from_pdf(file_name)
                extracted_text = extract_text_from_pdf(file_name)
                # extracted_text = []
                print(len(extracted_text), len(extracted_tables))
                # Combine text and tables into documents
                if extracted_text and extracted_tables:
                    pages.extend(extracted_text + extracted_tables)
                elif extracted_text:
                    pages.extend(extracted_text)
                elif extracted_tables:
                    pages.extend(extracted_tables)
            elif structured == "pymupdf":
                with pdfplumber.open(file_name) as pdf:
                    for page_num, page in enumerate(pdf.pages):
                        md_text = pymupdf4llm.to_markdown(file_name, pages=[page_num])
                        pages.append(Document(page_content=md_text))

    return pages  # Return the list of pages


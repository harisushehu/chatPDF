# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 15:47:14 2025

@author: HarisuShehu
"""

import os
from flask import Flask, render_template, request, jsonify
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI
from dotenv import load_dotenv
import tempfile
import numpy as np

# Load environment variables
load_dotenv("api.env")

app = Flask(__name__)

# Global variables to store the vector store and QA chain
vector_store = None
qa_chain = None

class PDFChatApp:
    @staticmethod
    def process_pdfs(pdf_files):
        """Process uploaded PDF files and create a vector store."""
        global vector_store, qa_chain
        
        documents = []
        temp_files = []  # Store temp file paths for cleanup
        
        for pdf_file in pdf_files:
            try:
                # Create temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(pdf_file.read())
                    temp_files.append(tmp_file.name)
                
                # Load PDF content
                loader = PyPDFLoader(temp_files[-1])
                documents.extend(loader.load())
            except Exception as e:
                print(f"Error processing file {pdf_file.filename}: {str(e)}")
            finally:
                # Clean up temp files
                for tmp_file in temp_files:
                    try:
                        if os.path.exists(tmp_file):
                            os.unlink(tmp_file)
                    except Exception as e:
                        print(f"Error deleting temp file {tmp_file}: {str(e)}")
        
        if not documents:
            raise ValueError("No valid content found in the uploaded PDFs")
        
        # Split documents into chunks
        text_splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.split_documents(documents)
        
        # Create embeddings and vector store
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.from_documents(texts, embeddings)
        
        # Create QA chain
        llm = OpenAI(temperature=0)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever()
        )

    @staticmethod
    def is_off_topic(question: str, context: str) -> bool:
        """Check if the question is off-topic using multiple criteria"""
        embeddings = OpenAIEmbeddings()
        
        try:
            q_embedding = embeddings.embed_query(question)
            c_embedding = embeddings.embed_query(context)
            
            # Calculate cosine similarity
            similarity = np.dot(q_embedding, c_embedding) / (
                np.linalg.norm(q_embedding) * np.linalg.norm(c_embedding))
            
            # Check if the context is too short or similarity is too low
            if similarity < 0.3 or len(context) < 50:
                return True
            
            return False
        except Exception as e:
            print(f"Error checking topic relevance: {str(e)}")
            return True

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'pdf_files' not in request.files:
            return jsonify({'error': 'No files uploaded'}), 400
        
        pdf_files = request.files.getlist('pdf_files')
        pdf_files = [f for f in pdf_files if f.filename != '']
        
        if not pdf_files:
            return jsonify({'error': 'No valid PDF files uploaded'}), 400
        
        try:
            PDFChatApp.process_pdfs(pdf_files)
            return jsonify({'message': 'PDFs processed successfully'}), 200
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask_question():
    global qa_chain, vector_store
    
    if not qa_chain:
        return jsonify({'error': 'Please upload PDF files first'}), 400
    
    data = request.get_json()
    question = data.get('question')
    
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    try:
        # Get the most relevant document chunk
        relevant_docs = vector_store.similarity_search(question, k=1)
        
        # Check if the question is relevant
        if not relevant_docs or PDFChatApp.is_off_topic(question, relevant_docs[0].page_content):
            return jsonify({
                'answer': "I'm sorry, but this question doesn't appear to be related to the content of the uploaded documents."
            }), 200
        
        # If relevant, proceed with answering
        result = qa_chain.run(question)
        return jsonify({'answer': result}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
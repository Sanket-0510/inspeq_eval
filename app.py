
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from inspeq.client import InspeqEval
import json
import os
load_dotenv()


inspeq_eval = InspeqEval(inspeq_api_key= os.getenv("INSPEQ_KEY"))

def eval_factual_consistency(docs, user_question, response):
    input_data = {
        "prompt": user_question,
        "context": docs[0].page_content,
        "response": response['output_text']
    }
    
    results_json = inspeq_eval.factual_consistency(input_data=input_data, task_name="QnA app")
    results = json.loads(results_json)

    st.write("Output Consistency")
    st.write(results)

def eval_do_not_use_keywords(docs,user_question, response):
    input_data = {
        "response": response['output_text']
    }
    
    config_input = {
       "threshold": 0.5,
       "custom_labels": ["web","react"],
       "label_thresholds": [0,1],
    }
    results = inspeq_eval.do_not_use_keywords(input_data= input_data ,config_input= config_input ,task_name="QnA_task")
    results = json.loads(results)
    st.write("Eval do not use keywords")
    st.write(results)
    
def eval_ans_relevance(docs, user_question, response):
    input_data = {
        "prompt": user_question,
        "response": response['output_text']
    }
    
    config_input = {
        "threshold": 0.5,
        "custom_labels": ["custom_label_1", "custom_label_2"],
        "label_thresholds": [0, 0.5, 1],
    }
    
    results = inspeq_eval.answer_relevance(input_data=input_data, config_input=config_input, task_name="your_task_name")
    results = json.loads(results)
    st.write("Eval Ans Relevance")
    st.write(results)



def eval_word_limt(docs, user_question, response):
    input_data = {
   "prompt": user_question,
   "response": response['output_text']
   }


    config_input= {
       "threshold": 0.5,
       "custom_labels": ["custom_label_1","custom_label_2"],
       "label_thresholds": [0,1],
   }
    
    results = inspeq_eval.word_limit_test(input_data= input_data ,config_input= config_input ,task_name="your_task_name")
    results = json.loads(results)
    st.write("Eval Word Limit")
    st.write(results)
    

def eval_response_tone(docs, user_question, response):
    input_data = {
   "response": response['output_text']
   }


    config_input= {
       "threshold": 0.5,
       "custom_labels": ["custom_label_1","custom_label_2"],
       "label_thresholds": [0,0.5, 1],
   }
    
    results = inspeq_eval.response_tone(input_data= input_data ,config_input= config_input ,task_name="your_task_name")
    results = json.loads(results)
    st.write("Eval Response Tone")
    st.write(results)
    
def eval_conceptual_similarity(docs, user_question, response):
    input_data = {
        "context":  docs[0].page_content,
        "response": response['output_text']
    }

    config_input = {
        "threshold": 0.5,
        "custom_labels": ["custom_label_1", "custom_label_2"],
        "label_thresholds": [0, 0.5, 1],
    }

    results = inspeq_eval.conceptual_similarity(input_data=input_data, config_input=config_input, task_name="your_task_name")
    results = json.loads(results)
    st.write(results)



st.title("Langchain PDF Text Analysis")
st.write("Upload a PDF file to analyze its content.")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file is not None:
    text = ""
    pdf_reader = PdfReader(uploaded_file)
    for page in pdf_reader.pages:
        text += page.extract_text()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GOOGLE_KEY"))
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

    prompt_template = """
    Answer the question as detailed as possible from the provided context.
    Context: \n{context}\n

    Question: \n{question}\n

    Answer:
    """
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=os.getenv("GOOGLE_KEY"))

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    user_question = st.text_input("Enter your question")
    

    if user_question:
        docs = new_db.similarity_search(user_question)
        response = chain(
            {
                "input_documents": docs,
                "question": user_question
            },
            return_only_outputs=True
        )
        
        st.write("Answer:", response['output_text'])
        
        eval_factual_consistency(docs, user_question, response)
        eval_do_not_use_keywords(docs, user_question, response)
        eval_ans_relevance(docs, user_question, response)
        eval_conceptual_similarity(docs, user_question, response)
        eval_word_limt(docs, user_question, response)
        eval_response_tone(docs, user_question, response)
        




    


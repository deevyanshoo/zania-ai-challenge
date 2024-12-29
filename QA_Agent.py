from doc_parser import doc_parser
import os
import tiktoken
import re
import pandas as pd
import openai
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import Document
from sentence_transformers import CrossEncoder
import fitz
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from langchain_openai import ChatOpenAI
import time
import requests
import json
from openai import OpenAI
import json
import ast
import requests
from operator import itemgetter
from langchain.retrievers import BM25Retriever as LangchainBM25
from ordered_set import OrderedSet
import os
import warnings
import json
from IPython.display import display, Markdown, Latex
from typing import List, Dict
from dotenv import load_dotenv
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
debug = True

client_open_ai = OpenAI(api_key=openai_api_key)

openai.api_key = openai_api_key

warnings.filterwarnings("ignore")

def chunk_content_data_parser_dataframe(df, chunk_length=1500, overlap_length=400):

    def chunk_content(row):
        content = row['content']
        return [
            {
            'header_id': int(row.get('header_id', -1)),
            'chunk_id': f"{int(row.get('header_id', -1))}_{j + 1}",
            'title': row.get('title', ''),
            'level': int(row.get('level', -1)),
            'start_page': int(row.get('start_page', -1)),
            'end_page': int(row.get('end_page', -1)),
            'chunked_content': content[i:i + chunk_length],
            'parent_titles': row.get('parent_titles', '')
        }
            for j, i in enumerate(range(0, len(content), chunk_length - overlap_length))
            if i < len(content)
        ]
 
    chunked_data = [chunk for _, row in df.iterrows() for chunk in chunk_content(row)]
    chunked_df = pd.DataFrame(chunked_data)
 
    return chunked_df

def create_embeddings(df_column, batch_size=2000, max_workers=10):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, model="text-embedding-3-large")
    def embed_batch(batch):
        return embeddings.embed_documents(batch)

    chunks = df_column.tolist()
    total_chunks = len(chunks)

    embedded_chunks = [None] * total_chunks

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        
        for i in range(0, total_chunks, batch_size):
            batch = chunks[i:i+batch_size]
            future = executor.submit(embed_batch, batch)
            futures.append((i, future))
 
        with tqdm(total=total_chunks, desc="Embedding chunks") as pbar:
            for i, future in futures:
                batch_embeddings = future.result()
                batch_size = len(batch_embeddings)
                embedded_chunks[i:i+batch_size] = batch_embeddings
                pbar.update(batch_size)
 
    return embedded_chunks

def create_faiss_index(df):
    embeddings_model = OpenAIEmbeddings(openai_api_key=openai_api_key, model="text-embedding-3-large")
    texts = df['chunked_content'].tolist()
    embeddings = df['embedding'].tolist()
    metadatas = [{'source': f"'Section: '{row['title']} 'Page: '{row['start_page']} "} for _, row in df.iterrows()]
    text_embeddings = list(zip(texts, embeddings))

    faiss_index = FAISS.from_embeddings(
        text_embeddings=text_embeddings,
        embedding=embeddings_model,
        metadatas=metadatas
    )

    return faiss_index

model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
def CrossEncoderRerank(query, docs_list, top_k):
    print("INSIDE RERANKER")
    pairs = [(query, docs) for docs in docs_list]
    scores = model.predict(pairs)
    result_pairs = [(docs, score) for docs, score in zip(docs_list, scores)]
    result_pairs = sorted(result_pairs, key=itemgetter(1), reverse=True)
    most_relevant_docs = [doc for doc, score in result_pairs]
    if top_k is not None:
        most_relevant_docs = most_relevant_docs[:top_k]
        
    return most_relevant_docs

def get_custom_prompt_template():

    prompt_template = """
    You are an AI assistant tasked with answering questions based on the provided document references. Your responses should be comprehensive, utilizing multiple references where relevant. If you cannot answer the question based on the references, respond with "Data Not Available"
    
    Instructions:

    Structure: Answer the following question using the provided references, ensuring each part of your answer is supported by specific REFERENCE ID from the references.
    Citations: In your response, it is critical to cite each sentence precisely using the specific REFERENCE ID provided to you in the References JSON below. Do not create any new REFERENCE ID on your own. Each REFERENCE ID must accurately refer to a specific reference from the References JSON I provide you. Accuracy is paramount — ensure that the ID used in your citations exactly match the REFERENCE ID from the References JSON.
    IMPORTANT: Only generate the REFERENCE ID in your response. Only tag the sentences in your response with relevant REFERENCE ID and do not generate your own reference list. I will know the reference by the REFERENCE ID in you will provide in your response

    The focus should be on correctly tagging the provided REFERENCE ID.

    Detail: Aim for a thorough answer that covers various aspects of the question, drawing on different references to support your statements.
    IMPORTANT: Note that each sentence should not be cited by more than 3 REFERENCE ID. Do not miss this detail.
    Conciseness: While being detailed, keep your response concise and focused solely on the question.


    Question: {question}

    References JSON:
    {context}

    Please provide a structured and informative answer with REFERENCE ID.
    """

    return PromptTemplate(input_variables=["question", "context"], template=prompt_template)

class BM25_Retriever:

    def __init__(self, df, top_k=100):
        self.df = df
        self.top_k = top_k  

    def bm25_retriever(self, query):

        documents = [Document(metadata={'source': f"'Section_bm25: '{row['title']} 'Page: '{row['start_page']} "}, page_content=row['chunked_content']) for _, row in self.df.iterrows()]
 
        bm25_retriever = LangchainBM25.from_documents(documents)
        bm25_retriever.k = self.top_k
        retrieved_docs = bm25_retriever.get_relevant_documents(query)

        return retrieved_docs

    def run(self, query):
        relevant_docs = self.bm25_retriever(query)
        return relevant_docs
    
def process_documents_pipeline(pdf_path):

    files_data_dict = {}

    for doc_id, file_path in enumerate(os.listdir(pdf_path)):
        absolute_file_path = os.path.abspath(os.path.join(pdf_path, file_path))
        file_name = file_path.split('\\')[-1]
        nested_toc, toc_df = doc_parser(absolute_file_path)
        toc_df['file_name'] = file_name
        files_data_dict[file_name] = {'nested_toc': nested_toc,
                                     'toc_df':toc_df}

    combined_toc_df = pd.concat([files_data_dict[file_name]['toc_df']
                                 for file_name in files_data_dict.keys()], ignore_index=True)

    chunked_df = chunk_content_data_parser_dataframe(combined_toc_df)
    chunked_df['embedding'] = create_embeddings(chunked_df['chunked_content'])
    faiss_index = create_faiss_index(chunked_df)

    return chunked_df,faiss_index

def prompt_4o():

    prompt_template = f"""
    You are an AI assistant tasked with answering questions based on the provided document references. Your responses should be comprehensive, utilizing multiple references where relevant. If you cannot answer the question based on the references, respond with "Data Not Available"
 
    Instructions:

    Structure: Answer the following question using the provided references, ensuring each part of your answer is supported by specific REFERENCE ID from the references.
    Citations: In your response, it is critical to cite each sentence precisely using the specific REFERENCE ID provided to you in the References JSON below. Do not create any new REFERENCE ID on your own. Each REFERENCE ID must accurately refer to a specific reference from the References JSON I provide you. Accuracy is paramount — ensure that the ID used in your citations exactly match the REFERENCE ID from the References JSON.

    IMPORTANT: Only generate the REFERENCE ID in your response. Only tag the sentences in your response with relevant REFERENCE ID and do not generate your own reference list. I will know the reference by the REFERENCE ID in you will provide in your response
    Format : Please ensure the each reference id number is mentioned invidiually in the response at the end of the sentence i.e, it should be like [1], [3], etc

    The focus should be on correctly tagging the provided REFERENCE ID.
 
    Detail: Aim for a thorough answer that covers various aspects of the question, drawing on different references to support your statements.
    IMPORTANT: Note that each sentence should not be cited by more than 3 REFERENCE ID. Do not miss this detail.
    Conciseness: While being detailed, keep your response concise and focused solely on the question.
    """
    return prompt_template

def answer_question_normal(faiss_index, df, question, pdfs_path,top_faiss=50, top_bm=50, rerank  = True, normal_mode= True): 
    start_time = time.perf_counter()
    docs_faiss = faiss_index.similarity_search(question, k=top_faiss)
    bm25_ret =  BM25_Retriever(df=df, top_k=top_bm)
    bm25_relevant_docs = bm25_ret.run(question)
 
    time_to_retrieve_chunks = round(time.perf_counter() - start_time, 3)
 
    bm25_chunks = []
    faiss_chunks = []

   
    for i, doc in enumerate(docs_faiss):
        file_name = doc.metadata['source']
        chunk_text = doc.page_content
        faiss_chunks.append({
            'chunk': chunk_text,
            'citation_number': i + 1
        })

    fais_len=len(docs_faiss)    

    for i, doc in enumerate(bm25_relevant_docs):
        file_name = doc.metadata['source']
        chunk_text = doc.page_content
        bm25_chunks.append({
            'chunk': chunk_text,
            'citation_number': fais_len + i + 1
        })
 
    relevant_documents_bm25 = [f"{chunk['chunk']}" for chunk in bm25_chunks]
   
    relevant_documents_faiss = [f"{chunk['chunk']}" for chunk in faiss_chunks]

    relevant_chunks_test = relevant_documents_faiss + relevant_documents_bm25

    most_relevant_documents = relevant_chunks_test

    rerank_start = time.perf_counter()

    if rerank:
        try:
            most_relevant_documents = CrossEncoderRerank(question, most_relevant_documents, top_k=top_bm+top_faiss)
        except Exception as e:
            print(e)
            print("Cross Encoder not working")

    time_to_rerank = round(time.perf_counter() - rerank_start, 3)
 
    if debug:
        print("Time to retrieve: ", time_to_retrieve_chunks)
        print("Time to rerank: ", time_to_rerank)
       
    most_relevant_documents_reordered_citation = []

    most_relevant_documents_ordered_set = OrderedSet(most_relevant_documents)

    for i, chunk in enumerate(most_relevant_documents_ordered_set):
        most_relevant_documents_reordered_citation.append({
            'REFERENCE ID': i+1,
            'CONTENT': chunk,
        })

    time_to_first_token = round(time.perf_counter() - start_time, 3)

    system_prompt_4o = prompt_4o()
 
    if True:
        print("Time to First Token: ",time_to_first_token)
       
    response_openai = client_open_ai.chat.completions.create(
        model='gpt-4o-mini',
        messages=[
            {
                "role": "system",
                "content": system_prompt_4o
            },
            {
                "role": "user",
                "content": f"""Your job is answer the query using the provided References JSON and refering the correct REFERENCE IDs.
                Question: {question}
                References JSON:
                {json.dumps(most_relevant_documents_reordered_citation, indent=2)}
                """
            }
        ]
    )

   
    print("Answer: \n")
    print(response_openai.choices[0].message.content)
    cited_numbers_response_openai = list({int(num.strip()) for match in re.findall(r'\[([\d,\s]+)\]', response_openai.choices[0].message.content) for num in match.split(',') if num.strip().isdigit()})

    openai_references = []
    most_relevant_documents_ordered_set = [f"[{i+1}] {x}" for i, x in enumerate(list(most_relevant_documents_ordered_set))]

    for openai_ref in cited_numbers_response_openai:
        openai_references.append(most_relevant_documents_ordered_set[openai_ref-1])

    cited_chunks_text_openai = "\n\nCited Reference:\n\n" + "\n\n".join(openai_references)
    
    return response_openai.choices[0].message.content + cited_chunks_text_openai


def answer_questions( questions,faiss_index,df,pdfs_path):

    answers = {}
    for question in questions:
        answer = answer_question_normal(faiss_index=faiss_index, df=df,question=question,pdfs_path=pdfs_path, normal_mode=True)
        answers[question] = answer.strip()
    return answers

def main():
    # Inputs
    pdfs_path = './docs'
    df, faiss_index = process_documents_pipeline('./docs')

    print("Please enter comma separated list of questions!")
    questions_input = input()
    questions = questions_input.split(',')

    # Step 2: Answer questions based on PDF content
    print("Answering questions...")
    answers = answer_questions(questions,faiss_index,df,pdfs_path)

    # Step 3: Output results as structured JSON
    output = json.dumps(answers, indent=4)
    print("Results:\n", output)

    # Optional: Save results to a file
    with open("results.json", "w") as f:
        f.write(output)

if __name__ == "__main__":
    main()

# PDF Question Answering Agent
This project demonstrates an AI agent capable of answering questions based on the content of a large PDF document. The implementation leverages OpenAI's GPT-4o-mini model to extract accurate answers with citations from PDF content.

## Features
- **Structure Preserving PDF Text Extraction**: Extracts the content from a PDF and stores relevant information in a tree based format to chunk similar content together
- **Context Augmentation**: Additional context has been added to each chunk for richer retrieval
- **Hybrid Retrieval**: Both semantic and keyword based retrieval followed by reranking for richer retrieval
- **Question Answering**: Answers questions using GPT-4o-mini. Low-confidence responses are handled with "Data Not Available."
- **JSON Output**: Generates a structured JSON blob mapping questions to answers.

## How It Works
1. **User Input**: Upload a PDF file in docs folder and add list of questions.
2. **Processing**: The Agent parses the document and preserves its structure. There is a hybrid retrieval on the list of questions which extracts the information using both semantic search and keyword matching. Final 50 chunks are sent to the LLM for the final answer generation.
3. **Output**: Results are returned as a structured JSON file.

![image](https://github.com/user-attachments/assets/906df44e-2ead-4ce3-a6eb-08df3649828e)


## Prerequisites
- Python 3.8 or later
- OpenAI API key

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/deevyanshoo/zania-ai-challenge.git
   cd zania-ai-challenge
2. Create and save a .env file in the root directory:
    OPEN_API_KEY = "your api key"
3. Install all the dependencies mentioned in requirements.txt file

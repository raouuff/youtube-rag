# Building a RAG application from scratch

A step-by-step Jupyter notebook ([`rag.ipynb`](rag.ipynb)) that builds a RAG (Retrieval-Augmented Generation) app to ask questions about any YouTube video, using LangChain, Pinecone, and Google's Gemini (`gemini-3.1-flash-lite`).

## How it works

1. **Fetch the transcript** — Get the video's transcript directly from YouTube with [youtube-transcript-api](https://github.com/jdepoix/youtube-transcript-api). No audio download or speech-to-text model needed.
2. **Chunk** — Split the transcription into overlapping chunks (`RecursiveCharacterTextSplitter`).
3. **Embed & store** — Embed the chunks with Google's `text-embedding-004` (768 dimensions) and store them in a vector store (in-memory `DocArrayInMemorySearch` for demos, [Pinecone](https://www.pinecone.io/) at scale).
4. **Retrieve & answer** — Find the chunks most similar to the question and pass them as context to Gemini through a LangChain chain.

Along the way, the notebook also covers LangChain fundamentals: chaining models with output parsers, prompt templates, and combining chains (e.g., translating the answer to another language).

The fetched transcript is cached in `transcription.txt`, so the YouTube fetch only runs when that file doesn't exist yet — delete it to re-fetch or switch videos.

## Setup

1. Create a virtual environment and install the required packages:

```bash
$ python3 -m venv .venv
$ source .venv/bin/activate
$ pip install -r requirements.txt
```

2. Create a free [Pinecone](https://www.pinecone.io/) account and get your API key, and a Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey).

3. Create a `.env` file with the following variables:

```bash
GOOGLE_API_KEY = [ENTER YOUR GEMINI API KEY HERE]
PINECONE_API_KEY = [ENTER YOUR PINECONE API KEY HERE]
PINECONE_API_ENV = [ENTER YOUR PINECONE API ENVIRONMENT HERE]
```

> Note: the embeddings are 768-dimensional, so your Pinecone index must be created with dimension `768`.

4. Run the notebook:

```bash
$ jupyter notebook rag.ipynb
```

> Run the cells in order (Restart & Run All). Later sections reuse variables like `chain` defined in earlier sections, so running cells out of order causes confusing errors.
# youtube-rag

# chroma-rag-colab

A Colab notebook with a Qwen chat model, file retrieval (RAG) on Chroma, and web search, served by a FastAPI endpoint and a Gradio UI.

## Components

| Part | Detail |
|---|---|
| Model | `Qwen/Qwen2.5-Coder-7B-Instruct`, 4-bit (bitsandbytes, nf4), free Colab GPU |
| RAG | LangChain `PyPDFLoader` / `TextLoader`, `RecursiveCharacterTextSplitter`, `sentence-transformers/all-MiniLM-L6-v2` embeddings, `Chroma.from_documents`, `similarity_search` for the top matches |
| Web search | DuckDuckGo search tool, switched on per request |
| API | FastAPI `POST /chat` with `{prompt, use_web}`, exposed through ngrok |
| UI | Gradio: file upload, "Process Files" button, chat, internet-search checkbox |

## Run output (saved in the notebook)

- Model load, FastAPI server and Gradio UI start.
- An API call with web search returns an answer (a Bitcoin price query).

## Run

Open the notebook in Colab with a GPU runtime, paste your own ngrok token where marked (`PASTE_YOUR_NGROK_TOKEN_HERE`) and run the cells in order. Library versions are pinned in the first cell.

# DoorDashGPT — Intelligent DoorDash Support Assistant

DoorDashGPT is an AI-powered assistant built with **LangGraph** + **LlamaIndex** that answers DoorDash-related queries accurately using a **RAG** (Retrieval-Augmented Generation) setup.

---

## 🚀 Features

- **DoorDash-focused RAG** → Uses vector search to retrieve relevant chunks from stored documentation.
- **Source-backed answers** → Always provides source links and page links when available.
- **Accurate & safe** → Does **not** invent information; responds only from indexed docs.
- **Tool-integrated** → Uses a knowledge base query tool to fetch context dynamically.

---

## 🛠️ Tech Stack

- **LangGraph** — Orchestrates conversational flow.
- **LlamaIndex** — Handles document retrieval & response synthesis.
- **PGVector + Postgres** — Stores and indexes embeddings.
- **Google Gemini API** — Provides embeddings + LLM.
- **Groq llama-3.1-8b-instant** - Provides Metadata Questions and Answers
- **Python 3.10+**

---

## Configure environment variables

### Create a .env file in the root directory:

- POSTGRES_CONNECTION_STRING=postgresql+psycopg2://user:password@localhost:5432/dbname
- POSTGRES_DB_NAME=doordash_rag
- GOOGLE_API_KEY=your_google_api_key
- GROQ_API_KEY=your_groq_api_key

---

🧩 How It Works

User asks a question → Sent to LangGraph.

Knowledge Base Tool → Queries LlamaIndex for top 50 chunks.

LLM synthesis → Gemini combines chunks into a concise, human-friendly answer.

Returns → Final response + source links.

---

⚠️ Notes

Strictly answers DoorDash-related queries.

If no relevant information exists, it clearly states so.

Includes direct links whenever available.

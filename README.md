# AI Travel Planning Assistant — Singapore

A context-aware travel assistant that combines **RAG** (document-based knowledge base) with **MCP tools** (live weather and currency data), built with **LangChain**.
---

## Artifact

### Artifact Link 
https://drive.google.com/file/d/1ENBD5rsCBQ7fhkV2ZY2-_jlnyGybbWXb/view?usp=sharing

### Video Link
https://drive.google.com/file/d/1WDc5nrhB8_3hbWjona-KnBS8NtWByMmB/view?usp=sharing

### Repo Link
https://github.com/KumVivekAI/travel_planning 

## Architecture

```
User (Streamlit UI)
        │
        ▼
LangChain Agent (GPT-4o-mini)
        │
   ┌────┴────────────────┐
   ▼                     ▼
RAG Pipeline          MCP Tools
   │                     │
   ├─ Load MD docs       ├─ Weather (Open-Meteo)
   ├─ Chunk (800/150)    └─ Currency (Frankfurter)
   ├─ Embed (MiniLM)
   ├─ FAISS store
   └─ Similarity search
```

### Information Flow

| Query Type | Source | Example |
|------------|--------|---------|
| Destination facts | RAG (Knowledge Base) | "Must-visit attractions?" |
| Weather / forecast | MCP Weather Tool | "Rain expected tomorrow?" |
| Currency / budget | MCP Currency Tool | "Convert INR 50,000 to SGD" |
| Combined | RAG + MCP + LLM | "3-day itinerary adjusted for weather" |

---

## Knowledge Base Sources

| # | Source | File | URL |
|---|--------|------|-----|
| 1 | Wikivoyage Singapore | `01_wikivoyage_singapore.md` | https://en.wikivoyage.org/wiki/Singapore |
| 2 | Visit Singapore Essential Info | `02_visit_singapore_essential.md` | https://www.visitsingapore.com/travel-tips-travelling-to-singapore/about-singapore/ |
| 3 | Visit Singapore Itineraries | `03_visit_singapore_itineraries.md` | https://www.visitsingapore.com/content/desktop/en/things-to-do/itineraries.html |
| 4 | Visit Singapore Things to Do | `04_visit_singapore_things_to_do.md` | https://www.visitsingapore.com/content/desktop/en/things-to-do.html |

Content is stored as Markdown with `source_title` and `source_url` metadata for citations.

---

## RAG Workflow

1. **Load** — Markdown files from `data/knowledge_base/`
2. **Chunk** — `RecursiveCharacterTextSplitter` (800 chars, 150 overlap)
3. **Embed** — `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace)
4. **Store** — FAISS vector index in `data/vector_store/`
5. **Retrieve** — Top-5 semantic similarity search per query
6. **Generate** — LLM answers grounded in retrieved chunks with source references

---

## MCP Tools

The application connects to MCP servers using **`langchain-mcp-adapters`** (`MultiServerMCPClient`, stdio transport). See `src/mcp_client.py`.

```
Streamlit UI → assistant.py → MCP client (stdio) → FastMCP servers → external APIs
```

### Weather Server (`mcp_servers/weather_server.py`)
- **API:** Open-Meteo (free, no API key)
- **Destination:** Singapore (1.3521°N, 103.8198°E)
- **Returns:** Day-wise temperature, precipitation, conditions, rain flag

### Currency Server (`mcp_servers/currency_server.py`)
- **API:** Frankfurter (`api.frankfurter.dev`, free, no API key)
- **Returns:** Live exchange rates with conversion date

### Running MCP Servers Standalone

```bash
python mcp_servers/weather_server.py
python mcp_servers/currency_server.py
```

The Streamlit app invokes these servers through the MCP client on each weather/currency request. If MCP transport fails, the assistant reports the failure and may fall back to the same API logic directly (without fabricating data).

---

## Context Strategy

- **Multi-turn memory:** The last **6** chat turns are injected into the combined prompt as conversation history.
- **User preferences** (family, budget, dates) are preserved in that history for follow-up questions.
- **RAG retrieval** runs on every query to ground destination facts.
- **Intent routing** selects weather and/or currency MCP tools using keyword detection and `src/currency_parser.py` (supports amounts, codes, and aliases such as “Singapore dollars”).

---

## Prompt Strategy

The system prompt instructs the LLM to:

1. Use **knowledge base** for destination facts only
2. Use **MCP tools** for time-sensitive weather and currency data
3. **Never fabricate** facts, rates, or forecasts
4. **Label sections:** From Knowledge Base | From MCP Tools | AI Recommendations
5. **Preserve conversation context** (family, budget, dates)
6. **Adjust itineraries** based on weather (indoor alternatives on rainy days)

See `src/prompts.py` for full templates.

---

## Project Structure

```
travel_planning_assistant/
├── app.py                          # Streamlit UI
├── requirements.txt
├── .env.example
├── data/
│   └── knowledge_base/             # 4 Singapore travel guides
├── data/vector_store/              # FAISS index (generated)
├── mcp_servers/
│   ├── weather_server.py
│   └── currency_server.py
├── scripts/
│   ├── build_kb.py
├── src/
│   ├── config.py
│   ├── prompts.py
│   ├── mcp_client.py
│   ├── currency_parser.py
│   ├── rag/
│   │   ├── ingestion.py
│   │   └── retriever.py
│   ├── mcp_tools/
│   │   ├── weather.py
│   │   └── currency.py
│   └── agent/
│       └── assistant.py
└── docs/
    ├── SAMPLE_QA.md
    └── DEMO.md
```

---

## Setup

### 1. Create virtual environment

```bash
cd travel_planning_assistant
python -m venv venv
venv\Scripts\activate        # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure OpenAI

The assistant uses **OpenAI** for answers (`langchain-openai` / `gpt-4o-mini` by default). Weather and currency use free public APIs; only the LLM step needs a paid OpenAI key.

```bash
copy .env.example .env
```

Edit `.env` and set your key:

```env
OPENAI_API_KEY=sk-proj-...
```

### 4. Build the knowledge base

```bash
python scripts/build_kb.py
```

### 5. Run the application

```bash
python -m streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## Demo Scenarios

| Scenario | What to Show |
|----------|--------------|
| **RAG** | "What are must-visit attractions?" → KB sources cited |
| **MCP Weather** | "Weather forecast for 3 days?" → Open-Meteo data |
| **MCP Currency** | "Convert INR 50,000 to SGD" → Frankfurter rate |
| **Combined** | "3-day itinerary adjusted for weather" → KB + weather + AI plan |
| **Multi-turn** | Ask about family trip, then follow up about rain |

See `docs/SAMPLE_QA.md` for example questions and `docs/DEMO.md` for a short demo script.

---

## Acceptance Criteria Checklist

- [x] Knowledge base from 3+ travel resources
- [x] Embedding-based semantic retrieval (FAISS + MiniLM)
- [x] Grounded answers with source references
- [x] Weather via MCP tool (Open-Meteo)
- [x] Currency conversion via MCP tool (Frankfurter)
- [x] Combined RAG + MCP response
- [x] Multi-turn conversation with context
- [x] Intent-based tool selection
- [x] Missing knowledge / tool failure handling
- [x] Simple Streamlit UI

---

## Technology Stack

| Component | Choice |
|-----------|--------|
| Orchestration | LangChain |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector Store | FAISS |
| LLM | OpenAI GPT-4o-mini |
| MCP Client | `langchain-mcp-adapters` (stdio) |
| MCP Servers | `mcp` (FastMCP) |
| UI | Streamlit |

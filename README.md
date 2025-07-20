# Research Assistant with LangChain, Groq, and Google Search

This is a Python-based research assistant that uses LangChain agents, Groq-powered LLaMA3 model, and Google Search to fetch and summarize recent academic research.

## 🚀 Features

- 🔍 Google Search integration for academic queries
- 🧠 Summarization using LLaMA3 via Groq API
- 🧮 Calculator tool
- 🤖 Agent orchestration using LangChain

## 🛠️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/research-assistant.git
cd research-assistant

```
### 2. Create a .env file
Add the following to a .env file in the root directory:
```
GROQ_API_KEY=your_groq_api_key
GOOGLEAPI_KEY=your_google_api_key
GOOGLECSE_ID=your_custom_search_engine_id
```
### 3. Install dependencies:
```
pip install -r requirements.txt
```
### 4. Run the script:
```
python research_aast.py
```


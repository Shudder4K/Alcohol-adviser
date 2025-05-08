
# Cocktail Advisor Chat

A simple FastAPI-based chat application that lets you:

- Ask for non-alcoholic or alcoholic cocktails by ingredient filters  
- Save and clear your favourite ingredients  
- Get top-5 most popular and rarest ingredients  
- **Attempt** to recommend a cocktail similar to a given one (by ingredient overlap)

## 🚀 Features

1. **Keyword-based filters**  
   - “What are the 5 non-alcoholic cocktails containing sugar?”  
   - “What are the 5 cocktails containing lemon?”  

2. **Favourites management**  
   - Save with:

     my favourite is Rum, Gin
     
- List with:

     What are my favourite ingredients?
     
- Clear with:

     clear my favourites
     
3. **Analytics**  
   - Top-5 most popular ingredients  
   - Top-5 rarest ingredients  

4. **“Similar to” recommendation**  
   - Tries to find cocktails sharing the most ingredients with the target.  
   - Falls back to semantic search (FAISS + SentenceTransformer).  
   - **Known issue:** the ingredient-overlap approach does not reliably return useful results for many cocktail names (often returns an empty list), so the fallback semantic search is used instead.

## 📁 Project Structure


cocktail\_advisor/

├── api/

│   ├── main.py           # FastAPI endpoints & request handlers

│   └── rag\_pipeline.py   # RAG logic, FAISS index, ingredient-overlap & embeddings

├── utils/

│   └── memory.py         # In-memory favourites store

├── chat\_ui/

│   └── index.html        # Minimal chat frontend (HTML+JS)

├── data/

│   └── final\_cocktails.csv

├── vectorstore/

│   └── faiss\_index.faiss # FAISS index (auto-generated)

│   └── faiss\_index.npy   # Corresponding embeddings

├── requirements.txt

└── README.md

`

## 🔧 Installation

1. Clone the repo:
bash
   git clone <your-repo-url>
   cd cocktail_advisor
`

2. Install dependencies:
bash
   pip install -r requirements.txt
   
3. Ensure `data/final_cocktails.csv` is present.

## ▶️ Running
bash
uvicorn api.main:app --reload
Open your browser at [http://127.0.0.1:8000/](http://127.0.0.1:8000/) and start chatting!

## 🐞 Known Issue

* **Similar-to search** using ingredient overlap often returns **no results** for many cocktail names (e.g. “Hot Creamy Bush”).
  As a fallback, the semantic FAISS search is performed, which usually produces meaningful recommendations.

Future improvements could include:

* Better normalization of ingredient names
* Incorporating title embeddings (not just ingredient overlap)
* Handling typos and partial name matches more gracefully

---

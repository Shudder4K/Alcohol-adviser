import os
import re
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from api.rag_pipeline import CocktailRAG
from utils.memory import save_favorite, get_favorites, clear_favorites

app = FastAPI()
rag = CocktailRAG("data/final_cocktails.csv")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def get_chat():
    path = os.path.join("chat_ui", "index.html")
    return HTMLResponse(open(path, "r", encoding="utf-8").read())

@app.post("/chat")
async def chat(message: str = Form(...), user_id: str = Form("user_1")):
    low = message.strip().lower()

    # 1) Clear favourites
    if re.match(r"^clear (my )?favourites?\.?$", low):
        clear_favorites(user_id)
        return {"response": "Your favourites have been cleared."}

    # 2) Save favourites
    if low.startswith("my favourite"):
        raw = message.split("my favourite", 1)[1]
        favs = [w.strip().title() for w in re.split(r"[,\s]+", raw) if w.strip()]
        save_favorite(user_id, favs)
        return {"response": f"Saved your favourites: {', '.join(favs)}"}

    # 3) Show favourites
    if "what are my favourite" in low:
        favs = get_favorites(user_id)
        return {"response": favs and ", ".join(favs) or "You have not set any favourites yet."}

    # 4) Recommend similar by ingredient overlap
    key = "recommend a cocktail similar to"
    if low.startswith(key):
        tail = message[len(key):].strip()
        for q in ('"', "'", '“', '”'):
            tail = tail.strip(q)
        target = tail.strip()
        sims = rag.search_similar(target, k=5)
        if sims:
            return {"response": sims}
        # semantic fallback
        sem = rag.retrieve(target, k=5)
        if sem:
            return {"response": [f"No exact ingredient overlap for '{target}'. Here are semantically similar cocktails:"] + sem}
        return {"response": [f"Sorry, I couldn’t find cocktails similar to '{target}'."]}

    # 5) Recommend by favourites
    if low.startswith("recommend 5 cocktails that contain my favourite ingredients"):
        favs = get_favorites(user_id)
        if not favs:
            return {"response": "You have no saved favourites yet."}
        res = rag.search_by_ingredients(favs, k=5)
        return {"response": res or ["No matching cocktails found."]}

    # 6) Top-5 popular ingredients
    if low.startswith("what are the 5 most popular ingredients"):
        top5 = rag.most_popular_ingredients(5)
        resp = [f"{ing.title()}: {cnt}" for ing, cnt in top5]
        return {"response": resp}

    # 7) Top-5 rarest ingredients
    if low.startswith("what is the rarest ingredient"):
        rare5 = rag.rarest_ingredients(5)
        resp = [f"{ing.title()}: {cnt}" for ing, cnt in rare5]
        return {"response": resp}

    # 8) Fallback semantic search
    res = rag.retrieve(message, k=5)
    return {"response": res or ["Sorry, I couldn’t find any cocktails matching that."]}

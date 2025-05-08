import os
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from collections import Counter

class CocktailRAG:
    def __init__(self,
                 csv_path: str,
                 index_path: str = "vectorstore/faiss_index.faiss"):
        # 1) Завантажуємо датасет і готуємо поля
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.data = pd.read_csv(csv_path)
        self.data['ingredients_list'] = self.data['ingredients']\
            .str.split(r',\s*', expand=False)
        self.data['name_lower'] = self.data['name'].str.lower()
        self.texts = self.data.apply(
            lambda x: f"{x['name']}. Ingredients: {', '.join(x['ingredients_list'])}",
            axis=1
        ).tolist()

        # 2) Підготовка FAISS-індексу
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        emb_path = index_path.replace(".faiss", ".npy")

        if os.path.exists(index_path) and os.path.exists(emb_path):
            # Підвантажуємо індекс і embeddings
            self.index = faiss.read_index(index_path)
            self.embeddings = np.load(emb_path)
        else:
            # Будуємо embeddings та індекс
            emb = self.model.encode(self.texts, show_progress_bar=True).astype('float32')
            self.embeddings = emb
            d = emb.shape[1]
            self.index = faiss.IndexFlatL2(d)
            self.index.add(emb)
            faiss.write_index(self.index, index_path)
            np.save(emb_path, emb)

    def retrieve(self, query: str, k: int = 5) -> list[str]:
        """Семантичний пошук через FAISS + embeddings."""
        vec = self.model.encode([query])[0].astype('float32')
        _, idxs = self.index.search(np.array([vec]), k)
        return [self.texts[i] for i in idxs[0] if i < len(self.texts)]

    def search_similar(self, name: str, k: int = 5) -> list[str]:
        """
        Ігноруємо назву — шукаємо коктейлі з найбільшим перетином інгредієнтів.
        Якщо не знайдено жодного перетину, повертаємо [].
        """
        target = name.strip().lower()
        match = self.data[self.data['name_lower'] == target]
        if match.empty:
            return []
        idx0 = match.index[0]
        target_ings = set(ing.lower() for ing in self.data.at[idx0, 'ingredients_list'])

        # Рахуємо, скільки інгредієнтів спільних із кожного іншого коктейлю
        scores = []
        for idx, row in self.data.iterrows():
            if idx == idx0:
                continue
            ings = set(ing.lower() for ing in row['ingredients_list'])
            common = target_ings & ings
            if common:
                scores.append((len(common), idx))

        # Відсортувати за спаданням кількості спільних інгредієнтів
        scores.sort(key=lambda x: x[0], reverse=True)

        # Повернути топ-k
        results = []
        for common_count, idx in scores[:k]:
            row = self.data.iloc[idx]
            results.append(
                f"{row['name']}. Ingredients: {', '.join(row['ingredients_list'])} "
                f"(common: {common_count})"
            )
        return results

    def search_by_ingredients(self, ingredients: list[str], k: int = 5) -> list[str]:
        """Коктейлі, що містять усі інгредієнти зі списку."""
        mask = self.data['ingredients_list']\
                   .apply(lambda lst: all(
                       ing.lower() in (x.lower() for x in lst)
                       for ing in ingredients))
        subset = self.data[mask].head(k)
        return [
            f"{r['name']}. Ingredients: {', '.join(r['ingredients_list'])}"
            for _, r in subset.iterrows()
        ]

    def most_popular_ingredients(self, n: int = 5) -> list[tuple[str,int]]:
        cnt = Counter(
            ing.lower()
            for lst in self.data['ingredients_list']
            for ing in lst
        )
        return cnt.most_common(n)

    def rarest_ingredients(self, n: int = 5) -> list[tuple[str,int]]:
        cnt = Counter(
            ing.lower()
            for lst in self.data['ingredients_list']
            for ing in lst
        )
        rare = sorted(cnt.items(), key=lambda x: x[1])
        return rare[:n]

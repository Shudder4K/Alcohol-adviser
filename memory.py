user_memory = {}

def save_favorite(user_id: str, ingredients: list):
    """
    Save a list of favorite ingredients for a user.
    """
    if user_id not in user_memory:
        user_memory[user_id] = {'favorites': []}
    for ing in ingredients:
        if ing not in user_memory[user_id]['favorites']:
            user_memory[user_id]['favorites'].append(ing)


def get_favorites(user_id: str) -> list:
    """
    Retrieve the list of favorite ingredients for a user.
    """
    return user_memory.get(user_id, {}).get('favorites', [])


def clear_favorites(user_id: str):
    """
    Clear all saved favorite ingredients for the user.
    """
    if user_id in user_memory:
        user_memory[user_id]['favorites'].clear()
import os

def create_folders():
    os.makedirs("images", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
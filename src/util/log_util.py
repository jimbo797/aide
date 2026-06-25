from src.globals import LOGGING

def log(alias: str, message: str):
    if not LOGGING: return

    print(f"[{alias}] {message}")

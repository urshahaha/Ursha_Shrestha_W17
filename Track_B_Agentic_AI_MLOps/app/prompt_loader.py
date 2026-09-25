from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def load_prompt(version:str="v3")->str:
    p=ROOT/"prompts"/f"prompt_{version}.txt"
    if not p.exists(): raise ValueError(f"Unknown prompt version: {version}")
    return p.read_text()

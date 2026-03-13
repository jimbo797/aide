"""Save and load RKTRoot skill trees as JSON."""

import json
from pathlib import Path
import os
from type import RKTRoot



def save_skill_tree(root: RKTRoot, path: str | Path) -> None:
    """Save a skill tree to a JSON file. Use load_skill_tree to restore it."""
    path = Path(path)
    data = root.model_dump(mode="json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_skill_tree(path: str | Path) -> RKTRoot:
    """Load a skill tree from a JSON file produced by save_skill_tree."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return RKTRoot.model_validate(data)

if __name__ == "__main__":
    dir = os.path.dirname(os.path.abspath(__file__))
    rubric_name = "open-ended-response"
    RUBRIC_PATH = f"{rubric_name}.csv"
    TREE_PATH = f"{rubric_name}.json"

    skill_tree = load_skill_tree(os.path.join(dir, "rubrics", TREE_PATH))
    print(skill_tree)

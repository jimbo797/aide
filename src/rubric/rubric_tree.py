from rubric_types import *

# Entry point for generating the rubric tree

def flatten_rubric_leaves(rubric: Rubric) -> list[RubricCriteria]:
    leaves = []
    for category in rubric.categories:
        leaves.extend(category.criteria)
    return leaves

def generate_rubric_tree(rubric_path: str) -> Rubric:
    pass
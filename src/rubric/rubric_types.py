from typing import List, Optional, Union, Literal, Annotated
from pydantic import BaseModel, Field

# This tree is generated from the raw rubric information, and the reviewed and amended by teaching staff

# Is decided whether it is met or not met by the eval agent
class RubricCriteria(BaseModel):
    id: int
    description: str

# Given a score
class RubricCategory(BaseModel):
    description: str # Description of the category
    weight: float # Percentage, how much this category is worth in the total score
    criteria: List[RubricCriteria] # Atomic criteria that are part of this category
    scoring_instructions: str # Instructions for aggregating the criteria scores and creating the total category score

# Total rubric tree
class Rubric(BaseModel):
    categories: List[RubricCategory]
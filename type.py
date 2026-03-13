from typing import List, Optional, Union, Literal, Annotated
from pydantic import BaseModel, Field


# class CategorySlot(BaseModel):
#   points: int
#   description: str

# class Rubric(BaseModel):
#   categories: list[CategorySlot]


# # autoscore
# class CategoryScore(BaseModel):
#   score: int
#   evidence: list[str]

# class Score(BaseModel):
#   categories: list[CategoryScore]


# class Feature(BaseModel):
#   feature_name: str
#   data_type: str

# class StructuredComponentRepresentation(BaseModel):
#   features: list[Feature]



# ratas rubric knowledge tree (RKT)
class RKTSimpleRule(BaseModel):
  type: Literal["simple"] = "simple rule"
  rule: str

Rule = Annotated[
  Union["RKTBasicRule", RKTSimpleRule],
  Field(discriminator="type")
]

class RKTBasicRule(BaseModel):
  type: Literal["basic"] = "basic rule"
  description: str
  children: List[Rule]

class RKTRubricLine(BaseModel):
  type: Literal["line"] = "rubric line"
  description: str
  children: List[Rule]

class RKTRoot(BaseModel):
  type: Literal["root"] = "root"
  rows: List[RKTRubricLine]

RKTBasicRule.model_rebuild()
RKTRubricLine.model_rebuild()

class Skill(BaseModel):
  description: str

class SkillList(BaseModel):
  skills: List[Skill]
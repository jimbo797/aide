from typing import List, Optional, Union, Literal, Annotated
from pydantic import BaseModel, Field

# ratas rubric knowledge tree (RKT)
# Structure: root -> rows (rubric lines) -> children (Rule*). Rule = basic | simple.
# Simple rules are atomic criteria (leaves, yes/no). Basic rules optionally umbrella over rules.
# A rubric line's children can be any mix of basic and simple rules.
class RKTSimpleRule(BaseModel):
  type: Literal["simple rule"] = "simple rule"
  rule: str  # atomic criterion text

Rule = Annotated[
  Union["RKTBasicRule", RKTSimpleRule],
  Field(discriminator="type")
]

class RKTBasicRule(BaseModel):
  type: Literal["basic rule"] = "basic rule"
  description: str  # group name; children are rules (basic or simple)
  children: List[Rule]

class RKTRubricLine(BaseModel):
  type: Literal["rubric line"] = "rubric line"
  description: str
  children: List[Rule]  # can be basic rules, simple rules, or both

class RKTRoot(BaseModel):
  type: Literal["root"] = "root"
  rows: List[RKTRubricLine]

RKTBasicRule.model_rebuild()
RKTRubricLine.model_rebuild()

class Skill(BaseModel):
  description: str

class SkillList(BaseModel):
  skills: List[Skill]


# assessment (using RKT rubric)
class CriterionAssessment(BaseModel):
  description: str
  met: bool
  evidence: Optional[str] = None


class CategoryAssessment(BaseModel):
  category_name: str
  criteria: List[CriterionAssessment]


class Assessment(BaseModel):
  categories: List[CategoryAssessment]
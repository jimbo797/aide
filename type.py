from typing import List, Optional, Union, Literal, Annotated
from pydantic import BaseModel, Field

# ratas rubric knowledge tree (RKT)
# Structure: root -> rows (rubric lines) -> children (Rule*). Rule = basic | simple.
# Simple rules are atomic criteria (leaves, yes/no). Basic rules optionally umbrella over rules.
# A rubric line's children can be any mix of basic and simple rules.
class RKTSimpleRule(BaseModel):
  type: Literal["simple rule"] = "simple rule"
  rule: str  # atomic criterion text
  weight: Optional[float] = None  # optional score weight / points for this leaf

Rule = Annotated[
  Union["RKTBasicRule", RKTSimpleRule],
  Field(discriminator="type")
]

class RKTBasicRule(BaseModel):
  type: Literal["basic rule"] = "basic rule"
  description: str  # group name; children are rules (basic or simple)
  children: List[Rule]
  weight: Optional[float] = None  # optional group weight (e.g. sub-criterion points)

class RKTRubricLine(BaseModel):
  type: Literal["rubric line"] = "rubric line"
  description: str
  children: List[Rule]  # can be basic rules, simple rules, or both
  weight: Optional[float] = None  # weight for this top-level criterion (e.g. rubric row)

class RKTRoot(BaseModel):
  type: Literal["root"] = "root"
  rows: List[RKTRubricLine]

RKTBasicRule.model_rebuild()
RKTRubricLine.model_rebuild()

class Skill(BaseModel):
  description: str

class SkillList(BaseModel):
  skills: List[Skill]


# Per-leaf assessment against an RKTRoot (each simple-rule leaf: met / not met / undetermined)
LeafVerdict = Literal["met", "not_met", "undetermined"]


class LeafAssessmentItem(BaseModel):
  """LLM output: one row per rubric leaf, keyed by leaf_id from the prompt."""

  leaf_id: int
  verdict: LeafVerdict
  evidence: Optional[str] = None


class LeafAssessmentBatch(BaseModel):
  results: List[LeafAssessmentItem]
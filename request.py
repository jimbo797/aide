import instructor
from dataclasses import dataclass

from type import (
  RKTRoot,
  SkillList,
  LeafAssessmentBatch,
  LeafVerdict,
)
from dotenv import load_dotenv
import os

load_dotenv()

client = instructor.from_provider("openai/gpt-5-nano")


def rubric_skill_extract(str_rubric):
  dir = os.path.dirname(os.path.abspath(__file__))
  path = os.path.join(dir, "prompts", "skill-extraction.txt")
  with open(path, 'r') as file:
    prompt = file.read()
  prompt = prompt.format(rubric_text=str_rubric)
  # print(prompt)

  out = client.create(
    response_model=SkillList,
    messages=[ 
      {
        "role": "user",
        "content": prompt,
      }
      ],
    )
  return out


def rubric_skill_tree_construct(
  skills,
  categories_block: str | None = None,
):
  dir = os.path.dirname(os.path.abspath(__file__))
  path = os.path.join(dir, "prompts", "skill-tree-construction.txt")
  with open(path, 'r') as file:
    prompt = file.read()
  skills_text = (
    skills.model_dump_json(indent=2)
    if hasattr(skills, "model_dump_json")
    else str(skills)
  )
  block = (categories_block or "").strip()
  if not block:
    block = (
      "(No fixed category list — infer rubric lines from the skills below; "
      "use null for top-level weights unless the skill text states a numeric weight.)"
    )
  prompt = prompt.format(skills=skills_text, categories_block=block)
  out = client.create(
    response_model=RKTRoot,
    messages=[ 
      {
        "role": "user",
        "content": prompt,
      }
      ],
    )
  return out


def _rule_text(rule) -> str:
  """Get criterion text from a rule (basic or simple)."""
  return (getattr(rule, "description", None) or getattr(rule, "rule", None) or "").strip()


def _collect_simple_criteria(
  rule, under_basic: str | None
) -> list[tuple[str | None, str, float | None]]:
  """
  Collect (optional_basic_description, criterion_text, optional_weight) for assessment.
  Only simple rules are atomic criteria; basic rules are traversed. Basic rules with no children
  are treated as one criterion (fallback for rubrics that have no simple-rule terminals yet).
  """
  out: list[tuple[str | None, str, float | None]] = []
  if getattr(rule, "type", None) == "simple rule":
    text = (getattr(rule, "rule", None) or "").strip()
    if text:
      w = getattr(rule, "weight", None)
      out.append((under_basic, text, w))
  else:
    # basic rule
    children = getattr(rule, "children", []) or []
    basic_desc = (getattr(rule, "description", None) or "").strip() or under_basic
    if not children:
      # Fallback: treat as one criterion so rubrics without simple terminals still work
      desc = (getattr(rule, "description", None) or "").strip()
      if desc:
        w = getattr(rule, "weight", None)
        out.append((under_basic, desc, w))
    else:
      for r in children:
        out.extend(_collect_simple_criteria(r, basic_desc))
  return out


@dataclass
class RubricLeafRef:
  """One assessable leaf (terminal criterion) in an RKTRoot."""

  leaf_id: int
  category: str
  category_weight: float | None
  basic_group: str | None
  rule_text: str
  leaf_weight: float | None


def flatten_rubric_leaves(root: RKTRoot) -> list[RubricLeafRef]:
  """Enumerate simple-rule leaves in tree order for per-leaf assessment."""
  leaves: list[RubricLeafRef] = []
  leaf_id = 0
  for row in getattr(root, "rows", []) or []:
    cat = (getattr(row, "description", None) or "").strip()
    rw = getattr(row, "weight", None)
    for child in getattr(row, "children", []) or []:
      for under_basic, text, cw in _collect_simple_criteria(child, None):
        if not text:
          continue
        leaves.append(
          RubricLeafRef(
            leaf_id=leaf_id,
            category=cat,
            category_weight=rw,
            basic_group=under_basic,
            rule_text=text,
            leaf_weight=cw,
          )
        )
        leaf_id += 1
  return leaves


def _format_leaves_block(leaves: list[RubricLeafRef]) -> str:
  lines: list[str] = []
  for L in leaves:
    parts = [f"{L.leaf_id}."]
    parts.append(f'Category: "{L.category}"')
    if L.category_weight is not None:
      parts.append(f"(category weight: {L.category_weight})")
    if L.basic_group:
      parts.append(f'[Basic: "{L.basic_group}"]')
    parts.append(f'Criterion: "{L.rule_text}"')
    if L.leaf_weight is not None:
      parts.append(f"(leaf weight: {L.leaf_weight})")
    lines.append(" ".join(parts))
  return "\n".join(lines)


def _merge_leaf_assessments(
  leaves: list[RubricLeafRef],
  batch: LeafAssessmentBatch,
) -> list[dict]:
  """Join LLM results with rubric refs; fill missing ids as undetermined."""
  by_id = {r.leaf_id: r for r in batch.results}
  merged: list[dict] = []
  for L in leaves:
    r = by_id.get(L.leaf_id)
    verdict: LeafVerdict = r.verdict if r else "undetermined"
    evidence = r.evidence if r else None
    merged.append(
      {
        "leaf_id": L.leaf_id,
        "category": L.category,
        "category_weight": L.category_weight,
        "basic_group": L.basic_group,
        "criterion": L.rule_text,
        "leaf_weight": L.leaf_weight,
        "verdict": verdict,
        "evidence": evidence,
      }
    )
  return merged


def assess_response_leaves(
  rubric: RKTRoot,
  response_text: str,
) -> tuple[list[dict], LeafAssessmentBatch]:
  """
  Assess the response against each simple-rule leaf. Returns (merged rows for export/UI, raw batch).
  """
  leaves = flatten_rubric_leaves(rubric)
  if not leaves:
    return [], LeafAssessmentBatch(results=[])

  dir = os.path.dirname(os.path.abspath(__file__))
  path = os.path.join(dir, "prompts", "leaf-assessment.txt")
  with open(path, "r", encoding="utf-8") as f:
    prompt_tpl = f.read()
  leaves_block = _format_leaves_block(leaves)
  prompt = prompt_tpl.format(leaves_block=leaves_block, response_text=response_text)
  out = client.create(
    response_model=LeafAssessmentBatch,
    messages=[{"role": "user", "content": prompt}],
  )
  merged = _merge_leaf_assessments(leaves, out)
  return merged, out


from openai import OpenAI
import instructor
from type import RKTRoot, SkillList, Assessment
from dotenv import load_dotenv
import os

load_dotenv()

# client = OpenAI()
client = instructor.from_provider("openai/gpt-5-nano")

# TODO: add the list input version that lets me pass in a system prompt
def make_formatted_request(input, response_model):
  response = client.responses.parse(
    model="gpt-5-nano",
    input=input,
    text_format=format,
  )
  print(response.output)
  return response.output_text
  # return response 


def rubric_RKT_construct(str_rubric):
  out = client.create(
    response_model=RKTRoot,
    messages=[ 
      {
        "role": "system", 
        "content": "Extract the rules/criteria in an assignment rubric and fill them into a tree structure. The root will have child nodes that are the rows/categories of the rubric. "
        "These rows may be divided into basic rules or simple rules. A basic rule is a more general rule that encompasses simple rules. Simple rules are binary decisions about "
        "the student's response, and should be answered with a single yes/no response. The rubric will most likely have point values assigned to them. If so, consider the aggregate ideas of points columns "
        "and place these into the tree"}, 
      {
        "role": "user",
        "content": str_rubric,
      }
      ],
    )
  return out


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


def rubric_skill_tree_construct(skills):
  dir = os.path.dirname(os.path.abspath(__file__))
  path = os.path.join(dir, "prompts", "skill-tree-construction.txt")
  with open(path, 'r') as file:
    prompt = file.read()
  prompt = prompt.format(skills=skills)
  # print(prompt)
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


def _collect_simple_criteria(rule, under_basic: str | None) -> list[tuple[str | None, str]]:
  """
  Collect (optional_basic_description, criterion_text) for assessment.
  Only simple rules are atomic criteria; basic rules are traversed. Basic rules with no children
  are treated as one criterion (fallback for rubrics that have no simple-rule terminals yet).
  """
  out: list[tuple[str | None, str]] = []
  if getattr(rule, "type", None) == "simple rule":
    text = (getattr(rule, "rule", None) or "").strip()
    if text:
      out.append((under_basic, text))
  else:
    # basic rule
    children = getattr(rule, "children", []) or []
    basic_desc = (getattr(rule, "description", None) or "").strip() or under_basic
    if not children:
      # Fallback: treat as one criterion so rubrics without simple terminals still work
      desc = (getattr(rule, "description", None) or "").strip()
      if desc:
        out.append((under_basic, desc))
    else:
      for r in children:
        out.extend(_collect_simple_criteria(r, basic_desc))
  return out


def _flatten_rubric_to_text(root: RKTRoot) -> str:
  """Format rubric for assessment: only simple rules (atomic criteria), preserving optional basic grouping."""
  lines = []
  for row in getattr(root, "rows", []) or []:
    cat = (getattr(row, "description", None) or "").strip()
    children = getattr(row, "children", []) or []
    criteria_with_basic: list[tuple[str | None, str]] = []
    for r in children:
      criteria_with_basic.extend(_collect_simple_criteria(r, None))
    if not criteria_with_basic:
      continue
    lines.append(f"Category: {cat}")
    current_basic = None
    for under_basic, criterion_text in criteria_with_basic:
      if under_basic != current_basic:
        current_basic = under_basic
        if under_basic:
          lines.append(f"  [Basic: {under_basic}]")
      lines.append(f"  - {criterion_text}")
    lines.append("")
  return "\n".join(lines).strip()


def assess_response(rubric: RKTRoot, response_text: str) -> Assessment:
  """Assess a student response against the given rubric (RKTRoot). Returns structured Assessment."""
  dir = os.path.dirname(os.path.abspath(__file__))
  path = os.path.join(dir, "prompts", "assessment.txt")
  with open(path, "r", encoding="utf-8") as f:
    prompt_tpl = f.read()
  rubric_text = _flatten_rubric_to_text(rubric)
  prompt = prompt_tpl.format(rubric_text=rubric_text, response_text=response_text)
  out = client.create(
    response_model=Assessment,
    messages=[{"role": "user", "content": prompt}],
  )
  return out


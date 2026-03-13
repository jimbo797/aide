from openai import OpenAI
import instructor
from type import RKTRoot, SkillList
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


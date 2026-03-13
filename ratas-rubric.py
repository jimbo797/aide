from rubrics import get_rubric_df
from request import *
from datetime import datetime
from tree_viz import render_skill_tree
from rkt_io import save_skill_tree

if __name__ == "__main__": 
  rubric_name = "open-ended-response"
  RUBRIC_PATH = f"{rubric_name}.csv"
  OUTPUT_PATH = f"{rubric_name}.json"
  rubric_df = get_rubric_df(RUBRIC_PATH)
  columns = rubric_df.columns.to_list()

  rubric_str = rubric_df.to_csv(index=False)
  start_time = datetime.now()

  skills = rubric_skill_extract(rubric_str)
  print(skills)
  end_time = datetime.now()
  print("Elapsed:", end_time - start_time)

  skill_tree = rubric_skill_tree_construct(skills)
  print(render_skill_tree(skill_tree))
  end_time = datetime.now()
  print("Elapsed:", end_time - start_time)

  print("Saving skill tree to file...")

  save_skill_tree(skill_tree, OUTPUT_PATH)
  skill_tree = load_skill_tree(OUTPUT_PATH)
  print(render_skill_tree(skill_tree))
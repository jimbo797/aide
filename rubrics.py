import pandas as pd
import os

dir = os.path.dirname(os.path.abspath(__file__))

def get_rubric_df(filename):
  df = pd.read_csv(os.path.join(dir, "rubrics", filename))
  return df

import pandas as pd
import os

dir = os.path.dirname(os.path.abspath(__file__))

def get_rubric_df(filepath):
  df = pd.read_csv(filepath)
  return df

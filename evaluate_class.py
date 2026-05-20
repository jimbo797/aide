# USING NEW REFACTORED EVALUATION CODE

from pathlib import Path
import pandas as pd
from datetime import datetime
import csv
from src.eval.preprocess.preprocess import preprocess_video
from src.eval.eval import eval_submission
from src.rubric.rubric_types import Rubric
from src.util import log

def evaluate_class(assigment_list_csv_path: Path, rubric: Rubric, preprocess_dir: Path, output_dir: Path, model: str) -> float:
    df = pd.read_csv(assigment_list_csv_path)

    class_results = []
    for index, row in df.iterrows():
        url = row["link"]
        alias = row["email"].split("@")[0]

        log(alias, "Starting")

        log(alias, "Preprocessing")
        preprocess_video(url, alias, preprocess_dir)

        log(alias, "Evaluating")
        score, results = eval_submission(alias, rubric, preprocess_dir=preprocess_dir, output_dir=output_dir, model=model)
        class_results.append((alias, score))
        
        with open(output_dir / f"class_results.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(["alias", "score"])
            for alias, score in class_results:
                writer.writerow([alias, score])

        log(alias, "Finished")

if __name__ == "__main__":
    start_time = datetime.now()
    print(f"Start time: {start_time.time()}")

    json_data = Path("rubrics/gsu-sumprod-manual.json").read_text()
    rubric = Rubric.model_validate_json(json_data)

    evaluate_class(
        assigment_list_csv_path=Path("student-responses/gsu-student-sumprod-video-list-short.csv"), 
        rubric=rubric, 
        preprocess_dir=Path("out/preprocess"), 
        output_dir=Path("out/results"),
        model="gpt-5.5"
        )

    end_time = datetime.now()
    print(f"End time: {end_time.time()}")
    print(f"Total time: {end_time - start_time}")



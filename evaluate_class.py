# ENTRY POINT FOR EVALUATING A CLASS
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
    times_taken = []
    for index, row in df.iterrows():
        start_time = datetime.now()
        # Just in case an error occurs, we don't want to stop the entire evaluation
        try:
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
        except Exception as e:
            log(alias, f"Error: {e}")
            continue

        end_time = datetime.now()
        delta_time = end_time - start_time
        times_taken.append(delta_time)
        log(alias, f"Finished. Time taken: {delta_time}")
    
    print(f"Finished evaluating class")
    print(f"Total time: {sum(times_taken)}")
    print(f"Average time: {sum(times_taken) / len(times_taken)}")

if __name__ == "__main__":
    json_data = Path("rubrics/gsu-sumprod-manual.json").read_text()
    rubric = Rubric.model_validate_json(json_data)

    evaluate_class(
        # assigment_list_csv_path=Path("student-responses/gsu-student-sumprod-video-list-short.csv"), 
        assigment_list_csv_path=Path("student-responses/gsu-student-sumprod-video-list-checkpoint.csv"), 
        rubric=rubric, 
        preprocess_dir=Path("out/preprocess"), 
        output_dir=Path("out/results"),
        model="gpt-5.5"
        )




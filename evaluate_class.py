# ENTRY POINT FOR EVALUATING A CLASS
from pathlib import Path
from datetime import datetime, timedelta
import csv
from src.eval.preprocess.preprocess_artifacts import preprocess_artifacts
from src.eval.eval import eval_submission
from src.rubric.rubric_types import Rubric
from src.util import log

def evaluate_class(rubric: Rubric, in_dir: Path, preprocess_dir: Path, output_dir: Path, model: str, sheet_name: str) -> None:
    class_results = []
    times_taken = []
    submission_dirs = sorted(path for path in in_dir.iterdir() if path.is_dir())

    for submission_dir in submission_dirs:
        alias = submission_dir.name
        start_time = datetime.now()
        # Just in case an error occurs, we don't want to stop the entire evaluation
        try:
            log(alias, "Starting")

            log(alias, "Preprocessing")
            preprocess_artifacts(
                alias,
                submissions_dir=in_dir,
                preprocess_dir=preprocess_dir,
                sheet_name=sheet_name,
            )
            
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
            # continue

        end_time = datetime.now()
        delta_time = end_time - start_time
        times_taken.append(delta_time)
        log(alias, f"Finished. Time taken: {delta_time}")
    
    print(f"Finished evaluating class")
    total_time = sum(times_taken, timedelta())
    print(f"Total time: {total_time}")
    if times_taken:
        print(f"Average time: {total_time / len(times_taken)}")

if __name__ == "__main__":
    # json_data = Path("rubrics/gsu-spring-forecast-manual.json").read_text()
    json_data = Path("rubrics/gsu-spring-carloan-manual.json").read_text()
    rubric = Rubric.model_validate_json(json_data)

    evaluate_class(
        rubric=rubric, 
        in_dir=Path("in"),
        preprocess_dir=Path("out/preprocess"), 
        output_dir=Path("out/results"),
        model="gpt-5.5",
        # sheet_name="Forecast"
        sheet_name="CarLoan"
        )




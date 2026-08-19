# ENTRY POINT FOR EVALUATING A CLASS
from pathlib import Path
from datetime import datetime, timedelta
import csv
import json
from src.eval.preprocess.preprocess_artifacts import preprocess_artifacts
from src.eval.eval import eval_submission
from src.rubric.rubric_types import Rubric
from src.util import log
from src.util.token_usage import aggregate_costs, student_costs_dict, track_token_usage

def evaluate_class(
    rubric: Rubric,
    in_dir: Path,
    preprocess_dir: Path,
    output_dir: Path,
    model: str,
    sheet_name: str | None = None,
    scene_threshold: float | None = 0.1,
    sample_interval_seconds: float | None = 30.0,
    max_loop_iters: int | None = None,
) -> None:
    class_results = []
    class_costs: list[dict] = []
    times_taken = []
    submission_dirs = sorted(path for path in in_dir.iterdir() if path.is_dir())

    for submission_dir in submission_dirs:
        alias = submission_dir.name
        start_time = datetime.now()
        token_costs = None
        # Just in case an error occurs, we don't want to stop the entire evaluation
        try:
            log(alias, "Starting")

            with track_token_usage(alias) as usage:
                log(alias, "Preprocessing")
                preprocess_artifacts(
                    alias,
                    submissions_dir=in_dir,
                    preprocess_dir=preprocess_dir,
                    sheet_name=sheet_name,
                    scene_threshold=scene_threshold,
                    sample_interval_seconds=sample_interval_seconds,
                )

                log(alias, "Evaluating")
                score, results, token_costs = eval_submission(
                    alias,
                    rubric,
                    preprocess_dir=preprocess_dir,
                    submissions_dir=in_dir,
                    output_dir=output_dir,
                    model=model,
                    max_loop_iters=max_loop_iters,
                )
                # Prefer the outer tracker so preprocess + eval are both included.
                token_costs = usage.to_dict()
            class_results.append((alias, score))
            
            with open(output_dir / f"class_results.csv", "w") as f:
                writer = csv.writer(f)
                writer.writerow(["alias", "score"])
                for result_alias, result_score in class_results:
                    writer.writerow([result_alias, result_score])
        except Exception as e:
            log(alias, f"Error: {e}")
            # continue

        end_time = datetime.now()
        delta_time = end_time - start_time
        times_taken.append(delta_time)
        log(alias, f"Finished. Time taken: {delta_time}")

        if token_costs is not None:
            costs = student_costs_dict(alias, delta_time, token_costs)
            student_out = output_dir / alias
            student_out.mkdir(parents=True, exist_ok=True)
            with open(student_out / "costs.json", "w") as f:
                json.dump(costs, f, indent=2)
            class_costs.append(costs)
            with open(output_dir / "class_costs.json", "w") as f:
                json.dump(aggregate_costs(class_costs), f, indent=2)
            log(
                alias,
                f"Tokens: {token_costs.get('total_tokens', 0)} "
                f"(prompt={token_costs.get('prompt_tokens', 0)}, "
                f"completion={token_costs.get('completion_tokens', 0)}, "
                f"calls={token_costs.get('api_calls', 0)}, "
                f"cost=${token_costs.get('cost_usd', 0):.4f})",
            )
    
    print(f"Finished evaluating class")
    total_time = sum(times_taken, timedelta())
    print(f"Total time: {total_time}")
    if times_taken:
        print(f"Average time: {total_time / len(times_taken)}")
    if class_costs:
        totals = aggregate_costs(class_costs)
        tc = totals["token_costs"]
        print(
            f"Total tokens: {tc['total_tokens']} "
            f"(prompt={tc['prompt_tokens']}, "
            f"completion={tc['completion_tokens']}, "
            f"calls={tc['api_calls']}, "
            f"cost=${tc.get('cost_usd', 0):.4f})"
        )

if __name__ == "__main__":
    json_data = Path("rubrics/gsu-summer-forecast.json").read_text()
    # json_data = Path("rubrics/asap2.json").read_text()
    rubric = Rubric.model_validate_json(json_data)

    # evaluate_class(
    #     rubric=rubric, 
    #     in_dir=Path("in"),
    #     preprocess_dir=Path("out/preprocess"), 
    #     output_dir=Path("out/results"),
    #     model="gpt-5.6-terra",
    #     max_loop_iters=3,
    #     # sheet_name="Forecast"
    #     # sheet_name="CarLoan"
    #     )

    evaluate_class(
        rubric=rubric, 
        in_dir=Path("old-runs/gsu-summer-in"),
        preprocess_dir=Path("old-runs/summer9/preprocess"), 
        output_dir=Path("old-runs/summer9/results"),
        model="gpt-5.6-terra",
        max_loop_iters=3
        )

    # experiments = [
    #                 (None, 30), (None, 90),
    #     (0.1, None), (0.1, 30), (0.1, 90),
    #     (0.3, None), (0.3, 30), (0.3, 90),
    #     ]
    # for threshold, interval in experiments:
    #     evaluate_class(
    #         rubric=rubric, 
    #         in_dir=Path("old-runs/gsu-summer-in"),
    #         preprocess_dir=Path(f"old-runs/frame-sampling/gsu-{threshold}-{interval}/preprocess"), 
    #         output_dir=Path(f"old-runs/frame-sampling/gsu-{threshold}-{interval}/results"),
    #         model="gpt-5.6-terra",
    #         max_loop_iters=3,
    #         scene_threshold=threshold,
    #         sample_interval_seconds=interval,
    #         )

    


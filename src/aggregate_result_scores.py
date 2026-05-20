# Collects all the results from the out/results directory and aggregates the scores

from pathlib import Path
import json
import csv
import os

def aggregate_result_scores():
    result_dir = Path("out/results")
    result_files = result_dir.glob("*.json")

    results= []
    for result_file in result_files:
        alias = result_file.stem
        total_points = 0
        with open(result_file, "r") as f:
            result = json.load(f)
            for category in result:
                total_points += category["category_points"]

        results.append((alias, total_points))

    with open("out/results/class_results.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["alias", "score"])
        for alias, score in results:
            writer.writerow([alias, score])


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent.parent)
    aggregate_result_scores()
            
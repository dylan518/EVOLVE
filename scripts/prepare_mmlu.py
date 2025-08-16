#!/usr/bin/env python3
import argparse
import csv
import os
from typing import List

from datasets import load_dataset, get_dataset_config_names


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "datas", "mmlu")


def to_letter(ans):
    if isinstance(ans, int):
        return ["A", "B", "C", "D"][ans]
    a = str(ans).strip().upper()
    if a in {"A", "B", "C", "D"}:
        return a
    try:
        idx = int(a)
        return ["A", "B", "C", "D"][idx]
    except Exception:
        return "C"


def write_rows(path: str, rows: List[dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["Subject", "Question", "A", "B", "C", "D", "Answer"],
        )
        w.writeheader()
        w.writerows(rows)


def _try_load(subject: str, split_candidates: List[str]):
    last_err = None
    for s in split_candidates:
        try:
            return load_dataset(
                "hendrycks_test",
                subject,
                split=s,
                trust_remote_code=True,
            )
        except Exception as e:
            last_err = e
            continue
    raise last_err


def collect_split(subjects: List[str], split: str, max_per_subject: int) -> List[dict]:
    data = []
    for subject in subjects:
        # hendrycks_test usually uses 'dev' and 'test'; some mirrors expose 'validation'
        split_candidates = [split]
        if split == "dev":
            split_candidates = ["dev", "validation"]
        ds = _try_load(subject, split_candidates)
        count = 0
        for item in ds:
            choices = item.get("choices") or item.get("options") or []
            if len(choices) != 4:
                continue
            row = {
                "Subject": subject,
                "Question": item["question"],
                "A": choices[0],
                "B": choices[1],
                "C": choices[2],
                "D": choices[3],
                "Answer": to_letter(item["answer"]),
            }
            data.append(row)
            count += 1
            if max_per_subject and count >= max_per_subject:
                break
    return data


def main():
    ap = argparse.ArgumentParser(description="Prepare real MMLU CSVs for this repo")
    ap.add_argument("--max_per_subject", type=int, default=0, help="Limit per subject (0 = full)")
    args = ap.parse_args()

    # Prefer CAIS MMLU (already consolidated) when available
    try:
        cais_valid = load_dataset("cais/mmlu", "all", split="validation")
        cais_test = load_dataset("cais/mmlu", "all", split="test")
        def rows_from(ds, limit_per_subject: int):
            counts = {}
            rows = []
            for item in ds:
                s = item["subject"]
                if limit_per_subject and counts.get(s, 0) >= limit_per_subject:
                    continue
                choices = item["choices"]
                if len(choices) != 4:
                    continue
                rows.append({
                    "Subject": s,
                    "Question": item["question"],
                    "A": choices[0],
                    "B": choices[1],
                    "C": choices[2],
                    "D": choices[3],
                    "Answer": to_letter(item["answer"]),
                })
                counts[s] = counts.get(s, 0) + 1
            return rows
        valid_rows = rows_from(cais_valid, args.max_per_subject)
        test_rows = rows_from(cais_test, args.max_per_subject)
    except Exception:
        subjects = get_dataset_config_names("hendrycks_test")
        valid_rows = collect_split(subjects, split="dev", max_per_subject=args.max_per_subject)
        test_rows = collect_split(subjects, split="test", max_per_subject=args.max_per_subject)

    write_rows(os.path.join(OUTPUT_DIR, "valid.csv"), valid_rows)
    write_rows(os.path.join(OUTPUT_DIR, "test.csv"), test_rows)

    # No official train; use dev as the surrogate for mmlu.csv expected by the loader
    write_rows(os.path.join(OUTPUT_DIR, "mmlu.csv"), valid_rows)

    print(f"Wrote valid={len(valid_rows)}, test={len(test_rows)} rows to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()



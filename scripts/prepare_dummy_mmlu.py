#!/usr/bin/env python3
import os
import csv

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "datas", "mmlu")

ROWS = [
    {
        "Subject": "general_knowledge",
        "Question": "Which planet is known as the Red Planet?",
        "A": "Earth",
        "B": "Venus",
        "C": "Mars",
        "D": "Jupiter",
        "Answer": "C",
    },
    {
        "Subject": "math",
        "Question": "What is 2 + 2?",
        "A": "3",
        "B": "5",
        "C": "4",
        "D": "22",
        "Answer": "C",
    },
    {
        "Subject": "science",
        "Question": "Water freezes at what temperature (Celsius)?",
        "A": "-10",
        "B": "10",
        "C": "0",
        "D": "100",
        "Answer": "C",
    },
    {
        "Subject": "history",
        "Question": "Who was the first president of the United States?",
        "A": "Abraham Lincoln",
        "B": "Thomas Jefferson",
        "C": "George Washington",
        "D": "John Adams",
        "Answer": "C",
    },
    {
        "Subject": "logic",
        "Question": "If all cats are animals and some animals are black, which must be true?",
        "A": "All cats are black",
        "B": "No cats are black",
        "C": "Some animals are black",
        "D": "All animals are cats",
        "Answer": "C",
    },
]

def write_csv(path: str, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Subject","Question","A","B","C","D","Answer"])
        writer.writeheader()
        writer.writerows(rows)

def main():
    valid_path = os.path.join(DATA_DIR, "valid.csv")
    test_path = os.path.join(DATA_DIR, "test.csv")
    train_path = os.path.join(DATA_DIR, "mmlu.csv")

    write_csv(valid_path, ROWS)
    write_csv(test_path, ROWS)
    write_csv(train_path, ROWS)
    print(f"Wrote dummy MMLU CSVs to {DATA_DIR}")

if __name__ == "__main__":
    main()




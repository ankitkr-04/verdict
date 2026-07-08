"""Self-test for the zero-LLM deterministic lane. Run: python eval/test_deterministic.py

The negative cases matter most: a handler firing on a word problem it merely
resembles would produce a confidently wrong final answer.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.solvers.deterministic import try_deterministic

MUST_ANSWER: list[tuple[str, str]] = [
    ("What is 847 * 36 + 129?", "30621"),
    ("Calculate 2^10", "1024"),
    ("what's (15 + 5) / 4?", "5"),
    ("Compute 7.5 * 4", "30"),
    ("What is 15% of 240?", "36"),
    ("What is 12.5 percent of 80?", "10"),
    ("What day of the week was 2024-03-15?", "Friday"),
    ("What day of the week is March 15, 2024?", "Friday"),
    ("Which day of the week was 15 March 2024?", "Friday"),
    ("How many days are there between 2024-01-01 and 2024-03-01?", "60"),
    ("How many days between January 1, 2020 and January 1, 2021?", "366"),
    ("45 days after March 3, 2021?", "2021-04-17"),
    ("What date is 10 days before 2024-01-05?", "2023-12-26"),
    ("How many times does the letter 'r' appear in 'strawberry'?", "3"),
    ("How many 'e's are in 'excellence'?", "4"),
    ("How many vowels are there in 'education'?", "5"),
    ("How many words are in 'the quick brown fox'?", "4"),
    ("Reverse the string 'hello'", "olleh"),
    ("What is 'stressed' spelled backwards?", "desserts"),
    ("What is the average of 4, 8, 15, 16?", "10.75"),
    ("Calculate the median of 3, 1 and 2", "2"),
    ("Find the sum of 10, 20, 30", "60"),
    ("What is the largest of 7, 42, 19?", "42"),
    ("What is the GCD of 12 and 18?", "6"),
    ("Find the least common multiple of 4 and 6?", "12"),
    ("What is the factorial of 6?", "720"),
    ("Compute 6 factorial", "720"),
    ("What is the square root of 144?", "12"),
    ("Find the cube root of 27", "3"),
    ("What is the percentage increase from 80 to 100?", "25%"),
    ("Calculate the percent decrease from 200 to 150", "25%"),
    ("Convert 100 kilometers to miles", "62.1371"),
    ("Convert 25 celsius to fahrenheit", "77"),
    ("What is 0 degrees Celsius in Kelvin?", "273.15"),
    ("Convert 10 kg to pounds", "22.0462"),
    ("Sort in alphabetical order: banana, apple, cherry", "apple, banana, cherry"),
]

MUST_PASS: list[str] = [
    # Word problems: computable, but only via the PAL solver — never by regex.
    "A shop sells a jacket for $80 after a 20% discount. Calculate the original price.",
    "If a train travels 240 km in 3 hours, how many kilometers does it travel in 5 hours?",
    "A project started on March 3, 2021 and lasted 45 days, with a break. When did it end?",
    "What is the meaning of life?",
    "What is 42?",  # single number, nothing to compute
    "How many days does it take Mars to orbit the sun?",  # factual, not date math
    "How many letters do children typically learn first?",  # no quoted operand
    "Calculate the derivative of x^2 + 3x",  # symbolic, not arithmetic
    "What day of the week do most meetings happen?",  # no date present
    "What is the average of the test scores if Amy got 80 and Bob did better?",  # word problem
    "Convert 100 dollars to euros",  # currency: rates change, never deterministic
    "Convert 5 kilograms to miles",  # cross-dimension: must defer
    "What is the factorial of 25?",  # too large — expected format ambiguous
    "What is the square root of -4?",  # imaginary
    "Sort in alphabetical order: the quick brown fox jumps",  # not a comma list
    "What is the percentage increase from 0 to 50?",  # division by zero
]


def main() -> int:
    failures: list[str] = []
    for prompt, expected in MUST_ANSWER:
        got = try_deterministic(prompt)
        if got != expected:
            failures.append(f"ANSWER  {prompt!r}: expected {expected!r}, got {got!r}")
    for prompt in MUST_PASS:
        got = try_deterministic(prompt)
        if got is not None:
            failures.append(f"PASS-UP {prompt!r}: must return None, got {got!r}")
    if failures:
        print("DETERMINISTIC TESTS FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"DETERMINISTIC TESTS OK: {len(MUST_ANSWER)} answered, {len(MUST_PASS)} passed up")
    return 0


if __name__ == "__main__":
    sys.exit(main())

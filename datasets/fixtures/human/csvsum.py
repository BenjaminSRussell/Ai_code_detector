import csv
import sys
from collections import Counter


def totals(path, key, val):
    out = Counter()
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                out[row[key]] += float(row[val] or 0)
            except ValueError:
                continue  # junk rows from the export, skip
    return out


if __name__ == "__main__":
    for k, v in totals(*sys.argv[1:4]).most_common(10):
        print(f"{k:<30} {v:>12,.2f}")

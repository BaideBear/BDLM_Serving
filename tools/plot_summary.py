"""Plot a manually filled frozen30 CSV without rebuilding the archived results."""
import argparse
import csv
import math
from pathlib import Path

from render_report import chart, verify_figure

COLUMNS = ("row", "series", "tpf", "token_efficiency_pct", "correct")


def read_grid(path):
    rows, series, cells = {}, {}, {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise ValueError(f"expected CSV header: {','.join(COLUMNS)}")
        for entry in reader:
            try:
                if None in entry or None in entry.values():
                    raise ValueError("wrong number of columns")
                row, col, tpf, pct, correct = (entry[k].strip() for k in COLUMNS)
                if not row or not col or (row, col) in cells:
                    raise ValueError("blank or duplicate row/series key")
                rows[row] = series[col] = None
                value = None
                if any((tpf, pct, correct)):
                    if not all((tpf, pct, correct)):
                        raise ValueError("fill all three metrics or leave all three blank")
                    t, p, c = float(tpf), float(pct), int(correct)
                    if not math.isfinite(t) or t < 0 or not 0 <= p <= 100 or not 0 <= c <= 30:
                        raise ValueError("expected finite TPF >= 0, efficiency in [0,100], correct in [0,30]")
                    value = dict(tpf=t, efficiency=p/100, correct=c)
                cells[row, col] = value
            except ValueError as exc:
                raise ValueError(f"{path}:{reader.line_num}: {exc}") from exc
    if not rows:
        raise ValueError("no configurations in CSV")
    return list(series), [(r, [cells.get((r, s)) for s in series]) for r in rows]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv", type=Path)
    p.add_argument("--out", type=Path, required=True, help="output path stem (writes .png and .svg)")
    p.add_argument("--title", default="Frozen 30: manual summary")
    args = p.parse_args()
    if args.out.suffix:
        p.error("--out must be a path stem without a file extension")
    try:
        grid = read_grid(args.csv)
    except (ValueError, OSError) as exc:
        p.error(str(exc))
    chart(args.out, None, grid, title=args.title.strip() or "Frozen 30: manual summary")
    verify_figure(args.out, grid)
    print(f"{args.out}.png\n{args.out}.svg")


if __name__ == "__main__":
    main()

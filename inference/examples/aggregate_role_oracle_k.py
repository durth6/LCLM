"""Aggregate three-seed Full500 offline role-oracle-K curves."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


METRICS = (
    "answer_recall",
    "key_answer_recall",
    "support_recall",
    "key_answer_group_fraction",
    "support_group_fraction",
    "joint_role_recall",
)


def _mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def aggregate(
    template: str,
    tags: list[str],
    seeds: list[int],
    expected_rows: int,
    expected_valid: int,
) -> dict[str, Any]:
    lengths = []
    for tag in tags:
        sources = []
        curve_by_ratio: dict[float, list[dict[str, Any]]] = {}
        minima: dict[str, list[int]] | None = None
        ratios: tuple[float, ...] | None = None
        for seed in seeds:
            path = Path(template.format(tag=tag, seed=seed))
            if not path.is_file():
                raise FileNotFoundError(path)
            report = json.loads(path.read_text())
            if len(report.get("rows", [])) != expected_rows:
                raise ValueError(f"{path}: expected {expected_rows} rows")
            if int(report.get("selector_metadata", {}).get("seed", -1)) != seed:
                raise ValueError(f"{path}: selector seed mismatch")
            ids = [str(row["id"]) for row in report["rows"]]
            if len(ids) != len(set(ids)):
                raise ValueError(f"{path}: duplicate query IDs")
            sources.append(str(path))
            report_ratios = tuple(float(value) for value in report["budget_ratios"])
            if ratios is None:
                ratios = report_ratios
                minima = {str(ratio): [] for ratio in ratios}
                curve_by_ratio = {ratio: [] for ratio in ratios}
            elif report_ratios != ratios:
                raise ValueError(f"{tag}: budget ratios differ across seeds")
            for point_index, ratio in enumerate(ratios):
                points = [row["budget_curve"][point_index] for row in report["rows"]]
                valid = [point for point in points if point.get("joint_role_recall") is not None]
                if len(valid) != expected_valid:
                    raise ValueError(f"{tag}/s{seed}: expected {expected_valid} role-valid rows")
                curve_by_ratio[ratio].append({
                    "seed": seed,
                    "role_valid_examples": len(valid),
                    **{
                        f"{metric}_mean": _mean([
                            float(point[metric]) for point in valid if point.get(metric) is not None
                        ])
                        for metric in METRICS
                    },
                    "joint_complete_fraction": _mean([
                        float(point["joint_role_recall"] >= 1.0) for point in valid
                    ]),
                })
            assert minima is not None
            values = [row.get("oracle_min_ratio") for row in report["rows"]]
            for ratio in ratios:
                minima[str(ratio)].append(sum(value == ratio for value in values))

        assert ratios is not None and minima is not None
        curve = []
        for ratio in ratios:
            seed_points = curve_by_ratio[ratio]
            aggregate_point: dict[str, Any] = {
                "ratio": ratio,
                "role_valid_examples_per_seed": expected_valid,
                "seeds": seed_points,
            }
            for metric in (*[f"{name}_mean" for name in METRICS], "joint_complete_fraction"):
                values = [float(point[metric]) for point in seed_points if point.get(metric) is not None]
                aggregate_point[metric] = _mean(values)
                aggregate_point[f"{metric}_sample_sd"] = (
                    statistics.stdev(values) if len(values) > 1 else 0.0
                )
            curve.append(aggregate_point)
        lengths.append({
            "length": tag,
            "sources": sources,
            "curve": curve,
            "oracle_min_ratio_distribution_per_seed": minima,
        })
    return {
        "schema_version": 1,
        "scope": "offline declared-budget role-coverage curve; not an online policy or global subset oracle",
        "seeds": seeds,
        "expected_rows_per_seed_length": expected_rows,
        "role_valid_rows_per_seed_length": expected_valid,
        "lengths": lengths,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", required=True)
    parser.add_argument("--tags", default="4k,8k,16k,32k")
    parser.add_argument("--seeds", default="6201,6202,6203")
    parser.add_argument("--expected-rows", type=int, default=6500)
    parser.add_argument("--expected-valid", type=int, default=4000)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = aggregate(
        args.template,
        [value for value in args.tags.split(",") if value],
        [int(value) for value in args.seeds.split(",") if value],
        args.expected_rows,
        args.expected_valid,
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"out": str(output), "lengths": len(report["lengths"])}))


if __name__ == "__main__":
    main()

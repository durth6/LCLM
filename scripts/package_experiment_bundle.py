"""Package completed experiment outputs, tables, and dependency-free SVG charts.

The bundle uses hard links for large immutable JSONL outputs when the filesystem
permits it. This keeps the package self-contained on the same filesystem without
doubling the multi-gigabyte answer artifacts.
"""

from __future__ import annotations

import csv
import html
import json
import math
import os
from pathlib import Path
import shutil
from statistics import mean
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
BUNDLE = ARTIFACTS / "experiment_bundle_2026-09-22"


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _link_or_copy(source: Path, target: Path) -> str:
    _ensure_parent(target)
    if target.exists():
        target.unlink()
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def _copy_snapshot(source: Path, target: Path) -> None:
    _ensure_parent(target)
    shutil.copy2(source, target)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_json(path: Path, value: Any) -> None:
    _ensure_parent(path)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _copy_direct_files(source_dir: Path, target_dir: Path) -> int:
    count = 0
    if not source_dir.is_dir():
        return count
    for source in sorted(source_dir.iterdir()):
        if source.is_file():
            _link_or_copy(source, target_dir / source.name)
            count += 1
    return count


def _copy_tree(source: Path, target: Path, skip_dirs: set[str] | None = None) -> int:
    skip_dirs = skip_dirs or set()
    count = 0
    if not source.is_dir():
        return count
    for current, dirs, files in os.walk(source):
        dirs[:] = sorted(name for name in dirs if name not in skip_dirs)
        current_path = Path(current)
        relative = current_path.relative_to(source)
        for filename in sorted(files):
            source_file = current_path / filename
            _link_or_copy(source_file, target / relative / filename)
            count += 1
    return count


def _summary_rows() -> list[dict[str, Any]]:
    rows = []
    for summary_path in sorted((ARTIFACTS / "runs").glob("*/summary.json")):
        try:
            summary = _read_json(summary_path)
        except (OSError, json.JSONDecodeError):
            continue
        rows.append({
            "result_dir": summary_path.parent.name,
            "summary_source": str(summary_path.relative_to(ROOT)),
            "num_queries": summary.get("num_queries"),
            "ruler_accuracy": summary.get("ruler_accuracy"),
            "selection_ratio_actual_mean": summary.get("selection_ratio_actual_mean"),
            "answer_nll_retention_mean": summary.get("answer_nll_retention_mean"),
            "answer_nll_delta_vs_full_mean": summary.get("answer_nll_delta_vs_full_mean"),
            "key_answer_joint_recall_mean": summary.get("key_answer_joint_recall_mean"),
            "support_latent_recall_mean": summary.get("support_latent_recall_mean"),
            "decoder_kv_cache_bytes_estimate_mean": summary.get("decoder_kv_cache_bytes_estimate_mean"),
            "decoder_prefill_latency_ms_mean": summary.get("decoder_prefill_latency_ms_mean"),
            "cache_resident_ttft_ms_mean": summary.get("cache_resident_ttft_ms_mean"),
            "selector_latency_ms_mean": summary.get("selector_latency_ms_mean"),
            "selection_policy": summary.get("selection_policy"),
            "selector_mode": summary.get("selector_mode"),
            "dynamic_k": summary.get("dynamic_k"),
            "position_policy": summary.get("position_policy"),
        })
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    _ensure_parent(path)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _fixed_budget_rows() -> list[dict[str, Any]]:
    report = _read_json(ARTIFACTS / "runs/full500-fixed-budget-pareto.json")
    rows = []
    for point in report["points"]:
        metrics = point["metrics"]
        rows.append({
            "length": point["length"],
            "budget_ratio": point["budget_ratio"],
            "ruler_mean": metrics["ruler_accuracy"]["mean"],
            "ruler_sd": metrics["ruler_accuracy"]["sample_sd"],
            "retention_mean": metrics["answer_nll_retention_mean"]["mean"],
            "gap_mean": metrics["answer_nll_delta_vs_full_mean"]["mean"],
            "kv_mib_mean": metrics["decoder_kv_cache_bytes_estimate_mean"]["mean"] / 1048576,
            "prefill_ms_mean": metrics["decoder_prefill_latency_ms_mean"]["mean"],
            "resident_ttft_ms_mean": metrics["cache_resident_ttft_ms_mean"]["mean"],
        })
    return rows


def _oracle_k_rows() -> list[dict[str, Any]]:
    report = _read_json(ARTIFACTS / "runs/role-oracle-k-full500-aggregate.json")
    rows = []
    for length in report["lengths"]:
        for point in length["curve"]:
            rows.append({
                "length": length["length"],
                "budget_ratio": point["ratio"],
                "answer_recall_mean": point["answer_recall_mean"],
                "key_answer_recall_mean": point["key_answer_recall_mean"],
                "support_recall_mean": point["support_recall_mean"],
                "joint_role_recall_mean": point["joint_role_recall_mean"],
                "joint_complete_fraction": point["joint_complete_fraction"],
            })
    return rows


def _dynamic_rows() -> list[dict[str, Any]]:
    report = _read_json(ARTIFACTS / "runs/full500-role-aware-dynamic-vs-fixed50.json")
    rows = []
    for length in report["lengths"]:
        dynamic = length["dynamic"]
        fixed = length["fixed_50"]
        delta = length["delta_dynamic_minus_fixed_50"]
        rows.append({
            "length": length["length"],
            "dynamic_km_mean": dynamic["selection_ratio_actual_mean"]["mean"],
            "fixed50_km_mean": fixed["selection_ratio_actual_mean"]["mean"],
            "delta_km": delta["selection_ratio_actual_mean"],
            "dynamic_ruler_mean": dynamic["ruler_accuracy"]["mean"],
            "fixed50_ruler_mean": fixed["ruler_accuracy"]["mean"],
            "delta_ruler": delta["ruler_accuracy"],
            "dynamic_retention_mean": dynamic["answer_nll_retention_mean"]["mean"],
            "fixed50_retention_mean": fixed["answer_nll_retention_mean"]["mean"],
            "delta_retention": delta["answer_nll_retention_mean"],
            "dynamic_ttft_ms_mean": dynamic["cache_resident_ttft_ms_mean"]["mean"],
            "fixed50_ttft_ms_mean": fixed["cache_resident_ttft_ms_mean"]["mean"],
            "delta_ttft_ms": delta["cache_resident_ttft_ms_mean"],
            "dynamic_prefill_ms_mean": dynamic["decoder_prefill_latency_ms_mean"]["mean"],
            "fixed50_prefill_ms_mean": fixed["decoder_prefill_latency_ms_mean"]["mean"],
            "delta_prefill_ms": delta["decoder_prefill_latency_ms_mean"],
        })
    return rows


def _svg_escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _line_svg(
    path: Path,
    title: str,
    x_labels: list[str],
    series: dict[str, list[float]],
    ylabel: str,
    y_min: float = 0.0,
    y_max: float | None = None,
) -> None:
    width, height = 1040, 600
    left, right, top, bottom = 92, 40, 76, 86
    plot_w, plot_h = width - left - right, height - top - bottom
    all_values = [value for values in series.values() for value in values]
    if y_max is None:
        y_max = max(all_values) * 1.12 if all_values else 1.0
    if y_max <= y_min:
        y_max = y_min + 1.0
    colors = ["#166534", "#2563eb", "#b45309", "#be123c", "#6d28d9", "#0f766e"]

    def px(index: int) -> float:
        return left + (plot_w * index / max(1, len(x_labels) - 1))

    def py(value: float) -> float:
        return top + plot_h * (y_max - value) / (y_max - y_min)

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2:.1f}" y="34" text-anchor="middle" font-family="Arial" font-size="22" font-weight="700">{_svg_escape(title)}</text>',
        f'<text x="20" y="{top + plot_h / 2:.1f}" transform="rotate(-90 20 {top + plot_h / 2:.1f})" text-anchor="middle" font-family="Arial" font-size="14">{_svg_escape(ylabel)}</text>',
    ]
    for tick in range(6):
        value = y_min + (y_max - y_min) * tick / 5
        y = py(value)
        svg.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        svg.append(f'<text x="{left - 12}" y="{y + 5:.1f}" text-anchor="end" font-family="Arial" font-size="12">{value:.2f}</text>')
    svg.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{width - right}" y2="{top + plot_h}" stroke="#111827"/>')
    for index, label in enumerate(x_labels):
        x = px(index)
        svg.append(f'<line x1="{x:.1f}" y1="{top + plot_h}" x2="{x:.1f}" y2="{top + plot_h + 6}" stroke="#111827"/>')
        svg.append(f'<text x="{x:.1f}" y="{top + plot_h + 27}" text-anchor="middle" font-family="Arial" font-size="13">{_svg_escape(label)}</text>')
    for series_index, (label, values) in enumerate(series.items()):
        color = colors[series_index % len(colors)]
        points = " ".join(f"{px(i):.1f},{py(float(value)):.1f}" for i, value in enumerate(values))
        svg.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>')
        for i, value in enumerate(values):
            svg.append(f'<circle cx="{px(i):.1f}" cy="{py(float(value)):.1f}" r="5" fill="{color}"/>')
    legend_x = left
    for series_index, label in enumerate(series):
        color = colors[series_index % len(colors)]
        x = legend_x + series_index * 170
        svg.append(f'<line x1="{x}" y1="{height - 28}" x2="{x + 25}" y2="{height - 28}" stroke="{color}" stroke-width="3"/>')
        svg.append(f'<text x="{x + 32}" y="{height - 23}" font-family="Arial" font-size="13">{_svg_escape(label)}</text>')
    svg.append('</svg>')
    _ensure_parent(path)
    path.write_text("\n".join(svg) + "\n")


def _grouped_bar_svg(path: Path, title: str, labels: list[str], groups: dict[str, list[float]], ylabel: str) -> None:
    width, height = 1100, 620
    left, right, top, bottom = 92, 40, 76, 90
    plot_w, plot_h = width - left - right, height - top - bottom
    maximum = max(value for values in groups.values() for value in values) * 1.18
    colors = ["#2563eb", "#b45309", "#166534", "#be123c"]
    group_count = len(groups)
    bar_width = plot_w / len(labels) / (group_count + 1)
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2:.1f}" y="34" text-anchor="middle" font-family="Arial" font-size="22" font-weight="700">{_svg_escape(title)}</text>',
        f'<text x="20" y="{top + plot_h / 2:.1f}" transform="rotate(-90 20 {top + plot_h / 2:.1f})" text-anchor="middle" font-family="Arial" font-size="14">{_svg_escape(ylabel)}</text>',
    ]
    for tick in range(6):
        value = maximum * tick / 5
        y = top + plot_h * (1 - value / maximum)
        svg.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        svg.append(f'<text x="{left - 12}" y="{y + 5:.1f}" text-anchor="end" font-family="Arial" font-size="12">{value:.2f}</text>')
    svg.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{width - right}" y2="{top + plot_h}" stroke="#111827"/>')
    for label_index, label in enumerate(labels):
        base = left + label_index * plot_w / len(labels) + bar_width / 2
        svg.append(f'<text x="{base + bar_width * group_count / 2:.1f}" y="{top + plot_h + 28}" text-anchor="middle" font-family="Arial" font-size="13">{_svg_escape(label)}</text>')
        for group_index, (group, values) in enumerate(groups.items()):
            value = float(values[label_index])
            x = base + group_index * bar_width
            y = top + plot_h * (1 - value / maximum)
            h = top + plot_h - y
            svg.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width * 0.86:.1f}" height="{h:.1f}" fill="{colors[group_index % len(colors)]}"/>')
    for group_index, group in enumerate(groups):
        x = left + group_index * 180
        color = colors[group_index % len(colors)]
        svg.append(f'<rect x="{x}" y="{height - 32}" width="16" height="16" fill="{color}"/>')
        svg.append(f'<text x="{x + 23}" y="{height - 19}" font-family="Arial" font-size="13">{_svg_escape(group)}</text>')
    svg.append('</svg>')
    _ensure_parent(path)
    path.write_text("\n".join(svg) + "\n")


def _heatmap_svg(path: Path, title: str, rows: list[str], columns: list[str], values: list[list[float]]) -> None:
    width, height = 1240, max(520, 150 + len(rows) * 28)
    left, top, cell_w, cell_h = 240, 90, 82, 26
    maximum = max(max(row) for row in values) if values else 1.0
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2:.1f}" y="38" text-anchor="middle" font-family="Arial" font-size="22" font-weight="700">{_svg_escape(title)}</text>',
    ]
    for column_index, column in enumerate(columns):
        x = left + column_index * cell_w + cell_w / 2
        svg.append(f'<text x="{x:.1f}" y="76" text-anchor="middle" font-family="Arial" font-size="12">{_svg_escape(column)}</text>')
    for row_index, row_label in enumerate(rows):
        y = top + row_index * cell_h
        svg.append(f'<text x="{left - 12}" y="{y + 18}" text-anchor="end" font-family="Arial" font-size="11">{_svg_escape(row_label)}</text>')
        for column_index, value in enumerate(values[row_index]):
            x = left + column_index * cell_w
            normalized = max(0.0, min(1.0, float(value) / max(100.0, maximum)))
            red = int(239 - normalized * 170)
            green = int(246 - normalized * 90)
            blue = int(255 - normalized * 15)
            color = f"rgb({red},{green},{blue})"
            svg.append(f'<rect x="{x}" y="{y}" width="{cell_w - 2}" height="{cell_h - 2}" fill="{color}" stroke="#ffffff"/>')
            svg.append(f'<text x="{x + cell_w / 2:.1f}" y="{y + 17}" text-anchor="middle" font-family="Arial" font-size="11">{float(value):.1f}</text>')
    svg.append('</svg>')
    _ensure_parent(path)
    path.write_text("\n".join(svg) + "\n")


def _build_charts() -> None:
    charts = BUNDLE / "charts"
    fixed = _fixed_budget_rows()
    lengths = ["4k", "8k", "16k", "32k"]
    budgets = [0.25, 0.5, 0.75, 1.0]
    _line_svg(
        charts / "full500_fixed_budget_ruler.svg",
        "Full500 fixed-budget RULER accuracy",
        ["25%", "50%", "75%", "100%"],
        {length: [next(row["ruler_mean"] for row in fixed if row["length"] == length and row["budget_ratio"] == budget) for budget in budgets] for length in lengths},
        "RULER accuracy (points)",
        y_max=100,
    )
    _line_svg(
        charts / "full500_fixed_budget_retention.svg",
        "Full500 fixed-budget answer-NLL retention",
        ["25%", "50%", "75%", "100%"],
        {length: [next(row["retention_mean"] for row in fixed if row["length"] == length and row["budget_ratio"] == budget) for budget in budgets] for length in lengths},
        "Retention",
        y_max=1.05,
    )
    dynamic = _dynamic_rows()
    _grouped_bar_svg(
        charts / "dynamic_vs_fixed50_budget.svg",
        "Guarded dynamic-K vs matched fixed-50 budget",
        lengths,
        {"dynamic K/M": [row["dynamic_km_mean"] * 100 for row in dynamic], "fixed 50% K/M": [row["fixed50_km_mean"] * 100 for row in dynamic]},
        "K/M (%)",
    )
    _grouped_bar_svg(
        charts / "dynamic_vs_fixed50_ruler.svg",
        "Guarded dynamic-K vs matched fixed-50 RULER",
        lengths,
        {"dynamic": [row["dynamic_ruler_mean"] for row in dynamic], "fixed 50%": [row["fixed50_ruler_mean"] for row in dynamic]},
        "RULER accuracy (points)",
    )
    _grouped_bar_svg(
        charts / "dynamic_vs_fixed50_ttft.svg",
        "Guarded dynamic-K vs matched fixed-50 resident TTFT",
        lengths,
        {"dynamic": [row["dynamic_ttft_ms_mean"] for row in dynamic], "fixed 50%": [row["fixed50_ttft_ms_mean"] for row in dynamic]},
        "Resident TTFT (ms)",
    )
    oracle = _oracle_k_rows()
    _line_svg(
        charts / "role_oracle_k_recall.svg",
        "Offline role-oracle K curve: answer recall",
        ["25%", "50%", "75%", "100%"],
        {length: [next(row["answer_recall_mean"] for row in oracle if row["length"] == length and row["budget_ratio"] == budget) for budget in budgets] for length in lengths},
        "All-answer latent recall",
        y_max=1.05,
    )
    _line_svg(
        charts / "role_oracle_k_joint_complete.svg",
        "Offline role-oracle K curve: joint role completion",
        ["25%", "50%", "75%", "100%"],
        {length: [next(row["joint_complete_fraction"] for row in oracle if row["length"] == length and row["budget_ratio"] == budget) for budget in budgets] for length in lengths},
        "Joint complete fraction",
        y_max=1.05,
    )

    task_values: dict[str, dict[str, list[float]]] = {}
    for length in lengths:
        task_values[length] = {}
        for seed in (6201, 6202, 6203):
            path = ARTIFACTS / "runs" / f"full500-group-relevance-v2-25pct-{length}-s{seed}" / "summary.json"
            if not path.is_file():
                continue
            summary = _read_json(path)
            for task, value in summary.get("per_task_accuracy", {}).items():
                task_values[length].setdefault(task, []).append(float(value))
    tasks = sorted({task for per_length in task_values.values() for task in per_length})
    heat_values = [[mean(task_values[length].get(task, [0.0])) for length in lengths] for task in tasks]
    _heatmap_svg(charts / "full500_group_relevance_task_heatmap.svg", "Full500 group relevance task accuracy", lengths, tasks, list(map(list, zip(*heat_values))) if False else heat_values)


def _package() -> dict[str, Any]:
    if BUNDLE.exists() and any(BUNDLE.iterdir()):
        raise RuntimeError(f"bundle already contains files: {BUNDLE}")
    for name in ("reports", "tables", "runs", "inputs", "checkpoints", "calibration", "cache_manifests", "logs", "charts"):
        (BUNDLE / name).mkdir(parents=True, exist_ok=True)
    copy_stats = {"hardlink": 0, "copy": 0, "files": 0}

    def add(source: Path, target: Path, snapshot: bool = False) -> None:
        if snapshot:
            _copy_snapshot(source, target)
            method = "copy"
        else:
            method = _link_or_copy(source, target)
        copy_stats[method] += 1
        copy_stats["files"] += 1

    for report_name in ("report.md", "report_results_2026-09-14.md"):
        add(ROOT / report_name, BUNDLE / "reports" / report_name, snapshot=True)

    for source in sorted((ARTIFACTS / "runs").glob("*.json")):
        add(source, BUNDLE / "tables" / "raw" / source.name)
    for source in sorted((ARTIFACTS / "runs/joint-role-recall-full500").glob("*.json")):
        add(source, BUNDLE / "tables" / "joint-role" / source.name)
    for source in sorted((ARTIFACTS / "calibration").rglob("*.json")):
        add(source, BUNDLE / "tables" / "calibration" / source.relative_to(ARTIFACTS / "calibration"))

    for run_dir in sorted((ARTIFACTS / "runs").iterdir()):
        if not run_dir.is_dir():
            continue
        if not any((run_dir / filename).is_file() for filename in ("answers.jsonl", "summary.json")):
            continue
        for source in sorted(run_dir.iterdir()):
            if source.is_file():
                add(source, BUNDLE / "runs" / run_dir.name / source.name)

    input_names = [
        "selector_train_mixed_4to32k",
        "dynamic_k_validation_v1",
        "dynamic_k_validation_v2",
    ]
    input_names.extend(path.name for path in sorted((ARTIFACTS / "data").glob("ruler_all_tasks_v2_*")))
    input_names.extend(path.name for path in sorted((ARTIFACTS / "data").glob("ruler_all_tasks_full500-v2_*")))
    for name in sorted(set(input_names)):
        source = ARTIFACTS / "data" / name
        for current, dirs, files in os.walk(source):
            dirs[:] = sorted(name for name in dirs if name != "cache")
            relative = Path(current).relative_to(source)
            for filename in sorted(files):
                add(Path(current) / filename, BUNDLE / "inputs" / name / relative / filename)

    for checkpoint_dir in sorted((ARTIFACTS / "checkpoints").iterdir()):
        if checkpoint_dir.is_dir():
            for source in sorted(checkpoint_dir.iterdir()):
                if source.is_file():
                    add(source, BUNDLE / "checkpoints" / checkpoint_dir.name / source.name)

    for source in sorted((ARTIFACTS / "logs").iterdir()):
        if source.is_file():
            add(source, BUNDLE / "logs" / source.name)

    cache_manifest_count = 0
    for base in (ARTIFACTS / "runs", ARTIFACTS / "data"):
        for source in sorted(base.rglob("manifest.json")):
            if "cache" not in source.parent.name and "cache" not in source.parent.parts:
                continue
            relative = source.relative_to(ROOT).with_suffix(".manifest.json")
            target_name = "__".join(relative.parts)
            add(source, BUNDLE / "cache_manifests" / target_name)
            cache_manifest_count += 1

    fixed_rows = _fixed_budget_rows()
    oracle_rows = _oracle_k_rows()
    dynamic_rows = _dynamic_rows()
    inventory = _summary_rows()
    _write_csv(BUNDLE / "tables/full500_fixed_budget.csv", fixed_rows, list(fixed_rows[0]))
    _write_csv(BUNDLE / "tables/role_oracle_k.csv", oracle_rows, list(oracle_rows[0]))
    _write_csv(BUNDLE / "tables/dynamic_vs_fixed50.csv", dynamic_rows, list(dynamic_rows[0]))
    _write_csv(BUNDLE / "tables/run_inventory.csv", inventory, list(inventory[0]))
    _build_charts()

    manifest = {
        "schema_version": 1,
        "created": "2026-09-22",
        "source_root": str(ROOT),
        "bundle_root": str(BUNDLE),
        "copy_stats": copy_stats,
        "result_run_directories": len({row["result_dir"] for row in inventory}),
        "result_summary_rows": len(inventory),
        "cache_manifests": cache_manifest_count,
        "raw_latent_cache_policy": "not duplicated; manifests are included under cache_manifests and retain source paths",
        "tables": [
            "tables/full500_fixed_budget.csv",
            "tables/role_oracle_k.csv",
            "tables/dynamic_vs_fixed50.csv",
            "tables/run_inventory.csv",
        ],
        "charts": sorted(path.name for path in (BUNDLE / "charts").glob("*.svg")),
    }
    _write_json(BUNDLE / "manifest.json", manifest)
    return manifest


def main() -> None:
    manifest = _package()
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

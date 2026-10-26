import json

from inference.examples.aggregate_dynamic_k_full500 import aggregate


def _write_run(path, *, dynamic, ratio, score):
    path.mkdir(parents=True)
    summary = {
        "num_queries": 2,
        "position_policy": "full_layout",
        "dynamic_k": dynamic,
        "selection_ratio_actual_mean": ratio,
        "ruler_accuracy": score,
        "answer_nll_retention_mean": 0.8,
        "answer_nll_delta_vs_full_mean": 0.2,
        "key_answer_joint_recall_mean": 0.9,
        "support_latent_recall_mean": 0.7,
        "decoder_kv_cache_bytes_estimate_mean": 1.0,
        "decoder_prefill_latency_ms_mean": 2.0,
        "cache_resident_ttft_ms_mean": 3.0,
        "selector_latency_ms_mean": 4.0,
        "per_task_accuracy": {"task": score},
    }
    if dynamic:
        summary.update({
            "dynamic_calibration_policy": "test",
            "dynamic_calibration_temperature": 1.0,
            "dynamic_calibration_qualified_against_fixed_50": True,
            "dynamic_calibration_fallback_ratio": None,
        })
    (path / "summary.json").write_text(json.dumps(summary))
    rows = []
    for index in range(2):
        rows.append({
            "id": str(index),
            "position_policy": "full_layout",
            "selected_latent_indices": [0, 1],
            "selection_ratio": ratio,
            "answer_nll_selected": 1.0,
            "selection_policy": "dynamic_coverage" if dynamic else "relevance_topk",
        })
    (path / "answers.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_dynamic_aggregate_audits_rows_and_computes_delta(tmp_path):
    for seed in (1, 2):
        _write_run(tmp_path / f"dynamic-4k-s{seed}", dynamic=True, ratio=0.25, score=60 + seed)
        _write_run(tmp_path / f"fixed-4k-s{seed}", dynamic=False, ratio=0.5, score=50 + seed)
    report = aggregate(
        "dynamic-4k-s{seed}",
        "fixed-4k-s{seed}",
        tmp_path,
        ["4k"],
        [1, 2],
        2,
    )
    point = report["lengths"][0]
    assert point["dynamic"]["ruler_accuracy"]["mean"] == 61.5
    assert point["fixed_50"]["ruler_accuracy"]["mean"] == 51.5
    assert point["delta_dynamic_minus_fixed_50"]["ruler_accuracy"] == 10.0
    assert point["dynamic_row_audits"][0]["budget_distribution"] == {"0.25": 2}

from argparse import Namespace

from inference.examples.eval_query_aware_niah import (
    _attach_role_metrics,
    _build_summary,
    _percentile,
)


def _args(**overrides):
    values = {
        "checkpoint": "checkpoint",
        "k_decoder": None,
        "selection_ratio": 0.25,
        "selector_mode": "learned",
        "neighbor_radius": 0,
        "repeat_each_query": 2,
        "max_resident_caches": 1,
        "reuse_queries": 4,
        "position_policy": "full_layout",
    }
    values.update(overrides)
    return Namespace(**values)


def _record(record_id, cache_hit, load_ms, ttft_ms, response="ok", first_load=None):
    return {
        "id": record_id,
        "task": "task",
        "answers": ["ok"],
        "response": response,
        "cache_hit": cache_hit,
        "cache_first_load": (not cache_hit) if first_load is None else first_load,
        "cache_load_latency_ms": load_ms,
        "cache_resident_ttft_ms": ttft_ms - load_ms,
        "ttft_ms": ttft_ms,
        "question_tokenize_latency_ms": 1.0,
        "selector_latency_ms": 2.0,
        "selector_peak_memory_bytes": 3,
        "adapter_latency_ms": 4.0,
        "adapter_peak_memory_bytes": 5,
        "decoder_input_latency_ms": 6.0,
        "decoder_prefill_latency_ms": 7.0,
        "decoder_decode_latency_ms": 8.0,
        "decoder_peak_memory_bytes": 9,
        "total_query_latency_ms": 20.0,
        "decoder_kv_cache_bytes_estimate": 10.0,
        "selection_ratio": 0.25,
        "oracle_latent_recall": 0.5,
    }


def test_percentile_uses_interpolation_and_empty_is_zero():
    records = [{"latency": value} for value in (10, 20, 40, 80)]
    assert _percentile(records, "latency", 50) == 30
    assert _percentile(records, "latency", 95) == 74
    assert _percentile([], "latency", 50) == 0.0


def test_summary_separates_cold_load_and_resident_ttft():
    records = [
        _record("a", False, 100.0, 130.0),
        _record("a", True, 0.0, 30.0),
        _record("b", False, 20.0, 50.0),
        _record("b", True, 0.0, 30.0),
    ]
    summary = _build_summary(
        _args(),
        [{"encoder_latency_ms": 400.0}],
        records,
    )

    assert summary["cache_hit_rate"] == 0.5
    assert summary["cache_cold_start_count"] == 2
    assert summary["cache_miss_count"] == 2
    assert summary["cache_resident_count"] == 2
    assert summary["cache_cold_start_load_latency_ms_p50"] == 60.0
    assert summary["cache_cold_start_load_latency_ms_p95"] == 96.0
    assert summary["cache_resident_ttft_ms_mean"] == 30.0
    assert summary["cache_hit_ttft_ms_mean"] == 30.0
    assert summary["encoder_latency_ms_amortized"] == 100.0
    assert summary["encoder_latency_ms_amortized_by_reuse"] == {
        "1": 400.0,
        "4": 100.0,
        "16": 25.0,
    }
    assert summary["selector_latency_ms_mean"] == 2.0
    assert summary["selector_peak_memory_bytes_max"] == 3
    assert summary["position_policy"] == "full_layout"


def test_summary_marks_unavailable_role_labels_as_null_with_valid_count():
    summary = _build_summary(
        _args(),
        [{"encoder_latency_ms": 1.0}],
        [_record("a", True, 0.0, 10.0)],
    )
    assert summary["key_answer_joint_recall_mean"] is None
    assert summary["support_latent_recall_mean"] is None
    assert summary["key_answer_joint_recall_valid_count"] == 0
    assert summary["support_latent_recall_valid_count"] == 0


def test_summary_averages_present_role_labels_only():
    records = [_record("a", True, 0.0, 10.0), _record("b", True, 0.0, 10.0)]
    records[0]["key_answer_joint_recall"] = 0.25
    records[1]["key_answer_joint_recall"] = 0.75
    summary = _build_summary(_args(), [{"encoder_latency_ms": 1.0}], records)
    assert summary["key_answer_joint_recall_mean"] == 0.5
    assert summary["key_answer_joint_recall_valid_count"] == 2


def test_summary_aggregates_selected_source_spans_globally_and_per_task():
    records = [_record("a", True, 0.0, 10.0), _record("b", True, 0.0, 10.0)]
    records[0].update({
        "selected_source_start_min": 4,
        "selected_source_end_max": 20,
        "selected_source_span_mean": 8.0,
        "selected_source_token_extent": 16,
        "selected_source_run_count": 1,
    })
    records[1].update({
        "selected_source_start_min": 12,
        "selected_source_end_max": 44,
        "selected_source_span_mean": 10.0,
        "selected_source_token_extent": 32,
        "selected_source_run_count": 3,
    })
    summary = _build_summary(_args(), [{"encoder_latency_ms": 1.0}], records)
    span = summary["selected_source_span_summary"]
    assert span["selected_source_token_extent"] == {
        "mean": 24.0, "p50": 24.0, "p95": 31.2, "valid_count": 2,
    }
    assert summary["selected_source_run_count_mean"] == 2.0
    assert summary["selected_source_run_count_p95"] == 2.9
    assert summary["per_task_source_span"]["task"]["selected_source_span_mean"]["mean"] == 9.0


def test_ruler_part_scoring_accepts_any_reference():
    from inference.examples.eval_query_aware_niah import _ruler_score

    assert _ruler_score(["alpha", "beta"], "BETA", "ruler_string_match_part") == 100.0
    assert _ruler_score(["alpha", "beta"], "alpha", "ruler_string_match_all") == 50.0


def test_role_join_enriches_historical_answers_without_source_text(tmp_path):
    roles = tmp_path / "oracle_latents_roles.jsonl"
    roles.write_text(
        '{"id":"q1","document_id":"d1","cache_fingerprint":"fp",'
        '"role_latent_indices":{"answer":[2],"all_answer":[2,3],'
        '"key_answer":[1,2,3],"support":[0,1,2,3]},'
        '"role_labels_available":true}\n'
    )
    records = [{
        "id": "q1",
        "document_id": "d1",
        "cache_fingerprint": "fp",
        "selected_latent_indices": [1, 2, 8],
    }]
    enriched = _attach_role_metrics(records, roles)
    assert enriched[0]["answer_latent_recall"] == 1.0
    assert enriched[0]["all_answer_latent_recall"] == 0.5
    assert enriched[0]["key_answer_joint_recall"] == 2 / 3
    assert enriched[0]["support_latent_recall"] == 0.5
    assert "context" not in enriched[0]


def test_role_join_keeps_unavailable_roles_as_none(tmp_path):
    roles = tmp_path / "roles.jsonl"
    roles.write_text(
        '{"id":"q1","document_id":"d1","oracle_latent_indices":[2],'
        '"role_labels_available":false}\n'
    )
    records = [{
        "id": "q1", "document_id": "d1", "selected_latent_indices": [2],
    }]
    enriched = _attach_role_metrics(records, roles)
    assert enriched[0]["answer_latent_recall"] == 1.0
    assert enriched[0]["all_answer_latent_recall"] == 1.0
    assert enriched[0]["key_answer_joint_recall"] is None
    assert enriched[0]["support_latent_recall"] is None

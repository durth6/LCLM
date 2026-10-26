import json

from inference.examples.analyze_joint_role_recall import analyze


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_joint_completion_fraction_excludes_role_unavailable_rows(tmp_path):
    answers = tmp_path / "answers.jsonl"
    roles = tmp_path / "roles.jsonl"
    _write_jsonl(answers, [
        {
            "id": "complete",
            "task": "niah",
            "answers": ["ok"],
            "response": "ok",
            "selected_latent_indices": [1, 3],
        },
        {
            "id": "missing",
            "task": "niah",
            "answers": ["ok"],
            "response": "no",
            "selected_latent_indices": [1],
        },
        {
            "id": "unavailable",
            "task": "qa",
            "answers": ["ok"],
            "response": "ok",
            "selected_latent_indices": [1],
        },
    ])
    _write_jsonl(roles, [
        {
            "id": "complete",
            "role_latent_indices": {"key_answer_0": [1], "support_0": [3]},
        },
        {
            "id": "missing",
            "role_latent_indices": {"key_answer_0": [1], "support_0": [3]},
        },
        {"id": "unavailable", "role_latent_indices": {}},
    ])

    report = analyze([("model", answers)], roles)
    summary = report["sources"]["model"]
    assert report["schema_version"] == 2
    assert summary["examples"] == 3
    assert summary["role_valid_examples"] == 2
    assert summary["role_unavailable_examples"] == 1
    assert summary["complete_joint_count"] == 1
    assert summary["complete_joint_fraction"] == 0.5

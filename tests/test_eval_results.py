import json

from evals.results import load_latest_result, load_model_results


def test_load_model_results_reports_missing_models(tmp_path) -> None:
    results = load_model_results(tmp_path)

    assert results["base"]["status"] == "missing"
    assert results["finetuned"]["precision"] is None
    assert results["gpt4o"]["samples"] == 0


def test_load_model_results_reads_existing_result(tmp_path) -> None:
    result_path = tmp_path / "base.json"
    result_path.write_text(
        json.dumps(
            {
                "model": "base",
                "samples": 10,
                "precision": 0.5,
                "recall": 0.4,
                "f1": 0.44,
                "quality": 2.1,
            }
        ),
        encoding="utf-8",
    )

    results = load_model_results(tmp_path)

    assert results["base"]["precision"] == 0.5
    assert results["finetuned"]["status"] == "missing"


def test_load_latest_result_reports_not_run_when_missing(tmp_path) -> None:
    latest = load_latest_result(tmp_path)

    assert latest["status"] == "not_run"

"""Port of packages/evals/test/comparison.test.ts."""

from pi.evals.report import format_eval_comparison_report, summarize_eval_observations


def scored(variant, run_number, score, **metrics):
    observation = {
        "evalSet": "tool access",
        "caseId": "create",
        "variant": variant,
        "model": "fixture/model",
        "runNumber": run_number,
        "outcome": "scored",
        "score": score,
        "totalTokens": 100,
        "toolCalls": 2,
        "totalMs": 1000,
        "estimatedCostUsd": 0.01,
    }
    observation.update(metrics)
    return observation


def errored(variant, run_number, total_tokens=None):
    observation = {
        "evalSet": "tool access",
        "caseId": "create",
        "variant": variant,
        "model": "fixture/model",
        "runNumber": run_number,
        "outcome": "errored",
    }
    if total_tokens is not None:
        observation["totalTokens"] = total_tokens
    return observation


def expected_for(run_numbers):
    expected = []
    for run_number in run_numbers:
        for variant in ("without_docs", "with_docs"):
            expected.append({"evalSet": "tool access", "caseId": "create", "variant": variant,
                             "model": "fixture/model", "runNumber": run_number})
    return expected


def test_computes_paired_lift_and_efficiency_deltas():
    observations = [
        scored("without_docs", 1, 0, totalTokens=100, toolCalls=3, totalMs=1000),
        scored("with_docs", 1, 1, totalTokens=120, toolCalls=2, totalMs=800),
        scored("without_docs", 2, 1, totalTokens=200),
        scored("with_docs", 2, 1, totalTokens=180),
    ]
    report = summarize_eval_observations("digest", expected_for([1, 2]), observations)
    comparison = report["comparisons"][0]
    assert comparison["evalSet"] == "tool access"
    assert comparison["totalPairs"] == 2
    assert comparison["eligiblePairs"] == 2
    assert comparison["blockedPairs"] == 0
    assert comparison["controlPassRate"] == 0.5
    assert comparison["treatmentPassRate"] == 1
    assert comparison["lift"] == 0.5
    assert comparison["totalTokens"] == {"eligiblePairs": 2, "controlMean": 150, "treatmentMean": 150, "meanDelta": 0}
    assert comparison["toolCalls"] == {"eligiblePairs": 2, "controlMean": 2.5, "treatmentMean": 2, "meanDelta": -0.5}


def test_fails_closed_for_incomplete_and_errored_pairs_while_retaining_totals():
    observations = [
        scored("without_docs", 1, 0),
        errored("with_docs", 1, 120),
        scored("without_docs", 2, 1, totalTokens=200),
    ]
    report = summarize_eval_observations("digest", expected_for([1, 2]), observations)
    comparison = report["comparisons"][0]
    assert comparison["totalPairs"] == 2
    assert comparison["eligiblePairs"] == 0
    assert comparison["blockedPairs"] == 2
    assert comparison["lift"] is None
    assert [(pair["runNumber"], pair["reasons"]) for pair in report["blockedPairs"]] == [
        (1, ["with_docs: errored"]),
        (2, ["with_docs: expected 1 observation, found 0"]),
    ]
    assert report["operationalTotals"][0]["totalTokens"] == {"availableRuns": 2, "total": 300}


def test_blocks_duplicate_observations_and_keeps_missing_metrics_distinct_from_zero():
    without_tokens = scored("without_docs", 1, 1)
    del without_tokens["totalTokens"]
    report = summarize_eval_observations("digest", expected_for([1]), [
        without_tokens,
        without_tokens,
        scored("with_docs", 1, 1, totalTokens=0),
    ])
    assert report["blockedPairs"][0]["reasons"] == ["without_docs: expected 1 observation, found 2"]
    assert report["operationalTotals"][0]["totalTokens"] == {"availableRuns": 0, "total": None}
    assert report["operationalTotals"][1]["totalTokens"] == {"availableRuns": 1, "total": 0}


def test_formats_blocked_comparisons_and_operational_totals():
    report = summarize_eval_observations("digest", expected_for([1, 2]), [
        scored("without_docs", 1, 1),
        scored("with_docs", 1, 1),
    ])
    formatted = format_eval_comparison_report(report)
    assert "Documentation Eval Comparisons" in formatted
    assert "Pass rate  withheld because pairs are blocked" in formatted
    assert "without_docs: 1 runs" in formatted

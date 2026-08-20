from spider.diversity import (
    audit_records,
    balanced_group_sample,
    macro_boolean_metric,
    sampling_unit,
)


def _records() -> list[dict]:
    records = []
    for domain, count in (("large.test", 20), ("medium.test", 8), ("small.test", 2)):
        for index in range(count):
            records.append(
                {
                    "id": f"{domain}-{index}",
                    "domain": domain,
                    "task": "qa",
                    "image": f"images/{domain}-{index // 2}.jpg",
                    "exact_match": index % 2 == 0,
                }
            )
    return records


def test_audit_reports_concentration() -> None:
    audit = audit_records(_records())
    assert audit["examples"] == 30
    assert audit["unique_domains"] == 3
    assert audit["max_domain_share"] == 20 / 30
    assert audit["unique_sampling_units"] == 15


def test_balanced_sample_is_deterministic_and_capped() -> None:
    records = _records()
    first = balanced_group_sample(
        records, 10, seed=7, temperature=0.5, max_domain_share=0.5, max_per_unit=1
    )
    second = balanced_group_sample(
        records, 10, seed=7, temperature=0.5, max_domain_share=0.5, max_per_unit=1
    )
    assert [record["id"] for record in first] == [record["id"] for record in second]
    audit = audit_records(first)
    assert audit["max_domain_share"] <= 0.5
    assert audit["unique_sampling_units"] == len(first)


def test_macro_metric_weights_domains_equally() -> None:
    records = [
        {"domain": "a", "correct": True},
        {"domain": "a", "correct": True},
        {"domain": "a", "correct": False},
        {"domain": "b", "correct": False},
    ]
    metric = macro_boolean_metric(records, "correct")
    assert metric["micro"] == 0.5
    assert metric["macro"] == 1 / 3


def test_action_uses_trajectory_as_sampling_unit() -> None:
    assert sampling_unit({"id": "x", "trajectory_id": "t", "image": "a.jpg"}) == (
        "trajectory:t"
    )

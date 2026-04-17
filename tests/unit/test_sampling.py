from collections import Counter

from ibm650_it.dataset.sampling import stable_limit_records, stable_weighted_band_sample


def _records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for band in ["B0", "B1", "B2"]:
        for index in range(4):
            records.append(
                {
                    "id": f"{band.lower()}_{index}",
                    "band": band,
                    "source": {"it_text_v1": f"accepted/{band}/{index:04d}/source.it"},
                }
            )
    return records


def test_stable_limit_records_balances_across_bands() -> None:
    selected = stable_limit_records(_records(), 6)

    counts = Counter(str(record["band"]) for record in selected)
    assert counts == {"B0": 2, "B1": 2, "B2": 2}


def test_stable_limit_records_is_deterministic() -> None:
    first = [str(record["id"]) for record in stable_limit_records(_records(), 5)]
    second = [str(record["id"]) for record in stable_limit_records(_records(), 5)]

    assert first == second


def test_stable_weighted_band_sample_biases_toward_weighted_band() -> None:
    selected = stable_weighted_band_sample(_records(), 6, band_weights={"B0": 3})

    counts = Counter(str(record["band"]) for record in selected)
    assert counts == {"B0": 4, "B1": 1, "B2": 1}


def test_stable_weighted_band_sample_is_deterministic() -> None:
    first = [str(record["id"]) for record in stable_weighted_band_sample(_records(), 7, band_weights={"B2": 4})]
    second = [str(record["id"]) for record in stable_weighted_band_sample(_records(), 7, band_weights={"B2": 4})]

    assert first == second

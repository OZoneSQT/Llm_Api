import json
import sys
import types
from pathlib import Path


dummy_datasets_module = types.ModuleType("datasets")


def _placeholder_load_dataset(*_args, **_kwargs):
    raise RuntimeError("load_dataset stub not patched")


setattr(dummy_datasets_module, 'load_dataset', _placeholder_load_dataset)
sys.modules.setdefault("datasets", dummy_datasets_module)


from Training.app.use_cases import dataset_preparation
from Training.domain.entities import DatasetPreparationRequest, DatasetSpec


class DummyRecords(list):
    def select(self, indices):
        return DummyRecords([self[i] for i in indices])


def test_prepare_dataset_aggregates_sources(monkeypatch, tmp_path: Path) -> None:
    sanitized = tmp_path / "sanitized.jsonl"
    sanitized.write_text(
        """
{"text": "clean-1"}
{"prompt": "clean-2"}
{"content": "clean-ignored"}
""".strip(),
        encoding="utf-8",
    )

    custom = tmp_path / "custom.txt"
    custom.write_text("first custom\nsecond custom\nthird custom\n", encoding="utf-8")

    good_records = DummyRecords(
        [
            {"text": "remote-good-1"},
            {"instruction": "remote-good-2"},
            {"content": "remote-good-extra"},
        ]
    )
    fallback_records = DummyRecords(
        [
            {"content": "fallback-1"},
            {"text": "fallback-2"},
        ]
    )

    load_calls = []

    def fake_load_dataset(repo_id, split=None):
        load_calls.append((repo_id, split))
        if repo_id == "good/dataset" and split is not None:
            limit = int(split.split(':')[1].strip('[]'))
            return DummyRecords(good_records[:limit])
        if repo_id == "fallback/dataset" and split is not None:
            raise RuntimeError("split load failure")
        if repo_id == "fallback/dataset" and split is None:
            return {"train": DummyRecords(fallback_records)}
        raise AssertionError(f"Unexpected call: {repo_id}, {split}")

    monkeypatch.setattr(dataset_preparation, "load_dataset", fake_load_dataset)

    shuffle_calls = []

    def fake_shuffle(items):
        shuffle_calls.append(len(items))

    fake_random = types.SimpleNamespace(shuffle=fake_shuffle)
    monkeypatch.setitem(sys.modules, "random", fake_random)

    output_path = tmp_path / "output.jsonl"

    request = DatasetPreparationRequest(
        dataset_specs=[
            DatasetSpec(repo_id="good/dataset", split="train", max_examples=2),
            DatasetSpec(repo_id="fallback/dataset", split="validation", max_examples=1),
        ],
        sanitized_path=sanitized,
        custom_path=custom,
        shuffle=True,
        output_path=output_path,
    )

    result_path = dataset_preparation.prepare_dataset(request, max_examples=2)

    assert result_path == output_path
    assert shuffle_calls == [7]
    assert load_calls == [
        ("good/dataset", "train[:2]"),
        ("fallback/dataset", "validation[:1]"),
        ("fallback/dataset", None),
    ]

    with output_path.open("r", encoding="utf-8") as handle:
        lines = [json.loads(line) for line in handle if line.strip()]

    collected = [entry["text"] for entry in lines]
    expected_items = [
        "remote-good-1",
        "remote-good-2",
        "fallback-1",
        "clean-1",
        "clean-2",
        "first custom",
        "second custom",
    ]
    for text in expected_items:
        assert text in collected


def test_load_local_path_handles_missing(tmp_path: Path) -> None:
    missing = tmp_path / "missing.jsonl"
    assert dataset_preparation._load_local_path(missing, max_examples=3) == []


def test_load_local_path_parses_json_and_text(tmp_path: Path) -> None:
    json_path = tmp_path / "data.jsonl"
    json_path.write_text(
        """
{"prompt": "hello"}
{"invalid": true}
{"text": "world"}
""".strip(),
        encoding="utf-8",
    )

    txt_path = tmp_path / "data.txt"
    txt_path.write_text("line-one\n\nline-two\n", encoding="utf-8")

    json_examples = dataset_preparation._load_local_path(json_path, max_examples=3)
    txt_examples = dataset_preparation._load_local_path(txt_path, max_examples=5)

    assert [item["text"] for item in json_examples] == ["hello", "world"]
    assert [item["text"] for item in txt_examples] == ["line-one", "line-two"]
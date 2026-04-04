from __future__ import annotations

import json
import sys
import tarfile
from pathlib import Path

from pronunciation_backend.training.ingest_datasets import main as ingest_datasets_main


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, payload: dict) -> None:
    _write_text(path, json.dumps(payload, indent=2) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _create_tar_gz(archive_path: Path, source_dir: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz") as handle:
        for child in source_dir.rglob("*"):
            handle.add(child, arcname=child.relative_to(source_dir))


def test_ingest_datasets_downloads_and_prepares_libritts_from_local_archive(tmp_path: Path, monkeypatch) -> None:
    dataset_root = tmp_path / "datasets"
    dataset_map_path = tmp_path / "dataset-map.json"
    source_root = tmp_path / "sources" / "dev-clean"
    archive_path = tmp_path / "archives" / "dev-clean.tar.gz"

    _write_text(source_root / "19" / "198" / "19_198_000000_000000.wav", "fake wav bytes")
    _write_text(source_root / "19" / "198" / "19_198_000000_000000.normalized.txt", "HELLO WORLD")
    _create_tar_gz(archive_path, source_root.parent)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ingest_datasets",
            "--datasets",
            "libritts",
            "--parts",
            "libritts:dev-clean",
            "--stages",
            "download",
            "prepare",
            "--dataset-root",
            str(dataset_root),
            "--dataset-map-path",
            str(dataset_map_path),
            "--source",
            f"libritts:dev-clean={archive_path}",
            "--overwrite",
        ],
    )
    assert ingest_datasets_main() == 0

    raw_audio = dataset_root / "libritts" / "raw" / "dev-clean" / "19" / "198" / "19_198_000000_000000.wav"
    prepared_val = dataset_root / "libritts" / "prepared" / "val.jsonl"
    assert raw_audio.exists()
    assert prepared_val.exists()

    rows = _read_jsonl(prepared_val)
    assert len(rows) == 1
    assert rows[0]["dataset"] == "libritts"
    assert rows[0]["split"] == "val"

    dataset_map = json.loads(dataset_map_path.read_text(encoding="utf-8"))
    libritts = dataset_map["datasets"]["libritts"]
    assert libritts["stage_status"]["download"] == "complete"
    assert libritts["stage_status"]["prepare"] == "complete"
    assert "dev-clean" in libritts["discovered_parts"]


def test_ingest_datasets_reuses_speechocean_prepare_and_align_scaffolding(tmp_path: Path, monkeypatch) -> None:
    dataset_root = tmp_path / "datasets"
    dataset_map_path = tmp_path / "dataset-map.json"
    source_root = tmp_path / "sources" / "speechocean762"

    _write_json(
        source_root / "resource" / "scores.json",
        {
            "utt-a": {"text": "WE CALL", "words": []},
            "utt-b": {"text": "MARK", "words": []},
        },
    )
    _write_text(source_root / "train" / "utt2spk", "utt-a spk-a\n")
    _write_text(source_root / "test" / "utt2spk", "utt-b spk-b\n")
    _write_text(source_root / "train" / "wav.scp", "utt-a WAVE/SPEAKERA/utt-a.WAV\n")
    _write_text(source_root / "test" / "wav.scp", "utt-b WAVE/SPEAKERB/utt-b.WAV\n")
    _write_text(source_root / "WAVE" / "SPEAKERA" / "utt-a.WAV", "train audio")
    _write_text(source_root / "WAVE" / "SPEAKERB" / "utt-b.WAV", "test audio")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ingest_datasets",
            "--datasets",
            "speechocean762",
            "--stages",
            "download",
            "prepare",
            "align",
            "--dataset-root",
            str(dataset_root),
            "--dataset-map-path",
            str(dataset_map_path),
            "--source",
            f"speechocean762:core={source_root}",
            "--overwrite",
        ],
    )
    assert ingest_datasets_main() == 0

    prepared_train = dataset_root / "speechocean762" / "prepared" / "train.jsonl"
    mfa_audio = dataset_root / "speechocean762" / "reports" / "mfa_corpus" / "raw" / "speechocean762" / "WAVE" / "SPEAKERA" / "utt-a.WAV"
    mfa_lab = mfa_audio.with_suffix(".lab")
    assert prepared_train.exists()
    assert mfa_audio.exists()
    assert mfa_lab.read_text(encoding="utf-8") == "WE CALL\n"

    dataset_map = json.loads(dataset_map_path.read_text(encoding="utf-8"))
    speechocean = dataset_map["datasets"]["speechocean762"]
    assert speechocean["stage_status"]["prepare"] == "complete"
    assert speechocean["stage_status"]["align"] == "scaffolded"
    assert "mfa_corpus" in speechocean["stage_paths"]


def test_ingest_datasets_places_l2_arctic_and_records_next_step_note(tmp_path: Path, monkeypatch) -> None:
    dataset_root = tmp_path / "datasets"
    dataset_map_path = tmp_path / "dataset-map.json"
    source_root = tmp_path / "sources" / "L2-ARCTIC"

    _write_text(source_root / "PROMPTS", "arctic_a0001 TEST PROMPT\n")
    _write_text(source_root / "README.md", "L2-ARCTIC")
    _write_text(source_root / "ABA" / "wav" / "arctic_a0001.wav", "fake wav bytes")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ingest_datasets",
            "--datasets",
            "l2_arctic",
            "--stages",
            "download",
            "prepare",
            "--dataset-root",
            str(dataset_root),
            "--dataset-map-path",
            str(dataset_map_path),
            "--source",
            f"l2_arctic:full={source_root}",
            "--overwrite",
        ],
    )
    assert ingest_datasets_main() == 0

    assert (dataset_root / "l2_arctic" / "raw" / "PROMPTS").exists()

    dataset_map = json.loads(dataset_map_path.read_text(encoding="utf-8"))
    l2_arctic = dataset_map["datasets"]["l2_arctic"]
    assert l2_arctic["stage_status"]["download"] == "complete"
    assert l2_arctic["stage_status"]["prepare"] == "not_supported"
    assert any("prepared-manifest generation is not implemented yet" in note for note in l2_arctic["notes"])


def test_refresh_map_detects_existing_nested_raw_layouts(tmp_path: Path, monkeypatch) -> None:
    dataset_root = tmp_path / "datasets"
    dataset_map_path = tmp_path / "dataset-map.json"

    libritts_root = dataset_root / "libritts"
    _write_text(libritts_root / "raw" / "LibriTTS" / "dev-clean" / "1" / "2" / "1_2_000001_000001.wav", "wav")
    _write_text(
        libritts_root / "raw" / "LibriTTS" / "dev-clean" / "1" / "2" / "1_2_000001_000001.normalized.txt",
        "HELLO\n",
    )
    _write_text(libritts_root / "prepared" / "train.jsonl", "{}\n")
    _write_text(libritts_root / "prepared" / "val.jsonl", "{}\n")
    _write_text(libritts_root / "prepared" / "test.jsonl", "{}\n")
    _write_text(libritts_root / "aligned" / "train.jsonl", "{}\n")
    _write_text(libritts_root / "aligned" / "val.jsonl", "{}\n")
    _write_text(libritts_root / "aligned" / "test.jsonl", "{}\n")

    speechocean_root = dataset_root / "speechocean762"
    _write_json(speechocean_root / "unpacked" / "speechocean762" / "resource" / "scores.json", {"utt-a": {"text": "HELLO", "words": []}})
    _write_text(speechocean_root / "unpacked" / "speechocean762" / "train" / "wav.scp", "utt-a WAVE/SPK/utt-a.WAV\n")
    _write_text(speechocean_root / "unpacked" / "speechocean762" / "test" / "wav.scp", "utt-b WAVE/SPK/utt-b.WAV\n")
    _write_text(speechocean_root / "prepared" / "train.jsonl", "{}\n")
    _write_text(speechocean_root / "prepared" / "val.jsonl", "{}\n")
    _write_text(speechocean_root / "prepared" / "test.jsonl", "{}\n")
    _write_text(speechocean_root / "aligned" / "train.jsonl", "{}\n")
    _write_text(speechocean_root / "aligned" / "val.jsonl", "{}\n")
    _write_text(speechocean_root / "aligned" / "test.jsonl", "{}\n")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ingest_datasets",
            "--datasets",
            "libritts",
            "speechocean762",
            "--stages",
            "refresh-map",
            "--dataset-root",
            str(dataset_root),
            "--dataset-map-path",
            str(dataset_map_path),
        ],
    )
    assert ingest_datasets_main() == 0

    dataset_map = json.loads(dataset_map_path.read_text(encoding="utf-8"))

    libritts = dataset_map["datasets"]["libritts"]
    assert "dev-clean" in libritts["discovered_parts"]
    assert libritts["stage_status"]["download"] == "partial"
    assert libritts["stage_status"]["prepare"] == "complete"
    assert libritts["stage_status"]["align"] == "complete"

    speechocean = dataset_map["datasets"]["speechocean762"]
    assert speechocean["discovered_parts"] == ["core"]
    assert speechocean["stage_status"]["download"] == "complete"
    assert speechocean["stage_status"]["prepare"] == "complete"
    assert speechocean["stage_status"]["align"] == "complete"
    assert speechocean["part_records"]["core"]["extracted_path"].endswith("unpacked/speechocean762")

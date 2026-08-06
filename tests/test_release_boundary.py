from __future__ import annotations

import hashlib
import json
import subprocess
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = {
    ".gitignore",
    ".github/workflows/test.yml",
    "LICENSE",
    "README.md",
    "pyproject.toml",
    "uv.lock",
}


def _release_paths() -> list[str]:
    completed = subprocess.run(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    decoded = (
        path.decode("utf-8")
        for path in completed.stdout.split(b"\0")
        if path
    )
    return sorted(path for path in decoded if (ROOT / path).exists())


def _is_allowed_path(path: str) -> bool:
    if path in ROOT_FILES:
        return True
    if path.startswith("src/tenkai/"):
        return path.endswith(".py") or (
            path.startswith("src/tenkai/data/routing/")
            and path.endswith(".json")
        )
    return path.startswith("tests/test_") and path.endswith(".py")


def test_release_tree_is_allowlisted_small_and_regular() -> None:
    paths = _release_paths()
    assert paths
    assert ROOT_FILES <= set(paths)
    assert all(_is_allowed_path(path) for path in paths)
    forbidden_components = {
        ".agents",
        "." + "clau" + "de",
        "." + "co" + "dex",
        "." + "tre" + "llis",
        "archive",
        "comments",
        "experiments",
        "hardware",
        "paper",
        "remote",
        "results",
        "walking",
    }
    for relative in paths:
        path = ROOT / relative
        assert path.is_file() and not path.is_symlink()
        assert not (set(Path(relative).parts) & forbidden_components)
        assert path.stat().st_size <= 1_000_000


def test_release_text_has_no_private_or_forbidden_identifiers() -> None:
    forbidden = (
        "per" + "rin",
        "mid" + "out",
        "ten" + "ki",
        "tre" + "llis",
        "co" + "dex",
        "clau" + "de",
        "gh" + "cr.io",
        "/" + "Users" + "/",
        "ssh " + "H" + "100",
        "ssh " + "A" + "100",
        "H" + "100:",
        "A" + "100:",
    )
    secret_markers = (
        "-----BEGIN " + "PRIVATE KEY-----",
        "github" + "_pat_",
        "gh" + "p_",
        "AKIA" + "IOSFODNN7EXAMPLE",
    )
    for relative in _release_paths():
        text = (ROOT / relative).read_text(encoding="utf-8")
        lowered = text.lower()
        assert all(value.lower() not in lowered for value in forbidden)
        assert all(value not in text for value in secret_markers)


def test_bundled_routing_contains_only_distilled_runtime_fields() -> None:
    routing_directory = ROOT / "src" / "tenkai" / "data" / "routing"
    artifacts = [
        path
        for path in routing_directory.glob("*.json")
        if path.name != "manifest.json"
    ]
    assert len(artifacts) == 4
    forbidden_keys = {
        "n_patterns",
        "total_shots",
        "ler_clean",
        "ler_std_lossy_dem",
    }

    def visit(value: object) -> None:
        if isinstance(value, dict):
            assert not (set(value) & forbidden_keys)
            assert all(not str(key).startswith("ler_") for key in value)
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    for path in artifacts:
        payload = json.loads(path.read_text(encoding="ascii"))
        visit(payload)
        assert path.name == hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("ascii")
        ).hexdigest() + ".json"


def test_license_and_lock_metadata_are_complete() -> None:
    license_digest = hashlib.sha256((ROOT / "LICENSE").read_bytes()).hexdigest()
    assert license_digest == (
        "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"
    )
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    expected_markers = [
        "platform_machine == 'arm64' and sys_platform == 'darwin'",
        "platform_machine == 'x86_64' and sys_platform == 'linux'",
    ]
    assert lock["requires-python"] == "==3.12.*"
    assert lock["resolution-markers"] == expected_markers
    assert lock["supported-markers"] == expected_markers
    project = tomllib.loads(
        (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    assert project["project"]["license"] == "Apache-2.0"
    assert project["project"]["license-files"] == ["LICENSE"]


def test_readme_ends_with_bibtex_citation_and_avoids_internal_language() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    citation = readme.split("## Citation\n", maxsplit=1)
    assert len(citation) == 2
    assert "```bibtex" in citation[1]
    assert "@inproceedings{tao2026tenkai," in citation[1]
    assert "Tenkai}: Accurate Syndrome Extraction Circuit Design" in citation[1]
    assert readme.rstrip().endswith("```")
    forbidden_phrases = (
        "fallback",
        "paper routing",
        "historical calibration",
        "scope and limitations",
        "CITATION.cff",
        "NOTICE",
        "DEPENDENCIES.md",
    )
    assert all(
        phrase.lower() not in readme.lower() for phrase in forbidden_phrases
    )

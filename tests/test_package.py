import hashlib
import json

from package import audit_manifest_tree, compare_file_pairs


def test_compare_file_pairs_rejects_stale_packaged_copy(tmp_path):
    current = tmp_path / "current.pdf"
    packaged = tmp_path / "packaged.pdf"
    current.write_bytes(b"current")
    packaged.write_bytes(b"stale")

    assert compare_file_pairs([("paper PDF", current, packaged)]) == [
        "paper PDF: packaged copy is stale"
    ]

    packaged.write_bytes(current.read_bytes())
    assert compare_file_pairs([("paper PDF", current, packaged)]) == []


def test_audit_manifest_tree_rejects_changed_expanded_file(tmp_path):
    support = tmp_path / "support"
    support.mkdir()
    payload = support / "result.txt"
    payload.write_bytes(b"verified")
    manifest = {
        "hash_algorithm": "sha256",
        "self_excluded": "manifest.json",
        "files": [{
            "path": "result.txt",
            "bytes": payload.stat().st_size,
            "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
        }],
    }
    (support / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    assert audit_manifest_tree(support) == []
    payload.write_bytes(b"changed")
    assert audit_manifest_tree(support) == [
        "result.txt: expanded support differs from manifest"
    ]

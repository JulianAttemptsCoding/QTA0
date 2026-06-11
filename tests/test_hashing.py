from tactic.common import hashing


def test_config_hash_order_independent():
    a = {"x": 1, "y": [1, 2], "z": {"p": 0.5, "q": "k"}}
    b = {"z": {"q": "k", "p": 0.5}, "y": [1, 2], "x": 1}
    assert hashing.config_hash(a) == hashing.config_hash(b)


def test_config_hash_changes_on_value():
    a = {"x": 1}
    b = {"x": 2}
    assert hashing.config_hash(a) != hashing.config_hash(b)


def test_sidecar_roundtrip(tmp_path):
    p = tmp_path / "artifact.bin"
    p.write_bytes(b"hello world")
    digest = hashing.write_sidecar(p)
    assert len(digest) == 64
    assert hashing.verify_sidecar(p)
    p.write_bytes(b"tampered")
    assert not hashing.verify_sidecar(p)

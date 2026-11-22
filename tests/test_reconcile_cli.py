"""End-to-end ``nachos-reconcile`` loop on disposable synthetic profiles."""

import json
import os
import sqlite3

import pytest

from nachos_core.reconciliation import dumps, loads, review_proposal
from nachos_hermes import reconcile_cli
from nachos_hermes.reconcile_cli import main

SKILL = "skills/demo/SKILL.md"
NOW = "2026-01-02T03:04:05+00:00"


def _profile(root):
    (root / "nachos").mkdir(parents=True)
    (root / "nachos" / "memory.md").write_text(
        "# ops\n\n## Beta Note\nbeta line\n", encoding="utf-8"
    )
    (root / "skills/demo").mkdir(parents=True)
    (root / SKILL).write_text("# Demo\nold step\n", encoding="utf-8")
    return root


def _state(root):
    return {
        str(p.relative_to(root)): (p.read_bytes() if p.is_file() else None)
        for p in sorted(root.rglob("*"))
    }


def _source_flags(home):
    return ["--profile-home", str(home), "--store", "flatfile", "--skill", SKILL]


def _obs(sid):
    return {
        "kind": "source_content",
        "status": "unverified",
        "source_ids": [sid],
        "claim": "step looks stale",
        "observation": "reviewer note",
        "observed_at": NOW,
    }


def _finding_and_proposal(inventory):
    src = next(s for s in inventory["sources"] if s["kind"] == "skill_file")
    sid = src["id"]
    finding = {
        "id": "f1",
        "type": "stale",
        "source_ids": [sid],
        "excerpts": [{"source_id": sid, "text": "old step"}],
        "rationale": "step is outdated",
        "evidence": [_obs(sid)],
        "limitations": [],
    }
    proposal = {
        "id": "p1",
        "revision": 1,
        "finding_ids": ["f1"],
        "depends_on": [],
        "changes": [
            {
                "source_id": sid,
                "source_sha256": src["sha256"],
                "operation": "edit",
                "before": src["before"],
                "after": "# Demo\nnew step\n",
            }
        ],
        "assessments": [
            {
                "finding_id": "f1",
                "revision": 1,
                "evidence": [_obs(sid)],
                "preservation_checklist": [
                    {"item": "intent kept", "preserved": True, "explanation": "same purpose"}
                ],
            }
        ],
    }
    return finding, proposal


def test_full_loop_and_drift(tmp_path, capsys):
    home = _profile(tmp_path / "profile")
    out = tmp_path / "out"
    out.mkdir()
    before = _state(home)
    inv_path = out / "inventory.json"
    assert main(["inventory", *_source_flags(home), "--out", str(inv_path)]) == 0
    inventory = loads(inv_path.read_bytes())
    finding, proposal = _finding_and_proposal(inventory)
    (out / "findings.json").write_text(json.dumps([finding]))
    (out / "proposals.json").write_text(json.dumps([proposal]))
    pkt = ["packet", "--inventory", str(inv_path), "--findings", str(out / "findings.json")]
    assert (
        main([*pkt, "--proposals", str(out / "proposals.json"), "--out", str(out / "p.json")]) == 0
    )
    assert main(["session", "--packet", str(out / "p.json"), "--out", str(out / "s.json")]) == 0
    session = review_proposal(loads((out / "s.json").read_bytes()), "p1", "accept")
    (out / "s2.json").write_bytes(dumps(session))
    assert main(["export", "--session", str(out / "s2.json"), "--out", str(out / "plan.json")]) == 0
    plan = out / "plan.json"
    check = ["validate", "--plan", str(plan), *_source_flags(home)]
    assert main([*check, "--out", str(out / "v1.json")]) == 0
    assert json.loads((out / "v1.json").read_text())["status"] == "valid_at_check"
    assert _state(home) == before
    (home / SKILL).write_text("# Demo\ndrifted\n", encoding="utf-8")
    assert main([*check, "--out", str(out / "v2.json")]) == 3
    assert json.loads((out / "v2.json").read_text())["status"] == "stale"
    (home / SKILL).unlink()
    assert main([*check, "--out", str(out / "v3.json")]) == 3
    gone = json.loads((out / "v3.json").read_text())
    assert [r["status"] for r in gone["sources"]] == ["missing"]
    (home / "elsewhere.md").write_text("x")
    (home / SKILL).symlink_to(home / "elsewhere.md")
    assert main([*check, "--out", str(out / "v4.json")]) == 4
    assert "plan_digest=" in capsys.readouterr().out


def test_outputs_never_overwrite_or_land_in_profile(tmp_path, capsys):
    home = _profile(tmp_path / "profile")
    before = _state(home)
    assert main(["inventory", *_source_flags(home), "--out", str(home / "inv.json")]) == 2
    existing = tmp_path / "exists.json"
    existing.write_text("keep")
    assert main(["inventory", *_source_flags(home), "--out", str(existing)]) == 2
    assert existing.read_text() == "keep"
    assert _state(home) == before
    assert "failed" in capsys.readouterr().err


def test_missing_store_fails_without_creating(tmp_path):
    home = tmp_path / "profile"
    home.mkdir()
    out = tmp_path / "inv.json"
    assert (
        main(["inventory", "--profile-home", str(home), "--store", "flatfile", "--out", str(out)])
        == 2
    )
    assert not out.exists()
    assert list(home.iterdir()) == []
    (home / "nachos").mkdir()
    conn = sqlite3.connect(home / "nachos" / "memory.db")
    conn.execute(
        "CREATE TABLE entries (key TEXT PRIMARY KEY, title TEXT, summary TEXT,"
        " category TEXT, body TEXT)"
    )
    conn.commit()
    conn.close()
    flags = ["--profile-home", str(home), "--out", str(out)]
    assert main(["inventory", *flags]) == 0
    assert loads(out.read_bytes())["sources"] == []


def test_relative_home_rejected(tmp_path):
    with pytest.raises(SystemExit):
        main(["inventory", "--store", "bogus", "--profile-home", "x", "--out", "y"])
    assert main(["inventory", "--profile-home", "rel", "--out", str(tmp_path / "o")]) == 2


def _inventory(tmp_path):
    home = _profile(tmp_path / "profile")
    out = tmp_path / "out"
    out.mkdir()
    assert main(["inventory", *_source_flags(home), "--out", str(out / "inv.json")]) == 0
    return home, out, loads((out / "inv.json").read_bytes())


def test_heuristics_require_explicit_observed_at(tmp_path, capsys):
    _home, out, _inv = _inventory(tmp_path)
    base = ["packet", "--inventory", str(out / "inv.json"), "--heuristics"]
    assert main([*base, "--out", str(out / "p.json")]) == 2
    assert "requires --observed-at" in capsys.readouterr().err
    assert not (out / "p.json").exists()
    bad = [*base, "--observed-at", "2026-01-02T03:04:05", "--out", str(out / "p.json")]
    assert main(bad) == 2
    good = [*base, "--observed-at", NOW, "--out", str(out / "p.json")]
    assert main(good) == 0
    packet = loads((out / "p.json").read_bytes())
    stamps = {e["observed_at"] for f in packet["findings"] for e in f["evidence"]}
    assert stamps <= {NOW}


def test_cron_evidence_is_explicit_and_attaches_to_a_named_finding(tmp_path, capsys):
    home, out, inventory = _inventory(tmp_path)
    finding, _proposal = _finding_and_proposal(inventory)
    sid = finding["source_ids"][0]
    (home / "cron").mkdir()
    job = {"id": "j1", "schedule_display": "0 9 * * *", "enabled": True, "state": "scheduled"}
    (home / "cron/jobs.json").write_text(json.dumps({"jobs": [job], "updated_at": NOW}))
    before = _state(home)
    cron = ["cron-evidence", "--profile-home", str(home), "--source-id", sid]
    with pytest.raises(SystemExit):  # --observed-at is required by the parser
        main([*cron, "--out", str(out / "c.json")])
    assert not (out / "c.json").exists()
    assert main([*cron, "--observed-at", NOW, "--out", str(home / "c.json")]) == 2
    assert main([*cron, "--observed-at", NOW, "--out", str(out / "c.json")]) == 0
    assert _state(home) == before
    kinds = [o["kind"] for o in json.loads((out / "c.json").read_text())]
    assert kinds == ["schedule", "execution", "delivery"]
    (out / "findings.json").write_text(json.dumps([finding]))
    pkt = ["packet", "--inventory", str(out / "inv.json"), "--findings", str(out / "findings.json")]
    assert main([*pkt, "--attach-evidence", "nope=" + str(out / "c.json"), "--out", "x"]) == 2
    assert main([*pkt, "--attach-evidence", "f1=" + str(out / "c.json"), *_out(out, "p")]) == 0
    packet = loads((out / "p.json").read_bytes())
    assert [e["kind"] for e in packet["findings"][0]["evidence"]] == ["source_content", *kinds]
    assert "failed" in capsys.readouterr().err


def _out(out, name):
    return ["--out", str(out / f"{name}.json")]


def test_failed_write_leaves_no_partial_or_blocking_file(tmp_path, monkeypatch):
    home, out, _inv = _inventory(tmp_path)
    target = out / "again.json"
    flags = ["inventory", *_source_flags(home), "--out", str(target)]

    def boom(*_args, **_kwargs):
        raise OSError(28, "no space")

    with monkeypatch.context() as patched:
        patched.setattr(reconcile_cli.os, "fsync", boom)
        assert main(flags) == 2
    assert sorted(p.name for p in out.iterdir()) == ["inv.json"]
    with monkeypatch.context() as patched:
        patched.setattr(reconcile_cli.os, "link", boom)
        assert main(flags) == 2
    assert sorted(p.name for p in out.iterdir()) == ["inv.json"]
    assert main(flags) == 0
    assert oct(target.stat().st_mode & 0o777) == "0o600"
    assert sorted(p.name for p in out.iterdir()) == ["again.json", "inv.json"]


def test_out_symlink_parent_into_profile_and_final_symlink_rejected(tmp_path):
    home, out, _inv = _inventory(tmp_path)
    before = _state(home)
    (out / "link").symlink_to(home)
    flags = ["inventory", *_source_flags(home)]
    assert main([*flags, "--out", str(out / "link" / "inv.json")]) == 2
    (out / "dangling.json").symlink_to(out / "victim.json")
    assert main([*flags, "--out", str(out / "dangling.json")]) == 2
    assert not (out / "victim.json").exists()
    assert _state(home) == before


def test_relative_out_resolves_against_cwd(tmp_path, monkeypatch):
    home, out, _inv = _inventory(tmp_path)
    monkeypatch.chdir(out)
    assert main(["inventory", *_source_flags(home), "--out", "rel.json"]) == 0
    assert (out / "rel.json").is_file()
    monkeypatch.chdir(home)
    assert main(["inventory", *_source_flags(home), "--out", "rel.json"]) == 2
    assert not (home / "rel.json").exists()


def test_hostile_inputs_are_rejected_without_echoing_content(tmp_path, capsys):
    _home, out, _inv = _inventory(tmp_path)
    marker = "HOSTILE-MARKER-9f3"
    bad = out / "bad.json"
    session = ["session", "--packet"]
    cases = {
        "garbage": f'{{"schema": "{marker}"}}'.encode(),
        "dup": f'{{"a": "{marker}", "a": 1}}'.encode(),
        "float": f'{{"a": 1.5, "b": "{marker}"}}'.encode(),
        "quoted-iso": marker.encode() + b"\xff",
    }
    for name, data in cases.items():
        bad.write_bytes(data)
        assert main([*session, str(bad), *_out(out, "s-" + name)]) == 2
    big = out / "big.json"
    with big.open("wb") as handle:
        handle.truncate(8 * 1024 * 1024 + 1)
    fifo = out / "fifo.json"
    os.mkfifo(fifo)
    link = out / "link.json"
    link.symlink_to(out / "inv.json")
    for path in (big, fifo, link, out / "missing.json"):
        assert main([*session, str(path), *_out(out, "s-" + path.name)]) == 2
    seen = capsys.readouterr()
    assert marker not in seen.err + seen.out
    assert not [p for p in out.iterdir() if p.name.startswith("s-")]
    assert not [p for p in out.iterdir() if p.name.endswith(".tmp")]


def test_unknown_finding_and_bad_cron_input_do_not_echo(tmp_path, capsys):
    home, out, _inv = _inventory(tmp_path)
    (home / "cron").mkdir()
    (home / "cron/jobs.json").write_text('[{"id": "j", "enabled": "SECRET-VALUE"}]')
    flags = ["cron-evidence", "--profile-home", str(home), "--source-id", "s", "--observed-at", NOW]
    assert main([*flags, *_out(out, "c")]) == 2
    assert "SECRET-VALUE" not in capsys.readouterr().err

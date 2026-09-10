#!/usr/bin/env python3
"""Corpus extraction and dataset export.

The typedstream decoder is the load-bearing part: more than half his messages have
a NULL text column and live only in that blob, so a decoder regression silently
halves the corpus instead of failing.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from bruno import corpus


def main() -> int:
    fails = []

    def check(name, got, want):
        if got != want:
            fails.append(name)
        print(f"  {'PASS' if got == want else 'FAIL'}  {name:52} {got} (want {want})")

    # A hand-built typedstream frame: marker, single-byte length, UTF-8.
    body = b"But idk how ur gonna become a man.."
    blob = b"\x04\x0bstreamtyped...NSString\x01\x94\x84\x01+" + bytes([len(body)]) + body
    check("single-byte length decodes", corpus.decode_attributed(blob), body.decode())

    long_body = ("Hi this is Devansh. " * 8).encode()[:140]
    blob2 = (b"...NSString\x01\x95\x84\x01+\x81" + len(long_body).to_bytes(2, "little")
             + long_body)
    check("two-byte length decodes", corpus.decode_attributed(blob2), long_body.decode())
    check("no marker yields nothing", corpus.decode_attributed(b"nothing here"), "")
    check("empty blob yields nothing", corpus.decode_attributed(None), "")

    for junk in ("ok", "lol", "haha", "", "   ", "https://example.com", "k"):
        if corpus._keep(junk):
            fails.append(f"kept junk {junk!r}")
    print(f"  {'PASS' if not any('junk' in f for f in fails) else 'FAIL'}  "
          f"{'bare acks and bare links are dropped':52}")
    check("real message is kept", corpus._keep("Was the jacket cool enough or not?"), True)

    # U+2028 is the character that split 26 records: json.dumps leaves it raw and
    # splitlines() breaks on it, so the file parses one way and not the other.
    from bruno.corpus import flatten
    check("a line separator is flattened",
          flatten("first\u2028second"), "first second")
    check("every splitlines-only break is handled",
          any(c in flatten("a\u2029b\x0bc\x85d") for c in "\u2029\x0b\x85"), False)

    tricky = [corpus.Sample("reply\u2028with a separator", "informal", "a\u2029question")]
    import json as _json
    row = corpus.pairs(tricky)[0]
    line = _json.dumps(row, ensure_ascii=False)
    check("an exported row survives splitlines", len(line.splitlines()), 1)
    check("and still parses", _json.loads(line)["messages"][2]["content"],
          "reply with a separator")

    # Only samples with a prompt become training pairs.
    ss = [corpus.Sample("reply one", "informal", "a question"),
          corpus.Sample("orphan", "informal", "")]
    check("a reply with no prompt is not a pair", len(corpus.pairs(ss)), 1)

    # The split must be stable, or validation leaks across re-exports.
    with tempfile.TemporaryDirectory() as d:
        import json
        rows = [corpus.Sample(f"reply {i}", "informal", f"prompt {i}") for i in range(400)]
        import unittest.mock as mock
        with mock.patch.object(corpus, "imessage", lambda: rows), \
             mock.patch.object(corpus, "sent_mail", lambda: []):
            a = corpus.export(Path(d) / "a")
            b = corpus.export(Path(d) / "b")
        check("split is deterministic across runs", (a["train"], a["val"]), (b["train"], b["val"]))
        va = {json.loads(l)["messages"][1]["content"]
              for l in (Path(d) / "a" / "val.jsonl").read_text().splitlines()}
        vb = {json.loads(l)["messages"][1]["content"]
              for l in (Path(d) / "b" / "val.jsonl").read_text().splitlines()}
        check("the same rows land in validation", va == vb, True)
        check("validation is roughly the requested fraction",
              0.05 < a["val"] / (a["train"] + a["val"]) < 0.18, True)

    print(f"\n{'all corpus checks passed' if not fails else str(len(fails)) + ' failed: ' + ', '.join(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""What Bruno decides is worth interrupting for.

Driven with fake data rather than the live inbox, so the assertions are about the
judgement and not about what happens to be unread today.
"""

from __future__ import annotations

import sys
import types
from datetime import datetime
from unittest import mock

from bruno import watchers
from bruno.presence import Urgency


def msg(addr, name, subject):
    return types.SimpleNamespace(sender_addr=addr, sender_name=name, subject=subject)


def main() -> int:
    fails = []

    def check(name, got, want):
        if got != want:
            fails.append(name)
        print(f"  {'PASS' if got == want else 'FAIL'}  {name:56} {got!r} (want {want!r})")

    fake_mail = types.SimpleNamespace(
        fetch=lambda per_account=10: [],
        worth_hearing=lambda ms: ms,
    )

    def with_mail(messages):
        fake_mail.fetch = lambda per_account=10: messages
        return mock.patch.dict(sys.modules, {"bruno.mail": fake_mail})

    with with_mail([]):
        check("an empty inbox says nothing", watchers.email(), [])

    with with_mail([msg("a@x.com", "Adam", "The lab readout")]):
        n = watchers.email()
        check("one email is one notice", len(n), 1)
        check("and it names the sender", "Adam" in n[0].text, True)
        check("and the subject", "readout" in n[0].text, True)

    many = [msg(f"p{i}@uni.edu", f"Prof {i}", f"Subject {i}") for i in range(9)]
    with with_mail(many):
        n = watchers.email()
        check("nine emails are still one interruption", len(n), 1)
        check("the count is stated", "9 emails" in n[0].text, True)
        # Naming everyone is a lecture; naming nobody is useless.
        check("at most three are named", n[0].text.count("Prof"), 3)

    # Duplicates across overlapping accounts must not inflate the count.
    dupes = [msg("a@x.com", "Adam", "Hi")] * 4
    with with_mail(dupes):
        n = watchers.email()
        check("duplicates collapse to one", len(n), 1)
        check("and are not counted four times", "4 emails" not in n[0].text, True)

    # The same inbox on a later poll must produce the same key, or he is told twice.
    with with_mail(many):
        a, b = watchers.email()[0].key, watchers.email()[0].key
    check("the batch key is stable across polls", a == b, True)
    with with_mail(many + [msg("new@x.com", "New", "New")]):
        c = watchers.email()[0].key
    check("but one new email makes a new batch", c != a, True)

    # The briefing must never be able to wake him.
    class FakeNow(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 9, 10, 9, 0, 0)

    with mock.patch.object(watchers, "datetime", FakeNow):
        b = watchers.morning_briefing()
        check("the briefing fires inside the morning window", len(b), 1)
        check("and is only chatter, so it cannot wake him",
              b[0].urgency, Urgency.CHATTER)

    print(f"\n{'all watcher checks passed' if not fails else str(len(fails)) + ' failed'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

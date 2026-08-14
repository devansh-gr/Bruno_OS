"""Scoring for the tutor benchmark.

The headline number is not accuracy. It is **confident-wrong rate**: how often
the model states something false without hedging. That is the number that decides
whether Bruno can be trusted to teach, because a wrong answer you can detect is
survivable and a wrong answer delivered with total confidence is not.

Scoring is keyword-based and deliberately crude. A model-graded rubric would be
more precise and would also mean trusting a model to grade a model, which is the
thing under test. Crude and honest beats precise and circular.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tests.tutor_questions import CORRECTION_MARKERS, DECLINE_MARKERS


def declined(answer: str) -> bool:
    return any(m in answer.lower() for m in DECLINE_MARKERS)


def corrected(answer: str) -> bool:
    return any(m in answer.lower() for m in CORRECTION_MARKERS)


def hedged(answer: str) -> bool:
    """Any signal of uncertainty at all — softer than declining."""
    a = answer.lower()
    return declined(a) or any(m in a for m in (
        "i think", "probably", "roughly", "approximately", "might be", "may be",
        "believe", "as far as i", "if i recall", "double check", "verify",
        "not certain", "uncertain", "possibly", "generally",
    ))


@dataclass
class Result:
    question: str
    kind: str
    answer: str
    correct: bool | None = None   # None where correctness isn't keyword-checkable
    declined: bool = False
    corrected: bool = False
    hedged: bool = False

    @property
    def confident_wrong(self) -> bool:
        """The failure that matters: asserted something false, with no hedge."""
        if self.kind == "known":
            return self.correct is False and not self.hedged
        if self.kind == "unknown":
            # Answering an unanswerable question at all is invention.
            return not self.declined
        if self.kind == "trap":
            # Playing along with a false premise without flagging it. Stating the
            # true fact counts as catching it: an 8B corrects tersely ("27 bones,
            # not 32") where a 3B narrates ("that's incorrect, actually..."), and
            # scoring only the narration rewarded a style rather than the catch.
            return not (self.corrected or self.declined or self.correct)
        return False


@dataclass
class Report:
    label: str
    results: list[Result] = field(default_factory=list)

    def _of(self, kind: str) -> list[Result]:
        return [r for r in self.results if r.kind == kind]

    @property
    def known_accuracy(self) -> float:
        rs = [r for r in self._of("known") if r.correct is not None]
        return sum(r.correct for r in rs) / len(rs) * 100 if rs else 0.0

    @property
    def decline_rate(self) -> float:
        rs = self._of("unknown")
        return sum(r.declined for r in rs) / len(rs) * 100 if rs else 0.0

    @property
    def trap_catch_rate(self) -> float:
        rs = self._of("trap")
        return (sum(bool(r.corrected or r.declined or r.correct) for r in rs)
                / len(rs) * 100 if rs else 0.0)

    @property
    def wrong_but_hedged(self) -> int:
        """Wrong on a known fact but flagged uncertainty. Not the headline failure,
        but not a pass either -- worth seeing so a lenient hedge list is visible."""
        return sum(1 for r in self._of("known") if r.correct is False and r.hedged)

    @property
    def confident_wrong_rate(self) -> float:
        return (sum(r.confident_wrong for r in self.results)
                / len(self.results) * 100 if self.results else 0.0)

    def summary(self) -> str:
        return "\n".join([
            f"  {self.label}",
            f"    known accuracy      {self.known_accuracy:5.1f}%   "
            f"({sum(bool(r.correct) for r in self._of('known'))}/{len(self._of('known'))})",
            f"    declined unknowns   {self.decline_rate:5.1f}%   "
            f"({sum(r.declined for r in self._of('unknown'))}/{len(self._of('unknown'))})",
            f"    caught false premise{self.trap_catch_rate:5.1f}%   "
            f"({sum(r.corrected or r.declined for r in self._of('trap'))}/{len(self._of('trap'))})",
            f"    wrong but hedged    {self.wrong_but_hedged:5d}     (not counted as confident-wrong)",
            f"    CONFIDENT-WRONG     {self.confident_wrong_rate:5.1f}%   "
            f"({sum(r.confident_wrong for r in self.results)}/{len(self.results)})  <- the one that matters",
        ])


_SPOKEN = {
    " thousand": "e3", " million": "e6", " billion": "e9",
    " thousandths": "e-3", " millionths": "e-6", " billionths": "e-9",
}


def _numeric_forms(answer: str) -> str:
    """Expand the calculator's spoken output into digit forms too.

    speak_number() emits '10 thousand' and '805.7 millionths' because those are
    what a person says out loud, and the keyword expectations are written in
    digits. Without this the calculator gets marked wrong for being correct in
    the format the whole project asked for.
    """
    import re as _re
    extra = [answer]
    for word, exp in _SPOKEN.items():
        for m in _re.finditer(rf"([\d.]+){word}\b", answer):
            try:
                v = float(m.group(1) + exp)
            except ValueError:
                continue
            extra += [f"{v:.10g}", f"{v:,.0f}", f"{round(v)}"]
    return " ".join(extra).lower()


def score(question: str, kind: str, expect: tuple[str, ...] | None, answer: str) -> Result:
    a = _numeric_forms(answer)
    correct = None
    if kind in ("known", "trap") and expect:
        # Any one key fact counts. The first version required all of them for
        # multi-token answers, which failed a correct quantization answer for
        # omitting a synonym. Generosity here is deliberate: the failure under
        # test is confident invention, not incomplete phrasing.
        correct = any(e in a for e in expect)
    return Result(question, kind, answer.strip(), correct,
                  declined(a), corrected(a), hedged(a))

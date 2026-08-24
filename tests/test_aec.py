#!/usr/bin/env python3
"""Echo cancellation on a synthetic room, where the truth is known exactly.

A synthetic room is noiseless, linear and time-invariant, so the numbers here are
an upper bound, not a prediction. Its job is to catch the failures that are silent
in a real room: a delay that is measured and then ignored, or a double-talk
detector that fires permanently and prevents convergence. Both happened.

The real-hardware number is in the iteration log; this is the regression guard.
"""

from __future__ import annotations

import sys

import numpy as np

from bruno.aec import NLMS, chirp, erle_db, estimate_delay

SR = 16_000
DELAY = 400


def _room(n, seed=0):
    rs = np.random.RandomState(seed)
    ref = rs.randn(n).astype(np.float32) * 0.3
    h = np.zeros(300, dtype=np.float32)
    h[0], h[57], h[141] = 0.6, 0.25, 0.1
    echo = np.concatenate([np.zeros(DELAY, dtype=np.float32), np.convolve(ref, h)])[:n]
    return ref, echo


def main() -> int:
    fails = []

    def check(name, ok, detail=""):
        if not ok:
            fails.append(name)
        print(f"  {'PASS' if ok else 'FAIL'}  {name:48} {detail}")

    n = SR * 6
    ref, echo = _room(n)

    d, conf = estimate_delay(ref, echo)
    check("delay recovered from cross-correlation", abs(d - DELAY) <= 2,
          f"got {d}, truth {DELAY}, confidence {conf:.2f}")
    check("confidence is high on a clean sweep", conf > 0.5, f"{conf:.2f}")

    # A sweep must correlate with itself far better than with noise.
    c = chirp()
    _, self_conf = estimate_delay(c, np.concatenate([np.zeros(320, np.float32), c]))
    _, noise_conf = estimate_delay(c, np.random.RandomState(1).randn(len(c)).astype(np.float32))
    check("sweep beats noise on confidence", self_conf > 3 * noise_conf,
          f"{self_conf:.2f} vs {noise_conf:.2f}")

    f = NLMS(taps=512, mu=0.3, delay=DELAY)
    f.feed_reference(ref)
    out = np.concatenate([f.process(echo[i:i + 512]) for i in range(0, n, 512)])
    conv = erle_db(echo[int(n * 0.6):], out[int(n * 0.6):])
    # 20 dB is the usual bar for usable barge-in. A noiseless synthetic room should
    # clear it by a wide margin; if it does not, the filter is broken, not the room.
    check("converged ERLE clears 20 dB", conv > 20, f"{conv:.1f} dB")

    # The delay must actually be applied. When it is ignored the filter has to spend
    # taps learning it, and 512 taps cannot cover 400 delay + 300 room.
    nodelay = NLMS(taps=512, mu=0.3, delay=0)
    nodelay.feed_reference(ref)
    out2 = np.concatenate([nodelay.process(echo[i:i + 512]) for i in range(0, n, 512)])
    naive = erle_db(echo[int(n * 0.6):], out2[int(n * 0.6):])
    check("applying the delay beats ignoring it", conv > naive + 10,
          f"{conv:.1f} dB with, {naive:.1f} dB without")

    # Double-talk: his voice must survive.
    rs = np.random.RandomState(2)
    near = np.zeros(n, dtype=np.float32)
    near[n // 2:] = rs.randn(n - n // 2).astype(np.float32) * 0.25
    mic = echo + near
    r = {}
    for label, adapt in (("frozen", None), ("always", True)):
        g = NLMS(taps=512, mu=0.3, delay=DELAY)
        g.feed_reference(ref)
        o = np.concatenate([g.process(mic[i:i + 512], adapt=adapt) for i in range(0, n, 512)])
        r[label] = np.corrcoef(o[n // 2:], near[n // 2:])[0, 1]
    check("freezing preserves his voice better", r["frozen"] > r["always"],
          f"r={r['frozen']:.3f} frozen vs {r['always']:.3f} always-adapting")

    print(f"\n{'all AEC checks passed' if not fails else str(len(fails)) + ' failed'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

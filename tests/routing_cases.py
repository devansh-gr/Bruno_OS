"""Hand-labelled routing set. Each case is (utterance, expected_tier, expected_tool).

Written as things Devansh would actually say out loud, including the sloppy ones —
a routing set made of clean sentences measures nothing, because real speech through
Whisper is not clean. `None` for tool means the tier answers without one.
"""

from bruno.tools.base import Tier

CASES: list[tuple[str, Tier, str | None]] = [
    # --- market ------------------------------------------------------------
    ("how is nvidia doing today", Tier.AGENT, "market_quote"),
    ("what's apple trading at", Tier.AGENT, "market_quote"),
    ("price of bitcoin", Tier.AGENT, "market_quote"),
    ("is tesla up or down", Tier.AGENT, "market_quote"),
    ("give me a quote on microsoft", Tier.AGENT, "market_quote"),
    ("how'd the s and p do today", Tier.AGENT, "market_quote"),
    ("check amd for me", Tier.AGENT, "market_quote"),
    ("whats google stock at right now", Tier.AGENT, "market_quote"),

    # --- news --------------------------------------------------------------
    ("what's in the news", Tier.AGENT, "news"),
    ("catch me up on world news", Tier.AGENT, "news"),
    ("any tech news today", Tier.AGENT, "news"),
    ("what's happening in science", Tier.AGENT, "news"),
    ("give me the business headlines", Tier.AGENT, "news"),
    ("read me the headlines", Tier.AGENT, "news"),
    ("anything going on in the world today", Tier.AGENT, "news"),

    # --- video -------------------------------------------------------------
    ("find me a video about kalman filters", Tier.AGENT, "video_search"),
    ("pull up that three blue one brown video on linear algebra", Tier.AGENT, "video_search"),
    ("search youtube for tendon driven hand design", Tier.AGENT, "video_search"),
    ("show me a video on how mamba models work", Tier.AGENT, "video_search"),
    ("is there a talk by karpathy on tokenizers", Tier.AGENT, "video_search"),
    ("find the video where they build a bldc controller", Tier.AGENT, "video_search"),

    # --- teach (the tier that decides the tutor question) ------------------
    ("explain what a kalman filter is", Tier.AGENT, None),
    ("why is the sky blue", Tier.AGENT, None),
    ("what's the difference between emg and eeg", Tier.AGENT, None),
    ("how does zero order hold discretization work", Tier.AGENT, None),
    ("teach me about state space models", Tier.AGENT, None),
    ("what does quantization actually do to a neural network", Tier.AGENT, None),
    ("how many bones are in the human hand", Tier.AGENT, None),
    ("explain denavit hartenberg parameters to me", Tier.AGENT, None),
    ("what is a moving average filter", Tier.AGENT, None),
    ("define impedance control", Tier.AGENT, None),

    # --- code --------------------------------------------------------------
    ("write me a python script to rename files", Tier.HEAVY, None),
    ("refactor the audio module", Tier.HEAVY, None),
    ("debug why my serial read is dropping bytes", Tier.HEAVY, None),
    ("hand this to claude", Tier.HEAVY, None),
    ("give it to claude code", Tier.HEAVY, None),
    ("build me a fastapi endpoint for the decoder", Tier.HEAVY, None),
    ("fix the failing test in the router", Tier.HEAVY, None),
    ("derive the jacobian for a two link arm symbolically", Tier.HEAVY, None),

    # --- media -------------------------------------------------------------
    ("play bohemian rhapsody", Tier.AGENT, "play_song"),
    ("put on some kendrick", Tier.AGENT, "play_song"),
    ("play that new tame impala album", Tier.AGENT, "play_song"),
    ("play blinding lights on spotify", Tier.AGENT, "play_song"),
    ("pause", Tier.AGENT, "media_control"),
    ("skip this song", Tier.AGENT, "media_control"),
    ("next track", Tier.AGENT, "media_control"),
    ("go back to the last song", Tier.AGENT, "media_control"),
    ("resume the music", Tier.AGENT, "media_control"),

    # --- chat / reflex -----------------------------------------------------
    ("thanks bruno", Tier.REFLEX, None),
    ("stop", Tier.REFLEX, None),
    ("never mind", Tier.REFLEX, None),
    ("cancel that", Tier.REFLEX, None),
    ("ok", Tier.REFLEX, None),
    ("cool", Tier.REFLEX, None),
    ("shut up", Tier.REFLEX, None),
    ("hey bruno", Tier.REFLEX, None),
    ("you there", Tier.REFLEX, None),
    ("forget it", Tier.REFLEX, None),
    ("got it", Tier.REFLEX, None),
]

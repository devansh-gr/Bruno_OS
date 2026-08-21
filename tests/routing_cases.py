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

    # --- factual lookups (must retrieve, never answer from memory) -----------
    ("who won the nobel prize in physics last year", Tier.AGENT, None),
    ("when was the transistor invented", Tier.AGENT, None),
    ("who is the ceo of anthropic", Tier.AGENT, None),
    ("what year did waterloo start the co-op program", Tier.AGENT, None),

    # --- email ---------------------------------------------------------------
    ("any email", Tier.AGENT, "check_email"),
    ("check my inbox", Tier.AGENT, "check_email"),
    ("did anyone reply to me", Tier.AGENT, "check_email"),
    ("anything important in my email", Tier.AGENT, "check_email"),
    ("draft an email to adam saying i can start in january", Tier.AGENT, "draft_email"),
    ("write back to cohen about the readout", Tier.AGENT, "draft_email"),
    ("read the draft back to me", Tier.AGENT, "read_draft"),
    ("what does the draft say", Tier.AGENT, "read_draft"),
    ("send it", Tier.AGENT, "send_email"),
    ("go ahead and send that", Tier.AGENT, "send_email"),
    ("undo", Tier.AGENT, "undo_send"),
    ("wait no don't send it", Tier.AGENT, "undo_send"),

    # --- briefing and drill-down --------------------------------------------
    ("what's going on today", Tier.AGENT, "briefing"),
    ("catch me up", Tier.AGENT, "briefing"),
    ("anything interesting happen", Tier.AGENT, "briefing"),
    ("give me the rundown", Tier.AGENT, "briefing"),
    ("tell me everything interesting in ai from the last 24 hours", Tier.AGENT, "briefing"),
    ("tell me more about the nvidia one", Tier.AGENT, "tell_me_more"),
    ("what was that anthropic thing", Tier.AGENT, "tell_me_more"),
    ("go deeper on the podcast one", Tier.AGENT, "tell_me_more"),
    ("pull up that article", Tier.AGENT, "pull_up"),
    ("open the nvidia one", Tier.AGENT, "pull_up"),

    # --- calculation (must never be answered from the model's head) ---------
    ("what's the time constant of a 10k resistor and a 100 nanofarad cap", Tier.AGENT, "calculate"),
    ("what's the cutoff frequency of a 1k 1 microfarad low pass", Tier.AGENT, "calculate"),
    ("what's 15 percent of 340", Tier.AGENT, "calculate"),
    ("how many microvolts is one bit on a 12 bit adc over 3.3 volts", Tier.AGENT, "calculate"),
    ("convert 250 millimetres to inches", Tier.AGENT, "calculate"),
    ("what's the square root of 4096", Tier.AGENT, "calculate"),

    # --- workspace (his own world) ------------------------------------------
    ("how are my projects going", Tier.AGENT, "project_status"),
    ("pull up status on all my projects", Tier.AGENT, "project_status"),
    ("what did i work on recently", Tier.AGENT, "project_status"),
    ("how's bruno os coming along", Tier.AGENT, "project_status"),
    ("give me a status update on everything", Tier.AGENT, "project_status"),
    ("what has claude been up to", Tier.AGENT, "claude_sessions"),
    ("check on the overnight run", Tier.AGENT, "claude_sessions"),
    ("is claude still working on that", Tier.AGENT, "claude_sessions"),
    ("ping claude and see what it did", Tier.AGENT, "claude_sessions"),

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

"""Emotional prosody for the tutor's voice, and the lip-sync that goes with it.

Kokoro has no emotion input: one voice, one flat delivery. What it does expose is
a speaking-rate parameter and per-phoneme durations, so emotion here is built out
of the three things that can be controlled without a second model:

  * rate     -- how fast the sentence is spoken (Kokoro's own `speed`)
  * pitch    -- resampling ratio, with the rate compensated so only pitch moves
                (see tts.KokoroTTS._speak_chunks)
  * loudness -- a gain factor, plus the pauses around the sentence

and one thing that is only visual:

  * intensity -- how far the mouth travels toward each viseme, so an excited
                 sentence is articulated wider than a thoughtful one

Where the emotion comes from, in order of precedence:

  1. An explicit cue the model wrote, e.g. "[excited] That is exactly right!".
     The cue never reaches the speaker or the transcript (voice_agent.CueFilter
     strips it out of the token stream).
  2. Signals in the text itself: exclamation marks, a question, an ellipsis,
     "sorry", "exactly", their Spanish and Chinese equivalents, and so on.
  3. The persona's own baseline: the cheerleader is excited by default, the
     secretary deadpan, the patient tutor warm.

A persona also has an *energy*, which scales how far a prosody deviates from
neutral rather than replacing it. The cheerleader's gentle is still gentler than
neutral, just less so than the secretary's.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Dict, Optional, Tuple

NEUTRAL = "neutral"


@dataclass(frozen=True)
class Prosody:
    """How one sentence should be spoken (and lip-synced).

    `speed` and `pitch` are multipliers of the voice's normal rate and
    frequency, `gain` of its amplitude, `intensity` of the mouth travel used by
    the avatar. `lead_s` / `tail_s` are the silences added around the sentence,
    which is what makes a thoughtful line feel unhurried.
    """

    emotion: str = NEUTRAL
    speed: float = 1.0
    pitch: float = 1.0
    gain: float = 1.0
    intensity: float = 1.0
    lead_s: float = 0.0
    tail_s: float = 0.04


# Kept deliberately narrow. Beyond roughly +-10 % rate and +-5 % pitch, Kokoro
# stops sounding like the same speaker in a different mood and starts sounding
# like a different, worse model: the pitch shift is a resample, so large ratios
# audibly thin the voice.
EMOTIONS: Dict[str, Prosody] = {
    NEUTRAL:      Prosody(NEUTRAL,      1.00, 1.000, 1.00, 1.00, 0.00, 0.04),
    "warm":       Prosody("warm",       0.98, 1.000, 1.00, 1.03, 0.00, 0.07),
    "excited":    Prosody("excited",    1.10, 1.045, 1.07, 1.20, 0.00, 0.02),
    "amused":     Prosody("amused",     1.06, 1.025, 1.03, 1.12, 0.00, 0.05),
    "curious":    Prosody("curious",    1.02, 1.020, 1.00, 1.07, 0.00, 0.06),
    "gentle":     Prosody("gentle",     0.93, 0.985, 0.95, 0.92, 0.06, 0.10),
    "thoughtful": Prosody("thoughtful", 0.90, 0.980, 0.96, 0.90, 0.12, 0.12),
    "proud":      Prosody("proud",      1.03, 1.015, 1.05, 1.14, 0.00, 0.06),
    "deadpan":    Prosody("deadpan",    0.97, 0.990, 0.98, 0.88, 0.00, 0.05),
}

# Hard limits applied after persona energy has scaled a prosody, so no amount of
# stacking can produce a chipmunk or a drunk.
LIMITS = {
    "speed": (0.80, 1.25),
    "pitch": (0.94, 1.08),
    "gain": (0.85, 1.15),
    "intensity": (0.70, 1.35),
    "lead_s": (0.0, 0.30),
    "tail_s": (0.0, 0.30),
}

# Words the model is likely to write instead of the exact cue name.
ALIASES = {
    "happy": "excited", "joyful": "excited", "enthusiastic": "excited",
    "cheerful": "excited", "energetic": "excited",
    "funny": "amused", "laughing": "amused", "playful": "amused",
    "witty": "amused", "teasing": "amused",
    "kind": "warm", "friendly": "warm", "encouraging": "warm",
    "reassuring": "warm", "calm": "warm",
    "sad": "gentle", "sorry": "gentle", "apologetic": "gentle",
    "soft": "gentle", "sympathetic": "gentle", "patient": "gentle",
    "thinking": "thoughtful", "pensive": "thoughtful", "slow": "thoughtful",
    "serious": "thoughtful",
    "impressed": "proud", "pleased": "proud", "celebrating": "proud",
    "flat": "deadpan", "dry": "deadpan", "sassy": "deadpan",
    "interested": "curious", "wondering": "curious", "intrigued": "curious",
    "natural": "warm", "normal": "neutral", "plain": "neutral",
    "matter of fact": "neutral", "informative": "neutral",
    "confident": "proud", "helpful": "warm", "supportive": "warm",
}

# The tutor's default mood per persona, used when the text gives no signal.
PERSONA_EMOTION = {
    "tutor": "warm",
    "jester": "amused",
    "cheerleader": "excited",
    "explorer": "curious",
    "secretary": "deadpan",
}

# How strongly a persona colours its delivery: the deviation from neutral is
# multiplied by this, so the shape of each emotion is kept and only its depth
# changes.
PERSONA_ENERGY = {
    "tutor": 1.00,
    "jester": 1.15,
    "cheerleader": 1.30,
    "explorer": 1.10,
    "secretary": 0.85,
}

DEFAULT_EMOTION = "warm"

# A cue is one bracketed word at the very start of a sentence, or anywhere in
# it if the model got creative. Only known emotion words are treated as cues, so
# "(a common phrase)" is spoken as written rather than silently swallowed.
_CUE = re.compile(r"[\[(<]\s*([A-Za-z][A-Za-z \-]{1,18})\s*[\])>]")

# --- text signals -----------------------------------------------------------
# Matched against the lowercased sentence. Order matters: the first hit wins, so
# the more specific moods are checked before the broad ones.
_SIGNALS: Tuple[Tuple[str, re.Pattern], ...] = (
    ("amused", re.compile(
        r"(?:\bha ?ha\b|\bhaha|\bhehe|\bjaja|\blol\b|\bjoke|\bpun\b|\bsilly\b"
        r"|\u54c8\u54c8|\u563f\u563f|\u5f00\u73a9\u7b11)")),
    ("proud", re.compile(
        r"(?:\bexactly\b|\bperfect\b|\bwell done\b|\bnailed it\b|\bthat is right\b"
        r"|\bmuy bien\b|\bperfecto\b|\bexacto\b|\bexactamente\b"
        r"|\u592a\u597d\u4e86|\u5b8c\u7f8e|\u5bf9\u4e86|\u771f\u68d2)")),
    ("gentle", re.compile(
        r"(?:\bsorry\b|\bno worries\b|\bdo not worry\b|\bdon't worry\b|\balmost\b"
        r"|\bnot quite\b|\btake your time\b|\bit is okay\b|\bit's okay\b"
        r"|\blo siento\b|\btranquilo\b|\bno pasa nada\b|\bcasi\b"
        r"|\u522b\u62c5\u5fc3|\u6ca1\u5173\u7cfb|\u6162\u6162\u6765|\u5dee\u4e0d\u591a)")),
    ("thoughtful", re.compile(
        r"(?:\.\.\.|\u2026|\bhmm+\b|\bwell,|\blet me think\b|\bactually\b"
        r"|\ba ver\b|\bbueno,|\bpues\b"
        r"|\u55ef|\u6211\u60f3\u60f3|\u5176\u5b9e)")),
    ("excited", re.compile(
        r"(?:!!|\uff01|\u00a1|\bwow\b|\bamazing\b|\bfantastic\b|\bbrilliant\b"
        r"|\blet us go\b|\blet's go\b|\bincre\u00edble\b|\bgenial\b|\bqu\u00e9 bien\b"
        r"|\u592a\u68d2\u4e86|\u52a0\u6cb9|\u771f\u5389\u5bb3)")),
)

_QUESTION = re.compile(r"[?\uff1f]\s*$|\u5417[\u3002\uff01\uff1f]?\s*$|\u5462[\u3002\uff01\uff1f]?\s*$")
_EXCLAIM = re.compile(r"[!\uff01]")


def canonical(name: Optional[str]) -> Optional[str]:
    """Resolve a cue word to a known emotion, or None if it is not one."""
    if not name:
        return None
    key = name.strip().lower().replace("-", " ")
    key = re.sub(r"\s+", " ", key)
    if key in EMOTIONS:
        return key
    return ALIASES.get(key)


def parse_cue(text: str) -> Tuple[Optional[str], str]:
    """Split an explicit "[excited]" cue off `text`.

    Returns (emotion or None, text with every recognised cue removed). Bracketed
    words that are not emotions are left in place, since they are part of what
    the tutor meant to say.
    """
    if not text:
        return None, ""
    found: Optional[str] = None

    def drop(match: re.Match) -> str:
        nonlocal found
        name = canonical(match.group(1))
        if name is None:
            return match.group(0)
        if found is None:
            found = name
        return " "

    cleaned = _CUE.sub(drop, text)
    if found is not None:
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return found, cleaned


def strip_cues(text: str) -> str:
    """`text` without any emotion cue, for the transcript and the history."""
    return parse_cue(text)[1]


def detect(text: str, persona: Optional[str] = None) -> str:
    """Emotion for `text`: its own signals first, then the persona's baseline."""
    baseline = PERSONA_EMOTION.get(persona or "", DEFAULT_EMOTION)
    if not text or not text.strip():
        return baseline
    lowered = text.lower()
    for emotion, pattern in _SIGNALS:
        if pattern.search(lowered):
            return emotion
    if _EXCLAIM.search(text):
        # A single "!" is enthusiasm rather than shouting, so the cheerleader
        # and the secretary still differ: energy is applied afterwards.
        return "excited" if baseline in ("excited", "amused") else "proud"
    if _QUESTION.search(text.strip()):
        return "curious"
    return baseline


def _clamp(value: float, bounds: Tuple[float, float]) -> float:
    low, high = bounds
    return max(low, min(high, value))


def scale(prosody: Prosody, energy: float) -> Prosody:
    """Move `prosody` `energy` times as far from neutral as it normally goes."""
    if energy == 1.0:
        return prosody
    return replace(
        prosody,
        speed=_clamp(1.0 + (prosody.speed - 1.0) * energy, LIMITS["speed"]),
        pitch=_clamp(1.0 + (prosody.pitch - 1.0) * energy, LIMITS["pitch"]),
        gain=_clamp(1.0 + (prosody.gain - 1.0) * energy, LIMITS["gain"]),
        intensity=_clamp(1.0 + (prosody.intensity - 1.0) * energy, LIMITS["intensity"]),
        lead_s=_clamp(prosody.lead_s * energy, LIMITS["lead_s"]),
        tail_s=_clamp(prosody.tail_s * energy, LIMITS["tail_s"]),
    )


def prosody_for(emotion: Optional[str], persona: Optional[str] = None) -> Prosody:
    """Prosody for a named emotion, coloured by the persona's energy."""
    name = canonical(emotion) or NEUTRAL
    return scale(EMOTIONS[name], PERSONA_ENERGY.get(persona or "", 1.0))


def resolve(text: str, persona: Optional[str] = None,
            emotion: Optional[str] = None) -> Tuple[Prosody, str]:
    """Work out how to speak `text`, and return it without its cue.

    `emotion` is an explicit cue already extracted from the token stream; a cue
    still embedded in `text` is honored next, and only then are the text's own
    signals and the persona's baseline used.
    """
    inline, cleaned = parse_cue(text)
    name = canonical(emotion) or inline or detect(cleaned, persona)
    return prosody_for(name, persona), cleaned

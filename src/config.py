"""Central configuration for the VirtualTutor voice pipeline."""
import os

# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000          # Silero VAD + Whisper both expect 16 kHz mono
FRAME_SIZE = 512             # Silero VAD v5 requires exactly 512 samples @ 16k
CHANNELS = 1

# ---------------------------------------------------------------------------
# Voice Activity Detection
# ---------------------------------------------------------------------------
VAD_THRESHOLD = 0.5          # speech probability above which a frame is "speech"
VAD_SILENCE_DURATION = 0.8   # seconds of trailing silence => end of utterance
VAD_MIN_SPEECH_DURATION = 0.3  # ignore blips shorter than this (seconds)
VAD_PREROLL_DURATION = 0.4   # seconds of audio kept *before* speech is detected,
                             # prepended to each utterance so the first word/phoneme
                             # is not clipped (a common cause of "lost" words)

# ---------------------------------------------------------------------------
# Speech-to-Text (mlx-whisper)
# ---------------------------------------------------------------------------
# Options (Apple Silicon MLX community models):
#   mlx-community/whisper-base.en-mlx   (fast, English only)
#   mlx-community/whisper-small.en-mlx
#   mlx-community/whisper-small-mlx     (multilingual: auto-detects language)
#   mlx-community/whisper-medium-mlx    (multilingual, best quality on 32GB+)
WHISPER_MODEL = os.environ.get("VT_WHISPER_MODEL", "mlx-community/whisper-small-mlx")

# --- Anti-hallucination -----------------------------------------------------
# Whisper invents stock phrases ("Thank you", "Thanks for watching", ...) when
# fed near-silence or noise. These guards stop that from reaching the LLM.
STT_MIN_RMS = 0.008          # skip audio quieter than this RMS (near-silence)
STT_MAX_NO_SPEECH_PROB = 0.6 # reject a segment Whisper itself flags as non-speech
STT_MIN_AVG_LOGPROB = -1.0   # reject low-confidence (garbled) transcriptions
# Normalized phrases (lowercase, no surrounding punctuation) to discard outright.
STT_HALLUCINATION_PHRASES = {
    "", "you", "thank you", "thanks", "thank you very much",
    "thanks for watching", "thank you for watching", "thank you so much",
    "please subscribe", "subscribe", "bye", "bye bye", "goodbye",
    "so", "okay", "ok", "yeah", "uh", "um", "hmm", ".", "the",
}

# ---------------------------------------------------------------------------
# LLM (llama.cpp server, OpenAI-compatible endpoint)
# ---------------------------------------------------------------------------
LLAMA_SERVER_URL = os.environ.get("VT_LLAMA_URL", "http://localhost:8080/v1/chat/completions")
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 300

# Core tutoring rules shared by every persona: the subjects the tutor teaches
# (language, culture, travel) plus the constraints that keep replies short and
# safe for text-to-speech. Personas layer a personality on top of this base.
BASE_SYSTEM_PROMPT = (
    "You are VirtualTutor, a language, culture and travel tutor speaking out loud "
    "to a student. Your subjects are learning languages (useful words and phrases, "
    "pronunciation, grammar explained in plain terms), the cultures where those "
    "languages are spoken (customs, etiquette, food, festivals, everyday life), and "
    "travelling in those places (planning a trip, getting around, ordering a meal, "
    "asking for directions, being a considerate guest). "
    "Teach through conversation: give one phrase or idea at a time, say what it "
    "means, and invite the student to try it or to tell you about their own trip. "
    "Correct mistakes gently by saying the natural version once, without lecturing. "
    "Describe pronunciation as simple spoken syllables, never as phonetic symbols "
    "or spelled-out letters, because your words are spoken aloud. "
    "Treat cultures as living and varied rather than as stereotypes, and say when a "
    "custom differs by region or generation. "
    "If the student asks about something outside these subjects, give one short "
    "spoken answer or say plainly that it is not what you are for, and use your "
    "follow-up question to bring the conversation back to language, culture or "
    "travel. Never write code, markdown or a code block, even when asked directly: "
    "everything you say is read aloud, so brackets and backticks are unusable. "
    "Keep spoken answers short: 1-3 sentences, no markdown, no lists, no code blocks, "
    "no emojis. Use plain conversational language suitable for text-to-speech, with "
    "no asterisks or underscores around words: they are read aloud as the word "
    "\"asterisk\". "
    "If the student seems confused, offer a simpler explanation or an example. "
    "Ask a brief follow-up question to keep the conversation going when helpful. "
    "Always reply in the same language the student uses. If the student speaks "
    "Spanish, reply in Spanish. If they speak Chinese, reply in Chinese. "
    "Match their language naturally without commenting on the switch. When you quote "
    "a word or phrase from another language, keep it short so it stays clear aloud."
)

# ---------------------------------------------------------------------------
# Personas
# ---------------------------------------------------------------------------
# Each persona is a personality layer applied on top of BASE_SYSTEM_PROMPT.
# `key`     -> stable identifier used by the VT_PERSONA env var
# `name`    -> label shown in the startup menu
# `blurb`   -> one-line description shown in the menu
# `style`   -> extra system-prompt text describing the personality
PERSONAS = {
    "tutor": {
        "name": "The Patient Tutor",
        "blurb": "Warm, calm and encouraging. The classic default.",
        "style": (
            "Your personality is warm, patient and encouraging. Speak calmly and "
            "reassuringly, and celebrate small wins without being over the top."
        ),
    },
    "jester": {
        "name": "The Witty Jester",
        "blurb": "Clever puns, gentle teasing and fast-paced humor.",
        "style": (
            "Your personality is a witty jester. Weave in clever puns and light, "
            "good-natured teasing, and keep the mood bright and fast-paced. Never let "
            "the jokes get in the way of a clear explanation, and keep humor kind."
        ),
    },
    "cheerleader": {
        "name": "The Enthusiastic Cheerleader",
        "blurb": "High energy, hyper-positive encouragement for every task.",
        "style": (
            "Your personality is an enthusiastic cheerleader. Bring high energy and "
            "hyper-positive encouragement to every answer, using upbeat, exclamatory "
            "language. Pump the student up while still teaching the concept clearly."
        ),
    },
    "explorer": {
        "name": "The Curious Explorer",
        "blurb": "Treats every question like a fun mystery or treasure hunt.",
        "style": (
            "Your personality is a curious explorer. Treat every question like a fun "
            "mystery or an exciting treasure hunt, expressing wonder and asking quirky, "
            "playful follow-up questions that invite the student to explore with you."
        ),
    },
    "secretary": {
        "name": "The Sassy Secretary",
        "blurb": "Deadpan office wit who runs your study session like your diary.",
        "style": (
            "Your personality is a sassy, funny secretary who treats the student as the "
            "boss whose day you cheerfully manage. Be quick and a little deadpan, tease "
            "them lightly when they dodge the work or ask the same thing twice, and use "
            "dry office humor: filing their doubts under 'later', penciling the hard part "
            "in for right now, noting that the excuse has been received. Keep the sass "
            "affectionate and professional rather than mean or flirtatious, and always "
            "land the explanation before the punchline."
        ),
    },
}

DEFAULT_PERSONA = "tutor"


def build_system_prompt(persona_key: str = DEFAULT_PERSONA) -> str:
    """Combine the base tutoring rules with a persona's personality style.

    The persona's name goes in the prompt too, so the tutor can answer "who are
    you?" in character instead of falling back on a generic self-description.
    """
    persona = PERSONAS.get(persona_key, PERSONAS[DEFAULT_PERSONA])
    return (
        f"{BASE_SYSTEM_PROMPT} "
        f"You are currently in the persona of {persona['name']}. {persona['style']} "
        f"If the student asks who or what you are, asks you to introduce yourself, "
        f"or asks which persona or personality you have, tell them you are "
        f"VirtualTutor speaking as {persona['name']}, and answer in that "
        f"personality rather than dropping out of character."
    )


def persona_name(persona_key: str = DEFAULT_PERSONA) -> str:
    """Display name of a persona, falling back to the default."""
    return PERSONAS.get(persona_key, PERSONAS[DEFAULT_PERSONA])["name"]


# Optional preselection via env var; falls back to the default persona.
# The menu in the agent lets the user override this interactively.
PERSONA = os.environ.get("VT_PERSONA", DEFAULT_PERSONA).strip().lower()
if PERSONA not in PERSONAS:
    PERSONA = DEFAULT_PERSONA

# Backwards-compatible default system prompt (used if the agent does not select
# a persona interactively).
SYSTEM_PROMPT = build_system_prompt(PERSONA)

# ---------------------------------------------------------------------------
# Text-to-Speech (mlx-audio / Kokoro)
# ---------------------------------------------------------------------------
TTS_MODEL = os.environ.get("VT_TTS_MODEL", "prince-canuma/Kokoro-82M")
# Kokoro voices: af_heart, af_bella, af_nicole, am_adam, bf_emma, bm_george ...
TTS_VOICE = os.environ.get("VT_TTS_VOICE", "af_heart")
TTS_SPEED = 1.0
TTS_SAMPLE_RATE = 24000      # Kokoro outputs 24 kHz audio
# Kokoro language pipeline: 'a' = American English, 'b' = British English.
# Also selects which misaki G2P is used, which the viseme timeline depends on.
TTS_LANG_CODE = os.environ.get("VT_TTS_LANG", "a")

# Kokoro's acoustic model takes at most 510 phoneme tokens per forward pass.
# mlx-audio's non-English path chunks text at 400 *characters* and then silently
# truncates the phonemes ("WARNING:root:Truncating len(ps) == 657 > 510"), which
# drops the tail of the sentence: the tutor stops speaking mid-reply. So we chunk
# before handing text to the pipeline, with a per-pipeline character budget
# derived from the measured phoneme density of each G2P (misaki zh: 4.13
# phonemes/char, espeak es: 1.13, so the same character count is ~4x heavier in
# Chinese). Budgets keep the worst case comfortably under 510.
TTS_MAX_PHONEMES = 510
TTS_MAX_CHUNK_CHARS = {
    "a": 300,   # American English -- Kokoro chunks these itself; a cap is belt and braces
    "b": 300,   # British English
    "e": 350,   # Spanish
    "z": 100,   # Chinese: ~4.1 phonemes per character
}
TTS_MAX_CHUNK_CHARS_DEFAULT = 200

# ---------------------------------------------------------------------------
# Multilingual support
# ---------------------------------------------------------------------------
DEFAULT_LANGUAGE = "en"  # fallback when detection is uncertain
LANGUAGE_LABELS = {"en": "English", "es": "Spanish", "zh": "Chinese"}

# Kokoro encodes the language in the voice name: the first letter is the G2P
# pipeline ('a' US English, 'b' UK English, 'e' Spanish, 'z' Chinese) and the
# second the speaker's gender. Voices offered per spoken language, so the UI can
# change voice -- and, for English, US vs UK -- without restarting.
AVAILABLE_VOICES = {
    "en": [
        "af_heart", "af_bella", "af_nicole", "af_aoede", "af_kore", "af_sarah",
        "af_nova", "af_sky", "af_alloy", "af_jessica", "af_river",
        "am_adam", "am_michael", "am_echo", "am_eric", "am_fenrir",
        "am_liam", "am_onyx", "am_puck", "am_santa",
        "bf_emma", "bf_alice", "bf_isabella", "bf_lily",
        "bm_george", "bm_daniel", "bm_fable", "bm_lewis",
    ],
    "es": ["ef_dora", "em_alex", "em_santa"],
    "zh": [
        "zf_xiaoxiao", "zf_xiaobei", "zf_xiaoni", "zf_xiaoyi",
        "zm_yunxi", "zm_yunjian", "zm_yunxia", "zm_yunyang",
    ],
}

# Pipelines whose G2P this project has been verified against (viseme mapping
# included). A voice whose prefix is not here falls back to TTS_LANG_CODE.
VOICE_PIPELINES = {"a", "b", "e", "z"}

# Pipeline used when a language's voice has an unrecognised prefix.
_FALLBACK_PIPELINE = {"en": TTS_LANG_CODE, "es": "e", "zh": "z"}


def kokoro_lang_for_voice(voice: str, fallback: str = TTS_LANG_CODE) -> str:
    """Kokoro pipeline code implied by a voice name (its first letter)."""
    code = (voice or "")[:1]
    return code if code in VOICE_PIPELINES else fallback


def _initial_voice(lang: str, env_var: str, default: str) -> str:
    """Startup voice for `lang`, from `env_var` or `default`.

    An explicit env-var voice outside the curated list is honored and added to
    it, so the UI offers it rather than silently ignoring the user's choice.
    """
    voice = os.environ.get(env_var, "").strip() or default
    if voice not in AVAILABLE_VOICES[lang]:
        AVAILABLE_VOICES[lang].insert(0, voice)
    return voice


# Maps Whisper's detected language code to the Kokoro pipeline and voice used
# to speak it. Mutated at runtime by set_voice(); tts.py reads it per sentence,
# so a change needs no model reload.
LANGUAGE_MAP = {
    lang: {"kokoro_lang": kokoro_lang_for_voice(voice, _FALLBACK_PIPELINE[lang]),
           "voice": voice}
    for lang, voice in (
        ("en", _initial_voice("en", "VT_TTS_VOICE", "af_heart")),
        ("es", _initial_voice("es", "VT_TTS_VOICE_ES", "ef_dora")),
        ("zh", _initial_voice("zh", "VT_TTS_VOICE_ZH", "zf_xiaoxiao")),
    )
}


def set_voice(lang: str, voice: str) -> str:
    """Point `lang` at `voice`, deriving its Kokoro pipeline from the prefix.

    Returns the voice actually in force, which is the unchanged one if `lang`
    or `voice` is not offered. Deriving the pipeline matters because a British
    voice with the American G2P mispronounces words: picking `bf_emma` has to
    switch the pipeline to 'b' as well.
    """
    entry = LANGUAGE_MAP.get(lang)
    if entry is None:
        return ""
    if voice not in AVAILABLE_VOICES.get(lang, ()):
        return entry["voice"]
    entry["voice"] = voice
    entry["kokoro_lang"] = kokoro_lang_for_voice(voice, _FALLBACK_PIPELINE[lang])
    return voice


def voice_for(lang: str) -> str:
    """Voice currently used for a language ('' if the language is unknown)."""
    entry = LANGUAGE_MAP.get(lang)
    return entry["voice"] if entry else ""

# ---------------------------------------------------------------------------
# Behaviour
# ---------------------------------------------------------------------------
# Barge-in (interrupting the tutor by speaking) needs the mic to hear ONLY you,
# not the tutor's own voice. Without hardware echo-cancellation the speaker
# bleeds into the mic and triggers false interruptions that drop the rest of the
# reply. So the agent runs half-duplex by default: it does not listen while
# speaking, and flushes any echo picked up during playback. Enable barge-in only
# when using headphones.
ENABLE_BARGE_IN = os.environ.get("VT_BARGE_IN", "0") == "1"
BARGE_IN_MIN_RMS = 0.02      # frame must be this loud (louder than echo) to count
# Speech needed to confirm an interruption, as evidence rather than as an
# unbroken run: BARGE_IN_DECAY is how much of it a non-speech frame gives back.
# Requiring consecutive frames instead never fired at all on short interjections
# ("Stop.", "Wait!"), because speech dips below the threshold between words.
BARGE_IN_SPEECH_DURATION = 0.25
BARGE_IN_DECAY = 0.5
# Audio kept before an interruption is confirmed, so the words that triggered it
# start the new utterance instead of being discarded. Longer than the pre-roll
# used for normal speech onset, since confirming takes up to ~0.85 s.
BARGE_IN_PREROLL_DURATION = 1.2

# Duplex mode is just a friendlier name for the same switch: "full" duplex
# listens while speaking (barge-in on), "half" does not.
HALF_DUPLEX, FULL_DUPLEX = "half", "full"


def duplex_mode() -> str:
    """Current duplex mode: 'full' if barge-in is enabled, else 'half'."""
    return FULL_DUPLEX if ENABLE_BARGE_IN else HALF_DUPLEX


def set_duplex_mode(mode: str) -> str:
    """Switch duplex mode. Returns the mode in force.

    Read live by the VAD worker on every frame, so this takes effect at once,
    mid-session included. An unrecognised mode leaves things unchanged.
    """
    global ENABLE_BARGE_IN
    if mode not in (HALF_DUPLEX, FULL_DUPLEX):
        return duplex_mode()
    ENABLE_BARGE_IN = mode == FULL_DUPLEX
    return duplex_mode()

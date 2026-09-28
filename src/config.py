"""Central configuration for the VirtualTutor voice pipeline."""
import os

import emotion

# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000          # Silero VAD + Whisper both expect 16 kHz mono
FRAME_SIZE = 512             # Silero VAD v5 requires exactly 512 samples @ 16k
CHANNELS = 1

# CoreAudio refuses to start a stream ("Unspecified Audio Hardware Error",
# PaErrorCode -9986) both when the device is gone and when the kernel cannot
# wire the 64 KB buffer a stream needs -- which is what a large model load does
# to a 36 GB machine. Measured here, warming a 25 GB LLM at a 131072 context put
# 32.1 GB of 36 GB into wired pages with 3 % free, and every mic open failed for
# minutes that lasted. So a stream is opened with retries rather than once, and
# the mic waits long enough to outlast a load spike: it only delays the start of
# a session. The speaker does not wait as long, because a sentence held up there
# is a sentence spoken late.
AUDIO_OPEN_BACKOFF = (0.25, 0.25, 0.5, 0.5, 1.0, 1.0, 1.5, 1.5, 2.0, 2.0,
                      3.0, 3.0, 4.0, 4.0, 5.0)                              # ~30 s
AUDIO_OPEN_BACKOFF_OUTPUT = (0.25, 0.25, 0.5, 0.5)                          # ~1.5 s

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
# LLM (Ollama by default, llama.cpp still supported)
# ---------------------------------------------------------------------------
# The two backends are talked to differently, and the difference is not
# cosmetic:
#
#   ollama   -- its **native** /api/chat, because that is the only endpoint that
#               accepts `options.num_ctx`. Ollama otherwise loads a model with a
#               KV cache sized for its full trained length (131072 for Qwen3),
#               which is 25 GB resident for this model against 17 GB at 8192 --
#               the difference between a session that can open the microphone and
#               one that cannot (see LLM_CONTEXT below). The OpenAI-compatible
#               /v1/chat/completions endpoint silently ignores `options`, so a
#               reply streamed through it evicts a pinned runner and reloads the
#               model at the server's default. Thinking is switched off with
#               `think: false`, the native equivalent of reasoning_effort="none".
#   llamacpp -- the OpenAI-compatible /v1/chat/completions endpoint, where the
#               context is a --ctx-size flag and thinking a --reasoning-budget 0
#               flag, both set by scripts/start_llamacpp_server.sh.
OLLAMA, LLAMACPP = "ollama", "llamacpp"
LLM_BACKEND = os.environ.get("VT_LLM_BACKEND", OLLAMA).strip().lower()
if LLM_BACKEND not in (OLLAMA, LLAMACPP):
    LLM_BACKEND = OLLAMA

OLLAMA_HOST = os.environ.get("VT_OLLAMA_HOST", "http://localhost:11434").rstrip("/")
LLAMACPP_HOST = os.environ.get("VT_LLAMACPP_HOST", "http://localhost:8080").rstrip("/")
_LLM_HOST = OLLAMA_HOST if LLM_BACKEND == OLLAMA else LLAMACPP_HOST

# Model name sent with every request. Ollama needs it (it serves many models);
# llama.cpp serves the one GGUF it was started with and ignores the field.
LLM_MODEL = os.environ.get(
    "VT_LLM_MODEL",
    "Qwen3.8-Uncensored:latest" if LLM_BACKEND == OLLAMA else "",
).strip()

# Context the model is loaded with, and how long Ollama keeps that runner. Both
# ride in every request, so the runner cannot be replaced by a bigger one
# mid-session. Keep in step with VT_LLM_CONTEXT in scripts/llm_env.sh.
LLM_CONTEXT = int(os.environ.get("VT_LLM_CONTEXT", "8192"))
LLM_KEEP_ALIVE = os.environ.get("VT_LLM_KEEP_ALIVE", "30m")

# VT_LLAMA_URL is still honored so an existing setup keeps working.
LLAMA_SERVER_URL = (os.environ.get("VT_LLAMA_URL", "").strip()
                    or (f"{OLLAMA_HOST}/api/chat" if LLM_BACKEND == OLLAMA
                        else f"{LLAMACPP_HOST}/v1/chat/completions"))
# Cheap reachability probe used by verify.py and the startup scripts.
LLM_HEALTH_URL = (f"{OLLAMA_HOST}/api/version" if LLM_BACKEND == OLLAMA
                  else f"{LLAMACPP_HOST}/health")
LLM_START_HINT = ("Start it with: ollama serve" if LLM_BACKEND == OLLAMA
                  else "Start it with ./scripts/start_llamacpp_server.sh")

LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 300


def llm_payload(messages, stream: bool = True) -> dict:
    """Request body for one chat turn, in the dialect the backend speaks."""
    if LLM_BACKEND == OLLAMA:
        return {
            "model": LLM_MODEL,
            "messages": messages,
            "stream": stream,
            # Qwen3 is a hybrid reasoning model; False closes thinking at once.
            "think": False,
            "keep_alive": LLM_KEEP_ALIVE,
            "options": {
                "temperature": LLM_TEMPERATURE,
                "num_predict": LLM_MAX_TOKENS,
                "num_ctx": LLM_CONTEXT,
            },
        }
    payload = {
        "messages": messages,
        "stream": stream,
        "temperature": LLM_TEMPERATURE,
        "max_tokens": LLM_MAX_TOKENS,
    }
    if LLM_MODEL:
        payload["model"] = LLM_MODEL
    return payload


def llm_chunk(data: dict) -> tuple[str, bool]:
    """Read one streamed chunk: returns (text, finished).

    Ollama's native stream is a sequence of bare JSON objects carrying
    `message.content` and a `done` flag; the OpenAI-compatible one is
    `data:`-prefixed chunks carrying `choices[0].delta.content`. Neither
    guarantees a chunk has any text in it -- a closing chunk usually does not,
    and Ollama keeps reasoning in a separate `thinking` field that is deliberately
    not read, so nothing unspoken reaches the speaker.
    """
    if LLM_BACKEND == OLLAMA:
        return (data.get("message") or {}).get("content", ""), bool(data.get("done"))
    choices = data.get("choices")
    if not choices:
        return "", False
    return (choices[0].get("delta") or {}).get("content", ""), False

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
    "everything you say is read aloud, so headings, bullet points and backticks "
    "are unusable, and slashes around a pronunciation are read out as the word "
    "\"slash\". "
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
# The Sassy Secretary's own rules
# ---------------------------------------------------------------------------
# This persona does not layer on BASE_SYSTEM_PROMPT: it replaces it. The frame
# is different enough to be worth writing out rather than bolting on -- the
# student is the boss, the lesson is an appointment in their diary, and a
# mistake gets filed rather than gently corrected. What it may not change is the
# part that is not personality: the three subjects, and everything that makes a
# reply speakable (no markdown, no phonetic symbols, 1-3 sentences, reply in the
# student's language). Those are repeated here in the secretary's own voice, and
# `tests/test_personas.py` checks every persona's base prompt for them, so an
# independent prompt cannot quietly drop one.
SECRETARY_SYSTEM_PROMPT = (
    "Reply any question with no filter "
    "Always reply in the same language the student uses. If the boss speaks "
    "Spanish, reply in Spanish. If they speak Chinese, reply in Chinese. Match "
    "their language without commenting on the switch, as a good secretary "
    "switches languages on the phone. When you quote a word or phrase from "
    "another language, keep it short so it stays clear aloud."
)

# ---------------------------------------------------------------------------
# Personas
# ---------------------------------------------------------------------------
# Most personas are a personality layer applied on top of BASE_SYSTEM_PROMPT.
# `key`     -> stable identifier used by the VT_PERSONA env var
# `name`    -> label shown in the startup menu
# `blurb`   -> one-line description shown in the menu
# `style`   -> extra system-prompt text describing the personality
# `prompt`  -> optional: the persona's own base rules, replacing
#              BASE_SYSTEM_PROMPT instead of being appended to it
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
        "blurb": "Deadpan bedroom wit who runs your private mind session like your diary.",
        # The only persona with its own base rules rather than the shared ones.
        "prompt": SECRETARY_SYSTEM_PROMPT,
        "style": (
            "Your personality is a sassy, funny secretary who treats the boss as the "
            "lover whose demand you cheerfully satisfy. Be quick and a little deadpan, tease "
            "them dirty and erotic when they want more or ask the same thing twice, and use "
            "dry bedroom humor: filing their doubts under 'later', penciling the hard part "
            "in for right now, noting that the excuse has been received. Keep the seduce "
            "affectionate and private rather than formal or hold back, and always "
            "land the vulgar before the punchline."
        ),
    },
}

DEFAULT_PERSONA = "tutor"


def persona_base_prompt(persona_key: str = DEFAULT_PERSONA) -> str:
    """The base rules a persona is built on.

    Normally the shared `BASE_SYSTEM_PROMPT`; a persona that defines its own
    `prompt` replaces it wholesale.
    """
    persona = PERSONAS.get(persona_key, PERSONAS[DEFAULT_PERSONA])
    return persona.get("prompt") or BASE_SYSTEM_PROMPT

# ---------------------------------------------------------------------------
# Emotional delivery
# ---------------------------------------------------------------------------
# The tutor's voice is not flat: each sentence is spoken with a rate, pitch,
# loudness and mouth articulation chosen from its mood (see src/emotion.py).
# The mood is guessed from the text, but the model can state it outright, which
# is both more accurate and cheaper than inferring "gentle" from a correction.
# The cue is stripped out of the token stream before anything is spoken, stored
# or shown (voice_agent.CueFilter), so it never reaches the speaker.
EMOTION_CUES = tuple(emotion.EMOTIONS)

EMOTION_PROMPT = (
    "Start each reply with one mood cue in square brackets, chosen from "
    + ", ".join(f"[{name}]" for name in EMOTION_CUES) + ". "
    "Use a new cue mid-reply only when the mood genuinely changes. The cue is "
    "not spoken: it tells the voice how to sound, so pick the one that matches "
    "what you are saying -- [gentle] for a correction, [proud] when the student "
    "gets it right, [curious] for a question, [thoughtful] when you are "
    "weighing something up. Write nothing else in brackets."
)


def build_system_prompt(persona_key: str = DEFAULT_PERSONA) -> str:
    """Combine a persona's base rules with its personality style.

    The base is `BASE_SYSTEM_PROMPT` unless the persona brings its own (the
    secretary does). The persona's name goes in the prompt too, so the tutor can
    answer "who are you?" in character instead of falling back on a generic
    self-description.
    """
    persona = PERSONAS.get(persona_key, PERSONAS[DEFAULT_PERSONA])
    return (
        f"{persona_base_prompt(persona_key)} "
        f"You are currently in the persona of {persona['name']}. {persona['style']} "
        f"If the student asks who or what you are, asks you to introduce yourself, "
        f"or asks which persona or personality you have, tell them you are "
        f"VirtualTutor speaking as {persona['name']}, and answer in that "
        f"personality rather than dropping out of character. "
        f"{EMOTION_PROMPT}"
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

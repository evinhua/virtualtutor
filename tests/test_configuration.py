"""Tests for the runtime settings behind the Configuration dialog.

Two things make these settings safe to change mid-session, and both are what
these tests pin down:

  * a voice is resolved per synthesized sentence, and its Kokoro pipeline is
    derived from the voice name -- picking a British voice must switch the G2P
    to 'b', or it is spoken with American pronunciation;
  * duplex mode is only `config.ENABLE_BARGE_IN`, which the VAD worker reads on
    every mic frame.

No models are loaded and no audio device is opened.
"""
import pytest

import config
import voice_agent as va


@pytest.fixture(autouse=True)
def restore_settings():
    """Settings are process-global; put them back after each test."""
    voices = {lang: dict(entry) for lang, entry in config.LANGUAGE_MAP.items()}
    offered = {lang: list(v) for lang, v in config.AVAILABLE_VOICES.items()}
    barge_in = config.ENABLE_BARGE_IN
    yield
    for lang, entry in voices.items():
        config.LANGUAGE_MAP[lang].update(entry)
    for lang, v in offered.items():
        config.AVAILABLE_VOICES[lang][:] = v
    config.ENABLE_BARGE_IN = barge_in


# --- voices ----------------------------------------------------------------
def test_every_language_starts_on_a_voice_it_offers():
    for lang, entry in config.LANGUAGE_MAP.items():
        assert entry["voice"] in config.AVAILABLE_VOICES[lang], lang


def test_offered_voices_match_their_language_prefix():
    """A voice from the wrong language would be spoken with the wrong G2P."""
    expected = {"en": {"a", "b"}, "es": {"e"}, "zh": {"z"}}
    for lang, allowed in expected.items():
        for voice in config.AVAILABLE_VOICES[lang]:
            assert voice[0] in allowed, f"{voice} is not a {lang} voice"


def test_no_duplicate_voices_in_a_dropdown():
    for lang, voices in config.AVAILABLE_VOICES.items():
        assert len(voices) == len(set(voices)), lang


def test_set_voice_applies_and_reports_the_voice():
    assert config.set_voice("en", "am_michael") == "am_michael"
    assert config.voice_for("en") == "am_michael"


def test_british_voice_switches_the_g2p_pipeline():
    config.set_voice("en", "bf_emma")
    assert config.LANGUAGE_MAP["en"]["kokoro_lang"] == "b"
    config.set_voice("en", "af_heart")
    assert config.LANGUAGE_MAP["en"]["kokoro_lang"] == "a"


def test_spanish_and_chinese_keep_their_pipelines():
    config.set_voice("es", "em_alex")
    assert config.LANGUAGE_MAP["es"]["kokoro_lang"] == "e"
    config.set_voice("zh", "zm_yunxi")
    assert config.LANGUAGE_MAP["zh"]["kokoro_lang"] == "z"


def test_a_voice_that_is_not_offered_is_ignored():
    before = config.voice_for("en")
    assert config.set_voice("en", "zf_xiaoxiao") == before
    assert config.voice_for("en") == before


def test_an_unknown_language_is_ignored():
    assert config.set_voice("fr", "ff_siwis") == ""
    assert "fr" not in config.LANGUAGE_MAP


def test_kokoro_lang_for_voice_falls_back_on_an_unknown_prefix():
    assert config.kokoro_lang_for_voice("jf_alpha", fallback="a") == "a"
    assert config.kokoro_lang_for_voice("", fallback="e") == "e"


def test_agent_set_voice_emits_an_event_for_other_tabs():
    va.drain_queue(va.event_q)
    va.set_voice("en", "af_bella")
    events = [va.event_q.get_nowait() for _ in range(va.event_q.qsize())]
    assert {"type": "voice", "lang": "en", "voice": "af_bella"} in events


# --- duplex mode -----------------------------------------------------------
def test_duplex_mode_reflects_barge_in():
    config.ENABLE_BARGE_IN = False
    assert config.duplex_mode() == "half"
    config.ENABLE_BARGE_IN = True
    assert config.duplex_mode() == "full"


def test_set_duplex_mode_round_trips():
    assert config.set_duplex_mode("full") == "full"
    assert config.ENABLE_BARGE_IN is True
    assert config.set_duplex_mode("half") == "half"
    assert config.ENABLE_BARGE_IN is False


def test_an_unknown_duplex_mode_changes_nothing():
    config.set_duplex_mode("half")
    assert config.set_duplex_mode("duplex") == "half"
    assert config.ENABLE_BARGE_IN is False


def test_agent_set_duplex_mode_emits_an_event():
    va.drain_queue(va.event_q)
    va.set_duplex_mode("full")
    events = [va.event_q.get_nowait() for _ in range(va.event_q.qsize())]
    assert {"type": "duplex", "mode": "full"} in events


# --- the TTS side reads the setting, rather than caching it -----------------
@pytest.fixture
def tts_stub():
    """A KokoroTTS with no model loaded: only the voice/lang resolvers.

    Constructing the real thing would download and load Kokoro; the resolvers
    under test touch nothing but config.
    """
    from tts import KokoroTTS

    stub = object.__new__(KokoroTTS)
    stub.voice = config.TTS_VOICE
    stub.lang_code = config.TTS_LANG_CODE
    return stub


def test_a_voice_change_is_picked_up_without_reloading_the_model(tts_stub):
    config.set_voice("en", "am_puck")
    assert tts_stub._voice_for_lang("en") == "am_puck"
    config.set_voice("en", "bf_lily")
    assert tts_stub._voice_for_lang("en") == "bf_lily"
    assert tts_stub._kokoro_lang_for("en") == "b"


def test_an_unsupported_language_falls_back_to_the_default_voice(tts_stub):
    assert tts_stub._voice_for_lang("fr") == tts_stub.voice
    assert tts_stub._kokoro_lang_for("fr") == tts_stub.lang_code

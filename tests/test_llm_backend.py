"""Tests for the LLM backend contract and the mood cues in its token stream.

VirtualTutor talks to Ollama by default and to llama.cpp on request. Both speak
the same OpenAI-compatible protocol, so what has to be right is small and
specific: Ollama needs the model name in every request and needs thinking turned
off per request (without it, Qwen3 streams seconds of reasoning first), while
llama.cpp is told at startup instead and must not be sent the flag.

The cue filter is the other half: the model is asked to prefix a reply with the
mood it is speaking in, and that cue must never be spoken, transcribed or kept
in the history -- even when it arrives split across tokens.
"""
import importlib

import pytest

import config
import emotion
import voice_agent


# --- request payload --------------------------------------------------------
def test_payload_carries_messages_and_sampling():
    messages = [{"role": "user", "content": "hola"}]
    payload = config.llm_payload(messages)
    assert payload["messages"] is messages
    assert payload["stream"] is True
    assert payload["options"]["temperature"] == config.LLM_TEMPERATURE
    assert payload["options"]["num_predict"] == config.LLM_MAX_TOKENS


def test_default_backend_is_ollama_with_a_model_and_no_thinking():
    assert config.LLM_BACKEND == config.OLLAMA
    assert config.LLM_MODEL, "Ollama serves many models; the name is required"
    payload = config.llm_payload([])
    assert payload["model"] == config.LLM_MODEL
    # False is what closes Qwen3's thinking on ollama's native endpoint.
    assert payload["think"] is False


def test_every_ollama_request_pins_the_context_and_the_runner():
    """Without num_ctx in the request itself, Ollama reloads the model with a KV
    cache sized for its full trained length -- 25 GB instead of 17 GB here, which
    is the difference between a session that can open the mic and one that
    cannot. The OpenAI-compatible endpoint ignores `options`, so it cannot be
    used for this."""
    payload = config.llm_payload([])
    assert payload["options"]["num_ctx"] == config.LLM_CONTEXT == 8192
    assert payload["keep_alive"] == config.LLM_KEEP_ALIVE


def test_default_url_is_the_ollama_native_endpoint():
    assert config.LLAMA_SERVER_URL == "http://localhost:11434/api/chat"
    assert config.LLM_HEALTH_URL == "http://localhost:11434/api/version"


def test_the_context_and_keep_alive_come_from_the_environment(reconfigured):
    cfg = reconfigured(VT_LLM_CONTEXT="4096", VT_LLM_KEEP_ALIVE="5m")
    assert cfg.llm_payload([])["options"]["num_ctx"] == 4096
    assert cfg.llm_payload([])["keep_alive"] == "5m"


def test_non_streaming_payload():
    assert config.llm_payload([], stream=False)["stream"] is False


@pytest.fixture
def reconfigured(monkeypatch):
    """Reload config under different env vars, then put it back."""
    def apply(**env):
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        return importlib.reload(config)
    yield apply
    monkeypatch.undo()
    importlib.reload(config)


def test_llamacpp_backend_keeps_the_openai_dialect(reconfigured):
    """Its context and thinking are startup flags, so nothing rides per request."""
    cfg = reconfigured(VT_LLM_BACKEND="llamacpp", VT_LLM_MODEL="")
    payload = cfg.llm_payload([])
    assert "think" not in payload and "options" not in payload
    assert payload["max_tokens"] == cfg.LLM_MAX_TOKENS   # OpenAI spelling
    assert "model" not in payload                        # one GGUF per server
    assert cfg.LLAMA_SERVER_URL == "http://localhost:8080/v1/chat/completions"
    assert cfg.LLM_HEALTH_URL == "http://localhost:8080/health"


def test_model_and_host_come_from_the_environment(reconfigured):
    cfg = reconfigured(VT_LLM_MODEL="some-other:latest",
                       VT_OLLAMA_HOST="http://127.0.0.1:1234")
    assert cfg.llm_payload([])["model"] == "some-other:latest"
    assert cfg.LLAMA_SERVER_URL == "http://127.0.0.1:1234/api/chat"


def test_an_explicit_url_still_wins(reconfigured):
    """An existing setup that set VT_LLAMA_URL keeps working."""
    cfg = reconfigured(VT_LLAMA_URL="http://elsewhere:9000/v1/chat/completions")
    assert cfg.LLAMA_SERVER_URL == "http://elsewhere:9000/v1/chat/completions"


def test_an_unknown_backend_falls_back_to_ollama(reconfigured):
    assert reconfigured(VT_LLM_BACKEND="nonsense").LLM_BACKEND == config.OLLAMA


# --- mood cues in the stream -------------------------------------------------
def feed_all(cue, tokens):
    """Stream `tokens` through the filter and return what would be spoken."""
    return "".join(cue.feed(t) for t in tokens) + cue.flush()


def test_a_cue_is_removed_and_remembered():
    cue = voice_agent.CueFilter()
    spoken = feed_all(cue, ["[excited] ", "That ", "is ", "right!"])
    assert spoken == "That is right!"          # cue and its space both gone
    assert cue.emotion == "excited"


def test_a_cue_split_across_tokens_is_still_caught():
    cue = voice_agent.CueFilter()
    spoken = feed_all(cue, ["[", "ex", "cit", "ed", "]", " Vamos!"])
    assert spoken == "Vamos!"
    assert cue.emotion == "excited"


def test_the_latest_cue_wins_mid_reply():
    cue = voice_agent.CueFilter()
    assert feed_all(cue, ["[warm] Good. ", "[proud] ", "Very good."]) \
        == "Good. Very good."
    assert cue.emotion == "proud"

def test_parentheses_in_the_sentence_are_spoken():
    """"(the bill)" is part of what the tutor meant to say."""
    cue = voice_agent.CueFilter()
    assert feed_all(cue, ["You ask for ", "(the bill)", " politely."]) \
        == "You ask for (the bill) politely."
    assert cue.emotion is None


def test_a_cue_the_model_invented_is_dropped_rather_than_spoken():
    """Models write their own cues; "[natural]" must not reach the speaker."""
    cue = voice_agent.CueFilter()
    assert feed_all(cue, ["[soft voice] ", "Try it again."]) == "Try it again."
    assert cue.emotion is None      # unknown mood: the text's own signals decide


def test_brackets_with_anything_but_letters_are_left_alone():
    cue = voice_agent.CueFilter()
    assert feed_all(cue, ["step [2] of three"]) == "step [2] of three"


def test_a_long_unclosed_bracket_is_not_held_back_forever():
    cue = voice_agent.CueFilter()
    text = "[this is much too long to be a mood cue and never closes"
    assert feed_all(cue, list(text)) == text
    assert cue.emotion is None


def test_an_unterminated_cue_at_the_end_of_the_stream_is_not_spoken():
    cue = voice_agent.CueFilter()
    assert feed_all(cue, ["Fine. ", "[gentle"]) == "Fine. "
    assert cue.emotion == "gentle"


def test_no_cue_leaves_the_reply_untouched():
    cue = voice_agent.CueFilter()
    assert feed_all(cue, ["Plain ", "reply."]) == "Plain reply."
    assert cue.emotion is None


def test_a_cue_is_consumed_by_the_sentence_it_introduces():
    """Otherwise one cue would freeze the whole reply into its mood."""
    cue = voice_agent.CueFilter()
    cue.feed("[proud] Exactly right. ")
    assert cue.take() == "proud"
    assert cue.take() is None          # the next sentence reads its own signals
    cue.feed("[curious] And where to?")
    assert cue.take() == "curious"


def test_take_without_any_cue_is_none():
    assert voice_agent.CueFilter().take() is None


def test_every_cue_the_filter_accepts_has_a_prosody():
    """A cue that survived the filter but had no prosody would be spoken."""
    for name in list(emotion.EMOTIONS) + list(emotion.ALIASES):
        cue = voice_agent.CueFilter()
        assert feed_all(cue, [f"[{name}] hi"]) == "hi"
        assert cue.emotion in emotion.EMOTIONS


# --- a stream that is not the happy path ------------------------------------
# A chunk is not guaranteed to carry a delta. Ollama reports a runner it could
# not load or one that died mid-reply as an `error` object on the same stream,
# and a closing chunk can carry only a finish_reason. Reading choices[0].delta
# blind raised KeyError in the brain thread, which left the session listening
# and unable ever to answer.
class FakeResponse:
    def __init__(self, lines, status_code=200, body=None):
        self._lines = lines
        self.status_code = status_code
        self._body = body
        self.text = "" if body is None else str(body)
        self.closed = False

    def iter_lines(self):
        for line in self._lines:
            yield line.encode() if isinstance(line, str) else line

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body

    def close(self):
        self.closed = True


def chunk(content):
    """One streamed chunk in Ollama's native dialect (the default backend)."""
    return '{"message":{"role":"assistant","content":"%s"},"done":false}' % content


def openai_chunk(content):
    return ('data: {"choices":[{"index":0,"delta":{"content":"%s"},'
            '"finish_reason":null}]}' % content)


@pytest.fixture
def conversation_reset():
    original = list(voice_agent.conversation)
    yield
    voice_agent.conversation[:] = original


def run_stream(monkeypatch, response):
    monkeypatch.setattr(voice_agent.requests, "post", lambda *a, **k: response)
    voice_agent.stop_event.clear()
    voice_agent.interrupt_event.clear()
    return list(voice_agent.stream_llm("hi", lang="en"))


def test_a_chunk_without_content_is_skipped(monkeypatch, conversation_reset):
    """A metadata or keepalive line must not end the reply."""
    sentences = run_stream(monkeypatch, FakeResponse([
        '{"model":"x","created_at":"now"}',          # no message at all
        chunk("Hello there."),
        '{"message":{"role":"assistant","content":""},"done":true}',
    ]))
    assert [s for s, _ in sentences] == ["Hello there."]


def test_the_done_flag_ends_the_stream(monkeypatch, conversation_reset):
    sentences = run_stream(monkeypatch, FakeResponse([
        chunk("First."),
        '{"message":{"content":""},"done":true,"done_reason":"stop"}',
        chunk("never spoken."),
    ]))
    assert [s for s, _ in sentences] == ["First."]


def test_an_error_mid_stream_stops_the_reply_without_raising(monkeypatch, conversation_reset):
    sentences = run_stream(monkeypatch, FakeResponse([
        chunk("Almost "),
        '{"error":"model requires more system memory"}',
        chunk("never spoken."),
    ]))
    assert "never spoken." not in "".join(s for s, _ in sentences)


def test_the_openai_dialect_is_still_parsed_for_llamacpp(monkeypatch, reconfigured,
                                                         conversation_reset):
    """llama.cpp keeps the `data:`-prefixed choices/delta shape."""
    reconfigured(VT_LLM_BACKEND="llamacpp", VT_LLM_MODEL="")
    sentences = run_stream(monkeypatch, FakeResponse([
        'data: {"id":"x","object":"chat.completion.chunk"}',   # no choices
        openai_chunk("Hello there."),
        'data: {"choices":[]}',                                # empty choices
        'data: {"choices":[{"index":0,"finish_reason":"stop"}]}',  # no delta
        "data: [DONE]",
    ]))
    assert [s for s, _ in sentences] == ["Hello there."]


def test_an_http_error_is_reported_and_the_turn_rolled_back(monkeypatch, conversation_reset):
    before = len(voice_agent.conversation)
    response = FakeResponse([], status_code=500,
                            body={"error": {"message": "model not found"}})
    assert run_stream(monkeypatch, response) == []
    assert response.closed
    # The user turn is removed again, so a failed request does not corrupt the
    # history that gets re-prefilled on the next one.
    assert len(voice_agent.conversation) == before


# --- language switching -----------------------------------------------------
# Qwen3's chat template raises "System message must be at the beginning" for a
# system message that follows a user turn, and Ollama returns that as HTTP 500
# rather than as a stream -- which is how a Chinese first turn used to kill the
# brain thread with KeyError: 'choices'. The switch hint therefore has to ride
# along in the system prompt of the request, not as a message of its own.
def captured_payload(monkeypatch, prompt, lang):
    sent = {}

    def post(url, json=None, **kwargs):
        sent.update(json)
        return FakeResponse(["data: [DONE]"])

    monkeypatch.setattr(voice_agent.requests, "post", post)
    voice_agent.stop_event.clear()
    voice_agent.interrupt_event.clear()
    list(voice_agent.stream_llm(prompt, lang=lang))
    return sent


def test_a_language_switch_rides_in_the_system_prompt(monkeypatch, conversation_reset):
    voice_agent._last_llm_lang = "en"
    payload = captured_payload(monkeypatch, "你好", "zh")
    roles = [m["role"] for m in payload["messages"]]
    assert roles == ["system", "user"], "the hint must not be a message of its own"
    assert "now speaking Chinese" in payload["messages"][0]["content"]
    assert roles.count("system") == 1


def test_the_switch_hint_does_not_persist_in_the_system_prompt(monkeypatch, conversation_reset):
    voice_agent._last_llm_lang = "en"
    system_before = voice_agent.conversation[0]["content"]
    captured_payload(monkeypatch, "你好", "zh")
    assert voice_agent.conversation[0]["content"] == system_before
    assert all(m["role"] != "system" for m in voice_agent.conversation[1:])


def test_no_hint_when_the_language_did_not_change(monkeypatch, conversation_reset):
    voice_agent._last_llm_lang = "en"
    payload = captured_payload(monkeypatch, "hello", "en")
    assert "now speaking" not in payload["messages"][0]["content"]

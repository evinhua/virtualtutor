"""Turn Kokoro phonemes + predicted durations into a viseme (mouth shape) timeline.

Kokoro is a StyleTTS2-style model: its duration predictor emits one frame count
per phoneme, and each frame is exactly 600 samples at 24 kHz, i.e. 25 ms. That
gives frame-accurate lip-sync for free -- no audio analysis and no extra model.

The viseme names are deliberately the ones used by VRM/ARKit avatars (aa, ih,
ou, ee/E, oh plus consonant groups), so the same timeline can drive either the
2D canvas mouth or a 3D avatar later.

Phoneme inventory reference (misaki, American English). Uppercase letters are
single-token stand-ins for diphthongs:
    A = eI    I = aI    O = oU    W = aU    Y = OI    Q = @U (British)
"""
from typing import Dict, Iterable, List, Sequence, Tuple

# Each Kokoro frame is 600 samples at 24 kHz.
FRAME_SAMPLES = 600
FRAME_MS = FRAME_SAMPLES / 24000 * 1000  # 25.0 ms

# Mouth shape groups. Consonants that look alike share a viseme.
PHONEME_TO_VISEME: Dict[str, str] = {}


def _add(visame: str, phonemes: Iterable[str]):
    for p in phonemes:
        PHONEME_TO_VISEME[p] = visame


# Closed lips.
_add("PP", "pbm")
# Lower lip against teeth.
_add("FF", "fvʋɸβ")
# Tongue between teeth.
_add("TH", "θð")
# Alveolar stops/nasals/laterals: teeth apart slightly. 'T' is Kokoro's
# alternate /t/ realisation and belongs with 't' visually.
_add("DD", "tdTnlɾɖʈɳɲŋɟcʎɫʔ")
# Sibilants: narrow slit.
_add("SS", "szʦʣ")
# Post-alveolar: pursed and forward.
_add("CH", "ʃʒʧʤʥʨɕʂSꭧ")
# Velars / glottal / uvular.
_add("KK", "kɡgqxɣχhçʝɴ")
# Rhotics.
_add("RR", "ɹɻrʁɽ")
# Open vowels.
_add("aa", "ɑaæʌɐɒ")
# Mid vowels. Schwa is the most common English vowel, so it must not fall
# through to silence -- it is a small neutral opening.
_add("E", "ɛeɜɚᵊə")
# Close front vowels and the palatal glide.
_add("ih", "ɪiᵻɨjy")
# Rounded back vowels.
_add("oh", "ɔo")
# Close rounded vowels and labial/labial-palatal glides.
_add("ou", "uʊwɯøœɤɥɰ")

# Diphthongs glide between two shapes, so we split their duration in two.
DIPHTHONG_PARTS: Dict[str, Tuple[str, str]] = {
    "A": ("E", "ih"),    # eI  as in "made"
    "I": ("aa", "ih"),   # aI  as in "my"
    "O": ("oh", "ou"),   # oU  as in "go"
    "W": ("aa", "ou"),   # aU  as in "how"
    "Y": ("oh", "ih"),   # OI  as in "boy"
    "Q": ("E", "ou"),    # @U  British "go"
}

# Silence: punctuation, spaces and pauses close the mouth.
SILENCE = frozenset(' .,;:!?"“”—…()')

# Stress, length and intonation marks carry no shape of their own; their frames
# belong to the phoneme that follows (stress marks precede their vowel).
CARRY = frozenset("ˈˌːʰʲ̃ᵝ→↓↗↘")

NEUTRAL = "sil"

# Span lengths are sums of floats, so a one-frame phoneme can measure
# 24.999999 ms instead of 25. Compare durations with this slack so exactly-
# minimum spans (most consonants) are never mistaken for sub-frame ones.
EPS_S = 1e-6


def viseme_for(phoneme: str) -> str:
    """Mouth shape for a single phoneme ('sil' when it makes no shape)."""
    if phoneme in SILENCE:
        return NEUTRAL
    return PHONEME_TO_VISEME.get(phoneme, NEUTRAL)


def build_timeline(
    phonemes: str,
    pred_dur: Sequence[int],
    vocab: Dict[str, int],
    offset_s: float = 0.0,
    min_duration_s: float = FRAME_MS / 1000.0,
) -> List[Tuple[str, float, float]]:
    """Build [(viseme, start_s, end_s), ...] from phonemes and frame durations.

    `pred_dur` is laid out as [<bos>, one entry per in-vocabulary phoneme, <eos>],
    so we walk the phonemes that Kokoro actually tokenized and pair them up in
    order. Adjacent identical visemes are merged.

    `min_duration_s` defaults to a single 25 ms frame, which keeps every real
    phoneme: many consonants last exactly one frame, and dropping them would
    erase visible articulation such as the /d/ in "made". Sub-frame spans (the
    halves of a one-frame diphthong) are absorbed instead. The renderer eases
    between shapes, so brief spans read as partial movement rather than flicker.
    """
    if pred_dur is None or len(pred_dur) < 3:
        return []

    # Only phonemes present in the vocabulary consume a duration slot.
    tokenized = [p for p in phonemes if vocab.get(p) is not None]
    frames = list(pred_dur)[1:-1]  # drop <bos>/<eos>
    n = min(len(tokenized), len(frames))

    raw: List[Tuple[str, float, float]] = []
    # <bos> frames are leading silence.
    t = offset_s + float(pred_dur[0]) * FRAME_MS / 1000.0
    carried = 0.0  # duration of stress marks waiting for their vowel

    for i in range(n):
        phoneme = tokenized[i]
        dur = float(frames[i]) * FRAME_MS / 1000.0

        if phoneme in CARRY:
            carried += dur
            continue

        dur += carried
        carried = 0.0

        if phoneme in DIPHTHONG_PARTS:
            first, second = DIPHTHONG_PARTS[phoneme]
            half = dur / 2.0
            raw.append((first, t, t + half))
            raw.append((second, t + half, t + dur))
        else:
            raw.append((viseme_for(phoneme), t, t + dur))
        t += dur

    if carried:  # trailing stress mark, keep the timeline contiguous
        raw.append((NEUTRAL, t, t + carried))
        t += carried

    # <eos> frames are trailing silence; including them makes the timeline span
    # exactly as long as the generated audio.
    tail = float(pred_dur[-1]) * FRAME_MS / 1000.0
    if tail > 0:
        raw.append((NEUTRAL, t, t + tail))

    return _merge(raw, min_duration_s)


def _merge(
    spans: List[Tuple[str, float, float]], min_duration_s: float
) -> List[Tuple[str, float, float]]:
    """Merge neighbouring identical visemes, then absorb ultra-short spans."""
    merged: List[List] = []
    for viseme, start, end in spans:
        if merged and merged[-1][0] == viseme:
            merged[-1][2] = end
        else:
            merged.append([viseme, start, end])

    # Absorb spans shorter than min_duration into the previous one, which keeps
    # the mouth from twitching on 25 ms consonants.
    out: List[List] = []
    for span in merged:
        if out and (span[2] - span[1]) < min_duration_s - EPS_S and span[0] != NEUTRAL:
            out[-1][2] = span[2]
        else:
            out.append(span)

    # Re-merge in case absorbing created new neighbours with the same shape.
    final: List[Tuple[str, float, float]] = []
    for viseme, start, end in out:
        if final and final[-1][0] == viseme:
            final[-1] = (viseme, final[-1][1], end)
        else:
            final.append((viseme, start, end))
    return final


def to_wire(timeline: Sequence[Tuple[str, float, float]]) -> List[List]:
    """Compact form for the browser: [[start_ms, viseme], ...].

    The end of each span is the start of the next, so only starts are sent. A
    trailing 'sil' marks the end of the utterance.
    """
    wire: List[List] = [[int(round(start * 1000)), viseme] for viseme, start, _ in timeline]
    if timeline:
        wire.append([int(round(timeline[-1][2] * 1000)), NEUTRAL])
    return wire

/* VirtualTutor frontend -----------------------------------------------------
   Talks to the local Python server: POST /api/start|stop to control the voice
   pipeline, and an SSE stream on /api/events for transcript and state updates.
--------------------------------------------------------------------------- */
"use strict";

const el = {
  chat: document.getElementById("chat"),
  empty: document.getElementById("empty-state"),
  toggle: document.getElementById("toggle"),
  toggleLabel: document.getElementById("toggle-label"),
  statusText: document.getElementById("status-text"),
  statusDot: document.getElementById("status-dot"),
  persona: document.getElementById("persona"),
  error: document.getElementById("error"),
  configOpen: document.getElementById("config-open"),
  configClose: document.getElementById("config-close"),
  configDialog: document.getElementById("config-dialog"),
  configError: document.getElementById("config-error"),
  voiceFields: document.getElementById("voice-fields"),
  duplex: document.getElementById("duplex"),
};

const STATUS_TEXT = {
  idle: "Idle",
  starting: "Starting\u2026",
  listening: "Listening",
  hearing: "Hearing you\u2026",
  thinking: "Thinking\u2026",
  speaking: "Speaking",
};

let running = false;
let busy = false;          // a start/stop request is in flight
let streamingBubble = null; // tutor bubble currently being appended to

/* --- chat ---------------------------------------------------------------- */
function addMessage(role, text) {
  if (el.empty) el.empty.remove();
  const node = document.createElement("div");
  node.className = `msg ${role}`;
  node.textContent = text;
  el.chat.appendChild(node);
  scrollChat();
  return node;
}

function scrollChat() {
  el.chat.scrollTop = el.chat.scrollHeight;
}

function appendTutorText(text) {
  if (!streamingBubble) {
    streamingBubble = addMessage("tutor", "");
    streamingBubble.classList.add("streaming");
  }
  streamingBubble.textContent += text;
  scrollChat();
}

function finishTutorTurn() {
  if (streamingBubble) {
    streamingBubble.classList.remove("streaming");
    // Drop a turn that produced no text at all (e.g. immediate barge-in).
    if (!streamingBubble.textContent.trim()) streamingBubble.remove();
    streamingBubble = null;
  }
}

/* --- status ------------------------------------------------------------- */
function setStatus(state) {
  el.statusText.textContent = STATUS_TEXT[state] || state;
  el.statusDot.className = `dot ${state}`;
  waves.setState(state);
  avatar.setState(state);
  if (state === "idle" || state === "listening") avatar.silence();
}

function showError(message) {
  el.error.textContent = message;
  el.error.hidden = false;
}

function clearError() {
  el.error.hidden = true;
}

/* --- start / stop ------------------------------------------------------- */
function setRunning(next) {
  running = next;
  el.toggle.setAttribute("aria-pressed", String(running));
  el.toggleLabel.textContent = running ? "Stop" : "Start";
  // The persona stays switchable while running: /api/persona rewrites the
  // system prompt in place, so the next reply is in the new personality.
  el.persona.disabled = false;
}

async function post(path, body) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok || data.ok === false) {
    throw new Error(data.error || `Request failed (${res.status})`);
  }
  return data;
}

/* Persona can be changed at any time; while a session is live the server
   rewrites the system prompt so the next reply changes personality. */
el.persona.addEventListener("change", async () => {
  clearError();
  try {
    const { persona_name: name } = await post("/api/persona", { persona: el.persona.value });
    addMessage("system", `Persona: ${name}`);
  } catch (err) {
    showError(err.message);
  }
});

/* --- configuration dialog -----------------------------------------------
   Voice per language and half/full duplex. Both are read by the pipeline as it
   runs -- the voice per synthesized sentence, the duplex mode per mic frame --
   so every change applies at once and there is nothing to save.
------------------------------------------------------------------------- */
const config = (() => {
  const selects = new Map();   // language code -> <select>

  function showConfigError(message) {
    el.configError.textContent = message;
    el.configError.hidden = false;
  }

  function clearConfigError() {
    el.configError.hidden = true;
  }

  /** Build one labelled select per language served by /api/config. */
  function renderVoices(languages) {
    el.voiceFields.textContent = "";
    selects.clear();
    for (const lang of languages) {
      const row = document.createElement("label");
      row.className = "field";

      const name = document.createElement("span");
      name.textContent = lang.label;
      row.appendChild(name);

      const select = document.createElement("select");
      for (const voice of lang.voices) {
        const opt = document.createElement("option");
        opt.value = voice;
        opt.textContent = voice;
        select.appendChild(opt);
      }
      select.value = lang.voice;
      select.addEventListener("change", () => apply({ voices: { [lang.code]: select.value } }));
      row.appendChild(select);

      selects.set(lang.code, select);
      el.voiceFields.appendChild(row);
    }
  }

  /** Reflect server state in the controls, so it is never merely optimistic. */
  function render(state) {
    if (state.languages) {
      if (selects.size === state.languages.length) {
        // Same set of languages: only update values, so an open dropdown and
        // keyboard focus survive a change made in another tab.
        for (const lang of state.languages) {
          const select = selects.get(lang.code);
          if (select) select.value = lang.voice;
        }
      } else {
        renderVoices(state.languages);
      }
    }
    if (state.duplex) el.duplex.checked = state.duplex === "full";
  }

  async function apply(patch) {
    clearConfigError();
    try {
      render(await post("/api/config", patch));
    } catch (err) {
      showConfigError(err.message);
      // Put the controls back to what the server actually has.
      load().catch(() => {});
    }
  }

  async function load() {
    render(await (await fetch("/api/config")).json());
  }

  el.duplex.addEventListener("change", () =>
    apply({ duplex: el.duplex.checked ? "full" : "half" }));

  el.configOpen.addEventListener("click", () => {
    clearConfigError();
    // Re-read on open: the CLI or another tab may have changed something.
    load().catch(() => showConfigError("Cannot reach the VirtualTutor server."));
    if (typeof el.configDialog.showModal === "function") el.configDialog.showModal();
    else el.configDialog.setAttribute("open", "");   // very old Safari
  });

  el.configClose.addEventListener("click", () => el.configDialog.close());

  // Click outside the card closes it, matching the Esc key <dialog> gives us.
  el.configDialog.addEventListener("click", (e) => {
    if (e.target === el.configDialog) el.configDialog.close();
  });

  return {
    load,
    /** Apply a change that came from the server (another tab, or the CLI). */
    setVoice(lang, voice) {
      const select = selects.get(lang);
      if (select && voice) select.value = voice;
    },
    setDuplex(mode) {
      el.duplex.checked = mode === "full";
    },
  };
})();

el.toggle.addEventListener("click", async () => {
  if (busy) return;
  busy = true;
  el.toggle.disabled = true;
  clearError();
  try {
    if (running) {
      await post("/api/stop");
      setRunning(false);
      setStatus("idle");
      finishTutorTurn();
    } else {
      setStatus("starting");
      await post("/api/start", { persona: el.persona.value });
      setRunning(true);
      setStatus("listening");
    }
  } catch (err) {
    showError(err.message);
    setStatus(running ? "listening" : "idle");
  } finally {
    busy = false;
    el.toggle.disabled = false;
  }
});

/* --- server events ----------------------------------------------------- */
function connectEvents() {
  const source = new EventSource("/api/events");

  source.onmessage = (msg) => {
    let event;
    try {
      event = JSON.parse(msg.data);
    } catch {
      return;
    }
    switch (event.type) {
      case "state":
        // Ignore stale states while our own request is settling.
        if (!busy) {
          setStatus(event.state);
          if (event.state === "idle" && running) setRunning(false);
          if (event.state !== "idle" && !running) setRunning(true);
        }
        break;
      case "user":
        finishTutorTurn();
        addMessage("user", event.text);
        break;
      case "assistant_delta":
        appendTutorText(event.text);
        break;
      case "assistant_done":
        finishTutorTurn();
        break;
      case "visemes":
        avatar.speak(event.start_in_ms, event.timeline,
                     { emotion: event.emotion, pace: event.pace });
        break;
      case "interrupted":
        finishTutorTurn();
        avatar.silence();
        break;
      case "level":
        waves.setLevel(event.value);
        break;
      // Config changes are broadcast, so a second tab does not show stale
      // settings after one of them changes a voice or the duplex mode.
      case "voice":
        config.setVoice(event.lang, event.voice);
        break;
      case "duplex":
        config.setDuplex(event.mode);
        break;
      case "error":
        showError(event.text);
        break;
    }
  };

  // EventSource reconnects on its own; surface only a hint via status.
  source.onerror = () => {
    if (!running) setStatus("idle");
  };
}

/* --- animated background ------------------------------------------------
   Layered sine waves whose hue drifts continuously and whose amplitude reacts
   to microphone level and pipeline state. All motion is skipped when the user
   prefers reduced motion.
------------------------------------------------------------------------- */
const waves = (() => {
  const canvas = document.getElementById("waves");
  const ctx = canvas.getContext("2d");
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // Base hue per state, so the background colour reflects what is happening.
  const STATE_HUE = {
    idle: 218, starting: 218, listening: 168,
    hearing: 214, thinking: 38, speaking: 268,
  };

  const layers = [
    { amp: 0.055, freq: 1.1, speed: 0.16, alpha: 0.42, offset: 0.68 },
    { amp: 0.075, freq: 0.8, speed: -0.11, alpha: 0.34, offset: 0.76 },
    { amp: 0.100, freq: 0.6, speed: 0.07, alpha: 0.26, offset: 0.85 },
  ];

  let width = 0, height = 0, dpr = 1;
  let hue = STATE_HUE.idle;
  let targetHue = hue;
  let level = 0, targetLevel = 0;
  let energy = 0.25, targetEnergy = 0.25;
  let start = performance.now();

  function resize() {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    width = canvas.clientWidth;
    height = canvas.clientHeight;
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function lerp(a, b, t) { return a + (b - a) * t; }

  function draw(now) {
    const t = (now - start) / 1000;

    // Ease every animated quantity so colour and shape changes stay smooth.
    hue = lerp(hue, targetHue, 0.012);
    level = lerp(level, targetLevel, 0.10);
    energy = lerp(energy, targetEnergy, 0.02);
    targetLevel *= 0.94; // decay so the waves settle when input goes quiet

    // Vertical wash that shifts with the current hue.
    const bg = ctx.createLinearGradient(0, 0, 0, height);
    bg.addColorStop(0, `hsl(${hue - 16} 62% 96%)`);
    bg.addColorStop(1, `hsl(${hue + 20} 58% 88%)`);
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, width, height);

    const drive = energy + Math.min(level * 7, 1.1);

    layers.forEach((layer, i) => {
      const layerHue = hue + i * 16;
      const fill = ctx.createLinearGradient(0, height * layer.offset, 0, height);
      fill.addColorStop(0, `hsla(${layerHue} 76% 62% / ${layer.alpha})`);
      fill.addColorStop(1, `hsla(${layerHue + 26} 72% 52% / ${layer.alpha * 0.5})`);

      ctx.beginPath();
      ctx.moveTo(0, height);

      const base = height * layer.offset;
      const amp = height * layer.amp * drive;
      const phase = t * layer.speed * Math.PI;

      for (let x = 0; x <= width; x += 6) {
        const p = x / width;
        // Two summed sines per layer give an organic, non-repeating shape.
        const y = base
          + Math.sin(p * Math.PI * 2 * layer.freq + phase) * amp
          + Math.sin(p * Math.PI * 2 * layer.freq * 1.9 - phase * 1.3) * amp * 0.4;
        ctx.lineTo(x, y);
      }

      ctx.lineTo(width, height);
      ctx.closePath();
      ctx.fillStyle = fill;
      ctx.fill();
    });

    if (!reduceMotion) requestAnimationFrame(draw);
  }

  window.addEventListener("resize", resize);
  resize();
  if (reduceMotion) {
    // Draw a single static frame instead of animating.
    draw(performance.now());
  } else {
    requestAnimationFrame(draw);
  }

  return {
    setState(state) {
      targetHue = STATE_HUE[state] ?? STATE_HUE.idle;
      targetEnergy = state === "speaking" ? 1.0
        : state === "hearing" ? 0.85
        : state === "thinking" ? 0.5
        : state === "listening" ? 0.4
        : 0.25;
    },
    setLevel(value) {
      targetLevel = Math.max(targetLevel, Number(value) || 0);
    },
  };
})();

/* --- avatar --------------------------------------------------------------
   Draws a face whose mouth follows the viseme timeline produced by Kokoro.
   The server sends [[start_ms, viseme, level], ...] plus start_in_ms, the delay
   until the audio is actually audible (PortAudio buffer + anything still
   playing), so the mouth lines up with what you hear even though audio plays
   server-side. `level` is how far the mouth should travel toward that shape
   (stress plus the sentence's mood); a two-element entry means 1. The event also
   carries the mood and its speaking rate, which colour the ring and set how
   quickly the mouth moves.

   If web/avatar/manifest.json exists (built by tools/build_photo_avatar.py) a
   photo avatar is used: one base face plus a small mouth patch per viseme,
   feathered in through an elliptical mask. Shipping patches instead of 14 whole
   faces keeps the download at ~170 KB rather than ~2.3 MB, and because
   everything outside the mouth comes from the base image the eyes never blink
   or glance around when the mouth changes. Without the manifest it falls back
   to the vector face, so the UI works with no assets built.
------------------------------------------------------------------------- */
const avatar = (() => {
  const canvas = document.getElementById("avatar");
  const ctx = canvas.getContext("2d");
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // Mouth geometry per viseme, as fractions of head size.
  // w = width, h = opening height, r = roundness (0 = wide slit, 1 = circle).
  const MOUTH = {
    sil: { w: 0.34, h: 0.03, r: 0.50 },
    PP:  { w: 0.30, h: 0.02, r: 0.50 },
    FF:  { w: 0.36, h: 0.07, r: 0.30 },
    TH:  { w: 0.34, h: 0.13, r: 0.30 },
    DD:  { w: 0.36, h: 0.15, r: 0.30 },
    SS:  { w: 0.32, h: 0.08, r: 0.25 },
    CH:  { w: 0.28, h: 0.17, r: 0.60 },
    KK:  { w: 0.36, h: 0.18, r: 0.35 },
    RR:  { w: 0.30, h: 0.15, r: 0.55 },
    aa:  { w: 0.42, h: 0.36, r: 0.40 },
    E:   { w: 0.40, h: 0.22, r: 0.35 },
    ih:  { w: 0.40, h: 0.14, r: 0.30 },
    oh:  { w: 0.30, h: 0.30, r: 0.75 },
    ou:  { w: 0.24, h: 0.22, r: 0.85 },
  };

  let timeline = [];      // [[start_ms, viseme], ...]
  let startAt = 0;        // performance.now() when timeline position 0 is heard
  let cursor = 0;         // index into timeline, advanced monotonically
  let shape = { ...MOUTH.sil };
  let state = "idle";
  let blinkAt = performance.now() + 2500;
  let blink = 0;          // 0 = open, 1 = closed
  let size = 168, dpr = 1;

  // --- photo mode ---------------------------------------------------------
  /* Mouth motion.

     Kokoro's timeline is phoneme-accurate, which means it contains plenty of
     25 ms spans. Snapping to each one -- or restarting a crossfade on every
     change -- looks like flicker, because a fade longer than the span never
     finishes before the next one begins.

     So the mouth has inertia instead: every viseme keeps a weight that rises
     toward 1 while it is the target and decays afterwards, and the rendered
     mouth is the weighted blend of the few strongest. A 25 ms consonant only
     gets part of the way to its shape before decaying, which is what real
     articulation does -- the lips never fully reach the target of a fleeting
     stop between two vowels. Opening is quicker than closing, as in speech.

     Time constants are in milliseconds and applied against real frame deltas,
     so motion is identical on 60 Hz and 120 Hz displays. */
  const ATTACK_MS = 55;      // toward the current viseme
  const RELEASE_MS = 95;     // away from previous ones
  const MAX_LAYERS = 3;      // strongest shapes blended per frame
  const WEIGHT_FLOOR = 0.02; // below this a shape is dropped from the blend

  /* Mood.

     The server sends the mood each sentence was spoken in, plus `pace` (its
     speaking-rate multiplier) and a per-span articulation level. The level is
     applied as openness rather than as weight: weights are normalised before
     rendering, so scaling them all would cancel out, whereas biasing the blend
     between the mouth shapes and silence really does open or close the mouth.

     Pace scales the transition times, so a fast, excited sentence gets snappier
     articulation and a slow, thoughtful one gets softer movement -- the same
     relationship speech has between rate and coarticulation. */
  const MOOD_HUE = {
    neutral: 268, warm: 28, excited: 330, amused: 300, curious: 196,
    gentle: 158, thoughtful: 236, proud: 46, deadpan: 250,
  };

  // Brow raise and inward slant per mood, as fractions of head size. Only the
  // drawn face uses these; the photo avatar's face comes from the video, so its
  // share of the mood is the ring colour and how wide the mouth articulates.
  const MOOD_BROW = {
    neutral: { raise: 0.00, tilt: 0.00 },
    warm: { raise: 0.02, tilt: 0.02 },
    excited: { raise: 0.09, tilt: 0.04 },
    amused: { raise: 0.06, tilt: -0.05 },
    curious: { raise: 0.07, tilt: 0.08 },
    gentle: { raise: -0.02, tilt: 0.06 },
    thoughtful: { raise: -0.04, tilt: -0.06 },
    proud: { raise: 0.05, tilt: 0.00 },
    deadpan: { raise: -0.05, tilt: -0.02 },
  };

  let photo = null;          // { base, patches, mask, mix, manifest } once loaded
  let weights = { sil: 1 };  // viseme -> 0..1 contribution to the current mouth
  let lastFrameAt = performance.now();
  let mood = "neutral";      // mood of the sentence being spoken
  let pace = 1;              // its speaking-rate multiplier
  let level = 1;             // articulation of the span being spoken
  let shownLevel = 1;        // eased, so a mood change does not jump the mouth
  let brow = { raise: 0, tilt: 0 };   // eased brow pose of the drawn face

  /** Alpha mask matching build_photo_avatar.py's mouth_mask, in patch space. */
  function buildMask(manifest) {
    const p = manifest.patch, m = manifest.mouth;
    const mask = document.createElement("canvas");
    mask.width = p.w;
    mask.height = p.h;
    const c = mask.getContext("2d");
    const alpha = c.createImageData(p.w, p.h);
    const ramp = m.feather / Math.min(m.rx, m.ry);
    for (let y = 0; y < p.h; y++) {
      for (let x = 0; x < p.w; x++) {
        const nx = (x + p.x - m.cx) / m.rx;
        const ny = (y + p.y - m.cy) / m.ry;
        const d = Math.sqrt(nx * nx + ny * ny);
        const t = Math.min(1, Math.max(0, (1 + ramp - d) / ramp));
        alpha.data[(y * p.w + x) * 4 + 3] = Math.round(255 * t * t * (3 - 2 * t));
      }
    }
    c.putImageData(alpha, 0, 0);
    return mask;
  }

  function loadImage(src) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error(`cannot load ${src}`));
      img.src = src;
    });
  }

  /** Load the photo sprite set; silently stay on the vector face if absent. */
  async function loadPhoto() {
    let manifest;
    try {
      const res = await fetch("avatar/manifest.json", { cache: "no-store" });
      if (!res.ok) return;
      manifest = await res.json();
      if (manifest.mode !== "patch") return;
    } catch {
      return;
    }
    try {
      const base = await loadImage(`avatar/${manifest.base}`);
      const patches = {};
      await Promise.all(Object.entries(manifest.images).map(async ([viseme, file]) => {
        patches[viseme] = await loadImage(`avatar/${file}`);
      }));
      // Patches stay unmasked so several can be averaged first; the mask is
      // applied once to the blend, which keeps the result identical to the
      // single-patch composite the builder produced.
      const mix = document.createElement("canvas");
      mix.width = manifest.patch.w;
      mix.height = manifest.patch.h;
      photo = { base, patches, mask: buildMask(manifest), mix, manifest };
      canvas.removeAttribute("aria-hidden");
      canvas.setAttribute("role", "img");
      canvas.setAttribute("aria-label", "Tutor avatar");
    } catch (e) {
      photo = null;      // a missing sprite must not break the whole UI
    }
  }

  function resize() {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    size = canvas.clientWidth || 168;
    canvas.width = Math.round(size * dpr);
    canvas.height = Math.round(size * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function lerp(a, b, t) { return a + (b - a) * t; }

  /** Viseme that should be visible at `now`, following the timeline.
      Also latches the articulation level of that span (default 1). */
  function visemeAt(now) {
    if (!timeline.length) return "sil";
    const t = now - startAt;
    if (t < 0) return "sil";                            // audio not audible yet
    // Advance the cursor to the span covering t (timeline is sorted).
    while (cursor + 1 < timeline.length && timeline[cursor + 1][0] <= t) cursor++;
    while (cursor > 0 && timeline[cursor][0] > t) cursor--;
    if (t > timeline[timeline.length - 1][0]) {          // utterance finished
      timeline = [];
      level = 1;
      return "sil";
    }
    const span = timeline[cursor];
    level = Number.isFinite(span[2]) ? span[2] : 1;
    return span[1] || "sil";
  }

  /** Advance each viseme's weight toward its target for a `dt` ms frame. */
  function updateWeights(target, dt) {
    // Faster speech moves the lips faster; `pace` is the sentence's rate.
    const attack = reduceMotion ? 1 : 1 - Math.exp(-dt * pace / ATTACK_MS);
    const decay = reduceMotion ? 0 : Math.exp(-dt * pace / RELEASE_MS);
    weights[target] = (weights[target] || 0) + (1 - (weights[target] || 0)) * attack;
    for (const v in weights) {
      if (v === target) continue;
      const w = weights[v] * decay;
      if (w < WEIGHT_FLOOR) delete weights[v];
      else weights[v] = w;
    }
    // Ease the articulation level so it never steps between two spans.
    const k = reduceMotion ? 1 : 1 - Math.exp(-dt / 70);
    shownLevel = lerp(shownLevel, timeline.length ? level : 1, k);
    return weights;
  }

  /** The strongest shapes, normalised so their weights sum to 1. */
  function blendLayers(available) {
    const layers = Object.entries(weights)
      .filter(([v]) => !available || available[v])
      .sort((a, b) => b[1] - a[1])
      .slice(0, MAX_LAYERS);
    const total = layers.reduce((sum, [, w]) => sum + w, 0);
    if (!total) return [];
    return layers.map(([v, w]) => [v, w / total]);
  }

  /** Bias a blend toward a wider or a more closed mouth.

      This is how the articulation level reaches the photo avatar, which cannot
      be stretched geometrically: an over-articulated sentence mixes in a little
      of the wide-open shape, a mumbled one a little of the closed one. Scaling
      the weights themselves would do nothing, since they are normalised. */
  function articulate(layers, available) {
    const bias = shownLevel - 1;
    if (Math.abs(bias) < 0.02 || !layers.length) return layers;
    const extra = bias > 0 ? "aa" : "sil";
    if (available && !available[extra]) return layers;
    const share = Math.min(0.35, Math.abs(bias));
    const out = layers.map(([v, w]) => [v, w * (1 - share)]);
    const existing = out.find(([v]) => v === extra);
    if (existing) existing[1] += share;
    else out.push([extra, share]);
    return out;
  }

  /** Photo avatar: base face + a blend of the strongest mouth patches. */
  function drawPhoto(now, dt) {
    updateWeights(visemeAt(now), dt);
    const layers = articulate(blendLayers(photo.patches), photo.patches);
    const p = photo.manifest.patch;
    const k = size / photo.manifest.size;        // exported px -> CSS px

    // Average the unmasked patches, then mask once. Each draw contributes its
    // share of the running total, which makes the result an exact weighted
    // mean rather than a stack of partial fades.
    const mc = photo.mix.getContext("2d");
    mc.globalCompositeOperation = "source-over";
    mc.clearRect(0, 0, p.w, p.h);
    let acc = 0;
    for (const [viseme, weight] of layers) {
      acc += weight;
      mc.globalAlpha = weight / acc;
      mc.drawImage(photo.patches[viseme], 0, 0, p.w, p.h);
    }
    mc.globalAlpha = 1;
    mc.globalCompositeOperation = "destination-in";
    mc.drawImage(photo.mask, 0, 0);
    mc.globalCompositeOperation = "source-over";

    ctx.clearRect(0, 0, size, size);
    ctx.drawImage(photo.base, 0, 0, size, size);
    if (layers.length) {
      ctx.drawImage(photo.mix, p.x * k, p.y * k, p.w * k, p.h * k);
    }

    // Thin ring tinted by pipeline state: the only chrome the photo needs.
    const hue = stateHue();
    ctx.strokeStyle = `hsl(${hue} 78% 62% / ${state === "idle" ? 0.35 : 0.75})`;
    ctx.lineWidth = Math.max(2, size * 0.018);
    const inset = ctx.lineWidth / 2;
    const radius = size * 0.14;
    ctx.beginPath();
    if (ctx.roundRect) {
      ctx.roundRect(inset, inset, size - 2 * inset, size - 2 * inset, radius);
    } else {
      ctx.rect(inset, inset, size - 2 * inset, size - 2 * inset);   // older Safari
    }
    ctx.stroke();
  }

  function stateHue() {
    // While speaking, the ring takes the mood's colour instead of the generic
    // "speaking" purple, so the mood is visible and not only audible.
    if (state === "speaking") return MOOD_HUE[mood] ?? 268;
    return state === "thinking" ? 38
      : state === "hearing" ? 214 : state === "listening" ? 168 : 218;
  }

  function draw(now) {
    // Real elapsed time, so motion is the same on 60 Hz and 120 Hz displays and
    // survives a tab being throttled in the background.
    const dt = Math.min(120, Math.max(1, now - lastFrameAt));
    lastFrameAt = now;

    if (photo) {
      drawPhoto(now, dt);
      requestAnimationFrame(draw);
      return;
    }
    // The drawn mouth blends the same weights the photo avatar uses, so a
    // fleeting consonant only nudges the shape instead of snapping to it.
    updateWeights(visemeAt(now), dt);
    const tgt = { w: 0, h: 0, r: 0 };
    for (const [viseme, weight] of blendLayers(MOUTH)) {
      const m = MOUTH[viseme];
      tgt.w += m.w * weight;
      tgt.h += m.h * weight;
      tgt.r += m.r * weight;
    }
    // Articulation: the drawn face can simply open wider, where the photo
    // avatar has to mix in another mouth patch. Width moves less than height,
    // as it does in speech.
    const open = Math.max(0.5, Math.min(1.6, shownLevel));
    tgt.h *= open;
    tgt.w *= 1 + (open - 1) * 0.35;
    // A little extra easing on top, time-based rather than per-frame.
    const k = reduceMotion ? 1 : 1 - Math.exp(-dt / 45);
    shape.w = lerp(shape.w, tgt.w, k);
    shape.h = lerp(shape.h, tgt.h, k);
    shape.r = lerp(shape.r, tgt.r, k);

    // Blink on an irregular schedule; suppressed under reduced motion.
    if (!reduceMotion) {
      const step = dt / 60;                    // ~60 ms to close or open
      if (now > blinkAt) {
        blink = Math.min(1, blink + step);
        if (blink >= 1) { blinkAt = now + 1800 + Math.random() * 3500; }
      } else {
        blink = Math.max(0, blink - step);
      }
    }

    const c = size / 2;
    // Gentle breathing bob while active.
    const bob = reduceMotion || state === "idle" ? 0 : Math.sin(now / 900) * size * 0.008;
    ctx.clearRect(0, 0, size, size);

    // Head.
    const head = size * 0.40;
    const g = ctx.createLinearGradient(0, c - head, 0, c + head);
    const hue = stateHue();
    g.addColorStop(0, `hsl(${hue} 72% 74%)`);
    g.addColorStop(1, `hsl(${hue + 18} 66% 58%)`);
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.ellipse(c, c + bob, head * 0.86, head, 0, 0, Math.PI * 2);
    ctx.fill();

    // Eyes.
    const eyeY = c + bob - head * 0.18;
    const eyeDx = head * 0.34;
    const eyeR = head * 0.10;
    ctx.fillStyle = "#12203a";
    for (const dx of [-eyeDx, eyeDx]) {
      ctx.beginPath();
      ctx.ellipse(c + dx, eyeY, eyeR, eyeR * (1 - 0.92 * blink), 0, 0, Math.PI * 2);
      ctx.fill();
    }

    // Brows: the drawn face's share of the mood. Raised and slanted inward for
    // excitement or curiosity, lowered and flattened for deadpan -- eased, so
    // the expression settles into a sentence rather than snapping per span.
    const target = MOOD_BROW[mood] || MOOD_BROW.neutral;
    const bk = reduceMotion ? 1 : 1 - Math.exp(-dt / 220);
    brow.raise = lerp(brow.raise, target.raise, bk);
    brow.tilt = lerp(brow.tilt, target.tilt, bk);
    ctx.strokeStyle = "#12203a";
    ctx.lineWidth = Math.max(1.5, head * 0.055);
    ctx.lineCap = "round";
    for (const side of [-1, 1]) {
      const bx = c + side * eyeDx;
      const by = eyeY - head * (0.24 + brow.raise);
      const half = head * 0.16;
      ctx.beginPath();
      ctx.moveTo(bx - half, by + head * brow.tilt * side);
      ctx.lineTo(bx + half, by - head * brow.tilt * side);
      ctx.stroke();
    }

    // Mouth: an ellipse whose height and roundness come from the viseme.
    const mw = head * shape.w;
    const mh = Math.max(size * 0.006, head * shape.h);
    const my = c + bob + head * 0.34;
    ctx.fillStyle = "#2a1420";
    ctx.beginPath();
    ctx.ellipse(c, my, mw / 2, mh / 2 + mw * 0.04 * shape.r, 0, 0, Math.PI * 2);
    ctx.fill();
    // Inner highlight suggests depth when the mouth is open.
    if (shape.h > 0.10) {
      ctx.fillStyle = "rgba(255,255,255,0.14)";
      ctx.beginPath();
      ctx.ellipse(c, my + mh * 0.16, mw * 0.30, mh * 0.22, 0, 0, Math.PI * 2);
      ctx.fill();
    }

    requestAnimationFrame(draw);
  }

  window.addEventListener("resize", resize);
  resize();
  loadPhoto();
  requestAnimationFrame(draw);

  return {
    setState(next) { state = next; },
    /** Start a sentence. `info` carries its mood and speaking rate. */
    speak(startInMs, wire, info) {
      timeline = Array.isArray(wire) ? wire : [];
      cursor = 0;
      startAt = performance.now() + (Number(startInMs) || 0);
      mood = (info && MOOD_BROW[info.emotion]) ? info.emotion : "neutral";
      pace = Math.max(0.6, Math.min(1.6, Number(info && info.pace) || 1));
    },
    silence() {
      timeline = [];
      cursor = 0;
      level = 1;
      pace = 1;
      mood = "neutral";
    },
  };
})();

/* --- boot -------------------------------------------------------------- */
async function init() {
  try {
    const { personas, default: fallback } = await (await fetch("/api/personas")).json();
    for (const p of personas) {
      const opt = document.createElement("option");
      opt.value = p.key;
      opt.textContent = p.name;
      opt.title = p.blurb;
      el.persona.appendChild(opt);
    }
    // Show the persona actually in force, not the config default, so a reload
    // mid-session does not misreport who the tutor currently is.
    const { running: live, persona } = await (await fetch("/api/status")).json();
    el.persona.value = persona || fallback;
    setRunning(live);
    setStatus(live ? "listening" : "idle");
    await config.load();
  } catch {
    showError("Cannot reach the VirtualTutor server.");
  }
  connectEvents();
}

init();

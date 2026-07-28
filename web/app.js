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
  el.persona.disabled = running;
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
      case "interrupted":
        finishTutorTurn();
        break;
      case "level":
        waves.setLevel(event.value);
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

/* --- boot -------------------------------------------------------------- */
async function init() {
  try {
    const { personas, default: fallback } = await (await fetch("/api/personas")).json();
    for (const p of personas) {
      const opt = document.createElement("option");
      opt.value = p.key;
      opt.textContent = p.name;
      opt.title = p.blurb;
      if (p.key === fallback) opt.selected = true;
      el.persona.appendChild(opt);
    }
    const { running: live } = await (await fetch("/api/status")).json();
    setRunning(live);
    setStatus(live ? "listening" : "idle");
  } catch {
    showError("Cannot reach the VirtualTutor server.");
  }
  connectEvents();
}

init();

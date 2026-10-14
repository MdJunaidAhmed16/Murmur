const $ = (id) => document.getElementById(id);
const STAGES = ["endpoint", "stt_final", "memory_recall", "llm_first_token", "llm_first_sentence", "tts_first_audio", "time_to_first_audio", "barge_in"];

let ws, captureCtx, playbackCtx, playbackNode, micStream;
let assistantLine = null;
let audioPlaying = false;

// A random id per browser: in demo mode it selects this visitor's own memory store.
const VISITOR = (() => {
  try {
    let id = localStorage.getItem("murmur-visitor");
    if (!id) {
      const bytes = crypto.getRandomValues(new Uint8Array(16));
      id = [...bytes].map((b) => b.toString(16).padStart(2, "0")).join("");
      localStorage.setItem("murmur-visitor", id);
    }
    return id;
  } catch {
    // Storage blocked (private window): a fresh id, so memory lasts for this page only.
    return [...crypto.getRandomValues(new Uint8Array(16))].map((b) => b.toString(16).padStart(2, "0")).join("");
  }
})();

async function refreshMemories() {
  try {
    const res = await fetch(`/api/memories?visitor=${VISITOR}`);
    if (!res.ok) return;
    const { memories } = await res.json();
    const list = $("memories");
    list.replaceChildren();
    if (!memories.length) {
      const li = document.createElement("li");
      li.className = "empty";
      li.textContent = "Nothing yet. Tell it something about yourself.";
      list.appendChild(li);
      return;
    }
    for (const m of memories) {
      const li = document.createElement("li");
      if (m.replaced_by) li.className = "old";
      li.textContent = m.text;
      const since = document.createElement("span");
      since.className = "since";
      since.textContent = m.replaced_by ? `replaced by: ${m.replaced_by}` : m.since;
      li.appendChild(since);
      list.appendChild(li);
    }
  } catch {
    // The panel is a view; a failed refresh leaves the last one up.
  }
}

function setStatus(text, state) {
  $("status").textContent = text;
  $("status").dataset.state = state;
}

function addLine(who, text) {
  const div = document.createElement("div");
  div.className = `line ${who}`;
  div.innerHTML = `<span class="who"></span><span class="text"></span>`;
  div.querySelector(".who").textContent = { user: "You", assistant: "Murmur", memory: "Recalled" }[who];
  div.querySelector(".text").textContent = text;
  $("transcript").appendChild(div);
  div.scrollIntoView({ block: "end" });
  return div.querySelector(".text");
}

function showMetrics(record) {
  const row = document.createElement("tr");
  for (const stage of STAGES) {
    const td = document.createElement("td");
    const v = record.ms[stage];
    td.textContent = v == null ? "—" : `${v}`;
    row.appendChild(td);
  }
  const td = document.createElement("td");
  td.textContent = record.barge_in ? "yes" : "";
  row.appendChild(td);
  $("metrics-body").prepend(row);
}

function onEvent(msg) {
  switch (msg.type) {
    case "config":
      $("backend").textContent = `· ${msg.backend} model`;
      $("demo-note").hidden = !msg.demo_mode;
      return startAudio(msg);
    case "speech_start":
      $("partial").textContent = "…";
      return;
    case "partial":
      $("partial").textContent = msg.text;
      return;
    case "user":
      $("partial").textContent = "";
      addLine("user", msg.text);
      assistantLine = null;
      return;
    case "recalled":
      addLine("memory", msg.memories.join(" · "));
      return;
    case "assistant_delta":
      if (!assistantLine) assistantLine = addLine("assistant", "");
      assistantLine.textContent += msg.text;
      return;
    case "pause":
    case "resume":
      playbackNode?.port.postMessage(msg.type);
      return;
    case "clear":
      playbackNode?.port.postMessage("clear");
      // The server clears defensively; only a reply still playing was actually cut off.
      if (assistantLine && audioPlaying) assistantLine.parentElement.classList.add("cut");
      assistantLine = null;
      return;
    case "metrics":
      showMetrics(msg);
      // Facts are written in the background after the turn; give the worker a moment.
      setTimeout(refreshMemories, 2500);
      return;
    case "error":
      return setStatus(`Error: ${msg.message}`, "error");
  }
}

async function startAudio(config) {
  micStream = await navigator.mediaDevices.getUserMedia({
    // Echo cancellation keeps the agent from hearing itself and barging in on its own voice.
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
  });

  captureCtx = new AudioContext({ sampleRate: config.input_sample_rate });
  await captureCtx.audioWorklet.addModule("/static/worklets.js");
  const capture = new AudioWorkletNode(captureCtx, "capture", {
    processorOptions: { frameSamples: config.frame_samples },
  });
  capture.port.onmessage = (e) => {
    if (ws.readyState === WebSocket.OPEN) ws.send(e.data);
  };
  captureCtx.createMediaStreamSource(micStream).connect(capture);

  playbackCtx = new AudioContext({ sampleRate: config.output_sample_rate });
  await playbackCtx.audioWorklet.addModule("/static/worklets.js");
  playbackNode = new AudioWorkletNode(playbackCtx, "playback", { outputChannelCount: [1] });
  playbackNode.port.onmessage = (e) => {
    if (ws.readyState !== WebSocket.OPEN) return;
    if ("cleared" in e.data) {
      ws.send(JSON.stringify({ type: "playback", discarded_samples: e.data.cleared }));
      return;
    }
    audioPlaying = e.data.playing;
    setStatus(e.data.playing ? "Speaking" : "Listening", "live");
  };
  playbackNode.connect(playbackCtx.destination);

  setStatus("Listening", "live");
}

function playPcm(buffer) {
  const pcm = new Int16Array(buffer);
  const floats = new Float32Array(pcm.length);
  for (let i = 0; i < pcm.length; i++) floats[i] = pcm[i] / 32768;
  playbackNode?.port.postMessage(floats, [floats.buffer]);
}

function start() {
  setStatus("Connecting", "idle");
  $("start").disabled = true;
  ws = new WebSocket(
    `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws?visitor=${VISITOR}`,
  );
  ws.binaryType = "arraybuffer";
  ws.onmessage = (e) => (typeof e.data === "string" ? onEvent(JSON.parse(e.data)) : playPcm(e.data));
  ws.onclose = (e) => {
    stop();
    setStatus(e.reason ? `Disconnected: ${e.reason}` : "Disconnected", "idle");
  };
  $("stop").disabled = false;
}

function stop() {
  micStream?.getTracks().forEach((t) => t.stop());
  captureCtx?.close();
  playbackCtx?.close();
  captureCtx = playbackCtx = playbackNode = micStream = null;
  if (ws && ws.readyState <= WebSocket.OPEN) ws.close();
  $("start").disabled = false;
  $("stop").disabled = true;
}

$("start").onclick = start;
$("stop").onclick = stop;
refreshMemories();

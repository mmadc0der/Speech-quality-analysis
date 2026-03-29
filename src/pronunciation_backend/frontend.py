from __future__ import annotations

import argparse
import json
import mimetypes
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, Response

BACKEND_URL_ENV = "PRONUNCIATION_FRONTEND_BACKEND_URL"
HOST_ENV = "PRONUNCIATION_FRONTEND_HOST"
PORT_ENV = "PRONUNCIATION_FRONTEND_PORT"

INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Pronunciation Scorer Debug UI</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #10151f;
      --panel: #192132;
      --panel-2: #212c40;
      --text: #e8eefc;
      --muted: #94a7c6;
      --accent: #7dd3fc;
      --good: #86efac;
      --warn: #fcd34d;
      --bad: #fca5a5;
      --border: #31405e;
      --shadow: 0 16px 48px rgba(0, 0, 0, 0.28);
      --radius: 16px;
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
      background:
        radial-gradient(circle at top left, rgba(125, 211, 252, 0.12), transparent 32%),
        linear-gradient(180deg, #0d131d, #131b29 55%, #0f1622);
      color: var(--text);
      min-height: 100vh;
    }

    main {
      width: min(1120px, calc(100% - 32px));
      margin: 32px auto 48px;
      display: grid;
      gap: 16px;
    }

    .hero, .panel {
      background: rgba(25, 33, 50, 0.92);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      backdrop-filter: blur(10px);
    }

    .hero {
      padding: 24px;
    }

    .hero h1 {
      margin: 0 0 8px;
      font-size: 1.8rem;
    }

    .hero p {
      margin: 0;
      color: var(--muted);
      max-width: 70ch;
      line-height: 1.45;
    }

    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 16px;
    }

    .panel {
      padding: 18px;
    }

    .panel h2 {
      margin: 0 0 12px;
      font-size: 1.05rem;
    }

    .row {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      margin-bottom: 12px;
    }

    .stack {
      display: grid;
      gap: 12px;
    }

    label {
      display: grid;
      gap: 6px;
      width: 100%;
      font-size: 0.95rem;
      color: var(--muted);
    }

    input[type="text"], input[type="number"], input[type="file"] {
      width: 100%;
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 12px 14px;
      background: var(--panel-2);
      color: var(--text);
      outline: none;
    }

    input[type="checkbox"] {
      width: 16px;
      height: 16px;
      accent-color: #38bdf8;
    }

    button {
      border: 1px solid transparent;
      border-radius: 12px;
      padding: 11px 14px;
      background: #2a3c58;
      color: var(--text);
      cursor: pointer;
      transition: transform 120ms ease, opacity 120ms ease, border-color 120ms ease;
      font-weight: 600;
    }

    button:hover:not(:disabled) {
      transform: translateY(-1px);
      border-color: rgba(125, 211, 252, 0.4);
    }

    button:disabled {
      cursor: not-allowed;
      opacity: 0.6;
    }

    button.primary {
      background: linear-gradient(135deg, #2563eb, #0891b2);
    }

    button.warn {
      background: linear-gradient(135deg, #b45309, #92400e);
    }

    button.danger {
      background: linear-gradient(135deg, #b91c1c, #991b1b);
    }

    .pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      border-radius: 999px;
      padding: 8px 12px;
      background: rgba(33, 44, 64, 0.95);
      border: 1px solid var(--border);
      color: var(--muted);
      font-size: 0.92rem;
    }

    .mono {
      font-family: Consolas, "Courier New", monospace;
    }

    .status-ok {
      color: var(--good);
    }

    .status-warn {
      color: var(--warn);
    }

    .status-bad {
      color: var(--bad);
    }

    .summary {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 12px;
    }

    .metric {
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 12px;
      background: rgba(33, 44, 64, 0.75);
    }

    .metric .label {
      display: block;
      color: var(--muted);
      font-size: 0.85rem;
      margin-bottom: 4px;
    }

    .metric .value {
      font-size: 1.35rem;
      font-weight: 700;
    }

    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.94rem;
    }

    th, td {
      text-align: left;
      padding: 10px 8px;
      border-bottom: 1px solid rgba(49, 64, 94, 0.8);
      vertical-align: top;
    }

    th {
      color: var(--muted);
      font-weight: 600;
    }

    pre {
      margin: 0;
      padding: 14px;
      border-radius: 14px;
      background: #0c1220;
      border: 1px solid var(--border);
      overflow: auto;
      max-height: 360px;
      white-space: pre-wrap;
      word-break: break-word;
    }

    audio {
      width: 100%;
    }

    .hidden {
      display: none;
    }

    .footer-note {
      color: var(--muted);
      font-size: 0.9rem;
      line-height: 1.45;
    }
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>Pronunciation Scorer Debug UI</h1>
      <p>Run this page on your workstation, point it at a remote backend, record a word, and inspect the phoneme-level output without browser CORS issues.</p>
    </section>

    <section class="grid">
      <div class="panel stack">
        <h2>Backend</h2>
        <div class="pill"><strong>Proxy target:</strong> <span id="backendUrl" class="mono"></span></div>
        <div class="row">
          <button id="pingButton">Ping backend</button>
          <button id="loadWordsButton">Load supported words</button>
        </div>
        <div id="backendStatus" class="footer-note">Backend status is not checked yet.</div>
      </div>

      <div class="panel stack">
        <h2>Target Word</h2>
        <label>
          Word
          <input id="wordInput" type="text" list="wordOptions" placeholder="Enter a supported word, e.g. thought">
        </label>
        <datalist id="wordOptions"></datalist>
        <div id="wordStatus" class="footer-note">Load the word list or type a word directly.</div>
      </div>
    </section>

    <section class="grid">
      <div class="panel stack">
        <h2>Recording</h2>
        <div class="row">
          <button id="recordButton" class="primary">Start recording</button>
          <button id="stopButton" class="danger" disabled>Stop</button>
          <button id="clearButton">Clear audio</button>
        </div>
        <div class="pill"><strong>Recorder:</strong> <span id="recordingState">idle</span></div>
        <div class="pill"><strong>Timer:</strong> <span id="recordingTimer" class="mono">0.0s</span></div>
        <audio id="audioPreview" controls class="hidden"></audio>
        <label>
          Or choose an existing audio file
          <input id="fileInput" type="file" accept="audio/*">
        </label>
        <label>
          <span class="row">
            <input id="frontendTrimEnabled" type="checkbox">
            <span>Apply manual frontend trim before upload</span>
          </span>
        </label>
        <div class="row">
          <label>
            Trim start (ms)
            <input id="trimStartMs" type="number" min="0" step="10" value="0">
          </label>
          <label>
            Trim end (ms, optional)
            <input id="trimEndMs" type="number" min="0" step="10" placeholder="to clip end">
          </label>
        </div>
        <div id="audioStatus" class="footer-note">Recording is captured as WAV in the browser for backend compatibility. You can optionally trim the clip locally before upload.</div>
      </div>

      <div class="panel stack">
        <h2>Score Request</h2>
        <label>
          Optional speaker id
          <input id="speakerInput" type="text" placeholder="leave empty for now">
        </label>
        <label>
          <span class="row">
            <input id="noTrimCheckbox" type="checkbox">
            <span>Skip backend auto-trim and trust uploaded audio (`noTrim`)</span>
          </span>
        </label>
        <div class="row">
          <button id="scoreButton" class="primary">Submit for scoring</button>
        </div>
        <div id="requestStatus" class="footer-note">No request sent yet.</div>
      </div>
    </section>

    <section class="panel stack">
      <h2>Result Summary</h2>
      <div id="summaryGrid" class="summary">
        <div class="metric"><span class="label">Overall score</span><span class="value">-</span></div>
        <div class="metric"><span class="label">Confidence</span><span class="value">-</span></div>
        <div class="metric"><span class="label">Primary issue</span><span class="value">-</span></div>
        <div class="metric"><span class="label">Audio quality</span><span class="value">-</span></div>
        <div class="metric"><span class="label">Trim window</span><span class="value">-</span></div>
      </div>
    </section>

    <section class="panel stack">
      <h2>Per-Phone Detail</h2>
      <table>
        <thead>
          <tr>
            <th>Phone</th>
            <th>Time</th>
            <th>Score</th>
            <th>Class</th>
            <th>Omission</th>
            <th>Confidence</th>
            <th>Class probabilities</th>
          </tr>
        </thead>
        <tbody id="phoneRows">
          <tr><td colspan="7" class="footer-note">No result yet.</td></tr>
        </tbody>
      </table>
    </section>

    <section class="panel stack">
      <h2>Raw Response</h2>
      <pre id="rawJson">No response yet.</pre>
    </section>
  </main>

  <script>
    const state = {
      mediaStream: null,
      audioContext: null,
      sourceNode: null,
      processorNode: null,
      chunks: [],
      timerHandle: null,
      recordingStartedAt: 0,
      recordingBlob: null,
      recordingFileName: "recording.wav",
    };

    const backendUrlEl = document.getElementById("backendUrl");
    const backendStatusEl = document.getElementById("backendStatus");
    const wordInputEl = document.getElementById("wordInput");
    const wordOptionsEl = document.getElementById("wordOptions");
    const wordStatusEl = document.getElementById("wordStatus");
    const recordButtonEl = document.getElementById("recordButton");
    const stopButtonEl = document.getElementById("stopButton");
    const clearButtonEl = document.getElementById("clearButton");
    const recordingStateEl = document.getElementById("recordingState");
    const recordingTimerEl = document.getElementById("recordingTimer");
    const audioPreviewEl = document.getElementById("audioPreview");
    const fileInputEl = document.getElementById("fileInput");
    const frontendTrimEnabledEl = document.getElementById("frontendTrimEnabled");
    const trimStartMsEl = document.getElementById("trimStartMs");
    const trimEndMsEl = document.getElementById("trimEndMs");
    const audioStatusEl = document.getElementById("audioStatus");
    const speakerInputEl = document.getElementById("speakerInput");
    const noTrimCheckboxEl = document.getElementById("noTrimCheckbox");
    const scoreButtonEl = document.getElementById("scoreButton");
    const requestStatusEl = document.getElementById("requestStatus");
    const summaryGridEl = document.getElementById("summaryGrid");
    const phoneRowsEl = document.getElementById("phoneRows");
    const rawJsonEl = document.getElementById("rawJson");

    function setText(element, text, className = "") {
      element.textContent = text;
      element.className = className;
    }

    function formatPercent(value) {
      return `${(Number(value) * 100).toFixed(1)}%`;
    }

    function formatScore(value) {
      return Number(value).toFixed(1);
    }

    function escapeHtml(text) {
      return String(text)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;");
    }

    async function fetchJson(url, options = {}) {
      const response = await fetch(url, options);
      const contentType = response.headers.get("content-type") || "";
      const payload = contentType.includes("application/json")
        ? await response.json()
        : { detail: await response.text() };
      if (!response.ok) {
        const message = payload.detail || payload.message || JSON.stringify(payload);
        throw new Error(message);
      }
      return payload;
    }

    async function loadConfig() {
      const payload = await fetchJson("/api/config");
      backendUrlEl.textContent = payload.backend_base_url;
    }

    async function pingBackend() {
      backendStatusEl.textContent = "Checking backend...";
      backendStatusEl.className = "footer-note";
      try {
        const payload = await fetchJson("/api/health");
        backendStatusEl.textContent =
          `status=${payload.status}, model_ready=${payload.model_ready}, backend=${payload.runtime_backend}, device=${payload.device}`;
        backendStatusEl.className = "footer-note status-ok";
      } catch (error) {
        backendStatusEl.textContent = `Backend check failed: ${error.message}`;
        backendStatusEl.className = "footer-note status-bad";
      }
    }

    async function loadWords() {
      wordStatusEl.textContent = "Loading supported words...";
      wordStatusEl.className = "footer-note";
      try {
        const payload = await fetchJson("/api/words");
        const words = Array.isArray(payload.words) ? payload.words : [];
        wordOptionsEl.innerHTML = words.map((word) => `<option value="${escapeHtml(word)}"></option>`).join("");
        wordStatusEl.textContent = `Loaded ${words.length} words.`;
        wordStatusEl.className = "footer-note status-ok";
        if (!wordInputEl.value && words.length > 0) {
          wordInputEl.value = words[0];
        }
      } catch (error) {
        wordStatusEl.textContent = `Unable to load words: ${error.message}`;
        wordStatusEl.className = "footer-note status-bad";
      }
    }

    function mergeChunks(chunks) {
      const totalLength = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
      const merged = new Float32Array(totalLength);
      let offset = 0;
      for (const chunk of chunks) {
        merged.set(chunk, offset);
        offset += chunk.length;
      }
      return merged;
    }

    function parseTrimBound(value, fallback = 0) {
      const numeric = Number(value);
      if (!Number.isFinite(numeric) || numeric < 0) {
        return fallback;
      }
      return Math.round(numeric);
    }

    function encodeWav(samples, sampleRate) {
      const bytesPerSample = 2;
      const buffer = new ArrayBuffer(44 + samples.length * bytesPerSample);
      const view = new DataView(buffer);

      function writeAscii(offset, text) {
        for (let index = 0; index < text.length; index += 1) {
          view.setUint8(offset + index, text.charCodeAt(index));
        }
      }

      writeAscii(0, "RIFF");
      view.setUint32(4, 36 + samples.length * bytesPerSample, true);
      writeAscii(8, "WAVE");
      writeAscii(12, "fmt ");
      view.setUint32(16, 16, true);
      view.setUint16(20, 1, true);
      view.setUint16(22, 1, true);
      view.setUint32(24, sampleRate, true);
      view.setUint32(28, sampleRate * bytesPerSample, true);
      view.setUint16(32, bytesPerSample, true);
      view.setUint16(34, 16, true);
      writeAscii(36, "data");
      view.setUint32(40, samples.length * bytesPerSample, true);

      let offset = 44;
      for (let index = 0; index < samples.length; index += 1) {
        const clamped = Math.max(-1, Math.min(1, samples[index]));
        const pcm = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;
        view.setInt16(offset, pcm, true);
        offset += bytesPerSample;
      }

      return new Blob([buffer], { type: "audio/wav" });
    }

    async function decodeAudioFile(file) {
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      if (!AudioContextClass) {
        throw new Error("This browser cannot decode audio for manual trimming.");
      }
      const context = new AudioContextClass();
      try {
        const buffer = await file.arrayBuffer();
        const audioBuffer = await context.decodeAudioData(buffer.slice(0));
        const channels = audioBuffer.numberOfChannels;
        const length = audioBuffer.length;
        const mono = new Float32Array(length);
        for (let channelIndex = 0; channelIndex < channels; channelIndex += 1) {
          const channel = audioBuffer.getChannelData(channelIndex);
          for (let sampleIndex = 0; sampleIndex < length; sampleIndex += 1) {
            mono[sampleIndex] += channel[sampleIndex] / channels;
          }
        }
        return { sampleRate: audioBuffer.sampleRate, samples: mono };
      } finally {
        await context.close();
      }
    }

    async function buildTrimmedAudioFile(file) {
      const trimStartMs = parseTrimBound(trimStartMsEl.value, 0);
      const trimEndMsRaw = trimEndMsEl.value.trim();
      const decoded = await decodeAudioFile(file);
      const totalMs = (decoded.samples.length / decoded.sampleRate) * 1000;
      const trimEndMs = trimEndMsRaw ? parseTrimBound(trimEndMsRaw, Math.round(totalMs)) : Math.round(totalMs);
      if (trimEndMs <= trimStartMs) {
        throw new Error("Trim end must be greater than trim start.");
      }

      const startSample = Math.max(0, Math.min(decoded.samples.length - 1, Math.round((trimStartMs / 1000) * decoded.sampleRate)));
      const endSample = Math.max(startSample + 1, Math.min(decoded.samples.length, Math.round((trimEndMs / 1000) * decoded.sampleRate)));
      const trimmed = decoded.samples.slice(startSample, endSample);
      const blob = encodeWav(trimmed, decoded.sampleRate);
      const baseName = (file.name || "recording").replace(/\\.[^/.]+$/, "");
      return new File([blob], `${baseName}_trimmed.wav`, { type: "audio/wav" });
    }

    async function startRecording() {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        audioStatusEl.textContent = "This browser does not support microphone capture.";
        audioStatusEl.className = "footer-note status-bad";
        return;
      }

      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const context = new AudioContext();
        const source = context.createMediaStreamSource(stream);
        const processor = context.createScriptProcessor(4096, 1, 1);

        state.mediaStream = stream;
        state.audioContext = context;
        state.sourceNode = source;
        state.processorNode = processor;
        state.chunks = [];
        state.recordingBlob = null;
        state.recordingStartedAt = Date.now();

        processor.onaudioprocess = (event) => {
          const input = event.inputBuffer.getChannelData(0);
          state.chunks.push(new Float32Array(input));
        };

        source.connect(processor);
        processor.connect(context.destination);

        if (state.timerHandle) {
          clearInterval(state.timerHandle);
        }
        state.timerHandle = setInterval(() => {
          const elapsed = (Date.now() - state.recordingStartedAt) / 1000;
          recordingTimerEl.textContent = `${elapsed.toFixed(1)}s`;
        }, 100);

        recordingStateEl.textContent = "recording";
        audioStatusEl.textContent = `Microphone active at ${context.sampleRate} Hz.`;
        audioStatusEl.className = "footer-note status-ok";
        recordButtonEl.disabled = true;
        stopButtonEl.disabled = false;
        fileInputEl.value = "";
      } catch (error) {
        audioStatusEl.textContent = `Unable to start recording: ${error.message}`;
        audioStatusEl.className = "footer-note status-bad";
      }
    }

    async function stopRecording() {
      if (!state.audioContext) {
        return;
      }

      if (state.timerHandle) {
        clearInterval(state.timerHandle);
        state.timerHandle = null;
      }

      state.processorNode.disconnect();
      state.sourceNode.disconnect();
      state.mediaStream.getTracks().forEach((track) => track.stop());

      const sampleRate = state.audioContext.sampleRate;
      await state.audioContext.close();

      const merged = mergeChunks(state.chunks);
      state.recordingBlob = encodeWav(merged, sampleRate);
      audioPreviewEl.src = URL.createObjectURL(state.recordingBlob);
      audioPreviewEl.classList.remove("hidden");

      state.audioContext = null;
      state.sourceNode = null;
      state.processorNode = null;
      state.mediaStream = null;
      state.chunks = [];

      recordingStateEl.textContent = "captured";
      audioStatusEl.textContent = `Captured ${((state.recordingBlob.size || 0) / 1024).toFixed(1)} KiB WAV file.`;
      audioStatusEl.className = "footer-note status-ok";
      recordButtonEl.disabled = false;
      stopButtonEl.disabled = true;
    }

    function clearAudio() {
      if (audioPreviewEl.src) {
        URL.revokeObjectURL(audioPreviewEl.src);
      }
      state.recordingBlob = null;
      audioPreviewEl.removeAttribute("src");
      audioPreviewEl.classList.add("hidden");
      fileInputEl.value = "";
      recordingStateEl.textContent = "idle";
      recordingTimerEl.textContent = "0.0s";
      audioStatusEl.textContent = "No audio selected.";
      audioStatusEl.className = "footer-note";
    }

    async function selectedAudioFile() {
      let file = null;
      if (fileInputEl.files && fileInputEl.files.length > 0) {
        file = fileInputEl.files[0];
      } else if (state.recordingBlob) {
        file = new File([state.recordingBlob], state.recordingFileName, { type: "audio/wav" });
      }
      if (!file) {
        return null;
      }
      if (!frontendTrimEnabledEl.checked) {
        return file;
      }
      return buildTrimmedAudioFile(file);
    }

    function renderSummary(result) {
      const issue = result.primary_issue
        ? `${result.primary_issue.phoneme} (${result.primary_issue.type})`
        : "none";
      const quality = result.audio_quality
        ? `${result.audio_quality.status}, ${result.audio_quality.duration_ms} ms`
        : "unknown";
      const trimInfo = result.audio_quality
        ? (result.audio_quality.trim_applied
          ? `${result.audio_quality.trim_start_ms}-${result.audio_quality.trim_end_ms} of ${result.audio_quality.original_duration_ms} ms`
          : `not applied (${result.audio_quality.original_duration_ms} ms clip)`)
        : "unknown";

      summaryGridEl.innerHTML = `
        <div class="metric"><span class="label">Overall score</span><span class="value">${formatScore(result.overall_score)}</span></div>
        <div class="metric"><span class="label">Confidence</span><span class="value">${formatPercent(result.confidence)}</span></div>
        <div class="metric"><span class="label">Primary issue</span><span class="value">${escapeHtml(issue)}</span></div>
        <div class="metric"><span class="label">Audio quality</span><span class="value">${escapeHtml(quality)}</span></div>
        <div class="metric"><span class="label">Trim window</span><span class="value">${escapeHtml(trimInfo)}</span></div>
      `;
    }

    function renderPhones(result) {
      const phones = Array.isArray(result.phonemes) ? result.phonemes : [];
      if (phones.length === 0) {
        phoneRowsEl.innerHTML = `<tr><td colspan="7" class="footer-note">No phoneme data returned.</td></tr>`;
        return;
      }

      phoneRowsEl.innerHTML = phones.map((phone) => {
        const probs = phone.quality_class_probs || {};
        return `
          <tr>
            <td><strong>${escapeHtml(phone.phoneme)}</strong></td>
            <td>${phone.start_ms}-${phone.end_ms} ms</td>
            <td>${formatScore(phone.expected_score)}</td>
            <td>${escapeHtml(phone.predicted_class)}</td>
            <td>${formatPercent(phone.omission_probability)}</td>
            <td>${formatPercent(phone.confidence)}</td>
            <td class="mono">w=${formatPercent(probs.wrong_or_missed || 0)} a=${formatPercent(probs.accented || 0)} c=${formatPercent(probs.correct || 0)}</td>
          </tr>
        `;
      }).join("");
    }

    function renderResult(result) {
      renderSummary(result);
      renderPhones(result);
      rawJsonEl.textContent = JSON.stringify(result, null, 2);
    }

    async function submitForScoring() {
      const word = wordInputEl.value.trim();
      if (!word) {
        requestStatusEl.textContent = "Enter a target word before scoring.";
        requestStatusEl.className = "footer-note status-bad";
        return;
      }

      const audioFile = await selectedAudioFile();
      if (!audioFile) {
        requestStatusEl.textContent = "Record audio or choose a file first.";
        requestStatusEl.className = "footer-note status-bad";
        return;
      }

      const formData = new FormData();
      formData.append("word", word);
      formData.append("audio", audioFile, audioFile.name || "recording.wav");
      if (speakerInputEl.value.trim()) {
        formData.append("speaker_id", speakerInputEl.value.trim());
      }
      if (noTrimCheckboxEl.checked) {
        formData.append("noTrim", "true");
      }

      scoreButtonEl.disabled = true;
      requestStatusEl.textContent = "Uploading audio and waiting for score...";
      requestStatusEl.className = "footer-note";

      try {
        const result = await fetchJson("/api/score", { method: "POST", body: formData });
        renderResult(result);
        requestStatusEl.textContent = `Scored word "${word}" successfully.`;
        requestStatusEl.className = "footer-note status-ok";
      } catch (error) {
        requestStatusEl.textContent = `Scoring failed: ${error.message}`;
        requestStatusEl.className = "footer-note status-bad";
      } finally {
        scoreButtonEl.disabled = false;
      }
    }

    document.getElementById("pingButton").addEventListener("click", pingBackend);
    document.getElementById("loadWordsButton").addEventListener("click", loadWords);
    recordButtonEl.addEventListener("click", startRecording);
    stopButtonEl.addEventListener("click", stopRecording);
    clearButtonEl.addEventListener("click", clearAudio);
    scoreButtonEl.addEventListener("click", submitForScoring);
    fileInputEl.addEventListener("change", () => {
      if (fileInputEl.files && fileInputEl.files.length > 0) {
        const selected = fileInputEl.files[0];
        audioPreviewEl.src = URL.createObjectURL(selected);
        audioPreviewEl.classList.remove("hidden");
        audioStatusEl.textContent = `Selected file ${selected.name}.`;
        audioStatusEl.className = "footer-note status-ok";
      }
    });

    loadConfig().then(pingBackend).then(loadWords).catch((error) => {
      backendStatusEl.textContent = `Initialization failed: ${error.message}`;
      backendStatusEl.className = "footer-note status-bad";
    });
  </script>
</body>
</html>
"""


@dataclass(frozen=True)
class FrontendSettings:
    backend_base_url: str


def _normalize_backend_url(value: str) -> str:
    normalized = value.strip().rstrip("/")
    if not normalized:
        raise ValueError("Backend URL must not be empty.")
    if not normalized.startswith(("http://", "https://")):
        raise ValueError("Backend URL must start with http:// or https://")
    return normalized


def _join_url(base_url: str, path: str) -> str:
    return f"{base_url}{path}"


def _proxy_request(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    data: bytes | None = None,
    timeout_seconds: float = 60.0,
) -> tuple[int, bytes, dict[str, str]]:
    request_headers = headers or {}
    request_obj = urllib.request.Request(url, data=data, headers=request_headers, method=method)
    try:
        with urllib.request.urlopen(request_obj, timeout=timeout_seconds) as response:
            return response.status, response.read(), dict(response.headers.items())
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), dict(exc.headers.items())
    except urllib.error.URLError as exc:
        payload = json.dumps({"detail": f"Unable to reach backend: {exc.reason}"}).encode("utf-8")
        return 502, payload, {"Content-Type": "application/json"}


def _multipart_body(
    *,
    fields: dict[str, str],
    files: list[tuple[str, str, str, bytes]],
) -> tuple[bytes, str]:
    boundary = f"----pronunciation-frontend-{uuid4().hex}"
    body: list[bytes] = []

    for name, value in fields.items():
        body.append(f"--{boundary}\r\n".encode("utf-8"))
        body.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"))
        body.append(value.encode("utf-8"))
        body.append(b"\r\n")

    for field_name, filename, content_type, content in files:
        body.append(f"--{boundary}\r\n".encode("utf-8"))
        body.append(
            (
                f'Content-Disposition: form-data; name="{field_name}"; '
                f'filename="{filename}"\r\n'
            ).encode("utf-8")
        )
        body.append(f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
        body.append(content)
        body.append(b"\r\n")

    body.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(body), f"multipart/form-data; boundary={boundary}"


def _proxied_response(status_code: int, content: bytes, headers: dict[str, str]) -> Response:
    content_type = headers.get("Content-Type", "application/json").split(";", maxsplit=1)[0]
    return Response(content=content, status_code=status_code, media_type=content_type)


def create_frontend_app(*, backend_base_url: str) -> FastAPI:
    settings = FrontendSettings(backend_base_url=_normalize_backend_url(backend_base_url))
    app = FastAPI(
        title="Pronunciation Frontend Debug UI",
        version="0.1.0",
        description="Lightweight local frontend for manual verification against a remote pronunciation backend.",
    )

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse(INDEX_HTML)

    @app.get("/api/config")
    def config() -> dict[str, str]:
        return {"backend_base_url": settings.backend_base_url}

    @app.get("/api/health")
    def health_proxy() -> Response:
        status_code, content, headers = _proxy_request(
            "GET",
            _join_url(settings.backend_base_url, "/health"),
            headers={"Accept": "application/json"},
        )
        return _proxied_response(status_code, content, headers)

    @app.get("/api/words")
    def words_proxy() -> Response:
        status_code, content, headers = _proxy_request(
            "GET",
            _join_url(settings.backend_base_url, "/v1/words"),
            headers={"Accept": "application/json"},
        )
        return _proxied_response(status_code, content, headers)

    @app.post("/api/score")
    async def score_proxy(
        word: str = Form(...),
        audio: UploadFile = File(...),
        speaker_id: str | None = Form(default=None),
        no_trim: bool = Form(default=False, alias="noTrim"),
    ) -> Response:
        audio_bytes = await audio.read()
        file_name = audio.filename or "recording.wav"
        content_type = audio.content_type or mimetypes.guess_type(file_name)[0] or "application/octet-stream"
        fields = {"word": word}
        if speaker_id:
            fields["speaker_id"] = speaker_id
        if no_trim:
            fields["noTrim"] = "true"
        body, body_content_type = _multipart_body(
            fields=fields,
            files=[("audio", file_name, content_type, audio_bytes)],
        )
        status_code, content, headers = _proxy_request(
            "POST",
            _join_url(settings.backend_base_url, "/v1/pronunciation/score"),
            headers={
                "Accept": "application/json",
                "Content-Type": body_content_type,
            },
            data=body,
        )
        return _proxied_response(status_code, content, headers)

    return app


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the lightweight local frontend for the pronunciation backend.")
    parser.add_argument(
        "--backend-url",
        default=os.getenv(BACKEND_URL_ENV),
        help=f"Remote backend base URL. Can also be provided via {BACKEND_URL_ENV}.",
    )
    parser.add_argument(
        "--host",
        default=os.getenv(HOST_ENV, "127.0.0.1"),
        help=f"Host interface for the local frontend. Can also be provided via {HOST_ENV}.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv(PORT_ENV, "3000")),
        help=f"Port for the local frontend. Can also be provided via {PORT_ENV}.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not args.backend_url:
        raise SystemExit(
            f"Set --backend-url or the {BACKEND_URL_ENV} environment variable to the remote backend endpoint."
        )
    app = create_frontend_app(backend_base_url=args.backend_url)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

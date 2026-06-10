const API_BASE = "http://127.0.0.1:8001/api";

const patientSelect = document.getElementById("patientSelect");
const patientSuggestions = document.getElementById("patientSuggestions");
const patientNameInput = document.getElementById("patientNameInput");
const scenarioInput = document.getElementById("scenarioInput");
const audioFileInput = document.getElementById("audioFile");
const runBtn = document.getElementById("runBtn");
const recordBtn = document.getElementById("recordBtn");
const recordStatus = document.getElementById("recordStatus");
const transcriptOutput = document.getElementById("transcriptOutput");
const normalizedOutput = document.getElementById("normalizedOutput");
const entitiesOutput = document.getElementById("entitiesOutput");
const analysisNotesOutput = document.getElementById("analysisNotesOutput");
const decisionOutput = document.getElementById("decisionOutput");
const refreshTimelineBtn = document.getElementById("refreshTimelineBtn");
const timelineHeader = document.getElementById("timelineHeader");
const timelineList = document.getElementById("timelineList");
const apiStatus = document.getElementById("apiStatus");

let mediaRecorder = null;
let recordedChunks = [];
let recordedBlob = null;
let isRecording = false;
let previewRequestInFlight = false;
let pendingPreviewBlob = null;
let audioContext = null;
let analyserNode = null;
let mediaStreamSource = null;
let speechMonitorId = null;
let lastSpeechDetectedAt = 0;
let currentInputLevel = 0;
const SPEECH_RMS_THRESHOLD = 0.035;
const SPEECH_HANGOVER_MS = 900;
const TELEPHONY_AUDIO_CONSTRAINTS = {
  channelCount: 1,
  sampleRate: 8000,
  noiseSuppression: true,
  echoCancellation: true,
  autoGainControl: true,
  voiceIsolation: true,
};

function nowMs() {
  return Date.now();
}

function isSpeechLikelyActive() {
  return nowMs() - lastSpeechDetectedAt <= SPEECH_HANGOVER_MS;
}

function formatInputLevel() {
  return `Speaker focus active\nMic level: ${currentInputLevel.toFixed(3)}\nThreshold: ${SPEECH_RMS_THRESHOLD.toFixed(3)}`;
}

function startSpeechMonitor(stream) {
  const AudioContextCtor = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextCtor) return;

  audioContext = new AudioContextCtor();
  analyserNode = audioContext.createAnalyser();
  analyserNode.fftSize = 2048;
  mediaStreamSource = audioContext.createMediaStreamSource(stream);
  mediaStreamSource.connect(analyserNode);

  const buffer = new Float32Array(analyserNode.fftSize);
  const tick = () => {
    if (!analyserNode) return;
    analyserNode.getFloatTimeDomainData(buffer);
    let sumSquares = 0;
    for (let i = 0; i < buffer.length; i += 1) {
      sumSquares += buffer[i] * buffer[i];
    }
    currentInputLevel = Math.sqrt(sumSquares / buffer.length);
    if (currentInputLevel >= SPEECH_RMS_THRESHOLD) {
      lastSpeechDetectedAt = nowMs();
    }
  };

  tick();
  speechMonitorId = window.setInterval(tick, 120);
}

function stopSpeechMonitor() {
  if (speechMonitorId) {
    window.clearInterval(speechMonitorId);
    speechMonitorId = null;
  }
  if (mediaStreamSource) {
    mediaStreamSource.disconnect();
    mediaStreamSource = null;
  }
  if (analyserNode) {
    analyserNode.disconnect();
    analyserNode = null;
  }
  if (audioContext) {
    audioContext.close();
    audioContext = null;
  }
}

async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) throw new Error("Health check failed");
    apiStatus.className = "pill pill-success";
    apiStatus.textContent = "Backend connected";
  } catch (error) {
    apiStatus.className = "pill pill-danger";
    apiStatus.textContent = "Backend offline";
  }
}

async function loadPatients() {
  const res = await fetch(`${API_BASE}/patients`);
  const patients = await res.json();
  patientSelect.innerHTML = patients.map((patient) => `<option value="${patient.id}">${patient.full_name}</option>`).join("");
  patientSuggestions.innerHTML = patients.map((patient) => `<option value="${patient.full_name}"></option>`).join("");
}

function renderEntities(entities) {
  entitiesOutput.innerHTML = "";
  if (!entities || entities.length === 0) {
    entitiesOutput.innerHTML = "<span class='pill pill-neutral'>No entities found</span>";
    return;
  }
  entities.forEach((entity) => {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.textContent = `${entity.label}: ${entity.value}`;
    entitiesOutput.appendChild(chip);
  });
}

function renderDecision(decision) {
  const className = decision.priority === "Urgent" || decision.priority === "High"
    ? "pill pill-danger"
    : decision.priority === "Medium"
    ? "pill pill-warning"
    : "pill pill-success";
  decisionOutput.innerHTML = `<span class="${className}">${decision.priority}</span><strong>${decision.title}</strong><p>${decision.description}</p>`;
}

async function sendPreviewChunk(blob) {
  if (!blob || blob.size === 0) return;
  if (previewRequestInFlight) {
    pendingPreviewBlob = blob;
    return;
  }

  previewRequestInFlight = true;
  const formData = new FormData();
  if (patientSelect.value) formData.append("patient_id", patientSelect.value);
  formData.append("important_terms", patientNameInput.value.trim());
  formData.append("audio", blob, "preview.webm");

  try {
    const res = await fetch(`${API_BASE}/transcribe-preview`, { method: "POST", body: formData });
    const payload = await res.json();
    if (res.ok && payload.transcript) {
      transcriptOutput.textContent = payload.transcript;
      analysisNotesOutput.textContent = `Live preview\nSTT model: ${payload.stt_model_used || "unknown"}\nAudio profile: ${payload.audio_profile_used || "unknown"}`;
    }
  } catch {
    // Final processing still runs on the full file.
  } finally {
    previewRequestInFlight = false;
    if (pendingPreviewBlob) {
      const nextBlob = pendingPreviewBlob;
      pendingPreviewBlob = null;
      sendPreviewChunk(nextBlob);
    }
  }
}

async function runPipeline() {
  const patientId = await ensurePatientSelectedOrCreated();
  if (!patientId) return alert("Select a patient first.");
  const selectedFile = audioFileInput.files[0];
  const fileToSend = recordedBlob || selectedFile;
  if (!fileToSend) return alert("Upload an audio file or record one first.");

  const formData = new FormData();
  formData.append("patient_id", patientId);
  formData.append("scenario", scenarioInput.value || "live demo");
  formData.append("important_terms", patientNameInput.value.trim());
  formData.append("audio", fileToSend, fileToSend.name || "recording.webm");

  runBtn.disabled = true;
  runBtn.textContent = "Processing...";
  transcriptOutput.textContent = "Transcribing...";
  normalizedOutput.textContent = "Normalizing...";
  analysisNotesOutput.textContent = "Running clinical extraction...";
  entitiesOutput.innerHTML = "";
  decisionOutput.innerHTML = "<span class='pill pill-neutral'>Processing</span>";

  try {
    const res = await fetch(`${API_BASE}/transcribe`, { method: "POST", body: formData });
    const payload = await res.json();
    if (!res.ok) throw new Error(payload.detail || "Failed to process audio");
    transcriptOutput.textContent = payload.transcript;
    normalizedOutput.textContent = payload.normalized_transcript;
    analysisNotesOutput.textContent = payload.analysis_notes
      ? `${payload.analysis_notes}\nSTT model: ${payload.stt_model_used || "unknown"}\nAudio profile: ${payload.audio_profile_used || "unknown"}`
      : `Clinical extraction completed.\nSTT model: ${payload.stt_model_used || "unknown"}\nAudio profile: ${payload.audio_profile_used || "unknown"}`;
    renderEntities(payload.entities);
    renderDecision(payload.decision);
    await loadTimeline();
  } catch (error) {
    transcriptOutput.textContent = `Error: ${error.message}`;
    normalizedOutput.textContent = "No output";
    analysisNotesOutput.textContent = "No analysis output";
    entitiesOutput.innerHTML = "<span class='pill pill-danger'>Request failed</span>";
    decisionOutput.innerHTML = "<span class='pill pill-danger'>Could not complete pipeline</span>";
  } finally {
    runBtn.disabled = false;
    runBtn.textContent = "Run CareCaller";
  }
}

async function loadTimeline() {
  const patientId = patientSelect.value;
  if (!patientId) return;
  const res = await fetch(`${API_BASE}/patients/${patientId}/timeline`);
  const payload = await res.json();
  patientNameInput.value = payload.patient.full_name;
  timelineHeader.innerHTML = `<strong>${payload.patient.full_name}</strong><br />Phone: ${payload.patient.phone_number || "Not added"}<br />Medications: ${payload.patient.medications || "None listed"}<br />Conditions: ${payload.patient.conditions || "None listed"}`;
  if (!payload.calls || payload.calls.length === 0) {
    timelineList.innerHTML = "<div class='timeline-item'>No calls yet for this patient.</div>";
    return;
  }
  timelineList.innerHTML = payload.calls.map((call) => `
    <article class="timeline-item">
      <div class="timeline-item-top">
        <strong>${new Date(call.created_at).toLocaleString()}</strong>
        <span class="pill ${call.priority === "Urgent" || call.priority === "High" ? "pill-danger" : call.priority === "Medium" ? "pill-warning" : "pill-success"}">${call.priority}</span>
      </div>
      <div><strong>${call.decision_title}</strong></div>
      <p>${call.decision_description}</p>
      <p><strong>Transcript:</strong> ${call.normalized_transcript}</p>
    </article>
  `).join("");
}

function findPatientOptionByName(name) {
  const typedName = name.trim().toLowerCase();
  if (!typedName) return null;
  return Array.from(patientSelect.options).find(
    (option) => option.textContent.trim().toLowerCase() === typedName,
  );
}

async function createPatientFromTypedName(name) {
  const res = await fetch(`${API_BASE}/patients`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ full_name: name.trim() }),
  });
  const patient = await res.json();
  if (!res.ok) {
    throw new Error(patient.detail || "Could not create patient");
  }
  await loadPatients();
  const exactMatch = findPatientOptionByName(patient.full_name);
  if (exactMatch) {
    patientSelect.value = exactMatch.value;
  }
  patientNameInput.value = patient.full_name;
  await loadTimeline();
  return patient.id;
}

async function ensurePatientSelectedOrCreated() {
  const typedName = patientNameInput.value.trim();
  if (!typedName) return null;

  const exactMatch = findPatientOptionByName(typedName);
  if (exactMatch) {
    patientSelect.value = exactMatch.value;
    return exactMatch.value;
  }

  const patient = await createPatientFromTypedName(typedName);
  return String(patient);
}

async function syncPatientNameToActivePatient() {
  const typedName = patientNameInput.value.trim();
  if (!typedName) {
    patientSelect.value = "";
    timelineHeader.textContent = "Type a patient name to load or create a timeline.";
    timelineList.innerHTML = "";
    return;
  }

  const exactMatch = findPatientOptionByName(typedName);
  if (exactMatch) {
    patientSelect.value = exactMatch.value;
    await loadTimeline();
    return;
  }

  try {
    await createPatientFromTypedName(typedName);
  } catch (error) {
    timelineHeader.textContent = error.message;
    timelineList.innerHTML = "";
  }
}

async function toggleRecording() {
  if (!isRecording) {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: TELEPHONY_AUDIO_CONSTRAINTS });
      startSpeechMonitor(stream);
      recordedChunks = [];
      recordedBlob = null;
      lastSpeechDetectedAt = nowMs();
      currentInputLevel = 0;
      mediaRecorder = new MediaRecorder(stream);
      transcriptOutput.textContent = "Listening for live preview...";
      normalizedOutput.textContent = "Normalized transcript will appear after full processing...";
      analysisNotesOutput.textContent = "Starting speaker-focused telephony preview...";
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0 && isSpeechLikelyActive()) {
          recordedChunks.push(event.data);
          sendPreviewChunk(event.data);
        } else if (event.data.size > 0) {
          analysisNotesOutput.textContent = `${formatInputLevel()}\nIgnoring low-confidence background audio.`;
        }
      };
      mediaRecorder.onstop = () => {
        recordedBlob = new Blob(recordedChunks, { type: "audio/webm" });
        recordedBlob.name = "browser-recording.webm";
        recordStatus.textContent = "Recording ready";
        stopSpeechMonitor();
        stream.getTracks().forEach((track) => track.stop());
      };
      mediaRecorder.start(1500);
      isRecording = true;
      recordBtn.textContent = "Stop recording";
      recordStatus.textContent = "Recording with speaker focus...";
    } catch {
      alert("Microphone access failed.");
    }
  } else {
    mediaRecorder.stop();
    isRecording = false;
    recordBtn.textContent = "Start recording";
  }
}

patientSelect.addEventListener("change", loadTimeline);
patientNameInput.addEventListener("change", syncPatientNameToActivePatient);
refreshTimelineBtn.addEventListener("click", loadTimeline);
runBtn.addEventListener("click", runPipeline);
recordBtn.addEventListener("click", toggleRecording);

checkHealth();
loadPatients();

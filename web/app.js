const SANDBOX_API_BASE = 'http://127.0.0.1:8082/api';
const SANDBOX_HEALTH_URL = 'http://127.0.0.1:8082/healthz';
const INFERENCE_V1_BASE = 'http://127.0.0.1:8081/v1';

const $ = (id) => document.getElementById(id);
let runId = '';
let events = [];
let pending = false;
let viewToken = 0;
let health = {};
let model = {};
let config = {};
let directModels = {};

async function json(url, options) {
  const response = await fetch(url, options);
  const data = await response.json().catch(() => ({}));
  data.http_status = response.status;
  return data;
}

function shown(event) {
  return ['user', 'assistant', 'error'].includes(event.kind);
}

function renderEvents() {
  const box = $('events');
  box.innerHTML = '';
  const list = events.filter(shown);
  if (!list.length) {
    const empty = document.createElement('p');
    empty.className = 'muted';
    empty.textContent = 'No messages yet.';
    box.appendChild(empty);
  }
  for (const event of list) {
    const node = document.createElement('div');
    node.className = `event ${event.kind}`;
    node.textContent = event.content || '';
    box.appendChild(node);
  }
  box.scrollTop = box.scrollHeight;
}

function chatReason() {
  if (model.chat_supported) return 'chat ready';
  if (model.artifact_kind === 'dense') return 'dense diagnostics only';
  if (model.artifact_kind === 'transformer') return 'transformer diagnostics only';
  return model.reachable ? 'decoder required' : 'model unavailable';
}

function applyState() {
  $('message').disabled = pending;
  $('send').disabled = pending;
  $('chatState').textContent = pending ? 'sending' : chatReason();
  $('modelLine').textContent =
    `${model.model || 'model'} / ${model.artifact_kind || 'unknown'} / ${config.tool_profile || 'tools pending'}`;
  $('status').textContent =
    `${health.status || 'unknown'} / ${model.reachable ? 'reachable' : 'not reachable'}`;
  $('modelJson').textContent = JSON.stringify({
    sandbox: { health, model, config },
    inference: { models: directModels }
  }, null, 2);
}

function startDraft() {
  viewToken += 1;
  runId = '';
  events = [];
  $('message').value = '';
  $('title').textContent = 'Draft run';
  renderEvents();
  applyState();
  loadRuns('');
}

function runButton(run, active) {
  const button = document.createElement('button');
  button.className = `run${active ? ' active' : ''}`;
  button.type = 'button';
  button.innerHTML =
    `<strong>${run.run_id}</strong><br><span class="muted">${run.last_kind || 'empty'}: ${(run.preview || '').slice(0, 72)}</span>`;
  button.onclick = () => loadRun(run.run_id);
  return button;
}

async function loadRuns(active) {
  const data = await json(`${SANDBOX_API_BASE}/runs?limit=20`).catch(() => ({ runs: [] }));
  $('runs').innerHTML = '';
  $('runState').textContent = data.runs && data.runs.length ? 'Recent runs' : 'No persisted runs';
  for (const run of data.runs || []) {
    $('runs').appendChild(runButton(run, run.run_id === active));
  }
  return data.runs || [];
}

async function loadRun(id) {
  const token = ++viewToken;
  const data = await json(`${SANDBOX_API_BASE}/runs/${encodeURIComponent(id)}`);
  if (token !== viewToken) return;
  runId = id;
  events = data.events || [];
  $('title').textContent = id;
  renderEvents();
  await loadRuns(id);
}

async function loadStatus() {
  [health, model, config, directModels] = await Promise.all([
    json(SANDBOX_HEALTH_URL).catch((error) => ({ error: String(error) })),
    json(`${SANDBOX_API_BASE}/model`).catch((error) => ({ error: String(error) })),
    json(`${SANDBOX_API_BASE}/config`).catch((error) => ({ error: String(error) })),
    json(`${INFERENCE_V1_BASE}/models`).catch((error) => ({ error: String(error) }))
  ]);
  applyState();
}

async function sendMessage(event) {
  event.preventDefault();
  const message = $('message').value.trim();
  if (!message || pending) return;
  pending = true;
  applyState();
  const body = { message };
  if (runId) body.run_id = runId;
  const data = await json(`${SANDBOX_API_BASE}/chat`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body)
  }).catch((error) => ({ error: String(error), events }));
  runId = data.run_id || runId;
  events = data.events || events;
  $('message').value = '';
  $('title').textContent = runId || 'Draft run';
  renderEvents();
  await loadRuns(runId);
  await loadStatus();
  pending = false;
  applyState();
}

$('newRun').onclick = startDraft;
$('composer').addEventListener('submit', sendMessage);
$('message').addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    $('composer').requestSubmit();
  }
});

loadStatus().then(async () => {
  const runs = await loadRuns('');
  if (runs[0] && viewToken === 0) await loadRun(runs[0].run_id);
  else renderEvents();
});

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
let lastChat = {};

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
  box.replaceChildren();
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

function directModelName() {
  return model.model || (((directModels.data || [])[0] || {}).id) || 'decoder-40m-3070';
}

function applyState() {
  $('message').disabled = pending;
  $('send').disabled = pending;
  $('chatState').textContent = pending ? 'sending' : chatReason();
  $('chatOutcome').textContent = lastChat.stop_reason
    ? `last ${lastChat.http_status || 'n/a'} / ${lastChat.stop_reason}`
    : 'no attempt';
  $('chatOutcome').className =
    lastChat.stop_reason === 'finish' ? 'outcome-ok' :
    lastChat.stop_reason ? 'outcome-bad' : 'muted';
  $('modelLine').textContent =
    `${model.model || 'model'} / ${model.artifact_kind || 'unknown'} / ${config.tool_profile || 'tools pending'}`;
  $('status').textContent =
    `${health.status || 'unknown'} / ${model.reachable ? 'reachable' : 'not reachable'}`;
  $('modelFacts').replaceChildren(...webState.modelFacts(model));
  $('modelJson').textContent = JSON.stringify({
    sandbox: { health, model, config },
    inference: { models: directModels },
    last_chat: lastChat
  }, null, 2);
}

function startDraft() {
  viewToken += 1;
  runId = '';
  events = [];
  lastChat = {};
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
  const title = document.createElement('strong');
  title.textContent = run.run_id || '';
  const lineBreak = document.createElement('br');
  const preview = document.createElement('span');
  preview.className = 'muted';
  preview.textContent =
    `${run.last_kind || 'empty'}: ${(run.preview || '').slice(0, 72)}`;
  button.append(title, lineBreak, preview);
  button.onclick = () => loadRun(run.run_id);
  return button;
}

async function loadRuns(active) {
  const data = await json(`${SANDBOX_API_BASE}/runs?limit=20`).catch(() => ({ runs: [] }));
  $('runs').replaceChildren();
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
  const previousEvents = events;
  const data = await json(`${SANDBOX_API_BASE}/chat`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body)
  }).catch((error) => ({ error: String(error), events: previousEvents, http_status: 0 }));
  runId = data.run_id || runId;
  events = data.events || previousEvents;
  lastChat = {
    http_status: data.http_status || 0,
    stop_reason: data.stop_reason || (data.error ? 'request_error' : ''),
    error: data.error || ''
  };
  if (lastChat.stop_reason && lastChat.stop_reason !== 'finish' &&
      !webState.hasAssistant(events)) {
    const direct = await webDirect.chat(INFERENCE_V1_BASE, directModelName(), message)
      .catch((error) => ({ error: String(error), http_status: 0 }));
    if (direct.content) {
      events = [...events, {
        kind: 'assistant',
        content: `[direct model] ${direct.content}`
      }];
      lastChat.direct = {
        http_status: direct.http_status || 0,
        stop_reason: direct.stop_reason || 'direct_response'
      };
    }
  }
  if (lastChat.stop_reason && lastChat.stop_reason !== 'finish' &&
      !webState.hasAssistant(events) && !webState.hasError(events)) {
    events = [...events, { kind: 'error', content: webState.failureMessage(data, model) }];
  }
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

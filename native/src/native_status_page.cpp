#include "native_status_page.hpp"

namespace lkjai {

std::string_view native_status_page_html() {
  return R"HTML(<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>lkjai operator</title>
<style>
body{margin:0;font-family:system-ui,sans-serif;color:#172026;background:#f6f7f4}
.wrap{max-width:1180px;margin:auto;padding:18px}
header{display:flex;justify-content:space-between;gap:12px;align-items:center}
h1{font-size:18px;margin:0}.status{font-size:13px;color:#506069}
main{display:grid;grid-template-columns:260px 1fr;gap:14px;margin-top:14px}
aside,.pane,details{background:#fff;border:1px solid #d8ddd8;border-radius:8px}
aside{padding:12px}.pane{min-height:520px;display:flex;flex-direction:column}
.bar{display:flex;justify-content:space-between;gap:8px;padding:12px;border-bottom:1px solid #e2e6e1}
.runs{display:grid;gap:6px}.run{border:1px solid #dfe4df;background:#fbfcfa;border-radius:6px;padding:8px;text-align:left;color:#172026}
.run.active{border-color:#246b5e;background:#eef7f3}.muted{color:#66757d;font-size:13px}
.events{flex:1;overflow:auto;padding:14px;display:grid;align-content:start;gap:10px}
.event{max-width:760px;border-radius:8px;padding:10px 12px;background:#f2f4f1}
.user{justify-self:end;background:#e8f1ee}.assistant{justify-self:start;background:#fff;border:1px solid #dfe4df}.error{border:1px solid #d19a9a;background:#fff4f2}
.composer{border-top:1px solid #e2e6e1;padding:12px;display:grid;grid-template-columns:1fr auto;gap:8px}
textarea{width:100%;min-height:78px;box-sizing:border-box;font:inherit;border:1px solid #cfd7d1;border-radius:6px;padding:10px}
button{height:36px;padding:0 12px;border:0;border-radius:6px;background:#174c43;color:#fff}
button:disabled,textarea:disabled{opacity:.55}.secondary{background:#52615f}
details{margin-top:14px;padding:12px}summary{cursor:pointer;font-weight:650}
pre{white-space:pre-wrap;word-break:break-word;margin:8px 0 0;font-size:12px}
.diag{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:10px}
.row{display:grid;grid-template-columns:56px 1fr 72px;gap:8px;align-items:center}
.meter{height:8px;background:#e1e7e2}.meter span{display:block;height:8px;background:#2f6b5f}
@media(max-width:820px){main,.diag{grid-template-columns:1fr}.wrap{padding:12px}.pane{min-height:480px}}
</style>
</head>
<body>
<div class="wrap">
<header><h1>lkjai operator</h1><div id="status" class="status">loading</div></header>
<main>
<aside>
<div class="bar"><b>Runs</b><button id="new" class="secondary">New</button></div>
<p class="muted" id="runState">loading runs</p><div class="runs" id="runs"></div>
</aside>
<section class="pane">
<div class="bar"><div><b id="title">Draft run</b><div class="muted" id="modelLine">model pending</div></div><div class="muted" id="chatState">chat pending</div></div>
<div class="events" id="events"></div>
<div class="composer"><textarea id="message" placeholder="Send an operator message"></textarea><button id="send">Send</button></div>
</section>
</main>
<details><summary>Advanced diagnostics</summary>
<div class="diag">
<div><label for="tokens">Dense token input</label><textarea id="tokens">1,2,3</textarea><p><button id="runDense">Run logits</button></p><div id="rows">top-k logits pending</div></div>
<div><b>Raw JSON</b><pre id="raw">diagnostics pending</pre></div>
</div>
</details>
</div>
<script>
const $=id=>document.getElementById(id);
let runId='',events=[],pending=false,viewToken=0,model={},health={},config={},dense={};
const shown=e=>['user','assistant','error'].includes(e.kind);
function eventNode(e){const d=document.createElement('div');d.className='event '+e.kind;d.textContent=e.content||'';return d}
function renderEvents(){const box=$('events');box.innerHTML='';const list=events.filter(shown);if(!list.length){const p=document.createElement('p');p.className='muted';p.textContent='No messages yet.';box.appendChild(p)}for(const e of list)box.appendChild(eventNode(e));box.scrollTop=box.scrollHeight}
function reason(){if(model.chat_supported)return '';const k=model.artifact_kind||'artifact';if(k==='dense')return 'Dense artifact loaded; chat requires a decoder artifact';if(k==='transformer')return 'Transformer artifact loaded; chat requires a decoder artifact';return 'Chat requires a loaded decoder artifact'}
function applyModel(){const can=!!model.chat_supported&&!pending;$('message').disabled=!can;$('send').disabled=!can;$('chatState').textContent=pending?'sending':reason()||'chat ready';$('modelLine').textContent=`${model.model||'model'} / ${model.artifact_kind||'unknown'} / ${config.tool_profile||model.tool_profile||'tool profile pending'}`;$('status').textContent=`${health.status||'unknown'} / ${model.loaded?'loaded':'not loaded'}`}
function startDraft(){viewToken++;runId='';events=[];$('message').value='';$('title').textContent='Draft run';renderEvents();loadRuns('')}
async function loadRun(id){const token=++viewToken;const data=await fetch('/api/runs/'+encodeURIComponent(id)).then(r=>r.json());if(token!==viewToken)return;runId=id;events=data.events||[];$('title').textContent=id;renderEvents();await loadRuns(id)}
function runButton(r,active){const b=document.createElement('button');b.className='run'+(active?' active':'');b.innerHTML=`<b>${r.run_id}</b><br><span class="muted">${r.last_kind||'empty'}: ${(r.preview||'').slice(0,72)}</span>`;b.onclick=()=>loadRun(r.run_id);return b}
async function loadRuns(active){const data=await fetch('/api/runs?limit=20').then(r=>r.json()).catch(()=>({runs:[]}));$('runs').innerHTML='';$('runState').textContent=active&&data.runs.length?'Recent runs':'No persisted runs';for(const r of data.runs)$('runs').appendChild(runButton(r,r.run_id===active));return data.runs}
async function load(){
  [health,model,config,dense]=await Promise.all([
    fetch('/healthz').then(r=>r.json()).catch(e=>({error:String(e)})),
    fetch('/api/model').then(r=>r.json()).catch(e=>({error:String(e)})),
    fetch('/api/config').then(r=>r.json()).catch(e=>({error:String(e)})),
    fetch('/api/dense/status').then(r=>r.json()).catch(e=>({error:String(e)}))
  ]);
  $('raw').textContent=JSON.stringify({health,model,config,dense},null,2);applyModel();
  const runs=await loadRuns('');if(runs[0]&&viewToken===0)await loadRun(runs[0].run_id);else renderEvents();
}
async function send(){
  const message=$('message').value.trim();if(!message||pending)return;pending=true;applyModel();
  const body={message};if(runId)body.run_id=runId;
  const data=await fetch('/api/chat',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(body)}).then(r=>r.json());
  runId=data.run_id||runId;events=data.events||events;$('message').value='';$('title').textContent=runId||'Draft run';renderEvents();await loadRuns(runId);pending=false;applyModel();
}
function renderRows(data){$('rows').innerHTML='';for(const item of data.top_k||[]){const row=document.createElement('div');row.className='row';const pct=Math.max(0,Math.min(100,(item.prob||0)*100));row.innerHTML=`<b>${item.id}</b><div class="meter"><span style="width:${pct}%"></span></div><code>${pct.toFixed(2)}%</code>`;$('rows').appendChild(row)}$('raw').textContent=JSON.stringify(data,null,2)}
$('new').onclick=startDraft;
$('send').onclick=send;
$('message').addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send()}});
$('runDense').onclick=async()=>{const tokens=$('tokens').value.split(',').map(s=>Number(s.trim())).filter(Number.isInteger);const body=JSON.stringify({tokens,top_k:8});const r=await fetch('/api/dense/next-token',{method:'POST',headers:{'content-type':'application/json'},body});renderRows(await r.json())};
load();
</script>
</body>
</html>)HTML";
}

}  // namespace lkjai

#include "native_status_page.hpp"

namespace lkjai {

std::string_view native_status_page_html() {
  return R"HTML(<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>lkjai native</title>
<style>
body{margin:0;font-family:system-ui,sans-serif;color:#172026;background:#f7f7f4}
.wrap{max-width:980px;margin:auto;padding:24px}
header{display:flex;justify-content:space-between;gap:16px;align-items:center}
h1{font-size:24px;margin:0}.status{font-size:13px;color:#4b5b63}
main{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:20px}
section{background:#fff;border:1px solid #d8ddd8;border-radius:8px;padding:16px}
label{display:block;font-size:13px;color:#4b5b63;margin-bottom:6px}
textarea{width:100%;min-height:92px;box-sizing:border-box;font:inherit}
button{height:36px;padding:0 14px;border:0;border-radius:6px;background:#174c43;color:#fff}
.row{display:grid;grid-template-columns:56px 1fr 72px;gap:8px;align-items:center}
.bar{height:8px;background:#e1e7e2}.bar span{display:block;height:8px;background:#2f6b5f}
pre{white-space:pre-wrap;word-break:break-word;margin:0;font-size:13px}.muted{color:#647277}
@media(max-width:760px){main{grid-template-columns:1fr}.wrap{padding:16px}}
</style>
</head>
<body>
<div class="wrap">
<header><h1>lkjai dense demo</h1><div id="status" class="status">loading</div></header>
<main>
<section><pre id="model">model status pending</pre><p class="muted" id="chat">chat state pending</p></section>
<section>
<label for="tokens">Token ids</label>
<textarea id="tokens">1,2,3</textarea>
<p><button id="run">Run</button></p>
<div id="rows">top-k pending</div>
<pre id="reply"></pre>
</section>
</main>
</div>
<script>
const $=id=>document.getElementById(id);
async function load(){
  const h=await fetch('/healthz').then(r=>r.json()).catch(e=>({error:String(e)}));
  const m=await fetch('/api/model').then(r=>r.json()).catch(e=>({error:String(e)}));
  const d=await fetch('/api/dense/status').then(r=>r.json()).catch(e=>({error:String(e)}));
  $('status').textContent=d.status==='ready'?'dense ready':'dense degraded';
  $('chat').textContent='chat decode: unsupported for dense artifacts';
  $('model').textContent=JSON.stringify({process:h.status,loaded:m.loaded,
    dense:d.dense_supported,checksum:d.weights_checksum,
    config_checksum:d.config_checksum,optimizer_steps:d.optimizer_steps,
    loss:d.loss,parameter_count:d.parameter_count,
    provenance:d.train_report_path||'none'},null,2);
}
function renderRows(data){
  $('rows').innerHTML='';
  for(const item of data.top_k||[]){
    const row=document.createElement('div'); row.className='row';
    const pct=Math.max(0,Math.min(100,(item.prob||0)*100));
    row.innerHTML=`<b>${item.id}</b><div class="bar"><span style="width:${pct}%"></span></div><code>${pct.toFixed(2)}%</code>`;
    $('rows').appendChild(row);
  }
}
$('run').onclick=async()=>{
  const tokens=$('tokens').value.split(',').map(s=>Number(s.trim())).filter(Number.isInteger);
  const body=JSON.stringify({tokens,top_k:8});
  const r=await fetch('/api/dense/next-token',{method:'POST',headers:{'content-type':'application/json'},body});
  const data=await r.json(); renderRows(data);
  $('reply').textContent=JSON.stringify({top_token:data.top_token,
    checksum:data.checksum,weights_checksum:data.weights_checksum,
    provenance:data.train_report_path||'none'},null,2);
};
load();
</script>
</body>
</html>)HTML";
}

}  // namespace lkjai

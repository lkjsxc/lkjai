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
pre{white-space:pre-wrap;word-break:break-word;margin:0;font-size:13px}
@media(max-width:760px){main{grid-template-columns:1fr}.wrap{padding:16px}}
</style>
</head>
<body>
<div class="wrap">
<header><h1>lkjai dense demo</h1><div id="status" class="status">loading</div></header>
<main>
<section><pre id="model">model status pending</pre></section>
<section>
<label for="tokens">Token ids</label>
<textarea id="tokens">1,2,3</textarea>
<p><button id="run">Run</button></p>
<pre id="reply">top-k pending</pre>
</section>
</main>
</div>
<script>
const $=id=>document.getElementById(id);
async function load(){
  const h=await fetch('/healthz').then(r=>r.json()).catch(e=>({error:String(e)}));
  const m=await fetch('/api/model').then(r=>r.json()).catch(e=>({error:String(e)}));
  const d=await fetch('/api/dense/status').then(r=>r.json()).catch(e=>({error:String(e)}));
  $('status').textContent=h.status==='ok'?'process ok':'process degraded';
  $('model').textContent=JSON.stringify({model:m,dense:d},null,2);
}
$('run').onclick=async()=>{
  const tokens=$('tokens').value.split(',').map(s=>Number(s.trim())).filter(Number.isInteger);
  const body=JSON.stringify({tokens,top_k:8});
  const r=await fetch('/api/dense/next-token',{method:'POST',headers:{'content-type':'application/json'},body});
  $('reply').textContent=JSON.stringify(await r.json(),null,2);
};
load();
</script>
</body>
</html>)HTML";
}

}  // namespace lkjai

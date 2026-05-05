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
:root{font-family:system-ui,sans-serif;color:#172026;background:#f7f7f4}
body{margin:0}.wrap{max-width:960px;margin:auto;padding:24px}
header{display:flex;justify-content:space-between;gap:16px;align-items:center}
h1{font-size:24px;margin:0}.status{font-size:13px;color:#4b5b63}
main{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:20px}
section{background:white;border:1px solid #d8ddd8;border-radius:8px;padding:16px}
textarea{width:100%;min-height:120px;box-sizing:border-box;font:inherit}
button{height:36px;padding:0 14px;border:0;border-radius:6px;background:#174c43;color:white}
pre{white-space:pre-wrap;word-break:break-word;margin:0;font-size:13px}
@media(max-width:760px){main{grid-template-columns:1fr}.wrap{padding:16px}}
</style>
</head>
<body>
<div class="wrap">
<header><h1>lkjai native</h1><div id="status" class="status">loading</div></header>
<main>
<section><pre id="model">model status pending</pre></section>
<section>
<textarea id="msg" placeholder="Message"></textarea>
<p><button id="send">Send</button></p>
<pre id="reply"></pre>
</section>
</main>
</div>
<script>
const $=id=>document.getElementById(id);
async function load(){
  const h=await fetch('/healthz').then(r=>r.json()).catch(e=>({error:String(e)}));
  const m=await fetch('/api/model').then(r=>r.json()).catch(e=>({error:String(e)}));
  $('status').textContent=h.status==='ok'?'process ok':'process degraded';
  $('model').textContent=JSON.stringify(m,null,2);
}
$('send').onclick=async()=>{
  const body=JSON.stringify({message:$('msg').value,max_steps:6});
  const r=await fetch('/api/chat',{method:'POST',headers:{'content-type':'application/json'},body});
  $('reply').textContent=JSON.stringify(await r.json(),null,2);
};
load();
</script>
</body>
</html>)HTML";
}

}  // namespace lkjai

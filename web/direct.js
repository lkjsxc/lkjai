const webDirect = (() => {
  async function chat(base, model, message) {
    const response = await fetch(`${base}/chat/completions`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        model,
        messages: [{ role: 'user', content: message }],
        max_tokens: 64,
        temperature: 0
      })
    });
    const data = await response.json().catch(() => ({}));
    const choice = (data.choices || [])[0] || {};
    return {
      http_status: response.status,
      stop_reason: choice.finish_reason || data.error || '',
      content: (choice.message && choice.message.content) || ''
    };
  }
  return { chat };
})();

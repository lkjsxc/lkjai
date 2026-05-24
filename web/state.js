function stateFact(label, value, state = '') {
  const span = document.createElement('span');
  span.className = `fact ${state}`.trim();
  span.textContent = `${label}: ${value}`;
  return span;
}

window.webState = {
  modelFacts(model) {
    return [
      stateFact('artifact', model.artifact_kind || 'unknown'),
      stateFact('decode', model.decode_supported ? 'yes' : 'no',
                model.decode_supported ? '' : 'warn'),
      stateFact('http', model.probe_status || 'n/a',
                model.probe_status === 200 ? '' : 'warn'),
      stateFact('degraded',
                model.degraded_reason || (model.degraded ? 'yes' : 'no'),
                model.degraded ? 'bad' : '')
    ];
  },

  hasAssistant(events) {
    return events.some((event) =>
      ['assistant', 'finish'].includes(event.kind) &&
      (event.content || '').trim());
  },

  hasError(events) {
    return events.some((event) =>
      event.kind === 'error' && (event.content || '').trim());
  },

  failureMessage(data, model) {
    const parts = [];
    if (data.http_status) parts.push(`HTTP ${data.http_status}`);
    if (data.stop_reason) parts.push(`stop_reason=${data.stop_reason}`);
    if (data.error) parts.push(String(data.error));
    if (model.degraded_reason) parts.push(`model=${model.degraded_reason}`);
    return parts.join(' / ') || 'chat attempt failed';
  }
};

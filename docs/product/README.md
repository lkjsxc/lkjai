# Product

Use this subtree for user-visible behavior: dense demo, chat, agent tools,
memory, and HTTP API contracts.

## Read This Section When

- You need multi-turn chat behavior.
- You need the dense logits browser demo.
- You need the structured tool surface.
- You need route, payload, and event contracts.
- You need runtime expectations for a real model-backed assistant.

Current native dense artifacts support readiness, logits, top-k, and checksum
surfaces, but they do not support autoregressive chat decode. Decoder artifacts
are the chat target and disclose accepted CUDA KV-cache decode only when route
evidence is present.

## Child Index

- [chat.md](chat.md): local multi-turn chat behavior and stop reasons
- [dense-demo.md](dense-demo.md): dense next-token browser demo contract
- [agent-tools.md](agent-tools.md): command, website, file, and memory tools
- [api.md](api.md): HTTP route, payload, and error contracts
- [kjxlkj-integration.md](kjxlkj-integration.md): future note-app handoff

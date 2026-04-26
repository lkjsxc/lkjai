import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .generation import LoadedModel


class State:
    model_name = os.environ.get("MODEL_NAME", "lkjai-scratch-40m")
    model_root = Path(os.environ.get("MODEL_ROOT", "/models"))
    loaded: LoadedModel | None = None
    error = ""


def main() -> None:
    load_model()
    host = os.environ.get("INFERENCE_HOST", "0.0.0.0")
    port = int(os.environ.get("INFERENCE_PORT", "8081"))
    ThreadingHTTPServer((host, port), Handler).serve_forever()


def load_model() -> None:
    model_dir = State.model_root / State.model_name
    required = ["manifest.json", "config.json", "tokenizer.json", "model.pt"]
    missing = [name for name in required if not (model_dir / name).is_file()]
    if missing:
        State.error = f"missing artifact files: {', '.join(missing)}"
        return
    try:
        State.loaded = LoadedModel(model_dir)
    except Exception as error:
        State.error = f"failed to load model: {error}"


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/healthz":
            return self.send_json(200, {"status": "ok", "loaded": State.loaded is not None, "error": State.error})
        if self.path == "/v1/models":
            if not State.loaded:
                return self.send_json(503, {"error": State.error})
            body = {"data": [{"id": State.model_name, "object": "model"}], **device_status()}
            return self.send_json(200, body)
        self.send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            return self.send_json(404, {"error": "not found"})
        if not State.loaded:
            return self.send_json(503, {"error": State.error})
        try:
            body = self.read_json()
            content = State.loaded.complete(
                body.get("messages", []),
                int(body.get("max_tokens") or 128),
                float(body.get("temperature") or 0.0),
            )
            message = {"role": "assistant", "content": content}
            self.send_json(200, {"choices": [{"message": message}]})
        except Exception as error:
            self.send_json(500, {"error": str(error)})

    def read_json(self) -> dict:
        size = int(self.headers.get("content-length", "0"))
        raw = self.rfile.read(size).decode("utf-8")
        return json.loads(raw) if raw else {}

    def send_json(self, status: int, body: dict) -> None:
        data = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args) -> None:
        return


def device_status() -> dict:
    try:
        import torch

        cuda = torch.cuda.is_available()
        gpu = torch.cuda.get_device_name(0) if cuda else ""
    except Exception:
        cuda, gpu = False, ""
    device = State.loaded.device if State.loaded else ""
    warning = "" if cuda else "CUDA unavailable; serving on CPU fallback"
    return {"device": device, "cuda_available": cuda, "gpu_name": gpu, "warning": warning}


if __name__ == "__main__":
    main()

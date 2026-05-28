"""Minimal OpenAI-compatible chat-completions server for Qwen3-VL-8B-Instruct.

Stands in for vLLM (which has no Windows wheel) so HY-World's worldgen
pipeline can run unmodified. Implements only the subset that
traj_generate.py / traj_render.py actually call:

    POST /v1/chat/completions
    body : {model, messages, max_tokens, temperature, seed}
    item : {type: "text", text}
         | {type: "image_url", image_url: {url: "data:image/...;base64,..."}}

Loads the model once at startup, single-threaded inference (no batching,
no paged attention). Slow but correct. ~17 GB VRAM in bf16.

Usage:
    python vlm_server.py
    # in another shell:
    .\\run_example.bat --module worldgen
"""

import argparse
import base64
import io
import os
import time
import uuid
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def _decode_data_url(url: str) -> Image.Image:
    if not url.startswith("data:"):
        raise HTTPException(400, f"image_url must be a data URL (got prefix: {url[:32]!r})")
    payload = url.split(",", 1)[-1]
    return Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")


class ContentItem(BaseModel):
    type: str
    text: str | None = None
    image_url: dict[str, Any] | None = None


class ChatMessage(BaseModel):
    role: str
    content: list[ContentItem] | str


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int = 1024
    temperature: float = 0.0
    seed: int | None = None


def _to_qwen_messages(req_messages):
    """Translate OpenAI vision schema -> Qwen3-VL chat-template schema.
    OpenAI : {type:"image_url", image_url:{url:"data:image/png;base64,..."}}
    Qwen3VL: {type:"image", image: PIL.Image}
    """
    out = []
    images = []
    for m in req_messages:
        if isinstance(m.content, str):
            out.append({"role": m.role, "content": m.content})
            continue
        items = []
        for c in m.content:
            if c.type == "text":
                items.append({"type": "text", "text": c.text or ""})
            elif c.type == "image_url":
                if not c.image_url or "url" not in c.image_url:
                    raise HTTPException(400, "image_url.image_url.url missing")
                img = _decode_data_url(c.image_url["url"])
                images.append(img)
                items.append({"type": "image", "image": img})
            else:
                raise HTTPException(400, f"unsupported content type: {c.type!r}")
        out.append({"role": m.role, "content": items})
    return out, images


def build_app(model_path: str, device: str = "cuda", dtype: str = "bfloat16"):
    print(f"[vlm_server] loading {model_path} on {device} ({dtype}) ...", flush=True)
    torch_dtype = getattr(torch, dtype)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch_dtype,
        device_map=device,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_path)
    print("[vlm_server] ready.", flush=True)

    app = FastAPI()

    @app.post("/v1/chat/completions")
    def chat(req: ChatRequest):
        qwen_msgs, images = _to_qwen_messages(req.messages)

        text = processor.apply_chat_template(qwen_msgs, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text],
            images=images if images else None,
            return_tensors="pt",
            padding=True,
        ).to(device)

        if req.seed is not None:
            torch.manual_seed(req.seed)

        gen_kwargs = {"max_new_tokens": req.max_tokens, "do_sample": req.temperature > 0}
        if req.temperature > 0:
            gen_kwargs["temperature"] = req.temperature
        with torch.inference_mode():
            output = model.generate(**inputs, **gen_kwargs)
        new_tokens = output[:, inputs["input_ids"].shape[1]:]
        text_out = processor.batch_decode(new_tokens, skip_special_tokens=True)[0]

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model or os.path.basename(model_path.rstrip(os.sep)),
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text_out},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": int(inputs["input_ids"].shape[1]),
                "completion_tokens": int(new_tokens.shape[1]),
                "total_tokens": int(output.shape[1]),
            },
        }

    return app


def main():
    p = argparse.ArgumentParser()
    # Default to the HF repo id so transformers resolves via the HF cache. The
    # repo-side ``checkpoint\Qwen3-VL-8B-Instruct`` directory is left over from
    # an earlier download_models.py run that only fetched 2/4 shards (the
    # script's "directory exists -> skip" guard masked the truncation). HF
    # cache at ~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct is
    # the canonical complete copy.
    p.add_argument("--model-path", default="Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    args = p.parse_args()

    app = build_app(args.model_path, device=args.device, dtype=args.dtype)

    # Shrink anyio's thread-pool. FastAPI runs sync handlers (our /v1/chat/
    # completions is a `def`, not `async def`) in anyio's worker pool, which
    # defaults to 40 threads + their lock primitives. On Windows with Qwen3-VL
    # already resident this leaks enough locks/handles to eventually hit
    # ``RuntimeError: can't allocate lock``. We only ever process one request
    # at a time anyway (model.generate is blocking), so 2 tokens is enough.
    @app.on_event("startup")
    async def _cap_thread_pool():
        import anyio.to_thread
        anyio.to_thread.current_default_thread_limiter().total_tokens = 2

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

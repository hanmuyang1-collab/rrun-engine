# R-Run

**One-command hosting & hot-swap serving engine for HuggingFace models.**
Sibling repo of [R-Build v2 (`rbuild-engine`)](https://github.com/hanmuyang1-collab/rbuild-engine) — R-Build trains the models, R-Run serves them.

## The promise

```bash
rrun swap GeoThinkAI/R-build-20b-a3.4b
```

That's the whole interface for changing models. R-Run will:

1. **wipe** the resident model — weights *and* KV cache fully released,
2. **load** the new model from HuggingFace,
3. **allocate a full KV cache** for it (never a shrunk cache to make swap look faster),
4. do all of it **without restarting the server** — same port, clients keep their connections.

> Why not carry the old KV cache over? KV cache is weights- and
> architecture-specific — it physically cannot transfer between different
> models. So "keeping full KV cache" in R-Run means: *every* swap ends with
> the new model holding a **full-capacity, freshly allocated** cache sized
> for its max context. Swap speed comes from fast wipe + reload, never from
> crippling the cache.

## Install

```bash
pip install git+https://github.com/hanmuyang1-collab/rrun-engine.git
# GPU host (fast path):
pip install "rrun-engine[vllm] @ git+https://github.com/hanmuyang1-collab/rrun-engine.git"
```

## Usage

```bash
rrun serve Qwen/Qwen3-32B            # host a model (starts server on :8000)
rrun swap  GeoThinkAI/R-build-20b-a3.4b   # wipe & swap, one command
rrun status                          # resident model + KV cache state
rrun wipe                            # unload everything, free VRAM
```

OpenAI-compatible API stays live across swaps:

```bash
curl localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "hello"}]
}'
```

Admin endpoints: `POST /admin/swap`, `POST /admin/wipe`, `GET /admin/status`.

## Backends

| backend | when | notes |
|---|---|---|
| `vllm` | GPU host, production | paged attention, `gpu_memory_utilization=0.90` full-cache default |
| `hf` | CPU / no vLLM | plain transformers, `use_cache=True` |
| `mock` | smoke tests / dev | no weights touched |

Auto-detected (`--backend auto`, default): vLLM → HF → mock.

## Smoke test

No GPU required:

```bash
python tests/smoke_test.py
```

Verifies the full swap contract: serve A → generate → swap to B (old cache
wiped, new full KV cache) → generate → second swap → HTTP layer consistency.

## Layout

```
rrun/
├── cli.py            # rrun serve / swap / status / wipe
├── engine.py         # RRunEngine — the swap sequence
├── kvcache.py        # full-wipe / full-allocate KV cache manager
├── server.py         # OpenAI-compatible FastAPI server (never restarts)
└── backends/         # vllm / hf / mock
```

Apache-2.0.

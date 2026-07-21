IMPORTANT: Ensure you’ve thoroughly reviewed the [AGENTS.md](AGENTS.md) file before beginning any work.

> AGENTS.md is upstream llama.cpp's contributor policy (no AI-generated PRs
> *upstream*). This is a **private fork** ("Private forks are exempt"); the
> sections below document what this fork adds and why.

# Fork purpose

This fork is **upstream llama.cpp plus a model-load performance patch set**,
tuned for GPU-only serving. It is consumed by the companion
`wow-look-at-my/ollama` fork, which pins this repo via its `LLAMA_CPP_VERSION`
and serves Gemma 4 models through `llama-server`.

Gemma 4 Multi-Token Prediction (MTP, speculative decoding with Google's
"drafter" model) is served by **upstream's implementation** — the fork's
earlier private `gemma4_assistant` dialect (own arch string, `--mtp-head`,
`llama_model_load_mtp_from_file`, `mtp.*` tensors) was dropped in the
2026-07 uprev in favor of it. The fork no longer carries any MTP code of
its own.

## Gemma 4 MTP (upstream implementation)

- Arch string **`gemma4-assistant`** (hyphen!) ↔ `LLM_ARCH_GEMMA4_ASSISTANT`.
  The drafter is a **separate GGUF** (`general.architecture =
  "gemma4-assistant"`), not embedded in the target file.
- Usage:

  ```
  llama-server -m gemma4-target.gguf --spec-type draft-mtp --spec-draft-model gemma4-assistant.gguf
  ```

  (`--spec-draft-model` aliases `-md`/`--model-draft`; type string
  `"draft-mtp"` is registered in `common/speculative.cpp`.)
- The draft file is loaded as a normal model; the server then creates a draft
  context with `ctx_type = LLAMA_CONTEXT_TYPE_MTP` and `ctx_other = <target
  ctx>` (`llama_context_params.ctx_other`). `LLM_ARCH_GEMMA4_ASSISTANT`
  requires `ctx_other` (`src/llama-context.cpp`; the warning during memory
  fitting is normal). The assistant's iswa KV cache **shares the target's
  memory** — drafter layers map to target layers `n_layer-2` (swa) /
  `n_layer-1` (full) (`src/llama-model.cpp`). Attention is Q-only
  (no `attn_k`/`attn_v`): it cross-attends the target's stored KV.
- Vocab gate: `common_speculative_are_compatible` compares vocab type,
  bos/eos, and token text over the shared range — the assistant GGUF must
  carry the **target's tokenizer**.

### On-disk contract (what the ollama converter must emit)

Source of truth: `src/models/gemma4-assistant.cpp`
(`load_arch_hparams`/`load_arch_tensors`) + the generic loads in
`src/llama-model.cpp` + `src/llama-arch.cpp` (tensor names now come from the
single global `LLM_TENSOR_NAMES` table). The producing converter is
`convert.ConvertGemma4MTPDraft` in `wow-look-at-my/ollama`
(`convert/convert_gemma4.go`) — **keep the two in sync** if tensor names or
KV keys change here. Reference parity: upstream's own Python converter,
`conversion/gemma.py` `Gemma4AssistantModel`.

Notable deltas vs the old fork dialect, and easy-to-miss requirements:

- KV prefix is `gemma4-assistant.*` (hyphen).
- `embedding_length_out` replaces the dialect's `n_embd_backbone` — must
  equal the **target's** hidden size and differ from `embedding_length`
  (`load_arch_tensors` throws if `n_embd_out() == n_embd`).
- `nextn_predict_layers` is new and must equal `block_count`
  (`GGML_ASSERT(n_layer_nextn == n_layer_all)`; the read is "optional" but
  defaults to 0, which asserts).
- Projections are `nextn.pre_projection.weight` `{2*n_embd_out, n_embd}` and
  `nextn.post_projection.weight` `{n_embd, n_embd_out}` (dialect called them
  `mtp.*`).
- `blk.%d.layer_output_scale.weight` is **required on every layer** (the
  dialect treated it as optional).
- Gemma 4's global-vs-sliding split must be emitted in full:
  `attention.head_count_kv` as a per-layer array when full/sliding KV heads
  differ; `attention.key_length`/`value_length` (full head dim, 512 on 31B)
  and `attention.key_length_swa`/`value_length_swa` (sliding head dim, 256;
  **required**); `attention.sliding_window` + `attention.sliding_window_pattern`
  (bool array, len = block_count; required); `attention.shared_kv_layers`.
- `rope.dimension_count_swa` is optional-with-a-silent-trap: it defaults to
  the full-attention `n_rot`, which would RoPE the wrong dim on sliding
  layers — always emit it. One global `rope_freqs.weight`
  `{head_dim_full/2}` (freq-factors trick: `1.0` for the rotated dims,
  `1e30` for the rest).
- `masked_embd_centroids.weight` / `masked_embd_ordering` (note: **no
  `.weight` suffix** on the latter) are accepted-and-**ignored** via the
  loader's unused-tensor path (`GGML_OP_NONE` → "model has unused tensor
  ... ignoring"). Any OTHER unexpected tensor is a hard
  `wrong number of tensors` throw. See TODO.md ("Centroid LM head").
- Dialect-only KV keys with no upstream reader (emit nothing):
  `vocab_size`, `n_embd_backbone`, `use_ordered_embeddings`, `n_centroids`,
  `centroid_top_k`, `attention.k_eq_v`, `requires_target_arch`,
  `final_logit_softcapping`.

**Breaking**: GGUFs built for the old dialect (`general.architecture =
"gemma4_assistant"`, `mtp.*` tensors) do not load on this tree — re-create
them with the updated ollama converter (`ollama create gemma4-mtp ...`).

## Model load path (fork changes)

The loader is tuned for GPU-only serving: no `MAP_POPULATE` (readahead via
`madvise(WILLNEED)` instead), GGUF metadata parsed zero-copy out of the
mapping (`gguf_init_from_buffer_borrow`; string values are `string_view`s),
vocab token texts and BPE merges are views into the retained mapping (the
metadata region of the main file stays mapped for the model's lifetime), and
device-bound weights are uploaded by DMA from the `cudaHostRegister`-pinned
mapping, falling back to the async pinned-staging streamer when registration
is unavailable. Every load phase logs a `timing:` line (grep the server log)
- see TODO.md for the deferred follow-ups (on-disk NUL-terminated strings,
flat tokenizer trie, centroid LM head).

Loader-perf files: `ggml/include/gguf.h` + `ggml/src/gguf.cpp` (borrowed
GGUF contexts, `_view` accessors), `src/llama-mmap.cpp` (no MAP_POPULATE),
`src/llama-model-loader.{h,cpp}` + `src/llama-model.cpp` (mapping pin +
DMA-vs-stream decision, timing), `src/llama-vocab.{h,cpp}` (string_view
token table with a lazy NUL-terminated shadow for the C API),
`src/llama-impl.{h,cpp}` (`gguf_kv_to_str` array preview),
`ggml/src/ggml-cuda/ggml-cuda.cu` (host-register calls ungated from
`GGML_CUDA_REGISTER_HOST`), plus `timing:` log lines in `src/llama.cpp` and
the warmup in `common/common.cpp`.

## CI policy (fork)

- Scheduled (cron) triggers are banned; `no-scheduled-workflows.yml` fails CI
  if one appears. Unused upstream workflows are parked as `*.yml.disabled`
  (rename to disable — do not delete, it keeps future merges trivial).
- The gating build is `build-cuda-ubuntu.yml` job `cuda` (full tree,
  `LLAMA_FATAL_WARNINGS=ON`, `LLAMA_BUILD_TESTS=ON`, CUDA 12.6). CPU-suite
  runs (`ctest -L main`) are local/dev only.
- During an uprev merge, resolve every `.github/workflows` conflict to the
  fork's set (the merge will try to resurrect upstream's active workflows,
  whose `schedule:` triggers then fail the guard).

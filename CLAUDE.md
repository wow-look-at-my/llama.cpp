IMPORTANT: Ensure you’ve thoroughly reviewed the [AGENTS.md](AGENTS.md) file before beginning any work.

> AGENTS.md is upstream llama.cpp's contributor policy (no AI-generated PRs
> *upstream*). This is a **private fork** ("Private forks are exempt"); the
> section below documents what this fork adds and why.

# Fork purpose: Gemma 4 Multi-Token Prediction (MTP)

This fork adds **`gemma4_assistant`**, the llama.cpp port of Google's Gemma 4 MTP
"drafter" (speculative decoding). It is consumed by the companion
`wow-look-at-my/ollama` fork, which pins this repo via its `LLAMA_CPP_VERSION`
and serves Gemma 4 models through `llama-server`.

## Design (mirrors Google's Gemma 4 drafter)

The drafter is a small autoregressive model that proposes several tokens which the
full Gemma 4 **target** verifies in one pass. Three enhancements (per
ai.google.dev/gemma/docs/mtp) shape the on-disk format:

- **Target activations**: the drafter consumes the target's last-layer hidden
  state, concatenated with the token embedding and down-projected
  (`mtp.pre_projection`), then up-projected back for later rounds
  (`mtp.post_projection`).
- **KV-cache sharing**: the drafter **cross-attends the target's KV cache** and
  builds none of its own — so its attention is **Q-only** (`attn_q`/`attn_output`/
  `attn_q_norm`, no `attn_k`/`attn_v`).
- **Centroid LM head** (efficient embedder): `mtp.centroids` +
  `mtp.token_ordering`, gated by `use_ordered_embeddings`.

## On-disk contract (what a `gemma4_assistant` GGUF must contain)

The assistant is a **separate** GGUF (`general.architecture = "gemma4_assistant"`),
NOT embedded in the target file. It is loaded via `llama_model_load_mtp_from_file`
(`src/llama.cpp`) and attached to the target; its vocab must match the target's
(`llama_mtp_vocab_matches` compares token text), so it carries the full tokenizer.

Tensors (see `src/models/gemma4-assistant.cpp` `load_arch_tensors`):
- Global: `token_embd.weight {n_embd, n_vocab}`, `mtp.pre_projection.weight
  {2*n_embd_backbone, n_embd}`, `mtp.post_projection.weight {n_embd,
  n_embd_backbone}`, `output_norm.weight {n_embd}`, and one global
  `rope_freqs.weight {n_embd_head/2}` (the per-layer `tn(ROPE_FREQS,…,i)` calls all
  resolve to the same name, reused via `TENSOR_DUPLICATED`).
- Per layer: `blk.i.{attn_norm,attn_q,attn_output,attn_q_norm,post_attention_norm,
  ffn_norm,ffn_gate,ffn_up,ffn_down,post_ffw_norm}.weight`; optional
  `blk.i.layer_output_scale.weight`.
- If `use_ordered_embeddings`: `mtp.centroids.weight {n_embd, n_centroids}`,
  `mtp.token_ordering.weight {n_vocab}`.

KV keys (`gemma4_assistant.*`; see `src/llama-arch.cpp` and `load_arch_hparams`):
standard `block_count`/`embedding_length`/`feed_forward_length`/
`attention.head_count`, plus Gemma 4's global-vs-local split which the converter
MUST emit in full:
- `attention.head_count_kv` — emit a **per-layer array** when
  `num_global_key_value_heads` differs from `num_key_value_heads` (full vs
  sliding layers), not a scalar.
- `attention.key_length`/`value_length` (global head dim) **and**
  `attention.key_length_swa`/`value_length_swa` (sliding head dim).
- `rope.freq_base`/`rope.dimension_count` (global) **and**
  `rope.freq_base_swa`/`rope.dimension_count_swa` (sliding). Omitting
  `rope.dimension_count_swa` is a *silent* bug: `load_hparams`
  (`src/llama-model.cpp`) defaults `n_rot_swa` to `n_rot_full`, so the drafter
  would RoPE the global head dim on the narrower sliding heads.
- `attention.sliding_window`, `attention.sliding_window_pattern` (`[]bool`,
  len = n_layer, **required**), `attention.shared_kv_layers`.

Assistant-specific: `n_embd_backbone` (**required, must equal the target's
`n_embd`**), `use_ordered_embeddings`, `n_centroids`, `centroid_top_k`,
`attention.k_eq_v`, `requires_target_arch`.

> Cross-checked against Google's `google/gemma-4-31B-it-assistant` checkpoint
> (`Gemma4AssistantForCausalLM`): `backbone_hidden_size` 5376 == target
> `gemma-4-31B-it` `hidden_size`; head dims 256 (sliding) / 512 (full); KV heads
> 16 (sliding) / 4 (full); `requires_target_arch` absent (so the converter omits
> it).

The ollama-side converter that produces this file is
`convert.ConvertGemma4MTPDraft` (`convert/convert_gemma4.go`) — keep the two in
sync if you change tensor names or KV keys here.

## Usage

```
llama-server -m gemma4-target.gguf --spec-type gemma4-mtp --mtp-head gemma4-assistant.gguf
```

## Key files

- `src/llama-arch.{h,cpp}` — `LLM_ARCH_GEMMA4_ASSISTANT`, `mtp.*` tensor names, `LLM_KV_GEMMA4_ASSISTANT_*`
- `src/models/gemma4-assistant.cpp` — hparams/tensor load + the one-step drafter graph (centroid head, in-graph argmax)
- `src/models/models.h` — `llama_model_gemma4::graph_mtp` + its `graph_mtp_params_owner` base (owns the `llm_graph_params` copy so the `llm_graph_context` references it binds don't dangle once `build_arch_graph`'s local copy dies)
- `src/llama.cpp` — `llama_model_load_mtp_from_file`, `llama_mtp_vocab_matches`
- `src/llama-graph.{h,cpp}` — `build_attn_mtp` (cross-attention into the target's KV)
- `common/speculative.cpp`, `common/arg.cpp` — gemma4-mtp speculative driver, `--mtp-head`
- `tests/test-mtp-graph-lifetime.cpp` — CPU-only regression test for the MTP graph-context use-after-free (the `graph_mtp` params-ownership / dangling-reference crash); static-asserts the ownership layout and checks the references survive the source `llm_graph_params` being destroyed

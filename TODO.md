# TODO

## Centroid LM head for the Gemma 4 MTP drafter (dropped in the uprev)

The old fork dialect implemented the drafter's "efficient embedder" centroid
LM head (`mtp.centroids` + `mtp.token_ordering`, gated by
`use_ordered_embeddings`): route the hidden state through centroids, then
score only the top-k token cluster instead of the full vocab matmul.
**Upstream's `gemma4-assistant` has no centroid head** — its logits are a
full `mul_mat` with the tied `token_embd`, and its own converter filters the
HF `masked_embedding.*` tensors out entirely.

Upstream's loader does accept-and-ignore the tensors by name
(`masked_embd_centroids.weight` / `masked_embd_ordering` — note: no
`.weight` suffix on the latter) through the unused-tensor path
(`GGML_OP_NONE` → "model has unused tensor ... ignoring"), so a GGUF
carrying them loads fine today. Re-adding the head therefore needs:

1. converter emission in `wow-look-at-my/ollama` (`convert/convert_gemma4.go`)
   under those exact upstream names,
2. the head implementation in `src/models/gemma4-assistant.cpp` (centroid
   routing + top-k scoring in the drafter graph),
3. a GPU A/B on the 31B checkpoint (draft quality + tok/s with and without
   the head) to prove it is worth carrying — not measurable in a CPU-only
   dev environment.

## GGUF format: strings are length-prefixed but the C API hands out `const char *`

**The mismatch.** GGUF stores every string (metadata values, and critically the
~262k-entry `tokenizer.ggml.tokens` array) as `u64 length + raw bytes`, with
**no NUL terminator** (see `gguf_reader::read(std::string &)` in
`ggml/src/gguf.cpp`). The consuming C API, however, traffics in C strings:

- `gguf_get_val_str()` / `gguf_get_arr_str()` return `const char *`
  (`ggml/include/gguf.h`)
- `llama_vocab_get_text()` returns `const char *` (`include/llama.h`)

Because the on-disk bytes are not NUL-terminated, none of these can ever point
directly at a memory-mapped model file. Every string crossing that boundary
must be copied into owned, NUL-terminated storage first. Internal code uses
`std::string_view` into the retained mapping, but the `const char *` boundary
always forces a copy or a lazily materialized NUL-terminated shadow.

**The fix (future, not now).** The C ABI can't reasonably change, so the
format should: a future GGUF revision (or a fork-local convention — we control
the converter in `wow-look-at-my/ollama` `convert/convert_gemma4.go`) should
**write a single NUL byte after every string's payload**, excluded from the
declared length. Readers that honor the length prefix are unaffected; loaders
that keep the file mapped can then serve `const char *` straight from the
mapping with zero copies and zero shadow buffers. Writer change is one line
per string (`gguf_writer::write(const std::string &)` in
`ggml/src/gguf.cpp`); cost is one byte per string.

Until then: `string_view` into the retained mapping internally, lazy
NUL-terminated shadow at the `const char *` entry points.

## Tokenizer trie: replace `naive_trie`'s per-character `std::map` nodes

`naive_trie` (src/llama-vocab.cpp) allocates one `std::map` red-black-tree node
per character of every inserted token. It is only built for UGM (T5-style),
RWKV, and PLaMo-2 vocabs - BPE/SPM (including Gemma 4) never construct it, and
the `timing: tokenizer initialized` marker confirms ~0 ms there. A flat
(sorted-range or double-array) replacement would cut load time for UGM/RWKV
models, but the repo has no UGM/RWKV tokenizer conformance vocabs
(models/ggml-vocab-*.gguf) to validate a rewrite against, so it was deferred
rather than shipped untested. Add a t5/rwkv vocab + .inp/.out pair first, then
rewrite the three traversal loops against a cursor-style flat trie.

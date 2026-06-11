# TODO

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

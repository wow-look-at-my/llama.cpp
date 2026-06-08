// Regression test for the Gemma 4 MTP graph-context use-after-free.
//
// What crashed
// ------------
// llm_graph_context binds its hparams / cparams / ubatch / cb_func members BY
// REFERENCE to the llm_graph_params it is constructed from (see
// llm_graph_context::llm_graph_context in src/llama-graph.cpp). Those references
// must therefore outlive the graph context.
//
// The Gemma 4 MTP path violated that. llama_model_gemma4::build_arch_graph()
// (src/models/gemma4.cpp) builds the draft graph from a LOCAL llm_graph_params
// copy -- it has to, because it swaps arch/hparams over to the assistant model:
//
//     llm_graph_params p = params;          // local copy
//     p.arch    = mtp_assistant->arch;
//     p.hparams = mtp_assistant->hparams;
//     p.gtype   = LLM_GRAPH_TYPE_MTP;
//     return std::make_unique<graph_mtp>(*this, *mtp_assistant, p);   // p dies here
//
// `p` is destroyed when build_arch_graph() returns, but llama_model::build_graph()
// keeps using the returned context afterwards (build_pooling / build_sampling /
// build_dense_out / set_outputs). Those calls dereference the now-dangling
// references -- most fatally cb_func, a std::function whose captured `this` is
// garbage, leading to model.n_gpu_layers() reading through a wild pointer (general
// protection fault). This is the crash exercised on the MTP reserve/decode path
// (llama_context::ensure_sched_mtp / process_ubatch_mtp -> model.build_graph).
//
// The fix
// -------
// src/models/models.h + src/models/gemma4-assistant.cpp give graph_mtp its own
// copy of the params via a graph_mtp_params_owner base that is initialized BEFORE
// llm_graph_context (base subobjects construct in declaration order), so the
// context's references bind to storage the graph object owns and they stay valid
// for its whole lifetime:
//
//     struct graph_mtp : private graph_mtp_params_owner, public llm_graph_context
//     graph_mtp(...) : graph_mtp_params_owner(params), llm_graph_context(params_owned), ...
//
// What this test checks
// ---------------------
// Pure CPU-side object-lifetime / pointer logic -- no GPU, no model weights, no
// graph compute. It (1) statically asserts the real graph_mtp still has the
// ownership layout the fix introduced, (2) checks graph_mtp_params_owner makes an
// independent deep copy, (3) reproduces the "source params destroyed before the
// context is used" condition through the SAME ownership layout and asserts the
// context's references point into the owned copy (not the dead local) and remain
// usable, and (4) demonstrates that the pre-fix layout would instead bind those
// references to the local copy that build_arch_graph destroys on return.

#include "llama.h"

// internal headers (same approach as test-llama-archs.cpp)
#include "../src/llama-arch.h"
#include "../src/llama-graph.h"
#include "../src/models/models.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <type_traits>

// Always-on check (the test is built with -DNDEBUG, so assert() would be a no-op).
static int g_failures = 0;
#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            fprintf(stderr, "FAILED: %s  (%s:%d)\n", #cond, __FILE__, __LINE__); \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

// ---------------------------------------------------------------------------
// (1) Compile-time guards on the REAL graph_mtp type.
//
// If the fix is reverted (graph_mtp_params_owner removed, graph_mtp back to a
// plain `: public llm_graph_context`), these stop compiling -- the strongest
// possible regression signal.
// ---------------------------------------------------------------------------
using mtp_owner_t = llama_model_gemma4::graph_mtp_params_owner;
using mtp_graph_t = llama_model_gemma4::graph_mtp;

static_assert(std::is_base_of<llm_graph_context, mtp_graph_t>::value,
              "graph_mtp must be an llm_graph_context");
static_assert(std::is_base_of<mtp_owner_t, mtp_graph_t>::value,
              "graph_mtp must own its params via graph_mtp_params_owner (the fix)");
// The owned member must be a full llm_graph_params held BY VALUE (a copy) -- not a
// reference or pointer into the caller's temporary. That copy is the whole point.
static_assert(std::is_same<decltype(mtp_owner_t::params_owned), llm_graph_params>::value,
              "graph_mtp_params_owner must hold an llm_graph_params copy by value");

// ---------------------------------------------------------------------------
// Faithful mirror of graph_mtp's ownership mechanism.
//
// We reuse the REAL graph_mtp_params_owner base and the REAL llm_graph_context,
// in the exact base order and constructor init order as the real graph_mtp. We
// cannot construct the real graph_mtp here (its constructor builds a full ggml
// graph that needs loaded model weights and a populated KV cache), so this mirror
// isolates just the lifetime/ownership mechanism the crash hinged on. Reusing the
// real owner base means this also stops compiling if the fix is reverted.
// ---------------------------------------------------------------------------
struct test_owned_graph : private mtp_owner_t, public llm_graph_context {
    explicit test_owned_graph(const llm_graph_params & p)
        : mtp_owner_t(p), llm_graph_context(params_owned) {}

    const llm_graph_params * owned() const { return &params_owned; }
};

// The pre-fix layout: no owner, so llm_graph_context binds its references directly
// to the constructor argument.
struct test_dangling_graph : public llm_graph_context {
    explicit test_dangling_graph(const llm_graph_params & p) : llm_graph_context(p) {}
};

// ---------------------------------------------------------------------------
// Minimal-but-valid fixtures (so the llm_graph_context constructor's derived
// scalar members can be computed without GGML_ABORT).
// ---------------------------------------------------------------------------
static llama_hparams make_min_hparams(uint32_t n_embd) {
    llama_hparams hp{};
    hp.n_embd             = n_embd;
    hp.n_layer_all        = 1;
    hp.n_layer_nextn      = 0;
    hp.n_head_arr[0]      = 1;
    hp.n_head_kv_arr[0]   = 1;
    hp.is_swa_impl[0]     = 0;   // dense layer 0 -> accessors use the *_full dims
    hp.swa_type           = LLAMA_SWA_TYPE_NONE;
    hp.n_embd_head_k_full = 8;
    hp.n_embd_head_v_full = 8;
    hp.n_rot_full         = 8;
    return hp;
}

static llm_graph_params make_params(llm_graph_result & res,
                                    const llama_hparams & hp,
                                    llm_arch arch,
                                    int * cb_calls) {
    llm_graph_params p{};
    p.arch        = arch;
    p.hparams     = hp;
    p.cparams.n_ctx = 128;
    p.gtype       = LLM_GRAPH_TYPE_DEFAULT;
    p.sched       = nullptr;
    p.backend_cpu = nullptr;
    p.cvec        = nullptr;
    p.loras       = nullptr;
    p.mctx        = nullptr;
    p.cross       = nullptr;
    p.n_outputs   = 1;
    if (cb_calls != nullptr) {
        p.cb = [cb_calls](const llama_ubatch &, ggml_tensor *, const char *, int) {
            ++(*cb_calls);
        };
    }
    p.res = &res;
    return p;
}

// ---------------------------------------------------------------------------
// Mirror of build_arch_graph()'s MTP branch: make a LOCAL copy of params (swapping
// in the assistant hparams + MTP gtype), build the graph object from that local
// copy, and return it. The local copy is destroyed on return -- exactly the
// condition that used to leave the context's references dangling.
// ---------------------------------------------------------------------------
static std::unique_ptr<test_owned_graph> build_owned_like_arch_graph(
        const llm_graph_params & params,
        const llama_hparams & assistant_hparams,
        const void ** out_local_hparams_addr) {
    llm_graph_params p = params;            // local copy (gemma4.cpp build_arch_graph)
    p.arch    = LLM_ARCH_GEMMA4_ASSISTANT;  // swap to the assistant model's arch
    p.hparams = assistant_hparams;          // swap to the assistant model's hparams
    p.gtype   = LLM_GRAPH_TYPE_MTP;
    if (out_local_hparams_addr != nullptr) {
        *out_local_hparams_addr = &p.hparams;
    }
    return std::make_unique<test_owned_graph>(p);   // p dies on return
}

// Same shape, but with the pre-fix layout. Captures the addresses while the local
// copy is still alive (we never dereference the dangling references afterwards).
static void capture_dangling_binding(
        const llm_graph_params & params,
        const llama_hparams & assistant_hparams,
        const void ** out_local_hparams_addr,
        const void ** out_ctx_hparams_addr) {
    llm_graph_params p = params;
    p.hparams = assistant_hparams;
    p.gtype   = LLM_GRAPH_TYPE_MTP;
    test_dangling_graph g(p);
    *out_local_hparams_addr = &p.hparams;
    *out_ctx_hparams_addr   = &g.hparams;   // reference -> &p.hparams
}

// ---------------------------------------------------------------------------
// (2) graph_mtp_params_owner makes an independent deep copy of the params.
// ---------------------------------------------------------------------------
static void test_owner_makes_independent_copy() {
    const int before = g_failures;
    llm_graph_result res(64);
    llama_hparams hp = make_min_hparams(/*n_embd*/ 4242);
    llm_graph_params src = make_params(res, hp, LLM_ARCH_GEMMA4_ASSISTANT, nullptr);

    mtp_owner_t owner(src);   // the REAL fix type

    CHECK(owner.params_owned.hparams.n_embd == 4242);
    CHECK(&owner.params_owned       != &src);          // distinct storage
    CHECK(&owner.params_owned.hparams != &src.hparams); // distinct storage

    // Mutating the source must not touch the owned copy.
    src.hparams.n_embd = 9999;
    CHECK(owner.params_owned.hparams.n_embd == 4242);

    printf("  %s: graph_mtp_params_owner makes an independent deep copy\n",
           g_failures == before ? "ok" : "FAIL");
}

// ---------------------------------------------------------------------------
// (3) The core regression: with the owning layout, the context's references
// survive the source params being destroyed and point into the owned copy.
// ---------------------------------------------------------------------------
static void test_references_survive_source_destruction() {
    const int before = g_failures;
    llm_graph_result res(64);
    const llama_hparams target_hp    = make_min_hparams(111);
    const llama_hparams assistant_hp = make_min_hparams(222);

    int cb_calls = 0;
    llm_graph_params params = make_params(res, target_hp, LLM_ARCH_GEMMA4, &cb_calls);
    params.ubatch.n_tokens = 7;   // sentinel carried through the copy

    const void * local_hparams_addr = nullptr;
    std::unique_ptr<test_owned_graph> ctx =
        build_owned_like_arch_graph(params, assistant_hp, &local_hparams_addr);

    // Everything below mirrors build_graph() continuing to use the context AFTER
    // build_arch_graph() returned and destroyed its local params copy.

    // The references must NOT point into the destroyed local copy...
    CHECK((const void *) &ctx->hparams != local_hparams_addr);

    // ...they must alias the graph object's OWN params copy instead.
    CHECK(&ctx->hparams == &ctx->owned()->hparams);
    CHECK(&ctx->cparams == &ctx->owned()->cparams);
    CHECK(&ctx->ubatch  == &ctx->owned()->ubatch);
    CHECK(&ctx->cb_func == &ctx->owned()->cb);

    // The owned copy captured the assistant arch/hparams build_arch_graph swapped
    // in, and the sentinel ubatch -- reading through the references is safe and
    // correct.
    CHECK(ctx->arch            == LLM_ARCH_GEMMA4_ASSISTANT);
    CHECK(ctx->hparams.n_embd  == 222);
    CHECK(ctx->n_embd          == 222);   // derived scalar, computed in the ctor
    CHECK(ctx->ubatch.n_tokens == 7);

    // Invoking cb() -- as build_pooling() does -- must be safe and dispatch through
    // the owned callback (this is the exact call that GP-faulted pre-fix).
    ctx->cb(nullptr, "mtp_test", -1);
    CHECK(cb_calls == 1);

    printf("  %s: graph context references survive source-params destruction\n",
           g_failures == before ? "ok" : "FAIL");
}

// ---------------------------------------------------------------------------
// (4) Discriminating control: the pre-fix layout DID bind the context's
// references to the local copy that build_arch_graph destroys on return.
// ---------------------------------------------------------------------------
static void test_prefix_layout_binds_to_dying_local() {
    const int before = g_failures;
    llm_graph_result res(64);
    const llama_hparams target_hp    = make_min_hparams(111);
    const llama_hparams assistant_hp = make_min_hparams(333);
    llm_graph_params params = make_params(res, target_hp, LLM_ARCH_GEMMA4, nullptr);

    const void * local_hparams_addr = nullptr;
    const void * ctx_hparams_addr   = nullptr;
    capture_dangling_binding(params, assistant_hp, &local_hparams_addr, &ctx_hparams_addr);

    // Without an owner, the context's hparams reference aliased the LOCAL copy --
    // i.e. exactly the pointer that would dangle once build_arch_graph returned.
    // (Both objects are already destroyed; we compare only addresses captured while
    // they were alive and never dereference them.)
    CHECK(local_hparams_addr != nullptr);
    CHECK(local_hparams_addr == ctx_hparams_addr);

    printf("  %s: pre-fix layout binds references to the dying local copy\n",
           g_failures == before ? "ok" : "FAIL");
}

int main() {
    test_owner_makes_independent_copy();
    test_references_survive_source_destruction();
    test_prefix_layout_binds_to_dying_local();

    if (g_failures != 0) {
        fprintf(stderr, "test-mtp-graph-lifetime: %d check(s) FAILED\n", g_failures);
        return 1;
    }
    printf("test-mtp-graph-lifetime: all checks passed\n");
    return 0;
}

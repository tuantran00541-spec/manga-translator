#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <map>
#include <string>
#include <vector>

struct capture_state {
    std::map<std::string, std::vector<float>> tensors;
    std::string error;
    bool done = false;
    std::string post_ffn_source;
};

static const char * TARGETS[] = {
    "model.input_embed",
    "attn_norm-0",
    "linear_attn_qkv_mixed-0",
    "z-0",
    "beta-0",
    "beta_sigmoid-0",
    "alpha-0",
    "a_softplus-0",
    "gate-0",
    "conv_output_silu-0",
    "q_conv_predelta-0",
    "k_conv_predelta-0",
    "v_conv_predelta-0",
    "linear_attn_out-0",
    "attn_residual-0",
    "attn_post_norm-0",
    "ffn_out-0",
    "post_ffn-0",
};

static bool wanted(const char * name) {
    if (!name) return false;
    for (const char * target : TARGETS) {
        if (std::strcmp(name, target) == 0) return true;
    }
    return false;
}

static bool capture_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * state = static_cast<capture_state *>(user_data);
    const char * name = ggml_get_name(t);
    const bool match = wanted(name);
    if (ask) return match;
    if (!match) return true;

    if (t->type != GGML_TYPE_F32) {
        state->error = std::string("checkpoint ") + (name ? name : "<unnamed>") +
            " has unexpected type " + ggml_type_name(t->type);
        return false;
    }
    const size_t n = ggml_nelements(t);
    std::vector<float> values(n);
    ggml_backend_tensor_get(t, values.data(), 0, n * sizeof(float));
    state->tensors[name] = std::move(values);
    if (name && std::strcmp(name, "post_ffn-0") == 0) {
        state->done = true;
        state->post_ffn_source = "cb_eval";
    }
    return true;
}

static void finalize_layer0(capture_state & state) {
    if (state.done) return;

    // qwen35.cpp defines post_ffn exactly as:
    //   post_ffn = ffn_out + attn_residual
    // The CPU graph optimizer may fuse that add so cb_eval never materializes a
    // tensor named post_ffn-0. Both inputs are observable F32 checkpoints, so
    // reconstruct the exact semantic checkpoint rather than requiring an
    // optimizer-visible node.
    const auto residual = state.tensors.find("attn_residual-0");
    const auto ffn = state.tensors.find("ffn_out-0");
    if (residual == state.tensors.end() || ffn == state.tensors.end()) return;
    if (residual->second.size() != ffn->second.size()) {
        state.error = "attn_residual-0 / ffn_out-0 size mismatch while deriving post_ffn-0";
        return;
    }

    std::vector<float> post_ffn(residual->second.size());
    for (size_t i = 0; i < post_ffn.size(); ++i) {
        post_ffn[i] = residual->second[i] + ffn->second[i];
    }
    state.tensors["post_ffn-0"] = std::move(post_ffn);
    state.done = true;
    state.post_ffn_source = "derived_fp32_add(attn_residual-0,ffn_out-0)";
}

static void write_json(const char * path, llama_token token, int decode_rc, const capture_state & state) {
    std::ofstream out(path);
    out << std::setprecision(9);
    out << "{\n";
    out << "  \"schema\": \"qwen38-llama-layer0-oracle-v1\",\n";
    out << "  \"llama_cpp_revision\": \"557614e0296ff4a5b6f649737a65ae2076eea2fd\",\n";
    out << "  \"token_id\": " << token << ",\n";
    out << "  \"decode_returncode\": " << decode_rc << ",\n";
    out << "  \"captured_complete_layer\": " << (state.done ? "true" : "false") << ",\n";
    out << "  \"post_ffn_source\": \"" << state.post_ffn_source << "\",\n";
    out << "  \"error\": \"" << state.error << "\",\n";
    out << "  \"checkpoints\": {\n";
    size_t ti = 0;
    for (const auto & kv : state.tensors) {
        out << "    \"" << kv.first << "\": [";
        for (size_t i = 0; i < kv.second.size(); ++i) {
            if (i) out << ',';
            out << kv.second[i];
        }
        out << "]" << (++ti == state.tensors.size() ? "\n" : ",\n");
    }
    out << "  }\n";
    out << "}\n";
}

int main(int argc, char ** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s MODEL.gguf OUTPUT.json\n", argv[0]);
        return 2;
    }
    ggml_backend_load_all();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) {
        std::fprintf(stderr, "failed to load model\n");
        return 3;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    llama_token token = llama_vocab_bos(vocab);
    if (token == LLAMA_TOKEN_NULL) {
        std::fprintf(stderr, "model has no BOS token\n");
        llama_model_free(model);
        return 4;
    }

    capture_state state;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 32;
    cp.n_batch = 1;
    cp.n_ubatch = 1;
    cp.n_threads = 2;
    cp.n_threads_batch = 2;
    cp.offload_kqv = false;
    cp.no_perf = true;
    cp.cb_eval = capture_cb;
    cp.cb_eval_user_data = &state;

    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        std::fprintf(stderr, "failed to create context\n");
        llama_model_free(model);
        return 5;
    }

    llama_batch batch = llama_batch_get_one(&token, 1);
    const int rc = llama_decode(ctx, batch);
    finalize_layer0(state);
    write_json(argv[2], token, rc, state);

    llama_free(ctx);
    llama_model_free(model);

    if (!state.error.empty()) {
        std::fprintf(stderr, "%s\n", state.error.c_str());
        return 6;
    }
    if (rc != 0) {
        std::fprintf(stderr, "llama_decode failed rc=%d\n", rc);
        return 8;
    }
    if (!state.done) {
        std::fprintf(stderr, "complete layer-0 checkpoint was not captured/derived; decode rc=%d\n", rc);
        return 7;
    }
    std::fprintf(stderr,
                 "QWEN38_LLAMA_LAYER0_ORACLE token=%d checkpoints=%zu decode_rc=%d post_ffn_source=%s\n",
                 (int) token, state.tensors.size(), rc, state.post_ffn_source.c_str());
    return 0;
}

#pragma once

#include "llama.h"

#include <nlohmann/json.hpp>

#include <array>
#include <cstdint>
#include <map>
#include <regex>
#include <string>
#include <unordered_map>
#include <vector>

enum common_hybrid_backend {
    COMMON_HYBRID_BACKEND_CPU = 0,
    COMMON_HYBRID_BACKEND_NPU = 1,
};

enum common_hybrid_cpu_policy {
    COMMON_HYBRID_CPU_POLICY_CPU_ONLY = 0,
    COMMON_HYBRID_CPU_POLICY_CPU_PREFERRED = 1,
    COMMON_HYBRID_CPU_POLICY_NPU_PREFERRED = 2,
};

enum common_hybrid_tensor_role {
    COMMON_HYBRID_TENSOR_ROLE_OTHER = 0,
    COMMON_HYBRID_TENSOR_ROLE_ATTN = 1,
    COMMON_HYBRID_TENSOR_ROLE_FFN_DENSE = 2,
    COMMON_HYBRID_TENSOR_ROLE_FFN_EXPERT = 3,
    COMMON_HYBRID_TENSOR_ROLE_SHARED_EXPERT = 4,
    COMMON_HYBRID_TENSOR_ROLE_EMBEDDING = 5,
    COMMON_HYBRID_TENSOR_ROLE_OUTPUT = 6,
};

enum common_hybrid_plan_source {
    COMMON_HYBRID_PLAN_SOURCE_MANIFEST = 0,
    COMMON_HYBRID_PLAN_SOURCE_OVERRIDE = 1,
    COMMON_HYBRID_PLAN_SOURCE_LEGACY = 2,
};

struct common_hybrid_model_hint {
    std::string arch;
    std::string name_regex;
    std::string name;
    int32_t n_layer = -1;
};

struct common_hybrid_shape_constraints {
    int32_t k_align = 0;
    int32_t n_align = 0;
    int32_t min_m = 0;
    int32_t min_n = 0;
};

struct common_hybrid_tensor_info {
    std::string name;
    ggml_type type = GGML_TYPE_F32;
    std::array<int64_t, GGML_MAX_DIMS> ne = { 0, 0, 0, 0 };
    int32_t n_dims = 0;
};

struct common_hybrid_rule {
    std::string name;
    std::string match;
    std::regex match_regex;

    bool has_layers = false;
    int32_t layer_start = -1;
    int32_t layer_end = -1;

    common_hybrid_backend backend = COMMON_HYBRID_BACKEND_CPU;
    std::string npu_pipeline;

    std::vector<ggml_type> source_quant_allow;
    common_hybrid_shape_constraints min_shape;

    std::string fallback = "cpu";
    common_hybrid_tensor_role role = COMMON_HYBRID_TENSOR_ROLE_OTHER;
    bool required = false;
};

struct common_hybrid_profile {
    std::string name;
    std::string extends;
    std::vector<common_hybrid_rule> rules;
};

struct common_hybrid_tensor_plan_entry {
    std::string tensor_name;
    int32_t layer_id = -1;
    common_hybrid_tensor_role role = COMMON_HYBRID_TENSOR_ROLE_OTHER;
    common_hybrid_backend backend = COMMON_HYBRID_BACKEND_CPU;
    std::string buffer_type;
    std::string npu_pipeline;
    bool strict = false;
    std::string fallback_reason;
    common_hybrid_plan_source source = COMMON_HYBRID_PLAN_SOURCE_LEGACY;
    std::string rule_name;
    ggml_type source_type = GGML_TYPE_F32;
};

struct common_hybrid_tensor_plan {
    std::vector<common_hybrid_tensor_plan_entry> entries;
    std::unordered_map<std::string, size_t> by_name;

    const common_hybrid_tensor_plan_entry * find(const std::string & name) const;
    bool empty() const { return entries.empty(); }
    std::string describe(size_t max_entries = 0) const;
};

struct common_hybrid_manifest {
    bool loaded = false;
    std::string source_path;
    int32_t version = 1;
    common_hybrid_model_hint model_hint;
    common_hybrid_cpu_policy default_cpu_policy = COMMON_HYBRID_CPU_POLICY_CPU_PREFERRED;
    std::map<std::string, common_hybrid_profile> profiles;
    std::string active_profile;
    std::vector<common_hybrid_rule> resolved_rules;
    nlohmann::ordered_json calibration;

    static common_hybrid_manifest load(const std::string & path, const std::string & profile = {});
    static common_hybrid_manifest load_for_model(const std::string & model_path, const std::string & explicit_path = {}, const std::string & profile = {});

    bool has_manifest() const { return loaded; }
    bool matches_model_hint(const std::string & model_name, int32_t n_layer) const;
    common_hybrid_tensor_plan resolve_plan(const std::vector<common_hybrid_tensor_info> & tensors, bool strict = false) const;
    std::string describe(const common_hybrid_tensor_plan * plan = nullptr) const;
};

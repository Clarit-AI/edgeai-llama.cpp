#include "ggml-impl.h"

#include "rknpu2-configuration.h"
#include "../../../common/log.h"

#include <arm_neon.h>
#include "../../../vendor/nlohmann/json.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <sstream>

using json = nlohmann::ordered_json;

// --- Anonymous namespace for chip-specific packing functions ---

namespace {
using rknpu2_configuration::Rknpu2HybridRule;

bool parse_rule_int_field(const json & obj, const char * key, int & dst, std::string & error, const std::string & field_name) {
    if (!obj.contains(key)) {
        return true;
    }
    if (!obj[key].is_number_integer()) {
        error = field_name + " must be an integer";
        return false;
    }
    dst = obj[key].get<int>();
    return true;
}

// Packing KxN FP16 (row-major: idx [k,n] -> k*N + n) into native RKNN for RK3588: (N/16, K/32, 16, 32)
void pack_B_rk3588_fp16(
    uint8_t* dst_u8, const uint8_t* src_u8,
    int K, int N_total, int n_offset, int n_segment) {

    auto dst = reinterpret_cast<uint16_t*>(dst_u8);
    auto src = reinterpret_cast<const uint16_t*>(src_u8);

    GGML_ASSERT(K % 32 == 0 && N_total > 0 && K > 0);
    GGML_ASSERT(n_offset % 16 == 0 && n_segment % 16 == 0 && n_offset + n_segment <= N_total);

    const int k_segment_limit = 8192;
    size_t packed_base = 0;

    for (int k_base = 0; k_base < K; k_base += k_segment_limit) {
        const int k_segment = std::min(k_segment_limit, K - k_base);
        const size_t s0 = (size_t)(k_segment / 32) * 16 * 32;
        const size_t s1 = 16 * 32;
        const size_t s2 = 32;
        uint16_t * dst_segment = dst + packed_base;

        for (int i = 0; i < n_segment / 16; ++i) {
            for (int j = 0; j < k_segment / 32; ++j) {
                const size_t dst_block = (size_t) i * s0 + (size_t) j * s1;
                for (int ii = 0; ii < 16; ++ii) {
                    const size_t n_global = (size_t)n_offset + (size_t)i * 16 + (size_t)ii;

                    const uint16_t * src_ptr = src + n_global * K + k_base + j * 32;
                    uint16_t * dst_ptr = dst_segment + dst_block + ii * s2;

                    uint16x8_t d0 = vld1q_u16(src_ptr + 0);
                    uint16x8_t d1 = vld1q_u16(src_ptr + 8);
                    uint16x8_t d2 = vld1q_u16(src_ptr + 16);
                    uint16x8_t d3 = vld1q_u16(src_ptr + 24);

                    vst1q_u16(dst_ptr + 0, d0);
                    vst1q_u16(dst_ptr + 8, d1);
                    vst1q_u16(dst_ptr + 16, d2);
                    vst1q_u16(dst_ptr + 24, d3);
                }
            }
        }

        packed_base += (size_t) n_segment * k_segment;
    }
}

// Packing KxN INT8 (row-major) into native RKNN for RK3588: (N/32, K/32, 32, 32)
void pack_B_rk3588_int8(
    uint8_t* dst_u8, const uint8_t* src_u8,
    int K, int N_total, int n_offset, int n_segment) {

    auto dst = reinterpret_cast<int8_t*>(dst_u8);
    auto src = reinterpret_cast<const int8_t*>(src_u8);

    GGML_ASSERT(K % 32 == 0 && N_total > 0 && K > 0);
    GGML_ASSERT(n_offset % 32 == 0 && n_segment % 32 == 0 && n_offset + n_segment <= N_total);

    const int k_segment_limit = 8192;
    size_t packed_base = 0;

    for (int k_base = 0; k_base < K; k_base += k_segment_limit) {
        const int k_segment = std::min(k_segment_limit, K - k_base);
        const size_t s0 = (size_t)(k_segment / 32) * 32 * 32;
        const size_t s1 = 32 * 32;
        const size_t s2 = 32;
        int8_t * dst_segment = dst + packed_base;

        for (int i = 0; i < n_segment / 32; ++i) {
            for (int j = 0; j < k_segment / 32; ++j) {
                const size_t dst_block = (size_t) i * s0 + (size_t) j * s1;
                for (int ii = 0; ii < 32; ++ii) {
                    const size_t n_global = (size_t)n_offset + (size_t)i * 32 + (size_t)ii;

                    const int8_t* src_ptr = src + n_global * K + k_base + j * 32;
                    int8_t* dst_ptr = dst_segment + dst_block + ii * s2;

                    int8x16_t d0 = vld1q_s8(src_ptr);
                    int8x16_t d1 = vld1q_s8(src_ptr + 16);

                    vst1q_s8(dst_ptr, d0);
                    vst1q_s8(dst_ptr + 16, d1);
                }
            }
        }

        packed_base += (size_t) n_segment * k_segment;
    }
}

// Packing KxN INT4 (row-major) into native RKNN for RK3588: (N/64, K/32, 64, 32)
void pack_B_rk3588_int4(
    uint8_t * dst, const uint8_t * src,
    int K, int N_total, int n_offset, int n_segment) {

    GGML_ASSERT(K % 32 == 0 && N_total > 0 && K > 0);
    GGML_ASSERT(n_offset % 64 == 0 && n_segment % 64 == 0 && n_offset + n_segment <= N_total);

    const size_t s0 = (size_t)(K / 32) * 64 * (32 / 2);
    const size_t s1 = 64 * (32 / 2);
    const size_t s2 = (32 / 2); 

    const size_t src_row_stride_bytes = (size_t)K / 2;

    for (int i = 0; i < n_segment / 64; ++i) {
        for (int j = 0; j < K / 32; ++j) {
            const size_t dst_block = (size_t) i * s0 + (size_t) j * s1;
            for (int ii = 0; ii < 64; ++ii) {
                const size_t n_global = (size_t)n_offset + (size_t)i * 64 + (size_t)ii;

                const uint8_t* src_ptr = src + n_global * src_row_stride_bytes + (j * 32) / 2;
                uint8_t* dst_ptr = dst + dst_block + ii * s2;

                uint8x16_t d0 = vld1q_u8(src_ptr);
                vst1q_u8(dst_ptr, d0);
            }
        }
    }
}

} // anonymous namespace

namespace {
    // Function for parsing ENV variable
    std::vector<std::string> split_string(const std::string& str, char delimiter) {
        std::vector<std::string> tokens;
        std::string token;
        std::istringstream tokenStream(str);
        while (std::getline(tokenStream, token, delimiter)) {
            if(!token.empty()) tokens.push_back(token);
        }
        return tokens;
    }

    static std::string to_upper(std::string value) {
        std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return (char) std::toupper(c); });
        return value;
    }

    static bool env_flag_enabled(const char * name) {
        const char * value = std::getenv(name);
        if (value == nullptr || value[0] == '\0') {
            return false;
        }
        const std::string s = to_upper(value);
        return s == "1" || s == "TRUE" || s == "YES" || s == "ON";
    }

    static std::string env_or_empty(const char * name) {
        const char * value = std::getenv(name);
        return value == nullptr ? std::string() : std::string(value);
    }

    static int parse_layer_id(const std::string & tensor_name) {
        static const std::regex layer_re(R"(^blk\.(\d+)\.)");
        std::smatch m;
        if (std::regex_search(tensor_name, m, layer_re)) {
            return std::stoi(m[1].str());
        }
        return -1;
    }

    static std::string derive_role(const std::string & tensor_name) {
        const std::string lower = to_upper(tensor_name);
        if (lower.find("FFN") != std::string::npos && lower.find("EXPS") != std::string::npos) {
            return lower.find("SHARED") != std::string::npos ? "shared_expert" : "ffn_expert";
        }
        if (lower.find("FFN") != std::string::npos) {
            return "ffn_dense";
        }
        if (lower.find("ATTN") != std::string::npos) {
            return "attention";
        }
        if (lower.find("TOKEN_EMBD") != std::string::npos || lower.find("TOK_EMBD") != std::string::npos || lower.find("EMBED") != std::string::npos) {
            return "embeddings";
        }
        if (lower.find("OUTPUT") != std::string::npos || lower.find("LM_HEAD") != std::string::npos) {
            return "output";
        }
        return "other";
    }

    static bool parse_layer_range(const json & value, int & begin, int & end) {
        if (value.is_array() && value.size() >= 2 && value[0].is_number_integer() && value[1].is_number_integer()) {
            begin = value[0].get<int>();
            end = value[1].get<int>();
            return true;
        }
        if (value.is_object()) {
            if (value.contains("begin") && value.contains("end") && value["begin"].is_number_integer() && value["end"].is_number_integer()) {
                begin = value["begin"].get<int>();
                end = value["end"].get<int>();
                return true;
            }
            if (value.contains("start") && value.contains("end") && value["start"].is_number_integer() && value["end"].is_number_integer()) {
                begin = value["start"].get<int>();
                end = value["end"].get<int>();
                return true;
            }
        }
        return false;
    }

    static bool parse_manifest_rule(const json & rule_json, Rknpu2HybridRule & rule, std::string & error) {
        if (!rule_json.is_object()) {
            error = "manifest rule is not an object";
            return false;
        }

        rule.name = rule_json.value("name", std::string());
        const std::string match = rule_json.value("match", rule_json.value("pattern", std::string()));
        if (match.empty()) {
            error = "manifest rule is missing match/pattern";
            return false;
        }

        try {
            rule.match = std::regex(match);
        } catch (const std::regex_error & e) {
            error = std::string("invalid manifest regex '") + match + "': " + e.what();
            return false;
        }

        const std::string target = to_upper(rule_json.value("target", std::string()));
        const std::string backend = to_upper(rule_json.value("backend", target.empty() ? std::string("NPU") : target));
        rule.force_cpu = backend == "CPU" || backend == "CPU_ONLY";
        rule.required = rule_json.value("required", false);
        rule.role = rule_json.value("role", std::string());
        rule.pipeline_name = rule_json.value("npu_pipeline", rule_json.value("pipeline", std::string()));

        if (rule_json.contains("layers")) {
            rule.has_layer_range = parse_layer_range(rule_json["layers"], rule.layer_begin, rule.layer_end);
            if (!rule.has_layer_range) {
                error = "manifest rule has invalid layers range";
                return false;
            }
        }

        const json * shape = nullptr;
        if (rule_json.contains("min_shape")) {
            shape = &rule_json["min_shape"];
        } else if (rule_json.contains("shape")) {
            shape = &rule_json["shape"];
        }

        if (shape != nullptr) {
            if (!shape->is_object()) {
                error = "shape/min_shape must be an object";
                return false;
            }
            if (!parse_rule_int_field(*shape, "k_divisible_by", rule.k_divisible_by, error, "shape/min_shape.k_divisible_by")) {
                return false;
            }
            if (!parse_rule_int_field(*shape, "k_align", rule.k_divisible_by, error, "shape/min_shape.k_align")) {
                return false;
            }
            if (!parse_rule_int_field(*shape, "n_divisible_by", rule.n_divisible_by, error, "shape/min_shape.n_divisible_by")) {
                return false;
            }
            if (!parse_rule_int_field(*shape, "n_align", rule.n_divisible_by, error, "shape/min_shape.n_align")) {
                return false;
            }
            if (shape->contains("min_m") || shape->contains("min_n")) {
                error = "shape/min_shape.min_m and min_n are not supported by the RKNPU backend manifest parser";
                return false;
            }
        }

        if (!parse_rule_int_field(rule_json, "k_align", rule.k_divisible_by, error, "rule.k_align")) {
            return false;
        }
        if (!parse_rule_int_field(rule_json, "n_align", rule.n_divisible_by, error, "rule.n_align")) {
            return false;
        }
        if (rule_json.contains("min_m") || rule_json.contains("min_n")) {
            error = "rule.min_m and min_n are not supported by the RKNPU backend manifest parser";
            return false;
        }

        if (rule_json.contains("source_quant_allow") || rule_json.contains("quant_allow")) {
            const auto & arr = rule_json.contains("source_quant_allow") ? rule_json["source_quant_allow"] : rule_json["quant_allow"];
            if (!arr.is_array()) {
                error = "source_quant_allow/quant_allow must be an array";
                return false;
            }
            for (const auto & item : arr) {
                if (!item.is_string()) {
                    error = "source_quant_allow/quant_allow entries must be strings";
                    return false;
                }
                rule.source_quant_allow.push_back(to_upper(item.get<std::string>()));
            }
        }

        return true;
    }

    static bool load_manifest_rules(
        const std::string & path,
        std::string & profile_name,
        std::string & default_policy,
        std::vector<Rknpu2HybridRule> & rules,
        std::string & error) {

        std::ifstream f(path);
        if (!f) {
            error = "failed to open hybrid manifest: " + path;
            return false;
        }

        json manifest;
        try {
            f >> manifest;
        } catch (const std::exception & e) {
            error = std::string("failed to parse hybrid manifest: ") + e.what();
            return false;
        }

        if (manifest.contains("default_cpu_policy") && manifest["default_cpu_policy"].is_string()) {
            default_policy = to_upper(manifest["default_cpu_policy"].get<std::string>());
        }

        json profile_obj = manifest;
        std::string effective_profile = profile_name;
        if (effective_profile.empty() && manifest.contains("active_profile") && manifest["active_profile"].is_string()) {
            effective_profile = manifest["active_profile"].get<std::string>();
        }

        if (manifest.contains("profiles") && manifest["profiles"].is_object()) {
            const auto & profiles = manifest["profiles"];
            if (!effective_profile.empty() && profiles.contains(effective_profile)) {
                profile_obj = profiles[effective_profile];
            } else if (!effective_profile.empty()) {
                // explicitly requested profile not found
                error = "hybrid manifest profile '" + effective_profile + "' not found";
                return false;
            } else if (profiles.contains("default")) {
                profile_obj = profiles["default"];
            } else if (!profiles.empty()) {
                profile_obj = profiles.begin().value();
                effective_profile = profiles.begin().key();
            } else {
                error = "hybrid manifest profiles object is empty";
                return false;
            }
        }

        if (profile_obj.contains("default_cpu_policy") && profile_obj["default_cpu_policy"].is_string()) {
            default_policy = to_upper(profile_obj["default_cpu_policy"].get<std::string>());
        }

        const json * rules_src = nullptr;
        if (profile_obj.contains("rules")) {
            rules_src = &profile_obj["rules"];
        } else if (profile_obj.contains("routes")) {
            rules_src = &profile_obj["routes"];
        } else if (manifest.contains("rules")) {
            rules_src = &manifest["rules"];
        } else if (manifest.contains("routes")) {
            rules_src = &manifest["routes"];
        }
        if (rules_src == nullptr || !rules_src->is_array()) {
            error = "hybrid manifest does not contain a rules/routes array";
            return false;
        }

        rules.clear();
        rules.reserve(rules_src->size());
        for (const auto & item : *rules_src) {
            Rknpu2HybridRule rule;
            if (!parse_manifest_rule(item, rule, error)) {
                return false;
            }
            rules.push_back(std::move(rule));
        }

        if (profile_name.empty()) {
            profile_name = effective_profile.empty() ? "default" : effective_profile;
        }

        return true;
    }

    static bool source_type_allowed(const Rknpu2HybridRule & rule, ggml_type type) {
        if (rule.source_quant_allow.empty()) {
            return true;
        }
        const std::string needle = to_upper(ggml_type_name(type));
        return std::find(rule.source_quant_allow.begin(), rule.source_quant_allow.end(), needle) != rule.source_quant_allow.end();
    }

    static bool rule_matches_tensor(const Rknpu2HybridRule & rule, const struct ggml_tensor * tensor, int layer_id) {
        if (tensor == nullptr) {
            return false;
        }

        const std::string name = tensor->name != nullptr ? tensor->name : "";
        if (!std::regex_search(name, rule.match)) {
            return false;
        }

        if (rule.k_divisible_by > 0 && tensor->ne[0] % rule.k_divisible_by != 0) {
            return false;
        }
        if (rule.n_divisible_by > 0 && tensor->ne[1] % rule.n_divisible_by != 0) {
            return false;
        }

        if (rule.has_layer_range && layer_id >= 0) {
            if (layer_id < rule.layer_begin || layer_id > rule.layer_end) {
                return false;
            }
        }

        return source_type_allowed(rule, tensor->type);
    }
} // anonymous namespace

namespace rknpu2_configuration {

Rknpu2ConfigManager& Rknpu2ConfigManager::get_instance() {
    static Rknpu2ConfigManager instance;
    return instance;
}

const std::vector<std::string>* Rknpu2DeviceConfig::get_active_pattern(int tensor_type) const {
    auto it = default_patterns.find(tensor_type);
    if (it == default_patterns.end()) {
        return nullptr;
    }

    if (use_custom_pattern && !custom_hybrid_pattern.empty()) {
        // Validate that at least one pipeline in custom_hybrid_pattern is compatible
        // with the allowed pipelines for this tensor type
        const auto& allowed_pipelines = it->second;
        bool has_compatible_pipeline = false;

        for (const auto& custom_pipeline : custom_hybrid_pattern) {
            for (const auto& allowed_pipeline : allowed_pipelines) {
                if (custom_pipeline == allowed_pipeline) {
                    has_compatible_pipeline = true;
                    break;
                }
            }
            if (has_compatible_pipeline) break;
        }

        // Only use custom pattern if at least one pipeline is compatible
        if (has_compatible_pipeline) {
            return &custom_hybrid_pattern;
        }
    }

    return &it->second;
}

const Rknpu2HardwarePipeline* Rknpu2DeviceConfig::find_pipeline(const std::string & name) const {
    for (const auto & pipe : hardware_pipelines) {
        if (pipe.pipeline_name == name) {
            return &pipe;
        }
    }
    return nullptr;
}

const Rknpu2HybridRoute* Rknpu2DeviceConfig::resolve_explicit_route(const struct ggml_tensor * w_tensor) const {
    if (w_tensor == nullptr) {
        return nullptr;
    }

    std::string name = w_tensor->name != nullptr ? w_tensor->name : "";
    if (name.empty()) {
        name = "ptr_" + std::to_string(reinterpret_cast<uintptr_t>(w_tensor));
    }

    std::lock_guard<std::mutex> lock(*pattern_mutex);
    auto cache_key = std::make_pair(current_model_id, name);
    auto it = explicit_route_cache.find(cache_key);
    if (it == explicit_route_cache.end()) {
        return nullptr;
    }
    return &it->second;
}

void Rknpu2DeviceConfig::clear_explicit_routes() const {
    std::lock_guard<std::mutex> lock(*pattern_mutex);
    explicit_route_cache.clear();
}

void Rknpu2DeviceConfig::register_explicit_route(const std::string & tensor_name, const std::string & pipeline_name, bool strict) const {
    std::lock_guard<std::mutex> lock(*pattern_mutex);

    Rknpu2HybridRoute route;
    route.from_loader = true;
    route.strict = strict;
    route.layer_id = parse_layer_id(tensor_name);
    route.role = derive_role(tensor_name);
    route.rule_name = "loader";

    const auto * pipeline = find_pipeline(pipeline_name);
    if (pipeline == nullptr) {
        route.valid = false;
        route.fallback_reason = "unknown explicit pipeline";
        if (strict) {
            throw std::runtime_error("loader hybrid route references unknown RKNPU pipeline '" + pipeline_name + "'");
        }
    } else {
        route.valid = true;
        route.pipeline_name = pipeline_name;
        route.pipeline = pipeline;
    }

    auto cache_key = std::make_pair(current_model_id, tensor_name);
    explicit_route_cache[cache_key] = std::move(route);
}

const Rknpu2HybridRoute* Rknpu2DeviceConfig::resolve_manifest_route(const struct ggml_tensor * w_tensor) const {
    if (!hybrid_manifest_loaded || w_tensor == nullptr) {
        return nullptr;
    }

    std::string name = w_tensor->name != nullptr ? w_tensor->name : "";
    if (name.empty()) {
        name = "ptr_" + std::to_string(reinterpret_cast<uintptr_t>(w_tensor));
    }

    std::lock_guard<std::mutex> lock(*pattern_mutex);
    auto cache_key = std::make_pair(current_model_id, name);
    auto it = hybrid_route_cache.find(cache_key);
    if (it != hybrid_route_cache.end()) {
        return &it->second;
    }

    Rknpu2HybridRoute route;
    route.layer_id = parse_layer_id(name);
    route.role = derive_role(name);

    for (const auto & rule : hybrid_manifest_rules) {
        if (!rule_matches_tensor(rule, w_tensor, route.layer_id)) {
            continue;
        }

        route.from_manifest = true;
        route.strict = hybrid_manifest_strict || rule.required;
        route.rule_name = rule.name.empty() ? rule.pipeline_name : rule.name;
        if (!rule.role.empty()) {
            route.role = rule.role;
        }

        if (rule.force_cpu) {
            route.force_cpu = true;
            route.valid = true;
            route.fallback_reason = "manifest rule requests CPU";
            hybrid_route_cache.emplace(cache_key, route);
            if (hybrid_manifest_dump_plan) {
                LOG_INF("RKNPU2: manifest route %s -> CPU (%s, role=%s, layer=%d)\n",
                    name.c_str(), route.rule_name.c_str(), route.role.c_str(), route.layer_id);
            }
            return &hybrid_route_cache.find(cache_key)->second;
        }

        if (rule.pipeline_name.empty()) {
            route.fallback_reason = "manifest rule has no pipeline";
            if (route.strict) {
                throw std::runtime_error("hybrid manifest rule '" + route.rule_name + "' matched tensor '" + name + "' but did not name an NPU pipeline");
            }
            hybrid_route_cache.emplace(cache_key, route);
            return &hybrid_route_cache.find(cache_key)->second;
        }

        const auto * pipeline = find_pipeline(rule.pipeline_name);
        if (pipeline == nullptr) {
            route.fallback_reason = "unknown manifest pipeline";
            if (route.strict) {
                throw std::runtime_error("hybrid manifest rule '" + route.rule_name + "' references unknown pipeline '" + rule.pipeline_name + "'");
            }
            hybrid_route_cache.emplace(cache_key, route);
            return &hybrid_route_cache.find(cache_key)->second;
        }

        route.valid = true;
        route.pipeline_name = pipeline->pipeline_name;
        route.pipeline = pipeline;
        hybrid_route_cache.emplace(cache_key, route);
        if (hybrid_manifest_dump_plan) {
            LOG_INF("RKNPU2: manifest route %s -> %s (%s, role=%s, layer=%d)\n",
                name.c_str(), route.pipeline_name.c_str(), route.rule_name.c_str(), route.role.c_str(), route.layer_id);
        }
        return &hybrid_route_cache.find(cache_key)->second;
    }

    return nullptr;
}

void Rknpu2DeviceConfig::dump_hybrid_manifest_summary() const {
    if (!hybrid_manifest_loaded) {
        return;
    }

    LOG_INF("RKNPU2: hybrid manifest path=%s profile=%s default_policy=%s rules=%zu strict=%d dump_plan=%d\n",
        hybrid_manifest_path.c_str(),
        hybrid_manifest_profile.c_str(),
        hybrid_manifest_default_policy.c_str(),
        hybrid_manifest_rules.size(),
        hybrid_manifest_strict ? 1 : 0,
        hybrid_manifest_dump_plan ? 1 : 0);
}

const Rknpu2HardwarePipeline* Rknpu2DeviceConfig::resolve_op_support(const struct ggml_tensor* w_tensor) const {
    if (!w_tensor) return nullptr;

    if (const auto * route = resolve_explicit_route(w_tensor)) {
        if (route->force_cpu || route->pipeline == nullptr || !route->valid) {
            return nullptr;
        }
        return route->pipeline;
    }

    if (hybrid_manifest_loaded) {
        const auto * route = resolve_manifest_route(w_tensor);
        if (route != nullptr) {
            if (route->force_cpu || route->pipeline == nullptr || !route->valid) {
                return nullptr;
            }
            // Return manifest-selected pipeline instead of falling through to legacy chooser
            return route->pipeline;
        }

        if (to_upper(hybrid_manifest_default_policy) == "CPU_ONLY") {
            return nullptr;
        }
    }

    // Retrieve active quantization pattern based on tensor type (or custom ENV variable)
    const std::vector<std::string>* pattern_ptr = get_active_pattern((int)w_tensor->type);
    
    // If no pattern is registered for this type and no global override exists, reject operation
    if (!pattern_ptr || pattern_ptr->empty()) {
        return nullptr;
    }

    const auto& pattern = *pattern_ptr;

    // Acquiring the lock on the pattern mutex for thread-safe tensor tracking
    std::lock_guard<std::mutex> lock(*pattern_mutex);

    // Retrieving the unique tensor name
    std::string name = w_tensor->name;
    if (name.empty()) {
        name = "ptr_" + std::to_string(reinterpret_cast<uintptr_t>(w_tensor));
    }

    // Assigning the next sequence number if this tensor is seen for the first time
    if (tensor_sequence_map.find(name) == tensor_sequence_map.end()) {
        tensor_sequence_map[name] = global_tensor_counter++;
    }

    // Selecting the pipeline cyclically based on the defined pattern
    int seq_id = tensor_sequence_map[name];
    size_t pattern_idx = seq_id % pattern.size();

    const std::string& selected_pipeline = pattern[pattern_idx];
    const auto* pipeline = find_pipeline(selected_pipeline);

    // If no hardware pipeline exists with this name, reject operation
    if (!pipeline) {
        return nullptr;
    }

    return pipeline;
}

Rknpu2ConfigManager::Rknpu2ConfigManager() {
    // Reading custom hybrid pattern ENV variable
    const char* env_pattern = std::getenv("HYBRID_PATTERN");
    bool use_custom_pattern = false;
    std::vector<std::string> custom_pattern;

    if (env_pattern != nullptr) {
        custom_pattern = split_string(env_pattern, ',');
        use_custom_pattern = true;
    }

    // --- Define RK3588 Configuration ---
    Rknpu2DeviceConfig rk3588_config;
    rk3588_config.device_name = "RK3588";
    rk3588_config.core_count = 3;
    rk3588_config.hardware_pipelines = {
        {
            /* .pipeline_name = */ "FP16_STANDARD",
            /* .npu_type_a    = */ NPU_TYPE_FP16,
            /* .npu_type_c    = */ NPU_TYPE_FP32,
            /* .mm_type       = */ RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32,
            /* .k_align       = */ 32,
            /* .n_align       = */ 16,
            /* .pack_func     = */ pack_B_rk3588_fp16,
            /* .use_hadamard  = */ false
        },
        {
            /* .pipeline_name = */ "FP16_HADAMARD",
            /* .npu_type_a    = */ NPU_TYPE_FP16,
            /* .npu_type_c    = */ NPU_TYPE_FP32,
            /* .mm_type       = */ RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32,
            /* .k_align       = */ 32,
            /* .n_align       = */ 16,
            /* .pack_func     = */ pack_B_rk3588_fp16,
            /* .use_hadamard  = */ true
        },
        {
            /* .pipeline_name = */ "INT8_STANDARD",
            /* .npu_type_a    = */ NPU_TYPE_INT8,
            /* .npu_type_c    = */ NPU_TYPE_INT32,
            /* .mm_type       = */ RKNN_INT8_MM_INT8_TO_INT32,
            /* .k_align       = */ 32,
            /* .n_align       = */ 32,
            /* .pack_func     = */ pack_B_rk3588_int8,
            /* .use_hadamard  = */ false
        },
        {
            /* .pipeline_name = */ "INT8_HADAMARD",
            /* .npu_type_a    = */ NPU_TYPE_INT8,
            /* .npu_type_c    = */ NPU_TYPE_INT32,
            /* .mm_type       = */ RKNN_INT8_MM_INT8_TO_INT32,
            /* .k_align       = */ 32,
            /* .n_align       = */ 32,
            /* .pack_func     = */ pack_B_rk3588_int8,
            /* .use_hadamard  = */ true
        },
        {
            /* .pipeline_name = */ "INT4_STANDARD",
            /* .npu_type_a    = */ NPU_TYPE_INT4,
            /* .npu_type_c    = */ NPU_TYPE_INT16,
            /* .mm_type       = */ RKNN_INT4_MM_INT4_TO_INT16,
            /* .k_align       = */ 32,
            /* .n_align       = */ 64,
            /* .pack_func     = */ pack_B_rk3588_int4,
            /* .use_hadamard  = */ false
        },
        {
            /* .pipeline_name = */ "INT4_HADAMARD",
            /* .npu_type_a    = */ NPU_TYPE_INT4,
            /* .npu_type_c    = */ NPU_TYPE_INT16,
            /* .mm_type       = */ RKNN_INT4_MM_INT4_TO_INT16,
            /* .k_align       = */ 32,
            /* .n_align       = */ 64,
            /* .pack_func     = */ pack_B_rk3588_int4,
            /* .use_hadamard  = */ true
        }
    };
    
    // Assigning custom variables
    rk3588_config.use_custom_pattern = use_custom_pattern;
    rk3588_config.custom_hybrid_pattern = custom_pattern;

    // Defining default quantization sequences for each supported ggml_type
    rk3588_config.default_patterns[(int)GGML_TYPE_F16]  = {"FP16_STANDARD"};
    rk3588_config.default_patterns[(int)GGML_TYPE_Q8_0] = {"INT8_STANDARD"};
    rk3588_config.default_patterns[(int)GGML_TYPE_Q6_K] = {"INT8_STANDARD", "INT4_HADAMARD"};
    rk3588_config.default_patterns[(int)GGML_TYPE_Q4_0] = {"INT4_HADAMARD"};

    // Optional env-driven hybrid manifest. This is a compatibility layer for the
    // current workspace slice where loader/common plumbing is intentionally left untouched.
    rk3588_config.hybrid_manifest_path = env_or_empty("HYBRID_MANIFEST");
    rk3588_config.hybrid_manifest_profile = env_or_empty("HYBRID_PROFILE");
    rk3588_config.hybrid_manifest_strict = env_flag_enabled("HYBRID_STRICT");
    rk3588_config.hybrid_manifest_dump_plan = env_flag_enabled("HYBRID_DUMP_PLAN");

    if (!rk3588_config.hybrid_manifest_path.empty()) {
        std::string error;
        if (load_manifest_rules(
                rk3588_config.hybrid_manifest_path,
                rk3588_config.hybrid_manifest_profile,
                rk3588_config.hybrid_manifest_default_policy,
                rk3588_config.hybrid_manifest_rules,
                error)) {
            rk3588_config.hybrid_manifest_loaded = true;
            // Generate unique model ID to avoid cache collisions across models
            rk3588_config.current_model_id = rk3588_config.hybrid_manifest_path + ":" + std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
        } else {
            if (rk3588_config.hybrid_manifest_strict) {
                throw std::runtime_error(error);
            }
            LOG_WRN("RKNPU2: ignoring hybrid manifest '%s': %s\n",
                rk3588_config.hybrid_manifest_path.c_str(), error.c_str());
        }
    }

    device_configs["RK3588"] = rk3588_config;

    // --- Define RK3588S Configuration ---
    // RK3588S is a variant with 2 NPU cores (vs 3 on RK3588)
    // Uses the same pipelines as RK3588
    Rknpu2DeviceConfig rk3588s_config = rk3588_config;  // Copy from RK3588
    rk3588s_config.device_name = "RK3588S";
    rk3588s_config.core_count = 2;
    device_configs["RK3588S"] = rk3588s_config;

    // --- Define RK3576 Configuration ---
    // RK3576 has 3 NPU cores, same architecture as RK3588
    // Uses the same pipelines as RK3588
    Rknpu2DeviceConfig rk3576_config = rk3588_config;  // Copy from RK3588
    rk3576_config.device_name = "RK3576";
    rk3576_config.core_count = 3;
    device_configs["RK3576"] = rk3576_config;

    // --- Define RK3566 Configuration (Placeholder) ---
    // Rknpu2DeviceConfig rk3566_config;
    // ... fill config for RK3566 ...
    // device_configs["RK3566"] = rk3566_config;

    // Select a default device with explicit deterministic fallback order
    if (!device_configs.empty()) {
        // Prefer RK3588, then RK3588S, then RK3576, otherwise first available
        if (device_configs.find("RK3588") != device_configs.end()) {
            select_device("RK3588");
        } else if (device_configs.find("RK3588S") != device_configs.end()) {
            select_device("RK3588S");
        } else if (device_configs.find("RK3576") != device_configs.end()) {
            select_device("RK3576");
        } else {
            // Fall back to first available device
            select_device(device_configs.begin()->first);
        }
    }
}

bool Rknpu2ConfigManager::select_device(const std::string& device_name) {
    auto it = device_configs.find(device_name);
    if (it != device_configs.end()) {
        // Reset tensor sequence state for this device to ensure consistent hybrid pattern
        // selection for each new model load
        it->second.tensor_sequence_map.clear();
        it->second.global_tensor_counter = 0;
        it->second.hybrid_route_cache.clear();
        current_config = &it->second;
        if (current_config->hybrid_manifest_loaded && current_config->hybrid_manifest_dump_plan) {
            current_config->dump_hybrid_manifest_summary();
        }
        return true;
    }
    return false;
}

void Rknpu2ConfigManager::clear_explicit_routes() {
    if (current_config != nullptr) {
        current_config->clear_explicit_routes();
    }
}

void Rknpu2ConfigManager::register_explicit_route(const std::string & tensor_name, const std::string & pipeline_name, bool strict) {
    if (current_config != nullptr) {
        current_config->register_explicit_route(tensor_name, pipeline_name, strict);
    }
}

const Rknpu2DeviceConfig& Rknpu2ConfigManager::get_current_config() const {
    GGML_ASSERT(current_config != nullptr && "No device configuration selected or available.");
    return *current_config;
}

void set_split_factor(int factor) {
    Rknpu2ConfigManager::get_instance().set_split_factor_internal(factor);
}

} // namespace rknpu2_configuration

#include "hybrid-manifest.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

using json = nlohmann::ordered_json;

namespace {

static std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

static std::string trim_copy(const std::string & s) {
    size_t begin = 0;
    while (begin < s.size() && std::isspace(static_cast<unsigned char>(s[begin]))) {
        ++begin;
    }
    size_t end = s.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(s[end - 1]))) {
        --end;
    }
    return s.substr(begin, end - begin);
}

static std::string upper_copy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    return s;
}

static bool contains_type(const std::vector<ggml_type> & types, ggml_type type) {
    return std::find(types.begin(), types.end(), type) != types.end();
}

static common_hybrid_backend parse_backend(const std::string & value) {
    const std::string v = to_lower(trim_copy(value));
    if (v == "cpu") return COMMON_HYBRID_BACKEND_CPU;
    if (v == "npu") return COMMON_HYBRID_BACKEND_NPU;
    throw std::runtime_error("invalid hybrid backend: " + value);
}

static std::string backend_to_string(common_hybrid_backend backend) {
    switch (backend) {
        case COMMON_HYBRID_BACKEND_CPU: return "cpu";
        case COMMON_HYBRID_BACKEND_NPU: return "npu";
    }
    return "cpu";
}

static common_hybrid_cpu_policy parse_cpu_policy(const std::string & value) {
    const std::string v = to_lower(trim_copy(value));
    if (v == "cpu_only") return COMMON_HYBRID_CPU_POLICY_CPU_ONLY;
    if (v == "cpu_preferred") return COMMON_HYBRID_CPU_POLICY_CPU_PREFERRED;
    if (v == "npu_preferred") return COMMON_HYBRID_CPU_POLICY_NPU_PREFERRED;
    throw std::runtime_error("invalid hybrid default_cpu_policy: " + value);
}

static std::string cpu_policy_to_string(common_hybrid_cpu_policy policy) {
    switch (policy) {
        case COMMON_HYBRID_CPU_POLICY_CPU_ONLY: return "cpu_only";
        case COMMON_HYBRID_CPU_POLICY_CPU_PREFERRED: return "cpu_preferred";
        case COMMON_HYBRID_CPU_POLICY_NPU_PREFERRED: return "npu_preferred";
    }
    return "cpu_preferred";
}

static common_hybrid_tensor_role parse_role(const std::string & value) {
    const std::string v = to_lower(trim_copy(value));
    if (v == "other") return COMMON_HYBRID_TENSOR_ROLE_OTHER;
    if (v == "attn" || v == "attention") return COMMON_HYBRID_TENSOR_ROLE_ATTN;
    if (v == "ffn_dense" || v == "dense_ffn") return COMMON_HYBRID_TENSOR_ROLE_FFN_DENSE;
    if (v == "ffn_expert" || v == "expert_ffn") return COMMON_HYBRID_TENSOR_ROLE_FFN_EXPERT;
    if (v == "shared_expert") return COMMON_HYBRID_TENSOR_ROLE_SHARED_EXPERT;
    if (v == "embedding" || v == "embeddings") return COMMON_HYBRID_TENSOR_ROLE_EMBEDDING;
    if (v == "output") return COMMON_HYBRID_TENSOR_ROLE_OUTPUT;
    throw std::runtime_error("invalid hybrid tensor role: " + value);
}

static std::string role_to_string(common_hybrid_tensor_role role) {
    switch (role) {
        case COMMON_HYBRID_TENSOR_ROLE_OTHER: return "other";
        case COMMON_HYBRID_TENSOR_ROLE_ATTN: return "attn";
        case COMMON_HYBRID_TENSOR_ROLE_FFN_DENSE: return "ffn_dense";
        case COMMON_HYBRID_TENSOR_ROLE_FFN_EXPERT: return "ffn_expert";
        case COMMON_HYBRID_TENSOR_ROLE_SHARED_EXPERT: return "shared_expert";
        case COMMON_HYBRID_TENSOR_ROLE_EMBEDDING: return "embedding";
        case COMMON_HYBRID_TENSOR_ROLE_OUTPUT: return "output";
    }
    return "other";
}

static ggml_type parse_ggml_type(const std::string & value) {
    const std::string v = to_lower(trim_copy(value));
    if (v == "f32" || v == "fp32") return GGML_TYPE_F32;
    if (v == "f16" || v == "fp16") return GGML_TYPE_F16;
    if (v == "bf16") return GGML_TYPE_BF16;
    if (v == "q8_0") return GGML_TYPE_Q8_0;
    if (v == "q6_k") return GGML_TYPE_Q6_K;
    if (v == "q4_0") return GGML_TYPE_Q4_0;
    if (v == "q8_kv") return GGML_TYPE_Q8_KV;
    if (v == "q8_0_r8") return GGML_TYPE_Q8_0_R8;
    if (v == "q6_k_r4") return GGML_TYPE_Q6_K_R4;
    if (v == "q4_0_r8") return GGML_TYPE_Q4_0_R8;
    if (v == "q8_kv_r8") return GGML_TYPE_Q8_KV_R8;
    if (v == "bf16_r16") return GGML_TYPE_BF16_R16;
    throw std::runtime_error("invalid hybrid source_quant_allow type: " + value);
}

static std::string ggml_type_to_manifest_name(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32: return "f32";
        case GGML_TYPE_F16: return "f16";
        case GGML_TYPE_BF16: return "bf16";
        case GGML_TYPE_Q8_0: return "q8_0";
        case GGML_TYPE_Q6_K: return "q6_k";
        case GGML_TYPE_Q4_0: return "q4_0";
        case GGML_TYPE_Q8_KV: return "q8_kv";
        case GGML_TYPE_Q8_0_R8: return "q8_0_r8";
        case GGML_TYPE_Q6_K_R4: return "q6_k_r4";
        case GGML_TYPE_Q4_0_R8: return "q4_0_r8";
        case GGML_TYPE_Q8_KV_R8: return "q8_kv_r8";
        case GGML_TYPE_BF16_R16: return "bf16_r16";
        default: return ggml_type_name(type);
    }
}

static common_hybrid_tensor_role infer_role_from_name(const std::string & name) {
    const std::string lower = to_lower(name);
    if (lower.find("token_embd") != std::string::npos || lower.find("embed") != std::string::npos) {
        return COMMON_HYBRID_TENSOR_ROLE_EMBEDDING;
    }
    if (lower.find("output") != std::string::npos || lower.find("lm_head") != std::string::npos) {
        return COMMON_HYBRID_TENSOR_ROLE_OUTPUT;
    }
    if (lower.find("shared") != std::string::npos && lower.find("expert") != std::string::npos) {
        return COMMON_HYBRID_TENSOR_ROLE_SHARED_EXPERT;
    }
    if (lower.find("exps") != std::string::npos || lower.find("expert") != std::string::npos) {
        return COMMON_HYBRID_TENSOR_ROLE_FFN_EXPERT;
    }
    if (lower.find("ffn") != std::string::npos || lower.find("feed_forward") != std::string::npos) {
        return COMMON_HYBRID_TENSOR_ROLE_FFN_DENSE;
    }
    if (lower.find("attn") != std::string::npos || lower.find("attention") != std::string::npos) {
        return COMMON_HYBRID_TENSOR_ROLE_ATTN;
    }
    return COMMON_HYBRID_TENSOR_ROLE_OTHER;
}

static int32_t infer_layer_id(const std::string & name) {
    static const std::regex layer_regex(R"(blk\.(\d+)\.)");
    std::smatch match;
    if (std::regex_search(name, match, layer_regex)) {
        return std::stoi(match[1].str());
    }
    return -1;
}

static bool shape_satisfies(const common_hybrid_shape_constraints & constraints, const common_hybrid_tensor_info & tensor) {
    if (constraints.k_align > 0 && (tensor.n_dims < 1 || tensor.ne[0] % constraints.k_align != 0)) {
        return false;
    }
    if (constraints.n_align > 0 && (tensor.n_dims < 2 || tensor.ne[1] % constraints.n_align != 0)) {
        return false;
    }
    if (constraints.min_n > 0 && (tensor.n_dims < 1 || tensor.ne[0] < constraints.min_n)) {
        return false;
    }
    if (constraints.min_m > 0 && (tensor.n_dims < 2 || tensor.ne[1] < constraints.min_m)) {
        return false;
    }
    return true;
}

static bool quant_type_allowed(const common_hybrid_rule & rule, ggml_type type) {
    return rule.source_quant_allow.empty() || contains_type(rule.source_quant_allow, type);
}

static bool rule_role_matches(const common_hybrid_rule & rule, common_hybrid_tensor_role role) {
    return !rule.role.has_value() || rule.role.value() == role;
}

static bool rule_layer_matches(const common_hybrid_rule & rule, int32_t layer_id) {
    if (!rule.has_layers) {
        return true;
    }
    return layer_id >= rule.layer_start && layer_id <= rule.layer_end;
}

static common_hybrid_tensor_plan_entry make_default_entry(const common_hybrid_tensor_info & tensor, common_hybrid_cpu_policy policy) {
    common_hybrid_tensor_plan_entry entry;
    entry.tensor_name = tensor.name;
    entry.layer_id = infer_layer_id(tensor.name);
    entry.role = infer_role_from_name(tensor.name);
    entry.source_type = tensor.type;
    entry.source = COMMON_HYBRID_PLAN_SOURCE_LEGACY;

    if (policy == COMMON_HYBRID_CPU_POLICY_NPU_PREFERRED) {
        switch (tensor.type) {
            case GGML_TYPE_F32:
            case GGML_TYPE_F16:
            case GGML_TYPE_BF16:
            case GGML_TYPE_Q8_0:
            case GGML_TYPE_Q8_0_R8:
            case GGML_TYPE_Q8_KV:
            case GGML_TYPE_Q8_KV_R8:
            case GGML_TYPE_Q6_K:
            case GGML_TYPE_Q6_K_R4:
            case GGML_TYPE_Q4_0:
            case GGML_TYPE_Q4_0_R8:
                entry.backend = COMMON_HYBRID_BACKEND_NPU;
                entry.buffer_type = "NPU";
                if (tensor.type == GGML_TYPE_Q4_0 || tensor.type == GGML_TYPE_Q4_0_R8) {
                    entry.npu_pipeline = "INT4_HADAMARD";
                } else if (tensor.type == GGML_TYPE_F32 || tensor.type == GGML_TYPE_F16 || tensor.type == GGML_TYPE_BF16) {
                    entry.npu_pipeline = "FP16_STANDARD";
                } else {
                    entry.npu_pipeline = "INT8_STANDARD";
                }
                break;
            default:
                entry.backend = COMMON_HYBRID_BACKEND_CPU;
                entry.buffer_type = "CPU";
                entry.fallback_reason = "source type not npu-preferred compatible";
                break;
        }
        return entry;
    }

    entry.backend = COMMON_HYBRID_BACKEND_CPU;
    entry.buffer_type = "CPU";
    return entry;
}

static std::vector<common_hybrid_rule> resolve_rules(
        const std::map<std::string, common_hybrid_profile> & profiles,
        const std::string & profile_name,
        std::unordered_set<std::string> & visiting) {
    const auto it = profiles.find(profile_name);
    if (it == profiles.end()) {
        throw std::runtime_error("unknown hybrid profile: " + profile_name);
    }
    if (!visiting.insert(profile_name).second) {
        throw std::runtime_error("hybrid profile cycle detected at: " + profile_name);
    }

    std::vector<common_hybrid_rule> rules;
    if (!it->second.extends.empty()) {
        auto parent_rules = resolve_rules(profiles, it->second.extends, visiting);
        rules.insert(rules.end(), parent_rules.begin(), parent_rules.end());
    }
    rules.insert(rules.end(), it->second.rules.begin(), it->second.rules.end());
    visiting.erase(profile_name);
    return rules;
}

static common_hybrid_rule parse_rule(const json & rule_json) {
    if (!rule_json.is_object()) {
        throw std::runtime_error("hybrid rule must be an object");
    }

    common_hybrid_rule rule;
    if (rule_json.contains("name")) {
        rule.name = rule_json.at("name").get<std::string>();
    }
    if (rule_json.contains("match")) {
        rule.match = rule_json.at("match").get<std::string>();
    }
    if (rule.match.empty()) {
        throw std::runtime_error("hybrid manifest rule missing 'match'");
    }
    rule.match_regex = std::regex(rule.match);

    if (rule_json.contains("layers") && !rule_json.at("layers").is_null()) {
        const auto & layers = rule_json.at("layers");
        if (!layers.is_array() || layers.size() != 2) {
            throw std::runtime_error("hybrid manifest rule 'layers' must be a 2-item array");
        }
        rule.has_layers = true;
        rule.layer_start = layers.at(0).get<int32_t>();
        rule.layer_end = layers.at(1).get<int32_t>();
        if (rule.layer_start > rule.layer_end) {
            throw std::runtime_error("hybrid manifest rule layer range is inverted");
        }
    }

    bool backend_explicitly_set = rule_json.contains("backend");
    if (backend_explicitly_set) {
        rule.backend = parse_backend(rule_json.at("backend").get<std::string>());
    }
    if (rule_json.contains("npu_pipeline") && !rule_json.at("npu_pipeline").is_null()) {
        rule.npu_pipeline = rule_json.at("npu_pipeline").get<std::string>();
        // If npu_pipeline exists and backend was not explicitly set (or is still CPU default), set to NPU
        if (!backend_explicitly_set || rule.backend == COMMON_HYBRID_BACKEND_CPU) {
            rule.backend = COMMON_HYBRID_BACKEND_NPU;
        } else if (rule.backend != COMMON_HYBRID_BACKEND_NPU) {
            // Backend explicitly set to non-NPU while npu_pipeline is present
            throw std::runtime_error("hybrid manifest rule has npu_pipeline but backend is not NPU");
        }
    }
    if (rule.backend == COMMON_HYBRID_BACKEND_NPU && rule.npu_pipeline.empty()) {
        throw std::runtime_error("hybrid manifest NPU rule requires npu_pipeline");
    }

    if (rule_json.contains("source_quant_allow") && !rule_json.at("source_quant_allow").is_null()) {
        const auto & allowed = rule_json.at("source_quant_allow");
        if (!allowed.is_array()) {
            throw std::runtime_error("hybrid manifest rule 'source_quant_allow' must be an array");
        }
        for (const auto & item : allowed) {
            if (!item.is_string()) {
                throw std::runtime_error("hybrid manifest source_quant_allow entries must be strings");
            }
            rule.source_quant_allow.push_back(parse_ggml_type(item.get<std::string>()));
        }
    }

    if (rule_json.contains("min_shape") && !rule_json.at("min_shape").is_null()) {
        const auto & ms = rule_json.at("min_shape");
        if (!ms.is_object()) {
            throw std::runtime_error("hybrid manifest rule 'min_shape' must be an object");
        }
        if (ms.contains("k_align")) rule.min_shape.k_align = ms.at("k_align").get<int32_t>();
        if (ms.contains("n_align")) rule.min_shape.n_align = ms.at("n_align").get<int32_t>();
        if (ms.contains("min_m")) rule.min_shape.min_m = ms.at("min_m").get<int32_t>();
        if (ms.contains("min_n")) rule.min_shape.min_n = ms.at("min_n").get<int32_t>();
    }

    if (rule_json.contains("fallback")) {
        rule.fallback = rule_json.at("fallback").get<std::string>();
    }
    if (rule_json.contains("role") && !rule_json.at("role").is_null()) {
        rule.role = parse_role(rule_json.at("role").get<std::string>());
    }
    if (rule_json.contains("required")) {
        rule.required = rule_json.at("required").get<bool>();
    }

    return rule;
}

static common_hybrid_model_hint parse_model_hint(const json & root) {
    common_hybrid_model_hint hint;
    if (!root.contains("model_hint") || root.at("model_hint").is_null()) {
        return hint;
    }
    const auto & mh = root.at("model_hint");
    if (!mh.is_object()) {
        throw std::runtime_error("hybrid manifest 'model_hint' must be an object");
    }
    if (mh.contains("arch")) hint.arch = mh.at("arch").get<std::string>();
    if (mh.contains("name_regex")) {
        hint.name_regex = mh.at("name_regex").get<std::string>();
        (void) std::regex(hint.name_regex);
    }
    if (mh.contains("name")) hint.name = mh.at("name").get<std::string>();
    if (mh.contains("n_layer")) hint.n_layer = mh.at("n_layer").get<int32_t>();
    return hint;
}

static void validate_manifest_schema(const json & root) {
    if (!root.is_object()) {
        throw std::runtime_error("hybrid manifest root must be an object");
    }
    if (root.contains("version") && !root.at("version").is_number_integer()) {
        throw std::runtime_error("hybrid manifest 'version' must be an integer");
    }
    if (root.contains("default_cpu_policy") && !root.at("default_cpu_policy").is_string()) {
        throw std::runtime_error("hybrid manifest 'default_cpu_policy' must be a string");
    }
    if (root.contains("active_profile") && !root.at("active_profile").is_string()) {
        throw std::runtime_error("hybrid manifest 'active_profile' must be a string");
    }
    if (!root.contains("profiles") || !root.at("profiles").is_object()) {
        throw std::runtime_error("hybrid manifest requires a profiles object");
    }
}

} // namespace

const common_hybrid_tensor_plan_entry * common_hybrid_tensor_plan::find(const std::string & name) const {
    const auto it = by_name.find(name);
    if (it == by_name.end()) {
        return nullptr;
    }
    return &entries[it->second];
}

std::string common_hybrid_tensor_plan::describe(size_t max_entries) const {
    std::ostringstream os;
    std::map<std::string, size_t> backend_counts;
    std::map<std::string, size_t> role_counts;

    for (const auto & entry : entries) {
        backend_counts[backend_to_string(entry.backend)]++;
        role_counts[role_to_string(entry.role)]++;
    }

    os << "hybrid plan: tensors=" << entries.size();
    os << " cpu=" << backend_counts["cpu"];
    os << " npu=" << backend_counts["npu"];
    os << "\nroles:";
    for (const auto & [role, count] : role_counts) {
        os << " " << role << "=" << count;
    }
    os << "\n";

    const size_t limit = max_entries == 0 ? entries.size() : std::min(max_entries, entries.size());
    for (size_t i = 0; i < limit; ++i) {
        const auto & entry = entries[i];
        os << " - " << entry.tensor_name
           << " role=" << role_to_string(entry.role)
           << " backend=" << backend_to_string(entry.backend)
           << " pipeline=" << (entry.npu_pipeline.empty() ? "-" : entry.npu_pipeline)
           << " source=" << (entry.source == COMMON_HYBRID_PLAN_SOURCE_MANIFEST ? "manifest" : entry.source == COMMON_HYBRID_PLAN_SOURCE_OVERRIDE ? "override" : "legacy");
        if (!entry.rule_name.empty()) {
            os << " rule=" << entry.rule_name;
        }
        if (!entry.fallback_reason.empty()) {
            os << " fallback=" << entry.fallback_reason;
        }
        os << "\n";
    }

    if (limit < entries.size()) {
        os << " ... " << (entries.size() - limit) << " more entries\n";
    }
    return os.str();
}

common_hybrid_manifest common_hybrid_manifest::load(const std::string & path, const std::string & profile) {
    if (path.empty()) {
        return {};
    }

    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("failed to open hybrid manifest: " + path);
    }

    json root;
    file >> root;
    validate_manifest_schema(root);

    common_hybrid_manifest manifest;
    manifest.loaded = true;
    manifest.source_path = path;
    manifest.version = root.value("version", 1);
    if (manifest.version != 1) {
        throw std::runtime_error("unsupported hybrid manifest version: " + std::to_string(manifest.version));
    }
    manifest.model_hint = parse_model_hint(root);
    manifest.default_cpu_policy = parse_cpu_policy(root.value("default_cpu_policy", std::string("cpu_preferred")));
    if (root.contains("calibration")) {
        manifest.calibration = root.at("calibration");
    }

    for (const auto & [profile_name, profile_json] : root.at("profiles").items()) {
        if (!profile_json.is_object()) {
            throw std::runtime_error("hybrid profile must be an object: " + profile_name);
        }
        common_hybrid_profile parsed;
        parsed.name = profile_name;
        if (profile_json.contains("extends") && !profile_json.at("extends").is_null()) {
            parsed.extends = profile_json.at("extends").get<std::string>();
        }
        if (!profile_json.contains("rules") || !profile_json.at("rules").is_array()) {
            throw std::runtime_error("hybrid profile missing rules array: " + profile_name);
        }
        for (const auto & rule_json : profile_json.at("rules")) {
            common_hybrid_rule rule = parse_rule(rule_json);
            parsed.rules.push_back(std::move(rule));
        }
        manifest.profiles.emplace(profile_name, std::move(parsed));
    }

    const std::string selected_profile = !profile.empty() ? profile : root.value("active_profile", std::string{});
    if (!selected_profile.empty()) {
        manifest.active_profile = selected_profile;
    } else if (manifest.profiles.size() == 1) {
        manifest.active_profile = manifest.profiles.begin()->first;
    } else {
        throw std::runtime_error("hybrid manifest requires active_profile or explicit --hybrid-profile when multiple profiles exist");
    }

    std::unordered_set<std::string> visiting;
    manifest.resolved_rules = resolve_rules(manifest.profiles, manifest.active_profile, visiting);
    return manifest;
}

common_hybrid_manifest common_hybrid_manifest::load_for_model(const std::string & model_path, const std::string & explicit_path, const std::string & profile) {
    if (!explicit_path.empty()) {
        return load(explicit_path, profile);
    }

    const std::string candidate = model_path + ".hybrid.json";
    std::ifstream file(candidate);
    if (!file) {
        return {};
    }
    return load(candidate, profile);
}

bool common_hybrid_manifest::matches_model_hint(const std::string & arch, const std::string & model_name, int32_t n_layer) const {
    if (!loaded) {
        return true;
    }
    if (!model_hint.arch.empty() && to_lower(model_hint.arch) != to_lower(arch)) {
        return false;
    }
    if (!model_hint.name.empty() && to_lower(model_hint.name) != to_lower(model_name)) {
        return false;
    }
    if (!model_hint.name_regex.empty()) {
        if (!std::regex_search(model_name, std::regex(model_hint.name_regex))) {
            return false;
        }
    }
    if (model_hint.n_layer >= 0 && n_layer >= 0 && model_hint.n_layer != n_layer) {
        return false;
    }
    return true;
}

common_hybrid_tensor_plan common_hybrid_manifest::resolve_plan(const std::vector<common_hybrid_tensor_info> & tensors, bool strict) const {
    common_hybrid_tensor_plan plan;
    plan.entries.reserve(tensors.size());

    for (const auto & tensor : tensors) {
        common_hybrid_tensor_plan_entry entry = make_default_entry(tensor, default_cpu_policy);
        const common_hybrid_rule * matched_rule = nullptr;

        for (const auto & rule : resolved_rules) {
            if (!std::regex_search(tensor.name, rule.match_regex)) {
                continue;
            }
            if (!rule_role_matches(rule, entry.role)) {
                continue;
            }
            if (!rule_layer_matches(rule, entry.layer_id)) {
                continue;
            }
            if (!quant_type_allowed(rule, tensor.type)) {
                continue;
            }
            if (!shape_satisfies(rule.min_shape, tensor)) {
                continue;
            }
            matched_rule = &rule;
            break;
        }

        if (matched_rule != nullptr) {
            entry.source = COMMON_HYBRID_PLAN_SOURCE_MANIFEST;
            entry.rule_name = matched_rule->name;
            entry.backend = matched_rule->backend;
            entry.strict = matched_rule->required || strict;
            if (matched_rule->backend == COMMON_HYBRID_BACKEND_CPU) {
                entry.buffer_type = "CPU";
                if (entry.fallback_reason.empty()) {
                    entry.fallback_reason = matched_rule->fallback;
                }
            } else {
                entry.buffer_type = "NPU";
                entry.npu_pipeline = matched_rule->npu_pipeline;
            }
        } else if (default_cpu_policy == COMMON_HYBRID_CPU_POLICY_NPU_PREFERRED) {
            if (entry.backend == COMMON_HYBRID_BACKEND_CPU) {
                entry.strict = strict;
                if (strict) {
                    throw std::runtime_error("hybrid manifest strict mode: no NPU-compatible route for tensor " + tensor.name);
                }
                entry.fallback_reason = "no compatible npu route";
            }
        }

        if (strict && entry.backend == COMMON_HYBRID_BACKEND_NPU && entry.npu_pipeline.empty()) {
            throw std::runtime_error("hybrid manifest strict mode: missing NPU pipeline for tensor " + tensor.name);
        }

        if (entry.buffer_type.empty()) {
            entry.buffer_type = entry.backend == COMMON_HYBRID_BACKEND_CPU ? "CPU" : "NPU";
        }

        plan.by_name[entry.tensor_name] = plan.entries.size();
        plan.entries.push_back(std::move(entry));
    }

    return plan;
}

std::string common_hybrid_manifest::describe(const common_hybrid_tensor_plan * plan) const {
    std::ostringstream os;
    os << "hybrid manifest";
    if (!loaded) {
        os << ": not loaded\n";
        return os.str();
    }

    os << ": version=" << version
       << " source=" << source_path
       << " active_profile=" << active_profile
       << " default_cpu_policy=" << cpu_policy_to_string(default_cpu_policy)
       << " rules=" << resolved_rules.size()
       << "\n";

    if (!model_hint.arch.empty() || !model_hint.name.empty() || !model_hint.name_regex.empty() || model_hint.n_layer >= 0) {
        os << "model_hint:";
        if (!model_hint.arch.empty()) os << " arch=" << model_hint.arch;
        if (!model_hint.name.empty()) os << " name=" << model_hint.name;
        if (!model_hint.name_regex.empty()) os << " name_regex=" << model_hint.name_regex;
        if (model_hint.n_layer >= 0) os << " n_layer=" << model_hint.n_layer;
        os << "\n";
    }

    for (const auto & [name, profile_obj] : profiles) {
        os << "profile " << name;
        if (!profile_obj.extends.empty()) {
            os << " extends=" << profile_obj.extends;
        }
        os << " rules=" << profile_obj.rules.size() << "\n";
    }

    if (!calibration.is_null()) {
        os << "calibration=" << calibration.dump() << "\n";
    }

    if (plan != nullptr) {
        os << plan->describe(0);
    }

    return os.str();
}
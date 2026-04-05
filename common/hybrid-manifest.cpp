#include "hybrid-manifest.h"

#include "common.h"

#define JSON_ASSERT GGML_ASSERT
#include <nlohmann/json.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <regex>
#include <set>
#include <string_view>
#include <stdexcept>

using json = nlohmann::ordered_json;

namespace common_hybrid_manifest {
namespace {

constexpr const char * TARGET_CPU     = "cpu";
constexpr const char * TARGET_DEFAULT = "default";

const std::set<std::string> & allowed_profiles() {
    static const std::set<std::string> profiles = {
        "dense-balanced",
        "dense-cpu-only",
        "moe-balanced",
    };
    return profiles;
}

const std::set<std::string> & known_pipelines() {
    static const std::set<std::string> pipelines = {
        "FP16_STANDARD",
        "FP16_HADAMARD",
        "INT8_STANDARD",
        "INT8_HADAMARD",
        "INT4_STANDARD",
        "INT4_HADAMARD",
    };
    return pipelines;
}

const std::set<std::string> & npu_compatible_quants() {
    static const std::set<std::string> quants = {
        "f16",
        "q4_0",
        "q6_k",
        "q8_0",
    };
    return quants;
}

bool file_exists(const std::string & path) {
    std::error_code ec;
    return std::filesystem::exists(path, ec);
}

void validate_positive_divisor(int value, const std::string & key, const std::string & rule_name) {
    if (value < 0) {
        throw std::runtime_error(format("hybrid manifest rule '%s' has invalid %s=%d", rule_name.c_str(), key.c_str(), value));
    }
    if (value == 0) {
        throw std::runtime_error(format("hybrid manifest rule '%s' has impossible %s=%d", rule_name.c_str(), key.c_str(), value));
    }
}

void validate_pipeline_name(const std::string & pipeline, const std::string & ctx) {
    if (pipeline.empty()) {
        return;
    }
    if (!known_pipelines().count(pipeline)) {
        throw std::runtime_error(format("hybrid manifest %s references unknown pipeline '%s'", ctx.c_str(), pipeline.c_str()));
    }
}

void validate_quant_allow(const route_rule & rule) {
    if (rule.quant_allow.empty()) {
        return;
    }

    for (const auto & quant : rule.quant_allow) {
        if (rule.target != TARGET_CPU && npu_compatible_quants().count(string_lower(quant)) == 0) {
            throw std::runtime_error(format(
                "hybrid manifest rule '%s' has incompatible quant allowlist for target '%s'",
                rule.name.c_str(),
                rule.target.c_str()));
        }
    }
}

std::optional<int> parse_generated_rule_number(std::string_view name) {
    constexpr std::string_view prefix = "rule-";
    if (!string_starts_with(std::string(name), std::string(prefix))) {
        return std::nullopt;
    }

    const std::string_view suffix = name.substr(prefix.size());
    if (suffix.empty()) {
        return std::nullopt;
    }
    for (char c : suffix) {
        if (c < '0' || c > '9') {
            return std::nullopt;
        }
    }

    return std::stoi(std::string(suffix));
}

bool plan_entry_less(const plan_entry & lhs, const plan_entry & rhs) {
    const auto lhs_num = parse_generated_rule_number(lhs.name);
    const auto rhs_num = parse_generated_rule_number(rhs.name);

    if (lhs_num.has_value() && rhs_num.has_value() && lhs_num.value() != rhs_num.value()) {
        return lhs_num.value() < rhs_num.value();
    }
    if (lhs_num.has_value() != rhs_num.has_value()) {
        return lhs_num.has_value();
    }
    if (lhs.name != rhs.name) {
        return lhs.name < rhs.name;
    }
    return lhs.match < rhs.match;
}

route_rule parse_route_rule(const json & item, int index) {
    route_rule rule;
    rule.name   = item.value("name", "rule-" + std::to_string(index));
    rule.match  = item.value("match", "");
    rule.target = string_lower(item.value("target", std::string(TARGET_DEFAULT)));
    rule.pipeline = item.value("pipeline", "");

    if (rule.match.empty()) {
        throw std::runtime_error(format("hybrid manifest rule '%s' is missing 'match'", rule.name.c_str()));
    }

    if (rule.target != TARGET_CPU && rule.target != TARGET_DEFAULT) {
        throw std::runtime_error(format("hybrid manifest rule '%s' has unsupported target '%s'", rule.name.c_str(), rule.target.c_str()));
    }

    try {
        (void) std::regex(rule.match);
    } catch (const std::regex_error & e) {
        throw std::runtime_error(format("hybrid manifest rule '%s' has invalid regex: %s", rule.name.c_str(), e.what()));
    }

    validate_pipeline_name(rule.pipeline, "route");

    if (item.contains("quant_allow")) {
        rule.quant_allow = item.at("quant_allow").get<std::vector<std::string>>();
    }

    if (item.contains("shape")) {
        const auto & shape = item.at("shape");
        if (shape.contains("k_divisible_by")) {
            rule.shape.k_divisible_by = shape.at("k_divisible_by").get<int>();
            validate_positive_divisor(rule.shape.k_divisible_by, "shape.k_divisible_by", rule.name);
        }
        if (shape.contains("n_divisible_by")) {
            rule.shape.n_divisible_by = shape.at("n_divisible_by").get<int>();
            validate_positive_divisor(rule.shape.n_divisible_by, "shape.n_divisible_by", rule.name);
        }
    }

    validate_quant_allow(rule);
    return rule;
}

} // namespace

std::string default_manifest_path(const std::string & model_path) {
    return model_path + ".hybrid.json";
}

std::optional<std::string> discover_manifest_path(const std::string & model_path, const std::string & explicit_path) {
    if (!explicit_path.empty()) {
        return explicit_path;
    }

    const std::string sidecar_path = default_manifest_path(model_path);
    if (file_exists(sidecar_path)) {
        return sidecar_path;
    }

    return std::nullopt;
}

manifest load_manifest(const std::string & path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error(format("failed to open hybrid manifest '%s'", path.c_str()));
    }

    json manifest_json = json::parse(file);

    manifest result;
    result.source_path = path;
    result.version = manifest_json.value("version", 1);
    result.profile = manifest_json.value("profile", "");
    result.strict  = manifest_json.value("strict", false);

    if (result.version != 1) {
        throw std::runtime_error(format("hybrid manifest '%s' has unsupported version %d", path.c_str(), result.version));
    }

    if (!allowed_profiles().count(result.profile)) {
        throw std::runtime_error(format("hybrid manifest '%s' has unknown profile '%s'", path.c_str(), result.profile.c_str()));
    }

    if (manifest_json.contains("npu_pattern")) {
        result.npu_pattern = manifest_json.at("npu_pattern").get<std::vector<std::string>>();
    }

    for (const auto & pipeline : result.npu_pattern) {
        validate_pipeline_name(pipeline, "npu_pattern");
    }

    if (!manifest_json.contains("routes") || !manifest_json.at("routes").is_array()) {
        throw std::runtime_error(format("hybrid manifest '%s' must contain a 'routes' array", path.c_str()));
    }

    int index = 0;
    for (const auto & item : manifest_json.at("routes")) {
        result.routes.push_back(parse_route_rule(item, index++));
    }

    if (result.routes.empty()) {
        throw std::runtime_error(format("hybrid manifest '%s' does not define any routes", path.c_str()));
    }

    return result;
}

plan resolve_plan(const manifest & manifest_data) {
    plan resolved_plan;
    resolved_plan.source_path = manifest_data.source_path;
    resolved_plan.profile     = manifest_data.profile;
    resolved_plan.strict      = manifest_data.strict;
    resolved_plan.npu_pattern = manifest_data.npu_pattern;

    for (const auto & rule : manifest_data.routes) {
        plan_entry entry;
        entry.name        = rule.name;
        entry.match       = rule.match;
        entry.target      = rule.target;
        entry.pipeline    = rule.pipeline;
        entry.quant_allow = rule.quant_allow;
        entry.shape       = rule.shape;
        resolved_plan.entries.push_back(entry);
    }

    std::stable_sort(resolved_plan.entries.begin(), resolved_plan.entries.end(), plan_entry_less);
    for (size_t i = 0; i < resolved_plan.entries.size(); ++i) {
        resolved_plan.entries[i].precedence = int(i) + 1;
    }

    return resolved_plan;
}

std::string get_hybrid_pattern_env(const plan & resolved_plan) {
    if (resolved_plan.npu_pattern.empty()) {
        return "";
    }
    return string_join(resolved_plan.npu_pattern, ",");
}

json plan_to_json(const plan & resolved_plan) {
    json entries = json::array();
    for (const auto & entry : resolved_plan.entries) {
        json item = {
            {"name", entry.name},
            {"match", entry.match},
            {"target", entry.target},
            {"precedence", entry.precedence},
        };
        if (!entry.pipeline.empty()) {
            item["pipeline"] = entry.pipeline;
        }
        if (!entry.quant_allow.empty()) {
            item["quant_allow"] = entry.quant_allow;
        }
        if (entry.shape.k_divisible_by > 0 || entry.shape.n_divisible_by > 0) {
            item["shape"] = json::object();
            if (entry.shape.k_divisible_by > 0) {
                item["shape"]["k_divisible_by"] = entry.shape.k_divisible_by;
            }
            if (entry.shape.n_divisible_by > 0) {
                item["shape"]["n_divisible_by"] = entry.shape.n_divisible_by;
            }
        }
        entries.push_back(std::move(item));
    }

    json output = {
        {"source_path", resolved_plan.source_path},
        {"profile", resolved_plan.profile},
        {"strict", resolved_plan.strict},
        {"npu_pattern", resolved_plan.npu_pattern},
        {"routes", std::move(entries)},
    };
    const std::string env = get_hybrid_pattern_env(resolved_plan);
    if (!env.empty()) {
        output["hybrid_pattern_env"] = env;
    }
    return output;
}

} // namespace common_hybrid_manifest

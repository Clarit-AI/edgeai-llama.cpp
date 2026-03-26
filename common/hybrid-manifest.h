#pragma once

#include <nlohmann/json_fwd.hpp>

#include <optional>
#include <string>
#include <vector>

namespace common_hybrid_manifest {

struct shape_requirements {
    int k_divisible_by = 0;
    int n_divisible_by = 0;
};

struct route_rule {
    std::string name;
    std::string match;
    std::string target;
    std::string pipeline;
    std::vector<std::string> quant_allow;
    shape_requirements shape;
};

struct manifest {
    int version = 1;
    std::string profile;
    bool strict = false;
    std::vector<std::string> npu_pattern;
    std::vector<route_rule> routes;
    std::string source_path;
};

struct plan_entry {
    std::string name;
    std::string match;
    std::string target;
    std::string pipeline;
    std::vector<std::string> quant_allow;
    shape_requirements shape;
    int precedence = 1;
};

struct plan {
    std::string source_path;
    std::string profile;
    bool strict = false;
    std::vector<std::string> npu_pattern;
    std::vector<plan_entry> entries;
};

std::string default_manifest_path(const std::string & model_path);
std::optional<std::string> discover_manifest_path(const std::string & model_path, const std::string & explicit_path);
manifest load_manifest(const std::string & path);
plan resolve_plan(const manifest & manifest_data);
std::string get_hybrid_pattern_env(const plan & resolved_plan);
nlohmann::ordered_json plan_to_json(const plan & resolved_plan);

} // namespace common_hybrid_manifest

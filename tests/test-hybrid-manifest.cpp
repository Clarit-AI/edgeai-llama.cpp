#include "hybrid-manifest.h"
#include "testing.h"

#include <nlohmann/json.hpp>

int main() {
    testing t;

    const std::string base = "tests/fixtures/hybrid-manifest/";

    t.test("loads example manifests", [&](const testing &) {
        const std::vector<std::string> files = {
            base + "examples/dense-balanced.json",
            base + "examples/dense-cpu-only.json",
            base + "examples/moe-balanced.json",
            base + "examples/strict-mode-validation-expected-pass.json",
        };

        for (const auto & file : files) {
            auto manifest = common_hybrid_manifest::load_manifest(file);
            auto plan = common_hybrid_manifest::resolve_plan(manifest);
            auto dumped = common_hybrid_manifest::plan_to_json(plan).dump(2);

            if (dumped.empty()) {
                throw std::runtime_error("expected non-empty plan dump for " + file);
            }
        }
    });

    t.test("plan dump is deterministic", [&](const testing &) {
        const auto manifest = common_hybrid_manifest::load_manifest(base + "examples/moe-balanced.json");
        const auto plan_a = common_hybrid_manifest::resolve_plan(manifest);
        const auto plan_b = common_hybrid_manifest::resolve_plan(manifest);

        const auto dump_a = common_hybrid_manifest::plan_to_json(plan_a).dump(2);
        const auto dump_b = common_hybrid_manifest::plan_to_json(plan_b).dump(2);

        if (dump_a != dump_b) {
            throw std::runtime_error("expected identical plan dumps across repeated resolution");
        }
    });

    t.test("strict-mode validation", [&](testing & tc) {
        const auto strict_manifest = common_hybrid_manifest::load_manifest(
            base + "examples/strict-mode-validation-expected-pass.json");
        const auto strict_plan = common_hybrid_manifest::resolve_plan(strict_manifest);
        const auto strict_dump = common_hybrid_manifest::plan_to_json(strict_plan);

        tc.assert_true("expected strict fixture to preserve strict=true on the manifest", strict_manifest.strict);
        tc.assert_true("expected strict fixture to preserve strict=true in the resolved plan", strict_plan.strict);
        tc.assert_equal("expected deterministic strict fixture profile", std::string("dense-balanced"), strict_plan.profile);
        tc.assert_equal("expected strict fixture route count", size_t(2), strict_plan.entries.size());
        tc.assert_equal("expected strict fixture pipeline", std::string("INT8_STANDARD"), strict_plan.entries.at(1).pipeline);
        tc.assert_true("expected serialized strict plan to include strict=true", strict_dump.value("strict", false));

        bool threw = false;
        try {
            (void) common_hybrid_manifest::load_manifest(base + "invalid/strict-mode-validation-expected-fail.json");
        } catch (const std::exception & e) {
            threw = true;
            const std::string message = string_lower(e.what());
            tc.assert_true("expected strict-mode failure message to mention pipeline",
                message.find("pipeline") != std::string::npos);
        }

        tc.assert_true("expected invalid strict-mode fixture to fail loudly", threw);
    });
    t.test("generated rule names sort numerically and set precedence", [&](testing & tc) {
        common_hybrid_manifest::manifest manifest;
        manifest.profile = "dense-balanced";
        manifest.routes = {
            {.name = "rule-10", .match = "c", .target = "cpu"},
            {.name = "rule-2", .match = "b", .target = "cpu"},
            {.name = "rule-1", .match = "a", .target = "cpu"},
        };

        const auto plan = common_hybrid_manifest::resolve_plan(manifest);

        tc.assert_equal(std::string("rule-1"), plan.entries.at(0).name);
        tc.assert_equal(std::string("rule-2"), plan.entries.at(1).name);
        tc.assert_equal(std::string("rule-10"), plan.entries.at(2).name);
        tc.assert_equal(1, plan.entries.at(0).precedence);
        tc.assert_equal(2, plan.entries.at(1).precedence);
        tc.assert_equal(3, plan.entries.at(2).precedence);
    });

    t.test("default sidecar path appends .hybrid.json", [&](testing & tc) {
        const auto path = common_hybrid_manifest::default_manifest_path("/tmp/model.gguf");
        tc.assert_equal("/tmp/model.gguf.hybrid.json", path);
    });

    t.test("invalid fixtures fail loudly", [&](const testing &) {
        const std::vector<std::pair<std::string, std::string>> files = {
            {base + "invalid/bad-pipeline.json", "pipeline"},
            {base + "invalid/bad-quant-allow.json", "quant"},
            {base + "invalid/bad-shape.json", "shape"},
            {base + "invalid/bad-profile.json", "profile"},
            {base + "invalid/strict-mode-validation-expected-fail.json", "pipeline"},
        };

        for (const auto & [file, expected_message] : files) {
            bool threw = false;
            try {
                (void) common_hybrid_manifest::load_manifest(file);
            } catch (const std::exception & e) {
                threw = true;
                const std::string message = string_lower(e.what());
                if (message.find(expected_message) == std::string::npos) {
                    throw std::runtime_error("unexpected validation error for " + file + ": " + e.what());
                }
            }
            if (!threw) {
                throw std::runtime_error("expected invalid fixture to fail: " + file);
            }
        }
    });

    return t.failures > 0 ? 1 : 0;
}

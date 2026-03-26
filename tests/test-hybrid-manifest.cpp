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

    t.test("default sidecar path appends .hybrid.json", [&](testing & tc) {
        const auto path = common_hybrid_manifest::default_manifest_path("/tmp/model.gguf");
        tc.assert_equal("/tmp/model.gguf.hybrid.json", path);
    });

    t.test("invalid fixtures fail loudly", [&](const testing &) {
        const std::vector<std::string> files = {
            base + "invalid/bad-pipeline.json",
            base + "invalid/bad-quant-allow.json",
            base + "invalid/bad-shape.json",
            base + "invalid/bad-profile.json",
        };

        for (const auto & file : files) {
            bool threw = false;
            try {
                (void) common_hybrid_manifest::load_manifest(file);
            } catch (const std::exception &) {
                threw = true;
            }
            if (!threw) {
                throw std::runtime_error("expected invalid fixture to fail: " + file);
            }
        }
    });

    return t.failures > 0 ? 1 : 0;
}

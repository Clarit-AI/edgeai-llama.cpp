#pragma once

#include "ggml.h"

#include <rknn_matmul_api.h>

#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <functional>
#include <cstdint>
#include <mutex>
#include <memory>
#include <regex>

namespace rknpu2_configuration {

/**
 * @brief Defines a function signature for packing a segment of the weight matrix (B)
 * from a standard row-major layout into the device-specific native layout.
 *
 * @param dst Pointer to the destination buffer for the packed data.
 * @param src Pointer to the source data in row-major layout (ggml format).
 * @param K The number of columns in the source matrix (inner dimension).
 * @param N_total The total number of rows in the source matrix.
 * @param n_offset The row offset from which to start packing.
 * @param n_segment The number of rows to pack.
 */
using PackingFunction = std::function<void(
    uint8_t* dst,
    const uint8_t* src,
    int K,
    int N_total,
    int n_offset,
    int n_segment
)>;

/**
 * @brief An abstract representation of data types used on the NPU side.
 * This helps decouple the main logic from specific ggml or rknn types.
 */
enum Rknpu2NpuType {
    NPU_TYPE_INT4,
    NPU_TYPE_INT8,
    NPU_TYPE_INT16,
    NPU_TYPE_INT32,
    NPU_TYPE_FP16,
    NPU_TYPE_FP32,
};

/**
 * @brief Describes a single supported hardware pipeline available on the NPU.
 *
 * This struct encapsulates the specific constraints and capabilities of a 
 * hardware execution path, including required data types, memory alignment 
 * rules, and the specific packing function needed to prepare tensor for 
 * the underlying matrix multiplication operation.
 */
struct Rknpu2HardwarePipeline {
    std::string pipeline_name;  // The unique hardware pipeline name

    Rknpu2NpuType npu_type_a;   // The target data type for activations on the NPU
    Rknpu2NpuType npu_type_c;   // The data type of the result from the NPU
    rknn_matmul_type mm_type;   // Corresponding RKNN operation type

    int k_align;                // Required alignment for the K dimension
    int n_align;                // Required alignment for the N dimension
    PackingFunction pack_func;  // Function to pack the weight matrix for this op

    bool use_hadamard;          // Flag for using Hadamard Transform
};

/**
 * @brief Describes one manifest rule resolved against a tensor name.
 *
 * The backend reads hybrid policy from environment variables in this slice,
 * so the resolved route is cached on demand here rather than in the loader.
 */
struct Rknpu2HybridRoute {
    bool from_manifest = false;
    bool from_loader = false;
    bool force_cpu = false;
    bool valid = false;
    bool strict = false;

    int layer_id = -1;
    std::string role;
    std::string rule_name;
    std::string fallback_reason;
    std::string pipeline_name;

    const Rknpu2HardwarePipeline * pipeline = nullptr;
};

struct Rknpu2HybridRule {
    std::string name;
    std::regex match;
    bool force_cpu = false;
    bool required = false;
    bool has_layer_range = false;
    int layer_begin = -1;
    int layer_end = -1;
    std::string role;
    std::string pipeline_name;
    std::vector<std::string> source_quant_allow;
};

/**
 * @brief Holds the complete hardware configuration for a specific Rockchip NPU.
 *
 * This includes the device name, number of available NPU cores, and a list of
 * all available hardware pipelines. It also manages layer-by-layer quantization strategies.
 */
struct Rknpu2DeviceConfig {
    std::string device_name;
    int core_count;
    std::vector<Rknpu2HardwarePipeline> hardware_pipelines;

    // Type-specific default patterns mapping
    std::map<int, std::vector<std::string>> default_patterns;

    // Custom global pattern
    bool use_custom_pattern = false;
    std::vector<std::string> custom_hybrid_pattern; 

    // Tensor-Quantization mapping
    mutable std::shared_ptr<std::mutex> pattern_mutex = std::make_shared<std::mutex>();
    mutable std::unordered_map<std::string, int> tensor_sequence_map;
    mutable int global_tensor_counter = 0;

    // Env-driven hybrid manifest support. This is intentionally backend-local in
    // this slice so it can be exercised without loader/common changes.
    bool hybrid_manifest_loaded = false;
    bool hybrid_manifest_strict = false;
    bool hybrid_manifest_dump_plan = false;
    std::string hybrid_manifest_path;
    std::string hybrid_manifest_profile = "default";
    std::string hybrid_manifest_default_policy = "legacy";
    std::vector<Rknpu2HybridRule> hybrid_manifest_rules;
    std::string current_model_id;
    mutable std::map<std::pair<std::string, std::string>, Rknpu2HybridRoute> hybrid_route_cache;
    mutable std::map<std::pair<std::string, std::string>, Rknpu2HybridRoute> explicit_route_cache;

    /**
     * @brief Resolves the appropriate hardware pipeline for a given tensor.
     * Applies layer-by-layer cyclical selection if the active pattern has multiple entries.
     */
    const Rknpu2HardwarePipeline* resolve_op_support(const struct ggml_tensor* w_tensor) const;

    /**
     * @brief Dumps a concise manifest summary if one is active.
     */
    void dump_hybrid_manifest_summary() const;

    /**
     * @brief Resolves an explicit manifest route, if any, for a tensor.
     *
     * This is intentionally exposed so the backend op support check can apply
     * strict manifest behavior before falling back to legacy patterns.
     */
    const Rknpu2HybridRoute* resolve_manifest_route(const struct ggml_tensor * w_tensor) const;
    const Rknpu2HybridRoute* resolve_explicit_route(const struct ggml_tensor * w_tensor) const;
    void clear_explicit_routes() const;
    void register_explicit_route(const std::string & tensor_name, const std::string & pipeline_name, bool strict) const;

private:
    /**
     * @brief Retrieves the active pattern for a given tensor type.
     * Prioritizes custom ENV variable pattern over default mapping.
     */
    const std::vector<std::string>* get_active_pattern(int tensor_type) const;

    /**
     * @brief Returns a pipeline pointer by name.
     */
    const Rknpu2HardwarePipeline* find_pipeline(const std::string & name) const;
};

/**
 * @brief Manages and provides access to hardware configurations for different Rockchip NPUs.
 *
 * This is a singleton class that should be initialized once. It allows selecting
 * a specific device configuration (e.g., for "RK3588") which can then be
 * accessed globally to drive backend logic.
 */
class Rknpu2ConfigManager {
public:
    /**
     * @brief Get the singleton instance of the manager.
     * @return Reference to the Rknpu2ConfigManager instance.
     */
    static Rknpu2ConfigManager& get_instance();

    /**
     * @brief Selects the active device configuration.
     * @param device_name The name of the device to activate (e.g., "RK3588").
     * @return True if the configuration was found and set, false otherwise.
     */
    bool select_device(const std::string& device_name);

    /**
     * @brief Get the currently active device configuration.
     * @return A constant reference to the active Rknpu2DeviceConfig.
     */
    const Rknpu2DeviceConfig& get_current_config() const;

    /**
     * @brief Get the current split factor.
     * @return The current split factor value.
     */
    int get_split_factor() const { return split_factor; }

    /**
     * @brief Set the split factor.
     * @param factor The split factor value to set.
     */
    void set_split_factor_internal(int factor) { split_factor = factor; }
    void clear_explicit_routes();
    void register_explicit_route(const std::string & tensor_name, const std::string & pipeline_name, bool strict);

private:
    // Private constructor, destructor, and copy operators to enforce singleton pattern
    Rknpu2ConfigManager();
    ~Rknpu2ConfigManager() = default;
    Rknpu2ConfigManager(const Rknpu2ConfigManager&) = delete;
    Rknpu2ConfigManager& operator=(const Rknpu2ConfigManager&) = delete;

    std::map<std::string, Rknpu2DeviceConfig> device_configs;
    const Rknpu2DeviceConfig* current_config = nullptr;

    // Split factor for IOVA exhaustion handling - set at backend init
    int split_factor = 1;
};

/**
 * @brief Set the global split factor for IOVA exhaustion handling.
 * @param factor The split factor value (1-16).
 */
void set_split_factor(int factor);

} // namespace rknpu2_configuration
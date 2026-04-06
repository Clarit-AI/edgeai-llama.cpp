#include "ggml-rknpu2.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml-quants.h"

#include "rknpu2-allocation.h"
#include "rknpu2-quantization.h"
#include "rknpu2-calibration.h"
#include "rknpu2-configuration.h"

#include <rknn_api.h>
#include <rknn_matmul_api.h>

#include <arm_neon.h>
#include <omp.h>

#include <chrono>
#include <atomic>
#include <cassert>
#include <cerrno>
#include <cstring>
#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>
#include <map>
#include <tuple>
#include <algorithm>
#include <memory>
#include <unordered_map>
#include <list>
#include <random>
#include <limits>
#include <stdexcept>
#include <unordered_set>

#define UNUSED(x) (void)(x)

// Macro for RKNN API calls
#define RKNN_CHECK(stmt, msg)                                           \
    do {                                                                \
        int ret = (stmt);                                               \
        if (ret < 0) {                                                  \
            fprintf(stderr,"RKNN error %d at %s:%d: %s\n", ret,         \
                __FILE__, __LINE__, msg);                               \
            assert(false);                                              \
        }                                                               \
    } while (0)

// --- Hashers ---

// Function for hash combinations
template <class T>
inline void hash_combine(std::size_t& seed, const T& v) {
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

// Hasher for std::pair
struct PairHasher {
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2>& p) const {
        std::size_t seed = 0;
        hash_combine(seed, p.first);
        hash_combine(seed, p.second);
        return seed;
    }
};

// Hasher for std::tuple
struct TupleHasher {
    template <typename... Ts>
    std::size_t operator()(const std::tuple<Ts...>& t) const {
        std::size_t seed = 0;
        std::apply([&](const auto&... args) {
            (hash_combine(seed, args), ...);
        }, t);
        return seed;
    }
};

template <typename Key, typename Value, typename Hasher>
struct LruCache {
    struct Entry {
        Value value;
        typename std::list<Key>::iterator lru_it;
    };

    size_t capacity = 64;
    std::list<Key> lru_keys;
    std::unordered_map<Key, Entry, Hasher> entries;

    Value * find(const Key & key) {
        auto it = entries.find(key);
        if (it == entries.end()) {
            return nullptr;
        }

        touch(it);
        return &it->second.value;
    }

    void insert(const Key & key, Value value) {
        auto it = entries.find(key);
        if (it != entries.end()) {
            it->second.value = std::move(value);
            touch(it);
            return;
        }

        lru_keys.push_front(key);
        entries.emplace(key, Entry{std::move(value), lru_keys.begin()});
        trim();
    }

private:
    void touch(typename std::unordered_map<Key, Entry, Hasher>::iterator it) {
        lru_keys.splice(lru_keys.begin(), lru_keys, it->second.lru_it);
        it->second.lru_it = lru_keys.begin();
    }

    void trim() {
        while (entries.size() > capacity && !lru_keys.empty()) {
            const Key & key = lru_keys.back();
            entries.erase(key);
            lru_keys.pop_back();
        }
    }
};

// --- Segmenters ---

// Matrix segment information
struct MatrixSegment {
    int offset_n;  // Segment offset
    int size_n;    // Segment size
    int core_id;   // Segment core ID
};

// Split B-matrix into segments
// split_factor: multiplies number of segments to reduce IOVA allocation size per segment
// Each segment is assigned to a core via round-robin (i % num_cores)
static std::vector<MatrixSegment> compute_matrix_segments(int N, int num_cores, int alignment, int split_factor = 1) {
    std::vector<MatrixSegment> segments;

    int total_segments = num_cores * split_factor;
    int base_segment_size = (N / total_segments / alignment) * alignment;
    int remaining = N - (base_segment_size * total_segments);

    int offset = 0;
    for (int i = 0; i < total_segments; i++) {
        MatrixSegment seg;
        seg.offset_n = offset;
        seg.size_n = base_segment_size;
        seg.core_id = i % num_cores;  // Round-robin core assignment

        if (i < remaining / alignment) {
            seg.size_n += alignment;
        }

        offset += seg.size_n;
        segments.push_back(seg);
    }

    return segments;
}

static const char * rknpu_debug_tensor_name() {
    static const char * name = getenv("RKNPU_DEBUG_TENSOR");
    return name;
}

static bool rknpu_should_debug_tensor(const ggml_tensor * tensor) {
    const char * debug_name = rknpu_debug_tensor_name();
    return debug_name && debug_name[0] != '\0' && strcmp(debug_name, tensor->name) == 0;
}

static bool rknpu_trace_runs_enabled() {
    const char * value = getenv("RKNPU_TRACE_RUNS");
    return value && value[0] != '\0' && strcmp(value, "0") != 0;
}

static bool rknpu_trace_discriminator_enabled() {
    const char * value = getenv("RKNPU_TRACE_DISCRIM");
    return value && value[0] != '\0' && strcmp(value, "0") != 0;
}

static bool rknpu_disable_b_cache() {
    const char * value = getenv("RKNPU_DISABLE_B_CACHE");
    return value && value[0] != '\0' && strcmp(value, "0") != 0;
}

static bool rknpu_disable_ac_cache() {
    const char * value = getenv("RKNPU_DISABLE_AC_CACHE");
    return value && value[0] != '\0' && strcmp(value, "0") != 0;
}

static bool rknpu_disable_c_cache() {
    const char * value = getenv("RKNPU_DISABLE_C_CACHE");
    return value && value[0] != '\0' && strcmp(value, "0") != 0;
}

static bool rknpu_disable_matmul_ctx_cache() {
    const char * value = getenv("RKNPU_DISABLE_MATMUL_CTX_CACHE");
    return value && value[0] != '\0' && strcmp(value, "0") != 0;
}

static bool rknpu_clear_caches_after_op() {
    const char * value = getenv("RKNPU_CLEAR_CACHES_AFTER_OP");
    return value && value[0] != '\0' && strcmp(value, "0") != 0;
}

static bool rknpu_clear_caches_after_graph() {
    const char * value = getenv("RKNPU_CLEAR_CACHES_AFTER_GRAPH");
    return value && value[0] != '\0' && strcmp(value, "0") != 0;
}

static bool rknpu_clear_b_cache_after_op() {
    const char * value = getenv("RKNPU_CLEAR_B_CACHE_AFTER_OP");
    return value && value[0] != '\0' && strcmp(value, "0") != 0;
}

static bool rknpu_clear_c_cache_after_op() {
    const char * value = getenv("RKNPU_CLEAR_C_CACHE_AFTER_OP");
    return value && value[0] != '\0' && strcmp(value, "0") != 0;
}

static bool rknpu_clear_a_cache_after_op() {
    const char * value = getenv("RKNPU_CLEAR_A_CACHE_AFTER_OP");
    return value && value[0] != '\0' && strcmp(value, "0") != 0;
}

static bool rknpu_clear_matmul_ctx_cache_after_op() {
    const char * value = getenv("RKNPU_CLEAR_MATMUL_CTX_CACHE_AFTER_OP");
    return value && value[0] != '\0' && strcmp(value, "0") != 0;
}

static bool rknpu_drop_local_refs_before_clear() {
    const char * value = getenv("RKNPU_DROP_LOCAL_REFS_BEFORE_CLEAR");
    return value && value[0] != '\0' && strcmp(value, "0") != 0;
}

static bool rknpu_trace_progress() {
    const char * value = getenv("RKNPU_TRACE_PROGRESS");
    return value && value[0] != '\0' && strcmp(value, "0") != 0;
}

static bool rknpu_b_use_fd_import() {
    const char * value = getenv("RKNPU_B_ALLOC_MODE");
    return value && strcmp(value, "fd") == 0;
}

static bool rknpu_serial_b_segments() {
    const char * value = getenv("RKNPU_SERIAL_B_SEGMENTS");
    return value && value[0] != '\0' && strcmp(value, "0") != 0;
}

static int8_t rknpu_read_native_int8_b(
    const int8_t * native_b,
    int K,
    int n_segment,
    int local_n,
    int k) {
    const int n_block = local_n / 32;
    const int n_in_block = local_n % 32;
    const int row_stride = 32;
    const int k_segment_limit = 8192;
    int k_remaining = K;
    int k_base = 0;
    size_t packed_base = 0;

    while (k_remaining > 0) {
        const int k_segment = std::min(k_segment_limit, k_remaining);
        if (k >= k_base && k < k_base + k_segment) {
            const int local_k = k - k_base;
            const int k_block = local_k / 32;
            const int k_in_block = local_k % 32;
            const size_t block_stride = (size_t) (k_segment / 32) * 32 * 32;
            const size_t k_stride = 32 * 32;
            const size_t offset = packed_base + (size_t) n_block * block_stride + (size_t) k_block * k_stride + (size_t) n_in_block * row_stride + (size_t) k_in_block;
            return native_b[offset];
        }

        packed_base += (size_t) n_segment * k_segment;
        k_base += k_segment;
        k_remaining -= k_segment;
    }

    GGML_ASSERT(false && "k index outside packed K range");
    return 0;
}

// --- Structs ---

// RKNN buffer context
struct ggml_backend_rknpu_buffer_context {
    rknpu2_allocation::DmaBuffer dma_buf;
    std::string name;

    // Per-tensor B scales for quantized weights.
    // INT8 stores one scale per output row/channel; other formats keep a single-element vector.
    std::unordered_map<const struct ggml_tensor *, std::vector<float>> quantized_tensor_scales;

    // Per-tensor random sign vector for Hadamard Transform
    std::unordered_map<const struct ggml_tensor *, std::vector<float>> hadamard_s_vectors;

    std::mutex mutex;
};

// RKNN matmul operation context
struct rknpu_matmul_context {
    rknn_matmul_info info;
    rknn_matmul_io_attr io_attr;
    rknn_matmul_ctx ctx = 0;

    rknpu_matmul_context(int M, int K, int N, rknn_matmul_type type) {
        memset(&info, 0, sizeof(info));
        info.M = M;
        info.K = K;
        info.N = N;
        info.type = type;
        info.B_layout = RKNN_MM_LAYOUT_NATIVE;
        info.AC_layout = RKNN_MM_LAYOUT_NORM;

        int ret = rknn_matmul_create(&ctx, &info, &io_attr);
        if (ret < 0) ctx = 0;
    }

    ~rknpu_matmul_context() {
        if (ctx != 0) {
            rknn_matmul_destroy(ctx);
        }
    }
};


// Backend main context
struct ggml_backend_rknpu_context {
    using matmul_cache_key = std::tuple<int, int, int, int, int>;
    using b_mem_cache_key = std::tuple<rknn_matmul_ctx, ggml_backend_buffer_t, size_t, size_t>;

    std::string name;
    std::mutex mutex;
    rknn_core_mask core_mask = RKNN_NPU_CORE_0;  // Selected via RKNN_CORE_MASK env var
    int split_factor = 1;  // Selected via RKNN_SPLIT_FACTOR env var
    size_t matmul_ctx_cache_size = 64;
    size_t b_mem_handle_cache_size = 64;

    // RKNN matmul contexts cache
    LruCache<matmul_cache_key, std::shared_ptr<rknpu_matmul_context>, TupleHasher> matmul_ctx_cache;

    // B-matrices handle cache (from fd)
    LruCache<b_mem_cache_key, std::shared_ptr<rknn_tensor_mem>, TupleHasher> b_mem_handle_cache;

    // A- and C-matrices cache (from create_mem)
    std::unordered_map<std::tuple<rknn_matmul_ctx, int, int, int, size_t>, std::shared_ptr<rknn_tensor_mem>, TupleHasher> a_buffer_cache;
    std::unordered_map<std::tuple<rknn_matmul_ctx, int, int, int, int, size_t>, std::shared_ptr<rknn_tensor_mem>, TupleHasher> c_buffer_cache;

    void clear_runtime_caches() {
        std::lock_guard<std::mutex> lock(mutex);
        matmul_ctx_cache.entries.clear();
        matmul_ctx_cache.lru_keys.clear();
        b_mem_handle_cache.entries.clear();
        b_mem_handle_cache.lru_keys.clear();
        a_buffer_cache.clear();
        c_buffer_cache.clear();
    }

    void clear_runtime_caches_selective(bool clear_matmul, bool clear_b, bool clear_a, bool clear_c) {
        std::lock_guard<std::mutex> lock(mutex);
        if (clear_matmul) {
            matmul_ctx_cache.entries.clear();
            matmul_ctx_cache.lru_keys.clear();
        }
        if (clear_b) {
            b_mem_handle_cache.entries.clear();
            b_mem_handle_cache.lru_keys.clear();
        }
        if (clear_a) {
            a_buffer_cache.clear();
        }
        if (clear_c) {
            c_buffer_cache.clear();
        }
    }

    void clear_b_mem_handle_cache() {
        std::lock_guard<std::mutex> lock(mutex);
        b_mem_handle_cache.entries.clear();
        b_mem_handle_cache.lru_keys.clear();
    }

    void clear_matmul_ctx_cache() {
        std::lock_guard<std::mutex> lock(mutex);
        matmul_ctx_cache.entries.clear();
        matmul_ctx_cache.lru_keys.clear();
    }

    void clear_a_buffer_cache() {
        std::lock_guard<std::mutex> lock(mutex);
        a_buffer_cache.clear();
    }

    void clear_c_buffer_cache() {
        std::lock_guard<std::mutex> lock(mutex);
        c_buffer_cache.clear();
    }

    std::shared_ptr<rknpu_matmul_context> get_matmul_ctx(int M, int K, int N, int core_id, rknn_matmul_type type) {
        std::lock_guard<std::mutex> lock(mutex);
        auto key = std::make_tuple(M, K, N, core_id, (int)type);
        const bool disable_ctx_cache = rknpu_disable_matmul_ctx_cache();
        if (!disable_ctx_cache) {
            if (auto * cached = matmul_ctx_cache.find(key)) {
                return *cached;
            }
        }

        auto ctx = std::make_shared<rknpu_matmul_context>(M, K, N, type);
        if (ctx->ctx == 0) {
            return nullptr;
        }

        if (rknpu_trace_discriminator_enabled()) {
            fprintf(stderr,
                    "RKNPU_DISCRIM matmul_ctx_create M=%d K=%d N=%d core_id=%d type=%d ctx=%p A_size=%u B_size=%u C_size=%u A_dims=[%u,%u,%u,%u] B_dims=[%u,%u,%u,%u] C_dims=[%u,%u,%u,%u]\n",
                    M, K, N, core_id, (int) type, (void *) ctx->ctx,
                    ctx->io_attr.A.size, ctx->io_attr.B.size, ctx->io_attr.C.size,
                    ctx->io_attr.A.dims[0], ctx->io_attr.A.dims[1], ctx->io_attr.A.dims[2], ctx->io_attr.A.dims[3],
                    ctx->io_attr.B.dims[0], ctx->io_attr.B.dims[1], ctx->io_attr.B.dims[2], ctx->io_attr.B.dims[3],
                    ctx->io_attr.C.dims[0], ctx->io_attr.C.dims[1], ctx->io_attr.C.dims[2], ctx->io_attr.C.dims[3]);
        }

        // Determine which core mask to use
        // If RKNN_CORE_MASK was set to a specific core (not AUTO), use it for all operations
        // Otherwise, map core_id to the appropriate single-core mask for load balancing
        rknn_core_mask selected_core_mask;
        if (this->core_mask != RKNN_NPU_CORE_AUTO) {
            // Use the user-specified core mask
            selected_core_mask = this->core_mask;
        } else {
            // Map core_id to the appropriate single-core mask for load balancing
            // core_id 0 -> RKNN_NPU_CORE_0 (mask 1)
            // core_id 1 -> RKNN_NPU_CORE_1 (mask 2)
            // core_id 2 -> RKNN_NPU_CORE_2 (mask 4)
            // This allows each matrix segment to run on a different NPU core
            switch (core_id) {
                case 0: selected_core_mask = RKNN_NPU_CORE_0; break;
                case 1: selected_core_mask = RKNN_NPU_CORE_1; break;
                case 2: selected_core_mask = RKNN_NPU_CORE_2; break;
                default: selected_core_mask = RKNN_NPU_CORE_0; break;
            }
        }

        int ret = rknn_matmul_set_core_mask(ctx->ctx, selected_core_mask);
        if (ret != RKNN_SUCC) {
            // Handle error - fall back to core 0
            rknn_matmul_set_core_mask(ctx->ctx, RKNN_NPU_CORE_0);
        }

        if (!disable_ctx_cache) {
            matmul_ctx_cache.insert(key, ctx);
        }
        return ctx;
    }

    std::shared_ptr<rknn_tensor_mem> get_b_mem_handle(
        ggml_backend_buffer_t src0_buffer,
        ggml_backend_rknpu_buffer_context * src0_buf_ctx,
        size_t total_offset,
        size_t segment_size_bytes,
        const std::shared_ptr<rknpu_matmul_context> & matmul_ctx,
        const struct ggml_tensor * tensor = nullptr) {
        std::lock_guard<std::mutex> lock(mutex);
        const auto key = std::make_tuple(matmul_ctx->ctx, src0_buffer, total_offset, segment_size_bytes);
        const bool disable_b_cache = rknpu_disable_b_cache();
        if (!disable_b_cache) {
            if (auto * cached = b_mem_handle_cache.find(key)) {
                return *cached;
            }
        }

        const bool debug_tensor = tensor != nullptr && rknpu_should_debug_tensor(tensor);
        if (debug_tensor) {
            fprintf(stderr,
                    "RKNPU_DEBUG b_alloc_begin tensor=%s mode=%s total_offset=%zu segment_size_bytes=%zu ctx=%p\n",
                    tensor->name,
                    rknpu_b_use_fd_import() ? "fd" : "copy",
                    total_offset,
                    segment_size_bytes,
                    (void *) matmul_ctx->ctx);
        }

        rknn_tensor_mem * mem = nullptr;
        if (rknpu_b_use_fd_import()) {
            mem = rknn_create_mem_from_fd(
                matmul_ctx->ctx,
                src0_buf_ctx->dma_buf.fd,
                src0_buf_ctx->dma_buf.virt_addr,
                segment_size_bytes,
                total_offset);
        } else {
            mem = rknn_create_mem(matmul_ctx->ctx, segment_size_bytes);
        }
        if (debug_tensor) {
            fprintf(stderr,
                    "RKNPU_DEBUG b_alloc_end tensor=%s mem=%p is_null=%d\n",
                    tensor->name,
                    (void *) mem,
                    mem == nullptr ? 1 : 0);
        }
        if (!mem) {
            return nullptr;
        }

        if (!rknpu_b_use_fd_import()) {
            memcpy(mem->virt_addr, (uint8_t *) src0_buf_ctx->dma_buf.virt_addr + total_offset, segment_size_bytes);
            if (debug_tensor) {
                fprintf(stderr,
                        "RKNPU_DEBUG b_sync_begin tensor=%s mem=%p segment_size_bytes=%zu\n",
                        tensor->name,
                        (void *) mem,
                        segment_size_bytes);
            }
            RKNN_CHECK(rknn_mem_sync(matmul_ctx->ctx, mem, RKNN_MEMORY_SYNC_TO_DEVICE), "sync B TO_DEVICE");
            if (debug_tensor) {
                fprintf(stderr,
                        "RKNPU_DEBUG b_sync_end tensor=%s mem=%p\n",
                        tensor->name,
                        (void *) mem);
            }
        }

        auto deleter = [matmul_ctx](rknn_tensor_mem * m) {
            if (m && matmul_ctx->ctx != 0) {
                rknn_destroy_mem(matmul_ctx->ctx, m);
            }
        };

        std::shared_ptr<rknn_tensor_mem> mem_shared(mem, deleter);
        if (!disable_b_cache) {
            b_mem_handle_cache.insert(key, mem_shared);
        }
        return mem_shared;
    }
};

static std::mutex g_rknpu_backend_registry_mutex;
static std::unordered_set<ggml_backend_rknpu_context *> g_rknpu_backend_registry;

static void ggml_backend_rknpu_register_context(ggml_backend_rknpu_context * ctx) {
    std::lock_guard<std::mutex> lock(g_rknpu_backend_registry_mutex);
    g_rknpu_backend_registry.insert(ctx);
}

static void ggml_backend_rknpu_unregister_context(ggml_backend_rknpu_context * ctx) {
    std::lock_guard<std::mutex> lock(g_rknpu_backend_registry_mutex);
    g_rknpu_backend_registry.erase(ctx);
}

// RKNN memory global context
struct rknpu_memory_context {
    rknn_matmul_ctx mem_ctx = 0;
    std::mutex mutex;

    rknpu_memory_context() {
        rknn_matmul_info dummy_info;
        memset(&dummy_info, 0, sizeof(dummy_info));
        dummy_info.M = 32;
        dummy_info.K = 32;
        dummy_info.N = 32;
        dummy_info.type = RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32;

        rknn_matmul_io_attr dummy_io_attr;
        int ret = rknn_matmul_create(&mem_ctx, &dummy_info, &dummy_io_attr);
        if (ret < 0) {
            fprintf(stderr, "RKNPU2: Failed to create memory context (ret=%d). NPU backend will be unavailable.\n", ret);
            mem_ctx = 0;
        }
    }

    ~rknpu_memory_context() {
        if (mem_ctx != 0) {
            rknn_matmul_destroy(mem_ctx);
        }
    }

    rknn_matmul_ctx get_ctx() {
        std::lock_guard<std::mutex> lock(mutex);
        return mem_ctx;
    }
};

static rknpu_memory_context & get_rknpu_memory_context() {
    static rknpu_memory_context g_mem_ctx;
    return g_mem_ctx;
}

// Parse RKNN_CORE_MASK environment variable
// Accepts: "0", "1", "2", "auto", or hex values like "0x1", "0x3", "0x7"
static rknn_core_mask parse_core_mask_env() {
    const char* env = getenv("RKNN_CORE_MASK");
    if (!env) return RKNN_NPU_CORE_AUTO;
    if (strcmp(env, "0") == 0) return RKNN_NPU_CORE_0;
    if (strcmp(env, "1") == 0) return RKNN_NPU_CORE_1;
    if (strcmp(env, "2") == 0) return RKNN_NPU_CORE_2;
    if (strcmp(env, "auto") == 0) return RKNN_NPU_CORE_AUTO;
    // Handle hex values like 0x1, 0x3, 0x7
    if (strncmp(env, "0x", 2) == 0 || strncmp(env, "0X", 2) == 0) {
        int val = strtol(env, NULL, 16);
        if (val >= 0x1 && val <= 0x7) {
            return (rknn_core_mask)val;
        }
    }
    fprintf(stderr, "RKNPU2: Unknown RKNN_CORE_MASK '%s', defaulting to AUTO\n", env);
    return RKNN_NPU_CORE_AUTO;
}

// Parse RKNN_SPLIT_FACTOR environment variable
// Splits each core's segments into split_factor pieces to reduce IOVA allocation size
static int parse_split_factor_env() {
    const char* env = getenv("RKNN_SPLIT_FACTOR");
    if (!env) return 1;
    int val = atoi(env);
    if (val < 1) val = 1;
    if (val > 16) val = 16;  // Cap at 16 to avoid excessive overhead
    return val;
}

static size_t parse_cache_size_env(const char * env_name, size_t default_value) {
    const char * env = getenv(env_name);
    if (!env) {
        return default_value;
    }

    char * end = nullptr;
    errno = 0;
    long long val = strtoll(env, &end, 10);
    if (errno != 0 || end == env || *end != '\0' || val <= 0) {
        fprintf(stderr, "RKNPU2: Invalid %s '%s', clamping to 1\n", env_name, env);
        return 1;
    }

    return (size_t) val;
}


//
// Backend
//

static const char * ggml_backend_rknpu_name(ggml_backend_t backend) {
    UNUSED(backend);
    return "RKNPU";
}

static ggml_guid_t ggml_backend_rknpu_guid() {
    static ggml_guid guid = {0x8c, 0x6d, 0x5f, 0x11, 0x73, 0x5a, 0x4c, 0x2e,
                             0xb1, 0x42, 0x6a, 0x93, 0xd8, 0x04, 0x7e, 0x51};
    return &guid;
}

static void ggml_backend_rknpu_free(ggml_backend_t backend) {
    ggml_backend_rknpu_context * ctx = (ggml_backend_rknpu_context *)backend->context;
    ggml_backend_rknpu_unregister_context(ctx);
    delete ctx;
    delete backend;
}

// Function for getting buffer from cache or creating new one
template <typename CacheKeyType>
static std::shared_ptr<rknn_tensor_mem> get_or_create_npu_buffer(
    ggml_backend_rknpu_context* backend_ctx,
    const std::shared_ptr<rknpu_matmul_context> & matmul_ctx,
    size_t size,
    const CacheKeyType& key,
    std::unordered_map<CacheKeyType, std::shared_ptr<rknn_tensor_mem>, TupleHasher>& cache,
    bool disable_cache
) {
    std::lock_guard<std::mutex> lock(backend_ctx->mutex);
    if (!disable_cache) {
        auto it = cache.find(key);
        if (it != cache.end()) {
            if (it->second->size >= size) {
                return it->second;
            }
        }
    }

    rknn_tensor_mem* mem = rknn_create_mem(matmul_ctx->ctx, size);
    if (!mem) { return nullptr; }

    auto deleter = [matmul_ctx](rknn_tensor_mem* m) {
        if (m && matmul_ctx->ctx != 0) {
            rknn_destroy_mem(matmul_ctx->ctx, m);
        }
    };

    std::shared_ptr<rknn_tensor_mem> mem_shared(mem, deleter);
    if (!disable_cache) {
        cache[key] = mem_shared;
    }
    return mem_shared;
}

static enum ggml_status ggml_backend_rknpu_graph_compute(ggml_backend_t backend, struct ggml_cgraph* cgraph) {
    auto* backend_ctx = (ggml_backend_rknpu_context*)backend->context;
    if (rknpu_trace_progress()) {
        fprintf(stderr, "RKNPU_PROGRESS graph_compute_begin nodes=%d\n", cgraph->n_nodes);
    }
    static std::mutex debug_mutex;
    static std::unordered_set<std::string> debug_logged_contracts;
    static std::unordered_set<std::string> debug_logged_numeric;

    // Getting the current device configuration once
    const auto& config = rknpu2_configuration::Rknpu2ConfigManager::get_instance().get_current_config();

    for (int node_i = 0; node_i < cgraph->n_nodes; node_i++) {
        struct ggml_tensor* node = cgraph->nodes[node_i];
        if (node->op != GGML_OP_MUL_MAT) continue;

        const struct ggml_tensor* src0 = node->src[0]; // Weights      :  (K x N)
        const struct ggml_tensor* src1 = node->src[1]; // Activations  :  (M x K)
        struct ggml_tensor* dst = node;

        const int M = (int)src1->ne[1];
        const int K = (int)src0->ne[0];
        const int N = (int)src0->ne[1];
        
        // Skipping zero-dimension matmuls
        if (M == 0 || K == 0 || N == 0) {
            continue;
        }

        const auto* pipeline = config.resolve_op_support(src0);
        if (!pipeline) continue;

        // Guard: skip if src0 is not on an RKNPU buffer (e.g. fell back to CPU
        // because the RKNPU DMA buffer was full).  Without this check the code
        // below would cast a CPU-buffer context to an RKNPU-buffer context,
        // causing UB / assertion failures on the quantized-scale lookup.
        if (!src0->buffer || src0->buffer->buft != ggml_backend_rknpu_buffer_type()) {
            continue;
        }

        const bool is_hadamard = (pipeline->use_hadamard);
        const int K_op = is_hadamard ? rknpu2_calibration::next_power_of_two(K) : K;

        const rknn_matmul_type matmul_type = pipeline->mm_type;
        const int alignment = pipeline->n_align;

        auto all_segments = compute_matrix_segments(N, config.core_count, alignment, backend_ctx->split_factor);

        std::vector<MatrixSegment> active_segments;
        for (const auto& seg : all_segments) {
            if (seg.size_n > 0) {
                active_segments.push_back(seg);
            }
        }

        if (active_segments.empty()) continue;

        const size_t num_active_segments = active_segments.size();
        std::vector<std::shared_ptr<rknpu_matmul_context>> matmul_ctxs(num_active_segments);
        std::vector<std::shared_ptr<rknn_tensor_mem>> mem_B_segments(num_active_segments);
        std::vector<std::shared_ptr<rknn_tensor_mem>> mem_A_segments(num_active_segments);
        std::vector<std::shared_ptr<rknn_tensor_mem>> mem_C_segments(num_active_segments);
        std::vector<size_t> active_segment_total_offsets(num_active_segments, 0);
        std::vector<size_t> active_segment_expected_bytes(num_active_segments, 0);
        std::vector<uint8_t> packed_A;
        const bool debug_tensor = rknpu_should_debug_tensor(src0);
        const bool discrim_trace = debug_tensor && rknpu_trace_discriminator_enabled();
        const bool serial_b_segments = rknpu_serial_b_segments();

        // ===========================================
        // ========== 1. Preparing contexts ==========
        // ===========================================
        {
            if (discrim_trace) {
                fprintf(stderr,
                        "RKNPU_DISCRIM op_begin tensor=%s pipeline=%s M=%d K=%d N=%d K_op=%d mm_type=%s segments=%zu\n",
                        src0->name,
                        pipeline->pipeline_name.c_str(),
                        M, K, N, K_op,
                        get_matmul_type_string(matmul_type),
                        num_active_segments);
                for (size_t idx = 0; idx < num_active_segments; ++idx) {
                    fprintf(stderr,
                            "RKNPU_DISCRIM segment tensor=%s idx=%zu offset_n=%d size_n=%d core_id=%d\n",
                            src0->name,
                            idx,
                            active_segments[idx].offset_n,
                            active_segments[idx].size_n,
                            active_segments[idx].core_id);
                }
            }
            for (size_t idx = 0; idx < num_active_segments; ++idx) {
                const auto& seg = active_segments[idx];
                matmul_ctxs[idx] = backend_ctx->get_matmul_ctx(M, K_op, seg.size_n, seg.core_id, matmul_type);
                if (!matmul_ctxs[idx] || matmul_ctxs[idx]->ctx == 0) return GGML_STATUS_FAILED;
            }
        }

        // ===========================================
        // ========== 2. Preparing B-matrix ==========
        // ===========================================
        ggml_backend_buffer_t src0_buffer = src0->buffer;
        auto* src0_buf_ctx = (ggml_backend_rknpu_buffer_context*)src0_buffer->context;
        size_t src0_base_offset_in_dma = (uintptr_t)src0->data - (uintptr_t)ggml_backend_buffer_get_base(src0_buffer);

        size_t type_size_packed;
        if (pipeline->npu_type_a == rknpu2_configuration::NPU_TYPE_FP16) type_size_packed = 2;
        else if (pipeline->npu_type_a == rknpu2_configuration::NPU_TYPE_INT8) type_size_packed = 1;
        else type_size_packed = 0;

        {
            size_t current_offset_in_tensor = 0;
            for (const auto& seg : all_segments) {
                for (size_t idx = 0; idx < num_active_segments; ++idx) {
                    if (active_segments[idx].offset_n == seg.offset_n) {
                        auto& matmul_ctx = matmul_ctxs[idx];
                        size_t segment_size_bytes = matmul_ctx->io_attr.B.size;
                        size_t total_offset = src0_base_offset_in_dma + current_offset_in_tensor;
                        const size_t expected_packed_segment_bytes =
                            type_size_packed > 0 ? (size_t) seg.size_n * K_op * type_size_packed : (size_t) seg.size_n * K_op / 2;
                        if (debug_tensor) {
                            fprintf(stderr,
                                    "RKNPU_DEBUG b_segment tensor=%s offset_n=%d size_n=%d io_attr_B_size=%zu expected_packed_segment_bytes=%zu total_offset=%zu\n",
                                    src0->name,
                                    seg.offset_n,
                                    seg.size_n,
                                    segment_size_bytes,
                                    expected_packed_segment_bytes,
                                    total_offset);
                        }
                        active_segment_total_offsets[idx] = total_offset;
                        active_segment_expected_bytes[idx] = expected_packed_segment_bytes;
                        
                        if (!serial_b_segments) {
                            mem_B_segments[idx] = backend_ctx->get_b_mem_handle(
                                src0_buffer,
                                src0_buf_ctx,
                                total_offset,
                                segment_size_bytes,
                                matmul_ctx,
                                src0);
                            if (!mem_B_segments[idx]) return GGML_STATUS_FAILED;
                            if (debug_tensor) {
                                fprintf(stderr,
                                        "RKNPU_DEBUG b_set_io_begin tensor=%s mem=%p ctx=%p io_attr_B_size=%u\n",
                                        src0->name,
                                        (void *) mem_B_segments[idx].get(),
                                        (void *) matmul_ctx->ctx,
                                        matmul_ctx->io_attr.B.size);
                            }
                            int ret_set_b = rknn_matmul_set_io_mem(matmul_ctx->ctx, mem_B_segments[idx].get(), &matmul_ctx->io_attr.B);
                            if (discrim_trace) {
                                fprintf(stderr,
                                        "RKNPU_DISCRIM set_b tensor=%s idx=%zu ret=%d ctx=%p mem=%p B_size=%u expected_bytes=%zu total_offset=%zu\n",
                                        src0->name,
                                        idx,
                                        ret_set_b,
                                        (void *) matmul_ctx->ctx,
                                        (void *) mem_B_segments[idx].get(),
                                        matmul_ctx->io_attr.B.size,
                                        expected_packed_segment_bytes,
                                        total_offset);
                            }
                            RKNN_CHECK(ret_set_b, "set_io_mem B segment");
                            if (debug_tensor) {
                                fprintf(stderr,
                                        "RKNPU_DEBUG b_set_io_end tensor=%s mem=%p ctx=%p\n",
                                        src0->name,
                                        (void *) mem_B_segments[idx].get(),
                                        (void *) matmul_ctx->ctx);
                            }
                        }
                        break;
                    }
                }
                current_offset_in_tensor += type_size_packed > 0 ? (size_t)seg.size_n * K_op * type_size_packed : (size_t)seg.size_n * K_op / 2;
            }
        }

        // ===========================================
        // ========== 3. Preparing A-matrix ==========
        // ===========================================
        std::vector<float> scales_A(M);
        std::vector<float> scales_B;
        {
            const float* x = (const float*)src1->data;
            const int row_stride = (int)(src1->nb[1] / sizeof(float));

            std::vector<float> s_vec;
            if (is_hadamard) {
                auto* src0_buf_ctx = (ggml_backend_rknpu_buffer_context*)src0->buffer->context;
                std::lock_guard<std::mutex> lock(src0_buf_ctx->mutex);
                auto it = src0_buf_ctx->hadamard_s_vectors.find(src0);
                GGML_ASSERT(it != src0_buf_ctx->hadamard_s_vectors.end() && "Hadamard 's' vector not found");
                s_vec = it->second;
            }

            packed_A.resize(matmul_ctxs[0]->io_attr.A.size);
            void * packed_A_base = packed_A.data();

            #pragma omp parallel for
            for (int m = 0; m < M; ++m) {
                const float* src_row = x + (size_t)m * row_stride;
                std::vector<float> ready_row(K_op);

                if (is_hadamard) {
                    std::vector<float> signed_row(K);
                    for(int k=0; k<K; ++k) signed_row[k] = src_row[k] * s_vec[k];
                    rknpu2_calibration::hadamard_transform(ready_row.data(), signed_row.data(), K, K_op);
                } else {
                    memcpy(ready_row.data(), src_row, K * sizeof(float));
                }

                if (pipeline->npu_type_a == rknpu2_configuration::NPU_TYPE_FP16) {
                    uint16_t* dst_ptr = (uint16_t*)packed_A_base;
                    uint16_t* dst_row = dst_ptr + (size_t)m * K_op;
                    rknpu2_quantization::convert_fp32_to_fp16(ready_row.data(), dst_row, K_op);
                } 
                else if (pipeline->npu_type_a == rknpu2_configuration::NPU_TYPE_INT8) {
                    float amax_m = 0.0f;
                    for (int k = 0; k < K_op; ++k) amax_m = std::max(amax_m, std::abs(ready_row[k]));
                    scales_A[m] = amax_m / 127.0f;

                    int8_t* dst_ptr = (int8_t*)packed_A_base;
                    int8_t* dst_row = dst_ptr + (size_t)m * K_op;
                    rknpu2_quantization::quantize_fp32_to_int8(ready_row.data(), dst_row, K_op, scales_A[m]);
                } 
                else if (pipeline->npu_type_a == rknpu2_configuration::NPU_TYPE_INT4) {
                    float amax_m = 0.0f;
                    for (int k = 0; k < K_op; ++k) amax_m = std::max(amax_m, std::abs(ready_row[k]));
                    scales_A[m] = amax_m / 7.0f;

                    uint8_t* dst_ptr = (uint8_t*)packed_A_base;
                    uint8_t* dst_row = dst_ptr + (size_t)m * (K_op / 2);
                    rknpu2_quantization::quantize_fp32_to_int4_packed(ready_row.data(), dst_row, K_op, scales_A[m]);
                }
            }

            if (pipeline->npu_type_a == rknpu2_configuration::NPU_TYPE_INT8 || pipeline->npu_type_a == rknpu2_configuration::NPU_TYPE_INT4) {
                ggml_backend_buffer_t src0_buffer = src0->buffer;
                auto* src0_buf_ctx = (ggml_backend_rknpu_buffer_context*)src0_buffer->context;
                {
                    std::lock_guard<std::mutex> lock(src0_buf_ctx->mutex);
                    auto it = src0_buf_ctx->quantized_tensor_scales.find(src0);
                    GGML_ASSERT(it != src0_buf_ctx->quantized_tensor_scales.end() && "Quantized scale not found");
                    scales_B = it->second;
                }
            }

            if (debug_tensor) {
                std::lock_guard<std::mutex> lock(debug_mutex);
                if (debug_logged_contracts.insert(src0->name).second) {
                    auto & ctx0 = matmul_ctxs[0];
                    rknn_quant_params qparams;
                    memset(&qparams, 0, sizeof(qparams));
                    float qscale = 0.0f;
                    int qret = rknn_matmul_get_quant_params(ctx0->ctx, &qparams, &qscale);
                    fprintf(stderr,
                            "RKNPU_DEBUG contract tensor=%s pipeline=%s mm_type=%s B_quant_type=%d AC_quant_type=%d A_type=%d B_type=%d C_type=%d A_dims=[%u,%u,%u,%u] B_dims=[%u,%u,%u,%u] C_dims=[%u,%u,%u,%u] qret=%d qscale=%g scale_len=%d zp_len=%d\n",
                            src0->name,
                            pipeline->pipeline_name.c_str(),
                            get_matmul_type_string(matmul_type),
                            (int) ctx0->info.B_quant_type,
                            (int) ctx0->info.AC_quant_type,
                            (int) ctx0->io_attr.A.type,
                            (int) ctx0->io_attr.B.type,
                            (int) ctx0->io_attr.C.type,
                            ctx0->io_attr.A.dims[0], ctx0->io_attr.A.dims[1], ctx0->io_attr.A.dims[2], ctx0->io_attr.A.dims[3],
                            ctx0->io_attr.B.dims[0], ctx0->io_attr.B.dims[1], ctx0->io_attr.B.dims[2], ctx0->io_attr.B.dims[3],
                            ctx0->io_attr.C.dims[0], ctx0->io_attr.C.dims[1], ctx0->io_attr.C.dims[2], ctx0->io_attr.C.dims[3],
                            qret, qscale, qparams.scale_len, qparams.zp_len);
                }
            }

            for (size_t idx = 0; idx < num_active_segments; idx++) {
                auto & matmul_ctx = matmul_ctxs[idx];
                auto cache_key = std::make_tuple(matmul_ctx->ctx, M, K_op, (int)pipeline->npu_type_a, (size_t) matmul_ctx->io_attr.A.size);
                mem_A_segments[idx] = get_or_create_npu_buffer(backend_ctx, matmul_ctx, matmul_ctx->io_attr.A.size, cache_key, backend_ctx->a_buffer_cache, rknpu_disable_ac_cache());
                if (discrim_trace) {
                    fprintf(stderr,
                            "RKNPU_DISCRIM alloc_a tensor=%s idx=%zu ctx=%p mem=%p A_size=%u expected_bytes=%zu\n",
                            src0->name,
                            idx,
                            (void *) matmul_ctx->ctx,
                            (void *) (mem_A_segments[idx] ? mem_A_segments[idx].get() : nullptr),
                            matmul_ctx->io_attr.A.size,
                            packed_A.size());
                }
                if (!mem_A_segments[idx]) return GGML_STATUS_FAILED;

                memcpy(mem_A_segments[idx]->virt_addr, packed_A.data(), packed_A.size());
                int ret_sync_a = rknn_mem_sync(matmul_ctx->ctx, mem_A_segments[idx].get(), RKNN_MEMORY_SYNC_TO_DEVICE);
                if (discrim_trace) {
                    fprintf(stderr,
                            "RKNPU_DISCRIM sync_a tensor=%s idx=%zu ret=%d ctx=%p mem=%p\n",
                            src0->name,
                            idx,
                            ret_sync_a,
                            (void *) matmul_ctx->ctx,
                            (void *) mem_A_segments[idx].get());
                }
                if (rknpu_trace_runs_enabled()) {
                    fprintf(stderr,
                            "RKNPU_TRACE sync_a tensor=%s segment=%zu ret=%d ctx=%p mem=%p\n",
                            src0->name,
                            idx,
                            ret_sync_a,
                            (void *) matmul_ctx->ctx,
                            (void *) mem_A_segments[idx].get());
                }
                RKNN_CHECK(ret_sync_a, "sync A TO_DEVICE");
                int ret_set_a = rknn_matmul_set_io_mem(matmul_ctx->ctx, mem_A_segments[idx].get(), &matmul_ctx->io_attr.A);
                if (discrim_trace) {
                    fprintf(stderr,
                            "RKNPU_DISCRIM set_a tensor=%s idx=%zu ret=%d ctx=%p mem=%p A_size=%u\n",
                            src0->name,
                            idx,
                            ret_set_a,
                            (void *) matmul_ctx->ctx,
                            (void *) mem_A_segments[idx].get(),
                            matmul_ctx->io_attr.A.size);
                }
                if (rknpu_trace_runs_enabled()) {
                    fprintf(stderr,
                            "RKNPU_TRACE set_a tensor=%s segment=%zu ret=%d ctx=%p mem=%p\n",
                            src0->name,
                            idx,
                            ret_set_a,
                            (void *) matmul_ctx->ctx,
                            (void *) mem_A_segments[idx].get());
                }
                RKNN_CHECK(ret_set_a, "set_io_mem A for core");
            }
        }

        float* dst_data = (float*)dst->data;
        const float hadamard_divisor = pipeline->use_hadamard ? (float)K_op : 1.0f;
        auto write_back_segment = [&](size_t idx) {
            const int N_offset = active_segments[idx].offset_n;
            const int N_segment = active_segments[idx].size_n;
            for (int m = 0; m < M; ++m) {
                switch (pipeline->npu_type_c) {
                    case rknpu2_configuration::NPU_TYPE_FP32: {
                        float * src_segment_base = (float *) mem_C_segments[idx]->virt_addr;
                        float * dst_ptr = dst_data + (size_t) m * N + N_offset;
                        float * src_ptr = src_segment_base + (size_t) m * N_segment;
                        if (pipeline->use_hadamard) {
                            for (int n = 0; n < N_segment; ++n) dst_ptr[n] = src_ptr[n] / hadamard_divisor;
                        } else {
                            memcpy(dst_ptr, src_ptr, N_segment * sizeof(float));
                        }
                        break;
                    }
                    case rknpu2_configuration::NPU_TYPE_INT32: {
                        float * dst_ptr = dst_data + (size_t) m * N + N_offset;
                        int32_t * src_segment_base = (int32_t *) mem_C_segments[idx]->virt_addr;
                        int32_t * src_ptr = src_segment_base + (size_t) m * N_segment;
                        if (scales_B.size() == 1) {
                            const float dequant_scale = (scales_A[m] * scales_B[0]) / hadamard_divisor;
                            rknpu2_quantization::dequantize_int32_to_fp32(src_ptr, dst_ptr, N_segment, dequant_scale);
                        } else {
                            for (int n = 0; n < N_segment; ++n) {
                                const float dequant_scale = (scales_A[m] * scales_B[N_offset + n]) / hadamard_divisor;
                                dst_ptr[n] = (float) src_ptr[n] * dequant_scale;
                            }
                        }
                        break;
                    }
                    case rknpu2_configuration::NPU_TYPE_INT16: {
                        float * dst_ptr = dst_data + (size_t) m * N + N_offset;
                        int16_t * src_segment_base = (int16_t *) mem_C_segments[idx]->virt_addr;
                        int16_t * src_ptr = src_segment_base + (size_t) m * N_segment;
                        if (scales_B.size() == 1) {
                            const float dequant_scale = (scales_A[m] * scales_B[0]) / hadamard_divisor;
                            rknpu2_quantization::dequantize_int16_to_fp32(src_ptr, dst_ptr, N_segment, dequant_scale);
                        } else {
                            for (int n = 0; n < N_segment; ++n) {
                                const float dequant_scale = (scales_A[m] * scales_B[N_offset + n]) / hadamard_divisor;
                                dst_ptr[n] = (float) src_ptr[n] * dequant_scale;
                            }
                        }
                        break;
                    }
                    default:
                        GGML_ASSERT(false && "Unsupported NPU output type");
                }
            }
        };

        // ===========================================
        // ========== 4. Preparing C-matrix ==========
        // ===========================================
        if (!serial_b_segments) {
            for (size_t idx = 0; idx < num_active_segments; idx++) {
                auto& matmul_ctx = matmul_ctxs[idx];
                auto cache_key = std::make_tuple(matmul_ctx->ctx, M, active_segments[idx].size_n, active_segments[idx].core_id, (int)pipeline->npu_type_c, (size_t) matmul_ctx->io_attr.C.size);
                mem_C_segments[idx] = get_or_create_npu_buffer(backend_ctx, matmul_ctx, matmul_ctx->io_attr.C.size, cache_key, backend_ctx->c_buffer_cache, rknpu_disable_ac_cache() || rknpu_disable_c_cache());
                if (discrim_trace) {
                    fprintf(stderr,
                            "RKNPU_DISCRIM alloc_c tensor=%s idx=%zu ctx=%p mem=%p C_size=%u\n",
                            src0->name,
                            idx,
                            (void *) matmul_ctx->ctx,
                            (void *) (mem_C_segments[idx] ? mem_C_segments[idx].get() : nullptr),
                            matmul_ctx->io_attr.C.size);
                }
                if (!mem_C_segments[idx]) return GGML_STATUS_FAILED;
                int ret_set_c = rknn_matmul_set_io_mem(matmul_ctx->ctx, mem_C_segments[idx].get(), &matmul_ctx->io_attr.C);
                if (discrim_trace) {
                    fprintf(stderr,
                            "RKNPU_DISCRIM set_c tensor=%s idx=%zu ret=%d ctx=%p mem=%p C_size=%u\n",
                            src0->name,
                            idx,
                            ret_set_c,
                            (void *) matmul_ctx->ctx,
                            (void *) mem_C_segments[idx].get(),
                            matmul_ctx->io_attr.C.size);
                }
                if (rknpu_trace_runs_enabled()) {
                    fprintf(stderr,
                            "RKNPU_TRACE set_c tensor=%s segment=%zu ret=%d ctx=%p mem=%p\n",
                            src0->name,
                            idx,
                            ret_set_c,
                            (void *) matmul_ctx->ctx,
                            (void *) mem_C_segments[idx].get());
                }
                RKNN_CHECK(ret_set_c, "set_io_mem C");
            }
        }

        // ==========================================
        // ========== 5. Running operation ==========
        // ==========================================
        if (serial_b_segments) {
            for (size_t idx = 0; idx < num_active_segments; ++idx) {
                auto & matmul_ctx = matmul_ctxs[idx];

                mem_B_segments[idx] = backend_ctx->get_b_mem_handle(
                    src0_buffer,
                    src0_buf_ctx,
                    active_segment_total_offsets[idx],
                    matmul_ctx->io_attr.B.size,
                    matmul_ctx,
                    src0);
                if (!mem_B_segments[idx]) return GGML_STATUS_FAILED;

                int ret_set_b = rknn_matmul_set_io_mem(matmul_ctx->ctx, mem_B_segments[idx].get(), &matmul_ctx->io_attr.B);
                if (discrim_trace) {
                    fprintf(stderr,
                            "RKNPU_DISCRIM set_b tensor=%s idx=%zu ret=%d ctx=%p mem=%p B_size=%u expected_bytes=%zu total_offset=%zu serial=1\n",
                            src0->name,
                            idx,
                            ret_set_b,
                            (void *) matmul_ctx->ctx,
                            (void *) mem_B_segments[idx].get(),
                            matmul_ctx->io_attr.B.size,
                            active_segment_expected_bytes[idx],
                            active_segment_total_offsets[idx]);
                }
                RKNN_CHECK(ret_set_b, "set_io_mem B segment");

                auto cache_key = std::make_tuple(matmul_ctx->ctx, M, active_segments[idx].size_n, active_segments[idx].core_id, (int)pipeline->npu_type_c, (size_t) matmul_ctx->io_attr.C.size);
                mem_C_segments[idx] = get_or_create_npu_buffer(backend_ctx, matmul_ctx, matmul_ctx->io_attr.C.size, cache_key, backend_ctx->c_buffer_cache, rknpu_disable_ac_cache() || rknpu_disable_c_cache());
                if (!mem_C_segments[idx]) return GGML_STATUS_FAILED;
                int ret_set_c = rknn_matmul_set_io_mem(matmul_ctx->ctx, mem_C_segments[idx].get(), &matmul_ctx->io_attr.C);
                if (discrim_trace) {
                    fprintf(stderr,
                            "RKNPU_DISCRIM set_c tensor=%s idx=%zu ret=%d ctx=%p mem=%p C_size=%u serial=1\n",
                            src0->name,
                            idx,
                            ret_set_c,
                            (void *) matmul_ctx->ctx,
                            (void *) mem_C_segments[idx].get(),
                            matmul_ctx->io_attr.C.size);
                }
                RKNN_CHECK(ret_set_c, "set_io_mem C");

                int ret = rknn_matmul_run(matmul_ctx->ctx);
                if (discrim_trace) {
                    fprintf(stderr,
                            "RKNPU_DISCRIM run tensor=%s idx=%zu ret=%d ctx=%p offset_n=%d size_n=%d serial=1\n",
                            src0->name,
                            idx,
                            ret,
                            (void *) matmul_ctx->ctx,
                            active_segments[idx].offset_n,
                            active_segments[idx].size_n);
                }
                if (ret != RKNN_SUCC) {
                    fprintf(stderr, "RKNPU2: rknn_matmul_run failed for segment %zu, ret=%d\n", idx, ret);
                    return GGML_STATUS_FAILED;
                }

                int ret_sync_c = rknn_mem_sync(matmul_ctx->ctx, mem_C_segments[idx].get(), RKNN_MEMORY_SYNC_FROM_DEVICE);
                RKNN_CHECK(ret_sync_c, "sync C FROM_DEVICE");
                write_back_segment(idx);
                mem_B_segments[idx].reset();
                mem_C_segments[idx].reset();
            }
        } else {
            std::atomic<bool> had_error{false};
            #pragma omp parallel for num_threads(num_active_segments)
            for (size_t idx = 0; idx < num_active_segments; idx++) {
                if (rknpu_trace_runs_enabled()) {
                    fprintf(stderr,
                            "RKNPU_TRACE run_begin tensor=%s segment=%zu offset_n=%d size_n=%d ctx=%p\n",
                            src0->name,
                            idx,
                            active_segments[idx].offset_n,
                            active_segments[idx].size_n,
                            (void *) matmul_ctxs[idx]->ctx);
                }
                int ret = rknn_matmul_run(matmul_ctxs[idx]->ctx);
                if (discrim_trace) {
                    fprintf(stderr,
                            "RKNPU_DISCRIM run tensor=%s idx=%zu ret=%d ctx=%p offset_n=%d size_n=%d\n",
                            src0->name,
                            idx,
                            ret,
                            (void *) matmul_ctxs[idx]->ctx,
                            active_segments[idx].offset_n,
                            active_segments[idx].size_n);
                }
                if (rknpu_trace_runs_enabled()) {
                    fprintf(stderr,
                            "RKNPU_TRACE run_end tensor=%s segment=%zu ret=%d ctx=%p\n",
                            src0->name,
                            idx,
                            ret,
                            (void *) matmul_ctxs[idx]->ctx);
                }
                if (ret != RKNN_SUCC) {
                    had_error = true;
                    fprintf(stderr, "RKNPU2: rknn_matmul_run failed for segment %zu, ret=%d\n", idx, ret);
                }
            }
            if (had_error) {
                return GGML_STATUS_FAILED;
            }
        }

        // ===========================================
        // ========== 6. Collecting results ==========
        // ===========================================
        if (!serial_b_segments) {
            for (size_t idx = 0; idx < num_active_segments; idx++) {
                int ret_sync_c = rknn_mem_sync(matmul_ctxs[idx]->ctx, mem_C_segments[idx].get(), RKNN_MEMORY_SYNC_FROM_DEVICE);
                if (rknpu_trace_runs_enabled()) {
                    fprintf(stderr,
                            "RKNPU_TRACE sync_c tensor=%s segment=%zu ret=%d ctx=%p mem=%p\n",
                            src0->name,
                            idx,
                            ret_sync_c,
                            (void *) matmul_ctxs[idx]->ctx,
                            (void *) mem_C_segments[idx].get());
                }
                RKNN_CHECK(ret_sync_c, "sync C FROM_DEVICE");
            }

            if (debug_tensor && pipeline->npu_type_a == rknpu2_configuration::NPU_TYPE_INT8 && pipeline->npu_type_c == rknpu2_configuration::NPU_TYPE_INT32 && !mem_B_segments.empty()) {
                std::lock_guard<std::mutex> lock(debug_mutex);
                if (debug_logged_numeric.insert(src0->name).second) {
                    const int debug_m = 0;
                    const size_t idx = 0;
                    const int N_segment = active_segments[idx].size_n;
                    const int debug_cols = std::min(4, N_segment);
                    const int8_t * a_row_exact = (const int8_t *) mem_A_segments[idx]->virt_addr + (size_t) debug_m * K_op;
                    const int8_t * b_native = (const int8_t *) mem_B_segments[idx]->virt_addr;
                    const int32_t * c_row = (const int32_t *) mem_C_segments[idx]->virt_addr + (size_t) debug_m * N_segment;
                    const auto & a_attr = matmul_ctxs[idx]->io_attr.A;
                    const size_t expected_a_bytes = (size_t) M * K_op;

                    fprintf(stderr,
                            "RKNPU_DEBUG numeric tensor=%s segment_offset=%d segment_size=%d M=%d K=%d N=%d scales_A0=%g scales_B0=%g A_dims=[%u,%u,%u,%u] A_size=%u A_expected_bytes=%zu\n",
                            src0->name,
                            active_segments[idx].offset_n,
                            N_segment,
                            M, K_op, N,
                            scales_A.empty() ? 0.0f : scales_A[0],
                            scales_B.empty() ? 0.0f : scales_B[0],
                            a_attr.dims[0], a_attr.dims[1], a_attr.dims[2], a_attr.dims[3],
                            a_attr.size,
                            expected_a_bytes);

                    for (int local_n = 0; local_n < debug_cols; ++local_n) {
                        int32_t cpu_exact_dot = 0;
                        for (int k = 0; k < K_op; ++k) {
                            cpu_exact_dot += (int32_t) a_row_exact[k] * (int32_t) rknpu_read_native_int8_b(b_native, K_op, N_segment, local_n, k);
                        }
                        fprintf(stderr,
                                "RKNPU_DEBUG compare tensor=%s m=%d n_global=%d n_local=%d raw_int32=%d cpu_exact_dot=%d\n",
                                src0->name,
                                debug_m,
                                active_segments[idx].offset_n + local_n,
                                local_n,
                                c_row[local_n],
                                cpu_exact_dot);
                    }
                }
            }

            #pragma omp parallel for
            for (int idx = 0; idx < (int) num_active_segments; ++idx) {
                write_back_segment((size_t) idx);
            }
        }

        if (rknpu_clear_caches_after_op()) {
            if (rknpu_drop_local_refs_before_clear()) {
                for (auto & mem : mem_B_segments) {
                    mem.reset();
                }
                for (auto & mem : mem_C_segments) {
                    mem.reset();
                }
                for (auto & mem : mem_A_segments) {
                    mem.reset();
                }
                mem_B_segments.clear();
                mem_C_segments.clear();
                mem_A_segments.clear();
                matmul_ctxs.clear();
            }
            const bool clear_matmul = rknpu_clear_matmul_ctx_cache_after_op();
            const bool clear_b = rknpu_clear_b_cache_after_op();
            const bool clear_c = rknpu_clear_c_cache_after_op();
            const bool clear_a = rknpu_clear_a_cache_after_op();

            if (!clear_matmul && !clear_b && !clear_c && !clear_a) {
                backend_ctx->clear_runtime_caches();
            } else {
                backend_ctx->clear_runtime_caches_selective(clear_matmul, clear_b, clear_a, clear_c);
            }
        }
    }

    if (rknpu_trace_progress()) {
        fprintf(stderr, "RKNPU_PROGRESS graph_compute_end status=success\n");
    }
    if (rknpu_clear_caches_after_graph()) {
        backend_ctx->clear_runtime_caches();
    }

    return GGML_STATUS_SUCCESS;
}


//
// Buffer
//

static const char * ggml_backend_rknpu_buffer_name(ggml_backend_buffer_t buffer) {
    return "RKNPU";

    GGML_UNUSED(buffer);
}

static void ggml_backend_rknpu_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_rknpu_buffer_context * ctx = (ggml_backend_rknpu_buffer_context *)buffer->context;
    rknpu2_allocation::free(ctx->dma_buf);
    delete ctx;
}

static void * ggml_backend_rknpu_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_rknpu_buffer_context * ctx = (ggml_backend_rknpu_buffer_context *)buffer->context;
    return ctx->dma_buf.virt_addr;
}

static void ggml_backend_rknpu_buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    UNUSED(buffer);
    UNUSED(tensor);
}

// Function for dequantizing GGUF format to FP32 and optionally applying Hadamard transform
static std::vector<float> dequantize_tensor(
    const struct ggml_tensor * tensor,
    ggml_backend_rknpu_buffer_context * ctx,
    const void * raw_data,
    int K, int N, int K_op, bool use_hadamard
) {
    std::vector<float> fp32_matrix((size_t)N * K_op);

    auto dequantize_row = [&](int n, float* row_out) {
        if (tensor->type == GGML_TYPE_F32) {
            const float* src = (const float*)raw_data;
            memcpy(row_out, src + (size_t)n * K, K * sizeof(float));
        } else if (tensor->type == GGML_TYPE_F16) {
            const ggml_fp16_t* src = (const ggml_fp16_t*)raw_data;
            const ggml_fp16_t* src_row = src + (size_t)n * K;
            for (int k = 0; k < K; ++k) row_out[k] = ggml_fp16_to_fp32(src_row[k]);
        } else if (tensor->type == GGML_TYPE_Q8_0) {
            const block_q8_0* src = (const block_q8_0*)raw_data;
            dequantize_row_q8_0(src + (size_t)n * (K / QK8_0), row_out, K);
        } else if (tensor->type == GGML_TYPE_Q6_K) {
            const block_q6_K* src = (const block_q6_K*)raw_data;
            dequantize_row_q6_K(src + (size_t)n * (K / QK_K), row_out, K);
        } else if (tensor->type == GGML_TYPE_Q4_0) {
            const block_q4_0* src = (const block_q4_0*)raw_data;
            dequantize_row_q4_0(src + (size_t)n * (K / QK4_0), row_out, K);
        } else {
            GGML_ASSERT(false && "Unsupported weight type for NPU pipeline");
        }
    };

    if (use_hadamard) {
        std::vector<float> s_vec(K_op, 1.0f);
        std::mt19937 gen(reinterpret_cast<uintptr_t>(tensor));
        std::uniform_int_distribution<int> distrib(0, 1);
        for(int k = 0; k < K_op; ++k) {
            s_vec[k] = (distrib(gen) == 0) ? -1.0f : 1.0f;
        }

        {
            std::lock_guard<std::mutex> lock(ctx->mutex);
            ctx->hadamard_s_vectors[tensor] = s_vec;
        }

        #pragma omp parallel for
        for (int n = 0; n < N; ++n) {
            std::vector<float> raw_row(K);
            dequantize_row(n, raw_row.data());

            std::vector<float> signed_row(K);
            for(int k=0; k<K; ++k) signed_row[k] = raw_row[k] * s_vec[k];

            rknpu2_calibration::hadamard_transform(fp32_matrix.data() + (size_t)n * K_op, signed_row.data(), K, K_op);
        }
    } else {
        #pragma omp parallel for
        for (int n = 0; n < N; ++n) {
            dequantize_row(n, fp32_matrix.data() + (size_t)n * K);
        }
    }

    return fp32_matrix;
}

// Function for quantizing FP32 matrix to target NPU format
static std::vector<uint8_t> quantize_tensor(
    const struct ggml_tensor * tensor,
    ggml_backend_rknpu_buffer_context * ctx,
    const std::vector<float>& fp32_matrix,
    int K_op, int N,
    rknpu2_configuration::Rknpu2NpuType npu_type
) {
    size_t n_elements = (size_t)N * K_op;

    // FP16
    if (npu_type == rknpu2_configuration::NPU_TYPE_FP16) {
        std::vector<uint8_t> npu_bytes(n_elements * sizeof(uint16_t));
        uint16_t* fp16_ptr = (uint16_t*)npu_bytes.data();

        #pragma omp parallel for
        for (int n = 0; n < N; ++n) {
            rknpu2_quantization::convert_fp32_to_fp16(
                fp32_matrix.data() + (size_t)n * K_op, 
                fp16_ptr + (size_t)n * K_op, 
                K_op);
        }
        return npu_bytes;
    }

    // INT8
    if (npu_type == rknpu2_configuration::NPU_TYPE_INT8) {
        std::vector<uint8_t> npu_bytes(n_elements);
        int8_t* int8_ptr = (int8_t*)npu_bytes.data();
        std::vector<float> row_scales((size_t) N);

        #pragma omp parallel for
        for (int n = 0; n < N; ++n) {
            float amax_n = 0.0f;
            const float * row_ptr = fp32_matrix.data() + (size_t)n * K_op;
            for (int k = 0; k < K_op; ++k) {
                amax_n = std::max(amax_n, std::abs(row_ptr[k]));
            }
            const float row_scale_b = amax_n / 127.0f;
            row_scales[n] = row_scale_b;
            rknpu2_quantization::quantize_fp32_to_int8(
                row_ptr,
                int8_ptr + (size_t)n * K_op, 
                K_op, row_scale_b);
        }

        {
            std::lock_guard<std::mutex> lock(ctx->mutex);
            ctx->quantized_tensor_scales[tensor] = std::move(row_scales);
        }
        return npu_bytes;
    }

    float amax = rknpu2_calibration::calculate_entropy_amax(fp32_matrix.data(), n_elements);
    float global_scale_b = amax / 7.0f;

    {
        std::lock_guard<std::mutex> lock(ctx->mutex);
        ctx->quantized_tensor_scales[tensor] = { global_scale_b };
    }

    // INT4
    std::vector<uint8_t> npu_bytes(n_elements / 2);
    #pragma omp parallel for
    for (int n = 0; n < N; ++n) {
        rknpu2_quantization::quantize_fp32_to_int4_packed(
            fp32_matrix.data() + (size_t)n * K_op, 
            npu_bytes.data() + (size_t)n * (K_op / 2), 
            K_op, global_scale_b);
    }
    return npu_bytes;
}

// Function for splitting into NPU-native layout segments and writing to DMA buffer
static void pack_tensor(
    const uint8_t* src_data,
    uint8_t* dst_dma_ptr,
    int K_op, int N, int core_count,
    const rknpu2_configuration::Rknpu2HardwarePipeline * pipeline,
    int split_factor
) {
    auto segments = compute_matrix_segments(N, core_count, pipeline->n_align, split_factor);
    uint8_t* current_write_ptr = dst_dma_ptr;
    std::vector<uint8_t> packed_temp;

    for (const auto& seg : segments) {
        if (seg.size_n == 0) continue;

        size_t segment_packed_size = 0;
        if (pipeline->npu_type_a == rknpu2_configuration::NPU_TYPE_FP16) {
            segment_packed_size = (size_t)seg.size_n * K_op * 2;
        } else if (pipeline->npu_type_a == rknpu2_configuration::NPU_TYPE_INT8) {
            segment_packed_size = (size_t)seg.size_n * K_op;
        } else if (pipeline->npu_type_a == rknpu2_configuration::NPU_TYPE_INT4) {
            segment_packed_size = (size_t)seg.size_n * K_op / 2;
        }

        packed_temp.resize(segment_packed_size);
        pipeline->pack_func(packed_temp.data(), src_data, K_op, N, seg.offset_n, seg.size_n);

        memcpy(current_write_ptr, packed_temp.data(), segment_packed_size);
        current_write_ptr += segment_packed_size;
    }
}

static void ggml_backend_rknpu_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    auto * ctx = (ggml_backend_rknpu_buffer_context *) buffer->context;
    uint8_t* dma_base = (uint8_t*)ctx->dma_buf.virt_addr;
    uint8_t* tensor_dma_ptr = dma_base + ((uintptr_t)tensor->data - (uintptr_t)ggml_backend_buffer_get_base(buffer));

    const auto& config = rknpu2_configuration::Rknpu2ConfigManager::get_instance().get_current_config();
    const auto* pipeline = config.resolve_op_support(tensor);

    if (pipeline && pipeline->pack_func) {
        const int K = (int)tensor->ne[0];
        const int N = (int)tensor->ne[1];

        // Validate tensor dimensions are compatible with the pipeline
        // If N is not aligned to n_align, fall back to CPU to avoid assertion failures in pack_func
        if (N % pipeline->n_align != 0) {
            fprintf(stderr, "RKNPU2: tensor '%s' has N=%d not aligned to %d, falling back to CPU\n",
                    tensor->name, N, pipeline->n_align);
            memcpy(tensor_dma_ptr + offset, data, size);
            return;
        }

        const int K_op = pipeline->use_hadamard ? rknpu2_calibration::next_power_of_two(K) : K;

        std::vector<float> fp32_matrix = dequantize_tensor(tensor, ctx, data, K, N, K_op, pipeline->use_hadamard);
        std::vector<uint8_t> npu_matrix = quantize_tensor(tensor, ctx, fp32_matrix, K_op, N, pipeline->npu_type_a);
        int split_factor = rknpu2_configuration::Rknpu2ConfigManager::get_instance().get_split_factor();
        pack_tensor(npu_matrix.data(), tensor_dma_ptr, K_op, N, config.core_count, pipeline, split_factor);
    } else {
        memcpy(tensor_dma_ptr + offset, data, size);
    }
}

static void ggml_backend_rknpu_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_rknpu_buffer_context * ctx = (ggml_backend_rknpu_buffer_context *)buffer->context;
    uint8_t* dma_base = (uint8_t*)ctx->dma_buf.virt_addr;
    uint8_t* tensor_dma_ptr = dma_base + ((uintptr_t)tensor->data - (uintptr_t)ggml_backend_buffer_get_base(buffer));
    memcpy(data, tensor_dma_ptr + offset, size);
}

static void ggml_backend_rknpu_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_rknpu_buffer_context * ctx = (ggml_backend_rknpu_buffer_context *)buffer->context;
    memset(ctx->dma_buf.virt_addr, value, ctx->dma_buf.size);
}


//
// Buffer Type
//

static const char * ggml_backend_rknpu_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    UNUSED(buft);
    return "RKNPU";
}

static ggml_backend_buffer_t ggml_backend_rknpu_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    UNUSED(buft);

    fprintf(stderr,
            "RKNPU_BUFFER_ALLOC: one-shot backend buffer request size=%zu bytes (%.2f MiB), alignment=%zu\n",
            size, size / 1024.0 / 1024.0, (size_t)64);

    rknpu2_allocation::DmaBuffer dma_buf = rknpu2_allocation::alloc(size);
    if (dma_buf.fd < 0) {
        return NULL;
    }

    ggml_backend_rknpu_buffer_context * ctx = new ggml_backend_rknpu_buffer_context{
        dma_buf, "rknpu_dma_buffer", {}, {}, {}
    };

    static const ggml_backend_buffer_i rknpu_buffer_interface = {
        /* .get_name      = */ ggml_backend_rknpu_buffer_name,
        /* .free_buffer   = */ ggml_backend_rknpu_buffer_free_buffer,
        /* .get_base      = */ ggml_backend_rknpu_buffer_get_base,
        /* .init_tensor   = */ ggml_backend_rknpu_buffer_init_tensor,
        /* .memset_tensor = */ NULL,
        /* .set_tensor    = */ ggml_backend_rknpu_buffer_set_tensor,
        /* .get_tensor    = */ ggml_backend_rknpu_buffer_get_tensor,
        /* .cpy_tensor    = */ NULL,
        /* .clear         = */ ggml_backend_rknpu_buffer_clear,
        /* .reset         = */ NULL,
    };

    return ggml_backend_buffer_init(buft, rknpu_buffer_interface, ctx, size);
}

static size_t ggml_backend_rknpu_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    UNUSED(buft);
    return 64;
}

static size_t ggml_backend_rknpu_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor) {
    UNUSED(buft);

    // Getting the current device configuration
    const auto& config = rknpu2_configuration::Rknpu2ConfigManager::get_instance().get_current_config();

    // Defining hardware pipeline for the tensor
    const auto* pipeline = config.resolve_op_support(tensor);

    if (pipeline) {
        const int K = (int)tensor->ne[0];
        const int N = (int)tensor->ne[1];

        // Check alignment - must match the check in set_tensor to avoid allocation/packing mismatch
        if (N % pipeline->n_align != 0) {
            // Fall back to CPU allocation size if not aligned
            return ggml_nbytes(tensor);
        }

        int split_factor = rknpu2_configuration::Rknpu2ConfigManager::get_instance().get_split_factor();
        auto segments = compute_matrix_segments(N, config.core_count, pipeline->n_align, split_factor);

        const int K_op = pipeline->use_hadamard ? rknpu2_calibration::next_power_of_two(K) : K;

        size_t total_size = 0;
        for (const auto& seg : segments) {
            if (seg.size_n > 0) {
                if (pipeline->npu_type_a == rknpu2_configuration::NPU_TYPE_INT4) {
                    total_size += (size_t)seg.size_n * K_op / 2;
                } else if (pipeline->npu_type_a == rknpu2_configuration::NPU_TYPE_INT8) {
                    total_size += (size_t)seg.size_n * K_op;
                } else if (pipeline->npu_type_a == rknpu2_configuration::NPU_TYPE_FP16) {
                    total_size += (size_t)seg.size_n * K_op * 2;
                }
            }
        }
        return total_size;
    }

    // Fallback to default size calculation for other types.
    return ggml_nbytes(tensor);
}

static ggml_backend_buffer_type_t ggml_backend_rknpu_buffer_type(void) {
    static const struct ggml_backend_buffer_type_i rknpu_buffer_type_interface = {
        /* .get_name       = */ ggml_backend_rknpu_buffer_type_get_name,
        /* .alloc_buffer   = */ ggml_backend_rknpu_buffer_type_alloc_buffer,
        /* .get_alignment  = */ ggml_backend_rknpu_buffer_type_get_alignment,
        /* .get_max_size   = */ NULL,
        /* .get_alloc_size = */ ggml_backend_rknpu_buffer_type_get_alloc_size,
        /* .is_host        = */ NULL,
    };

    static struct ggml_backend_buffer_type rknpu_buffer_type = {
        /* .iface   = */ rknpu_buffer_type_interface,
        /* .device  = */ NULL,
        /* .context = */ NULL,
    };

    return &rknpu_buffer_type;
}


//
// Device
//

static const char * ggml_backend_rknpu_device_get_name(ggml_backend_dev_t dev) {
    UNUSED(dev);
    return "RKNPU";
}

static const char * ggml_backend_rknpu_device_get_description(ggml_backend_dev_t dev) {
    UNUSED(dev);
    return "Rockchip NPU";
}

static void ggml_backend_rknpu_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    UNUSED(dev);
    *free = 0;
    *total = 0;
}

static enum ggml_backend_dev_type ggml_backend_rknpu_device_get_type(ggml_backend_dev_t dev) {
    UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
}

static void ggml_backend_rknpu_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name = ggml_backend_rknpu_device_get_name(dev);
    props->description = ggml_backend_rknpu_device_get_description(dev);
    props->type = ggml_backend_rknpu_device_get_type(dev);
    ggml_backend_rknpu_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->device_id = NULL;

    props->caps.async = false;
    props->caps.host_buffer = false;
    props->caps.buffer_from_host_ptr = false;
    props->caps.events = false;
}

static bool ggml_backend_rknpu_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    UNUSED(dev);

    switch (op->op) {
        case GGML_OP_NONE:
            return true;

        case GGML_OP_MUL_MAT: {
            const struct ggml_tensor * src0 = op->src[0]; // Weights
            const struct ggml_tensor * src1 = op->src[1]; // Activations
            const auto & config = rknpu2_configuration::Rknpu2ConfigManager::get_instance().get_current_config();
            const rknpu2_configuration::Rknpu2HybridRoute * manifest_route = nullptr;
            bool manifest_strict = false;
            bool manifest_mode = false;
            try {
                manifest_route = config.resolve_manifest_route(src0);
                manifest_strict = manifest_route != nullptr && manifest_route->strict;
                manifest_mode = manifest_route != nullptr;
            } catch (const std::exception &) {
                // Exception during manifest resolution - treat as no manifest
                manifest_mode = false;
            }

            if (manifest_mode) {
                if (manifest_route->force_cpu || !manifest_route->valid || manifest_route->pipeline == nullptr) {
                    return false;
                }
            }

            // Explicitly reject IQK quantization types - NPU cannot handle these
            // These types have different block structures than standard quants
            // CPU backend will handle these via its own optimized kernels
            switch (src0->type) {
                case GGML_TYPE_IQ2_XXS:
                case GGML_TYPE_IQ2_XS:
                case GGML_TYPE_IQ3_XXS:
                case GGML_TYPE_IQ1_S:
                case GGML_TYPE_IQ4_NL:
                case GGML_TYPE_IQ3_S:
                case GGML_TYPE_IQ2_S:
                case GGML_TYPE_IQ4_XS:
                case GGML_TYPE_IQ1_M:
                case GGML_TYPE_IQ2_K:
                case GGML_TYPE_IQ3_K:
                case GGML_TYPE_IQ4_K:
                case GGML_TYPE_IQ5_K:
                case GGML_TYPE_IQ6_K:
                case GGML_TYPE_IQ4_KS:
                case GGML_TYPE_IQ2_KS:
                case GGML_TYPE_IQ4_KSS:
                case GGML_TYPE_IQ5_KS:
                case GGML_TYPE_IQ2_KT:
                case GGML_TYPE_IQ3_KT:
                case GGML_TYPE_IQ4_KT:
                case GGML_TYPE_IQ3_KS:
                case GGML_TYPE_IQ2_KL:
                case GGML_TYPE_IQ1_KT:
                case GGML_TYPE_IQ2_XXS_R4:
                case GGML_TYPE_IQ2_XS_R4:
                case GGML_TYPE_IQ3_XXS_R4:
                case GGML_TYPE_IQ1_S_R4:
                case GGML_TYPE_IQ4_NL_R4:
                case GGML_TYPE_IQ3_S_R4:
                case GGML_TYPE_IQ2_S_R4:
                case GGML_TYPE_IQ4_XS_R8:
                case GGML_TYPE_IQ1_M_R4:
                case GGML_TYPE_IQ2_BN:
                case GGML_TYPE_IQ2_BN_R4:
                case GGML_TYPE_IQ2_K_R4:
                case GGML_TYPE_IQ3_K_R4:
                case GGML_TYPE_IQ4_K_R4:
                case GGML_TYPE_IQ5_K_R4:
                case GGML_TYPE_IQ4_KS_R4:
                case GGML_TYPE_IQ5_KS_R4:
                    return false;
                default:
                    break;
            }

            // Searching for available hardware pipeline for this tensor
            const auto* pipeline = manifest_mode ? manifest_route->pipeline : nullptr;
            if (!manifest_mode) {
                pipeline = config.resolve_op_support(src0);
            }
            if (!pipeline) {
                return false;
            }

            // Rejecting zero-dimension ops
            if (src0->ne[0] == 0 || src0->ne[1] == 0 ||
                src1->ne[0] == 0 || src1->ne[1] == 0) {
                return false;
            }

            // Checking if activation type matches the supported operation
            if (src1->type != GGML_TYPE_F32) {
                if (manifest_strict) {
                    fprintf(stderr, "RKNPU2 strict hybrid route rejected tensor '%s' because src1 is not F32\n", src0->name ? src0->name : "");
                }
                return false;
            }

            // Checking for K alignment
            if (src0->ne[0] % pipeline->k_align != 0) {
                if (manifest_strict) {
                    fprintf(stderr, "RKNPU2 strict hybrid route rejected tensor '%s' because K alignment does not match\n", src0->name ? src0->name : "");
                }
                return false;
            }

            // Checking for N alignment
            if (src0->ne[1] % pipeline->n_align != 0) {
                if (manifest_strict) {
                    fprintf(stderr, "RKNPU2 strict hybrid route rejected tensor '%s' because N alignment does not match\n", src0->name ? src0->name : "");
                }
                return false;
            }

            // Checking for exact dimensions
            if (src1->ne[0] != src0->ne[0]) {
                 if (manifest_strict) {
                    fprintf(stderr, "RKNPU2 strict hybrid route rejected tensor '%s' because src1->ne[0] != src0->ne[0]\n", src0->name ? src0->name : "");
                 }
                 return false;
            }

            // Checking contiguous memory
            if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1)) {
                if (manifest_strict) {
                    fprintf(stderr, "RKNPU2 strict hybrid route rejected tensor '%s' because tensors are not contiguous\n", src0->name ? src0->name : "");
                }
                return false;
            }

            return true;
        }
        default:
            return false;
    }
}

static ggml_backend_buffer_type_t ggml_backend_rknpu_get_default_buffer_type(ggml_backend_t backend) {
    UNUSED(backend);
    return ggml_backend_rknpu_buffer_type();
}

static bool ggml_backend_rknpu_supports_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    UNUSED(backend);
    return ggml_backend_rknpu_device_supports_op(NULL, op);
}

static bool ggml_backend_rknpu_supports_buft(ggml_backend_t backend, ggml_backend_buffer_type_t buft) {
    UNUSED(backend);
    return buft == ggml_backend_rknpu_buffer_type();
}

static ggml_backend_t ggml_backend_rknpu_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    UNUSED(dev);
    UNUSED(params);

    // Parse RKNN_DEVICE environment variable for device selection
    const char* device_name = getenv("RKNN_DEVICE");
    if (!device_name) device_name = "RK3588";

    if (!rknpu2_configuration::Rknpu2ConfigManager::get_instance().select_device(device_name)) {
        fprintf(stderr, "RKNPU2: Failed to select device '%s'\n", device_name);
        return NULL;
    }

    ggml_backend_rknpu_context * ctx = new ggml_backend_rknpu_context();
    ggml_backend_rknpu_register_context(ctx);
    ctx->core_mask = parse_core_mask_env();
    ctx->split_factor = parse_split_factor_env();
    ctx->b_mem_handle_cache_size = parse_cache_size_env("RKNPU_B_CACHE_SIZE", 64);
    ctx->matmul_ctx_cache_size = parse_cache_size_env("RKNPU_CTX_CACHE_SIZE", 64);
    ctx->b_mem_handle_cache.capacity = ctx->b_mem_handle_cache_size;
    ctx->matmul_ctx_cache.capacity = ctx->matmul_ctx_cache_size;
    rknpu2_configuration::set_split_factor(ctx->split_factor);
    fprintf(stderr,
            "RKNPU2: Using device '%s' with core_mask=%d, split_factor=%d, b_cache_size=%zu, ctx_cache_size=%zu\n",
            device_name,
            ctx->core_mask,
            ctx->split_factor,
            ctx->b_mem_handle_cache_size,
            ctx->matmul_ctx_cache_size);
    
    static const struct ggml_backend_i rknpu_backend_interface = {
        /* .get_name                = */ ggml_backend_rknpu_name,
        /* .free                    = */ ggml_backend_rknpu_free,
        /* .get_default_buffer_type = */ ggml_backend_rknpu_get_default_buffer_type,
        /* .set_tensor_async        = */ NULL,
        /* .get_tensor_async        = */ NULL,
        /* .cpy_tensor_async        = */ NULL,
        /* .synchronize             = */ NULL,
        /* .graph_plan_create       = */ NULL,
        /* .graph_plan_free         = */ NULL,
        /* .graph_plan_update       = */ NULL,
        /* .graph_plan_compute      = */ NULL,
        /* .graph_compute           = */ ggml_backend_rknpu_graph_compute,
        /* .supports_op             = */ ggml_backend_rknpu_supports_op,
        /* .supports_buft           = */ ggml_backend_rknpu_supports_buft,
        /* .offload_op              = */ NULL,
        /* .event_new               = */ NULL,
        /* .event_free              = */ NULL,
        /* .event_record            = */ NULL,
        /* .event_wait              = */ NULL,
        /* .event_synchronize       = */ NULL,
    };

    return new ggml_backend{
        /* .guid    = */ ggml_backend_rknpu_guid(),
        /* .iface   = */ rknpu_backend_interface,
        /* .context = */ ctx,
    };
}


//
// Registry
//

static const char * ggml_backend_rknpu_reg_get_name(ggml_backend_reg_t reg) {
    UNUSED(reg);
    return "RKNPU";
}

static size_t ggml_backend_rknpu_reg_get_device_count(ggml_backend_reg_t reg) {
    UNUSED(reg);
    if (get_rknpu_memory_context().get_ctx() != 0) {
        return 1;
    }
    return 0;
}

static ggml_backend_dev_t ggml_backend_rknpu_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    if (index != 0) {
        return NULL;
    }

    ggml_backend_buffer_type_t rknpu_buffer_type = ggml_backend_rknpu_buffer_type();

    static const struct ggml_backend_device_i rknpu_device_interface = {
        /* .get_name             = */ ggml_backend_rknpu_device_get_name,
        /* .get_description      = */ ggml_backend_rknpu_device_get_description,
        /* .get_memory           = */ ggml_backend_rknpu_device_get_memory,
        /* .get_type             = */ ggml_backend_rknpu_device_get_type,
        /* .get_props            = */ ggml_backend_rknpu_device_get_props,
        /* .init_backend         = */ ggml_backend_rknpu_device_init_backend,
        /* .get_buffer_type      = */ [](ggml_backend_dev_t dev) { UNUSED(dev); return ggml_backend_rknpu_buffer_type(); },
        /* .supports_op          = */ ggml_backend_rknpu_device_supports_op,
        /* .supports_buft        = */ [](ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) { UNUSED(dev); return buft == ggml_backend_rknpu_buffer_type(); },
        /* .offload_op           = */ NULL,
    };

    static struct ggml_backend_device rknpu_device = {
        /* .iface   = */ rknpu_device_interface,
        /* .reg     = */ reg,
        /* .context = */ NULL,
    };

    if (rknpu_buffer_type->device == NULL) {
        rknpu_buffer_type->device = &rknpu_device;
    }

    return &rknpu_device;
}


//
// Public API
//

GGML_API ggml_backend_reg_t ggml_backend_rknpu2_reg(void) {
    static const struct ggml_backend_reg_i rknpu_reg_interface = {
        /* .get_name         = */ ggml_backend_rknpu_reg_get_name,
        /* .get_device_count = */ ggml_backend_rknpu_reg_get_device_count,
        /* .get_device       = */ ggml_backend_rknpu_reg_get_device,
        /* .get_proc_address = */ NULL,
    };

    static struct ggml_backend_reg rknpu_backend_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ rknpu_reg_interface,
        /* .context     = */ NULL,
    };

    return &rknpu_backend_reg;
}

GGML_API void ggml_backend_rknpu2_clear_runtime_caches(void) {
    std::lock_guard<std::mutex> lock(g_rknpu_backend_registry_mutex);
    for (ggml_backend_rknpu_context * ctx : g_rknpu_backend_registry) {
        if (ctx != nullptr) {
            ctx->clear_runtime_caches();
        }
    }
}

#ifdef GGML_BACKEND_DL
GGML_BACKEND_DL_IMPL(ggml_backend_rknpu2_reg)
#endif

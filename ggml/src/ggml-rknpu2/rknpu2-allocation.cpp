#include "rknpu2-allocation.h"

#include <sys/ioctl.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <cerrno>
#include <cstdio>

// --- Anonymous namespace for implementation details ---

namespace {

// --- DMA-Heap specific structures and IOCTL commands ---

struct dma_heap_allocation_data {
    uint64_t len;
    uint32_t fd;
    uint32_t fd_flags;
    uint64_t heap_flags;
};

#define DMA_HEAP_IOC_MAGIC 'H'
#define DMA_HEAP_IOCTL_ALLOC _IOWR(DMA_HEAP_IOC_MAGIC, 0x0, struct dma_heap_allocation_data)

} // anonymous namespace

namespace rknpu2_allocation {

DmaBuffer alloc(size_t size) {
    DmaBuffer buffer;
    buffer.size = size;

    // Try CMA heap first for NPU compatibility with DRM GEM handles.
    // If the heap exists but cannot satisfy the request, retry on system heap.
    const char* paths[] = {"/dev/dma_heap/cma", "/dev/dma_heap/system"};
    const long page_size = sysconf(_SC_PAGESIZE);
    int last_open_errno = 0;
    int last_ioctl_errno = 0;

    for (const char* path : paths) {
        int dma_heap_fd = open(path, O_RDWR);
        if (dma_heap_fd < 0) {
            last_open_errno = errno;
            fprintf(stderr, "RKNPU_DMA_ALLOC: failed to open heap=%s: %s\n", path, strerror(errno));
            continue;
        }

        dma_heap_allocation_data buf_data;
        memset(&buf_data, 0, sizeof(buf_data));
        buf_data.len = size;
        buf_data.fd_flags = O_CLOEXEC | O_RDWR;

        fprintf(stderr,
                "RKNPU_DMA_ALLOC: request size=%zu bytes (%.2f MiB), page_size=%ld, heap=%s\n",
                size, size / 1024.0 / 1024.0, page_size, path);

        if (ioctl(dma_heap_fd, DMA_HEAP_IOCTL_ALLOC, &buf_data) < 0) {
            last_ioctl_errno = errno;
            fprintf(stderr,
                    "RKNPU_DMA_ALLOC: ioctl DMA_HEAP_IOCTL_ALLOC failed for size=%zu bytes on %s: %s\n",
                    size, path, strerror(errno));
            close(dma_heap_fd);
            continue;
        }

        close(dma_heap_fd);

        buffer.fd = buf_data.fd;
        buffer.virt_addr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, buffer.fd, 0);

        if (buffer.virt_addr == MAP_FAILED) {
            fprintf(stderr, "RKNPU_DMA_ALLOC: mmap failed on %s: %s\n", path, strerror(errno));
            close(buffer.fd);
            buffer.fd = -1;
            buffer.virt_addr = nullptr;
            continue;
        }

        return buffer;
    }

    if (last_ioctl_errno != 0) {
        errno = last_ioctl_errno;
    } else if (last_open_errno != 0) {
        errno = last_open_errno;
        fprintf(stderr, "RKNPU_DMA_ALLOC: Failed to open any DMA heap: %s\n", strerror(errno));
    }

    return buffer;
}

void free(const DmaBuffer& buffer) {
    if (buffer.virt_addr != nullptr) {
        munmap(buffer.virt_addr, buffer.size);
    }
    if (buffer.fd >= 0) {
        close(buffer.fd);
    }
}

} // namespace rknpu2_allocation

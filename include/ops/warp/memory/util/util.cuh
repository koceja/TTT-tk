/**
 * @file
 * @brief General memory utilities not specialized for either tiles or vectors.
 */

#pragma once

namespace kittens {

/* ----------   To prevent generic addressing, PTX  ---------- */

template<typename T> struct move {
    template<typename U> __device__ static inline void lds(T& dst, U* src);
    template<typename U> __device__ static inline void sts(U* dst, T& src);
};
// unpacked types
template<> struct move<bf16> {
    template<typename U> __device__ static inline void lds(bf16& dst, U* src) {
        asm volatile("ld.shared.b16 %0, [%1];\n" : "=h"(*(uint16_t*)&dst) : "l"(src));
    }
    template<typename U> __device__ static inline void sts(U* dst, bf16& src) {
        asm volatile("st.shared.b16 [%1], %0;\n" : : "h"(*(uint16_t*)&src), "l"(dst));
    }
};
template<> struct move<half> {
    template<typename U> __device__ static inline void lds(half& dst, U* src) {
        asm volatile("ld.shared.b16 %0, [%1];\n" : "=h"(*(uint16_t*)&dst) : "l"(src));
    }
    template<typename U> __device__ static inline void sts(U* dst, half& src) {
        asm volatile("st.shared.b16 [%1], %0;\n" : : "h"(*(uint16_t*)&src), "l"(dst));
    }
};
template<> struct move<float> {
    template<typename U> __device__ static inline void lds(float& dst, U* src) {
        asm volatile("ld.shared.f32 %0, [%1];\n" : "=f"(dst) : "l"(src));
    }
    template<typename U> __device__ static inline void sts(U* dst, float& src) {
        asm volatile("st.shared.f32 [%1], %0;\n" : : "f"(src), "l"(dst));
    }
};
// packed types
template<> struct move<bf16_2> {
    template<typename U> __device__ static inline void lds(bf16_2& dst, U* src) {
        asm volatile("ld.shared.b32 %0, [%1];\n" : "=r"(*(uint32_t*)&dst) : "l"(src));
    }
    template<typename U> __device__ static inline void sts(U* dst, bf16_2& src) {
        asm volatile("st.shared.b32 [%1], %0;\n" : : "r"(*(uint32_t*)&src), "l"(dst));
    }
};
template<> struct move<half_2> {
    template<typename U> __device__ static inline void lds(half_2& dst, U* src) {
        asm volatile("ld.shared.b32 %0, [%1];\n" : "=r"(*(uint32_t*)&dst) : "l"(src));
    }
    template<typename U> __device__ static inline void sts(U* dst, half_2& src) {
        asm volatile("st.shared.b32 [%1], %0;\n" : : "r"(*(uint32_t*)&src), "l"(dst));
    }
};
template<> struct move<float2> {
    template<typename U> __device__ static inline void lds(float2& dst, U* src) {
        asm volatile("ld.shared.v2.f32 {%0, %1}, [%2];\n" : "=f"(dst.x), "=f"(dst.y) : "l"(src));
    }
    template<typename U> __device__ static inline void sts(U* dst, float2& src) {
        asm volatile("st.shared.v2.f32 [%2], {%0, %1};\n" : : "f"(src.x), "f"(src.y), "l"(dst));
    }
};

/* ----------   Generic (non-Hopper specific) barrier functions  ---------- */

using barrier = uint64_t;

/**
 * @brief Initializes a synchronization barrier with a transaction count and sets the expected number of bytes.
 *
 * This function sets up a barrier that is used to synchronize threads within a block during asynchronous operations.
 * It initializes the barrier with a thread count barrier.
 *
 * Additionally, if it is given a shared tile type, it will also call `set_bytes` to prepare for the memory transaction.
 *
 * @param[out] barrier The barrier variable to initialize.
 * @param[in] tc The thread counter for the barrier.
 */
__device__ static inline void init_barrier(barrier& bar, int thread_count, int transaction_count) {
    if (::kittens::laneid() == 0) {
        void const* const ptr = &bar;
        uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr)); 

        asm volatile (
            "mbarrier.init.shared::cta.b64 [%0], %1;\n"
            :: "r"(bar_ptr), "r"(thread_count+transaction_count)
        );
    }
}

/**
* @brief Arrives at a barrier.
*
* Marks a thread arrival at an mbarrier
*
* @param barrier Reference to the barrier variable.
* @param kPhaseBit The phase bit used for the barrier.
*/
__device__ static inline void arrive(barrier& bar, uint32_t count=1) {
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&bar)); 
    asm volatile (
        "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], %1;\n"
        :
        : "r"(mbar_ptr), "r"(count)
        : "memory"
    );
}


/**
* @brief Waits for the requested barrier phase.
*
* @param barrier Reference to the barrier variable.
* @param kPhaseBit The phase bit used for the barrier.
*/
__device__ static inline void wait(barrier& bar, int kPhaseBit) {
    void const* const ptr = &bar;
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr)); 

#ifdef KITTENS_HOPPER
    asm volatile (
        "{\n"
        ".reg .pred                P1;\n"
        "LAB_WAIT:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
        "@P1                       bra.uni DONE;\n"
        "bra.uni                   LAB_WAIT;\n"
        "DONE:\n"
        "}\n"
        :: "r"(mbar_ptr),
        "r"(kPhaseBit)
    );
#else
    asm volatile (
        "{\n"
        ".reg .pred                P1;\n"
        "LAB_WAIT:\n"
        "mbarrier.test_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
        "@P1                       bra.uni DONE;\n"
        "nanosleep.u32 5;\n" // wait a few nanoseconds on pre-Hopper architectures to save instruction issue slots
        "bra.uni                   LAB_WAIT;\n"
        "DONE:\n"
        "}\n"
        :: "r"(mbar_ptr),
        "r"(kPhaseBit)
    );
#endif
}

__device__ static inline void arrive_and_wait(barrier& bar, int kPhaseBit) {
    arrive(bar);
    wait(bar, kPhaseBit);
}

// sizeof() can be unreliable when working with references to objects
// plus, template magic allows arrays of these objects to be copied, too.
namespace detail {
template<typename T, uint32_t... dims> struct size_info;
template<ducks::st::all ST> struct size_info<ST> {
    static constexpr uint32_t elements = ST::num_elements;
    static constexpr uint32_t bytes    = ST::num_elements * sizeof(typename ST::dtype);
};
template<ducks::sv::all SV> struct size_info<SV> {
    static constexpr uint32_t elements = SV::length;
    static constexpr uint32_t bytes    = SV::length * sizeof(typename SV::dtype);
};
template<typename T, uint32_t dim, uint32_t... rest_dims> struct size_info<T, dim, rest_dims...> {
    static constexpr uint32_t elements = dim*size_info<T, rest_dims...>::elements;
    static constexpr uint32_t bytes    = dim*size_info<T, rest_dims...>::bytes;
};
}
template<typename T, uint32_t... dims> constexpr uint32_t size_elements = detail::size_info<T, dims...>::elements;
template<typename T, uint32_t... dims> constexpr uint32_t size_bytes    = detail::size_info<T, dims...>::bytes;

} // namespace kittens

#ifdef KITTENS_HOPPER
#include "tma.cuh"
#endif
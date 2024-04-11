/**
 * @file
 * @brief Warp-scope conversions on shared vectors.
 */

#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"


namespace kittens {

/**
 * @brief Copies data from one shared vector to another, converting data types if necessary.
 *
 * This function copies data from the source shared vector `src` to the destination shared vector `dst`.
 * If the data types of `src` and `dst` are the same, it performs a direct memory copy. Otherwise, it
 * converts each element from the source data type to the destination data type using the appropriate
 * converter before copying.
 *
 * @tparam SV1 The type of the destination shared vector, must satisfy the ducks::sv::all concept.
 * @tparam SV2 The type of the source shared vector, must satisfy the ducks::sv::all concept.
 * @param[out] dst The destination shared vector.
 * @param[in] src The source shared vector.
 * @note The lengths of `src` and `dst` must be equal. This is enforced at compile time.
 */
template<ducks::sv::all SV1, ducks::sv::all SV2>
__device__ static inline void copy(SV1 &dst, const SV2 &src) {
    static_assert(dst.length == src.length, "Source and destination vectors must have the same length.");
    int lane = kittens::laneid();
    if constexpr (std::is_same_v<typename SV1::dtype, typename SV2::dtype>) { // just copy the memory over at wordsize.
        #pragma unroll
        for(int i = lane; i < sizeof(dst)/sizeof(int); i+=WARP_THREADS) {
            ((int*)(dst.data))[i] = ((int*)(src.data))[i];
        }
    }
    else { // convert
        #pragma unroll
        for(int i = lane; i < dst.length; i+=WARP_THREADS) {
            dst[i] = base_types::convertor<typename SV1::dtype, typename SV2::dtype>::convert(src[i]);
        }
    }
}

/* ----------  SUBVEC  ---------- */

/**
* @brief Returns a reference to a subvec of a given shared vector
*
* @tparam subvec_tiles The length, in subtiles, of the subvec.
* @tparam SV The type of the input vector, which must satisfy the ducks::sv::all concept.
* @param src The input tile.
* @param vec_idx The index of the subtile, in units of subvec_tiles*16 elements.
* @return A reference to the subvec.
*
* @note The subvec length must evenly divide the vector length.
*/
template<int subvec_tiles, ducks::sv::all SV>
__device__ inline typename SV::subvec<subvec_tiles> &subvec_inplace(SV &src, int vec_idx) {
    return *(typename SV::subvec<subvec_tiles>*)(&src[vec_idx*kittens::TILE_DIM*subvec_tiles]);
}

}
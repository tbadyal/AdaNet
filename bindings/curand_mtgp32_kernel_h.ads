pragma Ada_2005;
pragma Style_Checks (Off);

with Interfaces.C; use Interfaces.C;
with vector_types_h;
limited with curand_mtgp32_h;

package curand_mtgp32_kernel_h is

  -- * Copyright 2010-2014 NVIDIA Corporation.  All rights reserved.
  -- *
  -- * NOTICE TO LICENSEE:
  -- *
  -- * This source code and/or documentation ("Licensed Deliverables") are
  -- * subject to NVIDIA intellectual property rights under U.S. and
  -- * international Copyright laws.
  -- *
  -- * These Licensed Deliverables contained herein is PROPRIETARY and
  -- * CONFIDENTIAL to NVIDIA and is being provided under the terms and
  -- * conditions of a form of NVIDIA software license agreement by and
  -- * between NVIDIA and Licensee ("License Agreement") or electronically
  -- * accepted by Licensee.  Notwithstanding any terms or conditions to
  -- * the contrary in the License Agreement, reproduction or disclosure
  -- * of the Licensed Deliverables to any third party without the express
  -- * written consent of NVIDIA is prohibited.
  -- *
  -- * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
  -- * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
  -- * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
  -- * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
  -- * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
  -- * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
  -- * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
  -- * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
  -- * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
  -- * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
  -- * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
  -- * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
  -- * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
  -- * OF THESE LICENSED DELIVERABLES.
  -- *
  -- * U.S. Government End Users.  These Licensed Deliverables are a
  -- * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
  -- * 1995), consisting of "commercial computer software" and "commercial
  -- * computer software documentation" as such terms are used in 48
  -- * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
  -- * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
  -- * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
  -- * U.S. Government End Users acquire the Licensed Deliverables with
  -- * only those rights set forth herein.
  -- *
  -- * Any use of the Licensed Deliverables in individual and commercial
  -- * software must include, in the user documentation and internal
  -- * comments to the code, the above Disclaimer and U.S. Government End
  -- * Users Notice.
  --  

  -- * curand_mtgp32_kernel.h
  -- *
  -- *
  -- * MTGP32-11213
  -- *
  -- * Mersenne Twister RNG for the GPU
  -- * 
  -- * The period of generated integers is 2<sup>11213</sup>-1.
  -- *
  -- * This code generates 32-bit unsigned integers, and
  -- * single precision floating point numbers uniformly distributed 
  -- * in the range [1, 2). (float r; 1.0 <= r < 2.0)
  --  

  -- * Copyright (c) 2009, 2010 Mutsuo Saito, Makoto Matsumoto and Hiroshima
  -- * University.  All rights reserved.
  -- * Copyright (c) 2011 Mutsuo Saito, Makoto Matsumoto, Hiroshima
  -- * University and University of Tokyo.  All rights reserved.
  -- *
  -- * Redistribution and use in source and binary forms, with or without
  -- * modification, are permitted provided that the following conditions are
  -- * met:
  -- * 
  -- *     * Redistributions of source code must retain the above copyright
  -- *       notice, this list of conditions and the following disclaimer.
  -- *     * Redistributions in binary form must reproduce the above
  -- *       copyright notice, this list of conditions and the following
  -- *       disclaimer in the documentation and/or other materials provided
  -- *       with the distribution.
  -- *     * Neither the name of the Hiroshima University nor the names of
  -- *       its contributors may be used to endorse or promote products
  -- *       derived from this software without specific prior written
  -- *       permission.
  -- * 
  -- * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  -- * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  -- * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  -- * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  -- * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  -- * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  -- * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  -- * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  -- * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  -- * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  -- * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
  --  

  --#if !defined(QUALIFIERS)
  --#define QUALIFIERS static inline __device__
  --#endif
  --*
  -- * \addtogroup DEVICE Device API
  -- *
  -- * @{
  --  

  -- define blockDim and threadIdx for host compatibility call
   blockDim : aliased vector_types_h.Class_dim3.dim3;  -- /usr/local/cuda-8.0/include/curand_mtgp32_kernel.h:121
   pragma Import (C, blockDim, "blockDim");

   threadIdx : aliased vector_types_h.uint3;  -- /usr/local/cuda-8.0/include/curand_mtgp32_kernel.h:122
   pragma Import (C, threadIdx, "threadIdx");

  -- * The function of the recursion formula calculation.
  -- *
  -- * @param[in] X1 the farthest part of state array.
  -- * @param[in] X2 the second farthest part of state array.
  -- * @param[in] Y a part of state array.
  -- * @param[in] bid block id.
  -- * @return output
  --  

   function para_rec
     (k : access curand_mtgp32_h.mtgp32_kernel_params_t;
      X1 : unsigned;
      X2 : unsigned;
      Y : unsigned;
      bid : int) return unsigned;  -- /usr/local/cuda-8.0/include/curand_mtgp32_kernel.h:135
   pragma Import (CPP, para_rec, "_ZL8para_recP20mtgp32_kernel_paramsjjji");

  -- * The tempering function.
  -- *
  -- * @param[in] V the output value should be tempered.
  -- * @param[in] T the tempering helper value.
  -- * @param[in] bid block id.
  -- * @return the tempered value.
  --  

   function temper
     (k : access curand_mtgp32_h.mtgp32_kernel_params_t;
      V : unsigned;
      T : unsigned;
      bid : int) return unsigned;  -- /usr/local/cuda-8.0/include/curand_mtgp32_kernel.h:153
   pragma Import (CPP, temper, "_ZL6temperP20mtgp32_kernel_paramsjji");

  -- * The tempering and converting function.
  -- * By using the preset table, converting to IEEE format
  -- * and tempering are done simultaneously.
  -- *
  -- * @param[in] V the output value should be tempered.
  -- * @param[in] T the tempering helper value.
  -- * @param[in] bid block id.
  -- * @return the tempered and converted value.
  --  

   function temper_single
     (k : access curand_mtgp32_h.mtgp32_kernel_params_t;
      V : unsigned;
      T : unsigned;
      bid : int) return unsigned;  -- /usr/local/cuda-8.0/include/curand_mtgp32_kernel.h:172
   pragma Import (CPP, temper_single, "_ZL13temper_singleP20mtgp32_kernel_paramsjji");

  --*
  -- * \brief Return 32-bits of pseudorandomness from a mtgp32 generator.
  -- *
  -- * Return 32-bits of pseudorandomness from the mtgp32 generator in \p state,
  -- * increment position of generator by the number of threads in the block.
  -- * Note the number of threads in the block can not exceed 256.
  -- *
  -- * \param state - Pointer to state to update
  -- *
  -- * \return 32-bits of pseudorandomness as an unsigned int, all bits valid to use.
  --  

   function curand (state : access curand_mtgp32_h.curandStateMtgp32_t) return unsigned;  -- /usr/local/cuda-8.0/include/curand_mtgp32_kernel.h:194
   pragma Import (CPP, curand, "_ZL6curandP17curandStateMtgp32");

  --assert( d <= 256 );
  --*
  -- * \brief Return 32-bits of pseudorandomness from a specific position in a mtgp32 generator.
  -- *
  -- * Return 32-bits of pseudorandomness from position \p index of the mtgp32 generator in \p state,
  -- * increment position of generator by \p n positions, which must be the total number of positions
  -- * upddated in the state by the thread block, for this invocation.
  -- *
  -- * Note : 
  -- * Thread indices must range from 0...\ n - 1.
  -- * The number of positions updated may not exceed 256. 
  -- * A thread block may update more than one state, but a given state may not be updated by more than one thread block.
  -- *
  -- * \param state - Pointer to state to update
  -- * \param index - Index (0..255) of the position within the state to draw from and update
  -- * \param n - The total number of postions in this state that are being updated by this invocation
  -- * 
  -- * \return 32-bits of pseudorandomness as an unsigned int, all bits valid to use.
  --  

   function curand_mtgp32_specific
     (state : access curand_mtgp32_h.curandStateMtgp32_t;
      index : unsigned_char;
      n : unsigned_char) return unsigned;  -- /usr/local/cuda-8.0/include/curand_mtgp32_kernel.h:245
   pragma Import (CPP, curand_mtgp32_specific, "_ZL22curand_mtgp32_specificP17curandStateMtgp32hh");

  --*
  -- * \brief Return a uniformly distributed float from a mtgp32 generator.
  -- *
  -- * Return a uniformly distributed float between \p 0.0f and \p 1.0f 
  -- * from the mtgp32 generator in \p state, increment position of generator.
  -- * Output range excludes \p 0.0f but includes \p 1.0f.  Denormalized floating
  -- * point outputs are never returned.
  -- *
  -- * Note: This alternate derivation of a uniform float is provided for completeness 
  -- * with the original source
  -- *
  -- * \param state - Pointer to state to update
  -- *
  -- * \return uniformly distributed float between \p 0.0f and \p 1.0f
  --  

   function curand_mtgp32_single (state : access curand_mtgp32_h.curandStateMtgp32_t) return float;  -- /usr/local/cuda-8.0/include/curand_mtgp32_kernel.h:289
   pragma Import (CPP, curand_mtgp32_single, "_ZL20curand_mtgp32_singleP17curandStateMtgp32");

  --assert( d <= 256 );
  --*
  -- * \brief Return a uniformly distributed float from a specific position in a mtgp32 generator.
  -- *
  -- * Return a uniformly distributed float between \p 0.0f and \p 1.0f 
  -- * from position \p index of the mtgp32 generator in \p state, and
  -- * increment position of generator by \p n positions, which must be the total number of positions
  -- * upddated in the state by the thread block, for this invocation. 
  -- * Output range excludes \p 0.0f but includes \p 1.0f.  Denormalized floating
  -- * point outputs are never returned.
  -- *
  -- * Note 1:  
  -- * Thread indices must range from 0...\p n - 1.
  -- * The number of positions updated may not exceed 256. 
  -- * A thread block may update more than one state, but a given state may not be updated by more than one thread block.
  -- *
  -- * Note 2: This alternate derivation of a uniform float is provided for completeness 
  -- * with the original source
  -- *
  -- * \param state - Pointer to state to update
  -- * \param index - Index (0..255) of the position within the state to draw from and update
  -- * \param n - The total number of postions in this state that are being updated by this invocation
  -- * 
  -- * \return uniformly distributed float between \p 0.0f and \p 1.0f
  --  

   function curand_mtgp32_single_specific
     (state : access curand_mtgp32_h.curandStateMtgp32_t;
      index : unsigned_char;
      n : unsigned_char) return float;  -- /usr/local/cuda-8.0/include/curand_mtgp32_kernel.h:351
   pragma Import (CPP, curand_mtgp32_single_specific, "_ZL29curand_mtgp32_single_specificP17curandStateMtgp32hh");

  --* @}  
end curand_mtgp32_kernel_h;

pragma Ada_2005;
pragma Style_Checks (Off);

with Interfaces.C; use Interfaces.C;

package curand_mtgp32_h is

   MTGPDC_MEXP : constant := 11213;  --  /usr/local/cuda-8.0/include/curand_mtgp32.h:98
   MTGPDC_N : constant := 351;  --  /usr/local/cuda-8.0/include/curand_mtgp32.h:99
   MTGPDC_FLOOR_2P : constant := 256;  --  /usr/local/cuda-8.0/include/curand_mtgp32.h:100
   MTGPDC_CEIL_2P : constant := 512;  --  /usr/local/cuda-8.0/include/curand_mtgp32.h:101
   --  unsupported macro: MTGPDC_PARAM_TABLE mtgp32dc_params_fast_11213

   MTGP32_STATE_SIZE : constant := 1024;  --  /usr/local/cuda-8.0/include/curand_mtgp32.h:103
   MTGP32_STATE_MASK : constant := 1023;  --  /usr/local/cuda-8.0/include/curand_mtgp32.h:104
   CURAND_NUM_MTGP32_PARAMS : constant := 200;  --  /usr/local/cuda-8.0/include/curand_mtgp32.h:105
   MEXP : constant := 11213;  --  /usr/local/cuda-8.0/include/curand_mtgp32.h:106
   --  unsupported macro: THREAD_NUM MTGPDC_FLOOR_2P
   --  unsupported macro: LARGE_SIZE (THREAD_NUM * 3)
   --  unsupported macro: BLOCK_NUM_MAX CURAND_NUM_MTGP32_PARAMS

   TBL_SIZE : constant := 16;  --  /usr/local/cuda-8.0/include/curand_mtgp32.h:110

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

  -- * @file curand_mtgp32.h
  -- *
  -- * @brief Mersenne Twister for Graphic Processors (mtgp32), which
  -- * generates 32-bit unsigned integers and single precision floating
  -- * point numbers based on IEEE 754 format.
  -- *
  -- * @author Mutsuo Saito (Hiroshima University)
  -- * @author Makoto Matsumoto (Hiroshima University)
  -- *
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

  --*
  -- * \addtogroup DEVICE Device API
  -- *
  -- * @{
  --  

  -- * \struct MTGP32_PARAMS_FAST_T
  -- * MTGP32 parameters.
  -- * Some element is redundant to keep structure simple.
  -- *
  -- * \b pos is a pick up position which is selected to have good
  -- * performance on graphic processors.  3 < \b pos < Q, where Q is a
  -- * maximum number such that the size of status array - Q is a power of
  -- * 2.  For example, when \b mexp is 44497, size of 32-bit status array
  -- * is 696, and Q is 184, then \b pos is between 4 and 183. This means
  -- * 512 parallel calculations is allowed when \b mexp is 44497.
  -- *
  -- * \b poly_sha1 is SHA1 digest of the characteristic polynomial of
  -- * state transition function. SHA1 is calculated based on printing
  -- * form of the polynomial. This is important when we use parameters
  -- * generated by the dynamic creator which
  -- *
  -- * \b mask This is a mask to make the dimension of state space have
  -- * just Mersenne Prime. This is redundant.
  --  

  --< Mersenne exponent. This is redundant.  
   type mtgp32_params_fast_tbl_array is array (0 .. 15) of aliased unsigned;
   type mtgp32_params_fast_tmp_tbl_array is array (0 .. 15) of aliased unsigned;
   type mtgp32_params_fast_flt_tmp_tbl_array is array (0 .. 15) of aliased unsigned;
   type mtgp32_params_fast_poly_sha1_array is array (0 .. 20) of aliased unsigned_char;
   type mtgp32_params_fast is record
      mexp : aliased int;  -- /usr/local/cuda-8.0/include/curand_mtgp32.h:142
      pos : aliased int;  -- /usr/local/cuda-8.0/include/curand_mtgp32.h:143
      sh1 : aliased int;  -- /usr/local/cuda-8.0/include/curand_mtgp32.h:144
      sh2 : aliased int;  -- /usr/local/cuda-8.0/include/curand_mtgp32.h:145
      tbl : aliased mtgp32_params_fast_tbl_array;  -- /usr/local/cuda-8.0/include/curand_mtgp32.h:146
      tmp_tbl : aliased mtgp32_params_fast_tmp_tbl_array;  -- /usr/local/cuda-8.0/include/curand_mtgp32.h:147
      flt_tmp_tbl : aliased mtgp32_params_fast_flt_tmp_tbl_array;  -- /usr/local/cuda-8.0/include/curand_mtgp32.h:148
      mask : aliased unsigned;  -- /usr/local/cuda-8.0/include/curand_mtgp32.h:150
      poly_sha1 : aliased mtgp32_params_fast_poly_sha1_array;  -- /usr/local/cuda-8.0/include/curand_mtgp32.h:151
   end record;
   pragma Convention (C_Pass_By_Copy, mtgp32_params_fast);  -- /usr/local/cuda-8.0/include/curand_mtgp32.h:141

  --< pick up position.  
  --< shift value 1. 0 < sh1 < 32.  
  --< shift value 2. 0 < sh2 < 32.  
  --< a small matrix.  
  --< a small matrix for tempering.  
  --< a small matrix for tempering and
  --                 converting to float.  

  --< This is a mask for state space  
  --< SHA1 digest  
  --* \cond UNHIDE_TYPEDEFS  
   subtype mtgp32_params_fast_t is mtgp32_params_fast;

  --* \endcond  
  -- * Generator Parameters.
  --  

   type mtgp32_kernel_params_pos_tbl_array is array (0 .. 199) of aliased unsigned;
   type mtgp32_kernel_params_param_tbl_array is array (0 .. 199, 0 .. 15) of aliased unsigned;
   type mtgp32_kernel_params_temper_tbl_array is array (0 .. 199, 0 .. 15) of aliased unsigned;
   type mtgp32_kernel_params_single_temper_tbl_array is array (0 .. 199, 0 .. 15) of aliased unsigned;
   type mtgp32_kernel_params_sh1_tbl_array is array (0 .. 199) of aliased unsigned;
   type mtgp32_kernel_params_sh2_tbl_array is array (0 .. 199) of aliased unsigned;
   type mtgp32_kernel_params_mask_array is array (0 .. 0) of aliased unsigned;
   type mtgp32_kernel_params is record
      pos_tbl : aliased mtgp32_kernel_params_pos_tbl_array;  -- /usr/local/cuda-8.0/include/curand_mtgp32.h:163
      param_tbl : aliased mtgp32_kernel_params_param_tbl_array;  -- /usr/local/cuda-8.0/include/curand_mtgp32.h:164
      temper_tbl : aliased mtgp32_kernel_params_temper_tbl_array;  -- /usr/local/cuda-8.0/include/curand_mtgp32.h:165
      single_temper_tbl : aliased mtgp32_kernel_params_single_temper_tbl_array;  -- /usr/local/cuda-8.0/include/curand_mtgp32.h:166
      sh1_tbl : aliased mtgp32_kernel_params_sh1_tbl_array;  -- /usr/local/cuda-8.0/include/curand_mtgp32.h:167
      sh2_tbl : aliased mtgp32_kernel_params_sh2_tbl_array;  -- /usr/local/cuda-8.0/include/curand_mtgp32.h:168
      mask : aliased mtgp32_kernel_params_mask_array;  -- /usr/local/cuda-8.0/include/curand_mtgp32.h:169
   end record;
   pragma Convention (C_Pass_By_Copy, mtgp32_kernel_params);  -- /usr/local/cuda-8.0/include/curand_mtgp32.h:162

  --* \cond UNHIDE_TYPEDEFS  
   subtype mtgp32_kernel_params_t is mtgp32_kernel_params;

  --* \endcond  
  -- * kernel I/O
  -- * This structure must be initialized before first use.
  --  

  -- MTGP (Mersenne Twister) RNG  
  -- This generator uses the Mersenne Twister algorithm of
  -- * http://arxiv.org/abs/1005.4973v2
  -- * Has period 2^11213.
  -- 

  --*
  -- * CURAND MTGP32 state 
  --  

   type curandStateMtgp32_s_array is array (0 .. 1023) of aliased unsigned;
   type curandStateMtgp32 is record
      s : aliased curandStateMtgp32_s_array;  -- /usr/local/cuda-8.0/include/curand_mtgp32.h:195
      offset : aliased int;  -- /usr/local/cuda-8.0/include/curand_mtgp32.h:196
      pIdx : aliased int;  -- /usr/local/cuda-8.0/include/curand_mtgp32.h:197
      k : access mtgp32_kernel_params_t;  -- /usr/local/cuda-8.0/include/curand_mtgp32.h:198
      precise_double_flag : aliased int;  -- /usr/local/cuda-8.0/include/curand_mtgp32.h:199
   end record;
   pragma Convention (C_Pass_By_Copy, curandStateMtgp32);  -- /usr/local/cuda-8.0/include/curand_mtgp32.h:194

  -- * CURAND MTGP32 state 
  --  

  --* \cond UNHIDE_TYPEDEFS  
   subtype curandStateMtgp32_t is curandStateMtgp32;

  --* \endcond  
  --* @}  
end curand_mtgp32_h;

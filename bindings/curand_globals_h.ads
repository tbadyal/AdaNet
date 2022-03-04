pragma Ada_2005;
pragma Style_Checks (Off);

with Interfaces.C; use Interfaces.C;

package curand_globals_h is

   MAX_XOR_N : constant := (5);  --  /usr/local/cuda-8.0/include/curand_globals.h:51
   SKIPAHEAD_BLOCKSIZE : constant := (4);  --  /usr/local/cuda-8.0/include/curand_globals.h:52
   --  unsupported macro: SKIPAHEAD_MASK ((1<<SKIPAHEAD_BLOCKSIZE)-1)

   CURAND_2POW32 : constant := (4294967296.f);  --  /usr/local/cuda-8.0/include/curand_globals.h:54
   CURAND_2POW32_DOUBLE : constant := (4294967296.);  --  /usr/local/cuda-8.0/include/curand_globals.h:55
   CURAND_2POW32_INV : constant := (2.3283064e-10f);  --  /usr/local/cuda-8.0/include/curand_globals.h:56
   CURAND_2POW32_INV_DOUBLE : constant := (2.3283064365386963e-10);  --  /usr/local/cuda-8.0/include/curand_globals.h:57
   CURAND_2POW53_INV_DOUBLE : constant := (1.1102230246251565e-16);  --  /usr/local/cuda-8.0/include/curand_globals.h:58
   CURAND_2POW32_INV_2PI : constant := (2.3283064e-10f * 6.2831855f);  --  /usr/local/cuda-8.0/include/curand_globals.h:59
   CURAND_2PI : constant := (6.2831855f);  --  /usr/local/cuda-8.0/include/curand_globals.h:60
   CURAND_2POW53_INV_2PI_DOUBLE : constant := (1.1102230246251565e-16 * 6.2831853071795860);  --  /usr/local/cuda-8.0/include/curand_globals.h:61
   CURAND_PI_DOUBLE : constant := (3.1415926535897932);  --  /usr/local/cuda-8.0/include/curand_globals.h:62
   CURAND_2PI_DOUBLE : constant := (6.2831853071795860);  --  /usr/local/cuda-8.0/include/curand_globals.h:63
   CURAND_SQRT2 : constant := (-1.4142135f);  --  /usr/local/cuda-8.0/include/curand_globals.h:64
   CURAND_SQRT2_DOUBLE : constant := (-1.4142135623730951);  --  /usr/local/cuda-8.0/include/curand_globals.h:65

   SOBOL64_ITR_BINARY_DIVIDE : constant := 2;  --  /usr/local/cuda-8.0/include/curand_globals.h:67
   SOBOL_M2_BINARY_DIVIDE : constant := 10;  --  /usr/local/cuda-8.0/include/curand_globals.h:68
   MTGP32_M2_BINARY_DIVIDE : constant := 32;  --  /usr/local/cuda-8.0/include/curand_globals.h:69
   MAX_LAMBDA : constant := 400000;  --  /usr/local/cuda-8.0/include/curand_globals.h:70
   MIN_GAUSS_LAMBDA : constant := 2000;  --  /usr/local/cuda-8.0/include/curand_globals.h:71

  -- Copyright 2010-2014 NVIDIA Corporation.  All rights reserved.
  --  *
  --  * NOTICE TO LICENSEE:
  --  *
  --  * The source code and/or documentation ("Licensed Deliverables") are
  --  * subject to NVIDIA intellectual property rights under U.S. and
  --  * international Copyright laws.
  --  *
  --  * The Licensed Deliverables contained herein are PROPRIETARY and
  --  * CONFIDENTIAL to NVIDIA and are being provided under the terms and
  --  * conditions of a form of NVIDIA software license agreement by and
  --  * between NVIDIA and Licensee ("License Agreement") or electronically
  --  * accepted by Licensee.  Notwithstanding any terms or conditions to
  --  * the contrary in the License Agreement, reproduction or disclosure
  --  * of the Licensed Deliverables to any third party without the express
  --  * written consent of NVIDIA is prohibited.
  --  *
  --  * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
  --  * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
  --  * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  THEY ARE
  --  * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
  --  * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
  --  * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
  --  * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
  --  * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
  --  * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
  --  * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
  --  * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
  --  * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
  --  * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
  --  * OF THESE LICENSED DELIVERABLES.
  --  *
  --  * U.S. Government End Users.  These Licensed Deliverables are a
  --  * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
  --  * 1995), consisting of "commercial computer software" and "commercial
  --  * computer software documentation" as such terms are used in 48
  --  * C.F.R. 12.212 (SEPT 1995) and are provided to the U.S. Government
  --  * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
  --  * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
  --  * U.S. Government End Users acquire the Licensed Deliverables with
  --  * only those rights set forth herein.
  --  *
  --  * Any use of the Licensed Deliverables in individual and commercial
  --  * software must include, in the user documentation and internal
  --  * comments to the code, the above Disclaimer and U.S. Government End
  --  * Users Notice.
  --   

   type normal_args_st is record
      mean : aliased float;  -- /usr/local/cuda-8.0/include/curand_globals.h:74
      stddev : aliased float;  -- /usr/local/cuda-8.0/include/curand_globals.h:75
   end record;
   pragma Convention (C_Pass_By_Copy, normal_args_st);  -- /usr/local/cuda-8.0/include/curand_globals.h:73

   subtype normal_args_t is normal_args_st;

   type normal_args_double_st is record
      mean : aliased double;  -- /usr/local/cuda-8.0/include/curand_globals.h:81
      stddev : aliased double;  -- /usr/local/cuda-8.0/include/curand_globals.h:82
   end record;
   pragma Convention (C_Pass_By_Copy, normal_args_double_st);  -- /usr/local/cuda-8.0/include/curand_globals.h:80

   subtype normal_args_double_t is normal_args_double_st;

end curand_globals_h;

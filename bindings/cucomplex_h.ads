pragma Ada_2005;
pragma Style_Checks (Off);

with Interfaces.C; use Interfaces.C;
with vector_types_h;

package cuComplex_h is

  -- * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
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

  -- When trying to include C header file in C++ Code extern "C" is required
  -- * But the Standard QNX headers already have ifdef extern in them when compiling C++ Code
  -- * extern "C" cannot be nested
  -- * Hence keep the header out of extern "C" block
  --  

  -- import fabsf, sqrt  
   subtype cuFloatComplex is vector_types_h.float2;

   function cuCrealf (x : cuFloatComplex) return float;  -- /usr/local/cuda-8.0/include/cuComplex.h:69
   pragma Import (C, cuCrealf, "cuCrealf");

   function cuCimagf (x : cuFloatComplex) return float;  -- /usr/local/cuda-8.0/include/cuComplex.h:74
   pragma Import (C, cuCimagf, "cuCimagf");

   function make_cuFloatComplex (r : float; i : float) return cuFloatComplex;  -- /usr/local/cuda-8.0/include/cuComplex.h:79
   pragma Import (C, make_cuFloatComplex, "make_cuFloatComplex");

   function cuConjf (x : cuFloatComplex) return cuFloatComplex;  -- /usr/local/cuda-8.0/include/cuComplex.h:88
   pragma Import (C, cuConjf, "cuConjf");

   function cuCaddf (x : cuFloatComplex; y : cuFloatComplex) return cuFloatComplex;  -- /usr/local/cuda-8.0/include/cuComplex.h:92
   pragma Import (C, cuCaddf, "cuCaddf");

   function cuCsubf (x : cuFloatComplex; y : cuFloatComplex) return cuFloatComplex;  -- /usr/local/cuda-8.0/include/cuComplex.h:99
   pragma Import (C, cuCsubf, "cuCsubf");

  -- This implementation could suffer from intermediate overflow even though
  -- * the final result would be in range. However, various implementations do
  -- * not guard against this (presumably to avoid losing performance), so we 
  -- * don't do it either to stay competitive.
  --  

   function cuCmulf (x : cuFloatComplex; y : cuFloatComplex) return cuFloatComplex;  -- /usr/local/cuda-8.0/include/cuComplex.h:111
   pragma Import (C, cuCmulf, "cuCmulf");

  -- This implementation guards against intermediate underflow and overflow
  -- * by scaling. Such guarded implementations are usually the default for
  -- * complex library implementations, with some also offering an unguarded,
  -- * faster version.
  --  

   function cuCdivf (x : cuFloatComplex; y : cuFloatComplex) return cuFloatComplex;  -- /usr/local/cuda-8.0/include/cuComplex.h:127
   pragma Import (C, cuCdivf, "cuCdivf");

  -- 
  -- * We would like to call hypotf(), but it's not available on all platforms.
  -- * This discrete implementation guards against intermediate underflow and 
  -- * overflow by scaling. Otherwise we would lose half the exponent range. 
  -- * There are various ways of doing guarded computation. For now chose the 
  -- * simplest and fastest solution, however this may suffer from inaccuracies 
  -- * if sqrt and division are not IEEE compliant. 
  --  

   function cuCabsf (x : cuFloatComplex) return float;  -- /usr/local/cuda-8.0/include/cuComplex.h:152
   pragma Import (C, cuCabsf, "cuCabsf");

  -- Double precision  
   type cuDoubleComplex is new vector_types_h.double2;

   function cuCreal (x : cuDoubleComplex) return double;  -- /usr/local/cuda-8.0/include/cuComplex.h:178
   pragma Import (C, cuCreal, "cuCreal");

   function cuCimag (x : cuDoubleComplex) return double;  -- /usr/local/cuda-8.0/include/cuComplex.h:183
   pragma Import (C, cuCimag, "cuCimag");

   function make_cuDoubleComplex (r : double; i : double) return cuDoubleComplex;  -- /usr/local/cuda-8.0/include/cuComplex.h:188
   pragma Import (C, make_cuDoubleComplex, "make_cuDoubleComplex");

   function cuConj (x : cuDoubleComplex) return cuDoubleComplex;  -- /usr/local/cuda-8.0/include/cuComplex.h:197
   pragma Import (C, cuConj, "cuConj");

   function cuCadd (x : cuDoubleComplex; y : cuDoubleComplex) return cuDoubleComplex;  -- /usr/local/cuda-8.0/include/cuComplex.h:202
   pragma Import (C, cuCadd, "cuCadd");

   function cuCsub (x : cuDoubleComplex; y : cuDoubleComplex) return cuDoubleComplex;  -- /usr/local/cuda-8.0/include/cuComplex.h:209
   pragma Import (C, cuCsub, "cuCsub");

  -- This implementation could suffer from intermediate overflow even though
  -- * the final result would be in range. However, various implementations do
  -- * not guard against this (presumably to avoid losing performance), so we 
  -- * don't do it either to stay competitive.
  --  

   function cuCmul (x : cuDoubleComplex; y : cuDoubleComplex) return cuDoubleComplex;  -- /usr/local/cuda-8.0/include/cuComplex.h:221
   pragma Import (C, cuCmul, "cuCmul");

  -- This implementation guards against intermediate underflow and overflow
  -- * by scaling. Such guarded implementations are usually the default for
  -- * complex library implementations, with some also offering an unguarded,
  -- * faster version.
  --  

   function cuCdiv (x : cuDoubleComplex; y : cuDoubleComplex) return cuDoubleComplex;  -- /usr/local/cuda-8.0/include/cuComplex.h:237
   pragma Import (C, cuCdiv, "cuCdiv");

  -- This implementation guards against intermediate underflow and overflow
  -- * by scaling. Otherwise we would lose half the exponent range. There are
  -- * various ways of doing guarded computation. For now chose the simplest
  -- * and fastest solution, however this may suffer from inaccuracies if sqrt
  -- * and division are not IEEE compliant.
  --  

   function cuCabs (x : cuDoubleComplex) return double;  -- /usr/local/cuda-8.0/include/cuComplex.h:260
   pragma Import (C, cuCabs, "cuCabs");

  -- aliases  
   type cuComplex is new vector_types_h.float2;

   function make_cuComplex (x : float; y : float) return cuComplex;  -- /usr/local/cuda-8.0/include/cuComplex.h:290
   pragma Import (CPP, make_cuComplex, "_ZL14make_cuComplexff");

  -- float-to-double promotion  
   function cuComplexFloatToDouble (c : cuFloatComplex) return cuDoubleComplex;  -- /usr/local/cuda-8.0/include/cuComplex.h:297
   pragma Import (CPP, cuComplexFloatToDouble, "_ZL22cuComplexFloatToDouble6float2");

   function cuComplexDoubleToFloat (c : cuDoubleComplex) return cuFloatComplex;  -- /usr/local/cuda-8.0/include/cuComplex.h:303
   pragma Import (CPP, cuComplexDoubleToFloat, "_ZL22cuComplexDoubleToFloat7double2");

   function cuCfmaf
     (x : cuComplex;
      y : cuComplex;
      d : cuComplex) return cuComplex;  -- /usr/local/cuda-8.0/include/cuComplex.h:310
   pragma Import (CPP, cuCfmaf, "_ZL7cuCfmaf6float2S_S_");

   function cuCfma
     (x : cuDoubleComplex;
      y : cuDoubleComplex;
      d : cuDoubleComplex) return cuDoubleComplex;  -- /usr/local/cuda-8.0/include/cuComplex.h:324
   pragma Import (CPP, cuCfma, "_ZL6cuCfma7double2S_S_");

end cuComplex_h;

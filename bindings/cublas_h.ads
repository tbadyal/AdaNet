pragma Ada_2005;
pragma Style_Checks (Off);

with Interfaces.C; use Interfaces.C;
with cublas_api_h;
with System;
with driver_types_h;
with cuComplex_h;

package cublas_h is

   --  unsupported macro: cublasStatus cublasStatus_t
  -- * Copyright 1993-2014 NVIDIA Corporation.  All rights reserved.
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

  -- * This is the public header file for the CUBLAS library, defining the API
  -- *
  -- * CUBLAS is an implementation of BLAS (Basic Linear Algebra Subroutines) 
  -- * on top of the CUDA runtime. 
  --  

  -- CUBLAS data types  
   function cublasInit return cublas_api_h.cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas.h:86
   pragma Import (C, cublasInit, "cublasInit");

   function cublasShutdown return cublas_api_h.cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas.h:87
   pragma Import (C, cublasShutdown, "cublasShutdown");

   function cublasGetError return cublas_api_h.cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas.h:88
   pragma Import (C, cublasGetError, "cublasGetError");

   function cublasGetVersion (version : access int) return cublas_api_h.cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas.h:90
   pragma Import (C, cublasGetVersion, "cublasGetVersion");

   function cublasAlloc
     (n : int;
      elemSize : int;
      devicePtr : System.Address) return cublas_api_h.cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas.h:91
   pragma Import (C, cublasAlloc, "cublasAlloc");

   function cublasFree (devicePtr : System.Address) return cublas_api_h.cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas.h:93
   pragma Import (C, cublasFree, "cublasFree");

   function cublasSetKernelStream (stream : driver_types_h.cudaStream_t) return cublas_api_h.cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas.h:96
   pragma Import (C, cublasSetKernelStream, "cublasSetKernelStream");

  -- ---------------- CUBLAS BLAS1 functions ----------------  
  -- NRM2  
   function cublasSnrm2
     (n : int;
      x : access float;
      incx : int) return float;  -- /usr/local/cuda-8.0/include/cublas.h:102
   pragma Import (C, cublasSnrm2, "cublasSnrm2");

   function cublasDnrm2
     (n : int;
      x : access double;
      incx : int) return double;  -- /usr/local/cuda-8.0/include/cublas.h:103
   pragma Import (C, cublasDnrm2, "cublasDnrm2");

   function cublasScnrm2
     (n : int;
      x : access constant cuComplex_h.cuComplex;
      incx : int) return float;  -- /usr/local/cuda-8.0/include/cublas.h:104
   pragma Import (C, cublasScnrm2, "cublasScnrm2");

   function cublasDznrm2
     (n : int;
      x : access constant cuComplex_h.cuDoubleComplex;
      incx : int) return double;  -- /usr/local/cuda-8.0/include/cublas.h:105
   pragma Import (C, cublasDznrm2, "cublasDznrm2");

  -------------------------------------------------------------------------- 
  -- DOT  
   function cublasSdot
     (n : int;
      x : access float;
      incx : int;
      y : access float;
      incy : int) return float;  -- /usr/local/cuda-8.0/include/cublas.h:108
   pragma Import (C, cublasSdot, "cublasSdot");

   function cublasDdot
     (n : int;
      x : access double;
      incx : int;
      y : access double;
      incy : int) return double;  -- /usr/local/cuda-8.0/include/cublas.h:110
   pragma Import (C, cublasDdot, "cublasDdot");

   function cublasCdotu
     (n : int;
      x : access constant cuComplex_h.cuComplex;
      incx : int;
      y : access constant cuComplex_h.cuComplex;
      incy : int) return cuComplex_h.cuComplex;  -- /usr/local/cuda-8.0/include/cublas.h:112
   pragma Import (C, cublasCdotu, "cublasCdotu");

   function cublasCdotc
     (n : int;
      x : access constant cuComplex_h.cuComplex;
      incx : int;
      y : access constant cuComplex_h.cuComplex;
      incy : int) return cuComplex_h.cuComplex;  -- /usr/local/cuda-8.0/include/cublas.h:114
   pragma Import (C, cublasCdotc, "cublasCdotc");

   function cublasZdotu
     (n : int;
      x : access constant cuComplex_h.cuDoubleComplex;
      incx : int;
      y : access constant cuComplex_h.cuDoubleComplex;
      incy : int) return cuComplex_h.cuDoubleComplex;  -- /usr/local/cuda-8.0/include/cublas.h:116
   pragma Import (C, cublasZdotu, "cublasZdotu");

   function cublasZdotc
     (n : int;
      x : access constant cuComplex_h.cuDoubleComplex;
      incx : int;
      y : access constant cuComplex_h.cuDoubleComplex;
      incy : int) return cuComplex_h.cuDoubleComplex;  -- /usr/local/cuda-8.0/include/cublas.h:118
   pragma Import (C, cublasZdotc, "cublasZdotc");

  -------------------------------------------------------------------------- 
  -- SCAL  
   procedure cublasSscal
     (n : int;
      alpha : float;
      x : access float;
      incx : int);  -- /usr/local/cuda-8.0/include/cublas.h:122
   pragma Import (C, cublasSscal, "cublasSscal");

   procedure cublasDscal
     (n : int;
      alpha : double;
      x : access double;
      incx : int);  -- /usr/local/cuda-8.0/include/cublas.h:123
   pragma Import (C, cublasDscal, "cublasDscal");

   procedure cublasCscal
     (n : int;
      alpha : cuComplex_h.cuComplex;
      x : access cuComplex_h.cuComplex;
      incx : int);  -- /usr/local/cuda-8.0/include/cublas.h:124
   pragma Import (C, cublasCscal, "cublasCscal");

   procedure cublasZscal
     (n : int;
      alpha : cuComplex_h.cuDoubleComplex;
      x : access cuComplex_h.cuDoubleComplex;
      incx : int);  -- /usr/local/cuda-8.0/include/cublas.h:125
   pragma Import (C, cublasZscal, "cublasZscal");

   procedure cublasCsscal
     (n : int;
      alpha : float;
      x : access cuComplex_h.cuComplex;
      incx : int);  -- /usr/local/cuda-8.0/include/cublas.h:127
   pragma Import (C, cublasCsscal, "cublasCsscal");

   procedure cublasZdscal
     (n : int;
      alpha : double;
      x : access cuComplex_h.cuDoubleComplex;
      incx : int);  -- /usr/local/cuda-8.0/include/cublas.h:128
   pragma Import (C, cublasZdscal, "cublasZdscal");

  -------------------------------------------------------------------------- 
  -- AXPY  
   procedure cublasSaxpy
     (n : int;
      alpha : float;
      x : access float;
      incx : int;
      y : access float;
      incy : int);  -- /usr/local/cuda-8.0/include/cublas.h:131
   pragma Import (C, cublasSaxpy, "cublasSaxpy");

   procedure cublasDaxpy
     (n : int;
      alpha : double;
      x : access double;
      incx : int;
      y : access double;
      incy : int);  -- /usr/local/cuda-8.0/include/cublas.h:133
   pragma Import (C, cublasDaxpy, "cublasDaxpy");

   procedure cublasCaxpy
     (n : int;
      alpha : cuComplex_h.cuComplex;
      x : access constant cuComplex_h.cuComplex;
      incx : int;
      y : access cuComplex_h.cuComplex;
      incy : int);  -- /usr/local/cuda-8.0/include/cublas.h:135
   pragma Import (C, cublasCaxpy, "cublasCaxpy");

   procedure cublasZaxpy
     (n : int;
      alpha : cuComplex_h.cuDoubleComplex;
      x : access constant cuComplex_h.cuDoubleComplex;
      incx : int;
      y : access cuComplex_h.cuDoubleComplex;
      incy : int);  -- /usr/local/cuda-8.0/include/cublas.h:137
   pragma Import (C, cublasZaxpy, "cublasZaxpy");

  -------------------------------------------------------------------------- 
  -- COPY  
   procedure cublasScopy
     (n : int;
      x : access float;
      incx : int;
      y : access float;
      incy : int);  -- /usr/local/cuda-8.0/include/cublas.h:141
   pragma Import (C, cublasScopy, "cublasScopy");

   procedure cublasDcopy
     (n : int;
      x : access double;
      incx : int;
      y : access double;
      incy : int);  -- /usr/local/cuda-8.0/include/cublas.h:143
   pragma Import (C, cublasDcopy, "cublasDcopy");

   procedure cublasCcopy
     (n : int;
      x : access constant cuComplex_h.cuComplex;
      incx : int;
      y : access cuComplex_h.cuComplex;
      incy : int);  -- /usr/local/cuda-8.0/include/cublas.h:145
   pragma Import (C, cublasCcopy, "cublasCcopy");

   procedure cublasZcopy
     (n : int;
      x : access constant cuComplex_h.cuDoubleComplex;
      incx : int;
      y : access cuComplex_h.cuDoubleComplex;
      incy : int);  -- /usr/local/cuda-8.0/include/cublas.h:147
   pragma Import (C, cublasZcopy, "cublasZcopy");

  -------------------------------------------------------------------------- 
  -- SWAP  
   procedure cublasSswap
     (n : int;
      x : access float;
      incx : int;
      y : access float;
      incy : int);  -- /usr/local/cuda-8.0/include/cublas.h:151
   pragma Import (C, cublasSswap, "cublasSswap");

   procedure cublasDswap
     (n : int;
      x : access double;
      incx : int;
      y : access double;
      incy : int);  -- /usr/local/cuda-8.0/include/cublas.h:152
   pragma Import (C, cublasDswap, "cublasDswap");

   procedure cublasCswap
     (n : int;
      x : access cuComplex_h.cuComplex;
      incx : int;
      y : access cuComplex_h.cuComplex;
      incy : int);  -- /usr/local/cuda-8.0/include/cublas.h:153
   pragma Import (C, cublasCswap, "cublasCswap");

   procedure cublasZswap
     (n : int;
      x : access cuComplex_h.cuDoubleComplex;
      incx : int;
      y : access cuComplex_h.cuDoubleComplex;
      incy : int);  -- /usr/local/cuda-8.0/include/cublas.h:154
   pragma Import (C, cublasZswap, "cublasZswap");

  -------------------------------------------------------------------------- 
  -- AMAX  
   function cublasIsamax
     (n : int;
      x : access float;
      incx : int) return int;  -- /usr/local/cuda-8.0/include/cublas.h:157
   pragma Import (C, cublasIsamax, "cublasIsamax");

   function cublasIdamax
     (n : int;
      x : access double;
      incx : int) return int;  -- /usr/local/cuda-8.0/include/cublas.h:158
   pragma Import (C, cublasIdamax, "cublasIdamax");

   function cublasIcamax
     (n : int;
      x : access constant cuComplex_h.cuComplex;
      incx : int) return int;  -- /usr/local/cuda-8.0/include/cublas.h:159
   pragma Import (C, cublasIcamax, "cublasIcamax");

   function cublasIzamax
     (n : int;
      x : access constant cuComplex_h.cuDoubleComplex;
      incx : int) return int;  -- /usr/local/cuda-8.0/include/cublas.h:160
   pragma Import (C, cublasIzamax, "cublasIzamax");

  -------------------------------------------------------------------------- 
  -- AMIN  
   function cublasIsamin
     (n : int;
      x : access float;
      incx : int) return int;  -- /usr/local/cuda-8.0/include/cublas.h:163
   pragma Import (C, cublasIsamin, "cublasIsamin");

   function cublasIdamin
     (n : int;
      x : access double;
      incx : int) return int;  -- /usr/local/cuda-8.0/include/cublas.h:164
   pragma Import (C, cublasIdamin, "cublasIdamin");

   function cublasIcamin
     (n : int;
      x : access constant cuComplex_h.cuComplex;
      incx : int) return int;  -- /usr/local/cuda-8.0/include/cublas.h:166
   pragma Import (C, cublasIcamin, "cublasIcamin");

   function cublasIzamin
     (n : int;
      x : access constant cuComplex_h.cuDoubleComplex;
      incx : int) return int;  -- /usr/local/cuda-8.0/include/cublas.h:167
   pragma Import (C, cublasIzamin, "cublasIzamin");

  -------------------------------------------------------------------------- 
  -- ASUM  
   function cublasSasum
     (n : int;
      x : access float;
      incx : int) return float;  -- /usr/local/cuda-8.0/include/cublas.h:170
   pragma Import (C, cublasSasum, "cublasSasum");

   function cublasDasum
     (n : int;
      x : access double;
      incx : int) return double;  -- /usr/local/cuda-8.0/include/cublas.h:171
   pragma Import (C, cublasDasum, "cublasDasum");

   function cublasScasum
     (n : int;
      x : access constant cuComplex_h.cuComplex;
      incx : int) return float;  -- /usr/local/cuda-8.0/include/cublas.h:172
   pragma Import (C, cublasScasum, "cublasScasum");

   function cublasDzasum
     (n : int;
      x : access constant cuComplex_h.cuDoubleComplex;
      incx : int) return double;  -- /usr/local/cuda-8.0/include/cublas.h:173
   pragma Import (C, cublasDzasum, "cublasDzasum");

  -------------------------------------------------------------------------- 
  -- ROT  
   procedure cublasSrot
     (n : int;
      x : access float;
      incx : int;
      y : access float;
      incy : int;
      sc : float;
      ss : float);  -- /usr/local/cuda-8.0/include/cublas.h:176
   pragma Import (C, cublasSrot, "cublasSrot");

   procedure cublasDrot
     (n : int;
      x : access double;
      incx : int;
      y : access double;
      incy : int;
      sc : double;
      ss : double);  -- /usr/local/cuda-8.0/include/cublas.h:178
   pragma Import (C, cublasDrot, "cublasDrot");

   procedure cublasCrot
     (n : int;
      x : access cuComplex_h.cuComplex;
      incx : int;
      y : access cuComplex_h.cuComplex;
      incy : int;
      c : float;
      s : cuComplex_h.cuComplex);  -- /usr/local/cuda-8.0/include/cublas.h:180
   pragma Import (C, cublasCrot, "cublasCrot");

   procedure cublasZrot
     (n : int;
      x : access cuComplex_h.cuDoubleComplex;
      incx : int;
      y : access cuComplex_h.cuDoubleComplex;
      incy : int;
      sc : double;
      cs : cuComplex_h.cuDoubleComplex);  -- /usr/local/cuda-8.0/include/cublas.h:182
   pragma Import (C, cublasZrot, "cublasZrot");

   procedure cublasCsrot
     (n : int;
      x : access cuComplex_h.cuComplex;
      incx : int;
      y : access cuComplex_h.cuComplex;
      incy : int;
      c : float;
      s : float);  -- /usr/local/cuda-8.0/include/cublas.h:185
   pragma Import (C, cublasCsrot, "cublasCsrot");

   procedure cublasZdrot
     (n : int;
      x : access cuComplex_h.cuDoubleComplex;
      incx : int;
      y : access cuComplex_h.cuDoubleComplex;
      incy : int;
      c : double;
      s : double);  -- /usr/local/cuda-8.0/include/cublas.h:187
   pragma Import (C, cublasZdrot, "cublasZdrot");

  -------------------------------------------------------------------------- 
  -- ROTG  
   procedure cublasSrotg
     (sa : access float;
      sb : access float;
      sc : access float;
      ss : access float);  -- /usr/local/cuda-8.0/include/cublas.h:191
   pragma Import (C, cublasSrotg, "cublasSrotg");

   procedure cublasDrotg
     (sa : access double;
      sb : access double;
      sc : access double;
      ss : access double);  -- /usr/local/cuda-8.0/include/cublas.h:192
   pragma Import (C, cublasDrotg, "cublasDrotg");

   procedure cublasCrotg
     (ca : access cuComplex_h.cuComplex;
      cb : cuComplex_h.cuComplex;
      sc : access float;
      cs : access cuComplex_h.cuComplex);  -- /usr/local/cuda-8.0/include/cublas.h:193
   pragma Import (C, cublasCrotg, "cublasCrotg");

   procedure cublasZrotg
     (ca : access cuComplex_h.cuDoubleComplex;
      cb : cuComplex_h.cuDoubleComplex;
      sc : access double;
      cs : access cuComplex_h.cuDoubleComplex);  -- /usr/local/cuda-8.0/include/cublas.h:195
   pragma Import (C, cublasZrotg, "cublasZrotg");

  -------------------------------------------------------------------------- 
  -- ROTM  
   procedure cublasSrotm
     (n : int;
      x : access float;
      incx : int;
      y : access float;
      incy : int;
      sparam : access float);  -- /usr/local/cuda-8.0/include/cublas.h:199
   pragma Import (C, cublasSrotm, "cublasSrotm");

   procedure cublasDrotm
     (n : int;
      x : access double;
      incx : int;
      y : access double;
      incy : int;
      sparam : access double);  -- /usr/local/cuda-8.0/include/cublas.h:201
   pragma Import (C, cublasDrotm, "cublasDrotm");

  -------------------------------------------------------------------------- 
  -- ROTMG  
   procedure cublasSrotmg
     (sd1 : access float;
      sd2 : access float;
      sx1 : access float;
      sy1 : access float;
      sparam : access float);  -- /usr/local/cuda-8.0/include/cublas.h:205
   pragma Import (C, cublasSrotmg, "cublasSrotmg");

   procedure cublasDrotmg
     (sd1 : access double;
      sd2 : access double;
      sx1 : access double;
      sy1 : access double;
      sparam : access double);  -- /usr/local/cuda-8.0/include/cublas.h:207
   pragma Import (C, cublasDrotmg, "cublasDrotmg");

  -- --------------- CUBLAS BLAS2 functions  ----------------  
  -- GEMV  
   procedure cublasSgemv
     (trans : char;
      m : int;
      n : int;
      alpha : float;
      A : access float;
      lda : int;
      x : access float;
      incx : int;
      beta : float;
      y : access float;
      incy : int);  -- /usr/local/cuda-8.0/include/cublas.h:212
   pragma Import (C, cublasSgemv, "cublasSgemv");

   procedure cublasDgemv
     (trans : char;
      m : int;
      n : int;
      alpha : double;
      A : access double;
      lda : int;
      x : access double;
      incx : int;
      beta : double;
      y : access double;
      incy : int);  -- /usr/local/cuda-8.0/include/cublas.h:215
   pragma Import (C, cublasDgemv, "cublasDgemv");

   procedure cublasCgemv
     (trans : char;
      m : int;
      n : int;
      alpha : cuComplex_h.cuComplex;
      A : access constant cuComplex_h.cuComplex;
      lda : int;
      x : access constant cuComplex_h.cuComplex;
      incx : int;
      beta : cuComplex_h.cuComplex;
      y : access cuComplex_h.cuComplex;
      incy : int);  -- /usr/local/cuda-8.0/include/cublas.h:218
   pragma Import (C, cublasCgemv, "cublasCgemv");

   procedure cublasZgemv
     (trans : char;
      m : int;
      n : int;
      alpha : cuComplex_h.cuDoubleComplex;
      A : access constant cuComplex_h.cuDoubleComplex;
      lda : int;
      x : access constant cuComplex_h.cuDoubleComplex;
      incx : int;
      beta : cuComplex_h.cuDoubleComplex;
      y : access cuComplex_h.cuDoubleComplex;
      incy : int);  -- /usr/local/cuda-8.0/include/cublas.h:221
   pragma Import (C, cublasZgemv, "cublasZgemv");

  -------------------------------------------------------------------------- 
  -- GBMV  
   procedure cublasSgbmv
     (trans : char;
      m : int;
      n : int;
      kl : int;
      ku : int;
      alpha : float;
      A : access float;
      lda : int;
      x : access float;
      incx : int;
      beta : float;
      y : access float;
      incy : int);  -- /usr/local/cuda-8.0/include/cublas.h:226
   pragma Import (C, cublasSgbmv, "cublasSgbmv");

   procedure cublasDgbmv
     (trans : char;
      m : int;
      n : int;
      kl : int;
      ku : int;
      alpha : double;
      A : access double;
      lda : int;
      x : access double;
      incx : int;
      beta : double;
      y : access double;
      incy : int);  -- /usr/local/cuda-8.0/include/cublas.h:230
   pragma Import (C, cublasDgbmv, "cublasDgbmv");

   procedure cublasCgbmv
     (trans : char;
      m : int;
      n : int;
      kl : int;
      ku : int;
      alpha : cuComplex_h.cuComplex;
      A : access constant cuComplex_h.cuComplex;
      lda : int;
      x : access constant cuComplex_h.cuComplex;
      incx : int;
      beta : cuComplex_h.cuComplex;
      y : access cuComplex_h.cuComplex;
      incy : int);  -- /usr/local/cuda-8.0/include/cublas.h:234
   pragma Import (C, cublasCgbmv, "cublasCgbmv");

   procedure cublasZgbmv
     (trans : char;
      m : int;
      n : int;
      kl : int;
      ku : int;
      alpha : cuComplex_h.cuDoubleComplex;
      A : access constant cuComplex_h.cuDoubleComplex;
      lda : int;
      x : access constant cuComplex_h.cuDoubleComplex;
      incx : int;
      beta : cuComplex_h.cuDoubleComplex;
      y : access cuComplex_h.cuDoubleComplex;
      incy : int);  -- /usr/local/cuda-8.0/include/cublas.h:238
   pragma Import (C, cublasZgbmv, "cublasZgbmv");

  -------------------------------------------------------------------------- 
  -- TRMV  
   procedure cublasStrmv
     (uplo : char;
      trans : char;
      diag : char;
      n : int;
      A : access float;
      lda : int;
      x : access float;
      incx : int);  -- /usr/local/cuda-8.0/include/cublas.h:244
   pragma Import (C, cublasStrmv, "cublasStrmv");

   procedure cublasDtrmv
     (uplo : char;
      trans : char;
      diag : char;
      n : int;
      A : access double;
      lda : int;
      x : access double;
      incx : int);  -- /usr/local/cuda-8.0/include/cublas.h:246
   pragma Import (C, cublasDtrmv, "cublasDtrmv");

   procedure cublasCtrmv
     (uplo : char;
      trans : char;
      diag : char;
      n : int;
      A : access constant cuComplex_h.cuComplex;
      lda : int;
      x : access cuComplex_h.cuComplex;
      incx : int);  -- /usr/local/cuda-8.0/include/cublas.h:248
   pragma Import (C, cublasCtrmv, "cublasCtrmv");

   procedure cublasZtrmv
     (uplo : char;
      trans : char;
      diag : char;
      n : int;
      A : access constant cuComplex_h.cuDoubleComplex;
      lda : int;
      x : access cuComplex_h.cuDoubleComplex;
      incx : int);  -- /usr/local/cuda-8.0/include/cublas.h:250
   pragma Import (C, cublasZtrmv, "cublasZtrmv");

  -------------------------------------------------------------------------- 
  -- TBMV  
   procedure cublasStbmv
     (uplo : char;
      trans : char;
      diag : char;
      n : int;
      k : int;
      A : access float;
      lda : int;
      x : access float;
      incx : int);  -- /usr/local/cuda-8.0/include/cublas.h:254
   pragma Import (C, cublasStbmv, "cublasStbmv");

   procedure cublasDtbmv
     (uplo : char;
      trans : char;
      diag : char;
      n : int;
      k : int;
      A : access double;
      lda : int;
      x : access double;
      incx : int);  -- /usr/local/cuda-8.0/include/cublas.h:256
   pragma Import (C, cublasDtbmv, "cublasDtbmv");

   procedure cublasCtbmv
     (uplo : char;
      trans : char;
      diag : char;
      n : int;
      k : int;
      A : access constant cuComplex_h.cuComplex;
      lda : int;
      x : access cuComplex_h.cuComplex;
      incx : int);  -- /usr/local/cuda-8.0/include/cublas.h:258
   pragma Import (C, cublasCtbmv, "cublasCtbmv");

   procedure cublasZtbmv
     (uplo : char;
      trans : char;
      diag : char;
      n : int;
      k : int;
      A : access constant cuComplex_h.cuDoubleComplex;
      lda : int;
      x : access cuComplex_h.cuDoubleComplex;
      incx : int);  -- /usr/local/cuda-8.0/include/cublas.h:260
   pragma Import (C, cublasZtbmv, "cublasZtbmv");

  -------------------------------------------------------------------------- 
  -- TPMV  
   procedure cublasStpmv
     (uplo : char;
      trans : char;
      diag : char;
      n : int;
      AP : access float;
      x : access float;
      incx : int);  -- /usr/local/cuda-8.0/include/cublas.h:264
   pragma Import (C, cublasStpmv, "cublasStpmv");

   procedure cublasDtpmv
     (uplo : char;
      trans : char;
      diag : char;
      n : int;
      AP : access double;
      x : access double;
      incx : int);  -- /usr/local/cuda-8.0/include/cublas.h:266
   pragma Import (C, cublasDtpmv, "cublasDtpmv");

   procedure cublasCtpmv
     (uplo : char;
      trans : char;
      diag : char;
      n : int;
      AP : access constant cuComplex_h.cuComplex;
      x : access cuComplex_h.cuComplex;
      incx : int);  -- /usr/local/cuda-8.0/include/cublas.h:268
   pragma Import (C, cublasCtpmv, "cublasCtpmv");

   procedure cublasZtpmv
     (uplo : char;
      trans : char;
      diag : char;
      n : int;
      AP : access constant cuComplex_h.cuDoubleComplex;
      x : access cuComplex_h.cuDoubleComplex;
      incx : int);  -- /usr/local/cuda-8.0/include/cublas.h:270
   pragma Import (C, cublasZtpmv, "cublasZtpmv");

  -------------------------------------------------------------------------- 
  -- TRSV  
   procedure cublasStrsv
     (uplo : char;
      trans : char;
      diag : char;
      n : int;
      A : access float;
      lda : int;
      x : access float;
      incx : int);  -- /usr/local/cuda-8.0/include/cublas.h:273
   pragma Import (C, cublasStrsv, "cublasStrsv");

   procedure cublasDtrsv
     (uplo : char;
      trans : char;
      diag : char;
      n : int;
      A : access double;
      lda : int;
      x : access double;
      incx : int);  -- /usr/local/cuda-8.0/include/cublas.h:275
   pragma Import (C, cublasDtrsv, "cublasDtrsv");

   procedure cublasCtrsv
     (uplo : char;
      trans : char;
      diag : char;
      n : int;
      A : access constant cuComplex_h.cuComplex;
      lda : int;
      x : access cuComplex_h.cuComplex;
      incx : int);  -- /usr/local/cuda-8.0/include/cublas.h:277
   pragma Import (C, cublasCtrsv, "cublasCtrsv");

   procedure cublasZtrsv
     (uplo : char;
      trans : char;
      diag : char;
      n : int;
      A : access constant cuComplex_h.cuDoubleComplex;
      lda : int;
      x : access cuComplex_h.cuDoubleComplex;
      incx : int);  -- /usr/local/cuda-8.0/include/cublas.h:279
   pragma Import (C, cublasZtrsv, "cublasZtrsv");

  -------------------------------------------------------------------------- 
  -- TPSV  
   procedure cublasStpsv
     (uplo : char;
      trans : char;
      diag : char;
      n : int;
      AP : access float;
      x : access float;
      incx : int);  -- /usr/local/cuda-8.0/include/cublas.h:283
   pragma Import (C, cublasStpsv, "cublasStpsv");

   procedure cublasDtpsv
     (uplo : char;
      trans : char;
      diag : char;
      n : int;
      AP : access double;
      x : access double;
      incx : int);  -- /usr/local/cuda-8.0/include/cublas.h:286
   pragma Import (C, cublasDtpsv, "cublasDtpsv");

   procedure cublasCtpsv
     (uplo : char;
      trans : char;
      diag : char;
      n : int;
      AP : access constant cuComplex_h.cuComplex;
      x : access cuComplex_h.cuComplex;
      incx : int);  -- /usr/local/cuda-8.0/include/cublas.h:288
   pragma Import (C, cublasCtpsv, "cublasCtpsv");

   procedure cublasZtpsv
     (uplo : char;
      trans : char;
      diag : char;
      n : int;
      AP : access constant cuComplex_h.cuDoubleComplex;
      x : access cuComplex_h.cuDoubleComplex;
      incx : int);  -- /usr/local/cuda-8.0/include/cublas.h:290
   pragma Import (C, cublasZtpsv, "cublasZtpsv");

  -------------------------------------------------------------------------- 
  -- TBSV  
   procedure cublasStbsv
     (uplo : char;
      trans : char;
      diag : char;
      n : int;
      k : int;
      A : access float;
      lda : int;
      x : access float;
      incx : int);  -- /usr/local/cuda-8.0/include/cublas.h:294
   pragma Import (C, cublasStbsv, "cublasStbsv");

   procedure cublasDtbsv
     (uplo : char;
      trans : char;
      diag : char;
      n : int;
      k : int;
      A : access double;
      lda : int;
      x : access double;
      incx : int);  -- /usr/local/cuda-8.0/include/cublas.h:298
   pragma Import (C, cublasDtbsv, "cublasDtbsv");

   procedure cublasCtbsv
     (uplo : char;
      trans : char;
      diag : char;
      n : int;
      k : int;
      A : access constant cuComplex_h.cuComplex;
      lda : int;
      x : access cuComplex_h.cuComplex;
      incx : int);  -- /usr/local/cuda-8.0/include/cublas.h:301
   pragma Import (C, cublasCtbsv, "cublasCtbsv");

   procedure cublasZtbsv
     (uplo : char;
      trans : char;
      diag : char;
      n : int;
      k : int;
      A : access constant cuComplex_h.cuDoubleComplex;
      lda : int;
      x : access cuComplex_h.cuDoubleComplex;
      incx : int);  -- /usr/local/cuda-8.0/include/cublas.h:305
   pragma Import (C, cublasZtbsv, "cublasZtbsv");

  -------------------------------------------------------------------------- 
  -- SYMV/HEMV  
   procedure cublasSsymv
     (uplo : char;
      n : int;
      alpha : float;
      A : access float;
      lda : int;
      x : access float;
      incx : int;
      beta : float;
      y : access float;
      incy : int);  -- /usr/local/cuda-8.0/include/cublas.h:310
   pragma Import (C, cublasSsymv, "cublasSsymv");

   procedure cublasDsymv
     (uplo : char;
      n : int;
      alpha : double;
      A : access double;
      lda : int;
      x : access double;
      incx : int;
      beta : double;
      y : access double;
      incy : int);  -- /usr/local/cuda-8.0/include/cublas.h:313
   pragma Import (C, cublasDsymv, "cublasDsymv");

   procedure cublasChemv
     (uplo : char;
      n : int;
      alpha : cuComplex_h.cuComplex;
      A : access constant cuComplex_h.cuComplex;
      lda : int;
      x : access constant cuComplex_h.cuComplex;
      incx : int;
      beta : cuComplex_h.cuComplex;
      y : access cuComplex_h.cuComplex;
      incy : int);  -- /usr/local/cuda-8.0/include/cublas.h:316
   pragma Import (C, cublasChemv, "cublasChemv");

   procedure cublasZhemv
     (uplo : char;
      n : int;
      alpha : cuComplex_h.cuDoubleComplex;
      A : access constant cuComplex_h.cuDoubleComplex;
      lda : int;
      x : access constant cuComplex_h.cuDoubleComplex;
      incx : int;
      beta : cuComplex_h.cuDoubleComplex;
      y : access cuComplex_h.cuDoubleComplex;
      incy : int);  -- /usr/local/cuda-8.0/include/cublas.h:319
   pragma Import (C, cublasZhemv, "cublasZhemv");

  -------------------------------------------------------------------------- 
  -- SBMV/HBMV  
   procedure cublasSsbmv
     (uplo : char;
      n : int;
      k : int;
      alpha : float;
      A : access float;
      lda : int;
      x : access float;
      incx : int;
      beta : float;
      y : access float;
      incy : int);  -- /usr/local/cuda-8.0/include/cublas.h:324
   pragma Import (C, cublasSsbmv, "cublasSsbmv");

   procedure cublasDsbmv
     (uplo : char;
      n : int;
      k : int;
      alpha : double;
      A : access double;
      lda : int;
      x : access double;
      incx : int;
      beta : double;
      y : access double;
      incy : int);  -- /usr/local/cuda-8.0/include/cublas.h:327
   pragma Import (C, cublasDsbmv, "cublasDsbmv");

   procedure cublasChbmv
     (uplo : char;
      n : int;
      k : int;
      alpha : cuComplex_h.cuComplex;
      A : access constant cuComplex_h.cuComplex;
      lda : int;
      x : access constant cuComplex_h.cuComplex;
      incx : int;
      beta : cuComplex_h.cuComplex;
      y : access cuComplex_h.cuComplex;
      incy : int);  -- /usr/local/cuda-8.0/include/cublas.h:330
   pragma Import (C, cublasChbmv, "cublasChbmv");

   procedure cublasZhbmv
     (uplo : char;
      n : int;
      k : int;
      alpha : cuComplex_h.cuDoubleComplex;
      A : access constant cuComplex_h.cuDoubleComplex;
      lda : int;
      x : access constant cuComplex_h.cuDoubleComplex;
      incx : int;
      beta : cuComplex_h.cuDoubleComplex;
      y : access cuComplex_h.cuDoubleComplex;
      incy : int);  -- /usr/local/cuda-8.0/include/cublas.h:333
   pragma Import (C, cublasZhbmv, "cublasZhbmv");

  -------------------------------------------------------------------------- 
  -- SPMV/HPMV  
   procedure cublasSspmv
     (uplo : char;
      n : int;
      alpha : float;
      AP : access float;
      x : access float;
      incx : int;
      beta : float;
      y : access float;
      incy : int);  -- /usr/local/cuda-8.0/include/cublas.h:338
   pragma Import (C, cublasSspmv, "cublasSspmv");

   procedure cublasDspmv
     (uplo : char;
      n : int;
      alpha : double;
      AP : access double;
      x : access double;
      incx : int;
      beta : double;
      y : access double;
      incy : int);  -- /usr/local/cuda-8.0/include/cublas.h:341
   pragma Import (C, cublasDspmv, "cublasDspmv");

   procedure cublasChpmv
     (uplo : char;
      n : int;
      alpha : cuComplex_h.cuComplex;
      AP : access constant cuComplex_h.cuComplex;
      x : access constant cuComplex_h.cuComplex;
      incx : int;
      beta : cuComplex_h.cuComplex;
      y : access cuComplex_h.cuComplex;
      incy : int);  -- /usr/local/cuda-8.0/include/cublas.h:344
   pragma Import (C, cublasChpmv, "cublasChpmv");

   procedure cublasZhpmv
     (uplo : char;
      n : int;
      alpha : cuComplex_h.cuDoubleComplex;
      AP : access constant cuComplex_h.cuDoubleComplex;
      x : access constant cuComplex_h.cuDoubleComplex;
      incx : int;
      beta : cuComplex_h.cuDoubleComplex;
      y : access cuComplex_h.cuDoubleComplex;
      incy : int);  -- /usr/local/cuda-8.0/include/cublas.h:347
   pragma Import (C, cublasZhpmv, "cublasZhpmv");

  -------------------------------------------------------------------------- 
  -- GER  
   procedure cublasSger
     (m : int;
      n : int;
      alpha : float;
      x : access float;
      incx : int;
      y : access float;
      incy : int;
      A : access float;
      lda : int);  -- /usr/local/cuda-8.0/include/cublas.h:353
   pragma Import (C, cublasSger, "cublasSger");

   procedure cublasDger
     (m : int;
      n : int;
      alpha : double;
      x : access double;
      incx : int;
      y : access double;
      incy : int;
      A : access double;
      lda : int);  -- /usr/local/cuda-8.0/include/cublas.h:355
   pragma Import (C, cublasDger, "cublasDger");

   procedure cublasCgeru
     (m : int;
      n : int;
      alpha : cuComplex_h.cuComplex;
      x : access constant cuComplex_h.cuComplex;
      incx : int;
      y : access constant cuComplex_h.cuComplex;
      incy : int;
      A : access cuComplex_h.cuComplex;
      lda : int);  -- /usr/local/cuda-8.0/include/cublas.h:358
   pragma Import (C, cublasCgeru, "cublasCgeru");

   procedure cublasCgerc
     (m : int;
      n : int;
      alpha : cuComplex_h.cuComplex;
      x : access constant cuComplex_h.cuComplex;
      incx : int;
      y : access constant cuComplex_h.cuComplex;
      incy : int;
      A : access cuComplex_h.cuComplex;
      lda : int);  -- /usr/local/cuda-8.0/include/cublas.h:361
   pragma Import (C, cublasCgerc, "cublasCgerc");

   procedure cublasZgeru
     (m : int;
      n : int;
      alpha : cuComplex_h.cuDoubleComplex;
      x : access constant cuComplex_h.cuDoubleComplex;
      incx : int;
      y : access constant cuComplex_h.cuDoubleComplex;
      incy : int;
      A : access cuComplex_h.cuDoubleComplex;
      lda : int);  -- /usr/local/cuda-8.0/include/cublas.h:364
   pragma Import (C, cublasZgeru, "cublasZgeru");

   procedure cublasZgerc
     (m : int;
      n : int;
      alpha : cuComplex_h.cuDoubleComplex;
      x : access constant cuComplex_h.cuDoubleComplex;
      incx : int;
      y : access constant cuComplex_h.cuDoubleComplex;
      incy : int;
      A : access cuComplex_h.cuDoubleComplex;
      lda : int);  -- /usr/local/cuda-8.0/include/cublas.h:367
   pragma Import (C, cublasZgerc, "cublasZgerc");

  -------------------------------------------------------------------------- 
  -- SYR/HER  
   procedure cublasSsyr
     (uplo : char;
      n : int;
      alpha : float;
      x : access float;
      incx : int;
      A : access float;
      lda : int);  -- /usr/local/cuda-8.0/include/cublas.h:372
   pragma Import (C, cublasSsyr, "cublasSsyr");

   procedure cublasDsyr
     (uplo : char;
      n : int;
      alpha : double;
      x : access double;
      incx : int;
      A : access double;
      lda : int);  -- /usr/local/cuda-8.0/include/cublas.h:374
   pragma Import (C, cublasDsyr, "cublasDsyr");

   procedure cublasCher
     (uplo : char;
      n : int;
      alpha : float;
      x : access constant cuComplex_h.cuComplex;
      incx : int;
      A : access cuComplex_h.cuComplex;
      lda : int);  -- /usr/local/cuda-8.0/include/cublas.h:377
   pragma Import (C, cublasCher, "cublasCher");

   procedure cublasZher
     (uplo : char;
      n : int;
      alpha : double;
      x : access constant cuComplex_h.cuDoubleComplex;
      incx : int;
      A : access cuComplex_h.cuDoubleComplex;
      lda : int);  -- /usr/local/cuda-8.0/include/cublas.h:379
   pragma Import (C, cublasZher, "cublasZher");

  -------------------------------------------------------------------------- 
  -- SPR/HPR  
   procedure cublasSspr
     (uplo : char;
      n : int;
      alpha : float;
      x : access float;
      incx : int;
      AP : access float);  -- /usr/local/cuda-8.0/include/cublas.h:384
   pragma Import (C, cublasSspr, "cublasSspr");

   procedure cublasDspr
     (uplo : char;
      n : int;
      alpha : double;
      x : access double;
      incx : int;
      AP : access double);  -- /usr/local/cuda-8.0/include/cublas.h:386
   pragma Import (C, cublasDspr, "cublasDspr");

   procedure cublasChpr
     (uplo : char;
      n : int;
      alpha : float;
      x : access constant cuComplex_h.cuComplex;
      incx : int;
      AP : access cuComplex_h.cuComplex);  -- /usr/local/cuda-8.0/include/cublas.h:388
   pragma Import (C, cublasChpr, "cublasChpr");

   procedure cublasZhpr
     (uplo : char;
      n : int;
      alpha : double;
      x : access constant cuComplex_h.cuDoubleComplex;
      incx : int;
      AP : access cuComplex_h.cuDoubleComplex);  -- /usr/local/cuda-8.0/include/cublas.h:390
   pragma Import (C, cublasZhpr, "cublasZhpr");

  -------------------------------------------------------------------------- 
  -- SYR2/HER2  
   procedure cublasSsyr2
     (uplo : char;
      n : int;
      alpha : float;
      x : access float;
      incx : int;
      y : access float;
      incy : int;
      A : access float;
      lda : int);  -- /usr/local/cuda-8.0/include/cublas.h:394
   pragma Import (C, cublasSsyr2, "cublasSsyr2");

   procedure cublasDsyr2
     (uplo : char;
      n : int;
      alpha : double;
      x : access double;
      incx : int;
      y : access double;
      incy : int;
      A : access double;
      lda : int);  -- /usr/local/cuda-8.0/include/cublas.h:397
   pragma Import (C, cublasDsyr2, "cublasDsyr2");

   procedure cublasCher2
     (uplo : char;
      n : int;
      alpha : cuComplex_h.cuComplex;
      x : access constant cuComplex_h.cuComplex;
      incx : int;
      y : access constant cuComplex_h.cuComplex;
      incy : int;
      A : access cuComplex_h.cuComplex;
      lda : int);  -- /usr/local/cuda-8.0/include/cublas.h:400
   pragma Import (C, cublasCher2, "cublasCher2");

   procedure cublasZher2
     (uplo : char;
      n : int;
      alpha : cuComplex_h.cuDoubleComplex;
      x : access constant cuComplex_h.cuDoubleComplex;
      incx : int;
      y : access constant cuComplex_h.cuDoubleComplex;
      incy : int;
      A : access cuComplex_h.cuDoubleComplex;
      lda : int);  -- /usr/local/cuda-8.0/include/cublas.h:403
   pragma Import (C, cublasZher2, "cublasZher2");

  -------------------------------------------------------------------------- 
  -- SPR2/HPR2  
   procedure cublasSspr2
     (uplo : char;
      n : int;
      alpha : float;
      x : access float;
      incx : int;
      y : access float;
      incy : int;
      AP : access float);  -- /usr/local/cuda-8.0/include/cublas.h:409
   pragma Import (C, cublasSspr2, "cublasSspr2");

   procedure cublasDspr2
     (uplo : char;
      n : int;
      alpha : double;
      x : access double;
      incx : int;
      y : access double;
      incy : int;
      AP : access double);  -- /usr/local/cuda-8.0/include/cublas.h:411
   pragma Import (C, cublasDspr2, "cublasDspr2");

   procedure cublasChpr2
     (uplo : char;
      n : int;
      alpha : cuComplex_h.cuComplex;
      x : access constant cuComplex_h.cuComplex;
      incx : int;
      y : access constant cuComplex_h.cuComplex;
      incy : int;
      AP : access cuComplex_h.cuComplex);  -- /usr/local/cuda-8.0/include/cublas.h:414
   pragma Import (C, cublasChpr2, "cublasChpr2");

   procedure cublasZhpr2
     (uplo : char;
      n : int;
      alpha : cuComplex_h.cuDoubleComplex;
      x : access constant cuComplex_h.cuDoubleComplex;
      incx : int;
      y : access constant cuComplex_h.cuDoubleComplex;
      incy : int;
      AP : access cuComplex_h.cuDoubleComplex);  -- /usr/local/cuda-8.0/include/cublas.h:417
   pragma Import (C, cublasZhpr2, "cublasZhpr2");

  -- ------------------------BLAS3 Functions -------------------------------  
  -- GEMM  
   procedure cublasSgemm
     (transa : char;
      transb : char;
      m : int;
      n : int;
      k : int;
      alpha : float;
      A : access float;
      lda : int;
      B : access float;
      ldb : int;
      beta : float;
      C : access float;
      ldc : int);  -- /usr/local/cuda-8.0/include/cublas.h:422
   pragma Import (C, cublasSgemm, "cublasSgemm");

   procedure cublasDgemm
     (transa : char;
      transb : char;
      m : int;
      n : int;
      k : int;
      alpha : double;
      A : access double;
      lda : int;
      B : access double;
      ldb : int;
      beta : double;
      C : access double;
      ldc : int);  -- /usr/local/cuda-8.0/include/cublas.h:426
   pragma Import (C, cublasDgemm, "cublasDgemm");

   procedure cublasCgemm
     (transa : char;
      transb : char;
      m : int;
      n : int;
      k : int;
      alpha : cuComplex_h.cuComplex;
      A : access constant cuComplex_h.cuComplex;
      lda : int;
      B : access constant cuComplex_h.cuComplex;
      ldb : int;
      beta : cuComplex_h.cuComplex;
      C : access cuComplex_h.cuComplex;
      ldc : int);  -- /usr/local/cuda-8.0/include/cublas.h:430
   pragma Import (C, cublasCgemm, "cublasCgemm");

   procedure cublasZgemm
     (transa : char;
      transb : char;
      m : int;
      n : int;
      k : int;
      alpha : cuComplex_h.cuDoubleComplex;
      A : access constant cuComplex_h.cuDoubleComplex;
      lda : int;
      B : access constant cuComplex_h.cuDoubleComplex;
      ldb : int;
      beta : cuComplex_h.cuDoubleComplex;
      C : access cuComplex_h.cuDoubleComplex;
      ldc : int);  -- /usr/local/cuda-8.0/include/cublas.h:434
   pragma Import (C, cublasZgemm, "cublasZgemm");

  -- ------------------------------------------------------- 
  -- SYRK  
   procedure cublasSsyrk
     (uplo : char;
      trans : char;
      n : int;
      k : int;
      alpha : float;
      A : access float;
      lda : int;
      beta : float;
      C : access float;
      ldc : int);  -- /usr/local/cuda-8.0/include/cublas.h:442
   pragma Import (C, cublasSsyrk, "cublasSsyrk");

   procedure cublasDsyrk
     (uplo : char;
      trans : char;
      n : int;
      k : int;
      alpha : double;
      A : access double;
      lda : int;
      beta : double;
      C : access double;
      ldc : int);  -- /usr/local/cuda-8.0/include/cublas.h:445
   pragma Import (C, cublasDsyrk, "cublasDsyrk");

   procedure cublasCsyrk
     (uplo : char;
      trans : char;
      n : int;
      k : int;
      alpha : cuComplex_h.cuComplex;
      A : access constant cuComplex_h.cuComplex;
      lda : int;
      beta : cuComplex_h.cuComplex;
      C : access cuComplex_h.cuComplex;
      ldc : int);  -- /usr/local/cuda-8.0/include/cublas.h:449
   pragma Import (C, cublasCsyrk, "cublasCsyrk");

   procedure cublasZsyrk
     (uplo : char;
      trans : char;
      n : int;
      k : int;
      alpha : cuComplex_h.cuDoubleComplex;
      A : access constant cuComplex_h.cuDoubleComplex;
      lda : int;
      beta : cuComplex_h.cuDoubleComplex;
      C : access cuComplex_h.cuDoubleComplex;
      ldc : int);  -- /usr/local/cuda-8.0/include/cublas.h:452
   pragma Import (C, cublasZsyrk, "cublasZsyrk");

  -- -------------------------------------------------------  
  -- HERK  
   procedure cublasCherk
     (uplo : char;
      trans : char;
      n : int;
      k : int;
      alpha : float;
      A : access constant cuComplex_h.cuComplex;
      lda : int;
      beta : float;
      C : access cuComplex_h.cuComplex;
      ldc : int);  -- /usr/local/cuda-8.0/include/cublas.h:459
   pragma Import (C, cublasCherk, "cublasCherk");

   procedure cublasZherk
     (uplo : char;
      trans : char;
      n : int;
      k : int;
      alpha : double;
      A : access constant cuComplex_h.cuDoubleComplex;
      lda : int;
      beta : double;
      C : access cuComplex_h.cuDoubleComplex;
      ldc : int);  -- /usr/local/cuda-8.0/include/cublas.h:462
   pragma Import (C, cublasZherk, "cublasZherk");

  -- -------------------------------------------------------  
  -- SYR2K  
   procedure cublasSsyr2k
     (uplo : char;
      trans : char;
      n : int;
      k : int;
      alpha : float;
      A : access float;
      lda : int;
      B : access float;
      ldb : int;
      beta : float;
      C : access float;
      ldc : int);  -- /usr/local/cuda-8.0/include/cublas.h:469
   pragma Import (C, cublasSsyr2k, "cublasSsyr2k");

   procedure cublasDsyr2k
     (uplo : char;
      trans : char;
      n : int;
      k : int;
      alpha : double;
      A : access double;
      lda : int;
      B : access double;
      ldb : int;
      beta : double;
      C : access double;
      ldc : int);  -- /usr/local/cuda-8.0/include/cublas.h:473
   pragma Import (C, cublasDsyr2k, "cublasDsyr2k");

   procedure cublasCsyr2k
     (uplo : char;
      trans : char;
      n : int;
      k : int;
      alpha : cuComplex_h.cuComplex;
      A : access constant cuComplex_h.cuComplex;
      lda : int;
      B : access constant cuComplex_h.cuComplex;
      ldb : int;
      beta : cuComplex_h.cuComplex;
      C : access cuComplex_h.cuComplex;
      ldc : int);  -- /usr/local/cuda-8.0/include/cublas.h:477
   pragma Import (C, cublasCsyr2k, "cublasCsyr2k");

   procedure cublasZsyr2k
     (uplo : char;
      trans : char;
      n : int;
      k : int;
      alpha : cuComplex_h.cuDoubleComplex;
      A : access constant cuComplex_h.cuDoubleComplex;
      lda : int;
      B : access constant cuComplex_h.cuDoubleComplex;
      ldb : int;
      beta : cuComplex_h.cuDoubleComplex;
      C : access cuComplex_h.cuDoubleComplex;
      ldc : int);  -- /usr/local/cuda-8.0/include/cublas.h:482
   pragma Import (C, cublasZsyr2k, "cublasZsyr2k");

  -- -------------------------------------------------------  
  -- HER2K  
   procedure cublasCher2k
     (uplo : char;
      trans : char;
      n : int;
      k : int;
      alpha : cuComplex_h.cuComplex;
      A : access constant cuComplex_h.cuComplex;
      lda : int;
      B : access constant cuComplex_h.cuComplex;
      ldb : int;
      beta : float;
      C : access cuComplex_h.cuComplex;
      ldc : int);  -- /usr/local/cuda-8.0/include/cublas.h:488
   pragma Import (C, cublasCher2k, "cublasCher2k");

   procedure cublasZher2k
     (uplo : char;
      trans : char;
      n : int;
      k : int;
      alpha : cuComplex_h.cuDoubleComplex;
      A : access constant cuComplex_h.cuDoubleComplex;
      lda : int;
      B : access constant cuComplex_h.cuDoubleComplex;
      ldb : int;
      beta : double;
      C : access cuComplex_h.cuDoubleComplex;
      ldc : int);  -- /usr/local/cuda-8.0/include/cublas.h:493
   pragma Import (C, cublasZher2k, "cublasZher2k");

  -------------------------------------------------------------------------- 
  -- SYMM 
   procedure cublasSsymm
     (side : char;
      uplo : char;
      m : int;
      n : int;
      alpha : float;
      A : access float;
      lda : int;
      B : access float;
      ldb : int;
      beta : float;
      C : access float;
      ldc : int);  -- /usr/local/cuda-8.0/include/cublas.h:500
   pragma Import (C, cublasSsymm, "cublasSsymm");

   procedure cublasDsymm
     (side : char;
      uplo : char;
      m : int;
      n : int;
      alpha : double;
      A : access double;
      lda : int;
      B : access double;
      ldb : int;
      beta : double;
      C : access double;
      ldc : int);  -- /usr/local/cuda-8.0/include/cublas.h:503
   pragma Import (C, cublasDsymm, "cublasDsymm");

   procedure cublasCsymm
     (side : char;
      uplo : char;
      m : int;
      n : int;
      alpha : cuComplex_h.cuComplex;
      A : access constant cuComplex_h.cuComplex;
      lda : int;
      B : access constant cuComplex_h.cuComplex;
      ldb : int;
      beta : cuComplex_h.cuComplex;
      C : access cuComplex_h.cuComplex;
      ldc : int);  -- /usr/local/cuda-8.0/include/cublas.h:507
   pragma Import (C, cublasCsymm, "cublasCsymm");

   procedure cublasZsymm
     (side : char;
      uplo : char;
      m : int;
      n : int;
      alpha : cuComplex_h.cuDoubleComplex;
      A : access constant cuComplex_h.cuDoubleComplex;
      lda : int;
      B : access constant cuComplex_h.cuDoubleComplex;
      ldb : int;
      beta : cuComplex_h.cuDoubleComplex;
      C : access cuComplex_h.cuDoubleComplex;
      ldc : int);  -- /usr/local/cuda-8.0/include/cublas.h:511
   pragma Import (C, cublasZsymm, "cublasZsymm");

  -------------------------------------------------------------------------- 
  -- HEMM 
   procedure cublasChemm
     (side : char;
      uplo : char;
      m : int;
      n : int;
      alpha : cuComplex_h.cuComplex;
      A : access constant cuComplex_h.cuComplex;
      lda : int;
      B : access constant cuComplex_h.cuComplex;
      ldb : int;
      beta : cuComplex_h.cuComplex;
      C : access cuComplex_h.cuComplex;
      ldc : int);  -- /usr/local/cuda-8.0/include/cublas.h:516
   pragma Import (C, cublasChemm, "cublasChemm");

   procedure cublasZhemm
     (side : char;
      uplo : char;
      m : int;
      n : int;
      alpha : cuComplex_h.cuDoubleComplex;
      A : access constant cuComplex_h.cuDoubleComplex;
      lda : int;
      B : access constant cuComplex_h.cuDoubleComplex;
      ldb : int;
      beta : cuComplex_h.cuDoubleComplex;
      C : access cuComplex_h.cuDoubleComplex;
      ldc : int);  -- /usr/local/cuda-8.0/include/cublas.h:520
   pragma Import (C, cublasZhemm, "cublasZhemm");

  -------------------------------------------------------------------------- 
  -- TRSM 
   procedure cublasStrsm
     (side : char;
      uplo : char;
      transa : char;
      diag : char;
      m : int;
      n : int;
      alpha : float;
      A : access float;
      lda : int;
      B : access float;
      ldb : int);  -- /usr/local/cuda-8.0/include/cublas.h:527
   pragma Import (C, cublasStrsm, "cublasStrsm");

   procedure cublasDtrsm
     (side : char;
      uplo : char;
      transa : char;
      diag : char;
      m : int;
      n : int;
      alpha : double;
      A : access double;
      lda : int;
      B : access double;
      ldb : int);  -- /usr/local/cuda-8.0/include/cublas.h:531
   pragma Import (C, cublasDtrsm, "cublasDtrsm");

   procedure cublasCtrsm
     (side : char;
      uplo : char;
      transa : char;
      diag : char;
      m : int;
      n : int;
      alpha : cuComplex_h.cuComplex;
      A : access constant cuComplex_h.cuComplex;
      lda : int;
      B : access cuComplex_h.cuComplex;
      ldb : int);  -- /usr/local/cuda-8.0/include/cublas.h:536
   pragma Import (C, cublasCtrsm, "cublasCtrsm");

   procedure cublasZtrsm
     (side : char;
      uplo : char;
      transa : char;
      diag : char;
      m : int;
      n : int;
      alpha : cuComplex_h.cuDoubleComplex;
      A : access constant cuComplex_h.cuDoubleComplex;
      lda : int;
      B : access cuComplex_h.cuDoubleComplex;
      ldb : int);  -- /usr/local/cuda-8.0/include/cublas.h:540
   pragma Import (C, cublasZtrsm, "cublasZtrsm");

  -------------------------------------------------------------------------- 
  -- TRMM 
   procedure cublasStrmm
     (side : char;
      uplo : char;
      transa : char;
      diag : char;
      m : int;
      n : int;
      alpha : float;
      A : access float;
      lda : int;
      B : access float;
      ldb : int);  -- /usr/local/cuda-8.0/include/cublas.h:546
   pragma Import (C, cublasStrmm, "cublasStrmm");

   procedure cublasDtrmm
     (side : char;
      uplo : char;
      transa : char;
      diag : char;
      m : int;
      n : int;
      alpha : double;
      A : access double;
      lda : int;
      B : access double;
      ldb : int);  -- /usr/local/cuda-8.0/include/cublas.h:549
   pragma Import (C, cublasDtrmm, "cublasDtrmm");

   procedure cublasCtrmm
     (side : char;
      uplo : char;
      transa : char;
      diag : char;
      m : int;
      n : int;
      alpha : cuComplex_h.cuComplex;
      A : access constant cuComplex_h.cuComplex;
      lda : int;
      B : access cuComplex_h.cuComplex;
      ldb : int);  -- /usr/local/cuda-8.0/include/cublas.h:553
   pragma Import (C, cublasCtrmm, "cublasCtrmm");

   procedure cublasZtrmm
     (side : char;
      uplo : char;
      transa : char;
      diag : char;
      m : int;
      n : int;
      alpha : cuComplex_h.cuDoubleComplex;
      A : access constant cuComplex_h.cuDoubleComplex;
      lda : int;
      B : access cuComplex_h.cuDoubleComplex;
      ldb : int);  -- /usr/local/cuda-8.0/include/cublas.h:556
   pragma Import (C, cublasZtrmm, "cublasZtrmm");

end cublas_h;

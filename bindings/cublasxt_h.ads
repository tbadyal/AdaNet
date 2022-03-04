pragma Ada_2005;
pragma Style_Checks (Off);

with Interfaces.C; use Interfaces.C;
with System;
with cublas_api_h;
with stddef_h;
limited with cuComplex_h;

package cublasXt_h is

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

  --   cublasXt : Host API, Out of Core and Multi-GPU BLAS Library
  --  

  -- import complex data type  
   --  skipped empty struct cublasXtContext

   type cublasXtHandle_t is new System.Address;  -- /usr/local/cuda-8.0/include/cublasXt.h:67

   function cublasXtCreate (handle : System.Address) return cublas_api_h.cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublasXt.h:69
   pragma Import (C, cublasXtCreate, "cublasXtCreate");

   function cublasXtDestroy (handle : cublasXtHandle_t) return cublas_api_h.cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublasXt.h:70
   pragma Import (C, cublasXtDestroy, "cublasXtDestroy");

   function cublasXtGetNumBoards
     (nbDevices : int;
      deviceId : access int;
      nbBoards : access int) return cublas_api_h.cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublasXt.h:71
   pragma Import (C, cublasXtGetNumBoards, "cublasXtGetNumBoards");

   function cublasXtMaxBoards (nbGpuBoards : access int) return cublas_api_h.cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublasXt.h:72
   pragma Import (C, cublasXtMaxBoards, "cublasXtMaxBoards");

  -- This routine selects the Gpus that the user want to use for CUBLAS-XT  
   function cublasXtDeviceSelect
     (handle : cublasXtHandle_t;
      nbDevices : int;
      deviceId : access int) return cublas_api_h.cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublasXt.h:74
   pragma Import (C, cublasXtDeviceSelect, "cublasXtDeviceSelect");

  -- This routine allows to change the dimension of the tiles ( blockDim x blockDim )  
   function cublasXtSetBlockDim (handle : cublasXtHandle_t; blockDim : int) return cublas_api_h.cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublasXt.h:77
   pragma Import (C, cublasXtSetBlockDim, "cublasXtSetBlockDim");

   function cublasXtGetBlockDim (handle : cublasXtHandle_t; blockDim : access int) return cublas_api_h.cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublasXt.h:78
   pragma Import (C, cublasXtGetBlockDim, "cublasXtGetBlockDim");

   type cublasXtPinnedMemMode_t is 
     (CUBLASXT_PINNING_DISABLED,
      CUBLASXT_PINNING_ENABLED);
   pragma Convention (C, cublasXtPinnedMemMode_t);  -- /usr/local/cuda-8.0/include/cublasXt.h:83

  -- This routine allows to CUBLAS-XT to pin the Host memory if it find out that some of the matrix passed
  --   are not pinned : Pinning/Unpinning the Host memory is still a costly operation
  --   It is better if the user controls the memory on its own (by pinning/unpinning oly when necessary)
  -- 

   function cublasXtGetPinningMemMode (handle : cublasXtHandle_t; mode : access cublasXtPinnedMemMode_t) return cublas_api_h.cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublasXt.h:88
   pragma Import (C, cublasXtGetPinningMemMode, "cublasXtGetPinningMemMode");

   function cublasXtSetPinningMemMode (handle : cublasXtHandle_t; mode : cublasXtPinnedMemMode_t) return cublas_api_h.cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublasXt.h:89
   pragma Import (C, cublasXtSetPinningMemMode, "cublasXtSetPinningMemMode");

  -- This routines is to provide a CPU Blas routines, used for too small sizes or hybrid computation  
   type cublasXtOpType_t is 
     (CUBLASXT_FLOAT,
      CUBLASXT_DOUBLE,
      CUBLASXT_COMPLEX,
      CUBLASXT_DOUBLECOMPLEX);
   pragma Convention (C, cublasXtOpType_t);  -- /usr/local/cuda-8.0/include/cublasXt.h:98

   type cublasXtBlasOp_t is 
     (CUBLASXT_GEMM,
      CUBLASXT_SYRK,
      CUBLASXT_HERK,
      CUBLASXT_SYMM,
      CUBLASXT_HEMM,
      CUBLASXT_TRSM,
      CUBLASXT_SYR2K,
      CUBLASXT_HER2K,
      CUBLASXT_SPMM,
      CUBLASXT_SYRKX,
      CUBLASXT_HERKX,
      CUBLASXT_TRMM,
      CUBLASXT_ROUTINE_MAX);
   pragma Convention (C, cublasXtBlasOp_t);  -- /usr/local/cuda-8.0/include/cublasXt.h:116

  -- Currently only 32-bit integer BLAS routines are supported  
   function cublasXtSetCpuRoutine
     (handle : cublasXtHandle_t;
      blasOp : cublasXtBlasOp_t;
      c_type : cublasXtOpType_t;
      blasFunctor : System.Address) return cublas_api_h.cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublasXt.h:120
   pragma Import (C, cublasXtSetCpuRoutine, "cublasXtSetCpuRoutine");

  -- Specified the percentage of work that should done by the CPU, default is 0 (no work)  
   function cublasXtSetCpuRatio
     (handle : cublasXtHandle_t;
      blasOp : cublasXtBlasOp_t;
      c_type : cublasXtOpType_t;
      ratio : float) return cublas_api_h.cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublasXt.h:123
   pragma Import (C, cublasXtSetCpuRatio, "cublasXtSetCpuRatio");

  -- GEMM  
   function cublasXtSgemm
     (handle : cublasXtHandle_t;
      transa : cublas_api_h.cublasOperation_t;
      transb : cublas_api_h.cublasOperation_t;
      m : stddef_h.size_t;
      n : stddef_h.size_t;
      k : stddef_h.size_t;
      alpha : access float;
      A : access float;
      lda : stddef_h.size_t;
      B : access float;
      ldb : stddef_h.size_t;
      beta : access float;
      C : access float;
      ldc : stddef_h.size_t) return cublas_api_h.cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublasXt.h:127
   pragma Import (C, cublasXtSgemm, "cublasXtSgemm");

   function cublasXtDgemm
     (handle : cublasXtHandle_t;
      transa : cublas_api_h.cublasOperation_t;
      transb : cublas_api_h.cublasOperation_t;
      m : stddef_h.size_t;
      n : stddef_h.size_t;
      k : stddef_h.size_t;
      alpha : access double;
      A : access double;
      lda : stddef_h.size_t;
      B : access double;
      ldb : stddef_h.size_t;
      beta : access double;
      C : access double;
      ldc : stddef_h.size_t) return cublas_api_h.cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublasXt.h:142
   pragma Import (C, cublasXtDgemm, "cublasXtDgemm");

   function cublasXtCgemm
     (handle : cublasXtHandle_t;
      transa : cublas_api_h.cublasOperation_t;
      transb : cublas_api_h.cublasOperation_t;
      m : stddef_h.size_t;
      n : stddef_h.size_t;
      k : stddef_h.size_t;
      alpha : access constant cuComplex_h.cuComplex;
      A : access constant cuComplex_h.cuComplex;
      lda : stddef_h.size_t;
      B : access constant cuComplex_h.cuComplex;
      ldb : stddef_h.size_t;
      beta : access constant cuComplex_h.cuComplex;
      C : access cuComplex_h.cuComplex;
      ldc : stddef_h.size_t) return cublas_api_h.cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublasXt.h:157
   pragma Import (C, cublasXtCgemm, "cublasXtCgemm");

   function cublasXtZgemm
     (handle : cublasXtHandle_t;
      transa : cublas_api_h.cublasOperation_t;
      transb : cublas_api_h.cublasOperation_t;
      m : stddef_h.size_t;
      n : stddef_h.size_t;
      k : stddef_h.size_t;
      alpha : access constant cuComplex_h.cuDoubleComplex;
      A : access constant cuComplex_h.cuDoubleComplex;
      lda : stddef_h.size_t;
      B : access constant cuComplex_h.cuDoubleComplex;
      ldb : stddef_h.size_t;
      beta : access constant cuComplex_h.cuDoubleComplex;
      C : access cuComplex_h.cuDoubleComplex;
      ldc : stddef_h.size_t) return cublas_api_h.cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublasXt.h:172
   pragma Import (C, cublasXtZgemm, "cublasXtZgemm");

  -- -------------------------------------------------------  
  -- SYRK  
   function cublasXtSsyrk
     (handle : cublasXtHandle_t;
      uplo : cublas_api_h.cublasFillMode_t;
      trans : cublas_api_h.cublasOperation_t;
      n : stddef_h.size_t;
      k : stddef_h.size_t;
      alpha : access float;
      A : access float;
      lda : stddef_h.size_t;
      beta : access float;
      C : access float;
      ldc : stddef_h.size_t) return cublas_api_h.cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublasXt.h:188
   pragma Import (C, cublasXtSsyrk, "cublasXtSsyrk");

   function cublasXtDsyrk
     (handle : cublasXtHandle_t;
      uplo : cublas_api_h.cublasFillMode_t;
      trans : cublas_api_h.cublasOperation_t;
      n : stddef_h.size_t;
      k : stddef_h.size_t;
      alpha : access double;
      A : access double;
      lda : stddef_h.size_t;
      beta : access double;
      C : access double;
      ldc : stddef_h.size_t) return cublas_api_h.cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublasXt.h:200
   pragma Import (C, cublasXtDsyrk, "cublasXtDsyrk");

   function cublasXtCsyrk
     (handle : cublasXtHandle_t;
      uplo : cublas_api_h.cublasFillMode_t;
      trans : cublas_api_h.cublasOperation_t;
      n : stddef_h.size_t;
      k : stddef_h.size_t;
      alpha : access constant cuComplex_h.cuComplex;
      A : access constant cuComplex_h.cuComplex;
      lda : stddef_h.size_t;
      beta : access constant cuComplex_h.cuComplex;
      C : access cuComplex_h.cuComplex;
      ldc : stddef_h.size_t) return cublas_api_h.cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublasXt.h:212
   pragma Import (C, cublasXtCsyrk, "cublasXtCsyrk");

   function cublasXtZsyrk
     (handle : cublasXtHandle_t;
      uplo : cublas_api_h.cublasFillMode_t;
      trans : cublas_api_h.cublasOperation_t;
      n : stddef_h.size_t;
      k : stddef_h.size_t;
      alpha : access constant cuComplex_h.cuDoubleComplex;
      A : access constant cuComplex_h.cuDoubleComplex;
      lda : stddef_h.size_t;
      beta : access constant cuComplex_h.cuDoubleComplex;
      C : access cuComplex_h.cuDoubleComplex;
      ldc : stddef_h.size_t) return cublas_api_h.cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublasXt.h:224
   pragma Import (C, cublasXtZsyrk, "cublasXtZsyrk");

  -- --------------------------------------------------------------------  
  -- HERK  
   function cublasXtCherk
     (handle : cublasXtHandle_t;
      uplo : cublas_api_h.cublasFillMode_t;
      trans : cublas_api_h.cublasOperation_t;
      n : stddef_h.size_t;
      k : stddef_h.size_t;
      alpha : access float;
      A : access constant cuComplex_h.cuComplex;
      lda : stddef_h.size_t;
      beta : access float;
      C : access cuComplex_h.cuComplex;
      ldc : stddef_h.size_t) return cublas_api_h.cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublasXt.h:237
   pragma Import (C, cublasXtCherk, "cublasXtCherk");

   function cublasXtZherk
     (handle : cublasXtHandle_t;
      uplo : cublas_api_h.cublasFillMode_t;
      trans : cublas_api_h.cublasOperation_t;
      n : stddef_h.size_t;
      k : stddef_h.size_t;
      alpha : access double;
      A : access constant cuComplex_h.cuDoubleComplex;
      lda : stddef_h.size_t;
      beta : access double;
      C : access cuComplex_h.cuDoubleComplex;
      ldc : stddef_h.size_t) return cublas_api_h.cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublasXt.h:249
   pragma Import (C, cublasXtZherk, "cublasXtZherk");

  -- --------------------------------------------------------------------  
  -- SYR2K  
   function cublasXtSsyr2k
     (handle : cublasXtHandle_t;
      uplo : cublas_api_h.cublasFillMode_t;
      trans : cublas_api_h.cublasOperation_t;
      n : stddef_h.size_t;
      k : stddef_h.size_t;
      alpha : access float;
      A : access float;
      lda : stddef_h.size_t;
      B : access float;
      ldb : stddef_h.size_t;
      beta : access float;
      C : access float;
      ldc : stddef_h.size_t) return cublas_api_h.cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublasXt.h:262
   pragma Import (C, cublasXtSsyr2k, "cublasXtSsyr2k");

   function cublasXtDsyr2k
     (handle : cublasXtHandle_t;
      uplo : cublas_api_h.cublasFillMode_t;
      trans : cublas_api_h.cublasOperation_t;
      n : stddef_h.size_t;
      k : stddef_h.size_t;
      alpha : access double;
      A : access double;
      lda : stddef_h.size_t;
      B : access double;
      ldb : stddef_h.size_t;
      beta : access double;
      C : access double;
      ldc : stddef_h.size_t) return cublas_api_h.cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublasXt.h:276
   pragma Import (C, cublasXtDsyr2k, "cublasXtDsyr2k");

   function cublasXtCsyr2k
     (handle : cublasXtHandle_t;
      uplo : cublas_api_h.cublasFillMode_t;
      trans : cublas_api_h.cublasOperation_t;
      n : stddef_h.size_t;
      k : stddef_h.size_t;
      alpha : access constant cuComplex_h.cuComplex;
      A : access constant cuComplex_h.cuComplex;
      lda : stddef_h.size_t;
      B : access constant cuComplex_h.cuComplex;
      ldb : stddef_h.size_t;
      beta : access constant cuComplex_h.cuComplex;
      C : access cuComplex_h.cuComplex;
      ldc : stddef_h.size_t) return cublas_api_h.cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublasXt.h:290
   pragma Import (C, cublasXtCsyr2k, "cublasXtCsyr2k");

   function cublasXtZsyr2k
     (handle : cublasXtHandle_t;
      uplo : cublas_api_h.cublasFillMode_t;
      trans : cublas_api_h.cublasOperation_t;
      n : stddef_h.size_t;
      k : stddef_h.size_t;
      alpha : access constant cuComplex_h.cuDoubleComplex;
      A : access constant cuComplex_h.cuDoubleComplex;
      lda : stddef_h.size_t;
      B : access constant cuComplex_h.cuDoubleComplex;
      ldb : stddef_h.size_t;
      beta : access constant cuComplex_h.cuDoubleComplex;
      C : access cuComplex_h.cuDoubleComplex;
      ldc : stddef_h.size_t) return cublas_api_h.cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublasXt.h:304
   pragma Import (C, cublasXtZsyr2k, "cublasXtZsyr2k");

  -- --------------------------------------------------------------------  
  -- HERKX : variant extension of HERK  
   function cublasXtCherkx
     (handle : cublasXtHandle_t;
      uplo : cublas_api_h.cublasFillMode_t;
      trans : cublas_api_h.cublasOperation_t;
      n : stddef_h.size_t;
      k : stddef_h.size_t;
      alpha : access constant cuComplex_h.cuComplex;
      A : access constant cuComplex_h.cuComplex;
      lda : stddef_h.size_t;
      B : access constant cuComplex_h.cuComplex;
      ldb : stddef_h.size_t;
      beta : access float;
      C : access cuComplex_h.cuComplex;
      ldc : stddef_h.size_t) return cublas_api_h.cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublasXt.h:319
   pragma Import (C, cublasXtCherkx, "cublasXtCherkx");

   function cublasXtZherkx
     (handle : cublasXtHandle_t;
      uplo : cublas_api_h.cublasFillMode_t;
      trans : cublas_api_h.cublasOperation_t;
      n : stddef_h.size_t;
      k : stddef_h.size_t;
      alpha : access constant cuComplex_h.cuDoubleComplex;
      A : access constant cuComplex_h.cuDoubleComplex;
      lda : stddef_h.size_t;
      B : access constant cuComplex_h.cuDoubleComplex;
      ldb : stddef_h.size_t;
      beta : access double;
      C : access cuComplex_h.cuDoubleComplex;
      ldc : stddef_h.size_t) return cublas_api_h.cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublasXt.h:333
   pragma Import (C, cublasXtZherkx, "cublasXtZherkx");

  -- --------------------------------------------------------------------  
  -- TRSM  
   function cublasXtStrsm
     (handle : cublasXtHandle_t;
      side : cublas_api_h.cublasSideMode_t;
      uplo : cublas_api_h.cublasFillMode_t;
      trans : cublas_api_h.cublasOperation_t;
      diag : cublas_api_h.cublasDiagType_t;
      m : stddef_h.size_t;
      n : stddef_h.size_t;
      alpha : access float;
      A : access float;
      lda : stddef_h.size_t;
      B : access float;
      ldb : stddef_h.size_t) return cublas_api_h.cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublasXt.h:349
   pragma Import (C, cublasXtStrsm, "cublasXtStrsm");

   function cublasXtDtrsm
     (handle : cublasXtHandle_t;
      side : cublas_api_h.cublasSideMode_t;
      uplo : cublas_api_h.cublasFillMode_t;
      trans : cublas_api_h.cublasOperation_t;
      diag : cublas_api_h.cublasDiagType_t;
      m : stddef_h.size_t;
      n : stddef_h.size_t;
      alpha : access double;
      A : access double;
      lda : stddef_h.size_t;
      B : access double;
      ldb : stddef_h.size_t) return cublas_api_h.cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublasXt.h:363
   pragma Import (C, cublasXtDtrsm, "cublasXtDtrsm");

   function cublasXtCtrsm
     (handle : cublasXtHandle_t;
      side : cublas_api_h.cublasSideMode_t;
      uplo : cublas_api_h.cublasFillMode_t;
      trans : cublas_api_h.cublasOperation_t;
      diag : cublas_api_h.cublasDiagType_t;
      m : stddef_h.size_t;
      n : stddef_h.size_t;
      alpha : access constant cuComplex_h.cuComplex;
      A : access constant cuComplex_h.cuComplex;
      lda : stddef_h.size_t;
      B : access cuComplex_h.cuComplex;
      ldb : stddef_h.size_t) return cublas_api_h.cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublasXt.h:376
   pragma Import (C, cublasXtCtrsm, "cublasXtCtrsm");

   function cublasXtZtrsm
     (handle : cublasXtHandle_t;
      side : cublas_api_h.cublasSideMode_t;
      uplo : cublas_api_h.cublasFillMode_t;
      trans : cublas_api_h.cublasOperation_t;
      diag : cublas_api_h.cublasDiagType_t;
      m : stddef_h.size_t;
      n : stddef_h.size_t;
      alpha : access constant cuComplex_h.cuDoubleComplex;
      A : access constant cuComplex_h.cuDoubleComplex;
      lda : stddef_h.size_t;
      B : access cuComplex_h.cuDoubleComplex;
      ldb : stddef_h.size_t) return cublas_api_h.cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublasXt.h:389
   pragma Import (C, cublasXtZtrsm, "cublasXtZtrsm");

  -- --------------------------------------------------------------------  
  -- SYMM : Symmetric Multiply Matrix 
   function cublasXtSsymm
     (handle : cublasXtHandle_t;
      side : cublas_api_h.cublasSideMode_t;
      uplo : cublas_api_h.cublasFillMode_t;
      m : stddef_h.size_t;
      n : stddef_h.size_t;
      alpha : access float;
      A : access float;
      lda : stddef_h.size_t;
      B : access float;
      ldb : stddef_h.size_t;
      beta : access float;
      C : access float;
      ldc : stddef_h.size_t) return cublas_api_h.cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublasXt.h:403
   pragma Import (C, cublasXtSsymm, "cublasXtSsymm");

   function cublasXtDsymm
     (handle : cublasXtHandle_t;
      side : cublas_api_h.cublasSideMode_t;
      uplo : cublas_api_h.cublasFillMode_t;
      m : stddef_h.size_t;
      n : stddef_h.size_t;
      alpha : access double;
      A : access double;
      lda : stddef_h.size_t;
      B : access double;
      ldb : stddef_h.size_t;
      beta : access double;
      C : access double;
      ldc : stddef_h.size_t) return cublas_api_h.cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublasXt.h:417
   pragma Import (C, cublasXtDsymm, "cublasXtDsymm");

   function cublasXtCsymm
     (handle : cublasXtHandle_t;
      side : cublas_api_h.cublasSideMode_t;
      uplo : cublas_api_h.cublasFillMode_t;
      m : stddef_h.size_t;
      n : stddef_h.size_t;
      alpha : access constant cuComplex_h.cuComplex;
      A : access constant cuComplex_h.cuComplex;
      lda : stddef_h.size_t;
      B : access constant cuComplex_h.cuComplex;
      ldb : stddef_h.size_t;
      beta : access constant cuComplex_h.cuComplex;
      C : access cuComplex_h.cuComplex;
      ldc : stddef_h.size_t) return cublas_api_h.cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublasXt.h:431
   pragma Import (C, cublasXtCsymm, "cublasXtCsymm");

   function cublasXtZsymm
     (handle : cublasXtHandle_t;
      side : cublas_api_h.cublasSideMode_t;
      uplo : cublas_api_h.cublasFillMode_t;
      m : stddef_h.size_t;
      n : stddef_h.size_t;
      alpha : access constant cuComplex_h.cuDoubleComplex;
      A : access constant cuComplex_h.cuDoubleComplex;
      lda : stddef_h.size_t;
      B : access constant cuComplex_h.cuDoubleComplex;
      ldb : stddef_h.size_t;
      beta : access constant cuComplex_h.cuDoubleComplex;
      C : access cuComplex_h.cuDoubleComplex;
      ldc : stddef_h.size_t) return cublas_api_h.cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublasXt.h:445
   pragma Import (C, cublasXtZsymm, "cublasXtZsymm");

  -- --------------------------------------------------------------------  
  -- HEMM : Hermitian Matrix Multiply  
   function cublasXtChemm
     (handle : cublasXtHandle_t;
      side : cublas_api_h.cublasSideMode_t;
      uplo : cublas_api_h.cublasFillMode_t;
      m : stddef_h.size_t;
      n : stddef_h.size_t;
      alpha : access constant cuComplex_h.cuComplex;
      A : access constant cuComplex_h.cuComplex;
      lda : stddef_h.size_t;
      B : access constant cuComplex_h.cuComplex;
      ldb : stddef_h.size_t;
      beta : access constant cuComplex_h.cuComplex;
      C : access cuComplex_h.cuComplex;
      ldc : stddef_h.size_t) return cublas_api_h.cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublasXt.h:460
   pragma Import (C, cublasXtChemm, "cublasXtChemm");

   function cublasXtZhemm
     (handle : cublasXtHandle_t;
      side : cublas_api_h.cublasSideMode_t;
      uplo : cublas_api_h.cublasFillMode_t;
      m : stddef_h.size_t;
      n : stddef_h.size_t;
      alpha : access constant cuComplex_h.cuDoubleComplex;
      A : access constant cuComplex_h.cuDoubleComplex;
      lda : stddef_h.size_t;
      B : access constant cuComplex_h.cuDoubleComplex;
      ldb : stddef_h.size_t;
      beta : access constant cuComplex_h.cuDoubleComplex;
      C : access cuComplex_h.cuDoubleComplex;
      ldc : stddef_h.size_t) return cublas_api_h.cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublasXt.h:474
   pragma Import (C, cublasXtZhemm, "cublasXtZhemm");

  -- --------------------------------------------------------------------  
  -- SYRKX : variant extension of SYRK   
   function cublasXtSsyrkx
     (handle : cublasXtHandle_t;
      uplo : cublas_api_h.cublasFillMode_t;
      trans : cublas_api_h.cublasOperation_t;
      n : stddef_h.size_t;
      k : stddef_h.size_t;
      alpha : access float;
      A : access float;
      lda : stddef_h.size_t;
      B : access float;
      ldb : stddef_h.size_t;
      beta : access float;
      C : access float;
      ldc : stddef_h.size_t) return cublas_api_h.cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublasXt.h:490
   pragma Import (C, cublasXtSsyrkx, "cublasXtSsyrkx");

   function cublasXtDsyrkx
     (handle : cublasXtHandle_t;
      uplo : cublas_api_h.cublasFillMode_t;
      trans : cublas_api_h.cublasOperation_t;
      n : stddef_h.size_t;
      k : stddef_h.size_t;
      alpha : access double;
      A : access double;
      lda : stddef_h.size_t;
      B : access double;
      ldb : stddef_h.size_t;
      beta : access double;
      C : access double;
      ldc : stddef_h.size_t) return cublas_api_h.cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublasXt.h:504
   pragma Import (C, cublasXtDsyrkx, "cublasXtDsyrkx");

   function cublasXtCsyrkx
     (handle : cublasXtHandle_t;
      uplo : cublas_api_h.cublasFillMode_t;
      trans : cublas_api_h.cublasOperation_t;
      n : stddef_h.size_t;
      k : stddef_h.size_t;
      alpha : access constant cuComplex_h.cuComplex;
      A : access constant cuComplex_h.cuComplex;
      lda : stddef_h.size_t;
      B : access constant cuComplex_h.cuComplex;
      ldb : stddef_h.size_t;
      beta : access constant cuComplex_h.cuComplex;
      C : access cuComplex_h.cuComplex;
      ldc : stddef_h.size_t) return cublas_api_h.cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublasXt.h:518
   pragma Import (C, cublasXtCsyrkx, "cublasXtCsyrkx");

   function cublasXtZsyrkx
     (handle : cublasXtHandle_t;
      uplo : cublas_api_h.cublasFillMode_t;
      trans : cublas_api_h.cublasOperation_t;
      n : stddef_h.size_t;
      k : stddef_h.size_t;
      alpha : access constant cuComplex_h.cuDoubleComplex;
      A : access constant cuComplex_h.cuDoubleComplex;
      lda : stddef_h.size_t;
      B : access constant cuComplex_h.cuDoubleComplex;
      ldb : stddef_h.size_t;
      beta : access constant cuComplex_h.cuDoubleComplex;
      C : access cuComplex_h.cuDoubleComplex;
      ldc : stddef_h.size_t) return cublas_api_h.cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublasXt.h:532
   pragma Import (C, cublasXtZsyrkx, "cublasXtZsyrkx");

  -- --------------------------------------------------------------------  
  -- HER2K : variant extension of HERK   
   function cublasXtCher2k
     (handle : cublasXtHandle_t;
      uplo : cublas_api_h.cublasFillMode_t;
      trans : cublas_api_h.cublasOperation_t;
      n : stddef_h.size_t;
      k : stddef_h.size_t;
      alpha : access constant cuComplex_h.cuComplex;
      A : access constant cuComplex_h.cuComplex;
      lda : stddef_h.size_t;
      B : access constant cuComplex_h.cuComplex;
      ldb : stddef_h.size_t;
      beta : access float;
      C : access cuComplex_h.cuComplex;
      ldc : stddef_h.size_t) return cublas_api_h.cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublasXt.h:547
   pragma Import (C, cublasXtCher2k, "cublasXtCher2k");

   function cublasXtZher2k
     (handle : cublasXtHandle_t;
      uplo : cublas_api_h.cublasFillMode_t;
      trans : cublas_api_h.cublasOperation_t;
      n : stddef_h.size_t;
      k : stddef_h.size_t;
      alpha : access constant cuComplex_h.cuDoubleComplex;
      A : access constant cuComplex_h.cuDoubleComplex;
      lda : stddef_h.size_t;
      B : access constant cuComplex_h.cuDoubleComplex;
      ldb : stddef_h.size_t;
      beta : access double;
      C : access cuComplex_h.cuDoubleComplex;
      ldc : stddef_h.size_t) return cublas_api_h.cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublasXt.h:561
   pragma Import (C, cublasXtZher2k, "cublasXtZher2k");

  -- --------------------------------------------------------------------  
  -- SPMM : Symmetric Packed Multiply Matrix 
   function cublasXtSspmm
     (handle : cublasXtHandle_t;
      side : cublas_api_h.cublasSideMode_t;
      uplo : cublas_api_h.cublasFillMode_t;
      m : stddef_h.size_t;
      n : stddef_h.size_t;
      alpha : access float;
      AP : access float;
      B : access float;
      ldb : stddef_h.size_t;
      beta : access float;
      C : access float;
      ldc : stddef_h.size_t) return cublas_api_h.cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublasXt.h:578
   pragma Import (C, cublasXtSspmm, "cublasXtSspmm");

   function cublasXtDspmm
     (handle : cublasXtHandle_t;
      side : cublas_api_h.cublasSideMode_t;
      uplo : cublas_api_h.cublasFillMode_t;
      m : stddef_h.size_t;
      n : stddef_h.size_t;
      alpha : access double;
      AP : access double;
      B : access double;
      ldb : stddef_h.size_t;
      beta : access double;
      C : access double;
      ldc : stddef_h.size_t) return cublas_api_h.cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublasXt.h:591
   pragma Import (C, cublasXtDspmm, "cublasXtDspmm");

   function cublasXtCspmm
     (handle : cublasXtHandle_t;
      side : cublas_api_h.cublasSideMode_t;
      uplo : cublas_api_h.cublasFillMode_t;
      m : stddef_h.size_t;
      n : stddef_h.size_t;
      alpha : access constant cuComplex_h.cuComplex;
      AP : access constant cuComplex_h.cuComplex;
      B : access constant cuComplex_h.cuComplex;
      ldb : stddef_h.size_t;
      beta : access constant cuComplex_h.cuComplex;
      C : access cuComplex_h.cuComplex;
      ldc : stddef_h.size_t) return cublas_api_h.cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublasXt.h:604
   pragma Import (C, cublasXtCspmm, "cublasXtCspmm");

   function cublasXtZspmm
     (handle : cublasXtHandle_t;
      side : cublas_api_h.cublasSideMode_t;
      uplo : cublas_api_h.cublasFillMode_t;
      m : stddef_h.size_t;
      n : stddef_h.size_t;
      alpha : access constant cuComplex_h.cuDoubleComplex;
      AP : access constant cuComplex_h.cuDoubleComplex;
      B : access constant cuComplex_h.cuDoubleComplex;
      ldb : stddef_h.size_t;
      beta : access constant cuComplex_h.cuDoubleComplex;
      C : access cuComplex_h.cuDoubleComplex;
      ldc : stddef_h.size_t) return cublas_api_h.cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublasXt.h:617
   pragma Import (C, cublasXtZspmm, "cublasXtZspmm");

  -- --------------------------------------------------------------------  
  -- TRMM  
   function cublasXtStrmm
     (handle : cublasXtHandle_t;
      side : cublas_api_h.cublasSideMode_t;
      uplo : cublas_api_h.cublasFillMode_t;
      trans : cublas_api_h.cublasOperation_t;
      diag : cublas_api_h.cublasDiagType_t;
      m : stddef_h.size_t;
      n : stddef_h.size_t;
      alpha : access float;
      A : access float;
      lda : stddef_h.size_t;
      B : access float;
      ldb : stddef_h.size_t;
      C : access float;
      ldc : stddef_h.size_t) return cublas_api_h.cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublasXt.h:632
   pragma Import (C, cublasXtStrmm, "cublasXtStrmm");

   function cublasXtDtrmm
     (handle : cublasXtHandle_t;
      side : cublas_api_h.cublasSideMode_t;
      uplo : cublas_api_h.cublasFillMode_t;
      trans : cublas_api_h.cublasOperation_t;
      diag : cublas_api_h.cublasDiagType_t;
      m : stddef_h.size_t;
      n : stddef_h.size_t;
      alpha : access double;
      A : access double;
      lda : stddef_h.size_t;
      B : access double;
      ldb : stddef_h.size_t;
      C : access double;
      ldc : stddef_h.size_t) return cublas_api_h.cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublasXt.h:647
   pragma Import (C, cublasXtDtrmm, "cublasXtDtrmm");

   function cublasXtCtrmm
     (handle : cublasXtHandle_t;
      side : cublas_api_h.cublasSideMode_t;
      uplo : cublas_api_h.cublasFillMode_t;
      trans : cublas_api_h.cublasOperation_t;
      diag : cublas_api_h.cublasDiagType_t;
      m : stddef_h.size_t;
      n : stddef_h.size_t;
      alpha : access constant cuComplex_h.cuComplex;
      A : access constant cuComplex_h.cuComplex;
      lda : stddef_h.size_t;
      B : access constant cuComplex_h.cuComplex;
      ldb : stddef_h.size_t;
      C : access cuComplex_h.cuComplex;
      ldc : stddef_h.size_t) return cublas_api_h.cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublasXt.h:662
   pragma Import (C, cublasXtCtrmm, "cublasXtCtrmm");

   function cublasXtZtrmm
     (handle : cublasXtHandle_t;
      side : cublas_api_h.cublasSideMode_t;
      uplo : cublas_api_h.cublasFillMode_t;
      trans : cublas_api_h.cublasOperation_t;
      diag : cublas_api_h.cublasDiagType_t;
      m : stddef_h.size_t;
      n : stddef_h.size_t;
      alpha : access constant cuComplex_h.cuDoubleComplex;
      A : access constant cuComplex_h.cuDoubleComplex;
      lda : stddef_h.size_t;
      B : access constant cuComplex_h.cuDoubleComplex;
      ldb : stddef_h.size_t;
      C : access cuComplex_h.cuDoubleComplex;
      ldc : stddef_h.size_t) return cublas_api_h.cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublasXt.h:677
   pragma Import (C, cublasXtZtrmm, "cublasXtZtrmm");

end cublasXt_h;

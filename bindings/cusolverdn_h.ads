pragma Ada_2005;
pragma Style_Checks (Off);

with Interfaces.C; use Interfaces.C;
with System;
with cusolver_common_h;
with driver_types_h;
with cublas_api_h;
limited with cuComplex_h;

package cusolverDn_h is

  -- * Copyright 2014 NVIDIA Corporation.  All rights reserved.
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

  --   cuSolverDN : Dense Linear Algebra Library
  --  

  -- import complex data type  
   --  skipped empty struct cusolverDnContext

   type cusolverDnHandle_t is new System.Address;  -- /usr/local/cuda-8.0/include/cusolverDn.h:67

   function cusolverDnCreate (handle : System.Address) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:69
   pragma Import (C, cusolverDnCreate, "cusolverDnCreate");

   function cusolverDnDestroy (handle : cusolverDnHandle_t) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:70
   pragma Import (C, cusolverDnDestroy, "cusolverDnDestroy");

   function cusolverDnSetStream (handle : cusolverDnHandle_t; streamId : driver_types_h.cudaStream_t) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:71
   pragma Import (C, cusolverDnSetStream, "cusolverDnSetStream");

   function cusolverDnGetStream (handle : cusolverDnHandle_t; streamId : System.Address) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:72
   pragma Import (C, cusolverDnGetStream, "cusolverDnGetStream");

  -- Cholesky factorization and its solver  
   function cusolverDnSpotrf_bufferSize
     (handle : cusolverDnHandle_t;
      uplo : cublas_api_h.cublasFillMode_t;
      n : int;
      A : access float;
      lda : int;
      Lwork : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:75
   pragma Import (C, cusolverDnSpotrf_bufferSize, "cusolverDnSpotrf_bufferSize");

   function cusolverDnDpotrf_bufferSize
     (handle : cusolverDnHandle_t;
      uplo : cublas_api_h.cublasFillMode_t;
      n : int;
      A : access double;
      lda : int;
      Lwork : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:83
   pragma Import (C, cusolverDnDpotrf_bufferSize, "cusolverDnDpotrf_bufferSize");

   function cusolverDnCpotrf_bufferSize
     (handle : cusolverDnHandle_t;
      uplo : cublas_api_h.cublasFillMode_t;
      n : int;
      A : access cuComplex_h.cuComplex;
      lda : int;
      Lwork : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:91
   pragma Import (C, cusolverDnCpotrf_bufferSize, "cusolverDnCpotrf_bufferSize");

   function cusolverDnZpotrf_bufferSize
     (handle : cusolverDnHandle_t;
      uplo : cublas_api_h.cublasFillMode_t;
      n : int;
      A : access cuComplex_h.cuDoubleComplex;
      lda : int;
      Lwork : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:99
   pragma Import (C, cusolverDnZpotrf_bufferSize, "cusolverDnZpotrf_bufferSize");

   function cusolverDnSpotrf
     (handle : cusolverDnHandle_t;
      uplo : cublas_api_h.cublasFillMode_t;
      n : int;
      A : access float;
      lda : int;
      Workspace : access float;
      Lwork : int;
      devInfo : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:107
   pragma Import (C, cusolverDnSpotrf, "cusolverDnSpotrf");

   function cusolverDnDpotrf
     (handle : cusolverDnHandle_t;
      uplo : cublas_api_h.cublasFillMode_t;
      n : int;
      A : access double;
      lda : int;
      Workspace : access double;
      Lwork : int;
      devInfo : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:117
   pragma Import (C, cusolverDnDpotrf, "cusolverDnDpotrf");

   function cusolverDnCpotrf
     (handle : cusolverDnHandle_t;
      uplo : cublas_api_h.cublasFillMode_t;
      n : int;
      A : access cuComplex_h.cuComplex;
      lda : int;
      Workspace : access cuComplex_h.cuComplex;
      Lwork : int;
      devInfo : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:127
   pragma Import (C, cusolverDnCpotrf, "cusolverDnCpotrf");

   function cusolverDnZpotrf
     (handle : cusolverDnHandle_t;
      uplo : cublas_api_h.cublasFillMode_t;
      n : int;
      A : access cuComplex_h.cuDoubleComplex;
      lda : int;
      Workspace : access cuComplex_h.cuDoubleComplex;
      Lwork : int;
      devInfo : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:137
   pragma Import (C, cusolverDnZpotrf, "cusolverDnZpotrf");

   function cusolverDnSpotrs
     (handle : cusolverDnHandle_t;
      uplo : cublas_api_h.cublasFillMode_t;
      n : int;
      nrhs : int;
      A : access float;
      lda : int;
      B : access float;
      ldb : int;
      devInfo : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:148
   pragma Import (C, cusolverDnSpotrs, "cusolverDnSpotrs");

   function cusolverDnDpotrs
     (handle : cusolverDnHandle_t;
      uplo : cublas_api_h.cublasFillMode_t;
      n : int;
      nrhs : int;
      A : access double;
      lda : int;
      B : access double;
      ldb : int;
      devInfo : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:159
   pragma Import (C, cusolverDnDpotrs, "cusolverDnDpotrs");

   function cusolverDnCpotrs
     (handle : cusolverDnHandle_t;
      uplo : cublas_api_h.cublasFillMode_t;
      n : int;
      nrhs : int;
      A : access constant cuComplex_h.cuComplex;
      lda : int;
      B : access cuComplex_h.cuComplex;
      ldb : int;
      devInfo : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:170
   pragma Import (C, cusolverDnCpotrs, "cusolverDnCpotrs");

   function cusolverDnZpotrs
     (handle : cusolverDnHandle_t;
      uplo : cublas_api_h.cublasFillMode_t;
      n : int;
      nrhs : int;
      A : access constant cuComplex_h.cuDoubleComplex;
      lda : int;
      B : access cuComplex_h.cuDoubleComplex;
      ldb : int;
      devInfo : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:181
   pragma Import (C, cusolverDnZpotrs, "cusolverDnZpotrs");

  -- LU Factorization  
   function cusolverDnSgetrf_bufferSize
     (handle : cusolverDnHandle_t;
      m : int;
      n : int;
      A : access float;
      lda : int;
      Lwork : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:194
   pragma Import (C, cusolverDnSgetrf_bufferSize, "cusolverDnSgetrf_bufferSize");

   function cusolverDnDgetrf_bufferSize
     (handle : cusolverDnHandle_t;
      m : int;
      n : int;
      A : access double;
      lda : int;
      Lwork : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:202
   pragma Import (C, cusolverDnDgetrf_bufferSize, "cusolverDnDgetrf_bufferSize");

   function cusolverDnCgetrf_bufferSize
     (handle : cusolverDnHandle_t;
      m : int;
      n : int;
      A : access cuComplex_h.cuComplex;
      lda : int;
      Lwork : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:210
   pragma Import (C, cusolverDnCgetrf_bufferSize, "cusolverDnCgetrf_bufferSize");

   function cusolverDnZgetrf_bufferSize
     (handle : cusolverDnHandle_t;
      m : int;
      n : int;
      A : access cuComplex_h.cuDoubleComplex;
      lda : int;
      Lwork : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:218
   pragma Import (C, cusolverDnZgetrf_bufferSize, "cusolverDnZgetrf_bufferSize");

   function cusolverDnSgetrf
     (handle : cusolverDnHandle_t;
      m : int;
      n : int;
      A : access float;
      lda : int;
      Workspace : access float;
      devIpiv : access int;
      devInfo : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:227
   pragma Import (C, cusolverDnSgetrf, "cusolverDnSgetrf");

   function cusolverDnDgetrf
     (handle : cusolverDnHandle_t;
      m : int;
      n : int;
      A : access double;
      lda : int;
      Workspace : access double;
      devIpiv : access int;
      devInfo : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:237
   pragma Import (C, cusolverDnDgetrf, "cusolverDnDgetrf");

   function cusolverDnCgetrf
     (handle : cusolverDnHandle_t;
      m : int;
      n : int;
      A : access cuComplex_h.cuComplex;
      lda : int;
      Workspace : access cuComplex_h.cuComplex;
      devIpiv : access int;
      devInfo : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:247
   pragma Import (C, cusolverDnCgetrf, "cusolverDnCgetrf");

   function cusolverDnZgetrf
     (handle : cusolverDnHandle_t;
      m : int;
      n : int;
      A : access cuComplex_h.cuDoubleComplex;
      lda : int;
      Workspace : access cuComplex_h.cuDoubleComplex;
      devIpiv : access int;
      devInfo : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:257
   pragma Import (C, cusolverDnZgetrf, "cusolverDnZgetrf");

  -- Row pivoting  
   function cusolverDnSlaswp
     (handle : cusolverDnHandle_t;
      n : int;
      A : access float;
      lda : int;
      k1 : int;
      k2 : int;
      devIpiv : access int;
      incx : int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:268
   pragma Import (C, cusolverDnSlaswp, "cusolverDnSlaswp");

   function cusolverDnDlaswp
     (handle : cusolverDnHandle_t;
      n : int;
      A : access double;
      lda : int;
      k1 : int;
      k2 : int;
      devIpiv : access int;
      incx : int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:278
   pragma Import (C, cusolverDnDlaswp, "cusolverDnDlaswp");

   function cusolverDnClaswp
     (handle : cusolverDnHandle_t;
      n : int;
      A : access cuComplex_h.cuComplex;
      lda : int;
      k1 : int;
      k2 : int;
      devIpiv : access int;
      incx : int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:288
   pragma Import (C, cusolverDnClaswp, "cusolverDnClaswp");

   function cusolverDnZlaswp
     (handle : cusolverDnHandle_t;
      n : int;
      A : access cuComplex_h.cuDoubleComplex;
      lda : int;
      k1 : int;
      k2 : int;
      devIpiv : access int;
      incx : int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:298
   pragma Import (C, cusolverDnZlaswp, "cusolverDnZlaswp");

  -- LU solve  
   function cusolverDnSgetrs
     (handle : cusolverDnHandle_t;
      trans : cublas_api_h.cublasOperation_t;
      n : int;
      nrhs : int;
      A : access float;
      lda : int;
      devIpiv : access int;
      B : access float;
      ldb : int;
      devInfo : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:309
   pragma Import (C, cusolverDnSgetrs, "cusolverDnSgetrs");

   function cusolverDnDgetrs
     (handle : cusolverDnHandle_t;
      trans : cublas_api_h.cublasOperation_t;
      n : int;
      nrhs : int;
      A : access double;
      lda : int;
      devIpiv : access int;
      B : access double;
      ldb : int;
      devInfo : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:321
   pragma Import (C, cusolverDnDgetrs, "cusolverDnDgetrs");

   function cusolverDnCgetrs
     (handle : cusolverDnHandle_t;
      trans : cublas_api_h.cublasOperation_t;
      n : int;
      nrhs : int;
      A : access constant cuComplex_h.cuComplex;
      lda : int;
      devIpiv : access int;
      B : access cuComplex_h.cuComplex;
      ldb : int;
      devInfo : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:333
   pragma Import (C, cusolverDnCgetrs, "cusolverDnCgetrs");

   function cusolverDnZgetrs
     (handle : cusolverDnHandle_t;
      trans : cublas_api_h.cublasOperation_t;
      n : int;
      nrhs : int;
      A : access constant cuComplex_h.cuDoubleComplex;
      lda : int;
      devIpiv : access int;
      B : access cuComplex_h.cuDoubleComplex;
      ldb : int;
      devInfo : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:345
   pragma Import (C, cusolverDnZgetrs, "cusolverDnZgetrs");

  -- QR factorization  
   function cusolverDnSgeqrf_bufferSize
     (handle : cusolverDnHandle_t;
      m : int;
      n : int;
      A : access float;
      lda : int;
      lwork : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:359
   pragma Import (C, cusolverDnSgeqrf_bufferSize, "cusolverDnSgeqrf_bufferSize");

   function cusolverDnDgeqrf_bufferSize
     (handle : cusolverDnHandle_t;
      m : int;
      n : int;
      A : access double;
      lda : int;
      lwork : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:367
   pragma Import (C, cusolverDnDgeqrf_bufferSize, "cusolverDnDgeqrf_bufferSize");

   function cusolverDnCgeqrf_bufferSize
     (handle : cusolverDnHandle_t;
      m : int;
      n : int;
      A : access cuComplex_h.cuComplex;
      lda : int;
      lwork : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:375
   pragma Import (C, cusolverDnCgeqrf_bufferSize, "cusolverDnCgeqrf_bufferSize");

   function cusolverDnZgeqrf_bufferSize
     (handle : cusolverDnHandle_t;
      m : int;
      n : int;
      A : access cuComplex_h.cuDoubleComplex;
      lda : int;
      lwork : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:383
   pragma Import (C, cusolverDnZgeqrf_bufferSize, "cusolverDnZgeqrf_bufferSize");

   function cusolverDnSgeqrf
     (handle : cusolverDnHandle_t;
      m : int;
      n : int;
      A : access float;
      lda : int;
      TAU : access float;
      Workspace : access float;
      Lwork : int;
      devInfo : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:391
   pragma Import (C, cusolverDnSgeqrf, "cusolverDnSgeqrf");

   function cusolverDnDgeqrf
     (handle : cusolverDnHandle_t;
      m : int;
      n : int;
      A : access double;
      lda : int;
      TAU : access double;
      Workspace : access double;
      Lwork : int;
      devInfo : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:402
   pragma Import (C, cusolverDnDgeqrf, "cusolverDnDgeqrf");

   function cusolverDnCgeqrf
     (handle : cusolverDnHandle_t;
      m : int;
      n : int;
      A : access cuComplex_h.cuComplex;
      lda : int;
      TAU : access cuComplex_h.cuComplex;
      Workspace : access cuComplex_h.cuComplex;
      Lwork : int;
      devInfo : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:413
   pragma Import (C, cusolverDnCgeqrf, "cusolverDnCgeqrf");

   function cusolverDnZgeqrf
     (handle : cusolverDnHandle_t;
      m : int;
      n : int;
      A : access cuComplex_h.cuDoubleComplex;
      lda : int;
      TAU : access cuComplex_h.cuDoubleComplex;
      Workspace : access cuComplex_h.cuDoubleComplex;
      Lwork : int;
      devInfo : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:424
   pragma Import (C, cusolverDnZgeqrf, "cusolverDnZgeqrf");

  -- generate unitary matrix Q from QR factorization  
   function cusolverDnSorgqr_bufferSize
     (handle : cusolverDnHandle_t;
      m : int;
      n : int;
      k : int;
      A : access float;
      lda : int;
      tau : access float;
      lwork : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:437
   pragma Import (C, cusolverDnSorgqr_bufferSize, "cusolverDnSorgqr_bufferSize");

   function cusolverDnDorgqr_bufferSize
     (handle : cusolverDnHandle_t;
      m : int;
      n : int;
      k : int;
      A : access double;
      lda : int;
      tau : access double;
      lwork : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:447
   pragma Import (C, cusolverDnDorgqr_bufferSize, "cusolverDnDorgqr_bufferSize");

   function cusolverDnCungqr_bufferSize
     (handle : cusolverDnHandle_t;
      m : int;
      n : int;
      k : int;
      A : access constant cuComplex_h.cuComplex;
      lda : int;
      tau : access constant cuComplex_h.cuComplex;
      lwork : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:457
   pragma Import (C, cusolverDnCungqr_bufferSize, "cusolverDnCungqr_bufferSize");

   function cusolverDnZungqr_bufferSize
     (handle : cusolverDnHandle_t;
      m : int;
      n : int;
      k : int;
      A : access constant cuComplex_h.cuDoubleComplex;
      lda : int;
      tau : access constant cuComplex_h.cuDoubleComplex;
      lwork : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:467
   pragma Import (C, cusolverDnZungqr_bufferSize, "cusolverDnZungqr_bufferSize");

   function cusolverDnSorgqr
     (handle : cusolverDnHandle_t;
      m : int;
      n : int;
      k : int;
      A : access float;
      lda : int;
      tau : access float;
      work : access float;
      lwork : int;
      info : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:477
   pragma Import (C, cusolverDnSorgqr, "cusolverDnSorgqr");

   function cusolverDnDorgqr
     (handle : cusolverDnHandle_t;
      m : int;
      n : int;
      k : int;
      A : access double;
      lda : int;
      tau : access double;
      work : access double;
      lwork : int;
      info : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:489
   pragma Import (C, cusolverDnDorgqr, "cusolverDnDorgqr");

   function cusolverDnCungqr
     (handle : cusolverDnHandle_t;
      m : int;
      n : int;
      k : int;
      A : access cuComplex_h.cuComplex;
      lda : int;
      tau : access constant cuComplex_h.cuComplex;
      work : access cuComplex_h.cuComplex;
      lwork : int;
      info : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:501
   pragma Import (C, cusolverDnCungqr, "cusolverDnCungqr");

   function cusolverDnZungqr
     (handle : cusolverDnHandle_t;
      m : int;
      n : int;
      k : int;
      A : access cuComplex_h.cuDoubleComplex;
      lda : int;
      tau : access constant cuComplex_h.cuDoubleComplex;
      work : access cuComplex_h.cuDoubleComplex;
      lwork : int;
      info : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:513
   pragma Import (C, cusolverDnZungqr, "cusolverDnZungqr");

  -- compute Q**T*b in solve min||A*x = b||  
   function cusolverDnSormqr_bufferSize
     (handle : cusolverDnHandle_t;
      side : cublas_api_h.cublasSideMode_t;
      trans : cublas_api_h.cublasOperation_t;
      m : int;
      n : int;
      k : int;
      A : access float;
      lda : int;
      tau : access float;
      C : access float;
      ldc : int;
      lwork : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:528
   pragma Import (C, cusolverDnSormqr_bufferSize, "cusolverDnSormqr_bufferSize");

   function cusolverDnDormqr_bufferSize
     (handle : cusolverDnHandle_t;
      side : cublas_api_h.cublasSideMode_t;
      trans : cublas_api_h.cublasOperation_t;
      m : int;
      n : int;
      k : int;
      A : access double;
      lda : int;
      tau : access double;
      C : access double;
      ldc : int;
      lwork : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:542
   pragma Import (C, cusolverDnDormqr_bufferSize, "cusolverDnDormqr_bufferSize");

   function cusolverDnCunmqr_bufferSize
     (handle : cusolverDnHandle_t;
      side : cublas_api_h.cublasSideMode_t;
      trans : cublas_api_h.cublasOperation_t;
      m : int;
      n : int;
      k : int;
      A : access constant cuComplex_h.cuComplex;
      lda : int;
      tau : access constant cuComplex_h.cuComplex;
      C : access constant cuComplex_h.cuComplex;
      ldc : int;
      lwork : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:556
   pragma Import (C, cusolverDnCunmqr_bufferSize, "cusolverDnCunmqr_bufferSize");

   function cusolverDnZunmqr_bufferSize
     (handle : cusolverDnHandle_t;
      side : cublas_api_h.cublasSideMode_t;
      trans : cublas_api_h.cublasOperation_t;
      m : int;
      n : int;
      k : int;
      A : access constant cuComplex_h.cuDoubleComplex;
      lda : int;
      tau : access constant cuComplex_h.cuDoubleComplex;
      C : access constant cuComplex_h.cuDoubleComplex;
      ldc : int;
      lwork : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:570
   pragma Import (C, cusolverDnZunmqr_bufferSize, "cusolverDnZunmqr_bufferSize");

   function cusolverDnSormqr
     (handle : cusolverDnHandle_t;
      side : cublas_api_h.cublasSideMode_t;
      trans : cublas_api_h.cublasOperation_t;
      m : int;
      n : int;
      k : int;
      A : access float;
      lda : int;
      tau : access float;
      C : access float;
      ldc : int;
      work : access float;
      lwork : int;
      devInfo : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:584
   pragma Import (C, cusolverDnSormqr, "cusolverDnSormqr");

   function cusolverDnDormqr
     (handle : cusolverDnHandle_t;
      side : cublas_api_h.cublasSideMode_t;
      trans : cublas_api_h.cublasOperation_t;
      m : int;
      n : int;
      k : int;
      A : access double;
      lda : int;
      tau : access double;
      C : access double;
      ldc : int;
      work : access double;
      lwork : int;
      devInfo : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:600
   pragma Import (C, cusolverDnDormqr, "cusolverDnDormqr");

   function cusolverDnCunmqr
     (handle : cusolverDnHandle_t;
      side : cublas_api_h.cublasSideMode_t;
      trans : cublas_api_h.cublasOperation_t;
      m : int;
      n : int;
      k : int;
      A : access constant cuComplex_h.cuComplex;
      lda : int;
      tau : access constant cuComplex_h.cuComplex;
      C : access cuComplex_h.cuComplex;
      ldc : int;
      work : access cuComplex_h.cuComplex;
      lwork : int;
      devInfo : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:616
   pragma Import (C, cusolverDnCunmqr, "cusolverDnCunmqr");

   function cusolverDnZunmqr
     (handle : cusolverDnHandle_t;
      side : cublas_api_h.cublasSideMode_t;
      trans : cublas_api_h.cublasOperation_t;
      m : int;
      n : int;
      k : int;
      A : access constant cuComplex_h.cuDoubleComplex;
      lda : int;
      tau : access constant cuComplex_h.cuDoubleComplex;
      C : access cuComplex_h.cuDoubleComplex;
      ldc : int;
      work : access cuComplex_h.cuDoubleComplex;
      lwork : int;
      devInfo : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:632
   pragma Import (C, cusolverDnZunmqr, "cusolverDnZunmqr");

  -- L*D*L**T,U*D*U**T factorization  
   function cusolverDnSsytrf_bufferSize
     (handle : cusolverDnHandle_t;
      n : int;
      A : access float;
      lda : int;
      lwork : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:650
   pragma Import (C, cusolverDnSsytrf_bufferSize, "cusolverDnSsytrf_bufferSize");

   function cusolverDnDsytrf_bufferSize
     (handle : cusolverDnHandle_t;
      n : int;
      A : access double;
      lda : int;
      lwork : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:657
   pragma Import (C, cusolverDnDsytrf_bufferSize, "cusolverDnDsytrf_bufferSize");

   function cusolverDnCsytrf_bufferSize
     (handle : cusolverDnHandle_t;
      n : int;
      A : access cuComplex_h.cuComplex;
      lda : int;
      lwork : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:664
   pragma Import (C, cusolverDnCsytrf_bufferSize, "cusolverDnCsytrf_bufferSize");

   function cusolverDnZsytrf_bufferSize
     (handle : cusolverDnHandle_t;
      n : int;
      A : access cuComplex_h.cuDoubleComplex;
      lda : int;
      lwork : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:671
   pragma Import (C, cusolverDnZsytrf_bufferSize, "cusolverDnZsytrf_bufferSize");

   function cusolverDnSsytrf
     (handle : cusolverDnHandle_t;
      uplo : cublas_api_h.cublasFillMode_t;
      n : int;
      A : access float;
      lda : int;
      ipiv : access int;
      work : access float;
      lwork : int;
      info : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:678
   pragma Import (C, cusolverDnSsytrf, "cusolverDnSsytrf");

   function cusolverDnDsytrf
     (handle : cusolverDnHandle_t;
      uplo : cublas_api_h.cublasFillMode_t;
      n : int;
      A : access double;
      lda : int;
      ipiv : access int;
      work : access double;
      lwork : int;
      info : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:689
   pragma Import (C, cusolverDnDsytrf, "cusolverDnDsytrf");

   function cusolverDnCsytrf
     (handle : cusolverDnHandle_t;
      uplo : cublas_api_h.cublasFillMode_t;
      n : int;
      A : access cuComplex_h.cuComplex;
      lda : int;
      ipiv : access int;
      work : access cuComplex_h.cuComplex;
      lwork : int;
      info : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:700
   pragma Import (C, cusolverDnCsytrf, "cusolverDnCsytrf");

   function cusolverDnZsytrf
     (handle : cusolverDnHandle_t;
      uplo : cublas_api_h.cublasFillMode_t;
      n : int;
      A : access cuComplex_h.cuDoubleComplex;
      lda : int;
      ipiv : access int;
      work : access cuComplex_h.cuDoubleComplex;
      lwork : int;
      info : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:711
   pragma Import (C, cusolverDnZsytrf, "cusolverDnZsytrf");

  -- bidiagonal factorization  
   function cusolverDnSgebrd_bufferSize
     (handle : cusolverDnHandle_t;
      m : int;
      n : int;
      Lwork : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:724
   pragma Import (C, cusolverDnSgebrd_bufferSize, "cusolverDnSgebrd_bufferSize");

   function cusolverDnDgebrd_bufferSize
     (handle : cusolverDnHandle_t;
      m : int;
      n : int;
      Lwork : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:730
   pragma Import (C, cusolverDnDgebrd_bufferSize, "cusolverDnDgebrd_bufferSize");

   function cusolverDnCgebrd_bufferSize
     (handle : cusolverDnHandle_t;
      m : int;
      n : int;
      Lwork : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:736
   pragma Import (C, cusolverDnCgebrd_bufferSize, "cusolverDnCgebrd_bufferSize");

   function cusolverDnZgebrd_bufferSize
     (handle : cusolverDnHandle_t;
      m : int;
      n : int;
      Lwork : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:742
   pragma Import (C, cusolverDnZgebrd_bufferSize, "cusolverDnZgebrd_bufferSize");

   function cusolverDnSgebrd
     (handle : cusolverDnHandle_t;
      m : int;
      n : int;
      A : access float;
      lda : int;
      D : access float;
      E : access float;
      TAUQ : access float;
      TAUP : access float;
      Work : access float;
      Lwork : int;
      devInfo : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:748
   pragma Import (C, cusolverDnSgebrd, "cusolverDnSgebrd");

   function cusolverDnDgebrd
     (handle : cusolverDnHandle_t;
      m : int;
      n : int;
      A : access double;
      lda : int;
      D : access double;
      E : access double;
      TAUQ : access double;
      TAUP : access double;
      Work : access double;
      Lwork : int;
      devInfo : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:762
   pragma Import (C, cusolverDnDgebrd, "cusolverDnDgebrd");

   function cusolverDnCgebrd
     (handle : cusolverDnHandle_t;
      m : int;
      n : int;
      A : access cuComplex_h.cuComplex;
      lda : int;
      D : access float;
      E : access float;
      TAUQ : access cuComplex_h.cuComplex;
      TAUP : access cuComplex_h.cuComplex;
      Work : access cuComplex_h.cuComplex;
      Lwork : int;
      devInfo : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:776
   pragma Import (C, cusolverDnCgebrd, "cusolverDnCgebrd");

   function cusolverDnZgebrd
     (handle : cusolverDnHandle_t;
      m : int;
      n : int;
      A : access cuComplex_h.cuDoubleComplex;
      lda : int;
      D : access double;
      E : access double;
      TAUQ : access cuComplex_h.cuDoubleComplex;
      TAUP : access cuComplex_h.cuDoubleComplex;
      Work : access cuComplex_h.cuDoubleComplex;
      Lwork : int;
      devInfo : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:790
   pragma Import (C, cusolverDnZgebrd, "cusolverDnZgebrd");

  -- generates one of the unitary matrices Q or P**T determined by GEBRD 
   function cusolverDnSorgbr_bufferSize
     (handle : cusolverDnHandle_t;
      side : cublas_api_h.cublasSideMode_t;
      m : int;
      n : int;
      k : int;
      A : access float;
      lda : int;
      tau : access float;
      lwork : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:805
   pragma Import (C, cusolverDnSorgbr_bufferSize, "cusolverDnSorgbr_bufferSize");

   function cusolverDnDorgbr_bufferSize
     (handle : cusolverDnHandle_t;
      side : cublas_api_h.cublasSideMode_t;
      m : int;
      n : int;
      k : int;
      A : access double;
      lda : int;
      tau : access double;
      lwork : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:816
   pragma Import (C, cusolverDnDorgbr_bufferSize, "cusolverDnDorgbr_bufferSize");

   function cusolverDnCungbr_bufferSize
     (handle : cusolverDnHandle_t;
      side : cublas_api_h.cublasSideMode_t;
      m : int;
      n : int;
      k : int;
      A : access constant cuComplex_h.cuComplex;
      lda : int;
      tau : access constant cuComplex_h.cuComplex;
      lwork : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:827
   pragma Import (C, cusolverDnCungbr_bufferSize, "cusolverDnCungbr_bufferSize");

   function cusolverDnZungbr_bufferSize
     (handle : cusolverDnHandle_t;
      side : cublas_api_h.cublasSideMode_t;
      m : int;
      n : int;
      k : int;
      A : access constant cuComplex_h.cuDoubleComplex;
      lda : int;
      tau : access constant cuComplex_h.cuDoubleComplex;
      lwork : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:838
   pragma Import (C, cusolverDnZungbr_bufferSize, "cusolverDnZungbr_bufferSize");

   function cusolverDnSorgbr
     (handle : cusolverDnHandle_t;
      side : cublas_api_h.cublasSideMode_t;
      m : int;
      n : int;
      k : int;
      A : access float;
      lda : int;
      tau : access float;
      work : access float;
      lwork : int;
      info : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:849
   pragma Import (C, cusolverDnSorgbr, "cusolverDnSorgbr");

   function cusolverDnDorgbr
     (handle : cusolverDnHandle_t;
      side : cublas_api_h.cublasSideMode_t;
      m : int;
      n : int;
      k : int;
      A : access double;
      lda : int;
      tau : access double;
      work : access double;
      lwork : int;
      info : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:862
   pragma Import (C, cusolverDnDorgbr, "cusolverDnDorgbr");

   function cusolverDnCungbr
     (handle : cusolverDnHandle_t;
      side : cublas_api_h.cublasSideMode_t;
      m : int;
      n : int;
      k : int;
      A : access cuComplex_h.cuComplex;
      lda : int;
      tau : access constant cuComplex_h.cuComplex;
      work : access cuComplex_h.cuComplex;
      lwork : int;
      info : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:875
   pragma Import (C, cusolverDnCungbr, "cusolverDnCungbr");

   function cusolverDnZungbr
     (handle : cusolverDnHandle_t;
      side : cublas_api_h.cublasSideMode_t;
      m : int;
      n : int;
      k : int;
      A : access cuComplex_h.cuDoubleComplex;
      lda : int;
      tau : access constant cuComplex_h.cuDoubleComplex;
      work : access cuComplex_h.cuDoubleComplex;
      lwork : int;
      info : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:888
   pragma Import (C, cusolverDnZungbr, "cusolverDnZungbr");

  -- tridiagonal factorization  
   function cusolverDnSsytrd_bufferSize
     (handle : cusolverDnHandle_t;
      uplo : cublas_api_h.cublasFillMode_t;
      n : int;
      A : access float;
      lda : int;
      d : access float;
      e : access float;
      tau : access float;
      lwork : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:903
   pragma Import (C, cusolverDnSsytrd_bufferSize, "cusolverDnSsytrd_bufferSize");

   function cusolverDnDsytrd_bufferSize
     (handle : cusolverDnHandle_t;
      uplo : cublas_api_h.cublasFillMode_t;
      n : int;
      A : access double;
      lda : int;
      d : access double;
      e : access double;
      tau : access double;
      lwork : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:914
   pragma Import (C, cusolverDnDsytrd_bufferSize, "cusolverDnDsytrd_bufferSize");

   function cusolverDnChetrd_bufferSize
     (handle : cusolverDnHandle_t;
      uplo : cublas_api_h.cublasFillMode_t;
      n : int;
      A : access constant cuComplex_h.cuComplex;
      lda : int;
      d : access float;
      e : access float;
      tau : access constant cuComplex_h.cuComplex;
      lwork : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:925
   pragma Import (C, cusolverDnChetrd_bufferSize, "cusolverDnChetrd_bufferSize");

   function cusolverDnZhetrd_bufferSize
     (handle : cusolverDnHandle_t;
      uplo : cublas_api_h.cublasFillMode_t;
      n : int;
      A : access constant cuComplex_h.cuDoubleComplex;
      lda : int;
      d : access double;
      e : access double;
      tau : access constant cuComplex_h.cuDoubleComplex;
      lwork : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:936
   pragma Import (C, cusolverDnZhetrd_bufferSize, "cusolverDnZhetrd_bufferSize");

   function cusolverDnSsytrd
     (handle : cusolverDnHandle_t;
      uplo : cublas_api_h.cublasFillMode_t;
      n : int;
      A : access float;
      lda : int;
      d : access float;
      e : access float;
      tau : access float;
      work : access float;
      lwork : int;
      info : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:948
   pragma Import (C, cusolverDnSsytrd, "cusolverDnSsytrd");

   function cusolverDnDsytrd
     (handle : cusolverDnHandle_t;
      uplo : cublas_api_h.cublasFillMode_t;
      n : int;
      A : access double;
      lda : int;
      d : access double;
      e : access double;
      tau : access double;
      work : access double;
      lwork : int;
      info : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:961
   pragma Import (C, cusolverDnDsytrd, "cusolverDnDsytrd");

   function cusolverDnChetrd
     (handle : cusolverDnHandle_t;
      uplo : cublas_api_h.cublasFillMode_t;
      n : int;
      A : access cuComplex_h.cuComplex;
      lda : int;
      d : access float;
      e : access float;
      tau : access cuComplex_h.cuComplex;
      work : access cuComplex_h.cuComplex;
      lwork : int;
      info : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:974
   pragma Import (C, cusolverDnChetrd, "cusolverDnChetrd");

   function cusolverDnZhetrd
     (handle : cusolverDnHandle_t;
      uplo : cublas_api_h.cublasFillMode_t;
      n : int;
      A : access cuComplex_h.cuDoubleComplex;
      lda : int;
      d : access double;
      e : access double;
      tau : access cuComplex_h.cuDoubleComplex;
      work : access cuComplex_h.cuDoubleComplex;
      lwork : int;
      info : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:987
   pragma Import (C, cusolverDnZhetrd, "cusolverDnZhetrd");

  -- generate unitary Q comes from sytrd  
   function cusolverDnSorgtr_bufferSize
     (handle : cusolverDnHandle_t;
      uplo : cublas_api_h.cublasFillMode_t;
      n : int;
      A : access float;
      lda : int;
      tau : access float;
      lwork : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:1003
   pragma Import (C, cusolverDnSorgtr_bufferSize, "cusolverDnSorgtr_bufferSize");

   function cusolverDnDorgtr_bufferSize
     (handle : cusolverDnHandle_t;
      uplo : cublas_api_h.cublasFillMode_t;
      n : int;
      A : access double;
      lda : int;
      tau : access double;
      lwork : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:1012
   pragma Import (C, cusolverDnDorgtr_bufferSize, "cusolverDnDorgtr_bufferSize");

   function cusolverDnCungtr_bufferSize
     (handle : cusolverDnHandle_t;
      uplo : cublas_api_h.cublasFillMode_t;
      n : int;
      A : access constant cuComplex_h.cuComplex;
      lda : int;
      tau : access constant cuComplex_h.cuComplex;
      lwork : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:1021
   pragma Import (C, cusolverDnCungtr_bufferSize, "cusolverDnCungtr_bufferSize");

   function cusolverDnZungtr_bufferSize
     (handle : cusolverDnHandle_t;
      uplo : cublas_api_h.cublasFillMode_t;
      n : int;
      A : access constant cuComplex_h.cuDoubleComplex;
      lda : int;
      tau : access constant cuComplex_h.cuDoubleComplex;
      lwork : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:1030
   pragma Import (C, cusolverDnZungtr_bufferSize, "cusolverDnZungtr_bufferSize");

   function cusolverDnSorgtr
     (handle : cusolverDnHandle_t;
      uplo : cublas_api_h.cublasFillMode_t;
      n : int;
      A : access float;
      lda : int;
      tau : access float;
      work : access float;
      lwork : int;
      info : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:1039
   pragma Import (C, cusolverDnSorgtr, "cusolverDnSorgtr");

   function cusolverDnDorgtr
     (handle : cusolverDnHandle_t;
      uplo : cublas_api_h.cublasFillMode_t;
      n : int;
      A : access double;
      lda : int;
      tau : access double;
      work : access double;
      lwork : int;
      info : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:1050
   pragma Import (C, cusolverDnDorgtr, "cusolverDnDorgtr");

   function cusolverDnCungtr
     (handle : cusolverDnHandle_t;
      uplo : cublas_api_h.cublasFillMode_t;
      n : int;
      A : access cuComplex_h.cuComplex;
      lda : int;
      tau : access constant cuComplex_h.cuComplex;
      work : access cuComplex_h.cuComplex;
      lwork : int;
      info : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:1061
   pragma Import (C, cusolverDnCungtr, "cusolverDnCungtr");

   function cusolverDnZungtr
     (handle : cusolverDnHandle_t;
      uplo : cublas_api_h.cublasFillMode_t;
      n : int;
      A : access cuComplex_h.cuDoubleComplex;
      lda : int;
      tau : access constant cuComplex_h.cuDoubleComplex;
      work : access cuComplex_h.cuDoubleComplex;
      lwork : int;
      info : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:1072
   pragma Import (C, cusolverDnZungtr, "cusolverDnZungtr");

  -- compute op(Q)*C or C*op(Q) where Q comes from sytrd  
   function cusolverDnSormtr_bufferSize
     (handle : cusolverDnHandle_t;
      side : cublas_api_h.cublasSideMode_t;
      uplo : cublas_api_h.cublasFillMode_t;
      trans : cublas_api_h.cublasOperation_t;
      m : int;
      n : int;
      A : access float;
      lda : int;
      tau : access float;
      C : access float;
      ldc : int;
      lwork : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:1086
   pragma Import (C, cusolverDnSormtr_bufferSize, "cusolverDnSormtr_bufferSize");

   function cusolverDnDormtr_bufferSize
     (handle : cusolverDnHandle_t;
      side : cublas_api_h.cublasSideMode_t;
      uplo : cublas_api_h.cublasFillMode_t;
      trans : cublas_api_h.cublasOperation_t;
      m : int;
      n : int;
      A : access double;
      lda : int;
      tau : access double;
      C : access double;
      ldc : int;
      lwork : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:1100
   pragma Import (C, cusolverDnDormtr_bufferSize, "cusolverDnDormtr_bufferSize");

   function cusolverDnCunmtr_bufferSize
     (handle : cusolverDnHandle_t;
      side : cublas_api_h.cublasSideMode_t;
      uplo : cublas_api_h.cublasFillMode_t;
      trans : cublas_api_h.cublasOperation_t;
      m : int;
      n : int;
      A : access constant cuComplex_h.cuComplex;
      lda : int;
      tau : access constant cuComplex_h.cuComplex;
      C : access constant cuComplex_h.cuComplex;
      ldc : int;
      lwork : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:1114
   pragma Import (C, cusolverDnCunmtr_bufferSize, "cusolverDnCunmtr_bufferSize");

   function cusolverDnZunmtr_bufferSize
     (handle : cusolverDnHandle_t;
      side : cublas_api_h.cublasSideMode_t;
      uplo : cublas_api_h.cublasFillMode_t;
      trans : cublas_api_h.cublasOperation_t;
      m : int;
      n : int;
      A : access constant cuComplex_h.cuDoubleComplex;
      lda : int;
      tau : access constant cuComplex_h.cuDoubleComplex;
      C : access constant cuComplex_h.cuDoubleComplex;
      ldc : int;
      lwork : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:1128
   pragma Import (C, cusolverDnZunmtr_bufferSize, "cusolverDnZunmtr_bufferSize");

   function cusolverDnSormtr
     (handle : cusolverDnHandle_t;
      side : cublas_api_h.cublasSideMode_t;
      uplo : cublas_api_h.cublasFillMode_t;
      trans : cublas_api_h.cublasOperation_t;
      m : int;
      n : int;
      A : access float;
      lda : int;
      tau : access float;
      C : access float;
      ldc : int;
      work : access float;
      lwork : int;
      info : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:1142
   pragma Import (C, cusolverDnSormtr, "cusolverDnSormtr");

   function cusolverDnDormtr
     (handle : cusolverDnHandle_t;
      side : cublas_api_h.cublasSideMode_t;
      uplo : cublas_api_h.cublasFillMode_t;
      trans : cublas_api_h.cublasOperation_t;
      m : int;
      n : int;
      A : access double;
      lda : int;
      tau : access double;
      C : access double;
      ldc : int;
      work : access double;
      lwork : int;
      info : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:1158
   pragma Import (C, cusolverDnDormtr, "cusolverDnDormtr");

   function cusolverDnCunmtr
     (handle : cusolverDnHandle_t;
      side : cublas_api_h.cublasSideMode_t;
      uplo : cublas_api_h.cublasFillMode_t;
      trans : cublas_api_h.cublasOperation_t;
      m : int;
      n : int;
      A : access cuComplex_h.cuComplex;
      lda : int;
      tau : access cuComplex_h.cuComplex;
      C : access cuComplex_h.cuComplex;
      ldc : int;
      work : access cuComplex_h.cuComplex;
      lwork : int;
      info : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:1174
   pragma Import (C, cusolverDnCunmtr, "cusolverDnCunmtr");

   function cusolverDnZunmtr
     (handle : cusolverDnHandle_t;
      side : cublas_api_h.cublasSideMode_t;
      uplo : cublas_api_h.cublasFillMode_t;
      trans : cublas_api_h.cublasOperation_t;
      m : int;
      n : int;
      A : access cuComplex_h.cuDoubleComplex;
      lda : int;
      tau : access cuComplex_h.cuDoubleComplex;
      C : access cuComplex_h.cuDoubleComplex;
      ldc : int;
      work : access cuComplex_h.cuDoubleComplex;
      lwork : int;
      info : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:1190
   pragma Import (C, cusolverDnZunmtr, "cusolverDnZunmtr");

  -- singular value decomposition, A = U * Sigma * V^H  
   function cusolverDnSgesvd_bufferSize
     (handle : cusolverDnHandle_t;
      m : int;
      n : int;
      lwork : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:1209
   pragma Import (C, cusolverDnSgesvd_bufferSize, "cusolverDnSgesvd_bufferSize");

   function cusolverDnDgesvd_bufferSize
     (handle : cusolverDnHandle_t;
      m : int;
      n : int;
      lwork : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:1215
   pragma Import (C, cusolverDnDgesvd_bufferSize, "cusolverDnDgesvd_bufferSize");

   function cusolverDnCgesvd_bufferSize
     (handle : cusolverDnHandle_t;
      m : int;
      n : int;
      lwork : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:1221
   pragma Import (C, cusolverDnCgesvd_bufferSize, "cusolverDnCgesvd_bufferSize");

   function cusolverDnZgesvd_bufferSize
     (handle : cusolverDnHandle_t;
      m : int;
      n : int;
      lwork : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:1227
   pragma Import (C, cusolverDnZgesvd_bufferSize, "cusolverDnZgesvd_bufferSize");

   function cusolverDnSgesvd
     (handle : cusolverDnHandle_t;
      jobu : signed_char;
      jobvt : signed_char;
      m : int;
      n : int;
      A : access float;
      lda : int;
      S : access float;
      U : access float;
      ldu : int;
      VT : access float;
      ldvt : int;
      work : access float;
      lwork : int;
      rwork : access float;
      info : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:1233
   pragma Import (C, cusolverDnSgesvd, "cusolverDnSgesvd");

   function cusolverDnDgesvd
     (handle : cusolverDnHandle_t;
      jobu : signed_char;
      jobvt : signed_char;
      m : int;
      n : int;
      A : access double;
      lda : int;
      S : access double;
      U : access double;
      ldu : int;
      VT : access double;
      ldvt : int;
      work : access double;
      lwork : int;
      rwork : access double;
      info : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:1251
   pragma Import (C, cusolverDnDgesvd, "cusolverDnDgesvd");

   function cusolverDnCgesvd
     (handle : cusolverDnHandle_t;
      jobu : signed_char;
      jobvt : signed_char;
      m : int;
      n : int;
      A : access cuComplex_h.cuComplex;
      lda : int;
      S : access float;
      U : access cuComplex_h.cuComplex;
      ldu : int;
      VT : access cuComplex_h.cuComplex;
      ldvt : int;
      work : access cuComplex_h.cuComplex;
      lwork : int;
      rwork : access float;
      info : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:1269
   pragma Import (C, cusolverDnCgesvd, "cusolverDnCgesvd");

   function cusolverDnZgesvd
     (handle : cusolverDnHandle_t;
      jobu : signed_char;
      jobvt : signed_char;
      m : int;
      n : int;
      A : access cuComplex_h.cuDoubleComplex;
      lda : int;
      S : access double;
      U : access cuComplex_h.cuDoubleComplex;
      ldu : int;
      VT : access cuComplex_h.cuDoubleComplex;
      ldvt : int;
      work : access cuComplex_h.cuDoubleComplex;
      lwork : int;
      rwork : access double;
      info : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:1287
   pragma Import (C, cusolverDnZgesvd, "cusolverDnZgesvd");

  -- standard symmetric eigenvalue solver, A*x = lambda*x, by divide-and-conquer   
   function cusolverDnSsyevd_bufferSize
     (handle : cusolverDnHandle_t;
      jobz : cusolver_common_h.cusolverEigMode_t;
      uplo : cublas_api_h.cublasFillMode_t;
      n : int;
      A : access float;
      lda : int;
      W : access float;
      lwork : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:1307
   pragma Import (C, cusolverDnSsyevd_bufferSize, "cusolverDnSsyevd_bufferSize");

   function cusolverDnDsyevd_bufferSize
     (handle : cusolverDnHandle_t;
      jobz : cusolver_common_h.cusolverEigMode_t;
      uplo : cublas_api_h.cublasFillMode_t;
      n : int;
      A : access double;
      lda : int;
      W : access double;
      lwork : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:1317
   pragma Import (C, cusolverDnDsyevd_bufferSize, "cusolverDnDsyevd_bufferSize");

   function cusolverDnCheevd_bufferSize
     (handle : cusolverDnHandle_t;
      jobz : cusolver_common_h.cusolverEigMode_t;
      uplo : cublas_api_h.cublasFillMode_t;
      n : int;
      A : access constant cuComplex_h.cuComplex;
      lda : int;
      W : access float;
      lwork : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:1327
   pragma Import (C, cusolverDnCheevd_bufferSize, "cusolverDnCheevd_bufferSize");

   function cusolverDnZheevd_bufferSize
     (handle : cusolverDnHandle_t;
      jobz : cusolver_common_h.cusolverEigMode_t;
      uplo : cublas_api_h.cublasFillMode_t;
      n : int;
      A : access constant cuComplex_h.cuDoubleComplex;
      lda : int;
      W : access double;
      lwork : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:1337
   pragma Import (C, cusolverDnZheevd_bufferSize, "cusolverDnZheevd_bufferSize");

   function cusolverDnSsyevd
     (handle : cusolverDnHandle_t;
      jobz : cusolver_common_h.cusolverEigMode_t;
      uplo : cublas_api_h.cublasFillMode_t;
      n : int;
      A : access float;
      lda : int;
      W : access float;
      work : access float;
      lwork : int;
      info : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:1347
   pragma Import (C, cusolverDnSsyevd, "cusolverDnSsyevd");

   function cusolverDnDsyevd
     (handle : cusolverDnHandle_t;
      jobz : cusolver_common_h.cusolverEigMode_t;
      uplo : cublas_api_h.cublasFillMode_t;
      n : int;
      A : access double;
      lda : int;
      W : access double;
      work : access double;
      lwork : int;
      info : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:1359
   pragma Import (C, cusolverDnDsyevd, "cusolverDnDsyevd");

   function cusolverDnCheevd
     (handle : cusolverDnHandle_t;
      jobz : cusolver_common_h.cusolverEigMode_t;
      uplo : cublas_api_h.cublasFillMode_t;
      n : int;
      A : access cuComplex_h.cuComplex;
      lda : int;
      W : access float;
      work : access cuComplex_h.cuComplex;
      lwork : int;
      info : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:1371
   pragma Import (C, cusolverDnCheevd, "cusolverDnCheevd");

   function cusolverDnZheevd
     (handle : cusolverDnHandle_t;
      jobz : cusolver_common_h.cusolverEigMode_t;
      uplo : cublas_api_h.cublasFillMode_t;
      n : int;
      A : access cuComplex_h.cuDoubleComplex;
      lda : int;
      W : access double;
      work : access cuComplex_h.cuDoubleComplex;
      lwork : int;
      info : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:1383
   pragma Import (C, cusolverDnZheevd, "cusolverDnZheevd");

  -- generalized symmetric eigenvalue solver, A*x = lambda*B*x, by divide-and-conquer   
   function cusolverDnSsygvd_bufferSize
     (handle : cusolverDnHandle_t;
      itype : cusolver_common_h.cusolverEigType_t;
      jobz : cusolver_common_h.cusolverEigMode_t;
      uplo : cublas_api_h.cublasFillMode_t;
      n : int;
      A : access float;
      lda : int;
      B : access float;
      ldb : int;
      W : access float;
      lwork : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:1397
   pragma Import (C, cusolverDnSsygvd_bufferSize, "cusolverDnSsygvd_bufferSize");

   function cusolverDnDsygvd_bufferSize
     (handle : cusolverDnHandle_t;
      itype : cusolver_common_h.cusolverEigType_t;
      jobz : cusolver_common_h.cusolverEigMode_t;
      uplo : cublas_api_h.cublasFillMode_t;
      n : int;
      A : access double;
      lda : int;
      B : access double;
      ldb : int;
      W : access double;
      lwork : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:1410
   pragma Import (C, cusolverDnDsygvd_bufferSize, "cusolverDnDsygvd_bufferSize");

   function cusolverDnChegvd_bufferSize
     (handle : cusolverDnHandle_t;
      itype : cusolver_common_h.cusolverEigType_t;
      jobz : cusolver_common_h.cusolverEigMode_t;
      uplo : cublas_api_h.cublasFillMode_t;
      n : int;
      A : access constant cuComplex_h.cuComplex;
      lda : int;
      B : access constant cuComplex_h.cuComplex;
      ldb : int;
      W : access float;
      lwork : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:1423
   pragma Import (C, cusolverDnChegvd_bufferSize, "cusolverDnChegvd_bufferSize");

   function cusolverDnZhegvd_bufferSize
     (handle : cusolverDnHandle_t;
      itype : cusolver_common_h.cusolverEigType_t;
      jobz : cusolver_common_h.cusolverEigMode_t;
      uplo : cublas_api_h.cublasFillMode_t;
      n : int;
      A : access constant cuComplex_h.cuDoubleComplex;
      lda : int;
      B : access constant cuComplex_h.cuDoubleComplex;
      ldb : int;
      W : access double;
      lwork : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:1436
   pragma Import (C, cusolverDnZhegvd_bufferSize, "cusolverDnZhegvd_bufferSize");

   function cusolverDnSsygvd
     (handle : cusolverDnHandle_t;
      itype : cusolver_common_h.cusolverEigType_t;
      jobz : cusolver_common_h.cusolverEigMode_t;
      uplo : cublas_api_h.cublasFillMode_t;
      n : int;
      A : access float;
      lda : int;
      B : access float;
      ldb : int;
      W : access float;
      work : access float;
      lwork : int;
      info : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:1450
   pragma Import (C, cusolverDnSsygvd, "cusolverDnSsygvd");

   function cusolverDnDsygvd
     (handle : cusolverDnHandle_t;
      itype : cusolver_common_h.cusolverEigType_t;
      jobz : cusolver_common_h.cusolverEigMode_t;
      uplo : cublas_api_h.cublasFillMode_t;
      n : int;
      A : access double;
      lda : int;
      B : access double;
      ldb : int;
      W : access double;
      work : access double;
      lwork : int;
      info : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:1465
   pragma Import (C, cusolverDnDsygvd, "cusolverDnDsygvd");

   function cusolverDnChegvd
     (handle : cusolverDnHandle_t;
      itype : cusolver_common_h.cusolverEigType_t;
      jobz : cusolver_common_h.cusolverEigMode_t;
      uplo : cublas_api_h.cublasFillMode_t;
      n : int;
      A : access cuComplex_h.cuComplex;
      lda : int;
      B : access cuComplex_h.cuComplex;
      ldb : int;
      W : access float;
      work : access cuComplex_h.cuComplex;
      lwork : int;
      info : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:1480
   pragma Import (C, cusolverDnChegvd, "cusolverDnChegvd");

   function cusolverDnZhegvd
     (handle : cusolverDnHandle_t;
      itype : cusolver_common_h.cusolverEigType_t;
      jobz : cusolver_common_h.cusolverEigMode_t;
      uplo : cublas_api_h.cublasFillMode_t;
      n : int;
      A : access cuComplex_h.cuDoubleComplex;
      lda : int;
      B : access cuComplex_h.cuDoubleComplex;
      ldb : int;
      W : access double;
      work : access cuComplex_h.cuDoubleComplex;
      lwork : int;
      info : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverDn.h:1495
   pragma Import (C, cusolverDnZhegvd, "cusolverDnZhegvd");

end cusolverDn_h;

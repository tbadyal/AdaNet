pragma Ada_2005;
pragma Style_Checks (Off);

with Interfaces.C; use Interfaces.C;
with System;
with cusolver_common_h;
with cusolverSp_h;
with cusparse_h;
with stddef_h;
with cuComplex_h;

package cusolverSp_LOWLEVEL_PREVIEW_h is

  -- * Copyright 2015 NVIDIA Corporation.  All rights reserved.
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

   --  skipped empty struct csrluInfoHost

   type csrluInfoHost_t is new System.Address;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:61

   --  skipped empty struct csrqrInfoHost

   type csrqrInfoHost_t is new System.Address;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:65

   --  skipped empty struct csrcholInfoHost

   type csrcholInfoHost_t is new System.Address;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:69

   --  skipped empty struct csrcholInfo

   type csrcholInfo_t is new System.Address;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:73

  -- * Low level API for CPU LU
  -- * 
  --  

   function cusolverSpCreateCsrluInfoHost (info : System.Address) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:81
   pragma Import (C, cusolverSpCreateCsrluInfoHost, "cusolverSpCreateCsrluInfoHost");

   function cusolverSpDestroyCsrluInfoHost (info : csrluInfoHost_t) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:84
   pragma Import (C, cusolverSpDestroyCsrluInfoHost, "cusolverSpDestroyCsrluInfoHost");

   function cusolverSpXcsrluAnalysisHost
     (handle : cusolverSp_h.cusolverSpHandle_t;
      n : int;
      nnzA : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrRowPtrA : access int;
      csrColIndA : access int;
      info : csrluInfoHost_t) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:87
   pragma Import (C, cusolverSpXcsrluAnalysisHost, "cusolverSpXcsrluAnalysisHost");

   function cusolverSpScsrluBufferInfoHost
     (handle : cusolverSp_h.cusolverSpHandle_t;
      n : int;
      nnzA : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrValA : access float;
      csrRowPtrA : access int;
      csrColIndA : access int;
      info : csrluInfoHost_t;
      internalDataInBytes : access stddef_h.size_t;
      workspaceInBytes : access stddef_h.size_t) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:96
   pragma Import (C, cusolverSpScsrluBufferInfoHost, "cusolverSpScsrluBufferInfoHost");

   function cusolverSpDcsrluBufferInfoHost
     (handle : cusolverSp_h.cusolverSpHandle_t;
      n : int;
      nnzA : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrValA : access double;
      csrRowPtrA : access int;
      csrColIndA : access int;
      info : csrluInfoHost_t;
      internalDataInBytes : access stddef_h.size_t;
      workspaceInBytes : access stddef_h.size_t) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:108
   pragma Import (C, cusolverSpDcsrluBufferInfoHost, "cusolverSpDcsrluBufferInfoHost");

   function cusolverSpCcsrluBufferInfoHost
     (handle : cusolverSp_h.cusolverSpHandle_t;
      n : int;
      nnzA : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrValA : access constant cuComplex_h.cuComplex;
      csrRowPtrA : access int;
      csrColIndA : access int;
      info : csrluInfoHost_t;
      internalDataInBytes : access stddef_h.size_t;
      workspaceInBytes : access stddef_h.size_t) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:120
   pragma Import (C, cusolverSpCcsrluBufferInfoHost, "cusolverSpCcsrluBufferInfoHost");

   function cusolverSpZcsrluBufferInfoHost
     (handle : cusolverSp_h.cusolverSpHandle_t;
      n : int;
      nnzA : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrValA : access constant cuComplex_h.cuDoubleComplex;
      csrRowPtrA : access int;
      csrColIndA : access int;
      info : csrluInfoHost_t;
      internalDataInBytes : access stddef_h.size_t;
      workspaceInBytes : access stddef_h.size_t) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:132
   pragma Import (C, cusolverSpZcsrluBufferInfoHost, "cusolverSpZcsrluBufferInfoHost");

   function cusolverSpScsrluFactorHost
     (handle : cusolverSp_h.cusolverSpHandle_t;
      n : int;
      nnzA : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrValA : access float;
      csrRowPtrA : access int;
      csrColIndA : access int;
      info : csrluInfoHost_t;
      pivot_threshold : float;
      pBuffer : System.Address) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:144
   pragma Import (C, cusolverSpScsrluFactorHost, "cusolverSpScsrluFactorHost");

   function cusolverSpDcsrluFactorHost
     (handle : cusolverSp_h.cusolverSpHandle_t;
      n : int;
      nnzA : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrValA : access double;
      csrRowPtrA : access int;
      csrColIndA : access int;
      info : csrluInfoHost_t;
      pivot_threshold : double;
      pBuffer : System.Address) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:156
   pragma Import (C, cusolverSpDcsrluFactorHost, "cusolverSpDcsrluFactorHost");

   function cusolverSpCcsrluFactorHost
     (handle : cusolverSp_h.cusolverSpHandle_t;
      n : int;
      nnzA : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrValA : access constant cuComplex_h.cuComplex;
      csrRowPtrA : access int;
      csrColIndA : access int;
      info : csrluInfoHost_t;
      pivot_threshold : float;
      pBuffer : System.Address) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:168
   pragma Import (C, cusolverSpCcsrluFactorHost, "cusolverSpCcsrluFactorHost");

   function cusolverSpZcsrluFactorHost
     (handle : cusolverSp_h.cusolverSpHandle_t;
      n : int;
      nnzA : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrValA : access constant cuComplex_h.cuDoubleComplex;
      csrRowPtrA : access int;
      csrColIndA : access int;
      info : csrluInfoHost_t;
      pivot_threshold : double;
      pBuffer : System.Address) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:180
   pragma Import (C, cusolverSpZcsrluFactorHost, "cusolverSpZcsrluFactorHost");

   function cusolverSpScsrluZeroPivotHost
     (handle : cusolverSp_h.cusolverSpHandle_t;
      info : csrluInfoHost_t;
      tol : float;
      position : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:193
   pragma Import (C, cusolverSpScsrluZeroPivotHost, "cusolverSpScsrluZeroPivotHost");

   function cusolverSpDcsrluZeroPivotHost
     (handle : cusolverSp_h.cusolverSpHandle_t;
      info : csrluInfoHost_t;
      tol : double;
      position : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:199
   pragma Import (C, cusolverSpDcsrluZeroPivotHost, "cusolverSpDcsrluZeroPivotHost");

   function cusolverSpCcsrluZeroPivotHost
     (handle : cusolverSp_h.cusolverSpHandle_t;
      info : csrluInfoHost_t;
      tol : float;
      position : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:205
   pragma Import (C, cusolverSpCcsrluZeroPivotHost, "cusolverSpCcsrluZeroPivotHost");

   function cusolverSpZcsrluZeroPivotHost
     (handle : cusolverSp_h.cusolverSpHandle_t;
      info : csrluInfoHost_t;
      tol : double;
      position : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:211
   pragma Import (C, cusolverSpZcsrluZeroPivotHost, "cusolverSpZcsrluZeroPivotHost");

   function cusolverSpScsrluSolveHost
     (handle : cusolverSp_h.cusolverSpHandle_t;
      n : int;
      b : access float;
      x : access float;
      info : csrluInfoHost_t;
      pBuffer : System.Address) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:218
   pragma Import (C, cusolverSpScsrluSolveHost, "cusolverSpScsrluSolveHost");

   function cusolverSpDcsrluSolveHost
     (handle : cusolverSp_h.cusolverSpHandle_t;
      n : int;
      b : access double;
      x : access double;
      info : csrluInfoHost_t;
      pBuffer : System.Address) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:226
   pragma Import (C, cusolverSpDcsrluSolveHost, "cusolverSpDcsrluSolveHost");

   function cusolverSpCcsrluSolveHost
     (handle : cusolverSp_h.cusolverSpHandle_t;
      n : int;
      b : access constant cuComplex_h.cuComplex;
      x : access cuComplex_h.cuComplex;
      info : csrluInfoHost_t;
      pBuffer : System.Address) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:234
   pragma Import (C, cusolverSpCcsrluSolveHost, "cusolverSpCcsrluSolveHost");

   function cusolverSpZcsrluSolveHost
     (handle : cusolverSp_h.cusolverSpHandle_t;
      n : int;
      b : access constant cuComplex_h.cuDoubleComplex;
      x : access cuComplex_h.cuDoubleComplex;
      info : csrluInfoHost_t;
      pBuffer : System.Address) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:242
   pragma Import (C, cusolverSpZcsrluSolveHost, "cusolverSpZcsrluSolveHost");

   function cusolverSpXcsrluNnzHost
     (handle : cusolverSp_h.cusolverSpHandle_t;
      nnzLRef : access int;
      nnzURef : access int;
      info : csrluInfoHost_t) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:250
   pragma Import (C, cusolverSpXcsrluNnzHost, "cusolverSpXcsrluNnzHost");

   function cusolverSpScsrluExtractHost
     (handle : cusolverSp_h.cusolverSpHandle_t;
      P : access int;
      Q : access int;
      descrL : cusparse_h.cusparseMatDescr_t;
      csrValL : access float;
      csrRowPtrL : access int;
      csrColIndL : access int;
      descrU : cusparse_h.cusparseMatDescr_t;
      csrValU : access float;
      csrRowPtrU : access int;
      csrColIndU : access int;
      info : csrluInfoHost_t;
      pBuffer : System.Address) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:256
   pragma Import (C, cusolverSpScsrluExtractHost, "cusolverSpScsrluExtractHost");

   function cusolverSpDcsrluExtractHost
     (handle : cusolverSp_h.cusolverSpHandle_t;
      P : access int;
      Q : access int;
      descrL : cusparse_h.cusparseMatDescr_t;
      csrValL : access double;
      csrRowPtrL : access int;
      csrColIndL : access int;
      descrU : cusparse_h.cusparseMatDescr_t;
      csrValU : access double;
      csrRowPtrU : access int;
      csrColIndU : access int;
      info : csrluInfoHost_t;
      pBuffer : System.Address) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:271
   pragma Import (C, cusolverSpDcsrluExtractHost, "cusolverSpDcsrluExtractHost");

   function cusolverSpCcsrluExtractHost
     (handle : cusolverSp_h.cusolverSpHandle_t;
      P : access int;
      Q : access int;
      descrL : cusparse_h.cusparseMatDescr_t;
      csrValL : access cuComplex_h.cuComplex;
      csrRowPtrL : access int;
      csrColIndL : access int;
      descrU : cusparse_h.cusparseMatDescr_t;
      csrValU : access cuComplex_h.cuComplex;
      csrRowPtrU : access int;
      csrColIndU : access int;
      info : csrluInfoHost_t;
      pBuffer : System.Address) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:286
   pragma Import (C, cusolverSpCcsrluExtractHost, "cusolverSpCcsrluExtractHost");

   function cusolverSpZcsrluExtractHost
     (handle : cusolverSp_h.cusolverSpHandle_t;
      P : access int;
      Q : access int;
      descrL : cusparse_h.cusparseMatDescr_t;
      csrValL : access cuComplex_h.cuDoubleComplex;
      csrRowPtrL : access int;
      csrColIndL : access int;
      descrU : cusparse_h.cusparseMatDescr_t;
      csrValU : access cuComplex_h.cuDoubleComplex;
      csrRowPtrU : access int;
      csrColIndU : access int;
      info : csrluInfoHost_t;
      pBuffer : System.Address) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:301
   pragma Import (C, cusolverSpZcsrluExtractHost, "cusolverSpZcsrluExtractHost");

  -- * Low level API for CPU QR
  -- *
  --  

   function cusolverSpCreateCsrqrInfoHost (info : System.Address) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:321
   pragma Import (C, cusolverSpCreateCsrqrInfoHost, "cusolverSpCreateCsrqrInfoHost");

   function cusolverSpDestroyCsrqrInfoHost (info : csrqrInfoHost_t) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:324
   pragma Import (C, cusolverSpDestroyCsrqrInfoHost, "cusolverSpDestroyCsrqrInfoHost");

   function cusolverSpXcsrqrAnalysisHost
     (handle : cusolverSp_h.cusolverSpHandle_t;
      m : int;
      n : int;
      nnzA : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrRowPtrA : access int;
      csrColIndA : access int;
      info : csrqrInfoHost_t) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:327
   pragma Import (C, cusolverSpXcsrqrAnalysisHost, "cusolverSpXcsrqrAnalysisHost");

   function cusolverSpScsrqrBufferInfoHost
     (handle : cusolverSp_h.cusolverSpHandle_t;
      m : int;
      n : int;
      nnzA : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrValA : access float;
      csrRowPtrA : access int;
      csrColIndA : access int;
      info : csrqrInfoHost_t;
      internalDataInBytes : access stddef_h.size_t;
      workspaceInBytes : access stddef_h.size_t) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:337
   pragma Import (C, cusolverSpScsrqrBufferInfoHost, "cusolverSpScsrqrBufferInfoHost");

   function cusolverSpDcsrqrBufferInfoHost
     (handle : cusolverSp_h.cusolverSpHandle_t;
      m : int;
      n : int;
      nnzA : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrValA : access double;
      csrRowPtrA : access int;
      csrColIndA : access int;
      info : csrqrInfoHost_t;
      internalDataInBytes : access stddef_h.size_t;
      workspaceInBytes : access stddef_h.size_t) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:350
   pragma Import (C, cusolverSpDcsrqrBufferInfoHost, "cusolverSpDcsrqrBufferInfoHost");

   function cusolverSpCcsrqrBufferInfoHost
     (handle : cusolverSp_h.cusolverSpHandle_t;
      m : int;
      n : int;
      nnzA : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrValA : access constant cuComplex_h.cuComplex;
      csrRowPtrA : access int;
      csrColIndA : access int;
      info : csrqrInfoHost_t;
      internalDataInBytes : access stddef_h.size_t;
      workspaceInBytes : access stddef_h.size_t) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:363
   pragma Import (C, cusolverSpCcsrqrBufferInfoHost, "cusolverSpCcsrqrBufferInfoHost");

   function cusolverSpZcsrqrBufferInfoHost
     (handle : cusolverSp_h.cusolverSpHandle_t;
      m : int;
      n : int;
      nnzA : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrValA : access constant cuComplex_h.cuDoubleComplex;
      csrRowPtrA : access int;
      csrColIndA : access int;
      info : csrqrInfoHost_t;
      internalDataInBytes : access stddef_h.size_t;
      workspaceInBytes : access stddef_h.size_t) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:376
   pragma Import (C, cusolverSpZcsrqrBufferInfoHost, "cusolverSpZcsrqrBufferInfoHost");

   function cusolverSpScsrqrSetupHost
     (handle : cusolverSp_h.cusolverSpHandle_t;
      m : int;
      n : int;
      nnzA : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrValA : access float;
      csrRowPtrA : access int;
      csrColIndA : access int;
      mu : float;
      info : csrqrInfoHost_t) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:389
   pragma Import (C, cusolverSpScsrqrSetupHost, "cusolverSpScsrqrSetupHost");

   function cusolverSpDcsrqrSetupHost
     (handle : cusolverSp_h.cusolverSpHandle_t;
      m : int;
      n : int;
      nnzA : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrValA : access double;
      csrRowPtrA : access int;
      csrColIndA : access int;
      mu : double;
      info : csrqrInfoHost_t) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:401
   pragma Import (C, cusolverSpDcsrqrSetupHost, "cusolverSpDcsrqrSetupHost");

   function cusolverSpCcsrqrSetupHost
     (handle : cusolverSp_h.cusolverSpHandle_t;
      m : int;
      n : int;
      nnzA : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrValA : access constant cuComplex_h.cuComplex;
      csrRowPtrA : access int;
      csrColIndA : access int;
      mu : cuComplex_h.cuComplex;
      info : csrqrInfoHost_t) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:413
   pragma Import (C, cusolverSpCcsrqrSetupHost, "cusolverSpCcsrqrSetupHost");

   function cusolverSpZcsrqrSetupHost
     (handle : cusolverSp_h.cusolverSpHandle_t;
      m : int;
      n : int;
      nnzA : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrValA : access constant cuComplex_h.cuDoubleComplex;
      csrRowPtrA : access int;
      csrColIndA : access int;
      mu : cuComplex_h.cuDoubleComplex;
      info : csrqrInfoHost_t) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:425
   pragma Import (C, cusolverSpZcsrqrSetupHost, "cusolverSpZcsrqrSetupHost");

   function cusolverSpScsrqrFactorHost
     (handle : cusolverSp_h.cusolverSpHandle_t;
      m : int;
      n : int;
      nnzA : int;
      b : access float;
      x : access float;
      info : csrqrInfoHost_t;
      pBuffer : System.Address) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:437
   pragma Import (C, cusolverSpScsrqrFactorHost, "cusolverSpScsrqrFactorHost");

   function cusolverSpDcsrqrFactorHost
     (handle : cusolverSp_h.cusolverSpHandle_t;
      m : int;
      n : int;
      nnzA : int;
      b : access double;
      x : access double;
      info : csrqrInfoHost_t;
      pBuffer : System.Address) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:447
   pragma Import (C, cusolverSpDcsrqrFactorHost, "cusolverSpDcsrqrFactorHost");

   function cusolverSpCcsrqrFactorHost
     (handle : cusolverSp_h.cusolverSpHandle_t;
      m : int;
      n : int;
      nnzA : int;
      b : access cuComplex_h.cuComplex;
      x : access cuComplex_h.cuComplex;
      info : csrqrInfoHost_t;
      pBuffer : System.Address) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:457
   pragma Import (C, cusolverSpCcsrqrFactorHost, "cusolverSpCcsrqrFactorHost");

   function cusolverSpZcsrqrFactorHost
     (handle : cusolverSp_h.cusolverSpHandle_t;
      m : int;
      n : int;
      nnzA : int;
      b : access cuComplex_h.cuDoubleComplex;
      x : access cuComplex_h.cuDoubleComplex;
      info : csrqrInfoHost_t;
      pBuffer : System.Address) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:467
   pragma Import (C, cusolverSpZcsrqrFactorHost, "cusolverSpZcsrqrFactorHost");

   function cusolverSpScsrqrZeroPivotHost
     (handle : cusolverSp_h.cusolverSpHandle_t;
      info : csrqrInfoHost_t;
      tol : float;
      position : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:478
   pragma Import (C, cusolverSpScsrqrZeroPivotHost, "cusolverSpScsrqrZeroPivotHost");

   function cusolverSpDcsrqrZeroPivotHost
     (handle : cusolverSp_h.cusolverSpHandle_t;
      info : csrqrInfoHost_t;
      tol : double;
      position : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:484
   pragma Import (C, cusolverSpDcsrqrZeroPivotHost, "cusolverSpDcsrqrZeroPivotHost");

   function cusolverSpCcsrqrZeroPivotHost
     (handle : cusolverSp_h.cusolverSpHandle_t;
      info : csrqrInfoHost_t;
      tol : float;
      position : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:490
   pragma Import (C, cusolverSpCcsrqrZeroPivotHost, "cusolverSpCcsrqrZeroPivotHost");

   function cusolverSpZcsrqrZeroPivotHost
     (handle : cusolverSp_h.cusolverSpHandle_t;
      info : csrqrInfoHost_t;
      tol : double;
      position : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:496
   pragma Import (C, cusolverSpZcsrqrZeroPivotHost, "cusolverSpZcsrqrZeroPivotHost");

   function cusolverSpScsrqrSolveHost
     (handle : cusolverSp_h.cusolverSpHandle_t;
      m : int;
      n : int;
      b : access float;
      x : access float;
      info : csrqrInfoHost_t;
      pBuffer : System.Address) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:503
   pragma Import (C, cusolverSpScsrqrSolveHost, "cusolverSpScsrqrSolveHost");

   function cusolverSpDcsrqrSolveHost
     (handle : cusolverSp_h.cusolverSpHandle_t;
      m : int;
      n : int;
      b : access double;
      x : access double;
      info : csrqrInfoHost_t;
      pBuffer : System.Address) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:512
   pragma Import (C, cusolverSpDcsrqrSolveHost, "cusolverSpDcsrqrSolveHost");

   function cusolverSpCcsrqrSolveHost
     (handle : cusolverSp_h.cusolverSpHandle_t;
      m : int;
      n : int;
      b : access cuComplex_h.cuComplex;
      x : access cuComplex_h.cuComplex;
      info : csrqrInfoHost_t;
      pBuffer : System.Address) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:521
   pragma Import (C, cusolverSpCcsrqrSolveHost, "cusolverSpCcsrqrSolveHost");

   function cusolverSpZcsrqrSolveHost
     (handle : cusolverSp_h.cusolverSpHandle_t;
      m : int;
      n : int;
      b : access cuComplex_h.cuDoubleComplex;
      x : access cuComplex_h.cuDoubleComplex;
      info : csrqrInfoHost_t;
      pBuffer : System.Address) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:530
   pragma Import (C, cusolverSpZcsrqrSolveHost, "cusolverSpZcsrqrSolveHost");

  -- * Low level API for GPU QR
  -- *
  --  

   function cusolverSpXcsrqrAnalysis
     (handle : cusolverSp_h.cusolverSpHandle_t;
      m : int;
      n : int;
      nnzA : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrRowPtrA : access int;
      csrColIndA : access int;
      info : cusolverSp_h.csrqrInfo_t) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:544
   pragma Import (C, cusolverSpXcsrqrAnalysis, "cusolverSpXcsrqrAnalysis");

   function cusolverSpScsrqrBufferInfo
     (handle : cusolverSp_h.cusolverSpHandle_t;
      m : int;
      n : int;
      nnzA : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrValA : access float;
      csrRowPtrA : access int;
      csrColIndA : access int;
      info : cusolverSp_h.csrqrInfo_t;
      internalDataInBytes : access stddef_h.size_t;
      workspaceInBytes : access stddef_h.size_t) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:554
   pragma Import (C, cusolverSpScsrqrBufferInfo, "cusolverSpScsrqrBufferInfo");

   function cusolverSpDcsrqrBufferInfo
     (handle : cusolverSp_h.cusolverSpHandle_t;
      m : int;
      n : int;
      nnzA : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrValA : access double;
      csrRowPtrA : access int;
      csrColIndA : access int;
      info : cusolverSp_h.csrqrInfo_t;
      internalDataInBytes : access stddef_h.size_t;
      workspaceInBytes : access stddef_h.size_t) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:567
   pragma Import (C, cusolverSpDcsrqrBufferInfo, "cusolverSpDcsrqrBufferInfo");

   function cusolverSpCcsrqrBufferInfo
     (handle : cusolverSp_h.cusolverSpHandle_t;
      m : int;
      n : int;
      nnzA : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrValA : access constant cuComplex_h.cuComplex;
      csrRowPtrA : access int;
      csrColIndA : access int;
      info : cusolverSp_h.csrqrInfo_t;
      internalDataInBytes : access stddef_h.size_t;
      workspaceInBytes : access stddef_h.size_t) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:580
   pragma Import (C, cusolverSpCcsrqrBufferInfo, "cusolverSpCcsrqrBufferInfo");

   function cusolverSpZcsrqrBufferInfo
     (handle : cusolverSp_h.cusolverSpHandle_t;
      m : int;
      n : int;
      nnzA : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrValA : access constant cuComplex_h.cuDoubleComplex;
      csrRowPtrA : access int;
      csrColIndA : access int;
      info : cusolverSp_h.csrqrInfo_t;
      internalDataInBytes : access stddef_h.size_t;
      workspaceInBytes : access stddef_h.size_t) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:593
   pragma Import (C, cusolverSpZcsrqrBufferInfo, "cusolverSpZcsrqrBufferInfo");

   function cusolverSpScsrqrSetup
     (handle : cusolverSp_h.cusolverSpHandle_t;
      m : int;
      n : int;
      nnzA : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrValA : access float;
      csrRowPtrA : access int;
      csrColIndA : access int;
      mu : float;
      info : cusolverSp_h.csrqrInfo_t) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:606
   pragma Import (C, cusolverSpScsrqrSetup, "cusolverSpScsrqrSetup");

   function cusolverSpDcsrqrSetup
     (handle : cusolverSp_h.cusolverSpHandle_t;
      m : int;
      n : int;
      nnzA : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrValA : access double;
      csrRowPtrA : access int;
      csrColIndA : access int;
      mu : double;
      info : cusolverSp_h.csrqrInfo_t) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:618
   pragma Import (C, cusolverSpDcsrqrSetup, "cusolverSpDcsrqrSetup");

   function cusolverSpCcsrqrSetup
     (handle : cusolverSp_h.cusolverSpHandle_t;
      m : int;
      n : int;
      nnzA : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrValA : access constant cuComplex_h.cuComplex;
      csrRowPtrA : access int;
      csrColIndA : access int;
      mu : cuComplex_h.cuComplex;
      info : cusolverSp_h.csrqrInfo_t) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:630
   pragma Import (C, cusolverSpCcsrqrSetup, "cusolverSpCcsrqrSetup");

   function cusolverSpZcsrqrSetup
     (handle : cusolverSp_h.cusolverSpHandle_t;
      m : int;
      n : int;
      nnzA : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrValA : access constant cuComplex_h.cuDoubleComplex;
      csrRowPtrA : access int;
      csrColIndA : access int;
      mu : cuComplex_h.cuDoubleComplex;
      info : cusolverSp_h.csrqrInfo_t) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:642
   pragma Import (C, cusolverSpZcsrqrSetup, "cusolverSpZcsrqrSetup");

   function cusolverSpScsrqrFactor
     (handle : cusolverSp_h.cusolverSpHandle_t;
      m : int;
      n : int;
      nnzA : int;
      b : access float;
      x : access float;
      info : cusolverSp_h.csrqrInfo_t;
      pBuffer : System.Address) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:655
   pragma Import (C, cusolverSpScsrqrFactor, "cusolverSpScsrqrFactor");

   function cusolverSpDcsrqrFactor
     (handle : cusolverSp_h.cusolverSpHandle_t;
      m : int;
      n : int;
      nnzA : int;
      b : access double;
      x : access double;
      info : cusolverSp_h.csrqrInfo_t;
      pBuffer : System.Address) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:665
   pragma Import (C, cusolverSpDcsrqrFactor, "cusolverSpDcsrqrFactor");

   function cusolverSpCcsrqrFactor
     (handle : cusolverSp_h.cusolverSpHandle_t;
      m : int;
      n : int;
      nnzA : int;
      b : access cuComplex_h.cuComplex;
      x : access cuComplex_h.cuComplex;
      info : cusolverSp_h.csrqrInfo_t;
      pBuffer : System.Address) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:675
   pragma Import (C, cusolverSpCcsrqrFactor, "cusolverSpCcsrqrFactor");

   function cusolverSpZcsrqrFactor
     (handle : cusolverSp_h.cusolverSpHandle_t;
      m : int;
      n : int;
      nnzA : int;
      b : access cuComplex_h.cuDoubleComplex;
      x : access cuComplex_h.cuDoubleComplex;
      info : cusolverSp_h.csrqrInfo_t;
      pBuffer : System.Address) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:685
   pragma Import (C, cusolverSpZcsrqrFactor, "cusolverSpZcsrqrFactor");

   function cusolverSpScsrqrZeroPivot
     (handle : cusolverSp_h.cusolverSpHandle_t;
      info : cusolverSp_h.csrqrInfo_t;
      tol : float;
      position : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:696
   pragma Import (C, cusolverSpScsrqrZeroPivot, "cusolverSpScsrqrZeroPivot");

   function cusolverSpDcsrqrZeroPivot
     (handle : cusolverSp_h.cusolverSpHandle_t;
      info : cusolverSp_h.csrqrInfo_t;
      tol : double;
      position : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:702
   pragma Import (C, cusolverSpDcsrqrZeroPivot, "cusolverSpDcsrqrZeroPivot");

   function cusolverSpCcsrqrZeroPivot
     (handle : cusolverSp_h.cusolverSpHandle_t;
      info : cusolverSp_h.csrqrInfo_t;
      tol : float;
      position : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:708
   pragma Import (C, cusolverSpCcsrqrZeroPivot, "cusolverSpCcsrqrZeroPivot");

   function cusolverSpZcsrqrZeroPivot
     (handle : cusolverSp_h.cusolverSpHandle_t;
      info : cusolverSp_h.csrqrInfo_t;
      tol : double;
      position : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:714
   pragma Import (C, cusolverSpZcsrqrZeroPivot, "cusolverSpZcsrqrZeroPivot");

   function cusolverSpScsrqrSolve
     (handle : cusolverSp_h.cusolverSpHandle_t;
      m : int;
      n : int;
      b : access float;
      x : access float;
      info : cusolverSp_h.csrqrInfo_t;
      pBuffer : System.Address) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:721
   pragma Import (C, cusolverSpScsrqrSolve, "cusolverSpScsrqrSolve");

   function cusolverSpDcsrqrSolve
     (handle : cusolverSp_h.cusolverSpHandle_t;
      m : int;
      n : int;
      b : access double;
      x : access double;
      info : cusolverSp_h.csrqrInfo_t;
      pBuffer : System.Address) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:730
   pragma Import (C, cusolverSpDcsrqrSolve, "cusolverSpDcsrqrSolve");

   function cusolverSpCcsrqrSolve
     (handle : cusolverSp_h.cusolverSpHandle_t;
      m : int;
      n : int;
      b : access cuComplex_h.cuComplex;
      x : access cuComplex_h.cuComplex;
      info : cusolverSp_h.csrqrInfo_t;
      pBuffer : System.Address) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:739
   pragma Import (C, cusolverSpCcsrqrSolve, "cusolverSpCcsrqrSolve");

   function cusolverSpZcsrqrSolve
     (handle : cusolverSp_h.cusolverSpHandle_t;
      m : int;
      n : int;
      b : access cuComplex_h.cuDoubleComplex;
      x : access cuComplex_h.cuDoubleComplex;
      info : cusolverSp_h.csrqrInfo_t;
      pBuffer : System.Address) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:748
   pragma Import (C, cusolverSpZcsrqrSolve, "cusolverSpZcsrqrSolve");

  -- * Low level API for CPU Cholesky
  -- * 
  --  

   function cusolverSpCreateCsrcholInfoHost (info : System.Address) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:762
   pragma Import (C, cusolverSpCreateCsrcholInfoHost, "cusolverSpCreateCsrcholInfoHost");

   function cusolverSpDestroyCsrcholInfoHost (info : csrcholInfoHost_t) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:765
   pragma Import (C, cusolverSpDestroyCsrcholInfoHost, "cusolverSpDestroyCsrcholInfoHost");

   function cusolverSpXcsrcholAnalysisHost
     (handle : cusolverSp_h.cusolverSpHandle_t;
      n : int;
      nnzA : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrRowPtrA : access int;
      csrColIndA : access int;
      info : csrcholInfoHost_t) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:768
   pragma Import (C, cusolverSpXcsrcholAnalysisHost, "cusolverSpXcsrcholAnalysisHost");

   function cusolverSpScsrcholBufferInfoHost
     (handle : cusolverSp_h.cusolverSpHandle_t;
      n : int;
      nnzA : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrValA : access float;
      csrRowPtrA : access int;
      csrColIndA : access int;
      info : csrcholInfoHost_t;
      internalDataInBytes : access stddef_h.size_t;
      workspaceInBytes : access stddef_h.size_t) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:777
   pragma Import (C, cusolverSpScsrcholBufferInfoHost, "cusolverSpScsrcholBufferInfoHost");

   function cusolverSpDcsrcholBufferInfoHost
     (handle : cusolverSp_h.cusolverSpHandle_t;
      n : int;
      nnzA : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrValA : access double;
      csrRowPtrA : access int;
      csrColIndA : access int;
      info : csrcholInfoHost_t;
      internalDataInBytes : access stddef_h.size_t;
      workspaceInBytes : access stddef_h.size_t) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:789
   pragma Import (C, cusolverSpDcsrcholBufferInfoHost, "cusolverSpDcsrcholBufferInfoHost");

   function cusolverSpCcsrcholBufferInfoHost
     (handle : cusolverSp_h.cusolverSpHandle_t;
      n : int;
      nnzA : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrValA : access constant cuComplex_h.cuComplex;
      csrRowPtrA : access int;
      csrColIndA : access int;
      info : csrcholInfoHost_t;
      internalDataInBytes : access stddef_h.size_t;
      workspaceInBytes : access stddef_h.size_t) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:801
   pragma Import (C, cusolverSpCcsrcholBufferInfoHost, "cusolverSpCcsrcholBufferInfoHost");

   function cusolverSpZcsrcholBufferInfoHost
     (handle : cusolverSp_h.cusolverSpHandle_t;
      n : int;
      nnzA : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrValA : access constant cuComplex_h.cuDoubleComplex;
      csrRowPtrA : access int;
      csrColIndA : access int;
      info : csrcholInfoHost_t;
      internalDataInBytes : access stddef_h.size_t;
      workspaceInBytes : access stddef_h.size_t) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:813
   pragma Import (C, cusolverSpZcsrcholBufferInfoHost, "cusolverSpZcsrcholBufferInfoHost");

   function cusolverSpScsrcholFactorHost
     (handle : cusolverSp_h.cusolverSpHandle_t;
      n : int;
      nnzA : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrValA : access float;
      csrRowPtrA : access int;
      csrColIndA : access int;
      info : csrcholInfoHost_t;
      pBuffer : System.Address) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:826
   pragma Import (C, cusolverSpScsrcholFactorHost, "cusolverSpScsrcholFactorHost");

   function cusolverSpDcsrcholFactorHost
     (handle : cusolverSp_h.cusolverSpHandle_t;
      n : int;
      nnzA : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrValA : access double;
      csrRowPtrA : access int;
      csrColIndA : access int;
      info : csrcholInfoHost_t;
      pBuffer : System.Address) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:837
   pragma Import (C, cusolverSpDcsrcholFactorHost, "cusolverSpDcsrcholFactorHost");

   function cusolverSpCcsrcholFactorHost
     (handle : cusolverSp_h.cusolverSpHandle_t;
      n : int;
      nnzA : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrValA : access constant cuComplex_h.cuComplex;
      csrRowPtrA : access int;
      csrColIndA : access int;
      info : csrcholInfoHost_t;
      pBuffer : System.Address) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:848
   pragma Import (C, cusolverSpCcsrcholFactorHost, "cusolverSpCcsrcholFactorHost");

   function cusolverSpZcsrcholFactorHost
     (handle : cusolverSp_h.cusolverSpHandle_t;
      n : int;
      nnzA : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrValA : access constant cuComplex_h.cuDoubleComplex;
      csrRowPtrA : access int;
      csrColIndA : access int;
      info : csrcholInfoHost_t;
      pBuffer : System.Address) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:859
   pragma Import (C, cusolverSpZcsrcholFactorHost, "cusolverSpZcsrcholFactorHost");

   function cusolverSpScsrcholZeroPivotHost
     (handle : cusolverSp_h.cusolverSpHandle_t;
      info : csrcholInfoHost_t;
      tol : float;
      position : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:870
   pragma Import (C, cusolverSpScsrcholZeroPivotHost, "cusolverSpScsrcholZeroPivotHost");

   function cusolverSpDcsrcholZeroPivotHost
     (handle : cusolverSp_h.cusolverSpHandle_t;
      info : csrcholInfoHost_t;
      tol : double;
      position : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:876
   pragma Import (C, cusolverSpDcsrcholZeroPivotHost, "cusolverSpDcsrcholZeroPivotHost");

   function cusolverSpCcsrcholZeroPivotHost
     (handle : cusolverSp_h.cusolverSpHandle_t;
      info : csrcholInfoHost_t;
      tol : float;
      position : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:882
   pragma Import (C, cusolverSpCcsrcholZeroPivotHost, "cusolverSpCcsrcholZeroPivotHost");

   function cusolverSpZcsrcholZeroPivotHost
     (handle : cusolverSp_h.cusolverSpHandle_t;
      info : csrcholInfoHost_t;
      tol : double;
      position : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:888
   pragma Import (C, cusolverSpZcsrcholZeroPivotHost, "cusolverSpZcsrcholZeroPivotHost");

   function cusolverSpScsrcholSolveHost
     (handle : cusolverSp_h.cusolverSpHandle_t;
      n : int;
      b : access float;
      x : access float;
      info : csrcholInfoHost_t;
      pBuffer : System.Address) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:894
   pragma Import (C, cusolverSpScsrcholSolveHost, "cusolverSpScsrcholSolveHost");

   function cusolverSpDcsrcholSolveHost
     (handle : cusolverSp_h.cusolverSpHandle_t;
      n : int;
      b : access double;
      x : access double;
      info : csrcholInfoHost_t;
      pBuffer : System.Address) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:902
   pragma Import (C, cusolverSpDcsrcholSolveHost, "cusolverSpDcsrcholSolveHost");

   function cusolverSpCcsrcholSolveHost
     (handle : cusolverSp_h.cusolverSpHandle_t;
      n : int;
      b : access constant cuComplex_h.cuComplex;
      x : access cuComplex_h.cuComplex;
      info : csrcholInfoHost_t;
      pBuffer : System.Address) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:910
   pragma Import (C, cusolverSpCcsrcholSolveHost, "cusolverSpCcsrcholSolveHost");

   function cusolverSpZcsrcholSolveHost
     (handle : cusolverSp_h.cusolverSpHandle_t;
      n : int;
      b : access constant cuComplex_h.cuDoubleComplex;
      x : access cuComplex_h.cuDoubleComplex;
      info : csrcholInfoHost_t;
      pBuffer : System.Address) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:918
   pragma Import (C, cusolverSpZcsrcholSolveHost, "cusolverSpZcsrcholSolveHost");

  -- * Low level API for GPU Cholesky
  -- * 
  --  

   function cusolverSpCreateCsrcholInfo (info : System.Address) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:930
   pragma Import (C, cusolverSpCreateCsrcholInfo, "cusolverSpCreateCsrcholInfo");

   function cusolverSpDestroyCsrcholInfo (info : csrcholInfo_t) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:933
   pragma Import (C, cusolverSpDestroyCsrcholInfo, "cusolverSpDestroyCsrcholInfo");

   function cusolverSpXcsrcholAnalysis
     (handle : cusolverSp_h.cusolverSpHandle_t;
      n : int;
      nnzA : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrRowPtrA : access int;
      csrColIndA : access int;
      info : csrcholInfo_t) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:936
   pragma Import (C, cusolverSpXcsrcholAnalysis, "cusolverSpXcsrcholAnalysis");

   function cusolverSpScsrcholBufferInfo
     (handle : cusolverSp_h.cusolverSpHandle_t;
      n : int;
      nnzA : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrValA : access float;
      csrRowPtrA : access int;
      csrColIndA : access int;
      info : csrcholInfo_t;
      internalDataInBytes : access stddef_h.size_t;
      workspaceInBytes : access stddef_h.size_t) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:945
   pragma Import (C, cusolverSpScsrcholBufferInfo, "cusolverSpScsrcholBufferInfo");

   function cusolverSpDcsrcholBufferInfo
     (handle : cusolverSp_h.cusolverSpHandle_t;
      n : int;
      nnzA : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrValA : access double;
      csrRowPtrA : access int;
      csrColIndA : access int;
      info : csrcholInfo_t;
      internalDataInBytes : access stddef_h.size_t;
      workspaceInBytes : access stddef_h.size_t) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:957
   pragma Import (C, cusolverSpDcsrcholBufferInfo, "cusolverSpDcsrcholBufferInfo");

   function cusolverSpCcsrcholBufferInfo
     (handle : cusolverSp_h.cusolverSpHandle_t;
      n : int;
      nnzA : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrValA : access constant cuComplex_h.cuComplex;
      csrRowPtrA : access int;
      csrColIndA : access int;
      info : csrcholInfo_t;
      internalDataInBytes : access stddef_h.size_t;
      workspaceInBytes : access stddef_h.size_t) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:969
   pragma Import (C, cusolverSpCcsrcholBufferInfo, "cusolverSpCcsrcholBufferInfo");

   function cusolverSpZcsrcholBufferInfo
     (handle : cusolverSp_h.cusolverSpHandle_t;
      n : int;
      nnzA : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrValA : access constant cuComplex_h.cuDoubleComplex;
      csrRowPtrA : access int;
      csrColIndA : access int;
      info : csrcholInfo_t;
      internalDataInBytes : access stddef_h.size_t;
      workspaceInBytes : access stddef_h.size_t) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:981
   pragma Import (C, cusolverSpZcsrcholBufferInfo, "cusolverSpZcsrcholBufferInfo");

   function cusolverSpScsrcholFactor
     (handle : cusolverSp_h.cusolverSpHandle_t;
      n : int;
      nnzA : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrValA : access float;
      csrRowPtrA : access int;
      csrColIndA : access int;
      info : csrcholInfo_t;
      pBuffer : System.Address) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:993
   pragma Import (C, cusolverSpScsrcholFactor, "cusolverSpScsrcholFactor");

   function cusolverSpDcsrcholFactor
     (handle : cusolverSp_h.cusolverSpHandle_t;
      n : int;
      nnzA : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrValA : access double;
      csrRowPtrA : access int;
      csrColIndA : access int;
      info : csrcholInfo_t;
      pBuffer : System.Address) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:1004
   pragma Import (C, cusolverSpDcsrcholFactor, "cusolverSpDcsrcholFactor");

   function cusolverSpCcsrcholFactor
     (handle : cusolverSp_h.cusolverSpHandle_t;
      n : int;
      nnzA : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrValA : access constant cuComplex_h.cuComplex;
      csrRowPtrA : access int;
      csrColIndA : access int;
      info : csrcholInfo_t;
      pBuffer : System.Address) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:1015
   pragma Import (C, cusolverSpCcsrcholFactor, "cusolverSpCcsrcholFactor");

   function cusolverSpZcsrcholFactor
     (handle : cusolverSp_h.cusolverSpHandle_t;
      n : int;
      nnzA : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrValA : access constant cuComplex_h.cuDoubleComplex;
      csrRowPtrA : access int;
      csrColIndA : access int;
      info : csrcholInfo_t;
      pBuffer : System.Address) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:1026
   pragma Import (C, cusolverSpZcsrcholFactor, "cusolverSpZcsrcholFactor");

   function cusolverSpScsrcholZeroPivot
     (handle : cusolverSp_h.cusolverSpHandle_t;
      info : csrcholInfo_t;
      tol : float;
      position : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:1037
   pragma Import (C, cusolverSpScsrcholZeroPivot, "cusolverSpScsrcholZeroPivot");

   function cusolverSpDcsrcholZeroPivot
     (handle : cusolverSp_h.cusolverSpHandle_t;
      info : csrcholInfo_t;
      tol : double;
      position : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:1043
   pragma Import (C, cusolverSpDcsrcholZeroPivot, "cusolverSpDcsrcholZeroPivot");

   function cusolverSpCcsrcholZeroPivot
     (handle : cusolverSp_h.cusolverSpHandle_t;
      info : csrcholInfo_t;
      tol : float;
      position : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:1049
   pragma Import (C, cusolverSpCcsrcholZeroPivot, "cusolverSpCcsrcholZeroPivot");

   function cusolverSpZcsrcholZeroPivot
     (handle : cusolverSp_h.cusolverSpHandle_t;
      info : csrcholInfo_t;
      tol : double;
      position : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:1055
   pragma Import (C, cusolverSpZcsrcholZeroPivot, "cusolverSpZcsrcholZeroPivot");

   function cusolverSpScsrcholSolve
     (handle : cusolverSp_h.cusolverSpHandle_t;
      n : int;
      b : access float;
      x : access float;
      info : csrcholInfo_t;
      pBuffer : System.Address) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:1061
   pragma Import (C, cusolverSpScsrcholSolve, "cusolverSpScsrcholSolve");

   function cusolverSpDcsrcholSolve
     (handle : cusolverSp_h.cusolverSpHandle_t;
      n : int;
      b : access double;
      x : access double;
      info : csrcholInfo_t;
      pBuffer : System.Address) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:1069
   pragma Import (C, cusolverSpDcsrcholSolve, "cusolverSpDcsrcholSolve");

   function cusolverSpCcsrcholSolve
     (handle : cusolverSp_h.cusolverSpHandle_t;
      n : int;
      b : access constant cuComplex_h.cuComplex;
      x : access cuComplex_h.cuComplex;
      info : csrcholInfo_t;
      pBuffer : System.Address) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:1077
   pragma Import (C, cusolverSpCcsrcholSolve, "cusolverSpCcsrcholSolve");

   function cusolverSpZcsrcholSolve
     (handle : cusolverSp_h.cusolverSpHandle_t;
      n : int;
      b : access constant cuComplex_h.cuDoubleComplex;
      x : access cuComplex_h.cuDoubleComplex;
      info : csrcholInfo_t;
      pBuffer : System.Address) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp_LOWLEVEL_PREVIEW.h:1085
   pragma Import (C, cusolverSpZcsrcholSolve, "cusolverSpZcsrcholSolve");

end cusolverSp_LOWLEVEL_PREVIEW_h;

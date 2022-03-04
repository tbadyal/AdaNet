pragma Ada_2005;
pragma Style_Checks (Off);

with Interfaces.C; use Interfaces.C;
with System;
with library_types_h;
with driver_types_h;
with cuComplex_h;
with stddef_h;

package cusparse_h is

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

  -- import complex data type  
  -- CUSPARSE status type returns  
   type cusparseStatus_t is 
     (CUSPARSE_STATUS_SUCCESS,
      CUSPARSE_STATUS_NOT_INITIALIZED,
      CUSPARSE_STATUS_ALLOC_FAILED,
      CUSPARSE_STATUS_INVALID_VALUE,
      CUSPARSE_STATUS_ARCH_MISMATCH,
      CUSPARSE_STATUS_MAPPING_ERROR,
      CUSPARSE_STATUS_EXECUTION_FAILED,
      CUSPARSE_STATUS_INTERNAL_ERROR,
      CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED,
      CUSPARSE_STATUS_ZERO_PIVOT);
   pragma Convention (C, cusparseStatus_t);  -- /usr/local/cuda-8.0/include/cusparse.h:83

  -- Opaque structure holding CUSPARSE library context  
   --  skipped empty struct cusparseContext

   type cusparseHandle_t is new System.Address;  -- /usr/local/cuda-8.0/include/cusparse.h:87

  -- Opaque structure holding the matrix descriptor  
   --  skipped empty struct cusparseMatDescr

   type cusparseMatDescr_t is new System.Address;  -- /usr/local/cuda-8.0/include/cusparse.h:91

  -- Opaque structure holding the sparse triangular solve information  
   --  skipped empty struct cusparseSolveAnalysisInfo

   type cusparseSolveAnalysisInfo_t is new System.Address;  -- /usr/local/cuda-8.0/include/cusparse.h:95

  -- Opaque structures holding the sparse triangular solve information  
   --  skipped empty struct csrsv2Info

   type csrsv2Info_t is new System.Address;  -- /usr/local/cuda-8.0/include/cusparse.h:99

   --  skipped empty struct bsrsv2Info

   type bsrsv2Info_t is new System.Address;  -- /usr/local/cuda-8.0/include/cusparse.h:102

   --  skipped empty struct bsrsm2Info

   type bsrsm2Info_t is new System.Address;  -- /usr/local/cuda-8.0/include/cusparse.h:105

  -- Opaque structures holding incomplete Cholesky information  
   --  skipped empty struct csric02Info

   type csric02Info_t is new System.Address;  -- /usr/local/cuda-8.0/include/cusparse.h:109

   --  skipped empty struct bsric02Info

   type bsric02Info_t is new System.Address;  -- /usr/local/cuda-8.0/include/cusparse.h:112

  -- Opaque structures holding incomplete LU information  
   --  skipped empty struct csrilu02Info

   type csrilu02Info_t is new System.Address;  -- /usr/local/cuda-8.0/include/cusparse.h:116

   --  skipped empty struct bsrilu02Info

   type bsrilu02Info_t is new System.Address;  -- /usr/local/cuda-8.0/include/cusparse.h:119

  -- Opaque structures holding the hybrid (HYB) storage information  
   --  skipped empty struct cusparseHybMat

   type cusparseHybMat_t is new System.Address;  -- /usr/local/cuda-8.0/include/cusparse.h:123

  -- Opaque structures holding sparse gemm information  
   --  skipped empty struct csrgemm2Info

   type csrgemm2Info_t is new System.Address;  -- /usr/local/cuda-8.0/include/cusparse.h:127

  -- Opaque structure holding the sorting information  
   --  skipped empty struct csru2csrInfo

   type csru2csrInfo_t is new System.Address;  -- /usr/local/cuda-8.0/include/cusparse.h:131

  -- Opaque structure holding the coloring information  
   --  skipped empty struct cusparseColorInfo

   type cusparseColorInfo_t is new System.Address;  -- /usr/local/cuda-8.0/include/cusparse.h:135

  -- Types definitions  
   type cusparsePointerMode_t is 
     (CUSPARSE_POINTER_MODE_HOST,
      CUSPARSE_POINTER_MODE_DEVICE);
   pragma Convention (C, cusparsePointerMode_t);  -- /usr/local/cuda-8.0/include/cusparse.h:141

   type cusparseAction_t is 
     (CUSPARSE_ACTION_SYMBOLIC,
      CUSPARSE_ACTION_NUMERIC);
   pragma Convention (C, cusparseAction_t);  -- /usr/local/cuda-8.0/include/cusparse.h:146

   type cusparseMatrixType_t is 
     (CUSPARSE_MATRIX_TYPE_GENERAL,
      CUSPARSE_MATRIX_TYPE_SYMMETRIC,
      CUSPARSE_MATRIX_TYPE_HERMITIAN,
      CUSPARSE_MATRIX_TYPE_TRIANGULAR);
   pragma Convention (C, cusparseMatrixType_t);  -- /usr/local/cuda-8.0/include/cusparse.h:153

   type cusparseFillMode_t is 
     (CUSPARSE_FILL_MODE_LOWER,
      CUSPARSE_FILL_MODE_UPPER);
   pragma Convention (C, cusparseFillMode_t);  -- /usr/local/cuda-8.0/include/cusparse.h:158

   type cusparseDiagType_t is 
     (CUSPARSE_DIAG_TYPE_NON_UNIT,
      CUSPARSE_DIAG_TYPE_UNIT);
   pragma Convention (C, cusparseDiagType_t);  -- /usr/local/cuda-8.0/include/cusparse.h:163

   type cusparseIndexBase_t is 
     (CUSPARSE_INDEX_BASE_ZERO,
      CUSPARSE_INDEX_BASE_ONE);
   pragma Convention (C, cusparseIndexBase_t);  -- /usr/local/cuda-8.0/include/cusparse.h:168

   type cusparseOperation_t is 
     (CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_TRANSPOSE,
      CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE);
   pragma Convention (C, cusparseOperation_t);  -- /usr/local/cuda-8.0/include/cusparse.h:174

   type cusparseDirection_t is 
     (CUSPARSE_DIRECTION_ROW,
      CUSPARSE_DIRECTION_COLUMN);
   pragma Convention (C, cusparseDirection_t);  -- /usr/local/cuda-8.0/include/cusparse.h:179

  -- automatically decide how to split the data into regular/irregular part
  -- store data into regular part up to a user specified treshhold
  -- store all data in the regular part
   type cusparseHybPartition_t is 
     (CUSPARSE_HYB_PARTITION_AUTO,
      CUSPARSE_HYB_PARTITION_USER,
      CUSPARSE_HYB_PARTITION_MAX);
   pragma Convention (C, cusparseHybPartition_t);  -- /usr/local/cuda-8.0/include/cusparse.h:185

  -- used in csrsv2, csric02, and csrilu02
  -- no level information is generated, only reports structural zero.
   type cusparseSolvePolicy_t is 
     (CUSPARSE_SOLVE_POLICY_NO_LEVEL,
      CUSPARSE_SOLVE_POLICY_USE_LEVEL);
   pragma Convention (C, cusparseSolvePolicy_t);  -- /usr/local/cuda-8.0/include/cusparse.h:191

   type cusparseSideMode_t is 
     (CUSPARSE_SIDE_LEFT,
      CUSPARSE_SIDE_RIGHT);
   pragma Convention (C, cusparseSideMode_t);  -- /usr/local/cuda-8.0/include/cusparse.h:196

  -- default
   type cusparseColorAlg_t is 
     (CUSPARSE_COLOR_ALG0,
      CUSPARSE_COLOR_ALG1);
   pragma Convention (C, cusparseColorAlg_t);  -- /usr/local/cuda-8.0/include/cusparse.h:201

  --default, naive
  --merge path
   type cusparseAlgMode_t is 
     (CUSPARSE_ALG0,
      CUSPARSE_ALG1);
   pragma Convention (C, cusparseAlgMode_t);  -- /usr/local/cuda-8.0/include/cusparse.h:206

  -- CUSPARSE initialization and managment routines  
   function cusparseCreate (handle : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:209
   pragma Import (C, cusparseCreate, "cusparseCreate");

   function cusparseDestroy (handle : cusparseHandle_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:210
   pragma Import (C, cusparseDestroy, "cusparseDestroy");

   function cusparseGetVersion (handle : cusparseHandle_t; version : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:211
   pragma Import (C, cusparseGetVersion, "cusparseGetVersion");

   function cusparseGetProperty (c_type : library_types_h.libraryPropertyType; value : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:212
   pragma Import (C, cusparseGetProperty, "cusparseGetProperty");

   function cusparseSetStream (handle : cusparseHandle_t; streamId : driver_types_h.cudaStream_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:213
   pragma Import (C, cusparseSetStream, "cusparseSetStream");

   function cusparseGetStream (handle : cusparseHandle_t; streamId : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:214
   pragma Import (C, cusparseGetStream, "cusparseGetStream");

  -- CUSPARSE type creation, destruction, set and get routines  
   function cusparseGetPointerMode (handle : cusparseHandle_t; mode : access cusparsePointerMode_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:218
   pragma Import (C, cusparseGetPointerMode, "cusparseGetPointerMode");

   function cusparseSetPointerMode (handle : cusparseHandle_t; mode : cusparsePointerMode_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:219
   pragma Import (C, cusparseSetPointerMode, "cusparseSetPointerMode");

  -- sparse matrix descriptor  
  -- When the matrix descriptor is created, its fields are initialized to: 
  --   CUSPARSE_MATRIX_TYPE_GENERAL
  --   CUSPARSE_INDEX_BASE_ZERO
  --   All other fields are uninitialized
  -- 

   function cusparseCreateMatDescr (descrA : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:227
   pragma Import (C, cusparseCreateMatDescr, "cusparseCreateMatDescr");

   function cusparseDestroyMatDescr (descrA : cusparseMatDescr_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:228
   pragma Import (C, cusparseDestroyMatDescr, "cusparseDestroyMatDescr");

   function cusparseCopyMatDescr (dest : cusparseMatDescr_t; src : cusparseMatDescr_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:230
   pragma Import (C, cusparseCopyMatDescr, "cusparseCopyMatDescr");

   function cusparseSetMatType (descrA : cusparseMatDescr_t; c_type : cusparseMatrixType_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:232
   pragma Import (C, cusparseSetMatType, "cusparseSetMatType");

   function cusparseGetMatType (descrA : cusparseMatDescr_t) return cusparseMatrixType_t;  -- /usr/local/cuda-8.0/include/cusparse.h:233
   pragma Import (C, cusparseGetMatType, "cusparseGetMatType");

   function cusparseSetMatFillMode (descrA : cusparseMatDescr_t; fillMode : cusparseFillMode_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:235
   pragma Import (C, cusparseSetMatFillMode, "cusparseSetMatFillMode");

   function cusparseGetMatFillMode (descrA : cusparseMatDescr_t) return cusparseFillMode_t;  -- /usr/local/cuda-8.0/include/cusparse.h:236
   pragma Import (C, cusparseGetMatFillMode, "cusparseGetMatFillMode");

   function cusparseSetMatDiagType (descrA : cusparseMatDescr_t; diagType : cusparseDiagType_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:238
   pragma Import (C, cusparseSetMatDiagType, "cusparseSetMatDiagType");

   function cusparseGetMatDiagType (descrA : cusparseMatDescr_t) return cusparseDiagType_t;  -- /usr/local/cuda-8.0/include/cusparse.h:239
   pragma Import (C, cusparseGetMatDiagType, "cusparseGetMatDiagType");

   function cusparseSetMatIndexBase (descrA : cusparseMatDescr_t; base : cusparseIndexBase_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:241
   pragma Import (C, cusparseSetMatIndexBase, "cusparseSetMatIndexBase");

   function cusparseGetMatIndexBase (descrA : cusparseMatDescr_t) return cusparseIndexBase_t;  -- /usr/local/cuda-8.0/include/cusparse.h:242
   pragma Import (C, cusparseGetMatIndexBase, "cusparseGetMatIndexBase");

  -- sparse triangular solve and incomplete-LU and Cholesky (algorithm 1)  
   function cusparseCreateSolveAnalysisInfo (info : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:245
   pragma Import (C, cusparseCreateSolveAnalysisInfo, "cusparseCreateSolveAnalysisInfo");

   function cusparseDestroySolveAnalysisInfo (info : cusparseSolveAnalysisInfo_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:246
   pragma Import (C, cusparseDestroySolveAnalysisInfo, "cusparseDestroySolveAnalysisInfo");

   function cusparseGetLevelInfo
     (handle : cusparseHandle_t;
      info : cusparseSolveAnalysisInfo_t;
      nlevels : access int;
      levelPtr : System.Address;
      levelInd : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:247
   pragma Import (C, cusparseGetLevelInfo, "cusparseGetLevelInfo");

  -- sparse triangular solve (algorithm 2)  
   function cusparseCreateCsrsv2Info (info : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:254
   pragma Import (C, cusparseCreateCsrsv2Info, "cusparseCreateCsrsv2Info");

   function cusparseDestroyCsrsv2Info (info : csrsv2Info_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:255
   pragma Import (C, cusparseDestroyCsrsv2Info, "cusparseDestroyCsrsv2Info");

  -- incomplete Cholesky (algorithm 2) 
   function cusparseCreateCsric02Info (info : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:258
   pragma Import (C, cusparseCreateCsric02Info, "cusparseCreateCsric02Info");

   function cusparseDestroyCsric02Info (info : csric02Info_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:259
   pragma Import (C, cusparseDestroyCsric02Info, "cusparseDestroyCsric02Info");

   function cusparseCreateBsric02Info (info : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:261
   pragma Import (C, cusparseCreateBsric02Info, "cusparseCreateBsric02Info");

   function cusparseDestroyBsric02Info (info : bsric02Info_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:262
   pragma Import (C, cusparseDestroyBsric02Info, "cusparseDestroyBsric02Info");

  -- incomplete LU (algorithm 2)  
   function cusparseCreateCsrilu02Info (info : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:265
   pragma Import (C, cusparseCreateCsrilu02Info, "cusparseCreateCsrilu02Info");

   function cusparseDestroyCsrilu02Info (info : csrilu02Info_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:266
   pragma Import (C, cusparseDestroyCsrilu02Info, "cusparseDestroyCsrilu02Info");

   function cusparseCreateBsrilu02Info (info : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:268
   pragma Import (C, cusparseCreateBsrilu02Info, "cusparseCreateBsrilu02Info");

   function cusparseDestroyBsrilu02Info (info : bsrilu02Info_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:269
   pragma Import (C, cusparseDestroyBsrilu02Info, "cusparseDestroyBsrilu02Info");

  -- block-CSR triangular solve (algorithm 2)  
   function cusparseCreateBsrsv2Info (info : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:272
   pragma Import (C, cusparseCreateBsrsv2Info, "cusparseCreateBsrsv2Info");

   function cusparseDestroyBsrsv2Info (info : bsrsv2Info_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:273
   pragma Import (C, cusparseDestroyBsrsv2Info, "cusparseDestroyBsrsv2Info");

   function cusparseCreateBsrsm2Info (info : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:275
   pragma Import (C, cusparseCreateBsrsm2Info, "cusparseCreateBsrsm2Info");

   function cusparseDestroyBsrsm2Info (info : bsrsm2Info_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:276
   pragma Import (C, cusparseDestroyBsrsm2Info, "cusparseDestroyBsrsm2Info");

  -- hybrid (HYB) format  
   function cusparseCreateHybMat (hybA : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:279
   pragma Import (C, cusparseCreateHybMat, "cusparseCreateHybMat");

   function cusparseDestroyHybMat (hybA : cusparseHybMat_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:280
   pragma Import (C, cusparseDestroyHybMat, "cusparseDestroyHybMat");

  -- sorting information  
   function cusparseCreateCsru2csrInfo (info : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:283
   pragma Import (C, cusparseCreateCsru2csrInfo, "cusparseCreateCsru2csrInfo");

   function cusparseDestroyCsru2csrInfo (info : csru2csrInfo_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:284
   pragma Import (C, cusparseDestroyCsru2csrInfo, "cusparseDestroyCsru2csrInfo");

  -- coloring info  
   function cusparseCreateColorInfo (info : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:287
   pragma Import (C, cusparseCreateColorInfo, "cusparseCreateColorInfo");

   function cusparseDestroyColorInfo (info : cusparseColorInfo_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:288
   pragma Import (C, cusparseDestroyColorInfo, "cusparseDestroyColorInfo");

   function cusparseSetColorAlgs (info : cusparseColorInfo_t; alg : cusparseColorAlg_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:290
   pragma Import (C, cusparseSetColorAlgs, "cusparseSetColorAlgs");

   function cusparseGetColorAlgs (info : cusparseColorInfo_t; alg : access cusparseColorAlg_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:291
   pragma Import (C, cusparseGetColorAlgs, "cusparseGetColorAlgs");

  -- --- Sparse Level 1 routines ---  
  -- Description: Addition of a scalar multiple of a sparse vector x  
  --   and a dense vector y.  

   function cusparseSaxpyi
     (handle : cusparseHandle_t;
      nnz : int;
      alpha : access float;
      xVal : access float;
      xInd : access int;
      y : access float;
      idxBase : cusparseIndexBase_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:297
   pragma Import (C, cusparseSaxpyi, "cusparseSaxpyi");

   function cusparseDaxpyi
     (handle : cusparseHandle_t;
      nnz : int;
      alpha : access double;
      xVal : access double;
      xInd : access int;
      y : access double;
      idxBase : cusparseIndexBase_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:305
   pragma Import (C, cusparseDaxpyi, "cusparseDaxpyi");

   function cusparseCaxpyi
     (handle : cusparseHandle_t;
      nnz : int;
      alpha : access constant cuComplex_h.cuComplex;
      xVal : access constant cuComplex_h.cuComplex;
      xInd : access int;
      y : access cuComplex_h.cuComplex;
      idxBase : cusparseIndexBase_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:313
   pragma Import (C, cusparseCaxpyi, "cusparseCaxpyi");

   function cusparseZaxpyi
     (handle : cusparseHandle_t;
      nnz : int;
      alpha : access constant cuComplex_h.cuDoubleComplex;
      xVal : access constant cuComplex_h.cuDoubleComplex;
      xInd : access int;
      y : access cuComplex_h.cuDoubleComplex;
      idxBase : cusparseIndexBase_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:321
   pragma Import (C, cusparseZaxpyi, "cusparseZaxpyi");

  -- Description: dot product of a sparse vector x and a dense vector y.  
   function cusparseSdoti
     (handle : cusparseHandle_t;
      nnz : int;
      xVal : access float;
      xInd : access int;
      y : access float;
      resultDevHostPtr : access float;
      idxBase : cusparseIndexBase_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:330
   pragma Import (C, cusparseSdoti, "cusparseSdoti");

   function cusparseDdoti
     (handle : cusparseHandle_t;
      nnz : int;
      xVal : access double;
      xInd : access int;
      y : access double;
      resultDevHostPtr : access double;
      idxBase : cusparseIndexBase_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:338
   pragma Import (C, cusparseDdoti, "cusparseDdoti");

   function cusparseCdoti
     (handle : cusparseHandle_t;
      nnz : int;
      xVal : access constant cuComplex_h.cuComplex;
      xInd : access int;
      y : access constant cuComplex_h.cuComplex;
      resultDevHostPtr : access cuComplex_h.cuComplex;
      idxBase : cusparseIndexBase_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:346
   pragma Import (C, cusparseCdoti, "cusparseCdoti");

   function cusparseZdoti
     (handle : cusparseHandle_t;
      nnz : int;
      xVal : access constant cuComplex_h.cuDoubleComplex;
      xInd : access int;
      y : access constant cuComplex_h.cuDoubleComplex;
      resultDevHostPtr : access cuComplex_h.cuDoubleComplex;
      idxBase : cusparseIndexBase_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:354
   pragma Import (C, cusparseZdoti, "cusparseZdoti");

  -- Description: dot product of complex conjugate of a sparse vector x
  --   and a dense vector y.  

   function cusparseCdotci
     (handle : cusparseHandle_t;
      nnz : int;
      xVal : access constant cuComplex_h.cuComplex;
      xInd : access int;
      y : access constant cuComplex_h.cuComplex;
      resultDevHostPtr : access cuComplex_h.cuComplex;
      idxBase : cusparseIndexBase_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:364
   pragma Import (C, cusparseCdotci, "cusparseCdotci");

   function cusparseZdotci
     (handle : cusparseHandle_t;
      nnz : int;
      xVal : access constant cuComplex_h.cuDoubleComplex;
      xInd : access int;
      y : access constant cuComplex_h.cuDoubleComplex;
      resultDevHostPtr : access cuComplex_h.cuDoubleComplex;
      idxBase : cusparseIndexBase_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:372
   pragma Import (C, cusparseZdotci, "cusparseZdotci");

  -- Description: Gather of non-zero elements from dense vector y into 
  --   sparse vector x.  

   function cusparseSgthr
     (handle : cusparseHandle_t;
      nnz : int;
      y : access float;
      xVal : access float;
      xInd : access int;
      idxBase : cusparseIndexBase_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:383
   pragma Import (C, cusparseSgthr, "cusparseSgthr");

   function cusparseDgthr
     (handle : cusparseHandle_t;
      nnz : int;
      y : access double;
      xVal : access double;
      xInd : access int;
      idxBase : cusparseIndexBase_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:390
   pragma Import (C, cusparseDgthr, "cusparseDgthr");

   function cusparseCgthr
     (handle : cusparseHandle_t;
      nnz : int;
      y : access constant cuComplex_h.cuComplex;
      xVal : access cuComplex_h.cuComplex;
      xInd : access int;
      idxBase : cusparseIndexBase_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:397
   pragma Import (C, cusparseCgthr, "cusparseCgthr");

   function cusparseZgthr
     (handle : cusparseHandle_t;
      nnz : int;
      y : access constant cuComplex_h.cuDoubleComplex;
      xVal : access cuComplex_h.cuDoubleComplex;
      xInd : access int;
      idxBase : cusparseIndexBase_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:404
   pragma Import (C, cusparseZgthr, "cusparseZgthr");

  -- Description: Gather of non-zero elements from desne vector y into 
  --   sparse vector x (also replacing these elements in y by zeros).  

   function cusparseSgthrz
     (handle : cusparseHandle_t;
      nnz : int;
      y : access float;
      xVal : access float;
      xInd : access int;
      idxBase : cusparseIndexBase_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:413
   pragma Import (C, cusparseSgthrz, "cusparseSgthrz");

   function cusparseDgthrz
     (handle : cusparseHandle_t;
      nnz : int;
      y : access double;
      xVal : access double;
      xInd : access int;
      idxBase : cusparseIndexBase_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:420
   pragma Import (C, cusparseDgthrz, "cusparseDgthrz");

   function cusparseCgthrz
     (handle : cusparseHandle_t;
      nnz : int;
      y : access cuComplex_h.cuComplex;
      xVal : access cuComplex_h.cuComplex;
      xInd : access int;
      idxBase : cusparseIndexBase_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:427
   pragma Import (C, cusparseCgthrz, "cusparseCgthrz");

   function cusparseZgthrz
     (handle : cusparseHandle_t;
      nnz : int;
      y : access cuComplex_h.cuDoubleComplex;
      xVal : access cuComplex_h.cuDoubleComplex;
      xInd : access int;
      idxBase : cusparseIndexBase_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:434
   pragma Import (C, cusparseZgthrz, "cusparseZgthrz");

  -- Description: Scatter of elements of the sparse vector x into 
  --   dense vector y.  

   function cusparseSsctr
     (handle : cusparseHandle_t;
      nnz : int;
      xVal : access float;
      xInd : access int;
      y : access float;
      idxBase : cusparseIndexBase_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:443
   pragma Import (C, cusparseSsctr, "cusparseSsctr");

   function cusparseDsctr
     (handle : cusparseHandle_t;
      nnz : int;
      xVal : access double;
      xInd : access int;
      y : access double;
      idxBase : cusparseIndexBase_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:450
   pragma Import (C, cusparseDsctr, "cusparseDsctr");

   function cusparseCsctr
     (handle : cusparseHandle_t;
      nnz : int;
      xVal : access constant cuComplex_h.cuComplex;
      xInd : access int;
      y : access cuComplex_h.cuComplex;
      idxBase : cusparseIndexBase_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:457
   pragma Import (C, cusparseCsctr, "cusparseCsctr");

   function cusparseZsctr
     (handle : cusparseHandle_t;
      nnz : int;
      xVal : access constant cuComplex_h.cuDoubleComplex;
      xInd : access int;
      y : access cuComplex_h.cuDoubleComplex;
      idxBase : cusparseIndexBase_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:464
   pragma Import (C, cusparseZsctr, "cusparseZsctr");

  -- Description: Givens rotation, where c and s are cosine and sine, 
  --   x and y are sparse and dense vectors, respectively.  

   function cusparseSroti
     (handle : cusparseHandle_t;
      nnz : int;
      xVal : access float;
      xInd : access int;
      y : access float;
      c : access float;
      s : access float;
      idxBase : cusparseIndexBase_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:473
   pragma Import (C, cusparseSroti, "cusparseSroti");

   function cusparseDroti
     (handle : cusparseHandle_t;
      nnz : int;
      xVal : access double;
      xInd : access int;
      y : access double;
      c : access double;
      s : access double;
      idxBase : cusparseIndexBase_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:482
   pragma Import (C, cusparseDroti, "cusparseDroti");

  -- --- Sparse Level 2 routines ---  
   function cusparseSgemvi
     (handle : cusparseHandle_t;
      transA : cusparseOperation_t;
      m : int;
      n : int;
      alpha : access float;
      A : access float;
      lda : int;
      nnz : int;
      xVal : access float;
      xInd : access int;
      beta : access float;
      y : access float;
      idxBase : cusparseIndexBase_t;
      pBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:494
   pragma Import (C, cusparseSgemvi, "cusparseSgemvi");

  -- host or device pointer  
  -- host or device pointer  
   function cusparseSgemvi_bufferSize
     (handle : cusparseHandle_t;
      transA : cusparseOperation_t;
      m : int;
      n : int;
      nnz : int;
      pBufferSize : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:509
   pragma Import (C, cusparseSgemvi_bufferSize, "cusparseSgemvi_bufferSize");

   function cusparseDgemvi
     (handle : cusparseHandle_t;
      transA : cusparseOperation_t;
      m : int;
      n : int;
      alpha : access double;
      A : access double;
      lda : int;
      nnz : int;
      xVal : access double;
      xInd : access int;
      beta : access double;
      y : access double;
      idxBase : cusparseIndexBase_t;
      pBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:516
   pragma Import (C, cusparseDgemvi, "cusparseDgemvi");

  -- host or device pointer  
  -- host or device pointer  
   function cusparseDgemvi_bufferSize
     (handle : cusparseHandle_t;
      transA : cusparseOperation_t;
      m : int;
      n : int;
      nnz : int;
      pBufferSize : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:531
   pragma Import (C, cusparseDgemvi_bufferSize, "cusparseDgemvi_bufferSize");

   function cusparseCgemvi
     (handle : cusparseHandle_t;
      transA : cusparseOperation_t;
      m : int;
      n : int;
      alpha : access constant cuComplex_h.cuComplex;
      A : access constant cuComplex_h.cuComplex;
      lda : int;
      nnz : int;
      xVal : access constant cuComplex_h.cuComplex;
      xInd : access int;
      beta : access constant cuComplex_h.cuComplex;
      y : access cuComplex_h.cuComplex;
      idxBase : cusparseIndexBase_t;
      pBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:538
   pragma Import (C, cusparseCgemvi, "cusparseCgemvi");

  -- host or device pointer  
  -- host or device pointer  
   function cusparseCgemvi_bufferSize
     (handle : cusparseHandle_t;
      transA : cusparseOperation_t;
      m : int;
      n : int;
      nnz : int;
      pBufferSize : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:553
   pragma Import (C, cusparseCgemvi_bufferSize, "cusparseCgemvi_bufferSize");

   function cusparseZgemvi
     (handle : cusparseHandle_t;
      transA : cusparseOperation_t;
      m : int;
      n : int;
      alpha : access constant cuComplex_h.cuDoubleComplex;
      A : access constant cuComplex_h.cuDoubleComplex;
      lda : int;
      nnz : int;
      xVal : access constant cuComplex_h.cuDoubleComplex;
      xInd : access int;
      beta : access constant cuComplex_h.cuDoubleComplex;
      y : access cuComplex_h.cuDoubleComplex;
      idxBase : cusparseIndexBase_t;
      pBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:560
   pragma Import (C, cusparseZgemvi, "cusparseZgemvi");

  -- host or device pointer  
  -- host or device pointer  
   function cusparseZgemvi_bufferSize
     (handle : cusparseHandle_t;
      transA : cusparseOperation_t;
      m : int;
      n : int;
      nnz : int;
      pBufferSize : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:575
   pragma Import (C, cusparseZgemvi_bufferSize, "cusparseZgemvi_bufferSize");

  -- Description: Matrix-vector multiplication  y = alpha * op(A) * x  + beta * y, 
  --   where A is a sparse matrix in CSR storage format, x and y are dense vectors.  

   function cusparseScsrmv
     (handle : cusparseHandle_t;
      transA : cusparseOperation_t;
      m : int;
      n : int;
      nnz : int;
      alpha : access float;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access float;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      x : access float;
      beta : access float;
      y : access float) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:585
   pragma Import (C, cusparseScsrmv, "cusparseScsrmv");

   function cusparseDcsrmv
     (handle : cusparseHandle_t;
      transA : cusparseOperation_t;
      m : int;
      n : int;
      nnz : int;
      alpha : access double;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access double;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      x : access double;
      beta : access double;
      y : access double) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:599
   pragma Import (C, cusparseDcsrmv, "cusparseDcsrmv");

   function cusparseCcsrmv
     (handle : cusparseHandle_t;
      transA : cusparseOperation_t;
      m : int;
      n : int;
      nnz : int;
      alpha : access constant cuComplex_h.cuComplex;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access constant cuComplex_h.cuComplex;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      x : access constant cuComplex_h.cuComplex;
      beta : access constant cuComplex_h.cuComplex;
      y : access cuComplex_h.cuComplex) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:613
   pragma Import (C, cusparseCcsrmv, "cusparseCcsrmv");

   function cusparseZcsrmv
     (handle : cusparseHandle_t;
      transA : cusparseOperation_t;
      m : int;
      n : int;
      nnz : int;
      alpha : access constant cuComplex_h.cuDoubleComplex;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access constant cuComplex_h.cuDoubleComplex;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      x : access constant cuComplex_h.cuDoubleComplex;
      beta : access constant cuComplex_h.cuDoubleComplex;
      y : access cuComplex_h.cuDoubleComplex) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:627
   pragma Import (C, cusparseZcsrmv, "cusparseZcsrmv");

  --Returns number of bytes
   function cusparseCsrmvEx_bufferSize
     (handle : cusparseHandle_t;
      alg : cusparseAlgMode_t;
      transA : cusparseOperation_t;
      m : int;
      n : int;
      nnz : int;
      alpha : System.Address;
      alphatype : library_types_h.cudaDataType;
      descrA : cusparseMatDescr_t;
      csrValA : System.Address;
      csrValAtype : library_types_h.cudaDataType;
      csrRowPtrA : access int;
      csrColIndA : access int;
      x : System.Address;
      xtype : library_types_h.cudaDataType;
      beta : System.Address;
      betatype : library_types_h.cudaDataType;
      y : System.Address;
      ytype : library_types_h.cudaDataType;
      executiontype : library_types_h.cudaDataType;
      bufferSizeInBytes : access stddef_h.size_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:642
   pragma Import (C, cusparseCsrmvEx_bufferSize, "cusparseCsrmvEx_bufferSize");

   function cusparseCsrmvEx
     (handle : cusparseHandle_t;
      alg : cusparseAlgMode_t;
      transA : cusparseOperation_t;
      m : int;
      n : int;
      nnz : int;
      alpha : System.Address;
      alphatype : library_types_h.cudaDataType;
      descrA : cusparseMatDescr_t;
      csrValA : System.Address;
      csrValAtype : library_types_h.cudaDataType;
      csrRowPtrA : access int;
      csrColIndA : access int;
      x : System.Address;
      xtype : library_types_h.cudaDataType;
      beta : System.Address;
      betatype : library_types_h.cudaDataType;
      y : System.Address;
      ytype : library_types_h.cudaDataType;
      executiontype : library_types_h.cudaDataType;
      buffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:664
   pragma Import (C, cusparseCsrmvEx, "cusparseCsrmvEx");

  -- Description: Matrix-vector multiplication  y = alpha * op(A) * x  + beta * y, 
  --   where A is a sparse matrix in CSR storage format, x and y are dense vectors
  --   using a Merge Path load-balancing implementation.  

   function cusparseScsrmv_mp
     (handle : cusparseHandle_t;
      transA : cusparseOperation_t;
      m : int;
      n : int;
      nnz : int;
      alpha : access float;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access float;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      x : access float;
      beta : access float;
      y : access float) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:689
   pragma Import (C, cusparseScsrmv_mp, "cusparseScsrmv_mp");

   function cusparseDcsrmv_mp
     (handle : cusparseHandle_t;
      transA : cusparseOperation_t;
      m : int;
      n : int;
      nnz : int;
      alpha : access double;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access double;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      x : access double;
      beta : access double;
      y : access double) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:703
   pragma Import (C, cusparseDcsrmv_mp, "cusparseDcsrmv_mp");

   function cusparseCcsrmv_mp
     (handle : cusparseHandle_t;
      transA : cusparseOperation_t;
      m : int;
      n : int;
      nnz : int;
      alpha : access constant cuComplex_h.cuComplex;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access constant cuComplex_h.cuComplex;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      x : access constant cuComplex_h.cuComplex;
      beta : access constant cuComplex_h.cuComplex;
      y : access cuComplex_h.cuComplex) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:717
   pragma Import (C, cusparseCcsrmv_mp, "cusparseCcsrmv_mp");

   function cusparseZcsrmv_mp
     (handle : cusparseHandle_t;
      transA : cusparseOperation_t;
      m : int;
      n : int;
      nnz : int;
      alpha : access constant cuComplex_h.cuDoubleComplex;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access constant cuComplex_h.cuDoubleComplex;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      x : access constant cuComplex_h.cuDoubleComplex;
      beta : access constant cuComplex_h.cuDoubleComplex;
      y : access cuComplex_h.cuDoubleComplex) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:731
   pragma Import (C, cusparseZcsrmv_mp, "cusparseZcsrmv_mp");

  -- Description: Matrix-vector multiplication  y = alpha * op(A) * x  + beta * y, 
  --   where A is a sparse matrix in HYB storage format, x and y are dense vectors.  

   function cusparseShybmv
     (handle : cusparseHandle_t;
      transA : cusparseOperation_t;
      alpha : access float;
      descrA : cusparseMatDescr_t;
      hybA : cusparseHybMat_t;
      x : access float;
      beta : access float;
      y : access float) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:748
   pragma Import (C, cusparseShybmv, "cusparseShybmv");

   function cusparseDhybmv
     (handle : cusparseHandle_t;
      transA : cusparseOperation_t;
      alpha : access double;
      descrA : cusparseMatDescr_t;
      hybA : cusparseHybMat_t;
      x : access double;
      beta : access double;
      y : access double) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:757
   pragma Import (C, cusparseDhybmv, "cusparseDhybmv");

   function cusparseChybmv
     (handle : cusparseHandle_t;
      transA : cusparseOperation_t;
      alpha : access constant cuComplex_h.cuComplex;
      descrA : cusparseMatDescr_t;
      hybA : cusparseHybMat_t;
      x : access constant cuComplex_h.cuComplex;
      beta : access constant cuComplex_h.cuComplex;
      y : access cuComplex_h.cuComplex) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:766
   pragma Import (C, cusparseChybmv, "cusparseChybmv");

   function cusparseZhybmv
     (handle : cusparseHandle_t;
      transA : cusparseOperation_t;
      alpha : access constant cuComplex_h.cuDoubleComplex;
      descrA : cusparseMatDescr_t;
      hybA : cusparseHybMat_t;
      x : access constant cuComplex_h.cuDoubleComplex;
      beta : access constant cuComplex_h.cuDoubleComplex;
      y : access cuComplex_h.cuDoubleComplex) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:775
   pragma Import (C, cusparseZhybmv, "cusparseZhybmv");

  -- Description: Matrix-vector multiplication  y = alpha * op(A) * x  + beta * y, 
  --   where A is a sparse matrix in BSR storage format, x and y are dense vectors.  

   function cusparseSbsrmv
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      transA : cusparseOperation_t;
      mb : int;
      nb : int;
      nnzb : int;
      alpha : access float;
      descrA : cusparseMatDescr_t;
      bsrSortedValA : access float;
      bsrSortedRowPtrA : access int;
      bsrSortedColIndA : access int;
      blockDim : int;
      x : access float;
      beta : access float;
      y : access float) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:786
   pragma Import (C, cusparseSbsrmv, "cusparseSbsrmv");

   function cusparseDbsrmv
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      transA : cusparseOperation_t;
      mb : int;
      nb : int;
      nnzb : int;
      alpha : access double;
      descrA : cusparseMatDescr_t;
      bsrSortedValA : access double;
      bsrSortedRowPtrA : access int;
      bsrSortedColIndA : access int;
      blockDim : int;
      x : access double;
      beta : access double;
      y : access double) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:802
   pragma Import (C, cusparseDbsrmv, "cusparseDbsrmv");

   function cusparseCbsrmv
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      transA : cusparseOperation_t;
      mb : int;
      nb : int;
      nnzb : int;
      alpha : access constant cuComplex_h.cuComplex;
      descrA : cusparseMatDescr_t;
      bsrSortedValA : access constant cuComplex_h.cuComplex;
      bsrSortedRowPtrA : access int;
      bsrSortedColIndA : access int;
      blockDim : int;
      x : access constant cuComplex_h.cuComplex;
      beta : access constant cuComplex_h.cuComplex;
      y : access cuComplex_h.cuComplex) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:818
   pragma Import (C, cusparseCbsrmv, "cusparseCbsrmv");

   function cusparseZbsrmv
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      transA : cusparseOperation_t;
      mb : int;
      nb : int;
      nnzb : int;
      alpha : access constant cuComplex_h.cuDoubleComplex;
      descrA : cusparseMatDescr_t;
      bsrSortedValA : access constant cuComplex_h.cuDoubleComplex;
      bsrSortedRowPtrA : access int;
      bsrSortedColIndA : access int;
      blockDim : int;
      x : access constant cuComplex_h.cuDoubleComplex;
      beta : access constant cuComplex_h.cuDoubleComplex;
      y : access cuComplex_h.cuDoubleComplex) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:834
   pragma Import (C, cusparseZbsrmv, "cusparseZbsrmv");

  -- Description: Matrix-vector multiplication  y = alpha * op(A) * x  + beta * y, 
  --   where A is a sparse matrix in extended BSR storage format, x and y are dense 
  --   vectors.  

   function cusparseSbsrxmv
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      transA : cusparseOperation_t;
      sizeOfMask : int;
      mb : int;
      nb : int;
      nnzb : int;
      alpha : access float;
      descrA : cusparseMatDescr_t;
      bsrSortedValA : access float;
      bsrSortedMaskPtrA : access int;
      bsrSortedRowPtrA : access int;
      bsrSortedEndPtrA : access int;
      bsrSortedColIndA : access int;
      blockDim : int;
      x : access float;
      beta : access float;
      y : access float) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:853
   pragma Import (C, cusparseSbsrxmv, "cusparseSbsrxmv");

   function cusparseDbsrxmv
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      transA : cusparseOperation_t;
      sizeOfMask : int;
      mb : int;
      nb : int;
      nnzb : int;
      alpha : access double;
      descrA : cusparseMatDescr_t;
      bsrSortedValA : access double;
      bsrSortedMaskPtrA : access int;
      bsrSortedRowPtrA : access int;
      bsrSortedEndPtrA : access int;
      bsrSortedColIndA : access int;
      blockDim : int;
      x : access double;
      beta : access double;
      y : access double) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:873
   pragma Import (C, cusparseDbsrxmv, "cusparseDbsrxmv");

   function cusparseCbsrxmv
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      transA : cusparseOperation_t;
      sizeOfMask : int;
      mb : int;
      nb : int;
      nnzb : int;
      alpha : access constant cuComplex_h.cuComplex;
      descrA : cusparseMatDescr_t;
      bsrSortedValA : access constant cuComplex_h.cuComplex;
      bsrSortedMaskPtrA : access int;
      bsrSortedRowPtrA : access int;
      bsrSortedEndPtrA : access int;
      bsrSortedColIndA : access int;
      blockDim : int;
      x : access constant cuComplex_h.cuComplex;
      beta : access constant cuComplex_h.cuComplex;
      y : access cuComplex_h.cuComplex) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:892
   pragma Import (C, cusparseCbsrxmv, "cusparseCbsrxmv");

   function cusparseZbsrxmv
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      transA : cusparseOperation_t;
      sizeOfMask : int;
      mb : int;
      nb : int;
      nnzb : int;
      alpha : access constant cuComplex_h.cuDoubleComplex;
      descrA : cusparseMatDescr_t;
      bsrSortedValA : access constant cuComplex_h.cuDoubleComplex;
      bsrSortedMaskPtrA : access int;
      bsrSortedRowPtrA : access int;
      bsrSortedEndPtrA : access int;
      bsrSortedColIndA : access int;
      blockDim : int;
      x : access constant cuComplex_h.cuDoubleComplex;
      beta : access constant cuComplex_h.cuDoubleComplex;
      y : access cuComplex_h.cuDoubleComplex) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:912
   pragma Import (C, cusparseZbsrxmv, "cusparseZbsrxmv");

  -- Description: Solution of triangular linear system op(A) * x = alpha * f, 
  --   where A is a sparse matrix in CSR storage format, rhs f and solution x 
  --   are dense vectors. This routine implements algorithm 1 for the solve.  

   function cusparseCsrsv_analysisEx
     (handle : cusparseHandle_t;
      transA : cusparseOperation_t;
      m : int;
      nnz : int;
      descrA : cusparseMatDescr_t;
      csrSortedValA : System.Address;
      csrSortedValAtype : library_types_h.cudaDataType;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      info : cusparseSolveAnalysisInfo_t;
      executiontype : library_types_h.cudaDataType) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:934
   pragma Import (C, cusparseCsrsv_analysisEx, "cusparseCsrsv_analysisEx");

   function cusparseScsrsv_analysis
     (handle : cusparseHandle_t;
      transA : cusparseOperation_t;
      m : int;
      nnz : int;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access float;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      info : cusparseSolveAnalysisInfo_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:946
   pragma Import (C, cusparseScsrsv_analysis, "cusparseScsrsv_analysis");

   function cusparseDcsrsv_analysis
     (handle : cusparseHandle_t;
      transA : cusparseOperation_t;
      m : int;
      nnz : int;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access double;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      info : cusparseSolveAnalysisInfo_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:956
   pragma Import (C, cusparseDcsrsv_analysis, "cusparseDcsrsv_analysis");

   function cusparseCcsrsv_analysis
     (handle : cusparseHandle_t;
      transA : cusparseOperation_t;
      m : int;
      nnz : int;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access constant cuComplex_h.cuComplex;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      info : cusparseSolveAnalysisInfo_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:966
   pragma Import (C, cusparseCcsrsv_analysis, "cusparseCcsrsv_analysis");

   function cusparseZcsrsv_analysis
     (handle : cusparseHandle_t;
      transA : cusparseOperation_t;
      m : int;
      nnz : int;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access constant cuComplex_h.cuDoubleComplex;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      info : cusparseSolveAnalysisInfo_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:976
   pragma Import (C, cusparseZcsrsv_analysis, "cusparseZcsrsv_analysis");

   function cusparseCsrsv_solveEx
     (handle : cusparseHandle_t;
      transA : cusparseOperation_t;
      m : int;
      alpha : System.Address;
      alphatype : library_types_h.cudaDataType;
      descrA : cusparseMatDescr_t;
      csrSortedValA : System.Address;
      csrSortedValAtype : library_types_h.cudaDataType;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      info : cusparseSolveAnalysisInfo_t;
      f : System.Address;
      ftype : library_types_h.cudaDataType;
      x : System.Address;
      xtype : library_types_h.cudaDataType;
      executiontype : library_types_h.cudaDataType) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:986
   pragma Import (C, cusparseCsrsv_solveEx, "cusparseCsrsv_solveEx");

   function cusparseScsrsv_solve
     (handle : cusparseHandle_t;
      transA : cusparseOperation_t;
      m : int;
      alpha : access float;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access float;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      info : cusparseSolveAnalysisInfo_t;
      f : access float;
      x : access float) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:1003
   pragma Import (C, cusparseScsrsv_solve, "cusparseScsrsv_solve");

   function cusparseDcsrsv_solve
     (handle : cusparseHandle_t;
      transA : cusparseOperation_t;
      m : int;
      alpha : access double;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access double;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      info : cusparseSolveAnalysisInfo_t;
      f : access double;
      x : access double) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:1015
   pragma Import (C, cusparseDcsrsv_solve, "cusparseDcsrsv_solve");

   function cusparseCcsrsv_solve
     (handle : cusparseHandle_t;
      transA : cusparseOperation_t;
      m : int;
      alpha : access constant cuComplex_h.cuComplex;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access constant cuComplex_h.cuComplex;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      info : cusparseSolveAnalysisInfo_t;
      f : access constant cuComplex_h.cuComplex;
      x : access cuComplex_h.cuComplex) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:1027
   pragma Import (C, cusparseCcsrsv_solve, "cusparseCcsrsv_solve");

   function cusparseZcsrsv_solve
     (handle : cusparseHandle_t;
      transA : cusparseOperation_t;
      m : int;
      alpha : access constant cuComplex_h.cuDoubleComplex;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access constant cuComplex_h.cuDoubleComplex;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      info : cusparseSolveAnalysisInfo_t;
      f : access constant cuComplex_h.cuDoubleComplex;
      x : access cuComplex_h.cuDoubleComplex) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:1039
   pragma Import (C, cusparseZcsrsv_solve, "cusparseZcsrsv_solve");

  -- Description: Solution of triangular linear system op(A) * x = alpha * f, 
  --   where A is a sparse matrix in CSR storage format, rhs f and solution y 
  --   are dense vectors. This routine implements algorithm 1 for this problem. 
  --   Also, it provides a utility function to query size of buffer used.  

   function cusparseXcsrsv2_zeroPivot
     (handle : cusparseHandle_t;
      info : csrsv2Info_t;
      position : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:1055
   pragma Import (C, cusparseXcsrsv2_zeroPivot, "cusparseXcsrsv2_zeroPivot");

   function cusparseScsrsv2_bufferSize
     (handle : cusparseHandle_t;
      transA : cusparseOperation_t;
      m : int;
      nnz : int;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access float;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      info : csrsv2Info_t;
      pBufferSizeInBytes : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:1059
   pragma Import (C, cusparseScsrsv2_bufferSize, "cusparseScsrsv2_bufferSize");

   function cusparseDcsrsv2_bufferSize
     (handle : cusparseHandle_t;
      transA : cusparseOperation_t;
      m : int;
      nnz : int;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access double;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      info : csrsv2Info_t;
      pBufferSizeInBytes : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:1070
   pragma Import (C, cusparseDcsrsv2_bufferSize, "cusparseDcsrsv2_bufferSize");

   function cusparseCcsrsv2_bufferSize
     (handle : cusparseHandle_t;
      transA : cusparseOperation_t;
      m : int;
      nnz : int;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access cuComplex_h.cuComplex;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      info : csrsv2Info_t;
      pBufferSizeInBytes : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:1081
   pragma Import (C, cusparseCcsrsv2_bufferSize, "cusparseCcsrsv2_bufferSize");

   function cusparseZcsrsv2_bufferSize
     (handle : cusparseHandle_t;
      transA : cusparseOperation_t;
      m : int;
      nnz : int;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access cuComplex_h.cuDoubleComplex;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      info : csrsv2Info_t;
      pBufferSizeInBytes : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:1092
   pragma Import (C, cusparseZcsrsv2_bufferSize, "cusparseZcsrsv2_bufferSize");

   function cusparseScsrsv2_bufferSizeExt
     (handle : cusparseHandle_t;
      transA : cusparseOperation_t;
      m : int;
      nnz : int;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access float;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      info : csrsv2Info_t;
      pBufferSize : access stddef_h.size_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:1103
   pragma Import (C, cusparseScsrsv2_bufferSizeExt, "cusparseScsrsv2_bufferSizeExt");

   function cusparseDcsrsv2_bufferSizeExt
     (handle : cusparseHandle_t;
      transA : cusparseOperation_t;
      m : int;
      nnz : int;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access double;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      info : csrsv2Info_t;
      pBufferSize : access stddef_h.size_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:1114
   pragma Import (C, cusparseDcsrsv2_bufferSizeExt, "cusparseDcsrsv2_bufferSizeExt");

   function cusparseCcsrsv2_bufferSizeExt
     (handle : cusparseHandle_t;
      transA : cusparseOperation_t;
      m : int;
      nnz : int;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access cuComplex_h.cuComplex;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      info : csrsv2Info_t;
      pBufferSize : access stddef_h.size_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:1125
   pragma Import (C, cusparseCcsrsv2_bufferSizeExt, "cusparseCcsrsv2_bufferSizeExt");

   function cusparseZcsrsv2_bufferSizeExt
     (handle : cusparseHandle_t;
      transA : cusparseOperation_t;
      m : int;
      nnz : int;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access cuComplex_h.cuDoubleComplex;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      info : csrsv2Info_t;
      pBufferSize : access stddef_h.size_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:1136
   pragma Import (C, cusparseZcsrsv2_bufferSizeExt, "cusparseZcsrsv2_bufferSizeExt");

   function cusparseScsrsv2_analysis
     (handle : cusparseHandle_t;
      transA : cusparseOperation_t;
      m : int;
      nnz : int;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access float;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      info : csrsv2Info_t;
      policy : cusparseSolvePolicy_t;
      pBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:1148
   pragma Import (C, cusparseScsrsv2_analysis, "cusparseScsrsv2_analysis");

   function cusparseDcsrsv2_analysis
     (handle : cusparseHandle_t;
      transA : cusparseOperation_t;
      m : int;
      nnz : int;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access double;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      info : csrsv2Info_t;
      policy : cusparseSolvePolicy_t;
      pBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:1160
   pragma Import (C, cusparseDcsrsv2_analysis, "cusparseDcsrsv2_analysis");

   function cusparseCcsrsv2_analysis
     (handle : cusparseHandle_t;
      transA : cusparseOperation_t;
      m : int;
      nnz : int;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access constant cuComplex_h.cuComplex;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      info : csrsv2Info_t;
      policy : cusparseSolvePolicy_t;
      pBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:1172
   pragma Import (C, cusparseCcsrsv2_analysis, "cusparseCcsrsv2_analysis");

   function cusparseZcsrsv2_analysis
     (handle : cusparseHandle_t;
      transA : cusparseOperation_t;
      m : int;
      nnz : int;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access constant cuComplex_h.cuDoubleComplex;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      info : csrsv2Info_t;
      policy : cusparseSolvePolicy_t;
      pBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:1184
   pragma Import (C, cusparseZcsrsv2_analysis, "cusparseZcsrsv2_analysis");

   function cusparseScsrsv2_solve
     (handle : cusparseHandle_t;
      transA : cusparseOperation_t;
      m : int;
      nnz : int;
      alpha : access float;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access float;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      info : csrsv2Info_t;
      f : access float;
      x : access float;
      policy : cusparseSolvePolicy_t;
      pBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:1196
   pragma Import (C, cusparseScsrsv2_solve, "cusparseScsrsv2_solve");

   function cusparseDcsrsv2_solve
     (handle : cusparseHandle_t;
      transA : cusparseOperation_t;
      m : int;
      nnz : int;
      alpha : access double;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access double;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      info : csrsv2Info_t;
      f : access double;
      x : access double;
      policy : cusparseSolvePolicy_t;
      pBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:1211
   pragma Import (C, cusparseDcsrsv2_solve, "cusparseDcsrsv2_solve");

   function cusparseCcsrsv2_solve
     (handle : cusparseHandle_t;
      transA : cusparseOperation_t;
      m : int;
      nnz : int;
      alpha : access constant cuComplex_h.cuComplex;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access constant cuComplex_h.cuComplex;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      info : csrsv2Info_t;
      f : access constant cuComplex_h.cuComplex;
      x : access cuComplex_h.cuComplex;
      policy : cusparseSolvePolicy_t;
      pBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:1226
   pragma Import (C, cusparseCcsrsv2_solve, "cusparseCcsrsv2_solve");

   function cusparseZcsrsv2_solve
     (handle : cusparseHandle_t;
      transA : cusparseOperation_t;
      m : int;
      nnz : int;
      alpha : access constant cuComplex_h.cuDoubleComplex;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access constant cuComplex_h.cuDoubleComplex;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      info : csrsv2Info_t;
      f : access constant cuComplex_h.cuDoubleComplex;
      x : access cuComplex_h.cuDoubleComplex;
      policy : cusparseSolvePolicy_t;
      pBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:1241
   pragma Import (C, cusparseZcsrsv2_solve, "cusparseZcsrsv2_solve");

  -- Description: Solution of triangular linear system op(A) * x = alpha * f, 
  --   where A is a sparse matrix in block-CSR storage format, rhs f and solution y 
  --   are dense vectors. This routine implements algorithm 2 for this problem. 
  --   Also, it provides a utility function to query size of buffer used.  

   function cusparseXbsrsv2_zeroPivot
     (handle : cusparseHandle_t;
      info : bsrsv2Info_t;
      position : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:1260
   pragma Import (C, cusparseXbsrsv2_zeroPivot, "cusparseXbsrsv2_zeroPivot");

   function cusparseSbsrsv2_bufferSize
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      transA : cusparseOperation_t;
      mb : int;
      nnzb : int;
      descrA : cusparseMatDescr_t;
      bsrSortedValA : access float;
      bsrSortedRowPtrA : access int;
      bsrSortedColIndA : access int;
      blockDim : int;
      info : bsrsv2Info_t;
      pBufferSizeInBytes : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:1265
   pragma Import (C, cusparseSbsrsv2_bufferSize, "cusparseSbsrsv2_bufferSize");

   function cusparseDbsrsv2_bufferSize
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      transA : cusparseOperation_t;
      mb : int;
      nnzb : int;
      descrA : cusparseMatDescr_t;
      bsrSortedValA : access double;
      bsrSortedRowPtrA : access int;
      bsrSortedColIndA : access int;
      blockDim : int;
      info : bsrsv2Info_t;
      pBufferSizeInBytes : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:1278
   pragma Import (C, cusparseDbsrsv2_bufferSize, "cusparseDbsrsv2_bufferSize");

   function cusparseCbsrsv2_bufferSize
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      transA : cusparseOperation_t;
      mb : int;
      nnzb : int;
      descrA : cusparseMatDescr_t;
      bsrSortedValA : access cuComplex_h.cuComplex;
      bsrSortedRowPtrA : access int;
      bsrSortedColIndA : access int;
      blockDim : int;
      info : bsrsv2Info_t;
      pBufferSizeInBytes : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:1291
   pragma Import (C, cusparseCbsrsv2_bufferSize, "cusparseCbsrsv2_bufferSize");

   function cusparseZbsrsv2_bufferSize
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      transA : cusparseOperation_t;
      mb : int;
      nnzb : int;
      descrA : cusparseMatDescr_t;
      bsrSortedValA : access cuComplex_h.cuDoubleComplex;
      bsrSortedRowPtrA : access int;
      bsrSortedColIndA : access int;
      blockDim : int;
      info : bsrsv2Info_t;
      pBufferSizeInBytes : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:1304
   pragma Import (C, cusparseZbsrsv2_bufferSize, "cusparseZbsrsv2_bufferSize");

   function cusparseSbsrsv2_bufferSizeExt
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      transA : cusparseOperation_t;
      mb : int;
      nnzb : int;
      descrA : cusparseMatDescr_t;
      bsrSortedValA : access float;
      bsrSortedRowPtrA : access int;
      bsrSortedColIndA : access int;
      blockSize : int;
      info : bsrsv2Info_t;
      pBufferSize : access stddef_h.size_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:1317
   pragma Import (C, cusparseSbsrsv2_bufferSizeExt, "cusparseSbsrsv2_bufferSizeExt");

   function cusparseDbsrsv2_bufferSizeExt
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      transA : cusparseOperation_t;
      mb : int;
      nnzb : int;
      descrA : cusparseMatDescr_t;
      bsrSortedValA : access double;
      bsrSortedRowPtrA : access int;
      bsrSortedColIndA : access int;
      blockSize : int;
      info : bsrsv2Info_t;
      pBufferSize : access stddef_h.size_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:1330
   pragma Import (C, cusparseDbsrsv2_bufferSizeExt, "cusparseDbsrsv2_bufferSizeExt");

   function cusparseCbsrsv2_bufferSizeExt
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      transA : cusparseOperation_t;
      mb : int;
      nnzb : int;
      descrA : cusparseMatDescr_t;
      bsrSortedValA : access cuComplex_h.cuComplex;
      bsrSortedRowPtrA : access int;
      bsrSortedColIndA : access int;
      blockSize : int;
      info : bsrsv2Info_t;
      pBufferSize : access stddef_h.size_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:1343
   pragma Import (C, cusparseCbsrsv2_bufferSizeExt, "cusparseCbsrsv2_bufferSizeExt");

   function cusparseZbsrsv2_bufferSizeExt
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      transA : cusparseOperation_t;
      mb : int;
      nnzb : int;
      descrA : cusparseMatDescr_t;
      bsrSortedValA : access cuComplex_h.cuDoubleComplex;
      bsrSortedRowPtrA : access int;
      bsrSortedColIndA : access int;
      blockSize : int;
      info : bsrsv2Info_t;
      pBufferSize : access stddef_h.size_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:1356
   pragma Import (C, cusparseZbsrsv2_bufferSizeExt, "cusparseZbsrsv2_bufferSizeExt");

   function cusparseSbsrsv2_analysis
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      transA : cusparseOperation_t;
      mb : int;
      nnzb : int;
      descrA : cusparseMatDescr_t;
      bsrSortedValA : access float;
      bsrSortedRowPtrA : access int;
      bsrSortedColIndA : access int;
      blockDim : int;
      info : bsrsv2Info_t;
      policy : cusparseSolvePolicy_t;
      pBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:1370
   pragma Import (C, cusparseSbsrsv2_analysis, "cusparseSbsrsv2_analysis");

   function cusparseDbsrsv2_analysis
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      transA : cusparseOperation_t;
      mb : int;
      nnzb : int;
      descrA : cusparseMatDescr_t;
      bsrSortedValA : access double;
      bsrSortedRowPtrA : access int;
      bsrSortedColIndA : access int;
      blockDim : int;
      info : bsrsv2Info_t;
      policy : cusparseSolvePolicy_t;
      pBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:1384
   pragma Import (C, cusparseDbsrsv2_analysis, "cusparseDbsrsv2_analysis");

   function cusparseCbsrsv2_analysis
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      transA : cusparseOperation_t;
      mb : int;
      nnzb : int;
      descrA : cusparseMatDescr_t;
      bsrSortedValA : access constant cuComplex_h.cuComplex;
      bsrSortedRowPtrA : access int;
      bsrSortedColIndA : access int;
      blockDim : int;
      info : bsrsv2Info_t;
      policy : cusparseSolvePolicy_t;
      pBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:1398
   pragma Import (C, cusparseCbsrsv2_analysis, "cusparseCbsrsv2_analysis");

   function cusparseZbsrsv2_analysis
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      transA : cusparseOperation_t;
      mb : int;
      nnzb : int;
      descrA : cusparseMatDescr_t;
      bsrSortedValA : access constant cuComplex_h.cuDoubleComplex;
      bsrSortedRowPtrA : access int;
      bsrSortedColIndA : access int;
      blockDim : int;
      info : bsrsv2Info_t;
      policy : cusparseSolvePolicy_t;
      pBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:1412
   pragma Import (C, cusparseZbsrsv2_analysis, "cusparseZbsrsv2_analysis");

   function cusparseSbsrsv2_solve
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      transA : cusparseOperation_t;
      mb : int;
      nnzb : int;
      alpha : access float;
      descrA : cusparseMatDescr_t;
      bsrSortedValA : access float;
      bsrSortedRowPtrA : access int;
      bsrSortedColIndA : access int;
      blockDim : int;
      info : bsrsv2Info_t;
      f : access float;
      x : access float;
      policy : cusparseSolvePolicy_t;
      pBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:1427
   pragma Import (C, cusparseSbsrsv2_solve, "cusparseSbsrsv2_solve");

   function cusparseDbsrsv2_solve
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      transA : cusparseOperation_t;
      mb : int;
      nnzb : int;
      alpha : access double;
      descrA : cusparseMatDescr_t;
      bsrSortedValA : access double;
      bsrSortedRowPtrA : access int;
      bsrSortedColIndA : access int;
      blockDim : int;
      info : bsrsv2Info_t;
      f : access double;
      x : access double;
      policy : cusparseSolvePolicy_t;
      pBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:1445
   pragma Import (C, cusparseDbsrsv2_solve, "cusparseDbsrsv2_solve");

   function cusparseCbsrsv2_solve
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      transA : cusparseOperation_t;
      mb : int;
      nnzb : int;
      alpha : access constant cuComplex_h.cuComplex;
      descrA : cusparseMatDescr_t;
      bsrSortedValA : access constant cuComplex_h.cuComplex;
      bsrSortedRowPtrA : access int;
      bsrSortedColIndA : access int;
      blockDim : int;
      info : bsrsv2Info_t;
      f : access constant cuComplex_h.cuComplex;
      x : access cuComplex_h.cuComplex;
      policy : cusparseSolvePolicy_t;
      pBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:1463
   pragma Import (C, cusparseCbsrsv2_solve, "cusparseCbsrsv2_solve");

   function cusparseZbsrsv2_solve
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      transA : cusparseOperation_t;
      mb : int;
      nnzb : int;
      alpha : access constant cuComplex_h.cuDoubleComplex;
      descrA : cusparseMatDescr_t;
      bsrSortedValA : access constant cuComplex_h.cuDoubleComplex;
      bsrSortedRowPtrA : access int;
      bsrSortedColIndA : access int;
      blockDim : int;
      info : bsrsv2Info_t;
      f : access constant cuComplex_h.cuDoubleComplex;
      x : access cuComplex_h.cuDoubleComplex;
      policy : cusparseSolvePolicy_t;
      pBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:1481
   pragma Import (C, cusparseZbsrsv2_solve, "cusparseZbsrsv2_solve");

  -- Description: Solution of triangular linear system op(A) * x = alpha * f, 
  --   where A is a sparse matrix in HYB storage format, rhs f and solution x 
  --   are dense vectors.  

   function cusparseShybsv_analysis
     (handle : cusparseHandle_t;
      transA : cusparseOperation_t;
      descrA : cusparseMatDescr_t;
      hybA : cusparseHybMat_t;
      info : cusparseSolveAnalysisInfo_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:1501
   pragma Import (C, cusparseShybsv_analysis, "cusparseShybsv_analysis");

   function cusparseDhybsv_analysis
     (handle : cusparseHandle_t;
      transA : cusparseOperation_t;
      descrA : cusparseMatDescr_t;
      hybA : cusparseHybMat_t;
      info : cusparseSolveAnalysisInfo_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:1507
   pragma Import (C, cusparseDhybsv_analysis, "cusparseDhybsv_analysis");

   function cusparseChybsv_analysis
     (handle : cusparseHandle_t;
      transA : cusparseOperation_t;
      descrA : cusparseMatDescr_t;
      hybA : cusparseHybMat_t;
      info : cusparseSolveAnalysisInfo_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:1513
   pragma Import (C, cusparseChybsv_analysis, "cusparseChybsv_analysis");

   function cusparseZhybsv_analysis
     (handle : cusparseHandle_t;
      transA : cusparseOperation_t;
      descrA : cusparseMatDescr_t;
      hybA : cusparseHybMat_t;
      info : cusparseSolveAnalysisInfo_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:1519
   pragma Import (C, cusparseZhybsv_analysis, "cusparseZhybsv_analysis");

   function cusparseShybsv_solve
     (handle : cusparseHandle_t;
      trans : cusparseOperation_t;
      alpha : access float;
      descra : cusparseMatDescr_t;
      hybA : cusparseHybMat_t;
      info : cusparseSolveAnalysisInfo_t;
      f : access float;
      x : access float) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:1525
   pragma Import (C, cusparseShybsv_solve, "cusparseShybsv_solve");

   function cusparseChybsv_solve
     (handle : cusparseHandle_t;
      trans : cusparseOperation_t;
      alpha : access constant cuComplex_h.cuComplex;
      descra : cusparseMatDescr_t;
      hybA : cusparseHybMat_t;
      info : cusparseSolveAnalysisInfo_t;
      f : access constant cuComplex_h.cuComplex;
      x : access cuComplex_h.cuComplex) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:1534
   pragma Import (C, cusparseChybsv_solve, "cusparseChybsv_solve");

   function cusparseDhybsv_solve
     (handle : cusparseHandle_t;
      trans : cusparseOperation_t;
      alpha : access double;
      descra : cusparseMatDescr_t;
      hybA : cusparseHybMat_t;
      info : cusparseSolveAnalysisInfo_t;
      f : access double;
      x : access double) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:1543
   pragma Import (C, cusparseDhybsv_solve, "cusparseDhybsv_solve");

   function cusparseZhybsv_solve
     (handle : cusparseHandle_t;
      trans : cusparseOperation_t;
      alpha : access constant cuComplex_h.cuDoubleComplex;
      descra : cusparseMatDescr_t;
      hybA : cusparseHybMat_t;
      info : cusparseSolveAnalysisInfo_t;
      f : access constant cuComplex_h.cuDoubleComplex;
      x : access cuComplex_h.cuDoubleComplex) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:1552
   pragma Import (C, cusparseZhybsv_solve, "cusparseZhybsv_solve");

  -- --- Sparse Level 3 routines ---  
  -- Description: sparse - dense matrix multiplication C = alpha * op(A) * B  + beta * C, 
  --   where A is a sparse matrix in CSR format, B and C are dense tall matrices.   

   function cusparseScsrmm
     (handle : cusparseHandle_t;
      transA : cusparseOperation_t;
      m : int;
      n : int;
      k : int;
      nnz : int;
      alpha : access float;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access float;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      B : access float;
      ldb : int;
      beta : access float;
      C : access float;
      ldc : int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:1566
   pragma Import (C, cusparseScsrmm, "cusparseScsrmm");

   function cusparseDcsrmm
     (handle : cusparseHandle_t;
      transA : cusparseOperation_t;
      m : int;
      n : int;
      k : int;
      nnz : int;
      alpha : access double;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access double;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      B : access double;
      ldb : int;
      beta : access double;
      C : access double;
      ldc : int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:1583
   pragma Import (C, cusparseDcsrmm, "cusparseDcsrmm");

   function cusparseCcsrmm
     (handle : cusparseHandle_t;
      transA : cusparseOperation_t;
      m : int;
      n : int;
      k : int;
      nnz : int;
      alpha : access constant cuComplex_h.cuComplex;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access constant cuComplex_h.cuComplex;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      B : access constant cuComplex_h.cuComplex;
      ldb : int;
      beta : access constant cuComplex_h.cuComplex;
      C : access cuComplex_h.cuComplex;
      ldc : int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:1600
   pragma Import (C, cusparseCcsrmm, "cusparseCcsrmm");

   function cusparseZcsrmm
     (handle : cusparseHandle_t;
      transA : cusparseOperation_t;
      m : int;
      n : int;
      k : int;
      nnz : int;
      alpha : access constant cuComplex_h.cuDoubleComplex;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access constant cuComplex_h.cuDoubleComplex;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      B : access constant cuComplex_h.cuDoubleComplex;
      ldb : int;
      beta : access constant cuComplex_h.cuDoubleComplex;
      C : access cuComplex_h.cuDoubleComplex;
      ldc : int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:1617
   pragma Import (C, cusparseZcsrmm, "cusparseZcsrmm");

  -- Description: sparse - dense matrix multiplication C = alpha * op(A) * B  + beta * C, 
  --   where A is a sparse matrix in CSR format, B and C are dense tall matrices.
  --   This routine allows transposition of matrix B, which may improve performance.  

   function cusparseScsrmm2
     (handle : cusparseHandle_t;
      transA : cusparseOperation_t;
      transB : cusparseOperation_t;
      m : int;
      n : int;
      k : int;
      nnz : int;
      alpha : access float;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access float;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      B : access float;
      ldb : int;
      beta : access float;
      C : access float;
      ldc : int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:1637
   pragma Import (C, cusparseScsrmm2, "cusparseScsrmm2");

   function cusparseDcsrmm2
     (handle : cusparseHandle_t;
      transA : cusparseOperation_t;
      transB : cusparseOperation_t;
      m : int;
      n : int;
      k : int;
      nnz : int;
      alpha : access double;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access double;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      B : access double;
      ldb : int;
      beta : access double;
      C : access double;
      ldc : int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:1655
   pragma Import (C, cusparseDcsrmm2, "cusparseDcsrmm2");

   function cusparseCcsrmm2
     (handle : cusparseHandle_t;
      transA : cusparseOperation_t;
      transB : cusparseOperation_t;
      m : int;
      n : int;
      k : int;
      nnz : int;
      alpha : access constant cuComplex_h.cuComplex;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access constant cuComplex_h.cuComplex;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      B : access constant cuComplex_h.cuComplex;
      ldb : int;
      beta : access constant cuComplex_h.cuComplex;
      C : access cuComplex_h.cuComplex;
      ldc : int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:1673
   pragma Import (C, cusparseCcsrmm2, "cusparseCcsrmm2");

   function cusparseZcsrmm2
     (handle : cusparseHandle_t;
      transA : cusparseOperation_t;
      transB : cusparseOperation_t;
      m : int;
      n : int;
      k : int;
      nnz : int;
      alpha : access constant cuComplex_h.cuDoubleComplex;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access constant cuComplex_h.cuDoubleComplex;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      B : access constant cuComplex_h.cuDoubleComplex;
      ldb : int;
      beta : access constant cuComplex_h.cuDoubleComplex;
      C : access cuComplex_h.cuDoubleComplex;
      ldc : int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:1691
   pragma Import (C, cusparseZcsrmm2, "cusparseZcsrmm2");

  -- Description: sparse - dense matrix multiplication C = alpha * op(A) * B  + beta * C, 
  --   where A is a sparse matrix in block-CSR format, B and C are dense tall matrices.
  --   This routine allows transposition of matrix B, which may improve performance.  

   function cusparseSbsrmm
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      transA : cusparseOperation_t;
      transB : cusparseOperation_t;
      mb : int;
      n : int;
      kb : int;
      nnzb : int;
      alpha : access float;
      descrA : cusparseMatDescr_t;
      bsrSortedValA : access float;
      bsrSortedRowPtrA : access int;
      bsrSortedColIndA : access int;
      blockSize : int;
      B : access float;
      ldb : int;
      beta : access float;
      C : access float;
      ldc : int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:1712
   pragma Import (C, cusparseSbsrmm, "cusparseSbsrmm");

   function cusparseDbsrmm
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      transA : cusparseOperation_t;
      transB : cusparseOperation_t;
      mb : int;
      n : int;
      kb : int;
      nnzb : int;
      alpha : access double;
      descrA : cusparseMatDescr_t;
      bsrSortedValA : access double;
      bsrSortedRowPtrA : access int;
      bsrSortedColIndA : access int;
      blockSize : int;
      B : access double;
      ldb : int;
      beta : access double;
      C : access double;
      ldc : int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:1732
   pragma Import (C, cusparseDbsrmm, "cusparseDbsrmm");

   function cusparseCbsrmm
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      transA : cusparseOperation_t;
      transB : cusparseOperation_t;
      mb : int;
      n : int;
      kb : int;
      nnzb : int;
      alpha : access constant cuComplex_h.cuComplex;
      descrA : cusparseMatDescr_t;
      bsrSortedValA : access constant cuComplex_h.cuComplex;
      bsrSortedRowPtrA : access int;
      bsrSortedColIndA : access int;
      blockSize : int;
      B : access constant cuComplex_h.cuComplex;
      ldb : int;
      beta : access constant cuComplex_h.cuComplex;
      C : access cuComplex_h.cuComplex;
      ldc : int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:1752
   pragma Import (C, cusparseCbsrmm, "cusparseCbsrmm");

   function cusparseZbsrmm
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      transA : cusparseOperation_t;
      transB : cusparseOperation_t;
      mb : int;
      n : int;
      kb : int;
      nnzb : int;
      alpha : access constant cuComplex_h.cuDoubleComplex;
      descrA : cusparseMatDescr_t;
      bsrSortedValA : access constant cuComplex_h.cuDoubleComplex;
      bsrSortedRowPtrA : access int;
      bsrSortedColIndA : access int;
      blockSize : int;
      B : access constant cuComplex_h.cuDoubleComplex;
      ldb : int;
      beta : access constant cuComplex_h.cuDoubleComplex;
      C : access cuComplex_h.cuDoubleComplex;
      ldc : int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:1772
   pragma Import (C, cusparseZbsrmm, "cusparseZbsrmm");

  -- Description: dense - sparse matrix multiplication C = alpha * A * B  + beta * C, 
  --   where A is column-major dense matrix, B is a sparse matrix in CSC format, 
  --   and C is column-major dense matrix.  

   function cusparseSgemmi
     (handle : cusparseHandle_t;
      m : int;
      n : int;
      k : int;
      nnz : int;
      alpha : access float;
      A : access float;
      lda : int;
      cscValB : access float;
      cscColPtrB : access int;
      cscRowIndB : access int;
      beta : access float;
      C : access float;
      ldc : int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:1796
   pragma Import (C, cusparseSgemmi, "cusparseSgemmi");

  -- host or device pointer  
  -- host or device pointer  
   function cusparseDgemmi
     (handle : cusparseHandle_t;
      m : int;
      n : int;
      k : int;
      nnz : int;
      alpha : access double;
      A : access double;
      lda : int;
      cscValB : access double;
      cscColPtrB : access int;
      cscRowIndB : access int;
      beta : access double;
      C : access double;
      ldc : int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:1811
   pragma Import (C, cusparseDgemmi, "cusparseDgemmi");

  -- host or device pointer  
  -- host or device pointer  
   function cusparseCgemmi
     (handle : cusparseHandle_t;
      m : int;
      n : int;
      k : int;
      nnz : int;
      alpha : access constant cuComplex_h.cuComplex;
      A : access constant cuComplex_h.cuComplex;
      lda : int;
      cscValB : access constant cuComplex_h.cuComplex;
      cscColPtrB : access int;
      cscRowIndB : access int;
      beta : access constant cuComplex_h.cuComplex;
      C : access cuComplex_h.cuComplex;
      ldc : int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:1826
   pragma Import (C, cusparseCgemmi, "cusparseCgemmi");

  -- host or device pointer  
  -- host or device pointer  
   function cusparseZgemmi
     (handle : cusparseHandle_t;
      m : int;
      n : int;
      k : int;
      nnz : int;
      alpha : access constant cuComplex_h.cuDoubleComplex;
      A : access constant cuComplex_h.cuDoubleComplex;
      lda : int;
      cscValB : access constant cuComplex_h.cuDoubleComplex;
      cscColPtrB : access int;
      cscRowIndB : access int;
      beta : access constant cuComplex_h.cuDoubleComplex;
      C : access cuComplex_h.cuDoubleComplex;
      ldc : int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:1841
   pragma Import (C, cusparseZgemmi, "cusparseZgemmi");

  -- host or device pointer  
  -- host or device pointer  
  -- Description: Solution of triangular linear system op(A) * X = alpha * F, 
  --   with multiple right-hand-sides, where A is a sparse matrix in CSR storage 
  --   format, rhs F and solution X are dense tall matrices. 
  --   This routine implements algorithm 1 for this problem.  

   function cusparseScsrsm_analysis
     (handle : cusparseHandle_t;
      transA : cusparseOperation_t;
      m : int;
      nnz : int;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access float;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      info : cusparseSolveAnalysisInfo_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:1861
   pragma Import (C, cusparseScsrsm_analysis, "cusparseScsrsm_analysis");

   function cusparseDcsrsm_analysis
     (handle : cusparseHandle_t;
      transA : cusparseOperation_t;
      m : int;
      nnz : int;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access double;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      info : cusparseSolveAnalysisInfo_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:1871
   pragma Import (C, cusparseDcsrsm_analysis, "cusparseDcsrsm_analysis");

   function cusparseCcsrsm_analysis
     (handle : cusparseHandle_t;
      transA : cusparseOperation_t;
      m : int;
      nnz : int;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access constant cuComplex_h.cuComplex;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      info : cusparseSolveAnalysisInfo_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:1881
   pragma Import (C, cusparseCcsrsm_analysis, "cusparseCcsrsm_analysis");

   function cusparseZcsrsm_analysis
     (handle : cusparseHandle_t;
      transA : cusparseOperation_t;
      m : int;
      nnz : int;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access constant cuComplex_h.cuDoubleComplex;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      info : cusparseSolveAnalysisInfo_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:1891
   pragma Import (C, cusparseZcsrsm_analysis, "cusparseZcsrsm_analysis");

   function cusparseScsrsm_solve
     (handle : cusparseHandle_t;
      transA : cusparseOperation_t;
      m : int;
      n : int;
      alpha : access float;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access float;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      info : cusparseSolveAnalysisInfo_t;
      F : access float;
      ldf : int;
      X : access float;
      ldx : int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:1902
   pragma Import (C, cusparseScsrsm_solve, "cusparseScsrsm_solve");

   function cusparseDcsrsm_solve
     (handle : cusparseHandle_t;
      transA : cusparseOperation_t;
      m : int;
      n : int;
      alpha : access double;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access double;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      info : cusparseSolveAnalysisInfo_t;
      F : access double;
      ldf : int;
      X : access double;
      ldx : int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:1917
   pragma Import (C, cusparseDcsrsm_solve, "cusparseDcsrsm_solve");

   function cusparseCcsrsm_solve
     (handle : cusparseHandle_t;
      transA : cusparseOperation_t;
      m : int;
      n : int;
      alpha : access constant cuComplex_h.cuComplex;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access constant cuComplex_h.cuComplex;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      info : cusparseSolveAnalysisInfo_t;
      F : access constant cuComplex_h.cuComplex;
      ldf : int;
      X : access cuComplex_h.cuComplex;
      ldx : int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:1932
   pragma Import (C, cusparseCcsrsm_solve, "cusparseCcsrsm_solve");

   function cusparseZcsrsm_solve
     (handle : cusparseHandle_t;
      transA : cusparseOperation_t;
      m : int;
      n : int;
      alpha : access constant cuComplex_h.cuDoubleComplex;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access constant cuComplex_h.cuDoubleComplex;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      info : cusparseSolveAnalysisInfo_t;
      F : access constant cuComplex_h.cuDoubleComplex;
      ldf : int;
      X : access cuComplex_h.cuDoubleComplex;
      ldx : int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:1947
   pragma Import (C, cusparseZcsrsm_solve, "cusparseZcsrsm_solve");

  -- Description: Solution of triangular linear system op(A) * X = alpha * F, 
  --   with multiple right-hand-sides, where A is a sparse matrix in CSR storage 
  --   format, rhs F and solution X are dense tall matrices.
  --   This routine implements algorithm 2 for this problem.  

   function cusparseXbsrsm2_zeroPivot
     (handle : cusparseHandle_t;
      info : bsrsm2Info_t;
      position : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:1966
   pragma Import (C, cusparseXbsrsm2_zeroPivot, "cusparseXbsrsm2_zeroPivot");

   function cusparseSbsrsm2_bufferSize
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      transA : cusparseOperation_t;
      transXY : cusparseOperation_t;
      mb : int;
      n : int;
      nnzb : int;
      descrA : cusparseMatDescr_t;
      bsrSortedVal : access float;
      bsrSortedRowPtr : access int;
      bsrSortedColInd : access int;
      blockSize : int;
      info : bsrsm2Info_t;
      pBufferSizeInBytes : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:1970
   pragma Import (C, cusparseSbsrsm2_bufferSize, "cusparseSbsrsm2_bufferSize");

   function cusparseDbsrsm2_bufferSize
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      transA : cusparseOperation_t;
      transXY : cusparseOperation_t;
      mb : int;
      n : int;
      nnzb : int;
      descrA : cusparseMatDescr_t;
      bsrSortedVal : access double;
      bsrSortedRowPtr : access int;
      bsrSortedColInd : access int;
      blockSize : int;
      info : bsrsm2Info_t;
      pBufferSizeInBytes : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:1985
   pragma Import (C, cusparseDbsrsm2_bufferSize, "cusparseDbsrsm2_bufferSize");

   function cusparseCbsrsm2_bufferSize
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      transA : cusparseOperation_t;
      transXY : cusparseOperation_t;
      mb : int;
      n : int;
      nnzb : int;
      descrA : cusparseMatDescr_t;
      bsrSortedVal : access cuComplex_h.cuComplex;
      bsrSortedRowPtr : access int;
      bsrSortedColInd : access int;
      blockSize : int;
      info : bsrsm2Info_t;
      pBufferSizeInBytes : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:2000
   pragma Import (C, cusparseCbsrsm2_bufferSize, "cusparseCbsrsm2_bufferSize");

   function cusparseZbsrsm2_bufferSize
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      transA : cusparseOperation_t;
      transXY : cusparseOperation_t;
      mb : int;
      n : int;
      nnzb : int;
      descrA : cusparseMatDescr_t;
      bsrSortedVal : access cuComplex_h.cuDoubleComplex;
      bsrSortedRowPtr : access int;
      bsrSortedColInd : access int;
      blockSize : int;
      info : bsrsm2Info_t;
      pBufferSizeInBytes : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:2015
   pragma Import (C, cusparseZbsrsm2_bufferSize, "cusparseZbsrsm2_bufferSize");

   function cusparseSbsrsm2_bufferSizeExt
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      transA : cusparseOperation_t;
      transB : cusparseOperation_t;
      mb : int;
      n : int;
      nnzb : int;
      descrA : cusparseMatDescr_t;
      bsrSortedVal : access float;
      bsrSortedRowPtr : access int;
      bsrSortedColInd : access int;
      blockSize : int;
      info : bsrsm2Info_t;
      pBufferSize : access stddef_h.size_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:2031
   pragma Import (C, cusparseSbsrsm2_bufferSizeExt, "cusparseSbsrsm2_bufferSizeExt");

   function cusparseDbsrsm2_bufferSizeExt
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      transA : cusparseOperation_t;
      transB : cusparseOperation_t;
      mb : int;
      n : int;
      nnzb : int;
      descrA : cusparseMatDescr_t;
      bsrSortedVal : access double;
      bsrSortedRowPtr : access int;
      bsrSortedColInd : access int;
      blockSize : int;
      info : bsrsm2Info_t;
      pBufferSize : access stddef_h.size_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:2046
   pragma Import (C, cusparseDbsrsm2_bufferSizeExt, "cusparseDbsrsm2_bufferSizeExt");

   function cusparseCbsrsm2_bufferSizeExt
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      transA : cusparseOperation_t;
      transB : cusparseOperation_t;
      mb : int;
      n : int;
      nnzb : int;
      descrA : cusparseMatDescr_t;
      bsrSortedVal : access cuComplex_h.cuComplex;
      bsrSortedRowPtr : access int;
      bsrSortedColInd : access int;
      blockSize : int;
      info : bsrsm2Info_t;
      pBufferSize : access stddef_h.size_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:2061
   pragma Import (C, cusparseCbsrsm2_bufferSizeExt, "cusparseCbsrsm2_bufferSizeExt");

   function cusparseZbsrsm2_bufferSizeExt
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      transA : cusparseOperation_t;
      transB : cusparseOperation_t;
      mb : int;
      n : int;
      nnzb : int;
      descrA : cusparseMatDescr_t;
      bsrSortedVal : access cuComplex_h.cuDoubleComplex;
      bsrSortedRowPtr : access int;
      bsrSortedColInd : access int;
      blockSize : int;
      info : bsrsm2Info_t;
      pBufferSize : access stddef_h.size_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:2076
   pragma Import (C, cusparseZbsrsm2_bufferSizeExt, "cusparseZbsrsm2_bufferSizeExt");

   function cusparseSbsrsm2_analysis
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      transA : cusparseOperation_t;
      transXY : cusparseOperation_t;
      mb : int;
      n : int;
      nnzb : int;
      descrA : cusparseMatDescr_t;
      bsrSortedVal : access float;
      bsrSortedRowPtr : access int;
      bsrSortedColInd : access int;
      blockSize : int;
      info : bsrsm2Info_t;
      policy : cusparseSolvePolicy_t;
      pBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:2092
   pragma Import (C, cusparseSbsrsm2_analysis, "cusparseSbsrsm2_analysis");

   function cusparseDbsrsm2_analysis
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      transA : cusparseOperation_t;
      transXY : cusparseOperation_t;
      mb : int;
      n : int;
      nnzb : int;
      descrA : cusparseMatDescr_t;
      bsrSortedVal : access double;
      bsrSortedRowPtr : access int;
      bsrSortedColInd : access int;
      blockSize : int;
      info : bsrsm2Info_t;
      policy : cusparseSolvePolicy_t;
      pBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:2108
   pragma Import (C, cusparseDbsrsm2_analysis, "cusparseDbsrsm2_analysis");

   function cusparseCbsrsm2_analysis
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      transA : cusparseOperation_t;
      transXY : cusparseOperation_t;
      mb : int;
      n : int;
      nnzb : int;
      descrA : cusparseMatDescr_t;
      bsrSortedVal : access constant cuComplex_h.cuComplex;
      bsrSortedRowPtr : access int;
      bsrSortedColInd : access int;
      blockSize : int;
      info : bsrsm2Info_t;
      policy : cusparseSolvePolicy_t;
      pBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:2124
   pragma Import (C, cusparseCbsrsm2_analysis, "cusparseCbsrsm2_analysis");

   function cusparseZbsrsm2_analysis
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      transA : cusparseOperation_t;
      transXY : cusparseOperation_t;
      mb : int;
      n : int;
      nnzb : int;
      descrA : cusparseMatDescr_t;
      bsrSortedVal : access constant cuComplex_h.cuDoubleComplex;
      bsrSortedRowPtr : access int;
      bsrSortedColInd : access int;
      blockSize : int;
      info : bsrsm2Info_t;
      policy : cusparseSolvePolicy_t;
      pBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:2140
   pragma Import (C, cusparseZbsrsm2_analysis, "cusparseZbsrsm2_analysis");

   function cusparseSbsrsm2_solve
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      transA : cusparseOperation_t;
      transXY : cusparseOperation_t;
      mb : int;
      n : int;
      nnzb : int;
      alpha : access float;
      descrA : cusparseMatDescr_t;
      bsrSortedVal : access float;
      bsrSortedRowPtr : access int;
      bsrSortedColInd : access int;
      blockSize : int;
      info : bsrsm2Info_t;
      F : access float;
      ldf : int;
      X : access float;
      ldx : int;
      policy : cusparseSolvePolicy_t;
      pBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:2157
   pragma Import (C, cusparseSbsrsm2_solve, "cusparseSbsrsm2_solve");

   function cusparseDbsrsm2_solve
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      transA : cusparseOperation_t;
      transXY : cusparseOperation_t;
      mb : int;
      n : int;
      nnzb : int;
      alpha : access double;
      descrA : cusparseMatDescr_t;
      bsrSortedVal : access double;
      bsrSortedRowPtr : access int;
      bsrSortedColInd : access int;
      blockSize : int;
      info : bsrsm2Info_t;
      F : access double;
      ldf : int;
      X : access double;
      ldx : int;
      policy : cusparseSolvePolicy_t;
      pBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:2178
   pragma Import (C, cusparseDbsrsm2_solve, "cusparseDbsrsm2_solve");

   function cusparseCbsrsm2_solve
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      transA : cusparseOperation_t;
      transXY : cusparseOperation_t;
      mb : int;
      n : int;
      nnzb : int;
      alpha : access constant cuComplex_h.cuComplex;
      descrA : cusparseMatDescr_t;
      bsrSortedVal : access constant cuComplex_h.cuComplex;
      bsrSortedRowPtr : access int;
      bsrSortedColInd : access int;
      blockSize : int;
      info : bsrsm2Info_t;
      F : access constant cuComplex_h.cuComplex;
      ldf : int;
      X : access cuComplex_h.cuComplex;
      ldx : int;
      policy : cusparseSolvePolicy_t;
      pBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:2199
   pragma Import (C, cusparseCbsrsm2_solve, "cusparseCbsrsm2_solve");

   function cusparseZbsrsm2_solve
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      transA : cusparseOperation_t;
      transXY : cusparseOperation_t;
      mb : int;
      n : int;
      nnzb : int;
      alpha : access constant cuComplex_h.cuDoubleComplex;
      descrA : cusparseMatDescr_t;
      bsrSortedVal : access constant cuComplex_h.cuDoubleComplex;
      bsrSortedRowPtr : access int;
      bsrSortedColInd : access int;
      blockSize : int;
      info : bsrsm2Info_t;
      F : access constant cuComplex_h.cuDoubleComplex;
      ldf : int;
      X : access cuComplex_h.cuDoubleComplex;
      ldx : int;
      policy : cusparseSolvePolicy_t;
      pBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:2220
   pragma Import (C, cusparseZbsrsm2_solve, "cusparseZbsrsm2_solve");

  -- --- Preconditioners ---  
  -- Description: Compute the incomplete-LU factorization with 0 fill-in (ILU0)
  --   of the matrix A stored in CSR format based on the information in the opaque 
  --   structure info that was obtained from the analysis phase (csrsv_analysis). 
  --   This routine implements algorithm 1 for this problem.  

   function cusparseCsrilu0Ex
     (handle : cusparseHandle_t;
      trans : cusparseOperation_t;
      m : int;
      descrA : cusparseMatDescr_t;
      csrSortedValA_ValM : System.Address;
      csrSortedValA_ValMtype : library_types_h.cudaDataType;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      info : cusparseSolveAnalysisInfo_t;
      executiontype : library_types_h.cudaDataType) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:2248
   pragma Import (C, cusparseCsrilu0Ex, "cusparseCsrilu0Ex");

  -- matrix A values are updated inplace 
  --                                                 to be the preconditioner M values  

   function cusparseScsrilu0
     (handle : cusparseHandle_t;
      trans : cusparseOperation_t;
      m : int;
      descrA : cusparseMatDescr_t;
      csrSortedValA_ValM : access float;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      info : cusparseSolveAnalysisInfo_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:2261
   pragma Import (C, cusparseScsrilu0, "cusparseScsrilu0");

  -- matrix A values are updated inplace 
  --                                                 to be the preconditioner M values  

   function cusparseDcsrilu0
     (handle : cusparseHandle_t;
      trans : cusparseOperation_t;
      m : int;
      descrA : cusparseMatDescr_t;
      csrSortedValA_ValM : access double;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      info : cusparseSolveAnalysisInfo_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:2272
   pragma Import (C, cusparseDcsrilu0, "cusparseDcsrilu0");

  -- matrix A values are updated inplace 
  --                                                 to be the preconditioner M values  

   function cusparseCcsrilu0
     (handle : cusparseHandle_t;
      trans : cusparseOperation_t;
      m : int;
      descrA : cusparseMatDescr_t;
      csrSortedValA_ValM : access cuComplex_h.cuComplex;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      info : cusparseSolveAnalysisInfo_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:2283
   pragma Import (C, cusparseCcsrilu0, "cusparseCcsrilu0");

  -- matrix A values are updated inplace 
  --                                                 to be the preconditioner M values  

   function cusparseZcsrilu0
     (handle : cusparseHandle_t;
      trans : cusparseOperation_t;
      m : int;
      descrA : cusparseMatDescr_t;
      csrSortedValA_ValM : access cuComplex_h.cuDoubleComplex;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      info : cusparseSolveAnalysisInfo_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:2294
   pragma Import (C, cusparseZcsrilu0, "cusparseZcsrilu0");

  -- matrix A values are updated inplace 
  --                                                 to be the preconditioner M values  

  -- Description: Compute the incomplete-LU factorization with 0 fill-in (ILU0)
  --   of the matrix A stored in CSR format based on the information in the opaque 
  --   structure info that was obtained from the analysis phase (csrsv2_analysis).
  --   This routine implements algorithm 2 for this problem.  

   function cusparseScsrilu02_numericBoost
     (handle : cusparseHandle_t;
      info : csrilu02Info_t;
      enable_boost : int;
      tol : access double;
      boost_val : access float) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:2309
   pragma Import (C, cusparseScsrilu02_numericBoost, "cusparseScsrilu02_numericBoost");

   function cusparseDcsrilu02_numericBoost
     (handle : cusparseHandle_t;
      info : csrilu02Info_t;
      enable_boost : int;
      tol : access double;
      boost_val : access double) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:2315
   pragma Import (C, cusparseDcsrilu02_numericBoost, "cusparseDcsrilu02_numericBoost");

   function cusparseCcsrilu02_numericBoost
     (handle : cusparseHandle_t;
      info : csrilu02Info_t;
      enable_boost : int;
      tol : access double;
      boost_val : access cuComplex_h.cuComplex) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:2321
   pragma Import (C, cusparseCcsrilu02_numericBoost, "cusparseCcsrilu02_numericBoost");

   function cusparseZcsrilu02_numericBoost
     (handle : cusparseHandle_t;
      info : csrilu02Info_t;
      enable_boost : int;
      tol : access double;
      boost_val : access cuComplex_h.cuDoubleComplex) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:2327
   pragma Import (C, cusparseZcsrilu02_numericBoost, "cusparseZcsrilu02_numericBoost");

   function cusparseXcsrilu02_zeroPivot
     (handle : cusparseHandle_t;
      info : csrilu02Info_t;
      position : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:2333
   pragma Import (C, cusparseXcsrilu02_zeroPivot, "cusparseXcsrilu02_zeroPivot");

   function cusparseScsrilu02_bufferSize
     (handle : cusparseHandle_t;
      m : int;
      nnz : int;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access float;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      info : csrilu02Info_t;
      pBufferSizeInBytes : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:2337
   pragma Import (C, cusparseScsrilu02_bufferSize, "cusparseScsrilu02_bufferSize");

   function cusparseDcsrilu02_bufferSize
     (handle : cusparseHandle_t;
      m : int;
      nnz : int;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access double;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      info : csrilu02Info_t;
      pBufferSizeInBytes : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:2347
   pragma Import (C, cusparseDcsrilu02_bufferSize, "cusparseDcsrilu02_bufferSize");

   function cusparseCcsrilu02_bufferSize
     (handle : cusparseHandle_t;
      m : int;
      nnz : int;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access cuComplex_h.cuComplex;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      info : csrilu02Info_t;
      pBufferSizeInBytes : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:2357
   pragma Import (C, cusparseCcsrilu02_bufferSize, "cusparseCcsrilu02_bufferSize");

   function cusparseZcsrilu02_bufferSize
     (handle : cusparseHandle_t;
      m : int;
      nnz : int;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access cuComplex_h.cuDoubleComplex;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      info : csrilu02Info_t;
      pBufferSizeInBytes : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:2367
   pragma Import (C, cusparseZcsrilu02_bufferSize, "cusparseZcsrilu02_bufferSize");

   function cusparseScsrilu02_bufferSizeExt
     (handle : cusparseHandle_t;
      m : int;
      nnz : int;
      descrA : cusparseMatDescr_t;
      csrSortedVal : access float;
      csrSortedRowPtr : access int;
      csrSortedColInd : access int;
      info : csrilu02Info_t;
      pBufferSize : access stddef_h.size_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:2377
   pragma Import (C, cusparseScsrilu02_bufferSizeExt, "cusparseScsrilu02_bufferSizeExt");

   function cusparseDcsrilu02_bufferSizeExt
     (handle : cusparseHandle_t;
      m : int;
      nnz : int;
      descrA : cusparseMatDescr_t;
      csrSortedVal : access double;
      csrSortedRowPtr : access int;
      csrSortedColInd : access int;
      info : csrilu02Info_t;
      pBufferSize : access stddef_h.size_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:2387
   pragma Import (C, cusparseDcsrilu02_bufferSizeExt, "cusparseDcsrilu02_bufferSizeExt");

   function cusparseCcsrilu02_bufferSizeExt
     (handle : cusparseHandle_t;
      m : int;
      nnz : int;
      descrA : cusparseMatDescr_t;
      csrSortedVal : access cuComplex_h.cuComplex;
      csrSortedRowPtr : access int;
      csrSortedColInd : access int;
      info : csrilu02Info_t;
      pBufferSize : access stddef_h.size_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:2397
   pragma Import (C, cusparseCcsrilu02_bufferSizeExt, "cusparseCcsrilu02_bufferSizeExt");

   function cusparseZcsrilu02_bufferSizeExt
     (handle : cusparseHandle_t;
      m : int;
      nnz : int;
      descrA : cusparseMatDescr_t;
      csrSortedVal : access cuComplex_h.cuDoubleComplex;
      csrSortedRowPtr : access int;
      csrSortedColInd : access int;
      info : csrilu02Info_t;
      pBufferSize : access stddef_h.size_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:2407
   pragma Import (C, cusparseZcsrilu02_bufferSizeExt, "cusparseZcsrilu02_bufferSizeExt");

   function cusparseScsrilu02_analysis
     (handle : cusparseHandle_t;
      m : int;
      nnz : int;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access float;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      info : csrilu02Info_t;
      policy : cusparseSolvePolicy_t;
      pBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:2417
   pragma Import (C, cusparseScsrilu02_analysis, "cusparseScsrilu02_analysis");

   function cusparseDcsrilu02_analysis
     (handle : cusparseHandle_t;
      m : int;
      nnz : int;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access double;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      info : csrilu02Info_t;
      policy : cusparseSolvePolicy_t;
      pBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:2428
   pragma Import (C, cusparseDcsrilu02_analysis, "cusparseDcsrilu02_analysis");

   function cusparseCcsrilu02_analysis
     (handle : cusparseHandle_t;
      m : int;
      nnz : int;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access constant cuComplex_h.cuComplex;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      info : csrilu02Info_t;
      policy : cusparseSolvePolicy_t;
      pBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:2439
   pragma Import (C, cusparseCcsrilu02_analysis, "cusparseCcsrilu02_analysis");

   function cusparseZcsrilu02_analysis
     (handle : cusparseHandle_t;
      m : int;
      nnz : int;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access constant cuComplex_h.cuDoubleComplex;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      info : csrilu02Info_t;
      policy : cusparseSolvePolicy_t;
      pBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:2450
   pragma Import (C, cusparseZcsrilu02_analysis, "cusparseZcsrilu02_analysis");

   function cusparseScsrilu02
     (handle : cusparseHandle_t;
      m : int;
      nnz : int;
      descrA : cusparseMatDescr_t;
      csrSortedValA_valM : access float;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      info : csrilu02Info_t;
      policy : cusparseSolvePolicy_t;
      pBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:2461
   pragma Import (C, cusparseScsrilu02, "cusparseScsrilu02");

  -- matrix A values are updated inplace 
  --                                                  to be the preconditioner M values  

   function cusparseDcsrilu02
     (handle : cusparseHandle_t;
      m : int;
      nnz : int;
      descrA : cusparseMatDescr_t;
      csrSortedValA_valM : access double;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      info : csrilu02Info_t;
      policy : cusparseSolvePolicy_t;
      pBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:2474
   pragma Import (C, cusparseDcsrilu02, "cusparseDcsrilu02");

  -- matrix A values are updated inplace 
  --                                                  to be the preconditioner M values  

   function cusparseCcsrilu02
     (handle : cusparseHandle_t;
      m : int;
      nnz : int;
      descrA : cusparseMatDescr_t;
      csrSortedValA_valM : access cuComplex_h.cuComplex;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      info : csrilu02Info_t;
      policy : cusparseSolvePolicy_t;
      pBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:2487
   pragma Import (C, cusparseCcsrilu02, "cusparseCcsrilu02");

  -- matrix A values are updated inplace 
  --                                                  to be the preconditioner M values  

   function cusparseZcsrilu02
     (handle : cusparseHandle_t;
      m : int;
      nnz : int;
      descrA : cusparseMatDescr_t;
      csrSortedValA_valM : access cuComplex_h.cuDoubleComplex;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      info : csrilu02Info_t;
      policy : cusparseSolvePolicy_t;
      pBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:2500
   pragma Import (C, cusparseZcsrilu02, "cusparseZcsrilu02");

  -- matrix A values are updated inplace 
  --                                                  to be the preconditioner M values  

  -- Description: Compute the incomplete-LU factorization with 0 fill-in (ILU0)
  --   of the matrix A stored in block-CSR format based on the information in the opaque 
  --   structure info that was obtained from the analysis phase (bsrsv2_analysis).
  --   This routine implements algorithm 2 for this problem.  

   function cusparseSbsrilu02_numericBoost
     (handle : cusparseHandle_t;
      info : bsrilu02Info_t;
      enable_boost : int;
      tol : access double;
      boost_val : access float) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:2517
   pragma Import (C, cusparseSbsrilu02_numericBoost, "cusparseSbsrilu02_numericBoost");

   function cusparseDbsrilu02_numericBoost
     (handle : cusparseHandle_t;
      info : bsrilu02Info_t;
      enable_boost : int;
      tol : access double;
      boost_val : access double) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:2523
   pragma Import (C, cusparseDbsrilu02_numericBoost, "cusparseDbsrilu02_numericBoost");

   function cusparseCbsrilu02_numericBoost
     (handle : cusparseHandle_t;
      info : bsrilu02Info_t;
      enable_boost : int;
      tol : access double;
      boost_val : access cuComplex_h.cuComplex) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:2529
   pragma Import (C, cusparseCbsrilu02_numericBoost, "cusparseCbsrilu02_numericBoost");

   function cusparseZbsrilu02_numericBoost
     (handle : cusparseHandle_t;
      info : bsrilu02Info_t;
      enable_boost : int;
      tol : access double;
      boost_val : access cuComplex_h.cuDoubleComplex) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:2535
   pragma Import (C, cusparseZbsrilu02_numericBoost, "cusparseZbsrilu02_numericBoost");

   function cusparseXbsrilu02_zeroPivot
     (handle : cusparseHandle_t;
      info : bsrilu02Info_t;
      position : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:2541
   pragma Import (C, cusparseXbsrilu02_zeroPivot, "cusparseXbsrilu02_zeroPivot");

   function cusparseSbsrilu02_bufferSize
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      mb : int;
      nnzb : int;
      descrA : cusparseMatDescr_t;
      bsrSortedVal : access float;
      bsrSortedRowPtr : access int;
      bsrSortedColInd : access int;
      blockDim : int;
      info : bsrilu02Info_t;
      pBufferSizeInBytes : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:2545
   pragma Import (C, cusparseSbsrilu02_bufferSize, "cusparseSbsrilu02_bufferSize");

   function cusparseDbsrilu02_bufferSize
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      mb : int;
      nnzb : int;
      descrA : cusparseMatDescr_t;
      bsrSortedVal : access double;
      bsrSortedRowPtr : access int;
      bsrSortedColInd : access int;
      blockDim : int;
      info : bsrilu02Info_t;
      pBufferSizeInBytes : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:2557
   pragma Import (C, cusparseDbsrilu02_bufferSize, "cusparseDbsrilu02_bufferSize");

   function cusparseCbsrilu02_bufferSize
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      mb : int;
      nnzb : int;
      descrA : cusparseMatDescr_t;
      bsrSortedVal : access cuComplex_h.cuComplex;
      bsrSortedRowPtr : access int;
      bsrSortedColInd : access int;
      blockDim : int;
      info : bsrilu02Info_t;
      pBufferSizeInBytes : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:2569
   pragma Import (C, cusparseCbsrilu02_bufferSize, "cusparseCbsrilu02_bufferSize");

   function cusparseZbsrilu02_bufferSize
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      mb : int;
      nnzb : int;
      descrA : cusparseMatDescr_t;
      bsrSortedVal : access cuComplex_h.cuDoubleComplex;
      bsrSortedRowPtr : access int;
      bsrSortedColInd : access int;
      blockDim : int;
      info : bsrilu02Info_t;
      pBufferSizeInBytes : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:2581
   pragma Import (C, cusparseZbsrilu02_bufferSize, "cusparseZbsrilu02_bufferSize");

   function cusparseSbsrilu02_bufferSizeExt
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      mb : int;
      nnzb : int;
      descrA : cusparseMatDescr_t;
      bsrSortedVal : access float;
      bsrSortedRowPtr : access int;
      bsrSortedColInd : access int;
      blockSize : int;
      info : bsrilu02Info_t;
      pBufferSize : access stddef_h.size_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:2593
   pragma Import (C, cusparseSbsrilu02_bufferSizeExt, "cusparseSbsrilu02_bufferSizeExt");

   function cusparseDbsrilu02_bufferSizeExt
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      mb : int;
      nnzb : int;
      descrA : cusparseMatDescr_t;
      bsrSortedVal : access double;
      bsrSortedRowPtr : access int;
      bsrSortedColInd : access int;
      blockSize : int;
      info : bsrilu02Info_t;
      pBufferSize : access stddef_h.size_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:2605
   pragma Import (C, cusparseDbsrilu02_bufferSizeExt, "cusparseDbsrilu02_bufferSizeExt");

   function cusparseCbsrilu02_bufferSizeExt
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      mb : int;
      nnzb : int;
      descrA : cusparseMatDescr_t;
      bsrSortedVal : access cuComplex_h.cuComplex;
      bsrSortedRowPtr : access int;
      bsrSortedColInd : access int;
      blockSize : int;
      info : bsrilu02Info_t;
      pBufferSize : access stddef_h.size_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:2617
   pragma Import (C, cusparseCbsrilu02_bufferSizeExt, "cusparseCbsrilu02_bufferSizeExt");

   function cusparseZbsrilu02_bufferSizeExt
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      mb : int;
      nnzb : int;
      descrA : cusparseMatDescr_t;
      bsrSortedVal : access cuComplex_h.cuDoubleComplex;
      bsrSortedRowPtr : access int;
      bsrSortedColInd : access int;
      blockSize : int;
      info : bsrilu02Info_t;
      pBufferSize : access stddef_h.size_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:2629
   pragma Import (C, cusparseZbsrilu02_bufferSizeExt, "cusparseZbsrilu02_bufferSizeExt");

   function cusparseSbsrilu02_analysis
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      mb : int;
      nnzb : int;
      descrA : cusparseMatDescr_t;
      bsrSortedVal : access float;
      bsrSortedRowPtr : access int;
      bsrSortedColInd : access int;
      blockDim : int;
      info : bsrilu02Info_t;
      policy : cusparseSolvePolicy_t;
      pBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:2642
   pragma Import (C, cusparseSbsrilu02_analysis, "cusparseSbsrilu02_analysis");

   function cusparseDbsrilu02_analysis
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      mb : int;
      nnzb : int;
      descrA : cusparseMatDescr_t;
      bsrSortedVal : access double;
      bsrSortedRowPtr : access int;
      bsrSortedColInd : access int;
      blockDim : int;
      info : bsrilu02Info_t;
      policy : cusparseSolvePolicy_t;
      pBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:2655
   pragma Import (C, cusparseDbsrilu02_analysis, "cusparseDbsrilu02_analysis");

   function cusparseCbsrilu02_analysis
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      mb : int;
      nnzb : int;
      descrA : cusparseMatDescr_t;
      bsrSortedVal : access cuComplex_h.cuComplex;
      bsrSortedRowPtr : access int;
      bsrSortedColInd : access int;
      blockDim : int;
      info : bsrilu02Info_t;
      policy : cusparseSolvePolicy_t;
      pBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:2668
   pragma Import (C, cusparseCbsrilu02_analysis, "cusparseCbsrilu02_analysis");

   function cusparseZbsrilu02_analysis
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      mb : int;
      nnzb : int;
      descrA : cusparseMatDescr_t;
      bsrSortedVal : access cuComplex_h.cuDoubleComplex;
      bsrSortedRowPtr : access int;
      bsrSortedColInd : access int;
      blockDim : int;
      info : bsrilu02Info_t;
      policy : cusparseSolvePolicy_t;
      pBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:2681
   pragma Import (C, cusparseZbsrilu02_analysis, "cusparseZbsrilu02_analysis");

   function cusparseSbsrilu02
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      mb : int;
      nnzb : int;
      descra : cusparseMatDescr_t;
      bsrSortedVal : access float;
      bsrSortedRowPtr : access int;
      bsrSortedColInd : access int;
      blockDim : int;
      info : bsrilu02Info_t;
      policy : cusparseSolvePolicy_t;
      pBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:2695
   pragma Import (C, cusparseSbsrilu02, "cusparseSbsrilu02");

   function cusparseDbsrilu02
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      mb : int;
      nnzb : int;
      descra : cusparseMatDescr_t;
      bsrSortedVal : access double;
      bsrSortedRowPtr : access int;
      bsrSortedColInd : access int;
      blockDim : int;
      info : bsrilu02Info_t;
      policy : cusparseSolvePolicy_t;
      pBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:2708
   pragma Import (C, cusparseDbsrilu02, "cusparseDbsrilu02");

   function cusparseCbsrilu02
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      mb : int;
      nnzb : int;
      descra : cusparseMatDescr_t;
      bsrSortedVal : access cuComplex_h.cuComplex;
      bsrSortedRowPtr : access int;
      bsrSortedColInd : access int;
      blockDim : int;
      info : bsrilu02Info_t;
      policy : cusparseSolvePolicy_t;
      pBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:2721
   pragma Import (C, cusparseCbsrilu02, "cusparseCbsrilu02");

   function cusparseZbsrilu02
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      mb : int;
      nnzb : int;
      descra : cusparseMatDescr_t;
      bsrSortedVal : access cuComplex_h.cuDoubleComplex;
      bsrSortedRowPtr : access int;
      bsrSortedColInd : access int;
      blockDim : int;
      info : bsrilu02Info_t;
      policy : cusparseSolvePolicy_t;
      pBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:2734
   pragma Import (C, cusparseZbsrilu02, "cusparseZbsrilu02");

  -- Description: Compute the incomplete-Cholesky factorization with 0 fill-in (IC0)
  --   of the matrix A stored in CSR format based on the information in the opaque 
  --   structure info that was obtained from the analysis phase (csrsv_analysis). 
  --   This routine implements algorithm 1 for this problem.  

   function cusparseScsric0
     (handle : cusparseHandle_t;
      trans : cusparseOperation_t;
      m : int;
      descrA : cusparseMatDescr_t;
      csrSortedValA_ValM : access float;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      info : cusparseSolveAnalysisInfo_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:2751
   pragma Import (C, cusparseScsric0, "cusparseScsric0");

  -- matrix A values are updated inplace 
  --                                                 to be the preconditioner M values  

   function cusparseDcsric0
     (handle : cusparseHandle_t;
      trans : cusparseOperation_t;
      m : int;
      descrA : cusparseMatDescr_t;
      csrSortedValA_ValM : access double;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      info : cusparseSolveAnalysisInfo_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:2762
   pragma Import (C, cusparseDcsric0, "cusparseDcsric0");

  -- matrix A values are updated inplace 
  --                                                 to be the preconditioner M values  

   function cusparseCcsric0
     (handle : cusparseHandle_t;
      trans : cusparseOperation_t;
      m : int;
      descrA : cusparseMatDescr_t;
      csrSortedValA_ValM : access cuComplex_h.cuComplex;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      info : cusparseSolveAnalysisInfo_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:2773
   pragma Import (C, cusparseCcsric0, "cusparseCcsric0");

  -- matrix A values are updated inplace 
  --                                                 to be the preconditioner M values  

   function cusparseZcsric0
     (handle : cusparseHandle_t;
      trans : cusparseOperation_t;
      m : int;
      descrA : cusparseMatDescr_t;
      csrSortedValA_ValM : access cuComplex_h.cuDoubleComplex;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      info : cusparseSolveAnalysisInfo_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:2784
   pragma Import (C, cusparseZcsric0, "cusparseZcsric0");

  -- matrix A values are updated inplace 
  --                                                 to be the preconditioner M values  

  -- Description: Compute the incomplete-Cholesky factorization with 0 fill-in (IC0)
  --   of the matrix A stored in CSR format based on the information in the opaque 
  --   structure info that was obtained from the analysis phase (csrsv2_analysis). 
  --   This routine implements algorithm 2 for this problem.  

   function cusparseXcsric02_zeroPivot
     (handle : cusparseHandle_t;
      info : csric02Info_t;
      position : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:2799
   pragma Import (C, cusparseXcsric02_zeroPivot, "cusparseXcsric02_zeroPivot");

   function cusparseScsric02_bufferSize
     (handle : cusparseHandle_t;
      m : int;
      nnz : int;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access float;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      info : csric02Info_t;
      pBufferSizeInBytes : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:2803
   pragma Import (C, cusparseScsric02_bufferSize, "cusparseScsric02_bufferSize");

   function cusparseDcsric02_bufferSize
     (handle : cusparseHandle_t;
      m : int;
      nnz : int;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access double;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      info : csric02Info_t;
      pBufferSizeInBytes : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:2813
   pragma Import (C, cusparseDcsric02_bufferSize, "cusparseDcsric02_bufferSize");

   function cusparseCcsric02_bufferSize
     (handle : cusparseHandle_t;
      m : int;
      nnz : int;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access cuComplex_h.cuComplex;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      info : csric02Info_t;
      pBufferSizeInBytes : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:2823
   pragma Import (C, cusparseCcsric02_bufferSize, "cusparseCcsric02_bufferSize");

   function cusparseZcsric02_bufferSize
     (handle : cusparseHandle_t;
      m : int;
      nnz : int;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access cuComplex_h.cuDoubleComplex;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      info : csric02Info_t;
      pBufferSizeInBytes : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:2833
   pragma Import (C, cusparseZcsric02_bufferSize, "cusparseZcsric02_bufferSize");

   function cusparseScsric02_bufferSizeExt
     (handle : cusparseHandle_t;
      m : int;
      nnz : int;
      descrA : cusparseMatDescr_t;
      csrSortedVal : access float;
      csrSortedRowPtr : access int;
      csrSortedColInd : access int;
      info : csric02Info_t;
      pBufferSize : access stddef_h.size_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:2843
   pragma Import (C, cusparseScsric02_bufferSizeExt, "cusparseScsric02_bufferSizeExt");

   function cusparseDcsric02_bufferSizeExt
     (handle : cusparseHandle_t;
      m : int;
      nnz : int;
      descrA : cusparseMatDescr_t;
      csrSortedVal : access double;
      csrSortedRowPtr : access int;
      csrSortedColInd : access int;
      info : csric02Info_t;
      pBufferSize : access stddef_h.size_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:2853
   pragma Import (C, cusparseDcsric02_bufferSizeExt, "cusparseDcsric02_bufferSizeExt");

   function cusparseCcsric02_bufferSizeExt
     (handle : cusparseHandle_t;
      m : int;
      nnz : int;
      descrA : cusparseMatDescr_t;
      csrSortedVal : access cuComplex_h.cuComplex;
      csrSortedRowPtr : access int;
      csrSortedColInd : access int;
      info : csric02Info_t;
      pBufferSize : access stddef_h.size_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:2863
   pragma Import (C, cusparseCcsric02_bufferSizeExt, "cusparseCcsric02_bufferSizeExt");

   function cusparseZcsric02_bufferSizeExt
     (handle : cusparseHandle_t;
      m : int;
      nnz : int;
      descrA : cusparseMatDescr_t;
      csrSortedVal : access cuComplex_h.cuDoubleComplex;
      csrSortedRowPtr : access int;
      csrSortedColInd : access int;
      info : csric02Info_t;
      pBufferSize : access stddef_h.size_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:2873
   pragma Import (C, cusparseZcsric02_bufferSizeExt, "cusparseZcsric02_bufferSizeExt");

   function cusparseScsric02_analysis
     (handle : cusparseHandle_t;
      m : int;
      nnz : int;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access float;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      info : csric02Info_t;
      policy : cusparseSolvePolicy_t;
      pBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:2884
   pragma Import (C, cusparseScsric02_analysis, "cusparseScsric02_analysis");

   function cusparseDcsric02_analysis
     (handle : cusparseHandle_t;
      m : int;
      nnz : int;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access double;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      info : csric02Info_t;
      policy : cusparseSolvePolicy_t;
      pBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:2896
   pragma Import (C, cusparseDcsric02_analysis, "cusparseDcsric02_analysis");

   function cusparseCcsric02_analysis
     (handle : cusparseHandle_t;
      m : int;
      nnz : int;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access constant cuComplex_h.cuComplex;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      info : csric02Info_t;
      policy : cusparseSolvePolicy_t;
      pBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:2907
   pragma Import (C, cusparseCcsric02_analysis, "cusparseCcsric02_analysis");

   function cusparseZcsric02_analysis
     (handle : cusparseHandle_t;
      m : int;
      nnz : int;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access constant cuComplex_h.cuDoubleComplex;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      info : csric02Info_t;
      policy : cusparseSolvePolicy_t;
      pBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:2918
   pragma Import (C, cusparseZcsric02_analysis, "cusparseZcsric02_analysis");

   function cusparseScsric02
     (handle : cusparseHandle_t;
      m : int;
      nnz : int;
      descrA : cusparseMatDescr_t;
      csrSortedValA_valM : access float;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      info : csric02Info_t;
      policy : cusparseSolvePolicy_t;
      pBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:2929
   pragma Import (C, cusparseScsric02, "cusparseScsric02");

  -- matrix A values are updated inplace 
  --                                                 to be the preconditioner M values  

   function cusparseDcsric02
     (handle : cusparseHandle_t;
      m : int;
      nnz : int;
      descrA : cusparseMatDescr_t;
      csrSortedValA_valM : access double;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      info : csric02Info_t;
      policy : cusparseSolvePolicy_t;
      pBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:2942
   pragma Import (C, cusparseDcsric02, "cusparseDcsric02");

  -- matrix A values are updated inplace 
  --                                                 to be the preconditioner M values  

   function cusparseCcsric02
     (handle : cusparseHandle_t;
      m : int;
      nnz : int;
      descrA : cusparseMatDescr_t;
      csrSortedValA_valM : access cuComplex_h.cuComplex;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      info : csric02Info_t;
      policy : cusparseSolvePolicy_t;
      pBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:2955
   pragma Import (C, cusparseCcsric02, "cusparseCcsric02");

  -- matrix A values are updated inplace 
  --                                                 to be the preconditioner M values  

   function cusparseZcsric02
     (handle : cusparseHandle_t;
      m : int;
      nnz : int;
      descrA : cusparseMatDescr_t;
      csrSortedValA_valM : access cuComplex_h.cuDoubleComplex;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      info : csric02Info_t;
      policy : cusparseSolvePolicy_t;
      pBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:2968
   pragma Import (C, cusparseZcsric02, "cusparseZcsric02");

  -- matrix A values are updated inplace 
  --                                                 to be the preconditioner M values  

  -- Description: Compute the incomplete-Cholesky factorization with 0 fill-in (IC0)
  --   of the matrix A stored in block-CSR format based on the information in the opaque 
  --   structure info that was obtained from the analysis phase (bsrsv2_analysis). 
  --   This routine implements algorithm 1 for this problem.  

   function cusparseXbsric02_zeroPivot
     (handle : cusparseHandle_t;
      info : bsric02Info_t;
      position : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:2985
   pragma Import (C, cusparseXbsric02_zeroPivot, "cusparseXbsric02_zeroPivot");

   function cusparseSbsric02_bufferSize
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      mb : int;
      nnzb : int;
      descrA : cusparseMatDescr_t;
      bsrSortedVal : access float;
      bsrSortedRowPtr : access int;
      bsrSortedColInd : access int;
      blockDim : int;
      info : bsric02Info_t;
      pBufferSizeInBytes : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:2989
   pragma Import (C, cusparseSbsric02_bufferSize, "cusparseSbsric02_bufferSize");

   function cusparseDbsric02_bufferSize
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      mb : int;
      nnzb : int;
      descrA : cusparseMatDescr_t;
      bsrSortedVal : access double;
      bsrSortedRowPtr : access int;
      bsrSortedColInd : access int;
      blockDim : int;
      info : bsric02Info_t;
      pBufferSizeInBytes : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:3001
   pragma Import (C, cusparseDbsric02_bufferSize, "cusparseDbsric02_bufferSize");

   function cusparseCbsric02_bufferSize
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      mb : int;
      nnzb : int;
      descrA : cusparseMatDescr_t;
      bsrSortedVal : access cuComplex_h.cuComplex;
      bsrSortedRowPtr : access int;
      bsrSortedColInd : access int;
      blockDim : int;
      info : bsric02Info_t;
      pBufferSizeInBytes : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:3013
   pragma Import (C, cusparseCbsric02_bufferSize, "cusparseCbsric02_bufferSize");

   function cusparseZbsric02_bufferSize
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      mb : int;
      nnzb : int;
      descrA : cusparseMatDescr_t;
      bsrSortedVal : access cuComplex_h.cuDoubleComplex;
      bsrSortedRowPtr : access int;
      bsrSortedColInd : access int;
      blockDim : int;
      info : bsric02Info_t;
      pBufferSizeInBytes : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:3025
   pragma Import (C, cusparseZbsric02_bufferSize, "cusparseZbsric02_bufferSize");

   function cusparseSbsric02_bufferSizeExt
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      mb : int;
      nnzb : int;
      descrA : cusparseMatDescr_t;
      bsrSortedVal : access float;
      bsrSortedRowPtr : access int;
      bsrSortedColInd : access int;
      blockSize : int;
      info : bsric02Info_t;
      pBufferSize : access stddef_h.size_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:3037
   pragma Import (C, cusparseSbsric02_bufferSizeExt, "cusparseSbsric02_bufferSizeExt");

   function cusparseDbsric02_bufferSizeExt
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      mb : int;
      nnzb : int;
      descrA : cusparseMatDescr_t;
      bsrSortedVal : access double;
      bsrSortedRowPtr : access int;
      bsrSortedColInd : access int;
      blockSize : int;
      info : bsric02Info_t;
      pBufferSize : access stddef_h.size_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:3049
   pragma Import (C, cusparseDbsric02_bufferSizeExt, "cusparseDbsric02_bufferSizeExt");

   function cusparseCbsric02_bufferSizeExt
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      mb : int;
      nnzb : int;
      descrA : cusparseMatDescr_t;
      bsrSortedVal : access cuComplex_h.cuComplex;
      bsrSortedRowPtr : access int;
      bsrSortedColInd : access int;
      blockSize : int;
      info : bsric02Info_t;
      pBufferSize : access stddef_h.size_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:3061
   pragma Import (C, cusparseCbsric02_bufferSizeExt, "cusparseCbsric02_bufferSizeExt");

   function cusparseZbsric02_bufferSizeExt
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      mb : int;
      nnzb : int;
      descrA : cusparseMatDescr_t;
      bsrSortedVal : access cuComplex_h.cuDoubleComplex;
      bsrSortedRowPtr : access int;
      bsrSortedColInd : access int;
      blockSize : int;
      info : bsric02Info_t;
      pBufferSize : access stddef_h.size_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:3073
   pragma Import (C, cusparseZbsric02_bufferSizeExt, "cusparseZbsric02_bufferSizeExt");

   function cusparseSbsric02_analysis
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      mb : int;
      nnzb : int;
      descrA : cusparseMatDescr_t;
      bsrSortedVal : access float;
      bsrSortedRowPtr : access int;
      bsrSortedColInd : access int;
      blockDim : int;
      info : bsric02Info_t;
      policy : cusparseSolvePolicy_t;
      pInputBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:3087
   pragma Import (C, cusparseSbsric02_analysis, "cusparseSbsric02_analysis");

   function cusparseDbsric02_analysis
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      mb : int;
      nnzb : int;
      descrA : cusparseMatDescr_t;
      bsrSortedVal : access double;
      bsrSortedRowPtr : access int;
      bsrSortedColInd : access int;
      blockDim : int;
      info : bsric02Info_t;
      policy : cusparseSolvePolicy_t;
      pInputBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:3100
   pragma Import (C, cusparseDbsric02_analysis, "cusparseDbsric02_analysis");

   function cusparseCbsric02_analysis
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      mb : int;
      nnzb : int;
      descrA : cusparseMatDescr_t;
      bsrSortedVal : access constant cuComplex_h.cuComplex;
      bsrSortedRowPtr : access int;
      bsrSortedColInd : access int;
      blockDim : int;
      info : bsric02Info_t;
      policy : cusparseSolvePolicy_t;
      pInputBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:3113
   pragma Import (C, cusparseCbsric02_analysis, "cusparseCbsric02_analysis");

   function cusparseZbsric02_analysis
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      mb : int;
      nnzb : int;
      descrA : cusparseMatDescr_t;
      bsrSortedVal : access constant cuComplex_h.cuDoubleComplex;
      bsrSortedRowPtr : access int;
      bsrSortedColInd : access int;
      blockDim : int;
      info : bsric02Info_t;
      policy : cusparseSolvePolicy_t;
      pInputBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:3126
   pragma Import (C, cusparseZbsric02_analysis, "cusparseZbsric02_analysis");

   function cusparseSbsric02
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      mb : int;
      nnzb : int;
      descrA : cusparseMatDescr_t;
      bsrSortedVal : access float;
      bsrSortedRowPtr : access int;
      bsrSortedColInd : access int;
      blockDim : int;
      info : bsric02Info_t;
      policy : cusparseSolvePolicy_t;
      pBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:3139
   pragma Import (C, cusparseSbsric02, "cusparseSbsric02");

   function cusparseDbsric02
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      mb : int;
      nnzb : int;
      descrA : cusparseMatDescr_t;
      bsrSortedVal : access double;
      bsrSortedRowPtr : access int;
      bsrSortedColInd : access int;
      blockDim : int;
      info : bsric02Info_t;
      policy : cusparseSolvePolicy_t;
      pBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:3152
   pragma Import (C, cusparseDbsric02, "cusparseDbsric02");

   function cusparseCbsric02
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      mb : int;
      nnzb : int;
      descrA : cusparseMatDescr_t;
      bsrSortedVal : access cuComplex_h.cuComplex;
      bsrSortedRowPtr : access int;
      bsrSortedColInd : access int;
      blockDim : int;
      info : bsric02Info_t;
      policy : cusparseSolvePolicy_t;
      pBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:3165
   pragma Import (C, cusparseCbsric02, "cusparseCbsric02");

   function cusparseZbsric02
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      mb : int;
      nnzb : int;
      descrA : cusparseMatDescr_t;
      bsrSortedVal : access cuComplex_h.cuDoubleComplex;
      bsrSortedRowPtr : access int;
      bsrSortedColInd : access int;
      blockDim : int;
      info : bsric02Info_t;
      policy : cusparseSolvePolicy_t;
      pBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:3178
   pragma Import (C, cusparseZbsric02, "cusparseZbsric02");

  -- Description: Solution of tridiagonal linear system A * X = F, 
  --   with multiple right-hand-sides. The coefficient matrix A is 
  --   composed of lower (dl), main (d) and upper (du) diagonals, and 
  --   the right-hand-sides F are overwritten with the solution X. 
  --   These routine use pivoting.  

   function cusparseSgtsv
     (handle : cusparseHandle_t;
      m : int;
      n : int;
      dl : access float;
      d : access float;
      du : access float;
      B : access float;
      ldb : int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:3197
   pragma Import (C, cusparseSgtsv, "cusparseSgtsv");

   function cusparseDgtsv
     (handle : cusparseHandle_t;
      m : int;
      n : int;
      dl : access double;
      d : access double;
      du : access double;
      B : access double;
      ldb : int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:3206
   pragma Import (C, cusparseDgtsv, "cusparseDgtsv");

   function cusparseCgtsv
     (handle : cusparseHandle_t;
      m : int;
      n : int;
      dl : access constant cuComplex_h.cuComplex;
      d : access constant cuComplex_h.cuComplex;
      du : access constant cuComplex_h.cuComplex;
      B : access cuComplex_h.cuComplex;
      ldb : int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:3215
   pragma Import (C, cusparseCgtsv, "cusparseCgtsv");

   function cusparseZgtsv
     (handle : cusparseHandle_t;
      m : int;
      n : int;
      dl : access constant cuComplex_h.cuDoubleComplex;
      d : access constant cuComplex_h.cuDoubleComplex;
      du : access constant cuComplex_h.cuDoubleComplex;
      B : access cuComplex_h.cuDoubleComplex;
      ldb : int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:3224
   pragma Import (C, cusparseZgtsv, "cusparseZgtsv");

  -- Description: Solution of tridiagonal linear system A * X = F, 
  --   with multiple right-hand-sides. The coefficient matrix A is 
  --   composed of lower (dl), main (d) and upper (du) diagonals, and 
  --   the right-hand-sides F are overwritten with the solution X. 
  --   These routine does not use pivoting.  

   function cusparseSgtsv_nopivot
     (handle : cusparseHandle_t;
      m : int;
      n : int;
      dl : access float;
      d : access float;
      du : access float;
      B : access float;
      ldb : int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:3238
   pragma Import (C, cusparseSgtsv_nopivot, "cusparseSgtsv_nopivot");

   function cusparseDgtsv_nopivot
     (handle : cusparseHandle_t;
      m : int;
      n : int;
      dl : access double;
      d : access double;
      du : access double;
      B : access double;
      ldb : int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:3247
   pragma Import (C, cusparseDgtsv_nopivot, "cusparseDgtsv_nopivot");

   function cusparseCgtsv_nopivot
     (handle : cusparseHandle_t;
      m : int;
      n : int;
      dl : access constant cuComplex_h.cuComplex;
      d : access constant cuComplex_h.cuComplex;
      du : access constant cuComplex_h.cuComplex;
      B : access cuComplex_h.cuComplex;
      ldb : int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:3256
   pragma Import (C, cusparseCgtsv_nopivot, "cusparseCgtsv_nopivot");

   function cusparseZgtsv_nopivot
     (handle : cusparseHandle_t;
      m : int;
      n : int;
      dl : access constant cuComplex_h.cuDoubleComplex;
      d : access constant cuComplex_h.cuDoubleComplex;
      du : access constant cuComplex_h.cuDoubleComplex;
      B : access cuComplex_h.cuDoubleComplex;
      ldb : int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:3265
   pragma Import (C, cusparseZgtsv_nopivot, "cusparseZgtsv_nopivot");

  -- Description: Solution of a set of tridiagonal linear systems 
  --   A_{i} * x_{i} = f_{i} for i=1,...,batchCount. The coefficient 
  --   matrices A_{i} are composed of lower (dl), main (d) and upper (du) 
  --   diagonals and stored separated by a batchStride. Also, the 
  --   right-hand-sides/solutions f_{i}/x_{i} are separated by a batchStride.  

   function cusparseSgtsvStridedBatch
     (handle : cusparseHandle_t;
      m : int;
      dl : access float;
      d : access float;
      du : access float;
      x : access float;
      batchCount : int;
      batchStride : int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:3279
   pragma Import (C, cusparseSgtsvStridedBatch, "cusparseSgtsvStridedBatch");

   function cusparseDgtsvStridedBatch
     (handle : cusparseHandle_t;
      m : int;
      dl : access double;
      d : access double;
      du : access double;
      x : access double;
      batchCount : int;
      batchStride : int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:3289
   pragma Import (C, cusparseDgtsvStridedBatch, "cusparseDgtsvStridedBatch");

   function cusparseCgtsvStridedBatch
     (handle : cusparseHandle_t;
      m : int;
      dl : access constant cuComplex_h.cuComplex;
      d : access constant cuComplex_h.cuComplex;
      du : access constant cuComplex_h.cuComplex;
      x : access cuComplex_h.cuComplex;
      batchCount : int;
      batchStride : int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:3298
   pragma Import (C, cusparseCgtsvStridedBatch, "cusparseCgtsvStridedBatch");

   function cusparseZgtsvStridedBatch
     (handle : cusparseHandle_t;
      m : int;
      dl : access constant cuComplex_h.cuDoubleComplex;
      d : access constant cuComplex_h.cuDoubleComplex;
      du : access constant cuComplex_h.cuDoubleComplex;
      x : access cuComplex_h.cuDoubleComplex;
      batchCount : int;
      batchStride : int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:3307
   pragma Import (C, cusparseZgtsvStridedBatch, "cusparseZgtsvStridedBatch");

  -- --- Sparse Level 4 routines ---  
  -- Description: Compute sparse - sparse matrix multiplication for matrices 
  --   stored in CSR format.  

   function cusparseXcsrgemmNnz
     (handle : cusparseHandle_t;
      transA : cusparseOperation_t;
      transB : cusparseOperation_t;
      m : int;
      n : int;
      k : int;
      descrA : cusparseMatDescr_t;
      nnzA : int;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      descrB : cusparseMatDescr_t;
      nnzB : int;
      csrSortedRowPtrB : access int;
      csrSortedColIndB : access int;
      descrC : cusparseMatDescr_t;
      csrSortedRowPtrC : access int;
      nnzTotalDevHostPtr : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:3320
   pragma Import (C, cusparseXcsrgemmNnz, "cusparseXcsrgemmNnz");

   function cusparseScsrgemm
     (handle : cusparseHandle_t;
      transA : cusparseOperation_t;
      transB : cusparseOperation_t;
      m : int;
      n : int;
      k : int;
      descrA : cusparseMatDescr_t;
      nnzA : int;
      csrSortedValA : access float;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      descrB : cusparseMatDescr_t;
      nnzB : int;
      csrSortedValB : access float;
      csrSortedRowPtrB : access int;
      csrSortedColIndB : access int;
      descrC : cusparseMatDescr_t;
      csrSortedValC : access float;
      csrSortedRowPtrC : access int;
      csrSortedColIndC : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:3338
   pragma Import (C, cusparseScsrgemm, "cusparseScsrgemm");

   function cusparseDcsrgemm
     (handle : cusparseHandle_t;
      transA : cusparseOperation_t;
      transB : cusparseOperation_t;
      m : int;
      n : int;
      k : int;
      descrA : cusparseMatDescr_t;
      nnzA : int;
      csrSortedValA : access double;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      descrB : cusparseMatDescr_t;
      nnzB : int;
      csrSortedValB : access double;
      csrSortedRowPtrB : access int;
      csrSortedColIndB : access int;
      descrC : cusparseMatDescr_t;
      csrSortedValC : access double;
      csrSortedRowPtrC : access int;
      csrSortedColIndC : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:3359
   pragma Import (C, cusparseDcsrgemm, "cusparseDcsrgemm");

   function cusparseCcsrgemm
     (handle : cusparseHandle_t;
      transA : cusparseOperation_t;
      transB : cusparseOperation_t;
      m : int;
      n : int;
      k : int;
      descrA : cusparseMatDescr_t;
      nnzA : int;
      csrSortedValA : access constant cuComplex_h.cuComplex;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      descrB : cusparseMatDescr_t;
      nnzB : int;
      csrSortedValB : access constant cuComplex_h.cuComplex;
      csrSortedRowPtrB : access int;
      csrSortedColIndB : access int;
      descrC : cusparseMatDescr_t;
      csrSortedValC : access cuComplex_h.cuComplex;
      csrSortedRowPtrC : access int;
      csrSortedColIndC : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:3380
   pragma Import (C, cusparseCcsrgemm, "cusparseCcsrgemm");

   function cusparseZcsrgemm
     (handle : cusparseHandle_t;
      transA : cusparseOperation_t;
      transB : cusparseOperation_t;
      m : int;
      n : int;
      k : int;
      descrA : cusparseMatDescr_t;
      nnzA : int;
      csrSortedValA : access constant cuComplex_h.cuDoubleComplex;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      descrB : cusparseMatDescr_t;
      nnzB : int;
      csrSortedValB : access constant cuComplex_h.cuDoubleComplex;
      csrSortedRowPtrB : access int;
      csrSortedColIndB : access int;
      descrC : cusparseMatDescr_t;
      csrSortedValC : access cuComplex_h.cuDoubleComplex;
      csrSortedRowPtrC : access int;
      csrSortedColIndC : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:3401
   pragma Import (C, cusparseZcsrgemm, "cusparseZcsrgemm");

  -- Description: Compute sparse - sparse matrix multiplication for matrices 
  --   stored in CSR format.  

   function cusparseCreateCsrgemm2Info (info : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:3425
   pragma Import (C, cusparseCreateCsrgemm2Info, "cusparseCreateCsrgemm2Info");

   function cusparseDestroyCsrgemm2Info (info : csrgemm2Info_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:3427
   pragma Import (C, cusparseDestroyCsrgemm2Info, "cusparseDestroyCsrgemm2Info");

   function cusparseScsrgemm2_bufferSizeExt
     (handle : cusparseHandle_t;
      m : int;
      n : int;
      k : int;
      alpha : access float;
      descrA : cusparseMatDescr_t;
      nnzA : int;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      descrB : cusparseMatDescr_t;
      nnzB : int;
      csrSortedRowPtrB : access int;
      csrSortedColIndB : access int;
      beta : access float;
      descrD : cusparseMatDescr_t;
      nnzD : int;
      csrSortedRowPtrD : access int;
      csrSortedColIndD : access int;
      info : csrgemm2Info_t;
      pBufferSizeInBytes : access stddef_h.size_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:3429
   pragma Import (C, cusparseScsrgemm2_bufferSizeExt, "cusparseScsrgemm2_bufferSizeExt");

   function cusparseDcsrgemm2_bufferSizeExt
     (handle : cusparseHandle_t;
      m : int;
      n : int;
      k : int;
      alpha : access double;
      descrA : cusparseMatDescr_t;
      nnzA : int;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      descrB : cusparseMatDescr_t;
      nnzB : int;
      csrSortedRowPtrB : access int;
      csrSortedColIndB : access int;
      beta : access double;
      descrD : cusparseMatDescr_t;
      nnzD : int;
      csrSortedRowPtrD : access int;
      csrSortedColIndD : access int;
      info : csrgemm2Info_t;
      pBufferSizeInBytes : access stddef_h.size_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:3450
   pragma Import (C, cusparseDcsrgemm2_bufferSizeExt, "cusparseDcsrgemm2_bufferSizeExt");

   function cusparseCcsrgemm2_bufferSizeExt
     (handle : cusparseHandle_t;
      m : int;
      n : int;
      k : int;
      alpha : access constant cuComplex_h.cuComplex;
      descrA : cusparseMatDescr_t;
      nnzA : int;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      descrB : cusparseMatDescr_t;
      nnzB : int;
      csrSortedRowPtrB : access int;
      csrSortedColIndB : access int;
      beta : access constant cuComplex_h.cuComplex;
      descrD : cusparseMatDescr_t;
      nnzD : int;
      csrSortedRowPtrD : access int;
      csrSortedColIndD : access int;
      info : csrgemm2Info_t;
      pBufferSizeInBytes : access stddef_h.size_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:3471
   pragma Import (C, cusparseCcsrgemm2_bufferSizeExt, "cusparseCcsrgemm2_bufferSizeExt");

   function cusparseZcsrgemm2_bufferSizeExt
     (handle : cusparseHandle_t;
      m : int;
      n : int;
      k : int;
      alpha : access constant cuComplex_h.cuDoubleComplex;
      descrA : cusparseMatDescr_t;
      nnzA : int;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      descrB : cusparseMatDescr_t;
      nnzB : int;
      csrSortedRowPtrB : access int;
      csrSortedColIndB : access int;
      beta : access constant cuComplex_h.cuDoubleComplex;
      descrD : cusparseMatDescr_t;
      nnzD : int;
      csrSortedRowPtrD : access int;
      csrSortedColIndD : access int;
      info : csrgemm2Info_t;
      pBufferSizeInBytes : access stddef_h.size_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:3492
   pragma Import (C, cusparseZcsrgemm2_bufferSizeExt, "cusparseZcsrgemm2_bufferSizeExt");

   function cusparseXcsrgemm2Nnz
     (handle : cusparseHandle_t;
      m : int;
      n : int;
      k : int;
      descrA : cusparseMatDescr_t;
      nnzA : int;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      descrB : cusparseMatDescr_t;
      nnzB : int;
      csrSortedRowPtrB : access int;
      csrSortedColIndB : access int;
      descrD : cusparseMatDescr_t;
      nnzD : int;
      csrSortedRowPtrD : access int;
      csrSortedColIndD : access int;
      descrC : cusparseMatDescr_t;
      csrSortedRowPtrC : access int;
      nnzTotalDevHostPtr : access int;
      info : csrgemm2Info_t;
      pBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:3514
   pragma Import (C, cusparseXcsrgemm2Nnz, "cusparseXcsrgemm2Nnz");

   function cusparseScsrgemm2
     (handle : cusparseHandle_t;
      m : int;
      n : int;
      k : int;
      alpha : access float;
      descrA : cusparseMatDescr_t;
      nnzA : int;
      csrSortedValA : access float;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      descrB : cusparseMatDescr_t;
      nnzB : int;
      csrSortedValB : access float;
      csrSortedRowPtrB : access int;
      csrSortedColIndB : access int;
      beta : access float;
      descrD : cusparseMatDescr_t;
      nnzD : int;
      csrSortedValD : access float;
      csrSortedRowPtrD : access int;
      csrSortedColIndD : access int;
      descrC : cusparseMatDescr_t;
      csrSortedValC : access float;
      csrSortedRowPtrC : access int;
      csrSortedColIndC : access int;
      info : csrgemm2Info_t;
      pBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:3537
   pragma Import (C, cusparseScsrgemm2, "cusparseScsrgemm2");

   function cusparseDcsrgemm2
     (handle : cusparseHandle_t;
      m : int;
      n : int;
      k : int;
      alpha : access double;
      descrA : cusparseMatDescr_t;
      nnzA : int;
      csrSortedValA : access double;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      descrB : cusparseMatDescr_t;
      nnzB : int;
      csrSortedValB : access double;
      csrSortedRowPtrB : access int;
      csrSortedColIndB : access int;
      beta : access double;
      descrD : cusparseMatDescr_t;
      nnzD : int;
      csrSortedValD : access double;
      csrSortedRowPtrD : access int;
      csrSortedColIndD : access int;
      descrC : cusparseMatDescr_t;
      csrSortedValC : access double;
      csrSortedRowPtrC : access int;
      csrSortedColIndC : access int;
      info : csrgemm2Info_t;
      pBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:3565
   pragma Import (C, cusparseDcsrgemm2, "cusparseDcsrgemm2");

   function cusparseCcsrgemm2
     (handle : cusparseHandle_t;
      m : int;
      n : int;
      k : int;
      alpha : access constant cuComplex_h.cuComplex;
      descrA : cusparseMatDescr_t;
      nnzA : int;
      csrSortedValA : access constant cuComplex_h.cuComplex;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      descrB : cusparseMatDescr_t;
      nnzB : int;
      csrSortedValB : access constant cuComplex_h.cuComplex;
      csrSortedRowPtrB : access int;
      csrSortedColIndB : access int;
      beta : access constant cuComplex_h.cuComplex;
      descrD : cusparseMatDescr_t;
      nnzD : int;
      csrSortedValD : access constant cuComplex_h.cuComplex;
      csrSortedRowPtrD : access int;
      csrSortedColIndD : access int;
      descrC : cusparseMatDescr_t;
      csrSortedValC : access cuComplex_h.cuComplex;
      csrSortedRowPtrC : access int;
      csrSortedColIndC : access int;
      info : csrgemm2Info_t;
      pBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:3594
   pragma Import (C, cusparseCcsrgemm2, "cusparseCcsrgemm2");

   function cusparseZcsrgemm2
     (handle : cusparseHandle_t;
      m : int;
      n : int;
      k : int;
      alpha : access constant cuComplex_h.cuDoubleComplex;
      descrA : cusparseMatDescr_t;
      nnzA : int;
      csrSortedValA : access constant cuComplex_h.cuDoubleComplex;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      descrB : cusparseMatDescr_t;
      nnzB : int;
      csrSortedValB : access constant cuComplex_h.cuDoubleComplex;
      csrSortedRowPtrB : access int;
      csrSortedColIndB : access int;
      beta : access constant cuComplex_h.cuDoubleComplex;
      descrD : cusparseMatDescr_t;
      nnzD : int;
      csrSortedValD : access constant cuComplex_h.cuDoubleComplex;
      csrSortedRowPtrD : access int;
      csrSortedColIndD : access int;
      descrC : cusparseMatDescr_t;
      csrSortedValC : access cuComplex_h.cuDoubleComplex;
      csrSortedRowPtrC : access int;
      csrSortedColIndC : access int;
      info : csrgemm2Info_t;
      pBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:3623
   pragma Import (C, cusparseZcsrgemm2, "cusparseZcsrgemm2");

  -- Description: Compute sparse - sparse matrix addition of matrices 
  --   stored in CSR format  

   function cusparseXcsrgeamNnz
     (handle : cusparseHandle_t;
      m : int;
      n : int;
      descrA : cusparseMatDescr_t;
      nnzA : int;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      descrB : cusparseMatDescr_t;
      nnzB : int;
      csrSortedRowPtrB : access int;
      csrSortedColIndB : access int;
      descrC : cusparseMatDescr_t;
      csrSortedRowPtrC : access int;
      nnzTotalDevHostPtr : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:3654
   pragma Import (C, cusparseXcsrgeamNnz, "cusparseXcsrgeamNnz");

   function cusparseScsrgeam
     (handle : cusparseHandle_t;
      m : int;
      n : int;
      alpha : access float;
      descrA : cusparseMatDescr_t;
      nnzA : int;
      csrSortedValA : access float;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      beta : access float;
      descrB : cusparseMatDescr_t;
      nnzB : int;
      csrSortedValB : access float;
      csrSortedRowPtrB : access int;
      csrSortedColIndB : access int;
      descrC : cusparseMatDescr_t;
      csrSortedValC : access float;
      csrSortedRowPtrC : access int;
      csrSortedColIndC : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:3669
   pragma Import (C, cusparseScsrgeam, "cusparseScsrgeam");

   function cusparseDcsrgeam
     (handle : cusparseHandle_t;
      m : int;
      n : int;
      alpha : access double;
      descrA : cusparseMatDescr_t;
      nnzA : int;
      csrSortedValA : access double;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      beta : access double;
      descrB : cusparseMatDescr_t;
      nnzB : int;
      csrSortedValB : access double;
      csrSortedRowPtrB : access int;
      csrSortedColIndB : access int;
      descrC : cusparseMatDescr_t;
      csrSortedValC : access double;
      csrSortedRowPtrC : access int;
      csrSortedColIndC : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:3689
   pragma Import (C, cusparseDcsrgeam, "cusparseDcsrgeam");

   function cusparseCcsrgeam
     (handle : cusparseHandle_t;
      m : int;
      n : int;
      alpha : access constant cuComplex_h.cuComplex;
      descrA : cusparseMatDescr_t;
      nnzA : int;
      csrSortedValA : access constant cuComplex_h.cuComplex;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      beta : access constant cuComplex_h.cuComplex;
      descrB : cusparseMatDescr_t;
      nnzB : int;
      csrSortedValB : access constant cuComplex_h.cuComplex;
      csrSortedRowPtrB : access int;
      csrSortedColIndB : access int;
      descrC : cusparseMatDescr_t;
      csrSortedValC : access cuComplex_h.cuComplex;
      csrSortedRowPtrC : access int;
      csrSortedColIndC : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:3709
   pragma Import (C, cusparseCcsrgeam, "cusparseCcsrgeam");

   function cusparseZcsrgeam
     (handle : cusparseHandle_t;
      m : int;
      n : int;
      alpha : access constant cuComplex_h.cuDoubleComplex;
      descrA : cusparseMatDescr_t;
      nnzA : int;
      csrSortedValA : access constant cuComplex_h.cuDoubleComplex;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      beta : access constant cuComplex_h.cuDoubleComplex;
      descrB : cusparseMatDescr_t;
      nnzB : int;
      csrSortedValB : access constant cuComplex_h.cuDoubleComplex;
      csrSortedRowPtrB : access int;
      csrSortedColIndB : access int;
      descrC : cusparseMatDescr_t;
      csrSortedValC : access cuComplex_h.cuDoubleComplex;
      csrSortedRowPtrC : access int;
      csrSortedColIndC : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:3729
   pragma Import (C, cusparseZcsrgeam, "cusparseZcsrgeam");

  -- --- Sparse Matrix Reorderings ---  
  -- Description: Find an approximate coloring of a matrix stored in CSR format.  
   function cusparseScsrcolor
     (handle : cusparseHandle_t;
      m : int;
      nnz : int;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access float;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      fractionToColor : access float;
      ncolors : access int;
      coloring : access int;
      reordering : access int;
      info : cusparseColorInfo_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:3753
   pragma Import (C, cusparseScsrcolor, "cusparseScsrcolor");

   function cusparseDcsrcolor
     (handle : cusparseHandle_t;
      m : int;
      nnz : int;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access double;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      fractionToColor : access double;
      ncolors : access int;
      coloring : access int;
      reordering : access int;
      info : cusparseColorInfo_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:3766
   pragma Import (C, cusparseDcsrcolor, "cusparseDcsrcolor");

   function cusparseCcsrcolor
     (handle : cusparseHandle_t;
      m : int;
      nnz : int;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access constant cuComplex_h.cuComplex;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      fractionToColor : access float;
      ncolors : access int;
      coloring : access int;
      reordering : access int;
      info : cusparseColorInfo_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:3779
   pragma Import (C, cusparseCcsrcolor, "cusparseCcsrcolor");

   function cusparseZcsrcolor
     (handle : cusparseHandle_t;
      m : int;
      nnz : int;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access constant cuComplex_h.cuDoubleComplex;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      fractionToColor : access double;
      ncolors : access int;
      coloring : access int;
      reordering : access int;
      info : cusparseColorInfo_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:3792
   pragma Import (C, cusparseZcsrcolor, "cusparseZcsrcolor");

  -- --- Sparse Format Conversion ---  
  -- Description: This routine finds the total number of non-zero elements and 
  --   the number of non-zero elements per row or column in the dense matrix A.  

   function cusparseSnnz
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      m : int;
      n : int;
      descrA : cusparseMatDescr_t;
      A : access float;
      lda : int;
      nnzPerRowCol : access int;
      nnzTotalDevHostPtr : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:3809
   pragma Import (C, cusparseSnnz, "cusparseSnnz");

   function cusparseDnnz
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      m : int;
      n : int;
      descrA : cusparseMatDescr_t;
      A : access double;
      lda : int;
      nnzPerRowCol : access int;
      nnzTotalDevHostPtr : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:3819
   pragma Import (C, cusparseDnnz, "cusparseDnnz");

   function cusparseCnnz
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      m : int;
      n : int;
      descrA : cusparseMatDescr_t;
      A : access constant cuComplex_h.cuComplex;
      lda : int;
      nnzPerRowCol : access int;
      nnzTotalDevHostPtr : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:3829
   pragma Import (C, cusparseCnnz, "cusparseCnnz");

   function cusparseZnnz
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      m : int;
      n : int;
      descrA : cusparseMatDescr_t;
      A : access constant cuComplex_h.cuDoubleComplex;
      lda : int;
      nnzPerRowCol : access int;
      nnzTotalDevHostPtr : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:3839
   pragma Import (C, cusparseZnnz, "cusparseZnnz");

  -- --- Sparse Format Conversion ---  
  -- Description: This routine finds the total number of non-zero elements and 
  --   the number of non-zero elements per row in a noncompressed csr matrix A.  

   function cusparseSnnz_compress
     (handle : cusparseHandle_t;
      m : int;
      descr : cusparseMatDescr_t;
      values : access float;
      rowPtr : access int;
      nnzPerRow : access int;
      nnzTotal : access int;
      tol : float) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:3853
   pragma Import (C, cusparseSnnz_compress, "cusparseSnnz_compress");

   function cusparseDnnz_compress
     (handle : cusparseHandle_t;
      m : int;
      descr : cusparseMatDescr_t;
      values : access double;
      rowPtr : access int;
      nnzPerRow : access int;
      nnzTotal : access int;
      tol : double) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:3862
   pragma Import (C, cusparseDnnz_compress, "cusparseDnnz_compress");

   function cusparseCnnz_compress
     (handle : cusparseHandle_t;
      m : int;
      descr : cusparseMatDescr_t;
      values : access constant cuComplex_h.cuComplex;
      rowPtr : access int;
      nnzPerRow : access int;
      nnzTotal : access int;
      tol : cuComplex_h.cuComplex) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:3871
   pragma Import (C, cusparseCnnz_compress, "cusparseCnnz_compress");

   function cusparseZnnz_compress
     (handle : cusparseHandle_t;
      m : int;
      descr : cusparseMatDescr_t;
      values : access constant cuComplex_h.cuDoubleComplex;
      rowPtr : access int;
      nnzPerRow : access int;
      nnzTotal : access int;
      tol : cuComplex_h.cuDoubleComplex) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:3880
   pragma Import (C, cusparseZnnz_compress, "cusparseZnnz_compress");

  -- Description: This routine takes as input a csr form where the values may have 0 elements
  --   and compresses it to return a csr form with no zeros.  

   function cusparseScsr2csr_compress
     (handle : cusparseHandle_t;
      m : int;
      n : int;
      descra : cusparseMatDescr_t;
      inVal : access float;
      inColInd : access int;
      inRowPtr : access int;
      inNnz : int;
      nnzPerRow : access int;
      outVal : access float;
      outColInd : access int;
      outRowPtr : access int;
      tol : float) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:3891
   pragma Import (C, cusparseScsr2csr_compress, "cusparseScsr2csr_compress");

   function cusparseDcsr2csr_compress
     (handle : cusparseHandle_t;
      m : int;
      n : int;
      descra : cusparseMatDescr_t;
      inVal : access double;
      inColInd : access int;
      inRowPtr : access int;
      inNnz : int;
      nnzPerRow : access int;
      outVal : access double;
      outColInd : access int;
      outRowPtr : access int;
      tol : double) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:3905
   pragma Import (C, cusparseDcsr2csr_compress, "cusparseDcsr2csr_compress");

  --number of rows
  --csr values array-the elements which are below a certain tolerance will be remvoed
  --corresponding input noncompressed row pointer
  --output: returns number of nonzeros per row 
   function cusparseCcsr2csr_compress
     (handle : cusparseHandle_t;
      m : int;
      n : int;
      descra : cusparseMatDescr_t;
      inVal : access constant cuComplex_h.cuComplex;
      inColInd : access int;
      inRowPtr : access int;
      inNnz : int;
      nnzPerRow : access int;
      outVal : access cuComplex_h.cuComplex;
      outColInd : access int;
      outRowPtr : access int;
      tol : cuComplex_h.cuComplex) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:3919
   pragma Import (C, cusparseCcsr2csr_compress, "cusparseCcsr2csr_compress");

  --number of rows
  --csr values array-the elements which are below a certain tolerance will be remvoed
  --corresponding input noncompressed row pointer
  --output: returns number of nonzeros per row 
   function cusparseZcsr2csr_compress
     (handle : cusparseHandle_t;
      m : int;
      n : int;
      descra : cusparseMatDescr_t;
      inVal : access constant cuComplex_h.cuDoubleComplex;
      inColInd : access int;
      inRowPtr : access int;
      inNnz : int;
      nnzPerRow : access int;
      outVal : access cuComplex_h.cuDoubleComplex;
      outColInd : access int;
      outRowPtr : access int;
      tol : cuComplex_h.cuDoubleComplex) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:3933
   pragma Import (C, cusparseZcsr2csr_compress, "cusparseZcsr2csr_compress");

  --number of rows
  --csr values array-the elements which are below a certain tolerance will be remvoed
  --corresponding input noncompressed row pointer
  --output: returns number of nonzeros per row 
  -- Description: This routine converts a dense matrix to a sparse matrix 
  --   in the CSR storage format, using the information computed by the 
  --   nnz routine.  

   function cusparseSdense2csr
     (handle : cusparseHandle_t;
      m : int;
      n : int;
      descrA : cusparseMatDescr_t;
      A : access float;
      lda : int;
      nnzPerRow : access int;
      csrSortedValA : access float;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:3950
   pragma Import (C, cusparseSdense2csr, "cusparseSdense2csr");

   function cusparseDdense2csr
     (handle : cusparseHandle_t;
      m : int;
      n : int;
      descrA : cusparseMatDescr_t;
      A : access double;
      lda : int;
      nnzPerRow : access int;
      csrSortedValA : access double;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:3961
   pragma Import (C, cusparseDdense2csr, "cusparseDdense2csr");

   function cusparseCdense2csr
     (handle : cusparseHandle_t;
      m : int;
      n : int;
      descrA : cusparseMatDescr_t;
      A : access constant cuComplex_h.cuComplex;
      lda : int;
      nnzPerRow : access int;
      csrSortedValA : access cuComplex_h.cuComplex;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:3972
   pragma Import (C, cusparseCdense2csr, "cusparseCdense2csr");

   function cusparseZdense2csr
     (handle : cusparseHandle_t;
      m : int;
      n : int;
      descrA : cusparseMatDescr_t;
      A : access constant cuComplex_h.cuDoubleComplex;
      lda : int;
      nnzPerRow : access int;
      csrSortedValA : access cuComplex_h.cuDoubleComplex;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:3983
   pragma Import (C, cusparseZdense2csr, "cusparseZdense2csr");

  -- Description: This routine converts a sparse matrix in CSR storage format
  --   to a dense matrix.  

   function cusparseScsr2dense
     (handle : cusparseHandle_t;
      m : int;
      n : int;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access float;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      A : access float;
      lda : int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:3996
   pragma Import (C, cusparseScsr2dense, "cusparseScsr2dense");

   function cusparseDcsr2dense
     (handle : cusparseHandle_t;
      m : int;
      n : int;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access double;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      A : access double;
      lda : int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:4006
   pragma Import (C, cusparseDcsr2dense, "cusparseDcsr2dense");

   function cusparseCcsr2dense
     (handle : cusparseHandle_t;
      m : int;
      n : int;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access constant cuComplex_h.cuComplex;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      A : access cuComplex_h.cuComplex;
      lda : int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:4016
   pragma Import (C, cusparseCcsr2dense, "cusparseCcsr2dense");

   function cusparseZcsr2dense
     (handle : cusparseHandle_t;
      m : int;
      n : int;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access constant cuComplex_h.cuDoubleComplex;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      A : access cuComplex_h.cuDoubleComplex;
      lda : int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:4026
   pragma Import (C, cusparseZcsr2dense, "cusparseZcsr2dense");

  -- Description: This routine converts a dense matrix to a sparse matrix 
  --   in the CSC storage format, using the information computed by the 
  --   nnz routine.  

   function cusparseSdense2csc
     (handle : cusparseHandle_t;
      m : int;
      n : int;
      descrA : cusparseMatDescr_t;
      A : access float;
      lda : int;
      nnzPerCol : access int;
      cscSortedValA : access float;
      cscSortedRowIndA : access int;
      cscSortedColPtrA : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:4039
   pragma Import (C, cusparseSdense2csc, "cusparseSdense2csc");

   function cusparseDdense2csc
     (handle : cusparseHandle_t;
      m : int;
      n : int;
      descrA : cusparseMatDescr_t;
      A : access double;
      lda : int;
      nnzPerCol : access int;
      cscSortedValA : access double;
      cscSortedRowIndA : access int;
      cscSortedColPtrA : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:4050
   pragma Import (C, cusparseDdense2csc, "cusparseDdense2csc");

   function cusparseCdense2csc
     (handle : cusparseHandle_t;
      m : int;
      n : int;
      descrA : cusparseMatDescr_t;
      A : access constant cuComplex_h.cuComplex;
      lda : int;
      nnzPerCol : access int;
      cscSortedValA : access cuComplex_h.cuComplex;
      cscSortedRowIndA : access int;
      cscSortedColPtrA : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:4061
   pragma Import (C, cusparseCdense2csc, "cusparseCdense2csc");

   function cusparseZdense2csc
     (handle : cusparseHandle_t;
      m : int;
      n : int;
      descrA : cusparseMatDescr_t;
      A : access constant cuComplex_h.cuDoubleComplex;
      lda : int;
      nnzPerCol : access int;
      cscSortedValA : access cuComplex_h.cuDoubleComplex;
      cscSortedRowIndA : access int;
      cscSortedColPtrA : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:4072
   pragma Import (C, cusparseZdense2csc, "cusparseZdense2csc");

  -- Description: This routine converts a sparse matrix in CSC storage format
  --   to a dense matrix.  

   function cusparseScsc2dense
     (handle : cusparseHandle_t;
      m : int;
      n : int;
      descrA : cusparseMatDescr_t;
      cscSortedValA : access float;
      cscSortedRowIndA : access int;
      cscSortedColPtrA : access int;
      A : access float;
      lda : int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:4085
   pragma Import (C, cusparseScsc2dense, "cusparseScsc2dense");

   function cusparseDcsc2dense
     (handle : cusparseHandle_t;
      m : int;
      n : int;
      descrA : cusparseMatDescr_t;
      cscSortedValA : access double;
      cscSortedRowIndA : access int;
      cscSortedColPtrA : access int;
      A : access double;
      lda : int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:4095
   pragma Import (C, cusparseDcsc2dense, "cusparseDcsc2dense");

   function cusparseCcsc2dense
     (handle : cusparseHandle_t;
      m : int;
      n : int;
      descrA : cusparseMatDescr_t;
      cscSortedValA : access constant cuComplex_h.cuComplex;
      cscSortedRowIndA : access int;
      cscSortedColPtrA : access int;
      A : access cuComplex_h.cuComplex;
      lda : int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:4105
   pragma Import (C, cusparseCcsc2dense, "cusparseCcsc2dense");

   function cusparseZcsc2dense
     (handle : cusparseHandle_t;
      m : int;
      n : int;
      descrA : cusparseMatDescr_t;
      cscSortedValA : access constant cuComplex_h.cuDoubleComplex;
      cscSortedRowIndA : access int;
      cscSortedColPtrA : access int;
      A : access cuComplex_h.cuDoubleComplex;
      lda : int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:4115
   pragma Import (C, cusparseZcsc2dense, "cusparseZcsc2dense");

  -- Description: This routine compresses the indecis of rows or columns.
  --   It can be interpreted as a conversion from COO to CSR sparse storage
  --   format.  

   function cusparseXcoo2csr
     (handle : cusparseHandle_t;
      cooRowInd : access int;
      nnz : int;
      m : int;
      csrSortedRowPtr : access int;
      idxBase : cusparseIndexBase_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:4128
   pragma Import (C, cusparseXcoo2csr, "cusparseXcoo2csr");

  -- Description: This routine uncompresses the indecis of rows or columns.
  --   It can be interpreted as a conversion from CSR to COO sparse storage
  --   format.  

   function cusparseXcsr2coo
     (handle : cusparseHandle_t;
      csrSortedRowPtr : access int;
      nnz : int;
      m : int;
      cooRowInd : access int;
      idxBase : cusparseIndexBase_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:4138
   pragma Import (C, cusparseXcsr2coo, "cusparseXcsr2coo");

  -- Description: This routine converts a matrix from CSR to CSC sparse 
  --   storage format. The resulting matrix can be re-interpreted as a 
  --   transpose of the original matrix in CSR storage format.  

   function cusparseCsr2cscEx
     (handle : cusparseHandle_t;
      m : int;
      n : int;
      nnz : int;
      csrSortedVal : System.Address;
      csrSortedValtype : library_types_h.cudaDataType;
      csrSortedRowPtr : access int;
      csrSortedColInd : access int;
      cscSortedVal : System.Address;
      cscSortedValtype : library_types_h.cudaDataType;
      cscSortedRowInd : access int;
      cscSortedColPtr : access int;
      copyValues : cusparseAction_t;
      idxBase : cusparseIndexBase_t;
      executiontype : library_types_h.cudaDataType) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:4148
   pragma Import (C, cusparseCsr2cscEx, "cusparseCsr2cscEx");

   function cusparseScsr2csc
     (handle : cusparseHandle_t;
      m : int;
      n : int;
      nnz : int;
      csrSortedVal : access float;
      csrSortedRowPtr : access int;
      csrSortedColInd : access int;
      cscSortedVal : access float;
      cscSortedRowInd : access int;
      cscSortedColPtr : access int;
      copyValues : cusparseAction_t;
      idxBase : cusparseIndexBase_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:4164
   pragma Import (C, cusparseScsr2csc, "cusparseScsr2csc");

   function cusparseDcsr2csc
     (handle : cusparseHandle_t;
      m : int;
      n : int;
      nnz : int;
      csrSortedVal : access double;
      csrSortedRowPtr : access int;
      csrSortedColInd : access int;
      cscSortedVal : access double;
      cscSortedRowInd : access int;
      cscSortedColPtr : access int;
      copyValues : cusparseAction_t;
      idxBase : cusparseIndexBase_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:4177
   pragma Import (C, cusparseDcsr2csc, "cusparseDcsr2csc");

   function cusparseCcsr2csc
     (handle : cusparseHandle_t;
      m : int;
      n : int;
      nnz : int;
      csrSortedVal : access constant cuComplex_h.cuComplex;
      csrSortedRowPtr : access int;
      csrSortedColInd : access int;
      cscSortedVal : access cuComplex_h.cuComplex;
      cscSortedRowInd : access int;
      cscSortedColPtr : access int;
      copyValues : cusparseAction_t;
      idxBase : cusparseIndexBase_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:4190
   pragma Import (C, cusparseCcsr2csc, "cusparseCcsr2csc");

   function cusparseZcsr2csc
     (handle : cusparseHandle_t;
      m : int;
      n : int;
      nnz : int;
      csrSortedVal : access constant cuComplex_h.cuDoubleComplex;
      csrSortedRowPtr : access int;
      csrSortedColInd : access int;
      cscSortedVal : access cuComplex_h.cuDoubleComplex;
      cscSortedRowInd : access int;
      cscSortedColPtr : access int;
      copyValues : cusparseAction_t;
      idxBase : cusparseIndexBase_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:4203
   pragma Import (C, cusparseZcsr2csc, "cusparseZcsr2csc");

  -- Description: This routine converts a dense matrix to a sparse matrix 
  --   in HYB storage format.  

   function cusparseSdense2hyb
     (handle : cusparseHandle_t;
      m : int;
      n : int;
      descrA : cusparseMatDescr_t;
      A : access float;
      lda : int;
      nnzPerRow : access int;
      hybA : cusparseHybMat_t;
      userEllWidth : int;
      partitionType : cusparseHybPartition_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:4218
   pragma Import (C, cusparseSdense2hyb, "cusparseSdense2hyb");

   function cusparseDdense2hyb
     (handle : cusparseHandle_t;
      m : int;
      n : int;
      descrA : cusparseMatDescr_t;
      A : access double;
      lda : int;
      nnzPerRow : access int;
      hybA : cusparseHybMat_t;
      userEllWidth : int;
      partitionType : cusparseHybPartition_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:4229
   pragma Import (C, cusparseDdense2hyb, "cusparseDdense2hyb");

   function cusparseCdense2hyb
     (handle : cusparseHandle_t;
      m : int;
      n : int;
      descrA : cusparseMatDescr_t;
      A : access constant cuComplex_h.cuComplex;
      lda : int;
      nnzPerRow : access int;
      hybA : cusparseHybMat_t;
      userEllWidth : int;
      partitionType : cusparseHybPartition_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:4240
   pragma Import (C, cusparseCdense2hyb, "cusparseCdense2hyb");

   function cusparseZdense2hyb
     (handle : cusparseHandle_t;
      m : int;
      n : int;
      descrA : cusparseMatDescr_t;
      A : access constant cuComplex_h.cuDoubleComplex;
      lda : int;
      nnzPerRow : access int;
      hybA : cusparseHybMat_t;
      userEllWidth : int;
      partitionType : cusparseHybPartition_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:4251
   pragma Import (C, cusparseZdense2hyb, "cusparseZdense2hyb");

  -- Description: This routine converts a sparse matrix in HYB storage format
  --   to a dense matrix.  

   function cusparseShyb2dense
     (handle : cusparseHandle_t;
      descrA : cusparseMatDescr_t;
      hybA : cusparseHybMat_t;
      A : access float;
      lda : int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:4264
   pragma Import (C, cusparseShyb2dense, "cusparseShyb2dense");

   function cusparseDhyb2dense
     (handle : cusparseHandle_t;
      descrA : cusparseMatDescr_t;
      hybA : cusparseHybMat_t;
      A : access double;
      lda : int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:4270
   pragma Import (C, cusparseDhyb2dense, "cusparseDhyb2dense");

   function cusparseChyb2dense
     (handle : cusparseHandle_t;
      descrA : cusparseMatDescr_t;
      hybA : cusparseHybMat_t;
      A : access cuComplex_h.cuComplex;
      lda : int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:4276
   pragma Import (C, cusparseChyb2dense, "cusparseChyb2dense");

   function cusparseZhyb2dense
     (handle : cusparseHandle_t;
      descrA : cusparseMatDescr_t;
      hybA : cusparseHybMat_t;
      A : access cuComplex_h.cuDoubleComplex;
      lda : int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:4282
   pragma Import (C, cusparseZhyb2dense, "cusparseZhyb2dense");

  -- Description: This routine converts a sparse matrix in CSR storage format
  --   to a sparse matrix in HYB storage format.  

   function cusparseScsr2hyb
     (handle : cusparseHandle_t;
      m : int;
      n : int;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access float;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      hybA : cusparseHybMat_t;
      userEllWidth : int;
      partitionType : cusparseHybPartition_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:4290
   pragma Import (C, cusparseScsr2hyb, "cusparseScsr2hyb");

   function cusparseDcsr2hyb
     (handle : cusparseHandle_t;
      m : int;
      n : int;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access double;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      hybA : cusparseHybMat_t;
      userEllWidth : int;
      partitionType : cusparseHybPartition_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:4301
   pragma Import (C, cusparseDcsr2hyb, "cusparseDcsr2hyb");

   function cusparseCcsr2hyb
     (handle : cusparseHandle_t;
      m : int;
      n : int;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access constant cuComplex_h.cuComplex;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      hybA : cusparseHybMat_t;
      userEllWidth : int;
      partitionType : cusparseHybPartition_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:4312
   pragma Import (C, cusparseCcsr2hyb, "cusparseCcsr2hyb");

   function cusparseZcsr2hyb
     (handle : cusparseHandle_t;
      m : int;
      n : int;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access constant cuComplex_h.cuDoubleComplex;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      hybA : cusparseHybMat_t;
      userEllWidth : int;
      partitionType : cusparseHybPartition_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:4323
   pragma Import (C, cusparseZcsr2hyb, "cusparseZcsr2hyb");

  -- Description: This routine converts a sparse matrix in HYB storage format
  --   to a sparse matrix in CSR storage format.  

   function cusparseShyb2csr
     (handle : cusparseHandle_t;
      descrA : cusparseMatDescr_t;
      hybA : cusparseHybMat_t;
      csrSortedValA : access float;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:4336
   pragma Import (C, cusparseShyb2csr, "cusparseShyb2csr");

   function cusparseDhyb2csr
     (handle : cusparseHandle_t;
      descrA : cusparseMatDescr_t;
      hybA : cusparseHybMat_t;
      csrSortedValA : access double;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:4343
   pragma Import (C, cusparseDhyb2csr, "cusparseDhyb2csr");

   function cusparseChyb2csr
     (handle : cusparseHandle_t;
      descrA : cusparseMatDescr_t;
      hybA : cusparseHybMat_t;
      csrSortedValA : access cuComplex_h.cuComplex;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:4350
   pragma Import (C, cusparseChyb2csr, "cusparseChyb2csr");

   function cusparseZhyb2csr
     (handle : cusparseHandle_t;
      descrA : cusparseMatDescr_t;
      hybA : cusparseHybMat_t;
      csrSortedValA : access cuComplex_h.cuDoubleComplex;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:4357
   pragma Import (C, cusparseZhyb2csr, "cusparseZhyb2csr");

  -- Description: This routine converts a sparse matrix in CSC storage format
  --   to a sparse matrix in HYB storage format.  

   function cusparseScsc2hyb
     (handle : cusparseHandle_t;
      m : int;
      n : int;
      descrA : cusparseMatDescr_t;
      cscSortedValA : access float;
      cscSortedRowIndA : access int;
      cscSortedColPtrA : access int;
      hybA : cusparseHybMat_t;
      userEllWidth : int;
      partitionType : cusparseHybPartition_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:4366
   pragma Import (C, cusparseScsc2hyb, "cusparseScsc2hyb");

   function cusparseDcsc2hyb
     (handle : cusparseHandle_t;
      m : int;
      n : int;
      descrA : cusparseMatDescr_t;
      cscSortedValA : access double;
      cscSortedRowIndA : access int;
      cscSortedColPtrA : access int;
      hybA : cusparseHybMat_t;
      userEllWidth : int;
      partitionType : cusparseHybPartition_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:4377
   pragma Import (C, cusparseDcsc2hyb, "cusparseDcsc2hyb");

   function cusparseCcsc2hyb
     (handle : cusparseHandle_t;
      m : int;
      n : int;
      descrA : cusparseMatDescr_t;
      cscSortedValA : access constant cuComplex_h.cuComplex;
      cscSortedRowIndA : access int;
      cscSortedColPtrA : access int;
      hybA : cusparseHybMat_t;
      userEllWidth : int;
      partitionType : cusparseHybPartition_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:4388
   pragma Import (C, cusparseCcsc2hyb, "cusparseCcsc2hyb");

   function cusparseZcsc2hyb
     (handle : cusparseHandle_t;
      m : int;
      n : int;
      descrA : cusparseMatDescr_t;
      cscSortedValA : access constant cuComplex_h.cuDoubleComplex;
      cscSortedRowIndA : access int;
      cscSortedColPtrA : access int;
      hybA : cusparseHybMat_t;
      userEllWidth : int;
      partitionType : cusparseHybPartition_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:4399
   pragma Import (C, cusparseZcsc2hyb, "cusparseZcsc2hyb");

  -- Description: This routine converts a sparse matrix in HYB storage format
  --   to a sparse matrix in CSC storage format.  

   function cusparseShyb2csc
     (handle : cusparseHandle_t;
      descrA : cusparseMatDescr_t;
      hybA : cusparseHybMat_t;
      cscSortedVal : access float;
      cscSortedRowInd : access int;
      cscSortedColPtr : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:4412
   pragma Import (C, cusparseShyb2csc, "cusparseShyb2csc");

   function cusparseDhyb2csc
     (handle : cusparseHandle_t;
      descrA : cusparseMatDescr_t;
      hybA : cusparseHybMat_t;
      cscSortedVal : access double;
      cscSortedRowInd : access int;
      cscSortedColPtr : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:4419
   pragma Import (C, cusparseDhyb2csc, "cusparseDhyb2csc");

   function cusparseChyb2csc
     (handle : cusparseHandle_t;
      descrA : cusparseMatDescr_t;
      hybA : cusparseHybMat_t;
      cscSortedVal : access cuComplex_h.cuComplex;
      cscSortedRowInd : access int;
      cscSortedColPtr : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:4426
   pragma Import (C, cusparseChyb2csc, "cusparseChyb2csc");

   function cusparseZhyb2csc
     (handle : cusparseHandle_t;
      descrA : cusparseMatDescr_t;
      hybA : cusparseHybMat_t;
      cscSortedVal : access cuComplex_h.cuDoubleComplex;
      cscSortedRowInd : access int;
      cscSortedColPtr : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:4433
   pragma Import (C, cusparseZhyb2csc, "cusparseZhyb2csc");

  -- Description: This routine converts a sparse matrix in CSR storage format
  --   to a sparse matrix in block-CSR storage format.  

   function cusparseXcsr2bsrNnz
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      m : int;
      n : int;
      descrA : cusparseMatDescr_t;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      blockDim : int;
      descrC : cusparseMatDescr_t;
      bsrSortedRowPtrC : access int;
      nnzTotalDevHostPtr : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:4442
   pragma Import (C, cusparseXcsr2bsrNnz, "cusparseXcsr2bsrNnz");

   function cusparseScsr2bsr
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      m : int;
      n : int;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access float;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      blockDim : int;
      descrC : cusparseMatDescr_t;
      bsrSortedValC : access float;
      bsrSortedRowPtrC : access int;
      bsrSortedColIndC : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:4454
   pragma Import (C, cusparseScsr2bsr, "cusparseScsr2bsr");

   function cusparseDcsr2bsr
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      m : int;
      n : int;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access double;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      blockDim : int;
      descrC : cusparseMatDescr_t;
      bsrSortedValC : access double;
      bsrSortedRowPtrC : access int;
      bsrSortedColIndC : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:4468
   pragma Import (C, cusparseDcsr2bsr, "cusparseDcsr2bsr");

   function cusparseCcsr2bsr
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      m : int;
      n : int;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access constant cuComplex_h.cuComplex;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      blockDim : int;
      descrC : cusparseMatDescr_t;
      bsrSortedValC : access cuComplex_h.cuComplex;
      bsrSortedRowPtrC : access int;
      bsrSortedColIndC : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:4482
   pragma Import (C, cusparseCcsr2bsr, "cusparseCcsr2bsr");

   function cusparseZcsr2bsr
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      m : int;
      n : int;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access constant cuComplex_h.cuDoubleComplex;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      blockDim : int;
      descrC : cusparseMatDescr_t;
      bsrSortedValC : access cuComplex_h.cuDoubleComplex;
      bsrSortedRowPtrC : access int;
      bsrSortedColIndC : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:4496
   pragma Import (C, cusparseZcsr2bsr, "cusparseZcsr2bsr");

  -- Description: This routine converts a sparse matrix in block-CSR storage format
  --   to a sparse matrix in CSR storage format.  

   function cusparseSbsr2csr
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      mb : int;
      nb : int;
      descrA : cusparseMatDescr_t;
      bsrSortedValA : access float;
      bsrSortedRowPtrA : access int;
      bsrSortedColIndA : access int;
      blockDim : int;
      descrC : cusparseMatDescr_t;
      csrSortedValC : access float;
      csrSortedRowPtrC : access int;
      csrSortedColIndC : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:4512
   pragma Import (C, cusparseSbsr2csr, "cusparseSbsr2csr");

   function cusparseDbsr2csr
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      mb : int;
      nb : int;
      descrA : cusparseMatDescr_t;
      bsrSortedValA : access double;
      bsrSortedRowPtrA : access int;
      bsrSortedColIndA : access int;
      blockDim : int;
      descrC : cusparseMatDescr_t;
      csrSortedValC : access double;
      csrSortedRowPtrC : access int;
      csrSortedColIndC : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:4526
   pragma Import (C, cusparseDbsr2csr, "cusparseDbsr2csr");

   function cusparseCbsr2csr
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      mb : int;
      nb : int;
      descrA : cusparseMatDescr_t;
      bsrSortedValA : access constant cuComplex_h.cuComplex;
      bsrSortedRowPtrA : access int;
      bsrSortedColIndA : access int;
      blockDim : int;
      descrC : cusparseMatDescr_t;
      csrSortedValC : access cuComplex_h.cuComplex;
      csrSortedRowPtrC : access int;
      csrSortedColIndC : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:4540
   pragma Import (C, cusparseCbsr2csr, "cusparseCbsr2csr");

   function cusparseZbsr2csr
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      mb : int;
      nb : int;
      descrA : cusparseMatDescr_t;
      bsrSortedValA : access constant cuComplex_h.cuDoubleComplex;
      bsrSortedRowPtrA : access int;
      bsrSortedColIndA : access int;
      blockDim : int;
      descrC : cusparseMatDescr_t;
      csrSortedValC : access cuComplex_h.cuDoubleComplex;
      csrSortedRowPtrC : access int;
      csrSortedColIndC : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:4554
   pragma Import (C, cusparseZbsr2csr, "cusparseZbsr2csr");

  -- Description: This routine converts a sparse matrix in general block-CSR storage format
  --   to a sparse matrix in general block-CSC storage format.  

   function cusparseSgebsr2gebsc_bufferSize
     (handle : cusparseHandle_t;
      mb : int;
      nb : int;
      nnzb : int;
      bsrSortedVal : access float;
      bsrSortedRowPtr : access int;
      bsrSortedColInd : access int;
      rowBlockDim : int;
      colBlockDim : int;
      pBufferSizeInBytes : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:4570
   pragma Import (C, cusparseSgebsr2gebsc_bufferSize, "cusparseSgebsr2gebsc_bufferSize");

   function cusparseDgebsr2gebsc_bufferSize
     (handle : cusparseHandle_t;
      mb : int;
      nb : int;
      nnzb : int;
      bsrSortedVal : access double;
      bsrSortedRowPtr : access int;
      bsrSortedColInd : access int;
      rowBlockDim : int;
      colBlockDim : int;
      pBufferSizeInBytes : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:4581
   pragma Import (C, cusparseDgebsr2gebsc_bufferSize, "cusparseDgebsr2gebsc_bufferSize");

   function cusparseCgebsr2gebsc_bufferSize
     (handle : cusparseHandle_t;
      mb : int;
      nb : int;
      nnzb : int;
      bsrSortedVal : access constant cuComplex_h.cuComplex;
      bsrSortedRowPtr : access int;
      bsrSortedColInd : access int;
      rowBlockDim : int;
      colBlockDim : int;
      pBufferSizeInBytes : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:4592
   pragma Import (C, cusparseCgebsr2gebsc_bufferSize, "cusparseCgebsr2gebsc_bufferSize");

   function cusparseZgebsr2gebsc_bufferSize
     (handle : cusparseHandle_t;
      mb : int;
      nb : int;
      nnzb : int;
      bsrSortedVal : access constant cuComplex_h.cuDoubleComplex;
      bsrSortedRowPtr : access int;
      bsrSortedColInd : access int;
      rowBlockDim : int;
      colBlockDim : int;
      pBufferSizeInBytes : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:4603
   pragma Import (C, cusparseZgebsr2gebsc_bufferSize, "cusparseZgebsr2gebsc_bufferSize");

   function cusparseSgebsr2gebsc_bufferSizeExt
     (handle : cusparseHandle_t;
      mb : int;
      nb : int;
      nnzb : int;
      bsrSortedVal : access float;
      bsrSortedRowPtr : access int;
      bsrSortedColInd : access int;
      rowBlockDim : int;
      colBlockDim : int;
      pBufferSize : access stddef_h.size_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:4614
   pragma Import (C, cusparseSgebsr2gebsc_bufferSizeExt, "cusparseSgebsr2gebsc_bufferSizeExt");

   function cusparseDgebsr2gebsc_bufferSizeExt
     (handle : cusparseHandle_t;
      mb : int;
      nb : int;
      nnzb : int;
      bsrSortedVal : access double;
      bsrSortedRowPtr : access int;
      bsrSortedColInd : access int;
      rowBlockDim : int;
      colBlockDim : int;
      pBufferSize : access stddef_h.size_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:4625
   pragma Import (C, cusparseDgebsr2gebsc_bufferSizeExt, "cusparseDgebsr2gebsc_bufferSizeExt");

   function cusparseCgebsr2gebsc_bufferSizeExt
     (handle : cusparseHandle_t;
      mb : int;
      nb : int;
      nnzb : int;
      bsrSortedVal : access constant cuComplex_h.cuComplex;
      bsrSortedRowPtr : access int;
      bsrSortedColInd : access int;
      rowBlockDim : int;
      colBlockDim : int;
      pBufferSize : access stddef_h.size_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:4636
   pragma Import (C, cusparseCgebsr2gebsc_bufferSizeExt, "cusparseCgebsr2gebsc_bufferSizeExt");

   function cusparseZgebsr2gebsc_bufferSizeExt
     (handle : cusparseHandle_t;
      mb : int;
      nb : int;
      nnzb : int;
      bsrSortedVal : access constant cuComplex_h.cuDoubleComplex;
      bsrSortedRowPtr : access int;
      bsrSortedColInd : access int;
      rowBlockDim : int;
      colBlockDim : int;
      pBufferSize : access stddef_h.size_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:4647
   pragma Import (C, cusparseZgebsr2gebsc_bufferSizeExt, "cusparseZgebsr2gebsc_bufferSizeExt");

   function cusparseSgebsr2gebsc
     (handle : cusparseHandle_t;
      mb : int;
      nb : int;
      nnzb : int;
      bsrSortedVal : access float;
      bsrSortedRowPtr : access int;
      bsrSortedColInd : access int;
      rowBlockDim : int;
      colBlockDim : int;
      bscVal : access float;
      bscRowInd : access int;
      bscColPtr : access int;
      copyValues : cusparseAction_t;
      baseIdx : cusparseIndexBase_t;
      pBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:4659
   pragma Import (C, cusparseSgebsr2gebsc, "cusparseSgebsr2gebsc");

   function cusparseDgebsr2gebsc
     (handle : cusparseHandle_t;
      mb : int;
      nb : int;
      nnzb : int;
      bsrSortedVal : access double;
      bsrSortedRowPtr : access int;
      bsrSortedColInd : access int;
      rowBlockDim : int;
      colBlockDim : int;
      bscVal : access double;
      bscRowInd : access int;
      bscColPtr : access int;
      copyValues : cusparseAction_t;
      baseIdx : cusparseIndexBase_t;
      pBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:4675
   pragma Import (C, cusparseDgebsr2gebsc, "cusparseDgebsr2gebsc");

   function cusparseCgebsr2gebsc
     (handle : cusparseHandle_t;
      mb : int;
      nb : int;
      nnzb : int;
      bsrSortedVal : access constant cuComplex_h.cuComplex;
      bsrSortedRowPtr : access int;
      bsrSortedColInd : access int;
      rowBlockDim : int;
      colBlockDim : int;
      bscVal : access cuComplex_h.cuComplex;
      bscRowInd : access int;
      bscColPtr : access int;
      copyValues : cusparseAction_t;
      baseIdx : cusparseIndexBase_t;
      pBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:4691
   pragma Import (C, cusparseCgebsr2gebsc, "cusparseCgebsr2gebsc");

   function cusparseZgebsr2gebsc
     (handle : cusparseHandle_t;
      mb : int;
      nb : int;
      nnzb : int;
      bsrSortedVal : access constant cuComplex_h.cuDoubleComplex;
      bsrSortedRowPtr : access int;
      bsrSortedColInd : access int;
      rowBlockDim : int;
      colBlockDim : int;
      bscVal : access cuComplex_h.cuDoubleComplex;
      bscRowInd : access int;
      bscColPtr : access int;
      copyValues : cusparseAction_t;
      baseIdx : cusparseIndexBase_t;
      pBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:4707
   pragma Import (C, cusparseZgebsr2gebsc, "cusparseZgebsr2gebsc");

  -- Description: This routine converts a sparse matrix in general block-CSR storage format
  --   to a sparse matrix in CSR storage format.  

   function cusparseXgebsr2csr
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      mb : int;
      nb : int;
      descrA : cusparseMatDescr_t;
      bsrSortedRowPtrA : access int;
      bsrSortedColIndA : access int;
      rowBlockDim : int;
      colBlockDim : int;
      descrC : cusparseMatDescr_t;
      csrSortedRowPtrC : access int;
      csrSortedColIndC : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:4725
   pragma Import (C, cusparseXgebsr2csr, "cusparseXgebsr2csr");

   function cusparseSgebsr2csr
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      mb : int;
      nb : int;
      descrA : cusparseMatDescr_t;
      bsrSortedValA : access float;
      bsrSortedRowPtrA : access int;
      bsrSortedColIndA : access int;
      rowBlockDim : int;
      colBlockDim : int;
      descrC : cusparseMatDescr_t;
      csrSortedValC : access float;
      csrSortedRowPtrC : access int;
      csrSortedColIndC : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:4738
   pragma Import (C, cusparseSgebsr2csr, "cusparseSgebsr2csr");

   function cusparseDgebsr2csr
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      mb : int;
      nb : int;
      descrA : cusparseMatDescr_t;
      bsrSortedValA : access double;
      bsrSortedRowPtrA : access int;
      bsrSortedColIndA : access int;
      rowBlockDim : int;
      colBlockDim : int;
      descrC : cusparseMatDescr_t;
      csrSortedValC : access double;
      csrSortedRowPtrC : access int;
      csrSortedColIndC : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:4754
   pragma Import (C, cusparseDgebsr2csr, "cusparseDgebsr2csr");

   function cusparseCgebsr2csr
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      mb : int;
      nb : int;
      descrA : cusparseMatDescr_t;
      bsrSortedValA : access constant cuComplex_h.cuComplex;
      bsrSortedRowPtrA : access int;
      bsrSortedColIndA : access int;
      rowBlockDim : int;
      colBlockDim : int;
      descrC : cusparseMatDescr_t;
      csrSortedValC : access cuComplex_h.cuComplex;
      csrSortedRowPtrC : access int;
      csrSortedColIndC : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:4770
   pragma Import (C, cusparseCgebsr2csr, "cusparseCgebsr2csr");

   function cusparseZgebsr2csr
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      mb : int;
      nb : int;
      descrA : cusparseMatDescr_t;
      bsrSortedValA : access constant cuComplex_h.cuDoubleComplex;
      bsrSortedRowPtrA : access int;
      bsrSortedColIndA : access int;
      rowBlockDim : int;
      colBlockDim : int;
      descrC : cusparseMatDescr_t;
      csrSortedValC : access cuComplex_h.cuDoubleComplex;
      csrSortedRowPtrC : access int;
      csrSortedColIndC : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:4786
   pragma Import (C, cusparseZgebsr2csr, "cusparseZgebsr2csr");

  -- Description: This routine converts a sparse matrix in CSR storage format
  --   to a sparse matrix in general block-CSR storage format.  

   function cusparseScsr2gebsr_bufferSize
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      m : int;
      n : int;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access float;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      rowBlockDim : int;
      colBlockDim : int;
      pBufferSizeInBytes : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:4803
   pragma Import (C, cusparseScsr2gebsr_bufferSize, "cusparseScsr2gebsr_bufferSize");

   function cusparseDcsr2gebsr_bufferSize
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      m : int;
      n : int;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access double;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      rowBlockDim : int;
      colBlockDim : int;
      pBufferSizeInBytes : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:4815
   pragma Import (C, cusparseDcsr2gebsr_bufferSize, "cusparseDcsr2gebsr_bufferSize");

   function cusparseCcsr2gebsr_bufferSize
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      m : int;
      n : int;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access constant cuComplex_h.cuComplex;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      rowBlockDim : int;
      colBlockDim : int;
      pBufferSizeInBytes : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:4827
   pragma Import (C, cusparseCcsr2gebsr_bufferSize, "cusparseCcsr2gebsr_bufferSize");

   function cusparseZcsr2gebsr_bufferSize
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      m : int;
      n : int;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access constant cuComplex_h.cuDoubleComplex;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      rowBlockDim : int;
      colBlockDim : int;
      pBufferSizeInBytes : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:4839
   pragma Import (C, cusparseZcsr2gebsr_bufferSize, "cusparseZcsr2gebsr_bufferSize");

   function cusparseScsr2gebsr_bufferSizeExt
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      m : int;
      n : int;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access float;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      rowBlockDim : int;
      colBlockDim : int;
      pBufferSize : access stddef_h.size_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:4851
   pragma Import (C, cusparseScsr2gebsr_bufferSizeExt, "cusparseScsr2gebsr_bufferSizeExt");

   function cusparseDcsr2gebsr_bufferSizeExt
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      m : int;
      n : int;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access double;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      rowBlockDim : int;
      colBlockDim : int;
      pBufferSize : access stddef_h.size_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:4863
   pragma Import (C, cusparseDcsr2gebsr_bufferSizeExt, "cusparseDcsr2gebsr_bufferSizeExt");

   function cusparseCcsr2gebsr_bufferSizeExt
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      m : int;
      n : int;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access constant cuComplex_h.cuComplex;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      rowBlockDim : int;
      colBlockDim : int;
      pBufferSize : access stddef_h.size_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:4875
   pragma Import (C, cusparseCcsr2gebsr_bufferSizeExt, "cusparseCcsr2gebsr_bufferSizeExt");

   function cusparseZcsr2gebsr_bufferSizeExt
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      m : int;
      n : int;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access constant cuComplex_h.cuDoubleComplex;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      rowBlockDim : int;
      colBlockDim : int;
      pBufferSize : access stddef_h.size_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:4887
   pragma Import (C, cusparseZcsr2gebsr_bufferSizeExt, "cusparseZcsr2gebsr_bufferSizeExt");

   function cusparseXcsr2gebsrNnz
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      m : int;
      n : int;
      descrA : cusparseMatDescr_t;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      descrC : cusparseMatDescr_t;
      bsrSortedRowPtrC : access int;
      rowBlockDim : int;
      colBlockDim : int;
      nnzTotalDevHostPtr : access int;
      pBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:4901
   pragma Import (C, cusparseXcsr2gebsrNnz, "cusparseXcsr2gebsrNnz");

   function cusparseScsr2gebsr
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      m : int;
      n : int;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access float;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      descrC : cusparseMatDescr_t;
      bsrSortedValC : access float;
      bsrSortedRowPtrC : access int;
      bsrSortedColIndC : access int;
      rowBlockDim : int;
      colBlockDim : int;
      pBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:4915
   pragma Import (C, cusparseScsr2gebsr, "cusparseScsr2gebsr");

   function cusparseDcsr2gebsr
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      m : int;
      n : int;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access double;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      descrC : cusparseMatDescr_t;
      bsrSortedValC : access double;
      bsrSortedRowPtrC : access int;
      bsrSortedColIndC : access int;
      rowBlockDim : int;
      colBlockDim : int;
      pBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:4931
   pragma Import (C, cusparseDcsr2gebsr, "cusparseDcsr2gebsr");

   function cusparseCcsr2gebsr
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      m : int;
      n : int;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access constant cuComplex_h.cuComplex;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      descrC : cusparseMatDescr_t;
      bsrSortedValC : access cuComplex_h.cuComplex;
      bsrSortedRowPtrC : access int;
      bsrSortedColIndC : access int;
      rowBlockDim : int;
      colBlockDim : int;
      pBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:4947
   pragma Import (C, cusparseCcsr2gebsr, "cusparseCcsr2gebsr");

   function cusparseZcsr2gebsr
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      m : int;
      n : int;
      descrA : cusparseMatDescr_t;
      csrSortedValA : access constant cuComplex_h.cuDoubleComplex;
      csrSortedRowPtrA : access int;
      csrSortedColIndA : access int;
      descrC : cusparseMatDescr_t;
      bsrSortedValC : access cuComplex_h.cuDoubleComplex;
      bsrSortedRowPtrC : access int;
      bsrSortedColIndC : access int;
      rowBlockDim : int;
      colBlockDim : int;
      pBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:4963
   pragma Import (C, cusparseZcsr2gebsr, "cusparseZcsr2gebsr");

  -- Description: This routine converts a sparse matrix in general block-CSR storage format
  --   to a sparse matrix in general block-CSR storage format with different block size.  

   function cusparseSgebsr2gebsr_bufferSize
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      mb : int;
      nb : int;
      nnzb : int;
      descrA : cusparseMatDescr_t;
      bsrSortedValA : access float;
      bsrSortedRowPtrA : access int;
      bsrSortedColIndA : access int;
      rowBlockDimA : int;
      colBlockDimA : int;
      rowBlockDimC : int;
      colBlockDimC : int;
      pBufferSizeInBytes : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:4981
   pragma Import (C, cusparseSgebsr2gebsr_bufferSize, "cusparseSgebsr2gebsr_bufferSize");

   function cusparseDgebsr2gebsr_bufferSize
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      mb : int;
      nb : int;
      nnzb : int;
      descrA : cusparseMatDescr_t;
      bsrSortedValA : access double;
      bsrSortedRowPtrA : access int;
      bsrSortedColIndA : access int;
      rowBlockDimA : int;
      colBlockDimA : int;
      rowBlockDimC : int;
      colBlockDimC : int;
      pBufferSizeInBytes : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:4996
   pragma Import (C, cusparseDgebsr2gebsr_bufferSize, "cusparseDgebsr2gebsr_bufferSize");

   function cusparseCgebsr2gebsr_bufferSize
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      mb : int;
      nb : int;
      nnzb : int;
      descrA : cusparseMatDescr_t;
      bsrSortedValA : access constant cuComplex_h.cuComplex;
      bsrSortedRowPtrA : access int;
      bsrSortedColIndA : access int;
      rowBlockDimA : int;
      colBlockDimA : int;
      rowBlockDimC : int;
      colBlockDimC : int;
      pBufferSizeInBytes : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:5011
   pragma Import (C, cusparseCgebsr2gebsr_bufferSize, "cusparseCgebsr2gebsr_bufferSize");

   function cusparseZgebsr2gebsr_bufferSize
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      mb : int;
      nb : int;
      nnzb : int;
      descrA : cusparseMatDescr_t;
      bsrSortedValA : access constant cuComplex_h.cuDoubleComplex;
      bsrSortedRowPtrA : access int;
      bsrSortedColIndA : access int;
      rowBlockDimA : int;
      colBlockDimA : int;
      rowBlockDimC : int;
      colBlockDimC : int;
      pBufferSizeInBytes : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:5026
   pragma Import (C, cusparseZgebsr2gebsr_bufferSize, "cusparseZgebsr2gebsr_bufferSize");

   function cusparseSgebsr2gebsr_bufferSizeExt
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      mb : int;
      nb : int;
      nnzb : int;
      descrA : cusparseMatDescr_t;
      bsrSortedValA : access float;
      bsrSortedRowPtrA : access int;
      bsrSortedColIndA : access int;
      rowBlockDimA : int;
      colBlockDimA : int;
      rowBlockDimC : int;
      colBlockDimC : int;
      pBufferSize : access stddef_h.size_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:5042
   pragma Import (C, cusparseSgebsr2gebsr_bufferSizeExt, "cusparseSgebsr2gebsr_bufferSizeExt");

   function cusparseDgebsr2gebsr_bufferSizeExt
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      mb : int;
      nb : int;
      nnzb : int;
      descrA : cusparseMatDescr_t;
      bsrSortedValA : access double;
      bsrSortedRowPtrA : access int;
      bsrSortedColIndA : access int;
      rowBlockDimA : int;
      colBlockDimA : int;
      rowBlockDimC : int;
      colBlockDimC : int;
      pBufferSize : access stddef_h.size_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:5057
   pragma Import (C, cusparseDgebsr2gebsr_bufferSizeExt, "cusparseDgebsr2gebsr_bufferSizeExt");

   function cusparseCgebsr2gebsr_bufferSizeExt
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      mb : int;
      nb : int;
      nnzb : int;
      descrA : cusparseMatDescr_t;
      bsrSortedValA : access constant cuComplex_h.cuComplex;
      bsrSortedRowPtrA : access int;
      bsrSortedColIndA : access int;
      rowBlockDimA : int;
      colBlockDimA : int;
      rowBlockDimC : int;
      colBlockDimC : int;
      pBufferSize : access stddef_h.size_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:5072
   pragma Import (C, cusparseCgebsr2gebsr_bufferSizeExt, "cusparseCgebsr2gebsr_bufferSizeExt");

   function cusparseZgebsr2gebsr_bufferSizeExt
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      mb : int;
      nb : int;
      nnzb : int;
      descrA : cusparseMatDescr_t;
      bsrSortedValA : access constant cuComplex_h.cuDoubleComplex;
      bsrSortedRowPtrA : access int;
      bsrSortedColIndA : access int;
      rowBlockDimA : int;
      colBlockDimA : int;
      rowBlockDimC : int;
      colBlockDimC : int;
      pBufferSize : access stddef_h.size_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:5087
   pragma Import (C, cusparseZgebsr2gebsr_bufferSizeExt, "cusparseZgebsr2gebsr_bufferSizeExt");

   function cusparseXgebsr2gebsrNnz
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      mb : int;
      nb : int;
      nnzb : int;
      descrA : cusparseMatDescr_t;
      bsrSortedRowPtrA : access int;
      bsrSortedColIndA : access int;
      rowBlockDimA : int;
      colBlockDimA : int;
      descrC : cusparseMatDescr_t;
      bsrSortedRowPtrC : access int;
      rowBlockDimC : int;
      colBlockDimC : int;
      nnzTotalDevHostPtr : access int;
      pBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:5104
   pragma Import (C, cusparseXgebsr2gebsrNnz, "cusparseXgebsr2gebsrNnz");

   function cusparseSgebsr2gebsr
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      mb : int;
      nb : int;
      nnzb : int;
      descrA : cusparseMatDescr_t;
      bsrSortedValA : access float;
      bsrSortedRowPtrA : access int;
      bsrSortedColIndA : access int;
      rowBlockDimA : int;
      colBlockDimA : int;
      descrC : cusparseMatDescr_t;
      bsrSortedValC : access float;
      bsrSortedRowPtrC : access int;
      bsrSortedColIndC : access int;
      rowBlockDimC : int;
      colBlockDimC : int;
      pBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:5121
   pragma Import (C, cusparseSgebsr2gebsr, "cusparseSgebsr2gebsr");

   function cusparseDgebsr2gebsr
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      mb : int;
      nb : int;
      nnzb : int;
      descrA : cusparseMatDescr_t;
      bsrSortedValA : access double;
      bsrSortedRowPtrA : access int;
      bsrSortedColIndA : access int;
      rowBlockDimA : int;
      colBlockDimA : int;
      descrC : cusparseMatDescr_t;
      bsrSortedValC : access double;
      bsrSortedRowPtrC : access int;
      bsrSortedColIndC : access int;
      rowBlockDimC : int;
      colBlockDimC : int;
      pBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:5140
   pragma Import (C, cusparseDgebsr2gebsr, "cusparseDgebsr2gebsr");

   function cusparseCgebsr2gebsr
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      mb : int;
      nb : int;
      nnzb : int;
      descrA : cusparseMatDescr_t;
      bsrSortedValA : access constant cuComplex_h.cuComplex;
      bsrSortedRowPtrA : access int;
      bsrSortedColIndA : access int;
      rowBlockDimA : int;
      colBlockDimA : int;
      descrC : cusparseMatDescr_t;
      bsrSortedValC : access cuComplex_h.cuComplex;
      bsrSortedRowPtrC : access int;
      bsrSortedColIndC : access int;
      rowBlockDimC : int;
      colBlockDimC : int;
      pBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:5159
   pragma Import (C, cusparseCgebsr2gebsr, "cusparseCgebsr2gebsr");

   function cusparseZgebsr2gebsr
     (handle : cusparseHandle_t;
      dirA : cusparseDirection_t;
      mb : int;
      nb : int;
      nnzb : int;
      descrA : cusparseMatDescr_t;
      bsrSortedValA : access constant cuComplex_h.cuDoubleComplex;
      bsrSortedRowPtrA : access int;
      bsrSortedColIndA : access int;
      rowBlockDimA : int;
      colBlockDimA : int;
      descrC : cusparseMatDescr_t;
      bsrSortedValC : access cuComplex_h.cuDoubleComplex;
      bsrSortedRowPtrC : access int;
      bsrSortedColIndC : access int;
      rowBlockDimC : int;
      colBlockDimC : int;
      pBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:5178
   pragma Import (C, cusparseZgebsr2gebsr, "cusparseZgebsr2gebsr");

  -- --- Sparse Matrix Sorting ---  
  -- Description: Create a identity sequence p=[0,1,...,n-1].  
   function cusparseCreateIdentityPermutation
     (handle : cusparseHandle_t;
      n : int;
      p : access int) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:5200
   pragma Import (C, cusparseCreateIdentityPermutation, "cusparseCreateIdentityPermutation");

  -- Description: Sort sparse matrix stored in COO format  
   function cusparseXcoosort_bufferSizeExt
     (handle : cusparseHandle_t;
      m : int;
      n : int;
      nnz : int;
      cooRowsA : access int;
      cooColsA : access int;
      pBufferSizeInBytes : access stddef_h.size_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:5205
   pragma Import (C, cusparseXcoosort_bufferSizeExt, "cusparseXcoosort_bufferSizeExt");

   function cusparseXcoosortByRow
     (handle : cusparseHandle_t;
      m : int;
      n : int;
      nnz : int;
      cooRowsA : access int;
      cooColsA : access int;
      P : access int;
      pBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:5213
   pragma Import (C, cusparseXcoosortByRow, "cusparseXcoosortByRow");

   function cusparseXcoosortByColumn
     (handle : cusparseHandle_t;
      m : int;
      n : int;
      nnz : int;
      cooRowsA : access int;
      cooColsA : access int;
      P : access int;
      pBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:5222
   pragma Import (C, cusparseXcoosortByColumn, "cusparseXcoosortByColumn");

  -- Description: Sort sparse matrix stored in CSR format  
   function cusparseXcsrsort_bufferSizeExt
     (handle : cusparseHandle_t;
      m : int;
      n : int;
      nnz : int;
      csrRowPtrA : access int;
      csrColIndA : access int;
      pBufferSizeInBytes : access stddef_h.size_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:5232
   pragma Import (C, cusparseXcsrsort_bufferSizeExt, "cusparseXcsrsort_bufferSizeExt");

   function cusparseXcsrsort
     (handle : cusparseHandle_t;
      m : int;
      n : int;
      nnz : int;
      descrA : cusparseMatDescr_t;
      csrRowPtrA : access int;
      csrColIndA : access int;
      P : access int;
      pBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:5240
   pragma Import (C, cusparseXcsrsort, "cusparseXcsrsort");

  -- Description: Sort sparse matrix stored in CSC format  
   function cusparseXcscsort_bufferSizeExt
     (handle : cusparseHandle_t;
      m : int;
      n : int;
      nnz : int;
      cscColPtrA : access int;
      cscRowIndA : access int;
      pBufferSizeInBytes : access stddef_h.size_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:5251
   pragma Import (C, cusparseXcscsort_bufferSizeExt, "cusparseXcscsort_bufferSizeExt");

   function cusparseXcscsort
     (handle : cusparseHandle_t;
      m : int;
      n : int;
      nnz : int;
      descrA : cusparseMatDescr_t;
      cscColPtrA : access int;
      cscRowIndA : access int;
      P : access int;
      pBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:5259
   pragma Import (C, cusparseXcscsort, "cusparseXcscsort");

  -- Description: Wrapper that sorts sparse matrix stored in CSR format 
  --   (without exposing the permutation).  

   function cusparseScsru2csr_bufferSizeExt
     (handle : cusparseHandle_t;
      m : int;
      n : int;
      nnz : int;
      csrVal : access float;
      csrRowPtr : access int;
      csrColInd : access int;
      info : csru2csrInfo_t;
      pBufferSizeInBytes : access stddef_h.size_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:5271
   pragma Import (C, cusparseScsru2csr_bufferSizeExt, "cusparseScsru2csr_bufferSizeExt");

   function cusparseDcsru2csr_bufferSizeExt
     (handle : cusparseHandle_t;
      m : int;
      n : int;
      nnz : int;
      csrVal : access double;
      csrRowPtr : access int;
      csrColInd : access int;
      info : csru2csrInfo_t;
      pBufferSizeInBytes : access stddef_h.size_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:5281
   pragma Import (C, cusparseDcsru2csr_bufferSizeExt, "cusparseDcsru2csr_bufferSizeExt");

   function cusparseCcsru2csr_bufferSizeExt
     (handle : cusparseHandle_t;
      m : int;
      n : int;
      nnz : int;
      csrVal : access cuComplex_h.cuComplex;
      csrRowPtr : access int;
      csrColInd : access int;
      info : csru2csrInfo_t;
      pBufferSizeInBytes : access stddef_h.size_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:5291
   pragma Import (C, cusparseCcsru2csr_bufferSizeExt, "cusparseCcsru2csr_bufferSizeExt");

   function cusparseZcsru2csr_bufferSizeExt
     (handle : cusparseHandle_t;
      m : int;
      n : int;
      nnz : int;
      csrVal : access cuComplex_h.cuDoubleComplex;
      csrRowPtr : access int;
      csrColInd : access int;
      info : csru2csrInfo_t;
      pBufferSizeInBytes : access stddef_h.size_t) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:5301
   pragma Import (C, cusparseZcsru2csr_bufferSizeExt, "cusparseZcsru2csr_bufferSizeExt");

   function cusparseScsru2csr
     (handle : cusparseHandle_t;
      m : int;
      n : int;
      nnz : int;
      descrA : cusparseMatDescr_t;
      csrVal : access float;
      csrRowPtr : access int;
      csrColInd : access int;
      info : csru2csrInfo_t;
      pBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:5311
   pragma Import (C, cusparseScsru2csr, "cusparseScsru2csr");

   function cusparseDcsru2csr
     (handle : cusparseHandle_t;
      m : int;
      n : int;
      nnz : int;
      descrA : cusparseMatDescr_t;
      csrVal : access double;
      csrRowPtr : access int;
      csrColInd : access int;
      info : csru2csrInfo_t;
      pBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:5322
   pragma Import (C, cusparseDcsru2csr, "cusparseDcsru2csr");

   function cusparseCcsru2csr
     (handle : cusparseHandle_t;
      m : int;
      n : int;
      nnz : int;
      descrA : cusparseMatDescr_t;
      csrVal : access cuComplex_h.cuComplex;
      csrRowPtr : access int;
      csrColInd : access int;
      info : csru2csrInfo_t;
      pBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:5333
   pragma Import (C, cusparseCcsru2csr, "cusparseCcsru2csr");

   function cusparseZcsru2csr
     (handle : cusparseHandle_t;
      m : int;
      n : int;
      nnz : int;
      descrA : cusparseMatDescr_t;
      csrVal : access cuComplex_h.cuDoubleComplex;
      csrRowPtr : access int;
      csrColInd : access int;
      info : csru2csrInfo_t;
      pBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:5344
   pragma Import (C, cusparseZcsru2csr, "cusparseZcsru2csr");

  -- Description: Wrapper that un-sorts sparse matrix stored in CSR format 
  --   (without exposing the permutation).  

   function cusparseScsr2csru
     (handle : cusparseHandle_t;
      m : int;
      n : int;
      nnz : int;
      descrA : cusparseMatDescr_t;
      csrVal : access float;
      csrRowPtr : access int;
      csrColInd : access int;
      info : csru2csrInfo_t;
      pBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:5357
   pragma Import (C, cusparseScsr2csru, "cusparseScsr2csru");

   function cusparseDcsr2csru
     (handle : cusparseHandle_t;
      m : int;
      n : int;
      nnz : int;
      descrA : cusparseMatDescr_t;
      csrVal : access double;
      csrRowPtr : access int;
      csrColInd : access int;
      info : csru2csrInfo_t;
      pBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:5368
   pragma Import (C, cusparseDcsr2csru, "cusparseDcsr2csru");

   function cusparseCcsr2csru
     (handle : cusparseHandle_t;
      m : int;
      n : int;
      nnz : int;
      descrA : cusparseMatDescr_t;
      csrVal : access cuComplex_h.cuComplex;
      csrRowPtr : access int;
      csrColInd : access int;
      info : csru2csrInfo_t;
      pBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:5379
   pragma Import (C, cusparseCcsr2csru, "cusparseCcsr2csru");

   function cusparseZcsr2csru
     (handle : cusparseHandle_t;
      m : int;
      n : int;
      nnz : int;
      descrA : cusparseMatDescr_t;
      csrVal : access cuComplex_h.cuDoubleComplex;
      csrRowPtr : access int;
      csrColInd : access int;
      info : csru2csrInfo_t;
      pBuffer : System.Address) return cusparseStatus_t;  -- /usr/local/cuda-8.0/include/cusparse.h:5390
   pragma Import (C, cusparseZcsr2csru, "cusparseZcsr2csru");

end cusparse_h;

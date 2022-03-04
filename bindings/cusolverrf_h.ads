pragma Ada_2005;
pragma Style_Checks (Off);

with Interfaces.C; use Interfaces.C;
with System;
with cusolver_common_h;

package cusolverRf_h is

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

  -- CUSOLVERRF mode  
  --default   
   type cusolverRfResetValuesFastMode_t is 
     (CUSOLVERRF_RESET_VALUES_FAST_MODE_OFF,
      CUSOLVERRF_RESET_VALUES_FAST_MODE_ON);
   pragma Convention (C, cusolverRfResetValuesFastMode_t);  -- /usr/local/cuda-8.0/include/cusolverRf.h:65

  -- CUSOLVERRF matrix format  
  --default   
   type cusolverRfMatrixFormat_t is 
     (CUSOLVERRF_MATRIX_FORMAT_CSR,
      CUSOLVERRF_MATRIX_FORMAT_CSC);
   pragma Convention (C, cusolverRfMatrixFormat_t);  -- /usr/local/cuda-8.0/include/cusolverRf.h:71

  -- CUSOLVERRF unit diagonal  
  --default   
   type cusolverRfUnitDiagonal_t is 
     (CUSOLVERRF_UNIT_DIAGONAL_STORED_L,
      CUSOLVERRF_UNIT_DIAGONAL_STORED_U,
      CUSOLVERRF_UNIT_DIAGONAL_ASSUMED_L,
      CUSOLVERRF_UNIT_DIAGONAL_ASSUMED_U);
   pragma Convention (C, cusolverRfUnitDiagonal_t);  -- /usr/local/cuda-8.0/include/cusolverRf.h:79

  -- CUSOLVERRF factorization algorithm  
  -- default
   type cusolverRfFactorization_t is 
     (CUSOLVERRF_FACTORIZATION_ALG0,
      CUSOLVERRF_FACTORIZATION_ALG1,
      CUSOLVERRF_FACTORIZATION_ALG2);
   pragma Convention (C, cusolverRfFactorization_t);  -- /usr/local/cuda-8.0/include/cusolverRf.h:86

  -- CUSOLVERRF triangular solve algorithm  
  -- default
   type cusolverRfTriangularSolve_t is 
     (CUSOLVERRF_TRIANGULAR_SOLVE_ALG0,
      CUSOLVERRF_TRIANGULAR_SOLVE_ALG1,
      CUSOLVERRF_TRIANGULAR_SOLVE_ALG2,
      CUSOLVERRF_TRIANGULAR_SOLVE_ALG3);
   pragma Convention (C, cusolverRfTriangularSolve_t);  -- /usr/local/cuda-8.0/include/cusolverRf.h:94

  -- CUSOLVERRF numeric boost report  
  --default
   type cusolverRfNumericBoostReport_t is 
     (CUSOLVERRF_NUMERIC_BOOST_NOT_USED,
      CUSOLVERRF_NUMERIC_BOOST_USED);
   pragma Convention (C, cusolverRfNumericBoostReport_t);  -- /usr/local/cuda-8.0/include/cusolverRf.h:100

  -- Opaque structure holding CUSOLVERRF library common  
   --  skipped empty struct cusolverRfCommon

   type cusolverRfHandle_t is new System.Address;  -- /usr/local/cuda-8.0/include/cusolverRf.h:104

  -- CUSOLVERRF create (allocate memory) and destroy (free memory) in the handle  
   function cusolverRfCreate (handle : System.Address) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverRf.h:107
   pragma Import (C, cusolverRfCreate, "cusolverRfCreate");

   function cusolverRfDestroy (handle : cusolverRfHandle_t) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverRf.h:108
   pragma Import (C, cusolverRfDestroy, "cusolverRfDestroy");

  -- CUSOLVERRF set and get input format  
   function cusolverRfGetMatrixFormat
     (handle : cusolverRfHandle_t;
      format : access cusolverRfMatrixFormat_t;
      diag : access cusolverRfUnitDiagonal_t) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverRf.h:111
   pragma Import (C, cusolverRfGetMatrixFormat, "cusolverRfGetMatrixFormat");

   function cusolverRfSetMatrixFormat
     (handle : cusolverRfHandle_t;
      format : cusolverRfMatrixFormat_t;
      diag : cusolverRfUnitDiagonal_t) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverRf.h:115
   pragma Import (C, cusolverRfSetMatrixFormat, "cusolverRfSetMatrixFormat");

  -- CUSOLVERRF set and get numeric properties  
   function cusolverRfSetNumericProperties
     (handle : cusolverRfHandle_t;
      zero : double;
      boost : double) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverRf.h:120
   pragma Import (C, cusolverRfSetNumericProperties, "cusolverRfSetNumericProperties");

   function cusolverRfGetNumericProperties
     (handle : cusolverRfHandle_t;
      zero : access double;
      boost : access double) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverRf.h:124
   pragma Import (C, cusolverRfGetNumericProperties, "cusolverRfGetNumericProperties");

   function cusolverRfGetNumericBoostReport (handle : cusolverRfHandle_t; report : access cusolverRfNumericBoostReport_t) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverRf.h:128
   pragma Import (C, cusolverRfGetNumericBoostReport, "cusolverRfGetNumericBoostReport");

  -- CUSOLVERRF choose the triangular solve algorithm  
   function cusolverRfSetAlgs
     (handle : cusolverRfHandle_t;
      factAlg : cusolverRfFactorization_t;
      solveAlg : cusolverRfTriangularSolve_t) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverRf.h:132
   pragma Import (C, cusolverRfSetAlgs, "cusolverRfSetAlgs");

   function cusolverRfGetAlgs
     (handle : cusolverRfHandle_t;
      factAlg : access cusolverRfFactorization_t;
      solveAlg : access cusolverRfTriangularSolve_t) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverRf.h:136
   pragma Import (C, cusolverRfGetAlgs, "cusolverRfGetAlgs");

  -- CUSOLVERRF set and get fast mode  
   function cusolverRfGetResetValuesFastMode (handle : cusolverRfHandle_t; fastMode : access cusolverRfResetValuesFastMode_t) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverRf.h:141
   pragma Import (C, cusolverRfGetResetValuesFastMode, "cusolverRfGetResetValuesFastMode");

   function cusolverRfSetResetValuesFastMode (handle : cusolverRfHandle_t; fastMode : cusolverRfResetValuesFastMode_t) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverRf.h:144
   pragma Import (C, cusolverRfSetResetValuesFastMode, "cusolverRfSetResetValuesFastMode");

  --** Non-Batched Routines ** 
  -- CUSOLVERRF setup of internal structures from host or device memory  
  -- Input (in the host memory)  
   function cusolverRfSetupHost
     (n : int;
      nnzA : int;
      h_csrRowPtrA : access int;
      h_csrColIndA : access int;
      h_csrValA : access double;
      nnzL : int;
      h_csrRowPtrL : access int;
      h_csrColIndL : access int;
      h_csrValL : access double;
      nnzU : int;
      h_csrRowPtrU : access int;
      h_csrColIndU : access int;
      h_csrValU : access double;
      h_P : access int;
      h_Q : access int;
      handle : cusolverRfHandle_t) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverRf.h:149
   pragma Import (C, cusolverRfSetupHost, "cusolverRfSetupHost");

  -- Output  
  -- Input (in the device memory)  
   function cusolverRfSetupDevice
     (n : int;
      nnzA : int;
      csrRowPtrA : access int;
      csrColIndA : access int;
      csrValA : access double;
      nnzL : int;
      csrRowPtrL : access int;
      csrColIndL : access int;
      csrValL : access double;
      nnzU : int;
      csrRowPtrU : access int;
      csrColIndU : access int;
      csrValU : access double;
      P : access int;
      Q : access int;
      handle : cusolverRfHandle_t) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverRf.h:168
   pragma Import (C, cusolverRfSetupDevice, "cusolverRfSetupDevice");

  -- Output  
  -- CUSOLVERRF update the matrix values (assuming the reordering, pivoting 
  --   and consequently the sparsity pattern of L and U did not change),
  --   and zero out the remaining values.  

  -- Input (in the device memory)  
   function cusolverRfResetValues
     (n : int;
      nnzA : int;
      csrRowPtrA : access int;
      csrColIndA : access int;
      csrValA : access double;
      P : access int;
      Q : access int;
      handle : cusolverRfHandle_t) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverRf.h:190
   pragma Import (C, cusolverRfResetValues, "cusolverRfResetValues");

  -- Output  
  -- CUSOLVERRF analysis (for parallelism)  
   function cusolverRfAnalyze (handle : cusolverRfHandle_t) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverRf.h:202
   pragma Import (C, cusolverRfAnalyze, "cusolverRfAnalyze");

  -- CUSOLVERRF re-factorization (for parallelism)  
   function cusolverRfRefactor (handle : cusolverRfHandle_t) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverRf.h:205
   pragma Import (C, cusolverRfRefactor, "cusolverRfRefactor");

  -- CUSOLVERRF extraction: Get L & U packed into a single matrix M  
  -- Input  
   function cusolverRfAccessBundledFactorsDevice
     (handle : cusolverRfHandle_t;
      nnzM : access int;
      Mp : System.Address;
      Mi : System.Address;
      Mx : System.Address) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverRf.h:208
   pragma Import (C, cusolverRfAccessBundledFactorsDevice, "cusolverRfAccessBundledFactorsDevice");

  -- Output (in the host memory)  
  -- Output (in the device memory)  
  -- Input  
   function cusolverRfExtractBundledFactorsHost
     (handle : cusolverRfHandle_t;
      h_nnzM : access int;
      h_Mp : System.Address;
      h_Mi : System.Address;
      h_Mx : System.Address) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverRf.h:217
   pragma Import (C, cusolverRfExtractBundledFactorsHost, "cusolverRfExtractBundledFactorsHost");

  -- Output (in the host memory)  
  -- CUSOLVERRF extraction: Get L & U individually  
  -- Input  
   function cusolverRfExtractSplitFactorsHost
     (handle : cusolverRfHandle_t;
      h_nnzL : access int;
      h_csrRowPtrL : System.Address;
      h_csrColIndL : System.Address;
      h_csrValL : System.Address;
      h_nnzU : access int;
      h_csrRowPtrU : System.Address;
      h_csrColIndU : System.Address;
      h_csrValU : System.Address) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverRf.h:226
   pragma Import (C, cusolverRfExtractSplitFactorsHost, "cusolverRfExtractSplitFactorsHost");

  -- Output (in the host memory)  
  -- CUSOLVERRF (forward and backward triangular) solves  
  -- Input (in the device memory)  
   function cusolverRfSolve
     (handle : cusolverRfHandle_t;
      P : access int;
      Q : access int;
      nrhs : int;
      Temp : access double;
      ldt : int;
      XF : access double;
      ldxf : int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverRf.h:239
   pragma Import (C, cusolverRfSolve, "cusolverRfSolve");

  --only nrhs=1 is supported
  --of size ldt*nrhs (ldt>=n)
  -- Input/Output (in the device memory)  
  -- Input  
  --** Batched Routines ** 
  -- CUSOLVERRF-batch setup of internal structures from host  
  -- Input (in the host memory) 
   function cusolverRfBatchSetupHost
     (batchSize : int;
      n : int;
      nnzA : int;
      h_csrRowPtrA : access int;
      h_csrColIndA : access int;
      h_csrValA_array : System.Address;
      nnzL : int;
      h_csrRowPtrL : access int;
      h_csrColIndL : access int;
      h_csrValL : access double;
      nnzU : int;
      h_csrRowPtrU : access int;
      h_csrColIndU : access int;
      h_csrValU : access double;
      h_P : access int;
      h_Q : access int;
      handle : cusolverRfHandle_t) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverRf.h:253
   pragma Import (C, cusolverRfBatchSetupHost, "cusolverRfBatchSetupHost");

  -- Output (in the device memory)  
  -- CUSOLVERRF-batch update the matrix values (assuming the reordering, pivoting 
  --   and consequently the sparsity pattern of L and U did not change),
  --   and zero out the remaining values.  

  -- Input (in the device memory)  
   function cusolverRfBatchResetValues
     (batchSize : int;
      n : int;
      nnzA : int;
      csrRowPtrA : access int;
      csrColIndA : access int;
      csrValA_array : System.Address;
      P : access int;
      Q : access int;
      handle : cusolverRfHandle_t) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverRf.h:276
   pragma Import (C, cusolverRfBatchResetValues, "cusolverRfBatchResetValues");

  -- Output  
  -- CUSOLVERRF-batch analysis (for parallelism)  
   function cusolverRfBatchAnalyze (handle : cusolverRfHandle_t) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverRf.h:289
   pragma Import (C, cusolverRfBatchAnalyze, "cusolverRfBatchAnalyze");

  -- CUSOLVERRF-batch re-factorization (for parallelism)  
   function cusolverRfBatchRefactor (handle : cusolverRfHandle_t) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverRf.h:292
   pragma Import (C, cusolverRfBatchRefactor, "cusolverRfBatchRefactor");

  -- CUSOLVERRF-batch (forward and backward triangular) solves  
  -- Input (in the device memory)  
   function cusolverRfBatchSolve
     (handle : cusolverRfHandle_t;
      P : access int;
      Q : access int;
      nrhs : int;
      Temp : access double;
      ldt : int;
      XF_array : System.Address;
      ldxf : int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverRf.h:295
   pragma Import (C, cusolverRfBatchSolve, "cusolverRfBatchSolve");

  --only nrhs=1 is supported
  --of size 2*batchSize*(n*nrhs)
  --only ldt=n is supported
  -- Input/Output (in the device memory)  
  -- Input  
  -- CUSOLVERRF-batch obtain the position of zero pivot  
  -- Input  
   function cusolverRfBatchZeroPivot (handle : cusolverRfHandle_t; position : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverRf.h:308
   pragma Import (C, cusolverRfBatchZeroPivot, "cusolverRfBatchZeroPivot");

  -- Output (in the host memory)  
end cusolverRf_h;

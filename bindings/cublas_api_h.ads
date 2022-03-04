pragma Ada_2005;
pragma Style_Checks (Off);

with Interfaces.C; use Interfaces.C;
with library_types_h;
with System;
with driver_types_h;
with Interfaces.C.Strings;
limited with cuComplex_h;
limited with cuda_fp16_h;

package cublas_api_h is

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

  -- import complex data type  
  -- CUBLAS status type returns  
   subtype cublasStatus_t is unsigned;
   CUBLAS_STATUS_SUCCESS : constant cublasStatus_t := 0;
   CUBLAS_STATUS_NOT_INITIALIZED : constant cublasStatus_t := 1;
   CUBLAS_STATUS_ALLOC_FAILED : constant cublasStatus_t := 3;
   CUBLAS_STATUS_INVALID_VALUE : constant cublasStatus_t := 7;
   CUBLAS_STATUS_ARCH_MISMATCH : constant cublasStatus_t := 8;
   CUBLAS_STATUS_MAPPING_ERROR : constant cublasStatus_t := 11;
   CUBLAS_STATUS_EXECUTION_FAILED : constant cublasStatus_t := 13;
   CUBLAS_STATUS_INTERNAL_ERROR : constant cublasStatus_t := 14;
   CUBLAS_STATUS_NOT_SUPPORTED : constant cublasStatus_t := 15;
   CUBLAS_STATUS_LICENSE_ERROR : constant cublasStatus_t := 16;  -- /usr/local/cuda-8.0/include/cublas_api.h:93

   type cublasFillMode_t is 
     (CUBLAS_FILL_MODE_LOWER,
      CUBLAS_FILL_MODE_UPPER);
   pragma Convention (C, cublasFillMode_t);  -- /usr/local/cuda-8.0/include/cublas_api.h:99

   type cublasDiagType_t is 
     (CUBLAS_DIAG_NON_UNIT,
      CUBLAS_DIAG_UNIT);
   pragma Convention (C, cublasDiagType_t);  -- /usr/local/cuda-8.0/include/cublas_api.h:104

   type cublasSideMode_t is 
     (CUBLAS_SIDE_LEFT,
      CUBLAS_SIDE_RIGHT);
   pragma Convention (C, cublasSideMode_t);  -- /usr/local/cuda-8.0/include/cublas_api.h:109

   type cublasOperation_t is 
     (CUBLAS_OP_N,
      CUBLAS_OP_T,
      CUBLAS_OP_C);
   pragma Convention (C, cublasOperation_t);  -- /usr/local/cuda-8.0/include/cublas_api.h:116

   type cublasPointerMode_t is 
     (CUBLAS_POINTER_MODE_HOST,
      CUBLAS_POINTER_MODE_DEVICE);
   pragma Convention (C, cublasPointerMode_t);  -- /usr/local/cuda-8.0/include/cublas_api.h:122

   type cublasAtomicsMode_t is 
     (CUBLAS_ATOMICS_NOT_ALLOWED,
      CUBLAS_ATOMICS_ALLOWED);
   pragma Convention (C, cublasAtomicsMode_t);  -- /usr/local/cuda-8.0/include/cublas_api.h:127

  --For different GEMM algorithm  
   subtype cublasGemmAlgo_t is unsigned;
   CUBLAS_GEMM_DFALT : constant cublasGemmAlgo_t := -1;
   CUBLAS_GEMM_ALGO0 : constant cublasGemmAlgo_t := 0;
   CUBLAS_GEMM_ALGO1 : constant cublasGemmAlgo_t := 1;
   CUBLAS_GEMM_ALGO2 : constant cublasGemmAlgo_t := 2;
   CUBLAS_GEMM_ALGO3 : constant cublasGemmAlgo_t := 3;
   CUBLAS_GEMM_ALGO4 : constant cublasGemmAlgo_t := 4;
   CUBLAS_GEMM_ALGO5 : constant cublasGemmAlgo_t := 5;
   CUBLAS_GEMM_ALGO6 : constant cublasGemmAlgo_t := 6;
   CUBLAS_GEMM_ALGO7 : constant cublasGemmAlgo_t := 7;  -- /usr/local/cuda-8.0/include/cublas_api.h:140

  -- For backward compatibility purposes  
   subtype cublasDataType_t is library_types_h.cudaDataType_t;

  -- Opaque structure holding CUBLAS library context  
   --  skipped empty struct cublasContext

   type cublasHandle_t is new System.Address;  -- /usr/local/cuda-8.0/include/cublas_api.h:147

   function cublasCreate_v2 (handle : System.Address) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:149
   pragma Import (C, cublasCreate_v2, "cublasCreate_v2");

   function cublasDestroy_v2 (handle : cublasHandle_t) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:150
   pragma Import (C, cublasDestroy_v2, "cublasDestroy_v2");

   function cublasGetVersion_v2 (handle : cublasHandle_t; version : access int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:152
   pragma Import (C, cublasGetVersion_v2, "cublasGetVersion_v2");

   function cublasGetProperty (c_type : library_types_h.libraryPropertyType; value : access int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:153
   pragma Import (C, cublasGetProperty, "cublasGetProperty");

   function cublasSetStream_v2 (handle : cublasHandle_t; streamId : driver_types_h.cudaStream_t) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:155
   pragma Import (C, cublasSetStream_v2, "cublasSetStream_v2");

   function cublasGetStream_v2 (handle : cublasHandle_t; streamId : System.Address) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:156
   pragma Import (C, cublasGetStream_v2, "cublasGetStream_v2");

   function cublasGetPointerMode_v2 (handle : cublasHandle_t; mode : access cublasPointerMode_t) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:158
   pragma Import (C, cublasGetPointerMode_v2, "cublasGetPointerMode_v2");

   function cublasSetPointerMode_v2 (handle : cublasHandle_t; mode : cublasPointerMode_t) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:159
   pragma Import (C, cublasSetPointerMode_v2, "cublasSetPointerMode_v2");

   function cublasGetAtomicsMode (handle : cublasHandle_t; mode : access cublasAtomicsMode_t) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:161
   pragma Import (C, cublasGetAtomicsMode, "cublasGetAtomicsMode");

   function cublasSetAtomicsMode (handle : cublasHandle_t; mode : cublasAtomicsMode_t) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:162
   pragma Import (C, cublasSetAtomicsMode, "cublasSetAtomicsMode");

  -- 
  -- * cublasStatus_t 
  -- * cublasSetVector (int n, int elemSize, const void *x, int incx, 
  -- *                  void *y, int incy) 
  -- *
  -- * copies n elements from a vector x in CPU memory space to a vector y 
  -- * in GPU memory space. Elements in both vectors are assumed to have a 
  -- * size of elemSize bytes. Storage spacing between consecutive elements
  -- * is incx for the source vector x and incy for the destination vector
  -- * y. In general, y points to an object, or part of an object, allocated
  -- * via cublasAlloc(). Column major format for two-dimensional matrices
  -- * is assumed throughout CUBLAS. Therefore, if the increment for a vector 
  -- * is equal to 1, this access a column vector while using an increment 
  -- * equal to the leading dimension of the respective matrix accesses a 
  -- * row vector.
  -- *
  -- * Return Values
  -- * -------------
  -- * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library not been initialized
  -- * CUBLAS_STATUS_INVALID_VALUE    if incx, incy, or elemSize <= 0
  -- * CUBLAS_STATUS_MAPPING_ERROR    if an error occurred accessing GPU memory   
  -- * CUBLAS_STATUS_SUCCESS          if the operation completed successfully
  --  

   function cublasSetVector
     (n : int;
      elemSize : int;
      x : System.Address;
      incx : int;
      devicePtr : System.Address;
      incy : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:187
   pragma Import (C, cublasSetVector, "cublasSetVector");

  -- 
  -- * cublasStatus_t 
  -- * cublasGetVector (int n, int elemSize, const void *x, int incx, 
  -- *                  void *y, int incy)
  -- * 
  -- * copies n elements from a vector x in GPU memory space to a vector y 
  -- * in CPU memory space. Elements in both vectors are assumed to have a 
  -- * size of elemSize bytes. Storage spacing between consecutive elements
  -- * is incx for the source vector x and incy for the destination vector
  -- * y. In general, x points to an object, or part of an object, allocated
  -- * via cublasAlloc(). Column major format for two-dimensional matrices
  -- * is assumed throughout CUBLAS. Therefore, if the increment for a vector 
  -- * is equal to 1, this access a column vector while using an increment 
  -- * equal to the leading dimension of the respective matrix accesses a 
  -- * row vector.
  -- *
  -- * Return Values
  -- * -------------
  -- * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library not been initialized
  -- * CUBLAS_STATUS_INVALID_VALUE    if incx, incy, or elemSize <= 0
  -- * CUBLAS_STATUS_MAPPING_ERROR    if an error occurred accessing GPU memory   
  -- * CUBLAS_STATUS_SUCCESS          if the operation completed successfully
  --  

   function cublasGetVector
     (n : int;
      elemSize : int;
      x : System.Address;
      incx : int;
      y : System.Address;
      incy : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:213
   pragma Import (C, cublasGetVector, "cublasGetVector");

  -- * cublasStatus_t 
  -- * cublasSetMatrix (int rows, int cols, int elemSize, const void *A, 
  -- *                  int lda, void *B, int ldb)
  -- *
  -- * copies a tile of rows x cols elements from a matrix A in CPU memory
  -- * space to a matrix B in GPU memory space. Each element requires storage
  -- * of elemSize bytes. Both matrices are assumed to be stored in column 
  -- * major format, with the leading dimension (i.e. number of rows) of 
  -- * source matrix A provided in lda, and the leading dimension of matrix B
  -- * provided in ldb. In general, B points to an object, or part of an 
  -- * object, that was allocated via cublasAlloc().
  -- *
  -- * Return Values 
  -- * -------------
  -- * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
  -- * CUBLAS_STATUS_INVALID_VALUE    if rows or cols < 0, or elemSize, lda, or 
  -- *                                ldb <= 0
  -- * CUBLAS_STATUS_MAPPING_ERROR    if error occurred accessing GPU memory
  -- * CUBLAS_STATUS_SUCCESS          if the operation completed successfully
  --  

   function cublasSetMatrix
     (rows : int;
      cols : int;
      elemSize : int;
      A : System.Address;
      lda : int;
      B : System.Address;
      ldb : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:237
   pragma Import (C, cublasSetMatrix, "cublasSetMatrix");

  -- * cublasStatus_t 
  -- * cublasGetMatrix (int rows, int cols, int elemSize, const void *A, 
  -- *                  int lda, void *B, int ldb)
  -- *
  -- * copies a tile of rows x cols elements from a matrix A in GPU memory
  -- * space to a matrix B in CPU memory space. Each element requires storage
  -- * of elemSize bytes. Both matrices are assumed to be stored in column 
  -- * major format, with the leading dimension (i.e. number of rows) of 
  -- * source matrix A provided in lda, and the leading dimension of matrix B
  -- * provided in ldb. In general, A points to an object, or part of an 
  -- * object, that was allocated via cublasAlloc().
  -- *
  -- * Return Values 
  -- * -------------
  -- * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
  -- * CUBLAS_STATUS_INVALID_VALUE    if rows, cols, eleSize, lda, or ldb <= 0
  -- * CUBLAS_STATUS_MAPPING_ERROR    if error occurred accessing GPU memory
  -- * CUBLAS_STATUS_SUCCESS          if the operation completed successfully
  --  

   function cublasGetMatrix
     (rows : int;
      cols : int;
      elemSize : int;
      A : System.Address;
      lda : int;
      B : System.Address;
      ldb : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:261
   pragma Import (C, cublasGetMatrix, "cublasGetMatrix");

  -- 
  -- * cublasStatus 
  -- * cublasSetVectorAsync ( int n, int elemSize, const void *x, int incx, 
  -- *                       void *y, int incy, cudaStream_t stream );
  -- *
  -- * cublasSetVectorAsync has the same functionnality as cublasSetVector
  -- * but the transfer is done asynchronously within the CUDA stream passed
  -- * in parameter.
  -- *
  -- * Return Values
  -- * -------------
  -- * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library not been initialized
  -- * CUBLAS_STATUS_INVALID_VALUE    if incx, incy, or elemSize <= 0
  -- * CUBLAS_STATUS_MAPPING_ERROR    if an error occurred accessing GPU memory   
  -- * CUBLAS_STATUS_SUCCESS          if the operation completed successfully
  --  

   function cublasSetVectorAsync
     (n : int;
      elemSize : int;
      hostPtr : System.Address;
      incx : int;
      devicePtr : System.Address;
      incy : int;
      stream : driver_types_h.cudaStream_t) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:281
   pragma Import (C, cublasSetVectorAsync, "cublasSetVectorAsync");

  -- 
  -- * cublasStatus 
  -- * cublasGetVectorAsync( int n, int elemSize, const void *x, int incx, 
  -- *                       void *y, int incy, cudaStream_t stream)
  -- * 
  -- * cublasGetVectorAsync has the same functionnality as cublasGetVector
  -- * but the transfer is done asynchronously within the CUDA stream passed
  -- * in parameter.
  -- *
  -- * Return Values
  -- * -------------
  -- * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library not been initialized
  -- * CUBLAS_STATUS_INVALID_VALUE    if incx, incy, or elemSize <= 0
  -- * CUBLAS_STATUS_MAPPING_ERROR    if an error occurred accessing GPU memory   
  -- * CUBLAS_STATUS_SUCCESS          if the operation completed successfully
  --  

   function cublasGetVectorAsync
     (n : int;
      elemSize : int;
      devicePtr : System.Address;
      incx : int;
      hostPtr : System.Address;
      incy : int;
      stream : driver_types_h.cudaStream_t) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:301
   pragma Import (C, cublasGetVectorAsync, "cublasGetVectorAsync");

  -- * cublasStatus_t 
  -- * cublasSetMatrixAsync (int rows, int cols, int elemSize, const void *A, 
  -- *                       int lda, void *B, int ldb, cudaStream_t stream)
  -- *
  -- * cublasSetMatrixAsync has the same functionnality as cublasSetMatrix
  -- * but the transfer is done asynchronously within the CUDA stream passed
  -- * in parameter.
  -- *
  -- * Return Values 
  -- * -------------
  -- * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
  -- * CUBLAS_STATUS_INVALID_VALUE    if rows or cols < 0, or elemSize, lda, or 
  -- *                                ldb <= 0
  -- * CUBLAS_STATUS_MAPPING_ERROR    if error occurred accessing GPU memory
  -- * CUBLAS_STATUS_SUCCESS          if the operation completed successfully
  --  

   function cublasSetMatrixAsync
     (rows : int;
      cols : int;
      elemSize : int;
      A : System.Address;
      lda : int;
      B : System.Address;
      ldb : int;
      stream : driver_types_h.cudaStream_t) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:323
   pragma Import (C, cublasSetMatrixAsync, "cublasSetMatrixAsync");

  -- * cublasStatus_t 
  -- * cublasGetMatrixAsync (int rows, int cols, int elemSize, const void *A, 
  -- *                       int lda, void *B, int ldb, cudaStream_t stream)
  -- *
  -- * cublasGetMatrixAsync has the same functionnality as cublasGetMatrix
  -- * but the transfer is done asynchronously within the CUDA stream passed
  -- * in parameter.
  -- *
  -- * Return Values 
  -- * -------------
  -- * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
  -- * CUBLAS_STATUS_INVALID_VALUE    if rows, cols, eleSize, lda, or ldb <= 0
  -- * CUBLAS_STATUS_MAPPING_ERROR    if error occurred accessing GPU memory
  -- * CUBLAS_STATUS_SUCCESS          if the operation completed successfully
  --  

   function cublasGetMatrixAsync
     (rows : int;
      cols : int;
      elemSize : int;
      A : System.Address;
      lda : int;
      B : System.Address;
      ldb : int;
      stream : driver_types_h.cudaStream_t) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:343
   pragma Import (C, cublasGetMatrixAsync, "cublasGetMatrixAsync");

   procedure cublasXerbla (srName : Interfaces.C.Strings.chars_ptr; info : int);  -- /usr/local/cuda-8.0/include/cublas_api.h:348
   pragma Import (C, cublasXerbla, "cublasXerbla");

  -- ---------------- CUBLAS BLAS1 functions ----------------  
   function cublasNrm2Ex
     (handle : cublasHandle_t;
      n : int;
      x : System.Address;
      xType : library_types_h.cudaDataType;
      incx : int;
      result : System.Address;
      resultType : library_types_h.cudaDataType;
      executionType : library_types_h.cudaDataType) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:350
   pragma Import (C, cublasNrm2Ex, "cublasNrm2Ex");

  -- host or device pointer  
   function cublasSnrm2_v2
     (handle : cublasHandle_t;
      n : int;
      x : access float;
      incx : int;
      result : access float) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:358
   pragma Import (C, cublasSnrm2_v2, "cublasSnrm2_v2");

  -- host or device pointer  
   function cublasDnrm2_v2
     (handle : cublasHandle_t;
      n : int;
      x : access double;
      incx : int;
      result : access double) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:364
   pragma Import (C, cublasDnrm2_v2, "cublasDnrm2_v2");

  -- host or device pointer  
   function cublasScnrm2_v2
     (handle : cublasHandle_t;
      n : int;
      x : access constant cuComplex_h.cuComplex;
      incx : int;
      result : access float) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:370
   pragma Import (C, cublasScnrm2_v2, "cublasScnrm2_v2");

  -- host or device pointer  
   function cublasDznrm2_v2
     (handle : cublasHandle_t;
      n : int;
      x : access constant cuComplex_h.cuDoubleComplex;
      incx : int;
      result : access double) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:376
   pragma Import (C, cublasDznrm2_v2, "cublasDznrm2_v2");

  -- host or device pointer  
   function cublasDotEx
     (handle : cublasHandle_t;
      n : int;
      x : System.Address;
      xType : library_types_h.cudaDataType;
      incx : int;
      y : System.Address;
      yType : library_types_h.cudaDataType;
      incy : int;
      result : System.Address;
      resultType : library_types_h.cudaDataType;
      executionType : library_types_h.cudaDataType) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:382
   pragma Import (C, cublasDotEx, "cublasDotEx");

   function cublasDotcEx
     (handle : cublasHandle_t;
      n : int;
      x : System.Address;
      xType : library_types_h.cudaDataType;
      incx : int;
      y : System.Address;
      yType : library_types_h.cudaDataType;
      incy : int;
      result : System.Address;
      resultType : library_types_h.cudaDataType;
      executionType : library_types_h.cudaDataType) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:394
   pragma Import (C, cublasDotcEx, "cublasDotcEx");

   function cublasSdot_v2
     (handle : cublasHandle_t;
      n : int;
      x : access float;
      incx : int;
      y : access float;
      incy : int;
      result : access float) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:406
   pragma Import (C, cublasSdot_v2, "cublasSdot_v2");

  -- host or device pointer  
   function cublasDdot_v2
     (handle : cublasHandle_t;
      n : int;
      x : access double;
      incx : int;
      y : access double;
      incy : int;
      result : access double) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:414
   pragma Import (C, cublasDdot_v2, "cublasDdot_v2");

  -- host or device pointer  
   function cublasCdotu_v2
     (handle : cublasHandle_t;
      n : int;
      x : access constant cuComplex_h.cuComplex;
      incx : int;
      y : access constant cuComplex_h.cuComplex;
      incy : int;
      result : access cuComplex_h.cuComplex) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:422
   pragma Import (C, cublasCdotu_v2, "cublasCdotu_v2");

  -- host or device pointer  
   function cublasCdotc_v2
     (handle : cublasHandle_t;
      n : int;
      x : access constant cuComplex_h.cuComplex;
      incx : int;
      y : access constant cuComplex_h.cuComplex;
      incy : int;
      result : access cuComplex_h.cuComplex) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:430
   pragma Import (C, cublasCdotc_v2, "cublasCdotc_v2");

  -- host or device pointer  
   function cublasZdotu_v2
     (handle : cublasHandle_t;
      n : int;
      x : access constant cuComplex_h.cuDoubleComplex;
      incx : int;
      y : access constant cuComplex_h.cuDoubleComplex;
      incy : int;
      result : access cuComplex_h.cuDoubleComplex) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:438
   pragma Import (C, cublasZdotu_v2, "cublasZdotu_v2");

  -- host or device pointer  
   function cublasZdotc_v2
     (handle : cublasHandle_t;
      n : int;
      x : access constant cuComplex_h.cuDoubleComplex;
      incx : int;
      y : access constant cuComplex_h.cuDoubleComplex;
      incy : int;
      result : access cuComplex_h.cuDoubleComplex) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:446
   pragma Import (C, cublasZdotc_v2, "cublasZdotc_v2");

  -- host or device pointer  
   function cublasScalEx
     (handle : cublasHandle_t;
      n : int;
      alpha : System.Address;
      alphaType : library_types_h.cudaDataType;
      x : System.Address;
      xType : library_types_h.cudaDataType;
      incx : int;
      executionType : library_types_h.cudaDataType) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:454
   pragma Import (C, cublasScalEx, "cublasScalEx");

  -- host or device pointer  
   function cublasSscal_v2
     (handle : cublasHandle_t;
      n : int;
      alpha : access float;
      x : access float;
      incx : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:462
   pragma Import (C, cublasSscal_v2, "cublasSscal_v2");

  -- host or device pointer  
   function cublasDscal_v2
     (handle : cublasHandle_t;
      n : int;
      alpha : access double;
      x : access double;
      incx : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:468
   pragma Import (C, cublasDscal_v2, "cublasDscal_v2");

  -- host or device pointer  
   function cublasCscal_v2
     (handle : cublasHandle_t;
      n : int;
      alpha : access constant cuComplex_h.cuComplex;
      x : access cuComplex_h.cuComplex;
      incx : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:474
   pragma Import (C, cublasCscal_v2, "cublasCscal_v2");

  -- host or device pointer  
   function cublasCsscal_v2
     (handle : cublasHandle_t;
      n : int;
      alpha : access float;
      x : access cuComplex_h.cuComplex;
      incx : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:480
   pragma Import (C, cublasCsscal_v2, "cublasCsscal_v2");

  -- host or device pointer  
   function cublasZscal_v2
     (handle : cublasHandle_t;
      n : int;
      alpha : access constant cuComplex_h.cuDoubleComplex;
      x : access cuComplex_h.cuDoubleComplex;
      incx : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:486
   pragma Import (C, cublasZscal_v2, "cublasZscal_v2");

  -- host or device pointer  
   function cublasZdscal_v2
     (handle : cublasHandle_t;
      n : int;
      alpha : access double;
      x : access cuComplex_h.cuDoubleComplex;
      incx : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:492
   pragma Import (C, cublasZdscal_v2, "cublasZdscal_v2");

  -- host or device pointer  
   function cublasAxpyEx
     (handle : cublasHandle_t;
      n : int;
      alpha : System.Address;
      alphaType : library_types_h.cudaDataType;
      x : System.Address;
      xType : library_types_h.cudaDataType;
      incx : int;
      y : System.Address;
      yType : library_types_h.cudaDataType;
      incy : int;
      executiontype : library_types_h.cudaDataType) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:498
   pragma Import (C, cublasAxpyEx, "cublasAxpyEx");

  -- host or device pointer  
   function cublasSaxpy_v2
     (handle : cublasHandle_t;
      n : int;
      alpha : access float;
      x : access float;
      incx : int;
      y : access float;
      incy : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:510
   pragma Import (C, cublasSaxpy_v2, "cublasSaxpy_v2");

  -- host or device pointer  
   function cublasDaxpy_v2
     (handle : cublasHandle_t;
      n : int;
      alpha : access double;
      x : access double;
      incx : int;
      y : access double;
      incy : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:518
   pragma Import (C, cublasDaxpy_v2, "cublasDaxpy_v2");

  -- host or device pointer  
   function cublasCaxpy_v2
     (handle : cublasHandle_t;
      n : int;
      alpha : access constant cuComplex_h.cuComplex;
      x : access constant cuComplex_h.cuComplex;
      incx : int;
      y : access cuComplex_h.cuComplex;
      incy : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:526
   pragma Import (C, cublasCaxpy_v2, "cublasCaxpy_v2");

  -- host or device pointer  
   function cublasZaxpy_v2
     (handle : cublasHandle_t;
      n : int;
      alpha : access constant cuComplex_h.cuDoubleComplex;
      x : access constant cuComplex_h.cuDoubleComplex;
      incx : int;
      y : access cuComplex_h.cuDoubleComplex;
      incy : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:534
   pragma Import (C, cublasZaxpy_v2, "cublasZaxpy_v2");

  -- host or device pointer  
   function cublasScopy_v2
     (handle : cublasHandle_t;
      n : int;
      x : access float;
      incx : int;
      y : access float;
      incy : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:542
   pragma Import (C, cublasScopy_v2, "cublasScopy_v2");

   function cublasDcopy_v2
     (handle : cublasHandle_t;
      n : int;
      x : access double;
      incx : int;
      y : access double;
      incy : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:549
   pragma Import (C, cublasDcopy_v2, "cublasDcopy_v2");

   function cublasCcopy_v2
     (handle : cublasHandle_t;
      n : int;
      x : access constant cuComplex_h.cuComplex;
      incx : int;
      y : access cuComplex_h.cuComplex;
      incy : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:556
   pragma Import (C, cublasCcopy_v2, "cublasCcopy_v2");

   function cublasZcopy_v2
     (handle : cublasHandle_t;
      n : int;
      x : access constant cuComplex_h.cuDoubleComplex;
      incx : int;
      y : access cuComplex_h.cuDoubleComplex;
      incy : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:563
   pragma Import (C, cublasZcopy_v2, "cublasZcopy_v2");

   function cublasSswap_v2
     (handle : cublasHandle_t;
      n : int;
      x : access float;
      incx : int;
      y : access float;
      incy : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:570
   pragma Import (C, cublasSswap_v2, "cublasSswap_v2");

   function cublasDswap_v2
     (handle : cublasHandle_t;
      n : int;
      x : access double;
      incx : int;
      y : access double;
      incy : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:577
   pragma Import (C, cublasDswap_v2, "cublasDswap_v2");

   function cublasCswap_v2
     (handle : cublasHandle_t;
      n : int;
      x : access cuComplex_h.cuComplex;
      incx : int;
      y : access cuComplex_h.cuComplex;
      incy : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:584
   pragma Import (C, cublasCswap_v2, "cublasCswap_v2");

   function cublasZswap_v2
     (handle : cublasHandle_t;
      n : int;
      x : access cuComplex_h.cuDoubleComplex;
      incx : int;
      y : access cuComplex_h.cuDoubleComplex;
      incy : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:591
   pragma Import (C, cublasZswap_v2, "cublasZswap_v2");

   function cublasIsamax_v2
     (handle : cublasHandle_t;
      n : int;
      x : access float;
      incx : int;
      result : access int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:598
   pragma Import (C, cublasIsamax_v2, "cublasIsamax_v2");

  -- host or device pointer  
   function cublasIdamax_v2
     (handle : cublasHandle_t;
      n : int;
      x : access double;
      incx : int;
      result : access int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:604
   pragma Import (C, cublasIdamax_v2, "cublasIdamax_v2");

  -- host or device pointer  
   function cublasIcamax_v2
     (handle : cublasHandle_t;
      n : int;
      x : access constant cuComplex_h.cuComplex;
      incx : int;
      result : access int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:610
   pragma Import (C, cublasIcamax_v2, "cublasIcamax_v2");

  -- host or device pointer  
   function cublasIzamax_v2
     (handle : cublasHandle_t;
      n : int;
      x : access constant cuComplex_h.cuDoubleComplex;
      incx : int;
      result : access int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:616
   pragma Import (C, cublasIzamax_v2, "cublasIzamax_v2");

  -- host or device pointer  
   function cublasIsamin_v2
     (handle : cublasHandle_t;
      n : int;
      x : access float;
      incx : int;
      result : access int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:622
   pragma Import (C, cublasIsamin_v2, "cublasIsamin_v2");

  -- host or device pointer  
   function cublasIdamin_v2
     (handle : cublasHandle_t;
      n : int;
      x : access double;
      incx : int;
      result : access int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:628
   pragma Import (C, cublasIdamin_v2, "cublasIdamin_v2");

  -- host or device pointer  
   function cublasIcamin_v2
     (handle : cublasHandle_t;
      n : int;
      x : access constant cuComplex_h.cuComplex;
      incx : int;
      result : access int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:634
   pragma Import (C, cublasIcamin_v2, "cublasIcamin_v2");

  -- host or device pointer  
   function cublasIzamin_v2
     (handle : cublasHandle_t;
      n : int;
      x : access constant cuComplex_h.cuDoubleComplex;
      incx : int;
      result : access int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:640
   pragma Import (C, cublasIzamin_v2, "cublasIzamin_v2");

  -- host or device pointer  
   function cublasSasum_v2
     (handle : cublasHandle_t;
      n : int;
      x : access float;
      incx : int;
      result : access float) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:646
   pragma Import (C, cublasSasum_v2, "cublasSasum_v2");

  -- host or device pointer  
   function cublasDasum_v2
     (handle : cublasHandle_t;
      n : int;
      x : access double;
      incx : int;
      result : access double) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:652
   pragma Import (C, cublasDasum_v2, "cublasDasum_v2");

  -- host or device pointer  
   function cublasScasum_v2
     (handle : cublasHandle_t;
      n : int;
      x : access constant cuComplex_h.cuComplex;
      incx : int;
      result : access float) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:658
   pragma Import (C, cublasScasum_v2, "cublasScasum_v2");

  -- host or device pointer  
   function cublasDzasum_v2
     (handle : cublasHandle_t;
      n : int;
      x : access constant cuComplex_h.cuDoubleComplex;
      incx : int;
      result : access double) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:664
   pragma Import (C, cublasDzasum_v2, "cublasDzasum_v2");

  -- host or device pointer  
   function cublasSrot_v2
     (handle : cublasHandle_t;
      n : int;
      x : access float;
      incx : int;
      y : access float;
      incy : int;
      c : access float;
      s : access float) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:670
   pragma Import (C, cublasSrot_v2, "cublasSrot_v2");

  -- host or device pointer  
  -- host or device pointer  
   function cublasDrot_v2
     (handle : cublasHandle_t;
      n : int;
      x : access double;
      incx : int;
      y : access double;
      incy : int;
      c : access double;
      s : access double) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:679
   pragma Import (C, cublasDrot_v2, "cublasDrot_v2");

  -- host or device pointer  
  -- host or device pointer  
   function cublasCrot_v2
     (handle : cublasHandle_t;
      n : int;
      x : access cuComplex_h.cuComplex;
      incx : int;
      y : access cuComplex_h.cuComplex;
      incy : int;
      c : access float;
      s : access constant cuComplex_h.cuComplex) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:688
   pragma Import (C, cublasCrot_v2, "cublasCrot_v2");

  -- host or device pointer  
  -- host or device pointer  
   function cublasCsrot_v2
     (handle : cublasHandle_t;
      n : int;
      x : access cuComplex_h.cuComplex;
      incx : int;
      y : access cuComplex_h.cuComplex;
      incy : int;
      c : access float;
      s : access float) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:697
   pragma Import (C, cublasCsrot_v2, "cublasCsrot_v2");

  -- host or device pointer  
  -- host or device pointer  
   function cublasZrot_v2
     (handle : cublasHandle_t;
      n : int;
      x : access cuComplex_h.cuDoubleComplex;
      incx : int;
      y : access cuComplex_h.cuDoubleComplex;
      incy : int;
      c : access double;
      s : access constant cuComplex_h.cuDoubleComplex) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:706
   pragma Import (C, cublasZrot_v2, "cublasZrot_v2");

  -- host or device pointer  
  -- host or device pointer  
   function cublasZdrot_v2
     (handle : cublasHandle_t;
      n : int;
      x : access cuComplex_h.cuDoubleComplex;
      incx : int;
      y : access cuComplex_h.cuDoubleComplex;
      incy : int;
      c : access double;
      s : access double) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:715
   pragma Import (C, cublasZdrot_v2, "cublasZdrot_v2");

  -- host or device pointer  
  -- host or device pointer  
   function cublasSrotg_v2
     (handle : cublasHandle_t;
      a : access float;
      b : access float;
      c : access float;
      s : access float) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:724
   pragma Import (C, cublasSrotg_v2, "cublasSrotg_v2");

  -- host or device pointer  
  -- host or device pointer  
  -- host or device pointer  
  -- host or device pointer  
   function cublasDrotg_v2
     (handle : cublasHandle_t;
      a : access double;
      b : access double;
      c : access double;
      s : access double) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:730
   pragma Import (C, cublasDrotg_v2, "cublasDrotg_v2");

  -- host or device pointer  
  -- host or device pointer  
  -- host or device pointer  
  -- host or device pointer  
   function cublasCrotg_v2
     (handle : cublasHandle_t;
      a : access cuComplex_h.cuComplex;
      b : access cuComplex_h.cuComplex;
      c : access float;
      s : access cuComplex_h.cuComplex) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:736
   pragma Import (C, cublasCrotg_v2, "cublasCrotg_v2");

  -- host or device pointer  
  -- host or device pointer  
  -- host or device pointer  
  -- host or device pointer  
   function cublasZrotg_v2
     (handle : cublasHandle_t;
      a : access cuComplex_h.cuDoubleComplex;
      b : access cuComplex_h.cuDoubleComplex;
      c : access double;
      s : access cuComplex_h.cuDoubleComplex) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:742
   pragma Import (C, cublasZrotg_v2, "cublasZrotg_v2");

  -- host or device pointer  
  -- host or device pointer  
  -- host or device pointer  
  -- host or device pointer  
   function cublasSrotm_v2
     (handle : cublasHandle_t;
      n : int;
      x : access float;
      incx : int;
      y : access float;
      incy : int;
      param : access float) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:748
   pragma Import (C, cublasSrotm_v2, "cublasSrotm_v2");

  -- host or device pointer  
   function cublasDrotm_v2
     (handle : cublasHandle_t;
      n : int;
      x : access double;
      incx : int;
      y : access double;
      incy : int;
      param : access double) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:756
   pragma Import (C, cublasDrotm_v2, "cublasDrotm_v2");

  -- host or device pointer  
   function cublasSrotmg_v2
     (handle : cublasHandle_t;
      d1 : access float;
      d2 : access float;
      x1 : access float;
      y1 : access float;
      param : access float) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:764
   pragma Import (C, cublasSrotmg_v2, "cublasSrotmg_v2");

  -- host or device pointer  
  -- host or device pointer  
  -- host or device pointer  
  -- host or device pointer  
  -- host or device pointer  
   function cublasDrotmg_v2
     (handle : cublasHandle_t;
      d1 : access double;
      d2 : access double;
      x1 : access double;
      y1 : access double;
      param : access double) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:771
   pragma Import (C, cublasDrotmg_v2, "cublasDrotmg_v2");

  -- host or device pointer  
  -- host or device pointer  
  -- host or device pointer  
  -- host or device pointer  
  -- host or device pointer  
  -- --------------- CUBLAS BLAS2 functions  ----------------  
  -- GEMV  
   function cublasSgemv_v2
     (handle : cublasHandle_t;
      trans : cublasOperation_t;
      m : int;
      n : int;
      alpha : access float;
      A : access float;
      lda : int;
      x : access float;
      incx : int;
      beta : access float;
      y : access float;
      incy : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:781
   pragma Import (C, cublasSgemv_v2, "cublasSgemv_v2");

  -- host or device pointer  
  -- host or device pointer  
   function cublasDgemv_v2
     (handle : cublasHandle_t;
      trans : cublasOperation_t;
      m : int;
      n : int;
      alpha : access double;
      A : access double;
      lda : int;
      x : access double;
      incx : int;
      beta : access double;
      y : access double;
      incy : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:794
   pragma Import (C, cublasDgemv_v2, "cublasDgemv_v2");

  -- host or device pointer  
  -- host or device pointer  
   function cublasCgemv_v2
     (handle : cublasHandle_t;
      trans : cublasOperation_t;
      m : int;
      n : int;
      alpha : access constant cuComplex_h.cuComplex;
      A : access constant cuComplex_h.cuComplex;
      lda : int;
      x : access constant cuComplex_h.cuComplex;
      incx : int;
      beta : access constant cuComplex_h.cuComplex;
      y : access cuComplex_h.cuComplex;
      incy : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:807
   pragma Import (C, cublasCgemv_v2, "cublasCgemv_v2");

  -- host or device pointer  
  -- host or device pointer  
   function cublasZgemv_v2
     (handle : cublasHandle_t;
      trans : cublasOperation_t;
      m : int;
      n : int;
      alpha : access constant cuComplex_h.cuDoubleComplex;
      A : access constant cuComplex_h.cuDoubleComplex;
      lda : int;
      x : access constant cuComplex_h.cuDoubleComplex;
      incx : int;
      beta : access constant cuComplex_h.cuDoubleComplex;
      y : access cuComplex_h.cuDoubleComplex;
      incy : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:820
   pragma Import (C, cublasZgemv_v2, "cublasZgemv_v2");

  -- host or device pointer  
  -- host or device pointer  
  -- GBMV  
   function cublasSgbmv_v2
     (handle : cublasHandle_t;
      trans : cublasOperation_t;
      m : int;
      n : int;
      kl : int;
      ku : int;
      alpha : access float;
      A : access float;
      lda : int;
      x : access float;
      incx : int;
      beta : access float;
      y : access float;
      incy : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:833
   pragma Import (C, cublasSgbmv_v2, "cublasSgbmv_v2");

  -- host or device pointer  
  -- host or device pointer  
   function cublasDgbmv_v2
     (handle : cublasHandle_t;
      trans : cublasOperation_t;
      m : int;
      n : int;
      kl : int;
      ku : int;
      alpha : access double;
      A : access double;
      lda : int;
      x : access double;
      incx : int;
      beta : access double;
      y : access double;
      incy : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:848
   pragma Import (C, cublasDgbmv_v2, "cublasDgbmv_v2");

  -- host or device pointer  
  -- host or device pointer  
   function cublasCgbmv_v2
     (handle : cublasHandle_t;
      trans : cublasOperation_t;
      m : int;
      n : int;
      kl : int;
      ku : int;
      alpha : access constant cuComplex_h.cuComplex;
      A : access constant cuComplex_h.cuComplex;
      lda : int;
      x : access constant cuComplex_h.cuComplex;
      incx : int;
      beta : access constant cuComplex_h.cuComplex;
      y : access cuComplex_h.cuComplex;
      incy : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:863
   pragma Import (C, cublasCgbmv_v2, "cublasCgbmv_v2");

  -- host or device pointer  
  -- host or device pointer  
   function cublasZgbmv_v2
     (handle : cublasHandle_t;
      trans : cublasOperation_t;
      m : int;
      n : int;
      kl : int;
      ku : int;
      alpha : access constant cuComplex_h.cuDoubleComplex;
      A : access constant cuComplex_h.cuDoubleComplex;
      lda : int;
      x : access constant cuComplex_h.cuDoubleComplex;
      incx : int;
      beta : access constant cuComplex_h.cuDoubleComplex;
      y : access cuComplex_h.cuDoubleComplex;
      incy : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:878
   pragma Import (C, cublasZgbmv_v2, "cublasZgbmv_v2");

  -- host or device pointer  
  -- host or device pointer  
  -- TRMV  
   function cublasStrmv_v2
     (handle : cublasHandle_t;
      uplo : cublasFillMode_t;
      trans : cublasOperation_t;
      diag : cublasDiagType_t;
      n : int;
      A : access float;
      lda : int;
      x : access float;
      incx : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:894
   pragma Import (C, cublasStrmv_v2, "cublasStrmv_v2");

   function cublasDtrmv_v2
     (handle : cublasHandle_t;
      uplo : cublasFillMode_t;
      trans : cublasOperation_t;
      diag : cublasDiagType_t;
      n : int;
      A : access double;
      lda : int;
      x : access double;
      incx : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:904
   pragma Import (C, cublasDtrmv_v2, "cublasDtrmv_v2");

   function cublasCtrmv_v2
     (handle : cublasHandle_t;
      uplo : cublasFillMode_t;
      trans : cublasOperation_t;
      diag : cublasDiagType_t;
      n : int;
      A : access constant cuComplex_h.cuComplex;
      lda : int;
      x : access cuComplex_h.cuComplex;
      incx : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:914
   pragma Import (C, cublasCtrmv_v2, "cublasCtrmv_v2");

   function cublasZtrmv_v2
     (handle : cublasHandle_t;
      uplo : cublasFillMode_t;
      trans : cublasOperation_t;
      diag : cublasDiagType_t;
      n : int;
      A : access constant cuComplex_h.cuDoubleComplex;
      lda : int;
      x : access cuComplex_h.cuDoubleComplex;
      incx : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:924
   pragma Import (C, cublasZtrmv_v2, "cublasZtrmv_v2");

  -- TBMV  
   function cublasStbmv_v2
     (handle : cublasHandle_t;
      uplo : cublasFillMode_t;
      trans : cublasOperation_t;
      diag : cublasDiagType_t;
      n : int;
      k : int;
      A : access float;
      lda : int;
      x : access float;
      incx : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:935
   pragma Import (C, cublasStbmv_v2, "cublasStbmv_v2");

   function cublasDtbmv_v2
     (handle : cublasHandle_t;
      uplo : cublasFillMode_t;
      trans : cublasOperation_t;
      diag : cublasDiagType_t;
      n : int;
      k : int;
      A : access double;
      lda : int;
      x : access double;
      incx : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:946
   pragma Import (C, cublasDtbmv_v2, "cublasDtbmv_v2");

   function cublasCtbmv_v2
     (handle : cublasHandle_t;
      uplo : cublasFillMode_t;
      trans : cublasOperation_t;
      diag : cublasDiagType_t;
      n : int;
      k : int;
      A : access constant cuComplex_h.cuComplex;
      lda : int;
      x : access cuComplex_h.cuComplex;
      incx : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:957
   pragma Import (C, cublasCtbmv_v2, "cublasCtbmv_v2");

   function cublasZtbmv_v2
     (handle : cublasHandle_t;
      uplo : cublasFillMode_t;
      trans : cublasOperation_t;
      diag : cublasDiagType_t;
      n : int;
      k : int;
      A : access constant cuComplex_h.cuDoubleComplex;
      lda : int;
      x : access cuComplex_h.cuDoubleComplex;
      incx : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:968
   pragma Import (C, cublasZtbmv_v2, "cublasZtbmv_v2");

  -- TPMV  
   function cublasStpmv_v2
     (handle : cublasHandle_t;
      uplo : cublasFillMode_t;
      trans : cublasOperation_t;
      diag : cublasDiagType_t;
      n : int;
      AP : access float;
      x : access float;
      incx : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:980
   pragma Import (C, cublasStpmv_v2, "cublasStpmv_v2");

   function cublasDtpmv_v2
     (handle : cublasHandle_t;
      uplo : cublasFillMode_t;
      trans : cublasOperation_t;
      diag : cublasDiagType_t;
      n : int;
      AP : access double;
      x : access double;
      incx : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:989
   pragma Import (C, cublasDtpmv_v2, "cublasDtpmv_v2");

   function cublasCtpmv_v2
     (handle : cublasHandle_t;
      uplo : cublasFillMode_t;
      trans : cublasOperation_t;
      diag : cublasDiagType_t;
      n : int;
      AP : access constant cuComplex_h.cuComplex;
      x : access cuComplex_h.cuComplex;
      incx : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:998
   pragma Import (C, cublasCtpmv_v2, "cublasCtpmv_v2");

   function cublasZtpmv_v2
     (handle : cublasHandle_t;
      uplo : cublasFillMode_t;
      trans : cublasOperation_t;
      diag : cublasDiagType_t;
      n : int;
      AP : access constant cuComplex_h.cuDoubleComplex;
      x : access cuComplex_h.cuDoubleComplex;
      incx : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:1007
   pragma Import (C, cublasZtpmv_v2, "cublasZtpmv_v2");

  -- TRSV  
   function cublasStrsv_v2
     (handle : cublasHandle_t;
      uplo : cublasFillMode_t;
      trans : cublasOperation_t;
      diag : cublasDiagType_t;
      n : int;
      A : access float;
      lda : int;
      x : access float;
      incx : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:1017
   pragma Import (C, cublasStrsv_v2, "cublasStrsv_v2");

   function cublasDtrsv_v2
     (handle : cublasHandle_t;
      uplo : cublasFillMode_t;
      trans : cublasOperation_t;
      diag : cublasDiagType_t;
      n : int;
      A : access double;
      lda : int;
      x : access double;
      incx : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:1027
   pragma Import (C, cublasDtrsv_v2, "cublasDtrsv_v2");

   function cublasCtrsv_v2
     (handle : cublasHandle_t;
      uplo : cublasFillMode_t;
      trans : cublasOperation_t;
      diag : cublasDiagType_t;
      n : int;
      A : access constant cuComplex_h.cuComplex;
      lda : int;
      x : access cuComplex_h.cuComplex;
      incx : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:1037
   pragma Import (C, cublasCtrsv_v2, "cublasCtrsv_v2");

   function cublasZtrsv_v2
     (handle : cublasHandle_t;
      uplo : cublasFillMode_t;
      trans : cublasOperation_t;
      diag : cublasDiagType_t;
      n : int;
      A : access constant cuComplex_h.cuDoubleComplex;
      lda : int;
      x : access cuComplex_h.cuDoubleComplex;
      incx : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:1047
   pragma Import (C, cublasZtrsv_v2, "cublasZtrsv_v2");

  -- TPSV  
   function cublasStpsv_v2
     (handle : cublasHandle_t;
      uplo : cublasFillMode_t;
      trans : cublasOperation_t;
      diag : cublasDiagType_t;
      n : int;
      AP : access float;
      x : access float;
      incx : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:1058
   pragma Import (C, cublasStpsv_v2, "cublasStpsv_v2");

   function cublasDtpsv_v2
     (handle : cublasHandle_t;
      uplo : cublasFillMode_t;
      trans : cublasOperation_t;
      diag : cublasDiagType_t;
      n : int;
      AP : access double;
      x : access double;
      incx : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:1067
   pragma Import (C, cublasDtpsv_v2, "cublasDtpsv_v2");

   function cublasCtpsv_v2
     (handle : cublasHandle_t;
      uplo : cublasFillMode_t;
      trans : cublasOperation_t;
      diag : cublasDiagType_t;
      n : int;
      AP : access constant cuComplex_h.cuComplex;
      x : access cuComplex_h.cuComplex;
      incx : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:1076
   pragma Import (C, cublasCtpsv_v2, "cublasCtpsv_v2");

   function cublasZtpsv_v2
     (handle : cublasHandle_t;
      uplo : cublasFillMode_t;
      trans : cublasOperation_t;
      diag : cublasDiagType_t;
      n : int;
      AP : access constant cuComplex_h.cuDoubleComplex;
      x : access cuComplex_h.cuDoubleComplex;
      incx : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:1085
   pragma Import (C, cublasZtpsv_v2, "cublasZtpsv_v2");

  -- TBSV  
   function cublasStbsv_v2
     (handle : cublasHandle_t;
      uplo : cublasFillMode_t;
      trans : cublasOperation_t;
      diag : cublasDiagType_t;
      n : int;
      k : int;
      A : access float;
      lda : int;
      x : access float;
      incx : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:1094
   pragma Import (C, cublasStbsv_v2, "cublasStbsv_v2");

   function cublasDtbsv_v2
     (handle : cublasHandle_t;
      uplo : cublasFillMode_t;
      trans : cublasOperation_t;
      diag : cublasDiagType_t;
      n : int;
      k : int;
      A : access double;
      lda : int;
      x : access double;
      incx : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:1105
   pragma Import (C, cublasDtbsv_v2, "cublasDtbsv_v2");

   function cublasCtbsv_v2
     (handle : cublasHandle_t;
      uplo : cublasFillMode_t;
      trans : cublasOperation_t;
      diag : cublasDiagType_t;
      n : int;
      k : int;
      A : access constant cuComplex_h.cuComplex;
      lda : int;
      x : access cuComplex_h.cuComplex;
      incx : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:1116
   pragma Import (C, cublasCtbsv_v2, "cublasCtbsv_v2");

   function cublasZtbsv_v2
     (handle : cublasHandle_t;
      uplo : cublasFillMode_t;
      trans : cublasOperation_t;
      diag : cublasDiagType_t;
      n : int;
      k : int;
      A : access constant cuComplex_h.cuDoubleComplex;
      lda : int;
      x : access cuComplex_h.cuDoubleComplex;
      incx : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:1127
   pragma Import (C, cublasZtbsv_v2, "cublasZtbsv_v2");

  -- SYMV/HEMV  
   function cublasSsymv_v2
     (handle : cublasHandle_t;
      uplo : cublasFillMode_t;
      n : int;
      alpha : access float;
      A : access float;
      lda : int;
      x : access float;
      incx : int;
      beta : access float;
      y : access float;
      incy : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:1139
   pragma Import (C, cublasSsymv_v2, "cublasSsymv_v2");

  -- host or device pointer  
  -- host or device pointer  
   function cublasDsymv_v2
     (handle : cublasHandle_t;
      uplo : cublasFillMode_t;
      n : int;
      alpha : access double;
      A : access double;
      lda : int;
      x : access double;
      incx : int;
      beta : access double;
      y : access double;
      incy : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:1151
   pragma Import (C, cublasDsymv_v2, "cublasDsymv_v2");

  -- host or device pointer  
  -- host or device pointer  
   function cublasCsymv_v2
     (handle : cublasHandle_t;
      uplo : cublasFillMode_t;
      n : int;
      alpha : access constant cuComplex_h.cuComplex;
      A : access constant cuComplex_h.cuComplex;
      lda : int;
      x : access constant cuComplex_h.cuComplex;
      incx : int;
      beta : access constant cuComplex_h.cuComplex;
      y : access cuComplex_h.cuComplex;
      incy : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:1163
   pragma Import (C, cublasCsymv_v2, "cublasCsymv_v2");

  -- host or device pointer  
  -- host or device pointer  
   function cublasZsymv_v2
     (handle : cublasHandle_t;
      uplo : cublasFillMode_t;
      n : int;
      alpha : access constant cuComplex_h.cuDoubleComplex;
      A : access constant cuComplex_h.cuDoubleComplex;
      lda : int;
      x : access constant cuComplex_h.cuDoubleComplex;
      incx : int;
      beta : access constant cuComplex_h.cuDoubleComplex;
      y : access cuComplex_h.cuDoubleComplex;
      incy : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:1175
   pragma Import (C, cublasZsymv_v2, "cublasZsymv_v2");

  -- host or device pointer  
  -- host or device pointer  
   function cublasChemv_v2
     (handle : cublasHandle_t;
      uplo : cublasFillMode_t;
      n : int;
      alpha : access constant cuComplex_h.cuComplex;
      A : access constant cuComplex_h.cuComplex;
      lda : int;
      x : access constant cuComplex_h.cuComplex;
      incx : int;
      beta : access constant cuComplex_h.cuComplex;
      y : access cuComplex_h.cuComplex;
      incy : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:1187
   pragma Import (C, cublasChemv_v2, "cublasChemv_v2");

  -- host or device pointer  
  -- host or device pointer  
   function cublasZhemv_v2
     (handle : cublasHandle_t;
      uplo : cublasFillMode_t;
      n : int;
      alpha : access constant cuComplex_h.cuDoubleComplex;
      A : access constant cuComplex_h.cuDoubleComplex;
      lda : int;
      x : access constant cuComplex_h.cuDoubleComplex;
      incx : int;
      beta : access constant cuComplex_h.cuDoubleComplex;
      y : access cuComplex_h.cuDoubleComplex;
      incy : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:1199
   pragma Import (C, cublasZhemv_v2, "cublasZhemv_v2");

  -- host or device pointer  
  -- host or device pointer  
  -- SBMV/HBMV  
   function cublasSsbmv_v2
     (handle : cublasHandle_t;
      uplo : cublasFillMode_t;
      n : int;
      k : int;
      alpha : access float;
      A : access float;
      lda : int;
      x : access float;
      incx : int;
      beta : access float;
      y : access float;
      incy : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:1212
   pragma Import (C, cublasSsbmv_v2, "cublasSsbmv_v2");

  -- host or device pointer  
  -- host or device pointer  
   function cublasDsbmv_v2
     (handle : cublasHandle_t;
      uplo : cublasFillMode_t;
      n : int;
      k : int;
      alpha : access double;
      A : access double;
      lda : int;
      x : access double;
      incx : int;
      beta : access double;
      y : access double;
      incy : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:1225
   pragma Import (C, cublasDsbmv_v2, "cublasDsbmv_v2");

  -- host or device pointer  
  -- host or device pointer  
   function cublasChbmv_v2
     (handle : cublasHandle_t;
      uplo : cublasFillMode_t;
      n : int;
      k : int;
      alpha : access constant cuComplex_h.cuComplex;
      A : access constant cuComplex_h.cuComplex;
      lda : int;
      x : access constant cuComplex_h.cuComplex;
      incx : int;
      beta : access constant cuComplex_h.cuComplex;
      y : access cuComplex_h.cuComplex;
      incy : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:1238
   pragma Import (C, cublasChbmv_v2, "cublasChbmv_v2");

  -- host or device pointer  
  -- host or device pointer  
   function cublasZhbmv_v2
     (handle : cublasHandle_t;
      uplo : cublasFillMode_t;
      n : int;
      k : int;
      alpha : access constant cuComplex_h.cuDoubleComplex;
      A : access constant cuComplex_h.cuDoubleComplex;
      lda : int;
      x : access constant cuComplex_h.cuDoubleComplex;
      incx : int;
      beta : access constant cuComplex_h.cuDoubleComplex;
      y : access cuComplex_h.cuDoubleComplex;
      incy : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:1251
   pragma Import (C, cublasZhbmv_v2, "cublasZhbmv_v2");

  -- host or device pointer  
  -- host or device pointer  
  -- SPMV/HPMV  
   function cublasSspmv_v2
     (handle : cublasHandle_t;
      uplo : cublasFillMode_t;
      n : int;
      alpha : access float;
      AP : access float;
      x : access float;
      incx : int;
      beta : access float;
      y : access float;
      incy : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:1265
   pragma Import (C, cublasSspmv_v2, "cublasSspmv_v2");

  -- host or device pointer  
  -- host or device pointer  
   function cublasDspmv_v2
     (handle : cublasHandle_t;
      uplo : cublasFillMode_t;
      n : int;
      alpha : access double;
      AP : access double;
      x : access double;
      incx : int;
      beta : access double;
      y : access double;
      incy : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:1276
   pragma Import (C, cublasDspmv_v2, "cublasDspmv_v2");

  -- host or device pointer  
  -- host or device pointer  
   function cublasChpmv_v2
     (handle : cublasHandle_t;
      uplo : cublasFillMode_t;
      n : int;
      alpha : access constant cuComplex_h.cuComplex;
      AP : access constant cuComplex_h.cuComplex;
      x : access constant cuComplex_h.cuComplex;
      incx : int;
      beta : access constant cuComplex_h.cuComplex;
      y : access cuComplex_h.cuComplex;
      incy : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:1287
   pragma Import (C, cublasChpmv_v2, "cublasChpmv_v2");

  -- host or device pointer  
  -- host or device pointer  
   function cublasZhpmv_v2
     (handle : cublasHandle_t;
      uplo : cublasFillMode_t;
      n : int;
      alpha : access constant cuComplex_h.cuDoubleComplex;
      AP : access constant cuComplex_h.cuDoubleComplex;
      x : access constant cuComplex_h.cuDoubleComplex;
      incx : int;
      beta : access constant cuComplex_h.cuDoubleComplex;
      y : access cuComplex_h.cuDoubleComplex;
      incy : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:1298
   pragma Import (C, cublasZhpmv_v2, "cublasZhpmv_v2");

  -- host or device pointer  
  -- host or device pointer  
  -- GER  
   function cublasSger_v2
     (handle : cublasHandle_t;
      m : int;
      n : int;
      alpha : access float;
      x : access float;
      incx : int;
      y : access float;
      incy : int;
      A : access float;
      lda : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:1310
   pragma Import (C, cublasSger_v2, "cublasSger_v2");

  -- host or device pointer  
   function cublasDger_v2
     (handle : cublasHandle_t;
      m : int;
      n : int;
      alpha : access double;
      x : access double;
      incx : int;
      y : access double;
      incy : int;
      A : access double;
      lda : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:1321
   pragma Import (C, cublasDger_v2, "cublasDger_v2");

  -- host or device pointer  
   function cublasCgeru_v2
     (handle : cublasHandle_t;
      m : int;
      n : int;
      alpha : access constant cuComplex_h.cuComplex;
      x : access constant cuComplex_h.cuComplex;
      incx : int;
      y : access constant cuComplex_h.cuComplex;
      incy : int;
      A : access cuComplex_h.cuComplex;
      lda : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:1332
   pragma Import (C, cublasCgeru_v2, "cublasCgeru_v2");

  -- host or device pointer  
   function cublasCgerc_v2
     (handle : cublasHandle_t;
      m : int;
      n : int;
      alpha : access constant cuComplex_h.cuComplex;
      x : access constant cuComplex_h.cuComplex;
      incx : int;
      y : access constant cuComplex_h.cuComplex;
      incy : int;
      A : access cuComplex_h.cuComplex;
      lda : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:1343
   pragma Import (C, cublasCgerc_v2, "cublasCgerc_v2");

  -- host or device pointer  
   function cublasZgeru_v2
     (handle : cublasHandle_t;
      m : int;
      n : int;
      alpha : access constant cuComplex_h.cuDoubleComplex;
      x : access constant cuComplex_h.cuDoubleComplex;
      incx : int;
      y : access constant cuComplex_h.cuDoubleComplex;
      incy : int;
      A : access cuComplex_h.cuDoubleComplex;
      lda : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:1354
   pragma Import (C, cublasZgeru_v2, "cublasZgeru_v2");

  -- host or device pointer  
   function cublasZgerc_v2
     (handle : cublasHandle_t;
      m : int;
      n : int;
      alpha : access constant cuComplex_h.cuDoubleComplex;
      x : access constant cuComplex_h.cuDoubleComplex;
      incx : int;
      y : access constant cuComplex_h.cuDoubleComplex;
      incy : int;
      A : access cuComplex_h.cuDoubleComplex;
      lda : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:1365
   pragma Import (C, cublasZgerc_v2, "cublasZgerc_v2");

  -- host or device pointer  
  -- SYR/HER  
   function cublasSsyr_v2
     (handle : cublasHandle_t;
      uplo : cublasFillMode_t;
      n : int;
      alpha : access float;
      x : access float;
      incx : int;
      A : access float;
      lda : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:1377
   pragma Import (C, cublasSsyr_v2, "cublasSsyr_v2");

  -- host or device pointer  
   function cublasDsyr_v2
     (handle : cublasHandle_t;
      uplo : cublasFillMode_t;
      n : int;
      alpha : access double;
      x : access double;
      incx : int;
      A : access double;
      lda : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:1386
   pragma Import (C, cublasDsyr_v2, "cublasDsyr_v2");

  -- host or device pointer  
   function cublasCsyr_v2
     (handle : cublasHandle_t;
      uplo : cublasFillMode_t;
      n : int;
      alpha : access constant cuComplex_h.cuComplex;
      x : access constant cuComplex_h.cuComplex;
      incx : int;
      A : access cuComplex_h.cuComplex;
      lda : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:1395
   pragma Import (C, cublasCsyr_v2, "cublasCsyr_v2");

  -- host or device pointer  
   function cublasZsyr_v2
     (handle : cublasHandle_t;
      uplo : cublasFillMode_t;
      n : int;
      alpha : access constant cuComplex_h.cuDoubleComplex;
      x : access constant cuComplex_h.cuDoubleComplex;
      incx : int;
      A : access cuComplex_h.cuDoubleComplex;
      lda : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:1404
   pragma Import (C, cublasZsyr_v2, "cublasZsyr_v2");

  -- host or device pointer  
   function cublasCher_v2
     (handle : cublasHandle_t;
      uplo : cublasFillMode_t;
      n : int;
      alpha : access float;
      x : access constant cuComplex_h.cuComplex;
      incx : int;
      A : access cuComplex_h.cuComplex;
      lda : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:1413
   pragma Import (C, cublasCher_v2, "cublasCher_v2");

  -- host or device pointer  
   function cublasZher_v2
     (handle : cublasHandle_t;
      uplo : cublasFillMode_t;
      n : int;
      alpha : access double;
      x : access constant cuComplex_h.cuDoubleComplex;
      incx : int;
      A : access cuComplex_h.cuDoubleComplex;
      lda : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:1422
   pragma Import (C, cublasZher_v2, "cublasZher_v2");

  -- host or device pointer  
  -- SPR/HPR  
   function cublasSspr_v2
     (handle : cublasHandle_t;
      uplo : cublasFillMode_t;
      n : int;
      alpha : access float;
      x : access float;
      incx : int;
      AP : access float) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:1432
   pragma Import (C, cublasSspr_v2, "cublasSspr_v2");

  -- host or device pointer  
   function cublasDspr_v2
     (handle : cublasHandle_t;
      uplo : cublasFillMode_t;
      n : int;
      alpha : access double;
      x : access double;
      incx : int;
      AP : access double) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:1440
   pragma Import (C, cublasDspr_v2, "cublasDspr_v2");

  -- host or device pointer  
   function cublasChpr_v2
     (handle : cublasHandle_t;
      uplo : cublasFillMode_t;
      n : int;
      alpha : access float;
      x : access constant cuComplex_h.cuComplex;
      incx : int;
      AP : access cuComplex_h.cuComplex) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:1448
   pragma Import (C, cublasChpr_v2, "cublasChpr_v2");

  -- host or device pointer  
   function cublasZhpr_v2
     (handle : cublasHandle_t;
      uplo : cublasFillMode_t;
      n : int;
      alpha : access double;
      x : access constant cuComplex_h.cuDoubleComplex;
      incx : int;
      AP : access cuComplex_h.cuDoubleComplex) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:1456
   pragma Import (C, cublasZhpr_v2, "cublasZhpr_v2");

  -- host or device pointer  
  -- SYR2/HER2  
   function cublasSsyr2_v2
     (handle : cublasHandle_t;
      uplo : cublasFillMode_t;
      n : int;
      alpha : access float;
      x : access float;
      incx : int;
      y : access float;
      incy : int;
      A : access float;
      lda : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:1465
   pragma Import (C, cublasSsyr2_v2, "cublasSsyr2_v2");

  -- host or device pointer  
   function cublasDsyr2_v2
     (handle : cublasHandle_t;
      uplo : cublasFillMode_t;
      n : int;
      alpha : access double;
      x : access double;
      incx : int;
      y : access double;
      incy : int;
      A : access double;
      lda : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:1476
   pragma Import (C, cublasDsyr2_v2, "cublasDsyr2_v2");

  -- host or device pointer  
   function cublasCsyr2_v2
     (handle : cublasHandle_t;
      uplo : cublasFillMode_t;
      n : int;
      alpha : access constant cuComplex_h.cuComplex;
      x : access constant cuComplex_h.cuComplex;
      incx : int;
      y : access constant cuComplex_h.cuComplex;
      incy : int;
      A : access cuComplex_h.cuComplex;
      lda : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:1487
   pragma Import (C, cublasCsyr2_v2, "cublasCsyr2_v2");

  -- host or device pointer  
   function cublasZsyr2_v2
     (handle : cublasHandle_t;
      uplo : cublasFillMode_t;
      n : int;
      alpha : access constant cuComplex_h.cuDoubleComplex;
      x : access constant cuComplex_h.cuDoubleComplex;
      incx : int;
      y : access constant cuComplex_h.cuDoubleComplex;
      incy : int;
      A : access cuComplex_h.cuDoubleComplex;
      lda : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:1497
   pragma Import (C, cublasZsyr2_v2, "cublasZsyr2_v2");

  -- host or device pointer  
   function cublasCher2_v2
     (handle : cublasHandle_t;
      uplo : cublasFillMode_t;
      n : int;
      alpha : access constant cuComplex_h.cuComplex;
      x : access constant cuComplex_h.cuComplex;
      incx : int;
      y : access constant cuComplex_h.cuComplex;
      incy : int;
      A : access cuComplex_h.cuComplex;
      lda : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:1509
   pragma Import (C, cublasCher2_v2, "cublasCher2_v2");

  -- host or device pointer  
   function cublasZher2_v2
     (handle : cublasHandle_t;
      uplo : cublasFillMode_t;
      n : int;
      alpha : access constant cuComplex_h.cuDoubleComplex;
      x : access constant cuComplex_h.cuDoubleComplex;
      incx : int;
      y : access constant cuComplex_h.cuDoubleComplex;
      incy : int;
      A : access cuComplex_h.cuDoubleComplex;
      lda : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:1519
   pragma Import (C, cublasZher2_v2, "cublasZher2_v2");

  -- host or device pointer  
  -- SPR2/HPR2  
   function cublasSspr2_v2
     (handle : cublasHandle_t;
      uplo : cublasFillMode_t;
      n : int;
      alpha : access float;
      x : access float;
      incx : int;
      y : access float;
      incy : int;
      AP : access float) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:1531
   pragma Import (C, cublasSspr2_v2, "cublasSspr2_v2");

  -- host or device pointer  
   function cublasDspr2_v2
     (handle : cublasHandle_t;
      uplo : cublasFillMode_t;
      n : int;
      alpha : access double;
      x : access double;
      incx : int;
      y : access double;
      incy : int;
      AP : access double) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:1541
   pragma Import (C, cublasDspr2_v2, "cublasDspr2_v2");

  -- host or device pointer  
   function cublasChpr2_v2
     (handle : cublasHandle_t;
      uplo : cublasFillMode_t;
      n : int;
      alpha : access constant cuComplex_h.cuComplex;
      x : access constant cuComplex_h.cuComplex;
      incx : int;
      y : access constant cuComplex_h.cuComplex;
      incy : int;
      AP : access cuComplex_h.cuComplex) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:1552
   pragma Import (C, cublasChpr2_v2, "cublasChpr2_v2");

  -- host or device pointer  
   function cublasZhpr2_v2
     (handle : cublasHandle_t;
      uplo : cublasFillMode_t;
      n : int;
      alpha : access constant cuComplex_h.cuDoubleComplex;
      x : access constant cuComplex_h.cuDoubleComplex;
      incx : int;
      y : access constant cuComplex_h.cuDoubleComplex;
      incy : int;
      AP : access cuComplex_h.cuDoubleComplex) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:1562
   pragma Import (C, cublasZhpr2_v2, "cublasZhpr2_v2");

  -- host or device pointer  
  -- ---------------- CUBLAS BLAS3 functions ----------------  
  -- GEMM  
   function cublasSgemm_v2
     (handle : cublasHandle_t;
      transa : cublasOperation_t;
      transb : cublasOperation_t;
      m : int;
      n : int;
      k : int;
      alpha : access float;
      A : access float;
      lda : int;
      B : access float;
      ldb : int;
      beta : access float;
      C : access float;
      ldc : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:1575
   pragma Import (C, cublasSgemm_v2, "cublasSgemm_v2");

  -- host or device pointer  
  -- host or device pointer  
   function cublasDgemm_v2
     (handle : cublasHandle_t;
      transa : cublasOperation_t;
      transb : cublasOperation_t;
      m : int;
      n : int;
      k : int;
      alpha : access double;
      A : access double;
      lda : int;
      B : access double;
      ldb : int;
      beta : access double;
      C : access double;
      ldc : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:1590
   pragma Import (C, cublasDgemm_v2, "cublasDgemm_v2");

  -- host or device pointer  
  -- host or device pointer  
   function cublasCgemm_v2
     (handle : cublasHandle_t;
      transa : cublasOperation_t;
      transb : cublasOperation_t;
      m : int;
      n : int;
      k : int;
      alpha : access constant cuComplex_h.cuComplex;
      A : access constant cuComplex_h.cuComplex;
      lda : int;
      B : access constant cuComplex_h.cuComplex;
      ldb : int;
      beta : access constant cuComplex_h.cuComplex;
      C : access cuComplex_h.cuComplex;
      ldc : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:1605
   pragma Import (C, cublasCgemm_v2, "cublasCgemm_v2");

  -- host or device pointer  
  -- host or device pointer  
   function cublasCgemm3m
     (handle : cublasHandle_t;
      transa : cublasOperation_t;
      transb : cublasOperation_t;
      m : int;
      n : int;
      k : int;
      alpha : access constant cuComplex_h.cuComplex;
      A : access constant cuComplex_h.cuComplex;
      lda : int;
      B : access constant cuComplex_h.cuComplex;
      ldb : int;
      beta : access constant cuComplex_h.cuComplex;
      C : access cuComplex_h.cuComplex;
      ldc : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:1620
   pragma Import (C, cublasCgemm3m, "cublasCgemm3m");

  -- host or device pointer  
  -- host or device pointer  
   function cublasCgemm3mEx
     (handle : cublasHandle_t;
      transa : cublasOperation_t;
      transb : cublasOperation_t;
      m : int;
      n : int;
      k : int;
      alpha : access constant cuComplex_h.cuComplex;
      A : System.Address;
      Atype : library_types_h.cudaDataType;
      lda : int;
      B : System.Address;
      Btype : library_types_h.cudaDataType;
      ldb : int;
      beta : access constant cuComplex_h.cuComplex;
      C : System.Address;
      Ctype : library_types_h.cudaDataType;
      ldc : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:1634
   pragma Import (C, cublasCgemm3mEx, "cublasCgemm3mEx");

   function cublasZgemm_v2
     (handle : cublasHandle_t;
      transa : cublasOperation_t;
      transb : cublasOperation_t;
      m : int;
      n : int;
      k : int;
      alpha : access constant cuComplex_h.cuDoubleComplex;
      A : access constant cuComplex_h.cuDoubleComplex;
      lda : int;
      B : access constant cuComplex_h.cuDoubleComplex;
      ldb : int;
      beta : access constant cuComplex_h.cuDoubleComplex;
      C : access cuComplex_h.cuDoubleComplex;
      ldc : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:1650
   pragma Import (C, cublasZgemm_v2, "cublasZgemm_v2");

  -- host or device pointer  
  -- host or device pointer  
   function cublasZgemm3m
     (handle : cublasHandle_t;
      transa : cublasOperation_t;
      transb : cublasOperation_t;
      m : int;
      n : int;
      k : int;
      alpha : access constant cuComplex_h.cuDoubleComplex;
      A : access constant cuComplex_h.cuDoubleComplex;
      lda : int;
      B : access constant cuComplex_h.cuDoubleComplex;
      ldb : int;
      beta : access constant cuComplex_h.cuDoubleComplex;
      C : access cuComplex_h.cuDoubleComplex;
      ldc : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:1665
   pragma Import (C, cublasZgemm3m, "cublasZgemm3m");

  -- host or device pointer  
  -- host or device pointer  
   function cublasHgemm
     (handle : cublasHandle_t;
      transa : cublasOperation_t;
      transb : cublasOperation_t;
      m : int;
      n : int;
      k : int;
      alpha : access constant cuda_fp16_h.uu_half;
      A : access constant cuda_fp16_h.uu_half;
      lda : int;
      B : access constant cuda_fp16_h.uu_half;
      ldb : int;
      beta : access constant cuda_fp16_h.uu_half;
      C : access cuda_fp16_h.uu_half;
      ldc : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:1680
   pragma Import (C, cublasHgemm, "cublasHgemm");

  -- host or device pointer  
  -- host or device pointer  
  -- IO in FP16/FP32, computation in float  
   function cublasSgemmEx
     (handle : cublasHandle_t;
      transa : cublasOperation_t;
      transb : cublasOperation_t;
      m : int;
      n : int;
      k : int;
      alpha : access float;
      A : System.Address;
      Atype : library_types_h.cudaDataType;
      lda : int;
      B : System.Address;
      Btype : library_types_h.cudaDataType;
      ldb : int;
      beta : access float;
      C : System.Address;
      Ctype : library_types_h.cudaDataType;
      ldc : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:1695
   pragma Import (C, cublasSgemmEx, "cublasSgemmEx");

  -- host or device pointer  
  -- host or device pointer  
   function cublasGemmEx
     (handle : cublasHandle_t;
      transa : cublasOperation_t;
      transb : cublasOperation_t;
      m : int;
      n : int;
      k : int;
      alpha : System.Address;
      A : System.Address;
      Atype : library_types_h.cudaDataType;
      lda : int;
      B : System.Address;
      Btype : library_types_h.cudaDataType;
      ldb : int;
      beta : System.Address;
      C : System.Address;
      Ctype : library_types_h.cudaDataType;
      ldc : int;
      computeType : library_types_h.cudaDataType;
      algo : cublasGemmAlgo_t) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:1713
   pragma Import (C, cublasGemmEx, "cublasGemmEx");

  -- host or device pointer  
  -- host or device pointer  
  -- IO in Int8 complex/cuComplex, computation in cuComplex  
   function cublasCgemmEx
     (handle : cublasHandle_t;
      transa : cublasOperation_t;
      transb : cublasOperation_t;
      m : int;
      n : int;
      k : int;
      alpha : access constant cuComplex_h.cuComplex;
      A : System.Address;
      Atype : library_types_h.cudaDataType;
      lda : int;
      B : System.Address;
      Btype : library_types_h.cudaDataType;
      ldb : int;
      beta : access constant cuComplex_h.cuComplex;
      C : System.Address;
      Ctype : library_types_h.cudaDataType;
      ldc : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:1734
   pragma Import (C, cublasCgemmEx, "cublasCgemmEx");

   function cublasUint8gemmBias
     (handle : cublasHandle_t;
      transa : cublasOperation_t;
      transb : cublasOperation_t;
      transc : cublasOperation_t;
      m : int;
      n : int;
      k : int;
      A : access unsigned_char;
      A_bias : int;
      lda : int;
      B : access unsigned_char;
      B_bias : int;
      ldb : int;
      C : access unsigned_char;
      C_bias : int;
      ldc : int;
      C_mult : int;
      C_shift : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:1749
   pragma Import (C, cublasUint8gemmBias, "cublasUint8gemmBias");

  -- SYRK  
   function cublasSsyrk_v2
     (handle : cublasHandle_t;
      uplo : cublasFillMode_t;
      trans : cublasOperation_t;
      n : int;
      k : int;
      alpha : access float;
      A : access float;
      lda : int;
      beta : access float;
      C : access float;
      ldc : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:1758
   pragma Import (C, cublasSsyrk_v2, "cublasSsyrk_v2");

  -- host or device pointer  
  -- host or device pointer  
   function cublasDsyrk_v2
     (handle : cublasHandle_t;
      uplo : cublasFillMode_t;
      trans : cublasOperation_t;
      n : int;
      k : int;
      alpha : access double;
      A : access double;
      lda : int;
      beta : access double;
      C : access double;
      ldc : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:1770
   pragma Import (C, cublasDsyrk_v2, "cublasDsyrk_v2");

  -- host or device pointer  
  -- host or device pointer  
   function cublasCsyrk_v2
     (handle : cublasHandle_t;
      uplo : cublasFillMode_t;
      trans : cublasOperation_t;
      n : int;
      k : int;
      alpha : access constant cuComplex_h.cuComplex;
      A : access constant cuComplex_h.cuComplex;
      lda : int;
      beta : access constant cuComplex_h.cuComplex;
      C : access cuComplex_h.cuComplex;
      ldc : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:1782
   pragma Import (C, cublasCsyrk_v2, "cublasCsyrk_v2");

  -- host or device pointer  
  -- host or device pointer  
   function cublasZsyrk_v2
     (handle : cublasHandle_t;
      uplo : cublasFillMode_t;
      trans : cublasOperation_t;
      n : int;
      k : int;
      alpha : access constant cuComplex_h.cuDoubleComplex;
      A : access constant cuComplex_h.cuDoubleComplex;
      lda : int;
      beta : access constant cuComplex_h.cuDoubleComplex;
      C : access cuComplex_h.cuDoubleComplex;
      ldc : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:1794
   pragma Import (C, cublasZsyrk_v2, "cublasZsyrk_v2");

  -- host or device pointer  
  -- host or device pointer  
  -- IO in Int8 complex/cuComplex, computation in cuComplex  
   function cublasCsyrkEx
     (handle : cublasHandle_t;
      uplo : cublasFillMode_t;
      trans : cublasOperation_t;
      n : int;
      k : int;
      alpha : access constant cuComplex_h.cuComplex;
      A : System.Address;
      Atype : library_types_h.cudaDataType;
      lda : int;
      beta : access constant cuComplex_h.cuComplex;
      C : System.Address;
      Ctype : library_types_h.cudaDataType;
      ldc : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:1806
   pragma Import (C, cublasCsyrkEx, "cublasCsyrkEx");

  -- host or device pointer  
  -- host or device pointer  
  -- IO in Int8 complex/cuComplex, computation in cuComplex, Gaussian math  
   function cublasCsyrk3mEx
     (handle : cublasHandle_t;
      uplo : cublasFillMode_t;
      trans : cublasOperation_t;
      n : int;
      k : int;
      alpha : access constant cuComplex_h.cuComplex;
      A : System.Address;
      Atype : library_types_h.cudaDataType;
      lda : int;
      beta : access constant cuComplex_h.cuComplex;
      C : System.Address;
      Ctype : library_types_h.cudaDataType;
      ldc : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:1821
   pragma Import (C, cublasCsyrk3mEx, "cublasCsyrk3mEx");

  -- HERK  
   function cublasCherk_v2
     (handle : cublasHandle_t;
      uplo : cublasFillMode_t;
      trans : cublasOperation_t;
      n : int;
      k : int;
      alpha : access float;
      A : access constant cuComplex_h.cuComplex;
      lda : int;
      beta : access float;
      C : access cuComplex_h.cuComplex;
      ldc : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:1836
   pragma Import (C, cublasCherk_v2, "cublasCherk_v2");

  -- host or device pointer  
  -- host or device pointer  
   function cublasZherk_v2
     (handle : cublasHandle_t;
      uplo : cublasFillMode_t;
      trans : cublasOperation_t;
      n : int;
      k : int;
      alpha : access double;
      A : access constant cuComplex_h.cuDoubleComplex;
      lda : int;
      beta : access double;
      C : access cuComplex_h.cuDoubleComplex;
      ldc : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:1848
   pragma Import (C, cublasZherk_v2, "cublasZherk_v2");

  -- host or device pointer  
  -- host or device pointer  
  -- IO in Int8 complex/cuComplex, computation in cuComplex  
   function cublasCherkEx
     (handle : cublasHandle_t;
      uplo : cublasFillMode_t;
      trans : cublasOperation_t;
      n : int;
      k : int;
      alpha : access float;
      A : System.Address;
      Atype : library_types_h.cudaDataType;
      lda : int;
      beta : access float;
      C : System.Address;
      Ctype : library_types_h.cudaDataType;
      ldc : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:1861
   pragma Import (C, cublasCherkEx, "cublasCherkEx");

  -- host or device pointer  
  -- host or device pointer  
  -- IO in Int8 complex/cuComplex, computation in cuComplex, Gaussian math  
   function cublasCherk3mEx
     (handle : cublasHandle_t;
      uplo : cublasFillMode_t;
      trans : cublasOperation_t;
      n : int;
      k : int;
      alpha : access float;
      A : System.Address;
      Atype : library_types_h.cudaDataType;
      lda : int;
      beta : access float;
      C : System.Address;
      Ctype : library_types_h.cudaDataType;
      ldc : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:1876
   pragma Import (C, cublasCherk3mEx, "cublasCherk3mEx");

  -- SYR2K  
   function cublasSsyr2k_v2
     (handle : cublasHandle_t;
      uplo : cublasFillMode_t;
      trans : cublasOperation_t;
      n : int;
      k : int;
      alpha : access float;
      A : access float;
      lda : int;
      B : access float;
      ldb : int;
      beta : access float;
      C : access float;
      ldc : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:1892
   pragma Import (C, cublasSsyr2k_v2, "cublasSsyr2k_v2");

  -- host or device pointer  
  -- host or device pointer  
   function cublasDsyr2k_v2
     (handle : cublasHandle_t;
      uplo : cublasFillMode_t;
      trans : cublasOperation_t;
      n : int;
      k : int;
      alpha : access double;
      A : access double;
      lda : int;
      B : access double;
      ldb : int;
      beta : access double;
      C : access double;
      ldc : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:1906
   pragma Import (C, cublasDsyr2k_v2, "cublasDsyr2k_v2");

  -- host or device pointer  
  -- host or device pointer  
   function cublasCsyr2k_v2
     (handle : cublasHandle_t;
      uplo : cublasFillMode_t;
      trans : cublasOperation_t;
      n : int;
      k : int;
      alpha : access constant cuComplex_h.cuComplex;
      A : access constant cuComplex_h.cuComplex;
      lda : int;
      B : access constant cuComplex_h.cuComplex;
      ldb : int;
      beta : access constant cuComplex_h.cuComplex;
      C : access cuComplex_h.cuComplex;
      ldc : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:1920
   pragma Import (C, cublasCsyr2k_v2, "cublasCsyr2k_v2");

  -- host or device pointer  
  -- host or device pointer  
   function cublasZsyr2k_v2
     (handle : cublasHandle_t;
      uplo : cublasFillMode_t;
      trans : cublasOperation_t;
      n : int;
      k : int;
      alpha : access constant cuComplex_h.cuDoubleComplex;
      A : access constant cuComplex_h.cuDoubleComplex;
      lda : int;
      B : access constant cuComplex_h.cuDoubleComplex;
      ldb : int;
      beta : access constant cuComplex_h.cuDoubleComplex;
      C : access cuComplex_h.cuDoubleComplex;
      ldc : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:1934
   pragma Import (C, cublasZsyr2k_v2, "cublasZsyr2k_v2");

  -- host or device pointer  
  -- host or device pointer  
  -- HER2K  
   function cublasCher2k_v2
     (handle : cublasHandle_t;
      uplo : cublasFillMode_t;
      trans : cublasOperation_t;
      n : int;
      k : int;
      alpha : access constant cuComplex_h.cuComplex;
      A : access constant cuComplex_h.cuComplex;
      lda : int;
      B : access constant cuComplex_h.cuComplex;
      ldb : int;
      beta : access float;
      C : access cuComplex_h.cuComplex;
      ldc : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:1948
   pragma Import (C, cublasCher2k_v2, "cublasCher2k_v2");

  -- host or device pointer  
  -- host or device pointer  
   function cublasZher2k_v2
     (handle : cublasHandle_t;
      uplo : cublasFillMode_t;
      trans : cublasOperation_t;
      n : int;
      k : int;
      alpha : access constant cuComplex_h.cuDoubleComplex;
      A : access constant cuComplex_h.cuDoubleComplex;
      lda : int;
      B : access constant cuComplex_h.cuDoubleComplex;
      ldb : int;
      beta : access double;
      C : access cuComplex_h.cuDoubleComplex;
      ldc : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:1962
   pragma Import (C, cublasZher2k_v2, "cublasZher2k_v2");

  -- host or device pointer  
  -- host or device pointer  
  -- SYRKX : eXtended SYRK 
   function cublasSsyrkx
     (handle : cublasHandle_t;
      uplo : cublasFillMode_t;
      trans : cublasOperation_t;
      n : int;
      k : int;
      alpha : access float;
      A : access float;
      lda : int;
      B : access float;
      ldb : int;
      beta : access float;
      C : access float;
      ldc : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:1976
   pragma Import (C, cublasSsyrkx, "cublasSsyrkx");

  -- host or device pointer  
  -- host or device pointer  
   function cublasDsyrkx
     (handle : cublasHandle_t;
      uplo : cublasFillMode_t;
      trans : cublasOperation_t;
      n : int;
      k : int;
      alpha : access double;
      A : access double;
      lda : int;
      B : access double;
      ldb : int;
      beta : access double;
      C : access double;
      ldc : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:1990
   pragma Import (C, cublasDsyrkx, "cublasDsyrkx");

  -- host or device pointer  
  -- host or device pointer  
   function cublasCsyrkx
     (handle : cublasHandle_t;
      uplo : cublasFillMode_t;
      trans : cublasOperation_t;
      n : int;
      k : int;
      alpha : access constant cuComplex_h.cuComplex;
      A : access constant cuComplex_h.cuComplex;
      lda : int;
      B : access constant cuComplex_h.cuComplex;
      ldb : int;
      beta : access constant cuComplex_h.cuComplex;
      C : access cuComplex_h.cuComplex;
      ldc : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:2004
   pragma Import (C, cublasCsyrkx, "cublasCsyrkx");

  -- host or device pointer  
  -- host or device pointer  
   function cublasZsyrkx
     (handle : cublasHandle_t;
      uplo : cublasFillMode_t;
      trans : cublasOperation_t;
      n : int;
      k : int;
      alpha : access constant cuComplex_h.cuDoubleComplex;
      A : access constant cuComplex_h.cuDoubleComplex;
      lda : int;
      B : access constant cuComplex_h.cuDoubleComplex;
      ldb : int;
      beta : access constant cuComplex_h.cuDoubleComplex;
      C : access cuComplex_h.cuDoubleComplex;
      ldc : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:2018
   pragma Import (C, cublasZsyrkx, "cublasZsyrkx");

  -- host or device pointer  
  -- host or device pointer  
  -- HERKX : eXtended HERK  
   function cublasCherkx
     (handle : cublasHandle_t;
      uplo : cublasFillMode_t;
      trans : cublasOperation_t;
      n : int;
      k : int;
      alpha : access constant cuComplex_h.cuComplex;
      A : access constant cuComplex_h.cuComplex;
      lda : int;
      B : access constant cuComplex_h.cuComplex;
      ldb : int;
      beta : access float;
      C : access cuComplex_h.cuComplex;
      ldc : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:2032
   pragma Import (C, cublasCherkx, "cublasCherkx");

  -- host or device pointer  
  -- host or device pointer  
   function cublasZherkx
     (handle : cublasHandle_t;
      uplo : cublasFillMode_t;
      trans : cublasOperation_t;
      n : int;
      k : int;
      alpha : access constant cuComplex_h.cuDoubleComplex;
      A : access constant cuComplex_h.cuDoubleComplex;
      lda : int;
      B : access constant cuComplex_h.cuDoubleComplex;
      ldb : int;
      beta : access double;
      C : access cuComplex_h.cuDoubleComplex;
      ldc : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:2046
   pragma Import (C, cublasZherkx, "cublasZherkx");

  -- host or device pointer  
  -- host or device pointer  
  -- SYMM  
   function cublasSsymm_v2
     (handle : cublasHandle_t;
      side : cublasSideMode_t;
      uplo : cublasFillMode_t;
      m : int;
      n : int;
      alpha : access float;
      A : access float;
      lda : int;
      B : access float;
      ldb : int;
      beta : access float;
      C : access float;
      ldc : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:2060
   pragma Import (C, cublasSsymm_v2, "cublasSsymm_v2");

  -- host or device pointer  
  -- host or device pointer  
   function cublasDsymm_v2
     (handle : cublasHandle_t;
      side : cublasSideMode_t;
      uplo : cublasFillMode_t;
      m : int;
      n : int;
      alpha : access double;
      A : access double;
      lda : int;
      B : access double;
      ldb : int;
      beta : access double;
      C : access double;
      ldc : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:2074
   pragma Import (C, cublasDsymm_v2, "cublasDsymm_v2");

  -- host or device pointer  
  -- host or device pointer  
   function cublasCsymm_v2
     (handle : cublasHandle_t;
      side : cublasSideMode_t;
      uplo : cublasFillMode_t;
      m : int;
      n : int;
      alpha : access constant cuComplex_h.cuComplex;
      A : access constant cuComplex_h.cuComplex;
      lda : int;
      B : access constant cuComplex_h.cuComplex;
      ldb : int;
      beta : access constant cuComplex_h.cuComplex;
      C : access cuComplex_h.cuComplex;
      ldc : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:2088
   pragma Import (C, cublasCsymm_v2, "cublasCsymm_v2");

  -- host or device pointer  
  -- host or device pointer  
   function cublasZsymm_v2
     (handle : cublasHandle_t;
      side : cublasSideMode_t;
      uplo : cublasFillMode_t;
      m : int;
      n : int;
      alpha : access constant cuComplex_h.cuDoubleComplex;
      A : access constant cuComplex_h.cuDoubleComplex;
      lda : int;
      B : access constant cuComplex_h.cuDoubleComplex;
      ldb : int;
      beta : access constant cuComplex_h.cuDoubleComplex;
      C : access cuComplex_h.cuDoubleComplex;
      ldc : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:2102
   pragma Import (C, cublasZsymm_v2, "cublasZsymm_v2");

  -- host or device pointer  
  -- host or device pointer  
  -- HEMM  
   function cublasChemm_v2
     (handle : cublasHandle_t;
      side : cublasSideMode_t;
      uplo : cublasFillMode_t;
      m : int;
      n : int;
      alpha : access constant cuComplex_h.cuComplex;
      A : access constant cuComplex_h.cuComplex;
      lda : int;
      B : access constant cuComplex_h.cuComplex;
      ldb : int;
      beta : access constant cuComplex_h.cuComplex;
      C : access cuComplex_h.cuComplex;
      ldc : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:2117
   pragma Import (C, cublasChemm_v2, "cublasChemm_v2");

  -- host or device pointer  
  -- host or device pointer  
   function cublasZhemm_v2
     (handle : cublasHandle_t;
      side : cublasSideMode_t;
      uplo : cublasFillMode_t;
      m : int;
      n : int;
      alpha : access constant cuComplex_h.cuDoubleComplex;
      A : access constant cuComplex_h.cuDoubleComplex;
      lda : int;
      B : access constant cuComplex_h.cuDoubleComplex;
      ldb : int;
      beta : access constant cuComplex_h.cuDoubleComplex;
      C : access cuComplex_h.cuDoubleComplex;
      ldc : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:2131
   pragma Import (C, cublasZhemm_v2, "cublasZhemm_v2");

  -- host or device pointer  
  -- host or device pointer  
  -- TRSM  
   function cublasStrsm_v2
     (handle : cublasHandle_t;
      side : cublasSideMode_t;
      uplo : cublasFillMode_t;
      trans : cublasOperation_t;
      diag : cublasDiagType_t;
      m : int;
      n : int;
      alpha : access float;
      A : access float;
      lda : int;
      B : access float;
      ldb : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:2146
   pragma Import (C, cublasStrsm_v2, "cublasStrsm_v2");

  -- host or device pointer  
   function cublasDtrsm_v2
     (handle : cublasHandle_t;
      side : cublasSideMode_t;
      uplo : cublasFillMode_t;
      trans : cublasOperation_t;
      diag : cublasDiagType_t;
      m : int;
      n : int;
      alpha : access double;
      A : access double;
      lda : int;
      B : access double;
      ldb : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:2160
   pragma Import (C, cublasDtrsm_v2, "cublasDtrsm_v2");

  -- host or device pointer  
   function cublasCtrsm_v2
     (handle : cublasHandle_t;
      side : cublasSideMode_t;
      uplo : cublasFillMode_t;
      trans : cublasOperation_t;
      diag : cublasDiagType_t;
      m : int;
      n : int;
      alpha : access constant cuComplex_h.cuComplex;
      A : access constant cuComplex_h.cuComplex;
      lda : int;
      B : access cuComplex_h.cuComplex;
      ldb : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:2173
   pragma Import (C, cublasCtrsm_v2, "cublasCtrsm_v2");

  -- host or device pointer  
   function cublasZtrsm_v2
     (handle : cublasHandle_t;
      side : cublasSideMode_t;
      uplo : cublasFillMode_t;
      trans : cublasOperation_t;
      diag : cublasDiagType_t;
      m : int;
      n : int;
      alpha : access constant cuComplex_h.cuDoubleComplex;
      A : access constant cuComplex_h.cuDoubleComplex;
      lda : int;
      B : access cuComplex_h.cuDoubleComplex;
      ldb : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:2186
   pragma Import (C, cublasZtrsm_v2, "cublasZtrsm_v2");

  -- host or device pointer  
  -- TRMM  
   function cublasStrmm_v2
     (handle : cublasHandle_t;
      side : cublasSideMode_t;
      uplo : cublasFillMode_t;
      trans : cublasOperation_t;
      diag : cublasDiagType_t;
      m : int;
      n : int;
      alpha : access float;
      A : access float;
      lda : int;
      B : access float;
      ldb : int;
      C : access float;
      ldc : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:2200
   pragma Import (C, cublasStrmm_v2, "cublasStrmm_v2");

  -- host or device pointer  
   function cublasDtrmm_v2
     (handle : cublasHandle_t;
      side : cublasSideMode_t;
      uplo : cublasFillMode_t;
      trans : cublasOperation_t;
      diag : cublasDiagType_t;
      m : int;
      n : int;
      alpha : access double;
      A : access double;
      lda : int;
      B : access double;
      ldb : int;
      C : access double;
      ldc : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:2215
   pragma Import (C, cublasDtrmm_v2, "cublasDtrmm_v2");

  -- host or device pointer  
   function cublasCtrmm_v2
     (handle : cublasHandle_t;
      side : cublasSideMode_t;
      uplo : cublasFillMode_t;
      trans : cublasOperation_t;
      diag : cublasDiagType_t;
      m : int;
      n : int;
      alpha : access constant cuComplex_h.cuComplex;
      A : access constant cuComplex_h.cuComplex;
      lda : int;
      B : access constant cuComplex_h.cuComplex;
      ldb : int;
      C : access cuComplex_h.cuComplex;
      ldc : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:2230
   pragma Import (C, cublasCtrmm_v2, "cublasCtrmm_v2");

  -- host or device pointer  
   function cublasZtrmm_v2
     (handle : cublasHandle_t;
      side : cublasSideMode_t;
      uplo : cublasFillMode_t;
      trans : cublasOperation_t;
      diag : cublasDiagType_t;
      m : int;
      n : int;
      alpha : access constant cuComplex_h.cuDoubleComplex;
      A : access constant cuComplex_h.cuDoubleComplex;
      lda : int;
      B : access constant cuComplex_h.cuDoubleComplex;
      ldb : int;
      C : access cuComplex_h.cuDoubleComplex;
      ldc : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:2245
   pragma Import (C, cublasZtrmm_v2, "cublasZtrmm_v2");

  -- host or device pointer  
  -- BATCH GEMM  
   function cublasSgemmBatched
     (handle : cublasHandle_t;
      transa : cublasOperation_t;
      transb : cublasOperation_t;
      m : int;
      n : int;
      k : int;
      alpha : access float;
      Aarray : System.Address;
      lda : int;
      Barray : System.Address;
      ldb : int;
      beta : access float;
      Carray : System.Address;
      ldc : int;
      batchCount : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:2259
   pragma Import (C, cublasSgemmBatched, "cublasSgemmBatched");

  -- host or device pointer  
  -- host or device pointer  
   function cublasDgemmBatched
     (handle : cublasHandle_t;
      transa : cublasOperation_t;
      transb : cublasOperation_t;
      m : int;
      n : int;
      k : int;
      alpha : access double;
      Aarray : System.Address;
      lda : int;
      Barray : System.Address;
      ldb : int;
      beta : access double;
      Carray : System.Address;
      ldc : int;
      batchCount : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:2275
   pragma Import (C, cublasDgemmBatched, "cublasDgemmBatched");

  -- host or device pointer  
  -- host or device pointer  
   function cublasCgemmBatched
     (handle : cublasHandle_t;
      transa : cublasOperation_t;
      transb : cublasOperation_t;
      m : int;
      n : int;
      k : int;
      alpha : access constant cuComplex_h.cuComplex;
      Aarray : System.Address;
      lda : int;
      Barray : System.Address;
      ldb : int;
      beta : access constant cuComplex_h.cuComplex;
      Carray : System.Address;
      ldc : int;
      batchCount : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:2291
   pragma Import (C, cublasCgemmBatched, "cublasCgemmBatched");

  -- host or device pointer  
  -- host or device pointer  
   function cublasCgemm3mBatched
     (handle : cublasHandle_t;
      transa : cublasOperation_t;
      transb : cublasOperation_t;
      m : int;
      n : int;
      k : int;
      alpha : access constant cuComplex_h.cuComplex;
      Aarray : System.Address;
      lda : int;
      Barray : System.Address;
      ldb : int;
      beta : access constant cuComplex_h.cuComplex;
      Carray : System.Address;
      ldc : int;
      batchCount : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:2307
   pragma Import (C, cublasCgemm3mBatched, "cublasCgemm3mBatched");

  -- host or device pointer  
  -- host or device pointer  
   function cublasZgemmBatched
     (handle : cublasHandle_t;
      transa : cublasOperation_t;
      transb : cublasOperation_t;
      m : int;
      n : int;
      k : int;
      alpha : access constant cuComplex_h.cuDoubleComplex;
      Aarray : System.Address;
      lda : int;
      Barray : System.Address;
      ldb : int;
      beta : access constant cuComplex_h.cuDoubleComplex;
      Carray : System.Address;
      ldc : int;
      batchCount : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:2323
   pragma Import (C, cublasZgemmBatched, "cublasZgemmBatched");

  -- host or device pointer  
  -- host or device pointer  
   function cublasSgemmStridedBatched
     (handle : cublasHandle_t;
      transa : cublasOperation_t;
      transb : cublasOperation_t;
      m : int;
      n : int;
      k : int;
      alpha : access float;
      A : access float;
      lda : int;
      strideA : Long_Long_Integer;
      B : access float;
      ldb : int;
      strideB : Long_Long_Integer;
      beta : access float;
      C : access float;
      ldc : int;
      strideC : Long_Long_Integer;
      batchCount : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:2339
   pragma Import (C, cublasSgemmStridedBatched, "cublasSgemmStridedBatched");

  -- host or device pointer  
  -- purposely signed  
  -- host or device pointer  
   function cublasDgemmStridedBatched
     (handle : cublasHandle_t;
      transa : cublasOperation_t;
      transb : cublasOperation_t;
      m : int;
      n : int;
      k : int;
      alpha : access double;
      A : access double;
      lda : int;
      strideA : Long_Long_Integer;
      B : access double;
      ldb : int;
      strideB : Long_Long_Integer;
      beta : access double;
      C : access double;
      ldc : int;
      strideC : Long_Long_Integer;
      batchCount : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:2358
   pragma Import (C, cublasDgemmStridedBatched, "cublasDgemmStridedBatched");

  -- host or device pointer  
  -- purposely signed  
  -- host or device pointer  
   function cublasCgemmStridedBatched
     (handle : cublasHandle_t;
      transa : cublasOperation_t;
      transb : cublasOperation_t;
      m : int;
      n : int;
      k : int;
      alpha : access constant cuComplex_h.cuComplex;
      A : access constant cuComplex_h.cuComplex;
      lda : int;
      strideA : Long_Long_Integer;
      B : access constant cuComplex_h.cuComplex;
      ldb : int;
      strideB : Long_Long_Integer;
      beta : access constant cuComplex_h.cuComplex;
      C : access cuComplex_h.cuComplex;
      ldc : int;
      strideC : Long_Long_Integer;
      batchCount : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:2377
   pragma Import (C, cublasCgemmStridedBatched, "cublasCgemmStridedBatched");

  -- host or device pointer  
  -- purposely signed  
  -- host or device pointer  
   function cublasCgemm3mStridedBatched
     (handle : cublasHandle_t;
      transa : cublasOperation_t;
      transb : cublasOperation_t;
      m : int;
      n : int;
      k : int;
      alpha : access constant cuComplex_h.cuComplex;
      A : access constant cuComplex_h.cuComplex;
      lda : int;
      strideA : Long_Long_Integer;
      B : access constant cuComplex_h.cuComplex;
      ldb : int;
      strideB : Long_Long_Integer;
      beta : access constant cuComplex_h.cuComplex;
      C : access cuComplex_h.cuComplex;
      ldc : int;
      strideC : Long_Long_Integer;
      batchCount : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:2396
   pragma Import (C, cublasCgemm3mStridedBatched, "cublasCgemm3mStridedBatched");

  -- host or device pointer  
  -- purposely signed  
  -- host or device pointer  
   function cublasZgemmStridedBatched
     (handle : cublasHandle_t;
      transa : cublasOperation_t;
      transb : cublasOperation_t;
      m : int;
      n : int;
      k : int;
      alpha : access constant cuComplex_h.cuDoubleComplex;
      A : access constant cuComplex_h.cuDoubleComplex;
      lda : int;
      strideA : Long_Long_Integer;
      B : access constant cuComplex_h.cuDoubleComplex;
      ldb : int;
      strideB : Long_Long_Integer;
      beta : access constant cuComplex_h.cuDoubleComplex;
      C : access cuComplex_h.cuDoubleComplex;
      ldc : int;
      strideC : Long_Long_Integer;
      batchCount : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:2416
   pragma Import (C, cublasZgemmStridedBatched, "cublasZgemmStridedBatched");

  -- host or device pointer  
  -- purposely signed  
  -- host or device poi  
   function cublasHgemmStridedBatched
     (handle : cublasHandle_t;
      transa : cublasOperation_t;
      transb : cublasOperation_t;
      m : int;
      n : int;
      k : int;
      alpha : access constant cuda_fp16_h.uu_half;
      A : access constant cuda_fp16_h.uu_half;
      lda : int;
      strideA : Long_Long_Integer;
      B : access constant cuda_fp16_h.uu_half;
      ldb : int;
      strideB : Long_Long_Integer;
      beta : access constant cuda_fp16_h.uu_half;
      C : access cuda_fp16_h.uu_half;
      ldc : int;
      strideC : Long_Long_Integer;
      batchCount : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:2435
   pragma Import (C, cublasHgemmStridedBatched, "cublasHgemmStridedBatched");

  -- host or device pointer  
  -- purposely signed  
  -- host or device pointer  
  -- ---------------- CUBLAS BLAS-like extension ----------------  
  -- GEAM  
   function cublasSgeam
     (handle : cublasHandle_t;
      transa : cublasOperation_t;
      transb : cublasOperation_t;
      m : int;
      n : int;
      alpha : access float;
      A : access float;
      lda : int;
      beta : access float;
      B : access float;
      ldb : int;
      C : access float;
      ldc : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:2456
   pragma Import (C, cublasSgeam, "cublasSgeam");

  -- host or device pointer  
  -- host or device pointer  
   function cublasDgeam
     (handle : cublasHandle_t;
      transa : cublasOperation_t;
      transb : cublasOperation_t;
      m : int;
      n : int;
      alpha : access double;
      A : access double;
      lda : int;
      beta : access double;
      B : access double;
      ldb : int;
      C : access double;
      ldc : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:2470
   pragma Import (C, cublasDgeam, "cublasDgeam");

  -- host or device pointer  
  -- host or device pointer  
   function cublasCgeam
     (handle : cublasHandle_t;
      transa : cublasOperation_t;
      transb : cublasOperation_t;
      m : int;
      n : int;
      alpha : access constant cuComplex_h.cuComplex;
      A : access constant cuComplex_h.cuComplex;
      lda : int;
      beta : access constant cuComplex_h.cuComplex;
      B : access constant cuComplex_h.cuComplex;
      ldb : int;
      C : access cuComplex_h.cuComplex;
      ldc : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:2484
   pragma Import (C, cublasCgeam, "cublasCgeam");

  -- host or device pointer  
  -- host or device pointer  
   function cublasZgeam
     (handle : cublasHandle_t;
      transa : cublasOperation_t;
      transb : cublasOperation_t;
      m : int;
      n : int;
      alpha : access constant cuComplex_h.cuDoubleComplex;
      A : access constant cuComplex_h.cuDoubleComplex;
      lda : int;
      beta : access constant cuComplex_h.cuDoubleComplex;
      B : access constant cuComplex_h.cuDoubleComplex;
      ldb : int;
      C : access cuComplex_h.cuDoubleComplex;
      ldc : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:2498
   pragma Import (C, cublasZgeam, "cublasZgeam");

  -- host or device pointer  
  -- host or device pointer  
  -- Batched LU - GETRF 
   function cublasSgetrfBatched
     (handle : cublasHandle_t;
      n : int;
      A : System.Address;
      lda : int;
      P : access int;
      info : access int;
      batchSize : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:2513
   pragma Import (C, cublasSgetrfBatched, "cublasSgetrfBatched");

  --Device pointer 
  --Device Pointer 
  --Device Pointer 
   function cublasDgetrfBatched
     (handle : cublasHandle_t;
      n : int;
      A : System.Address;
      lda : int;
      P : access int;
      info : access int;
      batchSize : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:2521
   pragma Import (C, cublasDgetrfBatched, "cublasDgetrfBatched");

  --Device pointer 
  --Device Pointer 
  --Device Pointer 
   function cublasCgetrfBatched
     (handle : cublasHandle_t;
      n : int;
      A : System.Address;
      lda : int;
      P : access int;
      info : access int;
      batchSize : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:2529
   pragma Import (C, cublasCgetrfBatched, "cublasCgetrfBatched");

  --Device pointer 
  --Device Pointer 
  --Device Pointer 
   function cublasZgetrfBatched
     (handle : cublasHandle_t;
      n : int;
      A : System.Address;
      lda : int;
      P : access int;
      info : access int;
      batchSize : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:2537
   pragma Import (C, cublasZgetrfBatched, "cublasZgetrfBatched");

  --Device pointer 
  --Device Pointer 
  --Device Pointer 
  -- Batched inversion based on LU factorization from getrf  
   function cublasSgetriBatched
     (handle : cublasHandle_t;
      n : int;
      A : System.Address;
      lda : int;
      P : access int;
      C : System.Address;
      ldc : int;
      info : access int;
      batchSize : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:2546
   pragma Import (C, cublasSgetriBatched, "cublasSgetriBatched");

  --Device pointer 
  --Device pointer 
  --Device pointer 
   function cublasDgetriBatched
     (handle : cublasHandle_t;
      n : int;
      A : System.Address;
      lda : int;
      P : access int;
      C : System.Address;
      ldc : int;
      info : access int;
      batchSize : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:2556
   pragma Import (C, cublasDgetriBatched, "cublasDgetriBatched");

  --Device pointer 
  --Device pointer 
  --Device pointer 
   function cublasCgetriBatched
     (handle : cublasHandle_t;
      n : int;
      A : System.Address;
      lda : int;
      P : access int;
      C : System.Address;
      ldc : int;
      info : access int;
      batchSize : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:2566
   pragma Import (C, cublasCgetriBatched, "cublasCgetriBatched");

  --Device pointer 
  --Device pointer 
  --Device pointer 
   function cublasZgetriBatched
     (handle : cublasHandle_t;
      n : int;
      A : System.Address;
      lda : int;
      P : access int;
      C : System.Address;
      ldc : int;
      info : access int;
      batchSize : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:2576
   pragma Import (C, cublasZgetriBatched, "cublasZgetriBatched");

  --Device pointer 
  --Device pointer 
  --Device pointer 
  -- Batched solver based on LU factorization from getrf  
   function cublasSgetrsBatched
     (handle : cublasHandle_t;
      trans : cublasOperation_t;
      n : int;
      nrhs : int;
      Aarray : System.Address;
      lda : int;
      devIpiv : access int;
      Barray : System.Address;
      ldb : int;
      info : access int;
      batchSize : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:2588
   pragma Import (C, cublasSgetrsBatched, "cublasSgetrsBatched");

   function cublasDgetrsBatched
     (handle : cublasHandle_t;
      trans : cublasOperation_t;
      n : int;
      nrhs : int;
      Aarray : System.Address;
      lda : int;
      devIpiv : access int;
      Barray : System.Address;
      ldb : int;
      info : access int;
      batchSize : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:2600
   pragma Import (C, cublasDgetrsBatched, "cublasDgetrsBatched");

   function cublasCgetrsBatched
     (handle : cublasHandle_t;
      trans : cublasOperation_t;
      n : int;
      nrhs : int;
      Aarray : System.Address;
      lda : int;
      devIpiv : access int;
      Barray : System.Address;
      ldb : int;
      info : access int;
      batchSize : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:2612
   pragma Import (C, cublasCgetrsBatched, "cublasCgetrsBatched");

   function cublasZgetrsBatched
     (handle : cublasHandle_t;
      trans : cublasOperation_t;
      n : int;
      nrhs : int;
      Aarray : System.Address;
      lda : int;
      devIpiv : access int;
      Barray : System.Address;
      ldb : int;
      info : access int;
      batchSize : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:2625
   pragma Import (C, cublasZgetrsBatched, "cublasZgetrsBatched");

  -- TRSM - Batched Triangular Solver  
   function cublasStrsmBatched
     (handle : cublasHandle_t;
      side : cublasSideMode_t;
      uplo : cublasFillMode_t;
      trans : cublasOperation_t;
      diag : cublasDiagType_t;
      m : int;
      n : int;
      alpha : access float;
      A : System.Address;
      lda : int;
      B : System.Address;
      ldb : int;
      batchCount : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:2640
   pragma Import (C, cublasStrsmBatched, "cublasStrsmBatched");

  --Host or Device Pointer 
   function cublasDtrsmBatched
     (handle : cublasHandle_t;
      side : cublasSideMode_t;
      uplo : cublasFillMode_t;
      trans : cublasOperation_t;
      diag : cublasDiagType_t;
      m : int;
      n : int;
      alpha : access double;
      A : System.Address;
      lda : int;
      B : System.Address;
      ldb : int;
      batchCount : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:2654
   pragma Import (C, cublasDtrsmBatched, "cublasDtrsmBatched");

  --Host or Device Pointer 
   function cublasCtrsmBatched
     (handle : cublasHandle_t;
      side : cublasSideMode_t;
      uplo : cublasFillMode_t;
      trans : cublasOperation_t;
      diag : cublasDiagType_t;
      m : int;
      n : int;
      alpha : access constant cuComplex_h.cuComplex;
      A : System.Address;
      lda : int;
      B : System.Address;
      ldb : int;
      batchCount : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:2668
   pragma Import (C, cublasCtrsmBatched, "cublasCtrsmBatched");

  --Host or Device Pointer 
   function cublasZtrsmBatched
     (handle : cublasHandle_t;
      side : cublasSideMode_t;
      uplo : cublasFillMode_t;
      trans : cublasOperation_t;
      diag : cublasDiagType_t;
      m : int;
      n : int;
      alpha : access constant cuComplex_h.cuDoubleComplex;
      A : System.Address;
      lda : int;
      B : System.Address;
      ldb : int;
      batchCount : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:2682
   pragma Import (C, cublasZtrsmBatched, "cublasZtrsmBatched");

  --Host or Device Pointer 
  -- Batched - MATINV 
   function cublasSmatinvBatched
     (handle : cublasHandle_t;
      n : int;
      A : System.Address;
      lda : int;
      Ainv : System.Address;
      lda_inv : int;
      info : access int;
      batchSize : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:2697
   pragma Import (C, cublasSmatinvBatched, "cublasSmatinvBatched");

  --Device pointer 
  --Device pointer 
  --Device Pointer 
   function cublasDmatinvBatched
     (handle : cublasHandle_t;
      n : int;
      A : System.Address;
      lda : int;
      Ainv : System.Address;
      lda_inv : int;
      info : access int;
      batchSize : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:2706
   pragma Import (C, cublasDmatinvBatched, "cublasDmatinvBatched");

  --Device pointer 
  --Device pointer 
  --Device Pointer 
   function cublasCmatinvBatched
     (handle : cublasHandle_t;
      n : int;
      A : System.Address;
      lda : int;
      Ainv : System.Address;
      lda_inv : int;
      info : access int;
      batchSize : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:2715
   pragma Import (C, cublasCmatinvBatched, "cublasCmatinvBatched");

  --Device pointer 
  --Device pointer 
  --Device Pointer 
   function cublasZmatinvBatched
     (handle : cublasHandle_t;
      n : int;
      A : System.Address;
      lda : int;
      Ainv : System.Address;
      lda_inv : int;
      info : access int;
      batchSize : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:2724
   pragma Import (C, cublasZmatinvBatched, "cublasZmatinvBatched");

  --Device pointer 
  --Device pointer 
  --Device Pointer 
  -- Batch QR Factorization  
   function cublasSgeqrfBatched
     (handle : cublasHandle_t;
      m : int;
      n : int;
      Aarray : System.Address;
      lda : int;
      TauArray : System.Address;
      info : access int;
      batchSize : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:2734
   pragma Import (C, cublasSgeqrfBatched, "cublasSgeqrfBatched");

  --Device pointer 
  -- Device pointer 
   function cublasDgeqrfBatched
     (handle : cublasHandle_t;
      m : int;
      n : int;
      Aarray : System.Address;
      lda : int;
      TauArray : System.Address;
      info : access int;
      batchSize : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:2743
   pragma Import (C, cublasDgeqrfBatched, "cublasDgeqrfBatched");

  --Device pointer 
  -- Device pointer 
   function cublasCgeqrfBatched
     (handle : cublasHandle_t;
      m : int;
      n : int;
      Aarray : System.Address;
      lda : int;
      TauArray : System.Address;
      info : access int;
      batchSize : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:2752
   pragma Import (C, cublasCgeqrfBatched, "cublasCgeqrfBatched");

  --Device pointer 
  -- Device pointer 
   function cublasZgeqrfBatched
     (handle : cublasHandle_t;
      m : int;
      n : int;
      Aarray : System.Address;
      lda : int;
      TauArray : System.Address;
      info : access int;
      batchSize : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:2761
   pragma Import (C, cublasZgeqrfBatched, "cublasZgeqrfBatched");

  --Device pointer 
  -- Device pointer 
  -- Least Square Min only m >= n and Non-transpose supported  
   function cublasSgelsBatched
     (handle : cublasHandle_t;
      trans : cublasOperation_t;
      m : int;
      n : int;
      nrhs : int;
      Aarray : System.Address;
      lda : int;
      Carray : System.Address;
      ldc : int;
      info : access int;
      devInfoArray : access int;
      batchSize : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:2770
   pragma Import (C, cublasSgelsBatched, "cublasSgelsBatched");

  --Device pointer 
  -- Device pointer 
  -- Device pointer 
   function cublasDgelsBatched
     (handle : cublasHandle_t;
      trans : cublasOperation_t;
      m : int;
      n : int;
      nrhs : int;
      Aarray : System.Address;
      lda : int;
      Carray : System.Address;
      ldc : int;
      info : access int;
      devInfoArray : access int;
      batchSize : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:2783
   pragma Import (C, cublasDgelsBatched, "cublasDgelsBatched");

  --Device pointer 
  -- Device pointer 
  -- Device pointer 
   function cublasCgelsBatched
     (handle : cublasHandle_t;
      trans : cublasOperation_t;
      m : int;
      n : int;
      nrhs : int;
      Aarray : System.Address;
      lda : int;
      Carray : System.Address;
      ldc : int;
      info : access int;
      devInfoArray : access int;
      batchSize : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:2796
   pragma Import (C, cublasCgelsBatched, "cublasCgelsBatched");

  --Device pointer 
  -- Device pointer 
   function cublasZgelsBatched
     (handle : cublasHandle_t;
      trans : cublasOperation_t;
      m : int;
      n : int;
      nrhs : int;
      Aarray : System.Address;
      lda : int;
      Carray : System.Address;
      ldc : int;
      info : access int;
      devInfoArray : access int;
      batchSize : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:2809
   pragma Import (C, cublasZgelsBatched, "cublasZgelsBatched");

  --Device pointer 
  -- Device pointer 
  -- DGMM  
   function cublasSdgmm
     (handle : cublasHandle_t;
      mode : cublasSideMode_t;
      m : int;
      n : int;
      A : access float;
      lda : int;
      x : access float;
      incx : int;
      C : access float;
      ldc : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:2822
   pragma Import (C, cublasSdgmm, "cublasSdgmm");

   function cublasDdgmm
     (handle : cublasHandle_t;
      mode : cublasSideMode_t;
      m : int;
      n : int;
      A : access double;
      lda : int;
      x : access double;
      incx : int;
      C : access double;
      ldc : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:2833
   pragma Import (C, cublasDdgmm, "cublasDdgmm");

   function cublasCdgmm
     (handle : cublasHandle_t;
      mode : cublasSideMode_t;
      m : int;
      n : int;
      A : access constant cuComplex_h.cuComplex;
      lda : int;
      x : access constant cuComplex_h.cuComplex;
      incx : int;
      C : access cuComplex_h.cuComplex;
      ldc : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:2844
   pragma Import (C, cublasCdgmm, "cublasCdgmm");

   function cublasZdgmm
     (handle : cublasHandle_t;
      mode : cublasSideMode_t;
      m : int;
      n : int;
      A : access constant cuComplex_h.cuDoubleComplex;
      lda : int;
      x : access constant cuComplex_h.cuDoubleComplex;
      incx : int;
      C : access cuComplex_h.cuDoubleComplex;
      ldc : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:2855
   pragma Import (C, cublasZdgmm, "cublasZdgmm");

  -- TPTTR : Triangular Pack format to Triangular format  
   function cublasStpttr
     (handle : cublasHandle_t;
      uplo : cublasFillMode_t;
      n : int;
      AP : access float;
      A : access float;
      lda : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:2867
   pragma Import (C, cublasStpttr, "cublasStpttr");

   function cublasDtpttr
     (handle : cublasHandle_t;
      uplo : cublasFillMode_t;
      n : int;
      AP : access double;
      A : access double;
      lda : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:2874
   pragma Import (C, cublasDtpttr, "cublasDtpttr");

   function cublasCtpttr
     (handle : cublasHandle_t;
      uplo : cublasFillMode_t;
      n : int;
      AP : access constant cuComplex_h.cuComplex;
      A : access cuComplex_h.cuComplex;
      lda : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:2881
   pragma Import (C, cublasCtpttr, "cublasCtpttr");

   function cublasZtpttr
     (handle : cublasHandle_t;
      uplo : cublasFillMode_t;
      n : int;
      AP : access constant cuComplex_h.cuDoubleComplex;
      A : access cuComplex_h.cuDoubleComplex;
      lda : int) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:2888
   pragma Import (C, cublasZtpttr, "cublasZtpttr");

  -- TRTTP : Triangular format to Triangular Pack format  
   function cublasStrttp
     (handle : cublasHandle_t;
      uplo : cublasFillMode_t;
      n : int;
      A : access float;
      lda : int;
      AP : access float) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:2895
   pragma Import (C, cublasStrttp, "cublasStrttp");

   function cublasDtrttp
     (handle : cublasHandle_t;
      uplo : cublasFillMode_t;
      n : int;
      A : access double;
      lda : int;
      AP : access double) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:2902
   pragma Import (C, cublasDtrttp, "cublasDtrttp");

   function cublasCtrttp
     (handle : cublasHandle_t;
      uplo : cublasFillMode_t;
      n : int;
      A : access constant cuComplex_h.cuComplex;
      lda : int;
      AP : access cuComplex_h.cuComplex) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:2909
   pragma Import (C, cublasCtrttp, "cublasCtrttp");

   function cublasZtrttp
     (handle : cublasHandle_t;
      uplo : cublasFillMode_t;
      n : int;
      A : access constant cuComplex_h.cuDoubleComplex;
      lda : int;
      AP : access cuComplex_h.cuDoubleComplex) return cublasStatus_t;  -- /usr/local/cuda-8.0/include/cublas_api.h:2916
   pragma Import (C, cublasZtrttp, "cublasZtrttp");

end cublas_api_h;

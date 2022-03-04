pragma Ada_2005;
pragma Style_Checks (Off);

with Interfaces.C; use Interfaces.C;
with System;
with cusolver_common_h;
with driver_types_h;
with cusparse_h;
with cuComplex_h;
with stddef_h;

package cusolverSp_h is

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

   --  skipped empty struct cusolverSpContext

   type cusolverSpHandle_t is new System.Address;  -- /usr/local/cuda-8.0/include/cusolverSp.h:61

   --  skipped empty struct csrqrInfo

   type csrqrInfo_t is new System.Address;  -- /usr/local/cuda-8.0/include/cusolverSp.h:64

   function cusolverSpCreate (handle : System.Address) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp.h:66
   pragma Import (C, cusolverSpCreate, "cusolverSpCreate");

   function cusolverSpDestroy (handle : cusolverSpHandle_t) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp.h:67
   pragma Import (C, cusolverSpDestroy, "cusolverSpDestroy");

   function cusolverSpSetStream (handle : cusolverSpHandle_t; streamId : driver_types_h.cudaStream_t) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp.h:68
   pragma Import (C, cusolverSpSetStream, "cusolverSpSetStream");

   function cusolverSpGetStream (handle : cusolverSpHandle_t; streamId : System.Address) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp.h:69
   pragma Import (C, cusolverSpGetStream, "cusolverSpGetStream");

   function cusolverSpXcsrissymHost
     (handle : cusolverSpHandle_t;
      m : int;
      nnzA : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrRowPtrA : access int;
      csrEndPtrA : access int;
      csrColIndA : access int;
      issym : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp.h:71
   pragma Import (C, cusolverSpXcsrissymHost, "cusolverSpXcsrissymHost");

  -- -------- GPU linear solver based on LU factorization
  -- *       solve A*x = b, A can be singular 
  -- * [ls] stands for linear solve
  -- * [v] stands for vector
  -- * [lu] stands for LU factorization
  --  

   function cusolverSpScsrlsvluHost
     (handle : cusolverSpHandle_t;
      n : int;
      nnzA : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrValA : access float;
      csrRowPtrA : access int;
      csrColIndA : access int;
      b : access float;
      tol : float;
      reorder : int;
      x : access float;
      singularity : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp.h:87
   pragma Import (C, cusolverSpScsrlsvluHost, "cusolverSpScsrlsvluHost");

   function cusolverSpDcsrlsvluHost
     (handle : cusolverSpHandle_t;
      n : int;
      nnzA : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrValA : access double;
      csrRowPtrA : access int;
      csrColIndA : access int;
      b : access double;
      tol : double;
      reorder : int;
      x : access double;
      singularity : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp.h:101
   pragma Import (C, cusolverSpDcsrlsvluHost, "cusolverSpDcsrlsvluHost");

   function cusolverSpCcsrlsvluHost
     (handle : cusolverSpHandle_t;
      n : int;
      nnzA : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrValA : access constant cuComplex_h.cuComplex;
      csrRowPtrA : access int;
      csrColIndA : access int;
      b : access constant cuComplex_h.cuComplex;
      tol : float;
      reorder : int;
      x : access cuComplex_h.cuComplex;
      singularity : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp.h:115
   pragma Import (C, cusolverSpCcsrlsvluHost, "cusolverSpCcsrlsvluHost");

   function cusolverSpZcsrlsvluHost
     (handle : cusolverSpHandle_t;
      n : int;
      nnzA : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrValA : access constant cuComplex_h.cuDoubleComplex;
      csrRowPtrA : access int;
      csrColIndA : access int;
      b : access constant cuComplex_h.cuDoubleComplex;
      tol : double;
      reorder : int;
      x : access cuComplex_h.cuDoubleComplex;
      singularity : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp.h:129
   pragma Import (C, cusolverSpZcsrlsvluHost, "cusolverSpZcsrlsvluHost");

  -- -------- GPU linear solver based on QR factorization
  -- *       solve A*x = b, A can be singular 
  -- * [ls] stands for linear solve
  -- * [v] stands for vector
  -- * [qr] stands for QR factorization
  --  

   function cusolverSpScsrlsvqr
     (handle : cusolverSpHandle_t;
      m : int;
      nnz : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrVal : access float;
      csrRowPtr : access int;
      csrColInd : access int;
      b : access float;
      tol : float;
      reorder : int;
      x : access float;
      singularity : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp.h:150
   pragma Import (C, cusolverSpScsrlsvqr, "cusolverSpScsrlsvqr");

   function cusolverSpDcsrlsvqr
     (handle : cusolverSpHandle_t;
      m : int;
      nnz : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrVal : access double;
      csrRowPtr : access int;
      csrColInd : access int;
      b : access double;
      tol : double;
      reorder : int;
      x : access double;
      singularity : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp.h:164
   pragma Import (C, cusolverSpDcsrlsvqr, "cusolverSpDcsrlsvqr");

   function cusolverSpCcsrlsvqr
     (handle : cusolverSpHandle_t;
      m : int;
      nnz : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrVal : access constant cuComplex_h.cuComplex;
      csrRowPtr : access int;
      csrColInd : access int;
      b : access constant cuComplex_h.cuComplex;
      tol : float;
      reorder : int;
      x : access cuComplex_h.cuComplex;
      singularity : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp.h:178
   pragma Import (C, cusolverSpCcsrlsvqr, "cusolverSpCcsrlsvqr");

   function cusolverSpZcsrlsvqr
     (handle : cusolverSpHandle_t;
      m : int;
      nnz : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrVal : access constant cuComplex_h.cuDoubleComplex;
      csrRowPtr : access int;
      csrColInd : access int;
      b : access constant cuComplex_h.cuDoubleComplex;
      tol : double;
      reorder : int;
      x : access cuComplex_h.cuDoubleComplex;
      singularity : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp.h:192
   pragma Import (C, cusolverSpZcsrlsvqr, "cusolverSpZcsrlsvqr");

  -- -------- CPU linear solver based on QR factorization
  -- *       solve A*x = b, A can be singular 
  -- * [ls] stands for linear solve
  -- * [v] stands for vector
  -- * [qr] stands for QR factorization
  --  

   function cusolverSpScsrlsvqrHost
     (handle : cusolverSpHandle_t;
      m : int;
      nnz : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrValA : access float;
      csrRowPtrA : access int;
      csrColIndA : access int;
      b : access float;
      tol : float;
      reorder : int;
      x : access float;
      singularity : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp.h:214
   pragma Import (C, cusolverSpScsrlsvqrHost, "cusolverSpScsrlsvqrHost");

   function cusolverSpDcsrlsvqrHost
     (handle : cusolverSpHandle_t;
      m : int;
      nnz : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrValA : access double;
      csrRowPtrA : access int;
      csrColIndA : access int;
      b : access double;
      tol : double;
      reorder : int;
      x : access double;
      singularity : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp.h:228
   pragma Import (C, cusolverSpDcsrlsvqrHost, "cusolverSpDcsrlsvqrHost");

   function cusolverSpCcsrlsvqrHost
     (handle : cusolverSpHandle_t;
      m : int;
      nnz : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrValA : access constant cuComplex_h.cuComplex;
      csrRowPtrA : access int;
      csrColIndA : access int;
      b : access constant cuComplex_h.cuComplex;
      tol : float;
      reorder : int;
      x : access cuComplex_h.cuComplex;
      singularity : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp.h:242
   pragma Import (C, cusolverSpCcsrlsvqrHost, "cusolverSpCcsrlsvqrHost");

   function cusolverSpZcsrlsvqrHost
     (handle : cusolverSpHandle_t;
      m : int;
      nnz : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrValA : access constant cuComplex_h.cuDoubleComplex;
      csrRowPtrA : access int;
      csrColIndA : access int;
      b : access constant cuComplex_h.cuDoubleComplex;
      tol : double;
      reorder : int;
      x : access cuComplex_h.cuDoubleComplex;
      singularity : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp.h:256
   pragma Import (C, cusolverSpZcsrlsvqrHost, "cusolverSpZcsrlsvqrHost");

  -- -------- CPU linear solver based on Cholesky factorization
  -- *       solve A*x = b, A can be singular 
  -- * [ls] stands for linear solve
  -- * [v] stands for vector
  -- * [chol] stands for Cholesky factorization
  -- *
  -- * Only works for symmetric positive definite matrix.
  -- * The upper part of A is ignored.
  --  

   function cusolverSpScsrlsvcholHost
     (handle : cusolverSpHandle_t;
      m : int;
      nnz : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrVal : access float;
      csrRowPtr : access int;
      csrColInd : access int;
      b : access float;
      tol : float;
      reorder : int;
      x : access float;
      singularity : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp.h:280
   pragma Import (C, cusolverSpScsrlsvcholHost, "cusolverSpScsrlsvcholHost");

   function cusolverSpDcsrlsvcholHost
     (handle : cusolverSpHandle_t;
      m : int;
      nnz : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrVal : access double;
      csrRowPtr : access int;
      csrColInd : access int;
      b : access double;
      tol : double;
      reorder : int;
      x : access double;
      singularity : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp.h:294
   pragma Import (C, cusolverSpDcsrlsvcholHost, "cusolverSpDcsrlsvcholHost");

   function cusolverSpCcsrlsvcholHost
     (handle : cusolverSpHandle_t;
      m : int;
      nnz : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrVal : access constant cuComplex_h.cuComplex;
      csrRowPtr : access int;
      csrColInd : access int;
      b : access constant cuComplex_h.cuComplex;
      tol : float;
      reorder : int;
      x : access cuComplex_h.cuComplex;
      singularity : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp.h:308
   pragma Import (C, cusolverSpCcsrlsvcholHost, "cusolverSpCcsrlsvcholHost");

   function cusolverSpZcsrlsvcholHost
     (handle : cusolverSpHandle_t;
      m : int;
      nnz : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrVal : access constant cuComplex_h.cuDoubleComplex;
      csrRowPtr : access int;
      csrColInd : access int;
      b : access constant cuComplex_h.cuDoubleComplex;
      tol : double;
      reorder : int;
      x : access cuComplex_h.cuDoubleComplex;
      singularity : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp.h:322
   pragma Import (C, cusolverSpZcsrlsvcholHost, "cusolverSpZcsrlsvcholHost");

  -- -------- GPU linear solver based on Cholesky factorization
  -- *       solve A*x = b, A can be singular 
  -- * [ls] stands for linear solve
  -- * [v] stands for vector
  -- * [chol] stands for Cholesky factorization
  -- *
  -- * Only works for symmetric positive definite matrix.
  -- * The upper part of A is ignored.
  --  

   function cusolverSpScsrlsvchol
     (handle : cusolverSpHandle_t;
      m : int;
      nnz : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrVal : access float;
      csrRowPtr : access int;
      csrColInd : access int;
      b : access float;
      tol : float;
      reorder : int;
      x : access float;
      singularity : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp.h:345
   pragma Import (C, cusolverSpScsrlsvchol, "cusolverSpScsrlsvchol");

  -- output
   function cusolverSpDcsrlsvchol
     (handle : cusolverSpHandle_t;
      m : int;
      nnz : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrVal : access double;
      csrRowPtr : access int;
      csrColInd : access int;
      b : access double;
      tol : double;
      reorder : int;
      x : access double;
      singularity : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp.h:360
   pragma Import (C, cusolverSpDcsrlsvchol, "cusolverSpDcsrlsvchol");

  -- output
   function cusolverSpCcsrlsvchol
     (handle : cusolverSpHandle_t;
      m : int;
      nnz : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrVal : access constant cuComplex_h.cuComplex;
      csrRowPtr : access int;
      csrColInd : access int;
      b : access constant cuComplex_h.cuComplex;
      tol : float;
      reorder : int;
      x : access cuComplex_h.cuComplex;
      singularity : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp.h:375
   pragma Import (C, cusolverSpCcsrlsvchol, "cusolverSpCcsrlsvchol");

  -- output
   function cusolverSpZcsrlsvchol
     (handle : cusolverSpHandle_t;
      m : int;
      nnz : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrVal : access constant cuComplex_h.cuDoubleComplex;
      csrRowPtr : access int;
      csrColInd : access int;
      b : access constant cuComplex_h.cuDoubleComplex;
      tol : double;
      reorder : int;
      x : access cuComplex_h.cuDoubleComplex;
      singularity : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp.h:390
   pragma Import (C, cusolverSpZcsrlsvchol, "cusolverSpZcsrlsvchol");

  -- output
  -- ----------- CPU least square solver based on QR factorization
  -- *       solve min|b - A*x| 
  -- * [lsq] stands for least square
  -- * [v] stands for vector
  -- * [qr] stands for QR factorization
  --  

   function cusolverSpScsrlsqvqrHost
     (handle : cusolverSpHandle_t;
      m : int;
      n : int;
      nnz : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrValA : access float;
      csrRowPtrA : access int;
      csrColIndA : access int;
      b : access float;
      tol : float;
      rankA : access int;
      x : access float;
      p : access int;
      min_norm : access float) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp.h:413
   pragma Import (C, cusolverSpScsrlsqvqrHost, "cusolverSpScsrlsqvqrHost");

   function cusolverSpDcsrlsqvqrHost
     (handle : cusolverSpHandle_t;
      m : int;
      n : int;
      nnz : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrValA : access double;
      csrRowPtrA : access int;
      csrColIndA : access int;
      b : access double;
      tol : double;
      rankA : access int;
      x : access double;
      p : access int;
      min_norm : access double) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp.h:429
   pragma Import (C, cusolverSpDcsrlsqvqrHost, "cusolverSpDcsrlsqvqrHost");

   function cusolverSpCcsrlsqvqrHost
     (handle : cusolverSpHandle_t;
      m : int;
      n : int;
      nnz : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrValA : access constant cuComplex_h.cuComplex;
      csrRowPtrA : access int;
      csrColIndA : access int;
      b : access constant cuComplex_h.cuComplex;
      tol : float;
      rankA : access int;
      x : access cuComplex_h.cuComplex;
      p : access int;
      min_norm : access float) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp.h:445
   pragma Import (C, cusolverSpCcsrlsqvqrHost, "cusolverSpCcsrlsqvqrHost");

   function cusolverSpZcsrlsqvqrHost
     (handle : cusolverSpHandle_t;
      m : int;
      n : int;
      nnz : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrValA : access constant cuComplex_h.cuDoubleComplex;
      csrRowPtrA : access int;
      csrColIndA : access int;
      b : access constant cuComplex_h.cuDoubleComplex;
      tol : double;
      rankA : access int;
      x : access cuComplex_h.cuDoubleComplex;
      p : access int;
      min_norm : access double) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp.h:461
   pragma Import (C, cusolverSpZcsrlsqvqrHost, "cusolverSpZcsrlsqvqrHost");

  -- --------- CPU eigenvalue solver based on shift inverse
  -- *      solve A*x = lambda * x 
  -- *   where lambda is the eigenvalue nearest mu0.
  -- * [eig] stands for eigenvalue solver
  -- * [si] stands for shift-inverse
  --  

   function cusolverSpScsreigvsiHost
     (handle : cusolverSpHandle_t;
      m : int;
      nnz : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrValA : access float;
      csrRowPtrA : access int;
      csrColIndA : access int;
      mu0 : float;
      x0 : access float;
      maxite : int;
      tol : float;
      mu : access float;
      x : access float) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp.h:483
   pragma Import (C, cusolverSpScsreigvsiHost, "cusolverSpScsreigvsiHost");

   function cusolverSpDcsreigvsiHost
     (handle : cusolverSpHandle_t;
      m : int;
      nnz : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrValA : access double;
      csrRowPtrA : access int;
      csrColIndA : access int;
      mu0 : double;
      x0 : access double;
      maxite : int;
      tol : double;
      mu : access double;
      x : access double) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp.h:498
   pragma Import (C, cusolverSpDcsreigvsiHost, "cusolverSpDcsreigvsiHost");

   function cusolverSpCcsreigvsiHost
     (handle : cusolverSpHandle_t;
      m : int;
      nnz : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrValA : access constant cuComplex_h.cuComplex;
      csrRowPtrA : access int;
      csrColIndA : access int;
      mu0 : cuComplex_h.cuComplex;
      x0 : access constant cuComplex_h.cuComplex;
      maxite : int;
      tol : float;
      mu : access cuComplex_h.cuComplex;
      x : access cuComplex_h.cuComplex) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp.h:513
   pragma Import (C, cusolverSpCcsreigvsiHost, "cusolverSpCcsreigvsiHost");

   function cusolverSpZcsreigvsiHost
     (handle : cusolverSpHandle_t;
      m : int;
      nnz : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrValA : access constant cuComplex_h.cuDoubleComplex;
      csrRowPtrA : access int;
      csrColIndA : access int;
      mu0 : cuComplex_h.cuDoubleComplex;
      x0 : access constant cuComplex_h.cuDoubleComplex;
      maxite : int;
      tol : double;
      mu : access cuComplex_h.cuDoubleComplex;
      x : access cuComplex_h.cuDoubleComplex) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp.h:528
   pragma Import (C, cusolverSpZcsreigvsiHost, "cusolverSpZcsreigvsiHost");

  -- --------- GPU eigenvalue solver based on shift inverse
  -- *      solve A*x = lambda * x 
  -- *   where lambda is the eigenvalue nearest mu0.
  -- * [eig] stands for eigenvalue solver
  -- * [si] stands for shift-inverse
  --  

   function cusolverSpScsreigvsi
     (handle : cusolverSpHandle_t;
      m : int;
      nnz : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrValA : access float;
      csrRowPtrA : access int;
      csrColIndA : access int;
      mu0 : float;
      x0 : access float;
      maxite : int;
      eps : float;
      mu : access float;
      x : access float) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp.h:550
   pragma Import (C, cusolverSpScsreigvsi, "cusolverSpScsreigvsi");

   function cusolverSpDcsreigvsi
     (handle : cusolverSpHandle_t;
      m : int;
      nnz : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrValA : access double;
      csrRowPtrA : access int;
      csrColIndA : access int;
      mu0 : double;
      x0 : access double;
      maxite : int;
      eps : double;
      mu : access double;
      x : access double) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp.h:565
   pragma Import (C, cusolverSpDcsreigvsi, "cusolverSpDcsreigvsi");

   function cusolverSpCcsreigvsi
     (handle : cusolverSpHandle_t;
      m : int;
      nnz : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrValA : access constant cuComplex_h.cuComplex;
      csrRowPtrA : access int;
      csrColIndA : access int;
      mu0 : cuComplex_h.cuComplex;
      x0 : access constant cuComplex_h.cuComplex;
      maxite : int;
      eps : float;
      mu : access cuComplex_h.cuComplex;
      x : access cuComplex_h.cuComplex) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp.h:580
   pragma Import (C, cusolverSpCcsreigvsi, "cusolverSpCcsreigvsi");

   function cusolverSpZcsreigvsi
     (handle : cusolverSpHandle_t;
      m : int;
      nnz : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrValA : access constant cuComplex_h.cuDoubleComplex;
      csrRowPtrA : access int;
      csrColIndA : access int;
      mu0 : cuComplex_h.cuDoubleComplex;
      x0 : access constant cuComplex_h.cuDoubleComplex;
      maxite : int;
      eps : double;
      mu : access cuComplex_h.cuDoubleComplex;
      x : access cuComplex_h.cuDoubleComplex) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp.h:595
   pragma Import (C, cusolverSpZcsreigvsi, "cusolverSpZcsreigvsi");

  -- ----------- enclosed eigenvalues
   function cusolverSpScsreigsHost
     (handle : cusolverSpHandle_t;
      m : int;
      nnz : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrValA : access float;
      csrRowPtrA : access int;
      csrColIndA : access int;
      left_bottom_corner : cuComplex_h.cuComplex;
      right_upper_corner : cuComplex_h.cuComplex;
      num_eigs : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp.h:613
   pragma Import (C, cusolverSpScsreigsHost, "cusolverSpScsreigsHost");

   function cusolverSpDcsreigsHost
     (handle : cusolverSpHandle_t;
      m : int;
      nnz : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrValA : access double;
      csrRowPtrA : access int;
      csrColIndA : access int;
      left_bottom_corner : cuComplex_h.cuDoubleComplex;
      right_upper_corner : cuComplex_h.cuDoubleComplex;
      num_eigs : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp.h:625
   pragma Import (C, cusolverSpDcsreigsHost, "cusolverSpDcsreigsHost");

   function cusolverSpCcsreigsHost
     (handle : cusolverSpHandle_t;
      m : int;
      nnz : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrValA : access constant cuComplex_h.cuComplex;
      csrRowPtrA : access int;
      csrColIndA : access int;
      left_bottom_corner : cuComplex_h.cuComplex;
      right_upper_corner : cuComplex_h.cuComplex;
      num_eigs : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp.h:637
   pragma Import (C, cusolverSpCcsreigsHost, "cusolverSpCcsreigsHost");

   function cusolverSpZcsreigsHost
     (handle : cusolverSpHandle_t;
      m : int;
      nnz : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrValA : access constant cuComplex_h.cuDoubleComplex;
      csrRowPtrA : access int;
      csrColIndA : access int;
      left_bottom_corner : cuComplex_h.cuDoubleComplex;
      right_upper_corner : cuComplex_h.cuDoubleComplex;
      num_eigs : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp.h:649
   pragma Import (C, cusolverSpZcsreigsHost, "cusolverSpZcsreigsHost");

  -- --------- CPU symrcm
  -- *   Symmetric reverse Cuthill McKee permutation         
  -- *
  --  

   function cusolverSpXcsrsymrcmHost
     (handle : cusolverSpHandle_t;
      n : int;
      nnzA : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrRowPtrA : access int;
      csrColIndA : access int;
      p : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp.h:667
   pragma Import (C, cusolverSpXcsrsymrcmHost, "cusolverSpXcsrsymrcmHost");

  -- --------- CPU symmdq 
  -- *   Symmetric minimum degree algorithm based on quotient graph
  -- *
  --  

   function cusolverSpXcsrsymmdqHost
     (handle : cusolverSpHandle_t;
      n : int;
      nnzA : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrRowPtrA : access int;
      csrColIndA : access int;
      p : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp.h:680
   pragma Import (C, cusolverSpXcsrsymmdqHost, "cusolverSpXcsrsymmdqHost");

  -- --------- CPU symmdq 
  -- *   Symmetric Approximate minimum degree algorithm based on quotient graph
  -- *
  --  

   function cusolverSpXcsrsymamdHost
     (handle : cusolverSpHandle_t;
      n : int;
      nnzA : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrRowPtrA : access int;
      csrColIndA : access int;
      p : access int) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp.h:693
   pragma Import (C, cusolverSpXcsrsymamdHost, "cusolverSpXcsrsymamdHost");

  -- --------- CPU permuation
  -- *   P*A*Q^T        
  -- *
  --  

   function cusolverSpXcsrperm_bufferSizeHost
     (handle : cusolverSpHandle_t;
      m : int;
      n : int;
      nnzA : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrRowPtrA : access int;
      csrColIndA : access int;
      p : access int;
      q : access int;
      bufferSizeInBytes : access stddef_h.size_t) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp.h:707
   pragma Import (C, cusolverSpXcsrperm_bufferSizeHost, "cusolverSpXcsrperm_bufferSizeHost");

   function cusolverSpXcsrpermHost
     (handle : cusolverSpHandle_t;
      m : int;
      n : int;
      nnzA : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrRowPtrA : access int;
      csrColIndA : access int;
      p : access int;
      q : access int;
      map : access int;
      pBuffer : System.Address) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp.h:719
   pragma Import (C, cusolverSpXcsrpermHost, "cusolverSpXcsrpermHost");

  -- *  Low-level API: Batched QR
  -- *
  --  

   function cusolverSpCreateCsrqrInfo (info : System.Address) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp.h:739
   pragma Import (C, cusolverSpCreateCsrqrInfo, "cusolverSpCreateCsrqrInfo");

   function cusolverSpDestroyCsrqrInfo (info : csrqrInfo_t) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp.h:742
   pragma Import (C, cusolverSpDestroyCsrqrInfo, "cusolverSpDestroyCsrqrInfo");

   function cusolverSpXcsrqrAnalysisBatched
     (handle : cusolverSpHandle_t;
      m : int;
      n : int;
      nnzA : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrRowPtrA : access int;
      csrColIndA : access int;
      info : csrqrInfo_t) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp.h:746
   pragma Import (C, cusolverSpXcsrqrAnalysisBatched, "cusolverSpXcsrqrAnalysisBatched");

   function cusolverSpScsrqrBufferInfoBatched
     (handle : cusolverSpHandle_t;
      m : int;
      n : int;
      nnz : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrVal : access float;
      csrRowPtr : access int;
      csrColInd : access int;
      batchSize : int;
      info : csrqrInfo_t;
      internalDataInBytes : access stddef_h.size_t;
      workspaceInBytes : access stddef_h.size_t) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp.h:756
   pragma Import (C, cusolverSpScsrqrBufferInfoBatched, "cusolverSpScsrqrBufferInfoBatched");

   function cusolverSpDcsrqrBufferInfoBatched
     (handle : cusolverSpHandle_t;
      m : int;
      n : int;
      nnz : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrVal : access double;
      csrRowPtr : access int;
      csrColInd : access int;
      batchSize : int;
      info : csrqrInfo_t;
      internalDataInBytes : access stddef_h.size_t;
      workspaceInBytes : access stddef_h.size_t) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp.h:770
   pragma Import (C, cusolverSpDcsrqrBufferInfoBatched, "cusolverSpDcsrqrBufferInfoBatched");

   function cusolverSpCcsrqrBufferInfoBatched
     (handle : cusolverSpHandle_t;
      m : int;
      n : int;
      nnz : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrVal : access constant cuComplex_h.cuComplex;
      csrRowPtr : access int;
      csrColInd : access int;
      batchSize : int;
      info : csrqrInfo_t;
      internalDataInBytes : access stddef_h.size_t;
      workspaceInBytes : access stddef_h.size_t) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp.h:784
   pragma Import (C, cusolverSpCcsrqrBufferInfoBatched, "cusolverSpCcsrqrBufferInfoBatched");

   function cusolverSpZcsrqrBufferInfoBatched
     (handle : cusolverSpHandle_t;
      m : int;
      n : int;
      nnz : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrVal : access constant cuComplex_h.cuDoubleComplex;
      csrRowPtr : access int;
      csrColInd : access int;
      batchSize : int;
      info : csrqrInfo_t;
      internalDataInBytes : access stddef_h.size_t;
      workspaceInBytes : access stddef_h.size_t) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp.h:798
   pragma Import (C, cusolverSpZcsrqrBufferInfoBatched, "cusolverSpZcsrqrBufferInfoBatched");

   function cusolverSpScsrqrsvBatched
     (handle : cusolverSpHandle_t;
      m : int;
      n : int;
      nnz : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrValA : access float;
      csrRowPtrA : access int;
      csrColIndA : access int;
      b : access float;
      x : access float;
      batchSize : int;
      info : csrqrInfo_t;
      pBuffer : System.Address) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp.h:812
   pragma Import (C, cusolverSpScsrqrsvBatched, "cusolverSpScsrqrsvBatched");

   function cusolverSpDcsrqrsvBatched
     (handle : cusolverSpHandle_t;
      m : int;
      n : int;
      nnz : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrValA : access double;
      csrRowPtrA : access int;
      csrColIndA : access int;
      b : access double;
      x : access double;
      batchSize : int;
      info : csrqrInfo_t;
      pBuffer : System.Address) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp.h:827
   pragma Import (C, cusolverSpDcsrqrsvBatched, "cusolverSpDcsrqrsvBatched");

   function cusolverSpCcsrqrsvBatched
     (handle : cusolverSpHandle_t;
      m : int;
      n : int;
      nnz : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrValA : access constant cuComplex_h.cuComplex;
      csrRowPtrA : access int;
      csrColIndA : access int;
      b : access constant cuComplex_h.cuComplex;
      x : access cuComplex_h.cuComplex;
      batchSize : int;
      info : csrqrInfo_t;
      pBuffer : System.Address) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp.h:842
   pragma Import (C, cusolverSpCcsrqrsvBatched, "cusolverSpCcsrqrsvBatched");

   function cusolverSpZcsrqrsvBatched
     (handle : cusolverSpHandle_t;
      m : int;
      n : int;
      nnz : int;
      descrA : cusparse_h.cusparseMatDescr_t;
      csrValA : access constant cuComplex_h.cuDoubleComplex;
      csrRowPtrA : access int;
      csrColIndA : access int;
      b : access constant cuComplex_h.cuDoubleComplex;
      x : access cuComplex_h.cuDoubleComplex;
      batchSize : int;
      info : csrqrInfo_t;
      pBuffer : System.Address) return cusolver_common_h.cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolverSp.h:857
   pragma Import (C, cusolverSpZcsrqrsvBatched, "cusolverSpZcsrqrsvBatched");

end cusolverSp_h;

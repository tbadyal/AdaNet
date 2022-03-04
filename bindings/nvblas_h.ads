pragma Ada_2005;
pragma Style_Checks (Off);

with Interfaces.C; use Interfaces.C;
with Interfaces.C.Strings;
limited with cuComplex_h;

package nvblas_h is

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
  -- GEMM  
   procedure sgemm_u
     (transa : Interfaces.C.Strings.chars_ptr;
      transb : Interfaces.C.Strings.chars_ptr;
      m : access int;
      n : access int;
      k : access int;
      alpha : access float;
      a : access float;
      lda : access int;
      b : access float;
      ldb : access int;
      beta : access float;
      c : access float;
      ldc : access int);  -- /usr/local/cuda-8.0/include/nvblas.h:61
   pragma Import (C, sgemm_u, "sgemm_");

   procedure dgemm_u
     (transa : Interfaces.C.Strings.chars_ptr;
      transb : Interfaces.C.Strings.chars_ptr;
      m : access int;
      n : access int;
      k : access int;
      alpha : access double;
      a : access double;
      lda : access int;
      b : access double;
      ldb : access int;
      beta : access double;
      c : access double;
      ldc : access int);  -- /usr/local/cuda-8.0/include/nvblas.h:65
   pragma Import (C, dgemm_u, "dgemm_");

   procedure cgemm_u
     (transa : Interfaces.C.Strings.chars_ptr;
      transb : Interfaces.C.Strings.chars_ptr;
      m : access int;
      n : access int;
      k : access int;
      alpha : access constant cuComplex_h.cuComplex;
      a : access constant cuComplex_h.cuComplex;
      lda : access int;
      b : access constant cuComplex_h.cuComplex;
      ldb : access int;
      beta : access constant cuComplex_h.cuComplex;
      c : access cuComplex_h.cuComplex;
      ldc : access int);  -- /usr/local/cuda-8.0/include/nvblas.h:69
   pragma Import (C, cgemm_u, "cgemm_");

   procedure zgemm_u
     (transa : Interfaces.C.Strings.chars_ptr;
      transb : Interfaces.C.Strings.chars_ptr;
      m : access int;
      n : access int;
      k : access int;
      alpha : access constant cuComplex_h.cuDoubleComplex;
      a : access constant cuComplex_h.cuDoubleComplex;
      lda : access int;
      b : access constant cuComplex_h.cuDoubleComplex;
      ldb : access int;
      beta : access constant cuComplex_h.cuDoubleComplex;
      c : access cuComplex_h.cuDoubleComplex;
      ldc : access int);  -- /usr/local/cuda-8.0/include/nvblas.h:74
   pragma Import (C, zgemm_u, "zgemm_");

   procedure sgemm
     (transa : Interfaces.C.Strings.chars_ptr;
      transb : Interfaces.C.Strings.chars_ptr;
      m : access int;
      n : access int;
      k : access int;
      alpha : access float;
      a : access float;
      lda : access int;
      b : access float;
      ldb : access int;
      beta : access float;
      c : access float;
      ldc : access int);  -- /usr/local/cuda-8.0/include/nvblas.h:79
   pragma Import (C, sgemm, "sgemm");

   procedure dgemm
     (transa : Interfaces.C.Strings.chars_ptr;
      transb : Interfaces.C.Strings.chars_ptr;
      m : access int;
      n : access int;
      k : access int;
      alpha : access double;
      a : access double;
      lda : access int;
      b : access double;
      ldb : access int;
      beta : access double;
      c : access double;
      ldc : access int);  -- /usr/local/cuda-8.0/include/nvblas.h:83
   pragma Import (C, dgemm, "dgemm");

   procedure cgemm
     (transa : Interfaces.C.Strings.chars_ptr;
      transb : Interfaces.C.Strings.chars_ptr;
      m : access int;
      n : access int;
      k : access int;
      alpha : access constant cuComplex_h.cuComplex;
      a : access constant cuComplex_h.cuComplex;
      lda : access int;
      b : access constant cuComplex_h.cuComplex;
      ldb : access int;
      beta : access constant cuComplex_h.cuComplex;
      c : access cuComplex_h.cuComplex;
      ldc : access int);  -- /usr/local/cuda-8.0/include/nvblas.h:87
   pragma Import (C, cgemm, "cgemm");

   procedure zgemm
     (transa : Interfaces.C.Strings.chars_ptr;
      transb : Interfaces.C.Strings.chars_ptr;
      m : access int;
      n : access int;
      k : access int;
      alpha : access constant cuComplex_h.cuDoubleComplex;
      a : access constant cuComplex_h.cuDoubleComplex;
      lda : access int;
      b : access constant cuComplex_h.cuDoubleComplex;
      ldb : access int;
      beta : access constant cuComplex_h.cuDoubleComplex;
      c : access cuComplex_h.cuDoubleComplex;
      ldc : access int);  -- /usr/local/cuda-8.0/include/nvblas.h:92
   pragma Import (C, zgemm, "zgemm");

  -- SYRK  
   procedure ssyrk_u
     (uplo : Interfaces.C.Strings.chars_ptr;
      trans : Interfaces.C.Strings.chars_ptr;
      n : access int;
      k : access int;
      alpha : access float;
      a : access float;
      lda : access int;
      beta : access float;
      c : access float;
      ldc : access int);  -- /usr/local/cuda-8.0/include/nvblas.h:98
   pragma Import (C, ssyrk_u, "ssyrk_");

   procedure dsyrk_u
     (uplo : Interfaces.C.Strings.chars_ptr;
      trans : Interfaces.C.Strings.chars_ptr;
      n : access int;
      k : access int;
      alpha : access double;
      a : access double;
      lda : access int;
      beta : access double;
      c : access double;
      ldc : access int);  -- /usr/local/cuda-8.0/include/nvblas.h:102
   pragma Import (C, dsyrk_u, "dsyrk_");

   procedure csyrk_u
     (uplo : Interfaces.C.Strings.chars_ptr;
      trans : Interfaces.C.Strings.chars_ptr;
      n : access int;
      k : access int;
      alpha : access constant cuComplex_h.cuComplex;
      a : access constant cuComplex_h.cuComplex;
      lda : access int;
      beta : access constant cuComplex_h.cuComplex;
      c : access cuComplex_h.cuComplex;
      ldc : access int);  -- /usr/local/cuda-8.0/include/nvblas.h:106
   pragma Import (C, csyrk_u, "csyrk_");

   procedure zsyrk_u
     (uplo : Interfaces.C.Strings.chars_ptr;
      trans : Interfaces.C.Strings.chars_ptr;
      n : access int;
      k : access int;
      alpha : access constant cuComplex_h.cuDoubleComplex;
      a : access constant cuComplex_h.cuDoubleComplex;
      lda : access int;
      beta : access constant cuComplex_h.cuDoubleComplex;
      c : access cuComplex_h.cuDoubleComplex;
      ldc : access int);  -- /usr/local/cuda-8.0/include/nvblas.h:110
   pragma Import (C, zsyrk_u, "zsyrk_");

   procedure ssyrk
     (uplo : Interfaces.C.Strings.chars_ptr;
      trans : Interfaces.C.Strings.chars_ptr;
      n : access int;
      k : access int;
      alpha : access float;
      a : access float;
      lda : access int;
      beta : access float;
      c : access float;
      ldc : access int);  -- /usr/local/cuda-8.0/include/nvblas.h:114
   pragma Import (C, ssyrk, "ssyrk");

   procedure dsyrk
     (uplo : Interfaces.C.Strings.chars_ptr;
      trans : Interfaces.C.Strings.chars_ptr;
      n : access int;
      k : access int;
      alpha : access double;
      a : access double;
      lda : access int;
      beta : access double;
      c : access double;
      ldc : access int);  -- /usr/local/cuda-8.0/include/nvblas.h:118
   pragma Import (C, dsyrk, "dsyrk");

   procedure csyrk
     (uplo : Interfaces.C.Strings.chars_ptr;
      trans : Interfaces.C.Strings.chars_ptr;
      n : access int;
      k : access int;
      alpha : access constant cuComplex_h.cuComplex;
      a : access constant cuComplex_h.cuComplex;
      lda : access int;
      beta : access constant cuComplex_h.cuComplex;
      c : access cuComplex_h.cuComplex;
      ldc : access int);  -- /usr/local/cuda-8.0/include/nvblas.h:122
   pragma Import (C, csyrk, "csyrk");

   procedure zsyrk
     (uplo : Interfaces.C.Strings.chars_ptr;
      trans : Interfaces.C.Strings.chars_ptr;
      n : access int;
      k : access int;
      alpha : access constant cuComplex_h.cuDoubleComplex;
      a : access constant cuComplex_h.cuDoubleComplex;
      lda : access int;
      beta : access constant cuComplex_h.cuDoubleComplex;
      c : access cuComplex_h.cuDoubleComplex;
      ldc : access int);  -- /usr/local/cuda-8.0/include/nvblas.h:126
   pragma Import (C, zsyrk, "zsyrk");

  -- HERK  
   procedure cherk_u
     (uplo : Interfaces.C.Strings.chars_ptr;
      trans : Interfaces.C.Strings.chars_ptr;
      n : access int;
      k : access int;
      alpha : access float;
      a : access constant cuComplex_h.cuComplex;
      lda : access int;
      beta : access float;
      c : access cuComplex_h.cuComplex;
      ldc : access int);  -- /usr/local/cuda-8.0/include/nvblas.h:131
   pragma Import (C, cherk_u, "cherk_");

   procedure zherk_u
     (uplo : Interfaces.C.Strings.chars_ptr;
      trans : Interfaces.C.Strings.chars_ptr;
      n : access int;
      k : access int;
      alpha : access double;
      a : access constant cuComplex_h.cuDoubleComplex;
      lda : access int;
      beta : access double;
      c : access cuComplex_h.cuDoubleComplex;
      ldc : access int);  -- /usr/local/cuda-8.0/include/nvblas.h:135
   pragma Import (C, zherk_u, "zherk_");

   procedure cherk
     (uplo : Interfaces.C.Strings.chars_ptr;
      trans : Interfaces.C.Strings.chars_ptr;
      n : access int;
      k : access int;
      alpha : access float;
      a : access constant cuComplex_h.cuComplex;
      lda : access int;
      beta : access float;
      c : access cuComplex_h.cuComplex;
      ldc : access int);  -- /usr/local/cuda-8.0/include/nvblas.h:139
   pragma Import (C, cherk, "cherk");

   procedure zherk
     (uplo : Interfaces.C.Strings.chars_ptr;
      trans : Interfaces.C.Strings.chars_ptr;
      n : access int;
      k : access int;
      alpha : access double;
      a : access constant cuComplex_h.cuDoubleComplex;
      lda : access int;
      beta : access double;
      c : access cuComplex_h.cuDoubleComplex;
      ldc : access int);  -- /usr/local/cuda-8.0/include/nvblas.h:143
   pragma Import (C, zherk, "zherk");

  -- TRSM  
   procedure strsm_u
     (side : Interfaces.C.Strings.chars_ptr;
      uplo : Interfaces.C.Strings.chars_ptr;
      transa : Interfaces.C.Strings.chars_ptr;
      diag : Interfaces.C.Strings.chars_ptr;
      m : access int;
      n : access int;
      alpha : access float;
      a : access float;
      lda : access int;
      b : access float;
      ldb : access int);  -- /usr/local/cuda-8.0/include/nvblas.h:148
   pragma Import (C, strsm_u, "strsm_");

   procedure dtrsm_u
     (side : Interfaces.C.Strings.chars_ptr;
      uplo : Interfaces.C.Strings.chars_ptr;
      transa : Interfaces.C.Strings.chars_ptr;
      diag : Interfaces.C.Strings.chars_ptr;
      m : access int;
      n : access int;
      alpha : access double;
      a : access double;
      lda : access int;
      b : access double;
      ldb : access int);  -- /usr/local/cuda-8.0/include/nvblas.h:152
   pragma Import (C, dtrsm_u, "dtrsm_");

   procedure ctrsm_u
     (side : Interfaces.C.Strings.chars_ptr;
      uplo : Interfaces.C.Strings.chars_ptr;
      transa : Interfaces.C.Strings.chars_ptr;
      diag : Interfaces.C.Strings.chars_ptr;
      m : access int;
      n : access int;
      alpha : access constant cuComplex_h.cuComplex;
      a : access constant cuComplex_h.cuComplex;
      lda : access int;
      b : access cuComplex_h.cuComplex;
      ldb : access int);  -- /usr/local/cuda-8.0/include/nvblas.h:156
   pragma Import (C, ctrsm_u, "ctrsm_");

   procedure ztrsm_u
     (side : Interfaces.C.Strings.chars_ptr;
      uplo : Interfaces.C.Strings.chars_ptr;
      transa : Interfaces.C.Strings.chars_ptr;
      diag : Interfaces.C.Strings.chars_ptr;
      m : access int;
      n : access int;
      alpha : access constant cuComplex_h.cuDoubleComplex;
      a : access constant cuComplex_h.cuDoubleComplex;
      lda : access int;
      b : access cuComplex_h.cuDoubleComplex;
      ldb : access int);  -- /usr/local/cuda-8.0/include/nvblas.h:160
   pragma Import (C, ztrsm_u, "ztrsm_");

   procedure strsm
     (side : Interfaces.C.Strings.chars_ptr;
      uplo : Interfaces.C.Strings.chars_ptr;
      transa : Interfaces.C.Strings.chars_ptr;
      diag : Interfaces.C.Strings.chars_ptr;
      m : access int;
      n : access int;
      alpha : access float;
      a : access float;
      lda : access int;
      b : access float;
      ldb : access int);  -- /usr/local/cuda-8.0/include/nvblas.h:164
   pragma Import (C, strsm, "strsm");

   procedure dtrsm
     (side : Interfaces.C.Strings.chars_ptr;
      uplo : Interfaces.C.Strings.chars_ptr;
      transa : Interfaces.C.Strings.chars_ptr;
      diag : Interfaces.C.Strings.chars_ptr;
      m : access int;
      n : access int;
      alpha : access double;
      a : access double;
      lda : access int;
      b : access double;
      ldb : access int);  -- /usr/local/cuda-8.0/include/nvblas.h:168
   pragma Import (C, dtrsm, "dtrsm");

   procedure ctrsm
     (side : Interfaces.C.Strings.chars_ptr;
      uplo : Interfaces.C.Strings.chars_ptr;
      transa : Interfaces.C.Strings.chars_ptr;
      diag : Interfaces.C.Strings.chars_ptr;
      m : access int;
      n : access int;
      alpha : access constant cuComplex_h.cuComplex;
      a : access constant cuComplex_h.cuComplex;
      lda : access int;
      b : access cuComplex_h.cuComplex;
      ldb : access int);  -- /usr/local/cuda-8.0/include/nvblas.h:172
   pragma Import (C, ctrsm, "ctrsm");

   procedure ztrsm
     (side : Interfaces.C.Strings.chars_ptr;
      uplo : Interfaces.C.Strings.chars_ptr;
      transa : Interfaces.C.Strings.chars_ptr;
      diag : Interfaces.C.Strings.chars_ptr;
      m : access int;
      n : access int;
      alpha : access constant cuComplex_h.cuDoubleComplex;
      a : access constant cuComplex_h.cuDoubleComplex;
      lda : access int;
      b : access cuComplex_h.cuDoubleComplex;
      ldb : access int);  -- /usr/local/cuda-8.0/include/nvblas.h:176
   pragma Import (C, ztrsm, "ztrsm");

  -- SYMM  
   procedure ssymm_u
     (side : Interfaces.C.Strings.chars_ptr;
      uplo : Interfaces.C.Strings.chars_ptr;
      m : access int;
      n : access int;
      alpha : access float;
      a : access float;
      lda : access int;
      b : access float;
      ldb : access int;
      beta : access float;
      c : access float;
      ldc : access int);  -- /usr/local/cuda-8.0/include/nvblas.h:181
   pragma Import (C, ssymm_u, "ssymm_");

   procedure dsymm_u
     (side : Interfaces.C.Strings.chars_ptr;
      uplo : Interfaces.C.Strings.chars_ptr;
      m : access int;
      n : access int;
      alpha : access double;
      a : access double;
      lda : access int;
      b : access double;
      ldb : access int;
      beta : access double;
      c : access double;
      ldc : access int);  -- /usr/local/cuda-8.0/include/nvblas.h:185
   pragma Import (C, dsymm_u, "dsymm_");

   procedure csymm_u
     (side : Interfaces.C.Strings.chars_ptr;
      uplo : Interfaces.C.Strings.chars_ptr;
      m : access int;
      n : access int;
      alpha : access constant cuComplex_h.cuComplex;
      a : access constant cuComplex_h.cuComplex;
      lda : access int;
      b : access constant cuComplex_h.cuComplex;
      ldb : access int;
      beta : access constant cuComplex_h.cuComplex;
      c : access cuComplex_h.cuComplex;
      ldc : access int);  -- /usr/local/cuda-8.0/include/nvblas.h:189
   pragma Import (C, csymm_u, "csymm_");

   procedure zsymm_u
     (side : Interfaces.C.Strings.chars_ptr;
      uplo : Interfaces.C.Strings.chars_ptr;
      m : access int;
      n : access int;
      alpha : access constant cuComplex_h.cuDoubleComplex;
      a : access constant cuComplex_h.cuDoubleComplex;
      lda : access int;
      b : access constant cuComplex_h.cuDoubleComplex;
      ldb : access int;
      beta : access constant cuComplex_h.cuDoubleComplex;
      c : access cuComplex_h.cuDoubleComplex;
      ldc : access int);  -- /usr/local/cuda-8.0/include/nvblas.h:194
   pragma Import (C, zsymm_u, "zsymm_");

   procedure ssymm
     (side : Interfaces.C.Strings.chars_ptr;
      uplo : Interfaces.C.Strings.chars_ptr;
      m : access int;
      n : access int;
      alpha : access float;
      a : access float;
      lda : access int;
      b : access float;
      ldb : access int;
      beta : access float;
      c : access float;
      ldc : access int);  -- /usr/local/cuda-8.0/include/nvblas.h:199
   pragma Import (C, ssymm, "ssymm");

   procedure dsymm
     (side : Interfaces.C.Strings.chars_ptr;
      uplo : Interfaces.C.Strings.chars_ptr;
      m : access int;
      n : access int;
      alpha : access double;
      a : access double;
      lda : access int;
      b : access double;
      ldb : access int;
      beta : access double;
      c : access double;
      ldc : access int);  -- /usr/local/cuda-8.0/include/nvblas.h:203
   pragma Import (C, dsymm, "dsymm");

   procedure csymm
     (side : Interfaces.C.Strings.chars_ptr;
      uplo : Interfaces.C.Strings.chars_ptr;
      m : access int;
      n : access int;
      alpha : access constant cuComplex_h.cuComplex;
      a : access constant cuComplex_h.cuComplex;
      lda : access int;
      b : access constant cuComplex_h.cuComplex;
      ldb : access int;
      beta : access constant cuComplex_h.cuComplex;
      c : access cuComplex_h.cuComplex;
      ldc : access int);  -- /usr/local/cuda-8.0/include/nvblas.h:207
   pragma Import (C, csymm, "csymm");

   procedure zsymm
     (side : Interfaces.C.Strings.chars_ptr;
      uplo : Interfaces.C.Strings.chars_ptr;
      m : access int;
      n : access int;
      alpha : access constant cuComplex_h.cuDoubleComplex;
      a : access constant cuComplex_h.cuDoubleComplex;
      lda : access int;
      b : access constant cuComplex_h.cuDoubleComplex;
      ldb : access int;
      beta : access constant cuComplex_h.cuDoubleComplex;
      c : access cuComplex_h.cuDoubleComplex;
      ldc : access int);  -- /usr/local/cuda-8.0/include/nvblas.h:212
   pragma Import (C, zsymm, "zsymm");

  -- HEMM  
   procedure chemm_u
     (side : Interfaces.C.Strings.chars_ptr;
      uplo : Interfaces.C.Strings.chars_ptr;
      m : access int;
      n : access int;
      alpha : access constant cuComplex_h.cuComplex;
      a : access constant cuComplex_h.cuComplex;
      lda : access int;
      b : access constant cuComplex_h.cuComplex;
      ldb : access int;
      beta : access constant cuComplex_h.cuComplex;
      c : access cuComplex_h.cuComplex;
      ldc : access int);  -- /usr/local/cuda-8.0/include/nvblas.h:218
   pragma Import (C, chemm_u, "chemm_");

   procedure zhemm_u
     (side : Interfaces.C.Strings.chars_ptr;
      uplo : Interfaces.C.Strings.chars_ptr;
      m : access int;
      n : access int;
      alpha : access constant cuComplex_h.cuDoubleComplex;
      a : access constant cuComplex_h.cuDoubleComplex;
      lda : access int;
      b : access constant cuComplex_h.cuDoubleComplex;
      ldb : access int;
      beta : access constant cuComplex_h.cuDoubleComplex;
      c : access cuComplex_h.cuDoubleComplex;
      ldc : access int);  -- /usr/local/cuda-8.0/include/nvblas.h:223
   pragma Import (C, zhemm_u, "zhemm_");

  -- HEMM with no underscore 
   procedure chemm
     (side : Interfaces.C.Strings.chars_ptr;
      uplo : Interfaces.C.Strings.chars_ptr;
      m : access int;
      n : access int;
      alpha : access constant cuComplex_h.cuComplex;
      a : access constant cuComplex_h.cuComplex;
      lda : access int;
      b : access constant cuComplex_h.cuComplex;
      ldb : access int;
      beta : access constant cuComplex_h.cuComplex;
      c : access cuComplex_h.cuComplex;
      ldc : access int);  -- /usr/local/cuda-8.0/include/nvblas.h:228
   pragma Import (C, chemm, "chemm");

   procedure zhemm
     (side : Interfaces.C.Strings.chars_ptr;
      uplo : Interfaces.C.Strings.chars_ptr;
      m : access int;
      n : access int;
      alpha : access constant cuComplex_h.cuDoubleComplex;
      a : access constant cuComplex_h.cuDoubleComplex;
      lda : access int;
      b : access constant cuComplex_h.cuDoubleComplex;
      ldb : access int;
      beta : access constant cuComplex_h.cuDoubleComplex;
      c : access cuComplex_h.cuDoubleComplex;
      ldc : access int);  -- /usr/local/cuda-8.0/include/nvblas.h:232
   pragma Import (C, zhemm, "zhemm");

  -- SYR2K  
   procedure ssyr2k_u
     (uplo : Interfaces.C.Strings.chars_ptr;
      trans : Interfaces.C.Strings.chars_ptr;
      n : access int;
      k : access int;
      alpha : access float;
      a : access float;
      lda : access int;
      b : access float;
      ldb : access int;
      beta : access float;
      c : access float;
      ldc : access int);  -- /usr/local/cuda-8.0/include/nvblas.h:237
   pragma Import (C, ssyr2k_u, "ssyr2k_");

   procedure dsyr2k_u
     (uplo : Interfaces.C.Strings.chars_ptr;
      trans : Interfaces.C.Strings.chars_ptr;
      n : access int;
      k : access int;
      alpha : access double;
      a : access double;
      lda : access int;
      b : access double;
      ldb : access int;
      beta : access double;
      c : access double;
      ldc : access int);  -- /usr/local/cuda-8.0/include/nvblas.h:243
   pragma Import (C, dsyr2k_u, "dsyr2k_");

   procedure csyr2k_u
     (uplo : Interfaces.C.Strings.chars_ptr;
      trans : Interfaces.C.Strings.chars_ptr;
      n : access int;
      k : access int;
      alpha : access constant cuComplex_h.cuComplex;
      a : access constant cuComplex_h.cuComplex;
      lda : access int;
      b : access constant cuComplex_h.cuComplex;
      ldb : access int;
      beta : access constant cuComplex_h.cuComplex;
      c : access cuComplex_h.cuComplex;
      ldc : access int);  -- /usr/local/cuda-8.0/include/nvblas.h:250
   pragma Import (C, csyr2k_u, "csyr2k_");

   procedure zsyr2k_u
     (uplo : Interfaces.C.Strings.chars_ptr;
      trans : Interfaces.C.Strings.chars_ptr;
      n : access int;
      k : access int;
      alpha : access constant cuComplex_h.cuDoubleComplex;
      a : access constant cuComplex_h.cuDoubleComplex;
      lda : access int;
      b : access constant cuComplex_h.cuDoubleComplex;
      ldb : access int;
      beta : access constant cuComplex_h.cuDoubleComplex;
      c : access cuComplex_h.cuDoubleComplex;
      ldc : access int);  -- /usr/local/cuda-8.0/include/nvblas.h:256
   pragma Import (C, zsyr2k_u, "zsyr2k_");

  -- SYR2K no_underscore 
   procedure ssyr2k
     (uplo : Interfaces.C.Strings.chars_ptr;
      trans : Interfaces.C.Strings.chars_ptr;
      n : access int;
      k : access int;
      alpha : access float;
      a : access float;
      lda : access int;
      b : access float;
      ldb : access int;
      beta : access float;
      c : access float;
      ldc : access int);  -- /usr/local/cuda-8.0/include/nvblas.h:263
   pragma Import (C, ssyr2k, "ssyr2k");

   procedure dsyr2k
     (uplo : Interfaces.C.Strings.chars_ptr;
      trans : Interfaces.C.Strings.chars_ptr;
      n : access int;
      k : access int;
      alpha : access double;
      a : access double;
      lda : access int;
      b : access double;
      ldb : access int;
      beta : access double;
      c : access double;
      ldc : access int);  -- /usr/local/cuda-8.0/include/nvblas.h:268
   pragma Import (C, dsyr2k, "dsyr2k");

   procedure csyr2k
     (uplo : Interfaces.C.Strings.chars_ptr;
      trans : Interfaces.C.Strings.chars_ptr;
      n : access int;
      k : access int;
      alpha : access constant cuComplex_h.cuComplex;
      a : access constant cuComplex_h.cuComplex;
      lda : access int;
      b : access constant cuComplex_h.cuComplex;
      ldb : access int;
      beta : access constant cuComplex_h.cuComplex;
      c : access cuComplex_h.cuComplex;
      ldc : access int);  -- /usr/local/cuda-8.0/include/nvblas.h:275
   pragma Import (C, csyr2k, "csyr2k");

   procedure zsyr2k
     (uplo : Interfaces.C.Strings.chars_ptr;
      trans : Interfaces.C.Strings.chars_ptr;
      n : access int;
      k : access int;
      alpha : access constant cuComplex_h.cuDoubleComplex;
      a : access constant cuComplex_h.cuDoubleComplex;
      lda : access int;
      b : access constant cuComplex_h.cuDoubleComplex;
      ldb : access int;
      beta : access constant cuComplex_h.cuDoubleComplex;
      c : access cuComplex_h.cuDoubleComplex;
      ldc : access int);  -- /usr/local/cuda-8.0/include/nvblas.h:281
   pragma Import (C, zsyr2k, "zsyr2k");

  -- HERK  
   procedure cher2k_u
     (uplo : Interfaces.C.Strings.chars_ptr;
      trans : Interfaces.C.Strings.chars_ptr;
      n : access int;
      k : access int;
      alpha : access constant cuComplex_h.cuComplex;
      a : access constant cuComplex_h.cuComplex;
      lda : access int;
      b : access constant cuComplex_h.cuComplex;
      ldb : access int;
      beta : access float;
      c : access cuComplex_h.cuComplex;
      ldc : access int);  -- /usr/local/cuda-8.0/include/nvblas.h:288
   pragma Import (C, cher2k_u, "cher2k_");

   procedure zher2k_u
     (uplo : Interfaces.C.Strings.chars_ptr;
      trans : Interfaces.C.Strings.chars_ptr;
      n : access int;
      k : access int;
      alpha : access constant cuComplex_h.cuDoubleComplex;
      a : access constant cuComplex_h.cuDoubleComplex;
      lda : access int;
      b : access constant cuComplex_h.cuDoubleComplex;
      ldb : access int;
      beta : access double;
      c : access cuComplex_h.cuDoubleComplex;
      ldc : access int);  -- /usr/local/cuda-8.0/include/nvblas.h:294
   pragma Import (C, zher2k_u, "zher2k_");

  -- HER2K with no underscore  
   procedure cher2k
     (uplo : Interfaces.C.Strings.chars_ptr;
      trans : Interfaces.C.Strings.chars_ptr;
      n : access int;
      k : access int;
      alpha : access constant cuComplex_h.cuComplex;
      a : access constant cuComplex_h.cuComplex;
      lda : access int;
      b : access constant cuComplex_h.cuComplex;
      ldb : access int;
      beta : access float;
      c : access cuComplex_h.cuComplex;
      ldc : access int);  -- /usr/local/cuda-8.0/include/nvblas.h:301
   pragma Import (C, cher2k, "cher2k");

   procedure zher2k
     (uplo : Interfaces.C.Strings.chars_ptr;
      trans : Interfaces.C.Strings.chars_ptr;
      n : access int;
      k : access int;
      alpha : access constant cuComplex_h.cuDoubleComplex;
      a : access constant cuComplex_h.cuDoubleComplex;
      lda : access int;
      b : access constant cuComplex_h.cuDoubleComplex;
      ldb : access int;
      beta : access double;
      c : access cuComplex_h.cuDoubleComplex;
      ldc : access int);  -- /usr/local/cuda-8.0/include/nvblas.h:307
   pragma Import (C, zher2k, "zher2k");

  -- TRMM  
   procedure strmm_u
     (side : Interfaces.C.Strings.chars_ptr;
      uplo : Interfaces.C.Strings.chars_ptr;
      transa : Interfaces.C.Strings.chars_ptr;
      diag : Interfaces.C.Strings.chars_ptr;
      m : access int;
      n : access int;
      alpha : access float;
      a : access float;
      lda : access int;
      b : access float;
      ldb : access int);  -- /usr/local/cuda-8.0/include/nvblas.h:314
   pragma Import (C, strmm_u, "strmm_");

   procedure dtrmm_u
     (side : Interfaces.C.Strings.chars_ptr;
      uplo : Interfaces.C.Strings.chars_ptr;
      transa : Interfaces.C.Strings.chars_ptr;
      diag : Interfaces.C.Strings.chars_ptr;
      m : access int;
      n : access int;
      alpha : access double;
      a : access double;
      lda : access int;
      b : access double;
      ldb : access int);  -- /usr/local/cuda-8.0/include/nvblas.h:318
   pragma Import (C, dtrmm_u, "dtrmm_");

   procedure ctrmm_u
     (side : Interfaces.C.Strings.chars_ptr;
      uplo : Interfaces.C.Strings.chars_ptr;
      transa : Interfaces.C.Strings.chars_ptr;
      diag : Interfaces.C.Strings.chars_ptr;
      m : access int;
      n : access int;
      alpha : access constant cuComplex_h.cuComplex;
      a : access constant cuComplex_h.cuComplex;
      lda : access int;
      b : access cuComplex_h.cuComplex;
      ldb : access int);  -- /usr/local/cuda-8.0/include/nvblas.h:322
   pragma Import (C, ctrmm_u, "ctrmm_");

   procedure ztrmm_u
     (side : Interfaces.C.Strings.chars_ptr;
      uplo : Interfaces.C.Strings.chars_ptr;
      transa : Interfaces.C.Strings.chars_ptr;
      diag : Interfaces.C.Strings.chars_ptr;
      m : access int;
      n : access int;
      alpha : access constant cuComplex_h.cuDoubleComplex;
      a : access constant cuComplex_h.cuDoubleComplex;
      lda : access int;
      b : access cuComplex_h.cuDoubleComplex;
      ldb : access int);  -- /usr/local/cuda-8.0/include/nvblas.h:326
   pragma Import (C, ztrmm_u, "ztrmm_");

   procedure strmm
     (side : Interfaces.C.Strings.chars_ptr;
      uplo : Interfaces.C.Strings.chars_ptr;
      transa : Interfaces.C.Strings.chars_ptr;
      diag : Interfaces.C.Strings.chars_ptr;
      m : access int;
      n : access int;
      alpha : access float;
      a : access float;
      lda : access int;
      b : access float;
      ldb : access int);  -- /usr/local/cuda-8.0/include/nvblas.h:330
   pragma Import (C, strmm, "strmm");

   procedure dtrmm
     (side : Interfaces.C.Strings.chars_ptr;
      uplo : Interfaces.C.Strings.chars_ptr;
      transa : Interfaces.C.Strings.chars_ptr;
      diag : Interfaces.C.Strings.chars_ptr;
      m : access int;
      n : access int;
      alpha : access double;
      a : access double;
      lda : access int;
      b : access double;
      ldb : access int);  -- /usr/local/cuda-8.0/include/nvblas.h:334
   pragma Import (C, dtrmm, "dtrmm");

   procedure ctrmm
     (side : Interfaces.C.Strings.chars_ptr;
      uplo : Interfaces.C.Strings.chars_ptr;
      transa : Interfaces.C.Strings.chars_ptr;
      diag : Interfaces.C.Strings.chars_ptr;
      m : access int;
      n : access int;
      alpha : access constant cuComplex_h.cuComplex;
      a : access constant cuComplex_h.cuComplex;
      lda : access int;
      b : access cuComplex_h.cuComplex;
      ldb : access int);  -- /usr/local/cuda-8.0/include/nvblas.h:338
   pragma Import (C, ctrmm, "ctrmm");

   procedure ztrmm
     (side : Interfaces.C.Strings.chars_ptr;
      uplo : Interfaces.C.Strings.chars_ptr;
      transa : Interfaces.C.Strings.chars_ptr;
      diag : Interfaces.C.Strings.chars_ptr;
      m : access int;
      n : access int;
      alpha : access constant cuComplex_h.cuDoubleComplex;
      a : access constant cuComplex_h.cuDoubleComplex;
      lda : access int;
      b : access cuComplex_h.cuDoubleComplex;
      ldb : access int);  -- /usr/local/cuda-8.0/include/nvblas.h:342
   pragma Import (C, ztrmm, "ztrmm");

end nvblas_h;

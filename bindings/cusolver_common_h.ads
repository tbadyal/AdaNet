pragma Ada_2005;
pragma Style_Checks (Off);

with Interfaces.C; use Interfaces.C;
with library_types_h;

package cusolver_common_h is

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

   type cusolverStatus_t is 
     (CUSOLVER_STATUS_SUCCESS,
      CUSOLVER_STATUS_NOT_INITIALIZED,
      CUSOLVER_STATUS_ALLOC_FAILED,
      CUSOLVER_STATUS_INVALID_VALUE,
      CUSOLVER_STATUS_ARCH_MISMATCH,
      CUSOLVER_STATUS_MAPPING_ERROR,
      CUSOLVER_STATUS_EXECUTION_FAILED,
      CUSOLVER_STATUS_INTERNAL_ERROR,
      CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED,
      CUSOLVER_STATUS_NOT_SUPPORTED,
      CUSOLVER_STATUS_ZERO_PIVOT,
      CUSOLVER_STATUS_INVALID_LICENSE);
   pragma Convention (C, cusolverStatus_t);  -- /usr/local/cuda-8.0/include/cusolver_common.h:81

   subtype cusolverEigType_t is unsigned;
   CUSOLVER_EIG_TYPE_1 : constant cusolverEigType_t := 1;
   CUSOLVER_EIG_TYPE_2 : constant cusolverEigType_t := 2;
   CUSOLVER_EIG_TYPE_3 : constant cusolverEigType_t := 3;  -- /usr/local/cuda-8.0/include/cusolver_common.h:87

   type cusolverEigMode_t is 
     (CUSOLVER_EIG_MODE_NOVECTOR,
      CUSOLVER_EIG_MODE_VECTOR);
   pragma Convention (C, cusolverEigMode_t);  -- /usr/local/cuda-8.0/include/cusolver_common.h:92

   function cusolverGetProperty (c_type : library_types_h.libraryPropertyType; value : access int) return cusolverStatus_t;  -- /usr/local/cuda-8.0/include/cusolver_common.h:95
   pragma Import (C, cusolverGetProperty, "cusolverGetProperty");

end cusolver_common_h;

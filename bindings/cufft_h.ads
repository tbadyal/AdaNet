pragma Ada_2005;
pragma Style_Checks (Off);

with Interfaces.C; use Interfaces.C;
with vector_types_h;
with stddef_h;
with System;
with driver_types_h;
with library_types_h;

package cufft_h is

   --  unsupported macro: CUFFTAPI __attribute__ ((visibility ("default")))
   MAX_CUFFT_ERROR : constant := 16#11#;  --  /usr/local/cuda-8.0/include/cufft.h:99

   CUFFT_FORWARD : constant := -1;  --  /usr/local/cuda-8.0/include/cufft.h:117
   CUFFT_INVERSE : constant := 1;  --  /usr/local/cuda-8.0/include/cufft.h:118
   --  unsupported macro: CUFFT_COMPATIBILITY_DEFAULT CUFFT_COMPATIBILITY_FFTW_PADDING

   MAX_SHIM_RANK : constant := 3;  --  /usr/local/cuda-8.0/include/cufft.h:140

  -- Copyright 2005-2014 NVIDIA Corporation.  All rights reserved.
  --  *
  --  * NOTICE TO LICENSEE:
  --  *
  --  * The source code and/or documentation ("Licensed Deliverables") are
  --  * subject to NVIDIA intellectual property rights under U.S. and
  --  * international Copyright laws.
  --  *
  --  * The Licensed Deliverables contained herein are PROPRIETARY and
  --  * CONFIDENTIAL to NVIDIA and are being provided under the terms and
  --  * conditions of a form of NVIDIA software license agreement by and
  --  * between NVIDIA and Licensee ("License Agreement") or electronically
  --  * accepted by Licensee.  Notwithstanding any terms or conditions to
  --  * the contrary in the License Agreement, reproduction or disclosure
  --  * of the Licensed Deliverables to any third party without the express
  --  * written consent of NVIDIA is prohibited.
  --  *
  --  * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
  --  * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
  --  * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  THEY ARE
  --  * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
  --  * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
  --  * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
  --  * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
  --  * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
  --  * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
  --  * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
  --  * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
  --  * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
  --  * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
  --  * OF THESE LICENSED DELIVERABLES.
  --  *
  --  * U.S. Government End Users.  These Licensed Deliverables are a
  --  * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
  --  * 1995), consisting of "commercial computer software" and "commercial
  --  * computer software documentation" as such terms are used in 48
  --  * C.F.R. 12.212 (SEPT 1995) and are provided to the U.S. Government
  --  * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
  --  * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
  --  * U.S. Government End Users acquire the Licensed Deliverables with
  --  * only those rights set forth herein.
  --  *
  --  * Any use of the Licensed Deliverables in individual and commercial
  --  * software must include, in the user documentation and internal
  --  * comments to the code, the above Disclaimer and U.S. Government End
  --  * Users Notice.
  --   

  --!
  --* \file cufft.h  
  --* \brief Public header file for the NVIDIA CUDA FFT library (CUFFT)  
  -- 

  -- CUFFT API function return values 
   type cufftResult_t is 
     (CUFFT_SUCCESS,
      CUFFT_INVALID_PLAN,
      CUFFT_ALLOC_FAILED,
      CUFFT_INVALID_TYPE,
      CUFFT_INVALID_VALUE,
      CUFFT_INTERNAL_ERROR,
      CUFFT_EXEC_FAILED,
      CUFFT_SETUP_FAILED,
      CUFFT_INVALID_SIZE,
      CUFFT_UNALIGNED_DATA,
      CUFFT_INCOMPLETE_PARAMETER_LIST,
      CUFFT_INVALID_DEVICE,
      CUFFT_PARSE_ERROR,
      CUFFT_NO_WORKSPACE,
      CUFFT_NOT_IMPLEMENTED,
      CUFFT_LICENSE_ERROR,
      CUFFT_NOT_SUPPORTED);
   pragma Convention (C, cufftResult_t);  -- /usr/local/cuda-8.0/include/cufft.h:78

   subtype cufftResult is cufftResult_t;

  -- CUFFT defines and supports the following data types
  -- cufftReal is a single-precision, floating-point real data type.
  -- cufftDoubleReal is a double-precision, real data type.
   subtype cufftReal is float;  -- /usr/local/cuda-8.0/include/cufft.h:107

   subtype cufftDoubleReal is double;  -- /usr/local/cuda-8.0/include/cufft.h:108

  -- cufftComplex is a single-precision, floating-point complex data type that 
  -- consists of interleaved real and imaginary components.
  -- cufftDoubleComplex is the double-precision equivalent.
   subtype cufftComplex is vector_types_h.float2;

   subtype cufftDoubleComplex is vector_types_h.double2;

  -- CUFFT transform directions 
  -- CUFFT supports the following transform types 
   subtype cufftType_t is unsigned;
   CUFFT_R2C : constant cufftType_t := 42;
   CUFFT_C2R : constant cufftType_t := 44;
   CUFFT_C2C : constant cufftType_t := 41;
   CUFFT_D2Z : constant cufftType_t := 106;
   CUFFT_Z2D : constant cufftType_t := 108;
   CUFFT_Z2Z : constant cufftType_t := 105;  -- /usr/local/cuda-8.0/include/cufft.h:121

  -- Real to Complex (interleaved)
  -- Complex (interleaved) to Real
  -- Complex to Complex, interleaved
  -- Double to Double-Complex
  -- Double-Complex to Double
  -- Double-Complex to Double-Complex
   subtype cufftType is cufftType_t;

  -- CUFFT supports the following data layouts
   subtype cufftCompatibility_t is unsigned;
   CUFFT_COMPATIBILITY_FFTW_PADDING : constant cufftCompatibility_t := 1;  -- /usr/local/cuda-8.0/include/cufft.h:131

  -- The default value
   subtype cufftCompatibility is cufftCompatibility_t;

  -- structure definition used by the shim between old and new APIs
  -- cufftHandle is a handle type used to store and access CUFFT plans.
   subtype cufftHandle is int;  -- /usr/local/cuda-8.0/include/cufft.h:143

   function cufftPlan1d
     (plan : access cufftHandle;
      nx : int;
      c_type : cufftType;
      batch : int) return cufftResult;  -- /usr/local/cuda-8.0/include/cufft.h:146
   pragma Import (C, cufftPlan1d, "cufftPlan1d");

   function cufftPlan2d
     (plan : access cufftHandle;
      nx : int;
      ny : int;
      c_type : cufftType) return cufftResult;  -- /usr/local/cuda-8.0/include/cufft.h:151
   pragma Import (C, cufftPlan2d, "cufftPlan2d");

   function cufftPlan3d
     (plan : access cufftHandle;
      nx : int;
      ny : int;
      nz : int;
      c_type : cufftType) return cufftResult;  -- /usr/local/cuda-8.0/include/cufft.h:155
   pragma Import (C, cufftPlan3d, "cufftPlan3d");

   function cufftPlanMany
     (plan : access cufftHandle;
      rank : int;
      n : access int;
      inembed : access int;
      istride : int;
      idist : int;
      onembed : access int;
      ostride : int;
      odist : int;
      c_type : cufftType;
      batch : int) return cufftResult;  -- /usr/local/cuda-8.0/include/cufft.h:159
   pragma Import (C, cufftPlanMany, "cufftPlanMany");

   function cufftMakePlan1d
     (plan : cufftHandle;
      nx : int;
      c_type : cufftType;
      batch : int;
      workSize : access stddef_h.size_t) return cufftResult;  -- /usr/local/cuda-8.0/include/cufft.h:167
   pragma Import (C, cufftMakePlan1d, "cufftMakePlan1d");

   function cufftMakePlan2d
     (plan : cufftHandle;
      nx : int;
      ny : int;
      c_type : cufftType;
      workSize : access stddef_h.size_t) return cufftResult;  -- /usr/local/cuda-8.0/include/cufft.h:173
   pragma Import (C, cufftMakePlan2d, "cufftMakePlan2d");

   function cufftMakePlan3d
     (plan : cufftHandle;
      nx : int;
      ny : int;
      nz : int;
      c_type : cufftType;
      workSize : access stddef_h.size_t) return cufftResult;  -- /usr/local/cuda-8.0/include/cufft.h:178
   pragma Import (C, cufftMakePlan3d, "cufftMakePlan3d");

   function cufftMakePlanMany
     (plan : cufftHandle;
      rank : int;
      n : access int;
      inembed : access int;
      istride : int;
      idist : int;
      onembed : access int;
      ostride : int;
      odist : int;
      c_type : cufftType;
      batch : int;
      workSize : access stddef_h.size_t) return cufftResult;  -- /usr/local/cuda-8.0/include/cufft.h:183
   pragma Import (C, cufftMakePlanMany, "cufftMakePlanMany");

   function cufftMakePlanMany64
     (plan : cufftHandle;
      rank : int;
      n : access Long_Long_Integer;
      inembed : access Long_Long_Integer;
      istride : Long_Long_Integer;
      idist : Long_Long_Integer;
      onembed : access Long_Long_Integer;
      ostride : Long_Long_Integer;
      odist : Long_Long_Integer;
      c_type : cufftType;
      batch : Long_Long_Integer;
      workSize : access stddef_h.size_t) return cufftResult;  -- /usr/local/cuda-8.0/include/cufft.h:192
   pragma Import (C, cufftMakePlanMany64, "cufftMakePlanMany64");

   function cufftGetSizeMany64
     (plan : cufftHandle;
      rank : int;
      n : access Long_Long_Integer;
      inembed : access Long_Long_Integer;
      istride : Long_Long_Integer;
      idist : Long_Long_Integer;
      onembed : access Long_Long_Integer;
      ostride : Long_Long_Integer;
      odist : Long_Long_Integer;
      c_type : cufftType;
      batch : Long_Long_Integer;
      workSize : access stddef_h.size_t) return cufftResult;  -- /usr/local/cuda-8.0/include/cufft.h:204
   pragma Import (C, cufftGetSizeMany64, "cufftGetSizeMany64");

   function cufftEstimate1d
     (nx : int;
      c_type : cufftType;
      batch : int;
      workSize : access stddef_h.size_t) return cufftResult;  -- /usr/local/cuda-8.0/include/cufft.h:218
   pragma Import (C, cufftEstimate1d, "cufftEstimate1d");

   function cufftEstimate2d
     (nx : int;
      ny : int;
      c_type : cufftType;
      workSize : access stddef_h.size_t) return cufftResult;  -- /usr/local/cuda-8.0/include/cufft.h:223
   pragma Import (C, cufftEstimate2d, "cufftEstimate2d");

   function cufftEstimate3d
     (nx : int;
      ny : int;
      nz : int;
      c_type : cufftType;
      workSize : access stddef_h.size_t) return cufftResult;  -- /usr/local/cuda-8.0/include/cufft.h:227
   pragma Import (C, cufftEstimate3d, "cufftEstimate3d");

   function cufftEstimateMany
     (rank : int;
      n : access int;
      inembed : access int;
      istride : int;
      idist : int;
      onembed : access int;
      ostride : int;
      odist : int;
      c_type : cufftType;
      batch : int;
      workSize : access stddef_h.size_t) return cufftResult;  -- /usr/local/cuda-8.0/include/cufft.h:231
   pragma Import (C, cufftEstimateMany, "cufftEstimateMany");

   function cufftCreate (handle : access cufftHandle) return cufftResult;  -- /usr/local/cuda-8.0/include/cufft.h:239
   pragma Import (C, cufftCreate, "cufftCreate");

   function cufftGetSize1d
     (handle : cufftHandle;
      nx : int;
      c_type : cufftType;
      batch : int;
      workSize : access stddef_h.size_t) return cufftResult;  -- /usr/local/cuda-8.0/include/cufft.h:241
   pragma Import (C, cufftGetSize1d, "cufftGetSize1d");

   function cufftGetSize2d
     (handle : cufftHandle;
      nx : int;
      ny : int;
      c_type : cufftType;
      workSize : access stddef_h.size_t) return cufftResult;  -- /usr/local/cuda-8.0/include/cufft.h:247
   pragma Import (C, cufftGetSize2d, "cufftGetSize2d");

   function cufftGetSize3d
     (handle : cufftHandle;
      nx : int;
      ny : int;
      nz : int;
      c_type : cufftType;
      workSize : access stddef_h.size_t) return cufftResult;  -- /usr/local/cuda-8.0/include/cufft.h:252
   pragma Import (C, cufftGetSize3d, "cufftGetSize3d");

   function cufftGetSizeMany
     (handle : cufftHandle;
      rank : int;
      n : access int;
      inembed : access int;
      istride : int;
      idist : int;
      onembed : access int;
      ostride : int;
      odist : int;
      c_type : cufftType;
      batch : int;
      workArea : access stddef_h.size_t) return cufftResult;  -- /usr/local/cuda-8.0/include/cufft.h:257
   pragma Import (C, cufftGetSizeMany, "cufftGetSizeMany");

   function cufftGetSize (handle : cufftHandle; workSize : access stddef_h.size_t) return cufftResult;  -- /usr/local/cuda-8.0/include/cufft.h:263
   pragma Import (C, cufftGetSize, "cufftGetSize");

   function cufftSetWorkArea (plan : cufftHandle; workArea : System.Address) return cufftResult;  -- /usr/local/cuda-8.0/include/cufft.h:265
   pragma Import (C, cufftSetWorkArea, "cufftSetWorkArea");

   function cufftSetAutoAllocation (plan : cufftHandle; autoAllocate : int) return cufftResult;  -- /usr/local/cuda-8.0/include/cufft.h:267
   pragma Import (C, cufftSetAutoAllocation, "cufftSetAutoAllocation");

   function cufftExecC2C
     (plan : cufftHandle;
      idata : access cufftComplex;
      odata : access cufftComplex;
      direction : int) return cufftResult;  -- /usr/local/cuda-8.0/include/cufft.h:269
   pragma Import (C, cufftExecC2C, "cufftExecC2C");

   function cufftExecR2C
     (plan : cufftHandle;
      idata : access cufftReal;
      odata : access cufftComplex) return cufftResult;  -- /usr/local/cuda-8.0/include/cufft.h:274
   pragma Import (C, cufftExecR2C, "cufftExecR2C");

   function cufftExecC2R
     (plan : cufftHandle;
      idata : access cufftComplex;
      odata : access cufftReal) return cufftResult;  -- /usr/local/cuda-8.0/include/cufft.h:278
   pragma Import (C, cufftExecC2R, "cufftExecC2R");

   function cufftExecZ2Z
     (plan : cufftHandle;
      idata : access cufftDoubleComplex;
      odata : access cufftDoubleComplex;
      direction : int) return cufftResult;  -- /usr/local/cuda-8.0/include/cufft.h:282
   pragma Import (C, cufftExecZ2Z, "cufftExecZ2Z");

   function cufftExecD2Z
     (plan : cufftHandle;
      idata : access cufftDoubleReal;
      odata : access cufftDoubleComplex) return cufftResult;  -- /usr/local/cuda-8.0/include/cufft.h:287
   pragma Import (C, cufftExecD2Z, "cufftExecD2Z");

   function cufftExecZ2D
     (plan : cufftHandle;
      idata : access cufftDoubleComplex;
      odata : access cufftDoubleReal) return cufftResult;  -- /usr/local/cuda-8.0/include/cufft.h:291
   pragma Import (C, cufftExecZ2D, "cufftExecZ2D");

  -- utility functions
   function cufftSetStream (plan : cufftHandle; stream : driver_types_h.cudaStream_t) return cufftResult;  -- /usr/local/cuda-8.0/include/cufft.h:297
   pragma Import (C, cufftSetStream, "cufftSetStream");

   function cufftSetCompatibilityMode (plan : cufftHandle; mode : cufftCompatibility) return cufftResult;  -- /usr/local/cuda-8.0/include/cufft.h:300
   pragma Import (C, cufftSetCompatibilityMode, "cufftSetCompatibilityMode");

   function cufftDestroy (plan : cufftHandle) return cufftResult;  -- /usr/local/cuda-8.0/include/cufft.h:303
   pragma Import (C, cufftDestroy, "cufftDestroy");

   function cufftGetVersion (version : access int) return cufftResult;  -- /usr/local/cuda-8.0/include/cufft.h:305
   pragma Import (C, cufftGetVersion, "cufftGetVersion");

   function cufftGetProperty (c_type : library_types_h.libraryPropertyType; value : access int) return cufftResult;  -- /usr/local/cuda-8.0/include/cufft.h:307
   pragma Import (C, cufftGetProperty, "cufftGetProperty");

end cufft_h;

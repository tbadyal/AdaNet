pragma Ada_2005;
pragma Style_Checks (Off);

with Interfaces.C; use Interfaces.C;
with cufft_h;
with System;
limited with cudalibxt_h;
with stddef_h;
with library_types_h;

package cufftXt_h is

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
  --* \file cufftXt.h  
  --* \brief Public header file for the NVIDIA CUDA FFT library (CUFFT)  
  -- 

  -- cufftXtSubFormat identifies the data layout of 
  -- a memory descriptor owned by cufft.
  -- note that multi GPU cufft does not yet support out-of-place transforms
   type cufftXtSubFormat_t is 
     (CUFFT_XT_FORMAT_INPUT,
      CUFFT_XT_FORMAT_OUTPUT,
      CUFFT_XT_FORMAT_INPLACE,
      CUFFT_XT_FORMAT_INPLACE_SHUFFLED,
      CUFFT_XT_FORMAT_1D_INPUT_SHUFFLED,
      CUFFT_FORMAT_UNDEFINED);
   pragma Convention (C, cufftXtSubFormat_t);  -- /usr/local/cuda-8.0/include/cufftXt.h:79

  --by default input is in linear order across GPUs
  --by default output is in scrambled order depending on transform
  --by default inplace is input order, which is linear across GPUs
  --shuffled output order after execution of the transform
  --shuffled input order prior to execution of 1D transforms
   subtype cufftXtSubFormat is cufftXtSubFormat_t;

  -- cufftXtCopyType specifies the type of copy for cufftXtMemcpy
   type cufftXtCopyType_t is 
     (CUFFT_COPY_HOST_TO_DEVICE,
      CUFFT_COPY_DEVICE_TO_HOST,
      CUFFT_COPY_DEVICE_TO_DEVICE,
      CUFFT_COPY_UNDEFINED);
   pragma Convention (C, cufftXtCopyType_t);  -- /usr/local/cuda-8.0/include/cufftXt.h:91

   subtype cufftXtCopyType is cufftXtCopyType_t;

  -- cufftXtQueryType specifies the type of query for cufftXtQueryPlan
   type cufftXtQueryType_t is 
     (CUFFT_QUERY_1D_FACTORS,
      CUFFT_QUERY_UNDEFINED);
   pragma Convention (C, cufftXtQueryType_t);  -- /usr/local/cuda-8.0/include/cufftXt.h:101

   subtype cufftXtQueryType is cufftXtQueryType_t;

   type cufftXt1dFactors_t is record
      size : aliased Long_Long_Integer;  -- /usr/local/cuda-8.0/include/cufftXt.h:107
      stringCount : aliased Long_Long_Integer;  -- /usr/local/cuda-8.0/include/cufftXt.h:108
      stringLength : aliased Long_Long_Integer;  -- /usr/local/cuda-8.0/include/cufftXt.h:109
      substringLength : aliased Long_Long_Integer;  -- /usr/local/cuda-8.0/include/cufftXt.h:110
      factor1 : aliased Long_Long_Integer;  -- /usr/local/cuda-8.0/include/cufftXt.h:111
      factor2 : aliased Long_Long_Integer;  -- /usr/local/cuda-8.0/include/cufftXt.h:112
      stringMask : aliased Long_Long_Integer;  -- /usr/local/cuda-8.0/include/cufftXt.h:113
      substringMask : aliased Long_Long_Integer;  -- /usr/local/cuda-8.0/include/cufftXt.h:114
      factor1Mask : aliased Long_Long_Integer;  -- /usr/local/cuda-8.0/include/cufftXt.h:115
      factor2Mask : aliased Long_Long_Integer;  -- /usr/local/cuda-8.0/include/cufftXt.h:116
      stringShift : aliased int;  -- /usr/local/cuda-8.0/include/cufftXt.h:117
      substringShift : aliased int;  -- /usr/local/cuda-8.0/include/cufftXt.h:118
      factor1Shift : aliased int;  -- /usr/local/cuda-8.0/include/cufftXt.h:119
      factor2Shift : aliased int;  -- /usr/local/cuda-8.0/include/cufftXt.h:120
   end record;
   pragma Convention (C_Pass_By_Copy, cufftXt1dFactors_t);  -- /usr/local/cuda-8.0/include/cufftXt.h:106

   subtype cufftXt1dFactors is cufftXt1dFactors_t;

  -- multi-GPU routines
   function cufftXtSetGPUs
     (handle : cufft_h.cufftHandle;
      nGPUs : int;
      whichGPUs : access int) return cufft_h.cufftResult;  -- /usr/local/cuda-8.0/include/cufftXt.h:124
   pragma Import (C, cufftXtSetGPUs, "cufftXtSetGPUs");

   function cufftXtMalloc
     (plan : cufft_h.cufftHandle;
      descriptor : System.Address;
      format : cufftXtSubFormat) return cufft_h.cufftResult;  -- /usr/local/cuda-8.0/include/cufftXt.h:126
   pragma Import (C, cufftXtMalloc, "cufftXtMalloc");

   function cufftXtMemcpy
     (plan : cufft_h.cufftHandle;
      dstPointer : System.Address;
      srcPointer : System.Address;
      c_type : cufftXtCopyType) return cufft_h.cufftResult;  -- /usr/local/cuda-8.0/include/cufftXt.h:130
   pragma Import (C, cufftXtMemcpy, "cufftXtMemcpy");

   function cufftXtFree (descriptor : access cudalibxt_h.cudaLibXtDesc) return cufft_h.cufftResult;  -- /usr/local/cuda-8.0/include/cufftXt.h:135
   pragma Import (C, cufftXtFree, "cufftXtFree");

   function cufftXtSetWorkArea (plan : cufft_h.cufftHandle; workArea : System.Address) return cufft_h.cufftResult;  -- /usr/local/cuda-8.0/include/cufftXt.h:137
   pragma Import (C, cufftXtSetWorkArea, "cufftXtSetWorkArea");

   function cufftXtExecDescriptorC2C
     (plan : cufft_h.cufftHandle;
      input : access cudalibxt_h.cudaLibXtDesc;
      output : access cudalibxt_h.cudaLibXtDesc;
      direction : int) return cufft_h.cufftResult;  -- /usr/local/cuda-8.0/include/cufftXt.h:139
   pragma Import (C, cufftXtExecDescriptorC2C, "cufftXtExecDescriptorC2C");

   function cufftXtExecDescriptorR2C
     (plan : cufft_h.cufftHandle;
      input : access cudalibxt_h.cudaLibXtDesc;
      output : access cudalibxt_h.cudaLibXtDesc) return cufft_h.cufftResult;  -- /usr/local/cuda-8.0/include/cufftXt.h:144
   pragma Import (C, cufftXtExecDescriptorR2C, "cufftXtExecDescriptorR2C");

   function cufftXtExecDescriptorC2R
     (plan : cufft_h.cufftHandle;
      input : access cudalibxt_h.cudaLibXtDesc;
      output : access cudalibxt_h.cudaLibXtDesc) return cufft_h.cufftResult;  -- /usr/local/cuda-8.0/include/cufftXt.h:148
   pragma Import (C, cufftXtExecDescriptorC2R, "cufftXtExecDescriptorC2R");

   function cufftXtExecDescriptorZ2Z
     (plan : cufft_h.cufftHandle;
      input : access cudalibxt_h.cudaLibXtDesc;
      output : access cudalibxt_h.cudaLibXtDesc;
      direction : int) return cufft_h.cufftResult;  -- /usr/local/cuda-8.0/include/cufftXt.h:152
   pragma Import (C, cufftXtExecDescriptorZ2Z, "cufftXtExecDescriptorZ2Z");

   function cufftXtExecDescriptorD2Z
     (plan : cufft_h.cufftHandle;
      input : access cudalibxt_h.cudaLibXtDesc;
      output : access cudalibxt_h.cudaLibXtDesc) return cufft_h.cufftResult;  -- /usr/local/cuda-8.0/include/cufftXt.h:157
   pragma Import (C, cufftXtExecDescriptorD2Z, "cufftXtExecDescriptorD2Z");

   function cufftXtExecDescriptorZ2D
     (plan : cufft_h.cufftHandle;
      input : access cudalibxt_h.cudaLibXtDesc;
      output : access cudalibxt_h.cudaLibXtDesc) return cufft_h.cufftResult;  -- /usr/local/cuda-8.0/include/cufftXt.h:161
   pragma Import (C, cufftXtExecDescriptorZ2D, "cufftXtExecDescriptorZ2D");

  -- Utility functions
   function cufftXtQueryPlan
     (plan : cufft_h.cufftHandle;
      queryStruct : System.Address;
      queryType : cufftXtQueryType) return cufft_h.cufftResult;  -- /usr/local/cuda-8.0/include/cufftXt.h:167
   pragma Import (C, cufftXtQueryPlan, "cufftXtQueryPlan");

  -- callbacks
   type cufftXtCallbackType_t is 
     (CUFFT_CB_LD_COMPLEX,
      CUFFT_CB_LD_COMPLEX_DOUBLE,
      CUFFT_CB_LD_REAL,
      CUFFT_CB_LD_REAL_DOUBLE,
      CUFFT_CB_ST_COMPLEX,
      CUFFT_CB_ST_COMPLEX_DOUBLE,
      CUFFT_CB_ST_REAL,
      CUFFT_CB_ST_REAL_DOUBLE,
      CUFFT_CB_UNDEFINED);
   pragma Convention (C, cufftXtCallbackType_t);  -- /usr/local/cuda-8.0/include/cufftXt.h:173

   subtype cufftXtCallbackType is cufftXtCallbackType_t;

   type cufftCallbackLoadC is access function
        (arg1 : System.Address;
         arg2 : stddef_h.size_t;
         arg3 : System.Address;
         arg4 : System.Address) return cufft_h.cufftComplex;
   pragma Convention (C, cufftCallbackLoadC);  -- /usr/local/cuda-8.0/include/cufftXt.h:186

   type cufftCallbackLoadZ is access function
        (arg1 : System.Address;
         arg2 : stddef_h.size_t;
         arg3 : System.Address;
         arg4 : System.Address) return cufft_h.cufftDoubleComplex;
   pragma Convention (C, cufftCallbackLoadZ);  -- /usr/local/cuda-8.0/include/cufftXt.h:187

   type cufftCallbackLoadR is access function
        (arg1 : System.Address;
         arg2 : stddef_h.size_t;
         arg3 : System.Address;
         arg4 : System.Address) return cufft_h.cufftReal;
   pragma Convention (C, cufftCallbackLoadR);  -- /usr/local/cuda-8.0/include/cufftXt.h:188

   type cufftCallbackLoadD is access function
        (arg1 : System.Address;
         arg2 : stddef_h.size_t;
         arg3 : System.Address;
         arg4 : System.Address) return cufft_h.cufftDoubleReal;
   pragma Convention (C, cufftCallbackLoadD);  -- /usr/local/cuda-8.0/include/cufftXt.h:189

   type cufftCallbackStoreC is access procedure
        (arg1 : System.Address;
         arg2 : stddef_h.size_t;
         arg3 : cufft_h.cufftComplex;
         arg4 : System.Address;
         arg5 : System.Address);
   pragma Convention (C, cufftCallbackStoreC);  -- /usr/local/cuda-8.0/include/cufftXt.h:191

   type cufftCallbackStoreZ is access procedure
        (arg1 : System.Address;
         arg2 : stddef_h.size_t;
         arg3 : cufft_h.cufftDoubleComplex;
         arg4 : System.Address;
         arg5 : System.Address);
   pragma Convention (C, cufftCallbackStoreZ);  -- /usr/local/cuda-8.0/include/cufftXt.h:192

   type cufftCallbackStoreR is access procedure
        (arg1 : System.Address;
         arg2 : stddef_h.size_t;
         arg3 : cufft_h.cufftReal;
         arg4 : System.Address;
         arg5 : System.Address);
   pragma Convention (C, cufftCallbackStoreR);  -- /usr/local/cuda-8.0/include/cufftXt.h:193

   type cufftCallbackStoreD is access procedure
        (arg1 : System.Address;
         arg2 : stddef_h.size_t;
         arg3 : cufft_h.cufftDoubleReal;
         arg4 : System.Address;
         arg5 : System.Address);
   pragma Convention (C, cufftCallbackStoreD);  -- /usr/local/cuda-8.0/include/cufftXt.h:194

   function cufftXtSetCallback
     (plan : cufft_h.cufftHandle;
      callback_routine : System.Address;
      cbType : cufftXtCallbackType;
      caller_info : System.Address) return cufft_h.cufftResult;  -- /usr/local/cuda-8.0/include/cufftXt.h:197
   pragma Import (C, cufftXtSetCallback, "cufftXtSetCallback");

   function cufftXtClearCallback (plan : cufft_h.cufftHandle; cbType : cufftXtCallbackType) return cufft_h.cufftResult;  -- /usr/local/cuda-8.0/include/cufftXt.h:198
   pragma Import (C, cufftXtClearCallback, "cufftXtClearCallback");

   function cufftXtSetCallbackSharedSize
     (plan : cufft_h.cufftHandle;
      cbType : cufftXtCallbackType;
      sharedSize : stddef_h.size_t) return cufft_h.cufftResult;  -- /usr/local/cuda-8.0/include/cufftXt.h:199
   pragma Import (C, cufftXtSetCallbackSharedSize, "cufftXtSetCallbackSharedSize");

   function cufftXtMakePlanMany
     (plan : cufft_h.cufftHandle;
      rank : int;
      n : access Long_Long_Integer;
      inembed : access Long_Long_Integer;
      istride : Long_Long_Integer;
      idist : Long_Long_Integer;
      inputtype : library_types_h.cudaDataType;
      onembed : access Long_Long_Integer;
      ostride : Long_Long_Integer;
      odist : Long_Long_Integer;
      outputtype : library_types_h.cudaDataType;
      batch : Long_Long_Integer;
      workSize : access stddef_h.size_t;
      executiontype : library_types_h.cudaDataType) return cufft_h.cufftResult;  -- /usr/local/cuda-8.0/include/cufftXt.h:201
   pragma Import (C, cufftXtMakePlanMany, "cufftXtMakePlanMany");

   function cufftXtGetSizeMany
     (plan : cufft_h.cufftHandle;
      rank : int;
      n : access Long_Long_Integer;
      inembed : access Long_Long_Integer;
      istride : Long_Long_Integer;
      idist : Long_Long_Integer;
      inputtype : library_types_h.cudaDataType;
      onembed : access Long_Long_Integer;
      ostride : Long_Long_Integer;
      odist : Long_Long_Integer;
      outputtype : library_types_h.cudaDataType;
      batch : Long_Long_Integer;
      workSize : access stddef_h.size_t;
      executiontype : library_types_h.cudaDataType) return cufft_h.cufftResult;  -- /usr/local/cuda-8.0/include/cufftXt.h:216
   pragma Import (C, cufftXtGetSizeMany, "cufftXtGetSizeMany");

   function cufftXtExec
     (plan : cufft_h.cufftHandle;
      input : System.Address;
      output : System.Address;
      direction : int) return cufft_h.cufftResult;  -- /usr/local/cuda-8.0/include/cufftXt.h:231
   pragma Import (C, cufftXtExec, "cufftXtExec");

   function cufftXtExecDescriptor
     (plan : cufft_h.cufftHandle;
      input : access cudalibxt_h.cudaLibXtDesc;
      output : access cudalibxt_h.cudaLibXtDesc;
      direction : int) return cufft_h.cufftResult;  -- /usr/local/cuda-8.0/include/cufftXt.h:236
   pragma Import (C, cufftXtExecDescriptor, "cufftXtExecDescriptor");

end cufftXt_h;

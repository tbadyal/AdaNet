pragma Ada_2005;
pragma Style_Checks (Off);

with Interfaces.C; use Interfaces.C;
with System;
with stddef_h;

package cudalibxt_h is

   CUDA_XT_DESCRIPTOR_VERSION : constant := 16#01000000#;  --  /usr/local/cuda-8.0/include/cudalibxt.h:58

   MAX_CUDA_DESCRIPTOR_GPUS : constant := 64;  --  /usr/local/cuda-8.0/include/cudalibxt.h:74

  -- Copyright 2013,2014 NVIDIA Corporation.  All rights reserved.
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
  --* \file cudalibxt.h  
  --* \brief Public header file for the NVIDIA library multi-GPU support structures  
  -- 

   type cudaXtCopyType_t is 
     (LIB_XT_COPY_HOST_TO_DEVICE,
      LIB_XT_COPY_DEVICE_TO_HOST,
      LIB_XT_COPY_DEVICE_TO_DEVICE);
   pragma Convention (C, cudaXtCopyType_t);  -- /usr/local/cuda-8.0/include/cudalibxt.h:60

   subtype cudaLibXtCopyType is cudaXtCopyType_t;

   type libFormat_t is 
     (LIB_FORMAT_CUFFT,
      LIB_FORMAT_UNDEFINED);
   pragma Convention (C, libFormat_t);  -- /usr/local/cuda-8.0/include/cudalibxt.h:67

   subtype libFormat is libFormat_t;

  --descriptor version
   type cudaXtDesc_t_GPUs_array is array (0 .. 63) of aliased int;
   type cudaXtDesc_t_data_array is array (0 .. 63) of System.Address;
   type cudaXtDesc_t_size_array is array (0 .. 63) of aliased stddef_h.size_t;
   type cudaXtDesc_t is record
      version : aliased int;  -- /usr/local/cuda-8.0/include/cudalibxt.h:77
      nGPUs : aliased int;  -- /usr/local/cuda-8.0/include/cudalibxt.h:78
      GPUs : aliased cudaXtDesc_t_GPUs_array;  -- /usr/local/cuda-8.0/include/cudalibxt.h:79
      data : cudaXtDesc_t_data_array;  -- /usr/local/cuda-8.0/include/cudalibxt.h:80
      size : aliased cudaXtDesc_t_size_array;  -- /usr/local/cuda-8.0/include/cudalibxt.h:81
      cudaXtState : System.Address;  -- /usr/local/cuda-8.0/include/cudalibxt.h:82
   end record;
   pragma Convention (C_Pass_By_Copy, cudaXtDesc_t);  -- /usr/local/cuda-8.0/include/cudalibxt.h:76

  --number of GPUs 
  --array of device IDs
  --array of pointers to data, one per GPU
  --array of data sizes, one per GPU
  --opaque CUDA utility structure
   subtype cudaXtDesc is cudaXtDesc_t;

  --descriptor version
   type cudaLibXtDesc_t is record
      version : aliased int;  -- /usr/local/cuda-8.0/include/cudalibxt.h:87
      descriptor : access cudaXtDesc;  -- /usr/local/cuda-8.0/include/cudalibxt.h:88
      library : aliased libFormat;  -- /usr/local/cuda-8.0/include/cudalibxt.h:89
      subFormat : aliased int;  -- /usr/local/cuda-8.0/include/cudalibxt.h:90
      libDescriptor : System.Address;  -- /usr/local/cuda-8.0/include/cudalibxt.h:91
   end record;
   pragma Convention (C_Pass_By_Copy, cudaLibXtDesc_t);  -- /usr/local/cuda-8.0/include/cudalibxt.h:86

  --multi-GPU memory descriptor
  --which library recognizes the format
  --library specific enumerator of sub formats
  --library specific descriptor e.g. FFT transform plan object
   subtype cudaLibXtDesc is cudaLibXtDesc_t;

end cudalibxt_h;

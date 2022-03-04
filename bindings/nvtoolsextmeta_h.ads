pragma Ada_2005;
pragma Style_Checks (Off);

with Interfaces.C; use Interfaces.C;
with Interfaces.C.Strings;
with System;

package nvToolsExtMeta_h is

   --  arg-macro: procedure NVTX_PACK __Declaration__ __attribute__((__packed__))
   --    __Declaration__ __attribute__((__packed__))
  --* Copyright 2009-2016  NVIDIA Corporation.  All rights reserved.
  --*
  --* NOTICE TO USER:
  --*
  --* This source code is subject to NVIDIA ownership rights under U.S. and
  --* international Copyright laws.
  --*
  --* This software and the information contained herein is PROPRIETARY and
  --* CONFIDENTIAL to NVIDIA and is being provided under the terms and conditions
  --* of a form of NVIDIA software license agreement.
  --*
  --* NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
  --* CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
  --* IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
  --* REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
  --* MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
  --* IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
  --* OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
  --* OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
  --* OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
  --* OR PERFORMANCE OF THIS SOURCE CODE.
  --*
  --* U.S. Government End Users.   This source code is a "commercial item" as
  --* that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
  --* "commercial computer  software"  and "commercial computer software
  --* documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
  --* and is provided to the U.S. Government only as a commercial end item.
  --* Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
  --* 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
  --* source code with only those rights set forth herein.
  --*
  --* Any use of this source code in individual and commercial software must
  --* include, in the user documentation and internal comments to the code,
  --* the above Disclaimer and U.S. Government End Users Notice.
  -- 

  --* \mainpage
  -- * \section Introduction
  -- * The NVIDIA Tools Extension library is a set of functions that a
  -- * developer can use to provide additional information to tools.
  -- * The additional information is used by the tool to improve
  -- * analysis and visualization of data.
  -- *
  -- * The library introduces close to zero overhead if no tool is
  -- * attached to the application.  The overhead when a tool is
  -- * attached is specific to the tool.
  --  

  -- Structs defining parameters for NVTX API functions  
   type NvtxMarkEx is record
      null;
   end record;
   pragma Convention (C_Pass_By_Copy, NvtxMarkEx);  -- /usr/local/cuda-8.0/include/nvToolsExtMeta.h:64

   type NvtxDomainMarkEx is record
      core : aliased NvtxMarkEx;  -- /usr/local/cuda-8.0/include/nvToolsExtMeta.h:65
   end record;
   pragma Convention (C_Pass_By_Copy, NvtxDomainMarkEx);  -- /usr/local/cuda-8.0/include/nvToolsExtMeta.h:65

   type NvtxMarkA is record
      message : Interfaces.C.Strings.chars_ptr;  -- /usr/local/cuda-8.0/include/nvToolsExtMeta.h:68
   end record;
   pragma Convention (C_Pass_By_Copy, NvtxMarkA);  -- /usr/local/cuda-8.0/include/nvToolsExtMeta.h:68

   type NvtxMarkW is record
      message : access wchar_t;  -- /usr/local/cuda-8.0/include/nvToolsExtMeta.h:69
   end record;
   pragma Convention (C_Pass_By_Copy, NvtxMarkW);  -- /usr/local/cuda-8.0/include/nvToolsExtMeta.h:69

   type NvtxRangeStartEx is record
      null;
   end record;
   pragma Convention (C_Pass_By_Copy, NvtxRangeStartEx);  -- /usr/local/cuda-8.0/include/nvToolsExtMeta.h:70

   type NvtxDomainRangeStartEx is record
      core : aliased NvtxRangeStartEx;  -- /usr/local/cuda-8.0/include/nvToolsExtMeta.h:72
   end record;
   pragma Convention (C_Pass_By_Copy, NvtxDomainRangeStartEx);  -- /usr/local/cuda-8.0/include/nvToolsExtMeta.h:72

   type NvtxRangeStartA is record
      message : Interfaces.C.Strings.chars_ptr;  -- /usr/local/cuda-8.0/include/nvToolsExtMeta.h:75
   end record;
   pragma Convention (C_Pass_By_Copy, NvtxRangeStartA);  -- /usr/local/cuda-8.0/include/nvToolsExtMeta.h:75

   type NvtxRangeStartW is record
      message : access wchar_t;  -- /usr/local/cuda-8.0/include/nvToolsExtMeta.h:76
   end record;
   pragma Convention (C_Pass_By_Copy, NvtxRangeStartW);  -- /usr/local/cuda-8.0/include/nvToolsExtMeta.h:76

   type NvtxRangeEnd is record
      null;
   end record;
   pragma Convention (C_Pass_By_Copy, NvtxRangeEnd);  -- /usr/local/cuda-8.0/include/nvToolsExtMeta.h:77

   type NvtxDomainRangeEnd is record
      core : aliased NvtxRangeEnd;  -- /usr/local/cuda-8.0/include/nvToolsExtMeta.h:79
   end record;
   pragma Convention (C_Pass_By_Copy, NvtxDomainRangeEnd);  -- /usr/local/cuda-8.0/include/nvToolsExtMeta.h:79

   type NvtxRangePushEx is record
      null;
   end record;
   pragma Convention (C_Pass_By_Copy, NvtxRangePushEx);  -- /usr/local/cuda-8.0/include/nvToolsExtMeta.h:82

   type NvtxDomainRangePushEx is record
      core : aliased NvtxRangePushEx;  -- /usr/local/cuda-8.0/include/nvToolsExtMeta.h:84
   end record;
   pragma Convention (C_Pass_By_Copy, NvtxDomainRangePushEx);  -- /usr/local/cuda-8.0/include/nvToolsExtMeta.h:84

   type NvtxRangePushA is record
      message : Interfaces.C.Strings.chars_ptr;  -- /usr/local/cuda-8.0/include/nvToolsExtMeta.h:87
   end record;
   pragma Convention (C_Pass_By_Copy, NvtxRangePushA);  -- /usr/local/cuda-8.0/include/nvToolsExtMeta.h:87

   type NvtxRangePushW is record
      message : access wchar_t;  -- /usr/local/cuda-8.0/include/nvToolsExtMeta.h:88
   end record;
   pragma Convention (C_Pass_By_Copy, NvtxRangePushW);  -- /usr/local/cuda-8.0/include/nvToolsExtMeta.h:88

   type NvtxDomainRangePop is record
      null;
   end record;
   pragma Convention (C_Pass_By_Copy, NvtxDomainRangePop);  -- /usr/local/cuda-8.0/include/nvToolsExtMeta.h:89

  --     NvtxRangePop     - no parameters, params will be NULL.  
   type NvtxDomainResourceCreate is record
      null;
   end record;
   pragma Convention (C_Pass_By_Copy, NvtxDomainResourceCreate);  -- /usr/local/cuda-8.0/include/nvToolsExtMeta.h:91

   type NvtxDomainResourceDestroy is record
      null;
   end record;
   pragma Convention (C_Pass_By_Copy, NvtxDomainResourceDestroy);  -- /usr/local/cuda-8.0/include/nvToolsExtMeta.h:92

   type NvtxDomainRegisterString is record
      str : System.Address;  -- /usr/local/cuda-8.0/include/nvToolsExtMeta.h:93
   end record;
   pragma Convention (C_Pass_By_Copy, NvtxDomainRegisterString);  -- /usr/local/cuda-8.0/include/nvToolsExtMeta.h:93

   type NvtxDomainCreate is record
      name : System.Address;  -- /usr/local/cuda-8.0/include/nvToolsExtMeta.h:94
   end record;
   pragma Convention (C_Pass_By_Copy, NvtxDomainCreate);  -- /usr/local/cuda-8.0/include/nvToolsExtMeta.h:94

   type NvtxDomainDestroy is record
      null;
   end record;
   pragma Convention (C_Pass_By_Copy, NvtxDomainDestroy);  -- /usr/local/cuda-8.0/include/nvToolsExtMeta.h:95

  -- All other NVTX API functions are for naming resources. 
  -- * A generic params struct is used for all such functions,
  -- * passing all resource handles as a uint64_t.
  --  

   type NvtxNameResourceA is record
      name : Interfaces.C.Strings.chars_ptr;  -- /usr/local/cuda-8.0/include/nvToolsExtMeta.h:110
   end record;
   pragma Convention (C_Pass_By_Copy, NvtxNameResourceA);  -- /usr/local/cuda-8.0/include/nvToolsExtMeta.h:107

   type NvtxNameResourceW is record
      name : access wchar_t;  -- /usr/local/cuda-8.0/include/nvToolsExtMeta.h:116
   end record;
   pragma Convention (C_Pass_By_Copy, NvtxNameResourceW);  -- /usr/local/cuda-8.0/include/nvToolsExtMeta.h:113

   type NvtxDomainNameResourceA is record
      core : aliased NvtxNameResourceA;  -- /usr/local/cuda-8.0/include/nvToolsExtMeta.h:119
   end record;
   pragma Convention (C_Pass_By_Copy, NvtxDomainNameResourceA);  -- /usr/local/cuda-8.0/include/nvToolsExtMeta.h:119

   type NvtxDomainNameResourceW is record
      core : aliased NvtxNameResourceW;  -- /usr/local/cuda-8.0/include/nvToolsExtMeta.h:121
   end record;
   pragma Convention (C_Pass_By_Copy, NvtxDomainNameResourceW);  -- /usr/local/cuda-8.0/include/nvToolsExtMeta.h:121

end nvToolsExtMeta_h;

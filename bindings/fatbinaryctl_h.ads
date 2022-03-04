pragma Ada_2005;
pragma Style_Checks (Off);

with Interfaces.C; use Interfaces.C;
with Interfaces.C.Strings;
with System;
with stddef_h;
with Interfaces.C.Extensions;

package fatBinaryCtl_h is

   FBCTL_SET_BINARY : constant := 1;  --  /usr/local/cuda-8.0/include/fatBinaryCtl.h:62
   FBCTL_SET_TARGETSM : constant := 2;  --  /usr/local/cuda-8.0/include/fatBinaryCtl.h:63
   FBCTL_SET_FLAGS : constant := 3;  --  /usr/local/cuda-8.0/include/fatBinaryCtl.h:64
   FBCTL_SET_CMDOPTIONS : constant := 4;  --  /usr/local/cuda-8.0/include/fatBinaryCtl.h:65
   FBCTL_SET_POLICY : constant := 5;  --  /usr/local/cuda-8.0/include/fatBinaryCtl.h:66

   FBCTL_GET_CANDIDATE : constant := 10;  --  /usr/local/cuda-8.0/include/fatBinaryCtl.h:68

   FBCTL_GET_IDENTIFIER : constant := 11;  --  /usr/local/cuda-8.0/include/fatBinaryCtl.h:71
   FBCTL_HAS_DEBUG : constant := 12;  --  /usr/local/cuda-8.0/include/fatBinaryCtl.h:72
   FBCTL_GET_PTXAS_OPTIONS : constant := 13;  --  /usr/local/cuda-8.0/include/fatBinaryCtl.h:73

   FATBINC_MAGIC : constant := 16#466243B1#;  --  /usr/local/cuda-8.0/include/fatBinaryCtl.h:100
   FATBINC_VERSION : constant := 1;  --  /usr/local/cuda-8.0/include/fatBinaryCtl.h:101
   FATBINC_LINK_VERSION : constant := 2;  --  /usr/local/cuda-8.0/include/fatBinaryCtl.h:102

   FATBIN_CONTROL_SECTION_NAME : aliased constant String := ".nvFatBinSegment" & ASCII.NUL;  --  /usr/local/cuda-8.0/include/fatBinaryCtl.h:122

   FATBIN_DATA_SECTION_NAME : aliased constant String := ".nv_fatbin" & ASCII.NUL;  --  /usr/local/cuda-8.0/include/fatBinaryCtl.h:127

   FATBIN_PRELINK_DATA_SECTION_NAME : aliased constant String := "__nv_relfatbin" & ASCII.NUL;  --  /usr/local/cuda-8.0/include/fatBinaryCtl.h:130

  -- *  Copyright 2010-2016 NVIDIA Corporation.  All rights reserved.
  -- *
  -- *  NOTICE TO USER: The source code, and related code and software
  -- *  ("Code"), is copyrighted under U.S. and international laws.
  -- *
  -- *  NVIDIA Corporation owns the copyright and any patents issued or
  -- *  pending for the Code.
  -- *
  -- *  NVIDIA CORPORATION MAKES NO REPRESENTATION ABOUT THE SUITABILITY
  -- *  OF THIS CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS-IS" WITHOUT EXPRESS
  -- *  OR IMPLIED WARRANTY OF ANY KIND.  NVIDIA CORPORATION DISCLAIMS ALL
  -- *  WARRANTIES WITH REGARD TO THE CODE, INCLUDING NON-INFRINGEMENT, AND
  -- *  ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
  -- *  PURPOSE.  IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
  -- *  DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES
  -- *  WHATSOEVER ARISING OUT OF OR IN ANY WAY RELATED TO THE USE OR
  -- *  PERFORMANCE OF THE CODE, INCLUDING, BUT NOT LIMITED TO, INFRINGEMENT,
  -- *  LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT,
  -- *  NEGLIGENCE OR OTHER TORTIOUS ACTION, AND WHETHER OR NOT THE
  -- *  POSSIBILITY OF SUCH DAMAGES WERE KNOWN OR MADE KNOWN TO NVIDIA
  -- *  CORPORATION.
  -- *
  --  

  -- for size_t  
  -- 
  -- * These are routines for controlling the fat binary.
  -- * A void* object is used, with ioctl-style calls to set and get info from it.
  --  

  -- null pointer  
  -- unrecognized kind  
  -- no candidate found  
  -- no candidate found  
  -- unexpected internal error  
   type fatBinaryCtlError_t is 
     (FBCTL_ERROR_NONE,
      FBCTL_ERROR_NULL,
      FBCTL_ERROR_UNRECOGNIZED,
      FBCTL_ERROR_NO_CANDIDATE,
      FBCTL_ERROR_COMPILE_FAILED,
      FBCTL_ERROR_INTERNAL);
   pragma Convention (C, fatBinaryCtlError_t);  -- /usr/local/cuda-8.0/include/fatBinaryCtl.h:51

   function fatBinaryCtl_Errmsg (e : fatBinaryCtlError_t) return Interfaces.C.Strings.chars_ptr;  -- /usr/local/cuda-8.0/include/fatBinaryCtl.h:52
   pragma Import (C, fatBinaryCtl_Errmsg, "fatBinaryCtl_Errmsg");

   function fatBinaryCtl_Create (data : System.Address) return fatBinaryCtlError_t;  -- /usr/local/cuda-8.0/include/fatBinaryCtl.h:54
   pragma Import (C, fatBinaryCtl_Create, "fatBinaryCtl_Create");

   procedure fatBinaryCtl_Delete (data : System.Address);  -- /usr/local/cuda-8.0/include/fatBinaryCtl.h:56
   pragma Import (C, fatBinaryCtl_Delete, "fatBinaryCtl_Delete");

  -- use this control-call to set and get values  
   function fatBinaryCtl (data : System.Address; request : int  -- , ...
      ) return fatBinaryCtlError_t;  -- /usr/local/cuda-8.0/include/fatBinaryCtl.h:59
   pragma Import (C, fatBinaryCtl, "fatBinaryCtl");

  -- defined requests  
  -- get calls return value in arg, thus are all by reference  
  -- default  
  -- use sass if possible for compile-time savings  
  -- use ptx (mainly for testing)  
  -- use ptx if arch doesn't match  
   type fatBinary_CompilationPolicy is 
     (fatBinary_PreferBestCode,
      fatBinary_AvoidPTX,
      fatBinary_ForcePTX,
      fatBinary_JITIfNotMatch);
   pragma Convention (C, fatBinary_CompilationPolicy);  -- /usr/local/cuda-8.0/include/fatBinaryCtl.h:80

  -- 
  -- * Using the input values, pick the best candidate;
  -- * use subsequent Ctl requests to get info about that candidate.
  --  

   function fatBinaryCtl_PickCandidate (data : System.Address) return fatBinaryCtlError_t;  -- /usr/local/cuda-8.0/include/fatBinaryCtl.h:86
   pragma Import (C, fatBinaryCtl_PickCandidate, "fatBinaryCtl_PickCandidate");

  -- 
  -- * Using the previously chosen candidate, compile the code to elf,
  -- * returning elf image and size.
  -- * Note that because elf is allocated inside fatBinaryCtl, 
  -- * it will be freed when _Delete routine is called.
  --  

   function fatBinaryCtl_Compile
     (data : System.Address;
      elf : System.Address;
      esize : access stddef_h.size_t) return fatBinaryCtlError_t;  -- /usr/local/cuda-8.0/include/fatBinaryCtl.h:94
   pragma Import (C, fatBinaryCtl_Compile, "fatBinaryCtl_Compile");

  -- * These defines are for the fatbin.c runtime wrapper
  --  

   type uu_fatBinC_Wrapper_t is record
      magic : aliased int;  -- /usr/local/cuda-8.0/include/fatBinaryCtl.h:105
      version : aliased int;  -- /usr/local/cuda-8.0/include/fatBinaryCtl.h:106
      data : access Extensions.unsigned_long_long;  -- /usr/local/cuda-8.0/include/fatBinaryCtl.h:107
      filename_or_fatbins : System.Address;  -- /usr/local/cuda-8.0/include/fatBinaryCtl.h:108
   end record;
   pragma Convention (C_Pass_By_Copy, uu_fatBinC_Wrapper_t);  -- /usr/local/cuda-8.0/include/fatBinaryCtl.h:110

   --  skipped anonymous struct anon_3

  -- version 1: offline filename,
  --                               * version 2: array of prelinked fatbins  

  -- * The section that contains the fatbin control structure
  --  

  -- mach-o sections limited to 15 chars, and want __ prefix else strip complains, * so use a different name  
  -- only need segment name for mach-o  
  -- * The section that contains the fatbin data itself
  -- * (put in separate section so easy to find)
  --  

  -- section for pre-linked relocatable fatbin data  
end fatBinaryCtl_h;

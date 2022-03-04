pragma Ada_2005;
pragma Style_Checks (Off);

with Interfaces.C; use Interfaces.C;
with System;
with Interfaces.C.Extensions;

package fatbinary_h is

   FATBIN_MAGIC : constant := 16#BA55ED50#;  --  /usr/local/cuda-8.0/include/fatbinary.h:61
   OLD_STYLE_FATBIN_MAGIC : constant := 16#1EE55A01#;  --  /usr/local/cuda-8.0/include/fatbinary.h:62

   FATBIN_VERSION : constant := 16#0001#;  --  /usr/local/cuda-8.0/include/fatbinary.h:64

  -- *  Copyright 2010 NVIDIA Corporation.  All rights reserved.
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

  -- 
  -- * This is the fat binary header structure. 
  -- * Because all layout information is contained in all the structures, 
  -- * it is both forward and backward compatible. 
  -- * A new driver can interpret an old binary 
  -- * as it will not address fields that are present in the current version. 
  -- * An old driver can, for minor version differences, 
  -- * still interpret a new binary, 
  -- * as the new features in the binary will be ignored by the driver.
  -- *
  -- * This is the top level type for the binary format. 
  -- * It points to a fatBinaryHeader structure. 
  -- * It is followed by a number of code binaries.
  -- * The structures must be 8-byte aligned, 
  -- * and are the same on both 32bit and 64bit platforms.
  -- *
  -- * The details of the format for the binaries that follow the header
  -- * are in a separate internal header.
  --  

   type computeFatBinaryFormat_t is new System.Address;  -- /usr/local/cuda-8.0/include/fatbinary.h:49

  -- ensure 8-byte alignment  
  -- Magic numbers  
  -- * This is the fat binary header structure. 
  -- * The 'magic' field holds the magic number. 
  -- * A magic of OLD_STYLE_FATBIN_MAGIC indicates an old style fat binary. 
  -- * Because old style binaries are in little endian, we can just read 
  -- * the magic in a 32 bit container for both 32 and 64 bit platforms. 
  -- * The 'version' fields holds the fatbin version.
  -- * It should be the goal to never bump this version. 
  -- * The headerSize holds the size of the header (must be multiple of 8).
  -- * The 'fatSize' fields holds the size of the entire fat binary, 
  -- * excluding this header. It must be a multiple of 8.
  --  

   type fatBinaryHeader is record
      magic : aliased unsigned;  -- /usr/local/cuda-8.0/include/fatbinary.h:80
      version : aliased unsigned_short;  -- /usr/local/cuda-8.0/include/fatbinary.h:81
      headerSize : aliased unsigned_short;  -- /usr/local/cuda-8.0/include/fatbinary.h:82
      fatSize : aliased Extensions.unsigned_long_long;  -- /usr/local/cuda-8.0/include/fatbinary.h:83
   end record;
   pragma Convention (C_Pass_By_Copy, fatBinaryHeader);  -- /usr/local/cuda-8.0/include/fatbinary.h:78

  -- Code kinds supported by the driver  
   subtype fatBinaryCodeKind is unsigned;
   FATBIN_KIND_PTX : constant fatBinaryCodeKind := 1;
   FATBIN_KIND_ELF : constant fatBinaryCodeKind := 2;
   FATBIN_KIND_OLDCUBIN : constant fatBinaryCodeKind := 4;  -- /usr/local/cuda-8.0/include/fatbinary.h:91

end fatbinary_h;

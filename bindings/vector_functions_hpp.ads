pragma Ada_2005;
pragma Style_Checks (Off);

with Interfaces.C; use Interfaces.C;
with vector_types_h;
with Interfaces.C.Extensions;

package vector_functions_hpp is

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

  --******************************************************************************
  --*                                                                              *
  --*                                                                              *
  --*                                                                              *
  --****************************************************************************** 

  --******************************************************************************
  --*                                                                              *
  --*                                                                              *
  --*                                                                              *
  --****************************************************************************** 

   function make_char1 (x : signed_char) return vector_types_h.char1;  -- /usr/local/cuda-8.0/include/vector_functions.hpp:75
   pragma Import (CPP, make_char1, "_ZL10make_char1a");

   function make_uchar1 (x : unsigned_char) return vector_types_h.uchar1;  -- /usr/local/cuda-8.0/include/vector_functions.hpp:80
   pragma Import (CPP, make_uchar1, "_ZL11make_uchar1h");

   function make_char2 (x : signed_char; y : signed_char) return vector_types_h.char2;  -- /usr/local/cuda-8.0/include/vector_functions.hpp:85
   pragma Import (CPP, make_char2, "_ZL10make_char2aa");

   function make_uchar2 (x : unsigned_char; y : unsigned_char) return vector_types_h.uchar2;  -- /usr/local/cuda-8.0/include/vector_functions.hpp:90
   pragma Import (CPP, make_uchar2, "_ZL11make_uchar2hh");

   function make_char3
     (x : signed_char;
      y : signed_char;
      z : signed_char) return vector_types_h.char3;  -- /usr/local/cuda-8.0/include/vector_functions.hpp:95
   pragma Import (CPP, make_char3, "_ZL10make_char3aaa");

   function make_uchar3
     (x : unsigned_char;
      y : unsigned_char;
      z : unsigned_char) return vector_types_h.uchar3;  -- /usr/local/cuda-8.0/include/vector_functions.hpp:100
   pragma Import (CPP, make_uchar3, "_ZL11make_uchar3hhh");

   function make_char4
     (x : signed_char;
      y : signed_char;
      z : signed_char;
      w : signed_char) return vector_types_h.char4;  -- /usr/local/cuda-8.0/include/vector_functions.hpp:105
   pragma Import (CPP, make_char4, "_ZL10make_char4aaaa");

   function make_uchar4
     (x : unsigned_char;
      y : unsigned_char;
      z : unsigned_char;
      w : unsigned_char) return vector_types_h.uchar4;  -- /usr/local/cuda-8.0/include/vector_functions.hpp:110
   pragma Import (CPP, make_uchar4, "_ZL11make_uchar4hhhh");

   function make_short1 (x : short) return vector_types_h.short1;  -- /usr/local/cuda-8.0/include/vector_functions.hpp:115
   pragma Import (CPP, make_short1, "_ZL11make_short1s");

   function make_ushort1 (x : unsigned_short) return vector_types_h.ushort1;  -- /usr/local/cuda-8.0/include/vector_functions.hpp:120
   pragma Import (CPP, make_ushort1, "_ZL12make_ushort1t");

   function make_short2 (x : short; y : short) return vector_types_h.short2;  -- /usr/local/cuda-8.0/include/vector_functions.hpp:125
   pragma Import (CPP, make_short2, "_ZL11make_short2ss");

   function make_ushort2 (x : unsigned_short; y : unsigned_short) return vector_types_h.ushort2;  -- /usr/local/cuda-8.0/include/vector_functions.hpp:130
   pragma Import (CPP, make_ushort2, "_ZL12make_ushort2tt");

   function make_short3
     (x : short;
      y : short;
      z : short) return vector_types_h.short3;  -- /usr/local/cuda-8.0/include/vector_functions.hpp:135
   pragma Import (CPP, make_short3, "_ZL11make_short3sss");

   function make_ushort3
     (x : unsigned_short;
      y : unsigned_short;
      z : unsigned_short) return vector_types_h.ushort3;  -- /usr/local/cuda-8.0/include/vector_functions.hpp:140
   pragma Import (CPP, make_ushort3, "_ZL12make_ushort3ttt");

   function make_short4
     (x : short;
      y : short;
      z : short;
      w : short) return vector_types_h.short4;  -- /usr/local/cuda-8.0/include/vector_functions.hpp:145
   pragma Import (CPP, make_short4, "_ZL11make_short4ssss");

   function make_ushort4
     (x : unsigned_short;
      y : unsigned_short;
      z : unsigned_short;
      w : unsigned_short) return vector_types_h.ushort4;  -- /usr/local/cuda-8.0/include/vector_functions.hpp:150
   pragma Import (CPP, make_ushort4, "_ZL12make_ushort4tttt");

   function make_int1 (x : int) return vector_types_h.int1;  -- /usr/local/cuda-8.0/include/vector_functions.hpp:155
   pragma Import (CPP, make_int1, "_ZL9make_int1i");

   function make_uint1 (x : unsigned) return vector_types_h.uint1;  -- /usr/local/cuda-8.0/include/vector_functions.hpp:160
   pragma Import (CPP, make_uint1, "_ZL10make_uint1j");

   function make_int2 (x : int; y : int) return vector_types_h.int2;  -- /usr/local/cuda-8.0/include/vector_functions.hpp:165
   pragma Import (CPP, make_int2, "_ZL9make_int2ii");

   function make_uint2 (x : unsigned; y : unsigned) return vector_types_h.uint2;  -- /usr/local/cuda-8.0/include/vector_functions.hpp:170
   pragma Import (CPP, make_uint2, "_ZL10make_uint2jj");

   function make_int3
     (x : int;
      y : int;
      z : int) return vector_types_h.int3;  -- /usr/local/cuda-8.0/include/vector_functions.hpp:175
   pragma Import (CPP, make_int3, "_ZL9make_int3iii");

   function make_uint3
     (x : unsigned;
      y : unsigned;
      z : unsigned) return vector_types_h.uint3;  -- /usr/local/cuda-8.0/include/vector_functions.hpp:180
   pragma Import (CPP, make_uint3, "_ZL10make_uint3jjj");

   function make_int4
     (x : int;
      y : int;
      z : int;
      w : int) return vector_types_h.int4;  -- /usr/local/cuda-8.0/include/vector_functions.hpp:185
   pragma Import (CPP, make_int4, "_ZL9make_int4iiii");

   function make_uint4
     (x : unsigned;
      y : unsigned;
      z : unsigned;
      w : unsigned) return vector_types_h.uint4;  -- /usr/local/cuda-8.0/include/vector_functions.hpp:190
   pragma Import (CPP, make_uint4, "_ZL10make_uint4jjjj");

   function make_long1 (x : long) return vector_types_h.long1;  -- /usr/local/cuda-8.0/include/vector_functions.hpp:195
   pragma Import (CPP, make_long1, "_ZL10make_long1l");

   function make_ulong1 (x : unsigned_long) return vector_types_h.ulong1;  -- /usr/local/cuda-8.0/include/vector_functions.hpp:200
   pragma Import (CPP, make_ulong1, "_ZL11make_ulong1m");

   function make_long2 (x : long; y : long) return vector_types_h.long2;  -- /usr/local/cuda-8.0/include/vector_functions.hpp:205
   pragma Import (CPP, make_long2, "_ZL10make_long2ll");

   function make_ulong2 (x : unsigned_long; y : unsigned_long) return vector_types_h.ulong2;  -- /usr/local/cuda-8.0/include/vector_functions.hpp:210
   pragma Import (CPP, make_ulong2, "_ZL11make_ulong2mm");

   function make_long3
     (x : long;
      y : long;
      z : long) return vector_types_h.long3;  -- /usr/local/cuda-8.0/include/vector_functions.hpp:215
   pragma Import (CPP, make_long3, "_ZL10make_long3lll");

   function make_ulong3
     (x : unsigned_long;
      y : unsigned_long;
      z : unsigned_long) return vector_types_h.ulong3;  -- /usr/local/cuda-8.0/include/vector_functions.hpp:220
   pragma Import (CPP, make_ulong3, "_ZL11make_ulong3mmm");

   function make_long4
     (x : long;
      y : long;
      z : long;
      w : long) return vector_types_h.long4;  -- /usr/local/cuda-8.0/include/vector_functions.hpp:225
   pragma Import (CPP, make_long4, "_ZL10make_long4llll");

   function make_ulong4
     (x : unsigned_long;
      y : unsigned_long;
      z : unsigned_long;
      w : unsigned_long) return vector_types_h.ulong4;  -- /usr/local/cuda-8.0/include/vector_functions.hpp:230
   pragma Import (CPP, make_ulong4, "_ZL11make_ulong4mmmm");

   function make_float1 (x : float) return vector_types_h.float1;  -- /usr/local/cuda-8.0/include/vector_functions.hpp:235
   pragma Import (CPP, make_float1, "_ZL11make_float1f");

   function make_float2 (x : float; y : float) return vector_types_h.float2;  -- /usr/local/cuda-8.0/include/vector_functions.hpp:240
   pragma Import (CPP, make_float2, "_ZL11make_float2ff");

   function make_float3
     (x : float;
      y : float;
      z : float) return vector_types_h.float3;  -- /usr/local/cuda-8.0/include/vector_functions.hpp:245
   pragma Import (CPP, make_float3, "_ZL11make_float3fff");

   function make_float4
     (x : float;
      y : float;
      z : float;
      w : float) return vector_types_h.float4;  -- /usr/local/cuda-8.0/include/vector_functions.hpp:250
   pragma Import (CPP, make_float4, "_ZL11make_float4ffff");

   function make_longlong1 (x : Long_Long_Integer) return vector_types_h.longlong1;  -- /usr/local/cuda-8.0/include/vector_functions.hpp:255
   pragma Import (CPP, make_longlong1, "_ZL14make_longlong1x");

   function make_ulonglong1 (x : Extensions.unsigned_long_long) return vector_types_h.ulonglong1;  -- /usr/local/cuda-8.0/include/vector_functions.hpp:260
   pragma Import (CPP, make_ulonglong1, "_ZL15make_ulonglong1y");

   function make_longlong2 (x : Long_Long_Integer; y : Long_Long_Integer) return vector_types_h.longlong2;  -- /usr/local/cuda-8.0/include/vector_functions.hpp:265
   pragma Import (CPP, make_longlong2, "_ZL14make_longlong2xx");

   function make_ulonglong2 (x : Extensions.unsigned_long_long; y : Extensions.unsigned_long_long) return vector_types_h.ulonglong2;  -- /usr/local/cuda-8.0/include/vector_functions.hpp:270
   pragma Import (CPP, make_ulonglong2, "_ZL15make_ulonglong2yy");

   function make_longlong3
     (x : Long_Long_Integer;
      y : Long_Long_Integer;
      z : Long_Long_Integer) return vector_types_h.longlong3;  -- /usr/local/cuda-8.0/include/vector_functions.hpp:275
   pragma Import (CPP, make_longlong3, "_ZL14make_longlong3xxx");

   function make_ulonglong3
     (x : Extensions.unsigned_long_long;
      y : Extensions.unsigned_long_long;
      z : Extensions.unsigned_long_long) return vector_types_h.ulonglong3;  -- /usr/local/cuda-8.0/include/vector_functions.hpp:280
   pragma Import (CPP, make_ulonglong3, "_ZL15make_ulonglong3yyy");

   function make_longlong4
     (x : Long_Long_Integer;
      y : Long_Long_Integer;
      z : Long_Long_Integer;
      w : Long_Long_Integer) return vector_types_h.longlong4;  -- /usr/local/cuda-8.0/include/vector_functions.hpp:285
   pragma Import (CPP, make_longlong4, "_ZL14make_longlong4xxxx");

   function make_ulonglong4
     (x : Extensions.unsigned_long_long;
      y : Extensions.unsigned_long_long;
      z : Extensions.unsigned_long_long;
      w : Extensions.unsigned_long_long) return vector_types_h.ulonglong4;  -- /usr/local/cuda-8.0/include/vector_functions.hpp:290
   pragma Import (CPP, make_ulonglong4, "_ZL15make_ulonglong4yyyy");

   function make_double1 (x : double) return vector_types_h.double1;  -- /usr/local/cuda-8.0/include/vector_functions.hpp:295
   pragma Import (CPP, make_double1, "_ZL12make_double1d");

   function make_double2 (x : double; y : double) return vector_types_h.double2;  -- /usr/local/cuda-8.0/include/vector_functions.hpp:300
   pragma Import (CPP, make_double2, "_ZL12make_double2dd");

   function make_double3
     (x : double;
      y : double;
      z : double) return vector_types_h.double3;  -- /usr/local/cuda-8.0/include/vector_functions.hpp:305
   pragma Import (CPP, make_double3, "_ZL12make_double3ddd");

   function make_double4
     (x : double;
      y : double;
      z : double;
      w : double) return vector_types_h.double4;  -- /usr/local/cuda-8.0/include/vector_functions.hpp:310
   pragma Import (CPP, make_double4, "_ZL12make_double4dddd");

end vector_functions_hpp;

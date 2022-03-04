pragma Ada_2005;
pragma Style_Checks (Off);

with Interfaces.C; use Interfaces.C;
with Interfaces.C.Extensions;

package vector_types_h is

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

   type char1 is record
      x : aliased signed_char;  -- /usr/local/cuda-8.0/include/vector_types.h:100
   end record;
   pragma Convention (C_Pass_By_Copy, char1);  -- /usr/local/cuda-8.0/include/vector_types.h:98

   type uchar1 is record
      x : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/vector_types.h:105
   end record;
   pragma Convention (C_Pass_By_Copy, uchar1);  -- /usr/local/cuda-8.0/include/vector_types.h:103

   type char2 is record
      x : aliased signed_char;  -- /usr/local/cuda-8.0/include/vector_types.h:111
      y : aliased signed_char;  -- /usr/local/cuda-8.0/include/vector_types.h:111
   end record;
   pragma Convention (C_Pass_By_Copy, char2);  -- /usr/local/cuda-8.0/include/vector_types.h:109

   type uchar2 is record
      x : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/vector_types.h:116
      y : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/vector_types.h:116
   end record;
   pragma Convention (C_Pass_By_Copy, uchar2);  -- /usr/local/cuda-8.0/include/vector_types.h:114

   type char3 is record
      x : aliased signed_char;  -- /usr/local/cuda-8.0/include/vector_types.h:121
      y : aliased signed_char;  -- /usr/local/cuda-8.0/include/vector_types.h:121
      z : aliased signed_char;  -- /usr/local/cuda-8.0/include/vector_types.h:121
   end record;
   pragma Convention (C_Pass_By_Copy, char3);  -- /usr/local/cuda-8.0/include/vector_types.h:119

   type uchar3 is record
      x : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/vector_types.h:126
      y : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/vector_types.h:126
      z : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/vector_types.h:126
   end record;
   pragma Convention (C_Pass_By_Copy, uchar3);  -- /usr/local/cuda-8.0/include/vector_types.h:124

   type char4 is record
      x : aliased signed_char;  -- /usr/local/cuda-8.0/include/vector_types.h:131
      y : aliased signed_char;  -- /usr/local/cuda-8.0/include/vector_types.h:131
      z : aliased signed_char;  -- /usr/local/cuda-8.0/include/vector_types.h:131
      w : aliased signed_char;  -- /usr/local/cuda-8.0/include/vector_types.h:131
   end record;
   pragma Convention (C_Pass_By_Copy, char4);  -- /usr/local/cuda-8.0/include/vector_types.h:129

   type uchar4 is record
      x : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/vector_types.h:136
      y : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/vector_types.h:136
      z : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/vector_types.h:136
      w : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/vector_types.h:136
   end record;
   pragma Convention (C_Pass_By_Copy, uchar4);  -- /usr/local/cuda-8.0/include/vector_types.h:134

   type short1 is record
      x : aliased short;  -- /usr/local/cuda-8.0/include/vector_types.h:141
   end record;
   pragma Convention (C_Pass_By_Copy, short1);  -- /usr/local/cuda-8.0/include/vector_types.h:139

   type ushort1 is record
      x : aliased unsigned_short;  -- /usr/local/cuda-8.0/include/vector_types.h:146
   end record;
   pragma Convention (C_Pass_By_Copy, ushort1);  -- /usr/local/cuda-8.0/include/vector_types.h:144

   type short2 is record
      x : aliased short;  -- /usr/local/cuda-8.0/include/vector_types.h:151
      y : aliased short;  -- /usr/local/cuda-8.0/include/vector_types.h:151
   end record;
   pragma Convention (C_Pass_By_Copy, short2);  -- /usr/local/cuda-8.0/include/vector_types.h:149

   type ushort2 is record
      x : aliased unsigned_short;  -- /usr/local/cuda-8.0/include/vector_types.h:156
      y : aliased unsigned_short;  -- /usr/local/cuda-8.0/include/vector_types.h:156
   end record;
   pragma Convention (C_Pass_By_Copy, ushort2);  -- /usr/local/cuda-8.0/include/vector_types.h:154

   type short3 is record
      x : aliased short;  -- /usr/local/cuda-8.0/include/vector_types.h:161
      y : aliased short;  -- /usr/local/cuda-8.0/include/vector_types.h:161
      z : aliased short;  -- /usr/local/cuda-8.0/include/vector_types.h:161
   end record;
   pragma Convention (C_Pass_By_Copy, short3);  -- /usr/local/cuda-8.0/include/vector_types.h:159

   type ushort3 is record
      x : aliased unsigned_short;  -- /usr/local/cuda-8.0/include/vector_types.h:166
      y : aliased unsigned_short;  -- /usr/local/cuda-8.0/include/vector_types.h:166
      z : aliased unsigned_short;  -- /usr/local/cuda-8.0/include/vector_types.h:166
   end record;
   pragma Convention (C_Pass_By_Copy, ushort3);  -- /usr/local/cuda-8.0/include/vector_types.h:164

   type short4 is record
      x : aliased short;  -- /usr/local/cuda-8.0/include/vector_types.h:169
      y : aliased short;  -- /usr/local/cuda-8.0/include/vector_types.h:169
      z : aliased short;  -- /usr/local/cuda-8.0/include/vector_types.h:169
      w : aliased short;  -- /usr/local/cuda-8.0/include/vector_types.h:169
   end record;
   pragma Convention (C_Pass_By_Copy, short4);  -- /usr/local/cuda-8.0/include/vector_types.h:169

   type ushort4 is record
      x : aliased unsigned_short;  -- /usr/local/cuda-8.0/include/vector_types.h:170
      y : aliased unsigned_short;  -- /usr/local/cuda-8.0/include/vector_types.h:170
      z : aliased unsigned_short;  -- /usr/local/cuda-8.0/include/vector_types.h:170
      w : aliased unsigned_short;  -- /usr/local/cuda-8.0/include/vector_types.h:170
   end record;
   pragma Convention (C_Pass_By_Copy, ushort4);  -- /usr/local/cuda-8.0/include/vector_types.h:170

   type int1 is record
      x : aliased int;  -- /usr/local/cuda-8.0/include/vector_types.h:174
   end record;
   pragma Convention (C_Pass_By_Copy, int1);  -- /usr/local/cuda-8.0/include/vector_types.h:172

   type uint1 is record
      x : aliased unsigned;  -- /usr/local/cuda-8.0/include/vector_types.h:179
   end record;
   pragma Convention (C_Pass_By_Copy, uint1);  -- /usr/local/cuda-8.0/include/vector_types.h:177

   type int2 is record
      x : aliased int;  -- /usr/local/cuda-8.0/include/vector_types.h:182
      y : aliased int;  -- /usr/local/cuda-8.0/include/vector_types.h:182
   end record;
   pragma Convention (C_Pass_By_Copy, int2);  -- /usr/local/cuda-8.0/include/vector_types.h:182

   type uint2 is record
      x : aliased unsigned;  -- /usr/local/cuda-8.0/include/vector_types.h:183
      y : aliased unsigned;  -- /usr/local/cuda-8.0/include/vector_types.h:183
   end record;
   pragma Convention (C_Pass_By_Copy, uint2);  -- /usr/local/cuda-8.0/include/vector_types.h:183

   type int3 is record
      x : aliased int;  -- /usr/local/cuda-8.0/include/vector_types.h:187
      y : aliased int;  -- /usr/local/cuda-8.0/include/vector_types.h:187
      z : aliased int;  -- /usr/local/cuda-8.0/include/vector_types.h:187
   end record;
   pragma Convention (C_Pass_By_Copy, int3);  -- /usr/local/cuda-8.0/include/vector_types.h:185

   type uint3 is record
      x : aliased unsigned;  -- /usr/local/cuda-8.0/include/vector_types.h:192
      y : aliased unsigned;  -- /usr/local/cuda-8.0/include/vector_types.h:192
      z : aliased unsigned;  -- /usr/local/cuda-8.0/include/vector_types.h:192
   end record;
   pragma Convention (C_Pass_By_Copy, uint3);  -- /usr/local/cuda-8.0/include/vector_types.h:190

   type int4 is record
      x : aliased int;  -- /usr/local/cuda-8.0/include/vector_types.h:197
      y : aliased int;  -- /usr/local/cuda-8.0/include/vector_types.h:197
      z : aliased int;  -- /usr/local/cuda-8.0/include/vector_types.h:197
      w : aliased int;  -- /usr/local/cuda-8.0/include/vector_types.h:197
   end record;
   pragma Convention (C_Pass_By_Copy, int4);  -- /usr/local/cuda-8.0/include/vector_types.h:195

   type uint4 is record
      x : aliased unsigned;  -- /usr/local/cuda-8.0/include/vector_types.h:202
      y : aliased unsigned;  -- /usr/local/cuda-8.0/include/vector_types.h:202
      z : aliased unsigned;  -- /usr/local/cuda-8.0/include/vector_types.h:202
      w : aliased unsigned;  -- /usr/local/cuda-8.0/include/vector_types.h:202
   end record;
   pragma Convention (C_Pass_By_Copy, uint4);  -- /usr/local/cuda-8.0/include/vector_types.h:200

   type long1 is record
      x : aliased long;  -- /usr/local/cuda-8.0/include/vector_types.h:207
   end record;
   pragma Convention (C_Pass_By_Copy, long1);  -- /usr/local/cuda-8.0/include/vector_types.h:205

   type ulong1 is record
      x : aliased unsigned_long;  -- /usr/local/cuda-8.0/include/vector_types.h:212
   end record;
   pragma Convention (C_Pass_By_Copy, ulong1);  -- /usr/local/cuda-8.0/include/vector_types.h:210

   type long2 is record
      x : aliased long;  -- /usr/local/cuda-8.0/include/vector_types.h:222
      y : aliased long;  -- /usr/local/cuda-8.0/include/vector_types.h:222
   end record;
   pragma Convention (C_Pass_By_Copy, long2);  -- /usr/local/cuda-8.0/include/vector_types.h:220

   type ulong2 is record
      x : aliased unsigned_long;  -- /usr/local/cuda-8.0/include/vector_types.h:227
      y : aliased unsigned_long;  -- /usr/local/cuda-8.0/include/vector_types.h:227
   end record;
   pragma Convention (C_Pass_By_Copy, ulong2);  -- /usr/local/cuda-8.0/include/vector_types.h:225

   type long3 is record
      x : aliased long;  -- /usr/local/cuda-8.0/include/vector_types.h:234
      y : aliased long;  -- /usr/local/cuda-8.0/include/vector_types.h:234
      z : aliased long;  -- /usr/local/cuda-8.0/include/vector_types.h:234
   end record;
   pragma Convention (C_Pass_By_Copy, long3);  -- /usr/local/cuda-8.0/include/vector_types.h:232

   type ulong3 is record
      x : aliased unsigned_long;  -- /usr/local/cuda-8.0/include/vector_types.h:239
      y : aliased unsigned_long;  -- /usr/local/cuda-8.0/include/vector_types.h:239
      z : aliased unsigned_long;  -- /usr/local/cuda-8.0/include/vector_types.h:239
   end record;
   pragma Convention (C_Pass_By_Copy, ulong3);  -- /usr/local/cuda-8.0/include/vector_types.h:237

   type long4 is record
      x : aliased long;  -- /usr/local/cuda-8.0/include/vector_types.h:244
      y : aliased long;  -- /usr/local/cuda-8.0/include/vector_types.h:244
      z : aliased long;  -- /usr/local/cuda-8.0/include/vector_types.h:244
      w : aliased long;  -- /usr/local/cuda-8.0/include/vector_types.h:244
   end record;
   pragma Convention (C_Pass_By_Copy, long4);  -- /usr/local/cuda-8.0/include/vector_types.h:242

   type ulong4 is record
      x : aliased unsigned_long;  -- /usr/local/cuda-8.0/include/vector_types.h:249
      y : aliased unsigned_long;  -- /usr/local/cuda-8.0/include/vector_types.h:249
      z : aliased unsigned_long;  -- /usr/local/cuda-8.0/include/vector_types.h:249
      w : aliased unsigned_long;  -- /usr/local/cuda-8.0/include/vector_types.h:249
   end record;
   pragma Convention (C_Pass_By_Copy, ulong4);  -- /usr/local/cuda-8.0/include/vector_types.h:247

   type float1 is record
      x : aliased float;  -- /usr/local/cuda-8.0/include/vector_types.h:254
   end record;
   pragma Convention (C_Pass_By_Copy, float1);  -- /usr/local/cuda-8.0/include/vector_types.h:252

   type float2 is record
      x : aliased float;  -- /usr/local/cuda-8.0/include/vector_types.h:274
      y : aliased float;  -- /usr/local/cuda-8.0/include/vector_types.h:274
   end record;
   pragma Convention (C_Pass_By_Copy, float2);  -- /usr/local/cuda-8.0/include/vector_types.h:274

   type float3 is record
      x : aliased float;  -- /usr/local/cuda-8.0/include/vector_types.h:281
      y : aliased float;  -- /usr/local/cuda-8.0/include/vector_types.h:281
      z : aliased float;  -- /usr/local/cuda-8.0/include/vector_types.h:281
   end record;
   pragma Convention (C_Pass_By_Copy, float3);  -- /usr/local/cuda-8.0/include/vector_types.h:279

   type float4 is record
      x : aliased float;  -- /usr/local/cuda-8.0/include/vector_types.h:286
      y : aliased float;  -- /usr/local/cuda-8.0/include/vector_types.h:286
      z : aliased float;  -- /usr/local/cuda-8.0/include/vector_types.h:286
      w : aliased float;  -- /usr/local/cuda-8.0/include/vector_types.h:286
   end record;
   pragma Convention (C_Pass_By_Copy, float4);  -- /usr/local/cuda-8.0/include/vector_types.h:284

   type longlong1 is record
      x : aliased Long_Long_Integer;  -- /usr/local/cuda-8.0/include/vector_types.h:291
   end record;
   pragma Convention (C_Pass_By_Copy, longlong1);  -- /usr/local/cuda-8.0/include/vector_types.h:289

   type ulonglong1 is record
      x : aliased Extensions.unsigned_long_long;  -- /usr/local/cuda-8.0/include/vector_types.h:296
   end record;
   pragma Convention (C_Pass_By_Copy, ulonglong1);  -- /usr/local/cuda-8.0/include/vector_types.h:294

   type longlong2 is record
      x : aliased Long_Long_Integer;  -- /usr/local/cuda-8.0/include/vector_types.h:301
      y : aliased Long_Long_Integer;  -- /usr/local/cuda-8.0/include/vector_types.h:301
   end record;
   pragma Convention (C_Pass_By_Copy, longlong2);  -- /usr/local/cuda-8.0/include/vector_types.h:299

   type ulonglong2 is record
      x : aliased Extensions.unsigned_long_long;  -- /usr/local/cuda-8.0/include/vector_types.h:306
      y : aliased Extensions.unsigned_long_long;  -- /usr/local/cuda-8.0/include/vector_types.h:306
   end record;
   pragma Convention (C_Pass_By_Copy, ulonglong2);  -- /usr/local/cuda-8.0/include/vector_types.h:304

   type longlong3 is record
      x : aliased Long_Long_Integer;  -- /usr/local/cuda-8.0/include/vector_types.h:311
      y : aliased Long_Long_Integer;  -- /usr/local/cuda-8.0/include/vector_types.h:311
      z : aliased Long_Long_Integer;  -- /usr/local/cuda-8.0/include/vector_types.h:311
   end record;
   pragma Convention (C_Pass_By_Copy, longlong3);  -- /usr/local/cuda-8.0/include/vector_types.h:309

   type ulonglong3 is record
      x : aliased Extensions.unsigned_long_long;  -- /usr/local/cuda-8.0/include/vector_types.h:316
      y : aliased Extensions.unsigned_long_long;  -- /usr/local/cuda-8.0/include/vector_types.h:316
      z : aliased Extensions.unsigned_long_long;  -- /usr/local/cuda-8.0/include/vector_types.h:316
   end record;
   pragma Convention (C_Pass_By_Copy, ulonglong3);  -- /usr/local/cuda-8.0/include/vector_types.h:314

   type longlong4 is record
      x : aliased Long_Long_Integer;  -- /usr/local/cuda-8.0/include/vector_types.h:321
      y : aliased Long_Long_Integer;  -- /usr/local/cuda-8.0/include/vector_types.h:321
      z : aliased Long_Long_Integer;  -- /usr/local/cuda-8.0/include/vector_types.h:321
      w : aliased Long_Long_Integer;  -- /usr/local/cuda-8.0/include/vector_types.h:321
   end record;
   pragma Convention (C_Pass_By_Copy, longlong4);  -- /usr/local/cuda-8.0/include/vector_types.h:319

   type ulonglong4 is record
      x : aliased Extensions.unsigned_long_long;  -- /usr/local/cuda-8.0/include/vector_types.h:326
      y : aliased Extensions.unsigned_long_long;  -- /usr/local/cuda-8.0/include/vector_types.h:326
      z : aliased Extensions.unsigned_long_long;  -- /usr/local/cuda-8.0/include/vector_types.h:326
      w : aliased Extensions.unsigned_long_long;  -- /usr/local/cuda-8.0/include/vector_types.h:326
   end record;
   pragma Convention (C_Pass_By_Copy, ulonglong4);  -- /usr/local/cuda-8.0/include/vector_types.h:324

   type double1 is record
      x : aliased double;  -- /usr/local/cuda-8.0/include/vector_types.h:331
   end record;
   pragma Convention (C_Pass_By_Copy, double1);  -- /usr/local/cuda-8.0/include/vector_types.h:329

   type double2 is record
      x : aliased double;  -- /usr/local/cuda-8.0/include/vector_types.h:336
      y : aliased double;  -- /usr/local/cuda-8.0/include/vector_types.h:336
   end record;
   pragma Convention (C_Pass_By_Copy, double2);  -- /usr/local/cuda-8.0/include/vector_types.h:334

   type double3 is record
      x : aliased double;  -- /usr/local/cuda-8.0/include/vector_types.h:341
      y : aliased double;  -- /usr/local/cuda-8.0/include/vector_types.h:341
      z : aliased double;  -- /usr/local/cuda-8.0/include/vector_types.h:341
   end record;
   pragma Convention (C_Pass_By_Copy, double3);  -- /usr/local/cuda-8.0/include/vector_types.h:339

   type double4 is record
      x : aliased double;  -- /usr/local/cuda-8.0/include/vector_types.h:346
      y : aliased double;  -- /usr/local/cuda-8.0/include/vector_types.h:346
      z : aliased double;  -- /usr/local/cuda-8.0/include/vector_types.h:346
      w : aliased double;  -- /usr/local/cuda-8.0/include/vector_types.h:346
   end record;
   pragma Convention (C_Pass_By_Copy, double4);  -- /usr/local/cuda-8.0/include/vector_types.h:344

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

   package Class_dim3 is
      type dim3 is limited record
         x : aliased unsigned;  -- /usr/local/cuda-8.0/include/vector_types.h:419
         y : aliased unsigned;  -- /usr/local/cuda-8.0/include/vector_types.h:419
         z : aliased unsigned;  -- /usr/local/cuda-8.0/include/vector_types.h:419
      end record;
      pragma Import (CPP, dim3);

      function New_dim3
        (vx : unsigned;
         vy : unsigned;
         vz : unsigned) return dim3;  -- /usr/local/cuda-8.0/include/vector_types.h:421
      pragma CPP_Constructor (New_dim3, "_ZN4dim3C1Ejjj");

      function New_dim3 (v : uint3) return dim3;  -- /usr/local/cuda-8.0/include/vector_types.h:422
      pragma CPP_Constructor (New_dim3, "_ZN4dim3C1E5uint3");
      
      function New_dim3 return dim3; 
      pragma CPP_Constructor (New_dim3, "_ZN4dim3C1E5v");

      function operator_1 (this : access dim3) return uint3;  -- /usr/local/cuda-8.0/include/vector_types.h:423
      pragma Import (CPP, operator_1, "_ZN4dim3cv5uint3Ev");
   end;
   use Class_dim3;
end vector_types_h;

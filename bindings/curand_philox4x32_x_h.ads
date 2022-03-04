pragma Ada_2005;
pragma Style_Checks (Off);

with Interfaces.C; use Interfaces.C;
with vector_types_h;
with Interfaces.C.Extensions;

package curand_philox4x32_x_h is

   --  unsupported macro: QUALIFIERS static __forceinline__ __device__
   PHILOX_W32_0 : constant := (16#9E3779B9#);  --  /usr/local/cuda-8.0/include/curand_philox4x32_x.h:87
   PHILOX_W32_1 : constant := (16#BB67AE85#);  --  /usr/local/cuda-8.0/include/curand_philox4x32_x.h:88
   PHILOX_M4x32_0 : constant := (16#D2511F53#);  --  /usr/local/cuda-8.0/include/curand_philox4x32_x.h:89
   PHILOX_M4x32_1 : constant := (16#CD9E8D57#);  --  /usr/local/cuda-8.0/include/curand_philox4x32_x.h:90

  -- Copyright 2010-2014 NVIDIA Corporation.  All rights reserved.
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

  --Copyright 2010-2011, D. E. Shaw Research.
  --All rights reserved.
  --Redistribution and use in source and binary forms, with or without
  --modification, are permitted provided that the following conditions are
  --met:
  --* Redistributions of source code must retain the above copyright
  --  notice, this list of conditions, and the following disclaimer.
  --* Redistributions in binary form must reproduce the above copyright
  --  notice, this list of conditions, and the following disclaimer in the
  --  documentation and/or other materials provided with the distribution.
  --* Neither the name of D. E. Shaw Research nor the names of its
  --  contributors may be used to endorse or promote products derived from
  --  this software without specific prior written permission.
  --THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  --"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  --LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  --A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  --OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  --SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  --LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  --DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  --THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  --(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  --OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
  -- 

   type curandStatePhilox4_32_10 is record
      ctr : aliased vector_types_h.uint4;  -- /usr/local/cuda-8.0/include/curand_philox4x32_x.h:93
      output : aliased vector_types_h.uint4;  -- /usr/local/cuda-8.0/include/curand_philox4x32_x.h:94
      key : aliased vector_types_h.uint2;  -- /usr/local/cuda-8.0/include/curand_philox4x32_x.h:95
      STATE : aliased unsigned;  -- /usr/local/cuda-8.0/include/curand_philox4x32_x.h:96
      boxmuller_flag : aliased int;  -- /usr/local/cuda-8.0/include/curand_philox4x32_x.h:97
      boxmuller_flag_double : aliased int;  -- /usr/local/cuda-8.0/include/curand_philox4x32_x.h:98
      boxmuller_extra : aliased float;  -- /usr/local/cuda-8.0/include/curand_philox4x32_x.h:99
      boxmuller_extra_double : aliased double;  -- /usr/local/cuda-8.0/include/curand_philox4x32_x.h:100
   end record;
   pragma Convention (C_Pass_By_Copy, curandStatePhilox4_32_10);  -- /usr/local/cuda-8.0/include/curand_philox4x32_x.h:92

   subtype curandStatePhilox4_32_10_t is curandStatePhilox4_32_10;

   procedure Philox_State_Incr (s : access curandStatePhilox4_32_10_t; n : Extensions.unsigned_long_long);  -- /usr/local/cuda-8.0/include/curand_philox4x32_x.h:106
   pragma Import (CPP, Philox_State_Incr, "_ZL17Philox_State_IncrP24curandStatePhilox4_32_10y");

   procedure Philox_State_Incr_hi (s : access curandStatePhilox4_32_10_t; n : Extensions.unsigned_long_long);  -- /usr/local/cuda-8.0/include/curand_philox4x32_x.h:122
   pragma Import (CPP, Philox_State_Incr_hi, "_ZL20Philox_State_Incr_hiP24curandStatePhilox4_32_10y");

   procedure Philox_State_Incr (s : access curandStatePhilox4_32_10_t);  -- /usr/local/cuda-8.0/include/curand_philox4x32_x.h:136
   pragma Import (CPP, Philox_State_Incr, "_ZL17Philox_State_IncrP24curandStatePhilox4_32_10");

   function mulhilo32
     (a : unsigned;
      b : unsigned;
      hip : access unsigned) return unsigned;  -- /usr/local/cuda-8.0/include/curand_philox4x32_x.h:145
   pragma Import (CPP, mulhilo32, "_ZL9mulhilo32jjPj");

  -- host code
  -- device code
   --  skipped func _philox4x32round

   function curand_Philox4x32_10 (c : vector_types_h.uint4; k : vector_types_h.uint2) return vector_types_h.uint4;  -- /usr/local/cuda-8.0/include/curand_philox4x32_x.h:170
   pragma Import (CPP, curand_Philox4x32_10, "_ZL20curand_Philox4x32_105uint45uint2");

  -- 1 
  -- 2
  -- 3 
  -- 4 
  -- 5 
  -- 6 
  -- 7 
  -- 8 
  -- 9 
  -- 10
end curand_philox4x32_x_h;

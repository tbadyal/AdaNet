pragma Ada_2005;
pragma Style_Checks (Off);

with Interfaces.C; use Interfaces.C;
with curand_h;

package curand_discrete_h is

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

   type curandDistributionShift_st is record
      probability : curand_h.curandDistribution_t;  -- /usr/local/cuda-8.0/include/curand_discrete.h:53
      host_probability : curand_h.curandDistribution_t;  -- /usr/local/cuda-8.0/include/curand_discrete.h:54
      shift : aliased unsigned;  -- /usr/local/cuda-8.0/include/curand_discrete.h:55
      length : aliased unsigned;  -- /usr/local/cuda-8.0/include/curand_discrete.h:56
      host_gen : aliased unsigned;  -- /usr/local/cuda-8.0/include/curand_discrete.h:57
   end record;
   pragma Convention (C_Pass_By_Copy, curandDistributionShift_st);  -- /usr/local/cuda-8.0/include/curand_discrete.h:52

   type curandHistogramM2_st is record
      V : curand_h.curandHistogramM2V_t;  -- /usr/local/cuda-8.0/include/curand_discrete.h:61
      host_V : curand_h.curandHistogramM2V_t;  -- /usr/local/cuda-8.0/include/curand_discrete.h:62
      K : curand_h.curandHistogramM2K_t;  -- /usr/local/cuda-8.0/include/curand_discrete.h:63
      host_K : curand_h.curandHistogramM2K_t;  -- /usr/local/cuda-8.0/include/curand_discrete.h:64
      host_gen : aliased unsigned;  -- /usr/local/cuda-8.0/include/curand_discrete.h:65
   end record;
   pragma Convention (C_Pass_By_Copy, curandHistogramM2_st);  -- /usr/local/cuda-8.0/include/curand_discrete.h:60

   type curandDistributionM2Shift_st is record
      histogram : curand_h.curandHistogramM2_t;  -- /usr/local/cuda-8.0/include/curand_discrete.h:70
      host_histogram : curand_h.curandHistogramM2_t;  -- /usr/local/cuda-8.0/include/curand_discrete.h:71
      shift : aliased unsigned;  -- /usr/local/cuda-8.0/include/curand_discrete.h:72
      length : aliased unsigned;  -- /usr/local/cuda-8.0/include/curand_discrete.h:73
      host_gen : aliased unsigned;  -- /usr/local/cuda-8.0/include/curand_discrete.h:74
   end record;
   pragma Convention (C_Pass_By_Copy, curandDistributionM2Shift_st);  -- /usr/local/cuda-8.0/include/curand_discrete.h:69

   type curandDiscreteDistribution_st is record
      self_host_ptr : curand_h.curandDiscreteDistribution_t;  -- /usr/local/cuda-8.0/include/curand_discrete.h:78
      M2 : curand_h.curandDistributionM2Shift_t;  -- /usr/local/cuda-8.0/include/curand_discrete.h:79
      host_M2 : curand_h.curandDistributionM2Shift_t;  -- /usr/local/cuda-8.0/include/curand_discrete.h:80
      stddev : aliased double;  -- /usr/local/cuda-8.0/include/curand_discrete.h:81
      mean : aliased double;  -- /usr/local/cuda-8.0/include/curand_discrete.h:82
      method : aliased curand_h.curandMethod_t;  -- /usr/local/cuda-8.0/include/curand_discrete.h:83
      host_gen : aliased unsigned;  -- /usr/local/cuda-8.0/include/curand_discrete.h:84
   end record;
   pragma Convention (C_Pass_By_Copy, curandDiscreteDistribution_st);  -- /usr/local/cuda-8.0/include/curand_discrete.h:77

end curand_discrete_h;

pragma Ada_2005;
pragma Style_Checks (Off);

with Interfaces.C; use Interfaces.C;
limited with curand_kernel_h;
with curand_h;
limited with curand_philox4x32_x_h;
with vector_types_h;
limited with curand_mtgp32_h;

package curand_discrete2_h is

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

  --*
  -- * \defgroup DEVICE Device API
  -- *
  -- * @{
  --  

  --Round to nearest
  --Round to nearest
  --Round to nearest
  --Round to nearest
  --Round to nearest
  -- * \brief Return a discrete distributed unsigned int from a XORWOW generator.
  -- *
  -- * Return a single discrete distributed unsigned int derived from a
  -- * distribution defined by \p discrete_distribution from the XORWOW generator in \p state,
  -- * increment position of generator by one.
  -- *
  -- * \param state - Pointer to state to update
  -- * \param discrete_distribution - ancillary structure for discrete distribution
  -- *
  -- * \return unsigned int distributed by distribution defined by \p discrete_distribution.
  --  

   function curand_discrete (state : access curand_kernel_h.curandStateXORWOW_t; discrete_distribution : curand_h.curandDiscreteDistribution_t) return unsigned;  -- /usr/local/cuda-8.0/include/curand_discrete2.h:111
   pragma Import (CPP, curand_discrete, "_ZL15curand_discreteP17curandStateXORWOWP29curandDiscreteDistribution_st");

  -- * \brief Return a discrete distributed unsigned int from a Philox4_32_10 generator.
  -- *
  -- * Return a single discrete distributed unsigned int derived from a
  -- * distribution defined by \p discrete_distribution from the Philox4_32_10 generator in \p state,
  -- * increment position of generator by one.
  -- *
  -- * \param state - Pointer to state to update
  -- * \param discrete_distribution - ancillary structure for discrete distribution
  -- *
  -- * \return unsigned int distributed by distribution defined by \p discrete_distribution.
  --  

   function curand_discrete (state : access curand_philox4x32_x_h.curandStatePhilox4_32_10_t; discrete_distribution : curand_h.curandDiscreteDistribution_t) return unsigned;  -- /usr/local/cuda-8.0/include/curand_discrete2.h:128
   pragma Import (CPP, curand_discrete, "_ZL15curand_discreteP24curandStatePhilox4_32_10P29curandDiscreteDistribution_st");

  -- * \brief Return four discrete distributed unsigned ints from a Philox4_32_10 generator.
  -- *
  -- * Return four single discrete distributed unsigned ints derived from a
  -- * distribution defined by \p discrete_distribution from the Philox4_32_10 generator in \p state,
  -- * increment position of generator by one.
  -- *
  -- * \param state - Pointer to state to update
  -- * \param discrete_distribution - ancillary structure for discrete distribution
  -- *
  -- * \return unsigned int distributed by distribution defined by \p discrete_distribution.
  --  

   function curand_discrete4 (state : access curand_philox4x32_x_h.curandStatePhilox4_32_10_t; discrete_distribution : curand_h.curandDiscreteDistribution_t) return vector_types_h.uint4;  -- /usr/local/cuda-8.0/include/curand_discrete2.h:145
   pragma Import (CPP, curand_discrete4, "_ZL16curand_discrete4P24curandStatePhilox4_32_10P29curandDiscreteDistribution_st");

  -- * \brief Return a discrete distributed unsigned int from a MRG32k3a generator.
  -- *
  -- * Re turn a single discrete distributed unsigned int derived from a
  -- * distribution defined by \p discrete_distribution from the MRG32k3a generator in \p state,
  -- * increment position of generator by one.
  -- *
  -- * \param state - Pointer to state to update
  -- * \param discrete_distribution - ancillary structure for discrete distribution
  -- *
  -- * \return unsigned int distributed by distribution defined by \p discrete_distribution.
  --  

   function curand_discrete (state : access curand_kernel_h.curandStateMRG32k3a_t; discrete_distribution : curand_h.curandDiscreteDistribution_t) return unsigned;  -- /usr/local/cuda-8.0/include/curand_discrete2.h:161
   pragma Import (CPP, curand_discrete, "_ZL15curand_discreteP19curandStateMRG32k3aP29curandDiscreteDistribution_st");

  -- * \brief Return a discrete distributed unsigned int from a MTGP32 generator.
  -- *
  -- * Return a single discrete distributed unsigned int derived from a
  -- * distribution defined by \p discrete_distribution from the MTGP32 generator in \p state,
  -- * increment position of generator by one.
  -- *
  -- * \param state - Pointer to state to update
  -- * \param discrete_distribution - ancillary structure for discrete distribution
  -- *
  -- * \return unsigned int distributed by distribution defined by \p discrete_distribution.
  --  

   function curand_discrete (state : access curand_mtgp32_h.curandStateMtgp32_t; discrete_distribution : curand_h.curandDiscreteDistribution_t) return unsigned;  -- /usr/local/cuda-8.0/include/curand_discrete2.h:178
   pragma Import (CPP, curand_discrete, "_ZL15curand_discreteP17curandStateMtgp32P29curandDiscreteDistribution_st");

  -- * \brief Return a discrete distributed unsigned int from a Sobol32 generator.
  -- *
  -- * Return a single discrete distributed unsigned int derived from a
  -- * distribution defined by \p discrete_distribution from the Sobol32 generator in \p state,
  -- * increment position of generator by one.
  -- *
  -- * \param state - Pointer to state to update
  -- * \param discrete_distribution - ancillary structure for discrete distribution
  -- *
  -- * \return unsigned int distributed by distribution defined by \p discrete_distribution.
  --  

   function curand_discrete (state : access curand_kernel_h.curandStateSobol32_t; discrete_distribution : curand_h.curandDiscreteDistribution_t) return unsigned;  -- /usr/local/cuda-8.0/include/curand_discrete2.h:195
   pragma Import (CPP, curand_discrete, "_ZL15curand_discreteP18curandStateSobol32P29curandDiscreteDistribution_st");

  -- * \brief Return a discrete distributed unsigned int from a scrambled Sobol32 generator.
  -- *
  -- * Return a single discrete distributed unsigned int derived from a
  -- * distribution defined by \p discrete_distribution from the scrambled Sobol32 generator in \p state,
  -- * increment position of generator by one.
  -- *
  -- * \param state - Pointer to state to update
  -- * \param discrete_distribution - ancillary structure for discrete distribution
  -- *
  -- * \return unsigned int distributed by distribution defined by \p discrete_distribution.
  --  

   function curand_discrete (state : access curand_kernel_h.curandStateScrambledSobol32_t; discrete_distribution : curand_h.curandDiscreteDistribution_t) return unsigned;  -- /usr/local/cuda-8.0/include/curand_discrete2.h:212
   pragma Import (CPP, curand_discrete, "_ZL15curand_discreteP27curandStateScrambledSobol32P29curandDiscreteDistribution_st");

  -- * \brief Return a discrete distributed unsigned int from a Sobol64 generator.
  -- *
  -- * Return a single discrete distributed unsigned int derived from a
  -- * distribution defined by \p discrete_distribution from the Sobol64 generator in \p state,
  -- * increment position of generator by one.
  -- *
  -- * \param state - Pointer to state to update
  -- * \param discrete_distribution - ancillary structure for discrete distribution
  -- *
  -- * \return unsigned int distributed by distribution defined by \p discrete_distribution.
  --  

   function curand_discrete (state : access curand_kernel_h.curandStateSobol64_t; discrete_distribution : curand_h.curandDiscreteDistribution_t) return unsigned;  -- /usr/local/cuda-8.0/include/curand_discrete2.h:229
   pragma Import (CPP, curand_discrete, "_ZL15curand_discreteP18curandStateSobol64P29curandDiscreteDistribution_st");

  -- * \brief Return a discrete distributed unsigned int from a scrambled Sobol64 generator.
  -- *
  -- * Return a single discrete distributed unsigned int derived from a
  -- * distribution defined by \p discrete_distribution from the scrambled Sobol64 generator in \p state,
  -- * increment position of generator by one.
  -- *
  -- * \param state - Pointer to state to update
  -- * \param discrete_distribution - ancillary structure for discrete distribution
  -- *
  -- * \return unsigned int distributed by distribution defined by \p discrete_distribution.
  --  

   function curand_discrete (state : access curand_kernel_h.curandStateScrambledSobol64_t; discrete_distribution : curand_h.curandDiscreteDistribution_t) return unsigned;  -- /usr/local/cuda-8.0/include/curand_discrete2.h:246
   pragma Import (CPP, curand_discrete, "_ZL15curand_discreteP27curandStateScrambledSobol64P29curandDiscreteDistribution_st");

end curand_discrete2_h;

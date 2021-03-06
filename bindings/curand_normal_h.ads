pragma Ada_2005;
pragma Style_Checks (Off);

with Interfaces.C; use Interfaces.C;
with vector_types_h;

package curand_normal_h is

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

   --  skipped func _curand_box_muller

   curand_box_muller_mrg : aliased vector_types_h.float2;  -- /usr/local/cuda-8.0/include/curand_normal.h:85
   pragma Import (CPP, curand_box_muller_mrg, "_ZL21curand_box_muller_mrg");

   --  skipped func _curand_box_muller_double

   curand_box_muller_mrg_double : aliased vector_types_h.double2;  -- /usr/local/cuda-8.0/include/curand_normal.h:130
   pragma Import (CPP, curand_box_muller_mrg_double, "_ZL28curand_box_muller_mrg_double");

  --unsigned int y = curand(state);
  --QUALIFIERS float _curand_normal_icdf(unsigned int x)
  --{
  --#if __CUDA_ARCH__ > 0 || defined(HOST_HAVE_ERFCINVF)
  --    float s = CURAND_SQRT2;
  --    // Mirror to avoid loss of precision
  --    if(x > 0x80000000UL) {
  --        x = 0xffffffffUL - x;
  --        s = -s;
  --    }
  --    float p = x * CURAND_2POW32_INV + (CURAND_2POW32_INV/2.0f);
  --    // p is in (0, 0.5], 2p is in (0, 1]
  --    return s * erfcinvf(2.0f * p);
  --#else
  --    x++;    //suppress warnings
  --    return 0.0f;
  --#endif
  --}
  --QUALIFIERS float _curand_normal_icdf(unsigned long long x)
  --{
  --#if __CUDA_ARCH__ > 0 || defined(HOST_HAVE_ERFCINVF)
  --    unsigned int t = (unsigned int)(x >> 32);
  --    float s = CURAND_SQRT2;
  --    // Mirror to avoid loss of precision
  --    if(t > 0x80000000UL) {
  --        t = 0xffffffffUL - t;
  --        s = -s;
  --    }
  --    float p = t * CURAND_2POW32_INV + (CURAND_2POW32_INV/2.0f);
  --    // p is in (0, 0.5], 2p is in (0, 1]
  --    return s * erfcinvf(2.0f * p);
  --#else
  --    x++;
  --    return 0.0f;
  --#endif
  --}
  --QUALIFIERS double _curand_normal_icdf_double(unsigned int x)
  --{
  --#if __CUDA_ARCH__ > 0 || defined(HOST_HAVE_ERFCINVF)
  --    double s = CURAND_SQRT2_DOUBLE;
  --    // Mirror to avoid loss of precision
  --    if(x > 0x80000000UL) {
  --        x = 0xffffffffUL - x;
  --        s = -s;
  --    }
  --    double p = x * CURAND_2POW32_INV_DOUBLE + (CURAND_2POW32_INV_DOUBLE/2.0);
  --    // p is in (0, 0.5], 2p is in (0, 1]
  --    return s * erfcinv(2.0 * p);
  --#else
  --    x++;
  --    return 0.0;
  --#endif
  --}
  --QUALIFIERS double _curand_normal_icdf_double(unsigned long long x)
  --{
  --#if __CUDA_ARCH__ > 0 || defined(HOST_HAVE_ERFCINVF)
  --    double s = CURAND_SQRT2_DOUBLE;
  --    x >>= 11;
  --    // Mirror to avoid loss of precision
  --    if(x > 0x10000000000000UL) {
  --        x = 0x1fffffffffffffUL - x;
  --        s = -s;
  --    }
  --    double p = x * CURAND_2POW53_INV_DOUBLE + (CURAND_2POW53_INV_DOUBLE/2.0);
  --    // p is in (0, 0.5], 2p is in (0, 1]
  --    return s * erfcinv(2.0 * p);
  --#else
  --    x++;
  --    return 0.0;
  --#endif
  --}
  -- 
  --*
  -- * \brief Return a normally distributed float from an XORWOW generator.
  -- *
  -- * Return a single normally distributed float with mean \p 0.0f and
  -- * standard deviation \p 1.0f from the XORWOW generator in \p state,
  -- * increment position of generator by one.
  -- *
  -- * The implementation uses a Box-Muller transform to generate two
  -- * normally distributed results, then returns them one at a time.
  -- * See ::curand_normal2() for a more efficient version that returns
  -- * both results at once.
  -- *
  -- * \param state - Pointer to state to update
  -- *
  -- * \return Normally distributed float with mean \p 0.0f and standard deviation \p 1.0f
  --  

   curand_normal : aliased float;  -- /usr/local/cuda-8.0/include/curand_normal.h:308
   pragma Import (CPP, curand_normal, "_ZL13curand_normal");

  --*
  -- * \brief Return a normally distributed float from an Philox4_32_10 generator.
  -- *
  -- * Return a single normally distributed float with mean \p 0.0f and
  -- * standard deviation \p 1.0f from the Philox4_32_10 generator in \p state,
  -- * increment position of generator by one.
  -- *
  -- * The implementation uses a Box-Muller transform to generate two
  -- * normally distributed results, then returns them one at a time.
  -- * See ::curand_normal2() for a more efficient version that returns
  -- * both results at once.
  -- *
  -- * \param state - Pointer to state to update
  -- *
  -- * \return Normally distributed float with mean \p 0.0f and standard deviation \p 1.0f
  --  

  --*
  -- * \brief Return a normally distributed float from an MRG32k3a generator.
  -- *
  -- * Return a single normally distributed float with mean \p 0.0f and
  -- * standard deviation \p 1.0f from the MRG32k3a generator in \p state,
  -- * increment position of generator by one.
  -- *
  -- * The implementation uses a Box-Muller transform to generate two
  -- * normally distributed results, then returns them one at a time.
  -- * See ::curand_normal2() for a more efficient version that returns
  -- * both results at once.
  -- *
  -- * \param state - Pointer to state to update
  -- *
  -- * \return Normally distributed float with mean \p 0.0f and standard deviation \p 1.0f
  --  

  --*
  -- * \brief Return two normally distributed floats from an XORWOW generator.
  -- *
  -- * Return two normally distributed floats with mean \p 0.0f and
  -- * standard deviation \p 1.0f from the XORWOW generator in \p state,
  -- * increment position of generator by two.
  -- *
  -- * The implementation uses a Box-Muller transform to generate two
  -- * normally distributed results.
  -- *
  -- * \param state - Pointer to state to update
  -- *
  -- * \return Normally distributed float2 where each element is from a
  -- * distribution with mean \p 0.0f and standard deviation \p 1.0f
  --  

  --*
  -- * \brief Return two normally distributed floats from an Philox4_32_10 generator.
  -- *
  -- * Return two normally distributed floats with mean \p 0.0f and
  -- * standard deviation \p 1.0f from the Philox4_32_10 generator in \p state,
  -- * increment position of generator by two.
  -- *
  -- * The implementation uses a Box-Muller transform to generate two
  -- * normally distributed results.
  -- *
  -- * \param state - Pointer to state to update
  -- *
  -- * \return Normally distributed float2 where each element is from a
  -- * distribution with mean \p 0.0f and standard deviation \p 1.0f
  --  

  --*
  -- * \brief Return four normally distributed floats from an Philox4_32_10 generator.
  -- *
  -- * Return four normally distributed floats with mean \p 0.0f and
  -- * standard deviation \p 1.0f from the Philox4_32_10 generator in \p state,
  -- * increment position of generator by four.
  -- *
  -- * The implementation uses a Box-Muller transform to generate two
  -- * normally distributed results.
  -- *
  -- * \param state - Pointer to state to update
  -- *
  -- * \return Normally distributed float2 where each element is from a
  -- * distribution with mean \p 0.0f and standard deviation \p 1.0f
  --  

  --*
  -- * \brief Return two normally distributed floats from an MRG32k3a generator.
  -- *
  -- * Return two normally distributed floats with mean \p 0.0f and
  -- * standard deviation \p 1.0f from the MRG32k3a generator in \p state,
  -- * increment position of generator by two.
  -- *
  -- * The implementation uses a Box-Muller transform to generate two
  -- * normally distributed results.
  -- *
  -- * \param state - Pointer to state to update
  -- *
  -- * \return Normally distributed float2 where each element is from a
  -- * distribution with mean \p 0.0f and standard deviation \p 1.0f
  --  

  --*
  -- * \brief Return a normally distributed float from a MTGP32 generator.
  -- *
  -- * Return a single normally distributed float with mean \p 0.0f and
  -- * standard deviation \p 1.0f from the MTGP32 generator in \p state,
  -- * increment position of generator.
  -- *
  -- * The implementation uses the inverse cumulative distribution function
  -- * to generate normally distributed results.
  -- *
  -- * \param state - Pointer to state to update
  -- *
  -- * \return Normally distributed float with mean \p 0.0f and standard deviation \p 1.0f
  --  

  --*
  -- * \brief Return a normally distributed float from a Sobol32 generator.
  -- *
  -- * Return a single normally distributed float with mean \p 0.0f and
  -- * standard deviation \p 1.0f from the Sobol32 generator in \p state,
  -- * increment position of generator by one.
  -- *
  -- * The implementation uses the inverse cumulative distribution function
  -- * to generate normally distributed results.
  -- *
  -- * \param state - Pointer to state to update
  -- *
  -- * \return Normally distributed float with mean \p 0.0f and standard deviation \p 1.0f
  --  

  --*
  -- * \brief Return a normally distributed float from a scrambled Sobol32 generator.
  -- *
  -- * Return a single normally distributed float with mean \p 0.0f and
  -- * standard deviation \p 1.0f from the scrambled Sobol32 generator in \p state,
  -- * increment position of generator by one.
  -- *
  -- * The implementation uses the inverse cumulative distribution function
  -- * to generate normally distributed results.
  -- *
  -- * \param state - Pointer to state to update
  -- *
  -- * \return Normally distributed float with mean \p 0.0f and standard deviation \p 1.0f
  --  

  --*
  -- * \brief Return a normally distributed float from a Sobol64 generator.
  -- *
  -- * Return a single normally distributed float with mean \p 0.0f and
  -- * standard deviation \p 1.0f from the Sobol64 generator in \p state,
  -- * increment position of generator by one.
  -- *
  -- * The implementation uses the inverse cumulative distribution function
  -- * to generate normally distributed results.
  -- *
  -- * \param state - Pointer to state to update
  -- *
  -- * \return Normally distributed float with mean \p 0.0f and standard deviation \p 1.0f
  --  

  --*
  -- * \brief Return a normally distributed float from a scrambled Sobol64 generator.
  -- *
  -- * Return a single normally distributed float with mean \p 0.0f and
  -- * standard deviation \p 1.0f from the scrambled Sobol64 generator in \p state,
  -- * increment position of generator by one.
  -- *
  -- * The implementation uses the inverse cumulative distribution function
  -- * to generate normally distributed results.
  -- *
  -- * \param state - Pointer to state to update
  -- *
  -- * \return Normally distributed float with mean \p 0.0f and standard deviation \p 1.0f
  --  

  --*
  -- * \brief Return a normally distributed double from an XORWOW generator.
  -- *
  -- * Return a single normally distributed double with mean \p 0.0 and
  -- * standard deviation \p 1.0 from the XORWOW generator in \p state,
  -- * increment position of generator.
  -- *
  -- * The implementation uses a Box-Muller transform to generate two
  -- * normally distributed results, then returns them one at a time.
  -- * See ::curand_normal2_double() for a more efficient version that returns
  -- * both results at once.
  -- *
  -- * \param state - Pointer to state to update
  -- *
  -- * \return Normally distributed double with mean \p 0.0 and standard deviation \p 1.0
  --  

  --*
  -- * \brief Return a normally distributed double from an Philox4_32_10 generator.
  -- *
  -- * Return a single normally distributed double with mean \p 0.0 and
  -- * standard deviation \p 1.0 from the Philox4_32_10 generator in \p state,
  -- * increment position of generator.
  -- *
  -- * The implementation uses a Box-Muller transform to generate two
  -- * normally distributed results, then returns them one at a time.
  -- * See ::curand_normal2_double() for a more efficient version that returns
  -- * both results at once.
  -- *
  -- * \param state - Pointer to state to update
  -- *
  -- * \return Normally distributed double with mean \p 0.0 and standard deviation \p 1.0
  --  

  --*
  -- * \brief Return a normally distributed double from an MRG32k3a generator.
  -- *
  -- * Return a single normally distributed double with mean \p 0.0 and
  -- * standard deviation \p 1.0 from the XORWOW generator in \p state,
  -- * increment position of generator.
  -- *
  -- * The implementation uses a Box-Muller transform to generate two
  -- * normally distributed results, then returns them one at a time.
  -- * See ::curand_normal2_double() for a more efficient version that returns
  -- * both results at once.
  -- *
  -- * \param state - Pointer to state to update
  -- *
  -- * \return Normally distributed double with mean \p 0.0 and standard deviation \p 1.0
  --  

  --*
  -- * \brief Return two normally distributed doubles from an XORWOW generator.
  -- *
  -- * Return two normally distributed doubles with mean \p 0.0 and
  -- * standard deviation \p 1.0 from the XORWOW generator in \p state,
  -- * increment position of generator by 2.
  -- *
  -- * The implementation uses a Box-Muller transform to generate two
  -- * normally distributed results.
  -- *
  -- * \param state - Pointer to state to update
  -- *
  -- * \return Normally distributed double2 where each element is from a
  -- * distribution with mean \p 0.0 and standard deviation \p 1.0
  --  

  --*
  -- * \brief Return two normally distributed doubles from an Philox4_32_10 generator.
  -- *
  -- * Return two normally distributed doubles with mean \p 0.0 and
  -- * standard deviation \p 1.0 from the Philox4_32_10 generator in \p state,
  -- * increment position of generator by 2.
  -- *
  -- * The implementation uses a Box-Muller transform to generate two
  -- * normally distributed results.
  -- *
  -- * \param state - Pointer to state to update
  -- *
  -- * \return Normally distributed double2 where each element is from a
  -- * distribution with mean \p 0.0 and standard deviation \p 1.0
  --  

  -- not a part of API
  --*
  -- * \brief Return two normally distributed doubles from an MRG32k3a generator.
  -- *
  -- * Return two normally distributed doubles with mean \p 0.0 and
  -- * standard deviation \p 1.0 from the MRG32k3a generator in \p state,
  -- * increment position of generator.
  -- *
  -- * The implementation uses a Box-Muller transform to generate two
  -- * normally distributed results.
  -- *
  -- * \param state - Pointer to state to update
  -- *
  -- * \return Normally distributed double2 where each element is from a
  -- * distribution with mean \p 0.0 and standard deviation \p 1.0
  --  

  --*
  -- * \brief Return a normally distributed double from an MTGP32 generator.
  -- *
  -- * Return a single normally distributed double with mean \p 0.0 and
  -- * standard deviation \p 1.0 from the MTGP32 generator in \p state,
  -- * increment position of generator.
  -- *
  -- * The implementation uses the inverse cumulative distribution function
  -- * to generate normally distributed results.
  -- *
  -- * \param state - Pointer to state to update
  -- *
  -- * \return Normally distributed double with mean \p 0.0 and standard deviation \p 1.0
  --  

  --*
  -- * \brief Return a normally distributed double from an Sobol32 generator.
  -- *
  -- * Return a single normally distributed double with mean \p 0.0 and
  -- * standard deviation \p 1.0 from the Sobol32 generator in \p state,
  -- * increment position of generator by one.
  -- *
  -- * The implementation uses the inverse cumulative distribution function
  -- * to generate normally distributed results.
  -- *
  -- * \param state - Pointer to state to update
  -- *
  -- * \return Normally distributed double with mean \p 0.0 and standard deviation \p 1.0
  --  

  --*
  -- * \brief Return a normally distributed double from a scrambled Sobol32 generator.
  -- *
  -- * Return a single normally distributed double with mean \p 0.0 and
  -- * standard deviation \p 1.0 from the scrambled Sobol32 generator in \p state,
  -- * increment position of generator by one.
  -- *
  -- * The implementation uses the inverse cumulative distribution function
  -- * to generate normally distributed results.
  -- *
  -- * \param state - Pointer to state to update
  -- *
  -- * \return Normally distributed double with mean \p 0.0 and standard deviation \p 1.0
  --  

  --*
  -- * \brief Return a normally distributed double from a Sobol64 generator.
  -- *
  -- * Return a single normally distributed double with mean \p 0.0 and
  -- * standard deviation \p 1.0 from the Sobol64 generator in \p state,
  -- * increment position of generator by one.
  -- *
  -- * The implementation uses the inverse cumulative distribution function
  -- * to generate normally distributed results.
  -- *
  -- * \param state - Pointer to state to update
  -- *
  -- * \return Normally distributed double with mean \p 0.0 and standard deviation \p 1.0
  --  

  --*
  -- * \brief Return a normally distributed double from a scrambled Sobol64 generator.
  -- *
  -- * Return a single normally distributed double with mean \p 0.0 and
  -- * standard deviation \p 1.0 from the scrambled Sobol64 generator in \p state,
  -- * increment position of generator by one.
  -- *
  -- * The implementation uses the inverse cumulative distribution function
  -- * to generate normally distributed results.
  -- *
  -- * \param state - Pointer to state to update
  -- *
  -- * \return Normally distributed double with mean \p 0.0 and standard deviation \p 1.0
  --  

end curand_normal_h;

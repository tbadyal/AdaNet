pragma Ada_2005;
pragma Style_Checks (Off);

with Interfaces.C; use Interfaces.C;

package curand_lognormal_h is

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

  --*
  -- * \brief Return a log-normally distributed float from an XORWOW generator.
  -- *
  -- * Return a single log-normally distributed float derived from a normal
  -- * distribution with mean \p mean and standard deviation \p stddev 
  -- * from the XORWOW generator in \p state, 
  -- * increment position of generator by one.
  -- *
  -- * The implementation uses a Box-Muller transform to generate two
  -- * normally distributed results, transforms them to log-normal distribution,
  -- * then returns them one at a time.
  -- * See ::curand_log_normal2() for a more efficient version that returns
  -- * both results at once.
  -- *
  -- * \param state  - Pointer to state to update
  -- * \param mean   - Mean of the related normal distribution
  -- * \param stddev - Standard deviation of the related normal distribution
  -- *
  -- * \return Log-normally distributed float with mean \p mean and standard deviation \p stddev
  --  

   curand_log_normal : aliased float;  -- /usr/local/cuda-8.0/include/curand_lognormal.h:87
   pragma Import (CPP, curand_log_normal, "_ZL17curand_log_normal");

  --*
  -- * \brief Return a log-normally distributed float from an Philox4_32_10 generator.
  -- *
  -- * Return a single log-normally distributed float derived from a normal
  -- * distribution with mean \p mean and standard deviation \p stddev 
  -- * from the Philox4_32_10 generator in \p state, 
  -- * increment position of generator by one.
  -- *
  -- * The implementation uses a Box-Muller transform to generate two
  -- * normally distributed results, transforms them to log-normal distribution,
  -- * then returns them one at a time.
  -- * See ::curand_log_normal2() for a more efficient version that returns
  -- * both results at once.
  -- *
  -- * \param state  - Pointer to state to update
  -- * \param mean   - Mean of the related normal distribution
  -- * \param stddev - Standard deviation of the related normal distribution
  -- *
  -- * \return Log-normally distributed float with mean \p mean and standard deviation \p stddev
  --  

  --*
  -- * \brief Return two normally distributed floats from an XORWOW generator.
  -- *
  -- * Return two log-normally distributed floats derived from a normal
  -- * distribution with mean \p mean and standard deviation \p stddev 
  -- * from the XORWOW generator in \p state,
  -- * increment position of generator by two.
  -- *
  -- * The implementation uses a Box-Muller transform to generate two
  -- * normally distributed results, then transforms them to log-normal.
  -- *
  -- * \param state - Pointer to state to update
  -- * \param mean   - Mean of the related normal distribution
  -- * \param stddev - Standard deviation of the related normal distribution
  -- *
  -- * \return Log-normally distributed float2 where each element is from a
  -- * distribution with mean \p mean and standard deviation \p stddev
  --  

  --*
  -- * \brief Return two normally distributed floats from an Philox4_32_10 generator.
  -- *
  -- * Return two log-normally distributed floats derived from a normal
  -- * distribution with mean \p mean and standard deviation \p stddev 
  -- * from the Philox4_32_10 generator in \p state,
  -- * increment position of generator by two.
  -- *
  -- * The implementation uses a Box-Muller transform to generate two
  -- * normally distributed results, then transforms them to log-normal.
  -- *
  -- * \param state - Pointer to state to update
  -- * \param mean   - Mean of the related normal distribution
  -- * \param stddev - Standard deviation of the related normal distribution
  -- *
  -- * \return Log-normally distributed float2 where each element is from a
  -- * distribution with mean \p mean and standard deviation \p stddev
  --  

  --*
  -- * \brief Return four normally distributed floats from an Philox4_32_10 generator.
  -- *
  -- * Return four log-normally distributed floats derived from a normal
  -- * distribution with mean \p mean and standard deviation \p stddev 
  -- * from the Philox4_32_10 generator in \p state,
  -- * increment position of generator by four.
  -- *
  -- * The implementation uses a Box-Muller transform to generate two
  -- * normally distributed results, then transforms them to log-normal.
  -- *
  -- * \param state - Pointer to state to update
  -- * \param mean   - Mean of the related normal distribution
  -- * \param stddev - Standard deviation of the related normal distribution
  -- *
  -- * \return Log-normally distributed float4 where each element is from a
  -- * distribution with mean \p mean and standard deviation \p stddev
  --  

  --*
  -- * \brief Return a log-normally distributed float from an MRG32k3a generator.
  -- *
  -- * Return a single log-normally distributed float derived from a normal
  -- * distribution with mean \p mean and standard deviation \p stddev 
  -- * from the MRG32k3a generator in \p state, 
  -- * increment position of generator by one.
  -- *
  -- * The implementation uses a Box-Muller transform to generate two
  -- * normally distributed results, transforms them to log-normal distribution,
  -- * then returns them one at a time.
  -- * See ::curand_log_normal2() for a more efficient version that returns
  -- * both results at once.
  -- *
  -- * \param state  - Pointer to state to update
  -- * \param mean   - Mean of the related normal distribution
  -- * \param stddev - Standard deviation of the related normal distribution
  -- *
  -- * \return Log-normally distributed float with mean \p mean and standard deviation \p stddev
  --  

  --*
  -- * \brief Return two normally distributed floats from an MRG32k3a generator.
  -- *
  -- * Return two log-normally distributed floats derived from a normal
  -- * distribution with mean \p mean and standard deviation \p stddev 
  -- * from the MRG32k3a generator in \p state,
  -- * increment position of generator by two.
  -- *
  -- * The implementation uses a Box-Muller transform to generate two
  -- * normally distributed results, then transforms them to log-normal.
  -- *
  -- * \param state - Pointer to state to update
  -- * \param mean   - Mean of the related normal distribution
  -- * \param stddev - Standard deviation of the related normal distribution
  -- *
  -- * \return Log-normally distributed float2 where each element is from a
  -- * distribution with mean \p mean and standard deviation \p stddev
  --  

  --*
  -- * \brief Return a log-normally distributed float from an MTGP32 generator.
  -- *
  -- * Return a single log-normally distributed float derived from a normal
  -- * distribution with mean \p mean and standard deviation \p stddev 
  -- * from the MTGP32 generator in \p state,
  -- * increment position of generator.
  -- *
  -- * The implementation uses the inverse cumulative distribution function
  -- * to generate a normally distributed result, then transforms the result
  -- * to log-normal.
  -- *
  -- * \param state - Pointer to state to update
  -- * \param mean   - Mean of the related normal distribution
  -- * \param stddev - Standard deviation of the related normal distribution
  -- *
  -- * \return Log-normally distributed float with mean \p mean and standard deviation \p stddev
  --  

  --*
  -- * \brief Return a log-normally distributed float from a Sobol32 generator.
  -- *
  -- * Return a single log-normally distributed float derived from a normal
  -- * distribution with mean \p mean and standard deviation \p stddev 
  -- * from the Sobol32 generator in \p state,
  -- * increment position of generator by one.
  -- *
  -- * The implementation uses the inverse cumulative distribution function
  -- * to generate a normally distributed result, then transforms the result
  -- * to log-normal.
  -- *
  -- * \param state - Pointer to state to update
  -- * \param mean   - Mean of the related normal distribution
  -- * \param stddev - Standard deviation of the related normal distribution
  -- *
  -- * \return Log-normally distributed float with mean \p mean and standard deviation \p stddev
  --  

  --*
  -- * \brief Return a log-normally distributed float from a scrambled Sobol32 generator.
  -- *
  -- * Return a single log-normally distributed float derived from a normal
  -- * distribution with mean \p mean and standard deviation \p stddev 
  -- * from the scrambled Sobol32 generator in \p state,
  -- * increment position of generator by one.
  -- *
  -- * The implementation uses the inverse cumulative distribution function
  -- * to generate a normally distributed result, then transforms the result
  -- * to log-normal.
  -- *
  -- * \param state - Pointer to state to update
  -- * \param mean   - Mean of the related normal distribution
  -- * \param stddev - Standard deviation of the related normal distribution
  -- *
  -- * \return Log-normally distributed float with mean \p mean and standard deviation \p stddev
  --  

  --*
  -- * \brief Return a log-normally distributed float from a Sobol64 generator.
  -- *
  -- * Return a single log-normally distributed float derived from a normal
  -- * distribution with mean \p mean and standard deviation \p stddev 
  -- * from the Sobol64 generator in \p state,
  -- * increment position of generator by one.
  -- *
  -- * The implementation uses the inverse cumulative distribution function
  -- * to generate normally distributed results, then converts to log-normal
  -- * distribution.
  -- *
  -- * \param state - Pointer to state to update
  -- * \param mean   - Mean of the related normal distribution
  -- * \param stddev - Standard deviation of the related normal distribution
  -- *
  -- * \return Log-normally distributed float with mean \p mean and standard deviation \p stddev
  --  

  --*
  -- * \brief Return a log-normally distributed float from a scrambled Sobol64 generator.
  -- *
  -- * Return a single log-normally distributed float derived from a normal
  -- * distribution with mean \p mean and standard deviation \p stddev 
  -- * from the scrambled Sobol64 generator in \p state,
  -- * increment position of generator by one.
  -- *
  -- * The implementation uses the inverse cumulative distribution function
  -- * to generate normally distributed results, then converts to log-normal
  -- * distribution.
  -- *
  -- * \param state - Pointer to state to update
  -- * \param mean   - Mean of the related normal distribution
  -- * \param stddev - Standard deviation of the related normal distribution
  -- *
  -- * \return Log-normally distributed float with mean \p mean and standard deviation \p stddev
  --  

  --*
  -- * \brief Return a log-normally distributed double from an XORWOW generator.
  -- *
  -- * Return a single normally distributed double derived from a normal
  -- * distribution with mean \p mean and standard deviation \p stddev 
  -- * from the XORWOW generator in \p state,
  -- * increment position of generator.
  -- *
  -- * The implementation uses a Box-Muller transform to generate two
  -- * normally distributed results, transforms them to log-normal distribution,
  -- * then returns them one at a time.
  -- * See ::curand_log_normal2_double() for a more efficient version that returns
  -- * both results at once.
  -- *
  -- * \param state - Pointer to state to update
  -- * \param mean   - Mean of the related normal distribution
  -- * \param stddev - Standard deviation of the related normal distribution
  -- *
  -- * \return Log-normally distributed double with mean \p mean and standard deviation \p stddev
  --  

  --*
  -- * \brief Return a log-normally distributed double from an Philox4_32_10 generator.
  -- *
  -- * Return a single normally distributed double derived from a normal
  -- * distribution with mean \p mean and standard deviation \p stddev 
  -- * from the Philox4_32_10 generator in \p state,
  -- * increment position of generator.
  -- *
  -- * The implementation uses a Box-Muller transform to generate two
  -- * normally distributed results, transforms them to log-normal distribution,
  -- * then returns them one at a time.
  -- * See ::curand_log_normal2_double() for a more efficient version that returns
  -- * both results at once.
  -- *
  -- * \param state - Pointer to state to update
  -- * \param mean   - Mean of the related normal distribution
  -- * \param stddev - Standard deviation of the related normal distribution
  -- *
  -- * \return Log-normally distributed double with mean \p mean and standard deviation \p stddev
  --  

  --*
  -- * \brief Return two log-normally distributed doubles from an XORWOW generator.
  -- *
  -- * Return two log-normally distributed doubles derived from a normal
  -- * distribution with mean \p mean and standard deviation \p stddev 
  -- * from the XORWOW generator in \p state,
  -- * increment position of generator by two.
  -- *
  -- * The implementation uses a Box-Muller transform to generate two
  -- * normally distributed results, and transforms them to log-normal distribution,.
  -- *
  -- * \param state - Pointer to state to update
  -- * \param mean   - Mean of the related normal distribution
  -- * \param stddev - Standard deviation of the related normal distribution
  -- *
  -- * \return Log-normally distributed double2 where each element is from a
  -- * distribution with mean \p mean and standard deviation \p stddev
  --  

  --*
  -- * \brief Return two log-normally distributed doubles from an Philox4_32_10 generator.
  -- *
  -- * Return two log-normally distributed doubles derived from a normal
  -- * distribution with mean \p mean and standard deviation \p stddev 
  -- * from the Philox4_32_10 generator in \p state,
  -- * increment position of generator by four.
  -- *
  -- * The implementation uses a Box-Muller transform to generate two
  -- * normally distributed results, and transforms them to log-normal distribution,.
  -- *
  -- * \param state - Pointer to state to update
  -- * \param mean   - Mean of the related normal distribution
  -- * \param stddev - Standard deviation of the related normal distribution
  -- *
  -- * \return Log-normally distributed double4 where each element is from a
  -- * distribution with mean \p mean and standard deviation \p stddev
  --  

  -- nor part of API
  --*
  -- * \brief Return a log-normally distributed double from an MRG32k3a generator.
  -- *
  -- * Return a single normally distributed double derived from a normal
  -- * distribution with mean \p mean and standard deviation \p stddev 
  -- * from the MRG32k3a generator in \p state,
  -- * increment position of generator.
  -- *
  -- * The implementation uses a Box-Muller transform to generate two
  -- * normally distributed results, transforms them to log-normal distribution,
  -- * then returns them one at a time.
  -- * See ::curand_log_normal2_double() for a more efficient version that returns
  -- * both results at once.
  -- *
  -- * \param state - Pointer to state to update
  -- * \param mean   - Mean of the related normal distribution
  -- * \param stddev - Standard deviation of the related normal distribution
  -- *
  -- * \return Log-normally distributed double with mean \p mean and standard deviation \p stddev
  --  

  --*
  -- * \brief Return two log-normally distributed doubles from an MRG32k3a generator.
  -- *
  -- * Return two log-normally distributed doubles derived from a normal
  -- * distribution with mean \p mean and standard deviation \p stddev 
  -- * from the MRG32k3a generator in \p state,
  -- * increment position of generator by two.
  -- *
  -- * The implementation uses a Box-Muller transform to generate two
  -- * normally distributed results, and transforms them to log-normal distribution,.
  -- *
  -- * \param state - Pointer to state to update
  -- * \param mean   - Mean of the related normal distribution
  -- * \param stddev - Standard deviation of the related normal distribution
  -- *
  -- * \return Log-normally distributed double2 where each element is from a
  -- * distribution with mean \p mean and standard deviation \p stddev
  --  

  --*
  -- * \brief Return a log-normally distributed double from an MTGP32 generator.
  -- *
  -- * Return a single log-normally distributed double derived from a normal
  -- * distribution with mean \p mean and standard deviation \p stddev 
  -- * from the MTGP32 generator in \p state,
  -- * increment position of generator.
  -- *
  -- * The implementation uses the inverse cumulative distribution function
  -- * to generate normally distributed results, and transforms them into
  -- * log-normal distribution.
  -- *
  -- * \param state - Pointer to state to update
  -- * \param mean   - Mean of the related normal distribution
  -- * \param stddev - Standard deviation of the related normal distribution
  -- *
  -- * \return Log-normally distributed double with mean \p mean and standard deviation \p stddev
  --  

  --*
  -- * \brief Return a log-normally distributed double from a Sobol32 generator.
  -- *
  -- * Return a single log-normally distributed double derived from a normal
  -- * distribution with mean \p mean and standard deviation \p stddev 
  -- * from the Sobol32 generator in \p state,
  -- * increment position of generator by one.
  -- *
  -- * The implementation uses the inverse cumulative distribution function
  -- * to generate normally distributed results, and transforms them into
  -- * log-normal distribution.
  -- *
  -- * \param state - Pointer to state to update
  -- * \param mean   - Mean of the related normal distribution
  -- * \param stddev - Standard deviation of the related normal distribution
  -- *
  -- * \return Log-normally distributed double with mean \p mean and standard deviation \p stddev
  --  

  --*
  -- * \brief Return a log-normally distributed double from a scrambled Sobol32 generator.
  -- *
  -- * Return a single log-normally distributed double derived from a normal
  -- * distribution with mean \p mean and standard deviation \p stddev 
  -- * from the scrambled Sobol32 generator in \p state,
  -- * increment position of generator by one.
  -- *
  -- * The implementation uses the inverse cumulative distribution function
  -- * to generate normally distributed results, and transforms them into
  -- * log-normal distribution.
  -- *
  -- * \param state - Pointer to state to update
  -- * \param mean   - Mean of the related normal distribution
  -- * \param stddev - Standard deviation of the related normal distribution
  -- *
  -- * \return Log-normally distributed double with mean \p mean and standard deviation \p stddev
  --  

  --*
  -- * \brief Return a log-normally distributed double from a Sobol64 generator.
  -- *
  -- * Return a single normally distributed double derived from a normal
  -- * distribution with mean \p mean and standard deviation \p stddev 
  -- * from the Sobol64 generator in \p state,
  -- * increment position of generator by one.
  -- *
  -- * The implementation uses the inverse cumulative distribution function
  -- * to generate normally distributed results.
  -- *
  -- * \param state - Pointer to state to update
  -- * \param mean   - Mean of the related normal distribution
  -- * \param stddev - Standard deviation of the related normal distribution
  -- *
  -- * \return Log-normally distributed double with mean \p mean and standard deviation \p stddev
  --  

  --*
  -- * \brief Return a log-normally distributed double from a scrambled Sobol64 generator.
  -- *
  -- * Return a single normally distributed double derived from a normal
  -- * distribution with mean \p mean and standard deviation \p stddev 
  -- * from the scrambled Sobol64 generator in \p state,
  -- * increment position of generator by one.
  -- *
  -- * The implementation uses the inverse cumulative distribution function
  -- * to generate normally distributed results.
  -- *
  -- * \param state - Pointer to state to update
  -- * \param mean   - Mean of the related normal distribution
  -- * \param stddev - Standard deviation of the related normal distribution
  -- *
  -- * \return Log-normally distributed double with mean \p mean and standard deviation \p stddev
  --  

end curand_lognormal_h;

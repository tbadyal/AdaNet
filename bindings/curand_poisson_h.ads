pragma Ada_2005;
pragma Style_Checks (Off);

with Interfaces.C; use Interfaces.C;
limited with curand_philox4x32_x_h;
with vector_types_h;

package curand_poisson_h is

   CR_CUDART_PI : constant := 3.1415926535897931e+0;  --  /usr/local/cuda-8.0/include/curand_poisson.h:67
   CR_CUDART_TWO_TO_52 : constant := 4503599627370496.0;  --  /usr/local/cuda-8.0/include/curand_poisson.h:68

   KNUTH_FLOAT_CONST : constant := 60.0;  --  /usr/local/cuda-8.0/include/curand_poisson.h:428

   MAGIC_DOUBLE_CONST : constant := 500.0;  --  /usr/local/cuda-8.0/include/curand_poisson.h:592

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

   --  skipped func __cr_isnan

   --  skipped func __cr_isinf

   --  skipped func __cr_copysign

   --  skipped func __cr_rint

   --  skipped func __cr_rsqrt

   --  skipped func __cr_exp

   --  skipped func __cr_log

   --  skipped func __cr_rcp

  -- Computes regularized gamma function:  gammainc(a,x)/gamma(a)  
   --  skipped func __cr_pgammainc

  -- First level parametrization constants  
  -- Second level parametrization constants (depends only on a)  
  -- Final approximation (depends on a and x)  
  -- Negative a,x or a,x=NAN requires special handling  
  --t = !(x > 0 && a >= 0) ? 0.0 : t;
  -- Computes inverse of pgammainc  
   --  skipped func __cr_pgammaincinv

  -- First level parametrization constants  
  -- Second level parametrization constants (depends only on a)  
  -- Final approximation (depends on a and y)  
  -- Negative a,x or a,x=NAN requires special handling  
  --t = !(y > 0 && a >= 0) ? 0.0 : t;
   --  skipped func __cr_lgamma_integer

  -- Stirling approximation; coefficients from Hart et al, "Computer 
  --         * Approximations", Wiley 1968. Approximation 5404. 
  --          

   --  skipped func __cr_lgamma

  -- Stirling approximation; coefficients from Hart et al, "Computer 
  --       * Approximations", Wiley 1968. Approximation 5404. 
  --        

  -- a is an integer: return infinity  
  -- Donald E. Knuth Seminumerical Algorithms. The Art of Computer Programming, Volume 2
  -- Donald E. Knuth Seminumerical Algorithms. The Art of Computer Programming, Volume 2
  -- Marsaglia, Tsang, Wang Journal of Statistical Software, square histogram.
  --if (u < distributionM2->histogram->V[j]) return distributionM2->shift + j;
  --return distributionM2->shift + distributionM2->histogram->K[j];
  -- Marsaglia, Tsang, Wang Journal of Statistical Software, square histogram.
  --    int result;
  --return distributionM2->shift + distributionM2->histogram->K[j];
  --George S. Fishman Discrete-event simulation: modeling, programming, and analysis
  -- Rejection Method for Poisson distribution based on gammainc approximation  
  -- Rejection Method for Poisson distribution based on gammainc approximation  
  -- Note below that the round to nearest integer, where needed,is done in line with code that
  -- assumes the range of values is < 2**32
  --Round to nearest
  --Round to nearest
  --Round to nearest
  --Round to nearest
  --Round to nearest
  --Round to nearest
  --Round to nearest
  --Round to nearest
  --*
  -- * \brief Return a Poisson-distributed unsigned int from a XORWOW generator.
  -- *
  -- * Return a single unsigned int from a Poisson
  -- * distribution with lambda \p lambda from the XORWOW generator in \p state,
  -- * increment the  position of the generator by a variable amount, depending 
  -- * on the algorithm used.
  -- *
  -- * \param state - Pointer to state to update
  -- * \param lambda - Lambda of the Poisson distribution
  -- *
  -- * \return Poisson-distributed unsigned int with lambda \p lambda
  --  

   curand_poisson : aliased unsigned;  -- /usr/local/cuda-8.0/include/curand_poisson.h:750
   pragma Import (CPP, curand_poisson, "_ZL14curand_poisson");

  --Round to nearest
  --*
  -- * \brief Return a Poisson-distributed unsigned int from a Philox4_32_10 generator.
  -- *
  -- * Return a single unsigned int from a Poisson
  -- * distribution with lambda \p lambda from the Philox4_32_10 generator in \p state,
  -- * increment the  position of the generator by a variable amount, depending 
  -- * on the algorithm used.
  -- *
  -- * \param state - Pointer to state to update
  -- * \param lambda - Lambda of the Poisson distribution
  -- *
  -- * \return Poisson-distributed unsigned int with lambda \p lambda
  --  

  --Round to nearest
  --*
  -- * \brief Return four Poisson-distributed unsigned ints from a Philox4_32_10 generator.
  -- *
  -- * Return a four unsigned ints from a Poisson
  -- * distribution with lambda \p lambda from the Philox4_32_10 generator in \p state,
  -- * increment the  position of the generator by a variable amount, depending 
  -- * on the algorithm used.
  -- *
  -- * \param state - Pointer to state to update
  -- * \param lambda - Lambda of the Poisson distribution
  -- *
  -- * \return Poisson-distributed unsigned int with lambda \p lambda
  --  

   function curand_poisson4 (state : access curand_philox4x32_x_h.curandStatePhilox4_32_10_t; lambda : double) return vector_types_h.uint4;  -- /usr/local/cuda-8.0/include/curand_poisson.h:793
   pragma Import (CPP, curand_poisson4, "_ZL15curand_poisson4P24curandStatePhilox4_32_10d");

  --Round to nearest
  --Round to nearest
  --Round to nearest
  --Round to nearest
  --*
  -- * \brief Return a Poisson-distributed unsigned int from a MRG32k3A generator.
  -- *
  -- * Return a single unsigned int from a Poisson
  -- * distribution with lambda \p lambda from the MRG32k3a generator in \p state,
  -- * increment the position of the generator by a variable amount, depending 
  -- * on the algorithm used.
  -- *
  -- * \param state - Pointer to state to update
  -- * \param lambda - Lambda of the Poisson distribution
  -- *
  -- * \return Poisson-distributed unsigned int with lambda \p lambda
  --  

  --Round to nearest
  --*
  -- * \brief Return a Poisson-distributed unsigned int from a MTGP32 generator.
  -- *
  -- * Return a single int from a Poisson
  -- * distribution with lambda \p lambda from the MTGP32 generator in \p state,
  -- * increment the position of the generator by one.
  -- *
  -- * \param state - Pointer to state to update
  -- * \param lambda - Lambda of the Poisson distribution
  -- *
  -- * \return Poisson-distributed unsigned int with lambda \p lambda
  --  

  --*
  -- * \brief Return a Poisson-distributed unsigned int from a Sobol32 generator.
  -- *
  -- * Return a single unsigned int from a Poisson
  -- * distribution with lambda \p lambda from the Sobol32 generator in \p state,
  -- * increment the position of the generator by one.
  -- *
  -- * \param state - Pointer to state to update
  -- * \param lambda - Lambda of the Poisson distribution
  -- *
  -- * \return Poisson-distributed unsigned int with lambda \p lambda
  --  

  --*
  -- * \brief Return a Poisson-distributed unsigned int from a scrambled Sobol32 generator.
  -- *
  -- * Return a single unsigned int from a Poisson
  -- * distribution with lambda \p lambda from the scrambled Sobol32 generator in \p state,
  -- * increment the position of the generator by one.
  -- *
  -- * \param state - Pointer to state to update
  -- * \param lambda - Lambda of the Poisson distribution
  -- *
  -- * \return Poisson-distributed unsigned int with lambda \p lambda
  --  

  --*
  -- * \brief Return a Poisson-distributed unsigned int from a Sobol64 generator.
  -- *
  -- * Return a single unsigned int from a Poisson
  -- * distribution with lambda \p lambda from the Sobol64 generator in \p state,
  -- * increment position of generator by one.
  -- *
  -- * \param state - Pointer to state to update
  -- * \param lambda - Lambda of the Poisson distribution
  -- *
  -- * \return Poisson-distributed unsigned int with lambda \p lambda
  --  

  --*
  -- * \brief Return a Poisson-distributed unsigned int from a scrambled Sobol64 generator.
  -- *
  -- * Return a single unsigned int from a Poisson
  -- * distribution with lambda \p lambda from the scrambled Sobol64 generator in \p state,
  -- * increment position of generator by one.
  -- *
  -- * \param state - Pointer to state to update
  -- * \param lambda - Lambda of the Poisson distribution
  -- *
  -- * \return Poisson-distributed unsigned int with lambda \p lambda
  --  

end curand_poisson_h;

pragma Ada_2005;
pragma Style_Checks (Off);

with Interfaces.C; use Interfaces.C;
limited with curand_philox4x32_x_h;
with vector_types_h;

package curand_uniform_h is

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

   --  skipped func _curand_uniform

   --  skipped func _curand_uniform4

   --  skipped func _curand_uniform

   --  skipped func _curand_uniform_double

   --  skipped func _curand_uniform_double

   --  skipped func _curand_uniform4_double

   --  skipped func _curand_uniform_double_hq

   curand_uniform : aliased float;  -- /usr/local/cuda-8.0/include/curand_uniform.h:116
   pragma Import (CPP, curand_uniform, "_ZL14curand_uniform");

   curand_uniform_double : aliased double;  -- /usr/local/cuda-8.0/include/curand_uniform.h:121
   pragma Import (CPP, curand_uniform_double, "_ZL21curand_uniform_double");

  --*
  -- * \brief Return a uniformly distributed float from an XORWOW generator.
  -- *
  -- * Return a uniformly distributed float between \p 0.0f and \p 1.0f 
  -- * from the XORWOW generator in \p state, increment position of generator.
  -- * Output range excludes \p 0.0f but includes \p 1.0f.  Denormalized floating
  -- * point outputs are never returned.
  -- *
  -- * The implementation may use any number of calls to \p curand() to
  -- * get enough random bits to create the return value.  The current
  -- * implementation uses one call.
  -- *
  -- * \param state - Pointer to state to update
  -- *
  -- * \return uniformly distributed float between \p 0.0f and \p 1.0f
  --  

  --*
  -- * \brief Return a uniformly distributed double from an XORWOW generator.
  -- *
  -- * Return a uniformly distributed double between \p 0.0 and \p 1.0 
  -- * from the XORWOW generator in \p state, increment position of generator.
  -- * Output range excludes \p 0.0 but includes \p 1.0.  Denormalized floating
  -- * point outputs are never returned.
  -- *
  -- * The implementation may use any number of calls to \p curand() to
  -- * get enough random bits to create the return value.  The current
  -- * implementation uses exactly two calls.
  -- *
  -- * \param state - Pointer to state to update
  -- *
  -- * \return uniformly distributed double between \p 0.0 and \p 1.0
  --  

  --*
  -- * \brief Return a uniformly distributed float from an MRG32k3a generator.
  -- *
  -- * Return a uniformly distributed float between \p 0.0f and \p 1.0f 
  -- * from the MRG32k3a generator in \p state, increment position of generator.
  -- * Output range excludes \p 0.0f but includes \p 1.0f.  Denormalized floating
  -- * point outputs are never returned.
  -- *
  -- * The implementation returns up to 23 bits of mantissa, with the minimum 
  -- * return value \f$ 2^{-32} \f$ 
  -- *
  -- * \param state - Pointer to state to update
  -- *
  -- * \return uniformly distributed float between \p 0.0f and \p 1.0f
  --  

  --*
  -- * \brief Return a uniformly distributed double from an MRG32k3a generator.
  -- *
  -- * Return a uniformly distributed double between \p 0.0 and \p 1.0 
  -- * from the MRG32k3a generator in \p state, increment position of generator.
  -- * Output range excludes \p 0.0 but includes \p 1.0.  Denormalized floating
  -- * point outputs are never returned. 
  -- *
  -- * Note the implementation returns at most 32 random bits of mantissa as 
  -- * outlined in the seminal paper by L'Ecuyer.
  -- *
  -- * \param state - Pointer to state to update
  -- *
  -- * \return uniformly distributed double between \p 0.0 and \p 1.0
  --  

  --*
  -- * \brief Return a uniformly distributed tuple of 2 doubles from an Philox4_32_10 generator.
  -- *
  -- * Return a uniformly distributed 2 doubles (double4) between \p 0.0 and \p 1.0 
  -- * from the Philox4_32_10 generator in \p state, increment position of generator by 4.
  -- * Output range excludes \p 0.0 but includes \p 1.0.  Denormalized floating
  -- * point outputs are never returned. 
  -- *
  -- * \param state - Pointer to state to update
  -- *
  -- * \return 2 uniformly distributed doubles between \p 0.0 and \p 1.0
  --  

   function curand_uniform2_double (state : access curand_philox4x32_x_h.curandStatePhilox4_32_10_t) return vector_types_h.double2;  -- /usr/local/cuda-8.0/include/curand_uniform.h:225
   pragma Import (CPP, curand_uniform2_double, "_ZL22curand_uniform2_doubleP24curandStatePhilox4_32_10");

  -- not a part of API
   function curand_uniform4_double (state : access curand_philox4x32_x_h.curandStatePhilox4_32_10_t) return vector_types_h.double4;  -- /usr/local/cuda-8.0/include/curand_uniform.h:237
   pragma Import (CPP, curand_uniform4_double, "_ZL22curand_uniform4_doubleP24curandStatePhilox4_32_10");

  --*
  -- * \brief Return a uniformly distributed float from a Philox4_32_10 generator.
  -- *
  -- * Return a uniformly distributed float between \p 0.0f and \p 1.0f 
  -- * from the Philox4_32_10 generator in \p state, increment position of generator.
  -- * Output range excludes \p 0.0f but includes \p 1.0f.  Denormalized floating
  -- * point outputs are never returned.
  -- * 
  -- * \param state - Pointer to state to update
  -- *
  -- * \return uniformly distributed float between \p 0.0 and \p 1.0
  --  

  --*
  -- * \brief Return a uniformly distributed tuple of 4 floats from a Philox4_32_10 generator.
  -- *
  -- * Return a uniformly distributed 4 floats between \p 0.0f and \p 1.0f 
  -- * from the Philox4_32_10 generator in \p state, increment position of generator by 4.
  -- * Output range excludes \p 0.0f but includes \p 1.0f.  Denormalized floating
  -- * point outputs are never returned.
  -- * 
  -- * \param state - Pointer to state to update
  -- *
  -- * \return uniformly distributed float between \p 0.0 and \p 1.0
  --  

   function curand_uniform4 (state : access curand_philox4x32_x_h.curandStatePhilox4_32_10_t) return vector_types_h.float4;  -- /usr/local/cuda-8.0/include/curand_uniform.h:282
   pragma Import (CPP, curand_uniform4, "_ZL15curand_uniform4P24curandStatePhilox4_32_10");

  --*
  -- * \brief Return a uniformly distributed float from a MTGP32 generator.
  -- *
  -- * Return a uniformly distributed float between \p 0.0f and \p 1.0f 
  -- * from the MTGP32 generator in \p state, increment position of generator.
  -- * Output range excludes \p 0.0f but includes \p 1.0f.  Denormalized floating
  -- * point outputs are never returned.
  -- *
  -- * \param state - Pointer to state to update
  -- *
  -- * \return uniformly distributed float between \p 0.0f and \p 1.0f
  --  

  --*
  -- * \brief Return a uniformly distributed double from a MTGP32 generator.
  -- *
  -- * Return a uniformly distributed double between \p 0.0f and \p 1.0f 
  -- * from the MTGP32 generator in \p state, increment position of generator.
  -- * Output range excludes \p 0.0f but includes \p 1.0f.  Denormalized floating
  -- * point outputs are never returned.
  -- *
  -- * \param state - Pointer to state to update
  -- *
  -- * \return uniformly distributed double between \p 0.0f and \p 1.0f
  --  

  --*
  -- * \brief Return a uniformly distributed double from a Philox4_32_10 generator.
  -- *
  -- * Return a uniformly distributed double between \p 0.0f and \p 1.0f 
  -- * from the Philox4_32_10 generator in \p state, increment position of generator.
  -- * Output range excludes \p 0.0f but includes \p 1.0f.  Denormalized floating
  -- * point outputs are never returned.
  -- *
  -- * \param state - Pointer to state to update
  -- *
  -- * \return uniformly distributed double between \p 0.0f and \p 1.0f
  --  

  --*
  -- * \brief Return a uniformly distributed float from a Sobol32 generator.
  -- *
  -- * Return a uniformly distributed float between \p 0.0f and \p 1.0f 
  -- * from the Sobol32 generator in \p state, increment position of generator.
  -- * Output range excludes \p 0.0f but includes \p 1.0f.  Denormalized floating
  -- * point outputs are never returned.
  -- *
  -- * The implementation is guaranteed to use a single call to \p curand().
  -- *
  -- * \param state - Pointer to state to update
  -- *
  -- * \return uniformly distributed float between \p 0.0f and \p 1.0f
  --  

  --*
  -- * \brief Return a uniformly distributed double from a Sobol32 generator.
  -- *
  -- * Return a uniformly distributed double between \p 0.0 and \p 1.0 
  -- * from the Sobol32 generator in \p state, increment position of generator.
  -- * Output range excludes \p 0.0 but includes \p 1.0.  Denormalized floating
  -- * point outputs are never returned.
  -- *
  -- * The implementation is guaranteed to use a single call to \p curand()
  -- * to preserve the quasirandom properties of the sequence.
  -- *
  -- * \param state - Pointer to state to update
  -- *
  -- * \return uniformly distributed double between \p 0.0 and \p 1.0
  --  

  --*
  -- * \brief Return a uniformly distributed float from a scrambled Sobol32 generator.
  -- *
  -- * Return a uniformly distributed float between \p 0.0f and \p 1.0f 
  -- * from the scrambled Sobol32 generator in \p state, increment position of generator.
  -- * Output range excludes \p 0.0f but includes \p 1.0f.  Denormalized floating
  -- * point outputs are never returned.
  -- *
  -- * The implementation is guaranteed to use a single call to \p curand().
  -- *
  -- * \param state - Pointer to state to update
  -- *
  -- * \return uniformly distributed float between \p 0.0f and \p 1.0f
  --  

  --*
  -- * \brief Return a uniformly distributed double from a scrambled Sobol32 generator.
  -- *
  -- * Return a uniformly distributed double between \p 0.0 and \p 1.0 
  -- * from the scrambled Sobol32 generator in \p state, increment position of generator.
  -- * Output range excludes \p 0.0 but includes \p 1.0.  Denormalized floating
  -- * point outputs are never returned.
  -- *
  -- * The implementation is guaranteed to use a single call to \p curand()
  -- * to preserve the quasirandom properties of the sequence.
  -- *
  -- * \param state - Pointer to state to update
  -- *
  -- * \return uniformly distributed double between \p 0.0 and \p 1.0
  --  

  --*
  -- * \brief Return a uniformly distributed float from a Sobol64 generator.
  -- *
  -- * Return a uniformly distributed float between \p 0.0f and \p 1.0f 
  -- * from the Sobol64 generator in \p state, increment position of generator.
  -- * Output range excludes \p 0.0f but includes \p 1.0f.  Denormalized floating
  -- * point outputs are never returned.
  -- *
  -- * The implementation is guaranteed to use a single call to \p curand().
  -- *
  -- * \param state - Pointer to state to update
  -- *
  -- * \return uniformly distributed float between \p 0.0f and \p 1.0f
  --  

  --*
  -- * \brief Return a uniformly distributed double from a Sobol64 generator.
  -- *
  -- * Return a uniformly distributed double between \p 0.0 and \p 1.0 
  -- * from the Sobol64 generator in \p state, increment position of generator.
  -- * Output range excludes \p 0.0 but includes \p 1.0.  Denormalized floating
  -- * point outputs are never returned.
  -- *
  -- * The implementation is guaranteed to use a single call to \p curand()
  -- * to preserve the quasirandom properties of the sequence.
  -- *
  -- * \param state - Pointer to state to update
  -- *
  -- * \return uniformly distributed double between \p 0.0 and \p 1.0
  --  

  --*
  -- * \brief Return a uniformly distributed float from a scrambled Sobol64 generator.
  -- *
  -- * Return a uniformly distributed float between \p 0.0f and \p 1.0f 
  -- * from the scrambled Sobol64 generator in \p state, increment position of generator.
  -- * Output range excludes \p 0.0f but includes \p 1.0f.  Denormalized floating
  -- * point outputs are never returned.
  -- *
  -- * The implementation is guaranteed to use a single call to \p curand().
  -- *
  -- * \param state - Pointer to state to update
  -- *
  -- * \return uniformly distributed float between \p 0.0f and \p 1.0f
  --  

  --*
  -- * \brief Return a uniformly distributed double from a scrambled Sobol64 generator.
  -- *
  -- * Return a uniformly distributed double between \p 0.0 and \p 1.0 
  -- * from the scrambled Sobol64 generator in \p state, increment position of generator.
  -- * Output range excludes \p 0.0 but includes \p 1.0.  Denormalized floating
  -- * point outputs are never returned.
  -- *
  -- * The implementation is guaranteed to use a single call to \p curand()
  -- * to preserve the quasirandom properties of the sequence.
  -- *
  -- * \param state - Pointer to state to update
  -- *
  -- * \return uniformly distributed double between \p 0.0 and \p 1.0
  --  

end curand_uniform_h;

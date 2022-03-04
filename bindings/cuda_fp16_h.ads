pragma Ada_2005;
pragma Style_Checks (Off);

with Interfaces.C; use Interfaces.C;

package cuda_fp16_h is

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

  --*
  -- * \defgroup CUDA_MATH_INTRINSIC_HALF Half Precision Intrinsics
  -- * This section describes half precision intrinsic functions that are
  -- * only supported in device code.
  --  

  --*
  -- * \defgroup CUDA_MATH__HALF_ARITHMETIC Half Arithmetic Functions
  -- * \ingroup CUDA_MATH_INTRINSIC_HALF
  --  

  --*
  -- * \defgroup CUDA_MATH__HALF2_ARITHMETIC Half2 Arithmetic Functions
  -- * \ingroup CUDA_MATH_INTRINSIC_HALF
  --  

  --*
  -- * \defgroup CUDA_MATH__HALF_COMPARISON Half Comparison Functions
  -- * \ingroup CUDA_MATH_INTRINSIC_HALF
  --  

  --*
  -- * \defgroup CUDA_MATH__HALF2_COMPARISON Half2 Comparison Functions
  -- * \ingroup CUDA_MATH_INTRINSIC_HALF
  --  

  --*
  -- * \defgroup CUDA_MATH__HALF_MISC Half Precision Conversion And Data Movement
  -- * \ingroup CUDA_MATH_INTRINSIC_HALF
  --  

  --*
  -- * \defgroup CUDA_MATH__HALF_FUNCTIONS Half Math Functions
  -- * \ingroup CUDA_MATH_INTRINSIC_HALF
  --  

  --*
  -- * \defgroup CUDA_MATH__HALF2_FUNCTIONS Half2 Math Functions
  -- * \ingroup CUDA_MATH_INTRINSIC_HALF
  --  

   type uu_half is record
      x : aliased unsigned_short;  -- /usr/local/cuda-8.0/include/cuda_fp16.h:95
   end record;
   pragma Convention (C_Pass_By_Copy, uu_half);  -- /usr/local/cuda-8.0/include/cuda_fp16.h:96

   --  skipped anonymous struct anon_7

   type uu_half2 is record
      x : aliased unsigned;  -- /usr/local/cuda-8.0/include/cuda_fp16.h:99
   end record;
   pragma Convention (C_Pass_By_Copy, uu_half2);  -- /usr/local/cuda-8.0/include/cuda_fp16.h:100

   --  skipped anonymous struct anon_8

   subtype half is uu_half;

   subtype half2 is uu_half2;

  --*
  -- * \ingroup CUDA_MATH__HALF_MISC
  -- * \brief Converts float number to half precision in round-to-nearest-even mode
  -- * and returns \p half with converted value.
  -- *
  -- * Converts float number \p a to half precision in round-to-nearest-even mode.
  -- *
  -- * \return Returns \p half result with converted value.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_MISC
  -- * \brief Converts float number to half precision in round-towards-zero mode
  -- * and returns \p half with converted value.
  -- *
  -- * Converts float number \p a to half precision in round-towards-zero mode.
  -- *
  -- * \return Returns \p half result with converted value.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_MISC
  -- * \brief Converts float number to half precision in round-down mode
  -- * and returns \p half with converted value.
  -- *
  -- * Converts float number \p a to half precision in round-down mode.
  -- *
  -- * \return Returns \p half result with converted value.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_MISC
  -- * \brief Converts float number to half precision in round-up mode
  -- * and returns \p half with converted value.
  -- *
  -- * Converts float number \p a to half precision in round-up mode.
  -- *
  -- * \return Returns \p half result with converted value.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_MISC
  -- * \brief Converts \p half number to float.
  -- *
  -- * Converts half number \p a to float.
  -- *
  -- * \return Returns float result with converted value.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_MISC
  -- * \brief Convert a half to a signed integer in round-to-nearest-even mode.
  -- *
  -- * Convert the half-precision floating point value \p h to a signed integer in
  -- * round-to-nearest-even mode.
  -- *
  -- * \return Returns converted value.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_MISC
  -- * \brief Convert a half to a signed integer in round-towards-zero mode.
  -- *
  -- * Convert the half-precision floating point value \p h to a signed integer in
  -- * round-towards-zero mode.
  -- *
  -- * \return Returns converted value.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_MISC
  -- * \brief Convert a half to a signed integer in round-down mode.
  -- *
  -- * Convert the half-precision floating point value \p h to a signed integer in
  -- * round-down mode.
  -- *
  -- * \return Returns converted value.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_MISC
  -- * \brief Convert a half to a signed integer in round-up mode.
  -- *
  -- * Convert the half-precision floating point value \p h to a signed integer in
  -- * round-up mode.
  -- *
  -- * \return Returns converted value.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_MISC
  -- * \brief Convert a signed integer to a half in round-to-nearest-even mode.
  -- *
  -- * Convert the signed integer value \p i to a half-precision floating point
  -- * value in round-to-nearest-even mode.
  -- *
  -- * \return Returns converted value.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_MISC
  -- * \brief Convert a signed integer to a half in round-towards-zero mode.
  -- *
  -- * Convert the signed integer value \p i to a half-precision floating point
  -- * value in round-towards-zero mode.
  -- *
  -- * \return Returns converted value.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_MISC
  -- * \brief Convert a signed integer to a half in round-down mode.
  -- *
  -- * Convert the signed integer value \p i to a half-precision floating point
  -- * value in round-down mode.
  -- *
  -- * \return Returns converted value.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_MISC
  -- * \brief Convert a signed integer to a half in round-up mode.
  -- *
  -- * Convert the signed integer value \p i to a half-precision floating point
  -- * value in round-up mode.
  -- *
  -- * \return Returns converted value.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_MISC
  -- * \brief Convert a half to a signed short integer in round-to-nearest-even
  -- * mode.
  -- *
  -- * Convert the half-precision floating point value \p h to a signed short
  -- * integer in round-to-nearest-even mode.
  -- *
  -- * \return Returns converted value.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_MISC
  -- * \brief Convert a half to a signed short integer in round-towards-zero mode.
  -- *
  -- * Convert the half-precision floating point value \p h to a signed short
  -- * integer in round-towards-zero mode.
  -- *
  -- * \return Returns converted value.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_MISC
  -- * \brief Convert a half to a signed short integer in round-down mode.
  -- *
  -- * Convert the half-precision floating point value \p h to a signed short
  -- * integer in round-down mode.
  -- *
  -- * \return Returns converted value.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_MISC
  -- * \brief Convert a half to a signed short integer in round-up mode.
  -- *
  -- * Convert the half-precision floating point value \p h to a signed short
  -- * integer in round-up mode.
  -- *
  -- * \return Returns converted value.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_MISC
  -- * \brief Convert a signed short integer to a half in round-to-nearest-even
  -- * mode.
  -- *
  -- * Convert the signed short integer value \p i to a half-precision floating
  -- * point value in round-to-nearest-even mode.
  -- *
  -- * \return Returns converted value.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_MISC
  -- * \brief Convert a signed short integer to a half in round-towards-zero mode.
  -- *
  -- * Convert the signed short integer value \p i to a half-precision floating
  -- * point value in round-towards-zero mode.
  -- *
  -- * \return Returns converted value.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_MISC
  -- * \brief Convert a signed short integer to a half in round-down mode.
  -- *
  -- * Convert the signed short integer value \p i to a half-precision floating
  -- * point value in round-down mode.
  -- *
  -- * \return Returns converted value.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_MISC
  -- * \brief Convert a signed short integer to a half in round-up mode.
  -- *
  -- * Convert the signed short integer value \p i to a half-precision floating
  -- * point value in round-up mode.
  -- *
  -- * \return Returns converted value.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_MISC
  -- * \brief Convert a half to an unsigned integer in round-to-nearest-even mode.
  -- *
  -- * Convert the half-precision floating point value \p h to an unsigned integer
  -- * in round-to-nearest-even mode.
  -- *
  -- * \return Returns converted value.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_MISC
  -- * \brief Convert a half to an unsigned integer in round-towards-zero mode.
  -- *
  -- * Convert the half-precision floating point value \p h to an unsigned integer
  -- * in round-towards-zero mode.
  -- *
  -- * \return Returns converted value.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_MISC
  -- * \brief Convert a half to an unsigned integer in round-down mode.
  -- *
  -- * Convert the half-precision floating point value \p h to an unsigned integer
  -- * in round-down mode.
  -- *
  -- * \return Returns converted value.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_MISC
  -- * \brief Convert a half to an unsigned integer in round-up mode.
  -- *
  -- * Convert the half-precision floating point value \p h to an unsigned integer
  -- * in round-up mode.
  -- *
  -- * \return Returns converted value.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_MISC
  -- * \brief Convert an unsigned integer to a half in round-to-nearest-even mode.
  -- *
  -- * Convert the unsigned integer value \p i to a half-precision floating point
  -- * value in round-to-nearest-even mode.
  -- *
  -- * \return Returns converted value.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_MISC
  -- * \brief Convert an unsigned integer to a half in round-towards-zero mode.
  -- *
  -- * Convert the unsigned integer value \p i to a half-precision floating point
  -- * value in round-towards-zero mode.
  -- *
  -- * \return Returns converted value.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_MISC
  -- * \brief Convert an unsigned integer to a half in round-down mode.
  -- *
  -- * Convert the unsigned integer value \p i to a half-precision floating point
  -- * value in round-down mode.
  -- *
  -- * \return Returns converted value.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_MISC
  -- * \brief Convert an unsigned integer to a half in round-up mode.
  -- *
  -- * Convert the unsigned integer value \p i to a half-precision floating point
  -- * value in round-up mode.
  -- *
  -- * \return Returns converted value.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_MISC
  -- * \brief Convert a half to an unsigned short integer in round-to-nearest-even
  -- * mode.
  -- *
  -- * Convert the half-precision floating point value \p h to an unsigned short
  -- * integer in round-to-nearest-even mode.
  -- *
  -- * \return Returns converted value.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_MISC
  -- * \brief Convert a half to an unsigned short integer in round-towards-zero
  -- * mode.
  -- *
  -- * Convert the half-precision floating point value \p h to an unsigned short
  -- * integer in round-towards-zero mode.
  -- *
  -- * \return Returns converted value.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_MISC
  -- * \brief Convert a half to an unsigned short integer in round-down mode.
  -- *
  -- * Convert the half-precision floating point value \p h to an unsigned short
  -- * integer in round-down mode.
  -- *
  -- * \return Returns converted value.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_MISC
  -- * \brief Convert a half to an unsigned short integer in round-up mode.
  -- *
  -- * Convert the half-precision floating point value \p h to an unsigned short
  -- * integer in round-up mode.
  -- *
  -- * \return Returns converted value.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_MISC
  -- * \brief Convert an unsigned short integer to a half in round-to-nearest-even
  -- * mode.
  -- *
  -- * Convert the unsigned short integer value \p i to a half-precision floating
  -- * point value in round-to-nearest-even mode.
  -- *
  -- * \return Returns converted value.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_MISC
  -- * \brief Convert an unsigned short integer to a half in round-towards-zero
  -- * mode.
  -- *
  -- * Convert the unsigned short integer value \p i to a half-precision floating
  -- * point value in round-towards-zero mode.
  -- *
  -- * \return Returns converted value.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_MISC
  -- * \brief Convert an unsigned short integer to a half in round-down mode.
  -- *
  -- * Convert the unsigned short integer value \p i to a half-precision floating
  -- * point value in round-down mode.
  -- *
  -- * \return Returns converted value.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_MISC
  -- * \brief Convert an unsigned short integer to a half in round-up mode.
  -- *
  -- * Convert the unsigned short integer value \p i to a half-precision floating
  -- * point value in round-up mode.
  -- *
  -- * \return Returns converted value.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_MISC
  -- * \brief Convert a half to an unsigned 64-bit integer in round-to-nearest-even
  -- * mode.
  -- *
  -- * Convert the half-precision floating point value \p h to an unsigned 64-bit
  -- * integer in round-to-nearest-even mode.
  -- *
  -- * \return Returns converted value.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_MISC
  -- * \brief Convert a half to an unsigned 64-bit integer in round-towards-zero
  -- * mode.
  -- *
  -- * Convert the half-precision floating point value \p h to an unsigned 64-bit
  -- * integer in round-towards-zero mode.
  -- *
  -- * \return Returns converted value.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_MISC
  -- * \brief Convert a half to an unsigned 64-bit integer in round-down mode.
  -- *
  -- * Convert the half-precision floating point value \p h to an unsigned 64-bit
  -- * integer in round-down mode.
  -- *
  -- * \return Returns converted value.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_MISC
  -- * \brief Convert a half to an unsigned 64-bit integer in round-up mode.
  -- *
  -- * Convert the half-precision floating point value \p h to an unsigned 64-bit
  -- * integer in round-up mode.
  -- *
  -- * \return Returns converted value.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_MISC
  -- * \brief Convert an unsigned 64-bit integer to a half in round-to-nearest-even
  -- * mode.
  -- *
  -- * Convert the unsigned 64-bit integer value \p i to a half-precision floating
  -- * point value in round-to-nearest-even mode.
  -- *
  -- * \return Returns converted value.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_MISC
  -- * \brief Convert an unsigned 64-bit integer to a half in round-towards-zero
  -- * mode.
  -- *
  -- * Convert the unsigned 64-bit integer value \p i to a half-precision floating
  -- * point value in round-towards-zero mode.
  -- *
  -- * \return Returns converted value.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_MISC
  -- * \brief Convert an unsigned 64-bit integer to a half in round-down mode.
  -- *
  -- * Convert the unsigned 64-bit integer value \p i to a half-precision floating
  -- * point value in round-down mode.
  -- *
  -- * \return Returns converted value.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_MISC
  -- * \brief Convert an unsigned 64-bit integer to a half in round-up mode.
  -- *
  -- * Convert the unsigned 64-bit integer value \p i to a half-precision floating
  -- * point value in round-up mode.
  -- *
  -- * \return Returns converted value.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_MISC
  -- * \brief Convert a half to a signed 64-bit integer in round-to-nearest-even
  -- * mode.
  -- *
  -- * Convert the half-precision floating point value \p h to a signed 64-bit
  -- * integer in round-to-nearest-even mode.
  -- *
  -- * \return Returns converted value.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_MISC
  -- * \brief Convert a half to a signed 64-bit integer in round-towards-zero mode.
  -- *
  -- * Convert the half-precision floating point value \p h to a signed 64-bit
  -- * integer in round-towards-zero mode.
  -- *
  -- * \return Returns converted value.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_MISC
  -- * \brief Convert a half to a signed 64-bit integer in round-down mode.
  -- *
  -- * Convert the half-precision floating point value \p h to a signed 64-bit
  -- * integer in round-down mode.
  -- *
  -- * \return Returns converted value.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_MISC
  -- * \brief Convert a half to a signed 64-bit integer in round-up mode.
  -- *
  -- * Convert the half-precision floating point value \p h to a signed 64-bit
  -- * integer in round-up mode.
  -- *
  -- * \return Returns converted value.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_MISC
  -- * \brief Convert a signed 64-bit integer to a half in round-to-nearest-even
  -- * mode.
  -- *
  -- * Convert the signed 64-bit integer value \p i to a half-precision floating
  -- * point value in round-to-nearest-even mode.
  -- *
  -- * \return Returns converted value.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_MISC
  -- * \brief Convert a signed 64-bit integer to a half in round-towards-zero mode.
  -- *
  -- * Convert the signed 64-bit integer value \p i to a half-precision floating
  -- * point value in round-towards-zero mode.
  -- *
  -- * \return Returns converted value.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_MISC
  -- * \brief Convert a signed 64-bit integer to a half in round-down mode.
  -- *
  -- * Convert the signed 64-bit integer value \p i to a half-precision floating
  -- * point value in round-down mode.
  -- *
  -- * \return Returns converted value.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_MISC
  -- * \brief Convert a signed 64-bit integer to a half in round-up mode.
  -- *
  -- * Convert the signed 64-bit integer value \p i to a half-precision floating
  -- * point value in round-up mode.
  -- *
  -- * \return Returns converted value.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_FUNCTIONS
  -- * \brief Truncate input argument to the integral part.
  -- *
  -- * Round \p h to the nearest integer value that does not exceed \p h in
  -- * magnitude.
  -- *
  -- * \return Returns truncated integer value.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_FUNCTIONS
  -- * \brief Calculate ceiling of the input argument.
  -- *
  -- * Compute the smallest integer value not less than \p h.
  -- *
  -- * \return Returns ceiling expressed as a half-precision floating point number.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_FUNCTIONS
  -- * \brief Calculate the largest integer less than or equal to \p h.
  -- *
  -- * Calculate the largest integer value which is less than or equal to \p h.
  -- *
  -- * \return Returns floor expressed as half-precision floating point number.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_FUNCTIONS
  -- * \brief Round input to nearest integer value in half-precision floating point
  -- * number.
  -- *
  -- * Round \p h to the nearest integer value in half-precision floating point
  -- * format, with halfway cases rounded to the nearest even integer value.
  -- *
  -- * \return Returns rounded integer value expressed as half-precision floating
  -- * point number.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF2_FUNCTIONS
  -- * \brief Truncate \p half2 vector input argument to the integral part.
  -- *
  -- * Round each component of vector \p h to the nearest integer value that does
  -- * not exceed \p h in magnitude.
  -- *
  -- * \return Returns \p half2 vector truncated integer value.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF2_FUNCTIONS
  -- * \brief Calculate \p half2 vector ceiling of the input argument.
  -- *
  -- * For each component of vector \p h compute the smallest integer value not less
  -- * than \p h.
  -- *
  -- * \return Returns \p half2 vector ceiling expressed as a pair of half-precision
  -- * floating point numbers.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF2_FUNCTIONS
  -- * \brief Calculate the largest integer less than or equal to \p h.
  -- *
  -- * For each component of vector \p h calculate the largest integer value which
  -- * is less than or equal to \p h.
  -- *
  -- * \return Returns \p half2 vector floor expressed as a pair of half-precision
  -- * floating point number.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF2_FUNCTIONS
  -- * \brief Round input to nearest integer value in half-precision floating point
  -- * number.
  -- *
  -- * Round each component of \p half2 vector \p h to the nearest integer value in
  -- * half-precision floating point format, with halfway cases rounded to the
  -- * nearest even integer value.
  -- *
  -- * \return Returns \p half2 vector of rounded integer values expressed as
  -- * half-precision floating point numbers.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_MISC
  -- * \brief Converts input to half precision in round-to-nearest-even mode and
  -- * populates both halves of \p half2 with converted value.
  -- *
  -- * Converts input \p a to half precision in round-to-nearest-even mode and
  -- * populates both halves of \p half2 with converted value.
  -- *
  -- * \return Returns \p half2 with both halves equal to the converted half
  -- * precision number.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_MISC
  -- * \brief Converts both input floats to half precision in round-to-nearest-even
  -- * mode and returns \p half2 with converted values.
  -- *
  -- * Converts both input floats to half precision in round-to-nearest-even mode
  -- * and combines the results into one \p half2 number. Low 16 bits of the return
  -- * value correspond to the input \p a, high 16 bits correspond to the input \p
  -- * b.
  -- *
  -- * \return Returns \p half2 which has corresponding halves equal to the
  -- * converted input floats.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_MISC
  -- * \brief Converts both components of float2 number to half precision in
  -- * round-to-nearest-even mode and returns \p half2 with converted values.
  -- *
  -- * Converts both components of float2 to half precision in round-to-nearest
  -- * mode and combines the results into one \p half2 number. Low 16 bits of the
  -- * return value correspond to \p a.x and high 16 bits of the return value
  -- * correspond to \p a.y.
  -- *
  -- * \return Returns \p half2 which has corresponding halves equal to the
  -- * converted float2 components.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_MISC
  -- * \brief Converts both halves of \p half2 to float2 and returns the result.
  -- *
  -- * Converts both halves of \p half2 input \p a to float2 and returns the
  -- * result.
  -- *
  -- * \return Returns converted float2.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_MISC
  -- * \brief Converts low 16 bits of \p half2 to float and returns the result
  -- *
  -- * Converts low 16 bits of \p half2 input \p a to 32 bit floating point number
  -- * and returns the result.
  -- *
  -- * \return Returns low 16 bits of \p a converted to float.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_MISC
  -- * \brief Returns \p half2 with both halves equal to the input value.
  -- *
  -- * Returns \p half2 number with both halves equal to the input \p a \p half
  -- * number.
  -- *
  -- * \return Returns \p half2 with both halves equal to the input \p a.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_MISC
  -- * \brief Converts high 16 bits of \p half2 to float and returns the result
  -- *
  -- * Converts high 16 bits of \p half2 input \p a to 32 bit floating point number
  -- * and returns the result.
  -- *
  -- * \return Returns high 16 bits of \p a converted to float.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_MISC
  -- * \brief Swaps both halves of the \p half2 input.
  -- *
  -- * Swaps both halves of the \p half2 input and returns a new \p half2 number
  -- * with swapped halves.
  -- *
  -- * \return Returns \p half2 with halves swapped.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_MISC
  -- * \brief Extracts low 16 bits from each of the two \p half2 inputs and combines
  -- * into one \p half2 number.
  -- *
  -- * Extracts low 16 bits from each of the two \p half2 inputs and combines into
  -- * one \p half2 number. Low 16 bits from input \p a is stored in low 16 bits of
  -- * the return value, low 16 bits from input \p b is stored in high 16 bits of
  -- * the return value.
  -- *
  -- * \return Returns \p half2 which contains low 16 bits from \p a and \p b.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_MISC
  -- * \brief Extracts high 16 bits from each of the two \p half2 inputs and
  -- * combines into one \p half2 number.
  -- *
  -- * Extracts high 16 bits from each of the two \p half2 inputs and combines into
  -- * one \p half2 number. High 16 bits from input \p a is stored in low 16 bits of
  -- * the return value, high 16 bits from input \p b is stored in high 16 bits of
  -- * the return value.
  -- *
  -- * \return Returns \p half2 which contains high 16 bits from \p a and \p b.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_MISC
  -- * \brief Returns high 16 bits of \p half2 input.
  -- *
  -- * Returns high 16 bits of \p half2 input \p a.
  -- *
  -- * \return Returns \p half which contains high 16 bits of the input.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_MISC
  -- * \brief Returns low 16 bits of \p half2 input.
  -- *
  -- * Returns low 16 bits of \p half2 input \p a.
  -- *
  -- * \return Returns \p half which contains low 16 bits of the input.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_COMPARISON
  -- * \brief Checks if the input \p half number is infinite.
  -- *
  -- * Checks if the input \p half number \p a is infinite.
  -- *
  -- * \return Returns -1 iff \p a is equal to negative infinity, 1 iff \p a is
  -- * equal to positive infinity and 0 otherwise.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_MISC
  -- * \brief Combines two \p half numbers into one \p half2 number.
  -- *
  -- * Combines two input \p half number \p a and \p b into one \p half2 number.
  -- * Input \p a is stored in low 16 bits of the return value, input \p b is stored
  -- * in high 16 bits of the return value.
  -- *
  -- * \return Returns \p half2 number which has one half equal to \p a and the
  -- * other to \p b.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_MISC
  -- * \brief Extracts low 16 bits from \p half2 input.
  -- *
  -- * Extracts low 16 bits from \p half2 input \p a and returns a new \p half2
  -- * number which has both halves equal to the extracted bits.
  -- *
  -- * \return Returns \p half2 with both halves equal to low 16 bits from the
  -- * input.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_MISC
  -- * \brief Extracts high 16 bits from \p half2 input.
  -- *
  -- * Extracts high 16 bits from \p half2 input \p a and returns a new \p half2
  -- * number which has both halves equal to the extracted bits.
  -- *
  -- * \return Returns \p half2 with both halves equal to high 16 bits from the
  -- * input.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_MISC
  -- * \brief Reinterprets bits in a \p half as a signed short integer.
  -- *
  -- * Reinterprets the bits in the half-precision floating point value \p h
  -- * as a signed short integer.
  -- *
  -- * \return Returns reinterpreted value.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_MISC
  -- * \brief Reinterprets bits in a \p half as an unsigned short integer.
  -- *
  -- * Reinterprets the bits in the half-precision floating point value \p h
  -- * as an unsigned short integer.
  -- *
  -- * \return Returns reinterpreted value.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_MISC
  -- * \brief Reinterprets bits in a signed short integer as a \p half.
  -- *
  -- * Reinterprets the bits in the signed short integer value \p i as a
  -- * half-precision floating point value.
  -- *
  -- * \return Returns reinterpreted value.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_MISC
  -- * \brief Reinterprets bits in an unsigned short integer as a \p half.
  -- *
  -- * Reinterprets the bits in the unsigned short integer value \p i as a
  -- * half-precision floating point value.
  -- *
  -- * \return Returns reinterpreted value.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF2_COMPARISON
  -- * \brief Performs half2 vector if-equal comparison.
  -- *
  -- * Performs \p half2 vector if-equal comparison of inputs \p a and \p b.
  -- * The corresponding \p half results are set to 1.0 for true, or 0.0 for false.
  -- * NaN inputs generate false results.
  -- *
  -- * \return Returns the \p half2 vector result of if-equal comparison of vectors
  -- * \p a and \p b.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF2_COMPARISON
  -- * \brief Performs \p half2 vector not-equal comparison.
  -- *
  -- * Performs \p half2 vector not-equal comparison of inputs \p a and \p b.
  -- * The corresponding \p half results are set to 1.0 for true, or 0.0 for false.
  -- * NaN inputs generate false results.
  -- *
  -- * \return Returns the \p half2 vector result of not-equal comparison of vectors
  -- * \p a and \p b.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF2_COMPARISON
  -- * \brief Performs \p half2 vector less-equal comparison.
  -- *
  -- * Performs \p half2 vector less-equal comparison of inputs \p a and \p b.
  -- * The corresponding \p half results are set to 1.0 for true, or 0.0 for false.
  -- * NaN inputs generate false results.
  -- *
  -- * \return Returns the \p half2 vector result of less-equal comparison of
  -- * vectors \p a and \p b.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF2_COMPARISON
  -- * \brief Performs \p half2 vector greater-equal comparison.
  -- *
  -- * Performs \p half2 vector greater-equal comparison of inputs \p a and \p b.
  -- * The corresponding \p half results are set to 1.0 for true, or 0.0 for false.
  -- * NaN inputs generate false results.
  -- *
  -- * \return Returns the \p half2 vector result of greater-equal comparison of
  -- * vectors \p a and \p b.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF2_COMPARISON
  -- * \brief Performs \p half2 vector less-than comparison.
  -- *
  -- * Performs \p half2 vector less-than comparison of inputs \p a and \p b.
  -- * The corresponding \p half results are set to 1.0 for true, or 0.0 for false.
  -- * NaN inputs generate false results.
  -- *
  -- * \return Returns the \p half2 vector result of less-than comparison of vectors
  -- * \p a and \p b.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF2_COMPARISON
  -- * \brief Performs \p half2 vector greater-than comparison.
  -- *
  -- * Performs \p half2 vector greater-than comparison of inputs \p a and \p b.
  -- * The corresponding \p half results are set to 1.0 for true, or 0.0 for false.
  -- * NaN inputs generate false results.
  -- *
  -- * \return Returns the half2 vector result of greater-than comparison of vectors
  -- * \p a and \p b.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF2_COMPARISON
  -- * \brief Performs \p half2 vector unordered if-equal comparison.
  -- *
  -- * Performs \p half2 vector if-equal comparison of inputs \p a and \p b.
  -- * The corresponding \p half results are set to 1.0 for true, or 0.0 for false.
  -- * NaN inputs generate true results.
  -- *
  -- * \return Returns the \p half2 vector result of unordered if-equal comparison
  -- * of vectors \p a and \p b.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF2_COMPARISON
  -- * \brief Performs \p half2 vector unordered not-equal comparison.
  -- *
  -- * Performs \p half2 vector not-equal comparison of inputs \p a and \p b.
  -- * The corresponding \p half results are set to 1.0 for true, or 0.0 for false.
  -- * NaN inputs generate true results.
  -- *
  -- * \return Returns the \p half2 vector result of unordered not-equal comparison
  -- * of vectors \p a and \p b.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF2_COMPARISON
  -- * \brief Performs \p half2 vector unordered less-equal comparison.
  -- *
  -- * Performs \p half2 vector less-equal comparison of inputs \p a and \p b.
  -- * The corresponding \p half results are set to 1.0 for true, or 0.0 for false.
  -- * NaN inputs generate true results.
  -- *
  -- * \return Returns the \p half2 vector result of unordered less-equal comparison
  -- * of vectors \p a and \p b.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF2_COMPARISON
  -- * \brief Performs \p half2 vector unordered greater-equal comparison.
  -- *
  -- * Performs \p half2 vector greater-equal comparison of inputs \p a and \p b.
  -- * The corresponding \p half results are set to 1.0 for true, or 0.0 for false.
  -- * NaN inputs generate true results.
  -- *
  -- * \return Returns the \p half2 vector result of unordered greater-equal
  -- * comparison of vectors \p a and \p b.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF2_COMPARISON
  -- * \brief Performs \p half2 vector unordered less-than comparison.
  -- *
  -- * Performs \p half2 vector less-than comparison of inputs \p a and \p b.
  -- * The corresponding \p half results are set to 1.0 for true, or 0.0 for false.
  -- * NaN inputs generate true results.
  -- *
  -- * \return Returns the \p half2 vector result of unordered less-than comparison
  -- * of vectors \p a and \p b.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF2_COMPARISON
  -- * \brief Performs \p half2 vector unordered greater-than comparison.
  -- *
  -- * Performs \p half2 vector greater-than comparison of inputs \p a and \p b.
  -- * The corresponding \p half results are set to 1.0 for true, or 0.0 for false.
  -- * NaN inputs generate true results.
  -- *
  -- * \return Returns the \p half2 vector result of unordered greater-than
  -- * comparison of vectors \p a and \p b.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF2_COMPARISON
  -- * \brief Determine whether \p half2 argument is a NaN.
  -- *
  -- * Determine whether each half of input \p half2 number \p a is a NaN.
  -- *
  -- * \return Returns \p half2 which has the corresponding \p half results set to
  -- * 1.0 for true, or 0.0 for false.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF2_ARITHMETIC
  -- * \brief Performs \p half2 vector addition in round-to-nearest-even mode.
  -- *
  -- * Performs \p half2 vector add of inputs \p a and \p b, in round-to-nearest
  -- * mode.
  -- *
  -- * \return Returns the \p half2 vector result of adding vectors \p a and \p b.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF2_ARITHMETIC
  -- * \brief Performs \p half2 vector subtraction in round-to-nearest-even mode.
  -- *
  -- * Subtracts \p half2 input vector \p b from input vector \p a in
  -- * round-to-nearest-even mode.
  -- *
  -- * \return Returns the \p half2 vector result of subtraction vector \p b from \p
  -- * a.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF2_ARITHMETIC
  -- * \brief Performs \p half2 vector multiplication in round-to-nearest-even mode.
  -- *
  -- * Performs \p half2 vector multiplication of inputs \p a and \p b, in
  -- * round-to-nearest-even mode.
  -- *
  -- * \return Returns the \p half2 vector result of multiplying vectors \p a and \p
  -- * b.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_ARITHMETIC
  -- * \brief Performs \p half2 vector division in round-to-nearest-even mode.
  -- *
  -- * Divides \p half2 input vector \p a by input vector \p b in round-to-nearest
  -- * mode.
  -- *
  -- * \return Returns the \p half2 vector result of division \p a by \p b.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF2_ARITHMETIC
  -- * \brief Performs \p half2 vector addition in round-to-nearest-even mode, with
  -- * saturation to [0.0, 1.0].
  -- *
  -- * Performs \p half2 vector add of inputs \p a and \p b, in round-to-nearest
  -- * mode, and clamps the results to range [0.0, 1.0]. NaN results are flushed to
  -- * +0.0.
  -- *
  -- * \return Returns the \p half2 vector result of adding vectors \p a and \p b
  -- * with saturation.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF2_ARITHMETIC
  -- * \brief Performs \p half2 vector subtraction in round-to-nearest-even mode,
  -- * with saturation to [0.0, 1.0].
  -- *
  -- * Subtracts \p half2 input vector \p b from input vector \p a in
  -- * round-to-nearest-even mode, and clamps the results to range [0.0, 1.0]. NaN
  -- * results are flushed to +0.0.
  -- *
  -- * \return Returns the \p half2 vector result of subtraction vector \p b from \p
  -- * a with saturation.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF2_ARITHMETIC
  -- * \brief Performs \p half2 vector multiplication in round-to-nearest-even mode,
  -- * with saturation to [0.0, 1.0].
  -- *
  -- * Performs \p half2 vector multiplication of inputs \p a and \p b, in
  -- * round-to-nearest-even mode, and clamps the results to range [0.0, 1.0]. NaN
  -- * results are flushed to +0.0.
  -- *
  -- * \return Returns the \p half2 vector result of multiplying vectors \p a and \p
  -- * b with saturation.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF2_ARITHMETIC
  -- * \brief Performs \p half2 vector fused multiply-add in round-to-nearest-even
  -- * mode.
  -- *
  -- * Performs \p half2 vector multiply on inputs \p a and \p b,
  -- * then performs a \p half2 vector add of the result with \p c,
  -- * rounding the result once in round-to-nearest-even mode.
  -- *
  -- * \return Returns the \p half2 vector result of the fused multiply-add
  -- * operation on vectors \p a, \p b, and \p c.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF2_ARITHMETIC
  -- * \brief Performs \p half2 vector fused multiply-add in round-to-nearest-even
  -- * mode, with saturation to [0.0, 1.0].
  -- *
  -- * Performs \p half2 vector multiply on inputs \p a and \p b,
  -- * then performs a \p half2 vector add of the result with \p c,
  -- * rounding the result once in round-to-nearest-even mode, and clamps the
  -- * results to range [0.0, 1.0]. NaN results are flushed to +0.0.
  -- *
  -- * \return Returns the \p half2 vector result of the fused multiply-add
  -- * operation on vectors \p a, \p b, and \p c with saturation.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF2_ARITHMETIC
  -- * \brief Negates both halves of the input \p half2 number and returns the
  -- * result.
  -- *
  -- * Negates both halves of the input \p half2 number \p a and returns the result.
  -- *
  -- * \return Returns \p half2 number with both halves negated.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_ARITHMETIC
  -- * \brief Performs \p half addition in round-to-nearest-even mode.
  -- *
  -- * Performs \p half addition of inputs \p a and \p b, in round-to-nearest-even
  -- * mode.
  -- *
  -- * \return Returns the \p half result of adding \p a and \p b.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_ARITHMETIC
  -- * \brief Performs \p half subtraction in round-to-nearest-even mode.
  -- *
  -- * Subtracts \p half input \p b from input \p a in round-to-nearest
  -- * mode.
  -- *
  -- * \return Returns the \p half result of subtraction \p b from \p a.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_ARITHMETIC
  -- * \brief Performs \p half multiplication in round-to-nearest-even mode.
  -- *
  -- * Performs \p half multiplication of inputs \p a and \p b, in round-to-nearest
  -- * mode.
  -- *
  -- * \return Returns the \p half result of multiplying \p a and \p b.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_ARITHMETIC
  -- * \brief Performs \p half division in round-to-nearest-even mode.
  -- *
  -- * Divides \p half input \p a by input \p b in round-to-nearest
  -- * mode.
  -- *
  -- * \return Returns the \p half result of division \p a by \p b.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_ARITHMETIC
  -- * \brief Performs \p half addition in round-to-nearest-even mode, with
  -- * saturation to [0.0, 1.0].
  -- *
  -- * Performs \p half add of inputs \p a and \p b, in round-to-nearest-even mode,
  -- * and clamps the result to range [0.0, 1.0]. NaN results are flushed to +0.0.
  -- *
  -- * \return Returns the \p half result of adding \p a and \p b with saturation.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_ARITHMETIC
  -- * \brief Performs \p half subtraction in round-to-nearest-even mode, with
  -- * saturation to [0.0, 1.0].
  -- *
  -- * Subtracts \p half input \p b from input \p a in round-to-nearest
  -- * mode,
  -- * and clamps the result to range [0.0, 1.0]. NaN results are flushed to +0.0.
  -- *
  -- * \return Returns the \p half result of subtraction \p b from \p a
  -- * with saturation.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_ARITHMETIC
  -- * \brief Performs \p half multiplication in round-to-nearest-even mode, with
  -- * saturation to [0.0, 1.0].
  -- *
  -- * Performs \p half multiplication of inputs \p a and \p b, in round-to-nearest
  -- * mode, and clamps the result to range [0.0, 1.0]. NaN results are flushed to
  -- * +0.0.
  -- *
  -- * \return Returns the \p half result of multiplying \p a and \p b with
  -- * saturation.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_ARITHMETIC
  -- * \brief Performs \p half fused multiply-add in round-to-nearest-even mode.
  -- *
  -- * Performs \p half multiply on inputs \p a and \p b,
  -- * then performs a \p half add of the result with \p c,
  -- * rounding the result once in round-to-nearest-even mode.
  -- *
  -- * \return Returns the \p half result of the fused multiply-add operation on \p
  -- * a, \p b, and \p c.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_ARITHMETIC
  -- * \brief Performs \p half fused multiply-add in round-to-nearest-even mode,
  -- * with saturation to [0.0, 1.0].
  -- *
  -- * Performs \p half multiply on inputs \p a and \p b,
  -- * then performs a \p half add of the result with \p c,
  -- * rounding the result once in round-to-nearest-even mode, and clamps the result
  -- * to range [0.0, 1.0]. NaN results are flushed to +0.0.
  -- *
  -- * \return Returns the \p half result of the fused multiply-add operation on \p
  -- * a, \p b, and \p c with saturation.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_ARITHMETIC
  -- * \brief Negates input \p half number and returns the result.
  -- *
  -- * Negates input \p half number and returns the result.
  -- *
  -- * \return Returns negated \p half input \p a.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF2_COMPARISON
  -- * \brief Performs \p half2 vector if-equal comparison, and returns boolean true
  -- * iff both \p half results are true, boolean false otherwise.
  -- *
  -- * Performs \p half2 vector if-equal comparison of inputs \p a and \p b.
  -- * The bool result is set to true only if both \p half if-equal comparisons
  -- * evaluate to true, or false otherwise.
  -- * NaN inputs generate false results.
  -- *
  -- * \return Returns boolean true if both \p half results of if-equal comparison
  -- * of vectors \p a and \p b are true, boolean false otherwise.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF2_COMPARISON
  -- * \brief Performs \p half2 vector not-equal comparison, and returns boolean
  -- * true iff both \p half results are true, boolean false otherwise.
  -- *
  -- * Performs \p half2 vector not-equal comparison of inputs \p a and \p b.
  -- * The bool result is set to true only if both \p half not-equal comparisons
  -- * evaluate to true, or false otherwise.
  -- * NaN inputs generate false results.
  -- *
  -- * \return Returns boolean true if both \p half results of not-equal comparison
  -- * of vectors \p a and \p b are true, boolean false otherwise.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF2_COMPARISON
  -- * \brief Performs \p half2 vector less-equal comparison, and returns boolean
  -- * true iff both \p half results are true, boolean false otherwise.
  -- *
  -- * Performs \p half2 vector less-equal comparison of inputs \p a and \p b.
  -- * The bool result is set to true only if both \p half less-equal comparisons
  -- * evaluate to true, or false otherwise.
  -- * NaN inputs generate false results.
  -- *
  -- * \return Returns boolean true if both \p half results of less-equal comparison
  -- * of vectors \p a and \p b are true, boolean false otherwise.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF2_COMPARISON
  -- * \brief Performs \p half2 vector greater-equal comparison, and returns boolean
  -- * true iff both \p half results are true, boolean false otherwise.
  -- *
  -- * Performs \p half2 vector greater-equal comparison of inputs \p a and \p b.
  -- * The bool result is set to true only if both \p half greater-equal comparisons
  -- * evaluate to true, or false otherwise.
  -- * NaN inputs generate false results.
  -- *
  -- * \return Returns boolean true if both \p half results of greater-equal
  -- * comparison of vectors \p a and \p b are true, boolean false otherwise.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF2_COMPARISON
  -- * \brief Performs \p half2 vector less-than comparison, and returns boolean
  -- * true iff both \p half results are true, boolean false otherwise.
  -- *
  -- * Performs \p half2 vector less-than comparison of inputs \p a and \p b.
  -- * The bool result is set to true only if both \p half less-than comparisons
  -- * evaluate to true, or false otherwise.
  -- * NaN inputs generate false results.
  -- *
  -- * \return Returns boolean true if both \p half results of less-than comparison
  -- * of vectors \p a and \p b are true, boolean false otherwise.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF2_COMPARISON
  -- * \brief Performs \p half2 vector greater-than comparison, and returns boolean
  -- * true iff both \p half results are true, boolean false otherwise.
  -- *
  -- * Performs \p half2 vector greater-than comparison of inputs \p a and \p b.
  -- * The bool result is set to true only if both \p half greater-than comparisons
  -- * evaluate to true, or false otherwise.
  -- * NaN inputs generate false results.
  -- *
  -- * \return Returns boolean true if both \p half results of greater-than
  -- * comparison of vectors \p a and \p b are true, boolean false otherwise.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF2_COMPARISON
  -- * \brief Performs \p half2 vector unordered if-equal comparison, and returns
  -- * boolean true iff both \p half results are true, boolean false otherwise.
  -- *
  -- * Performs \p half2 vector if-equal comparison of inputs \p a and \p b.
  -- * The bool result is set to true only if both \p half if-equal comparisons
  -- * evaluate to true, or false otherwise.
  -- * NaN inputs generate true results.
  -- *
  -- * \return Returns boolean true if both \p half results of unordered if-equal
  -- * comparison of vectors \p a and \p b are true, boolean false otherwise.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF2_COMPARISON
  -- * \brief Performs \p half2 vector unordered not-equal comparison, and returns
  -- * boolean true iff both \p half results are true, boolean false otherwise.
  -- *
  -- * Performs \p half2 vector not-equal comparison of inputs \p a and \p b.
  -- * The bool result is set to true only if both \p half not-equal comparisons
  -- * evaluate to true, or false otherwise.
  -- * NaN inputs generate true results.
  -- *
  -- * \return Returns boolean true if both \p half results of unordered not-equal
  -- * comparison of vectors \p a and \p b are true, boolean false otherwise.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF2_COMPARISON
  -- * \brief Performs \p half2 vector unordered less-equal comparison, and returns
  -- * boolean true iff both \p half results are true, boolean false otherwise.
  -- *
  -- * Performs \p half2 vector less-equal comparison of inputs \p a and \p b.
  -- * The bool result is set to true only if both \p half less-equal comparisons
  -- * evaluate to true, or false otherwise.
  -- * NaN inputs generate true results.
  -- *
  -- * \return Returns boolean true if both \p half results of unordered less-equal
  -- * comparison of vectors \p a and \p b are true, boolean false otherwise.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF2_COMPARISON
  -- * \brief Performs \p half2 vector unordered greater-equal comparison, and
  -- * returns boolean true iff both \p half results are true, boolean false
  -- * otherwise.
  -- *
  -- * Performs \p half2 vector greater-equal comparison of inputs \p a and \p b.
  -- * The bool result is set to true only if both \p half greater-equal comparisons
  -- * evaluate to true, or false otherwise.
  -- * NaN inputs generate true results.
  -- *
  -- * \return Returns boolean true if both \p half results of unordered
  -- * greater-equal comparison of vectors \p a and \p b are true, boolean false
  -- * otherwise.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF2_COMPARISON
  -- * \brief Performs \p half2 vector unordered less-than comparison, and returns
  -- * boolean true iff both \p half results are true, boolean false otherwise.
  -- *
  -- * Performs \p half2 vector less-than comparison of inputs \p a and \p b.
  -- * The bool result is set to true only if both \p half less-than comparisons
  -- * evaluate to true, or false otherwise.
  -- * NaN inputs generate true results.
  -- *
  -- * \return Returns boolean true if both \p half results of unordered less-than
  -- * comparison of vectors \p a and \p b are true, boolean false otherwise.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF2_COMPARISON
  -- * \brief Performs \p half2 vector unordered greater-than comparison, and
  -- * returns boolean true iff both \p half results are true, boolean false
  -- * otherwise.
  -- *
  -- * Performs \p half2 vector greater-than comparison of inputs \p a and \p b.
  -- * The bool result is set to true only if both \p half greater-than comparisons
  -- * evaluate to true, or false otherwise.
  -- * NaN inputs generate true results.
  -- *
  -- * \return Returns boolean true if both \p half results of unordered
  -- * greater-than comparison of vectors \p a and \p b are true, boolean false
  -- * otherwise.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_COMPARISON
  -- * \brief Performs \p half if-equal comparison.
  -- *
  -- * Performs \p half if-equal comparison of inputs \p a and \p b.
  -- * NaN inputs generate false results.
  -- *
  -- * \return Returns boolean result of if-equal comparison of \p a and \p b.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_COMPARISON
  -- * \brief Performs \p half not-equal comparison.
  -- *
  -- * Performs \p half not-equal comparison of inputs \p a and \p b.
  -- * NaN inputs generate false results.
  -- *
  -- * \return Returns boolean result of not-equal comparison of \p a and \p b.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_COMPARISON
  -- * \brief Performs \p half less-equal comparison.
  -- *
  -- * Performs \p half less-equal comparison of inputs \p a and \p b.
  -- * NaN inputs generate false results.
  -- *
  -- * \return Returns boolean result of less-equal comparison of \p a and \p b.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_COMPARISON
  -- * \brief Performs \p half greater-equal comparison.
  -- *
  -- * Performs \p half greater-equal comparison of inputs \p a and \p b.
  -- * NaN inputs generate false results.
  -- *
  -- * \return Returns boolean result of greater-equal comparison of \p a and \p b.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_COMPARISON
  -- * \brief Performs \p half less-than comparison.
  -- *
  -- * Performs \p half less-than comparison of inputs \p a and \p b.
  -- * NaN inputs generate false results.
  -- *
  -- * \return Returns boolean result of less-than comparison of \p a and \p b.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_COMPARISON
  -- * \brief Performs \p half greater-than comparison.
  -- *
  -- * Performs \p half greater-than comparison of inputs \p a and \p b.
  -- * NaN inputs generate false results.
  -- *
  -- * \return Returns boolean result of greater-than comparison of \p a and \p b.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_COMPARISON
  -- * \brief Performs \p half unordered if-equal comparison.
  -- *
  -- * Performs \p half if-equal comparison of inputs \p a and \p b.
  -- * NaN inputs generate true results.
  -- *
  -- * \return Returns boolean result of unordered if-equal comparison of \p a and
  -- * \p b.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_COMPARISON
  -- * \brief Performs \p half unordered not-equal comparison.
  -- *
  -- * Performs \p half not-equal comparison of inputs \p a and \p b.
  -- * NaN inputs generate true results.
  -- *
  -- * \return Returns boolean result of unordered not-equal comparison of \p a and
  -- * \p b.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_COMPARISON
  -- * \brief Performs \p half unordered less-equal comparison.
  -- *
  -- * Performs \p half less-equal comparison of inputs \p a and \p b.
  -- * NaN inputs generate true results.
  -- *
  -- * \return Returns boolean result of unordered less-equal comparison of \p a and
  -- * \p b.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_COMPARISON
  -- * \brief Performs \p half unordered greater-equal comparison.
  -- *
  -- * Performs \p half greater-equal comparison of inputs \p a and \p b.
  -- * NaN inputs generate true results.
  -- *
  -- * \return Returns boolean result of unordered greater-equal comparison of \p a
  -- * and \p b.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_COMPARISON
  -- * \brief Performs \p half unordered less-than comparison.
  -- *
  -- * Performs \p half less-than comparison of inputs \p a and \p b.
  -- * NaN inputs generate true results.
  -- *
  -- * \return Returns boolean result of unordered less-than comparison of \p a and
  -- * \p b.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_COMPARISON
  -- * \brief Performs \p half unordered greater-than comparison.
  -- *
  -- * Performs \p half greater-than comparison of inputs \p a and \p b.
  -- * NaN inputs generate true results.
  -- *
  -- * \return Returns boolean result of unordered greater-than comparison of \p a
  -- * and \p b.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_COMPARISON
  -- * \brief Determine whether \p half argument is a NaN.
  -- *
  -- * Determine whether \p half value \p a is a NaN.
  -- *
  -- * \return Returns boolean true iff argument is a NaN, boolean false otherwise.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_FUNCTIONS
  -- * \brief Calculates \p half square root in round-to-nearest-even mode.
  -- *
  -- * Calculates \p half square root of input \p a in round-to-nearest-even mode.
  -- *
  -- * \return Returns \p half square root of \p a.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_FUNCTIONS
  -- * \brief Calculates \p half reciprocal square root in round-to-nearest-even
  -- * mode.
  -- *
  -- * Calculates \p half reciprocal square root of input \p a in round-to-nearest
  -- * mode.
  -- *
  -- * \return Returns \p half reciprocal square root of \p a.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_FUNCTIONS
  -- * \brief Calculates \p half reciprocal in round-to-nearest-even mode.
  -- *
  -- * Calculates \p half reciprocal of input \p a in round-to-nearest-even mode.
  -- *
  -- * \return Returns \p half reciprocal of \p a.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_FUNCTIONS
  -- * \brief Calculates \p half natural logarithm in round-to-nearest-even mode.
  -- *
  -- * Calculates \p half natural logarithm of input \p a in round-to-nearest-even
  -- * mode.
  -- *
  -- * \return Returns \p half natural logarithm of \p a.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_FUNCTIONS
  -- * \brief Calculates \p half binary logarithm in round-to-nearest-even mode.
  -- *
  -- * Calculates \p half binary logarithm of input \p a in round-to-nearest-even
  -- * mode.
  -- *
  -- * \return Returns \p half binary logarithm of \p a.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_FUNCTIONS
  -- * \brief Calculates \p half decimal logarithm in round-to-nearest-even mode.
  -- *
  -- * Calculates \p half decimal logarithm of input \p a in round-to-nearest-even
  -- * mode.
  -- *
  -- * \return Returns \p half decimal logarithm of \p a.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_FUNCTIONS
  -- * \brief Calculates \p half natural exponential function in round-to-nearest
  -- * mode.
  -- *
  -- * Calculates \p half natural exponential function of input \p a in
  -- * round-to-nearest-even mode.
  -- *
  -- * \return Returns \p half natural exponential function of \p a.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_FUNCTIONS
  -- * \brief Calculates \p half binary exponential function in round-to-nearest
  -- * mode.
  -- *
  -- * Calculates \p half binary exponential function of input \p a in
  -- * round-to-nearest-even mode.
  -- *
  -- * \return Returns \p half binary exponential function of \p a.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_FUNCTIONS
  -- * \brief Calculates \p half decimal exponential function in round-to-nearest
  -- * mode.
  -- *
  -- * Calculates \p half decimal exponential function of input \p a in
  -- * round-to-nearest-even mode.
  -- *
  -- * \return Returns \p half decimal exponential function of \p a.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_FUNCTIONS
  -- * \brief Calculates \p half cosine in round-to-nearest-even mode.
  -- *
  -- * Calculates \p half cosine of input \p a in round-to-nearest-even mode.
  -- *
  -- * \return Returns \p half cosine of \p a.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF_FUNCTIONS
  -- * \brief Calculates \p half sine in round-to-nearest-even mode.
  -- *
  -- * Calculates \p half sine of input \p a in round-to-nearest-even mode.
  -- *
  -- * \return Returns \p half sine of \p a.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF2_FUNCTIONS
  -- * \brief Calculates \p half2 vector square root in round-to-nearest-even mode.
  -- *
  -- * Calculates \p half2 square root of input vector \p a in round-to-nearest
  -- * mode.
  -- *
  -- * \return Returns \p half2 square root of vector \p a.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF2_FUNCTIONS
  -- * \brief Calculates \p half2 vector reciprocal square root in round-to-nearest
  -- * mode.
  -- *
  -- * Calculates \p half2 reciprocal square root of input vector \p a in
  -- * round-to-nearest-even mode.
  -- *
  -- * \return Returns \p half2 reciprocal square root of vector \p a.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF2_FUNCTIONS
  -- * \brief Calculates \p half2 vector reciprocal in round-to-nearest-even mode.
  -- *
  -- * Calculates \p half2 reciprocal of input vector \p a in round-to-nearest-even
  -- * mode.
  -- *
  -- * \return Returns \p half2 reciprocal of vector \p a.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF2_FUNCTIONS
  -- * \brief Calculates \p half2 vector natural logarithm in round-to-nearest-even
  -- * mode.
  -- *
  -- * Calculates \p half2 natural logarithm of input vector \p a in
  -- * round-to-nearest-even mode.
  -- *
  -- * \return Returns \p half2 natural logarithm of vector \p a.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF2_FUNCTIONS
  -- * \brief Calculates \p half2 vector binary logarithm in round-to-nearest-even
  -- * mode.
  -- *
  -- * Calculates \p half2 binary logarithm of input vector \p a in round-to-nearest
  -- * mode.
  -- *
  -- * \return Returns \p half2 binary logarithm of vector \p a.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF2_FUNCTIONS
  -- * \brief Calculates \p half2 vector decimal logarithm in round-to-nearest-even
  -- * mode.
  -- *
  -- * Calculates \p half2 decimal logarithm of input vector \p a in
  -- * round-to-nearest-even mode.
  -- *
  -- * \return Returns \p half2 decimal logarithm of vector \p a.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF2_FUNCTIONS
  -- * \brief Calculates \p half2 vector exponential function in round-to-nearest
  -- * mode.
  -- *
  -- * Calculates \p half2 exponential function of input vector \p a in
  -- * round-to-nearest-even mode.
  -- *
  -- * \return Returns \p half2 exponential function of vector \p a.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF2_FUNCTIONS
  -- * \brief Calculates \p half2 vector binary exponential function in
  -- * round-to-nearest-even mode.
  -- *
  -- * Calculates \p half2 binary exponential function of input vector \p a in
  -- * round-to-nearest-even mode.
  -- *
  -- * \return Returns \p half2 binary exponential function of vector \p a.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF2_FUNCTIONS
  -- * \brief Calculates \p half2 vector decimal exponential function in
  -- * round-to-nearest-even mode.
  -- *
  -- * Calculates \p half2 decimal exponential function of input vector \p a in 
  -- * round-to-nearest-even mode.
  -- *
  -- * \return Returns \p half2 decimal exponential function of vector \p a.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF2_FUNCTIONS
  -- * \brief Calculates \p half2 vector cosine in round-to-nearest-even mode.
  -- *
  -- * Calculates \p half2 cosine of input vector \p a in round-to-nearest-even
  -- * mode.
  -- *
  -- * \return Returns \p half2 cosine of vector \p a.
  --  

  --*
  -- * \ingroup CUDA_MATH__HALF2_FUNCTIONS
  -- * \brief Calculates \p half2 vector sine in round-to-nearest-even mode.
  -- *
  -- * Calculates \p half2 sine of input vector \p a in round-to-nearest-even mode.
  -- *
  -- * \return Returns \p half2 sine of vector \p a.
  --  

  --*****************************************************************************
  -- *                           __half, __half2 warp shuffle                     *
  -- ***************************************************************************** 

  --*****************************************************************************
  -- *               __half and __half2 __ldg,__ldcg,__ldca,__ldcs                *
  -- ***************************************************************************** 

  --*****************************************************************************
  -- *                             __half2 comparison                             *
  -- ***************************************************************************** 

  --*****************************************************************************
  -- *                             __half comparison                              *
  -- ***************************************************************************** 

  --*****************************************************************************
  -- *                            __half2 arithmetic                             *
  -- ***************************************************************************** 

  --*****************************************************************************
  -- *                             __half arithmetic                             *
  -- ***************************************************************************** 

  --*****************************************************************************
  -- *                             __half2 functions                  *
  -- ***************************************************************************** 

end cuda_fp16_h;

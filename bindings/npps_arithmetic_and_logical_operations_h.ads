pragma Ada_2005;
pragma Style_Checks (Off);

with Interfaces.C; use Interfaces.C;
with nppdefs_h;

package npps_arithmetic_and_logical_operations_h is

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
  -- * \file npps_arithmetic_and_logical_operations.h
  -- * Signal Arithmetic and Logical Operations.
  --  

  --* 
  -- * @defgroup signal_arithmetic_and_logical_operations Arithmetic and Logical Operations
  -- * @ingroup npps
  -- *
  -- * @{
  -- *
  --  

  --* 
  -- * @defgroup signal_arithmetic Arithmetic Operations
  -- *
  -- * @{
  -- *
  --  

  --* 
  -- * @defgroup signal_addc AddC
  -- * Adds a constant value to each sample of a signal.
  -- *
  -- * @{
  -- *
  --  

  --* 
  -- * 8-bit unsigned char in place signal add constant,
  -- * scale, then clamp to saturated value
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nValue Constant value to be added to each vector element
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAddC_8u_ISfs
     (nValue : nppdefs_h.Npp8u;
      pSrcDst : access nppdefs_h.Npp8u;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:97
   pragma Import (C, nppsAddC_8u_ISfs, "nppsAddC_8u_ISfs");

  --* 
  -- * 8-bit unsigned charvector add constant, scale, then clamp to saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nValue Constant value to be added to each vector element
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAddC_8u_Sfs
     (pSrc : access nppdefs_h.Npp8u;
      nValue : nppdefs_h.Npp8u;
      pDst : access nppdefs_h.Npp8u;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:109
   pragma Import (C, nppsAddC_8u_Sfs, "nppsAddC_8u_Sfs");

  --* 
  -- * 16-bit unsigned short in place signal add constant, scale, then clamp to saturated value.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nValue Constant value to be added to each vector element
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAddC_16u_ISfs
     (nValue : nppdefs_h.Npp16u;
      pSrcDst : access nppdefs_h.Npp16u;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:120
   pragma Import (C, nppsAddC_16u_ISfs, "nppsAddC_16u_ISfs");

  --* 
  -- * 16-bit unsigned short vector add constant, scale, then clamp to saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nValue Constant value to be added to each vector element
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAddC_16u_Sfs
     (pSrc : access nppdefs_h.Npp16u;
      nValue : nppdefs_h.Npp16u;
      pDst : access nppdefs_h.Npp16u;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:132
   pragma Import (C, nppsAddC_16u_Sfs, "nppsAddC_16u_Sfs");

  --* 
  -- * 16-bit signed short in place  signal add constant, scale, then clamp to saturated value.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nValue Constant value to be added to each vector element
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAddC_16s_ISfs
     (nValue : nppdefs_h.Npp16s;
      pSrcDst : access nppdefs_h.Npp16s;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:143
   pragma Import (C, nppsAddC_16s_ISfs, "nppsAddC_16s_ISfs");

  --* 
  -- * 16-bit signed short signal add constant, scale, then clamp to saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nValue Constant value to be added to each vector element
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAddC_16s_Sfs
     (pSrc : access nppdefs_h.Npp16s;
      nValue : nppdefs_h.Npp16s;
      pDst : access nppdefs_h.Npp16s;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:155
   pragma Import (C, nppsAddC_16s_Sfs, "nppsAddC_16s_Sfs");

  --* 
  -- * 16-bit integer complex number (16 bit real, 16 bit imaginary)signal add constant, 
  -- * scale, then clamp to saturated value.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nValue Constant value to be added to each vector element
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAddC_16sc_ISfs
     (nValue : nppdefs_h.Npp16sc;
      pSrcDst : access nppdefs_h.Npp16sc;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:167
   pragma Import (C, nppsAddC_16sc_ISfs, "nppsAddC_16sc_ISfs");

  --* 
  -- * 16-bit integer complex number (16 bit real, 16 bit imaginary) signal add constant,
  -- * scale, then clamp to saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nValue Constant value to be added to each vector element
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAddC_16sc_Sfs
     (pSrc : access constant nppdefs_h.Npp16sc;
      nValue : nppdefs_h.Npp16sc;
      pDst : access nppdefs_h.Npp16sc;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:180
   pragma Import (C, nppsAddC_16sc_Sfs, "nppsAddC_16sc_Sfs");

  --* 
  -- * 32-bit signed integer in place signal add constant and scale.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nValue Constant value to be added to each vector element
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAddC_32s_ISfs
     (nValue : nppdefs_h.Npp32s;
      pSrcDst : access nppdefs_h.Npp32s;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:191
   pragma Import (C, nppsAddC_32s_ISfs, "nppsAddC_32s_ISfs");

  --* 
  -- * 32-bit signed integersignal add constant and scale.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nValue Constant value to be added to each vector element
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAddC_32s_Sfs
     (pSrc : access nppdefs_h.Npp32s;
      nValue : nppdefs_h.Npp32s;
      pDst : access nppdefs_h.Npp32s;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:203
   pragma Import (C, nppsAddC_32s_Sfs, "nppsAddC_32s_Sfs");

  --* 
  -- * 32-bit integer complex number (32 bit real, 32 bit imaginary) in place signal
  -- * add constant and scale.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nValue Constant value to be added to each vector element
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAddC_32sc_ISfs
     (nValue : nppdefs_h.Npp32sc;
      pSrcDst : access nppdefs_h.Npp32sc;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:215
   pragma Import (C, nppsAddC_32sc_ISfs, "nppsAddC_32sc_ISfs");

  --* 
  -- * 32-bit integer complex number (32 bit real, 32 bit imaginary) signal add constant
  -- * and scale.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nValue Constant value to be added to each vector element
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAddC_32sc_Sfs
     (pSrc : access constant nppdefs_h.Npp32sc;
      nValue : nppdefs_h.Npp32sc;
      pDst : access nppdefs_h.Npp32sc;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:228
   pragma Import (C, nppsAddC_32sc_Sfs, "nppsAddC_32sc_Sfs");

  --* 
  -- * 32-bit floating point in place signal add constant.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nValue Constant value to be added to each vector element
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAddC_32f_I
     (nValue : nppdefs_h.Npp32f;
      pSrcDst : access nppdefs_h.Npp32f;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:238
   pragma Import (C, nppsAddC_32f_I, "nppsAddC_32f_I");

  --* 
  -- * 32-bit floating point signal add constant.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nValue Constant value to be added to each vector element
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAddC_32f
     (pSrc : access nppdefs_h.Npp32f;
      nValue : nppdefs_h.Npp32f;
      pDst : access nppdefs_h.Npp32f;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:249
   pragma Import (C, nppsAddC_32f, "nppsAddC_32f");

  --* 
  -- * 32-bit floating point complex number (32 bit real, 32 bit imaginary) in
  -- * place signal add constant.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nValue Constant value to be added to each vector element
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAddC_32fc_I
     (nValue : nppdefs_h.Npp32fc;
      pSrcDst : access nppdefs_h.Npp32fc;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:260
   pragma Import (C, nppsAddC_32fc_I, "nppsAddC_32fc_I");

  --* 
  -- * 32-bit floating point complex number (32 bit real, 32 bit imaginary) signal
  -- * add constant.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nValue Constant value to be added to each vector element
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAddC_32fc
     (pSrc : access constant nppdefs_h.Npp32fc;
      nValue : nppdefs_h.Npp32fc;
      pDst : access nppdefs_h.Npp32fc;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:272
   pragma Import (C, nppsAddC_32fc, "nppsAddC_32fc");

  --* 
  -- * 64-bit floating point, in place signal add constant.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nValue Constant value to be added to each vector element
  -- * \param nLength Length of the vectors, number of items.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAddC_64f_I
     (nValue : nppdefs_h.Npp64f;
      pSrcDst : access nppdefs_h.Npp64f;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:282
   pragma Import (C, nppsAddC_64f_I, "nppsAddC_64f_I");

  --* 
  -- * 64-bit floating pointsignal add constant.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nValue Constant value to be added to each vector element
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAddC_64f
     (pSrc : access nppdefs_h.Npp64f;
      nValue : nppdefs_h.Npp64f;
      pDst : access nppdefs_h.Npp64f;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:293
   pragma Import (C, nppsAddC_64f, "nppsAddC_64f");

  --* 
  -- * 64-bit floating point complex number (64 bit real, 64 bit imaginary) in
  -- * place signal add constant.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nValue Constant value to be added to each vector element
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAddC_64fc_I
     (nValue : nppdefs_h.Npp64fc;
      pSrcDst : access nppdefs_h.Npp64fc;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:304
   pragma Import (C, nppsAddC_64fc_I, "nppsAddC_64fc_I");

  --* 
  -- * 64-bit floating point complex number (64 bit real, 64 bit imaginary) signal
  -- * add constant.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nValue Constant value to be added to each vector element
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAddC_64fc
     (pSrc : access constant nppdefs_h.Npp64fc;
      nValue : nppdefs_h.Npp64fc;
      pDst : access nppdefs_h.Npp64fc;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:316
   pragma Import (C, nppsAddC_64fc, "nppsAddC_64fc");

  --* @} signal_addc  
  --* 
  -- * @defgroup signal_addproductc AddProductC
  -- * Adds product of a constant and each sample of a source signal to the each sample of destination signal.
  -- *
  -- * @{
  -- *
  --  

  --* 
  -- * 32-bit floating point signal add product of signal times constant to destination signal.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nValue Constant value to be multiplied by each vector element
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAddProductC_32f
     (pSrc : access nppdefs_h.Npp32f;
      nValue : nppdefs_h.Npp32f;
      pDst : access nppdefs_h.Npp32f;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:337
   pragma Import (C, nppsAddProductC_32f, "nppsAddProductC_32f");

  --* @} signal_addproductc  
  --* 
  -- * @defgroup signal_mulc MulC
  -- *
  -- * Multiplies each sample of a signal by a constant value.
  -- *
  -- * @{
  -- *
  --  

  --* 
  -- * 8-bit unsigned char in place signal times constant,
  -- * scale, then clamp to saturated value
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nValue Constant value to be multiplied by each vector element
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMulC_8u_ISfs
     (nValue : nppdefs_h.Npp8u;
      pSrcDst : access nppdefs_h.Npp8u;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:360
   pragma Import (C, nppsMulC_8u_ISfs, "nppsMulC_8u_ISfs");

  --* 
  -- * 8-bit unsigned char signal times constant, scale, then clamp to saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nValue Constant value to be multiplied by each vector element
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMulC_8u_Sfs
     (pSrc : access nppdefs_h.Npp8u;
      nValue : nppdefs_h.Npp8u;
      pDst : access nppdefs_h.Npp8u;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:372
   pragma Import (C, nppsMulC_8u_Sfs, "nppsMulC_8u_Sfs");

  --* 
  -- * 16-bit unsigned short in place signal times constant, scale, then clamp to saturated value.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nValue Constant value to be multiplied by each vector element
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMulC_16u_ISfs
     (nValue : nppdefs_h.Npp16u;
      pSrcDst : access nppdefs_h.Npp16u;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:383
   pragma Import (C, nppsMulC_16u_ISfs, "nppsMulC_16u_ISfs");

  --* 
  -- * 16-bit unsigned short signal times constant, scale, then clamp to saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nValue Constant value to be multiplied by each vector element
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMulC_16u_Sfs
     (pSrc : access nppdefs_h.Npp16u;
      nValue : nppdefs_h.Npp16u;
      pDst : access nppdefs_h.Npp16u;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:395
   pragma Import (C, nppsMulC_16u_Sfs, "nppsMulC_16u_Sfs");

  --* 
  -- * 16-bit signed short in place signal times constant, scale, then clamp to saturated value.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nValue Constant value to be multiplied by each vector element
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMulC_16s_ISfs
     (nValue : nppdefs_h.Npp16s;
      pSrcDst : access nppdefs_h.Npp16s;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:406
   pragma Import (C, nppsMulC_16s_ISfs, "nppsMulC_16s_ISfs");

  --* 
  -- * 16-bit signed short signal times constant, scale, then clamp to saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nValue Constant value to be multiplied by each vector element
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMulC_16s_Sfs
     (pSrc : access nppdefs_h.Npp16s;
      nValue : nppdefs_h.Npp16s;
      pDst : access nppdefs_h.Npp16s;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:418
   pragma Import (C, nppsMulC_16s_Sfs, "nppsMulC_16s_Sfs");

  --* 
  -- * 16-bit integer complex number (16 bit real, 16 bit imaginary)signal times constant, 
  -- * scale, then clamp to saturated value.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nValue Constant value to be multiplied by each vector element
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMulC_16sc_ISfs
     (nValue : nppdefs_h.Npp16sc;
      pSrcDst : access nppdefs_h.Npp16sc;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:430
   pragma Import (C, nppsMulC_16sc_ISfs, "nppsMulC_16sc_ISfs");

  --* 
  -- * 16-bit integer complex number (16 bit real, 16 bit imaginary)signal times constant,
  -- * scale, then clamp to saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nValue Constant value to be multiplied by each vector element
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMulC_16sc_Sfs
     (pSrc : access constant nppdefs_h.Npp16sc;
      nValue : nppdefs_h.Npp16sc;
      pDst : access nppdefs_h.Npp16sc;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:443
   pragma Import (C, nppsMulC_16sc_Sfs, "nppsMulC_16sc_Sfs");

  --* 
  -- * 32-bit signed integer in place signal times constant and scale.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nValue Constant value to be multiplied by each vector element
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMulC_32s_ISfs
     (nValue : nppdefs_h.Npp32s;
      pSrcDst : access nppdefs_h.Npp32s;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:454
   pragma Import (C, nppsMulC_32s_ISfs, "nppsMulC_32s_ISfs");

  --* 
  -- * 32-bit signed integer signal times constant and scale.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nValue Constant value to be multiplied by each vector element
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMulC_32s_Sfs
     (pSrc : access nppdefs_h.Npp32s;
      nValue : nppdefs_h.Npp32s;
      pDst : access nppdefs_h.Npp32s;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:466
   pragma Import (C, nppsMulC_32s_Sfs, "nppsMulC_32s_Sfs");

  --* 
  -- * 32-bit integer complex number (32 bit real, 32 bit imaginary) in place signal
  -- * times constant and scale.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nValue Constant value to be multiplied by each vector element
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMulC_32sc_ISfs
     (nValue : nppdefs_h.Npp32sc;
      pSrcDst : access nppdefs_h.Npp32sc;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:478
   pragma Import (C, nppsMulC_32sc_ISfs, "nppsMulC_32sc_ISfs");

  --* 
  -- * 32-bit integer complex number (32 bit real, 32 bit imaginary) signal times constant
  -- * and scale.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nValue Constant value to be multiplied by each vector element
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMulC_32sc_Sfs
     (pSrc : access constant nppdefs_h.Npp32sc;
      nValue : nppdefs_h.Npp32sc;
      pDst : access nppdefs_h.Npp32sc;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:491
   pragma Import (C, nppsMulC_32sc_Sfs, "nppsMulC_32sc_Sfs");

  --* 
  -- * 32-bit floating point in place signal times constant.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nValue Constant value to be multiplied by each vector element
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMulC_32f_I
     (nValue : nppdefs_h.Npp32f;
      pSrcDst : access nppdefs_h.Npp32f;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:501
   pragma Import (C, nppsMulC_32f_I, "nppsMulC_32f_I");

  --* 
  -- * 32-bit floating point signal times constant.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nValue Constant value to be multiplied by each vector element
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMulC_32f
     (pSrc : access nppdefs_h.Npp32f;
      nValue : nppdefs_h.Npp32f;
      pDst : access nppdefs_h.Npp32f;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:512
   pragma Import (C, nppsMulC_32f, "nppsMulC_32f");

  --* 
  -- * 32-bit floating point signal times constant with output converted to 16-bit signed integer.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nValue Constant value to be multiplied by each vector element
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMulC_Low_32f16s
     (pSrc : access nppdefs_h.Npp32f;
      nValue : nppdefs_h.Npp32f;
      pDst : access nppdefs_h.Npp16s;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:523
   pragma Import (C, nppsMulC_Low_32f16s, "nppsMulC_Low_32f16s");

  --* 
  -- * 32-bit floating point signal times constant with output converted to 16-bit signed integer
  -- * with scaling and saturation of output result.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nValue Constant value to be multiplied by each vector element
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMulC_32f16s_Sfs
     (pSrc : access nppdefs_h.Npp32f;
      nValue : nppdefs_h.Npp32f;
      pDst : access nppdefs_h.Npp16s;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:536
   pragma Import (C, nppsMulC_32f16s_Sfs, "nppsMulC_32f16s_Sfs");

  --* 
  -- * 32-bit floating point complex number (32 bit real, 32 bit imaginary) in
  -- * place signal times constant.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nValue Constant value to be multiplied by each vector element
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMulC_32fc_I
     (nValue : nppdefs_h.Npp32fc;
      pSrcDst : access nppdefs_h.Npp32fc;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:547
   pragma Import (C, nppsMulC_32fc_I, "nppsMulC_32fc_I");

  --* 
  -- * 32-bit floating point complex number (32 bit real, 32 bit imaginary) signal
  -- * times constant.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nValue Constant value to be multiplied by each vector element
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMulC_32fc
     (pSrc : access constant nppdefs_h.Npp32fc;
      nValue : nppdefs_h.Npp32fc;
      pDst : access nppdefs_h.Npp32fc;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:559
   pragma Import (C, nppsMulC_32fc, "nppsMulC_32fc");

  --* 
  -- * 64-bit floating point, in place signal times constant.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nValue Constant value to be multiplied by each vector element
  -- * \param nLength Length of the vectors, number of items.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMulC_64f_I
     (nValue : nppdefs_h.Npp64f;
      pSrcDst : access nppdefs_h.Npp64f;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:569
   pragma Import (C, nppsMulC_64f_I, "nppsMulC_64f_I");

  --* 
  -- * 64-bit floating point signal times constant.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nValue Constant value to be multiplied by each vector element
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMulC_64f
     (pSrc : access nppdefs_h.Npp64f;
      nValue : nppdefs_h.Npp64f;
      pDst : access nppdefs_h.Npp64f;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:580
   pragma Import (C, nppsMulC_64f, "nppsMulC_64f");

  --* 
  -- * 64-bit floating point signal times constant with in place conversion to 64-bit signed integer
  -- * and with scaling and saturation of output result.
  -- * \param nValue Constant value to be multiplied by each vector element
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMulC_64f64s_ISfs
     (nValue : nppdefs_h.Npp64f;
      pDst : access nppdefs_h.Npp64s;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:592
   pragma Import (C, nppsMulC_64f64s_ISfs, "nppsMulC_64f64s_ISfs");

  --* 
  -- * 64-bit floating point complex number (64 bit real, 64 bit imaginary) in
  -- * place signal times constant.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nValue Constant value to be multiplied by each vector element
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMulC_64fc_I
     (nValue : nppdefs_h.Npp64fc;
      pSrcDst : access nppdefs_h.Npp64fc;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:603
   pragma Import (C, nppsMulC_64fc_I, "nppsMulC_64fc_I");

  --* 
  -- * 64-bit floating point complex number (64 bit real, 64 bit imaginary) signal
  -- * times constant.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nValue Constant value to be multiplied by each vector element
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMulC_64fc
     (pSrc : access constant nppdefs_h.Npp64fc;
      nValue : nppdefs_h.Npp64fc;
      pDst : access nppdefs_h.Npp64fc;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:615
   pragma Import (C, nppsMulC_64fc, "nppsMulC_64fc");

  --* @} signal_mulc  
  --* 
  -- * @defgroup signal_subc SubC
  -- *
  -- * Subtracts a constant from each sample of a signal.
  -- *
  -- * @{
  -- *
  --  

  --* 
  -- * 8-bit unsigned char in place signal subtract constant,
  -- * scale, then clamp to saturated value
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nValue Constant value to be subtracted from each vector element
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSubC_8u_ISfs
     (nValue : nppdefs_h.Npp8u;
      pSrcDst : access nppdefs_h.Npp8u;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:638
   pragma Import (C, nppsSubC_8u_ISfs, "nppsSubC_8u_ISfs");

  --* 
  -- * 8-bit unsigned char signal subtract constant, scale, then clamp to saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nValue Constant value to be subtracted from each vector element
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSubC_8u_Sfs
     (pSrc : access nppdefs_h.Npp8u;
      nValue : nppdefs_h.Npp8u;
      pDst : access nppdefs_h.Npp8u;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:650
   pragma Import (C, nppsSubC_8u_Sfs, "nppsSubC_8u_Sfs");

  --* 
  -- * 16-bit unsigned short in place signal subtract constant, scale, then clamp to saturated value.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nValue Constant value to be subtracted from each vector element
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSubC_16u_ISfs
     (nValue : nppdefs_h.Npp16u;
      pSrcDst : access nppdefs_h.Npp16u;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:661
   pragma Import (C, nppsSubC_16u_ISfs, "nppsSubC_16u_ISfs");

  --* 
  -- * 16-bit unsigned short signal subtract constant, scale, then clamp to saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nValue Constant value to be subtracted from each vector element
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSubC_16u_Sfs
     (pSrc : access nppdefs_h.Npp16u;
      nValue : nppdefs_h.Npp16u;
      pDst : access nppdefs_h.Npp16u;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:673
   pragma Import (C, nppsSubC_16u_Sfs, "nppsSubC_16u_Sfs");

  --* 
  -- * 16-bit signed short in place signal subtract constant, scale, then clamp to saturated value.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nValue Constant value to be subtracted from each vector element
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSubC_16s_ISfs
     (nValue : nppdefs_h.Npp16s;
      pSrcDst : access nppdefs_h.Npp16s;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:684
   pragma Import (C, nppsSubC_16s_ISfs, "nppsSubC_16s_ISfs");

  --* 
  -- * 16-bit signed short signal subtract constant, scale, then clamp to saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nValue Constant value to be subtracted from each vector element
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSubC_16s_Sfs
     (pSrc : access nppdefs_h.Npp16s;
      nValue : nppdefs_h.Npp16s;
      pDst : access nppdefs_h.Npp16s;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:696
   pragma Import (C, nppsSubC_16s_Sfs, "nppsSubC_16s_Sfs");

  --* 
  -- * 16-bit integer complex number (16 bit real, 16 bit imaginary) signal subtract constant, 
  -- * scale, then clamp to saturated value.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nValue Constant value to be subtracted from each vector element
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSubC_16sc_ISfs
     (nValue : nppdefs_h.Npp16sc;
      pSrcDst : access nppdefs_h.Npp16sc;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:708
   pragma Import (C, nppsSubC_16sc_ISfs, "nppsSubC_16sc_ISfs");

  --* 
  -- * 16-bit integer complex number (16 bit real, 16 bit imaginary) signal subtract constant,
  -- * scale, then clamp to saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nValue Constant value to be subtracted from each vector element
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSubC_16sc_Sfs
     (pSrc : access constant nppdefs_h.Npp16sc;
      nValue : nppdefs_h.Npp16sc;
      pDst : access nppdefs_h.Npp16sc;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:721
   pragma Import (C, nppsSubC_16sc_Sfs, "nppsSubC_16sc_Sfs");

  --* 
  -- * 32-bit signed integer in place signal subtract constant and scale.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nValue Constant value to be subtracted from each vector element
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSubC_32s_ISfs
     (nValue : nppdefs_h.Npp32s;
      pSrcDst : access nppdefs_h.Npp32s;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:732
   pragma Import (C, nppsSubC_32s_ISfs, "nppsSubC_32s_ISfs");

  --* 
  -- * 32-bit signed integer signal subtract constant and scale.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nValue Constant value to be subtracted from each vector element
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSubC_32s_Sfs
     (pSrc : access nppdefs_h.Npp32s;
      nValue : nppdefs_h.Npp32s;
      pDst : access nppdefs_h.Npp32s;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:744
   pragma Import (C, nppsSubC_32s_Sfs, "nppsSubC_32s_Sfs");

  --* 
  -- * 32-bit integer complex number (32 bit real, 32 bit imaginary) in place signal
  -- * subtract constant and scale.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nValue Constant value to be subtracted from each vector element
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSubC_32sc_ISfs
     (nValue : nppdefs_h.Npp32sc;
      pSrcDst : access nppdefs_h.Npp32sc;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:756
   pragma Import (C, nppsSubC_32sc_ISfs, "nppsSubC_32sc_ISfs");

  --* 
  -- * 32-bit integer complex number (32 bit real, 32 bit imaginary)signal subtract constant
  -- * and scale.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nValue Constant value to be subtracted from each vector element
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSubC_32sc_Sfs
     (pSrc : access constant nppdefs_h.Npp32sc;
      nValue : nppdefs_h.Npp32sc;
      pDst : access nppdefs_h.Npp32sc;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:769
   pragma Import (C, nppsSubC_32sc_Sfs, "nppsSubC_32sc_Sfs");

  --* 
  -- * 32-bit floating point in place signal subtract constant.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nValue Constant value to be subtracted from each vector element
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSubC_32f_I
     (nValue : nppdefs_h.Npp32f;
      pSrcDst : access nppdefs_h.Npp32f;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:779
   pragma Import (C, nppsSubC_32f_I, "nppsSubC_32f_I");

  --* 
  -- * 32-bit floating point signal subtract constant.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nValue Constant value to be subtracted from each vector element
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSubC_32f
     (pSrc : access nppdefs_h.Npp32f;
      nValue : nppdefs_h.Npp32f;
      pDst : access nppdefs_h.Npp32f;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:790
   pragma Import (C, nppsSubC_32f, "nppsSubC_32f");

  --* 
  -- * 32-bit floating point complex number (32 bit real, 32 bit imaginary) in
  -- * place signal subtract constant.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nValue Constant value to be subtracted from each vector element
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSubC_32fc_I
     (nValue : nppdefs_h.Npp32fc;
      pSrcDst : access nppdefs_h.Npp32fc;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:801
   pragma Import (C, nppsSubC_32fc_I, "nppsSubC_32fc_I");

  --* 
  -- * 32-bit floating point complex number (32 bit real, 32 bit imaginary) signal
  -- * subtract constant.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nValue Constant value to be subtracted from each vector element
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSubC_32fc
     (pSrc : access constant nppdefs_h.Npp32fc;
      nValue : nppdefs_h.Npp32fc;
      pDst : access nppdefs_h.Npp32fc;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:813
   pragma Import (C, nppsSubC_32fc, "nppsSubC_32fc");

  --* 
  -- * 64-bit floating point, in place signal subtract constant.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nValue Constant value to be subtracted from each vector element
  -- * \param nLength Length of the vectors, number of items.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSubC_64f_I
     (nValue : nppdefs_h.Npp64f;
      pSrcDst : access nppdefs_h.Npp64f;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:823
   pragma Import (C, nppsSubC_64f_I, "nppsSubC_64f_I");

  --* 
  -- * 64-bit floating point signal subtract constant.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nValue Constant value to be subtracted from each vector element
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSubC_64f
     (pSrc : access nppdefs_h.Npp64f;
      nValue : nppdefs_h.Npp64f;
      pDst : access nppdefs_h.Npp64f;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:834
   pragma Import (C, nppsSubC_64f, "nppsSubC_64f");

  --* 
  -- * 64-bit floating point complex number (64 bit real, 64 bit imaginary) in
  -- * place signal subtract constant.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nValue Constant value to be subtracted from each vector element
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSubC_64fc_I
     (nValue : nppdefs_h.Npp64fc;
      pSrcDst : access nppdefs_h.Npp64fc;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:845
   pragma Import (C, nppsSubC_64fc_I, "nppsSubC_64fc_I");

  --* 
  -- * 64-bit floating point complex number (64 bit real, 64 bit imaginary) signal
  -- * subtract constant.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nValue Constant value to be subtracted from each vector element
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSubC_64fc
     (pSrc : access constant nppdefs_h.Npp64fc;
      nValue : nppdefs_h.Npp64fc;
      pDst : access nppdefs_h.Npp64fc;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:857
   pragma Import (C, nppsSubC_64fc, "nppsSubC_64fc");

  --* @} signal_subc  
  --* 
  -- * @defgroup signal_subcrev SubCRev
  -- *
  -- * Subtracts each sample of a signal from a constant.
  -- *
  -- * @{
  -- *
  --  

  --* 
  -- * 8-bit unsigned char in place signal subtract from constant,
  -- * scale, then clamp to saturated value
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nValue Constant value each vector element is to be subtracted from
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSubCRev_8u_ISfs
     (nValue : nppdefs_h.Npp8u;
      pSrcDst : access nppdefs_h.Npp8u;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:880
   pragma Import (C, nppsSubCRev_8u_ISfs, "nppsSubCRev_8u_ISfs");

  --* 
  -- * 8-bit unsigned char signal subtract from constant, scale, then clamp to saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nValue Constant value each vector element is to be subtracted from
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSubCRev_8u_Sfs
     (pSrc : access nppdefs_h.Npp8u;
      nValue : nppdefs_h.Npp8u;
      pDst : access nppdefs_h.Npp8u;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:892
   pragma Import (C, nppsSubCRev_8u_Sfs, "nppsSubCRev_8u_Sfs");

  --* 
  -- * 16-bit unsigned short in place signal subtract from constant, scale, then clamp to saturated value.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nValue Constant value each vector element is to be subtracted from
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSubCRev_16u_ISfs
     (nValue : nppdefs_h.Npp16u;
      pSrcDst : access nppdefs_h.Npp16u;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:903
   pragma Import (C, nppsSubCRev_16u_ISfs, "nppsSubCRev_16u_ISfs");

  --* 
  -- * 16-bit unsigned short signal subtract from constant, scale, then clamp to saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nValue Constant value each vector element is to be subtracted from
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSubCRev_16u_Sfs
     (pSrc : access nppdefs_h.Npp16u;
      nValue : nppdefs_h.Npp16u;
      pDst : access nppdefs_h.Npp16u;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:915
   pragma Import (C, nppsSubCRev_16u_Sfs, "nppsSubCRev_16u_Sfs");

  --* 
  -- * 16-bit signed short in place signal subtract from constant, scale, then clamp to saturated value.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nValue Constant value each vector element is to be subtracted from
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSubCRev_16s_ISfs
     (nValue : nppdefs_h.Npp16s;
      pSrcDst : access nppdefs_h.Npp16s;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:926
   pragma Import (C, nppsSubCRev_16s_ISfs, "nppsSubCRev_16s_ISfs");

  --* 
  -- * 16-bit signed short signal subtract from constant, scale, then clamp to saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nValue Constant value each vector element is to be subtracted from
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSubCRev_16s_Sfs
     (pSrc : access nppdefs_h.Npp16s;
      nValue : nppdefs_h.Npp16s;
      pDst : access nppdefs_h.Npp16s;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:938
   pragma Import (C, nppsSubCRev_16s_Sfs, "nppsSubCRev_16s_Sfs");

  --* 
  -- * 16-bit integer complex number (16 bit real, 16 bit imaginary) signal subtract from constant, 
  -- * scale, then clamp to saturated value.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nValue Constant value each vector element is to be subtracted from
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSubCRev_16sc_ISfs
     (nValue : nppdefs_h.Npp16sc;
      pSrcDst : access nppdefs_h.Npp16sc;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:950
   pragma Import (C, nppsSubCRev_16sc_ISfs, "nppsSubCRev_16sc_ISfs");

  --* 
  -- * 16-bit integer complex number (16 bit real, 16 bit imaginary) signal subtract from constant,
  -- * scale, then clamp to saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nValue Constant value each vector element is to be subtracted from
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSubCRev_16sc_Sfs
     (pSrc : access constant nppdefs_h.Npp16sc;
      nValue : nppdefs_h.Npp16sc;
      pDst : access nppdefs_h.Npp16sc;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:963
   pragma Import (C, nppsSubCRev_16sc_Sfs, "nppsSubCRev_16sc_Sfs");

  --* 
  -- * 32-bit signed integer in place signal subtract from constant and scale.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nValue Constant value each vector element is to be subtracted from
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSubCRev_32s_ISfs
     (nValue : nppdefs_h.Npp32s;
      pSrcDst : access nppdefs_h.Npp32s;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:974
   pragma Import (C, nppsSubCRev_32s_ISfs, "nppsSubCRev_32s_ISfs");

  --* 
  -- * 32-bit signed integersignal subtract from constant and scale.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nValue Constant value each vector element is to be subtracted from
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSubCRev_32s_Sfs
     (pSrc : access nppdefs_h.Npp32s;
      nValue : nppdefs_h.Npp32s;
      pDst : access nppdefs_h.Npp32s;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:986
   pragma Import (C, nppsSubCRev_32s_Sfs, "nppsSubCRev_32s_Sfs");

  --* 
  -- * 32-bit integer complex number (32 bit real, 32 bit imaginary) in place signal
  -- * subtract from constant and scale.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nValue Constant value each vector element is to be subtracted from
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSubCRev_32sc_ISfs
     (nValue : nppdefs_h.Npp32sc;
      pSrcDst : access nppdefs_h.Npp32sc;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:998
   pragma Import (C, nppsSubCRev_32sc_ISfs, "nppsSubCRev_32sc_ISfs");

  --* 
  -- * 32-bit integer complex number (32 bit real, 32 bit imaginary) signal subtract from constant
  -- * and scale.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nValue Constant value each vector element is to be subtracted from
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSubCRev_32sc_Sfs
     (pSrc : access constant nppdefs_h.Npp32sc;
      nValue : nppdefs_h.Npp32sc;
      pDst : access nppdefs_h.Npp32sc;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:1011
   pragma Import (C, nppsSubCRev_32sc_Sfs, "nppsSubCRev_32sc_Sfs");

  --* 
  -- * 32-bit floating point in place signal subtract from constant.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nValue Constant value each vector element is to be subtracted from
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSubCRev_32f_I
     (nValue : nppdefs_h.Npp32f;
      pSrcDst : access nppdefs_h.Npp32f;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:1021
   pragma Import (C, nppsSubCRev_32f_I, "nppsSubCRev_32f_I");

  --* 
  -- * 32-bit floating point signal subtract from constant.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nValue Constant value each vector element is to be subtracted from
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSubCRev_32f
     (pSrc : access nppdefs_h.Npp32f;
      nValue : nppdefs_h.Npp32f;
      pDst : access nppdefs_h.Npp32f;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:1032
   pragma Import (C, nppsSubCRev_32f, "nppsSubCRev_32f");

  --* 
  -- * 32-bit floating point complex number (32 bit real, 32 bit imaginary) in
  -- * place signal subtract from constant.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nValue Constant value each vector element is to be subtracted from
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSubCRev_32fc_I
     (nValue : nppdefs_h.Npp32fc;
      pSrcDst : access nppdefs_h.Npp32fc;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:1043
   pragma Import (C, nppsSubCRev_32fc_I, "nppsSubCRev_32fc_I");

  --* 
  -- * 32-bit floating point complex number (32 bit real, 32 bit imaginary) signal
  -- * subtract from constant.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nValue Constant value each vector element is to be subtracted from
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSubCRev_32fc
     (pSrc : access constant nppdefs_h.Npp32fc;
      nValue : nppdefs_h.Npp32fc;
      pDst : access nppdefs_h.Npp32fc;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:1055
   pragma Import (C, nppsSubCRev_32fc, "nppsSubCRev_32fc");

  --* 
  -- * 64-bit floating point, in place signal subtract from constant.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nValue Constant value each vector element is to be subtracted from
  -- * \param nLength Length of the vectors, number of items.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSubCRev_64f_I
     (nValue : nppdefs_h.Npp64f;
      pSrcDst : access nppdefs_h.Npp64f;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:1065
   pragma Import (C, nppsSubCRev_64f_I, "nppsSubCRev_64f_I");

  --* 
  -- * 64-bit floating point signal subtract from constant.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nValue Constant value each vector element is to be subtracted from
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSubCRev_64f
     (pSrc : access nppdefs_h.Npp64f;
      nValue : nppdefs_h.Npp64f;
      pDst : access nppdefs_h.Npp64f;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:1076
   pragma Import (C, nppsSubCRev_64f, "nppsSubCRev_64f");

  --* 
  -- * 64-bit floating point complex number (64 bit real, 64 bit imaginary) in
  -- * place signal subtract from constant.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nValue Constant value each vector element is to be subtracted from
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSubCRev_64fc_I
     (nValue : nppdefs_h.Npp64fc;
      pSrcDst : access nppdefs_h.Npp64fc;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:1087
   pragma Import (C, nppsSubCRev_64fc_I, "nppsSubCRev_64fc_I");

  --* 
  -- * 64-bit floating point complex number (64 bit real, 64 bit imaginary) signal
  -- * subtract from constant.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nValue Constant value each vector element is to be subtracted from
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSubCRev_64fc
     (pSrc : access constant nppdefs_h.Npp64fc;
      nValue : nppdefs_h.Npp64fc;
      pDst : access nppdefs_h.Npp64fc;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:1099
   pragma Import (C, nppsSubCRev_64fc, "nppsSubCRev_64fc");

  --* @} signal_subcrev  
  --* 
  -- * @defgroup signal_divc DivC
  -- *
  -- * Divides each sample of a signal by a constant.
  -- *
  -- * @{
  -- *
  --  

  --* 
  -- * 8-bit unsigned char in place signal divided by constant,
  -- * scale, then clamp to saturated value
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nValue Constant value to be divided into each vector element
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsDivC_8u_ISfs
     (nValue : nppdefs_h.Npp8u;
      pSrcDst : access nppdefs_h.Npp8u;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:1122
   pragma Import (C, nppsDivC_8u_ISfs, "nppsDivC_8u_ISfs");

  --* 
  -- * 8-bit unsigned char signal divided by constant, scale, then clamp to saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nValue Constant value to be divided into each vector element
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsDivC_8u_Sfs
     (pSrc : access nppdefs_h.Npp8u;
      nValue : nppdefs_h.Npp8u;
      pDst : access nppdefs_h.Npp8u;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:1134
   pragma Import (C, nppsDivC_8u_Sfs, "nppsDivC_8u_Sfs");

  --* 
  -- * 16-bit unsigned short in place signal divided by constant, scale, then clamp to saturated value.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nValue Constant value to be divided into each vector element
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsDivC_16u_ISfs
     (nValue : nppdefs_h.Npp16u;
      pSrcDst : access nppdefs_h.Npp16u;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:1145
   pragma Import (C, nppsDivC_16u_ISfs, "nppsDivC_16u_ISfs");

  --* 
  -- * 16-bit unsigned short signal divided by constant, scale, then clamp to saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nValue Constant value to be divided into each vector element
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsDivC_16u_Sfs
     (pSrc : access nppdefs_h.Npp16u;
      nValue : nppdefs_h.Npp16u;
      pDst : access nppdefs_h.Npp16u;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:1157
   pragma Import (C, nppsDivC_16u_Sfs, "nppsDivC_16u_Sfs");

  --* 
  -- * 16-bit signed short in place signal divided by constant, scale, then clamp to saturated value.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nValue Constant value to be divided into each vector element
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsDivC_16s_ISfs
     (nValue : nppdefs_h.Npp16s;
      pSrcDst : access nppdefs_h.Npp16s;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:1168
   pragma Import (C, nppsDivC_16s_ISfs, "nppsDivC_16s_ISfs");

  --* 
  -- * 16-bit signed short signal divided by constant, scale, then clamp to saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nValue Constant value to be divided into each vector element
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsDivC_16s_Sfs
     (pSrc : access nppdefs_h.Npp16s;
      nValue : nppdefs_h.Npp16s;
      pDst : access nppdefs_h.Npp16s;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:1180
   pragma Import (C, nppsDivC_16s_Sfs, "nppsDivC_16s_Sfs");

  --* 
  -- * 16-bit integer complex number (16 bit real, 16 bit imaginary)signal divided by constant, 
  -- * scale, then clamp to saturated value.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nValue Constant value to be divided into each vector element
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsDivC_16sc_ISfs
     (nValue : nppdefs_h.Npp16sc;
      pSrcDst : access nppdefs_h.Npp16sc;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:1192
   pragma Import (C, nppsDivC_16sc_ISfs, "nppsDivC_16sc_ISfs");

  --* 
  -- * 16-bit integer complex number (16 bit real, 16 bit imaginary) signal divided by constant,
  -- * scale, then clamp to saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nValue Constant value to be divided into each vector element
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsDivC_16sc_Sfs
     (pSrc : access constant nppdefs_h.Npp16sc;
      nValue : nppdefs_h.Npp16sc;
      pDst : access nppdefs_h.Npp16sc;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:1205
   pragma Import (C, nppsDivC_16sc_Sfs, "nppsDivC_16sc_Sfs");

  --* 
  -- * 32-bit floating point in place signal divided by constant.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nValue Constant value to be divided into each vector element
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsDivC_32f_I
     (nValue : nppdefs_h.Npp32f;
      pSrcDst : access nppdefs_h.Npp32f;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:1215
   pragma Import (C, nppsDivC_32f_I, "nppsDivC_32f_I");

  --* 
  -- * 32-bit floating point signal divided by constant.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nValue Constant value to be divided into each vector element
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsDivC_32f
     (pSrc : access nppdefs_h.Npp32f;
      nValue : nppdefs_h.Npp32f;
      pDst : access nppdefs_h.Npp32f;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:1226
   pragma Import (C, nppsDivC_32f, "nppsDivC_32f");

  --* 
  -- * 32-bit floating point complex number (32 bit real, 32 bit imaginary) in
  -- * place signal divided by constant.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nValue Constant value to be divided into each vector element
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsDivC_32fc_I
     (nValue : nppdefs_h.Npp32fc;
      pSrcDst : access nppdefs_h.Npp32fc;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:1237
   pragma Import (C, nppsDivC_32fc_I, "nppsDivC_32fc_I");

  --* 
  -- * 32-bit floating point complex number (32 bit real, 32 bit imaginary) signal
  -- * divided by constant.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nValue Constant value to be divided into each vector element
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsDivC_32fc
     (pSrc : access constant nppdefs_h.Npp32fc;
      nValue : nppdefs_h.Npp32fc;
      pDst : access nppdefs_h.Npp32fc;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:1249
   pragma Import (C, nppsDivC_32fc, "nppsDivC_32fc");

  --* 
  -- * 64-bit floating point in place signal divided by constant.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nValue Constant value to be divided into each vector element
  -- * \param nLength Length of the vectors, number of items.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsDivC_64f_I
     (nValue : nppdefs_h.Npp64f;
      pSrcDst : access nppdefs_h.Npp64f;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:1259
   pragma Import (C, nppsDivC_64f_I, "nppsDivC_64f_I");

  --* 
  -- * 64-bit floating point signal divided by constant.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nValue Constant value to be divided into each vector element
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsDivC_64f
     (pSrc : access nppdefs_h.Npp64f;
      nValue : nppdefs_h.Npp64f;
      pDst : access nppdefs_h.Npp64f;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:1270
   pragma Import (C, nppsDivC_64f, "nppsDivC_64f");

  --* 
  -- * 64-bit floating point complex number (64 bit real, 64 bit imaginary) in
  -- * place signal divided by constant.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nValue Constant value to be divided into each vector element
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsDivC_64fc_I
     (nValue : nppdefs_h.Npp64fc;
      pSrcDst : access nppdefs_h.Npp64fc;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:1281
   pragma Import (C, nppsDivC_64fc_I, "nppsDivC_64fc_I");

  --* 
  -- * 64-bit floating point complex number (64 bit real, 64 bit imaginary) signal
  -- * divided by constant.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nValue Constant value to be divided into each vector element
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsDivC_64fc
     (pSrc : access constant nppdefs_h.Npp64fc;
      nValue : nppdefs_h.Npp64fc;
      pDst : access nppdefs_h.Npp64fc;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:1293
   pragma Import (C, nppsDivC_64fc, "nppsDivC_64fc");

  --* @} signal_divc  
  --* 
  -- * @defgroup signal_divcrev DivCRev
  -- *
  -- * Divides a constant by each sample of a signal.
  -- *
  -- * @{
  -- *
  --  

  --* 
  -- * 16-bit unsigned short in place constant divided by signal, then clamp to saturated value.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nValue Constant value to be divided by each vector element
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsDivCRev_16u_I
     (nValue : nppdefs_h.Npp16u;
      pSrcDst : access nppdefs_h.Npp16u;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:1314
   pragma Import (C, nppsDivCRev_16u_I, "nppsDivCRev_16u_I");

  --* 
  -- * 16-bit unsigned short signal divided by constant, then clamp to saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nValue Constant value to be divided by each vector element
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsDivCRev_16u
     (pSrc : access nppdefs_h.Npp16u;
      nValue : nppdefs_h.Npp16u;
      pDst : access nppdefs_h.Npp16u;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:1325
   pragma Import (C, nppsDivCRev_16u, "nppsDivCRev_16u");

  --* 
  -- * 32-bit floating point in place constant divided by signal.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nValue Constant value to be divided by each vector element
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsDivCRev_32f_I
     (nValue : nppdefs_h.Npp32f;
      pSrcDst : access nppdefs_h.Npp32f;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:1335
   pragma Import (C, nppsDivCRev_32f_I, "nppsDivCRev_32f_I");

  --* 
  -- * 32-bit floating point constant divided by signal.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nValue Constant value to be divided by each vector element
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsDivCRev_32f
     (pSrc : access nppdefs_h.Npp32f;
      nValue : nppdefs_h.Npp32f;
      pDst : access nppdefs_h.Npp32f;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:1346
   pragma Import (C, nppsDivCRev_32f, "nppsDivCRev_32f");

  --* @} signal_divcrev  
  --* 
  -- * @defgroup signal_add Add
  -- *
  -- * Sample by sample addition of two signals.
  -- *
  -- * @{
  -- *
  --  

  --* 
  -- * 16-bit signed short signal add signal,
  -- * then clamp to saturated value.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer. signal2 elements to be added to signal1 elements
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAdd_16s
     (pSrc1 : access nppdefs_h.Npp16s;
      pSrc2 : access nppdefs_h.Npp16s;
      pDst : access nppdefs_h.Npp16s;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:1369
   pragma Import (C, nppsAdd_16s, "nppsAdd_16s");

  --* 
  -- * 16-bit unsigned short signal add signal,
  -- * then clamp to saturated value.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer. signal2 elements to be added to signal1 elements
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAdd_16u
     (pSrc1 : access nppdefs_h.Npp16u;
      pSrc2 : access nppdefs_h.Npp16u;
      pDst : access nppdefs_h.Npp16u;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:1381
   pragma Import (C, nppsAdd_16u, "nppsAdd_16u");

  --* 
  -- * 32-bit unsigned int signal add signal,
  -- * then clamp to saturated value.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer. signal2 elements to be added to signal1 elements
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAdd_32u
     (pSrc1 : access nppdefs_h.Npp32u;
      pSrc2 : access nppdefs_h.Npp32u;
      pDst : access nppdefs_h.Npp32u;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:1393
   pragma Import (C, nppsAdd_32u, "nppsAdd_32u");

  --* 
  -- * 32-bit floating point signal add signal,
  -- * then clamp to saturated value.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer. signal2 elements to be added to signal1 elements
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAdd_32f
     (pSrc1 : access nppdefs_h.Npp32f;
      pSrc2 : access nppdefs_h.Npp32f;
      pDst : access nppdefs_h.Npp32f;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:1405
   pragma Import (C, nppsAdd_32f, "nppsAdd_32f");

  --* 
  -- * 64-bit floating point signal add signal,
  -- * then clamp to saturated value.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer. signal2 elements to be added to signal1 elements
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAdd_64f
     (pSrc1 : access nppdefs_h.Npp64f;
      pSrc2 : access nppdefs_h.Npp64f;
      pDst : access nppdefs_h.Npp64f;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:1417
   pragma Import (C, nppsAdd_64f, "nppsAdd_64f");

  --* 
  -- * 32-bit complex floating point signal add signal,
  -- * then clamp to saturated value.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer. signal2 elements to be added to signal1 elements
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAdd_32fc
     (pSrc1 : access constant nppdefs_h.Npp32fc;
      pSrc2 : access constant nppdefs_h.Npp32fc;
      pDst : access nppdefs_h.Npp32fc;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:1429
   pragma Import (C, nppsAdd_32fc, "nppsAdd_32fc");

  --* 
  -- * 64-bit complex floating point signal add signal,
  -- * then clamp to saturated value.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer. signal2 elements to be added to signal1 elements
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAdd_64fc
     (pSrc1 : access constant nppdefs_h.Npp64fc;
      pSrc2 : access constant nppdefs_h.Npp64fc;
      pDst : access nppdefs_h.Npp64fc;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:1441
   pragma Import (C, nppsAdd_64fc, "nppsAdd_64fc");

  --* 
  -- * 8-bit unsigned char signal add signal with 16-bit unsigned result,
  -- * then clamp to saturated value.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer. signal2 elements to be added to signal1 elements
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAdd_8u16u
     (pSrc1 : access nppdefs_h.Npp8u;
      pSrc2 : access nppdefs_h.Npp8u;
      pDst : access nppdefs_h.Npp16u;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:1453
   pragma Import (C, nppsAdd_8u16u, "nppsAdd_8u16u");

  --* 
  -- * 16-bit signed short signal add signal with 32-bit floating point result,
  -- * then clamp to saturated value.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer. signal2 elements to be added to signal1 elements
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAdd_16s32f
     (pSrc1 : access nppdefs_h.Npp16s;
      pSrc2 : access nppdefs_h.Npp16s;
      pDst : access nppdefs_h.Npp32f;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:1465
   pragma Import (C, nppsAdd_16s32f, "nppsAdd_16s32f");

  --* 
  -- * 8-bit unsigned char add signal, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer, signal2 elements to be added to signal1 elements.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAdd_8u_Sfs
     (pSrc1 : access nppdefs_h.Npp8u;
      pSrc2 : access nppdefs_h.Npp8u;
      pDst : access nppdefs_h.Npp8u;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:1477
   pragma Import (C, nppsAdd_8u_Sfs, "nppsAdd_8u_Sfs");

  --* 
  -- * 16-bit unsigned short add signal, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer, signal2 elements to be added to signal1 elements.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAdd_16u_Sfs
     (pSrc1 : access nppdefs_h.Npp16u;
      pSrc2 : access nppdefs_h.Npp16u;
      pDst : access nppdefs_h.Npp16u;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:1489
   pragma Import (C, nppsAdd_16u_Sfs, "nppsAdd_16u_Sfs");

  --* 
  -- * 16-bit signed short add signal, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer, signal2 elements to be added to signal1 elements.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAdd_16s_Sfs
     (pSrc1 : access nppdefs_h.Npp16s;
      pSrc2 : access nppdefs_h.Npp16s;
      pDst : access nppdefs_h.Npp16s;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:1501
   pragma Import (C, nppsAdd_16s_Sfs, "nppsAdd_16s_Sfs");

  --* 
  -- * 32-bit signed integer add signal, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer, signal2 elements to be added to signal1 elements.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAdd_32s_Sfs
     (pSrc1 : access nppdefs_h.Npp32s;
      pSrc2 : access nppdefs_h.Npp32s;
      pDst : access nppdefs_h.Npp32s;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:1513
   pragma Import (C, nppsAdd_32s_Sfs, "nppsAdd_32s_Sfs");

  --* 
  -- * 64-bit signed integer add signal, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer, signal2 elements to be added to signal1 elements.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAdd_64s_Sfs
     (pSrc1 : access nppdefs_h.Npp64s;
      pSrc2 : access nppdefs_h.Npp64s;
      pDst : access nppdefs_h.Npp64s;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:1525
   pragma Import (C, nppsAdd_64s_Sfs, "nppsAdd_64s_Sfs");

  --* 
  -- * 16-bit signed complex short add signal, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer, signal2 elements to be added to signal1 elements.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAdd_16sc_Sfs
     (pSrc1 : access constant nppdefs_h.Npp16sc;
      pSrc2 : access constant nppdefs_h.Npp16sc;
      pDst : access nppdefs_h.Npp16sc;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:1537
   pragma Import (C, nppsAdd_16sc_Sfs, "nppsAdd_16sc_Sfs");

  --* 
  -- * 32-bit signed complex integer add signal, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer, signal2 elements to be added to signal1 elements.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAdd_32sc_Sfs
     (pSrc1 : access constant nppdefs_h.Npp32sc;
      pSrc2 : access constant nppdefs_h.Npp32sc;
      pDst : access nppdefs_h.Npp32sc;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:1549
   pragma Import (C, nppsAdd_32sc_Sfs, "nppsAdd_32sc_Sfs");

  --* 
  -- * 16-bit signed short in place signal add signal,
  -- * then clamp to saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be added to signal1 elements
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAdd_16s_I
     (pSrc : access nppdefs_h.Npp16s;
      pSrcDst : access nppdefs_h.Npp16s;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:1560
   pragma Import (C, nppsAdd_16s_I, "nppsAdd_16s_I");

  --* 
  -- * 32-bit floating point in place signal add signal,
  -- * then clamp to saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be added to signal1 elements
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAdd_32f_I
     (pSrc : access nppdefs_h.Npp32f;
      pSrcDst : access nppdefs_h.Npp32f;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:1571
   pragma Import (C, nppsAdd_32f_I, "nppsAdd_32f_I");

  --* 
  -- * 64-bit floating point in place signal add signal,
  -- * then clamp to saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be added to signal1 elements
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAdd_64f_I
     (pSrc : access nppdefs_h.Npp64f;
      pSrcDst : access nppdefs_h.Npp64f;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:1582
   pragma Import (C, nppsAdd_64f_I, "nppsAdd_64f_I");

  --* 
  -- * 32-bit complex floating point in place signal add signal,
  -- * then clamp to saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be added to signal1 elements
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAdd_32fc_I
     (pSrc : access constant nppdefs_h.Npp32fc;
      pSrcDst : access nppdefs_h.Npp32fc;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:1593
   pragma Import (C, nppsAdd_32fc_I, "nppsAdd_32fc_I");

  --* 
  -- * 64-bit complex floating point in place signal add signal,
  -- * then clamp to saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be added to signal1 elements
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAdd_64fc_I
     (pSrc : access constant nppdefs_h.Npp64fc;
      pSrcDst : access nppdefs_h.Npp64fc;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:1604
   pragma Import (C, nppsAdd_64fc_I, "nppsAdd_64fc_I");

  --* 
  -- * 16/32-bit signed short in place signal add signal with 32-bit signed integer results,
  -- * then clamp to saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be added to signal1 elements
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAdd_16s32s_I
     (pSrc : access nppdefs_h.Npp16s;
      pSrcDst : access nppdefs_h.Npp32s;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:1615
   pragma Import (C, nppsAdd_16s32s_I, "nppsAdd_16s32s_I");

  --* 
  -- * 8-bit unsigned char in place signal add signal, with scaling,
  -- * then clamp to saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be added to signal1 elements
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAdd_8u_ISfs
     (pSrc : access nppdefs_h.Npp8u;
      pSrcDst : access nppdefs_h.Npp8u;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:1627
   pragma Import (C, nppsAdd_8u_ISfs, "nppsAdd_8u_ISfs");

  --* 
  -- * 16-bit unsigned short in place signal add signal, with scaling,
  -- * then clamp to saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be added to signal1 elements
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAdd_16u_ISfs
     (pSrc : access nppdefs_h.Npp16u;
      pSrcDst : access nppdefs_h.Npp16u;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:1639
   pragma Import (C, nppsAdd_16u_ISfs, "nppsAdd_16u_ISfs");

  --* 
  -- * 16-bit signed short in place signal add signal, with scaling, 
  -- * then clamp to saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be added to signal1 elements
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAdd_16s_ISfs
     (pSrc : access nppdefs_h.Npp16s;
      pSrcDst : access nppdefs_h.Npp16s;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:1651
   pragma Import (C, nppsAdd_16s_ISfs, "nppsAdd_16s_ISfs");

  --* 
  -- * 32-bit signed integer in place signal add signal, with scaling, 
  -- * then clamp to saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be added to signal1 elements
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAdd_32s_ISfs
     (pSrc : access nppdefs_h.Npp32s;
      pSrcDst : access nppdefs_h.Npp32s;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:1663
   pragma Import (C, nppsAdd_32s_ISfs, "nppsAdd_32s_ISfs");

  --* 
  -- * 16-bit complex signed short in place signal add signal, with scaling, 
  -- * then clamp to saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be added to signal1 elements
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAdd_16sc_ISfs
     (pSrc : access constant nppdefs_h.Npp16sc;
      pSrcDst : access nppdefs_h.Npp16sc;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:1675
   pragma Import (C, nppsAdd_16sc_ISfs, "nppsAdd_16sc_ISfs");

  --* 
  -- * 32-bit complex signed integer in place signal add signal, with scaling, 
  -- * then clamp to saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be added to signal1 elements
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAdd_32sc_ISfs
     (pSrc : access constant nppdefs_h.Npp32sc;
      pSrcDst : access nppdefs_h.Npp32sc;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:1687
   pragma Import (C, nppsAdd_32sc_ISfs, "nppsAdd_32sc_ISfs");

  --* @} signal_add  
  --* 
  -- * @defgroup signal_addproduct AddProduct
  -- *
  -- * Adds sample by sample product of two signals to the destination signal.
  -- *
  -- * @{
  -- *
  --  

  --* 
  -- * 32-bit floating point signal add product of source signal times destination signal to destination signal,
  -- * then clamp to saturated value.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer. product of source1 and source2 signal elements to be added to destination elements
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAddProduct_32f
     (pSrc1 : access nppdefs_h.Npp32f;
      pSrc2 : access nppdefs_h.Npp32f;
      pDst : access nppdefs_h.Npp32f;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:1710
   pragma Import (C, nppsAddProduct_32f, "nppsAddProduct_32f");

  --* 
  -- * 64-bit floating point signal add product of source signal times destination signal to destination signal,
  -- * then clamp to saturated value.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer. product of source1 and source2 signal elements to be added to destination elements
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAddProduct_64f
     (pSrc1 : access nppdefs_h.Npp64f;
      pSrc2 : access nppdefs_h.Npp64f;
      pDst : access nppdefs_h.Npp64f;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:1722
   pragma Import (C, nppsAddProduct_64f, "nppsAddProduct_64f");

  --* 
  -- * 32-bit complex floating point signal add product of source signal times destination signal to destination signal,
  -- * then clamp to saturated value.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer. product of source1 and source2 signal elements to be added to destination elements
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAddProduct_32fc
     (pSrc1 : access constant nppdefs_h.Npp32fc;
      pSrc2 : access constant nppdefs_h.Npp32fc;
      pDst : access nppdefs_h.Npp32fc;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:1734
   pragma Import (C, nppsAddProduct_32fc, "nppsAddProduct_32fc");

  --* 
  -- * 64-bit complex floating point signal add product of source signal times destination signal to destination signal,
  -- * then clamp to saturated value.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer. product of source1 and source2 signal elements to be added to destination elements
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAddProduct_64fc
     (pSrc1 : access constant nppdefs_h.Npp64fc;
      pSrc2 : access constant nppdefs_h.Npp64fc;
      pDst : access nppdefs_h.Npp64fc;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:1746
   pragma Import (C, nppsAddProduct_64fc, "nppsAddProduct_64fc");

  --* 
  -- * 16-bit signed short signal add product of source signal1 times source signal2 to destination signal,
  -- * with scaling, then clamp to saturated value.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer. product of source1 and source2 signal elements to be added to destination elements
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAddProduct_16s_Sfs
     (pSrc1 : access nppdefs_h.Npp16s;
      pSrc2 : access nppdefs_h.Npp16s;
      pDst : access nppdefs_h.Npp16s;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:1759
   pragma Import (C, nppsAddProduct_16s_Sfs, "nppsAddProduct_16s_Sfs");

  --* 
  -- * 32-bit signed short signal add product of source signal1 times source signal2 to destination signal,
  -- * with scaling, then clamp to saturated value.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer. product of source1 and source2 signal elements to be added to destination elements
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAddProduct_32s_Sfs
     (pSrc1 : access nppdefs_h.Npp32s;
      pSrc2 : access nppdefs_h.Npp32s;
      pDst : access nppdefs_h.Npp32s;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:1772
   pragma Import (C, nppsAddProduct_32s_Sfs, "nppsAddProduct_32s_Sfs");

  --* 
  -- * 16-bit signed short signal add product of source signal1 times source signal2 to 32-bit signed integer destination signal,
  -- * with scaling, then clamp to saturated value.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer. product of source1 and source2 signal elements to be added to destination elements
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAddProduct_16s32s_Sfs
     (pSrc1 : access nppdefs_h.Npp16s;
      pSrc2 : access nppdefs_h.Npp16s;
      pDst : access nppdefs_h.Npp32s;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:1785
   pragma Import (C, nppsAddProduct_16s32s_Sfs, "nppsAddProduct_16s32s_Sfs");

  --* @} signal_addproduct  
  --* 
  -- * @defgroup signal_mul Mul
  -- *
  -- * Sample by sample multiplication the samples of two signals.
  -- *
  -- * @{
  -- *
  --  

  --* 
  -- * 16-bit signed short signal times signal,
  -- * then clamp to saturated value.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer. signal2 elements to be multiplied by signal1 elements
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMul_16s
     (pSrc1 : access nppdefs_h.Npp16s;
      pSrc2 : access nppdefs_h.Npp16s;
      pDst : access nppdefs_h.Npp16s;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:1808
   pragma Import (C, nppsMul_16s, "nppsMul_16s");

  --* 
  -- * 32-bit floating point signal times signal,
  -- * then clamp to saturated value.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer. signal2 elements to be multiplied by signal1 elements
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMul_32f
     (pSrc1 : access nppdefs_h.Npp32f;
      pSrc2 : access nppdefs_h.Npp32f;
      pDst : access nppdefs_h.Npp32f;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:1820
   pragma Import (C, nppsMul_32f, "nppsMul_32f");

  --* 
  -- * 64-bit floating point signal times signal,
  -- * then clamp to saturated value.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer. signal2 elements to be multiplied by signal1 elements
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMul_64f
     (pSrc1 : access nppdefs_h.Npp64f;
      pSrc2 : access nppdefs_h.Npp64f;
      pDst : access nppdefs_h.Npp64f;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:1832
   pragma Import (C, nppsMul_64f, "nppsMul_64f");

  --* 
  -- * 32-bit complex floating point signal times signal,
  -- * then clamp to saturated value.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer. signal2 elements to be multiplied by signal1 elements
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMul_32fc
     (pSrc1 : access constant nppdefs_h.Npp32fc;
      pSrc2 : access constant nppdefs_h.Npp32fc;
      pDst : access nppdefs_h.Npp32fc;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:1844
   pragma Import (C, nppsMul_32fc, "nppsMul_32fc");

  --* 
  -- * 64-bit complex floating point signal times signal,
  -- * then clamp to saturated value.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer. signal2 elements to be multiplied by signal1 elements
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMul_64fc
     (pSrc1 : access constant nppdefs_h.Npp64fc;
      pSrc2 : access constant nppdefs_h.Npp64fc;
      pDst : access nppdefs_h.Npp64fc;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:1856
   pragma Import (C, nppsMul_64fc, "nppsMul_64fc");

  --* 
  -- * 8-bit unsigned char signal times signal with 16-bit unsigned result,
  -- * then clamp to saturated value.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer. signal2 elements to be multiplied by signal1 elements
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMul_8u16u
     (pSrc1 : access nppdefs_h.Npp8u;
      pSrc2 : access nppdefs_h.Npp8u;
      pDst : access nppdefs_h.Npp16u;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:1868
   pragma Import (C, nppsMul_8u16u, "nppsMul_8u16u");

  --* 
  -- * 16-bit signed short signal times signal with 32-bit floating point result,
  -- * then clamp to saturated value.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer. signal2 elements to be multiplied by signal1 elements
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMul_16s32f
     (pSrc1 : access nppdefs_h.Npp16s;
      pSrc2 : access nppdefs_h.Npp16s;
      pDst : access nppdefs_h.Npp32f;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:1880
   pragma Import (C, nppsMul_16s32f, "nppsMul_16s32f");

  --* 
  -- * 32-bit floating point signal times 32-bit complex floating point signal with complex 32-bit floating point result,
  -- * then clamp to saturated value.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer. signal2 elements to be multiplied by signal1 elements
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMul_32f32fc
     (pSrc1 : access nppdefs_h.Npp32f;
      pSrc2 : access constant nppdefs_h.Npp32fc;
      pDst : access nppdefs_h.Npp32fc;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:1892
   pragma Import (C, nppsMul_32f32fc, "nppsMul_32f32fc");

  --* 
  -- * 8-bit unsigned char signal times signal, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer, signal2 elements to be multiplied by signal1 elements.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMul_8u_Sfs
     (pSrc1 : access nppdefs_h.Npp8u;
      pSrc2 : access nppdefs_h.Npp8u;
      pDst : access nppdefs_h.Npp8u;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:1904
   pragma Import (C, nppsMul_8u_Sfs, "nppsMul_8u_Sfs");

  --* 
  -- * 16-bit unsigned short signal time signal, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer, signal2 elements to be multiplied by signal1 elements.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMul_16u_Sfs
     (pSrc1 : access nppdefs_h.Npp16u;
      pSrc2 : access nppdefs_h.Npp16u;
      pDst : access nppdefs_h.Npp16u;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:1916
   pragma Import (C, nppsMul_16u_Sfs, "nppsMul_16u_Sfs");

  --* 
  -- * 16-bit signed short signal times signal, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer, signal2 elements to be multiplied by signal1 elements.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMul_16s_Sfs
     (pSrc1 : access nppdefs_h.Npp16s;
      pSrc2 : access nppdefs_h.Npp16s;
      pDst : access nppdefs_h.Npp16s;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:1928
   pragma Import (C, nppsMul_16s_Sfs, "nppsMul_16s_Sfs");

  --* 
  -- * 32-bit signed integer signal times signal, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer, signal2 elements to be multiplied by signal1 elements.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMul_32s_Sfs
     (pSrc1 : access nppdefs_h.Npp32s;
      pSrc2 : access nppdefs_h.Npp32s;
      pDst : access nppdefs_h.Npp32s;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:1940
   pragma Import (C, nppsMul_32s_Sfs, "nppsMul_32s_Sfs");

  --* 
  -- * 16-bit signed complex short signal times signal, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer, signal2 elements to be multiplied by signal1 elements.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMul_16sc_Sfs
     (pSrc1 : access constant nppdefs_h.Npp16sc;
      pSrc2 : access constant nppdefs_h.Npp16sc;
      pDst : access nppdefs_h.Npp16sc;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:1952
   pragma Import (C, nppsMul_16sc_Sfs, "nppsMul_16sc_Sfs");

  --* 
  -- * 32-bit signed complex integer signal times signal, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer, signal2 elements to be multiplied by signal1 elements.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMul_32sc_Sfs
     (pSrc1 : access constant nppdefs_h.Npp32sc;
      pSrc2 : access constant nppdefs_h.Npp32sc;
      pDst : access nppdefs_h.Npp32sc;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:1964
   pragma Import (C, nppsMul_32sc_Sfs, "nppsMul_32sc_Sfs");

  --* 
  -- * 16-bit unsigned short signal times 16-bit signed short signal, scale, then clamp to 16-bit signed saturated value.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer, signal2 elements to be multiplied by signal1 elements.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMul_16u16s_Sfs
     (pSrc1 : access nppdefs_h.Npp16u;
      pSrc2 : access nppdefs_h.Npp16s;
      pDst : access nppdefs_h.Npp16s;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:1976
   pragma Import (C, nppsMul_16u16s_Sfs, "nppsMul_16u16s_Sfs");

  --* 
  -- * 16-bit signed short signal times signal, scale, then clamp to 32-bit signed saturated value.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer, signal2 elements to be multiplied by signal1 elements.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMul_16s32s_Sfs
     (pSrc1 : access nppdefs_h.Npp16s;
      pSrc2 : access nppdefs_h.Npp16s;
      pDst : access nppdefs_h.Npp32s;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:1988
   pragma Import (C, nppsMul_16s32s_Sfs, "nppsMul_16s32s_Sfs");

  --* 
  -- * 32-bit signed integer signal times 32-bit complex signed integer signal, scale, then clamp to 32-bit complex integer saturated value.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer, signal2 elements to be multiplied by signal1 elements.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMul_32s32sc_Sfs
     (pSrc1 : access nppdefs_h.Npp32s;
      pSrc2 : access constant nppdefs_h.Npp32sc;
      pDst : access nppdefs_h.Npp32sc;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:2000
   pragma Import (C, nppsMul_32s32sc_Sfs, "nppsMul_32s32sc_Sfs");

  --* 
  -- * 32-bit signed integer signal times signal, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer, signal2 elements to be multiplied by signal1 elements.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMul_Low_32s_Sfs
     (pSrc1 : access nppdefs_h.Npp32s;
      pSrc2 : access nppdefs_h.Npp32s;
      pDst : access nppdefs_h.Npp32s;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:2012
   pragma Import (C, nppsMul_Low_32s_Sfs, "nppsMul_Low_32s_Sfs");

  --* 
  -- * 16-bit signed short in place signal times signal,
  -- * then clamp to saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be multiplied by signal1 elements
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMul_16s_I
     (pSrc : access nppdefs_h.Npp16s;
      pSrcDst : access nppdefs_h.Npp16s;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:2023
   pragma Import (C, nppsMul_16s_I, "nppsMul_16s_I");

  --* 
  -- * 32-bit floating point in place signal times signal,
  -- * then clamp to saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be multiplied by signal1 elements
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMul_32f_I
     (pSrc : access nppdefs_h.Npp32f;
      pSrcDst : access nppdefs_h.Npp32f;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:2034
   pragma Import (C, nppsMul_32f_I, "nppsMul_32f_I");

  --* 
  -- * 64-bit floating point in place signal times signal,
  -- * then clamp to saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be multiplied by signal1 elements
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMul_64f_I
     (pSrc : access nppdefs_h.Npp64f;
      pSrcDst : access nppdefs_h.Npp64f;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:2045
   pragma Import (C, nppsMul_64f_I, "nppsMul_64f_I");

  --* 
  -- * 32-bit complex floating point in place signal times signal,
  -- * then clamp to saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be multiplied by signal1 elements
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMul_32fc_I
     (pSrc : access constant nppdefs_h.Npp32fc;
      pSrcDst : access nppdefs_h.Npp32fc;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:2056
   pragma Import (C, nppsMul_32fc_I, "nppsMul_32fc_I");

  --* 
  -- * 64-bit complex floating point in place signal times signal,
  -- * then clamp to saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be multiplied by signal1 elements
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMul_64fc_I
     (pSrc : access constant nppdefs_h.Npp64fc;
      pSrcDst : access nppdefs_h.Npp64fc;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:2067
   pragma Import (C, nppsMul_64fc_I, "nppsMul_64fc_I");

  --* 
  -- * 32-bit complex floating point in place signal times 32-bit floating point signal,
  -- * then clamp to 32-bit complex floating point saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be multiplied by signal1 elements
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMul_32f32fc_I
     (pSrc : access nppdefs_h.Npp32f;
      pSrcDst : access nppdefs_h.Npp32fc;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:2078
   pragma Import (C, nppsMul_32f32fc_I, "nppsMul_32f32fc_I");

  --* 
  -- * 8-bit unsigned char in place signal times signal, with scaling,
  -- * then clamp to saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be multiplied by signal1 elements
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMul_8u_ISfs
     (pSrc : access nppdefs_h.Npp8u;
      pSrcDst : access nppdefs_h.Npp8u;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:2090
   pragma Import (C, nppsMul_8u_ISfs, "nppsMul_8u_ISfs");

  --* 
  -- * 16-bit unsigned short in place signal times signal, with scaling,
  -- * then clamp to saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be multiplied by signal1 elements
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMul_16u_ISfs
     (pSrc : access nppdefs_h.Npp16u;
      pSrcDst : access nppdefs_h.Npp16u;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:2102
   pragma Import (C, nppsMul_16u_ISfs, "nppsMul_16u_ISfs");

  --* 
  -- * 16-bit signed short in place signal times signal, with scaling, 
  -- * then clamp to saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be multiplied by signal1 elements
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMul_16s_ISfs
     (pSrc : access nppdefs_h.Npp16s;
      pSrcDst : access nppdefs_h.Npp16s;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:2114
   pragma Import (C, nppsMul_16s_ISfs, "nppsMul_16s_ISfs");

  --* 
  -- * 32-bit signed integer in place signal times signal, with scaling, 
  -- * then clamp to saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be multiplied by signal1 elements
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMul_32s_ISfs
     (pSrc : access nppdefs_h.Npp32s;
      pSrcDst : access nppdefs_h.Npp32s;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:2126
   pragma Import (C, nppsMul_32s_ISfs, "nppsMul_32s_ISfs");

  --* 
  -- * 16-bit complex signed short in place signal times signal, with scaling, 
  -- * then clamp to saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be multiplied by signal1 elements
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMul_16sc_ISfs
     (pSrc : access constant nppdefs_h.Npp16sc;
      pSrcDst : access nppdefs_h.Npp16sc;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:2138
   pragma Import (C, nppsMul_16sc_ISfs, "nppsMul_16sc_ISfs");

  --* 
  -- * 32-bit complex signed integer in place signal times signal, with scaling, 
  -- * then clamp to saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be multiplied by signal1 elements
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMul_32sc_ISfs
     (pSrc : access constant nppdefs_h.Npp32sc;
      pSrcDst : access nppdefs_h.Npp32sc;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:2150
   pragma Import (C, nppsMul_32sc_ISfs, "nppsMul_32sc_ISfs");

  --* 
  -- * 32-bit complex signed integer in place signal times 32-bit signed integer signal, with scaling, 
  -- * then clamp to saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be multiplied by signal1 elements
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMul_32s32sc_ISfs
     (pSrc : access nppdefs_h.Npp32s;
      pSrcDst : access nppdefs_h.Npp32sc;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:2162
   pragma Import (C, nppsMul_32s32sc_ISfs, "nppsMul_32s32sc_ISfs");

  --* @} signal_mul  
  --* 
  -- * @defgroup signal_sub Sub
  -- *
  -- * Sample by sample subtraction of the samples of two signals.
  -- *
  -- * @{
  -- *
  --  

  --* 
  -- * 16-bit signed short signal subtract signal,
  -- * then clamp to saturated value.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer. signal1 elements to be subtracted from signal2 elements
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSub_16s
     (pSrc1 : access nppdefs_h.Npp16s;
      pSrc2 : access nppdefs_h.Npp16s;
      pDst : access nppdefs_h.Npp16s;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:2185
   pragma Import (C, nppsSub_16s, "nppsSub_16s");

  --* 
  -- * 32-bit floating point signal subtract signal,
  -- * then clamp to saturated value.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer. signal1 elements to be subtracted from signal2 elements
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSub_32f
     (pSrc1 : access nppdefs_h.Npp32f;
      pSrc2 : access nppdefs_h.Npp32f;
      pDst : access nppdefs_h.Npp32f;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:2197
   pragma Import (C, nppsSub_32f, "nppsSub_32f");

  --* 
  -- * 64-bit floating point signal subtract signal,
  -- * then clamp to saturated value.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer. signal1 elements to be subtracted from signal2 elements
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSub_64f
     (pSrc1 : access nppdefs_h.Npp64f;
      pSrc2 : access nppdefs_h.Npp64f;
      pDst : access nppdefs_h.Npp64f;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:2209
   pragma Import (C, nppsSub_64f, "nppsSub_64f");

  --* 
  -- * 32-bit complex floating point signal subtract signal,
  -- * then clamp to saturated value.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer. signal1 elements to be subtracted from signal2 elements
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSub_32fc
     (pSrc1 : access constant nppdefs_h.Npp32fc;
      pSrc2 : access constant nppdefs_h.Npp32fc;
      pDst : access nppdefs_h.Npp32fc;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:2221
   pragma Import (C, nppsSub_32fc, "nppsSub_32fc");

  --* 
  -- * 64-bit complex floating point signal subtract signal,
  -- * then clamp to saturated value.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer. signal1 elements to be subtracted from signal2 elements
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSub_64fc
     (pSrc1 : access constant nppdefs_h.Npp64fc;
      pSrc2 : access constant nppdefs_h.Npp64fc;
      pDst : access nppdefs_h.Npp64fc;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:2233
   pragma Import (C, nppsSub_64fc, "nppsSub_64fc");

  --* 
  -- * 16-bit signed short signal subtract 16-bit signed short signal,
  -- * then clamp and convert to 32-bit floating point saturated value.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer. signal1 elements to be subtracted from signal2 elements
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSub_16s32f
     (pSrc1 : access nppdefs_h.Npp16s;
      pSrc2 : access nppdefs_h.Npp16s;
      pDst : access nppdefs_h.Npp32f;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:2245
   pragma Import (C, nppsSub_16s32f, "nppsSub_16s32f");

  --* 
  -- * 8-bit unsigned char signal subtract signal, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer, signal1 elements to be subtracted from signal2 elements.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSub_8u_Sfs
     (pSrc1 : access nppdefs_h.Npp8u;
      pSrc2 : access nppdefs_h.Npp8u;
      pDst : access nppdefs_h.Npp8u;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:2257
   pragma Import (C, nppsSub_8u_Sfs, "nppsSub_8u_Sfs");

  --* 
  -- * 16-bit unsigned short signal subtract signal, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer, signal1 elements to be subtracted from signal2 elements.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSub_16u_Sfs
     (pSrc1 : access nppdefs_h.Npp16u;
      pSrc2 : access nppdefs_h.Npp16u;
      pDst : access nppdefs_h.Npp16u;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:2269
   pragma Import (C, nppsSub_16u_Sfs, "nppsSub_16u_Sfs");

  --* 
  -- * 16-bit signed short signal subtract signal, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer, signal1 elements to be subtracted from signal2 elements.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSub_16s_Sfs
     (pSrc1 : access nppdefs_h.Npp16s;
      pSrc2 : access nppdefs_h.Npp16s;
      pDst : access nppdefs_h.Npp16s;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:2281
   pragma Import (C, nppsSub_16s_Sfs, "nppsSub_16s_Sfs");

  --* 
  -- * 32-bit signed integer signal subtract signal, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer, signal1 elements to be subtracted from signal2 elements.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSub_32s_Sfs
     (pSrc1 : access nppdefs_h.Npp32s;
      pSrc2 : access nppdefs_h.Npp32s;
      pDst : access nppdefs_h.Npp32s;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:2293
   pragma Import (C, nppsSub_32s_Sfs, "nppsSub_32s_Sfs");

  --* 
  -- * 16-bit signed complex short signal subtract signal, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer, signal1 elements to be subtracted from signal2 elements.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSub_16sc_Sfs
     (pSrc1 : access constant nppdefs_h.Npp16sc;
      pSrc2 : access constant nppdefs_h.Npp16sc;
      pDst : access nppdefs_h.Npp16sc;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:2305
   pragma Import (C, nppsSub_16sc_Sfs, "nppsSub_16sc_Sfs");

  --* 
  -- * 32-bit signed complex integer signal subtract signal, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer, signal1 elements to be subtracted from signal2 elements.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSub_32sc_Sfs
     (pSrc1 : access constant nppdefs_h.Npp32sc;
      pSrc2 : access constant nppdefs_h.Npp32sc;
      pDst : access nppdefs_h.Npp32sc;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:2317
   pragma Import (C, nppsSub_32sc_Sfs, "nppsSub_32sc_Sfs");

  --* 
  -- * 16-bit signed short in place signal subtract signal,
  -- * then clamp to saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pSrcDst \ref in_place_signal_pointer. signal1 elements to be subtracted from signal2 elements
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSub_16s_I
     (pSrc : access nppdefs_h.Npp16s;
      pSrcDst : access nppdefs_h.Npp16s;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:2328
   pragma Import (C, nppsSub_16s_I, "nppsSub_16s_I");

  --* 
  -- * 32-bit floating point in place signal subtract signal,
  -- * then clamp to saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pSrcDst \ref in_place_signal_pointer. signal1 elements to be subtracted from signal2 elements
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSub_32f_I
     (pSrc : access nppdefs_h.Npp32f;
      pSrcDst : access nppdefs_h.Npp32f;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:2339
   pragma Import (C, nppsSub_32f_I, "nppsSub_32f_I");

  --* 
  -- * 64-bit floating point in place signal subtract signal,
  -- * then clamp to saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pSrcDst \ref in_place_signal_pointer. signal1 elements to be subtracted from signal2 elements
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSub_64f_I
     (pSrc : access nppdefs_h.Npp64f;
      pSrcDst : access nppdefs_h.Npp64f;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:2350
   pragma Import (C, nppsSub_64f_I, "nppsSub_64f_I");

  --* 
  -- * 32-bit complex floating point in place signal subtract signal,
  -- * then clamp to saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pSrcDst \ref in_place_signal_pointer. signal1 elements to be subtracted from signal2 elements
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSub_32fc_I
     (pSrc : access constant nppdefs_h.Npp32fc;
      pSrcDst : access nppdefs_h.Npp32fc;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:2361
   pragma Import (C, nppsSub_32fc_I, "nppsSub_32fc_I");

  --* 
  -- * 64-bit complex floating point in place signal subtract signal,
  -- * then clamp to saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pSrcDst \ref in_place_signal_pointer. signal1 elements to be subtracted from signal2 elements
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSub_64fc_I
     (pSrc : access constant nppdefs_h.Npp64fc;
      pSrcDst : access nppdefs_h.Npp64fc;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:2372
   pragma Import (C, nppsSub_64fc_I, "nppsSub_64fc_I");

  --* 
  -- * 8-bit unsigned char in place signal subtract signal, with scaling,
  -- * then clamp to saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pSrcDst \ref in_place_signal_pointer. signal1 elements to be subtracted from signal2 elements
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSub_8u_ISfs
     (pSrc : access nppdefs_h.Npp8u;
      pSrcDst : access nppdefs_h.Npp8u;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:2384
   pragma Import (C, nppsSub_8u_ISfs, "nppsSub_8u_ISfs");

  --* 
  -- * 16-bit unsigned short in place signal subtract signal, with scaling,
  -- * then clamp to saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pSrcDst \ref in_place_signal_pointer. signal1 elements to be subtracted from signal2 elements
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSub_16u_ISfs
     (pSrc : access nppdefs_h.Npp16u;
      pSrcDst : access nppdefs_h.Npp16u;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:2396
   pragma Import (C, nppsSub_16u_ISfs, "nppsSub_16u_ISfs");

  --* 
  -- * 16-bit signed short in place signal subtract signal, with scaling, 
  -- * then clamp to saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pSrcDst \ref in_place_signal_pointer. signal1 elements to be subtracted from signal2 elements
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSub_16s_ISfs
     (pSrc : access nppdefs_h.Npp16s;
      pSrcDst : access nppdefs_h.Npp16s;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:2408
   pragma Import (C, nppsSub_16s_ISfs, "nppsSub_16s_ISfs");

  --* 
  -- * 32-bit signed integer in place signal subtract signal, with scaling, 
  -- * then clamp to saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pSrcDst \ref in_place_signal_pointer. signal1 elements to be subtracted from signal2 elements
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSub_32s_ISfs
     (pSrc : access nppdefs_h.Npp32s;
      pSrcDst : access nppdefs_h.Npp32s;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:2420
   pragma Import (C, nppsSub_32s_ISfs, "nppsSub_32s_ISfs");

  --* 
  -- * 16-bit complex signed short in place signal subtract signal, with scaling, 
  -- * then clamp to saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pSrcDst \ref in_place_signal_pointer. signal1 elements to be subtracted from signal2 elements
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSub_16sc_ISfs
     (pSrc : access constant nppdefs_h.Npp16sc;
      pSrcDst : access nppdefs_h.Npp16sc;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:2432
   pragma Import (C, nppsSub_16sc_ISfs, "nppsSub_16sc_ISfs");

  --* 
  -- * 32-bit complex signed integer in place signal subtract signal, with scaling, 
  -- * then clamp to saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pSrcDst \ref in_place_signal_pointer. signal1 elements to be subtracted from signal2 elements
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSub_32sc_ISfs
     (pSrc : access constant nppdefs_h.Npp32sc;
      pSrcDst : access nppdefs_h.Npp32sc;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:2444
   pragma Import (C, nppsSub_32sc_ISfs, "nppsSub_32sc_ISfs");

  --* @} signal_sub  
  --*
  -- * @defgroup signal_div Div
  -- *
  -- * Sample by sample division of the samples of two signals.
  -- *
  -- * @{
  -- *
  --  

  --* 
  -- * 8-bit unsigned char signal divide signal, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer, signal1 divisor elements to be divided into signal2 dividend elements.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsDiv_8u_Sfs
     (pSrc1 : access nppdefs_h.Npp8u;
      pSrc2 : access nppdefs_h.Npp8u;
      pDst : access nppdefs_h.Npp8u;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:2467
   pragma Import (C, nppsDiv_8u_Sfs, "nppsDiv_8u_Sfs");

  --* 
  -- * 16-bit unsigned short signal divide signal, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer, signal1 divisor elements to be divided into signal2 dividend elements.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsDiv_16u_Sfs
     (pSrc1 : access nppdefs_h.Npp16u;
      pSrc2 : access nppdefs_h.Npp16u;
      pDst : access nppdefs_h.Npp16u;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:2479
   pragma Import (C, nppsDiv_16u_Sfs, "nppsDiv_16u_Sfs");

  --* 
  -- * 16-bit signed short signal divide signal, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer, signal1 divisor elements to be divided into signal2 dividend elements.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsDiv_16s_Sfs
     (pSrc1 : access nppdefs_h.Npp16s;
      pSrc2 : access nppdefs_h.Npp16s;
      pDst : access nppdefs_h.Npp16s;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:2491
   pragma Import (C, nppsDiv_16s_Sfs, "nppsDiv_16s_Sfs");

  --* 
  -- * 32-bit signed integer signal divide signal, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer, signal1 divisor elements to be divided into signal2 dividend elements.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsDiv_32s_Sfs
     (pSrc1 : access nppdefs_h.Npp32s;
      pSrc2 : access nppdefs_h.Npp32s;
      pDst : access nppdefs_h.Npp32s;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:2503
   pragma Import (C, nppsDiv_32s_Sfs, "nppsDiv_32s_Sfs");

  --* 
  -- * 16-bit signed complex short signal divide signal, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer, signal1 divisor elements to be divided into signal2 dividend elements.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsDiv_16sc_Sfs
     (pSrc1 : access constant nppdefs_h.Npp16sc;
      pSrc2 : access constant nppdefs_h.Npp16sc;
      pDst : access nppdefs_h.Npp16sc;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:2515
   pragma Import (C, nppsDiv_16sc_Sfs, "nppsDiv_16sc_Sfs");

  --* 
  -- * 32-bit signed integer signal divided by 16-bit signed short signal, scale, then clamp to 16-bit signed short saturated value.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer, signal1 divisor elements to be divided into signal2 dividend elements.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsDiv_32s16s_Sfs
     (pSrc1 : access nppdefs_h.Npp16s;
      pSrc2 : access nppdefs_h.Npp32s;
      pDst : access nppdefs_h.Npp16s;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:2527
   pragma Import (C, nppsDiv_32s16s_Sfs, "nppsDiv_32s16s_Sfs");

  --* 
  -- * 32-bit floating point signal divide signal,
  -- * then clamp to saturated value.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer, signal1 divisor elements to be divided into signal2 dividend elements.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsDiv_32f
     (pSrc1 : access nppdefs_h.Npp32f;
      pSrc2 : access nppdefs_h.Npp32f;
      pDst : access nppdefs_h.Npp32f;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:2539
   pragma Import (C, nppsDiv_32f, "nppsDiv_32f");

  --* 
  -- * 64-bit floating point signal divide signal,
  -- * then clamp to saturated value.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer, signal1 divisor elements to be divided into signal2 dividend elements.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsDiv_64f
     (pSrc1 : access nppdefs_h.Npp64f;
      pSrc2 : access nppdefs_h.Npp64f;
      pDst : access nppdefs_h.Npp64f;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:2551
   pragma Import (C, nppsDiv_64f, "nppsDiv_64f");

  --* 
  -- * 32-bit complex floating point signal divide signal,
  -- * then clamp to saturated value.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer, signal1 divisor elements to be divided into signal2 dividend elements.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsDiv_32fc
     (pSrc1 : access constant nppdefs_h.Npp32fc;
      pSrc2 : access constant nppdefs_h.Npp32fc;
      pDst : access nppdefs_h.Npp32fc;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:2563
   pragma Import (C, nppsDiv_32fc, "nppsDiv_32fc");

  --* 
  -- * 64-bit complex floating point signal divide signal,
  -- * then clamp to saturated value.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer, signal1 divisor elements to be divided into signal2 dividend elements.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsDiv_64fc
     (pSrc1 : access constant nppdefs_h.Npp64fc;
      pSrc2 : access constant nppdefs_h.Npp64fc;
      pDst : access nppdefs_h.Npp64fc;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:2575
   pragma Import (C, nppsDiv_64fc, "nppsDiv_64fc");

  --* 
  -- * 8-bit unsigned char in place signal divide signal, with scaling,
  -- * then clamp to saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pSrcDst \ref in_place_signal_pointer. signal1 divisor elements to be divided into signal2 dividend elements
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsDiv_8u_ISfs
     (pSrc : access nppdefs_h.Npp8u;
      pSrcDst : access nppdefs_h.Npp8u;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:2587
   pragma Import (C, nppsDiv_8u_ISfs, "nppsDiv_8u_ISfs");

  --* 
  -- * 16-bit unsigned short in place signal divide signal, with scaling,
  -- * then clamp to saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pSrcDst \ref in_place_signal_pointer. signal1 divisor elements to be divided into signal2 dividend elements
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsDiv_16u_ISfs
     (pSrc : access nppdefs_h.Npp16u;
      pSrcDst : access nppdefs_h.Npp16u;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:2599
   pragma Import (C, nppsDiv_16u_ISfs, "nppsDiv_16u_ISfs");

  --* 
  -- * 16-bit signed short in place signal divide signal, with scaling, 
  -- * then clamp to saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pSrcDst \ref in_place_signal_pointer. signal1 divisor elements to be divided into signal2 dividend elements
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsDiv_16s_ISfs
     (pSrc : access nppdefs_h.Npp16s;
      pSrcDst : access nppdefs_h.Npp16s;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:2611
   pragma Import (C, nppsDiv_16s_ISfs, "nppsDiv_16s_ISfs");

  --* 
  -- * 16-bit complex signed short in place signal divide signal, with scaling, 
  -- * then clamp to saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pSrcDst \ref in_place_signal_pointer. signal1 divisor elements to be divided into signal2 dividend elements
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsDiv_16sc_ISfs
     (pSrc : access constant nppdefs_h.Npp16sc;
      pSrcDst : access nppdefs_h.Npp16sc;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:2623
   pragma Import (C, nppsDiv_16sc_ISfs, "nppsDiv_16sc_ISfs");

  --* 
  -- * 32-bit signed integer in place signal divide signal, with scaling, 
  -- * then clamp to saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pSrcDst \ref in_place_signal_pointer. signal1 divisor elements to be divided into signal2 dividend elements
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsDiv_32s_ISfs
     (pSrc : access nppdefs_h.Npp32s;
      pSrcDst : access nppdefs_h.Npp32s;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:2635
   pragma Import (C, nppsDiv_32s_ISfs, "nppsDiv_32s_ISfs");

  --* 
  -- * 32-bit floating point in place signal divide signal,
  -- * then clamp to saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pSrcDst \ref in_place_signal_pointer. signal1 divisor elements to be divided into signal2 dividend elements
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsDiv_32f_I
     (pSrc : access nppdefs_h.Npp32f;
      pSrcDst : access nppdefs_h.Npp32f;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:2646
   pragma Import (C, nppsDiv_32f_I, "nppsDiv_32f_I");

  --* 
  -- * 64-bit floating point in place signal divide signal,
  -- * then clamp to saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pSrcDst \ref in_place_signal_pointer. signal1 divisor elements to be divided into signal2 dividend elements
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsDiv_64f_I
     (pSrc : access nppdefs_h.Npp64f;
      pSrcDst : access nppdefs_h.Npp64f;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:2657
   pragma Import (C, nppsDiv_64f_I, "nppsDiv_64f_I");

  --* 
  -- * 32-bit complex floating point in place signal divide signal,
  -- * then clamp to saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pSrcDst \ref in_place_signal_pointer. signal1 divisor elements to be divided into signal2 dividend elements
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsDiv_32fc_I
     (pSrc : access constant nppdefs_h.Npp32fc;
      pSrcDst : access nppdefs_h.Npp32fc;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:2668
   pragma Import (C, nppsDiv_32fc_I, "nppsDiv_32fc_I");

  --* 
  -- * 64-bit complex floating point in place signal divide signal,
  -- * then clamp to saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pSrcDst \ref in_place_signal_pointer. signal1 divisor elements to be divided into signal2 dividend elements
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsDiv_64fc_I
     (pSrc : access constant nppdefs_h.Npp64fc;
      pSrcDst : access nppdefs_h.Npp64fc;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:2679
   pragma Import (C, nppsDiv_64fc_I, "nppsDiv_64fc_I");

  --* @} signal_div  
  --* 
  -- * @defgroup signal_divround Div_Round
  -- *
  -- * Sample by sample division of the samples of two signals with rounding.
  -- *
  -- * @{
  -- *
  --  

  --* 
  -- * 8-bit unsigned char signal divide signal, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer, signal1 divisor elements to be divided into signal2 dividend elements.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nRndMode various rounding modes.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsDiv_Round_8u_Sfs
     (pSrc1 : access nppdefs_h.Npp8u;
      pSrc2 : access nppdefs_h.Npp8u;
      pDst : access nppdefs_h.Npp8u;
      nLength : int;
      nRndMode : nppdefs_h.NppRoundMode;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:2703
   pragma Import (C, nppsDiv_Round_8u_Sfs, "nppsDiv_Round_8u_Sfs");

  --* 
  -- * 16-bit unsigned short signal divide signal, scale, round, then clamp to saturated value.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer, signal1 divisor elements to be divided into signal2 dividend elements.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nRndMode various rounding modes.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsDiv_Round_16u_Sfs
     (pSrc1 : access nppdefs_h.Npp16u;
      pSrc2 : access nppdefs_h.Npp16u;
      pDst : access nppdefs_h.Npp16u;
      nLength : int;
      nRndMode : nppdefs_h.NppRoundMode;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:2716
   pragma Import (C, nppsDiv_Round_16u_Sfs, "nppsDiv_Round_16u_Sfs");

  --* 
  -- * 16-bit signed short signal divide signal, scale, round, then clamp to saturated value.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer, signal1 divisor elements to be divided into signal2 dividend elements.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nRndMode various rounding modes.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsDiv_Round_16s_Sfs
     (pSrc1 : access nppdefs_h.Npp16s;
      pSrc2 : access nppdefs_h.Npp16s;
      pDst : access nppdefs_h.Npp16s;
      nLength : int;
      nRndMode : nppdefs_h.NppRoundMode;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:2729
   pragma Import (C, nppsDiv_Round_16s_Sfs, "nppsDiv_Round_16s_Sfs");

  --* 
  -- * 8-bit unsigned char in place signal divide signal, with scaling, rounding
  -- * then clamp to saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pSrcDst \ref in_place_signal_pointer. signal1 divisor elements to be divided into signal2 dividend elements
  -- * \param nLength \ref length_specification.
  -- * \param nRndMode various rounding modes.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsDiv_Round_8u_ISfs
     (pSrc : access nppdefs_h.Npp8u;
      pSrcDst : access nppdefs_h.Npp8u;
      nLength : int;
      nRndMode : nppdefs_h.NppRoundMode;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:2742
   pragma Import (C, nppsDiv_Round_8u_ISfs, "nppsDiv_Round_8u_ISfs");

  --* 
  -- * 16-bit unsigned short in place signal divide signal, with scaling, rounding
  -- * then clamp to saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pSrcDst \ref in_place_signal_pointer. signal1 divisor elements to be divided into signal2 dividend elements
  -- * \param nLength \ref length_specification.
  -- * \param nRndMode various rounding modes.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsDiv_Round_16u_ISfs
     (pSrc : access nppdefs_h.Npp16u;
      pSrcDst : access nppdefs_h.Npp16u;
      nLength : int;
      nRndMode : nppdefs_h.NppRoundMode;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:2755
   pragma Import (C, nppsDiv_Round_16u_ISfs, "nppsDiv_Round_16u_ISfs");

  --* 
  -- * 16-bit signed short in place signal divide signal, with scaling, rounding
  -- * then clamp to saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pSrcDst \ref in_place_signal_pointer. signal1 divisor elements to be divided into signal2 dividend elements
  -- * \param nLength \ref length_specification.
  -- * \param nRndMode various rounding modes.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsDiv_Round_16s_ISfs
     (pSrc : access nppdefs_h.Npp16s;
      pSrcDst : access nppdefs_h.Npp16s;
      nLength : int;
      nRndMode : nppdefs_h.NppRoundMode;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:2768
   pragma Import (C, nppsDiv_Round_16s_ISfs, "nppsDiv_Round_16s_ISfs");

  --* @} signal_divround  
  --* 
  -- * @defgroup signal_abs Abs
  -- *
  -- * Absolute value of each sample of a signal.
  -- *
  -- * @{
  -- *
  --  

  --* 
  -- * 16-bit signed short signal absolute value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAbs_16s
     (pSrc : access nppdefs_h.Npp16s;
      pDst : access nppdefs_h.Npp16s;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:2789
   pragma Import (C, nppsAbs_16s, "nppsAbs_16s");

  --* 
  -- * 32-bit signed integer signal absolute value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAbs_32s
     (pSrc : access nppdefs_h.Npp32s;
      pDst : access nppdefs_h.Npp32s;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:2799
   pragma Import (C, nppsAbs_32s, "nppsAbs_32s");

  --* 
  -- * 32-bit floating point signal absolute value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAbs_32f
     (pSrc : access nppdefs_h.Npp32f;
      pDst : access nppdefs_h.Npp32f;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:2809
   pragma Import (C, nppsAbs_32f, "nppsAbs_32f");

  --* 
  -- * 64-bit floating point signal absolute value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAbs_64f
     (pSrc : access nppdefs_h.Npp64f;
      pDst : access nppdefs_h.Npp64f;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:2819
   pragma Import (C, nppsAbs_64f, "nppsAbs_64f");

  --* 
  -- * 16-bit signed short signal absolute value.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAbs_16s_I (pSrcDst : access nppdefs_h.Npp16s; nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:2828
   pragma Import (C, nppsAbs_16s_I, "nppsAbs_16s_I");

  --* 
  -- * 32-bit signed integer signal absolute value.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAbs_32s_I (pSrcDst : access nppdefs_h.Npp32s; nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:2837
   pragma Import (C, nppsAbs_32s_I, "nppsAbs_32s_I");

  --* 
  -- * 32-bit floating point signal absolute value.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAbs_32f_I (pSrcDst : access nppdefs_h.Npp32f; nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:2846
   pragma Import (C, nppsAbs_32f_I, "nppsAbs_32f_I");

  --* 
  -- * 64-bit floating point signal absolute value.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAbs_64f_I (pSrcDst : access nppdefs_h.Npp64f; nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:2855
   pragma Import (C, nppsAbs_64f_I, "nppsAbs_64f_I");

  --* @} signal_abs  
  --* 
  -- * @defgroup signal_square Sqr
  -- *
  -- * Squares each sample of a signal.
  -- *
  -- * @{
  -- *
  --  

  --* 
  -- * 32-bit floating point signal squared.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSqr_32f
     (pSrc : access nppdefs_h.Npp32f;
      pDst : access nppdefs_h.Npp32f;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:2876
   pragma Import (C, nppsSqr_32f, "nppsSqr_32f");

  --* 
  -- * 64-bit floating point signal squared.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSqr_64f
     (pSrc : access nppdefs_h.Npp64f;
      pDst : access nppdefs_h.Npp64f;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:2886
   pragma Import (C, nppsSqr_64f, "nppsSqr_64f");

  --* 
  -- * 32-bit complex floating point signal squared.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSqr_32fc
     (pSrc : access constant nppdefs_h.Npp32fc;
      pDst : access nppdefs_h.Npp32fc;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:2896
   pragma Import (C, nppsSqr_32fc, "nppsSqr_32fc");

  --* 
  -- * 64-bit complex floating point signal squared.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSqr_64fc
     (pSrc : access constant nppdefs_h.Npp64fc;
      pDst : access nppdefs_h.Npp64fc;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:2906
   pragma Import (C, nppsSqr_64fc, "nppsSqr_64fc");

  --* 
  -- * 32-bit floating point signal squared.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSqr_32f_I (pSrcDst : access nppdefs_h.Npp32f; nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:2915
   pragma Import (C, nppsSqr_32f_I, "nppsSqr_32f_I");

  --* 
  -- * 64-bit floating point signal squared.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSqr_64f_I (pSrcDst : access nppdefs_h.Npp64f; nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:2924
   pragma Import (C, nppsSqr_64f_I, "nppsSqr_64f_I");

  --* 
  -- * 32-bit complex floating point signal squared.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSqr_32fc_I (pSrcDst : access nppdefs_h.Npp32fc; nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:2933
   pragma Import (C, nppsSqr_32fc_I, "nppsSqr_32fc_I");

  --* 
  -- * 64-bit complex floating point signal squared.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSqr_64fc_I (pSrcDst : access nppdefs_h.Npp64fc; nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:2942
   pragma Import (C, nppsSqr_64fc_I, "nppsSqr_64fc_I");

  --* 
  -- * 8-bit unsigned char signal squared, scale, then clamp to saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSqr_8u_Sfs
     (pSrc : access nppdefs_h.Npp8u;
      pDst : access nppdefs_h.Npp8u;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:2953
   pragma Import (C, nppsSqr_8u_Sfs, "nppsSqr_8u_Sfs");

  --* 
  -- * 16-bit unsigned short signal squared, scale, then clamp to saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSqr_16u_Sfs
     (pSrc : access nppdefs_h.Npp16u;
      pDst : access nppdefs_h.Npp16u;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:2964
   pragma Import (C, nppsSqr_16u_Sfs, "nppsSqr_16u_Sfs");

  --* 
  -- * 16-bit signed short signal squared, scale, then clamp to saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSqr_16s_Sfs
     (pSrc : access nppdefs_h.Npp16s;
      pDst : access nppdefs_h.Npp16s;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:2975
   pragma Import (C, nppsSqr_16s_Sfs, "nppsSqr_16s_Sfs");

  --* 
  -- * 16-bit complex signed short signal squared, scale, then clamp to saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSqr_16sc_Sfs
     (pSrc : access constant nppdefs_h.Npp16sc;
      pDst : access nppdefs_h.Npp16sc;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:2986
   pragma Import (C, nppsSqr_16sc_Sfs, "nppsSqr_16sc_Sfs");

  --* 
  -- * 8-bit unsigned char signal squared, scale, then clamp to saturated value.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSqr_8u_ISfs
     (pSrcDst : access nppdefs_h.Npp8u;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:2996
   pragma Import (C, nppsSqr_8u_ISfs, "nppsSqr_8u_ISfs");

  --* 
  -- * 16-bit unsigned short signal squared, scale, then clamp to saturated value.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSqr_16u_ISfs
     (pSrcDst : access nppdefs_h.Npp16u;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:3006
   pragma Import (C, nppsSqr_16u_ISfs, "nppsSqr_16u_ISfs");

  --* 
  -- * 16-bit signed short signal squared, scale, then clamp to saturated value.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSqr_16s_ISfs
     (pSrcDst : access nppdefs_h.Npp16s;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:3016
   pragma Import (C, nppsSqr_16s_ISfs, "nppsSqr_16s_ISfs");

  --* 
  -- * 16-bit complex signed short signal squared, scale, then clamp to saturated value.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSqr_16sc_ISfs
     (pSrcDst : access nppdefs_h.Npp16sc;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:3026
   pragma Import (C, nppsSqr_16sc_ISfs, "nppsSqr_16sc_ISfs");

  --* @} signal_square  
  --* 
  -- * @defgroup signal_sqrt Sqrt
  -- *
  -- * Square root of each sample of a signal.
  -- *
  -- * @{
  -- *
  --  

  --* 
  -- * 32-bit floating point signal square root.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSqrt_32f
     (pSrc : access nppdefs_h.Npp32f;
      pDst : access nppdefs_h.Npp32f;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:3047
   pragma Import (C, nppsSqrt_32f, "nppsSqrt_32f");

  --* 
  -- * 64-bit floating point signal square root.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSqrt_64f
     (pSrc : access nppdefs_h.Npp64f;
      pDst : access nppdefs_h.Npp64f;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:3057
   pragma Import (C, nppsSqrt_64f, "nppsSqrt_64f");

  --* 
  -- * 32-bit complex floating point signal square root.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSqrt_32fc
     (pSrc : access constant nppdefs_h.Npp32fc;
      pDst : access nppdefs_h.Npp32fc;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:3067
   pragma Import (C, nppsSqrt_32fc, "nppsSqrt_32fc");

  --* 
  -- * 64-bit complex floating point signal square root.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSqrt_64fc
     (pSrc : access constant nppdefs_h.Npp64fc;
      pDst : access nppdefs_h.Npp64fc;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:3077
   pragma Import (C, nppsSqrt_64fc, "nppsSqrt_64fc");

  --* 
  -- * 32-bit floating point signal square root.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSqrt_32f_I (pSrcDst : access nppdefs_h.Npp32f; nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:3086
   pragma Import (C, nppsSqrt_32f_I, "nppsSqrt_32f_I");

  --* 
  -- * 64-bit floating point signal square root.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSqrt_64f_I (pSrcDst : access nppdefs_h.Npp64f; nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:3095
   pragma Import (C, nppsSqrt_64f_I, "nppsSqrt_64f_I");

  --* 
  -- * 32-bit complex floating point signal square root.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSqrt_32fc_I (pSrcDst : access nppdefs_h.Npp32fc; nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:3104
   pragma Import (C, nppsSqrt_32fc_I, "nppsSqrt_32fc_I");

  --* 
  -- * 64-bit complex floating point signal square root.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSqrt_64fc_I (pSrcDst : access nppdefs_h.Npp64fc; nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:3113
   pragma Import (C, nppsSqrt_64fc_I, "nppsSqrt_64fc_I");

  --* 
  -- * 8-bit unsigned char signal square root, scale, then clamp to saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSqrt_8u_Sfs
     (pSrc : access nppdefs_h.Npp8u;
      pDst : access nppdefs_h.Npp8u;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:3124
   pragma Import (C, nppsSqrt_8u_Sfs, "nppsSqrt_8u_Sfs");

  --* 
  -- * 16-bit unsigned short signal square root, scale, then clamp to saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSqrt_16u_Sfs
     (pSrc : access nppdefs_h.Npp16u;
      pDst : access nppdefs_h.Npp16u;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:3135
   pragma Import (C, nppsSqrt_16u_Sfs, "nppsSqrt_16u_Sfs");

  --* 
  -- * 16-bit signed short signal square root, scale, then clamp to saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSqrt_16s_Sfs
     (pSrc : access nppdefs_h.Npp16s;
      pDst : access nppdefs_h.Npp16s;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:3146
   pragma Import (C, nppsSqrt_16s_Sfs, "nppsSqrt_16s_Sfs");

  --* 
  -- * 16-bit complex signed short signal square root, scale, then clamp to saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSqrt_16sc_Sfs
     (pSrc : access constant nppdefs_h.Npp16sc;
      pDst : access nppdefs_h.Npp16sc;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:3157
   pragma Import (C, nppsSqrt_16sc_Sfs, "nppsSqrt_16sc_Sfs");

  --* 
  -- * 64-bit signed integer signal square root, scale, then clamp to saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSqrt_64s_Sfs
     (pSrc : access nppdefs_h.Npp64s;
      pDst : access nppdefs_h.Npp64s;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:3168
   pragma Import (C, nppsSqrt_64s_Sfs, "nppsSqrt_64s_Sfs");

  --* 
  -- * 32-bit signed integer signal square root, scale, then clamp to 16-bit signed integer saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSqrt_32s16s_Sfs
     (pSrc : access nppdefs_h.Npp32s;
      pDst : access nppdefs_h.Npp16s;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:3179
   pragma Import (C, nppsSqrt_32s16s_Sfs, "nppsSqrt_32s16s_Sfs");

  --* 
  -- * 64-bit signed integer signal square root, scale, then clamp to 16-bit signed integer saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSqrt_64s16s_Sfs
     (pSrc : access nppdefs_h.Npp64s;
      pDst : access nppdefs_h.Npp16s;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:3190
   pragma Import (C, nppsSqrt_64s16s_Sfs, "nppsSqrt_64s16s_Sfs");

  --* 
  -- * 8-bit unsigned char signal square root, scale, then clamp to saturated value.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSqrt_8u_ISfs
     (pSrcDst : access nppdefs_h.Npp8u;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:3200
   pragma Import (C, nppsSqrt_8u_ISfs, "nppsSqrt_8u_ISfs");

  --* 
  -- * 16-bit unsigned short signal square root, scale, then clamp to saturated value.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSqrt_16u_ISfs
     (pSrcDst : access nppdefs_h.Npp16u;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:3210
   pragma Import (C, nppsSqrt_16u_ISfs, "nppsSqrt_16u_ISfs");

  --* 
  -- * 16-bit signed short signal square root, scale, then clamp to saturated value.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSqrt_16s_ISfs
     (pSrcDst : access nppdefs_h.Npp16s;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:3220
   pragma Import (C, nppsSqrt_16s_ISfs, "nppsSqrt_16s_ISfs");

  --* 
  -- * 16-bit complex signed short signal square root, scale, then clamp to saturated value.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSqrt_16sc_ISfs
     (pSrcDst : access nppdefs_h.Npp16sc;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:3230
   pragma Import (C, nppsSqrt_16sc_ISfs, "nppsSqrt_16sc_ISfs");

  --* 
  -- * 64-bit signed integer signal square root, scale, then clamp to saturated value.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSqrt_64s_ISfs
     (pSrcDst : access nppdefs_h.Npp64s;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:3240
   pragma Import (C, nppsSqrt_64s_ISfs, "nppsSqrt_64s_ISfs");

  --* @} signal_sqrt  
  --* 
  -- * @defgroup signal_cuberoot Cubrt
  -- *
  -- * Cube root of each sample of a signal.
  -- *
  -- * @{
  -- *
  --  

  --* 
  -- * 32-bit floating point signal cube root.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsCubrt_32f
     (pSrc : access nppdefs_h.Npp32f;
      pDst : access nppdefs_h.Npp32f;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:3261
   pragma Import (C, nppsCubrt_32f, "nppsCubrt_32f");

  --* 
  -- * 32-bit signed integer signal cube root, scale, then clamp to 16-bit signed integer saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsCubrt_32s16s_Sfs
     (pSrc : access nppdefs_h.Npp32s;
      pDst : access nppdefs_h.Npp16s;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:3272
   pragma Import (C, nppsCubrt_32s16s_Sfs, "nppsCubrt_32s16s_Sfs");

  --* @} signal_cuberoot  
  --* 
  -- * @defgroup signal_exp Exp
  -- *
  -- * E raised to the power of each sample of a signal.
  -- *
  -- * @{
  -- *
  --  

  --* 
  -- * 32-bit floating point signal exponent.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsExp_32f
     (pSrc : access nppdefs_h.Npp32f;
      pDst : access nppdefs_h.Npp32f;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:3293
   pragma Import (C, nppsExp_32f, "nppsExp_32f");

  --* 
  -- * 64-bit floating point signal exponent.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsExp_64f
     (pSrc : access nppdefs_h.Npp64f;
      pDst : access nppdefs_h.Npp64f;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:3303
   pragma Import (C, nppsExp_64f, "nppsExp_64f");

  --* 
  -- * 32-bit floating point signal exponent with 64-bit floating point result.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsExp_32f64f
     (pSrc : access nppdefs_h.Npp32f;
      pDst : access nppdefs_h.Npp64f;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:3313
   pragma Import (C, nppsExp_32f64f, "nppsExp_32f64f");

  --* 
  -- * 32-bit floating point signal exponent.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsExp_32f_I (pSrcDst : access nppdefs_h.Npp32f; nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:3322
   pragma Import (C, nppsExp_32f_I, "nppsExp_32f_I");

  --* 
  -- * 64-bit floating point signal exponent.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsExp_64f_I (pSrcDst : access nppdefs_h.Npp64f; nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:3331
   pragma Import (C, nppsExp_64f_I, "nppsExp_64f_I");

  --* 
  -- * 16-bit signed short signal exponent, scale, then clamp to saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsExp_16s_Sfs
     (pSrc : access nppdefs_h.Npp16s;
      pDst : access nppdefs_h.Npp16s;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:3342
   pragma Import (C, nppsExp_16s_Sfs, "nppsExp_16s_Sfs");

  --* 
  -- * 32-bit signed integer signal exponent, scale, then clamp to saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsExp_32s_Sfs
     (pSrc : access nppdefs_h.Npp32s;
      pDst : access nppdefs_h.Npp32s;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:3353
   pragma Import (C, nppsExp_32s_Sfs, "nppsExp_32s_Sfs");

  --* 
  -- * 64-bit signed integer signal exponent, scale, then clamp to saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsExp_64s_Sfs
     (pSrc : access nppdefs_h.Npp64s;
      pDst : access nppdefs_h.Npp64s;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:3364
   pragma Import (C, nppsExp_64s_Sfs, "nppsExp_64s_Sfs");

  --* 
  -- * 16-bit signed short signal exponent, scale, then clamp to saturated value.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsExp_16s_ISfs
     (pSrcDst : access nppdefs_h.Npp16s;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:3374
   pragma Import (C, nppsExp_16s_ISfs, "nppsExp_16s_ISfs");

  --* 
  -- * 32-bit signed integer signal exponent, scale, then clamp to saturated value.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsExp_32s_ISfs
     (pSrcDst : access nppdefs_h.Npp32s;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:3384
   pragma Import (C, nppsExp_32s_ISfs, "nppsExp_32s_ISfs");

  --* 
  -- * 64-bit signed integer signal exponent, scale, then clamp to saturated value.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsExp_64s_ISfs
     (pSrcDst : access nppdefs_h.Npp64s;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:3394
   pragma Import (C, nppsExp_64s_ISfs, "nppsExp_64s_ISfs");

  --* @} signal_exp  
  --* 
  -- * @defgroup signal_ln Ln
  -- *
  -- * Natural logarithm of each sample of a signal.
  -- *
  -- * @{
  -- *
  --  

  --* 
  -- * 32-bit floating point signal natural logarithm.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsLn_32f
     (pSrc : access nppdefs_h.Npp32f;
      pDst : access nppdefs_h.Npp32f;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:3415
   pragma Import (C, nppsLn_32f, "nppsLn_32f");

  --* 
  -- * 64-bit floating point signal natural logarithm.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsLn_64f
     (pSrc : access nppdefs_h.Npp64f;
      pDst : access nppdefs_h.Npp64f;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:3425
   pragma Import (C, nppsLn_64f, "nppsLn_64f");

  --* 
  -- * 64-bit floating point signal natural logarithm with 32-bit floating point result.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsLn_64f32f
     (pSrc : access nppdefs_h.Npp64f;
      pDst : access nppdefs_h.Npp32f;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:3435
   pragma Import (C, nppsLn_64f32f, "nppsLn_64f32f");

  --* 
  -- * 32-bit floating point signal natural logarithm.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsLn_32f_I (pSrcDst : access nppdefs_h.Npp32f; nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:3444
   pragma Import (C, nppsLn_32f_I, "nppsLn_32f_I");

  --* 
  -- * 64-bit floating point signal natural logarithm.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsLn_64f_I (pSrcDst : access nppdefs_h.Npp64f; nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:3453
   pragma Import (C, nppsLn_64f_I, "nppsLn_64f_I");

  --* 
  -- * 16-bit signed short signal natural logarithm, scale, then clamp to saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsLn_16s_Sfs
     (pSrc : access nppdefs_h.Npp16s;
      pDst : access nppdefs_h.Npp16s;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:3464
   pragma Import (C, nppsLn_16s_Sfs, "nppsLn_16s_Sfs");

  --* 
  -- * 32-bit signed integer signal natural logarithm, scale, then clamp to saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsLn_32s_Sfs
     (pSrc : access nppdefs_h.Npp32s;
      pDst : access nppdefs_h.Npp32s;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:3475
   pragma Import (C, nppsLn_32s_Sfs, "nppsLn_32s_Sfs");

  --* 
  -- * 32-bit signed integer signal natural logarithm, scale, then clamp to 16-bit signed short saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsLn_32s16s_Sfs
     (pSrc : access nppdefs_h.Npp32s;
      pDst : access nppdefs_h.Npp16s;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:3486
   pragma Import (C, nppsLn_32s16s_Sfs, "nppsLn_32s16s_Sfs");

  --* 
  -- * 16-bit signed short signal natural logarithm, scale, then clamp to saturated value.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsLn_16s_ISfs
     (pSrcDst : access nppdefs_h.Npp16s;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:3496
   pragma Import (C, nppsLn_16s_ISfs, "nppsLn_16s_ISfs");

  --* 
  -- * 32-bit signed integer signal natural logarithm, scale, then clamp to saturated value.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsLn_32s_ISfs
     (pSrcDst : access nppdefs_h.Npp32s;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:3506
   pragma Import (C, nppsLn_32s_ISfs, "nppsLn_32s_ISfs");

  --* @} signal_ln  
  --* 
  -- * @defgroup signal_10log10 10Log10
  -- *
  -- * Ten times the decimal logarithm of each sample of a signal.
  -- *
  -- * @{
  -- *
  --  

  --* 
  -- * 32-bit signed integer signal 10 times base 10 logarithm, scale, then clamp to saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function npps10Log10_32s_Sfs
     (pSrc : access nppdefs_h.Npp32s;
      pDst : access nppdefs_h.Npp32s;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:3528
   pragma Import (C, npps10Log10_32s_Sfs, "npps10Log10_32s_Sfs");

  --* 
  -- * 32-bit signed integer signal 10 times base 10 logarithm, scale, then clamp to saturated value.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function npps10Log10_32s_ISfs
     (pSrcDst : access nppdefs_h.Npp32s;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:3538
   pragma Import (C, npps10Log10_32s_ISfs, "npps10Log10_32s_ISfs");

  --* @} signal_10log10  
  --* 
  -- * @defgroup signal_sumln SumLn
  -- *
  -- * Sums up the natural logarithm of each sample of a signal.
  -- *
  -- * @{
  -- *
  --  

  --* 
  -- * Device scratch buffer size (in bytes) for 32f SumLn.
  -- * This primitive provides the correct buffer size for nppsSumLn_32f.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
  -- *        <em>host pointer.</em> \ref general_scratch_buffer.
  -- * \return NPP_SUCCESS
  --  

  -- host pointer  
   function nppsSumLnGetBufferSize_32f (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:3560
   pragma Import (C, nppsSumLnGetBufferSize_32f, "nppsSumLnGetBufferSize_32f");

  --* 
  -- * 32-bit floating point signal sum natural logarithm.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pDst Pointer to the output result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSumLn_32f
     (pSrc : access nppdefs_h.Npp32f;
      nLength : int;
      pDst : access nppdefs_h.Npp32f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:3571
   pragma Import (C, nppsSumLn_32f, "nppsSumLn_32f");

  --* 
  -- * Device scratch buffer size (in bytes) for 64f SumLn.
  -- * This primitive provides the correct buffer size for nppsSumLn_64f.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
  -- *        <em>host pointer.</em> \ref general_scratch_buffer.
  -- * \return NPP_SUCCESS
  --  

  -- host pointer  
   function nppsSumLnGetBufferSize_64f (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:3582
   pragma Import (C, nppsSumLnGetBufferSize_64f, "nppsSumLnGetBufferSize_64f");

  --* 
  -- * 64-bit floating point signal sum natural logarithm.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pDst Pointer to the output result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSumLn_64f
     (pSrc : access nppdefs_h.Npp64f;
      nLength : int;
      pDst : access nppdefs_h.Npp64f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:3593
   pragma Import (C, nppsSumLn_64f, "nppsSumLn_64f");

  --* 
  -- * Device scratch buffer size (in bytes) for 32f64f SumLn.
  -- * This primitive provides the correct buffer size for nppsSumLn_32f64f.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
  -- *        <em>host pointer.</em> \ref general_scratch_buffer.
  -- * \return NPP_SUCCESS
  --  

  -- host pointer  
   function nppsSumLnGetBufferSize_32f64f (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:3604
   pragma Import (C, nppsSumLnGetBufferSize_32f64f, "nppsSumLnGetBufferSize_32f64f");

  --* 
  -- * 32-bit flaoting point input, 64-bit floating point output signal sum natural logarithm.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pDst Pointer to the output result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSumLn_32f64f
     (pSrc : access nppdefs_h.Npp32f;
      nLength : int;
      pDst : access nppdefs_h.Npp64f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:3615
   pragma Import (C, nppsSumLn_32f64f, "nppsSumLn_32f64f");

  --* 
  -- * Device scratch buffer size (in bytes) for 16s32f SumLn.
  -- * This primitive provides the correct buffer size for nppsSumLn_16s32f.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
  -- *        <em>host pointer.</em> \ref general_scratch_buffer.
  -- * \return NPP_SUCCESS
  --  

  -- host pointer  
   function nppsSumLnGetBufferSize_16s32f (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:3626
   pragma Import (C, nppsSumLnGetBufferSize_16s32f, "nppsSumLnGetBufferSize_16s32f");

  --* 
  -- * 16-bit signed short integer input, 32-bit floating point output signal sum natural logarithm.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pDst Pointer to the output result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSumLn_16s32f
     (pSrc : access nppdefs_h.Npp16s;
      nLength : int;
      pDst : access nppdefs_h.Npp32f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:3637
   pragma Import (C, nppsSumLn_16s32f, "nppsSumLn_16s32f");

  --* @} signal_sumln  
  --* 
  -- * @defgroup signal_inversetan Arctan
  -- *
  -- * Inverse tangent of each sample of a signal.
  -- *
  -- * @{
  -- *
  --  

  --* 
  -- * 32-bit floating point signal inverse tangent.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsArctan_32f
     (pSrc : access nppdefs_h.Npp32f;
      pDst : access nppdefs_h.Npp32f;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:3658
   pragma Import (C, nppsArctan_32f, "nppsArctan_32f");

  --* 
  -- * 64-bit floating point signal inverse tangent.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsArctan_64f
     (pSrc : access nppdefs_h.Npp64f;
      pDst : access nppdefs_h.Npp64f;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:3668
   pragma Import (C, nppsArctan_64f, "nppsArctan_64f");

  --* 
  -- * 32-bit floating point signal inverse tangent.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsArctan_32f_I (pSrcDst : access nppdefs_h.Npp32f; nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:3677
   pragma Import (C, nppsArctan_32f_I, "nppsArctan_32f_I");

  --* 
  -- * 64-bit floating point signal inverse tangent.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsArctan_64f_I (pSrcDst : access nppdefs_h.Npp64f; nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:3686
   pragma Import (C, nppsArctan_64f_I, "nppsArctan_64f_I");

  --* @} signal_inversetan  
  --* 
  -- * @defgroup signal_normalize Normalize
  -- *
  -- * Normalize each sample of a real or complex signal using offset and division operations.
  -- *
  -- * @{
  -- *
  --  

  --* 
  -- * 32-bit floating point signal normalize.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param vSub value subtracted from each signal element before division
  -- * \param vDiv divisor of post-subtracted signal element dividend
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsNormalize_32f
     (pSrc : access nppdefs_h.Npp32f;
      pDst : access nppdefs_h.Npp32f;
      nLength : int;
      vSub : nppdefs_h.Npp32f;
      vDiv : nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:3709
   pragma Import (C, nppsNormalize_32f, "nppsNormalize_32f");

  --* 
  -- * 32-bit complex floating point signal normalize.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param vSub value subtracted from each signal element before division
  -- * \param vDiv divisor of post-subtracted signal element dividend
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsNormalize_32fc
     (pSrc : access constant nppdefs_h.Npp32fc;
      pDst : access nppdefs_h.Npp32fc;
      nLength : int;
      vSub : nppdefs_h.Npp32fc;
      vDiv : nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:3721
   pragma Import (C, nppsNormalize_32fc, "nppsNormalize_32fc");

  --* 
  -- * 64-bit floating point signal normalize.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param vSub value subtracted from each signal element before division
  -- * \param vDiv divisor of post-subtracted signal element dividend
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsNormalize_64f
     (pSrc : access nppdefs_h.Npp64f;
      pDst : access nppdefs_h.Npp64f;
      nLength : int;
      vSub : nppdefs_h.Npp64f;
      vDiv : nppdefs_h.Npp64f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:3733
   pragma Import (C, nppsNormalize_64f, "nppsNormalize_64f");

  --* 
  -- * 64-bit complex floating point signal normalize.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param vSub value subtracted from each signal element before division
  -- * \param vDiv divisor of post-subtracted signal element dividend
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsNormalize_64fc
     (pSrc : access constant nppdefs_h.Npp64fc;
      pDst : access nppdefs_h.Npp64fc;
      nLength : int;
      vSub : nppdefs_h.Npp64fc;
      vDiv : nppdefs_h.Npp64f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:3745
   pragma Import (C, nppsNormalize_64fc, "nppsNormalize_64fc");

  --* 
  -- * 16-bit signed short signal normalize, scale, then clamp to saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param vSub value subtracted from each signal element before division
  -- * \param vDiv divisor of post-subtracted signal element dividend
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsNormalize_16s_Sfs
     (pSrc : access nppdefs_h.Npp16s;
      pDst : access nppdefs_h.Npp16s;
      nLength : int;
      vSub : nppdefs_h.Npp16s;
      vDiv : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:3758
   pragma Import (C, nppsNormalize_16s_Sfs, "nppsNormalize_16s_Sfs");

  --* 
  -- * 16-bit complex signed short signal normalize, scale, then clamp to saturated value.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param vSub value subtracted from each signal element before division
  -- * \param vDiv divisor of post-subtracted signal element dividend
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsNormalize_16sc_Sfs
     (pSrc : access constant nppdefs_h.Npp16sc;
      pDst : access nppdefs_h.Npp16sc;
      nLength : int;
      vSub : nppdefs_h.Npp16sc;
      vDiv : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:3771
   pragma Import (C, nppsNormalize_16sc_Sfs, "nppsNormalize_16sc_Sfs");

  --* @} signal_normalize  
  --* 
  -- * @defgroup signal_cauchy Cauchy, CauchyD, and CauchyDD2
  -- *
  -- * Determine Cauchy robust error function and its first and second derivatives for each sample of a signal.
  -- *
  -- * @{
  -- *
  --  

  --* 
  -- * 32-bit floating point signal Cauchy error calculation.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nParam constant used in Cauchy formula
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsCauchy_32f_I
     (pSrcDst : access nppdefs_h.Npp32f;
      nLength : int;
      nParam : nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:3792
   pragma Import (C, nppsCauchy_32f_I, "nppsCauchy_32f_I");

  --* 
  -- * 32-bit floating point signal Cauchy first derivative.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nParam constant used in Cauchy formula
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsCauchyD_32f_I
     (pSrcDst : access nppdefs_h.Npp32f;
      nLength : int;
      nParam : nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:3802
   pragma Import (C, nppsCauchyD_32f_I, "nppsCauchyD_32f_I");

  --* 
  -- * 32-bit floating point signal Cauchy first and second derivatives.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param pD2FVal \ref source_signal_pointer. This signal contains the second derivative
  -- *      of the source signal.
  -- * \param nLength \ref length_specification.
  -- * \param nParam constant used in Cauchy formula
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsCauchyDD2_32f_I
     (pSrcDst : access nppdefs_h.Npp32f;
      pD2FVal : access nppdefs_h.Npp32f;
      nLength : int;
      nParam : nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:3814
   pragma Import (C, nppsCauchyDD2_32f_I, "nppsCauchyDD2_32f_I");

  --* @} signal_cauchy  
  --* @} signal_arithmetic_operations  
  --* 
  -- * @defgroup signal_logical_and_shift_operations Logical And Shift Operations
  -- *
  -- * @{
  -- *
  --  

  --* 
  -- * @defgroup signal_andc AndC
  -- *
  -- * Bitwise AND of a constant and each sample of a signal.
  -- *
  -- * @{
  -- *
  --  

  --* 
  -- * 8-bit unsigned char signal and with constant.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nValue Constant value to be anded with each vector element
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAndC_8u
     (pSrc : access nppdefs_h.Npp8u;
      nValue : nppdefs_h.Npp8u;
      pDst : access nppdefs_h.Npp8u;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:3845
   pragma Import (C, nppsAndC_8u, "nppsAndC_8u");

  --* 
  -- * 16-bit unsigned short signal and with constant.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nValue Constant value to be anded with each vector element
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAndC_16u
     (pSrc : access nppdefs_h.Npp16u;
      nValue : nppdefs_h.Npp16u;
      pDst : access nppdefs_h.Npp16u;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:3856
   pragma Import (C, nppsAndC_16u, "nppsAndC_16u");

  --* 
  -- * 32-bit unsigned integer signal and with constant.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nValue Constant value to be anded with each vector element
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAndC_32u
     (pSrc : access nppdefs_h.Npp32u;
      nValue : nppdefs_h.Npp32u;
      pDst : access nppdefs_h.Npp32u;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:3867
   pragma Import (C, nppsAndC_32u, "nppsAndC_32u");

  --* 
  -- * 8-bit unsigned char in place signal and with constant.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nValue Constant value to be anded with each vector element
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAndC_8u_I
     (nValue : nppdefs_h.Npp8u;
      pSrcDst : access nppdefs_h.Npp8u;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:3877
   pragma Import (C, nppsAndC_8u_I, "nppsAndC_8u_I");

  --* 
  -- * 16-bit unsigned short in place signal and with constant.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nValue Constant value to be anded with each vector element
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAndC_16u_I
     (nValue : nppdefs_h.Npp16u;
      pSrcDst : access nppdefs_h.Npp16u;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:3887
   pragma Import (C, nppsAndC_16u_I, "nppsAndC_16u_I");

  --* 
  -- * 32-bit unsigned signed integer in place signal and with constant.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nValue Constant value to be anded with each vector element
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAndC_32u_I
     (nValue : nppdefs_h.Npp32u;
      pSrcDst : access nppdefs_h.Npp32u;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:3897
   pragma Import (C, nppsAndC_32u_I, "nppsAndC_32u_I");

  --* @} signal_andc  
  --* 
  -- * @defgroup signal_and And
  -- *
  -- * Sample by sample bitwise AND of samples from two signals.
  -- *
  -- * @{
  -- *
  --  

  --* 
  -- * 8-bit unsigned char signal and with signal.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer. signal2 elements to be anded with signal1 elements
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAnd_8u
     (pSrc1 : access nppdefs_h.Npp8u;
      pSrc2 : access nppdefs_h.Npp8u;
      pDst : access nppdefs_h.Npp8u;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:3919
   pragma Import (C, nppsAnd_8u, "nppsAnd_8u");

  --* 
  -- * 16-bit unsigned short signal and with signal.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer. signal2 elements to be anded with signal1 elements
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAnd_16u
     (pSrc1 : access nppdefs_h.Npp16u;
      pSrc2 : access nppdefs_h.Npp16u;
      pDst : access nppdefs_h.Npp16u;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:3930
   pragma Import (C, nppsAnd_16u, "nppsAnd_16u");

  --* 
  -- * 32-bit unsigned integer signal and with signal.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer. signal2 elements to be anded with signal1 elements
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAnd_32u
     (pSrc1 : access nppdefs_h.Npp32u;
      pSrc2 : access nppdefs_h.Npp32u;
      pDst : access nppdefs_h.Npp32u;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:3941
   pragma Import (C, nppsAnd_32u, "nppsAnd_32u");

  --* 
  -- * 8-bit unsigned char in place signal and with signal.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be anded with signal1 elements
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAnd_8u_I
     (pSrc : access nppdefs_h.Npp8u;
      pSrcDst : access nppdefs_h.Npp8u;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:3951
   pragma Import (C, nppsAnd_8u_I, "nppsAnd_8u_I");

  --* 
  -- * 16-bit unsigned short in place signal and with signal.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be anded with signal1 elements
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAnd_16u_I
     (pSrc : access nppdefs_h.Npp16u;
      pSrcDst : access nppdefs_h.Npp16u;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:3961
   pragma Import (C, nppsAnd_16u_I, "nppsAnd_16u_I");

  --* 
  -- * 32-bit unsigned integer in place signal and with signal.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be anded with signal1 elements
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAnd_32u_I
     (pSrc : access nppdefs_h.Npp32u;
      pSrcDst : access nppdefs_h.Npp32u;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:3971
   pragma Import (C, nppsAnd_32u_I, "nppsAnd_32u_I");

  --* @} signal_and  
  --* 
  -- * @defgroup signal_orc OrC
  -- *
  -- * Bitwise OR of a constant and each sample of a signal.
  -- *
  -- * @{
  -- *
  --  

  --* 
  -- * 8-bit unsigned char signal or with constant.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nValue Constant value to be ored with each vector element
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsOrC_8u
     (pSrc : access nppdefs_h.Npp8u;
      nValue : nppdefs_h.Npp8u;
      pDst : access nppdefs_h.Npp8u;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:3993
   pragma Import (C, nppsOrC_8u, "nppsOrC_8u");

  --* 
  -- * 16-bit unsigned short signal or with constant.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nValue Constant value to be ored with each vector element
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsOrC_16u
     (pSrc : access nppdefs_h.Npp16u;
      nValue : nppdefs_h.Npp16u;
      pDst : access nppdefs_h.Npp16u;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:4004
   pragma Import (C, nppsOrC_16u, "nppsOrC_16u");

  --* 
  -- * 32-bit unsigned integer signal or with constant.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nValue Constant value to be ored with each vector element
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsOrC_32u
     (pSrc : access nppdefs_h.Npp32u;
      nValue : nppdefs_h.Npp32u;
      pDst : access nppdefs_h.Npp32u;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:4015
   pragma Import (C, nppsOrC_32u, "nppsOrC_32u");

  --* 
  -- * 8-bit unsigned char in place signal or with constant.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nValue Constant value to be ored with each vector element
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsOrC_8u_I
     (nValue : nppdefs_h.Npp8u;
      pSrcDst : access nppdefs_h.Npp8u;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:4025
   pragma Import (C, nppsOrC_8u_I, "nppsOrC_8u_I");

  --* 
  -- * 16-bit unsigned short in place signal or with constant.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nValue Constant value to be ored with each vector element
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsOrC_16u_I
     (nValue : nppdefs_h.Npp16u;
      pSrcDst : access nppdefs_h.Npp16u;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:4035
   pragma Import (C, nppsOrC_16u_I, "nppsOrC_16u_I");

  --* 
  -- * 32-bit unsigned signed integer in place signal or with constant.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nValue Constant value to be ored with each vector element
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsOrC_32u_I
     (nValue : nppdefs_h.Npp32u;
      pSrcDst : access nppdefs_h.Npp32u;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:4045
   pragma Import (C, nppsOrC_32u_I, "nppsOrC_32u_I");

  --* @} signal_orc  
  --* 
  -- * @defgroup signal_or Or
  -- *
  -- * Sample by sample bitwise OR of the samples from two signals.
  -- *
  -- * @{
  -- *
  --  

  --* 
  -- * 8-bit unsigned char signal or with signal.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer. signal2 elements to be ored with signal1 elements
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsOr_8u
     (pSrc1 : access nppdefs_h.Npp8u;
      pSrc2 : access nppdefs_h.Npp8u;
      pDst : access nppdefs_h.Npp8u;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:4067
   pragma Import (C, nppsOr_8u, "nppsOr_8u");

  --* 
  -- * 16-bit unsigned short signal or with signal.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer. signal2 elements to be ored with signal1 elements
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsOr_16u
     (pSrc1 : access nppdefs_h.Npp16u;
      pSrc2 : access nppdefs_h.Npp16u;
      pDst : access nppdefs_h.Npp16u;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:4078
   pragma Import (C, nppsOr_16u, "nppsOr_16u");

  --* 
  -- * 32-bit unsigned integer signal or with signal.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer. signal2 elements to be ored with signal1 elements
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsOr_32u
     (pSrc1 : access nppdefs_h.Npp32u;
      pSrc2 : access nppdefs_h.Npp32u;
      pDst : access nppdefs_h.Npp32u;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:4089
   pragma Import (C, nppsOr_32u, "nppsOr_32u");

  --* 
  -- * 8-bit unsigned char in place signal or with signal.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be ored with signal1 elements
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsOr_8u_I
     (pSrc : access nppdefs_h.Npp8u;
      pSrcDst : access nppdefs_h.Npp8u;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:4099
   pragma Import (C, nppsOr_8u_I, "nppsOr_8u_I");

  --* 
  -- * 16-bit unsigned short in place signal or with signal.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be ored with signal1 elements
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsOr_16u_I
     (pSrc : access nppdefs_h.Npp16u;
      pSrcDst : access nppdefs_h.Npp16u;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:4109
   pragma Import (C, nppsOr_16u_I, "nppsOr_16u_I");

  --* 
  -- * 32-bit unsigned integer in place signal or with signal.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be ored with signal1 elements
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsOr_32u_I
     (pSrc : access nppdefs_h.Npp32u;
      pSrcDst : access nppdefs_h.Npp32u;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:4119
   pragma Import (C, nppsOr_32u_I, "nppsOr_32u_I");

  --* @} signal_or  
  --* 
  -- * @defgroup signal_xorc XorC
  -- *
  -- * Bitwise XOR of a constant and each sample of a signal.
  -- *
  -- * @{
  -- *
  --  

  --* 
  -- * 8-bit unsigned char signal exclusive or with constant.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nValue Constant value to be exclusive ored with each vector element
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsXorC_8u
     (pSrc : access nppdefs_h.Npp8u;
      nValue : nppdefs_h.Npp8u;
      pDst : access nppdefs_h.Npp8u;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:4141
   pragma Import (C, nppsXorC_8u, "nppsXorC_8u");

  --* 
  -- * 16-bit unsigned short signal exclusive or with constant.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nValue Constant value to be exclusive ored with each vector element
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsXorC_16u
     (pSrc : access nppdefs_h.Npp16u;
      nValue : nppdefs_h.Npp16u;
      pDst : access nppdefs_h.Npp16u;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:4152
   pragma Import (C, nppsXorC_16u, "nppsXorC_16u");

  --* 
  -- * 32-bit unsigned integer signal exclusive or with constant.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nValue Constant value to be exclusive ored with each vector element
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsXorC_32u
     (pSrc : access nppdefs_h.Npp32u;
      nValue : nppdefs_h.Npp32u;
      pDst : access nppdefs_h.Npp32u;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:4163
   pragma Import (C, nppsXorC_32u, "nppsXorC_32u");

  --* 
  -- * 8-bit unsigned char in place signal exclusive or with constant.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nValue Constant value to be exclusive ored with each vector element
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsXorC_8u_I
     (nValue : nppdefs_h.Npp8u;
      pSrcDst : access nppdefs_h.Npp8u;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:4173
   pragma Import (C, nppsXorC_8u_I, "nppsXorC_8u_I");

  --* 
  -- * 16-bit unsigned short in place signal exclusive or with constant.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nValue Constant value to be exclusive ored with each vector element
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsXorC_16u_I
     (nValue : nppdefs_h.Npp16u;
      pSrcDst : access nppdefs_h.Npp16u;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:4183
   pragma Import (C, nppsXorC_16u_I, "nppsXorC_16u_I");

  --* 
  -- * 32-bit unsigned signed integer in place signal exclusive or with constant.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nValue Constant value to be exclusive ored with each vector element
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsXorC_32u_I
     (nValue : nppdefs_h.Npp32u;
      pSrcDst : access nppdefs_h.Npp32u;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:4193
   pragma Import (C, nppsXorC_32u_I, "nppsXorC_32u_I");

  --* @} signal_xorc  
  --*
  -- * @defgroup signal_xor Xor
  -- *
  -- * Sample by sample bitwise XOR of the samples from two signals.
  -- *
  -- * @{
  -- *
  --  

  --* 
  -- * 8-bit unsigned char signal exclusive or with signal.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer. signal2 elements to be exclusive ored with signal1 elements
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsXor_8u
     (pSrc1 : access nppdefs_h.Npp8u;
      pSrc2 : access nppdefs_h.Npp8u;
      pDst : access nppdefs_h.Npp8u;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:4215
   pragma Import (C, nppsXor_8u, "nppsXor_8u");

  --* 
  -- * 16-bit unsigned short signal exclusive or with signal.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer. signal2 elements to be exclusive ored with signal1 elements
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsXor_16u
     (pSrc1 : access nppdefs_h.Npp16u;
      pSrc2 : access nppdefs_h.Npp16u;
      pDst : access nppdefs_h.Npp16u;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:4226
   pragma Import (C, nppsXor_16u, "nppsXor_16u");

  --* 
  -- * 32-bit unsigned integer signal exclusive or with signal.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer. signal2 elements to be exclusive ored with signal1 elements
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsXor_32u
     (pSrc1 : access nppdefs_h.Npp32u;
      pSrc2 : access nppdefs_h.Npp32u;
      pDst : access nppdefs_h.Npp32u;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:4237
   pragma Import (C, nppsXor_32u, "nppsXor_32u");

  --* 
  -- * 8-bit unsigned char in place signal exclusive or with signal.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be exclusive ored with signal1 elements
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsXor_8u_I
     (pSrc : access nppdefs_h.Npp8u;
      pSrcDst : access nppdefs_h.Npp8u;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:4247
   pragma Import (C, nppsXor_8u_I, "nppsXor_8u_I");

  --* 
  -- * 16-bit unsigned short in place signal exclusive or with signal.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be exclusive ored with signal1 elements
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsXor_16u_I
     (pSrc : access nppdefs_h.Npp16u;
      pSrcDst : access nppdefs_h.Npp16u;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:4257
   pragma Import (C, nppsXor_16u_I, "nppsXor_16u_I");

  --* 
  -- * 32-bit unsigned integer in place signal exclusive or with signal.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be exclusive ored with signal1 elements
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsXor_32u_I
     (pSrc : access nppdefs_h.Npp32u;
      pSrcDst : access nppdefs_h.Npp32u;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:4267
   pragma Import (C, nppsXor_32u_I, "nppsXor_32u_I");

  --* @} signal_xor  
  --* 
  -- * @defgroup signal_not Not
  -- *
  -- * Bitwise NOT of each sample of a signal.
  -- *
  -- * @{
  -- *
  --  

  --* 
  -- * 8-bit unsigned char not signal.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsNot_8u
     (pSrc : access nppdefs_h.Npp8u;
      pDst : access nppdefs_h.Npp8u;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:4288
   pragma Import (C, nppsNot_8u, "nppsNot_8u");

  --* 
  -- * 16-bit unsigned short not signal.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsNot_16u
     (pSrc : access nppdefs_h.Npp16u;
      pDst : access nppdefs_h.Npp16u;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:4298
   pragma Import (C, nppsNot_16u, "nppsNot_16u");

  --* 
  -- * 32-bit unsigned integer not signal.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsNot_32u
     (pSrc : access nppdefs_h.Npp32u;
      pDst : access nppdefs_h.Npp32u;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:4308
   pragma Import (C, nppsNot_32u, "nppsNot_32u");

  --* 
  -- * 8-bit unsigned char in place not signal.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsNot_8u_I (pSrcDst : access nppdefs_h.Npp8u; nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:4317
   pragma Import (C, nppsNot_8u_I, "nppsNot_8u_I");

  --* 
  -- * 16-bit unsigned short in place not signal.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsNot_16u_I (pSrcDst : access nppdefs_h.Npp16u; nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:4326
   pragma Import (C, nppsNot_16u_I, "nppsNot_16u_I");

  --* 
  -- * 32-bit unsigned signed integer in place not signal.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsNot_32u_I (pSrcDst : access nppdefs_h.Npp32u; nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:4335
   pragma Import (C, nppsNot_32u_I, "nppsNot_32u_I");

  --* @} signal_not  
  --* 
  -- * @defgroup signal_lshiftc LShiftC
  -- *
  -- * Left shifts the bits of each sample of a signal by a constant amount.
  -- *
  -- * @{
  -- *
  --  

  --* 
  -- * 8-bit unsigned char signal left shift with constant.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nValue Constant value to be used to left shift each vector element
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsLShiftC_8u
     (pSrc : access nppdefs_h.Npp8u;
      nValue : int;
      pDst : access nppdefs_h.Npp8u;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:4357
   pragma Import (C, nppsLShiftC_8u, "nppsLShiftC_8u");

  --* 
  -- * 16-bit unsigned short signal left shift with constant.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nValue Constant value to be used to left shift each vector element
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsLShiftC_16u
     (pSrc : access nppdefs_h.Npp16u;
      nValue : int;
      pDst : access nppdefs_h.Npp16u;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:4368
   pragma Import (C, nppsLShiftC_16u, "nppsLShiftC_16u");

  --* 
  -- * 16-bit signed short signal left shift with constant.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nValue Constant value to be used to left shift each vector element
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsLShiftC_16s
     (pSrc : access nppdefs_h.Npp16s;
      nValue : int;
      pDst : access nppdefs_h.Npp16s;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:4379
   pragma Import (C, nppsLShiftC_16s, "nppsLShiftC_16s");

  --* 
  -- * 32-bit unsigned integer signal left shift with constant.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nValue Constant value to be used to left shift each vector element
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsLShiftC_32u
     (pSrc : access nppdefs_h.Npp32u;
      nValue : int;
      pDst : access nppdefs_h.Npp32u;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:4390
   pragma Import (C, nppsLShiftC_32u, "nppsLShiftC_32u");

  --* 
  -- * 32-bit signed integer signal left shift with constant.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nValue Constant value to be used to left shift each vector element
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsLShiftC_32s
     (pSrc : access nppdefs_h.Npp32s;
      nValue : int;
      pDst : access nppdefs_h.Npp32s;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:4401
   pragma Import (C, nppsLShiftC_32s, "nppsLShiftC_32s");

  --* 
  -- * 8-bit unsigned char in place signal left shift with constant.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nValue Constant value to be used to left shift each vector element
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsLShiftC_8u_I
     (nValue : int;
      pSrcDst : access nppdefs_h.Npp8u;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:4411
   pragma Import (C, nppsLShiftC_8u_I, "nppsLShiftC_8u_I");

  --* 
  -- * 16-bit unsigned short in place signal left shift with constant.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nValue Constant value to be used to left shift each vector element
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsLShiftC_16u_I
     (nValue : int;
      pSrcDst : access nppdefs_h.Npp16u;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:4421
   pragma Import (C, nppsLShiftC_16u_I, "nppsLShiftC_16u_I");

  --* 
  -- * 16-bit signed short in place signal left shift with constant.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nValue Constant value to be used to left shift each vector element
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsLShiftC_16s_I
     (nValue : int;
      pSrcDst : access nppdefs_h.Npp16s;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:4431
   pragma Import (C, nppsLShiftC_16s_I, "nppsLShiftC_16s_I");

  --* 
  -- * 32-bit unsigned signed integer in place signal left shift with constant.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nValue Constant value to be used to left shift each vector element
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsLShiftC_32u_I
     (nValue : int;
      pSrcDst : access nppdefs_h.Npp32u;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:4441
   pragma Import (C, nppsLShiftC_32u_I, "nppsLShiftC_32u_I");

  --* 
  -- * 32-bit signed signed integer in place signal left shift with constant.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nValue Constant value to be used to left shift each vector element
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsLShiftC_32s_I
     (nValue : int;
      pSrcDst : access nppdefs_h.Npp32s;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:4451
   pragma Import (C, nppsLShiftC_32s_I, "nppsLShiftC_32s_I");

  --* @} signal_lshiftc  
  --* 
  -- * @defgroup signal_rshiftc RShiftC
  -- *
  -- * Right shifts the bits of each sample of a signal by a constant amount.
  -- *
  -- * @{
  -- *
  --  

  --* 
  -- * 8-bit unsigned char signal right shift with constant.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nValue Constant value to be used to right shift each vector element
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsRShiftC_8u
     (pSrc : access nppdefs_h.Npp8u;
      nValue : int;
      pDst : access nppdefs_h.Npp8u;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:4473
   pragma Import (C, nppsRShiftC_8u, "nppsRShiftC_8u");

  --* 
  -- * 16-bit unsigned short signal right shift with constant.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nValue Constant value to be used to right shift each vector element
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsRShiftC_16u
     (pSrc : access nppdefs_h.Npp16u;
      nValue : int;
      pDst : access nppdefs_h.Npp16u;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:4484
   pragma Import (C, nppsRShiftC_16u, "nppsRShiftC_16u");

  --* 
  -- * 16-bit signed short signal right shift with constant.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nValue Constant value to be used to right shift each vector element
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsRShiftC_16s
     (pSrc : access nppdefs_h.Npp16s;
      nValue : int;
      pDst : access nppdefs_h.Npp16s;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:4495
   pragma Import (C, nppsRShiftC_16s, "nppsRShiftC_16s");

  --* 
  -- * 32-bit unsigned integer signal right shift with constant.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nValue Constant value to be used to right shift each vector element
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsRShiftC_32u
     (pSrc : access nppdefs_h.Npp32u;
      nValue : int;
      pDst : access nppdefs_h.Npp32u;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:4506
   pragma Import (C, nppsRShiftC_32u, "nppsRShiftC_32u");

  --* 
  -- * 32-bit signed integer signal right shift with constant.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nValue Constant value to be used to right shift each vector element
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsRShiftC_32s
     (pSrc : access nppdefs_h.Npp32s;
      nValue : int;
      pDst : access nppdefs_h.Npp32s;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:4517
   pragma Import (C, nppsRShiftC_32s, "nppsRShiftC_32s");

  --* 
  -- * 8-bit unsigned char in place signal right shift with constant.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nValue Constant value to be used to right shift each vector element
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsRShiftC_8u_I
     (nValue : int;
      pSrcDst : access nppdefs_h.Npp8u;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:4527
   pragma Import (C, nppsRShiftC_8u_I, "nppsRShiftC_8u_I");

  --* 
  -- * 16-bit unsigned short in place signal right shift with constant.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nValue Constant value to be used to right shift each vector element
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsRShiftC_16u_I
     (nValue : int;
      pSrcDst : access nppdefs_h.Npp16u;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:4537
   pragma Import (C, nppsRShiftC_16u_I, "nppsRShiftC_16u_I");

  --* 
  -- * 16-bit signed short in place signal right shift with constant.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nValue Constant value to be used to right shift each vector element
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsRShiftC_16s_I
     (nValue : int;
      pSrcDst : access nppdefs_h.Npp16s;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:4547
   pragma Import (C, nppsRShiftC_16s_I, "nppsRShiftC_16s_I");

  --* 
  -- * 32-bit unsigned signed integer in place signal right shift with constant.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nValue Constant value to be used to right shift each vector element
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsRShiftC_32u_I
     (nValue : int;
      pSrcDst : access nppdefs_h.Npp32u;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:4557
   pragma Import (C, nppsRShiftC_32u_I, "nppsRShiftC_32u_I");

  --* 
  -- * 32-bit signed signed integer in place signal right shift with constant.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nValue Constant value to be used to right shift each vector element
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsRShiftC_32s_I
     (nValue : int;
      pSrcDst : access nppdefs_h.Npp32s;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_arithmetic_and_logical_operations.h:4567
   pragma Import (C, nppsRShiftC_32s_I, "nppsRShiftC_32s_I");

  --* @} signal_rshiftc  
  --* @} signal_logical_and_shift_operations  
  --* @} signal_arithmetic_and_logical_operations  
  -- extern "C"  
end npps_arithmetic_and_logical_operations_h;

pragma Ada_2005;
pragma Style_Checks (Off);

with Interfaces.C; use Interfaces.C;
with nppdefs_h;

package npps_statistics_functions_h is

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
  -- * \file npps_statistics_functions.h
  -- * NPP Signal Processing Functionality.
  --  

  --* @defgroup signal_statistical_functions Statistical Functions
  -- *  @ingroup npps
  -- * Functions that provide global signal statistics like: sum, mean, standard
  -- * deviation, min, max, etc.
  -- *
  -- * @{
  -- *
  --  

  --* @defgroup signal_min_every_or_max_every MinEvery And MaxEvery Functions
  -- * Performs the min or max operation on the samples of a signal.
  -- *
  -- * @{  
  -- *
  --  

  --* 
  -- * 8-bit in place min value for each pair of elements.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMinEvery_8u_I
     (pSrc : access nppdefs_h.Npp8u;
      pSrcDst : access nppdefs_h.Npp8u;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:88
   pragma Import (C, nppsMinEvery_8u_I, "nppsMinEvery_8u_I");

  --* 
  -- * 16-bit unsigned short integer in place min value for each pair of elements.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMinEvery_16u_I
     (pSrc : access nppdefs_h.Npp16u;
      pSrcDst : access nppdefs_h.Npp16u;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:98
   pragma Import (C, nppsMinEvery_16u_I, "nppsMinEvery_16u_I");

  --* 
  -- * 16-bit signed short integer in place min value for each pair of elements.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMinEvery_16s_I
     (pSrc : access nppdefs_h.Npp16s;
      pSrcDst : access nppdefs_h.Npp16s;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:108
   pragma Import (C, nppsMinEvery_16s_I, "nppsMinEvery_16s_I");

  --* 
  -- * 32-bit signed integer in place min value for each pair of elements.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMinEvery_32s_I
     (pSrc : access nppdefs_h.Npp32s;
      pSrcDst : access nppdefs_h.Npp32s;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:118
   pragma Import (C, nppsMinEvery_32s_I, "nppsMinEvery_32s_I");

  --* 
  -- * 32-bit floating point in place min value for each pair of elements.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMinEvery_32f_I
     (pSrc : access nppdefs_h.Npp32f;
      pSrcDst : access nppdefs_h.Npp32f;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:128
   pragma Import (C, nppsMinEvery_32f_I, "nppsMinEvery_32f_I");

  --* 
  -- * 64-bit floating point in place min value for each pair of elements.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMinEvery_64f_I
     (pSrc : access nppdefs_h.Npp64f;
      pSrcDst : access nppdefs_h.Npp64f;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:138
   pragma Import (C, nppsMinEvery_64f_I, "nppsMinEvery_64f_I");

  --* 
  -- * 8-bit in place max value for each pair of elements.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMaxEvery_8u_I
     (pSrc : access nppdefs_h.Npp8u;
      pSrcDst : access nppdefs_h.Npp8u;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:148
   pragma Import (C, nppsMaxEvery_8u_I, "nppsMaxEvery_8u_I");

  --* 
  -- * 16-bit unsigned short integer in place max value for each pair of elements.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMaxEvery_16u_I
     (pSrc : access nppdefs_h.Npp16u;
      pSrcDst : access nppdefs_h.Npp16u;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:158
   pragma Import (C, nppsMaxEvery_16u_I, "nppsMaxEvery_16u_I");

  --* 
  -- * 16-bit signed short integer in place max value for each pair of elements.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMaxEvery_16s_I
     (pSrc : access nppdefs_h.Npp16s;
      pSrcDst : access nppdefs_h.Npp16s;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:168
   pragma Import (C, nppsMaxEvery_16s_I, "nppsMaxEvery_16s_I");

  --* 
  -- * 32-bit signed integer in place max value for each pair of elements.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMaxEvery_32s_I
     (pSrc : access nppdefs_h.Npp32s;
      pSrcDst : access nppdefs_h.Npp32s;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:178
   pragma Import (C, nppsMaxEvery_32s_I, "nppsMaxEvery_32s_I");

  --* 
  -- * 32-bit floating point in place max value for each pair of elements.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMaxEvery_32f_I
     (pSrc : access nppdefs_h.Npp32f;
      pSrcDst : access nppdefs_h.Npp32f;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:188
   pragma Import (C, nppsMaxEvery_32f_I, "nppsMaxEvery_32f_I");

  --* 
  -- *
  -- * @} signal_min_every_or_max_every
  -- *
  --  

  --* @defgroup signal_sum Sum
  -- *
  -- * @{  
  -- *
  --  

  --* 
  -- * Device scratch buffer size (in bytes) for nppsSum_32f.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
  -- *        <em>host pointer.</em> \ref general_scratch_buffer.
  -- * \return NPP_SUCCESS
  --  

  -- host pointer  
   function nppsSumGetBufferSize_32f (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:210
   pragma Import (C, nppsSumGetBufferSize_32f, "nppsSumGetBufferSize_32f");

  --* 
  -- * Device scratch buffer size (in bytes) for nppsSum_32fc.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
  -- *        <em>host pointer.</em> \ref general_scratch_buffer.
  -- * \return NPP_SUCCESS
  --  

  -- host pointer  
   function nppsSumGetBufferSize_32fc (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:220
   pragma Import (C, nppsSumGetBufferSize_32fc, "nppsSumGetBufferSize_32fc");

  --* 
  -- * Device scratch buffer size (in bytes) for nppsSum_64f.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
  -- *        <em>host pointer.</em> \ref general_scratch_buffer.
  -- * \return NPP_SUCCESS
  --  

  -- host pointer  
   function nppsSumGetBufferSize_64f (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:230
   pragma Import (C, nppsSumGetBufferSize_64f, "nppsSumGetBufferSize_64f");

  --* 
  -- * Device scratch buffer size (in bytes) for nppsSum_64fc.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
  -- *        <em>host pointer.</em> \ref general_scratch_buffer.
  -- * \return NPP_SUCCESS
  --  

  -- host pointer  
   function nppsSumGetBufferSize_64fc (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:240
   pragma Import (C, nppsSumGetBufferSize_64fc, "nppsSumGetBufferSize_64fc");

  --* 
  -- * Device scratch buffer size (in bytes) for nppsSum_16s_Sfs.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
  -- *        <em>host pointer.</em> \ref general_scratch_buffer.
  -- * \return NPP_SUCCESS
  --  

  -- host pointer  
   function nppsSumGetBufferSize_16s_Sfs (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:250
   pragma Import (C, nppsSumGetBufferSize_16s_Sfs, "nppsSumGetBufferSize_16s_Sfs");

  --* 
  -- * Device scratch buffer size (in bytes) for nppsSum_16sc_Sfs.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
  -- *        <em>host pointer.</em> \ref general_scratch_buffer.
  -- * \return NPP_SUCCESS
  --  

  -- host pointer  
   function nppsSumGetBufferSize_16sc_Sfs (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:260
   pragma Import (C, nppsSumGetBufferSize_16sc_Sfs, "nppsSumGetBufferSize_16sc_Sfs");

  --* 
  -- * Device scratch buffer size (in bytes) for nppsSum_16sc32sc_Sfs.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
  -- *        <em>host pointer.</em> \ref general_scratch_buffer.
  -- * \return NPP_SUCCESS
  --  

  -- host pointer  
   function nppsSumGetBufferSize_16sc32sc_Sfs (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:270
   pragma Import (C, nppsSumGetBufferSize_16sc32sc_Sfs, "nppsSumGetBufferSize_16sc32sc_Sfs");

  --* 
  -- * Device scratch buffer size (in bytes) for nppsSum_32s_Sfs.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
  -- *        <em>host pointer.</em> \ref general_scratch_buffer.
  -- * \return NPP_SUCCESS
  --  

  -- host pointer  
   function nppsSumGetBufferSize_32s_Sfs (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:280
   pragma Import (C, nppsSumGetBufferSize_32s_Sfs, "nppsSumGetBufferSize_32s_Sfs");

  --* 
  -- * Device scratch buffer size (in bytes) for nppsSum_16s32s_Sfs.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
  -- *        <em>host pointer.</em> \ref general_scratch_buffer.
  -- * \return NPP_SUCCESS
  --  

  -- host pointer  
   function nppsSumGetBufferSize_16s32s_Sfs (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:290
   pragma Import (C, nppsSumGetBufferSize_16s32s_Sfs, "nppsSumGetBufferSize_16s32s_Sfs");

  --* 
  -- * 32-bit float vector sum method
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pSum Pointer to the output result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsSumGetBufferSize_32f to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSum_32f
     (pSrc : access nppdefs_h.Npp32f;
      nLength : int;
      pSum : access nppdefs_h.Npp32f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:302
   pragma Import (C, nppsSum_32f, "nppsSum_32f");

  --* 
  -- * 32-bit float complex vector sum method
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pSum Pointer to the output result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsSumGetBufferSize_32fc to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSum_32fc
     (pSrc : access constant nppdefs_h.Npp32fc;
      nLength : int;
      pSum : access nppdefs_h.Npp32fc;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:314
   pragma Import (C, nppsSum_32fc, "nppsSum_32fc");

  --* 
  -- * 64-bit double vector sum method
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pSum Pointer to the output result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsSumGetBufferSize_64f to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSum_64f
     (pSrc : access nppdefs_h.Npp64f;
      nLength : int;
      pSum : access nppdefs_h.Npp64f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:326
   pragma Import (C, nppsSum_64f, "nppsSum_64f");

  --* 
  -- * 64-bit double complex vector sum method
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pSum Pointer to the output result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsSumGetBufferSize_64fc to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSum_64fc
     (pSrc : access constant nppdefs_h.Npp64fc;
      nLength : int;
      pSum : access nppdefs_h.Npp64fc;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:338
   pragma Import (C, nppsSum_64fc, "nppsSum_64fc");

  --* 
  -- * 16-bit short vector sum with integer scaling method
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pSum Pointer to the output result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsSumGetBufferSize_16s_Sfs to determine the minium number of bytes required.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSum_16s_Sfs
     (pSrc : access nppdefs_h.Npp16s;
      nLength : int;
      pSum : access nppdefs_h.Npp16s;
      nScaleFactor : int;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:351
   pragma Import (C, nppsSum_16s_Sfs, "nppsSum_16s_Sfs");

  --* 
  -- * 32-bit integer vector sum with integer scaling method
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pSum Pointer to the output result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsSumGetBufferSize_32s_Sfs to determine the minium number of bytes required.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSum_32s_Sfs
     (pSrc : access nppdefs_h.Npp32s;
      nLength : int;
      pSum : access nppdefs_h.Npp32s;
      nScaleFactor : int;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:365
   pragma Import (C, nppsSum_32s_Sfs, "nppsSum_32s_Sfs");

  --* 
  -- * 16-bit short complex vector sum with integer scaling method
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pSum Pointer to the output result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsSumGetBufferSize_16sc_Sfs to determine the minium number of bytes required.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSum_16sc_Sfs
     (pSrc : access constant nppdefs_h.Npp16sc;
      nLength : int;
      pSum : access nppdefs_h.Npp16sc;
      nScaleFactor : int;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:379
   pragma Import (C, nppsSum_16sc_Sfs, "nppsSum_16sc_Sfs");

  --* 
  -- * 16-bit short complex vector sum (32bit int complex) with integer scaling
  -- * method
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pSum Pointer to the output result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsSumGetBufferSize_16sc32sc_Sfs to determine the minium number of bytes required.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSum_16sc32sc_Sfs
     (pSrc : access constant nppdefs_h.Npp16sc;
      nLength : int;
      pSum : access nppdefs_h.Npp32sc;
      nScaleFactor : int;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:394
   pragma Import (C, nppsSum_16sc32sc_Sfs, "nppsSum_16sc32sc_Sfs");

  --* 
  -- * 16-bit integer vector sum (32bit) with integer scaling method
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pSum Pointer to the output result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsSumGetBufferSize_16s32s_Sfs to determine the minium number of bytes required.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSum_16s32s_Sfs
     (pSrc : access nppdefs_h.Npp16s;
      nLength : int;
      pSum : access nppdefs_h.Npp32s;
      nScaleFactor : int;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:408
   pragma Import (C, nppsSum_16s32s_Sfs, "nppsSum_16s32s_Sfs");

  --* @} signal_sum  
  --* @defgroup signal_max Maximum
  -- *
  -- * @{
  -- *
  --  

  --* 
  -- * Device scratch buffer size (in bytes) for nppsMax_16s.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
  -- *        <em>host pointer.</em> \ref general_scratch_buffer.
  -- * \return NPP_SUCCESS
  --  

  -- host pointer  
   function nppsMaxGetBufferSize_16s (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:428
   pragma Import (C, nppsMaxGetBufferSize_16s, "nppsMaxGetBufferSize_16s");

  --* 
  -- * Device scratch buffer size (in bytes) for nppsMax_32s.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
  -- *        <em>host pointer.</em> \ref general_scratch_buffer.
  -- * \return NPP_SUCCESS
  --  

  -- host pointer  
   function nppsMaxGetBufferSize_32s (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:438
   pragma Import (C, nppsMaxGetBufferSize_32s, "nppsMaxGetBufferSize_32s");

  --* 
  -- * Device scratch buffer size (in bytes) for nppsMax_32f.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
  -- *        <em>host pointer.</em> \ref general_scratch_buffer.
  -- * \return NPP_SUCCESS
  --  

  -- host pointer  
   function nppsMaxGetBufferSize_32f (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:448
   pragma Import (C, nppsMaxGetBufferSize_32f, "nppsMaxGetBufferSize_32f");

  --* 
  -- * Device scratch buffer size (in bytes) for nppsMax_64f.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
  -- *        <em>host pointer.</em> \ref general_scratch_buffer.
  -- * \return NPP_SUCCESS
  --  

  -- host pointer  
   function nppsMaxGetBufferSize_64f (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:458
   pragma Import (C, nppsMaxGetBufferSize_64f, "nppsMaxGetBufferSize_64f");

  --* 
  -- * 16-bit integer vector max method
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pMax Pointer to the output result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsMaxGetBufferSize_16s to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMax_16s
     (pSrc : access nppdefs_h.Npp16s;
      nLength : int;
      pMax : access nppdefs_h.Npp16s;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:470
   pragma Import (C, nppsMax_16s, "nppsMax_16s");

  --* 
  -- * 32-bit integer vector max method
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pMax Pointer to the output result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsMaxGetBufferSize_32s to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMax_32s
     (pSrc : access nppdefs_h.Npp32s;
      nLength : int;
      pMax : access nppdefs_h.Npp32s;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:482
   pragma Import (C, nppsMax_32s, "nppsMax_32s");

  --* 
  -- * 32-bit float vector max method
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pMax Pointer to the output result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsMaxGetBufferSize_32f to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMax_32f
     (pSrc : access nppdefs_h.Npp32f;
      nLength : int;
      pMax : access nppdefs_h.Npp32f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:494
   pragma Import (C, nppsMax_32f, "nppsMax_32f");

  --* 
  -- * 64-bit float vector max method
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pMax Pointer to the output result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsMaxGetBufferSize_64f to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMax_64f
     (pSrc : access nppdefs_h.Npp64f;
      nLength : int;
      pMax : access nppdefs_h.Npp64f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:506
   pragma Import (C, nppsMax_64f, "nppsMax_64f");

  --* 
  -- * Device scratch buffer size (in bytes) for nppsMaxIndx_16s.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
  -- *        <em>host pointer.</em> \ref general_scratch_buffer.
  -- * \return NPP_SUCCESS
  --  

  -- host pointer  
   function nppsMaxIndxGetBufferSize_16s (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:516
   pragma Import (C, nppsMaxIndxGetBufferSize_16s, "nppsMaxIndxGetBufferSize_16s");

  --* 
  -- * Device scratch buffer size (in bytes) for nppsMaxIndx_32s.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
  -- *        <em>host pointer.</em> \ref general_scratch_buffer.
  -- * \return NPP_SUCCESS
  --  

  -- host pointer  
   function nppsMaxIndxGetBufferSize_32s (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:526
   pragma Import (C, nppsMaxIndxGetBufferSize_32s, "nppsMaxIndxGetBufferSize_32s");

  --* 
  -- * Device scratch buffer size (in bytes) for nppsMaxIndx_32f.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
  -- *        <em>host pointer.</em> \ref general_scratch_buffer.
  -- * \return NPP_SUCCESS
  --  

  -- host pointer  
   function nppsMaxIndxGetBufferSize_32f (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:536
   pragma Import (C, nppsMaxIndxGetBufferSize_32f, "nppsMaxIndxGetBufferSize_32f");

  --* 
  -- * Device scratch buffer size (in bytes) for nppsMaxIndx_64f.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
  -- *        <em>host pointer.</em> \ref general_scratch_buffer.
  -- * \return NPP_SUCCESS
  --  

  -- host pointer  
   function nppsMaxIndxGetBufferSize_64f (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:546
   pragma Import (C, nppsMaxIndxGetBufferSize_64f, "nppsMaxIndxGetBufferSize_64f");

  --* 
  -- * 16-bit integer vector max index method
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pMax Pointer to the output result.
  -- * \param pIndx Pointer to the index value of the first maximum element.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsMaxIndxGetBufferSize_16s to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMaxIndx_16s
     (pSrc : access nppdefs_h.Npp16s;
      nLength : int;
      pMax : access nppdefs_h.Npp16s;
      pIndx : access int;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:559
   pragma Import (C, nppsMaxIndx_16s, "nppsMaxIndx_16s");

  --* 
  -- * 32-bit integer vector max index method
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pMax Pointer to the output result.
  -- * \param pIndx Pointer to the index value of the first maximum element.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsMaxIndxGetBufferSize_32s to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMaxIndx_32s
     (pSrc : access nppdefs_h.Npp32s;
      nLength : int;
      pMax : access nppdefs_h.Npp32s;
      pIndx : access int;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:572
   pragma Import (C, nppsMaxIndx_32s, "nppsMaxIndx_32s");

  --* 
  -- * 32-bit float vector max index method
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pMax Pointer to the output result.
  -- * \param pIndx Pointer to the index value of the first maximum element.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsMaxIndxGetBufferSize_32f to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMaxIndx_32f
     (pSrc : access nppdefs_h.Npp32f;
      nLength : int;
      pMax : access nppdefs_h.Npp32f;
      pIndx : access int;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:585
   pragma Import (C, nppsMaxIndx_32f, "nppsMaxIndx_32f");

  --* 
  -- * 64-bit float vector max index method
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pMax Pointer to the output result.
  -- * \param pIndx Pointer to the index value of the first maximum element.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsMaxIndxGetBufferSize_64f to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMaxIndx_64f
     (pSrc : access nppdefs_h.Npp64f;
      nLength : int;
      pMax : access nppdefs_h.Npp64f;
      pIndx : access int;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:598
   pragma Import (C, nppsMaxIndx_64f, "nppsMaxIndx_64f");

  --* 
  -- * Device scratch buffer size (in bytes) for nppsMaxAbs_16s.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
  -- *        <em>host pointer.</em> \ref general_scratch_buffer.
  -- * \return NPP_SUCCESS
  --  

  -- host pointer  
   function nppsMaxAbsGetBufferSize_16s (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:608
   pragma Import (C, nppsMaxAbsGetBufferSize_16s, "nppsMaxAbsGetBufferSize_16s");

  --* 
  -- * Device scratch buffer size (in bytes) for nppsMaxAbs_32s.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
  -- *        <em>host pointer.</em> \ref general_scratch_buffer.
  -- * \return NPP_SUCCESS
  --  

  -- host pointer  
   function nppsMaxAbsGetBufferSize_32s (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:618
   pragma Import (C, nppsMaxAbsGetBufferSize_32s, "nppsMaxAbsGetBufferSize_32s");

  --* 
  -- * 16-bit integer vector max absolute method
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pMaxAbs Pointer to the output result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsMaxAbsGetBufferSize_16s to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMaxAbs_16s
     (pSrc : access nppdefs_h.Npp16s;
      nLength : int;
      pMaxAbs : access nppdefs_h.Npp16s;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:630
   pragma Import (C, nppsMaxAbs_16s, "nppsMaxAbs_16s");

  --* 
  -- * 32-bit integer vector max absolute method
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pMaxAbs Pointer to the output result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsMaxAbsGetBufferSize_32s to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMaxAbs_32s
     (pSrc : access nppdefs_h.Npp32s;
      nLength : int;
      pMaxAbs : access nppdefs_h.Npp32s;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:642
   pragma Import (C, nppsMaxAbs_32s, "nppsMaxAbs_32s");

  --* 
  -- * Device scratch buffer size (in bytes) for nppsMaxAbsIndx_16s.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
  -- *        <em>host pointer.</em> \ref general_scratch_buffer.
  -- * \return NPP_SUCCESS
  --  

  -- host pointer  
   function nppsMaxAbsIndxGetBufferSize_16s (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:652
   pragma Import (C, nppsMaxAbsIndxGetBufferSize_16s, "nppsMaxAbsIndxGetBufferSize_16s");

  --* 
  -- * Device scratch buffer size (in bytes) for nppsMaxAbsIndx_32s.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
  -- *        <em>host pointer.</em> \ref general_scratch_buffer.
  -- * \return NPP_SUCCESS
  --  

  -- host pointer  
   function nppsMaxAbsIndxGetBufferSize_32s (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:662
   pragma Import (C, nppsMaxAbsIndxGetBufferSize_32s, "nppsMaxAbsIndxGetBufferSize_32s");

  --* 
  -- * 16-bit integer vector max absolute index method
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pMaxAbs Pointer to the output result.
  -- * \param pIndx Pointer to the index value of the first maximum element.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsMaxAbsIndxGetBufferSize_16s to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMaxAbsIndx_16s
     (pSrc : access nppdefs_h.Npp16s;
      nLength : int;
      pMaxAbs : access nppdefs_h.Npp16s;
      pIndx : access int;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:675
   pragma Import (C, nppsMaxAbsIndx_16s, "nppsMaxAbsIndx_16s");

  --* 
  -- * 32-bit integer vector max absolute index method
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pMaxAbs Pointer to the output result.
  -- * \param pIndx Pointer to the index value of the first maximum element.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsMaxAbsIndxGetBufferSize_32s to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMaxAbsIndx_32s
     (pSrc : access nppdefs_h.Npp32s;
      nLength : int;
      pMaxAbs : access nppdefs_h.Npp32s;
      pIndx : access int;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:688
   pragma Import (C, nppsMaxAbsIndx_32s, "nppsMaxAbsIndx_32s");

  --* @} signal_max  
  --* @defgroup signal_min Minimum
  -- *
  -- * @{
  -- *
  --  

  --* 
  -- * Device scratch buffer size (in bytes) for nppsMin_16s.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
  -- *        <em>host pointer.</em> \ref general_scratch_buffer.
  -- * \return NPP_SUCCESS
  --  

  -- host pointer  
   function nppsMinGetBufferSize_16s (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:706
   pragma Import (C, nppsMinGetBufferSize_16s, "nppsMinGetBufferSize_16s");

  --* 
  -- * Device scratch buffer size (in bytes) for nppsMin_32s.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
  -- *        <em>host pointer.</em> \ref general_scratch_buffer.
  -- * \return NPP_SUCCESS
  --  

  -- host pointer  
   function nppsMinGetBufferSize_32s (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:716
   pragma Import (C, nppsMinGetBufferSize_32s, "nppsMinGetBufferSize_32s");

  --* 
  -- * Device scratch buffer size (in bytes) for nppsMin_32f.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
  -- *        <em>host pointer.</em> \ref general_scratch_buffer.
  -- * \return NPP_SUCCESS
  --  

  -- host pointer  
   function nppsMinGetBufferSize_32f (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:726
   pragma Import (C, nppsMinGetBufferSize_32f, "nppsMinGetBufferSize_32f");

  --* 
  -- * Device scratch buffer size (in bytes) for nppsMin_64f.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
  -- *        <em>host pointer.</em> \ref general_scratch_buffer.
  -- * \return NPP_SUCCESS
  --  

  -- host pointer  
   function nppsMinGetBufferSize_64f (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:736
   pragma Import (C, nppsMinGetBufferSize_64f, "nppsMinGetBufferSize_64f");

  --* 
  -- * 16-bit integer vector min method
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pMin Pointer to the output result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsMinGetBufferSize_16s to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMin_16s
     (pSrc : access nppdefs_h.Npp16s;
      nLength : int;
      pMin : access nppdefs_h.Npp16s;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:748
   pragma Import (C, nppsMin_16s, "nppsMin_16s");

  --* 
  -- * 32-bit integer vector min method
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pMin Pointer to the output result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsMinGetBufferSize_32s to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMin_32s
     (pSrc : access nppdefs_h.Npp32s;
      nLength : int;
      pMin : access nppdefs_h.Npp32s;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:760
   pragma Import (C, nppsMin_32s, "nppsMin_32s");

  --* 
  -- * 32-bit integer vector min method
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pMin Pointer to the output result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsMinGetBufferSize_32f to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMin_32f
     (pSrc : access nppdefs_h.Npp32f;
      nLength : int;
      pMin : access nppdefs_h.Npp32f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:772
   pragma Import (C, nppsMin_32f, "nppsMin_32f");

  --* 
  -- * 64-bit integer vector min method
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pMin Pointer to the output result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsMinGetBufferSize_64f to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMin_64f
     (pSrc : access nppdefs_h.Npp64f;
      nLength : int;
      pMin : access nppdefs_h.Npp64f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:784
   pragma Import (C, nppsMin_64f, "nppsMin_64f");

  --* 
  -- * Device scratch buffer size (in bytes) for nppsMinIndx_16s.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
  -- *        <em>host pointer.</em> \ref general_scratch_buffer.
  -- * \return NPP_SUCCESS
  --  

  -- host pointer  
   function nppsMinIndxGetBufferSize_16s (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:794
   pragma Import (C, nppsMinIndxGetBufferSize_16s, "nppsMinIndxGetBufferSize_16s");

  --* 
  -- * Device scratch buffer size (in bytes) for nppsMinIndx_32s.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
  -- *        <em>host pointer.</em> \ref general_scratch_buffer.
  -- * \return NPP_SUCCESS
  --  

  -- host pointer  
   function nppsMinIndxGetBufferSize_32s (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:804
   pragma Import (C, nppsMinIndxGetBufferSize_32s, "nppsMinIndxGetBufferSize_32s");

  --* 
  -- * Device scratch buffer size (in bytes) for nppsMinIndx_32f.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
  -- *        <em>host pointer.</em> \ref general_scratch_buffer.
  -- * \return NPP_SUCCESS
  --  

  -- host pointer  
   function nppsMinIndxGetBufferSize_32f (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:814
   pragma Import (C, nppsMinIndxGetBufferSize_32f, "nppsMinIndxGetBufferSize_32f");

  --* 
  -- * Device scratch buffer size (in bytes) for nppsMinIndx_64f.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
  -- *        <em>host pointer.</em> \ref general_scratch_buffer.
  -- * \return NPP_SUCCESS
  --  

  -- host pointer  
   function nppsMinIndxGetBufferSize_64f (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:824
   pragma Import (C, nppsMinIndxGetBufferSize_64f, "nppsMinIndxGetBufferSize_64f");

  --* 
  -- * 16-bit integer vector min index method
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pMin Pointer to the output result.
  -- * \param pIndx Pointer to the index value of the first minimum element.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsMinIndxGetBufferSize_16s to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMinIndx_16s
     (pSrc : access nppdefs_h.Npp16s;
      nLength : int;
      pMin : access nppdefs_h.Npp16s;
      pIndx : access int;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:837
   pragma Import (C, nppsMinIndx_16s, "nppsMinIndx_16s");

  --* 
  -- * 32-bit integer vector min index method
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pMin Pointer to the output result.
  -- * \param pIndx Pointer to the index value of the first minimum element.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsMinIndxGetBufferSize_32s to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMinIndx_32s
     (pSrc : access nppdefs_h.Npp32s;
      nLength : int;
      pMin : access nppdefs_h.Npp32s;
      pIndx : access int;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:850
   pragma Import (C, nppsMinIndx_32s, "nppsMinIndx_32s");

  --* 
  -- * 32-bit float vector min index method
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pMin Pointer to the output result.
  -- * \param pIndx Pointer to the index value of the first minimum element.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsMinIndxGetBufferSize_32f to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMinIndx_32f
     (pSrc : access nppdefs_h.Npp32f;
      nLength : int;
      pMin : access nppdefs_h.Npp32f;
      pIndx : access int;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:863
   pragma Import (C, nppsMinIndx_32f, "nppsMinIndx_32f");

  --* 
  -- * 64-bit float vector min index method
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pMin Pointer to the output result.
  -- * \param pIndx Pointer to the index value of the first minimum element.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsMinIndxGetBufferSize_64f to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMinIndx_64f
     (pSrc : access nppdefs_h.Npp64f;
      nLength : int;
      pMin : access nppdefs_h.Npp64f;
      pIndx : access int;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:876
   pragma Import (C, nppsMinIndx_64f, "nppsMinIndx_64f");

  --* 
  -- * Device scratch buffer size (in bytes) for nppsMinAbs_16s.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
  -- *        <em>host pointer.</em> \ref general_scratch_buffer.
  -- * \return NPP_SUCCESS
  --  

  -- host pointer  
   function nppsMinAbsGetBufferSize_16s (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:886
   pragma Import (C, nppsMinAbsGetBufferSize_16s, "nppsMinAbsGetBufferSize_16s");

  --* 
  -- * Device scratch buffer size (in bytes) for nppsMinAbs_32s.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
  -- *        <em>host pointer.</em> \ref general_scratch_buffer.
  -- * \return NPP_SUCCESS
  --  

  -- host pointer  
   function nppsMinAbsGetBufferSize_32s (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:896
   pragma Import (C, nppsMinAbsGetBufferSize_32s, "nppsMinAbsGetBufferSize_32s");

  --* 
  -- * 16-bit integer vector min absolute method
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pMinAbs Pointer to the output result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsMinAbsGetBufferSize_16s to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMinAbs_16s
     (pSrc : access nppdefs_h.Npp16s;
      nLength : int;
      pMinAbs : access nppdefs_h.Npp16s;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:908
   pragma Import (C, nppsMinAbs_16s, "nppsMinAbs_16s");

  --* 
  -- * 32-bit integer vector min absolute method
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pMinAbs Pointer to the output result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsMinAbsGetBufferSize_16s to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMinAbs_32s
     (pSrc : access nppdefs_h.Npp32s;
      nLength : int;
      pMinAbs : access nppdefs_h.Npp32s;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:920
   pragma Import (C, nppsMinAbs_32s, "nppsMinAbs_32s");

  --* 
  -- * Device scratch buffer size (in bytes) for nppsMinAbsIndx_16s.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
  -- *        <em>host pointer.</em> \ref general_scratch_buffer.
  -- * \return NPP_SUCCESS
  --  

  -- host pointer  
   function nppsMinAbsIndxGetBufferSize_16s (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:930
   pragma Import (C, nppsMinAbsIndxGetBufferSize_16s, "nppsMinAbsIndxGetBufferSize_16s");

  --* 
  -- * Device scratch buffer size (in bytes) for nppsMinAbsIndx_32s.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
  -- *        <em>host pointer.</em> \ref general_scratch_buffer.
  -- * \return NPP_SUCCESS
  --  

  -- host pointer  
   function nppsMinAbsIndxGetBufferSize_32s (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:940
   pragma Import (C, nppsMinAbsIndxGetBufferSize_32s, "nppsMinAbsIndxGetBufferSize_32s");

  --* 
  -- * 16-bit integer vector min absolute index method
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pMinAbs Pointer to the output result.
  -- * \param pIndx Pointer to the index value of the first minimum element.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsMinAbsIndxGetBufferSize_16s to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMinAbsIndx_16s
     (pSrc : access nppdefs_h.Npp16s;
      nLength : int;
      pMinAbs : access nppdefs_h.Npp16s;
      pIndx : access int;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:953
   pragma Import (C, nppsMinAbsIndx_16s, "nppsMinAbsIndx_16s");

  --* 
  -- * 32-bit integer vector min absolute index method
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pMinAbs Pointer to the output result.
  -- * \param pIndx Pointer to the index value of the first minimum element.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsMinAbsIndxGetBufferSize_32s to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMinAbsIndx_32s
     (pSrc : access nppdefs_h.Npp32s;
      nLength : int;
      pMinAbs : access nppdefs_h.Npp32s;
      pIndx : access int;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:966
   pragma Import (C, nppsMinAbsIndx_32s, "nppsMinAbsIndx_32s");

  --* @} signal_min  
  --* @defgroup signal_mean Mean
  -- *
  -- * @{
  -- *
  --  

  --* 
  -- * Device scratch buffer size (in bytes) for nppsMean_32f.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
  -- *        <em>host pointer.</em> \ref general_scratch_buffer.
  -- * \return NPP_SUCCESS
  --  

  -- host pointer  
   function nppsMeanGetBufferSize_32f (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:984
   pragma Import (C, nppsMeanGetBufferSize_32f, "nppsMeanGetBufferSize_32f");

  --* 
  -- * Device scratch buffer size (in bytes) for nppsMean_32fc.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
  -- *        <em>host pointer.</em> \ref general_scratch_buffer.
  -- * \return NPP_SUCCESS
  --  

  -- host pointer  
   function nppsMeanGetBufferSize_32fc (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:994
   pragma Import (C, nppsMeanGetBufferSize_32fc, "nppsMeanGetBufferSize_32fc");

  --* 
  -- * Device scratch buffer size (in bytes) for nppsMean_64f.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
  -- *        <em>host pointer.</em> \ref general_scratch_buffer.
  -- * \return NPP_SUCCESS
  --  

  -- host pointer  
   function nppsMeanGetBufferSize_64f (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:1004
   pragma Import (C, nppsMeanGetBufferSize_64f, "nppsMeanGetBufferSize_64f");

  --* 
  -- * Device scratch buffer size (in bytes) for nppsMean_64fc.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
  -- *        <em>host pointer.</em> \ref general_scratch_buffer.
  -- * \return NPP_SUCCESS
  --  

  -- host pointer  
   function nppsMeanGetBufferSize_64fc (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:1014
   pragma Import (C, nppsMeanGetBufferSize_64fc, "nppsMeanGetBufferSize_64fc");

  --* 
  -- * Device scratch buffer size (in bytes) for nppsMean_16s_Sfs.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
  -- *        <em>host pointer.</em> \ref general_scratch_buffer.
  -- * \return NPP_SUCCESS
  --  

  -- host pointer  
   function nppsMeanGetBufferSize_16s_Sfs (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:1024
   pragma Import (C, nppsMeanGetBufferSize_16s_Sfs, "nppsMeanGetBufferSize_16s_Sfs");

  --* 
  -- * Device scratch buffer size (in bytes) for nppsMean_32s_Sfs.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
  -- *        <em>host pointer.</em> \ref general_scratch_buffer.
  -- * \return NPP_SUCCESS
  --  

  -- host pointer  
   function nppsMeanGetBufferSize_32s_Sfs (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:1034
   pragma Import (C, nppsMeanGetBufferSize_32s_Sfs, "nppsMeanGetBufferSize_32s_Sfs");

  --* 
  -- * Device scratch buffer size (in bytes) for nppsMean_16sc_Sfs.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
  -- *        <em>host pointer.</em> \ref general_scratch_buffer.
  -- * \return NPP_SUCCESS
  --  

  -- host pointer  
   function nppsMeanGetBufferSize_16sc_Sfs (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:1044
   pragma Import (C, nppsMeanGetBufferSize_16sc_Sfs, "nppsMeanGetBufferSize_16sc_Sfs");

  --* 
  -- * 32-bit float vector mean method
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pMean Pointer to the output result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsMeanGetBufferSize_32f to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMean_32f
     (pSrc : access nppdefs_h.Npp32f;
      nLength : int;
      pMean : access nppdefs_h.Npp32f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:1056
   pragma Import (C, nppsMean_32f, "nppsMean_32f");

  --* 
  -- * 32-bit float complex vector mean method
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pMean Pointer to the output result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsMeanGetBufferSize_32fc to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMean_32fc
     (pSrc : access constant nppdefs_h.Npp32fc;
      nLength : int;
      pMean : access nppdefs_h.Npp32fc;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:1068
   pragma Import (C, nppsMean_32fc, "nppsMean_32fc");

  --* 
  -- * 64-bit double vector mean method
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pMean Pointer to the output result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsMeanGetBufferSize_64f to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMean_64f
     (pSrc : access nppdefs_h.Npp64f;
      nLength : int;
      pMean : access nppdefs_h.Npp64f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:1080
   pragma Import (C, nppsMean_64f, "nppsMean_64f");

  --* 
  -- * 64-bit double complex vector mean method
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pMean Pointer to the output result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsMeanGetBufferSize_64fc to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMean_64fc
     (pSrc : access constant nppdefs_h.Npp64fc;
      nLength : int;
      pMean : access nppdefs_h.Npp64fc;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:1092
   pragma Import (C, nppsMean_64fc, "nppsMean_64fc");

  --* 
  -- * 16-bit short vector mean with integer scaling method
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pMean Pointer to the output result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsMeanGetBufferSize_16s_Sfs to determine the minium number of bytes required.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMean_16s_Sfs
     (pSrc : access nppdefs_h.Npp16s;
      nLength : int;
      pMean : access nppdefs_h.Npp16s;
      nScaleFactor : int;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:1105
   pragma Import (C, nppsMean_16s_Sfs, "nppsMean_16s_Sfs");

  --* 
  -- * 32-bit integer vector mean with integer scaling method
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pMean Pointer to the output result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsMeanGetBufferSize_32s_Sfs to determine the minium number of bytes required.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMean_32s_Sfs
     (pSrc : access nppdefs_h.Npp32s;
      nLength : int;
      pMean : access nppdefs_h.Npp32s;
      nScaleFactor : int;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:1119
   pragma Import (C, nppsMean_32s_Sfs, "nppsMean_32s_Sfs");

  --* 
  -- * 16-bit short complex vector mean with integer scaling method
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pMean Pointer to the output result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsMeanGetBufferSize_16sc_Sfs to determine the minium number of bytes required.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMean_16sc_Sfs
     (pSrc : access constant nppdefs_h.Npp16sc;
      nLength : int;
      pMean : access nppdefs_h.Npp16sc;
      nScaleFactor : int;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:1133
   pragma Import (C, nppsMean_16sc_Sfs, "nppsMean_16sc_Sfs");

  --* @} signal_mean  
  --* @defgroup signal_standard_deviation Standard Deviation
  -- *
  -- * @{
  -- *
  --  

  --* 
  -- * Device scratch buffer size (in bytes) for nppsStdDev_32f.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
  -- *        <em>host pointer.</em> \ref general_scratch_buffer.
  -- * \return NPP_SUCCESS
  --  

  -- host pointer  
   function nppsStdDevGetBufferSize_32f (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:1152
   pragma Import (C, nppsStdDevGetBufferSize_32f, "nppsStdDevGetBufferSize_32f");

  --* 
  -- * Device scratch buffer size (in bytes) for nppsStdDev_64f.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
  -- *        <em>host pointer.</em> \ref general_scratch_buffer.
  -- * \return NPP_SUCCESS
  --  

  -- host pointer  
   function nppsStdDevGetBufferSize_64f (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:1162
   pragma Import (C, nppsStdDevGetBufferSize_64f, "nppsStdDevGetBufferSize_64f");

  --* 
  -- * Device scratch buffer size (in bytes) for nppsStdDev_16s32s_Sfs.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
  -- *        <em>host pointer.</em> \ref general_scratch_buffer.
  -- * \return NPP_SUCCESS
  --  

  -- host pointer  
   function nppsStdDevGetBufferSize_16s32s_Sfs (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:1172
   pragma Import (C, nppsStdDevGetBufferSize_16s32s_Sfs, "nppsStdDevGetBufferSize_16s32s_Sfs");

  --* 
  -- * Device scratch buffer size (in bytes) for nppsStdDev_16s_Sfs.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
  -- *        <em>host pointer.</em> \ref general_scratch_buffer.
  -- * \return NPP_SUCCESS
  --  

  -- host pointer  
   function nppsStdDevGetBufferSize_16s_Sfs (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:1182
   pragma Import (C, nppsStdDevGetBufferSize_16s_Sfs, "nppsStdDevGetBufferSize_16s_Sfs");

  --* 
  -- * 32-bit float vector standard deviation method
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pStdDev Pointer to the output result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsStdDevGetBufferSize_32f to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsStdDev_32f
     (pSrc : access nppdefs_h.Npp32f;
      nLength : int;
      pStdDev : access nppdefs_h.Npp32f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:1194
   pragma Import (C, nppsStdDev_32f, "nppsStdDev_32f");

  --* 
  -- * 64-bit float vector standard deviation method
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pStdDev Pointer to the output result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsStdDevGetBufferSize_64f to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsStdDev_64f
     (pSrc : access nppdefs_h.Npp64f;
      nLength : int;
      pStdDev : access nppdefs_h.Npp64f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:1206
   pragma Import (C, nppsStdDev_64f, "nppsStdDev_64f");

  --* 
  -- * 16-bit float vector standard deviation method (return value is 32-bit)
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pStdDev Pointer to the output result.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsStdDevGetBufferSize_16s32s_Sfs to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsStdDev_16s32s_Sfs
     (pSrc : access nppdefs_h.Npp16s;
      nLength : int;
      pStdDev : access nppdefs_h.Npp32s;
      nScaleFactor : int;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:1219
   pragma Import (C, nppsStdDev_16s32s_Sfs, "nppsStdDev_16s32s_Sfs");

  --* 
  -- * 16-bit float vector standard deviation method (return value is also 16-bit)
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pStdDev Pointer to the output result.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsStdDevGetBufferSize_16s_Sfs to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsStdDev_16s_Sfs
     (pSrc : access nppdefs_h.Npp16s;
      nLength : int;
      pStdDev : access nppdefs_h.Npp16s;
      nScaleFactor : int;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:1233
   pragma Import (C, nppsStdDev_16s_Sfs, "nppsStdDev_16s_Sfs");

  --* @} signal_standard_deviation  
  --* @defgroup signal_mean_and_standard_deviation Mean And Standard Deviation
  -- *
  -- * @{
  -- *
  --  

  --* 
  -- * Device scratch buffer size (in bytes) for nppsMeanStdDev_32f.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
  -- *        <em>host pointer.</em> \ref general_scratch_buffer.
  -- * \return NPP_SUCCESS
  --  

  -- host pointer  
   function nppsMeanStdDevGetBufferSize_32f (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:1252
   pragma Import (C, nppsMeanStdDevGetBufferSize_32f, "nppsMeanStdDevGetBufferSize_32f");

  --* 
  -- * Device scratch buffer size (in bytes) for nppsMeanStdDev_64f.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
  -- *        <em>host pointer.</em> \ref general_scratch_buffer.
  -- * \return NPP_SUCCESS
  --  

  -- host pointer  
   function nppsMeanStdDevGetBufferSize_64f (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:1262
   pragma Import (C, nppsMeanStdDevGetBufferSize_64f, "nppsMeanStdDevGetBufferSize_64f");

  --* 
  -- * Device scratch buffer size (in bytes) for nppsMeanStdDev_16s32s_Sfs.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
  -- *        <em>host pointer.</em> \ref general_scratch_buffer.
  -- * \return NPP_SUCCESS
  --  

  -- host pointer  
   function nppsMeanStdDevGetBufferSize_16s32s_Sfs (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:1272
   pragma Import (C, nppsMeanStdDevGetBufferSize_16s32s_Sfs, "nppsMeanStdDevGetBufferSize_16s32s_Sfs");

  --* 
  -- * Device scratch buffer size (in bytes) for nppsMeanStdDev_16s_Sfs.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
  -- *        <em>host pointer.</em> \ref general_scratch_buffer.
  -- * \return NPP_SUCCESS
  --  

  -- host pointer  
   function nppsMeanStdDevGetBufferSize_16s_Sfs (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:1282
   pragma Import (C, nppsMeanStdDevGetBufferSize_16s_Sfs, "nppsMeanStdDevGetBufferSize_16s_Sfs");

  --* 
  -- * 32-bit float vector mean and standard deviation method
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pMean Pointer to the output mean value.
  -- * \param pStdDev Pointer to the output standard deviation value.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsMeanStdDevGetBufferSize_32f to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMeanStdDev_32f
     (pSrc : access nppdefs_h.Npp32f;
      nLength : int;
      pMean : access nppdefs_h.Npp32f;
      pStdDev : access nppdefs_h.Npp32f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:1295
   pragma Import (C, nppsMeanStdDev_32f, "nppsMeanStdDev_32f");

  --* 
  -- * 64-bit float vector mean and standard deviation method
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pMean Pointer to the output mean value.
  -- * \param pStdDev Pointer to the output standard deviation value.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsMeanStdDevGetBufferSize_64f to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMeanStdDev_64f
     (pSrc : access nppdefs_h.Npp64f;
      nLength : int;
      pMean : access nppdefs_h.Npp64f;
      pStdDev : access nppdefs_h.Npp64f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:1308
   pragma Import (C, nppsMeanStdDev_64f, "nppsMeanStdDev_64f");

  --* 
  -- * 16-bit float vector mean and standard deviation method (return values are 32-bit)
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pMean Pointer to the output mean value.
  -- * \param pStdDev Pointer to the output standard deviation value.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsMeanStdDevGetBufferSize_16s32s_Sfs to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMeanStdDev_16s32s_Sfs
     (pSrc : access nppdefs_h.Npp16s;
      nLength : int;
      pMean : access nppdefs_h.Npp32s;
      pStdDev : access nppdefs_h.Npp32s;
      nScaleFactor : int;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:1322
   pragma Import (C, nppsMeanStdDev_16s32s_Sfs, "nppsMeanStdDev_16s32s_Sfs");

  --* 
  -- * 16-bit float vector mean and standard deviation method (return values are also 16-bit)
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pMean Pointer to the output mean value.
  -- * \param pStdDev Pointer to the output standard deviation value.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsMeanStdDevGetBufferSize_16s_Sfs to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMeanStdDev_16s_Sfs
     (pSrc : access nppdefs_h.Npp16s;
      nLength : int;
      pMean : access nppdefs_h.Npp16s;
      pStdDev : access nppdefs_h.Npp16s;
      nScaleFactor : int;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:1337
   pragma Import (C, nppsMeanStdDev_16s_Sfs, "nppsMeanStdDev_16s_Sfs");

  --* @} signal_mean_and_standard_deviation  
  --* @defgroup signal_min_max Minimum_Maximum
  -- *
  -- * @{
  -- *
  --  

  --* 
  -- * Device-buffer size (in bytes) for nppsMinMax_8u.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsMinMaxGetBufferSize_8u (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:1356
   pragma Import (C, nppsMinMaxGetBufferSize_8u, "nppsMinMaxGetBufferSize_8u");

  --* 
  -- * Device-buffer size (in bytes) for nppsMinMax_16s.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsMinMaxGetBufferSize_16s (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:1366
   pragma Import (C, nppsMinMaxGetBufferSize_16s, "nppsMinMaxGetBufferSize_16s");

  --* 
  -- * Device-buffer size (in bytes) for nppsMinMax_16u.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsMinMaxGetBufferSize_16u (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:1376
   pragma Import (C, nppsMinMaxGetBufferSize_16u, "nppsMinMaxGetBufferSize_16u");

  --* 
  -- * Device-buffer size (in bytes) for nppsMinMax_32s.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsMinMaxGetBufferSize_32s (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:1386
   pragma Import (C, nppsMinMaxGetBufferSize_32s, "nppsMinMaxGetBufferSize_32s");

  --* 
  -- * Device-buffer size (in bytes) for nppsMinMax_32u.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsMinMaxGetBufferSize_32u (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:1396
   pragma Import (C, nppsMinMaxGetBufferSize_32u, "nppsMinMaxGetBufferSize_32u");

  --* 
  -- * Device-buffer size (in bytes) for nppsMinMax_32f.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsMinMaxGetBufferSize_32f (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:1406
   pragma Import (C, nppsMinMaxGetBufferSize_32f, "nppsMinMaxGetBufferSize_32f");

  --* 
  -- * Device-buffer size (in bytes) for nppsMinMax_64f.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsMinMaxGetBufferSize_64f (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:1416
   pragma Import (C, nppsMinMaxGetBufferSize_64f, "nppsMinMaxGetBufferSize_64f");

  --* 
  -- * 8-bit char vector min and max method
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pMin Pointer to the min output result.
  -- * \param pMax Pointer to the max output result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsMinMaxGetBufferSize_8u to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMinMax_8u
     (pSrc : access nppdefs_h.Npp8u;
      nLength : int;
      pMin : access nppdefs_h.Npp8u;
      pMax : access nppdefs_h.Npp8u;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:1429
   pragma Import (C, nppsMinMax_8u, "nppsMinMax_8u");

  --* 
  -- * 16-bit signed short vector min and max method
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pMin Pointer to the min output result.
  -- * \param pMax Pointer to the max output result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsMinMaxGetBufferSize_16s to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMinMax_16s
     (pSrc : access nppdefs_h.Npp16s;
      nLength : int;
      pMin : access nppdefs_h.Npp16s;
      pMax : access nppdefs_h.Npp16s;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:1443
   pragma Import (C, nppsMinMax_16s, "nppsMinMax_16s");

  --* 
  -- * 16-bit unsigned short vector min and max method
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pMin Pointer to the min output result.
  -- * \param pMax Pointer to the max output result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsMinMaxGetBufferSize_16u to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMinMax_16u
     (pSrc : access nppdefs_h.Npp16u;
      nLength : int;
      pMin : access nppdefs_h.Npp16u;
      pMax : access nppdefs_h.Npp16u;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:1457
   pragma Import (C, nppsMinMax_16u, "nppsMinMax_16u");

  --* 
  -- * 32-bit unsigned int vector min and max method
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pMin Pointer to the min output result.
  -- * \param pMax Pointer to the max output result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsMinMaxGetBufferSize_32u to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMinMax_32u
     (pSrc : access nppdefs_h.Npp32u;
      nLength : int;
      pMin : access nppdefs_h.Npp32u;
      pMax : access nppdefs_h.Npp32u;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:1471
   pragma Import (C, nppsMinMax_32u, "nppsMinMax_32u");

  --* 
  -- * 32-bit signed int vector min and max method
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pMin Pointer to the min output result.
  -- * \param pMax Pointer to the max output result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsMinMaxGetBufferSize_32s to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMinMax_32s
     (pSrc : access nppdefs_h.Npp32s;
      nLength : int;
      pMin : access nppdefs_h.Npp32s;
      pMax : access nppdefs_h.Npp32s;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:1485
   pragma Import (C, nppsMinMax_32s, "nppsMinMax_32s");

  --* 
  -- * 32-bit float vector min and max method
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pMin Pointer to the min output result.
  -- * \param pMax Pointer to the max output result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsMinMaxGetBufferSize_32f to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMinMax_32f
     (pSrc : access nppdefs_h.Npp32f;
      nLength : int;
      pMin : access nppdefs_h.Npp32f;
      pMax : access nppdefs_h.Npp32f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:1499
   pragma Import (C, nppsMinMax_32f, "nppsMinMax_32f");

  --* 
  -- * 64-bit double vector min and max method
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pMin Pointer to the min output result.
  -- * \param pMax Pointer to the max output result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsMinMaxGetBufferSize_64f to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMinMax_64f
     (pSrc : access nppdefs_h.Npp64f;
      nLength : int;
      pMin : access nppdefs_h.Npp64f;
      pMax : access nppdefs_h.Npp64f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:1513
   pragma Import (C, nppsMinMax_64f, "nppsMinMax_64f");

  --* 
  -- * Device-buffer size (in bytes) for nppsMinMaxIndx_8u.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsMinMaxIndxGetBufferSize_8u (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:1524
   pragma Import (C, nppsMinMaxIndxGetBufferSize_8u, "nppsMinMaxIndxGetBufferSize_8u");

  --* 
  -- * Device-buffer size (in bytes) for nppsMinMaxIndx_16s.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsMinMaxIndxGetBufferSize_16s (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:1534
   pragma Import (C, nppsMinMaxIndxGetBufferSize_16s, "nppsMinMaxIndxGetBufferSize_16s");

  --* 
  -- * Device-buffer size (in bytes) for nppsMinMaxIndx_16u.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsMinMaxIndxGetBufferSize_16u (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:1544
   pragma Import (C, nppsMinMaxIndxGetBufferSize_16u, "nppsMinMaxIndxGetBufferSize_16u");

  --* 
  -- * Device-buffer size (in bytes) for nppsMinMaxIndx_32s.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsMinMaxIndxGetBufferSize_32s (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:1554
   pragma Import (C, nppsMinMaxIndxGetBufferSize_32s, "nppsMinMaxIndxGetBufferSize_32s");

  --* 
  -- * Device-buffer size (in bytes) for nppsMinMaxIndx_32u.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsMinMaxIndxGetBufferSize_32u (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:1564
   pragma Import (C, nppsMinMaxIndxGetBufferSize_32u, "nppsMinMaxIndxGetBufferSize_32u");

  --* 
  -- * Device-buffer size (in bytes) for nppsMinMaxIndx_32f.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsMinMaxIndxGetBufferSize_32f (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:1574
   pragma Import (C, nppsMinMaxIndxGetBufferSize_32f, "nppsMinMaxIndxGetBufferSize_32f");

  --* 
  -- * Device-buffer size (in bytes) for nppsMinMaxIndx_64f.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsMinMaxIndxGetBufferSize_64f (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:1584
   pragma Import (C, nppsMinMaxIndxGetBufferSize_64f, "nppsMinMaxIndxGetBufferSize_64f");

  --* 
  -- * 8-bit char vector min and max with indices method
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pMin Pointer to the min output result.
  -- * \param pMinIndx Pointer to the index of the first min value.
  -- * \param pMax Pointer to the max output result.
  -- * \param pMaxIndx Pointer to the index of the first max value.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsMinMaxIndxGetBufferSize_8u to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMinMaxIndx_8u
     (pSrc : access nppdefs_h.Npp8u;
      nLength : int;
      pMin : access nppdefs_h.Npp8u;
      pMinIndx : access int;
      pMax : access nppdefs_h.Npp8u;
      pMaxIndx : access int;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:1599
   pragma Import (C, nppsMinMaxIndx_8u, "nppsMinMaxIndx_8u");

  --* 
  -- * 16-bit signed short vector min and max with indices method
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pMin Pointer to the min output result.
  -- * \param pMinIndx Pointer to the index of the first min value.
  -- * \param pMax Pointer to the max output result.
  -- * \param pMaxIndx Pointer to the index of the first max value.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsMinMaxIndxGetBufferSize_16s to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMinMaxIndx_16s
     (pSrc : access nppdefs_h.Npp16s;
      nLength : int;
      pMin : access nppdefs_h.Npp16s;
      pMinIndx : access int;
      pMax : access nppdefs_h.Npp16s;
      pMaxIndx : access int;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:1615
   pragma Import (C, nppsMinMaxIndx_16s, "nppsMinMaxIndx_16s");

  --* 
  -- * 16-bit unsigned short vector min and max with indices method
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pMin Pointer to the min output result.
  -- * \param pMinIndx Pointer to the index of the first min value.
  -- * \param pMax Pointer to the max output result.
  -- * \param pMaxIndx Pointer to the index of the first max value.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsMinMaxIndxGetBufferSize_16u to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMinMaxIndx_16u
     (pSrc : access nppdefs_h.Npp16u;
      nLength : int;
      pMin : access nppdefs_h.Npp16u;
      pMinIndx : access int;
      pMax : access nppdefs_h.Npp16u;
      pMaxIndx : access int;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:1631
   pragma Import (C, nppsMinMaxIndx_16u, "nppsMinMaxIndx_16u");

  --* 
  -- * 32-bit signed short vector min and max with indices method
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pMin Pointer to the min output result.
  -- * \param pMinIndx Pointer to the index of the first min value.
  -- * \param pMax Pointer to the max output result.
  -- * \param pMaxIndx Pointer to the index of the first max value.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsMinMaxIndxGetBufferSize_32s to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMinMaxIndx_32s
     (pSrc : access nppdefs_h.Npp32s;
      nLength : int;
      pMin : access nppdefs_h.Npp32s;
      pMinIndx : access int;
      pMax : access nppdefs_h.Npp32s;
      pMaxIndx : access int;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:1647
   pragma Import (C, nppsMinMaxIndx_32s, "nppsMinMaxIndx_32s");

  --* 
  -- * 32-bit unsigned short vector min and max with indices method
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pMin Pointer to the min output result.
  -- * \param pMinIndx Pointer to the index of the first min value.
  -- * \param pMax Pointer to the max output result.
  -- * \param pMaxIndx Pointer to the index of the first max value.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsMinMaxIndxGetBufferSize_32u to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMinMaxIndx_32u
     (pSrc : access nppdefs_h.Npp32u;
      nLength : int;
      pMin : access nppdefs_h.Npp32u;
      pMinIndx : access int;
      pMax : access nppdefs_h.Npp32u;
      pMaxIndx : access int;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:1663
   pragma Import (C, nppsMinMaxIndx_32u, "nppsMinMaxIndx_32u");

  --* 
  -- * 32-bit float vector min and max with indices method
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pMin Pointer to the min output result.
  -- * \param pMinIndx Pointer to the index of the first min value.
  -- * \param pMax Pointer to the max output result.
  -- * \param pMaxIndx Pointer to the index of the first max value.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsMinMaxIndxGetBufferSize_32f to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMinMaxIndx_32f
     (pSrc : access nppdefs_h.Npp32f;
      nLength : int;
      pMin : access nppdefs_h.Npp32f;
      pMinIndx : access int;
      pMax : access nppdefs_h.Npp32f;
      pMaxIndx : access int;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:1679
   pragma Import (C, nppsMinMaxIndx_32f, "nppsMinMaxIndx_32f");

  --* 
  -- * 64-bit float vector min and max with indices method
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pMin Pointer to the min output result.
  -- * \param pMinIndx Pointer to the index of the first min value.
  -- * \param pMax Pointer to the max output result.
  -- * \param pMaxIndx Pointer to the index of the first max value.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsMinMaxIndxGetBufferSize_64f to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMinMaxIndx_64f
     (pSrc : access nppdefs_h.Npp64f;
      nLength : int;
      pMin : access nppdefs_h.Npp64f;
      pMinIndx : access int;
      pMax : access nppdefs_h.Npp64f;
      pMaxIndx : access int;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:1695
   pragma Import (C, nppsMinMaxIndx_64f, "nppsMinMaxIndx_64f");

  --* @} signal_min_max  
  --* @defgroup signal_infinity_norm Infinity Norm
  -- *
  -- * @{
  -- *
  --  

  --* 
  -- * Device-buffer size (in bytes) for nppsNorm_Inf_32f.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsNormInfGetBufferSize_32f (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:1714
   pragma Import (C, nppsNormInfGetBufferSize_32f, "nppsNormInfGetBufferSize_32f");

  --* 
  -- * 32-bit float vector C norm method
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pNorm Pointer to the norm result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsNormInfGetBufferSize_32f to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsNorm_Inf_32f
     (pSrc : access nppdefs_h.Npp32f;
      nLength : int;
      pNorm : access nppdefs_h.Npp32f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:1726
   pragma Import (C, nppsNorm_Inf_32f, "nppsNorm_Inf_32f");

  --* 
  -- * Device-buffer size (in bytes) for nppsNorm_Inf_64f.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsNormInfGetBufferSize_64f (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:1737
   pragma Import (C, nppsNormInfGetBufferSize_64f, "nppsNormInfGetBufferSize_64f");

  --* 
  -- * 64-bit float vector C norm method
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pNorm Pointer to the norm result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsNormInfGetBufferSize_64f to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsNorm_Inf_64f
     (pSrc : access nppdefs_h.Npp64f;
      nLength : int;
      pNorm : access nppdefs_h.Npp64f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:1749
   pragma Import (C, nppsNorm_Inf_64f, "nppsNorm_Inf_64f");

  --* 
  -- * Device-buffer size (in bytes) for nppsNorm_Inf_16s32f.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsNormInfGetBufferSize_16s32f (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:1760
   pragma Import (C, nppsNormInfGetBufferSize_16s32f, "nppsNormInfGetBufferSize_16s32f");

  --* 
  -- * 16-bit signed short integer vector C norm method, return value is 32-bit float.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pNorm Pointer to the norm result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsNormInfGetBufferSize_16s32f to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsNorm_Inf_16s32f
     (pSrc : access nppdefs_h.Npp16s;
      nLength : int;
      pNorm : access nppdefs_h.Npp32f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:1772
   pragma Import (C, nppsNorm_Inf_16s32f, "nppsNorm_Inf_16s32f");

  --* 
  -- * Device-buffer size (in bytes) for nppsNorm_Inf_32fc32f.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsNormInfGetBufferSize_32fc32f (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:1783
   pragma Import (C, nppsNormInfGetBufferSize_32fc32f, "nppsNormInfGetBufferSize_32fc32f");

  --* 
  -- * 32-bit float complex vector C norm method, return value is 32-bit float.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pNorm Pointer to the norm result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsNormInfGetBufferSize_32fc32f to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsNorm_Inf_32fc32f
     (pSrc : access constant nppdefs_h.Npp32fc;
      nLength : int;
      pNorm : access nppdefs_h.Npp32f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:1795
   pragma Import (C, nppsNorm_Inf_32fc32f, "nppsNorm_Inf_32fc32f");

  --* 
  -- * Device-buffer size (in bytes) for nppsNorm_Inf_64fc64f.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsNormInfGetBufferSize_64fc64f (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:1806
   pragma Import (C, nppsNormInfGetBufferSize_64fc64f, "nppsNormInfGetBufferSize_64fc64f");

  --* 
  -- * 64-bit float complex vector C norm method, return value is 64-bit float.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pNorm Pointer to the norm result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsNormInfGetBufferSize_64fc64f to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsNorm_Inf_64fc64f
     (pSrc : access constant nppdefs_h.Npp64fc;
      nLength : int;
      pNorm : access nppdefs_h.Npp64f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:1818
   pragma Import (C, nppsNorm_Inf_64fc64f, "nppsNorm_Inf_64fc64f");

  --* 
  -- * Device-buffer size (in bytes) for nppsNorm_Inf_16s32s_Sfs.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsNormInfGetBufferSize_16s32s_Sfs (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:1829
   pragma Import (C, nppsNormInfGetBufferSize_16s32s_Sfs, "nppsNormInfGetBufferSize_16s32s_Sfs");

  --* 
  -- * 16-bit signed short integer vector C norm method, return value is 32-bit signed integer.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pNorm Pointer to the norm result.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsNormInfGetBufferSize_16s32s_Sfs to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsNorm_Inf_16s32s_Sfs
     (pSrc : access nppdefs_h.Npp16s;
      nLength : int;
      pNorm : access nppdefs_h.Npp32s;
      nScaleFactor : int;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:1842
   pragma Import (C, nppsNorm_Inf_16s32s_Sfs, "nppsNorm_Inf_16s32s_Sfs");

  --* @} signal_infinity_norm  
  --* @defgroup signal_L1_norm L1 Norm
  -- *
  -- * @{
  -- *
  --  

  --* 
  -- * Device-buffer size (in bytes) for nppsNorm_L1_32f.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsNormL1GetBufferSize_32f (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:1861
   pragma Import (C, nppsNormL1GetBufferSize_32f, "nppsNormL1GetBufferSize_32f");

  --* 
  -- * 32-bit float vector L1 norm method
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pNorm Pointer to the norm result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsNormL1GetBufferSize_32f to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsNorm_L1_32f
     (pSrc : access nppdefs_h.Npp32f;
      nLength : int;
      pNorm : access nppdefs_h.Npp32f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:1873
   pragma Import (C, nppsNorm_L1_32f, "nppsNorm_L1_32f");

  --* 
  -- * Device-buffer size (in bytes) for nppsNorm_L1_64f.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsNormL1GetBufferSize_64f (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:1884
   pragma Import (C, nppsNormL1GetBufferSize_64f, "nppsNormL1GetBufferSize_64f");

  --* 
  -- * 64-bit float vector L1 norm method
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pNorm Pointer to the norm result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsNormL1GetBufferSize_64f to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsNorm_L1_64f
     (pSrc : access nppdefs_h.Npp64f;
      nLength : int;
      pNorm : access nppdefs_h.Npp64f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:1896
   pragma Import (C, nppsNorm_L1_64f, "nppsNorm_L1_64f");

  --* 
  -- * Device-buffer size (in bytes) for nppsNorm_L1_16s32f.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsNormL1GetBufferSize_16s32f (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:1907
   pragma Import (C, nppsNormL1GetBufferSize_16s32f, "nppsNormL1GetBufferSize_16s32f");

  --* 
  -- * 16-bit signed short integer vector L1 norm method, return value is 32-bit float.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pNorm Pointer to the L1 norm result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsNormL1GetBufferSize_16s32f to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsNorm_L1_16s32f
     (pSrc : access nppdefs_h.Npp16s;
      nLength : int;
      pNorm : access nppdefs_h.Npp32f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:1919
   pragma Import (C, nppsNorm_L1_16s32f, "nppsNorm_L1_16s32f");

  --* 
  -- * Device-buffer size (in bytes) for nppsNorm_L1_32fc64f.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsNormL1GetBufferSize_32fc64f (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:1930
   pragma Import (C, nppsNormL1GetBufferSize_32fc64f, "nppsNormL1GetBufferSize_32fc64f");

  --* 
  -- * 32-bit float complex vector L1 norm method, return value is 64-bit float.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pNorm Pointer to the norm result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsNormL1GetBufferSize_32fc64f to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsNorm_L1_32fc64f
     (pSrc : access constant nppdefs_h.Npp32fc;
      nLength : int;
      pNorm : access nppdefs_h.Npp64f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:1942
   pragma Import (C, nppsNorm_L1_32fc64f, "nppsNorm_L1_32fc64f");

  --* 
  -- * Device-buffer size (in bytes) for nppsNorm_L1_64fc64f.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsNormL1GetBufferSize_64fc64f (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:1953
   pragma Import (C, nppsNormL1GetBufferSize_64fc64f, "nppsNormL1GetBufferSize_64fc64f");

  --* 
  -- * 64-bit float complex vector L1 norm method, return value is 64-bit float.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pNorm Pointer to the norm result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsNormL1GetBufferSize_64fc64f to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsNorm_L1_64fc64f
     (pSrc : access constant nppdefs_h.Npp64fc;
      nLength : int;
      pNorm : access nppdefs_h.Npp64f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:1965
   pragma Import (C, nppsNorm_L1_64fc64f, "nppsNorm_L1_64fc64f");

  --* 
  -- * Device-buffer size (in bytes) for nppsNorm_L1_16s32s_Sfs.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsNormL1GetBufferSize_16s32s_Sfs (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:1976
   pragma Import (C, nppsNormL1GetBufferSize_16s32s_Sfs, "nppsNormL1GetBufferSize_16s32s_Sfs");

  --* 
  -- * 16-bit signed short integer vector L1 norm method, return value is 32-bit signed integer.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pNorm Pointer to the norm result.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsNormL1GetBufferSize_16s32s_Sfs to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsNorm_L1_16s32s_Sfs
     (pSrc : access nppdefs_h.Npp16s;
      nLength : int;
      pNorm : access nppdefs_h.Npp32s;
      nScaleFactor : int;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:1989
   pragma Import (C, nppsNorm_L1_16s32s_Sfs, "nppsNorm_L1_16s32s_Sfs");

  --* 
  -- * Device-buffer size (in bytes) for nppsNorm_L1_16s64s_Sfs.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsNormL1GetBufferSize_16s64s_Sfs (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:2000
   pragma Import (C, nppsNormL1GetBufferSize_16s64s_Sfs, "nppsNormL1GetBufferSize_16s64s_Sfs");

  --* 
  -- * 16-bit signed short integer vector L1 norm method, return value is 64-bit signed integer.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pNorm Pointer to the norm result.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsNormL1GetBufferSize_16s64s_Sfs to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsNorm_L1_16s64s_Sfs
     (pSrc : access nppdefs_h.Npp16s;
      nLength : int;
      pNorm : access nppdefs_h.Npp64s;
      nScaleFactor : int;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:2013
   pragma Import (C, nppsNorm_L1_16s64s_Sfs, "nppsNorm_L1_16s64s_Sfs");

  --* @} signal_L1_norm  
  --* @defgroup signal_L2_norm L2 Norm
  -- *
  -- * @{
  -- *
  --  

  --* 
  -- * Device-buffer size (in bytes) for nppsNorm_L2_32f.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsNormL2GetBufferSize_32f (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:2032
   pragma Import (C, nppsNormL2GetBufferSize_32f, "nppsNormL2GetBufferSize_32f");

  --* 
  -- * 32-bit float vector L2 norm method
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pNorm Pointer to the norm result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsNormL2GetBufferSize_32f to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsNorm_L2_32f
     (pSrc : access nppdefs_h.Npp32f;
      nLength : int;
      pNorm : access nppdefs_h.Npp32f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:2044
   pragma Import (C, nppsNorm_L2_32f, "nppsNorm_L2_32f");

  --* 
  -- * Device-buffer size (in bytes) for nppsNorm_L2_64f.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsNormL2GetBufferSize_64f (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:2054
   pragma Import (C, nppsNormL2GetBufferSize_64f, "nppsNormL2GetBufferSize_64f");

  --* 
  -- * 64-bit float vector L2 norm method
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pNorm Pointer to the norm result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsNormL2GetBufferSize_64f to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsNorm_L2_64f
     (pSrc : access nppdefs_h.Npp64f;
      nLength : int;
      pNorm : access nppdefs_h.Npp64f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:2066
   pragma Import (C, nppsNorm_L2_64f, "nppsNorm_L2_64f");

  --* 
  -- * Device-buffer size (in bytes) for nppsNorm_L2_16s32f.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsNormL2GetBufferSize_16s32f (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:2077
   pragma Import (C, nppsNormL2GetBufferSize_16s32f, "nppsNormL2GetBufferSize_16s32f");

  --* 
  -- * 16-bit signed short integer vector L2 norm method, return value is 32-bit float.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pNorm Pointer to the norm result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsNormL2GetBufferSize_16s32f to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsNorm_L2_16s32f
     (pSrc : access nppdefs_h.Npp16s;
      nLength : int;
      pNorm : access nppdefs_h.Npp32f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:2089
   pragma Import (C, nppsNorm_L2_16s32f, "nppsNorm_L2_16s32f");

  --* 
  -- * Device-buffer size (in bytes) for nppsNorm_L2_32fc64f.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsNormL2GetBufferSize_32fc64f (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:2100
   pragma Import (C, nppsNormL2GetBufferSize_32fc64f, "nppsNormL2GetBufferSize_32fc64f");

  --* 
  -- * 32-bit float complex vector L2 norm method, return value is 64-bit float.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pNorm Pointer to the norm result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsNormL2GetBufferSize_32fc64f to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsNorm_L2_32fc64f
     (pSrc : access constant nppdefs_h.Npp32fc;
      nLength : int;
      pNorm : access nppdefs_h.Npp64f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:2112
   pragma Import (C, nppsNorm_L2_32fc64f, "nppsNorm_L2_32fc64f");

  --* 
  -- * Device-buffer size (in bytes) for nppsNorm_L2_64fc64f.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsNormL2GetBufferSize_64fc64f (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:2123
   pragma Import (C, nppsNormL2GetBufferSize_64fc64f, "nppsNormL2GetBufferSize_64fc64f");

  --* 
  -- * 64-bit float complex vector L2 norm method, return value is 64-bit float.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pNorm Pointer to the norm result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsNormL2GetBufferSize_64fc64f to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsNorm_L2_64fc64f
     (pSrc : access constant nppdefs_h.Npp64fc;
      nLength : int;
      pNorm : access nppdefs_h.Npp64f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:2135
   pragma Import (C, nppsNorm_L2_64fc64f, "nppsNorm_L2_64fc64f");

  --* 
  -- * Device-buffer size (in bytes) for nppsNorm_L2_16s32s_Sfs.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsNormL2GetBufferSize_16s32s_Sfs (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:2146
   pragma Import (C, nppsNormL2GetBufferSize_16s32s_Sfs, "nppsNormL2GetBufferSize_16s32s_Sfs");

  --* 
  -- * 16-bit signed short integer vector L2 norm method, return value is 32-bit signed integer.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pNorm Pointer to the norm result.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsNormL2GetBufferSize_16s32s_Sfs to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsNorm_L2_16s32s_Sfs
     (pSrc : access nppdefs_h.Npp16s;
      nLength : int;
      pNorm : access nppdefs_h.Npp32s;
      nScaleFactor : int;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:2159
   pragma Import (C, nppsNorm_L2_16s32s_Sfs, "nppsNorm_L2_16s32s_Sfs");

  --* 
  -- * Device-buffer size (in bytes) for nppsNorm_L2Sqr_16s64s_Sfs.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsNormL2SqrGetBufferSize_16s64s_Sfs (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:2170
   pragma Import (C, nppsNormL2SqrGetBufferSize_16s64s_Sfs, "nppsNormL2SqrGetBufferSize_16s64s_Sfs");

  --* 
  -- * 16-bit signed short integer vector L2 Square norm method, return value is 64-bit signed integer.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pNorm Pointer to the norm result.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsNormL2SqrGetBufferSize_16s64s_Sfs to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsNorm_L2Sqr_16s64s_Sfs
     (pSrc : access nppdefs_h.Npp16s;
      nLength : int;
      pNorm : access nppdefs_h.Npp64s;
      nScaleFactor : int;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:2183
   pragma Import (C, nppsNorm_L2Sqr_16s64s_Sfs, "nppsNorm_L2Sqr_16s64s_Sfs");

  --* @} signal_L2_norm  
  --* @defgroup signal_infinity_norm_diff Infinity Norm Diff
  -- *
  -- * @{
  -- *
  --  

  --* 
  -- * Device-buffer size (in bytes) for nppsNormDiff_Inf_32f.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsNormDiffInfGetBufferSize_32f (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:2202
   pragma Import (C, nppsNormDiffInfGetBufferSize_32f, "nppsNormDiffInfGetBufferSize_32f");

  --* 
  -- * 32-bit float C norm method on two vectors' difference
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pNorm Pointer to the norm result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsNormDiffInfGetBufferSize_32f to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsNormDiff_Inf_32f
     (pSrc1 : access nppdefs_h.Npp32f;
      pSrc2 : access nppdefs_h.Npp32f;
      nLength : int;
      pNorm : access nppdefs_h.Npp32f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:2215
   pragma Import (C, nppsNormDiff_Inf_32f, "nppsNormDiff_Inf_32f");

  --* 
  -- * Device-buffer size (in bytes) for nppsNormDiff_Inf_64f.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsNormDiffInfGetBufferSize_64f (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:2226
   pragma Import (C, nppsNormDiffInfGetBufferSize_64f, "nppsNormDiffInfGetBufferSize_64f");

  --* 
  -- * 64-bit float C norm method on two vectors' difference
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pNorm Pointer to the norm result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsNormDiffInfGetBufferSize_64f to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsNormDiff_Inf_64f
     (pSrc1 : access nppdefs_h.Npp64f;
      pSrc2 : access nppdefs_h.Npp64f;
      nLength : int;
      pNorm : access nppdefs_h.Npp64f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:2239
   pragma Import (C, nppsNormDiff_Inf_64f, "nppsNormDiff_Inf_64f");

  --* 
  -- * Device-buffer size (in bytes) for nppsNormDiff_Inf_16s32f.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsNormDiffInfGetBufferSize_16s32f (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:2250
   pragma Import (C, nppsNormDiffInfGetBufferSize_16s32f, "nppsNormDiffInfGetBufferSize_16s32f");

  --* 
  -- * 16-bit signed short integer C norm method on two vectors' difference, return value is 32-bit float.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pNorm Pointer to the norm result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsNormDiffInfGetBufferSize_16s32f to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsNormDiff_Inf_16s32f
     (pSrc1 : access nppdefs_h.Npp16s;
      pSrc2 : access nppdefs_h.Npp16s;
      nLength : int;
      pNorm : access nppdefs_h.Npp32f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:2263
   pragma Import (C, nppsNormDiff_Inf_16s32f, "nppsNormDiff_Inf_16s32f");

  --* 
  -- * Device-buffer size (in bytes) for nppsNormDiff_Inf_32fc32f.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsNormDiffInfGetBufferSize_32fc32f (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:2274
   pragma Import (C, nppsNormDiffInfGetBufferSize_32fc32f, "nppsNormDiffInfGetBufferSize_32fc32f");

  --* 
  -- * 32-bit float complex C norm method on two vectors' difference, return value is 32-bit float.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pNorm Pointer to the norm result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsNormDiffInfGetBufferSize_32fc32f to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsNormDiff_Inf_32fc32f
     (pSrc1 : access constant nppdefs_h.Npp32fc;
      pSrc2 : access constant nppdefs_h.Npp32fc;
      nLength : int;
      pNorm : access nppdefs_h.Npp32f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:2287
   pragma Import (C, nppsNormDiff_Inf_32fc32f, "nppsNormDiff_Inf_32fc32f");

  --* 
  -- * Device-buffer size (in bytes) for nppsNormDiff_Inf_64fc64f.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsNormDiffInfGetBufferSize_64fc64f (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:2298
   pragma Import (C, nppsNormDiffInfGetBufferSize_64fc64f, "nppsNormDiffInfGetBufferSize_64fc64f");

  --* 
  -- * 64-bit float complex C norm method on two vectors' difference, return value is 64-bit float.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pNorm Pointer to the norm result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsNormDiffInfGetBufferSize_64fc64f to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsNormDiff_Inf_64fc64f
     (pSrc1 : access constant nppdefs_h.Npp64fc;
      pSrc2 : access constant nppdefs_h.Npp64fc;
      nLength : int;
      pNorm : access nppdefs_h.Npp64f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:2311
   pragma Import (C, nppsNormDiff_Inf_64fc64f, "nppsNormDiff_Inf_64fc64f");

  --* 
  -- * Device-buffer size (in bytes) for nppsNormDiff_Inf_16s32s_Sfs.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsNormDiffInfGetBufferSize_16s32s_Sfs (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:2322
   pragma Import (C, nppsNormDiffInfGetBufferSize_16s32s_Sfs, "nppsNormDiffInfGetBufferSize_16s32s_Sfs");

  --* 
  -- * 16-bit signed short integer C norm method on two vectors' difference, return value is 32-bit signed integer.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pNorm Pointer to the norm result.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsNormDiffInfGetBufferSize_16s32s_Sfs to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsNormDiff_Inf_16s32s_Sfs
     (pSrc1 : access nppdefs_h.Npp16s;
      pSrc2 : access nppdefs_h.Npp16s;
      nLength : int;
      pNorm : access nppdefs_h.Npp32s;
      nScaleFactor : int;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:2336
   pragma Import (C, nppsNormDiff_Inf_16s32s_Sfs, "nppsNormDiff_Inf_16s32s_Sfs");

  --* @} signal_infinity_norm_diff  
  --* @defgroup signal_L1_norm_diff L1 Norm Diff
  -- *
  -- * @{
  -- *
  --  

  --* 
  -- * Device-buffer size (in bytes) for nppsNormDiff_L1_32f.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsNormDiffL1GetBufferSize_32f (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:2355
   pragma Import (C, nppsNormDiffL1GetBufferSize_32f, "nppsNormDiffL1GetBufferSize_32f");

  --* 
  -- * 32-bit float L1 norm method on two vectors' difference
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pNorm Pointer to the norm result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsNormDiffL1GetBufferSize_32f to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsNormDiff_L1_32f
     (pSrc1 : access nppdefs_h.Npp32f;
      pSrc2 : access nppdefs_h.Npp32f;
      nLength : int;
      pNorm : access nppdefs_h.Npp32f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:2368
   pragma Import (C, nppsNormDiff_L1_32f, "nppsNormDiff_L1_32f");

  --* 
  -- * Device-buffer size (in bytes) for nppsNormDiff_L1_64f.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsNormDiffL1GetBufferSize_64f (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:2379
   pragma Import (C, nppsNormDiffL1GetBufferSize_64f, "nppsNormDiffL1GetBufferSize_64f");

  --* 
  -- * 64-bit float L1 norm method on two vectors' difference
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pNorm Pointer to the norm result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsNormDiffL1GetBufferSize_64f to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsNormDiff_L1_64f
     (pSrc1 : access nppdefs_h.Npp64f;
      pSrc2 : access nppdefs_h.Npp64f;
      nLength : int;
      pNorm : access nppdefs_h.Npp64f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:2392
   pragma Import (C, nppsNormDiff_L1_64f, "nppsNormDiff_L1_64f");

  --* 
  -- * Device-buffer size (in bytes) for nppsNormDiff_L1_16s32f.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsNormDiffL1GetBufferSize_16s32f (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:2403
   pragma Import (C, nppsNormDiffL1GetBufferSize_16s32f, "nppsNormDiffL1GetBufferSize_16s32f");

  --* 
  -- * 16-bit signed short integer L1 norm method on two vectors' difference, return value is 32-bit float.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pNorm Pointer to the L1 norm result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsNormDiffL1GetBufferSize_16s32f to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsNormDiff_L1_16s32f
     (pSrc1 : access nppdefs_h.Npp16s;
      pSrc2 : access nppdefs_h.Npp16s;
      nLength : int;
      pNorm : access nppdefs_h.Npp32f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:2416
   pragma Import (C, nppsNormDiff_L1_16s32f, "nppsNormDiff_L1_16s32f");

  --* 
  -- * Device-buffer size (in bytes) for nppsNormDiff_L1_32fc64f.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsNormDiffL1GetBufferSize_32fc64f (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:2427
   pragma Import (C, nppsNormDiffL1GetBufferSize_32fc64f, "nppsNormDiffL1GetBufferSize_32fc64f");

  --* 
  -- * 32-bit float complex L1 norm method on two vectors' difference, return value is 64-bit float.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pNorm Pointer to the norm result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsNormDiffL1GetBufferSize_32fc64f to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsNormDiff_L1_32fc64f
     (pSrc1 : access constant nppdefs_h.Npp32fc;
      pSrc2 : access constant nppdefs_h.Npp32fc;
      nLength : int;
      pNorm : access nppdefs_h.Npp64f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:2440
   pragma Import (C, nppsNormDiff_L1_32fc64f, "nppsNormDiff_L1_32fc64f");

  --* 
  -- * Device-buffer size (in bytes) for nppsNormDiff_L1_64fc64f.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsNormDiffL1GetBufferSize_64fc64f (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:2451
   pragma Import (C, nppsNormDiffL1GetBufferSize_64fc64f, "nppsNormDiffL1GetBufferSize_64fc64f");

  --* 
  -- * 64-bit float complex L1 norm method on two vectors' difference, return value is 64-bit float.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pNorm Pointer to the norm result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsNormDiffL1GetBufferSize_64fc64f to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsNormDiff_L1_64fc64f
     (pSrc1 : access constant nppdefs_h.Npp64fc;
      pSrc2 : access constant nppdefs_h.Npp64fc;
      nLength : int;
      pNorm : access nppdefs_h.Npp64f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:2464
   pragma Import (C, nppsNormDiff_L1_64fc64f, "nppsNormDiff_L1_64fc64f");

  --* 
  -- * Device-buffer size (in bytes) for nppsNormDiff_L1_16s32s_Sfs.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsNormDiffL1GetBufferSize_16s32s_Sfs (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:2475
   pragma Import (C, nppsNormDiffL1GetBufferSize_16s32s_Sfs, "nppsNormDiffL1GetBufferSize_16s32s_Sfs");

  --* 
  -- * 16-bit signed short integer L1 norm method on two vectors' difference, return value is 32-bit signed integer.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer..
  -- * \param nLength \ref length_specification.
  -- * \param pNorm Pointer to the norm result.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsNormDiffL1GetBufferSize_16s32s_Sfs to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsNormDiff_L1_16s32s_Sfs
     (pSrc1 : access nppdefs_h.Npp16s;
      pSrc2 : access nppdefs_h.Npp16s;
      nLength : int;
      pNorm : access nppdefs_h.Npp32s;
      nScaleFactor : int;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:2489
   pragma Import (C, nppsNormDiff_L1_16s32s_Sfs, "nppsNormDiff_L1_16s32s_Sfs");

  --* 
  -- * Device-buffer size (in bytes) for nppsNormDiff_L1_16s64s_Sfs.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsNormDiffL1GetBufferSize_16s64s_Sfs (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:2500
   pragma Import (C, nppsNormDiffL1GetBufferSize_16s64s_Sfs, "nppsNormDiffL1GetBufferSize_16s64s_Sfs");

  --* 
  -- * 16-bit signed short integer L1 norm method on two vectors' difference, return value is 64-bit signed integer.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pNorm Pointer to the norm result.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsNormDiffL1GetBufferSize_16s64s_Sfs to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsNormDiff_L1_16s64s_Sfs
     (pSrc1 : access nppdefs_h.Npp16s;
      pSrc2 : access nppdefs_h.Npp16s;
      nLength : int;
      pNorm : access nppdefs_h.Npp64s;
      nScaleFactor : int;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:2514
   pragma Import (C, nppsNormDiff_L1_16s64s_Sfs, "nppsNormDiff_L1_16s64s_Sfs");

  --* @} signal_L1_norm_diff  
  --* @defgroup signal_L2_norm_diff L2 Norm Diff
  -- *
  -- * @{
  -- *
  --  

  --* 
  -- * Device-buffer size (in bytes) for nppsNormDiff_L2_32f.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsNormDiffL2GetBufferSize_32f (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:2533
   pragma Import (C, nppsNormDiffL2GetBufferSize_32f, "nppsNormDiffL2GetBufferSize_32f");

  --* 
  -- * 32-bit float L2 norm method on two vectors' difference
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pNorm Pointer to the norm result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsNormDiffL2GetBufferSize_32f to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsNormDiff_L2_32f
     (pSrc1 : access nppdefs_h.Npp32f;
      pSrc2 : access nppdefs_h.Npp32f;
      nLength : int;
      pNorm : access nppdefs_h.Npp32f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:2546
   pragma Import (C, nppsNormDiff_L2_32f, "nppsNormDiff_L2_32f");

  --* 
  -- * Device-buffer size (in bytes) for nppsNormDiff_L2_64f.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsNormDiffL2GetBufferSize_64f (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:2557
   pragma Import (C, nppsNormDiffL2GetBufferSize_64f, "nppsNormDiffL2GetBufferSize_64f");

  --* 
  -- * 64-bit float L2 norm method on two vectors' difference
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pNorm Pointer to the norm result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsNormDiffL2GetBufferSize_64f to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsNormDiff_L2_64f
     (pSrc1 : access nppdefs_h.Npp64f;
      pSrc2 : access nppdefs_h.Npp64f;
      nLength : int;
      pNorm : access nppdefs_h.Npp64f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:2570
   pragma Import (C, nppsNormDiff_L2_64f, "nppsNormDiff_L2_64f");

  --* 
  -- * Device-buffer size (in bytes) for nppsNormDiff_L2_16s32f.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsNormDiffL2GetBufferSize_16s32f (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:2581
   pragma Import (C, nppsNormDiffL2GetBufferSize_16s32f, "nppsNormDiffL2GetBufferSize_16s32f");

  --* 
  -- * 16-bit signed short integer L2 norm method on two vectors' difference, return value is 32-bit float.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pNorm Pointer to the norm result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsNormDiffL2GetBufferSize_16s32f to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsNormDiff_L2_16s32f
     (pSrc1 : access nppdefs_h.Npp16s;
      pSrc2 : access nppdefs_h.Npp16s;
      nLength : int;
      pNorm : access nppdefs_h.Npp32f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:2594
   pragma Import (C, nppsNormDiff_L2_16s32f, "nppsNormDiff_L2_16s32f");

  --* 
  -- * Device-buffer size (in bytes) for nppsNormDiff_L2_32fc64f.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsNormDiffL2GetBufferSize_32fc64f (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:2605
   pragma Import (C, nppsNormDiffL2GetBufferSize_32fc64f, "nppsNormDiffL2GetBufferSize_32fc64f");

  --* 
  -- * 32-bit float complex L2 norm method on two vectors' difference, return value is 64-bit float.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pNorm Pointer to the norm result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsNormDiffL2GetBufferSize_32fc64f to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsNormDiff_L2_32fc64f
     (pSrc1 : access constant nppdefs_h.Npp32fc;
      pSrc2 : access constant nppdefs_h.Npp32fc;
      nLength : int;
      pNorm : access nppdefs_h.Npp64f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:2618
   pragma Import (C, nppsNormDiff_L2_32fc64f, "nppsNormDiff_L2_32fc64f");

  --* 
  -- * Device-buffer size (in bytes) for nppsNormDiff_L2_64fc64f.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsNormDiffL2GetBufferSize_64fc64f (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:2629
   pragma Import (C, nppsNormDiffL2GetBufferSize_64fc64f, "nppsNormDiffL2GetBufferSize_64fc64f");

  --* 
  -- * 64-bit float complex L2 norm method on two vectors' difference, return value is 64-bit float.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pNorm Pointer to the norm result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsNormDiffL2GetBufferSize_64fc64f to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsNormDiff_L2_64fc64f
     (pSrc1 : access constant nppdefs_h.Npp64fc;
      pSrc2 : access constant nppdefs_h.Npp64fc;
      nLength : int;
      pNorm : access nppdefs_h.Npp64f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:2642
   pragma Import (C, nppsNormDiff_L2_64fc64f, "nppsNormDiff_L2_64fc64f");

  --* 
  -- * Device-buffer size (in bytes) for nppsNormDiff_L2_16s32s_Sfs.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsNormDiffL2GetBufferSize_16s32s_Sfs (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:2653
   pragma Import (C, nppsNormDiffL2GetBufferSize_16s32s_Sfs, "nppsNormDiffL2GetBufferSize_16s32s_Sfs");

  --* 
  -- * 16-bit signed short integer L2 norm method on two vectors' difference, return value is 32-bit signed integer.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pNorm Pointer to the norm result.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsNormDiffL2GetBufferSize_16s32s_Sfs to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsNormDiff_L2_16s32s_Sfs
     (pSrc1 : access nppdefs_h.Npp16s;
      pSrc2 : access nppdefs_h.Npp16s;
      nLength : int;
      pNorm : access nppdefs_h.Npp32s;
      nScaleFactor : int;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:2667
   pragma Import (C, nppsNormDiff_L2_16s32s_Sfs, "nppsNormDiff_L2_16s32s_Sfs");

  --* 
  -- * Device-buffer size (in bytes) for nppsNormDiff_L2Sqr_16s64s_Sfs.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsNormDiffL2SqrGetBufferSize_16s64s_Sfs (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:2678
   pragma Import (C, nppsNormDiffL2SqrGetBufferSize_16s64s_Sfs, "nppsNormDiffL2SqrGetBufferSize_16s64s_Sfs");

  --* 
  -- * 16-bit signed short integer L2 Square norm method on two vectors' difference, return value is 64-bit signed integer.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pNorm Pointer to the norm result.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsNormDiffL2SqrGetBufferSize_16s64s_Sfs to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsNormDiff_L2Sqr_16s64s_Sfs
     (pSrc1 : access nppdefs_h.Npp16s;
      pSrc2 : access nppdefs_h.Npp16s;
      nLength : int;
      pNorm : access nppdefs_h.Npp64s;
      nScaleFactor : int;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:2692
   pragma Import (C, nppsNormDiff_L2Sqr_16s64s_Sfs, "nppsNormDiff_L2Sqr_16s64s_Sfs");

  --* @} signal_l2_norm_diff  
  --* @defgroup signal_dot_product Dot Product
  -- *
  -- * @{
  -- *
  --  

  --* 
  -- * Device-buffer size (in bytes) for nppsDotProd_32f.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsDotProdGetBufferSize_32f (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:2711
   pragma Import (C, nppsDotProdGetBufferSize_32f, "nppsDotProdGetBufferSize_32f");

  --* 
  -- * 32-bit float dot product method, return value is 32-bit float.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pDp Pointer to the dot product result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsDotProdGetBufferSize_32f to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsDotProd_32f
     (pSrc1 : access nppdefs_h.Npp32f;
      pSrc2 : access nppdefs_h.Npp32f;
      nLength : int;
      pDp : access nppdefs_h.Npp32f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:2724
   pragma Import (C, nppsDotProd_32f, "nppsDotProd_32f");

  --* 
  -- * Device-buffer size (in bytes) for nppsDotProd_32fc.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsDotProdGetBufferSize_32fc (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:2735
   pragma Import (C, nppsDotProdGetBufferSize_32fc, "nppsDotProdGetBufferSize_32fc");

  --* 
  -- * 32-bit float complex dot product method, return value is 32-bit float complex.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pDp Pointer to the dot product result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsDotProdGetBufferSize_32fc to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsDotProd_32fc
     (pSrc1 : access constant nppdefs_h.Npp32fc;
      pSrc2 : access constant nppdefs_h.Npp32fc;
      nLength : int;
      pDp : access nppdefs_h.Npp32fc;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:2748
   pragma Import (C, nppsDotProd_32fc, "nppsDotProd_32fc");

  --* 
  -- * Device-buffer size (in bytes) for nppsDotProd_32f32fc.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsDotProdGetBufferSize_32f32fc (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:2759
   pragma Import (C, nppsDotProdGetBufferSize_32f32fc, "nppsDotProdGetBufferSize_32f32fc");

  --* 
  -- * 32-bit float and 32-bit float complex dot product method, return value is 32-bit float complex.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pDp Pointer to the dot product result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsDotProdGetBufferSize_32f32fc to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsDotProd_32f32fc
     (pSrc1 : access nppdefs_h.Npp32f;
      pSrc2 : access constant nppdefs_h.Npp32fc;
      nLength : int;
      pDp : access nppdefs_h.Npp32fc;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:2772
   pragma Import (C, nppsDotProd_32f32fc, "nppsDotProd_32f32fc");

  --* 
  -- * Device-buffer size (in bytes) for nppsDotProd_32f64f.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsDotProdGetBufferSize_32f64f (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:2783
   pragma Import (C, nppsDotProdGetBufferSize_32f64f, "nppsDotProdGetBufferSize_32f64f");

  --* 
  -- * 32-bit float dot product method, return value is 64-bit float.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pDp Pointer to the dot product result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsDotProdGetBufferSize_32f64f to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsDotProd_32f64f
     (pSrc1 : access nppdefs_h.Npp32f;
      pSrc2 : access nppdefs_h.Npp32f;
      nLength : int;
      pDp : access nppdefs_h.Npp64f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:2796
   pragma Import (C, nppsDotProd_32f64f, "nppsDotProd_32f64f");

  --* 
  -- * Device-buffer size (in bytes) for nppsDotProd_32fc64fc.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsDotProdGetBufferSize_32fc64fc (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:2807
   pragma Import (C, nppsDotProdGetBufferSize_32fc64fc, "nppsDotProdGetBufferSize_32fc64fc");

  --* 
  -- * 32-bit float complex dot product method, return value is 64-bit float complex.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pDp Pointer to the dot product result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsDotProdGetBufferSize_32fc64fc to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsDotProd_32fc64fc
     (pSrc1 : access constant nppdefs_h.Npp32fc;
      pSrc2 : access constant nppdefs_h.Npp32fc;
      nLength : int;
      pDp : access nppdefs_h.Npp64fc;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:2820
   pragma Import (C, nppsDotProd_32fc64fc, "nppsDotProd_32fc64fc");

  --* 
  -- * Device-buffer size (in bytes) for nppsDotProd_32f32fc64fc.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsDotProdGetBufferSize_32f32fc64fc (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:2831
   pragma Import (C, nppsDotProdGetBufferSize_32f32fc64fc, "nppsDotProdGetBufferSize_32f32fc64fc");

  --* 
  -- * 32-bit float and 32-bit float complex dot product method, return value is 64-bit float complex.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pDp Pointer to the dot product result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsDotProdGetBufferSize_32f32fc64fc to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsDotProd_32f32fc64fc
     (pSrc1 : access nppdefs_h.Npp32f;
      pSrc2 : access constant nppdefs_h.Npp32fc;
      nLength : int;
      pDp : access nppdefs_h.Npp64fc;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:2844
   pragma Import (C, nppsDotProd_32f32fc64fc, "nppsDotProd_32f32fc64fc");

  --* 
  -- * Device-buffer size (in bytes) for nppsDotProd_64f.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsDotProdGetBufferSize_64f (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:2855
   pragma Import (C, nppsDotProdGetBufferSize_64f, "nppsDotProdGetBufferSize_64f");

  --* 
  -- * 64-bit float dot product method, return value is 64-bit float.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pDp Pointer to the dot product result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsDotProdGetBufferSize_64f to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsDotProd_64f
     (pSrc1 : access nppdefs_h.Npp64f;
      pSrc2 : access nppdefs_h.Npp64f;
      nLength : int;
      pDp : access nppdefs_h.Npp64f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:2868
   pragma Import (C, nppsDotProd_64f, "nppsDotProd_64f");

  --* 
  -- * Device-buffer size (in bytes) for nppsDotProd_64fc.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsDotProdGetBufferSize_64fc (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:2879
   pragma Import (C, nppsDotProdGetBufferSize_64fc, "nppsDotProdGetBufferSize_64fc");

  --* 
  -- * 64-bit float complex dot product method, return value is 64-bit float complex.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pDp Pointer to the dot product result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsDotProdGetBufferSize_64fc to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsDotProd_64fc
     (pSrc1 : access constant nppdefs_h.Npp64fc;
      pSrc2 : access constant nppdefs_h.Npp64fc;
      nLength : int;
      pDp : access nppdefs_h.Npp64fc;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:2892
   pragma Import (C, nppsDotProd_64fc, "nppsDotProd_64fc");

  --* 
  -- * Device-buffer size (in bytes) for nppsDotProd_64f64fc.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsDotProdGetBufferSize_64f64fc (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:2903
   pragma Import (C, nppsDotProdGetBufferSize_64f64fc, "nppsDotProdGetBufferSize_64f64fc");

  --* 
  -- * 64-bit float and 64-bit float complex dot product method, return value is 64-bit float complex.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pDp Pointer to the dot product result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsDotProdGetBufferSize_64f64fc to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsDotProd_64f64fc
     (pSrc1 : access nppdefs_h.Npp64f;
      pSrc2 : access constant nppdefs_h.Npp64fc;
      nLength : int;
      pDp : access nppdefs_h.Npp64fc;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:2916
   pragma Import (C, nppsDotProd_64f64fc, "nppsDotProd_64f64fc");

  --* 
  -- * Device-buffer size (in bytes) for nppsDotProd_16s64s.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsDotProdGetBufferSize_16s64s (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:2927
   pragma Import (C, nppsDotProdGetBufferSize_16s64s, "nppsDotProdGetBufferSize_16s64s");

  --* 
  -- * 16-bit signed short integer dot product method, return value is 64-bit signed integer.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pDp Pointer to the dot product result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsDotProdGetBufferSize_16s64s to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsDotProd_16s64s
     (pSrc1 : access nppdefs_h.Npp16s;
      pSrc2 : access nppdefs_h.Npp16s;
      nLength : int;
      pDp : access nppdefs_h.Npp64s;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:2940
   pragma Import (C, nppsDotProd_16s64s, "nppsDotProd_16s64s");

  --* 
  -- * Device-buffer size (in bytes) for nppsDotProd_16sc64sc.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsDotProdGetBufferSize_16sc64sc (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:2951
   pragma Import (C, nppsDotProdGetBufferSize_16sc64sc, "nppsDotProdGetBufferSize_16sc64sc");

  --* 
  -- * 16-bit signed short integer complex dot product method, return value is 64-bit signed integer complex.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pDp Pointer to the dot product result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsDotProdGetBufferSize_16sc64sc to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsDotProd_16sc64sc
     (pSrc1 : access constant nppdefs_h.Npp16sc;
      pSrc2 : access constant nppdefs_h.Npp16sc;
      nLength : int;
      pDp : access nppdefs_h.Npp64sc;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:2964
   pragma Import (C, nppsDotProd_16sc64sc, "nppsDotProd_16sc64sc");

  --* 
  -- * Device-buffer size (in bytes) for nppsDotProd_16s16sc64sc.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsDotProdGetBufferSize_16s16sc64sc (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:2975
   pragma Import (C, nppsDotProdGetBufferSize_16s16sc64sc, "nppsDotProdGetBufferSize_16s16sc64sc");

  --* 
  -- * 16-bit signed short integer and 16-bit signed short integer short dot product method, return value is 64-bit signed integer complex.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pDp Pointer to the dot product result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsDotProdGetBufferSize_16s16sc64sc to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsDotProd_16s16sc64sc
     (pSrc1 : access nppdefs_h.Npp16s;
      pSrc2 : access constant nppdefs_h.Npp16sc;
      nLength : int;
      pDp : access nppdefs_h.Npp64sc;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:2988
   pragma Import (C, nppsDotProd_16s16sc64sc, "nppsDotProd_16s16sc64sc");

  --* 
  -- * Device-buffer size (in bytes) for nppsDotProd_16s32f.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsDotProdGetBufferSize_16s32f (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:2999
   pragma Import (C, nppsDotProdGetBufferSize_16s32f, "nppsDotProdGetBufferSize_16s32f");

  --* 
  -- * 16-bit signed short integer dot product method, return value is 32-bit float.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pDp Pointer to the dot product result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsDotProdGetBufferSize_16s32f to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsDotProd_16s32f
     (pSrc1 : access nppdefs_h.Npp16s;
      pSrc2 : access nppdefs_h.Npp16s;
      nLength : int;
      pDp : access nppdefs_h.Npp32f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:3012
   pragma Import (C, nppsDotProd_16s32f, "nppsDotProd_16s32f");

  --* 
  -- * Device-buffer size (in bytes) for nppsDotProd_16sc32fc.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsDotProdGetBufferSize_16sc32fc (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:3023
   pragma Import (C, nppsDotProdGetBufferSize_16sc32fc, "nppsDotProdGetBufferSize_16sc32fc");

  --* 
  -- * 16-bit signed short integer complex dot product method, return value is 32-bit float complex.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pDp Pointer to the dot product result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsDotProdGetBufferSize_16sc32fc to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsDotProd_16sc32fc
     (pSrc1 : access constant nppdefs_h.Npp16sc;
      pSrc2 : access constant nppdefs_h.Npp16sc;
      nLength : int;
      pDp : access nppdefs_h.Npp32fc;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:3036
   pragma Import (C, nppsDotProd_16sc32fc, "nppsDotProd_16sc32fc");

  --* 
  -- * Device-buffer size (in bytes) for nppsDotProd_16s16sc32fc.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsDotProdGetBufferSize_16s16sc32fc (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:3047
   pragma Import (C, nppsDotProdGetBufferSize_16s16sc32fc, "nppsDotProdGetBufferSize_16s16sc32fc");

  --* 
  -- * 16-bit signed short integer and 16-bit signed short integer complex dot product method, return value is 32-bit float complex.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pDp Pointer to the dot product result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsDotProdGetBufferSize_16s16sc32fc to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsDotProd_16s16sc32fc
     (pSrc1 : access nppdefs_h.Npp16s;
      pSrc2 : access constant nppdefs_h.Npp16sc;
      nLength : int;
      pDp : access nppdefs_h.Npp32fc;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:3060
   pragma Import (C, nppsDotProd_16s16sc32fc, "nppsDotProd_16s16sc32fc");

  --* 
  -- * Device-buffer size (in bytes) for nppsDotProd_16s_Sfs.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsDotProdGetBufferSize_16s_Sfs (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:3071
   pragma Import (C, nppsDotProdGetBufferSize_16s_Sfs, "nppsDotProdGetBufferSize_16s_Sfs");

  --* 
  -- * 16-bit signed short integer dot product method, return value is 16-bit signed short integer.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pDp Pointer to the dot product result.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsDotProdGetBufferSize_16s_Sfs to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsDotProd_16s_Sfs
     (pSrc1 : access nppdefs_h.Npp16s;
      pSrc2 : access nppdefs_h.Npp16s;
      nLength : int;
      pDp : access nppdefs_h.Npp16s;
      nScaleFactor : int;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:3085
   pragma Import (C, nppsDotProd_16s_Sfs, "nppsDotProd_16s_Sfs");

  --* 
  -- * Device-buffer size (in bytes) for nppsDotProd_16sc_Sfs.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsDotProdGetBufferSize_16sc_Sfs (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:3096
   pragma Import (C, nppsDotProdGetBufferSize_16sc_Sfs, "nppsDotProdGetBufferSize_16sc_Sfs");

  --* 
  -- * 16-bit signed short integer complex dot product method, return value is 16-bit signed short integer complex.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pDp Pointer to the dot product result.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsDotProdGetBufferSize_16sc_Sfs to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsDotProd_16sc_Sfs
     (pSrc1 : access constant nppdefs_h.Npp16sc;
      pSrc2 : access constant nppdefs_h.Npp16sc;
      nLength : int;
      pDp : access nppdefs_h.Npp16sc;
      nScaleFactor : int;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:3110
   pragma Import (C, nppsDotProd_16sc_Sfs, "nppsDotProd_16sc_Sfs");

  --* 
  -- * Device-buffer size (in bytes) for nppsDotProd_32s_Sfs.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsDotProdGetBufferSize_32s_Sfs (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:3121
   pragma Import (C, nppsDotProdGetBufferSize_32s_Sfs, "nppsDotProdGetBufferSize_32s_Sfs");

  --* 
  -- * 32-bit signed integer dot product method, return value is 32-bit signed integer.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pDp Pointer to the dot product result.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsDotProdGetBufferSize_32s_Sfs to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsDotProd_32s_Sfs
     (pSrc1 : access nppdefs_h.Npp32s;
      pSrc2 : access nppdefs_h.Npp32s;
      nLength : int;
      pDp : access nppdefs_h.Npp32s;
      nScaleFactor : int;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:3135
   pragma Import (C, nppsDotProd_32s_Sfs, "nppsDotProd_32s_Sfs");

  --* 
  -- * Device-buffer size (in bytes) for nppsDotProd_32sc_Sfs.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsDotProdGetBufferSize_32sc_Sfs (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:3146
   pragma Import (C, nppsDotProdGetBufferSize_32sc_Sfs, "nppsDotProdGetBufferSize_32sc_Sfs");

  --* 
  -- * 32-bit signed integer complex dot product method, return value is 32-bit signed integer complex.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pDp Pointer to the dot product result.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsDotProdGetBufferSize_32sc_Sfs to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsDotProd_32sc_Sfs
     (pSrc1 : access constant nppdefs_h.Npp32sc;
      pSrc2 : access constant nppdefs_h.Npp32sc;
      nLength : int;
      pDp : access nppdefs_h.Npp32sc;
      nScaleFactor : int;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:3160
   pragma Import (C, nppsDotProd_32sc_Sfs, "nppsDotProd_32sc_Sfs");

  --* 
  -- * Device-buffer size (in bytes) for nppsDotProd_16s32s_Sfs.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsDotProdGetBufferSize_16s32s_Sfs (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:3171
   pragma Import (C, nppsDotProdGetBufferSize_16s32s_Sfs, "nppsDotProdGetBufferSize_16s32s_Sfs");

  --* 
  -- * 16-bit signed short integer dot product method, return value is 32-bit signed integer.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pDp Pointer to the dot product result. 
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsDotProdGetBufferSize_16s32s_Sfs to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsDotProd_16s32s_Sfs
     (pSrc1 : access nppdefs_h.Npp16s;
      pSrc2 : access nppdefs_h.Npp16s;
      nLength : int;
      pDp : access nppdefs_h.Npp32s;
      nScaleFactor : int;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:3185
   pragma Import (C, nppsDotProd_16s32s_Sfs, "nppsDotProd_16s32s_Sfs");

  --* 
  -- * Device-buffer size (in bytes) for nppsDotProd_16s16sc32sc_Sfs.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsDotProdGetBufferSize_16s16sc32sc_Sfs (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:3196
   pragma Import (C, nppsDotProdGetBufferSize_16s16sc32sc_Sfs, "nppsDotProdGetBufferSize_16s16sc32sc_Sfs");

  --* 
  -- * 16-bit signed short integer and 16-bit signed short integer complex dot product method, return value is 32-bit signed integer complex.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pDp Pointer to the dot product result.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsDotProdGetBufferSize_16s16sc32sc_Sfs to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsDotProd_16s16sc32sc_Sfs
     (pSrc1 : access nppdefs_h.Npp16s;
      pSrc2 : access constant nppdefs_h.Npp16sc;
      nLength : int;
      pDp : access nppdefs_h.Npp32sc;
      nScaleFactor : int;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:3210
   pragma Import (C, nppsDotProd_16s16sc32sc_Sfs, "nppsDotProd_16s16sc32sc_Sfs");

  --* 
  -- * Device-buffer size (in bytes) for nppsDotProd_16s32s32s_Sfs.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsDotProdGetBufferSize_16s32s32s_Sfs (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:3221
   pragma Import (C, nppsDotProdGetBufferSize_16s32s32s_Sfs, "nppsDotProdGetBufferSize_16s32s32s_Sfs");

  --* 
  -- * 16-bit signed short integer and 32-bit signed integer dot product method, return value is 32-bit signed integer.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pDp Pointer to the dot product result.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsDotProdGetBufferSize_16s32s32s_Sfs to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsDotProd_16s32s32s_Sfs
     (pSrc1 : access nppdefs_h.Npp16s;
      pSrc2 : access nppdefs_h.Npp32s;
      nLength : int;
      pDp : access nppdefs_h.Npp32s;
      nScaleFactor : int;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:3235
   pragma Import (C, nppsDotProd_16s32s32s_Sfs, "nppsDotProd_16s32s32s_Sfs");

  --* 
  -- * Device-buffer size (in bytes) for nppsDotProd_16s16sc_Sfs.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsDotProdGetBufferSize_16s16sc_Sfs (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:3246
   pragma Import (C, nppsDotProdGetBufferSize_16s16sc_Sfs, "nppsDotProdGetBufferSize_16s16sc_Sfs");

  --* 
  -- * 16-bit signed short integer and 16-bit signed short integer complex dot product method, return value is 16-bit signed short integer complex.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pDp Pointer to the dot product result.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsDotProdGetBufferSize_16s16sc_Sfs to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsDotProd_16s16sc_Sfs
     (pSrc1 : access nppdefs_h.Npp16s;
      pSrc2 : access constant nppdefs_h.Npp16sc;
      nLength : int;
      pDp : access nppdefs_h.Npp16sc;
      nScaleFactor : int;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:3260
   pragma Import (C, nppsDotProd_16s16sc_Sfs, "nppsDotProd_16s16sc_Sfs");

  --* 
  -- * Device-buffer size (in bytes) for nppsDotProd_16sc32sc_Sfs.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsDotProdGetBufferSize_16sc32sc_Sfs (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:3271
   pragma Import (C, nppsDotProdGetBufferSize_16sc32sc_Sfs, "nppsDotProdGetBufferSize_16sc32sc_Sfs");

  --* 
  -- * 16-bit signed short integer complex dot product method, return value is 32-bit signed integer complex.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pDp Pointer to the dot product result.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsDotProdGetBufferSize_16sc32sc_Sfs to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsDotProd_16sc32sc_Sfs
     (pSrc1 : access constant nppdefs_h.Npp16sc;
      pSrc2 : access constant nppdefs_h.Npp16sc;
      nLength : int;
      pDp : access nppdefs_h.Npp32sc;
      nScaleFactor : int;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:3285
   pragma Import (C, nppsDotProd_16sc32sc_Sfs, "nppsDotProd_16sc32sc_Sfs");

  --* 
  -- * Device-buffer size (in bytes) for nppsDotProd_32s32sc_Sfs.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsDotProdGetBufferSize_32s32sc_Sfs (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:3296
   pragma Import (C, nppsDotProdGetBufferSize_32s32sc_Sfs, "nppsDotProdGetBufferSize_32s32sc_Sfs");

  --* 
  -- * 32-bit signed short integer and 32-bit signed short integer complex dot product method, return value is 32-bit signed integer complex.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pDp Pointer to the dot product result.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsDotProdGetBufferSize_32s32sc_Sfs to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsDotProd_32s32sc_Sfs
     (pSrc1 : access nppdefs_h.Npp32s;
      pSrc2 : access constant nppdefs_h.Npp32sc;
      nLength : int;
      pDp : access nppdefs_h.Npp32sc;
      nScaleFactor : int;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:3310
   pragma Import (C, nppsDotProd_32s32sc_Sfs, "nppsDotProd_32s32sc_Sfs");

  --* @} signal_dot_product  
  --* @defgroup signal_count_in_range Count In Range
  -- *
  -- * @{
  -- *
  --  

  --* 
  -- * Device-buffer size (in bytes) for nppsCountInRange_32s.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsCountInRangeGetBufferSize_32s (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:3329
   pragma Import (C, nppsCountInRangeGetBufferSize_32s, "nppsCountInRangeGetBufferSize_32s");

  --* 
  -- * Computes the number of elements whose values fall into the specified range on a 32-bit signed integer array.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pCounts Pointer to the number of elements.
  -- * \param nLowerBound Lower bound of the specified range.
  -- * \param nUpperBound Upper bound of the specified range.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsCountInRangeGetBufferSize_32s to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsCountInRange_32s
     (pSrc : access nppdefs_h.Npp32s;
      nLength : int;
      pCounts : access int;
      nLowerBound : nppdefs_h.Npp32s;
      nUpperBound : nppdefs_h.Npp32s;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:3343
   pragma Import (C, nppsCountInRange_32s, "nppsCountInRange_32s");

  --* @} signal_count_in_range  
  --* @defgroup signal_count_zero_crossings Count Zero Crossings
  -- *
  -- * @{
  -- *
  --  

  --* 
  -- * Device-buffer size (in bytes) for nppsZeroCrossing_16s32f.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsZeroCrossingGetBufferSize_16s32f (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:3362
   pragma Import (C, nppsZeroCrossingGetBufferSize_16s32f, "nppsZeroCrossingGetBufferSize_16s32f");

  --* 
  -- * 16-bit signed short integer zero crossing method, return value is 32-bit floating point.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pValZC Pointer to the output result.
  -- * \param tZCType Type of the zero crossing measure: nppZCR, nppZCXor or nppZCC.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsZeroCrossingGetBufferSize_16s32f to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsZeroCrossing_16s32f
     (pSrc : access nppdefs_h.Npp16s;
      nLength : int;
      pValZC : access nppdefs_h.Npp32f;
      tZCType : nppdefs_h.NppsZCType;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:3375
   pragma Import (C, nppsZeroCrossing_16s32f, "nppsZeroCrossing_16s32f");

  --* 
  -- * Device-buffer size (in bytes) for nppsZeroCrossing_32f.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsZeroCrossingGetBufferSize_32f (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:3386
   pragma Import (C, nppsZeroCrossingGetBufferSize_32f, "nppsZeroCrossingGetBufferSize_32f");

  --* 
  -- * 32-bit floating-point zero crossing method, return value is 32-bit floating point.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pValZC Pointer to the output result.
  -- * \param tZCType Type of the zero crossing measure: nppZCR, nppZCXor or nppZCC.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsZeroCrossingGetBufferSize_32f to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsZeroCrossing_32f
     (pSrc : access nppdefs_h.Npp32f;
      nLength : int;
      pValZC : access nppdefs_h.Npp32f;
      tZCType : nppdefs_h.NppsZCType;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:3399
   pragma Import (C, nppsZeroCrossing_32f, "nppsZeroCrossing_32f");

  --* @} signal_count_zero_crossings  
  --* @defgroup signal_maximum_error MaximumError
  -- * Primitives for computing the maximum error between two signals.
  -- * Given two signals \f$pSrc1\f$ and \f$pSrc2\f$ both with length \f$N\f$, 
  -- * the maximum error is defined as the largest absolute difference between the corresponding
  -- * elements of two signals.
  -- *
  -- * If the signal is in complex format, the absolute value of the complex number is used.
  -- * @{
  -- *
  --  

  --* 
  -- * 8-bit unsigned char maximum method.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pDst Pointer to the error result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsMaximumErrorGetBufferSize_8u to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMaximumError_8u
     (pSrc1 : access nppdefs_h.Npp8u;
      pSrc2 : access nppdefs_h.Npp8u;
      nLength : int;
      pDst : access nppdefs_h.Npp64f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:3425
   pragma Import (C, nppsMaximumError_8u, "nppsMaximumError_8u");

  --* 
  -- * 8-bit signed char maximum method.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pDst Pointer to the error result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsMaximumErrorGetBufferSize_8s to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMaximumError_8s
     (pSrc1 : access nppdefs_h.Npp8s;
      pSrc2 : access nppdefs_h.Npp8s;
      nLength : int;
      pDst : access nppdefs_h.Npp64f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:3438
   pragma Import (C, nppsMaximumError_8s, "nppsMaximumError_8s");

  --* 
  -- * 16-bit unsigned short integer maximum method.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pDst Pointer to the error result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsMaximumErrorGetBufferSize_16u to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMaximumError_16u
     (pSrc1 : access nppdefs_h.Npp16u;
      pSrc2 : access nppdefs_h.Npp16u;
      nLength : int;
      pDst : access nppdefs_h.Npp64f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:3451
   pragma Import (C, nppsMaximumError_16u, "nppsMaximumError_16u");

  --* 
  -- * 16-bit signed short integer maximum method.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pDst Pointer to the error result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsMaximumErrorGetBufferSize_16s to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMaximumError_16s
     (pSrc1 : access nppdefs_h.Npp16s;
      pSrc2 : access nppdefs_h.Npp16s;
      nLength : int;
      pDst : access nppdefs_h.Npp64f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:3464
   pragma Import (C, nppsMaximumError_16s, "nppsMaximumError_16s");

  --* 
  -- * 16-bit unsigned short complex integer maximum method.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pDst Pointer to the error result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsMaximumErrorGetBufferSize_16sc to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMaximumError_16sc
     (pSrc1 : access constant nppdefs_h.Npp16sc;
      pSrc2 : access constant nppdefs_h.Npp16sc;
      nLength : int;
      pDst : access nppdefs_h.Npp64f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:3477
   pragma Import (C, nppsMaximumError_16sc, "nppsMaximumError_16sc");

  --* 
  -- * 32-bit unsigned short integer maximum method.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pDst Pointer to the error result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsMaximumErrorGetBufferSize_32u to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMaximumError_32u
     (pSrc1 : access nppdefs_h.Npp32u;
      pSrc2 : access nppdefs_h.Npp32u;
      nLength : int;
      pDst : access nppdefs_h.Npp64f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:3490
   pragma Import (C, nppsMaximumError_32u, "nppsMaximumError_32u");

  --* 
  -- * 32-bit signed short integer maximum method.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pDst Pointer to the error result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsMaximumErrorGetBufferSize_32s to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMaximumError_32s
     (pSrc1 : access nppdefs_h.Npp32s;
      pSrc2 : access nppdefs_h.Npp32s;
      nLength : int;
      pDst : access nppdefs_h.Npp64f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:3503
   pragma Import (C, nppsMaximumError_32s, "nppsMaximumError_32s");

  --* 
  -- * 32-bit unsigned short complex integer maximum method.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pDst Pointer to the error result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsMaximumErrorGetBufferSize_32sc to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMaximumError_32sc
     (pSrc1 : access constant nppdefs_h.Npp32sc;
      pSrc2 : access constant nppdefs_h.Npp32sc;
      nLength : int;
      pDst : access nppdefs_h.Npp64f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:3516
   pragma Import (C, nppsMaximumError_32sc, "nppsMaximumError_32sc");

  --* 
  -- * 64-bit signed short integer maximum method.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pDst Pointer to the error result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsMaximumErrorGetBufferSize_64s to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMaximumError_64s
     (pSrc1 : access nppdefs_h.Npp64s;
      pSrc2 : access nppdefs_h.Npp64s;
      nLength : int;
      pDst : access nppdefs_h.Npp64f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:3529
   pragma Import (C, nppsMaximumError_64s, "nppsMaximumError_64s");

  --* 
  -- * 64-bit unsigned short complex integer maximum method.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pDst Pointer to the error result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsMaximumErrorGetBufferSize_64sc to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMaximumError_64sc
     (pSrc1 : access constant nppdefs_h.Npp64sc;
      pSrc2 : access constant nppdefs_h.Npp64sc;
      nLength : int;
      pDst : access nppdefs_h.Npp64f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:3542
   pragma Import (C, nppsMaximumError_64sc, "nppsMaximumError_64sc");

  --* 
  -- * 32-bit floating point maximum method.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pDst Pointer to the error result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsMaximumErrorGetBufferSize_32f to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMaximumError_32f
     (pSrc1 : access nppdefs_h.Npp32f;
      pSrc2 : access nppdefs_h.Npp32f;
      nLength : int;
      pDst : access nppdefs_h.Npp64f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:3555
   pragma Import (C, nppsMaximumError_32f, "nppsMaximumError_32f");

  --* 
  -- * 32-bit floating point complex maximum method.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pDst Pointer to the error result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsMaximumErrorGetBufferSize_32fc to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMaximumError_32fc
     (pSrc1 : access constant nppdefs_h.Npp32fc;
      pSrc2 : access constant nppdefs_h.Npp32fc;
      nLength : int;
      pDst : access nppdefs_h.Npp64f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:3568
   pragma Import (C, nppsMaximumError_32fc, "nppsMaximumError_32fc");

  --* 
  -- * 64-bit floating point maximum method.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pDst Pointer to the error result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsMaximumErrorGetBufferSize_64f to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMaximumError_64f
     (pSrc1 : access nppdefs_h.Npp64f;
      pSrc2 : access nppdefs_h.Npp64f;
      nLength : int;
      pDst : access nppdefs_h.Npp64f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:3581
   pragma Import (C, nppsMaximumError_64f, "nppsMaximumError_64f");

  --* 
  -- * 64-bit floating point complex maximum method.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pDst Pointer to the error result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsMaximumErrorGetBufferSize_64fc to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMaximumError_64fc
     (pSrc1 : access constant nppdefs_h.Npp64fc;
      pSrc2 : access constant nppdefs_h.Npp64fc;
      nLength : int;
      pDst : access nppdefs_h.Npp64f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:3594
   pragma Import (C, nppsMaximumError_64fc, "nppsMaximumError_64fc");

  --* 
  -- * Device-buffer size (in bytes) for nppsMaximumError_8u.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsMaximumErrorGetBufferSize_8u (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:3604
   pragma Import (C, nppsMaximumErrorGetBufferSize_8u, "nppsMaximumErrorGetBufferSize_8u");

  --* 
  -- * Device-buffer size (in bytes) for nppsMaximumError_8s.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsMaximumErrorGetBufferSize_8s (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:3614
   pragma Import (C, nppsMaximumErrorGetBufferSize_8s, "nppsMaximumErrorGetBufferSize_8s");

  --* 
  -- * Device-buffer size (in bytes) for nppsMaximumError_16u.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsMaximumErrorGetBufferSize_16u (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:3624
   pragma Import (C, nppsMaximumErrorGetBufferSize_16u, "nppsMaximumErrorGetBufferSize_16u");

  --* 
  -- * Device-buffer size (in bytes) for nppsMaximumError_16s.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsMaximumErrorGetBufferSize_16s (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:3634
   pragma Import (C, nppsMaximumErrorGetBufferSize_16s, "nppsMaximumErrorGetBufferSize_16s");

  --* 
  -- * Device-buffer size (in bytes) for nppsMaximumError_16sc.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsMaximumErrorGetBufferSize_16sc (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:3644
   pragma Import (C, nppsMaximumErrorGetBufferSize_16sc, "nppsMaximumErrorGetBufferSize_16sc");

  --* 
  -- * Device-buffer size (in bytes) for nppsMaximumError_32u.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsMaximumErrorGetBufferSize_32u (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:3654
   pragma Import (C, nppsMaximumErrorGetBufferSize_32u, "nppsMaximumErrorGetBufferSize_32u");

  --* 
  -- * Device-buffer size (in bytes) for nppsMaximumError_32s.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsMaximumErrorGetBufferSize_32s (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:3664
   pragma Import (C, nppsMaximumErrorGetBufferSize_32s, "nppsMaximumErrorGetBufferSize_32s");

  --* 
  -- * Device-buffer size (in bytes) for nppsMaximumError_32sc.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsMaximumErrorGetBufferSize_32sc (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:3674
   pragma Import (C, nppsMaximumErrorGetBufferSize_32sc, "nppsMaximumErrorGetBufferSize_32sc");

  --* 
  -- * Device-buffer size (in bytes) for nppsMaximumError_64s.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsMaximumErrorGetBufferSize_64s (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:3684
   pragma Import (C, nppsMaximumErrorGetBufferSize_64s, "nppsMaximumErrorGetBufferSize_64s");

  --* 
  -- * Device-buffer size (in bytes) for nppsMaximumError_64sc.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsMaximumErrorGetBufferSize_64sc (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:3694
   pragma Import (C, nppsMaximumErrorGetBufferSize_64sc, "nppsMaximumErrorGetBufferSize_64sc");

  --* 
  -- * Device-buffer size (in bytes) for nppsMaximumError_32f.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsMaximumErrorGetBufferSize_32f (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:3704
   pragma Import (C, nppsMaximumErrorGetBufferSize_32f, "nppsMaximumErrorGetBufferSize_32f");

  --* 
  -- * Device-buffer size (in bytes) for nppsMaximumError_32fc.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsMaximumErrorGetBufferSize_32fc (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:3714
   pragma Import (C, nppsMaximumErrorGetBufferSize_32fc, "nppsMaximumErrorGetBufferSize_32fc");

  --* 
  -- * Device-buffer size (in bytes) for nppsMaximumError_64f.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsMaximumErrorGetBufferSize_64f (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:3724
   pragma Import (C, nppsMaximumErrorGetBufferSize_64f, "nppsMaximumErrorGetBufferSize_64f");

  --* 
  -- * Device-buffer size (in bytes) for nppsMaximumError_64fc.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsMaximumErrorGetBufferSize_64fc (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:3734
   pragma Import (C, nppsMaximumErrorGetBufferSize_64fc, "nppsMaximumErrorGetBufferSize_64fc");

  --* @}  
  --* @defgroup signal_average_error AverageError
  -- * Primitives for computing the Average error between two signals.
  -- * Given two signals \f$pSrc1\f$ and \f$pSrc2\f$ both with length \f$N\f$, 
  -- * the average error is defined as
  -- * \f[Average Error = \frac{1}{N}\sum_{n=0}^{N-1}\left|pSrc1(n) - pSrc2(n)\right|\f]
  -- *
  -- * If the signal is in complex format, the absolute value of the complex number is used.
  -- * @{
  -- *
  --  

  --* 
  -- * 8-bit unsigned char Average method.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pDst Pointer to the error result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsAverageErrorGetBufferSize_8u to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAverageError_8u
     (pSrc1 : access nppdefs_h.Npp8u;
      pSrc2 : access nppdefs_h.Npp8u;
      nLength : int;
      pDst : access nppdefs_h.Npp64f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:3758
   pragma Import (C, nppsAverageError_8u, "nppsAverageError_8u");

  --* 
  -- * 8-bit signed char Average method.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pDst Pointer to the error result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsAverageErrorGetBufferSize_8s to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAverageError_8s
     (pSrc1 : access nppdefs_h.Npp8s;
      pSrc2 : access nppdefs_h.Npp8s;
      nLength : int;
      pDst : access nppdefs_h.Npp64f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:3771
   pragma Import (C, nppsAverageError_8s, "nppsAverageError_8s");

  --* 
  -- * 16-bit unsigned short integer Average method.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pDst Pointer to the error result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsAverageErrorGetBufferSize_16u to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAverageError_16u
     (pSrc1 : access nppdefs_h.Npp16u;
      pSrc2 : access nppdefs_h.Npp16u;
      nLength : int;
      pDst : access nppdefs_h.Npp64f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:3784
   pragma Import (C, nppsAverageError_16u, "nppsAverageError_16u");

  --* 
  -- * 16-bit signed short integer Average method.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pDst Pointer to the error result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsAverageErrorGetBufferSize_16s to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAverageError_16s
     (pSrc1 : access nppdefs_h.Npp16s;
      pSrc2 : access nppdefs_h.Npp16s;
      nLength : int;
      pDst : access nppdefs_h.Npp64f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:3797
   pragma Import (C, nppsAverageError_16s, "nppsAverageError_16s");

  --* 
  -- * 16-bit unsigned short complex integer Average method.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pDst Pointer to the error result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsAverageErrorGetBufferSize_16sc to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAverageError_16sc
     (pSrc1 : access constant nppdefs_h.Npp16sc;
      pSrc2 : access constant nppdefs_h.Npp16sc;
      nLength : int;
      pDst : access nppdefs_h.Npp64f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:3810
   pragma Import (C, nppsAverageError_16sc, "nppsAverageError_16sc");

  --* 
  -- * 32-bit unsigned short integer Average method.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pDst Pointer to the error result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsAverageErrorGetBufferSize_32u to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAverageError_32u
     (pSrc1 : access nppdefs_h.Npp32u;
      pSrc2 : access nppdefs_h.Npp32u;
      nLength : int;
      pDst : access nppdefs_h.Npp64f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:3823
   pragma Import (C, nppsAverageError_32u, "nppsAverageError_32u");

  --* 
  -- * 32-bit signed short integer Average method.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pDst Pointer to the error result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsAverageErrorGetBufferSize_32s to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAverageError_32s
     (pSrc1 : access nppdefs_h.Npp32s;
      pSrc2 : access nppdefs_h.Npp32s;
      nLength : int;
      pDst : access nppdefs_h.Npp64f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:3836
   pragma Import (C, nppsAverageError_32s, "nppsAverageError_32s");

  --* 
  -- * 32-bit unsigned short complex integer Average method.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pDst Pointer to the error result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsAverageErrorGetBufferSize_32sc to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAverageError_32sc
     (pSrc1 : access constant nppdefs_h.Npp32sc;
      pSrc2 : access constant nppdefs_h.Npp32sc;
      nLength : int;
      pDst : access nppdefs_h.Npp64f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:3849
   pragma Import (C, nppsAverageError_32sc, "nppsAverageError_32sc");

  --* 
  -- * 64-bit signed short integer Average method.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pDst Pointer to the error result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsAverageErrorGetBufferSize_64s to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAverageError_64s
     (pSrc1 : access nppdefs_h.Npp64s;
      pSrc2 : access nppdefs_h.Npp64s;
      nLength : int;
      pDst : access nppdefs_h.Npp64f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:3862
   pragma Import (C, nppsAverageError_64s, "nppsAverageError_64s");

  --* 
  -- * 64-bit unsigned short complex integer Average method.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pDst Pointer to the error result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsAverageErrorGetBufferSize_64sc to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAverageError_64sc
     (pSrc1 : access constant nppdefs_h.Npp64sc;
      pSrc2 : access constant nppdefs_h.Npp64sc;
      nLength : int;
      pDst : access nppdefs_h.Npp64f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:3875
   pragma Import (C, nppsAverageError_64sc, "nppsAverageError_64sc");

  --* 
  -- * 32-bit floating point Average method.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pDst Pointer to the error result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsAverageErrorGetBufferSize_32f to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAverageError_32f
     (pSrc1 : access nppdefs_h.Npp32f;
      pSrc2 : access nppdefs_h.Npp32f;
      nLength : int;
      pDst : access nppdefs_h.Npp64f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:3888
   pragma Import (C, nppsAverageError_32f, "nppsAverageError_32f");

  --* 
  -- * 32-bit floating point complex Average method.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pDst Pointer to the error result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsAverageErrorGetBufferSize_32fc to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAverageError_32fc
     (pSrc1 : access constant nppdefs_h.Npp32fc;
      pSrc2 : access constant nppdefs_h.Npp32fc;
      nLength : int;
      pDst : access nppdefs_h.Npp64f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:3901
   pragma Import (C, nppsAverageError_32fc, "nppsAverageError_32fc");

  --* 
  -- * 64-bit floating point Average method.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pDst Pointer to the error result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsAverageErrorGetBufferSize_64f to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAverageError_64f
     (pSrc1 : access nppdefs_h.Npp64f;
      pSrc2 : access nppdefs_h.Npp64f;
      nLength : int;
      pDst : access nppdefs_h.Npp64f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:3914
   pragma Import (C, nppsAverageError_64f, "nppsAverageError_64f");

  --* 
  -- * 64-bit floating point complex Average method.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pDst Pointer to the error result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsAverageErrorGetBufferSize_64fc to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAverageError_64fc
     (pSrc1 : access constant nppdefs_h.Npp64fc;
      pSrc2 : access constant nppdefs_h.Npp64fc;
      nLength : int;
      pDst : access nppdefs_h.Npp64f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:3927
   pragma Import (C, nppsAverageError_64fc, "nppsAverageError_64fc");

  --* 
  -- * Device-buffer size (in bytes) for nppsAverageError_8u.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsAverageErrorGetBufferSize_8u (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:3937
   pragma Import (C, nppsAverageErrorGetBufferSize_8u, "nppsAverageErrorGetBufferSize_8u");

  --* 
  -- * Device-buffer size (in bytes) for nppsAverageError_8s.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsAverageErrorGetBufferSize_8s (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:3947
   pragma Import (C, nppsAverageErrorGetBufferSize_8s, "nppsAverageErrorGetBufferSize_8s");

  --* 
  -- * Device-buffer size (in bytes) for nppsAverageError_16u.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsAverageErrorGetBufferSize_16u (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:3957
   pragma Import (C, nppsAverageErrorGetBufferSize_16u, "nppsAverageErrorGetBufferSize_16u");

  --* 
  -- * Device-buffer size (in bytes) for nppsAverageError_16s.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsAverageErrorGetBufferSize_16s (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:3967
   pragma Import (C, nppsAverageErrorGetBufferSize_16s, "nppsAverageErrorGetBufferSize_16s");

  --* 
  -- * Device-buffer size (in bytes) for nppsAverageError_16sc.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsAverageErrorGetBufferSize_16sc (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:3977
   pragma Import (C, nppsAverageErrorGetBufferSize_16sc, "nppsAverageErrorGetBufferSize_16sc");

  --* 
  -- * Device-buffer size (in bytes) for nppsAverageError_32u.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsAverageErrorGetBufferSize_32u (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:3987
   pragma Import (C, nppsAverageErrorGetBufferSize_32u, "nppsAverageErrorGetBufferSize_32u");

  --* 
  -- * Device-buffer size (in bytes) for nppsAverageError_32s.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsAverageErrorGetBufferSize_32s (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:3997
   pragma Import (C, nppsAverageErrorGetBufferSize_32s, "nppsAverageErrorGetBufferSize_32s");

  --* 
  -- * Device-buffer size (in bytes) for nppsAverageError_32sc.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsAverageErrorGetBufferSize_32sc (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:4007
   pragma Import (C, nppsAverageErrorGetBufferSize_32sc, "nppsAverageErrorGetBufferSize_32sc");

  --* 
  -- * Device-buffer size (in bytes) for nppsAverageError_64s.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsAverageErrorGetBufferSize_64s (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:4017
   pragma Import (C, nppsAverageErrorGetBufferSize_64s, "nppsAverageErrorGetBufferSize_64s");

  --* 
  -- * Device-buffer size (in bytes) for nppsAverageError_64sc.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsAverageErrorGetBufferSize_64sc (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:4027
   pragma Import (C, nppsAverageErrorGetBufferSize_64sc, "nppsAverageErrorGetBufferSize_64sc");

  --* 
  -- * Device-buffer size (in bytes) for nppsAverageError_32f.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsAverageErrorGetBufferSize_32f (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:4037
   pragma Import (C, nppsAverageErrorGetBufferSize_32f, "nppsAverageErrorGetBufferSize_32f");

  --* 
  -- * Device-buffer size (in bytes) for nppsAverageError_32fc.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsAverageErrorGetBufferSize_32fc (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:4047
   pragma Import (C, nppsAverageErrorGetBufferSize_32fc, "nppsAverageErrorGetBufferSize_32fc");

  --* 
  -- * Device-buffer size (in bytes) for nppsAverageError_64f.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsAverageErrorGetBufferSize_64f (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:4057
   pragma Import (C, nppsAverageErrorGetBufferSize_64f, "nppsAverageErrorGetBufferSize_64f");

  --* 
  -- * Device-buffer size (in bytes) for nppsAverageError_64fc.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsAverageErrorGetBufferSize_64fc (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:4067
   pragma Import (C, nppsAverageErrorGetBufferSize_64fc, "nppsAverageErrorGetBufferSize_64fc");

  --* @}  
  --* @defgroup signal_maximum_relative_error MaximumRelativeError
  -- * Primitives for computing the MaximumRelative error between two signals.
  -- * Given two signals \f$pSrc1\f$ and \f$pSrc2\f$ both with length \f$N\f$, 
  -- * the maximum relative error is defined as
  -- * \f[MaximumRelativeError = max{\frac{\left|pSrc1(n) - pSrc2(n)\right|}{max(\left|pSrc1(n)\right|, \left|pSrc2(n)\right|)}}\f]
  -- *
  -- * If the signal is in complex format, the absolute value of the complex number is used.
  -- * @{
  -- *
  --  

  --* 
  -- * 8-bit unsigned char MaximumRelative method.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pDst Pointer to the error result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsMaximumRelativeErrorGetBufferSize_8u to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMaximumRelativeError_8u
     (pSrc1 : access nppdefs_h.Npp8u;
      pSrc2 : access nppdefs_h.Npp8u;
      nLength : int;
      pDst : access nppdefs_h.Npp64f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:4091
   pragma Import (C, nppsMaximumRelativeError_8u, "nppsMaximumRelativeError_8u");

  --* 
  -- * 8-bit signed char MaximumRelative method.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pDst Pointer to the error result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsMaximumRelativeErrorGetBufferSize_8s to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMaximumRelativeError_8s
     (pSrc1 : access nppdefs_h.Npp8s;
      pSrc2 : access nppdefs_h.Npp8s;
      nLength : int;
      pDst : access nppdefs_h.Npp64f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:4104
   pragma Import (C, nppsMaximumRelativeError_8s, "nppsMaximumRelativeError_8s");

  --* 
  -- * 16-bit unsigned short integer MaximumRelative method.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pDst Pointer to the error result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsMaximumRelativeErrorGetBufferSize_16u to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMaximumRelativeError_16u
     (pSrc1 : access nppdefs_h.Npp16u;
      pSrc2 : access nppdefs_h.Npp16u;
      nLength : int;
      pDst : access nppdefs_h.Npp64f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:4117
   pragma Import (C, nppsMaximumRelativeError_16u, "nppsMaximumRelativeError_16u");

  --* 
  -- * 16-bit signed short integer MaximumRelative method.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pDst Pointer to the error result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsMaximumRelativeErrorGetBufferSize_16s to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMaximumRelativeError_16s
     (pSrc1 : access nppdefs_h.Npp16s;
      pSrc2 : access nppdefs_h.Npp16s;
      nLength : int;
      pDst : access nppdefs_h.Npp64f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:4130
   pragma Import (C, nppsMaximumRelativeError_16s, "nppsMaximumRelativeError_16s");

  --* 
  -- * 16-bit unsigned short complex integer MaximumRelative method.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pDst Pointer to the error result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsMaximumRelativeErrorGetBufferSize_16sc to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMaximumRelativeError_16sc
     (pSrc1 : access constant nppdefs_h.Npp16sc;
      pSrc2 : access constant nppdefs_h.Npp16sc;
      nLength : int;
      pDst : access nppdefs_h.Npp64f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:4143
   pragma Import (C, nppsMaximumRelativeError_16sc, "nppsMaximumRelativeError_16sc");

  --* 
  -- * 32-bit unsigned short integer MaximumRelative method.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pDst Pointer to the error result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsMaximumRelativeErrorGetBufferSize_32u to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMaximumRelativeError_32u
     (pSrc1 : access nppdefs_h.Npp32u;
      pSrc2 : access nppdefs_h.Npp32u;
      nLength : int;
      pDst : access nppdefs_h.Npp64f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:4156
   pragma Import (C, nppsMaximumRelativeError_32u, "nppsMaximumRelativeError_32u");

  --* 
  -- * 32-bit signed short integer MaximumRelative method.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pDst Pointer to the error result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsMaximumRelativeErrorGetBufferSize_32s to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMaximumRelativeError_32s
     (pSrc1 : access nppdefs_h.Npp32s;
      pSrc2 : access nppdefs_h.Npp32s;
      nLength : int;
      pDst : access nppdefs_h.Npp64f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:4169
   pragma Import (C, nppsMaximumRelativeError_32s, "nppsMaximumRelativeError_32s");

  --* 
  -- * 32-bit unsigned short complex integer MaximumRelative method.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pDst Pointer to the error result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsMaximumRelativeErrorGetBufferSize_32sc to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMaximumRelativeError_32sc
     (pSrc1 : access constant nppdefs_h.Npp32sc;
      pSrc2 : access constant nppdefs_h.Npp32sc;
      nLength : int;
      pDst : access nppdefs_h.Npp64f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:4182
   pragma Import (C, nppsMaximumRelativeError_32sc, "nppsMaximumRelativeError_32sc");

  --* 
  -- * 64-bit signed short integer MaximumRelative method.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pDst Pointer to the error result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsMaximumRelativeErrorGetBufferSize_64s to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMaximumRelativeError_64s
     (pSrc1 : access nppdefs_h.Npp64s;
      pSrc2 : access nppdefs_h.Npp64s;
      nLength : int;
      pDst : access nppdefs_h.Npp64f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:4195
   pragma Import (C, nppsMaximumRelativeError_64s, "nppsMaximumRelativeError_64s");

  --* 
  -- * 64-bit unsigned short complex integer MaximumRelative method.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pDst Pointer to the error result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsMaximumRelativeErrorGetBufferSize_64sc to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMaximumRelativeError_64sc
     (pSrc1 : access constant nppdefs_h.Npp64sc;
      pSrc2 : access constant nppdefs_h.Npp64sc;
      nLength : int;
      pDst : access nppdefs_h.Npp64f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:4208
   pragma Import (C, nppsMaximumRelativeError_64sc, "nppsMaximumRelativeError_64sc");

  --* 
  -- * 32-bit floating point MaximumRelative method.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pDst Pointer to the error result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsMaximumRelativeErrorGetBufferSize_32f to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMaximumRelativeError_32f
     (pSrc1 : access nppdefs_h.Npp32f;
      pSrc2 : access nppdefs_h.Npp32f;
      nLength : int;
      pDst : access nppdefs_h.Npp64f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:4221
   pragma Import (C, nppsMaximumRelativeError_32f, "nppsMaximumRelativeError_32f");

  --* 
  -- * 32-bit floating point complex MaximumRelative method.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pDst Pointer to the error result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsMaximumRelativeErrorGetBufferSize_32fc to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMaximumRelativeError_32fc
     (pSrc1 : access constant nppdefs_h.Npp32fc;
      pSrc2 : access constant nppdefs_h.Npp32fc;
      nLength : int;
      pDst : access nppdefs_h.Npp64f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:4234
   pragma Import (C, nppsMaximumRelativeError_32fc, "nppsMaximumRelativeError_32fc");

  --* 
  -- * 64-bit floating point MaximumRelative method.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pDst Pointer to the error result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsMaximumRelativeErrorGetBufferSize_64f to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMaximumRelativeError_64f
     (pSrc1 : access nppdefs_h.Npp64f;
      pSrc2 : access nppdefs_h.Npp64f;
      nLength : int;
      pDst : access nppdefs_h.Npp64f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:4247
   pragma Import (C, nppsMaximumRelativeError_64f, "nppsMaximumRelativeError_64f");

  --* 
  -- * 64-bit floating point complex MaximumRelative method.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pDst Pointer to the error result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsMaximumRelativeErrorGetBufferSize_64fc to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsMaximumRelativeError_64fc
     (pSrc1 : access constant nppdefs_h.Npp64fc;
      pSrc2 : access constant nppdefs_h.Npp64fc;
      nLength : int;
      pDst : access nppdefs_h.Npp64f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:4260
   pragma Import (C, nppsMaximumRelativeError_64fc, "nppsMaximumRelativeError_64fc");

  --* 
  -- * Device-buffer size (in bytes) for nppsMaximumRelativeError_8u.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsMaximumRelativeErrorGetBufferSize_8u (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:4270
   pragma Import (C, nppsMaximumRelativeErrorGetBufferSize_8u, "nppsMaximumRelativeErrorGetBufferSize_8u");

  --* 
  -- * Device-buffer size (in bytes) for nppsMaximumRelativeError_8s.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsMaximumRelativeErrorGetBufferSize_8s (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:4280
   pragma Import (C, nppsMaximumRelativeErrorGetBufferSize_8s, "nppsMaximumRelativeErrorGetBufferSize_8s");

  --* 
  -- * Device-buffer size (in bytes) for nppsMaximumRelativeError_16u.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsMaximumRelativeErrorGetBufferSize_16u (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:4290
   pragma Import (C, nppsMaximumRelativeErrorGetBufferSize_16u, "nppsMaximumRelativeErrorGetBufferSize_16u");

  --* 
  -- * Device-buffer size (in bytes) for nppsMaximumRelativeError_16s.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsMaximumRelativeErrorGetBufferSize_16s (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:4300
   pragma Import (C, nppsMaximumRelativeErrorGetBufferSize_16s, "nppsMaximumRelativeErrorGetBufferSize_16s");

  --* 
  -- * Device-buffer size (in bytes) for nppsMaximumRelativeError_16sc.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsMaximumRelativeErrorGetBufferSize_16sc (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:4310
   pragma Import (C, nppsMaximumRelativeErrorGetBufferSize_16sc, "nppsMaximumRelativeErrorGetBufferSize_16sc");

  --* 
  -- * Device-buffer size (in bytes) for nppsMaximumRelativeError_32u.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsMaximumRelativeErrorGetBufferSize_32u (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:4320
   pragma Import (C, nppsMaximumRelativeErrorGetBufferSize_32u, "nppsMaximumRelativeErrorGetBufferSize_32u");

  --* 
  -- * Device-buffer size (in bytes) for nppsMaximumRelativeError_32s.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsMaximumRelativeErrorGetBufferSize_32s (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:4330
   pragma Import (C, nppsMaximumRelativeErrorGetBufferSize_32s, "nppsMaximumRelativeErrorGetBufferSize_32s");

  --* 
  -- * Device-buffer size (in bytes) for nppsMaximumRelativeError_32sc.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsMaximumRelativeErrorGetBufferSize_32sc (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:4340
   pragma Import (C, nppsMaximumRelativeErrorGetBufferSize_32sc, "nppsMaximumRelativeErrorGetBufferSize_32sc");

  --* 
  -- * Device-buffer size (in bytes) for nppsMaximumRelativeError_64s.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsMaximumRelativeErrorGetBufferSize_64s (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:4350
   pragma Import (C, nppsMaximumRelativeErrorGetBufferSize_64s, "nppsMaximumRelativeErrorGetBufferSize_64s");

  --* 
  -- * Device-buffer size (in bytes) for nppsMaximumRelativeError_64sc.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsMaximumRelativeErrorGetBufferSize_64sc (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:4360
   pragma Import (C, nppsMaximumRelativeErrorGetBufferSize_64sc, "nppsMaximumRelativeErrorGetBufferSize_64sc");

  --* 
  -- * Device-buffer size (in bytes) for nppsMaximumRelativeError_32f.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsMaximumRelativeErrorGetBufferSize_32f (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:4370
   pragma Import (C, nppsMaximumRelativeErrorGetBufferSize_32f, "nppsMaximumRelativeErrorGetBufferSize_32f");

  --* 
  -- * Device-buffer size (in bytes) for nppsMaximumRelativeError_32fc.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsMaximumRelativeErrorGetBufferSize_32fc (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:4380
   pragma Import (C, nppsMaximumRelativeErrorGetBufferSize_32fc, "nppsMaximumRelativeErrorGetBufferSize_32fc");

  --* 
  -- * Device-buffer size (in bytes) for nppsMaximumRelativeError_64f.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsMaximumRelativeErrorGetBufferSize_64f (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:4390
   pragma Import (C, nppsMaximumRelativeErrorGetBufferSize_64f, "nppsMaximumRelativeErrorGetBufferSize_64f");

  --* 
  -- * Device-buffer size (in bytes) for nppsMaximumRelativeError_64fc.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsMaximumRelativeErrorGetBufferSize_64fc (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:4400
   pragma Import (C, nppsMaximumRelativeErrorGetBufferSize_64fc, "nppsMaximumRelativeErrorGetBufferSize_64fc");

  --* @}  
  --* @defgroup signal_average_relative_error AverageRelativeError
  -- * Primitives for computing the AverageRelative error between two signals.
  -- * Given two signals \f$pSrc1\f$ and \f$pSrc2\f$ both with length \f$N\f$, 
  -- * the average relative error is defined as
  -- * \f[AverageRelativeError = \frac{1}{N}\sum_{n=0}^{N-1}\frac{\left|pSrc1(n) - pSrc2(n)\right|}{max(\left|pSrc1(n)\right|, \left|pSrc2(n)\right|)}\f]
  -- *
  -- * If the signal is in complex format, the absolute value of the complex number is used.
  -- * @{
  -- *
  --  

  --* 
  -- * 8-bit unsigned char AverageRelative method.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pDst Pointer to the error result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsAverageRelativeErrorGetBufferSize_8u to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAverageRelativeError_8u
     (pSrc1 : access nppdefs_h.Npp8u;
      pSrc2 : access nppdefs_h.Npp8u;
      nLength : int;
      pDst : access nppdefs_h.Npp64f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:4424
   pragma Import (C, nppsAverageRelativeError_8u, "nppsAverageRelativeError_8u");

  --* 
  -- * 8-bit signed char AverageRelative method.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pDst Pointer to the error result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsAverageRelativeErrorGetBufferSize_8s to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAverageRelativeError_8s
     (pSrc1 : access nppdefs_h.Npp8s;
      pSrc2 : access nppdefs_h.Npp8s;
      nLength : int;
      pDst : access nppdefs_h.Npp64f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:4437
   pragma Import (C, nppsAverageRelativeError_8s, "nppsAverageRelativeError_8s");

  --* 
  -- * 16-bit unsigned short integer AverageRelative method.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pDst Pointer to the error result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsAverageRelativeErrorGetBufferSize_16u to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAverageRelativeError_16u
     (pSrc1 : access nppdefs_h.Npp16u;
      pSrc2 : access nppdefs_h.Npp16u;
      nLength : int;
      pDst : access nppdefs_h.Npp64f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:4450
   pragma Import (C, nppsAverageRelativeError_16u, "nppsAverageRelativeError_16u");

  --* 
  -- * 16-bit signed short integer AverageRelative method.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pDst Pointer to the error result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsAverageRelativeErrorGetBufferSize_16s to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAverageRelativeError_16s
     (pSrc1 : access nppdefs_h.Npp16s;
      pSrc2 : access nppdefs_h.Npp16s;
      nLength : int;
      pDst : access nppdefs_h.Npp64f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:4463
   pragma Import (C, nppsAverageRelativeError_16s, "nppsAverageRelativeError_16s");

  --* 
  -- * 16-bit unsigned short complex integer AverageRelative method.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pDst Pointer to the error result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsAverageRelativeErrorGetBufferSize_16sc to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAverageRelativeError_16sc
     (pSrc1 : access constant nppdefs_h.Npp16sc;
      pSrc2 : access constant nppdefs_h.Npp16sc;
      nLength : int;
      pDst : access nppdefs_h.Npp64f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:4476
   pragma Import (C, nppsAverageRelativeError_16sc, "nppsAverageRelativeError_16sc");

  --* 
  -- * 32-bit unsigned short integer AverageRelative method.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pDst Pointer to the error result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsAverageRelativeErrorGetBufferSize_32u to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAverageRelativeError_32u
     (pSrc1 : access nppdefs_h.Npp32u;
      pSrc2 : access nppdefs_h.Npp32u;
      nLength : int;
      pDst : access nppdefs_h.Npp64f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:4489
   pragma Import (C, nppsAverageRelativeError_32u, "nppsAverageRelativeError_32u");

  --* 
  -- * 32-bit signed short integer AverageRelative method.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pDst Pointer to the error result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsAverageRelativeErrorGetBufferSize_32s to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAverageRelativeError_32s
     (pSrc1 : access nppdefs_h.Npp32s;
      pSrc2 : access nppdefs_h.Npp32s;
      nLength : int;
      pDst : access nppdefs_h.Npp64f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:4502
   pragma Import (C, nppsAverageRelativeError_32s, "nppsAverageRelativeError_32s");

  --* 
  -- * 32-bit unsigned short complex integer AverageRelative method.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pDst Pointer to the error result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsAverageRelativeErrorGetBufferSize_32sc to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAverageRelativeError_32sc
     (pSrc1 : access constant nppdefs_h.Npp32sc;
      pSrc2 : access constant nppdefs_h.Npp32sc;
      nLength : int;
      pDst : access nppdefs_h.Npp64f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:4515
   pragma Import (C, nppsAverageRelativeError_32sc, "nppsAverageRelativeError_32sc");

  --* 
  -- * 64-bit signed short integer AverageRelative method.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pDst Pointer to the error result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsAverageRelativeErrorGetBufferSize_64s to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAverageRelativeError_64s
     (pSrc1 : access nppdefs_h.Npp64s;
      pSrc2 : access nppdefs_h.Npp64s;
      nLength : int;
      pDst : access nppdefs_h.Npp64f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:4528
   pragma Import (C, nppsAverageRelativeError_64s, "nppsAverageRelativeError_64s");

  --* 
  -- * 64-bit unsigned short complex integer AverageRelative method.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pDst Pointer to the error result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsAverageRelativeErrorGetBufferSize_64sc to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAverageRelativeError_64sc
     (pSrc1 : access constant nppdefs_h.Npp64sc;
      pSrc2 : access constant nppdefs_h.Npp64sc;
      nLength : int;
      pDst : access nppdefs_h.Npp64f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:4541
   pragma Import (C, nppsAverageRelativeError_64sc, "nppsAverageRelativeError_64sc");

  --* 
  -- * 32-bit floating point AverageRelative method.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pDst Pointer to the error result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsAverageRelativeErrorGetBufferSize_32f to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAverageRelativeError_32f
     (pSrc1 : access nppdefs_h.Npp32f;
      pSrc2 : access nppdefs_h.Npp32f;
      nLength : int;
      pDst : access nppdefs_h.Npp64f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:4554
   pragma Import (C, nppsAverageRelativeError_32f, "nppsAverageRelativeError_32f");

  --* 
  -- * 32-bit floating point complex AverageRelative method.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pDst Pointer to the error result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsAverageRelativeErrorGetBufferSize_32fc to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAverageRelativeError_32fc
     (pSrc1 : access constant nppdefs_h.Npp32fc;
      pSrc2 : access constant nppdefs_h.Npp32fc;
      nLength : int;
      pDst : access nppdefs_h.Npp64f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:4567
   pragma Import (C, nppsAverageRelativeError_32fc, "nppsAverageRelativeError_32fc");

  --* 
  -- * 64-bit floating point AverageRelative method.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pDst Pointer to the error result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsAverageRelativeErrorGetBufferSize_64f to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAverageRelativeError_64f
     (pSrc1 : access nppdefs_h.Npp64f;
      pSrc2 : access nppdefs_h.Npp64f;
      nLength : int;
      pDst : access nppdefs_h.Npp64f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:4580
   pragma Import (C, nppsAverageRelativeError_64f, "nppsAverageRelativeError_64f");

  --* 
  -- * 64-bit floating point complex AverageRelative method.
  -- * \param pSrc1 \ref source_signal_pointer.
  -- * \param pSrc2 \ref source_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param pDst Pointer to the error result.
  -- * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
  -- *        Use \ref nppsAverageRelativeErrorGetBufferSize_64fc to determine the minium number of bytes required.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsAverageRelativeError_64fc
     (pSrc1 : access constant nppdefs_h.Npp64fc;
      pSrc2 : access constant nppdefs_h.Npp64fc;
      nLength : int;
      pDst : access nppdefs_h.Npp64f;
      pDeviceBuffer : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:4593
   pragma Import (C, nppsAverageRelativeError_64fc, "nppsAverageRelativeError_64fc");

  --* 
  -- * Device-buffer size (in bytes) for nppsAverageRelativeError_8u.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsAverageRelativeErrorGetBufferSize_8u (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:4603
   pragma Import (C, nppsAverageRelativeErrorGetBufferSize_8u, "nppsAverageRelativeErrorGetBufferSize_8u");

  --* 
  -- * Device-buffer size (in bytes) for nppsAverageRelativeError_8s.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsAverageRelativeErrorGetBufferSize_8s (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:4613
   pragma Import (C, nppsAverageRelativeErrorGetBufferSize_8s, "nppsAverageRelativeErrorGetBufferSize_8s");

  --* 
  -- * Device-buffer size (in bytes) for nppsAverageRelativeError_16u.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsAverageRelativeErrorGetBufferSize_16u (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:4623
   pragma Import (C, nppsAverageRelativeErrorGetBufferSize_16u, "nppsAverageRelativeErrorGetBufferSize_16u");

  --* 
  -- * Device-buffer size (in bytes) for nppsAverageRelativeError_16s.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsAverageRelativeErrorGetBufferSize_16s (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:4633
   pragma Import (C, nppsAverageRelativeErrorGetBufferSize_16s, "nppsAverageRelativeErrorGetBufferSize_16s");

  --* 
  -- * Device-buffer size (in bytes) for nppsAverageRelativeError_16sc.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsAverageRelativeErrorGetBufferSize_16sc (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:4643
   pragma Import (C, nppsAverageRelativeErrorGetBufferSize_16sc, "nppsAverageRelativeErrorGetBufferSize_16sc");

  --* 
  -- * Device-buffer size (in bytes) for nppsAverageRelativeError_32u.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsAverageRelativeErrorGetBufferSize_32u (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:4653
   pragma Import (C, nppsAverageRelativeErrorGetBufferSize_32u, "nppsAverageRelativeErrorGetBufferSize_32u");

  --* 
  -- * Device-buffer size (in bytes) for nppsAverageRelativeError_32s.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsAverageRelativeErrorGetBufferSize_32s (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:4663
   pragma Import (C, nppsAverageRelativeErrorGetBufferSize_32s, "nppsAverageRelativeErrorGetBufferSize_32s");

  --* 
  -- * Device-buffer size (in bytes) for nppsAverageRelativeError_32sc.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsAverageRelativeErrorGetBufferSize_32sc (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:4673
   pragma Import (C, nppsAverageRelativeErrorGetBufferSize_32sc, "nppsAverageRelativeErrorGetBufferSize_32sc");

  --* 
  -- * Device-buffer size (in bytes) for nppsAverageRelativeError_64s.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsAverageRelativeErrorGetBufferSize_64s (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:4683
   pragma Import (C, nppsAverageRelativeErrorGetBufferSize_64s, "nppsAverageRelativeErrorGetBufferSize_64s");

  --* 
  -- * Device-buffer size (in bytes) for nppsAverageRelativeError_64sc.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsAverageRelativeErrorGetBufferSize_64sc (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:4693
   pragma Import (C, nppsAverageRelativeErrorGetBufferSize_64sc, "nppsAverageRelativeErrorGetBufferSize_64sc");

  --* 
  -- * Device-buffer size (in bytes) for nppsAverageRelativeError_32f.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsAverageRelativeErrorGetBufferSize_32f (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:4703
   pragma Import (C, nppsAverageRelativeErrorGetBufferSize_32f, "nppsAverageRelativeErrorGetBufferSize_32f");

  --* 
  -- * Device-buffer size (in bytes) for nppsAverageRelativeError_32fc.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsAverageRelativeErrorGetBufferSize_32fc (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:4713
   pragma Import (C, nppsAverageRelativeErrorGetBufferSize_32fc, "nppsAverageRelativeErrorGetBufferSize_32fc");

  --* 
  -- * Device-buffer size (in bytes) for nppsAverageRelativeError_64f.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsAverageRelativeErrorGetBufferSize_64f (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:4723
   pragma Import (C, nppsAverageRelativeErrorGetBufferSize_64f, "nppsAverageRelativeErrorGetBufferSize_64f");

  --* 
  -- * Device-buffer size (in bytes) for nppsAverageRelativeError_64fc.
  -- * \param nLength \ref length_specification.
  -- * \param hpBufferSize Required buffer size.  Important: 
  -- *        hpBufferSize is a <em>host pointer.</em>
  -- * \return NPP_SUCCESS
  --  

   function nppsAverageRelativeErrorGetBufferSize_64fc (nLength : int; hpBufferSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_statistics_functions.h:4733
   pragma Import (C, nppsAverageRelativeErrorGetBufferSize_64fc, "nppsAverageRelativeErrorGetBufferSize_64fc");

  --* @}  
  --* @} signal_statistical_functions  
  -- extern "C"  
end npps_statistics_functions_h;

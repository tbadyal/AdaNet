pragma Ada_2005;
pragma Style_Checks (Off);

with Interfaces.C; use Interfaces.C;
with nppdefs_h;

package npps_conversion_functions_h is

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
  -- * \file npps_conversion_functions.h
  -- * NPP Signal Processing Functionality.
  --  

  --* @defgroup signal_conversion_functions Conversion Functions
  -- *  @ingroup npps
  -- *
  -- * @{
  -- *
  --  

  --* @defgroup signal_convert Convert
  -- *
  -- * @{
  -- *
  --  

  --* @name Convert
  -- * Routines for converting the sample-data type of signals.
  -- *
  -- * @{
  -- *
  --  

   function nppsConvert_8s16s
     (pSrc : access nppdefs_h.Npp8s;
      pDst : access nppdefs_h.Npp16s;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_conversion_functions.h:85
   pragma Import (C, nppsConvert_8s16s, "nppsConvert_8s16s");

   function nppsConvert_8s32f
     (pSrc : access nppdefs_h.Npp8s;
      pDst : access nppdefs_h.Npp32f;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_conversion_functions.h:88
   pragma Import (C, nppsConvert_8s32f, "nppsConvert_8s32f");

   function nppsConvert_8u32f
     (pSrc : access nppdefs_h.Npp8u;
      pDst : access nppdefs_h.Npp32f;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_conversion_functions.h:91
   pragma Import (C, nppsConvert_8u32f, "nppsConvert_8u32f");

   function nppsConvert_16s8s_Sfs
     (pSrc : access nppdefs_h.Npp16s;
      pDst : access nppdefs_h.Npp8s;
      nLength : nppdefs_h.Npp32u;
      eRoundMode : nppdefs_h.NppRoundMode;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_conversion_functions.h:94
   pragma Import (C, nppsConvert_16s8s_Sfs, "nppsConvert_16s8s_Sfs");

   function nppsConvert_16s32s
     (pSrc : access nppdefs_h.Npp16s;
      pDst : access nppdefs_h.Npp32s;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_conversion_functions.h:97
   pragma Import (C, nppsConvert_16s32s, "nppsConvert_16s32s");

   function nppsConvert_16s32f
     (pSrc : access nppdefs_h.Npp16s;
      pDst : access nppdefs_h.Npp32f;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_conversion_functions.h:100
   pragma Import (C, nppsConvert_16s32f, "nppsConvert_16s32f");

   function nppsConvert_16u32f
     (pSrc : access nppdefs_h.Npp16u;
      pDst : access nppdefs_h.Npp32f;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_conversion_functions.h:103
   pragma Import (C, nppsConvert_16u32f, "nppsConvert_16u32f");

   function nppsConvert_32s16s
     (pSrc : access nppdefs_h.Npp32s;
      pDst : access nppdefs_h.Npp16s;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_conversion_functions.h:106
   pragma Import (C, nppsConvert_32s16s, "nppsConvert_32s16s");

   function nppsConvert_32s32f
     (pSrc : access nppdefs_h.Npp32s;
      pDst : access nppdefs_h.Npp32f;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_conversion_functions.h:109
   pragma Import (C, nppsConvert_32s32f, "nppsConvert_32s32f");

   function nppsConvert_32s64f
     (pSrc : access nppdefs_h.Npp32s;
      pDst : access nppdefs_h.Npp64f;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_conversion_functions.h:112
   pragma Import (C, nppsConvert_32s64f, "nppsConvert_32s64f");

   function nppsConvert_32f64f
     (pSrc : access nppdefs_h.Npp32f;
      pDst : access nppdefs_h.Npp64f;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_conversion_functions.h:115
   pragma Import (C, nppsConvert_32f64f, "nppsConvert_32f64f");

   function nppsConvert_64s64f
     (pSrc : access nppdefs_h.Npp64s;
      pDst : access nppdefs_h.Npp64f;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_conversion_functions.h:118
   pragma Import (C, nppsConvert_64s64f, "nppsConvert_64s64f");

   function nppsConvert_64f32f
     (pSrc : access nppdefs_h.Npp64f;
      pDst : access nppdefs_h.Npp32f;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_conversion_functions.h:121
   pragma Import (C, nppsConvert_64f32f, "nppsConvert_64f32f");

   function nppsConvert_16s32f_Sfs
     (pSrc : access nppdefs_h.Npp16s;
      pDst : access nppdefs_h.Npp32f;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_conversion_functions.h:124
   pragma Import (C, nppsConvert_16s32f_Sfs, "nppsConvert_16s32f_Sfs");

   function nppsConvert_16s64f_Sfs
     (pSrc : access nppdefs_h.Npp16s;
      pDst : access nppdefs_h.Npp64f;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_conversion_functions.h:127
   pragma Import (C, nppsConvert_16s64f_Sfs, "nppsConvert_16s64f_Sfs");

   function nppsConvert_32s16s_Sfs
     (pSrc : access nppdefs_h.Npp32s;
      pDst : access nppdefs_h.Npp16s;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_conversion_functions.h:130
   pragma Import (C, nppsConvert_32s16s_Sfs, "nppsConvert_32s16s_Sfs");

   function nppsConvert_32s32f_Sfs
     (pSrc : access nppdefs_h.Npp32s;
      pDst : access nppdefs_h.Npp32f;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_conversion_functions.h:133
   pragma Import (C, nppsConvert_32s32f_Sfs, "nppsConvert_32s32f_Sfs");

   function nppsConvert_32s64f_Sfs
     (pSrc : access nppdefs_h.Npp32s;
      pDst : access nppdefs_h.Npp64f;
      nLength : int;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_conversion_functions.h:136
   pragma Import (C, nppsConvert_32s64f_Sfs, "nppsConvert_32s64f_Sfs");

   function nppsConvert_32f8s_Sfs
     (pSrc : access nppdefs_h.Npp32f;
      pDst : access nppdefs_h.Npp8s;
      nLength : int;
      eRoundMode : nppdefs_h.NppRoundMode;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_conversion_functions.h:139
   pragma Import (C, nppsConvert_32f8s_Sfs, "nppsConvert_32f8s_Sfs");

   function nppsConvert_32f8u_Sfs
     (pSrc : access nppdefs_h.Npp32f;
      pDst : access nppdefs_h.Npp8u;
      nLength : int;
      eRoundMode : nppdefs_h.NppRoundMode;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_conversion_functions.h:142
   pragma Import (C, nppsConvert_32f8u_Sfs, "nppsConvert_32f8u_Sfs");

   function nppsConvert_32f16s_Sfs
     (pSrc : access nppdefs_h.Npp32f;
      pDst : access nppdefs_h.Npp16s;
      nLength : int;
      eRoundMode : nppdefs_h.NppRoundMode;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_conversion_functions.h:145
   pragma Import (C, nppsConvert_32f16s_Sfs, "nppsConvert_32f16s_Sfs");

   function nppsConvert_32f16u_Sfs
     (pSrc : access nppdefs_h.Npp32f;
      pDst : access nppdefs_h.Npp16u;
      nLength : int;
      eRoundMode : nppdefs_h.NppRoundMode;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_conversion_functions.h:148
   pragma Import (C, nppsConvert_32f16u_Sfs, "nppsConvert_32f16u_Sfs");

   function nppsConvert_32f32s_Sfs
     (pSrc : access nppdefs_h.Npp32f;
      pDst : access nppdefs_h.Npp32s;
      nLength : int;
      eRoundMode : nppdefs_h.NppRoundMode;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_conversion_functions.h:151
   pragma Import (C, nppsConvert_32f32s_Sfs, "nppsConvert_32f32s_Sfs");

   function nppsConvert_64s32s_Sfs
     (pSrc : access nppdefs_h.Npp64s;
      pDst : access nppdefs_h.Npp32s;
      nLength : int;
      eRoundMode : nppdefs_h.NppRoundMode;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_conversion_functions.h:154
   pragma Import (C, nppsConvert_64s32s_Sfs, "nppsConvert_64s32s_Sfs");

   function nppsConvert_64f16s_Sfs
     (pSrc : access nppdefs_h.Npp64f;
      pDst : access nppdefs_h.Npp16s;
      nLength : int;
      eRoundMode : nppdefs_h.NppRoundMode;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_conversion_functions.h:157
   pragma Import (C, nppsConvert_64f16s_Sfs, "nppsConvert_64f16s_Sfs");

   function nppsConvert_64f32s_Sfs
     (pSrc : access nppdefs_h.Npp64f;
      pDst : access nppdefs_h.Npp32s;
      nLength : int;
      eRoundMode : nppdefs_h.NppRoundMode;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_conversion_functions.h:160
   pragma Import (C, nppsConvert_64f32s_Sfs, "nppsConvert_64f32s_Sfs");

   function nppsConvert_64f64s_Sfs
     (pSrc : access nppdefs_h.Npp64f;
      pDst : access nppdefs_h.Npp64s;
      nLength : int;
      eRoundMode : nppdefs_h.NppRoundMode;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_conversion_functions.h:163
   pragma Import (C, nppsConvert_64f64s_Sfs, "nppsConvert_64f64s_Sfs");

  --* @} end of Convert  
  --* @} signal_convert  
  --* @defgroup signal_threshold Threshold
  -- *
  -- * @{
  -- *
  --  

  --* @name Threshold Functions
  -- * Performs the threshold operation on the samples of a signal by limiting the sample values by a specified constant value.
  -- *
  -- * @{
  -- *
  --  

  --* 
  -- * 16-bit signed short signal threshold with constant level.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nLevel Constant threshold value to be used to limit each signal sample
  -- * \param nRelOp NppCmpOp type of thresholding operation (NPP_CMP_LESS or NPP_CMP_GREATER only).
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsThreshold_16s
     (pSrc : access nppdefs_h.Npp16s;
      pDst : access nppdefs_h.Npp16s;
      nLength : int;
      nLevel : nppdefs_h.Npp16s;
      nRelOp : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_conversion_functions.h:192
   pragma Import (C, nppsThreshold_16s, "nppsThreshold_16s");

  --* 
  -- * 16-bit in place signed short signal threshold with constant level.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nLevel Constant threshold value to be used to limit each signal sample
  -- * \param nRelOp NppCmpOp type of thresholding operation (NPP_CMP_LESS or NPP_CMP_GREATER only).
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsThreshold_16s_I
     (pSrcDst : access nppdefs_h.Npp16s;
      nLength : int;
      nLevel : nppdefs_h.Npp16s;
      nRelOp : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_conversion_functions.h:203
   pragma Import (C, nppsThreshold_16s_I, "nppsThreshold_16s_I");

  --* 
  -- * 16-bit signed short complex number signal threshold with constant level.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
  -- * \param nRelOp NppCmpOp type of thresholding operation (NPP_CMP_LESS or NPP_CMP_GREATER only).
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsThreshold_16sc
     (pSrc : access constant nppdefs_h.Npp16sc;
      pDst : access nppdefs_h.Npp16sc;
      nLength : int;
      nLevel : nppdefs_h.Npp16s;
      nRelOp : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_conversion_functions.h:215
   pragma Import (C, nppsThreshold_16sc, "nppsThreshold_16sc");

  --* 
  -- * 16-bit in place signed short complex number signal threshold with constant level.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
  -- * \param nRelOp NppCmpOp type of thresholding operation (NPP_CMP_LESS or NPP_CMP_GREATER only).
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsThreshold_16sc_I
     (pSrcDst : access nppdefs_h.Npp16sc;
      nLength : int;
      nLevel : nppdefs_h.Npp16s;
      nRelOp : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_conversion_functions.h:226
   pragma Import (C, nppsThreshold_16sc_I, "nppsThreshold_16sc_I");

  --* 
  -- * 32-bit floating point signal threshold with constant level.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nLevel Constant threshold value to be used to limit each signal sample
  -- * \param nRelOp NppCmpOp type of thresholding operation (NPP_CMP_LESS or NPP_CMP_GREATER only).
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsThreshold_32f
     (pSrc : access nppdefs_h.Npp32f;
      pDst : access nppdefs_h.Npp32f;
      nLength : int;
      nLevel : nppdefs_h.Npp32f;
      nRelOp : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_conversion_functions.h:238
   pragma Import (C, nppsThreshold_32f, "nppsThreshold_32f");

  --* 
  -- * 32-bit in place floating point signal threshold with constant level.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nLevel Constant threshold value to be used to limit each signal sample
  -- * \param nRelOp NppCmpOp type of thresholding operation (NPP_CMP_LESS or NPP_CMP_GREATER only).
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsThreshold_32f_I
     (pSrcDst : access nppdefs_h.Npp32f;
      nLength : int;
      nLevel : nppdefs_h.Npp32f;
      nRelOp : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_conversion_functions.h:249
   pragma Import (C, nppsThreshold_32f_I, "nppsThreshold_32f_I");

  --* 
  -- * 32-bit floating point complex number signal threshold with constant level.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
  -- * \param nRelOp NppCmpOp type of thresholding operation (NPP_CMP_LESS or NPP_CMP_GREATER only).
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsThreshold_32fc
     (pSrc : access constant nppdefs_h.Npp32fc;
      pDst : access nppdefs_h.Npp32fc;
      nLength : int;
      nLevel : nppdefs_h.Npp32f;
      nRelOp : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_conversion_functions.h:261
   pragma Import (C, nppsThreshold_32fc, "nppsThreshold_32fc");

  --* 
  -- * 32-bit in place floating point complex number signal threshold with constant level.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
  -- * \param nRelOp NppCmpOp type of thresholding operation (NPP_CMP_LESS or NPP_CMP_GREATER only).
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsThreshold_32fc_I
     (pSrcDst : access nppdefs_h.Npp32fc;
      nLength : int;
      nLevel : nppdefs_h.Npp32f;
      nRelOp : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_conversion_functions.h:272
   pragma Import (C, nppsThreshold_32fc_I, "nppsThreshold_32fc_I");

  --* 
  -- * 64-bit floating point signal threshold with constant level.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nLevel Constant threshold value to be used to limit each signal sample
  -- * \param nRelOp NppCmpOp type of thresholding operation (NPP_CMP_LESS or NPP_CMP_GREATER only).
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsThreshold_64f
     (pSrc : access nppdefs_h.Npp64f;
      pDst : access nppdefs_h.Npp64f;
      nLength : int;
      nLevel : nppdefs_h.Npp64f;
      nRelOp : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_conversion_functions.h:284
   pragma Import (C, nppsThreshold_64f, "nppsThreshold_64f");

  --* 
  -- * 64-bit in place floating point signal threshold with constant level.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nLevel Constant threshold value to be used to limit each signal sample
  -- * \param nRelOp NppCmpOp type of thresholding operation (NPP_CMP_LESS or NPP_CMP_GREATER only).
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsThreshold_64f_I
     (pSrcDst : access nppdefs_h.Npp64f;
      nLength : int;
      nLevel : nppdefs_h.Npp64f;
      nRelOp : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_conversion_functions.h:295
   pragma Import (C, nppsThreshold_64f_I, "nppsThreshold_64f_I");

  --* 
  -- * 64-bit floating point complex number signal threshold with constant level.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
  -- * \param nRelOp NppCmpOp type of thresholding operation (NPP_CMP_LESS or NPP_CMP_GREATER only).
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsThreshold_64fc
     (pSrc : access constant nppdefs_h.Npp64fc;
      pDst : access nppdefs_h.Npp64fc;
      nLength : int;
      nLevel : nppdefs_h.Npp64f;
      nRelOp : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_conversion_functions.h:307
   pragma Import (C, nppsThreshold_64fc, "nppsThreshold_64fc");

  --* 
  -- * 64-bit in place floating point complex number signal threshold with constant level.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
  -- * \param nRelOp NppCmpOp type of thresholding operation (NPP_CMP_LESS or NPP_CMP_GREATER only).
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsThreshold_64fc_I
     (pSrcDst : access nppdefs_h.Npp64fc;
      nLength : int;
      nLevel : nppdefs_h.Npp64f;
      nRelOp : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_conversion_functions.h:318
   pragma Import (C, nppsThreshold_64fc_I, "nppsThreshold_64fc_I");

  --* 
  -- * 16-bit signed short signal NPP_CMP_LESS threshold with constant level.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nLevel Constant threshold value to be used to limit each signal sample
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsThreshold_LT_16s
     (pSrc : access nppdefs_h.Npp16s;
      pDst : access nppdefs_h.Npp16s;
      nLength : int;
      nLevel : nppdefs_h.Npp16s) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_conversion_functions.h:329
   pragma Import (C, nppsThreshold_LT_16s, "nppsThreshold_LT_16s");

  --* 
  -- * 16-bit in place signed short signal NPP_CMP_LESS threshold with constant level.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nLevel Constant threshold value to be used to limit each signal sample
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsThreshold_LT_16s_I
     (pSrcDst : access nppdefs_h.Npp16s;
      nLength : int;
      nLevel : nppdefs_h.Npp16s) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_conversion_functions.h:339
   pragma Import (C, nppsThreshold_LT_16s_I, "nppsThreshold_LT_16s_I");

  --* 
  -- * 16-bit signed short complex number signal NPP_CMP_LESS threshold with constant level.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsThreshold_LT_16sc
     (pSrc : access constant nppdefs_h.Npp16sc;
      pDst : access nppdefs_h.Npp16sc;
      nLength : int;
      nLevel : nppdefs_h.Npp16s) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_conversion_functions.h:350
   pragma Import (C, nppsThreshold_LT_16sc, "nppsThreshold_LT_16sc");

  --* 
  -- * 16-bit in place signed short complex number signal NPP_CMP_LESS threshold with constant level.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsThreshold_LT_16sc_I
     (pSrcDst : access nppdefs_h.Npp16sc;
      nLength : int;
      nLevel : nppdefs_h.Npp16s) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_conversion_functions.h:360
   pragma Import (C, nppsThreshold_LT_16sc_I, "nppsThreshold_LT_16sc_I");

  --* 
  -- * 32-bit floating point signal NPP_CMP_LESS threshold with constant level.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nLevel Constant threshold value to be used to limit each signal sample
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsThreshold_LT_32f
     (pSrc : access nppdefs_h.Npp32f;
      pDst : access nppdefs_h.Npp32f;
      nLength : int;
      nLevel : nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_conversion_functions.h:371
   pragma Import (C, nppsThreshold_LT_32f, "nppsThreshold_LT_32f");

  --* 
  -- * 32-bit in place floating point signal NPP_CMP_LESS threshold with constant level.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nLevel Constant threshold value to be used to limit each signal sample
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsThreshold_LT_32f_I
     (pSrcDst : access nppdefs_h.Npp32f;
      nLength : int;
      nLevel : nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_conversion_functions.h:381
   pragma Import (C, nppsThreshold_LT_32f_I, "nppsThreshold_LT_32f_I");

  --* 
  -- * 32-bit floating point complex number signal NPP_CMP_LESS threshold with constant level.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsThreshold_LT_32fc
     (pSrc : access constant nppdefs_h.Npp32fc;
      pDst : access nppdefs_h.Npp32fc;
      nLength : int;
      nLevel : nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_conversion_functions.h:392
   pragma Import (C, nppsThreshold_LT_32fc, "nppsThreshold_LT_32fc");

  --* 
  -- * 32-bit in place floating point complex number signal NPP_CMP_LESS threshold with constant level.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsThreshold_LT_32fc_I
     (pSrcDst : access nppdefs_h.Npp32fc;
      nLength : int;
      nLevel : nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_conversion_functions.h:402
   pragma Import (C, nppsThreshold_LT_32fc_I, "nppsThreshold_LT_32fc_I");

  --* 
  -- * 64-bit floating point signal NPP_CMP_LESS threshold with constant level.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nLevel Constant threshold value to be used to limit each signal sample
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsThreshold_LT_64f
     (pSrc : access nppdefs_h.Npp64f;
      pDst : access nppdefs_h.Npp64f;
      nLength : int;
      nLevel : nppdefs_h.Npp64f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_conversion_functions.h:413
   pragma Import (C, nppsThreshold_LT_64f, "nppsThreshold_LT_64f");

  --* 
  -- * 64-bit in place floating point signal NPP_CMP_LESS threshold with constant level.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nLevel Constant threshold value to be used to limit each signal sample
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsThreshold_LT_64f_I
     (pSrcDst : access nppdefs_h.Npp64f;
      nLength : int;
      nLevel : nppdefs_h.Npp64f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_conversion_functions.h:423
   pragma Import (C, nppsThreshold_LT_64f_I, "nppsThreshold_LT_64f_I");

  --* 
  -- * 64-bit floating point complex number signal NPP_CMP_LESS threshold with constant level.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsThreshold_LT_64fc
     (pSrc : access constant nppdefs_h.Npp64fc;
      pDst : access nppdefs_h.Npp64fc;
      nLength : int;
      nLevel : nppdefs_h.Npp64f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_conversion_functions.h:434
   pragma Import (C, nppsThreshold_LT_64fc, "nppsThreshold_LT_64fc");

  --* 
  -- * 64-bit in place floating point complex number signal NPP_CMP_LESS threshold with constant level.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsThreshold_LT_64fc_I
     (pSrcDst : access nppdefs_h.Npp64fc;
      nLength : int;
      nLevel : nppdefs_h.Npp64f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_conversion_functions.h:444
   pragma Import (C, nppsThreshold_LT_64fc_I, "nppsThreshold_LT_64fc_I");

  --* 
  -- * 16-bit signed short signal NPP_CMP_GREATER threshold with constant level.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nLevel Constant threshold value to be used to limit each signal sample
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsThreshold_GT_16s
     (pSrc : access nppdefs_h.Npp16s;
      pDst : access nppdefs_h.Npp16s;
      nLength : int;
      nLevel : nppdefs_h.Npp16s) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_conversion_functions.h:455
   pragma Import (C, nppsThreshold_GT_16s, "nppsThreshold_GT_16s");

  --* 
  -- * 16-bit in place signed short signal NPP_CMP_GREATER threshold with constant level.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nLevel Constant threshold value to be used to limit each signal sample
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsThreshold_GT_16s_I
     (pSrcDst : access nppdefs_h.Npp16s;
      nLength : int;
      nLevel : nppdefs_h.Npp16s) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_conversion_functions.h:465
   pragma Import (C, nppsThreshold_GT_16s_I, "nppsThreshold_GT_16s_I");

  --* 
  -- * 16-bit signed short complex number signal NPP_CMP_GREATER threshold with constant level.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsThreshold_GT_16sc
     (pSrc : access constant nppdefs_h.Npp16sc;
      pDst : access nppdefs_h.Npp16sc;
      nLength : int;
      nLevel : nppdefs_h.Npp16s) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_conversion_functions.h:476
   pragma Import (C, nppsThreshold_GT_16sc, "nppsThreshold_GT_16sc");

  --* 
  -- * 16-bit in place signed short complex number signal NPP_CMP_GREATER threshold with constant level.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsThreshold_GT_16sc_I
     (pSrcDst : access nppdefs_h.Npp16sc;
      nLength : int;
      nLevel : nppdefs_h.Npp16s) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_conversion_functions.h:486
   pragma Import (C, nppsThreshold_GT_16sc_I, "nppsThreshold_GT_16sc_I");

  --* 
  -- * 32-bit floating point signal NPP_CMP_GREATER threshold with constant level.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nLevel Constant threshold value to be used to limit each signal sample
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsThreshold_GT_32f
     (pSrc : access nppdefs_h.Npp32f;
      pDst : access nppdefs_h.Npp32f;
      nLength : int;
      nLevel : nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_conversion_functions.h:497
   pragma Import (C, nppsThreshold_GT_32f, "nppsThreshold_GT_32f");

  --* 
  -- * 32-bit in place floating point signal NPP_CMP_GREATER threshold with constant level.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nLevel Constant threshold value to be used to limit each signal sample
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsThreshold_GT_32f_I
     (pSrcDst : access nppdefs_h.Npp32f;
      nLength : int;
      nLevel : nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_conversion_functions.h:507
   pragma Import (C, nppsThreshold_GT_32f_I, "nppsThreshold_GT_32f_I");

  --* 
  -- * 32-bit floating point complex number signal NPP_CMP_GREATER threshold with constant level.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsThreshold_GT_32fc
     (pSrc : access constant nppdefs_h.Npp32fc;
      pDst : access nppdefs_h.Npp32fc;
      nLength : int;
      nLevel : nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_conversion_functions.h:518
   pragma Import (C, nppsThreshold_GT_32fc, "nppsThreshold_GT_32fc");

  --* 
  -- * 32-bit in place floating point complex number signal NPP_CMP_GREATER threshold with constant level.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsThreshold_GT_32fc_I
     (pSrcDst : access nppdefs_h.Npp32fc;
      nLength : int;
      nLevel : nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_conversion_functions.h:528
   pragma Import (C, nppsThreshold_GT_32fc_I, "nppsThreshold_GT_32fc_I");

  --* 
  -- * 64-bit floating point signal NPP_CMP_GREATER threshold with constant level.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nLevel Constant threshold value to be used to limit each signal sample
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsThreshold_GT_64f
     (pSrc : access nppdefs_h.Npp64f;
      pDst : access nppdefs_h.Npp64f;
      nLength : int;
      nLevel : nppdefs_h.Npp64f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_conversion_functions.h:539
   pragma Import (C, nppsThreshold_GT_64f, "nppsThreshold_GT_64f");

  --* 
  -- * 64-bit in place floating point signal NPP_CMP_GREATER threshold with constant level.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nLevel Constant threshold value to be used to limit each signal sample
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsThreshold_GT_64f_I
     (pSrcDst : access nppdefs_h.Npp64f;
      nLength : int;
      nLevel : nppdefs_h.Npp64f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_conversion_functions.h:549
   pragma Import (C, nppsThreshold_GT_64f_I, "nppsThreshold_GT_64f_I");

  --* 
  -- * 64-bit floating point complex number signal NPP_CMP_GREATER threshold with constant level.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsThreshold_GT_64fc
     (pSrc : access constant nppdefs_h.Npp64fc;
      pDst : access nppdefs_h.Npp64fc;
      nLength : int;
      nLevel : nppdefs_h.Npp64f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_conversion_functions.h:560
   pragma Import (C, nppsThreshold_GT_64fc, "nppsThreshold_GT_64fc");

  --* 
  -- * 64-bit in place floating point complex number signal NPP_CMP_GREATER threshold with constant level.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsThreshold_GT_64fc_I
     (pSrcDst : access nppdefs_h.Npp64fc;
      nLength : int;
      nLevel : nppdefs_h.Npp64f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_conversion_functions.h:570
   pragma Import (C, nppsThreshold_GT_64fc_I, "nppsThreshold_GT_64fc_I");

  --* 
  -- * 16-bit signed short signal NPP_CMP_LESS threshold with constant level.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nLevel Constant threshold value to be used to limit each signal sample
  -- * \param nValue Constant value to replace source value when threshold test is true.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsThreshold_LTVal_16s
     (pSrc : access nppdefs_h.Npp16s;
      pDst : access nppdefs_h.Npp16s;
      nLength : int;
      nLevel : nppdefs_h.Npp16s;
      nValue : nppdefs_h.Npp16s) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_conversion_functions.h:582
   pragma Import (C, nppsThreshold_LTVal_16s, "nppsThreshold_LTVal_16s");

  --* 
  -- * 16-bit in place signed short signal NPP_CMP_LESS threshold with constant level.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nLevel Constant threshold value to be used to limit each signal sample
  -- * \param nValue Constant value to replace source value when threshold test is true.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsThreshold_LTVal_16s_I
     (pSrcDst : access nppdefs_h.Npp16s;
      nLength : int;
      nLevel : nppdefs_h.Npp16s;
      nValue : nppdefs_h.Npp16s) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_conversion_functions.h:593
   pragma Import (C, nppsThreshold_LTVal_16s_I, "nppsThreshold_LTVal_16s_I");

  --* 
  -- * 16-bit signed short complex number signal NPP_CMP_LESS threshold with constant level.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
  -- * \param nValue Constant value to replace source value when threshold test is true.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsThreshold_LTVal_16sc
     (pSrc : access constant nppdefs_h.Npp16sc;
      pDst : access nppdefs_h.Npp16sc;
      nLength : int;
      nLevel : nppdefs_h.Npp16s;
      nValue : nppdefs_h.Npp16sc) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_conversion_functions.h:605
   pragma Import (C, nppsThreshold_LTVal_16sc, "nppsThreshold_LTVal_16sc");

  --* 
  -- * 16-bit in place signed short complex number signal NPP_CMP_LESS threshold with constant level.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
  -- * \param nValue Constant value to replace source value when threshold test is true.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsThreshold_LTVal_16sc_I
     (pSrcDst : access nppdefs_h.Npp16sc;
      nLength : int;
      nLevel : nppdefs_h.Npp16s;
      nValue : nppdefs_h.Npp16sc) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_conversion_functions.h:616
   pragma Import (C, nppsThreshold_LTVal_16sc_I, "nppsThreshold_LTVal_16sc_I");

  --* 
  -- * 32-bit floating point signal NPP_CMP_LESS threshold with constant level.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nLevel Constant threshold value to be used to limit each signal sample
  -- * \param nValue Constant value to replace source value when threshold test is true.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsThreshold_LTVal_32f
     (pSrc : access nppdefs_h.Npp32f;
      pDst : access nppdefs_h.Npp32f;
      nLength : int;
      nLevel : nppdefs_h.Npp32f;
      nValue : nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_conversion_functions.h:628
   pragma Import (C, nppsThreshold_LTVal_32f, "nppsThreshold_LTVal_32f");

  --* 
  -- * 32-bit in place floating point signal NPP_CMP_LESS threshold with constant level.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nLevel Constant threshold value to be used to limit each signal sample
  -- * \param nValue Constant value to replace source value when threshold test is true.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsThreshold_LTVal_32f_I
     (pSrcDst : access nppdefs_h.Npp32f;
      nLength : int;
      nLevel : nppdefs_h.Npp32f;
      nValue : nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_conversion_functions.h:639
   pragma Import (C, nppsThreshold_LTVal_32f_I, "nppsThreshold_LTVal_32f_I");

  --* 
  -- * 32-bit floating point complex number signal NPP_CMP_LESS threshold with constant level.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
  -- * \param nValue Constant value to replace source value when threshold test is true.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsThreshold_LTVal_32fc
     (pSrc : access constant nppdefs_h.Npp32fc;
      pDst : access nppdefs_h.Npp32fc;
      nLength : int;
      nLevel : nppdefs_h.Npp32f;
      nValue : nppdefs_h.Npp32fc) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_conversion_functions.h:651
   pragma Import (C, nppsThreshold_LTVal_32fc, "nppsThreshold_LTVal_32fc");

  --* 
  -- * 32-bit in place floating point complex number signal NPP_CMP_LESS threshold with constant level.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
  -- * \param nValue Constant value to replace source value when threshold test is true.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsThreshold_LTVal_32fc_I
     (pSrcDst : access nppdefs_h.Npp32fc;
      nLength : int;
      nLevel : nppdefs_h.Npp32f;
      nValue : nppdefs_h.Npp32fc) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_conversion_functions.h:662
   pragma Import (C, nppsThreshold_LTVal_32fc_I, "nppsThreshold_LTVal_32fc_I");

  --* 
  -- * 64-bit floating point signal NPP_CMP_LESS threshold with constant level.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nLevel Constant threshold value to be used to limit each signal sample
  -- * \param nValue Constant value to replace source value when threshold test is true.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsThreshold_LTVal_64f
     (pSrc : access nppdefs_h.Npp64f;
      pDst : access nppdefs_h.Npp64f;
      nLength : int;
      nLevel : nppdefs_h.Npp64f;
      nValue : nppdefs_h.Npp64f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_conversion_functions.h:674
   pragma Import (C, nppsThreshold_LTVal_64f, "nppsThreshold_LTVal_64f");

  --* 
  -- * 64-bit in place floating point signal NPP_CMP_LESS threshold with constant level.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nLevel Constant threshold value to be used to limit each signal sample
  -- * \param nValue Constant value to replace source value when threshold test is true.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsThreshold_LTVal_64f_I
     (pSrcDst : access nppdefs_h.Npp64f;
      nLength : int;
      nLevel : nppdefs_h.Npp64f;
      nValue : nppdefs_h.Npp64f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_conversion_functions.h:685
   pragma Import (C, nppsThreshold_LTVal_64f_I, "nppsThreshold_LTVal_64f_I");

  --* 
  -- * 64-bit floating point complex number signal NPP_CMP_LESS threshold with constant level.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
  -- * \param nValue Constant value to replace source value when threshold test is true.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsThreshold_LTVal_64fc
     (pSrc : access constant nppdefs_h.Npp64fc;
      pDst : access nppdefs_h.Npp64fc;
      nLength : int;
      nLevel : nppdefs_h.Npp64f;
      nValue : nppdefs_h.Npp64fc) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_conversion_functions.h:697
   pragma Import (C, nppsThreshold_LTVal_64fc, "nppsThreshold_LTVal_64fc");

  --* 
  -- * 64-bit in place floating point complex number signal NPP_CMP_LESS threshold with constant level.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
  -- * \param nValue Constant value to replace source value when threshold test is true.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsThreshold_LTVal_64fc_I
     (pSrcDst : access nppdefs_h.Npp64fc;
      nLength : int;
      nLevel : nppdefs_h.Npp64f;
      nValue : nppdefs_h.Npp64fc) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_conversion_functions.h:708
   pragma Import (C, nppsThreshold_LTVal_64fc_I, "nppsThreshold_LTVal_64fc_I");

  --* 
  -- * 16-bit signed short signal NPP_CMP_GREATER threshold with constant level.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nLevel Constant threshold value to be used to limit each signal sample
  -- * \param nValue Constant value to replace source value when threshold test is true.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsThreshold_GTVal_16s
     (pSrc : access nppdefs_h.Npp16s;
      pDst : access nppdefs_h.Npp16s;
      nLength : int;
      nLevel : nppdefs_h.Npp16s;
      nValue : nppdefs_h.Npp16s) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_conversion_functions.h:720
   pragma Import (C, nppsThreshold_GTVal_16s, "nppsThreshold_GTVal_16s");

  --* 
  -- * 16-bit in place signed short signal NPP_CMP_GREATER threshold with constant level.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nLevel Constant threshold value to be used to limit each signal sample
  -- * \param nValue Constant value to replace source value when threshold test is true.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsThreshold_GTVal_16s_I
     (pSrcDst : access nppdefs_h.Npp16s;
      nLength : int;
      nLevel : nppdefs_h.Npp16s;
      nValue : nppdefs_h.Npp16s) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_conversion_functions.h:731
   pragma Import (C, nppsThreshold_GTVal_16s_I, "nppsThreshold_GTVal_16s_I");

  --* 
  -- * 16-bit signed short complex number signal NPP_CMP_GREATER threshold with constant level.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
  -- * \param nValue Constant value to replace source value when threshold test is true.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsThreshold_GTVal_16sc
     (pSrc : access constant nppdefs_h.Npp16sc;
      pDst : access nppdefs_h.Npp16sc;
      nLength : int;
      nLevel : nppdefs_h.Npp16s;
      nValue : nppdefs_h.Npp16sc) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_conversion_functions.h:743
   pragma Import (C, nppsThreshold_GTVal_16sc, "nppsThreshold_GTVal_16sc");

  --* 
  -- * 16-bit in place signed short complex number signal NPP_CMP_GREATER threshold with constant level.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
  -- * \param nValue Constant value to replace source value when threshold test is true.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsThreshold_GTVal_16sc_I
     (pSrcDst : access nppdefs_h.Npp16sc;
      nLength : int;
      nLevel : nppdefs_h.Npp16s;
      nValue : nppdefs_h.Npp16sc) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_conversion_functions.h:754
   pragma Import (C, nppsThreshold_GTVal_16sc_I, "nppsThreshold_GTVal_16sc_I");

  --* 
  -- * 32-bit floating point signal NPP_CMP_GREATER threshold with constant level.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nLevel Constant threshold value to be used to limit each signal sample
  -- * \param nValue Constant value to replace source value when threshold test is true.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsThreshold_GTVal_32f
     (pSrc : access nppdefs_h.Npp32f;
      pDst : access nppdefs_h.Npp32f;
      nLength : int;
      nLevel : nppdefs_h.Npp32f;
      nValue : nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_conversion_functions.h:766
   pragma Import (C, nppsThreshold_GTVal_32f, "nppsThreshold_GTVal_32f");

  --* 
  -- * 32-bit in place floating point signal NPP_CMP_GREATER threshold with constant level.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nLevel Constant threshold value to be used to limit each signal sample
  -- * \param nValue Constant value to replace source value when threshold test is true.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsThreshold_GTVal_32f_I
     (pSrcDst : access nppdefs_h.Npp32f;
      nLength : int;
      nLevel : nppdefs_h.Npp32f;
      nValue : nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_conversion_functions.h:777
   pragma Import (C, nppsThreshold_GTVal_32f_I, "nppsThreshold_GTVal_32f_I");

  --* 
  -- * 32-bit floating point complex number signal NPP_CMP_GREATER threshold with constant level.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
  -- * \param nValue Constant value to replace source value when threshold test is true.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsThreshold_GTVal_32fc
     (pSrc : access constant nppdefs_h.Npp32fc;
      pDst : access nppdefs_h.Npp32fc;
      nLength : int;
      nLevel : nppdefs_h.Npp32f;
      nValue : nppdefs_h.Npp32fc) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_conversion_functions.h:789
   pragma Import (C, nppsThreshold_GTVal_32fc, "nppsThreshold_GTVal_32fc");

  --* 
  -- * 32-bit in place floating point complex number signal NPP_CMP_GREATER threshold with constant level.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
  -- * \param nValue Constant value to replace source value when threshold test is true.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsThreshold_GTVal_32fc_I
     (pSrcDst : access nppdefs_h.Npp32fc;
      nLength : int;
      nLevel : nppdefs_h.Npp32f;
      nValue : nppdefs_h.Npp32fc) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_conversion_functions.h:800
   pragma Import (C, nppsThreshold_GTVal_32fc_I, "nppsThreshold_GTVal_32fc_I");

  --* 
  -- * 64-bit floating point signal NPP_CMP_GREATER threshold with constant level.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nLevel Constant threshold value to be used to limit each signal sample
  -- * \param nValue Constant value to replace source value when threshold test is true.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsThreshold_GTVal_64f
     (pSrc : access nppdefs_h.Npp64f;
      pDst : access nppdefs_h.Npp64f;
      nLength : int;
      nLevel : nppdefs_h.Npp64f;
      nValue : nppdefs_h.Npp64f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_conversion_functions.h:812
   pragma Import (C, nppsThreshold_GTVal_64f, "nppsThreshold_GTVal_64f");

  --* 
  -- * 64-bit in place floating point signal NPP_CMP_GREATER threshold with constant level.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nLevel Constant threshold value to be used to limit each signal sample
  -- * \param nValue Constant value to replace source value when threshold test is true.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsThreshold_GTVal_64f_I
     (pSrcDst : access nppdefs_h.Npp64f;
      nLength : int;
      nLevel : nppdefs_h.Npp64f;
      nValue : nppdefs_h.Npp64f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_conversion_functions.h:823
   pragma Import (C, nppsThreshold_GTVal_64f_I, "nppsThreshold_GTVal_64f_I");

  --* 
  -- * 64-bit floating point complex number signal NPP_CMP_GREATER threshold with constant level.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
  -- * \param nValue Constant value to replace source value when threshold test is true.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsThreshold_GTVal_64fc
     (pSrc : access constant nppdefs_h.Npp64fc;
      pDst : access nppdefs_h.Npp64fc;
      nLength : int;
      nLevel : nppdefs_h.Npp64f;
      nValue : nppdefs_h.Npp64fc) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_conversion_functions.h:835
   pragma Import (C, nppsThreshold_GTVal_64fc, "nppsThreshold_GTVal_64fc");

  --* 
  -- * 64-bit in place floating point complex number signal NPP_CMP_GREATER threshold with constant level.
  -- * \param pSrcDst \ref in_place_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
  -- * \param nValue Constant value to replace source value when threshold test is true.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsThreshold_GTVal_64fc_I
     (pSrcDst : access nppdefs_h.Npp64fc;
      nLength : int;
      nLevel : nppdefs_h.Npp64f;
      nValue : nppdefs_h.Npp64fc) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_conversion_functions.h:846
   pragma Import (C, nppsThreshold_GTVal_64fc_I, "nppsThreshold_GTVal_64fc_I");

  --* @} end of Threshold  
  --* @} signal_threshold  
  --* @} signal_conversion_functions  
  -- extern "C"  
end npps_conversion_functions_h;

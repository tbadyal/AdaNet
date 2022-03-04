pragma Ada_2005;
pragma Style_Checks (Off);

with Interfaces.C; use Interfaces.C;
with nppdefs_h;

package nppi_threshold_and_compare_operations_h is

  -- Copyright 2009-2014 NVIDIA Corporation.  All rights reserved. 
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
  -- * \file nppi_threshold_and_compare_operations.h
  -- * NPP Image Processing Functionality.
  --  

  --* @defgroup image_threshold_and_compare_operations Threshold and Compare Operations
  -- *  @ingroup nppi
  -- *
  -- * Methods for pixel-wise threshold and compare operations.
  -- *
  -- * @{
  -- *
  -- * These functions can be found in either the nppi or nppitc libraries. Linking to only the sub-libraries that you use can significantly
  -- * save link time, application load time, and CUDA runtime startup time when using dynamic libraries.
  -- *
  --  

  --* 
  -- * @defgroup image_threshold_operations Threshold Operations
  -- *
  -- * Threshold image pixels.
  -- *
  -- * @{
  -- *
  --  

  --* 
  -- * 1 channel 8-bit unsigned char threshold.
  -- * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
  -- * to nThreshold, otherwise it is set to sourcePixel.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nThreshold The threshold value.
  -- * \param eComparisonOperation The type of comparison operation to be used. The only valid
  -- *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
  -- * comparison operation type is specified.
  --  

   function nppiThreshold_8u_C1R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nThreshold : nppdefs_h.Npp8u;
      eComparisonOperation : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:100
   pragma Import (C, nppiThreshold_8u_C1R, "nppiThreshold_8u_C1R");

  --* 
  -- * 1 channel 8-bit unsigned char in place threshold.
  -- * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
  -- * to nThreshold, otherwise it is set to sourcePixel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nThreshold The threshold value.
  -- * \param eComparisonOperation The type of comparison operation to be used. The only valid
  -- *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
  -- * comparison operation type is specified.
  --  

   function nppiThreshold_8u_C1IR
     (pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nThreshold : nppdefs_h.Npp8u;
      eComparisonOperation : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:118
   pragma Import (C, nppiThreshold_8u_C1IR, "nppiThreshold_8u_C1IR");

  --* 
  -- * 1 channel 16-bit unsigned short threshold.
  -- * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
  -- * to nThreshold, otherwise it is set to sourcePixel.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nThreshold The threshold value.
  -- * \param eComparisonOperation The type of comparison operation to be used. The only valid
  -- *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
  -- * comparison operation type is specified.
  --  

   function nppiThreshold_16u_C1R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nThreshold : nppdefs_h.Npp16u;
      eComparisonOperation : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:137
   pragma Import (C, nppiThreshold_16u_C1R, "nppiThreshold_16u_C1R");

  --* 
  -- * 1 channel 16-bit unsigned short in place threshold.
  -- * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
  -- * to nThreshold, otherwise it is set to sourcePixel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nThreshold The threshold value.
  -- * \param eComparisonOperation The type of comparison operation to be used. The only valid
  -- *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
  -- * comparison operation type is specified.
  --  

   function nppiThreshold_16u_C1IR
     (pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nThreshold : nppdefs_h.Npp16u;
      eComparisonOperation : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:155
   pragma Import (C, nppiThreshold_16u_C1IR, "nppiThreshold_16u_C1IR");

  --* 
  -- * 1 channel 16-bit signed short threshold.
  -- * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
  -- * to nThreshold, otherwise it is set to sourcePixel.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nThreshold The threshold value.
  -- * \param eComparisonOperation The type of comparison operation to be used. The only valid
  -- *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
  -- * comparison operation type is specified.
  --  

   function nppiThreshold_16s_C1R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nThreshold : nppdefs_h.Npp16s;
      eComparisonOperation : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:174
   pragma Import (C, nppiThreshold_16s_C1R, "nppiThreshold_16s_C1R");

  --* 
  -- * 1 channel 16-bit signed short in place threshold.
  -- * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
  -- * to nThreshold, otherwise it is set to sourcePixel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nThreshold The threshold value.
  -- * \param eComparisonOperation The type of comparison operation to be used. The only valid
  -- *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
  -- * comparison operation type is specified.
  --  

   function nppiThreshold_16s_C1IR
     (pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nThreshold : nppdefs_h.Npp16s;
      eComparisonOperation : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:192
   pragma Import (C, nppiThreshold_16s_C1IR, "nppiThreshold_16s_C1IR");

  --* 
  -- * 1 channel 32-bit floating point threshold.
  -- * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
  -- * to nThreshold, otherwise it is set to sourcePixel.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nThreshold The threshold value.
  -- * \param eComparisonOperation The type of comparison operation to be used. The only valid
  -- *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
  -- * comparison operation type is specified.
  --  

   function nppiThreshold_32f_C1R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nThreshold : nppdefs_h.Npp32f;
      eComparisonOperation : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:211
   pragma Import (C, nppiThreshold_32f_C1R, "nppiThreshold_32f_C1R");

  --* 
  -- * 1 channel 32-bit floating point in place threshold.
  -- * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
  -- * to nThreshold, otherwise it is set to sourcePixel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nThreshold The threshold value.
  -- * \param eComparisonOperation The type of comparison operation to be used. The only valid
  -- *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
  -- * comparison operation type is specified.
  --  

   function nppiThreshold_32f_C1IR
     (pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nThreshold : nppdefs_h.Npp32f;
      eComparisonOperation : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:229
   pragma Import (C, nppiThreshold_32f_C1IR, "nppiThreshold_32f_C1IR");

  --* 
  -- * 3 channel 8-bit unsigned char threshold.
  -- * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
  -- * to nThreshold, otherwise it is set to sourcePixel.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \param eComparisonOperation The type of comparison operation to be used. The only valid
  -- *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
  -- * comparison operation type is specified.
  --  

   function nppiThreshold_8u_C3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp8u;
      eComparisonOperation : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:248
   pragma Import (C, nppiThreshold_8u_C3R, "nppiThreshold_8u_C3R");

  --* 
  -- * 3 channel 8-bit unsigned char in place threshold.
  -- * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
  -- * to nThreshold, otherwise it is set to sourcePixel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \param eComparisonOperation The type of comparison operation to be used. The only valid
  -- *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
  -- * comparison operation type is specified.
  --  

   function nppiThreshold_8u_C3IR
     (pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp8u;
      eComparisonOperation : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:266
   pragma Import (C, nppiThreshold_8u_C3IR, "nppiThreshold_8u_C3IR");

  --* 
  -- * 3 channel 16-bit unsigned short threshold.
  -- * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
  -- * to nThreshold, otherwise it is set to sourcePixel.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \param eComparisonOperation The type of comparison operation to be used. The only valid
  -- *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
  -- * comparison operation type is specified.
  --  

   function nppiThreshold_16u_C3R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp16u;
      eComparisonOperation : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:285
   pragma Import (C, nppiThreshold_16u_C3R, "nppiThreshold_16u_C3R");

  --* 
  -- * 3 channel 16-bit unsigned short in place threshold.
  -- * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
  -- * to nThreshold, otherwise it is set to sourcePixel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \param eComparisonOperation The type of comparison operation to be used. The only valid
  -- *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
  -- * comparison operation type is specified.
  --  

   function nppiThreshold_16u_C3IR
     (pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp16u;
      eComparisonOperation : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:303
   pragma Import (C, nppiThreshold_16u_C3IR, "nppiThreshold_16u_C3IR");

  --* 
  -- * 3 channel 16-bit signed short threshold.
  -- * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
  -- * to nThreshold, otherwise it is set to sourcePixel.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \param eComparisonOperation The type of comparison operation to be used. The only valid
  -- *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
  -- * comparison operation type is specified.
  --  

   function nppiThreshold_16s_C3R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp16s;
      eComparisonOperation : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:322
   pragma Import (C, nppiThreshold_16s_C3R, "nppiThreshold_16s_C3R");

  --* 
  -- * 3 channel 16-bit signed short in place threshold.
  -- * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
  -- * to nThreshold, otherwise it is set to sourcePixel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \param eComparisonOperation The type of comparison operation to be used. The only valid
  -- *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
  -- * comparison operation type is specified.
  --  

   function nppiThreshold_16s_C3IR
     (pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp16s;
      eComparisonOperation : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:340
   pragma Import (C, nppiThreshold_16s_C3IR, "nppiThreshold_16s_C3IR");

  --* 
  -- * 3 channel 32-bit floating point threshold.
  -- * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
  -- * to nThreshold, otherwise it is set to sourcePixel.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \param eComparisonOperation The type of comparison operation to be used. The only valid
  -- *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
  -- * comparison operation type is specified.
  --  

   function nppiThreshold_32f_C3R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp32f;
      eComparisonOperation : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:359
   pragma Import (C, nppiThreshold_32f_C3R, "nppiThreshold_32f_C3R");

  --* 
  -- * 3 channel 32-bit floating point in place threshold.
  -- * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
  -- * to nThreshold, otherwise it is set to sourcePixel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \param eComparisonOperation The type of comparison operation to be used. The only valid
  -- *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
  -- * comparison operation type is specified.
  --  

   function nppiThreshold_32f_C3IR
     (pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp32f;
      eComparisonOperation : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:377
   pragma Import (C, nppiThreshold_32f_C3IR, "nppiThreshold_32f_C3IR");

  --* 
  -- * 4 channel 8-bit unsigned char image threshold, not affecting Alpha.
  -- * If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel
  -- * value is set to nThreshold, otherwise it is set to sourcePixel.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \param eComparisonOperation The type of comparison operation to be used. The only valid
  -- *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
  -- * comparison operation type is specified.
  --  

   function nppiThreshold_8u_AC4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp8u;
      eComparisonOperation : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:397
   pragma Import (C, nppiThreshold_8u_AC4R, "nppiThreshold_8u_AC4R");

  --* 
  -- * 4 channel 8-bit unsigned char in place image threshold, not affecting Alpha.
  -- * If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel
  -- * value is set to nThreshold, otherwise it is set to sourcePixel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \param eComparisonOperation The type of comparison operation to be used. The only valid
  -- *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
  -- * comparison operation type is specified.
  --  

   function nppiThreshold_8u_AC4IR
     (pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp8u;
      eComparisonOperation : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:415
   pragma Import (C, nppiThreshold_8u_AC4IR, "nppiThreshold_8u_AC4IR");

  --* 
  -- * 4 channel 16-bit unsigned short image threshold, not affecting Alpha.
  -- * If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel
  -- * value is set to nThreshold, otherwise it is set to sourcePixel.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \param eComparisonOperation The type of comparison operation to be used. The only valid
  -- *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
  -- * comparison operation type is specified.
  --  

   function nppiThreshold_16u_AC4R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp16u;
      eComparisonOperation : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:434
   pragma Import (C, nppiThreshold_16u_AC4R, "nppiThreshold_16u_AC4R");

  --* 
  -- * 4 channel 16-bit unsigned short in place image threshold, not affecting Alpha.
  -- * If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel
  -- * value is set to nThreshold, otherwise it is set to sourcePixel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \param eComparisonOperation The type of comparison operation to be used. The only valid
  -- *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
  -- * comparison operation type is specified.
  --  

   function nppiThreshold_16u_AC4IR
     (pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp16u;
      eComparisonOperation : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:452
   pragma Import (C, nppiThreshold_16u_AC4IR, "nppiThreshold_16u_AC4IR");

  --* 
  -- * 4 channel 16-bit signed short image threshold, not affecting Alpha.
  -- * If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel
  -- * value is set to nThreshold, otherwise it is set to sourcePixel.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \param eComparisonOperation The type of comparison operation to be used. The only valid
  -- *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
  -- * comparison operation type is specified.
  --  

   function nppiThreshold_16s_AC4R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp16s;
      eComparisonOperation : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:471
   pragma Import (C, nppiThreshold_16s_AC4R, "nppiThreshold_16s_AC4R");

  --* 
  -- * 4 channel 16-bit signed short in place image threshold, not affecting Alpha.
  -- * If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel
  -- * value is set to nThreshold, otherwise it is set to sourcePixel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \param eComparisonOperation The type of comparison operation to be used. The only valid
  -- *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
  -- * comparison operation type is specified.
  --  

   function nppiThreshold_16s_AC4IR
     (pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp16s;
      eComparisonOperation : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:489
   pragma Import (C, nppiThreshold_16s_AC4IR, "nppiThreshold_16s_AC4IR");

  --* 
  -- * 4 channel 32-bit floating point image threshold, not affecting Alpha.
  -- * If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel
  -- * value is set to nThreshold, otherwise it is set to sourcePixel.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \param eComparisonOperation The type of comparison operation to be used. The only valid
  -- *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
  -- * comparison operation type is specified.
  --  

   function nppiThreshold_32f_AC4R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp32f;
      eComparisonOperation : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:508
   pragma Import (C, nppiThreshold_32f_AC4R, "nppiThreshold_32f_AC4R");

  --* 
  -- * 4 channel 32-bit floating point in place image threshold, not affecting Alpha.
  -- * If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel
  -- * value is set to nThreshold, otherwise it is set to sourcePixel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \param eComparisonOperation The type of comparison operation to be used. The only valid
  -- *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
  -- * comparison operation type is specified.
  --  

   function nppiThreshold_32f_AC4IR
     (pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp32f;
      eComparisonOperation : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:526
   pragma Import (C, nppiThreshold_32f_AC4IR, "nppiThreshold_32f_AC4IR");

  --* 
  -- * 1 channel 8-bit unsigned char threshold.
  -- * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
  -- * to nThreshold, otherwise it is set to sourcePixel.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nThreshold The threshold value.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_GT_8u_C1R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nThreshold : nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:542
   pragma Import (C, nppiThreshold_GT_8u_C1R, "nppiThreshold_GT_8u_C1R");

  --* 
  -- * 1 channel 8-bit unsigned char in place threshold.
  -- * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
  -- * to nThreshold, otherwise it is set to sourcePixel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nThreshold The threshold value.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_GT_8u_C1IR
     (pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nThreshold : nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:557
   pragma Import (C, nppiThreshold_GT_8u_C1IR, "nppiThreshold_GT_8u_C1IR");

  --* 
  -- * 1 channel 16-bit unsigned short threshold.
  -- * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
  -- * to nThreshold, otherwise it is set to sourcePixel.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nThreshold The threshold value.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_GT_16u_C1R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nThreshold : nppdefs_h.Npp16u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:573
   pragma Import (C, nppiThreshold_GT_16u_C1R, "nppiThreshold_GT_16u_C1R");

  --* 
  -- * 1 channel 16-bit unsigned short in place threshold.
  -- * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
  -- * to nThreshold, otherwise it is set to sourcePixel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nThreshold The threshold value.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_GT_16u_C1IR
     (pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nThreshold : nppdefs_h.Npp16u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:588
   pragma Import (C, nppiThreshold_GT_16u_C1IR, "nppiThreshold_GT_16u_C1IR");

  --* 
  -- * 1 channel 16-bit signed short threshold.
  -- * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
  -- * to nThreshold, otherwise it is set to sourcePixel.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nThreshold The threshold value.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_GT_16s_C1R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nThreshold : nppdefs_h.Npp16s) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:604
   pragma Import (C, nppiThreshold_GT_16s_C1R, "nppiThreshold_GT_16s_C1R");

  --* 
  -- * 1 channel 16-bit signed short in place threshold.
  -- * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
  -- * to nThreshold, otherwise it is set to sourcePixel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nThreshold The threshold value.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_GT_16s_C1IR
     (pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nThreshold : nppdefs_h.Npp16s) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:619
   pragma Import (C, nppiThreshold_GT_16s_C1IR, "nppiThreshold_GT_16s_C1IR");

  --* 
  -- * 1 channel 32-bit floating point threshold.
  -- * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
  -- * to nThreshold, otherwise it is set to sourcePixel.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nThreshold The threshold value.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_GT_32f_C1R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nThreshold : nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:635
   pragma Import (C, nppiThreshold_GT_32f_C1R, "nppiThreshold_GT_32f_C1R");

  --* 
  -- * 1 channel 32-bit floating point in place threshold.
  -- * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
  -- * to nThreshold, otherwise it is set to sourcePixel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nThreshold The threshold value.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_GT_32f_C1IR
     (pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nThreshold : nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:650
   pragma Import (C, nppiThreshold_GT_32f_C1IR, "nppiThreshold_GT_32f_C1IR");

  --* 
  -- * 3 channel 8-bit unsigned char threshold.
  -- * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
  -- * to nThreshold, otherwise it is set to sourcePixel.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_GT_8u_C3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:666
   pragma Import (C, nppiThreshold_GT_8u_C3R, "nppiThreshold_GT_8u_C3R");

  --* 
  -- * 3 channel 8-bit unsigned char in place threshold.
  -- * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
  -- * to nThreshold, otherwise it is set to sourcePixel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_GT_8u_C3IR
     (pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:681
   pragma Import (C, nppiThreshold_GT_8u_C3IR, "nppiThreshold_GT_8u_C3IR");

  --* 
  -- * 3 channel 16-bit unsigned short threshold.
  -- * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
  -- * to nThreshold, otherwise it is set to sourcePixel.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_GT_16u_C3R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp16u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:697
   pragma Import (C, nppiThreshold_GT_16u_C3R, "nppiThreshold_GT_16u_C3R");

  --* 
  -- * 3 channel 16-bit unsigned short in place threshold.
  -- * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
  -- * to nThreshold, otherwise it is set to sourcePixel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_GT_16u_C3IR
     (pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp16u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:712
   pragma Import (C, nppiThreshold_GT_16u_C3IR, "nppiThreshold_GT_16u_C3IR");

  --* 
  -- * 3 channel 16-bit signed short threshold.
  -- * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
  -- * to nThreshold, otherwise it is set to sourcePixel.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_GT_16s_C3R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp16s) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:728
   pragma Import (C, nppiThreshold_GT_16s_C3R, "nppiThreshold_GT_16s_C3R");

  --* 
  -- * 3 channel 16-bit signed short in place threshold.
  -- * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
  -- * to nThreshold, otherwise it is set to sourcePixel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_GT_16s_C3IR
     (pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp16s) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:743
   pragma Import (C, nppiThreshold_GT_16s_C3IR, "nppiThreshold_GT_16s_C3IR");

  --* 
  -- * 3 channel 32-bit floating point threshold.
  -- * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
  -- * to nThreshold, otherwise it is set to sourcePixel.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_GT_32f_C3R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:759
   pragma Import (C, nppiThreshold_GT_32f_C3R, "nppiThreshold_GT_32f_C3R");

  --* 
  -- * 3 channel 32-bit floating point in place threshold.
  -- * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
  -- * to nThreshold, otherwise it is set to sourcePixel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_GT_32f_C3IR
     (pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:774
   pragma Import (C, nppiThreshold_GT_32f_C3IR, "nppiThreshold_GT_32f_C3IR");

  --* 
  -- * 4 channel 8-bit unsigned char image threshold, not affecting Alpha.
  -- * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
  -- * value is set to nThreshold, otherwise it is set to sourcePixel.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_GT_8u_AC4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:791
   pragma Import (C, nppiThreshold_GT_8u_AC4R, "nppiThreshold_GT_8u_AC4R");

  --* 
  -- * 4 channel 8-bit unsigned char in place image threshold, not affecting Alpha.
  -- * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
  -- * value is set to nThreshold, otherwise it is set to sourcePixel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_GT_8u_AC4IR
     (pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:806
   pragma Import (C, nppiThreshold_GT_8u_AC4IR, "nppiThreshold_GT_8u_AC4IR");

  --* 
  -- * 4 channel 16-bit unsigned short image threshold, not affecting Alpha.
  -- * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
  -- * value is set to nThreshold, otherwise it is set to sourcePixel.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_GT_16u_AC4R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp16u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:822
   pragma Import (C, nppiThreshold_GT_16u_AC4R, "nppiThreshold_GT_16u_AC4R");

  --* 
  -- * 4 channel 16-bit unsigned short in place image threshold, not affecting Alpha.
  -- * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
  -- * value is set to nThreshold, otherwise it is set to sourcePixel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_GT_16u_AC4IR
     (pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp16u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:837
   pragma Import (C, nppiThreshold_GT_16u_AC4IR, "nppiThreshold_GT_16u_AC4IR");

  --* 
  -- * 4 channel 16-bit signed short image threshold, not affecting Alpha.
  -- * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
  -- * value is set to nThreshold, otherwise it is set to sourcePixel.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_GT_16s_AC4R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp16s) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:853
   pragma Import (C, nppiThreshold_GT_16s_AC4R, "nppiThreshold_GT_16s_AC4R");

  --* 
  -- * 4 channel 16-bit signed short in place image threshold, not affecting Alpha.
  -- * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
  -- * value is set to nThreshold, otherwise it is set to sourcePixel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_GT_16s_AC4IR
     (pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp16s) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:868
   pragma Import (C, nppiThreshold_GT_16s_AC4IR, "nppiThreshold_GT_16s_AC4IR");

  --* 
  -- * 4 channel 32-bit floating point image threshold, not affecting Alpha.
  -- * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
  -- * value is set to nThreshold, otherwise it is set to sourcePixel.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_GT_32f_AC4R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:884
   pragma Import (C, nppiThreshold_GT_32f_AC4R, "nppiThreshold_GT_32f_AC4R");

  --* 
  -- * 4 channel 32-bit floating point in place image threshold, not affecting Alpha.
  -- * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
  -- * value is set to nThreshold, otherwise it is set to sourcePixel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_GT_32f_AC4IR
     (pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:899
   pragma Import (C, nppiThreshold_GT_32f_AC4IR, "nppiThreshold_GT_32f_AC4IR");

  --* 
  -- * 1 channel 8-bit unsigned char threshold.
  -- * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
  -- * to nThreshold, otherwise it is set to sourcePixel.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nThreshold The threshold value.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_LT_8u_C1R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nThreshold : nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:916
   pragma Import (C, nppiThreshold_LT_8u_C1R, "nppiThreshold_LT_8u_C1R");

  --* 
  -- * 1 channel 8-bit unsigned char in place threshold.
  -- * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
  -- * to nThreshold, otherwise it is set to sourcePixel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nThreshold The threshold value.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_LT_8u_C1IR
     (pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nThreshold : nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:931
   pragma Import (C, nppiThreshold_LT_8u_C1IR, "nppiThreshold_LT_8u_C1IR");

  --* 
  -- * 1 channel 16-bit unsigned short threshold.
  -- * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
  -- * to nThreshold, otherwise it is set to sourcePixel.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nThreshold The threshold value.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_LT_16u_C1R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nThreshold : nppdefs_h.Npp16u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:947
   pragma Import (C, nppiThreshold_LT_16u_C1R, "nppiThreshold_LT_16u_C1R");

  --* 
  -- * 1 channel 16-bit unsigned short in place threshold.
  -- * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
  -- * to nThreshold, otherwise it is set to sourcePixel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nThreshold The threshold value.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_LT_16u_C1IR
     (pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nThreshold : nppdefs_h.Npp16u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:962
   pragma Import (C, nppiThreshold_LT_16u_C1IR, "nppiThreshold_LT_16u_C1IR");

  --* 
  -- * 1 channel 16-bit signed short threshold.
  -- * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
  -- * to nThreshold, otherwise it is set to sourcePixel.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nThreshold The threshold value.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_LT_16s_C1R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nThreshold : nppdefs_h.Npp16s) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:978
   pragma Import (C, nppiThreshold_LT_16s_C1R, "nppiThreshold_LT_16s_C1R");

  --* 
  -- * 1 channel 16-bit signed short in place threshold.
  -- * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
  -- * to nThreshold, otherwise it is set to sourcePixel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nThreshold The threshold value.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_LT_16s_C1IR
     (pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nThreshold : nppdefs_h.Npp16s) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:993
   pragma Import (C, nppiThreshold_LT_16s_C1IR, "nppiThreshold_LT_16s_C1IR");

  --* 
  -- * 1 channel 32-bit floating point threshold.
  -- * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
  -- * to nThreshold, otherwise it is set to sourcePixel.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nThreshold The threshold value.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_LT_32f_C1R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nThreshold : nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:1009
   pragma Import (C, nppiThreshold_LT_32f_C1R, "nppiThreshold_LT_32f_C1R");

  --* 
  -- * 1 channel 32-bit floating point in place threshold.
  -- * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
  -- * to nThreshold, otherwise it is set to sourcePixel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nThreshold The threshold value.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_LT_32f_C1IR
     (pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nThreshold : nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:1024
   pragma Import (C, nppiThreshold_LT_32f_C1IR, "nppiThreshold_LT_32f_C1IR");

  --* 
  -- * 3 channel 8-bit unsigned char threshold.
  -- * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
  -- * to nThreshold, otherwise it is set to sourcePixel.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_LT_8u_C3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:1040
   pragma Import (C, nppiThreshold_LT_8u_C3R, "nppiThreshold_LT_8u_C3R");

  --* 
  -- * 3 channel 8-bit unsigned char in place threshold.
  -- * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
  -- * to nThreshold, otherwise it is set to sourcePixel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_LT_8u_C3IR
     (pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:1055
   pragma Import (C, nppiThreshold_LT_8u_C3IR, "nppiThreshold_LT_8u_C3IR");

  --* 
  -- * 3 channel 16-bit unsigned short threshold.
  -- * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
  -- * to nThreshold, otherwise it is set to sourcePixel.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_LT_16u_C3R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp16u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:1071
   pragma Import (C, nppiThreshold_LT_16u_C3R, "nppiThreshold_LT_16u_C3R");

  --* 
  -- * 3 channel 16-bit unsigned short in place threshold.
  -- * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
  -- * to nThreshold, otherwise it is set to sourcePixel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_LT_16u_C3IR
     (pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp16u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:1086
   pragma Import (C, nppiThreshold_LT_16u_C3IR, "nppiThreshold_LT_16u_C3IR");

  --* 
  -- * 3 channel 16-bit signed short threshold.
  -- * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
  -- * to nThreshold, otherwise it is set to sourcePixel.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_LT_16s_C3R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp16s) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:1102
   pragma Import (C, nppiThreshold_LT_16s_C3R, "nppiThreshold_LT_16s_C3R");

  --* 
  -- * 3 channel 16-bit signed short in place threshold.
  -- * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
  -- * to nThreshold, otherwise it is set to sourcePixel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_LT_16s_C3IR
     (pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp16s) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:1117
   pragma Import (C, nppiThreshold_LT_16s_C3IR, "nppiThreshold_LT_16s_C3IR");

  --* 
  -- * 3 channel 32-bit floating point threshold.
  -- * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
  -- * to nThreshold, otherwise it is set to sourcePixel.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_LT_32f_C3R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:1133
   pragma Import (C, nppiThreshold_LT_32f_C3R, "nppiThreshold_LT_32f_C3R");

  --* 
  -- * 3 channel 32-bit floating point in place threshold.
  -- * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
  -- * to nThreshold, otherwise it is set to sourcePixel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_LT_32f_C3IR
     (pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:1148
   pragma Import (C, nppiThreshold_LT_32f_C3IR, "nppiThreshold_LT_32f_C3IR");

  --* 
  -- * 4 channel 8-bit unsigned char image threshold, not affecting Alpha.
  -- * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
  -- * value is set to nThreshold, otherwise it is set to sourcePixel.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_LT_8u_AC4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:1165
   pragma Import (C, nppiThreshold_LT_8u_AC4R, "nppiThreshold_LT_8u_AC4R");

  --* 
  -- * 4 channel 8-bit unsigned char in place image threshold, not affecting Alpha.
  -- * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
  -- * value is set to nThreshold, otherwise it is set to sourcePixel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_LT_8u_AC4IR
     (pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:1180
   pragma Import (C, nppiThreshold_LT_8u_AC4IR, "nppiThreshold_LT_8u_AC4IR");

  --* 
  -- * 4 channel 16-bit unsigned short image threshold, not affecting Alpha.
  -- * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
  -- * value is set to nThreshold, otherwise it is set to sourcePixel.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_LT_16u_AC4R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp16u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:1196
   pragma Import (C, nppiThreshold_LT_16u_AC4R, "nppiThreshold_LT_16u_AC4R");

  --* 
  -- * 4 channel 16-bit unsigned short in place image threshold, not affecting Alpha.
  -- * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
  -- * value is set to nThreshold, otherwise it is set to sourcePixel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_LT_16u_AC4IR
     (pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp16u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:1211
   pragma Import (C, nppiThreshold_LT_16u_AC4IR, "nppiThreshold_LT_16u_AC4IR");

  --* 
  -- * 4 channel 16-bit signed short image threshold, not affecting Alpha.
  -- * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
  -- * value is set to nThreshold, otherwise it is set to sourcePixel.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_LT_16s_AC4R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp16s) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:1227
   pragma Import (C, nppiThreshold_LT_16s_AC4R, "nppiThreshold_LT_16s_AC4R");

  --* 
  -- * 4 channel 16-bit signed short in place image threshold, not affecting Alpha.
  -- * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
  -- * value is set to nThreshold, otherwise it is set to sourcePixel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_LT_16s_AC4IR
     (pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp16s) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:1242
   pragma Import (C, nppiThreshold_LT_16s_AC4IR, "nppiThreshold_LT_16s_AC4IR");

  --* 
  -- * 4 channel 32-bit floating point image threshold, not affecting Alpha.
  -- * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
  -- * value is set to nThreshold, otherwise it is set to sourcePixel.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_LT_32f_AC4R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:1258
   pragma Import (C, nppiThreshold_LT_32f_AC4R, "nppiThreshold_LT_32f_AC4R");

  --* 
  -- * 4 channel 32-bit floating point in place image threshold, not affecting Alpha.
  -- * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
  -- * value is set to nThreshold, otherwise it is set to sourcePixel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_LT_32f_AC4IR
     (pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:1273
   pragma Import (C, nppiThreshold_LT_32f_AC4IR, "nppiThreshold_LT_32f_AC4IR");

  --* 
  -- * 1 channel 8-bit unsigned char threshold.
  -- * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
  -- * to nValue, otherwise it is set to sourcePixel.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nThreshold The threshold value.
  -- * \param nValue The threshold replacement value.
  -- * \param eComparisonOperation The type of comparison operation to be used. The only valid
  -- *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
  -- * comparison operation type is specified.
  --  

   function nppiThreshold_Val_8u_C1R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nThreshold : nppdefs_h.Npp8u;
      nValue : nppdefs_h.Npp8u;
      eComparisonOperation : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:1294
   pragma Import (C, nppiThreshold_Val_8u_C1R, "nppiThreshold_Val_8u_C1R");

  --* 
  -- * 1 channel 8-bit unsigned char in place threshold.
  -- * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
  -- * to nValue, otherwise it is set to sourcePixel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nThreshold The threshold value.
  -- * \param nValue The threshold replacement value.
  -- * \param eComparisonOperation The type of comparison operation to be used. The only valid
  -- *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
  -- * comparison operation type is specified.
  --  

   function nppiThreshold_Val_8u_C1IR
     (pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nThreshold : nppdefs_h.Npp8u;
      nValue : nppdefs_h.Npp8u;
      eComparisonOperation : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:1313
   pragma Import (C, nppiThreshold_Val_8u_C1IR, "nppiThreshold_Val_8u_C1IR");

  --* 
  -- * 1 channel 16-bit unsigned short threshold.
  -- * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
  -- * to nValue, otherwise it is set to sourcePixel.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nThreshold The threshold value.
  -- * \param nValue The threshold replacement value.
  -- * \param eComparisonOperation The type of comparison operation to be used. The only valid
  -- *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
  -- * comparison operation type is specified.
  --  

   function nppiThreshold_Val_16u_C1R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nThreshold : nppdefs_h.Npp16u;
      nValue : nppdefs_h.Npp16u;
      eComparisonOperation : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:1333
   pragma Import (C, nppiThreshold_Val_16u_C1R, "nppiThreshold_Val_16u_C1R");

  --* 
  -- * 1 channel 16-bit unsigned short in place threshold.
  -- * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
  -- * to nValue, otherwise it is set to sourcePixel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nThreshold The threshold value.
  -- * \param nValue The threshold replacement value.
  -- * \param eComparisonOperation The type of comparison operation to be used. The only valid
  -- *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
  -- * comparison operation type is specified.
  --  

   function nppiThreshold_Val_16u_C1IR
     (pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nThreshold : nppdefs_h.Npp16u;
      nValue : nppdefs_h.Npp16u;
      eComparisonOperation : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:1352
   pragma Import (C, nppiThreshold_Val_16u_C1IR, "nppiThreshold_Val_16u_C1IR");

  --* 
  -- * 1 channel 16-bit signed short threshold.
  -- * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
  -- * to nValue, otherwise it is set to sourcePixel.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nThreshold The threshold value.
  -- * \param nValue The threshold replacement value.
  -- * \param eComparisonOperation The type of comparison operation to be used. The only valid
  -- *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
  -- * comparison operation type is specified.
  --  

   function nppiThreshold_Val_16s_C1R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nThreshold : nppdefs_h.Npp16s;
      nValue : nppdefs_h.Npp16s;
      eComparisonOperation : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:1372
   pragma Import (C, nppiThreshold_Val_16s_C1R, "nppiThreshold_Val_16s_C1R");

  --* 
  -- * 1 channel 16-bit signed short in place threshold.
  -- * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
  -- * to nValue, otherwise it is set to sourcePixel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nThreshold The threshold value.
  -- * \param nValue The threshold replacement value.
  -- * \param eComparisonOperation The type of comparison operation to be used. The only valid
  -- *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
  -- * comparison operation type is specified.
  --  

   function nppiThreshold_Val_16s_C1IR
     (pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nThreshold : nppdefs_h.Npp16s;
      nValue : nppdefs_h.Npp16s;
      eComparisonOperation : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:1391
   pragma Import (C, nppiThreshold_Val_16s_C1IR, "nppiThreshold_Val_16s_C1IR");

  --* 
  -- * 1 channel 32-bit floating point threshold.
  -- * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
  -- * to nValue, otherwise it is set to sourcePixel.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nThreshold The threshold value.
  -- * \param nValue The threshold replacement value.
  -- * \param eComparisonOperation The type of comparison operation to be used. The only valid
  -- *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
  -- * comparison operation type is specified.
  --  

   function nppiThreshold_Val_32f_C1R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nThreshold : nppdefs_h.Npp32f;
      nValue : nppdefs_h.Npp32f;
      eComparisonOperation : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:1411
   pragma Import (C, nppiThreshold_Val_32f_C1R, "nppiThreshold_Val_32f_C1R");

  --* 
  -- * 1 channel 32-bit floating point in place threshold.
  -- * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
  -- * to nValue, otherwise it is set to sourcePixel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nThreshold The threshold value.
  -- * \param nValue The threshold replacement value.
  -- * \param eComparisonOperation The type of comparison operation to be used. The only valid
  -- *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
  -- * comparison operation type is specified.
  --  

   function nppiThreshold_Val_32f_C1IR
     (pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nThreshold : nppdefs_h.Npp32f;
      nValue : nppdefs_h.Npp32f;
      eComparisonOperation : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:1430
   pragma Import (C, nppiThreshold_Val_32f_C1IR, "nppiThreshold_Val_32f_C1IR");

  --* 
  -- * 3 channel 8-bit unsigned char threshold.
  -- * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
  -- * to nValue, otherwise it is set to sourcePixel.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \param rValues The threshold replacement values, one per color channel.
  -- * \param eComparisonOperation The type of comparison operation to be used. The only valid
  -- *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
  -- * comparison operation type is specified.
  --  

   function nppiThreshold_Val_8u_C3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp8u;
      rValues : access nppdefs_h.Npp8u;
      eComparisonOperation : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:1450
   pragma Import (C, nppiThreshold_Val_8u_C3R, "nppiThreshold_Val_8u_C3R");

  --* 
  -- * 3 channel 8-bit unsigned char in place threshold.
  -- * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
  -- * to nValue, otherwise it is set to sourcePixel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \param rValues The threshold replacement values, one per color channel.
  -- * \param eComparisonOperation The type of comparison operation to be used. The only valid
  -- *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
  -- * comparison operation type is specified.
  --  

   function nppiThreshold_Val_8u_C3IR
     (pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp8u;
      rValues : access nppdefs_h.Npp8u;
      eComparisonOperation : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:1469
   pragma Import (C, nppiThreshold_Val_8u_C3IR, "nppiThreshold_Val_8u_C3IR");

  --* 
  -- * 3 channel 16-bit unsigned short threshold.
  -- * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
  -- * to nValue, otherwise it is set to sourcePixel.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \param rValues The threshold replacement values, one per color channel.
  -- * \param eComparisonOperation The type of comparison operation to be used. The only valid
  -- *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
  -- * comparison operation type is specified.
  --  

   function nppiThreshold_Val_16u_C3R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp16u;
      rValues : access nppdefs_h.Npp16u;
      eComparisonOperation : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:1489
   pragma Import (C, nppiThreshold_Val_16u_C3R, "nppiThreshold_Val_16u_C3R");

  --* 
  -- * 3 channel 16-bit unsigned short in place threshold.
  -- * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
  -- * to nValue, otherwise it is set to sourcePixel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \param rValues The threshold replacement values, one per color channel.
  -- * \param eComparisonOperation The type of comparison operation to be used. The only valid
  -- *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
  -- * comparison operation type is specified.
  --  

   function nppiThreshold_Val_16u_C3IR
     (pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp16u;
      rValues : access nppdefs_h.Npp16u;
      eComparisonOperation : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:1508
   pragma Import (C, nppiThreshold_Val_16u_C3IR, "nppiThreshold_Val_16u_C3IR");

  --* 
  -- * 3 channel 16-bit signed short threshold.
  -- * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
  -- * to nValue, otherwise it is set to sourcePixel.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \param rValues The threshold replacement values, one per color channel.
  -- * \param eComparisonOperation The type of comparison operation to be used. The only valid
  -- *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
  -- * comparison operation type is specified.
  --  

   function nppiThreshold_Val_16s_C3R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp16s;
      rValues : access nppdefs_h.Npp16s;
      eComparisonOperation : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:1528
   pragma Import (C, nppiThreshold_Val_16s_C3R, "nppiThreshold_Val_16s_C3R");

  --* 
  -- * 3 channel 16-bit signed short in place threshold.
  -- * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
  -- * to nValue, otherwise it is set to sourcePixel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \param rValues The threshold replacement values, one per color channel.
  -- * \param eComparisonOperation The type of comparison operation to be used. The only valid
  -- *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
  -- * comparison operation type is specified.
  --  

   function nppiThreshold_Val_16s_C3IR
     (pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp16s;
      rValues : access nppdefs_h.Npp16s;
      eComparisonOperation : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:1547
   pragma Import (C, nppiThreshold_Val_16s_C3IR, "nppiThreshold_Val_16s_C3IR");

  --* 
  -- * 3 channel 32-bit floating point threshold.
  -- * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
  -- * to nValue, otherwise it is set to sourcePixel.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \param rValues The threshold replacement values, one per color channel.
  -- * \param eComparisonOperation The type of comparison operation to be used. The only valid
  -- *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
  -- * comparison operation type is specified.
  --  

   function nppiThreshold_Val_32f_C3R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp32f;
      rValues : access nppdefs_h.Npp32f;
      eComparisonOperation : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:1567
   pragma Import (C, nppiThreshold_Val_32f_C3R, "nppiThreshold_Val_32f_C3R");

  --* 
  -- * 3 channel 32-bit floating point in place threshold.
  -- * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
  -- * to nValue, otherwise it is set to sourcePixel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \param rValues The threshold replacement values, one per color channel.
  -- * \param eComparisonOperation The type of comparison operation to be used. The only valid
  -- *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
  -- * comparison operation type is specified.
  --  

   function nppiThreshold_Val_32f_C3IR
     (pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp32f;
      rValues : access nppdefs_h.Npp32f;
      eComparisonOperation : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:1586
   pragma Import (C, nppiThreshold_Val_32f_C3IR, "nppiThreshold_Val_32f_C3IR");

  --* 
  -- * 4 channel 8-bit unsigned char image threshold, not affecting Alpha.
  -- * If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel
  -- * value is set to nValue, otherwise it is set to sourcePixel.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \param rValues The threshold replacement values, one per color channel.
  -- * \param eComparisonOperation The type of comparison operation to be used. The only valid
  -- *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
  -- * comparison operation type is specified.
  --  

   function nppiThreshold_Val_8u_AC4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp8u;
      rValues : access nppdefs_h.Npp8u;
      eComparisonOperation : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:1607
   pragma Import (C, nppiThreshold_Val_8u_AC4R, "nppiThreshold_Val_8u_AC4R");

  --* 
  -- * 4 channel 8-bit unsigned char in place image threshold, not affecting Alpha.
  -- * If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel
  -- * value is set to nValue, otherwise it is set to sourcePixel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \param rValues The threshold replacement values, one per color channel.
  -- * \param eComparisonOperation The type of comparison operation to be used. The only valid
  -- *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
  -- * comparison operation type is specified.
  --  

   function nppiThreshold_Val_8u_AC4IR
     (pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp8u;
      rValues : access nppdefs_h.Npp8u;
      eComparisonOperation : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:1626
   pragma Import (C, nppiThreshold_Val_8u_AC4IR, "nppiThreshold_Val_8u_AC4IR");

  --* 
  -- * 4 channel 16-bit unsigned short image threshold, not affecting Alpha.
  -- * If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel
  -- * value is set to nValue, otherwise it is set to sourcePixel.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \param rValues The threshold replacement values, one per color channel.
  -- * \param eComparisonOperation The type of comparison operation to be used. The only valid
  -- *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
  -- * comparison operation type is specified.
  --  

   function nppiThreshold_Val_16u_AC4R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp16u;
      rValues : access nppdefs_h.Npp16u;
      eComparisonOperation : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:1646
   pragma Import (C, nppiThreshold_Val_16u_AC4R, "nppiThreshold_Val_16u_AC4R");

  --* 
  -- * 4 channel 16-bit unsigned short in place image threshold, not affecting Alpha.
  -- * If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel
  -- * value is set to nValue, otherwise it is set to sourcePixel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \param rValues The threshold replacement values, one per color channel.
  -- * \param eComparisonOperation The type of comparison operation to be used. The only valid
  -- *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
  -- * comparison operation type is specified.
  --  

   function nppiThreshold_Val_16u_AC4IR
     (pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp16u;
      rValues : access nppdefs_h.Npp16u;
      eComparisonOperation : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:1665
   pragma Import (C, nppiThreshold_Val_16u_AC4IR, "nppiThreshold_Val_16u_AC4IR");

  --* 
  -- * 4 channel 16-bit signed short image threshold, not affecting Alpha.
  -- * If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel
  -- * value is set to nValue, otherwise it is set to sourcePixel.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \param rValues The threshold replacement values, one per color channel.
  -- * \param eComparisonOperation The type of comparison operation to be used. The only valid
  -- *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
  -- * comparison operation type is specified.
  --  

   function nppiThreshold_Val_16s_AC4R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp16s;
      rValues : access nppdefs_h.Npp16s;
      eComparisonOperation : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:1685
   pragma Import (C, nppiThreshold_Val_16s_AC4R, "nppiThreshold_Val_16s_AC4R");

  --* 
  -- * 4 channel 16-bit signed short in place image threshold, not affecting Alpha.
  -- * If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel
  -- * value is set to nValue, otherwise it is set to sourcePixel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \param rValues The threshold replacement values, one per color channel.
  -- * \param eComparisonOperation The type of comparison operation to be used. The only valid
  -- *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
  -- * comparison operation type is specified.
  --  

   function nppiThreshold_Val_16s_AC4IR
     (pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp16s;
      rValues : access nppdefs_h.Npp16s;
      eComparisonOperation : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:1704
   pragma Import (C, nppiThreshold_Val_16s_AC4IR, "nppiThreshold_Val_16s_AC4IR");

  --* 
  -- * 4 channel 32-bit floating point image threshold, not affecting Alpha.
  -- * If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel
  -- * value is set to nValue, otherwise it is set to sourcePixel.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \param rValues The threshold replacement values, one per color channel.
  -- * \param eComparisonOperation The type of comparison operation to be used. The only valid
  -- *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
  -- * comparison operation type is specified.
  --  

   function nppiThreshold_Val_32f_AC4R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp32f;
      rValues : access nppdefs_h.Npp32f;
      eComparisonOperation : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:1724
   pragma Import (C, nppiThreshold_Val_32f_AC4R, "nppiThreshold_Val_32f_AC4R");

  --* 
  -- * 4 channel 32-bit floating point in place image threshold, not affecting Alpha.
  -- * If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel
  -- * value is set to nValue, otherwise it is set to sourcePixel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \param rValues The threshold replacement values, one per color channel.
  -- * \param eComparisonOperation The type of comparison operation to be used. The only valid
  -- *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
  -- * comparison operation type is specified.
  --  

   function nppiThreshold_Val_32f_AC4IR
     (pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp32f;
      rValues : access nppdefs_h.Npp32f;
      eComparisonOperation : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:1743
   pragma Import (C, nppiThreshold_Val_32f_AC4IR, "nppiThreshold_Val_32f_AC4IR");

  --* 
  -- * 1 channel 8-bit unsigned char threshold.
  -- * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
  -- * to nValue, otherwise it is set to sourcePixel.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nThreshold The threshold value.
  -- * \param nValue The threshold replacement value.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_GTVal_8u_C1R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nThreshold : nppdefs_h.Npp8u;
      nValue : nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:1760
   pragma Import (C, nppiThreshold_GTVal_8u_C1R, "nppiThreshold_GTVal_8u_C1R");

  --* 
  -- * 1 channel 8-bit unsigned char in place threshold.
  -- * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
  -- * to nValue, otherwise it is set to sourcePixel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nThreshold The threshold value.
  -- * \param nValue The threshold replacement value.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_GTVal_8u_C1IR
     (pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nThreshold : nppdefs_h.Npp8u;
      nValue : nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:1776
   pragma Import (C, nppiThreshold_GTVal_8u_C1IR, "nppiThreshold_GTVal_8u_C1IR");

  --* 
  -- * 1 channel 16-bit unsigned short threshold.
  -- * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
  -- * to nValue, otherwise it is set to sourcePixel.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nThreshold The threshold value.
  -- * \param nValue The threshold replacement value.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_GTVal_16u_C1R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nThreshold : nppdefs_h.Npp16u;
      nValue : nppdefs_h.Npp16u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:1793
   pragma Import (C, nppiThreshold_GTVal_16u_C1R, "nppiThreshold_GTVal_16u_C1R");

  --* 
  -- * 1 channel 16-bit unsigned short in place threshold.
  -- * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
  -- * to nValue, otherwise it is set to sourcePixel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nThreshold The threshold value.
  -- * \param nValue The threshold replacement value.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_GTVal_16u_C1IR
     (pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nThreshold : nppdefs_h.Npp16u;
      nValue : nppdefs_h.Npp16u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:1809
   pragma Import (C, nppiThreshold_GTVal_16u_C1IR, "nppiThreshold_GTVal_16u_C1IR");

  --* 
  -- * 1 channel 16-bit signed short threshold.
  -- * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
  -- * to nValue, otherwise it is set to sourcePixel.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nThreshold The threshold value.
  -- * \param nValue The threshold replacement value.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_GTVal_16s_C1R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nThreshold : nppdefs_h.Npp16s;
      nValue : nppdefs_h.Npp16s) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:1826
   pragma Import (C, nppiThreshold_GTVal_16s_C1R, "nppiThreshold_GTVal_16s_C1R");

  --* 
  -- * 1 channel 16-bit signed short in place threshold.
  -- * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
  -- * to nValue, otherwise it is set to sourcePixel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nThreshold The threshold value.
  -- * \param nValue The threshold replacement value.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_GTVal_16s_C1IR
     (pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nThreshold : nppdefs_h.Npp16s;
      nValue : nppdefs_h.Npp16s) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:1842
   pragma Import (C, nppiThreshold_GTVal_16s_C1IR, "nppiThreshold_GTVal_16s_C1IR");

  --* 
  -- * 1 channel 32-bit floating point threshold.
  -- * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
  -- * to nValue, otherwise it is set to sourcePixel.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nThreshold The threshold value.
  -- * \param nValue The threshold replacement value.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_GTVal_32f_C1R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nThreshold : nppdefs_h.Npp32f;
      nValue : nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:1859
   pragma Import (C, nppiThreshold_GTVal_32f_C1R, "nppiThreshold_GTVal_32f_C1R");

  --* 
  -- * 1 channel 32-bit floating point in place threshold.
  -- * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
  -- * to nValue, otherwise it is set to sourcePixel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nThreshold The threshold value.
  -- * \param nValue The threshold replacement values.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_GTVal_32f_C1IR
     (pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nThreshold : nppdefs_h.Npp32f;
      nValue : nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:1875
   pragma Import (C, nppiThreshold_GTVal_32f_C1IR, "nppiThreshold_GTVal_32f_C1IR");

  --* 
  -- * 3 channel 8-bit unsigned char threshold.
  -- * If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set
  -- * to rValue, otherwise it is set to sourcePixel.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \param rValues The threshold replacement values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_GTVal_8u_C3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp8u;
      rValues : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:1892
   pragma Import (C, nppiThreshold_GTVal_8u_C3R, "nppiThreshold_GTVal_8u_C3R");

  --* 
  -- * 3 channel 8-bit unsigned char in place threshold.
  -- * If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set
  -- * to rValue, otherwise it is set to sourcePixel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \param rValues The threshold replacement values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_GTVal_8u_C3IR
     (pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp8u;
      rValues : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:1908
   pragma Import (C, nppiThreshold_GTVal_8u_C3IR, "nppiThreshold_GTVal_8u_C3IR");

  --* 
  -- * 3 channel 16-bit unsigned short threshold.
  -- * If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set
  -- * to rValue, otherwise it is set to sourcePixel.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \param rValues The threshold replacement values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_GTVal_16u_C3R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp16u;
      rValues : access nppdefs_h.Npp16u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:1925
   pragma Import (C, nppiThreshold_GTVal_16u_C3R, "nppiThreshold_GTVal_16u_C3R");

  --* 
  -- * 3 channel 16-bit unsigned short in place threshold.
  -- * If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set
  -- * to rValue, otherwise it is set to sourcePixel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \param rValues The threshold replacement values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_GTVal_16u_C3IR
     (pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp16u;
      rValues : access nppdefs_h.Npp16u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:1941
   pragma Import (C, nppiThreshold_GTVal_16u_C3IR, "nppiThreshold_GTVal_16u_C3IR");

  --* 
  -- * 3 channel 16-bit signed short threshold.
  -- * If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set
  -- * to rValue, otherwise it is set to sourcePixel.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \param rValues The threshold replacement values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_GTVal_16s_C3R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp16s;
      rValues : access nppdefs_h.Npp16s) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:1958
   pragma Import (C, nppiThreshold_GTVal_16s_C3R, "nppiThreshold_GTVal_16s_C3R");

  --* 
  -- * 3 channel 16-bit signed short in place threshold.
  -- * If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set
  -- * to rValue, otherwise it is set to sourcePixel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \param rValues The threshold replacement values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_GTVal_16s_C3IR
     (pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp16s;
      rValues : access nppdefs_h.Npp16s) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:1974
   pragma Import (C, nppiThreshold_GTVal_16s_C3IR, "nppiThreshold_GTVal_16s_C3IR");

  --* 
  -- * 3 channel 32-bit floating point threshold.
  -- * If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set
  -- * to rValue, otherwise it is set to sourcePixel.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \param rValues The threshold replacement values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_GTVal_32f_C3R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp32f;
      rValues : access nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:1991
   pragma Import (C, nppiThreshold_GTVal_32f_C3R, "nppiThreshold_GTVal_32f_C3R");

  --* 
  -- * 3 channel 32-bit floating point in place threshold.
  -- * If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set
  -- * to rValue, otherwise it is set to sourcePixel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \param rValues The threshold replacement values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_GTVal_32f_C3IR
     (pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp32f;
      rValues : access nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:2007
   pragma Import (C, nppiThreshold_GTVal_32f_C3IR, "nppiThreshold_GTVal_32f_C3IR");

  --* 
  -- * 4 channel 8-bit unsigned char image threshold, not affecting Alpha.
  -- * If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set
  -- * value is set to rValue, otherwise it is set to sourcePixel.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \param rValues The threshold replacement values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_GTVal_8u_AC4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp8u;
      rValues : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:2024
   pragma Import (C, nppiThreshold_GTVal_8u_AC4R, "nppiThreshold_GTVal_8u_AC4R");

  --* 
  -- * 4 channel 8-bit unsigned char in place image threshold, not affecting Alpha.
  -- * If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set
  -- * value is set to rValue, otherwise it is set to sourcePixel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \param rValues The threshold replacement values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_GTVal_8u_AC4IR
     (pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp8u;
      rValues : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:2040
   pragma Import (C, nppiThreshold_GTVal_8u_AC4IR, "nppiThreshold_GTVal_8u_AC4IR");

  --* 
  -- * 4 channel 16-bit unsigned short image threshold, not affecting Alpha.
  -- * If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set
  -- * value is set to rValue, otherwise it is set to sourcePixel.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \param rValues The threshold replacement values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_GTVal_16u_AC4R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp16u;
      rValues : access nppdefs_h.Npp16u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:2057
   pragma Import (C, nppiThreshold_GTVal_16u_AC4R, "nppiThreshold_GTVal_16u_AC4R");

  --* 
  -- * 4 channel 16-bit unsigned short in place image threshold, not affecting Alpha.
  -- * If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set
  -- * value is set to rValue, otherwise it is set to sourcePixel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \param rValues The threshold replacement values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_GTVal_16u_AC4IR
     (pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp16u;
      rValues : access nppdefs_h.Npp16u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:2073
   pragma Import (C, nppiThreshold_GTVal_16u_AC4IR, "nppiThreshold_GTVal_16u_AC4IR");

  --* 
  -- * 4 channel 16-bit signed short image threshold, not affecting Alpha.
  -- * If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set
  -- * value is set to rValue, otherwise it is set to sourcePixel.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \param rValues The threshold replacement values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_GTVal_16s_AC4R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp16s;
      rValues : access nppdefs_h.Npp16s) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:2090
   pragma Import (C, nppiThreshold_GTVal_16s_AC4R, "nppiThreshold_GTVal_16s_AC4R");

  --* 
  -- * 4 channel 16-bit signed short in place image threshold, not affecting Alpha.
  -- * If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set
  -- * value is set to rValue, otherwise it is set to sourcePixel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \param rValues The threshold replacement values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_GTVal_16s_AC4IR
     (pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp16s;
      rValues : access nppdefs_h.Npp16s) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:2106
   pragma Import (C, nppiThreshold_GTVal_16s_AC4IR, "nppiThreshold_GTVal_16s_AC4IR");

  --* 
  -- * 4 channel 32-bit floating point image threshold, not affecting Alpha.
  -- * If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set
  -- * value is set to rValue, otherwise it is set to sourcePixel.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \param rValues The threshold replacement values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_GTVal_32f_AC4R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp32f;
      rValues : access nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:2123
   pragma Import (C, nppiThreshold_GTVal_32f_AC4R, "nppiThreshold_GTVal_32f_AC4R");

  --* 
  -- * 4 channel 32-bit floating point in place image threshold, not affecting Alpha.
  -- * If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set
  -- * value is set to rValue, otherwise it is set to sourcePixel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \param rValues The threshold replacement values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_GTVal_32f_AC4IR
     (pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp32f;
      rValues : access nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:2139
   pragma Import (C, nppiThreshold_GTVal_32f_AC4IR, "nppiThreshold_GTVal_32f_AC4IR");

  --* 
  -- * 1 channel 8-bit unsigned char threshold.
  -- * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
  -- * to nValue, otherwise it is set to sourcePixel.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nThreshold The threshold value.
  -- * \param nValue The threshold replacement value.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_LTVal_8u_C1R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nThreshold : nppdefs_h.Npp8u;
      nValue : nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:2157
   pragma Import (C, nppiThreshold_LTVal_8u_C1R, "nppiThreshold_LTVal_8u_C1R");

  --* 
  -- * 1 channel 8-bit unsigned char in place threshold.
  -- * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
  -- * to nValue, otherwise it is set to sourcePixel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nThreshold The threshold value.
  -- * \param nValue The threshold replacement value.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_LTVal_8u_C1IR
     (pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nThreshold : nppdefs_h.Npp8u;
      nValue : nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:2173
   pragma Import (C, nppiThreshold_LTVal_8u_C1IR, "nppiThreshold_LTVal_8u_C1IR");

  --* 
  -- * 1 channel 16-bit unsigned short threshold.
  -- * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
  -- * to nValue, otherwise it is set to sourcePixel.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nThreshold The threshold value.
  -- * \param nValue The threshold replacement value.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_LTVal_16u_C1R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nThreshold : nppdefs_h.Npp16u;
      nValue : nppdefs_h.Npp16u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:2190
   pragma Import (C, nppiThreshold_LTVal_16u_C1R, "nppiThreshold_LTVal_16u_C1R");

  --* 
  -- * 1 channel 16-bit unsigned short in place threshold.
  -- * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
  -- * to nValue, otherwise it is set to sourcePixel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nThreshold The threshold value.
  -- * \param nValue The threshold replacement value.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_LTVal_16u_C1IR
     (pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nThreshold : nppdefs_h.Npp16u;
      nValue : nppdefs_h.Npp16u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:2206
   pragma Import (C, nppiThreshold_LTVal_16u_C1IR, "nppiThreshold_LTVal_16u_C1IR");

  --* 
  -- * 1 channel 16-bit signed short threshold.
  -- * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
  -- * to nValue, otherwise it is set to sourcePixel.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nThreshold The threshold value.
  -- * \param nValue The threshold replacement value.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_LTVal_16s_C1R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nThreshold : nppdefs_h.Npp16s;
      nValue : nppdefs_h.Npp16s) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:2223
   pragma Import (C, nppiThreshold_LTVal_16s_C1R, "nppiThreshold_LTVal_16s_C1R");

  --* 
  -- * 1 channel 16-bit signed short in place threshold.
  -- * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
  -- * to nValue, otherwise it is set to sourcePixel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nThreshold The threshold value.
  -- * \param nValue The threshold replacement value.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_LTVal_16s_C1IR
     (pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nThreshold : nppdefs_h.Npp16s;
      nValue : nppdefs_h.Npp16s) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:2239
   pragma Import (C, nppiThreshold_LTVal_16s_C1IR, "nppiThreshold_LTVal_16s_C1IR");

  --* 
  -- * 1 channel 32-bit floating point threshold.
  -- * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
  -- * to nValue, otherwise it is set to sourcePixel.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nThreshold The threshold value.
  -- * \param nValue The threshold replacement value.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_LTVal_32f_C1R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nThreshold : nppdefs_h.Npp32f;
      nValue : nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:2256
   pragma Import (C, nppiThreshold_LTVal_32f_C1R, "nppiThreshold_LTVal_32f_C1R");

  --* 
  -- * 1 channel 32-bit floating point in place threshold.
  -- * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
  -- * to nValue, otherwise it is set to sourcePixel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nThreshold The threshold value.
  -- * \param nValue The threshold replacement value.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_LTVal_32f_C1IR
     (pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nThreshold : nppdefs_h.Npp32f;
      nValue : nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:2272
   pragma Import (C, nppiThreshold_LTVal_32f_C1IR, "nppiThreshold_LTVal_32f_C1IR");

  --* 
  -- * 3 channel 8-bit unsigned char threshold.
  -- * If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set
  -- * to rValue, otherwise it is set to sourcePixel.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \param rValues The threshold replacement values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_LTVal_8u_C3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp8u;
      rValues : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:2289
   pragma Import (C, nppiThreshold_LTVal_8u_C3R, "nppiThreshold_LTVal_8u_C3R");

  --* 
  -- * 3 channel 8-bit unsigned char in place threshold.
  -- * If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set
  -- * to rValue, otherwise it is set to sourcePixel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \param rValues The threshold replacement values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_LTVal_8u_C3IR
     (pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp8u;
      rValues : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:2305
   pragma Import (C, nppiThreshold_LTVal_8u_C3IR, "nppiThreshold_LTVal_8u_C3IR");

  --* 
  -- * 3 channel 16-bit unsigned short threshold.
  -- * If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set
  -- * to rValue, otherwise it is set to sourcePixel.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \param rValues The threshold replacement values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_LTVal_16u_C3R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp16u;
      rValues : access nppdefs_h.Npp16u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:2322
   pragma Import (C, nppiThreshold_LTVal_16u_C3R, "nppiThreshold_LTVal_16u_C3R");

  --* 
  -- * 3 channel 16-bit unsigned short in place threshold.
  -- * If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set
  -- * to rValue, otherwise it is set to sourcePixel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \param rValues The threshold replacement values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_LTVal_16u_C3IR
     (pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp16u;
      rValues : access nppdefs_h.Npp16u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:2338
   pragma Import (C, nppiThreshold_LTVal_16u_C3IR, "nppiThreshold_LTVal_16u_C3IR");

  --* 
  -- * 3 channel 16-bit signed short threshold.
  -- * If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set
  -- * to rValue, otherwise it is set to sourcePixel.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \param rValues The threshold replacement values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_LTVal_16s_C3R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp16s;
      rValues : access nppdefs_h.Npp16s) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:2355
   pragma Import (C, nppiThreshold_LTVal_16s_C3R, "nppiThreshold_LTVal_16s_C3R");

  --* 
  -- * 3 channel 16-bit signed short in place threshold.
  -- * If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set
  -- * to rValue, otherwise it is set to sourcePixel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \param rValues The threshold replacement values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_LTVal_16s_C3IR
     (pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp16s;
      rValues : access nppdefs_h.Npp16s) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:2371
   pragma Import (C, nppiThreshold_LTVal_16s_C3IR, "nppiThreshold_LTVal_16s_C3IR");

  --* 
  -- * 3 channel 32-bit floating point threshold.
  -- * If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set
  -- * to rValue, otherwise it is set to sourcePixel.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \param rValues The threshold replacement values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_LTVal_32f_C3R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp32f;
      rValues : access nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:2388
   pragma Import (C, nppiThreshold_LTVal_32f_C3R, "nppiThreshold_LTVal_32f_C3R");

  --* 
  -- * 3 channel 32-bit floating point in place threshold.
  -- * If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set
  -- * to rValue, otherwise it is set to sourcePixel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \param rValues The threshold replacement values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_LTVal_32f_C3IR
     (pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp32f;
      rValues : access nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:2404
   pragma Import (C, nppiThreshold_LTVal_32f_C3IR, "nppiThreshold_LTVal_32f_C3IR");

  --* 
  -- * 4 channel 8-bit unsigned char image threshold, not affecting Alpha.
  -- * If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set
  -- * value is set to rValue, otherwise it is set to sourcePixel.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \param rValues The threshold replacement values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_LTVal_8u_AC4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp8u;
      rValues : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:2421
   pragma Import (C, nppiThreshold_LTVal_8u_AC4R, "nppiThreshold_LTVal_8u_AC4R");

  --* 
  -- * 4 channel 8-bit unsigned char in place image threshold, not affecting Alpha.
  -- * If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set
  -- * value is set to rValue, otherwise it is set to sourcePixel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \param rValues The threshold replacement values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_LTVal_8u_AC4IR
     (pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp8u;
      rValues : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:2437
   pragma Import (C, nppiThreshold_LTVal_8u_AC4IR, "nppiThreshold_LTVal_8u_AC4IR");

  --* 
  -- * 4 channel 16-bit unsigned short image threshold, not affecting Alpha.
  -- * If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set
  -- * value is set to rValue, otherwise it is set to sourcePixel.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \param rValues The threshold replacement values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_LTVal_16u_AC4R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp16u;
      rValues : access nppdefs_h.Npp16u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:2454
   pragma Import (C, nppiThreshold_LTVal_16u_AC4R, "nppiThreshold_LTVal_16u_AC4R");

  --* 
  -- * 4 channel 16-bit unsigned short in place image threshold, not affecting Alpha.
  -- * If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set
  -- * value is set to rValue, otherwise it is set to sourcePixel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \param rValues The threshold replacement values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_LTVal_16u_AC4IR
     (pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp16u;
      rValues : access nppdefs_h.Npp16u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:2470
   pragma Import (C, nppiThreshold_LTVal_16u_AC4IR, "nppiThreshold_LTVal_16u_AC4IR");

  --* 
  -- * 4 channel 16-bit signed short image threshold, not affecting Alpha.
  -- * If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set
  -- * value is set to rValue, otherwise it is set to sourcePixel.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \param rValues The threshold replacement values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_LTVal_16s_AC4R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp16s;
      rValues : access nppdefs_h.Npp16s) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:2487
   pragma Import (C, nppiThreshold_LTVal_16s_AC4R, "nppiThreshold_LTVal_16s_AC4R");

  --* 
  -- * 4 channel 16-bit signed short in place image threshold, not affecting Alpha.
  -- * If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set
  -- * value is set to rValue, otherwise it is set to sourcePixel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \param rValues The threshold replacement values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_LTVal_16s_AC4IR
     (pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp16s;
      rValues : access nppdefs_h.Npp16s) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:2503
   pragma Import (C, nppiThreshold_LTVal_16s_AC4IR, "nppiThreshold_LTVal_16s_AC4IR");

  --* 
  -- * 4 channel 32-bit floating point image threshold, not affecting Alpha.
  -- * If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set
  -- * value is set to rValue, otherwise it is set to sourcePixel.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \param rValues The threshold replacement values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_LTVal_32f_AC4R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp32f;
      rValues : access nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:2520
   pragma Import (C, nppiThreshold_LTVal_32f_AC4R, "nppiThreshold_LTVal_32f_AC4R");

  --* 
  -- * 4 channel 32-bit floating point in place image threshold, not affecting Alpha.
  -- * If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set
  -- * value is set to rValue, otherwise it is set to sourcePixel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholds The threshold values, one per color channel.
  -- * \param rValues The threshold replacement values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_LTVal_32f_AC4IR
     (pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholds : access nppdefs_h.Npp32f;
      rValues : access nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:2536
   pragma Import (C, nppiThreshold_LTVal_32f_AC4IR, "nppiThreshold_LTVal_32f_AC4IR");

  --* 
  -- * 1 channel 8-bit unsigned char threshold.
  -- * If for a comparison operations sourcePixel is less than nThresholdLT is true, the pixel is set
  -- * to nValueLT, else if sourcePixel is greater than nThresholdGT the pixel is set to nValueGT, otherwise it is set to sourcePixel.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nThresholdLT The thresholdLT value.
  -- * \param nValueLT The thresholdLT replacement value.
  -- * \param nThresholdGT The thresholdGT value.
  -- * \param nValueGT The thresholdGT replacement value.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_LTValGTVal_8u_C1R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nThresholdLT : nppdefs_h.Npp8u;
      nValueLT : nppdefs_h.Npp8u;
      nThresholdGT : nppdefs_h.Npp8u;
      nValueGT : nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:2555
   pragma Import (C, nppiThreshold_LTValGTVal_8u_C1R, "nppiThreshold_LTValGTVal_8u_C1R");

  --* 
  -- * 1 channel 8-bit unsigned char in place threshold.
  -- * If for a comparison operations sourcePixel is less than nThresholdLT is true, the pixel is set
  -- * to nValueLT, else if sourcePixel is greater than nThresholdGT the pixel is set to nValueGT, otherwise it is set to sourcePixel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nThresholdLT The thresholdLT value.
  -- * \param nValueLT The thresholdLT replacement value.
  -- * \param nThresholdGT The thresholdGT value.
  -- * \param nValueGT The thresholdGT replacement value.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_LTValGTVal_8u_C1IR
     (pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nThresholdLT : nppdefs_h.Npp8u;
      nValueLT : nppdefs_h.Npp8u;
      nThresholdGT : nppdefs_h.Npp8u;
      nValueGT : nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:2573
   pragma Import (C, nppiThreshold_LTValGTVal_8u_C1IR, "nppiThreshold_LTValGTVal_8u_C1IR");

  --* 
  -- * 1 channel 16-bit unsigned short threshold.
  -- * If for a comparison operations sourcePixel is less than nThresholdLT is true, the pixel is set
  -- * to nValueLT, else if sourcePixel is greater than nThresholdGT the pixel is set to nValueGT, otherwise it is set to sourcePixel.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nThresholdLT The thresholdLT value.
  -- * \param nValueLT The thresholdLT replacement value.
  -- * \param nThresholdGT The thresholdGT value.
  -- * \param nValueGT The thresholdGT replacement value.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_LTValGTVal_16u_C1R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nThresholdLT : nppdefs_h.Npp16u;
      nValueLT : nppdefs_h.Npp16u;
      nThresholdGT : nppdefs_h.Npp16u;
      nValueGT : nppdefs_h.Npp16u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:2592
   pragma Import (C, nppiThreshold_LTValGTVal_16u_C1R, "nppiThreshold_LTValGTVal_16u_C1R");

  --* 
  -- * 1 channel 16-bit unsigned short in place threshold.
  -- * If for a comparison operations sourcePixel is less than nThresholdLT is true, the pixel is set
  -- * to nValueLT, else if sourcePixel is greater than nThresholdGT the pixel is set to nValueGT, otherwise it is set to sourcePixel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nThresholdLT The thresholdLT value.
  -- * \param nValueLT The thresholdLT replacement value.
  -- * \param nThresholdGT The thresholdGT value.
  -- * \param nValueGT The thresholdGT replacement value.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_LTValGTVal_16u_C1IR
     (pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nThresholdLT : nppdefs_h.Npp16u;
      nValueLT : nppdefs_h.Npp16u;
      nThresholdGT : nppdefs_h.Npp16u;
      nValueGT : nppdefs_h.Npp16u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:2610
   pragma Import (C, nppiThreshold_LTValGTVal_16u_C1IR, "nppiThreshold_LTValGTVal_16u_C1IR");

  --* 
  -- * 1 channel 16-bit signed short threshold.
  -- * If for a comparison operations sourcePixel is less than nThresholdLT is true, the pixel is set
  -- * to nValueLT, else if sourcePixel is greater than nThresholdGT the pixel is set to nValueGT, otherwise it is set to sourcePixel.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nThresholdLT The thresholdLT value.
  -- * \param nValueLT The thresholdLT replacement value.
  -- * \param nThresholdGT The thresholdGT value.
  -- * \param nValueGT The thresholdGT replacement value.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_LTValGTVal_16s_C1R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nThresholdLT : nppdefs_h.Npp16s;
      nValueLT : nppdefs_h.Npp16s;
      nThresholdGT : nppdefs_h.Npp16s;
      nValueGT : nppdefs_h.Npp16s) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:2629
   pragma Import (C, nppiThreshold_LTValGTVal_16s_C1R, "nppiThreshold_LTValGTVal_16s_C1R");

  --* 
  -- * 1 channel 16-bit signed short in place threshold.
  -- * If for a comparison operations sourcePixel is less than nThresholdLT is true, the pixel is set
  -- * to nValueLT, else if sourcePixel is greater than nThresholdGT the pixel is set to nValueGT, otherwise it is set to sourcePixel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nThresholdLT The thresholdLT value.
  -- * \param nValueLT The thresholdLT replacement value.
  -- * \param nThresholdGT The thresholdGT value.
  -- * \param nValueGT The thresholdGT replacement value.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_LTValGTVal_16s_C1IR
     (pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nThresholdLT : nppdefs_h.Npp16s;
      nValueLT : nppdefs_h.Npp16s;
      nThresholdGT : nppdefs_h.Npp16s;
      nValueGT : nppdefs_h.Npp16s) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:2647
   pragma Import (C, nppiThreshold_LTValGTVal_16s_C1IR, "nppiThreshold_LTValGTVal_16s_C1IR");

  --* 
  -- * 1 channel 32-bit floating point threshold.
  -- * If for a comparison operations sourcePixel is less than nThresholdLT is true, the pixel is set
  -- * to nValueLT, else if sourcePixel is greater than nThresholdGT the pixel is set to nValueGT, otherwise it is set to sourcePixel.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nThresholdLT The thresholdLT value.
  -- * \param nValueLT The thresholdLT replacement value.
  -- * \param nThresholdGT The thresholdGT value.
  -- * \param nValueGT The thresholdGT replacement value.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_LTValGTVal_32f_C1R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nThresholdLT : nppdefs_h.Npp32f;
      nValueLT : nppdefs_h.Npp32f;
      nThresholdGT : nppdefs_h.Npp32f;
      nValueGT : nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:2666
   pragma Import (C, nppiThreshold_LTValGTVal_32f_C1R, "nppiThreshold_LTValGTVal_32f_C1R");

  --* 
  -- * 1 channel 32-bit floating point in place threshold.
  -- * If for a comparison operations sourcePixel is less than nThresholdLT is true, the pixel is set
  -- * to nValueLT, else if sourcePixel is greater than nThresholdGT the pixel is set to nValueGT, otherwise it is set to sourcePixel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nThresholdLT The thresholdLT value.
  -- * \param nValueLT The thresholdLT replacement value.
  -- * \param nThresholdGT The thresholdGT value.
  -- * \param nValueGT The thresholdGT replacement value.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_LTValGTVal_32f_C1IR
     (pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nThresholdLT : nppdefs_h.Npp32f;
      nValueLT : nppdefs_h.Npp32f;
      nThresholdGT : nppdefs_h.Npp32f;
      nValueGT : nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:2684
   pragma Import (C, nppiThreshold_LTValGTVal_32f_C1IR, "nppiThreshold_LTValGTVal_32f_C1IR");

  --* 
  -- * 3 channel 8-bit unsigned char threshold.
  -- * If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set
  -- * to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholdsLT The thresholdLT values, one per color channel.
  -- * \param rValuesLT The thresholdLT replacement values, one per color channel.
  -- * \param rThresholdsGT The thresholdGT values, one per channel.
  -- * \param rValuesGT The thresholdGT replacement values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_LTValGTVal_8u_C3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholdsLT : access nppdefs_h.Npp8u;
      rValuesLT : access nppdefs_h.Npp8u;
      rThresholdsGT : access nppdefs_h.Npp8u;
      rValuesGT : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:2703
   pragma Import (C, nppiThreshold_LTValGTVal_8u_C3R, "nppiThreshold_LTValGTVal_8u_C3R");

  --* 
  -- * 3 channel 8-bit unsigned char in place threshold.
  -- * If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set
  -- * to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
  -- * \param pSrcDst \ref destination_image_pointer.
  -- * \param nSrcDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholdsLT The thresholdLT values, one per color channel.
  -- * \param rValuesLT The thresholdLT replacement values, one per color channel.
  -- * \param rThresholdsGT The thresholdGT values, one per channel.
  -- * \param rValuesGT The thresholdGT replacement values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_LTValGTVal_8u_C3IR
     (pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholdsLT : access nppdefs_h.Npp8u;
      rValuesLT : access nppdefs_h.Npp8u;
      rThresholdsGT : access nppdefs_h.Npp8u;
      rValuesGT : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:2721
   pragma Import (C, nppiThreshold_LTValGTVal_8u_C3IR, "nppiThreshold_LTValGTVal_8u_C3IR");

  --* 
  -- * 3 channel 16-bit unsigned short threshold.
  -- * If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set
  -- * to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholdsLT The thresholdLT values, one per color channel.
  -- * \param rValuesLT The thresholdLT replacement values, one per color channel.
  -- * \param rThresholdsGT The thresholdGT values, one per channel.
  -- * \param rValuesGT The thresholdGT replacement values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_LTValGTVal_16u_C3R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholdsLT : access nppdefs_h.Npp16u;
      rValuesLT : access nppdefs_h.Npp16u;
      rThresholdsGT : access nppdefs_h.Npp16u;
      rValuesGT : access nppdefs_h.Npp16u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:2740
   pragma Import (C, nppiThreshold_LTValGTVal_16u_C3R, "nppiThreshold_LTValGTVal_16u_C3R");

  --* 
  -- * 3 channel 16-bit unsigned short in place threshold.
  -- * If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set
  -- * to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholdsLT The thresholdLT values, one per color channel.
  -- * \param rValuesLT The thresholdLT replacement values, one per color channel.
  -- * \param rThresholdsGT The thresholdGT values, one per channel.
  -- * \param rValuesGT The thresholdGT replacement values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_LTValGTVal_16u_C3IR
     (pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholdsLT : access nppdefs_h.Npp16u;
      rValuesLT : access nppdefs_h.Npp16u;
      rThresholdsGT : access nppdefs_h.Npp16u;
      rValuesGT : access nppdefs_h.Npp16u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:2758
   pragma Import (C, nppiThreshold_LTValGTVal_16u_C3IR, "nppiThreshold_LTValGTVal_16u_C3IR");

  --* 
  -- * 3 channel 16-bit signed short threshold.
  -- * If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set
  -- * to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholdsLT The thresholdLT values, one per color channel.
  -- * \param rValuesLT The thresholdLT replacement values, one per color channel.
  -- * \param rThresholdsGT The thresholdGT values, one per channel.
  -- * \param rValuesGT The thresholdGT replacement values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_LTValGTVal_16s_C3R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholdsLT : access nppdefs_h.Npp16s;
      rValuesLT : access nppdefs_h.Npp16s;
      rThresholdsGT : access nppdefs_h.Npp16s;
      rValuesGT : access nppdefs_h.Npp16s) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:2777
   pragma Import (C, nppiThreshold_LTValGTVal_16s_C3R, "nppiThreshold_LTValGTVal_16s_C3R");

  --* 
  -- * 3 channel 16-bit signed short in place threshold.
  -- * If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set
  -- * to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholdsLT The thresholdLT values, one per color channel.
  -- * \param rValuesLT The thresholdLT replacement values, one per color channel.
  -- * \param rThresholdsGT The thresholdGT values, one per channel.
  -- * \param rValuesGT The thresholdGT replacement values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_LTValGTVal_16s_C3IR
     (pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholdsLT : access nppdefs_h.Npp16s;
      rValuesLT : access nppdefs_h.Npp16s;
      rThresholdsGT : access nppdefs_h.Npp16s;
      rValuesGT : access nppdefs_h.Npp16s) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:2795
   pragma Import (C, nppiThreshold_LTValGTVal_16s_C3IR, "nppiThreshold_LTValGTVal_16s_C3IR");

  --* 
  -- * 3 channel 32-bit floating point threshold.
  -- * If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set
  -- * to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholdsLT The thresholdLT values, one per color channel.
  -- * \param rValuesLT The thresholdLT replacement values, one per color channel.
  -- * \param rThresholdsGT The thresholdGT values, one per channel.
  -- * \param rValuesGT The thresholdGT replacement values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_LTValGTVal_32f_C3R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholdsLT : access nppdefs_h.Npp32f;
      rValuesLT : access nppdefs_h.Npp32f;
      rThresholdsGT : access nppdefs_h.Npp32f;
      rValuesGT : access nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:2814
   pragma Import (C, nppiThreshold_LTValGTVal_32f_C3R, "nppiThreshold_LTValGTVal_32f_C3R");

  --* 
  -- * 3 channel 32-bit floating point in place threshold.
  -- * If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set
  -- * to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholdsLT The thresholdLT values, one per color channel.
  -- * \param rValuesLT The thresholdLT replacement values, one per color channel.
  -- * \param rThresholdsGT The thresholdGT values, one per channel.
  -- * \param rValuesGT The thresholdGT replacement values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_LTValGTVal_32f_C3IR
     (pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholdsLT : access nppdefs_h.Npp32f;
      rValuesLT : access nppdefs_h.Npp32f;
      rThresholdsGT : access nppdefs_h.Npp32f;
      rValuesGT : access nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:2832
   pragma Import (C, nppiThreshold_LTValGTVal_32f_C3IR, "nppiThreshold_LTValGTVal_32f_C3IR");

  --* 
  -- * 4 channel 8-bit unsigned char image threshold, not affecting Alpha.
  -- * If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set
  -- * value is set to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholdsLT The thresholdLT values, one per color channel.
  -- * \param rValuesLT The thresholdLT replacement values, one per color channel.
  -- * \param rThresholdsGT The thresholdGT values, one per channel.
  -- * \param rValuesGT The thresholdGT replacement values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_LTValGTVal_8u_AC4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholdsLT : access nppdefs_h.Npp8u;
      rValuesLT : access nppdefs_h.Npp8u;
      rThresholdsGT : access nppdefs_h.Npp8u;
      rValuesGT : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:2851
   pragma Import (C, nppiThreshold_LTValGTVal_8u_AC4R, "nppiThreshold_LTValGTVal_8u_AC4R");

  --* 
  -- * 4 channel 8-bit unsigned char in place image threshold, not affecting Alpha.
  -- * If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set
  -- * value is set to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholdsLT The thresholdLT values, one per color channel.
  -- * \param rValuesLT The thresholdLT replacement values, one per color channel.
  -- * \param rThresholdsGT The thresholdGT values, one per channel.
  -- * \param rValuesGT The thresholdGT replacement values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_LTValGTVal_8u_AC4IR
     (pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholdsLT : access nppdefs_h.Npp8u;
      rValuesLT : access nppdefs_h.Npp8u;
      rThresholdsGT : access nppdefs_h.Npp8u;
      rValuesGT : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:2869
   pragma Import (C, nppiThreshold_LTValGTVal_8u_AC4IR, "nppiThreshold_LTValGTVal_8u_AC4IR");

  --* 
  -- * 4 channel 16-bit unsigned short image threshold, not affecting Alpha.
  -- * If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set
  -- * value is set to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholdsLT The thresholdLT values, one per color channel.
  -- * \param rValuesLT The thresholdLT replacement values, one per color channel.
  -- * \param rThresholdsGT The thresholdGT values, one per channel.
  -- * \param rValuesGT The thresholdGT replacement values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_LTValGTVal_16u_AC4R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholdsLT : access nppdefs_h.Npp16u;
      rValuesLT : access nppdefs_h.Npp16u;
      rThresholdsGT : access nppdefs_h.Npp16u;
      rValuesGT : access nppdefs_h.Npp16u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:2888
   pragma Import (C, nppiThreshold_LTValGTVal_16u_AC4R, "nppiThreshold_LTValGTVal_16u_AC4R");

  --* 
  -- * 4 channel 16-bit unsigned short in place image threshold, not affecting Alpha.
  -- * If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set
  -- * value is set to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholdsLT The thresholdLT values, one per color channel.
  -- * \param rValuesLT The thresholdLT replacement values, one per color channel.
  -- * \param rThresholdsGT The thresholdGT values, one per channel.
  -- * \param rValuesGT The thresholdGT replacement values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_LTValGTVal_16u_AC4IR
     (pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholdsLT : access nppdefs_h.Npp16u;
      rValuesLT : access nppdefs_h.Npp16u;
      rThresholdsGT : access nppdefs_h.Npp16u;
      rValuesGT : access nppdefs_h.Npp16u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:2906
   pragma Import (C, nppiThreshold_LTValGTVal_16u_AC4IR, "nppiThreshold_LTValGTVal_16u_AC4IR");

  --* 
  -- * 4 channel 16-bit signed short image threshold, not affecting Alpha.
  -- * If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set
  -- * value is set to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholdsLT The thresholdLT values, one per color channel.
  -- * \param rValuesLT The thresholdLT replacement values, one per color channel.
  -- * \param rThresholdsGT The thresholdGT values, one per channel.
  -- * \param rValuesGT The thresholdGT replacement values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_LTValGTVal_16s_AC4R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholdsLT : access nppdefs_h.Npp16s;
      rValuesLT : access nppdefs_h.Npp16s;
      rThresholdsGT : access nppdefs_h.Npp16s;
      rValuesGT : access nppdefs_h.Npp16s) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:2925
   pragma Import (C, nppiThreshold_LTValGTVal_16s_AC4R, "nppiThreshold_LTValGTVal_16s_AC4R");

  --* 
  -- * 4 channel 16-bit signed short in place image threshold, not affecting Alpha.
  -- * If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set
  -- * value is set to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholdsLT The thresholdLT values, one per color channel.
  -- * \param rValuesLT The thresholdLT replacement values, one per color channel.
  -- * \param rThresholdsGT The thresholdGT values, one per channel.
  -- * \param rValuesGT The thresholdGT replacement values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_LTValGTVal_16s_AC4IR
     (pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholdsLT : access nppdefs_h.Npp16s;
      rValuesLT : access nppdefs_h.Npp16s;
      rThresholdsGT : access nppdefs_h.Npp16s;
      rValuesGT : access nppdefs_h.Npp16s) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:2943
   pragma Import (C, nppiThreshold_LTValGTVal_16s_AC4IR, "nppiThreshold_LTValGTVal_16s_AC4IR");

  --* 
  -- * 4 channel 32-bit floating point image threshold, not affecting Alpha.
  -- * If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set
  -- * value is set to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholdsLT The thresholdLT values, one per color channel.
  -- * \param rValuesLT The thresholdLT replacement values, one per color channel.
  -- * \param rThresholdsGT The thresholdGT values, one per channel.
  -- * \param rValuesGT The thresholdGT replacement values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_LTValGTVal_32f_AC4R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholdsLT : access nppdefs_h.Npp32f;
      rValuesLT : access nppdefs_h.Npp32f;
      rThresholdsGT : access nppdefs_h.Npp32f;
      rValuesGT : access nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:2962
   pragma Import (C, nppiThreshold_LTValGTVal_32f_AC4R, "nppiThreshold_LTValGTVal_32f_AC4R");

  --* 
  -- * 4 channel 32-bit floating point in place image threshold, not affecting Alpha.
  -- * If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set
  -- * value is set to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rThresholdsLT The thresholdLT values, one per color channel.
  -- * \param rValuesLT The thresholdLT replacement values, one per color channel.
  -- * \param rThresholdsGT The thresholdGT values, one per channel.
  -- * \param rValuesGT The thresholdGT replacement values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes.
  --  

   function nppiThreshold_LTValGTVal_32f_AC4IR
     (pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rThresholdsLT : access nppdefs_h.Npp32f;
      rValuesLT : access nppdefs_h.Npp32f;
      rThresholdsGT : access nppdefs_h.Npp32f;
      rValuesGT : access nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:2980
   pragma Import (C, nppiThreshold_LTValGTVal_32f_AC4IR, "nppiThreshold_LTValGTVal_32f_AC4IR");

  --* @} image_threshold_operations  
  --* @defgroup image_compare_operations Compare Operations
  -- * Compare the pixels of two images and create a binary result image. In case of multi-channel
  -- * image types, the condition must be fulfilled for all channels, otherwise the comparison
  -- * is considered false.
  -- * The "binary" result image is of type 8u_C1. False is represented by 0, true by NPP_MAX_8U.
  -- *
  -- * @{
  -- *
  --  

  --* 
  -- * 1 channel 8-bit unsigned char image compare.
  -- * Compare pSrc1's pixels with corresponding pixels in pSrc2. 
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eComparisonOperation Specifies the comparison operation to be used in the pixel comparison.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCompare_8u_C1R
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp8u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eComparisonOperation : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:3009
   pragma Import (C, nppiCompare_8u_C1R, "nppiCompare_8u_C1R");

  --* 
  -- * 3 channel 8-bit unsigned char image compare.
  -- * Compare pSrc1's pixels with corresponding pixels in pSrc2. 
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eComparisonOperation Specifies the comparison operation to be used in the pixel comparison.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCompare_8u_C3R
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp8u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eComparisonOperation : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:3027
   pragma Import (C, nppiCompare_8u_C3R, "nppiCompare_8u_C3R");

  --* 
  -- * 4 channel 8-bit unsigned char image compare.
  -- * Compare pSrc1's pixels with corresponding pixels in pSrc2. 
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eComparisonOperation Specifies the comparison operation to be used in the pixel comparison.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCompare_8u_C4R
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp8u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eComparisonOperation : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:3045
   pragma Import (C, nppiCompare_8u_C4R, "nppiCompare_8u_C4R");

  --* 
  -- * 4 channel 8-bit unsigned char image compare, not affecting Alpha.
  -- * Compare pSrc1's pixels with corresponding pixels in pSrc2. 
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eComparisonOperation Specifies the comparison operation to be used in the pixel comparison.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCompare_8u_AC4R
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp8u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eComparisonOperation : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:3063
   pragma Import (C, nppiCompare_8u_AC4R, "nppiCompare_8u_AC4R");

  --* 
  -- * 1 channel 16-bit unsigned short image compare.
  -- * Compare pSrc1's pixels with corresponding pixels in pSrc2. 
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eComparisonOperation Specifies the comparison operation to be used in the pixel comparison.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCompare_16u_C1R
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp16u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eComparisonOperation : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:3081
   pragma Import (C, nppiCompare_16u_C1R, "nppiCompare_16u_C1R");

  --* 
  -- * 3 channel 16-bit unsigned short image compare.
  -- * Compare pSrc1's pixels with corresponding pixels in pSrc2. 
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eComparisonOperation Specifies the comparison operation to be used in the pixel comparison.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCompare_16u_C3R
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp16u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eComparisonOperation : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:3099
   pragma Import (C, nppiCompare_16u_C3R, "nppiCompare_16u_C3R");

  --* 
  -- * 4 channel 16-bit unsigned short image compare.
  -- * Compare pSrc1's pixels with corresponding pixels in pSrc2. 
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eComparisonOperation Specifies the comparison operation to be used in the pixel comparison.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCompare_16u_C4R
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp16u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eComparisonOperation : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:3117
   pragma Import (C, nppiCompare_16u_C4R, "nppiCompare_16u_C4R");

  --* 
  -- * 4 channel 16-bit unsigned short image compare, not affecting Alpha.
  -- * Compare pSrc1's pixels with corresponding pixels in pSrc2. 
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eComparisonOperation Specifies the comparison operation to be used in the pixel comparison.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCompare_16u_AC4R
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp16u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eComparisonOperation : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:3135
   pragma Import (C, nppiCompare_16u_AC4R, "nppiCompare_16u_AC4R");

  --* 
  -- * 1 channel 16-bit signed short image compare.
  -- * Compare pSrc1's pixels with corresponding pixels in pSrc2. 
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eComparisonOperation Specifies the comparison operation to be used in the pixel comparison.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCompare_16s_C1R
     (pSrc1 : access nppdefs_h.Npp16s;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp16s;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eComparisonOperation : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:3153
   pragma Import (C, nppiCompare_16s_C1R, "nppiCompare_16s_C1R");

  --* 
  -- * 3 channel 16-bit signed short image compare.
  -- * Compare pSrc1's pixels with corresponding pixels in pSrc2. 
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eComparisonOperation Specifies the comparison operation to be used in the pixel comparison.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCompare_16s_C3R
     (pSrc1 : access nppdefs_h.Npp16s;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp16s;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eComparisonOperation : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:3171
   pragma Import (C, nppiCompare_16s_C3R, "nppiCompare_16s_C3R");

  --* 
  -- * 4 channel 16-bit signed short image compare.
  -- * Compare pSrc1's pixels with corresponding pixels in pSrc2. 
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eComparisonOperation Specifies the comparison operation to be used in the pixel comparison.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCompare_16s_C4R
     (pSrc1 : access nppdefs_h.Npp16s;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp16s;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eComparisonOperation : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:3189
   pragma Import (C, nppiCompare_16s_C4R, "nppiCompare_16s_C4R");

  --* 
  -- * 4 channel 16-bit signed short image compare, not affecting Alpha.
  -- * Compare pSrc1's pixels with corresponding pixels in pSrc2. 
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eComparisonOperation Specifies the comparison operation to be used in the pixel comparison.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCompare_16s_AC4R
     (pSrc1 : access nppdefs_h.Npp16s;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp16s;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eComparisonOperation : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:3207
   pragma Import (C, nppiCompare_16s_AC4R, "nppiCompare_16s_AC4R");

  --* 
  -- * 1 channel 32-bit floating point image compare.
  -- * Compare pSrc1's pixels with corresponding pixels in pSrc2. 
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eComparisonOperation Specifies the comparison operation to be used in the pixel comparison.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCompare_32f_C1R
     (pSrc1 : access nppdefs_h.Npp32f;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp32f;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eComparisonOperation : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:3225
   pragma Import (C, nppiCompare_32f_C1R, "nppiCompare_32f_C1R");

  --* 
  -- * 3 channel 32-bit floating point image compare.
  -- * Compare pSrc1's pixels with corresponding pixels in pSrc2. 
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eComparisonOperation Specifies the comparison operation to be used in the pixel comparison.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCompare_32f_C3R
     (pSrc1 : access nppdefs_h.Npp32f;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp32f;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eComparisonOperation : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:3243
   pragma Import (C, nppiCompare_32f_C3R, "nppiCompare_32f_C3R");

  --* 
  -- * 4 channel 32-bit floating point image compare.
  -- * Compare pSrc1's pixels with corresponding pixels in pSrc2. 
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eComparisonOperation Specifies the comparison operation to be used in the pixel comparison.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCompare_32f_C4R
     (pSrc1 : access nppdefs_h.Npp32f;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp32f;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eComparisonOperation : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:3261
   pragma Import (C, nppiCompare_32f_C4R, "nppiCompare_32f_C4R");

  --* 
  -- * 4 channel 32-bit signed floating point compare, not affecting Alpha.
  -- * Compare pSrc1's pixels with corresponding pixels in pSrc2. 
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eComparisonOperation Specifies the comparison operation to be used in the pixel comparison.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCompare_32f_AC4R
     (pSrc1 : access nppdefs_h.Npp32f;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp32f;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eComparisonOperation : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:3279
   pragma Import (C, nppiCompare_32f_AC4R, "nppiCompare_32f_AC4R");

  --* 
  -- * 1 channel 8-bit unsigned char image compare with constant value.
  -- * Compare pSrc's pixels with constant value. 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param nConstant constant value.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eComparisonOperation Specifies the comparison operation to be used in the pixel comparison.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCompareC_8u_C1R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      nConstant : nppdefs_h.Npp8u;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eComparisonOperation : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:3296
   pragma Import (C, nppiCompareC_8u_C1R, "nppiCompareC_8u_C1R");

  --* 
  -- * 3 channel 8-bit unsigned char image compare with constant value.
  -- * Compare pSrc's pixels with constant value. 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pConstants pointer to a list of constant values, one per color channel..
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eComparisonOperation Specifies the comparison operation to be used in the pixel comparison.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCompareC_8u_C3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pConstants : access nppdefs_h.Npp8u;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eComparisonOperation : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:3313
   pragma Import (C, nppiCompareC_8u_C3R, "nppiCompareC_8u_C3R");

  --* 
  -- * 4 channel 8-bit unsigned char image compare with constant value.
  -- * Compare pSrc's pixels with constant value. 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pConstants pointer to a list of constants, one per color channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eComparisonOperation Specifies the comparison operation to be used in the pixel comparison.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCompareC_8u_C4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pConstants : access nppdefs_h.Npp8u;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eComparisonOperation : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:3330
   pragma Import (C, nppiCompareC_8u_C4R, "nppiCompareC_8u_C4R");

  --* 
  -- * 4 channel 8-bit unsigned char image compare, not affecting Alpha.
  -- * Compare pSrc's pixels with constant value. 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pConstants pointer to a list of constants, one per color channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eComparisonOperation Specifies the comparison operation to be used in the pixel comparison.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCompareC_8u_AC4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pConstants : access nppdefs_h.Npp8u;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eComparisonOperation : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:3347
   pragma Import (C, nppiCompareC_8u_AC4R, "nppiCompareC_8u_AC4R");

  --* 
  -- * 1 channel 16-bit unsigned short image compare with constant value.
  -- * Compare pSrc's pixels with constant value. 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param nConstant constant value
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eComparisonOperation Specifies the comparison operation to be used in the pixel comparison.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCompareC_16u_C1R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      nConstant : nppdefs_h.Npp16u;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eComparisonOperation : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:3364
   pragma Import (C, nppiCompareC_16u_C1R, "nppiCompareC_16u_C1R");

  --* 
  -- * 3 channel 16-bit unsigned short image compare with constant value.
  -- * Compare pSrc's pixels with constant value. 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pConstants pointer to a list of constants, one per color channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eComparisonOperation Specifies the comparison operation to be used in the pixel comparison.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCompareC_16u_C3R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pConstants : access nppdefs_h.Npp16u;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eComparisonOperation : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:3381
   pragma Import (C, nppiCompareC_16u_C3R, "nppiCompareC_16u_C3R");

  --* 
  -- * 4 channel 16-bit unsigned short image compare with constant value.
  -- * Compare pSrc's pixels with constant value. 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pConstants pointer to a list of constants, one per color channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eComparisonOperation Specifies the comparison operation to be used in the pixel comparison.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCompareC_16u_C4R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pConstants : access nppdefs_h.Npp16u;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eComparisonOperation : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:3398
   pragma Import (C, nppiCompareC_16u_C4R, "nppiCompareC_16u_C4R");

  --* 
  -- * 4 channel 16-bit unsigned short image compare, not affecting Alpha.
  -- * Compare pSrc's pixels with constant value. 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pConstants pointer to a list of constants, one per color channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eComparisonOperation Specifies the comparison operation to be used in the pixel comparison.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCompareC_16u_AC4R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pConstants : access nppdefs_h.Npp16u;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eComparisonOperation : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:3415
   pragma Import (C, nppiCompareC_16u_AC4R, "nppiCompareC_16u_AC4R");

  --* 
  -- * 1 channel 16-bit signed short image compare with constant value.
  -- * Compare pSrc's pixels with constant value. 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param nConstant constant value.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eComparisonOperation Specifies the comparison operation to be used in the pixel comparison.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCompareC_16s_C1R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      nConstant : nppdefs_h.Npp16s;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eComparisonOperation : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:3432
   pragma Import (C, nppiCompareC_16s_C1R, "nppiCompareC_16s_C1R");

  --* 
  -- * 3 channel 16-bit signed short image compare with constant value.
  -- * Compare pSrc's pixels with constant value. 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pConstants pointer to a list of constants, one per color channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eComparisonOperation Specifies the comparison operation to be used in the pixel comparison.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCompareC_16s_C3R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pConstants : access nppdefs_h.Npp16s;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eComparisonOperation : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:3449
   pragma Import (C, nppiCompareC_16s_C3R, "nppiCompareC_16s_C3R");

  --* 
  -- * 4 channel 16-bit signed short image compare with constant value.
  -- * Compare pSrc's pixels with constant value. 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pConstants pointer to a list of constants, one per color channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eComparisonOperation Specifies the comparison operation to be used in the pixel comparison.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCompareC_16s_C4R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pConstants : access nppdefs_h.Npp16s;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eComparisonOperation : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:3466
   pragma Import (C, nppiCompareC_16s_C4R, "nppiCompareC_16s_C4R");

  --* 
  -- * 4 channel 16-bit signed short image compare, not affecting Alpha.
  -- * Compare pSrc's pixels with constant value. 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pConstants pointer to a list of constants, one per color channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eComparisonOperation Specifies the comparison operation to be used in the pixel comparison.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCompareC_16s_AC4R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pConstants : access nppdefs_h.Npp16s;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eComparisonOperation : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:3483
   pragma Import (C, nppiCompareC_16s_AC4R, "nppiCompareC_16s_AC4R");

  --* 
  -- * 1 channel 32-bit floating point image compare with constant value.
  -- * Compare pSrc's pixels with constant value. 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param nConstant constant value
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eComparisonOperation Specifies the comparison operation to be used in the pixel comparison.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCompareC_32f_C1R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      nConstant : nppdefs_h.Npp32f;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eComparisonOperation : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:3500
   pragma Import (C, nppiCompareC_32f_C1R, "nppiCompareC_32f_C1R");

  --* 
  -- * 3 channel 32-bit floating point image compare with constant value.
  -- * Compare pSrc's pixels with constant value. 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pConstants pointer to a list of constants, one per color channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eComparisonOperation Specifies the comparison operation to be used in the pixel comparison.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCompareC_32f_C3R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pConstants : access nppdefs_h.Npp32f;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eComparisonOperation : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:3517
   pragma Import (C, nppiCompareC_32f_C3R, "nppiCompareC_32f_C3R");

  --* 
  -- * 4 channel 32-bit floating point image compare with constant value.
  -- * Compare pSrc's pixels with constant value. 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pConstants pointer to a list of constants, one per color channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eComparisonOperation Specifies the comparison operation to be used in the pixel comparison.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCompareC_32f_C4R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pConstants : access nppdefs_h.Npp32f;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eComparisonOperation : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:3534
   pragma Import (C, nppiCompareC_32f_C4R, "nppiCompareC_32f_C4R");

  --* 
  -- * 4 channel 32-bit signed floating point compare, not affecting Alpha.
  -- * Compare pSrc's pixels with constant value. 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pConstants pointer to a list of constants, one per color channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eComparisonOperation Specifies the comparison operation to be used in the pixel comparison.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCompareC_32f_AC4R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pConstants : access nppdefs_h.Npp32f;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eComparisonOperation : nppdefs_h.NppCmpOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:3551
   pragma Import (C, nppiCompareC_32f_AC4R, "nppiCompareC_32f_AC4R");

  --* 
  -- * 1 channel 32-bit floating point image compare whether two images are equal within epsilon.
  -- * Compare pSrc1's pixels with corresponding pixels in pSrc2 to determine whether they are equal with a difference of epsilon. 
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nEpsilon epsilon tolerance value to compare to pixel absolute differences
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCompareEqualEps_32f_C1R
     (pSrc1 : access nppdefs_h.Npp32f;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp32f;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nEpsilon : nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:3570
   pragma Import (C, nppiCompareEqualEps_32f_C1R, "nppiCompareEqualEps_32f_C1R");

  --* 
  -- * 3 channel 32-bit floating point image compare whether two images are equal within epsilon.
  -- * Compare pSrc1's pixels with corresponding pixels in pSrc2 to determine whether they are equal with a difference of epsilon. 
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nEpsilon epsilon tolerance value to compare to per color channel pixel absolute differences
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCompareEqualEps_32f_C3R
     (pSrc1 : access nppdefs_h.Npp32f;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp32f;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nEpsilon : nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:3588
   pragma Import (C, nppiCompareEqualEps_32f_C3R, "nppiCompareEqualEps_32f_C3R");

  --* 
  -- * 4 channel 32-bit floating point image compare whether two images are equal within epsilon.
  -- * Compare pSrc1's pixels with corresponding pixels in pSrc2 to determine whether they are equal with a difference of epsilon. 
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nEpsilon epsilon tolerance value to compare to per color channel pixel absolute differences
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCompareEqualEps_32f_C4R
     (pSrc1 : access nppdefs_h.Npp32f;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp32f;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nEpsilon : nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:3606
   pragma Import (C, nppiCompareEqualEps_32f_C4R, "nppiCompareEqualEps_32f_C4R");

  --* 
  -- * 4 channel 32-bit signed floating point compare whether two images are equal within epsilon, not affecting Alpha.
  -- * Compare pSrc1's pixels with corresponding pixels in pSrc2 to determine whether they are equal with a difference of epsilon. 
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nEpsilon epsilon tolerance value to compare to per color channel pixel absolute differences
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCompareEqualEps_32f_AC4R
     (pSrc1 : access nppdefs_h.Npp32f;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp32f;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nEpsilon : nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:3624
   pragma Import (C, nppiCompareEqualEps_32f_AC4R, "nppiCompareEqualEps_32f_AC4R");

  --* 
  -- * 1 channel 32-bit floating point image compare whether image and constant are equal within epsilon.
  -- * Compare pSrc's pixels with constant value to determine whether they are equal within a difference of epsilon. 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param nConstant constant value
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nEpsilon epsilon tolerance value to compare to pixel absolute differences
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCompareEqualEpsC_32f_C1R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      nConstant : nppdefs_h.Npp32f;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nEpsilon : nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:3641
   pragma Import (C, nppiCompareEqualEpsC_32f_C1R, "nppiCompareEqualEpsC_32f_C1R");

  --* 
  -- * 3 channel 32-bit floating point image compare whether image and constant are equal within epsilon.
  -- * Compare pSrc's pixels with constant value to determine whether they are equal within a difference of epsilon. 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pConstants pointer to a list of constants, one per color channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nEpsilon epsilon tolerance value to compare to per color channel pixel absolute differences
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCompareEqualEpsC_32f_C3R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pConstants : access nppdefs_h.Npp32f;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nEpsilon : nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:3658
   pragma Import (C, nppiCompareEqualEpsC_32f_C3R, "nppiCompareEqualEpsC_32f_C3R");

  --* 
  -- * 4 channel 32-bit floating point image compare whether image and constant are equal within epsilon.
  -- * Compare pSrc's pixels with constant value to determine whether they are equal within a difference of epsilon. 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pConstants pointer to a list of constants, one per color channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nEpsilon epsilon tolerance value to compare to per color channel pixel absolute differences
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCompareEqualEpsC_32f_C4R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pConstants : access nppdefs_h.Npp32f;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nEpsilon : nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:3675
   pragma Import (C, nppiCompareEqualEpsC_32f_C4R, "nppiCompareEqualEpsC_32f_C4R");

  --* 
  -- * 4 channel 32-bit signed floating point compare whether image and constant are equal within epsilon, not affecting Alpha.
  -- * Compare pSrc's pixels with constant value to determine whether they are equal within a difference of epsilon. 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pConstants pointer to a list of constants, one per color channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nEpsilon epsilon tolerance value to compare to per color channel pixel absolute differences
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCompareEqualEpsC_32f_AC4R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pConstants : access nppdefs_h.Npp32f;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nEpsilon : nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_threshold_and_compare_operations.h:3692
   pragma Import (C, nppiCompareEqualEpsC_32f_AC4R, "nppiCompareEqualEpsC_32f_AC4R");

  --* @} image_compare_operations  
  --* @} image_threshold_and_compare_operations  
  -- extern "C"  
end nppi_threshold_and_compare_operations_h;

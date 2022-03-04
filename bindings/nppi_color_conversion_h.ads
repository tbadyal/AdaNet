pragma Ada_2005;
pragma Style_Checks (Off);

with Interfaces.C; use Interfaces.C;
with nppdefs_h;
with System;

package nppi_color_conversion_h is

  -- Copyright 2009-2016 NVIDIA Corporation.  All rights reserved. 
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
  -- * \file nppi_color_conversion.h
  -- * NPP Image Processing Functionality.
  --  

  --* @defgroup image_color_conversion Color and Sampling Conversion
  -- *  @ingroup nppi
  -- *
  -- * Routines manipulating an image's color model and sampling format.
  -- *
  -- * @{
  -- *
  -- * These functions can be found in either the nppi or nppicc libraries. Linking to only the sub-libraries that you use can significantly
  -- * save link time, application load time, and CUDA runtime startup time when using dynamic libraries.
  -- *
  --  

  --* @defgroup image_color_model_conversion Color Model Conversion
  -- *
  -- * Routines for converting between various image color models.
  -- *
  -- * @{ 
  -- *
  --  

  --* @name RGBToYUV 
  -- *  RGB to YUV color conversion.
  -- *
  -- *  Here is how NPP converts gamma corrected RGB or BGR to YUV. For digital RGB values in the range [0..255], 
  -- *  Y has the range [0..255], U varies in the range [-112..+112], 
  -- *  and V in the range [-157..+157]. To fit in the range of [0..255], a constant value
  -- *  of 128 is added to computed U and V values, and V is then saturated.
  -- *
  -- *  \code   
  -- *  Npp32f nY =  0.299F * R + 0.587F * G + 0.114F * B; 
  -- *  Npp32f nU = (0.492F * ((Npp32f)nB - nY)) + 128.0F;
  -- *  Npp32f nV = (0.877F * ((Npp32f)nR - nY)) + 128.0F;
  -- *  if (nV > 255.0F) 
  -- *      nV = 255.0F;
  -- *  \endcode
  -- *
  -- * @{
  -- *
  --  

  --*
  -- * 3 channel 8-bit unsigned packed RGB to 3 channel 8-bit unsigned packed YUV color conversion.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiRGBToYUV_8u_C3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:114
   pragma Import (C, nppiRGBToYUV_8u_C3R, "nppiRGBToYUV_8u_C3R");

  --*
  -- * 4 channel 8-bit unsigned packed RGB with alpha to 4 channel 8-bit unsigned packed YUV color conversion with alpha, not affecting alpha.
  -- * images.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiRGBToYUV_8u_AC4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:127
   pragma Import (C, nppiRGBToYUV_8u_AC4R, "nppiRGBToYUV_8u_AC4R");

  --*
  -- * 3 channel 8-bit unsigned planar RGB to 3 channel 8-bit unsigned planar YUV color conversion.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_planar_image_pointer_array.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiRGBToYUV_8u_P3R
     (pSrc : System.Address;
      nSrcStep : int;
      pDst : System.Address;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:139
   pragma Import (C, nppiRGBToYUV_8u_P3R, "nppiRGBToYUV_8u_P3R");

  --*
  -- * 3 channel 8-bit unsigned packed RGB to 3 channel 8-bit unsigned planar YUV color conversion.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_planar_image_pointer_array.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiRGBToYUV_8u_C3P3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : System.Address;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:151
   pragma Import (C, nppiRGBToYUV_8u_C3P3R, "nppiRGBToYUV_8u_C3P3R");

  --*
  -- * 4 channel 8-bit unsigned packed RGB with alpha to 4 channel 8-bit unsigned planar YUV color conversion with alpha.
  -- * images.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_planar_image_pointer_array.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiRGBToYUV_8u_AC4P4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : System.Address;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:164
   pragma Import (C, nppiRGBToYUV_8u_AC4P4R, "nppiRGBToYUV_8u_AC4P4R");

  --* @}  
  --* @name BGRToYUV 
  -- *  BGR to YUV color conversion.
  -- *
  -- *  Here is how NPP converts gamma corrected RGB or BGR to YUV. For digital RGB values in the range [0..255], 
  -- *  Y has the range [0..255], U varies in the range [-112..+112], 
  -- *  and V in the range [-157..+157]. To fit in the range of [0..255], a constant value
  -- *  of 128 is added to computed U and V values, and V is then saturated.
  -- *
  -- *  \code   
  -- *  Npp32f nY =  0.299F * R + 0.587F * G + 0.114F * B; 
  -- *  Npp32f nU = (0.492F * ((Npp32f)nB - nY)) + 128.0F;
  -- *  Npp32f nV = (0.877F * ((Npp32f)nR - nY)) + 128.0F;
  -- *  if (nV > 255.0F) 
  -- *      nV = 255.0F;
  -- *  \endcode
  -- *
  -- * @{
  -- *
  --  

  --*
  -- * 3 channel 8-bit unsigned packed BGR to 3 channel 8-bit unsigned packed YUV color conversion.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiBGRToYUV_8u_C3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:198
   pragma Import (C, nppiBGRToYUV_8u_C3R, "nppiBGRToYUV_8u_C3R");

  --*
  -- * 4 channel 8-bit unsigned packed BGR with alpha to 4 channel 8-bit unsigned packed YUV color conversion with alpha, not affecting alpha.
  -- * images.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiBGRToYUV_8u_AC4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:211
   pragma Import (C, nppiBGRToYUV_8u_AC4R, "nppiBGRToYUV_8u_AC4R");

  --*
  -- * 3 channel 8-bit unsigned planar BGR to 3 channel 8-bit unsigned planar YUV color conversion.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_planar_image_pointer_array.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiBGRToYUV_8u_P3R
     (pSrc : System.Address;
      nSrcStep : int;
      pDst : System.Address;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:223
   pragma Import (C, nppiBGRToYUV_8u_P3R, "nppiBGRToYUV_8u_P3R");

  --*
  -- * 3 channel 8-bit unsigned packed BGR to 3 channel 8-bit unsigned planar YUV color conversion.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_planar_image_pointer_array.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiBGRToYUV_8u_C3P3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : System.Address;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:235
   pragma Import (C, nppiBGRToYUV_8u_C3P3R, "nppiBGRToYUV_8u_C3P3R");

  --*
  -- * 4 channel 8-bit unsigned packed BGR with alpha to 4 channel 8-bit unsigned planar YUV color conversion with alpha.
  -- * images.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_planar_image_pointer_array.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiBGRToYUV_8u_AC4P4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : System.Address;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:248
   pragma Import (C, nppiBGRToYUV_8u_AC4P4R, "nppiBGRToYUV_8u_AC4P4R");

  --* @}  
  --* @name YUVToRGB 
  -- *  YUV to RGB color conversion.
  -- *
  -- *  Here is how NPP converts YUV to gamma corrected RGB or BGR.
  -- *
  -- *  \code
  -- *  Npp32f nY = (Npp32f)Y;
  -- *  Npp32f nU = (Npp32f)U - 128.0F;
  -- *  Npp32f nV = (Npp32f)V - 128.0F;
  -- *  Npp32f nR = nY + 1.140F * nV; 
  -- *  if (nR < 0.0F)
  -- *      nR = 0.0F;
  -- *  if (nR > 255.0F)
  -- *      nR = 255.0F;    
  -- *  Npp32f nG = nY - 0.394F * nU - 0.581F * nV;
  -- *  if (nG < 0.0F)
  -- *      nG = 0.0F;
  -- *  if (nG > 255.0F)
  -- *      nG = 255.0F;    
  -- *  Npp32f nB = nY + 2.032F * nU;
  -- *  if (nB < 0.0F)
  -- *      nB = 0.0F;
  -- *  if (nB > 255.0F)
  -- *      nB = 255.0F;    
  -- *  \endcode
  -- *
  -- * @{
  -- *
  --  

  --*
  -- * 3 channel 8-bit unsigned packed YUV to 3 channel 8-bit unsigned packed RGB color conversion.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYUVToRGB_8u_C3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:292
   pragma Import (C, nppiYUVToRGB_8u_C3R, "nppiYUVToRGB_8u_C3R");

  --*
  -- * 4 channel 8-bit packed YUV with alpha to 4 channel 8-bit unsigned packed RGB color conversion with alpha, not affecting alpha.
  -- * images.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYUVToRGB_8u_AC4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:305
   pragma Import (C, nppiYUVToRGB_8u_AC4R, "nppiYUVToRGB_8u_AC4R");

  --*
  -- * 3 channel 8-bit unsigned planar YUV to 3 channel 8-bit unsigned planar RGB color conversion.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_planar_image_pointer_array.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYUVToRGB_8u_P3R
     (pSrc : System.Address;
      nSrcStep : int;
      pDst : System.Address;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:317
   pragma Import (C, nppiYUVToRGB_8u_P3R, "nppiYUVToRGB_8u_P3R");

  --*
  -- * 3 channel 8-bit unsigned planar YUV to 3 channel 8-bit unsigned packed RGB color conversion.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYUVToRGB_8u_P3C3R
     (pSrc : System.Address;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:329
   pragma Import (C, nppiYUVToRGB_8u_P3C3R, "nppiYUVToRGB_8u_P3C3R");

  --* @}  
  --* @name YUVToBGR 
  -- *  YUV to BGR color conversion.
  -- *
  -- *  Here is how NPP converts YUV to gamma corrected RGB or BGR.
  -- *
  -- *  \code
  -- *  Npp32f nY = (Npp32f)Y;
  -- *  Npp32f nU = (Npp32f)U - 128.0F;
  -- *  Npp32f nV = (Npp32f)V - 128.0F;
  -- *  Npp32f nR = nY + 1.140F * nV; 
  -- *  if (nR < 0.0F)
  -- *      nR = 0.0F;
  -- *  if (nR > 255.0F)
  -- *      nR = 255.0F;    
  -- *  Npp32f nG = nY - 0.394F * nU - 0.581F * nV;
  -- *  if (nG < 0.0F)
  -- *      nG = 0.0F;
  -- *  if (nG > 255.0F)
  -- *      nG = 255.0F;    
  -- *  Npp32f nB = nY + 2.032F * nU;
  -- *  if (nB < 0.0F)
  -- *      nB = 0.0F;
  -- *  if (nB > 255.0F)
  -- *      nB = 255.0F;    
  -- *  \endcode
  -- *
  -- * @{
  -- *
  --  

  --*
  -- * 3 channel 8-bit unsigned packed YUV to 3 channel 8-bit unsigned packed BGR color conversion.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYUVToBGR_8u_C3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:373
   pragma Import (C, nppiYUVToBGR_8u_C3R, "nppiYUVToBGR_8u_C3R");

  --*
  -- * 4 channel 8-bit packed YUV with alpha to 4 channel 8-bit unsigned packed BGR color conversion with alpha, not affecting alpha.
  -- * images.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYUVToBGR_8u_AC4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:386
   pragma Import (C, nppiYUVToBGR_8u_AC4R, "nppiYUVToBGR_8u_AC4R");

  --*
  -- * 3 channel 8-bit unsigned planar YUV to 3 channel 8-bit unsigned planar BGR color conversion.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_planar_image_pointer_array.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYUVToBGR_8u_P3R
     (pSrc : System.Address;
      nSrcStep : int;
      pDst : System.Address;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:398
   pragma Import (C, nppiYUVToBGR_8u_P3R, "nppiYUVToBGR_8u_P3R");

  --*
  -- * 3 channel 8-bit unsigned planar YUV to 3 channel 8-bit unsigned packed BGR color conversion.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYUVToBGR_8u_P3C3R
     (pSrc : System.Address;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:410
   pragma Import (C, nppiYUVToBGR_8u_P3C3R, "nppiYUVToBGR_8u_P3C3R");

  --* @}  
  --* @name RGBToYUV422 
  -- *  RGB to YUV422 color conversion.
  -- *
  -- * @{
  -- *
  --  

  --*
  -- * 3 channel 8-bit unsigned packed RGB to 2 channel 8-bit unsigned packed YUV422 color conversion.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiRGBToYUV422_8u_C3C2R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:431
   pragma Import (C, nppiRGBToYUV422_8u_C3C2R, "nppiRGBToYUV422_8u_C3C2R");

  --*
  -- * 3 channel 8-bit unsigned planar RGB to 3 channel 8-bit unsigned planar YUV422 color conversion.
  -- * images.
  -- *                         
  -- * \param pSrc \ref source_planar_image_pointer_array.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_planar_image_pointer_array.
  -- * \param rDstStep \ref destination_planar_image_line_step_array.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiRGBToYUV422_8u_P3R
     (pSrc : System.Address;
      nSrcStep : int;
      pDst : System.Address;
      rDstStep : access int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:444
   pragma Import (C, nppiRGBToYUV422_8u_P3R, "nppiRGBToYUV422_8u_P3R");

  --*
  -- * 3 channel 8-bit unsigned packed RGB to 3 channel 8-bit unsigned planar YUV422 color conversion.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_planar_image_pointer_array.
  -- * \param rDstStep \ref destination_planar_image_line_step_array.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiRGBToYUV422_8u_C3P3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : System.Address;
      rDstStep : access int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:456
   pragma Import (C, nppiRGBToYUV422_8u_C3P3R, "nppiRGBToYUV422_8u_C3P3R");

  --* @}  
  --* @name YUV422ToRGB 
  -- *  YUV422 to RGB color conversion.
  -- *
  -- * @{
  -- *
  --  

  --*
  -- * 2 channel 8-bit unsigned packed YUV422 to 3 channel 8-bit unsigned packed RGB color conversion.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYUV422ToRGB_8u_C2C3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:477
   pragma Import (C, nppiYUV422ToRGB_8u_C2C3R, "nppiYUV422ToRGB_8u_C2C3R");

  --*
  -- * 3 channel 8-bit unsigned planar YUV422 to 3 channel 8-bit unsigned planar RGB color conversion.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array.
  -- * \param rSrcStep \ref source_planar_image_line_step_array.
  -- * \param pDst \ref destination_planar_image_pointer_array.
  -- * \param nDstStep \ref destination_planar_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYUV422ToRGB_8u_P3R
     (pSrc : System.Address;
      rSrcStep : access int;
      pDst : System.Address;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:489
   pragma Import (C, nppiYUV422ToRGB_8u_P3R, "nppiYUV422ToRGB_8u_P3R");

  --*
  -- * 3 channel 8-bit unsigned planar YUV422 to 3 channel 8-bit unsigned packed RGB color conversion.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array.
  -- * \param rSrcStep \ref source_planar_image_line_step_array.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYUV422ToRGB_8u_P3C3R
     (pSrc : System.Address;
      rSrcStep : access int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:501
   pragma Import (C, nppiYUV422ToRGB_8u_P3C3R, "nppiYUV422ToRGB_8u_P3C3R");

  --*
  -- * 3 channel 8-bit unsigned planar YUV422 to 4 channel 8-bit unsigned packed RGB color conversion with alpha.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array.
  -- * \param rSrcStep \ref source_planar_image_line_step_array.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYUV422ToRGB_8u_P3AC4R
     (pSrc : System.Address;
      rSrcStep : access int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:513
   pragma Import (C, nppiYUV422ToRGB_8u_P3AC4R, "nppiYUV422ToRGB_8u_P3AC4R");

  --* @}  
  --* @name RGBToYUV420 
  -- *  RGB to YUV420 color conversion.
  -- *
  -- * @{
  -- *
  --  

  --*
  -- * 3 channel 8-bit unsigned planar RGB to 3 channel 8-bit unsigned planar YUV420 color conversion.
  -- * images.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_planar_image_pointer_array.
  -- * \param rDstStep \ref destination_planar_image_line_step_array.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiRGBToYUV420_8u_P3R
     (pSrc : System.Address;
      nSrcStep : int;
      pDst : System.Address;
      rDstStep : access int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:535
   pragma Import (C, nppiRGBToYUV420_8u_P3R, "nppiRGBToYUV420_8u_P3R");

  --*
  -- * 3 channel 8-bit unsigned packed RGB to 3 channel 8-bit unsigned planar YUV420 color conversion.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_planar_image_pointer_array.
  -- * \param rDstStep \ref destination_planar_image_line_step_array.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiRGBToYUV420_8u_C3P3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : System.Address;
      rDstStep : access int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:547
   pragma Import (C, nppiRGBToYUV420_8u_C3P3R, "nppiRGBToYUV420_8u_C3P3R");

  --* @}  
  --* @name YUV420ToRGB 
  -- *  YUV420 to RGB color conversion.
  -- *
  -- * @{
  -- *
  --  

  --*
  -- * 3 channel 8-bit unsigned planar YUV420 to 3 channel 8-bit unsigned planar RGB color conversion.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array.
  -- * \param rSrcStep \ref source_planar_image_line_step_array.
  -- * \param pDst \ref destination_planar_image_pointer_array.
  -- * \param nDstStep \ref destination_planar_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYUV420ToRGB_8u_P3R
     (pSrc : System.Address;
      rSrcStep : access int;
      pDst : System.Address;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:568
   pragma Import (C, nppiYUV420ToRGB_8u_P3R, "nppiYUV420ToRGB_8u_P3R");

  --*
  -- * 3 channel 8-bit unsigned planar YUV420 to 3 channel 8-bit unsigned packed RGB color conversion.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array.
  -- * \param rSrcStep \ref source_planar_image_line_step_array.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYUV420ToRGB_8u_P3C3R
     (pSrc : System.Address;
      rSrcStep : access int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:580
   pragma Import (C, nppiYUV420ToRGB_8u_P3C3R, "nppiYUV420ToRGB_8u_P3C3R");

  --*
  -- * 3 channel 8-bit unsigned planar YUV420 to 4 channel 8-bit unsigned packed RGB color conversion with constant alpha (0xFF).
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array.
  -- * \param rSrcStep \ref source_planar_image_line_step_array.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYUV420ToRGB_8u_P3C4R
     (pSrc : System.Address;
      rSrcStep : access int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:592
   pragma Import (C, nppiYUV420ToRGB_8u_P3C4R, "nppiYUV420ToRGB_8u_P3C4R");

  --*
  -- * 3 channel 8-bit unsigned planar YUV420 to 4 channel 8-bit unsigned packed RGB color conversion with alpha.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array.
  -- * \param rSrcStep \ref source_planar_image_line_step_array.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYUV420ToRGB_8u_P3AC4R
     (pSrc : System.Address;
      rSrcStep : access int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:604
   pragma Import (C, nppiYUV420ToRGB_8u_P3AC4R, "nppiYUV420ToRGB_8u_P3AC4R");

  --* @}  
  --* @name NV21ToRGB 
  -- *  NV21 to RGB color conversion.
  -- *
  -- * @{
  -- *
  --  

  --*
  -- *  2 channel 8-bit unsigned planar NV21 to 4 channel 8-bit unsigned packed RGBA color conversion with constant alpha (0xFF).
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array (one for Y plane, one for VU plane).
  -- * \param rSrcStep \ref source_planar_image_line_step_array.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiNV21ToRGB_8u_P2C4R
     (pSrc : System.Address;
      rSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:625
   pragma Import (C, nppiNV21ToRGB_8u_P2C4R, "nppiNV21ToRGB_8u_P2C4R");

  --* @}  
  --* @name BGRToYUV420 
  -- *  BGR to YUV420 color conversion.
  -- *
  -- * @{
  -- *
  --  

  --*
  -- * 4 channel 8-bit unsigned pacmed BGR with alpha to 3 channel 8-bit unsigned planar YUV420 color conversion.
  -- * images.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_planar_image_pointer_array.
  -- * \param rDstStep \ref destination_planar_image_line_step_array.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiBGRToYUV420_8u_AC4P3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : System.Address;
      rDstStep : access int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:647
   pragma Import (C, nppiBGRToYUV420_8u_AC4P3R, "nppiBGRToYUV420_8u_AC4P3R");

  --* @}  
  --* @name YUV420ToBGR 
  -- *  YUV420 to BGR color conversion.
  -- *
  -- * @{
  -- *
  --  

  --*
  -- * 3 channel 8-bit unsigned planar YUV420 to 3 channel 8-bit unsigned packed BGR color conversion.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array.
  -- * \param rSrcStep \ref source_planar_image_line_step_array.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYUV420ToBGR_8u_P3C3R
     (pSrc : System.Address;
      rSrcStep : access int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:668
   pragma Import (C, nppiYUV420ToBGR_8u_P3C3R, "nppiYUV420ToBGR_8u_P3C3R");

  --*
  -- * 3 channel 8-bit unsigned planar YUV420 to 4 channel 8-bit unsigned packed BGR color conversion with constant alpha (0xFF).
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array.
  -- * \param rSrcStep \ref source_planar_image_line_step_array.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYUV420ToBGR_8u_P3C4R
     (pSrc : System.Address;
      rSrcStep : access int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:680
   pragma Import (C, nppiYUV420ToBGR_8u_P3C4R, "nppiYUV420ToBGR_8u_P3C4R");

  --* @}  
  --* @name NV21ToBGR 
  -- *  NV21 to BGR color conversion.
  -- *
  -- * @{
  -- *
  --  

  --*
  -- *  2 channel 8-bit unsigned planar NV21 to 4 channel 8-bit unsigned packed BGRA color conversion with constant alpha (0xFF).
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array (one for Y plane, one for VU plane).
  -- * \param rSrcStep \ref source_planar_image_line_step_array.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiNV21ToBGR_8u_P2C4R
     (pSrc : System.Address;
      rSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:701
   pragma Import (C, nppiNV21ToBGR_8u_P2C4R, "nppiNV21ToBGR_8u_P2C4R");

  --* @}  
  --* @name RGBToYCbCr 
  -- *  RGB to YCbCr color conversion.
  -- *
  -- *  Here is how NPP converts gamma corrected RGB or BGR to YCbCr.  In the YCbCr model, 
  -- *  Y is defined to have a nominal range [16..235], while Cb and Cr are defined
  -- *  to have a range [16..240], with the value of 128 as corresponding to zero.
  -- *
  -- *  \code
  -- *  Npp32f nY  =  0.257F * R + 0.504F * G + 0.098F * B + 16.0F; 
  -- *  Npp32f nCb = -0.148F * R - 0.291F * G + 0.439F * B + 128.0F;
  -- *  Npp32f nCr =  0.439F * R - 0.368F * G - 0.071F * B + 128.0F;
  -- *  \endcode
  -- *
  -- * @{
  -- *
  --  

  --*
  -- * 3 channel 8-bit unsigned packed RGB to 3 channel unsigned 8-bit packed YCbCr color conversion.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiRGBToYCbCr_8u_C3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:733
   pragma Import (C, nppiRGBToYCbCr_8u_C3R, "nppiRGBToYCbCr_8u_C3R");

  --*
  -- * 4 channel 8-bit unsigned packed RGB with alpha to 4 channel unsigned 8-bit packed YCbCr with alpha color conversion, not affecting alpha.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiRGBToYCbCr_8u_AC4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:745
   pragma Import (C, nppiRGBToYCbCr_8u_AC4R, "nppiRGBToYCbCr_8u_AC4R");

  --*
  -- * 3 channel planar 8-bit unsigned RGB to 3 channel planar 8-bit YCbCr color conversion.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_planar_image_pointer_array.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiRGBToYCbCr_8u_P3R
     (pSrc : System.Address;
      nSrcStep : int;
      pDst : System.Address;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:757
   pragma Import (C, nppiRGBToYCbCr_8u_P3R, "nppiRGBToYCbCr_8u_P3R");

  --*
  -- * 3 channel 8-bit unsigned packed RGB to 3 channel unsigned 8-bit planar YCbCr color conversion.
  -- * images.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiRGBToYCbCr_8u_C3P3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : System.Address;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:770
   pragma Import (C, nppiRGBToYCbCr_8u_C3P3R, "nppiRGBToYCbCr_8u_C3P3R");

  --*
  -- * 4 channel 8-bit unsigned packed RGB with alpha to 3 channel 8-bit unsigned planar YCbCr color conversion.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_planar_image_pointer_array.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiRGBToYCbCr_8u_AC4P3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : System.Address;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:782
   pragma Import (C, nppiRGBToYCbCr_8u_AC4P3R, "nppiRGBToYCbCr_8u_AC4P3R");

  --* @}  
  --* @name YCbCrToRGB 
  -- *  YCbCr to RGB color conversion.
  -- *
  -- *  Here is how NPP converts YCbCr to gamma corrected RGB or BGR.  The output RGB values are saturated to the range [0..255].
  -- *
  -- *  \code
  -- *  Npp32f nY = 1.164F * ((Npp32f)Y - 16.0F);
  -- *  Npp32f nR = ((Npp32f)Cr - 128.0F);
  -- *  Npp32f nB = ((Npp32f)Cb - 128.0F);
  -- *  Npp32f nG = nY - 0.813F * nR - 0.392F * nB;
  -- *  if (nG > 255.0F)
  -- *      nG = 255.0F;
  -- *  nR = nY + 1.596F * nR; 
  -- *  if (nR > 255.0F)
  -- *      nR = 255.0F;
  -- *  nB = nY + 2.017F * nB;
  -- *  if (nB > 255.0F)
  -- *      nB = 255.0F;
  -- *  \endcode
  -- *
  -- * @{
  -- *
  --  

  --*
  -- * 3 channel 8-bit unsigned packed YCbCr to 3 channel 8-bit unsigned packed RGB color conversion.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYCbCrToRGB_8u_C3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:820
   pragma Import (C, nppiYCbCrToRGB_8u_C3R, "nppiYCbCrToRGB_8u_C3R");

  --*
  -- * 4 channel 8-bit unsigned packed YCbCr with alpha to 4 channel 8-bit unsigned packed RGB with alpha color conversion, not affecting alpha.
  -- * Alpha channel is the last channel and is not processed.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYCbCrToRGB_8u_AC4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:833
   pragma Import (C, nppiYCbCrToRGB_8u_AC4R, "nppiYCbCrToRGB_8u_AC4R");

  --*
  -- * 3 channel 8-bit unsigned planar YCbCr to 3 channel 8-bit unsigned planar RGB color conversion.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_planar_image_pointer_array.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYCbCrToRGB_8u_P3R
     (pSrc : System.Address;
      nSrcStep : int;
      pDst : System.Address;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:845
   pragma Import (C, nppiYCbCrToRGB_8u_P3R, "nppiYCbCrToRGB_8u_P3R");

  --*
  -- * 3 channel 8-bit unsigned planar YCbCr to 3 channel 8-bit unsigned packed RGB color conversion.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYCbCrToRGB_8u_P3C3R
     (pSrc : System.Address;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:857
   pragma Import (C, nppiYCbCrToRGB_8u_P3C3R, "nppiYCbCrToRGB_8u_P3C3R");

  --*
  -- * 3 channel 8-bit unsigned planar YCbCr to 4 channel 8-bit unsigned packed RGB color conversion with constant alpha.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nAval 8-bit unsigned alpha constant.                                         
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYCbCrToRGB_8u_P3C4R
     (pSrc : System.Address;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nAval : nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:870
   pragma Import (C, nppiYCbCrToRGB_8u_P3C4R, "nppiYCbCrToRGB_8u_P3C4R");

  --* @}  
  --* @name YCbCrToBGR 
  -- *  YCbCr to BGR color conversion.
  -- *
  -- * @{
  -- *
  --  

  --*
  -- * 3 channel 8-bit unsigned planar YCbCr to 3 channel 8-bit unsigned packed BGR color conversion.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYCbCrToBGR_8u_P3C3R
     (pSrc : System.Address;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:891
   pragma Import (C, nppiYCbCrToBGR_8u_P3C3R, "nppiYCbCrToBGR_8u_P3C3R");

  --*
  -- * 3 channel 8-bit unsigned planar YCbCr to 4 channel 8-bit unsigned packed BGR color conversion with constant alpha.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nAval 8-bit unsigned alpha constant.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYCbCrToBGR_8u_P3C4R
     (pSrc : System.Address;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nAval : nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:904
   pragma Import (C, nppiYCbCrToBGR_8u_P3C4R, "nppiYCbCrToBGR_8u_P3C4R");

  --* @}  
  --* @name YCbCrToBGR_709CSC 
  -- *  YCbCr to BGR_709CSC color conversion.
  -- *
  -- * @{
  -- *
  --  

  --*
  -- * 3 channel 8-bit unsigned planar YCbCr to 3 channel 8-bit unsigned packed BGR_709CSC color conversion.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYCbCrToBGR_709CSC_8u_P3C3R
     (pSrc : System.Address;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:925
   pragma Import (C, nppiYCbCrToBGR_709CSC_8u_P3C3R, "nppiYCbCrToBGR_709CSC_8u_P3C3R");

  --*
  -- * 3 channel 8-bit unsigned planar YCbCr to 4 channel 8-bit unsigned packed BGR_709CSC color conversion with constant alpha.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nAval 8-bit unsigned alpha constant.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYCbCrToBGR_709CSC_8u_P3C4R
     (pSrc : System.Address;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nAval : nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:938
   pragma Import (C, nppiYCbCrToBGR_709CSC_8u_P3C4R, "nppiYCbCrToBGR_709CSC_8u_P3C4R");

  --* @}  
  --* @name RGBToYCbCr422 
  -- *  RGB to YCbCr422 color conversion.
  -- *
  -- * @{
  -- *
  --  

  --*
  -- * 3 channel 8-bit unsigned packed RGB to 2 channel 8-bit unsigned packed YCbCr422 color conversion.
  -- * images.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiRGBToYCbCr422_8u_C3C2R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:961
   pragma Import (C, nppiRGBToYCbCr422_8u_C3C2R, "nppiRGBToYCbCr422_8u_C3C2R");

  --*
  -- * 3 channel 8-bit unsigned packed RGB to 3 channel 8-bit unsigned planar YCbCr422 color conversion.
  -- * images.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_planar_image_pointer_array.
  -- * \param rDstStep \ref destination_planar_image_line_step_array.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiRGBToYCbCr422_8u_C3P3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : System.Address;
      rDstStep : access int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:974
   pragma Import (C, nppiRGBToYCbCr422_8u_C3P3R, "nppiRGBToYCbCr422_8u_C3P3R");

  --*
  -- * 3 channel 8-bit unsigned planar RGB to 2 channel 8-bit unsigned packed YCbCr422 color conversion.
  -- * images.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiRGBToYCbCr422_8u_P3C2R
     (pSrc : System.Address;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:987
   pragma Import (C, nppiRGBToYCbCr422_8u_P3C2R, "nppiRGBToYCbCr422_8u_P3C2R");

  --* @}  
  --* @name YCbCr422ToRGB 
  -- *  YCbCr422 to RGB color conversion.
  -- *
  -- * @{
  -- *
  --  

  --*
  -- * 2 channel 8-bit unsigned packed YCbCr422 to 3 channel 8-bit unsigned packed RGB color conversion.
  -- * images.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYCbCr422ToRGB_8u_C2C3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:1009
   pragma Import (C, nppiYCbCr422ToRGB_8u_C2C3R, "nppiYCbCr422ToRGB_8u_C2C3R");

  --*
  -- * 2 channel 8-bit unsigned packed YCbCr422 to 3 channel 8-bit unsigned planar RGB color conversion.
  -- * images.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_planar_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYCbCr422ToRGB_8u_C2P3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : System.Address;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:1022
   pragma Import (C, nppiYCbCr422ToRGB_8u_C2P3R, "nppiYCbCr422ToRGB_8u_C2P3R");

  --*
  -- * 3 channel 8-bit unsigned planar YCbCr422 to 3 channel 8-bit unsigned packed RGB color conversion.
  -- * images.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array.
  -- * \param rSrcStep \ref source_planar_image_line_step_array.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYCbCr422ToRGB_8u_P3C3R
     (pSrc : System.Address;
      rSrcStep : access int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:1035
   pragma Import (C, nppiYCbCr422ToRGB_8u_P3C3R, "nppiYCbCr422ToRGB_8u_P3C3R");

  --* @}  
  --* @name RGBToYCrCb422 
  -- *  RGB to YCrCb422 color conversion.
  -- *
  -- * @{
  -- *
  --  

  --*
  -- * 3 channel 8-bit unsigned packed RGB to 2 channel 8-bit unsigned packed YCrCb422 color conversion.
  -- * images.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiRGBToYCrCb422_8u_C3C2R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:1057
   pragma Import (C, nppiRGBToYCrCb422_8u_C3C2R, "nppiRGBToYCrCb422_8u_C3C2R");

  --*
  -- * 3 channel 8-bit unsigned planar RGB to 2 channel 8-bit unsigned packed YCrCb422 color conversion.
  -- * images.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiRGBToYCrCb422_8u_P3C2R
     (pSrc : System.Address;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:1070
   pragma Import (C, nppiRGBToYCrCb422_8u_P3C2R, "nppiRGBToYCrCb422_8u_P3C2R");

  --* @}  
  --* @name YCrCb422ToRGB 
  -- *  YCrCb422 to RGB color conversion.
  -- *
  -- * @{
  -- *
  --  

  --*
  -- * 2 channel 8-bit unsigned packed YCrCb422 to 3 channel 8-bit unsigned packed RGB color conversion.
  -- * images.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYCrCb422ToRGB_8u_C2C3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:1092
   pragma Import (C, nppiYCrCb422ToRGB_8u_C2C3R, "nppiYCrCb422ToRGB_8u_C2C3R");

  --*
  -- * 2 channel 8-bit unsigned packed YCrCb422 to 3 channel 8-bit unsigned planar RGB color conversion.
  -- * images.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_planar_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYCrCb422ToRGB_8u_C2P3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : System.Address;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:1105
   pragma Import (C, nppiYCrCb422ToRGB_8u_C2P3R, "nppiYCrCb422ToRGB_8u_C2P3R");

  --* @}  
  --* @name BGRToYCbCr422 
  -- *  BGR to YCbCr422 color conversion.
  -- *
  -- * @{
  -- *
  --  

  --*
  -- * 3 channel 8-bit unsigned packed BGR to 2 channel 8-bit unsigned packed YCrCb422 color conversion.
  -- * images.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiBGRToYCbCr422_8u_C3C2R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:1127
   pragma Import (C, nppiBGRToYCbCr422_8u_C3C2R, "nppiBGRToYCbCr422_8u_C3C2R");

  --*
  -- * 4 channel 8-bit unsigned packed BGR with alpha to 2 channel 8-bit unsigned packed YCrCb422 color conversion.
  -- * images.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiBGRToYCbCr422_8u_AC4C2R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:1140
   pragma Import (C, nppiBGRToYCbCr422_8u_AC4C2R, "nppiBGRToYCbCr422_8u_AC4C2R");

  --*
  -- * 3 channel 8-bit unsigned packed BGR to 3 channel 8-bit unsigned planar YCbCr422 color conversion.
  -- * images.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_planar_image_pointer_array.
  -- * \param rDstStep \ref destination_planar_image_line_step_array.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiBGRToYCbCr422_8u_C3P3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : System.Address;
      rDstStep : access int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:1153
   pragma Import (C, nppiBGRToYCbCr422_8u_C3P3R, "nppiBGRToYCbCr422_8u_C3P3R");

  --*
  -- * 4 channel 8-bit unsigned packed BGR with alpha to 3 channel 8-bit unsigned planar YCbCr422 color conversion.
  -- * images.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_planar_image_pointer_array.
  -- * \param rDstStep \ref destination_planar_image_line_step_array.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiBGRToYCbCr422_8u_AC4P3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : System.Address;
      rDstStep : access int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:1166
   pragma Import (C, nppiBGRToYCbCr422_8u_AC4P3R, "nppiBGRToYCbCr422_8u_AC4P3R");

  --* @}  
  --* @name YCbCr422ToBGR 
  -- *  YCbCr422 to BGR color conversion.
  -- *
  -- * @{
  -- *
  --  

  --*
  -- * 2 channel 8-bit unsigned packed YCrCb422 to 3 channel 8-bit unsigned packed BGR color conversion.
  -- * images.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYCbCr422ToBGR_8u_C2C3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:1188
   pragma Import (C, nppiYCbCr422ToBGR_8u_C2C3R, "nppiYCbCr422ToBGR_8u_C2C3R");

  --*
  -- * 2 channel 8-bit unsigned packed YCrCb422 to 4 channel 8-bit unsigned packed BGR color conversion with constant alpha.
  -- * images.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nAval 8-bit unsigned alpha constant.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYCbCr422ToBGR_8u_C2C4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nAval : nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:1202
   pragma Import (C, nppiYCbCr422ToBGR_8u_C2C4R, "nppiYCbCr422ToBGR_8u_C2C4R");

  --*
  -- * 3 channel 8-bit unsigned planar YCbCr422 to 3 channel 8-bit unsigned packed BGR color conversion.
  -- * images.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array.
  -- * \param rSrcStep \ref source_planar_image_line_step_array.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYCbCr422ToBGR_8u_P3C3R
     (pSrc : System.Address;
      rSrcStep : access int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:1215
   pragma Import (C, nppiYCbCr422ToBGR_8u_P3C3R, "nppiYCbCr422ToBGR_8u_P3C3R");

  --* @}  
  --* @name RGBToCbYCr422 
  -- *  RGB to CbYCr422 color conversion.
  -- *
  -- * @{
  -- *
  --  

  --*
  -- * 3 channel 8-bit unsigned packed RGB to 2 channel 8-bit unsigned packed CbYCr422 color conversion.
  -- * images.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiRGBToCbYCr422_8u_C3C2R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:1237
   pragma Import (C, nppiRGBToCbYCr422_8u_C3C2R, "nppiRGBToCbYCr422_8u_C3C2R");

  --*
  -- * 3 channel 8-bit unsigned packed RGB first gets forward gamma corrected then converted to 2 channel 8-bit unsigned packed CbYCr422 color conversion.
  -- * images.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiRGBToCbYCr422Gamma_8u_C3C2R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:1250
   pragma Import (C, nppiRGBToCbYCr422Gamma_8u_C3C2R, "nppiRGBToCbYCr422Gamma_8u_C3C2R");

  --* @}  
  --* @name CbYCr422ToRGB 
  -- *  CbYCr422 to RGB color conversion.
  -- *
  -- * @{
  -- *
  --  

  --*
  -- * 2 channel 8-bit unsigned packed CbYCrC22 to 3 channel 8-bit unsigned packed RGB color conversion.
  -- * images.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCbYCr422ToRGB_8u_C2C3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:1272
   pragma Import (C, nppiCbYCr422ToRGB_8u_C2C3R, "nppiCbYCr422ToRGB_8u_C2C3R");

  --* @}  
  --* @name BGRToCbYCr422 
  -- *  BGR to CbYCr422 color conversion.
  -- *
  -- * @{
  -- *
  --  

  --*
  -- * 4 channel 8-bit unsigned packed BGR with alpha to 2 channel 8-bit unsigned packed CbYCr422 color conversion.
  -- * images.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiBGRToCbYCr422_8u_AC4C2R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:1294
   pragma Import (C, nppiBGRToCbYCr422_8u_AC4C2R, "nppiBGRToCbYCr422_8u_AC4C2R");

  --* @}  
  --* @name BGRToCbYCr422_709HDTV 
  -- *  BGR to CbYCr422_709HDTV color conversion.
  -- *
  -- * @{
  -- *
  --  

  --*
  -- * 3 channel 8-bit unsigned packed BGR to 2 channel 8-bit unsigned packed CbYCr422_709HDTV color conversion.
  -- * images.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiBGRToCbYCr422_709HDTV_8u_C3C2R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:1316
   pragma Import (C, nppiBGRToCbYCr422_709HDTV_8u_C3C2R, "nppiBGRToCbYCr422_709HDTV_8u_C3C2R");

  --*
  -- * 4 channel 8-bit unsigned packed BGR with alpha to 2 channel 8-bit unsigned packed CbYCr422_709HDTV color conversion.
  -- * images.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiBGRToCbYCr422_709HDTV_8u_AC4C2R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:1329
   pragma Import (C, nppiBGRToCbYCr422_709HDTV_8u_AC4C2R, "nppiBGRToCbYCr422_709HDTV_8u_AC4C2R");

  --* @}  
  --* @name CbYCr422ToBGR 
  -- *  CbYCr422 to BGR color conversion.
  -- *
  -- * @{
  -- *
  --  

  --*
  -- * 2 channel 8-bit unsigned packed CbYCr422 to 4 channel 8-bit unsigned packed BGR color conversion with alpha.
  -- * images.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nAval 8-bit unsigned alpha constant.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCbYCr422ToBGR_8u_C2C4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nAval : nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:1352
   pragma Import (C, nppiCbYCr422ToBGR_8u_C2C4R, "nppiCbYCr422ToBGR_8u_C2C4R");

  --* @}  
  --* @name CbYCr422ToBGR_709HDTV 
  -- *  CbYCr422 to BGR_709HDTV color conversion.
  -- *
  -- * @{
  -- *
  --  

  --*
  -- * 2 channel 8-bit unsigned packed CbYCr422 to 3 channel 8-bit unsigned packed BGR_709HDTV color conversion.
  -- * images.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCbYCr422ToBGR_709HDTV_8u_C2C3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:1374
   pragma Import (C, nppiCbYCr422ToBGR_709HDTV_8u_C2C3R, "nppiCbYCr422ToBGR_709HDTV_8u_C2C3R");

  --*
  -- * 2 channel 8-bit unsigned packed CbYCr422 to 4 channel 8-bit unsigned packed BGR_709HDTV color conversion with constant alpha.
  -- * images.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nAval 8-bit unsigned alpha constant.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCbYCr422ToBGR_709HDTV_8u_C2C4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nAval : nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:1388
   pragma Import (C, nppiCbYCr422ToBGR_709HDTV_8u_C2C4R, "nppiCbYCr422ToBGR_709HDTV_8u_C2C4R");

  --* @}  
  --* @name RGBToYCbCr420 
  -- *  RGB to YCbCr420 color conversion.
  -- *
  -- * @{
  -- *
  --  

  --*
  -- * 3 channel 8-bit unsigned packed RGB to 3 channel 8-bit unsigned planar YCbCr420 color conversion.
  -- * images.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_planar_image_pointer_array.
  -- * \param rDstStep \ref destination_planar_image_line_step_array.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiRGBToYCbCr420_8u_C3P3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : System.Address;
      rDstStep : access int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:1410
   pragma Import (C, nppiRGBToYCbCr420_8u_C3P3R, "nppiRGBToYCbCr420_8u_C3P3R");

  --* @}  
  --* @name YCbCr420ToRGB 
  -- *  YCbCr420 to RGB color conversion.
  -- *
  -- * @{
  -- *
  --  

  --*
  -- * 3 channel 8-bit unsigned planar YCbCr420 to 3 channel 8-bit unsigned packed RGB color conversion.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array.
  -- * \param rSrcStep \ref source_planar_image_line_step_array.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYCbCr420ToRGB_8u_P3C3R
     (pSrc : System.Address;
      rSrcStep : access int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:1431
   pragma Import (C, nppiYCbCr420ToRGB_8u_P3C3R, "nppiYCbCr420ToRGB_8u_P3C3R");

  --* @}  
  --* @name RGBToYCrCb420 
  -- *  RGB to YCrCb420 color conversion.
  -- *
  -- * @{
  -- *
  --  

  --*
  -- * 4 channel 8-bit unsigned packed RGB with alpha to 3 channel 8-bit unsigned planar YCrCb420 color conversion.
  -- * images.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_planar_image_pointer_array.
  -- * \param rDstStep \ref destination_planar_image_line_step_array.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiRGBToYCrCb420_8u_AC4P3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : System.Address;
      rDstStep : access int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:1453
   pragma Import (C, nppiRGBToYCrCb420_8u_AC4P3R, "nppiRGBToYCrCb420_8u_AC4P3R");

  --* @}  
  --* @name YCrCb420ToRGB 
  -- *  YCrCb420 to RGB color conversion.
  -- *
  -- * @{
  -- *
  --  

  --*
  -- * 3 channel 8-bit unsigned planar YCrCb420 to 4 channel 8-bit unsigned packed RGB color conversion with constant alpha.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array.
  -- * \param rSrcStep \ref source_planar_image_line_step_array.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nAval 8-bit unsigned alpha constant.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYCrCb420ToRGB_8u_P3C4R
     (pSrc : System.Address;
      rSrcStep : access int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nAval : nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:1475
   pragma Import (C, nppiYCrCb420ToRGB_8u_P3C4R, "nppiYCrCb420ToRGB_8u_P3C4R");

  --* @}  
  --* @name BGRToYCbCr420 
  -- *  BGR to YCbCr420 color conversion.
  -- *
  -- * @{
  -- *
  --  

  --*
  -- * 3 channel 8-bit unsigned packed BGR to 3 channel 8-bit unsigned planar YCbCr420 color conversion.
  -- * images.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_planar_image_pointer_array.
  -- * \param rDstStep \ref destination_planar_image_line_step_array.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiBGRToYCbCr420_8u_C3P3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : System.Address;
      rDstStep : access int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:1497
   pragma Import (C, nppiBGRToYCbCr420_8u_C3P3R, "nppiBGRToYCbCr420_8u_C3P3R");

  --*
  -- * 4 channel 8-bit unsigned packed BGR with alpha to 3 channel 8-bit unsigned planar YCbCr420 color conversion.
  -- * images.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_planar_image_pointer_array.
  -- * \param rDstStep \ref destination_planar_image_line_step_array.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiBGRToYCbCr420_8u_AC4P3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : System.Address;
      rDstStep : access int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:1510
   pragma Import (C, nppiBGRToYCbCr420_8u_AC4P3R, "nppiBGRToYCbCr420_8u_AC4P3R");

  --* @}  
  --* @name BGRToYCbCr420_709CSC 
  -- *  BGR to YCbCr420_709CSC color conversion.
  -- *
  -- * @{
  -- *
  --  

  --*
  -- * 3 channel 8-bit unsigned packed BGR to 3 channel 8-bit unsigned planar YCbCr420_709CSC color conversion.
  -- * images.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_planar_image_pointer_array.
  -- * \param rDstStep \ref destination_planar_image_line_step_array.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiBGRToYCbCr420_709CSC_8u_C3P3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : System.Address;
      rDstStep : access int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:1532
   pragma Import (C, nppiBGRToYCbCr420_709CSC_8u_C3P3R, "nppiBGRToYCbCr420_709CSC_8u_C3P3R");

  --*
  -- * 4 channel 8-bit unsigned packed BGR with alpha to 3 channel 8-bit unsigned planar YCbCr420_709CSC color conversion.
  -- * images.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_planar_image_pointer_array.
  -- * \param rDstStep \ref destination_planar_image_line_step_array.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiBGRToYCbCr420_709CSC_8u_AC4P3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : System.Address;
      rDstStep : access int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:1545
   pragma Import (C, nppiBGRToYCbCr420_709CSC_8u_AC4P3R, "nppiBGRToYCbCr420_709CSC_8u_AC4P3R");

  --* @}  
  --* @name BGRToYCbCr420_709HDTV 
  -- *  BGR to YCbCr420_709HDTV color conversion.
  -- *
  -- * @{
  -- *
  --  

  --*
  -- * 4 channel 8-bit unsigned packed BGR with alpha to 3 channel 8-bit unsigned planar YCbCr420_709HDTV color conversion.
  -- * images.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_planar_image_pointer_array.
  -- * \param rDstStep \ref destination_planar_image_line_step_array.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiBGRToYCbCr420_709HDTV_8u_AC4P3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : System.Address;
      rDstStep : access int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:1567
   pragma Import (C, nppiBGRToYCbCr420_709HDTV_8u_AC4P3R, "nppiBGRToYCbCr420_709HDTV_8u_AC4P3R");

  --* @}  
  --* @name BGRToYCrCb420_709CSC 
  -- *  BGR to YCrCb420_709CSC color conversion.
  -- * @{
  --  

  --*
  -- * 3 channel 8-bit unsigned packed BGR to 3 channel 8-bit unsigned planar YCrCb420_709CSC color conversion.
  -- * images.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_planar_image_pointer_array.
  -- * \param rDstStep \ref destination_planar_image_line_step_array.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiBGRToYCrCb420_709CSC_8u_C3P3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : System.Address;
      rDstStep : access int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:1587
   pragma Import (C, nppiBGRToYCrCb420_709CSC_8u_C3P3R, "nppiBGRToYCrCb420_709CSC_8u_C3P3R");

  --*
  -- * 4 channel 8-bit unsigned packed BGR with alpha to 3 channel 8-bit unsigned planar YCrCb420_709CSC color conversion.
  -- * images.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_planar_image_pointer_array.
  -- * \param rDstStep \ref destination_planar_image_line_step_array.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiBGRToYCrCb420_709CSC_8u_AC4P3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : System.Address;
      rDstStep : access int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:1600
   pragma Import (C, nppiBGRToYCrCb420_709CSC_8u_AC4P3R, "nppiBGRToYCrCb420_709CSC_8u_AC4P3R");

  --* @}  
  --* @name YCbCr420ToBGR 
  -- *  YCbCr420 to BGR color conversion.
  -- * @{
  --  

  --*
  -- * 3 channel 8-bit unsigned planar YCbCr420 to 3 channel 8-bit unsigned packed BGR color conversion.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array.
  -- * \param rSrcStep \ref source_planar_image_line_step_array.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYCbCr420ToBGR_8u_P3C3R
     (pSrc : System.Address;
      rSrcStep : access int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:1618
   pragma Import (C, nppiYCbCr420ToBGR_8u_P3C3R, "nppiYCbCr420ToBGR_8u_P3C3R");

  --*
  -- * 3 channel 8-bit unsigned planar YCbCr420 to 4 channel 8-bit unsigned packed BGR color conversion with constant alpha.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array.
  -- * \param rSrcStep \ref source_planar_image_line_step_array.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nAval 8-bit unsigned alpha constant.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYCbCr420ToBGR_8u_P3C4R
     (pSrc : System.Address;
      rSrcStep : access int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nAval : nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:1631
   pragma Import (C, nppiYCbCr420ToBGR_8u_P3C4R, "nppiYCbCr420ToBGR_8u_P3C4R");

  --* @}  
  --* @name YCbCr420ToBGR_709CSC 
  -- *  YCbCr420_709CSC to BGR color conversion.
  -- * @{
  --  

  --*
  -- * 3 channel 8-bit unsigned planar YCbCr420 to 3 channel 8-bit unsigned packed BGR_709CSC color conversion.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array.
  -- * \param rSrcStep \ref source_planar_image_line_step_array.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYCbCr420ToBGR_709CSC_8u_P3C3R
     (pSrc : System.Address;
      rSrcStep : access int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:1649
   pragma Import (C, nppiYCbCr420ToBGR_709CSC_8u_P3C3R, "nppiYCbCr420ToBGR_709CSC_8u_P3C3R");

  --* @}  
  --* @name YCbCr420ToBGR_709HDTV 
  -- *  YCbCr420_709HDTV to BGR color conversion.
  -- * @{
  --  

  --*
  -- * 3 channel 8-bit unsigned planar YCbCr420 to 4 channel 8-bit unsigned packed BGR_709HDTV color conversion with constant alpha.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array.
  -- * \param rSrcStep \ref source_planar_image_line_step_array.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nAval 8-bit unsigned alpha constant.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYCbCr420ToBGR_709HDTV_8u_P3C4R
     (pSrc : System.Address;
      rSrcStep : access int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nAval : nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:1668
   pragma Import (C, nppiYCbCr420ToBGR_709HDTV_8u_P3C4R, "nppiYCbCr420ToBGR_709HDTV_8u_P3C4R");

  --* @}  
  --* @name BGRToYCrCb420 
  -- *  BGR to YCrCb420 color conversion.
  -- * @{
  --  

  --*
  -- * 3 channel 8-bit unsigned packed BGR to 3 channel 8-bit unsigned planar YCrCb420 color conversion.
  -- * images.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_planar_image_pointer_array.
  -- * \param rDstStep \ref destination_planar_image_line_step_array.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiBGRToYCrCb420_8u_C3P3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : System.Address;
      rDstStep : access int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:1687
   pragma Import (C, nppiBGRToYCrCb420_8u_C3P3R, "nppiBGRToYCrCb420_8u_C3P3R");

  --*
  -- * 4 channel 8-bit unsigned packed BGR with alpha to 3 channel 8-bit unsigned planar YCrCb420 color conversion.
  -- * images.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_planar_image_pointer_array.
  -- * \param rDstStep \ref destination_planar_image_line_step_array.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiBGRToYCrCb420_8u_AC4P3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : System.Address;
      rDstStep : access int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:1700
   pragma Import (C, nppiBGRToYCrCb420_8u_AC4P3R, "nppiBGRToYCrCb420_8u_AC4P3R");

  --* @}  
  --* @name BGRToYCbCr411 
  -- *  BGR to YCbCr411 color conversion.
  -- * @{
  --  

  --*
  -- * 3 channel 8-bit unsigned packed BGR to 3 channel 8-bit unsigned planar YCbCr411 color conversion.
  -- * images.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_planar_image_pointer_array.
  -- * \param rDstStep \ref destination_planar_image_line_step_array.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiBGRToYCbCr411_8u_C3P3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : System.Address;
      rDstStep : access int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:1719
   pragma Import (C, nppiBGRToYCbCr411_8u_C3P3R, "nppiBGRToYCbCr411_8u_C3P3R");

  --*
  -- * 4 channel 8-bit unsigned packed BGR with alpha to 3 channel 8-bit unsigned planar YCbCr411 color conversion.
  -- * images.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_planar_image_pointer_array.
  -- * \param rDstStep \ref destination_planar_image_line_step_array.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiBGRToYCbCr411_8u_AC4P3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : System.Address;
      rDstStep : access int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:1732
   pragma Import (C, nppiBGRToYCbCr411_8u_AC4P3R, "nppiBGRToYCbCr411_8u_AC4P3R");

  --* @}  
  --* @name RGBToYCbCr411 
  -- *  RGB to YCbCr411 color conversion.
  -- * @{
  --  

  --*
  -- * 3 channel 8-bit unsigned packed RGB to 3 channel 8-bit unsigned planar YCbCr411 color conversion.
  -- * images.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_planar_image_pointer_array.
  -- * \param rDstStep \ref destination_planar_image_line_step_array.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiRGBToYCbCr411_8u_C3P3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : System.Address;
      rDstStep : access int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:1751
   pragma Import (C, nppiRGBToYCbCr411_8u_C3P3R, "nppiRGBToYCbCr411_8u_C3P3R");

  --*
  -- * 4 channel 8-bit unsigned packed RGB with alpha to 3 channel 8-bit unsigned planar YCbCr411 color conversion.
  -- * images.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_planar_image_pointer_array.
  -- * \param rDstStep \ref destination_planar_image_line_step_array.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiRGBToYCbCr411_8u_AC4P3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : System.Address;
      rDstStep : access int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:1764
   pragma Import (C, nppiRGBToYCbCr411_8u_AC4P3R, "nppiRGBToYCbCr411_8u_AC4P3R");

  --* @}  
  --* @name BGRToYCbCr 
  -- *  BGR to YCbCr color conversion.
  -- * @{
  --  

  --*
  -- * 3 channel 8-bit unsigned packed BGR to 3 channel 8-bit unsigned planar YCbCr color conversion.
  -- * images.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_planar_image_pointer_array.
  -- * \param nDstStep \ref destination_planar_image_line_step_array.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiBGRToYCbCr_8u_C3P3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : System.Address;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:1783
   pragma Import (C, nppiBGRToYCbCr_8u_C3P3R, "nppiBGRToYCbCr_8u_C3P3R");

  --*
  -- * 4 channel 8-bit unsigned packed BGR with alpha to 3 channel 8-bit unsigned planar YCbCr color conversion.
  -- * images.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_planar_image_pointer_array.
  -- * \param nDstStep \ref destination_planar_image_line_step_array.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiBGRToYCbCr_8u_AC4P3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : System.Address;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:1796
   pragma Import (C, nppiBGRToYCbCr_8u_AC4P3R, "nppiBGRToYCbCr_8u_AC4P3R");

  --*
  -- * 4 channel 8-bit unsigned packed BGR with alpha to 4 channel 8-bit unsigned planar YCbCr color conversion.
  -- * images.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_planar_image_pointer_array.
  -- * \param nDstStep \ref destination_planar_image_line_step_array.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiBGRToYCbCr_8u_AC4P4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : System.Address;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:1809
   pragma Import (C, nppiBGRToYCbCr_8u_AC4P4R, "nppiBGRToYCbCr_8u_AC4P4R");

  --* @}  
  --* @name YCbCr411ToBGR 
  -- *  YCbCr411 to BGR color conversion.
  -- * @{
  --  

  --*
  -- * 3 channel 8-bit unsigned planar YCbCr411 to 3 channel 8-bit unsigned packed BGR color conversion.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array.
  -- * \param rSrcStep \ref source_planar_image_line_step_array.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYCbCr411ToBGR_8u_P3C3R
     (pSrc : System.Address;
      rSrcStep : access int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:1827
   pragma Import (C, nppiYCbCr411ToBGR_8u_P3C3R, "nppiYCbCr411ToBGR_8u_P3C3R");

  --*
  -- * 3 channel 8-bit unsigned planar YCbCr411 to 4 channel 8-bit unsigned packed BGR color conversion with constant alpha.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array.
  -- * \param rSrcStep \ref source_planar_image_line_step_array.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nAval 8-bit unsigned alpha constant.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYCbCr411ToBGR_8u_P3C4R
     (pSrc : System.Address;
      rSrcStep : access int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nAval : nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:1840
   pragma Import (C, nppiYCbCr411ToBGR_8u_P3C4R, "nppiYCbCr411ToBGR_8u_P3C4R");

  --* @}  
  --* @name YCbCr411ToRGB 
  -- *  YCbCr411 to RGB color conversion.
  -- * @{
  --  

  --*
  -- * 3 channel 8-bit unsigned planar YCbCr411 to 3 channel 8-bit unsigned packed RGB color conversion.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array.
  -- * \param rSrcStep \ref source_planar_image_line_step_array.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYCbCr411ToRGB_8u_P3C3R
     (pSrc : System.Address;
      rSrcStep : access int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:1858
   pragma Import (C, nppiYCbCr411ToRGB_8u_P3C3R, "nppiYCbCr411ToRGB_8u_P3C3R");

  --*
  -- * 3 channel 8-bit unsigned planar YCbCr411 to 4 channel 8-bit unsigned packed RGB color conversion with constant alpha.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array.
  -- * \param rSrcStep \ref source_planar_image_line_step_array.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nAval 8-bit unsigned alpha constant.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYCbCr411ToRGB_8u_P3C4R
     (pSrc : System.Address;
      rSrcStep : access int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nAval : nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:1871
   pragma Import (C, nppiYCbCr411ToRGB_8u_P3C4R, "nppiYCbCr411ToRGB_8u_P3C4R");

  --* @}  
  --* @name RGBToXYZ 
  -- *  RGB to XYZ color conversion.
  -- *
  -- *  Here is how NPP converts gamma corrected RGB or BGR to XYZ.
  -- *
  -- *  \code
  -- *  Npp32f nNormalizedR = (Npp32f)R * 0.003921569F; // / 255.0F
  -- *  Npp32f nNormalizedG = (Npp32f)G * 0.003921569F;
  -- *  Npp32f nNormalizedB = (Npp32f)B * 0.003921569F;
  -- *  Npp32f nX = 0.412453F * nNormalizedR + 0.35758F  * nNormalizedG + 0.180423F * nNormalizedB; 
  -- *  if (nX > 1.0F)
  -- *      nX = 1.0F;
  -- *  Npp32f nY = 0.212671F * nNormalizedR + 0.71516F  * nNormalizedG + 0.072169F * nNormalizedB;
  -- *  if (nY > 1.0F)
  -- *      nY = 1.0F;
  -- *  Npp32f nZ = 0.019334F * nNormalizedR + 0.119193F * nNormalizedG + 0.950227F * nNormalizedB;
  -- *  if (nZ > 1.0F)
  -- *      nZ = 1.0F;
  -- *  X = (Npp8u)(nX * 255.0F);
  -- *  Y = (Npp8u)(nY * 255.0F);
  -- *  Z = (Npp8u)(nZ * 255.0F);
  -- *  \endcode
  -- *
  -- * @{
  --  

  --*
  -- * 3 channel 8-bit unsigned packed RGB to 3 channel 8-bit unsigned packed XYZ color conversion.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiRGBToXYZ_8u_C3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:1910
   pragma Import (C, nppiRGBToXYZ_8u_C3R, "nppiRGBToXYZ_8u_C3R");

  --*
  -- * 4 channel 8-bit unsigned packed RGB with alpha to 4 channel 8-bit unsigned packed XYZ with alpha color conversion.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiRGBToXYZ_8u_AC4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:1922
   pragma Import (C, nppiRGBToXYZ_8u_AC4R, "nppiRGBToXYZ_8u_AC4R");

  --* @}  
  --* @name XYZToRGB 
  -- *  XYZ to RGB color conversion.
  -- *
  -- *  Here is how NPP converts XYZ to gamma corrected RGB or BGR.  The code assumes that X,Y, and Z values are in the range [0..1].
  -- *
  -- *  \code
  -- *  Npp32f nNormalizedX = (Npp32f)X * 0.003921569F; // / 255.0F
  -- *  Npp32f nNormalizedY = (Npp32f)Y * 0.003921569F;
  -- *  Npp32f nNormalizedZ = (Npp32f)Z * 0.003921569F;
  -- *  Npp32f nR = 3.240479F * nNormalizedX - 1.53715F  * nNormalizedY - 0.498535F * nNormalizedZ; 
  -- *  if (nR > 1.0F)
  -- *      nR = 1.0F;
  -- *  Npp32f nG = -0.969256F * nNormalizedX + 1.875991F  * nNormalizedY + 0.041556F * nNormalizedZ;
  -- *  if (nG > 1.0F)
  -- *      nG = 1.0F;
  -- *  Npp32f nB = 0.055648F * nNormalizedX - 0.204043F * nNormalizedY + 1.057311F * nNormalizedZ;
  -- *  if (nB > 1.0F)
  -- *      nB = 1.0F;
  -- *  R = (Npp8u)(nR * 255.0F);
  -- *  G = (Npp8u)(nG * 255.0F);
  -- *  B = (Npp8u)(nB * 255.0F);
  -- *  \endcode
  -- *
  -- * @{
  --  

  --*
  -- * 3 channel 8-bit unsigned packed XYZ to 3 channel 8-bit unsigned packed RGB color conversion.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiXYZToRGB_8u_C3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:1961
   pragma Import (C, nppiXYZToRGB_8u_C3R, "nppiXYZToRGB_8u_C3R");

  --*
  -- * 4 channel 8-bit unsigned packed XYZ with alpha to 4 channel 8-bit unsigned packed RGB with alpha color conversion.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiXYZToRGB_8u_AC4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:1973
   pragma Import (C, nppiXYZToRGB_8u_AC4R, "nppiXYZToRGB_8u_AC4R");

  --* @}  
  --* @name RGBToLUV 
  -- *  RGB to LUV color conversion.
  -- *
  -- *  Here is how NPP converts gamma corrected RGB or BGR to CIE LUV using the CIE XYZ D65 white point with a Y luminance of 1.0.
  -- *  The computed values of the L component are in the range [0..100], U component in the range [-134..220], 
  -- *  and V component in the range [-140..122]. The code uses cbrtf() the 32 bit floating point cube root math function.
  -- *
  -- *  \code
  -- *  // use CIE D65 chromaticity coordinates
  -- *  #define nCIE_XYZ_D65_xn 0.312713F
  -- *  #define nCIE_XYZ_D65_yn 0.329016F
  -- *  #define nn_DIVISOR (-2.0F * nCIE_XYZ_D65_xn + 12.0F * nCIE_XYZ_D65_yn + 3.0F)
  -- *  #define nun (4.0F * nCIE_XYZ_D65_xn / nn_DIVISOR)
  -- *  #define nvn (9.0F * nCIE_XYZ_D65_yn / nn_DIVISOR)
  -- *    
  -- *  // First convert to XYZ
  -- *  Npp32f nNormalizedR = (Npp32f)R * 0.003921569F; // / 255.0F
  -- *  Npp32f nNormalizedG = (Npp32f)G * 0.003921569F;
  -- *  Npp32f nNormalizedB = (Npp32f)B * 0.003921569F;
  -- *  Npp32f nX = 0.412453F * nNormalizedR + 0.35758F  * nNormalizedG + 0.180423F * nNormalizedB; 
  -- *  Npp32f nY = 0.212671F * nNormalizedR + 0.71516F  * nNormalizedG + 0.072169F * nNormalizedB;
  -- *  Npp32f nZ = 0.019334F * nNormalizedR + 0.119193F * nNormalizedG + 0.950227F * nNormalizedB;
  -- *  // Now calculate LUV from the XYZ value
  -- *  Npp32f nTemp = nX + 15.0F * nY + 3.0F * nZ;
  -- *  Npp32f nu = 4.0F * nX / nTemp;
  -- *  Npp32f nv = 9.0F * nY / nTemp;
  -- *  Npp32f nL = 116.0F * cbrtf(nY) - 16.0F;
  -- *  if (nL < 0.0F)
  -- *      nL = 0.0F;
  -- *  if (nL > 100.0F)
  -- *      nL = 100.0F;
  -- *  nTemp = 13.0F * nL;
  -- *  Npp32f nU = nTemp * (nu - nun);
  -- *  if (nU < -134.0F)
  -- *      nU = -134.0F;
  -- *  if (nU > 220.0F)
  -- *      nU = 220.0F;
  -- *  Npp32f nV = nTemp * (nv - nvn);
  -- *  if (nV < -140.0F)
  -- *      nV = -140.0F;
  -- *  if (nV > 122.0F)
  -- *      nV = 122.0F;
  -- *  L = (Npp8u)(nL * 255.0F * 0.01F); // / 100.0F
  -- *  U = (Npp8u)((nU + 134.0F) * 255.0F * 0.0028249F); // / 354.0F
  -- *  V = (Npp8u)((nV + 140.0F) * 255.0F * 0.0038168F); // / 262.0F
  -- *  \endcode
  -- *
  -- * @{
  --  

  --*
  -- * 3 channel 8-bit unsigned packed RGB to 3 channel 8-bit unsigned packed LUV color conversion.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiRGBToLUV_8u_C3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:2036
   pragma Import (C, nppiRGBToLUV_8u_C3R, "nppiRGBToLUV_8u_C3R");

  --*
  -- * 4 channel 8-bit unsigned packed RGB with alpha to 4 channel 8-bit unsigned packed LUV with alpha color conversion.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiRGBToLUV_8u_AC4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:2048
   pragma Import (C, nppiRGBToLUV_8u_AC4R, "nppiRGBToLUV_8u_AC4R");

  --* @}  
  --* @name LUVToRGB 
  -- *  LUV to RGB color conversion.
  -- *
  -- *  Here is how NPP converts CIE LUV to gamma corrected RGB or BGR using the CIE XYZ D65 white point with a Y luminance of 1.0.
  -- *  The code uses powf() the 32 bit floating point power math function. 
  -- *
  -- *  \code
  -- *  // use CIE D65 chromaticity coordinates
  -- *  #define nCIE_XYZ_D65_xn 0.312713F
  -- *  #define nCIE_XYZ_D65_yn 0.329016F
  -- *  #define nn_DIVISOR (-2.0F * nCIE_XYZ_D65_xn + 12.0F * nCIE_XYZ_D65_yn + 3.0F)
  -- *  #define nun (4.0F * nCIE_XYZ_D65_xn / nn_DIVISOR)
  -- *  #define nvn (9.0F * nCIE_XYZ_D65_yn / nn_DIVISOR)
  -- *    
  -- *  // First convert normalized LUV back to original CIE LUV range
  -- *  Npp32f nL = (Npp32f)L * 100.0F * 0.003921569F;  // / 255.0F
  -- *  Npp32f nU = ((Npp32f)U * 354.0F * 0.003921569F) - 134.0F;
  -- *  Npp32f nV = ((Npp32f)V * 262.0F * 0.003921569F) - 140.0F;
  -- *  // Now convert LUV to CIE XYZ
  -- *  Npp32f nTemp = 13.0F * nL;
  -- *  Npp32f nu = nU / nTemp + nun;
  -- *  Npp32f nv = nV / nTemp + nvn;
  -- *  Npp32f nNormalizedY;
  -- *  if (nL > 7.9996248F)
  -- *  {
  -- *      nNormalizedY = (nL + 16.0F) * 0.008621F; // / 116.0F
  -- *      nNormalizedY = powf(nNormalizedY, 3.0F);
  -- *  }
  -- *  else
  -- *  {    
  -- *      nNormalizedY = nL * 0.001107F; // / 903.3F
  -- *  }    
  -- *  Npp32f nNormalizedX = (-9.0F * nNormalizedY * nu) / ((nu - 4.0F) * nv - nu * nv);
  -- *  Npp32f nNormalizedZ = (9.0F * nNormalizedY - 15.0F * nv * nNormalizedY - nv * nNormalizedX) / (3.0F * nv);
  -- *  Npp32f nR = 3.240479F * nNormalizedX - 1.53715F  * nNormalizedY - 0.498535F * nNormalizedZ; 
  -- *  if (nR > 1.0F)
  -- *      nR = 1.0F;
  -- *  if (nR < 0.0F)
  -- *      nR = 0.0F;
  -- *  Npp32f nG = -0.969256F * nNormalizedX + 1.875991F  * nNormalizedY + 0.041556F * nNormalizedZ;
  -- *  if (nG > 1.0F)
  -- *      nG = 1.0F;
  -- *  if (nG < 0.0F)
  -- *      nG = 0.0F;
  -- *  Npp32f nB = 0.055648F * nNormalizedX - 0.204043F * nNormalizedY + 1.057311F * nNormalizedZ;
  -- *  if (nB > 1.0F)
  -- *      nB = 1.0F;
  -- *  if (nB < 0.0F)
  -- *      nB = 0.0F;
  -- *  R = (Npp8u)(nR * 255.0F);
  -- *  G = (Npp8u)(nG * 255.0F);
  -- *  B = (Npp8u)(nB * 255.0F);
  -- *  \endcode
  -- *
  -- * @{
  --  

  --*
  -- * 3 channel 8-bit unsigned packed LUV to 3 channel 8-bit unsigned packed RGB color conversion.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiLUVToRGB_8u_C3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:2118
   pragma Import (C, nppiLUVToRGB_8u_C3R, "nppiLUVToRGB_8u_C3R");

  --*
  -- * 4 channel 8-bit unsigned packed LUV with alpha to 4 channel 8-bit unsigned packed RGB with alpha color conversion.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiLUVToRGB_8u_AC4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:2130
   pragma Import (C, nppiLUVToRGB_8u_AC4R, "nppiLUVToRGB_8u_AC4R");

  --* @}  
  --* @name BGRToLab 
  -- *  BGR to Lab color conversion.
  -- *
  -- *  This is how NPP converts gamma corrected BGR or RGB to Lab using the CIE Lab D65 white point with a Y luminance of 1.0.
  -- *  The computed values of the L component are in the range [0..100], a and b component values are in the range [-128..127].
  -- *  The code uses cbrtf() the 32 bit floating point cube root math function.
  -- *
  -- *  \code
  -- *  // use CIE Lab chromaticity coordinates
  -- *  #define nCIE_LAB_D65_xn 0.950455F
  -- *  #define nCIE_LAB_D65_yn 1.0F
  -- *  #define nCIE_LAB_D65_zn 1.088753F
  -- *  // First convert to XYZ
  -- *  Npp32f nNormalizedR = (Npp32f)R * 0.003921569F; // / 255.0F
  -- *  Npp32f nNormalizedG = (Npp32f)G * 0.003921569F;
  -- *  Npp32f nNormalizedB = (Npp32f)B * 0.003921569F;
  -- *  Npp32f nX = 0.412453F * nNormalizedR + 0.35758F  * nNormalizedG + 0.180423F * nNormalizedB; 
  -- *  Npp32f nY = 0.212671F * nNormalizedR + 0.71516F  * nNormalizedG + 0.072169F * nNormalizedB;
  -- *  Npp32f nZ = 0.019334F * nNormalizedR + 0.119193F * nNormalizedG + 0.950227F * nNormalizedB;
  -- *  Npp32f nL = cbrtf(nY);
  -- *  Npp32f nA;
  -- *  Npp32f nB;
  -- *  Npp32f nfX = nX * 1.052128F; // / nCIE_LAB_D65_xn; 
  -- *  Npp32f nfY = nY;
  -- *  Npp32f nfZ = nZ * 0.918482F; // / nCIE_LAB_D65_zn;
  -- *  nfY = nL - 16.0F;
  -- *  nL = 116.0F * nL - 16.0F;
  -- *  nA = cbrtf(nfX) - 16.0F;
  -- *  nA = 500.0F * (nA - nfY);
  -- *  nB = cbrtf(nfZ) - 16.0F;
  -- *  nB = 200.0F * (nfY - nB);
  -- *  // Now scale Lab range
  -- *  nL = nL * 255.0F * 0.01F; // / 100.0F
  -- *  nA = nA + 128.0F;
  -- *  nB = nB + 128.0F;
  -- *  L = (Npp8u)nL;
  -- *  a = (Npp8u)nA;
  -- *  b = (Npp8u)nB;
  -- *  \endcode
  -- * 
  -- * @{
  --  

  --*
  -- * 3 channel 8-bit unsigned packed BGR to 3 channel 8-bit unsigned packed Lab color conversion.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiBGRToLab_8u_C3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:2186
   pragma Import (C, nppiBGRToLab_8u_C3R, "nppiBGRToLab_8u_C3R");

  --* @}  
  --* @name LabToBGR 
  -- *  Lab to BGR color conversion.
  -- *
  -- *  This is how NPP converts Lab to gamma corrected BGR or RGB using the CIE Lab D65 white point with a Y luminance of 1.0.
  -- *  The code uses powf() the 32 bit floating point power math function. 
  -- *
  -- *  \code
  -- *  // use CIE Lab chromaticity coordinates
  -- *  #define nCIE_LAB_D65_xn 0.950455F
  -- *  #define nCIE_LAB_D65_yn 1.0F
  -- *  #define nCIE_LAB_D65_zn 1.088753F
  -- *  // First convert Lab back to original range then to CIE XYZ
  -- *  Npp32f nL = (Npp32f)L * 100.0F * 0.003921569F;  // / 255.0F
  -- *  Npp32f nA = (Npp32f)a - 128.0F;
  -- *  Npp32f nB = (Npp32f)b - 128.0F;
  -- *  Npp32f nP = (nL + 16.0F) * 0.008621F; // / 116.0F
  -- *  Npp32f nNormalizedY = nP * nP * nP; // powf(nP, 3.0F);
  -- *  Npp32f nNormalizedX = nCIE_LAB_D65_xn * powf((nP + nA * 0.002F), 3.0F); // / 500.0F
  -- *  Npp32f nNormalizedZ = nCIE_LAB_D65_zn * powf((nP - nB * 0.005F), 3.0F); // / 200.0F
  -- *  Npp32f nR = 3.240479F * nNormalizedX - 1.53715F  * nNormalizedY - 0.498535F * nNormalizedZ; 
  -- *  if (nR > 1.0F)
  -- *      nR = 1.0F;
  -- *  Npp32f nG = -0.969256F * nNormalizedX + 1.875991F  * nNormalizedY + 0.041556F * nNormalizedZ;
  -- *  if (nG > 1.0F)
  -- *      nG = 1.0F;
  -- *  nB = 0.055648F * nNormalizedX - 0.204043F * nNormalizedY + 1.057311F * nNormalizedZ;
  -- *  if (nB > 1.0F)
  -- *      nB = 1.0F;
  -- *  R = (Npp8u)(nR * 255.0F);
  -- *  G = (Npp8u)(nG * 255.0F);
  -- *  B = (Npp8u)(nB * 255.0F);
  -- *  \endcode
  -- *
  -- * @{
  --  

  --*
  -- * 3 channel 8-bit unsigned packed Lab to 3 channel 8-bit unsigned packed BGR color conversion.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiLabToBGR_8u_C3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:2235
   pragma Import (C, nppiLabToBGR_8u_C3R, "nppiLabToBGR_8u_C3R");

  --* @}  
  --* @name RGBToYCC 
  -- *  RGB to PhotoYCC color conversion.
  -- *
  -- *  This is how NPP converts gamma corrected BGR or RGB to PhotoYCC.
  -- *  The computed Y, C1, C2 values are then quantized and converted to fit in the range [0..1] before expanding to 8 bits.
  -- *
  -- *  \code
  -- *  Npp32f nNormalizedR = (Npp32f)R * 0.003921569F; // / 255.0F
  -- *  Npp32f nNormalizedG = (Npp32f)G * 0.003921569F;
  -- *  Npp32f nNormalizedB = (Npp32f)B * 0.003921569F;
  -- *  Npp32f nY = 0.299F * nNormalizedR + 0.587F  * nNormalizedG + 0.114F * nNormalizedB; 
  -- *  Npp32f nC1 = nNormalizedB - nY;
  -- *  nC1 = 111.4F * 0.003921569F * nC1 + 156.0F * 0.003921569F;
  -- *  Npp32f nC2 = nNormalizedR - nY;
  -- *  nC2 = 135.64F * 0.003921569F * nC2 + 137.0F * 0.003921569F;
  -- *  nY = 1.0F * 0.713267F * nY; // / 1.402F
  -- *  Y = (Npp8u)(nY  * 255.0F);
  -- *  C1 = (Npp8u)(nC1 * 255.0F);
  -- *  C2 = (Npp8u)(nC2 * 255.0F);
  -- *  \endcode
  -- * 
  -- * @{
  --  

  --*
  -- * 3 channel 8-bit unsigned packed RGB to 3 channel 8-bit unsigned packed YCC color conversion.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiRGBToYCC_8u_C3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:2272
   pragma Import (C, nppiRGBToYCC_8u_C3R, "nppiRGBToYCC_8u_C3R");

  --*
  -- * 4 channel 8-bit unsigned packed RGB with alpha to 4 channel 8-bit unsigned packed YCC with alpha color conversion.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiRGBToYCC_8u_AC4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:2284
   pragma Import (C, nppiRGBToYCC_8u_AC4R, "nppiRGBToYCC_8u_AC4R");

  --* @}  
  --* @name YCCToRGB 
  -- *  PhotoYCC to RGB color conversion.
  -- *
  -- *  This is how NPP converts PhotoYCC to gamma corrected RGB or BGR.
  -- *
  -- *  \code
  -- *  Npp32f nNormalizedY  = ((Npp32f)Y * 0.003921569F) * 1.3584F;  // / 255.0F
  -- *  Npp32f nNormalizedC1 = (((Npp32f)C1 * 0.003921569F) - 156.0F * 0.003921569F) * 2.2179F;
  -- *  Npp32f nNormalizedC2 = (((Npp32f)C2 * 0.003921569F) - 137.0F * 0.003921569F) * 1.8215F;
  -- *  Npp32f nR = nNormalizedY + nNormalizedC2;
  -- *  if (nR > 1.0F)
  -- *      nR = 1.0F;
  -- *  Npp32f nG = nNormalizedY - 0.194F  * nNormalizedC1 - 0.509F * nNormalizedC2;
  -- *  if (nG > 1.0F)
  -- *      nG = 1.0F;
  -- *  Npp32f nB = nNormalizedY + nNormalizedC1;
  -- *  if (nB > 1.0F)
  -- *      nB = 1.0F;
  -- *  R = (Npp8u)(nR * 255.0F);
  -- *  G = (Npp8u)(nG * 255.0F);
  -- *  B = (Npp8u)(nB * 255.0F);
  -- *  \endcode
  -- *
  -- * @{
  --  

  --*
  -- * 3 channel 8-bit unsigned packed YCC to 3 channel 8-bit unsigned packed RGB color conversion.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYCCToRGB_8u_C3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:2323
   pragma Import (C, nppiYCCToRGB_8u_C3R, "nppiYCCToRGB_8u_C3R");

  --*
  -- * 4 channel 8-bit unsigned packed YCC with alpha to 4 channel 8-bit unsigned packed RGB with alpha color conversion.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYCCToRGB_8u_AC4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:2335
   pragma Import (C, nppiYCCToRGB_8u_AC4R, "nppiYCCToRGB_8u_AC4R");

  --* @}  
  --* @name RGBToHLS 
  -- *  RGB to HLS color conversion.
  -- *
  -- *  This is how NPP converts gamma corrected RGB or BGR to HLS.
  -- *  This code uses the fmaxf() and fminf() 32 bit floating point math functions.
  -- *
  -- *  \code
  -- *  Npp32f nNormalizedR = (Npp32f)R * 0.003921569F;  // / 255.0F
  -- *  Npp32f nNormalizedG = (Npp32f)G * 0.003921569F;
  -- *  Npp32f nNormalizedB = (Npp32f)B * 0.003921569F;
  -- *  Npp32f nS;
  -- *  Npp32f nH;
  -- *  // Lightness
  -- *  Npp32f nMax = fmaxf(nNormalizedR, nNormalizedG);
  -- *         nMax = fmaxf(nMax, nNormalizedB);
  -- *  Npp32f nMin = fminf(nNormalizedR, nNormalizedG);
  -- *         nMin = fminf(nMin, nNormalizedB);
  -- *  Npp32f nL = (nMax + nMin) * 0.5F;
  -- *  Npp32f nDivisor = nMax - nMin;
  -- *  // Saturation
  -- *  if (nDivisor == 0.0F) // achromatics case
  -- *  {
  -- *      nS = 0.0F;
  -- *      nH = 0.0F;
  -- *  }
  -- *  else // chromatics case
  -- *  {
  -- *      if (nL > 0.5F)
  -- *          nS = nDivisor / (1.0F - (nMax + nMin - 1.0F));
  -- *      else
  -- *          nS = nDivisor / (nMax + nMin);
  -- *  }
  -- *  // Hue
  -- *  Npp32f nCr = (nMax - nNormalizedR) / nDivisor;
  -- *  Npp32f nCg = (nMax - nNormalizedG) / nDivisor;
  -- *  Npp32f nCb = (nMax - nNormalizedB) / nDivisor;
  -- *  if (nNormalizedR == nMax)
  -- *      nH = nCb - nCg;
  -- *  else if (nNormalizedG == nMax)
  -- *      nH = 2.0F + nCr - nCb;
  -- *  else if (nNormalizedB == nMax)
  -- *      nH = 4.0F + nCg - nCr;
  -- *  nH = nH * 0.166667F; // / 6.0F       
  -- *  if (nH < 0.0F)
  -- *      nH = nH + 1.0F;
  -- *  H = (Npp8u)(nH * 255.0F);
  -- *  L = (Npp8u)(nL * 255.0F);
  -- *  S = (Npp8u)(nS * 255.0F);
  -- *  \endcode
  -- *
  -- * @{
  --  

  --*
  -- * 3 channel 8-bit unsigned packed RGB to 3 channel 8-bit unsigned packed HLS color conversion.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiRGBToHLS_8u_C3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:2401
   pragma Import (C, nppiRGBToHLS_8u_C3R, "nppiRGBToHLS_8u_C3R");

  --*
  -- * 4 channel 8-bit unsigned packed RGB with alpha to 4 channel 8-bit unsigned packed HLS with alpha color conversion.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiRGBToHLS_8u_AC4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:2413
   pragma Import (C, nppiRGBToHLS_8u_AC4R, "nppiRGBToHLS_8u_AC4R");

  --* @}  
  --* @name HLSToRGB 
  -- *  HLS to RGB color conversion.
  -- *
  -- *  This is how NPP converts HLS to gamma corrected RGB or BGR.
  -- *
  -- *  \code
  -- *  Npp32f nNormalizedH = (Npp32f)H * 0.003921569F;  // / 255.0F
  -- *  Npp32f nNormalizedL = (Npp32f)L * 0.003921569F;
  -- *  Npp32f nNormalizedS = (Npp32f)S * 0.003921569F;
  -- *  Npp32f nM1;
  -- *  Npp32f nM2;
  -- *  Npp32f nR;
  -- *  Npp32f nG;
  -- *  Npp32f nB;
  -- *  Npp32f nh = 0.0F;
  -- *  if (nNormalizedL <= 0.5F)
  -- *      nM2 = nNormalizedL * (1.0F + nNormalizedS);
  -- *  else
  -- *      nM2 = nNormalizedL + nNormalizedS - nNormalizedL * nNormalizedS;
  -- *  nM1 = 2.0F * nNormalizedL - nM2;
  -- *  if (nNormalizedS == 0.0F)
  -- *      nR = nG = nB = nNormalizedL;
  -- *  else
  -- *  {
  -- *      nh = nNormalizedH + 0.3333F;
  -- *      if (nh > 1.0F)
  -- *          nh -= 1.0F;
  -- *  }
  -- *  Npp32f nMDiff = nM2 - nM1;
  -- *  if (0.6667F < nh)
  -- *      nR = nM1;
  -- *  else
  -- *  {    
  -- *      if (nh < 0.1667F)
  -- *          nR = (nM1 + nMDiff * nh * 6.0F); // / 0.1667F
  -- *      else if (nh < 0.5F)
  -- *          nR = nM2;
  -- *      else
  -- *          nR = nM1 + nMDiff * ( 0.6667F - nh ) * 6.0F; // / 0.1667F
  -- *  }
  -- *  if (nR > 1.0F)
  -- *      nR = 1.0F;     
  -- *  nh = nNormalizedH;
  -- *  if (0.6667F < nh)
  -- *      nG = nM1;
  -- *  else
  -- *  {
  -- *      if (nh < 0.1667F)
  -- *          nG = (nM1 + nMDiff * nh * 6.0F); // / 0.1667F
  -- *      else if (nh < 0.5F)
  -- *          nG = nM2;
  -- *      else
  -- *          nG = nM1 + nMDiff * (0.6667F - nh ) * 6.0F; // / 0.1667F
  -- *  }
  -- *  if (nG > 1.0F)
  -- *      nG = 1.0F;     
  -- *  nh = nNormalizedH - 0.3333F;
  -- *  if (nh < 0.0F)
  -- *      nh += 1.0F;
  -- *  if (0.6667F < nh)
  -- *      nB = nM1;
  -- *  else
  -- *  {
  -- *      if (nh < 0.1667F)
  -- *          nB = (nM1 + nMDiff * nh * 6.0F); // / 0.1667F
  -- *      else if (nh < 0.5F)
  -- *          nB = nM2;
  -- *      else
  -- *          nB = nM1 + nMDiff * (0.6667F - nh ) * 6.0F; // / 0.1667F
  -- *  }        
  -- *  if (nB > 1.0F)
  -- *      nB = 1.0F;     
  -- *  R = (Npp8u)(nR * 255.0F);
  -- *  G = (Npp8u)(nG * 255.0F);
  -- *  B = (Npp8u)(nB * 255.0F);
  -- *  \endcode
  -- *
  -- * @{
  --  

  --*
  -- * 3 channel 8-bit unsigned packed HLS to 3 channel 8-bit unsigned packed RGB color conversion.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiHLSToRGB_8u_C3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:2506
   pragma Import (C, nppiHLSToRGB_8u_C3R, "nppiHLSToRGB_8u_C3R");

  --*
  -- * 4 channel 8-bit unsigned packed HLS with alpha to 4 channel 8-bit unsigned packed RGB with alpha color conversion.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiHLSToRGB_8u_AC4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:2518
   pragma Import (C, nppiHLSToRGB_8u_AC4R, "nppiHLSToRGB_8u_AC4R");

  --* @}  
  --* @name BGRToHLS 
  -- *  BGR to HLS color conversion.
  -- * @{
  --  

  --*
  -- * 4 channel 8-bit unsigned packed BGR with alpha to 4 channel 8-bit unsigned packed HLS with alpha color conversion.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiBGRToHLS_8u_AC4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:2536
   pragma Import (C, nppiBGRToHLS_8u_AC4R, "nppiBGRToHLS_8u_AC4R");

  --*
  -- * 3 channel 8-bit unsigned packed BGR to 3 channel 8-bit unsigned planar HLS color conversion.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_planar_image_pointer_array.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiBGRToHLS_8u_C3P3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : System.Address;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:2548
   pragma Import (C, nppiBGRToHLS_8u_C3P3R, "nppiBGRToHLS_8u_C3P3R");

  --*
  -- * 4 channel 8-bit unsigned packed BGR with alpha to 4 channel 8-bit unsigned planar HLS with alpha color conversion.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_planar_image_pointer_array.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiBGRToHLS_8u_AC4P4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : System.Address;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:2560
   pragma Import (C, nppiBGRToHLS_8u_AC4P4R, "nppiBGRToHLS_8u_AC4P4R");

  --*
  -- * 3 channel 8-bit unsigned planar BGR to 3 channel 8-bit unsigned packed HLS color conversion.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiBGRToHLS_8u_P3C3R
     (pSrc : System.Address;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:2572
   pragma Import (C, nppiBGRToHLS_8u_P3C3R, "nppiBGRToHLS_8u_P3C3R");

  --*
  -- * 4 channel 8-bit unsigned planar BGR with alpha to 4 channel 8-bit unsigned packed HLS with alpha color conversion.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiBGRToHLS_8u_AP4C4R
     (pSrc : System.Address;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:2584
   pragma Import (C, nppiBGRToHLS_8u_AP4C4R, "nppiBGRToHLS_8u_AP4C4R");

  --*
  -- * 3 channel 8-bit unsigned planar BGR to 3 channel 8-bit unsigned planar HLS color conversion.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_planar_image_pointer_array.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiBGRToHLS_8u_P3R
     (pSrc : System.Address;
      nSrcStep : int;
      pDst : System.Address;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:2596
   pragma Import (C, nppiBGRToHLS_8u_P3R, "nppiBGRToHLS_8u_P3R");

  --*
  -- * 4 channel 8-bit unsigned planar BGR with alpha to 4 channel 8-bit unsigned planar HLS with alpha color conversion.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_planar_image_pointer_array.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiBGRToHLS_8u_AP4R
     (pSrc : System.Address;
      nSrcStep : int;
      pDst : System.Address;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:2608
   pragma Import (C, nppiBGRToHLS_8u_AP4R, "nppiBGRToHLS_8u_AP4R");

  --* @}  
  --* @name HLSToBGR 
  -- *  HLS to BGR color conversion.
  -- * @{
  --  

  --*
  -- * 3 channel 8-bit unsigned packed HLS to 3 channel 8-bit unsigned planar BGR color conversion.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_planar_image_pointer_array.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiHLSToBGR_8u_C3P3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : System.Address;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:2626
   pragma Import (C, nppiHLSToBGR_8u_C3P3R, "nppiHLSToBGR_8u_C3P3R");

  --*
  -- * 4 channel 8-bit unsigned packed HLS with alpha to 4 channel 8-bit unsigned planar BGR with alpha color conversion.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_planar_image_pointer_array.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiHLSToBGR_8u_AC4P4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : System.Address;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:2638
   pragma Import (C, nppiHLSToBGR_8u_AC4P4R, "nppiHLSToBGR_8u_AC4P4R");

  --*
  -- * 3 channel 8-bit unsigned planar HLS to 3 channel 8-bit unsigned planar BGR color conversion.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_planar_image_pointer_array.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiHLSToBGR_8u_P3R
     (pSrc : System.Address;
      nSrcStep : int;
      pDst : System.Address;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:2650
   pragma Import (C, nppiHLSToBGR_8u_P3R, "nppiHLSToBGR_8u_P3R");

  --*
  -- * 4 channel 8-bit unsigned planar HLS with alpha to 4 channel 8-bit unsigned planar BGR with alpha color conversion.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_planar_image_pointer_array.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiHLSToBGR_8u_AP4R
     (pSrc : System.Address;
      nSrcStep : int;
      pDst : System.Address;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:2662
   pragma Import (C, nppiHLSToBGR_8u_AP4R, "nppiHLSToBGR_8u_AP4R");

  --*
  -- * 4 channel 8-bit unsigned packed HLS with alpha to 4 channel 8-bit unsigned packed BGR with alpha color conversion.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiHLSToBGR_8u_AC4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:2674
   pragma Import (C, nppiHLSToBGR_8u_AC4R, "nppiHLSToBGR_8u_AC4R");

  --*
  -- * 3 channel 8-bit unsigned planar HLS to 3 channel 8-bit unsigned packed BGR color conversion.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiHLSToBGR_8u_P3C3R
     (pSrc : System.Address;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:2686
   pragma Import (C, nppiHLSToBGR_8u_P3C3R, "nppiHLSToBGR_8u_P3C3R");

  --*
  -- * 4 channel 8-bit unsigned planar HLS with alpha to 4 channel 8-bit unsigned packed BGR with alpha color conversion.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiHLSToBGR_8u_AP4C4R
     (pSrc : System.Address;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:2698
   pragma Import (C, nppiHLSToBGR_8u_AP4C4R, "nppiHLSToBGR_8u_AP4C4R");

  --* @}  
  --* @name RGBToHSV 
  -- *  RGB to HSV color conversion.
  -- *
  -- *  This is how NPP converts gamma corrected RGB or BGR to HSV.
  -- *  This code uses the fmaxf() and fminf() 32 bit floating point math functions.
  -- *
  -- * \code
  -- *  Npp32f nNormalizedR = (Npp32f)R * 0.003921569F; // / 255.0F
  -- *  Npp32f nNormalizedG = (Npp32f)G * 0.003921569F;
  -- *  Npp32f nNormalizedB = (Npp32f)B * 0.003921569F;
  -- *  Npp32f nS;
  -- *  Npp32f nH;
  -- *  // Value
  -- *  Npp32f nV = fmaxf(nNormalizedR, nNormalizedG);
  -- *         nV = fmaxf(nV, nNormalizedB);
  -- *  // Saturation
  -- *  Npp32f nTemp = fminf(nNormalizedR, nNormalizedG);
  -- *         nTemp = fminf(nTemp, nNormalizedB);
  -- *  Npp32f nDivisor = nV - nTemp;
  -- *  if (nV == 0.0F) // achromatics case
  -- *  {
  -- *      nS = 0.0F;
  -- *      nH = 0.0F;
  -- *  }    
  -- *  else // chromatics case
  -- *      nS = nDivisor / nV;
  -- *  // Hue:
  -- *  Npp32f nCr = (nV - nNormalizedR) / nDivisor;
  -- *  Npp32f nCg = (nV - nNormalizedG) / nDivisor;
  -- *  Npp32f nCb = (nV - nNormalizedB) / nDivisor;
  -- *  if (nNormalizedR == nV)
  -- *      nH = nCb - nCg;
  -- *  else if (nNormalizedG == nV)
  -- *      nH = 2.0F + nCr - nCb;
  -- *  else if (nNormalizedB == nV)
  -- *      nH = 4.0F + nCg - nCr;
  -- *  nH = nH * 0.166667F; // / 6.0F       
  -- *  if (nH < 0.0F)
  -- *      nH = nH + 1.0F;
  -- *  H = (Npp8u)(nH * 255.0F);
  -- *  S = (Npp8u)(nS * 255.0F);
  -- *  V = (Npp8u)(nV * 255.0F);
  -- * \endcode
  -- *
  -- * @{
  --  

  --*
  -- * 3 channel 8-bit unsigned packed RGB to 3 channel 8-bit unsigned packed HSV color conversion.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiRGBToHSV_8u_C3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:2758
   pragma Import (C, nppiRGBToHSV_8u_C3R, "nppiRGBToHSV_8u_C3R");

  --*
  -- * 4 channel 8-bit unsigned packed RGB with alpha to 4 channel 8-bit unsigned packed HSV with alpha color conversion.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiRGBToHSV_8u_AC4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:2770
   pragma Import (C, nppiRGBToHSV_8u_AC4R, "nppiRGBToHSV_8u_AC4R");

  --* @}  
  --* @name HSVToRGB 
  -- *  HSV to RGB color conversion.
  -- *
  -- *  This is how NPP converts HSV to gamma corrected RGB or BGR.
  -- *  This code uses the floorf() 32 bit floating point math function.
  -- *
  -- *  \code
  -- *  Npp32f nNormalizedH = (Npp32f)H * 0.003921569F; // / 255.0F
  -- *  Npp32f nNormalizedS = (Npp32f)S * 0.003921569F;
  -- *  Npp32f nNormalizedV = (Npp32f)V * 0.003921569F;
  -- *  Npp32f nR;
  -- *  Npp32f nG;
  -- *  Npp32f nB;
  -- *  if (nNormalizedS == 0.0F)
  -- *  {
  -- *      nR = nG = nB = nNormalizedV;
  -- *  }
  -- *  else
  -- *  {
  -- *      if (nNormalizedH == 1.0F)
  -- *          nNormalizedH = 0.0F;
  -- *      else
  -- *          nNormalizedH = nNormalizedH * 6.0F; // / 0.1667F
  -- *  }
  -- *  Npp32f nI = floorf(nNormalizedH);
  -- *  Npp32f nF = nNormalizedH - nI;
  -- *  Npp32f nM = nNormalizedV * (1.0F - nNormalizedS);
  -- *  Npp32f nN = nNormalizedV * (1.0F - nNormalizedS * nF);
  -- *  Npp32f nK = nNormalizedV * (1.0F - nNormalizedS * (1.0F - nF));
  -- *  if (nI == 0.0F)
  -- *      { nR = nNormalizedV; nG = nK; nB = nM; }
  -- *  else if (nI == 1.0F)
  -- *      { nR = nN; nG = nNormalizedV; nB = nM; }
  -- *  else if (nI == 2.0F)
  -- *      { nR = nM; nG = nNormalizedV; nB = nK; }
  -- *  else if (nI == 3.0F)
  -- *      { nR = nM; nG = nN; nB = nNormalizedV; }
  -- *  else if (nI == 4.0F)
  -- *      { nR = nK; nG = nM; nB = nNormalizedV; }
  -- *  else if (nI == 5.0F)
  -- *      { nR = nNormalizedV; nG = nM; nB = nN; }
  -- *  R = (Npp8u)(nR * 255.0F);
  -- *  G = (Npp8u)(nG * 255.0F);
  -- *  B = (Npp8u)(nB * 255.0F);
  -- *  \endcode
  -- *
  -- * @{
  --  

  --*
  -- * 3 channel 8-bit unsigned packed HSV to 3 channel 8-bit unsigned packed RGB color conversion.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiHSVToRGB_8u_C3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:2832
   pragma Import (C, nppiHSVToRGB_8u_C3R, "nppiHSVToRGB_8u_C3R");

  --*
  -- * 4 channel 8-bit unsigned packed HSV with alpha to 4 channel 8-bit unsigned packed RGB with alpha color conversion.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiHSVToRGB_8u_AC4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:2844
   pragma Import (C, nppiHSVToRGB_8u_AC4R, "nppiHSVToRGB_8u_AC4R");

  --* @}  
  --* @name RGBToGray 
  -- *  RGB to CCIR601 Gray conversion.
  -- *
  -- *  Here is how NPP converts gamma corrected RGB to CCIR601 Gray.
  -- *
  -- *  \code   
  -- *   nGray =  0.299F * R + 0.587F * G + 0.114F * B; 
  -- *  \endcode
  -- *
  -- * @{
  -- *
  --  

  --*
  -- * 3 channel 8-bit unsigned packed RGB to 1 channel 8-bit unsigned packed Gray conversion.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiRGBToGray_8u_C3C1R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:2871
   pragma Import (C, nppiRGBToGray_8u_C3C1R, "nppiRGBToGray_8u_C3C1R");

  --*
  -- * 4 channel 8-bit unsigned packed RGB with alpha to 1 channel 8-bit unsigned packed Gray conversion.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiRGBToGray_8u_AC4C1R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:2883
   pragma Import (C, nppiRGBToGray_8u_AC4C1R, "nppiRGBToGray_8u_AC4C1R");

  --*
  -- * 3 channel 16-bit unsigned packed RGB to 1 channel 16-bit unsigned packed Gray conversion.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiRGBToGray_16u_C3C1R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:2895
   pragma Import (C, nppiRGBToGray_16u_C3C1R, "nppiRGBToGray_16u_C3C1R");

  --*
  -- * 4 channel 16-bit unsigned packed RGB with alpha to 1 channel 16-bit unsigned packed Gray conversion.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiRGBToGray_16u_AC4C1R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:2907
   pragma Import (C, nppiRGBToGray_16u_AC4C1R, "nppiRGBToGray_16u_AC4C1R");

  --*
  -- * 3 channel 16-bit signed packed RGB to 1 channel 16-bit signed packed Gray conversion.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiRGBToGray_16s_C3C1R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:2919
   pragma Import (C, nppiRGBToGray_16s_C3C1R, "nppiRGBToGray_16s_C3C1R");

  --*
  -- * 4 channel 16-bit signed packed RGB with alpha to 1 channel 16-bit signed packed Gray conversion.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiRGBToGray_16s_AC4C1R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:2931
   pragma Import (C, nppiRGBToGray_16s_AC4C1R, "nppiRGBToGray_16s_AC4C1R");

  --*
  -- * 3 channel 32-bit floating point packed RGB to 1 channel 32-bit floating point packed Gray conversion.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiRGBToGray_32f_C3C1R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:2943
   pragma Import (C, nppiRGBToGray_32f_C3C1R, "nppiRGBToGray_32f_C3C1R");

  --*
  -- * 4 channel 32-bit floating point packed RGB with alpha to 1 channel 32-bit floating point packed Gray conversion.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiRGBToGray_32f_AC4C1R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:2955
   pragma Import (C, nppiRGBToGray_32f_AC4C1R, "nppiRGBToGray_32f_AC4C1R");

  --* @}  
  --* @name ColorToGray 
  -- *  RGB Color to Gray conversion using user supplied conversion coefficients.
  -- *
  -- *  Here is how NPP converts gamma corrected RGB Color to Gray using user supplied conversion coefficients.
  -- *
  -- *  \code   
  -- *   nGray =  aCoeffs[0] * R + aCoeffs[1] * G + aCoeffs[2] * B;
  -- *  \endcode
  -- *
  -- *  For the C4C1R versions of the functions the calculations are as follows.  
  -- *  For BGRA or other formats with alpha just rearrange the coefficients accordingly.
  -- *
  -- *  \code   
  -- *   nGray =  aCoeffs[0] * R + aCoeffs[1] * G + aCoeffs[2] * B + aCoeffs[3] * A;
  -- *  \endcode
  -- *
  -- *
  -- * @{
  -- *
  --  

  --*
  -- * 3 channel 8-bit unsigned packed RGB to 1 channel 8-bit unsigned packed Gray conversion.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aCoeffs fixed size array of constant floating point conversion coefficient values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiColorToGray_8u_C3C1R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aCoeffs : access nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:2991
   pragma Import (C, nppiColorToGray_8u_C3C1R, "nppiColorToGray_8u_C3C1R");

  --*
  -- * 4 channel 8-bit unsigned packed RGB with alpha to 1 channel 8-bit unsigned packed Gray conversion.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aCoeffs fixed size array of constant floating point conversion coefficient values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiColorToGray_8u_AC4C1R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aCoeffs : access nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:3004
   pragma Import (C, nppiColorToGray_8u_AC4C1R, "nppiColorToGray_8u_AC4C1R");

  --*
  -- * 4 channel 8-bit unsigned packed RGBA to 1 channel 8-bit unsigned packed Gray conversion.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aCoeffs fixed size array of constant floating point conversion coefficient values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiColorToGray_8u_C4C1R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aCoeffs : access nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:3017
   pragma Import (C, nppiColorToGray_8u_C4C1R, "nppiColorToGray_8u_C4C1R");

  --*
  -- * 3 channel 16-bit unsigned packed RGB to 1 channel 16-bit unsigned packed Gray conversion.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aCoeffs fixed size array of constant floating point conversion coefficient values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiColorToGray_16u_C3C1R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aCoeffs : access nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:3030
   pragma Import (C, nppiColorToGray_16u_C3C1R, "nppiColorToGray_16u_C3C1R");

  --*
  -- * 4 channel 16-bit unsigned packed RGB with alpha to 1 channel 16-bit unsigned packed Gray conversion.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aCoeffs fixed size array of constant floating point conversion coefficient values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiColorToGray_16u_AC4C1R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aCoeffs : access nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:3043
   pragma Import (C, nppiColorToGray_16u_AC4C1R, "nppiColorToGray_16u_AC4C1R");

  --*
  -- * 4 channel 16-bit unsigned packed RGBA to 1 channel 16-bit unsigned packed Gray conversion.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aCoeffs fixed size array of constant floating point conversion coefficient values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiColorToGray_16u_C4C1R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aCoeffs : access nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:3056
   pragma Import (C, nppiColorToGray_16u_C4C1R, "nppiColorToGray_16u_C4C1R");

  --*
  -- * 3 channel 16-bit signed packed RGB to 1 channel 16-bit signed packed Gray conversion.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aCoeffs fixed size array of constant floating point conversion coefficient values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiColorToGray_16s_C3C1R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aCoeffs : access nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:3069
   pragma Import (C, nppiColorToGray_16s_C3C1R, "nppiColorToGray_16s_C3C1R");

  --*
  -- * 4 channel 16-bit signed packed RGB with alpha to 1 channel 16-bit signed packed Gray conversion.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aCoeffs fixed size array of constant floating point conversion coefficient values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiColorToGray_16s_AC4C1R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aCoeffs : access nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:3082
   pragma Import (C, nppiColorToGray_16s_AC4C1R, "nppiColorToGray_16s_AC4C1R");

  --*
  -- * 4 channel 16-bit signed packed RGBA to 1 channel 16-bit signed packed Gray conversion.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aCoeffs fixed size array of constant floating point conversion coefficient values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiColorToGray_16s_C4C1R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aCoeffs : access nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:3095
   pragma Import (C, nppiColorToGray_16s_C4C1R, "nppiColorToGray_16s_C4C1R");

  --*
  -- * 3 channel 32-bit floating point packed RGB to 1 channel 32-bit floating point packed Gray conversion.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aCoeffs fixed size array of constant floating point conversion coefficient values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiColorToGray_32f_C3C1R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aCoeffs : access nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:3108
   pragma Import (C, nppiColorToGray_32f_C3C1R, "nppiColorToGray_32f_C3C1R");

  --*
  -- * 4 channel 32-bit floating point packed RGB with alpha to 1 channel 32-bit floating point packed Gray conversion.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aCoeffs fixed size array of constant floating point conversion coefficient values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiColorToGray_32f_AC4C1R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aCoeffs : access nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:3121
   pragma Import (C, nppiColorToGray_32f_AC4C1R, "nppiColorToGray_32f_AC4C1R");

  --*
  -- * 4 channel 32-bit floating point packed RGBA to 1 channel 32-bit floating point packed Gray conversion.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aCoeffs fixed size array of constant floating point conversion coefficient values, one per color channel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiColorToGray_32f_C4C1R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aCoeffs : access nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:3134
   pragma Import (C, nppiColorToGray_32f_C4C1R, "nppiColorToGray_32f_C4C1R");

  --* @}  
  --* @name GradientColorToGray 
  -- *  RGB Color to Gray Gradient conversion using user selected gradient distance method.
  -- *
  -- * @{
  -- *
  --  

  --*
  -- * 3 channel 8-bit unsigned packed RGB to 1 channel 8-bit unsigned packed Gray Gradient conversion.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eNorm Gradient distance method to use.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiGradientColorToGray_8u_C3C1R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eNorm : nppdefs_h.NppiNorm) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:3156
   pragma Import (C, nppiGradientColorToGray_8u_C3C1R, "nppiGradientColorToGray_8u_C3C1R");

  --*
  -- * 3 channel 16-bit unsigned packed RGB to 1 channel 16-bit unsigned packed Gray Gradient conversion.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eNorm Gradient distance method to use.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiGradientColorToGray_16u_C3C1R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eNorm : nppdefs_h.NppiNorm) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:3169
   pragma Import (C, nppiGradientColorToGray_16u_C3C1R, "nppiGradientColorToGray_16u_C3C1R");

  --*
  -- * 3 channel 16-bit signed packed RGB to 1 channel 16-bit signed packed Gray Gradient conversion.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eNorm Gradient distance method to use.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiGradientColorToGray_16s_C3C1R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eNorm : nppdefs_h.NppiNorm) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:3182
   pragma Import (C, nppiGradientColorToGray_16s_C3C1R, "nppiGradientColorToGray_16s_C3C1R");

  --*
  -- * 3 channel 32-bit floating point packed RGB to 1 channel 32-bit floating point packed Gray Gradient conversion.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eNorm Gradient distance method to use.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiGradientColorToGray_32f_C3C1R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eNorm : nppdefs_h.NppiNorm) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:3195
   pragma Import (C, nppiGradientColorToGray_32f_C3C1R, "nppiGradientColorToGray_32f_C3C1R");

  --* @}  
  --* @name ColorDebayer 
  -- *  Grayscale Color Filter Array to RGB Color Debayer conversion. Generates one RGB color pixel for every grayscale source pixel.
  -- *  Source and destination images must have even width and height.  Missing pixel colors are generated using bilinear interpolation
  -- *  with chroma correlation of generated green values (eInterpolation MUST be set to 0). eGrid allows the user to specify the Bayer grid 
  -- *  registration position at source image location oSrcROI.x, oSrcROI.y relative to pSrc. Possible registration positions are:
  -- *
  -- *  \code
  -- *  NPPI_BAYER_BGGR  NPPI_BAYER_RGGB  NPPI_BAYER_GBRG  NPPI_BAYER_GRBG
  -- *
  -- *        B G              R G              G B              G R
  -- *        G R              G B              R G              B G
  -- *
  -- *  \endcode
  -- *
  -- *  If it becomes necessary to access source pixels outside source image then the source image borders are mirrored.
  -- *
  -- *  Here is how the algorithm works.  R, G, and B base pixels from the source image are used unmodified.  To generate R values for those
  -- *  G pixels, the average of R(x - 1, y) and R(x + 1, y) or R(x, y - 1) and R(x, y + 1) is used depending on whether the left and right
  -- *  or top and bottom pixels are R base pixels.  To generate B values for those G pixels, the same algorithm is used using nearest B values.
  -- *  For an R base pixel, if there are no B values in the upper, lower, left, or right adjacent pixels then B is the average of B values
  -- *  in the 4 diagonal (G base) pixels.  The same algorithm is used using R values to generate the R value of a B base pixel. 
  -- *  Chroma correlation is applied to generated G values only, for a B base pixel G(x - 1, y) and G(x + 1, y) are averaged or G(x, y - 1)
  -- *  and G(x, y + 1) are averaged depending on whether the absolute difference between B(x, y) and the average of B(x - 2, y) and B(x + 2, y)
  -- *  is smaller than the absolute difference between B(x, y) and the average of B(x, y - 2) and B(x, y + 2). For an R base pixel the same
  -- *  algorithm is used testing against the surrounding R values at those offsets.  If the horizontal and vertical differences are the same
  -- *  at one of those pixels then the average of the four left, right, upper and lower G values is used instead.
  -- *  
  -- *
  -- * @{
  -- *
  --  

  --*
  -- * 1 channel 8-bit unsigned packed CFA grayscale Bayer pattern to 3 channel 8-bit unsigned packed RGB conversion.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize full source image width and height relative to pSrc.
  -- * \param oSrcROI rectangle specifying starting source image pixel x and y location relative to pSrc and ROI width and height. 
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param eGrid enumeration value specifying bayer grid registration position at location oSrcROI.x, oSrcROI.y relative to pSrc.
  -- * \param eInterpolation MUST be NPPI_INTER_UNDEFINED
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCFAToRGB_8u_C1C3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      oSrcSize : nppdefs_h.NppiSize;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      eGrid : nppdefs_h.NppiBayerGridPosition;
      eInterpolation : nppdefs_h.NppiInterpolationMode) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:3245
   pragma Import (C, nppiCFAToRGB_8u_C1C3R, "nppiCFAToRGB_8u_C1C3R");

  --*
  -- * 1 channel 8-bit unsigned packed CFA grayscale Bayer pattern to 4 channel 8-bit unsigned packed RGB conversion with alpha.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize full source image width and height relative to pSrc.
  -- * \param oSrcROI rectangle specifying starting source image pixel x and y location relative to pSrc and ROI width and height. 
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param eGrid enumeration value specifying bayer grid registration position at location oSrcROI.x, oSrcROI.y relative to pSrc.
  -- * \param eInterpolation MUST be NPPI_INTER_UNDEFINED
  -- * \param nAlpha constant alpha value to be written to each destination pixel
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCFAToRGBA_8u_C1AC4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      oSrcSize : nppdefs_h.NppiSize;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      eGrid : nppdefs_h.NppiBayerGridPosition;
      eInterpolation : nppdefs_h.NppiInterpolationMode;
      nAlpha : nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:3263
   pragma Import (C, nppiCFAToRGBA_8u_C1AC4R, "nppiCFAToRGBA_8u_C1AC4R");

  --*
  -- * 1 channel 16-bit unsigned packed CFA grayscale Bayer pattern to 3 channel 16-bit unsigned packed RGB conversion.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize full source image width and height relative to pSrc.
  -- * \param oSrcROI rectangle specifying starting source image pixel x and y location relative to pSrc and ROI width and height. 
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param eGrid enumeration value specifying bayer grid registration position at location oSrcROI.x, oSrcROI.y relative to pSrc.
  -- * \param eInterpolation MUST be NPPI_INTER_UNDEFINED
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCFAToRGB_16u_C1C3R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      oSrcSize : nppdefs_h.NppiSize;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      eGrid : nppdefs_h.NppiBayerGridPosition;
      eInterpolation : nppdefs_h.NppiInterpolationMode) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:3280
   pragma Import (C, nppiCFAToRGB_16u_C1C3R, "nppiCFAToRGB_16u_C1C3R");

  --*
  -- * 1 channel 16-bit unsigned packed CFA grayscale Bayer pattern to 4 channel 16-bit unsigned packed RGB conversion with alpha.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize full source image width and height relative to pSrc.
  -- * \param oSrcROI rectangle specifying starting source image pixel x and y location relative to pSrc and ROI width and height. 
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param eGrid enumeration value specifying bayer grid registration position at location oSrcROI.x, oSrcROI.y relative to pSrc.
  -- * \param eInterpolation MUST be NPPI_INTER_UNDEFINED
  -- * \param nAlpha constant alpha value to be written to each destination pixel
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCFAToRGBA_16u_C1AC4R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      oSrcSize : nppdefs_h.NppiSize;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      eGrid : nppdefs_h.NppiBayerGridPosition;
      eInterpolation : nppdefs_h.NppiInterpolationMode;
      nAlpha : nppdefs_h.Npp16u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:3298
   pragma Import (C, nppiCFAToRGBA_16u_C1AC4R, "nppiCFAToRGBA_16u_C1AC4R");

  --* @}  
  --* @} image_color_model_conversion  
  --* @defgroup image_color_sampling_format_conversion Color Sampling Format Conversion
  -- *
  -- * Routines for converting between various image color sampling formats.
  -- *
  -- *
  -- * @{                                         
  --  

  --* @name YCbCr420ToYCbCr411 
  -- *  YCbCr420 to YCbCr411 sampling format conversion.
  -- * @{
  --  

  --*
  -- * 3 channel 8-bit unsigned planar YCbCr420 to 2 channel 8-bit unsigned planar YCbCr411 sampling format conversion.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array.
  -- * \param rSrcStep \ref source_planar_image_line_step_array.
  -- * \param pDstY \ref destination_planar_image_pointer.
  -- * \param nDstYStep \ref destination_planar_image_line_step.
  -- * \param pDstCbCr \ref destination_planar_image_pointer.
  -- * \param nDstCbCrStep \ref destination_planar_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYCbCr420ToYCbCr411_8u_P3P2R
     (pSrc : System.Address;
      rSrcStep : access int;
      pDstY : access nppdefs_h.Npp8u;
      nDstYStep : int;
      pDstCbCr : access nppdefs_h.Npp8u;
      nDstCbCrStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:3329
   pragma Import (C, nppiYCbCr420ToYCbCr411_8u_P3P2R, "nppiYCbCr420ToYCbCr411_8u_P3P2R");

  --*
  -- * 2 channel 8-bit unsigned planar YCbCr420 to 3 channel 8-bit unsigned planar YCbCr411 sampling format conversion.
  -- *
  -- * \param pSrcY \ref source_planar_image_pointer.
  -- * \param nSrcYStep \ref source_planar_image_line_step.
  -- * \param pSrcCbCr \ref source_planar_image_pointer.
  -- * \param nSrcCbCrStep \ref source_planar_image_line_step.
  -- * \param pDst \ref destination_planar_image_pointer_array.
  -- * \param rDstStep \ref destination_planar_image_line_step_array.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYCbCr420ToYCbCr411_8u_P2P3R
     (pSrcY : access nppdefs_h.Npp8u;
      nSrcYStep : int;
      pSrcCbCr : access nppdefs_h.Npp8u;
      nSrcCbCrStep : int;
      pDst : System.Address;
      rDstStep : access int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:3343
   pragma Import (C, nppiYCbCr420ToYCbCr411_8u_P2P3R, "nppiYCbCr420ToYCbCr411_8u_P2P3R");

  --* @}  
  --* @name YCbCr422ToYCbCr422 
  -- *  YCbCr422 to YCbCr422 sampling format conversion.
  -- * @{
  --  

  --*
  -- * 2 channel 8-bit unsigned packed YCbCr422 to 3 channel 8-bit unsigned planar YCbCr422 sampling format conversion.
  -- * images.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_planar_image_pointer_array.
  -- * \param rDstStep \ref destination_planar_image_line_step_array.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYCbCr422_8u_C2P3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : System.Address;
      rDstStep : access int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:3362
   pragma Import (C, nppiYCbCr422_8u_C2P3R, "nppiYCbCr422_8u_C2P3R");

  --*
  -- * 3 channel 8-bit unsigned planar YCbCr422 to 2 channel 8-bit unsigned packed YCbCr422 sampling format conversion.
  -- * images.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array.
  -- * \param rSrcStep \ref source_planar_image_line_step_array.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYCbCr422_8u_P3C2R
     (pSrc : System.Address;
      rSrcStep : access int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:3375
   pragma Import (C, nppiYCbCr422_8u_P3C2R, "nppiYCbCr422_8u_P3C2R");

  --* @}  
  --* @name YCbCr422ToYCrCb422 
  -- *  YCbCr422 to YCrCb422 sampling format conversion.
  -- * @{
  --  

  --*
  -- * 2 channel 8-bit unsigned packed YCbCr422 to 2 channel 8-bit unsigned packed YCrCb422 sampling format conversion.
  -- * images.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYCbCr422ToYCrCb422_8u_C2R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:3394
   pragma Import (C, nppiYCbCr422ToYCrCb422_8u_C2R, "nppiYCbCr422ToYCrCb422_8u_C2R");

  --*
  -- * 3 channel 8-bit unsigned planar YCbCr422 to 2 channel 8-bit unsigned packed YCrCb422 sampling format conversion.
  -- * images.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array.
  -- * \param rSrcStep \ref source_planar_image_line_step_array.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYCbCr422ToYCrCb422_8u_P3C2R
     (pSrc : System.Address;
      rSrcStep : access int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:3407
   pragma Import (C, nppiYCbCr422ToYCrCb422_8u_P3C2R, "nppiYCbCr422ToYCrCb422_8u_P3C2R");

  --* @}  
  --* @name YCbCr422ToCbYCr422 
  -- *  YCbCr422 to CbYCr422 sampling format conversion.
  -- * @{
  --  

  --*
  -- * 2 channel 8-bit unsigned packed YCbCr422 to 2 channel 8-bit unsigned packed CbYCr422 sampling format conversion.
  -- * images.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYCbCr422ToCbYCr422_8u_C2R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:3426
   pragma Import (C, nppiYCbCr422ToCbYCr422_8u_C2R, "nppiYCbCr422ToCbYCr422_8u_C2R");

  --* @}  
  --* @name CbYCr422ToYCbCr411 
  -- *  CbYCr422 to YCbCr411 sampling format conversion.
  -- * @{
  --  

  --*
  -- * 2 channel 8-bit unsigned packed CbYCr422 to 3 channel 8-bit unsigned planar YCbCr411 sampling format conversion.
  -- * images.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_planar_image_pointer_array.
  -- * \param rDstStep \ref destination_planar_image_line_step_array.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCbYCr422ToYCbCr411_8u_C2P3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : System.Address;
      rDstStep : access int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:3445
   pragma Import (C, nppiCbYCr422ToYCbCr411_8u_C2P3R, "nppiCbYCr422ToYCbCr411_8u_C2P3R");

  --* @}  
  --* @name YCbCr422ToYCbCr420 
  -- *  YCbCr422 to YCbCr420 sampling format conversion.
  -- * @{
  --  

  --*
  -- * 3 channel 8-bit unsigned planar YCbCr422 to 3 channel 8-bit unsigned planar YCbCr420 sampling format conversion.
  -- * images.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array.
  -- * \param rSrcStep \ref source_planar_image_line_step_array.
  -- * \param pDst \ref destination_planar_image_pointer_array.
  -- * \param nDstStep \ref destination_planar_image_line_step_array.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYCbCr422ToYCbCr420_8u_P3R
     (pSrc : System.Address;
      rSrcStep : access int;
      pDst : System.Address;
      nDstStep : access int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:3464
   pragma Import (C, nppiYCbCr422ToYCbCr420_8u_P3R, "nppiYCbCr422ToYCbCr420_8u_P3R");

  --*
  -- * 3 channel 8-bit unsigned planar YCbCr422 to 2 channel 8-bit unsigned planar YCbCr420 sampling format conversion.
  -- * images.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array.
  -- * \param rSrcStep \ref source_planar_image_line_step_array.
  -- * \param pDstY \ref destination_planar_image_pointer.
  -- * \param nDstYStep \ref destination_planar_image_line_step.
  -- * \param pDstCbCr \ref destination_planar_image_pointer.
  -- * \param nDstCbCrStep \ref destination_planar_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYCbCr422ToYCbCr420_8u_P3P2R
     (pSrc : System.Address;
      rSrcStep : access int;
      pDstY : access nppdefs_h.Npp8u;
      nDstYStep : int;
      pDstCbCr : access nppdefs_h.Npp8u;
      nDstCbCrStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:3479
   pragma Import (C, nppiYCbCr422ToYCbCr420_8u_P3P2R, "nppiYCbCr422ToYCbCr420_8u_P3P2R");

  --*
  -- * 2 channel 8-bit unsigned packed YCbCr422 to 3 channel 8-bit unsigned planar YCbCr420 sampling format conversion.
  -- * images.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_planar_image_pointer_array.
  -- * \param rDstStep \ref destination_planar_image_line_step_array.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYCbCr422ToYCbCr420_8u_C2P3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : System.Address;
      rDstStep : access int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:3492
   pragma Import (C, nppiYCbCr422ToYCbCr420_8u_C2P3R, "nppiYCbCr422ToYCbCr420_8u_C2P3R");

  --*
  -- * 2 channel 8-bit unsigned packed YCbCr422 to 2 channel 8-bit unsigned planar YCbCr420 sampling format conversion.
  -- * images.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDstY \ref destination_planar_image_pointer.
  -- * \param nDstYStep \ref destination_planar_image_line_step.
  -- * \param pDstCbCr \ref destination_planar_image_pointer.
  -- * \param nDstCbCrStep \ref destination_planar_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYCbCr422ToYCbCr420_8u_C2P2R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDstY : access nppdefs_h.Npp8u;
      nDstYStep : int;
      pDstCbCr : access nppdefs_h.Npp8u;
      nDstCbCrStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:3507
   pragma Import (C, nppiYCbCr422ToYCbCr420_8u_C2P2R, "nppiYCbCr422ToYCbCr420_8u_C2P2R");

  --* @}  
  --* @name YCrCb420ToYCbCr422 
  -- *  YCrCb420 to YCbCr422 sampling format conversion.
  -- * @{
  --  

  --*
  -- * 3 channel 8-bit unsigned planar YCrCb420 to 3 channel 8-bit unsigned planar YCbCr422 sampling format conversion.
  -- * images.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array.
  -- * \param rSrcStep \ref source_planar_image_line_step_array.
  -- * \param pDst \ref destination_planar_image_pointer_array.
  -- * \param rDstStep \ref destination_planar_image_line_step_array.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYCrCb420ToYCbCr422_8u_P3R
     (pSrc : System.Address;
      rSrcStep : access int;
      pDst : System.Address;
      rDstStep : access int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:3526
   pragma Import (C, nppiYCrCb420ToYCbCr422_8u_P3R, "nppiYCrCb420ToYCbCr422_8u_P3R");

  --*
  -- * 3 channel 8-bit unsigned planar YCrCb420 to 2 channel 8-bit unsigned packed YCbCr422 sampling format conversion.
  -- * images.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array.
  -- * \param rSrcStep \ref source_planar_image_line_step_array.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYCrCb420ToYCbCr422_8u_P3C2R
     (pSrc : System.Address;
      rSrcStep : access int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:3539
   pragma Import (C, nppiYCrCb420ToYCbCr422_8u_P3C2R, "nppiYCrCb420ToYCbCr422_8u_P3C2R");

  --* @}  
  --* @name YCbCr422ToYCrCb420 
  -- *  YCbCr422 to YCrCb420 sampling format conversion.
  -- * @{
  --  

  --*
  -- * 2 channel 8-bit unsigned packed YCbCr422 to 3 channel 8-bit unsigned planar YCrCb420 sampling format conversion.
  -- * images.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_planar_image_pointer_array.
  -- * \param rDstStep \ref destination_planar_image_line_step_array.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYCbCr422ToYCrCb420_8u_C2P3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : System.Address;
      rDstStep : access int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:3558
   pragma Import (C, nppiYCbCr422ToYCrCb420_8u_C2P3R, "nppiYCbCr422ToYCrCb420_8u_C2P3R");

  --* @}  
  --* @name YCbCr422ToYCbCr411 
  -- *  YCbCr422 to YCbCr411 sampling format conversion.
  -- * @{
  --  

  --*
  -- * 3 channel 8-bit unsigned planar YCbCr422 to 3 channel 8-bit unsigned planar YCbCr411 sampling format conversion.
  -- * images.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array.
  -- * \param rSrcStep \ref source_planar_image_line_step_array.
  -- * \param pDst \ref destination_planar_image_pointer_array.
  -- * \param rDstStep \ref destination_planar_image_line_step_array.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYCbCr422ToYCbCr411_8u_P3R
     (pSrc : System.Address;
      rSrcStep : access int;
      pDst : System.Address;
      rDstStep : access int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:3577
   pragma Import (C, nppiYCbCr422ToYCbCr411_8u_P3R, "nppiYCbCr422ToYCbCr411_8u_P3R");

  --*
  -- * 3 channel 8-bit unsigned planar YCbCr422 to 2 channel 8-bit unsigned planar YCbCr411 sampling format conversion.
  -- * images.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array.
  -- * \param rSrcStep \ref source_planar_image_line_step_array.
  -- * \param pDstY \ref destination_planar_image_pointer.
  -- * \param nDstYStep \ref destination_planar_image_line_step.
  -- * \param pDstCbCr \ref destination_planar_image_pointer.
  -- * \param nDstCbCrStep \ref destination_planar_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYCbCr422ToYCbCr411_8u_P3P2R
     (pSrc : System.Address;
      rSrcStep : access int;
      pDstY : access nppdefs_h.Npp8u;
      nDstYStep : int;
      pDstCbCr : access nppdefs_h.Npp8u;
      nDstCbCrStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:3592
   pragma Import (C, nppiYCbCr422ToYCbCr411_8u_P3P2R, "nppiYCbCr422ToYCbCr411_8u_P3P2R");

  --*
  -- * 2 channel 8-bit unsigned packed YCbCr422 to 3 channel 8-bit unsigned planar YCbCr411 sampling format conversion.
  -- * images.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_planar_image_pointer_array.
  -- * \param rDstStep \ref destination_planar_image_line_step_array.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYCbCr422ToYCbCr411_8u_C2P3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : System.Address;
      rDstStep : access int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:3605
   pragma Import (C, nppiYCbCr422ToYCbCr411_8u_C2P3R, "nppiYCbCr422ToYCbCr411_8u_C2P3R");

  --*
  -- * 2 channel 8-bit unsigned packed YCbCr422 to 2 channel 8-bit unsigned planar YCbCr411 sampling format conversion.
  -- * images.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDstY \ref destination_planar_image_pointer.
  -- * \param nDstYStep \ref destination_planar_image_line_step.
  -- * \param pDstCbCr \ref destination_planar_image_pointer.
  -- * \param nDstCbCrStep \ref destination_planar_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYCbCr422ToYCbCr411_8u_C2P2R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDstY : access nppdefs_h.Npp8u;
      nDstYStep : int;
      pDstCbCr : access nppdefs_h.Npp8u;
      nDstCbCrStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:3620
   pragma Import (C, nppiYCbCr422ToYCbCr411_8u_C2P2R, "nppiYCbCr422ToYCbCr411_8u_C2P2R");

  --* @}  
  --* @name YCrCb422ToYCbCr422 
  -- *  YCrCb422 to YCbCr422 sampling format conversion.
  -- * @{
  --  

  --*
  -- * 2 channel 8-bit unsigned packed YCrCb422 to 3 channel 8-bit unsigned planar YCbCr422 sampling format conversion.
  -- * images.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_planar_image_pointer_array.
  -- * \param rDstStep \ref destination_planar_image_line_step_array.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYCrCb422ToYCbCr422_8u_C2P3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : System.Address;
      rDstStep : access int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:3639
   pragma Import (C, nppiYCrCb422ToYCbCr422_8u_C2P3R, "nppiYCrCb422ToYCbCr422_8u_C2P3R");

  --* @}  
  --* @name YCrCb422ToYCbCr420 
  -- *  YCrCb422 to YCbCr420 sampling format conversion.
  -- * @{
  --  

  --*
  -- * 2 channel 8-bit unsigned packed YCrCb422 to 3 channel 8-bit unsigned planar YCbCr420 sampling format conversion.
  -- * images.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_planar_image_pointer_array.
  -- * \param rDstStep \ref destination_planar_image_line_step_array.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYCrCb422ToYCbCr420_8u_C2P3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : System.Address;
      rDstStep : access int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:3658
   pragma Import (C, nppiYCrCb422ToYCbCr420_8u_C2P3R, "nppiYCrCb422ToYCbCr420_8u_C2P3R");

  --* @}  
  --* @name YCrCb422ToYCbCr411 
  -- *  YCrCb422 to YCbCr411 sampling format conversion.
  -- * @{
  --  

  --*
  -- * 2 channel 8-bit unsigned packed YCrCb422 to 3 channel 8-bit unsigned planar YCbCr411 sampling format conversion.
  -- * images.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_planar_image_pointer_array.
  -- * \param rDstStep \ref destination_planar_image_line_step_array.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYCrCb422ToYCbCr411_8u_C2P3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : System.Address;
      rDstStep : access int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:3677
   pragma Import (C, nppiYCrCb422ToYCbCr411_8u_C2P3R, "nppiYCrCb422ToYCbCr411_8u_C2P3R");

  --* @}  
  --* @name CbYCr422ToYCbCr422 
  -- *  CbYCr422 to YCbCr422 sampling format conversion.
  -- * @{
  --  

  --*
  -- * 2 channel 8-bit unsigned packed CbYCr422 to 2 channel 8-bit unsigned packed YCbCr422 sampling format conversion.
  -- * images.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCbYCr422ToYCbCr422_8u_C2R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:3696
   pragma Import (C, nppiCbYCr422ToYCbCr422_8u_C2R, "nppiCbYCr422ToYCbCr422_8u_C2R");

  --*
  -- * 2 channel 8-bit unsigned packed CbYCr422 to 3 channel 8-bit unsigned planar YCbCr422 sampling format conversion.
  -- * images.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_planar_image_pointer_array.
  -- * \param rDstStep \ref destination_planar_image_line_step_array.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCbYCr422ToYCbCr422_8u_C2P3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : System.Address;
      rDstStep : access int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:3709
   pragma Import (C, nppiCbYCr422ToYCbCr422_8u_C2P3R, "nppiCbYCr422ToYCbCr422_8u_C2P3R");

  --* @}  
  --* @name CbYCr422ToYCbCr420 
  -- *  CbYCr422 to YCbCr420 sampling format conversion.
  -- * @{
  --  

  --*
  -- * 2 channel 8-bit unsigned packed CbYCr422 to 3 channel 8-bit unsigned planar YCbCr420 sampling format conversion.
  -- * images.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_planar_image_pointer_array.
  -- * \param rDstStep \ref destination_planar_image_line_step_array.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCbYCr422ToYCbCr420_8u_C2P3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : System.Address;
      rDstStep : access int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:3728
   pragma Import (C, nppiCbYCr422ToYCbCr420_8u_C2P3R, "nppiCbYCr422ToYCbCr420_8u_C2P3R");

  --*
  -- * 2 channel 8-bit unsigned packed CbYCr422 to 2 channel 8-bit unsigned planar YCbCr420 sampling format conversion.
  -- * images.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDstY \ref destination_planar_image_pointer.
  -- * \param nDstYStep \ref destination_planar_image_line_step.
  -- * \param pDstCbCr \ref destination_planar_image_pointer.
  -- * \param nDstCbCrStep \ref destination_planar_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCbYCr422ToYCbCr420_8u_C2P2R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDstY : access nppdefs_h.Npp8u;
      nDstYStep : int;
      pDstCbCr : access nppdefs_h.Npp8u;
      nDstCbCrStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:3743
   pragma Import (C, nppiCbYCr422ToYCbCr420_8u_C2P2R, "nppiCbYCr422ToYCbCr420_8u_C2P2R");

  --* @}  
  --* @name CbYCr422ToYCrCb420 
  -- *  CbYCr422 to YCrCb420 sampling format conversion.
  -- * @{
  --  

  --*
  -- * 2 channel 8-bit unsigned packed CbYCr422 to 3 channel 8-bit unsigned planar YCrCb420 sampling format conversion.
  -- * images.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_planar_image_pointer_array.
  -- * \param rDstStep \ref destination_planar_image_line_step_array.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCbYCr422ToYCrCb420_8u_C2P3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : System.Address;
      rDstStep : access int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:3762
   pragma Import (C, nppiCbYCr422ToYCrCb420_8u_C2P3R, "nppiCbYCr422ToYCrCb420_8u_C2P3R");

  --* @}  
  --* @name YCbCr420ToYCbCr420 
  -- *  YCbCr420 to YCbCr420 sampling format conversion.
  -- * @{
  --  

  --*
  -- * 3 channel 8-bit unsigned planar YCbCr420 to 2 channel 8-bit unsigned planar YCbCr420 sampling format conversion.
  -- * images.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array.
  -- * \param rSrcStep \ref source_planar_image_line_step_array.
  -- * \param pDstY \ref destination_planar_image_pointer.
  -- * \param nDstYStep \ref destination_planar_image_line_step.
  -- * \param pDstCbCr \ref destination_planar_image_pointer.
  -- * \param nDstCbCrStep \ref destination_planar_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYCbCr420_8u_P3P2R
     (pSrc : System.Address;
      rSrcStep : access int;
      pDstY : access nppdefs_h.Npp8u;
      nDstYStep : int;
      pDstCbCr : access nppdefs_h.Npp8u;
      nDstCbCrStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:3783
   pragma Import (C, nppiYCbCr420_8u_P3P2R, "nppiYCbCr420_8u_P3P2R");

  --*
  -- * 2 channel 8-bit unsigned planar YCbCr420 to 3 channel 8-bit unsigned planar YCbCr420 sampling format conversion.
  -- *
  -- * \param pSrcY \ref source_planar_image_pointer.
  -- * \param nSrcYStep \ref source_planar_image_line_step.
  -- * \param pSrcCbCr \ref source_planar_image_pointer.
  -- * \param nSrcCbCrStep \ref source_planar_image_line_step.
  -- * \param pDst \ref destination_planar_image_pointer_array.
  -- * \param rDstStep \ref destination_planar_image_line_step_array.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYCbCr420_8u_P2P3R
     (pSrcY : access nppdefs_h.Npp8u;
      nSrcYStep : int;
      pSrcCbCr : access nppdefs_h.Npp8u;
      nSrcCbCrStep : int;
      pDst : System.Address;
      rDstStep : access int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:3797
   pragma Import (C, nppiYCbCr420_8u_P2P3R, "nppiYCbCr420_8u_P2P3R");

  --* @}  
  --* @name YCbCr420ToYCbCr422 
  -- *  YCbCr420 to YCbCr422 sampling format conversion.
  -- * @{
  --  

  --*
  -- * 3 channel 8-bit unsigned planar YCbCr420 to 3 channel 8-bit unsigned planar YCbCr422 sampling format conversion.
  -- * images.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array.
  -- * \param rSrcStep \ref source_planar_image_line_step_array.
  -- * \param pDst \ref destination_planar_image_pointer_array.
  -- * \param nDstStep \ref destination_planar_image_line_step_array.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYCbCr420ToYCbCr422_8u_P3R
     (pSrc : System.Address;
      rSrcStep : access int;
      pDst : System.Address;
      nDstStep : access int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:3816
   pragma Import (C, nppiYCbCr420ToYCbCr422_8u_P3R, "nppiYCbCr420ToYCbCr422_8u_P3R");

  --*
  -- * 2 channel 8-bit unsigned planar YCbCr420 to 3 channel 8-bit unsigned planar YCbCr422 sampling format conversion.
  -- *
  -- * \param pSrcY \ref source_planar_image_pointer.
  -- * \param nSrcYStep \ref source_planar_image_line_step.
  -- * \param pSrcCbCr \ref source_planar_image_pointer.
  -- * \param nSrcCbCrStep \ref source_planar_image_line_step.
  -- * \param pDst \ref destination_planar_image_pointer_array.
  -- * \param rDstStep \ref destination_planar_image_line_step_array.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYCbCr420ToYCbCr422_8u_P2P3R
     (pSrcY : access nppdefs_h.Npp8u;
      nSrcYStep : int;
      pSrcCbCr : access nppdefs_h.Npp8u;
      nSrcCbCrStep : int;
      pDst : System.Address;
      rDstStep : access int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:3830
   pragma Import (C, nppiYCbCr420ToYCbCr422_8u_P2P3R, "nppiYCbCr420ToYCbCr422_8u_P2P3R");

  --*
  -- * 2 channel 8-bit unsigned planar YCbCr420 to 2 channel 8-bit unsigned packed YCbCr422 sampling format conversion.
  -- *
  -- * \param pSrcY \ref source_planar_image_pointer.
  -- * \param nSrcYStep \ref source_planar_image_line_step.
  -- * \param pSrcCbCr \ref source_planar_image_pointer.
  -- * \param nSrcCbCrStep \ref source_planar_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYCbCr420ToYCbCr422_8u_P2C2R
     (pSrcY : access nppdefs_h.Npp8u;
      nSrcYStep : int;
      pSrcCbCr : access nppdefs_h.Npp8u;
      nSrcCbCrStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:3844
   pragma Import (C, nppiYCbCr420ToYCbCr422_8u_P2C2R, "nppiYCbCr420ToYCbCr422_8u_P2C2R");

  --* @}  
  --* @name YCbCr420ToCbYCr422 
  -- *  YCbCr420 to CbYCr422 sampling format conversion.
  -- * @{
  --  

  --*
  -- * 2 channel 8-bit unsigned planar YCbCr420 to 2 channel 8-bit unsigned packed CbYCr422 sampling format conversion.
  -- *
  -- * \param pSrcY \ref source_planar_image_pointer.
  -- * \param nSrcYStep \ref source_planar_image_line_step.
  -- * \param pSrcCbCr \ref source_planar_image_pointer.
  -- * \param nSrcCbCrStep \ref source_planar_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYCbCr420ToCbYCr422_8u_P2C2R
     (pSrcY : access nppdefs_h.Npp8u;
      nSrcYStep : int;
      pSrcCbCr : access nppdefs_h.Npp8u;
      nSrcCbCrStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:3864
   pragma Import (C, nppiYCbCr420ToCbYCr422_8u_P2C2R, "nppiYCbCr420ToCbYCr422_8u_P2C2R");

  --* @}  
  --* @name YCbCr420ToYCrCb420 
  -- *  YCbCr420 to YCrCb420 sampling format conversion.
  -- * @{
  --  

  --*
  -- * 2 channel 8-bit unsigned planar YCbCr420 to 3 channel 8-bit unsigned planar YCrCb420 sampling format conversion.
  -- *
  -- * \param pSrcY \ref source_planar_image_pointer.
  -- * \param nSrcYStep \ref source_planar_image_line_step.
  -- * \param pSrcCbCr \ref source_planar_image_pointer.
  -- * \param nSrcCbCrStep \ref source_planar_image_line_step.
  -- * \param pDst \ref destination_planar_image_pointer_array.
  -- * \param rDstStep \ref destination_planar_image_line_step_array.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYCbCr420ToYCrCb420_8u_P2P3R
     (pSrcY : access nppdefs_h.Npp8u;
      nSrcYStep : int;
      pSrcCbCr : access nppdefs_h.Npp8u;
      nSrcCbCrStep : int;
      pDst : System.Address;
      rDstStep : access int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:3884
   pragma Import (C, nppiYCbCr420ToYCrCb420_8u_P2P3R, "nppiYCbCr420ToYCrCb420_8u_P2P3R");

  --* @}  
  --* @name YCrCb420ToCbYCr422 
  -- *  YCrCb420 to CbYCr422 sampling format conversion.
  -- * @{
  --  

  --*
  -- * 3 channel 8-bit unsigned planar YCrCb420 to 2 channel 8-bit unsigned packed CbYCr422 sampling format conversion.
  -- * images.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array.
  -- * \param rSrcStep \ref source_planar_image_line_step_array.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYCrCb420ToCbYCr422_8u_P3C2R
     (pSrc : System.Address;
      rSrcStep : access int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:3903
   pragma Import (C, nppiYCrCb420ToCbYCr422_8u_P3C2R, "nppiYCrCb420ToCbYCr422_8u_P3C2R");

  --* @}  
  --* @name YCrCb420ToYCbCr420 
  -- *  YCrCb420 to YCbCr420 sampling format conversion.
  -- * @{
  --  

  --*
  -- * 3 channel 8-bit unsigned planar YCrCb420 to 2 channel 8-bit unsigned planar YCbCr420 sampling format conversion.
  -- * images.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array.
  -- * \param rSrcStep \ref source_planar_image_line_step_array.
  -- * \param pDstY \ref destination_planar_image_pointer.
  -- * \param nDstYStep \ref destination_planar_image_line_step.
  -- * \param pDstCbCr \ref destination_planar_image_pointer.
  -- * \param nDstCbCrStep \ref destination_planar_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYCrCb420ToYCbCr420_8u_P3P2R
     (pSrc : System.Address;
      rSrcStep : access int;
      pDstY : access nppdefs_h.Npp8u;
      nDstYStep : int;
      pDstCbCr : access nppdefs_h.Npp8u;
      nDstCbCrStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:3924
   pragma Import (C, nppiYCrCb420ToYCbCr420_8u_P3P2R, "nppiYCrCb420ToYCbCr420_8u_P3P2R");

  --* @}  
  --* @name YCrCb420ToYCbCr411 
  -- *  YCrCb420 to YCbCr411 sampling format conversion.
  -- * @{
  --  

  --*
  -- * 3 channel 8-bit unsigned planar YCrCb420 to 2 channel 8-bit unsigned planar YCbCr411 sampling format conversion.
  -- * images.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array.
  -- * \param rSrcStep \ref source_planar_image_line_step_array.
  -- * \param pDstY \ref destination_planar_image_pointer.
  -- * \param nDstYStep \ref destination_planar_image_line_step.
  -- * \param pDstCbCr \ref destination_planar_image_pointer.
  -- * \param nDstCbCrStep \ref destination_planar_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYCrCb420ToYCbCr411_8u_P3P2R
     (pSrc : System.Address;
      rSrcStep : access int;
      pDstY : access nppdefs_h.Npp8u;
      nDstYStep : int;
      pDstCbCr : access nppdefs_h.Npp8u;
      nDstCbCrStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:3945
   pragma Import (C, nppiYCrCb420ToYCbCr411_8u_P3P2R, "nppiYCrCb420ToYCbCr411_8u_P3P2R");

  --* @}  
  --* @name YCbCr411ToYCbCr411 
  -- *  YCbCr411 to YCbCr411 sampling format conversion.
  -- * @{
  --  

  --*
  -- * 3 channel 8-bit unsigned planar YCbCr411 to 2 channel 8-bit unsigned planar YCbCr411 sampling format conversion.
  -- * images.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array.
  -- * \param rSrcStep \ref source_planar_image_line_step_array.
  -- * \param pDstY \ref destination_planar_image_pointer.
  -- * \param nDstYStep \ref destination_planar_image_line_step.
  -- * \param pDstCbCr \ref destination_planar_image_pointer.
  -- * \param nDstCbCrStep \ref destination_planar_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYCbCr411_8u_P3P2R
     (pSrc : System.Address;
      rSrcStep : access int;
      pDstY : access nppdefs_h.Npp8u;
      nDstYStep : int;
      pDstCbCr : access nppdefs_h.Npp8u;
      nDstCbCrStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:3966
   pragma Import (C, nppiYCbCr411_8u_P3P2R, "nppiYCbCr411_8u_P3P2R");

  --*
  -- * 2 channel 8-bit unsigned planar YCbCr411 to 3 channel 8-bit unsigned planar YCbCr411 sampling format conversion.
  -- *
  -- * \param pSrcY \ref source_planar_image_pointer.
  -- * \param nSrcYStep \ref source_planar_image_line_step.
  -- * \param pSrcCbCr \ref source_planar_image_pointer.
  -- * \param nSrcCbCrStep \ref source_planar_image_line_step.
  -- * \param pDst \ref destination_planar_image_pointer_array.
  -- * \param rDstStep \ref destination_planar_image_line_step_array.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYCbCr411_8u_P2P3R
     (pSrcY : access nppdefs_h.Npp8u;
      nSrcYStep : int;
      pSrcCbCr : access nppdefs_h.Npp8u;
      nSrcCbCrStep : int;
      pDst : System.Address;
      rDstStep : access int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:3980
   pragma Import (C, nppiYCbCr411_8u_P2P3R, "nppiYCbCr411_8u_P2P3R");

  --* @}  
  --* @name YCbCr411ToYCbCr422 
  -- *  YCbCr411 to YCbCr422 sampling format conversion.
  -- * @{
  --  

  --*
  -- * 3 channel 8-bit unsigned planar YCbCr411 to 3 channel 8-bit unsigned planar YCbCr422 sampling format conversion.
  -- * images.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array.
  -- * \param rSrcStep \ref source_planar_image_line_step_array.
  -- * \param pDst \ref destination_planar_image_pointer_array.
  -- * \param nDstStep \ref destination_planar_image_line_step_array.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYCbCr411ToYCbCr422_8u_P3R
     (pSrc : System.Address;
      rSrcStep : access int;
      pDst : System.Address;
      nDstStep : access int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:3999
   pragma Import (C, nppiYCbCr411ToYCbCr422_8u_P3R, "nppiYCbCr411ToYCbCr422_8u_P3R");

  --*
  -- * 3 channel 8-bit unsigned planar YCbCr411 to 2 channel 8-bit unsigned packed YCbCr422 sampling format conversion.
  -- * images.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array.
  -- * \param rSrcStep \ref source_planar_image_line_step_array.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYCbCr411ToYCbCr422_8u_P3C2R
     (pSrc : System.Address;
      rSrcStep : access int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:4012
   pragma Import (C, nppiYCbCr411ToYCbCr422_8u_P3C2R, "nppiYCbCr411ToYCbCr422_8u_P3C2R");

  --*
  -- * 2 channel 8-bit unsigned planar YCbCr411 to 3 channel 8-bit unsigned planar YCbCr422 sampling format conversion.
  -- *
  -- * \param pSrcY \ref source_planar_image_pointer.
  -- * \param nSrcYStep \ref source_planar_image_line_step.
  -- * \param pSrcCbCr \ref source_planar_image_pointer.
  -- * \param nSrcCbCrStep \ref source_planar_image_line_step.
  -- * \param pDst \ref destination_planar_image_pointer_array.
  -- * \param rDstStep \ref destination_planar_image_line_step_array.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYCbCr411ToYCbCr422_8u_P2P3R
     (pSrcY : access nppdefs_h.Npp8u;
      nSrcYStep : int;
      pSrcCbCr : access nppdefs_h.Npp8u;
      nSrcCbCrStep : int;
      pDst : System.Address;
      rDstStep : access int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:4026
   pragma Import (C, nppiYCbCr411ToYCbCr422_8u_P2P3R, "nppiYCbCr411ToYCbCr422_8u_P2P3R");

  --*
  -- * 2 channel 8-bit unsigned planar YCbCr411 to 2 channel 8-bit unsigned packed YCbCr422 sampling format conversion.
  -- *
  -- * \param pSrcY \ref source_planar_image_pointer.
  -- * \param nSrcYStep \ref source_planar_image_line_step.
  -- * \param pSrcCbCr \ref source_planar_image_pointer.
  -- * \param nSrcCbCrStep \ref source_planar_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYCbCr411ToYCbCr422_8u_P2C2R
     (pSrcY : access nppdefs_h.Npp8u;
      nSrcYStep : int;
      pSrcCbCr : access nppdefs_h.Npp8u;
      nSrcCbCrStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:4040
   pragma Import (C, nppiYCbCr411ToYCbCr422_8u_P2C2R, "nppiYCbCr411ToYCbCr422_8u_P2C2R");

  --* @}  
  --* @name YCbCr411ToYCrCb422 
  -- *  YCbCr411 to YCrCb422 sampling format conversion.
  -- * @{
  --  

  --*
  -- * 3 channel 8-bit unsigned planar YCbCr411 to 3 channel 8-bit unsigned planar YCrCb422 sampling format conversion.
  -- * images.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array.
  -- * \param rSrcStep \ref source_planar_image_line_step_array.
  -- * \param pDst \ref destination_planar_image_pointer_array.
  -- * \param nDstStep \ref destination_planar_image_line_step_array.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYCbCr411ToYCrCb422_8u_P3R
     (pSrc : System.Address;
      rSrcStep : access int;
      pDst : System.Address;
      nDstStep : access int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:4059
   pragma Import (C, nppiYCbCr411ToYCrCb422_8u_P3R, "nppiYCbCr411ToYCrCb422_8u_P3R");

  --*
  -- * 3 channel 8-bit unsigned planar YCbCr411 to 2 channel 8-bit unsigned packed YCrCb422 sampling format conversion.
  -- * images.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array.
  -- * \param rSrcStep \ref source_planar_image_line_step_array.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYCbCr411ToYCrCb422_8u_P3C2R
     (pSrc : System.Address;
      rSrcStep : access int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:4072
   pragma Import (C, nppiYCbCr411ToYCrCb422_8u_P3C2R, "nppiYCbCr411ToYCrCb422_8u_P3C2R");

  --* @}  
  --* @name YCbCr411ToYCbCr420 
  -- *  YCbCr411 to YCbCr420 sampling format conversion.
  -- * @{
  --  

  --*
  -- * 3 channel 8-bit unsigned planar YCbCr411 to 3 channel 8-bit unsigned planar YCbCr420 sampling format conversion.
  -- * images.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array.
  -- * \param rSrcStep \ref source_planar_image_line_step_array.
  -- * \param pDst \ref destination_planar_image_pointer_array.
  -- * \param nDstStep \ref destination_planar_image_line_step_array.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYCbCr411ToYCbCr420_8u_P3R
     (pSrc : System.Address;
      rSrcStep : access int;
      pDst : System.Address;
      nDstStep : access int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:4091
   pragma Import (C, nppiYCbCr411ToYCbCr420_8u_P3R, "nppiYCbCr411ToYCbCr420_8u_P3R");

  --*
  -- * 3 channel 8-bit unsigned planar YCbCr411 to 2 channel 8-bit unsigned planar YCbCr420 sampling format conversion.
  -- * images.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array.
  -- * \param rSrcStep \ref source_planar_image_line_step_array.
  -- * \param pDstY \ref destination_planar_image_pointer.
  -- * \param nDstYStep \ref destination_planar_image_line_step.
  -- * \param pDstCbCr \ref destination_planar_image_pointer.
  -- * \param nDstCbCrStep \ref destination_planar_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYCbCr411ToYCbCr420_8u_P3P2R
     (pSrc : System.Address;
      rSrcStep : access int;
      pDstY : access nppdefs_h.Npp8u;
      nDstYStep : int;
      pDstCbCr : access nppdefs_h.Npp8u;
      nDstCbCrStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:4106
   pragma Import (C, nppiYCbCr411ToYCbCr420_8u_P3P2R, "nppiYCbCr411ToYCbCr420_8u_P3P2R");

  --*
  -- * 2 channel 8-bit unsigned planar YCbCr411 to 3 channel 8-bit unsigned planar YCbCr420 sampling format conversion.
  -- *
  -- * \param pSrcY \ref source_planar_image_pointer.
  -- * \param nSrcYStep \ref source_planar_image_line_step.
  -- * \param pSrcCbCr \ref source_planar_image_pointer.
  -- * \param nSrcCbCrStep \ref source_planar_image_line_step.
  -- * \param pDst \ref destination_planar_image_pointer_array.
  -- * \param rDstStep \ref destination_planar_image_line_step_array.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYCbCr411ToYCbCr420_8u_P2P3R
     (pSrcY : access nppdefs_h.Npp8u;
      nSrcYStep : int;
      pSrcCbCr : access nppdefs_h.Npp8u;
      nSrcCbCrStep : int;
      pDst : System.Address;
      rDstStep : access int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:4120
   pragma Import (C, nppiYCbCr411ToYCbCr420_8u_P2P3R, "nppiYCbCr411ToYCbCr420_8u_P2P3R");

  --* @}  
  --* @name YCbCr411ToYCrCb420 
  -- *  YCbCr411 to YCrCb420 sampling format conversion.
  -- * @{
  --  

  --*
  -- * 2 channel 8-bit unsigned planar YCbCr411 to 3 channel 8-bit unsigned planar YCrCb420 sampling format conversion.
  -- *
  -- * \param pSrcY \ref source_planar_image_pointer.
  -- * \param nSrcYStep \ref source_planar_image_line_step.
  -- * \param pSrcCbCr \ref source_planar_image_pointer.
  -- * \param nSrcCbCrStep \ref source_planar_image_line_step.
  -- * \param pDst \ref destination_planar_image_pointer_array.
  -- * \param rDstStep \ref destination_planar_image_line_step_array.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiYCbCr411ToYCrCb420_8u_P2P3R
     (pSrcY : access nppdefs_h.Npp8u;
      nSrcYStep : int;
      pSrcCbCr : access nppdefs_h.Npp8u;
      nSrcCbCrStep : int;
      pDst : System.Address;
      rDstStep : access int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:4140
   pragma Import (C, nppiYCbCr411ToYCrCb420_8u_P2P3R, "nppiYCbCr411ToYCrCb420_8u_P2P3R");

  --* @}  
  --* @} image_color_sampling_format_conversion  
  --* @defgroup image_color_gamma_correction Color Gamma Correction
  -- *
  -- * Routines for correcting image color gamma.
  -- *
  -- * @{                                         
  -- *
  --  

  --* @name GammaFwd 
  -- *  Forward gamma correction.
  -- * @{
  --  

  --*
  -- * 3 channel 8-bit unsigned packed color not in place forward gamma correction.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiGammaFwd_8u_C3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:4168
   pragma Import (C, nppiGammaFwd_8u_C3R, "nppiGammaFwd_8u_C3R");

  --*
  -- * 3 channel 8-bit unsigned packed color in place forward gamma correction.
  -- *
  -- * \param pSrcDst in place packed pixel image pointer.
  -- * \param nSrcDstStep in place packed pixel format image line step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiGammaFwd_8u_C3IR
     (pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:4178
   pragma Import (C, nppiGammaFwd_8u_C3IR, "nppiGammaFwd_8u_C3IR");

  --*
  -- * 4 channel 8-bit unsigned packed color with alpha not in place forward gamma correction.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiGammaFwd_8u_AC4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:4190
   pragma Import (C, nppiGammaFwd_8u_AC4R, "nppiGammaFwd_8u_AC4R");

  --*
  -- * 4 channel 8-bit unsigned packed color with alpha in place forward gamma correction.
  -- *
  -- * \param pSrcDst in place packed pixel format image pointer.
  -- * \param nSrcDstStep in place packed pixel format image line step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiGammaFwd_8u_AC4IR
     (pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:4200
   pragma Import (C, nppiGammaFwd_8u_AC4IR, "nppiGammaFwd_8u_AC4IR");

  --*
  -- * 3 channel 8-bit unsigned planar color not in place forward gamma correction.
  -- *
  -- * \param pSrc source planar pixel format image pointer array.
  -- * \param nSrcStep source planar pixel format image line step.
  -- * \param pDst destination planar pixel format image pointer array.
  -- * \param nDstStep destination planar pixel format image line step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiGammaFwd_8u_P3R
     (pSrc : System.Address;
      nSrcStep : int;
      pDst : System.Address;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:4212
   pragma Import (C, nppiGammaFwd_8u_P3R, "nppiGammaFwd_8u_P3R");

  --*
  -- * 3 channel 8-bit unsigned planar color in place forward gamma correction.
  -- *
  -- * \param pSrcDst in place planar pixel format image pointer array.
  -- * \param nSrcDstStep in place planar pixel format image line step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiGammaFwd_8u_IP3R
     (pSrcDst : System.Address;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:4222
   pragma Import (C, nppiGammaFwd_8u_IP3R, "nppiGammaFwd_8u_IP3R");

  --* @}  
  --* @name GammaInv 
  -- *  Inverse gamma correction.
  -- * @{
  --  

  --*
  -- * 3 channel 8-bit unsigned packed color not in place inverse gamma correction.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiGammaInv_8u_C3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:4240
   pragma Import (C, nppiGammaInv_8u_C3R, "nppiGammaInv_8u_C3R");

  --*
  -- * 3 channel 8-bit unsigned packed color in place inverse gamma correction.
  -- *
  -- * \param pSrcDst in place packed pixel format image pointer.
  -- * \param nSrcDstStep in place packed pixel format image line step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiGammaInv_8u_C3IR
     (pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:4250
   pragma Import (C, nppiGammaInv_8u_C3IR, "nppiGammaInv_8u_C3IR");

  --*
  -- * 4 channel 8-bit unsigned packed color with alpha not in place inverse gamma correction.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiGammaInv_8u_AC4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:4262
   pragma Import (C, nppiGammaInv_8u_AC4R, "nppiGammaInv_8u_AC4R");

  --*
  -- * 4 channel 8-bit unsigned packed color with alpha in place inverse gamma correction.
  -- *
  -- * \param pSrcDst in place packed pixel format image pointer.
  -- * \param nSrcDstStep in place packed pixel format image line step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiGammaInv_8u_AC4IR
     (pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:4272
   pragma Import (C, nppiGammaInv_8u_AC4IR, "nppiGammaInv_8u_AC4IR");

  --*
  -- * 3 channel 8-bit unsigned planar color not in place inverse gamma correction.
  -- *
  -- * \param pSrc source planar pixel format image pointer array.
  -- * \param nSrcStep source planar pixel format image line step.
  -- * \param pDst destination planar pixel format image pointer array.
  -- * \param nDstStep destination planar pixel format image line step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiGammaInv_8u_P3R
     (pSrc : System.Address;
      nSrcStep : int;
      pDst : System.Address;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:4284
   pragma Import (C, nppiGammaInv_8u_P3R, "nppiGammaInv_8u_P3R");

  --*
  -- * 3 channel 8-bit unsigned planar color in place inverse gamma correction.
  -- *
  -- * \param pSrcDst in place planar pixel format image pointer array.
  -- * \param nSrcDstStep in place planar pixel format image line step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiGammaInv_8u_IP3R
     (pSrcDst : System.Address;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:4294
   pragma Import (C, nppiGammaInv_8u_IP3R, "nppiGammaInv_8u_IP3R");

  --* @}  
  --* @} image_color_gamma_correction  
  --* @defgroup image_complement_color_key Complement Color Key
  -- *
  -- * Routines for performing complement color key replacement.
  -- *
  -- * @{                                         
  -- *
  --  

  --* @name CompColorKey 
  -- *  Complement color key replacement.
  -- * @{
  --  

  --*
  -- * 1 channel 8-bit unsigned packed color complement color key replacement of source image 1 by source image 2.
  -- *
  -- * \param pSrc1 source1 packed pixel format image pointer.
  -- * \param nSrc1Step source1 packed pixel format image line step.
  -- * \param pSrc2 source2 packed pixel format image pointer.
  -- * \param nSrc2Step source2 packed pixel format image line step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nColorKeyConst color key constant
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCompColorKey_8u_C1R
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp8u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nColorKeyConst : nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:4325
   pragma Import (C, nppiCompColorKey_8u_C1R, "nppiCompColorKey_8u_C1R");

  --*
  -- * 3 channel 8-bit unsigned packed color complement color key replacement of source image 1 by source image 2.
  -- *
  -- * \param pSrc1 source1 packed pixel format image pointer.
  -- * \param nSrc1Step source1 packed pixel format image line step.
  -- * \param pSrc2 source2 packed pixel format image pointer.
  -- * \param nSrc2Step source2 packed pixel format image line step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nColorKeyConst color key constant array
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCompColorKey_8u_C3R
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp8u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nColorKeyConst : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:4340
   pragma Import (C, nppiCompColorKey_8u_C3R, "nppiCompColorKey_8u_C3R");

  --*
  -- * 4 channel 8-bit unsigned packed color complement color key replacement of source image 1 by source image 2.
  -- *
  -- * \param pSrc1 source1 packed pixel format image pointer.
  -- * \param nSrc1Step source1 packed pixel format image line step.
  -- * \param pSrc2 source2 packed pixel format image pointer.
  -- * \param nSrc2Step source2 packed pixel format image line step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nColorKeyConst color key constant array
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCompColorKey_8u_C4R
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp8u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nColorKeyConst : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:4355
   pragma Import (C, nppiCompColorKey_8u_C4R, "nppiCompColorKey_8u_C4R");

  --*
  -- * 4 channel 8-bit unsigned packed color complement color key replacement of source image 1 by source image 2 with alpha blending.
  -- *
  -- * \param pSrc1 source1 packed pixel format image pointer.
  -- * \param nSrc1Step source1 packed pixel format image line step.
  -- * \param nAlpha1 source1 image alpha opacity (0 - max channel pixel value).
  -- * \param pSrc2 source2 packed pixel format image pointer.
  -- * \param nSrc2Step source2 packed pixel format image line step.
  -- * \param nAlpha2 source2 image alpha opacity (0 - max channel pixel value).
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nColorKeyConst color key constant array
  -- * \param nppAlphaOp NppiAlphaOp alpha compositing operation selector  (excluding premul ops).
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAlphaCompColorKey_8u_AC4R
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      nAlpha1 : nppdefs_h.Npp8u;
      pSrc2 : access nppdefs_h.Npp8u;
      nSrc2Step : int;
      nAlpha2 : nppdefs_h.Npp8u;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nColorKeyConst : access nppdefs_h.Npp8u;
      nppAlphaOp : nppdefs_h.NppiAlphaOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:4373
   pragma Import (C, nppiAlphaCompColorKey_8u_AC4R, "nppiAlphaCompColorKey_8u_AC4R");

  --* @}  
  --* @} image_complement_color_key  
  --* @defgroup image_color_processing Color Processing
  -- *
  -- * Routines for performing image color manipulation.
  -- *
  -- * @{                                         
  -- *
  --  

  --* @name ColorTwist
  -- * 
  -- *  Perform color twist pixel processing.  Color twist consists of applying the following formula to each
  -- *  image pixel using coefficients from the user supplied color twist host matrix array as follows where 
  -- *  dst[x] and src[x] represent destination pixel and source pixel channel or plane x. The full sized
  -- *  coefficient matrix should be sent for all pixel channel sizes, the function will process the appropriate
  -- *  coefficients and channels for the corresponding pixel size.
  -- *
  -- *  \code
  -- *      dst[0] = aTwist[0][0] * src[0] + aTwist[0][1] * src[1] + aTwist[0][2] * src[2] + aTwist[0][3]
  -- *      dst[1] = aTwist[1][0] * src[0] + aTwist[1][1] * src[1] + aTwist[1][2] * src[2] + aTwist[1][3]
  -- *      dst[2] = aTwist[2][0] * src[0] + aTwist[2][1] * src[1] + aTwist[2][2] * src[2] + aTwist[2][3]
  -- *  \endcode
  -- *
  -- * @{
  --  

  --*
  -- * 1 channel 8-bit unsigned color twist.
  -- * 
  -- * An input color twist matrix with floating-point coefficient values is applied
  -- * within ROI.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aTwist The color twist matrix with floating-point coefficient values.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiColorTwist32f_8u_C1R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aTwist : System.Address) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:4419
   pragma Import (C, nppiColorTwist32f_8u_C1R, "nppiColorTwist32f_8u_C1R");

  --*
  -- * 1 channel 8-bit unsigned in place color twist.
  -- * 
  -- * An input color twist matrix with floating-point coefficient values is applied
  -- * within ROI.
  -- *
  -- * \param pSrcDst in place packed pixel format image pointer.
  -- * \param nSrcDstStep in place packed pixel format image line step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aTwist The color twist matrix with floating-point coefficient values.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiColorTwist32f_8u_C1IR
     (pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aTwist : System.Address) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:4435
   pragma Import (C, nppiColorTwist32f_8u_C1IR, "nppiColorTwist32f_8u_C1IR");

  --*
  -- * 2 channel 8-bit unsigned color twist.
  -- * 
  -- * An input color twist matrix with floating-point coefficient values is applied
  -- * within ROI.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aTwist The color twist matrix with floating-point coefficient values.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiColorTwist32f_8u_C2R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aTwist : System.Address) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:4453
   pragma Import (C, nppiColorTwist32f_8u_C2R, "nppiColorTwist32f_8u_C2R");

  --*
  -- * 2 channel 8-bit unsigned in place color twist.
  -- * 
  -- * An input color twist matrix with floating-point coefficient values is applied
  -- * within ROI.
  -- *
  -- * \param pSrcDst in place packed pixel format image pointer.
  -- * \param nSrcDstStep in place packed pixel format image line step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aTwist The color twist matrix with floating-point coefficient values.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiColorTwist32f_8u_C2IR
     (pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aTwist : System.Address) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:4469
   pragma Import (C, nppiColorTwist32f_8u_C2IR, "nppiColorTwist32f_8u_C2IR");

  --*
  -- * 3 channel 8-bit unsigned color twist.
  -- * 
  -- * An input color twist matrix with floating-point coefficient values is applied
  -- * within ROI.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aTwist The color twist matrix with floating-point coefficient values.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiColorTwist32f_8u_C3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aTwist : System.Address) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:4487
   pragma Import (C, nppiColorTwist32f_8u_C3R, "nppiColorTwist32f_8u_C3R");

  --*
  -- * 3 channel 8-bit unsigned in place color twist.
  -- * 
  -- * An input color twist matrix with floating-point coefficient values is applied
  -- * within ROI.
  -- *
  -- * \param pSrcDst in place packed pixel format image pointer.
  -- * \param nSrcDstStep in place packed pixel format image line step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aTwist The color twist matrix with floating-point coefficient values.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiColorTwist32f_8u_C3IR
     (pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aTwist : System.Address) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:4503
   pragma Import (C, nppiColorTwist32f_8u_C3IR, "nppiColorTwist32f_8u_C3IR");

  --*
  -- * 4 channel 8-bit unsigned color twist, with alpha copy.
  -- *
  -- * An input color twist matrix with floating-point coefficient values is applied with
  -- * in ROI.
  -- * Alpha channel is the last channel and is copied unmodified from the source pixel to the destination pixel.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aTwist The color twist matrix with floating-point coefficient values.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiColorTwist32f_8u_C4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aTwist : System.Address) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:4522
   pragma Import (C, nppiColorTwist32f_8u_C4R, "nppiColorTwist32f_8u_C4R");

  --*
  -- * 4 channel 8-bit unsigned in place color twist, not affecting Alpha.
  -- *
  -- * An input color twist matrix with floating-point coefficient values is applied with
  -- * in ROI.
  -- * Alpha channel is the last channel and is unmodified.
  -- *
  -- * \param pSrcDst in place packed pixel format image pointer.
  -- * \param nSrcDstStep in place packed pixel format image line step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aTwist The color twist matrix with floating-point coefficient values.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiColorTwist32f_8u_C4IR
     (pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aTwist : System.Address) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:4539
   pragma Import (C, nppiColorTwist32f_8u_C4IR, "nppiColorTwist32f_8u_C4IR");

  --*
  -- * 4 channel 8-bit unsigned color twist, not affecting Alpha.
  -- *
  -- * An input color twist matrix with floating-point coefficient values is applied with
  -- * in ROI.
  -- * Alpha channel is the last channel and is not processed.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aTwist The color twist matrix with floating-point coefficient values.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiColorTwist32f_8u_AC4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aTwist : System.Address) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:4558
   pragma Import (C, nppiColorTwist32f_8u_AC4R, "nppiColorTwist32f_8u_AC4R");

  --*
  -- * 4 channel 8-bit unsigned in place color twist, not affecting Alpha.
  -- *
  -- * An input color twist matrix with floating-point coefficient values is applied with
  -- * in ROI.
  -- * Alpha channel is the last channel and is not processed.
  -- *
  -- * \param pSrcDst in place packed pixel format image pointer.
  -- * \param nSrcDstStep in place packed pixel format image line step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aTwist The color twist matrix with floating-point coefficient values.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiColorTwist32f_8u_AC4IR
     (pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aTwist : System.Address) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:4575
   pragma Import (C, nppiColorTwist32f_8u_AC4IR, "nppiColorTwist32f_8u_AC4IR");

  --*
  -- * 4 channel 8-bit unsigned color twist with 4x4 matrix and constant vector addition.
  -- *
  -- * An input 4x4 color twist matrix with floating-point coefficient values with an additional constant vector addition
  -- * is applied within ROI.  For this particular version of the function the result is generated as shown below.
  -- *
  -- *  \code
  -- *      dst[0] = aTwist[0][0] * src[0] + aTwist[0][1] * src[1] + aTwist[0][2] * src[2] + aTwist[0][3] * src[3] + aConstants[0]
  -- *      dst[1] = aTwist[1][0] * src[0] + aTwist[1][1] * src[1] + aTwist[1][2] * src[2] + aTwist[1][3] * src[3] + aConstants[1]
  -- *      dst[2] = aTwist[2][0] * src[0] + aTwist[2][1] * src[1] + aTwist[2][2] * src[2] + aTwist[2][3] * src[3] + aConstants[2]
  -- *      dst[3] = aTwist[3][0] * src[0] + aTwist[3][1] * src[1] + aTwist[3][2] * src[2] + aTwist[3][3] * src[3] + aConstants[3]
  -- *  \endcode
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aTwist The color twist matrix with floating-point coefficient values.
  -- * \param aConstants fixed size array of constant values, one per channel..
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiColorTwist32fC_8u_C4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aTwist : System.Address;
      aConstants : access nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:4601
   pragma Import (C, nppiColorTwist32fC_8u_C4R, "nppiColorTwist32fC_8u_C4R");

  --*
  -- * 4 channel 8-bit unsigned in place color twist with 4x4 matrix and an additional constant vector addition.
  -- *
  -- * An input 4x4 color twist matrix with floating-point coefficient values with an additional constant vector addition
  -- * is applied within ROI.  For this particular version of the function the result is generated as shown below.
  -- *
  -- *  \code
  -- *      dst[0] = aTwist[0][0] * src[0] + aTwist[0][1] * src[1] + aTwist[0][2] * src[2] + aTwist[0][3] * src[3] + aConstants[0]
  -- *      dst[1] = aTwist[1][0] * src[0] + aTwist[1][1] * src[1] + aTwist[1][2] * src[2] + aTwist[1][3] * src[3] + aConstants[1]
  -- *      dst[2] = aTwist[2][0] * src[0] + aTwist[2][1] * src[1] + aTwist[2][2] * src[2] + aTwist[2][3] * src[3] + aConstants[2]
  -- *      dst[3] = aTwist[3][0] * src[0] + aTwist[3][1] * src[1] + aTwist[3][2] * src[2] + aTwist[3][3] * src[3] + aConstants[3]
  -- *  \endcode
  -- *
  -- * \param pSrcDst in place packed pixel format image pointer.
  -- * \param nSrcDstStep in place packed pixel format image line step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aTwist The color twist matrix with floating-point coefficient values.
  -- * \param aConstants fixed size array of constant values, one per channel..
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiColorTwist32fC_8u_C4IR
     (pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aTwist : System.Address;
      aConstants : access nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:4625
   pragma Import (C, nppiColorTwist32fC_8u_C4IR, "nppiColorTwist32fC_8u_C4IR");

  --*
  -- * 3 channel 8-bit unsigned planar color twist.
  -- *
  -- * An input color twist matrix with floating-point coefficient values is applied
  -- * within ROI.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aTwist The color twist matrix with floating-point coefficient values.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiColorTwist32f_8u_P3R
     (pSrc : System.Address;
      nSrcStep : int;
      pDst : System.Address;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aTwist : System.Address) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:4643
   pragma Import (C, nppiColorTwist32f_8u_P3R, "nppiColorTwist32f_8u_P3R");

  --*
  -- * 3 channel 8-bit unsigned planar in place color twist.
  -- * 
  -- * An input color twist matrix with floating-point coefficient values is applied
  -- * within ROI.
  -- *
  -- * \param pSrcDst in place planar pixel format image pointer array, one pointer per plane.
  -- * \param nSrcDstStep in place planar pixel format image line step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aTwist The color twist matrix with floating-point coefficient values.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiColorTwist32f_8u_IP3R
     (pSrcDst : System.Address;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aTwist : System.Address) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:4659
   pragma Import (C, nppiColorTwist32f_8u_IP3R, "nppiColorTwist32f_8u_IP3R");

  --*
  -- * 1 channel 8-bit signed color twist.
  -- * 
  -- * An input color twist matrix with floating-point coefficient values is applied
  -- * within ROI.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aTwist The color twist matrix with floating-point coefficient values.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiColorTwist32f_8s_C1R
     (pSrc : access nppdefs_h.Npp8s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aTwist : System.Address) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:4677
   pragma Import (C, nppiColorTwist32f_8s_C1R, "nppiColorTwist32f_8s_C1R");

  --*
  -- * 1 channel 8-bit signed in place color twist.
  -- * 
  -- * An input color twist matrix with floating-point coefficient values is applied
  -- * within ROI.
  -- *
  -- * \param pSrcDst in place packed pixel format image pointer.
  -- * \param nSrcDstStep in place packed pixel format image line step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aTwist The color twist matrix with floating-point coefficient values.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiColorTwist32f_8s_C1IR
     (pSrcDst : access nppdefs_h.Npp8s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aTwist : System.Address) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:4693
   pragma Import (C, nppiColorTwist32f_8s_C1IR, "nppiColorTwist32f_8s_C1IR");

  --*
  -- * 2 channel 8-bit signed color twist.
  -- * 
  -- * An input color twist matrix with floating-point coefficient values is applied
  -- * within ROI.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aTwist The color twist matrix with floating-point coefficient values.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiColorTwist32f_8s_C2R
     (pSrc : access nppdefs_h.Npp8s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aTwist : System.Address) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:4711
   pragma Import (C, nppiColorTwist32f_8s_C2R, "nppiColorTwist32f_8s_C2R");

  --*
  -- * 2 channel 8-bit signed in place color twist.
  -- * 
  -- * An input color twist matrix with floating-point coefficient values is applied
  -- * within ROI.
  -- *
  -- * \param pSrcDst in place packed pixel format image pointer.
  -- * \param nSrcDstStep in place packed pixel format image line step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aTwist The color twist matrix with floating-point coefficient values.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiColorTwist32f_8s_C2IR
     (pSrcDst : access nppdefs_h.Npp8s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aTwist : System.Address) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:4727
   pragma Import (C, nppiColorTwist32f_8s_C2IR, "nppiColorTwist32f_8s_C2IR");

  --*
  -- * 3 channel 8-bit signed color twist.
  -- * 
  -- * An input color twist matrix with floating-point coefficient values is applied
  -- * within ROI.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aTwist The color twist matrix with floating-point coefficient values.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiColorTwist32f_8s_C3R
     (pSrc : access nppdefs_h.Npp8s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aTwist : System.Address) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:4745
   pragma Import (C, nppiColorTwist32f_8s_C3R, "nppiColorTwist32f_8s_C3R");

  --*
  -- * 3 channel 8-bit signed in place color twist.
  -- * 
  -- * An input color twist matrix with floating-point coefficient values is applied
  -- * within ROI.
  -- *
  -- * \param pSrcDst in place packed pixel format image pointer.
  -- * \param nSrcDstStep in place packed pixel format image line step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aTwist The color twist matrix with floating-point coefficient values.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiColorTwist32f_8s_C3IR
     (pSrcDst : access nppdefs_h.Npp8s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aTwist : System.Address) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:4761
   pragma Import (C, nppiColorTwist32f_8s_C3IR, "nppiColorTwist32f_8s_C3IR");

  --*
  -- * 4 channel 8-bit signed color twist, with alpha copy.
  -- *
  -- * An input color twist matrix with floating-point coefficient values is applied with
  -- * in ROI.
  -- * Alpha channel is the last channel and is copied unmodified from the source pixel to the destination pixel.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aTwist The color twist matrix with floating-point coefficient values.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiColorTwist32f_8s_C4R
     (pSrc : access nppdefs_h.Npp8s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aTwist : System.Address) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:4780
   pragma Import (C, nppiColorTwist32f_8s_C4R, "nppiColorTwist32f_8s_C4R");

  --*
  -- * 4 channel 8-bit signed in place color twist, not affecting Alpha.
  -- *
  -- * An input color twist matrix with floating-point coefficient values is applied with
  -- * in ROI.
  -- * Alpha channel is the last channel and is unmodified.
  -- *
  -- * \param pSrcDst in place packed pixel format image pointer.
  -- * \param nSrcDstStep in place packed pixel format image line step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aTwist The color twist matrix with floating-point coefficient values.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiColorTwist32f_8s_C4IR
     (pSrcDst : access nppdefs_h.Npp8s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aTwist : System.Address) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:4797
   pragma Import (C, nppiColorTwist32f_8s_C4IR, "nppiColorTwist32f_8s_C4IR");

  --*
  -- * 4 channel 8-bit signed color twist, not affecting Alpha.
  -- *
  -- * An input color twist matrix with floating-point coefficient values is applied with
  -- * in ROI.
  -- * Alpha channel is the last channel and is not processed.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aTwist The color twist matrix with floating-point coefficient values.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiColorTwist32f_8s_AC4R
     (pSrc : access nppdefs_h.Npp8s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aTwist : System.Address) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:4816
   pragma Import (C, nppiColorTwist32f_8s_AC4R, "nppiColorTwist32f_8s_AC4R");

  --*
  -- * 4 channel 8-bit signed in place color twist, not affecting Alpha.
  -- *
  -- * An input color twist matrix with floating-point coefficient values is applied with
  -- * in ROI.
  -- * Alpha channel is the last channel and is not processed.
  -- *
  -- * \param pSrcDst in place packed pixel format image pointer.
  -- * \param nSrcDstStep in place packed pixel format image line step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aTwist The color twist matrix with floating-point coefficient values.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiColorTwist32f_8s_AC4IR
     (pSrcDst : access nppdefs_h.Npp8s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aTwist : System.Address) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:4833
   pragma Import (C, nppiColorTwist32f_8s_AC4IR, "nppiColorTwist32f_8s_AC4IR");

  --*
  -- * 3 channel 8-bit signed planar color twist.
  -- *
  -- * An input color twist matrix with floating-point coefficient values is applied
  -- * within ROI.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aTwist The color twist matrix with floating-point coefficient values.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiColorTwist32f_8s_P3R
     (pSrc : System.Address;
      nSrcStep : int;
      pDst : System.Address;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aTwist : System.Address) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:4851
   pragma Import (C, nppiColorTwist32f_8s_P3R, "nppiColorTwist32f_8s_P3R");

  --*
  -- * 3 channel 8-bit signed planar in place color twist.
  -- * 
  -- * An input color twist matrix with floating-point coefficient values is applied
  -- * within ROI.
  -- *
  -- * \param pSrcDst in place planar pixel format image pointer array, one pointer per plane.
  -- * \param nSrcDstStep in place planar pixel format image line step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aTwist The color twist matrix with floating-point coefficient values.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiColorTwist32f_8s_IP3R
     (pSrcDst : System.Address;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aTwist : System.Address) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:4867
   pragma Import (C, nppiColorTwist32f_8s_IP3R, "nppiColorTwist32f_8s_IP3R");

  --*
  -- * 1 channel 16-bit unsigned color twist.
  -- * 
  -- * An input color twist matrix with floating-point coefficient values is applied
  -- * within ROI.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aTwist The color twist matrix with floating-point coefficient values.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiColorTwist32f_16u_C1R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aTwist : System.Address) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:4886
   pragma Import (C, nppiColorTwist32f_16u_C1R, "nppiColorTwist32f_16u_C1R");

  --*
  -- * 1 channel 16-bit unsigned in place color twist.
  -- * 
  -- * An input color twist matrix with floating-point coefficient values is applied
  -- * within ROI.
  -- *
  -- * \param pSrcDst in place packed pixel format image pointer.
  -- * \param nSrcDstStep in place packed pixel format image line step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aTwist The color twist matrix with floating-point coefficient values.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiColorTwist32f_16u_C1IR
     (pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aTwist : System.Address) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:4902
   pragma Import (C, nppiColorTwist32f_16u_C1IR, "nppiColorTwist32f_16u_C1IR");

  --*
  -- * 2 channel 16-bit unsigned color twist.
  -- * 
  -- * An input color twist matrix with floating-point coefficient values is applied
  -- * within ROI.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aTwist The color twist matrix with floating-point coefficient values.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiColorTwist32f_16u_C2R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aTwist : System.Address) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:4920
   pragma Import (C, nppiColorTwist32f_16u_C2R, "nppiColorTwist32f_16u_C2R");

  --*
  -- * 2 channel 16-bit unsigned in place color twist.
  -- * 
  -- * An input color twist matrix with floating-point coefficient values is applied
  -- * within ROI.
  -- *
  -- * \param pSrcDst in place packed pixel format image pointer.
  -- * \param nSrcDstStep in place packed pixel format image line step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aTwist The color twist matrix with floating-point coefficient values.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiColorTwist32f_16u_C2IR
     (pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aTwist : System.Address) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:4936
   pragma Import (C, nppiColorTwist32f_16u_C2IR, "nppiColorTwist32f_16u_C2IR");

  --*
  -- * 3 channel 16-bit unsigned color twist.
  -- * 
  -- * An input color twist matrix with floating-point coefficient values is applied
  -- * within ROI.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aTwist The color twist matrix with floating-point coefficient values.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiColorTwist32f_16u_C3R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aTwist : System.Address) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:4954
   pragma Import (C, nppiColorTwist32f_16u_C3R, "nppiColorTwist32f_16u_C3R");

  --*
  -- * 3 channel 16-bit unsigned in place color twist.
  -- * 
  -- * An input color twist matrix with floating-point coefficient values is applied
  -- * within ROI.
  -- *
  -- * \param pSrcDst in place packed pixel format image pointer.
  -- * \param nSrcDstStep in place packed pixel format image line step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aTwist The color twist matrix with floating-point coefficient values.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiColorTwist32f_16u_C3IR
     (pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aTwist : System.Address) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:4970
   pragma Import (C, nppiColorTwist32f_16u_C3IR, "nppiColorTwist32f_16u_C3IR");

  --*
  -- * 4 channel 16-bit unsigned color twist, not affecting Alpha.
  -- *
  -- * An input color twist matrix with floating-point coefficient values is applied with
  -- * in ROI.
  -- * Alpha channel is the last channel and is not processed.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aTwist The color twist matrix with floating-point coefficient values.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiColorTwist32f_16u_AC4R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aTwist : System.Address) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:4989
   pragma Import (C, nppiColorTwist32f_16u_AC4R, "nppiColorTwist32f_16u_AC4R");

  --*
  -- * 4 channel 16-bit unsigned in place color twist, not affecting Alpha.
  -- *
  -- * An input color twist matrix with floating-point coefficient values is applied with
  -- * in ROI.
  -- * Alpha channel is the last channel and is not processed.
  -- *
  -- * \param pSrcDst in place packed pixel format image pointer.
  -- * \param nSrcDstStep in place packed pixel format image line step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aTwist The color twist matrix with floating-point coefficient values.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiColorTwist32f_16u_AC4IR
     (pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aTwist : System.Address) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:5006
   pragma Import (C, nppiColorTwist32f_16u_AC4IR, "nppiColorTwist32f_16u_AC4IR");

  --*
  -- * 3 channel 16-bit unsigned planar color twist.
  -- *
  -- * An input color twist matrix with floating-point coefficient values is applied
  -- * within ROI.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aTwist The color twist matrix with floating-point coefficient values.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiColorTwist32f_16u_P3R
     (pSrc : System.Address;
      nSrcStep : int;
      pDst : System.Address;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aTwist : System.Address) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:5024
   pragma Import (C, nppiColorTwist32f_16u_P3R, "nppiColorTwist32f_16u_P3R");

  --*
  -- * 3 channel 16-bit unsigned planar in place color twist.
  -- * 
  -- * An input color twist matrix with floating-point coefficient values is applied
  -- * within ROI.
  -- *
  -- * \param pSrcDst in place planar pixel format image pointer array, one pointer per plane.
  -- * \param nSrcDstStep in place planar pixel format image line step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aTwist The color twist matrix with floating-point coefficient values.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiColorTwist32f_16u_IP3R
     (pSrcDst : System.Address;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aTwist : System.Address) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:5040
   pragma Import (C, nppiColorTwist32f_16u_IP3R, "nppiColorTwist32f_16u_IP3R");

  --*
  -- * 1 channel 16-bit signed color twist.
  -- * 
  -- * An input color twist matrix with floating-point coefficient values is applied
  -- * within ROI.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aTwist The color twist matrix with floating-point coefficient values.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiColorTwist32f_16s_C1R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aTwist : System.Address) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:5058
   pragma Import (C, nppiColorTwist32f_16s_C1R, "nppiColorTwist32f_16s_C1R");

  --*
  -- * 1 channel 16-bit signed in place color twist.
  -- * 
  -- * An input color twist matrix with floating-point coefficient values is applied
  -- * within ROI.
  -- *
  -- * \param pSrcDst in place packed pixel format image pointer.
  -- * \param nSrcDstStep in place packed pixel format image line step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aTwist The color twist matrix with floating-point coefficient values.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiColorTwist32f_16s_C1IR
     (pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aTwist : System.Address) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:5074
   pragma Import (C, nppiColorTwist32f_16s_C1IR, "nppiColorTwist32f_16s_C1IR");

  --*
  -- * 2 channel 16-bit signed color twist.
  -- * 
  -- * An input color twist matrix with floating-point coefficient values is applied
  -- * within ROI.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aTwist The color twist matrix with floating-point coefficient values.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiColorTwist32f_16s_C2R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aTwist : System.Address) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:5092
   pragma Import (C, nppiColorTwist32f_16s_C2R, "nppiColorTwist32f_16s_C2R");

  --*
  -- * 2 channel 16-bit signed in place color twist.
  -- * 
  -- * An input color twist matrix with floating-point coefficient values is applied
  -- * within ROI.
  -- *
  -- * \param pSrcDst in place packed pixel format image pointer.
  -- * \param nSrcDstStep in place packed pixel format image line step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aTwist The color twist matrix with floating-point coefficient values.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiColorTwist32f_16s_C2IR
     (pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aTwist : System.Address) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:5108
   pragma Import (C, nppiColorTwist32f_16s_C2IR, "nppiColorTwist32f_16s_C2IR");

  --*
  -- * 3 channel 16-bit signed color twist.
  -- * 
  -- * An input color twist matrix with floating-point coefficient values is applied
  -- * within ROI.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aTwist The color twist matrix with floating-point coefficient values.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiColorTwist32f_16s_C3R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aTwist : System.Address) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:5126
   pragma Import (C, nppiColorTwist32f_16s_C3R, "nppiColorTwist32f_16s_C3R");

  --*
  -- * 3 channel 16-bit signed in place color twist.
  -- * 
  -- * An input color twist matrix with floating-point coefficient values is applied
  -- * within ROI.
  -- *
  -- * \param pSrcDst in place packed pixel format image pointer.
  -- * \param nSrcDstStep in place packed pixel format image line step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aTwist The color twist matrix with floating-point coefficient values.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiColorTwist32f_16s_C3IR
     (pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aTwist : System.Address) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:5142
   pragma Import (C, nppiColorTwist32f_16s_C3IR, "nppiColorTwist32f_16s_C3IR");

  --*
  -- * 4 channel 16-bit signed color twist, not affecting Alpha.
  -- *
  -- * An input color twist matrix with floating-point coefficient values is applied with
  -- * in ROI.
  -- * Alpha channel is the last channel and is not processed.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aTwist The color twist matrix with floating-point coefficient values.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiColorTwist32f_16s_AC4R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aTwist : System.Address) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:5161
   pragma Import (C, nppiColorTwist32f_16s_AC4R, "nppiColorTwist32f_16s_AC4R");

  --*
  -- * 4 channel 16-bit signed in place color twist, not affecting Alpha.
  -- *
  -- * An input color twist matrix with floating-point coefficient values is applied with
  -- * in ROI.
  -- * Alpha channel is the last channel and is not processed.
  -- *
  -- * \param pSrcDst in place packed pixel format image pointer.
  -- * \param nSrcDstStep in place packed pixel format image line step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aTwist The color twist matrix with floating-point coefficient values.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiColorTwist32f_16s_AC4IR
     (pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aTwist : System.Address) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:5178
   pragma Import (C, nppiColorTwist32f_16s_AC4IR, "nppiColorTwist32f_16s_AC4IR");

  --*
  -- * 3 channel 16-bit signed planar color twist.
  -- *
  -- * An input color twist matrix with floating-point coefficient values is applied
  -- * within ROI.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aTwist The color twist matrix with floating-point coefficient values.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiColorTwist32f_16s_P3R
     (pSrc : System.Address;
      nSrcStep : int;
      pDst : System.Address;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aTwist : System.Address) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:5196
   pragma Import (C, nppiColorTwist32f_16s_P3R, "nppiColorTwist32f_16s_P3R");

  --*
  -- * 3 channel 16-bit signed planar in place color twist.
  -- * 
  -- * An input color twist matrix with floating-point coefficient values is applied
  -- * within ROI.
  -- *
  -- * \param pSrcDst in place planar pixel format image pointer array, one pointer per plane.
  -- * \param nSrcDstStep in place planar pixel format image line step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aTwist The color twist matrix with floating-point coefficient values.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiColorTwist32f_16s_IP3R
     (pSrcDst : System.Address;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aTwist : System.Address) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:5212
   pragma Import (C, nppiColorTwist32f_16s_IP3R, "nppiColorTwist32f_16s_IP3R");

  --*
  -- * 1 channel 32-bit floating point color twist.
  -- * 
  -- * An input color twist matrix with floating-point coefficient values is applied
  -- * within ROI.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aTwist The color twist matrix with floating-point coefficient values.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiColorTwist_32f_C1R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aTwist : System.Address) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:5230
   pragma Import (C, nppiColorTwist_32f_C1R, "nppiColorTwist_32f_C1R");

  --*
  -- * 1 channel 32-bit floating point in place color twist.
  -- * 
  -- * An input color twist matrix with floating-point coefficient values is applied
  -- * within ROI.
  -- *
  -- * \param pSrcDst in place packed pixel format image pointer.
  -- * \param nSrcDstStep in place packed pixel format image line step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aTwist The color twist matrix with floating-point coefficient values.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiColorTwist_32f_C1IR
     (pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aTwist : System.Address) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:5246
   pragma Import (C, nppiColorTwist_32f_C1IR, "nppiColorTwist_32f_C1IR");

  --*
  -- * 2 channel 32-bit floating point color twist.
  -- * 
  -- * An input color twist matrix with floating-point coefficient values is applied
  -- * within ROI.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aTwist The color twist matrix with floating-point coefficient values.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiColorTwist_32f_C2R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aTwist : System.Address) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:5264
   pragma Import (C, nppiColorTwist_32f_C2R, "nppiColorTwist_32f_C2R");

  --*
  -- * 2 channel 32-bit floating point in place color twist.
  -- * 
  -- * An input color twist matrix with floating-point coefficient values is applied
  -- * within ROI.
  -- *
  -- * \param pSrcDst in place packed pixel format image pointer.
  -- * \param nSrcDstStep in place packed pixel format image line step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aTwist The color twist matrix with floating-point coefficient values.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiColorTwist_32f_C2IR
     (pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aTwist : System.Address) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:5280
   pragma Import (C, nppiColorTwist_32f_C2IR, "nppiColorTwist_32f_C2IR");

  --*
  -- * 3 channel 32-bit floating point color twist.
  -- * 
  -- * An input color twist matrix with floating-point coefficient values is applied
  -- * within ROI.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aTwist The color twist matrix with floating-point coefficient values.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiColorTwist_32f_C3R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aTwist : System.Address) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:5298
   pragma Import (C, nppiColorTwist_32f_C3R, "nppiColorTwist_32f_C3R");

  --*
  -- * 3 channel 32-bit floating point in place color twist.
  -- * 
  -- * An input color twist matrix with floating-point coefficient values is applied
  -- * within ROI.
  -- *
  -- * \param pSrcDst in place packed pixel format image pointer.
  -- * \param nSrcDstStep in place packed pixel format image line step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aTwist The color twist matrix with floating-point coefficient values.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiColorTwist_32f_C3IR
     (pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aTwist : System.Address) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:5314
   pragma Import (C, nppiColorTwist_32f_C3IR, "nppiColorTwist_32f_C3IR");

  --*
  -- * 4 channel 32-bit floating point color twist, with alpha copy.
  -- *
  -- * An input color twist matrix with floating-point coefficient values is applied with
  -- * in ROI.
  -- * Alpha channel is the last channel and is copied unmodified from the source pixel to the destination pixel.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aTwist The color twist matrix with floating-point coefficient values.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiColorTwist_32f_C4R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aTwist : System.Address) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:5333
   pragma Import (C, nppiColorTwist_32f_C4R, "nppiColorTwist_32f_C4R");

  --*
  -- * 4 channel 32-bit floating point in place color twist, not affecting Alpha.
  -- *
  -- * An input color twist matrix with floating-point coefficient values is applied with
  -- * in ROI.
  -- * Alpha channel is the last channel and is not modified.
  -- *
  -- * \param pSrcDst in place packed pixel format image pointer.
  -- * \param nSrcDstStep in place packed pixel format image line step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aTwist The color twist matrix with floating-point coefficient values.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiColorTwist_32f_C4IR
     (pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aTwist : System.Address) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:5350
   pragma Import (C, nppiColorTwist_32f_C4IR, "nppiColorTwist_32f_C4IR");

  --*
  -- * 4 channel 32-bit floating point color twist, not affecting Alpha.
  -- *
  -- * An input color twist matrix with floating-point coefficient values is applied with
  -- * in ROI.
  -- * Alpha channel is the last channel and is not processed.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aTwist The color twist matrix with floating-point coefficient values.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiColorTwist_32f_AC4R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aTwist : System.Address) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:5369
   pragma Import (C, nppiColorTwist_32f_AC4R, "nppiColorTwist_32f_AC4R");

  --*
  -- * 4 channel 32-bit floating point in place color twist, not affecting Alpha.
  -- *
  -- * An input color twist matrix with floating-point coefficient values is applied with
  -- * in ROI.
  -- * Alpha channel is the last channel and is not processed.
  -- *
  -- * \param pSrcDst in place packed pixel format image pointer.
  -- * \param nSrcDstStep in place packed pixel format image line step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aTwist The color twist matrix with floating-point coefficient values.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiColorTwist_32f_AC4IR
     (pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aTwist : System.Address) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:5386
   pragma Import (C, nppiColorTwist_32f_AC4IR, "nppiColorTwist_32f_AC4IR");

  --*
  -- * 4 channel 32-bit floating point color twist with 4x4 matrix and constant vector addition.
  -- *
  -- * An input 4x4 color twist matrix with floating-point coefficient values with an additional constant vector addition
  -- * is applied within ROI.  For this particular version of the function the result is generated as shown below.
  -- *
  -- *  \code
  -- *      dst[0] = aTwist[0][0] * src[0] + aTwist[0][1] * src[1] + aTwist[0][2] * src[2] + aTwist[0][3] * src[3] + aConstants[0]
  -- *      dst[1] = aTwist[1][0] * src[0] + aTwist[1][1] * src[1] + aTwist[1][2] * src[2] + aTwist[1][3] * src[3] + aConstants[1]
  -- *      dst[2] = aTwist[2][0] * src[0] + aTwist[2][1] * src[1] + aTwist[2][2] * src[2] + aTwist[2][3] * src[3] + aConstants[2]
  -- *      dst[3] = aTwist[3][0] * src[0] + aTwist[3][1] * src[1] + aTwist[3][2] * src[2] + aTwist[3][3] * src[3] + aConstants[3]
  -- *  \endcode
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aTwist The color twist matrix with floating-point coefficient values.
  -- * \param aConstants fixed size array of constant values, one per channel..
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiColorTwist_32fC_C4R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aTwist : System.Address;
      aConstants : access nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:5412
   pragma Import (C, nppiColorTwist_32fC_C4R, "nppiColorTwist_32fC_C4R");

  --*
  -- * 4 channel 32-bit floating point in place color twist with 4x4 matrix and an additional constant vector addition.
  -- *
  -- * An input 4x4 color twist matrix with floating-point coefficient values with an additional constant vector addition
  -- * is applied within ROI.  For this particular version of the function the result is generated as shown below.
  -- *
  -- *  \code
  -- *      dst[0] = aTwist[0][0] * src[0] + aTwist[0][1] * src[1] + aTwist[0][2] * src[2] + aTwist[0][3] * src[3] + aConstants[0]
  -- *      dst[1] = aTwist[1][0] * src[0] + aTwist[1][1] * src[1] + aTwist[1][2] * src[2] + aTwist[1][3] * src[3] + aConstants[1]
  -- *      dst[2] = aTwist[2][0] * src[0] + aTwist[2][1] * src[1] + aTwist[2][2] * src[2] + aTwist[2][3] * src[3] + aConstants[2]
  -- *      dst[3] = aTwist[3][0] * src[0] + aTwist[3][1] * src[1] + aTwist[3][2] * src[2] + aTwist[3][3] * src[3] + aConstants[3]
  -- *  \endcode
  -- *
  -- * \param pSrcDst in place packed pixel format image pointer.
  -- * \param nSrcDstStep in place packed pixel format image line step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aTwist The color twist matrix with floating-point coefficient values.
  -- * \param aConstants fixed size array of constant values, one per channel..
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiColorTwist_32fC_C4IR
     (pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aTwist : System.Address;
      aConstants : access nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:5436
   pragma Import (C, nppiColorTwist_32fC_C4IR, "nppiColorTwist_32fC_C4IR");

  --*
  -- * 3 channel 32-bit floating point planar color twist.
  -- *
  -- * An input color twist matrix with floating-point coefficient values is applied
  -- * within ROI.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aTwist The color twist matrix with floating-point coefficient values.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiColorTwist_32f_P3R
     (pSrc : System.Address;
      nSrcStep : int;
      pDst : System.Address;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aTwist : System.Address) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:5454
   pragma Import (C, nppiColorTwist_32f_P3R, "nppiColorTwist_32f_P3R");

  --*
  -- * 3 channel 32-bit floating point planar in place color twist.
  -- * 
  -- * An input color twist matrix with floating-point coefficient values is applied
  -- * within ROI.
  -- *
  -- * \param pSrcDst in place planar pixel format image pointer array, one pointer per plane.
  -- * \param nSrcDstStep in place planar pixel format image line step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aTwist The color twist matrix with floating-point coefficient values.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiColorTwist_32f_IP3R
     (pSrcDst : System.Address;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aTwist : System.Address) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:5470
   pragma Import (C, nppiColorTwist_32f_IP3R, "nppiColorTwist_32f_IP3R");

  --* @}  
  --* @name ColorLUT
  -- * 
  -- *  Perform image color processing using members of various types of color look up tables.
  -- * @{
  --  

  --*
  -- * 8-bit unsigned look-up-table color conversion.
  -- *
  -- * The LUT is derived from a set of user defined mapping points with no interpolation. 
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Pointer to an array of user defined OUTPUT values (this is a device memory pointer)
  -- * \param pLevels Pointer to an array of user defined INPUT values  (this is a device memory pointer)
  -- * \param nLevels Number of user defined number of input/output mapping points (levels)
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 256.
  --  

   function nppiLUT_8u_C1R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : access nppdefs_h.Npp32s;
      pLevels : access nppdefs_h.Npp32s;
      nLevels : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:5498
   pragma Import (C, nppiLUT_8u_C1R, "nppiLUT_8u_C1R");

  --*
  -- * 8-bit unsigned look-up-table in place color conversion.
  -- *
  -- * The LUT is derived from a set of user defined mapping points with no interpolation. 
  -- *
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Pointer to an array of user defined OUTPUT values (this is a device memory pointer)
  -- * \param pLevels Pointer to an array of user defined INPUT values  (this is a device memory pointer)
  -- * \param nLevels Number of user defined number of input/output mapping points (levels)
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 256.
  --  

   function nppiLUT_8u_C1IR
     (pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : access nppdefs_h.Npp32s;
      pLevels : access nppdefs_h.Npp32s;
      nLevels : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:5516
   pragma Import (C, nppiLUT_8u_C1IR, "nppiLUT_8u_C1IR");

  --*
  -- * 3 channel 8-bit unsigned look-up-table color conversion.
  -- * 
  -- * The LUT is derived from a set of user defined mapping points using no interpolation. 
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
  -- * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
  -- * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 256.
  --  

   function nppiLUT_8u_C3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : System.Address;
      pLevels : System.Address;
      nLevels : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:5536
   pragma Import (C, nppiLUT_8u_C3R, "nppiLUT_8u_C3R");

  --*
  -- * 3 channel 8-bit unsigned look-up-table in place color conversion.
  -- * 
  -- * The LUT is derived from a set of user defined mapping points with no interpolation. 
  -- *
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
  -- * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
  -- * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 256.
  --  

   function nppiLUT_8u_C3IR
     (pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : System.Address;
      pLevels : System.Address;
      nLevels : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:5554
   pragma Import (C, nppiLUT_8u_C3IR, "nppiLUT_8u_C3IR");

  --*
  -- * 4 channel 8-bit unsigned look-up-table color conversion.
  -- * 
  -- * The LUT is derived from a set of user defined mapping points with no interpolation. 
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
  -- * \param pLevels Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
  -- * \param nLevels Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 256.
  --  

   function nppiLUT_8u_C4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : System.Address;
      pLevels : System.Address;
      nLevels : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:5574
   pragma Import (C, nppiLUT_8u_C4R, "nppiLUT_8u_C4R");

  --*
  -- * 4 channel 8-bit unsigned look-up-table in place color conversion.
  -- * 
  -- * The LUT is derived from a set of user defined mapping points using no interpolation. 
  -- *
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
  -- * \param pLevels Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
  -- * \param nLevels Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 256.
  --  

   function nppiLUT_8u_C4IR
     (pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : System.Address;
      pLevels : System.Address;
      nLevels : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:5592
   pragma Import (C, nppiLUT_8u_C4IR, "nppiLUT_8u_C4IR");

  --*
  -- * 4 channel 8-bit unsigned look-up-table color conversion, not affecting Alpha.
  -- *
  -- * The LUT is derived from a set of user defined mapping points using no interpolation. 
  -- * Alpha channel is the last channel and is not processed.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
  -- * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
  -- * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 256.
  --  

   function nppiLUT_8u_AC4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : System.Address;
      pLevels : System.Address;
      nLevels : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:5613
   pragma Import (C, nppiLUT_8u_AC4R, "nppiLUT_8u_AC4R");

  --*
  -- * 4 channel 8-bit unsigned look-up-table in place color conversion, not affecting Alpha.
  -- *
  -- * The LUT is derived from a set of user defined mapping points using no interpolation. 
  -- * Alpha channel is the last channel and is not processed.
  -- *
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
  -- * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
  -- * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 256.
  --  

   function nppiLUT_8u_AC4IR
     (pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : System.Address;
      pLevels : System.Address;
      nLevels : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:5632
   pragma Import (C, nppiLUT_8u_AC4IR, "nppiLUT_8u_AC4IR");

  --*
  -- * 16-bit unsigned look-up-table color conversion.
  -- *
  -- * The LUT is derived from a set of user defined mapping points with no interpolation. 
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Pointer to an array of user defined OUTPUT values (this is a device memory pointer)
  -- * \param pLevels Pointer to an array of user defined INPUT values  (this is a device memory pointer)
  -- * \param nLevels Number of user defined number of input/output mapping points (levels)
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
  --  

   function nppiLUT_16u_C1R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : access nppdefs_h.Npp32s;
      pLevels : access nppdefs_h.Npp32s;
      nLevels : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:5653
   pragma Import (C, nppiLUT_16u_C1R, "nppiLUT_16u_C1R");

  --*
  -- * 16-bit unsigned look-up-table in place color conversion.
  -- *
  -- * The LUT is derived from a set of user defined mapping points with no interpolation. 
  -- *
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Pointer to an array of user defined OUTPUT values (this is a device memory pointer)
  -- * \param pLevels Pointer to an array of user defined INPUT values  (this is a device memory pointer)
  -- * \param nLevels Number of user defined number of input/output mapping points (levels)
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
  --  

   function nppiLUT_16u_C1IR
     (pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : access nppdefs_h.Npp32s;
      pLevels : access nppdefs_h.Npp32s;
      nLevels : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:5671
   pragma Import (C, nppiLUT_16u_C1IR, "nppiLUT_16u_C1IR");

  --*
  -- * 3 channel 16-bit unsigned look-up-table color conversion.
  -- * 
  -- * The LUT is derived from a set of user defined mapping points using no interpolation. 
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
  -- * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
  -- * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
  --  

   function nppiLUT_16u_C3R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : System.Address;
      pLevels : System.Address;
      nLevels : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:5691
   pragma Import (C, nppiLUT_16u_C3R, "nppiLUT_16u_C3R");

  --*
  -- * 3 channel 16-bit unsigned look-up-table in place color conversion.
  -- * 
  -- * The LUT is derived from a set of user defined mapping points with no interpolation. 
  -- *
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
  -- * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
  -- * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
  --  

   function nppiLUT_16u_C3IR
     (pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : System.Address;
      pLevels : System.Address;
      nLevels : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:5709
   pragma Import (C, nppiLUT_16u_C3IR, "nppiLUT_16u_C3IR");

  --*
  -- * 4 channel 16-bit unsigned look-up-table color conversion.
  -- * 
  -- * The LUT is derived from a set of user defined mapping points with no interpolation. 
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
  -- * \param pLevels Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
  -- * \param nLevels Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
  --  

   function nppiLUT_16u_C4R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : System.Address;
      pLevels : System.Address;
      nLevels : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:5729
   pragma Import (C, nppiLUT_16u_C4R, "nppiLUT_16u_C4R");

  --*
  -- * 4 channel 16-bit unsigned look-up-table in place color conversion.
  -- * 
  -- * The LUT is derived from a set of user defined mapping points using no interpolation. 
  -- *
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
  -- * \param pLevels Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
  -- * \param nLevels Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
  --  

   function nppiLUT_16u_C4IR
     (pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : System.Address;
      pLevels : System.Address;
      nLevels : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:5747
   pragma Import (C, nppiLUT_16u_C4IR, "nppiLUT_16u_C4IR");

  --*
  -- * 4 channel 16-bit unsigned look-up-table color conversion, not affecting Alpha.
  -- *
  -- * The LUT is derived from a set of user defined mapping points using no interpolation. 
  -- * Alpha channel is the last channel and is not processed.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
  -- * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
  -- * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
  --  

   function nppiLUT_16u_AC4R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : System.Address;
      pLevels : System.Address;
      nLevels : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:5768
   pragma Import (C, nppiLUT_16u_AC4R, "nppiLUT_16u_AC4R");

  --*
  -- * 4 channel 16-bit unsigned look-up-table in place color conversion, not affecting Alpha.
  -- *
  -- * The LUT is derived from a set of user defined mapping points using no interpolation. 
  -- * Alpha channel is the last channel and is not processed.
  -- *
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
  -- * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
  -- * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
  --  

   function nppiLUT_16u_AC4IR
     (pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : System.Address;
      pLevels : System.Address;
      nLevels : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:5787
   pragma Import (C, nppiLUT_16u_AC4IR, "nppiLUT_16u_AC4IR");

  --*
  -- * 16-bit signed look-up-table color conversion.
  -- *
  -- * The LUT is derived from a set of user defined mapping points with no interpolation. 
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Pointer to an array of user defined OUTPUT values (this is a device memory pointer)
  -- * \param pLevels Pointer to an array of user defined INPUT values  (this is a device memory pointer)
  -- * \param nLevels Number of user defined number of input/output mapping points (levels)
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
  --  

   function nppiLUT_16s_C1R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : access nppdefs_h.Npp32s;
      pLevels : access nppdefs_h.Npp32s;
      nLevels : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:5808
   pragma Import (C, nppiLUT_16s_C1R, "nppiLUT_16s_C1R");

  --*
  -- * 16-bit signed look-up-table in place color conversion.
  -- *
  -- * The LUT is derived from a set of user defined mapping points with no interpolation. 
  -- *
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Pointer to an array of user defined OUTPUT values (this is a device memory pointer)
  -- * \param pLevels Pointer to an array of user defined INPUT values  (this is a device memory pointer)
  -- * \param nLevels Number of user defined number of input/output mapping points (levels)
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
  --  

   function nppiLUT_16s_C1IR
     (pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : access nppdefs_h.Npp32s;
      pLevels : access nppdefs_h.Npp32s;
      nLevels : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:5826
   pragma Import (C, nppiLUT_16s_C1IR, "nppiLUT_16s_C1IR");

  --*
  -- * 3 channel 16-bit signed look-up-table color conversion.
  -- * 
  -- * The LUT is derived from a set of user defined mapping points using no interpolation. 
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
  -- * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
  -- * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
  --  

   function nppiLUT_16s_C3R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : System.Address;
      pLevels : System.Address;
      nLevels : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:5846
   pragma Import (C, nppiLUT_16s_C3R, "nppiLUT_16s_C3R");

  --*
  -- * 3 channel 16-bit signed look-up-table in place color conversion.
  -- * 
  -- * The LUT is derived from a set of user defined mapping points with no interpolation. 
  -- *
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
  -- * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
  -- * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
  --  

   function nppiLUT_16s_C3IR
     (pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : System.Address;
      pLevels : System.Address;
      nLevels : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:5864
   pragma Import (C, nppiLUT_16s_C3IR, "nppiLUT_16s_C3IR");

  --*
  -- * 4 channel 16-bit signed look-up-table color conversion.
  -- * 
  -- * The LUT is derived from a set of user defined mapping points with no interpolation. 
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
  -- * \param pLevels Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
  -- * \param nLevels Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
  --  

   function nppiLUT_16s_C4R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : System.Address;
      pLevels : System.Address;
      nLevels : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:5884
   pragma Import (C, nppiLUT_16s_C4R, "nppiLUT_16s_C4R");

  --*
  -- * 4 channel 16-bit signed look-up-table in place color conversion.
  -- * 
  -- * The LUT is derived from a set of user defined mapping points using no interpolation. 
  -- *
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
  -- * \param pLevels Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
  -- * \param nLevels Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
  --  

   function nppiLUT_16s_C4IR
     (pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : System.Address;
      pLevels : System.Address;
      nLevels : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:5902
   pragma Import (C, nppiLUT_16s_C4IR, "nppiLUT_16s_C4IR");

  --*
  -- * 4 channel 16-bit signed look-up-table color conversion, not affecting Alpha.
  -- *
  -- * The LUT is derived from a set of user defined mapping points using no interpolation. 
  -- * Alpha channel is the last channel and is not processed.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
  -- * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
  -- * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
  --  

   function nppiLUT_16s_AC4R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : System.Address;
      pLevels : System.Address;
      nLevels : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:5923
   pragma Import (C, nppiLUT_16s_AC4R, "nppiLUT_16s_AC4R");

  --*
  -- * 4 channel 16-bit signed look-up-table in place color conversion, not affecting Alpha.
  -- *
  -- * The LUT is derived from a set of user defined mapping points using no interpolation. 
  -- * Alpha channel is the last channel and is not processed.
  -- *
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
  -- * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
  -- * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
  --  

   function nppiLUT_16s_AC4IR
     (pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : System.Address;
      pLevels : System.Address;
      nLevels : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:5942
   pragma Import (C, nppiLUT_16s_AC4IR, "nppiLUT_16s_AC4IR");

  --*
  -- * 32-bit floating point look-up-table color conversion.
  -- *
  -- * The LUT is derived from a set of user defined mapping points with no interpolation. 
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Pointer to an array of user defined OUTPUT values (this is a device memory pointer)
  -- * \param pLevels Pointer to an array of user defined INPUT values  (this is a device memory pointer)
  -- * \param nLevels Number of user defined number of input/output mapping points (levels)
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
  --  

   function nppiLUT_32f_C1R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : access nppdefs_h.Npp32f;
      pLevels : access nppdefs_h.Npp32f;
      nLevels : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:5963
   pragma Import (C, nppiLUT_32f_C1R, "nppiLUT_32f_C1R");

  --*
  -- * 32-bit floating point look-up-table in place color conversion.
  -- *
  -- * The LUT is derived from a set of user defined mapping points with no interpolation. 
  -- *
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Pointer to an array of user defined OUTPUT values (this is a device memory pointer)
  -- * \param pLevels Pointer to an array of user defined INPUT values  (this is a device memory pointer)
  -- * \param nLevels Number of user defined number of input/output mapping points (levels)
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
  --  

   function nppiLUT_32f_C1IR
     (pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : access nppdefs_h.Npp32f;
      pLevels : access nppdefs_h.Npp32f;
      nLevels : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:5981
   pragma Import (C, nppiLUT_32f_C1IR, "nppiLUT_32f_C1IR");

  --*
  -- * 3 channel 32-bit floating point look-up-table color conversion.
  -- * 
  -- * The LUT is derived from a set of user defined mapping points using no interpolation. 
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
  -- * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
  -- * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
  --  

   function nppiLUT_32f_C3R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : System.Address;
      pLevels : System.Address;
      nLevels : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:6001
   pragma Import (C, nppiLUT_32f_C3R, "nppiLUT_32f_C3R");

  --*
  -- * 3 channel 32-bit floating point look-up-table in place color conversion.
  -- * 
  -- * The LUT is derived from a set of user defined mapping points with no interpolation. 
  -- *
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
  -- * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
  -- * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
  --  

   function nppiLUT_32f_C3IR
     (pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : System.Address;
      pLevels : System.Address;
      nLevels : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:6019
   pragma Import (C, nppiLUT_32f_C3IR, "nppiLUT_32f_C3IR");

  --*
  -- * 4 channel 32-bit floating point look-up-table color conversion.
  -- * 
  -- * The LUT is derived from a set of user defined mapping points with no interpolation. 
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
  -- * \param pLevels Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
  -- * \param nLevels Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
  --  

   function nppiLUT_32f_C4R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : System.Address;
      pLevels : System.Address;
      nLevels : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:6039
   pragma Import (C, nppiLUT_32f_C4R, "nppiLUT_32f_C4R");

  --*
  -- * 4 channel 32-bit floating point look-up-table in place color conversion.
  -- * 
  -- * The LUT is derived from a set of user defined mapping points using no interpolation. 
  -- *
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
  -- * \param pLevels Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
  -- * \param nLevels Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
  --  

   function nppiLUT_32f_C4IR
     (pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : System.Address;
      pLevels : System.Address;
      nLevels : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:6057
   pragma Import (C, nppiLUT_32f_C4IR, "nppiLUT_32f_C4IR");

  --*
  -- * 4 channel 32-bit floating point look-up-table color conversion, not affecting Alpha.
  -- *
  -- * The LUT is derived from a set of user defined mapping points using no interpolation. 
  -- * Alpha channel is the last channel and is not processed.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
  -- * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
  -- * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
  --  

   function nppiLUT_32f_AC4R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : System.Address;
      pLevels : System.Address;
      nLevels : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:6078
   pragma Import (C, nppiLUT_32f_AC4R, "nppiLUT_32f_AC4R");

  --*
  -- * 4 channel 32-bit floating point look-up-table in place color conversion, not affecting Alpha.
  -- *
  -- * The LUT is derived from a set of user defined mapping points using no interpolation. 
  -- * Alpha channel is the last channel and is not processed.
  -- *
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
  -- * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
  -- * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
  --  

   function nppiLUT_32f_AC4IR
     (pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : System.Address;
      pLevels : System.Address;
      nLevels : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:6097
   pragma Import (C, nppiLUT_32f_AC4IR, "nppiLUT_32f_AC4IR");

  --* @}  
  --* @name ColorLUT_Linear
  -- * 
  -- * Perform image color processing using linear interpolation between members of various types of color look up tables.
  -- * @{
  --  

  --*
  -- * 8-bit unsigned linear interpolated look-up-table color conversion.
  -- *
  -- * The LUT is derived from a set of user defined mapping points through linear interpolation. 
  -- *
  -- * >>>>>>> ATTENTION ATTENTION <<<<<<<
  -- *
  -- * NOTE: As of the 5.0 release of NPP, the pValues and pLevels pointers need to be device memory pointers.
  -- *
  -- * >>>>>>>                     <<<<<<<
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Pointer to an array of user defined OUTPUT values (this is now a device memory pointer)
  -- * \param pLevels Pointer to an array of user defined INPUT values  (this is now a device memory pointer)
  -- * \param nLevels Number of user defined number of input/output mapping points (levels)
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 256.
  --  

   function nppiLUT_Linear_8u_C1R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : access nppdefs_h.Npp32s;
      pLevels : access nppdefs_h.Npp32s;
      nLevels : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:6130
   pragma Import (C, nppiLUT_Linear_8u_C1R, "nppiLUT_Linear_8u_C1R");

  --*
  -- * 8-bit unsigned linear interpolated look-up-table in place color conversion.
  -- *
  -- * The LUT is derived from a set of user defined mapping points through linear interpolation. 
  -- *
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Pointer to an array of user defined OUTPUT values (this is a device memory pointer)
  -- * \param pLevels Pointer to an array of user defined INPUT values  (this is a device memory pointer)
  -- * \param nLevels Number of user defined number of input/output mapping points (levels)
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 256.
  --  

   function nppiLUT_Linear_8u_C1IR
     (pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : access nppdefs_h.Npp32s;
      pLevels : access nppdefs_h.Npp32s;
      nLevels : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:6148
   pragma Import (C, nppiLUT_Linear_8u_C1IR, "nppiLUT_Linear_8u_C1IR");

  --*
  -- * 3 channel 8-bit unsigned linear interpolated look-up-table color conversion.
  -- * 
  -- * The LUT is derived from a set of user defined mapping points through linear interpolation. 
  -- *
  -- * >>>>>>> ATTENTION ATTENTION <<<<<<<
  -- *
  -- * NOTE: As of the 5.0 release of NPP, the pValues and pLevels pointers need to be host memory pointers to arrays of device memory pointers.
  -- *
  -- * >>>>>>>                     <<<<<<<
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
  -- * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
  -- * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 256.
  --  

   function nppiLUT_Linear_8u_C3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : System.Address;
      pLevels : System.Address;
      nLevels : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:6174
   pragma Import (C, nppiLUT_Linear_8u_C3R, "nppiLUT_Linear_8u_C3R");

  --*
  -- * 3 channel 8-bit unsigned linear interpolated look-up-table in place color conversion.
  -- * 
  -- * The LUT is derived from a set of user defined mapping points through linear interpolation. 
  -- *
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
  -- * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
  -- * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 256.
  --  

   function nppiLUT_Linear_8u_C3IR
     (pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : System.Address;
      pLevels : System.Address;
      nLevels : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:6192
   pragma Import (C, nppiLUT_Linear_8u_C3IR, "nppiLUT_Linear_8u_C3IR");

  --*
  -- * 4 channel 8-bit unsigned linear interpolated look-up-table color conversion.
  -- * 
  -- * The LUT is derived from a set of user defined mapping points through linear interpolation. 
  -- *
  -- * >>>>>>> ATTENTION ATTENTION <<<<<<<
  -- *
  -- * NOTE: As of the 5.0 release of NPP, the pValues and pLevels pointers need to be host memory pointers to arrays of device memory pointers.
  -- *
  -- * >>>>>>>                     <<<<<<<
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
  -- * \param pLevels Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
  -- * \param nLevels Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 256.
  --  

   function nppiLUT_Linear_8u_C4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : System.Address;
      pLevels : System.Address;
      nLevels : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:6218
   pragma Import (C, nppiLUT_Linear_8u_C4R, "nppiLUT_Linear_8u_C4R");

  --*
  -- * 4 channel 8-bit unsigned linear interpolated look-up-table in place color conversion.
  -- * 
  -- * The LUT is derived from a set of user defined mapping points through linear interpolation. 
  -- *
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
  -- * \param pLevels Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
  -- * \param nLevels Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 256.
  --  

   function nppiLUT_Linear_8u_C4IR
     (pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : System.Address;
      pLevels : System.Address;
      nLevels : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:6236
   pragma Import (C, nppiLUT_Linear_8u_C4IR, "nppiLUT_Linear_8u_C4IR");

  --*
  -- * 4 channel 8-bit unsigned linear interpolated look-up-table color conversion, not affecting Alpha.
  -- *
  -- * The LUT is derived from a set of user defined mapping points through linear interpolation. 
  -- * Alpha channel is the last channel and is not processed.
  -- *
  -- * >>>>>>> ATTENTION ATTENTION <<<<<<<
  -- *
  -- * NOTE: As of the 5.0 release of NPP, the pValues and pLevels pointers need to be host memory pointers to arrays of device memory pointers.
  -- *
  -- * >>>>>>>                     <<<<<<<
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
  -- * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
  -- * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 256.
  --  

   function nppiLUT_Linear_8u_AC4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : System.Address;
      pLevels : System.Address;
      nLevels : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:6263
   pragma Import (C, nppiLUT_Linear_8u_AC4R, "nppiLUT_Linear_8u_AC4R");

  --*
  -- * 4 channel 8-bit unsigned linear interpolated look-up-table in place color conversion, not affecting Alpha.
  -- *
  -- * The LUT is derived from a set of user defined mapping points through linear interpolation. 
  -- * Alpha channel is the last channel and is not processed.
  -- *
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
  -- * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
  -- * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 256.
  --  

   function nppiLUT_Linear_8u_AC4IR
     (pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : System.Address;
      pLevels : System.Address;
      nLevels : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:6282
   pragma Import (C, nppiLUT_Linear_8u_AC4IR, "nppiLUT_Linear_8u_AC4IR");

  --*
  -- * 16-bit unsigned look-up-table color conversion.
  -- *
  -- * The LUT is derived from a set of user defined mapping points using linear interpolation. 
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Pointer to an array of user defined OUTPUT values (this is a device memory pointer)
  -- * \param pLevels Pointer to an array of user defined INPUT values  (this is a device memory pointer)
  -- * \param nLevels Number of user defined number of input/output mapping points (levels)
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
  --  

   function nppiLUT_Linear_16u_C1R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : access nppdefs_h.Npp32s;
      pLevels : access nppdefs_h.Npp32s;
      nLevels : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:6302
   pragma Import (C, nppiLUT_Linear_16u_C1R, "nppiLUT_Linear_16u_C1R");

  --*
  -- * 16-bit unsigned look-up-table in place color conversion.
  -- *
  -- * The LUT is derived from a set of user defined mapping points using linear interpolation. 
  -- *
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Pointer to an array of user defined OUTPUT values (this is a device memory pointer)
  -- * \param pLevels Pointer to an array of user defined INPUT values  (this is a device memory pointer)
  -- * \param nLevels Number of user defined number of input/output mapping points (levels)
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
  --  

   function nppiLUT_Linear_16u_C1IR
     (pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : access nppdefs_h.Npp32s;
      pLevels : access nppdefs_h.Npp32s;
      nLevels : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:6320
   pragma Import (C, nppiLUT_Linear_16u_C1IR, "nppiLUT_Linear_16u_C1IR");

  --*
  -- * 3 channel 16-bit unsigned look-up-table color conversion.
  -- * 
  -- * The LUT is derived from a set of user defined mapping points using no interpolation. 
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
  -- * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
  -- * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
  --  

   function nppiLUT_Linear_16u_C3R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : System.Address;
      pLevels : System.Address;
      nLevels : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:6340
   pragma Import (C, nppiLUT_Linear_16u_C3R, "nppiLUT_Linear_16u_C3R");

  --*
  -- * 3 channel 16-bit unsigned look-up-table in place color conversion.
  -- * 
  -- * The LUT is derived from a set of user defined mapping points using linear interpolation. 
  -- *
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
  -- * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
  -- * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
  --  

   function nppiLUT_Linear_16u_C3IR
     (pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : System.Address;
      pLevels : System.Address;
      nLevels : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:6358
   pragma Import (C, nppiLUT_Linear_16u_C3IR, "nppiLUT_Linear_16u_C3IR");

  --*
  -- * 4 channel 16-bit unsigned look-up-table color conversion.
  -- * 
  -- * The LUT is derived from a set of user defined mapping points using linear interpolation. 
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
  -- * \param pLevels Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
  -- * \param nLevels Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
  --  

   function nppiLUT_Linear_16u_C4R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : System.Address;
      pLevels : System.Address;
      nLevels : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:6378
   pragma Import (C, nppiLUT_Linear_16u_C4R, "nppiLUT_Linear_16u_C4R");

  --*
  -- * 4 channel 16-bit unsigned look-up-table in place color conversion.
  -- * 
  -- * The LUT is derived from a set of user defined mapping points using no interpolation. 
  -- *
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
  -- * \param pLevels Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
  -- * \param nLevels Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
  --  

   function nppiLUT_Linear_16u_C4IR
     (pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : System.Address;
      pLevels : System.Address;
      nLevels : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:6396
   pragma Import (C, nppiLUT_Linear_16u_C4IR, "nppiLUT_Linear_16u_C4IR");

  --*
  -- * 4 channel 16-bit unsigned look-up-table color conversion, not affecting Alpha.
  -- *
  -- * The LUT is derived from a set of user defined mapping points using no interpolation. 
  -- * Alpha channel is the last channel and is not processed.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
  -- * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
  -- * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
  --  

   function nppiLUT_Linear_16u_AC4R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : System.Address;
      pLevels : System.Address;
      nLevels : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:6417
   pragma Import (C, nppiLUT_Linear_16u_AC4R, "nppiLUT_Linear_16u_AC4R");

  --*
  -- * 4 channel 16-bit unsigned look-up-table in place color conversion, not affecting Alpha.
  -- *
  -- * The LUT is derived from a set of user defined mapping points using no interpolation. 
  -- * Alpha channel is the last channel and is not processed.
  -- *
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
  -- * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
  -- * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
  --  

   function nppiLUT_Linear_16u_AC4IR
     (pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : System.Address;
      pLevels : System.Address;
      nLevels : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:6436
   pragma Import (C, nppiLUT_Linear_16u_AC4IR, "nppiLUT_Linear_16u_AC4IR");

  --*
  -- * 16-bit signed look-up-table color conversion.
  -- *
  -- * The LUT is derived from a set of user defined mapping points using linear interpolation. 
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Pointer to an array of user defined OUTPUT values (this is a device memory pointer)
  -- * \param pLevels Pointer to an array of user defined INPUT values  (this is a device memory pointer)
  -- * \param nLevels Number of user defined number of input/output mapping points (levels)
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
  --  

   function nppiLUT_Linear_16s_C1R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : access nppdefs_h.Npp32s;
      pLevels : access nppdefs_h.Npp32s;
      nLevels : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:6457
   pragma Import (C, nppiLUT_Linear_16s_C1R, "nppiLUT_Linear_16s_C1R");

  --*
  -- * 16-bit signed look-up-table in place color conversion.
  -- *
  -- * The LUT is derived from a set of user defined mapping points using linear interpolation. 
  -- *
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Pointer to an array of user defined OUTPUT values (this is a device memory pointer)
  -- * \param pLevels Pointer to an array of user defined INPUT values  (this is a device memory pointer)
  -- * \param nLevels Number of user defined number of input/output mapping points (levels)
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
  --  

   function nppiLUT_Linear_16s_C1IR
     (pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : access nppdefs_h.Npp32s;
      pLevels : access nppdefs_h.Npp32s;
      nLevels : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:6475
   pragma Import (C, nppiLUT_Linear_16s_C1IR, "nppiLUT_Linear_16s_C1IR");

  --*
  -- * 3 channel 16-bit signed look-up-table color conversion.
  -- * 
  -- * The LUT is derived from a set of user defined mapping points using no interpolation. 
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
  -- * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
  -- * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
  --  

   function nppiLUT_Linear_16s_C3R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : System.Address;
      pLevels : System.Address;
      nLevels : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:6495
   pragma Import (C, nppiLUT_Linear_16s_C3R, "nppiLUT_Linear_16s_C3R");

  --*
  -- * 3 channel 16-bit signed look-up-table in place color conversion.
  -- * 
  -- * The LUT is derived from a set of user defined mapping points using linear interpolation. 
  -- *
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
  -- * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
  -- * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
  --  

   function nppiLUT_Linear_16s_C3IR
     (pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : System.Address;
      pLevels : System.Address;
      nLevels : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:6513
   pragma Import (C, nppiLUT_Linear_16s_C3IR, "nppiLUT_Linear_16s_C3IR");

  --*
  -- * 4 channel 16-bit signed look-up-table color conversion.
  -- * 
  -- * The LUT is derived from a set of user defined mapping points using linear interpolation. 
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
  -- * \param pLevels Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
  -- * \param nLevels Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
  --  

   function nppiLUT_Linear_16s_C4R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : System.Address;
      pLevels : System.Address;
      nLevels : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:6533
   pragma Import (C, nppiLUT_Linear_16s_C4R, "nppiLUT_Linear_16s_C4R");

  --*
  -- * 4 channel 16-bit signed look-up-table in place color conversion.
  -- * 
  -- * The LUT is derived from a set of user defined mapping points using no interpolation. 
  -- *
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
  -- * \param pLevels Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
  -- * \param nLevels Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
  --  

   function nppiLUT_Linear_16s_C4IR
     (pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : System.Address;
      pLevels : System.Address;
      nLevels : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:6551
   pragma Import (C, nppiLUT_Linear_16s_C4IR, "nppiLUT_Linear_16s_C4IR");

  --*
  -- * 4 channel 16-bit signed look-up-table color conversion, not affecting Alpha.
  -- *
  -- * The LUT is derived from a set of user defined mapping points using no interpolation. 
  -- * Alpha channel is the last channel and is not processed.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
  -- * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
  -- * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
  --  

   function nppiLUT_Linear_16s_AC4R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : System.Address;
      pLevels : System.Address;
      nLevels : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:6572
   pragma Import (C, nppiLUT_Linear_16s_AC4R, "nppiLUT_Linear_16s_AC4R");

  --*
  -- * 4 channel 16-bit signed look-up-table in place color conversion, not affecting Alpha.
  -- *
  -- * The LUT is derived from a set of user defined mapping points using no interpolation. 
  -- * Alpha channel is the last channel and is not processed.
  -- *
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
  -- * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
  -- * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
  --  

   function nppiLUT_Linear_16s_AC4IR
     (pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : System.Address;
      pLevels : System.Address;
      nLevels : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:6591
   pragma Import (C, nppiLUT_Linear_16s_AC4IR, "nppiLUT_Linear_16s_AC4IR");

  --*
  -- * 32-bit floating point look-up-table color conversion.
  -- *
  -- * The LUT is derived from a set of user defined mapping points using linear interpolation. 
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Pointer to an array of user defined OUTPUT values (this is a device memory pointer)
  -- * \param pLevels Pointer to an array of user defined INPUT values  (this is a device memory pointer)
  -- * \param nLevels Number of user defined number of input/output mapping points (levels)
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
  --  

   function nppiLUT_Linear_32f_C1R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : access nppdefs_h.Npp32f;
      pLevels : access nppdefs_h.Npp32f;
      nLevels : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:6612
   pragma Import (C, nppiLUT_Linear_32f_C1R, "nppiLUT_Linear_32f_C1R");

  --*
  -- * 32-bit floating point look-up-table in place color conversion.
  -- *
  -- * The LUT is derived from a set of user defined mapping points using linear interpolation. 
  -- *
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Pointer to an array of user defined OUTPUT values (this is a device memory pointer)
  -- * \param pLevels Pointer to an array of user defined INPUT values  (this is a device memory pointer)
  -- * \param nLevels Number of user defined number of input/output mapping points (levels)
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
  --  

   function nppiLUT_Linear_32f_C1IR
     (pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : access nppdefs_h.Npp32f;
      pLevels : access nppdefs_h.Npp32f;
      nLevels : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:6630
   pragma Import (C, nppiLUT_Linear_32f_C1IR, "nppiLUT_Linear_32f_C1IR");

  --*
  -- * 3 channel 32-bit floating point look-up-table color conversion.
  -- * 
  -- * The LUT is derived from a set of user defined mapping points using no interpolation. 
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
  -- * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
  -- * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
  --  

   function nppiLUT_Linear_32f_C3R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : System.Address;
      pLevels : System.Address;
      nLevels : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:6650
   pragma Import (C, nppiLUT_Linear_32f_C3R, "nppiLUT_Linear_32f_C3R");

  --*
  -- * 3 channel 32-bit floating point look-up-table in place color conversion.
  -- * 
  -- * The LUT is derived from a set of user defined mapping points using linear interpolation. 
  -- *
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
  -- * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
  -- * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
  --  

   function nppiLUT_Linear_32f_C3IR
     (pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : System.Address;
      pLevels : System.Address;
      nLevels : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:6668
   pragma Import (C, nppiLUT_Linear_32f_C3IR, "nppiLUT_Linear_32f_C3IR");

  --*
  -- * 4 channel 32-bit floating point look-up-table color conversion.
  -- * 
  -- * The LUT is derived from a set of user defined mapping points using linear interpolation. 
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
  -- * \param pLevels Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
  -- * \param nLevels Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
  --  

   function nppiLUT_Linear_32f_C4R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : System.Address;
      pLevels : System.Address;
      nLevels : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:6688
   pragma Import (C, nppiLUT_Linear_32f_C4R, "nppiLUT_Linear_32f_C4R");

  --*
  -- * 4 channel 32-bit floating point look-up-table in place color conversion.
  -- * 
  -- * The LUT is derived from a set of user defined mapping points using no interpolation. 
  -- *
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
  -- * \param pLevels Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
  -- * \param nLevels Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
  --  

   function nppiLUT_Linear_32f_C4IR
     (pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : System.Address;
      pLevels : System.Address;
      nLevels : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:6706
   pragma Import (C, nppiLUT_Linear_32f_C4IR, "nppiLUT_Linear_32f_C4IR");

  --*
  -- * 4 channel 32-bit floating point look-up-table color conversion, not affecting Alpha.
  -- *
  -- * The LUT is derived from a set of user defined mapping points using no interpolation. 
  -- * Alpha channel is the last channel and is not processed.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
  -- * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
  -- * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
  --  

   function nppiLUT_Linear_32f_AC4R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : System.Address;
      pLevels : System.Address;
      nLevels : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:6727
   pragma Import (C, nppiLUT_Linear_32f_AC4R, "nppiLUT_Linear_32f_AC4R");

  --*
  -- * 4 channel 32-bit floating point look-up-table in place color conversion, not affecting Alpha.
  -- *
  -- * The LUT is derived from a set of user defined mapping points using no interpolation. 
  -- * Alpha channel is the last channel and is not processed.
  -- *
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
  -- * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
  -- * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
  --  

   function nppiLUT_Linear_32f_AC4IR
     (pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : System.Address;
      pLevels : System.Address;
      nLevels : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:6746
   pragma Import (C, nppiLUT_Linear_32f_AC4IR, "nppiLUT_Linear_32f_AC4IR");

  --* @}  
  --* @name ColorLUT_Cubic
  -- * 
  -- *  Perform image color processing using linear interpolation between members of various types of color look up tables.
  -- * @{
  --  

  --*
  -- * 8-bit unsigned cubic interpolated look-up-table color conversion.
  -- *
  -- * The LUT is derived from a set of user defined mapping points through cubic interpolation. 
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Pointer to an array of user defined OUTPUT values (this is a device memory pointer)
  -- * \param pLevels Pointer to an array of user defined INPUT values  (this is a device memory pointer)
  -- * \param nLevels Number of user defined number of input/output mapping points (levels)
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 256.
  --  

   function nppiLUT_Cubic_8u_C1R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : access nppdefs_h.Npp32s;
      pLevels : access nppdefs_h.Npp32s;
      nLevels : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:6773
   pragma Import (C, nppiLUT_Cubic_8u_C1R, "nppiLUT_Cubic_8u_C1R");

  --*
  -- * 8-bit unsigned cubic interpolated look-up-table in place color conversion.
  -- *
  -- * The LUT is derived from a set of user defined mapping points through cubic interpolation. 
  -- *
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Pointer to an array of user defined OUTPUT values (this is a device memory pointer)
  -- * \param pLevels Pointer to an array of user defined INPUT values  (this is a device memory pointer)
  -- * \param nLevels Number of user defined number of input/output mapping points (levels)
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 256.
  --  

   function nppiLUT_Cubic_8u_C1IR
     (pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : access nppdefs_h.Npp32s;
      pLevels : access nppdefs_h.Npp32s;
      nLevels : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:6791
   pragma Import (C, nppiLUT_Cubic_8u_C1IR, "nppiLUT_Cubic_8u_C1IR");

  --*
  -- * 3 channel 8-bit unsigned cubic interpolated look-up-table color conversion.
  -- * 
  -- * The LUT is derived from a set of user defined mapping points through cubic interpolation. 
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
  -- * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
  -- * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 256.
  --  

   function nppiLUT_Cubic_8u_C3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : System.Address;
      pLevels : System.Address;
      nLevels : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:6811
   pragma Import (C, nppiLUT_Cubic_8u_C3R, "nppiLUT_Cubic_8u_C3R");

  --*
  -- * 3 channel 8-bit unsigned cubic interpolated look-up-table in place color conversion.
  -- * 
  -- * The LUT is derived from a set of user defined mapping points through cubic interpolation. 
  -- *
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
  -- * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
  -- * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 256.
  --  

   function nppiLUT_Cubic_8u_C3IR
     (pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : System.Address;
      pLevels : System.Address;
      nLevels : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:6829
   pragma Import (C, nppiLUT_Cubic_8u_C3IR, "nppiLUT_Cubic_8u_C3IR");

  --*
  -- * 4 channel 8-bit unsigned cubic interpolated look-up-table color conversion.
  -- * 
  -- * The LUT is derived from a set of user defined mapping points through cubic interpolation. 
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
  -- * \param pLevels Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
  -- * \param nLevels Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 256.
  --  

   function nppiLUT_Cubic_8u_C4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : System.Address;
      pLevels : System.Address;
      nLevels : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:6849
   pragma Import (C, nppiLUT_Cubic_8u_C4R, "nppiLUT_Cubic_8u_C4R");

  --*
  -- * 4 channel 8-bit unsigned cubic interpolated look-up-table in place color conversion.
  -- * 
  -- * The LUT is derived from a set of user defined mapping points through cubic interpolation. 
  -- *
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
  -- * \param pLevels Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
  -- * \param nLevels Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 256.
  --  

   function nppiLUT_Cubic_8u_C4IR
     (pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : System.Address;
      pLevels : System.Address;
      nLevels : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:6867
   pragma Import (C, nppiLUT_Cubic_8u_C4IR, "nppiLUT_Cubic_8u_C4IR");

  --*
  -- * 4 channel 8-bit unsigned cubic interpolated look-up-table color conversion, not affecting Alpha.
  -- *
  -- * The LUT is derived from a set of user defined mapping points through cubic interpolation. 
  -- * Alpha channel is the last channel and is not processed.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
  -- * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
  -- * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 256.
  --  

   function nppiLUT_Cubic_8u_AC4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : System.Address;
      pLevels : System.Address;
      nLevels : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:6888
   pragma Import (C, nppiLUT_Cubic_8u_AC4R, "nppiLUT_Cubic_8u_AC4R");

  --*
  -- * 4 channel 8-bit unsigned cubic interpolated look-up-table in place color conversion, not affecting Alpha.
  -- *
  -- * The LUT is derived from a set of user defined mapping points through cubic interpolation. 
  -- * Alpha channel is the last channel and is not processed.
  -- *
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
  -- * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
  -- * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 256.
  --  

   function nppiLUT_Cubic_8u_AC4IR
     (pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : System.Address;
      pLevels : System.Address;
      nLevels : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:6907
   pragma Import (C, nppiLUT_Cubic_8u_AC4IR, "nppiLUT_Cubic_8u_AC4IR");

  --*
  -- * 16-bit unsigned look-up-table color conversion.
  -- *
  -- * The LUT is derived from a set of user defined mapping points through cubic interpolation. 
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Pointer to an array of user defined OUTPUT values (this is a device memory pointer)
  -- * \param pLevels Pointer to an array of user defined INPUT values  (this is a device memory pointer)
  -- * \param nLevels Number of user defined number of input/output mapping points (levels)
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
  --  

   function nppiLUT_Cubic_16u_C1R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : access nppdefs_h.Npp32s;
      pLevels : access nppdefs_h.Npp32s;
      nLevels : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:6927
   pragma Import (C, nppiLUT_Cubic_16u_C1R, "nppiLUT_Cubic_16u_C1R");

  --*
  -- * 16-bit unsigned look-up-table in place color conversion.
  -- *
  -- * The LUT is derived from a set of user defined mapping points through cubic interpolation. 
  -- *
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Pointer to an array of user defined OUTPUT values (this is a device memory pointer)
  -- * \param pLevels Pointer to an array of user defined INPUT values  (this is a device memory pointer)
  -- * \param nLevels Number of user defined number of input/output mapping points (levels)
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
  --  

   function nppiLUT_Cubic_16u_C1IR
     (pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : access nppdefs_h.Npp32s;
      pLevels : access nppdefs_h.Npp32s;
      nLevels : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:6945
   pragma Import (C, nppiLUT_Cubic_16u_C1IR, "nppiLUT_Cubic_16u_C1IR");

  --*
  -- * 3 channel 16-bit unsigned look-up-table color conversion.
  -- * 
  -- * The LUT is derived from a set of user defined mapping points using no interpolation. 
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
  -- * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
  -- * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
  --  

   function nppiLUT_Cubic_16u_C3R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : System.Address;
      pLevels : System.Address;
      nLevels : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:6965
   pragma Import (C, nppiLUT_Cubic_16u_C3R, "nppiLUT_Cubic_16u_C3R");

  --*
  -- * 3 channel 16-bit unsigned look-up-table in place color conversion.
  -- * 
  -- * The LUT is derived from a set of user defined mapping points through cubic interpolation. 
  -- *
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
  -- * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
  -- * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
  --  

   function nppiLUT_Cubic_16u_C3IR
     (pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : System.Address;
      pLevels : System.Address;
      nLevels : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:6983
   pragma Import (C, nppiLUT_Cubic_16u_C3IR, "nppiLUT_Cubic_16u_C3IR");

  --*
  -- * 4 channel 16-bit unsigned look-up-table color conversion.
  -- * 
  -- * The LUT is derived from a set of user defined mapping points through cubic interpolation. 
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
  -- * \param pLevels Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
  -- * \param nLevels Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
  --  

   function nppiLUT_Cubic_16u_C4R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : System.Address;
      pLevels : System.Address;
      nLevels : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:7003
   pragma Import (C, nppiLUT_Cubic_16u_C4R, "nppiLUT_Cubic_16u_C4R");

  --*
  -- * 4 channel 16-bit unsigned look-up-table in place color conversion.
  -- * 
  -- * The LUT is derived from a set of user defined mapping points using no interpolation. 
  -- *
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
  -- * \param pLevels Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
  -- * \param nLevels Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
  --  

   function nppiLUT_Cubic_16u_C4IR
     (pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : System.Address;
      pLevels : System.Address;
      nLevels : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:7021
   pragma Import (C, nppiLUT_Cubic_16u_C4IR, "nppiLUT_Cubic_16u_C4IR");

  --*
  -- * 4 channel 16-bit unsigned look-up-table color conversion, not affecting Alpha.
  -- *
  -- * The LUT is derived from a set of user defined mapping points using no interpolation. 
  -- * Alpha channel is the last channel and is not processed.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
  -- * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
  -- * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
  --  

   function nppiLUT_Cubic_16u_AC4R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : System.Address;
      pLevels : System.Address;
      nLevels : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:7042
   pragma Import (C, nppiLUT_Cubic_16u_AC4R, "nppiLUT_Cubic_16u_AC4R");

  --*
  -- * 4 channel 16-bit unsigned look-up-table in place color conversion, not affecting Alpha.
  -- *
  -- * The LUT is derived from a set of user defined mapping points using no interpolation. 
  -- * Alpha channel is the last channel and is not processed.
  -- *
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
  -- * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
  -- * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
  --  

   function nppiLUT_Cubic_16u_AC4IR
     (pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : System.Address;
      pLevels : System.Address;
      nLevels : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:7061
   pragma Import (C, nppiLUT_Cubic_16u_AC4IR, "nppiLUT_Cubic_16u_AC4IR");

  --*
  -- * 16-bit signed look-up-table color conversion.
  -- *
  -- * The LUT is derived from a set of user defined mapping points through cubic interpolation. 
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Pointer to an array of user defined OUTPUT values (this is a device memory pointer)
  -- * \param pLevels Pointer to an array of user defined INPUT values  (this is a device memory pointer)
  -- * \param nLevels Number of user defined number of input/output mapping points (levels)
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
  --  

   function nppiLUT_Cubic_16s_C1R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : access nppdefs_h.Npp32s;
      pLevels : access nppdefs_h.Npp32s;
      nLevels : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:7082
   pragma Import (C, nppiLUT_Cubic_16s_C1R, "nppiLUT_Cubic_16s_C1R");

  --*
  -- * 16-bit signed look-up-table in place color conversion.
  -- *
  -- * The LUT is derived from a set of user defined mapping points through cubic interpolation. 
  -- *
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Pointer to an array of user defined OUTPUT values (this is a device memory pointer)
  -- * \param pLevels Pointer to an array of user defined INPUT values  (this is a device memory pointer)
  -- * \param nLevels Number of user defined number of input/output mapping points (levels)
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
  --  

   function nppiLUT_Cubic_16s_C1IR
     (pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : access nppdefs_h.Npp32s;
      pLevels : access nppdefs_h.Npp32s;
      nLevels : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:7100
   pragma Import (C, nppiLUT_Cubic_16s_C1IR, "nppiLUT_Cubic_16s_C1IR");

  --*
  -- * 3 channel 16-bit signed look-up-table color conversion.
  -- * 
  -- * The LUT is derived from a set of user defined mapping points using no interpolation. 
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
  -- * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
  -- * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
  --  

   function nppiLUT_Cubic_16s_C3R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : System.Address;
      pLevels : System.Address;
      nLevels : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:7120
   pragma Import (C, nppiLUT_Cubic_16s_C3R, "nppiLUT_Cubic_16s_C3R");

  --*
  -- * 3 channel 16-bit signed look-up-table in place color conversion.
  -- * 
  -- * The LUT is derived from a set of user defined mapping points through cubic interpolation. 
  -- *
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
  -- * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
  -- * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
  --  

   function nppiLUT_Cubic_16s_C3IR
     (pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : System.Address;
      pLevels : System.Address;
      nLevels : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:7138
   pragma Import (C, nppiLUT_Cubic_16s_C3IR, "nppiLUT_Cubic_16s_C3IR");

  --*
  -- * 4 channel 16-bit signed look-up-table color conversion.
  -- * 
  -- * The LUT is derived from a set of user defined mapping points through cubic interpolation. 
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
  -- * \param pLevels Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
  -- * \param nLevels Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
  --  

   function nppiLUT_Cubic_16s_C4R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : System.Address;
      pLevels : System.Address;
      nLevels : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:7158
   pragma Import (C, nppiLUT_Cubic_16s_C4R, "nppiLUT_Cubic_16s_C4R");

  --*
  -- * 4 channel 16-bit signed look-up-table in place color conversion.
  -- * 
  -- * The LUT is derived from a set of user defined mapping points using no interpolation. 
  -- *
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
  -- * \param pLevels Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
  -- * \param nLevels Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
  --  

   function nppiLUT_Cubic_16s_C4IR
     (pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : System.Address;
      pLevels : System.Address;
      nLevels : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:7176
   pragma Import (C, nppiLUT_Cubic_16s_C4IR, "nppiLUT_Cubic_16s_C4IR");

  --*
  -- * 4 channel 16-bit signed look-up-table color conversion, not affecting Alpha.
  -- *
  -- * The LUT is derived from a set of user defined mapping points using no interpolation. 
  -- * Alpha channel is the last channel and is not processed.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
  -- * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
  -- * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
  --  

   function nppiLUT_Cubic_16s_AC4R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : System.Address;
      pLevels : System.Address;
      nLevels : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:7197
   pragma Import (C, nppiLUT_Cubic_16s_AC4R, "nppiLUT_Cubic_16s_AC4R");

  --*
  -- * 4 channel 16-bit signed look-up-table in place color conversion, not affecting Alpha.
  -- *
  -- * The LUT is derived from a set of user defined mapping points using no interpolation. 
  -- * Alpha channel is the last channel and is not processed.
  -- *
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
  -- * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
  -- * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
  --  

   function nppiLUT_Cubic_16s_AC4IR
     (pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : System.Address;
      pLevels : System.Address;
      nLevels : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:7216
   pragma Import (C, nppiLUT_Cubic_16s_AC4IR, "nppiLUT_Cubic_16s_AC4IR");

  --*
  -- * 32-bit floating point look-up-table color conversion.
  -- *
  -- * The LUT is derived from a set of user defined mapping points through cubic interpolation. 
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Pointer to an array of user defined OUTPUT values (this is a device memory pointer)
  -- * \param pLevels Pointer to an array of user defined INPUT values  (this is a device memory pointer)
  -- * \param nLevels Number of user defined number of input/output mapping points (levels)
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
  --  

   function nppiLUT_Cubic_32f_C1R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : access nppdefs_h.Npp32f;
      pLevels : access nppdefs_h.Npp32f;
      nLevels : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:7237
   pragma Import (C, nppiLUT_Cubic_32f_C1R, "nppiLUT_Cubic_32f_C1R");

  --*
  -- * 32-bit floating point look-up-table in place color conversion.
  -- *
  -- * The LUT is derived from a set of user defined mapping points through cubic interpolation. 
  -- *
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Pointer to an array of user defined OUTPUT values (this is a device memory pointer)
  -- * \param pLevels Pointer to an array of user defined INPUT values  (this is a device memory pointer)
  -- * \param nLevels Number of user defined number of input/output mapping points (levels)
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
  --  

   function nppiLUT_Cubic_32f_C1IR
     (pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : access nppdefs_h.Npp32f;
      pLevels : access nppdefs_h.Npp32f;
      nLevels : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:7255
   pragma Import (C, nppiLUT_Cubic_32f_C1IR, "nppiLUT_Cubic_32f_C1IR");

  --*
  -- * 3 channel 32-bit floating point look-up-table color conversion.
  -- * 
  -- * The LUT is derived from a set of user defined mapping points using no interpolation. 
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
  -- * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
  -- * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
  --  

   function nppiLUT_Cubic_32f_C3R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : System.Address;
      pLevels : System.Address;
      nLevels : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:7275
   pragma Import (C, nppiLUT_Cubic_32f_C3R, "nppiLUT_Cubic_32f_C3R");

  --*
  -- * 3 channel 32-bit floating point look-up-table in place color conversion.
  -- * 
  -- * The LUT is derived from a set of user defined mapping points through cubic interpolation. 
  -- *
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
  -- * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
  -- * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
  --  

   function nppiLUT_Cubic_32f_C3IR
     (pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : System.Address;
      pLevels : System.Address;
      nLevels : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:7293
   pragma Import (C, nppiLUT_Cubic_32f_C3IR, "nppiLUT_Cubic_32f_C3IR");

  --*
  -- * 4 channel 32-bit floating point look-up-table color conversion.
  -- * 
  -- * The LUT is derived from a set of user defined mapping points through cubic interpolation. 
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
  -- * \param pLevels Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
  -- * \param nLevels Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
  --  

   function nppiLUT_Cubic_32f_C4R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : System.Address;
      pLevels : System.Address;
      nLevels : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:7313
   pragma Import (C, nppiLUT_Cubic_32f_C4R, "nppiLUT_Cubic_32f_C4R");

  --*
  -- * 4 channel 32-bit floating point look-up-table in place color conversion.
  -- * 
  -- * The LUT is derived from a set of user defined mapping points using no interpolation. 
  -- *
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
  -- * \param pLevels Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
  -- * \param nLevels Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
  --  

   function nppiLUT_Cubic_32f_C4IR
     (pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : System.Address;
      pLevels : System.Address;
      nLevels : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:7331
   pragma Import (C, nppiLUT_Cubic_32f_C4IR, "nppiLUT_Cubic_32f_C4IR");

  --*
  -- * 4 channel 32-bit floating point look-up-table color conversion, not affecting Alpha.
  -- *
  -- * The LUT is derived from a set of user defined mapping points using no interpolation. 
  -- * Alpha channel is the last channel and is not processed.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
  -- * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
  -- * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
  --  

   function nppiLUT_Cubic_32f_AC4R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : System.Address;
      pLevels : System.Address;
      nLevels : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:7352
   pragma Import (C, nppiLUT_Cubic_32f_AC4R, "nppiLUT_Cubic_32f_AC4R");

  --*
  -- * 4 channel 32-bit floating point look-up-table in place color conversion, not affecting Alpha.
  -- *
  -- * The LUT is derived from a set of user defined mapping points using no interpolation. 
  -- * Alpha channel is the last channel and is not processed.
  -- *
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
  -- * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
  -- * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
  --  

   function nppiLUT_Cubic_32f_AC4IR
     (pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : System.Address;
      pLevels : System.Address;
      nLevels : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:7371
   pragma Import (C, nppiLUT_Cubic_32f_AC4IR, "nppiLUT_Cubic_32f_AC4IR");

  --* @}  
  --* @name ColorLUT_Trilinear
  -- * 
  -- *  Perform image color processing using 3D trilinear interpolation between members of various types of color look up tables.
  -- * @{
  --  

  --*
  -- * Four channel 8-bit unsigned 3D trilinear interpolated look-up-table color conversion, with alpha copy.
  -- * Alpha channel is the last channel and is copied to the destination unmodified.
  -- *
  -- * The LUT is derived from a set of user defined mapping points through trilinear interpolation. 
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Device pointer to aLevels[2] number of contiguous 2D x,y planes of 4-byte packed RGBX values
  -- *        containing the user defined base OUTPUT values at that x,y, and z (R,G,B) level location. Each level must contain x * y 4-byte
  -- *        packed pixel values (4th byte is used for alignement only and is ignored) in row (x) order.
  -- * \param pLevels Host pointer to an array of 3 host pointers, one per cube edge, pointing to user defined INPUT level values.
  -- * \param aLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per 3D cube edge.
  -- *        aLevels[0] represents the number of x axis levels (Red), aLevels[1] represents the number of y axis levels (Green), 
  -- *        and aLevels[2] represets the number of z axis levels (Blue).
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 256.
  --  

   function nppiLUT_Trilinear_8u_C4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : access nppdefs_h.Npp32u;
      pLevels : System.Address;
      aLevels : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:7404
   pragma Import (C, nppiLUT_Trilinear_8u_C4R, "nppiLUT_Trilinear_8u_C4R");

  --*
  -- * Four channel 8-bit unsigned 3D trilinear interpolated look-up-table color conversion, not affecting alpha.
  -- * Alpha channel is the last channel and is not processed.
  -- *
  -- * The LUT is derived from a set of user defined mapping points through trilinear interpolation. 
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Device pointer to aLevels[2] number of contiguous 2D x,y planes of 4-byte packed RGBX values
  -- *        containing the user defined base OUTPUT values at that x,y, and z (R,G,B) level location. Each level must contain x * y 4-byte
  -- *        packed pixel values (4th byte is used for alignement only and is ignored) in row (x) order.
  -- * \param pLevels Host pointer to an array of 3 host pointers, one per cube edge, pointing to user defined INPUT level values.
  -- * \param aLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per 3D cube edge.
  -- *        aLevels[0] represents the number of x axis levels (Red), aLevels[1] represents the number of y axis levels (Green), 
  -- *        and aLevels[2] represets the number of z axis levels (Blue).
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 256.
  --  

   function nppiLUT_Trilinear_8u_AC4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : access nppdefs_h.Npp32u;
      pLevels : System.Address;
      aLevels : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:7429
   pragma Import (C, nppiLUT_Trilinear_8u_AC4R, "nppiLUT_Trilinear_8u_AC4R");

  --*
  -- * Four channel 8-bit unsigned 3D trilinear interpolated look-up-table in place color conversion, not affecting alpha.
  -- * Alpha channel is the last channel and is not processed.
  -- *
  -- * The LUT is derived from a set of user defined mapping points through trilinear interpolation. 
  -- *
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pValues Device pointer aLevels[2] number of contiguous 2D x,y planes of 4-byte packed RGBX values
  -- *        containing the user defined base OUTPUT values at that x,y, and z (R,G,B) level location. Each level must contain x * y 4-byte
  -- *        packed pixel values (4th byte is used for alignement only and is ignored) in row (x) order.
  -- * \param pLevels Host pointer to an array of 3 host pointers, one per cube edge, pointing to user defined INPUT level values.
  -- * \param aLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per 3D cube edge.
  -- *        aLevels[0] represents the number of x axis levels (Red), aLevels[1] represents the number of y axis levels (Green), 
  -- *        and aLevels[2] represets the number of z axis levels (Blue).
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 256.
  --  

   function nppiLUT_Trilinear_8u_AC4IR
     (pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pValues : access nppdefs_h.Npp32u;
      pLevels : System.Address;
      aLevels : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:7452
   pragma Import (C, nppiLUT_Trilinear_8u_AC4IR, "nppiLUT_Trilinear_8u_AC4IR");

  --* @}  
  --* @name ColorLUTPalette
  -- * 
  -- *  Perform image color processing using various types of bit range restricted palette color look up tables.
  -- * @{
  --  

  --*
  -- * One channel 8-bit unsigned bit range restricted palette look-up-table color conversion.
  -- *
  -- * The LUT is derived from a set of user defined mapping points in a palette and 
  -- * source pixels are then processed using a restricted bit range when looking up palette values.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pTable Pointer to an array of user defined OUTPUT palette values (this is a device memory pointer)
  -- * \param nBitSize Number of least significant bits (must be > 0 and <= 8) of each source pixel value to use as index into palette table during conversion.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_PALETTE_BITSIZE_ERROR if nBitSize is < 1 or > 8.
  --  

   function nppiLUTPalette_8u_C1R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pTable : access nppdefs_h.Npp8u;
      nBitSize : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:7480
   pragma Import (C, nppiLUTPalette_8u_C1R, "nppiLUTPalette_8u_C1R");

  --*
  -- * One channel 8-bit unsigned bit range restricted 24-bit palette look-up-table color conversion with 24-bit destination output per pixel.
  -- *
  -- * The LUT is derived from a set of user defined mapping points in a palette and 
  -- * source pixels are then processed using a restricted bit range when looking up palette values.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step (3 bytes per pixel).
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pTable Pointer to an array of user defined OUTPUT palette values (this is a device memory pointer)
  -- * \param nBitSize Number of least significant bits (must be > 0 and <= 8) of each source pixel value to use as index into palette table during conversion.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_PALETTE_BITSIZE_ERROR if nBitSize is < 1 or > 8.
  --  

   function nppiLUTPalette_8u24u_C1R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pTable : access nppdefs_h.Npp8u;
      nBitSize : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:7500
   pragma Import (C, nppiLUTPalette_8u24u_C1R, "nppiLUTPalette_8u24u_C1R");

  --*
  -- * One channel 8-bit unsigned bit range restricted 32-bit palette look-up-table color conversion with 32-bit destination output per pixel.
  -- *
  -- * The LUT is derived from a set of user defined mapping points in a palette and 
  -- * source pixels are then processed using a restricted bit range when looking up palette values.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step (4 bytes per pixel).
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pTable Pointer to an array of user defined OUTPUT palette values (this is a device memory pointer)
  -- * \param nBitSize Number of least significant bits (must be > 0 and <= 8) of each source pixel value to use as index into palette table during conversion.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_PALETTE_BITSIZE_ERROR if nBitSize is < 1 or > 8.
  --  

   function nppiLUTPalette_8u32u_C1R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pTable : access nppdefs_h.Npp32u;
      nBitSize : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:7520
   pragma Import (C, nppiLUTPalette_8u32u_C1R, "nppiLUTPalette_8u32u_C1R");

  --*
  -- * Three channel 8-bit unsigned bit range restricted palette look-up-table color conversion.
  -- * 
  -- * The LUT is derived from a set of user defined mapping points in a palette and 
  -- * source pixels are then processed using a restricted bit range when looking up palette values.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pTables Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT palette values.
  -- * \param nBitSize Number of least significant bits (must be > 0 and <= 8) of each source pixel value to use as index into palette table during conversion.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_PALETTE_BITSIZE_ERROR if nBitSize is < 1 or > 8.
  --  

   function nppiLUTPalette_8u_C3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pTables : System.Address;
      nBitSize : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:7540
   pragma Import (C, nppiLUTPalette_8u_C3R, "nppiLUTPalette_8u_C3R");

  --*
  -- * Four channel 8-bit unsigned bit range restricted palette look-up-table color conversion.
  -- * 
  -- * The LUT is derived from a set of user defined mapping points in a palette and 
  -- * source pixels are then processed using a restricted bit range when looking up palette values.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pTables Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT palette values.
  -- * \param nBitSize Number of least significant bits (must be > 0 and <= 8) of each source pixel value to use as index into palette table during conversion.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_PALETTE_BITSIZE_ERROR if nBitSize is < 1 or > 8.
  --  

   function nppiLUTPalette_8u_C4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pTables : System.Address;
      nBitSize : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:7560
   pragma Import (C, nppiLUTPalette_8u_C4R, "nppiLUTPalette_8u_C4R");

  --*
  -- * Four channel 8-bit unsigned bit range restricted palette look-up-table color conversion, not affecting Alpha.
  -- *
  -- * The LUT is derived from a set of user defined mapping points in a palette and 
  -- * source pixels are then processed using a restricted bit range when looking up palette values.
  -- * Alpha channel is the last channel and is not processed.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pTables Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT palette values.
  -- * \param nBitSize Number of least significant bits (must be > 0 and <= 8) of each source pixel value to use as index into palette table during conversion.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_PALETTE_BITSIZE_ERROR if nBitSize is < 1 or > 8.
  --  

   function nppiLUTPalette_8u_AC4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pTables : System.Address;
      nBitSize : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:7581
   pragma Import (C, nppiLUTPalette_8u_AC4R, "nppiLUTPalette_8u_AC4R");

  --*
  -- * One channel 16-bit unsigned bit range restricted palette look-up-table color conversion.
  -- *
  -- * The LUT is derived from a set of user defined mapping points in a palette and 
  -- * source pixels are then processed using a restricted bit range when looking up palette values.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pTable Pointer to an array of user defined OUTPUT palette values (this is a device memory pointer)
  -- * \param nBitSize Number of least significant bits (must be > 0 and <= 16) of each source pixel value to use as index into palette table during conversion.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_PALETTE_BITSIZE_ERROR if nBitSize is < 1 or > 16.
  --  

   function nppiLUTPalette_16u_C1R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pTable : access nppdefs_h.Npp16u;
      nBitSize : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:7601
   pragma Import (C, nppiLUTPalette_16u_C1R, "nppiLUTPalette_16u_C1R");

  --*
  -- * One channel 16-bit unsigned bit range restricted 8-bit unsigned palette look-up-table color conversion with 8-bit unsigned destination output per pixel.
  -- *
  -- * The LUT is derived from a set of user defined mapping points in a palette and 
  -- * source pixels are then processed using a restricted bit range when looking up palette values.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step (1 unsigned byte per pixel).
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pTable Pointer to an array of user defined OUTPUT palette values (this is a device memory pointer)
  -- * \param nBitSize Number of least significant bits (must be > 0 and <= 16) of each source pixel value to use as index into palette table during conversion.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_PALETTE_BITSIZE_ERROR if nBitSize is < 1 or > 16.
  --  

   function nppiLUTPalette_16u8u_C1R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pTable : access nppdefs_h.Npp8u;
      nBitSize : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:7621
   pragma Import (C, nppiLUTPalette_16u8u_C1R, "nppiLUTPalette_16u8u_C1R");

  --*
  -- * One channel 16-bit unsigned bit range restricted 24-bit unsigned palette look-up-table color conversion with 24-bit unsigned destination output per pixel.
  -- *
  -- * The LUT is derived from a set of user defined mapping points in a palette and 
  -- * source pixels are then processed using a restricted bit range when looking up palette values.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step (3 unsigned bytes per pixel).
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pTable Pointer to an array of user defined OUTPUT palette values (this is a device memory pointer)
  -- * \param nBitSize Number of least significant bits (must be > 0 and <= 16) of each source pixel value to use as index into palette table during conversion.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_PALETTE_BITSIZE_ERROR if nBitSize is < 1 or > 16.
  --  

   function nppiLUTPalette_16u24u_C1R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pTable : access nppdefs_h.Npp8u;
      nBitSize : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:7641
   pragma Import (C, nppiLUTPalette_16u24u_C1R, "nppiLUTPalette_16u24u_C1R");

  --*
  -- * One channel 16-bit unsigned bit range restricted 32-bit palette look-up-table color conversion with 32-bit unsigned destination output per pixel.
  -- *
  -- * The LUT is derived from a set of user defined mapping points in a palette and 
  -- * source pixels are then processed using a restricted bit range when looking up palette values.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step (4 bytes per pixel).
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pTable Pointer to an array of user defined OUTPUT palette values (this is a device memory pointer)
  -- * \param nBitSize Number of least significant bits (must be > 0 and <= 16) of each source pixel value to use as index into palette table during conversion.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_PALETTE_BITSIZE_ERROR if nBitSize is < 1 or > 16.
  --  

   function nppiLUTPalette_16u32u_C1R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pTable : access nppdefs_h.Npp32u;
      nBitSize : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:7661
   pragma Import (C, nppiLUTPalette_16u32u_C1R, "nppiLUTPalette_16u32u_C1R");

  --*
  -- * Three channel 16-bit unsigned bit range restricted palette look-up-table color conversion.
  -- * 
  -- * The LUT is derived from a set of user defined mapping points in a palette and 
  -- * source pixels are then processed using a restricted bit range when looking up palette values.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pTables Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT palette values.
  -- * \param nBitSize Number of least significant bits (must be > 0 and <= 16) of each source pixel value to use as index into palette table during conversion.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_PALETTE_BITSIZE_ERROR if nBitSize is < 1 or > 16.
  --  

   function nppiLUTPalette_16u_C3R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pTables : System.Address;
      nBitSize : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:7681
   pragma Import (C, nppiLUTPalette_16u_C3R, "nppiLUTPalette_16u_C3R");

  --*
  -- * Four channel 16-bit unsigned bit range restricted palette look-up-table color conversion.
  -- * 
  -- * The LUT is derived from a set of user defined mapping points in a palette and 
  -- * source pixels are then processed using a restricted bit range when looking up palette values.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pTables Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT palette values.
  -- * \param nBitSize Number of least significant bits (must be > 0 and <= 16) of each source pixel value to use as index into palette table during conversion.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_PALETTE_BITSIZE_ERROR if nBitSize is < 1 or > 16.
  --  

   function nppiLUTPalette_16u_C4R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pTables : System.Address;
      nBitSize : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:7701
   pragma Import (C, nppiLUTPalette_16u_C4R, "nppiLUTPalette_16u_C4R");

  --*
  -- * Four channel 16-bit unsigned bit range restricted palette look-up-table color conversion, not affecting Alpha.
  -- *
  -- * The LUT is derived from a set of user defined mapping points in a palette and 
  -- * source pixels are then processed using a restricted bit range when looking up palette values.
  -- * Alpha channel is the last channel and is not processed.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pTables Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT palette values.
  -- * \param nBitSize Number of least significant bits (must be > 0 and <= 16) of each source pixel value to use as index into palette table during conversion.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_PALETTE_BITSIZE_ERROR if nBitSize is < 1 or > 16.
  --  

   function nppiLUTPalette_16u_AC4R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pTables : System.Address;
      nBitSize : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:7722
   pragma Import (C, nppiLUTPalette_16u_AC4R, "nppiLUTPalette_16u_AC4R");

  --*
  -- * Three channel 8-bit unsigned source bit range restricted palette look-up-table color conversion to four channel 8-bit unsigned destination output with alpha.
  -- *
  -- * The LUT is derived from a set of user defined mapping points in a palette and 
  -- * source pixels are then processed using a restricted bit range when looking up palette values.
  -- * This function also reverses the source pixel channel order in the destination so the Alpha channel is the first channel.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step (3 bytes per pixel).
  -- * \param nAlphaValue Signed alpha value that will be used to initialize the pixel alpha channel position in all modified destination pixels.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step (4 bytes per pixel with alpha).
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pTables Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT palette values.
  -- * Alpha values < 0 or > 255 will cause destination pixel alpha channel values to be unmodified.
  -- * \param nBitSize Number of least significant bits (must be > 0 and <= 8) of each source pixel value to use as index into palette table during conversion.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_PALETTE_BITSIZE_ERROR if nBitSize is < 1 or > 8.
  --  

   function nppiLUTPaletteSwap_8u_C3A0C4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      nAlphaValue : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pTables : System.Address;
      nBitSize : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:7745
   pragma Import (C, nppiLUTPaletteSwap_8u_C3A0C4R, "nppiLUTPaletteSwap_8u_C3A0C4R");

  --*
  -- * Three channel 16-bit unsigned source bit range restricted palette look-up-table color conversion to four channel 16-bit unsigned destination output with alpha.
  -- *
  -- * The LUT is derived from a set of user defined mapping points in a palette and 
  -- * source pixels are then processed using a restricted bit range when looking up palette values.
  -- * This function also reverses the source pixel channel order in the destination so the Alpha channel is the first channel.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step (3 unsigned short integers per pixel).
  -- * \param nAlphaValue Signed alpha value that will be used to initialize the pixel alpha channel position in all modified destination pixels.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step (4 unsigned short integers per pixel with alpha).
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pTables Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT palette values.
  -- * Alpha values < 0 or > 65535 will cause destination pixel alpha channel values to be unmodified.
  -- * \param nBitSize Number of least significant bits (must be > 0 and <= 16) of each source pixel value to use as index into palette table during conversion.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- *        - ::NPP_LUT_PALETTE_BITSIZE_ERROR if nBitSize is < 1 or > 16.
  --  

   function nppiLUTPaletteSwap_16u_C3A0C4R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      nAlphaValue : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pTables : System.Address;
      nBitSize : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_color_conversion.h:7768
   pragma Import (C, nppiLUTPaletteSwap_16u_C3A0C4R, "nppiLUTPaletteSwap_16u_C3A0C4R");

  --* @}  
  --* @} image_color_processing  
  --* @} image_color_conversion  
  -- extern "C"  
end nppi_color_conversion_h;

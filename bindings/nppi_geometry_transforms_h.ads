pragma Ada_2005;
pragma Style_Checks (Off);

with Interfaces.C; use Interfaces.C;
with nppdefs_h;
with System;

package nppi_geometry_transforms_h is

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
  -- * \file nppi_geometry_transforms.h
  -- * Image Geometry Transform Primitives.
  --  

  --* @defgroup image_geometry_transforms Geometry Transforms
  -- *  @ingroup nppi
  -- *
  -- * Routines manipulating an image's geometry.
  -- *
  -- * These functions can be found in either the nppi or nppig libraries. Linking to only the sub-libraries that you use can significantly
  -- * save link time, application load time, and CUDA runtime startup time when using dynamic libraries.
  -- *
  -- * \section geometric_transform_api Geometric Transform API Specifics
  -- *
  -- * This section covers some of the unique API features common to the
  -- * geometric transform primitives.
  -- *
  -- * \subsection geometric_transform_roi Geometric Transforms and ROIs
  -- *
  -- * Geometric transforms operate on source and destination ROIs. The way
  -- * these ROIs affect the processing of pixels differs from other (non
  -- * geometric) image-processing primitives: Only pixels in the intersection
  -- * of the destination ROI and the transformed source ROI are being
  -- * processed.
  -- *
  -- * The typical processing proceedes as follows:
  -- * -# Transform the rectangular source ROI (given in source image coordinates)
  -- *		into the destination image space. This yields a quadrilateral.
  -- * -# Write only pixels in the intersection of the transformed source ROI and
  -- *		the destination ROI.
  -- *
  -- * \subsection geometric_transforms_interpolation Pixel Interpolation
  -- *
  -- * The majority of image geometry transform operation need to perform a 
  -- * resampling of the source image as source and destination pixels are not
  -- * coincident.
  -- *
  -- * NPP supports the following pixel inerpolation modes (in order from fastest to 
  -- * slowest and lowest to highest quality):
  -- * - nearest neighbor
  -- * - linear interpolation
  -- * - cubic convolution
  -- * - supersampling
  -- * - interpolation using Lanczos window function
  -- *
  -- * @{
  -- *
  --  

  --* @defgroup image_resize_square_pixel ResizeSqrPixel
  -- *
  -- * ResizeSqrPixel supports the following interpolation modes:
  -- *
  -- * \code
  -- *   NPPI_INTER_NN
  -- *   NPPI_INTER_LINEAR
  -- *   NPPI_INTER_CUBIC
  -- *   NPPI_INTER_CUBIC2P_BSPLINE
  -- *   NPPI_INTER_CUBIC2P_CATMULLROM
  -- *   NPPI_INTER_CUBIC2P_B05C03
  -- *   NPPI_INTER_SUPER
  -- *   NPPI_INTER_LANCZOS
  -- * \endcode
  -- *
  -- * ResizeSqrPixel attempts to choose source pixels that would approximately represent the center of the destination pixels.
  -- * It does so by using the following scaling formula to select source pixels for interpolation:
  -- *
  -- * \code
  -- *   nAdjustedXFactor = 1.0 / nXFactor;
  -- *   nAdjustedYFactor = 1.0 / nYFactor;
  -- *   nAdjustedXShift = nXShift * nAdjustedXFactor + ((1.0 - nAdjustedXFactor) * 0.5);
  -- *   nAdjustedYShift = nYShift * nAdjustedYFactor + ((1.0 - nAdjustedYFactor) * 0.5);
  -- *   nSrcX = nAdjustedXFactor * nDstX - nAdjustedXShift;
  -- *   nSrcY = nAdjustedYFactor * nDstY - nAdjustedYShift;
  -- * \endcode
  -- *
  -- * In the ResizeSqrPixel functions below source image clip checking is handled as follows:
  -- *
  -- * If the source pixel fractional x and y coordinates are greater than or equal to oSizeROI.x and less than oSizeROI.x + oSizeROI.width and
  -- * greater than or equal to oSizeROI.y and less than oSizeROI.y + oSizeROI.height then the source pixel is considered to be within
  -- * the source image clip rectangle and the source image is sampled.  Otherwise the source image is not sampled and a destination pixel is not
  -- * written to the destination image. 
  -- *
  -- * \section resize_error_codes Error Codes
  -- * The resize primitives return the following error codes:
  -- *
  -- *         - ::NPP_WRONG_INTERSECTION_ROI_ERROR indicates an error condition if
  -- *           srcROIRect has no intersection with the source image.
  -- *         - ::NPP_RESIZE_NO_OPERATION_ERROR if either destination ROI width or
  -- *           height is less than 1 pixel.
  -- *         - ::NPP_RESIZE_FACTOR_ERROR Indicates an error condition if either nXFactor or
  -- *           nYFactor is less than or equal to zero.
  -- *         - ::NPP_INTERPOLATION_ERROR if eInterpolation has an illegal value.
  -- *         - ::NPP_SIZE_ERROR if source size width or height is less than 2 pixels.
  -- *
  -- * @{
  -- *
  --  

  --* @name GetResizeRect
  -- * Returns NppiRect which represents the offset and size of the destination rectangle that would be generated by
  -- * resizing the source NppiRect by the requested scale factors and shifts.
  -- *                                    
  -- * @{
  -- *
  --  

  --*
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pDstRect User supplied host memory pointer to an NppiRect structure that will be filled in by this function with the region of interest in the destination image.
  -- * \param nXFactor Factor by which x dimension is changed. 
  -- * \param nYFactor Factor by which y dimension is changed. 
  -- * \param nXShift Source pixel shift in x-direction.
  -- * \param nYShift Source pixel shift in y-direction.
  -- * \param eInterpolation The type of eInterpolation to perform resampling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref resize_error_codes
  --  

   function nppiGetResizeRect
     (oSrcROI : nppdefs_h.NppiRect;
      pDstRect : access nppdefs_h.NppiRect;
      nXFactor : double;
      nYFactor : double;
      nXShift : double;
      nYShift : double;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:178
   pragma Import (C, nppiGetResizeRect, "nppiGetResizeRect");

  --* @}  
  --* @name ResizeSqrPixel
  -- * Resizes images.
  -- *                                    
  -- * @{
  -- *
  --  

  --*
  -- * 1 channel 8-bit unsigned image resize.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image.
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Region of interest in the destination image.
  -- * \param nXFactor Factor by which x dimension is changed. 
  -- * \param nYFactor Factor by which y dimension is changed. 
  -- * \param nXShift Source pixel shift in x-direction.
  -- * \param nYShift Source pixel shift in y-direction.
  -- * \param eInterpolation The type of eInterpolation to perform resampling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref resize_error_codes
  --  

   function nppiResizeSqrPixel_8u_C1R
     (pSrc : access nppdefs_h.Npp8u;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      nXFactor : double;
      nYFactor : double;
      nXShift : double;
      nYShift : double;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:208
   pragma Import (C, nppiResizeSqrPixel_8u_C1R, "nppiResizeSqrPixel_8u_C1R");

  --*
  -- * 3 channel 8-bit unsigned image resize.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image.
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Region of interest in the destination image.
  -- * \param nXFactor Factor by which x dimension is changed. 
  -- * \param nYFactor Factor by which y dimension is changed. 
  -- * \param nXShift Source pixel shift in x-direction.
  -- * \param nYShift Source pixel shift in y-direction.
  -- * \param eInterpolation The type of eInterpolation to perform resampling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref resize_error_codes
  --  

   function nppiResizeSqrPixel_8u_C3R
     (pSrc : access nppdefs_h.Npp8u;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      nXFactor : double;
      nYFactor : double;
      nXShift : double;
      nYShift : double;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:230
   pragma Import (C, nppiResizeSqrPixel_8u_C3R, "nppiResizeSqrPixel_8u_C3R");

  --*
  -- * 4 channel 8-bit unsigned image resize.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image.
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Region of interest in the destination image.
  -- * \param nXFactor Factor by which x dimension is changed. 
  -- * \param nYFactor Factor by which y dimension is changed. 
  -- * \param nXShift Source pixel shift in x-direction.
  -- * \param nYShift Source pixel shift in y-direction.
  -- * \param eInterpolation The type of eInterpolation to perform resampling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref resize_error_codes
  --  

   function nppiResizeSqrPixel_8u_C4R
     (pSrc : access nppdefs_h.Npp8u;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      nXFactor : double;
      nYFactor : double;
      nXShift : double;
      nYShift : double;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:252
   pragma Import (C, nppiResizeSqrPixel_8u_C4R, "nppiResizeSqrPixel_8u_C4R");

  --*
  -- * 4 channel 8-bit unsigned image resize not affecting alpha.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image.
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Region of interest in the destination image.
  -- * \param nXFactor Factor by which x dimension is changed. 
  -- * \param nYFactor Factor by which y dimension is changed. 
  -- * \param nXShift Source pixel shift in x-direction.
  -- * \param nYShift Source pixel shift in y-direction.
  -- * \param eInterpolation The type of interpolation to perform resampling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref resize_error_codes
  --  

   function nppiResizeSqrPixel_8u_AC4R
     (pSrc : access nppdefs_h.Npp8u;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      nXFactor : double;
      nYFactor : double;
      nXShift : double;
      nYShift : double;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:274
   pragma Import (C, nppiResizeSqrPixel_8u_AC4R, "nppiResizeSqrPixel_8u_AC4R");

  --*
  -- * 3 channel 8-bit unsigned planar image resize.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array (host memory array containing device memory image plane pointers).
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image.
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pDst \ref destination_planar_image_pointer_array (host memory array containing device memory image plane pointers).
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Region of interest in the destination image.
  -- * \param nXFactor Factor by which x dimension is changed. 
  -- * \param nYFactor Factor by which y dimension is changed. 
  -- * \param nXShift Source pixel shift in x-direction.
  -- * \param nYShift Source pixel shift in y-direction.
  -- * \param eInterpolation The type of eInterpolation to perform resampling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref resize_error_codes
  --  

   function nppiResizeSqrPixel_8u_P3R
     (pSrc : System.Address;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : System.Address;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      nXFactor : double;
      nYFactor : double;
      nXShift : double;
      nYShift : double;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:296
   pragma Import (C, nppiResizeSqrPixel_8u_P3R, "nppiResizeSqrPixel_8u_P3R");

  --*
  -- * 4 channel 8-bit unsigned planar image resize.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array (host memory array containing device memory image plane pointers).
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image.
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pDst \ref destination_planar_image_pointer_array (host memory array containing device memory image plane pointers).
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Region of interest in the destination image.
  -- * \param nXFactor Factor by which x dimension is changed. 
  -- * \param nYFactor Factor by which y dimension is changed. 
  -- * \param nXShift Source pixel shift in x-direction.
  -- * \param nYShift Source pixel shift in y-direction.
  -- * \param eInterpolation The type of eInterpolation to perform resampling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref resize_error_codes
  --  

   function nppiResizeSqrPixel_8u_P4R
     (pSrc : System.Address;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : System.Address;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      nXFactor : double;
      nYFactor : double;
      nXShift : double;
      nYShift : double;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:318
   pragma Import (C, nppiResizeSqrPixel_8u_P4R, "nppiResizeSqrPixel_8u_P4R");

  --*
  -- * 1 channel 16-bit unsigned image resize.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image.
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Region of interest in the destination image.
  -- * \param nXFactor Factor by which x dimension is changed. 
  -- * \param nYFactor Factor by which y dimension is changed. 
  -- * \param nXShift Source pixel shift in x-direction.
  -- * \param nYShift Source pixel shift in y-direction.
  -- * \param eInterpolation The type of eInterpolation to perform resampling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref resize_error_codes
  --  

   function nppiResizeSqrPixel_16u_C1R
     (pSrc : access nppdefs_h.Npp16u;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      nXFactor : double;
      nYFactor : double;
      nXShift : double;
      nYShift : double;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:340
   pragma Import (C, nppiResizeSqrPixel_16u_C1R, "nppiResizeSqrPixel_16u_C1R");

  --*
  -- * 3 channel 16-bit unsigned image resize.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image.
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Region of interest in the destination image.
  -- * \param nXFactor Factor by which x dimension is changed. 
  -- * \param nYFactor Factor by which y dimension is changed. 
  -- * \param nXShift Source pixel shift in x-direction.
  -- * \param nYShift Source pixel shift in y-direction.
  -- * \param eInterpolation The type of eInterpolation to perform resampling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref resize_error_codes
  --  

   function nppiResizeSqrPixel_16u_C3R
     (pSrc : access nppdefs_h.Npp16u;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      nXFactor : double;
      nYFactor : double;
      nXShift : double;
      nYShift : double;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:362
   pragma Import (C, nppiResizeSqrPixel_16u_C3R, "nppiResizeSqrPixel_16u_C3R");

  --*
  -- * 4 channel 16-bit unsigned image resize.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image.
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Region of interest in the destination image.
  -- * \param nXFactor Factor by which x dimension is changed. 
  -- * \param nYFactor Factor by which y dimension is changed. 
  -- * \param nXShift Source pixel shift in x-direction.
  -- * \param nYShift Source pixel shift in y-direction.
  -- * \param eInterpolation The type of eInterpolation to perform resampling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref resize_error_codes
  --  

   function nppiResizeSqrPixel_16u_C4R
     (pSrc : access nppdefs_h.Npp16u;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      nXFactor : double;
      nYFactor : double;
      nXShift : double;
      nYShift : double;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:384
   pragma Import (C, nppiResizeSqrPixel_16u_C4R, "nppiResizeSqrPixel_16u_C4R");

  --*
  -- * 4 channel 16-bit unsigned image resize not affecting alpha.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image.
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Region of interest in the destination image.
  -- * \param nXFactor Factor by which x dimension is changed. 
  -- * \param nYFactor Factor by which y dimension is changed. 
  -- * \param nXShift Source pixel shift in x-direction.
  -- * \param nYShift Source pixel shift in y-direction.
  -- * \param eInterpolation The type of interpolation to perform resampling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref resize_error_codes
  --  

   function nppiResizeSqrPixel_16u_AC4R
     (pSrc : access nppdefs_h.Npp16u;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      nXFactor : double;
      nYFactor : double;
      nXShift : double;
      nYShift : double;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:406
   pragma Import (C, nppiResizeSqrPixel_16u_AC4R, "nppiResizeSqrPixel_16u_AC4R");

  --*
  -- * 3 channel 16-bit unsigned planar image resize.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array (host memory array containing device memory image plane pointers).
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image.
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pDst \ref destination_planar_image_pointer_array (host memory array containing device memory image plane pointers).
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Region of interest in the destination image.
  -- * \param nXFactor Factor by which x dimension is changed. 
  -- * \param nYFactor Factor by which y dimension is changed. 
  -- * \param nXShift Source pixel shift in x-direction.
  -- * \param nYShift Source pixel shift in y-direction.
  -- * \param eInterpolation The type of eInterpolation to perform resampling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref resize_error_codes
  --  

   function nppiResizeSqrPixel_16u_P3R
     (pSrc : System.Address;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : System.Address;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      nXFactor : double;
      nYFactor : double;
      nXShift : double;
      nYShift : double;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:428
   pragma Import (C, nppiResizeSqrPixel_16u_P3R, "nppiResizeSqrPixel_16u_P3R");

  --*
  -- * 4 channel 16-bit unsigned planar image resize.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array (host memory array containing device memory image plane pointers).
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image.
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pDst \ref destination_planar_image_pointer_array (host memory array containing device memory image plane pointers).
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Region of interest in the destination image.
  -- * \param nXFactor Factor by which x dimension is changed. 
  -- * \param nYFactor Factor by which y dimension is changed. 
  -- * \param nXShift Source pixel shift in x-direction.
  -- * \param nYShift Source pixel shift in y-direction.
  -- * \param eInterpolation The type of eInterpolation to perform resampling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref resize_error_codes
  --  

   function nppiResizeSqrPixel_16u_P4R
     (pSrc : System.Address;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : System.Address;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      nXFactor : double;
      nYFactor : double;
      nXShift : double;
      nYShift : double;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:450
   pragma Import (C, nppiResizeSqrPixel_16u_P4R, "nppiResizeSqrPixel_16u_P4R");

  --*
  -- * 1 channel 16-bit signed image resize.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image.
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Region of interest in the destination image.
  -- * \param nXFactor Factor by which x dimension is changed. 
  -- * \param nYFactor Factor by which y dimension is changed. 
  -- * \param nXShift Source pixel shift in x-direction.
  -- * \param nYShift Source pixel shift in y-direction.
  -- * \param eInterpolation The type of eInterpolation to perform resampling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref resize_error_codes
  --  

   function nppiResizeSqrPixel_16s_C1R
     (pSrc : access nppdefs_h.Npp16s;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      nXFactor : double;
      nYFactor : double;
      nXShift : double;
      nYShift : double;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:472
   pragma Import (C, nppiResizeSqrPixel_16s_C1R, "nppiResizeSqrPixel_16s_C1R");

  --*
  -- * 3 channel 16-bit signed image resize.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image.
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Region of interest in the destination image.
  -- * \param nXFactor Factor by which x dimension is changed. 
  -- * \param nYFactor Factor by which y dimension is changed. 
  -- * \param nXShift Source pixel shift in x-direction.
  -- * \param nYShift Source pixel shift in y-direction.
  -- * \param eInterpolation The type of eInterpolation to perform resampling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref resize_error_codes
  --  

   function nppiResizeSqrPixel_16s_C3R
     (pSrc : access nppdefs_h.Npp16s;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      nXFactor : double;
      nYFactor : double;
      nXShift : double;
      nYShift : double;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:494
   pragma Import (C, nppiResizeSqrPixel_16s_C3R, "nppiResizeSqrPixel_16s_C3R");

  --*
  -- * 4 channel 16-bit signed image resize.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image.
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Region of interest in the destination image.
  -- * \param nXFactor Factor by which x dimension is changed. 
  -- * \param nYFactor Factor by which y dimension is changed. 
  -- * \param nXShift Source pixel shift in x-direction.
  -- * \param nYShift Source pixel shift in y-direction.
  -- * \param eInterpolation The type of eInterpolation to perform resampling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref resize_error_codes
  --  

   function nppiResizeSqrPixel_16s_C4R
     (pSrc : access nppdefs_h.Npp16s;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      nXFactor : double;
      nYFactor : double;
      nXShift : double;
      nYShift : double;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:516
   pragma Import (C, nppiResizeSqrPixel_16s_C4R, "nppiResizeSqrPixel_16s_C4R");

  --*
  -- * 4 channel 16-bit signed image resize not affecting alpha.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image.
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Region of interest in the destination image.
  -- * \param nXFactor Factor by which x dimension is changed. 
  -- * \param nYFactor Factor by which y dimension is changed. 
  -- * \param nXShift Source pixel shift in x-direction.
  -- * \param nYShift Source pixel shift in y-direction.
  -- * \param eInterpolation The type of interpolation to perform resampling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref resize_error_codes
  --  

   function nppiResizeSqrPixel_16s_AC4R
     (pSrc : access nppdefs_h.Npp16s;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      nXFactor : double;
      nYFactor : double;
      nXShift : double;
      nYShift : double;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:538
   pragma Import (C, nppiResizeSqrPixel_16s_AC4R, "nppiResizeSqrPixel_16s_AC4R");

  --*
  -- * 3 channel 16-bit signed planar image resize.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array (host memory array containing device memory image plane pointers).
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image.
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pDst \ref destination_planar_image_pointer_array (host memory array containing device memory image plane pointers).
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Region of interest in the destination image.
  -- * \param nXFactor Factor by which x dimension is changed. 
  -- * \param nYFactor Factor by which y dimension is changed. 
  -- * \param nXShift Source pixel shift in x-direction.
  -- * \param nYShift Source pixel shift in y-direction.
  -- * \param eInterpolation The type of eInterpolation to perform resampling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref resize_error_codes
  --  

   function nppiResizeSqrPixel_16s_P3R
     (pSrc : System.Address;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : System.Address;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      nXFactor : double;
      nYFactor : double;
      nXShift : double;
      nYShift : double;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:560
   pragma Import (C, nppiResizeSqrPixel_16s_P3R, "nppiResizeSqrPixel_16s_P3R");

  --*
  -- * 4 channel 16-bit signed planar image resize.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array (host memory array containing device memory image plane pointers).
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image.
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pDst \ref destination_planar_image_pointer_array (host memory array containing device memory image plane pointers).
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Region of interest in the destination image.
  -- * \param nXFactor Factor by which x dimension is changed. 
  -- * \param nYFactor Factor by which y dimension is changed. 
  -- * \param nXShift Source pixel shift in x-direction.
  -- * \param nYShift Source pixel shift in y-direction.
  -- * \param eInterpolation The type of eInterpolation to perform resampling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref resize_error_codes
  --  

   function nppiResizeSqrPixel_16s_P4R
     (pSrc : System.Address;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : System.Address;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      nXFactor : double;
      nYFactor : double;
      nXShift : double;
      nYShift : double;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:582
   pragma Import (C, nppiResizeSqrPixel_16s_P4R, "nppiResizeSqrPixel_16s_P4R");

  --*
  -- * 1 channel 32-bit floating point image resize.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image.
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Region of interest in the destination image.
  -- * \param nXFactor Factor by which x dimension is changed. 
  -- * \param nYFactor Factor by which y dimension is changed. 
  -- * \param nXShift Source pixel shift in x-direction.
  -- * \param nYShift Source pixel shift in y-direction.
  -- * \param eInterpolation The type of eInterpolation to perform resampling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref resize_error_codes
  --  

   function nppiResizeSqrPixel_32f_C1R
     (pSrc : access nppdefs_h.Npp32f;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      nXFactor : double;
      nYFactor : double;
      nXShift : double;
      nYShift : double;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:604
   pragma Import (C, nppiResizeSqrPixel_32f_C1R, "nppiResizeSqrPixel_32f_C1R");

  --*
  -- * 3 channel 32-bit floating point image resize.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image.
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Region of interest in the destination image.
  -- * \param nXFactor Factor by which x dimension is changed. 
  -- * \param nYFactor Factor by which y dimension is changed. 
  -- * \param nXShift Source pixel shift in x-direction.
  -- * \param nYShift Source pixel shift in y-direction.
  -- * \param eInterpolation The type of eInterpolation to perform resampling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref resize_error_codes
  --  

   function nppiResizeSqrPixel_32f_C3R
     (pSrc : access nppdefs_h.Npp32f;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      nXFactor : double;
      nYFactor : double;
      nXShift : double;
      nYShift : double;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:626
   pragma Import (C, nppiResizeSqrPixel_32f_C3R, "nppiResizeSqrPixel_32f_C3R");

  --*
  -- * 4 channel 32-bit floating point image resize.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image.
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Region of interest in the destination image.
  -- * \param nXFactor Factor by which x dimension is changed. 
  -- * \param nYFactor Factor by which y dimension is changed. 
  -- * \param nXShift Source pixel shift in x-direction.
  -- * \param nYShift Source pixel shift in y-direction.
  -- * \param eInterpolation The type of eInterpolation to perform resampling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref resize_error_codes
  --  

   function nppiResizeSqrPixel_32f_C4R
     (pSrc : access nppdefs_h.Npp32f;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      nXFactor : double;
      nYFactor : double;
      nXShift : double;
      nYShift : double;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:648
   pragma Import (C, nppiResizeSqrPixel_32f_C4R, "nppiResizeSqrPixel_32f_C4R");

  --*
  -- * 4 channel 32-bit floating point image resize not affecting alpha.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image.
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Region of interest in the destination image.
  -- * \param nXFactor Factor by which x dimension is changed. 
  -- * \param nYFactor Factor by which y dimension is changed. 
  -- * \param nXShift Source pixel shift in x-direction.
  -- * \param nYShift Source pixel shift in y-direction.
  -- * \param eInterpolation The type of interpolation to perform resampling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref resize_error_codes
  --  

   function nppiResizeSqrPixel_32f_AC4R
     (pSrc : access nppdefs_h.Npp32f;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      nXFactor : double;
      nYFactor : double;
      nXShift : double;
      nYShift : double;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:670
   pragma Import (C, nppiResizeSqrPixel_32f_AC4R, "nppiResizeSqrPixel_32f_AC4R");

  --*
  -- * 3 channel 32-bit floating point planar image resize.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array (host memory array containing device memory image plane pointers).
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image.
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pDst \ref destination_planar_image_pointer_array (host memory array containing device memory image plane pointers).
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Region of interest in the destination image.
  -- * \param nXFactor Factor by which x dimension is changed. 
  -- * \param nYFactor Factor by which y dimension is changed. 
  -- * \param nXShift Source pixel shift in x-direction.
  -- * \param nYShift Source pixel shift in y-direction.
  -- * \param eInterpolation The type of eInterpolation to perform resampling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref resize_error_codes
  --  

   function nppiResizeSqrPixel_32f_P3R
     (pSrc : System.Address;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : System.Address;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      nXFactor : double;
      nYFactor : double;
      nXShift : double;
      nYShift : double;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:692
   pragma Import (C, nppiResizeSqrPixel_32f_P3R, "nppiResizeSqrPixel_32f_P3R");

  --*
  -- * 4 channel 32-bit floating point planar image resize.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array (host memory array containing device memory image plane pointers).
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image.
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pDst \ref destination_planar_image_pointer_array (host memory array containing device memory image plane pointers).
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Region of interest in the destination image.
  -- * \param nXFactor Factor by which x dimension is changed. 
  -- * \param nYFactor Factor by which y dimension is changed. 
  -- * \param nXShift Source pixel shift in x-direction.
  -- * \param nYShift Source pixel shift in y-direction.
  -- * \param eInterpolation The type of eInterpolation to perform resampling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref resize_error_codes
  --  

   function nppiResizeSqrPixel_32f_P4R
     (pSrc : System.Address;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : System.Address;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      nXFactor : double;
      nYFactor : double;
      nXShift : double;
      nYShift : double;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:714
   pragma Import (C, nppiResizeSqrPixel_32f_P4R, "nppiResizeSqrPixel_32f_P4R");

  --*
  -- * 1 channel 64-bit floating point image resize.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image.
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Region of interest in the destination image.
  -- * \param nXFactor Factor by which x dimension is changed. 
  -- * \param nYFactor Factor by which y dimension is changed. 
  -- * \param nXShift Source pixel shift in x-direction.
  -- * \param nYShift Source pixel shift in y-direction.
  -- * \param eInterpolation The type of eInterpolation to perform resampling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref resize_error_codes
  --  

   function nppiResizeSqrPixel_64f_C1R
     (pSrc : access nppdefs_h.Npp64f;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp64f;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      nXFactor : double;
      nYFactor : double;
      nXShift : double;
      nYShift : double;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:736
   pragma Import (C, nppiResizeSqrPixel_64f_C1R, "nppiResizeSqrPixel_64f_C1R");

  --*
  -- * 3 channel 64-bit floating point image resize.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image.
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Region of interest in the destination image.
  -- * \param nXFactor Factor by which x dimension is changed. 
  -- * \param nYFactor Factor by which y dimension is changed. 
  -- * \param nXShift Source pixel shift in x-direction.
  -- * \param nYShift Source pixel shift in y-direction.
  -- * \param eInterpolation The type of eInterpolation to perform resampling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref resize_error_codes
  --  

   function nppiResizeSqrPixel_64f_C3R
     (pSrc : access nppdefs_h.Npp64f;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp64f;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      nXFactor : double;
      nYFactor : double;
      nXShift : double;
      nYShift : double;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:758
   pragma Import (C, nppiResizeSqrPixel_64f_C3R, "nppiResizeSqrPixel_64f_C3R");

  --*
  -- * 4 channel 64-bit floating point image resize.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image.
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Region of interest in the destination image.
  -- * \param nXFactor Factor by which x dimension is changed. 
  -- * \param nYFactor Factor by which y dimension is changed. 
  -- * \param nXShift Source pixel shift in x-direction.
  -- * \param nYShift Source pixel shift in y-direction.
  -- * \param eInterpolation The type of eInterpolation to perform resampling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref resize_error_codes
  --  

   function nppiResizeSqrPixel_64f_C4R
     (pSrc : access nppdefs_h.Npp64f;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp64f;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      nXFactor : double;
      nYFactor : double;
      nXShift : double;
      nYShift : double;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:780
   pragma Import (C, nppiResizeSqrPixel_64f_C4R, "nppiResizeSqrPixel_64f_C4R");

  --*
  -- * 4 channel 64-bit floating point image resize not affecting alpha.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image.
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Region of interest in the destination image.
  -- * \param nXFactor Factor by which x dimension is changed. 
  -- * \param nYFactor Factor by which y dimension is changed. 
  -- * \param nXShift Source pixel shift in x-direction.
  -- * \param nYShift Source pixel shift in y-direction.
  -- * \param eInterpolation The type of interpolation to perform resampling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref resize_error_codes
  --  

   function nppiResizeSqrPixel_64f_AC4R
     (pSrc : access nppdefs_h.Npp64f;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp64f;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      nXFactor : double;
      nYFactor : double;
      nXShift : double;
      nYShift : double;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:802
   pragma Import (C, nppiResizeSqrPixel_64f_AC4R, "nppiResizeSqrPixel_64f_AC4R");

  --*
  -- * 3 channel 64-bit floating point planar image resize.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array (host memory array containing device memory image plane pointers).
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image.
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pDst \ref destination_planar_image_pointer_array (host memory array containing device memory image plane pointers).
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Region of interest in the destination image.
  -- * \param nXFactor Factor by which x dimension is changed. 
  -- * \param nYFactor Factor by which y dimension is changed. 
  -- * \param nXShift Source pixel shift in x-direction.
  -- * \param nYShift Source pixel shift in y-direction.
  -- * \param eInterpolation The type of eInterpolation to perform resampling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref resize_error_codes
  --  

   function nppiResizeSqrPixel_64f_P3R
     (pSrc : System.Address;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : System.Address;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      nXFactor : double;
      nYFactor : double;
      nXShift : double;
      nYShift : double;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:824
   pragma Import (C, nppiResizeSqrPixel_64f_P3R, "nppiResizeSqrPixel_64f_P3R");

  --*
  -- * 4 channel 64-bit floating point planar image resize.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array (host memory array containing device memory image plane pointers).
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image.
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pDst \ref destination_planar_image_pointer_array (host memory array containing device memory image plane pointers).
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Region of interest in the destination image.
  -- * \param nXFactor Factor by which x dimension is changed. 
  -- * \param nYFactor Factor by which y dimension is changed. 
  -- * \param nXShift Source pixel shift in x-direction.
  -- * \param nYShift Source pixel shift in y-direction.
  -- * \param eInterpolation The type of eInterpolation to perform resampling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref resize_error_codes
  --  

   function nppiResizeSqrPixel_64f_P4R
     (pSrc : System.Address;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : System.Address;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      nXFactor : double;
      nYFactor : double;
      nXShift : double;
      nYShift : double;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:846
   pragma Import (C, nppiResizeSqrPixel_64f_P4R, "nppiResizeSqrPixel_64f_P4R");

  --*
  -- * Buffer size for \ref nppiResizeSqrPixel_8u_C1R_Advanced.
  -- * \param oSrcROI \ref roi_specification.
  -- * \param oDstROI \ref roi_specification.
  -- * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
  -- *        <em>host pointer.</em> \ref general_scratch_buffer.
  -- * \param eInterpolationMode The type of eInterpolation to perform resampling. Currently only supports NPPI_INTER_LANCZOS3_Advanced.
  -- * \return NPP_NULL_POINTER_ERROR if hpBufferSize is 0 (NULL),  \ref roi_error_codes.
  --  

  -- host pointer  
   function nppiResizeAdvancedGetBufferHostSize_8u_C1R
     (oSrcROI : nppdefs_h.NppiSize;
      oDstROI : nppdefs_h.NppiSize;
      hpBufferSize : access int;
      eInterpolationMode : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:860
   pragma Import (C, nppiResizeAdvancedGetBufferHostSize_8u_C1R, "nppiResizeAdvancedGetBufferHostSize_8u_C1R");

  --*
  -- * 1 channel 8-bit unsigned image resize. This primitive matches the behavior of GraphicsMagick++.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image.
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Region of interest in the destination image.
  -- * \param nXFactor Factor by which x dimension is changed. 
  -- * \param nYFactor Factor by which y dimension is changed. 
  -- * \param pBuffer Device buffer that is used during calculations.
  -- * \param eInterpolationMode The type of eInterpolation to perform resampling. Currently only supports NPPI_INTER_LANCZOS3_Advanced.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref resize_error_codes
  --  

   function nppiResizeSqrPixel_8u_C1R_Advanced
     (pSrc : access nppdefs_h.Npp8u;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      nXFactor : double;
      nYFactor : double;
      pBuffer : access nppdefs_h.Npp8u;
      eInterpolationMode : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:879
   pragma Import (C, nppiResizeSqrPixel_8u_C1R_Advanced, "nppiResizeSqrPixel_8u_C1R_Advanced");

  --* @}  
  --* @} image_resize_square_pixel  
  --* @defgroup image_resize Resize
  -- *
  -- * This function has been deprecated.  ResizeSqrPixel provides the same functionality and more.
  -- *
  -- * Resize supports the following interpolation modes:
  -- *
  -- * \code
  -- *   NPPI_INTER_NN
  -- *   NPPI_INTER_LINEAR
  -- *   NPPI_INTER_CUBIC
  -- *   NPPI_INTER_SUPER
  -- *   NPPI_INTER_LANCZOS
  -- * \endcode
  -- *
  -- * Resize uses the following scaling formula to select source pixels for interpolation:
  -- *
  -- * \code
  -- *   scaledSrcSize.width = nXFactor * srcRectROI.width;
  -- *   scaledSrcSize.height = nYFactor * srcRectROI.height;
  -- *   nAdjustedXFactor = (srcRectROI.width - 1) / (scaledSrcSize.width - 1);
  -- *   nAdjustedYFactor = (srcRectROI.height - 1) / (scaledSrcSize.height - 1);
  -- *   nSrcX = nAdjustedXFactor * nDstX;
  -- *   nSrcY = nAdjustedYFactor * nDstY;
  -- * \endcode
  -- *
  -- * In the Resize functions below source image clip checking is handled as follows:
  -- *
  -- * If the source pixel fractional x and y coordinates are greater than or equal to oSizeROI.x and less than oSizeROI.x + oSizeROI.width and
  -- * greater than or equal to oSizeROI.y and less than oSizeROI.y + oSizeROI.height then the source pixel is considered to be within
  -- * the source image clip rectangle and the source image is sampled.  Otherwise the source image is not sampled and a destination pixel is not
  -- * written to the destination image. 
  -- *
  -- * \section resize_error_codes Error Codes
  -- * The resize primitives return the following error codes:
  -- *
  -- *         - ::NPP_WRONG_INTERSECTION_ROI_ERROR indicates an error condition if
  -- *           srcROIRect has no intersection with the source image.
  -- *         - ::NPP_RESIZE_NO_OPERATION_ERROR if either destination ROI width or
  -- *           height is less than 1 pixel.
  -- *         - ::NPP_RESIZE_FACTOR_ERROR Indicates an error condition if either nXFactor or
  -- *           nYFactor is less than or equal to zero.
  -- *         - ::NPP_INTERPOLATION_ERROR if eInterpolation has an illegal value.
  -- *         - ::NPP_SIZE_ERROR if source size width or height is less than 2 pixels.
  -- *
  -- * @{
  -- *
  --  

  --* @name Resize
  -- * Resizes images.
  -- *                                    
  -- * @{
  -- *
  --  

  --*
  -- * 1 channel 8-bit unsigned image resize.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param dstROISize Size in pixels of the destination image.
  -- * \param nXFactor Factor by which x dimension is changed. 
  -- * \param nYFactor Factor by which y dimension is changed. 
  -- * \param eInterpolation The type of eInterpolation to perform resampling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref resize_error_codes
  --  

   function nppiResize_8u_C1R
     (pSrc : access nppdefs_h.Npp8u;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      dstROISize : nppdefs_h.NppiSize;
      nXFactor : double;
      nYFactor : double;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:957
   pragma Import (C, nppiResize_8u_C1R, "nppiResize_8u_C1R");

  --*
  -- * 3 channel 8-bit unsigned image resize.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param dstROISize Size in pixels of the destination image.
  -- * \param nXFactor Factor by which x dimension is changed. 
  -- * \param nYFactor Factor by which y dimension is changed. 
  -- * \param eInterpolation The type of eInterpolation to perform resampling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref resize_error_codes
  --  

   function nppiResize_8u_C3R
     (pSrc : access nppdefs_h.Npp8u;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      dstROISize : nppdefs_h.NppiSize;
      nXFactor : double;
      nYFactor : double;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:977
   pragma Import (C, nppiResize_8u_C3R, "nppiResize_8u_C3R");

  --*
  -- * 4 channel 8-bit unsigned image resize.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param dstROISize Size in pixels of the destination image.
  -- * \param nXFactor Factor by which x dimension is changed. 
  -- * \param nYFactor Factor by which y dimension is changed. 
  -- * \param eInterpolation The type of eInterpolation to perform resampling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref resize_error_codes
  --  

   function nppiResize_8u_C4R
     (pSrc : access nppdefs_h.Npp8u;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      dstROISize : nppdefs_h.NppiSize;
      nXFactor : double;
      nYFactor : double;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:997
   pragma Import (C, nppiResize_8u_C4R, "nppiResize_8u_C4R");

  --*
  -- * 4 channel 8-bit unsigned image resize not affecting alpha.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param dstROISize Size in pixels of the destination image.
  -- * \param nXFactor Factor by which x dimension is changed. 
  -- * \param nYFactor Factor by which y dimension is changed. 
  -- * \param eInterpolation The type of interpolation to perform resampling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref resize_error_codes
  --  

   function nppiResize_8u_AC4R
     (pSrc : access nppdefs_h.Npp8u;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      dstROISize : nppdefs_h.NppiSize;
      nXFactor : double;
      nYFactor : double;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:1017
   pragma Import (C, nppiResize_8u_AC4R, "nppiResize_8u_AC4R");

  --*
  -- * 3 channel 8-bit unsigned planar image resize.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array (host memory array containing device memory image plane pointers).
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image.
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pDst \ref destination_planar_image_pointer_array (host memory array containing device memory image plane pointers).
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param dstROISize Size in pixels of the destination image
  -- * \param nXFactor Factor by which x dimension is changed. 
  -- * \param nYFactor Factor by which y dimension is changed. 
  -- * \param eInterpolation The type of eInterpolation to perform resampling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref resize_error_codes
  --  

   function nppiResize_8u_P3R
     (pSrc : System.Address;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : System.Address;
      nDstStep : int;
      dstROISize : nppdefs_h.NppiSize;
      nXFactor : double;
      nYFactor : double;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:1037
   pragma Import (C, nppiResize_8u_P3R, "nppiResize_8u_P3R");

  --*
  -- * 4 channel 8-bit unsigned planar image resize.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array (host memory array containing device memory image plane pointers).
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image.
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pDst \ref destination_planar_image_pointer_array (host memory array containing device memory image plane pointers).
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param dstROISize Size in pixels of the destination image.
  -- * \param nXFactor Factor by which x dimension is changed. 
  -- * \param nYFactor Factor by which y dimension is changed. 
  -- * \param eInterpolation The type of eInterpolation to perform resampling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref resize_error_codes
  --  

   function nppiResize_8u_P4R
     (pSrc : System.Address;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : System.Address;
      nDstStep : int;
      dstROISize : nppdefs_h.NppiSize;
      nXFactor : double;
      nYFactor : double;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:1057
   pragma Import (C, nppiResize_8u_P4R, "nppiResize_8u_P4R");

  --*
  -- * 1 channel 16-bit unsigned image resize.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param dstROISize Size in pixels of the destination image.
  -- * \param nXFactor Factor by which x dimension is changed. 
  -- * \param nYFactor Factor by which y dimension is changed. 
  -- * \param eInterpolation The type of eInterpolation to perform resampling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref resize_error_codes
  --  

   function nppiResize_16u_C1R
     (pSrc : access nppdefs_h.Npp16u;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      dstROISize : nppdefs_h.NppiSize;
      nXFactor : double;
      nYFactor : double;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:1077
   pragma Import (C, nppiResize_16u_C1R, "nppiResize_16u_C1R");

  --*
  -- * 3 channel 16-bit unsigned image resize.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param dstROISize Size in pixels of the destination image.
  -- * \param nXFactor Factor by which x dimension is changed.
  -- * \param nYFactor Factor by which y dimension is changed. 
  -- * \param eInterpolation The type of eInterpolation to perform resampling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref resize_error_codes
  --  

   function nppiResize_16u_C3R
     (pSrc : access nppdefs_h.Npp16u;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      dstROISize : nppdefs_h.NppiSize;
      nXFactor : double;
      nYFactor : double;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:1097
   pragma Import (C, nppiResize_16u_C3R, "nppiResize_16u_C3R");

  --*
  -- * 4 channel 16-bit unsigned image resize.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param dstROISize Size in pixels of the destination image.
  -- * \param nXFactor Factor by which x dimension is changed. 
  -- * \param nYFactor Factor by which y dimension is changed. 
  -- * \param eInterpolation The type of eInterpolation to perform resampling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref resize_error_codes
  --  

   function nppiResize_16u_C4R
     (pSrc : access nppdefs_h.Npp16u;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      dstROISize : nppdefs_h.NppiSize;
      nXFactor : double;
      nYFactor : double;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:1117
   pragma Import (C, nppiResize_16u_C4R, "nppiResize_16u_C4R");

  --*
  -- * 4 channel 16-bit unsigned image resize not affecting alpha.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param dstROISize Size in pixels of the destination image.
  -- * \param nXFactor Factor by which x dimension is changed. 
  -- * \param nYFactor Factor by which y dimension is changed. 
  -- * \param eInterpolation The type of interpolation to perform resampling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref resize_error_codes
  --  

   function nppiResize_16u_AC4R
     (pSrc : access nppdefs_h.Npp16u;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      dstROISize : nppdefs_h.NppiSize;
      nXFactor : double;
      nYFactor : double;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:1137
   pragma Import (C, nppiResize_16u_AC4R, "nppiResize_16u_AC4R");

  --*
  -- * 3 channel 16-bit unsigned planar image resize.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array (host memory array containing device memory image plane pointers).
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image.
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pDst \ref destination_planar_image_pointer_array (host memory array containing device memory image plane pointers).
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param dstROISize Size in pixels of the destination image.
  -- * \param nXFactor Factor by which x dimension is changed. 
  -- * \param nYFactor Factor by which y dimension is changed. 
  -- * \param eInterpolation The type of eInterpolation to perform resampling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref resize_error_codes
  --  

   function nppiResize_16u_P3R
     (pSrc : System.Address;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : System.Address;
      nDstStep : int;
      dstROISize : nppdefs_h.NppiSize;
      nXFactor : double;
      nYFactor : double;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:1157
   pragma Import (C, nppiResize_16u_P3R, "nppiResize_16u_P3R");

  --*
  -- * 4 channel 16-bit unsigned planar image resize.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array (host memory array containing device memory image plane pointers).
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image.
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pDst \ref destination_planar_image_pointer_array (host memory array containing device memory image plane pointers).
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param dstROISize Size in pixels of the destination image.
  -- * \param nXFactor Factor by which x dimension is changed. 
  -- * \param nYFactor Factor by which y dimension is changed. 
  -- * \param eInterpolation The type of eInterpolation to perform resampling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref resize_error_codes
  --  

   function nppiResize_16u_P4R
     (pSrc : System.Address;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : System.Address;
      nDstStep : int;
      dstROISize : nppdefs_h.NppiSize;
      nXFactor : double;
      nYFactor : double;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:1177
   pragma Import (C, nppiResize_16u_P4R, "nppiResize_16u_P4R");

  --*
  -- * 1 channel 32-bit floating point image resize.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param dstROISize Size in pixels of the destination image.
  -- * \param nXFactor Factor by which x dimension is changed. 
  -- * \param nYFactor Factor by which y dimension is changed. 
  -- * \param eInterpolation The type of eInterpolation to perform resampling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref resize_error_codes
  --  

   function nppiResize_32f_C1R
     (pSrc : access nppdefs_h.Npp32f;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      dstROISize : nppdefs_h.NppiSize;
      nXFactor : double;
      nYFactor : double;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:1197
   pragma Import (C, nppiResize_32f_C1R, "nppiResize_32f_C1R");

  --*
  -- * 3 channel 32-bit floating point image resize.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param dstROISize Size in pixels of the destination image.
  -- * \param nXFactor Factor by which x dimension is changed. 
  -- * \param nYFactor Factor by which y dimension is changed. 
  -- * \param eInterpolation The type of eInterpolation to perform resampling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref resize_error_codes
  --  

   function nppiResize_32f_C3R
     (pSrc : access nppdefs_h.Npp32f;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      dstROISize : nppdefs_h.NppiSize;
      nXFactor : double;
      nYFactor : double;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:1217
   pragma Import (C, nppiResize_32f_C3R, "nppiResize_32f_C3R");

  --*
  -- * 4 channel 32-bit floating point image resize.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param dstROISize Size in pixels of the destination image.
  -- * \param nXFactor Factor by which x dimension is changed. 
  -- * \param nYFactor Factor by which y dimension is changed. 
  -- * \param eInterpolation The type of eInterpolation to perform resampling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref resize_error_codes
  --  

   function nppiResize_32f_C4R
     (pSrc : access nppdefs_h.Npp32f;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      dstROISize : nppdefs_h.NppiSize;
      nXFactor : double;
      nYFactor : double;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:1237
   pragma Import (C, nppiResize_32f_C4R, "nppiResize_32f_C4R");

  --*
  -- * 4 channel 32-bit floating point image resize not affecting alpha.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param dstROISize Size in pixels of the destination image.
  -- * \param nXFactor Factor by which x dimension is changed. 
  -- * \param nYFactor Factor by which y dimension is changed. 
  -- * \param eInterpolation The type of interpolation to perform resampling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref resize_error_codes
  --  

   function nppiResize_32f_AC4R
     (pSrc : access nppdefs_h.Npp32f;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      dstROISize : nppdefs_h.NppiSize;
      nXFactor : double;
      nYFactor : double;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:1257
   pragma Import (C, nppiResize_32f_AC4R, "nppiResize_32f_AC4R");

  --*
  -- * 3 channel 32-bit floating point planar image resize.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array (host memory array containing device memory image plane pointers).
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image.
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pDst \ref destination_planar_image_pointer_array (host memory array containing device memory image plane pointers).
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param dstROISize Size in pixels of the destination image.
  -- * \param nXFactor Factor by which x dimension is changed. 
  -- * \param nYFactor Factor by which y dimension is changed. 
  -- * \param eInterpolation The type of eInterpolation to perform resampling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref resize_error_codes
  --  

   function nppiResize_32f_P3R
     (pSrc : System.Address;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : System.Address;
      nDstStep : int;
      dstROISize : nppdefs_h.NppiSize;
      nXFactor : double;
      nYFactor : double;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:1277
   pragma Import (C, nppiResize_32f_P3R, "nppiResize_32f_P3R");

  --*
  -- * 4 channel 32-bit floating point planar image resize.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array (host memory array containing device memory image plane pointers).
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image.
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pDst \ref destination_planar_image_pointer_array (host memory array containing device memory image plane pointers).
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param dstROISize Size in pixels of the destination image.
  -- * \param nXFactor Factor by which x dimension is changed. 
  -- * \param nYFactor Factor by which y dimension is changed. 
  -- * \param eInterpolation The type of eInterpolation to perform resampling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref resize_error_codes
  --  

   function nppiResize_32f_P4R
     (pSrc : System.Address;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : System.Address;
      nDstStep : int;
      dstROISize : nppdefs_h.NppiSize;
      nXFactor : double;
      nYFactor : double;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:1297
   pragma Import (C, nppiResize_32f_P4R, "nppiResize_32f_P4R");

  --* @}  
  --* @} image_resize  
  --* @defgroup image_remap Remap
  -- *
  -- * Remap supports the following interpolation modes:
  -- *
  -- *   NPPI_INTER_NN
  -- *   NPPI_INTER_LINEAR
  -- *   NPPI_INTER_CUBIC
  -- *   NPPI_INTER_CUBIC2P_BSPLINE
  -- *   NPPI_INTER_CUBIC2P_CATMULLROM
  -- *   NPPI_INTER_CUBIC2P_B05C03
  -- *   NPPI_INTER_LANCZOS
  -- *
  -- * Remap chooses source pixels using pixel coordinates explicitely supplied in two 2D device memory image arrays pointed to by the pXMap and pYMap pointers.
  -- * The pXMap array contains the X coordinated and the pYMap array contains the Y coordinate of the corresponding source image pixel to
  -- * use as input.   These coordinates are in floating point format so fraction pixel positions can be used. The coordinates of the source
  -- * pixel to sample are determined as follows:
  -- *
  -- *   nSrcX = pxMap[nDstX, nDstY]
  -- *   nSrcY = pyMap[nDstX, nDstY]
  -- *
  -- * In the Remap functions below source image clip checking is handled as follows:
  -- *
  -- * If the source pixel fractional x and y coordinates are greater than or equal to oSizeROI.x and less than oSizeROI.x + oSizeROI.width and
  -- * greater than or equal to oSizeROI.y and less than oSizeROI.y + oSizeROI.height then the source pixel is considered to be within
  -- * the source image clip rectangle and the source image is sampled.  Otherwise the source image is not sampled and a destination pixel is not
  -- * written to the destination image. 
  -- *
  -- * \section remap_error_codes Error Codes
  -- * The remap primitives return the following error codes:
  -- *
  -- *         - ::NPP_WRONG_INTERSECTION_ROI_ERROR indicates an error condition if
  -- *           srcROIRect has no intersection with the source image.
  -- *         - ::NPP_INTERPOLATION_ERROR if eInterpolation has an illegal value.
  -- *
  -- * @{
  -- *
  --  

  --* @name Remap
  -- * Remaps images.
  -- *                                    
  -- * @{
  -- *
  --  

  --*
  -- * 1 channel 8-bit unsigned image remap.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image.
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pXMap Device memory pointer to 2D image array of X coordinate values to be used when sampling source image. 
  -- * \param nXMapStep pXMap image array line step in bytes.
  -- * \param pYMap Device memory pointer to 2D image array of Y coordinate values to be used when sampling source image. 
  -- * \param nYMapStep pYMap image array line step in bytes.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Region of interest size in the destination image.
  -- * \param eInterpolation The type of eInterpolation to perform resampling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref remap_error_codes
  --  

   function nppiRemap_8u_C1R
     (pSrc : access nppdefs_h.Npp8u;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pXMap : access nppdefs_h.Npp32f;
      nXMapStep : int;
      pYMap : access nppdefs_h.Npp32f;
      nYMapStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:1368
   pragma Import (C, nppiRemap_8u_C1R, "nppiRemap_8u_C1R");

  --*
  -- * 3 channel 8-bit unsigned image remap.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image.
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pXMap Device memory pointer to 2D image array of X coordinate values to be used when sampling source image. 
  -- * \param nXMapStep pXMap image array line step in bytes.
  -- * \param pYMap Device memory pointer to 2D image array of Y coordinate values to be used when sampling source image. 
  -- * \param nYMapStep pYMap image array line step in bytes.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Region of interest size in the destination image.
  -- * \param eInterpolation The type of eInterpolation to perform resampling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref remap_error_codes
  --  

   function nppiRemap_8u_C3R
     (pSrc : access nppdefs_h.Npp8u;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pXMap : access nppdefs_h.Npp32f;
      nXMapStep : int;
      pYMap : access nppdefs_h.Npp32f;
      nYMapStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:1390
   pragma Import (C, nppiRemap_8u_C3R, "nppiRemap_8u_C3R");

  --*
  -- * 4 channel 8-bit unsigned image remap.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image.
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pXMap Device memory pointer to 2D image array of X coordinate values to be used when sampling source image. 
  -- * \param nXMapStep pXMap image array line step in bytes.
  -- * \param pYMap Device memory pointer to 2D image array of Y coordinate values to be used when sampling source image. 
  -- * \param nYMapStep pYMap image array line step in bytes.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Region of interest size in the destination image.
  -- * \param eInterpolation The type of eInterpolation to perform resampling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref remap_error_codes
  --  

   function nppiRemap_8u_C4R
     (pSrc : access nppdefs_h.Npp8u;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pXMap : access nppdefs_h.Npp32f;
      nXMapStep : int;
      pYMap : access nppdefs_h.Npp32f;
      nYMapStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:1412
   pragma Import (C, nppiRemap_8u_C4R, "nppiRemap_8u_C4R");

  --*
  -- * 4 channel 8-bit unsigned image remap not affecting alpha.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image.
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pXMap Device memory pointer to 2D image array of X coordinate values to be used when sampling source image. 
  -- * \param nXMapStep pXMap image array line step in bytes.
  -- * \param pYMap Device memory pointer to 2D image array of Y coordinate values to be used when sampling source image. 
  -- * \param nYMapStep pYMap image array line step in bytes.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Region of interest size in the destination image.
  -- * \param eInterpolation The type of interpolation to perform resampling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref remap_error_codes
  --  

   function nppiRemap_8u_AC4R
     (pSrc : access nppdefs_h.Npp8u;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pXMap : access nppdefs_h.Npp32f;
      nXMapStep : int;
      pYMap : access nppdefs_h.Npp32f;
      nYMapStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:1434
   pragma Import (C, nppiRemap_8u_AC4R, "nppiRemap_8u_AC4R");

  --*
  -- * 3 channel 8-bit unsigned planar image remap.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image.
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pXMap Device memory pointer to 2D image array of X coordinate values to be used when sampling source image. 
  -- * \param nXMapStep pXMap image array line step in bytes.
  -- * \param pYMap Device memory pointer to 2D image array of Y coordinate values to be used when sampling source image. 
  -- * \param nYMapStep pYMap image array line step in bytes.
  -- * \param pDst \ref destination_planar_image_pointer_array.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Region of interest size in the destination image.
  -- * \param eInterpolation The type of eInterpolation to perform resampling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref remap_error_codes
  --  

   function nppiRemap_8u_P3R
     (pSrc : System.Address;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pXMap : access nppdefs_h.Npp32f;
      nXMapStep : int;
      pYMap : access nppdefs_h.Npp32f;
      nYMapStep : int;
      pDst : System.Address;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:1456
   pragma Import (C, nppiRemap_8u_P3R, "nppiRemap_8u_P3R");

  --*
  -- * 4 channel 8-bit unsigned planar image remap.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image.
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pXMap Device memory pointer to 2D image array of X coordinate values to be used when sampling source image. 
  -- * \param nXMapStep pXMap image array line step in bytes.
  -- * \param pYMap Device memory pointer to 2D image array of Y coordinate values to be used when sampling source image. 
  -- * \param nYMapStep pYMap image array line step in bytes.
  -- * \param pDst \ref destination_planar_image_pointer_array.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Region of interest size in the destination image.
  -- * \param eInterpolation The type of eInterpolation to perform resampling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref remap_error_codes
  --  

   function nppiRemap_8u_P4R
     (pSrc : System.Address;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pXMap : access nppdefs_h.Npp32f;
      nXMapStep : int;
      pYMap : access nppdefs_h.Npp32f;
      nYMapStep : int;
      pDst : System.Address;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:1478
   pragma Import (C, nppiRemap_8u_P4R, "nppiRemap_8u_P4R");

  --*
  -- * 1 channel 16-bit unsigned image remap.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image.
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pXMap Device memory pointer to 2D image array of X coordinate values to be used when sampling source image. 
  -- * \param nXMapStep pXMap image array line step in bytes.
  -- * \param pYMap Device memory pointer to 2D image array of Y coordinate values to be used when sampling source image. 
  -- * \param nYMapStep pYMap image array line step in bytes.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Region of interest size in the destination image.
  -- * \param eInterpolation The type of eInterpolation to perform resampling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref remap_error_codes
  --  

   function nppiRemap_16u_C1R
     (pSrc : access nppdefs_h.Npp16u;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pXMap : access nppdefs_h.Npp32f;
      nXMapStep : int;
      pYMap : access nppdefs_h.Npp32f;
      nYMapStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:1500
   pragma Import (C, nppiRemap_16u_C1R, "nppiRemap_16u_C1R");

  --*
  -- * 3 channel 16-bit unsigned image remap.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image.
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pXMap Device memory pointer to 2D image array of X coordinate values to be used when sampling source image. 
  -- * \param nXMapStep pXMap image array line step in bytes.
  -- * \param pYMap Device memory pointer to 2D image array of Y coordinate values to be used when sampling source image. 
  -- * \param nYMapStep pYMap image array line step in bytes.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Region of interest size in the destination image.
  -- * \param eInterpolation The type of eInterpolation to perform resampling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref remap_error_codes
  --  

   function nppiRemap_16u_C3R
     (pSrc : access nppdefs_h.Npp16u;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pXMap : access nppdefs_h.Npp32f;
      nXMapStep : int;
      pYMap : access nppdefs_h.Npp32f;
      nYMapStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:1522
   pragma Import (C, nppiRemap_16u_C3R, "nppiRemap_16u_C3R");

  --*
  -- * 4 channel 16-bit unsigned image remap.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image.
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pXMap Device memory pointer to 2D image array of X coordinate values to be used when sampling source image. 
  -- * \param nXMapStep pXMap image array line step in bytes.
  -- * \param pYMap Device memory pointer to 2D image array of Y coordinate values to be used when sampling source image. 
  -- * \param nYMapStep pYMap image array line step in bytes.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Region of interest size in the destination image.
  -- * \param eInterpolation The type of eInterpolation to perform resampling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref remap_error_codes
  --  

   function nppiRemap_16u_C4R
     (pSrc : access nppdefs_h.Npp16u;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pXMap : access nppdefs_h.Npp32f;
      nXMapStep : int;
      pYMap : access nppdefs_h.Npp32f;
      nYMapStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:1544
   pragma Import (C, nppiRemap_16u_C4R, "nppiRemap_16u_C4R");

  --*
  -- * 4 channel 16-bit unsigned image remap not affecting alpha.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image.
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pXMap Device memory pointer to 2D image array of X coordinate values to be used when sampling source image. 
  -- * \param nXMapStep pXMap image array line step in bytes.
  -- * \param pYMap Device memory pointer to 2D image array of Y coordinate values to be used when sampling source image. 
  -- * \param nYMapStep pYMap image array line step in bytes.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Region of interest size in the destination image.
  -- * \param eInterpolation The type of interpolation to perform resampling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref remap_error_codes
  --  

   function nppiRemap_16u_AC4R
     (pSrc : access nppdefs_h.Npp16u;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pXMap : access nppdefs_h.Npp32f;
      nXMapStep : int;
      pYMap : access nppdefs_h.Npp32f;
      nYMapStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:1566
   pragma Import (C, nppiRemap_16u_AC4R, "nppiRemap_16u_AC4R");

  --*
  -- * 3 channel 16-bit unsigned planar image remap.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image.
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pXMap Device memory pointer to 2D image array of X coordinate values to be used when sampling source image. 
  -- * \param nXMapStep pXMap image array line step in bytes.
  -- * \param pYMap Device memory pointer to 2D image array of Y coordinate values to be used when sampling source image. 
  -- * \param nYMapStep pYMap image array line step in bytes.
  -- * \param pDst \ref destination_planar_image_pointer_array.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Region of interest size in the destination image.
  -- * \param eInterpolation The type of eInterpolation to perform resampling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref remap_error_codes
  --  

   function nppiRemap_16u_P3R
     (pSrc : System.Address;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pXMap : access nppdefs_h.Npp32f;
      nXMapStep : int;
      pYMap : access nppdefs_h.Npp32f;
      nYMapStep : int;
      pDst : System.Address;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:1588
   pragma Import (C, nppiRemap_16u_P3R, "nppiRemap_16u_P3R");

  --*
  -- * 4 channel 16-bit unsigned planar image remap.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image.
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pXMap Device memory pointer to 2D image array of X coordinate values to be used when sampling source image. 
  -- * \param nXMapStep pXMap image array line step in bytes.
  -- * \param pYMap Device memory pointer to 2D image array of Y coordinate values to be used when sampling source image. 
  -- * \param nYMapStep pYMap image array line step in bytes.
  -- * \param pDst \ref destination_planar_image_pointer_array.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Region of interest size in the destination image.
  -- * \param eInterpolation The type of eInterpolation to perform resampling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref remap_error_codes
  --  

   function nppiRemap_16u_P4R
     (pSrc : System.Address;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pXMap : access nppdefs_h.Npp32f;
      nXMapStep : int;
      pYMap : access nppdefs_h.Npp32f;
      nYMapStep : int;
      pDst : System.Address;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:1610
   pragma Import (C, nppiRemap_16u_P4R, "nppiRemap_16u_P4R");

  --*
  -- * 1 channel 16-bit signed image remap.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image.
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pXMap Device memory pointer to 2D image array of X coordinate values to be used when sampling source image. 
  -- * \param nXMapStep pXMap image array line step in bytes.
  -- * \param pYMap Device memory pointer to 2D image array of Y coordinate values to be used when sampling source image. 
  -- * \param nYMapStep pYMap image array line step in bytes.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Region of interest size in the destination image.
  -- * \param eInterpolation The type of eInterpolation to perform resampling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref remap_error_codes
  --  

   function nppiRemap_16s_C1R
     (pSrc : access nppdefs_h.Npp16s;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pXMap : access nppdefs_h.Npp32f;
      nXMapStep : int;
      pYMap : access nppdefs_h.Npp32f;
      nYMapStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:1632
   pragma Import (C, nppiRemap_16s_C1R, "nppiRemap_16s_C1R");

  --*
  -- * 3 channel 16-bit signed image remap.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image.
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pXMap Device memory pointer to 2D image array of X coordinate values to be used when sampling source image. 
  -- * \param nXMapStep pXMap image array line step in bytes.
  -- * \param pYMap Device memory pointer to 2D image array of Y coordinate values to be used when sampling source image. 
  -- * \param nYMapStep pYMap image array line step in bytes.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Region of interest size in the destination image.
  -- * \param eInterpolation The type of eInterpolation to perform resampling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref remap_error_codes
  --  

   function nppiRemap_16s_C3R
     (pSrc : access nppdefs_h.Npp16s;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pXMap : access nppdefs_h.Npp32f;
      nXMapStep : int;
      pYMap : access nppdefs_h.Npp32f;
      nYMapStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:1654
   pragma Import (C, nppiRemap_16s_C3R, "nppiRemap_16s_C3R");

  --*
  -- * 4 channel 16-bit signed image remap.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image.
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pXMap Device memory pointer to 2D image array of X coordinate values to be used when sampling source image. 
  -- * \param nXMapStep pXMap image array line step in bytes.
  -- * \param pYMap Device memory pointer to 2D image array of Y coordinate values to be used when sampling source image. 
  -- * \param nYMapStep pYMap image array line step in bytes.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Region of interest size in the destination image.
  -- * \param eInterpolation The type of eInterpolation to perform resampling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref remap_error_codes
  --  

   function nppiRemap_16s_C4R
     (pSrc : access nppdefs_h.Npp16s;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pXMap : access nppdefs_h.Npp32f;
      nXMapStep : int;
      pYMap : access nppdefs_h.Npp32f;
      nYMapStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:1676
   pragma Import (C, nppiRemap_16s_C4R, "nppiRemap_16s_C4R");

  --*
  -- * 4 channel 16-bit signed image remap not affecting alpha.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image.
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pXMap Device memory pointer to 2D image array of X coordinate values to be used when sampling source image. 
  -- * \param nXMapStep pXMap image array line step in bytes.
  -- * \param pYMap Device memory pointer to 2D image array of Y coordinate values to be used when sampling source image. 
  -- * \param nYMapStep pYMap image array line step in bytes.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Region of interest size in the destination image.
  -- * \param eInterpolation The type of interpolation to perform resampling
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref remap_error_codes
  --  

   function nppiRemap_16s_AC4R
     (pSrc : access nppdefs_h.Npp16s;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pXMap : access nppdefs_h.Npp32f;
      nXMapStep : int;
      pYMap : access nppdefs_h.Npp32f;
      nYMapStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:1698
   pragma Import (C, nppiRemap_16s_AC4R, "nppiRemap_16s_AC4R");

  --*
  -- * 3 channel 16-bit signed planar image remap.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image.
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pXMap Device memory pointer to 2D image array of X coordinate values to be used when sampling source image. 
  -- * \param nXMapStep pXMap image array line step in bytes.
  -- * \param pYMap Device memory pointer to 2D image array of Y coordinate values to be used when sampling source image. 
  -- * \param nYMapStep pYMap image array line step in bytes.
  -- * \param pDst \ref destination_planar_image_pointer_array.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Region of interest size in the destination image.
  -- * \param eInterpolation The type of eInterpolation to perform resampling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref remap_error_codes
  --  

   function nppiRemap_16s_P3R
     (pSrc : System.Address;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pXMap : access nppdefs_h.Npp32f;
      nXMapStep : int;
      pYMap : access nppdefs_h.Npp32f;
      nYMapStep : int;
      pDst : System.Address;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:1720
   pragma Import (C, nppiRemap_16s_P3R, "nppiRemap_16s_P3R");

  --*
  -- * 4 channel 16-bit signed planar image remap.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image.
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pXMap Device memory pointer to 2D image array of X coordinate values to be used when sampling source image. 
  -- * \param nXMapStep pXMap image array line step in bytes.
  -- * \param pYMap Device memory pointer to 2D image array of Y coordinate values to be used when sampling source image. 
  -- * \param nYMapStep pYMap image array line step in bytes.
  -- * \param pDst \ref destination_planar_image_pointer_array.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Region of interest size in the destination image.
  -- * \param eInterpolation The type of eInterpolation to perform resampling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref remap_error_codes
  --  

   function nppiRemap_16s_P4R
     (pSrc : System.Address;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pXMap : access nppdefs_h.Npp32f;
      nXMapStep : int;
      pYMap : access nppdefs_h.Npp32f;
      nYMapStep : int;
      pDst : System.Address;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:1742
   pragma Import (C, nppiRemap_16s_P4R, "nppiRemap_16s_P4R");

  --*
  -- * 1 channel 32-bit floating point image remap.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image.
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pXMap Device memory pointer to 2D image array of X coordinate values to be used when sampling source image. 
  -- * \param nXMapStep pXMap image array line step in bytes.
  -- * \param pYMap Device memory pointer to 2D image array of Y coordinate values to be used when sampling source image. 
  -- * \param nYMapStep pYMap image array line step in bytes.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Region of interest size in the destination image.
  -- * \param eInterpolation The type of eInterpolation to perform resampling
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref remap_error_codes
  --  

   function nppiRemap_32f_C1R
     (pSrc : access nppdefs_h.Npp32f;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pXMap : access nppdefs_h.Npp32f;
      nXMapStep : int;
      pYMap : access nppdefs_h.Npp32f;
      nYMapStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:1764
   pragma Import (C, nppiRemap_32f_C1R, "nppiRemap_32f_C1R");

  --*
  -- * 3 channel 32-bit floating point image remap.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image.
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pXMap Device memory pointer to 2D image array of X coordinate values to be used when sampling source image. 
  -- * \param nXMapStep pXMap image array line step in bytes.
  -- * \param pYMap Device memory pointer to 2D image array of Y coordinate values to be used when sampling source image. 
  -- * \param nYMapStep pYMap image array line step in bytes.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Region of interest size in the destination image.
  -- * \param eInterpolation The type of eInterpolation to perform resampling
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref remap_error_codes
  --  

   function nppiRemap_32f_C3R
     (pSrc : access nppdefs_h.Npp32f;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pXMap : access nppdefs_h.Npp32f;
      nXMapStep : int;
      pYMap : access nppdefs_h.Npp32f;
      nYMapStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:1786
   pragma Import (C, nppiRemap_32f_C3R, "nppiRemap_32f_C3R");

  --*
  -- * 4 channel 32-bit floating point image remap.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image.
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pXMap Device memory pointer to 2D image array of X coordinate values to be used when sampling source image. 
  -- * \param nXMapStep pXMap image array line step in bytes.
  -- * \param pYMap Device memory pointer to 2D image array of Y coordinate values to be used when sampling source image. 
  -- * \param nYMapStep pYMap image array line step in bytes.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Region of interest size in the destination image.
  -- * \param eInterpolation The type of eInterpolation to perform resampling
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref remap_error_codes
  --  

   function nppiRemap_32f_C4R
     (pSrc : access nppdefs_h.Npp32f;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pXMap : access nppdefs_h.Npp32f;
      nXMapStep : int;
      pYMap : access nppdefs_h.Npp32f;
      nYMapStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:1808
   pragma Import (C, nppiRemap_32f_C4R, "nppiRemap_32f_C4R");

  --*
  -- * 4 channel 32-bit floating point image remap not affecting alpha.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image.
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pXMap Device memory pointer to 2D image array of X coordinate values to be used when sampling source image. 
  -- * \param nXMapStep pXMap image array line step in bytes.
  -- * \param pYMap Device memory pointer to 2D image array of Y coordinate values to be used when sampling source image. 
  -- * \param nYMapStep pYMap image array line step in bytes.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Region of interest size in the destination image.
  -- * \param eInterpolation The type of interpolation to perform resampling
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref remap_error_codes
  --  

   function nppiRemap_32f_AC4R
     (pSrc : access nppdefs_h.Npp32f;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pXMap : access nppdefs_h.Npp32f;
      nXMapStep : int;
      pYMap : access nppdefs_h.Npp32f;
      nYMapStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:1830
   pragma Import (C, nppiRemap_32f_AC4R, "nppiRemap_32f_AC4R");

  --*
  -- * 3 channel 32-bit floating point planar image remap.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image.
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pXMap Device memory pointer to 2D image array of X coordinate values to be used when sampling source image. 
  -- * \param nXMapStep pXMap image array line step in bytes.
  -- * \param pYMap Device memory pointer to 2D image array of Y coordinate values to be used when sampling source image. 
  -- * \param nYMapStep pYMap image array line step in bytes.
  -- * \param pDst \ref destination_planar_image_pointer_array.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Region of interest size in the destination image.
  -- * \param eInterpolation The type of eInterpolation to perform resampling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref remap_error_codes
  --  

   function nppiRemap_32f_P3R
     (pSrc : System.Address;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pXMap : access nppdefs_h.Npp32f;
      nXMapStep : int;
      pYMap : access nppdefs_h.Npp32f;
      nYMapStep : int;
      pDst : System.Address;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:1852
   pragma Import (C, nppiRemap_32f_P3R, "nppiRemap_32f_P3R");

  --*
  -- * 4 channel 32-bit floating point planar image remap.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image.
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pXMap Device memory pointer to 2D image array of X coordinate values to be used when sampling source image. 
  -- * \param nXMapStep pXMap image array line step in bytes.
  -- * \param pYMap Device memory pointer to 2D image array of Y coordinate values to be used when sampling source image. 
  -- * \param nYMapStep pYMap image array line step in bytes.
  -- * \param pDst \ref destination_planar_image_pointer_array.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Region of interest size in the destination image.
  -- * \param eInterpolation The type of eInterpolation to perform resampling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref remap_error_codes
  --  

   function nppiRemap_32f_P4R
     (pSrc : System.Address;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pXMap : access nppdefs_h.Npp32f;
      nXMapStep : int;
      pYMap : access nppdefs_h.Npp32f;
      nYMapStep : int;
      pDst : System.Address;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:1874
   pragma Import (C, nppiRemap_32f_P4R, "nppiRemap_32f_P4R");

  --*
  -- * 1 channel 64-bit floating point image remap.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image.
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pXMap Device memory pointer to 2D image array of X coordinate values to be used when sampling source image. 
  -- * \param nXMapStep pXMap image array line step in bytes.
  -- * \param pYMap Device memory pointer to 2D image array of Y coordinate values to be used when sampling source image. 
  -- * \param nYMapStep pYMap image array line step in bytes.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Region of interest size in the destination image.
  -- * \param eInterpolation The type of eInterpolation to perform resampling
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref remap_error_codes
  --  

   function nppiRemap_64f_C1R
     (pSrc : access nppdefs_h.Npp64f;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pXMap : access nppdefs_h.Npp64f;
      nXMapStep : int;
      pYMap : access nppdefs_h.Npp64f;
      nYMapStep : int;
      pDst : access nppdefs_h.Npp64f;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:1896
   pragma Import (C, nppiRemap_64f_C1R, "nppiRemap_64f_C1R");

  --*
  -- * 3 channel 64-bit floating point image remap.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image.
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pXMap Device memory pointer to 2D image array of X coordinate values to be used when sampling source image. 
  -- * \param nXMapStep pXMap image array line step in bytes.
  -- * \param pYMap Device memory pointer to 2D image array of Y coordinate values to be used when sampling source image. 
  -- * \param nYMapStep pYMap image array line step in bytes.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Region of interest size in the destination image.
  -- * \param eInterpolation The type of eInterpolation to perform resampling
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref remap_error_codes
  --  

   function nppiRemap_64f_C3R
     (pSrc : access nppdefs_h.Npp64f;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pXMap : access nppdefs_h.Npp64f;
      nXMapStep : int;
      pYMap : access nppdefs_h.Npp64f;
      nYMapStep : int;
      pDst : access nppdefs_h.Npp64f;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:1918
   pragma Import (C, nppiRemap_64f_C3R, "nppiRemap_64f_C3R");

  --*
  -- * 4 channel 64-bit floating point image remap.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image.
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pXMap Device memory pointer to 2D image array of X coordinate values to be used when sampling source image. 
  -- * \param nXMapStep pXMap image array line step in bytes.
  -- * \param pYMap Device memory pointer to 2D image array of Y coordinate values to be used when sampling source image. 
  -- * \param nYMapStep pYMap image array line step in bytes.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Region of interest size in the destination image.
  -- * \param eInterpolation The type of eInterpolation to perform resampling
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref remap_error_codes
  --  

   function nppiRemap_64f_C4R
     (pSrc : access nppdefs_h.Npp64f;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pXMap : access nppdefs_h.Npp64f;
      nXMapStep : int;
      pYMap : access nppdefs_h.Npp64f;
      nYMapStep : int;
      pDst : access nppdefs_h.Npp64f;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:1940
   pragma Import (C, nppiRemap_64f_C4R, "nppiRemap_64f_C4R");

  --*
  -- * 4 channel 64-bit floating point image remap not affecting alpha.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image.
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pXMap Device memory pointer to 2D image array of X coordinate values to be used when sampling source image. 
  -- * \param nXMapStep pXMap image array line step in bytes.
  -- * \param pYMap Device memory pointer to 2D image array of Y coordinate values to be used when sampling source image. 
  -- * \param nYMapStep pYMap image array line step in bytes.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Region of interest size in the destination image.
  -- * \param eInterpolation The type of interpolation to perform resampling
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref remap_error_codes
  --  

   function nppiRemap_64f_AC4R
     (pSrc : access nppdefs_h.Npp64f;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pXMap : access nppdefs_h.Npp64f;
      nXMapStep : int;
      pYMap : access nppdefs_h.Npp64f;
      nYMapStep : int;
      pDst : access nppdefs_h.Npp64f;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:1962
   pragma Import (C, nppiRemap_64f_AC4R, "nppiRemap_64f_AC4R");

  --*
  -- * 3 channel 64-bit floating point planar image remap.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image.
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pXMap Device memory pointer to 2D image array of X coordinate values to be used when sampling source image. 
  -- * \param nXMapStep pXMap image array line step in bytes.
  -- * \param pYMap Device memory pointer to 2D image array of Y coordinate values to be used when sampling source image. 
  -- * \param nYMapStep pYMap image array line step in bytes.
  -- * \param pDst \ref destination_planar_image_pointer_array.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Region of interest size in the destination image.
  -- * \param eInterpolation The type of eInterpolation to perform resampling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref remap_error_codes
  --  

   function nppiRemap_64f_P3R
     (pSrc : System.Address;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pXMap : access nppdefs_h.Npp64f;
      nXMapStep : int;
      pYMap : access nppdefs_h.Npp64f;
      nYMapStep : int;
      pDst : System.Address;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:1984
   pragma Import (C, nppiRemap_64f_P3R, "nppiRemap_64f_P3R");

  --*
  -- * 4 channel 64-bit floating point planar image remap.
  -- *
  -- * \param pSrc \ref source_planar_image_pointer_array.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image.
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pXMap Device memory pointer to 2D image array of X coordinate values to be used when sampling source image. 
  -- * \param nXMapStep pXMap image array line step in bytes.
  -- * \param pYMap Device memory pointer to 2D image array of Y coordinate values to be used when sampling source image. 
  -- * \param nYMapStep pYMap image array line step in bytes.
  -- * \param pDst \ref destination_planar_image_pointer_array.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Region of interest size in the destination image.
  -- * \param eInterpolation The type of eInterpolation to perform resampling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref remap_error_codes
  --  

   function nppiRemap_64f_P4R
     (pSrc : System.Address;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pXMap : access nppdefs_h.Npp64f;
      nXMapStep : int;
      pYMap : access nppdefs_h.Npp64f;
      nYMapStep : int;
      pDst : System.Address;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:2006
   pragma Import (C, nppiRemap_64f_P4R, "nppiRemap_64f_P4R");

  --* @}  
  --* @} image_remap  
  --* @defgroup image_rotate Rotate
  -- *
  -- *  Rotates an image around the origin (0,0) and then shifts it.
  -- *
  -- * \section rotate_error_codes Rotate Error Codes
  -- * - ::NPP_INTERPOLATION_ERROR if eInterpolation has an illegal value.
  -- * - ::NPP_RECTANGLE_ERROR Indicates an error condition if width or height of
  -- *   the intersection of the oSrcROI and source image is less than or
  -- *   equal to 1.
  -- * - ::NPP_WRONG_INTERSECTION_ROI_ERROR indicates an error condition if
  -- *   srcROIRect has no intersection with the source image.
  -- * - ::NPP_WRONG_INTERSECTION_QUAD_WARNING indicates a warning that no
  -- *   operation is performed if the transformed source ROI does not
  -- *   intersect the destination ROI.
  -- *
  -- * @{
  -- *
  --  

  --* @name Utility Functions
  -- *
  -- * @{
  -- *
  --  

  --*
  -- * Compute shape of rotated image.
  -- * 
  -- * \param oSrcROI Region-of-interest of the source image.
  -- * \param aQuad Array of 2D points. These points are the locations of the corners
  -- *      of the rotated ROI. 
  -- * \param nAngle The rotation nAngle.
  -- * \param nShiftX Post-rotation shift in x-direction
  -- * \param nShiftY Post-rotation shift in y-direction
  -- * \return \ref roi_error_codes.
  --  

   function nppiGetRotateQuad
     (oSrcROI : nppdefs_h.NppiRect;
      aQuad : System.Address;
      nAngle : double;
      nShiftX : double;
      nShiftY : double) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:2051
   pragma Import (C, nppiGetRotateQuad, "nppiGetRotateQuad");

  --*
  -- * Compute bounding-box of rotated image.
  -- * \param oSrcROI Region-of-interest of the source image.
  -- * \param aBoundingBox Two 2D points representing the bounding-box of the rotated image. All four points
  -- *      from nppiGetRotateQuad are contained inside the axis-aligned rectangle spanned by the the two
  -- *      points of this bounding box.
  -- * \param nAngle The rotation angle.
  -- * \param nShiftX Post-rotation shift in x-direction.
  -- * \param nShiftY Post-rotation shift in y-direction.
  -- *
  -- * \return \ref roi_error_codes.
  --  

   function nppiGetRotateBound
     (oSrcROI : nppdefs_h.NppiRect;
      aBoundingBox : System.Address;
      nAngle : double;
      nShiftX : double;
      nShiftY : double) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:2066
   pragma Import (C, nppiGetRotateBound, "nppiGetRotateBound");

  --* @} Utility Functions  
  --* @name Rotate
  -- *
  -- * @{
  -- *
  --  

  --*
  -- * 8-bit unsigned image rotate.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Region of interest in the destination image.
  -- * \param nAngle The angle of rotation in degrees.
  -- * \param nShiftX Shift along horizontal axis 
  -- * \param nShiftY Shift along vertical axis 
  -- * \param eInterpolation The type of interpolation to perform resampling
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref rotate_error_codes
  --  

   function nppiRotate_8u_C1R
     (pSrc : access nppdefs_h.Npp8u;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      nAngle : double;
      nShiftX : double;
      nShiftY : double;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:2093
   pragma Import (C, nppiRotate_8u_C1R, "nppiRotate_8u_C1R");

  --*
  -- * 3 channel 8-bit unsigned image rotate.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Region of interest in the destination image.
  -- * \param nAngle The angle of rotation in degrees.
  -- * \param nShiftX Shift along horizontal axis 
  -- * \param nShiftY Shift along vertical axis 
  -- * \param eInterpolation The type of interpolation to perform resampling
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref rotate_error_codes
  --  

   function nppiRotate_8u_C3R
     (pSrc : access nppdefs_h.Npp8u;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      nAngle : double;
      nShiftX : double;
      nShiftY : double;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:2114
   pragma Import (C, nppiRotate_8u_C3R, "nppiRotate_8u_C3R");

  --*
  -- * 4 channel 8-bit unsigned image rotate.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Region of interest in the destination image.
  -- * \param nAngle The angle of rotation in degrees.
  -- * \param nShiftX Shift along horizontal axis 
  -- * \param nShiftY Shift along vertical axis 
  -- * \param eInterpolation The type of interpolation to perform resampling
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref rotate_error_codes
  --  

   function nppiRotate_8u_C4R
     (pSrc : access nppdefs_h.Npp8u;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      nAngle : double;
      nShiftX : double;
      nShiftY : double;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:2135
   pragma Import (C, nppiRotate_8u_C4R, "nppiRotate_8u_C4R");

  --*
  -- * 4 channel 8-bit unsigned image rotate ignoring alpha channel.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Region of interest in the destination image.
  -- * \param nAngle The angle of rotation in degrees.
  -- * \param nShiftX Shift along horizontal axis 
  -- * \param nShiftY Shift along vertical axis 
  -- * \param eInterpolation The type of interpolation to perform resampling
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref rotate_error_codes
  --  

   function nppiRotate_8u_AC4R
     (pSrc : access nppdefs_h.Npp8u;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      nAngle : double;
      nShiftX : double;
      nShiftY : double;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:2156
   pragma Import (C, nppiRotate_8u_AC4R, "nppiRotate_8u_AC4R");

  --*
  -- * 16-bit unsigned image rotate.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Region of interest in the destination image.
  -- * \param nAngle The angle of rotation in degrees.
  -- * \param nShiftX Shift along horizontal axis 
  -- * \param nShiftY Shift along vertical axis 
  -- * \param eInterpolation The type of interpolation to perform resampling
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref rotate_error_codes
  --  

   function nppiRotate_16u_C1R
     (pSrc : access nppdefs_h.Npp16u;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      nAngle : double;
      nShiftX : double;
      nShiftY : double;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:2177
   pragma Import (C, nppiRotate_16u_C1R, "nppiRotate_16u_C1R");

  --*
  -- * 3 channel 16-bit unsigned image rotate.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Region of interest in the destination image.
  -- * \param nAngle The angle of rotation in degrees.
  -- * \param nShiftX Shift along horizontal axis 
  -- * \param nShiftY Shift along vertical axis 
  -- * \param eInterpolation The type of interpolation to perform resampling
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref rotate_error_codes
  --  

   function nppiRotate_16u_C3R
     (pSrc : access nppdefs_h.Npp16u;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      nAngle : double;
      nShiftX : double;
      nShiftY : double;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:2198
   pragma Import (C, nppiRotate_16u_C3R, "nppiRotate_16u_C3R");

  --*
  -- * 4 channel 16-bit unsigned image rotate.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Region of interest in the destination image.
  -- * \param nAngle The angle of rotation in degrees.
  -- * \param nShiftX Shift along horizontal axis 
  -- * \param nShiftY Shift along vertical axis 
  -- * \param eInterpolation The type of interpolation to perform resampling
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref rotate_error_codes
  --  

   function nppiRotate_16u_C4R
     (pSrc : access nppdefs_h.Npp16u;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      nAngle : double;
      nShiftX : double;
      nShiftY : double;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:2219
   pragma Import (C, nppiRotate_16u_C4R, "nppiRotate_16u_C4R");

  --*
  -- * 4 channel 16-bit unsigned image rotate ignoring alpha channel.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Region of interest in the destination image.
  -- * \param nAngle The angle of rotation in degrees.
  -- * \param nShiftX Shift along horizontal axis 
  -- * \param nShiftY Shift along vertical axis 
  -- * \param eInterpolation The type of interpolation to perform resampling
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref rotate_error_codes
  --  

   function nppiRotate_16u_AC4R
     (pSrc : access nppdefs_h.Npp16u;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      nAngle : double;
      nShiftX : double;
      nShiftY : double;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:2240
   pragma Import (C, nppiRotate_16u_AC4R, "nppiRotate_16u_AC4R");

  --*
  -- * 32-bit float image rotate.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Region of interest in the destination image.
  -- * \param nAngle The angle of rotation in degrees.
  -- * \param nShiftX Shift along horizontal axis 
  -- * \param nShiftY Shift along vertical axis 
  -- * \param eInterpolation The type of interpolation to perform resampling
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref rotate_error_codes
  --  

   function nppiRotate_32f_C1R
     (pSrc : access nppdefs_h.Npp32f;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      nAngle : double;
      nShiftX : double;
      nShiftY : double;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:2261
   pragma Import (C, nppiRotate_32f_C1R, "nppiRotate_32f_C1R");

  --*
  -- * 3 channel 32-bit float image rotate.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Region of interest in the destination image.
  -- * \param nAngle The angle of rotation in degrees.
  -- * \param nShiftX Shift along horizontal axis 
  -- * \param nShiftY Shift along vertical axis 
  -- * \param eInterpolation The type of interpolation to perform resampling
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref rotate_error_codes
  --  

   function nppiRotate_32f_C3R
     (pSrc : access nppdefs_h.Npp32f;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      nAngle : double;
      nShiftX : double;
      nShiftY : double;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:2282
   pragma Import (C, nppiRotate_32f_C3R, "nppiRotate_32f_C3R");

  --*
  -- * 4 channel 32-bit float image rotate.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Region of interest in the destination image.
  -- * \param nAngle The angle of rotation in degrees.
  -- * \param nShiftX Shift along horizontal axis 
  -- * \param nShiftY Shift along vertical axis 
  -- * \param eInterpolation The type of interpolation to perform resampling
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref rotate_error_codes
  --  

   function nppiRotate_32f_C4R
     (pSrc : access nppdefs_h.Npp32f;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      nAngle : double;
      nShiftX : double;
      nShiftY : double;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:2303
   pragma Import (C, nppiRotate_32f_C4R, "nppiRotate_32f_C4R");

  --*
  -- * 4 channel 32-bit float image rotate ignoring alpha channel.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Size in pixels of the source image
  -- * \param oSrcROI Region of interest in the source image.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Region of interest in the destination image.
  -- * \param nAngle The angle of rotation in degrees.
  -- * \param nShiftX Shift along horizontal axis 
  -- * \param nShiftY Shift along vertical axis 
  -- * \param eInterpolation The type of interpolation to perform resampling
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref rotate_error_codes
  --  

   function nppiRotate_32f_AC4R
     (pSrc : access nppdefs_h.Npp32f;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      nAngle : double;
      nShiftX : double;
      nShiftY : double;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:2324
   pragma Import (C, nppiRotate_32f_AC4R, "nppiRotate_32f_AC4R");

  --* @}  
  --* @} image_rotate  
  --* @defgroup image_mirror Mirror
  -- * \section mirror_error_codes Mirror Error Codes
  -- *         - ::NPP_MIRROR_FLIP_ERR if flip has an illegal value.
  -- *
  -- * @{
  -- *
  --  

  --* @name Mirror
  -- *  Mirrors images horizontally, vertically and diagonally.
  -- *
  -- * @{
  -- *
  --  

  --*
  -- * 1 channel 8-bit unsigned image mirror.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oROI \ref roi_specification.
  -- * \param flip Specifies the axis about which the image is to be mirrored.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref mirror_error_codes
  --  

   function nppiMirror_8u_C1R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oROI : nppdefs_h.NppiSize;
      flip : nppdefs_h.NppiAxis) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:2358
   pragma Import (C, nppiMirror_8u_C1R, "nppiMirror_8u_C1R");

  --*
  -- * 1 channel 8-bit unsigned in place image mirror.
  -- *
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oROI \ref roi_specification.
  -- * \param flip Specifies the axis about which the image is to be mirrored.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref mirror_error_codes
  --  

   function nppiMirror_8u_C1IR
     (pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oROI : nppdefs_h.NppiSize;
      flip : nppdefs_h.NppiAxis) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:2372
   pragma Import (C, nppiMirror_8u_C1IR, "nppiMirror_8u_C1IR");

  --*
  -- * 3 channel 8-bit unsigned image mirror.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oROI \ref roi_specification.
  -- * \param flip Specifies the axis about which the image is to be mirrored.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref mirror_error_codes
  --  

   function nppiMirror_8u_C3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oROI : nppdefs_h.NppiSize;
      flip : nppdefs_h.NppiAxis) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:2387
   pragma Import (C, nppiMirror_8u_C3R, "nppiMirror_8u_C3R");

  --*
  -- * 3 channel 8-bit unsigned in place image mirror.
  -- *
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oROI \ref roi_specification.
  -- * \param flip Specifies the axis about which the image is to be mirrored.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref mirror_error_codes
  --  

   function nppiMirror_8u_C3IR
     (pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oROI : nppdefs_h.NppiSize;
      flip : nppdefs_h.NppiAxis) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:2401
   pragma Import (C, nppiMirror_8u_C3IR, "nppiMirror_8u_C3IR");

  --*
  -- * 4 channel 8-bit unsigned image mirror.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep Distance in bytes between starts of consecutive lines of the
  -- *        destination image.
  -- * \param oROI \ref roi_specification.
  -- * \param flip Specifies the axis about which the image is to be mirrored.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref mirror_error_codes
  --  

   function nppiMirror_8u_C4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oROI : nppdefs_h.NppiSize;
      flip : nppdefs_h.NppiAxis) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:2417
   pragma Import (C, nppiMirror_8u_C4R, "nppiMirror_8u_C4R");

  --*
  -- * 4 channel 8-bit unsigned in place image mirror.
  -- *
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oROI \ref roi_specification.
  -- * \param flip Specifies the axis about which the image is to be mirrored.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref mirror_error_codes
  --  

   function nppiMirror_8u_C4IR
     (pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oROI : nppdefs_h.NppiSize;
      flip : nppdefs_h.NppiAxis) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:2431
   pragma Import (C, nppiMirror_8u_C4IR, "nppiMirror_8u_C4IR");

  --*
  -- * 4 channel 8-bit unsigned image mirror not affecting alpha.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep Distance in bytes between starts of consecutive lines of the
  -- *        destination image.
  -- * \param oROI \ref roi_specification.
  -- * \param flip Specifies the axis about which the image is to be mirrored.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref mirror_error_codes
  --  

   function nppiMirror_8u_AC4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oROI : nppdefs_h.NppiSize;
      flip : nppdefs_h.NppiAxis) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:2447
   pragma Import (C, nppiMirror_8u_AC4R, "nppiMirror_8u_AC4R");

  --*
  -- * 4 channel 8-bit unsigned in place image mirror not affecting alpha.
  -- *
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oROI \ref roi_specification.
  -- * \param flip Specifies the axis about which the image is to be mirrored.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref mirror_error_codes
  --  

   function nppiMirror_8u_AC4IR
     (pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oROI : nppdefs_h.NppiSize;
      flip : nppdefs_h.NppiAxis) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:2461
   pragma Import (C, nppiMirror_8u_AC4IR, "nppiMirror_8u_AC4IR");

  --*
  -- * 1 channel 16-bit unsigned image mirror.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oROI \ref roi_specification.
  -- * \param flip Specifies the axis about which the image is to be mirrored.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref mirror_error_codes
  --  

   function nppiMirror_16u_C1R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oROI : nppdefs_h.NppiSize;
      flip : nppdefs_h.NppiAxis) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:2475
   pragma Import (C, nppiMirror_16u_C1R, "nppiMirror_16u_C1R");

  --*
  -- * 1 channel 16-bit unsigned in place image mirror.
  -- *
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oROI \ref roi_specification.
  -- * \param flip Specifies the axis about which the image is to be mirrored.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref mirror_error_codes
  --  

   function nppiMirror_16u_C1IR
     (pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oROI : nppdefs_h.NppiSize;
      flip : nppdefs_h.NppiAxis) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:2489
   pragma Import (C, nppiMirror_16u_C1IR, "nppiMirror_16u_C1IR");

  --*
  -- * 3 channel 16-bit unsigned image mirror.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oROI \ref roi_specification.
  -- * \param flip Specifies the axis about which the image is to be mirrored.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref mirror_error_codes
  --  

   function nppiMirror_16u_C3R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oROI : nppdefs_h.NppiSize;
      flip : nppdefs_h.NppiAxis) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:2504
   pragma Import (C, nppiMirror_16u_C3R, "nppiMirror_16u_C3R");

  --*
  -- * 3 channel 16-bit unsigned in place image mirror.
  -- *
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oROI \ref roi_specification.
  -- * \param flip Specifies the axis about which the image is to be mirrored.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref mirror_error_codes
  --  

   function nppiMirror_16u_C3IR
     (pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oROI : nppdefs_h.NppiSize;
      flip : nppdefs_h.NppiAxis) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:2518
   pragma Import (C, nppiMirror_16u_C3IR, "nppiMirror_16u_C3IR");

  --*
  -- * 4 channel 16-bit unsigned image mirror.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep Distance in bytes between starts of consecutive lines of the
  -- *        destination image.
  -- * \param oROI \ref roi_specification.
  -- * \param flip Specifies the axis about which the image is to be mirrored.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref mirror_error_codes
  --  

   function nppiMirror_16u_C4R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oROI : nppdefs_h.NppiSize;
      flip : nppdefs_h.NppiAxis) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:2534
   pragma Import (C, nppiMirror_16u_C4R, "nppiMirror_16u_C4R");

  --*
  -- * 4 channel 16-bit unsigned in place image mirror.
  -- *
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oROI \ref roi_specification.
  -- * \param flip Specifies the axis about which the image is to be mirrored.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref mirror_error_codes
  --  

   function nppiMirror_16u_C4IR
     (pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oROI : nppdefs_h.NppiSize;
      flip : nppdefs_h.NppiAxis) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:2548
   pragma Import (C, nppiMirror_16u_C4IR, "nppiMirror_16u_C4IR");

  --*
  -- * 4 channel 16-bit unsigned image mirror not affecting alpha.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep Distance in bytes between starts of consecutive lines of the
  -- *        destination image.
  -- * \param oROI \ref roi_specification.
  -- * \param flip Specifies the axis about which the image is to be mirrored.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref mirror_error_codes
  --  

   function nppiMirror_16u_AC4R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oROI : nppdefs_h.NppiSize;
      flip : nppdefs_h.NppiAxis) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:2564
   pragma Import (C, nppiMirror_16u_AC4R, "nppiMirror_16u_AC4R");

  --*
  -- * 4 channel 16-bit unsigned in place image mirror not affecting alpha.
  -- *
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oROI \ref roi_specification.
  -- * \param flip Specifies the axis about which the image is to be mirrored.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref mirror_error_codes
  --  

   function nppiMirror_16u_AC4IR
     (pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oROI : nppdefs_h.NppiSize;
      flip : nppdefs_h.NppiAxis) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:2578
   pragma Import (C, nppiMirror_16u_AC4IR, "nppiMirror_16u_AC4IR");

  --*
  -- * 1 channel 16-bit signed image mirror.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oROI \ref roi_specification.
  -- * \param flip Specifies the axis about which the image is to be mirrored.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref mirror_error_codes
  --  

   function nppiMirror_16s_C1R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oROI : nppdefs_h.NppiSize;
      flip : nppdefs_h.NppiAxis) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:2593
   pragma Import (C, nppiMirror_16s_C1R, "nppiMirror_16s_C1R");

  --*
  -- * 1 channel 16-bit signed in place image mirror.
  -- *
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oROI \ref roi_specification.
  -- * \param flip Specifies the axis about which the image is to be mirrored.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref mirror_error_codes
  --  

   function nppiMirror_16s_C1IR
     (pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oROI : nppdefs_h.NppiSize;
      flip : nppdefs_h.NppiAxis) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:2607
   pragma Import (C, nppiMirror_16s_C1IR, "nppiMirror_16s_C1IR");

  --*
  -- * 3 channel 16-bit signed image mirror.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oROI \ref roi_specification.
  -- * \param flip Specifies the axis about which the image is to be mirrored.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref mirror_error_codes
  --  

   function nppiMirror_16s_C3R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oROI : nppdefs_h.NppiSize;
      flip : nppdefs_h.NppiAxis) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:2622
   pragma Import (C, nppiMirror_16s_C3R, "nppiMirror_16s_C3R");

  --*
  -- * 3 channel 16-bit signed in place image mirror.
  -- *
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oROI \ref roi_specification.
  -- * \param flip Specifies the axis about which the image is to be mirrored.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref mirror_error_codes
  --  

   function nppiMirror_16s_C3IR
     (pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oROI : nppdefs_h.NppiSize;
      flip : nppdefs_h.NppiAxis) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:2636
   pragma Import (C, nppiMirror_16s_C3IR, "nppiMirror_16s_C3IR");

  --*
  -- * 4 channel 16-bit signed image mirror.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep Distance in bytes between starts of consecutive lines of the
  -- *        destination image.
  -- * \param oROI \ref roi_specification.
  -- * \param flip Specifies the axis about which the image is to be mirrored.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref mirror_error_codes
  --  

   function nppiMirror_16s_C4R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oROI : nppdefs_h.NppiSize;
      flip : nppdefs_h.NppiAxis) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:2652
   pragma Import (C, nppiMirror_16s_C4R, "nppiMirror_16s_C4R");

  --*
  -- * 4 channel 16-bit signed in place image mirror.
  -- *
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oROI \ref roi_specification.
  -- * \param flip Specifies the axis about which the image is to be mirrored.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref mirror_error_codes
  --  

   function nppiMirror_16s_C4IR
     (pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oROI : nppdefs_h.NppiSize;
      flip : nppdefs_h.NppiAxis) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:2666
   pragma Import (C, nppiMirror_16s_C4IR, "nppiMirror_16s_C4IR");

  --*
  -- * 4 channel 16-bit signed image mirror not affecting alpha.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep Distance in bytes between starts of consecutive lines of the
  -- *        destination image.
  -- * \param oROI \ref roi_specification.
  -- * \param flip Specifies the axis about which the image is to be mirrored.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref mirror_error_codes
  --  

   function nppiMirror_16s_AC4R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oROI : nppdefs_h.NppiSize;
      flip : nppdefs_h.NppiAxis) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:2682
   pragma Import (C, nppiMirror_16s_AC4R, "nppiMirror_16s_AC4R");

  --*
  -- * 4 channel 16-bit signed in place image mirror not affecting alpha.
  -- *
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oROI \ref roi_specification.
  -- * \param flip Specifies the axis about which the image is to be mirrored.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref mirror_error_codes
  --  

   function nppiMirror_16s_AC4IR
     (pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oROI : nppdefs_h.NppiSize;
      flip : nppdefs_h.NppiAxis) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:2696
   pragma Import (C, nppiMirror_16s_AC4IR, "nppiMirror_16s_AC4IR");

  --*
  -- * 1 channel 32-bit image mirror.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oROI \ref roi_specification.
  -- * \param flip Specifies the axis about which the image is to be mirrored.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref mirror_error_codes
  --  

   function nppiMirror_32s_C1R
     (pSrc : access nppdefs_h.Npp32s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oROI : nppdefs_h.NppiSize;
      flip : nppdefs_h.NppiAxis) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:2711
   pragma Import (C, nppiMirror_32s_C1R, "nppiMirror_32s_C1R");

  --*
  -- * 1 channel 32-bit signed in place image mirror.
  -- *
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oROI \ref roi_specification.
  -- * \param flip Specifies the axis about which the image is to be mirrored.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref mirror_error_codes
  --  

   function nppiMirror_32s_C1IR
     (pSrcDst : access nppdefs_h.Npp32s;
      nSrcDstStep : int;
      oROI : nppdefs_h.NppiSize;
      flip : nppdefs_h.NppiAxis) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:2725
   pragma Import (C, nppiMirror_32s_C1IR, "nppiMirror_32s_C1IR");

  --*
  -- * 3 channel 32-bit image mirror.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oROI \ref roi_specification.
  -- * \param flip Specifies the axis about which the image is to be mirrored.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref mirror_error_codes
  --  

   function nppiMirror_32s_C3R
     (pSrc : access nppdefs_h.Npp32s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oROI : nppdefs_h.NppiSize;
      flip : nppdefs_h.NppiAxis) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:2740
   pragma Import (C, nppiMirror_32s_C3R, "nppiMirror_32s_C3R");

  --*
  -- * 3 channel 32-bit signed in place image mirror.
  -- *
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oROI \ref roi_specification.
  -- * \param flip Specifies the axis about which the image is to be mirrored.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref mirror_error_codes
  --  

   function nppiMirror_32s_C3IR
     (pSrcDst : access nppdefs_h.Npp32s;
      nSrcDstStep : int;
      oROI : nppdefs_h.NppiSize;
      flip : nppdefs_h.NppiAxis) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:2754
   pragma Import (C, nppiMirror_32s_C3IR, "nppiMirror_32s_C3IR");

  --*
  -- * 4 channel 32-bit image mirror.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep Distance in bytes between starts of consecutive lines of the
  -- *        destination image.
  -- * \param oROI \ref roi_specification.
  -- * \param flip Specifies the axis about which the image is to be mirrored.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref mirror_error_codes
  --  

   function nppiMirror_32s_C4R
     (pSrc : access nppdefs_h.Npp32s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oROI : nppdefs_h.NppiSize;
      flip : nppdefs_h.NppiAxis) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:2770
   pragma Import (C, nppiMirror_32s_C4R, "nppiMirror_32s_C4R");

  --*
  -- * 4 channel 32-bit signed in place image mirror.
  -- *
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oROI \ref roi_specification.
  -- * \param flip Specifies the axis about which the image is to be mirrored.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref mirror_error_codes
  --  

   function nppiMirror_32s_C4IR
     (pSrcDst : access nppdefs_h.Npp32s;
      nSrcDstStep : int;
      oROI : nppdefs_h.NppiSize;
      flip : nppdefs_h.NppiAxis) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:2784
   pragma Import (C, nppiMirror_32s_C4IR, "nppiMirror_32s_C4IR");

  --*
  -- * 4 channel 32-bit image mirror not affecting alpha.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep Distance in bytes between starts of consecutive lines of the
  -- *        destination image.
  -- * \param oROI \ref roi_specification.
  -- * \param flip Specifies the axis about which the image is to be mirrored.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref mirror_error_codes
  --  

   function nppiMirror_32s_AC4R
     (pSrc : access nppdefs_h.Npp32s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oROI : nppdefs_h.NppiSize;
      flip : nppdefs_h.NppiAxis) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:2801
   pragma Import (C, nppiMirror_32s_AC4R, "nppiMirror_32s_AC4R");

  --*
  -- * 4 channel 32-bit signed in place image mirror not affecting alpha.
  -- *
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oROI \ref roi_specification.
  -- * \param flip Specifies the axis about which the image is to be mirrored.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref mirror_error_codes
  --  

   function nppiMirror_32s_AC4IR
     (pSrcDst : access nppdefs_h.Npp32s;
      nSrcDstStep : int;
      oROI : nppdefs_h.NppiSize;
      flip : nppdefs_h.NppiAxis) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:2815
   pragma Import (C, nppiMirror_32s_AC4IR, "nppiMirror_32s_AC4IR");

  --*
  -- * 1 channel 32-bit float image mirror.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oROI \ref roi_specification.
  -- * \param flip Specifies the axis about which the image is to be mirrored.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref mirror_error_codes
  --  

   function nppiMirror_32f_C1R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oROI : nppdefs_h.NppiSize;
      flip : nppdefs_h.NppiAxis) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:2831
   pragma Import (C, nppiMirror_32f_C1R, "nppiMirror_32f_C1R");

  --*
  -- * 1 channel 32-bit float in place image mirror.
  -- *
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oROI \ref roi_specification.
  -- * \param flip Specifies the axis about which the image is to be mirrored.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref mirror_error_codes
  --  

   function nppiMirror_32f_C1IR
     (pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oROI : nppdefs_h.NppiSize;
      flip : nppdefs_h.NppiAxis) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:2845
   pragma Import (C, nppiMirror_32f_C1IR, "nppiMirror_32f_C1IR");

  --*
  -- * 3 channel 32-bit float image mirror.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oROI \ref roi_specification.
  -- * \param flip Specifies the axis about which the image is to be mirrored.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref mirror_error_codes
  --  

   function nppiMirror_32f_C3R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oROI : nppdefs_h.NppiSize;
      flip : nppdefs_h.NppiAxis) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:2861
   pragma Import (C, nppiMirror_32f_C3R, "nppiMirror_32f_C3R");

  --*
  -- * 3 channel 32-bit float in place image mirror.
  -- *
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oROI \ref roi_specification.
  -- * \param flip Specifies the axis about which the image is to be mirrored.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref mirror_error_codes
  --  

   function nppiMirror_32f_C3IR
     (pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oROI : nppdefs_h.NppiSize;
      flip : nppdefs_h.NppiAxis) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:2875
   pragma Import (C, nppiMirror_32f_C3IR, "nppiMirror_32f_C3IR");

  --*
  -- * 4 channel 32-bit float image mirror.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep Distance in bytes between starts of consecutive lines of the
  -- *        destination image.
  -- * \param oROI \ref roi_specification.
  -- * \param flip Specifies the axis about which the image is to be mirrored.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref mirror_error_codes
  --  

   function nppiMirror_32f_C4R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oROI : nppdefs_h.NppiSize;
      flip : nppdefs_h.NppiAxis) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:2891
   pragma Import (C, nppiMirror_32f_C4R, "nppiMirror_32f_C4R");

  --*
  -- * 4 channel 32-bit float in place image mirror.
  -- *
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oROI \ref roi_specification.
  -- * \param flip Specifies the axis about which the image is to be mirrored.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref mirror_error_codes
  --  

   function nppiMirror_32f_C4IR
     (pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oROI : nppdefs_h.NppiSize;
      flip : nppdefs_h.NppiAxis) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:2905
   pragma Import (C, nppiMirror_32f_C4IR, "nppiMirror_32f_C4IR");

  --*
  -- * 4 channel 32-bit float image mirror not affecting alpha.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep Distance in bytes between starts of consecutive lines of the
  -- *        destination image.
  -- * \param oROI \ref roi_specification.
  -- * \param flip Specifies the axis about which the image is to be mirrored.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref mirror_error_codes
  --  

   function nppiMirror_32f_AC4R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oROI : nppdefs_h.NppiSize;
      flip : nppdefs_h.NppiAxis) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:2921
   pragma Import (C, nppiMirror_32f_AC4R, "nppiMirror_32f_AC4R");

  --*
  -- * 4 channel 32-bit float in place image mirror not affecting alpha.
  -- *
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oROI \ref roi_specification.
  -- * \param flip Specifies the axis about which the image is to be mirrored.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref mirror_error_codes
  --  

   function nppiMirror_32f_AC4IR
     (pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oROI : nppdefs_h.NppiSize;
      flip : nppdefs_h.NppiAxis) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:2935
   pragma Import (C, nppiMirror_32f_AC4IR, "nppiMirror_32f_AC4IR");

  --* @}  
  --* @} image_mirror  
  --* @defgroup image_affine_transform Affine Transforms
  -- *
  -- * \section affine_transform_error_codes Affine Transform Error Codes
  -- *
  -- *         - ::NPP_RECT_ERROR Indicates an error condition if width or height of
  -- *           the intersection of the oSrcROI and source image is less than or
  -- *           equal to 1
  -- *         - ::NPP_WRONG_INTERSECTION_ROI_ERROR Indicates an error condition if
  -- *           oSrcROI has no intersection with the source image
  -- *         - ::NPP_INTERPOLATION_ERROR Indicates an error condition if
  -- *           interpolation has an illegal value
  -- *         - ::NPP_COEFF_ERROR Indicates an error condition if coefficient values
  -- *           are invalid
  -- *         - ::NPP_WRONG_INTERSECTION_QUAD_WARNING Indicates a warning that no
  -- *           operation is performed if the transformed source ROI has no
  -- *           intersection with the destination ROI
  -- *
  -- * @{
  -- *
  --  

  --* @name Utility Functions
  -- *
  -- * @{
  -- *
  --  

  --*
  -- * Computes affine transform coefficients based on source ROI and destination quadrilateral.
  -- *
  -- * The function computes the coefficients of an affine transformation that maps the
  -- * given source ROI (axis aligned rectangle with integer coordinates) to a quadrilateral
  -- * in the destination image.
  -- *
  -- * An affine transform in 2D is fully determined by the mapping of just three vertices.
  -- * This function's API allows for passing a complete quadrilateral effectively making the 
  -- * prolem overdetermined. What this means in practice is, that for certain quadrilaterals it is
  -- * not possible to find an affine transform that would map all four corners of the source
  -- * ROI to the four vertices of that quadrilateral.
  -- *
  -- * The function circumvents this problem by only looking at the first three vertices of
  -- * the destination image quadrilateral to determine the affine transformation's coefficients.
  -- * If the destination quadrilateral is indeed one that cannot be mapped using an affine
  -- * transformation the functions informs the user of this situation by returning a 
  -- * ::NPP_AFFINE_QUAD_INCORRECT_WARNING.
  -- *
  -- * \param oSrcROI The source ROI. This rectangle needs to be at least one pixel wide and
  -- *          high. If either width or hight are less than one an ::NPP_RECT_ERROR is returned.
  -- * \param aQuad The destination quadrilateral.
  -- * \param aCoeffs The resulting affine transform coefficients.
  -- * \return Error codes:
  -- *         - ::NPP_SIZE_ERROR Indicates an error condition if any image dimension
  -- *           has zero or negative value
  -- *         - ::NPP_RECT_ERROR Indicates an error condition if width or height of
  -- *           the intersection of the oSrcROI and source image is less than or
  -- *           equal to 1
  -- *         - ::NPP_COEFF_ERROR Indicates an error condition if coefficient values
  -- *           are invalid
  -- *         - ::NPP_AFFINE_QUAD_INCORRECT_WARNING Indicates a warning when quad
  -- *           does not conform to the transform properties. Fourth vertex is
  -- *           ignored, internally computed coordinates are used instead
  --  

   function nppiGetAffineTransform
     (oSrcROI : nppdefs_h.NppiRect;
      aQuad : System.Address;
      aCoeffs : System.Address) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:3005
   pragma Import (C, nppiGetAffineTransform, "nppiGetAffineTransform");

  --*
  -- * Compute shape of transformed image.
  -- *
  -- * This method computes the quadrilateral in the destination image that 
  -- * the source ROI is transformed into by the affine transformation expressed
  -- * by the coefficients array (aCoeffs).
  -- *
  -- * \param oSrcROI The source ROI.
  -- * \param aQuad The resulting destination quadrangle.
  -- * \param aCoeffs The afine transform coefficients.
  -- * \return Error codes:
  -- *         - ::NPP_SIZE_ERROR Indicates an error condition if any image dimension
  -- *           has zero or negative value
  -- *         - ::NPP_RECT_ERROR Indicates an error condition if width or height of
  -- *           the intersection of the oSrcROI and source image is less than or
  -- *           equal to 1
  -- *         - ::NPP_COEFF_ERROR Indicates an error condition if coefficient values
  -- *           are invalid
  --  

   function nppiGetAffineQuad
     (oSrcROI : nppdefs_h.NppiRect;
      aQuad : System.Address;
      aCoeffs : System.Address) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:3028
   pragma Import (C, nppiGetAffineQuad, "nppiGetAffineQuad");

  --*
  -- * Compute bounding-box of transformed image.
  -- *
  -- * The method effectively computes the bounding box (axis aligned rectangle) of
  -- * the transformed source ROI (see nppiGetAffineQuad()). 
  -- *
  -- * \param oSrcROI The source ROI.
  -- * \param aBound The resulting bounding box.
  -- * \param aCoeffs The afine transform coefficients.
  -- * \return Error codes:
  -- *         - ::NPP_SIZE_ERROR Indicates an error condition if any image dimension
  -- *           has zero or negative value
  -- *         - ::NPP_RECT_ERROR Indicates an error condition if width or height of
  -- *           the intersection of the oSrcROI and source image is less than or
  -- *           equal to 1
  -- *         - ::NPP_COEFF_ERROR Indicates an error condition if coefficient values
  -- *           are invalid
  --  

   function nppiGetAffineBound
     (oSrcROI : nppdefs_h.NppiRect;
      aBound : System.Address;
      aCoeffs : System.Address) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:3050
   pragma Import (C, nppiGetAffineBound, "nppiGetAffineBound");

  --* @} Utility Functions Section  
  --* @name Affine Transform
  -- * Transforms (warps) an image based on an affine transform. The affine
  -- * transform is given as a \f$2\times 3\f$ matrix C. A pixel location \f$(x, y)\f$ in the
  -- * source image is mapped to the location \f$(x', y')\f$ in the destination image.
  -- * The destination image coorodinates are computed as follows:
  -- * \f[
  -- * x' = c_{00} * x + c_{01} * y + c_{02} \qquad
  -- * y' = c_{10} * x + c_{11} * y + c_{12} \qquad
  -- * C = \left[ \matrix{c_{00} & c_{01} & c_{02} \cr 
  --                      c_{10} & c_{11} & c_{12} } \right]
  -- * \f]
  -- * Affine transforms can be understood as a linear transformation (traditional
  -- * matrix multiplication) and a shift operation. The \f$2\times 2\f$ matrix 
  -- * \f[
  -- *    L = \left[ \matrix{c_{00} & c_{01} \cr 
  -- *                       c_{10} & c_{11} } \right]
  -- * \f]
  -- * represents the linear transform portion of the affine transformation. The
  -- * vector
  -- * \f[
  -- *      v = \left( \matrix{c_{02} \cr
  --                           c_{12} } \right)
  -- * \f]
  -- * represents the post-transform shift, i.e. after the pixel location is transformed
  -- * by \f$L\f$ it is translated by \f$v\f$.
  -- * 
  -- * @{
  -- *
  --  

  --*
  -- * Single-channel 8-bit unsigned affine warp.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Affine transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref affine_transform_error_codes
  --  

   function nppiWarpAffine_8u_C1R
     (pSrc : access nppdefs_h.Npp8u;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:3101
   pragma Import (C, nppiWarpAffine_8u_C1R, "nppiWarpAffine_8u_C1R");

  --*
  -- * Three-channel 8-bit unsigned affine warp.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Affine transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref affine_transform_error_codes
  --  

   function nppiWarpAffine_8u_C3R
     (pSrc : access nppdefs_h.Npp8u;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:3121
   pragma Import (C, nppiWarpAffine_8u_C3R, "nppiWarpAffine_8u_C3R");

  --*
  -- * Four-channel 8-bit unsigned affine warp.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Affine transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref affine_transform_error_codes
  --  

   function nppiWarpAffine_8u_C4R
     (pSrc : access nppdefs_h.Npp8u;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:3142
   pragma Import (C, nppiWarpAffine_8u_C4R, "nppiWarpAffine_8u_C4R");

  --*
  -- * Four-channel 8-bit unsigned affine warp, ignoring alpha channel.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Affine transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref affine_transform_error_codes
  --  

   function nppiWarpAffine_8u_AC4R
     (pSrc : access nppdefs_h.Npp8u;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:3163
   pragma Import (C, nppiWarpAffine_8u_AC4R, "nppiWarpAffine_8u_AC4R");

  --*
  -- * Three-channel planar 8-bit unsigned affine warp.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Affine transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref affine_transform_error_codes
  --  

   function nppiWarpAffine_8u_P3R
     (pSrc : System.Address;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : System.Address;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:3184
   pragma Import (C, nppiWarpAffine_8u_P3R, "nppiWarpAffine_8u_P3R");

  --*
  -- * Four-channel planar 8-bit unsigned affine warp.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Affine transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref affine_transform_error_codes
  --  

   function nppiWarpAffine_8u_P4R
     (pSrc : System.Address;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : System.Address;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:3204
   pragma Import (C, nppiWarpAffine_8u_P4R, "nppiWarpAffine_8u_P4R");

  --*
  -- * Single-channel 16-bit unsigned affine warp.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Affine transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref affine_transform_error_codes
  --  

   function nppiWarpAffine_16u_C1R
     (pSrc : access nppdefs_h.Npp16u;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:3224
   pragma Import (C, nppiWarpAffine_16u_C1R, "nppiWarpAffine_16u_C1R");

  --*
  -- * Three-channel 16-bit unsigned affine warp.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Affine transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref affine_transform_error_codes
  --  

   function nppiWarpAffine_16u_C3R
     (pSrc : access nppdefs_h.Npp16u;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:3244
   pragma Import (C, nppiWarpAffine_16u_C3R, "nppiWarpAffine_16u_C3R");

  --*
  -- * Four-channel 16-bit unsigned affine warp.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Affine transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref affine_transform_error_codes
  --  

   function nppiWarpAffine_16u_C4R
     (pSrc : access nppdefs_h.Npp16u;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:3264
   pragma Import (C, nppiWarpAffine_16u_C4R, "nppiWarpAffine_16u_C4R");

  --*
  -- * Four-channel 16-bit unsigned affine warp, ignoring alpha channel.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Affine transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref affine_transform_error_codes
  --  

   function nppiWarpAffine_16u_AC4R
     (pSrc : access nppdefs_h.Npp16u;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:3284
   pragma Import (C, nppiWarpAffine_16u_AC4R, "nppiWarpAffine_16u_AC4R");

  --*
  -- * Three-channel planar 16-bit unsigned affine warp.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Affine transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref affine_transform_error_codes
  --  

   function nppiWarpAffine_16u_P3R
     (pSrc : System.Address;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : System.Address;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:3304
   pragma Import (C, nppiWarpAffine_16u_P3R, "nppiWarpAffine_16u_P3R");

  --*
  -- * Four-channel planar 16-bit unsigned affine warp.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Affine transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref affine_transform_error_codes
  --  

   function nppiWarpAffine_16u_P4R
     (pSrc : System.Address;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : System.Address;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:3324
   pragma Import (C, nppiWarpAffine_16u_P4R, "nppiWarpAffine_16u_P4R");

  --*
  -- * Single-channel 32-bit signed affine warp.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Affine transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref affine_transform_error_codes
  --  

   function nppiWarpAffine_32s_C1R
     (pSrc : access nppdefs_h.Npp32s;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:3344
   pragma Import (C, nppiWarpAffine_32s_C1R, "nppiWarpAffine_32s_C1R");

  --*
  -- * Three-channel 32-bit signed affine warp.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Affine transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref affine_transform_error_codes
  --  

   function nppiWarpAffine_32s_C3R
     (pSrc : access nppdefs_h.Npp32s;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:3364
   pragma Import (C, nppiWarpAffine_32s_C3R, "nppiWarpAffine_32s_C3R");

  --*
  -- * Four-channel 32-bit signed affine warp.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Affine transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref affine_transform_error_codes
  --  

   function nppiWarpAffine_32s_C4R
     (pSrc : access nppdefs_h.Npp32s;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:3384
   pragma Import (C, nppiWarpAffine_32s_C4R, "nppiWarpAffine_32s_C4R");

  --*
  -- * Four-channel 32-bit signed affine warp, ignoring alpha channel.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Affine transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref affine_transform_error_codes
  --  

   function nppiWarpAffine_32s_AC4R
     (pSrc : access nppdefs_h.Npp32s;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:3404
   pragma Import (C, nppiWarpAffine_32s_AC4R, "nppiWarpAffine_32s_AC4R");

  --*
  -- * Three-channel planar 32-bit signed affine warp.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Affine transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref affine_transform_error_codes
  --  

   function nppiWarpAffine_32s_P3R
     (pSrc : System.Address;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : System.Address;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:3424
   pragma Import (C, nppiWarpAffine_32s_P3R, "nppiWarpAffine_32s_P3R");

  --*
  -- * Four-channel planar 32-bit signed affine warp.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Affine transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref affine_transform_error_codes
  --  

   function nppiWarpAffine_32s_P4R
     (pSrc : System.Address;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : System.Address;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:3444
   pragma Import (C, nppiWarpAffine_32s_P4R, "nppiWarpAffine_32s_P4R");

  --*
  -- * Single-channel 32-bit floating-point affine warp.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Affine transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref affine_transform_error_codes
  --  

   function nppiWarpAffine_32f_C1R
     (pSrc : access nppdefs_h.Npp32f;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:3464
   pragma Import (C, nppiWarpAffine_32f_C1R, "nppiWarpAffine_32f_C1R");

  --*
  -- * Three-channel 32-bit floating-point affine warp.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Affine transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref affine_transform_error_codes
  --  

   function nppiWarpAffine_32f_C3R
     (pSrc : access nppdefs_h.Npp32f;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:3484
   pragma Import (C, nppiWarpAffine_32f_C3R, "nppiWarpAffine_32f_C3R");

  --*
  -- * Four-channel 32-bit floating-point affine warp.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Affine transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref affine_transform_error_codes
  --  

   function nppiWarpAffine_32f_C4R
     (pSrc : access nppdefs_h.Npp32f;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:3504
   pragma Import (C, nppiWarpAffine_32f_C4R, "nppiWarpAffine_32f_C4R");

  --*
  -- * Four-channel 32-bit floating-point affine warp, ignoring alpha channel.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Affine transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref affine_transform_error_codes
  --  

   function nppiWarpAffine_32f_AC4R
     (pSrc : access nppdefs_h.Npp32f;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:3524
   pragma Import (C, nppiWarpAffine_32f_AC4R, "nppiWarpAffine_32f_AC4R");

  --*
  -- * Three-channel planar 32-bit floating-point affine warp.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Affine transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref affine_transform_error_codes
  --  

   function nppiWarpAffine_32f_P3R
     (pSrc : System.Address;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : System.Address;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:3544
   pragma Import (C, nppiWarpAffine_32f_P3R, "nppiWarpAffine_32f_P3R");

  --*
  -- * Four-channel planar 32-bit floating-point affine warp.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Affine transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref affine_transform_error_codes
  --  

   function nppiWarpAffine_32f_P4R
     (pSrc : System.Address;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : System.Address;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:3564
   pragma Import (C, nppiWarpAffine_32f_P4R, "nppiWarpAffine_32f_P4R");

  --*
  -- * Single-channel 64-bit floating-point affine warp.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Affine transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref affine_transform_error_codes
  --  

   function nppiWarpAffine_64f_C1R
     (pSrc : access nppdefs_h.Npp64f;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp64f;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:3585
   pragma Import (C, nppiWarpAffine_64f_C1R, "nppiWarpAffine_64f_C1R");

  --*
  -- * Three-channel 64-bit floating-point affine warp.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Affine transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref affine_transform_error_codes
  --  

   function nppiWarpAffine_64f_C3R
     (pSrc : access nppdefs_h.Npp64f;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp64f;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:3605
   pragma Import (C, nppiWarpAffine_64f_C3R, "nppiWarpAffine_64f_C3R");

  --*
  -- * Four-channel 64-bit floating-point affine warp.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Affine transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref affine_transform_error_codes
  --  

   function nppiWarpAffine_64f_C4R
     (pSrc : access nppdefs_h.Npp64f;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp64f;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:3625
   pragma Import (C, nppiWarpAffine_64f_C4R, "nppiWarpAffine_64f_C4R");

  --*
  -- * Four-channel 64-bit floating-point affine warp, ignoring alpha channel.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Affine transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref affine_transform_error_codes
  --  

   function nppiWarpAffine_64f_AC4R
     (pSrc : access nppdefs_h.Npp64f;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp64f;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:3645
   pragma Import (C, nppiWarpAffine_64f_AC4R, "nppiWarpAffine_64f_AC4R");

  --*
  -- * Three-channel planar 64-bit floating-point affine warp.
  -- * 
  -- * \param aSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param aDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Affine transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref affine_transform_error_codes
  --  

   function nppiWarpAffine_64f_P3R
     (aSrc : System.Address;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      aDst : System.Address;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:3665
   pragma Import (C, nppiWarpAffine_64f_P3R, "nppiWarpAffine_64f_P3R");

  --*
  -- * Four-channel planar 64-bit floating-point affine warp.
  -- * 
  -- * \param aSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param aDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Affine transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref affine_transform_error_codes
  --  

   function nppiWarpAffine_64f_P4R
     (aSrc : System.Address;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      aDst : System.Address;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:3685
   pragma Import (C, nppiWarpAffine_64f_P4R, "nppiWarpAffine_64f_P4R");

  --* @} Affine Transform Section  
  --* @name Backwards Affine Transform
  -- * Transforms (warps) an image based on an affine transform. The affine
  -- * transform is given as a \f$2\times 3\f$ matrix C. A pixel location \f$(x, y)\f$ in the
  -- * source image is mapped to the location \f$(x', y')\f$ in the destination image.
  -- * The destination image coorodinates fullfil the following properties:
  -- * \f[
  -- * x = c_{00} * x' + c_{01} * y' + c_{02} \qquad
  -- * y = c_{10} * x' + c_{11} * y' + c_{12} \qquad
  -- * C = \left[ \matrix{c_{00} & c_{01} & c_{02} \cr 
  --                      c_{10} & c_{11} & c_{12} } \right]
  -- * \f]
  -- * In other words, given matrix \f$C\f$ the source image's shape is transfored to the destination image
  -- * using the inverse matrix \f$C^{-1}\f$:
  -- * \f[
  -- * M = C^{-1} = \left[ \matrix{m_{00} & m_{01} & m_{02} \cr 
  --                               m_{10} & m_{11} & m_{12} } \right]
  -- * x' = m_{00} * x + m_{01} * y + m_{02} \qquad
  -- * y' = m_{10} * x + m_{11} * y + m_{12} \qquad
  -- * \f]
  -- *
  -- * @{
  -- *
  --  

  --*
  -- * Single-channel 8-bit unsigned integer backwards affine warp.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Affine transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref affine_transform_error_codes
  --  

   function nppiWarpAffineBack_8u_C1R
     (pSrc : access nppdefs_h.Npp8u;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:3732
   pragma Import (C, nppiWarpAffineBack_8u_C1R, "nppiWarpAffineBack_8u_C1R");

  --*
  -- * Three-channel 8-bit unsigned integer backwards affine warp.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Affine transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref affine_transform_error_codes
  --  

   function nppiWarpAffineBack_8u_C3R
     (pSrc : access nppdefs_h.Npp8u;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:3752
   pragma Import (C, nppiWarpAffineBack_8u_C3R, "nppiWarpAffineBack_8u_C3R");

  --*
  -- * Four-channel 8-bit unsigned integer backwards affine warp.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Affine transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref affine_transform_error_codes
  --  

   function nppiWarpAffineBack_8u_C4R
     (pSrc : access nppdefs_h.Npp8u;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:3772
   pragma Import (C, nppiWarpAffineBack_8u_C4R, "nppiWarpAffineBack_8u_C4R");

  --*
  -- * Four-channel 8-bit unsigned integer backwards affine warp, ignoring alpha channel.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Affine transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref affine_transform_error_codes
  --  

   function nppiWarpAffineBack_8u_AC4R
     (pSrc : access nppdefs_h.Npp8u;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:3792
   pragma Import (C, nppiWarpAffineBack_8u_AC4R, "nppiWarpAffineBack_8u_AC4R");

  --*
  -- * Three-channel planar 8-bit unsigned integer backwards affine warp.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Affine transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref affine_transform_error_codes
  --  

   function nppiWarpAffineBack_8u_P3R
     (pSrc : System.Address;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : System.Address;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:3812
   pragma Import (C, nppiWarpAffineBack_8u_P3R, "nppiWarpAffineBack_8u_P3R");

  --*
  -- * Four-channel planar 8-bit unsigned integer backwards affine warp.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Affine transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref affine_transform_error_codes
  --  

   function nppiWarpAffineBack_8u_P4R
     (pSrc : System.Address;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : System.Address;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:3832
   pragma Import (C, nppiWarpAffineBack_8u_P4R, "nppiWarpAffineBack_8u_P4R");

  --*
  -- * Single-channel 16-bit unsigned integer backwards affine warp.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Affine transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref affine_transform_error_codes
  --  

   function nppiWarpAffineBack_16u_C1R
     (pSrc : access nppdefs_h.Npp16u;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:3852
   pragma Import (C, nppiWarpAffineBack_16u_C1R, "nppiWarpAffineBack_16u_C1R");

  --*
  -- * Three-channel 16-bit unsigned integer backwards affine warp.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Affine transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref affine_transform_error_codes
  --  

   function nppiWarpAffineBack_16u_C3R
     (pSrc : access nppdefs_h.Npp16u;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:3872
   pragma Import (C, nppiWarpAffineBack_16u_C3R, "nppiWarpAffineBack_16u_C3R");

  --*
  -- * Four-channel 16-bit unsigned integer backwards affine warp.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Affine transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref affine_transform_error_codes
  --  

   function nppiWarpAffineBack_16u_C4R
     (pSrc : access nppdefs_h.Npp16u;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:3892
   pragma Import (C, nppiWarpAffineBack_16u_C4R, "nppiWarpAffineBack_16u_C4R");

  --*
  -- * Four-channel 16-bit unsigned integer backwards affine warp, ignoring alpha channel.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Affine transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref affine_transform_error_codes
  --  

   function nppiWarpAffineBack_16u_AC4R
     (pSrc : access nppdefs_h.Npp16u;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:3912
   pragma Import (C, nppiWarpAffineBack_16u_AC4R, "nppiWarpAffineBack_16u_AC4R");

  --*
  -- * Three-channel planar 16-bit unsigned integer backwards affine warp.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Affine transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref affine_transform_error_codes
  --  

   function nppiWarpAffineBack_16u_P3R
     (pSrc : System.Address;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : System.Address;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:3932
   pragma Import (C, nppiWarpAffineBack_16u_P3R, "nppiWarpAffineBack_16u_P3R");

  --*
  -- * Four-channel planar 16-bit unsigned integer backwards affine warp.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Affine transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref affine_transform_error_codes
  --  

   function nppiWarpAffineBack_16u_P4R
     (pSrc : System.Address;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : System.Address;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:3952
   pragma Import (C, nppiWarpAffineBack_16u_P4R, "nppiWarpAffineBack_16u_P4R");

  --*
  -- * Single-channel 32-bit signed integer backwards affine warp.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Affine transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref affine_transform_error_codes
  --  

   function nppiWarpAffineBack_32s_C1R
     (pSrc : access nppdefs_h.Npp32s;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:3972
   pragma Import (C, nppiWarpAffineBack_32s_C1R, "nppiWarpAffineBack_32s_C1R");

  --*
  -- * Three-channel 32-bit signed integer backwards affine warp.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Affine transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref affine_transform_error_codes
  --  

   function nppiWarpAffineBack_32s_C3R
     (pSrc : access nppdefs_h.Npp32s;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:3992
   pragma Import (C, nppiWarpAffineBack_32s_C3R, "nppiWarpAffineBack_32s_C3R");

  --*
  -- * Four-channel 32-bit signed integer backwards affine warp.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Affine transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref affine_transform_error_codes
  --  

   function nppiWarpAffineBack_32s_C4R
     (pSrc : access nppdefs_h.Npp32s;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:4012
   pragma Import (C, nppiWarpAffineBack_32s_C4R, "nppiWarpAffineBack_32s_C4R");

  --*
  -- * Four-channel 32-bit signed integer backwards affine warp, ignoring alpha channel.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Affine transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref affine_transform_error_codes
  --  

   function nppiWarpAffineBack_32s_AC4R
     (pSrc : access nppdefs_h.Npp32s;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:4032
   pragma Import (C, nppiWarpAffineBack_32s_AC4R, "nppiWarpAffineBack_32s_AC4R");

  --*
  -- * Three-channel planar 32-bit signed integer backwards affine warp.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Affine transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref affine_transform_error_codes
  --  

   function nppiWarpAffineBack_32s_P3R
     (pSrc : System.Address;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : System.Address;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:4052
   pragma Import (C, nppiWarpAffineBack_32s_P3R, "nppiWarpAffineBack_32s_P3R");

  --*
  -- * Four-channel planar 32-bit signed integer backwards affine warp.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Affine transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref affine_transform_error_codes
  --  

   function nppiWarpAffineBack_32s_P4R
     (pSrc : System.Address;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : System.Address;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:4072
   pragma Import (C, nppiWarpAffineBack_32s_P4R, "nppiWarpAffineBack_32s_P4R");

  --*
  -- * Single-channel 32-bit floating-point backwards affine warp.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Affine transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref affine_transform_error_codes
  --  

   function nppiWarpAffineBack_32f_C1R
     (pSrc : access nppdefs_h.Npp32f;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:4092
   pragma Import (C, nppiWarpAffineBack_32f_C1R, "nppiWarpAffineBack_32f_C1R");

  --*
  -- * Three-channel 32-bit floating-point backwards affine warp.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Affine transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref affine_transform_error_codes
  --  

   function nppiWarpAffineBack_32f_C3R
     (pSrc : access nppdefs_h.Npp32f;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:4112
   pragma Import (C, nppiWarpAffineBack_32f_C3R, "nppiWarpAffineBack_32f_C3R");

  --*
  -- * Four-channel 32-bit floating-point backwards affine warp.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Affine transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref affine_transform_error_codes
  --  

   function nppiWarpAffineBack_32f_C4R
     (pSrc : access nppdefs_h.Npp32f;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:4132
   pragma Import (C, nppiWarpAffineBack_32f_C4R, "nppiWarpAffineBack_32f_C4R");

  --*
  -- * Four-channel 32-bit floating-point backwards affine warp, ignoring alpha channel.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Affine transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref affine_transform_error_codes
  --  

   function nppiWarpAffineBack_32f_AC4R
     (pSrc : access nppdefs_h.Npp32f;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:4152
   pragma Import (C, nppiWarpAffineBack_32f_AC4R, "nppiWarpAffineBack_32f_AC4R");

  --*
  -- * Three-channel planar 32-bit floating-point backwards affine warp.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Affine transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref affine_transform_error_codes
  --  

   function nppiWarpAffineBack_32f_P3R
     (pSrc : System.Address;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : System.Address;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:4172
   pragma Import (C, nppiWarpAffineBack_32f_P3R, "nppiWarpAffineBack_32f_P3R");

  --*
  -- * Four-channel planar 32-bit floating-point backwards affine warp.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Affine transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref affine_transform_error_codes
  --  

   function nppiWarpAffineBack_32f_P4R
     (pSrc : System.Address;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : System.Address;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:4192
   pragma Import (C, nppiWarpAffineBack_32f_P4R, "nppiWarpAffineBack_32f_P4R");

  --* @} Backwards Affine Transform Section  
  --* @name Quad-Based Affine Transform
  -- * Transforms (warps) an image based on an affine transform. The affine
  -- * transform is computed such that it maps a quadrilateral in source image space to a 
  -- * quadrilateral in destination image space. 
  -- *
  -- * An affine transform is fully determined by the mapping of 3 discrete points.
  -- * The following primitives compute an affine transformation matrix that maps 
  -- * the first three corners of the source quad are mapped to the first three 
  -- * vertices of the destination image quad. If the fourth vertices do not match
  -- * the transform, an ::NPP_AFFINE_QUAD_INCORRECT_WARNING is returned by the primitive.
  -- *
  -- *
  -- * @{
  -- *
  --  

  --*
  -- * Single-channel 32-bit floating-point quad-based affine warp.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param aSrcQuad Source quad.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aDstQuad Destination quad.
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref affine_transform_error_codes
  --  

   function nppiWarpAffineQuad_8u_C1R
     (pSrc : access nppdefs_h.Npp8u;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      aSrcQuad : System.Address;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aDstQuad : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:4232
   pragma Import (C, nppiWarpAffineQuad_8u_C1R, "nppiWarpAffineQuad_8u_C1R");

  --*
  -- * Three-channel 8-bit unsigned integer quad-based affine warp.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param aSrcQuad Source quad.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aDstQuad Destination quad.
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref affine_transform_error_codes
  --  

   function nppiWarpAffineQuad_8u_C3R
     (pSrc : access nppdefs_h.Npp8u;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      aSrcQuad : System.Address;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aDstQuad : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:4254
   pragma Import (C, nppiWarpAffineQuad_8u_C3R, "nppiWarpAffineQuad_8u_C3R");

  --*
  -- * Four-channel 8-bit unsigned integer quad-based affine warp.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param aSrcQuad Source quad.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aDstQuad Destination quad.
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref affine_transform_error_codes
  --  

   function nppiWarpAffineQuad_8u_C4R
     (pSrc : access nppdefs_h.Npp8u;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      aSrcQuad : System.Address;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aDstQuad : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:4275
   pragma Import (C, nppiWarpAffineQuad_8u_C4R, "nppiWarpAffineQuad_8u_C4R");

  --*
  -- * Four-channel 8-bit unsigned integer quad-based affine warp, ignoring alpha channel.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param aSrcQuad Source quad.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aDstQuad Destination quad.
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref affine_transform_error_codes
  --  

   function nppiWarpAffineQuad_8u_AC4R
     (pSrc : access nppdefs_h.Npp8u;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      aSrcQuad : System.Address;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aDstQuad : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:4296
   pragma Import (C, nppiWarpAffineQuad_8u_AC4R, "nppiWarpAffineQuad_8u_AC4R");

  --*
  -- * Three-channel planar 8-bit unsigned integer quad-based affine warp.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param aSrcQuad Source quad.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aDstQuad Destination quad.
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref affine_transform_error_codes
  --  

   function nppiWarpAffineQuad_8u_P3R
     (pSrc : System.Address;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      aSrcQuad : System.Address;
      pDst : System.Address;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aDstQuad : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:4317
   pragma Import (C, nppiWarpAffineQuad_8u_P3R, "nppiWarpAffineQuad_8u_P3R");

  --*
  -- * Four-channel planar 8-bit unsigned integer quad-based affine warp.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param aSrcQuad Source quad.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aDstQuad Destination quad.
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref affine_transform_error_codes
  --  

   function nppiWarpAffineQuad_8u_P4R
     (pSrc : System.Address;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      aSrcQuad : System.Address;
      pDst : System.Address;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aDstQuad : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:4338
   pragma Import (C, nppiWarpAffineQuad_8u_P4R, "nppiWarpAffineQuad_8u_P4R");

  --*
  -- * Single-channel 16-bit unsigned integer quad-based affine warp.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param aSrcQuad Source quad.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aDstQuad Destination quad.
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref affine_transform_error_codes
  --  

   function nppiWarpAffineQuad_16u_C1R
     (pSrc : access nppdefs_h.Npp16u;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      aSrcQuad : System.Address;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aDstQuad : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:4359
   pragma Import (C, nppiWarpAffineQuad_16u_C1R, "nppiWarpAffineQuad_16u_C1R");

  --*
  -- * Three-channel 16-bit unsigned integer quad-based affine warp.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param aSrcQuad Source quad.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aDstQuad Destination quad.
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref affine_transform_error_codes
  --  

   function nppiWarpAffineQuad_16u_C3R
     (pSrc : access nppdefs_h.Npp16u;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      aSrcQuad : System.Address;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aDstQuad : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:4380
   pragma Import (C, nppiWarpAffineQuad_16u_C3R, "nppiWarpAffineQuad_16u_C3R");

  --*
  -- * Four-channel 16-bit unsigned integer quad-based affine warp.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param aSrcQuad Source quad.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aDstQuad Destination quad.
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref affine_transform_error_codes
  --  

   function nppiWarpAffineQuad_16u_C4R
     (pSrc : access nppdefs_h.Npp16u;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      aSrcQuad : System.Address;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aDstQuad : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:4401
   pragma Import (C, nppiWarpAffineQuad_16u_C4R, "nppiWarpAffineQuad_16u_C4R");

  --*
  -- * Four-channel 16-bit unsigned integer quad-based affine warp, ignoring alpha channel.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param aSrcQuad Source quad.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aDstQuad Destination quad.
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref affine_transform_error_codes
  --  

   function nppiWarpAffineQuad_16u_AC4R
     (pSrc : access nppdefs_h.Npp16u;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      aSrcQuad : System.Address;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aDstQuad : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:4422
   pragma Import (C, nppiWarpAffineQuad_16u_AC4R, "nppiWarpAffineQuad_16u_AC4R");

  --*
  -- * Three-channel planar 16-bit unsigned integer quad-based affine warp.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param aSrcQuad Source quad.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aDstQuad Destination quad.
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref affine_transform_error_codes
  --  

   function nppiWarpAffineQuad_16u_P3R
     (pSrc : System.Address;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      aSrcQuad : System.Address;
      pDst : System.Address;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aDstQuad : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:4443
   pragma Import (C, nppiWarpAffineQuad_16u_P3R, "nppiWarpAffineQuad_16u_P3R");

  --*
  -- * Four-channel planar 16-bit unsigned integer quad-based affine warp.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param aSrcQuad Source quad.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aDstQuad Destination quad.
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref affine_transform_error_codes
  --  

   function nppiWarpAffineQuad_16u_P4R
     (pSrc : System.Address;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      aSrcQuad : System.Address;
      pDst : System.Address;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aDstQuad : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:4464
   pragma Import (C, nppiWarpAffineQuad_16u_P4R, "nppiWarpAffineQuad_16u_P4R");

  --*
  -- * Single-channel 32-bit signed integer quad-based affine warp.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param aSrcQuad Source quad.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aDstQuad Destination quad.
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref affine_transform_error_codes
  --  

   function nppiWarpAffineQuad_32s_C1R
     (pSrc : access nppdefs_h.Npp32s;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      aSrcQuad : System.Address;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aDstQuad : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:4485
   pragma Import (C, nppiWarpAffineQuad_32s_C1R, "nppiWarpAffineQuad_32s_C1R");

  --*
  -- * Three-channel 32-bit signed integer quad-based affine warp.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param aSrcQuad Source quad.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aDstQuad Destination quad.
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref affine_transform_error_codes
  --  

   function nppiWarpAffineQuad_32s_C3R
     (pSrc : access nppdefs_h.Npp32s;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      aSrcQuad : System.Address;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aDstQuad : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:4506
   pragma Import (C, nppiWarpAffineQuad_32s_C3R, "nppiWarpAffineQuad_32s_C3R");

  --*
  -- * Four-channel 32-bit signed integer quad-based affine warp.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param aSrcQuad Source quad.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aDstQuad Destination quad.
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref affine_transform_error_codes
  --  

   function nppiWarpAffineQuad_32s_C4R
     (pSrc : access nppdefs_h.Npp32s;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      aSrcQuad : System.Address;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aDstQuad : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:4527
   pragma Import (C, nppiWarpAffineQuad_32s_C4R, "nppiWarpAffineQuad_32s_C4R");

  --*
  -- * Four-channel 32-bit signed integer quad-based affine warp, ignoring alpha channel.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param aSrcQuad Source quad.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aDstQuad Destination quad.
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref affine_transform_error_codes
  --  

   function nppiWarpAffineQuad_32s_AC4R
     (pSrc : access nppdefs_h.Npp32s;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      aSrcQuad : System.Address;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aDstQuad : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:4548
   pragma Import (C, nppiWarpAffineQuad_32s_AC4R, "nppiWarpAffineQuad_32s_AC4R");

  --*
  -- * Three-channel planar 32-bit signed integer quad-based affine warp.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param aSrcQuad Source quad.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aDstQuad Destination quad.
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref affine_transform_error_codes
  --  

   function nppiWarpAffineQuad_32s_P3R
     (pSrc : System.Address;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      aSrcQuad : System.Address;
      pDst : System.Address;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aDstQuad : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:4569
   pragma Import (C, nppiWarpAffineQuad_32s_P3R, "nppiWarpAffineQuad_32s_P3R");

  --*
  -- * Four-channel planar 32-bit signed integer quad-based affine warp.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param aSrcQuad Source quad.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aDstQuad Destination quad.
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref affine_transform_error_codes
  --  

   function nppiWarpAffineQuad_32s_P4R
     (pSrc : System.Address;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      aSrcQuad : System.Address;
      pDst : System.Address;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aDstQuad : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:4590
   pragma Import (C, nppiWarpAffineQuad_32s_P4R, "nppiWarpAffineQuad_32s_P4R");

  --*
  -- * Single-channel 32-bit floating-point quad-based affine warp.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param aSrcQuad Source quad.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aDstQuad Destination quad.
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref affine_transform_error_codes
  --  

   function nppiWarpAffineQuad_32f_C1R
     (pSrc : access nppdefs_h.Npp32f;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      aSrcQuad : System.Address;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aDstQuad : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:4611
   pragma Import (C, nppiWarpAffineQuad_32f_C1R, "nppiWarpAffineQuad_32f_C1R");

  --*
  -- * Three-channel 32-bit floating-point quad-based affine warp.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param aSrcQuad Source quad.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aDstQuad Destination quad.
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref affine_transform_error_codes
  --  

   function nppiWarpAffineQuad_32f_C3R
     (pSrc : access nppdefs_h.Npp32f;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      aSrcQuad : System.Address;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aDstQuad : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:4632
   pragma Import (C, nppiWarpAffineQuad_32f_C3R, "nppiWarpAffineQuad_32f_C3R");

  --*
  -- * Four-channel 32-bit floating-point quad-based affine warp.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param aSrcQuad Source quad.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aDstQuad Destination quad.
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref affine_transform_error_codes
  --  

   function nppiWarpAffineQuad_32f_C4R
     (pSrc : access nppdefs_h.Npp32f;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      aSrcQuad : System.Address;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aDstQuad : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:4653
   pragma Import (C, nppiWarpAffineQuad_32f_C4R, "nppiWarpAffineQuad_32f_C4R");

  --*
  -- * Four-channel 32-bit floating-point quad-based affine warp, ignoring alpha channel.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param aSrcQuad Source quad.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aDstQuad Destination quad.
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref affine_transform_error_codes
  --  

   function nppiWarpAffineQuad_32f_AC4R
     (pSrc : access nppdefs_h.Npp32f;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      aSrcQuad : System.Address;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aDstQuad : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:4674
   pragma Import (C, nppiWarpAffineQuad_32f_AC4R, "nppiWarpAffineQuad_32f_AC4R");

  --*
  -- * Three-channel planar 32-bit floating-point quad-based affine warp.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param aSrcQuad Source quad.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aDstQuad Destination quad.
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref affine_transform_error_codes
  --  

   function nppiWarpAffineQuad_32f_P3R
     (pSrc : System.Address;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      aSrcQuad : System.Address;
      pDst : System.Address;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aDstQuad : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:4695
   pragma Import (C, nppiWarpAffineQuad_32f_P3R, "nppiWarpAffineQuad_32f_P3R");

  --*
  -- * Four-channel planar 32-bit floating-point quad-based affine warp.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param aSrcQuad Source quad.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aDstQuad Destination quad.
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref affine_transform_error_codes
  --  

   function nppiWarpAffineQuad_32f_P4R
     (pSrc : System.Address;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      aSrcQuad : System.Address;
      pDst : System.Address;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aDstQuad : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:4716
   pragma Import (C, nppiWarpAffineQuad_32f_P4R, "nppiWarpAffineQuad_32f_P4R");

  --* @} Quad-Based Affine Transform Section  
  --* @} image_affine_transforms  
  --* @defgroup image_perspective_transforms Perspective Transform
  -- *
  -- * \section perspective_transform_error_codes Perspective Transform Error Codes
  -- *
  -- *         - ::NPP_RECT_ERROR Indicates an error condition if width or height of
  -- *           the intersection of the oSrcROI and source image is less than or
  -- *           equal to 1
  -- *         - ::NPP_WRONG_INTERSECTION_ROI_ERROR Indicates an error condition if
  -- *           oSrcROI has no intersection with the source image
  -- *         - ::NPP_INTERPOLATION_ERROR Indicates an error condition if
  -- *           interpolation has an illegal value
  -- *         - ::NPP_COEFF_ERROR Indicates an error condition if coefficient values
  -- *           are invalid
  -- *         - ::NPP_WRONG_INTERSECTION_QUAD_WARNING Indicates a warning that no
  -- *           operation is performed if the transformed source ROI has no
  -- *           intersection with the destination ROI
  -- *
  -- * @{
  -- *
  --  

  --* @name Utility Functions
  -- *
  -- * @{
  -- *
  --  

  --*
  -- * Calculates perspective transform coefficients given source rectangular ROI
  -- * and its destination quadrangle projection
  -- *
  -- * \param oSrcROI Source ROI
  -- * \param quad Destination quadrangle
  -- * \param aCoeffs Perspective transform coefficients
  -- * \return Error codes:
  -- *         - ::NPP_SIZE_ERROR Indicates an error condition if any image dimension
  -- *           has zero or negative value
  -- *         - ::NPP_RECT_ERROR Indicates an error condition if width or height of
  -- *           the intersection of the oSrcROI and source image is less than or
  -- *           equal to 1
  -- *         - ::NPP_COEFF_ERROR Indicates an error condition if coefficient values
  -- *           are invalid
  --  

   function nppiGetPerspectiveTransform
     (oSrcROI : nppdefs_h.NppiRect;
      quad : System.Address;
      aCoeffs : System.Address) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:4769
   pragma Import (C, nppiGetPerspectiveTransform, "nppiGetPerspectiveTransform");

  --*
  -- * Calculates perspective transform projection of given source rectangular
  -- * ROI
  -- *
  -- * \param oSrcROI Source ROI
  -- * \param quad Destination quadrangle
  -- * \param aCoeffs Perspective transform coefficients
  -- * \return Error codes:
  -- *         - ::NPP_SIZE_ERROR Indicates an error condition if any image dimension
  -- *           has zero or negative value
  -- *         - ::NPP_RECT_ERROR Indicates an error condition if width or height of
  -- *           the intersection of the oSrcROI and source image is less than or
  -- *           equal to 1
  -- *         - ::NPP_COEFF_ERROR Indicates an error condition if coefficient values
  -- *           are invalid
  --  

   function nppiGetPerspectiveQuad
     (oSrcROI : nppdefs_h.NppiRect;
      quad : System.Address;
      aCoeffs : System.Address) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:4789
   pragma Import (C, nppiGetPerspectiveQuad, "nppiGetPerspectiveQuad");

  --*
  -- * Calculates bounding box of the perspective transform projection of the
  -- * given source rectangular ROI
  -- *
  -- * \param oSrcROI Source ROI
  -- * \param bound Bounding box of the transformed source ROI
  -- * \param aCoeffs Perspective transform coefficients
  -- * \return Error codes:
  -- *         - ::NPP_SIZE_ERROR Indicates an error condition if any image dimension
  -- *           has zero or negative value
  -- *         - ::NPP_RECT_ERROR Indicates an error condition if width or height of
  -- *           the intersection of the oSrcROI and source image is less than or
  -- *           equal to 1
  -- *         - ::NPP_COEFF_ERROR Indicates an error condition if coefficient values
  -- *           are invalid
  --  

   function nppiGetPerspectiveBound
     (oSrcROI : nppdefs_h.NppiRect;
      bound : System.Address;
      aCoeffs : System.Address) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:4809
   pragma Import (C, nppiGetPerspectiveBound, "nppiGetPerspectiveBound");

  --* @} Utility Functions Section  
  --* @name Perspective Transform
  -- * Transforms (warps) an image based on a perspective transform. The perspective
  -- * transform is given as a \f$3\times 3\f$ matrix C. A pixel location \f$(x, y)\f$ in the
  -- * source image is mapped to the location \f$(x', y')\f$ in the destination image.
  -- * The destination image coorodinates are computed as follows:
  -- * \f[
  -- * x' = \frac{c_{00} * x + c_{01} * y + c_{02}}{c_{20} * x + c_{21} * y + c_{22}} \qquad
  -- * y' = \frac{c_{10} * x + c_{11} * y + c_{12}}{c_{20} * x + c_{21} * y + c_{22}}
  -- * \f]
  -- * \f[
  -- * C = \left[ \matrix{c_{00} & c_{01} & c_{02}   \cr 
  --                      c_{10} & c_{11} & c_{12}   \cr 
  --                      c_{20} & c_{21} & c_{22} } \right]
  -- * \f]
  -- *
  -- * @{
  -- *
  --  

  --*
  -- * Single-channel 8-bit unsigned integer perspective warp.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Perspective transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref perspective_transform_error_codes
  --  

   function nppiWarpPerspective_8u_C1R
     (pSrc : access nppdefs_h.Npp8u;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:4848
   pragma Import (C, nppiWarpPerspective_8u_C1R, "nppiWarpPerspective_8u_C1R");

  --*
  -- * Three-channel 8-bit unsigned integer perspective warp.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Perspective transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref perspective_transform_error_codes
  --  

   function nppiWarpPerspective_8u_C3R
     (pSrc : access nppdefs_h.Npp8u;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:4868
   pragma Import (C, nppiWarpPerspective_8u_C3R, "nppiWarpPerspective_8u_C3R");

  --*
  -- * Four-channel 8-bit unsigned integer perspective warp.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Perspective transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref perspective_transform_error_codes
  --  

   function nppiWarpPerspective_8u_C4R
     (pSrc : access nppdefs_h.Npp8u;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:4888
   pragma Import (C, nppiWarpPerspective_8u_C4R, "nppiWarpPerspective_8u_C4R");

  --*
  -- * Four-channel 8-bit unsigned integer perspective warp, ignoring alpha channel.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Perspective transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref perspective_transform_error_codes
  --  

   function nppiWarpPerspective_8u_AC4R
     (pSrc : access nppdefs_h.Npp8u;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:4908
   pragma Import (C, nppiWarpPerspective_8u_AC4R, "nppiWarpPerspective_8u_AC4R");

  --*
  -- * Three-channel planar 8-bit unsigned integer perspective warp.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Perspective transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref perspective_transform_error_codes
  --  

   function nppiWarpPerspective_8u_P3R
     (pSrc : System.Address;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : System.Address;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:4928
   pragma Import (C, nppiWarpPerspective_8u_P3R, "nppiWarpPerspective_8u_P3R");

  --*
  -- * Four-channel planar 8-bit unsigned integer perspective warp.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Perspective transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref perspective_transform_error_codes
  --  

   function nppiWarpPerspective_8u_P4R
     (pSrc : System.Address;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : System.Address;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:4948
   pragma Import (C, nppiWarpPerspective_8u_P4R, "nppiWarpPerspective_8u_P4R");

  --*
  -- * Single-channel 16-bit unsigned integer perspective warp.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Perspective transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref perspective_transform_error_codes
  --  

   function nppiWarpPerspective_16u_C1R
     (pSrc : access nppdefs_h.Npp16u;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:4968
   pragma Import (C, nppiWarpPerspective_16u_C1R, "nppiWarpPerspective_16u_C1R");

  --*
  -- * Three-channel 16-bit unsigned integer perspective warp.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Perspective transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref perspective_transform_error_codes
  --  

   function nppiWarpPerspective_16u_C3R
     (pSrc : access nppdefs_h.Npp16u;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:4988
   pragma Import (C, nppiWarpPerspective_16u_C3R, "nppiWarpPerspective_16u_C3R");

  --*
  -- * Four-channel 16-bit unsigned integer perspective warp.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Perspective transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref perspective_transform_error_codes
  --  

   function nppiWarpPerspective_16u_C4R
     (pSrc : access nppdefs_h.Npp16u;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:5008
   pragma Import (C, nppiWarpPerspective_16u_C4R, "nppiWarpPerspective_16u_C4R");

  --*
  -- * Four-channel 16-bit unsigned integer perspective warp, igoring alpha channel.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Perspective transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref perspective_transform_error_codes
  --  

   function nppiWarpPerspective_16u_AC4R
     (pSrc : access nppdefs_h.Npp16u;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:5028
   pragma Import (C, nppiWarpPerspective_16u_AC4R, "nppiWarpPerspective_16u_AC4R");

  --*
  -- * Three-channel planar 16-bit unsigned integer perspective warp.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Perspective transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref perspective_transform_error_codes
  --  

   function nppiWarpPerspective_16u_P3R
     (pSrc : System.Address;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : System.Address;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:5048
   pragma Import (C, nppiWarpPerspective_16u_P3R, "nppiWarpPerspective_16u_P3R");

  --*
  -- * Four-channel planar 16-bit unsigned integer perspective warp.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Perspective transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref perspective_transform_error_codes
  --  

   function nppiWarpPerspective_16u_P4R
     (pSrc : System.Address;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : System.Address;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:5068
   pragma Import (C, nppiWarpPerspective_16u_P4R, "nppiWarpPerspective_16u_P4R");

  --*
  -- * Single-channel 32-bit signed integer perspective warp.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Perspective transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref perspective_transform_error_codes
  --  

   function nppiWarpPerspective_32s_C1R
     (pSrc : access nppdefs_h.Npp32s;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:5088
   pragma Import (C, nppiWarpPerspective_32s_C1R, "nppiWarpPerspective_32s_C1R");

  --*
  -- * Three-channel 32-bit signed integer perspective warp.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Perspective transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref perspective_transform_error_codes
  --  

   function nppiWarpPerspective_32s_C3R
     (pSrc : access nppdefs_h.Npp32s;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:5108
   pragma Import (C, nppiWarpPerspective_32s_C3R, "nppiWarpPerspective_32s_C3R");

  --*
  -- * Four-channel 32-bit signed integer perspective warp.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Perspective transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref perspective_transform_error_codes
  --  

   function nppiWarpPerspective_32s_C4R
     (pSrc : access nppdefs_h.Npp32s;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:5128
   pragma Import (C, nppiWarpPerspective_32s_C4R, "nppiWarpPerspective_32s_C4R");

  --*
  -- * Four-channel 32-bit signed integer perspective warp, igoring alpha channel.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Perspective transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref perspective_transform_error_codes
  --  

   function nppiWarpPerspective_32s_AC4R
     (pSrc : access nppdefs_h.Npp32s;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:5148
   pragma Import (C, nppiWarpPerspective_32s_AC4R, "nppiWarpPerspective_32s_AC4R");

  --*
  -- * Three-channel planar 32-bit signed integer perspective warp.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Perspective transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref perspective_transform_error_codes
  --  

   function nppiWarpPerspective_32s_P3R
     (pSrc : System.Address;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : System.Address;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:5168
   pragma Import (C, nppiWarpPerspective_32s_P3R, "nppiWarpPerspective_32s_P3R");

  --*
  -- * Four-channel planar 32-bit signed integer perspective warp.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Perspective transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref perspective_transform_error_codes
  --  

   function nppiWarpPerspective_32s_P4R
     (pSrc : System.Address;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : System.Address;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:5188
   pragma Import (C, nppiWarpPerspective_32s_P4R, "nppiWarpPerspective_32s_P4R");

  --*
  -- * Single-channel 32-bit floating-point perspective warp.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Perspective transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref perspective_transform_error_codes
  --  

   function nppiWarpPerspective_32f_C1R
     (pSrc : access nppdefs_h.Npp32f;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:5208
   pragma Import (C, nppiWarpPerspective_32f_C1R, "nppiWarpPerspective_32f_C1R");

  --*
  -- * Three-channel 32-bit floating-point perspective warp.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Perspective transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref perspective_transform_error_codes
  --  

   function nppiWarpPerspective_32f_C3R
     (pSrc : access nppdefs_h.Npp32f;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:5228
   pragma Import (C, nppiWarpPerspective_32f_C3R, "nppiWarpPerspective_32f_C3R");

  --*
  -- * Four-channel 32-bit floating-point perspective warp.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Perspective transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref perspective_transform_error_codes
  --  

   function nppiWarpPerspective_32f_C4R
     (pSrc : access nppdefs_h.Npp32f;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:5248
   pragma Import (C, nppiWarpPerspective_32f_C4R, "nppiWarpPerspective_32f_C4R");

  --*
  -- * Four-channel 32-bit floating-point perspective warp, ignoring alpha channel.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Perspective transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref perspective_transform_error_codes
  --  

   function nppiWarpPerspective_32f_AC4R
     (pSrc : access nppdefs_h.Npp32f;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:5268
   pragma Import (C, nppiWarpPerspective_32f_AC4R, "nppiWarpPerspective_32f_AC4R");

  --*
  -- * Three-channel planar 32-bit floating-point perspective warp.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Perspective transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref perspective_transform_error_codes
  --  

   function nppiWarpPerspective_32f_P3R
     (pSrc : System.Address;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : System.Address;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:5288
   pragma Import (C, nppiWarpPerspective_32f_P3R, "nppiWarpPerspective_32f_P3R");

  --*
  -- * Four-channel planar 32-bit floating-point perspective warp.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Perspective transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref perspective_transform_error_codes
  --  

   function nppiWarpPerspective_32f_P4R
     (pSrc : System.Address;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : System.Address;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:5308
   pragma Import (C, nppiWarpPerspective_32f_P4R, "nppiWarpPerspective_32f_P4R");

  --* @} Perspective Transform Section  
  --* @name Backwards Perspective Transform
  -- * Transforms (warps) an image based on a perspective transform. The perspective
  -- * transform is given as a \f$3\times 3\f$ matrix C. A pixel location \f$(x, y)\f$ in the
  -- * source image is mapped to the location \f$(x', y')\f$ in the destination image.
  -- * The destination image coorodinates fullfil the following properties:
  -- * \f[
  -- * x = \frac{c_{00} * x' + c_{01} * y' + c_{02}}{c_{20} * x' + c_{21} * y' + c_{22}} \qquad
  -- * y = \frac{c_{10} * x' + c_{11} * y' + c_{12}}{c_{20} * x' + c_{21} * y' + c_{22}}
  -- * \f]
  -- * \f[
  -- * C = \left[ \matrix{c_{00} & c_{01} & c_{02}   \cr 
  --                      c_{10} & c_{11} & c_{12}   \cr 
  --                      c_{20} & c_{21} & c_{22} } \right]
  -- * \f]
  -- * In other words, given matrix \f$C\f$ the source image's shape is transfored to the destination image
  -- * using the inverse matrix \f$C^{-1}\f$:
  -- * \f[
  -- * M = C^{-1} = \left[ \matrix{m_{00} & m_{01} & m_{02} \cr 
  --                               m_{10} & m_{11} & m_{12} \cr 
  --                               m_{20} & m_{21} & m_{22} } \right]
  -- * x' = \frac{c_{00} * x + c_{01} * y + c_{02}}{c_{20} * x + c_{21} * y + c_{22}} \qquad
  -- * y' = \frac{c_{10} * x + c_{11} * y + c_{12}}{c_{20} * x + c_{21} * y + c_{22}}
  -- * \f]
  -- *
  -- * @{
  -- *
  --  

  --*
  -- * Single-channel 8-bit unsigned integer backwards perspective warp.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Perspective transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref perspective_transform_error_codes
  --  

   function nppiWarpPerspectiveBack_8u_C1R
     (pSrc : access nppdefs_h.Npp8u;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:5360
   pragma Import (C, nppiWarpPerspectiveBack_8u_C1R, "nppiWarpPerspectiveBack_8u_C1R");

  --*
  -- * Three-channel 8-bit unsigned integer backwards perspective warp.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Perspective transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref perspective_transform_error_codes
  --  

   function nppiWarpPerspectiveBack_8u_C3R
     (pSrc : access nppdefs_h.Npp8u;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:5380
   pragma Import (C, nppiWarpPerspectiveBack_8u_C3R, "nppiWarpPerspectiveBack_8u_C3R");

  --*
  -- * Four-channel 8-bit unsigned integer backwards perspective warp.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Perspective transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref perspective_transform_error_codes
  --  

   function nppiWarpPerspectiveBack_8u_C4R
     (pSrc : access nppdefs_h.Npp8u;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:5400
   pragma Import (C, nppiWarpPerspectiveBack_8u_C4R, "nppiWarpPerspectiveBack_8u_C4R");

  --*
  -- * Four-channel 8-bit unsigned integer backwards perspective warp, igoring alpha channel.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Perspective transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref perspective_transform_error_codes
  --  

   function nppiWarpPerspectiveBack_8u_AC4R
     (pSrc : access nppdefs_h.Npp8u;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:5420
   pragma Import (C, nppiWarpPerspectiveBack_8u_AC4R, "nppiWarpPerspectiveBack_8u_AC4R");

  --*
  -- * Three-channel planar 8-bit unsigned integer backwards perspective warp.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Perspective transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref perspective_transform_error_codes
  --  

   function nppiWarpPerspectiveBack_8u_P3R
     (pSrc : System.Address;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : System.Address;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:5440
   pragma Import (C, nppiWarpPerspectiveBack_8u_P3R, "nppiWarpPerspectiveBack_8u_P3R");

  --*
  -- * Four-channel planar 8-bit unsigned integer backwards perspective warp.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Perspective transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref perspective_transform_error_codes
  --  

   function nppiWarpPerspectiveBack_8u_P4R
     (pSrc : System.Address;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : System.Address;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:5460
   pragma Import (C, nppiWarpPerspectiveBack_8u_P4R, "nppiWarpPerspectiveBack_8u_P4R");

  --*
  -- * Single-channel 16-bit unsigned integer backwards perspective warp.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Perspective transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref perspective_transform_error_codes
  --  

   function nppiWarpPerspectiveBack_16u_C1R
     (pSrc : access nppdefs_h.Npp16u;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:5481
   pragma Import (C, nppiWarpPerspectiveBack_16u_C1R, "nppiWarpPerspectiveBack_16u_C1R");

  --*
  -- * Three-channel 16-bit unsigned integer backwards perspective warp.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Perspective transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref perspective_transform_error_codes
  --  

   function nppiWarpPerspectiveBack_16u_C3R
     (pSrc : access nppdefs_h.Npp16u;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:5501
   pragma Import (C, nppiWarpPerspectiveBack_16u_C3R, "nppiWarpPerspectiveBack_16u_C3R");

  --*
  -- * Four-channel 16-bit unsigned integer backwards perspective warp.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Perspective transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref perspective_transform_error_codes
  --  

   function nppiWarpPerspectiveBack_16u_C4R
     (pSrc : access nppdefs_h.Npp16u;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:5521
   pragma Import (C, nppiWarpPerspectiveBack_16u_C4R, "nppiWarpPerspectiveBack_16u_C4R");

  --*
  -- * Four-channel 16-bit unsigned integer backwards perspective warp, ignoring alpha channel.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Perspective transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref perspective_transform_error_codes
  --  

   function nppiWarpPerspectiveBack_16u_AC4R
     (pSrc : access nppdefs_h.Npp16u;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:5541
   pragma Import (C, nppiWarpPerspectiveBack_16u_AC4R, "nppiWarpPerspectiveBack_16u_AC4R");

  --*
  -- * Four-channel planar 16-bit unsigned integer backwards perspective warp.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Perspective transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref perspective_transform_error_codes
  --  

   function nppiWarpPerspectiveBack_16u_P3R
     (pSrc : System.Address;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : System.Address;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:5561
   pragma Import (C, nppiWarpPerspectiveBack_16u_P3R, "nppiWarpPerspectiveBack_16u_P3R");

  --*
  -- * Four-channel planar 16-bit unsigned integer backwards perspective warp.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Perspective transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref perspective_transform_error_codes
  --  

   function nppiWarpPerspectiveBack_16u_P4R
     (pSrc : System.Address;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : System.Address;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:5581
   pragma Import (C, nppiWarpPerspectiveBack_16u_P4R, "nppiWarpPerspectiveBack_16u_P4R");

  --*
  -- * Single-channel 32-bit signed integer backwards perspective warp.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Perspective transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref perspective_transform_error_codes
  --  

   function nppiWarpPerspectiveBack_32s_C1R
     (pSrc : access nppdefs_h.Npp32s;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:5601
   pragma Import (C, nppiWarpPerspectiveBack_32s_C1R, "nppiWarpPerspectiveBack_32s_C1R");

  --*
  -- * Three-channel 32-bit signed integer backwards perspective warp.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Perspective transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref perspective_transform_error_codes
  --  

   function nppiWarpPerspectiveBack_32s_C3R
     (pSrc : access nppdefs_h.Npp32s;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:5621
   pragma Import (C, nppiWarpPerspectiveBack_32s_C3R, "nppiWarpPerspectiveBack_32s_C3R");

  --*
  -- * Four-channel 32-bit signed integer backwards perspective warp.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Perspective transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref perspective_transform_error_codes
  --  

   function nppiWarpPerspectiveBack_32s_C4R
     (pSrc : access nppdefs_h.Npp32s;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:5641
   pragma Import (C, nppiWarpPerspectiveBack_32s_C4R, "nppiWarpPerspectiveBack_32s_C4R");

  --*
  -- * Four-channel 32-bit signed integer backwards perspective warp, ignoring alpha channel.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Perspective transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref perspective_transform_error_codes
  --  

   function nppiWarpPerspectiveBack_32s_AC4R
     (pSrc : access nppdefs_h.Npp32s;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:5661
   pragma Import (C, nppiWarpPerspectiveBack_32s_AC4R, "nppiWarpPerspectiveBack_32s_AC4R");

  --*
  -- * Three-channel planar 32-bit signed integer backwards perspective warp.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Perspective transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref perspective_transform_error_codes
  --  

   function nppiWarpPerspectiveBack_32s_P3R
     (pSrc : System.Address;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : System.Address;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:5681
   pragma Import (C, nppiWarpPerspectiveBack_32s_P3R, "nppiWarpPerspectiveBack_32s_P3R");

  --*
  -- * Four-channel planar 32-bit signed integer backwards perspective warp.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Perspective transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref perspective_transform_error_codes
  --  

   function nppiWarpPerspectiveBack_32s_P4R
     (pSrc : System.Address;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : System.Address;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:5701
   pragma Import (C, nppiWarpPerspectiveBack_32s_P4R, "nppiWarpPerspectiveBack_32s_P4R");

  --*
  -- * Single-channel 32-bit floating-point backwards perspective warp.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Perspective transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref perspective_transform_error_codes
  --  

   function nppiWarpPerspectiveBack_32f_C1R
     (pSrc : access nppdefs_h.Npp32f;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:5721
   pragma Import (C, nppiWarpPerspectiveBack_32f_C1R, "nppiWarpPerspectiveBack_32f_C1R");

  --*
  -- * Three-channel 32-bit floating-point backwards perspective warp.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Perspective transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref perspective_transform_error_codes
  --  

   function nppiWarpPerspectiveBack_32f_C3R
     (pSrc : access nppdefs_h.Npp32f;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:5741
   pragma Import (C, nppiWarpPerspectiveBack_32f_C3R, "nppiWarpPerspectiveBack_32f_C3R");

  --*
  -- * Four-channel 32-bit floating-point backwards perspective warp.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Perspective transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref perspective_transform_error_codes
  --  

   function nppiWarpPerspectiveBack_32f_C4R
     (pSrc : access nppdefs_h.Npp32f;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:5761
   pragma Import (C, nppiWarpPerspectiveBack_32f_C4R, "nppiWarpPerspectiveBack_32f_C4R");

  --*
  -- * Four-channel 32-bit floating-point backwards perspective warp, ignorning alpha channel.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Perspective transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref perspective_transform_error_codes
  --  

   function nppiWarpPerspectiveBack_32f_AC4R
     (pSrc : access nppdefs_h.Npp32f;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:5781
   pragma Import (C, nppiWarpPerspectiveBack_32f_AC4R, "nppiWarpPerspectiveBack_32f_AC4R");

  --*
  -- * Three-channel planar 32-bit floating-point backwards perspective warp.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Perspective transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref perspective_transform_error_codes
  --  

   function nppiWarpPerspectiveBack_32f_P3R
     (pSrc : System.Address;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : System.Address;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:5801
   pragma Import (C, nppiWarpPerspectiveBack_32f_P3R, "nppiWarpPerspectiveBack_32f_P3R");

  --*
  -- * Four-channel planar 32-bit floating-point backwards perspective warp.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aCoeffs Perspective transform coefficients
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref perspective_transform_error_codes
  --  

   function nppiWarpPerspectiveBack_32f_P4R
     (pSrc : System.Address;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      pDst : System.Address;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aCoeffs : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:5821
   pragma Import (C, nppiWarpPerspectiveBack_32f_P4R, "nppiWarpPerspectiveBack_32f_P4R");

  --* @} Backwards Perspective Transform Section  
  --* @name Quad-Based Perspective Transform
  -- * Transforms (warps) an image based on an perspective transform. The perspective
  -- * transform is computed such that it maps a quadrilateral in source image space to a 
  -- * quadrilateral in destination image space. 
  -- *
  -- * @{
  -- *
  --  

  --*
  -- * Single-channel 8-bit unsigned integer quad-based perspective warp.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param aSrcQuad Source quad.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aDstQuad Destination quad.
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref perspective_transform_error_codes
  --  

   function nppiWarpPerspectiveQuad_8u_C1R
     (pSrc : access nppdefs_h.Npp8u;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      aSrcQuad : System.Address;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aDstQuad : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:5853
   pragma Import (C, nppiWarpPerspectiveQuad_8u_C1R, "nppiWarpPerspectiveQuad_8u_C1R");

  --*
  -- * Three-channel 8-bit unsigned integer quad-based perspective warp.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param aSrcQuad Source quad.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aDstQuad Destination quad.
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref perspective_transform_error_codes
  --  

   function nppiWarpPerspectiveQuad_8u_C3R
     (pSrc : access nppdefs_h.Npp8u;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      aSrcQuad : System.Address;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aDstQuad : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:5873
   pragma Import (C, nppiWarpPerspectiveQuad_8u_C3R, "nppiWarpPerspectiveQuad_8u_C3R");

  --*
  -- * Four-channel 8-bit unsigned integer quad-based perspective warp.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param aSrcQuad Source quad.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aDstQuad Destination quad.
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref perspective_transform_error_codes
  --  

   function nppiWarpPerspectiveQuad_8u_C4R
     (pSrc : access nppdefs_h.Npp8u;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      aSrcQuad : System.Address;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aDstQuad : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:5893
   pragma Import (C, nppiWarpPerspectiveQuad_8u_C4R, "nppiWarpPerspectiveQuad_8u_C4R");

  --*
  -- * Four-channel 8-bit unsigned integer quad-based perspective warp, ignoring alpha channel.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param aSrcQuad Source quad.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aDstQuad Destination quad.
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref perspective_transform_error_codes
  --  

   function nppiWarpPerspectiveQuad_8u_AC4R
     (pSrc : access nppdefs_h.Npp8u;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      aSrcQuad : System.Address;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aDstQuad : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:5913
   pragma Import (C, nppiWarpPerspectiveQuad_8u_AC4R, "nppiWarpPerspectiveQuad_8u_AC4R");

  --*
  -- * Three-channel planar 8-bit unsigned integer quad-based perspective warp.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param aSrcQuad Source quad.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aDstQuad Destination quad.
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref perspective_transform_error_codes
  --  

   function nppiWarpPerspectiveQuad_8u_P3R
     (pSrc : System.Address;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      aSrcQuad : System.Address;
      pDst : System.Address;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aDstQuad : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:5933
   pragma Import (C, nppiWarpPerspectiveQuad_8u_P3R, "nppiWarpPerspectiveQuad_8u_P3R");

  --*
  -- * Four-channel planar 8-bit unsigned integer quad-based perspective warp.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param aSrcQuad Source quad.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aDstQuad Destination quad.
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref perspective_transform_error_codes
  --  

   function nppiWarpPerspectiveQuad_8u_P4R
     (pSrc : System.Address;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      aSrcQuad : System.Address;
      pDst : System.Address;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aDstQuad : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:5953
   pragma Import (C, nppiWarpPerspectiveQuad_8u_P4R, "nppiWarpPerspectiveQuad_8u_P4R");

  --*
  -- * Single-channel 16-bit unsigned integer quad-based perspective warp.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param aSrcQuad Source quad.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aDstQuad Destination quad.
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref perspective_transform_error_codes
  --  

   function nppiWarpPerspectiveQuad_16u_C1R
     (pSrc : access nppdefs_h.Npp16u;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      aSrcQuad : System.Address;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aDstQuad : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:5973
   pragma Import (C, nppiWarpPerspectiveQuad_16u_C1R, "nppiWarpPerspectiveQuad_16u_C1R");

  --*
  -- * Three-channel 16-bit unsigned integer quad-based perspective warp.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param aSrcQuad Source quad.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aDstQuad Destination quad.
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref perspective_transform_error_codes
  --  

   function nppiWarpPerspectiveQuad_16u_C3R
     (pSrc : access nppdefs_h.Npp16u;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      aSrcQuad : System.Address;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aDstQuad : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:5993
   pragma Import (C, nppiWarpPerspectiveQuad_16u_C3R, "nppiWarpPerspectiveQuad_16u_C3R");

  --*
  -- * Four-channel 16-bit unsigned integer quad-based perspective warp.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param aSrcQuad Source quad.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aDstQuad Destination quad.
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref perspective_transform_error_codes
  --  

   function nppiWarpPerspectiveQuad_16u_C4R
     (pSrc : access nppdefs_h.Npp16u;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      aSrcQuad : System.Address;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aDstQuad : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:6013
   pragma Import (C, nppiWarpPerspectiveQuad_16u_C4R, "nppiWarpPerspectiveQuad_16u_C4R");

  --*
  -- * Four-channel 16-bit unsigned integer quad-based perspective warp, ignoring alpha channel.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param aSrcQuad Source quad.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aDstQuad Destination quad.
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref perspective_transform_error_codes
  --  

   function nppiWarpPerspectiveQuad_16u_AC4R
     (pSrc : access nppdefs_h.Npp16u;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      aSrcQuad : System.Address;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aDstQuad : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:6033
   pragma Import (C, nppiWarpPerspectiveQuad_16u_AC4R, "nppiWarpPerspectiveQuad_16u_AC4R");

  --*
  -- * Three-channel planar 16-bit unsigned integer quad-based perspective warp.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param aSrcQuad Source quad.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aDstQuad Destination quad.
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref perspective_transform_error_codes
  --  

   function nppiWarpPerspectiveQuad_16u_P3R
     (pSrc : System.Address;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      aSrcQuad : System.Address;
      pDst : System.Address;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aDstQuad : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:6053
   pragma Import (C, nppiWarpPerspectiveQuad_16u_P3R, "nppiWarpPerspectiveQuad_16u_P3R");

  --*
  -- * Four-channel planar 16-bit unsigned integer quad-based perspective warp.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param aSrcQuad Source quad.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aDstQuad Destination quad.
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref perspective_transform_error_codes
  --  

   function nppiWarpPerspectiveQuad_16u_P4R
     (pSrc : System.Address;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      aSrcQuad : System.Address;
      pDst : System.Address;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aDstQuad : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:6073
   pragma Import (C, nppiWarpPerspectiveQuad_16u_P4R, "nppiWarpPerspectiveQuad_16u_P4R");

  --*
  -- * Single-channel 32-bit signed integer quad-based perspective warp.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param aSrcQuad Source quad.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aDstQuad Destination quad.
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref perspective_transform_error_codes
  --  

   function nppiWarpPerspectiveQuad_32s_C1R
     (pSrc : access nppdefs_h.Npp32s;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      aSrcQuad : System.Address;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aDstQuad : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:6093
   pragma Import (C, nppiWarpPerspectiveQuad_32s_C1R, "nppiWarpPerspectiveQuad_32s_C1R");

  --*
  -- * Three-channel 32-bit signed integer quad-based perspective warp.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param aSrcQuad Source quad.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aDstQuad Destination quad.
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref perspective_transform_error_codes
  --  

   function nppiWarpPerspectiveQuad_32s_C3R
     (pSrc : access nppdefs_h.Npp32s;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      aSrcQuad : System.Address;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aDstQuad : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:6113
   pragma Import (C, nppiWarpPerspectiveQuad_32s_C3R, "nppiWarpPerspectiveQuad_32s_C3R");

  --*
  -- * Four-channel 32-bit signed integer quad-based perspective warp.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param aSrcQuad Source quad.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aDstQuad Destination quad.
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref perspective_transform_error_codes
  --  

   function nppiWarpPerspectiveQuad_32s_C4R
     (pSrc : access nppdefs_h.Npp32s;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      aSrcQuad : System.Address;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aDstQuad : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:6133
   pragma Import (C, nppiWarpPerspectiveQuad_32s_C4R, "nppiWarpPerspectiveQuad_32s_C4R");

  --*
  -- * Four-channel 32-bit signed integer quad-based perspective warp, ignoring alpha channel.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param aSrcQuad Source quad.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aDstQuad Destination quad.
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref perspective_transform_error_codes
  --  

   function nppiWarpPerspectiveQuad_32s_AC4R
     (pSrc : access nppdefs_h.Npp32s;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      aSrcQuad : System.Address;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aDstQuad : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:6153
   pragma Import (C, nppiWarpPerspectiveQuad_32s_AC4R, "nppiWarpPerspectiveQuad_32s_AC4R");

  --*
  -- * Three-channel planar 32-bit signed integer quad-based perspective warp.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param aSrcQuad Source quad.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aDstQuad Destination quad.
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref perspective_transform_error_codes
  --  

   function nppiWarpPerspectiveQuad_32s_P3R
     (pSrc : System.Address;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      aSrcQuad : System.Address;
      pDst : System.Address;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aDstQuad : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:6173
   pragma Import (C, nppiWarpPerspectiveQuad_32s_P3R, "nppiWarpPerspectiveQuad_32s_P3R");

  --*
  -- * Four-channel planar 32-bit signed integer quad-based perspective warp.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param aSrcQuad Source quad.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aDstQuad Destination quad.
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref perspective_transform_error_codes
  --  

   function nppiWarpPerspectiveQuad_32s_P4R
     (pSrc : System.Address;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      aSrcQuad : System.Address;
      pDst : System.Address;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aDstQuad : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:6193
   pragma Import (C, nppiWarpPerspectiveQuad_32s_P4R, "nppiWarpPerspectiveQuad_32s_P4R");

  --*
  -- * Single-channel 32-bit floating-point quad-based perspective warp.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param aSrcQuad Source quad.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aDstQuad Destination quad.
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref perspective_transform_error_codes
  --  

   function nppiWarpPerspectiveQuad_32f_C1R
     (pSrc : access nppdefs_h.Npp32f;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      aSrcQuad : System.Address;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aDstQuad : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:6213
   pragma Import (C, nppiWarpPerspectiveQuad_32f_C1R, "nppiWarpPerspectiveQuad_32f_C1R");

  --*
  -- * Three-channel 32-bit floating-point quad-based perspective warp.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param aSrcQuad Source quad.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aDstQuad Destination quad.
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref perspective_transform_error_codes
  --  

   function nppiWarpPerspectiveQuad_32f_C3R
     (pSrc : access nppdefs_h.Npp32f;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      aSrcQuad : System.Address;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aDstQuad : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:6233
   pragma Import (C, nppiWarpPerspectiveQuad_32f_C3R, "nppiWarpPerspectiveQuad_32f_C3R");

  --*
  -- * Four-channel 32-bit floating-point quad-based perspective warp.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param aSrcQuad Source quad.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aDstQuad Destination quad.
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref perspective_transform_error_codes
  --  

   function nppiWarpPerspectiveQuad_32f_C4R
     (pSrc : access nppdefs_h.Npp32f;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      aSrcQuad : System.Address;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aDstQuad : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:6253
   pragma Import (C, nppiWarpPerspectiveQuad_32f_C4R, "nppiWarpPerspectiveQuad_32f_C4R");

  --*
  -- * Four-channel 32-bit floating-point quad-based perspective warp, ignoring alpha channel.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param aSrcQuad Source quad.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aDstQuad Destination quad.
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref perspective_transform_error_codes
  --  

   function nppiWarpPerspectiveQuad_32f_AC4R
     (pSrc : access nppdefs_h.Npp32f;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      aSrcQuad : System.Address;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aDstQuad : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:6273
   pragma Import (C, nppiWarpPerspectiveQuad_32f_AC4R, "nppiWarpPerspectiveQuad_32f_AC4R");

  --*
  -- * Three-channel planar 32-bit floating-point quad-based perspective warp.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param aSrcQuad Source quad.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aDstQuad Destination quad.
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref perspective_transform_error_codes
  --  

   function nppiWarpPerspectiveQuad_32f_P3R
     (pSrc : System.Address;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      aSrcQuad : System.Address;
      pDst : System.Address;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aDstQuad : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:6293
   pragma Import (C, nppiWarpPerspectiveQuad_32f_P3R, "nppiWarpPerspectiveQuad_32f_P3R");

  --*
  -- * Four-channel planar 32-bit floating-point quad-based perspective warp.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param oSrcSize Size of source image in pixels
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcROI Source ROI
  -- * \param aSrcQuad Source quad.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstROI Destination ROI
  -- * \param aDstQuad Destination quad.
  -- * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
  -- *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, \ref perspective_transform_error_codes
  --  

   function nppiWarpPerspectiveQuad_32f_P4R
     (pSrc : System.Address;
      oSrcSize : nppdefs_h.NppiSize;
      nSrcStep : int;
      oSrcROI : nppdefs_h.NppiRect;
      aSrcQuad : System.Address;
      pDst : System.Address;
      nDstStep : int;
      oDstROI : nppdefs_h.NppiRect;
      aDstQuad : System.Address;
      eInterpolation : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_geometry_transforms.h:6313
   pragma Import (C, nppiWarpPerspectiveQuad_32f_P4R, "nppiWarpPerspectiveQuad_32f_P4R");

  --* @} Quad-Based Perspective Transform Section  
  --* @} image_perspective_transforms  
  --* @} image_geometry_transforms  
  -- extern "C"  
end nppi_geometry_transforms_h;

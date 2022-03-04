pragma Ada_2005;
pragma Style_Checks (Off);

with Interfaces.C; use Interfaces.C;
with nppdefs_h;

package nppi_morphological_operations_h is

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
  -- * \file nppi_morphological_operations.h
  -- * NPP Image Processing Functionality.
  --  

  --* @defgroup image_morphological_operations Morphological Operations
  -- *  @ingroup nppi
  -- *
  -- * Morphological image operations. 
  -- *
  -- * Morphological operations are classified as \ref neighborhood_operations. 
  -- *
  -- * @{
  -- *
  -- * These functions can be found in either the nppi or nppim libraries. Linking to only the sub-libraries that you use can significantly
  -- * save link time, application load time, and CUDA runtime startup time when using dynamic libraries.
  -- *
  --  

  --* @defgroup image_dilate Dilation
  -- *
  -- * Dilation computes the output pixel as the maximum pixel value of the pixels
  -- * under the mask. Pixels who's corresponding mask values are zero do not 
  -- * participate in the maximum search.
  -- *
  -- * It is the user's responsibility to avoid \ref sampling_beyond_image_boundaries.
  -- *
  -- * @{
  -- *
  --  

  --*
  -- * Single-channel 8-bit unsigned integer dilation.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pMask Pointer to the start address of the mask array
  -- * \param oMaskSize Width and Height mask array.
  -- * \param oAnchor X and Y offsets of the mask origin frame of reference
  -- *        w.r.t the source pixel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDilate_8u_C1R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : nppdefs_h.Npp32s;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : nppdefs_h.Npp32s;
      oSizeROI : nppdefs_h.NppiSize;
      pMask : access nppdefs_h.Npp8u;
      oMaskSize : nppdefs_h.NppiSize;
      oAnchor : nppdefs_h.NppiPoint) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:104
   pragma Import (C, nppiDilate_8u_C1R, "nppiDilate_8u_C1R");

  --*
  -- * Three-channel 8-bit unsigned integer dilation.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pMask Pointer to the start address of the mask array
  -- * \param oMaskSize Width and Height mask array.
  -- * \param oAnchor X and Y offsets of the mask origin frame of reference
  -- *        w.r.t the source pixel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDilate_8u_C3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : nppdefs_h.Npp32s;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : nppdefs_h.Npp32s;
      oSizeROI : nppdefs_h.NppiSize;
      pMask : access nppdefs_h.Npp8u;
      oMaskSize : nppdefs_h.NppiSize;
      oAnchor : nppdefs_h.NppiPoint) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:122
   pragma Import (C, nppiDilate_8u_C3R, "nppiDilate_8u_C3R");

  --*
  -- * Four-channel 8-bit unsigned integer dilation.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pMask Pointer to the start address of the mask array
  -- * \param oMaskSize Width and Height mask array.
  -- * \param oAnchor X and Y offsets of the mask origin frame of reference
  -- *        w.r.t the source pixel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDilate_8u_C4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pMask : access nppdefs_h.Npp8u;
      oMaskSize : nppdefs_h.NppiSize;
      oAnchor : nppdefs_h.NppiPoint) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:140
   pragma Import (C, nppiDilate_8u_C4R, "nppiDilate_8u_C4R");

  --*
  -- * Four-channel 8-bit unsigned integer dilation, ignoring alpha-channel.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pMask Pointer to the start address of the mask array
  -- * \param oMaskSize Width and Height mask array.
  -- * \param oAnchor X and Y offsets of the mask origin frame of reference
  -- *        w.r.t the source pixel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDilate_8u_AC4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pMask : access nppdefs_h.Npp8u;
      oMaskSize : nppdefs_h.NppiSize;
      oAnchor : nppdefs_h.NppiPoint) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:158
   pragma Import (C, nppiDilate_8u_AC4R, "nppiDilate_8u_AC4R");

  --*
  -- * Single-channel 16-bit unsigned integer dilation.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pMask Pointer to the start address of the mask array
  -- * \param oMaskSize Width and Height mask array.
  -- * \param oAnchor X and Y offsets of the mask origin frame of reference
  -- *        w.r.t the source pixel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDilate_16u_C1R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : nppdefs_h.Npp32s;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : nppdefs_h.Npp32s;
      oSizeROI : nppdefs_h.NppiSize;
      pMask : access nppdefs_h.Npp8u;
      oMaskSize : nppdefs_h.NppiSize;
      oAnchor : nppdefs_h.NppiPoint) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:177
   pragma Import (C, nppiDilate_16u_C1R, "nppiDilate_16u_C1R");

  --*
  -- * Three-channel 16-bit unsigned integer dilation.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pMask Pointer to the start address of the mask array
  -- * \param oMaskSize Width and Height mask array.
  -- * \param oAnchor X and Y offsets of the mask origin frame of reference
  -- *        w.r.t the source pixel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDilate_16u_C3R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : nppdefs_h.Npp32s;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : nppdefs_h.Npp32s;
      oSizeROI : nppdefs_h.NppiSize;
      pMask : access nppdefs_h.Npp8u;
      oMaskSize : nppdefs_h.NppiSize;
      oAnchor : nppdefs_h.NppiPoint) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:195
   pragma Import (C, nppiDilate_16u_C3R, "nppiDilate_16u_C3R");

  --*
  -- * Four-channel 16-bit unsigned integer dilation.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pMask Pointer to the start address of the mask array
  -- * \param oMaskSize Width and Height mask array.
  -- * \param oAnchor X and Y offsets of the mask origin frame of reference
  -- *        w.r.t the source pixel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDilate_16u_C4R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pMask : access nppdefs_h.Npp8u;
      oMaskSize : nppdefs_h.NppiSize;
      oAnchor : nppdefs_h.NppiPoint) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:213
   pragma Import (C, nppiDilate_16u_C4R, "nppiDilate_16u_C4R");

  --*
  -- * Four-channel 16-bit unsigned integer dilation, ignoring alpha-channel.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pMask Pointer to the start address of the mask array
  -- * \param oMaskSize Width and Height mask array.
  -- * \param oAnchor X and Y offsets of the mask origin frame of reference
  -- *        w.r.t the source pixel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDilate_16u_AC4R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pMask : access nppdefs_h.Npp8u;
      oMaskSize : nppdefs_h.NppiSize;
      oAnchor : nppdefs_h.NppiPoint) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:231
   pragma Import (C, nppiDilate_16u_AC4R, "nppiDilate_16u_AC4R");

  --*
  -- * Single-channel 32-bit floating-point dilation.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pMask Pointer to the start address of the mask array
  -- * \param oMaskSize Width and Height mask array.
  -- * \param oAnchor X and Y offsets of the mask origin frame of reference
  -- *        w.r.t the source pixel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDilate_32f_C1R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : nppdefs_h.Npp32s;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : nppdefs_h.Npp32s;
      oSizeROI : nppdefs_h.NppiSize;
      pMask : access nppdefs_h.Npp8u;
      oMaskSize : nppdefs_h.NppiSize;
      oAnchor : nppdefs_h.NppiPoint) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:250
   pragma Import (C, nppiDilate_32f_C1R, "nppiDilate_32f_C1R");

  --*
  -- * Three-channel 32-bit floating-point dilation.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pMask Pointer to the start address of the mask array
  -- * \param oMaskSize Width and Height mask array.
  -- * \param oAnchor X and Y offsets of the mask origin frame of reference
  -- *        w.r.t the source pixel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDilate_32f_C3R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : nppdefs_h.Npp32s;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : nppdefs_h.Npp32s;
      oSizeROI : nppdefs_h.NppiSize;
      pMask : access nppdefs_h.Npp8u;
      oMaskSize : nppdefs_h.NppiSize;
      oAnchor : nppdefs_h.NppiPoint) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:268
   pragma Import (C, nppiDilate_32f_C3R, "nppiDilate_32f_C3R");

  --*
  -- * Four-channel 32-bit floating-point dilation.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pMask Pointer to the start address of the mask array
  -- * \param oMaskSize Width and Height mask array.
  -- * \param oAnchor X and Y offsets of the mask origin frame of reference
  -- *        w.r.t the source pixel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDilate_32f_C4R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pMask : access nppdefs_h.Npp8u;
      oMaskSize : nppdefs_h.NppiSize;
      oAnchor : nppdefs_h.NppiPoint) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:286
   pragma Import (C, nppiDilate_32f_C4R, "nppiDilate_32f_C4R");

  --*
  -- * Four-channel 32-bit floating-point dilation, ignoring alpha-channel.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pMask Pointer to the start address of the mask array
  -- * \param oMaskSize Width and Height mask array.
  -- * \param oAnchor X and Y offsets of the mask origin frame of reference
  -- *        w.r.t the source pixel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDilate_32f_AC4R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pMask : access nppdefs_h.Npp8u;
      oMaskSize : nppdefs_h.NppiSize;
      oAnchor : nppdefs_h.NppiPoint) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:304
   pragma Import (C, nppiDilate_32f_AC4R, "nppiDilate_32f_AC4R");

  --* @} image_dilate  
  --* @defgroup image_dilate_border Dilation with border control
  -- *
  -- * Dilation computes the output pixel as the maximum pixel value of the pixels
  -- * under the mask. Pixels who's corresponding mask values are zero do not 
  -- * participate in the maximum search.
  -- *
  -- * If any portion of the mask overlaps the source image boundary the requested border type 
  -- * operation is applied to all mask pixels which fall outside of the source image.
  -- *
  -- * Currently only the NPP_BORDER_REPLICATE border type operation is supported.
  -- *
  -- * @{
  -- *
  --  

  --*
  -- * Single-channel 8-bit unsigned integer dilation with border control.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Source image width and height in pixels relative to pSrc.
  -- * \param oSrcOffset Source image starting point relative to pSrc. 
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pMask Pointer to the start address of the mask array
  -- * \param oMaskSize Width and Height mask array.
  -- * \param oAnchor X and Y offsets of the mask origin frame of reference
  -- *        w.r.t the source pixel.
  -- * \param eBorderType The border type operation to be applied at source image border boundaries.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDilateBorder_8u_C1R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : nppdefs_h.Npp32s;
      oSrcSize : nppdefs_h.NppiSize;
      oSrcOffset : nppdefs_h.NppiPoint;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : nppdefs_h.Npp32s;
      oSizeROI : nppdefs_h.NppiSize;
      pMask : access nppdefs_h.Npp8u;
      oMaskSize : nppdefs_h.NppiSize;
      oAnchor : nppdefs_h.NppiPoint;
      eBorderType : nppdefs_h.NppiBorderType) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:343
   pragma Import (C, nppiDilateBorder_8u_C1R, "nppiDilateBorder_8u_C1R");

  --*
  -- * Three-channel 8-bit unsigned integer dilation with border control.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Source image width and height in pixels relative to pSrc.
  -- * \param oSrcOffset Source image starting point relative to pSrc. 
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pMask Pointer to the start address of the mask array
  -- * \param oMaskSize Width and Height mask array.
  -- * \param oAnchor X and Y offsets of the mask origin frame of reference
  -- *        w.r.t the source pixel.
  -- * \param eBorderType The border type operation to be applied at source image border boundaries.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDilateBorder_8u_C3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : nppdefs_h.Npp32s;
      oSrcSize : nppdefs_h.NppiSize;
      oSrcOffset : nppdefs_h.NppiPoint;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : nppdefs_h.Npp32s;
      oSizeROI : nppdefs_h.NppiSize;
      pMask : access nppdefs_h.Npp8u;
      oMaskSize : nppdefs_h.NppiSize;
      oAnchor : nppdefs_h.NppiPoint;
      eBorderType : nppdefs_h.NppiBorderType) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:364
   pragma Import (C, nppiDilateBorder_8u_C3R, "nppiDilateBorder_8u_C3R");

  --*
  -- * Four-channel 8-bit unsigned integer dilation with border control.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Source image width and height in pixels relative to pSrc.
  -- * \param oSrcOffset Source image starting point relative to pSrc. 
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pMask Pointer to the start address of the mask array
  -- * \param oMaskSize Width and Height mask array.
  -- * \param oAnchor X and Y offsets of the mask origin frame of reference
  -- *        w.r.t the source pixel.
  -- * \param eBorderType The border type operation to be applied at source image border boundaries.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDilateBorder_8u_C4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      oSrcSize : nppdefs_h.NppiSize;
      oSrcOffset : nppdefs_h.NppiPoint;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pMask : access nppdefs_h.Npp8u;
      oMaskSize : nppdefs_h.NppiSize;
      oAnchor : nppdefs_h.NppiPoint;
      eBorderType : nppdefs_h.NppiBorderType) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:385
   pragma Import (C, nppiDilateBorder_8u_C4R, "nppiDilateBorder_8u_C4R");

  --*
  -- * Four-channel 8-bit unsigned integer dilation with border control, ignoring alpha-channel.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Source image width and height in pixels relative to pSrc.
  -- * \param oSrcOffset Source image starting point relative to pSrc. 
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pMask Pointer to the start address of the mask array
  -- * \param oMaskSize Width and Height mask array.
  -- * \param oAnchor X and Y offsets of the mask origin frame of reference
  -- *        w.r.t the source pixel.
  -- * \param eBorderType The border type operation to be applied at source image border boundaries.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDilateBorder_8u_AC4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      oSrcSize : nppdefs_h.NppiSize;
      oSrcOffset : nppdefs_h.NppiPoint;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pMask : access nppdefs_h.Npp8u;
      oMaskSize : nppdefs_h.NppiSize;
      oAnchor : nppdefs_h.NppiPoint;
      eBorderType : nppdefs_h.NppiBorderType) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:406
   pragma Import (C, nppiDilateBorder_8u_AC4R, "nppiDilateBorder_8u_AC4R");

  --*
  -- * Single-channel 16-bit unsigned integer dilation with border control.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Source image width and height in pixels relative to pSrc.
  -- * \param oSrcOffset Source image starting point relative to pSrc. 
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pMask Pointer to the start address of the mask array
  -- * \param oMaskSize Width and Height mask array.
  -- * \param oAnchor X and Y offsets of the mask origin frame of reference
  -- *        w.r.t the source pixel.
  -- * \param eBorderType The border type operation to be applied at source image border boundaries.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDilateBorder_16u_C1R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : nppdefs_h.Npp32s;
      oSrcSize : nppdefs_h.NppiSize;
      oSrcOffset : nppdefs_h.NppiPoint;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : nppdefs_h.Npp32s;
      oSizeROI : nppdefs_h.NppiSize;
      pMask : access nppdefs_h.Npp8u;
      oMaskSize : nppdefs_h.NppiSize;
      oAnchor : nppdefs_h.NppiPoint;
      eBorderType : nppdefs_h.NppiBorderType) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:428
   pragma Import (C, nppiDilateBorder_16u_C1R, "nppiDilateBorder_16u_C1R");

  --*
  -- * Three-channel 16-bit unsigned integer dilation with border control.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Source image width and height in pixels relative to pSrc.
  -- * \param oSrcOffset Source image starting point relative to pSrc. 
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pMask Pointer to the start address of the mask array
  -- * \param oMaskSize Width and Height mask array.
  -- * \param oAnchor X and Y offsets of the mask origin frame of reference
  -- *        w.r.t the source pixel.
  -- * \param eBorderType The border type operation to be applied at source image border boundaries.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDilateBorder_16u_C3R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : nppdefs_h.Npp32s;
      oSrcSize : nppdefs_h.NppiSize;
      oSrcOffset : nppdefs_h.NppiPoint;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : nppdefs_h.Npp32s;
      oSizeROI : nppdefs_h.NppiSize;
      pMask : access nppdefs_h.Npp8u;
      oMaskSize : nppdefs_h.NppiSize;
      oAnchor : nppdefs_h.NppiPoint;
      eBorderType : nppdefs_h.NppiBorderType) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:449
   pragma Import (C, nppiDilateBorder_16u_C3R, "nppiDilateBorder_16u_C3R");

  --*
  -- * Four-channel 16-bit unsigned integer dilation with border control.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Source image width and height in pixels relative to pSrc.
  -- * \param oSrcOffset Source image starting point relative to pSrc. 
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pMask Pointer to the start address of the mask array
  -- * \param oMaskSize Width and Height mask array.
  -- * \param oAnchor X and Y offsets of the mask origin frame of reference
  -- *        w.r.t the source pixel.
  -- * \param eBorderType The border type operation to be applied at source image border boundaries.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDilateBorder_16u_C4R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      oSrcSize : nppdefs_h.NppiSize;
      oSrcOffset : nppdefs_h.NppiPoint;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pMask : access nppdefs_h.Npp8u;
      oMaskSize : nppdefs_h.NppiSize;
      oAnchor : nppdefs_h.NppiPoint;
      eBorderType : nppdefs_h.NppiBorderType) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:470
   pragma Import (C, nppiDilateBorder_16u_C4R, "nppiDilateBorder_16u_C4R");

  --*
  -- * Four-channel 16-bit unsigned integer dilation with border control, ignoring alpha-channel.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Source image width and height in pixels relative to pSrc.
  -- * \param oSrcOffset Source image starting point relative to pSrc. 
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pMask Pointer to the start address of the mask array
  -- * \param oMaskSize Width and Height mask array.
  -- * \param oAnchor X and Y offsets of the mask origin frame of reference
  -- *        w.r.t the source pixel.
  -- * \param eBorderType The border type operation to be applied at source image border boundaries.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDilateBorder_16u_AC4R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      oSrcSize : nppdefs_h.NppiSize;
      oSrcOffset : nppdefs_h.NppiPoint;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pMask : access nppdefs_h.Npp8u;
      oMaskSize : nppdefs_h.NppiSize;
      oAnchor : nppdefs_h.NppiPoint;
      eBorderType : nppdefs_h.NppiBorderType) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:491
   pragma Import (C, nppiDilateBorder_16u_AC4R, "nppiDilateBorder_16u_AC4R");

  --*
  -- * Single-channel 32-bit floating-point dilation with border control.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Source image width and height in pixels relative to pSrc.
  -- * \param oSrcOffset Source image starting point relative to pSrc. 
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pMask Pointer to the start address of the mask array
  -- * \param oMaskSize Width and Height mask array.
  -- * \param oAnchor X and Y offsets of the mask origin frame of reference
  -- *        w.r.t the source pixel.
  -- * \param eBorderType The border type operation to be applied at source image border boundaries.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDilateBorder_32f_C1R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : nppdefs_h.Npp32s;
      oSrcSize : nppdefs_h.NppiSize;
      oSrcOffset : nppdefs_h.NppiPoint;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : nppdefs_h.Npp32s;
      oSizeROI : nppdefs_h.NppiSize;
      pMask : access nppdefs_h.Npp8u;
      oMaskSize : nppdefs_h.NppiSize;
      oAnchor : nppdefs_h.NppiPoint;
      eBorderType : nppdefs_h.NppiBorderType) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:513
   pragma Import (C, nppiDilateBorder_32f_C1R, "nppiDilateBorder_32f_C1R");

  --*
  -- * Three-channel 32-bit floating-point dilation with border control.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Source image width and height in pixels relative to pSrc.
  -- * \param oSrcOffset Source image starting point relative to pSrc. 
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pMask Pointer to the start address of the mask array
  -- * \param oMaskSize Width and Height mask array.
  -- * \param oAnchor X and Y offsets of the mask origin frame of reference
  -- *        w.r.t the source pixel.
  -- * \param eBorderType The border type operation to be applied at source image border boundaries.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDilateBorder_32f_C3R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : nppdefs_h.Npp32s;
      oSrcSize : nppdefs_h.NppiSize;
      oSrcOffset : nppdefs_h.NppiPoint;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : nppdefs_h.Npp32s;
      oSizeROI : nppdefs_h.NppiSize;
      pMask : access nppdefs_h.Npp8u;
      oMaskSize : nppdefs_h.NppiSize;
      oAnchor : nppdefs_h.NppiPoint;
      eBorderType : nppdefs_h.NppiBorderType) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:534
   pragma Import (C, nppiDilateBorder_32f_C3R, "nppiDilateBorder_32f_C3R");

  --*
  -- * Four-channel 32-bit floating-point dilation with border control.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Source image width and height in pixels relative to pSrc.
  -- * \param oSrcOffset Source image starting point relative to pSrc. 
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pMask Pointer to the start address of the mask array
  -- * \param oMaskSize Width and Height mask array.
  -- * \param oAnchor X and Y offsets of the mask origin frame of reference
  -- *        w.r.t the source pixel.
  -- * \param eBorderType The border type operation to be applied at source image border boundaries.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDilateBorder_32f_C4R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      oSrcSize : nppdefs_h.NppiSize;
      oSrcOffset : nppdefs_h.NppiPoint;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pMask : access nppdefs_h.Npp8u;
      oMaskSize : nppdefs_h.NppiSize;
      oAnchor : nppdefs_h.NppiPoint;
      eBorderType : nppdefs_h.NppiBorderType) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:555
   pragma Import (C, nppiDilateBorder_32f_C4R, "nppiDilateBorder_32f_C4R");

  --*
  -- * Four-channel 32-bit floating-point dilation with border control, ignoring alpha-channel.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Source image width and height in pixels relative to pSrc.
  -- * \param oSrcOffset Source image starting point relative to pSrc. 
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pMask Pointer to the start address of the mask array
  -- * \param oMaskSize Width and Height mask array.
  -- * \param oAnchor X and Y offsets of the mask origin frame of reference
  -- *        w.r.t the source pixel.
  -- * \param eBorderType The border type operation to be applied at source image border boundaries.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDilateBorder_32f_AC4R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      oSrcSize : nppdefs_h.NppiSize;
      oSrcOffset : nppdefs_h.NppiPoint;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pMask : access nppdefs_h.Npp8u;
      oMaskSize : nppdefs_h.NppiSize;
      oAnchor : nppdefs_h.NppiPoint;
      eBorderType : nppdefs_h.NppiBorderType) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:576
   pragma Import (C, nppiDilateBorder_32f_AC4R, "nppiDilateBorder_32f_AC4R");

  --* @} image_dilate_border  
  --* @defgroup image_dilate_3x3 Dilate3x3
  -- *
  -- * Dilation using a 3x3 mask with the anchor at its center pixel.
  -- *
  -- * It is the user's responsibility to avoid \ref sampling_beyond_image_boundaries.
  -- *
  -- * @{
  -- *
  --  

  --*
  -- * Single-channel 8-bit unsigned integer 3x3 dilation.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDilate3x3_8u_C1R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : nppdefs_h.Npp32s;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : nppdefs_h.Npp32s;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:603
   pragma Import (C, nppiDilate3x3_8u_C1R, "nppiDilate3x3_8u_C1R");

  --*
  -- * Three-channel 8-bit unsigned integer 3x3 dilation.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDilate3x3_8u_C3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : nppdefs_h.Npp32s;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : nppdefs_h.Npp32s;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:616
   pragma Import (C, nppiDilate3x3_8u_C3R, "nppiDilate3x3_8u_C3R");

  --*
  -- * Four-channel 8-bit unsigned integer 3x3 dilation.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDilate3x3_8u_C4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:629
   pragma Import (C, nppiDilate3x3_8u_C4R, "nppiDilate3x3_8u_C4R");

  --*
  -- * Four-channel 8-bit unsigned integer 3x3 dilation, ignoring alpha-channel.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDilate3x3_8u_AC4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:642
   pragma Import (C, nppiDilate3x3_8u_AC4R, "nppiDilate3x3_8u_AC4R");

  --*
  -- * Single-channel 16-bit unsigned integer 3x3 dilation.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDilate3x3_16u_C1R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : nppdefs_h.Npp32s;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : nppdefs_h.Npp32s;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:656
   pragma Import (C, nppiDilate3x3_16u_C1R, "nppiDilate3x3_16u_C1R");

  --*
  -- * Three-channel 16-bit unsigned integer 3x3 dilation.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDilate3x3_16u_C3R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : nppdefs_h.Npp32s;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : nppdefs_h.Npp32s;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:669
   pragma Import (C, nppiDilate3x3_16u_C3R, "nppiDilate3x3_16u_C3R");

  --*
  -- * Four-channel 16-bit unsigned integer 3x3 dilation.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDilate3x3_16u_C4R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:682
   pragma Import (C, nppiDilate3x3_16u_C4R, "nppiDilate3x3_16u_C4R");

  --*
  -- * Four-channel 16-bit unsigned integer 3x3 dilation, ignoring alpha-channel.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDilate3x3_16u_AC4R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:695
   pragma Import (C, nppiDilate3x3_16u_AC4R, "nppiDilate3x3_16u_AC4R");

  --*
  -- * Single-channel 32-bit floating-point 3x3 dilation.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDilate3x3_32f_C1R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : nppdefs_h.Npp32s;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : nppdefs_h.Npp32s;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:709
   pragma Import (C, nppiDilate3x3_32f_C1R, "nppiDilate3x3_32f_C1R");

  --*
  -- * Three-channel 32-bit floating-point 3x3 dilation.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDilate3x3_32f_C3R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : nppdefs_h.Npp32s;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : nppdefs_h.Npp32s;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:722
   pragma Import (C, nppiDilate3x3_32f_C3R, "nppiDilate3x3_32f_C3R");

  --*
  -- * Four-channel 32-bit floating-point 3x3 dilation.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDilate3x3_32f_C4R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:735
   pragma Import (C, nppiDilate3x3_32f_C4R, "nppiDilate3x3_32f_C4R");

  --*
  -- * Four-channel 32-bit floating-point 3x3 dilation, ignoring alpha-channel.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDilate3x3_32f_AC4R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:748
   pragma Import (C, nppiDilate3x3_32f_AC4R, "nppiDilate3x3_32f_AC4R");

  --*
  -- * Single-channel 64-bit floating-point 3x3 dilation.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDilate3x3_64f_C1R
     (pSrc : access nppdefs_h.Npp64f;
      nSrcStep : nppdefs_h.Npp32s;
      pDst : access nppdefs_h.Npp64f;
      nDstStep : nppdefs_h.Npp32s;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:761
   pragma Import (C, nppiDilate3x3_64f_C1R, "nppiDilate3x3_64f_C1R");

  --* @} image_dilate_3x3  
  --* @defgroup image_dilate_3x3_border Dilate3x3Border
  -- *
  -- * Dilation using a 3x3 mask with the anchor at its center pixel with border control.
  -- *
  -- * If any portion of the mask overlaps the source image boundary the requested border type 
  -- * operation is applied to all mask pixels which fall outside of the source image.
  -- *
  -- * Currently only the NPP_BORDER_REPLICATE border type operation is supported.
  -- *
  -- * @{
  -- *
  --  

  --*
  -- * Single-channel 8-bit unsigned integer 3x3 dilation with border control.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Source image width and height in pixels relative to pSrc.
  -- * \param oSrcOffset Source image starting point relative to pSrc. 
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eBorderType The border type operation to be applied at source image border boundaries.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDilate3x3Border_8u_C1R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : nppdefs_h.Npp32s;
      oSrcSize : nppdefs_h.NppiSize;
      oSrcOffset : nppdefs_h.NppiPoint;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : nppdefs_h.Npp32s;
      oSizeROI : nppdefs_h.NppiSize;
      eBorderType : nppdefs_h.NppiBorderType) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:792
   pragma Import (C, nppiDilate3x3Border_8u_C1R, "nppiDilate3x3Border_8u_C1R");

  --*
  -- * Three-channel 8-bit unsigned integer 3x3 dilation with border control.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Source image width and height in pixels relative to pSrc.
  -- * \param oSrcOffset Source image starting point relative to pSrc. 
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eBorderType The border type operation to be applied at source image border boundaries.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDilate3x3Border_8u_C3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : nppdefs_h.Npp32s;
      oSrcSize : nppdefs_h.NppiSize;
      oSrcOffset : nppdefs_h.NppiPoint;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : nppdefs_h.Npp32s;
      oSizeROI : nppdefs_h.NppiSize;
      eBorderType : nppdefs_h.NppiBorderType) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:808
   pragma Import (C, nppiDilate3x3Border_8u_C3R, "nppiDilate3x3Border_8u_C3R");

  --*
  -- * Four-channel 8-bit unsigned integer 3x3 dilation with border control.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Source image width and height in pixels relative to pSrc.
  -- * \param oSrcOffset Source image starting point relative to pSrc. 
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eBorderType The border type operation to be applied at source image border boundaries.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDilate3x3Border_8u_C4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      oSrcSize : nppdefs_h.NppiSize;
      oSrcOffset : nppdefs_h.NppiPoint;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eBorderType : nppdefs_h.NppiBorderType) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:824
   pragma Import (C, nppiDilate3x3Border_8u_C4R, "nppiDilate3x3Border_8u_C4R");

  --*
  -- * Four-channel 8-bit unsigned integer 3x3 dilation with border control, ignoring alpha-channel.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Source image width and height in pixels relative to pSrc.
  -- * \param oSrcOffset Source image starting point relative to pSrc. 
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eBorderType The border type operation to be applied at source image border boundaries.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDilate3x3Border_8u_AC4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      oSrcSize : nppdefs_h.NppiSize;
      oSrcOffset : nppdefs_h.NppiPoint;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eBorderType : nppdefs_h.NppiBorderType) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:840
   pragma Import (C, nppiDilate3x3Border_8u_AC4R, "nppiDilate3x3Border_8u_AC4R");

  --*
  -- * Single-channel 16-bit unsigned integer 3x3 dilation with border control.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Source image width and height in pixels relative to pSrc.
  -- * \param oSrcOffset Source image starting point relative to pSrc. 
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eBorderType The border type operation to be applied at source image border boundaries.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDilate3x3Border_16u_C1R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : nppdefs_h.Npp32s;
      oSrcSize : nppdefs_h.NppiSize;
      oSrcOffset : nppdefs_h.NppiPoint;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : nppdefs_h.Npp32s;
      oSizeROI : nppdefs_h.NppiSize;
      eBorderType : nppdefs_h.NppiBorderType) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:857
   pragma Import (C, nppiDilate3x3Border_16u_C1R, "nppiDilate3x3Border_16u_C1R");

  --*
  -- * Three-channel 16-bit unsigned integer 3x3 dilation with border control.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Source image width and height in pixels relative to pSrc.
  -- * \param oSrcOffset Source image starting point relative to pSrc. 
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eBorderType The border type operation to be applied at source image border boundaries.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDilate3x3Border_16u_C3R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : nppdefs_h.Npp32s;
      oSrcSize : nppdefs_h.NppiSize;
      oSrcOffset : nppdefs_h.NppiPoint;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : nppdefs_h.Npp32s;
      oSizeROI : nppdefs_h.NppiSize;
      eBorderType : nppdefs_h.NppiBorderType) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:873
   pragma Import (C, nppiDilate3x3Border_16u_C3R, "nppiDilate3x3Border_16u_C3R");

  --*
  -- * Four-channel 16-bit unsigned integer 3x3 dilation with border control.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Source image width and height in pixels relative to pSrc.
  -- * \param oSrcOffset Source image starting point relative to pSrc. 
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eBorderType The border type operation to be applied at source image border boundaries.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDilate3x3Border_16u_C4R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      oSrcSize : nppdefs_h.NppiSize;
      oSrcOffset : nppdefs_h.NppiPoint;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eBorderType : nppdefs_h.NppiBorderType) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:889
   pragma Import (C, nppiDilate3x3Border_16u_C4R, "nppiDilate3x3Border_16u_C4R");

  --*
  -- * Four-channel 16-bit unsigned integer 3x3 dilation with border control, ignoring alpha-channel.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Source image width and height in pixels relative to pSrc.
  -- * \param oSrcOffset Source image starting point relative to pSrc. 
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eBorderType The border type operation to be applied at source image border boundaries.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDilate3x3Border_16u_AC4R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      oSrcSize : nppdefs_h.NppiSize;
      oSrcOffset : nppdefs_h.NppiPoint;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eBorderType : nppdefs_h.NppiBorderType) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:905
   pragma Import (C, nppiDilate3x3Border_16u_AC4R, "nppiDilate3x3Border_16u_AC4R");

  --*
  -- * Single-channel 32-bit floating-point 3x3 dilation with border control.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Source image width and height in pixels relative to pSrc.
  -- * \param oSrcOffset Source image starting point relative to pSrc. 
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eBorderType The border type operation to be applied at source image border boundaries.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDilate3x3Border_32f_C1R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : nppdefs_h.Npp32s;
      oSrcSize : nppdefs_h.NppiSize;
      oSrcOffset : nppdefs_h.NppiPoint;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : nppdefs_h.Npp32s;
      oSizeROI : nppdefs_h.NppiSize;
      eBorderType : nppdefs_h.NppiBorderType) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:922
   pragma Import (C, nppiDilate3x3Border_32f_C1R, "nppiDilate3x3Border_32f_C1R");

  --*
  -- * Three-channel 32-bit floating-point 3x3 dilation with border control.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Source image width and height in pixels relative to pSrc.
  -- * \param oSrcOffset Source image starting point relative to pSrc. 
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eBorderType The border type operation to be applied at source image border boundaries.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDilate3x3Border_32f_C3R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : nppdefs_h.Npp32s;
      oSrcSize : nppdefs_h.NppiSize;
      oSrcOffset : nppdefs_h.NppiPoint;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : nppdefs_h.Npp32s;
      oSizeROI : nppdefs_h.NppiSize;
      eBorderType : nppdefs_h.NppiBorderType) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:938
   pragma Import (C, nppiDilate3x3Border_32f_C3R, "nppiDilate3x3Border_32f_C3R");

  --*
  -- * Four-channel 32-bit floating-point 3x3 dilation with border control.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Source image width and height in pixels relative to pSrc.
  -- * \param oSrcOffset Source image starting point relative to pSrc. 
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eBorderType The border type operation to be applied at source image border boundaries.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDilate3x3Border_32f_C4R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      oSrcSize : nppdefs_h.NppiSize;
      oSrcOffset : nppdefs_h.NppiPoint;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eBorderType : nppdefs_h.NppiBorderType) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:954
   pragma Import (C, nppiDilate3x3Border_32f_C4R, "nppiDilate3x3Border_32f_C4R");

  --*
  -- * Four-channel 32-bit floating-point 3x3 dilation with border control, ignoring alpha-channel.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Source image width and height in pixels relative to pSrc.
  -- * \param oSrcOffset Source image starting point relative to pSrc. 
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eBorderType The border type operation to be applied at source image border boundaries.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDilate3x3Border_32f_AC4R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      oSrcSize : nppdefs_h.NppiSize;
      oSrcOffset : nppdefs_h.NppiPoint;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eBorderType : nppdefs_h.NppiBorderType) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:970
   pragma Import (C, nppiDilate3x3Border_32f_AC4R, "nppiDilate3x3Border_32f_AC4R");

  --* @} image_dilate_3x3_border  
  --* @defgroup image_erode Erode
  -- *
  -- * Erosion computes the output pixel as the minimum pixel value of the pixels
  -- * under the mask. Pixels who's corresponding mask values are zero do not 
  -- * participate in the maximum search.
  -- *
  -- * It is the user's responsibility to avoid \ref sampling_beyond_image_boundaries.
  -- *
  -- * @{
  -- *
  --  

  --*
  -- * Single-channel 8-bit unsigned integer erosion.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pMask Pointer to the start address of the mask array
  -- * \param oMaskSize Width and Height mask array.
  -- * \param oAnchor X and Y offsets of the mask origin frame of reference
  -- *        w.r.t the source pixel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiErode_8u_C1R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : nppdefs_h.Npp32s;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : nppdefs_h.Npp32s;
      oSizeROI : nppdefs_h.NppiSize;
      pMask : access nppdefs_h.Npp8u;
      oMaskSize : nppdefs_h.NppiSize;
      oAnchor : nppdefs_h.NppiPoint) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:1002
   pragma Import (C, nppiErode_8u_C1R, "nppiErode_8u_C1R");

  --*
  -- * Three-channel 8-bit unsigned integer erosion.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pMask Pointer to the start address of the mask array
  -- * \param oMaskSize Width and Height mask array.
  -- * \param oAnchor X and Y offsets of the mask origin frame of reference
  -- *        w.r.t the source pixel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiErode_8u_C3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : nppdefs_h.Npp32s;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : nppdefs_h.Npp32s;
      oSizeROI : nppdefs_h.NppiSize;
      pMask : access nppdefs_h.Npp8u;
      oMaskSize : nppdefs_h.NppiSize;
      oAnchor : nppdefs_h.NppiPoint) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:1020
   pragma Import (C, nppiErode_8u_C3R, "nppiErode_8u_C3R");

  --*
  -- * Four-channel 8-bit unsigned integer erosion.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pMask Pointer to the start address of the mask array
  -- * \param oMaskSize Width and Height mask array.
  -- * \param oAnchor X and Y offsets of the mask origin frame of reference
  -- *        w.r.t the source pixel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiErode_8u_C4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pMask : access nppdefs_h.Npp8u;
      oMaskSize : nppdefs_h.NppiSize;
      oAnchor : nppdefs_h.NppiPoint) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:1038
   pragma Import (C, nppiErode_8u_C4R, "nppiErode_8u_C4R");

  --*
  -- * Four-channel 8-bit unsigned integer erosion, ignoring alpha-channel.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pMask Pointer to the start address of the mask array
  -- * \param oMaskSize Width and Height mask array.
  -- * \param oAnchor X and Y offsets of the mask origin frame of reference
  -- *        w.r.t the source pixel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiErode_8u_AC4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pMask : access nppdefs_h.Npp8u;
      oMaskSize : nppdefs_h.NppiSize;
      oAnchor : nppdefs_h.NppiPoint) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:1056
   pragma Import (C, nppiErode_8u_AC4R, "nppiErode_8u_AC4R");

  --*
  -- * Single-channel 16-bit unsigned integer erosion.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pMask Pointer to the start address of the mask array
  -- * \param oMaskSize Width and Height mask array.
  -- * \param oAnchor X and Y offsets of the mask origin frame of reference
  -- *        w.r.t the source pixel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiErode_16u_C1R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : nppdefs_h.Npp32s;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : nppdefs_h.Npp32s;
      oSizeROI : nppdefs_h.NppiSize;
      pMask : access nppdefs_h.Npp8u;
      oMaskSize : nppdefs_h.NppiSize;
      oAnchor : nppdefs_h.NppiPoint) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:1075
   pragma Import (C, nppiErode_16u_C1R, "nppiErode_16u_C1R");

  --*
  -- * Three-channel 16-bit unsigned integer erosion.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pMask Pointer to the start address of the mask array
  -- * \param oMaskSize Width and Height mask array.
  -- * \param oAnchor X and Y offsets of the mask origin frame of reference
  -- *        w.r.t the source pixel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiErode_16u_C3R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : nppdefs_h.Npp32s;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : nppdefs_h.Npp32s;
      oSizeROI : nppdefs_h.NppiSize;
      pMask : access nppdefs_h.Npp8u;
      oMaskSize : nppdefs_h.NppiSize;
      oAnchor : nppdefs_h.NppiPoint) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:1093
   pragma Import (C, nppiErode_16u_C3R, "nppiErode_16u_C3R");

  --*
  -- * Four-channel 16-bit unsigned integer erosion.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pMask Pointer to the start address of the mask array
  -- * \param oMaskSize Width and Height mask array.
  -- * \param oAnchor X and Y offsets of the mask origin frame of reference
  -- *        w.r.t the source pixel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiErode_16u_C4R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pMask : access nppdefs_h.Npp8u;
      oMaskSize : nppdefs_h.NppiSize;
      oAnchor : nppdefs_h.NppiPoint) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:1111
   pragma Import (C, nppiErode_16u_C4R, "nppiErode_16u_C4R");

  --*
  -- * Four-channel 16-bit unsigned integer erosion, ignoring alpha-channel.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pMask Pointer to the start address of the mask array
  -- * \param oMaskSize Width and Height mask array.
  -- * \param oAnchor X and Y offsets of the mask origin frame of reference
  -- *        w.r.t the source pixel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiErode_16u_AC4R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pMask : access nppdefs_h.Npp8u;
      oMaskSize : nppdefs_h.NppiSize;
      oAnchor : nppdefs_h.NppiPoint) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:1129
   pragma Import (C, nppiErode_16u_AC4R, "nppiErode_16u_AC4R");

  --*
  -- * Single-channel 32-bit floating-point erosion.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pMask Pointer to the start address of the mask array
  -- * \param oMaskSize Width and Height mask array.
  -- * \param oAnchor X and Y offsets of the mask origin frame of reference
  -- *        w.r.t the source pixel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiErode_32f_C1R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : nppdefs_h.Npp32s;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : nppdefs_h.Npp32s;
      oSizeROI : nppdefs_h.NppiSize;
      pMask : access nppdefs_h.Npp8u;
      oMaskSize : nppdefs_h.NppiSize;
      oAnchor : nppdefs_h.NppiPoint) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:1148
   pragma Import (C, nppiErode_32f_C1R, "nppiErode_32f_C1R");

  --*
  -- * Three-channel 32-bit floating-point erosion.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pMask Pointer to the start address of the mask array
  -- * \param oMaskSize Width and Height mask array.
  -- * \param oAnchor X and Y offsets of the mask origin frame of reference
  -- *        w.r.t the source pixel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiErode_32f_C3R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : nppdefs_h.Npp32s;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : nppdefs_h.Npp32s;
      oSizeROI : nppdefs_h.NppiSize;
      pMask : access nppdefs_h.Npp8u;
      oMaskSize : nppdefs_h.NppiSize;
      oAnchor : nppdefs_h.NppiPoint) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:1166
   pragma Import (C, nppiErode_32f_C3R, "nppiErode_32f_C3R");

  --*
  -- * Four-channel 32-bit floating-point erosion.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pMask Pointer to the start address of the mask array
  -- * \param oMaskSize Width and Height mask array.
  -- * \param oAnchor X and Y offsets of the mask origin frame of reference
  -- *        w.r.t the source pixel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiErode_32f_C4R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pMask : access nppdefs_h.Npp8u;
      oMaskSize : nppdefs_h.NppiSize;
      oAnchor : nppdefs_h.NppiPoint) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:1184
   pragma Import (C, nppiErode_32f_C4R, "nppiErode_32f_C4R");

  --*
  -- * Four-channel 32-bit floating-point erosion, ignoring alpha-channel.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pMask Pointer to the start address of the mask array
  -- * \param oMaskSize Width and Height mask array.
  -- * \param oAnchor X and Y offsets of the mask origin frame of reference
  -- *        w.r.t the source pixel.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiErode_32f_AC4R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pMask : access nppdefs_h.Npp8u;
      oMaskSize : nppdefs_h.NppiSize;
      oAnchor : nppdefs_h.NppiPoint) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:1202
   pragma Import (C, nppiErode_32f_AC4R, "nppiErode_32f_AC4R");

  --* @} image_erode  
  --* @defgroup image_erode_border Erosion with border control
  -- *
  -- * Erosion computes the output pixel as the minimum pixel value of the pixels
  -- * under the mask. Pixels who's corresponding mask values are zero do not 
  -- * participate in the minimum search.
  -- *
  -- * If any portion of the mask overlaps the source image boundary the requested border type 
  -- * operation is applied to all mask pixels which fall outside of the source image.
  -- *
  -- * Currently only the NPP_BORDER_REPLICATE border type operation is supported.
  -- *
  -- * @{
  -- *
  --  

  --*
  -- * Single-channel 8-bit unsigned integer erosion with border control.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Source image width and height in pixels relative to pSrc.
  -- * \param oSrcOffset Source image starting point relative to pSrc. 
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pMask Pointer to the start address of the mask array
  -- * \param oMaskSize Width and Height mask array.
  -- * \param oAnchor X and Y offsets of the mask origin frame of reference
  -- *        w.r.t the source pixel.
  -- * \param eBorderType The border type operation to be applied at source image border boundaries.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiErodeBorder_8u_C1R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : nppdefs_h.Npp32s;
      oSrcSize : nppdefs_h.NppiSize;
      oSrcOffset : nppdefs_h.NppiPoint;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : nppdefs_h.Npp32s;
      oSizeROI : nppdefs_h.NppiSize;
      pMask : access nppdefs_h.Npp8u;
      oMaskSize : nppdefs_h.NppiSize;
      oAnchor : nppdefs_h.NppiPoint;
      eBorderType : nppdefs_h.NppiBorderType) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:1241
   pragma Import (C, nppiErodeBorder_8u_C1R, "nppiErodeBorder_8u_C1R");

  --*
  -- * Three-channel 8-bit unsigned integer erosion with border control.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Source image width and height in pixels relative to pSrc.
  -- * \param oSrcOffset Source image starting point relative to pSrc. 
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pMask Pointer to the start address of the mask array
  -- * \param oMaskSize Width and Height mask array.
  -- * \param oAnchor X and Y offsets of the mask origin frame of reference
  -- *        w.r.t the source pixel.
  -- * \param eBorderType The border type operation to be applied at source image border boundaries.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiErodeBorder_8u_C3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : nppdefs_h.Npp32s;
      oSrcSize : nppdefs_h.NppiSize;
      oSrcOffset : nppdefs_h.NppiPoint;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : nppdefs_h.Npp32s;
      oSizeROI : nppdefs_h.NppiSize;
      pMask : access nppdefs_h.Npp8u;
      oMaskSize : nppdefs_h.NppiSize;
      oAnchor : nppdefs_h.NppiPoint;
      eBorderType : nppdefs_h.NppiBorderType) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:1262
   pragma Import (C, nppiErodeBorder_8u_C3R, "nppiErodeBorder_8u_C3R");

  --*
  -- * Four-channel 8-bit unsigned integer erosion with border control.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Source image width and height in pixels relative to pSrc.
  -- * \param oSrcOffset Source image starting point relative to pSrc. 
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pMask Pointer to the start address of the mask array
  -- * \param oMaskSize Width and Height mask array.
  -- * \param oAnchor X and Y offsets of the mask origin frame of reference
  -- *        w.r.t the source pixel.
  -- * \param eBorderType The border type operation to be applied at source image border boundaries.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiErodeBorder_8u_C4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      oSrcSize : nppdefs_h.NppiSize;
      oSrcOffset : nppdefs_h.NppiPoint;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pMask : access nppdefs_h.Npp8u;
      oMaskSize : nppdefs_h.NppiSize;
      oAnchor : nppdefs_h.NppiPoint;
      eBorderType : nppdefs_h.NppiBorderType) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:1283
   pragma Import (C, nppiErodeBorder_8u_C4R, "nppiErodeBorder_8u_C4R");

  --*
  -- * Four-channel 8-bit unsigned integer erosion with border control, ignoring alpha-channel.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Source image width and height in pixels relative to pSrc.
  -- * \param oSrcOffset Source image starting point relative to pSrc. 
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pMask Pointer to the start address of the mask array
  -- * \param oMaskSize Width and Height mask array.
  -- * \param oAnchor X and Y offsets of the mask origin frame of reference
  -- *        w.r.t the source pixel.
  -- * \param eBorderType The border type operation to be applied at source image border boundaries.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiErodeBorder_8u_AC4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      oSrcSize : nppdefs_h.NppiSize;
      oSrcOffset : nppdefs_h.NppiPoint;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pMask : access nppdefs_h.Npp8u;
      oMaskSize : nppdefs_h.NppiSize;
      oAnchor : nppdefs_h.NppiPoint;
      eBorderType : nppdefs_h.NppiBorderType) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:1304
   pragma Import (C, nppiErodeBorder_8u_AC4R, "nppiErodeBorder_8u_AC4R");

  --*
  -- * Single-channel 16-bit unsigned integer erosion with border control.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Source image width and height in pixels relative to pSrc.
  -- * \param oSrcOffset Source image starting point relative to pSrc. 
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pMask Pointer to the start address of the mask array
  -- * \param oMaskSize Width and Height mask array.
  -- * \param oAnchor X and Y offsets of the mask origin frame of reference
  -- *        w.r.t the source pixel.
  -- * \param eBorderType The border type operation to be applied at source image border boundaries.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiErodeBorder_16u_C1R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : nppdefs_h.Npp32s;
      oSrcSize : nppdefs_h.NppiSize;
      oSrcOffset : nppdefs_h.NppiPoint;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : nppdefs_h.Npp32s;
      oSizeROI : nppdefs_h.NppiSize;
      pMask : access nppdefs_h.Npp8u;
      oMaskSize : nppdefs_h.NppiSize;
      oAnchor : nppdefs_h.NppiPoint;
      eBorderType : nppdefs_h.NppiBorderType) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:1326
   pragma Import (C, nppiErodeBorder_16u_C1R, "nppiErodeBorder_16u_C1R");

  --*
  -- * Three-channel 16-bit unsigned integer erosion with border control.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Source image width and height in pixels relative to pSrc.
  -- * \param oSrcOffset Source image starting point relative to pSrc. 
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pMask Pointer to the start address of the mask array
  -- * \param oMaskSize Width and Height mask array.
  -- * \param oAnchor X and Y offsets of the mask origin frame of reference
  -- *        w.r.t the source pixel.
  -- * \param eBorderType The border type operation to be applied at source image border boundaries.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiErodeBorder_16u_C3R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : nppdefs_h.Npp32s;
      oSrcSize : nppdefs_h.NppiSize;
      oSrcOffset : nppdefs_h.NppiPoint;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : nppdefs_h.Npp32s;
      oSizeROI : nppdefs_h.NppiSize;
      pMask : access nppdefs_h.Npp8u;
      oMaskSize : nppdefs_h.NppiSize;
      oAnchor : nppdefs_h.NppiPoint;
      eBorderType : nppdefs_h.NppiBorderType) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:1347
   pragma Import (C, nppiErodeBorder_16u_C3R, "nppiErodeBorder_16u_C3R");

  --*
  -- * Four-channel 16-bit unsigned integer erosion with border control.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Source image width and height in pixels relative to pSrc.
  -- * \param oSrcOffset Source image starting point relative to pSrc. 
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pMask Pointer to the start address of the mask array
  -- * \param oMaskSize Width and Height mask array.
  -- * \param oAnchor X and Y offsets of the mask origin frame of reference
  -- *        w.r.t the source pixel.
  -- * \param eBorderType The border type operation to be applied at source image border boundaries.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiErodeBorder_16u_C4R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      oSrcSize : nppdefs_h.NppiSize;
      oSrcOffset : nppdefs_h.NppiPoint;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pMask : access nppdefs_h.Npp8u;
      oMaskSize : nppdefs_h.NppiSize;
      oAnchor : nppdefs_h.NppiPoint;
      eBorderType : nppdefs_h.NppiBorderType) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:1368
   pragma Import (C, nppiErodeBorder_16u_C4R, "nppiErodeBorder_16u_C4R");

  --*
  -- * Four-channel 16-bit unsigned integer erosion with border control, ignoring alpha-channel.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Source image width and height in pixels relative to pSrc.
  -- * \param oSrcOffset Source image starting point relative to pSrc. 
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pMask Pointer to the start address of the mask array
  -- * \param oMaskSize Width and Height mask array.
  -- * \param oAnchor X and Y offsets of the mask origin frame of reference
  -- *        w.r.t the source pixel.
  -- * \param eBorderType The border type operation to be applied at source image border boundaries.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiErodeBorder_16u_AC4R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      oSrcSize : nppdefs_h.NppiSize;
      oSrcOffset : nppdefs_h.NppiPoint;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pMask : access nppdefs_h.Npp8u;
      oMaskSize : nppdefs_h.NppiSize;
      oAnchor : nppdefs_h.NppiPoint;
      eBorderType : nppdefs_h.NppiBorderType) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:1389
   pragma Import (C, nppiErodeBorder_16u_AC4R, "nppiErodeBorder_16u_AC4R");

  --*
  -- * Single-channel 32-bit floating-point erosion with border control.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Source image width and height in pixels relative to pSrc.
  -- * \param oSrcOffset Source image starting point relative to pSrc. 
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pMask Pointer to the start address of the mask array
  -- * \param oMaskSize Width and Height mask array.
  -- * \param oAnchor X and Y offsets of the mask origin frame of reference
  -- *        w.r.t the source pixel.
  -- * \param eBorderType The border type operation to be applied at source image border boundaries.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiErodeBorder_32f_C1R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : nppdefs_h.Npp32s;
      oSrcSize : nppdefs_h.NppiSize;
      oSrcOffset : nppdefs_h.NppiPoint;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : nppdefs_h.Npp32s;
      oSizeROI : nppdefs_h.NppiSize;
      pMask : access nppdefs_h.Npp8u;
      oMaskSize : nppdefs_h.NppiSize;
      oAnchor : nppdefs_h.NppiPoint;
      eBorderType : nppdefs_h.NppiBorderType) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:1411
   pragma Import (C, nppiErodeBorder_32f_C1R, "nppiErodeBorder_32f_C1R");

  --*
  -- * Three-channel 32-bit floating-point erosion with border control.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Source image width and height in pixels relative to pSrc.
  -- * \param oSrcOffset Source image starting point relative to pSrc. 
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pMask Pointer to the start address of the mask array
  -- * \param oMaskSize Width and Height mask array.
  -- * \param oAnchor X and Y offsets of the mask origin frame of reference
  -- *        w.r.t the source pixel.
  -- * \param eBorderType The border type operation to be applied at source image border boundaries.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiErodeBorder_32f_C3R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : nppdefs_h.Npp32s;
      oSrcSize : nppdefs_h.NppiSize;
      oSrcOffset : nppdefs_h.NppiPoint;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : nppdefs_h.Npp32s;
      oSizeROI : nppdefs_h.NppiSize;
      pMask : access nppdefs_h.Npp8u;
      oMaskSize : nppdefs_h.NppiSize;
      oAnchor : nppdefs_h.NppiPoint;
      eBorderType : nppdefs_h.NppiBorderType) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:1432
   pragma Import (C, nppiErodeBorder_32f_C3R, "nppiErodeBorder_32f_C3R");

  --*
  -- * Four-channel 32-bit floating-point erosion with border control.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Source image width and height in pixels relative to pSrc.
  -- * \param oSrcOffset Source image starting point relative to pSrc. 
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pMask Pointer to the start address of the mask array
  -- * \param oMaskSize Width and Height mask array.
  -- * \param oAnchor X and Y offsets of the mask origin frame of reference
  -- *        w.r.t the source pixel.
  -- * \param eBorderType The border type operation to be applied at source image border boundaries.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiErodeBorder_32f_C4R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      oSrcSize : nppdefs_h.NppiSize;
      oSrcOffset : nppdefs_h.NppiPoint;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pMask : access nppdefs_h.Npp8u;
      oMaskSize : nppdefs_h.NppiSize;
      oAnchor : nppdefs_h.NppiPoint;
      eBorderType : nppdefs_h.NppiBorderType) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:1453
   pragma Import (C, nppiErodeBorder_32f_C4R, "nppiErodeBorder_32f_C4R");

  --*
  -- * Four-channel 32-bit floating-point erosion with border control, ignoring alpha-channel.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Source image width and height in pixels relative to pSrc.
  -- * \param oSrcOffset Source image starting point relative to pSrc. 
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pMask Pointer to the start address of the mask array
  -- * \param oMaskSize Width and Height mask array.
  -- * \param oAnchor X and Y offsets of the mask origin frame of reference
  -- *        w.r.t the source pixel.
  -- * \param eBorderType The border type operation to be applied at source image border boundaries.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiErodeBorder_32f_AC4R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      oSrcSize : nppdefs_h.NppiSize;
      oSrcOffset : nppdefs_h.NppiPoint;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pMask : access nppdefs_h.Npp8u;
      oMaskSize : nppdefs_h.NppiSize;
      oAnchor : nppdefs_h.NppiPoint;
      eBorderType : nppdefs_h.NppiBorderType) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:1474
   pragma Import (C, nppiErodeBorder_32f_AC4R, "nppiErodeBorder_32f_AC4R");

  --* @} image_erode_border  
  --* @defgroup image_erode_3x3 Erode3x3
  -- *
  -- * Erosion using a 3x3 mask with the anchor at its center pixel.
  -- *
  -- * It is the user's responsibility to avoid \ref sampling_beyond_image_boundaries.
  -- *
  -- * @{
  -- *
  --  

  --*
  -- * Single-channel 8-bit unsigned integer 3x3 erosion.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiErode3x3_8u_C1R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : nppdefs_h.Npp32s;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : nppdefs_h.Npp32s;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:1502
   pragma Import (C, nppiErode3x3_8u_C1R, "nppiErode3x3_8u_C1R");

  --*
  -- * Three-channel 8-bit unsigned integer 3x3 erosion.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiErode3x3_8u_C3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : nppdefs_h.Npp32s;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : nppdefs_h.Npp32s;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:1515
   pragma Import (C, nppiErode3x3_8u_C3R, "nppiErode3x3_8u_C3R");

  --*
  -- * Four-channel 8-bit unsigned integer 3x3 erosion.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiErode3x3_8u_C4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:1528
   pragma Import (C, nppiErode3x3_8u_C4R, "nppiErode3x3_8u_C4R");

  --*
  -- * Four-channel 8-bit unsigned integer 3x3 erosion, ignoring alpha-channel.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiErode3x3_8u_AC4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:1541
   pragma Import (C, nppiErode3x3_8u_AC4R, "nppiErode3x3_8u_AC4R");

  --*
  -- * Single-channel 16-bit unsigned integer 3x3 erosion.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiErode3x3_16u_C1R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : nppdefs_h.Npp32s;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : nppdefs_h.Npp32s;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:1555
   pragma Import (C, nppiErode3x3_16u_C1R, "nppiErode3x3_16u_C1R");

  --*
  -- * Three-channel 16-bit unsigned integer 3x3 erosion.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiErode3x3_16u_C3R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : nppdefs_h.Npp32s;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : nppdefs_h.Npp32s;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:1568
   pragma Import (C, nppiErode3x3_16u_C3R, "nppiErode3x3_16u_C3R");

  --*
  -- * Four-channel 16-bit unsigned integer 3x3 erosion.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiErode3x3_16u_C4R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:1581
   pragma Import (C, nppiErode3x3_16u_C4R, "nppiErode3x3_16u_C4R");

  --*
  -- * Four-channel 16-bit unsigned integer 3x3 erosion, ignoring alpha-channel.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiErode3x3_16u_AC4R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:1594
   pragma Import (C, nppiErode3x3_16u_AC4R, "nppiErode3x3_16u_AC4R");

  --*
  -- * Single-channel 32-bit floating-point 3x3 erosion.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiErode3x3_32f_C1R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : nppdefs_h.Npp32s;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : nppdefs_h.Npp32s;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:1608
   pragma Import (C, nppiErode3x3_32f_C1R, "nppiErode3x3_32f_C1R");

  --*
  -- * Three-channel 32-bit floating-point 3x3 erosion.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiErode3x3_32f_C3R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : nppdefs_h.Npp32s;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : nppdefs_h.Npp32s;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:1621
   pragma Import (C, nppiErode3x3_32f_C3R, "nppiErode3x3_32f_C3R");

  --*
  -- * Four-channel 32-bit floating-point 3x3 erosion.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiErode3x3_32f_C4R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:1634
   pragma Import (C, nppiErode3x3_32f_C4R, "nppiErode3x3_32f_C4R");

  --*
  -- * Four-channel 32-bit floating-point 3x3 erosion, ignoring alpha-channel.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiErode3x3_32f_AC4R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:1647
   pragma Import (C, nppiErode3x3_32f_AC4R, "nppiErode3x3_32f_AC4R");

  --*
  -- * Single-channel 64-bit floating-point 3x3 erosion.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiErode3x3_64f_C1R
     (pSrc : access nppdefs_h.Npp64f;
      nSrcStep : nppdefs_h.Npp32s;
      pDst : access nppdefs_h.Npp64f;
      nDstStep : nppdefs_h.Npp32s;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:1660
   pragma Import (C, nppiErode3x3_64f_C1R, "nppiErode3x3_64f_C1R");

  --* @} image_erode  
  --* @defgroup image_erode_3x3_border Erode3x3Border
  -- *
  -- * Erosion using a 3x3 mask with the anchor at its center pixel with border control.
  -- *
  -- * If any portion of the mask overlaps the source image boundary the requested border type 
  -- * operation is applied to all mask pixels which fall outside of the source image.
  -- *
  -- * Currently only the NPP_BORDER_REPLICATE border type operation is supported.
  -- *
  -- * @{
  -- *
  --  

  --*
  -- * Single-channel 8-bit unsigned integer 3x3 erosion with border control.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Source image width and height in pixels relative to pSrc.
  -- * \param oSrcOffset Source image starting point relative to pSrc. 
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eBorderType The border type operation to be applied at source image border boundaries.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiErode3x3Border_8u_C1R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : nppdefs_h.Npp32s;
      oSrcSize : nppdefs_h.NppiSize;
      oSrcOffset : nppdefs_h.NppiPoint;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : nppdefs_h.Npp32s;
      oSizeROI : nppdefs_h.NppiSize;
      eBorderType : nppdefs_h.NppiBorderType) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:1691
   pragma Import (C, nppiErode3x3Border_8u_C1R, "nppiErode3x3Border_8u_C1R");

  --*
  -- * Three-channel 8-bit unsigned integer 3x3 erosion with border control.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Source image width and height in pixels relative to pSrc.
  -- * \param oSrcOffset Source image starting point relative to pSrc. 
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eBorderType The border type operation to be applied at source image border boundaries.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiErode3x3Border_8u_C3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : nppdefs_h.Npp32s;
      oSrcSize : nppdefs_h.NppiSize;
      oSrcOffset : nppdefs_h.NppiPoint;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : nppdefs_h.Npp32s;
      oSizeROI : nppdefs_h.NppiSize;
      eBorderType : nppdefs_h.NppiBorderType) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:1708
   pragma Import (C, nppiErode3x3Border_8u_C3R, "nppiErode3x3Border_8u_C3R");

  --*
  -- * Four-channel 8-bit unsigned integer 3x3 erosion with border control.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Source image width and height in pixels relative to pSrc.
  -- * \param oSrcOffset Source image starting point relative to pSrc. 
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eBorderType The border type operation to be applied at source image border boundaries.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiErode3x3Border_8u_C4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      oSrcSize : nppdefs_h.NppiSize;
      oSrcOffset : nppdefs_h.NppiPoint;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eBorderType : nppdefs_h.NppiBorderType) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:1725
   pragma Import (C, nppiErode3x3Border_8u_C4R, "nppiErode3x3Border_8u_C4R");

  --*
  -- * Four-channel 8-bit unsigned integer 3x3 erosion with border control, ignoring alpha-channel.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Source image width and height in pixels relative to pSrc.
  -- * \param oSrcOffset Source image starting point relative to pSrc. 
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eBorderType The border type operation to be applied at source image border boundaries.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiErode3x3Border_8u_AC4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      oSrcSize : nppdefs_h.NppiSize;
      oSrcOffset : nppdefs_h.NppiPoint;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eBorderType : nppdefs_h.NppiBorderType) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:1742
   pragma Import (C, nppiErode3x3Border_8u_AC4R, "nppiErode3x3Border_8u_AC4R");

  --*
  -- * Single-channel 16-bit unsigned integer 3x3 erosion with border control.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Source image width and height in pixels relative to pSrc.
  -- * \param oSrcOffset Source image starting point relative to pSrc. 
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eBorderType The border type operation to be applied at source image border boundaries.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiErode3x3Border_16u_C1R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : nppdefs_h.Npp32s;
      oSrcSize : nppdefs_h.NppiSize;
      oSrcOffset : nppdefs_h.NppiPoint;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : nppdefs_h.Npp32s;
      oSizeROI : nppdefs_h.NppiSize;
      eBorderType : nppdefs_h.NppiBorderType) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:1760
   pragma Import (C, nppiErode3x3Border_16u_C1R, "nppiErode3x3Border_16u_C1R");

  --*
  -- * Three-channel 16-bit unsigned integer 3x3 erosion with border control.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Source image width and height in pixels relative to pSrc.
  -- * \param oSrcOffset Source image starting point relative to pSrc. 
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eBorderType The border type operation to be applied at source image border boundaries.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiErode3x3Border_16u_C3R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : nppdefs_h.Npp32s;
      oSrcSize : nppdefs_h.NppiSize;
      oSrcOffset : nppdefs_h.NppiPoint;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : nppdefs_h.Npp32s;
      oSizeROI : nppdefs_h.NppiSize;
      eBorderType : nppdefs_h.NppiBorderType) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:1777
   pragma Import (C, nppiErode3x3Border_16u_C3R, "nppiErode3x3Border_16u_C3R");

  --*
  -- * Four-channel 16-bit unsigned integer 3x3 erosion with border control.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Source image width and height in pixels relative to pSrc.
  -- * \param oSrcOffset Source image starting point relative to pSrc. 
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eBorderType The border type operation to be applied at source image border boundaries.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiErode3x3Border_16u_C4R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      oSrcSize : nppdefs_h.NppiSize;
      oSrcOffset : nppdefs_h.NppiPoint;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eBorderType : nppdefs_h.NppiBorderType) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:1794
   pragma Import (C, nppiErode3x3Border_16u_C4R, "nppiErode3x3Border_16u_C4R");

  --*
  -- * Four-channel 16-bit unsigned integer 3x3 erosion with border control, ignoring alpha-channel.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Source image width and height in pixels relative to pSrc.
  -- * \param oSrcOffset Source image starting point relative to pSrc. 
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eBorderType The border type operation to be applied at source image border boundaries.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiErode3x3Border_16u_AC4R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      oSrcSize : nppdefs_h.NppiSize;
      oSrcOffset : nppdefs_h.NppiPoint;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eBorderType : nppdefs_h.NppiBorderType) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:1811
   pragma Import (C, nppiErode3x3Border_16u_AC4R, "nppiErode3x3Border_16u_AC4R");

  --*
  -- * Single-channel 32-bit floating-point 3x3 erosion with border control.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Source image width and height in pixels relative to pSrc.
  -- * \param oSrcOffset Source image starting point relative to pSrc. 
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eBorderType The border type operation to be applied at source image border boundaries.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiErode3x3Border_32f_C1R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : nppdefs_h.Npp32s;
      oSrcSize : nppdefs_h.NppiSize;
      oSrcOffset : nppdefs_h.NppiPoint;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : nppdefs_h.Npp32s;
      oSizeROI : nppdefs_h.NppiSize;
      eBorderType : nppdefs_h.NppiBorderType) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:1829
   pragma Import (C, nppiErode3x3Border_32f_C1R, "nppiErode3x3Border_32f_C1R");

  --*
  -- * Three-channel 32-bit floating-point 3x3 erosion with border control.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Source image width and height in pixels relative to pSrc.
  -- * \param oSrcOffset Source image starting point relative to pSrc. 
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eBorderType The border type operation to be applied at source image border boundaries.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiErode3x3Border_32f_C3R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : nppdefs_h.Npp32s;
      oSrcSize : nppdefs_h.NppiSize;
      oSrcOffset : nppdefs_h.NppiPoint;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : nppdefs_h.Npp32s;
      oSizeROI : nppdefs_h.NppiSize;
      eBorderType : nppdefs_h.NppiBorderType) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:1846
   pragma Import (C, nppiErode3x3Border_32f_C3R, "nppiErode3x3Border_32f_C3R");

  --*
  -- * Four-channel 32-bit floating-point 3x3 erosion with border control.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Source image width and height in pixels relative to pSrc.
  -- * \param oSrcOffset Source image starting point relative to pSrc. 
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eBorderType The border type operation to be applied at source image border boundaries.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiErode3x3Border_32f_C4R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      oSrcSize : nppdefs_h.NppiSize;
      oSrcOffset : nppdefs_h.NppiPoint;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eBorderType : nppdefs_h.NppiBorderType) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:1863
   pragma Import (C, nppiErode3x3Border_32f_C4R, "nppiErode3x3Border_32f_C4R");

  --*
  -- * Four-channel 32-bit floating-point 3x3 erosion with border control, ignoring alpha-channel.
  -- * 
  -- * \param pSrc  \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSize Source image width and height in pixels relative to pSrc.
  -- * \param oSrcOffset Source image starting point relative to pSrc. 
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eBorderType The border type operation to be applied at source image border boundaries.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiErode3x3Border_32f_AC4R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      oSrcSize : nppdefs_h.NppiSize;
      oSrcOffset : nppdefs_h.NppiPoint;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eBorderType : nppdefs_h.NppiBorderType) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_morphological_operations.h:1880
   pragma Import (C, nppiErode3x3Border_32f_AC4R, "nppiErode3x3Border_32f_AC4R");

  --* @} image_erode_3x3_border  
  --* @} image_morphological_operations  
  -- extern "C"  
end nppi_morphological_operations_h;

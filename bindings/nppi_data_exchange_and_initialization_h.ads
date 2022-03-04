pragma Ada_2005;
pragma Style_Checks (Off);

with Interfaces.C; use Interfaces.C;
with nppdefs_h;
with System;

package nppi_data_exchange_and_initialization_h is

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
  -- * \file nppi_data_exchange_and_initialization.h
  -- * NPP Image Processing Functionality.
  --  

  --* @defgroup image_data_exchange_and_initialization Data Exchange and Initialization
  -- *  @ingroup nppi
  -- *
  -- * Primitives for initializting, copying and converting image data.
  -- *
  -- * @{
  -- *
  -- * These functions can be found in either the nppi or nppidei libraries. Linking to only the sub-libraries that you use can significantly
  -- * save link time, application load time, and CUDA runtime startup time when using dynamic libraries.
  -- *
  --  

  --* 
  -- * @defgroup image_set Set
  -- *
  -- * Primitives for setting pixels to a specific value.
  -- *
  -- * @{
  -- *
  --  

  --* @name Set 
  -- *
  -- * Set all pixels within the ROI to a specific value.
  -- *
  -- * @{
  -- *
  --  

  --* 
  -- * 8-bit image set.
  -- * \param nValue The pixel value to be set.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSet_8s_C1R
     (nValue : nppdefs_h.Npp8s;
      pDst : access nppdefs_h.Npp8s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:102
   pragma Import (C, nppiSet_8s_C1R, "nppiSet_8s_C1R");

  --* 
  -- * 8-bit two-channel image set.
  -- * \param aValue The pixel value to be set.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSet_8s_C2R
     (aValue : access nppdefs_h.Npp8s;
      pDst : access nppdefs_h.Npp8s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:113
   pragma Import (C, nppiSet_8s_C2R, "nppiSet_8s_C2R");

  --* 
  -- * 8-bit three-channel image set.
  -- * \param aValue The pixel value to be set.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSet_8s_C3R
     (aValue : access nppdefs_h.Npp8s;
      pDst : access nppdefs_h.Npp8s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:124
   pragma Import (C, nppiSet_8s_C3R, "nppiSet_8s_C3R");

  --* 
  -- * 8-bit four-channel image set.
  -- * \param aValue The pixel value to be set.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSet_8s_C4R
     (aValue : access nppdefs_h.Npp8s;
      pDst : access nppdefs_h.Npp8s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:135
   pragma Import (C, nppiSet_8s_C4R, "nppiSet_8s_C4R");

  --* 
  -- * 8-bit four-channel image set ignoring alpha channel.
  -- * \param aValue The pixel value to be set.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSet_8s_AC4R
     (aValue : access nppdefs_h.Npp8s;
      pDst : access nppdefs_h.Npp8s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:146
   pragma Import (C, nppiSet_8s_AC4R, "nppiSet_8s_AC4R");

  --* 
  -- * 8-bit unsigned image set.
  -- * \param nValue The pixel value to be set.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSet_8u_C1R
     (nValue : nppdefs_h.Npp8u;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:157
   pragma Import (C, nppiSet_8u_C1R, "nppiSet_8u_C1R");

  --* 
  -- * 2 channel 8-bit unsigned image set.
  -- * \param aValue The pixel value to be set.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSet_8u_C2R
     (aValue : access nppdefs_h.Npp8u;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:168
   pragma Import (C, nppiSet_8u_C2R, "nppiSet_8u_C2R");

  --* 
  -- * 3 channel 8-bit unsigned image set.
  -- * \param aValue The pixel value to be set.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSet_8u_C3R
     (aValue : access nppdefs_h.Npp8u;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:179
   pragma Import (C, nppiSet_8u_C3R, "nppiSet_8u_C3R");

  --* 
  -- * 4 channel 8-bit unsigned image set.
  -- * \param aValue The pixel-value to be set.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSet_8u_C4R
     (aValue : access nppdefs_h.Npp8u;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:190
   pragma Import (C, nppiSet_8u_C4R, "nppiSet_8u_C4R");

  --* 
  -- * 4 channel 8-bit unsigned image set method, not affecting Alpha channel.
  -- * \param aValue The pixel-value to be set.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSet_8u_AC4R
     (aValue : access nppdefs_h.Npp8u;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:201
   pragma Import (C, nppiSet_8u_AC4R, "nppiSet_8u_AC4R");

  --* 
  -- * 16-bit unsigned image set.
  -- * \param nValue The pixel-value to be set.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSet_16u_C1R
     (nValue : nppdefs_h.Npp16u;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:212
   pragma Import (C, nppiSet_16u_C1R, "nppiSet_16u_C1R");

  --* 
  -- * 2 channel 16-bit unsigned image set.
  -- * \param aValue The pixel-value to be set.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSet_16u_C2R
     (aValue : access nppdefs_h.Npp16u;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:223
   pragma Import (C, nppiSet_16u_C2R, "nppiSet_16u_C2R");

  --* 
  -- * 3 channel 16-bit unsigned image set.
  -- * \param aValue The pixel-value to be set.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSet_16u_C3R
     (aValue : access nppdefs_h.Npp16u;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:234
   pragma Import (C, nppiSet_16u_C3R, "nppiSet_16u_C3R");

  --* 
  -- * 4 channel 16-bit unsigned image set.
  -- * \param aValue The pixel-value to be set.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSet_16u_C4R
     (aValue : access nppdefs_h.Npp16u;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:245
   pragma Import (C, nppiSet_16u_C4R, "nppiSet_16u_C4R");

  --* 
  -- * 4 channel 16-bit unsigned image set method, not affecting Alpha channel.
  -- * \param aValue The pixel-value to be set.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSet_16u_AC4R
     (aValue : access nppdefs_h.Npp16u;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:256
   pragma Import (C, nppiSet_16u_AC4R, "nppiSet_16u_AC4R");

  --* 
  -- * 16-bit image set.
  -- * \param nValue The pixel-value to be set.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSet_16s_C1R
     (nValue : nppdefs_h.Npp16s;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:267
   pragma Import (C, nppiSet_16s_C1R, "nppiSet_16s_C1R");

  --* 
  -- * 2 channel 16-bit image set.
  -- * \param aValue The pixel-value to be set.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSet_16s_C2R
     (aValue : access nppdefs_h.Npp16s;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:278
   pragma Import (C, nppiSet_16s_C2R, "nppiSet_16s_C2R");

  --* 
  -- * 3 channel 16-bit image set.
  -- * \param aValue The pixel-value to be set.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSet_16s_C3R
     (aValue : access nppdefs_h.Npp16s;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:289
   pragma Import (C, nppiSet_16s_C3R, "nppiSet_16s_C3R");

  --* 
  -- * 4 channel 16-bit image set.
  -- * \param aValue The pixel-value to be set.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSet_16s_C4R
     (aValue : access nppdefs_h.Npp16s;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:300
   pragma Import (C, nppiSet_16s_C4R, "nppiSet_16s_C4R");

  --* 
  -- * 4 channel 16-bit image set method, not affecting Alpha channel.
  -- * \param aValue The pixel-value to be set.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSet_16s_AC4R
     (aValue : access nppdefs_h.Npp16s;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:311
   pragma Import (C, nppiSet_16s_AC4R, "nppiSet_16s_AC4R");

  --* 
  -- * 16-bit complex integer image set.
  -- * \param oValue The pixel-value to be set.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSet_16sc_C1R
     (oValue : nppdefs_h.Npp16sc;
      pDst : access nppdefs_h.Npp16sc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:322
   pragma Import (C, nppiSet_16sc_C1R, "nppiSet_16sc_C1R");

  --* 
  -- * 16-bit complex integer two-channel image set.
  -- * \param aValue The pixel-value to be set.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSet_16sc_C2R
     (aValue : access constant nppdefs_h.Npp16sc;
      pDst : access nppdefs_h.Npp16sc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:333
   pragma Import (C, nppiSet_16sc_C2R, "nppiSet_16sc_C2R");

  --* 
  -- * 16-bit complex integer three-channel image set.
  -- * \param aValue The pixel-value to be set.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSet_16sc_C3R
     (aValue : access constant nppdefs_h.Npp16sc;
      pDst : access nppdefs_h.Npp16sc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:344
   pragma Import (C, nppiSet_16sc_C3R, "nppiSet_16sc_C3R");

  --* 
  -- * 16-bit complex integer four-channel image set.
  -- * \param aValue The pixel-value to be set.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSet_16sc_C4R
     (aValue : access constant nppdefs_h.Npp16sc;
      pDst : access nppdefs_h.Npp16sc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:355
   pragma Import (C, nppiSet_16sc_C4R, "nppiSet_16sc_C4R");

  --* 
  -- * 16-bit complex integer four-channel image set ignoring alpha.
  -- * \param aValue The pixel-value to be set.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSet_16sc_AC4R
     (aValue : access constant nppdefs_h.Npp16sc;
      pDst : access nppdefs_h.Npp16sc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:366
   pragma Import (C, nppiSet_16sc_AC4R, "nppiSet_16sc_AC4R");

  --* 
  -- * 32-bit image set.
  -- * \param nValue The pixel-value to be set.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSet_32s_C1R
     (nValue : nppdefs_h.Npp32s;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:377
   pragma Import (C, nppiSet_32s_C1R, "nppiSet_32s_C1R");

  --* 
  -- * 2 channel 32-bit image set.
  -- * \param aValue The pixel-value to be set.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSet_32s_C2R
     (aValue : access nppdefs_h.Npp32s;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:388
   pragma Import (C, nppiSet_32s_C2R, "nppiSet_32s_C2R");

  --* 
  -- * 3 channel 32-bit image set.
  -- * \param aValue The pixel-value to be set.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSet_32s_C3R
     (aValue : access nppdefs_h.Npp32s;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:399
   pragma Import (C, nppiSet_32s_C3R, "nppiSet_32s_C3R");

  --* 
  -- * 4 channel 32-bit image set.
  -- * \param aValue The pixel-value to be set.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSet_32s_C4R
     (aValue : access nppdefs_h.Npp32s;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:410
   pragma Import (C, nppiSet_32s_C4R, "nppiSet_32s_C4R");

  --* 
  -- * 4 channel 32-bit image set method, not affecting Alpha channel.
  -- * \param aValue The pixel-value to be set.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSet_32s_AC4R
     (aValue : access nppdefs_h.Npp32s;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:421
   pragma Import (C, nppiSet_32s_AC4R, "nppiSet_32s_AC4R");

  --* 
  -- * 32-bit unsigned image set.
  -- * \param nValue The pixel-value to be set.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSet_32u_C1R
     (nValue : nppdefs_h.Npp32u;
      pDst : access nppdefs_h.Npp32u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:432
   pragma Import (C, nppiSet_32u_C1R, "nppiSet_32u_C1R");

  --* 
  -- * 2 channel 32-bit unsigned image set.
  -- * \param aValue The pixel-value to be set.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSet_32u_C2R
     (aValue : access nppdefs_h.Npp32u;
      pDst : access nppdefs_h.Npp32u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:443
   pragma Import (C, nppiSet_32u_C2R, "nppiSet_32u_C2R");

  --* 
  -- * 3 channel 32-bit unsigned image set.
  -- * \param aValue The pixel-value to be set.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSet_32u_C3R
     (aValue : access nppdefs_h.Npp32u;
      pDst : access nppdefs_h.Npp32u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:454
   pragma Import (C, nppiSet_32u_C3R, "nppiSet_32u_C3R");

  --* 
  -- * 4 channel 32-bit unsigned image set.
  -- * \param aValue The pixel-value to be set.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSet_32u_C4R
     (aValue : access nppdefs_h.Npp32u;
      pDst : access nppdefs_h.Npp32u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:465
   pragma Import (C, nppiSet_32u_C4R, "nppiSet_32u_C4R");

  --* 
  -- * 4 channel 32-bit unsigned image set method, not affecting Alpha channel.
  -- * \param aValue The pixel-value to be set.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSet_32u_AC4R
     (aValue : access nppdefs_h.Npp32u;
      pDst : access nppdefs_h.Npp32u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:476
   pragma Import (C, nppiSet_32u_AC4R, "nppiSet_32u_AC4R");

  --* 
  -- * Single channel 32-bit complex integer image set.
  -- * \param oValue The pixel-value to be set.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSet_32sc_C1R
     (oValue : nppdefs_h.Npp32sc;
      pDst : access nppdefs_h.Npp32sc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:487
   pragma Import (C, nppiSet_32sc_C1R, "nppiSet_32sc_C1R");

  --* 
  -- * Two channel 32-bit complex integer image set.
  -- * \param aValue The pixel-value to be set.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSet_32sc_C2R
     (aValue : access constant nppdefs_h.Npp32sc;
      pDst : access nppdefs_h.Npp32sc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:498
   pragma Import (C, nppiSet_32sc_C2R, "nppiSet_32sc_C2R");

  --* 
  -- * Three channel 32-bit complex integer image set.
  -- * \param aValue The pixel-value to be set.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSet_32sc_C3R
     (aValue : access constant nppdefs_h.Npp32sc;
      pDst : access nppdefs_h.Npp32sc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:509
   pragma Import (C, nppiSet_32sc_C3R, "nppiSet_32sc_C3R");

  --* 
  -- * Four channel 32-bit complex integer image set.
  -- * \param aValue The pixel-value to be set.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSet_32sc_C4R
     (aValue : access constant nppdefs_h.Npp32sc;
      pDst : access nppdefs_h.Npp32sc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:520
   pragma Import (C, nppiSet_32sc_C4R, "nppiSet_32sc_C4R");

  --* 
  -- * 32-bit complex integer four-channel image set ignoring alpha.
  -- * \param aValue The pixel-value to be set.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSet_32sc_AC4R
     (aValue : access constant nppdefs_h.Npp32sc;
      pDst : access nppdefs_h.Npp32sc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:531
   pragma Import (C, nppiSet_32sc_AC4R, "nppiSet_32sc_AC4R");

  --* 
  -- * 32-bit floating point image set.
  -- * \param nValue The pixel-value to be set.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSet_32f_C1R
     (nValue : nppdefs_h.Npp32f;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:543
   pragma Import (C, nppiSet_32f_C1R, "nppiSet_32f_C1R");

  --* 
  -- * 2 channel 32-bit floating point image set.
  -- * \param aValue The pixel-value to be set.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSet_32f_C2R
     (aValue : access nppdefs_h.Npp32f;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:554
   pragma Import (C, nppiSet_32f_C2R, "nppiSet_32f_C2R");

  --* 
  -- * 3 channel 32-bit floating point image set.
  -- * \param aValue The pixel-value to be set.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSet_32f_C3R
     (aValue : access nppdefs_h.Npp32f;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:565
   pragma Import (C, nppiSet_32f_C3R, "nppiSet_32f_C3R");

  --* 
  -- * 4 channel 32-bit floating point image set.
  -- * \param aValue The pixel-value to be set.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSet_32f_C4R
     (aValue : access nppdefs_h.Npp32f;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:576
   pragma Import (C, nppiSet_32f_C4R, "nppiSet_32f_C4R");

  --* 
  -- * 4 channel 32-bit floating point image set method, not affecting Alpha channel.
  -- * \param aValue The pixel-value to be set.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSet_32f_AC4R
     (aValue : access nppdefs_h.Npp32f;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:587
   pragma Import (C, nppiSet_32f_AC4R, "nppiSet_32f_AC4R");

  --* 
  -- * Single channel 32-bit complex image set.
  -- * \param oValue The pixel-value to be set.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSet_32fc_C1R
     (oValue : nppdefs_h.Npp32fc;
      pDst : access nppdefs_h.Npp32fc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:599
   pragma Import (C, nppiSet_32fc_C1R, "nppiSet_32fc_C1R");

  --* 
  -- * Two channel 32-bit complex image set.
  -- * \param aValue The pixel-value to be set.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSet_32fc_C2R
     (aValue : access constant nppdefs_h.Npp32fc;
      pDst : access nppdefs_h.Npp32fc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:610
   pragma Import (C, nppiSet_32fc_C2R, "nppiSet_32fc_C2R");

  --* 
  -- * Three channel 32-bit complex image set.
  -- * \param aValue The pixel-value to be set.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSet_32fc_C3R
     (aValue : access constant nppdefs_h.Npp32fc;
      pDst : access nppdefs_h.Npp32fc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:621
   pragma Import (C, nppiSet_32fc_C3R, "nppiSet_32fc_C3R");

  --* 
  -- * Four channel 32-bit complex image set.
  -- * \param aValue The pixel-value to be set.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSet_32fc_C4R
     (aValue : access constant nppdefs_h.Npp32fc;
      pDst : access nppdefs_h.Npp32fc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:632
   pragma Import (C, nppiSet_32fc_C4R, "nppiSet_32fc_C4R");

  --* 
  -- * 32-bit complex four-channel image set ignoring alpha.
  -- * \param aValue The pixel-value to be set.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSet_32fc_AC4R
     (aValue : access constant nppdefs_h.Npp32fc;
      pDst : access nppdefs_h.Npp32fc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:643
   pragma Import (C, nppiSet_32fc_AC4R, "nppiSet_32fc_AC4R");

  --* @} Set  
  --* @name Masked Set
  -- * 
  -- * The masked set primitives have an additional "mask image" input. The mask  
  -- * controls which pixels within the ROI are set. For details see \ref masked_operation.
  -- *
  -- * @{
  -- *
  --  

  --* 
  -- * Masked 8-bit unsigned image set. 
  -- * \param nValue The pixel value to be set.
  -- * \param pDst Pointer \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pMask \ref mask_image_pointer.
  -- * \param nMaskStep \ref mask_image_line_step.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSet_8u_C1MR
     (nValue : nppdefs_h.Npp8u;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pMask : access nppdefs_h.Npp8u;
      nMaskStep : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:667
   pragma Import (C, nppiSet_8u_C1MR, "nppiSet_8u_C1MR");

  --* 
  -- * Masked 3 channel 8-bit unsigned image set.
  -- * \param aValue The pixel-value to be set.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pMask \ref mask_image_pointer.
  -- * \param nMaskStep \ref mask_image_line_step.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSet_8u_C3MR
     (aValue : access nppdefs_h.Npp8u;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pMask : access nppdefs_h.Npp8u;
      nMaskStep : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:680
   pragma Import (C, nppiSet_8u_C3MR, "nppiSet_8u_C3MR");

  --* 
  -- * Masked 4 channel 8-bit unsigned image set.
  -- * \param aValue The pixel-value to be set.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pMask \ref mask_image_pointer.
  -- * \param nMaskStep \ref mask_image_line_step.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSet_8u_C4MR
     (aValue : access nppdefs_h.Npp8u;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pMask : access nppdefs_h.Npp8u;
      nMaskStep : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:694
   pragma Import (C, nppiSet_8u_C4MR, "nppiSet_8u_C4MR");

  --* 
  -- * Masked 4 channel 8-bit unsigned image set method, not affecting Alpha channel.
  -- * \param aValue The pixel-value to be set.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pMask \ref mask_image_pointer.
  -- * \param nMaskStep \ref mask_image_line_step.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSet_8u_AC4MR
     (aValue : access nppdefs_h.Npp8u;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pMask : access nppdefs_h.Npp8u;
      nMaskStep : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:708
   pragma Import (C, nppiSet_8u_AC4MR, "nppiSet_8u_AC4MR");

  --* 
  -- * Masked 16-bit unsigned image set.
  -- * \param nValue The pixel-value to be set.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pMask \ref mask_image_pointer.
  -- * \param nMaskStep \ref mask_image_line_step.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSet_16u_C1MR
     (nValue : nppdefs_h.Npp16u;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pMask : access nppdefs_h.Npp8u;
      nMaskStep : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:723
   pragma Import (C, nppiSet_16u_C1MR, "nppiSet_16u_C1MR");

  --* 
  -- * Masked 3 channel 16-bit unsigned image set.
  -- * \param aValue The pixel-value to be set.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pMask \ref mask_image_pointer.
  -- * \param nMaskStep \ref mask_image_line_step.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSet_16u_C3MR
     (aValue : access nppdefs_h.Npp16u;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pMask : access nppdefs_h.Npp8u;
      nMaskStep : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:736
   pragma Import (C, nppiSet_16u_C3MR, "nppiSet_16u_C3MR");

  --* 
  -- * Masked 4 channel 16-bit unsigned image set.
  -- * \param aValue The pixel-value to be set.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pMask \ref mask_image_pointer.
  -- * \param nMaskStep \ref mask_image_line_step.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSet_16u_C4MR
     (aValue : access nppdefs_h.Npp16u;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pMask : access nppdefs_h.Npp8u;
      nMaskStep : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:751
   pragma Import (C, nppiSet_16u_C4MR, "nppiSet_16u_C4MR");

  --* 
  -- * Masked 4 channel 16-bit unsigned image set method, not affecting Alpha channel.
  -- * \param aValue The pixel-value to be set.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pMask \ref mask_image_pointer.
  -- * \param nMaskStep \ref mask_image_line_step.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSet_16u_AC4MR
     (aValue : access nppdefs_h.Npp16u;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pMask : access nppdefs_h.Npp8u;
      nMaskStep : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:766
   pragma Import (C, nppiSet_16u_AC4MR, "nppiSet_16u_AC4MR");

  --* 
  -- * Masked 16-bit image set.
  -- * \param nValue The pixel-value to be set.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pMask \ref mask_image_pointer.
  -- * \param nMaskStep \ref mask_image_line_step.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSet_16s_C1MR
     (nValue : nppdefs_h.Npp16s;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pMask : access nppdefs_h.Npp8u;
      nMaskStep : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:781
   pragma Import (C, nppiSet_16s_C1MR, "nppiSet_16s_C1MR");

  --* 
  -- * Masked 3 channel 16-bit image set.
  -- * \param aValue The pixel-value to be set.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pMask \ref mask_image_pointer.
  -- * \param nMaskStep \ref mask_image_line_step.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSet_16s_C3MR
     (aValue : access nppdefs_h.Npp16s;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pMask : access nppdefs_h.Npp8u;
      nMaskStep : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:794
   pragma Import (C, nppiSet_16s_C3MR, "nppiSet_16s_C3MR");

  --* 
  -- * Masked 4 channel 16-bit image set.
  -- * \param aValue The pixel-value to be set.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pMask \ref mask_image_pointer.
  -- * \param nMaskStep \ref mask_image_line_step.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSet_16s_C4MR
     (aValue : access nppdefs_h.Npp16s;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pMask : access nppdefs_h.Npp8u;
      nMaskStep : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:809
   pragma Import (C, nppiSet_16s_C4MR, "nppiSet_16s_C4MR");

  --* 
  -- * Masked 4 channel 16-bit image set method, not affecting Alpha channel.
  -- * \param aValue The pixel-value to be set.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pMask \ref mask_image_pointer.
  -- * \param nMaskStep \ref mask_image_line_step.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSet_16s_AC4MR
     (aValue : access nppdefs_h.Npp16s;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pMask : access nppdefs_h.Npp8u;
      nMaskStep : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:824
   pragma Import (C, nppiSet_16s_AC4MR, "nppiSet_16s_AC4MR");

  --* 
  -- * Masked 32-bit image set.
  -- * \param nValue The pixel-value to be set.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pMask \ref mask_image_pointer.
  -- * \param nMaskStep \ref mask_image_line_step.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSet_32s_C1MR
     (nValue : nppdefs_h.Npp32s;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pMask : access nppdefs_h.Npp8u;
      nMaskStep : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:839
   pragma Import (C, nppiSet_32s_C1MR, "nppiSet_32s_C1MR");

  --* 
  -- * Masked 3 channel 32-bit image set.
  -- * \param aValue The pixel-value to be set.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pMask \ref mask_image_pointer.
  -- * \param nMaskStep \ref mask_image_line_step.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSet_32s_C3MR
     (aValue : access nppdefs_h.Npp32s;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pMask : access nppdefs_h.Npp8u;
      nMaskStep : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:852
   pragma Import (C, nppiSet_32s_C3MR, "nppiSet_32s_C3MR");

  --* 
  -- * Masked 4 channel 32-bit image set.
  -- * \param aValue The pixel-value to be set.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pMask \ref mask_image_pointer.
  -- * \param nMaskStep \ref mask_image_line_step.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSet_32s_C4MR
     (aValue : access nppdefs_h.Npp32s;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pMask : access nppdefs_h.Npp8u;
      nMaskStep : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:867
   pragma Import (C, nppiSet_32s_C4MR, "nppiSet_32s_C4MR");

  --* 
  -- * Masked 4 channel 16-bit image set method, not affecting Alpha channel.
  -- * \param aValue The pixel-value to be set.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pMask \ref mask_image_pointer.
  -- * \param nMaskStep \ref mask_image_line_step.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSet_32s_AC4MR
     (aValue : access nppdefs_h.Npp32s;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pMask : access nppdefs_h.Npp8u;
      nMaskStep : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:882
   pragma Import (C, nppiSet_32s_AC4MR, "nppiSet_32s_AC4MR");

  --* 
  -- * Masked 32-bit floating point image set.
  -- * \param nValue The pixel-value to be set.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pMask \ref mask_image_pointer.
  -- * \param nMaskStep \ref mask_image_line_step.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSet_32f_C1MR
     (nValue : nppdefs_h.Npp32f;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pMask : access nppdefs_h.Npp8u;
      nMaskStep : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:897
   pragma Import (C, nppiSet_32f_C1MR, "nppiSet_32f_C1MR");

  --* 
  -- * Masked 3 channel 32-bit floating point image set.
  -- * \param aValue The pixel-value to be set.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pMask \ref mask_image_pointer.
  -- * \param nMaskStep \ref mask_image_line_step.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSet_32f_C3MR
     (aValue : access nppdefs_h.Npp32f;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pMask : access nppdefs_h.Npp8u;
      nMaskStep : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:910
   pragma Import (C, nppiSet_32f_C3MR, "nppiSet_32f_C3MR");

  --* 
  -- * Masked 4 channel 32-bit floating point image set.
  -- * \param aValue The pixel-value to be set.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pMask \ref mask_image_pointer.
  -- * \param nMaskStep \ref mask_image_line_step.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSet_32f_C4MR
     (aValue : access nppdefs_h.Npp32f;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pMask : access nppdefs_h.Npp8u;
      nMaskStep : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:925
   pragma Import (C, nppiSet_32f_C4MR, "nppiSet_32f_C4MR");

  --* 
  -- * Masked 4 channel 32-bit floating point image set method, not affecting Alpha channel.
  -- * \param aValue The pixel-value to be set.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pMask \ref mask_image_pointer.
  -- * \param nMaskStep \ref mask_image_line_step.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSet_32f_AC4MR
     (aValue : access nppdefs_h.Npp32f;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pMask : access nppdefs_h.Npp8u;
      nMaskStep : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:940
   pragma Import (C, nppiSet_32f_AC4MR, "nppiSet_32f_AC4MR");

  --* @} Masked Set  
  --* @name Channel Set
  -- * 
  -- * The select-channel set primitives set a single color channel in multi-channel images
  -- * to a given value. The channel is selected by adjusting the pDst pointer to point to 
  -- * the desired color channel (see \ref channel_of_interest).
  -- *
  -- * @{
  -- *
  --  

  --* 
  -- * 3 channel 8-bit unsigned image set affecting only single channel.
  -- * \param nValue The pixel-value to be set.
  -- * \param pDst \ref select_destination_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSet_8u_C3CR
     (nValue : nppdefs_h.Npp8u;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:966
   pragma Import (C, nppiSet_8u_C3CR, "nppiSet_8u_C3CR");

  --* 
  -- * 4 channel 8-bit unsigned image set affecting only single channel.
  -- * \param nValue The pixel-value to be set.
  -- * \param pDst \ref select_destination_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSet_8u_C4CR
     (nValue : nppdefs_h.Npp8u;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:977
   pragma Import (C, nppiSet_8u_C4CR, "nppiSet_8u_C4CR");

  --* 
  -- * 3 channel 16-bit unsigned image set affecting only single channel.
  -- * \param nValue The pixel-value to be set.
  -- * \param pDst \ref select_destination_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSet_16u_C3CR
     (nValue : nppdefs_h.Npp16u;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:988
   pragma Import (C, nppiSet_16u_C3CR, "nppiSet_16u_C3CR");

  --* 
  -- * 4 channel 16-bit unsigned image set affecting only single channel.
  -- * \param nValue The pixel-value to be set.
  -- * \param pDst \ref select_destination_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSet_16u_C4CR
     (nValue : nppdefs_h.Npp16u;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:999
   pragma Import (C, nppiSet_16u_C4CR, "nppiSet_16u_C4CR");

  --* 
  -- * 3 channel 16-bit signed image set affecting only single channel.
  -- * \param nValue The pixel-value to be set.
  -- * \param pDst \ref select_destination_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSet_16s_C3CR
     (nValue : nppdefs_h.Npp16s;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:1010
   pragma Import (C, nppiSet_16s_C3CR, "nppiSet_16s_C3CR");

  --* 
  -- * 4 channel 16-bit signed image set affecting only single channel.
  -- * \param nValue The pixel-value to be set.
  -- * \param pDst \ref select_destination_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSet_16s_C4CR
     (nValue : nppdefs_h.Npp16s;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:1021
   pragma Import (C, nppiSet_16s_C4CR, "nppiSet_16s_C4CR");

  --* 
  -- * 3 channel 32-bit unsigned image set affecting only single channel.
  -- * \param nValue The pixel-value to be set.
  -- * \param pDst \ref select_destination_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSet_32s_C3CR
     (nValue : nppdefs_h.Npp32s;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:1032
   pragma Import (C, nppiSet_32s_C3CR, "nppiSet_32s_C3CR");

  --* 
  -- * 4 channel 32-bit unsigned image set affecting only single channel.
  -- * \param nValue The pixel-value to be set.
  -- * \param pDst \ref select_destination_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSet_32s_C4CR
     (nValue : nppdefs_h.Npp32s;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:1043
   pragma Import (C, nppiSet_32s_C4CR, "nppiSet_32s_C4CR");

  --* 
  -- * 3 channel 32-bit floating point image set affecting only single channel.
  -- * \param nValue The pixel-value to be set.
  -- * \param pDst \ref select_destination_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSet_32f_C3CR
     (nValue : nppdefs_h.Npp32f;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:1054
   pragma Import (C, nppiSet_32f_C3CR, "nppiSet_32f_C3CR");

  --* 
  -- * 4 channel 32-bit floating point image set affecting only single channel.
  -- * \param nValue The pixel-value to be set.
  -- * \param pDst \ref select_destination_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSet_32f_C4CR
     (nValue : nppdefs_h.Npp32f;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:1065
   pragma Import (C, nppiSet_32f_C4CR, "nppiSet_32f_C4CR");

  --* @} Channel Set  
  --* @} image_set  
  --* 
  -- * @defgroup image_copy Copy
  -- *
  -- * @{
  -- *
  --  

  --* @name Copy
  -- *
  -- * Copy pixels from one image to another.
  -- * 
  -- * @{
  -- *
  --  

  --* 
  -- * 8-bit image copy.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_8s_C1R
     (pSrc : access nppdefs_h.Npp8s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:1098
   pragma Import (C, nppiCopy_8s_C1R, "nppiCopy_8s_C1R");

  --* 
  -- * Two-channel 8-bit image copy.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_8s_C2R
     (pSrc : access nppdefs_h.Npp8s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:1110
   pragma Import (C, nppiCopy_8s_C2R, "nppiCopy_8s_C2R");

  --* 
  -- * Three-channel 8-bit image copy.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_8s_C3R
     (pSrc : access nppdefs_h.Npp8s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:1122
   pragma Import (C, nppiCopy_8s_C3R, "nppiCopy_8s_C3R");

  --* 
  -- * Four-channel 8-bit image copy.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_8s_C4R
     (pSrc : access nppdefs_h.Npp8s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:1134
   pragma Import (C, nppiCopy_8s_C4R, "nppiCopy_8s_C4R");

  --* 
  -- * Four-channel 8-bit image copy, ignoring alpha channel.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_8s_AC4R
     (pSrc : access nppdefs_h.Npp8s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:1146
   pragma Import (C, nppiCopy_8s_AC4R, "nppiCopy_8s_AC4R");

  --* 
  -- * 8-bit unsigned image copy.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_8u_C1R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:1158
   pragma Import (C, nppiCopy_8u_C1R, "nppiCopy_8u_C1R");

  --* 
  -- * Three channel 8-bit unsigned image copy.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_8u_C3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:1170
   pragma Import (C, nppiCopy_8u_C3R, "nppiCopy_8u_C3R");

  --* 
  -- * 4 channel 8-bit unsigned image copy.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_8u_C4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:1182
   pragma Import (C, nppiCopy_8u_C4R, "nppiCopy_8u_C4R");

  --* 
  -- * 4 channel 8-bit unsigned image copy, not affecting Alpha channel.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_8u_AC4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:1194
   pragma Import (C, nppiCopy_8u_AC4R, "nppiCopy_8u_AC4R");

  --* 
  -- * 16-bit unsigned image copy.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_16u_C1R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:1206
   pragma Import (C, nppiCopy_16u_C1R, "nppiCopy_16u_C1R");

  --* 
  -- * Three channel 16-bit unsigned image copy.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_16u_C3R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:1218
   pragma Import (C, nppiCopy_16u_C3R, "nppiCopy_16u_C3R");

  --* 
  -- * 4 channel 16-bit unsigned image copy.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_16u_C4R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:1230
   pragma Import (C, nppiCopy_16u_C4R, "nppiCopy_16u_C4R");

  --* 
  -- * 4 channel 16-bit unsigned image copy, not affecting Alpha channel.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_16u_AC4R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:1242
   pragma Import (C, nppiCopy_16u_AC4R, "nppiCopy_16u_AC4R");

  --* 
  -- * 16-bit image copy.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_16s_C1R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:1254
   pragma Import (C, nppiCopy_16s_C1R, "nppiCopy_16s_C1R");

  --* 
  -- * Three channel 16-bit image copy.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_16s_C3R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:1266
   pragma Import (C, nppiCopy_16s_C3R, "nppiCopy_16s_C3R");

  --* 
  -- * 4 channel 16-bit image copy.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_16s_C4R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:1278
   pragma Import (C, nppiCopy_16s_C4R, "nppiCopy_16s_C4R");

  --* 
  -- * 4 channel 16-bit image copy, not affecting Alpha.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_16s_AC4R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:1290
   pragma Import (C, nppiCopy_16s_AC4R, "nppiCopy_16s_AC4R");

  --* 
  -- * 16-bit complex image copy.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_16sc_C1R
     (pSrc : access constant nppdefs_h.Npp16sc;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16sc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:1302
   pragma Import (C, nppiCopy_16sc_C1R, "nppiCopy_16sc_C1R");

  --* 
  -- * Two-channel 16-bit complex image copy.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_16sc_C2R
     (pSrc : access constant nppdefs_h.Npp16sc;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16sc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:1314
   pragma Import (C, nppiCopy_16sc_C2R, "nppiCopy_16sc_C2R");

  --* 
  -- * Three-channel 16-bit complex image copy.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_16sc_C3R
     (pSrc : access constant nppdefs_h.Npp16sc;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16sc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:1326
   pragma Import (C, nppiCopy_16sc_C3R, "nppiCopy_16sc_C3R");

  --* 
  -- * Four-channel 16-bit complex image copy.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_16sc_C4R
     (pSrc : access constant nppdefs_h.Npp16sc;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16sc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:1338
   pragma Import (C, nppiCopy_16sc_C4R, "nppiCopy_16sc_C4R");

  --* 
  -- * Four-channel 16-bit complex image copy, ignoring alpha.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_16sc_AC4R
     (pSrc : access constant nppdefs_h.Npp16sc;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16sc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:1350
   pragma Import (C, nppiCopy_16sc_AC4R, "nppiCopy_16sc_AC4R");

  --* 
  -- * 32-bit image copy.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_32s_C1R
     (pSrc : access nppdefs_h.Npp32s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:1363
   pragma Import (C, nppiCopy_32s_C1R, "nppiCopy_32s_C1R");

  --* 
  -- * Three channel 32-bit image copy.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_32s_C3R
     (pSrc : access nppdefs_h.Npp32s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:1375
   pragma Import (C, nppiCopy_32s_C3R, "nppiCopy_32s_C3R");

  --* 
  -- * 4 channel 32-bit image copy.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_32s_C4R
     (pSrc : access nppdefs_h.Npp32s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:1387
   pragma Import (C, nppiCopy_32s_C4R, "nppiCopy_32s_C4R");

  --* 
  -- * 4 channel 32-bit image copy, not affecting Alpha.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_32s_AC4R
     (pSrc : access nppdefs_h.Npp32s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:1399
   pragma Import (C, nppiCopy_32s_AC4R, "nppiCopy_32s_AC4R");

  --* 
  -- * 32-bit complex image copy.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_32sc_C1R
     (pSrc : access constant nppdefs_h.Npp32sc;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32sc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:1411
   pragma Import (C, nppiCopy_32sc_C1R, "nppiCopy_32sc_C1R");

  --* 
  -- * Two-channel 32-bit complex image copy.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_32sc_C2R
     (pSrc : access constant nppdefs_h.Npp32sc;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32sc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:1423
   pragma Import (C, nppiCopy_32sc_C2R, "nppiCopy_32sc_C2R");

  --* 
  -- * Three-channel 32-bit complex image copy.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_32sc_C3R
     (pSrc : access constant nppdefs_h.Npp32sc;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32sc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:1435
   pragma Import (C, nppiCopy_32sc_C3R, "nppiCopy_32sc_C3R");

  --* 
  -- * Four-channel 32-bit complex image copy.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_32sc_C4R
     (pSrc : access constant nppdefs_h.Npp32sc;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32sc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:1447
   pragma Import (C, nppiCopy_32sc_C4R, "nppiCopy_32sc_C4R");

  --* 
  -- * Four-channel 32-bit complex image copy, ignoring alpha.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_32sc_AC4R
     (pSrc : access constant nppdefs_h.Npp32sc;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32sc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:1459
   pragma Import (C, nppiCopy_32sc_AC4R, "nppiCopy_32sc_AC4R");

  --* 
  -- * 32-bit floating point image copy.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_32f_C1R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:1472
   pragma Import (C, nppiCopy_32f_C1R, "nppiCopy_32f_C1R");

  --* 
  -- * Three channel 32-bit floating point image copy.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_32f_C3R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:1484
   pragma Import (C, nppiCopy_32f_C3R, "nppiCopy_32f_C3R");

  --* 
  -- * 4 channel 32-bit floating point image copy.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_32f_C4R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:1496
   pragma Import (C, nppiCopy_32f_C4R, "nppiCopy_32f_C4R");

  --* 
  -- * 4 channel 32-bit floating point image copy, not affecting Alpha.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_32f_AC4R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:1508
   pragma Import (C, nppiCopy_32f_AC4R, "nppiCopy_32f_AC4R");

  --* 
  -- * 32-bit floating-point complex image copy.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_32fc_C1R
     (pSrc : access constant nppdefs_h.Npp32fc;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32fc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:1521
   pragma Import (C, nppiCopy_32fc_C1R, "nppiCopy_32fc_C1R");

  --* 
  -- * Two-channel 32-bit floating-point complex image copy.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_32fc_C2R
     (pSrc : access constant nppdefs_h.Npp32fc;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32fc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:1533
   pragma Import (C, nppiCopy_32fc_C2R, "nppiCopy_32fc_C2R");

  --* 
  -- * Three-channel 32-bit floating-point complex image copy.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_32fc_C3R
     (pSrc : access constant nppdefs_h.Npp32fc;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32fc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:1545
   pragma Import (C, nppiCopy_32fc_C3R, "nppiCopy_32fc_C3R");

  --* 
  -- * Four-channel 32-bit floating-point complex image copy.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_32fc_C4R
     (pSrc : access constant nppdefs_h.Npp32fc;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32fc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:1557
   pragma Import (C, nppiCopy_32fc_C4R, "nppiCopy_32fc_C4R");

  --* 
  -- * Four-channel 32-bit floating-point complex image copy, ignoring alpha.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_32fc_AC4R
     (pSrc : access constant nppdefs_h.Npp32fc;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32fc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:1569
   pragma Import (C, nppiCopy_32fc_AC4R, "nppiCopy_32fc_AC4R");

  --* @} Copy  
  --* @name Masked Copy
  -- * 
  -- * The masked copy primitives have an additional "mask image" input. The mask  
  -- * controls which pixels within the ROI are copied. For details see \ref masked_operation.
  -- *
  -- * @{
  -- *
  --  

  --* 
  -- * \ref masked_operation 8-bit unsigned image copy.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param pMask \ref mask_image_pointer.
  -- * \param nMaskStep \ref mask_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_8u_C1MR
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pMask : access nppdefs_h.Npp8u;
      nMaskStep : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:1594
   pragma Import (C, nppiCopy_8u_C1MR, "nppiCopy_8u_C1MR");

  --* 
  -- * \ref masked_operation three channel 8-bit unsigned image copy.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param pMask \ref mask_image_pointer.
  -- * \param nMaskStep \ref mask_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_8u_C3MR
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pMask : access nppdefs_h.Npp8u;
      nMaskStep : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:1609
   pragma Import (C, nppiCopy_8u_C3MR, "nppiCopy_8u_C3MR");

  --* 
  -- * \ref masked_operation four channel 8-bit unsigned image copy.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param pMask \ref mask_image_pointer.
  -- * \param nMaskStep \ref mask_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_8u_C4MR
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pMask : access nppdefs_h.Npp8u;
      nMaskStep : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:1624
   pragma Import (C, nppiCopy_8u_C4MR, "nppiCopy_8u_C4MR");

  --* 
  -- * \ref masked_operation four channel 8-bit unsigned image copy, ignoring alpha.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param pMask \ref mask_image_pointer.
  -- * \param nMaskStep \ref mask_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_8u_AC4MR
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pMask : access nppdefs_h.Npp8u;
      nMaskStep : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:1639
   pragma Import (C, nppiCopy_8u_AC4MR, "nppiCopy_8u_AC4MR");

  --* 
  -- * \ref masked_operation 16-bit unsigned image copy.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param pMask \ref mask_image_pointer.
  -- * \param nMaskStep \ref mask_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_16u_C1MR
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pMask : access nppdefs_h.Npp8u;
      nMaskStep : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:1654
   pragma Import (C, nppiCopy_16u_C1MR, "nppiCopy_16u_C1MR");

  --* 
  -- * \ref masked_operation three channel 16-bit unsigned image copy.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param pMask \ref mask_image_pointer.
  -- * \param nMaskStep \ref mask_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_16u_C3MR
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pMask : access nppdefs_h.Npp8u;
      nMaskStep : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:1669
   pragma Import (C, nppiCopy_16u_C3MR, "nppiCopy_16u_C3MR");

  --* 
  -- * \ref masked_operation four channel 16-bit unsigned image copy.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param pMask \ref mask_image_pointer.
  -- * \param nMaskStep \ref mask_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_16u_C4MR
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pMask : access nppdefs_h.Npp8u;
      nMaskStep : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:1684
   pragma Import (C, nppiCopy_16u_C4MR, "nppiCopy_16u_C4MR");

  --* 
  -- * \ref masked_operation four channel 16-bit unsigned image copy, ignoring alpha.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param pMask \ref mask_image_pointer.
  -- * \param nMaskStep \ref mask_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_16u_AC4MR
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pMask : access nppdefs_h.Npp8u;
      nMaskStep : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:1699
   pragma Import (C, nppiCopy_16u_AC4MR, "nppiCopy_16u_AC4MR");

  --* 
  -- * \ref masked_operation 16-bit signed image copy.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param pMask \ref mask_image_pointer.
  -- * \param nMaskStep \ref mask_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_16s_C1MR
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pMask : access nppdefs_h.Npp8u;
      nMaskStep : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:1714
   pragma Import (C, nppiCopy_16s_C1MR, "nppiCopy_16s_C1MR");

  --* 
  -- * \ref masked_operation three channel 16-bit signed image copy.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param pMask \ref mask_image_pointer.
  -- * \param nMaskStep \ref mask_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_16s_C3MR
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pMask : access nppdefs_h.Npp8u;
      nMaskStep : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:1729
   pragma Import (C, nppiCopy_16s_C3MR, "nppiCopy_16s_C3MR");

  --* 
  -- * \ref masked_operation four channel 16-bit signed image copy.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param pMask \ref mask_image_pointer.
  -- * \param nMaskStep \ref mask_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_16s_C4MR
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pMask : access nppdefs_h.Npp8u;
      nMaskStep : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:1744
   pragma Import (C, nppiCopy_16s_C4MR, "nppiCopy_16s_C4MR");

  --* 
  -- * \ref masked_operation four channel 16-bit signed image copy, ignoring alpha.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param pMask \ref mask_image_pointer.
  -- * \param nMaskStep \ref mask_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_16s_AC4MR
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pMask : access nppdefs_h.Npp8u;
      nMaskStep : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:1759
   pragma Import (C, nppiCopy_16s_AC4MR, "nppiCopy_16s_AC4MR");

  --* 
  -- * \ref masked_operation 32-bit signed image copy.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param pMask \ref mask_image_pointer.
  -- * \param nMaskStep \ref mask_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_32s_C1MR
     (pSrc : access nppdefs_h.Npp32s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pMask : access nppdefs_h.Npp8u;
      nMaskStep : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:1774
   pragma Import (C, nppiCopy_32s_C1MR, "nppiCopy_32s_C1MR");

  --* 
  -- * \ref masked_operation three channel 32-bit signed image copy.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param pMask \ref mask_image_pointer.
  -- * \param nMaskStep \ref mask_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_32s_C3MR
     (pSrc : access nppdefs_h.Npp32s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pMask : access nppdefs_h.Npp8u;
      nMaskStep : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:1789
   pragma Import (C, nppiCopy_32s_C3MR, "nppiCopy_32s_C3MR");

  --* 
  -- * \ref masked_operation four channel 32-bit signed image copy.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param pMask \ref mask_image_pointer.
  -- * \param nMaskStep \ref mask_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_32s_C4MR
     (pSrc : access nppdefs_h.Npp32s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pMask : access nppdefs_h.Npp8u;
      nMaskStep : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:1804
   pragma Import (C, nppiCopy_32s_C4MR, "nppiCopy_32s_C4MR");

  --* 
  -- * \ref masked_operation four channel 32-bit signed image copy, ignoring alpha.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param pMask \ref mask_image_pointer.
  -- * \param nMaskStep \ref mask_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_32s_AC4MR
     (pSrc : access nppdefs_h.Npp32s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pMask : access nppdefs_h.Npp8u;
      nMaskStep : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:1819
   pragma Import (C, nppiCopy_32s_AC4MR, "nppiCopy_32s_AC4MR");

  --* 
  -- * \ref masked_operation 32-bit float image copy.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param pMask \ref mask_image_pointer.
  -- * \param nMaskStep \ref mask_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_32f_C1MR
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pMask : access nppdefs_h.Npp8u;
      nMaskStep : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:1834
   pragma Import (C, nppiCopy_32f_C1MR, "nppiCopy_32f_C1MR");

  --* 
  -- * \ref masked_operation three channel 32-bit float image copy.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param pMask \ref mask_image_pointer.
  -- * \param nMaskStep \ref mask_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_32f_C3MR
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pMask : access nppdefs_h.Npp8u;
      nMaskStep : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:1849
   pragma Import (C, nppiCopy_32f_C3MR, "nppiCopy_32f_C3MR");

  --* 
  -- * \ref masked_operation four channel 32-bit float image copy.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param pMask \ref mask_image_pointer.
  -- * \param nMaskStep \ref mask_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_32f_C4MR
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pMask : access nppdefs_h.Npp8u;
      nMaskStep : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:1864
   pragma Import (C, nppiCopy_32f_C4MR, "nppiCopy_32f_C4MR");

  --* 
  -- * \ref masked_operation four channel 32-bit float image copy, ignoring alpha.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param pMask \ref mask_image_pointer.
  -- * \param nMaskStep \ref mask_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_32f_AC4MR
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      pMask : access nppdefs_h.Npp8u;
      nMaskStep : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:1879
   pragma Import (C, nppiCopy_32f_AC4MR, "nppiCopy_32f_AC4MR");

  --* @} Masked Copy  
  --* @name Channel Copy
  -- * 
  -- * The channel copy primitives copy a single color channel from a multi-channel source image
  -- * to any other color channel in a multi-channel destination image. The channel is selected 
  -- * by adjusting the respective image  pointers to point to the desired color channel 
  -- * (see \ref channel_of_interest).
  -- *
  -- * @{
  -- *
  --  

  --* 
  -- * Select-channel 8-bit unsigned image copy for three-channel images.
  -- * \param pSrc \ref select_source_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref select_destination_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_8u_C3CR
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:1906
   pragma Import (C, nppiCopy_8u_C3CR, "nppiCopy_8u_C3CR");

  --* 
  -- * Select-channel 8-bit unsigned image copy for four-channel images.
  -- * \param pSrc \ref select_source_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref select_destination_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_8u_C4CR
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:1918
   pragma Import (C, nppiCopy_8u_C4CR, "nppiCopy_8u_C4CR");

  --* 
  -- * Select-channel 16-bit signed image copy for three-channel images.
  -- * \param pSrc \ref select_source_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref select_destination_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_16s_C3CR
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:1930
   pragma Import (C, nppiCopy_16s_C3CR, "nppiCopy_16s_C3CR");

  --* 
  -- * Select-channel 16-bit signed image copy for four-channel images.
  -- * \param pSrc \ref select_source_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref select_destination_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_16s_C4CR
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:1942
   pragma Import (C, nppiCopy_16s_C4CR, "nppiCopy_16s_C4CR");

  --* 
  -- * Select-channel 16-bit unsigned image copy for three-channel images.
  -- * \param pSrc \ref select_source_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref select_destination_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_16u_C3CR
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:1954
   pragma Import (C, nppiCopy_16u_C3CR, "nppiCopy_16u_C3CR");

  --* 
  -- * Select-channel 16-bit unsigned image copy for four-channel images.
  -- * \param pSrc \ref select_source_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref select_destination_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_16u_C4CR
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:1966
   pragma Import (C, nppiCopy_16u_C4CR, "nppiCopy_16u_C4CR");

  --* 
  -- * Select-channel 32-bit signed image copy for three-channel images.
  -- * \param pSrc \ref select_source_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref select_destination_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_32s_C3CR
     (pSrc : access nppdefs_h.Npp32s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:1978
   pragma Import (C, nppiCopy_32s_C3CR, "nppiCopy_32s_C3CR");

  --* 
  -- * Select-channel 32-bit signed image copy for four-channel images.
  -- * \param pSrc \ref select_source_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref select_destination_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_32s_C4CR
     (pSrc : access nppdefs_h.Npp32s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:1990
   pragma Import (C, nppiCopy_32s_C4CR, "nppiCopy_32s_C4CR");

  --* 
  -- * Select-channel 32-bit float image copy for three-channel images.
  -- * \param pSrc \ref select_source_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref select_destination_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_32f_C3CR
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:2002
   pragma Import (C, nppiCopy_32f_C3CR, "nppiCopy_32f_C3CR");

  --* 
  -- * Select-channel 32-bit float image copy for four-channel images.
  -- * \param pSrc \ref select_source_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref select_destination_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_32f_C4CR
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:2014
   pragma Import (C, nppiCopy_32f_C4CR, "nppiCopy_32f_C4CR");

  --* @} Channel Copy  
  --* @name Extract Channel Copy
  -- * 
  -- * The channel extract primitives copy a single color channel from a multi-channel source image
  -- * to singl-channel destination image. The channel is selected by adjusting the source image pointer
  -- * to point to the desired color channel (see \ref channel_of_interest).
  -- *
  -- * @{
  -- *
  --  

  --* 
  -- * Three-channel to single-channel 8-bit unsigned image copy.
  -- * \param pSrc \ref select_source_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_8u_C3C1R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:2040
   pragma Import (C, nppiCopy_8u_C3C1R, "nppiCopy_8u_C3C1R");

  --* 
  -- * Four-channel to single-channel 8-bit unsigned image copy.
  -- * \param pSrc \ref select_source_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_8u_C4C1R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:2052
   pragma Import (C, nppiCopy_8u_C4C1R, "nppiCopy_8u_C4C1R");

  --* 
  -- * Three-channel to single-channel 16-bit signed image copy.
  -- * \param pSrc \ref select_source_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_16s_C3C1R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:2064
   pragma Import (C, nppiCopy_16s_C3C1R, "nppiCopy_16s_C3C1R");

  --* 
  -- * Four-channel to single-channel 16-bit signed image copy.
  -- * \param pSrc \ref select_source_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_16s_C4C1R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:2076
   pragma Import (C, nppiCopy_16s_C4C1R, "nppiCopy_16s_C4C1R");

  --* 
  -- * Three-channel to single-channel 16-bit unsigned image copy.
  -- * \param pSrc \ref select_source_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_16u_C3C1R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:2088
   pragma Import (C, nppiCopy_16u_C3C1R, "nppiCopy_16u_C3C1R");

  --* 
  -- * Four-channel to single-channel 16-bit unsigned image copy.
  -- * \param pSrc \ref select_source_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_16u_C4C1R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:2100
   pragma Import (C, nppiCopy_16u_C4C1R, "nppiCopy_16u_C4C1R");

  --* 
  -- * Three-channel to single-channel 32-bit signed image copy.
  -- * \param pSrc \ref select_source_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_32s_C3C1R
     (pSrc : access nppdefs_h.Npp32s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:2112
   pragma Import (C, nppiCopy_32s_C3C1R, "nppiCopy_32s_C3C1R");

  --* 
  -- * Four-channel to single-channel 32-bit signed image copy.
  -- * \param pSrc \ref select_source_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_32s_C4C1R
     (pSrc : access nppdefs_h.Npp32s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:2124
   pragma Import (C, nppiCopy_32s_C4C1R, "nppiCopy_32s_C4C1R");

  --* 
  -- * Three-channel to single-channel 32-bit float image copy.
  -- * \param pSrc \ref select_source_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_32f_C3C1R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:2136
   pragma Import (C, nppiCopy_32f_C3C1R, "nppiCopy_32f_C3C1R");

  --* 
  -- * Four-channel to single-channel 32-bit float image copy.
  -- * \param pSrc \ref select_source_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_32f_C4C1R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:2148
   pragma Import (C, nppiCopy_32f_C4C1R, "nppiCopy_32f_C4C1R");

  --* @} Extract Channel Copy  
  --* @name Insert Channel Copy
  -- * 
  -- * The channel insert primitives copy a single-channel source image into one of the color channels
  -- * in a multi-channel destination image. The channel is selected by adjusting the destination image pointer
  -- * to point to the desired color channel (see \ref channel_of_interest).
  -- *
  -- * @{
  -- *
  --  

  --* 
  -- * Single-channel to three-channel 8-bit unsigned image copy.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref select_destination_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_8u_C1C3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:2172
   pragma Import (C, nppiCopy_8u_C1C3R, "nppiCopy_8u_C1C3R");

  --* 
  -- * Single-channel to four-channel 8-bit unsigned image copy.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref select_destination_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_8u_C1C4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:2184
   pragma Import (C, nppiCopy_8u_C1C4R, "nppiCopy_8u_C1C4R");

  --* 
  -- * Single-channel to three-channel 16-bit signed image copy.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref select_destination_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_16s_C1C3R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:2196
   pragma Import (C, nppiCopy_16s_C1C3R, "nppiCopy_16s_C1C3R");

  --* 
  -- * Single-channel to four-channel 16-bit signed image copy.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref select_destination_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_16s_C1C4R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:2208
   pragma Import (C, nppiCopy_16s_C1C4R, "nppiCopy_16s_C1C4R");

  --* 
  -- * Single-channel to three-channel 16-bit unsigned image copy.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref select_destination_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_16u_C1C3R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:2220
   pragma Import (C, nppiCopy_16u_C1C3R, "nppiCopy_16u_C1C3R");

  --* 
  -- * Single-channel to four-channel 16-bit unsigned image copy.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref select_destination_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_16u_C1C4R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:2232
   pragma Import (C, nppiCopy_16u_C1C4R, "nppiCopy_16u_C1C4R");

  --* 
  -- * Single-channel to three-channel 32-bit signed image copy.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref select_destination_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_32s_C1C3R
     (pSrc : access nppdefs_h.Npp32s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:2244
   pragma Import (C, nppiCopy_32s_C1C3R, "nppiCopy_32s_C1C3R");

  --* 
  -- * Single-channel to four-channel 32-bit signed image copy.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref select_destination_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_32s_C1C4R
     (pSrc : access nppdefs_h.Npp32s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:2256
   pragma Import (C, nppiCopy_32s_C1C4R, "nppiCopy_32s_C1C4R");

  --* 
  -- * Single-channel to three-channel 32-bit float image copy.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref select_destination_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_32f_C1C3R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:2268
   pragma Import (C, nppiCopy_32f_C1C3R, "nppiCopy_32f_C1C3R");

  --* 
  -- * Single-channel to four-channel 32-bit float image copy.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref select_destination_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_32f_C1C4R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:2280
   pragma Import (C, nppiCopy_32f_C1C4R, "nppiCopy_32f_C1C4R");

  --* @} Insert Channel Copy  
  --* @name Packed-to-Planar Copy
  -- * 
  -- * Split a packed multi-channel image into a planar image.
  -- *
  -- * E.g. copy the three channels of an RGB image into three separate single-channel
  -- * images.
  -- *
  -- * @{
  -- *
  --  

  --* 
  -- * Three-channel 8-bit unsigned packed to planar image copy.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param aDst \ref destination_planar_image_pointer_array.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_8u_C3P3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      aDst : System.Address;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:2306
   pragma Import (C, nppiCopy_8u_C3P3R, "nppiCopy_8u_C3P3R");

  --* 
  -- * Four-channel 8-bit unsigned packed to planar image copy.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param aDst \ref destination_planar_image_pointer_array.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_8u_C4P4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      aDst : System.Address;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:2318
   pragma Import (C, nppiCopy_8u_C4P4R, "nppiCopy_8u_C4P4R");

  --* 
  -- * Three-channel 16-bit signed packed to planar image copy.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param aDst \ref destination_planar_image_pointer_array.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_16s_C3P3R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      aDst : System.Address;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:2330
   pragma Import (C, nppiCopy_16s_C3P3R, "nppiCopy_16s_C3P3R");

  --* 
  -- * Four-channel 16-bit signed packed to planar image copy.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param aDst \ref destination_planar_image_pointer_array.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_16s_C4P4R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      aDst : System.Address;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:2342
   pragma Import (C, nppiCopy_16s_C4P4R, "nppiCopy_16s_C4P4R");

  --* 
  -- * Three-channel 16-bit unsigned packed to planar image copy.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param aDst \ref destination_planar_image_pointer_array.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_16u_C3P3R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      aDst : System.Address;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:2354
   pragma Import (C, nppiCopy_16u_C3P3R, "nppiCopy_16u_C3P3R");

  --* 
  -- * Four-channel 16-bit unsigned packed to planar image copy.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param aDst \ref destination_planar_image_pointer_array.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_16u_C4P4R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      aDst : System.Address;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:2366
   pragma Import (C, nppiCopy_16u_C4P4R, "nppiCopy_16u_C4P4R");

  --* 
  -- * Three-channel 32-bit signed packed to planar image copy.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param aDst \ref destination_planar_image_pointer_array.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_32s_C3P3R
     (pSrc : access nppdefs_h.Npp32s;
      nSrcStep : int;
      aDst : System.Address;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:2378
   pragma Import (C, nppiCopy_32s_C3P3R, "nppiCopy_32s_C3P3R");

  --* 
  -- * Four-channel 32-bit signed packed to planar image copy.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param aDst \ref destination_planar_image_pointer_array.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_32s_C4P4R
     (pSrc : access nppdefs_h.Npp32s;
      nSrcStep : int;
      aDst : System.Address;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:2390
   pragma Import (C, nppiCopy_32s_C4P4R, "nppiCopy_32s_C4P4R");

  --* 
  -- * Three-channel 32-bit float packed to planar image copy.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param aDst \ref destination_planar_image_pointer_array.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_32f_C3P3R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      aDst : System.Address;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:2402
   pragma Import (C, nppiCopy_32f_C3P3R, "nppiCopy_32f_C3P3R");

  --* 
  -- * Four-channel 32-bit float packed to planar image copy.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param aDst \ref destination_planar_image_pointer_array.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_32f_C4P4R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      aDst : System.Address;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:2414
   pragma Import (C, nppiCopy_32f_C4P4R, "nppiCopy_32f_C4P4R");

  --* @} Packed-to-Planar Copy  
  --* @name Planar-to-Packed Copy
  -- * 
  -- * Combine multiple image planes into a packed multi-channel image.
  -- *
  -- * E.g. copy three single-channel images into a single 3-channel image.
  -- *
  -- * @{
  -- *
  --  

  --* 
  -- * Three-channel 8-bit unsigned planar to packed image copy.
  -- * \param aSrc Planar \ref source_image_pointer.
  -- * \param nSrcStep \ref source_planar_image_pointer_array.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_8u_P3C3R
     (aSrc : System.Address;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:2439
   pragma Import (C, nppiCopy_8u_P3C3R, "nppiCopy_8u_P3C3R");

  --* 
  -- * Four-channel 8-bit unsigned planar to packed image copy.
  -- * \param aSrc Planar \ref source_planar_image_pointer_array.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_8u_P4C4R
     (aSrc : System.Address;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:2451
   pragma Import (C, nppiCopy_8u_P4C4R, "nppiCopy_8u_P4C4R");

  --* 
  -- * Three-channel 16-bit unsigned planar to packed image copy.
  -- * \param aSrc Planar \ref source_planar_image_pointer_array.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_16u_P3C3R
     (aSrc : System.Address;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:2463
   pragma Import (C, nppiCopy_16u_P3C3R, "nppiCopy_16u_P3C3R");

  --* 
  -- * Four-channel 16-bit unsigned planar to packed image copy.
  -- * \param aSrc Planar \ref source_planar_image_pointer_array.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_16u_P4C4R
     (aSrc : System.Address;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:2475
   pragma Import (C, nppiCopy_16u_P4C4R, "nppiCopy_16u_P4C4R");

  --* 
  -- * Three-channel 16-bit signed planar to packed image copy.
  -- * \param aSrc Planar \ref source_planar_image_pointer_array.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_16s_P3C3R
     (aSrc : System.Address;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:2487
   pragma Import (C, nppiCopy_16s_P3C3R, "nppiCopy_16s_P3C3R");

  --* 
  -- * Four-channel 16-bit signed planar to packed image copy.
  -- * \param aSrc Planar \ref source_planar_image_pointer_array.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_16s_P4C4R
     (aSrc : System.Address;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:2499
   pragma Import (C, nppiCopy_16s_P4C4R, "nppiCopy_16s_P4C4R");

  --* 
  -- * Three-channel 32-bit signed planar to packed image copy.
  -- * \param aSrc Planar \ref source_planar_image_pointer_array.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_32s_P3C3R
     (aSrc : System.Address;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:2511
   pragma Import (C, nppiCopy_32s_P3C3R, "nppiCopy_32s_P3C3R");

  --* 
  -- * Four-channel 32-bit signed planar to packed image copy.
  -- * \param aSrc Planar \ref source_planar_image_pointer_array.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_32s_P4C4R
     (aSrc : System.Address;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:2523
   pragma Import (C, nppiCopy_32s_P4C4R, "nppiCopy_32s_P4C4R");

  --* 
  -- * Three-channel 32-bit float planar to packed image copy.
  -- * \param aSrc Planar \ref source_planar_image_pointer_array.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_32f_P3C3R
     (aSrc : System.Address;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:2535
   pragma Import (C, nppiCopy_32f_P3C3R, "nppiCopy_32f_P3C3R");

  --* 
  -- * Four-channel 32-bit float planar to packed image copy.
  -- * \param aSrc Planar \ref source_planar_image_pointer_array.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopy_32f_P4C4R
     (aSrc : System.Address;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:2547
   pragma Import (C, nppiCopy_32f_P4C4R, "nppiCopy_32f_P4C4R");

  --* @} Planar-to-Packed Copy  
  --* @} image_copy  
  --* 
  -- * @defgroup image_convert Convert
  -- *
  -- * @{
  -- *
  --  

  --* 
  -- * @name Convert to Increase Bit-Depth
  -- *
  -- * The integer conversion methods do not involve any scaling. Also, even when increasing the bit-depth
  -- * loss of information may occur:
  -- * - When converting integers (e.g. Npp32u) to float (e.g. Npp32f) integervalue not accurately representable 
  -- *   by the float are rounded to the closest floating-point value.
  -- * - When converting signed integers to unsigned integers all negative values are lost (saturated to 0).
  -- *
  -- * @{
  -- *
  --  

  --* 
  -- * Single channel 8-bit unsigned to 16-bit unsigned conversion.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_8u16u_C1R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:2584
   pragma Import (C, nppiConvert_8u16u_C1R, "nppiConvert_8u16u_C1R");

  --* 
  -- * Three channel 8-bit unsigned to 16-bit unsigned  conversion.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_8u16u_C3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:2597
   pragma Import (C, nppiConvert_8u16u_C3R, "nppiConvert_8u16u_C3R");

  --* 
  -- * Four channel 8-bit unsigned to 16-bit unsigned  conversion.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_8u16u_C4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:2610
   pragma Import (C, nppiConvert_8u16u_C4R, "nppiConvert_8u16u_C4R");

  --* 
  -- * Four channel 8-bit unsigned to 16-bit unsigned conversion, not affecting Alpha.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_8u16u_AC4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:2623
   pragma Import (C, nppiConvert_8u16u_AC4R, "nppiConvert_8u16u_AC4R");

  --* 
  -- * Single channel 8-bit unsigned to 16-bit signed conversion.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_8u16s_C1R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:2636
   pragma Import (C, nppiConvert_8u16s_C1R, "nppiConvert_8u16s_C1R");

  --* 
  -- * Three channel 8-bit unsigned to 16-bit signed conversion.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_8u16s_C3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:2649
   pragma Import (C, nppiConvert_8u16s_C3R, "nppiConvert_8u16s_C3R");

  --* 
  -- * Four channel 8-bit unsigned to 16-bit signed conversion.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_8u16s_C4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:2662
   pragma Import (C, nppiConvert_8u16s_C4R, "nppiConvert_8u16s_C4R");

  --* 
  -- * Four channel 8-bit unsigned to 16-bit signed conversion, not affecting Alpha.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_8u16s_AC4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:2675
   pragma Import (C, nppiConvert_8u16s_AC4R, "nppiConvert_8u16s_AC4R");

  --* 
  -- * Single channel 8-bit unsigned to 32-bit signed conversion.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_8u32s_C1R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:2688
   pragma Import (C, nppiConvert_8u32s_C1R, "nppiConvert_8u32s_C1R");

  --* 
  -- * Three channel 8-bit unsigned to 32-bit signed conversion.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_8u32s_C3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:2701
   pragma Import (C, nppiConvert_8u32s_C3R, "nppiConvert_8u32s_C3R");

  --* 
  -- * Four channel 8-bit unsigned to 32-bit signed conversion.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_8u32s_C4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:2714
   pragma Import (C, nppiConvert_8u32s_C4R, "nppiConvert_8u32s_C4R");

  --* 
  -- * Four channel 8-bit unsigned to 32-bit signed conversion, not affecting Alpha.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_8u32s_AC4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:2727
   pragma Import (C, nppiConvert_8u32s_AC4R, "nppiConvert_8u32s_AC4R");

  --* 
  -- * Single channel 8-bit unsigned to 32-bit floating-point conversion.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_8u32f_C1R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:2740
   pragma Import (C, nppiConvert_8u32f_C1R, "nppiConvert_8u32f_C1R");

  --* 
  -- * Three channel 8-bit unsigned to 32-bit floating-point conversion.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_8u32f_C3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:2753
   pragma Import (C, nppiConvert_8u32f_C3R, "nppiConvert_8u32f_C3R");

  --* 
  -- * Four channel 8-bit unsigned to 32-bit floating-point conversion.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_8u32f_C4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:2766
   pragma Import (C, nppiConvert_8u32f_C4R, "nppiConvert_8u32f_C4R");

  --* 
  -- * Four channel 8-bit unsigned to 32-bit floating-point conversion, not affecting Alpha.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_8u32f_AC4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:2779
   pragma Import (C, nppiConvert_8u32f_AC4R, "nppiConvert_8u32f_AC4R");

  --* 
  -- * Single channel 8-bit signed to 32-bit signed conversion.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_8s32s_C1R
     (pSrc : access nppdefs_h.Npp8s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:2792
   pragma Import (C, nppiConvert_8s32s_C1R, "nppiConvert_8s32s_C1R");

  --* 
  -- * Three channel 8-bit signed to 32-bit signed conversion.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_8s32s_C3R
     (pSrc : access nppdefs_h.Npp8s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:2805
   pragma Import (C, nppiConvert_8s32s_C3R, "nppiConvert_8s32s_C3R");

  --* 
  -- * Four channel 8-bit signed to 32-bit signed conversion.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_8s32s_C4R
     (pSrc : access nppdefs_h.Npp8s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:2818
   pragma Import (C, nppiConvert_8s32s_C4R, "nppiConvert_8s32s_C4R");

  --* 
  -- * Four channel 8-bit signed to 32-bit signed conversion, not affecting Alpha.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_8s32s_AC4R
     (pSrc : access nppdefs_h.Npp8s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:2831
   pragma Import (C, nppiConvert_8s32s_AC4R, "nppiConvert_8s32s_AC4R");

  --* 
  -- * Single channel 8-bit signed to 32-bit floating-point conversion.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_8s32f_C1R
     (pSrc : access nppdefs_h.Npp8s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:2844
   pragma Import (C, nppiConvert_8s32f_C1R, "nppiConvert_8s32f_C1R");

  --* 
  -- * Three channel 8-bit signed to 32-bit floating-point conversion.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_8s32f_C3R
     (pSrc : access nppdefs_h.Npp8s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:2857
   pragma Import (C, nppiConvert_8s32f_C3R, "nppiConvert_8s32f_C3R");

  --* 
  -- * Four channel 8-bit signed to 32-bit floating-point conversion.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_8s32f_C4R
     (pSrc : access nppdefs_h.Npp8s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:2870
   pragma Import (C, nppiConvert_8s32f_C4R, "nppiConvert_8s32f_C4R");

  --* 
  -- * Four channel 8-bit signed to 32-bit floating-point conversion, not affecting Alpha.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_8s32f_AC4R
     (pSrc : access nppdefs_h.Npp8s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:2883
   pragma Import (C, nppiConvert_8s32f_AC4R, "nppiConvert_8s32f_AC4R");

  --* 
  -- * Single channel 16-bit unsigned to 32-bit signed conversion.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_16u32s_C1R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:2896
   pragma Import (C, nppiConvert_16u32s_C1R, "nppiConvert_16u32s_C1R");

  --* 
  -- * Three channel 16-bit unsigned to 32-bit signed conversion.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_16u32s_C3R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:2909
   pragma Import (C, nppiConvert_16u32s_C3R, "nppiConvert_16u32s_C3R");

  --* 
  -- * Four channel 16-bit unsigned to 32-bit signed conversion.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_16u32s_C4R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:2922
   pragma Import (C, nppiConvert_16u32s_C4R, "nppiConvert_16u32s_C4R");

  --* 
  -- * Four channel 16-bit unsigned to 32-bit signed conversion, not affecting Alpha.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_16u32s_AC4R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:2935
   pragma Import (C, nppiConvert_16u32s_AC4R, "nppiConvert_16u32s_AC4R");

  --* 
  -- * Single channel 16-bit unsigned to 32-bit floating-point conversion.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_16u32f_C1R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:2948
   pragma Import (C, nppiConvert_16u32f_C1R, "nppiConvert_16u32f_C1R");

  --* 
  -- * Three channel 16-bit unsigned to 32-bit floating-point conversion.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_16u32f_C3R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:2961
   pragma Import (C, nppiConvert_16u32f_C3R, "nppiConvert_16u32f_C3R");

  --* 
  -- * Four channel 16-bit unsigned to 32-bit floating-point conversion.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_16u32f_C4R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:2974
   pragma Import (C, nppiConvert_16u32f_C4R, "nppiConvert_16u32f_C4R");

  --* 
  -- * Four channel 16-bit unsigned to 32-bit floating-point conversion, not affecting Alpha.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_16u32f_AC4R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:2987
   pragma Import (C, nppiConvert_16u32f_AC4R, "nppiConvert_16u32f_AC4R");

  --* 
  -- * Single channel 16-bit signed to 32-bit signed conversion.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_16s32s_C1R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:3000
   pragma Import (C, nppiConvert_16s32s_C1R, "nppiConvert_16s32s_C1R");

  --* 
  -- * Three channel 16-bit signed to 32-bit signed conversion.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_16s32s_C3R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:3013
   pragma Import (C, nppiConvert_16s32s_C3R, "nppiConvert_16s32s_C3R");

  --* 
  -- * Four channel 16-bit signed to 32-bit signed conversion.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_16s32s_C4R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:3026
   pragma Import (C, nppiConvert_16s32s_C4R, "nppiConvert_16s32s_C4R");

  --* 
  -- * Four channel 16-bit signed to 32-bit signed conversion, not affecting Alpha.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_16s32s_AC4R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:3039
   pragma Import (C, nppiConvert_16s32s_AC4R, "nppiConvert_16s32s_AC4R");

  --* 
  -- * Single channel 16-bit signed to 32-bit floating-point conversion.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_16s32f_C1R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:3052
   pragma Import (C, nppiConvert_16s32f_C1R, "nppiConvert_16s32f_C1R");

  --* 
  -- * Three channel 16-bit signed to 32-bit floating-point conversion.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_16s32f_C3R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:3065
   pragma Import (C, nppiConvert_16s32f_C3R, "nppiConvert_16s32f_C3R");

  --* 
  -- * Four channel 16-bit signed to 32-bit floating-point conversion.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_16s32f_C4R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:3078
   pragma Import (C, nppiConvert_16s32f_C4R, "nppiConvert_16s32f_C4R");

  --* 
  -- * Four channel 16-bit signed to 32-bit floating-point conversion, not affecting Alpha.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_16s32f_AC4R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:3091
   pragma Import (C, nppiConvert_16s32f_AC4R, "nppiConvert_16s32f_AC4R");

  --* 
  -- * Single channel 8-bit signed to 8-bit unsigned conversion with saturation.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_8s8u_C1Rs
     (pSrc : access nppdefs_h.Npp8s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:3104
   pragma Import (C, nppiConvert_8s8u_C1Rs, "nppiConvert_8s8u_C1Rs");

  --* 
  -- * Single channel 8-bit signed to 16-bit unsigned conversion with saturation.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_8s16u_C1Rs
     (pSrc : access nppdefs_h.Npp8s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:3117
   pragma Import (C, nppiConvert_8s16u_C1Rs, "nppiConvert_8s16u_C1Rs");

  --* 
  -- * Single channel 8-bit signed to 16-bit signed conversion.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_8s16s_C1R
     (pSrc : access nppdefs_h.Npp8s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:3130
   pragma Import (C, nppiConvert_8s16s_C1R, "nppiConvert_8s16s_C1R");

  --* 
  -- * Single channel 8-bit signed to 32-bit unsigned conversion with saturation.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_8s32u_C1Rs
     (pSrc : access nppdefs_h.Npp8s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:3143
   pragma Import (C, nppiConvert_8s32u_C1Rs, "nppiConvert_8s32u_C1Rs");

  --* 
  -- * Single channel 16-bit signed to 16-bit unsigned conversion with saturation.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_16s16u_C1Rs
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:3156
   pragma Import (C, nppiConvert_16s16u_C1Rs, "nppiConvert_16s16u_C1Rs");

  --* 
  -- * Single channel 16-bit signed to 32-bit unsigned conversion with saturation.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_16s32u_C1Rs
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:3169
   pragma Import (C, nppiConvert_16s32u_C1Rs, "nppiConvert_16s32u_C1Rs");

  --* 
  -- * Single channel 16-bit unsigned to 32-bit unsigned conversion.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_16u32u_C1R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:3182
   pragma Import (C, nppiConvert_16u32u_C1R, "nppiConvert_16u32u_C1R");

  --* 
  -- * Single channel 32-bit signed to 32-bit unsigned conversion with saturation.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_32s32u_C1Rs
     (pSrc : access nppdefs_h.Npp32s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:3196
   pragma Import (C, nppiConvert_32s32u_C1Rs, "nppiConvert_32s32u_C1Rs");

  --* 
  -- * Single channel 32-bit signed to 32-bit floating-point conversion.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_32s32f_C1R
     (pSrc : access nppdefs_h.Npp32s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:3209
   pragma Import (C, nppiConvert_32s32f_C1R, "nppiConvert_32s32f_C1R");

  --* 
  -- * Single channel 32-bit unsigned to 32-bit floating-point conversion.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_32u32f_C1R
     (pSrc : access nppdefs_h.Npp32u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:3222
   pragma Import (C, nppiConvert_32u32f_C1R, "nppiConvert_32u32f_C1R");

  --* @} Convert to Increase Bit-Depth  
  --* 
  -- * @name Convert to Decrease Bit-Depth
  -- *
  -- * The integer conversion methods do not involve any scaling. When converting floating-point values
  -- * to integers the user may choose the most appropriate rounding-mode. Typically information is lost when
  -- * converting to lower bit depth:
  -- * - All converted values are saturated to the destination type's range. E.g. any values larger than
  -- *   the largest value of the destination type are clamped to the destination's maximum.
  -- * - Converting floating-point values to integer also involves rounding, effectively loosing all
  -- *   fractional value information in the process. 
  -- *
  -- * @{
  -- *
  --  

  --* 
  -- * Single channel 16-bit unsigned to 8-bit unsigned conversion.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_16u8u_C1R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:3253
   pragma Import (C, nppiConvert_16u8u_C1R, "nppiConvert_16u8u_C1R");

  --* 
  -- * Three channel 16-bit unsigned to 8-bit unsigned conversion.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_16u8u_C3R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:3266
   pragma Import (C, nppiConvert_16u8u_C3R, "nppiConvert_16u8u_C3R");

  --* 
  -- * Four channel 16-bit unsigned to 8-bit unsigned conversion.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_16u8u_C4R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:3279
   pragma Import (C, nppiConvert_16u8u_C4R, "nppiConvert_16u8u_C4R");

  --* 
  -- * Four channel 16-bit unsigned to 8-bit unsigned conversion, not affecting Alpha.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_16u8u_AC4R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:3292
   pragma Import (C, nppiConvert_16u8u_AC4R, "nppiConvert_16u8u_AC4R");

  --* 
  -- * Single channel 16-bit signed to 8-bit unsigned conversion.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_16s8u_C1R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:3306
   pragma Import (C, nppiConvert_16s8u_C1R, "nppiConvert_16s8u_C1R");

  --* 
  -- * Three channel 16-bit signed to 8-bit unsigned conversion.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_16s8u_C3R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:3319
   pragma Import (C, nppiConvert_16s8u_C3R, "nppiConvert_16s8u_C3R");

  --* 
  -- * Four channel 16-bit signed to 8-bit unsigned conversion.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_16s8u_C4R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:3332
   pragma Import (C, nppiConvert_16s8u_C4R, "nppiConvert_16s8u_C4R");

  --* 
  -- * Four channel 16-bit signed to 8-bit unsigned conversion, not affecting Alpha.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_16s8u_AC4R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:3345
   pragma Import (C, nppiConvert_16s8u_AC4R, "nppiConvert_16s8u_AC4R");

  --* 
  -- * Single channel 32-bit signed to 8-bit unsigned conversion.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_32s8u_C1R
     (pSrc : access nppdefs_h.Npp32s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:3359
   pragma Import (C, nppiConvert_32s8u_C1R, "nppiConvert_32s8u_C1R");

  --* 
  -- * Three channel 32-bit signed to 8-bit unsigned conversion.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_32s8u_C3R
     (pSrc : access nppdefs_h.Npp32s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:3372
   pragma Import (C, nppiConvert_32s8u_C3R, "nppiConvert_32s8u_C3R");

  --* 
  -- * Four channel 32-bit signed to 8-bit unsigned conversion.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_32s8u_C4R
     (pSrc : access nppdefs_h.Npp32s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:3385
   pragma Import (C, nppiConvert_32s8u_C4R, "nppiConvert_32s8u_C4R");

  --* 
  -- * Four channel 32-bit signed to 8-bit unsigned conversion, not affecting Alpha.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_32s8u_AC4R
     (pSrc : access nppdefs_h.Npp32s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:3398
   pragma Import (C, nppiConvert_32s8u_AC4R, "nppiConvert_32s8u_AC4R");

  --* 
  -- * Single channel 32-bit signed to 8-bit signed conversion.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_32s8s_C1R
     (pSrc : access nppdefs_h.Npp32s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:3412
   pragma Import (C, nppiConvert_32s8s_C1R, "nppiConvert_32s8s_C1R");

  --* 
  -- * Three channel 32-bit signed to 8-bit signed conversion.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_32s8s_C3R
     (pSrc : access nppdefs_h.Npp32s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:3425
   pragma Import (C, nppiConvert_32s8s_C3R, "nppiConvert_32s8s_C3R");

  --* 
  -- * Four channel 32-bit signed to 8-bit signed conversion.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_32s8s_C4R
     (pSrc : access nppdefs_h.Npp32s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:3438
   pragma Import (C, nppiConvert_32s8s_C4R, "nppiConvert_32s8s_C4R");

  --* 
  -- * Four channel 32-bit signed to 8-bit signed conversion, not affecting Alpha.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_32s8s_AC4R
     (pSrc : access nppdefs_h.Npp32s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:3451
   pragma Import (C, nppiConvert_32s8s_AC4R, "nppiConvert_32s8s_AC4R");

  --* 
  -- * Single channel 8-bit unsigned to 8-bit signed conversion.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eRoundMode \ref rounding_mode_parameter.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_8u8s_C1RSfs
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eRoundMode : nppdefs_h.NppRoundMode;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:3466
   pragma Import (C, nppiConvert_8u8s_C1RSfs, "nppiConvert_8u8s_C1RSfs");

  --* 
  -- * Single channel 16-bit unsigned to 8-bit signed conversion.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eRoundMode \ref rounding_mode_parameter.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_16u8s_C1RSfs
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eRoundMode : nppdefs_h.NppRoundMode;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:3481
   pragma Import (C, nppiConvert_16u8s_C1RSfs, "nppiConvert_16u8s_C1RSfs");

  --* 
  -- * Single channel 16-bit signed to 8-bit signed conversion.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eRoundMode \ref rounding_mode_parameter.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_16s8s_C1RSfs
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eRoundMode : nppdefs_h.NppRoundMode;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:3496
   pragma Import (C, nppiConvert_16s8s_C1RSfs, "nppiConvert_16s8s_C1RSfs");

  --* 
  -- * Single channel 16-bit unsigned to 16-bit signed conversion.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eRoundMode \ref rounding_mode_parameter.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_16u16s_C1RSfs
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eRoundMode : nppdefs_h.NppRoundMode;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:3511
   pragma Import (C, nppiConvert_16u16s_C1RSfs, "nppiConvert_16u16s_C1RSfs");

  --* 
  -- * Single channel 32-bit unsigned to 8-bit unsigned conversion.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eRoundMode \ref rounding_mode_parameter.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_32u8u_C1RSfs
     (pSrc : access nppdefs_h.Npp32u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eRoundMode : nppdefs_h.NppRoundMode;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:3526
   pragma Import (C, nppiConvert_32u8u_C1RSfs, "nppiConvert_32u8u_C1RSfs");

  --* 
  -- * Single channel 32-bit unsigned to 8-bit signed conversion.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eRoundMode \ref rounding_mode_parameter.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_32u8s_C1RSfs
     (pSrc : access nppdefs_h.Npp32u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eRoundMode : nppdefs_h.NppRoundMode;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:3541
   pragma Import (C, nppiConvert_32u8s_C1RSfs, "nppiConvert_32u8s_C1RSfs");

  --* 
  -- * Single channel 32-bit unsigned to 16-bit unsigned conversion.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eRoundMode \ref rounding_mode_parameter.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_32u16u_C1RSfs
     (pSrc : access nppdefs_h.Npp32u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eRoundMode : nppdefs_h.NppRoundMode;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:3556
   pragma Import (C, nppiConvert_32u16u_C1RSfs, "nppiConvert_32u16u_C1RSfs");

  --* 
  -- * Single channel 32-bit unsigned to 16-bit signed conversion.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eRoundMode \ref rounding_mode_parameter.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_32u16s_C1RSfs
     (pSrc : access nppdefs_h.Npp32u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eRoundMode : nppdefs_h.NppRoundMode;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:3571
   pragma Import (C, nppiConvert_32u16s_C1RSfs, "nppiConvert_32u16s_C1RSfs");

  --* 
  -- * Single channel 32-bit unsigned to 32-bit signed conversion.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eRoundMode \ref rounding_mode_parameter.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_32u32s_C1RSfs
     (pSrc : access nppdefs_h.Npp32u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eRoundMode : nppdefs_h.NppRoundMode;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:3586
   pragma Import (C, nppiConvert_32u32s_C1RSfs, "nppiConvert_32u32s_C1RSfs");

  --* 
  -- * Single channel 32-bit unsigned to 16-bit unsigned conversion.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eRoundMode \ref rounding_mode_parameter.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_32s16u_C1RSfs
     (pSrc : access nppdefs_h.Npp32s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eRoundMode : nppdefs_h.NppRoundMode;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:3601
   pragma Import (C, nppiConvert_32s16u_C1RSfs, "nppiConvert_32s16u_C1RSfs");

  --* 
  -- * Single channel 32-bit unsigned to 16-bit signed conversion.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eRoundMode \ref rounding_mode_parameter.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_32s16s_C1RSfs
     (pSrc : access nppdefs_h.Npp32s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eRoundMode : nppdefs_h.NppRoundMode;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:3616
   pragma Import (C, nppiConvert_32s16s_C1RSfs, "nppiConvert_32s16s_C1RSfs");

  --* 
  -- * Single channel 32-bit floating point to 8-bit unsigned conversion.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eRoundMode Flag specifying how fractional float values are rounded to integer values.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_32f8u_C1R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eRoundMode : nppdefs_h.NppRoundMode) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:3630
   pragma Import (C, nppiConvert_32f8u_C1R, "nppiConvert_32f8u_C1R");

  --* 
  -- * Three channel 32-bit floating point to 8-bit unsigned conversion.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eRoundMode Flag specifying how fractional float values are rounded to integer values.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_32f8u_C3R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eRoundMode : nppdefs_h.NppRoundMode) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:3644
   pragma Import (C, nppiConvert_32f8u_C3R, "nppiConvert_32f8u_C3R");

  --* 
  -- * Four channel 32-bit floating point to 8-bit unsigned conversion.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eRoundMode Flag specifying how fractional float values are rounded to integer values.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_32f8u_C4R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eRoundMode : nppdefs_h.NppRoundMode) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:3658
   pragma Import (C, nppiConvert_32f8u_C4R, "nppiConvert_32f8u_C4R");

  --* 
  -- * Four channel 32-bit floating point to 8-bit unsigned conversion, not affecting Alpha.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eRoundMode Flag specifying how fractional float values are rounded to integer values.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_32f8u_AC4R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eRoundMode : nppdefs_h.NppRoundMode) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:3672
   pragma Import (C, nppiConvert_32f8u_AC4R, "nppiConvert_32f8u_AC4R");

  --* 
  -- * Single channel 32-bit floating point to 8-bit signed conversion.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eRoundMode Flag specifying how fractional float values are rounded to integer values.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_32f8s_C1R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eRoundMode : nppdefs_h.NppRoundMode) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:3686
   pragma Import (C, nppiConvert_32f8s_C1R, "nppiConvert_32f8s_C1R");

  --* 
  -- * Three channel 32-bit floating point to 8-bit signed conversion.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eRoundMode Flag specifying how fractional float values are rounded to integer values.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_32f8s_C3R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eRoundMode : nppdefs_h.NppRoundMode) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:3700
   pragma Import (C, nppiConvert_32f8s_C3R, "nppiConvert_32f8s_C3R");

  --* 
  -- * Four channel 32-bit floating point to 8-bit signed conversion.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eRoundMode Flag specifying how fractional float values are rounded to integer values.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_32f8s_C4R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eRoundMode : nppdefs_h.NppRoundMode) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:3714
   pragma Import (C, nppiConvert_32f8s_C4R, "nppiConvert_32f8s_C4R");

  --* 
  -- * Four channel 32-bit floating point to 8-bit signed conversion, not affecting Alpha.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eRoundMode Flag specifying how fractional float values are rounded to integer values.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_32f8s_AC4R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eRoundMode : nppdefs_h.NppRoundMode) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:3728
   pragma Import (C, nppiConvert_32f8s_AC4R, "nppiConvert_32f8s_AC4R");

  --* 
  -- * Single channel 32-bit floating point to 16-bit unsigned conversion.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eRoundMode Flag specifying how fractional float values are rounded to integer values.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_32f16u_C1R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eRoundMode : nppdefs_h.NppRoundMode) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:3742
   pragma Import (C, nppiConvert_32f16u_C1R, "nppiConvert_32f16u_C1R");

  --* 
  -- * Three channel 32-bit floating point to 16-bit unsigned conversion.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eRoundMode Flag specifying how fractional float values are rounded to integer values.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_32f16u_C3R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eRoundMode : nppdefs_h.NppRoundMode) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:3756
   pragma Import (C, nppiConvert_32f16u_C3R, "nppiConvert_32f16u_C3R");

  --* 
  -- * Four channel 32-bit floating point to 16-bit unsigned conversion.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eRoundMode Flag specifying how fractional float values are rounded to integer values.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_32f16u_C4R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eRoundMode : nppdefs_h.NppRoundMode) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:3770
   pragma Import (C, nppiConvert_32f16u_C4R, "nppiConvert_32f16u_C4R");

  --* 
  -- * Four channel 32-bit floating point to 16-bit unsigned conversion, not affecting Alpha.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eRoundMode Flag specifying how fractional float values are rounded to integer values.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_32f16u_AC4R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eRoundMode : nppdefs_h.NppRoundMode) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:3784
   pragma Import (C, nppiConvert_32f16u_AC4R, "nppiConvert_32f16u_AC4R");

  --* 
  -- * Single channel 32-bit floating point to 16-bit signed conversion.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eRoundMode Flag specifying how fractional float values are rounded to integer values.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_32f16s_C1R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eRoundMode : nppdefs_h.NppRoundMode) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:3798
   pragma Import (C, nppiConvert_32f16s_C1R, "nppiConvert_32f16s_C1R");

  --* 
  -- * Three channel 32-bit floating point to 16-bit signed conversion.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eRoundMode Flag specifying how fractional float values are rounded to integer values.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_32f16s_C3R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eRoundMode : nppdefs_h.NppRoundMode) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:3812
   pragma Import (C, nppiConvert_32f16s_C3R, "nppiConvert_32f16s_C3R");

  --* 
  -- * Four channel 32-bit floating point to 16-bit signed conversion.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eRoundMode Flag specifying how fractional float values are rounded to integer values.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_32f16s_C4R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eRoundMode : nppdefs_h.NppRoundMode) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:3826
   pragma Import (C, nppiConvert_32f16s_C4R, "nppiConvert_32f16s_C4R");

  --* 
  -- * Four channel 32-bit floating point to 16-bit signed conversion.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eRoundMode Flag specifying how fractional float values are rounded to integer values.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_32f16s_AC4R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eRoundMode : nppdefs_h.NppRoundMode) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:3840
   pragma Import (C, nppiConvert_32f16s_AC4R, "nppiConvert_32f16s_AC4R");

  --* 
  -- * Single channel 32-bit floating point to 8-bit unsigned conversion.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eRoundMode Flag specifying how fractional float values are rounded to integer values.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_32f8u_C1RSfs
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eRoundMode : nppdefs_h.NppRoundMode;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:3856
   pragma Import (C, nppiConvert_32f8u_C1RSfs, "nppiConvert_32f8u_C1RSfs");

  --* 
  -- * Single channel 32-bit floating point to 8-bit signed conversion.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eRoundMode Flag specifying how fractional float values are rounded to integer values.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_32f8s_C1RSfs
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eRoundMode : nppdefs_h.NppRoundMode;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:3871
   pragma Import (C, nppiConvert_32f8s_C1RSfs, "nppiConvert_32f8s_C1RSfs");

  --* 
  -- * Single channel 32-bit floating point to 16-bit unsigned conversion.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eRoundMode Flag specifying how fractional float values are rounded to integer values.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_32f16u_C1RSfs
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eRoundMode : nppdefs_h.NppRoundMode;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:3886
   pragma Import (C, nppiConvert_32f16u_C1RSfs, "nppiConvert_32f16u_C1RSfs");

  --* 
  -- * Single channel 32-bit floating point to 16-bit signed conversion.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eRoundMode Flag specifying how fractional float values are rounded to integer values.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_32f16s_C1RSfs
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eRoundMode : nppdefs_h.NppRoundMode;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:3901
   pragma Import (C, nppiConvert_32f16s_C1RSfs, "nppiConvert_32f16s_C1RSfs");

  --* 
  -- * Single channel 32-bit floating point to 32-bit unsigned conversion.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eRoundMode Flag specifying how fractional float values are rounded to integer values.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_32f32u_C1RSfs
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eRoundMode : nppdefs_h.NppRoundMode;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:3916
   pragma Import (C, nppiConvert_32f32u_C1RSfs, "nppiConvert_32f32u_C1RSfs");

  --* 
  -- * Single channel 32-bit floating point to 32-bit signed conversion.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eRoundMode Flag specifying how fractional float values are rounded to integer values.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiConvert_32f32s_C1RSfs
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eRoundMode : nppdefs_h.NppRoundMode;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:3931
   pragma Import (C, nppiConvert_32f32s_C1RSfs, "nppiConvert_32f32s_C1RSfs");

  --* @} Convert to Decrease Bit-Depth  
  --* @} image_convert  
  --* 
  -- * @defgroup image_scale Scale
  -- *
  -- * @{
  -- *
  --  

  --*
  -- * @name Scaled Bit-Depth Conversion
  -- * Scale bit-depth up and down.
  -- *
  -- * To map source pixel srcPixelValue to destination pixel dstPixelValue the following equation is used:
  -- * 
  -- *      dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)
  -- *
  -- * where scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue).
  -- *
  -- * For conversions between integer data types, the entire integer numeric range of the input data type is mapped onto 
  -- * the entire integer numeric range of the output data type.
  -- *
  -- * For conversions to floating point data types the floating point data range is defined by the user supplied floating point values
  -- * of nMax and nMin which are used as the dstMaxRangeValue and dstMinRangeValue respectively in the scaleFactor and dstPixelValue 
  -- * calculations and also as the saturation values to which output data is clamped.
  -- *
  -- * When converting from floating-point values to integer values, nMax and nMin are used as the srcMaxRangeValue and srcMinRangeValue
  -- * respectively in the scaleFactor and dstPixelValue calculations. Output values are saturated and clamped to the full output integer
  -- * pixel value range.
  -- *
  -- * @{
  -- *
  --  

  --* 
  -- * Single channel 8-bit unsigned to 16-bit unsigned conversion.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiScale_8u16u_C1R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:3981
   pragma Import (C, nppiScale_8u16u_C1R, "nppiScale_8u16u_C1R");

  --* 
  -- * Three channel 8-bit unsigned to 16-bit unsigned  conversion.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiScale_8u16u_C3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:3994
   pragma Import (C, nppiScale_8u16u_C3R, "nppiScale_8u16u_C3R");

  --* 
  -- * Four channel 8-bit unsigned to 16-bit unsigned  conversion.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiScale_8u16u_C4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:4007
   pragma Import (C, nppiScale_8u16u_C4R, "nppiScale_8u16u_C4R");

  --* 
  -- * Four channel 8-bit unsigned to 16-bit unsigned conversion, not affecting Alpha.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiScale_8u16u_AC4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:4020
   pragma Import (C, nppiScale_8u16u_AC4R, "nppiScale_8u16u_AC4R");

  --* 
  -- * Single channel 8-bit unsigned to 16-bit signed conversion.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiScale_8u16s_C1R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:4033
   pragma Import (C, nppiScale_8u16s_C1R, "nppiScale_8u16s_C1R");

  --* 
  -- * Three channel 8-bit unsigned to 16-bit signed conversion.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiScale_8u16s_C3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:4046
   pragma Import (C, nppiScale_8u16s_C3R, "nppiScale_8u16s_C3R");

  --* 
  -- * Four channel 8-bit unsigned to 16-bit signed conversion.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiScale_8u16s_C4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:4059
   pragma Import (C, nppiScale_8u16s_C4R, "nppiScale_8u16s_C4R");

  --* 
  -- * Four channel 8-bit unsigned to 16-bit signed conversion, not affecting Alpha.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiScale_8u16s_AC4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:4072
   pragma Import (C, nppiScale_8u16s_AC4R, "nppiScale_8u16s_AC4R");

  --* 
  -- * Single channel 8-bit unsigned to 32-bit signed conversion.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiScale_8u32s_C1R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:4085
   pragma Import (C, nppiScale_8u32s_C1R, "nppiScale_8u32s_C1R");

  --* 
  -- * Three channel 8-bit unsigned to 32-bit signed conversion.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiScale_8u32s_C3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:4098
   pragma Import (C, nppiScale_8u32s_C3R, "nppiScale_8u32s_C3R");

  --* 
  -- * Four channel 8-bit unsigned to 32-bit signed conversion.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiScale_8u32s_C4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:4111
   pragma Import (C, nppiScale_8u32s_C4R, "nppiScale_8u32s_C4R");

  --* 
  -- * Four channel 8-bit unsigned to 32-bit signed conversion, not affecting Alpha.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiScale_8u32s_AC4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:4124
   pragma Import (C, nppiScale_8u32s_AC4R, "nppiScale_8u32s_AC4R");

  --* 
  -- * Single channel 8-bit unsigned to 32-bit floating-point conversion.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nMin specifies the minimum saturation value to which every output value will be clamped.
  -- * \param nMax specifies the maximum saturation value to which every output value will be clamped.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, ::NPP_SCALE_RANGE_ERROR indicates an error condition if nMax <= nMin.
  --  

   function nppiScale_8u32f_C1R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nMin : nppdefs_h.Npp32f;
      nMax : nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:4139
   pragma Import (C, nppiScale_8u32f_C1R, "nppiScale_8u32f_C1R");

  --* 
  -- * Three channel 8-bit unsigned to 32-bit floating-point conversion.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nMin specifies the minimum saturation value to which every output value will be clamped.
  -- * \param nMax specifies the maximum saturation value to which every output value will be clamped.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, ::NPP_SCALE_RANGE_ERROR indicates an error condition if nMax <= nMin.
  --  

   function nppiScale_8u32f_C3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nMin : nppdefs_h.Npp32f;
      nMax : nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:4154
   pragma Import (C, nppiScale_8u32f_C3R, "nppiScale_8u32f_C3R");

  --* 
  -- * Four channel 8-bit unsigned to 32-bit floating-point conversion.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nMin specifies the minimum saturation value to which every output value will be clamped.
  -- * \param nMax specifies the maximum saturation value to which every output value will be clamped.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, ::NPP_SCALE_RANGE_ERROR indicates an error condition if nMax <= nMin.
  --  

   function nppiScale_8u32f_C4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nMin : nppdefs_h.Npp32f;
      nMax : nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:4169
   pragma Import (C, nppiScale_8u32f_C4R, "nppiScale_8u32f_C4R");

  --* 
  -- * Four channel 8-bit unsigned to 32-bit floating-point conversion, not affecting Alpha.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nMin specifies the minimum saturation value to which every output value will be clamped.
  -- * \param nMax specifies the maximum saturation value to which every output value will be clamped.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, ::NPP_SCALE_RANGE_ERROR indicates an error condition if nMax <= nMin.
  --  

   function nppiScale_8u32f_AC4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nMin : nppdefs_h.Npp32f;
      nMax : nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:4184
   pragma Import (C, nppiScale_8u32f_AC4R, "nppiScale_8u32f_AC4R");

  --* 
  -- * Single channel 16-bit unsigned to 8-bit unsigned conversion.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param hint algorithm performance or accuracy selector, currently ignored
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiScale_16u8u_C1R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      hint : nppdefs_h.NppHintAlgorithm) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:4198
   pragma Import (C, nppiScale_16u8u_C1R, "nppiScale_16u8u_C1R");

  --* 
  -- * Three channel 16-bit unsigned to 8-bit unsigned conversion.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param hint algorithm performance or accuracy selector, currently ignored
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiScale_16u8u_C3R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      hint : nppdefs_h.NppHintAlgorithm) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:4212
   pragma Import (C, nppiScale_16u8u_C3R, "nppiScale_16u8u_C3R");

  --* 
  -- * Four channel 16-bit unsigned to 8-bit unsigned conversion.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param hint algorithm performance or accuracy selector, currently ignored
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiScale_16u8u_C4R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      hint : nppdefs_h.NppHintAlgorithm) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:4226
   pragma Import (C, nppiScale_16u8u_C4R, "nppiScale_16u8u_C4R");

  --* 
  -- * Four channel 16-bit unsigned to 8-bit unsigned conversion, not affecting Alpha.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param hint algorithm performance or accuracy selector, currently ignored
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiScale_16u8u_AC4R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      hint : nppdefs_h.NppHintAlgorithm) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:4240
   pragma Import (C, nppiScale_16u8u_AC4R, "nppiScale_16u8u_AC4R");

  --* 
  -- * Single channel 16-bit signed to 8-bit unsigned conversion.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param hint algorithm performance or accuracy selector, currently ignored
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiScale_16s8u_C1R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      hint : nppdefs_h.NppHintAlgorithm) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:4255
   pragma Import (C, nppiScale_16s8u_C1R, "nppiScale_16s8u_C1R");

  --* 
  -- * Three channel 16-bit signed to 8-bit unsigned conversion.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param hint algorithm performance or accuracy selector, currently ignored
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiScale_16s8u_C3R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      hint : nppdefs_h.NppHintAlgorithm) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:4269
   pragma Import (C, nppiScale_16s8u_C3R, "nppiScale_16s8u_C3R");

  --* 
  -- * Four channel 16-bit signed to 8-bit unsigned conversion.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param hint algorithm performance or accuracy selector, currently ignored
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiScale_16s8u_C4R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      hint : nppdefs_h.NppHintAlgorithm) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:4283
   pragma Import (C, nppiScale_16s8u_C4R, "nppiScale_16s8u_C4R");

  --* 
  -- * Four channel 16-bit signed to 8-bit unsigned conversion, not affecting Alpha.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param hint algorithm performance or accuracy selector, currently ignored
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiScale_16s8u_AC4R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      hint : nppdefs_h.NppHintAlgorithm) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:4297
   pragma Import (C, nppiScale_16s8u_AC4R, "nppiScale_16s8u_AC4R");

  --* 
  -- * Single channel 32-bit signed to 8-bit unsigned conversion.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param hint algorithm performance or accuracy selector, currently ignored
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiScale_32s8u_C1R
     (pSrc : access nppdefs_h.Npp32s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      hint : nppdefs_h.NppHintAlgorithm) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:4312
   pragma Import (C, nppiScale_32s8u_C1R, "nppiScale_32s8u_C1R");

  --* 
  -- * Three channel 32-bit signed to 8-bit unsigned conversion.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param hint algorithm performance or accuracy selector, currently ignored
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiScale_32s8u_C3R
     (pSrc : access nppdefs_h.Npp32s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      hint : nppdefs_h.NppHintAlgorithm) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:4326
   pragma Import (C, nppiScale_32s8u_C3R, "nppiScale_32s8u_C3R");

  --* 
  -- * Four channel 32-bit signed to 8-bit unsigned conversion.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param hint algorithm performance or accuracy selector, currently ignored
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiScale_32s8u_C4R
     (pSrc : access nppdefs_h.Npp32s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      hint : nppdefs_h.NppHintAlgorithm) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:4340
   pragma Import (C, nppiScale_32s8u_C4R, "nppiScale_32s8u_C4R");

  --* 
  -- * Four channel 32-bit signed to 8-bit unsigned conversion, not affecting Alpha.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param hint algorithm performance or accuracy selector, currently ignored
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiScale_32s8u_AC4R
     (pSrc : access nppdefs_h.Npp32s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      hint : nppdefs_h.NppHintAlgorithm) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:4354
   pragma Import (C, nppiScale_32s8u_AC4R, "nppiScale_32s8u_AC4R");

  --* 
  -- * Single channel 32-bit floating point to 8-bit unsigned conversion.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nMin specifies the minimum saturation value to which every output value will be clamped.
  -- * \param nMax specifies the maximum saturation value to which every output value will be clamped.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, ::NPP_SCALE_RANGE_ERROR indicates an error condition if nMax <= nMin.
  --  

   function nppiScale_32f8u_C1R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nMin : nppdefs_h.Npp32f;
      nMax : nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:4369
   pragma Import (C, nppiScale_32f8u_C1R, "nppiScale_32f8u_C1R");

  --* 
  -- * Three channel 32-bit floating point to 8-bit unsigned conversion.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nMin specifies the minimum saturation value to which every output value will be clamped.
  -- * \param nMax specifies the maximum saturation value to which every output value will be clamped.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, ::NPP_SCALE_RANGE_ERROR indicates an error condition if nMax <= nMin.
  --  

   function nppiScale_32f8u_C3R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nMin : nppdefs_h.Npp32f;
      nMax : nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:4384
   pragma Import (C, nppiScale_32f8u_C3R, "nppiScale_32f8u_C3R");

  --* 
  -- * Four channel 32-bit floating point to 8-bit unsigned conversion.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nMin specifies the minimum saturation value to which every output value will be clamped.
  -- * \param nMax specifies the maximum saturation value to which every output value will be clamped.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, ::NPP_SCALE_RANGE_ERROR indicates an error condition if nMax <= nMin.
  --  

   function nppiScale_32f8u_C4R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nMin : nppdefs_h.Npp32f;
      nMax : nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:4399
   pragma Import (C, nppiScale_32f8u_C4R, "nppiScale_32f8u_C4R");

  --* 
  -- * Four channel 32-bit floating point to 8-bit unsigned conversion, not affecting Alpha.
  -- * 
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nMin specifies the minimum saturation value to which every output value will be clamped.
  -- * \param nMax specifies the maximum saturation value to which every output value will be clamped.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes, ::NPP_SCALE_RANGE_ERROR indicates an error condition if nMax <= nMin.
  --  

   function nppiScale_32f8u_AC4R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nMin : nppdefs_h.Npp32f;
      nMax : nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:4414
   pragma Import (C, nppiScale_32f8u_AC4R, "nppiScale_32f8u_AC4R");

  --* @} Scaled Bit-Depth Conversion  
  --* @} image_scale  
  --* 
  -- * @defgroup image_copy_constant_border Copy Constant Border
  -- * 
  -- * @{
  -- *
  --  

  --* @name CopyConstBorder
  -- * 
  -- * Methods for copying images and padding borders with a constant, user-specifiable color.
  -- *
  -- * @{
  -- *
  --  

  --* 
  -- * 1 channel 8-bit unsigned integer image copy with constant border color.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSizeROI Size of the source region of pixels.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size (width, height) of the destination region, i.e. the region that gets filled with
  -- *      data from the source image (inner part) and constant border color (outer part).
  -- * \param nTopBorderHeight Height (in pixels) of the top border. The number of pixel rows at the top of the
  -- *      destination ROI that will be filled with the constant border color.
  -- *      nBottomBorderHeight = oDstSizeROI.height - nTopBorderHeight - oSrcSizeROI.height.
  -- * \param nLeftBorderWidth Width (in pixels) of the left border. The width of the border at the right side of the
  -- *      destination ROI is implicitly defined by the size of the source ROI:
  -- *      nRightBorderWidth = oDstSizeROI.width - nLeftBorderWidth - oSrcSizeROI.width.
  -- * \param nValue The pixel value to be set for border pixels.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopyConstBorder_8u_C1R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      oSrcSizeROI : nppdefs_h.NppiSize;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      nTopBorderHeight : int;
      nLeftBorderWidth : int;
      nValue : nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:4453
   pragma Import (C, nppiCopyConstBorder_8u_C1R, "nppiCopyConstBorder_8u_C1R");

  --*
  -- * 3 channel 8-bit unsigned integer image copy with constant border color.
  -- * See nppiCopyConstBorder_8u_C1R() for detailed documentation.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSizeROI Size of the source region-of-interest.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size of the destination region-of-interest.
  -- * \param nTopBorderHeight Height of top border.
  -- * \param nLeftBorderWidth Width of left border.
  -- * \param aValue Vector of the RGBA values of the border pixels to be set.
  --  * \return \ref image_data_error_codes, \ref roi_error_codes
  -- 

   function nppiCopyConstBorder_8u_C3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      oSrcSizeROI : nppdefs_h.NppiSize;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      nTopBorderHeight : int;
      nLeftBorderWidth : int;
      aValue : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:4472
   pragma Import (C, nppiCopyConstBorder_8u_C3R, "nppiCopyConstBorder_8u_C3R");

  --*
  -- * 4 channel 8-bit unsigned integer image copy with constant border color.
  -- * See nppiCopyConstBorder_8u_C1R() for detailed documentation.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSizeROI Size of the source region-of-interest.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size of the destination region-of-interest.
  -- * \param nTopBorderHeight Height of top border.
  -- * \param nLeftBorderWidth Width of left border.
  -- * \param aValue Vector of the RGBA values of the border pixels to be set.
  --  * \return \ref image_data_error_codes, \ref roi_error_codes
  -- 

   function nppiCopyConstBorder_8u_C4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      oSrcSizeROI : nppdefs_h.NppiSize;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      nTopBorderHeight : int;
      nLeftBorderWidth : int;
      aValue : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:4491
   pragma Import (C, nppiCopyConstBorder_8u_C4R, "nppiCopyConstBorder_8u_C4R");

  --*
  -- * 4 channel 8-bit unsigned integer image copy with constant border color with alpha channel unaffected.
  -- * See nppiCopyConstBorder_8u_C1R() for detailed documentation.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSizeROI Size of the source region-of-interest.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size of the destination region-of-interest.
  -- * \param nTopBorderHeight Height of top border.
  -- * \param nLeftBorderWidth Width of left border.
  -- * \param aValue Vector of the RGB values of the border pixels. Because this method does not
  -- *      affect the destination image's alpha channel, only three components of the border color
  -- *      are needed.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopyConstBorder_8u_AC4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      oSrcSizeROI : nppdefs_h.NppiSize;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      nTopBorderHeight : int;
      nLeftBorderWidth : int;
      aValue : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:4512
   pragma Import (C, nppiCopyConstBorder_8u_AC4R, "nppiCopyConstBorder_8u_AC4R");

  --* 
  -- * 1 channel 16-bit unsigned integer image copy with constant border color.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSizeROI Size of the source region of pixels.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size (width, height) of the destination region, i.e. the region that gets filled with
  -- *      data from the source image (inner part) and constant border color (outer part).
  -- * \param nTopBorderHeight Height (in pixels) of the top border. The number of pixel rows at the top of the
  -- *      destination ROI that will be filled with the constant border color.
  -- *      nBottomBorderHeight = oDstSizeROI.height - nTopBorderHeight - oSrcSizeROI.height.
  -- * \param nLeftBorderWidth Width (in pixels) of the left border. The width of the border at the right side of the
  -- *      destination ROI is implicitly defined by the size of the source ROI:
  -- *      nRightBorderWidth = oDstSizeROI.width - nLeftBorderWidth - oSrcSizeROI.width.
  -- * \param nValue The pixel value to be set for border pixels.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopyConstBorder_16u_C1R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      oSrcSizeROI : nppdefs_h.NppiSize;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      nTopBorderHeight : int;
      nLeftBorderWidth : int;
      nValue : nppdefs_h.Npp16u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:4535
   pragma Import (C, nppiCopyConstBorder_16u_C1R, "nppiCopyConstBorder_16u_C1R");

  --*
  -- * 3 channel 16-bit unsigned integer image copy with constant border color.
  -- * See nppiCopyConstBorder_16u_C1R() for detailed documentation.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSizeROI Size of the source region-of-interest.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size of the destination region-of-interest.
  -- * \param nTopBorderHeight Height of top border.
  -- * \param nLeftBorderWidth Width of left border.
  -- * \param aValue Vector of the RGBA values of the border pixels to be set.
  --  * \return \ref image_data_error_codes, \ref roi_error_codes
  -- 

   function nppiCopyConstBorder_16u_C3R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      oSrcSizeROI : nppdefs_h.NppiSize;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      nTopBorderHeight : int;
      nLeftBorderWidth : int;
      aValue : access nppdefs_h.Npp16u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:4554
   pragma Import (C, nppiCopyConstBorder_16u_C3R, "nppiCopyConstBorder_16u_C3R");

  --*
  -- * 4 channel 16-bit unsigned integer image copy with constant border color.
  -- * See nppiCopyConstBorder_16u_C1R() for detailed documentation.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSizeROI Size of the source region-of-interest.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size of the destination region-of-interest.
  -- * \param nTopBorderHeight Height of top border.
  -- * \param nLeftBorderWidth Width of left border.
  -- * \param aValue Vector of the RGBA values of the border pixels to be set.
  --  * \return \ref image_data_error_codes, \ref roi_error_codes
  -- 

   function nppiCopyConstBorder_16u_C4R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      oSrcSizeROI : nppdefs_h.NppiSize;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      nTopBorderHeight : int;
      nLeftBorderWidth : int;
      aValue : access nppdefs_h.Npp16u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:4573
   pragma Import (C, nppiCopyConstBorder_16u_C4R, "nppiCopyConstBorder_16u_C4R");

  --*
  -- * 4 channel 16-bit unsigned integer image copy with constant border color with alpha channel unaffected.
  -- * See nppiCopyConstBorder_16u_C1R() for detailed documentation.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSizeROI Size of the source region-of-interest.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size of the destination region-of-interest.
  -- * \param nTopBorderHeight Height of top border.
  -- * \param nLeftBorderWidth Width of left border.
  -- * \param aValue Vector of the RGB values of the border pixels. Because this method does not
  -- *      affect the destination image's alpha channel, only three components of the border color
  -- *      are needed.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopyConstBorder_16u_AC4R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      oSrcSizeROI : nppdefs_h.NppiSize;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      nTopBorderHeight : int;
      nLeftBorderWidth : int;
      aValue : access nppdefs_h.Npp16u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:4594
   pragma Import (C, nppiCopyConstBorder_16u_AC4R, "nppiCopyConstBorder_16u_AC4R");

  --* 
  -- * 1 channel 16-bit signed integer image copy with constant border color.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSizeROI Size of the source region of pixels.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size (width, height) of the destination region, i.e. the region that gets filled with
  -- *      data from the source image (inner part) and constant border color (outer part).
  -- * \param nTopBorderHeight Height (in pixels) of the top border. The number of pixel rows at the top of the
  -- *      destination ROI that will be filled with the constant border color.
  -- *      nBottomBorderHeight = oDstSizeROI.height - nTopBorderHeight - oSrcSizeROI.height.
  -- * \param nLeftBorderWidth Width (in pixels) of the left border. The width of the border at the right side of the
  -- *      destination ROI is implicitly defined by the size of the source ROI:
  -- *      nRightBorderWidth = oDstSizeROI.width - nLeftBorderWidth - oSrcSizeROI.width.
  -- * \param nValue The pixel value to be set for border pixels.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopyConstBorder_16s_C1R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      oSrcSizeROI : nppdefs_h.NppiSize;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      nTopBorderHeight : int;
      nLeftBorderWidth : int;
      nValue : nppdefs_h.Npp16s) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:4617
   pragma Import (C, nppiCopyConstBorder_16s_C1R, "nppiCopyConstBorder_16s_C1R");

  --*
  -- * 3 channel 16-bit signed integer image copy with constant border color.
  -- * See nppiCopyConstBorder_16s_C1R() for detailed documentation.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSizeROI Size of the source region-of-interest.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size of the destination region-of-interest.
  -- * \param nTopBorderHeight Height of top border.
  -- * \param nLeftBorderWidth Width of left border.
  -- * \param aValue Vector of the RGBA values of the border pixels to be set.
  --  * \return \ref image_data_error_codes, \ref roi_error_codes
  -- 

   function nppiCopyConstBorder_16s_C3R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      oSrcSizeROI : nppdefs_h.NppiSize;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      nTopBorderHeight : int;
      nLeftBorderWidth : int;
      aValue : access nppdefs_h.Npp16s) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:4636
   pragma Import (C, nppiCopyConstBorder_16s_C3R, "nppiCopyConstBorder_16s_C3R");

  --*
  -- * 4 channel 16-bit signed integer image copy with constant border color.
  -- * See nppiCopyConstBorder_16s_C1R() for detailed documentation.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSizeROI Size of the source region-of-interest.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size of the destination region-of-interest.
  -- * \param nTopBorderHeight Height of top border.
  -- * \param nLeftBorderWidth Width of left border.
  -- * \param aValue Vector of the RGBA values of the border pixels to be set.
  --  * \return \ref image_data_error_codes, \ref roi_error_codes
  -- 

   function nppiCopyConstBorder_16s_C4R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      oSrcSizeROI : nppdefs_h.NppiSize;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      nTopBorderHeight : int;
      nLeftBorderWidth : int;
      aValue : access nppdefs_h.Npp16s) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:4655
   pragma Import (C, nppiCopyConstBorder_16s_C4R, "nppiCopyConstBorder_16s_C4R");

  --*
  -- * 4 channel 16-bit signed integer image copy with constant border color with alpha channel unaffected.
  -- * See nppiCopyConstBorder_16s_C1R() for detailed documentation.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSizeROI Size of the source region-of-interest.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size of the destination region-of-interest.
  -- * \param nTopBorderHeight Height of top border.
  -- * \param nLeftBorderWidth Width of left border.
  -- * \param aValue Vector of the RGB values of the border pixels. Because this method does not
  -- *      affect the destination image's alpha channel, only three components of the border color
  -- *      are needed.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopyConstBorder_16s_AC4R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      oSrcSizeROI : nppdefs_h.NppiSize;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      nTopBorderHeight : int;
      nLeftBorderWidth : int;
      aValue : access nppdefs_h.Npp16s) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:4676
   pragma Import (C, nppiCopyConstBorder_16s_AC4R, "nppiCopyConstBorder_16s_AC4R");

  --* 
  -- * 1 channel 32-bit signed integer image copy with constant border color.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSizeROI Size of the source region of pixels.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size (width, height) of the destination region, i.e. the region that gets filled with
  -- *      data from the source image (inner part) and constant border color (outer part).
  -- * \param nTopBorderHeight Height (in pixels) of the top border. The number of pixel rows at the top of the
  -- *      destination ROI that will be filled with the constant border color.
  -- *      nBottomBorderHeight = oDstSizeROI.height - nTopBorderHeight - oSrcSizeROI.height.
  -- * \param nLeftBorderWidth Width (in pixels) of the left border. The width of the border at the right side of the
  -- *      destination ROI is implicitly defined by the size of the source ROI:
  -- *      nRightBorderWidth = oDstSizeROI.width - nLeftBorderWidth - oSrcSizeROI.width.
  -- * \param nValue The pixel value to be set for border pixels.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopyConstBorder_32s_C1R
     (pSrc : access nppdefs_h.Npp32s;
      nSrcStep : int;
      oSrcSizeROI : nppdefs_h.NppiSize;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      nTopBorderHeight : int;
      nLeftBorderWidth : int;
      nValue : nppdefs_h.Npp32s) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:4699
   pragma Import (C, nppiCopyConstBorder_32s_C1R, "nppiCopyConstBorder_32s_C1R");

  --*
  -- * 3 channel 32-bit signed integer image copy with constant border color.
  -- * See nppiCopyConstBorder_32s_C1R() for detailed documentation.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSizeROI Size of the source region-of-interest.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size of the destination region-of-interest.
  -- * \param nTopBorderHeight Height of top border.
  -- * \param nLeftBorderWidth Width of left border.
  -- * \param aValue Vector of the RGBA values of the border pixels to be set.
  --  * \return \ref image_data_error_codes, \ref roi_error_codes
  -- 

   function nppiCopyConstBorder_32s_C3R
     (pSrc : access nppdefs_h.Npp32s;
      nSrcStep : int;
      oSrcSizeROI : nppdefs_h.NppiSize;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      nTopBorderHeight : int;
      nLeftBorderWidth : int;
      aValue : access nppdefs_h.Npp32s) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:4718
   pragma Import (C, nppiCopyConstBorder_32s_C3R, "nppiCopyConstBorder_32s_C3R");

  --*
  -- * 4 channel 32-bit signed integer image copy with constant border color.
  -- * See nppiCopyConstBorder_32s_C1R() for detailed documentation.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSizeROI Size of the source region-of-interest.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size of the destination region-of-interest.
  -- * \param nTopBorderHeight Height of top border.
  -- * \param nLeftBorderWidth Width of left border.
  -- * \param aValue Vector of the RGBA values of the border pixels to be set.
  --  * \return \ref image_data_error_codes, \ref roi_error_codes
  -- 

   function nppiCopyConstBorder_32s_C4R
     (pSrc : access nppdefs_h.Npp32s;
      nSrcStep : int;
      oSrcSizeROI : nppdefs_h.NppiSize;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      nTopBorderHeight : int;
      nLeftBorderWidth : int;
      aValue : access nppdefs_h.Npp32s) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:4737
   pragma Import (C, nppiCopyConstBorder_32s_C4R, "nppiCopyConstBorder_32s_C4R");

  --*
  -- * 4 channel 32-bit signed integer image copy with constant border color with alpha channel unaffected.
  -- * See nppiCopyConstBorder_32s_C1R() for detailed documentation.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSizeROI Size of the source region-of-interest.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size of the destination region-of-interest.
  -- * \param nTopBorderHeight Height of top border.
  -- * \param nLeftBorderWidth Width of left border.
  -- * \param aValue Vector of the RGB values of the border pixels. Because this method does not
  -- *      affect the destination image's alpha channel, only three components of the border color
  -- *      are needed.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopyConstBorder_32s_AC4R
     (pSrc : access nppdefs_h.Npp32s;
      nSrcStep : int;
      oSrcSizeROI : nppdefs_h.NppiSize;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      nTopBorderHeight : int;
      nLeftBorderWidth : int;
      aValue : access nppdefs_h.Npp32s) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:4758
   pragma Import (C, nppiCopyConstBorder_32s_AC4R, "nppiCopyConstBorder_32s_AC4R");

  --* 
  -- * 1 channel 32-bit floating point image copy with constant border color.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSizeROI Size of the source region of pixels.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size (width, height) of the destination region, i.e. the region that gets filled with
  -- *      data from the source image (inner part) and constant border color (outer part).
  -- * \param nTopBorderHeight Height (in pixels) of the top border. The number of pixel rows at the top of the
  -- *      destination ROI that will be filled with the constant border color.
  -- *      nBottomBorderHeight = oDstSizeROI.height - nTopBorderHeight - oSrcSizeROI.height.
  -- * \param nLeftBorderWidth Width (in pixels) of the left border. The width of the border at the right side of the
  -- *      destination ROI is implicitly defined by the size of the source ROI:
  -- *      nRightBorderWidth = oDstSizeROI.width - nLeftBorderWidth - oSrcSizeROI.width.
  -- * \param nValue The pixel value to be set for border pixels.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopyConstBorder_32f_C1R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      oSrcSizeROI : nppdefs_h.NppiSize;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      nTopBorderHeight : int;
      nLeftBorderWidth : int;
      nValue : nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:4781
   pragma Import (C, nppiCopyConstBorder_32f_C1R, "nppiCopyConstBorder_32f_C1R");

  --*
  -- * 3 channel 32-bit floating point image copy with constant border color.
  -- * See nppiCopyConstBorder_32f_C1R() for detailed documentation.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSizeROI Size of the source region-of-interest.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size of the destination region-of-interest.
  -- * \param nTopBorderHeight Height of top border.
  -- * \param nLeftBorderWidth Width of left border.
  -- * \param aValue Vector of the RGBA values of the border pixels to be set.
  --  * \return \ref image_data_error_codes, \ref roi_error_codes
  -- 

   function nppiCopyConstBorder_32f_C3R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      oSrcSizeROI : nppdefs_h.NppiSize;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      nTopBorderHeight : int;
      nLeftBorderWidth : int;
      aValue : access nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:4800
   pragma Import (C, nppiCopyConstBorder_32f_C3R, "nppiCopyConstBorder_32f_C3R");

  --*
  -- * 4 channel 32-bit floating point image copy with constant border color.
  -- * See nppiCopyConstBorder_32f_C1R() for detailed documentation.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSizeROI Size of the source region-of-interest.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size of the destination region-of-interest.
  -- * \param nTopBorderHeight Height of top border.
  -- * \param nLeftBorderWidth Width of left border.
  -- * \param aValue Vector of the RGBA values of the border pixels to be set.
  --  * \return \ref image_data_error_codes, \ref roi_error_codes
  -- 

   function nppiCopyConstBorder_32f_C4R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      oSrcSizeROI : nppdefs_h.NppiSize;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      nTopBorderHeight : int;
      nLeftBorderWidth : int;
      aValue : access nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:4819
   pragma Import (C, nppiCopyConstBorder_32f_C4R, "nppiCopyConstBorder_32f_C4R");

  --*
  -- * 4 channel 32-bit floating point image copy with constant border color with alpha channel unaffected.
  -- * See nppiCopyConstBorder_32f_C1R() for detailed documentation.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSizeROI Size of the source region-of-interest.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size of the destination region-of-interest.
  -- * \param nTopBorderHeight Height of top border.
  -- * \param nLeftBorderWidth Width of left border.
  -- * \param aValue Vector of the RGB values of the border pixels. Because this method does not
  -- *      affect the destination image's alpha channel, only three components of the border color
  -- *      are needed.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopyConstBorder_32f_AC4R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      oSrcSizeROI : nppdefs_h.NppiSize;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      nTopBorderHeight : int;
      nLeftBorderWidth : int;
      aValue : access nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:4840
   pragma Import (C, nppiCopyConstBorder_32f_AC4R, "nppiCopyConstBorder_32f_AC4R");

  --* @} CopyConstBorder 
  --* @} image_copy_constant_border  
  --* 
  -- * @defgroup image_copy_replicate_border Copy Replicate Border
  -- *
  -- * @{
  -- *
  --  

  --* @name CopyReplicateBorder
  -- * Methods for copying images and padding borders with a replicates of the nearest source image pixel color.
  -- *
  -- * @{
  -- *
  --  

  --* 
  -- * 1 channel 8-bit unsigned integer image copy with nearest source image pixel color.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSizeROI Size of the source region of pixels.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size (width, height) of the destination region, i.e. the region that gets filled with
  -- *      data from the source image (inner part) and nearest source image pixel color (outer part).
  -- * \param nTopBorderHeight Height (in pixels) of the top border. The number of pixel rows at the top of the
  -- *      destination ROI that will be filled with the nearest source image pixel color.
  -- *      nBottomBorderHeight = oDstSizeROI.height - nTopBorderHeight - oSrcSizeROI.height.
  -- * \param nLeftBorderWidth Width (in pixels) of the left border. The width of the border at the right side of the
  -- *      destination ROI is implicitly defined by the size of the source ROI:
  -- *      nRightBorderWidth = oDstSizeROI.width - nLeftBorderWidth - oSrcSizeROI.width.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopyReplicateBorder_8u_C1R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      oSrcSizeROI : nppdefs_h.NppiSize;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      nTopBorderHeight : int;
      nLeftBorderWidth : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:4880
   pragma Import (C, nppiCopyReplicateBorder_8u_C1R, "nppiCopyReplicateBorder_8u_C1R");

  --*
  -- * 3 channel 8-bit unsigned integer image copy with nearest source image pixel color.
  -- * See nppiCopyReplicateBorder_8u_C1R() for detailed documentation.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSizeROI Size of the source region-of-interest.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size of the destination region-of-interest.
  -- * \param nTopBorderHeight Height of top border.
  -- * \param nLeftBorderWidth Width of left border.
  --  * \return \ref image_data_error_codes, \ref roi_error_codes
  -- 

   function nppiCopyReplicateBorder_8u_C3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      oSrcSizeROI : nppdefs_h.NppiSize;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      nTopBorderHeight : int;
      nLeftBorderWidth : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:4897
   pragma Import (C, nppiCopyReplicateBorder_8u_C3R, "nppiCopyReplicateBorder_8u_C3R");

  --*
  -- * 4 channel 8-bit unsigned integer image copy with nearest source image pixel color.
  -- * See nppiCopyReplicateBorder_8u_C1R() for detailed documentation.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSizeROI Size of the source region-of-interest.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size of the destination region-of-interest.
  -- * \param nTopBorderHeight Height of top border.
  -- * \param nLeftBorderWidth Width of left border.
  --  * \return \ref image_data_error_codes, \ref roi_error_codes
  -- 

   function nppiCopyReplicateBorder_8u_C4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      oSrcSizeROI : nppdefs_h.NppiSize;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      nTopBorderHeight : int;
      nLeftBorderWidth : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:4914
   pragma Import (C, nppiCopyReplicateBorder_8u_C4R, "nppiCopyReplicateBorder_8u_C4R");

  --*
  -- * 4 channel 8-bit unsigned integer image copy with nearest source image pixel color with alpha channel unaffected.
  -- * See nppiCopyReplicateBorder_8u_C1R() for detailed documentation.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSizeROI Size of the source region-of-interest.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size of the destination region-of-interest.
  -- * \param nTopBorderHeight Height of top border.
  -- * \param nLeftBorderWidth Width of left border.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopyReplicateBorder_8u_AC4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      oSrcSizeROI : nppdefs_h.NppiSize;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      nTopBorderHeight : int;
      nLeftBorderWidth : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:4931
   pragma Import (C, nppiCopyReplicateBorder_8u_AC4R, "nppiCopyReplicateBorder_8u_AC4R");

  --* 
  -- * 1 channel 16-bit unsigned integer image copy with nearest source image pixel color.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSizeROI Size of the source region of pixels.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size (width, height) of the destination region, i.e. the region that gets filled with
  -- *      data from the source image (inner part) and nearest source image pixel color (outer part).
  -- * \param nTopBorderHeight Height (in pixels) of the top border. The number of pixel rows at the top of the
  -- *      destination ROI that will be filled with the nearest source image pixel color.
  -- *      nBottomBorderHeight = oDstSizeROI.height - nTopBorderHeight - oSrcSizeROI.height.
  -- * \param nLeftBorderWidth Width (in pixels) of the left border. The width of the border at the right side of the
  -- *      destination ROI is implicitly defined by the size of the source ROI:
  -- *      nRightBorderWidth = oDstSizeROI.width - nLeftBorderWidth - oSrcSizeROI.width.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopyReplicateBorder_16u_C1R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      oSrcSizeROI : nppdefs_h.NppiSize;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      nTopBorderHeight : int;
      nLeftBorderWidth : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:4952
   pragma Import (C, nppiCopyReplicateBorder_16u_C1R, "nppiCopyReplicateBorder_16u_C1R");

  --*
  -- * 3 channel 16-bit unsigned integer image copy with nearest source image pixel color.
  -- * See nppiCopyReplicateBorder_16u_C1R() for detailed documentation.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSizeROI Size of the source region-of-interest.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size of the destination region-of-interest.
  -- * \param nTopBorderHeight Height of top border.
  -- * \param nLeftBorderWidth Width of left border.
  --  * \return \ref image_data_error_codes, \ref roi_error_codes
  -- 

   function nppiCopyReplicateBorder_16u_C3R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      oSrcSizeROI : nppdefs_h.NppiSize;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      nTopBorderHeight : int;
      nLeftBorderWidth : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:4969
   pragma Import (C, nppiCopyReplicateBorder_16u_C3R, "nppiCopyReplicateBorder_16u_C3R");

  --*
  -- * 4 channel 16-bit unsigned integer image copy with nearest source image pixel color.
  -- * See nppiCopyReplicateBorder_16u_C1R() for detailed documentation.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSizeROI Size of the source region-of-interest.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size of the destination region-of-interest.
  -- * \param nTopBorderHeight Height of top border.
  -- * \param nLeftBorderWidth Width of left border.
  --  * \return \ref image_data_error_codes, \ref roi_error_codes
  -- 

   function nppiCopyReplicateBorder_16u_C4R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      oSrcSizeROI : nppdefs_h.NppiSize;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      nTopBorderHeight : int;
      nLeftBorderWidth : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:4986
   pragma Import (C, nppiCopyReplicateBorder_16u_C4R, "nppiCopyReplicateBorder_16u_C4R");

  --*
  -- * 4 channel 16-bit unsigned image copy with nearest source image pixel color with alpha channel unaffected.
  -- * See nppiCopyReplicateBorder_16u_C1R() for detailed documentation.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSizeROI Size of the source region-of-interest.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size of the destination region-of-interest.
  -- * \param nTopBorderHeight Height of top border.
  -- * \param nLeftBorderWidth Width of left border.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopyReplicateBorder_16u_AC4R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      oSrcSizeROI : nppdefs_h.NppiSize;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      nTopBorderHeight : int;
      nLeftBorderWidth : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:5003
   pragma Import (C, nppiCopyReplicateBorder_16u_AC4R, "nppiCopyReplicateBorder_16u_AC4R");

  --* 
  -- * 1 channel 16-bit signed integer image copy with nearest source image pixel color.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSizeROI Size of the source region of pixels.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size (width, height) of the destination region, i.e. the region that gets filled with
  -- *      data from the source image (inner part) and nearest source image pixel color (outer part).
  -- * \param nTopBorderHeight Height (in pixels) of the top border. The number of pixel rows at the top of the
  -- *      destination ROI that will be filled with the nearest source image pixel color.
  -- *      nBottomBorderHeight = oDstSizeROI.height - nTopBorderHeight - oSrcSizeROI.height.
  -- * \param nLeftBorderWidth Width (in pixels) of the left border. The width of the border at the right side of the
  -- *      destination ROI is implicitly defined by the size of the source ROI:
  -- *      nRightBorderWidth = oDstSizeROI.width - nLeftBorderWidth - oSrcSizeROI.width.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopyReplicateBorder_16s_C1R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      oSrcSizeROI : nppdefs_h.NppiSize;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      nTopBorderHeight : int;
      nLeftBorderWidth : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:5024
   pragma Import (C, nppiCopyReplicateBorder_16s_C1R, "nppiCopyReplicateBorder_16s_C1R");

  --*
  -- * 3 channel 16-bit signed integer image copy with nearest source image pixel color.
  -- * See nppiCopyReplicateBorder_16s_C1R() for detailed documentation.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSizeROI Size of the source region-of-interest.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size of the destination region-of-interest.
  -- * \param nTopBorderHeight Height of top border.
  -- * \param nLeftBorderWidth Width of left border.
  --  * \return \ref image_data_error_codes, \ref roi_error_codes
  -- 

   function nppiCopyReplicateBorder_16s_C3R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      oSrcSizeROI : nppdefs_h.NppiSize;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      nTopBorderHeight : int;
      nLeftBorderWidth : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:5041
   pragma Import (C, nppiCopyReplicateBorder_16s_C3R, "nppiCopyReplicateBorder_16s_C3R");

  --*
  -- * 4 channel 16-bit signed integer image copy with nearest source image pixel color.
  -- * See nppiCopyReplicateBorder_16s_C1R() for detailed documentation.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSizeROI Size of the source region-of-interest.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size of the destination region-of-interest.
  -- * \param nTopBorderHeight Height of top border.
  -- * \param nLeftBorderWidth Width of left border.
  --  * \return \ref image_data_error_codes, \ref roi_error_codes
  -- 

   function nppiCopyReplicateBorder_16s_C4R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      oSrcSizeROI : nppdefs_h.NppiSize;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      nTopBorderHeight : int;
      nLeftBorderWidth : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:5058
   pragma Import (C, nppiCopyReplicateBorder_16s_C4R, "nppiCopyReplicateBorder_16s_C4R");

  --*
  -- * 4 channel 16-bit signed integer image copy with nearest source image pixel color with alpha channel unaffected.
  -- * See nppiCopyReplicateBorder_16s_C1R() for detailed documentation.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSizeROI Size of the source region-of-interest.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size of the destination region-of-interest.
  -- * \param nTopBorderHeight Height of top border.
  -- * \param nLeftBorderWidth Width of left border.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopyReplicateBorder_16s_AC4R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      oSrcSizeROI : nppdefs_h.NppiSize;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      nTopBorderHeight : int;
      nLeftBorderWidth : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:5075
   pragma Import (C, nppiCopyReplicateBorder_16s_AC4R, "nppiCopyReplicateBorder_16s_AC4R");

  --* 
  -- * 1 channel 32-bit signed integer image copy with nearest source image pixel color.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSizeROI Size of the source region of pixels.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size (width, height) of the destination region, i.e. the region that gets filled with
  -- *      data from the source image (inner part) and nearest source image pixel color (outer part).
  -- * \param nTopBorderHeight Height (in pixels) of the top border. The number of pixel rows at the top of the
  -- *      destination ROI that will be filled with the nearest source image pixel color.
  -- *      nBottomBorderHeight = oDstSizeROI.height - nTopBorderHeight - oSrcSizeROI.height.
  -- * \param nLeftBorderWidth Width (in pixels) of the left border. The width of the border at the right side of the
  -- *      destination ROI is implicitly defined by the size of the source ROI:
  -- *      nRightBorderWidth = oDstSizeROI.width - nLeftBorderWidth - oSrcSizeROI.width.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopyReplicateBorder_32s_C1R
     (pSrc : access nppdefs_h.Npp32s;
      nSrcStep : int;
      oSrcSizeROI : nppdefs_h.NppiSize;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      nTopBorderHeight : int;
      nLeftBorderWidth : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:5096
   pragma Import (C, nppiCopyReplicateBorder_32s_C1R, "nppiCopyReplicateBorder_32s_C1R");

  --*
  -- * 3 channel 32-bit signed image copy with nearest source image pixel color.
  -- * See nppiCopyReplicateBorder_32s_C1R() for detailed documentation.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSizeROI Size of the source region-of-interest.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size of the destination region-of-interest.
  -- * \param nTopBorderHeight Height of top border.
  -- * \param nLeftBorderWidth Width of left border.
  --  * \return \ref image_data_error_codes, \ref roi_error_codes
  -- 

   function nppiCopyReplicateBorder_32s_C3R
     (pSrc : access nppdefs_h.Npp32s;
      nSrcStep : int;
      oSrcSizeROI : nppdefs_h.NppiSize;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      nTopBorderHeight : int;
      nLeftBorderWidth : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:5113
   pragma Import (C, nppiCopyReplicateBorder_32s_C3R, "nppiCopyReplicateBorder_32s_C3R");

  --*
  -- * 4 channel 32-bit signed integer image copy with nearest source image pixel color.
  -- * See nppiCopyReplicateBorder_32s_C1R() for detailed documentation.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSizeROI Size of the source region-of-interest.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size of the destination region-of-interest.
  -- * \param nTopBorderHeight Height of top border.
  -- * \param nLeftBorderWidth Width of left border.
  --  * \return \ref image_data_error_codes, \ref roi_error_codes
  -- 

   function nppiCopyReplicateBorder_32s_C4R
     (pSrc : access nppdefs_h.Npp32s;
      nSrcStep : int;
      oSrcSizeROI : nppdefs_h.NppiSize;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      nTopBorderHeight : int;
      nLeftBorderWidth : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:5130
   pragma Import (C, nppiCopyReplicateBorder_32s_C4R, "nppiCopyReplicateBorder_32s_C4R");

  --*
  -- * 4 channel 32-bit signed integer image copy with nearest source image pixel color with alpha channel unaffected.
  -- * See nppiCopyReplicateBorder_32s_C1R() for detailed documentation.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSizeROI Size of the source region-of-interest.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size of the destination region-of-interest.
  -- * \param nTopBorderHeight Height of top border.
  -- * \param nLeftBorderWidth Width of left border.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopyReplicateBorder_32s_AC4R
     (pSrc : access nppdefs_h.Npp32s;
      nSrcStep : int;
      oSrcSizeROI : nppdefs_h.NppiSize;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      nTopBorderHeight : int;
      nLeftBorderWidth : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:5147
   pragma Import (C, nppiCopyReplicateBorder_32s_AC4R, "nppiCopyReplicateBorder_32s_AC4R");

  --* 
  -- * 1 channel 32-bit floating point image copy with nearest source image pixel color.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSizeROI Size of the source region of pixels.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size (width, height) of the destination region, i.e. the region that gets filled with
  -- *      data from the source image (inner part) and nearest source image pixel color (outer part).
  -- * \param nTopBorderHeight Height (in pixels) of the top border. The number of pixel rows at the top of the
  -- *      destination ROI that will be filled with the nearest source image pixel color.
  -- *      nBottomBorderHeight = oDstSizeROI.height - nTopBorderHeight - oSrcSizeROI.height.
  -- * \param nLeftBorderWidth Width (in pixels) of the left border. The width of the border at the right side of the
  -- *      destination ROI is implicitly defined by the size of the source ROI:
  -- *      nRightBorderWidth = oDstSizeROI.width - nLeftBorderWidth - oSrcSizeROI.width.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopyReplicateBorder_32f_C1R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      oSrcSizeROI : nppdefs_h.NppiSize;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      nTopBorderHeight : int;
      nLeftBorderWidth : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:5168
   pragma Import (C, nppiCopyReplicateBorder_32f_C1R, "nppiCopyReplicateBorder_32f_C1R");

  --*
  -- * 3 channel 32-bit floating point image copy with nearest source image pixel color.
  -- * See nppiCopyReplicateBorder_32f_C1R() for detailed documentation.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSizeROI Size of the source region-of-interest.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size of the destination region-of-interest.
  -- * \param nTopBorderHeight Height of top border.
  -- * \param nLeftBorderWidth Width of left border.
  --  * \return \ref image_data_error_codes, \ref roi_error_codes
  -- 

   function nppiCopyReplicateBorder_32f_C3R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      oSrcSizeROI : nppdefs_h.NppiSize;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      nTopBorderHeight : int;
      nLeftBorderWidth : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:5185
   pragma Import (C, nppiCopyReplicateBorder_32f_C3R, "nppiCopyReplicateBorder_32f_C3R");

  --*
  -- * 4 channel 32-bit floating point image copy with nearest source image pixel color.
  -- * See nppiCopyReplicateBorder_32f_C1R() for detailed documentation.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSizeROI Size of the source region-of-interest.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size of the destination region-of-interest.
  -- * \param nTopBorderHeight Height of top border.
  -- * \param nLeftBorderWidth Width of left border.
  --  * \return \ref image_data_error_codes, \ref roi_error_codes
  -- 

   function nppiCopyReplicateBorder_32f_C4R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      oSrcSizeROI : nppdefs_h.NppiSize;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      nTopBorderHeight : int;
      nLeftBorderWidth : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:5202
   pragma Import (C, nppiCopyReplicateBorder_32f_C4R, "nppiCopyReplicateBorder_32f_C4R");

  --*
  -- * 4 channel 32-bit floating point image copy with nearest source image pixel color with alpha channel unaffected.
  -- * See nppiCopyReplicateBorder_32f_C1R() for detailed documentation.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSizeROI Size of the source region-of-interest.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size of the destination region-of-interest.
  -- * \param nTopBorderHeight Height of top border.
  -- * \param nLeftBorderWidth Width of left border.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopyReplicateBorder_32f_AC4R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      oSrcSizeROI : nppdefs_h.NppiSize;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      nTopBorderHeight : int;
      nLeftBorderWidth : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:5219
   pragma Import (C, nppiCopyReplicateBorder_32f_AC4R, "nppiCopyReplicateBorder_32f_AC4R");

  --* @} CopyReplicateBorder  
  --* @} image_copy_replicate_border  
  --* 
  -- * @defgroup image_copy_wrap_border Copy Wrap Border
  -- *
  -- * @{
  -- *
  --  

  --* @name CopyWrapBorder
  -- * 
  -- * Methods for copying images and padding borders with wrapped replications of the source image pixel colors.
  -- *
  -- * @{
  -- *
  --  

  --* 
  -- * 1 channel 8-bit unsigned integer image copy with the borders wrapped by replication of source image pixel colors.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSizeROI Size of the source region of pixels.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size (width, height) of the destination region, i.e. the region that gets filled with
  -- *      data from the source image (inner part) and a border consisting of wrapped replication of the source image pixel colors (outer part).
  -- * \param nTopBorderHeight Height (in pixels) of the top border. The number of pixel rows at the top of the
  -- *      destination ROI that will be filled with the wrapped replication of the corresponding column of source image pixels colors.
  -- *      nBottomBorderHeight = oDstSizeROI.height - nTopBorderHeight - oSrcSizeROI.height.
  -- * \param nLeftBorderWidth Width (in pixels) of the left border. The width of the border at the right side of the
  -- *      destination ROI is implicitly defined by the size of the source ROI:
  -- *      nRightBorderWidth = oDstSizeROI.width - nLeftBorderWidth - oSrcSizeROI.width.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopyWrapBorder_8u_C1R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      oSrcSizeROI : nppdefs_h.NppiSize;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      nTopBorderHeight : int;
      nLeftBorderWidth : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:5259
   pragma Import (C, nppiCopyWrapBorder_8u_C1R, "nppiCopyWrapBorder_8u_C1R");

  --*
  -- * 3 channel 8-bit unsigned integer image copy with the borders wrapped by replication of source image pixel colors.
  -- * See nppiCopyWrapBorder_8u_C1R() for detailed documentation.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSizeROI Size of the source region-of-interest.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size of the destination region-of-interest.
  -- * \param nTopBorderHeight Height of top border.
  -- * \param nLeftBorderWidth Width of left border.
  --  * \return \ref image_data_error_codes, \ref roi_error_codes
  -- 

   function nppiCopyWrapBorder_8u_C3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      oSrcSizeROI : nppdefs_h.NppiSize;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      nTopBorderHeight : int;
      nLeftBorderWidth : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:5276
   pragma Import (C, nppiCopyWrapBorder_8u_C3R, "nppiCopyWrapBorder_8u_C3R");

  --*
  -- * 4 channel 8-bit unsigned integer image copy with the borders wrapped by replication of source image pixel colors.
  -- * See nppiCopyWrapBorder_8u_C1R() for detailed documentation.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSizeROI Size of the source region-of-interest.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size of the destination region-of-interest.
  -- * \param nTopBorderHeight Height of top border.
  -- * \param nLeftBorderWidth Width of left border.
  --  * \return \ref image_data_error_codes, \ref roi_error_codes
  -- 

   function nppiCopyWrapBorder_8u_C4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      oSrcSizeROI : nppdefs_h.NppiSize;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      nTopBorderHeight : int;
      nLeftBorderWidth : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:5293
   pragma Import (C, nppiCopyWrapBorder_8u_C4R, "nppiCopyWrapBorder_8u_C4R");

  --*
  -- * 4 channel 8-bit unsigned integer image copy with the borders wrapped by replication of source image pixel colors with alpha channel unaffected.
  -- * See nppiCopyWrapBorder_8u_C1R() for detailed documentation.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSizeROI Size of the source region-of-interest.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size of the destination region-of-interest.
  -- * \param nTopBorderHeight Height of top border.
  -- * \param nLeftBorderWidth Width of left border.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopyWrapBorder_8u_AC4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      oSrcSizeROI : nppdefs_h.NppiSize;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      nTopBorderHeight : int;
      nLeftBorderWidth : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:5310
   pragma Import (C, nppiCopyWrapBorder_8u_AC4R, "nppiCopyWrapBorder_8u_AC4R");

  --* 
  -- * 1 channel 16-bit unsigned integer image copy with the borders wrapped by replication of source image pixel colors.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSizeROI Size of the source region of pixels.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size (width, height) of the destination region, i.e. the region that gets filled with
  -- *      data from the source image (inner part) and a border consisting of wrapped replication of the source image pixel colors (outer part).
  -- * \param nTopBorderHeight Height (in pixels) of the top border. The number of pixel rows at the top of the
  -- *      destination ROI that will be filled with the wrapped replication of the corresponding column of source image pixels colors.
  -- *      nBottomBorderHeight = oDstSizeROI.height - nTopBorderHeight - oSrcSizeROI.height.
  -- * \param nLeftBorderWidth Width (in pixels) of the left border. The width of the border at the right side of the
  -- *      destination ROI is implicitly defined by the size of the source ROI:
  -- *      nRightBorderWidth = oDstSizeROI.width - nLeftBorderWidth - oSrcSizeROI.width.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopyWrapBorder_16u_C1R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      oSrcSizeROI : nppdefs_h.NppiSize;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      nTopBorderHeight : int;
      nLeftBorderWidth : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:5331
   pragma Import (C, nppiCopyWrapBorder_16u_C1R, "nppiCopyWrapBorder_16u_C1R");

  --*
  -- * 3 channel 16-bit unsigned integer image copy with the borders wrapped by replication of source image pixel colors.
  -- * See nppiCopyWrapBorder_16u_C1R() for detailed documentation.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSizeROI Size of the source region-of-interest.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size of the destination region-of-interest.
  -- * \param nTopBorderHeight Height of top border.
  -- * \param nLeftBorderWidth Width of left border.
  --  * \return \ref image_data_error_codes, \ref roi_error_codes
  -- 

   function nppiCopyWrapBorder_16u_C3R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      oSrcSizeROI : nppdefs_h.NppiSize;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      nTopBorderHeight : int;
      nLeftBorderWidth : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:5348
   pragma Import (C, nppiCopyWrapBorder_16u_C3R, "nppiCopyWrapBorder_16u_C3R");

  --*
  -- * 4 channel 16-bit unsigned integer image copy with the borders wrapped by replication of source image pixel colors.
  -- * See nppiCopyWrapBorder_16u_C1R() for detailed documentation.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSizeROI Size of the source region-of-interest.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size of the destination region-of-interest.
  -- * \param nTopBorderHeight Height of top border.
  -- * \param nLeftBorderWidth Width of left border.
  --  * \return \ref image_data_error_codes, \ref roi_error_codes
  -- 

   function nppiCopyWrapBorder_16u_C4R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      oSrcSizeROI : nppdefs_h.NppiSize;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      nTopBorderHeight : int;
      nLeftBorderWidth : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:5365
   pragma Import (C, nppiCopyWrapBorder_16u_C4R, "nppiCopyWrapBorder_16u_C4R");

  --*
  -- * 4 channel 16-bit unsigned integer image copy with the borders wrapped by replication of source image pixel colors with alpha channel unaffected.
  -- * See nppiCopyWrapBorder_16u_C1R() for detailed documentation.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSizeROI Size of the source region-of-interest.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size of the destination region-of-interest.
  -- * \param nTopBorderHeight Height of top border.
  -- * \param nLeftBorderWidth Width of left border.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopyWrapBorder_16u_AC4R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      oSrcSizeROI : nppdefs_h.NppiSize;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      nTopBorderHeight : int;
      nLeftBorderWidth : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:5382
   pragma Import (C, nppiCopyWrapBorder_16u_AC4R, "nppiCopyWrapBorder_16u_AC4R");

  --* 
  -- * 1 channel 16-bit signed integer image copy with the borders wrapped by replication of source image pixel colors.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSizeROI Size of the source region of pixels.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size (width, height) of the destination region, i.e. the region that gets filled with
  -- *      data from the source image (inner part) and a border consisting of wrapped replication of the source image pixel colors (outer part).
  -- * \param nTopBorderHeight Height (in pixels) of the top border. The number of pixel rows at the top of the
  -- *      destination ROI that will be filled with the wrapped replication of the corresponding column of source image pixels colors.
  -- *      nBottomBorderHeight = oDstSizeROI.height - nTopBorderHeight - oSrcSizeROI.height.
  -- * \param nLeftBorderWidth Width (in pixels) of the left border. The width of the border at the right side of the
  -- *      destination ROI is implicitly defined by the size of the source ROI:
  -- *      nRightBorderWidth = oDstSizeROI.width - nLeftBorderWidth - oSrcSizeROI.width.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopyWrapBorder_16s_C1R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      oSrcSizeROI : nppdefs_h.NppiSize;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      nTopBorderHeight : int;
      nLeftBorderWidth : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:5403
   pragma Import (C, nppiCopyWrapBorder_16s_C1R, "nppiCopyWrapBorder_16s_C1R");

  --*
  -- * 3 channel 16-bit signed integer image copy with the borders wrapped by replication of source image pixel colors.
  -- * See nppiCopyWrapBorder_16s_C1R() for detailed documentation.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSizeROI Size of the source region-of-interest.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size of the destination region-of-interest.
  -- * \param nTopBorderHeight Height of top border.
  -- * \param nLeftBorderWidth Width of left border.
  --  * \return \ref image_data_error_codes, \ref roi_error_codes
  -- 

   function nppiCopyWrapBorder_16s_C3R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      oSrcSizeROI : nppdefs_h.NppiSize;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      nTopBorderHeight : int;
      nLeftBorderWidth : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:5420
   pragma Import (C, nppiCopyWrapBorder_16s_C3R, "nppiCopyWrapBorder_16s_C3R");

  --*
  -- * 4 channel 16-bit signed integer image copy with the borders wrapped by replication of source image pixel colors.
  -- * See nppiCopyWrapBorder_16s_C1R() for detailed documentation.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSizeROI Size of the source region-of-interest.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size of the destination region-of-interest.
  -- * \param nTopBorderHeight Height of top border.
  -- * \param nLeftBorderWidth Width of left border.
  --  * \return \ref image_data_error_codes, \ref roi_error_codes
  -- 

   function nppiCopyWrapBorder_16s_C4R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      oSrcSizeROI : nppdefs_h.NppiSize;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      nTopBorderHeight : int;
      nLeftBorderWidth : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:5437
   pragma Import (C, nppiCopyWrapBorder_16s_C4R, "nppiCopyWrapBorder_16s_C4R");

  --*
  -- * 4 channel 16-bit signed integer image copy with the borders wrapped by replication of source image pixel colors with alpha channel unaffected.
  -- * See nppiCopyWrapBorder_16s_C1R() for detailed documentation.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSizeROI Size of the source region-of-interest.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size of the destination region-of-interest.
  -- * \param nTopBorderHeight Height of top border.
  -- * \param nLeftBorderWidth Width of left border.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopyWrapBorder_16s_AC4R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      oSrcSizeROI : nppdefs_h.NppiSize;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      nTopBorderHeight : int;
      nLeftBorderWidth : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:5454
   pragma Import (C, nppiCopyWrapBorder_16s_AC4R, "nppiCopyWrapBorder_16s_AC4R");

  --* 
  -- * 1 channel 32-bit signed integer image copy with the borders wrapped by replication of source image pixel colors.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSizeROI Size of the source region of pixels.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size (width, height) of the destination region, i.e. the region that gets filled with
  -- *      data from the source image (inner part) and a border consisting of wrapped replication of the source image pixel colors (outer part).
  -- * \param nTopBorderHeight Height (in pixels) of the top border. The number of pixel rows at the top of the
  -- *      destination ROI that will be filled with the wrapped replication of the corresponding column of source image pixels colors.
  -- *      nBottomBorderHeight = oDstSizeROI.height - nTopBorderHeight - oSrcSizeROI.height.
  -- * \param nLeftBorderWidth Width (in pixels) of the left border. The width of the border at the right side of the
  -- *      destination ROI is implicitly defined by the size of the source ROI:
  -- *      nRightBorderWidth = oDstSizeROI.width - nLeftBorderWidth - oSrcSizeROI.width.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopyWrapBorder_32s_C1R
     (pSrc : access nppdefs_h.Npp32s;
      nSrcStep : int;
      oSrcSizeROI : nppdefs_h.NppiSize;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      nTopBorderHeight : int;
      nLeftBorderWidth : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:5475
   pragma Import (C, nppiCopyWrapBorder_32s_C1R, "nppiCopyWrapBorder_32s_C1R");

  --*
  -- * 3 channel 32-bit signed integer image copy with the borders wrapped by replication of source image pixel colors.
  -- * See nppiCopyWrapBorder_32s_C1R() for detailed documentation.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSizeROI Size of the source region-of-interest.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size of the destination region-of-interest.
  -- * \param nTopBorderHeight Height of top border.
  -- * \param nLeftBorderWidth Width of left border.
  --  * \return \ref image_data_error_codes, \ref roi_error_codes
  -- 

   function nppiCopyWrapBorder_32s_C3R
     (pSrc : access nppdefs_h.Npp32s;
      nSrcStep : int;
      oSrcSizeROI : nppdefs_h.NppiSize;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      nTopBorderHeight : int;
      nLeftBorderWidth : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:5492
   pragma Import (C, nppiCopyWrapBorder_32s_C3R, "nppiCopyWrapBorder_32s_C3R");

  --*
  -- * 4 channel 32-bit signed integer image copy with the borders wrapped by replication of source image pixel colors.
  -- * See nppiCopyWrapBorder_32s_C1R() for detailed documentation.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSizeROI Size of the source region-of-interest.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size of the destination region-of-interest.
  -- * \param nTopBorderHeight Height of top border.
  -- * \param nLeftBorderWidth Width of left border.
  --  * \return \ref image_data_error_codes, \ref roi_error_codes
  -- 

   function nppiCopyWrapBorder_32s_C4R
     (pSrc : access nppdefs_h.Npp32s;
      nSrcStep : int;
      oSrcSizeROI : nppdefs_h.NppiSize;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      nTopBorderHeight : int;
      nLeftBorderWidth : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:5509
   pragma Import (C, nppiCopyWrapBorder_32s_C4R, "nppiCopyWrapBorder_32s_C4R");

  --*
  -- * 4 channel 32-bit signed integer image copy with the borders wrapped by replication of source image pixel colors with alpha channel unaffected.
  -- * See nppiCopyWrapBorder_32s_C1R() for detailed documentation.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSizeROI Size of the source region-of-interest.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size of the destination region-of-interest.
  -- * \param nTopBorderHeight Height of top border.
  -- * \param nLeftBorderWidth Width of left border.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopyWrapBorder_32s_AC4R
     (pSrc : access nppdefs_h.Npp32s;
      nSrcStep : int;
      oSrcSizeROI : nppdefs_h.NppiSize;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      nTopBorderHeight : int;
      nLeftBorderWidth : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:5526
   pragma Import (C, nppiCopyWrapBorder_32s_AC4R, "nppiCopyWrapBorder_32s_AC4R");

  --* 
  -- * 1 channel 32-bit floating point image copy with the borders wrapped by replication of source image pixel colors.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSizeROI Size of the source region of pixels.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size (width, height) of the destination region, i.e. the region that gets filled with
  -- *      data from the source image (inner part) and a border consisting of wrapped replication of the source image pixel colors (outer part).
  -- * \param nTopBorderHeight Height (in pixels) of the top border. The number of pixel rows at the top of the
  -- *      destination ROI that will be filled with the wrapped replication of the corresponding column of source image pixels colors.
  -- *      nBottomBorderHeight = oDstSizeROI.height - nTopBorderHeight - oSrcSizeROI.height.
  -- * \param nLeftBorderWidth Width (in pixels) of the left border. The width of the border at the right side of the
  -- *      destination ROI is implicitly defined by the size of the source ROI:
  -- *      nRightBorderWidth = oDstSizeROI.width - nLeftBorderWidth - oSrcSizeROI.width.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopyWrapBorder_32f_C1R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      oSrcSizeROI : nppdefs_h.NppiSize;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      nTopBorderHeight : int;
      nLeftBorderWidth : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:5547
   pragma Import (C, nppiCopyWrapBorder_32f_C1R, "nppiCopyWrapBorder_32f_C1R");

  --*
  -- * 3 channel 32-bit floating point image copy with the borders wrapped by replication of source image pixel colors.
  -- * See nppiCopyWrapBorder_32f_C1R() for detailed documentation.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSizeROI Size of the source region-of-interest.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size of the destination region-of-interest.
  -- * \param nTopBorderHeight Height of top border.
  -- * \param nLeftBorderWidth Width of left border.
  --  * \return \ref image_data_error_codes, \ref roi_error_codes
  -- 

   function nppiCopyWrapBorder_32f_C3R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      oSrcSizeROI : nppdefs_h.NppiSize;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      nTopBorderHeight : int;
      nLeftBorderWidth : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:5564
   pragma Import (C, nppiCopyWrapBorder_32f_C3R, "nppiCopyWrapBorder_32f_C3R");

  --*
  -- * 4 channel 32-bit floating point image copy with the borders wrapped by replication of source image pixel colors.
  -- * See nppiCopyWrapBorder_32f_C1R() for detailed documentation.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSizeROI Size of the source region-of-interest.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size of the destination region-of-interest.
  -- * \param nTopBorderHeight Height of top border.
  -- * \param nLeftBorderWidth Width of left border.
  --  * \return \ref image_data_error_codes, \ref roi_error_codes
  -- 

   function nppiCopyWrapBorder_32f_C4R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      oSrcSizeROI : nppdefs_h.NppiSize;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      nTopBorderHeight : int;
      nLeftBorderWidth : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:5581
   pragma Import (C, nppiCopyWrapBorder_32f_C4R, "nppiCopyWrapBorder_32f_C4R");

  --*
  -- * 1 channel 32-bit floating point image copy with the borders wrapped by replication of source image pixel colors with alpha channel unaffected.
  -- * See nppiCopyWrapBorder_32f_C1R() for detailed documentation.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param oSrcSizeROI Size of the source region-of-interest.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size of the destination region-of-interest.
  -- * \param nTopBorderHeight Height of top border.
  -- * \param nLeftBorderWidth Width of left border.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopyWrapBorder_32f_AC4R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      oSrcSizeROI : nppdefs_h.NppiSize;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      nTopBorderHeight : int;
      nLeftBorderWidth : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:5598
   pragma Import (C, nppiCopyWrapBorder_32f_AC4R, "nppiCopyWrapBorder_32f_AC4R");

  --* @} CopyWrapBorder  
  --* @} image_copy_wrap_border  
  --* 
  -- * @defgroup image_copy_sub_pixel Copy Sub-Pixel
  -- *
  -- * @{
  -- *
  --  

  --* @name CopySubpix
  -- *
  -- * Functions for copying linearly interpolated images using source image subpixel coordinates
  -- *
  -- * @{
  -- *
  --  

  --* 
  -- * 1 channel 8-bit unsigned integer linearly interpolated source image subpixel coordinate color copy.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size (width, height) of the destination region, i.e. the region that gets filled with
  -- *      data from the source image, source image ROI is assumed to be same as destination image ROI.
  -- * \param nDx Fractional part of source image X coordinate.
  -- * \param nDy Fractional part of source image Y coordinate.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopySubpix_8u_C1R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      nDx : nppdefs_h.Npp32f;
      nDy : nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:5633
   pragma Import (C, nppiCopySubpix_8u_C1R, "nppiCopySubpix_8u_C1R");

  --*
  -- * 3 channel 8-bit unsigned integer linearly interpolated source image subpixel coordinate color copy.
  -- * See nppiCopySubpix_8u_C1R() for detailed documentation.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size of the destination region-of-interest.
  -- * \param nDx Fractional part of source image X coordinate.
  -- * \param nDy Fractional part of source image Y coordinate.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  -- 

   function nppiCopySubpix_8u_C3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      nDx : nppdefs_h.Npp32f;
      nDy : nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:5649
   pragma Import (C, nppiCopySubpix_8u_C3R, "nppiCopySubpix_8u_C3R");

  --*
  -- * 4 channel 8-bit unsigned integer linearly interpolated source image subpixel coordinate color copy.
  -- * See nppiCopySubpix_8u_C1R() for detailed documentation.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size of the destination region-of-interest.
  -- * \param nDx Fractional part of source image X coordinate.
  -- * \param nDy Fractional part of source image Y coordinate.
  --  * \return \ref image_data_error_codes, \ref roi_error_codes
  -- 

   function nppiCopySubpix_8u_C4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      nDx : nppdefs_h.Npp32f;
      nDy : nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:5665
   pragma Import (C, nppiCopySubpix_8u_C4R, "nppiCopySubpix_8u_C4R");

  --*
  -- * 4 channel 8-bit unsigned integer linearly interpolated source image subpixel coordinate color copy with alpha channel unaffected.
  -- * See nppiCopySubpix_8u_C1R() for detailed documentation.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size of the destination region-of-interest.
  -- * \param nDx Fractional part of source image X coordinate.
  -- * \param nDy Fractional part of source image Y coordinate.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopySubpix_8u_AC4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      nDx : nppdefs_h.Npp32f;
      nDy : nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:5681
   pragma Import (C, nppiCopySubpix_8u_AC4R, "nppiCopySubpix_8u_AC4R");

  --* 
  -- * 1 channel 16-bit unsigned integer linearly interpolated source image subpixel coordinate color copy.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size (width, height) of the destination region, i.e. the region that gets filled with
  -- *      data from the source image, source image ROI is assumed to be same as destination image ROI.
  -- * \param nDx Fractional part of source image X coordinate.
  -- * \param nDy Fractional part of source image Y coordinate.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopySubpix_16u_C1R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      nDx : nppdefs_h.Npp32f;
      nDy : nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:5697
   pragma Import (C, nppiCopySubpix_16u_C1R, "nppiCopySubpix_16u_C1R");

  --*
  -- * 3 channel 16-bit unsigned integer linearly interpolated source image subpixel coordinate color copy.
  -- * See nppiCopySubpix_16u_C1R() for detailed documentation.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size of the destination region-of-interest.
  -- * \param nDx Fractional part of source image X coordinate.
  -- * \param nDy Fractional part of source image Y coordinate.
  --  * \return \ref image_data_error_codes, \ref roi_error_codes
  -- 

   function nppiCopySubpix_16u_C3R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      nDx : nppdefs_h.Npp32f;
      nDy : nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:5713
   pragma Import (C, nppiCopySubpix_16u_C3R, "nppiCopySubpix_16u_C3R");

  --*
  -- * 4 channel 16-bit unsigned integer linearly interpolated source image subpixel coordinate color copy.
  -- * See nppiCopySubpix_16u_C1R() for detailed documentation.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size of the destination region-of-interest.
  -- * \param nDx Fractional part of source image X coordinate.
  -- * \param nDy Fractional part of source image Y coordinate.
  --  * \return \ref image_data_error_codes, \ref roi_error_codes
  -- 

   function nppiCopySubpix_16u_C4R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      nDx : nppdefs_h.Npp32f;
      nDy : nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:5729
   pragma Import (C, nppiCopySubpix_16u_C4R, "nppiCopySubpix_16u_C4R");

  --*
  -- * 4 channel 16-bit unsigned linearly interpolated source image subpixel coordinate color copy with alpha channel unaffected.
  -- * See nppiCopySubpix_16u_C1R() for detailed documentation.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size of the destination region-of-interest.
  -- * \param nDx Fractional part of source image X coordinate.
  -- * \param nDy Fractional part of source image Y coordinate.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopySubpix_16u_AC4R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      nDx : nppdefs_h.Npp32f;
      nDy : nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:5745
   pragma Import (C, nppiCopySubpix_16u_AC4R, "nppiCopySubpix_16u_AC4R");

  --* 
  -- * 1 channel 16-bit signed integer linearly interpolated source image subpixel coordinate color copy.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size (width, height) of the destination region, i.e. the region that gets filled with
  -- *      data from the source image, source image ROI is assumed to be same as destination image ROI.
  -- * \param nDx Fractional part of source image X coordinate.
  -- * \param nDy Fractional part of source image Y coordinate.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopySubpix_16s_C1R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      nDx : nppdefs_h.Npp32f;
      nDy : nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:5761
   pragma Import (C, nppiCopySubpix_16s_C1R, "nppiCopySubpix_16s_C1R");

  --*
  -- * 3 channel 16-bit signed integer linearly interpolated source image subpixel coordinate color copy.
  -- * See nppiCopySubpix_16s_C1R() for detailed documentation.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size of the destination region-of-interest.
  -- * \param nDx Fractional part of source image X coordinate.
  -- * \param nDy Fractional part of source image Y coordinate.
  --  * \return \ref image_data_error_codes, \ref roi_error_codes
  -- 

   function nppiCopySubpix_16s_C3R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      nDx : nppdefs_h.Npp32f;
      nDy : nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:5777
   pragma Import (C, nppiCopySubpix_16s_C3R, "nppiCopySubpix_16s_C3R");

  --*
  -- * 4 channel 16-bit signed integer linearly interpolated source image subpixel coordinate color copy.
  -- * See nppiCopySubpix_16s_C1R() for detailed documentation.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size of the destination region-of-interest.
  -- * \param nDx Fractional part of source image X coordinate.
  -- * \param nDy Fractional part of source image Y coordinate.
  --  * \return \ref image_data_error_codes, \ref roi_error_codes
  -- 

   function nppiCopySubpix_16s_C4R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      nDx : nppdefs_h.Npp32f;
      nDy : nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:5793
   pragma Import (C, nppiCopySubpix_16s_C4R, "nppiCopySubpix_16s_C4R");

  --*
  -- * 4 channel 16-bit signed integer linearly interpolated source image subpixel coordinate color copy with alpha channel unaffected.
  -- * See nppiCopySubpix_16s_C1R() for detailed documentation.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size of the destination region-of-interest.
  -- * \param nDx Fractional part of source image X coordinate.
  -- * \param nDy Fractional part of source image Y coordinate.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopySubpix_16s_AC4R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      nDx : nppdefs_h.Npp32f;
      nDy : nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:5809
   pragma Import (C, nppiCopySubpix_16s_AC4R, "nppiCopySubpix_16s_AC4R");

  --* 
  -- * 1 channel 32-bit signed integer linearly interpolated source image subpixel coordinate color copy.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size (width, height) of the destination region, i.e. the region that gets filled with
  -- *      data from the source image, source image ROI is assumed to be same as destination image ROI.
  -- * \param nDx Fractional part of source image X coordinate.
  -- * \param nDy Fractional part of source image Y coordinate.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopySubpix_32s_C1R
     (pSrc : access nppdefs_h.Npp32s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      nDx : nppdefs_h.Npp32f;
      nDy : nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:5825
   pragma Import (C, nppiCopySubpix_32s_C1R, "nppiCopySubpix_32s_C1R");

  --*
  -- * 3 channel 32-bit signed linearly interpolated source image subpixel coordinate color copy.
  -- * See nppiCopySubpix_32s_C1R() for detailed documentation.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size of the destination region-of-interest.
  -- * \param nDx Fractional part of source image X coordinate.
  -- * \param nDy Fractional part of source image Y coordinate.
  --  * \return \ref image_data_error_codes, \ref roi_error_codes
  -- 

   function nppiCopySubpix_32s_C3R
     (pSrc : access nppdefs_h.Npp32s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      nDx : nppdefs_h.Npp32f;
      nDy : nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:5841
   pragma Import (C, nppiCopySubpix_32s_C3R, "nppiCopySubpix_32s_C3R");

  --*
  -- * 4 channel 32-bit signed integer linearly interpolated source image subpixel coordinate color copy.
  -- * See nppiCopySubpix_32s_C1R() for detailed documentation.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size of the destination region-of-interest.
  -- * \param nDx Fractional part of source image X coordinate.
  -- * \param nDy Fractional part of source image Y coordinate.
  --  * \return \ref image_data_error_codes, \ref roi_error_codes
  -- 

   function nppiCopySubpix_32s_C4R
     (pSrc : access nppdefs_h.Npp32s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      nDx : nppdefs_h.Npp32f;
      nDy : nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:5857
   pragma Import (C, nppiCopySubpix_32s_C4R, "nppiCopySubpix_32s_C4R");

  --*
  -- * 4 channel 32-bit signed integer linearly interpolated source image subpixel coordinate color copy with alpha channel unaffected.
  -- * See nppiCopySubpix_32s_C1R() for detailed documentation.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size of the destination region-of-interest.
  -- * \param nDx Fractional part of source image X coordinate.
  -- * \param nDy Fractional part of source image Y coordinate.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopySubpix_32s_AC4R
     (pSrc : access nppdefs_h.Npp32s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      nDx : nppdefs_h.Npp32f;
      nDy : nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:5873
   pragma Import (C, nppiCopySubpix_32s_AC4R, "nppiCopySubpix_32s_AC4R");

  --* 
  -- * 1 channel 32-bit floating point linearly interpolated source image subpixel coordinate color copy.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size (width, height) of the destination region, i.e. the region that gets filled with
  -- *      data from the source image, source image ROI is assumed to be same as destination image ROI.
  -- * \param nDx Fractional part of source image X coordinate.
  -- * \param nDy Fractional part of source image Y coordinate.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopySubpix_32f_C1R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      nDx : nppdefs_h.Npp32f;
      nDy : nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:5889
   pragma Import (C, nppiCopySubpix_32f_C1R, "nppiCopySubpix_32f_C1R");

  --*
  -- * 3 channel 32-bit floating point linearly interpolated source image subpixel coordinate color copy.
  -- * See nppiCopySubpix_32f_C1R() for detailed documentation.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size of the destination region-of-interest.
  -- * \param nDx Fractional part of source image X coordinate.
  -- * \param nDy Fractional part of source image Y coordinate.
  --  * \return \ref image_data_error_codes, \ref roi_error_codes
  -- 

   function nppiCopySubpix_32f_C3R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      nDx : nppdefs_h.Npp32f;
      nDy : nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:5905
   pragma Import (C, nppiCopySubpix_32f_C3R, "nppiCopySubpix_32f_C3R");

  --*
  -- * 4 channel 32-bit floating point linearly interpolated source image subpixel coordinate color copy.
  -- * See nppiCopySubpix_32f_C1R() for detailed documentation.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size of the destination region-of-interest.
  -- * \param nDx Fractional part of source image X coordinate.
  -- * \param nDy Fractional part of source image Y coordinate.
  --  * \return \ref image_data_error_codes, \ref roi_error_codes
  -- 

   function nppiCopySubpix_32f_C4R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      nDx : nppdefs_h.Npp32f;
      nDy : nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:5921
   pragma Import (C, nppiCopySubpix_32f_C4R, "nppiCopySubpix_32f_C4R");

  --*
  -- * 4 channel 32-bit floating point linearly interpolated source image subpixel coordinate color copy with alpha channel unaffected.
  -- * See nppiCopySubpix_32f_C1R() for detailed documentation.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size of the destination region-of-interest.
  -- * \param nDx Fractional part of source image X coordinate.
  -- * \param nDy Fractional part of source image Y coordinate.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiCopySubpix_32f_AC4R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize;
      nDx : nppdefs_h.Npp32f;
      nDy : nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:5937
   pragma Import (C, nppiCopySubpix_32f_AC4R, "nppiCopySubpix_32f_AC4R");

  --* @} CopySubpix  
  --* @} image_copy_subpix  
  --* 
  -- * @defgroup image_duplicate_channel Duplicate Channel
  -- *
  -- * @{
  -- *
  --  

  --* @name Dup
  -- * 
  -- * Functions for duplicating a single channel image in a multiple channel image.
  -- *
  -- * @{
  -- *
  --  

  --* 
  -- * 1 channel 8-bit unsigned integer source image duplicated in all 3 channels of destination image.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size (width, height) of the destination region, i.e. the region that gets filled with
  -- *      data from the source image, source image ROI is assumed to be same as destination image ROI.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDup_8u_C1C3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:5970
   pragma Import (C, nppiDup_8u_C1C3R, "nppiDup_8u_C1C3R");

  --*
  -- * 1 channel 8-bit unsigned integer source image duplicated in all 4 channels of destination image.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size of the destination region-of-interest.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDup_8u_C1C4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:5982
   pragma Import (C, nppiDup_8u_C1C4R, "nppiDup_8u_C1C4R");

  --*
  -- * 1 channel 8-bit unsigned integer source image duplicated in 3 channels of 4 channel destination image with alpha channel unaffected.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size of the destination region-of-interest.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDup_8u_C1AC4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:5994
   pragma Import (C, nppiDup_8u_C1AC4R, "nppiDup_8u_C1AC4R");

  --* 
  -- * 1 channel 16-bit unsigned integer source image duplicated in all 3 channels of destination image.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size (width, height) of the destination region, i.e. the region that gets filled with
  -- *      data from the source image, source image ROI is assumed to be same as destination image ROI.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDup_16u_C1C3R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:6007
   pragma Import (C, nppiDup_16u_C1C3R, "nppiDup_16u_C1C3R");

  --*
  -- * 1 channel 16-bit unsigned integer source image duplicated in all 4 channels of destination image.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size of the destination region-of-interest.
  --  * \return \ref image_data_error_codes, \ref roi_error_codes
  -- 

   function nppiDup_16u_C1C4R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:6019
   pragma Import (C, nppiDup_16u_C1C4R, "nppiDup_16u_C1C4R");

  --*
  -- * 1 channel 16-bit unsigned integer source image duplicated in 3 channels of 4 channel destination image with alpha channel unaffected.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size of the destination region-of-interest.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDup_16u_C1AC4R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:6031
   pragma Import (C, nppiDup_16u_C1AC4R, "nppiDup_16u_C1AC4R");

  --* 
  -- * 1 channel 16-bit signed integer source image duplicated in all 3 channels of destination image.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size (width, height) of the destination region, i.e. the region that gets filled with
  -- *      data from the source image, source image ROI is assumed to be same as destination image ROI.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDup_16s_C1C3R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:6044
   pragma Import (C, nppiDup_16s_C1C3R, "nppiDup_16s_C1C3R");

  --*
  -- * 1 channel 16-bit signed integer source image duplicated in all 4 channels of destination image.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size of the destination region-of-interest.
  --  * \return \ref image_data_error_codes, \ref roi_error_codes
  -- 

   function nppiDup_16s_C1C4R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:6056
   pragma Import (C, nppiDup_16s_C1C4R, "nppiDup_16s_C1C4R");

  --*
  -- * 1 channel 16-bit signed integer source image duplicated in 3 channels of 4 channel destination image with alpha channel unaffected.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size of the destination region-of-interest.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDup_16s_C1AC4R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:6068
   pragma Import (C, nppiDup_16s_C1AC4R, "nppiDup_16s_C1AC4R");

  --* 
  -- * 1 channel 32-bit signed integer source image duplicated in all 3 channels of destination image.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size (width, height) of the destination region, i.e. the region that gets filled with
  -- *      data from the source image, source image ROI is assumed to be same as destination image ROI.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDup_32s_C1C3R
     (pSrc : access nppdefs_h.Npp32s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:6081
   pragma Import (C, nppiDup_32s_C1C3R, "nppiDup_32s_C1C3R");

  --*
  -- * 1 channel 32-bit signed integer source image duplicated in all 4 channels of destination image.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size of the destination region-of-interest.
  --  * \return \ref image_data_error_codes, \ref roi_error_codes
  -- 

   function nppiDup_32s_C1C4R
     (pSrc : access nppdefs_h.Npp32s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:6093
   pragma Import (C, nppiDup_32s_C1C4R, "nppiDup_32s_C1C4R");

  --*
  -- * 1 channel 32-bit signed integer source image duplicated in 3 channels of 4 channel destination image with alpha channel unaffected.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size of the destination region-of-interest.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDup_32s_C1AC4R
     (pSrc : access nppdefs_h.Npp32s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:6105
   pragma Import (C, nppiDup_32s_C1AC4R, "nppiDup_32s_C1AC4R");

  --* 
  -- * 1 channel 32-bit floating point source image duplicated in all 3 channels of destination image.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size (width, height) of the destination region, i.e. the region that gets filled with
  -- *      data from the source image, source image ROI is assumed to be same as destination image ROI.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDup_32f_C1C3R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:6118
   pragma Import (C, nppiDup_32f_C1C3R, "nppiDup_32f_C1C3R");

  --*
  -- * 1 channel 32-bit floating point source image duplicated in all 4 channels of destination image.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size of the destination region-of-interest.
  --  * \return \ref image_data_error_codes, \ref roi_error_codes
  -- 

   function nppiDup_32f_C1C4R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:6130
   pragma Import (C, nppiDup_32f_C1C4R, "nppiDup_32f_C1C4R");

  --*
  -- * 1 channel 32-bit floating point source image duplicated in 3 channels of 4 channel destination image with alpha channel unaffected.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oDstSizeROI Size of the destination region-of-interest.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDup_32f_C1AC4R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oDstSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:6142
   pragma Import (C, nppiDup_32f_C1AC4R, "nppiDup_32f_C1AC4R");

  --* @} Dup  
  --* @} image_duplicate_channel  
  --* 
  -- * @defgroup image_transpose Transpose 
  -- * 
  -- * @{
  -- *
  --  

  --* @name Transpose
  -- * Methods for transposing images of various types. Like matrix transpose, image transpose is a mirror along the image's
  -- * diagonal (upper-left to lower-right corner).
  -- *
  -- * @{
  -- *
  --  

  --*
  -- * 1 channel 8-bit unsigned int image transpose.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst Pointer to the destination ROI.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSrcROI \ref roi_specification.
  -- *
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiTranspose_8u_C1R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSrcROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:6176
   pragma Import (C, nppiTranspose_8u_C1R, "nppiTranspose_8u_C1R");

  --*
  -- * 3 channel 8-bit unsigned int image transpose.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst Pointer to the destination ROI.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSrcROI \ref roi_specification.
  -- *
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiTranspose_8u_C3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSrcROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:6190
   pragma Import (C, nppiTranspose_8u_C3R, "nppiTranspose_8u_C3R");

  --*
  -- * 4 channel 8-bit unsigned int image transpose.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst Pointer to the destination ROI.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSrcROI \ref roi_specification.
  -- *
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiTranspose_8u_C4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSrcROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:6204
   pragma Import (C, nppiTranspose_8u_C4R, "nppiTranspose_8u_C4R");

  --*
  -- * 1 channel 16-bit unsigned int image transpose.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst Pointer to the destination ROI.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSrcROI \ref roi_specification.
  -- *
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiTranspose_16u_C1R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSrcROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:6218
   pragma Import (C, nppiTranspose_16u_C1R, "nppiTranspose_16u_C1R");

  --*
  -- * 3 channel 16-bit unsigned int image transpose.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst Pointer to the destination ROI.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSrcROI \ref roi_specification.
  -- *
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiTranspose_16u_C3R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSrcROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:6232
   pragma Import (C, nppiTranspose_16u_C3R, "nppiTranspose_16u_C3R");

  --*
  -- * 4 channel 16-bit unsigned int image transpose.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst Pointer to the destination ROI.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSrcROI \ref roi_specification.
  -- *
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiTranspose_16u_C4R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSrcROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:6246
   pragma Import (C, nppiTranspose_16u_C4R, "nppiTranspose_16u_C4R");

  --*
  -- * 1 channel 16-bit signed int image transpose.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst Pointer to the destination ROI.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSrcROI \ref roi_specification.
  -- *
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiTranspose_16s_C1R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSrcROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:6260
   pragma Import (C, nppiTranspose_16s_C1R, "nppiTranspose_16s_C1R");

  --*
  -- * 3 channel 16-bit signed int image transpose.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst Pointer to the destination ROI.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSrcROI \ref roi_specification.
  -- *
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiTranspose_16s_C3R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSrcROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:6274
   pragma Import (C, nppiTranspose_16s_C3R, "nppiTranspose_16s_C3R");

  --*
  -- * 4 channel 16-bit signed int image transpose.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst Pointer to the destination ROI.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSrcROI \ref roi_specification.
  -- *
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiTranspose_16s_C4R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSrcROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:6288
   pragma Import (C, nppiTranspose_16s_C4R, "nppiTranspose_16s_C4R");

  --*
  -- * 1 channel 32-bit signed int image transpose.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst Pointer to the destination ROI.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSrcROI \ref roi_specification.
  -- *
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiTranspose_32s_C1R
     (pSrc : access nppdefs_h.Npp32s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSrcROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:6302
   pragma Import (C, nppiTranspose_32s_C1R, "nppiTranspose_32s_C1R");

  --*
  -- * 3 channel 32-bit signed int image transpose.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst Pointer to the destination ROI.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSrcROI \ref roi_specification.
  -- *
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiTranspose_32s_C3R
     (pSrc : access nppdefs_h.Npp32s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSrcROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:6316
   pragma Import (C, nppiTranspose_32s_C3R, "nppiTranspose_32s_C3R");

  --*
  -- * 4 channel 32-bit signed int image transpose.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst Pointer to the destination ROI.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSrcROI \ref roi_specification.
  -- *
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiTranspose_32s_C4R
     (pSrc : access nppdefs_h.Npp32s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSrcROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:6330
   pragma Import (C, nppiTranspose_32s_C4R, "nppiTranspose_32s_C4R");

  --*
  -- * 1 channel 32-bit floating point image transpose.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst Pointer to the destination ROI.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSrcROI \ref roi_specification.
  -- *
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiTranspose_32f_C1R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSrcROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:6344
   pragma Import (C, nppiTranspose_32f_C1R, "nppiTranspose_32f_C1R");

  --*
  -- * 3 channel 32-bit floating point image transpose.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst Pointer to the destination ROI.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSrcROI \ref roi_specification.
  -- *
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiTranspose_32f_C3R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSrcROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:6358
   pragma Import (C, nppiTranspose_32f_C3R, "nppiTranspose_32f_C3R");

  --*
  -- * 4 channel 32-bit floating point image transpose.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst Pointer to the destination ROI.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSrcROI \ref roi_specification.
  -- *
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiTranspose_32f_C4R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSrcROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:6372
   pragma Import (C, nppiTranspose_32f_C4R, "nppiTranspose_32f_C4R");

  --* @} Transpose  
  --* @} image_transpose  
  --* 
  -- * @defgroup image_swap_channels Swap Channels
  -- *
  -- * @{
  -- *
  --  

  --* @name SwapChannels
  -- * 
  -- * Functions for swapping and duplicating channels in multiple channel images. 
  -- * The methods support arbitrary permutations of the original channels, including replication and
  -- * setting one or more channels to a constant value.
  -- *
  -- * @{
  -- *
  --  

  --* 
  -- * 3 channel 8-bit unsigned integer source image to 3 channel destination image.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
  -- *      of the array contains the number of the channel that is stored in the n-th channel of
  -- *      the output image. E.g. Given an RGB image, aDstOrder = [2,1,0] converts this to BGR
  -- *      channel order.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSwapChannels_8u_C3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aDstOrder : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:6409
   pragma Import (C, nppiSwapChannels_8u_C3R, "nppiSwapChannels_8u_C3R");

  --* 
  -- * 3 channel 8-bit unsigned integer in place image.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
  -- *      of the array contains the number of the channel that is stored in the n-th channel of
  -- *      the output image. E.g. Given an RGB image, aDstOrder = [2,1,0] converts this to BGR
  -- *      channel order.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSwapChannels_8u_C3IR
     (pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aDstOrder : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:6423
   pragma Import (C, nppiSwapChannels_8u_C3IR, "nppiSwapChannels_8u_C3IR");

  --* 
  -- * 4 channel 8-bit unsigned integer source image to 3 channel destination image.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
  -- *      of the array contains the number of the channel that is stored in the n-th channel of
  -- *      the output image. E.g. Given an RGBA image, aDstOrder = [2,1,0] converts this to a 3 channel BGR
  -- *      channel order.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSwapChannels_8u_C4C3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aDstOrder : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:6438
   pragma Import (C, nppiSwapChannels_8u_C4C3R, "nppiSwapChannels_8u_C4C3R");

  --*
  -- * 4 channel 8-bit unsigned integer source image to 4 channel destination image.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
  -- *      of the array contains the number of the channel that is stored in the n-th channel of
  -- *      the output image. E.g. Given an ARGB image, aDstOrder = [3,2,1,0] converts this to BGRA
  -- *      channel order.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSwapChannels_8u_C4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aDstOrder : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:6454
   pragma Import (C, nppiSwapChannels_8u_C4R, "nppiSwapChannels_8u_C4R");

  --* 
  -- * 4 channel 8-bit unsigned integer in place image.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
  -- *      of the array contains the number of the channel that is stored in the n-th channel of
  -- *      the output image. E.g. Given an ARGB image, aDstOrder = [3,2,1,0] converts this to BGRA
  -- *      channel order.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSwapChannels_8u_C4IR
     (pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aDstOrder : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:6468
   pragma Import (C, nppiSwapChannels_8u_C4IR, "nppiSwapChannels_8u_C4IR");

  --* 
  -- * 3 channel 8-bit unsigned integer source image to 4 channel destination image.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
  -- *      of the array contains the number of the channel that is stored in the n-th channel of
  -- *      the output image. E.g. Given an RGB image, aDstOrder = [3,2,1,0] converts this to VBGR
  -- *      channel order.
  -- * \param nValue (V) Single channel constant value that can be replicated in one or more of the 4 destination channels.
  -- *      nValue is either written or not written to a particular channel depending on the aDstOrder entry for that destination
  -- *      channel. An aDstOrder value of 3 will output nValue to that channel, an aDstOrder value greater than 3 will leave that
  -- *      particular destination channel value unmodified.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSwapChannels_8u_C3C4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aDstOrder : access int;
      nValue : nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:6487
   pragma Import (C, nppiSwapChannels_8u_C3C4R, "nppiSwapChannels_8u_C3C4R");

  --*
  -- * 4 channel 8-bit unsigned integer source image to 4 channel destination image with destination alpha channel unaffected.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
  -- *      of the array contains the number of the channel that is stored in the n-th channel of
  -- *      the output image. E.g. Given an RGB image, aDstOrder = [3,2,1,0] converts this to VBGR
  -- *      channel order.
  -- *      of the array contains the number of the channel that is stored in the n-th channel of
  -- *      the output image. E.g. Given an RGBA image, aDstOrder = [2,1,0] converts this to BGRA
  -- *      channel order. In the AC4R case, the alpha channel is always assumed to be channel 3.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSwapChannels_8u_AC4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aDstOrder : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:6506
   pragma Import (C, nppiSwapChannels_8u_AC4R, "nppiSwapChannels_8u_AC4R");

  --* 
  -- * 3 channel 16-bit unsigned integer source image to 3 channel destination image.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
  -- *      of the array contains the number of the channel that is stored in the n-th channel of
  -- *      the output image. E.g. Given an RGB image, aDstOrder = [2,1,0] converts this to BGR
  -- *      channel order.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSwapChannels_16u_C3R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aDstOrder : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:6522
   pragma Import (C, nppiSwapChannels_16u_C3R, "nppiSwapChannels_16u_C3R");

  --* 
  -- * 3 channel 16-bit unsigned integer in place image.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
  -- *      of the array contains the number of the channel that is stored in the n-th channel of
  -- *      the output image. E.g. Given an RGB image, aDstOrder = [2,1,0] converts this to BGR
  -- *      channel order.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSwapChannels_16u_C3IR
     (pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aDstOrder : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:6536
   pragma Import (C, nppiSwapChannels_16u_C3IR, "nppiSwapChannels_16u_C3IR");

  --* 
  -- * 4 channel 16-bit unsigned integer source image to 3 channel destination image.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
  -- *      of the array contains the number of the channel that is stored in the n-th channel of
  -- *      the output image. E.g. Given an RGBA image, aDstOrder = [2,1,0] converts this to a 3 channel BGR
  -- *      channel order.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSwapChannels_16u_C4C3R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aDstOrder : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:6551
   pragma Import (C, nppiSwapChannels_16u_C4C3R, "nppiSwapChannels_16u_C4C3R");

  --*
  -- * 4 channel 16-bit unsigned integer source image to 4 channel destination image.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
  -- *      of the array contains the number of the channel that is stored in the n-th channel of
  -- *      the output image. E.g. Given an ARGB image, aDstOrder = [3,2,1,0] converts this to BGRA
  -- *      channel order.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSwapChannels_16u_C4R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aDstOrder : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:6567
   pragma Import (C, nppiSwapChannels_16u_C4R, "nppiSwapChannels_16u_C4R");

  --* 
  -- * 4 channel 16-bit unsigned integer in place image.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
  -- *      of the array contains the number of the channel that is stored in the n-th channel of
  -- *      the output image. E.g. Given an ARGB image, aDstOrder = [3,2,1,0] converts this to BGRA
  -- *      channel order.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSwapChannels_16u_C4IR
     (pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aDstOrder : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:6581
   pragma Import (C, nppiSwapChannels_16u_C4IR, "nppiSwapChannels_16u_C4IR");

  --* 
  -- * 3 channel 16-bit unsigned integer source image to 4 channel destination image.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
  -- *      of the array contains the number of the channel that is stored in the n-th channel of
  -- *      the output image. E.g. Given an RGB image, aDstOrder = [3,2,1,0] converts this to VBGR
  -- *      channel order.
  -- * \param nValue (V) Single channel constant value that can be replicated in one or more of the 4 destination channels.
  -- *      nValue is either written or not written to a particular channel depending on the aDstOrder entry for that destination
  -- *      channel. An aDstOrder value of 3 will output nValue to that channel, an aDstOrder value greater than 3 will leave that
  -- *      particular destination channel value unmodified.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSwapChannels_16u_C3C4R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aDstOrder : access int;
      nValue : nppdefs_h.Npp16u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:6600
   pragma Import (C, nppiSwapChannels_16u_C3C4R, "nppiSwapChannels_16u_C3C4R");

  --*
  -- * 4 channel 16-bit unsigned integer source image to 4 channel destination image with destination alpha channel unaffected.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
  -- *      of the array contains the number of the channel that is stored in the n-th channel of
  -- *      the output image. E.g. Given an RGBA image, aDstOrder = [2,1,0] converts this to BGRA
  -- *      channel order. In the AC4R case, the alpha channel is always assumed to be channel 3.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSwapChannels_16u_AC4R
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aDstOrder : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:6616
   pragma Import (C, nppiSwapChannels_16u_AC4R, "nppiSwapChannels_16u_AC4R");

  --* 
  -- * 3 channel 16-bit signed integer source image to 3 channel destination image.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
  -- *      of the array contains the number of the channel that is stored in the n-th channel of
  -- *      the output image. E.g. Given an RGB image, aDstOrder = [2,1,0] converts this to BGR
  -- *      channel order.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSwapChannels_16s_C3R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aDstOrder : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:6632
   pragma Import (C, nppiSwapChannels_16s_C3R, "nppiSwapChannels_16s_C3R");

  --* 
  -- * 3 channel 16-bit signed integer in place image.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
  -- *      of the array contains the number of the channel that is stored in the n-th channel of
  -- *      the output image. E.g. Given an RGB image, aDstOrder = [2,1,0] converts this to BGR
  -- *      channel order.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSwapChannels_16s_C3IR
     (pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aDstOrder : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:6646
   pragma Import (C, nppiSwapChannels_16s_C3IR, "nppiSwapChannels_16s_C3IR");

  --* 
  -- * 4 channel 16-bit signed integer source image to 3 channel destination image.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
  -- *      of the array contains the number of the channel that is stored in the n-th channel of
  -- *      the output image. E.g. Given an RGBA image, aDstOrder = [2,1,0] converts this to a 3 channel BGR
  -- *      channel order.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSwapChannels_16s_C4C3R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aDstOrder : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:6661
   pragma Import (C, nppiSwapChannels_16s_C4C3R, "nppiSwapChannels_16s_C4C3R");

  --*
  -- * 4 channel 16-bit signed integer source image to 4 channel destination image.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
  -- *      of the array contains the number of the channel that is stored in the n-th channel of
  -- *      the output image. E.g. Given an ARGB image, aDstOrder = [3,2,1,0] converts this to BGRA
  -- *      channel order.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSwapChannels_16s_C4R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aDstOrder : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:6677
   pragma Import (C, nppiSwapChannels_16s_C4R, "nppiSwapChannels_16s_C4R");

  --* 
  -- * 4 channel 16-bit signed integer in place image.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
  -- *      of the array contains the number of the channel that is stored in the n-th channel of
  -- *      the output image. E.g. Given an ARGB image, aDstOrder = [3,2,1,0] converts this to BGRA
  -- *      channel order.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSwapChannels_16s_C4IR
     (pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aDstOrder : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:6691
   pragma Import (C, nppiSwapChannels_16s_C4IR, "nppiSwapChannels_16s_C4IR");

  --* 
  -- * 3 channel 16-bit signed integer source image to 4 channel destination image.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
  -- *      of the array contains the number of the channel that is stored in the n-th channel of
  -- *      the output image. E.g. Given an RGB image, aDstOrder = [3,2,1,0] converts this to VBGR
  -- *      channel order.
  -- * \param nValue (V) Single channel constant value that can be replicated in one or more of the 4 destination channels.
  -- *      nValue is either written or not written to a particular channel depending on the aDstOrder entry for that destination
  -- *      channel. An aDstOrder value of 3 will output nValue to that channel, an aDstOrder value greater than 3 will leave that
  -- *      particular destination channel value unmodified.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSwapChannels_16s_C3C4R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aDstOrder : access int;
      nValue : nppdefs_h.Npp16s) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:6710
   pragma Import (C, nppiSwapChannels_16s_C3C4R, "nppiSwapChannels_16s_C3C4R");

  --*
  -- * 4 channel 16-bit signed integer source image to 4 channel destination image with destination alpha channel unaffected.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
  -- *      of the array contains the number of the channel that is stored in the n-th channel of
  -- *      the output image. E.g. Given an RGBA image, aDstOrder = [2,1,0] converts this to BGRA
  -- *      channel order. In the AC4R case, the alpha channel is always assumed to be channel 3.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSwapChannels_16s_AC4R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aDstOrder : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:6726
   pragma Import (C, nppiSwapChannels_16s_AC4R, "nppiSwapChannels_16s_AC4R");

  --* 
  -- * 3 channel 32-bit signed integer source image to 3 channel destination image.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
  -- *      of the array contains the number of the channel that is stored in the n-th channel of
  -- *      the output image. E.g. Given an RGB image, aDstOrder = [2,1,0] converts this to BGR
  -- *      channel order.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSwapChannels_32s_C3R
     (pSrc : access nppdefs_h.Npp32s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aDstOrder : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:6742
   pragma Import (C, nppiSwapChannels_32s_C3R, "nppiSwapChannels_32s_C3R");

  --* 
  -- * 3 channel 32-bit signed integer in place image.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
  -- *      of the array contains the number of the channel that is stored in the n-th channel of
  -- *      the output image. E.g. Given an RGB image, aDstOrder = [2,1,0] converts this to BGR
  -- *      channel order.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSwapChannels_32s_C3IR
     (pSrcDst : access nppdefs_h.Npp32s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aDstOrder : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:6756
   pragma Import (C, nppiSwapChannels_32s_C3IR, "nppiSwapChannels_32s_C3IR");

  --* 
  -- * 4 channel 32-bit signed integer source image to 3 channel destination image.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
  -- *      of the array contains the number of the channel that is stored in the n-th channel of
  -- *      the output image. E.g. Given an RGBA image, aDstOrder = [2,1,0] converts this to a 3 channel BGR
  -- *      channel order.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSwapChannels_32s_C4C3R
     (pSrc : access nppdefs_h.Npp32s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aDstOrder : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:6771
   pragma Import (C, nppiSwapChannels_32s_C4C3R, "nppiSwapChannels_32s_C4C3R");

  --*
  -- * 4 channel 32-bit signed integer source image to 4 channel destination image.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
  -- *      of the array contains the number of the channel that is stored in the n-th channel of
  -- *      the output image. E.g. Given an ARGB image, aDstOrder = [3,2,1,0] converts this to BGRA
  -- *      channel order.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSwapChannels_32s_C4R
     (pSrc : access nppdefs_h.Npp32s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aDstOrder : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:6787
   pragma Import (C, nppiSwapChannels_32s_C4R, "nppiSwapChannels_32s_C4R");

  --* 
  -- * 4 channel 32-bit signed integer in place image.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
  -- *      of the array contains the number of the channel that is stored in the n-th channel of
  -- *      the output image. E.g. Given an ARGB image, aDstOrder = [3,2,1,0] converts this to BGRA
  -- *      channel order.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSwapChannels_32s_C4IR
     (pSrcDst : access nppdefs_h.Npp32s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aDstOrder : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:6801
   pragma Import (C, nppiSwapChannels_32s_C4IR, "nppiSwapChannels_32s_C4IR");

  --* 
  -- * 3 channel 32-bit signed integer source image to 4 channel destination image.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
  -- *      of the array contains the number of the channel that is stored in the n-th channel of
  -- *      the output image. E.g. Given an RGB image, aDstOrder = [3,2,1,0] converts this to VBGR
  -- *      channel order.
  -- * \param nValue (V) Single channel constant value that can be replicated in one or more of the 4 destination channels.
  -- *      nValue is either written or not written to a particular channel depending on the aDstOrder entry for that destination
  -- *      channel. An aDstOrder value of 3 will output nValue to that channel, an aDstOrder value greater than 3 will leave that
  -- *      particular destination channel value unmodified.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSwapChannels_32s_C3C4R
     (pSrc : access nppdefs_h.Npp32s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aDstOrder : access int;
      nValue : nppdefs_h.Npp32s) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:6820
   pragma Import (C, nppiSwapChannels_32s_C3C4R, "nppiSwapChannels_32s_C3C4R");

  --*
  -- * 4 channel 32-bit signed integer source image to 4 channel destination image with destination alpha channel unaffected.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
  -- *      of the array contains the number of the channel that is stored in the n-th channel of
  -- *      the output image. E.g. Given an RGBA image, aDstOrder = [2,1,0] converts this to BGRA
  -- *      channel order. In the AC4R case, the alpha channel is always assumed to be channel 3.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSwapChannels_32s_AC4R
     (pSrc : access nppdefs_h.Npp32s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aDstOrder : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:6836
   pragma Import (C, nppiSwapChannels_32s_AC4R, "nppiSwapChannels_32s_AC4R");

  --* 
  -- * 3 channel 32-bit floating point source image to 3 channel destination image.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
  -- *      of the array contains the number of the channel that is stored in the n-th channel of
  -- *      the output image. E.g. Given an RGB image, aDstOrder = [2,1,0] converts this to BGR
  -- *      channel order.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSwapChannels_32f_C3R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aDstOrder : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:6852
   pragma Import (C, nppiSwapChannels_32f_C3R, "nppiSwapChannels_32f_C3R");

  --* 
  -- * 3 channel 32-bit floating point in place image.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI oSizeROI \ref roi_specification.
  -- * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
  -- *      of the array contains the number of the channel that is stored in the n-th channel of
  -- *      the output image. E.g. Given an RGB image, aDstOrder = [2,1,0] converts this to BGR
  -- *      channel order.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSwapChannels_32f_C3IR
     (pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aDstOrder : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:6866
   pragma Import (C, nppiSwapChannels_32f_C3IR, "nppiSwapChannels_32f_C3IR");

  --* 
  -- * 4 channel 32-bit floating point source image to 3 channel destination image.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
  -- *      of the array contains the number of the channel that is stored in the n-th channel of
  -- *      the output image. E.g. Given an RGBA image, aDstOrder = [2,1,0] converts this to a 3 channel BGR
  -- *      channel order.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSwapChannels_32f_C4C3R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aDstOrder : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:6881
   pragma Import (C, nppiSwapChannels_32f_C4C3R, "nppiSwapChannels_32f_C4C3R");

  --*
  -- * 4 channel 32-bit floating point source image to 4 channel destination image.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
  -- *      of the array contains the number of the channel that is stored in the n-th channel of
  -- *      the output image. E.g. Given an ARGB image, aDstOrder = [3,2,1,0] converts this to BGRA
  -- *      channel order.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSwapChannels_32f_C4R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aDstOrder : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:6897
   pragma Import (C, nppiSwapChannels_32f_C4R, "nppiSwapChannels_32f_C4R");

  --* 
  -- * 4 channel 32-bit floating point in place image.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
  -- *      of the array contains the number of the channel that is stored in the n-th channel of
  -- *      the output image. E.g. Given an ARGB image, aDstOrder = [3,2,1,0] converts this to BGRA
  -- *      channel order.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSwapChannels_32f_C4IR
     (pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aDstOrder : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:6911
   pragma Import (C, nppiSwapChannels_32f_C4IR, "nppiSwapChannels_32f_C4IR");

  --* 
  -- * 3 channel 32-bit floating point source image to 4 channel destination image.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
  -- *      of the array contains the number of the channel that is stored in the n-th channel of
  -- *      the output image. E.g. Given an RGB image, aDstOrder = [3,2,1,0] converts this to VBGR
  -- *      channel order.
  -- * \param nValue (V) Single channel constant value that can be replicated in one or more of the 4 destination channels.
  -- *      nValue is either written or not written to a particular channel depending on the aDstOrder entry for that destination
  -- *      channel. An aDstOrder value of 3 will output nValue to that channel, an aDstOrder value greater than 3 will leave that
  -- *      particular destination channel value unmodified.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSwapChannels_32f_C3C4R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aDstOrder : access int;
      nValue : nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:6930
   pragma Import (C, nppiSwapChannels_32f_C3C4R, "nppiSwapChannels_32f_C3C4R");

  --*
  -- * 4 channel 32-bit floating point source image to 4 channel destination image with destination alpha channel unaffected.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
  -- *      of the array contains the number of the channel that is stored in the n-th channel of
  -- *      the output image. E.g. Given an RGBA image, aDstOrder = [2,1,0] converts this to BGRA
  -- *      channel order. In the AC4R case, the alpha channel is always assumed to be channel 3.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSwapChannels_32f_AC4R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      aDstOrder : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_data_exchange_and_initialization.h:6946
   pragma Import (C, nppiSwapChannels_32f_AC4R, "nppiSwapChannels_32f_AC4R");

  --* @} SwapChannels  
  --* @} image_swap_channels  
  --* @} image_data_exchange_and_initialization  
  -- extern "C"  
end nppi_data_exchange_and_initialization_h;

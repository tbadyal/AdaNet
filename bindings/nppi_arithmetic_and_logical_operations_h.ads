pragma Ada_2005;
pragma Style_Checks (Off);

with Interfaces.C; use Interfaces.C;
with nppdefs_h;

package nppi_arithmetic_and_logical_operations_h is

  -- Copyright 2009-2015 NVIDIA Corporation.  All rights reserved. 
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
  -- * \file nppi_arithmetic_and_logical_operations.h
  -- * Image Arithmetic and Logical Operations.
  --  

  --* 
  -- * @defgroup image_arithmetic_and_logical_operations Arithmetic and Logical Operations
  -- * @ingroup nppi
  -- * @{
  -- *
  -- * These functions can be found in either the nppi or nppial libraries. Linking to only the sub-libraries that you use can significantly
  -- * save link time, application load time, and CUDA runtime startup time when using dynamic libraries.
  --  

  --* 
  -- * @defgroup image_arithmetic_operations Arithmetic Operations
  -- * @{
  --  

  --* 
  -- * @defgroup image_addc AddC
  -- *
  -- * Adds a constant value to each pixel of an image.
  -- *
  -- * @{
  --  

  --* 
  -- * One 8-bit unsigned char channel image add constant, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param nConstant Constant.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAddC_8u_C1RSfs
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      nConstant : nppdefs_h.Npp8u;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:98
   pragma Import (C, nppiAddC_8u_C1RSfs, "nppiAddC_8u_C1RSfs");

  --* 
  -- * One 8-bit unsigned char channel in place image add constant, scale, then clamp to saturated value.
  -- * \param nConstant Constant.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAddC_8u_C1IRSfs
     (nConstant : nppdefs_h.Npp8u;
      pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:111
   pragma Import (C, nppiAddC_8u_C1IRSfs, "nppiAddC_8u_C1IRSfs");

  --* 
  -- * Three 8-bit unsigned char channel image add constant, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel..
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAddC_8u_C3RSfs
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp8u;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:125
   pragma Import (C, nppiAddC_8u_C3RSfs, "nppiAddC_8u_C3RSfs");

  --* 
  -- * Three 8-bit unsigned char channel 8-bit unsigned char in place image add constant, scale, then clamp to saturated value.
  -- * \param aConstants fixed size array of constant values, one per channel..
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAddC_8u_C3IRSfs
     (aConstants : access nppdefs_h.Npp8u;
      pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:138
   pragma Import (C, nppiAddC_8u_C3IRSfs, "nppiAddC_8u_C3IRSfs");

  --* 
  -- * Four 8-bit unsigned char channel with unmodified alpha image add constant, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel..
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAddC_8u_AC4RSfs
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp8u;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:152
   pragma Import (C, nppiAddC_8u_AC4RSfs, "nppiAddC_8u_AC4RSfs");

  --* 
  -- * Four 8-bit unsigned char channel with unmodified alpha in place image add constant, scale, then clamp to saturated value.
  -- * \param aConstants fixed size array of constant values, one per channel..
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAddC_8u_AC4IRSfs
     (aConstants : access nppdefs_h.Npp8u;
      pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:165
   pragma Import (C, nppiAddC_8u_AC4IRSfs, "nppiAddC_8u_AC4IRSfs");

  --* 
  -- * Four 8-bit unsigned char channel image add constant, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel..
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAddC_8u_C4RSfs
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp8u;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:179
   pragma Import (C, nppiAddC_8u_C4RSfs, "nppiAddC_8u_C4RSfs");

  --* 
  -- * Four 8-bit unsigned char channel in place image add constant, scale, then clamp to saturated value.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAddC_8u_C4IRSfs
     (aConstants : access nppdefs_h.Npp8u;
      pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:192
   pragma Import (C, nppiAddC_8u_C4IRSfs, "nppiAddC_8u_C4IRSfs");

  --* 
  -- * One 16-bit unsigned short channel image add constant, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param nConstant Constant.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAddC_16u_C1RSfs
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      nConstant : nppdefs_h.Npp16u;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:206
   pragma Import (C, nppiAddC_16u_C1RSfs, "nppiAddC_16u_C1RSfs");

  --* 
  -- * One 16-bit unsigned short channel in place image add constant, scale, then clamp to saturated value.
  -- * \param nConstant Constant.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAddC_16u_C1IRSfs
     (nConstant : nppdefs_h.Npp16u;
      pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:219
   pragma Import (C, nppiAddC_16u_C1IRSfs, "nppiAddC_16u_C1IRSfs");

  --* 
  -- * Three 16-bit unsigned short channel image add constant, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAddC_16u_C3RSfs
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp16u;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:233
   pragma Import (C, nppiAddC_16u_C3RSfs, "nppiAddC_16u_C3RSfs");

  --* 
  -- * Three 16-bit unsigned short channel in place image add constant, scale, then clamp to saturated value.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAddC_16u_C3IRSfs
     (aConstants : access nppdefs_h.Npp16u;
      pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:246
   pragma Import (C, nppiAddC_16u_C3IRSfs, "nppiAddC_16u_C3IRSfs");

  --* 
  -- * Four 16-bit unsigned short channel with unmodified alpha image add constant, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAddC_16u_AC4RSfs
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp16u;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:260
   pragma Import (C, nppiAddC_16u_AC4RSfs, "nppiAddC_16u_AC4RSfs");

  --* 
  -- * Four 16-bit unsigned short channel with unmodified alpha in place image add constant, scale, then clamp to saturated value.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAddC_16u_AC4IRSfs
     (aConstants : access nppdefs_h.Npp16u;
      pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:273
   pragma Import (C, nppiAddC_16u_AC4IRSfs, "nppiAddC_16u_AC4IRSfs");

  --* 
  -- * Four 16-bit unsigned short channel image add constant, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAddC_16u_C4RSfs
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp16u;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:287
   pragma Import (C, nppiAddC_16u_C4RSfs, "nppiAddC_16u_C4RSfs");

  --* 
  -- * Four 16-bit unsigned short channel in place image add constant, scale, then clamp to saturated value.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAddC_16u_C4IRSfs
     (aConstants : access nppdefs_h.Npp16u;
      pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:300
   pragma Import (C, nppiAddC_16u_C4IRSfs, "nppiAddC_16u_C4IRSfs");

  --* 
  -- * One 16-bit signed short channel image add constant, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param nConstant Constant.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAddC_16s_C1RSfs
     (pSrc1 : access nppdefs_h.Npp16s;
      nSrc1Step : int;
      nConstant : nppdefs_h.Npp16s;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:314
   pragma Import (C, nppiAddC_16s_C1RSfs, "nppiAddC_16s_C1RSfs");

  --* 
  -- * One 16-bit signed short channel in place image add constant, scale, then clamp to saturated value.
  -- * \param nConstant Constant.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAddC_16s_C1IRSfs
     (nConstant : nppdefs_h.Npp16s;
      pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:327
   pragma Import (C, nppiAddC_16s_C1IRSfs, "nppiAddC_16s_C1IRSfs");

  --* 
  -- * Three 16-bit signed short channel image add constant, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAddC_16s_C3RSfs
     (pSrc1 : access nppdefs_h.Npp16s;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp16s;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:341
   pragma Import (C, nppiAddC_16s_C3RSfs, "nppiAddC_16s_C3RSfs");

  --* 
  -- * Three 16-bit signed short channel in place image add constant, scale, then clamp to saturated value.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAddC_16s_C3IRSfs
     (aConstants : access nppdefs_h.Npp16s;
      pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:354
   pragma Import (C, nppiAddC_16s_C3IRSfs, "nppiAddC_16s_C3IRSfs");

  --* 
  -- * Four 16-bit signed short channel with unmodified alpha image add constant, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAddC_16s_AC4RSfs
     (pSrc1 : access nppdefs_h.Npp16s;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp16s;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:368
   pragma Import (C, nppiAddC_16s_AC4RSfs, "nppiAddC_16s_AC4RSfs");

  --* 
  -- * Four 16-bit signed short channel with unmodified alpha in place image add constant, scale, then clamp to saturated value.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAddC_16s_AC4IRSfs
     (aConstants : access nppdefs_h.Npp16s;
      pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:381
   pragma Import (C, nppiAddC_16s_AC4IRSfs, "nppiAddC_16s_AC4IRSfs");

  --* 
  -- * Four 16-bit signed short channel image add constant, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAddC_16s_C4RSfs
     (pSrc1 : access nppdefs_h.Npp16s;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp16s;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:395
   pragma Import (C, nppiAddC_16s_C4RSfs, "nppiAddC_16s_C4RSfs");

  --* 
  -- * Four 16-bit signed short channel in place image add constant, scale, then clamp to saturated value.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAddC_16s_C4IRSfs
     (aConstants : access nppdefs_h.Npp16s;
      pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:408
   pragma Import (C, nppiAddC_16s_C4IRSfs, "nppiAddC_16s_C4IRSfs");

  --* 
  -- * One 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel image add constant, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param nConstant Constant.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAddC_16sc_C1RSfs
     (pSrc1 : access constant nppdefs_h.Npp16sc;
      nSrc1Step : int;
      nConstant : nppdefs_h.Npp16sc;
      pDst : access nppdefs_h.Npp16sc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:422
   pragma Import (C, nppiAddC_16sc_C1RSfs, "nppiAddC_16sc_C1RSfs");

  --* 
  -- * One 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel in place image add constant, scale, then clamp to saturated value.
  -- * \param nConstant Constant.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAddC_16sc_C1IRSfs
     (nConstant : nppdefs_h.Npp16sc;
      pSrcDst : access nppdefs_h.Npp16sc;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:435
   pragma Import (C, nppiAddC_16sc_C1IRSfs, "nppiAddC_16sc_C1IRSfs");

  --* 
  -- * Three 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel image add constant, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAddC_16sc_C3RSfs
     (pSrc1 : access constant nppdefs_h.Npp16sc;
      nSrc1Step : int;
      aConstants : access constant nppdefs_h.Npp16sc;
      pDst : access nppdefs_h.Npp16sc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:449
   pragma Import (C, nppiAddC_16sc_C3RSfs, "nppiAddC_16sc_C3RSfs");

  --* 
  -- * Three 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel in place image add constant, scale, then clamp to saturated value.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAddC_16sc_C3IRSfs
     (aConstants : access constant nppdefs_h.Npp16sc;
      pSrcDst : access nppdefs_h.Npp16sc;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:462
   pragma Import (C, nppiAddC_16sc_C3IRSfs, "nppiAddC_16sc_C3IRSfs");

  --* 
  -- * Four 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel with unmodified alpha image add constant, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAddC_16sc_AC4RSfs
     (pSrc1 : access constant nppdefs_h.Npp16sc;
      nSrc1Step : int;
      aConstants : access constant nppdefs_h.Npp16sc;
      pDst : access nppdefs_h.Npp16sc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:476
   pragma Import (C, nppiAddC_16sc_AC4RSfs, "nppiAddC_16sc_AC4RSfs");

  --* 
  -- * Four 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel with unmodified alpha in place image add constant, scale, then clamp to saturated value.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAddC_16sc_AC4IRSfs
     (aConstants : access constant nppdefs_h.Npp16sc;
      pSrcDst : access nppdefs_h.Npp16sc;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:489
   pragma Import (C, nppiAddC_16sc_AC4IRSfs, "nppiAddC_16sc_AC4IRSfs");

  --* 
  -- * One 32-bit signed integer channel image add constant, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param nConstant Constant.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAddC_32s_C1RSfs
     (pSrc1 : access nppdefs_h.Npp32s;
      nSrc1Step : int;
      nConstant : nppdefs_h.Npp32s;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:503
   pragma Import (C, nppiAddC_32s_C1RSfs, "nppiAddC_32s_C1RSfs");

  --* 
  -- * One 32-bit signed integer channel in place image add constant, scale, then clamp to saturated value.
  -- * \param nConstant Constant.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAddC_32s_C1IRSfs
     (nConstant : nppdefs_h.Npp32s;
      pSrcDst : access nppdefs_h.Npp32s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:516
   pragma Import (C, nppiAddC_32s_C1IRSfs, "nppiAddC_32s_C1IRSfs");

  --* 
  -- * Three 32-bit signed integer channel image add constant, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAddC_32s_C3RSfs
     (pSrc1 : access nppdefs_h.Npp32s;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp32s;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:530
   pragma Import (C, nppiAddC_32s_C3RSfs, "nppiAddC_32s_C3RSfs");

  --* 
  -- * Three 32-bit signed integer channel in place image add constant, scale, then clamp to saturated value.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAddC_32s_C3IRSfs
     (aConstants : access nppdefs_h.Npp32s;
      pSrcDst : access nppdefs_h.Npp32s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:543
   pragma Import (C, nppiAddC_32s_C3IRSfs, "nppiAddC_32s_C3IRSfs");

  --* 
  -- * One 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel image add constant, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param nConstant Constant.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAddC_32sc_C1RSfs
     (pSrc1 : access constant nppdefs_h.Npp32sc;
      nSrc1Step : int;
      nConstant : nppdefs_h.Npp32sc;
      pDst : access nppdefs_h.Npp32sc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:557
   pragma Import (C, nppiAddC_32sc_C1RSfs, "nppiAddC_32sc_C1RSfs");

  --* 
  -- * One 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel in place image add constant, scale, then clamp to saturated value.
  -- * \param nConstant Constant.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAddC_32sc_C1IRSfs
     (nConstant : nppdefs_h.Npp32sc;
      pSrcDst : access nppdefs_h.Npp32sc;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:570
   pragma Import (C, nppiAddC_32sc_C1IRSfs, "nppiAddC_32sc_C1IRSfs");

  --* 
  -- * Three 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel image add constant, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAddC_32sc_C3RSfs
     (pSrc1 : access constant nppdefs_h.Npp32sc;
      nSrc1Step : int;
      aConstants : access constant nppdefs_h.Npp32sc;
      pDst : access nppdefs_h.Npp32sc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:584
   pragma Import (C, nppiAddC_32sc_C3RSfs, "nppiAddC_32sc_C3RSfs");

  --* 
  -- * Three 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel in place image add constant, scale, then clamp to saturated value.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAddC_32sc_C3IRSfs
     (aConstants : access constant nppdefs_h.Npp32sc;
      pSrcDst : access nppdefs_h.Npp32sc;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:597
   pragma Import (C, nppiAddC_32sc_C3IRSfs, "nppiAddC_32sc_C3IRSfs");

  --* 
  -- * Four 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel with unmodified alpha image add constant, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAddC_32sc_AC4RSfs
     (pSrc1 : access constant nppdefs_h.Npp32sc;
      nSrc1Step : int;
      aConstants : access constant nppdefs_h.Npp32sc;
      pDst : access nppdefs_h.Npp32sc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:611
   pragma Import (C, nppiAddC_32sc_AC4RSfs, "nppiAddC_32sc_AC4RSfs");

  --* 
  -- * Four 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel with unmodified alpha in place image add constant, scale, then clamp to saturated value.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAddC_32sc_AC4IRSfs
     (aConstants : access constant nppdefs_h.Npp32sc;
      pSrcDst : access nppdefs_h.Npp32sc;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:624
   pragma Import (C, nppiAddC_32sc_AC4IRSfs, "nppiAddC_32sc_AC4IRSfs");

  --* 
  -- * One 32-bit floating point channel image add constant.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param nConstant Constant.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAddC_32f_C1R
     (pSrc1 : access nppdefs_h.Npp32f;
      nSrc1Step : int;
      nConstant : nppdefs_h.Npp32f;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:637
   pragma Import (C, nppiAddC_32f_C1R, "nppiAddC_32f_C1R");

  --* 
  -- * One 32-bit floating point channel in place image add constant.
  -- * \param nConstant Constant.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAddC_32f_C1IR
     (nConstant : nppdefs_h.Npp32f;
      pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:649
   pragma Import (C, nppiAddC_32f_C1IR, "nppiAddC_32f_C1IR");

  --* 
  -- * Three 32-bit floating point channel image add constant.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAddC_32f_C3R
     (pSrc1 : access nppdefs_h.Npp32f;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp32f;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:662
   pragma Import (C, nppiAddC_32f_C3R, "nppiAddC_32f_C3R");

  --* 
  -- * Three 32-bit floating point channel in place image add constant.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAddC_32f_C3IR
     (aConstants : access nppdefs_h.Npp32f;
      pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:674
   pragma Import (C, nppiAddC_32f_C3IR, "nppiAddC_32f_C3IR");

  --* 
  -- * Four 32-bit floating point channel with unmodified alpha image add constant.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAddC_32f_AC4R
     (pSrc1 : access nppdefs_h.Npp32f;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp32f;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:687
   pragma Import (C, nppiAddC_32f_AC4R, "nppiAddC_32f_AC4R");

  --* 
  -- * Four 32-bit floating point channel with unmodified alpha in place image add constant.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAddC_32f_AC4IR
     (aConstants : access nppdefs_h.Npp32f;
      pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:699
   pragma Import (C, nppiAddC_32f_AC4IR, "nppiAddC_32f_AC4IR");

  --* 
  -- * Four 32-bit floating point channel image add constant.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAddC_32f_C4R
     (pSrc1 : access nppdefs_h.Npp32f;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp32f;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:712
   pragma Import (C, nppiAddC_32f_C4R, "nppiAddC_32f_C4R");

  --* 
  -- * Four 32-bit floating point channel in place image add constant.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAddC_32f_C4IR
     (aConstants : access nppdefs_h.Npp32f;
      pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:724
   pragma Import (C, nppiAddC_32f_C4IR, "nppiAddC_32f_C4IR");

  --* 
  -- * One 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel image add constant.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param nConstant Constant.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAddC_32fc_C1R
     (pSrc1 : access constant nppdefs_h.Npp32fc;
      nSrc1Step : int;
      nConstant : nppdefs_h.Npp32fc;
      pDst : access nppdefs_h.Npp32fc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:737
   pragma Import (C, nppiAddC_32fc_C1R, "nppiAddC_32fc_C1R");

  --* 
  -- * One 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel in place image add constant.
  -- * \param nConstant Constant.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAddC_32fc_C1IR
     (nConstant : nppdefs_h.Npp32fc;
      pSrcDst : access nppdefs_h.Npp32fc;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:749
   pragma Import (C, nppiAddC_32fc_C1IR, "nppiAddC_32fc_C1IR");

  --* 
  -- * Three 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel image add constant.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAddC_32fc_C3R
     (pSrc1 : access constant nppdefs_h.Npp32fc;
      nSrc1Step : int;
      aConstants : access constant nppdefs_h.Npp32fc;
      pDst : access nppdefs_h.Npp32fc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:762
   pragma Import (C, nppiAddC_32fc_C3R, "nppiAddC_32fc_C3R");

  --* 
  -- * Three 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel in place image add constant.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAddC_32fc_C3IR
     (aConstants : access constant nppdefs_h.Npp32fc;
      pSrcDst : access nppdefs_h.Npp32fc;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:774
   pragma Import (C, nppiAddC_32fc_C3IR, "nppiAddC_32fc_C3IR");

  --* 
  -- * Four 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel with unmodified alpha image add constant.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAddC_32fc_AC4R
     (pSrc1 : access constant nppdefs_h.Npp32fc;
      nSrc1Step : int;
      aConstants : access constant nppdefs_h.Npp32fc;
      pDst : access nppdefs_h.Npp32fc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:787
   pragma Import (C, nppiAddC_32fc_AC4R, "nppiAddC_32fc_AC4R");

  --* 
  -- * Four 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel with unmodified alpha in place image add constant.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAddC_32fc_AC4IR
     (aConstants : access constant nppdefs_h.Npp32fc;
      pSrcDst : access nppdefs_h.Npp32fc;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:799
   pragma Import (C, nppiAddC_32fc_AC4IR, "nppiAddC_32fc_AC4IR");

  --* 
  -- * Four 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel image add constant.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAddC_32fc_C4R
     (pSrc1 : access constant nppdefs_h.Npp32fc;
      nSrc1Step : int;
      aConstants : access constant nppdefs_h.Npp32fc;
      pDst : access nppdefs_h.Npp32fc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:812
   pragma Import (C, nppiAddC_32fc_C4R, "nppiAddC_32fc_C4R");

  --* 
  -- * Four 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel in place image add constant.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAddC_32fc_C4IR
     (aConstants : access constant nppdefs_h.Npp32fc;
      pSrcDst : access nppdefs_h.Npp32fc;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:824
   pragma Import (C, nppiAddC_32fc_C4IR, "nppiAddC_32fc_C4IR");

  --* @} image_addc  
  --* 
  -- * @defgroup image_mulc MulC
  -- *
  -- * Multiplies each pixel of an image by a constant value.
  -- *
  -- * @{
  --  

  --* 
  -- * One 8-bit unsigned char channel image multiply by constant, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param nConstant Constant.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMulC_8u_C1RSfs
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      nConstant : nppdefs_h.Npp8u;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:849
   pragma Import (C, nppiMulC_8u_C1RSfs, "nppiMulC_8u_C1RSfs");

  --* 
  -- * One 8-bit unsigned char channel in place image multiply by constant, scale, then clamp to saturated value.
  -- * \param nConstant Constant.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMulC_8u_C1IRSfs
     (nConstant : nppdefs_h.Npp8u;
      pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:862
   pragma Import (C, nppiMulC_8u_C1IRSfs, "nppiMulC_8u_C1IRSfs");

  --* 
  -- * Three 8-bit unsigned char channel image multiply by constant, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMulC_8u_C3RSfs
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp8u;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:876
   pragma Import (C, nppiMulC_8u_C3RSfs, "nppiMulC_8u_C3RSfs");

  --* 
  -- * Three 8-bit unsigned char channel 8-bit unsigned char in place image multiply by constant, scale, then clamp to saturated value.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMulC_8u_C3IRSfs
     (aConstants : access nppdefs_h.Npp8u;
      pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:889
   pragma Import (C, nppiMulC_8u_C3IRSfs, "nppiMulC_8u_C3IRSfs");

  --* 
  -- * Four 8-bit unsigned char channel with unmodified alpha image multiply by constant, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMulC_8u_AC4RSfs
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp8u;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:903
   pragma Import (C, nppiMulC_8u_AC4RSfs, "nppiMulC_8u_AC4RSfs");

  --* 
  -- * Four 8-bit unsigned char channel with unmodified alpha in place image multiply by constant, scale, then clamp to saturated value.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMulC_8u_AC4IRSfs
     (aConstants : access nppdefs_h.Npp8u;
      pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:916
   pragma Import (C, nppiMulC_8u_AC4IRSfs, "nppiMulC_8u_AC4IRSfs");

  --* 
  -- * Four 8-bit unsigned char channel image multiply by constant, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMulC_8u_C4RSfs
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp8u;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:930
   pragma Import (C, nppiMulC_8u_C4RSfs, "nppiMulC_8u_C4RSfs");

  --* 
  -- * Four 8-bit unsigned char channel in place image multiply by constant, scale, then clamp to saturated value.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMulC_8u_C4IRSfs
     (aConstants : access nppdefs_h.Npp8u;
      pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:943
   pragma Import (C, nppiMulC_8u_C4IRSfs, "nppiMulC_8u_C4IRSfs");

  --* 
  -- * One 16-bit unsigned short channel image multiply by constant, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param nConstant Constant.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMulC_16u_C1RSfs
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      nConstant : nppdefs_h.Npp16u;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:957
   pragma Import (C, nppiMulC_16u_C1RSfs, "nppiMulC_16u_C1RSfs");

  --* 
  -- * One 16-bit unsigned short channel in place image multiply by constant, scale, then clamp to saturated value.
  -- * \param nConstant Constant.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMulC_16u_C1IRSfs
     (nConstant : nppdefs_h.Npp16u;
      pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:970
   pragma Import (C, nppiMulC_16u_C1IRSfs, "nppiMulC_16u_C1IRSfs");

  --* 
  -- * Three 16-bit unsigned short channel image multiply by constant, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMulC_16u_C3RSfs
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp16u;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:984
   pragma Import (C, nppiMulC_16u_C3RSfs, "nppiMulC_16u_C3RSfs");

  --* 
  -- * Three 16-bit unsigned short channel in place image multiply by constant, scale, then clamp to saturated value.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMulC_16u_C3IRSfs
     (aConstants : access nppdefs_h.Npp16u;
      pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:997
   pragma Import (C, nppiMulC_16u_C3IRSfs, "nppiMulC_16u_C3IRSfs");

  --* 
  -- * Four 16-bit unsigned short channel with unmodified alpha image multiply by constant, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMulC_16u_AC4RSfs
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp16u;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:1011
   pragma Import (C, nppiMulC_16u_AC4RSfs, "nppiMulC_16u_AC4RSfs");

  --* 
  -- * Four 16-bit unsigned short channel with unmodified alpha in place image multiply by constant, scale, then clamp to saturated value.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMulC_16u_AC4IRSfs
     (aConstants : access nppdefs_h.Npp16u;
      pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:1024
   pragma Import (C, nppiMulC_16u_AC4IRSfs, "nppiMulC_16u_AC4IRSfs");

  --* 
  -- * Four 16-bit unsigned short channel image multiply by constant, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMulC_16u_C4RSfs
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp16u;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:1038
   pragma Import (C, nppiMulC_16u_C4RSfs, "nppiMulC_16u_C4RSfs");

  --* 
  -- * Four 16-bit unsigned short channel in place image multiply by constant, scale, then clamp to saturated value.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMulC_16u_C4IRSfs
     (aConstants : access nppdefs_h.Npp16u;
      pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:1051
   pragma Import (C, nppiMulC_16u_C4IRSfs, "nppiMulC_16u_C4IRSfs");

  --* 
  -- * One 16-bit signed short channel image multiply by constant, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param nConstant Constant.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMulC_16s_C1RSfs
     (pSrc1 : access nppdefs_h.Npp16s;
      nSrc1Step : int;
      nConstant : nppdefs_h.Npp16s;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:1065
   pragma Import (C, nppiMulC_16s_C1RSfs, "nppiMulC_16s_C1RSfs");

  --* 
  -- * One 16-bit signed short channel in place image multiply by constant, scale, then clamp to saturated value.
  -- * \param nConstant Constant.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMulC_16s_C1IRSfs
     (nConstant : nppdefs_h.Npp16s;
      pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:1078
   pragma Import (C, nppiMulC_16s_C1IRSfs, "nppiMulC_16s_C1IRSfs");

  --* 
  -- * Three 16-bit signed short channel image multiply by constant, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMulC_16s_C3RSfs
     (pSrc1 : access nppdefs_h.Npp16s;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp16s;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:1092
   pragma Import (C, nppiMulC_16s_C3RSfs, "nppiMulC_16s_C3RSfs");

  --* 
  -- * Three 16-bit signed short channel in place image multiply by constant, scale, then clamp to saturated value.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMulC_16s_C3IRSfs
     (aConstants : access nppdefs_h.Npp16s;
      pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:1105
   pragma Import (C, nppiMulC_16s_C3IRSfs, "nppiMulC_16s_C3IRSfs");

  --* 
  -- * Four 16-bit signed short channel with unmodified alpha image multiply by constant, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMulC_16s_AC4RSfs
     (pSrc1 : access nppdefs_h.Npp16s;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp16s;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:1119
   pragma Import (C, nppiMulC_16s_AC4RSfs, "nppiMulC_16s_AC4RSfs");

  --* 
  -- * Four 16-bit signed short channel with unmodified alpha in place image multiply by constant, scale, then clamp to saturated value.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMulC_16s_AC4IRSfs
     (aConstants : access nppdefs_h.Npp16s;
      pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:1132
   pragma Import (C, nppiMulC_16s_AC4IRSfs, "nppiMulC_16s_AC4IRSfs");

  --* 
  -- * Four 16-bit signed short channel image multiply by constant, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMulC_16s_C4RSfs
     (pSrc1 : access nppdefs_h.Npp16s;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp16s;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:1146
   pragma Import (C, nppiMulC_16s_C4RSfs, "nppiMulC_16s_C4RSfs");

  --* 
  -- * Four 16-bit signed short channel in place image multiply by constant, scale, then clamp to saturated value.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMulC_16s_C4IRSfs
     (aConstants : access nppdefs_h.Npp16s;
      pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:1159
   pragma Import (C, nppiMulC_16s_C4IRSfs, "nppiMulC_16s_C4IRSfs");

  --* 
  -- * One 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel image multiply by constant, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param nConstant Constant.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMulC_16sc_C1RSfs
     (pSrc1 : access constant nppdefs_h.Npp16sc;
      nSrc1Step : int;
      nConstant : nppdefs_h.Npp16sc;
      pDst : access nppdefs_h.Npp16sc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:1173
   pragma Import (C, nppiMulC_16sc_C1RSfs, "nppiMulC_16sc_C1RSfs");

  --* 
  -- * One 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel in place image multiply by constant, scale, then clamp to saturated value.
  -- * \param nConstant Constant.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMulC_16sc_C1IRSfs
     (nConstant : nppdefs_h.Npp16sc;
      pSrcDst : access nppdefs_h.Npp16sc;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:1186
   pragma Import (C, nppiMulC_16sc_C1IRSfs, "nppiMulC_16sc_C1IRSfs");

  --* 
  -- * Three 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel image multiply by constant, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMulC_16sc_C3RSfs
     (pSrc1 : access constant nppdefs_h.Npp16sc;
      nSrc1Step : int;
      aConstants : access constant nppdefs_h.Npp16sc;
      pDst : access nppdefs_h.Npp16sc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:1200
   pragma Import (C, nppiMulC_16sc_C3RSfs, "nppiMulC_16sc_C3RSfs");

  --* 
  -- * Three 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel in place image multiply by constant, scale, then clamp to saturated value.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMulC_16sc_C3IRSfs
     (aConstants : access constant nppdefs_h.Npp16sc;
      pSrcDst : access nppdefs_h.Npp16sc;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:1213
   pragma Import (C, nppiMulC_16sc_C3IRSfs, "nppiMulC_16sc_C3IRSfs");

  --* 
  -- * Four 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel with unmodified alpha image multiply by constant, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMulC_16sc_AC4RSfs
     (pSrc1 : access constant nppdefs_h.Npp16sc;
      nSrc1Step : int;
      aConstants : access constant nppdefs_h.Npp16sc;
      pDst : access nppdefs_h.Npp16sc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:1227
   pragma Import (C, nppiMulC_16sc_AC4RSfs, "nppiMulC_16sc_AC4RSfs");

  --* 
  -- * Four 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel with unmodified alpha in place image multiply by constant, scale, then clamp to saturated value.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMulC_16sc_AC4IRSfs
     (aConstants : access constant nppdefs_h.Npp16sc;
      pSrcDst : access nppdefs_h.Npp16sc;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:1240
   pragma Import (C, nppiMulC_16sc_AC4IRSfs, "nppiMulC_16sc_AC4IRSfs");

  --* 
  -- * One 32-bit signed integer channel image multiply by constant, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param nConstant Constant.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMulC_32s_C1RSfs
     (pSrc1 : access nppdefs_h.Npp32s;
      nSrc1Step : int;
      nConstant : nppdefs_h.Npp32s;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:1254
   pragma Import (C, nppiMulC_32s_C1RSfs, "nppiMulC_32s_C1RSfs");

  --* 
  -- * One 32-bit signed integer channel in place image multiply by constant, scale, then clamp to saturated value.
  -- * \param nConstant Constant.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMulC_32s_C1IRSfs
     (nConstant : nppdefs_h.Npp32s;
      pSrcDst : access nppdefs_h.Npp32s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:1267
   pragma Import (C, nppiMulC_32s_C1IRSfs, "nppiMulC_32s_C1IRSfs");

  --* 
  -- * Three 32-bit signed integer channel image multiply by constant, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMulC_32s_C3RSfs
     (pSrc1 : access nppdefs_h.Npp32s;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp32s;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:1281
   pragma Import (C, nppiMulC_32s_C3RSfs, "nppiMulC_32s_C3RSfs");

  --* 
  -- * Three 32-bit signed integer channel in place image multiply by constant, scale, then clamp to saturated value.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMulC_32s_C3IRSfs
     (aConstants : access nppdefs_h.Npp32s;
      pSrcDst : access nppdefs_h.Npp32s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:1294
   pragma Import (C, nppiMulC_32s_C3IRSfs, "nppiMulC_32s_C3IRSfs");

  --* 
  -- * One 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel image multiply by constant, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param nConstant Constant.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMulC_32sc_C1RSfs
     (pSrc1 : access constant nppdefs_h.Npp32sc;
      nSrc1Step : int;
      nConstant : nppdefs_h.Npp32sc;
      pDst : access nppdefs_h.Npp32sc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:1308
   pragma Import (C, nppiMulC_32sc_C1RSfs, "nppiMulC_32sc_C1RSfs");

  --* 
  -- * One 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel in place image multiply by constant, scale, then clamp to saturated value.
  -- * \param nConstant Constant.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMulC_32sc_C1IRSfs
     (nConstant : nppdefs_h.Npp32sc;
      pSrcDst : access nppdefs_h.Npp32sc;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:1321
   pragma Import (C, nppiMulC_32sc_C1IRSfs, "nppiMulC_32sc_C1IRSfs");

  --* 
  -- * Three 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel image multiply by constant, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMulC_32sc_C3RSfs
     (pSrc1 : access constant nppdefs_h.Npp32sc;
      nSrc1Step : int;
      aConstants : access constant nppdefs_h.Npp32sc;
      pDst : access nppdefs_h.Npp32sc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:1335
   pragma Import (C, nppiMulC_32sc_C3RSfs, "nppiMulC_32sc_C3RSfs");

  --* 
  -- * Three 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel in place image multiply by constant, scale, then clamp to saturated value.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMulC_32sc_C3IRSfs
     (aConstants : access constant nppdefs_h.Npp32sc;
      pSrcDst : access nppdefs_h.Npp32sc;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:1348
   pragma Import (C, nppiMulC_32sc_C3IRSfs, "nppiMulC_32sc_C3IRSfs");

  --* 
  -- * Four 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel with unmodified alpha image multiply by constant, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMulC_32sc_AC4RSfs
     (pSrc1 : access constant nppdefs_h.Npp32sc;
      nSrc1Step : int;
      aConstants : access constant nppdefs_h.Npp32sc;
      pDst : access nppdefs_h.Npp32sc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:1362
   pragma Import (C, nppiMulC_32sc_AC4RSfs, "nppiMulC_32sc_AC4RSfs");

  --* 
  -- * Four 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel with unmodified alpha in place image multiply by constant, scale, then clamp to saturated value.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMulC_32sc_AC4IRSfs
     (aConstants : access constant nppdefs_h.Npp32sc;
      pSrcDst : access nppdefs_h.Npp32sc;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:1375
   pragma Import (C, nppiMulC_32sc_AC4IRSfs, "nppiMulC_32sc_AC4IRSfs");

  --* 
  -- * One 32-bit floating point channel image multiply by constant.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param nConstant Constant.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMulC_32f_C1R
     (pSrc1 : access nppdefs_h.Npp32f;
      nSrc1Step : int;
      nConstant : nppdefs_h.Npp32f;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:1388
   pragma Import (C, nppiMulC_32f_C1R, "nppiMulC_32f_C1R");

  --* 
  -- * One 32-bit floating point channel in place image multiply by constant.
  -- * \param nConstant Constant.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMulC_32f_C1IR
     (nConstant : nppdefs_h.Npp32f;
      pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:1400
   pragma Import (C, nppiMulC_32f_C1IR, "nppiMulC_32f_C1IR");

  --* 
  -- * Three 32-bit floating point channel image multiply by constant.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMulC_32f_C3R
     (pSrc1 : access nppdefs_h.Npp32f;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp32f;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:1413
   pragma Import (C, nppiMulC_32f_C3R, "nppiMulC_32f_C3R");

  --* 
  -- * Three 32-bit floating point channel in place image multiply by constant.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMulC_32f_C3IR
     (aConstants : access nppdefs_h.Npp32f;
      pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:1425
   pragma Import (C, nppiMulC_32f_C3IR, "nppiMulC_32f_C3IR");

  --* 
  -- * Four 32-bit floating point channel with unmodified alpha image multiply by constant.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMulC_32f_AC4R
     (pSrc1 : access nppdefs_h.Npp32f;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp32f;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:1438
   pragma Import (C, nppiMulC_32f_AC4R, "nppiMulC_32f_AC4R");

  --* 
  -- * Four 32-bit floating point channel with unmodified alpha in place image multiply by constant.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMulC_32f_AC4IR
     (aConstants : access nppdefs_h.Npp32f;
      pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:1450
   pragma Import (C, nppiMulC_32f_AC4IR, "nppiMulC_32f_AC4IR");

  --* 
  -- * Four 32-bit floating point channel image multiply by constant.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMulC_32f_C4R
     (pSrc1 : access nppdefs_h.Npp32f;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp32f;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:1463
   pragma Import (C, nppiMulC_32f_C4R, "nppiMulC_32f_C4R");

  --* 
  -- * Four 32-bit floating point channel in place image multiply by constant.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMulC_32f_C4IR
     (aConstants : access nppdefs_h.Npp32f;
      pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:1475
   pragma Import (C, nppiMulC_32f_C4IR, "nppiMulC_32f_C4IR");

  --* 
  -- * One 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel image multiply by constant.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param nConstant Constant.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMulC_32fc_C1R
     (pSrc1 : access constant nppdefs_h.Npp32fc;
      nSrc1Step : int;
      nConstant : nppdefs_h.Npp32fc;
      pDst : access nppdefs_h.Npp32fc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:1488
   pragma Import (C, nppiMulC_32fc_C1R, "nppiMulC_32fc_C1R");

  --* 
  -- * One 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel in place image multiply by constant.
  -- * \param nConstant Constant.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMulC_32fc_C1IR
     (nConstant : nppdefs_h.Npp32fc;
      pSrcDst : access nppdefs_h.Npp32fc;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:1500
   pragma Import (C, nppiMulC_32fc_C1IR, "nppiMulC_32fc_C1IR");

  --* 
  -- * Three 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel image multiply by constant.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMulC_32fc_C3R
     (pSrc1 : access constant nppdefs_h.Npp32fc;
      nSrc1Step : int;
      aConstants : access constant nppdefs_h.Npp32fc;
      pDst : access nppdefs_h.Npp32fc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:1513
   pragma Import (C, nppiMulC_32fc_C3R, "nppiMulC_32fc_C3R");

  --* 
  -- * Three 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel in place image multiply by constant.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMulC_32fc_C3IR
     (aConstants : access constant nppdefs_h.Npp32fc;
      pSrcDst : access nppdefs_h.Npp32fc;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:1525
   pragma Import (C, nppiMulC_32fc_C3IR, "nppiMulC_32fc_C3IR");

  --* 
  -- * Four 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel with unmodified alpha image multiply by constant.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMulC_32fc_AC4R
     (pSrc1 : access constant nppdefs_h.Npp32fc;
      nSrc1Step : int;
      aConstants : access constant nppdefs_h.Npp32fc;
      pDst : access nppdefs_h.Npp32fc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:1538
   pragma Import (C, nppiMulC_32fc_AC4R, "nppiMulC_32fc_AC4R");

  --* 
  -- * Four 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel with unmodified alpha in place image multiply by constant.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMulC_32fc_AC4IR
     (aConstants : access constant nppdefs_h.Npp32fc;
      pSrcDst : access nppdefs_h.Npp32fc;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:1550
   pragma Import (C, nppiMulC_32fc_AC4IR, "nppiMulC_32fc_AC4IR");

  --* 
  -- * Four 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel image multiply by constant.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMulC_32fc_C4R
     (pSrc1 : access constant nppdefs_h.Npp32fc;
      nSrc1Step : int;
      aConstants : access constant nppdefs_h.Npp32fc;
      pDst : access nppdefs_h.Npp32fc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:1563
   pragma Import (C, nppiMulC_32fc_C4R, "nppiMulC_32fc_C4R");

  --* 
  -- * Four 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel in place image multiply by constant.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMulC_32fc_C4IR
     (aConstants : access constant nppdefs_h.Npp32fc;
      pSrcDst : access nppdefs_h.Npp32fc;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:1575
   pragma Import (C, nppiMulC_32fc_C4IR, "nppiMulC_32fc_C4IR");

  --* @} image_mulc  
  --* 
  -- * @defgroup image_mulcscale MulCScale
  -- *
  -- * Multiplies each pixel of an image by a constant value then scales the result
  -- * by the maximum value for the data bit width.
  -- *
  -- * @{
  --  

  --* 
  -- * One 8-bit unsigned char channel image multiply by constant and scale by max bit width value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param nConstant Constant.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMulCScale_8u_C1R
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      nConstant : nppdefs_h.Npp8u;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:1599
   pragma Import (C, nppiMulCScale_8u_C1R, "nppiMulCScale_8u_C1R");

  --* 
  -- * One 8-bit unsigned char channel in place image multiply by constant and scale by max bit width value.
  -- * \param nConstant Constant.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMulCScale_8u_C1IR
     (nConstant : nppdefs_h.Npp8u;
      pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:1611
   pragma Import (C, nppiMulCScale_8u_C1IR, "nppiMulCScale_8u_C1IR");

  --* 
  -- * Three 8-bit unsigned char channel image multiply by constant and scale by max bit width value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMulCScale_8u_C3R
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp8u;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:1624
   pragma Import (C, nppiMulCScale_8u_C3R, "nppiMulCScale_8u_C3R");

  --* 
  -- * Three 8-bit unsigned char channel 8-bit unsigned char in place image multiply by constant and scale by max bit width value.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMulCScale_8u_C3IR
     (aConstants : access nppdefs_h.Npp8u;
      pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:1636
   pragma Import (C, nppiMulCScale_8u_C3IR, "nppiMulCScale_8u_C3IR");

  --* 
  -- * Four 8-bit unsigned char channel with unmodified alpha image multiply by constant and scale by max bit width value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMulCScale_8u_AC4R
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp8u;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:1649
   pragma Import (C, nppiMulCScale_8u_AC4R, "nppiMulCScale_8u_AC4R");

  --* 
  -- * Four 8-bit unsigned char channel with unmodified alpha in place image multiply by constant, scale and scale by max bit width value.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMulCScale_8u_AC4IR
     (aConstants : access nppdefs_h.Npp8u;
      pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:1661
   pragma Import (C, nppiMulCScale_8u_AC4IR, "nppiMulCScale_8u_AC4IR");

  --* 
  -- * Four 8-bit unsigned char channel image multiply by constant and scale by max bit width value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMulCScale_8u_C4R
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp8u;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:1674
   pragma Import (C, nppiMulCScale_8u_C4R, "nppiMulCScale_8u_C4R");

  --* 
  -- * Four 8-bit unsigned char channel in place image multiply by constant and scale by max bit width value.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMulCScale_8u_C4IR
     (aConstants : access nppdefs_h.Npp8u;
      pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:1686
   pragma Import (C, nppiMulCScale_8u_C4IR, "nppiMulCScale_8u_C4IR");

  --* 
  -- * One 16-bit unsigned short channel image multiply by constant and scale by max bit width value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param nConstant Constant.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMulCScale_16u_C1R
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      nConstant : nppdefs_h.Npp16u;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:1699
   pragma Import (C, nppiMulCScale_16u_C1R, "nppiMulCScale_16u_C1R");

  --* 
  -- * One 16-bit unsigned short channel in place image multiply by constant and scale by max bit width value.
  -- * \param nConstant Constant.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMulCScale_16u_C1IR
     (nConstant : nppdefs_h.Npp16u;
      pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:1711
   pragma Import (C, nppiMulCScale_16u_C1IR, "nppiMulCScale_16u_C1IR");

  --* 
  -- * Three 16-bit unsigned short channel image multiply by constant and scale by max bit width value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMulCScale_16u_C3R
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp16u;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:1724
   pragma Import (C, nppiMulCScale_16u_C3R, "nppiMulCScale_16u_C3R");

  --* 
  -- * Three 16-bit unsigned short channel in place image multiply by constant and scale by max bit width value.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMulCScale_16u_C3IR
     (aConstants : access nppdefs_h.Npp16u;
      pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:1736
   pragma Import (C, nppiMulCScale_16u_C3IR, "nppiMulCScale_16u_C3IR");

  --* 
  -- * Four 16-bit unsigned short channel with unmodified alpha image multiply by constant and scale by max bit width value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMulCScale_16u_AC4R
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp16u;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:1749
   pragma Import (C, nppiMulCScale_16u_AC4R, "nppiMulCScale_16u_AC4R");

  --* 
  -- * Four 16-bit unsigned short channel with unmodified alpha in place image multiply by constant and scale by max bit width value.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMulCScale_16u_AC4IR
     (aConstants : access nppdefs_h.Npp16u;
      pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:1761
   pragma Import (C, nppiMulCScale_16u_AC4IR, "nppiMulCScale_16u_AC4IR");

  --* 
  -- * Four 16-bit unsigned short channel image multiply by constant and scale by max bit width value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMulCScale_16u_C4R
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp16u;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:1774
   pragma Import (C, nppiMulCScale_16u_C4R, "nppiMulCScale_16u_C4R");

  --* 
  -- * Four 16-bit unsigned short channel in place image multiply by constant and scale by max bit width value.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMulCScale_16u_C4IR
     (aConstants : access nppdefs_h.Npp16u;
      pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:1786
   pragma Import (C, nppiMulCScale_16u_C4IR, "nppiMulCScale_16u_C4IR");

  --* @} image_mulcscale  
  --* @defgroup image_subc SubC
  -- * Subtracts a constant value from each pixel of an image.
  -- * @{
  --  

  --* 
  -- * One 8-bit unsigned char channel image subtract constant, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param nConstant Constant.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSubC_8u_C1RSfs
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      nConstant : nppdefs_h.Npp8u;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:1807
   pragma Import (C, nppiSubC_8u_C1RSfs, "nppiSubC_8u_C1RSfs");

  --* 
  -- * One 8-bit unsigned char channel in place image subtract constant, scale, then clamp to saturated value.
  -- * \param nConstant Constant.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSubC_8u_C1IRSfs
     (nConstant : nppdefs_h.Npp8u;
      pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:1820
   pragma Import (C, nppiSubC_8u_C1IRSfs, "nppiSubC_8u_C1IRSfs");

  --* 
  -- * Three 8-bit unsigned char channel image subtract constant, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSubC_8u_C3RSfs
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp8u;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:1834
   pragma Import (C, nppiSubC_8u_C3RSfs, "nppiSubC_8u_C3RSfs");

  --* 
  -- * Three 8-bit unsigned char channel 8-bit unsigned char in place image subtract constant, scale, then clamp to saturated value.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSubC_8u_C3IRSfs
     (aConstants : access nppdefs_h.Npp8u;
      pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:1847
   pragma Import (C, nppiSubC_8u_C3IRSfs, "nppiSubC_8u_C3IRSfs");

  --* 
  -- * Four 8-bit unsigned char channel with unmodified alpha image subtract constant, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSubC_8u_AC4RSfs
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp8u;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:1861
   pragma Import (C, nppiSubC_8u_AC4RSfs, "nppiSubC_8u_AC4RSfs");

  --* 
  -- * Four 8-bit unsigned char channel with unmodified alpha in place image subtract constant, scale, then clamp to saturated value.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSubC_8u_AC4IRSfs
     (aConstants : access nppdefs_h.Npp8u;
      pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:1874
   pragma Import (C, nppiSubC_8u_AC4IRSfs, "nppiSubC_8u_AC4IRSfs");

  --* 
  -- * Four 8-bit unsigned char channel image subtract constant, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSubC_8u_C4RSfs
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp8u;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:1888
   pragma Import (C, nppiSubC_8u_C4RSfs, "nppiSubC_8u_C4RSfs");

  --* 
  -- * Four 8-bit unsigned char channel in place image subtract constant, scale, then clamp to saturated value.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSubC_8u_C4IRSfs
     (aConstants : access nppdefs_h.Npp8u;
      pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:1901
   pragma Import (C, nppiSubC_8u_C4IRSfs, "nppiSubC_8u_C4IRSfs");

  --* 
  -- * One 16-bit unsigned short channel image subtract constant, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param nConstant Constant.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSubC_16u_C1RSfs
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      nConstant : nppdefs_h.Npp16u;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:1915
   pragma Import (C, nppiSubC_16u_C1RSfs, "nppiSubC_16u_C1RSfs");

  --* 
  -- * One 16-bit unsigned short channel in place image subtract constant, scale, then clamp to saturated value.
  -- * \param nConstant Constant.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSubC_16u_C1IRSfs
     (nConstant : nppdefs_h.Npp16u;
      pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:1928
   pragma Import (C, nppiSubC_16u_C1IRSfs, "nppiSubC_16u_C1IRSfs");

  --* 
  -- * Three 16-bit unsigned short channel image subtract constant, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSubC_16u_C3RSfs
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp16u;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:1942
   pragma Import (C, nppiSubC_16u_C3RSfs, "nppiSubC_16u_C3RSfs");

  --* 
  -- * Three 16-bit unsigned short channel in place image subtract constant, scale, then clamp to saturated value.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSubC_16u_C3IRSfs
     (aConstants : access nppdefs_h.Npp16u;
      pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:1955
   pragma Import (C, nppiSubC_16u_C3IRSfs, "nppiSubC_16u_C3IRSfs");

  --* 
  -- * Four 16-bit unsigned short channel with unmodified alpha image subtract constant, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSubC_16u_AC4RSfs
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp16u;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:1969
   pragma Import (C, nppiSubC_16u_AC4RSfs, "nppiSubC_16u_AC4RSfs");

  --* 
  -- * Four 16-bit unsigned short channel with unmodified alpha in place image subtract constant, scale, then clamp to saturated value.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSubC_16u_AC4IRSfs
     (aConstants : access nppdefs_h.Npp16u;
      pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:1982
   pragma Import (C, nppiSubC_16u_AC4IRSfs, "nppiSubC_16u_AC4IRSfs");

  --* 
  -- * Four 16-bit unsigned short channel image subtract constant, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSubC_16u_C4RSfs
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp16u;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:1996
   pragma Import (C, nppiSubC_16u_C4RSfs, "nppiSubC_16u_C4RSfs");

  --* 
  -- * Four 16-bit unsigned short channel in place image subtract constant, scale, then clamp to saturated value.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSubC_16u_C4IRSfs
     (aConstants : access nppdefs_h.Npp16u;
      pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:2009
   pragma Import (C, nppiSubC_16u_C4IRSfs, "nppiSubC_16u_C4IRSfs");

  --* 
  -- * One 16-bit signed short channel image subtract constant, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param nConstant Constant.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSubC_16s_C1RSfs
     (pSrc1 : access nppdefs_h.Npp16s;
      nSrc1Step : int;
      nConstant : nppdefs_h.Npp16s;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:2023
   pragma Import (C, nppiSubC_16s_C1RSfs, "nppiSubC_16s_C1RSfs");

  --* 
  -- * One 16-bit signed short channel in place image subtract constant, scale, then clamp to saturated value.
  -- * \param nConstant Constant.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSubC_16s_C1IRSfs
     (nConstant : nppdefs_h.Npp16s;
      pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:2036
   pragma Import (C, nppiSubC_16s_C1IRSfs, "nppiSubC_16s_C1IRSfs");

  --* 
  -- * Three 16-bit signed short channel image subtract constant, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSubC_16s_C3RSfs
     (pSrc1 : access nppdefs_h.Npp16s;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp16s;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:2050
   pragma Import (C, nppiSubC_16s_C3RSfs, "nppiSubC_16s_C3RSfs");

  --* 
  -- * Three 16-bit signed short channel in place image subtract constant, scale, then clamp to saturated value.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSubC_16s_C3IRSfs
     (aConstants : access nppdefs_h.Npp16s;
      pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:2063
   pragma Import (C, nppiSubC_16s_C3IRSfs, "nppiSubC_16s_C3IRSfs");

  --* 
  -- * Four 16-bit signed short channel with unmodified alpha image subtract constant, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSubC_16s_AC4RSfs
     (pSrc1 : access nppdefs_h.Npp16s;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp16s;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:2077
   pragma Import (C, nppiSubC_16s_AC4RSfs, "nppiSubC_16s_AC4RSfs");

  --* 
  -- * Four 16-bit signed short channel with unmodified alpha in place image subtract constant, scale, then clamp to saturated value.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSubC_16s_AC4IRSfs
     (aConstants : access nppdefs_h.Npp16s;
      pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:2090
   pragma Import (C, nppiSubC_16s_AC4IRSfs, "nppiSubC_16s_AC4IRSfs");

  --* 
  -- * Four 16-bit signed short channel image subtract constant, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSubC_16s_C4RSfs
     (pSrc1 : access nppdefs_h.Npp16s;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp16s;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:2104
   pragma Import (C, nppiSubC_16s_C4RSfs, "nppiSubC_16s_C4RSfs");

  --* 
  -- * Four 16-bit signed short channel in place image subtract constant, scale, then clamp to saturated value.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSubC_16s_C4IRSfs
     (aConstants : access nppdefs_h.Npp16s;
      pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:2117
   pragma Import (C, nppiSubC_16s_C4IRSfs, "nppiSubC_16s_C4IRSfs");

  --* 
  -- * One 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel image subtract constant, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param nConstant Constant.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSubC_16sc_C1RSfs
     (pSrc1 : access constant nppdefs_h.Npp16sc;
      nSrc1Step : int;
      nConstant : nppdefs_h.Npp16sc;
      pDst : access nppdefs_h.Npp16sc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:2131
   pragma Import (C, nppiSubC_16sc_C1RSfs, "nppiSubC_16sc_C1RSfs");

  --* 
  -- * One 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel in place image subtract constant, scale, then clamp to saturated value.
  -- * \param nConstant Constant.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSubC_16sc_C1IRSfs
     (nConstant : nppdefs_h.Npp16sc;
      pSrcDst : access nppdefs_h.Npp16sc;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:2144
   pragma Import (C, nppiSubC_16sc_C1IRSfs, "nppiSubC_16sc_C1IRSfs");

  --* 
  -- * Three 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel image subtract constant, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSubC_16sc_C3RSfs
     (pSrc1 : access constant nppdefs_h.Npp16sc;
      nSrc1Step : int;
      aConstants : access constant nppdefs_h.Npp16sc;
      pDst : access nppdefs_h.Npp16sc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:2158
   pragma Import (C, nppiSubC_16sc_C3RSfs, "nppiSubC_16sc_C3RSfs");

  --* 
  -- * Three 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel in place image subtract constant, scale, then clamp to saturated value.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSubC_16sc_C3IRSfs
     (aConstants : access constant nppdefs_h.Npp16sc;
      pSrcDst : access nppdefs_h.Npp16sc;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:2171
   pragma Import (C, nppiSubC_16sc_C3IRSfs, "nppiSubC_16sc_C3IRSfs");

  --* 
  -- * Four 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel with unmodified alpha image subtract constant, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSubC_16sc_AC4RSfs
     (pSrc1 : access constant nppdefs_h.Npp16sc;
      nSrc1Step : int;
      aConstants : access constant nppdefs_h.Npp16sc;
      pDst : access nppdefs_h.Npp16sc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:2185
   pragma Import (C, nppiSubC_16sc_AC4RSfs, "nppiSubC_16sc_AC4RSfs");

  --* 
  -- * Four 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel with unmodified alpha in place image subtract constant, scale, then clamp to saturated value.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSubC_16sc_AC4IRSfs
     (aConstants : access constant nppdefs_h.Npp16sc;
      pSrcDst : access nppdefs_h.Npp16sc;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:2198
   pragma Import (C, nppiSubC_16sc_AC4IRSfs, "nppiSubC_16sc_AC4IRSfs");

  --* 
  -- * One 32-bit signed integer channel image subtract constant, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param nConstant Constant.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSubC_32s_C1RSfs
     (pSrc1 : access nppdefs_h.Npp32s;
      nSrc1Step : int;
      nConstant : nppdefs_h.Npp32s;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:2212
   pragma Import (C, nppiSubC_32s_C1RSfs, "nppiSubC_32s_C1RSfs");

  --* 
  -- * One 32-bit signed integer channel in place image subtract constant, scale, then clamp to saturated value.
  -- * \param nConstant Constant.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSubC_32s_C1IRSfs
     (nConstant : nppdefs_h.Npp32s;
      pSrcDst : access nppdefs_h.Npp32s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:2225
   pragma Import (C, nppiSubC_32s_C1IRSfs, "nppiSubC_32s_C1IRSfs");

  --* 
  -- * Three 32-bit signed integer channel image subtract constant, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSubC_32s_C3RSfs
     (pSrc1 : access nppdefs_h.Npp32s;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp32s;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:2239
   pragma Import (C, nppiSubC_32s_C3RSfs, "nppiSubC_32s_C3RSfs");

  --* 
  -- * Three 32-bit signed integer channel in place image subtract constant, scale, then clamp to saturated value.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSubC_32s_C3IRSfs
     (aConstants : access nppdefs_h.Npp32s;
      pSrcDst : access nppdefs_h.Npp32s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:2252
   pragma Import (C, nppiSubC_32s_C3IRSfs, "nppiSubC_32s_C3IRSfs");

  --* 
  -- * One 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel image subtract constant, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param nConstant Constant.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSubC_32sc_C1RSfs
     (pSrc1 : access constant nppdefs_h.Npp32sc;
      nSrc1Step : int;
      nConstant : nppdefs_h.Npp32sc;
      pDst : access nppdefs_h.Npp32sc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:2266
   pragma Import (C, nppiSubC_32sc_C1RSfs, "nppiSubC_32sc_C1RSfs");

  --* 
  -- * One 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel in place image subtract constant, scale, then clamp to saturated value.
  -- * \param nConstant Constant.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSubC_32sc_C1IRSfs
     (nConstant : nppdefs_h.Npp32sc;
      pSrcDst : access nppdefs_h.Npp32sc;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:2279
   pragma Import (C, nppiSubC_32sc_C1IRSfs, "nppiSubC_32sc_C1IRSfs");

  --* 
  -- * Three 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel image subtract constant, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSubC_32sc_C3RSfs
     (pSrc1 : access constant nppdefs_h.Npp32sc;
      nSrc1Step : int;
      aConstants : access constant nppdefs_h.Npp32sc;
      pDst : access nppdefs_h.Npp32sc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:2293
   pragma Import (C, nppiSubC_32sc_C3RSfs, "nppiSubC_32sc_C3RSfs");

  --* 
  -- * Three 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel in place image subtract constant, scale, then clamp to saturated value.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSubC_32sc_C3IRSfs
     (aConstants : access constant nppdefs_h.Npp32sc;
      pSrcDst : access nppdefs_h.Npp32sc;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:2306
   pragma Import (C, nppiSubC_32sc_C3IRSfs, "nppiSubC_32sc_C3IRSfs");

  --* 
  -- * Four 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel with unmodified alpha image subtract constant, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSubC_32sc_AC4RSfs
     (pSrc1 : access constant nppdefs_h.Npp32sc;
      nSrc1Step : int;
      aConstants : access constant nppdefs_h.Npp32sc;
      pDst : access nppdefs_h.Npp32sc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:2320
   pragma Import (C, nppiSubC_32sc_AC4RSfs, "nppiSubC_32sc_AC4RSfs");

  --* 
  -- * Four 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel with unmodified alpha in place image subtract constant, scale, then clamp to saturated value.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSubC_32sc_AC4IRSfs
     (aConstants : access constant nppdefs_h.Npp32sc;
      pSrcDst : access nppdefs_h.Npp32sc;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:2333
   pragma Import (C, nppiSubC_32sc_AC4IRSfs, "nppiSubC_32sc_AC4IRSfs");

  --* 
  -- * One 32-bit floating point channel image subtract constant.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param nConstant Constant.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSubC_32f_C1R
     (pSrc1 : access nppdefs_h.Npp32f;
      nSrc1Step : int;
      nConstant : nppdefs_h.Npp32f;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:2346
   pragma Import (C, nppiSubC_32f_C1R, "nppiSubC_32f_C1R");

  --* 
  -- * One 32-bit floating point channel in place image subtract constant.
  -- * \param nConstant Constant.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSubC_32f_C1IR
     (nConstant : nppdefs_h.Npp32f;
      pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:2358
   pragma Import (C, nppiSubC_32f_C1IR, "nppiSubC_32f_C1IR");

  --* 
  -- * Three 32-bit floating point channel image subtract constant.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSubC_32f_C3R
     (pSrc1 : access nppdefs_h.Npp32f;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp32f;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:2371
   pragma Import (C, nppiSubC_32f_C3R, "nppiSubC_32f_C3R");

  --* 
  -- * Three 32-bit floating point channel in place image subtract constant.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSubC_32f_C3IR
     (aConstants : access nppdefs_h.Npp32f;
      pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:2383
   pragma Import (C, nppiSubC_32f_C3IR, "nppiSubC_32f_C3IR");

  --* 
  -- * Four 32-bit floating point channel with unmodified alpha image subtract constant.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSubC_32f_AC4R
     (pSrc1 : access nppdefs_h.Npp32f;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp32f;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:2396
   pragma Import (C, nppiSubC_32f_AC4R, "nppiSubC_32f_AC4R");

  --* 
  -- * Four 32-bit floating point channel with unmodified alpha in place image subtract constant.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSubC_32f_AC4IR
     (aConstants : access nppdefs_h.Npp32f;
      pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:2408
   pragma Import (C, nppiSubC_32f_AC4IR, "nppiSubC_32f_AC4IR");

  --* 
  -- * Four 32-bit floating point channel image subtract constant.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSubC_32f_C4R
     (pSrc1 : access nppdefs_h.Npp32f;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp32f;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:2421
   pragma Import (C, nppiSubC_32f_C4R, "nppiSubC_32f_C4R");

  --* 
  -- * Four 32-bit floating point channel in place image subtract constant.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSubC_32f_C4IR
     (aConstants : access nppdefs_h.Npp32f;
      pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:2433
   pragma Import (C, nppiSubC_32f_C4IR, "nppiSubC_32f_C4IR");

  --* 
  -- * One 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel image subtract constant.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param nConstant Constant.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSubC_32fc_C1R
     (pSrc1 : access constant nppdefs_h.Npp32fc;
      nSrc1Step : int;
      nConstant : nppdefs_h.Npp32fc;
      pDst : access nppdefs_h.Npp32fc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:2446
   pragma Import (C, nppiSubC_32fc_C1R, "nppiSubC_32fc_C1R");

  --* 
  -- * One 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel in place image subtract constant.
  -- * \param nConstant Constant.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSubC_32fc_C1IR
     (nConstant : nppdefs_h.Npp32fc;
      pSrcDst : access nppdefs_h.Npp32fc;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:2458
   pragma Import (C, nppiSubC_32fc_C1IR, "nppiSubC_32fc_C1IR");

  --* 
  -- * Three 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel image subtract constant.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSubC_32fc_C3R
     (pSrc1 : access constant nppdefs_h.Npp32fc;
      nSrc1Step : int;
      aConstants : access constant nppdefs_h.Npp32fc;
      pDst : access nppdefs_h.Npp32fc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:2471
   pragma Import (C, nppiSubC_32fc_C3R, "nppiSubC_32fc_C3R");

  --* 
  -- * Three 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel in place image subtract constant.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSubC_32fc_C3IR
     (aConstants : access constant nppdefs_h.Npp32fc;
      pSrcDst : access nppdefs_h.Npp32fc;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:2483
   pragma Import (C, nppiSubC_32fc_C3IR, "nppiSubC_32fc_C3IR");

  --* 
  -- * Four 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel with unmodified alpha image subtract constant.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSubC_32fc_AC4R
     (pSrc1 : access constant nppdefs_h.Npp32fc;
      nSrc1Step : int;
      aConstants : access constant nppdefs_h.Npp32fc;
      pDst : access nppdefs_h.Npp32fc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:2496
   pragma Import (C, nppiSubC_32fc_AC4R, "nppiSubC_32fc_AC4R");

  --* 
  -- * Four 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel with unmodified alpha in place image subtract constant.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSubC_32fc_AC4IR
     (aConstants : access constant nppdefs_h.Npp32fc;
      pSrcDst : access nppdefs_h.Npp32fc;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:2508
   pragma Import (C, nppiSubC_32fc_AC4IR, "nppiSubC_32fc_AC4IR");

  --* 
  -- * Four 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel image subtract constant.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSubC_32fc_C4R
     (pSrc1 : access constant nppdefs_h.Npp32fc;
      nSrc1Step : int;
      aConstants : access constant nppdefs_h.Npp32fc;
      pDst : access nppdefs_h.Npp32fc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:2521
   pragma Import (C, nppiSubC_32fc_C4R, "nppiSubC_32fc_C4R");

  --* 
  -- * Four 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel in place image subtract constant.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSubC_32fc_C4IR
     (aConstants : access constant nppdefs_h.Npp32fc;
      pSrcDst : access nppdefs_h.Npp32fc;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:2533
   pragma Import (C, nppiSubC_32fc_C4IR, "nppiSubC_32fc_C4IR");

  --* @} image_subc  
  --* 
  -- * @defgroup image_divc DivC
  -- *
  -- * Divides each pixel of an image by a constant value.
  -- *
  -- * @{
  --  

  --* 
  -- * One 8-bit unsigned char channel image divided by constant, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param nConstant Constant.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDivC_8u_C1RSfs
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      nConstant : nppdefs_h.Npp8u;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:2557
   pragma Import (C, nppiDivC_8u_C1RSfs, "nppiDivC_8u_C1RSfs");

  --* 
  -- * One 8-bit unsigned char channel in place image divided by constant, scale, then clamp to saturated value.
  -- * \param nConstant Constant.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDivC_8u_C1IRSfs
     (nConstant : nppdefs_h.Npp8u;
      pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:2570
   pragma Import (C, nppiDivC_8u_C1IRSfs, "nppiDivC_8u_C1IRSfs");

  --* 
  -- * Three 8-bit unsigned char channel image divided by constant, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDivC_8u_C3RSfs
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp8u;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:2584
   pragma Import (C, nppiDivC_8u_C3RSfs, "nppiDivC_8u_C3RSfs");

  --* 
  -- * Three 8-bit unsigned char channel 8-bit unsigned char in place image divided by constant, scale, then clamp to saturated value.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDivC_8u_C3IRSfs
     (aConstants : access nppdefs_h.Npp8u;
      pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:2597
   pragma Import (C, nppiDivC_8u_C3IRSfs, "nppiDivC_8u_C3IRSfs");

  --* 
  -- * Four 8-bit unsigned char channel with unmodified alpha image divided by constant, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDivC_8u_AC4RSfs
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp8u;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:2611
   pragma Import (C, nppiDivC_8u_AC4RSfs, "nppiDivC_8u_AC4RSfs");

  --* 
  -- * Four 8-bit unsigned char channel with unmodified alpha in place image divided by constant, scale, then clamp to saturated value.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDivC_8u_AC4IRSfs
     (aConstants : access nppdefs_h.Npp8u;
      pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:2624
   pragma Import (C, nppiDivC_8u_AC4IRSfs, "nppiDivC_8u_AC4IRSfs");

  --* 
  -- * Four 8-bit unsigned char channel image divided by constant, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDivC_8u_C4RSfs
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp8u;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:2638
   pragma Import (C, nppiDivC_8u_C4RSfs, "nppiDivC_8u_C4RSfs");

  --* 
  -- * Four 8-bit unsigned char channel in place image divided by constant, scale, then clamp to saturated value.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDivC_8u_C4IRSfs
     (aConstants : access nppdefs_h.Npp8u;
      pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:2651
   pragma Import (C, nppiDivC_8u_C4IRSfs, "nppiDivC_8u_C4IRSfs");

  --* 
  -- * One 16-bit unsigned short channel image divided by constant, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param nConstant Constant.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDivC_16u_C1RSfs
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      nConstant : nppdefs_h.Npp16u;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:2665
   pragma Import (C, nppiDivC_16u_C1RSfs, "nppiDivC_16u_C1RSfs");

  --* 
  -- * One 16-bit unsigned short channel in place image divided by constant, scale, then clamp to saturated value.
  -- * \param nConstant Constant.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDivC_16u_C1IRSfs
     (nConstant : nppdefs_h.Npp16u;
      pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:2678
   pragma Import (C, nppiDivC_16u_C1IRSfs, "nppiDivC_16u_C1IRSfs");

  --* 
  -- * Three 16-bit unsigned short channel image divided by constant, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDivC_16u_C3RSfs
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp16u;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:2692
   pragma Import (C, nppiDivC_16u_C3RSfs, "nppiDivC_16u_C3RSfs");

  --* 
  -- * Three 16-bit unsigned short channel in place image divided by constant, scale, then clamp to saturated value.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDivC_16u_C3IRSfs
     (aConstants : access nppdefs_h.Npp16u;
      pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:2705
   pragma Import (C, nppiDivC_16u_C3IRSfs, "nppiDivC_16u_C3IRSfs");

  --* 
  -- * Four 16-bit unsigned short channel with unmodified alpha image divided by constant, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDivC_16u_AC4RSfs
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp16u;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:2719
   pragma Import (C, nppiDivC_16u_AC4RSfs, "nppiDivC_16u_AC4RSfs");

  --* 
  -- * Four 16-bit unsigned short channel with unmodified alpha in place image divided by constant, scale, then clamp to saturated value.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDivC_16u_AC4IRSfs
     (aConstants : access nppdefs_h.Npp16u;
      pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:2732
   pragma Import (C, nppiDivC_16u_AC4IRSfs, "nppiDivC_16u_AC4IRSfs");

  --* 
  -- * Four 16-bit unsigned short channel image divided by constant, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDivC_16u_C4RSfs
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp16u;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:2746
   pragma Import (C, nppiDivC_16u_C4RSfs, "nppiDivC_16u_C4RSfs");

  --* 
  -- * Four 16-bit unsigned short channel in place image divided by constant, scale, then clamp to saturated value.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDivC_16u_C4IRSfs
     (aConstants : access nppdefs_h.Npp16u;
      pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:2759
   pragma Import (C, nppiDivC_16u_C4IRSfs, "nppiDivC_16u_C4IRSfs");

  --* 
  -- * One 16-bit signed short channel image divided by constant, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param nConstant Constant.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDivC_16s_C1RSfs
     (pSrc1 : access nppdefs_h.Npp16s;
      nSrc1Step : int;
      nConstant : nppdefs_h.Npp16s;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:2773
   pragma Import (C, nppiDivC_16s_C1RSfs, "nppiDivC_16s_C1RSfs");

  --* 
  -- * One 16-bit signed short channel in place image divided by constant, scale, then clamp to saturated value.
  -- * \param nConstant Constant.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDivC_16s_C1IRSfs
     (nConstant : nppdefs_h.Npp16s;
      pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:2786
   pragma Import (C, nppiDivC_16s_C1IRSfs, "nppiDivC_16s_C1IRSfs");

  --* 
  -- * Three 16-bit signed short channel image divided by constant, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDivC_16s_C3RSfs
     (pSrc1 : access nppdefs_h.Npp16s;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp16s;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:2800
   pragma Import (C, nppiDivC_16s_C3RSfs, "nppiDivC_16s_C3RSfs");

  --* 
  -- * Three 16-bit signed short channel in place image divided by constant, scale, then clamp to saturated value.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDivC_16s_C3IRSfs
     (aConstants : access nppdefs_h.Npp16s;
      pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:2813
   pragma Import (C, nppiDivC_16s_C3IRSfs, "nppiDivC_16s_C3IRSfs");

  --* 
  -- * Four 16-bit signed short channel with unmodified alpha image divided by constant, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDivC_16s_AC4RSfs
     (pSrc1 : access nppdefs_h.Npp16s;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp16s;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:2827
   pragma Import (C, nppiDivC_16s_AC4RSfs, "nppiDivC_16s_AC4RSfs");

  --* 
  -- * Four 16-bit signed short channel with unmodified alpha in place image divided by constant, scale, then clamp to saturated value.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDivC_16s_AC4IRSfs
     (aConstants : access nppdefs_h.Npp16s;
      pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:2840
   pragma Import (C, nppiDivC_16s_AC4IRSfs, "nppiDivC_16s_AC4IRSfs");

  --* 
  -- * Four 16-bit signed short channel image divided by constant, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDivC_16s_C4RSfs
     (pSrc1 : access nppdefs_h.Npp16s;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp16s;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:2854
   pragma Import (C, nppiDivC_16s_C4RSfs, "nppiDivC_16s_C4RSfs");

  --* 
  -- * Four 16-bit signed short channel in place image divided by constant, scale, then clamp to saturated value.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDivC_16s_C4IRSfs
     (aConstants : access nppdefs_h.Npp16s;
      pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:2867
   pragma Import (C, nppiDivC_16s_C4IRSfs, "nppiDivC_16s_C4IRSfs");

  --* 
  -- * One 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel image divided by constant, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param nConstant Constant.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDivC_16sc_C1RSfs
     (pSrc1 : access constant nppdefs_h.Npp16sc;
      nSrc1Step : int;
      nConstant : nppdefs_h.Npp16sc;
      pDst : access nppdefs_h.Npp16sc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:2881
   pragma Import (C, nppiDivC_16sc_C1RSfs, "nppiDivC_16sc_C1RSfs");

  --* 
  -- * One 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel in place image divided by constant, scale, then clamp to saturated value.
  -- * \param nConstant Constant.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDivC_16sc_C1IRSfs
     (nConstant : nppdefs_h.Npp16sc;
      pSrcDst : access nppdefs_h.Npp16sc;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:2894
   pragma Import (C, nppiDivC_16sc_C1IRSfs, "nppiDivC_16sc_C1IRSfs");

  --* 
  -- * Three 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel image divided by constant, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDivC_16sc_C3RSfs
     (pSrc1 : access constant nppdefs_h.Npp16sc;
      nSrc1Step : int;
      aConstants : access constant nppdefs_h.Npp16sc;
      pDst : access nppdefs_h.Npp16sc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:2908
   pragma Import (C, nppiDivC_16sc_C3RSfs, "nppiDivC_16sc_C3RSfs");

  --* 
  -- * Three 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel in place image divided by constant, scale, then clamp to saturated value.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDivC_16sc_C3IRSfs
     (aConstants : access constant nppdefs_h.Npp16sc;
      pSrcDst : access nppdefs_h.Npp16sc;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:2921
   pragma Import (C, nppiDivC_16sc_C3IRSfs, "nppiDivC_16sc_C3IRSfs");

  --* 
  -- * Four 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel with unmodified alpha image divided by constant, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDivC_16sc_AC4RSfs
     (pSrc1 : access constant nppdefs_h.Npp16sc;
      nSrc1Step : int;
      aConstants : access constant nppdefs_h.Npp16sc;
      pDst : access nppdefs_h.Npp16sc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:2935
   pragma Import (C, nppiDivC_16sc_AC4RSfs, "nppiDivC_16sc_AC4RSfs");

  --* 
  -- * Four 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel with unmodified alpha in place image divided by constant, scale, then clamp to saturated value.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDivC_16sc_AC4IRSfs
     (aConstants : access constant nppdefs_h.Npp16sc;
      pSrcDst : access nppdefs_h.Npp16sc;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:2948
   pragma Import (C, nppiDivC_16sc_AC4IRSfs, "nppiDivC_16sc_AC4IRSfs");

  --* 
  -- * One 32-bit signed integer channel image divided by constant, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param nConstant Constant.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDivC_32s_C1RSfs
     (pSrc1 : access nppdefs_h.Npp32s;
      nSrc1Step : int;
      nConstant : nppdefs_h.Npp32s;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:2962
   pragma Import (C, nppiDivC_32s_C1RSfs, "nppiDivC_32s_C1RSfs");

  --* 
  -- * One 32-bit signed integer channel in place image divided by constant, scale, then clamp to saturated value.
  -- * \param nConstant Constant.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDivC_32s_C1IRSfs
     (nConstant : nppdefs_h.Npp32s;
      pSrcDst : access nppdefs_h.Npp32s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:2975
   pragma Import (C, nppiDivC_32s_C1IRSfs, "nppiDivC_32s_C1IRSfs");

  --* 
  -- * Three 32-bit signed integer channel image divided by constant, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDivC_32s_C3RSfs
     (pSrc1 : access nppdefs_h.Npp32s;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp32s;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:2989
   pragma Import (C, nppiDivC_32s_C3RSfs, "nppiDivC_32s_C3RSfs");

  --* 
  -- * Three 32-bit signed integer channel in place image divided by constant, scale, then clamp to saturated value.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDivC_32s_C3IRSfs
     (aConstants : access nppdefs_h.Npp32s;
      pSrcDst : access nppdefs_h.Npp32s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:3002
   pragma Import (C, nppiDivC_32s_C3IRSfs, "nppiDivC_32s_C3IRSfs");

  --* 
  -- * One 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel image divided by constant, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param nConstant Constant.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDivC_32sc_C1RSfs
     (pSrc1 : access constant nppdefs_h.Npp32sc;
      nSrc1Step : int;
      nConstant : nppdefs_h.Npp32sc;
      pDst : access nppdefs_h.Npp32sc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:3016
   pragma Import (C, nppiDivC_32sc_C1RSfs, "nppiDivC_32sc_C1RSfs");

  --* 
  -- * One 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel in place image divided by constant, scale, then clamp to saturated value.
  -- * \param nConstant Constant.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDivC_32sc_C1IRSfs
     (nConstant : nppdefs_h.Npp32sc;
      pSrcDst : access nppdefs_h.Npp32sc;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:3029
   pragma Import (C, nppiDivC_32sc_C1IRSfs, "nppiDivC_32sc_C1IRSfs");

  --* 
  -- * Three 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel image divided by constant, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDivC_32sc_C3RSfs
     (pSrc1 : access constant nppdefs_h.Npp32sc;
      nSrc1Step : int;
      aConstants : access constant nppdefs_h.Npp32sc;
      pDst : access nppdefs_h.Npp32sc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:3043
   pragma Import (C, nppiDivC_32sc_C3RSfs, "nppiDivC_32sc_C3RSfs");

  --* 
  -- * Three 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel in place image divided by constant, scale, then clamp to saturated value.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDivC_32sc_C3IRSfs
     (aConstants : access constant nppdefs_h.Npp32sc;
      pSrcDst : access nppdefs_h.Npp32sc;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:3056
   pragma Import (C, nppiDivC_32sc_C3IRSfs, "nppiDivC_32sc_C3IRSfs");

  --* 
  -- * Four 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel with unmodified alpha image divided by constant, scale, then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDivC_32sc_AC4RSfs
     (pSrc1 : access constant nppdefs_h.Npp32sc;
      nSrc1Step : int;
      aConstants : access constant nppdefs_h.Npp32sc;
      pDst : access nppdefs_h.Npp32sc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:3070
   pragma Import (C, nppiDivC_32sc_AC4RSfs, "nppiDivC_32sc_AC4RSfs");

  --* 
  -- * Four 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel with unmodified alpha in place image divided by constant, scale, then clamp to saturated value.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDivC_32sc_AC4IRSfs
     (aConstants : access constant nppdefs_h.Npp32sc;
      pSrcDst : access nppdefs_h.Npp32sc;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:3083
   pragma Import (C, nppiDivC_32sc_AC4IRSfs, "nppiDivC_32sc_AC4IRSfs");

  --* 
  -- * One 32-bit floating point channel image divided by constant.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param nConstant Constant.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDivC_32f_C1R
     (pSrc1 : access nppdefs_h.Npp32f;
      nSrc1Step : int;
      nConstant : nppdefs_h.Npp32f;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:3096
   pragma Import (C, nppiDivC_32f_C1R, "nppiDivC_32f_C1R");

  --* 
  -- * One 32-bit floating point channel in place image divided by constant.
  -- * \param nConstant Constant.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDivC_32f_C1IR
     (nConstant : nppdefs_h.Npp32f;
      pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:3108
   pragma Import (C, nppiDivC_32f_C1IR, "nppiDivC_32f_C1IR");

  --* 
  -- * Three 32-bit floating point channel image divided by constant.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDivC_32f_C3R
     (pSrc1 : access nppdefs_h.Npp32f;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp32f;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:3121
   pragma Import (C, nppiDivC_32f_C3R, "nppiDivC_32f_C3R");

  --* 
  -- * Three 32-bit floating point channel in place image divided by constant.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDivC_32f_C3IR
     (aConstants : access nppdefs_h.Npp32f;
      pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:3133
   pragma Import (C, nppiDivC_32f_C3IR, "nppiDivC_32f_C3IR");

  --* 
  -- * Four 32-bit floating point channel with unmodified alpha image divided by constant.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDivC_32f_AC4R
     (pSrc1 : access nppdefs_h.Npp32f;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp32f;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:3146
   pragma Import (C, nppiDivC_32f_AC4R, "nppiDivC_32f_AC4R");

  --* 
  -- * Four 32-bit floating point channel with unmodified alpha in place image divided by constant.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDivC_32f_AC4IR
     (aConstants : access nppdefs_h.Npp32f;
      pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:3158
   pragma Import (C, nppiDivC_32f_AC4IR, "nppiDivC_32f_AC4IR");

  --* 
  -- * Four 32-bit floating point channel image divided by constant.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDivC_32f_C4R
     (pSrc1 : access nppdefs_h.Npp32f;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp32f;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:3171
   pragma Import (C, nppiDivC_32f_C4R, "nppiDivC_32f_C4R");

  --* 
  -- * Four 32-bit floating point channel in place image divided by constant.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDivC_32f_C4IR
     (aConstants : access nppdefs_h.Npp32f;
      pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:3183
   pragma Import (C, nppiDivC_32f_C4IR, "nppiDivC_32f_C4IR");

  --* 
  -- * One 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel image divided by constant.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param nConstant Constant.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDivC_32fc_C1R
     (pSrc1 : access constant nppdefs_h.Npp32fc;
      nSrc1Step : int;
      nConstant : nppdefs_h.Npp32fc;
      pDst : access nppdefs_h.Npp32fc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:3196
   pragma Import (C, nppiDivC_32fc_C1R, "nppiDivC_32fc_C1R");

  --* 
  -- * One 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel in place image divided by constant.
  -- * \param nConstant Constant.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDivC_32fc_C1IR
     (nConstant : nppdefs_h.Npp32fc;
      pSrcDst : access nppdefs_h.Npp32fc;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:3208
   pragma Import (C, nppiDivC_32fc_C1IR, "nppiDivC_32fc_C1IR");

  --* 
  -- * Three 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel image divided by constant.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDivC_32fc_C3R
     (pSrc1 : access constant nppdefs_h.Npp32fc;
      nSrc1Step : int;
      aConstants : access constant nppdefs_h.Npp32fc;
      pDst : access nppdefs_h.Npp32fc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:3221
   pragma Import (C, nppiDivC_32fc_C3R, "nppiDivC_32fc_C3R");

  --* 
  -- * Three 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel in place image divided by constant.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDivC_32fc_C3IR
     (aConstants : access constant nppdefs_h.Npp32fc;
      pSrcDst : access nppdefs_h.Npp32fc;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:3233
   pragma Import (C, nppiDivC_32fc_C3IR, "nppiDivC_32fc_C3IR");

  --* 
  -- * Four 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel with unmodified alpha image divided by constant.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDivC_32fc_AC4R
     (pSrc1 : access constant nppdefs_h.Npp32fc;
      nSrc1Step : int;
      aConstants : access constant nppdefs_h.Npp32fc;
      pDst : access nppdefs_h.Npp32fc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:3246
   pragma Import (C, nppiDivC_32fc_AC4R, "nppiDivC_32fc_AC4R");

  --* 
  -- * Four 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel with unmodified alpha in place image divided by constant.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDivC_32fc_AC4IR
     (aConstants : access constant nppdefs_h.Npp32fc;
      pSrcDst : access nppdefs_h.Npp32fc;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:3258
   pragma Import (C, nppiDivC_32fc_AC4IR, "nppiDivC_32fc_AC4IR");

  --* 
  -- * Four 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel image divided by constant.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDivC_32fc_C4R
     (pSrc1 : access constant nppdefs_h.Npp32fc;
      nSrc1Step : int;
      aConstants : access constant nppdefs_h.Npp32fc;
      pDst : access nppdefs_h.Npp32fc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:3271
   pragma Import (C, nppiDivC_32fc_C4R, "nppiDivC_32fc_C4R");

  --* 
  -- * Four 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel in place image divided by constant.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDivC_32fc_C4IR
     (aConstants : access constant nppdefs_h.Npp32fc;
      pSrcDst : access nppdefs_h.Npp32fc;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:3283
   pragma Import (C, nppiDivC_32fc_C4IR, "nppiDivC_32fc_C4IR");

  --* @} image_divc  
  --* 
  -- * @defgroup image_absdiffc AbsDiffC
  -- *
  -- * Determines absolute difference between each pixel of an image and a constant value.
  -- *
  -- * @{
  --  

  --* 
  -- * One 8-bit unsigned char channel image absolute difference with constant.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param nConstant Constant.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAbsDiffC_8u_C1R
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nConstant : nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:3306
   pragma Import (C, nppiAbsDiffC_8u_C1R, "nppiAbsDiffC_8u_C1R");

  --* 
  -- * One 16-bit unsigned short channel image absolute difference with constant.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param nConstant Constant.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAbsDiffC_16u_C1R
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nConstant : nppdefs_h.Npp16u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:3319
   pragma Import (C, nppiAbsDiffC_16u_C1R, "nppiAbsDiffC_16u_C1R");

  --* 
  -- * One 32-bit floating point channel image absolute difference with constant.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param nConstant Constant.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAbsDiffC_32f_C1R
     (pSrc1 : access nppdefs_h.Npp32f;
      nSrc1Step : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nConstant : nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:3332
   pragma Import (C, nppiAbsDiffC_32f_C1R, "nppiAbsDiffC_32f_C1R");

  --* @} image_absdiffc  
  --* 
  -- * @defgroup image_add Add
  -- *
  -- * Pixel by pixel addition of two images.
  -- *
  -- * @{
  --  

  --* 
  -- * One 8-bit unsigned char channel image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAdd_8u_C1RSfs
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp8u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:3357
   pragma Import (C, nppiAdd_8u_C1RSfs, "nppiAdd_8u_C1RSfs");

  --* 
  -- * One 8-bit unsigned char channel in place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAdd_8u_C1IRSfs
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:3371
   pragma Import (C, nppiAdd_8u_C1IRSfs, "nppiAdd_8u_C1IRSfs");

  --* 
  -- * Three 8-bit unsigned char channel image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAdd_8u_C3RSfs
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp8u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:3387
   pragma Import (C, nppiAdd_8u_C3RSfs, "nppiAdd_8u_C3RSfs");

  --* 
  -- * Three 8-bit unsigned char channel in place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAdd_8u_C3IRSfs
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:3401
   pragma Import (C, nppiAdd_8u_C3IRSfs, "nppiAdd_8u_C3IRSfs");

  --* 
  -- * Four 8-bit unsigned char channel with unmodified alpha image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAdd_8u_AC4RSfs
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp8u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:3417
   pragma Import (C, nppiAdd_8u_AC4RSfs, "nppiAdd_8u_AC4RSfs");

  --* 
  -- * Four 8-bit unsigned char channel with unmodified alpha in place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAdd_8u_AC4IRSfs
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:3431
   pragma Import (C, nppiAdd_8u_AC4IRSfs, "nppiAdd_8u_AC4IRSfs");

  --* 
  -- * Four 8-bit unsigned char channel image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAdd_8u_C4RSfs
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp8u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:3447
   pragma Import (C, nppiAdd_8u_C4RSfs, "nppiAdd_8u_C4RSfs");

  --* 
  -- * Four 8-bit unsigned char channel in place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAdd_8u_C4IRSfs
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:3461
   pragma Import (C, nppiAdd_8u_C4IRSfs, "nppiAdd_8u_C4IRSfs");

  --* 
  -- * One 16-bit unsigned short channel image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAdd_16u_C1RSfs
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp16u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:3477
   pragma Import (C, nppiAdd_16u_C1RSfs, "nppiAdd_16u_C1RSfs");

  --* 
  -- * One 16-bit unsigned short channel in place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAdd_16u_C1IRSfs
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:3491
   pragma Import (C, nppiAdd_16u_C1IRSfs, "nppiAdd_16u_C1IRSfs");

  --* 
  -- * Three 16-bit unsigned short channel image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAdd_16u_C3RSfs
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp16u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:3507
   pragma Import (C, nppiAdd_16u_C3RSfs, "nppiAdd_16u_C3RSfs");

  --* 
  -- * Three 16-bit unsigned short channel in place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAdd_16u_C3IRSfs
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:3521
   pragma Import (C, nppiAdd_16u_C3IRSfs, "nppiAdd_16u_C3IRSfs");

  --* 
  -- * Four 16-bit unsigned short channel with unmodified alpha image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAdd_16u_AC4RSfs
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp16u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:3537
   pragma Import (C, nppiAdd_16u_AC4RSfs, "nppiAdd_16u_AC4RSfs");

  --* 
  -- * Four 16-bit unsigned short channel with unmodified alpha in place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAdd_16u_AC4IRSfs
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:3551
   pragma Import (C, nppiAdd_16u_AC4IRSfs, "nppiAdd_16u_AC4IRSfs");

  --* 
  -- * Four 16-bit unsigned short channel image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAdd_16u_C4RSfs
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp16u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:3567
   pragma Import (C, nppiAdd_16u_C4RSfs, "nppiAdd_16u_C4RSfs");

  --* 
  -- * Four 16-bit unsigned short channel in place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAdd_16u_C4IRSfs
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:3581
   pragma Import (C, nppiAdd_16u_C4IRSfs, "nppiAdd_16u_C4IRSfs");

  --* 
  -- * One 16-bit signed short channel image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAdd_16s_C1RSfs
     (pSrc1 : access nppdefs_h.Npp16s;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp16s;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:3597
   pragma Import (C, nppiAdd_16s_C1RSfs, "nppiAdd_16s_C1RSfs");

  --* 
  -- * One 16-bit signed short channel in place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAdd_16s_C1IRSfs
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:3611
   pragma Import (C, nppiAdd_16s_C1IRSfs, "nppiAdd_16s_C1IRSfs");

  --* 
  -- * Three 16-bit signed short channel image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAdd_16s_C3RSfs
     (pSrc1 : access nppdefs_h.Npp16s;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp16s;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:3627
   pragma Import (C, nppiAdd_16s_C3RSfs, "nppiAdd_16s_C3RSfs");

  --* 
  -- * Three 16-bit signed short channel in place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAdd_16s_C3IRSfs
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:3641
   pragma Import (C, nppiAdd_16s_C3IRSfs, "nppiAdd_16s_C3IRSfs");

  --* 
  -- * Four 16-bit signed short channel with unmodified alpha image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAdd_16s_AC4RSfs
     (pSrc1 : access nppdefs_h.Npp16s;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp16s;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:3657
   pragma Import (C, nppiAdd_16s_AC4RSfs, "nppiAdd_16s_AC4RSfs");

  --* 
  -- * Four 16-bit signed short channel with unmodified alpha in place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAdd_16s_AC4IRSfs
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:3671
   pragma Import (C, nppiAdd_16s_AC4IRSfs, "nppiAdd_16s_AC4IRSfs");

  --* 
  -- * Four 16-bit signed short channel image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAdd_16s_C4RSfs
     (pSrc1 : access nppdefs_h.Npp16s;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp16s;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:3687
   pragma Import (C, nppiAdd_16s_C4RSfs, "nppiAdd_16s_C4RSfs");

  --* 
  -- * Four 16-bit signed short channel in place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAdd_16s_C4IRSfs
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:3701
   pragma Import (C, nppiAdd_16s_C4IRSfs, "nppiAdd_16s_C4IRSfs");

  --* 
  -- * One 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAdd_16sc_C1RSfs
     (pSrc1 : access constant nppdefs_h.Npp16sc;
      nSrc1Step : int;
      pSrc2 : access constant nppdefs_h.Npp16sc;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp16sc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:3717
   pragma Import (C, nppiAdd_16sc_C1RSfs, "nppiAdd_16sc_C1RSfs");

  --* 
  -- * One 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel in place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAdd_16sc_C1IRSfs
     (pSrc : access constant nppdefs_h.Npp16sc;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp16sc;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:3731
   pragma Import (C, nppiAdd_16sc_C1IRSfs, "nppiAdd_16sc_C1IRSfs");

  --* 
  -- * Three 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAdd_16sc_C3RSfs
     (pSrc1 : access constant nppdefs_h.Npp16sc;
      nSrc1Step : int;
      pSrc2 : access constant nppdefs_h.Npp16sc;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp16sc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:3747
   pragma Import (C, nppiAdd_16sc_C3RSfs, "nppiAdd_16sc_C3RSfs");

  --* 
  -- * Three 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel in place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAdd_16sc_C3IRSfs
     (pSrc : access constant nppdefs_h.Npp16sc;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp16sc;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:3761
   pragma Import (C, nppiAdd_16sc_C3IRSfs, "nppiAdd_16sc_C3IRSfs");

  --* 
  -- * Four 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel with unmodified alpha image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAdd_16sc_AC4RSfs
     (pSrc1 : access constant nppdefs_h.Npp16sc;
      nSrc1Step : int;
      pSrc2 : access constant nppdefs_h.Npp16sc;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp16sc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:3777
   pragma Import (C, nppiAdd_16sc_AC4RSfs, "nppiAdd_16sc_AC4RSfs");

  --* 
  -- * Four 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel with unmodified alpha in place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAdd_16sc_AC4IRSfs
     (pSrc : access constant nppdefs_h.Npp16sc;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp16sc;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:3791
   pragma Import (C, nppiAdd_16sc_AC4IRSfs, "nppiAdd_16sc_AC4IRSfs");

  --* 
  -- * One 32-bit signed integer channel image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAdd_32s_C1RSfs
     (pSrc1 : access nppdefs_h.Npp32s;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp32s;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:3807
   pragma Import (C, nppiAdd_32s_C1RSfs, "nppiAdd_32s_C1RSfs");

  --*
  -- * Note: This function is to be deprecated in future NPP releases, use the function above with a scale factor of 0 instead. 
  -- * 32-bit image add.
  -- * Add the pixel values of corresponding pixels in the ROI and write them to the output image.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAdd_32s_C1R
     (pSrc1 : access nppdefs_h.Npp32s;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp32s;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:3823
   pragma Import (C, nppiAdd_32s_C1R, "nppiAdd_32s_C1R");

  --* 
  -- * One 32-bit signed integer channel in place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAdd_32s_C1IRSfs
     (pSrc : access nppdefs_h.Npp32s;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp32s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:3839
   pragma Import (C, nppiAdd_32s_C1IRSfs, "nppiAdd_32s_C1IRSfs");

  --* 
  -- * Three 32-bit signed integer channel image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAdd_32s_C3RSfs
     (pSrc1 : access nppdefs_h.Npp32s;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp32s;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:3855
   pragma Import (C, nppiAdd_32s_C3RSfs, "nppiAdd_32s_C3RSfs");

  --* 
  -- * Three 32-bit signed integer channel in place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAdd_32s_C3IRSfs
     (pSrc : access nppdefs_h.Npp32s;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp32s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:3869
   pragma Import (C, nppiAdd_32s_C3IRSfs, "nppiAdd_32s_C3IRSfs");

  --* 
  -- * One 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAdd_32sc_C1RSfs
     (pSrc1 : access constant nppdefs_h.Npp32sc;
      nSrc1Step : int;
      pSrc2 : access constant nppdefs_h.Npp32sc;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp32sc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:3885
   pragma Import (C, nppiAdd_32sc_C1RSfs, "nppiAdd_32sc_C1RSfs");

  --* 
  -- * One 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel in place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAdd_32sc_C1IRSfs
     (pSrc : access constant nppdefs_h.Npp32sc;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp32sc;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:3899
   pragma Import (C, nppiAdd_32sc_C1IRSfs, "nppiAdd_32sc_C1IRSfs");

  --* 
  -- * Three 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAdd_32sc_C3RSfs
     (pSrc1 : access constant nppdefs_h.Npp32sc;
      nSrc1Step : int;
      pSrc2 : access constant nppdefs_h.Npp32sc;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp32sc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:3915
   pragma Import (C, nppiAdd_32sc_C3RSfs, "nppiAdd_32sc_C3RSfs");

  --* 
  -- * Three 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel in place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAdd_32sc_C3IRSfs
     (pSrc : access constant nppdefs_h.Npp32sc;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp32sc;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:3929
   pragma Import (C, nppiAdd_32sc_C3IRSfs, "nppiAdd_32sc_C3IRSfs");

  --* 
  -- * Four 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel with unmodified alpha image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAdd_32sc_AC4RSfs
     (pSrc1 : access constant nppdefs_h.Npp32sc;
      nSrc1Step : int;
      pSrc2 : access constant nppdefs_h.Npp32sc;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp32sc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:3945
   pragma Import (C, nppiAdd_32sc_AC4RSfs, "nppiAdd_32sc_AC4RSfs");

  --* 
  -- * Four 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel with unmodified alpha in place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAdd_32sc_AC4IRSfs
     (pSrc : access constant nppdefs_h.Npp32sc;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp32sc;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:3959
   pragma Import (C, nppiAdd_32sc_AC4IRSfs, "nppiAdd_32sc_AC4IRSfs");

  --* 
  -- * One 32-bit floating point channel image addition.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAdd_32f_C1R
     (pSrc1 : access nppdefs_h.Npp32f;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp32f;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:3974
   pragma Import (C, nppiAdd_32f_C1R, "nppiAdd_32f_C1R");

  --* 
  -- * One 32-bit floating point channel in place image addition.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAdd_32f_C1IR
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:3987
   pragma Import (C, nppiAdd_32f_C1IR, "nppiAdd_32f_C1IR");

  --* 
  -- * Three 32-bit floating point channel image addition.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAdd_32f_C3R
     (pSrc1 : access nppdefs_h.Npp32f;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp32f;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:4002
   pragma Import (C, nppiAdd_32f_C3R, "nppiAdd_32f_C3R");

  --* 
  -- * Three 32-bit floating point channel in place image addition.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAdd_32f_C3IR
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:4015
   pragma Import (C, nppiAdd_32f_C3IR, "nppiAdd_32f_C3IR");

  --* 
  -- * Four 32-bit floating point channel with unmodified alpha image addition.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAdd_32f_AC4R
     (pSrc1 : access nppdefs_h.Npp32f;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp32f;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:4030
   pragma Import (C, nppiAdd_32f_AC4R, "nppiAdd_32f_AC4R");

  --* 
  -- * Four 32-bit floating point channel with unmodified alpha in place image addition.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAdd_32f_AC4IR
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:4043
   pragma Import (C, nppiAdd_32f_AC4IR, "nppiAdd_32f_AC4IR");

  --* 
  -- * Four 32-bit floating point channel image addition.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAdd_32f_C4R
     (pSrc1 : access nppdefs_h.Npp32f;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp32f;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:4058
   pragma Import (C, nppiAdd_32f_C4R, "nppiAdd_32f_C4R");

  --* 
  -- * Four 32-bit floating point channel in place image addition.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAdd_32f_C4IR
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:4071
   pragma Import (C, nppiAdd_32f_C4IR, "nppiAdd_32f_C4IR");

  --* 
  -- * One 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel image addition.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAdd_32fc_C1R
     (pSrc1 : access constant nppdefs_h.Npp32fc;
      nSrc1Step : int;
      pSrc2 : access constant nppdefs_h.Npp32fc;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp32fc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:4086
   pragma Import (C, nppiAdd_32fc_C1R, "nppiAdd_32fc_C1R");

  --* 
  -- * One 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel in place image addition.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAdd_32fc_C1IR
     (pSrc : access constant nppdefs_h.Npp32fc;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp32fc;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:4099
   pragma Import (C, nppiAdd_32fc_C1IR, "nppiAdd_32fc_C1IR");

  --* 
  -- * Three 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel image addition.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAdd_32fc_C3R
     (pSrc1 : access constant nppdefs_h.Npp32fc;
      nSrc1Step : int;
      pSrc2 : access constant nppdefs_h.Npp32fc;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp32fc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:4114
   pragma Import (C, nppiAdd_32fc_C3R, "nppiAdd_32fc_C3R");

  --* 
  -- * Three 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel in place image addition.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAdd_32fc_C3IR
     (pSrc : access constant nppdefs_h.Npp32fc;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp32fc;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:4127
   pragma Import (C, nppiAdd_32fc_C3IR, "nppiAdd_32fc_C3IR");

  --* 
  -- * Four 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel with unmodified alpha image addition.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAdd_32fc_AC4R
     (pSrc1 : access constant nppdefs_h.Npp32fc;
      nSrc1Step : int;
      pSrc2 : access constant nppdefs_h.Npp32fc;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp32fc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:4142
   pragma Import (C, nppiAdd_32fc_AC4R, "nppiAdd_32fc_AC4R");

  --* 
  -- * Four 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel with unmodified alpha in place image addition.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAdd_32fc_AC4IR
     (pSrc : access constant nppdefs_h.Npp32fc;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp32fc;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:4155
   pragma Import (C, nppiAdd_32fc_AC4IR, "nppiAdd_32fc_AC4IR");

  --* 
  -- * Four 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel image addition.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAdd_32fc_C4R
     (pSrc1 : access constant nppdefs_h.Npp32fc;
      nSrc1Step : int;
      pSrc2 : access constant nppdefs_h.Npp32fc;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp32fc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:4170
   pragma Import (C, nppiAdd_32fc_C4R, "nppiAdd_32fc_C4R");

  --* 
  -- * Four 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel in place image addition.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAdd_32fc_C4IR
     (pSrc : access constant nppdefs_h.Npp32fc;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp32fc;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:4183
   pragma Import (C, nppiAdd_32fc_C4IR, "nppiAdd_32fc_C4IR");

  --* @} image_add  
  --* 
  -- * @defgroup image_addsquare AddSquare
  -- *
  -- * Pixel by pixel addition of squared pixels from source image to floating point
  -- * pixel values of destination image.
  -- *
  -- * @{
  --  

  --* 
  -- * One 8-bit unsigned char channel image squared then added to in place floating point destination image using filter mask (updates destination when mask is non-zero).
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pMask \ref mask_image_pointer.
  -- * \param nMaskStep \ref mask_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAddSquare_8u32f_C1IMR
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pMask : access nppdefs_h.Npp8u;
      nMaskStep : int;
      pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:4209
   pragma Import (C, nppiAddSquare_8u32f_C1IMR, "nppiAddSquare_8u32f_C1IMR");

  --* 
  -- * One 8-bit unsigned char channel image squared then added to in place floating point destination image.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAddSquare_8u32f_C1IR
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:4222
   pragma Import (C, nppiAddSquare_8u32f_C1IR, "nppiAddSquare_8u32f_C1IR");

  --* 
  -- * One 16-bit unsigned short channel image squared then added to in place floating point destination image using filter mask (updates destination when mask is non-zero).
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pMask \ref mask_image_pointer.
  -- * \param nMaskStep \ref mask_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAddSquare_16u32f_C1IMR
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pMask : access nppdefs_h.Npp8u;
      nMaskStep : int;
      pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:4237
   pragma Import (C, nppiAddSquare_16u32f_C1IMR, "nppiAddSquare_16u32f_C1IMR");

  --* 
  -- * One 16-bit unsigned short channel image squared then added to in place floating point destination image.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAddSquare_16u32f_C1IR
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:4250
   pragma Import (C, nppiAddSquare_16u32f_C1IR, "nppiAddSquare_16u32f_C1IR");

  --* 
  -- * One 32-bit floating point channel image squared then added to in place floating point destination image using filter mask (updates destination when mask is non-zero).
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pMask \ref mask_image_pointer.
  -- * \param nMaskStep \ref mask_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAddSquare_32f_C1IMR
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pMask : access nppdefs_h.Npp8u;
      nMaskStep : int;
      pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:4265
   pragma Import (C, nppiAddSquare_32f_C1IMR, "nppiAddSquare_32f_C1IMR");

  --* 
  -- * One 32-bit floating point channel image squared then added to in place floating point destination image.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAddSquare_32f_C1IR
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:4278
   pragma Import (C, nppiAddSquare_32f_C1IR, "nppiAddSquare_32f_C1IR");

  --* @} image_addsquare  
  --* 
  -- * @defgroup image_addproduct AddProduct
  -- * Pixel by pixel addition of product of pixels from two source images to
  -- * floating point pixel values of destination image.
  -- *
  -- * @{
  --  

  --* 
  -- * One 8-bit unsigned char channel image product added to in place floating point destination image using filter mask (updates destination when mask is non-zero).
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pMask \ref mask_image_pointer.
  -- * \param nMaskStep \ref mask_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAddProduct_8u32f_C1IMR
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp8u;
      nSrc2Step : int;
      pMask : access nppdefs_h.Npp8u;
      nMaskStep : int;
      pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:4305
   pragma Import (C, nppiAddProduct_8u32f_C1IMR, "nppiAddProduct_8u32f_C1IMR");

  --* 
  -- * One 8-bit unsigned char channel image product added to in place floating point destination image.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAddProduct_8u32f_C1IR
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp8u;
      nSrc2Step : int;
      pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:4320
   pragma Import (C, nppiAddProduct_8u32f_C1IR, "nppiAddProduct_8u32f_C1IR");

  --* 
  -- * One 16-bit unsigned short channel image product added to in place floating point destination image using filter mask (updates destination when mask is non-zero).
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pMask \ref mask_image_pointer.
  -- * \param nMaskStep \ref mask_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAddProduct_16u32f_C1IMR
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp16u;
      nSrc2Step : int;
      pMask : access nppdefs_h.Npp8u;
      nMaskStep : int;
      pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:4337
   pragma Import (C, nppiAddProduct_16u32f_C1IMR, "nppiAddProduct_16u32f_C1IMR");

  --* 
  -- * One 16-bit unsigned short channel image product added to in place floating point destination image.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAddProduct_16u32f_C1IR
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp16u;
      nSrc2Step : int;
      pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:4352
   pragma Import (C, nppiAddProduct_16u32f_C1IR, "nppiAddProduct_16u32f_C1IR");

  --* 
  -- * One 32-bit floating point channel image product added to in place floating point destination image using filter mask (updates destination when mask is non-zero).
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pMask \ref mask_image_pointer.
  -- * \param nMaskStep \ref mask_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAddProduct_32f_C1IMR
     (pSrc1 : access nppdefs_h.Npp32f;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp32f;
      nSrc2Step : int;
      pMask : access nppdefs_h.Npp8u;
      nMaskStep : int;
      pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:4369
   pragma Import (C, nppiAddProduct_32f_C1IMR, "nppiAddProduct_32f_C1IMR");

  --* 
  -- * One 32-bit floating point channel image product added to in place floating point destination image.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAddProduct_32f_C1IR
     (pSrc1 : access nppdefs_h.Npp32f;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp32f;
      nSrc2Step : int;
      pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:4384
   pragma Import (C, nppiAddProduct_32f_C1IR, "nppiAddProduct_32f_C1IR");

  --* @} image_addproduct  
  --* 
  -- * @defgroup image_addweighted AddWeighted
  -- * Pixel by pixel addition of alpha weighted pixel values from a source image to
  -- * floating point pixel values of destination image.
  -- *
  -- * @{
  --  

  --* 
  -- * One 8-bit unsigned char channel alpha weighted image added to in place floating point destination image using filter mask (updates destination when mask is non-zero).
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pMask \ref mask_image_pointer.
  -- * \param nMaskStep \ref mask_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nAlpha Alpha weight to be applied to source image pixels (0.0F to 1.0F)
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAddWeighted_8u32f_C1IMR
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pMask : access nppdefs_h.Npp8u;
      nMaskStep : int;
      pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nAlpha : nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:4410
   pragma Import (C, nppiAddWeighted_8u32f_C1IMR, "nppiAddWeighted_8u32f_C1IMR");

  --* 
  -- * One 8-bit unsigned char channel alpha weighted image added to in place floating point destination image.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nAlpha Alpha weight to be applied to source image pixels (0.0F to 1.0F)
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAddWeighted_8u32f_C1IR
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nAlpha : nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:4424
   pragma Import (C, nppiAddWeighted_8u32f_C1IR, "nppiAddWeighted_8u32f_C1IR");

  --* 
  -- * One 16-bit unsigned short channel alpha weighted image added to in place floating point destination image using filter mask (updates destination when mask is non-zero).
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pMask \ref mask_image_pointer.
  -- * \param nMaskStep \ref mask_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nAlpha Alpha weight to be applied to source image pixels (0.0F to 1.0F)
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAddWeighted_16u32f_C1IMR
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pMask : access nppdefs_h.Npp8u;
      nMaskStep : int;
      pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nAlpha : nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:4440
   pragma Import (C, nppiAddWeighted_16u32f_C1IMR, "nppiAddWeighted_16u32f_C1IMR");

  --* 
  -- * One 16-bit unsigned short channel alpha weighted image added to in place floating point destination image.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nAlpha Alpha weight to be applied to source image pixels (0.0F to 1.0F)
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAddWeighted_16u32f_C1IR
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nAlpha : nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:4454
   pragma Import (C, nppiAddWeighted_16u32f_C1IR, "nppiAddWeighted_16u32f_C1IR");

  --* 
  -- * One 32-bit floating point channel alpha weighted image added to in place floating point destination image using filter mask (updates destination when mask is non-zero).
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pMask \ref mask_image_pointer.
  -- * \param nMaskStep \ref mask_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nAlpha Alpha weight to be applied to source image pixels (0.0F to 1.0F)
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAddWeighted_32f_C1IMR
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pMask : access nppdefs_h.Npp8u;
      nMaskStep : int;
      pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nAlpha : nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:4470
   pragma Import (C, nppiAddWeighted_32f_C1IMR, "nppiAddWeighted_32f_C1IMR");

  --* 
  -- * One 32-bit floating point channel alpha weighted image added to in place floating point destination image.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nAlpha Alpha weight to be applied to source image pixels (0.0F to 1.0F)
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAddWeighted_32f_C1IR
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nAlpha : nppdefs_h.Npp32f) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:4484
   pragma Import (C, nppiAddWeighted_32f_C1IR, "nppiAddWeighted_32f_C1IR");

  --* @} image_addweighted  
  --* 
  -- * @defgroup image_mul Mul
  -- *
  -- * Pixel by pixel multiply of two images.
  -- *
  -- * @{
  --  

  --* 
  -- * One 8-bit unsigned char channel image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMul_8u_C1RSfs
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp8u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:4510
   pragma Import (C, nppiMul_8u_C1RSfs, "nppiMul_8u_C1RSfs");

  --* 
  -- * One 8-bit unsigned char channel in place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMul_8u_C1IRSfs
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:4524
   pragma Import (C, nppiMul_8u_C1IRSfs, "nppiMul_8u_C1IRSfs");

  --* 
  -- * Three 8-bit unsigned char channel image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMul_8u_C3RSfs
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp8u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:4540
   pragma Import (C, nppiMul_8u_C3RSfs, "nppiMul_8u_C3RSfs");

  --* 
  -- * Three 8-bit unsigned char channel in place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMul_8u_C3IRSfs
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:4554
   pragma Import (C, nppiMul_8u_C3IRSfs, "nppiMul_8u_C3IRSfs");

  --* 
  -- * Four 8-bit unsigned char channel with unmodified alpha image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMul_8u_AC4RSfs
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp8u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:4570
   pragma Import (C, nppiMul_8u_AC4RSfs, "nppiMul_8u_AC4RSfs");

  --* 
  -- * Four 8-bit unsigned char channel with unmodified alpha in place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMul_8u_AC4IRSfs
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:4584
   pragma Import (C, nppiMul_8u_AC4IRSfs, "nppiMul_8u_AC4IRSfs");

  --* 
  -- * Four 8-bit unsigned char channel image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMul_8u_C4RSfs
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp8u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:4600
   pragma Import (C, nppiMul_8u_C4RSfs, "nppiMul_8u_C4RSfs");

  --* 
  -- * Four 8-bit unsigned char channel in place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMul_8u_C4IRSfs
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:4614
   pragma Import (C, nppiMul_8u_C4IRSfs, "nppiMul_8u_C4IRSfs");

  --* 
  -- * One 16-bit unsigned short channel image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMul_16u_C1RSfs
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp16u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:4630
   pragma Import (C, nppiMul_16u_C1RSfs, "nppiMul_16u_C1RSfs");

  --* 
  -- * One 16-bit unsigned short channel in place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMul_16u_C1IRSfs
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:4644
   pragma Import (C, nppiMul_16u_C1IRSfs, "nppiMul_16u_C1IRSfs");

  --* 
  -- * Three 16-bit unsigned short channel image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMul_16u_C3RSfs
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp16u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:4660
   pragma Import (C, nppiMul_16u_C3RSfs, "nppiMul_16u_C3RSfs");

  --* 
  -- * Three 16-bit unsigned short channel in place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMul_16u_C3IRSfs
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:4674
   pragma Import (C, nppiMul_16u_C3IRSfs, "nppiMul_16u_C3IRSfs");

  --* 
  -- * Four 16-bit unsigned short channel with unmodified alpha image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMul_16u_AC4RSfs
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp16u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:4690
   pragma Import (C, nppiMul_16u_AC4RSfs, "nppiMul_16u_AC4RSfs");

  --* 
  -- * Four 16-bit unsigned short channel with unmodified alpha in place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMul_16u_AC4IRSfs
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:4704
   pragma Import (C, nppiMul_16u_AC4IRSfs, "nppiMul_16u_AC4IRSfs");

  --* 
  -- * Four 16-bit unsigned short channel image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMul_16u_C4RSfs
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp16u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:4720
   pragma Import (C, nppiMul_16u_C4RSfs, "nppiMul_16u_C4RSfs");

  --* 
  -- * Four 16-bit unsigned short channel in place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMul_16u_C4IRSfs
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:4734
   pragma Import (C, nppiMul_16u_C4IRSfs, "nppiMul_16u_C4IRSfs");

  --* 
  -- * One 16-bit signed short channel image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMul_16s_C1RSfs
     (pSrc1 : access nppdefs_h.Npp16s;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp16s;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:4750
   pragma Import (C, nppiMul_16s_C1RSfs, "nppiMul_16s_C1RSfs");

  --* 
  -- * One 16-bit signed short channel in place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMul_16s_C1IRSfs
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:4764
   pragma Import (C, nppiMul_16s_C1IRSfs, "nppiMul_16s_C1IRSfs");

  --* 
  -- * Three 16-bit signed short channel image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMul_16s_C3RSfs
     (pSrc1 : access nppdefs_h.Npp16s;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp16s;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:4780
   pragma Import (C, nppiMul_16s_C3RSfs, "nppiMul_16s_C3RSfs");

  --* 
  -- * Three 16-bit signed short channel in place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMul_16s_C3IRSfs
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:4794
   pragma Import (C, nppiMul_16s_C3IRSfs, "nppiMul_16s_C3IRSfs");

  --* 
  -- * Four 16-bit signed short channel with unmodified alpha image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMul_16s_AC4RSfs
     (pSrc1 : access nppdefs_h.Npp16s;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp16s;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:4810
   pragma Import (C, nppiMul_16s_AC4RSfs, "nppiMul_16s_AC4RSfs");

  --* 
  -- * Four 16-bit signed short channel with unmodified alpha in place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMul_16s_AC4IRSfs
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:4824
   pragma Import (C, nppiMul_16s_AC4IRSfs, "nppiMul_16s_AC4IRSfs");

  --* 
  -- * Four 16-bit signed short channel image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMul_16s_C4RSfs
     (pSrc1 : access nppdefs_h.Npp16s;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp16s;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:4840
   pragma Import (C, nppiMul_16s_C4RSfs, "nppiMul_16s_C4RSfs");

  --* 
  -- * Four 16-bit signed short channel in place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMul_16s_C4IRSfs
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:4854
   pragma Import (C, nppiMul_16s_C4IRSfs, "nppiMul_16s_C4IRSfs");

  --* 
  -- * One 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMul_16sc_C1RSfs
     (pSrc1 : access constant nppdefs_h.Npp16sc;
      nSrc1Step : int;
      pSrc2 : access constant nppdefs_h.Npp16sc;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp16sc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:4870
   pragma Import (C, nppiMul_16sc_C1RSfs, "nppiMul_16sc_C1RSfs");

  --* 
  -- * One 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel in place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMul_16sc_C1IRSfs
     (pSrc : access constant nppdefs_h.Npp16sc;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp16sc;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:4884
   pragma Import (C, nppiMul_16sc_C1IRSfs, "nppiMul_16sc_C1IRSfs");

  --* 
  -- * Three 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMul_16sc_C3RSfs
     (pSrc1 : access constant nppdefs_h.Npp16sc;
      nSrc1Step : int;
      pSrc2 : access constant nppdefs_h.Npp16sc;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp16sc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:4900
   pragma Import (C, nppiMul_16sc_C3RSfs, "nppiMul_16sc_C3RSfs");

  --* 
  -- * Three 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel in place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMul_16sc_C3IRSfs
     (pSrc : access constant nppdefs_h.Npp16sc;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp16sc;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:4914
   pragma Import (C, nppiMul_16sc_C3IRSfs, "nppiMul_16sc_C3IRSfs");

  --* 
  -- * Four 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel with unmodified alpha image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMul_16sc_AC4RSfs
     (pSrc1 : access constant nppdefs_h.Npp16sc;
      nSrc1Step : int;
      pSrc2 : access constant nppdefs_h.Npp16sc;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp16sc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:4930
   pragma Import (C, nppiMul_16sc_AC4RSfs, "nppiMul_16sc_AC4RSfs");

  --* 
  -- * Four 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel with unmodified alpha in place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMul_16sc_AC4IRSfs
     (pSrc : access constant nppdefs_h.Npp16sc;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp16sc;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:4944
   pragma Import (C, nppiMul_16sc_AC4IRSfs, "nppiMul_16sc_AC4IRSfs");

  --* 
  -- * One 32-bit signed integer channel image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMul_32s_C1RSfs
     (pSrc1 : access nppdefs_h.Npp32s;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp32s;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:4960
   pragma Import (C, nppiMul_32s_C1RSfs, "nppiMul_32s_C1RSfs");

  --* 
  -- * Note: This function is to be deprecated in future NPP releases, use the function above with a scale factor of 0 instead.
  -- * 1 channel 32-bit image multiplication.
  -- * Multiply corresponding pixels in ROI. 
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMul_32s_C1R
     (pSrc1 : access nppdefs_h.Npp32s;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp32s;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:4976
   pragma Import (C, nppiMul_32s_C1R, "nppiMul_32s_C1R");

  --* 
  -- * One 32-bit signed integer channel in place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMul_32s_C1IRSfs
     (pSrc : access nppdefs_h.Npp32s;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp32s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:4992
   pragma Import (C, nppiMul_32s_C1IRSfs, "nppiMul_32s_C1IRSfs");

  --* 
  -- * Three 32-bit signed integer channel image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMul_32s_C3RSfs
     (pSrc1 : access nppdefs_h.Npp32s;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp32s;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:5008
   pragma Import (C, nppiMul_32s_C3RSfs, "nppiMul_32s_C3RSfs");

  --* 
  -- * Three 32-bit signed integer channel in place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMul_32s_C3IRSfs
     (pSrc : access nppdefs_h.Npp32s;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp32s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:5022
   pragma Import (C, nppiMul_32s_C3IRSfs, "nppiMul_32s_C3IRSfs");

  --* 
  -- * One 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMul_32sc_C1RSfs
     (pSrc1 : access constant nppdefs_h.Npp32sc;
      nSrc1Step : int;
      pSrc2 : access constant nppdefs_h.Npp32sc;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp32sc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:5038
   pragma Import (C, nppiMul_32sc_C1RSfs, "nppiMul_32sc_C1RSfs");

  --* 
  -- * One 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel in place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMul_32sc_C1IRSfs
     (pSrc : access constant nppdefs_h.Npp32sc;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp32sc;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:5052
   pragma Import (C, nppiMul_32sc_C1IRSfs, "nppiMul_32sc_C1IRSfs");

  --* 
  -- * Three 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMul_32sc_C3RSfs
     (pSrc1 : access constant nppdefs_h.Npp32sc;
      nSrc1Step : int;
      pSrc2 : access constant nppdefs_h.Npp32sc;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp32sc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:5068
   pragma Import (C, nppiMul_32sc_C3RSfs, "nppiMul_32sc_C3RSfs");

  --* 
  -- * Three 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel in place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMul_32sc_C3IRSfs
     (pSrc : access constant nppdefs_h.Npp32sc;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp32sc;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:5082
   pragma Import (C, nppiMul_32sc_C3IRSfs, "nppiMul_32sc_C3IRSfs");

  --* 
  -- * Four 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel with unmodified alpha image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMul_32sc_AC4RSfs
     (pSrc1 : access constant nppdefs_h.Npp32sc;
      nSrc1Step : int;
      pSrc2 : access constant nppdefs_h.Npp32sc;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp32sc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:5098
   pragma Import (C, nppiMul_32sc_AC4RSfs, "nppiMul_32sc_AC4RSfs");

  --* 
  -- * Four 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel with unmodified alpha in place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMul_32sc_AC4IRSfs
     (pSrc : access constant nppdefs_h.Npp32sc;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp32sc;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:5112
   pragma Import (C, nppiMul_32sc_AC4IRSfs, "nppiMul_32sc_AC4IRSfs");

  --* 
  -- * One 32-bit floating point channel image multiplication.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMul_32f_C1R
     (pSrc1 : access nppdefs_h.Npp32f;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp32f;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:5127
   pragma Import (C, nppiMul_32f_C1R, "nppiMul_32f_C1R");

  --* 
  -- * One 32-bit floating point channel in place image multiplication.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMul_32f_C1IR
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:5140
   pragma Import (C, nppiMul_32f_C1IR, "nppiMul_32f_C1IR");

  --* 
  -- * Three 32-bit floating point channel image multiplication.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMul_32f_C3R
     (pSrc1 : access nppdefs_h.Npp32f;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp32f;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:5155
   pragma Import (C, nppiMul_32f_C3R, "nppiMul_32f_C3R");

  --* 
  -- * Three 32-bit floating point channel in place image multiplication.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMul_32f_C3IR
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:5168
   pragma Import (C, nppiMul_32f_C3IR, "nppiMul_32f_C3IR");

  --* 
  -- * Four 32-bit floating point channel with unmodified alpha image multiplication.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMul_32f_AC4R
     (pSrc1 : access nppdefs_h.Npp32f;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp32f;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:5183
   pragma Import (C, nppiMul_32f_AC4R, "nppiMul_32f_AC4R");

  --* 
  -- * Four 32-bit floating point channel with unmodified alpha in place image multiplication.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMul_32f_AC4IR
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:5196
   pragma Import (C, nppiMul_32f_AC4IR, "nppiMul_32f_AC4IR");

  --* 
  -- * Four 32-bit floating point channel image multiplication.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMul_32f_C4R
     (pSrc1 : access nppdefs_h.Npp32f;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp32f;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:5211
   pragma Import (C, nppiMul_32f_C4R, "nppiMul_32f_C4R");

  --* 
  -- * Four 32-bit floating point channel in place image multiplication.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMul_32f_C4IR
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:5224
   pragma Import (C, nppiMul_32f_C4IR, "nppiMul_32f_C4IR");

  --* 
  -- * One 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel image multiplication.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMul_32fc_C1R
     (pSrc1 : access constant nppdefs_h.Npp32fc;
      nSrc1Step : int;
      pSrc2 : access constant nppdefs_h.Npp32fc;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp32fc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:5239
   pragma Import (C, nppiMul_32fc_C1R, "nppiMul_32fc_C1R");

  --* 
  -- * One 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel in place image multiplication.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMul_32fc_C1IR
     (pSrc : access constant nppdefs_h.Npp32fc;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp32fc;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:5252
   pragma Import (C, nppiMul_32fc_C1IR, "nppiMul_32fc_C1IR");

  --* 
  -- * Three 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel image multiplication.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMul_32fc_C3R
     (pSrc1 : access constant nppdefs_h.Npp32fc;
      nSrc1Step : int;
      pSrc2 : access constant nppdefs_h.Npp32fc;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp32fc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:5267
   pragma Import (C, nppiMul_32fc_C3R, "nppiMul_32fc_C3R");

  --* 
  -- * Three 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel in place image multiplication.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMul_32fc_C3IR
     (pSrc : access constant nppdefs_h.Npp32fc;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp32fc;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:5280
   pragma Import (C, nppiMul_32fc_C3IR, "nppiMul_32fc_C3IR");

  --* 
  -- * Four 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel with unmodified alpha image multiplication.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMul_32fc_AC4R
     (pSrc1 : access constant nppdefs_h.Npp32fc;
      nSrc1Step : int;
      pSrc2 : access constant nppdefs_h.Npp32fc;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp32fc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:5295
   pragma Import (C, nppiMul_32fc_AC4R, "nppiMul_32fc_AC4R");

  --* 
  -- * Four 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel with unmodified alpha in place image multiplication.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMul_32fc_AC4IR
     (pSrc : access constant nppdefs_h.Npp32fc;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp32fc;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:5308
   pragma Import (C, nppiMul_32fc_AC4IR, "nppiMul_32fc_AC4IR");

  --* 
  -- * Four 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel image multiplication.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMul_32fc_C4R
     (pSrc1 : access constant nppdefs_h.Npp32fc;
      nSrc1Step : int;
      pSrc2 : access constant nppdefs_h.Npp32fc;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp32fc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:5323
   pragma Import (C, nppiMul_32fc_C4R, "nppiMul_32fc_C4R");

  --* 
  -- * Four 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel in place image multiplication.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMul_32fc_C4IR
     (pSrc : access constant nppdefs_h.Npp32fc;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp32fc;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:5336
   pragma Import (C, nppiMul_32fc_C4IR, "nppiMul_32fc_C4IR");

  --* @} image_mul  
  --* 
  -- * @defgroup image_mulscale MulScale
  -- *
  -- * Pixel by pixel multiplies each pixel of two images then scales the result by
  -- * the maximum value for the data bit width.
  -- *
  -- * @{
  --  

  --* 
  -- * One 8-bit unsigned char channel image multiplication then scale by maximum value for pixel bit width.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMulScale_8u_C1R
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp8u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:5362
   pragma Import (C, nppiMulScale_8u_C1R, "nppiMulScale_8u_C1R");

  --* 
  -- * One 8-bit unsigned char channel in place image multiplication then scale by maximum value for pixel bit width.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMulScale_8u_C1IR
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:5375
   pragma Import (C, nppiMulScale_8u_C1IR, "nppiMulScale_8u_C1IR");

  --* 
  -- * Three 8-bit unsigned char channel image multiplication then scale by maximum value for pixel bit width.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMulScale_8u_C3R
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp8u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:5390
   pragma Import (C, nppiMulScale_8u_C3R, "nppiMulScale_8u_C3R");

  --* 
  -- * Three 8-bit unsigned char channel in place image multiplication then scale by maximum value for pixel bit width.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMulScale_8u_C3IR
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:5403
   pragma Import (C, nppiMulScale_8u_C3IR, "nppiMulScale_8u_C3IR");

  --* 
  -- * Four 8-bit unsigned char channel with unmodified alpha image multiplication then scale by maximum value for pixel bit width.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMulScale_8u_AC4R
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp8u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:5418
   pragma Import (C, nppiMulScale_8u_AC4R, "nppiMulScale_8u_AC4R");

  --* 
  -- * Four 8-bit unsigned char channel with unmodified alpha in place image multiplication then scale by maximum value for pixel bit width.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMulScale_8u_AC4IR
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:5431
   pragma Import (C, nppiMulScale_8u_AC4IR, "nppiMulScale_8u_AC4IR");

  --* 
  -- * Four 8-bit unsigned char channel image multiplication then scale by maximum value for pixel bit width.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMulScale_8u_C4R
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp8u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:5446
   pragma Import (C, nppiMulScale_8u_C4R, "nppiMulScale_8u_C4R");

  --* 
  -- * Four 8-bit unsigned char channel in place image multiplication then scale by maximum value for pixel bit width.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMulScale_8u_C4IR
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:5459
   pragma Import (C, nppiMulScale_8u_C4IR, "nppiMulScale_8u_C4IR");

  --* 
  -- * One 16-bit unsigned short channel image multiplication then scale by maximum value for pixel bit width.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMulScale_16u_C1R
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp16u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:5474
   pragma Import (C, nppiMulScale_16u_C1R, "nppiMulScale_16u_C1R");

  --* 
  -- * One 16-bit unsigned short channel in place image multiplication then scale by maximum value for pixel bit width.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMulScale_16u_C1IR
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:5487
   pragma Import (C, nppiMulScale_16u_C1IR, "nppiMulScale_16u_C1IR");

  --* 
  -- * Three 16-bit unsigned short channel image multiplication then scale by maximum value for pixel bit width.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMulScale_16u_C3R
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp16u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:5502
   pragma Import (C, nppiMulScale_16u_C3R, "nppiMulScale_16u_C3R");

  --* 
  -- * Three 16-bit unsigned short channel in place image multiplication then scale by maximum value for pixel bit width.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMulScale_16u_C3IR
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:5515
   pragma Import (C, nppiMulScale_16u_C3IR, "nppiMulScale_16u_C3IR");

  --* 
  -- * Four 16-bit unsigned short channel with unmodified alpha image multiplication then scale by maximum value for pixel bit width.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMulScale_16u_AC4R
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp16u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:5530
   pragma Import (C, nppiMulScale_16u_AC4R, "nppiMulScale_16u_AC4R");

  --* 
  -- * Four 16-bit unsigned short channel with unmodified alpha in place image multiplication then scale by maximum value for pixel bit width.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMulScale_16u_AC4IR
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:5543
   pragma Import (C, nppiMulScale_16u_AC4IR, "nppiMulScale_16u_AC4IR");

  --* 
  -- * Four 16-bit unsigned short channel image multiplication then scale by maximum value for pixel bit width.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMulScale_16u_C4R
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp16u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:5558
   pragma Import (C, nppiMulScale_16u_C4R, "nppiMulScale_16u_C4R");

  --* 
  -- * Four 16-bit unsigned short channel in place image multiplication then scale by maximum value for pixel bit width.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiMulScale_16u_C4IR
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:5571
   pragma Import (C, nppiMulScale_16u_C4IR, "nppiMulScale_16u_C4IR");

  --* @} image_mulscale  
  --* 
  -- * @defgroup image_sub Sub
  -- *
  -- * Pixel by pixel subtraction of two images.
  -- *
  -- * @{
  --  

  --* 
  -- * One 8-bit unsigned char channel image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSub_8u_C1RSfs
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp8u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:5597
   pragma Import (C, nppiSub_8u_C1RSfs, "nppiSub_8u_C1RSfs");

  --* 
  -- * One 8-bit unsigned char channel in place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSub_8u_C1IRSfs
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:5611
   pragma Import (C, nppiSub_8u_C1IRSfs, "nppiSub_8u_C1IRSfs");

  --* 
  -- * Three 8-bit unsigned char channel image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSub_8u_C3RSfs
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp8u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:5627
   pragma Import (C, nppiSub_8u_C3RSfs, "nppiSub_8u_C3RSfs");

  --* 
  -- * Three 8-bit unsigned char channel in place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSub_8u_C3IRSfs
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:5641
   pragma Import (C, nppiSub_8u_C3IRSfs, "nppiSub_8u_C3IRSfs");

  --* 
  -- * Four 8-bit unsigned char channel with unmodified alpha image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSub_8u_AC4RSfs
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp8u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:5657
   pragma Import (C, nppiSub_8u_AC4RSfs, "nppiSub_8u_AC4RSfs");

  --* 
  -- * Four 8-bit unsigned char channel with unmodified alpha in place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSub_8u_AC4IRSfs
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:5671
   pragma Import (C, nppiSub_8u_AC4IRSfs, "nppiSub_8u_AC4IRSfs");

  --* 
  -- * Four 8-bit unsigned char channel image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSub_8u_C4RSfs
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp8u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:5687
   pragma Import (C, nppiSub_8u_C4RSfs, "nppiSub_8u_C4RSfs");

  --* 
  -- * Four 8-bit unsigned char channel in place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSub_8u_C4IRSfs
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:5701
   pragma Import (C, nppiSub_8u_C4IRSfs, "nppiSub_8u_C4IRSfs");

  --* 
  -- * One 16-bit unsigned short channel image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSub_16u_C1RSfs
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp16u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:5717
   pragma Import (C, nppiSub_16u_C1RSfs, "nppiSub_16u_C1RSfs");

  --* 
  -- * One 16-bit unsigned short channel in place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSub_16u_C1IRSfs
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:5731
   pragma Import (C, nppiSub_16u_C1IRSfs, "nppiSub_16u_C1IRSfs");

  --* 
  -- * Three 16-bit unsigned short channel image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSub_16u_C3RSfs
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp16u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:5747
   pragma Import (C, nppiSub_16u_C3RSfs, "nppiSub_16u_C3RSfs");

  --* 
  -- * Three 16-bit unsigned short channel in place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSub_16u_C3IRSfs
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:5761
   pragma Import (C, nppiSub_16u_C3IRSfs, "nppiSub_16u_C3IRSfs");

  --* 
  -- * Four 16-bit unsigned short channel with unmodified alpha image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSub_16u_AC4RSfs
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp16u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:5777
   pragma Import (C, nppiSub_16u_AC4RSfs, "nppiSub_16u_AC4RSfs");

  --* 
  -- * Four 16-bit unsigned short channel with unmodified alpha in place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSub_16u_AC4IRSfs
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:5791
   pragma Import (C, nppiSub_16u_AC4IRSfs, "nppiSub_16u_AC4IRSfs");

  --* 
  -- * Four 16-bit unsigned short channel image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSub_16u_C4RSfs
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp16u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:5807
   pragma Import (C, nppiSub_16u_C4RSfs, "nppiSub_16u_C4RSfs");

  --* 
  -- * Four 16-bit unsigned short channel in place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSub_16u_C4IRSfs
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:5821
   pragma Import (C, nppiSub_16u_C4IRSfs, "nppiSub_16u_C4IRSfs");

  --* 
  -- * One 16-bit signed short channel image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSub_16s_C1RSfs
     (pSrc1 : access nppdefs_h.Npp16s;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp16s;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:5837
   pragma Import (C, nppiSub_16s_C1RSfs, "nppiSub_16s_C1RSfs");

  --* 
  -- * One 16-bit signed short channel in place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSub_16s_C1IRSfs
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:5851
   pragma Import (C, nppiSub_16s_C1IRSfs, "nppiSub_16s_C1IRSfs");

  --* 
  -- * Three 16-bit signed short channel image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSub_16s_C3RSfs
     (pSrc1 : access nppdefs_h.Npp16s;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp16s;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:5867
   pragma Import (C, nppiSub_16s_C3RSfs, "nppiSub_16s_C3RSfs");

  --* 
  -- * Three 16-bit signed short channel in place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSub_16s_C3IRSfs
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:5881
   pragma Import (C, nppiSub_16s_C3IRSfs, "nppiSub_16s_C3IRSfs");

  --* 
  -- * Four 16-bit signed short channel with unmodified alpha image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSub_16s_AC4RSfs
     (pSrc1 : access nppdefs_h.Npp16s;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp16s;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:5897
   pragma Import (C, nppiSub_16s_AC4RSfs, "nppiSub_16s_AC4RSfs");

  --* 
  -- * Four 16-bit signed short channel with unmodified alpha in place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSub_16s_AC4IRSfs
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:5911
   pragma Import (C, nppiSub_16s_AC4IRSfs, "nppiSub_16s_AC4IRSfs");

  --* 
  -- * Four 16-bit signed short channel image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSub_16s_C4RSfs
     (pSrc1 : access nppdefs_h.Npp16s;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp16s;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:5927
   pragma Import (C, nppiSub_16s_C4RSfs, "nppiSub_16s_C4RSfs");

  --* 
  -- * Four 16-bit signed short channel in place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSub_16s_C4IRSfs
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:5941
   pragma Import (C, nppiSub_16s_C4IRSfs, "nppiSub_16s_C4IRSfs");

  --* 
  -- * One 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSub_16sc_C1RSfs
     (pSrc1 : access constant nppdefs_h.Npp16sc;
      nSrc1Step : int;
      pSrc2 : access constant nppdefs_h.Npp16sc;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp16sc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:5957
   pragma Import (C, nppiSub_16sc_C1RSfs, "nppiSub_16sc_C1RSfs");

  --* 
  -- * One 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel in place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSub_16sc_C1IRSfs
     (pSrc : access constant nppdefs_h.Npp16sc;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp16sc;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:5971
   pragma Import (C, nppiSub_16sc_C1IRSfs, "nppiSub_16sc_C1IRSfs");

  --* 
  -- * Three 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSub_16sc_C3RSfs
     (pSrc1 : access constant nppdefs_h.Npp16sc;
      nSrc1Step : int;
      pSrc2 : access constant nppdefs_h.Npp16sc;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp16sc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:5987
   pragma Import (C, nppiSub_16sc_C3RSfs, "nppiSub_16sc_C3RSfs");

  --* 
  -- * Three 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel in place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSub_16sc_C3IRSfs
     (pSrc : access constant nppdefs_h.Npp16sc;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp16sc;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:6001
   pragma Import (C, nppiSub_16sc_C3IRSfs, "nppiSub_16sc_C3IRSfs");

  --* 
  -- * Four 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel with unmodified alpha image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSub_16sc_AC4RSfs
     (pSrc1 : access constant nppdefs_h.Npp16sc;
      nSrc1Step : int;
      pSrc2 : access constant nppdefs_h.Npp16sc;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp16sc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:6017
   pragma Import (C, nppiSub_16sc_AC4RSfs, "nppiSub_16sc_AC4RSfs");

  --* 
  -- * Four 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel with unmodified alpha in place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSub_16sc_AC4IRSfs
     (pSrc : access constant nppdefs_h.Npp16sc;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp16sc;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:6031
   pragma Import (C, nppiSub_16sc_AC4IRSfs, "nppiSub_16sc_AC4IRSfs");

  --* 
  -- * One 32-bit signed integer channel image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSub_32s_C1RSfs
     (pSrc1 : access nppdefs_h.Npp32s;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp32s;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:6047
   pragma Import (C, nppiSub_32s_C1RSfs, "nppiSub_32s_C1RSfs");

  --*
  -- * Note: This function is to be deprecated in future NPP releases, use the function above with a scale factor of 0 instead. 
  -- * 32-bit image subtraction.
  -- * Subtract pSrc1's pixels from corresponding pixels in pSrc2. 
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSub_32s_C1R
     (pSrc1 : access nppdefs_h.Npp32s;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp32s;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:6063
   pragma Import (C, nppiSub_32s_C1R, "nppiSub_32s_C1R");

  --* 
  -- * One 32-bit signed integer channel in place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSub_32s_C1IRSfs
     (pSrc : access nppdefs_h.Npp32s;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp32s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:6079
   pragma Import (C, nppiSub_32s_C1IRSfs, "nppiSub_32s_C1IRSfs");

  --* 
  -- * Three 32-bit signed integer channel image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSub_32s_C3RSfs
     (pSrc1 : access nppdefs_h.Npp32s;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp32s;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:6095
   pragma Import (C, nppiSub_32s_C3RSfs, "nppiSub_32s_C3RSfs");

  --* 
  -- * Three 32-bit signed integer channel in place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSub_32s_C3IRSfs
     (pSrc : access nppdefs_h.Npp32s;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp32s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:6109
   pragma Import (C, nppiSub_32s_C3IRSfs, "nppiSub_32s_C3IRSfs");

  --* 
  -- * Four 32-bit signed integer channel image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSub_32s_C4RSfs
     (pSrc1 : access nppdefs_h.Npp32s;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp32s;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:6125
   pragma Import (C, nppiSub_32s_C4RSfs, "nppiSub_32s_C4RSfs");

  --* 
  -- * Four 32-bit signed integer channel in place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSub_32s_C4IRSfs
     (pSrc : access nppdefs_h.Npp32s;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp32s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:6139
   pragma Import (C, nppiSub_32s_C4IRSfs, "nppiSub_32s_C4IRSfs");

  --* 
  -- * One 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSub_32sc_C1RSfs
     (pSrc1 : access constant nppdefs_h.Npp32sc;
      nSrc1Step : int;
      pSrc2 : access constant nppdefs_h.Npp32sc;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp32sc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:6155
   pragma Import (C, nppiSub_32sc_C1RSfs, "nppiSub_32sc_C1RSfs");

  --* 
  -- * One 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel in place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSub_32sc_C1IRSfs
     (pSrc : access constant nppdefs_h.Npp32sc;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp32sc;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:6169
   pragma Import (C, nppiSub_32sc_C1IRSfs, "nppiSub_32sc_C1IRSfs");

  --* 
  -- * Three 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSub_32sc_C3RSfs
     (pSrc1 : access constant nppdefs_h.Npp32sc;
      nSrc1Step : int;
      pSrc2 : access constant nppdefs_h.Npp32sc;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp32sc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:6185
   pragma Import (C, nppiSub_32sc_C3RSfs, "nppiSub_32sc_C3RSfs");

  --* 
  -- * Three 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel in place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSub_32sc_C3IRSfs
     (pSrc : access constant nppdefs_h.Npp32sc;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp32sc;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:6199
   pragma Import (C, nppiSub_32sc_C3IRSfs, "nppiSub_32sc_C3IRSfs");

  --* 
  -- * Four 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel with unmodified alpha image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSub_32sc_AC4RSfs
     (pSrc1 : access constant nppdefs_h.Npp32sc;
      nSrc1Step : int;
      pSrc2 : access constant nppdefs_h.Npp32sc;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp32sc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:6215
   pragma Import (C, nppiSub_32sc_AC4RSfs, "nppiSub_32sc_AC4RSfs");

  --* 
  -- * Four 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel with unmodified alpha in place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSub_32sc_AC4IRSfs
     (pSrc : access constant nppdefs_h.Npp32sc;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp32sc;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:6229
   pragma Import (C, nppiSub_32sc_AC4IRSfs, "nppiSub_32sc_AC4IRSfs");

  --* 
  -- * One 32-bit floating point channel image subtraction.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSub_32f_C1R
     (pSrc1 : access nppdefs_h.Npp32f;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp32f;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:6244
   pragma Import (C, nppiSub_32f_C1R, "nppiSub_32f_C1R");

  --* 
  -- * One 32-bit floating point channel in place image subtraction.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSub_32f_C1IR
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:6257
   pragma Import (C, nppiSub_32f_C1IR, "nppiSub_32f_C1IR");

  --* 
  -- * Three 32-bit floating point channel image subtraction.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSub_32f_C3R
     (pSrc1 : access nppdefs_h.Npp32f;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp32f;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:6272
   pragma Import (C, nppiSub_32f_C3R, "nppiSub_32f_C3R");

  --* 
  -- * Three 32-bit floating point channel in place image subtraction.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSub_32f_C3IR
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:6285
   pragma Import (C, nppiSub_32f_C3IR, "nppiSub_32f_C3IR");

  --* 
  -- * Four 32-bit floating point channel with unmodified alpha image subtraction.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSub_32f_AC4R
     (pSrc1 : access nppdefs_h.Npp32f;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp32f;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:6300
   pragma Import (C, nppiSub_32f_AC4R, "nppiSub_32f_AC4R");

  --* 
  -- * Four 32-bit floating point channel with unmodified alpha in place image subtraction.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSub_32f_AC4IR
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:6313
   pragma Import (C, nppiSub_32f_AC4IR, "nppiSub_32f_AC4IR");

  --* 
  -- * Four 32-bit floating point channel image subtraction.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSub_32f_C4R
     (pSrc1 : access nppdefs_h.Npp32f;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp32f;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:6328
   pragma Import (C, nppiSub_32f_C4R, "nppiSub_32f_C4R");

  --* 
  -- * Four 32-bit floating point channel in place image subtraction.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSub_32f_C4IR
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:6341
   pragma Import (C, nppiSub_32f_C4IR, "nppiSub_32f_C4IR");

  --* 
  -- * One 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel image subtraction.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSub_32fc_C1R
     (pSrc1 : access constant nppdefs_h.Npp32fc;
      nSrc1Step : int;
      pSrc2 : access constant nppdefs_h.Npp32fc;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp32fc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:6356
   pragma Import (C, nppiSub_32fc_C1R, "nppiSub_32fc_C1R");

  --* 
  -- * One 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel in place image subtraction.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSub_32fc_C1IR
     (pSrc : access constant nppdefs_h.Npp32fc;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp32fc;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:6369
   pragma Import (C, nppiSub_32fc_C1IR, "nppiSub_32fc_C1IR");

  --* 
  -- * Three 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel image subtraction.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSub_32fc_C3R
     (pSrc1 : access constant nppdefs_h.Npp32fc;
      nSrc1Step : int;
      pSrc2 : access constant nppdefs_h.Npp32fc;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp32fc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:6384
   pragma Import (C, nppiSub_32fc_C3R, "nppiSub_32fc_C3R");

  --* 
  -- * Three 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel in place image subtraction.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSub_32fc_C3IR
     (pSrc : access constant nppdefs_h.Npp32fc;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp32fc;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:6397
   pragma Import (C, nppiSub_32fc_C3IR, "nppiSub_32fc_C3IR");

  --* 
  -- * Four 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel with unmodified alpha image subtraction.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSub_32fc_AC4R
     (pSrc1 : access constant nppdefs_h.Npp32fc;
      nSrc1Step : int;
      pSrc2 : access constant nppdefs_h.Npp32fc;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp32fc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:6412
   pragma Import (C, nppiSub_32fc_AC4R, "nppiSub_32fc_AC4R");

  --* 
  -- * Four 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel with unmodified alpha in place image subtraction.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSub_32fc_AC4IR
     (pSrc : access constant nppdefs_h.Npp32fc;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp32fc;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:6425
   pragma Import (C, nppiSub_32fc_AC4IR, "nppiSub_32fc_AC4IR");

  --* 
  -- * Four 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel image subtraction.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSub_32fc_C4R
     (pSrc1 : access constant nppdefs_h.Npp32fc;
      nSrc1Step : int;
      pSrc2 : access constant nppdefs_h.Npp32fc;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp32fc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:6440
   pragma Import (C, nppiSub_32fc_C4R, "nppiSub_32fc_C4R");

  --* 
  -- * Four 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel in place image subtraction.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSub_32fc_C4IR
     (pSrc : access constant nppdefs_h.Npp32fc;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp32fc;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:6453
   pragma Import (C, nppiSub_32fc_C4IR, "nppiSub_32fc_C4IR");

  --* @} image_sub  
  --* 
  -- * @defgroup image_div Div
  -- *
  -- * Pixel by pixel division of two images.
  -- *
  -- * @{
  --  

  --* 
  -- * One 8-bit unsigned char channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDiv_8u_C1RSfs
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp8u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:6479
   pragma Import (C, nppiDiv_8u_C1RSfs, "nppiDiv_8u_C1RSfs");

  --* 
  -- * One 8-bit unsigned char channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDiv_8u_C1IRSfs
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:6493
   pragma Import (C, nppiDiv_8u_C1IRSfs, "nppiDiv_8u_C1IRSfs");

  --* 
  -- * Three 8-bit unsigned char channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDiv_8u_C3RSfs
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp8u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:6509
   pragma Import (C, nppiDiv_8u_C3RSfs, "nppiDiv_8u_C3RSfs");

  --* 
  -- * Three 8-bit unsigned char channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDiv_8u_C3IRSfs
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:6523
   pragma Import (C, nppiDiv_8u_C3IRSfs, "nppiDiv_8u_C3IRSfs");

  --* 
  -- * Four 8-bit unsigned char channel with unmodified alpha image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDiv_8u_AC4RSfs
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp8u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:6539
   pragma Import (C, nppiDiv_8u_AC4RSfs, "nppiDiv_8u_AC4RSfs");

  --* 
  -- * Four 8-bit unsigned char channel with unmodified alpha in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDiv_8u_AC4IRSfs
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:6553
   pragma Import (C, nppiDiv_8u_AC4IRSfs, "nppiDiv_8u_AC4IRSfs");

  --* 
  -- * Four 8-bit unsigned char channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDiv_8u_C4RSfs
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp8u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:6569
   pragma Import (C, nppiDiv_8u_C4RSfs, "nppiDiv_8u_C4RSfs");

  --* 
  -- * Four 8-bit unsigned char channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDiv_8u_C4IRSfs
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:6583
   pragma Import (C, nppiDiv_8u_C4IRSfs, "nppiDiv_8u_C4IRSfs");

  --* 
  -- * One 16-bit unsigned short channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDiv_16u_C1RSfs
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp16u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:6599
   pragma Import (C, nppiDiv_16u_C1RSfs, "nppiDiv_16u_C1RSfs");

  --* 
  -- * One 16-bit unsigned short channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDiv_16u_C1IRSfs
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:6613
   pragma Import (C, nppiDiv_16u_C1IRSfs, "nppiDiv_16u_C1IRSfs");

  --* 
  -- * Three 16-bit unsigned short channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDiv_16u_C3RSfs
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp16u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:6629
   pragma Import (C, nppiDiv_16u_C3RSfs, "nppiDiv_16u_C3RSfs");

  --* 
  -- * Three 16-bit unsigned short channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDiv_16u_C3IRSfs
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:6643
   pragma Import (C, nppiDiv_16u_C3IRSfs, "nppiDiv_16u_C3IRSfs");

  --* 
  -- * Four 16-bit unsigned short channel with unmodified alpha image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDiv_16u_AC4RSfs
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp16u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:6659
   pragma Import (C, nppiDiv_16u_AC4RSfs, "nppiDiv_16u_AC4RSfs");

  --* 
  -- * Four 16-bit unsigned short channel with unmodified alpha in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDiv_16u_AC4IRSfs
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:6673
   pragma Import (C, nppiDiv_16u_AC4IRSfs, "nppiDiv_16u_AC4IRSfs");

  --* 
  -- * Four 16-bit unsigned short channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDiv_16u_C4RSfs
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp16u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:6689
   pragma Import (C, nppiDiv_16u_C4RSfs, "nppiDiv_16u_C4RSfs");

  --* 
  -- * Four 16-bit unsigned short channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDiv_16u_C4IRSfs
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:6703
   pragma Import (C, nppiDiv_16u_C4IRSfs, "nppiDiv_16u_C4IRSfs");

  --* 
  -- * One 16-bit signed short channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDiv_16s_C1RSfs
     (pSrc1 : access nppdefs_h.Npp16s;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp16s;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:6719
   pragma Import (C, nppiDiv_16s_C1RSfs, "nppiDiv_16s_C1RSfs");

  --* 
  -- * One 16-bit signed short channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDiv_16s_C1IRSfs
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:6733
   pragma Import (C, nppiDiv_16s_C1IRSfs, "nppiDiv_16s_C1IRSfs");

  --* 
  -- * Three 16-bit signed short channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDiv_16s_C3RSfs
     (pSrc1 : access nppdefs_h.Npp16s;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp16s;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:6749
   pragma Import (C, nppiDiv_16s_C3RSfs, "nppiDiv_16s_C3RSfs");

  --* 
  -- * Three 16-bit signed short channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDiv_16s_C3IRSfs
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:6763
   pragma Import (C, nppiDiv_16s_C3IRSfs, "nppiDiv_16s_C3IRSfs");

  --* 
  -- * Four 16-bit signed short channel with unmodified alpha image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDiv_16s_AC4RSfs
     (pSrc1 : access nppdefs_h.Npp16s;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp16s;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:6779
   pragma Import (C, nppiDiv_16s_AC4RSfs, "nppiDiv_16s_AC4RSfs");

  --* 
  -- * Four 16-bit signed short channel with unmodified alpha in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDiv_16s_AC4IRSfs
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:6793
   pragma Import (C, nppiDiv_16s_AC4IRSfs, "nppiDiv_16s_AC4IRSfs");

  --* 
  -- * Four 16-bit signed short channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDiv_16s_C4RSfs
     (pSrc1 : access nppdefs_h.Npp16s;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp16s;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:6809
   pragma Import (C, nppiDiv_16s_C4RSfs, "nppiDiv_16s_C4RSfs");

  --* 
  -- * Four 16-bit signed short channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDiv_16s_C4IRSfs
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:6823
   pragma Import (C, nppiDiv_16s_C4IRSfs, "nppiDiv_16s_C4IRSfs");

  --* 
  -- * One 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDiv_16sc_C1RSfs
     (pSrc1 : access constant nppdefs_h.Npp16sc;
      nSrc1Step : int;
      pSrc2 : access constant nppdefs_h.Npp16sc;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp16sc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:6839
   pragma Import (C, nppiDiv_16sc_C1RSfs, "nppiDiv_16sc_C1RSfs");

  --* 
  -- * One 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDiv_16sc_C1IRSfs
     (pSrc : access constant nppdefs_h.Npp16sc;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp16sc;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:6853
   pragma Import (C, nppiDiv_16sc_C1IRSfs, "nppiDiv_16sc_C1IRSfs");

  --* 
  -- * Three 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDiv_16sc_C3RSfs
     (pSrc1 : access constant nppdefs_h.Npp16sc;
      nSrc1Step : int;
      pSrc2 : access constant nppdefs_h.Npp16sc;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp16sc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:6869
   pragma Import (C, nppiDiv_16sc_C3RSfs, "nppiDiv_16sc_C3RSfs");

  --* 
  -- * Three 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDiv_16sc_C3IRSfs
     (pSrc : access constant nppdefs_h.Npp16sc;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp16sc;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:6883
   pragma Import (C, nppiDiv_16sc_C3IRSfs, "nppiDiv_16sc_C3IRSfs");

  --* 
  -- * Four 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel with unmodified alpha image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDiv_16sc_AC4RSfs
     (pSrc1 : access constant nppdefs_h.Npp16sc;
      nSrc1Step : int;
      pSrc2 : access constant nppdefs_h.Npp16sc;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp16sc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:6899
   pragma Import (C, nppiDiv_16sc_AC4RSfs, "nppiDiv_16sc_AC4RSfs");

  --* 
  -- * Four 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel with unmodified alpha in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDiv_16sc_AC4IRSfs
     (pSrc : access constant nppdefs_h.Npp16sc;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp16sc;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:6913
   pragma Import (C, nppiDiv_16sc_AC4IRSfs, "nppiDiv_16sc_AC4IRSfs");

  --* 
  -- * One 32-bit signed integer channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDiv_32s_C1RSfs
     (pSrc1 : access nppdefs_h.Npp32s;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp32s;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:6929
   pragma Import (C, nppiDiv_32s_C1RSfs, "nppiDiv_32s_C1RSfs");

  --*
  -- * Note: This function is to be deprecated in future NPP releases, use the function above with a scale factor of 0 instead. 
  -- * 32-bit image division.
  -- * Divide pixels in pSrc2 by pSrc1's pixels. 
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDiv_32s_C1R
     (pSrc1 : access nppdefs_h.Npp32s;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp32s;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:6945
   pragma Import (C, nppiDiv_32s_C1R, "nppiDiv_32s_C1R");

  --* 
  -- * One 32-bit signed integer channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDiv_32s_C1IRSfs
     (pSrc : access nppdefs_h.Npp32s;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp32s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:6961
   pragma Import (C, nppiDiv_32s_C1IRSfs, "nppiDiv_32s_C1IRSfs");

  --* 
  -- * Three 32-bit signed integer channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDiv_32s_C3RSfs
     (pSrc1 : access nppdefs_h.Npp32s;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp32s;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:6977
   pragma Import (C, nppiDiv_32s_C3RSfs, "nppiDiv_32s_C3RSfs");

  --* 
  -- * Three 32-bit signed integer channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDiv_32s_C3IRSfs
     (pSrc : access nppdefs_h.Npp32s;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp32s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:6991
   pragma Import (C, nppiDiv_32s_C3IRSfs, "nppiDiv_32s_C3IRSfs");

  --* 
  -- * One 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDiv_32sc_C1RSfs
     (pSrc1 : access constant nppdefs_h.Npp32sc;
      nSrc1Step : int;
      pSrc2 : access constant nppdefs_h.Npp32sc;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp32sc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:7007
   pragma Import (C, nppiDiv_32sc_C1RSfs, "nppiDiv_32sc_C1RSfs");

  --* 
  -- * One 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDiv_32sc_C1IRSfs
     (pSrc : access constant nppdefs_h.Npp32sc;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp32sc;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:7021
   pragma Import (C, nppiDiv_32sc_C1IRSfs, "nppiDiv_32sc_C1IRSfs");

  --* 
  -- * Three 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDiv_32sc_C3RSfs
     (pSrc1 : access constant nppdefs_h.Npp32sc;
      nSrc1Step : int;
      pSrc2 : access constant nppdefs_h.Npp32sc;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp32sc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:7037
   pragma Import (C, nppiDiv_32sc_C3RSfs, "nppiDiv_32sc_C3RSfs");

  --* 
  -- * Three 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDiv_32sc_C3IRSfs
     (pSrc : access constant nppdefs_h.Npp32sc;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp32sc;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:7051
   pragma Import (C, nppiDiv_32sc_C3IRSfs, "nppiDiv_32sc_C3IRSfs");

  --* 
  -- * Four 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel with unmodified alpha image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDiv_32sc_AC4RSfs
     (pSrc1 : access constant nppdefs_h.Npp32sc;
      nSrc1Step : int;
      pSrc2 : access constant nppdefs_h.Npp32sc;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp32sc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:7067
   pragma Import (C, nppiDiv_32sc_AC4RSfs, "nppiDiv_32sc_AC4RSfs");

  --* 
  -- * Four 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel with unmodified alpha in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDiv_32sc_AC4IRSfs
     (pSrc : access constant nppdefs_h.Npp32sc;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp32sc;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:7081
   pragma Import (C, nppiDiv_32sc_AC4IRSfs, "nppiDiv_32sc_AC4IRSfs");

  --* 
  -- * One 32-bit floating point channel image division.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDiv_32f_C1R
     (pSrc1 : access nppdefs_h.Npp32f;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp32f;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:7096
   pragma Import (C, nppiDiv_32f_C1R, "nppiDiv_32f_C1R");

  --* 
  -- * One 32-bit floating point channel in place image division.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDiv_32f_C1IR
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:7109
   pragma Import (C, nppiDiv_32f_C1IR, "nppiDiv_32f_C1IR");

  --* 
  -- * Three 32-bit floating point channel image division.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDiv_32f_C3R
     (pSrc1 : access nppdefs_h.Npp32f;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp32f;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:7124
   pragma Import (C, nppiDiv_32f_C3R, "nppiDiv_32f_C3R");

  --* 
  -- * Three 32-bit floating point channel in place image division.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDiv_32f_C3IR
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:7137
   pragma Import (C, nppiDiv_32f_C3IR, "nppiDiv_32f_C3IR");

  --* 
  -- * Four 32-bit floating point channel with unmodified alpha image division.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDiv_32f_AC4R
     (pSrc1 : access nppdefs_h.Npp32f;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp32f;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:7152
   pragma Import (C, nppiDiv_32f_AC4R, "nppiDiv_32f_AC4R");

  --* 
  -- * Four 32-bit floating point channel with unmodified alpha in place image division.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDiv_32f_AC4IR
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:7165
   pragma Import (C, nppiDiv_32f_AC4IR, "nppiDiv_32f_AC4IR");

  --* 
  -- * Four 32-bit floating point channel image division.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDiv_32f_C4R
     (pSrc1 : access nppdefs_h.Npp32f;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp32f;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:7180
   pragma Import (C, nppiDiv_32f_C4R, "nppiDiv_32f_C4R");

  --* 
  -- * Four 32-bit floating point channel in place image division.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDiv_32f_C4IR
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:7193
   pragma Import (C, nppiDiv_32f_C4IR, "nppiDiv_32f_C4IR");

  --* 
  -- * One 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel image division.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDiv_32fc_C1R
     (pSrc1 : access constant nppdefs_h.Npp32fc;
      nSrc1Step : int;
      pSrc2 : access constant nppdefs_h.Npp32fc;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp32fc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:7208
   pragma Import (C, nppiDiv_32fc_C1R, "nppiDiv_32fc_C1R");

  --* 
  -- * One 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel in place image division.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDiv_32fc_C1IR
     (pSrc : access constant nppdefs_h.Npp32fc;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp32fc;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:7221
   pragma Import (C, nppiDiv_32fc_C1IR, "nppiDiv_32fc_C1IR");

  --* 
  -- * Three 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel image division.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDiv_32fc_C3R
     (pSrc1 : access constant nppdefs_h.Npp32fc;
      nSrc1Step : int;
      pSrc2 : access constant nppdefs_h.Npp32fc;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp32fc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:7236
   pragma Import (C, nppiDiv_32fc_C3R, "nppiDiv_32fc_C3R");

  --* 
  -- * Three 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel in place image division.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDiv_32fc_C3IR
     (pSrc : access constant nppdefs_h.Npp32fc;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp32fc;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:7249
   pragma Import (C, nppiDiv_32fc_C3IR, "nppiDiv_32fc_C3IR");

  --* 
  -- * Four 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel with unmodified alpha image division.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDiv_32fc_AC4R
     (pSrc1 : access constant nppdefs_h.Npp32fc;
      nSrc1Step : int;
      pSrc2 : access constant nppdefs_h.Npp32fc;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp32fc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:7264
   pragma Import (C, nppiDiv_32fc_AC4R, "nppiDiv_32fc_AC4R");

  --* 
  -- * Four 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel with unmodified alpha in place image division.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDiv_32fc_AC4IR
     (pSrc : access constant nppdefs_h.Npp32fc;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp32fc;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:7277
   pragma Import (C, nppiDiv_32fc_AC4IR, "nppiDiv_32fc_AC4IR");

  --* 
  -- * Four 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel image division.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDiv_32fc_C4R
     (pSrc1 : access constant nppdefs_h.Npp32fc;
      nSrc1Step : int;
      pSrc2 : access constant nppdefs_h.Npp32fc;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp32fc;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:7292
   pragma Import (C, nppiDiv_32fc_C4R, "nppiDiv_32fc_C4R");

  --* 
  -- * Four 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel in place image division.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDiv_32fc_C4IR
     (pSrc : access constant nppdefs_h.Npp32fc;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp32fc;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:7305
   pragma Import (C, nppiDiv_32fc_C4IR, "nppiDiv_32fc_C4IR");

  --* @} image_div  
  --* 
  -- * @defgroup image_divround Div_Round
  -- *
  -- * Pixel by pixel division of two images using result rounding modes.
  -- *
  -- * @{
  --  

  --* 
  -- * One 8-bit unsigned char channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rndMode Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDiv_Round_8u_C1RSfs
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp8u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rndMode : nppdefs_h.NppRoundMode;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:7332
   pragma Import (C, nppiDiv_Round_8u_C1RSfs, "nppiDiv_Round_8u_C1RSfs");

  --* 
  -- * One 8-bit unsigned char channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rndMode Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDiv_Round_8u_C1IRSfs
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rndMode : nppdefs_h.NppRoundMode;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:7347
   pragma Import (C, nppiDiv_Round_8u_C1IRSfs, "nppiDiv_Round_8u_C1IRSfs");

  --* 
  -- * Three 8-bit unsigned char channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rndMode Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDiv_Round_8u_C3RSfs
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp8u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rndMode : nppdefs_h.NppRoundMode;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:7364
   pragma Import (C, nppiDiv_Round_8u_C3RSfs, "nppiDiv_Round_8u_C3RSfs");

  --* 
  -- * Three 8-bit unsigned char channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rndMode Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDiv_Round_8u_C3IRSfs
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rndMode : nppdefs_h.NppRoundMode;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:7379
   pragma Import (C, nppiDiv_Round_8u_C3IRSfs, "nppiDiv_Round_8u_C3IRSfs");

  --* 
  -- * Four 8-bit unsigned char channel image division with unmodified alpha, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rndMode Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDiv_Round_8u_AC4RSfs
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp8u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rndMode : nppdefs_h.NppRoundMode;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:7396
   pragma Import (C, nppiDiv_Round_8u_AC4RSfs, "nppiDiv_Round_8u_AC4RSfs");

  --* 
  -- * Four 8-bit unsigned char channel in place image division with unmodified alpha, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rndMode Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDiv_Round_8u_AC4IRSfs
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rndMode : nppdefs_h.NppRoundMode;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:7411
   pragma Import (C, nppiDiv_Round_8u_AC4IRSfs, "nppiDiv_Round_8u_AC4IRSfs");

  --* 
  -- * Four 8-bit unsigned char channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rndMode Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDiv_Round_8u_C4RSfs
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp8u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rndMode : nppdefs_h.NppRoundMode;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:7428
   pragma Import (C, nppiDiv_Round_8u_C4RSfs, "nppiDiv_Round_8u_C4RSfs");

  --* 
  -- * Four 8-bit unsigned char channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rndMode Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDiv_Round_8u_C4IRSfs
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rndMode : nppdefs_h.NppRoundMode;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:7443
   pragma Import (C, nppiDiv_Round_8u_C4IRSfs, "nppiDiv_Round_8u_C4IRSfs");

  --* 
  -- * One 16-bit unsigned short channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rndMode Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDiv_Round_16u_C1RSfs
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp16u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rndMode : nppdefs_h.NppRoundMode;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:7460
   pragma Import (C, nppiDiv_Round_16u_C1RSfs, "nppiDiv_Round_16u_C1RSfs");

  --* 
  -- * One 16-bit unsigned short channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rndMode Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDiv_Round_16u_C1IRSfs
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rndMode : nppdefs_h.NppRoundMode;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:7475
   pragma Import (C, nppiDiv_Round_16u_C1IRSfs, "nppiDiv_Round_16u_C1IRSfs");

  --* 
  -- * Three 16-bit unsigned short channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rndMode Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDiv_Round_16u_C3RSfs
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp16u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rndMode : nppdefs_h.NppRoundMode;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:7492
   pragma Import (C, nppiDiv_Round_16u_C3RSfs, "nppiDiv_Round_16u_C3RSfs");

  --* 
  -- * Three 16-bit unsigned short channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rndMode Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDiv_Round_16u_C3IRSfs
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rndMode : nppdefs_h.NppRoundMode;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:7507
   pragma Import (C, nppiDiv_Round_16u_C3IRSfs, "nppiDiv_Round_16u_C3IRSfs");

  --* 
  -- * Four 16-bit unsigned short channel image division with unmodified alpha, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rndMode Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDiv_Round_16u_AC4RSfs
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp16u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rndMode : nppdefs_h.NppRoundMode;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:7524
   pragma Import (C, nppiDiv_Round_16u_AC4RSfs, "nppiDiv_Round_16u_AC4RSfs");

  --* 
  -- * Four 16-bit unsigned short channel in place image division with unmodified alpha, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rndMode Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDiv_Round_16u_AC4IRSfs
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rndMode : nppdefs_h.NppRoundMode;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:7539
   pragma Import (C, nppiDiv_Round_16u_AC4IRSfs, "nppiDiv_Round_16u_AC4IRSfs");

  --* 
  -- * Four 16-bit unsigned short channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rndMode Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDiv_Round_16u_C4RSfs
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp16u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rndMode : nppdefs_h.NppRoundMode;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:7556
   pragma Import (C, nppiDiv_Round_16u_C4RSfs, "nppiDiv_Round_16u_C4RSfs");

  --* 
  -- * Four 16-bit unsigned short channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rndMode Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDiv_Round_16u_C4IRSfs
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rndMode : nppdefs_h.NppRoundMode;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:7571
   pragma Import (C, nppiDiv_Round_16u_C4IRSfs, "nppiDiv_Round_16u_C4IRSfs");

  --* 
  -- * One 16-bit signed short channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rndMode Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDiv_Round_16s_C1RSfs
     (pSrc1 : access nppdefs_h.Npp16s;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp16s;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rndMode : nppdefs_h.NppRoundMode;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:7588
   pragma Import (C, nppiDiv_Round_16s_C1RSfs, "nppiDiv_Round_16s_C1RSfs");

  --* 
  -- * One 16-bit signed short channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rndMode Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDiv_Round_16s_C1IRSfs
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rndMode : nppdefs_h.NppRoundMode;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:7603
   pragma Import (C, nppiDiv_Round_16s_C1IRSfs, "nppiDiv_Round_16s_C1IRSfs");

  --* 
  -- * Three 16-bit signed short channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rndMode Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDiv_Round_16s_C3RSfs
     (pSrc1 : access nppdefs_h.Npp16s;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp16s;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rndMode : nppdefs_h.NppRoundMode;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:7620
   pragma Import (C, nppiDiv_Round_16s_C3RSfs, "nppiDiv_Round_16s_C3RSfs");

  --* 
  -- * Three 16-bit signed short channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rndMode Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDiv_Round_16s_C3IRSfs
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rndMode : nppdefs_h.NppRoundMode;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:7635
   pragma Import (C, nppiDiv_Round_16s_C3IRSfs, "nppiDiv_Round_16s_C3IRSfs");

  --* 
  -- * Four 16-bit signed short channel image division with unmodified alpha, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rndMode Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDiv_Round_16s_AC4RSfs
     (pSrc1 : access nppdefs_h.Npp16s;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp16s;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rndMode : nppdefs_h.NppRoundMode;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:7652
   pragma Import (C, nppiDiv_Round_16s_AC4RSfs, "nppiDiv_Round_16s_AC4RSfs");

  --* 
  -- * Four 16-bit signed short channel in place image division with unmodified alpha, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rndMode Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDiv_Round_16s_AC4IRSfs
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rndMode : nppdefs_h.NppRoundMode;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:7667
   pragma Import (C, nppiDiv_Round_16s_AC4IRSfs, "nppiDiv_Round_16s_AC4IRSfs");

  --* 
  -- * Four 16-bit signed short channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rndMode Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDiv_Round_16s_C4RSfs
     (pSrc1 : access nppdefs_h.Npp16s;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp16s;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rndMode : nppdefs_h.NppRoundMode;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:7684
   pragma Import (C, nppiDiv_Round_16s_C4RSfs, "nppiDiv_Round_16s_C4RSfs");

  --* 
  -- * Four 16-bit signed short channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param rndMode Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiDiv_Round_16s_C4IRSfs
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      rndMode : nppdefs_h.NppRoundMode;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:7699
   pragma Import (C, nppiDiv_Round_16s_C4IRSfs, "nppiDiv_Round_16s_C4IRSfs");

  --* @} image_divround  
  --* 
  -- * @defgroup image_abs Abs
  -- *
  -- * Absolute value of each pixel value in an image.
  -- *
  -- * @{
  --  

  --* 
  -- * One 16-bit signed short channel image absolute value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAbs_16s_C1R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:7722
   pragma Import (C, nppiAbs_16s_C1R, "nppiAbs_16s_C1R");

  --* 
  -- * One 16-bit signed short channel in place image absolute value.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAbs_16s_C1IR
     (pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:7732
   pragma Import (C, nppiAbs_16s_C1IR, "nppiAbs_16s_C1IR");

  --* 
  -- * Three 16-bit signed short channel image absolute value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAbs_16s_C3R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:7744
   pragma Import (C, nppiAbs_16s_C3R, "nppiAbs_16s_C3R");

  --* 
  -- * Three 16-bit signed short channel in place image absolute value.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAbs_16s_C3IR
     (pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:7754
   pragma Import (C, nppiAbs_16s_C3IR, "nppiAbs_16s_C3IR");

  --* 
  -- * Four 16-bit signed short channel image absolute value with unmodified alpha.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAbs_16s_AC4R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:7766
   pragma Import (C, nppiAbs_16s_AC4R, "nppiAbs_16s_AC4R");

  --* 
  -- * Four 16-bit signed short channel in place image absolute value with unmodified alpha.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAbs_16s_AC4IR
     (pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:7776
   pragma Import (C, nppiAbs_16s_AC4IR, "nppiAbs_16s_AC4IR");

  --* 
  -- * Four 16-bit signed short channel image absolute value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAbs_16s_C4R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:7788
   pragma Import (C, nppiAbs_16s_C4R, "nppiAbs_16s_C4R");

  --* 
  -- * Four 16-bit signed short channel in place image absolute value.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAbs_16s_C4IR
     (pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:7798
   pragma Import (C, nppiAbs_16s_C4IR, "nppiAbs_16s_C4IR");

  --* 
  -- * One 32-bit floating point channel image absolute value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAbs_32f_C1R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:7810
   pragma Import (C, nppiAbs_32f_C1R, "nppiAbs_32f_C1R");

  --* 
  -- * One 32-bit floating point channel in place image absolute value.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAbs_32f_C1IR
     (pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:7820
   pragma Import (C, nppiAbs_32f_C1IR, "nppiAbs_32f_C1IR");

  --* 
  -- * Three 32-bit floating point channel image absolute value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAbs_32f_C3R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:7832
   pragma Import (C, nppiAbs_32f_C3R, "nppiAbs_32f_C3R");

  --* 
  -- * Three 32-bit floating point channel in place image absolute value.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAbs_32f_C3IR
     (pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:7842
   pragma Import (C, nppiAbs_32f_C3IR, "nppiAbs_32f_C3IR");

  --* 
  -- * Four 32-bit floating point channel image absolute value with unmodified alpha.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAbs_32f_AC4R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:7854
   pragma Import (C, nppiAbs_32f_AC4R, "nppiAbs_32f_AC4R");

  --* 
  -- * Four 32-bit floating point channel in place image absolute value with unmodified alpha.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAbs_32f_AC4IR
     (pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:7864
   pragma Import (C, nppiAbs_32f_AC4IR, "nppiAbs_32f_AC4IR");

  --* 
  -- * Four 32-bit floating point channel image absolute value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAbs_32f_C4R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:7876
   pragma Import (C, nppiAbs_32f_C4R, "nppiAbs_32f_C4R");

  --* 
  -- * Four 32-bit floating point channel in place image absolute value.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAbs_32f_C4IR
     (pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:7886
   pragma Import (C, nppiAbs_32f_C4IR, "nppiAbs_32f_C4IR");

  --* @} image_abs  
  --* 
  -- * @defgroup image_absdiff AbsDiff
  -- *
  -- * Pixel by pixel absolute difference between two images.
  -- *
  -- * @{
  --  

  --* 
  -- * One 8-bit unsigned char channel absolute difference of image1 minus image2.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAbsDiff_8u_C1R
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp8u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:7910
   pragma Import (C, nppiAbsDiff_8u_C1R, "nppiAbsDiff_8u_C1R");

  --* 
  -- * Three 8-bit unsigned char channels absolute difference of image1 minus image2.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAbsDiff_8u_C3R
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp8u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:7924
   pragma Import (C, nppiAbsDiff_8u_C3R, "nppiAbsDiff_8u_C3R");

  --* 
  -- * Four 8-bit unsigned char channels absolute difference of image1 minus image2.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAbsDiff_8u_C4R
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp8u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:7938
   pragma Import (C, nppiAbsDiff_8u_C4R, "nppiAbsDiff_8u_C4R");

  --* 
  -- * One 16-bit unsigned short channel absolute difference of image1 minus image2.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAbsDiff_16u_C1R
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp16u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:7952
   pragma Import (C, nppiAbsDiff_16u_C1R, "nppiAbsDiff_16u_C1R");

  --* 
  -- * One 32-bit floating point channel absolute difference of image1 minus image2.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAbsDiff_32f_C1R
     (pSrc1 : access nppdefs_h.Npp32f;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp32f;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:7966
   pragma Import (C, nppiAbsDiff_32f_C1R, "nppiAbsDiff_32f_C1R");

  --* @} image_absdiff  
  --* 
  -- * @defgroup image_sqr Sqr
  -- *
  -- * Square each pixel in an image.
  -- *
  -- * @{
  --  

  --* 
  -- * One 8-bit unsigned char channel image squared, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSqr_8u_C1RSfs
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:7989
   pragma Import (C, nppiSqr_8u_C1RSfs, "nppiSqr_8u_C1RSfs");

  --* 
  -- * One 8-bit unsigned char channel in place image squared, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSqr_8u_C1IRSfs
     (pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:8000
   pragma Import (C, nppiSqr_8u_C1IRSfs, "nppiSqr_8u_C1IRSfs");

  --* 
  -- * Three 8-bit unsigned char channel image squared, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSqr_8u_C3RSfs
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:8013
   pragma Import (C, nppiSqr_8u_C3RSfs, "nppiSqr_8u_C3RSfs");

  --* 
  -- * Three 8-bit unsigned char channel in place image squared, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSqr_8u_C3IRSfs
     (pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:8024
   pragma Import (C, nppiSqr_8u_C3IRSfs, "nppiSqr_8u_C3IRSfs");

  --* 
  -- * Four 8-bit unsigned char channel image squared with unmodified alpha, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSqr_8u_AC4RSfs
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:8037
   pragma Import (C, nppiSqr_8u_AC4RSfs, "nppiSqr_8u_AC4RSfs");

  --* 
  -- * Four 8-bit unsigned char channel in place image squared with unmodified alpha, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSqr_8u_AC4IRSfs
     (pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:8048
   pragma Import (C, nppiSqr_8u_AC4IRSfs, "nppiSqr_8u_AC4IRSfs");

  --* 
  -- * Four 8-bit unsigned char channel image squared, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSqr_8u_C4RSfs
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:8061
   pragma Import (C, nppiSqr_8u_C4RSfs, "nppiSqr_8u_C4RSfs");

  --* 
  -- * Four 8-bit unsigned char channel in place image squared, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSqr_8u_C4IRSfs
     (pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:8072
   pragma Import (C, nppiSqr_8u_C4IRSfs, "nppiSqr_8u_C4IRSfs");

  --* 
  -- * One 16-bit unsigned short channel image squared, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSqr_16u_C1RSfs
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:8085
   pragma Import (C, nppiSqr_16u_C1RSfs, "nppiSqr_16u_C1RSfs");

  --* 
  -- * One 16-bit unsigned short channel in place image squared, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSqr_16u_C1IRSfs
     (pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:8096
   pragma Import (C, nppiSqr_16u_C1IRSfs, "nppiSqr_16u_C1IRSfs");

  --* 
  -- * Three 16-bit unsigned short channel image squared, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSqr_16u_C3RSfs
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:8109
   pragma Import (C, nppiSqr_16u_C3RSfs, "nppiSqr_16u_C3RSfs");

  --* 
  -- * Three 16-bit unsigned short channel in place image squared, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSqr_16u_C3IRSfs
     (pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:8120
   pragma Import (C, nppiSqr_16u_C3IRSfs, "nppiSqr_16u_C3IRSfs");

  --* 
  -- * Four 16-bit unsigned short channel image squared with unmodified alpha, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSqr_16u_AC4RSfs
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:8133
   pragma Import (C, nppiSqr_16u_AC4RSfs, "nppiSqr_16u_AC4RSfs");

  --* 
  -- * Four 16-bit unsigned short channel in place image squared with unmodified alpha, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSqr_16u_AC4IRSfs
     (pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:8144
   pragma Import (C, nppiSqr_16u_AC4IRSfs, "nppiSqr_16u_AC4IRSfs");

  --* 
  -- * Four 16-bit unsigned short channel image squared, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSqr_16u_C4RSfs
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:8157
   pragma Import (C, nppiSqr_16u_C4RSfs, "nppiSqr_16u_C4RSfs");

  --* 
  -- * Four 16-bit unsigned short channel in place image squared, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSqr_16u_C4IRSfs
     (pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:8168
   pragma Import (C, nppiSqr_16u_C4IRSfs, "nppiSqr_16u_C4IRSfs");

  --* 
  -- * One 16-bit signed short channel image squared, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSqr_16s_C1RSfs
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:8181
   pragma Import (C, nppiSqr_16s_C1RSfs, "nppiSqr_16s_C1RSfs");

  --* 
  -- * One 16-bit signed short channel in place image squared, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSqr_16s_C1IRSfs
     (pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:8192
   pragma Import (C, nppiSqr_16s_C1IRSfs, "nppiSqr_16s_C1IRSfs");

  --* 
  -- * Three 16-bit signed short channel image squared, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSqr_16s_C3RSfs
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:8205
   pragma Import (C, nppiSqr_16s_C3RSfs, "nppiSqr_16s_C3RSfs");

  --* 
  -- * Three 16-bit signed short channel in place image squared, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSqr_16s_C3IRSfs
     (pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:8216
   pragma Import (C, nppiSqr_16s_C3IRSfs, "nppiSqr_16s_C3IRSfs");

  --* 
  -- * Four 16-bit signed short channel image squared with unmodified alpha, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSqr_16s_AC4RSfs
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:8229
   pragma Import (C, nppiSqr_16s_AC4RSfs, "nppiSqr_16s_AC4RSfs");

  --* 
  -- * Four 16-bit signed short channel in place image squared with unmodified alpha, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSqr_16s_AC4IRSfs
     (pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:8240
   pragma Import (C, nppiSqr_16s_AC4IRSfs, "nppiSqr_16s_AC4IRSfs");

  --* 
  -- * Four 16-bit signed short channel image squared, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSqr_16s_C4RSfs
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:8253
   pragma Import (C, nppiSqr_16s_C4RSfs, "nppiSqr_16s_C4RSfs");

  --* 
  -- * Four 16-bit signed short channel in place image squared, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSqr_16s_C4IRSfs
     (pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:8264
   pragma Import (C, nppiSqr_16s_C4IRSfs, "nppiSqr_16s_C4IRSfs");

  --* 
  -- * One 32-bit floating point channel image squared.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSqr_32f_C1R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:8276
   pragma Import (C, nppiSqr_32f_C1R, "nppiSqr_32f_C1R");

  --* 
  -- * One 32-bit floating point channel in place image squared.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSqr_32f_C1IR
     (pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:8286
   pragma Import (C, nppiSqr_32f_C1IR, "nppiSqr_32f_C1IR");

  --* 
  -- * Three 32-bit floating point channel image squared.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSqr_32f_C3R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:8298
   pragma Import (C, nppiSqr_32f_C3R, "nppiSqr_32f_C3R");

  --* 
  -- * Three 32-bit floating point channel in place image squared.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSqr_32f_C3IR
     (pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:8308
   pragma Import (C, nppiSqr_32f_C3IR, "nppiSqr_32f_C3IR");

  --* 
  -- * Four 32-bit floating point channel image squared with unmodified alpha.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSqr_32f_AC4R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:8320
   pragma Import (C, nppiSqr_32f_AC4R, "nppiSqr_32f_AC4R");

  --* 
  -- * Four 32-bit floating point channel in place image squared with unmodified alpha.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSqr_32f_AC4IR
     (pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:8330
   pragma Import (C, nppiSqr_32f_AC4IR, "nppiSqr_32f_AC4IR");

  --* 
  -- * Four 32-bit floating point channel image squared.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSqr_32f_C4R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:8342
   pragma Import (C, nppiSqr_32f_C4R, "nppiSqr_32f_C4R");

  --* 
  -- * Four 32-bit floating point channel in place image squared.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSqr_32f_C4IR
     (pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:8352
   pragma Import (C, nppiSqr_32f_C4IR, "nppiSqr_32f_C4IR");

  --* @} image_sqr  
  --* @defgroup image_sqrt Sqrt
  -- *
  -- * Pixel by pixel square root of each pixel in an image.
  -- *
  -- * @{
  --  

  --* 
  -- * One 8-bit unsigned char channel image square root, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSqrt_8u_C1RSfs
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:8374
   pragma Import (C, nppiSqrt_8u_C1RSfs, "nppiSqrt_8u_C1RSfs");

  --* 
  -- * One 8-bit unsigned char channel in place image square root, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSqrt_8u_C1IRSfs
     (pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:8385
   pragma Import (C, nppiSqrt_8u_C1IRSfs, "nppiSqrt_8u_C1IRSfs");

  --* 
  -- * Three 8-bit unsigned char channel image square root, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSqrt_8u_C3RSfs
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:8398
   pragma Import (C, nppiSqrt_8u_C3RSfs, "nppiSqrt_8u_C3RSfs");

  --* 
  -- * Three 8-bit unsigned char channel in place image square root, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSqrt_8u_C3IRSfs
     (pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:8409
   pragma Import (C, nppiSqrt_8u_C3IRSfs, "nppiSqrt_8u_C3IRSfs");

  --* 
  -- * Four 8-bit unsigned char channel image square root with unmodified alpha, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSqrt_8u_AC4RSfs
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:8422
   pragma Import (C, nppiSqrt_8u_AC4RSfs, "nppiSqrt_8u_AC4RSfs");

  --* 
  -- * Four 8-bit unsigned char channel in place image square root with unmodified alpha, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSqrt_8u_AC4IRSfs
     (pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:8433
   pragma Import (C, nppiSqrt_8u_AC4IRSfs, "nppiSqrt_8u_AC4IRSfs");

  --* 
  -- * One 16-bit unsigned short channel image square root, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSqrt_16u_C1RSfs
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:8446
   pragma Import (C, nppiSqrt_16u_C1RSfs, "nppiSqrt_16u_C1RSfs");

  --* 
  -- * One 16-bit unsigned short channel in place image square root, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSqrt_16u_C1IRSfs
     (pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:8457
   pragma Import (C, nppiSqrt_16u_C1IRSfs, "nppiSqrt_16u_C1IRSfs");

  --* 
  -- * Three 16-bit unsigned short channel image square root, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSqrt_16u_C3RSfs
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:8470
   pragma Import (C, nppiSqrt_16u_C3RSfs, "nppiSqrt_16u_C3RSfs");

  --* 
  -- * Three 16-bit unsigned short channel in place image square root, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSqrt_16u_C3IRSfs
     (pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:8481
   pragma Import (C, nppiSqrt_16u_C3IRSfs, "nppiSqrt_16u_C3IRSfs");

  --* 
  -- * Four 16-bit unsigned short channel image square root with unmodified alpha, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSqrt_16u_AC4RSfs
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:8494
   pragma Import (C, nppiSqrt_16u_AC4RSfs, "nppiSqrt_16u_AC4RSfs");

  --* 
  -- * Four 16-bit unsigned short channel in place image square root with unmodified alpha, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSqrt_16u_AC4IRSfs
     (pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:8505
   pragma Import (C, nppiSqrt_16u_AC4IRSfs, "nppiSqrt_16u_AC4IRSfs");

  --* 
  -- * One 16-bit signed short channel image square root, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSqrt_16s_C1RSfs
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:8518
   pragma Import (C, nppiSqrt_16s_C1RSfs, "nppiSqrt_16s_C1RSfs");

  --* 
  -- * One 16-bit signed short channel in place image square root, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSqrt_16s_C1IRSfs
     (pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:8529
   pragma Import (C, nppiSqrt_16s_C1IRSfs, "nppiSqrt_16s_C1IRSfs");

  --* 
  -- * Three 16-bit signed short channel image square root, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSqrt_16s_C3RSfs
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:8542
   pragma Import (C, nppiSqrt_16s_C3RSfs, "nppiSqrt_16s_C3RSfs");

  --* 
  -- * Three 16-bit signed short channel in place image square root, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSqrt_16s_C3IRSfs
     (pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:8553
   pragma Import (C, nppiSqrt_16s_C3IRSfs, "nppiSqrt_16s_C3IRSfs");

  --* 
  -- * Four 16-bit signed short channel image square root with unmodified alpha, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSqrt_16s_AC4RSfs
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:8566
   pragma Import (C, nppiSqrt_16s_AC4RSfs, "nppiSqrt_16s_AC4RSfs");

  --* 
  -- * Four 16-bit signed short channel in place image square root with unmodified alpha, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSqrt_16s_AC4IRSfs
     (pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:8577
   pragma Import (C, nppiSqrt_16s_AC4IRSfs, "nppiSqrt_16s_AC4IRSfs");

  --* 
  -- * One 32-bit floating point channel image square root.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSqrt_32f_C1R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:8589
   pragma Import (C, nppiSqrt_32f_C1R, "nppiSqrt_32f_C1R");

  --* 
  -- * One 32-bit floating point channel in place image square root.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSqrt_32f_C1IR
     (pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:8599
   pragma Import (C, nppiSqrt_32f_C1IR, "nppiSqrt_32f_C1IR");

  --* 
  -- * Three 32-bit floating point channel image square root.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSqrt_32f_C3R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:8611
   pragma Import (C, nppiSqrt_32f_C3R, "nppiSqrt_32f_C3R");

  --* 
  -- * Three 32-bit floating point channel in place image square root.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSqrt_32f_C3IR
     (pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:8621
   pragma Import (C, nppiSqrt_32f_C3IR, "nppiSqrt_32f_C3IR");

  --* 
  -- * Four 32-bit floating point channel image square root with unmodified alpha.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSqrt_32f_AC4R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:8633
   pragma Import (C, nppiSqrt_32f_AC4R, "nppiSqrt_32f_AC4R");

  --* 
  -- * Four 32-bit floating point channel in place image square root with unmodified alpha.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSqrt_32f_AC4IR
     (pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:8643
   pragma Import (C, nppiSqrt_32f_AC4IR, "nppiSqrt_32f_AC4IR");

  --* 
  -- * Four 32-bit floating point channel image square root.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSqrt_32f_C4R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:8655
   pragma Import (C, nppiSqrt_32f_C4R, "nppiSqrt_32f_C4R");

  --* 
  -- * Four 32-bit floating point channel in place image square root.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiSqrt_32f_C4IR
     (pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:8665
   pragma Import (C, nppiSqrt_32f_C4IR, "nppiSqrt_32f_C4IR");

  --* @} image_sqrt  
  --* @defgroup image_ln Ln
  -- *
  -- * Pixel by pixel natural logarithm of each pixel in an image.
  -- *
  -- * @{
  --  

  --* 
  -- * One 8-bit unsigned char channel image natural logarithm, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiLn_8u_C1RSfs
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:8687
   pragma Import (C, nppiLn_8u_C1RSfs, "nppiLn_8u_C1RSfs");

  --* 
  -- * One 8-bit unsigned char channel in place image natural logarithm, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiLn_8u_C1IRSfs
     (pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:8698
   pragma Import (C, nppiLn_8u_C1IRSfs, "nppiLn_8u_C1IRSfs");

  --* 
  -- * Three 8-bit unsigned char channel image natural logarithm, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiLn_8u_C3RSfs
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:8711
   pragma Import (C, nppiLn_8u_C3RSfs, "nppiLn_8u_C3RSfs");

  --* 
  -- * Three 8-bit unsigned char channel in place image natural logarithm, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiLn_8u_C3IRSfs
     (pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:8722
   pragma Import (C, nppiLn_8u_C3IRSfs, "nppiLn_8u_C3IRSfs");

  --* 
  -- * One 16-bit unsigned short channel image natural logarithm, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiLn_16u_C1RSfs
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:8735
   pragma Import (C, nppiLn_16u_C1RSfs, "nppiLn_16u_C1RSfs");

  --* 
  -- * One 16-bit unsigned short channel in place image natural logarithm, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiLn_16u_C1IRSfs
     (pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:8746
   pragma Import (C, nppiLn_16u_C1IRSfs, "nppiLn_16u_C1IRSfs");

  --* 
  -- * Three 16-bit unsigned short channel image natural logarithm, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiLn_16u_C3RSfs
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:8759
   pragma Import (C, nppiLn_16u_C3RSfs, "nppiLn_16u_C3RSfs");

  --* 
  -- * Three 16-bit unsigned short channel in place image natural logarithm, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiLn_16u_C3IRSfs
     (pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:8770
   pragma Import (C, nppiLn_16u_C3IRSfs, "nppiLn_16u_C3IRSfs");

  --* 
  -- * One 16-bit signed short channel image natural logarithm, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiLn_16s_C1RSfs
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:8783
   pragma Import (C, nppiLn_16s_C1RSfs, "nppiLn_16s_C1RSfs");

  --* 
  -- * One 16-bit signed short channel in place image natural logarithm, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiLn_16s_C1IRSfs
     (pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:8794
   pragma Import (C, nppiLn_16s_C1IRSfs, "nppiLn_16s_C1IRSfs");

  --* 
  -- * Three 16-bit signed short channel image natural logarithm, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiLn_16s_C3RSfs
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:8807
   pragma Import (C, nppiLn_16s_C3RSfs, "nppiLn_16s_C3RSfs");

  --* 
  -- * Three 16-bit signed short channel in place image natural logarithm, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiLn_16s_C3IRSfs
     (pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:8818
   pragma Import (C, nppiLn_16s_C3IRSfs, "nppiLn_16s_C3IRSfs");

  --* 
  -- * One 32-bit floating point channel image natural logarithm.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiLn_32f_C1R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:8830
   pragma Import (C, nppiLn_32f_C1R, "nppiLn_32f_C1R");

  --* 
  -- * One 32-bit floating point channel in place image natural logarithm.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiLn_32f_C1IR
     (pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:8840
   pragma Import (C, nppiLn_32f_C1IR, "nppiLn_32f_C1IR");

  --* 
  -- * Three 32-bit floating point channel image natural logarithm.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiLn_32f_C3R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:8852
   pragma Import (C, nppiLn_32f_C3R, "nppiLn_32f_C3R");

  --* 
  -- * Three 32-bit floating point channel in place image natural logarithm.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiLn_32f_C3IR
     (pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:8862
   pragma Import (C, nppiLn_32f_C3IR, "nppiLn_32f_C3IR");

  --* @} image_ln  
  --* 
  -- * @defgroup image_exp Exp
  -- *
  -- * Exponential value of each pixel in an image.
  -- *
  -- * @{
  --  

  --* 
  -- * One 8-bit unsigned char channel image exponential, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiExp_8u_C1RSfs
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:8885
   pragma Import (C, nppiExp_8u_C1RSfs, "nppiExp_8u_C1RSfs");

  --* 
  -- * One 8-bit unsigned char channel in place image exponential, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiExp_8u_C1IRSfs
     (pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:8896
   pragma Import (C, nppiExp_8u_C1IRSfs, "nppiExp_8u_C1IRSfs");

  --* 
  -- * Three 8-bit unsigned char channel image exponential, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiExp_8u_C3RSfs
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:8909
   pragma Import (C, nppiExp_8u_C3RSfs, "nppiExp_8u_C3RSfs");

  --* 
  -- * Three 8-bit unsigned char channel in place image exponential, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiExp_8u_C3IRSfs
     (pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:8920
   pragma Import (C, nppiExp_8u_C3IRSfs, "nppiExp_8u_C3IRSfs");

  --* 
  -- * One 16-bit unsigned short channel image exponential, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiExp_16u_C1RSfs
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:8933
   pragma Import (C, nppiExp_16u_C1RSfs, "nppiExp_16u_C1RSfs");

  --* 
  -- * One 16-bit unsigned short channel in place image exponential, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiExp_16u_C1IRSfs
     (pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:8944
   pragma Import (C, nppiExp_16u_C1IRSfs, "nppiExp_16u_C1IRSfs");

  --* 
  -- * Three 16-bit unsigned short channel image exponential, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiExp_16u_C3RSfs
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:8957
   pragma Import (C, nppiExp_16u_C3RSfs, "nppiExp_16u_C3RSfs");

  --* 
  -- * Three 16-bit unsigned short channel in place image exponential, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiExp_16u_C3IRSfs
     (pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:8968
   pragma Import (C, nppiExp_16u_C3IRSfs, "nppiExp_16u_C3IRSfs");

  --* 
  -- * One 16-bit signed short channel image exponential, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiExp_16s_C1RSfs
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:8981
   pragma Import (C, nppiExp_16s_C1RSfs, "nppiExp_16s_C1RSfs");

  --* 
  -- * One 16-bit signed short channel in place image exponential, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiExp_16s_C1IRSfs
     (pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:8992
   pragma Import (C, nppiExp_16s_C1IRSfs, "nppiExp_16s_C1IRSfs");

  --* 
  -- * Three 16-bit signed short channel image exponential, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiExp_16s_C3RSfs
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:9005
   pragma Import (C, nppiExp_16s_C3RSfs, "nppiExp_16s_C3RSfs");

  --* 
  -- * Three 16-bit signed short channel in place image exponential, scale by 2^(-nScaleFactor), then clamp to saturated value.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param nScaleFactor \ref integer_result_scaling.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiExp_16s_C3IRSfs
     (pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      nScaleFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:9016
   pragma Import (C, nppiExp_16s_C3IRSfs, "nppiExp_16s_C3IRSfs");

  --* 
  -- * One 32-bit floating point channel image exponential.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiExp_32f_C1R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:9028
   pragma Import (C, nppiExp_32f_C1R, "nppiExp_32f_C1R");

  --* 
  -- * One 32-bit floating point channel in place image exponential.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiExp_32f_C1IR
     (pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:9038
   pragma Import (C, nppiExp_32f_C1IR, "nppiExp_32f_C1IR");

  --* 
  -- * Three 32-bit floating point channel image exponential.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiExp_32f_C3R
     (pSrc : access nppdefs_h.Npp32f;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:9050
   pragma Import (C, nppiExp_32f_C3R, "nppiExp_32f_C3R");

  --* 
  -- * Three 32-bit floating point channel in place image exponential.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiExp_32f_C3IR
     (pSrcDst : access nppdefs_h.Npp32f;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:9060
   pragma Import (C, nppiExp_32f_C3IR, "nppiExp_32f_C3IR");

  --* @} image_exp  
  --* @} image_arithmetic_operations  
  --* 
  -- * @defgroup image_logical_operations Logical Operations
  -- *
  -- * @{
  --  

  --* @defgroup image_andc AndC
  -- *
  -- * Pixel by pixel logical and of an image with a constant.
  -- *
  -- * @{
  --  

  --* 
  -- * One 8-bit unsigned char channel image logical and with constant.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param nConstant Constant.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAndC_8u_C1R
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      nConstant : nppdefs_h.Npp8u;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:9090
   pragma Import (C, nppiAndC_8u_C1R, "nppiAndC_8u_C1R");

  --* 
  -- * One 8-bit unsigned char channel in place image logical and with constant.
  -- * \param nConstant Constant.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAndC_8u_C1IR
     (nConstant : nppdefs_h.Npp8u;
      pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:9102
   pragma Import (C, nppiAndC_8u_C1IR, "nppiAndC_8u_C1IR");

  --* 
  -- * Three 8-bit unsigned char channel image logical and with constant.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAndC_8u_C3R
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp8u;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:9115
   pragma Import (C, nppiAndC_8u_C3R, "nppiAndC_8u_C3R");

  --* 
  -- * Three 8-bit unsigned char channel in place image logical and with constant.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAndC_8u_C3IR
     (aConstants : access nppdefs_h.Npp8u;
      pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:9127
   pragma Import (C, nppiAndC_8u_C3IR, "nppiAndC_8u_C3IR");

  --* 
  -- * Four 8-bit unsigned char channel image logical and with constant with unmodified alpha.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAndC_8u_AC4R
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp8u;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:9140
   pragma Import (C, nppiAndC_8u_AC4R, "nppiAndC_8u_AC4R");

  --* 
  -- * Four 8-bit unsigned char channel in place image logical and with constant with unmodified alpha.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAndC_8u_AC4IR
     (aConstants : access nppdefs_h.Npp8u;
      pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:9152
   pragma Import (C, nppiAndC_8u_AC4IR, "nppiAndC_8u_AC4IR");

  --* 
  -- * Four 8-bit unsigned char channel image logical and with constant.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAndC_8u_C4R
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp8u;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:9165
   pragma Import (C, nppiAndC_8u_C4R, "nppiAndC_8u_C4R");

  --* 
  -- * Four 8-bit unsigned char channel in place image logical and with constant.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAndC_8u_C4IR
     (aConstants : access nppdefs_h.Npp8u;
      pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:9177
   pragma Import (C, nppiAndC_8u_C4IR, "nppiAndC_8u_C4IR");

  --* 
  -- * One 16-bit unsigned short channel image logical and with constant.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param nConstant Constant.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAndC_16u_C1R
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      nConstant : nppdefs_h.Npp16u;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:9190
   pragma Import (C, nppiAndC_16u_C1R, "nppiAndC_16u_C1R");

  --* 
  -- * One 16-bit unsigned short channel in place image logical and with constant.
  -- * \param nConstant Constant.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAndC_16u_C1IR
     (nConstant : nppdefs_h.Npp16u;
      pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:9202
   pragma Import (C, nppiAndC_16u_C1IR, "nppiAndC_16u_C1IR");

  --* 
  -- * Three 16-bit unsigned short channel image logical and with constant.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAndC_16u_C3R
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp16u;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:9215
   pragma Import (C, nppiAndC_16u_C3R, "nppiAndC_16u_C3R");

  --* 
  -- * Three 16-bit unsigned short channel in place image logical and with constant.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAndC_16u_C3IR
     (aConstants : access nppdefs_h.Npp16u;
      pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:9227
   pragma Import (C, nppiAndC_16u_C3IR, "nppiAndC_16u_C3IR");

  --* 
  -- * Four 16-bit unsigned short channel image logical and with constant with unmodified alpha.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAndC_16u_AC4R
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp16u;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:9240
   pragma Import (C, nppiAndC_16u_AC4R, "nppiAndC_16u_AC4R");

  --* 
  -- * Four 16-bit unsigned short channel in place image logical and with constant with unmodified alpha.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAndC_16u_AC4IR
     (aConstants : access nppdefs_h.Npp16u;
      pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:9252
   pragma Import (C, nppiAndC_16u_AC4IR, "nppiAndC_16u_AC4IR");

  --* 
  -- * Four 16-bit unsigned short channel image logical and with constant.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAndC_16u_C4R
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp16u;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:9265
   pragma Import (C, nppiAndC_16u_C4R, "nppiAndC_16u_C4R");

  --* 
  -- * Four 16-bit unsigned short channel in place image logical and with constant.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAndC_16u_C4IR
     (aConstants : access nppdefs_h.Npp16u;
      pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:9277
   pragma Import (C, nppiAndC_16u_C4IR, "nppiAndC_16u_C4IR");

  --* 
  -- * One 32-bit signed integer channel image logical and with constant.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param nConstant Constant.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAndC_32s_C1R
     (pSrc1 : access nppdefs_h.Npp32s;
      nSrc1Step : int;
      nConstant : nppdefs_h.Npp32s;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:9290
   pragma Import (C, nppiAndC_32s_C1R, "nppiAndC_32s_C1R");

  --* 
  -- * One 32-bit signed integer channel in place image logical and with constant.
  -- * \param nConstant Constant.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAndC_32s_C1IR
     (nConstant : nppdefs_h.Npp32s;
      pSrcDst : access nppdefs_h.Npp32s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:9302
   pragma Import (C, nppiAndC_32s_C1IR, "nppiAndC_32s_C1IR");

  --* 
  -- * Three 32-bit signed integer channel image logical and with constant.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAndC_32s_C3R
     (pSrc1 : access nppdefs_h.Npp32s;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp32s;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:9315
   pragma Import (C, nppiAndC_32s_C3R, "nppiAndC_32s_C3R");

  --* 
  -- * Three 32-bit signed integer channel in place image logical and with constant.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAndC_32s_C3IR
     (aConstants : access nppdefs_h.Npp32s;
      pSrcDst : access nppdefs_h.Npp32s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:9327
   pragma Import (C, nppiAndC_32s_C3IR, "nppiAndC_32s_C3IR");

  --* 
  -- * Four 32-bit signed integer channel image logical and with constant with unmodified alpha.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAndC_32s_AC4R
     (pSrc1 : access nppdefs_h.Npp32s;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp32s;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:9340
   pragma Import (C, nppiAndC_32s_AC4R, "nppiAndC_32s_AC4R");

  --* 
  -- * Four 32-bit signed integer channel in place image logical and with constant with unmodified alpha.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAndC_32s_AC4IR
     (aConstants : access nppdefs_h.Npp32s;
      pSrcDst : access nppdefs_h.Npp32s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:9352
   pragma Import (C, nppiAndC_32s_AC4IR, "nppiAndC_32s_AC4IR");

  --* 
  -- * Four 32-bit signed integer channel image logical and with constant.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAndC_32s_C4R
     (pSrc1 : access nppdefs_h.Npp32s;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp32s;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:9365
   pragma Import (C, nppiAndC_32s_C4R, "nppiAndC_32s_C4R");

  --* 
  -- * Four 32-bit signed integer channel in place image logical and with constant.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAndC_32s_C4IR
     (aConstants : access nppdefs_h.Npp32s;
      pSrcDst : access nppdefs_h.Npp32s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:9377
   pragma Import (C, nppiAndC_32s_C4IR, "nppiAndC_32s_C4IR");

  --* @} image_andc  
  --* 
  -- * @defgroup image_orc OrC
  -- *
  -- * Pixel by pixel logical or of an image with a constant.
  -- *
  -- * @{
  --  

  --* 
  -- * One 8-bit unsigned char channel image logical or with constant.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param nConstant Constant.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiOrC_8u_C1R
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      nConstant : nppdefs_h.Npp8u;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:9401
   pragma Import (C, nppiOrC_8u_C1R, "nppiOrC_8u_C1R");

  --* 
  -- * One 8-bit unsigned char channel in place image logical or with constant.
  -- * \param nConstant Constant.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiOrC_8u_C1IR
     (nConstant : nppdefs_h.Npp8u;
      pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:9413
   pragma Import (C, nppiOrC_8u_C1IR, "nppiOrC_8u_C1IR");

  --* 
  -- * Three 8-bit unsigned char channel image logical or with constant.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiOrC_8u_C3R
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp8u;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:9426
   pragma Import (C, nppiOrC_8u_C3R, "nppiOrC_8u_C3R");

  --* 
  -- * Three 8-bit unsigned char channel in place image logical or with constant.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiOrC_8u_C3IR
     (aConstants : access nppdefs_h.Npp8u;
      pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:9438
   pragma Import (C, nppiOrC_8u_C3IR, "nppiOrC_8u_C3IR");

  --* 
  -- * Four 8-bit unsigned char channel image logical or with constant with unmodified alpha.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiOrC_8u_AC4R
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp8u;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:9451
   pragma Import (C, nppiOrC_8u_AC4R, "nppiOrC_8u_AC4R");

  --* 
  -- * Four 8-bit unsigned char channel in place image logical or with constant with unmodified alpha.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiOrC_8u_AC4IR
     (aConstants : access nppdefs_h.Npp8u;
      pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:9463
   pragma Import (C, nppiOrC_8u_AC4IR, "nppiOrC_8u_AC4IR");

  --* 
  -- * Four 8-bit unsigned char channel image logical or with constant.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiOrC_8u_C4R
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp8u;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:9476
   pragma Import (C, nppiOrC_8u_C4R, "nppiOrC_8u_C4R");

  --* 
  -- * Four 8-bit unsigned char channel in place image logical or with constant.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiOrC_8u_C4IR
     (aConstants : access nppdefs_h.Npp8u;
      pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:9488
   pragma Import (C, nppiOrC_8u_C4IR, "nppiOrC_8u_C4IR");

  --* 
  -- * One 16-bit unsigned short channel image logical or with constant.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param nConstant Constant.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiOrC_16u_C1R
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      nConstant : nppdefs_h.Npp16u;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:9501
   pragma Import (C, nppiOrC_16u_C1R, "nppiOrC_16u_C1R");

  --* 
  -- * One 16-bit unsigned short channel in place image logical or with constant.
  -- * \param nConstant Constant.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiOrC_16u_C1IR
     (nConstant : nppdefs_h.Npp16u;
      pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:9513
   pragma Import (C, nppiOrC_16u_C1IR, "nppiOrC_16u_C1IR");

  --* 
  -- * Three 16-bit unsigned short channel image logical or with constant.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiOrC_16u_C3R
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp16u;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:9526
   pragma Import (C, nppiOrC_16u_C3R, "nppiOrC_16u_C3R");

  --* 
  -- * Three 16-bit unsigned short channel in place image logical or with constant.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiOrC_16u_C3IR
     (aConstants : access nppdefs_h.Npp16u;
      pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:9538
   pragma Import (C, nppiOrC_16u_C3IR, "nppiOrC_16u_C3IR");

  --* 
  -- * Four 16-bit unsigned short channel image logical or with constant with unmodified alpha.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiOrC_16u_AC4R
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp16u;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:9551
   pragma Import (C, nppiOrC_16u_AC4R, "nppiOrC_16u_AC4R");

  --* 
  -- * Four 16-bit unsigned short channel in place image logical or with constant with unmodified alpha.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiOrC_16u_AC4IR
     (aConstants : access nppdefs_h.Npp16u;
      pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:9563
   pragma Import (C, nppiOrC_16u_AC4IR, "nppiOrC_16u_AC4IR");

  --* 
  -- * Four 16-bit unsigned short channel image logical or with constant.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiOrC_16u_C4R
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp16u;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:9576
   pragma Import (C, nppiOrC_16u_C4R, "nppiOrC_16u_C4R");

  --* 
  -- * Four 16-bit unsigned short channel in place image logical or with constant.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiOrC_16u_C4IR
     (aConstants : access nppdefs_h.Npp16u;
      pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:9588
   pragma Import (C, nppiOrC_16u_C4IR, "nppiOrC_16u_C4IR");

  --* 
  -- * One 32-bit signed integer channel image logical or with constant.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param nConstant Constant.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiOrC_32s_C1R
     (pSrc1 : access nppdefs_h.Npp32s;
      nSrc1Step : int;
      nConstant : nppdefs_h.Npp32s;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:9601
   pragma Import (C, nppiOrC_32s_C1R, "nppiOrC_32s_C1R");

  --* 
  -- * One 32-bit signed integer channel in place image logical or with constant.
  -- * \param nConstant Constant.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiOrC_32s_C1IR
     (nConstant : nppdefs_h.Npp32s;
      pSrcDst : access nppdefs_h.Npp32s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:9613
   pragma Import (C, nppiOrC_32s_C1IR, "nppiOrC_32s_C1IR");

  --* 
  -- * Three 32-bit signed integer channel image logical or with constant.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiOrC_32s_C3R
     (pSrc1 : access nppdefs_h.Npp32s;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp32s;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:9626
   pragma Import (C, nppiOrC_32s_C3R, "nppiOrC_32s_C3R");

  --* 
  -- * Three 32-bit signed integer channel in place image logical or with constant.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiOrC_32s_C3IR
     (aConstants : access nppdefs_h.Npp32s;
      pSrcDst : access nppdefs_h.Npp32s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:9638
   pragma Import (C, nppiOrC_32s_C3IR, "nppiOrC_32s_C3IR");

  --* 
  -- * Four 32-bit signed integer channel image logical or with constant with unmodified alpha.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiOrC_32s_AC4R
     (pSrc1 : access nppdefs_h.Npp32s;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp32s;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:9651
   pragma Import (C, nppiOrC_32s_AC4R, "nppiOrC_32s_AC4R");

  --* 
  -- * Four 32-bit signed integer channel in place image logical or with constant with unmodified alpha.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiOrC_32s_AC4IR
     (aConstants : access nppdefs_h.Npp32s;
      pSrcDst : access nppdefs_h.Npp32s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:9663
   pragma Import (C, nppiOrC_32s_AC4IR, "nppiOrC_32s_AC4IR");

  --* 
  -- * Four 32-bit signed integer channel image logical or with constant.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiOrC_32s_C4R
     (pSrc1 : access nppdefs_h.Npp32s;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp32s;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:9676
   pragma Import (C, nppiOrC_32s_C4R, "nppiOrC_32s_C4R");

  --* 
  -- * Four 32-bit signed integer channel in place image logical or with constant.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiOrC_32s_C4IR
     (aConstants : access nppdefs_h.Npp32s;
      pSrcDst : access nppdefs_h.Npp32s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:9688
   pragma Import (C, nppiOrC_32s_C4IR, "nppiOrC_32s_C4IR");

  --* @} image_orc  
  --* 
  -- * @defgroup image_xorc XorC
  -- *
  -- * Pixel by pixel logical exclusive or of an image with a constant.
  -- *
  -- * @{
  --  

  --* 
  -- * One 8-bit unsigned char channel image logical exclusive or with constant.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param nConstant Constant.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiXorC_8u_C1R
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      nConstant : nppdefs_h.Npp8u;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:9711
   pragma Import (C, nppiXorC_8u_C1R, "nppiXorC_8u_C1R");

  --* 
  -- * One 8-bit unsigned char channel in place image logical exclusive or with constant.
  -- * \param nConstant Constant.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiXorC_8u_C1IR
     (nConstant : nppdefs_h.Npp8u;
      pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:9723
   pragma Import (C, nppiXorC_8u_C1IR, "nppiXorC_8u_C1IR");

  --* 
  -- * Three 8-bit unsigned char channel image logical exclusive or with constant.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiXorC_8u_C3R
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp8u;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:9736
   pragma Import (C, nppiXorC_8u_C3R, "nppiXorC_8u_C3R");

  --* 
  -- * Three 8-bit unsigned char channel in place image logical exclusive or with constant.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiXorC_8u_C3IR
     (aConstants : access nppdefs_h.Npp8u;
      pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:9748
   pragma Import (C, nppiXorC_8u_C3IR, "nppiXorC_8u_C3IR");

  --* 
  -- * Four 8-bit unsigned char channel image logical exclusive or with constant with unmodified alpha.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiXorC_8u_AC4R
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp8u;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:9761
   pragma Import (C, nppiXorC_8u_AC4R, "nppiXorC_8u_AC4R");

  --* 
  -- * Four 8-bit unsigned char channel in place image logical exclusive or with constant with unmodified alpha.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiXorC_8u_AC4IR
     (aConstants : access nppdefs_h.Npp8u;
      pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:9773
   pragma Import (C, nppiXorC_8u_AC4IR, "nppiXorC_8u_AC4IR");

  --* 
  -- * Four 8-bit unsigned char channel image logical exclusive or with constant.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiXorC_8u_C4R
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp8u;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:9786
   pragma Import (C, nppiXorC_8u_C4R, "nppiXorC_8u_C4R");

  --* 
  -- * Four 8-bit unsigned char channel in place image logical exclusive or with constant.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiXorC_8u_C4IR
     (aConstants : access nppdefs_h.Npp8u;
      pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:9798
   pragma Import (C, nppiXorC_8u_C4IR, "nppiXorC_8u_C4IR");

  --* 
  -- * One 16-bit unsigned short channel image logical exclusive or with constant.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param nConstant Constant.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiXorC_16u_C1R
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      nConstant : nppdefs_h.Npp16u;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:9811
   pragma Import (C, nppiXorC_16u_C1R, "nppiXorC_16u_C1R");

  --* 
  -- * One 16-bit unsigned short channel in place image logical exclusive or with constant.
  -- * \param nConstant Constant.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiXorC_16u_C1IR
     (nConstant : nppdefs_h.Npp16u;
      pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:9823
   pragma Import (C, nppiXorC_16u_C1IR, "nppiXorC_16u_C1IR");

  --* 
  -- * Three 16-bit unsigned short channel image logical exclusive or with constant.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiXorC_16u_C3R
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp16u;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:9836
   pragma Import (C, nppiXorC_16u_C3R, "nppiXorC_16u_C3R");

  --* 
  -- * Three 16-bit unsigned short channel in place image logical exclusive or with constant.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiXorC_16u_C3IR
     (aConstants : access nppdefs_h.Npp16u;
      pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:9848
   pragma Import (C, nppiXorC_16u_C3IR, "nppiXorC_16u_C3IR");

  --* 
  -- * Four 16-bit unsigned short channel image logical exclusive or with constant with unmodified alpha.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiXorC_16u_AC4R
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp16u;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:9861
   pragma Import (C, nppiXorC_16u_AC4R, "nppiXorC_16u_AC4R");

  --* 
  -- * Four 16-bit unsigned short channel in place image logical exclusive or with constant with unmodified alpha.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiXorC_16u_AC4IR
     (aConstants : access nppdefs_h.Npp16u;
      pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:9873
   pragma Import (C, nppiXorC_16u_AC4IR, "nppiXorC_16u_AC4IR");

  --* 
  -- * Four 16-bit unsigned short channel image logical exclusive or with constant.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiXorC_16u_C4R
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp16u;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:9886
   pragma Import (C, nppiXorC_16u_C4R, "nppiXorC_16u_C4R");

  --* 
  -- * Four 16-bit unsigned short channel in place image logical exclusive or with constant.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiXorC_16u_C4IR
     (aConstants : access nppdefs_h.Npp16u;
      pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:9898
   pragma Import (C, nppiXorC_16u_C4IR, "nppiXorC_16u_C4IR");

  --* 
  -- * One 32-bit signed integer channel image logical exclusive or with constant.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param nConstant Constant.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiXorC_32s_C1R
     (pSrc1 : access nppdefs_h.Npp32s;
      nSrc1Step : int;
      nConstant : nppdefs_h.Npp32s;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:9911
   pragma Import (C, nppiXorC_32s_C1R, "nppiXorC_32s_C1R");

  --* 
  -- * One 32-bit signed integer channel in place image logical exclusive or with constant.
  -- * \param nConstant Constant.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiXorC_32s_C1IR
     (nConstant : nppdefs_h.Npp32s;
      pSrcDst : access nppdefs_h.Npp32s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:9923
   pragma Import (C, nppiXorC_32s_C1IR, "nppiXorC_32s_C1IR");

  --* 
  -- * Three 32-bit signed integer channel image logical exclusive or with constant.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiXorC_32s_C3R
     (pSrc1 : access nppdefs_h.Npp32s;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp32s;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:9936
   pragma Import (C, nppiXorC_32s_C3R, "nppiXorC_32s_C3R");

  --* 
  -- * Three 32-bit signed integer channel in place image logical exclusive or with constant.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiXorC_32s_C3IR
     (aConstants : access nppdefs_h.Npp32s;
      pSrcDst : access nppdefs_h.Npp32s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:9948
   pragma Import (C, nppiXorC_32s_C3IR, "nppiXorC_32s_C3IR");

  --* 
  -- * Four 32-bit signed integer channel image logical exclusive or with constant with unmodified alpha.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiXorC_32s_AC4R
     (pSrc1 : access nppdefs_h.Npp32s;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp32s;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:9961
   pragma Import (C, nppiXorC_32s_AC4R, "nppiXorC_32s_AC4R");

  --* 
  -- * Four 32-bit signed integer channel in place image logical exclusive or with constant with unmodified alpha.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiXorC_32s_AC4IR
     (aConstants : access nppdefs_h.Npp32s;
      pSrcDst : access nppdefs_h.Npp32s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:9973
   pragma Import (C, nppiXorC_32s_AC4IR, "nppiXorC_32s_AC4IR");

  --* 
  -- * Four 32-bit signed integer channel image logical exclusive or with constant.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiXorC_32s_C4R
     (pSrc1 : access nppdefs_h.Npp32s;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp32s;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:9986
   pragma Import (C, nppiXorC_32s_C4R, "nppiXorC_32s_C4R");

  --* 
  -- * Four 32-bit signed integer channel in place image logical exclusive or with constant.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiXorC_32s_C4IR
     (aConstants : access nppdefs_h.Npp32s;
      pSrcDst : access nppdefs_h.Npp32s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:9998
   pragma Import (C, nppiXorC_32s_C4IR, "nppiXorC_32s_C4IR");

  --* @} image_xorc  
  --* 
  -- * @defgroup image_rshiftc RShiftC
  -- *
  -- * Pixel by pixel right shift of an image by a constant value.
  -- *
  -- * @{
  --  

  --* 
  -- * One 8-bit unsigned char channel image right shift by constant.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param nConstant Constant.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiRShiftC_8u_C1R
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      nConstant : nppdefs_h.Npp32u;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:10021
   pragma Import (C, nppiRShiftC_8u_C1R, "nppiRShiftC_8u_C1R");

  --* 
  -- * One 8-bit unsigned char channel in place image right shift by constant.
  -- * \param nConstant Constant.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiRShiftC_8u_C1IR
     (nConstant : nppdefs_h.Npp32u;
      pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:10033
   pragma Import (C, nppiRShiftC_8u_C1IR, "nppiRShiftC_8u_C1IR");

  --* 
  -- * Three 8-bit unsigned char channel image right shift by constant.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiRShiftC_8u_C3R
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp32u;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:10046
   pragma Import (C, nppiRShiftC_8u_C3R, "nppiRShiftC_8u_C3R");

  --* 
  -- * Three 8-bit unsigned char channel in place image right shift by constant.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiRShiftC_8u_C3IR
     (aConstants : access nppdefs_h.Npp32u;
      pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:10058
   pragma Import (C, nppiRShiftC_8u_C3IR, "nppiRShiftC_8u_C3IR");

  --* 
  -- * Four 8-bit unsigned char channel image right shift by constant with unmodified alpha.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiRShiftC_8u_AC4R
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp32u;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:10071
   pragma Import (C, nppiRShiftC_8u_AC4R, "nppiRShiftC_8u_AC4R");

  --* 
  -- * Four 8-bit unsigned char channel in place image right shift by constant with unmodified alpha.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiRShiftC_8u_AC4IR
     (aConstants : access nppdefs_h.Npp32u;
      pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:10083
   pragma Import (C, nppiRShiftC_8u_AC4IR, "nppiRShiftC_8u_AC4IR");

  --* 
  -- * Four 8-bit unsigned char channel image right shift by constant.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiRShiftC_8u_C4R
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp32u;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:10096
   pragma Import (C, nppiRShiftC_8u_C4R, "nppiRShiftC_8u_C4R");

  --* 
  -- * Four 8-bit unsigned char channel in place image right shift by constant.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiRShiftC_8u_C4IR
     (aConstants : access nppdefs_h.Npp32u;
      pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:10108
   pragma Import (C, nppiRShiftC_8u_C4IR, "nppiRShiftC_8u_C4IR");

  --* 
  -- * One 8-bit signed char channel image right shift by constant.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param nConstant Constant.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiRShiftC_8s_C1R
     (pSrc1 : access nppdefs_h.Npp8s;
      nSrc1Step : int;
      nConstant : nppdefs_h.Npp32u;
      pDst : access nppdefs_h.Npp8s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:10121
   pragma Import (C, nppiRShiftC_8s_C1R, "nppiRShiftC_8s_C1R");

  --* 
  -- * One 8-bit signed char channel in place image right shift by constant.
  -- * \param nConstant Constant.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiRShiftC_8s_C1IR
     (nConstant : nppdefs_h.Npp32u;
      pSrcDst : access nppdefs_h.Npp8s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:10133
   pragma Import (C, nppiRShiftC_8s_C1IR, "nppiRShiftC_8s_C1IR");

  --* 
  -- * Three 8-bit signed char channel image right shift by constant.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiRShiftC_8s_C3R
     (pSrc1 : access nppdefs_h.Npp8s;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp32u;
      pDst : access nppdefs_h.Npp8s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:10146
   pragma Import (C, nppiRShiftC_8s_C3R, "nppiRShiftC_8s_C3R");

  --* 
  -- * Three 8-bit signed char channel in place image right shift by constant.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiRShiftC_8s_C3IR
     (aConstants : access nppdefs_h.Npp32u;
      pSrcDst : access nppdefs_h.Npp8s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:10158
   pragma Import (C, nppiRShiftC_8s_C3IR, "nppiRShiftC_8s_C3IR");

  --* 
  -- * Four 8-bit signed char channel image right shift by constant with unmodified alpha.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiRShiftC_8s_AC4R
     (pSrc1 : access nppdefs_h.Npp8s;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp32u;
      pDst : access nppdefs_h.Npp8s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:10171
   pragma Import (C, nppiRShiftC_8s_AC4R, "nppiRShiftC_8s_AC4R");

  --* 
  -- * Four 8-bit signed char channel in place image right shift by constant with unmodified alpha.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiRShiftC_8s_AC4IR
     (aConstants : access nppdefs_h.Npp32u;
      pSrcDst : access nppdefs_h.Npp8s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:10183
   pragma Import (C, nppiRShiftC_8s_AC4IR, "nppiRShiftC_8s_AC4IR");

  --* 
  -- * Four 8-bit signed char channel image right shift by constant.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiRShiftC_8s_C4R
     (pSrc1 : access nppdefs_h.Npp8s;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp32u;
      pDst : access nppdefs_h.Npp8s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:10196
   pragma Import (C, nppiRShiftC_8s_C4R, "nppiRShiftC_8s_C4R");

  --* 
  -- * Four 8-bit signed char channel in place image right shift by constant.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiRShiftC_8s_C4IR
     (aConstants : access nppdefs_h.Npp32u;
      pSrcDst : access nppdefs_h.Npp8s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:10208
   pragma Import (C, nppiRShiftC_8s_C4IR, "nppiRShiftC_8s_C4IR");

  --* 
  -- * One 16-bit unsigned short channel image right shift by constant.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param nConstant Constant.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiRShiftC_16u_C1R
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      nConstant : nppdefs_h.Npp32u;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:10221
   pragma Import (C, nppiRShiftC_16u_C1R, "nppiRShiftC_16u_C1R");

  --* 
  -- * One 16-bit unsigned short channel in place image right shift by constant.
  -- * \param nConstant Constant.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiRShiftC_16u_C1IR
     (nConstant : nppdefs_h.Npp32u;
      pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:10233
   pragma Import (C, nppiRShiftC_16u_C1IR, "nppiRShiftC_16u_C1IR");

  --* 
  -- * Three 16-bit unsigned short channel image right shift by constant.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiRShiftC_16u_C3R
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp32u;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:10246
   pragma Import (C, nppiRShiftC_16u_C3R, "nppiRShiftC_16u_C3R");

  --* 
  -- * Three 16-bit unsigned short channel in place image right shift by constant.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiRShiftC_16u_C3IR
     (aConstants : access nppdefs_h.Npp32u;
      pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:10258
   pragma Import (C, nppiRShiftC_16u_C3IR, "nppiRShiftC_16u_C3IR");

  --* 
  -- * Four 16-bit unsigned short channel image right shift by constant with unmodified alpha.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiRShiftC_16u_AC4R
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp32u;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:10271
   pragma Import (C, nppiRShiftC_16u_AC4R, "nppiRShiftC_16u_AC4R");

  --* 
  -- * Four 16-bit unsigned short channel in place image right shift by constant with unmodified alpha.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiRShiftC_16u_AC4IR
     (aConstants : access nppdefs_h.Npp32u;
      pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:10283
   pragma Import (C, nppiRShiftC_16u_AC4IR, "nppiRShiftC_16u_AC4IR");

  --* 
  -- * Four 16-bit unsigned short channel image right shift by constant.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiRShiftC_16u_C4R
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp32u;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:10296
   pragma Import (C, nppiRShiftC_16u_C4R, "nppiRShiftC_16u_C4R");

  --* 
  -- * Four 16-bit unsigned short channel in place image right shift by constant.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiRShiftC_16u_C4IR
     (aConstants : access nppdefs_h.Npp32u;
      pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:10308
   pragma Import (C, nppiRShiftC_16u_C4IR, "nppiRShiftC_16u_C4IR");

  --* 
  -- * One 16-bit signed short channel image right shift by constant.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param nConstant Constant.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiRShiftC_16s_C1R
     (pSrc1 : access nppdefs_h.Npp16s;
      nSrc1Step : int;
      nConstant : nppdefs_h.Npp32u;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:10321
   pragma Import (C, nppiRShiftC_16s_C1R, "nppiRShiftC_16s_C1R");

  --* 
  -- * One 16-bit signed short channel in place image right shift by constant.
  -- * \param nConstant Constant.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiRShiftC_16s_C1IR
     (nConstant : nppdefs_h.Npp32u;
      pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:10333
   pragma Import (C, nppiRShiftC_16s_C1IR, "nppiRShiftC_16s_C1IR");

  --* 
  -- * Three 16-bit signed short channel image right shift by constant.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiRShiftC_16s_C3R
     (pSrc1 : access nppdefs_h.Npp16s;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp32u;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:10346
   pragma Import (C, nppiRShiftC_16s_C3R, "nppiRShiftC_16s_C3R");

  --* 
  -- * Three 16-bit signed short channel in place image right shift by constant.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiRShiftC_16s_C3IR
     (aConstants : access nppdefs_h.Npp32u;
      pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:10358
   pragma Import (C, nppiRShiftC_16s_C3IR, "nppiRShiftC_16s_C3IR");

  --* 
  -- * Four 16-bit signed short channel image right shift by constant with unmodified alpha.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiRShiftC_16s_AC4R
     (pSrc1 : access nppdefs_h.Npp16s;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp32u;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:10371
   pragma Import (C, nppiRShiftC_16s_AC4R, "nppiRShiftC_16s_AC4R");

  --* 
  -- * Four 16-bit signed short channel in place image right shift by constant with unmodified alpha.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiRShiftC_16s_AC4IR
     (aConstants : access nppdefs_h.Npp32u;
      pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:10383
   pragma Import (C, nppiRShiftC_16s_AC4IR, "nppiRShiftC_16s_AC4IR");

  --* 
  -- * Four 16-bit signed short channel image right shift by constant.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiRShiftC_16s_C4R
     (pSrc1 : access nppdefs_h.Npp16s;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp32u;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:10396
   pragma Import (C, nppiRShiftC_16s_C4R, "nppiRShiftC_16s_C4R");

  --* 
  -- * Four 16-bit signed short channel in place image right shift by constant.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiRShiftC_16s_C4IR
     (aConstants : access nppdefs_h.Npp32u;
      pSrcDst : access nppdefs_h.Npp16s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:10408
   pragma Import (C, nppiRShiftC_16s_C4IR, "nppiRShiftC_16s_C4IR");

  --* 
  -- * One 32-bit signed integer channel image right shift by constant.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param nConstant Constant.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiRShiftC_32s_C1R
     (pSrc1 : access nppdefs_h.Npp32s;
      nSrc1Step : int;
      nConstant : nppdefs_h.Npp32u;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:10421
   pragma Import (C, nppiRShiftC_32s_C1R, "nppiRShiftC_32s_C1R");

  --* 
  -- * One 32-bit signed integer channel in place image right shift by constant.
  -- * \param nConstant Constant.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiRShiftC_32s_C1IR
     (nConstant : nppdefs_h.Npp32u;
      pSrcDst : access nppdefs_h.Npp32s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:10433
   pragma Import (C, nppiRShiftC_32s_C1IR, "nppiRShiftC_32s_C1IR");

  --* 
  -- * Three 32-bit signed integer channel image right shift by constant.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiRShiftC_32s_C3R
     (pSrc1 : access nppdefs_h.Npp32s;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp32u;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:10446
   pragma Import (C, nppiRShiftC_32s_C3R, "nppiRShiftC_32s_C3R");

  --* 
  -- * Three 32-bit signed integer channel in place image right shift by constant.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiRShiftC_32s_C3IR
     (aConstants : access nppdefs_h.Npp32u;
      pSrcDst : access nppdefs_h.Npp32s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:10458
   pragma Import (C, nppiRShiftC_32s_C3IR, "nppiRShiftC_32s_C3IR");

  --* 
  -- * Four 32-bit signed integer channel image right shift by constant with unmodified alpha.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiRShiftC_32s_AC4R
     (pSrc1 : access nppdefs_h.Npp32s;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp32u;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:10471
   pragma Import (C, nppiRShiftC_32s_AC4R, "nppiRShiftC_32s_AC4R");

  --* 
  -- * Four 32-bit signed integer channel in place image right shift by constant with unmodified alpha.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiRShiftC_32s_AC4IR
     (aConstants : access nppdefs_h.Npp32u;
      pSrcDst : access nppdefs_h.Npp32s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:10483
   pragma Import (C, nppiRShiftC_32s_AC4IR, "nppiRShiftC_32s_AC4IR");

  --* 
  -- * Four 32-bit signed integer channel image right shift by constant.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiRShiftC_32s_C4R
     (pSrc1 : access nppdefs_h.Npp32s;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp32u;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:10496
   pragma Import (C, nppiRShiftC_32s_C4R, "nppiRShiftC_32s_C4R");

  --* 
  -- * Four 32-bit signed integer channel in place image right shift by constant.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiRShiftC_32s_C4IR
     (aConstants : access nppdefs_h.Npp32u;
      pSrcDst : access nppdefs_h.Npp32s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:10508
   pragma Import (C, nppiRShiftC_32s_C4IR, "nppiRShiftC_32s_C4IR");

  --* @} image_rshiftc  
  --* 
  -- * @defgroup image_lshiftc LShiftC
  -- *
  -- * Pixel by pixel left shift of an image by a constant value.
  -- *
  -- * @{
  --  

  --* 
  -- * One 8-bit unsigned char channel image left shift by constant.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param nConstant Constant.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiLShiftC_8u_C1R
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      nConstant : nppdefs_h.Npp32u;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:10531
   pragma Import (C, nppiLShiftC_8u_C1R, "nppiLShiftC_8u_C1R");

  --* 
  -- * One 8-bit unsigned char channel in place image left shift by constant.
  -- * \param nConstant Constant.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiLShiftC_8u_C1IR
     (nConstant : nppdefs_h.Npp32u;
      pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:10543
   pragma Import (C, nppiLShiftC_8u_C1IR, "nppiLShiftC_8u_C1IR");

  --* 
  -- * Three 8-bit unsigned char channel image left shift by constant.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiLShiftC_8u_C3R
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp32u;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:10556
   pragma Import (C, nppiLShiftC_8u_C3R, "nppiLShiftC_8u_C3R");

  --* 
  -- * Three 8-bit unsigned char channel in place image left shift by constant.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiLShiftC_8u_C3IR
     (aConstants : access nppdefs_h.Npp32u;
      pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:10568
   pragma Import (C, nppiLShiftC_8u_C3IR, "nppiLShiftC_8u_C3IR");

  --* 
  -- * Four 8-bit unsigned char channel image left shift by constant with unmodified alpha.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiLShiftC_8u_AC4R
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp32u;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:10581
   pragma Import (C, nppiLShiftC_8u_AC4R, "nppiLShiftC_8u_AC4R");

  --* 
  -- * Four 8-bit unsigned char channel in place image left shift by constant with unmodified alpha.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiLShiftC_8u_AC4IR
     (aConstants : access nppdefs_h.Npp32u;
      pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:10593
   pragma Import (C, nppiLShiftC_8u_AC4IR, "nppiLShiftC_8u_AC4IR");

  --* 
  -- * Four 8-bit unsigned char channel image left shift by constant.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiLShiftC_8u_C4R
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp32u;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:10606
   pragma Import (C, nppiLShiftC_8u_C4R, "nppiLShiftC_8u_C4R");

  --* 
  -- * Four 8-bit unsigned char channel in place image left shift by constant.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiLShiftC_8u_C4IR
     (aConstants : access nppdefs_h.Npp32u;
      pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:10618
   pragma Import (C, nppiLShiftC_8u_C4IR, "nppiLShiftC_8u_C4IR");

  --* 
  -- * One 16-bit unsigned short channel image left shift by constant.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param nConstant Constant
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiLShiftC_16u_C1R
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      nConstant : nppdefs_h.Npp32u;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:10631
   pragma Import (C, nppiLShiftC_16u_C1R, "nppiLShiftC_16u_C1R");

  --* 
  -- * One 16-bit unsigned short channel in place image left shift by constant.
  -- * \param nConstant Constant.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiLShiftC_16u_C1IR
     (nConstant : nppdefs_h.Npp32u;
      pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:10643
   pragma Import (C, nppiLShiftC_16u_C1IR, "nppiLShiftC_16u_C1IR");

  --* 
  -- * Three 16-bit unsigned short channel image left shift by constant.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiLShiftC_16u_C3R
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp32u;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:10656
   pragma Import (C, nppiLShiftC_16u_C3R, "nppiLShiftC_16u_C3R");

  --* 
  -- * Three 16-bit unsigned short channel in place image left shift by constant.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiLShiftC_16u_C3IR
     (aConstants : access nppdefs_h.Npp32u;
      pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:10668
   pragma Import (C, nppiLShiftC_16u_C3IR, "nppiLShiftC_16u_C3IR");

  --* 
  -- * Four 16-bit unsigned short channel image left shift by constant with unmodified alpha.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiLShiftC_16u_AC4R
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp32u;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:10681
   pragma Import (C, nppiLShiftC_16u_AC4R, "nppiLShiftC_16u_AC4R");

  --* 
  -- * Four 16-bit unsigned short channel in place image left shift by constant with unmodified alpha.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiLShiftC_16u_AC4IR
     (aConstants : access nppdefs_h.Npp32u;
      pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:10693
   pragma Import (C, nppiLShiftC_16u_AC4IR, "nppiLShiftC_16u_AC4IR");

  --* 
  -- * Four 16-bit unsigned short channel image left shift by constant.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiLShiftC_16u_C4R
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp32u;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:10706
   pragma Import (C, nppiLShiftC_16u_C4R, "nppiLShiftC_16u_C4R");

  --* 
  -- * Four 16-bit unsigned short channel in place image left shift by constant.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiLShiftC_16u_C4IR
     (aConstants : access nppdefs_h.Npp32u;
      pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:10718
   pragma Import (C, nppiLShiftC_16u_C4IR, "nppiLShiftC_16u_C4IR");

  --* 
  -- * One 32-bit signed integer channel image left shift by constant.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param nConstant Constant.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiLShiftC_32s_C1R
     (pSrc1 : access nppdefs_h.Npp32s;
      nSrc1Step : int;
      nConstant : nppdefs_h.Npp32u;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:10731
   pragma Import (C, nppiLShiftC_32s_C1R, "nppiLShiftC_32s_C1R");

  --* 
  -- * One 32-bit signed integer channel in place image left shift by constant.
  -- * \param nConstant Constant.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiLShiftC_32s_C1IR
     (nConstant : nppdefs_h.Npp32u;
      pSrcDst : access nppdefs_h.Npp32s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:10743
   pragma Import (C, nppiLShiftC_32s_C1IR, "nppiLShiftC_32s_C1IR");

  --* 
  -- * Three 32-bit signed integer channel image left shift by constant.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiLShiftC_32s_C3R
     (pSrc1 : access nppdefs_h.Npp32s;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp32u;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:10756
   pragma Import (C, nppiLShiftC_32s_C3R, "nppiLShiftC_32s_C3R");

  --* 
  -- * Three 32-bit signed integer channel in place image left shift by constant.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiLShiftC_32s_C3IR
     (aConstants : access nppdefs_h.Npp32u;
      pSrcDst : access nppdefs_h.Npp32s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:10768
   pragma Import (C, nppiLShiftC_32s_C3IR, "nppiLShiftC_32s_C3IR");

  --* 
  -- * Four 32-bit signed integer channel image left shift by constant with unmodified alpha.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiLShiftC_32s_AC4R
     (pSrc1 : access nppdefs_h.Npp32s;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp32u;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:10781
   pragma Import (C, nppiLShiftC_32s_AC4R, "nppiLShiftC_32s_AC4R");

  --* 
  -- * Four 32-bit signed integer channel in place image left shift by constant with unmodified alpha.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiLShiftC_32s_AC4IR
     (aConstants : access nppdefs_h.Npp32u;
      pSrcDst : access nppdefs_h.Npp32s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:10793
   pragma Import (C, nppiLShiftC_32s_AC4IR, "nppiLShiftC_32s_AC4IR");

  --* 
  -- * Four 32-bit signed integer channel image left shift by constant.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiLShiftC_32s_C4R
     (pSrc1 : access nppdefs_h.Npp32s;
      nSrc1Step : int;
      aConstants : access nppdefs_h.Npp32u;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:10806
   pragma Import (C, nppiLShiftC_32s_C4R, "nppiLShiftC_32s_C4R");

  --* 
  -- * Four 32-bit signed integer channel in place image left shift by constant.
  -- * \param aConstants fixed size array of constant values, one per channel.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiLShiftC_32s_C4IR
     (aConstants : access nppdefs_h.Npp32u;
      pSrcDst : access nppdefs_h.Npp32s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:10818
   pragma Import (C, nppiLShiftC_32s_C4IR, "nppiLShiftC_32s_C4IR");

  --* @} image_lshiftc  
  --* 
  -- * @defgroup image_and And
  -- *
  -- * Pixel by pixel logical and of images.
  -- *
  -- * @{
  --  

  --* 
  -- * One 8-bit unsigned char channel image logical and.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAnd_8u_C1R
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp8u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:10842
   pragma Import (C, nppiAnd_8u_C1R, "nppiAnd_8u_C1R");

  --* 
  -- * One 8-bit unsigned char channel in place image logical and.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAnd_8u_C1IR
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:10854
   pragma Import (C, nppiAnd_8u_C1IR, "nppiAnd_8u_C1IR");

  --* 
  -- * Three 8-bit unsigned char channel image logical and.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAnd_8u_C3R
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp8u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:10869
   pragma Import (C, nppiAnd_8u_C3R, "nppiAnd_8u_C3R");

  --* 
  -- * Three 8-bit unsigned char channel in place image logical and.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAnd_8u_C3IR
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:10881
   pragma Import (C, nppiAnd_8u_C3IR, "nppiAnd_8u_C3IR");

  --* 
  -- * Four 8-bit unsigned char channel image logical and with unmodified alpha.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAnd_8u_AC4R
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp8u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:10896
   pragma Import (C, nppiAnd_8u_AC4R, "nppiAnd_8u_AC4R");

  --* 
  -- * Four 8-bit unsigned char channel in place image logical and with unmodified alpha.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAnd_8u_AC4IR
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:10908
   pragma Import (C, nppiAnd_8u_AC4IR, "nppiAnd_8u_AC4IR");

  --* 
  -- * Four 8-bit unsigned char channel image logical and.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAnd_8u_C4R
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp8u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:10923
   pragma Import (C, nppiAnd_8u_C4R, "nppiAnd_8u_C4R");

  --* 
  -- * Four 8-bit unsigned char channel in place image logical and.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAnd_8u_C4IR
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:10935
   pragma Import (C, nppiAnd_8u_C4IR, "nppiAnd_8u_C4IR");

  --* 
  -- * One 16-bit unsigned short channel image logical and.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAnd_16u_C1R
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp16u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:10950
   pragma Import (C, nppiAnd_16u_C1R, "nppiAnd_16u_C1R");

  --* 
  -- * One 16-bit unsigned short channel in place image logical and.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAnd_16u_C1IR
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:10962
   pragma Import (C, nppiAnd_16u_C1IR, "nppiAnd_16u_C1IR");

  --* 
  -- * Three 16-bit unsigned short channel image logical and.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAnd_16u_C3R
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp16u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:10977
   pragma Import (C, nppiAnd_16u_C3R, "nppiAnd_16u_C3R");

  --* 
  -- * Three 16-bit unsigned short channel in place image logical and.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAnd_16u_C3IR
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:10989
   pragma Import (C, nppiAnd_16u_C3IR, "nppiAnd_16u_C3IR");

  --* 
  -- * Four 16-bit unsigned short channel image logical and with unmodified alpha.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAnd_16u_AC4R
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp16u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:11004
   pragma Import (C, nppiAnd_16u_AC4R, "nppiAnd_16u_AC4R");

  --* 
  -- * Four 16-bit unsigned short channel in place image logical and with unmodified alpha.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAnd_16u_AC4IR
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:11016
   pragma Import (C, nppiAnd_16u_AC4IR, "nppiAnd_16u_AC4IR");

  --* 
  -- * Four 16-bit unsigned short channel image logical and.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAnd_16u_C4R
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp16u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:11031
   pragma Import (C, nppiAnd_16u_C4R, "nppiAnd_16u_C4R");

  --* 
  -- * Four 16-bit unsigned short channel in place image logical and.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAnd_16u_C4IR
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:11043
   pragma Import (C, nppiAnd_16u_C4IR, "nppiAnd_16u_C4IR");

  --* 
  -- * One 32-bit signed integer channel image logical and.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAnd_32s_C1R
     (pSrc1 : access nppdefs_h.Npp32s;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp32s;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:11058
   pragma Import (C, nppiAnd_32s_C1R, "nppiAnd_32s_C1R");

  --* 
  -- * One 32-bit signed integer channel in place image logical and.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAnd_32s_C1IR
     (pSrc : access nppdefs_h.Npp32s;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp32s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:11070
   pragma Import (C, nppiAnd_32s_C1IR, "nppiAnd_32s_C1IR");

  --* 
  -- * Three 32-bit signed integer channel image logical and.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAnd_32s_C3R
     (pSrc1 : access nppdefs_h.Npp32s;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp32s;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:11085
   pragma Import (C, nppiAnd_32s_C3R, "nppiAnd_32s_C3R");

  --* 
  -- * Three 32-bit signed integer channel in place image logical and.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAnd_32s_C3IR
     (pSrc : access nppdefs_h.Npp32s;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp32s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:11097
   pragma Import (C, nppiAnd_32s_C3IR, "nppiAnd_32s_C3IR");

  --* 
  -- * Four 32-bit signed integer channel image logical and with unmodified alpha.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAnd_32s_AC4R
     (pSrc1 : access nppdefs_h.Npp32s;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp32s;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:11112
   pragma Import (C, nppiAnd_32s_AC4R, "nppiAnd_32s_AC4R");

  --* 
  -- * Four 32-bit signed integer channel in place image logical and with unmodified alpha.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAnd_32s_AC4IR
     (pSrc : access nppdefs_h.Npp32s;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp32s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:11124
   pragma Import (C, nppiAnd_32s_AC4IR, "nppiAnd_32s_AC4IR");

  --* 
  -- * Four 32-bit signed integer channel image logical and.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAnd_32s_C4R
     (pSrc1 : access nppdefs_h.Npp32s;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp32s;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:11139
   pragma Import (C, nppiAnd_32s_C4R, "nppiAnd_32s_C4R");

  --* 
  -- * Four 32-bit signed integer channel in place image logical and.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAnd_32s_C4IR
     (pSrc : access nppdefs_h.Npp32s;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp32s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:11151
   pragma Import (C, nppiAnd_32s_C4IR, "nppiAnd_32s_C4IR");

  --* @} image_and  
  --* 
  -- * @defgroup image_or Or
  -- *
  -- * Pixel by pixel logical or of images.
  -- *
  -- * @{
  --  

  --* 
  -- * One 8-bit unsigned char channel image logical or.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiOr_8u_C1R
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp8u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:11176
   pragma Import (C, nppiOr_8u_C1R, "nppiOr_8u_C1R");

  --* 
  -- * One 8-bit unsigned char channel in place image logical or.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiOr_8u_C1IR
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:11188
   pragma Import (C, nppiOr_8u_C1IR, "nppiOr_8u_C1IR");

  --* 
  -- * Three 8-bit unsigned char channel image logical or.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiOr_8u_C3R
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp8u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:11203
   pragma Import (C, nppiOr_8u_C3R, "nppiOr_8u_C3R");

  --* 
  -- * Three 8-bit unsigned char channel in place image logical or.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiOr_8u_C3IR
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:11215
   pragma Import (C, nppiOr_8u_C3IR, "nppiOr_8u_C3IR");

  --* 
  -- * Four 8-bit unsigned char channel image logical or with unmodified alpha.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiOr_8u_AC4R
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp8u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:11230
   pragma Import (C, nppiOr_8u_AC4R, "nppiOr_8u_AC4R");

  --* 
  -- * Four 8-bit unsigned char channel in place image logical or with unmodified alpha.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiOr_8u_AC4IR
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:11242
   pragma Import (C, nppiOr_8u_AC4IR, "nppiOr_8u_AC4IR");

  --* 
  -- * Four 8-bit unsigned char channel image logical or.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiOr_8u_C4R
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp8u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:11257
   pragma Import (C, nppiOr_8u_C4R, "nppiOr_8u_C4R");

  --* 
  -- * Four 8-bit unsigned char channel in place image logical or.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiOr_8u_C4IR
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:11269
   pragma Import (C, nppiOr_8u_C4IR, "nppiOr_8u_C4IR");

  --* 
  -- * One 16-bit unsigned short channel image logical or.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiOr_16u_C1R
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp16u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:11284
   pragma Import (C, nppiOr_16u_C1R, "nppiOr_16u_C1R");

  --* 
  -- * One 16-bit unsigned short channel in place image logical or.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiOr_16u_C1IR
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:11296
   pragma Import (C, nppiOr_16u_C1IR, "nppiOr_16u_C1IR");

  --* 
  -- * Three 16-bit unsigned short channel image logical or.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiOr_16u_C3R
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp16u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:11311
   pragma Import (C, nppiOr_16u_C3R, "nppiOr_16u_C3R");

  --* 
  -- * Three 16-bit unsigned short channel in place image logical or.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiOr_16u_C3IR
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:11323
   pragma Import (C, nppiOr_16u_C3IR, "nppiOr_16u_C3IR");

  --* 
  -- * Four 16-bit unsigned short channel image logical or with unmodified alpha.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiOr_16u_AC4R
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp16u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:11338
   pragma Import (C, nppiOr_16u_AC4R, "nppiOr_16u_AC4R");

  --* 
  -- * Four 16-bit unsigned short channel in place image logical or with unmodified alpha.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiOr_16u_AC4IR
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:11350
   pragma Import (C, nppiOr_16u_AC4IR, "nppiOr_16u_AC4IR");

  --* 
  -- * Four 16-bit unsigned short channel image logical or.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiOr_16u_C4R
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp16u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:11365
   pragma Import (C, nppiOr_16u_C4R, "nppiOr_16u_C4R");

  --* 
  -- * Four 16-bit unsigned short channel in place image logical or.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiOr_16u_C4IR
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:11377
   pragma Import (C, nppiOr_16u_C4IR, "nppiOr_16u_C4IR");

  --* 
  -- * One 32-bit signed integer channel image logical or.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiOr_32s_C1R
     (pSrc1 : access nppdefs_h.Npp32s;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp32s;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:11392
   pragma Import (C, nppiOr_32s_C1R, "nppiOr_32s_C1R");

  --* 
  -- * One 32-bit signed integer channel in place image logical or.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiOr_32s_C1IR
     (pSrc : access nppdefs_h.Npp32s;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp32s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:11404
   pragma Import (C, nppiOr_32s_C1IR, "nppiOr_32s_C1IR");

  --* 
  -- * Three 32-bit signed integer channel image logical or.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiOr_32s_C3R
     (pSrc1 : access nppdefs_h.Npp32s;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp32s;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:11419
   pragma Import (C, nppiOr_32s_C3R, "nppiOr_32s_C3R");

  --* 
  -- * Three 32-bit signed integer channel in place image logical or.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiOr_32s_C3IR
     (pSrc : access nppdefs_h.Npp32s;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp32s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:11431
   pragma Import (C, nppiOr_32s_C3IR, "nppiOr_32s_C3IR");

  --* 
  -- * Four 32-bit signed integer channel image logical or with unmodified alpha.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiOr_32s_AC4R
     (pSrc1 : access nppdefs_h.Npp32s;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp32s;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:11446
   pragma Import (C, nppiOr_32s_AC4R, "nppiOr_32s_AC4R");

  --* 
  -- * Four 32-bit signed integer channel in place image logical or with unmodified alpha.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiOr_32s_AC4IR
     (pSrc : access nppdefs_h.Npp32s;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp32s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:11458
   pragma Import (C, nppiOr_32s_AC4IR, "nppiOr_32s_AC4IR");

  --* 
  -- * Four 32-bit signed integer channel image logical or.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiOr_32s_C4R
     (pSrc1 : access nppdefs_h.Npp32s;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp32s;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:11473
   pragma Import (C, nppiOr_32s_C4R, "nppiOr_32s_C4R");

  --* 
  -- * Four 32-bit signed integer channel in place image logical or.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiOr_32s_C4IR
     (pSrc : access nppdefs_h.Npp32s;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp32s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:11485
   pragma Import (C, nppiOr_32s_C4IR, "nppiOr_32s_C4IR");

  --* @} image_or  
  --* 
  -- * @defgroup image_xor Xor
  -- *
  -- * Pixel by pixel logical exclusive or of images.
  -- *
  -- * @{
  --  

  --* 
  -- * One 8-bit unsigned char channel image logical exclusive or.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiXor_8u_C1R
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp8u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:11510
   pragma Import (C, nppiXor_8u_C1R, "nppiXor_8u_C1R");

  --* 
  -- * One 8-bit unsigned char channel in place image logical exclusive or.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiXor_8u_C1IR
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:11522
   pragma Import (C, nppiXor_8u_C1IR, "nppiXor_8u_C1IR");

  --* 
  -- * Three 8-bit unsigned char channel image logical exclusive or.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiXor_8u_C3R
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp8u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:11537
   pragma Import (C, nppiXor_8u_C3R, "nppiXor_8u_C3R");

  --* 
  -- * Three 8-bit unsigned char channel in place image logical exclusive or.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiXor_8u_C3IR
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:11549
   pragma Import (C, nppiXor_8u_C3IR, "nppiXor_8u_C3IR");

  --* 
  -- * Four 8-bit unsigned char channel image logical exclusive or with unmodified alpha.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiXor_8u_AC4R
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp8u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:11564
   pragma Import (C, nppiXor_8u_AC4R, "nppiXor_8u_AC4R");

  --* 
  -- * Four 8-bit unsigned char channel in place image logical exclusive or with unmodified alpha.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiXor_8u_AC4IR
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:11576
   pragma Import (C, nppiXor_8u_AC4IR, "nppiXor_8u_AC4IR");

  --* 
  -- * Four 8-bit unsigned char channel image logical exclusive or.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiXor_8u_C4R
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp8u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:11591
   pragma Import (C, nppiXor_8u_C4R, "nppiXor_8u_C4R");

  --* 
  -- * Four 8-bit unsigned char channel in place image logical exclusive or.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiXor_8u_C4IR
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:11603
   pragma Import (C, nppiXor_8u_C4IR, "nppiXor_8u_C4IR");

  --* 
  -- * One 16-bit unsigned short channel image logical exclusive or.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiXor_16u_C1R
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp16u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:11618
   pragma Import (C, nppiXor_16u_C1R, "nppiXor_16u_C1R");

  --* 
  -- * One 16-bit unsigned short channel in place image logical exclusive or.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiXor_16u_C1IR
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:11630
   pragma Import (C, nppiXor_16u_C1IR, "nppiXor_16u_C1IR");

  --* 
  -- * Three 16-bit unsigned short channel image logical exclusive or.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiXor_16u_C3R
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp16u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:11645
   pragma Import (C, nppiXor_16u_C3R, "nppiXor_16u_C3R");

  --* 
  -- * Three 16-bit unsigned short channel in place image logical exclusive or.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiXor_16u_C3IR
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:11657
   pragma Import (C, nppiXor_16u_C3IR, "nppiXor_16u_C3IR");

  --* 
  -- * Four 16-bit unsigned short channel image logical exclusive or with unmodified alpha.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiXor_16u_AC4R
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp16u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:11672
   pragma Import (C, nppiXor_16u_AC4R, "nppiXor_16u_AC4R");

  --* 
  -- * Four 16-bit unsigned short channel in place image logical exclusive or with unmodified alpha.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiXor_16u_AC4IR
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:11684
   pragma Import (C, nppiXor_16u_AC4IR, "nppiXor_16u_AC4IR");

  --* 
  -- * Four 16-bit unsigned short channel image logical exclusive or.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiXor_16u_C4R
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp16u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:11699
   pragma Import (C, nppiXor_16u_C4R, "nppiXor_16u_C4R");

  --* 
  -- * Four 16-bit unsigned short channel in place image logical exclusive or.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiXor_16u_C4IR
     (pSrc : access nppdefs_h.Npp16u;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:11711
   pragma Import (C, nppiXor_16u_C4IR, "nppiXor_16u_C4IR");

  --* 
  -- * One 32-bit signed integer channel image logical exclusive or.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiXor_32s_C1R
     (pSrc1 : access nppdefs_h.Npp32s;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp32s;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:11726
   pragma Import (C, nppiXor_32s_C1R, "nppiXor_32s_C1R");

  --* 
  -- * One 32-bit signed integer channel in place image logical exclusive or.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiXor_32s_C1IR
     (pSrc : access nppdefs_h.Npp32s;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp32s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:11738
   pragma Import (C, nppiXor_32s_C1IR, "nppiXor_32s_C1IR");

  --* 
  -- * Three 32-bit signed integer channel image logical exclusive or.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiXor_32s_C3R
     (pSrc1 : access nppdefs_h.Npp32s;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp32s;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:11753
   pragma Import (C, nppiXor_32s_C3R, "nppiXor_32s_C3R");

  --* 
  -- * Three 32-bit signed integer channel in place image logical exclusive or.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiXor_32s_C3IR
     (pSrc : access nppdefs_h.Npp32s;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp32s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:11765
   pragma Import (C, nppiXor_32s_C3IR, "nppiXor_32s_C3IR");

  --* 
  -- * Four 32-bit signed integer channel image logical exclusive or with unmodified alpha.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiXor_32s_AC4R
     (pSrc1 : access nppdefs_h.Npp32s;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp32s;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:11780
   pragma Import (C, nppiXor_32s_AC4R, "nppiXor_32s_AC4R");

  --* 
  -- * Four 32-bit signed integer channel in place image logical exclusive or with unmodified alpha.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiXor_32s_AC4IR
     (pSrc : access nppdefs_h.Npp32s;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp32s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:11792
   pragma Import (C, nppiXor_32s_AC4IR, "nppiXor_32s_AC4IR");

  --* 
  -- * Four 32-bit signed integer channel image logical exclusive or.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiXor_32s_C4R
     (pSrc1 : access nppdefs_h.Npp32s;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp32s;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:11807
   pragma Import (C, nppiXor_32s_C4R, "nppiXor_32s_C4R");

  --* 
  -- * Four 32-bit signed integer channel in place image logical exclusive or.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiXor_32s_C4IR
     (pSrc : access nppdefs_h.Npp32s;
      nSrcStep : int;
      pSrcDst : access nppdefs_h.Npp32s;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:11819
   pragma Import (C, nppiXor_32s_C4IR, "nppiXor_32s_C4IR");

  --* @} image_xor  
  --* 
  -- * @defgroup image_not Not
  -- *
  -- * Pixel by pixel logical not of image.
  -- *
  -- * @{
  --  

  --* 
  -- * One 8-bit unsigned char channel image logical not.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiNot_8u_C1R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:11842
   pragma Import (C, nppiNot_8u_C1R, "nppiNot_8u_C1R");

  --* 
  -- * One 8-bit unsigned char channel in place image logical not.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiNot_8u_C1IR
     (pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:11852
   pragma Import (C, nppiNot_8u_C1IR, "nppiNot_8u_C1IR");

  --* 
  -- * Three 8-bit unsigned char channel image logical not.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiNot_8u_C3R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:11864
   pragma Import (C, nppiNot_8u_C3R, "nppiNot_8u_C3R");

  --* 
  -- * Three 8-bit unsigned char channel in place image logical not.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiNot_8u_C3IR
     (pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:11874
   pragma Import (C, nppiNot_8u_C3IR, "nppiNot_8u_C3IR");

  --* 
  -- * Four 8-bit unsigned char channel image logical not with unmodified alpha.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiNot_8u_AC4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:11886
   pragma Import (C, nppiNot_8u_AC4R, "nppiNot_8u_AC4R");

  --* 
  -- * Four 8-bit unsigned char channel in place image logical not with unmodified alpha.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiNot_8u_AC4IR
     (pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:11896
   pragma Import (C, nppiNot_8u_AC4IR, "nppiNot_8u_AC4IR");

  --* 
  -- * Four 8-bit unsigned char channel image logical not.
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiNot_8u_C4R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:11908
   pragma Import (C, nppiNot_8u_C4R, "nppiNot_8u_C4R");

  --* 
  -- * Four 8-bit unsigned char channel in place image logical not.
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiNot_8u_C4IR
     (pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:11918
   pragma Import (C, nppiNot_8u_C4IR, "nppiNot_8u_C4IR");

  --* @} image_not  
  --* @} image_logical_operations  
  --* 
  -- * @defgroup image_alpha_composition_operations Alpha Composition
  -- * @{
  --  

  --* 
  -- * @defgroup image_alphacompc AlphaCompC
  -- *
  -- * Composite two images using constant alpha values.
  -- *
  -- * @{
  --  

  --* 
  -- * One 8-bit unsigned char channel image composition using constant alpha.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param nAlpha1 Image alpha opacity (0 - max channel pixel value).
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param nAlpha2 Image alpha opacity (0 - max channel pixel value).
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eAlphaOp alpha-blending operation..
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAlphaCompC_8u_C1R
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      nAlpha1 : nppdefs_h.Npp8u;
      pSrc2 : access nppdefs_h.Npp8u;
      nSrc2Step : int;
      nAlpha2 : nppdefs_h.Npp8u;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eAlphaOp : nppdefs_h.NppiAlphaOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:11952
   pragma Import (C, nppiAlphaCompC_8u_C1R, "nppiAlphaCompC_8u_C1R");

  --* 
  -- * Three 8-bit unsigned char channel image composition using constant alpha.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param nAlpha1 Image alpha opacity (0 - max channel pixel value).
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param nAlpha2 Image alpha opacity (0 - max channel pixel value).
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eAlphaOp alpha-blending operation..
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAlphaCompC_8u_C3R
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      nAlpha1 : nppdefs_h.Npp8u;
      pSrc2 : access nppdefs_h.Npp8u;
      nSrc2Step : int;
      nAlpha2 : nppdefs_h.Npp8u;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eAlphaOp : nppdefs_h.NppiAlphaOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:11970
   pragma Import (C, nppiAlphaCompC_8u_C3R, "nppiAlphaCompC_8u_C3R");

  --* 
  -- * Four 8-bit unsigned char channel image composition using constant alpha.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param nAlpha1 Image alpha opacity (0 - max channel pixel value).
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param nAlpha2 Image alpha opacity (0 - max channel pixel value).
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eAlphaOp alpha-blending operation..
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAlphaCompC_8u_C4R
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      nAlpha1 : nppdefs_h.Npp8u;
      pSrc2 : access nppdefs_h.Npp8u;
      nSrc2Step : int;
      nAlpha2 : nppdefs_h.Npp8u;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eAlphaOp : nppdefs_h.NppiAlphaOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:11988
   pragma Import (C, nppiAlphaCompC_8u_C4R, "nppiAlphaCompC_8u_C4R");

  --* 
  -- * Four 8-bit unsigned char channel image composition with alpha using constant source alpha.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param nAlpha1 Image alpha opacity (0 - max channel pixel value).
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param nAlpha2 Image alpha opacity (0 - max channel pixel value).
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eAlphaOp alpha-blending operation..
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAlphaCompC_8u_AC4R
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      nAlpha1 : nppdefs_h.Npp8u;
      pSrc2 : access nppdefs_h.Npp8u;
      nSrc2Step : int;
      nAlpha2 : nppdefs_h.Npp8u;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eAlphaOp : nppdefs_h.NppiAlphaOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:12006
   pragma Import (C, nppiAlphaCompC_8u_AC4R, "nppiAlphaCompC_8u_AC4R");

  --* 
  -- * One 8-bit signed char channel image composition using constant alpha.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param nAlpha1 Image alpha opacity (0 - max channel pixel value).
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param nAlpha2 Image alpha opacity (0 - max channel pixel value).
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eAlphaOp alpha-blending operation..
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAlphaCompC_8s_C1R
     (pSrc1 : access nppdefs_h.Npp8s;
      nSrc1Step : int;
      nAlpha1 : nppdefs_h.Npp8s;
      pSrc2 : access nppdefs_h.Npp8s;
      nSrc2Step : int;
      nAlpha2 : nppdefs_h.Npp8s;
      pDst : access nppdefs_h.Npp8s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eAlphaOp : nppdefs_h.NppiAlphaOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:12024
   pragma Import (C, nppiAlphaCompC_8s_C1R, "nppiAlphaCompC_8s_C1R");

  --* 
  -- * One 16-bit unsigned short channel image composition using constant alpha.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param nAlpha1 Image alpha opacity (0 - max channel pixel value).
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param nAlpha2 Image alpha opacity (0 - max channel pixel value).
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eAlphaOp alpha-blending operation..
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAlphaCompC_16u_C1R
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      nAlpha1 : nppdefs_h.Npp16u;
      pSrc2 : access nppdefs_h.Npp16u;
      nSrc2Step : int;
      nAlpha2 : nppdefs_h.Npp16u;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eAlphaOp : nppdefs_h.NppiAlphaOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:12042
   pragma Import (C, nppiAlphaCompC_16u_C1R, "nppiAlphaCompC_16u_C1R");

  --* 
  -- * Three 16-bit unsigned short channel image composition using constant alpha.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param nAlpha1 Image alpha opacity (0 - max channel pixel value).
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param nAlpha2 Image alpha opacity (0 - max channel pixel value).
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eAlphaOp alpha-blending operation..
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAlphaCompC_16u_C3R
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      nAlpha1 : nppdefs_h.Npp16u;
      pSrc2 : access nppdefs_h.Npp16u;
      nSrc2Step : int;
      nAlpha2 : nppdefs_h.Npp16u;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eAlphaOp : nppdefs_h.NppiAlphaOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:12060
   pragma Import (C, nppiAlphaCompC_16u_C3R, "nppiAlphaCompC_16u_C3R");

  --* 
  -- * Four 16-bit unsigned short channel image composition using constant alpha.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param nAlpha1 Image alpha opacity (0 - max channel pixel value).
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param nAlpha2 Image alpha opacity (0 - max channel pixel value).
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eAlphaOp alpha-blending operation..
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAlphaCompC_16u_C4R
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      nAlpha1 : nppdefs_h.Npp16u;
      pSrc2 : access nppdefs_h.Npp16u;
      nSrc2Step : int;
      nAlpha2 : nppdefs_h.Npp16u;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eAlphaOp : nppdefs_h.NppiAlphaOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:12078
   pragma Import (C, nppiAlphaCompC_16u_C4R, "nppiAlphaCompC_16u_C4R");

  --* 
  -- * Four 16-bit unsigned short channel image composition with alpha using constant source alpha.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param nAlpha1 Image alpha opacity (0 - max channel pixel value).
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param nAlpha2 Image alpha opacity (0 - max channel pixel value).
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eAlphaOp alpha-blending operation..
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAlphaCompC_16u_AC4R
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      nAlpha1 : nppdefs_h.Npp16u;
      pSrc2 : access nppdefs_h.Npp16u;
      nSrc2Step : int;
      nAlpha2 : nppdefs_h.Npp16u;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eAlphaOp : nppdefs_h.NppiAlphaOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:12096
   pragma Import (C, nppiAlphaCompC_16u_AC4R, "nppiAlphaCompC_16u_AC4R");

  --* 
  -- * One 16-bit signed short channel image composition using constant alpha.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param nAlpha1 Image alpha opacity (0 - max channel pixel value).
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param nAlpha2 Image alpha opacity (0 - max channel pixel value).
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eAlphaOp alpha-blending operation..
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAlphaCompC_16s_C1R
     (pSrc1 : access nppdefs_h.Npp16s;
      nSrc1Step : int;
      nAlpha1 : nppdefs_h.Npp16s;
      pSrc2 : access nppdefs_h.Npp16s;
      nSrc2Step : int;
      nAlpha2 : nppdefs_h.Npp16s;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eAlphaOp : nppdefs_h.NppiAlphaOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:12114
   pragma Import (C, nppiAlphaCompC_16s_C1R, "nppiAlphaCompC_16s_C1R");

  --* 
  -- * One 32-bit unsigned integer channel image composition using constant alpha.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param nAlpha1 Image alpha opacity (0 - max channel pixel value).
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param nAlpha2 Image alpha opacity (0 - max channel pixel value).
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eAlphaOp alpha-blending operation..
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAlphaCompC_32u_C1R
     (pSrc1 : access nppdefs_h.Npp32u;
      nSrc1Step : int;
      nAlpha1 : nppdefs_h.Npp32u;
      pSrc2 : access nppdefs_h.Npp32u;
      nSrc2Step : int;
      nAlpha2 : nppdefs_h.Npp32u;
      pDst : access nppdefs_h.Npp32u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eAlphaOp : nppdefs_h.NppiAlphaOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:12132
   pragma Import (C, nppiAlphaCompC_32u_C1R, "nppiAlphaCompC_32u_C1R");

  --* 
  -- * One 32-bit signed integer channel image composition using constant alpha.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param nAlpha1 Image alpha opacity (0 - max channel pixel value).
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param nAlpha2 Image alpha opacity (0 - max channel pixel value).
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eAlphaOp alpha-blending operation..
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAlphaCompC_32s_C1R
     (pSrc1 : access nppdefs_h.Npp32s;
      nSrc1Step : int;
      nAlpha1 : nppdefs_h.Npp32s;
      pSrc2 : access nppdefs_h.Npp32s;
      nSrc2Step : int;
      nAlpha2 : nppdefs_h.Npp32s;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eAlphaOp : nppdefs_h.NppiAlphaOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:12150
   pragma Import (C, nppiAlphaCompC_32s_C1R, "nppiAlphaCompC_32s_C1R");

  --* 
  -- * One 32-bit floating point channel image composition using constant alpha.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param nAlpha1 Image alpha opacity (0.0 - 1.0).
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param nAlpha2 Image alpha opacity (0.0 - 1.0).
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eAlphaOp alpha-blending operation..
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAlphaCompC_32f_C1R
     (pSrc1 : access nppdefs_h.Npp32f;
      nSrc1Step : int;
      nAlpha1 : nppdefs_h.Npp32f;
      pSrc2 : access nppdefs_h.Npp32f;
      nSrc2Step : int;
      nAlpha2 : nppdefs_h.Npp32f;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eAlphaOp : nppdefs_h.NppiAlphaOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:12168
   pragma Import (C, nppiAlphaCompC_32f_C1R, "nppiAlphaCompC_32f_C1R");

  --* @} image_alphacompc  
  --* 
  -- * @defgroup image_alphapremulc AlphaPremulC
  -- * 
  -- * Premultiplies pixels of an image using a constant alpha value.
  -- *
  -- * @{
  --  

  --* 
  -- * One 8-bit unsigned char channel image premultiplication using constant alpha.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param nAlpha1 Image alpha opacity (0 - max channel pixel value).
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAlphaPremulC_8u_C1R
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      nAlpha1 : nppdefs_h.Npp8u;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:12192
   pragma Import (C, nppiAlphaPremulC_8u_C1R, "nppiAlphaPremulC_8u_C1R");

  --* 
  -- * One 8-bit unsigned char channel in place image premultiplication using constant alpha.
  -- * \param nAlpha1 Image alpha opacity (0 - max channel pixel value).
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAlphaPremulC_8u_C1IR
     (nAlpha1 : nppdefs_h.Npp8u;
      pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:12203
   pragma Import (C, nppiAlphaPremulC_8u_C1IR, "nppiAlphaPremulC_8u_C1IR");

  --* 
  -- * Three 8-bit unsigned char channel image premultiplication using constant alpha.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param nAlpha1 Image alpha opacity (0 - max channel pixel value).
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAlphaPremulC_8u_C3R
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      nAlpha1 : nppdefs_h.Npp8u;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:12216
   pragma Import (C, nppiAlphaPremulC_8u_C3R, "nppiAlphaPremulC_8u_C3R");

  --* 
  -- * Three 8-bit unsigned char channel in place image premultiplication using constant alpha.
  -- * \param nAlpha1 Image alpha opacity (0 - max channel pixel value).
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAlphaPremulC_8u_C3IR
     (nAlpha1 : nppdefs_h.Npp8u;
      pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:12227
   pragma Import (C, nppiAlphaPremulC_8u_C3IR, "nppiAlphaPremulC_8u_C3IR");

  --* 
  -- * Four 8-bit unsigned char channel image premultiplication using constant alpha.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param nAlpha1 Image alpha opacity (0 - max channel pixel value).
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAlphaPremulC_8u_C4R
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      nAlpha1 : nppdefs_h.Npp8u;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:12240
   pragma Import (C, nppiAlphaPremulC_8u_C4R, "nppiAlphaPremulC_8u_C4R");

  --* 
  -- * Four 8-bit unsigned char channel in place image premultiplication using constant alpha.
  -- * \param nAlpha1 Image alpha opacity (0 - max channel pixel value).
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAlphaPremulC_8u_C4IR
     (nAlpha1 : nppdefs_h.Npp8u;
      pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:12251
   pragma Import (C, nppiAlphaPremulC_8u_C4IR, "nppiAlphaPremulC_8u_C4IR");

  --* 
  -- * Four 8-bit unsigned char channel image premultiplication with alpha using constant alpha.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param nAlpha1 Image alpha opacity (0 - max channel pixel value).
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAlphaPremulC_8u_AC4R
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      nAlpha1 : nppdefs_h.Npp8u;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:12264
   pragma Import (C, nppiAlphaPremulC_8u_AC4R, "nppiAlphaPremulC_8u_AC4R");

  --* 
  -- * Four 8-bit unsigned char channel in place image premultiplication with alpha using constant alpha.
  -- * \param nAlpha1 Image alpha opacity (0 - max channel pixel value).
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAlphaPremulC_8u_AC4IR
     (nAlpha1 : nppdefs_h.Npp8u;
      pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:12275
   pragma Import (C, nppiAlphaPremulC_8u_AC4IR, "nppiAlphaPremulC_8u_AC4IR");

  --* 
  -- * One 16-bit unsigned short channel image premultiplication using constant alpha.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param nAlpha1 Image alpha opacity (0 - max channel pixel value).
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAlphaPremulC_16u_C1R
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      nAlpha1 : nppdefs_h.Npp16u;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:12288
   pragma Import (C, nppiAlphaPremulC_16u_C1R, "nppiAlphaPremulC_16u_C1R");

  --* 
  -- * One 16-bit unsigned short channel in place image premultiplication using constant alpha.
  -- * \param nAlpha1 Image alpha opacity (0 - max channel pixel value).
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAlphaPremulC_16u_C1IR
     (nAlpha1 : nppdefs_h.Npp16u;
      pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:12299
   pragma Import (C, nppiAlphaPremulC_16u_C1IR, "nppiAlphaPremulC_16u_C1IR");

  --* 
  -- * Three 16-bit unsigned short channel image premultiplication using constant alpha.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param nAlpha1 Image alpha opacity (0 - max channel pixel value).
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAlphaPremulC_16u_C3R
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      nAlpha1 : nppdefs_h.Npp16u;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:12312
   pragma Import (C, nppiAlphaPremulC_16u_C3R, "nppiAlphaPremulC_16u_C3R");

  --* 
  -- * Three 16-bit unsigned short channel in place image premultiplication using constant alpha.
  -- * \param nAlpha1 Image alpha opacity (0 - max channel pixel value).
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAlphaPremulC_16u_C3IR
     (nAlpha1 : nppdefs_h.Npp16u;
      pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:12323
   pragma Import (C, nppiAlphaPremulC_16u_C3IR, "nppiAlphaPremulC_16u_C3IR");

  --* 
  -- * Four 16-bit unsigned short channel image premultiplication using constant alpha.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param nAlpha1 Image alpha opacity (0 - max channel pixel value).
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAlphaPremulC_16u_C4R
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      nAlpha1 : nppdefs_h.Npp16u;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:12336
   pragma Import (C, nppiAlphaPremulC_16u_C4R, "nppiAlphaPremulC_16u_C4R");

  --* 
  -- * Four 16-bit unsigned short channel in place image premultiplication using constant alpha.
  -- * \param nAlpha1 Image alpha opacity (0 - max channel pixel value).
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAlphaPremulC_16u_C4IR
     (nAlpha1 : nppdefs_h.Npp16u;
      pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:12347
   pragma Import (C, nppiAlphaPremulC_16u_C4IR, "nppiAlphaPremulC_16u_C4IR");

  --* 
  -- * Four 16-bit unsigned short channel image premultiplication with alpha using constant alpha.
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param nAlpha1 Image alpha opacity (0 - max channel pixel value).
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAlphaPremulC_16u_AC4R
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      nAlpha1 : nppdefs_h.Npp16u;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:12360
   pragma Import (C, nppiAlphaPremulC_16u_AC4R, "nppiAlphaPremulC_16u_AC4R");

  --* 
  -- * Four 16-bit unsigned short channel in place image premultiplication with alpha using constant alpha.
  -- * \param nAlpha1 Image alpha opacity (0 - max channel pixel value).
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAlphaPremulC_16u_AC4IR
     (nAlpha1 : nppdefs_h.Npp16u;
      pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:12371
   pragma Import (C, nppiAlphaPremulC_16u_AC4IR, "nppiAlphaPremulC_16u_AC4IR");

  --* @} image_alphapremulc  
  --* 
  -- * @defgroup image_alphacomp AlphaComp
  -- *
  -- * Composite two images using alpha opacity values contained in each image.
  -- *
  -- * @{
  --  

  --* 
  -- * One 8-bit unsigned char channel image composition using image alpha values (0 - max channel pixel value).
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eAlphaOp alpha-blending operation..
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAlphaComp_8u_AC1R
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp8u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eAlphaOp : nppdefs_h.NppiAlphaOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:12396
   pragma Import (C, nppiAlphaComp_8u_AC1R, "nppiAlphaComp_8u_AC1R");

  --* 
  -- * Four 8-bit unsigned char channel image composition using image alpha values (0 - max channel pixel value).
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eAlphaOp alpha-blending operation..
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAlphaComp_8u_AC4R
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp8u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eAlphaOp : nppdefs_h.NppiAlphaOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:12412
   pragma Import (C, nppiAlphaComp_8u_AC4R, "nppiAlphaComp_8u_AC4R");

  --* 
  -- * One 8-bit signed char channel image composition using image alpha values (0 - max channel pixel value).
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eAlphaOp alpha-blending operation..
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAlphaComp_8s_AC1R
     (pSrc1 : access nppdefs_h.Npp8s;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp8s;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp8s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eAlphaOp : nppdefs_h.NppiAlphaOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:12428
   pragma Import (C, nppiAlphaComp_8s_AC1R, "nppiAlphaComp_8s_AC1R");

  --* 
  -- * One 16-bit unsigned short channel image composition using image alpha values (0 - max channel pixel value).
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eAlphaOp alpha-blending operation..
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAlphaComp_16u_AC1R
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp16u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eAlphaOp : nppdefs_h.NppiAlphaOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:12444
   pragma Import (C, nppiAlphaComp_16u_AC1R, "nppiAlphaComp_16u_AC1R");

  --* 
  -- * Four 16-bit unsigned short channel image composition using image alpha values (0 - max channel pixel value).
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eAlphaOp alpha-blending operation..
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAlphaComp_16u_AC4R
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp16u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eAlphaOp : nppdefs_h.NppiAlphaOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:12460
   pragma Import (C, nppiAlphaComp_16u_AC4R, "nppiAlphaComp_16u_AC4R");

  --* 
  -- * One 16-bit signed short channel image composition using image alpha values (0 - max channel pixel value).
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eAlphaOp alpha-blending operation..
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAlphaComp_16s_AC1R
     (pSrc1 : access nppdefs_h.Npp16s;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp16s;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eAlphaOp : nppdefs_h.NppiAlphaOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:12476
   pragma Import (C, nppiAlphaComp_16s_AC1R, "nppiAlphaComp_16s_AC1R");

  --* 
  -- * One 32-bit unsigned integer channel image composition using image alpha values (0 - max channel pixel value).
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eAlphaOp alpha-blending operation..
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAlphaComp_32u_AC1R
     (pSrc1 : access nppdefs_h.Npp32u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp32u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp32u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eAlphaOp : nppdefs_h.NppiAlphaOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:12492
   pragma Import (C, nppiAlphaComp_32u_AC1R, "nppiAlphaComp_32u_AC1R");

  --* 
  -- * Four 32-bit unsigned integer channel image composition using image alpha values (0 - max channel pixel value).
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eAlphaOp alpha-blending operation..
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAlphaComp_32u_AC4R
     (pSrc1 : access nppdefs_h.Npp32u;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp32u;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp32u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eAlphaOp : nppdefs_h.NppiAlphaOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:12508
   pragma Import (C, nppiAlphaComp_32u_AC4R, "nppiAlphaComp_32u_AC4R");

  --* 
  -- * One 32-bit signed integer channel image composition using image alpha values (0 - max channel pixel value).
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eAlphaOp alpha-blending operation..
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAlphaComp_32s_AC1R
     (pSrc1 : access nppdefs_h.Npp32s;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp32s;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eAlphaOp : nppdefs_h.NppiAlphaOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:12524
   pragma Import (C, nppiAlphaComp_32s_AC1R, "nppiAlphaComp_32s_AC1R");

  --* 
  -- * Four 32-bit signed integer channel image composition using image alpha values (0 - max channel pixel value).
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eAlphaOp alpha-blending operation..
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAlphaComp_32s_AC4R
     (pSrc1 : access nppdefs_h.Npp32s;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp32s;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp32s;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eAlphaOp : nppdefs_h.NppiAlphaOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:12540
   pragma Import (C, nppiAlphaComp_32s_AC4R, "nppiAlphaComp_32s_AC4R");

  --* 
  -- * One 32-bit floating point channel image composition using image alpha values (0.0 - 1.0).
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eAlphaOp alpha-blending operation..
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAlphaComp_32f_AC1R
     (pSrc1 : access nppdefs_h.Npp32f;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp32f;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eAlphaOp : nppdefs_h.NppiAlphaOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:12556
   pragma Import (C, nppiAlphaComp_32f_AC1R, "nppiAlphaComp_32f_AC1R");

  --* 
  -- * Four 32-bit floating point channel image composition using image alpha values (0.0 - 1.0).
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pSrc2 \ref source_image_pointer.
  -- * \param nSrc2Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param eAlphaOp alpha-blending operation..
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAlphaComp_32f_AC4R
     (pSrc1 : access nppdefs_h.Npp32f;
      nSrc1Step : int;
      pSrc2 : access nppdefs_h.Npp32f;
      nSrc2Step : int;
      pDst : access nppdefs_h.Npp32f;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize;
      eAlphaOp : nppdefs_h.NppiAlphaOp) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:12572
   pragma Import (C, nppiAlphaComp_32f_AC4R, "nppiAlphaComp_32f_AC4R");

  --* @} image_alphacomp  
  --* 
  -- * @defgroup image_alphapremul AlphaPremul
  -- * 
  -- * Premultiplies image pixels by image alpha opacity values.
  -- *
  -- * @{
  --  

  --* 
  -- * Four 8-bit unsigned char channel image premultiplication with pixel alpha (0 - max channel pixel value).
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAlphaPremul_8u_AC4R
     (pSrc1 : access nppdefs_h.Npp8u;
      nSrc1Step : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:12595
   pragma Import (C, nppiAlphaPremul_8u_AC4R, "nppiAlphaPremul_8u_AC4R");

  --* 
  -- * Four 8-bit unsigned char channel in place image premultiplication with pixel alpha (0 - max channel pixel value).
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAlphaPremul_8u_AC4IR
     (pSrcDst : access nppdefs_h.Npp8u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:12605
   pragma Import (C, nppiAlphaPremul_8u_AC4IR, "nppiAlphaPremul_8u_AC4IR");

  --* 
  -- * Four 16-bit unsigned short channel image premultiplication with pixel alpha (0 - max channel pixel value).
  -- * \param pSrc1 \ref source_image_pointer.
  -- * \param nSrc1Step \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAlphaPremul_16u_AC4R
     (pSrc1 : access nppdefs_h.Npp16u;
      nSrc1Step : int;
      pDst : access nppdefs_h.Npp16u;
      nDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:12617
   pragma Import (C, nppiAlphaPremul_16u_AC4R, "nppiAlphaPremul_16u_AC4R");

  --* 
  -- * Four 16-bit unsigned short channel in place image premultiplication with pixel alpha (0 - max channel pixel value).
  -- * \param pSrcDst \ref in_place_image_pointer.
  -- * \param nSrcDstStep \ref in_place_image_line_step.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return \ref image_data_error_codes, \ref roi_error_codes
  --  

   function nppiAlphaPremul_16u_AC4IR
     (pSrcDst : access nppdefs_h.Npp16u;
      nSrcDstStep : int;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_arithmetic_and_logical_operations.h:12627
   pragma Import (C, nppiAlphaPremul_16u_AC4IR, "nppiAlphaPremul_16u_AC4IR");

  --* @} image_alphapremul  
  --* @} image_alpha_composition 
  --* @} image_arithmetic_and_logical_operations  
  -- extern "C"  
end nppi_arithmetic_and_logical_operations_h;

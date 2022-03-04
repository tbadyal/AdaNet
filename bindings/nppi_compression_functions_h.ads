pragma Ada_2005;
pragma Style_Checks (Off);

with Interfaces.C; use Interfaces.C;
with nppdefs_h;
with System;

package nppi_compression_functions_h is

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
  -- * \file nppi_compression_functions.h
  -- * NPP Image Processing Functionality.
  --  

  --* @defgroup image_compression Compression
  -- *  @ingroup nppi
  -- *
  -- * Image compression primitives.
  -- *
  -- * The JPEG standard defines a flow of level shift, DCT and quantization for
  -- * forward JPEG transform and inverse level shift, IDCT and de-quantization
  -- * for inverse JPEG transform. This group has the functions for both forward
  -- * and inverse functions. 
  -- *
  -- * @{
  -- *
  -- * These functions can be found in either the nppi or nppicom libraries. Linking to only the sub-libraries that you use can significantly
  -- * save link time, application load time, and CUDA runtime startup time when using dynamic libraries.
  -- *
  --  

  --* @defgroup image_quantization Quantization Functions
  -- *
  -- * @{
  -- *
  --  

  --*
  -- * Apply quality factor to raw 8-bit quantization table.
  -- *
  -- * This is effectively and in-place method that modifies a given raw
  -- * quantization table based on a quality factor.
  -- * Note that this method is a host method and that the pointer to the
  -- * raw quantization table is a host pointer.
  -- *
  -- * \param hpQuantRawTable Raw quantization table.
  -- * \param nQualityFactor Quality factor for the table. Range is [1:100].
  -- * \return Error code:
  -- *      ::NPP_NULL_POINTER_ERROR is returned if hpQuantRawTable is 0.
  --  

   function nppiQuantFwdRawTableInit_JPEG_8u (hpQuantRawTable : access nppdefs_h.Npp8u; nQualityFactor : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_compression_functions.h:99
   pragma Import (C, nppiQuantFwdRawTableInit_JPEG_8u, "nppiQuantFwdRawTableInit_JPEG_8u");

  --*
  -- * Initializes a quantization table for nppiDCTQuantFwd8x8LS_JPEG_8u16s_C1R().
  -- *    The method creates a 16-bit version of the raw table and converts the 
  -- * data order from zigzag layout to original row-order layout since raw
  -- * quantization tables are typically stored in zigzag format.
  -- *
  -- * This method is a host method. It consumes and produces host data. I.e. the pointers
  -- * passed to this function must be host pointers. The resulting table needs to be
  -- * transferred to device memory in order to be used with nppiDCTQuantFwd8x8LS_JPEG_8u16s_C1R()
  -- * function.
  -- *
  -- * \param hpQuantRawTable Host pointer to raw quantization table as returned by 
  -- *      nppiQuantFwdRawTableInit_JPEG_8u(). The raw quantization table is assumed to be in
  -- *      zigzag order.
  -- * \param hpQuantFwdRawTable Forward quantization table for use with nppiDCTQuantFwd8x8LS_JPEG_8u16s_C1R().
  -- * \return Error code:
  -- *      ::NPP_NULL_POINTER_ERROR pQuantRawTable is 0.
  --  

   function nppiQuantFwdTableInit_JPEG_8u16u (hpQuantRawTable : access nppdefs_h.Npp8u; hpQuantFwdRawTable : access nppdefs_h.Npp16u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_compression_functions.h:120
   pragma Import (C, nppiQuantFwdTableInit_JPEG_8u16u, "nppiQuantFwdTableInit_JPEG_8u16u");

  --*
  -- * Initializes a quantization table for nppiDCTQuantInv8x8LS_JPEG_16s8u_C1R().
  -- *      The nppiDCTQuantFwd8x8LS_JPEG_8u16s_C1R() method uses a quantization table
  -- * in a 16-bit format allowing for faster processing. In addition it converts the 
  -- * data order from zigzag layout to original row-order layout. Typically raw
  -- * quantization tables are stored in zigzag format.
  -- *
  -- * This method is a host method and consumes and produces host data. I.e. the pointers
  -- * passed to this function must be host pointers. The resulting table needs to be
  -- * transferred to device memory in order to be used with nppiDCTQuantFwd8x8LS_JPEG_8u16s_C1R()
  -- * function.
  -- *
  -- * \param hpQuantRawTable Raw quantization table.
  -- * \param hpQuantFwdRawTable Inverse quantization table.
  -- * \return ::NPP_NULL_POINTER_ERROR pQuantRawTable or pQuantFwdRawTable is0.
  --  

   function nppiQuantInvTableInit_JPEG_8u16u (hpQuantRawTable : access nppdefs_h.Npp8u; hpQuantFwdRawTable : access nppdefs_h.Npp16u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_compression_functions.h:139
   pragma Import (C, nppiQuantInvTableInit_JPEG_8u16u, "nppiQuantInvTableInit_JPEG_8u16u");

  --*
  -- * Forward DCT, quantization and level shift part of the JPEG encoding.
  -- * Input is expected in 8x8 macro blocks and output is expected to be in 64x1
  -- * macro blocks.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param pQuantFwdTable Forward quantization tables for JPEG encoding created
  -- *          using nppiQuantInvTableInit_JPEG_8u16u().
  -- * \param oSizeROI \ref roi_specification.
  -- * \return Error codes:
  -- *         - ::NPP_SIZE_ERROR For negative input height/width or not a multiple of
  -- *           8 width/height.
  -- *         - ::NPP_STEP_ERROR If input image width is not multiple of 8 or does not
  -- *           match ROI.
  -- *         - ::NPP_NULL_POINTER_ERROR If the destination pointer is 0.
  --  

   function nppiDCTQuantFwd8x8LS_JPEG_8u16s_C1R
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      pQuantFwdTable : access nppdefs_h.Npp16u;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_compression_functions.h:162
   pragma Import (C, nppiDCTQuantFwd8x8LS_JPEG_8u16s_C1R, "nppiDCTQuantFwd8x8LS_JPEG_8u16s_C1R");

  --*
  -- * Inverse DCT, de-quantization and level shift part of the JPEG decoding.
  -- * Input is expected in 64x1 macro blocks and output is expected to be in 8x8
  -- * macro blocks.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep Image width in pixels x 8 x sizeof(Npp16s).
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep Image width in pixels x 8 x sizeof(Npp16s).
  -- * \param pQuantInvTable Inverse quantization tables for JPEG decoding created
  -- *           using nppiQuantInvTableInit_JPEG_8u16u().
  -- * \param oSizeROI \ref roi_specification.
  -- * \return Error codes:
  -- *         - ::NPP_SIZE_ERROR For negative input height/width or not a multiple of
  -- *           8 width/height.
  -- *         - ::NPP_STEP_ERROR If input image width is not multiple of 8 or does not
  -- *           match ROI.
  -- *         - ::NPP_NULL_POINTER_ERROR If the destination pointer is 0.
  --  

   function nppiDCTQuantInv8x8LS_JPEG_16s8u_C1R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      pQuantInvTable : access nppdefs_h.Npp16u;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_compression_functions.h:186
   pragma Import (C, nppiDCTQuantInv8x8LS_JPEG_16s8u_C1R, "nppiDCTQuantInv8x8LS_JPEG_16s8u_C1R");

   --  skipped empty struct NppiDCTState

  --*
  -- * Initializes DCT state structure and allocates additional resources.
  -- *
  -- * \see nppiDCTQuantFwd8x8LS_JPEG_8u16s_C1R_NEW(), nppiDCTQuantInv8x8LS_JPEG_16s8u_C1R_NEW.
  -- * 
  -- * \param ppState Pointer to pointer to DCT state structure. 
  -- *
  -- * \return NPP_SUCCESS Indicates no error. Any other value indicates an error
  -- *         or a warning
  -- * \return NPP_SIZE_ERROR Indicates an error condition if any image dimension
  -- *         has zero or negative value
  -- * \return NPP_NULL_POINTER_ERROR Indicates an error condition if pBufSize
  -- *         pointer is NULL
  --  

   function nppiDCTInitAlloc (ppState : System.Address) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_compression_functions.h:213
   pragma Import (C, nppiDCTInitAlloc, "nppiDCTInitAlloc");

  --*
  -- * Frees the additional resources of the DCT state structure.
  -- *
  -- * \see nppiDCTInitAlloc
  -- * 
  -- * \param pState Pointer to DCT state structure. 
  -- *
  -- * \return NPP_SUCCESS Indicates no error. Any other value indicates an error
  -- *         or a warning
  -- * \return NPP_SIZE_ERROR Indicates an error condition if any image dimension
  -- *         has zero or negative value
  -- * \return NPP_NULL_POINTER_ERROR Indicates an error condition if pState
  -- *         pointer is NULL
  --  

   function nppiDCTFree (pState : System.Address) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_compression_functions.h:229
   pragma Import (C, nppiDCTFree, "nppiDCTFree");

  --*
  -- * Forward DCT, quantization and level shift part of the JPEG encoding.
  -- * Input is expected in 8x8 macro blocks and output is expected to be in 64x1
  -- * macro blocks. The new version of the primitive takes the ROI in image pixel size and
  -- * works with DCT coefficients that are in zig-zag order.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep \ref source_image_line_step.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep Image width in pixels x 8 x sizeof(Npp16s).
  -- * \param pQuantizationTable Quantization Table in zig-zag order.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pState Pointer to DCT state structure. This structure must be
  -- *          initialized allocated and initialized using nppiDCTInitAlloc(). 
  -- * \return Error codes:
  -- *         - ::NPP_SIZE_ERROR For negative input height/width or not a multiple of
  -- *           8 width/height.
  -- *         - ::NPP_STEP_ERROR If input image width is not multiple of 8 or does not
  -- *           match ROI.
  -- *         - ::NPP_NULL_POINTER_ERROR If the destination pointer is 0.
  --  

   function nppiDCTQuantFwd8x8LS_JPEG_8u16s_C1R_NEW
     (pSrc : access nppdefs_h.Npp8u;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : int;
      pQuantizationTable : access nppdefs_h.Npp8u;
      oSizeROI : nppdefs_h.NppiSize;
      pState : System.Address) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_compression_functions.h:253
   pragma Import (C, nppiDCTQuantFwd8x8LS_JPEG_8u16s_C1R_NEW, "nppiDCTQuantFwd8x8LS_JPEG_8u16s_C1R_NEW");

  --*
  -- * Inverse DCT, de-quantization and level shift part of the JPEG decoding.
  -- * Input is expected in 64x1 macro blocks and output is expected to be in 8x8
  -- * macro blocks. The new version of the primitive takes the ROI in image pixel size and
  -- * works with DCT coefficients that are in zig-zag order.
  -- *
  -- * \param pSrc \ref source_image_pointer.
  -- * \param nSrcStep Image width in pixels x 8 x sizeof(Npp16s).
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param pQuantizationTable Quantization Table in zig-zag order.
  -- * \param oSizeROI \ref roi_specification.
  -- * \param pState Pointer to DCT state structure. This structure must be
  -- *          initialized allocated and initialized using nppiDCTInitAlloc().  
  -- * \return Error codes:
  -- *         - ::NPP_SIZE_ERROR For negative input height/width or not a multiple of
  -- *           8 width/height.
  -- *         - ::NPP_STEP_ERROR If input image width is not multiple of 8 or does not
  -- *           match ROI.
  -- *         - ::NPP_NULL_POINTER_ERROR If the destination pointer is 0.
  --  

   function nppiDCTQuantInv8x8LS_JPEG_16s8u_C1R_NEW
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : int;
      pDst : access nppdefs_h.Npp8u;
      nDstStep : int;
      pQuantizationTable : access nppdefs_h.Npp8u;
      oSizeROI : nppdefs_h.NppiSize;
      pState : System.Address) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_compression_functions.h:282
   pragma Import (C, nppiDCTQuantInv8x8LS_JPEG_16s8u_C1R_NEW, "nppiDCTQuantInv8x8LS_JPEG_16s8u_C1R_NEW");

  --* @} image_quantization  
   --  skipped empty struct NppiDecodeHuffmanSpec

  --*
  -- * Returns the length of the NppiDecodeHuffmanSpec structure.
  -- * \param pSize Pointer to a variable that will receive the length of the NppiDecodeHuffmanSpec structure.
  -- * \return Error codes:
  -- *         - ::NPP_NULL_POINTER_ERROR If one of the pointers is 0.
  --* 

   function nppiDecodeHuffmanSpecGetBufSize_JPEG (pSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_compression_functions.h:303
   pragma Import (C, nppiDecodeHuffmanSpecGetBufSize_JPEG, "nppiDecodeHuffmanSpecGetBufSize_JPEG");

  --*
  -- * Creates a Huffman table in a format that is suitable for the decoder on the host.
  -- * \param pRawHuffmanTable Huffman table formated as specified in the JPEG standard.
  -- * \param eTableType Enum specifying type of table (nppiDCTable or nppiACTable).
  -- * \param pHuffmanSpec Pointer to the Huffman table for the decoder
  -- * \return Error codes:
  -- *         - ::NPP_NULL_POINTER_ERROR If one of the pointers is 0.
  --* 

   function nppiDecodeHuffmanSpecInitHost_JPEG
     (pRawHuffmanTable : access nppdefs_h.Npp8u;
      eTableType : nppdefs_h.NppiHuffmanTableType;
      pHuffmanSpec : System.Address) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_compression_functions.h:314
   pragma Import (C, nppiDecodeHuffmanSpecInitHost_JPEG, "nppiDecodeHuffmanSpecInitHost_JPEG");

  --*
  -- * Allocates memory and creates a Huffman table in a format that is suitable for the decoder on the host.
  -- * \param pRawHuffmanTable Huffman table formated as specified in the JPEG standard.
  -- * \param eTableType Enum specifying type of table (nppiDCTable or nppiACTable).
  -- * \param ppHuffmanSpec Pointer to returned pointer to the Huffman table for the decoder
  -- * \return Error codes:
  -- *         - ::NPP_NULL_POINTER_ERROR If one of the pointers is 0.
  --* 

   function nppiDecodeHuffmanSpecInitAllocHost_JPEG
     (pRawHuffmanTable : access nppdefs_h.Npp8u;
      eTableType : nppdefs_h.NppiHuffmanTableType;
      ppHuffmanSpec : System.Address) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_compression_functions.h:325
   pragma Import (C, nppiDecodeHuffmanSpecInitAllocHost_JPEG, "nppiDecodeHuffmanSpecInitAllocHost_JPEG");

  --*
  -- * Frees the host memory allocated by nppiDecodeHuffmanSpecInitAllocHost_JPEG.
  -- * \param pHuffmanSpec Pointer to the Huffman table for the decoder
  --* 

   function nppiDecodeHuffmanSpecFreeHost_JPEG (pHuffmanSpec : System.Address) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_compression_functions.h:333
   pragma Import (C, nppiDecodeHuffmanSpecFreeHost_JPEG, "nppiDecodeHuffmanSpecFreeHost_JPEG");

  --*
  -- * Huffman Decoding of the JPEG decoding on the host.
  -- * Input is expected in byte stuffed huffman encoded JPEG scan and output is expected to be 64x1 macro blocks.
  -- *
  -- * \param pSrc Byte-stuffed huffman encoded JPEG scan.
  -- * \param nLength Byte length of the input.
  -- * \param restartInterval Restart Interval, see JPEG standard.
  -- * \param Ss Start Coefficient, see JPEG standard.
  -- * \param Se End Coefficient, see JPEG standard.
  -- * \param Ah Bit Approximation High, see JPEG standard.
  -- * \param Al Bit Approximation Low, see JPEG standard.
  -- * \param pDst \ref destination_image_pointer.
  -- * \param nDstStep \ref destination_image_line_step.
  -- * \param pHuffmanTableDC DC Huffman table.
  -- * \param pHuffmanTableAC AC Huffman table.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return Error codes:
  -- *         - ::NPP_SIZE_ERROR For negative input height/width or not a multiple of
  -- *           8 width/height.
  -- *         - ::NPP_STEP_ERROR If input image width is not multiple of 8 or does not
  -- *           match ROI.
  -- *         - ::NPP_NULL_POINTER_ERROR If the destination pointer is 0.
  --  

   function nppiDecodeHuffmanScanHost_JPEG_8u16s_P1R
     (pSrc : access nppdefs_h.Npp8u;
      nLength : nppdefs_h.Npp32s;
      restartInterval : nppdefs_h.Npp32s;
      Ss : nppdefs_h.Npp32s;
      Se : nppdefs_h.Npp32s;
      Ah : nppdefs_h.Npp32s;
      Al : nppdefs_h.Npp32s;
      pDst : access nppdefs_h.Npp16s;
      nDstStep : nppdefs_h.Npp32s;
      pHuffmanTableDC : System.Address;
      pHuffmanTableAC : System.Address;
      oSizeROI : nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_compression_functions.h:359
   pragma Import (C, nppiDecodeHuffmanScanHost_JPEG_8u16s_P1R, "nppiDecodeHuffmanScanHost_JPEG_8u16s_P1R");

  --*
  -- * Huffman Decoding of the JPEG decoding on the host.
  -- * Input is expected in byte stuffed huffman encoded JPEG scan and output is expected to be 64x1 macro blocks.
  -- *
  -- * \param pSrc Byte-stuffed huffman encoded JPEG scan.
  -- * \param nLength Byte length of the input.
  -- * \param nRestartInterval Restart Interval, see JPEG standard. 
  -- * \param nSs Start Coefficient, see JPEG standard.
  -- * \param nSe End Coefficient, see JPEG standard.
  -- * \param nAh Bit Approximation High, see JPEG standard.
  -- * \param nAl Bit Approximation Low, see JPEG standard.
  -- * \param apDst \ref destination_image_pointer.
  -- * \param aDstStep \ref destination_image_line_step.
  -- * \param apHuffmanDCTable DC Huffman tables.
  -- * \param apHuffmanACTable AC Huffman tables.
  -- * \param aSizeROI \ref roi_specification.
  -- * \return Error codes:
  -- *         - ::NPP_SIZE_ERROR For negative input height/width or not a multiple of
  -- *           8 width/height.
  -- *         - ::NPP_STEP_ERROR If input image width is not multiple of 8 or does not
  -- *           match ROI.
  -- *         - ::NPP_NULL_POINTER_ERROR If the destination pointer is 0.
  --  

   function nppiDecodeHuffmanScanHost_JPEG_8u16s_P3R
     (pSrc : access nppdefs_h.Npp8u;
      nLength : nppdefs_h.Npp32s;
      nRestartInterval : nppdefs_h.Npp32s;
      nSs : nppdefs_h.Npp32s;
      nSe : nppdefs_h.Npp32s;
      nAh : nppdefs_h.Npp32s;
      nAl : nppdefs_h.Npp32s;
      apDst : System.Address;
      aDstStep : access nppdefs_h.Npp32s;
      apHuffmanDCTable : System.Address;
      apHuffmanACTable : System.Address;
      aSizeROI : access nppdefs_h.NppiSize) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_compression_functions.h:391
   pragma Import (C, nppiDecodeHuffmanScanHost_JPEG_8u16s_P3R, "nppiDecodeHuffmanScanHost_JPEG_8u16s_P3R");

  --* @} image_compression  
   --  skipped empty struct NppiEncodeHuffmanSpec

  --*
  -- * Returns the length of the NppiEncodeHuffmanSpec structure.
  -- * \param pSize Pointer to a variable that will receive the length of the NppiEncodeHuffmanSpec structure.
  -- * \return Error codes:
  -- *         - ::NPP_NULL_POINTER_ERROR If one of the pointers is 0.
  --* 

   function nppiEncodeHuffmanSpecGetBufSize_JPEG (pSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_compression_functions.h:414
   pragma Import (C, nppiEncodeHuffmanSpecGetBufSize_JPEG, "nppiEncodeHuffmanSpecGetBufSize_JPEG");

  --*
  -- * Creates a Huffman table in a format that is suitable for the encoder.
  -- * \param pRawHuffmanTable Huffman table formated as specified in the JPEG standard.
  -- * \param eTableType Enum specifying type of table (nppiDCTable or nppiACTable).
  -- * \param pHuffmanSpec Pointer to the Huffman table for the decoder
  -- * \return Error codes:
  -- *         - ::NPP_NULL_POINTER_ERROR If one of the pointers is 0.
  --* 

   function nppiEncodeHuffmanSpecInit_JPEG
     (pRawHuffmanTable : access nppdefs_h.Npp8u;
      eTableType : nppdefs_h.NppiHuffmanTableType;
      pHuffmanSpec : System.Address) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_compression_functions.h:425
   pragma Import (C, nppiEncodeHuffmanSpecInit_JPEG, "nppiEncodeHuffmanSpecInit_JPEG");

  --*
  -- * Allocates memory and creates a Huffman table in a format that is suitable for the encoder.
  -- * \param pRawHuffmanTable Huffman table formated as specified in the JPEG standard.
  -- * \param eTableType Enum specifying type of table (nppiDCTable or nppiACTable).
  -- * \param ppHuffmanSpec Pointer to returned pointer to the Huffman table for the encoder
  -- * \return Error codes:
  -- *         - ::NPP_NULL_POINTER_ERROR If one of the pointers is 0.
  --* 

   function nppiEncodeHuffmanSpecInitAlloc_JPEG
     (pRawHuffmanTable : access nppdefs_h.Npp8u;
      eTableType : nppdefs_h.NppiHuffmanTableType;
      ppHuffmanSpec : System.Address) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_compression_functions.h:436
   pragma Import (C, nppiEncodeHuffmanSpecInitAlloc_JPEG, "nppiEncodeHuffmanSpecInitAlloc_JPEG");

  --*
  -- * Frees the memory allocated by nppiEncodeHuffmanSpecInitAlloc_JPEG.
  -- * \param pHuffmanSpec Pointer to the Huffman table for the encoder
  --* 

   function nppiEncodeHuffmanSpecFree_JPEG (pHuffmanSpec : System.Address) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_compression_functions.h:444
   pragma Import (C, nppiEncodeHuffmanSpecFree_JPEG, "nppiEncodeHuffmanSpecFree_JPEG");

  --*
  -- * Huffman Encoding of the JPEG Encoding.
  -- * Input is expected to be 64x1 macro blocks and output is expected as byte stuffed huffman encoded JPEG scan.
  -- *
  -- * \param pSrc \ref destination_image_pointer.
  -- * \param nSrcStep \ref destination_image_line_step.
  -- * \param nRestartInterval Restart Interval, see JPEG standard. Currently only values <=0 are supported.
  -- * \param nSs Start Coefficient, see JPEG standard.
  -- * \param nSe End Coefficient, see JPEG standard.
  -- * \param nAh Bit Approximation High, see JPEG standard.
  -- * \param nAl Bit Approximation Low, see JPEG standard.
  -- * \param pDst Byte-stuffed huffman encoded JPEG scan.
  -- * \param nLength Byte length of the huffman encoded JPEG scan.
  -- * \param pHuffmanTableDC DC Huffman table.
  -- * \param pHuffmanTableAC AC Huffman table.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return Error codes:
  -- *         - ::NPP_SIZE_ERROR For negative input height/width or not a multiple of
  -- *           8 width/height.
  -- *         - ::NPP_STEP_ERROR If input image width is not multiple of 8 or does not
  -- *           match ROI.
  -- *         - ::NPP_NULL_POINTER_ERROR If the destination pointer is 0.
  -- *         - ::NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY If the device has compute capability < 2.0. 
  --  

   function nppiEncodeHuffmanScan_JPEG_8u16s_P1R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : nppdefs_h.Npp32s;
      restartInterval : nppdefs_h.Npp32s;
      Ss : nppdefs_h.Npp32s;
      Se : nppdefs_h.Npp32s;
      Ah : nppdefs_h.Npp32s;
      Al : nppdefs_h.Npp32s;
      pDst : access nppdefs_h.Npp8u;
      nLength : access nppdefs_h.Npp32s;
      pHuffmanTableDC : System.Address;
      pHuffmanTableAC : System.Address;
      oSizeROI : nppdefs_h.NppiSize;
      pTempStorage : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_compression_functions.h:471
   pragma Import (C, nppiEncodeHuffmanScan_JPEG_8u16s_P1R, "nppiEncodeHuffmanScan_JPEG_8u16s_P1R");

  --*
  -- * Huffman Encoding of the JPEG Encoding.
  -- * Input is expected to be 64x1 macro blocks and output is expected as byte stuffed huffman encoded JPEG scan.
  -- *
  -- * \param apSrc \ref destination_image_pointer.
  -- * \param aSrcStep \ref destination_image_line_step.
  -- * \param nRestartInterval Restart Interval, see JPEG standard. Currently only values <=0 are supported.
  -- * \param nSs Start Coefficient, see JPEG standard.
  -- * \param nSe End Coefficient, see JPEG standard.
  -- * \param nAh Bit Approximation High, see JPEG standard.
  -- * \param nAl Bit Approximation Low, see JPEG standard.
  -- * \param pDst Byte-stuffed huffman encoded JPEG scan.
  -- * \param nLength Byte length of the huffman encoded JPEG scan.
  -- * \param apHuffmanTableDC DC Huffman tables.
  -- * \param apHuffmanTableAC AC Huffman tables.
  -- * \param aSizeROI \ref roi_specification.
  -- * \return Error codes:
  -- *         - ::NPP_SIZE_ERROR For negative input height/width or not a multiple of
  -- *           8 width/height.
  -- *         - ::NPP_STEP_ERROR If input image width is not multiple of 8 or does not
  -- *           match ROI.
  -- *         - ::NPP_NULL_POINTER_ERROR If the destination pointer is 0.
  -- *         - ::NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY If the device has compute capability < 2.0.
  --  

   function nppiEncodeHuffmanScan_JPEG_8u16s_P3R
     (apSrc : System.Address;
      aSrcStep : access nppdefs_h.Npp32s;
      nRestartInterval : nppdefs_h.Npp32s;
      nSs : nppdefs_h.Npp32s;
      nSe : nppdefs_h.Npp32s;
      nAh : nppdefs_h.Npp32s;
      nAl : nppdefs_h.Npp32s;
      pDst : access nppdefs_h.Npp8u;
      nLength : access nppdefs_h.Npp32s;
      apHuffmanDCTable : System.Address;
      apHuffmanACTable : System.Address;
      aSizeROI : access nppdefs_h.NppiSize;
      pTempStorage : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_compression_functions.h:505
   pragma Import (C, nppiEncodeHuffmanScan_JPEG_8u16s_P3R, "nppiEncodeHuffmanScan_JPEG_8u16s_P3R");

  --*
  -- * Optimize Huffman Encoding of the JPEG Encoding.
  -- * Input is expected to be 64x1 macro blocks and output is expected as byte stuffed huffman encoded JPEG scan.
  -- *
  -- * \param pSrc \ref destination_image_pointer.
  -- * \param nSrcStep \ref destination_image_line_step.
  -- * \param nRestartInterval Restart Interval, see JPEG standard. Currently only values <=0 are supported.
  -- * \param nSs Start Coefficient, see JPEG standard.
  -- * \param nSe End Coefficient, see JPEG standard.
  -- * \param nAh Bit Approximation High, see JPEG standard.
  -- * \param nAl Bit Approximation Low, see JPEG standard.
  -- * \param pDst Byte-stuffed huffman encoded JPEG scan.
  -- * \param pLength Pointer to the byte length of the huffman encoded JPEG scan.
  -- * \param hpCodesDC Host pointer to the code of the huffman tree for DC component.
  -- * \param hpTableDC Host pointer to the table of the huffman tree for DC component.
  -- * \param hpCodesAC Host pointer to the code of the huffman tree for AC component.
  -- * \param hpTableAC Host pointer to the table of the huffman tree for AC component.
  --* \param pHuffmanTableDC DC Huffman table.
  -- * \param pHuffmanTableAC AC Huffman table.
  -- * \param oSizeROI \ref roi_specification.
  -- * \return Error codes:
  -- *         - ::NPP_SIZE_ERROR For negative input height/width or not a multiple of
  -- *           8 width/height.
  -- *         - ::NPP_STEP_ERROR If input image width is not multiple of 8 or does not
  -- *           match ROI.
  -- *         - ::NPP_NULL_POINTER_ERROR If the destination pointer is 0.
  -- *         - ::NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY If the device has compute capability < 2.0. 
  --  

   function nppiEncodeOptimizeHuffmanScan_JPEG_8u16s_P1R
     (pSrc : access nppdefs_h.Npp16s;
      nSrcStep : nppdefs_h.Npp32s;
      nRestartInterval : nppdefs_h.Npp32s;
      nSs : nppdefs_h.Npp32s;
      nSe : nppdefs_h.Npp32s;
      nAh : nppdefs_h.Npp32s;
      nAl : nppdefs_h.Npp32s;
      pDst : access nppdefs_h.Npp8u;
      pLength : access nppdefs_h.Npp32s;
      hpCodesDC : access nppdefs_h.Npp8u;
      hpTableDC : access nppdefs_h.Npp8u;
      hpCodesAC : access nppdefs_h.Npp8u;
      hpTableAC : access nppdefs_h.Npp8u;
      aHuffmanDCTable : System.Address;
      aHuffmanACTable : System.Address;
      oSizeROI : nppdefs_h.NppiSize;
      pTempStorage : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_compression_functions.h:542
   pragma Import (C, nppiEncodeOptimizeHuffmanScan_JPEG_8u16s_P1R, "nppiEncodeOptimizeHuffmanScan_JPEG_8u16s_P1R");

  --*
  -- * Optimize Huffman Encoding of the JPEG Encoding.
  -- * Input is expected to be 64x1 macro blocks and output is expected as byte stuffed huffman encoded JPEG scan.
  -- *
  -- * \param apSrc \ref destination_image_pointer.
  -- * \param aSrcStep \ref destination_image_line_step.
  -- * \param nRestartInterval Restart Interval, see JPEG standard. Currently only values <=0 are supported.
  -- * \param nSs Start Coefficient, see JPEG standard.
  -- * \param nSe End Coefficient, see JPEG standard.
  -- * \param nAh Bit Approximation High, see JPEG standard.
  -- * \param nAl Bit Approximation Low, see JPEG standard.
  -- * \param pDst Byte-stuffed huffman encoded JPEG scan.
  -- * \param pLength Pointer to the byte length of the huffman encoded JPEG scan.
  -- * \param hpCodesDC Host pointer to the code of the huffman tree for DC component.
  -- * \param hpTableDC Host pointer to the table of the huffman tree for DC component.
  -- * \param hpCodesAC Host pointer to the code of the huffman tree for AC component.
  -- * \param hpTableAC Host pointer to the table of the huffman tree for AC component.
  -- * \param apHuffmanTableDC DC Huffman tables.
  -- * \param apHuffmanTableAC AC Huffman tables.
  -- * \param aSizeROI \ref roi_specification.
  -- * \return Error codes:
  -- *         - ::NPP_SIZE_ERROR For negative input height/width or not a multiple of
  -- *           8 width/height.
  -- *         - ::NPP_STEP_ERROR If input image width is not multiple of 8 or does not
  -- *           match ROI.
  -- *         - ::NPP_NULL_POINTER_ERROR If the destination pointer is 0.
  -- *         - ::NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY If the device has compute capability < 2.0.
  --  

   function nppiEncodeOptimizeHuffmanScan_JPEG_8u16s_P3R
     (pSrc : System.Address;
      aSrcStep : access nppdefs_h.Npp32s;
      nRestartInterval : nppdefs_h.Npp32s;
      nSs : nppdefs_h.Npp32s;
      nSe : nppdefs_h.Npp32s;
      nAh : nppdefs_h.Npp32s;
      nAl : nppdefs_h.Npp32s;
      pDst : access nppdefs_h.Npp8u;
      pLength : access nppdefs_h.Npp32s;
      hpCodesDC : System.Address;
      hpTableDC : System.Address;
      hpCodesAC : System.Address;
      hpTableAC : System.Address;
      aHuffmanDCTable : System.Address;
      aHuffmanACTable : System.Address;
      oSizeROI : access nppdefs_h.NppiSize;
      pTempStorage : access nppdefs_h.Npp8u) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_compression_functions.h:581
   pragma Import (C, nppiEncodeOptimizeHuffmanScan_JPEG_8u16s_P3R, "nppiEncodeOptimizeHuffmanScan_JPEG_8u16s_P3R");

  --*
  -- * Calculates the size of the temporary buffer for baseline Huffman encoding.
  -- *
  -- * \see nppiEncodeHuffmanScan_JPEG_8u16s_P1R(), nppiEncodeHuffmanScan_JPEG_8u16s_P3R().
  -- * 
  -- * \param oSize Image Dimension.
  -- * \param pBufSize Pointer to variable that returns the size of the
  -- *        temporary buffer. 
  -- *
  -- * \return NPP_SUCCESS Indicates no error. Any other value indicates an error
  -- *         or a warning
  -- * \return NPP_SIZE_ERROR Indicates an error condition if any image dimension
  -- *         has zero or negative value
  -- * \return NPP_NULL_POINTER_ERROR Indicates an error condition if pBufSize
  -- *         pointer is NULL
  --  

   function nppiEncodeHuffmanGetSize
     (oSize : nppdefs_h.NppiSize;
      nChannels : int;
      pBufSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_compression_functions.h:607
   pragma Import (C, nppiEncodeHuffmanGetSize, "nppiEncodeHuffmanGetSize");

  --*
  -- * Calculates the size of the temporary buffer for optimize Huffman coding.
  -- *
  -- * See \ref nppiGenerateOptimizeHuffmanTable_JPEG.
  -- * 
  -- * \param oSize Image Dimension.
  -- * \param nChannels Number of channels in the image.
  -- * \param pBufSize Pointer to variable that returns the size of the
  -- *        temporary buffer. 
  -- *
  -- * \return NPP_SUCCESS Indicates no error. Any other value indicates an error
  -- *         or a warning
  -- * \return NPP_SIZE_ERROR Indicates an error condition if any image dimension
  -- *         has zero or negative value
  -- * \return NPP_NULL_POINTER_ERROR Indicates an error condition if pBufSize
  -- *         pointer is NULL
  --  

   function nppiEncodeOptimizeHuffmanGetSize
     (oSize : nppdefs_h.NppiSize;
      nChannels : int;
      pBufSize : access int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/nppi_compression_functions.h:626
   pragma Import (C, nppiEncodeOptimizeHuffmanGetSize, "nppiEncodeOptimizeHuffmanGetSize");

  --* @} image_compression  
  -- extern "C"  
end nppi_compression_functions_h;

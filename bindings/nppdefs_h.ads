pragma Ada_2005;
pragma Style_Checks (Off);

with Interfaces.C; use Interfaces.C;
with Interfaces.C.Extensions;
with stddef_h;

package nppdefs_h is

   --  unsupported macro: NPP_ALIGN_8 __align__(8)
   --  unsupported macro: NPP_ALIGN_16 __align__(16)
   NPP_MIN_8U : constant := ( 0 );  --  /usr/local/cuda-8.0/include/nppdefs.h:368
   NPP_MAX_8U : constant := ( 255 );  --  /usr/local/cuda-8.0/include/nppdefs.h:369
   NPP_MIN_16U : constant := ( 0 );  --  /usr/local/cuda-8.0/include/nppdefs.h:370
   NPP_MAX_16U : constant := ( 65535 );  --  /usr/local/cuda-8.0/include/nppdefs.h:371
   NPP_MIN_32U : constant := ( 0 );  --  /usr/local/cuda-8.0/include/nppdefs.h:372
   NPP_MAX_32U : constant := ( 4294967295 );  --  /usr/local/cuda-8.0/include/nppdefs.h:373
   NPP_MIN_64U : constant := ( 0 );  --  /usr/local/cuda-8.0/include/nppdefs.h:374
   NPP_MAX_64U : constant := ( 18446744073709551615 );  --  /usr/local/cuda-8.0/include/nppdefs.h:375

   NPP_MIN_8S : constant := (-127 - 1 );  --  /usr/local/cuda-8.0/include/nppdefs.h:377
   NPP_MAX_8S : constant := ( 127 );  --  /usr/local/cuda-8.0/include/nppdefs.h:378
   NPP_MIN_16S : constant := (-32767 - 1 );  --  /usr/local/cuda-8.0/include/nppdefs.h:379
   NPP_MAX_16S : constant := ( 32767 );  --  /usr/local/cuda-8.0/include/nppdefs.h:380
   NPP_MIN_32S : constant := (-2147483647 - 1 );  --  /usr/local/cuda-8.0/include/nppdefs.h:381
   NPP_MAX_32S : constant := ( 2147483647 );  --  /usr/local/cuda-8.0/include/nppdefs.h:382
   NPP_MAX_64S : constant := ( 9223372036854775807 );  --  /usr/local/cuda-8.0/include/nppdefs.h:383
   NPP_MIN_64S : constant := (-9223372036854775807 - 1);  --  /usr/local/cuda-8.0/include/nppdefs.h:384

   NPP_MINABS_32F : constant := ( 1.175494351e-38f );  --  /usr/local/cuda-8.0/include/nppdefs.h:386
   NPP_MAXABS_32F : constant := ( 3.402823466e+38f );  --  /usr/local/cuda-8.0/include/nppdefs.h:387
   NPP_MINABS_64F : constant := ( 2.2250738585072014e-308 );  --  /usr/local/cuda-8.0/include/nppdefs.h:388
   NPP_MAXABS_64F : constant := ( 1.7976931348623158e+308 );  --  /usr/local/cuda-8.0/include/nppdefs.h:389

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
  -- * \file nppdefs.h
  -- * Typedefinitions and macros for NPP library.
  --  

  -- If this is a 32-bit Windows compile, don't align to 16-byte at all
  -- and use a "union-trick" to create 8-byte alignment.
  -- On 32-bit Windows platforms, do not force 8-byte alignment.
  --   This is a consequence of a limitation of that platform.
  -- On 32-bit Windows platforms, do not force 16-byte alignment.
  --   This is a consequence of a limitation of that platform.
  --* \defgroup typedefs_npp NPP Type Definitions and Constants
  -- * @{
  --  

  --* 
  -- * Filtering methods.
  --  

  --*<  Nearest neighbor filtering.  
  --*<  Linear interpolation.  
  --*<  Cubic interpolation.  
  --*<  Two-parameter cubic filter (B=1, C=0)  
  --*<  Two-parameter cubic filter (B=0, C=1/2)  
  --*<  Two-parameter cubic filter (B=1/2, C=3/10)  
  --*<  Super sampling.  
  --*<  Lanczos filtering.  
  --*<  Generic Lanczos filtering with order 3.  
  --*<  Smooth edge filtering.  
   subtype NppiInterpolationMode is unsigned;
   NPPI_INTER_UNDEFINED : constant NppiInterpolationMode := 0;
   NPPI_INTER_NN : constant NppiInterpolationMode := 1;
   NPPI_INTER_LINEAR : constant NppiInterpolationMode := 2;
   NPPI_INTER_CUBIC : constant NppiInterpolationMode := 4;
   NPPI_INTER_CUBIC2P_BSPLINE : constant NppiInterpolationMode := 5;
   NPPI_INTER_CUBIC2P_CATMULLROM : constant NppiInterpolationMode := 6;
   NPPI_INTER_CUBIC2P_B05C03 : constant NppiInterpolationMode := 7;
   NPPI_INTER_SUPER : constant NppiInterpolationMode := 8;
   NPPI_INTER_LANCZOS : constant NppiInterpolationMode := 16;
   NPPI_INTER_LANCZOS3_ADVANCED : constant NppiInterpolationMode := 17;
   NPPI_SMOOTH_EDGE : constant NppiInterpolationMode := -2147483648;  -- /usr/local/cuda-8.0/include/nppdefs.h:103

  --* 
  -- * Bayer Grid Position Registration.
  --  

  --*<  Default registration position.  
   type NppiBayerGridPosition is 
     (NPPI_BAYER_BGGR,
      NPPI_BAYER_RGGB,
      NPPI_BAYER_GBRG,
      NPPI_BAYER_GRBG);
   pragma Convention (C, NppiBayerGridPosition);  -- /usr/local/cuda-8.0/include/nppdefs.h:114

  --*
  -- * Fixed filter-kernel sizes.
  --  

  -- leaving space for more 1 X N type enum values 
  -- leaving space for more N X 1 type enum values
   subtype NppiMaskSize is unsigned;
   NPP_MASK_SIZE_1_X_3 : constant NppiMaskSize := 0;
   NPP_MASK_SIZE_1_X_5 : constant NppiMaskSize := 1;
   NPP_MASK_SIZE_3_X_1 : constant NppiMaskSize := 100;
   NPP_MASK_SIZE_5_X_1 : constant NppiMaskSize := 101;
   NPP_MASK_SIZE_3_X_3 : constant NppiMaskSize := 200;
   NPP_MASK_SIZE_5_X_5 : constant NppiMaskSize := 201;
   NPP_MASK_SIZE_7_X_7 : constant NppiMaskSize := 400;
   NPP_MASK_SIZE_9_X_9 : constant NppiMaskSize := 500;
   NPP_MASK_SIZE_11_X_11 : constant NppiMaskSize := 600;
   NPP_MASK_SIZE_13_X_13 : constant NppiMaskSize := 700;
   NPP_MASK_SIZE_15_X_15 : constant NppiMaskSize := 800;  -- /usr/local/cuda-8.0/include/nppdefs.h:132

  --* 
  -- * Differential Filter types
  --  

   type NppiDifferentialKernel is 
     (NPP_FILTER_SOBEL,
      NPP_FILTER_SCHARR);
   pragma Convention (C, NppiDifferentialKernel);  -- /usr/local/cuda-8.0/include/nppdefs.h:142

  --*
  -- * Error Status Codes
  -- *
  -- * Almost all NPP function return error-status information using
  -- * these return codes.
  -- * Negative return codes indicate errors, positive return codes indicate
  -- * warnings, a return code of 0 indicates success.
  --  

  -- negative return-codes indicate errors  
  --*<  ZeroCrossing mode not supported   
  --*< Unsupported round mode 
  --*< Image pixels are constant for quality index  
  --*< One of the output image dimensions is less than 1 pixel  
  --*< Number overflows the upper or lower limit of the data type  
  --*< Step value is not pixel multiple  
  --*< Number of levels for histogram is less than 2  
  --*< Number of levels for LUT is less than 2  
  --*< Processed data is corrupted  
  --*< Wrong order of the destination channels  
  --*< All values of the mask are zero  
  --*< The quadrangle is nonconvex or degenerates into triangle, line or point  
  --*< Size of the rectangle region is less than or equal to 1  
  --*< Unallowable values of the transformation coefficients    
  --*< Bad or unsupported number of channels  
  --*< Channel of interest is not 1, 2, or 3  
  --*< Divisor is equal to zero  
  --*< Illegal channel index  
  --*< Stride is less than the row length  
  --*< Anchor point is outside mask  
  --*< Lower bound is larger than upper bound  
  --*<  Step is less or equal zero  
  -- success  
  --*<  Error free operation  
  --*<  Successful operation (same as NPP_NO_ERROR)  
  -- positive return-codes indicate warnings  
  --*<  Indicates that no operation was performed  
  --*<  Divisor is zero however does not terminate the execution  
  --*<  Indicates that the quadrangle passed to one of affine warping functions doesn't have necessary properties. First 3 vertices are used, the fourth vertex discarded.  
  --*<  The given ROI has no interestion with either the source or destination ROI. Thus no operation was performed.  
  --*<  The given quadrangle has no intersection with either the source or destination ROI. Thus no operation was performed.  
  --*<  Image size isn't multiple of two. Indicates that in case of 422/411/420 sampling the ROI width/height was modified for proper processing.  
  --*<  Speed reduction due to uncoalesced memory accesses warning.  
   subtype NppStatus is unsigned;
   NPP_NOT_SUPPORTED_MODE_ERROR : constant NppStatus := -9999;
   NPP_INVALID_HOST_POINTER_ERROR : constant NppStatus := -1032;
   NPP_INVALID_DEVICE_POINTER_ERROR : constant NppStatus := -1031;
   NPP_LUT_PALETTE_BITSIZE_ERROR : constant NppStatus := -1030;
   NPP_ZC_MODE_NOT_SUPPORTED_ERROR : constant NppStatus := -1028;
   NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY : constant NppStatus := -1027;
   NPP_TEXTURE_BIND_ERROR : constant NppStatus := -1024;
   NPP_WRONG_INTERSECTION_ROI_ERROR : constant NppStatus := -1020;
   NPP_HAAR_CLASSIFIER_PIXEL_MATCH_ERROR : constant NppStatus := -1006;
   NPP_MEMFREE_ERROR : constant NppStatus := -1005;
   NPP_MEMSET_ERROR : constant NppStatus := -1004;
   NPP_MEMCPY_ERROR : constant NppStatus := -1003;
   NPP_ALIGNMENT_ERROR : constant NppStatus := -1002;
   NPP_CUDA_KERNEL_EXECUTION_ERROR : constant NppStatus := -1000;
   NPP_ROUND_MODE_NOT_SUPPORTED_ERROR : constant NppStatus := -213;
   NPP_QUALITY_INDEX_ERROR : constant NppStatus := -210;
   NPP_RESIZE_NO_OPERATION_ERROR : constant NppStatus := -201;
   NPP_OVERFLOW_ERROR : constant NppStatus := -109;
   NPP_NOT_EVEN_STEP_ERROR : constant NppStatus := -108;
   NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR : constant NppStatus := -107;
   NPP_LUT_NUMBER_OF_LEVELS_ERROR : constant NppStatus := -106;
   NPP_CORRUPTED_DATA_ERROR : constant NppStatus := -61;
   NPP_CHANNEL_ORDER_ERROR : constant NppStatus := -60;
   NPP_ZERO_MASK_VALUE_ERROR : constant NppStatus := -59;
   NPP_QUADRANGLE_ERROR : constant NppStatus := -58;
   NPP_RECTANGLE_ERROR : constant NppStatus := -57;
   NPP_COEFFICIENT_ERROR : constant NppStatus := -56;
   NPP_NUMBER_OF_CHANNELS_ERROR : constant NppStatus := -53;
   NPP_COI_ERROR : constant NppStatus := -52;
   NPP_DIVISOR_ERROR : constant NppStatus := -51;
   NPP_CHANNEL_ERROR : constant NppStatus := -47;
   NPP_STRIDE_ERROR : constant NppStatus := -37;
   NPP_ANCHOR_ERROR : constant NppStatus := -34;
   NPP_MASK_SIZE_ERROR : constant NppStatus := -33;
   NPP_RESIZE_FACTOR_ERROR : constant NppStatus := -23;
   NPP_INTERPOLATION_ERROR : constant NppStatus := -22;
   NPP_MIRROR_FLIP_ERROR : constant NppStatus := -21;
   NPP_MOMENT_00_ZERO_ERROR : constant NppStatus := -20;
   NPP_THRESHOLD_NEGATIVE_LEVEL_ERROR : constant NppStatus := -19;
   NPP_THRESHOLD_ERROR : constant NppStatus := -18;
   NPP_CONTEXT_MATCH_ERROR : constant NppStatus := -17;
   NPP_FFT_FLAG_ERROR : constant NppStatus := -16;
   NPP_FFT_ORDER_ERROR : constant NppStatus := -15;
   NPP_STEP_ERROR : constant NppStatus := -14;
   NPP_SCALE_RANGE_ERROR : constant NppStatus := -13;
   NPP_DATA_TYPE_ERROR : constant NppStatus := -12;
   NPP_OUT_OFF_RANGE_ERROR : constant NppStatus := -11;
   NPP_DIVIDE_BY_ZERO_ERROR : constant NppStatus := -10;
   NPP_MEMORY_ALLOCATION_ERR : constant NppStatus := -9;
   NPP_NULL_POINTER_ERROR : constant NppStatus := -8;
   NPP_RANGE_ERROR : constant NppStatus := -7;
   NPP_SIZE_ERROR : constant NppStatus := -6;
   NPP_BAD_ARGUMENT_ERROR : constant NppStatus := -5;
   NPP_NO_MEMORY_ERROR : constant NppStatus := -4;
   NPP_NOT_IMPLEMENTED_ERROR : constant NppStatus := -3;
   NPP_ERROR : constant NppStatus := -2;
   NPP_ERROR_RESERVED : constant NppStatus := -1;
   NPP_NO_ERROR : constant NppStatus := 0;
   NPP_SUCCESS : constant NppStatus := 0;
   NPP_NO_OPERATION_WARNING : constant NppStatus := 1;
   NPP_DIVIDE_BY_ZERO_WARNING : constant NppStatus := 6;
   NPP_AFFINE_QUAD_INCORRECT_WARNING : constant NppStatus := 28;
   NPP_WRONG_INTERSECTION_ROI_WARNING : constant NppStatus := 29;
   NPP_WRONG_INTERSECTION_QUAD_WARNING : constant NppStatus := 30;
   NPP_DOUBLE_SIZE_WARNING : constant NppStatus := 35;
   NPP_MISALIGNED_DST_ROI_WARNING : constant NppStatus := 10000;  -- /usr/local/cuda-8.0/include/nppdefs.h:237

  --*<  Indicates that the compute-capability query failed  
  --*<  Indicates that no CUDA capable device was found  
  --*<  Indicates that CUDA 1.0 capable device is machine's default device  
  --*<  Indicates that CUDA 1.1 capable device is machine's default device  
  --*<  Indicates that CUDA 1.2 capable device is machine's default device  
  --*<  Indicates that CUDA 1.3 capable device is machine's default device  
  --*<  Indicates that CUDA 2.0 capable device is machine's default device  
  --*<  Indicates that CUDA 2.1 capable device is machine's default device  
  --*<  Indicates that CUDA 3.0 capable device is machine's default device  
  --*<  Indicates that CUDA 3.2 capable device is machine's default device  
  --*<  Indicates that CUDA 3.5 capable device is machine's default device  
  --*<  Indicates that CUDA 3.7 capable device is machine's default device  
  --*<  Indicates that CUDA 5.0 capable device is machine's default device  
  --*<  Indicates that CUDA 5.2 capable device is machine's default device  
  --*<  Indicates that CUDA 5.3 capable device is machine's default device  
  --*<  Indicates that CUDA 6.0 capable device is machine's default device  
  --*<  Indicates that CUDA 6.1 capable device is machine's default device  
  --*<  Indicates that CUDA 6.2 capable device is machine's default device  
  --*<  Indicates that CUDA 6.3 or better is machine's default device  
   subtype NppGpuComputeCapability is unsigned;
   NPP_CUDA_UNKNOWN_VERSION : constant NppGpuComputeCapability := -1;
   NPP_CUDA_NOT_CAPABLE : constant NppGpuComputeCapability := 0;
   NPP_CUDA_1_0 : constant NppGpuComputeCapability := 100;
   NPP_CUDA_1_1 : constant NppGpuComputeCapability := 110;
   NPP_CUDA_1_2 : constant NppGpuComputeCapability := 120;
   NPP_CUDA_1_3 : constant NppGpuComputeCapability := 130;
   NPP_CUDA_2_0 : constant NppGpuComputeCapability := 200;
   NPP_CUDA_2_1 : constant NppGpuComputeCapability := 210;
   NPP_CUDA_3_0 : constant NppGpuComputeCapability := 300;
   NPP_CUDA_3_2 : constant NppGpuComputeCapability := 320;
   NPP_CUDA_3_5 : constant NppGpuComputeCapability := 350;
   NPP_CUDA_3_7 : constant NppGpuComputeCapability := 370;
   NPP_CUDA_5_0 : constant NppGpuComputeCapability := 500;
   NPP_CUDA_5_2 : constant NppGpuComputeCapability := 520;
   NPP_CUDA_5_3 : constant NppGpuComputeCapability := 530;
   NPP_CUDA_6_0 : constant NppGpuComputeCapability := 600;
   NPP_CUDA_6_1 : constant NppGpuComputeCapability := 610;
   NPP_CUDA_6_2 : constant NppGpuComputeCapability := 620;
   NPP_CUDA_6_3 : constant NppGpuComputeCapability := 630;  -- /usr/local/cuda-8.0/include/nppdefs.h:260

  --*<  Major version number  
   type NppLibraryVersion is record
      major : aliased int;  -- /usr/local/cuda-8.0/include/nppdefs.h:264
      minor : aliased int;  -- /usr/local/cuda-8.0/include/nppdefs.h:265
      build : aliased int;  -- /usr/local/cuda-8.0/include/nppdefs.h:266
   end record;
   pragma Convention (C_Pass_By_Copy, NppLibraryVersion);  -- /usr/local/cuda-8.0/include/nppdefs.h:267

   --  skipped anonymous struct anon_25

  --*<  Minor version number  
  --*<  Build number. This reflects the nightly build this release was made from.  
  --* \defgroup npp_basic_types Basic NPP Data Types
  -- * @{
  --  

  --*<  8-bit unsigned chars  
   subtype Npp8u is unsigned_char;  -- /usr/local/cuda-8.0/include/nppdefs.h:274

  --*<  8-bit signed chars  
   subtype Npp8s is signed_char;  -- /usr/local/cuda-8.0/include/nppdefs.h:275

  --*<  16-bit unsigned integers  
   subtype Npp16u is unsigned_short;  -- /usr/local/cuda-8.0/include/nppdefs.h:276

  --*<  16-bit signed integers  
   subtype Npp16s is short;  -- /usr/local/cuda-8.0/include/nppdefs.h:277

  --*<  32-bit unsigned integers  
   subtype Npp32u is unsigned;  -- /usr/local/cuda-8.0/include/nppdefs.h:278

  --*<  32-bit signed integers  
   subtype Npp32s is int;  -- /usr/local/cuda-8.0/include/nppdefs.h:279

  --*<  64-bit unsigned integers  
   subtype Npp64u is Extensions.unsigned_long_long;  -- /usr/local/cuda-8.0/include/nppdefs.h:280

  --*<  64-bit signed integers  
   subtype Npp64s is Long_Long_Integer;  -- /usr/local/cuda-8.0/include/nppdefs.h:281

  --*<  32-bit (IEEE) floating-point numbers  
   subtype Npp32f is float;  -- /usr/local/cuda-8.0/include/nppdefs.h:282

  --*<  64-bit floating-point numbers  
   subtype Npp64f is double;  -- /usr/local/cuda-8.0/include/nppdefs.h:283

  --*
  -- * Complex Number
  -- * This struct represents an unsigned char complex number.
  --  

  --*<  Real part  
   type Npp8uc is record
      re : aliased Npp8u;  -- /usr/local/cuda-8.0/include/nppdefs.h:292
      im : aliased Npp8u;  -- /usr/local/cuda-8.0/include/nppdefs.h:293
   end record;
   pragma Convention (C_Pass_By_Copy, Npp8uc);  -- /usr/local/cuda-8.0/include/nppdefs.h:294

   --  skipped anonymous struct anon_26

  --*<  Imaginary part  
  --*
  -- * Complex Number
  -- * This struct represents an unsigned short complex number.
  --  

  --*<  Real part  
   type Npp16uc is record
      re : aliased Npp16u;  -- /usr/local/cuda-8.0/include/nppdefs.h:302
      im : aliased Npp16u;  -- /usr/local/cuda-8.0/include/nppdefs.h:303
   end record;
   pragma Convention (C_Pass_By_Copy, Npp16uc);  -- /usr/local/cuda-8.0/include/nppdefs.h:304

   --  skipped anonymous struct anon_27

  --*<  Imaginary part  
  --*
  -- * Complex Number
  -- * This struct represents a short complex number.
  --  

  --*<  Real part  
   type Npp16sc is record
      re : aliased Npp16s;  -- /usr/local/cuda-8.0/include/nppdefs.h:312
      im : aliased Npp16s;  -- /usr/local/cuda-8.0/include/nppdefs.h:313
   end record;
   pragma Convention (C_Pass_By_Copy, Npp16sc);  -- /usr/local/cuda-8.0/include/nppdefs.h:314

   --  skipped anonymous struct anon_28

  --*<  Imaginary part  
  --*
  -- * Complex Number
  -- * This struct represents an unsigned int complex number.
  --  

  --*<  Real part  
   type Npp32uc is record
      re : aliased Npp32u;  -- /usr/local/cuda-8.0/include/nppdefs.h:322
      im : aliased Npp32u;  -- /usr/local/cuda-8.0/include/nppdefs.h:323
   end record;
   pragma Convention (C_Pass_By_Copy, Npp32uc);  -- /usr/local/cuda-8.0/include/nppdefs.h:324

   --  skipped anonymous struct anon_29

  --*<  Imaginary part  
  --*
  -- * Complex Number
  -- * This struct represents a signed int complex number.
  --  

  --*<  Real part  
   type Npp32sc is record
      re : aliased Npp32s;  -- /usr/local/cuda-8.0/include/nppdefs.h:332
      im : aliased Npp32s;  -- /usr/local/cuda-8.0/include/nppdefs.h:333
   end record;
   pragma Convention (C_Pass_By_Copy, Npp32sc);  -- /usr/local/cuda-8.0/include/nppdefs.h:334

   --  skipped anonymous struct anon_30

  --*<  Imaginary part  
  --*
  -- * Complex Number
  -- * This struct represents a single floating-point complex number.
  --  

  --*<  Real part  
   type Npp32fc is record
      re : aliased Npp32f;  -- /usr/local/cuda-8.0/include/nppdefs.h:342
      im : aliased Npp32f;  -- /usr/local/cuda-8.0/include/nppdefs.h:343
   end record;
   pragma Convention (C_Pass_By_Copy, Npp32fc);  -- /usr/local/cuda-8.0/include/nppdefs.h:344

   --  skipped anonymous struct anon_31

  --*<  Imaginary part  
  --*
  -- * Complex Number
  -- * This struct represents a long long complex number.
  --  

  --*<  Real part  
   type Npp64sc is record
      re : aliased Npp64s;  -- /usr/local/cuda-8.0/include/nppdefs.h:352
      im : aliased Npp64s;  -- /usr/local/cuda-8.0/include/nppdefs.h:353
   end record;
   pragma Convention (C_Pass_By_Copy, Npp64sc);  -- /usr/local/cuda-8.0/include/nppdefs.h:354

   --  skipped anonymous struct anon_32

  --*<  Imaginary part  
  --*
  -- * Complex Number
  -- * This struct represents a double floating-point complex number.
  --  

  --*<  Real part  
   type Npp64fc is record
      re : aliased Npp64f;  -- /usr/local/cuda-8.0/include/nppdefs.h:362
      im : aliased Npp64f;  -- /usr/local/cuda-8.0/include/nppdefs.h:363
   end record;
   pragma Convention (C_Pass_By_Copy, Npp64fc);  -- /usr/local/cuda-8.0/include/nppdefs.h:364

   --  skipped anonymous struct anon_33

  --*<  Imaginary part  
  --@} 
  --* 
  -- * 2D Point
  --  

  --*<  x-coordinate.  
   type NppiPoint is record
      x : aliased int;  -- /usr/local/cuda-8.0/include/nppdefs.h:397
      y : aliased int;  -- /usr/local/cuda-8.0/include/nppdefs.h:398
   end record;
   pragma Convention (C_Pass_By_Copy, NppiPoint);  -- /usr/local/cuda-8.0/include/nppdefs.h:399

   --  skipped anonymous struct anon_34

  --*<  y-coordinate.  
  --*
  -- * 2D Size
  -- * This struct typically represents the size of a a rectangular region in
  -- * two space.
  --  

  --*<  Rectangle width.  
   type NppiSize is record
      width : aliased int;  -- /usr/local/cuda-8.0/include/nppdefs.h:408
      height : aliased int;  -- /usr/local/cuda-8.0/include/nppdefs.h:409
   end record;
   pragma Convention (C_Pass_By_Copy, NppiSize);  -- /usr/local/cuda-8.0/include/nppdefs.h:410

   --  skipped anonymous struct anon_35

  --*<  Rectangle height.  
  --*
  -- * 2D Rectangle
  -- * This struct contains position and size information of a rectangle in 
  -- * two space.
  -- * The rectangle's position is usually signified by the coordinate of its
  -- * upper-left corner.
  --  

  --*<  x-coordinate of upper left corner (lowest memory address).  
   type NppiRect is record
      x : aliased int;  -- /usr/local/cuda-8.0/include/nppdefs.h:421
      y : aliased int;  -- /usr/local/cuda-8.0/include/nppdefs.h:422
      width : aliased int;  -- /usr/local/cuda-8.0/include/nppdefs.h:423
      height : aliased int;  -- /usr/local/cuda-8.0/include/nppdefs.h:424
   end record;
   pragma Convention (C_Pass_By_Copy, NppiRect);  -- /usr/local/cuda-8.0/include/nppdefs.h:425

   --  skipped anonymous struct anon_36

  --*<  y-coordinate of upper left corner (lowest memory address).  
  --*<  Rectangle width.  
  --*<  Rectangle height.  
   type NppiAxis is 
     (NPP_HORIZONTAL_AXIS,
      NPP_VERTICAL_AXIS,
      NPP_BOTH_AXIS);
   pragma Convention (C, NppiAxis);  -- /usr/local/cuda-8.0/include/nppdefs.h:432

   type NppCmpOp is 
     (NPP_CMP_LESS,
      NPP_CMP_LESS_EQ,
      NPP_CMP_EQ,
      NPP_CMP_GREATER_EQ,
      NPP_CMP_GREATER);
   pragma Convention (C, NppCmpOp);  -- /usr/local/cuda-8.0/include/nppdefs.h:441

  --*
  -- * Rounding Modes
  -- *
  -- * The enumerated rounding modes are used by a large number of NPP primitives
  -- * to allow the user to specify the method by which fractional values are converted
  -- * to integer values. Also see \ref rounding_modes.
  -- *
  -- * For NPP release 5.5 new names for the three rounding modes are introduced that are
  -- * based on the naming conventions for rounding modes set forth in the IEEE-754
  -- * floating-point standard. Developers are encouraged to use the new, longer names
  -- * to be future proof as the legacy names will be deprecated in subsequent NPP releases.
  -- *
  --  

  --* 
  --     * Round to the nearest even integer.
  --     * All fractional numbers are rounded to their nearest integer. The ambiguous
  --     * cases (i.e. \<integer\>.5) are rounded to the closest even integer.
  --     * E.g.
  --     * - roundNear(0.5) = 0
  --     * - roundNear(0.6) = 1
  --     * - roundNear(1.5) = 2
  --     * - roundNear(-1.5) = -2
  --      

  --/< Alias name for ::NPP_RND_NEAR.
  --* 
  --     * Round according to financial rule.
  --     * All fractional numbers are rounded to their nearest integer. The ambiguous
  --     * cases (i.e. \<integer\>.5) are rounded away from zero.
  --     * E.g.
  --     * - roundFinancial(0.4)  = 0
  --     * - roundFinancial(0.5)  = 1
  --     * - roundFinancial(-1.5) = -2
  --      

  --/< Alias name for ::NPP_RND_FINANCIAL. 
  --*
  --     * Round towards zero (truncation). 
  --     * All fractional numbers of the form \<integer\>.\<decimals\> are truncated to
  --     * \<integer\>.
  --     * - roundZero(1.5) = 1
  --     * - roundZero(1.9) = 1
  --     * - roundZero(-2.5) = -2
  --      

  --/< Alias name for ::NPP_RND_ZERO. 
  --     * Other rounding modes supported by IEEE-754 (2008) floating-point standard:
  --     *
  --     * - NPP_ROUND_TOWARD_INFINITY // ceiling
  --     * - NPP_ROUND_TOWARD_NEGATIVE_INFINITY // floor
  --     *
  --      

   subtype NppRoundMode is unsigned;
   NPP_RND_NEAR : constant NppRoundMode := 0;
   NPP_ROUND_NEAREST_TIES_TO_EVEN : constant NppRoundMode := 0;
   NPP_RND_FINANCIAL : constant NppRoundMode := 1;
   NPP_ROUND_NEAREST_TIES_AWAY_FROM_ZERO : constant NppRoundMode := 1;
   NPP_RND_ZERO : constant NppRoundMode := 2;
   NPP_ROUND_TOWARD_ZERO : constant NppRoundMode := 2;  -- /usr/local/cuda-8.0/include/nppdefs.h:499

   subtype NppiBorderType is unsigned;
   NPP_BORDER_UNDEFINED : constant NppiBorderType := 0;
   NPP_BORDER_NONE : constant NppiBorderType := 0;
   NPP_BORDER_CONSTANT : constant NppiBorderType := 1;
   NPP_BORDER_REPLICATE : constant NppiBorderType := 2;
   NPP_BORDER_WRAP : constant NppiBorderType := 3;
   NPP_BORDER_MIRROR : constant NppiBorderType := 4;  -- /usr/local/cuda-8.0/include/nppdefs.h:509

   type NppHintAlgorithm is 
     (NPP_ALG_HINT_NONE,
      NPP_ALG_HINT_FAST,
      NPP_ALG_HINT_ACCURATE);
   pragma Convention (C, NppHintAlgorithm);  -- /usr/local/cuda-8.0/include/nppdefs.h:516

  -- Alpha composition controls  
   type NppiAlphaOp is 
     (NPPI_OP_ALPHA_OVER,
      NPPI_OP_ALPHA_IN,
      NPPI_OP_ALPHA_OUT,
      NPPI_OP_ALPHA_ATOP,
      NPPI_OP_ALPHA_XOR,
      NPPI_OP_ALPHA_PLUS,
      NPPI_OP_ALPHA_OVER_PREMUL,
      NPPI_OP_ALPHA_IN_PREMUL,
      NPPI_OP_ALPHA_OUT_PREMUL,
      NPPI_OP_ALPHA_ATOP_PREMUL,
      NPPI_OP_ALPHA_XOR_PREMUL,
      NPPI_OP_ALPHA_PLUS_PREMUL,
      NPPI_OP_ALPHA_PREMUL);
   pragma Convention (C, NppiAlphaOp);  -- /usr/local/cuda-8.0/include/nppdefs.h:534

  --*<  number of classifiers  
   type NppiHaarClassifier_32f is record
      numClassifiers : aliased int;  -- /usr/local/cuda-8.0/include/nppdefs.h:540
      classifiers : access Npp32s;  -- /usr/local/cuda-8.0/include/nppdefs.h:541
      classifierStep : aliased stddef_h.size_t;  -- /usr/local/cuda-8.0/include/nppdefs.h:542
      classifierSize : aliased NppiSize;  -- /usr/local/cuda-8.0/include/nppdefs.h:543
      counterDevice : access Npp32s;  -- /usr/local/cuda-8.0/include/nppdefs.h:544
   end record;
   pragma Convention (C_Pass_By_Copy, NppiHaarClassifier_32f);  -- /usr/local/cuda-8.0/include/nppdefs.h:545

   --  skipped anonymous struct anon_43

  --*<  packed classifier data 40 bytes each  
  --*<  size of the buffer  
   type NppiHaarBuffer is record
      haarBufferSize : aliased int;  -- /usr/local/cuda-8.0/include/nppdefs.h:549
      haarBuffer : access Npp32s;  -- /usr/local/cuda-8.0/include/nppdefs.h:550
   end record;
   pragma Convention (C_Pass_By_Copy, NppiHaarBuffer);  -- /usr/local/cuda-8.0/include/nppdefs.h:552

   --  skipped anonymous struct anon_44

  --*<  buffer  
  --*<  sign change  
  --*<  sign change XOR  
  --*<  sign change count_0  
   type NppsZCType is 
     (nppZCR,
      nppZCXor,
      nppZCC);
   pragma Convention (C, NppsZCType);  -- /usr/local/cuda-8.0/include/nppdefs.h:558

  --*<  DC Table  
  --*<  AC Table  
   type NppiHuffmanTableType is 
     (nppiDCTable,
      nppiACTable);
   pragma Convention (C, NppiHuffmanTableType);  -- /usr/local/cuda-8.0/include/nppdefs.h:563

  --*<  maximum  
  --*<  sum  
  --*<  square root of sum of squares  
   type NppiNorm is 
     (nppiNormInf,
      nppiNormL1,
      nppiNormL2);
   pragma Convention (C, NppiNorm);  -- /usr/local/cuda-8.0/include/nppdefs.h:569

  -- extern "C"  
  --@} 
end nppdefs_h;

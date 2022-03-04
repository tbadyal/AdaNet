pragma Ada_2005;
pragma Style_Checks (Off);

with Interfaces.C; use Interfaces.C;
with nppdefs_h;

package npps_initialization_h is

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
  -- * \file npps_initialization.h
  -- * NPP Signal Processing Functionality.
  --  

  --* @defgroup signal_initialization Initialization
  -- * @ingroup npps
  -- *
  -- * @{
  --  

  --* \defgroup signal_set Set
  -- *
  -- * @{
  -- *
  --  

  --* @name Set 
  -- * Set methods for 1D vectors of various types. The copy methods operate on vector data given
  -- * as a pointer to the underlying data-type (e.g. 8-bit vectors would
  -- * be passed as pointers to Npp8u type) and length of the vectors, i.e. the number of items.
  -- *
  -- * @{
  -- *
  --  

  --* 
  -- * 8-bit unsigned char, vector set method.
  -- * \param nValue Value used to initialize the vector pDst.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSet_8u
     (nValue : nppdefs_h.Npp8u;
      pDst : access nppdefs_h.Npp8u;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_initialization.h:93
   pragma Import (C, nppsSet_8u, "nppsSet_8u");

  --* 
  -- * 8-bit signed char, vector set method.
  -- * \param nValue Value used to initialize the vector pDst.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSet_8s
     (nValue : nppdefs_h.Npp8s;
      pDst : access nppdefs_h.Npp8s;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_initialization.h:103
   pragma Import (C, nppsSet_8s, "nppsSet_8s");

  --* 
  -- * 16-bit unsigned integer, vector set method.
  -- * \param nValue Value used to initialize the vector pDst.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSet_16u
     (nValue : nppdefs_h.Npp16u;
      pDst : access nppdefs_h.Npp16u;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_initialization.h:113
   pragma Import (C, nppsSet_16u, "nppsSet_16u");

  --* 
  -- * 16-bit signed integer, vector set method.
  -- * \param nValue Value used to initialize the vector pDst.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSet_16s
     (nValue : nppdefs_h.Npp16s;
      pDst : access nppdefs_h.Npp16s;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_initialization.h:123
   pragma Import (C, nppsSet_16s, "nppsSet_16s");

  --* 
  -- * 16-bit integer complex, vector set method.
  -- * \param nValue Value used to initialize the vector pDst.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSet_16sc
     (nValue : nppdefs_h.Npp16sc;
      pDst : access nppdefs_h.Npp16sc;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_initialization.h:133
   pragma Import (C, nppsSet_16sc, "nppsSet_16sc");

  --* 
  -- * 32-bit unsigned integer, vector set method.
  -- * \param nValue Value used to initialize the vector pDst.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSet_32u
     (nValue : nppdefs_h.Npp32u;
      pDst : access nppdefs_h.Npp32u;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_initialization.h:143
   pragma Import (C, nppsSet_32u, "nppsSet_32u");

  --* 
  -- * 32-bit signed integer, vector set method.
  -- * \param nValue Value used to initialize the vector pDst.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSet_32s
     (nValue : nppdefs_h.Npp32s;
      pDst : access nppdefs_h.Npp32s;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_initialization.h:153
   pragma Import (C, nppsSet_32s, "nppsSet_32s");

  --* 
  -- * 32-bit integer complex, vector set method.
  -- * \param nValue Value used to initialize the vector pDst.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSet_32sc
     (nValue : nppdefs_h.Npp32sc;
      pDst : access nppdefs_h.Npp32sc;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_initialization.h:163
   pragma Import (C, nppsSet_32sc, "nppsSet_32sc");

  --* 
  -- * 32-bit float, vector set method.
  -- * \param nValue Value used to initialize the vector pDst.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSet_32f
     (nValue : nppdefs_h.Npp32f;
      pDst : access nppdefs_h.Npp32f;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_initialization.h:173
   pragma Import (C, nppsSet_32f, "nppsSet_32f");

  --* 
  -- * 32-bit float complex, vector set method.
  -- * \param nValue Value used to initialize the vector pDst.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSet_32fc
     (nValue : nppdefs_h.Npp32fc;
      pDst : access nppdefs_h.Npp32fc;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_initialization.h:183
   pragma Import (C, nppsSet_32fc, "nppsSet_32fc");

  --* 
  -- * 64-bit long long integer, vector set method.
  -- * \param nValue Value used to initialize the vector pDst.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSet_64s
     (nValue : nppdefs_h.Npp64s;
      pDst : access nppdefs_h.Npp64s;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_initialization.h:193
   pragma Import (C, nppsSet_64s, "nppsSet_64s");

  --* 
  -- * 64-bit long long integer complex, vector set method.
  -- * \param nValue Value used to initialize the vector pDst.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSet_64sc
     (nValue : nppdefs_h.Npp64sc;
      pDst : access nppdefs_h.Npp64sc;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_initialization.h:203
   pragma Import (C, nppsSet_64sc, "nppsSet_64sc");

  --* 
  -- * 64-bit double, vector set method.
  -- * \param nValue Value used to initialize the vector pDst.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSet_64f
     (nValue : nppdefs_h.Npp64f;
      pDst : access nppdefs_h.Npp64f;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_initialization.h:213
   pragma Import (C, nppsSet_64f, "nppsSet_64f");

  --* 
  -- * 64-bit double complex, vector set method.
  -- * \param nValue Value used to initialize the vector pDst.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsSet_64fc
     (nValue : nppdefs_h.Npp64fc;
      pDst : access nppdefs_h.Npp64fc;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_initialization.h:223
   pragma Import (C, nppsSet_64fc, "nppsSet_64fc");

  --* @} end of Signal Set  
  --* @} signal_set  
  --* \defgroup signal_zero Zero
  -- *
  -- * @{
  -- *
  --  

  --* @name Zero
  -- * Set signals to zero.
  -- *
  -- * @{
  -- *
  --  

  --* 
  -- * 8-bit unsigned char, vector zero method.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsZero_8u (pDst : access nppdefs_h.Npp8u; nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_initialization.h:248
   pragma Import (C, nppsZero_8u, "nppsZero_8u");

  --* 
  -- * 16-bit integer, vector zero method.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsZero_16s (pDst : access nppdefs_h.Npp16s; nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_initialization.h:257
   pragma Import (C, nppsZero_16s, "nppsZero_16s");

  --* 
  -- * 16-bit integer complex, vector zero method.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsZero_16sc (pDst : access nppdefs_h.Npp16sc; nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_initialization.h:266
   pragma Import (C, nppsZero_16sc, "nppsZero_16sc");

  --* 
  -- * 32-bit integer, vector zero method.
  --  * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsZero_32s (pDst : access nppdefs_h.Npp32s; nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_initialization.h:275
   pragma Import (C, nppsZero_32s, "nppsZero_32s");

  --* 
  -- * 32-bit integer complex, vector zero method.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsZero_32sc (pDst : access nppdefs_h.Npp32sc; nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_initialization.h:284
   pragma Import (C, nppsZero_32sc, "nppsZero_32sc");

  --* 
  -- * 32-bit float, vector zero method.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsZero_32f (pDst : access nppdefs_h.Npp32f; nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_initialization.h:293
   pragma Import (C, nppsZero_32f, "nppsZero_32f");

  --* 
  -- * 32-bit float complex, vector zero method.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsZero_32fc (pDst : access nppdefs_h.Npp32fc; nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_initialization.h:302
   pragma Import (C, nppsZero_32fc, "nppsZero_32fc");

  --* 
  -- * 64-bit long long integer, vector zero method.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsZero_64s (pDst : access nppdefs_h.Npp64s; nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_initialization.h:311
   pragma Import (C, nppsZero_64s, "nppsZero_64s");

  --* 
  -- * 64-bit long long integer complex, vector zero method.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsZero_64sc (pDst : access nppdefs_h.Npp64sc; nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_initialization.h:320
   pragma Import (C, nppsZero_64sc, "nppsZero_64sc");

  --* 
  -- * 64-bit double, vector zero method.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsZero_64f (pDst : access nppdefs_h.Npp64f; nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_initialization.h:329
   pragma Import (C, nppsZero_64f, "nppsZero_64f");

  --* 
  -- * 64-bit double complex, vector zero method.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsZero_64fc (pDst : access nppdefs_h.Npp64fc; nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_initialization.h:338
   pragma Import (C, nppsZero_64fc, "nppsZero_64fc");

  --* @} end of Zero  
  --* @} signal_zero  
  --* \defgroup signal_copy Copy
  -- *
  -- * @{
  -- *
  --  

  --* @name Copy
  -- * Copy methods for various type signals. Copy methods operate on
  -- * signal data given as a pointer to the underlying data-type (e.g. 8-bit
  -- * vectors would be passed as pointers to Npp8u type) and length of the
  -- * vectors, i.e. the number of items. 
  -- *
  -- * @{
  -- *
  --  

  --* 
  -- * 8-bit unsigned char, vector copy method
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsCopy_8u
     (pSrc : access nppdefs_h.Npp8u;
      pDst : access nppdefs_h.Npp8u;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_initialization.h:368
   pragma Import (C, nppsCopy_8u, "nppsCopy_8u");

  --* 
  -- * 16-bit signed short, vector copy method.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsCopy_16s
     (pSrc : access nppdefs_h.Npp16s;
      pDst : access nppdefs_h.Npp16s;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_initialization.h:378
   pragma Import (C, nppsCopy_16s, "nppsCopy_16s");

  --* 
  -- * 32-bit signed integer, vector copy method.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsCopy_32s
     (pSrc : access nppdefs_h.Npp32s;
      pDst : access nppdefs_h.Npp32s;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_initialization.h:388
   pragma Import (C, nppsCopy_32s, "nppsCopy_32s");

  --* 
  -- * 32-bit float, vector copy method.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsCopy_32f
     (pSrc : access nppdefs_h.Npp32f;
      pDst : access nppdefs_h.Npp32f;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_initialization.h:398
   pragma Import (C, nppsCopy_32f, "nppsCopy_32f");

  --* 
  -- * 64-bit signed integer, vector copy method.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsCopy_64s
     (pSrc : access nppdefs_h.Npp64s;
      pDst : access nppdefs_h.Npp64s;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_initialization.h:408
   pragma Import (C, nppsCopy_64s, "nppsCopy_64s");

  --* 
  -- * 16-bit complex short, vector copy method.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsCopy_16sc
     (pSrc : access constant nppdefs_h.Npp16sc;
      pDst : access nppdefs_h.Npp16sc;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_initialization.h:418
   pragma Import (C, nppsCopy_16sc, "nppsCopy_16sc");

  --* 
  -- * 32-bit complex signed integer, vector copy method.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsCopy_32sc
     (pSrc : access constant nppdefs_h.Npp32sc;
      pDst : access nppdefs_h.Npp32sc;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_initialization.h:428
   pragma Import (C, nppsCopy_32sc, "nppsCopy_32sc");

  --* 
  -- * 32-bit complex float, vector copy method.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsCopy_32fc
     (pSrc : access constant nppdefs_h.Npp32fc;
      pDst : access nppdefs_h.Npp32fc;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_initialization.h:438
   pragma Import (C, nppsCopy_32fc, "nppsCopy_32fc");

  --* 
  -- * 64-bit complex signed integer, vector copy method.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsCopy_64sc
     (pSrc : access constant nppdefs_h.Npp64sc;
      pDst : access nppdefs_h.Npp64sc;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_initialization.h:448
   pragma Import (C, nppsCopy_64sc, "nppsCopy_64sc");

  --* 
  -- * 64-bit complex double, vector copy method.
  -- * \param pSrc \ref source_signal_pointer.
  -- * \param pDst \ref destination_signal_pointer.
  -- * \param nLength \ref length_specification.
  -- * \return \ref signal_data_error_codes, \ref length_error_codes.
  --  

   function nppsCopy_64fc
     (pSrc : access constant nppdefs_h.Npp64fc;
      pDst : access nppdefs_h.Npp64fc;
      nLength : int) return nppdefs_h.NppStatus;  -- /usr/local/cuda-8.0/include/npps_initialization.h:458
   pragma Import (C, nppsCopy_64fc, "nppsCopy_64fc");

  --* @} end of Copy  
  --* @} signal_copy  
  --* @} signal_initialization  
  -- extern "C"  
end npps_initialization_h;

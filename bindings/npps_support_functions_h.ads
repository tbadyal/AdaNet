pragma Ada_2005;
pragma Style_Checks (Off);

with Interfaces.C; use Interfaces.C;
with nppdefs_h;
with System;

package npps_support_functions_h is

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
  -- * \file npps_support_functions.h
  -- * Signal Processing Support Functions.
  --  

  --* @defgroup signal_memory_management Memory Management
  -- *  @ingroup npps
  -- *
  -- * @{
  --  

  --* @defgroup signal_malloc Malloc
  -- * Signal-allocator methods for allocating 1D arrays of data in device memory.
  -- * All allocators have size parameters to specify the size of the signal (1D array)
  -- * being allocated.
  -- *
  -- * The allocator methods return a pointer to the newly allocated memory of appropriate
  -- * type. If device-memory allocation is not possible due to resource constaints
  -- * the allocators return 0 (i.e. NULL pointer). 
  -- *
  -- * All signal allocators allocate memory aligned such that it is  beneficial to the 
  -- * performance of the majority of the signal-processing primitives. 
  -- * It is no mandatory however to use these allocators. Any valid
  -- * CUDA device-memory pointers can be passed to NPP primitives. 
  -- *
  -- * @{
  --  

  --*
  -- * 8-bit unsigned signal allocator.
  -- * \param nSize Number of unsigned chars in the new signal.
  -- * \return A pointer to the new signal. 0 (NULL-pointer) indicates
  -- *         that an error occurred during allocation.
  --  

   function nppsMalloc_8u (nSize : int) return access nppdefs_h.Npp8u;  -- /usr/local/cuda-8.0/include/npps_support_functions.h:94
   pragma Import (C, nppsMalloc_8u, "nppsMalloc_8u");

  --*
  -- * 8-bit signed signal allocator.
  -- * \param nSize Number of (signed) chars in the new signal.
  -- * \return A pointer to the new signal. 0 (NULL-pointer) indicates
  -- *         that an error occurred during allocation.
  --  

   function nppsMalloc_8s (nSize : int) return access nppdefs_h.Npp8s;  -- /usr/local/cuda-8.0/include/npps_support_functions.h:103
   pragma Import (C, nppsMalloc_8s, "nppsMalloc_8s");

  --*
  -- * 16-bit unsigned signal allocator.
  -- * \param nSize Number of unsigned shorts in the new signal.
  -- * \return A pointer to the new signal. 0 (NULL-pointer) indicates
  -- *         that an error occurred during allocation.
  --  

   function nppsMalloc_16u (nSize : int) return access nppdefs_h.Npp16u;  -- /usr/local/cuda-8.0/include/npps_support_functions.h:112
   pragma Import (C, nppsMalloc_16u, "nppsMalloc_16u");

  --*
  -- * 16-bit signal allocator.
  -- * \param nSize Number of shorts in the new signal.
  -- * \return A pointer to the new signal. 0 (NULL-pointer) indicates
  -- *         that an error occurred during allocation.
  --  

   function nppsMalloc_16s (nSize : int) return access nppdefs_h.Npp16s;  -- /usr/local/cuda-8.0/include/npps_support_functions.h:121
   pragma Import (C, nppsMalloc_16s, "nppsMalloc_16s");

  --*
  -- * 16-bit complex-value signal allocator.
  -- * \param nSize Number of 16-bit complex numbers in the new signal.
  -- * \return A pointer to the new signal. 0 (NULL-pointer) indicates
  -- *         that an error occurred during allocation.
  --  

   function nppsMalloc_16sc (nSize : int) return access nppdefs_h.Npp16sc;  -- /usr/local/cuda-8.0/include/npps_support_functions.h:130
   pragma Import (C, nppsMalloc_16sc, "nppsMalloc_16sc");

  --*
  -- * 32-bit unsigned signal allocator.
  -- * \param nSize Number of unsigned ints in the new signal.
  -- * \return A pointer to the new signal. 0 (NULL-pointer) indicates
  -- *         that an error occurred during allocation.
  --  

   function nppsMalloc_32u (nSize : int) return access nppdefs_h.Npp32u;  -- /usr/local/cuda-8.0/include/npps_support_functions.h:139
   pragma Import (C, nppsMalloc_32u, "nppsMalloc_32u");

  --*
  -- * 32-bit integer signal allocator.
  -- * \param nSize Number of ints in the new signal.
  -- * \return A pointer to the new signal. 0 (NULL-pointer) indicates
  -- *         that an error occurred during allocation.
  --  

   function nppsMalloc_32s (nSize : int) return access nppdefs_h.Npp32s;  -- /usr/local/cuda-8.0/include/npps_support_functions.h:148
   pragma Import (C, nppsMalloc_32s, "nppsMalloc_32s");

  --*
  -- * 32-bit complex integer signal allocator.
  -- * \param nSize Number of complex integner values in the new signal.
  -- * \return A pointer to the new signal. 0 (NULL-pointer) indicates
  -- *         that an error occurred during allocation.
  --  

   function nppsMalloc_32sc (nSize : int) return access nppdefs_h.Npp32sc;  -- /usr/local/cuda-8.0/include/npps_support_functions.h:157
   pragma Import (C, nppsMalloc_32sc, "nppsMalloc_32sc");

  --*
  -- * 32-bit float signal allocator.
  -- * \param nSize Number of floats in the new signal.
  -- * \return A pointer to the new signal. 0 (NULL-pointer) indicates
  -- *         that an error occurred during allocation.
  --  

   function nppsMalloc_32f (nSize : int) return access nppdefs_h.Npp32f;  -- /usr/local/cuda-8.0/include/npps_support_functions.h:166
   pragma Import (C, nppsMalloc_32f, "nppsMalloc_32f");

  --*
  -- * 32-bit complex float signal allocator.
  -- * \param nSize Number of complex float values in the new signal.
  -- * \return A pointer to the new signal. 0 (NULL-pointer) indicates
  -- *         that an error occurred during allocation.
  --  

   function nppsMalloc_32fc (nSize : int) return access nppdefs_h.Npp32fc;  -- /usr/local/cuda-8.0/include/npps_support_functions.h:175
   pragma Import (C, nppsMalloc_32fc, "nppsMalloc_32fc");

  --*
  -- * 64-bit long integer signal allocator.
  -- * \param nSize Number of long ints in the new signal.
  -- * \return A pointer to the new signal. 0 (NULL-pointer) indicates
  -- *         that an error occurred during allocation.
  --  

   function nppsMalloc_64s (nSize : int) return access nppdefs_h.Npp64s;  -- /usr/local/cuda-8.0/include/npps_support_functions.h:184
   pragma Import (C, nppsMalloc_64s, "nppsMalloc_64s");

  --*
  -- * 64-bit complex long integer signal allocator.
  -- * \param nSize Number of complex long int values in the new signal.
  -- * \return A pointer to the new signal. 0 (NULL-pointer) indicates
  -- *         that an error occurred during allocation.
  --  

   function nppsMalloc_64sc (nSize : int) return access nppdefs_h.Npp64sc;  -- /usr/local/cuda-8.0/include/npps_support_functions.h:193
   pragma Import (C, nppsMalloc_64sc, "nppsMalloc_64sc");

  --*
  -- * 64-bit float (double) signal allocator.
  -- * \param nSize Number of doubles in the new signal.
  -- * \return A pointer to the new signal. 0 (NULL-pointer) indicates
  -- *         that an error occurred during allocation.
  --  

   function nppsMalloc_64f (nSize : int) return access nppdefs_h.Npp64f;  -- /usr/local/cuda-8.0/include/npps_support_functions.h:202
   pragma Import (C, nppsMalloc_64f, "nppsMalloc_64f");

  --*
  -- * 64-bit complex complex signal allocator.
  -- * \param nSize Number of complex double valuess in the new signal.
  -- * \return A pointer to the new signal. 0 (NULL-pointer) indicates
  -- *         that an error occurred during allocation.
  --  

   function nppsMalloc_64fc (nSize : int) return access nppdefs_h.Npp64fc;  -- /usr/local/cuda-8.0/include/npps_support_functions.h:211
   pragma Import (C, nppsMalloc_64fc, "nppsMalloc_64fc");

  --* @} signal_malloc  
  --* @defgroup signal_free Free
  -- * Free  signal memory.
  -- *
  -- * @{
  --  

  --*
  -- * Free method for any signal memory.
  -- * \param pValues A pointer to memory allocated using nppiMalloc_<modifier>.
  --  

   procedure nppsFree (pValues : System.Address);  -- /usr/local/cuda-8.0/include/npps_support_functions.h:225
   pragma Import (C, nppsFree, "nppsFree");

  --* @} signal_free  
  --* end of Memory management functions
  -- * 
  -- * @}
  -- *
  --  

  -- extern "C"  
end npps_support_functions_h;

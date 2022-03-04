pragma Ada_2005;
pragma Style_Checks (Off);

with Interfaces.C; use Interfaces.C;
with Interfaces.C.Strings;
with driver_types_h;

package nvToolsExtCudaRt_h is

   NVTX_RESOURCE_CLASS_CUDART : constant := 5;  --  /usr/local/cuda-8.0/include/nvToolsExtCudaRt.h:67
   --  unsupported macro: nvtxNameCudaDevice nvtxNameCudaDeviceA
   --  unsupported macro: nvtxNameCudaStream nvtxNameCudaStreamA
   --  unsupported macro: nvtxNameCudaEvent nvtxNameCudaEventA

  --* Copyright 2009-2016  NVIDIA Corporation.  All rights reserved.
  --*
  --* NOTICE TO USER:
  --*
  --* This source code is subject to NVIDIA ownership rights under U.S. and
  --* international Copyright laws.
  --*
  --* This software and the information contained herein is PROPRIETARY and
  --* CONFIDENTIAL to NVIDIA and is being provided under the terms and conditions
  --* of a form of NVIDIA software license agreement.
  --*
  --* NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
  --* CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
  --* IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
  --* REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
  --* MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
  --* IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
  --* OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
  --* OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
  --* OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
  --* OR PERFORMANCE OF THIS SOURCE CODE.
  --*
  --* U.S. Government End Users.   This source code is a "commercial item" as
  --* that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
  --* "commercial computer  software"  and "commercial computer software
  --* documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
  --* and is provided to the U.S. Government only as a commercial end item.
  --* Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
  --* 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
  --* source code with only those rights set forth herein.
  --*
  --* Any use of this source code in individual and commercial software must
  --* include, in the user documentation and internal comments to the code,
  --* the above Disclaimer and U.S. Government End Users Notice.
  -- 

  -- =========================================================================  
  --* \name Functions for CUDA Resource Naming
  -- 

  --* \addtogroup RESOURCE_NAMING
  -- * \section RESOURCE_NAMING_CUDART CUDA Runtime Resource Naming
  -- *
  -- * This section covers the API functions that allow to annotate CUDA resources
  -- * with user-provided names.
  -- *
  -- * @{
  --  

  --  -------------------------------------------------------------------------  
  -- \cond SHOW_HIDDEN 
  --* \brief Used to build a non-colliding value for resource types separated class
  --* \version \NVTX_VERSION_2
  -- 

  --* \endcond  
  --  -------------------------------------------------------------------------  
  --* \brief Resource types for CUDART
  -- 

   subtype nvtxResourceCUDARTType_t is unsigned;
   NVTX_RESOURCE_TYPE_CUDART_DEVICE : constant nvtxResourceCUDARTType_t := 327680;
   NVTX_RESOURCE_TYPE_CUDART_STREAM : constant nvtxResourceCUDARTType_t := 327681;
   NVTX_RESOURCE_TYPE_CUDART_EVENT : constant nvtxResourceCUDARTType_t := 327682;  -- /usr/local/cuda-8.0/include/nvToolsExtCudaRt.h:73

  -- int device  
  -- cudaStream_t  
  -- cudaEvent_t  
  -- -------------------------------------------------------------------------  
  --* \brief Annotates a CUDA device.
  -- *
  -- * Allows the user to associate a CUDA device with a user-provided name.
  -- *
  -- * \param device - The id of the CUDA device to name.
  -- * \param name   - The name of the CUDA device.
  -- *
  -- * \version \NVTX_VERSION_1
  -- * @{  

   procedure nvtxNameCudaDeviceA (device : int; name : Interfaces.C.Strings.chars_ptr);  -- /usr/local/cuda-8.0/include/nvToolsExtCudaRt.h:91
   pragma Import (C, nvtxNameCudaDeviceA, "nvtxNameCudaDeviceA");

   procedure nvtxNameCudaDeviceW (device : int; name : access wchar_t);  -- /usr/local/cuda-8.0/include/nvToolsExtCudaRt.h:92
   pragma Import (C, nvtxNameCudaDeviceW, "nvtxNameCudaDeviceW");

  --* @}  
  -- -------------------------------------------------------------------------  
  --* \brief Annotates a CUDA stream.
  -- *
  -- * Allows the user to associate a CUDA stream with a user-provided name.
  -- *
  -- * \param stream - The handle of the CUDA stream to name.
  -- * \param name   - The name of the CUDA stream.
  -- *
  -- * \version \NVTX_VERSION_1
  -- * @{  

   procedure nvtxNameCudaStreamA (stream : driver_types_h.cudaStream_t; name : Interfaces.C.Strings.chars_ptr);  -- /usr/local/cuda-8.0/include/nvToolsExtCudaRt.h:105
   pragma Import (C, nvtxNameCudaStreamA, "nvtxNameCudaStreamA");

   procedure nvtxNameCudaStreamW (stream : driver_types_h.cudaStream_t; name : access wchar_t);  -- /usr/local/cuda-8.0/include/nvToolsExtCudaRt.h:106
   pragma Import (C, nvtxNameCudaStreamW, "nvtxNameCudaStreamW");

  --* @}  
  -- -------------------------------------------------------------------------  
  --* \brief Annotates a CUDA event.
  -- *
  -- * Allows the user to associate a CUDA event with a user-provided name.
  -- *
  -- * \param event - The handle of the CUDA event to name.
  -- * \param name  - The name of the CUDA event.
  -- *
  -- * \version \NVTX_VERSION_1
  -- * @{  

   procedure nvtxNameCudaEventA (event : driver_types_h.cudaEvent_t; name : Interfaces.C.Strings.chars_ptr);  -- /usr/local/cuda-8.0/include/nvToolsExtCudaRt.h:119
   pragma Import (C, nvtxNameCudaEventA, "nvtxNameCudaEventA");

   procedure nvtxNameCudaEventW (event : driver_types_h.cudaEvent_t; name : access wchar_t);  -- /usr/local/cuda-8.0/include/nvToolsExtCudaRt.h:120
   pragma Import (C, nvtxNameCudaEventW, "nvtxNameCudaEventW");

  --* @}  
  --* @}  
  -- END RESOURCE_NAMING  
  -- =========================================================================  
end nvToolsExtCudaRt_h;

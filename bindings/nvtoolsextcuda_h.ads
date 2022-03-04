pragma Ada_2005;
pragma Style_Checks (Off);

with Interfaces.C; use Interfaces.C;
with cuda_h;
with Interfaces.C.Strings;

package nvToolsExtCuda_h is

   NVTX_RESOURCE_CLASS_CUDA : constant := 4;  --  /usr/local/cuda-8.0/include/nvToolsExtCuda.h:66
   --  unsupported macro: nvtxNameCuDevice nvtxNameCuDeviceA
   --  unsupported macro: nvtxNameCuContext nvtxNameCuContextA
   --  unsupported macro: nvtxNameCuStream nvtxNameCuStreamA
   --  unsupported macro: nvtxNameCuEvent nvtxNameCuEventA

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
  -- * \section RESOURCE_NAMING_CUDA CUDA Resource Naming
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
  --* \brief Resource types for CUDA
  -- 

   subtype nvtxResourceCUDAType_t is unsigned;
   NVTX_RESOURCE_TYPE_CUDA_DEVICE : constant nvtxResourceCUDAType_t := 262145;
   NVTX_RESOURCE_TYPE_CUDA_CONTEXT : constant nvtxResourceCUDAType_t := 262146;
   NVTX_RESOURCE_TYPE_CUDA_STREAM : constant nvtxResourceCUDAType_t := 262147;
   NVTX_RESOURCE_TYPE_CUDA_EVENT : constant nvtxResourceCUDAType_t := 262148;  -- /usr/local/cuda-8.0/include/nvToolsExtCuda.h:72

  -- CUdevice  
  -- CUcontext  
  -- CUstream  
  -- CUevent  
  -- -------------------------------------------------------------------------  
  --* \brief Annotates a CUDA device.
  -- *
  -- * Allows the user to associate a CUDA device with a user-provided name.
  -- *
  -- * \param device - The handle of the CUDA device to name.
  -- * \param name   - The name of the CUDA device.
  -- *
  -- * \version \NVTX_VERSION_1
  -- * @{  

   procedure nvtxNameCuDeviceA (device : cuda_h.CUdevice; name : Interfaces.C.Strings.chars_ptr);  -- /usr/local/cuda-8.0/include/nvToolsExtCuda.h:91
   pragma Import (C, nvtxNameCuDeviceA, "nvtxNameCuDeviceA");

   procedure nvtxNameCuDeviceW (device : cuda_h.CUdevice; name : access wchar_t);  -- /usr/local/cuda-8.0/include/nvToolsExtCuda.h:92
   pragma Import (C, nvtxNameCuDeviceW, "nvtxNameCuDeviceW");

  --* @}  
  -- -------------------------------------------------------------------------  
  --* \brief Annotates a CUDA context.
  -- *
  -- * Allows the user to associate a CUDA context with a user-provided name.
  -- *
  -- * \param context - The handle of the CUDA context to name.
  -- * \param name    - The name of the CUDA context.
  -- *
  -- * \par Example:
  -- * \code
  -- * CUresult status = cuCtxCreate( &cuContext, 0, cuDevice );
  -- * if ( CUDA_SUCCESS != status )
  -- *     goto Error;
  -- * nvtxNameCuContext(cuContext, "CTX_NAME");
  -- * \endcode
  -- *
  -- * \version \NVTX_VERSION_1
  -- * @{  

   procedure nvtxNameCuContextA (context : cuda_h.CUcontext; name : Interfaces.C.Strings.chars_ptr);  -- /usr/local/cuda-8.0/include/nvToolsExtCuda.h:113
   pragma Import (C, nvtxNameCuContextA, "nvtxNameCuContextA");

   procedure nvtxNameCuContextW (context : cuda_h.CUcontext; name : access wchar_t);  -- /usr/local/cuda-8.0/include/nvToolsExtCuda.h:114
   pragma Import (C, nvtxNameCuContextW, "nvtxNameCuContextW");

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

   procedure nvtxNameCuStreamA (stream : cuda_h.CUstream; name : Interfaces.C.Strings.chars_ptr);  -- /usr/local/cuda-8.0/include/nvToolsExtCuda.h:127
   pragma Import (C, nvtxNameCuStreamA, "nvtxNameCuStreamA");

   procedure nvtxNameCuStreamW (stream : cuda_h.CUstream; name : access wchar_t);  -- /usr/local/cuda-8.0/include/nvToolsExtCuda.h:128
   pragma Import (C, nvtxNameCuStreamW, "nvtxNameCuStreamW");

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

   procedure nvtxNameCuEventA (event : cuda_h.CUevent; name : Interfaces.C.Strings.chars_ptr);  -- /usr/local/cuda-8.0/include/nvToolsExtCuda.h:141
   pragma Import (C, nvtxNameCuEventA, "nvtxNameCuEventA");

   procedure nvtxNameCuEventW (event : cuda_h.CUevent; name : access wchar_t);  -- /usr/local/cuda-8.0/include/nvToolsExtCuda.h:142
   pragma Import (C, nvtxNameCuEventW, "nvtxNameCuEventW");

  --* @}  
  --* @}  
  -- END RESOURCE_NAMING  
  -- =========================================================================  
end nvToolsExtCuda_h;

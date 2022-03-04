pragma Ada_2005;
pragma Style_Checks (Off);

with Interfaces.C; use Interfaces.C;
with driver_types_h;
with Interfaces.C.Extensions;

package surface_types_h is

   cudaSurfaceType1D : constant := 16#01#;  --  /usr/local/cuda-8.0/include/surface_types.h:73
   cudaSurfaceType2D : constant := 16#02#;  --  /usr/local/cuda-8.0/include/surface_types.h:74
   cudaSurfaceType3D : constant := 16#03#;  --  /usr/local/cuda-8.0/include/surface_types.h:75
   cudaSurfaceTypeCubemap : constant := 16#0C#;  --  /usr/local/cuda-8.0/include/surface_types.h:76
   cudaSurfaceType1DLayered : constant := 16#F1#;  --  /usr/local/cuda-8.0/include/surface_types.h:77
   cudaSurfaceType2DLayered : constant := 16#F2#;  --  /usr/local/cuda-8.0/include/surface_types.h:78
   cudaSurfaceTypeCubemapLayered : constant := 16#FC#;  --  /usr/local/cuda-8.0/include/surface_types.h:79

  -- * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
  -- *
  -- * NOTICE TO LICENSEE:
  -- *
  -- * This source code and/or documentation ("Licensed Deliverables") are
  -- * subject to NVIDIA intellectual property rights under U.S. and
  -- * international Copyright laws.
  -- *
  -- * These Licensed Deliverables contained herein is PROPRIETARY and
  -- * CONFIDENTIAL to NVIDIA and is being provided under the terms and
  -- * conditions of a form of NVIDIA software license agreement by and
  -- * between NVIDIA and Licensee ("License Agreement") or electronically
  -- * accepted by Licensee.  Notwithstanding any terms or conditions to
  -- * the contrary in the License Agreement, reproduction or disclosure
  -- * of the Licensed Deliverables to any third party without the express
  -- * written consent of NVIDIA is prohibited.
  -- *
  -- * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
  -- * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
  -- * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
  -- * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
  -- * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
  -- * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
  -- * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
  -- * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
  -- * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
  -- * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
  -- * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
  -- * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
  -- * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
  -- * OF THESE LICENSED DELIVERABLES.
  -- *
  -- * U.S. Government End Users.  These Licensed Deliverables are a
  -- * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
  -- * 1995), consisting of "commercial computer software" and "commercial
  -- * computer software documentation" as such terms are used in 48
  -- * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
  -- * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
  -- * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
  -- * U.S. Government End Users acquire the Licensed Deliverables with
  -- * only those rights set forth herein.
  -- *
  -- * Any use of the Licensed Deliverables in individual and commercial
  -- * software must include, in the user documentation and internal
  -- * comments to the code, the above Disclaimer and U.S. Government End
  -- * Users Notice.
  --  

  --******************************************************************************
  --*                                                                              *
  --*                                                                              *
  --*                                                                              *
  --****************************************************************************** 

  --*
  -- * \addtogroup CUDART_TYPES
  -- *
  -- * @{
  --  

  --******************************************************************************
  --*                                                                              *
  --*                                                                              *
  --*                                                                              *
  --****************************************************************************** 

  --*
  -- * CUDA Surface boundary modes
  --  

   type cudaSurfaceBoundaryMode is 
     (cudaBoundaryModeZero,
      cudaBoundaryModeClamp,
      cudaBoundaryModeTrap);
   pragma Convention (C, cudaSurfaceBoundaryMode);  -- /usr/local/cuda-8.0/include/surface_types.h:84

  --*< Zero boundary mode  
  --*< Clamp boundary mode  
  --*< Trap boundary mode  
  --*
  -- * CUDA Surface format modes
  --  

   type cudaSurfaceFormatMode is 
     (cudaFormatModeForced,
      cudaFormatModeAuto);
   pragma Convention (C, cudaSurfaceFormatMode);  -- /usr/local/cuda-8.0/include/surface_types.h:94

  --*< Forced format mode  
  --*< Auto format mode  
  --*
  -- * CUDA Surface reference
  --  

  --*
  --     * Channel descriptor for surface reference
  --      

   type surfaceReference is record
      channelDesc : aliased driver_types_h.cudaChannelFormatDesc;  -- /usr/local/cuda-8.0/include/surface_types.h:108
   end record;
   pragma Convention (C_Pass_By_Copy, surfaceReference);  -- /usr/local/cuda-8.0/include/surface_types.h:103

  --*
  -- * An opaque value that represents a CUDA Surface object
  --  

   subtype cudaSurfaceObject_t is Extensions.unsigned_long_long;  -- /usr/local/cuda-8.0/include/surface_types.h:114

  --* @}  
  --* @}  
  -- END CUDART_TYPES  
end surface_types_h;

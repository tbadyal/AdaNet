pragma Ada_2005;
pragma Style_Checks (Off);

with Interfaces.C; use Interfaces.C;
with driver_types_h;
with Interfaces.C.Extensions;

package texture_types_h is

   cudaTextureType1D : constant := 16#01#;  --  /usr/local/cuda-8.0/include/texture_types.h:73
   cudaTextureType2D : constant := 16#02#;  --  /usr/local/cuda-8.0/include/texture_types.h:74
   cudaTextureType3D : constant := 16#03#;  --  /usr/local/cuda-8.0/include/texture_types.h:75
   cudaTextureTypeCubemap : constant := 16#0C#;  --  /usr/local/cuda-8.0/include/texture_types.h:76
   cudaTextureType1DLayered : constant := 16#F1#;  --  /usr/local/cuda-8.0/include/texture_types.h:77
   cudaTextureType2DLayered : constant := 16#F2#;  --  /usr/local/cuda-8.0/include/texture_types.h:78
   cudaTextureTypeCubemapLayered : constant := 16#FC#;  --  /usr/local/cuda-8.0/include/texture_types.h:79

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
  -- * CUDA texture address modes
  --  

   type cudaTextureAddressMode is 
     (cudaAddressModeWrap,
      cudaAddressModeClamp,
      cudaAddressModeMirror,
      cudaAddressModeBorder);
   pragma Convention (C, cudaTextureAddressMode);  -- /usr/local/cuda-8.0/include/texture_types.h:84

  --*< Wrapping address mode  
  --*< Clamp to edge address mode  
  --*< Mirror address mode  
  --*< Border address mode  
  --*
  -- * CUDA texture filter modes
  --  

   type cudaTextureFilterMode is 
     (cudaFilterModePoint,
      cudaFilterModeLinear);
   pragma Convention (C, cudaTextureFilterMode);  -- /usr/local/cuda-8.0/include/texture_types.h:95

  --*< Point filter mode  
  --*< Linear filter mode  
  --*
  -- * CUDA texture read modes
  --  

   type cudaTextureReadMode is 
     (cudaReadModeElementType,
      cudaReadModeNormalizedFloat);
   pragma Convention (C, cudaTextureReadMode);  -- /usr/local/cuda-8.0/include/texture_types.h:104

  --*< Read texture as specified element type  
  --*< Read texture as normalized float  
  --*
  -- * CUDA texture reference
  --  

  --*
  --     * Indicates whether texture reads are normalized or not
  --      

   type textureReference_addressMode_array is array (0 .. 2) of aliased cudaTextureAddressMode;
   type textureReference_uu_cudaReserved_array is array (0 .. 14) of aliased int;
   type textureReference is record
      normalized : aliased int;  -- /usr/local/cuda-8.0/include/texture_types.h:118
      filterMode : aliased cudaTextureFilterMode;  -- /usr/local/cuda-8.0/include/texture_types.h:122
      addressMode : aliased textureReference_addressMode_array;  -- /usr/local/cuda-8.0/include/texture_types.h:126
      channelDesc : aliased driver_types_h.cudaChannelFormatDesc;  -- /usr/local/cuda-8.0/include/texture_types.h:130
      sRGB : aliased int;  -- /usr/local/cuda-8.0/include/texture_types.h:134
      maxAnisotropy : aliased unsigned;  -- /usr/local/cuda-8.0/include/texture_types.h:138
      mipmapFilterMode : aliased cudaTextureFilterMode;  -- /usr/local/cuda-8.0/include/texture_types.h:142
      mipmapLevelBias : aliased float;  -- /usr/local/cuda-8.0/include/texture_types.h:146
      minMipmapLevelClamp : aliased float;  -- /usr/local/cuda-8.0/include/texture_types.h:150
      maxMipmapLevelClamp : aliased float;  -- /usr/local/cuda-8.0/include/texture_types.h:154
      uu_cudaReserved : aliased textureReference_uu_cudaReserved_array;  -- /usr/local/cuda-8.0/include/texture_types.h:155
   end record;
   pragma Convention (C_Pass_By_Copy, textureReference);  -- /usr/local/cuda-8.0/include/texture_types.h:113

  --*
  --     * Texture filter mode
  --      

  --*
  --     * Texture address mode for up to 3 dimensions
  --      

  --*
  --     * Channel descriptor for the texture reference
  --      

  --*
  --     * Perform sRGB->linear conversion during texture read
  --      

  --*
  --     * Limit to the anisotropy ratio
  --      

  --*
  --     * Mipmap filter mode
  --      

  --*
  --     * Offset applied to the supplied mipmap level
  --      

  --*
  --     * Lower end of the mipmap level range to clamp access to
  --      

  --*
  --     * Upper end of the mipmap level range to clamp access to
  --      

  --*
  -- * CUDA texture descriptor
  --  

  --*
  --     * Texture address mode for up to 3 dimensions
  --      

   type cudaTextureDesc_addressMode_array is array (0 .. 2) of aliased cudaTextureAddressMode;
   type cudaTextureDesc_borderColor_array is array (0 .. 3) of aliased float;
   type cudaTextureDesc is record
      addressMode : aliased cudaTextureDesc_addressMode_array;  -- /usr/local/cuda-8.0/include/texture_types.h:166
      filterMode : aliased cudaTextureFilterMode;  -- /usr/local/cuda-8.0/include/texture_types.h:170
      readMode : aliased cudaTextureReadMode;  -- /usr/local/cuda-8.0/include/texture_types.h:174
      sRGB : aliased int;  -- /usr/local/cuda-8.0/include/texture_types.h:178
      borderColor : aliased cudaTextureDesc_borderColor_array;  -- /usr/local/cuda-8.0/include/texture_types.h:182
      normalizedCoords : aliased int;  -- /usr/local/cuda-8.0/include/texture_types.h:186
      maxAnisotropy : aliased unsigned;  -- /usr/local/cuda-8.0/include/texture_types.h:190
      mipmapFilterMode : aliased cudaTextureFilterMode;  -- /usr/local/cuda-8.0/include/texture_types.h:194
      mipmapLevelBias : aliased float;  -- /usr/local/cuda-8.0/include/texture_types.h:198
      minMipmapLevelClamp : aliased float;  -- /usr/local/cuda-8.0/include/texture_types.h:202
      maxMipmapLevelClamp : aliased float;  -- /usr/local/cuda-8.0/include/texture_types.h:206
   end record;
   pragma Convention (C_Pass_By_Copy, cudaTextureDesc);  -- /usr/local/cuda-8.0/include/texture_types.h:161

  --*
  --     * Texture filter mode
  --      

  --*
  --     * Texture read mode
  --      

  --*
  --     * Perform sRGB->linear conversion during texture read
  --      

  --*
  --     * Texture Border Color
  --      

  --*
  --     * Indicates whether texture reads are normalized or not
  --      

  --*
  --     * Limit to the anisotropy ratio
  --      

  --*
  --     * Mipmap filter mode
  --      

  --*
  --     * Offset applied to the supplied mipmap level
  --      

  --*
  --     * Lower end of the mipmap level range to clamp access to
  --      

  --*
  --     * Upper end of the mipmap level range to clamp access to
  --      

  --*
  -- * An opaque value that represents a CUDA texture object
  --  

   subtype cudaTextureObject_t is Extensions.unsigned_long_long;  -- /usr/local/cuda-8.0/include/texture_types.h:212

  --* @}  
  --* @}  
  -- END CUDART_TYPES  
end texture_types_h;

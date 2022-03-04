pragma Ada_2005;
pragma Style_Checks (Off);

with Interfaces.C; use Interfaces.C;
with driver_types_h;

package channel_descriptor_h is

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

  --******************************************************************************
  --*                                                                              *
  --*                                                                              *
  --*                                                                              *
  --****************************************************************************** 

  --*
  -- * \addtogroup CUDART_HIGHLEVEL
  -- *
  -- * @{
  --  

  --*
  -- * \brief \hl Returns a channel descriptor using the specified format
  -- *
  -- * Returns a channel descriptor with format \p f and number of bits of each
  -- * component \p x, \p y, \p z, and \p w.  The ::cudaChannelFormatDesc is
  -- * defined as:
  -- * \code
  --  struct cudaChannelFormatDesc {
  --    int x, y, z, w;
  --    enum cudaChannelFormatKind f;
  --  };
  -- * \endcode
  -- *
  -- * where ::cudaChannelFormatKind is one of ::cudaChannelFormatKindSigned,
  -- * ::cudaChannelFormatKindUnsigned, or ::cudaChannelFormatKindFloat.
  -- *
  -- * \return
  -- * Channel descriptor with format \p f
  -- *
  -- * \sa \ref ::cudaCreateChannelDesc(int,int,int,int,cudaChannelFormatKind) "cudaCreateChannelDesc (Low level)",
  -- * ::cudaGetChannelDesc, ::cudaGetTextureReference,
  -- * \ref ::cudaBindTexture(size_t*, const struct texture< T, dim, readMode>&, const void*, const struct cudaChannelFormatDesc&, size_t) "cudaBindTexture (High level)",
  -- * \ref ::cudaBindTexture(size_t*, const struct texture< T, dim, readMode>&, const void*, size_t) "cudaBindTexture (High level, inherited channel descriptor)",
  -- * \ref ::cudaBindTexture2D(size_t*, const struct texture< T, dim, readMode>&, const void*, const struct cudaChannelFormatDesc&, size_t, size_t, size_t) "cudaBindTexture2D (High level)",
  -- * \ref ::cudaBindTextureToArray(const struct texture< T, dim, readMode>&, cudaArray_const_t, const struct cudaChannelFormatDesc&) "cudaBindTextureToArray (High level)",
  -- * \ref ::cudaBindTextureToArray(const struct texture< T, dim, readMode>&, cudaArray_const_t) "cudaBindTextureToArray (High level, inherited channel descriptor)",
  -- * \ref ::cudaUnbindTexture(const struct texture< T, dim, readMode>&) "cudaUnbindTexture (High level)",
  -- * \ref ::cudaGetTextureAlignmentOffset(size_t*, const struct texture< T, dim, readMode>&) "cudaGetTextureAlignmentOffset (High level)"
  --  

   function cudaCreateChannelDescHalf return driver_types_h.cudaChannelFormatDesc;  -- /usr/local/cuda-8.0/include/channel_descriptor.h:112
   pragma Import (CPP, cudaCreateChannelDescHalf, "_ZL25cudaCreateChannelDescHalfv");

   function cudaCreateChannelDescHalf1 return driver_types_h.cudaChannelFormatDesc;  -- /usr/local/cuda-8.0/include/channel_descriptor.h:119
   pragma Import (CPP, cudaCreateChannelDescHalf1, "_ZL26cudaCreateChannelDescHalf1v");

   function cudaCreateChannelDescHalf2 return driver_types_h.cudaChannelFormatDesc;  -- /usr/local/cuda-8.0/include/channel_descriptor.h:126
   pragma Import (CPP, cudaCreateChannelDescHalf2, "_ZL26cudaCreateChannelDescHalf2v");

   function cudaCreateChannelDescHalf4 return driver_types_h.cudaChannelFormatDesc;  -- /usr/local/cuda-8.0/include/channel_descriptor.h:133
   pragma Import (CPP, cudaCreateChannelDescHalf4, "_ZL26cudaCreateChannelDescHalf4v");

  --* @}  
  --* @}  
  -- END CUDART_TEXTURE_HL  
end channel_descriptor_h;

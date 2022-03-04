pragma Ada_2005;
pragma Style_Checks (Off);

with Interfaces.C; use Interfaces.C;
with stdint_h;
with Interfaces.C.Extensions;
with System;
with stddef_h;
with Interfaces.C.Strings;

package cuda_h is

   --  unsupported macro: cuDeviceTotalMem cuDeviceTotalMem_v2
   --  unsupported macro: cuCtxCreate cuCtxCreate_v2
   --  unsupported macro: cuModuleGetGlobal cuModuleGetGlobal_v2
   --  unsupported macro: cuMemGetInfo cuMemGetInfo_v2
   --  unsupported macro: cuMemAlloc cuMemAlloc_v2
   --  unsupported macro: cuMemAllocPitch cuMemAllocPitch_v2
   --  unsupported macro: cuMemFree cuMemFree_v2
   --  unsupported macro: cuMemGetAddressRange cuMemGetAddressRange_v2
   --  unsupported macro: cuMemAllocHost cuMemAllocHost_v2
   --  unsupported macro: cuMemHostGetDevicePointer cuMemHostGetDevicePointer_v2
   --  unsupported macro: cuMemcpyHtoD __CUDA_API_PTDS(cuMemcpyHtoD_v2)
   --  unsupported macro: cuMemcpyDtoH __CUDA_API_PTDS(cuMemcpyDtoH_v2)
   --  unsupported macro: cuMemcpyDtoD __CUDA_API_PTDS(cuMemcpyDtoD_v2)
   --  unsupported macro: cuMemcpyDtoA __CUDA_API_PTDS(cuMemcpyDtoA_v2)
   --  unsupported macro: cuMemcpyAtoD __CUDA_API_PTDS(cuMemcpyAtoD_v2)
   --  unsupported macro: cuMemcpyHtoA __CUDA_API_PTDS(cuMemcpyHtoA_v2)
   --  unsupported macro: cuMemcpyAtoH __CUDA_API_PTDS(cuMemcpyAtoH_v2)
   --  unsupported macro: cuMemcpyAtoA __CUDA_API_PTDS(cuMemcpyAtoA_v2)
   --  unsupported macro: cuMemcpyHtoAAsync __CUDA_API_PTSZ(cuMemcpyHtoAAsync_v2)
   --  unsupported macro: cuMemcpyAtoHAsync __CUDA_API_PTSZ(cuMemcpyAtoHAsync_v2)
   --  unsupported macro: cuMemcpy2D __CUDA_API_PTDS(cuMemcpy2D_v2)
   --  unsupported macro: cuMemcpy2DUnaligned __CUDA_API_PTDS(cuMemcpy2DUnaligned_v2)
   --  unsupported macro: cuMemcpy3D __CUDA_API_PTDS(cuMemcpy3D_v2)
   --  unsupported macro: cuMemcpyHtoDAsync __CUDA_API_PTSZ(cuMemcpyHtoDAsync_v2)
   --  unsupported macro: cuMemcpyDtoHAsync __CUDA_API_PTSZ(cuMemcpyDtoHAsync_v2)
   --  unsupported macro: cuMemcpyDtoDAsync __CUDA_API_PTSZ(cuMemcpyDtoDAsync_v2)
   --  unsupported macro: cuMemcpy2DAsync __CUDA_API_PTSZ(cuMemcpy2DAsync_v2)
   --  unsupported macro: cuMemcpy3DAsync __CUDA_API_PTSZ(cuMemcpy3DAsync_v2)
   --  unsupported macro: cuMemsetD8 __CUDA_API_PTDS(cuMemsetD8_v2)
   --  unsupported macro: cuMemsetD16 __CUDA_API_PTDS(cuMemsetD16_v2)
   --  unsupported macro: cuMemsetD32 __CUDA_API_PTDS(cuMemsetD32_v2)
   --  unsupported macro: cuMemsetD2D8 __CUDA_API_PTDS(cuMemsetD2D8_v2)
   --  unsupported macro: cuMemsetD2D16 __CUDA_API_PTDS(cuMemsetD2D16_v2)
   --  unsupported macro: cuMemsetD2D32 __CUDA_API_PTDS(cuMemsetD2D32_v2)
   --  unsupported macro: cuArrayCreate cuArrayCreate_v2
   --  unsupported macro: cuArrayGetDescriptor cuArrayGetDescriptor_v2
   --  unsupported macro: cuArray3DCreate cuArray3DCreate_v2
   --  unsupported macro: cuArray3DGetDescriptor cuArray3DGetDescriptor_v2
   --  unsupported macro: cuTexRefSetAddress cuTexRefSetAddress_v2
   --  unsupported macro: cuTexRefGetAddress cuTexRefGetAddress_v2
   --  unsupported macro: cuGraphicsResourceGetMappedPointer cuGraphicsResourceGetMappedPointer_v2
   --  unsupported macro: cuCtxDestroy cuCtxDestroy_v2
   --  unsupported macro: cuCtxPopCurrent cuCtxPopCurrent_v2
   --  unsupported macro: cuCtxPushCurrent cuCtxPushCurrent_v2
   --  unsupported macro: cuStreamDestroy cuStreamDestroy_v2
   --  unsupported macro: cuEventDestroy cuEventDestroy_v2
   --  unsupported macro: cuTexRefSetAddress2D cuTexRefSetAddress2D_v3
   --  unsupported macro: cuLinkCreate cuLinkCreate_v2
   --  unsupported macro: cuLinkAddData cuLinkAddData_v2
   --  unsupported macro: cuLinkAddFile cuLinkAddFile_v2
   --  unsupported macro: cuMemHostRegister cuMemHostRegister_v2
   --  unsupported macro: cuGraphicsResourceSetMapFlags cuGraphicsResourceSetMapFlags_v2
   CUDA_VERSION : constant := 8000;  --  /usr/local/cuda-8.0/include/cuda.h:208

   CU_IPC_HANDLE_SIZE : constant := 64;  --  /usr/local/cuda-8.0/include/cuda.h:252
   --  unsupported macro: CU_STREAM_LEGACY ((CUstream)0x1)
   --  unsupported macro: CU_STREAM_PER_THREAD ((CUstream)0x2)

   CU_MEMHOSTALLOC_PORTABLE : constant := 16#01#;  --  /usr/local/cuda-8.0/include/cuda.h:1426

   CU_MEMHOSTALLOC_DEVICEMAP : constant := 16#02#;  --  /usr/local/cuda-8.0/include/cuda.h:1433

   CU_MEMHOSTALLOC_WRITECOMBINED : constant := 16#04#;  --  /usr/local/cuda-8.0/include/cuda.h:1441

   CU_MEMHOSTREGISTER_PORTABLE : constant := 16#01#;  --  /usr/local/cuda-8.0/include/cuda.h:1447

   CU_MEMHOSTREGISTER_DEVICEMAP : constant := 16#02#;  --  /usr/local/cuda-8.0/include/cuda.h:1454

   CU_MEMHOSTREGISTER_IOMEMORY : constant := 16#04#;  --  /usr/local/cuda-8.0/include/cuda.h:1468

   CUDA_ARRAY3D_LAYERED : constant := 16#01#;  --  /usr/local/cuda-8.0/include/cuda.h:1719

   CUDA_ARRAY3D_2DARRAY : constant := 16#01#;  --  /usr/local/cuda-8.0/include/cuda.h:1724

   CUDA_ARRAY3D_SURFACE_LDST : constant := 16#02#;  --  /usr/local/cuda-8.0/include/cuda.h:1730

   CUDA_ARRAY3D_CUBEMAP : constant := 16#04#;  --  /usr/local/cuda-8.0/include/cuda.h:1738

   CUDA_ARRAY3D_TEXTURE_GATHER : constant := 16#08#;  --  /usr/local/cuda-8.0/include/cuda.h:1744

   CUDA_ARRAY3D_DEPTH_TEXTURE : constant := 16#10#;  --  /usr/local/cuda-8.0/include/cuda.h:1750

   CU_TRSA_OVERRIDE_FORMAT : constant := 16#01#;  --  /usr/local/cuda-8.0/include/cuda.h:1756

   CU_TRSF_READ_AS_INTEGER : constant := 16#01#;  --  /usr/local/cuda-8.0/include/cuda.h:1763

   CU_TRSF_NORMALIZED_COORDINATES : constant := 16#02#;  --  /usr/local/cuda-8.0/include/cuda.h:1769

   CU_TRSF_SRGB : constant := 16#10#;  --  /usr/local/cuda-8.0/include/cuda.h:1775
   --  unsupported macro: CU_LAUNCH_PARAM_END ((void*)0x00)
   --  unsupported macro: CU_LAUNCH_PARAM_BUFFER_POINTER ((void*)0x01)
   --  unsupported macro: CU_LAUNCH_PARAM_BUFFER_SIZE ((void*)0x02)

   CU_PARAM_TR_DEFAULT : constant := -1;  --  /usr/local/cuda-8.0/include/cuda.h:1808
   --  unsupported macro: CU_DEVICE_CPU ((CUdevice)-1)
   --  unsupported macro: CU_DEVICE_INVALID ((CUdevice)-2)

  -- * Copyright 1993-2014 NVIDIA Corporation.  All rights reserved.
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

   subtype cuuint32_t is stdint_h.uint32_t;  -- /usr/local/cuda-8.0/include/cuda.h:59

   subtype cuuint64_t is stdint_h.uint64_t;  -- /usr/local/cuda-8.0/include/cuda.h:60

  --*
  -- * CUDA API versioning support
  --  

  --*
  -- * \file cuda.h
  -- * \brief Header file for the CUDA Toolkit application programming interface.
  -- *
  -- * \file cudaGL.h
  -- * \brief Header file for the OpenGL interoperability functions of the
  -- * low-level CUDA driver application programming interface.
  -- *
  -- * \file cudaD3D9.h
  -- * \brief Header file for the Direct3D 9 interoperability functions of the
  -- * low-level CUDA driver application programming interface.
  --  

  --*
  -- * \defgroup CUDA_TYPES Data types used by CUDA driver
  -- * @{
  --  

  --*
  -- * CUDA API version number
  --  

  --*
  -- * CUDA device pointer
  -- * CUdeviceptr is defined as an unsigned integer type whose size matches the size of a pointer on the target platform.
  --  

   subtype CUdeviceptr is Extensions.unsigned_long_long;  -- /usr/local/cuda-8.0/include/cuda.h:221

  --*< CUDA device  
   subtype CUdevice is int;  -- /usr/local/cuda-8.0/include/cuda.h:228

  --*< CUDA context  
   --  skipped empty struct CUctx_st

   type CUcontext is new System.Address;  -- /usr/local/cuda-8.0/include/cuda.h:229

  --*< CUDA module  
   --  skipped empty struct CUmod_st

   type CUmodule is new System.Address;  -- /usr/local/cuda-8.0/include/cuda.h:230

  --*< CUDA function  
   --  skipped empty struct CUfunc_st

   type CUfunction is new System.Address;  -- /usr/local/cuda-8.0/include/cuda.h:231

  --*< CUDA array  
   --  skipped empty struct CUarray_st

   type CUarray is new System.Address;  -- /usr/local/cuda-8.0/include/cuda.h:232

  --*< CUDA mipmapped array  
   --  skipped empty struct CUmipmappedArray_st

   type CUmipmappedArray is new System.Address;  -- /usr/local/cuda-8.0/include/cuda.h:233

  --*< CUDA texture reference  
   --  skipped empty struct CUtexref_st

   type CUtexref is new System.Address;  -- /usr/local/cuda-8.0/include/cuda.h:234

  --*< CUDA surface reference  
   --  skipped empty struct CUsurfref_st

   type CUsurfref is new System.Address;  -- /usr/local/cuda-8.0/include/cuda.h:235

  --*< CUDA event  
   --  skipped empty struct CUevent_st

   type CUevent is new System.Address;  -- /usr/local/cuda-8.0/include/cuda.h:236

  --*< CUDA stream  
   --  skipped empty struct CUstream_st

   type CUstream is new System.Address;  -- /usr/local/cuda-8.0/include/cuda.h:237

  --*< CUDA graphics interop resource  
   --  skipped empty struct CUgraphicsResource_st

   type CUgraphicsResource is new System.Address;  -- /usr/local/cuda-8.0/include/cuda.h:238

  --*< An opaque value that represents a CUDA texture object  
   subtype CUtexObject is Extensions.unsigned_long_long;  -- /usr/local/cuda-8.0/include/cuda.h:239

  --*< An opaque value that represents a CUDA surface object  
   subtype CUsurfObject is Extensions.unsigned_long_long;  -- /usr/local/cuda-8.0/include/cuda.h:240

  --*< CUDA definition of UUID  
   subtype CUuuid_st_bytes_array is Interfaces.C.char_array (0 .. 15);
   type CUuuid_st is record
      bytes : aliased CUuuid_st_bytes_array;  -- /usr/local/cuda-8.0/include/cuda.h:243
   end record;
   pragma Convention (C_Pass_By_Copy, CUuuid_st);  -- /usr/local/cuda-8.0/include/cuda.h:242

   subtype CUuuid is CUuuid_st;

  --*
  -- * CUDA IPC handle size 
  --  

  --*
  -- * CUDA IPC event handle
  --  

   subtype CUipcEventHandle_st_reserved_array is Interfaces.C.char_array (0 .. 63);
   type CUipcEventHandle_st is record
      reserved : aliased CUipcEventHandle_st_reserved_array;  -- /usr/local/cuda-8.0/include/cuda.h:258
   end record;
   pragma Convention (C_Pass_By_Copy, CUipcEventHandle_st);  -- /usr/local/cuda-8.0/include/cuda.h:257

   subtype CUipcEventHandle is CUipcEventHandle_st;

  --*
  -- * CUDA IPC mem handle
  --  

   subtype CUipcMemHandle_st_reserved_array is Interfaces.C.char_array (0 .. 63);
   type CUipcMemHandle_st is record
      reserved : aliased CUipcMemHandle_st_reserved_array;  -- /usr/local/cuda-8.0/include/cuda.h:265
   end record;
   pragma Convention (C_Pass_By_Copy, CUipcMemHandle_st);  -- /usr/local/cuda-8.0/include/cuda.h:264

   subtype CUipcMemHandle is CUipcMemHandle_st;

  --*
  -- * CUDA Ipc Mem Flags
  --  

   subtype CUipcMem_flags_enum is unsigned;
   CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS : constant CUipcMem_flags_enum := 1;  -- /usr/local/cuda-8.0/include/cuda.h:271

  --*< Automatically enable peer access between remote devices as needed  
   subtype CUipcMem_flags is CUipcMem_flags_enum;

  --*
  -- * CUDA Mem Attach Flags
  --  

   subtype CUmemAttach_flags_enum is unsigned;
   CU_MEM_ATTACH_GLOBAL : constant CUmemAttach_flags_enum := 1;
   CU_MEM_ATTACH_HOST : constant CUmemAttach_flags_enum := 2;
   CU_MEM_ATTACH_SINGLE : constant CUmemAttach_flags_enum := 4;  -- /usr/local/cuda-8.0/include/cuda.h:280

  --*< Memory can be accessed by any stream on any device  
  --*< Memory cannot be accessed by any stream on any device  
  --*< Memory can only be accessed by a single stream on the associated device  
   subtype CUmemAttach_flags is CUmemAttach_flags_enum;

  --*
  -- * Context creation flags
  --  

   subtype CUctx_flags_enum is unsigned;
   CU_CTX_SCHED_AUTO : constant CUctx_flags_enum := 0;
   CU_CTX_SCHED_SPIN : constant CUctx_flags_enum := 1;
   CU_CTX_SCHED_YIELD : constant CUctx_flags_enum := 2;
   CU_CTX_SCHED_BLOCKING_SYNC : constant CUctx_flags_enum := 4;
   CU_CTX_BLOCKING_SYNC : constant CUctx_flags_enum := 4;
   CU_CTX_SCHED_MASK : constant CUctx_flags_enum := 7;
   CU_CTX_MAP_HOST : constant CUctx_flags_enum := 8;
   CU_CTX_LMEM_RESIZE_TO_MAX : constant CUctx_flags_enum := 16;
   CU_CTX_FLAGS_MASK : constant CUctx_flags_enum := 31;  -- /usr/local/cuda-8.0/include/cuda.h:289

  --*< Automatic scheduling  
  --*< Set spin as default scheduling  
  --*< Set yield as default scheduling  
  --*< Set blocking synchronization as default scheduling  
  --*< Set blocking synchronization as default scheduling
  --                                         *  \deprecated This flag was deprecated as of CUDA 4.0
  --                                         *  and was replaced with ::CU_CTX_SCHED_BLOCKING_SYNC.  

  --*< Support mapped pinned allocations  
  --*< Keep local memory allocation after launch  
   subtype CUctx_flags is CUctx_flags_enum;

  --*
  -- * Stream creation flags
  --  

   type CUstream_flags_enum is 
     (CU_STREAM_DEFAULT,
      CU_STREAM_NON_BLOCKING);
   pragma Convention (C, CUstream_flags_enum);  -- /usr/local/cuda-8.0/include/cuda.h:306

  --*< Default stream flag  
  --*< Stream does not synchronize with stream 0 (the NULL stream)  
   subtype CUstream_flags is CUstream_flags_enum;

  --*
  -- * Legacy stream handle
  -- *
  -- * Stream handle that can be passed as a CUstream to use an implicit stream
  -- * with legacy synchronization behavior.
  -- *
  -- * See details of the \link_sync_behavior
  --  

  --*
  -- * Per-thread stream handle
  -- *
  -- * Stream handle that can be passed as a CUstream to use an implicit stream
  -- * with per-thread synchronization behavior.
  -- *
  -- * See details of the \link_sync_behavior
  --  

  --*
  -- * Event creation flags
  --  

   subtype CUevent_flags_enum is unsigned;
   CU_EVENT_DEFAULT : constant CUevent_flags_enum := 0;
   CU_EVENT_BLOCKING_SYNC : constant CUevent_flags_enum := 1;
   CU_EVENT_DISABLE_TIMING : constant CUevent_flags_enum := 2;
   CU_EVENT_INTERPROCESS : constant CUevent_flags_enum := 4;  -- /usr/local/cuda-8.0/include/cuda.h:334

  --*< Default event flag  
  --*< Event uses blocking synchronization  
  --*< Event will not record timing data  
  --*< Event is suitable for interprocess use. CU_EVENT_DISABLE_TIMING must be set  
   subtype CUevent_flags is CUevent_flags_enum;

  --*
  -- * Flags for ::cuStreamWaitValue32
  --  

   subtype CUstreamWaitValue_flags_enum is unsigned;
   CU_STREAM_WAIT_VALUE_GEQ : constant CUstreamWaitValue_flags_enum := 0;
   CU_STREAM_WAIT_VALUE_EQ : constant CUstreamWaitValue_flags_enum := 1;
   CU_STREAM_WAIT_VALUE_AND : constant CUstreamWaitValue_flags_enum := 2;
   CU_STREAM_WAIT_VALUE_FLUSH : constant CUstreamWaitValue_flags_enum := 1073741824;  -- /usr/local/cuda-8.0/include/cuda.h:345

  --*< Wait until (int32_t)(*addr - value) >= 0. Note this is a
  --                                             cyclic comparison which ignores wraparound. (Default behavior.)  

  --*< Wait until *addr == value.  
  --*< Wait until (*addr & value) != 0.  
  --*< Follow the wait operation with a flush of outstanding remote writes. This
  --                                             means that, if a remote write operation is guaranteed to have reached the
  --                                             device before the wait can be satisfied, that write is guaranteed to be
  --                                             visible to downstream device work. The device is permitted to reorder
  --                                             remote writes internally. For example, this flag would be required if
  --                                             two remote writes arrive in a defined order, the wait is satisfied by the
  --                                             second write, and downstream work needs to observe the first write.  

   subtype CUstreamWaitValue_flags is CUstreamWaitValue_flags_enum;

  --*
  -- * Flags for ::cuStreamWriteValue32
  --  

   type CUstreamWriteValue_flags_enum is 
     (CU_STREAM_WRITE_VALUE_DEFAULT,
      CU_STREAM_WRITE_VALUE_NO_MEMORY_BARRIER);
   pragma Convention (C, CUstreamWriteValue_flags_enum);  -- /usr/local/cuda-8.0/include/cuda.h:362

  --*< Default behavior  
  --*< Permits the write to be reordered with writes which were issued
  --                                                        before it, as a performance optimization. Normally,
  --                                                        ::cuStreamWriteValue32 will provide a memory fence before the
  --                                                        write, which has similar semantics to
  --                                                        __threadfence_system() but is scoped to the stream
  --                                                        rather than a CUDA thread.  

   subtype CUstreamWriteValue_flags is CUstreamWriteValue_flags_enum;

  --*
  -- * Operations for ::cuStreamBatchMemOp
  --  

   subtype CUstreamBatchMemOpType_enum is unsigned;
   CU_STREAM_MEM_OP_WAIT_VALUE_32 : constant CUstreamBatchMemOpType_enum := 1;
   CU_STREAM_MEM_OP_WRITE_VALUE_32 : constant CUstreamBatchMemOpType_enum := 2;
   CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES : constant CUstreamBatchMemOpType_enum := 3;  -- /usr/local/cuda-8.0/include/cuda.h:375

  --*< Represents a ::cuStreamWaitValue32 operation  
  --*< Represents a ::cuStreamWriteValue32 operation  
  --*< This has the same effect as ::CU_STREAM_WAIT_VALUE_FLUSH, but as a
  --                                                  standalone operation.  

   subtype CUstreamBatchMemOpType is CUstreamBatchMemOpType_enum;

  --*
  -- * Per-operation parameters for ::cuStreamBatchMemOp
  --  

   type CUstreamBatchMemOpParams_union;
   type CUstreamMemOpWaitValueParams_st;
   type anon_19 (discr : unsigned := 0) is record
      case discr is
         when 0 =>
            value : aliased cuuint32_t;  -- /usr/local/cuda-8.0/include/cuda.h:391
         when others =>
            pad : aliased cuuint64_t;  -- /usr/local/cuda-8.0/include/cuda.h:392
      end case;
   end record;
   pragma Convention (C_Pass_By_Copy, anon_19);
   pragma Unchecked_Union (anon_19);type CUstreamMemOpWaitValueParams_st is record
      operation : aliased CUstreamBatchMemOpType;  -- /usr/local/cuda-8.0/include/cuda.h:388
      address : aliased CUdeviceptr;  -- /usr/local/cuda-8.0/include/cuda.h:389
      field_3 : aliased anon_19;
      flags : aliased unsigned;  -- /usr/local/cuda-8.0/include/cuda.h:394
      alias : aliased CUdeviceptr;  -- /usr/local/cuda-8.0/include/cuda.h:395
   end record;
   pragma Convention (C_Pass_By_Copy, CUstreamMemOpWaitValueParams_st);
   type CUstreamMemOpWriteValueParams_st;
   type anon_20 (discr : unsigned := 0) is record
      case discr is
         when 0 =>
            value : aliased cuuint32_t;  -- /usr/local/cuda-8.0/include/cuda.h:401
         when others =>
            pad : aliased cuuint64_t;  -- /usr/local/cuda-8.0/include/cuda.h:402
      end case;
   end record;
   pragma Convention (C_Pass_By_Copy, anon_20);
   pragma Unchecked_Union (anon_20);type CUstreamMemOpWriteValueParams_st is record
      operation : aliased CUstreamBatchMemOpType;  -- /usr/local/cuda-8.0/include/cuda.h:398
      address : aliased CUdeviceptr;  -- /usr/local/cuda-8.0/include/cuda.h:399
      field_3 : aliased anon_20;
      flags : aliased unsigned;  -- /usr/local/cuda-8.0/include/cuda.h:404
      alias : aliased CUdeviceptr;  -- /usr/local/cuda-8.0/include/cuda.h:405
   end record;
   pragma Convention (C_Pass_By_Copy, CUstreamMemOpWriteValueParams_st);
   type CUstreamMemOpFlushRemoteWritesParams_st is record
      operation : aliased CUstreamBatchMemOpType;  -- /usr/local/cuda-8.0/include/cuda.h:408
      flags : aliased unsigned;  -- /usr/local/cuda-8.0/include/cuda.h:409
   end record;
   pragma Convention (C_Pass_By_Copy, CUstreamMemOpFlushRemoteWritesParams_st);
   type CUstreamBatchMemOpParams_union_pad_array is array (0 .. 5) of aliased cuuint64_t;
   type CUstreamBatchMemOpParams_union (discr : unsigned := 0) is record
      case discr is
         when 0 =>
            operation : aliased CUstreamBatchMemOpType;  -- /usr/local/cuda-8.0/include/cuda.h:386
         when 1 =>
            waitValue : aliased CUstreamMemOpWaitValueParams_st;  -- /usr/local/cuda-8.0/include/cuda.h:396
         when 2 =>
            writeValue : aliased CUstreamMemOpWriteValueParams_st;  -- /usr/local/cuda-8.0/include/cuda.h:406
         when 3 =>
            flushRemoteWrites : aliased CUstreamMemOpFlushRemoteWritesParams_st;  -- /usr/local/cuda-8.0/include/cuda.h:410
         when others =>
            pad : aliased CUstreamBatchMemOpParams_union_pad_array;  -- /usr/local/cuda-8.0/include/cuda.h:411
      end case;
   end record;
   pragma Convention (C_Pass_By_Copy, CUstreamBatchMemOpParams_union);
   pragma Unchecked_Union (CUstreamBatchMemOpParams_union);  -- /usr/local/cuda-8.0/include/cuda.h:385

  --*< For driver internal use. Initial value is unimportant.  
  --*< For driver internal use. Initial value is unimportant.  
   subtype CUstreamBatchMemOpParams is CUstreamBatchMemOpParams_union;

  --*
  -- * Occupancy calculator flag
  --  

   type CUoccupancy_flags_enum is 
     (CU_OCCUPANCY_DEFAULT,
      CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE);
   pragma Convention (C, CUoccupancy_flags_enum);  -- /usr/local/cuda-8.0/include/cuda.h:418

  --*< Default behavior  
  --*< Assume global caching is enabled and cannot be automatically turned off  
   subtype CUoccupancy_flags is CUoccupancy_flags_enum;

  --*
  -- * Array formats
  --  

   subtype CUarray_format_enum is unsigned;
   CU_AD_FORMAT_UNSIGNED_INT8 : constant CUarray_format_enum := 1;
   CU_AD_FORMAT_UNSIGNED_INT16 : constant CUarray_format_enum := 2;
   CU_AD_FORMAT_UNSIGNED_INT32 : constant CUarray_format_enum := 3;
   CU_AD_FORMAT_SIGNED_INT8 : constant CUarray_format_enum := 8;
   CU_AD_FORMAT_SIGNED_INT16 : constant CUarray_format_enum := 9;
   CU_AD_FORMAT_SIGNED_INT32 : constant CUarray_format_enum := 10;
   CU_AD_FORMAT_HALF : constant CUarray_format_enum := 16;
   CU_AD_FORMAT_FLOAT : constant CUarray_format_enum := 32;  -- /usr/local/cuda-8.0/include/cuda.h:426

  --*< Unsigned 8-bit integers  
  --*< Unsigned 16-bit integers  
  --*< Unsigned 32-bit integers  
  --*< Signed 8-bit integers  
  --*< Signed 16-bit integers  
  --*< Signed 32-bit integers  
  --*< 16-bit floating point  
  --*< 32-bit floating point  
   subtype CUarray_format is CUarray_format_enum;

  --*
  -- * Texture reference addressing modes
  --  

   type CUaddress_mode_enum is 
     (CU_TR_ADDRESS_MODE_WRAP,
      CU_TR_ADDRESS_MODE_CLAMP,
      CU_TR_ADDRESS_MODE_MIRROR,
      CU_TR_ADDRESS_MODE_BORDER);
   pragma Convention (C, CUaddress_mode_enum);  -- /usr/local/cuda-8.0/include/cuda.h:440

  --*< Wrapping address mode  
  --*< Clamp to edge address mode  
  --*< Mirror address mode  
  --*< Border address mode  
   subtype CUaddress_mode is CUaddress_mode_enum;

  --*
  -- * Texture reference filtering modes
  --  

   type CUfilter_mode_enum is 
     (CU_TR_FILTER_MODE_POINT,
      CU_TR_FILTER_MODE_LINEAR);
   pragma Convention (C, CUfilter_mode_enum);  -- /usr/local/cuda-8.0/include/cuda.h:450

  --*< Point filter mode  
  --*< Linear filter mode  
   subtype CUfilter_mode is CUfilter_mode_enum;

  --*
  -- * Device properties
  --  

   subtype CUdevice_attribute_enum is unsigned;
   CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK : constant CUdevice_attribute_enum := 1;
   CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X : constant CUdevice_attribute_enum := 2;
   CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y : constant CUdevice_attribute_enum := 3;
   CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z : constant CUdevice_attribute_enum := 4;
   CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X : constant CUdevice_attribute_enum := 5;
   CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y : constant CUdevice_attribute_enum := 6;
   CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z : constant CUdevice_attribute_enum := 7;
   CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK : constant CUdevice_attribute_enum := 8;
   CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK : constant CUdevice_attribute_enum := 8;
   CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY : constant CUdevice_attribute_enum := 9;
   CU_DEVICE_ATTRIBUTE_WARP_SIZE : constant CUdevice_attribute_enum := 10;
   CU_DEVICE_ATTRIBUTE_MAX_PITCH : constant CUdevice_attribute_enum := 11;
   CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK : constant CUdevice_attribute_enum := 12;
   CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK : constant CUdevice_attribute_enum := 12;
   CU_DEVICE_ATTRIBUTE_CLOCK_RATE : constant CUdevice_attribute_enum := 13;
   CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT : constant CUdevice_attribute_enum := 14;
   CU_DEVICE_ATTRIBUTE_GPU_OVERLAP : constant CUdevice_attribute_enum := 15;
   CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT : constant CUdevice_attribute_enum := 16;
   CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT : constant CUdevice_attribute_enum := 17;
   CU_DEVICE_ATTRIBUTE_INTEGRATED : constant CUdevice_attribute_enum := 18;
   CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY : constant CUdevice_attribute_enum := 19;
   CU_DEVICE_ATTRIBUTE_COMPUTE_MODE : constant CUdevice_attribute_enum := 20;
   CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH : constant CUdevice_attribute_enum := 21;
   CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH : constant CUdevice_attribute_enum := 22;
   CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT : constant CUdevice_attribute_enum := 23;
   CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH : constant CUdevice_attribute_enum := 24;
   CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT : constant CUdevice_attribute_enum := 25;
   CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH : constant CUdevice_attribute_enum := 26;
   CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH : constant CUdevice_attribute_enum := 27;
   CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT : constant CUdevice_attribute_enum := 28;
   CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS : constant CUdevice_attribute_enum := 29;
   CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH : constant CUdevice_attribute_enum := 27;
   CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT : constant CUdevice_attribute_enum := 28;
   CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES : constant CUdevice_attribute_enum := 29;
   CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT : constant CUdevice_attribute_enum := 30;
   CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS : constant CUdevice_attribute_enum := 31;
   CU_DEVICE_ATTRIBUTE_ECC_ENABLED : constant CUdevice_attribute_enum := 32;
   CU_DEVICE_ATTRIBUTE_PCI_BUS_ID : constant CUdevice_attribute_enum := 33;
   CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID : constant CUdevice_attribute_enum := 34;
   CU_DEVICE_ATTRIBUTE_TCC_DRIVER : constant CUdevice_attribute_enum := 35;
   CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE : constant CUdevice_attribute_enum := 36;
   CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH : constant CUdevice_attribute_enum := 37;
   CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE : constant CUdevice_attribute_enum := 38;
   CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR : constant CUdevice_attribute_enum := 39;
   CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT : constant CUdevice_attribute_enum := 40;
   CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING : constant CUdevice_attribute_enum := 41;
   CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH : constant CUdevice_attribute_enum := 42;
   CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS : constant CUdevice_attribute_enum := 43;
   CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER : constant CUdevice_attribute_enum := 44;
   CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH : constant CUdevice_attribute_enum := 45;
   CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT : constant CUdevice_attribute_enum := 46;
   CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE : constant CUdevice_attribute_enum := 47;
   CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE : constant CUdevice_attribute_enum := 48;
   CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE : constant CUdevice_attribute_enum := 49;
   CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID : constant CUdevice_attribute_enum := 50;
   CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT : constant CUdevice_attribute_enum := 51;
   CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH : constant CUdevice_attribute_enum := 52;
   CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH : constant CUdevice_attribute_enum := 53;
   CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS : constant CUdevice_attribute_enum := 54;
   CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH : constant CUdevice_attribute_enum := 55;
   CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH : constant CUdevice_attribute_enum := 56;
   CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT : constant CUdevice_attribute_enum := 57;
   CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH : constant CUdevice_attribute_enum := 58;
   CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT : constant CUdevice_attribute_enum := 59;
   CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH : constant CUdevice_attribute_enum := 60;
   CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH : constant CUdevice_attribute_enum := 61;
   CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS : constant CUdevice_attribute_enum := 62;
   CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH : constant CUdevice_attribute_enum := 63;
   CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT : constant CUdevice_attribute_enum := 64;
   CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS : constant CUdevice_attribute_enum := 65;
   CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH : constant CUdevice_attribute_enum := 66;
   CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH : constant CUdevice_attribute_enum := 67;
   CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS : constant CUdevice_attribute_enum := 68;
   CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH : constant CUdevice_attribute_enum := 69;
   CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH : constant CUdevice_attribute_enum := 70;
   CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT : constant CUdevice_attribute_enum := 71;
   CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH : constant CUdevice_attribute_enum := 72;
   CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH : constant CUdevice_attribute_enum := 73;
   CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT : constant CUdevice_attribute_enum := 74;
   CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR : constant CUdevice_attribute_enum := 75;
   CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR : constant CUdevice_attribute_enum := 76;
   CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH : constant CUdevice_attribute_enum := 77;
   CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED : constant CUdevice_attribute_enum := 78;
   CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED : constant CUdevice_attribute_enum := 79;
   CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED : constant CUdevice_attribute_enum := 80;
   CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR : constant CUdevice_attribute_enum := 81;
   CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR : constant CUdevice_attribute_enum := 82;
   CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY : constant CUdevice_attribute_enum := 83;
   CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD : constant CUdevice_attribute_enum := 84;
   CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID : constant CUdevice_attribute_enum := 85;
   CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED : constant CUdevice_attribute_enum := 86;
   CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO : constant CUdevice_attribute_enum := 87;
   CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS : constant CUdevice_attribute_enum := 88;
   CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS : constant CUdevice_attribute_enum := 89;
   CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED : constant CUdevice_attribute_enum := 90;
   CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM : constant CUdevice_attribute_enum := 91;
   CU_DEVICE_ATTRIBUTE_MAX : constant CUdevice_attribute_enum := 92;  -- /usr/local/cuda-8.0/include/cuda.h:458

  --*< Maximum number of threads per block  
  --*< Maximum block dimension X  
  --*< Maximum block dimension Y  
  --*< Maximum block dimension Z  
  --*< Maximum grid dimension X  
  --*< Maximum grid dimension Y  
  --*< Maximum grid dimension Z  
  --*< Maximum shared memory available per block in bytes  
  --*< Deprecated, use CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK  
  --*< Memory available on device for __constant__ variables in a CUDA C kernel in bytes  
  --*< Warp size in threads  
  --*< Maximum pitch in bytes allowed by memory copies  
  --*< Maximum number of 32-bit registers available per block  
  --*< Deprecated, use CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK  
  --*< Typical clock frequency in kilohertz  
  --*< Alignment requirement for textures  
  --*< Device can possibly copy memory and execute a kernel concurrently. Deprecated. Use instead CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT.  
  --*< Number of multiprocessors on device  
  --*< Specifies whether there is a run time limit on kernels  
  --*< Device is integrated with host memory  
  --*< Device can map host memory into CUDA address space  
  --*< Compute mode (See ::CUcomputemode for details)  
  --*< Maximum 1D texture width  
  --*< Maximum 2D texture width  
  --*< Maximum 2D texture height  
  --*< Maximum 3D texture width  
  --*< Maximum 3D texture height  
  --*< Maximum 3D texture depth  
  --*< Maximum 2D layered texture width  
  --*< Maximum 2D layered texture height  
  --*< Maximum layers in a 2D layered texture  
  --*< Deprecated, use CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH  
  --*< Deprecated, use CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT  
  --*< Deprecated, use CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS  
  --*< Alignment requirement for surfaces  
  --*< Device can possibly execute multiple kernels concurrently  
  --*< Device has ECC support enabled  
  --*< PCI bus ID of the device  
  --*< PCI device ID of the device  
  --*< Device is using TCC driver model  
  --*< Peak memory clock frequency in kilohertz  
  --*< Global memory bus width in bits  
  --*< Size of L2 cache in bytes  
  --*< Maximum resident threads per multiprocessor  
  --*< Number of asynchronous engines  
  --*< Device shares a unified address space with the host  
  --*< Maximum 1D layered texture width  
  --*< Maximum layers in a 1D layered texture  
  --*< Deprecated, do not use.  
  --*< Maximum 2D texture width if CUDA_ARRAY3D_TEXTURE_GATHER is set  
  --*< Maximum 2D texture height if CUDA_ARRAY3D_TEXTURE_GATHER is set  
  --*< Alternate maximum 3D texture width  
  --*< Alternate maximum 3D texture height  
  --*< Alternate maximum 3D texture depth  
  --*< PCI domain ID of the device  
  --*< Pitch alignment requirement for textures  
  --*< Maximum cubemap texture width/height  
  --*< Maximum cubemap layered texture width/height  
  --*< Maximum layers in a cubemap layered texture  
  --*< Maximum 1D surface width  
  --*< Maximum 2D surface width  
  --*< Maximum 2D surface height  
  --*< Maximum 3D surface width  
  --*< Maximum 3D surface height  
  --*< Maximum 3D surface depth  
  --*< Maximum 1D layered surface width  
  --*< Maximum layers in a 1D layered surface  
  --*< Maximum 2D layered surface width  
  --*< Maximum 2D layered surface height  
  --*< Maximum layers in a 2D layered surface  
  --*< Maximum cubemap surface width  
  --*< Maximum cubemap layered surface width  
  --*< Maximum layers in a cubemap layered surface  
  --*< Maximum 1D linear texture width  
  --*< Maximum 2D linear texture width  
  --*< Maximum 2D linear texture height  
  --*< Maximum 2D linear texture pitch in bytes  
  --*< Maximum mipmapped 2D texture width  
  --*< Maximum mipmapped 2D texture height  
  --*< Major compute capability version number  
  --*< Minor compute capability version number  
  --*< Maximum mipmapped 1D texture width  
  --*< Device supports stream priorities  
  --*< Device supports caching globals in L1  
  --*< Device supports caching locals in L1  
  --*< Maximum shared memory available per multiprocessor in bytes  
  --*< Maximum number of 32-bit registers available per multiprocessor  
  --*< Device can allocate managed memory on this system  
  --*< Device is on a multi-GPU board  
  --*< Unique id for a group of devices on the same multi-GPU board  
  --*< Link between the device and the host supports native atomic operations (this is a placeholder attribute, and is not supported on any current hardware) 
  --*< Ratio of single precision performance (in floating-point operations per second) to double precision performance  
  --*< Device supports coherently accessing pageable memory without calling cudaHostRegister on it  
  --*< Device can coherently access managed memory concurrently with the CPU  
  --*< Device supports compute preemption.  
  --*< Device can access host registered memory at the same virtual address as the CPU  
   subtype CUdevice_attribute is CUdevice_attribute_enum;

  --*
  -- * Legacy device properties
  --  

  --*< Maximum number of threads per block  
   type CUdevprop_st_maxThreadsDim_array is array (0 .. 2) of aliased int;
   type CUdevprop_st_maxGridSize_array is array (0 .. 2) of aliased int;
   type CUdevprop_st is record
      maxThreadsPerBlock : aliased int;  -- /usr/local/cuda-8.0/include/cuda.h:562
      maxThreadsDim : aliased CUdevprop_st_maxThreadsDim_array;  -- /usr/local/cuda-8.0/include/cuda.h:563
      maxGridSize : aliased CUdevprop_st_maxGridSize_array;  -- /usr/local/cuda-8.0/include/cuda.h:564
      sharedMemPerBlock : aliased int;  -- /usr/local/cuda-8.0/include/cuda.h:565
      totalConstantMemory : aliased int;  -- /usr/local/cuda-8.0/include/cuda.h:566
      SIMDWidth : aliased int;  -- /usr/local/cuda-8.0/include/cuda.h:567
      memPitch : aliased int;  -- /usr/local/cuda-8.0/include/cuda.h:568
      regsPerBlock : aliased int;  -- /usr/local/cuda-8.0/include/cuda.h:569
      clockRate : aliased int;  -- /usr/local/cuda-8.0/include/cuda.h:570
      textureAlign : aliased int;  -- /usr/local/cuda-8.0/include/cuda.h:571
   end record;
   pragma Convention (C_Pass_By_Copy, CUdevprop_st);  -- /usr/local/cuda-8.0/include/cuda.h:561

  --*< Maximum size of each dimension of a block  
  --*< Maximum size of each dimension of a grid  
  --*< Shared memory available per block in bytes  
  --*< Constant memory available on device in bytes  
  --*< Warp size in threads  
  --*< Maximum pitch in bytes allowed by memory copies  
  --*< 32-bit registers available per block  
  --*< Clock frequency in kilohertz  
  --*< Alignment requirement for textures  
   subtype CUdevprop is CUdevprop_st;

  --*
  -- * Pointer information
  --  

   subtype CUpointer_attribute_enum is unsigned;
   CU_POINTER_ATTRIBUTE_CONTEXT : constant CUpointer_attribute_enum := 1;
   CU_POINTER_ATTRIBUTE_MEMORY_TYPE : constant CUpointer_attribute_enum := 2;
   CU_POINTER_ATTRIBUTE_DEVICE_POINTER : constant CUpointer_attribute_enum := 3;
   CU_POINTER_ATTRIBUTE_HOST_POINTER : constant CUpointer_attribute_enum := 4;
   CU_POINTER_ATTRIBUTE_P2P_TOKENS : constant CUpointer_attribute_enum := 5;
   CU_POINTER_ATTRIBUTE_SYNC_MEMOPS : constant CUpointer_attribute_enum := 6;
   CU_POINTER_ATTRIBUTE_BUFFER_ID : constant CUpointer_attribute_enum := 7;
   CU_POINTER_ATTRIBUTE_IS_MANAGED : constant CUpointer_attribute_enum := 8;  -- /usr/local/cuda-8.0/include/cuda.h:577

  --*< The ::CUcontext on which a pointer was allocated or registered  
  --*< The ::CUmemorytype describing the physical location of a pointer  
  --*< The address at which a pointer's memory may be accessed on the device  
  --*< The address at which a pointer's memory may be accessed on the host  
  --*< A pair of tokens for use with the nv-p2p.h Linux kernel interface  
  --*< Synchronize every synchronous memory operation initiated on this region  
  --*< A process-wide unique ID for an allocated memory region 
  --*< Indicates if the pointer points to managed memory  
   subtype CUpointer_attribute is CUpointer_attribute_enum;

  --*
  -- * Function properties
  --  

   type CUfunction_attribute_enum is 
     (CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
      CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
      CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES,
      CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
      CU_FUNC_ATTRIBUTE_NUM_REGS,
      CU_FUNC_ATTRIBUTE_PTX_VERSION,
      CU_FUNC_ATTRIBUTE_BINARY_VERSION,
      CU_FUNC_ATTRIBUTE_CACHE_MODE_CA,
      CU_FUNC_ATTRIBUTE_MAX);
   pragma Convention (C, CUfunction_attribute_enum);  -- /usr/local/cuda-8.0/include/cuda.h:591

  --*
  --     * The maximum number of threads per block, beyond which a launch of the
  --     * function would fail. This number depends on both the function and the
  --     * device on which the function is currently loaded.
  --      

  --*
  --     * The size in bytes of statically-allocated shared memory required by
  --     * this function. This does not include dynamically-allocated shared
  --     * memory requested by the user at runtime.
  --      

  --*
  --     * The size in bytes of user-allocated constant memory required by this
  --     * function.
  --      

  --*
  --     * The size in bytes of local memory used by each thread of this function.
  --      

  --*
  --     * The number of registers used by each thread of this function.
  --      

  --*
  --     * The PTX virtual architecture version for which the function was
  --     * compiled. This value is the major PTX version * 10 + the minor PTX
  --     * version, so a PTX version 1.3 function would return the value 13.
  --     * Note that this may return the undefined value of 0 for cubins
  --     * compiled prior to CUDA 3.0.
  --      

  --*
  --     * The binary architecture version for which the function was compiled.
  --     * This value is the major binary version * 10 + the minor binary version,
  --     * so a binary version 1.3 function would return the value 13. Note that
  --     * this will return a value of 10 for legacy cubins that do not have a
  --     * properly-encoded binary architecture version.
  --      

  --*
  --     * The attribute to indicate whether the function has been compiled with 
  --     * user specified option "-Xptxas --dlcm=ca" set .
  --      

   subtype CUfunction_attribute is CUfunction_attribute_enum;

  --*
  -- * Function cache configurations
  --  

   type CUfunc_cache_enum is 
     (CU_FUNC_CACHE_PREFER_NONE,
      CU_FUNC_CACHE_PREFER_SHARED,
      CU_FUNC_CACHE_PREFER_L1,
      CU_FUNC_CACHE_PREFER_EQUAL);
   pragma Convention (C, CUfunc_cache_enum);  -- /usr/local/cuda-8.0/include/cuda.h:652

  --*< no preference for shared memory or L1 (default)  
  --*< prefer larger shared memory and smaller L1 cache  
  --*< prefer larger L1 cache and smaller shared memory  
  --*< prefer equal sized L1 cache and shared memory  
   subtype CUfunc_cache is CUfunc_cache_enum;

  --*
  -- * Shared memory configurations
  --  

   type CUsharedconfig_enum is 
     (CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE,
      CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE,
      CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE);
   pragma Convention (C, CUsharedconfig_enum);  -- /usr/local/cuda-8.0/include/cuda.h:662

  --*< set default shared memory bank size  
  --*< set shared memory bank width to four bytes  
  --*< set shared memory bank width to eight bytes  
   subtype CUsharedconfig is CUsharedconfig_enum;

  --*
  -- * Memory types
  --  

   subtype CUmemorytype_enum is unsigned;
   CU_MEMORYTYPE_HOST : constant CUmemorytype_enum := 1;
   CU_MEMORYTYPE_DEVICE : constant CUmemorytype_enum := 2;
   CU_MEMORYTYPE_ARRAY : constant CUmemorytype_enum := 3;
   CU_MEMORYTYPE_UNIFIED : constant CUmemorytype_enum := 4;  -- /usr/local/cuda-8.0/include/cuda.h:671

  --*< Host memory  
  --*< Device memory  
  --*< Array memory  
  --*< Unified device or host memory  
   subtype CUmemorytype is CUmemorytype_enum;

  --*
  -- * Compute Modes
  --  

   subtype CUcomputemode_enum is unsigned;
   CU_COMPUTEMODE_DEFAULT : constant CUcomputemode_enum := 0;
   CU_COMPUTEMODE_PROHIBITED : constant CUcomputemode_enum := 2;
   CU_COMPUTEMODE_EXCLUSIVE_PROCESS : constant CUcomputemode_enum := 3;  -- /usr/local/cuda-8.0/include/cuda.h:681

  --*< Default compute mode (Multiple contexts allowed per device)  
  --*< Compute-prohibited mode (No contexts can be created on this device at this time)  
  --*< Compute-exclusive-process mode (Only one context used by a single process can be present on this device at a time)  
   subtype CUcomputemode is CUcomputemode_enum;

  --*
  -- * Memory advise values
  --  

   subtype CUmem_advise_enum is unsigned;
   CU_MEM_ADVISE_SET_READ_MOSTLY : constant CUmem_advise_enum := 1;
   CU_MEM_ADVISE_UNSET_READ_MOSTLY : constant CUmem_advise_enum := 2;
   CU_MEM_ADVISE_SET_PREFERRED_LOCATION : constant CUmem_advise_enum := 3;
   CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION : constant CUmem_advise_enum := 4;
   CU_MEM_ADVISE_SET_ACCESSED_BY : constant CUmem_advise_enum := 5;
   CU_MEM_ADVISE_UNSET_ACCESSED_BY : constant CUmem_advise_enum := 6;  -- /usr/local/cuda-8.0/include/cuda.h:690

  --*< Data will mostly be read and only occassionally be written to  
  --*< Undo the effect of ::CU_MEM_ADVISE_SET_READ_MOSTLY  
  --*< Set the preferred location for the data as the specified device  
  --*< Clear the preferred location for the data  
  --*< Data will be accessed by the specified device, so prevent page faults as much as possible  
  --*< Let the Unified Memory subsystem decide on the page faulting policy for the specified device  
   subtype CUmem_advise is CUmem_advise_enum;

   subtype CUmem_range_attribute_enum is unsigned;
   CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY : constant CUmem_range_attribute_enum := 1;
   CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION : constant CUmem_range_attribute_enum := 2;
   CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY : constant CUmem_range_attribute_enum := 3;
   CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION : constant CUmem_range_attribute_enum := 4;  -- /usr/local/cuda-8.0/include/cuda.h:699

  --*< Whether the range will mostly be read and only occassionally be written to  
  --*< The preferred location of the range  
  --*< Memory range has ::CU_MEM_ADVISE_SET_ACCESSED_BY set for specified device  
  --*< The last location to which the range was prefetched  
   subtype CUmem_range_attribute is CUmem_range_attribute_enum;

  --*
  -- * Online compiler and linker options
  --  

   type CUjit_option_enum is 
     (CU_JIT_MAX_REGISTERS,
      CU_JIT_THREADS_PER_BLOCK,
      CU_JIT_WALL_TIME,
      CU_JIT_INFO_LOG_BUFFER,
      CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
      CU_JIT_ERROR_LOG_BUFFER,
      CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
      CU_JIT_OPTIMIZATION_LEVEL,
      CU_JIT_TARGET_FROM_CUCONTEXT,
      CU_JIT_TARGET,
      CU_JIT_FALLBACK_STRATEGY,
      CU_JIT_GENERATE_DEBUG_INFO,
      CU_JIT_LOG_VERBOSE,
      CU_JIT_GENERATE_LINE_INFO,
      CU_JIT_CACHE_MODE,
      CU_JIT_NEW_SM3X_OPT,
      CU_JIT_FAST_COMPILE,
      CU_JIT_NUM_OPTIONS);
   pragma Convention (C, CUjit_option_enum);  -- /usr/local/cuda-8.0/include/cuda.h:709

  --*
  --     * Max number of registers that a thread may use.\n
  --     * Option type: unsigned int\n
  --     * Applies to: compiler only
  --      

  --*
  --     * IN: Specifies minimum number of threads per block to target compilation
  --     * for\n
  --     * OUT: Returns the number of threads the compiler actually targeted.
  --     * This restricts the resource utilization fo the compiler (e.g. max
  --     * registers) such that a block with the given number of threads should be
  --     * able to launch based on register limitations. Note, this option does not
  --     * currently take into account any other resource limitations, such as
  --     * shared memory utilization.\n
  --     * Cannot be combined with ::CU_JIT_TARGET.\n
  --     * Option type: unsigned int\n
  --     * Applies to: compiler only
  --      

  --*
  --     * Overwrites the option value with the total wall clock time, in
  --     * milliseconds, spent in the compiler and linker\n
  --     * Option type: float\n
  --     * Applies to: compiler and linker
  --      

  --*
  --     * Pointer to a buffer in which to print any log messages
  --     * that are informational in nature (the buffer size is specified via
  --     * option ::CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES)\n
  --     * Option type: char *\n
  --     * Applies to: compiler and linker
  --      

  --*
  --     * IN: Log buffer size in bytes.  Log messages will be capped at this size
  --     * (including null terminator)\n
  --     * OUT: Amount of log buffer filled with messages\n
  --     * Option type: unsigned int\n
  --     * Applies to: compiler and linker
  --      

  --*
  --     * Pointer to a buffer in which to print any log messages that
  --     * reflect errors (the buffer size is specified via option
  --     * ::CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES)\n
  --     * Option type: char *\n
  --     * Applies to: compiler and linker
  --      

  --*
  --     * IN: Log buffer size in bytes.  Log messages will be capped at this size
  --     * (including null terminator)\n
  --     * OUT: Amount of log buffer filled with messages\n
  --     * Option type: unsigned int\n
  --     * Applies to: compiler and linker
  --      

  --*
  --     * Level of optimizations to apply to generated code (0 - 4), with 4
  --     * being the default and highest level of optimizations.\n
  --     * Option type: unsigned int\n
  --     * Applies to: compiler only
  --      

  --*
  --     * No option value required. Determines the target based on the current
  --     * attached context (default)\n
  --     * Option type: No option value needed\n
  --     * Applies to: compiler and linker
  --      

  --*
  --     * Target is chosen based on supplied ::CUjit_target.  Cannot be
  --     * combined with ::CU_JIT_THREADS_PER_BLOCK.\n
  --     * Option type: unsigned int for enumerated type ::CUjit_target\n
  --     * Applies to: compiler and linker
  --      

  --*
  --     * Specifies choice of fallback strategy if matching cubin is not found.
  --     * Choice is based on supplied ::CUjit_fallback.  This option cannot be
  --     * used with cuLink* APIs as the linker requires exact matches.\n
  --     * Option type: unsigned int for enumerated type ::CUjit_fallback\n
  --     * Applies to: compiler only
  --      

  --*
  --     * Specifies whether to create debug information in output (-g)
  --     * (0: false, default)\n
  --     * Option type: int\n
  --     * Applies to: compiler and linker
  --      

  --*
  --     * Generate verbose log messages (0: false, default)\n
  --     * Option type: int\n
  --     * Applies to: compiler and linker
  --      

  --*
  --     * Generate line number information (-lineinfo) (0: false, default)\n
  --     * Option type: int\n
  --     * Applies to: compiler only
  --      

  --*
  --     * Specifies whether to enable caching explicitly (-dlcm) \n
  --     * Choice is based on supplied ::CUjit_cacheMode_enum.\n
  --     * Option type: unsigned int for enumerated type ::CUjit_cacheMode_enum\n
  --     * Applies to: compiler only
  --      

  --*
  --     * The below jit options are used for internal purposes only, in this version of CUDA
  --      

   subtype CUjit_option is CUjit_option_enum;

  --*
  -- * Online compilation targets
  --  

   subtype CUjit_target_enum is unsigned;
   CU_TARGET_COMPUTE_10 : constant CUjit_target_enum := 10;
   CU_TARGET_COMPUTE_11 : constant CUjit_target_enum := 11;
   CU_TARGET_COMPUTE_12 : constant CUjit_target_enum := 12;
   CU_TARGET_COMPUTE_13 : constant CUjit_target_enum := 13;
   CU_TARGET_COMPUTE_20 : constant CUjit_target_enum := 20;
   CU_TARGET_COMPUTE_21 : constant CUjit_target_enum := 21;
   CU_TARGET_COMPUTE_30 : constant CUjit_target_enum := 30;
   CU_TARGET_COMPUTE_32 : constant CUjit_target_enum := 32;
   CU_TARGET_COMPUTE_35 : constant CUjit_target_enum := 35;
   CU_TARGET_COMPUTE_37 : constant CUjit_target_enum := 37;
   CU_TARGET_COMPUTE_50 : constant CUjit_target_enum := 50;
   CU_TARGET_COMPUTE_52 : constant CUjit_target_enum := 52;
   CU_TARGET_COMPUTE_53 : constant CUjit_target_enum := 53;
   CU_TARGET_COMPUTE_60 : constant CUjit_target_enum := 60;
   CU_TARGET_COMPUTE_61 : constant CUjit_target_enum := 61;
   CU_TARGET_COMPUTE_62 : constant CUjit_target_enum := 62;  -- /usr/local/cuda-8.0/include/cuda.h:853

  --*< Compute device class 1.0  
  --*< Compute device class 1.1  
  --*< Compute device class 1.2  
  --*< Compute device class 1.3  
  --*< Compute device class 2.0  
  --*< Compute device class 2.1  
  --*< Compute device class 3.0  
  --*< Compute device class 3.2  
  --*< Compute device class 3.5  
  --*< Compute device class 3.7  
  --*< Compute device class 5.0  
  --*< Compute device class 5.2  
  --*< Compute device class 5.3  
  --*< Compute device class 6.0. This must be removed for CUDA 7.0 toolkit. See bug 1518217.  
  --*< Compute device class 6.1. This must be removed for CUDA 7.0 toolkit. 
  --*< Compute device class 6.2. This must be removed for CUDA 7.0 toolkit. 
   subtype CUjit_target is CUjit_target_enum;

  --*
  -- * Cubin matching fallback strategies
  --  

   type CUjit_fallback_enum is 
     (CU_PREFER_PTX,
      CU_PREFER_BINARY);
   pragma Convention (C, CUjit_fallback_enum);  -- /usr/local/cuda-8.0/include/cuda.h:876

  --*< Prefer to compile ptx if exact binary match not found  
  --*< Prefer to fall back to compatible binary code if exact match not found  
   subtype CUjit_fallback is CUjit_fallback_enum;

  --*
  -- * Caching modes for dlcm 
  --  

   type CUjit_cacheMode_enum is 
     (CU_JIT_CACHE_OPTION_NONE,
      CU_JIT_CACHE_OPTION_CG,
      CU_JIT_CACHE_OPTION_CA);
   pragma Convention (C, CUjit_cacheMode_enum);  -- /usr/local/cuda-8.0/include/cuda.h:887

  --*< Compile with no -dlcm flag specified  
  --*< Compile with L1 cache disabled  
  --*< Compile with L1 cache enabled  
   subtype CUjit_cacheMode is CUjit_cacheMode_enum;

  --*
  -- * Device code formats
  --  

   type CUjitInputType_enum is 
     (CU_JIT_INPUT_CUBIN,
      CU_JIT_INPUT_PTX,
      CU_JIT_INPUT_FATBINARY,
      CU_JIT_INPUT_OBJECT,
      CU_JIT_INPUT_LIBRARY,
      CU_JIT_NUM_INPUT_TYPES);
   pragma Convention (C, CUjitInputType_enum);  -- /usr/local/cuda-8.0/include/cuda.h:897

  --*
  --     * Compiled device-class-specific device code\n
  --     * Applicable options: none
  --      

  --*
  --     * PTX source code\n
  --     * Applicable options: PTX compiler options
  --      

  --*
  --     * Bundle of multiple cubins and/or PTX of some device code\n
  --     * Applicable options: PTX compiler options, ::CU_JIT_FALLBACK_STRATEGY
  --      

  --*
  --     * Host object with embedded device code\n
  --     * Applicable options: PTX compiler options, ::CU_JIT_FALLBACK_STRATEGY
  --      

  --*
  --     * Archive of host objects with embedded device code\n
  --     * Applicable options: PTX compiler options, ::CU_JIT_FALLBACK_STRATEGY
  --      

   subtype CUjitInputType is CUjitInputType_enum;

   --  skipped empty struct CUlinkState_st

   type CUlinkState is new System.Address;  -- /usr/local/cuda-8.0/include/cuda.h:933

  --*
  -- * Flags to register a graphics resource
  --  

   subtype CUgraphicsRegisterFlags_enum is unsigned;
   CU_GRAPHICS_REGISTER_FLAGS_NONE : constant CUgraphicsRegisterFlags_enum := 0;
   CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY : constant CUgraphicsRegisterFlags_enum := 1;
   CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD : constant CUgraphicsRegisterFlags_enum := 2;
   CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST : constant CUgraphicsRegisterFlags_enum := 4;
   CU_GRAPHICS_REGISTER_FLAGS_TEXTURE_GATHER : constant CUgraphicsRegisterFlags_enum := 8;  -- /usr/local/cuda-8.0/include/cuda.h:939

   subtype CUgraphicsRegisterFlags is CUgraphicsRegisterFlags_enum;

  --*
  -- * Flags for mapping and unmapping interop resources
  --  

   type CUgraphicsMapResourceFlags_enum is 
     (CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE,
      CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY,
      CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD);
   pragma Convention (C, CUgraphicsMapResourceFlags_enum);  -- /usr/local/cuda-8.0/include/cuda.h:950

   subtype CUgraphicsMapResourceFlags is CUgraphicsMapResourceFlags_enum;

  --*
  -- * Array indices for cube faces
  --  

   type CUarray_cubemap_face_enum is 
     (CU_CUBEMAP_FACE_POSITIVE_X,
      CU_CUBEMAP_FACE_NEGATIVE_X,
      CU_CUBEMAP_FACE_POSITIVE_Y,
      CU_CUBEMAP_FACE_NEGATIVE_Y,
      CU_CUBEMAP_FACE_POSITIVE_Z,
      CU_CUBEMAP_FACE_NEGATIVE_Z);
   pragma Convention (C, CUarray_cubemap_face_enum);  -- /usr/local/cuda-8.0/include/cuda.h:959

  --*< Positive X face of cubemap  
  --*< Negative X face of cubemap  
  --*< Positive Y face of cubemap  
  --*< Negative Y face of cubemap  
  --*< Positive Z face of cubemap  
  --*< Negative Z face of cubemap  
   subtype CUarray_cubemap_face is CUarray_cubemap_face_enum;

  --*
  -- * Limits
  --  

   type CUlimit_enum is 
     (CU_LIMIT_STACK_SIZE,
      CU_LIMIT_PRINTF_FIFO_SIZE,
      CU_LIMIT_MALLOC_HEAP_SIZE,
      CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH,
      CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT,
      CU_LIMIT_MAX);
   pragma Convention (C, CUlimit_enum);  -- /usr/local/cuda-8.0/include/cuda.h:971

  --*< GPU thread stack size  
  --*< GPU printf FIFO size  
  --*< GPU malloc heap size  
  --*< GPU device runtime launch synchronize depth  
  --*< GPU device runtime pending launch count  
   subtype CUlimit is CUlimit_enum;

  --*
  -- * Resource types
  --  

   type CUresourcetype_enum is 
     (CU_RESOURCE_TYPE_ARRAY,
      CU_RESOURCE_TYPE_MIPMAPPED_ARRAY,
      CU_RESOURCE_TYPE_LINEAR,
      CU_RESOURCE_TYPE_PITCH2D);
   pragma Convention (C, CUresourcetype_enum);  -- /usr/local/cuda-8.0/include/cuda.h:983

  --*< Array resoure  
  --*< Mipmapped array resource  
  --*< Linear resource  
  --*< Pitch 2D resource  
   subtype CUresourcetype is CUresourcetype_enum;

  --*
  -- * Error codes
  --  

   subtype cudaError_enum is unsigned;
   CUDA_SUCCESS : constant cudaError_enum := 0;
   CUDA_ERROR_INVALID_VALUE : constant cudaError_enum := 1;
   CUDA_ERROR_OUT_OF_MEMORY : constant cudaError_enum := 2;
   CUDA_ERROR_NOT_INITIALIZED : constant cudaError_enum := 3;
   CUDA_ERROR_DEINITIALIZED : constant cudaError_enum := 4;
   CUDA_ERROR_PROFILER_DISABLED : constant cudaError_enum := 5;
   CUDA_ERROR_PROFILER_NOT_INITIALIZED : constant cudaError_enum := 6;
   CUDA_ERROR_PROFILER_ALREADY_STARTED : constant cudaError_enum := 7;
   CUDA_ERROR_PROFILER_ALREADY_STOPPED : constant cudaError_enum := 8;
   CUDA_ERROR_NO_DEVICE : constant cudaError_enum := 100;
   CUDA_ERROR_INVALID_DEVICE : constant cudaError_enum := 101;
   CUDA_ERROR_INVALID_IMAGE : constant cudaError_enum := 200;
   CUDA_ERROR_INVALID_CONTEXT : constant cudaError_enum := 201;
   CUDA_ERROR_CONTEXT_ALREADY_CURRENT : constant cudaError_enum := 202;
   CUDA_ERROR_MAP_FAILED : constant cudaError_enum := 205;
   CUDA_ERROR_UNMAP_FAILED : constant cudaError_enum := 206;
   CUDA_ERROR_ARRAY_IS_MAPPED : constant cudaError_enum := 207;
   CUDA_ERROR_ALREADY_MAPPED : constant cudaError_enum := 208;
   CUDA_ERROR_NO_BINARY_FOR_GPU : constant cudaError_enum := 209;
   CUDA_ERROR_ALREADY_ACQUIRED : constant cudaError_enum := 210;
   CUDA_ERROR_NOT_MAPPED : constant cudaError_enum := 211;
   CUDA_ERROR_NOT_MAPPED_AS_ARRAY : constant cudaError_enum := 212;
   CUDA_ERROR_NOT_MAPPED_AS_POINTER : constant cudaError_enum := 213;
   CUDA_ERROR_ECC_UNCORRECTABLE : constant cudaError_enum := 214;
   CUDA_ERROR_UNSUPPORTED_LIMIT : constant cudaError_enum := 215;
   CUDA_ERROR_CONTEXT_ALREADY_IN_USE : constant cudaError_enum := 216;
   CUDA_ERROR_PEER_ACCESS_UNSUPPORTED : constant cudaError_enum := 217;
   CUDA_ERROR_INVALID_PTX : constant cudaError_enum := 218;
   CUDA_ERROR_INVALID_GRAPHICS_CONTEXT : constant cudaError_enum := 219;
   CUDA_ERROR_NVLINK_UNCORRECTABLE : constant cudaError_enum := 220;
   CUDA_ERROR_INVALID_SOURCE : constant cudaError_enum := 300;
   CUDA_ERROR_FILE_NOT_FOUND : constant cudaError_enum := 301;
   CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND : constant cudaError_enum := 302;
   CUDA_ERROR_SHARED_OBJECT_INIT_FAILED : constant cudaError_enum := 303;
   CUDA_ERROR_OPERATING_SYSTEM : constant cudaError_enum := 304;
   CUDA_ERROR_INVALID_HANDLE : constant cudaError_enum := 400;
   CUDA_ERROR_NOT_FOUND : constant cudaError_enum := 500;
   CUDA_ERROR_NOT_READY : constant cudaError_enum := 600;
   CUDA_ERROR_ILLEGAL_ADDRESS : constant cudaError_enum := 700;
   CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES : constant cudaError_enum := 701;
   CUDA_ERROR_LAUNCH_TIMEOUT : constant cudaError_enum := 702;
   CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING : constant cudaError_enum := 703;
   CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED : constant cudaError_enum := 704;
   CUDA_ERROR_PEER_ACCESS_NOT_ENABLED : constant cudaError_enum := 705;
   CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE : constant cudaError_enum := 708;
   CUDA_ERROR_CONTEXT_IS_DESTROYED : constant cudaError_enum := 709;
   CUDA_ERROR_ASSERT : constant cudaError_enum := 710;
   CUDA_ERROR_TOO_MANY_PEERS : constant cudaError_enum := 711;
   CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED : constant cudaError_enum := 712;
   CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED : constant cudaError_enum := 713;
   CUDA_ERROR_HARDWARE_STACK_ERROR : constant cudaError_enum := 714;
   CUDA_ERROR_ILLEGAL_INSTRUCTION : constant cudaError_enum := 715;
   CUDA_ERROR_MISALIGNED_ADDRESS : constant cudaError_enum := 716;
   CUDA_ERROR_INVALID_ADDRESS_SPACE : constant cudaError_enum := 717;
   CUDA_ERROR_INVALID_PC : constant cudaError_enum := 718;
   CUDA_ERROR_LAUNCH_FAILED : constant cudaError_enum := 719;
   CUDA_ERROR_NOT_PERMITTED : constant cudaError_enum := 800;
   CUDA_ERROR_NOT_SUPPORTED : constant cudaError_enum := 801;
   CUDA_ERROR_UNKNOWN : constant cudaError_enum := 999;  -- /usr/local/cuda-8.0/include/cuda.h:993

  --*
  --     * The API call returned with no errors. In the case of query calls, this
  --     * can also mean that the operation being queried is complete (see
  --     * ::cuEventQuery() and ::cuStreamQuery()).
  --      

  --*
  --     * This indicates that one or more of the parameters passed to the API call
  --     * is not within an acceptable range of values.
  --      

  --*
  --     * The API call failed because it was unable to allocate enough memory to
  --     * perform the requested operation.
  --      

  --*
  --     * This indicates that the CUDA driver has not been initialized with
  --     * ::cuInit() or that initialization has failed.
  --      

  --*
  --     * This indicates that the CUDA driver is in the process of shutting down.
  --      

  --*
  --     * This indicates profiler is not initialized for this run. This can
  --     * happen when the application is running with external profiling tools
  --     * like visual profiler.
  --      

  --*
  --     * \deprecated
  --     * This error return is deprecated as of CUDA 5.0. It is no longer an error
  --     * to attempt to enable/disable the profiling via ::cuProfilerStart or
  --     * ::cuProfilerStop without initialization.
  --      

  --*
  --     * \deprecated
  --     * This error return is deprecated as of CUDA 5.0. It is no longer an error
  --     * to call cuProfilerStart() when profiling is already enabled.
  --      

  --*
  --     * \deprecated
  --     * This error return is deprecated as of CUDA 5.0. It is no longer an error
  --     * to call cuProfilerStop() when profiling is already disabled.
  --      

  --*
  --     * This indicates that no CUDA-capable devices were detected by the installed
  --     * CUDA driver.
  --      

  --*
  --     * This indicates that the device ordinal supplied by the user does not
  --     * correspond to a valid CUDA device.
  --      

  --*
  --     * This indicates that the device kernel image is invalid. This can also
  --     * indicate an invalid CUDA module.
  --      

  --*
  --     * This most frequently indicates that there is no context bound to the
  --     * current thread. This can also be returned if the context passed to an
  --     * API call is not a valid handle (such as a context that has had
  --     * ::cuCtxDestroy() invoked on it). This can also be returned if a user
  --     * mixes different API versions (i.e. 3010 context with 3020 API calls).
  --     * See ::cuCtxGetApiVersion() for more details.
  --      

  --*
  --     * This indicated that the context being supplied as a parameter to the
  --     * API call was already the active context.
  --     * \deprecated
  --     * This error return is deprecated as of CUDA 3.2. It is no longer an
  --     * error to attempt to push the active context via ::cuCtxPushCurrent().
  --      

  --*
  --     * This indicates that a map or register operation has failed.
  --      

  --*
  --     * This indicates that an unmap or unregister operation has failed.
  --      

  --*
  --     * This indicates that the specified array is currently mapped and thus
  --     * cannot be destroyed.
  --      

  --*
  --     * This indicates that the resource is already mapped.
  --      

  --*
  --     * This indicates that there is no kernel image available that is suitable
  --     * for the device. This can occur when a user specifies code generation
  --     * options for a particular CUDA source file that do not include the
  --     * corresponding device configuration.
  --      

  --*
  --     * This indicates that a resource has already been acquired.
  --      

  --*
  --     * This indicates that a resource is not mapped.
  --      

  --*
  --     * This indicates that a mapped resource is not available for access as an
  --     * array.
  --      

  --*
  --     * This indicates that a mapped resource is not available for access as a
  --     * pointer.
  --      

  --*
  --     * This indicates that an uncorrectable ECC error was detected during
  --     * execution.
  --      

  --*
  --     * This indicates that the ::CUlimit passed to the API call is not
  --     * supported by the active device.
  --      

  --*
  --     * This indicates that the ::CUcontext passed to the API call can
  --     * only be bound to a single CPU thread at a time but is already 
  --     * bound to a CPU thread.
  --      

  --*
  --     * This indicates that peer access is not supported across the given
  --     * devices.
  --      

  --*
  --     * This indicates that a PTX JIT compilation failed.
  --      

  --*
  --     * This indicates an error with OpenGL or DirectX context.
  --      

  --*
  --    * This indicates that an uncorrectable NVLink error was detected during the
  --    * execution.
  --     

  --*
  --     * This indicates that the device kernel source is invalid.
  --      

  --*
  --     * This indicates that the file specified was not found.
  --      

  --*
  --     * This indicates that a link to a shared object failed to resolve.
  --      

  --*
  --     * This indicates that initialization of a shared object failed.
  --      

  --*
  --     * This indicates that an OS call failed.
  --      

  --*
  --     * This indicates that a resource handle passed to the API call was not
  --     * valid. Resource handles are opaque types like ::CUstream and ::CUevent.
  --      

  --*
  --     * This indicates that a named symbol was not found. Examples of symbols
  --     * are global/constant variable names, texture names, and surface names.
  --      

  --*
  --     * This indicates that asynchronous operations issued previously have not
  --     * completed yet. This result is not actually an error, but must be indicated
  --     * differently than ::CUDA_SUCCESS (which indicates completion). Calls that
  --     * may return this value include ::cuEventQuery() and ::cuStreamQuery().
  --      

  --*
  --     * While executing a kernel, the device encountered a
  --     * load or store instruction on an invalid memory address.
  --     * This leaves the process in an inconsistent state and any further CUDA work
  --     * will return the same error. To continue using CUDA, the process must be terminated
  --     * and relaunched.
  --      

  --*
  --     * This indicates that a launch did not occur because it did not have
  --     * appropriate resources. This error usually indicates that the user has
  --     * attempted to pass too many arguments to the device kernel, or the
  --     * kernel launch specifies too many threads for the kernel's register
  --     * count. Passing arguments of the wrong size (i.e. a 64-bit pointer
  --     * when a 32-bit int is expected) is equivalent to passing too many
  --     * arguments and can also result in this error.
  --      

  --*
  --     * This indicates that the device kernel took too long to execute. This can
  --     * only occur if timeouts are enabled - see the device attribute
  --     * ::CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT for more information.
  --     * This leaves the process in an inconsistent state and any further CUDA work
  --     * will return the same error. To continue using CUDA, the process must be terminated
  --     * and relaunched.
  --      

  --*
  --     * This error indicates a kernel launch that uses an incompatible texturing
  --     * mode.
  --      

  --*
  --     * This error indicates that a call to ::cuCtxEnablePeerAccess() is
  --     * trying to re-enable peer access to a context which has already
  --     * had peer access to it enabled.
  --      

  --*
  --     * This error indicates that ::cuCtxDisablePeerAccess() is 
  --     * trying to disable peer access which has not been enabled yet 
  --     * via ::cuCtxEnablePeerAccess(). 
  --      

  --*
  --     * This error indicates that the primary context for the specified device
  --     * has already been initialized.
  --      

  --*
  --     * This error indicates that the context current to the calling thread
  --     * has been destroyed using ::cuCtxDestroy, or is a primary context which
  --     * has not yet been initialized.
  --      

  --*
  --     * A device-side assert triggered during kernel execution. The context
  --     * cannot be used anymore, and must be destroyed. All existing device 
  --     * memory allocations from this context are invalid and must be 
  --     * reconstructed if the program is to continue using CUDA.
  --      

  --*
  --     * This error indicates that the hardware resources required to enable
  --     * peer access have been exhausted for one or more of the devices 
  --     * passed to ::cuCtxEnablePeerAccess().
  --      

  --*
  --     * This error indicates that the memory range passed to ::cuMemHostRegister()
  --     * has already been registered.
  --      

  --*
  --     * This error indicates that the pointer passed to ::cuMemHostUnregister()
  --     * does not correspond to any currently registered memory region.
  --      

  --*
  --     * While executing a kernel, the device encountered a stack error.
  --     * This can be due to stack corruption or exceeding the stack size limit.
  --     * This leaves the process in an inconsistent state and any further CUDA work
  --     * will return the same error. To continue using CUDA, the process must be terminated
  --     * and relaunched.
  --      

  --*
  --     * While executing a kernel, the device encountered an illegal instruction.
  --     * This leaves the process in an inconsistent state and any further CUDA work
  --     * will return the same error. To continue using CUDA, the process must be terminated
  --     * and relaunched.
  --      

  --*
  --     * While executing a kernel, the device encountered a load or store instruction
  --     * on a memory address which is not aligned.
  --     * This leaves the process in an inconsistent state and any further CUDA work
  --     * will return the same error. To continue using CUDA, the process must be terminated
  --     * and relaunched.
  --      

  --*
  --     * While executing a kernel, the device encountered an instruction
  --     * which can only operate on memory locations in certain address spaces
  --     * (global, shared, or local), but was supplied a memory address not
  --     * belonging to an allowed address space.
  --     * This leaves the process in an inconsistent state and any further CUDA work
  --     * will return the same error. To continue using CUDA, the process must be terminated
  --     * and relaunched.
  --      

  --*
  --     * While executing a kernel, the device program counter wrapped its address space.
  --     * This leaves the process in an inconsistent state and any further CUDA work
  --     * will return the same error. To continue using CUDA, the process must be terminated
  --     * and relaunched.
  --      

  --*
  --     * An exception occurred on the device while executing a kernel. Common
  --     * causes include dereferencing an invalid device pointer and accessing
  --     * out of bounds shared memory.
  --     * This leaves the process in an inconsistent state and any further CUDA work
  --     * will return the same error. To continue using CUDA, the process must be terminated
  --     * and relaunched.
  --      

  --*
  --     * This error indicates that the attempted operation is not permitted.
  --      

  --*
  --     * This error indicates that the attempted operation is not supported
  --     * on the current system or device.
  --      

  --*
  --     * This indicates that an unknown internal error has occurred.
  --      

   subtype CUresult is cudaError_enum;

  --*
  -- * P2P Attributes
  --  

   subtype CUdevice_P2PAttribute_enum is unsigned;
   CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK : constant CUdevice_P2PAttribute_enum := 1;
   CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED : constant CUdevice_P2PAttribute_enum := 2;
   CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED : constant CUdevice_P2PAttribute_enum := 3;  -- /usr/local/cuda-8.0/include/cuda.h:1394

  --*< A relative value indicating the performance of the link between two devices  
  --*< P2P Access is enable  
  --*< Atomic operation over the link supported  
   subtype CUdevice_P2PAttribute is CUdevice_P2PAttribute_enum;

  --*
  -- * CUDA stream callback
  -- * \param hStream The stream the callback was added to, as passed to ::cuStreamAddCallback.  May be NULL.
  -- * \param status ::CUDA_SUCCESS or any persistent error on the stream.
  -- * \param userData User parameter provided at registration.
  --  

   type CUstreamCallback is access procedure
        (arg1 : CUstream;
         arg2 : CUresult;
         arg3 : System.Address);
   pragma Convention (C, CUstreamCallback);  -- /usr/local/cuda-8.0/include/cuda.h:1412

  --*
  -- * Block size to per-block dynamic shared memory mapping for a certain
  -- * kernel \param blockSize Block size of the kernel.
  -- *
  -- * \return The dynamic shared memory needed by a block.
  --  

   type CUoccupancyB2DSize is access function (arg1 : int) return stddef_h.size_t;
   pragma Convention (C, CUoccupancyB2DSize);  -- /usr/local/cuda-8.0/include/cuda.h:1420

  --*
  -- * If set, host memory is portable between CUDA contexts.
  -- * Flag for ::cuMemHostAlloc()
  --  

  --*
  -- * If set, host memory is mapped into CUDA address space and
  -- * ::cuMemHostGetDevicePointer() may be called on the host pointer.
  -- * Flag for ::cuMemHostAlloc()
  --  

  --*
  -- * If set, host memory is allocated as write-combined - fast to write,
  -- * faster to DMA, slow to read except via SSE4 streaming load instruction
  -- * (MOVNTDQA).
  -- * Flag for ::cuMemHostAlloc()
  --  

  --*
  -- * If set, host memory is portable between CUDA contexts.
  -- * Flag for ::cuMemHostRegister()
  --  

  --*
  -- * If set, host memory is mapped into CUDA address space and
  -- * ::cuMemHostGetDevicePointer() may be called on the host pointer.
  -- * Flag for ::cuMemHostRegister()
  --  

  --*
  -- * If set, the passed memory pointer is treated as pointing to some
  -- * memory-mapped I/O space, e.g. belonging to a third-party PCIe device.
  -- * On Windows the flag is a no-op.
  -- * On Linux that memory is marked as non cache-coherent for the GPU and
  -- * is expected to be physically contiguous. It may return
  -- * CUDA_ERROR_NOT_PERMITTED if run as an unprivileged user,
  -- * CUDA_ERROR_NOT_SUPPORTED on older Linux kernel versions.
  -- * On all other platforms, it is not supported and CUDA_ERROR_NOT_SUPPORTED
  -- * is returned.
  -- * Flag for ::cuMemHostRegister()
  --  

  --*
  -- * 2D memory copy parameters
  --  

  --*< Source X in bytes  
   type CUDA_MEMCPY2D_st is record
      srcXInBytes : aliased stddef_h.size_t;  -- /usr/local/cuda-8.0/include/cuda.h:1476
      srcY : aliased stddef_h.size_t;  -- /usr/local/cuda-8.0/include/cuda.h:1477
      srcMemoryType : aliased CUmemorytype;  -- /usr/local/cuda-8.0/include/cuda.h:1479
      srcHost : System.Address;  -- /usr/local/cuda-8.0/include/cuda.h:1480
      srcDevice : aliased CUdeviceptr;  -- /usr/local/cuda-8.0/include/cuda.h:1481
      srcArray : CUarray;  -- /usr/local/cuda-8.0/include/cuda.h:1482
      srcPitch : aliased stddef_h.size_t;  -- /usr/local/cuda-8.0/include/cuda.h:1483
      dstXInBytes : aliased stddef_h.size_t;  -- /usr/local/cuda-8.0/include/cuda.h:1485
      dstY : aliased stddef_h.size_t;  -- /usr/local/cuda-8.0/include/cuda.h:1486
      dstMemoryType : aliased CUmemorytype;  -- /usr/local/cuda-8.0/include/cuda.h:1488
      dstHost : System.Address;  -- /usr/local/cuda-8.0/include/cuda.h:1489
      dstDevice : aliased CUdeviceptr;  -- /usr/local/cuda-8.0/include/cuda.h:1490
      dstArray : CUarray;  -- /usr/local/cuda-8.0/include/cuda.h:1491
      dstPitch : aliased stddef_h.size_t;  -- /usr/local/cuda-8.0/include/cuda.h:1492
      WidthInBytes : aliased stddef_h.size_t;  -- /usr/local/cuda-8.0/include/cuda.h:1494
      Height : aliased stddef_h.size_t;  -- /usr/local/cuda-8.0/include/cuda.h:1495
   end record;
   pragma Convention (C_Pass_By_Copy, CUDA_MEMCPY2D_st);  -- /usr/local/cuda-8.0/include/cuda.h:1475

  --*< Source Y  
  --*< Source memory type (host, device, array)  
  --*< Source host pointer  
  --*< Source device pointer  
  --*< Source array reference  
  --*< Source pitch (ignored when src is array)  
  --*< Destination X in bytes  
  --*< Destination Y  
  --*< Destination memory type (host, device, array)  
  --*< Destination host pointer  
  --*< Destination device pointer  
  --*< Destination array reference  
  --*< Destination pitch (ignored when dst is array)  
  --*< Width of 2D memory copy in bytes  
  --*< Height of 2D memory copy  
   subtype CUDA_MEMCPY2D is CUDA_MEMCPY2D_st;

  --*
  -- * 3D memory copy parameters
  --  

  --*< Source X in bytes  
   type CUDA_MEMCPY3D_st is record
      srcXInBytes : aliased stddef_h.size_t;  -- /usr/local/cuda-8.0/include/cuda.h:1502
      srcY : aliased stddef_h.size_t;  -- /usr/local/cuda-8.0/include/cuda.h:1503
      srcZ : aliased stddef_h.size_t;  -- /usr/local/cuda-8.0/include/cuda.h:1504
      srcLOD : aliased stddef_h.size_t;  -- /usr/local/cuda-8.0/include/cuda.h:1505
      srcMemoryType : aliased CUmemorytype;  -- /usr/local/cuda-8.0/include/cuda.h:1506
      srcHost : System.Address;  -- /usr/local/cuda-8.0/include/cuda.h:1507
      srcDevice : aliased CUdeviceptr;  -- /usr/local/cuda-8.0/include/cuda.h:1508
      srcArray : CUarray;  -- /usr/local/cuda-8.0/include/cuda.h:1509
      reserved0 : System.Address;  -- /usr/local/cuda-8.0/include/cuda.h:1510
      srcPitch : aliased stddef_h.size_t;  -- /usr/local/cuda-8.0/include/cuda.h:1511
      srcHeight : aliased stddef_h.size_t;  -- /usr/local/cuda-8.0/include/cuda.h:1512
      dstXInBytes : aliased stddef_h.size_t;  -- /usr/local/cuda-8.0/include/cuda.h:1514
      dstY : aliased stddef_h.size_t;  -- /usr/local/cuda-8.0/include/cuda.h:1515
      dstZ : aliased stddef_h.size_t;  -- /usr/local/cuda-8.0/include/cuda.h:1516
      dstLOD : aliased stddef_h.size_t;  -- /usr/local/cuda-8.0/include/cuda.h:1517
      dstMemoryType : aliased CUmemorytype;  -- /usr/local/cuda-8.0/include/cuda.h:1518
      dstHost : System.Address;  -- /usr/local/cuda-8.0/include/cuda.h:1519
      dstDevice : aliased CUdeviceptr;  -- /usr/local/cuda-8.0/include/cuda.h:1520
      dstArray : CUarray;  -- /usr/local/cuda-8.0/include/cuda.h:1521
      reserved1 : System.Address;  -- /usr/local/cuda-8.0/include/cuda.h:1522
      dstPitch : aliased stddef_h.size_t;  -- /usr/local/cuda-8.0/include/cuda.h:1523
      dstHeight : aliased stddef_h.size_t;  -- /usr/local/cuda-8.0/include/cuda.h:1524
      WidthInBytes : aliased stddef_h.size_t;  -- /usr/local/cuda-8.0/include/cuda.h:1526
      Height : aliased stddef_h.size_t;  -- /usr/local/cuda-8.0/include/cuda.h:1527
      Depth : aliased stddef_h.size_t;  -- /usr/local/cuda-8.0/include/cuda.h:1528
   end record;
   pragma Convention (C_Pass_By_Copy, CUDA_MEMCPY3D_st);  -- /usr/local/cuda-8.0/include/cuda.h:1501

  --*< Source Y  
  --*< Source Z  
  --*< Source LOD  
  --*< Source memory type (host, device, array)  
  --*< Source host pointer  
  --*< Source device pointer  
  --*< Source array reference  
  --*< Must be NULL  
  --*< Source pitch (ignored when src is array)  
  --*< Source height (ignored when src is array; may be 0 if Depth==1)  
  --*< Destination X in bytes  
  --*< Destination Y  
  --*< Destination Z  
  --*< Destination LOD  
  --*< Destination memory type (host, device, array)  
  --*< Destination host pointer  
  --*< Destination device pointer  
  --*< Destination array reference  
  --*< Must be NULL  
  --*< Destination pitch (ignored when dst is array)  
  --*< Destination height (ignored when dst is array; may be 0 if Depth==1)  
  --*< Width of 3D memory copy in bytes  
  --*< Height of 3D memory copy  
  --*< Depth of 3D memory copy  
   subtype CUDA_MEMCPY3D is CUDA_MEMCPY3D_st;

  --*
  -- * 3D memory cross-context copy parameters
  --  

  --*< Source X in bytes  
   type CUDA_MEMCPY3D_PEER_st is record
      srcXInBytes : aliased stddef_h.size_t;  -- /usr/local/cuda-8.0/include/cuda.h:1535
      srcY : aliased stddef_h.size_t;  -- /usr/local/cuda-8.0/include/cuda.h:1536
      srcZ : aliased stddef_h.size_t;  -- /usr/local/cuda-8.0/include/cuda.h:1537
      srcLOD : aliased stddef_h.size_t;  -- /usr/local/cuda-8.0/include/cuda.h:1538
      srcMemoryType : aliased CUmemorytype;  -- /usr/local/cuda-8.0/include/cuda.h:1539
      srcHost : System.Address;  -- /usr/local/cuda-8.0/include/cuda.h:1540
      srcDevice : aliased CUdeviceptr;  -- /usr/local/cuda-8.0/include/cuda.h:1541
      srcArray : CUarray;  -- /usr/local/cuda-8.0/include/cuda.h:1542
      srcContext : CUcontext;  -- /usr/local/cuda-8.0/include/cuda.h:1543
      srcPitch : aliased stddef_h.size_t;  -- /usr/local/cuda-8.0/include/cuda.h:1544
      srcHeight : aliased stddef_h.size_t;  -- /usr/local/cuda-8.0/include/cuda.h:1545
      dstXInBytes : aliased stddef_h.size_t;  -- /usr/local/cuda-8.0/include/cuda.h:1547
      dstY : aliased stddef_h.size_t;  -- /usr/local/cuda-8.0/include/cuda.h:1548
      dstZ : aliased stddef_h.size_t;  -- /usr/local/cuda-8.0/include/cuda.h:1549
      dstLOD : aliased stddef_h.size_t;  -- /usr/local/cuda-8.0/include/cuda.h:1550
      dstMemoryType : aliased CUmemorytype;  -- /usr/local/cuda-8.0/include/cuda.h:1551
      dstHost : System.Address;  -- /usr/local/cuda-8.0/include/cuda.h:1552
      dstDevice : aliased CUdeviceptr;  -- /usr/local/cuda-8.0/include/cuda.h:1553
      dstArray : CUarray;  -- /usr/local/cuda-8.0/include/cuda.h:1554
      dstContext : CUcontext;  -- /usr/local/cuda-8.0/include/cuda.h:1555
      dstPitch : aliased stddef_h.size_t;  -- /usr/local/cuda-8.0/include/cuda.h:1556
      dstHeight : aliased stddef_h.size_t;  -- /usr/local/cuda-8.0/include/cuda.h:1557
      WidthInBytes : aliased stddef_h.size_t;  -- /usr/local/cuda-8.0/include/cuda.h:1559
      Height : aliased stddef_h.size_t;  -- /usr/local/cuda-8.0/include/cuda.h:1560
      Depth : aliased stddef_h.size_t;  -- /usr/local/cuda-8.0/include/cuda.h:1561
   end record;
   pragma Convention (C_Pass_By_Copy, CUDA_MEMCPY3D_PEER_st);  -- /usr/local/cuda-8.0/include/cuda.h:1534

  --*< Source Y  
  --*< Source Z  
  --*< Source LOD  
  --*< Source memory type (host, device, array)  
  --*< Source host pointer  
  --*< Source device pointer  
  --*< Source array reference  
  --*< Source context (ignored with srcMemoryType is ::CU_MEMORYTYPE_ARRAY)  
  --*< Source pitch (ignored when src is array)  
  --*< Source height (ignored when src is array; may be 0 if Depth==1)  
  --*< Destination X in bytes  
  --*< Destination Y  
  --*< Destination Z  
  --*< Destination LOD  
  --*< Destination memory type (host, device, array)  
  --*< Destination host pointer  
  --*< Destination device pointer  
  --*< Destination array reference  
  --*< Destination context (ignored with dstMemoryType is ::CU_MEMORYTYPE_ARRAY)  
  --*< Destination pitch (ignored when dst is array)  
  --*< Destination height (ignored when dst is array; may be 0 if Depth==1)  
  --*< Width of 3D memory copy in bytes  
  --*< Height of 3D memory copy  
  --*< Depth of 3D memory copy  
   subtype CUDA_MEMCPY3D_PEER is CUDA_MEMCPY3D_PEER_st;

  --*
  -- * Array descriptor
  --  

  --*< Width of array  
   type CUDA_ARRAY_DESCRIPTOR_st is record
      Width : aliased stddef_h.size_t;  -- /usr/local/cuda-8.0/include/cuda.h:1569
      Height : aliased stddef_h.size_t;  -- /usr/local/cuda-8.0/include/cuda.h:1570
      Format : aliased CUarray_format;  -- /usr/local/cuda-8.0/include/cuda.h:1572
      NumChannels : aliased unsigned;  -- /usr/local/cuda-8.0/include/cuda.h:1573
   end record;
   pragma Convention (C_Pass_By_Copy, CUDA_ARRAY_DESCRIPTOR_st);  -- /usr/local/cuda-8.0/include/cuda.h:1567

  --*< Height of array  
  --*< Array format  
  --*< Channels per array element  
   subtype CUDA_ARRAY_DESCRIPTOR is CUDA_ARRAY_DESCRIPTOR_st;

  --*
  -- * 3D array descriptor
  --  

  --*< Width of 3D array  
   type CUDA_ARRAY3D_DESCRIPTOR_st is record
      Width : aliased stddef_h.size_t;  -- /usr/local/cuda-8.0/include/cuda.h:1581
      Height : aliased stddef_h.size_t;  -- /usr/local/cuda-8.0/include/cuda.h:1582
      Depth : aliased stddef_h.size_t;  -- /usr/local/cuda-8.0/include/cuda.h:1583
      Format : aliased CUarray_format;  -- /usr/local/cuda-8.0/include/cuda.h:1585
      NumChannels : aliased unsigned;  -- /usr/local/cuda-8.0/include/cuda.h:1586
      Flags : aliased unsigned;  -- /usr/local/cuda-8.0/include/cuda.h:1587
   end record;
   pragma Convention (C_Pass_By_Copy, CUDA_ARRAY3D_DESCRIPTOR_st);  -- /usr/local/cuda-8.0/include/cuda.h:1579

  --*< Height of 3D array  
  --*< Depth of 3D array  
  --*< Array format  
  --*< Channels per array element  
  --*< Flags  
   subtype CUDA_ARRAY3D_DESCRIPTOR is CUDA_ARRAY3D_DESCRIPTOR_st;

  --*
  -- * CUDA Resource descriptor
  --  

  --*< Resource type  
   type CUDA_RESOURCE_DESC_st;
   type anon_21;
   type anon_22 is record
      hArray : CUarray;  -- /usr/local/cuda-8.0/include/cuda.h:1603
   end record;
   pragma Convention (C_Pass_By_Copy, anon_22);
   type anon_23 is record
      hMipmappedArray : CUmipmappedArray;  -- /usr/local/cuda-8.0/include/cuda.h:1606
   end record;
   pragma Convention (C_Pass_By_Copy, anon_23);
   type anon_24 is record
      devPtr : aliased CUdeviceptr;  -- /usr/local/cuda-8.0/include/cuda.h:1609
      format : aliased CUarray_format;  -- /usr/local/cuda-8.0/include/cuda.h:1610
      numChannels : aliased unsigned;  -- /usr/local/cuda-8.0/include/cuda.h:1611
      sizeInBytes : aliased stddef_h.size_t;  -- /usr/local/cuda-8.0/include/cuda.h:1612
   end record;
   pragma Convention (C_Pass_By_Copy, anon_24);
   type anon_25 is record
      devPtr : aliased CUdeviceptr;  -- /usr/local/cuda-8.0/include/cuda.h:1615
      format : aliased CUarray_format;  -- /usr/local/cuda-8.0/include/cuda.h:1616
      numChannels : aliased unsigned;  -- /usr/local/cuda-8.0/include/cuda.h:1617
      width : aliased stddef_h.size_t;  -- /usr/local/cuda-8.0/include/cuda.h:1618
      height : aliased stddef_h.size_t;  -- /usr/local/cuda-8.0/include/cuda.h:1619
      pitchInBytes : aliased stddef_h.size_t;  -- /usr/local/cuda-8.0/include/cuda.h:1620
   end record;
   pragma Convention (C_Pass_By_Copy, anon_25);
   type anon_21_reserved_array is array (0 .. 31) of aliased int;
   type anon_26 is record
      reserved : aliased anon_21_reserved_array;  -- /usr/local/cuda-8.0/include/cuda.h:1623
   end record;
   pragma Convention (C_Pass_By_Copy, anon_26);
   type anon_21 (discr : unsigned := 0) is record
      case discr is
         when 0 =>
            c_array : aliased anon_22;  -- /usr/local/cuda-8.0/include/cuda.h:1604
         when 1 =>
            mipmap : aliased anon_23;  -- /usr/local/cuda-8.0/include/cuda.h:1607
         when 2 =>
            linear : aliased anon_24;  -- /usr/local/cuda-8.0/include/cuda.h:1613
         when 3 =>
            pitch2D : aliased anon_25;  -- /usr/local/cuda-8.0/include/cuda.h:1621
         when others =>
            reserved : aliased anon_26;  -- /usr/local/cuda-8.0/include/cuda.h:1624
      end case;
   end record;
   pragma Convention (C_Pass_By_Copy, anon_21);
   pragma Unchecked_Union (anon_21);type CUDA_RESOURCE_DESC_st is record
      resType : aliased CUresourcetype;  -- /usr/local/cuda-8.0/include/cuda.h:1599
      res : aliased anon_21;  -- /usr/local/cuda-8.0/include/cuda.h:1625
      flags : aliased unsigned;  -- /usr/local/cuda-8.0/include/cuda.h:1627
   end record;
   pragma Convention (C_Pass_By_Copy, CUDA_RESOURCE_DESC_st);  -- /usr/local/cuda-8.0/include/cuda.h:1597

  --*< CUDA array  
  --*< CUDA mipmapped array  
  --*< Device pointer  
  --*< Array format  
  --*< Channels per array element  
  --*< Size in bytes  
  --*< Device pointer  
  --*< Array format  
  --*< Channels per array element  
  --*< Width of the array in elements  
  --*< Height of the array in elements  
  --*< Pitch between two rows in bytes  
  --*< Flags (must be zero)  
   subtype CUDA_RESOURCE_DESC is CUDA_RESOURCE_DESC_st;

  --*
  -- * Texture descriptor
  --  

  --*< Address modes  
   type CUDA_TEXTURE_DESC_st_addressMode_array is array (0 .. 2) of aliased CUaddress_mode;
   type CUDA_TEXTURE_DESC_st_borderColor_array is array (0 .. 3) of aliased float;
   type CUDA_TEXTURE_DESC_st_reserved_array is array (0 .. 11) of aliased int;
   type CUDA_TEXTURE_DESC_st is record
      addressMode : aliased CUDA_TEXTURE_DESC_st_addressMode_array;  -- /usr/local/cuda-8.0/include/cuda.h:1634
      filterMode : aliased CUfilter_mode;  -- /usr/local/cuda-8.0/include/cuda.h:1635
      flags : aliased unsigned;  -- /usr/local/cuda-8.0/include/cuda.h:1636
      maxAnisotropy : aliased unsigned;  -- /usr/local/cuda-8.0/include/cuda.h:1637
      mipmapFilterMode : aliased CUfilter_mode;  -- /usr/local/cuda-8.0/include/cuda.h:1638
      mipmapLevelBias : aliased float;  -- /usr/local/cuda-8.0/include/cuda.h:1639
      minMipmapLevelClamp : aliased float;  -- /usr/local/cuda-8.0/include/cuda.h:1640
      maxMipmapLevelClamp : aliased float;  -- /usr/local/cuda-8.0/include/cuda.h:1641
      borderColor : aliased CUDA_TEXTURE_DESC_st_borderColor_array;  -- /usr/local/cuda-8.0/include/cuda.h:1642
      reserved : aliased CUDA_TEXTURE_DESC_st_reserved_array;  -- /usr/local/cuda-8.0/include/cuda.h:1643
   end record;
   pragma Convention (C_Pass_By_Copy, CUDA_TEXTURE_DESC_st);  -- /usr/local/cuda-8.0/include/cuda.h:1633

  --*< Filter mode  
  --*< Flags  
  --*< Maximum anisotropy ratio  
  --*< Mipmap filter mode  
  --*< Mipmap level bias  
  --*< Mipmap minimum level clamp  
  --*< Mipmap maximum level clamp  
  --*< Border Color  
   subtype CUDA_TEXTURE_DESC is CUDA_TEXTURE_DESC_st;

  --*
  -- * Resource view format
  --  

   type CUresourceViewFormat_enum is 
     (CU_RES_VIEW_FORMAT_NONE,
      CU_RES_VIEW_FORMAT_UINT_1X8,
      CU_RES_VIEW_FORMAT_UINT_2X8,
      CU_RES_VIEW_FORMAT_UINT_4X8,
      CU_RES_VIEW_FORMAT_SINT_1X8,
      CU_RES_VIEW_FORMAT_SINT_2X8,
      CU_RES_VIEW_FORMAT_SINT_4X8,
      CU_RES_VIEW_FORMAT_UINT_1X16,
      CU_RES_VIEW_FORMAT_UINT_2X16,
      CU_RES_VIEW_FORMAT_UINT_4X16,
      CU_RES_VIEW_FORMAT_SINT_1X16,
      CU_RES_VIEW_FORMAT_SINT_2X16,
      CU_RES_VIEW_FORMAT_SINT_4X16,
      CU_RES_VIEW_FORMAT_UINT_1X32,
      CU_RES_VIEW_FORMAT_UINT_2X32,
      CU_RES_VIEW_FORMAT_UINT_4X32,
      CU_RES_VIEW_FORMAT_SINT_1X32,
      CU_RES_VIEW_FORMAT_SINT_2X32,
      CU_RES_VIEW_FORMAT_SINT_4X32,
      CU_RES_VIEW_FORMAT_FLOAT_1X16,
      CU_RES_VIEW_FORMAT_FLOAT_2X16,
      CU_RES_VIEW_FORMAT_FLOAT_4X16,
      CU_RES_VIEW_FORMAT_FLOAT_1X32,
      CU_RES_VIEW_FORMAT_FLOAT_2X32,
      CU_RES_VIEW_FORMAT_FLOAT_4X32,
      CU_RES_VIEW_FORMAT_UNSIGNED_BC1,
      CU_RES_VIEW_FORMAT_UNSIGNED_BC2,
      CU_RES_VIEW_FORMAT_UNSIGNED_BC3,
      CU_RES_VIEW_FORMAT_UNSIGNED_BC4,
      CU_RES_VIEW_FORMAT_SIGNED_BC4,
      CU_RES_VIEW_FORMAT_UNSIGNED_BC5,
      CU_RES_VIEW_FORMAT_SIGNED_BC5,
      CU_RES_VIEW_FORMAT_UNSIGNED_BC6H,
      CU_RES_VIEW_FORMAT_SIGNED_BC6H,
      CU_RES_VIEW_FORMAT_UNSIGNED_BC7);
   pragma Convention (C, CUresourceViewFormat_enum);  -- /usr/local/cuda-8.0/include/cuda.h:1649

  --*< No resource view format (use underlying resource format)  
  --*< 1 channel unsigned 8-bit integers  
  --*< 2 channel unsigned 8-bit integers  
  --*< 4 channel unsigned 8-bit integers  
  --*< 1 channel signed 8-bit integers  
  --*< 2 channel signed 8-bit integers  
  --*< 4 channel signed 8-bit integers  
  --*< 1 channel unsigned 16-bit integers  
  --*< 2 channel unsigned 16-bit integers  
  --*< 4 channel unsigned 16-bit integers  
  --*< 1 channel signed 16-bit integers  
  --*< 2 channel signed 16-bit integers  
  --*< 4 channel signed 16-bit integers  
  --*< 1 channel unsigned 32-bit integers  
  --*< 2 channel unsigned 32-bit integers  
  --*< 4 channel unsigned 32-bit integers  
  --*< 1 channel signed 32-bit integers  
  --*< 2 channel signed 32-bit integers  
  --*< 4 channel signed 32-bit integers  
  --*< 1 channel 16-bit floating point  
  --*< 2 channel 16-bit floating point  
  --*< 4 channel 16-bit floating point  
  --*< 1 channel 32-bit floating point  
  --*< 2 channel 32-bit floating point  
  --*< 4 channel 32-bit floating point  
  --*< Block compressed 1  
  --*< Block compressed 2  
  --*< Block compressed 3  
  --*< Block compressed 4 unsigned  
  --*< Block compressed 4 signed  
  --*< Block compressed 5 unsigned  
  --*< Block compressed 5 signed  
  --*< Block compressed 6 unsigned half-float  
  --*< Block compressed 6 signed half-float  
  --*< Block compressed 7  
   subtype CUresourceViewFormat is CUresourceViewFormat_enum;

  --*
  -- * Resource view descriptor
  --  

  --*< Resource view format  
   type CUDA_RESOURCE_VIEW_DESC_st_reserved_array is array (0 .. 15) of aliased unsigned;
   type CUDA_RESOURCE_VIEW_DESC_st is record
      format : aliased CUresourceViewFormat;  -- /usr/local/cuda-8.0/include/cuda.h:1693
      width : aliased stddef_h.size_t;  -- /usr/local/cuda-8.0/include/cuda.h:1694
      height : aliased stddef_h.size_t;  -- /usr/local/cuda-8.0/include/cuda.h:1695
      depth : aliased stddef_h.size_t;  -- /usr/local/cuda-8.0/include/cuda.h:1696
      firstMipmapLevel : aliased unsigned;  -- /usr/local/cuda-8.0/include/cuda.h:1697
      lastMipmapLevel : aliased unsigned;  -- /usr/local/cuda-8.0/include/cuda.h:1698
      firstLayer : aliased unsigned;  -- /usr/local/cuda-8.0/include/cuda.h:1699
      lastLayer : aliased unsigned;  -- /usr/local/cuda-8.0/include/cuda.h:1700
      reserved : aliased CUDA_RESOURCE_VIEW_DESC_st_reserved_array;  -- /usr/local/cuda-8.0/include/cuda.h:1701
   end record;
   pragma Convention (C_Pass_By_Copy, CUDA_RESOURCE_VIEW_DESC_st);  -- /usr/local/cuda-8.0/include/cuda.h:1691

  --*< Width of the resource view  
  --*< Height of the resource view  
  --*< Depth of the resource view  
  --*< First defined mipmap level  
  --*< Last defined mipmap level  
  --*< First layer index  
  --*< Last layer index  
   subtype CUDA_RESOURCE_VIEW_DESC is CUDA_RESOURCE_VIEW_DESC_st;

  --*
  -- * GPU Direct v3 tokens
  --  

   type CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_st is record
      p2pToken : aliased Extensions.unsigned_long_long;  -- /usr/local/cuda-8.0/include/cuda.h:1708
      vaSpaceToken : aliased unsigned;  -- /usr/local/cuda-8.0/include/cuda.h:1709
   end record;
   pragma Convention (C_Pass_By_Copy, CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_st);  -- /usr/local/cuda-8.0/include/cuda.h:1707

   subtype CUDA_POINTER_ATTRIBUTE_P2P_TOKENS is CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_st;

  --*
  -- * If set, the CUDA array is a collection of layers, where each layer is either a 1D
  -- * or a 2D array and the Depth member of CUDA_ARRAY3D_DESCRIPTOR specifies the number 
  -- * of layers, not the depth of a 3D array.
  --  

  --*
  -- * Deprecated, use CUDA_ARRAY3D_LAYERED
  --  

  --*
  -- * This flag must be set in order to bind a surface reference
  -- * to the CUDA array
  --  

  --*
  -- * If set, the CUDA array is a collection of six 2D arrays, representing faces of a cube. The
  -- * width of such a CUDA array must be equal to its height, and Depth must be six.
  -- * If ::CUDA_ARRAY3D_LAYERED flag is also set, then the CUDA array is a collection of cubemaps
  -- * and Depth must be a multiple of six.
  --  

  --*
  -- * This flag must be set in order to perform texture gather operations
  -- * on a CUDA array.
  --  

  --*
  -- * This flag if set indicates that the CUDA
  -- * array is a DEPTH_TEXTURE.
  -- 

  --*
  -- * Override the texref format with a format inferred from the array.
  -- * Flag for ::cuTexRefSetArray()
  --  

  --*
  -- * Read the texture as integers rather than promoting the values to floats
  -- * in the range [0,1].
  -- * Flag for ::cuTexRefSetFlags()
  --  

  --*
  -- * Use normalized texture coordinates in the range [0,1) instead of [0,dim).
  -- * Flag for ::cuTexRefSetFlags()
  --  

  --*
  -- * Perform sRGB->linear conversion during texture read.
  -- * Flag for ::cuTexRefSetFlags()
  --  

  --*
  -- * End of array terminator for the \p extra parameter to
  -- * ::cuLaunchKernel
  --  

  --*
  -- * Indicator that the next value in the \p extra parameter to
  -- * ::cuLaunchKernel will be a pointer to a buffer containing all kernel
  -- * parameters used for launching kernel \p f.  This buffer needs to
  -- * honor all alignment/padding requirements of the individual parameters.
  -- * If ::CU_LAUNCH_PARAM_BUFFER_SIZE is not also specified in the
  -- * \p extra array, then ::CU_LAUNCH_PARAM_BUFFER_POINTER will have no
  -- * effect.
  --  

  --*
  -- * Indicator that the next value in the \p extra parameter to
  -- * ::cuLaunchKernel will be a pointer to a size_t which contains the
  -- * size of the buffer specified with ::CU_LAUNCH_PARAM_BUFFER_POINTER.
  -- * It is required that ::CU_LAUNCH_PARAM_BUFFER_POINTER also be specified
  -- * in the \p extra array if the value associated with
  -- * ::CU_LAUNCH_PARAM_BUFFER_SIZE is not zero.
  --  

  --*
  -- * For texture references loaded into the module, use default texunit from
  -- * texture reference.
  --  

  --*
  -- * Device that represents the CPU
  --  

  --*
  -- * Device that represents an invalid device
  --  

  --* @}  
  -- END CUDA_TYPES  
  --*
  -- * \defgroup CUDA_ERROR Error Handling
  -- *
  -- * ___MANBRIEF___ error handling functions of the low-level CUDA driver API
  -- * (___CURRENT_FILE___) ___ENDMANBRIEF___
  -- *
  -- * This section describes the error handling functions of the low-level CUDA
  -- * driver application programming interface.
  -- *
  -- * @{
  --  

  --*
  -- * \brief Gets the string description of an error code
  -- *
  -- * Sets \p *pStr to the address of a NULL-terminated string description
  -- * of the error code \p error.
  -- * If the error code is not recognized, ::CUDA_ERROR_INVALID_VALUE
  -- * will be returned and \p *pStr will be set to the NULL address.
  -- *
  -- * \param error - Error code to convert to string
  -- * \param pStr - Address of the string pointer.
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- *
  -- * \sa ::CUresult
  --  

   function cuGetErrorString (error : CUresult; pStr : System.Address) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:1857
   pragma Import (C, cuGetErrorString, "cuGetErrorString");

  --*
  -- * \brief Gets the string representation of an error code enum name
  -- *
  -- * Sets \p *pStr to the address of a NULL-terminated string representation
  -- * of the name of the enum error code \p error.
  -- * If the error code is not recognized, ::CUDA_ERROR_INVALID_VALUE
  -- * will be returned and \p *pStr will be set to the NULL address.
  -- *
  -- * \param error - Error code to convert to string
  -- * \param pStr - Address of the string pointer.
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- *
  -- * \sa ::CUresult
  --  

   function cuGetErrorName (error : CUresult; pStr : System.Address) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:1876
   pragma Import (C, cuGetErrorName, "cuGetErrorName");

  --* @}  
  -- END CUDA_ERROR  
  --*
  -- * \defgroup CUDA_INITIALIZE Initialization
  -- *
  -- * ___MANBRIEF___ initialization functions of the low-level CUDA driver API
  -- * (___CURRENT_FILE___) ___ENDMANBRIEF___
  -- *
  -- * This section describes the initialization functions of the low-level CUDA
  -- * driver application programming interface.
  -- *
  -- * @{
  --  

  --*
  -- * \brief Initialize the CUDA driver API
  -- *
  -- * Initializes the driver API and must be called before any other function from
  -- * the driver API. Currently, the \p Flags parameter must be 0. If ::cuInit()
  -- * has not been called, any function from the driver API will return
  -- * ::CUDA_ERROR_NOT_INITIALIZED.
  -- *
  -- * \param Flags - Initialization flag for CUDA.
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_INVALID_VALUE,
  -- * ::CUDA_ERROR_INVALID_DEVICE
  -- * \notefnerr
  --  

   function cuInit (Flags : unsigned) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:1908
   pragma Import (C, cuInit, "cuInit");

  --* @}  
  -- END CUDA_INITIALIZE  
  --*
  -- * \defgroup CUDA_VERSION Version Management
  -- *
  -- * ___MANBRIEF___ version management functions of the low-level CUDA driver
  -- * API (___CURRENT_FILE___) ___ENDMANBRIEF___
  -- *
  -- * This section describes the version management functions of the low-level
  -- * CUDA driver application programming interface.
  -- *
  -- * @{
  --  

  --*
  -- * \brief Returns the CUDA driver version
  -- *
  -- * Returns in \p *driverVersion the version number of the installed CUDA
  -- * driver. This function automatically returns ::CUDA_ERROR_INVALID_VALUE if
  -- * the \p driverVersion argument is NULL.
  -- *
  -- * \param driverVersion - Returns the CUDA driver version
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- * \notefnerr
  --  

   function cuDriverGetVersion (driverVersion : access int) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:1938
   pragma Import (C, cuDriverGetVersion, "cuDriverGetVersion");

  --* @}  
  -- END CUDA_VERSION  
  --*
  -- * \defgroup CUDA_DEVICE Device Management
  -- *
  -- * ___MANBRIEF___ device management functions of the low-level CUDA driver API
  -- * (___CURRENT_FILE___) ___ENDMANBRIEF___
  -- *
  -- * This section describes the device management functions of the low-level
  -- * CUDA driver application programming interface.
  -- *
  -- * @{
  --  

  --*
  -- * \brief Returns a handle to a compute device
  -- *
  -- * Returns in \p *device a device handle given an ordinal in the range <b>[0,
  -- * ::cuDeviceGetCount()-1]</b>.
  -- *
  -- * \param device  - Returned device handle
  -- * \param ordinal - Device number to get handle for
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE,
  -- * ::CUDA_ERROR_INVALID_DEVICE
  -- * \notefnerr
  -- *
  -- * \sa
  -- * ::cuDeviceGetAttribute,
  -- * ::cuDeviceGetCount,
  -- * ::cuDeviceGetName,
  -- * ::cuDeviceTotalMem
  --  

   function cuDeviceGet (device : access CUdevice; ordinal : int) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:1978
   pragma Import (C, cuDeviceGet, "cuDeviceGet");

  --*
  -- * \brief Returns the number of compute-capable devices
  -- *
  -- * Returns in \p *count the number of devices with compute capability greater
  -- * than or equal to 1.0 that are available for execution. If there is no such
  -- * device, ::cuDeviceGetCount() returns 0.
  -- *
  -- * \param count - Returned number of compute-capable devices
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- * \notefnerr
  -- *
  -- * \sa
  -- * ::cuDeviceGetAttribute,
  -- * ::cuDeviceGetName,
  -- * ::cuDeviceGet,
  -- * ::cuDeviceTotalMem
  --  

   function cuDeviceGetCount (count : access int) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:2003
   pragma Import (C, cuDeviceGetCount, "cuDeviceGetCount");

  --*
  -- * \brief Returns an identifer string for the device
  -- *
  -- * Returns an ASCII string identifying the device \p dev in the NULL-terminated
  -- * string pointed to by \p name. \p len specifies the maximum length of the
  -- * string that may be returned.
  -- *
  -- * \param name - Returned identifier string for the device
  -- * \param len  - Maximum length of string to store in \p name
  -- * \param dev  - Device to get identifier string for
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE,
  -- * ::CUDA_ERROR_INVALID_DEVICE
  -- * \notefnerr
  -- *
  -- * \sa
  -- * ::cuDeviceGetAttribute,
  -- * ::cuDeviceGetCount,
  -- * ::cuDeviceGet,
  -- * ::cuDeviceTotalMem
  --  

   function cuDeviceGetName
     (name : Interfaces.C.Strings.chars_ptr;
      len : int;
      dev : CUdevice) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:2031
   pragma Import (C, cuDeviceGetName, "cuDeviceGetName");

  --*
  -- * \brief Returns the total amount of memory on the device
  -- *
  -- * Returns in \p *bytes the total amount of memory available on the device
  -- * \p dev in bytes.
  -- *
  -- * \param bytes - Returned memory available on device in bytes
  -- * \param dev   - Device handle
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE,
  -- * ::CUDA_ERROR_INVALID_DEVICE
  -- * \notefnerr
  -- *
  -- * \sa
  -- * ::cuDeviceGetAttribute,
  -- * ::cuDeviceGetCount,
  -- * ::cuDeviceGetName,
  -- * ::cuDeviceGet,
  --  

   function cuDeviceTotalMem_v2 (bytes : access stddef_h.size_t; dev : CUdevice) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:2058
   pragma Import (C, cuDeviceTotalMem_v2, "cuDeviceTotalMem_v2");

  --*
  -- * \brief Returns information about the device
  -- *
  -- * Returns in \p *pi the integer value of the attribute \p attrib on device
  -- * \p dev. The supported attributes are:
  -- * - ::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK: Maximum number of threads per
  -- *   block;
  -- * - ::CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X: Maximum x-dimension of a block;
  -- * - ::CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y: Maximum y-dimension of a block;
  -- * - ::CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z: Maximum z-dimension of a block;
  -- * - ::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X: Maximum x-dimension of a grid;
  -- * - ::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y: Maximum y-dimension of a grid;
  -- * - ::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z: Maximum z-dimension of a grid;
  -- * - ::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK: Maximum amount of
  -- *   shared memory available to a thread block in bytes;
  -- * - ::CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY: Memory available on device for
  -- *   __constant__ variables in a CUDA C kernel in bytes;
  -- * - ::CU_DEVICE_ATTRIBUTE_WARP_SIZE: Warp size in threads;
  -- * - ::CU_DEVICE_ATTRIBUTE_MAX_PITCH: Maximum pitch in bytes allowed by the
  -- *   memory copy functions that involve memory regions allocated through
  -- *   ::cuMemAllocPitch();
  -- * - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH: Maximum 1D 
  -- *  texture width;
  -- * - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH: Maximum width
  -- *  for a 1D texture bound to linear memory;
  -- * - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH: Maximum 
  -- *  mipmapped 1D texture width;
  -- * - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH: Maximum 2D 
  -- *  texture width;
  -- * - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT: Maximum 2D 
  -- *  texture height;
  -- * - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH: Maximum width
  -- *  for a 2D texture bound to linear memory;
  -- * - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT: Maximum height
  -- *  for a 2D texture bound to linear memory;
  -- * - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH: Maximum pitch
  -- *  in bytes for a 2D texture bound to linear memory;
  -- * - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH: Maximum 
  -- *  mipmapped 2D texture width;
  -- * - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT: Maximum
  -- *  mipmapped 2D texture height;
  -- * - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH: Maximum 3D 
  -- *  texture width;
  -- * - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT: Maximum 3D 
  -- *  texture height;
  -- * - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH: Maximum 3D 
  -- *  texture depth;
  -- * - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE: 
  -- *  Alternate maximum 3D texture width, 0 if no alternate
  -- *  maximum 3D texture size is supported;
  -- * - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE: 
  -- *  Alternate maximum 3D texture height, 0 if no alternate
  -- *  maximum 3D texture size is supported;
  -- * - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE: 
  -- *  Alternate maximum 3D texture depth, 0 if no alternate
  -- *  maximum 3D texture size is supported;
  -- * - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH:
  -- *  Maximum cubemap texture width or height;
  -- * - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH: 
  -- *  Maximum 1D layered texture width;
  -- * - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS: 
  -- *   Maximum layers in a 1D layered texture;
  -- * - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH: 
  -- *  Maximum 2D layered texture width;
  -- * - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT: 
  -- *   Maximum 2D layered texture height;
  -- * - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS: 
  -- *   Maximum layers in a 2D layered texture;
  -- * - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH: 
  -- *   Maximum cubemap layered texture width or height;
  -- * - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS: 
  -- *   Maximum layers in a cubemap layered texture;
  -- * - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH:
  -- *   Maximum 1D surface width;
  -- * - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH:
  -- *   Maximum 2D surface width;
  -- * - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT:
  -- *   Maximum 2D surface height;
  -- * - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH:
  -- *   Maximum 3D surface width;
  -- * - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT:
  -- *   Maximum 3D surface height;
  -- * - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH:
  -- *   Maximum 3D surface depth;
  -- * - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH:
  -- *   Maximum 1D layered surface width;
  -- * - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS:
  -- *   Maximum layers in a 1D layered surface;
  -- * - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH:
  -- *   Maximum 2D layered surface width;
  -- * - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT:
  -- *   Maximum 2D layered surface height;
  -- * - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS:
  -- *   Maximum layers in a 2D layered surface;
  -- * - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH:
  -- *   Maximum cubemap surface width;
  -- * - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH:
  -- *   Maximum cubemap layered surface width;
  -- * - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS:
  -- *   Maximum layers in a cubemap layered surface;
  -- * - ::CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK: Maximum number of 32-bit
  -- *   registers available to a thread block;
  -- * - ::CU_DEVICE_ATTRIBUTE_CLOCK_RATE: The typical clock frequency in kilohertz;
  -- * - ::CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT: Alignment requirement; texture
  -- *   base addresses aligned to ::textureAlign bytes do not need an offset
  -- *   applied to texture fetches;
  -- * - ::CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT: Pitch alignment requirement
  -- *   for 2D texture references bound to pitched memory;
  -- * - ::CU_DEVICE_ATTRIBUTE_GPU_OVERLAP: 1 if the device can concurrently copy
  -- *   memory between host and device while executing a kernel, or 0 if not;
  -- * - ::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: Number of multiprocessors on
  -- *   the device;
  -- * - ::CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT: 1 if there is a run time limit
  -- *   for kernels executed on the device, or 0 if not;
  -- * - ::CU_DEVICE_ATTRIBUTE_INTEGRATED: 1 if the device is integrated with the
  -- *   memory subsystem, or 0 if not;
  -- * - ::CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY: 1 if the device can map host
  -- *   memory into the CUDA address space, or 0 if not;
  -- * - ::CU_DEVICE_ATTRIBUTE_COMPUTE_MODE: Compute mode that device is currently
  -- *   in. Available modes are as follows:
  -- *   - ::CU_COMPUTEMODE_DEFAULT: Default mode - Device is not restricted and
  -- *     can have multiple CUDA contexts present at a single time.
  -- *   - ::CU_COMPUTEMODE_PROHIBITED: Compute-prohibited mode - Device is
  -- *     prohibited from creating new CUDA contexts.
  -- *   - ::CU_COMPUTEMODE_EXCLUSIVE_PROCESS:  Compute-exclusive-process mode - Device
  -- *     can have only one context used by a single process at a time.
  -- * - ::CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS: 1 if the device supports
  -- *   executing multiple kernels within the same context simultaneously, or 0 if
  -- *   not. It is not guaranteed that multiple kernels will be resident
  -- *   on the device concurrently so this feature should not be relied upon for
  -- *   correctness;
  -- * - ::CU_DEVICE_ATTRIBUTE_ECC_ENABLED: 1 if error correction is enabled on the
  -- *    device, 0 if error correction is disabled or not supported by the device;
  -- * - ::CU_DEVICE_ATTRIBUTE_PCI_BUS_ID: PCI bus identifier of the device;
  -- * - ::CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID: PCI device (also known as slot) identifier
  -- *   of the device;
  -- * - ::CU_DEVICE_ATTRIBUTE_TCC_DRIVER: 1 if the device is using a TCC driver. TCC
  -- *    is only available on Tesla hardware running Windows Vista or later;
  -- * - ::CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE: Peak memory clock frequency in kilohertz;
  -- * - ::CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH: Global memory bus width in bits;
  -- * - ::CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE: Size of L2 cache in bytes. 0 if the device doesn't have L2 cache;
  -- * - ::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR: Maximum resident threads per multiprocessor;
  -- * - ::CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING: 1 if the device shares a unified address space with 
  -- *   the host, or 0 if not;
  -- * - ::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: Major compute capability version number;
  -- * - ::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: Minor compute capability version number;
  -- * - ::CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED: 1 if device supports caching globals 
  -- *    in L1 cache, 0 if caching globals in L1 cache is not supported by the device;
  -- * - ::CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED: 1 if device supports caching locals 
  -- *    in L1 cache, 0 if caching locals in L1 cache is not supported by the device;
  -- * - ::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR: Maximum amount of
  -- *   shared memory available to a multiprocessor in bytes; this amount is shared
  -- *   by all thread blocks simultaneously resident on a multiprocessor;
  -- * - ::CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR: Maximum number of 32-bit
  -- *   registers available to a multiprocessor; this number is shared by all thread
  -- *   blocks simultaneously resident on a multiprocessor;
  -- * - ::CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY: 1 if device supports allocating managed memory
  -- *   on this system, 0 if allocating managed memory is not supported by the device on this system.
  -- * - ::CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD: 1 if device is on a multi-GPU board, 0 if not.
  -- * - ::CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID: Unique identifier for a group of devices
  -- *   associated with the same board. Devices on the same multi-GPU board will share the same identifier.
  -- * - ::CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED: 1 if Link between the device and the host
  -- *   supports native atomic operations.
  -- * - ::CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO: Ratio of single precision performance
  -- *   (in floating-point operations per second) to double precision performance.
  -- * - ::CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS: Device suppports coherently accessing
  -- *   pageable memory without calling cudaHostRegister on it.
  -- * - ::CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS: Device can coherently access managed memory
  -- *   concurrently with the CPU.
  -- * - ::CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED: Device supports Compute Preemption.
  -- * - ::CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM: Device can access host registered
  -- *   memory at the same virtual address as the CPU.
  -- *
  -- * \param pi     - Returned device attribute value
  -- * \param attrib - Device attribute to query
  -- * \param dev    - Device handle
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE,
  -- * ::CUDA_ERROR_INVALID_DEVICE
  -- * \notefnerr
  -- *
  -- * \sa
  -- * ::cuDeviceGetCount,
  -- * ::cuDeviceGetName,
  -- * ::cuDeviceGet,
  -- * ::cuDeviceTotalMem
  --  

   function cuDeviceGetAttribute
     (pi : access int;
      attrib : CUdevice_attribute;
      dev : CUdevice) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:2253
   pragma Import (C, cuDeviceGetAttribute, "cuDeviceGetAttribute");

  --* @}  
  -- END CUDA_DEVICE  
  --*
  -- * \defgroup CUDA_DEVICE_DEPRECATED Device Management [DEPRECATED]
  -- *
  -- * ___MANBRIEF___ deprecated device management functions of the low-level CUDA
  -- * driver API (___CURRENT_FILE___) ___ENDMANBRIEF___
  -- *
  -- * This section describes the device management functions of the low-level
  -- * CUDA driver application programming interface.
  -- *
  -- * @{
  --  

  --*
  -- * \brief Returns properties for a selected device
  -- *
  -- * \deprecated
  -- *
  -- * This function was deprecated as of CUDA 5.0 and replaced by ::cuDeviceGetAttribute().
  -- *
  -- * Returns in \p *prop the properties of device \p dev. The ::CUdevprop
  -- * structure is defined as:
  -- *
  -- * \code
  --     typedef struct CUdevprop_st {
  --     int maxThreadsPerBlock;
  --     int maxThreadsDim[3];
  --     int maxGridSize[3];
  --     int sharedMemPerBlock;
  --     int totalConstantMemory;
  --     int SIMDWidth;
  --     int memPitch;
  --     int regsPerBlock;
  --     int clockRate;
  --     int textureAlign
  --  } CUdevprop;
  -- * \endcode
  -- * where:
  -- *
  -- * - ::maxThreadsPerBlock is the maximum number of threads per block;
  -- * - ::maxThreadsDim[3] is the maximum sizes of each dimension of a block;
  -- * - ::maxGridSize[3] is the maximum sizes of each dimension of a grid;
  -- * - ::sharedMemPerBlock is the total amount of shared memory available per
  -- *   block in bytes;
  -- * - ::totalConstantMemory is the total amount of constant memory available on
  -- *   the device in bytes;
  -- * - ::SIMDWidth is the warp size;
  -- * - ::memPitch is the maximum pitch allowed by the memory copy functions that
  -- *   involve memory regions allocated through ::cuMemAllocPitch();
  -- * - ::regsPerBlock is the total number of registers available per block;
  -- * - ::clockRate is the clock frequency in kilohertz;
  -- * - ::textureAlign is the alignment requirement; texture base addresses that
  -- *   are aligned to ::textureAlign bytes do not need an offset applied to
  -- *   texture fetches.
  -- *
  -- * \param prop - Returned properties of device
  -- * \param dev  - Device to get properties for
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE,
  -- * ::CUDA_ERROR_INVALID_DEVICE
  -- * \notefnerr
  -- *
  -- * \sa
  -- * ::cuDeviceGetAttribute,
  -- * ::cuDeviceGetCount,
  -- * ::cuDeviceGetName,
  -- * ::cuDeviceGet,
  -- * ::cuDeviceTotalMem
  --  

   function cuDeviceGetProperties (prop : access CUdevprop; dev : CUdevice) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:2330
   pragma Import (C, cuDeviceGetProperties, "cuDeviceGetProperties");

  --*
  -- * \brief Returns the compute capability of the device
  -- *
  -- * \deprecated
  -- *
  -- * This function was deprecated as of CUDA 5.0 and its functionality superceded
  -- * by ::cuDeviceGetAttribute(). 
  -- *
  -- * Returns in \p *major and \p *minor the major and minor revision numbers that
  -- * define the compute capability of the device \p dev.
  -- *
  -- * \param major - Major revision number
  -- * \param minor - Minor revision number
  -- * \param dev   - Device handle
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE,
  -- * ::CUDA_ERROR_INVALID_DEVICE
  -- * \notefnerr
  -- *
  -- * \sa
  -- * ::cuDeviceGetAttribute,
  -- * ::cuDeviceGetCount,
  -- * ::cuDeviceGetName,
  -- * ::cuDeviceGet,
  -- * ::cuDeviceTotalMem
  --  

   function cuDeviceComputeCapability
     (major : access int;
      minor : access int;
      dev : CUdevice) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:2363
   pragma Import (C, cuDeviceComputeCapability, "cuDeviceComputeCapability");

  --* @}  
  -- END CUDA_DEVICE_DEPRECATED  
  --*
  -- * \defgroup CUDA_PRIMARY_CTX Primary Context Management
  -- *
  -- * ___MANBRIEF___ primary context management functions of the low-level CUDA driver
  -- * API (___CURRENT_FILE___) ___ENDMANBRIEF___
  -- *
  -- * This section describes the primary context management functions of the low-level
  -- * CUDA driver application programming interface.
  -- *
  -- * The primary context unique per device and it's shared with CUDA runtime API.
  -- * Those functions allows seemless integration with other libraries using CUDA.
  -- *
  -- * @{
  --  

  --*
  -- * \brief Retain the primary context on the GPU
  -- *
  -- * Retains the primary context on the device, creating it if necessary,
  -- * increasing its usage count. The caller must call
  -- * ::cuDevicePrimaryCtxRelease() when done using the context.
  -- * Unlike ::cuCtxCreate() the newly created context is not pushed onto the stack.
  -- *
  -- * Context creation will fail with ::CUDA_ERROR_UNKNOWN if the compute mode of
  -- * the device is ::CU_COMPUTEMODE_PROHIBITED.  The function ::cuDeviceGetAttribute() 
  -- * can be used with ::CU_DEVICE_ATTRIBUTE_COMPUTE_MODE to determine the compute mode 
  -- * of the device. 
  -- * The <i>nvidia-smi</i> tool can be used to set the compute mode for
  -- * devices. Documentation for <i>nvidia-smi</i> can be obtained by passing a
  -- * -h option to it.
  -- *
  -- * Please note that the primary context always supports pinned allocations. Other
  -- * flags can be specified by ::cuDevicePrimaryCtxSetFlags().
  -- *
  -- * \param pctx  - Returned context handle of the new context
  -- * \param dev   - Device for which primary context is requested
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_DEVICE,
  -- * ::CUDA_ERROR_INVALID_VALUE,
  -- * ::CUDA_ERROR_OUT_OF_MEMORY,
  -- * ::CUDA_ERROR_UNKNOWN
  -- * \notefnerr
  -- *
  -- * \sa ::cuDevicePrimaryCtxRelease,
  -- * ::cuDevicePrimaryCtxSetFlags,
  -- * ::cuCtxCreate,
  -- * ::cuCtxGetApiVersion,
  -- * ::cuCtxGetCacheConfig,
  -- * ::cuCtxGetDevice,
  -- * ::cuCtxGetFlags,
  -- * ::cuCtxGetLimit,
  -- * ::cuCtxPopCurrent,
  -- * ::cuCtxPushCurrent,
  -- * ::cuCtxSetCacheConfig,
  -- * ::cuCtxSetLimit,
  -- * ::cuCtxSynchronize
  --  

   function cuDevicePrimaryCtxRetain (pctx : System.Address; dev : CUdevice) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:2431
   pragma Import (C, cuDevicePrimaryCtxRetain, "cuDevicePrimaryCtxRetain");

  --*
  -- * \brief Release the primary context on the GPU
  -- *
  -- * Releases the primary context interop on the device by decreasing the usage
  -- * count by 1. If the usage drops to 0 the primary context of device \p dev
  -- * will be destroyed regardless of how many threads it is current to.
  -- *
  -- * Please note that unlike ::cuCtxDestroy() this method does not pop the context
  -- * from stack in any circumstances.
  -- *
  -- * \param dev - Device which primary context is released
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_DEVICE
  -- * \notefnerr
  -- *
  -- * \sa ::cuDevicePrimaryCtxRetain,
  -- * ::cuCtxDestroy,
  -- * ::cuCtxGetApiVersion,
  -- * ::cuCtxGetCacheConfig,
  -- * ::cuCtxGetDevice,
  -- * ::cuCtxGetFlags,
  -- * ::cuCtxGetLimit,
  -- * ::cuCtxPopCurrent,
  -- * ::cuCtxPushCurrent,
  -- * ::cuCtxSetCacheConfig,
  -- * ::cuCtxSetLimit,
  -- * ::cuCtxSynchronize
  --  

   function cuDevicePrimaryCtxRelease (dev : CUdevice) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:2465
   pragma Import (C, cuDevicePrimaryCtxRelease, "cuDevicePrimaryCtxRelease");

  --*
  -- * \brief Set flags for the primary context
  -- *
  -- * Sets the flags for the primary context on the device overwriting perviously
  -- * set ones. If the primary context is already created
  -- * ::CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE is returned.
  -- *
  -- * The three LSBs of the \p flags parameter can be used to control how the OS
  -- * thread, which owns the CUDA context at the time of an API call, interacts
  -- * with the OS scheduler when waiting for results from the GPU. Only one of
  -- * the scheduling flags can be set when creating a context.
  -- *
  -- * - ::CU_CTX_SCHED_SPIN: Instruct CUDA to actively spin when waiting for
  -- * results from the GPU. This can decrease latency when waiting for the GPU,
  -- * but may lower the performance of CPU threads if they are performing work in
  -- * parallel with the CUDA thread.
  -- *
  -- * - ::CU_CTX_SCHED_YIELD: Instruct CUDA to yield its thread when waiting for
  -- * results from the GPU. This can increase latency when waiting for the GPU,
  -- * but can increase the performance of CPU threads performing work in parallel
  -- * with the GPU.
  -- *
  -- * - ::CU_CTX_SCHED_BLOCKING_SYNC: Instruct CUDA to block the CPU thread on a
  -- * synchronization primitive when waiting for the GPU to finish work.
  -- *
  -- * - ::CU_CTX_BLOCKING_SYNC: Instruct CUDA to block the CPU thread on a
  -- * synchronization primitive when waiting for the GPU to finish work. <br>
  -- * <b>Deprecated:</b> This flag was deprecated as of CUDA 4.0 and was
  -- * replaced with ::CU_CTX_SCHED_BLOCKING_SYNC.
  -- *
  -- * - ::CU_CTX_SCHED_AUTO: The default value if the \p flags parameter is zero,
  -- * uses a heuristic based on the number of active CUDA contexts in the
  -- * process \e C and the number of logical processors in the system \e P. If
  -- * \e C > \e P, then CUDA will yield to other OS threads when waiting for
  -- * the GPU (::CU_CTX_SCHED_YIELD), otherwise CUDA will not yield while
  -- * waiting for results and actively spin on the processor (::CU_CTX_SCHED_SPIN).
  -- * However, on low power devices like Tegra, it always defaults to
  -- * ::CU_CTX_SCHED_BLOCKING_SYNC.
  -- *
  -- * - ::CU_CTX_LMEM_RESIZE_TO_MAX: Instruct CUDA to not reduce local memory
  -- * after resizing local memory for a kernel. This can prevent thrashing by
  -- * local memory allocations when launching many kernels with high local
  -- * memory usage at the cost of potentially increased memory usage.
  -- *
  -- * \param dev   - Device for which the primary context flags are set
  -- * \param flags - New flags for the device
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_DEVICE,
  -- * ::CUDA_ERROR_INVALID_VALUE,
  -- * ::CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE
  -- * \notefnerr
  -- *
  -- * \sa ::cuDevicePrimaryCtxRetain,
  -- * ::cuDevicePrimaryCtxGetState,
  -- * ::cuCtxCreate,
  -- * ::cuCtxGetFlags
  --  

   function cuDevicePrimaryCtxSetFlags (dev : CUdevice; flags : unsigned) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:2528
   pragma Import (C, cuDevicePrimaryCtxSetFlags, "cuDevicePrimaryCtxSetFlags");

  --*
  -- * \brief Get the state of the primary context
  -- *
  -- * Returns in \p *flags the flags for the primary context of \p dev, and in
  -- * \p *active whether it is active.  See ::cuDevicePrimaryCtxSetFlags for flag
  -- * values.
  -- *
  -- * \param dev    - Device to get primary context flags for
  -- * \param flags  - Pointer to store flags
  -- * \param active - Pointer to store context state; 0 = inactive, 1 = active
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_DEVICE,
  -- * ::CUDA_ERROR_INVALID_VALUE,
  -- * \notefnerr
  -- *
  -- * \sa ::cuDevicePrimaryCtxSetFlags,
  -- * ::cuCtxGetFlags
  --  

   function cuDevicePrimaryCtxGetState
     (dev : CUdevice;
      flags : access unsigned;
      active : access int) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:2552
   pragma Import (C, cuDevicePrimaryCtxGetState, "cuDevicePrimaryCtxGetState");

  --*
  -- * \brief Destroy all allocations and reset all state on the primary context
  -- *
  -- * Explicitly destroys and cleans up all resources associated with the current
  -- * device in the current process.
  -- *
  -- * Note that it is responsibility of the calling function to ensure that no
  -- * other module in the process is using the device any more. For that reason
  -- * it is recommended to use ::cuDevicePrimaryCtxRelease() in most cases.
  -- * However it is safe for other modules to call ::cuDevicePrimaryCtxRelease()
  -- * even after resetting the device.
  -- *
  -- * \param dev - Device for which primary context is destroyed
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_DEVICE,
  -- * ::CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE
  -- * \notefnerr
  -- *
  -- * \sa ::cuDevicePrimaryCtxRetain,
  -- * ::cuDevicePrimaryCtxRelease,
  -- * ::cuCtxGetApiVersion,
  -- * ::cuCtxGetCacheConfig,
  -- * ::cuCtxGetDevice,
  -- * ::cuCtxGetFlags,
  -- * ::cuCtxGetLimit,
  -- * ::cuCtxPopCurrent,
  -- * ::cuCtxPushCurrent,
  -- * ::cuCtxSetCacheConfig,
  -- * ::cuCtxSetLimit,
  -- * ::cuCtxSynchronize
  -- *
  --  

   function cuDevicePrimaryCtxReset (dev : CUdevice) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:2590
   pragma Import (C, cuDevicePrimaryCtxReset, "cuDevicePrimaryCtxReset");

  --* @}  
  -- END CUDA_PRIMARY_CTX  
  --*
  -- * \defgroup CUDA_CTX Context Management
  -- *
  -- * ___MANBRIEF___ context management functions of the low-level CUDA driver
  -- * API (___CURRENT_FILE___) ___ENDMANBRIEF___
  -- *
  -- * This section describes the context management functions of the low-level
  -- * CUDA driver application programming interface.
  -- *
  -- * @{
  --  

  --*
  -- * \brief Create a CUDA context
  -- *
  -- * Creates a new CUDA context and associates it with the calling thread. The
  -- * \p flags parameter is described below. The context is created with a usage
  -- * count of 1 and the caller of ::cuCtxCreate() must call ::cuCtxDestroy() or
  -- * when done using the context. If a context is already current to the thread, 
  -- * it is supplanted by the newly created context and may be restored by a subsequent 
  -- * call to ::cuCtxPopCurrent().
  -- *
  -- * The three LSBs of the \p flags parameter can be used to control how the OS
  -- * thread, which owns the CUDA context at the time of an API call, interacts
  -- * with the OS scheduler when waiting for results from the GPU. Only one of
  -- * the scheduling flags can be set when creating a context.
  -- *
  -- * - ::CU_CTX_SCHED_SPIN: Instruct CUDA to actively spin when waiting for
  -- * results from the GPU. This can decrease latency when waiting for the GPU,
  -- * but may lower the performance of CPU threads if they are performing work in
  -- * parallel with the CUDA thread.
  -- *
  -- * - ::CU_CTX_SCHED_YIELD: Instruct CUDA to yield its thread when waiting for
  -- * results from the GPU. This can increase latency when waiting for the GPU,
  -- * but can increase the performance of CPU threads performing work in parallel
  -- * with the GPU.
  -- * 
  -- * - ::CU_CTX_SCHED_BLOCKING_SYNC: Instruct CUDA to block the CPU thread on a
  -- * synchronization primitive when waiting for the GPU to finish work.
  -- *
  -- * - ::CU_CTX_BLOCKING_SYNC: Instruct CUDA to block the CPU thread on a
  -- * synchronization primitive when waiting for the GPU to finish work. <br>
  -- * <b>Deprecated:</b> This flag was deprecated as of CUDA 4.0 and was
  -- * replaced with ::CU_CTX_SCHED_BLOCKING_SYNC. 
  -- *
  -- * - ::CU_CTX_SCHED_AUTO: The default value if the \p flags parameter is zero,
  -- * uses a heuristic based on the number of active CUDA contexts in the
  -- * process \e C and the number of logical processors in the system \e P. If
  -- * \e C > \e P, then CUDA will yield to other OS threads when waiting for 
  -- * the GPU (::CU_CTX_SCHED_YIELD), otherwise CUDA will not yield while 
  -- * waiting for results and actively spin on the processor (::CU_CTX_SCHED_SPIN). 
  -- * However, on low power devices like Tegra, it always defaults to 
  -- * ::CU_CTX_SCHED_BLOCKING_SYNC.
  -- *
  -- * - ::CU_CTX_MAP_HOST: Instruct CUDA to support mapped pinned allocations.
  -- * This flag must be set in order to allocate pinned host memory that is
  -- * accessible to the GPU.
  -- *
  -- * - ::CU_CTX_LMEM_RESIZE_TO_MAX: Instruct CUDA to not reduce local memory
  -- * after resizing local memory for a kernel. This can prevent thrashing by
  -- * local memory allocations when launching many kernels with high local
  -- * memory usage at the cost of potentially increased memory usage.
  -- *
  -- * Context creation will fail with ::CUDA_ERROR_UNKNOWN if the compute mode of
  -- * the device is ::CU_COMPUTEMODE_PROHIBITED. The function ::cuDeviceGetAttribute() 
  -- * can be used with ::CU_DEVICE_ATTRIBUTE_COMPUTE_MODE to determine the 
  -- * compute mode of the device. The <i>nvidia-smi</i> tool can be used to set 
  -- * the compute mode for * devices. 
  -- * Documentation for <i>nvidia-smi</i> can be obtained by passing a
  -- * -h option to it.
  -- *
  -- * \param pctx  - Returned context handle of the new context
  -- * \param flags - Context creation flags
  -- * \param dev   - Device to create context on
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_DEVICE,
  -- * ::CUDA_ERROR_INVALID_VALUE,
  -- * ::CUDA_ERROR_OUT_OF_MEMORY,
  -- * ::CUDA_ERROR_UNKNOWN
  -- * \notefnerr
  -- *
  -- * \sa ::cuCtxDestroy,
  -- * ::cuCtxGetApiVersion,
  -- * ::cuCtxGetCacheConfig,
  -- * ::cuCtxGetDevice,
  -- * ::cuCtxGetFlags,
  -- * ::cuCtxGetLimit,
  -- * ::cuCtxPopCurrent,
  -- * ::cuCtxPushCurrent,
  -- * ::cuCtxSetCacheConfig,
  -- * ::cuCtxSetLimit,
  -- * ::cuCtxSynchronize
  --  

   function cuCtxCreate_v2
     (pctx : System.Address;
      flags : unsigned;
      dev : CUdevice) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:2696
   pragma Import (C, cuCtxCreate_v2, "cuCtxCreate_v2");

  --*
  -- * \brief Destroy a CUDA context
  -- *
  -- * Destroys the CUDA context specified by \p ctx.  The context \p ctx will be
  -- * destroyed regardless of how many threads it is current to.
  -- * It is the responsibility of the calling function to ensure that no API
  -- * call issues using \p ctx while ::cuCtxDestroy() is executing.
  -- *
  -- * If \p ctx is current to the calling thread then \p ctx will also be 
  -- * popped from the current thread's context stack (as though ::cuCtxPopCurrent()
  -- * were called).  If \p ctx is current to other threads, then \p ctx will
  -- * remain current to those threads, and attempting to access \p ctx from
  -- * those threads will result in the error ::CUDA_ERROR_CONTEXT_IS_DESTROYED.
  -- *
  -- * \param ctx - Context to destroy
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- * \notefnerr
  -- *
  -- * \sa ::cuCtxCreate,
  -- * ::cuCtxGetApiVersion,
  -- * ::cuCtxGetCacheConfig,
  -- * ::cuCtxGetDevice,
  -- * ::cuCtxGetFlags,
  -- * ::cuCtxGetLimit,
  -- * ::cuCtxPopCurrent,
  -- * ::cuCtxPushCurrent,
  -- * ::cuCtxSetCacheConfig,
  -- * ::cuCtxSetLimit,
  -- * ::cuCtxSynchronize
  --  

   function cuCtxDestroy_v2 (ctx : CUcontext) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:2736
   pragma Import (C, cuCtxDestroy_v2, "cuCtxDestroy_v2");

  --*
  -- * \brief Pushes a context on the current CPU thread
  -- *
  -- * Pushes the given context \p ctx onto the CPU thread's stack of current
  -- * contexts. The specified context becomes the CPU thread's current context, so
  -- * all CUDA functions that operate on the current context are affected.
  -- *
  -- * The previous current context may be made current again by calling
  -- * ::cuCtxDestroy() or ::cuCtxPopCurrent().
  -- *
  -- * \param ctx - Context to push
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- * \notefnerr
  -- *
  -- * \sa ::cuCtxCreate,
  -- * ::cuCtxDestroy,
  -- * ::cuCtxGetApiVersion,
  -- * ::cuCtxGetCacheConfig,
  -- * ::cuCtxGetDevice,
  -- * ::cuCtxGetFlags,
  -- * ::cuCtxGetLimit,
  -- * ::cuCtxPopCurrent,
  -- * ::cuCtxSetCacheConfig,
  -- * ::cuCtxSetLimit,
  -- * ::cuCtxSynchronize
  --  

   function cuCtxPushCurrent_v2 (ctx : CUcontext) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:2772
   pragma Import (C, cuCtxPushCurrent_v2, "cuCtxPushCurrent_v2");

  --*
  -- * \brief Pops the current CUDA context from the current CPU thread.
  -- *
  -- * Pops the current CUDA context from the CPU thread and passes back the 
  -- * old context handle in \p *pctx. That context may then be made current 
  -- * to a different CPU thread by calling ::cuCtxPushCurrent().
  -- *
  -- * If a context was current to the CPU thread before ::cuCtxCreate() or
  -- * ::cuCtxPushCurrent() was called, this function makes that context current to
  -- * the CPU thread again.
  -- *
  -- * \param pctx - Returned new context handle
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT
  -- * \notefnerr
  -- *
  -- * \sa ::cuCtxCreate,
  -- * ::cuCtxDestroy,
  -- * ::cuCtxGetApiVersion,
  -- * ::cuCtxGetCacheConfig,
  -- * ::cuCtxGetDevice,
  -- * ::cuCtxGetFlags,
  -- * ::cuCtxGetLimit,
  -- * ::cuCtxPushCurrent,
  -- * ::cuCtxSetCacheConfig,
  -- * ::cuCtxSetLimit,
  -- * ::cuCtxSynchronize
  --  

   function cuCtxPopCurrent_v2 (pctx : System.Address) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:2806
   pragma Import (C, cuCtxPopCurrent_v2, "cuCtxPopCurrent_v2");

  --*
  -- * \brief Binds the specified CUDA context to the calling CPU thread
  -- *
  -- * Binds the specified CUDA context to the calling CPU thread.
  -- * If \p ctx is NULL then the CUDA context previously bound to the
  -- * calling CPU thread is unbound and ::CUDA_SUCCESS is returned.
  -- *
  -- * If there exists a CUDA context stack on the calling CPU thread, this
  -- * will replace the top of that stack with \p ctx.  
  -- * If \p ctx is NULL then this will be equivalent to popping the top
  -- * of the calling CPU thread's CUDA context stack (or a no-op if the
  -- * calling CPU thread's CUDA context stack is empty).
  -- *
  -- * \param ctx - Context to bind to the calling CPU thread
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT
  -- * \notefnerr
  -- *
  -- * \sa ::cuCtxGetCurrent, ::cuCtxCreate, ::cuCtxDestroy
  --  

   function cuCtxSetCurrent (ctx : CUcontext) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:2832
   pragma Import (C, cuCtxSetCurrent, "cuCtxSetCurrent");

  --*
  -- * \brief Returns the CUDA context bound to the calling CPU thread.
  -- *
  -- * Returns in \p *pctx the CUDA context bound to the calling CPU thread.
  -- * If no context is bound to the calling CPU thread then \p *pctx is
  -- * set to NULL and ::CUDA_SUCCESS is returned.
  -- *
  -- * \param pctx - Returned context handle
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * \notefnerr
  -- *
  -- * \sa ::cuCtxSetCurrent, ::cuCtxCreate, ::cuCtxDestroy
  --  

   function cuCtxGetCurrent (pctx : System.Address) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:2851
   pragma Import (C, cuCtxGetCurrent, "cuCtxGetCurrent");

  --*
  -- * \brief Returns the device ID for the current context
  -- *
  -- * Returns in \p *device the ordinal of the current context's device.
  -- *
  -- * \param device - Returned device ID for the current context
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE,
  -- * \notefnerr
  -- *
  -- * \sa ::cuCtxCreate,
  -- * ::cuCtxDestroy,
  -- * ::cuCtxGetApiVersion,
  -- * ::cuCtxGetCacheConfig,
  -- * ::cuCtxGetFlags,
  -- * ::cuCtxGetLimit,
  -- * ::cuCtxPopCurrent,
  -- * ::cuCtxPushCurrent,
  -- * ::cuCtxSetCacheConfig,
  -- * ::cuCtxSetLimit,
  -- * ::cuCtxSynchronize
  --  

   function cuCtxGetDevice (device : access CUdevice) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:2881
   pragma Import (C, cuCtxGetDevice, "cuCtxGetDevice");

  --*
  -- * \brief Returns the flags for the current context
  -- *
  -- * Returns in \p *flags the flags of the current context. See ::cuCtxCreate
  -- * for flag values.
  -- *
  -- * \param flags - Pointer to store flags of current context
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE,
  -- * \notefnerr
  -- *
  -- * \sa ::cuCtxCreate,
  -- * ::cuCtxGetApiVersion,
  -- * ::cuCtxGetCacheConfig,
  -- * ::cuCtxGetCurrent,
  -- * ::cuCtxGetDevice
  -- * ::cuCtxGetLimit,
  -- * ::cuCtxGetSharedMemConfig,
  -- * ::cuCtxGetStreamPriorityRange
  --  

   function cuCtxGetFlags (flags : access unsigned) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:2909
   pragma Import (C, cuCtxGetFlags, "cuCtxGetFlags");

  --*
  -- * \brief Block for a context's tasks to complete
  -- *
  -- * Blocks until the device has completed all preceding requested tasks.
  -- * ::cuCtxSynchronize() returns an error if one of the preceding tasks failed.
  -- * If the context was created with the ::CU_CTX_SCHED_BLOCKING_SYNC flag, the 
  -- * CPU thread will block until the GPU context has finished its work.
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT
  -- * \notefnerr
  -- *
  -- * \sa ::cuCtxCreate,
  -- * ::cuCtxDestroy,
  -- * ::cuCtxGetApiVersion,
  -- * ::cuCtxGetCacheConfig,
  -- * ::cuCtxGetDevice,
  -- * ::cuCtxGetFlags,
  -- * ::cuCtxGetLimit,
  -- * ::cuCtxPopCurrent,
  -- * ::cuCtxPushCurrent,
  -- * ::cuCtxSetCacheConfig,
  -- * ::cuCtxSetLimit
  --  

   function cuCtxSynchronize return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:2939
   pragma Import (C, cuCtxSynchronize, "cuCtxSynchronize");

  --*
  -- * \brief Set resource limits
  -- *
  -- * Setting \p limit to \p value is a request by the application to update
  -- * the current limit maintained by the context. The driver is free to
  -- * modify the requested value to meet h/w requirements (this could be
  -- * clamping to minimum or maximum values, rounding up to nearest element
  -- * size, etc). The application can use ::cuCtxGetLimit() to find out exactly
  -- * what the limit has been set to.
  -- *
  -- * Setting each ::CUlimit has its own specific restrictions, so each is
  -- * discussed here.
  -- *
  -- * - ::CU_LIMIT_STACK_SIZE controls the stack size in bytes of each GPU thread.
  -- *   This limit is only applicable to devices of compute capability 2.0 and
  -- *   higher. Attempting to set this limit on devices of compute capability
  -- *   less than 2.0 will result in the error ::CUDA_ERROR_UNSUPPORTED_LIMIT
  -- *   being returned.
  -- *
  -- * - ::CU_LIMIT_PRINTF_FIFO_SIZE controls the size in bytes of the FIFO used
  -- *   by the ::printf() device system call. Setting ::CU_LIMIT_PRINTF_FIFO_SIZE
  -- *   must be performed before launching any kernel that uses the ::printf()
  -- *   device system call, otherwise ::CUDA_ERROR_INVALID_VALUE will be returned.
  -- *   This limit is only applicable to devices of compute capability 2.0 and
  -- *   higher. Attempting to set this limit on devices of compute capability
  -- *   less than 2.0 will result in the error ::CUDA_ERROR_UNSUPPORTED_LIMIT
  -- *   being returned.
  -- *
  -- * - ::CU_LIMIT_MALLOC_HEAP_SIZE controls the size in bytes of the heap used
  -- *   by the ::malloc() and ::free() device system calls. Setting
  -- *   ::CU_LIMIT_MALLOC_HEAP_SIZE must be performed before launching any kernel
  -- *   that uses the ::malloc() or ::free() device system calls, otherwise
  -- *   ::CUDA_ERROR_INVALID_VALUE will be returned. This limit is only applicable
  -- *   to devices of compute capability 2.0 and higher. Attempting to set this
  -- *   limit on devices of compute capability less than 2.0 will result in the
  -- *   error ::CUDA_ERROR_UNSUPPORTED_LIMIT being returned.
  -- *
  -- * - ::CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH controls the maximum nesting depth of
  -- *   a grid at which a thread can safely call ::cudaDeviceSynchronize(). Setting
  -- *   this limit must be performed before any launch of a kernel that uses the 
  -- *   device runtime and calls ::cudaDeviceSynchronize() above the default sync
  -- *   depth, two levels of grids. Calls to ::cudaDeviceSynchronize() will fail 
  -- *   with error code ::cudaErrorSyncDepthExceeded if the limitation is 
  -- *   violated. This limit can be set smaller than the default or up the maximum
  -- *   launch depth of 24. When setting this limit, keep in mind that additional
  -- *   levels of sync depth require the driver to reserve large amounts of device
  -- *   memory which can no longer be used for user allocations. If these 
  -- *   reservations of device memory fail, ::cuCtxSetLimit will return 
  -- *   ::CUDA_ERROR_OUT_OF_MEMORY, and the limit can be reset to a lower value.
  -- *   This limit is only applicable to devices of compute capability 3.5 and
  -- *   higher. Attempting to set this limit on devices of compute capability less
  -- *   than 3.5 will result in the error ::CUDA_ERROR_UNSUPPORTED_LIMIT being 
  -- *   returned.
  -- *
  -- * - ::CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT controls the maximum number of
  -- *   outstanding device runtime launches that can be made from the current
  -- *   context. A grid is outstanding from the point of launch up until the grid
  -- *   is known to have been completed. Device runtime launches which violate 
  -- *   this limitation fail and return ::cudaErrorLaunchPendingCountExceeded when
  -- *   ::cudaGetLastError() is called after launch. If more pending launches than
  -- *   the default (2048 launches) are needed for a module using the device
  -- *   runtime, this limit can be increased. Keep in mind that being able to
  -- *   sustain additional pending launches will require the driver to reserve
  -- *   larger amounts of device memory upfront which can no longer be used for
  -- *   allocations. If these reservations fail, ::cuCtxSetLimit will return
  -- *   ::CUDA_ERROR_OUT_OF_MEMORY, and the limit can be reset to a lower value.
  -- *   This limit is only applicable to devices of compute capability 3.5 and
  -- *   higher. Attempting to set this limit on devices of compute capability less
  -- *   than 3.5 will result in the error ::CUDA_ERROR_UNSUPPORTED_LIMIT being
  -- *   returned.
  -- *
  -- * \param limit - Limit to set
  -- * \param value - Size of limit
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_INVALID_VALUE,
  -- * ::CUDA_ERROR_UNSUPPORTED_LIMIT,
  -- * ::CUDA_ERROR_OUT_OF_MEMORY
  -- * \notefnerr
  -- *
  -- * \sa ::cuCtxCreate,
  -- * ::cuCtxDestroy,
  -- * ::cuCtxGetApiVersion,
  -- * ::cuCtxGetCacheConfig,
  -- * ::cuCtxGetDevice,
  -- * ::cuCtxGetFlags,
  -- * ::cuCtxGetLimit,
  -- * ::cuCtxPopCurrent,
  -- * ::cuCtxPushCurrent,
  -- * ::cuCtxSetCacheConfig,
  -- * ::cuCtxSynchronize
  --  

   function cuCtxSetLimit (limit : CUlimit; value : stddef_h.size_t) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:3034
   pragma Import (C, cuCtxSetLimit, "cuCtxSetLimit");

  --*
  -- * \brief Returns resource limits
  -- *
  -- * Returns in \p *pvalue the current size of \p limit.  The supported
  -- * ::CUlimit values are:
  -- * - ::CU_LIMIT_STACK_SIZE: stack size in bytes of each GPU thread.
  -- * - ::CU_LIMIT_PRINTF_FIFO_SIZE: size in bytes of the FIFO used by the
  -- *   ::printf() device system call.
  -- * - ::CU_LIMIT_MALLOC_HEAP_SIZE: size in bytes of the heap used by the
  -- *   ::malloc() and ::free() device system calls.
  -- * - ::CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH: maximum grid depth at which a thread
  -- *   can issue the device runtime call ::cudaDeviceSynchronize() to wait on
  -- *   child grid launches to complete.
  -- * - ::CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT: maximum number of outstanding
  -- *   device runtime launches that can be made from this context.
  -- *
  -- * \param limit  - Limit to query
  -- * \param pvalue - Returned size of limit
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_INVALID_VALUE,
  -- * ::CUDA_ERROR_UNSUPPORTED_LIMIT
  -- * \notefnerr
  -- *
  -- * \sa ::cuCtxCreate,
  -- * ::cuCtxDestroy,
  -- * ::cuCtxGetApiVersion,
  -- * ::cuCtxGetCacheConfig,
  -- * ::cuCtxGetDevice,
  -- * ::cuCtxGetFlags,
  -- * ::cuCtxPopCurrent,
  -- * ::cuCtxPushCurrent,
  -- * ::cuCtxSetCacheConfig,
  -- * ::cuCtxSetLimit,
  -- * ::cuCtxSynchronize
  --  

   function cuCtxGetLimit (pvalue : access stddef_h.size_t; limit : CUlimit) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:3073
   pragma Import (C, cuCtxGetLimit, "cuCtxGetLimit");

  --*
  -- * \brief Returns the preferred cache configuration for the current context.
  -- *
  -- * On devices where the L1 cache and shared memory use the same hardware
  -- * resources, this function returns through \p pconfig the preferred cache configuration
  -- * for the current context. This is only a preference. The driver will use
  -- * the requested configuration if possible, but it is free to choose a different
  -- * configuration if required to execute functions.
  -- *
  -- * This will return a \p pconfig of ::CU_FUNC_CACHE_PREFER_NONE on devices
  -- * where the size of the L1 cache and shared memory are fixed.
  -- *
  -- * The supported cache configurations are:
  -- * - ::CU_FUNC_CACHE_PREFER_NONE: no preference for shared memory or L1 (default)
  -- * - ::CU_FUNC_CACHE_PREFER_SHARED: prefer larger shared memory and smaller L1 cache
  -- * - ::CU_FUNC_CACHE_PREFER_L1: prefer larger L1 cache and smaller shared memory
  -- * - ::CU_FUNC_CACHE_PREFER_EQUAL: prefer equal sized L1 cache and shared memory
  -- *
  -- * \param pconfig - Returned cache configuration
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- * \notefnerr
  -- *
  -- * \sa ::cuCtxCreate,
  -- * ::cuCtxDestroy,
  -- * ::cuCtxGetApiVersion,
  -- * ::cuCtxGetDevice,
  -- * ::cuCtxGetFlags,
  -- * ::cuCtxGetLimit,
  -- * ::cuCtxPopCurrent,
  -- * ::cuCtxPushCurrent,
  -- * ::cuCtxSetCacheConfig,
  -- * ::cuCtxSetLimit,
  -- * ::cuCtxSynchronize,
  -- * ::cuFuncSetCacheConfig
  --  

   function cuCtxGetCacheConfig (pconfig : access CUfunc_cache) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:3116
   pragma Import (C, cuCtxGetCacheConfig, "cuCtxGetCacheConfig");

  --*
  -- * \brief Sets the preferred cache configuration for the current context.
  -- *
  -- * On devices where the L1 cache and shared memory use the same hardware
  -- * resources, this sets through \p config the preferred cache configuration for
  -- * the current context. This is only a preference. The driver will use
  -- * the requested configuration if possible, but it is free to choose a different
  -- * configuration if required to execute the function. Any function preference
  -- * set via ::cuFuncSetCacheConfig() will be preferred over this context-wide
  -- * setting. Setting the context-wide cache configuration to
  -- * ::CU_FUNC_CACHE_PREFER_NONE will cause subsequent kernel launches to prefer
  -- * to not change the cache configuration unless required to launch the kernel.
  -- *
  -- * This setting does nothing on devices where the size of the L1 cache and
  -- * shared memory are fixed.
  -- *
  -- * Launching a kernel with a different preference than the most recent
  -- * preference setting may insert a device-side synchronization point.
  -- *
  -- * The supported cache configurations are:
  -- * - ::CU_FUNC_CACHE_PREFER_NONE: no preference for shared memory or L1 (default)
  -- * - ::CU_FUNC_CACHE_PREFER_SHARED: prefer larger shared memory and smaller L1 cache
  -- * - ::CU_FUNC_CACHE_PREFER_L1: prefer larger L1 cache and smaller shared memory
  -- * - ::CU_FUNC_CACHE_PREFER_EQUAL: prefer equal sized L1 cache and shared memory
  -- *
  -- * \param config - Requested cache configuration
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- * \notefnerr
  -- *
  -- * \sa ::cuCtxCreate,
  -- * ::cuCtxDestroy,
  -- * ::cuCtxGetApiVersion,
  -- * ::cuCtxGetCacheConfig,
  -- * ::cuCtxGetDevice,
  -- * ::cuCtxGetFlags,
  -- * ::cuCtxGetLimit,
  -- * ::cuCtxPopCurrent,
  -- * ::cuCtxPushCurrent,
  -- * ::cuCtxSetLimit,
  -- * ::cuCtxSynchronize,
  -- * ::cuFuncSetCacheConfig
  --  

   function cuCtxSetCacheConfig (config : CUfunc_cache) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:3166
   pragma Import (C, cuCtxSetCacheConfig, "cuCtxSetCacheConfig");

  --*
  -- * \brief Returns the current shared memory configuration for the current context.
  -- *
  -- * This function will return in \p pConfig the current size of shared memory banks
  -- * in the current context. On devices with configurable shared memory banks, 
  -- * ::cuCtxSetSharedMemConfig can be used to change this setting, so that all 
  -- * subsequent kernel launches will by default use the new bank size. When 
  -- * ::cuCtxGetSharedMemConfig is called on devices without configurable shared 
  -- * memory, it will return the fixed bank size of the hardware.
  -- *
  -- * The returned bank configurations can be either:
  -- * - ::CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE:  shared memory bank width is 
  -- *   four bytes.
  -- * - ::CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE: shared memory bank width will
  -- *   eight bytes.
  -- *
  -- * \param pConfig - returned shared memory configuration
  -- * \return 
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- * \notefnerr
  -- *
  -- * \sa ::cuCtxCreate,
  -- * ::cuCtxDestroy,
  -- * ::cuCtxGetApiVersion,
  -- * ::cuCtxGetCacheConfig,
  -- * ::cuCtxGetDevice,
  -- * ::cuCtxGetFlags,
  -- * ::cuCtxGetLimit,
  -- * ::cuCtxPopCurrent,
  -- * ::cuCtxPushCurrent,
  -- * ::cuCtxSetLimit,
  -- * ::cuCtxSynchronize,
  -- * ::cuCtxGetSharedMemConfig,
  -- * ::cuFuncSetCacheConfig,
  --  

   function cuCtxGetSharedMemConfig (pConfig : access CUsharedconfig) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:3208
   pragma Import (C, cuCtxGetSharedMemConfig, "cuCtxGetSharedMemConfig");

  --*
  -- * \brief Sets the shared memory configuration for the current context.
  -- *
  -- * On devices with configurable shared memory banks, this function will set
  -- * the context's shared memory bank size which is used for subsequent kernel 
  -- * launches. 
  -- *
  -- * Changed the shared memory configuration between launches may insert a device
  -- * side synchronization point between those launches.
  -- *
  -- * Changing the shared memory bank size will not increase shared memory usage
  -- * or affect occupancy of kernels, but may have major effects on performance. 
  -- * Larger bank sizes will allow for greater potential bandwidth to shared memory,
  -- * but will change what kinds of accesses to shared memory will result in bank 
  -- * conflicts.
  -- *
  -- * This function will do nothing on devices with fixed shared memory bank size.
  -- *
  -- * The supported bank configurations are:
  -- * - ::CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE: set bank width to the default initial
  -- *   setting (currently, four bytes).
  -- * - ::CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE: set shared memory bank width to
  -- *   be natively four bytes.
  -- * - ::CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE: set shared memory bank width to
  -- *   be natively eight bytes.
  -- *
  -- * \param config - requested shared memory configuration
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- * \notefnerr
  -- *
  -- * \sa ::cuCtxCreate,
  -- * ::cuCtxDestroy,
  -- * ::cuCtxGetApiVersion,
  -- * ::cuCtxGetCacheConfig,
  -- * ::cuCtxGetDevice,
  -- * ::cuCtxGetFlags,
  -- * ::cuCtxGetLimit,
  -- * ::cuCtxPopCurrent,
  -- * ::cuCtxPushCurrent,
  -- * ::cuCtxSetLimit,
  -- * ::cuCtxSynchronize,
  -- * ::cuCtxGetSharedMemConfig,
  -- * ::cuFuncSetCacheConfig,
  --  

   function cuCtxSetSharedMemConfig (config : CUsharedconfig) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:3260
   pragma Import (C, cuCtxSetSharedMemConfig, "cuCtxSetSharedMemConfig");

  --*
  -- * \brief Gets the context's API version.
  -- *
  -- * Returns a version number in \p version corresponding to the capabilities of
  -- * the context (e.g. 3010 or 3020), which library developers can use to direct
  -- * callers to a specific API version. If \p ctx is NULL, returns the API version
  -- * used to create the currently bound context.
  -- *
  -- * Note that new API versions are only introduced when context capabilities are
  -- * changed that break binary compatibility, so the API version and driver version
  -- * may be different. For example, it is valid for the API version to be 3020 while
  -- * the driver version is 4020.
  -- *
  -- * \param ctx     - Context to check
  -- * \param version - Pointer to version
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_UNKNOWN
  -- * \notefnerr
  -- *
  -- * \sa ::cuCtxCreate,
  -- * ::cuCtxDestroy,
  -- * ::cuCtxGetDevice,
  -- * ::cuCtxGetFlags,
  -- * ::cuCtxGetLimit,
  -- * ::cuCtxPopCurrent,
  -- * ::cuCtxPushCurrent,
  -- * ::cuCtxSetCacheConfig,
  -- * ::cuCtxSetLimit,
  -- * ::cuCtxSynchronize
  --  

   function cuCtxGetApiVersion (ctx : CUcontext; version : access unsigned) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:3298
   pragma Import (C, cuCtxGetApiVersion, "cuCtxGetApiVersion");

  --*
  -- * \brief Returns numerical values that correspond to the least and
  -- * greatest stream priorities.
  -- *
  -- * Returns in \p *leastPriority and \p *greatestPriority the numerical values that correspond
  -- * to the least and greatest stream priorities respectively. Stream priorities
  -- * follow a convention where lower numbers imply greater priorities. The range of
  -- * meaningful stream priorities is given by [\p *greatestPriority, \p *leastPriority].
  -- * If the user attempts to create a stream with a priority value that is
  -- * outside the meaningful range as specified by this API, the priority is
  -- * automatically clamped down or up to either \p *leastPriority or \p *greatestPriority
  -- * respectively. See ::cuStreamCreateWithPriority for details on creating a
  -- * priority stream.
  -- * A NULL may be passed in for \p *leastPriority or \p *greatestPriority if the value
  -- * is not desired.
  -- *
  -- * This function will return '0' in both \p *leastPriority and \p *greatestPriority if
  -- * the current context's device does not support stream priorities
  -- * (see ::cuDeviceGetAttribute).
  -- *
  -- * \param leastPriority    - Pointer to an int in which the numerical value for least
  -- *                           stream priority is returned
  -- * \param greatestPriority - Pointer to an int in which the numerical value for greatest
  -- *                           stream priority is returned
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_INVALID_VALUE,
  -- * \notefnerr
  -- *
  -- * \sa ::cuStreamCreateWithPriority,
  -- * ::cuStreamGetPriority,
  -- * ::cuCtxGetDevice,
  -- * ::cuCtxGetFlags,
  -- * ::cuCtxSetLimit,
  -- * ::cuCtxSynchronize
  --  

   function cuCtxGetStreamPriorityRange (leastPriority : access int; greatestPriority : access int) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:3337
   pragma Import (C, cuCtxGetStreamPriorityRange, "cuCtxGetStreamPriorityRange");

  --* @}  
  -- END CUDA_CTX  
  --*
  -- * \defgroup CUDA_CTX_DEPRECATED Context Management [DEPRECATED]
  -- *
  -- * ___MANBRIEF___ deprecated context management functions of the low-level CUDA
  -- * driver API (___CURRENT_FILE___) ___ENDMANBRIEF___
  -- *
  -- * This section describes the deprecated context management functions of the low-level
  -- * CUDA driver application programming interface.
  -- *
  -- * @{
  --  

  --*
  -- * \brief Increment a context's usage-count
  -- *
  -- * \deprecated
  -- *
  -- * Note that this function is deprecated and should not be used.
  -- *
  -- * Increments the usage count of the context and passes back a context handle
  -- * in \p *pctx that must be passed to ::cuCtxDetach() when the application is
  -- * done with the context. ::cuCtxAttach() fails if there is no context current
  -- * to the thread.
  -- *
  -- * Currently, the \p flags parameter must be 0.
  -- *
  -- * \param pctx  - Returned context handle of the current context
  -- * \param flags - Context attach flags (must be 0)
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- * \notefnerr
  -- *
  -- * \sa ::cuCtxCreate,
  -- * ::cuCtxDestroy,
  -- * ::cuCtxDetach,
  -- * ::cuCtxGetApiVersion,
  -- * ::cuCtxGetCacheConfig,
  -- * ::cuCtxGetDevice,
  -- * ::cuCtxGetFlags,
  -- * ::cuCtxGetLimit,
  -- * ::cuCtxPopCurrent,
  -- * ::cuCtxPushCurrent,
  -- * ::cuCtxSetCacheConfig,
  -- * ::cuCtxSetLimit,
  -- * ::cuCtxSynchronize
  --  

   function cuCtxAttach (pctx : System.Address; flags : unsigned) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:3392
   pragma Import (C, cuCtxAttach, "cuCtxAttach");

  --*
  -- * \brief Decrement a context's usage-count
  -- *
  -- * \deprecated
  -- *
  -- * Note that this function is deprecated and should not be used.
  -- *
  -- * Decrements the usage count of the context \p ctx, and destroys the context
  -- * if the usage count goes to 0. The context must be a handle that was passed
  -- * back by ::cuCtxCreate() or ::cuCtxAttach(), and must be current to the
  -- * calling thread.
  -- *
  -- * \param ctx - Context to destroy
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT
  -- * \notefnerr
  -- *
  -- * \sa ::cuCtxCreate,
  -- * ::cuCtxDestroy,
  -- * ::cuCtxGetApiVersion,
  -- * ::cuCtxGetCacheConfig,
  -- * ::cuCtxGetDevice,
  -- * ::cuCtxGetFlags,
  -- * ::cuCtxGetLimit,
  -- * ::cuCtxPopCurrent,
  -- * ::cuCtxPushCurrent,
  -- * ::cuCtxSetCacheConfig,
  -- * ::cuCtxSetLimit,
  -- * ::cuCtxSynchronize
  --  

   function cuCtxDetach (ctx : CUcontext) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:3428
   pragma Import (C, cuCtxDetach, "cuCtxDetach");

  --* @}  
  -- END CUDA_CTX_DEPRECATED  
  --*
  -- * \defgroup CUDA_MODULE Module Management
  -- *
  -- * ___MANBRIEF___ module management functions of the low-level CUDA driver API
  -- * (___CURRENT_FILE___) ___ENDMANBRIEF___
  -- *
  -- * This section describes the module management functions of the low-level CUDA
  -- * driver application programming interface.
  -- *
  -- * @{
  --  

  --*
  -- * \brief Loads a compute module
  -- *
  -- * Takes a filename \p fname and loads the corresponding module \p module into
  -- * the current context. The CUDA driver API does not attempt to lazily
  -- * allocate the resources needed by a module; if the memory for functions and
  -- * data (constant and global) needed by the module cannot be allocated,
  -- * ::cuModuleLoad() fails. The file should be a \e cubin file as output by
  -- * \b nvcc, or a \e PTX file either as output by \b nvcc or handwritten, or
  -- * a \e fatbin file as output by \b nvcc from toolchain 4.0 or later.
  -- *
  -- * \param module - Returned module
  -- * \param fname  - Filename of module to load
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE,
  -- * ::CUDA_ERROR_INVALID_PTX,
  -- * ::CUDA_ERROR_NOT_FOUND,
  -- * ::CUDA_ERROR_OUT_OF_MEMORY,
  -- * ::CUDA_ERROR_FILE_NOT_FOUND,
  -- * ::CUDA_ERROR_NO_BINARY_FOR_GPU,
  -- * ::CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND,
  -- * ::CUDA_ERROR_SHARED_OBJECT_INIT_FAILED
  -- * \notefnerr
  -- *
  -- * \sa ::cuModuleGetFunction,
  -- * ::cuModuleGetGlobal,
  -- * ::cuModuleGetTexRef,
  -- * ::cuModuleLoadData,
  -- * ::cuModuleLoadDataEx,
  -- * ::cuModuleLoadFatBinary,
  -- * ::cuModuleUnload
  --  

   function cuModuleLoad (module : System.Address; fname : Interfaces.C.Strings.chars_ptr) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:3482
   pragma Import (C, cuModuleLoad, "cuModuleLoad");

  --*
  -- * \brief Load a module's data
  -- *
  -- * Takes a pointer \p image and loads the corresponding module \p module into
  -- * the current context. The pointer may be obtained by mapping a \e cubin or
  -- * \e PTX or \e fatbin file, passing a \e cubin or \e PTX or \e fatbin file
  -- * as a NULL-terminated text string, or incorporating a \e cubin or \e fatbin
  -- * object into the executable resources and using operating system calls such
  -- * as Windows \c FindResource() to obtain the pointer.
  -- *
  -- * \param module - Returned module
  -- * \param image  - Module data to load
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE,
  -- * ::CUDA_ERROR_INVALID_PTX,
  -- * ::CUDA_ERROR_OUT_OF_MEMORY,
  -- * ::CUDA_ERROR_NO_BINARY_FOR_GPU,
  -- * ::CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND,
  -- * ::CUDA_ERROR_SHARED_OBJECT_INIT_FAILED
  -- * \notefnerr
  -- *
  -- * \sa ::cuModuleGetFunction,
  -- * ::cuModuleGetGlobal,
  -- * ::cuModuleGetTexRef,
  -- * ::cuModuleLoad,
  -- * ::cuModuleLoadDataEx,
  -- * ::cuModuleLoadFatBinary,
  -- * ::cuModuleUnload
  --  

   function cuModuleLoadData (module : System.Address; image : System.Address) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:3518
   pragma Import (C, cuModuleLoadData, "cuModuleLoadData");

  --*
  -- * \brief Load a module's data with options
  -- *
  -- * Takes a pointer \p image and loads the corresponding module \p module into
  -- * the current context. The pointer may be obtained by mapping a \e cubin or
  -- * \e PTX or \e fatbin file, passing a \e cubin or \e PTX or \e fatbin file
  -- * as a NULL-terminated text string, or incorporating a \e cubin or \e fatbin
  -- * object into the executable resources and using operating system calls such
  -- * as Windows \c FindResource() to obtain the pointer. Options are passed as
  -- * an array via \p options and any corresponding parameters are passed in
  -- * \p optionValues. The number of total options is supplied via \p numOptions.
  -- * Any outputs will be returned via \p optionValues. 
  -- *
  -- * \param module       - Returned module
  -- * \param image        - Module data to load
  -- * \param numOptions   - Number of options
  -- * \param options      - Options for JIT
  -- * \param optionValues - Option values for JIT
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE,
  -- * ::CUDA_ERROR_INVALID_PTX,
  -- * ::CUDA_ERROR_OUT_OF_MEMORY,
  -- * ::CUDA_ERROR_NO_BINARY_FOR_GPU,
  -- * ::CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND,
  -- * ::CUDA_ERROR_SHARED_OBJECT_INIT_FAILED
  -- * \notefnerr
  -- *
  -- * \sa ::cuModuleGetFunction,
  -- * ::cuModuleGetGlobal,
  -- * ::cuModuleGetTexRef,
  -- * ::cuModuleLoad,
  -- * ::cuModuleLoadData,
  -- * ::cuModuleLoadFatBinary,
  -- * ::cuModuleUnload
  --  

   function cuModuleLoadDataEx
     (module : System.Address;
      image : System.Address;
      numOptions : unsigned;
      options : access CUjit_option;
      optionValues : System.Address) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:3560
   pragma Import (C, cuModuleLoadDataEx, "cuModuleLoadDataEx");

  --*
  -- * \brief Load a module's data
  -- *
  -- * Takes a pointer \p fatCubin and loads the corresponding module \p module
  -- * into the current context. The pointer represents a <i>fat binary</i> object,
  -- * which is a collection of different \e cubin and/or \e PTX files, all
  -- * representing the same device code, but compiled and optimized for different
  -- * architectures.
  -- *
  -- * Prior to CUDA 4.0, there was no documented API for constructing and using
  -- * fat binary objects by programmers.  Starting with CUDA 4.0, fat binary
  -- * objects can be constructed by providing the <i>-fatbin option</i> to \b nvcc.
  -- * More information can be found in the \b nvcc document.
  -- *
  -- * \param module   - Returned module
  -- * \param fatCubin - Fat binary to load
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE,
  -- * ::CUDA_ERROR_INVALID_PTX,
  -- * ::CUDA_ERROR_NOT_FOUND,
  -- * ::CUDA_ERROR_OUT_OF_MEMORY,
  -- * ::CUDA_ERROR_NO_BINARY_FOR_GPU,
  -- * ::CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND,
  -- * ::CUDA_ERROR_SHARED_OBJECT_INIT_FAILED
  -- * \notefnerr
  -- *
  -- * \sa ::cuModuleGetFunction,
  -- * ::cuModuleGetGlobal,
  -- * ::cuModuleGetTexRef,
  -- * ::cuModuleLoad,
  -- * ::cuModuleLoadData,
  -- * ::cuModuleLoadDataEx,
  -- * ::cuModuleUnload
  --  

   function cuModuleLoadFatBinary (module : System.Address; fatCubin : System.Address) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:3601
   pragma Import (C, cuModuleLoadFatBinary, "cuModuleLoadFatBinary");

  --*
  -- * \brief Unloads a module
  -- *
  -- * Unloads a module \p hmod from the current context.
  -- *
  -- * \param hmod - Module to unload
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- * \notefnerr
  -- *
  -- * \sa ::cuModuleGetFunction,
  -- * ::cuModuleGetGlobal,
  -- * ::cuModuleGetTexRef,
  -- * ::cuModuleLoad,
  -- * ::cuModuleLoadData,
  -- * ::cuModuleLoadDataEx,
  -- * ::cuModuleLoadFatBinary
  --  

   function cuModuleUnload (hmod : CUmodule) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:3626
   pragma Import (C, cuModuleUnload, "cuModuleUnload");

  --*
  -- * \brief Returns a function handle
  -- *
  -- * Returns in \p *hfunc the handle of the function of name \p name located in
  -- * module \p hmod. If no function of that name exists, ::cuModuleGetFunction()
  -- * returns ::CUDA_ERROR_NOT_FOUND.
  -- *
  -- * \param hfunc - Returned function handle
  -- * \param hmod  - Module to retrieve function from
  -- * \param name  - Name of function to retrieve
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE,
  -- * ::CUDA_ERROR_NOT_FOUND
  -- * \notefnerr
  -- *
  -- * \sa ::cuModuleGetGlobal,
  -- * ::cuModuleGetTexRef,
  -- * ::cuModuleLoad,
  -- * ::cuModuleLoadData,
  -- * ::cuModuleLoadDataEx,
  -- * ::cuModuleLoadFatBinary,
  -- * ::cuModuleUnload
  --  

   function cuModuleGetFunction
     (hfunc : System.Address;
      hmod : CUmodule;
      name : Interfaces.C.Strings.chars_ptr) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:3656
   pragma Import (C, cuModuleGetFunction, "cuModuleGetFunction");

  --*
  -- * \brief Returns a global pointer from a module
  -- *
  -- * Returns in \p *dptr and \p *bytes the base pointer and size of the
  -- * global of name \p name located in module \p hmod. If no variable of that name
  -- * exists, ::cuModuleGetGlobal() returns ::CUDA_ERROR_NOT_FOUND. Both
  -- * parameters \p dptr and \p bytes are optional. If one of them is
  -- * NULL, it is ignored.
  -- *
  -- * \param dptr  - Returned global device pointer
  -- * \param bytes - Returned global size in bytes
  -- * \param hmod  - Module to retrieve global from
  -- * \param name  - Name of global to retrieve
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE,
  -- * ::CUDA_ERROR_NOT_FOUND
  -- * \notefnerr
  -- *
  -- * \sa ::cuModuleGetFunction,
  -- * ::cuModuleGetTexRef,
  -- * ::cuModuleLoad,
  -- * ::cuModuleLoadData,
  -- * ::cuModuleLoadDataEx,
  -- * ::cuModuleLoadFatBinary,
  -- * ::cuModuleUnload
  --  

   function cuModuleGetGlobal_v2
     (dptr : access CUdeviceptr;
      bytes : access stddef_h.size_t;
      hmod : CUmodule;
      name : Interfaces.C.Strings.chars_ptr) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:3690
   pragma Import (C, cuModuleGetGlobal_v2, "cuModuleGetGlobal_v2");

  --*
  -- * \brief Returns a handle to a texture reference
  -- *
  -- * Returns in \p *pTexRef the handle of the texture reference of name \p name
  -- * in the module \p hmod. If no texture reference of that name exists,
  -- * ::cuModuleGetTexRef() returns ::CUDA_ERROR_NOT_FOUND. This texture reference
  -- * handle should not be destroyed, since it will be destroyed when the module
  -- * is unloaded.
  -- *
  -- * \param pTexRef  - Returned texture reference
  -- * \param hmod     - Module to retrieve texture reference from
  -- * \param name     - Name of texture reference to retrieve
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE,
  -- * ::CUDA_ERROR_NOT_FOUND
  -- * \notefnerr
  -- *
  -- * \sa ::cuModuleGetFunction,
  -- * ::cuModuleGetGlobal,
  -- * ::cuModuleGetSurfRef,
  -- * ::cuModuleLoad,
  -- * ::cuModuleLoadData,
  -- * ::cuModuleLoadDataEx,
  -- * ::cuModuleLoadFatBinary,
  -- * ::cuModuleUnload
  --  

   function cuModuleGetTexRef
     (pTexRef : System.Address;
      hmod : CUmodule;
      name : Interfaces.C.Strings.chars_ptr) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:3724
   pragma Import (C, cuModuleGetTexRef, "cuModuleGetTexRef");

  --*
  -- * \brief Returns a handle to a surface reference
  -- *
  -- * Returns in \p *pSurfRef the handle of the surface reference of name \p name
  -- * in the module \p hmod. If no surface reference of that name exists,
  -- * ::cuModuleGetSurfRef() returns ::CUDA_ERROR_NOT_FOUND.
  -- *
  -- * \param pSurfRef  - Returned surface reference
  -- * \param hmod     - Module to retrieve surface reference from
  -- * \param name     - Name of surface reference to retrieve
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE,
  -- * ::CUDA_ERROR_NOT_FOUND
  -- * \notefnerr
  -- *
  -- * \sa ::cuModuleGetFunction,
  -- * ::cuModuleGetGlobal,
  -- * ::cuModuleGetTexRef,
  -- * ::cuModuleLoad,
  -- * ::cuModuleLoadData,
  -- * ::cuModuleLoadDataEx,
  -- * ::cuModuleLoadFatBinary,
  -- * ::cuModuleUnload
  --  

   function cuModuleGetSurfRef
     (pSurfRef : System.Address;
      hmod : CUmodule;
      name : Interfaces.C.Strings.chars_ptr) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:3755
   pragma Import (C, cuModuleGetSurfRef, "cuModuleGetSurfRef");

  --*
  -- * \brief Creates a pending JIT linker invocation.
  -- *
  -- * If the call is successful, the caller owns the returned CUlinkState, which
  -- * should eventually be destroyed with ::cuLinkDestroy.  The
  -- * device code machine size (32 or 64 bit) will match the calling application.
  -- *
  -- * Both linker and compiler options may be specified.  Compiler options will
  -- * be applied to inputs to this linker action which must be compiled from PTX.
  -- * The options ::CU_JIT_WALL_TIME,
  -- * ::CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES, and ::CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES
  -- * will accumulate data until the CUlinkState is destroyed.
  -- *
  -- * \p optionValues must remain valid for the life of the CUlinkState if output
  -- * options are used.  No other references to inputs are maintained after this
  -- * call returns.
  -- *
  -- * \param numOptions   Size of options arrays
  -- * \param options      Array of linker and compiler options
  -- * \param optionValues Array of option values, each cast to void *
  -- * \param stateOut     On success, this will contain a CUlinkState to specify
  -- *                     and complete this action
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE,
  -- * ::CUDA_ERROR_OUT_OF_MEMORY
  -- * \notefnerr
  -- *
  -- * \sa ::cuLinkAddData,
  -- * ::cuLinkAddFile,
  -- * ::cuLinkComplete,
  -- * ::cuLinkDestroy
  --  

   function cuLinkCreate_v2
     (numOptions : unsigned;
      options : access CUjit_option;
      optionValues : System.Address;
      stateOut : System.Address) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:3797
   pragma Import (C, cuLinkCreate_v2, "cuLinkCreate_v2");

  --*
  -- * \brief Add an input to a pending linker invocation
  -- *
  -- * Ownership of \p data is retained by the caller.  No reference is retained to any
  -- * inputs after this call returns.
  -- *
  -- * This method accepts only compiler options, which are used if the data must
  -- * be compiled from PTX, and does not accept any of
  -- * ::CU_JIT_WALL_TIME, ::CU_JIT_INFO_LOG_BUFFER, ::CU_JIT_ERROR_LOG_BUFFER,
  -- * ::CU_JIT_TARGET_FROM_CUCONTEXT, or ::CU_JIT_TARGET.
  -- *
  -- * \param state        A pending linker action.
  -- * \param type         The type of the input data.
  -- * \param data         The input data.  PTX must be NULL-terminated.
  -- * \param size         The length of the input data.
  -- * \param name         An optional name for this input in log messages.
  -- * \param numOptions   Size of options.
  -- * \param options      Options to be applied only for this input (overrides options from ::cuLinkCreate).
  -- * \param optionValues Array of option values, each cast to void *.
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_INVALID_HANDLE,
  -- * ::CUDA_ERROR_INVALID_VALUE,
  -- * ::CUDA_ERROR_INVALID_IMAGE,
  -- * ::CUDA_ERROR_INVALID_PTX,
  -- * ::CUDA_ERROR_OUT_OF_MEMORY,
  -- * ::CUDA_ERROR_NO_BINARY_FOR_GPU
  -- *
  -- * \sa ::cuLinkCreate,
  -- * ::cuLinkAddFile,
  -- * ::cuLinkComplete,
  -- * ::cuLinkDestroy
  --  

   function cuLinkAddData_v2
     (state : CUlinkState;
      c_type : CUjitInputType;
      data : System.Address;
      size : stddef_h.size_t;
      name : Interfaces.C.Strings.chars_ptr;
      numOptions : unsigned;
      options : access CUjit_option;
      optionValues : System.Address) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:3834
   pragma Import (C, cuLinkAddData_v2, "cuLinkAddData_v2");

  --*
  -- * \brief Add a file input to a pending linker invocation
  -- *
  -- * No reference is retained to any inputs after this call returns.
  -- *
  -- * This method accepts only compiler options, which are used if the input
  -- * must be compiled from PTX, and does not accept any of
  -- * ::CU_JIT_WALL_TIME, ::CU_JIT_INFO_LOG_BUFFER, ::CU_JIT_ERROR_LOG_BUFFER,
  -- * ::CU_JIT_TARGET_FROM_CUCONTEXT, or ::CU_JIT_TARGET.
  -- *
  -- * This method is equivalent to invoking ::cuLinkAddData on the contents
  -- * of the file.
  -- *
  -- * \param state        A pending linker action
  -- * \param type         The type of the input data
  -- * \param path         Path to the input file
  -- * \param numOptions   Size of options
  -- * \param options      Options to be applied only for this input (overrides options from ::cuLinkCreate)
  -- * \param optionValues Array of option values, each cast to void *
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_FILE_NOT_FOUND
  -- * ::CUDA_ERROR_INVALID_HANDLE,
  -- * ::CUDA_ERROR_INVALID_VALUE,
  -- * ::CUDA_ERROR_INVALID_IMAGE,
  -- * ::CUDA_ERROR_INVALID_PTX,
  -- * ::CUDA_ERROR_OUT_OF_MEMORY,
  -- * ::CUDA_ERROR_NO_BINARY_FOR_GPU
  -- *
  -- * \sa ::cuLinkCreate,
  -- * ::cuLinkAddData,
  -- * ::cuLinkComplete,
  -- * ::cuLinkDestroy
  --  

   function cuLinkAddFile_v2
     (state : CUlinkState;
      c_type : CUjitInputType;
      path : Interfaces.C.Strings.chars_ptr;
      numOptions : unsigned;
      options : access CUjit_option;
      optionValues : System.Address) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:3873
   pragma Import (C, cuLinkAddFile_v2, "cuLinkAddFile_v2");

  --*
  -- * \brief Complete a pending linker invocation
  -- *
  -- * Completes the pending linker action and returns the cubin image for the linked
  -- * device code, which can be used with ::cuModuleLoadData.  The cubin is owned by
  -- * \p state, so it should be loaded before \p state is destroyed via ::cuLinkDestroy.
  -- * This call does not destroy \p state.
  -- *
  -- * \param state    A pending linker invocation
  -- * \param cubinOut On success, this will point to the output image
  -- * \param sizeOut  Optional parameter to receive the size of the generated image
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_INVALID_HANDLE,
  -- * ::CUDA_ERROR_OUT_OF_MEMORY
  -- *
  -- * \sa ::cuLinkCreate,
  -- * ::cuLinkAddData,
  -- * ::cuLinkAddFile,
  -- * ::cuLinkDestroy,
  -- * ::cuModuleLoadData
  --  

   function cuLinkComplete
     (state : CUlinkState;
      cubinOut : System.Address;
      sizeOut : access stddef_h.size_t) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:3900
   pragma Import (C, cuLinkComplete, "cuLinkComplete");

  --*
  -- * \brief Destroys state for a JIT linker invocation.
  -- *
  -- * \param state State object for the linker invocation
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_INVALID_HANDLE
  -- *
  -- * \sa ::cuLinkCreate
  --  

   function cuLinkDestroy (state : CUlinkState) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:3914
   pragma Import (C, cuLinkDestroy, "cuLinkDestroy");

  --* @}  
  -- END CUDA_MODULE  
  --*
  -- * \defgroup CUDA_MEM Memory Management
  -- *
  -- * ___MANBRIEF___ memory management functions of the low-level CUDA driver API
  -- * (___CURRENT_FILE___) ___ENDMANBRIEF___
  -- *
  -- * This section describes the memory management functions of the low-level CUDA
  -- * driver application programming interface.
  -- *
  -- * @{
  --  

  --*
  -- * \brief Gets free and total memory
  -- *
  -- * Returns in \p *free and \p *total respectively, the free and total amount of
  -- * memory available for allocation by the CUDA context, in bytes.
  -- *
  -- * \param free  - Returned free memory in bytes
  -- * \param total - Returned total memory in bytes
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- * \notefnerr
  -- *
  -- * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
  -- * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
  -- * ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
  -- * ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
  -- * ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
  -- * ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
  -- * ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
  -- * ::cuMemGetAddressRange, ::cuMemHostAlloc,
  -- * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D16,
  -- * ::cuMemsetD2D32, ::cuMemsetD8, ::cuMemsetD16, ::cuMemsetD32
  --  

   function cuMemGetInfo_v2 (free : access stddef_h.size_t; total : access stddef_h.size_t) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:3962
   pragma Import (C, cuMemGetInfo_v2, "cuMemGetInfo_v2");

  --*
  -- * \brief Allocates device memory
  -- *
  -- * Allocates \p bytesize bytes of linear memory on the device and returns in
  -- * \p *dptr a pointer to the allocated memory. The allocated memory is suitably
  -- * aligned for any kind of variable. The memory is not cleared. If \p bytesize
  -- * is 0, ::cuMemAlloc() returns ::CUDA_ERROR_INVALID_VALUE.
  -- *
  -- * \param dptr     - Returned device pointer
  -- * \param bytesize - Requested allocation size in bytes
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE,
  -- * ::CUDA_ERROR_OUT_OF_MEMORY
  -- * \notefnerr
  -- *
  -- * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
  -- * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAllocHost,
  -- * ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
  -- * ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
  -- * ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
  -- * ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
  -- * ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
  -- * ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
  -- * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D16,
  -- * ::cuMemsetD2D32, ::cuMemsetD8, ::cuMemsetD16, ::cuMemsetD32
  --  

   function cuMemAlloc_v2 (dptr : access CUdeviceptr; bytesize : stddef_h.size_t) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:3995
   pragma Import (C, cuMemAlloc_v2, "cuMemAlloc_v2");

  --*
  -- * \brief Allocates pitched device memory
  -- *
  -- * Allocates at least \p WidthInBytes * \p Height bytes of linear memory on
  -- * the device and returns in \p *dptr a pointer to the allocated memory. The
  -- * function may pad the allocation to ensure that corresponding pointers in
  -- * any given row will continue to meet the alignment requirements for
  -- * coalescing as the address is updated from row to row. \p ElementSizeBytes
  -- * specifies the size of the largest reads and writes that will be performed
  -- * on the memory range. \p ElementSizeBytes may be 4, 8 or 16 (since coalesced
  -- * memory transactions are not possible on other data sizes). If
  -- * \p ElementSizeBytes is smaller than the actual read/write size of a kernel,
  -- * the kernel will run correctly, but possibly at reduced speed. The pitch
  -- * returned in \p *pPitch by ::cuMemAllocPitch() is the width in bytes of the
  -- * allocation. The intended usage of pitch is as a separate parameter of the
  -- * allocation, used to compute addresses within the 2D array. Given the row
  -- * and column of an array element of type \b T, the address is computed as:
  -- * \code
  --   T* pElement = (T*)((char*)BaseAddress + Row * Pitch) + Column;
  -- * \endcode
  -- *
  -- * The pitch returned by ::cuMemAllocPitch() is guaranteed to work with
  -- * ::cuMemcpy2D() under all circumstances. For allocations of 2D arrays, it is
  -- * recommended that programmers consider performing pitch allocations using
  -- * ::cuMemAllocPitch(). Due to alignment restrictions in the hardware, this is
  -- * especially true if the application will be performing 2D memory copies
  -- * between different regions of device memory (whether linear memory or CUDA
  -- * arrays).
  -- *
  -- * The byte alignment of the pitch returned by ::cuMemAllocPitch() is guaranteed
  -- * to match or exceed the alignment requirement for texture binding with
  -- * ::cuTexRefSetAddress2D().
  -- *
  -- * \param dptr             - Returned device pointer
  -- * \param pPitch           - Returned pitch of allocation in bytes
  -- * \param WidthInBytes     - Requested allocation width in bytes
  -- * \param Height           - Requested allocation height in rows
  -- * \param ElementSizeBytes - Size of largest reads/writes for range
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE,
  -- * ::CUDA_ERROR_OUT_OF_MEMORY
  -- * \notefnerr
  -- *
  -- * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
  -- * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
  -- * ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
  -- * ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
  -- * ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
  -- * ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
  -- * ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
  -- * ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
  -- * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D16,
  -- * ::cuMemsetD2D32, ::cuMemsetD8, ::cuMemsetD16, ::cuMemsetD32
  --  

   function cuMemAllocPitch_v2
     (dptr : access CUdeviceptr;
      pPitch : access stddef_h.size_t;
      WidthInBytes : stddef_h.size_t;
      Height : stddef_h.size_t;
      ElementSizeBytes : unsigned) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:4056
   pragma Import (C, cuMemAllocPitch_v2, "cuMemAllocPitch_v2");

  --*
  -- * \brief Frees device memory
  -- *
  -- * Frees the memory space pointed to by \p dptr, which must have been returned
  -- * by a previous call to ::cuMemAlloc() or ::cuMemAllocPitch().
  -- *
  -- * \param dptr - Pointer to memory to free
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- * \notefnerr
  -- *
  -- * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
  -- * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
  -- * ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
  -- * ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
  -- * ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
  -- * ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
  -- * ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFreeHost,
  -- * ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
  -- * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D16,
  -- * ::cuMemsetD2D32, ::cuMemsetD8, ::cuMemsetD16, ::cuMemsetD32
  --  

   function cuMemFree_v2 (dptr : CUdeviceptr) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:4085
   pragma Import (C, cuMemFree_v2, "cuMemFree_v2");

  --*
  -- * \brief Get information on memory allocations
  -- *
  -- * Returns the base address in \p *pbase and size in \p *psize of the
  -- * allocation by ::cuMemAlloc() or ::cuMemAllocPitch() that contains the input
  -- * pointer \p dptr. Both parameters \p pbase and \p psize are optional. If one
  -- * of them is NULL, it is ignored.
  -- *
  -- * \param pbase - Returned base address
  -- * \param psize - Returned size of device memory allocation
  -- * \param dptr  - Device pointer to query
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- * \notefnerr
  -- *
  -- * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
  -- * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
  -- * ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
  -- * ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
  -- * ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
  -- * ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
  -- * ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
  -- * ::cuMemGetInfo, ::cuMemHostAlloc,
  -- * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D16,
  -- * ::cuMemsetD2D32, ::cuMemsetD8, ::cuMemsetD16, ::cuMemsetD32
  --  

   function cuMemGetAddressRange_v2
     (pbase : access CUdeviceptr;
      psize : access stddef_h.size_t;
      dptr : CUdeviceptr) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:4118
   pragma Import (C, cuMemGetAddressRange_v2, "cuMemGetAddressRange_v2");

  --*
  -- * \brief Allocates page-locked host memory
  -- *
  -- * Allocates \p bytesize bytes of host memory that is page-locked and
  -- * accessible to the device. The driver tracks the virtual memory ranges
  -- * allocated with this function and automatically accelerates calls to
  -- * functions such as ::cuMemcpy(). Since the memory can be accessed directly by
  -- * the device, it can be read or written with much higher bandwidth than
  -- * pageable memory obtained with functions such as ::malloc(). Allocating
  -- * excessive amounts of memory with ::cuMemAllocHost() may degrade system
  -- * performance, since it reduces the amount of memory available to the system
  -- * for paging. As a result, this function is best used sparingly to allocate
  -- * staging areas for data exchange between host and device.
  -- *
  -- * Note all host memory allocated using ::cuMemHostAlloc() will automatically
  -- * be immediately accessible to all contexts on all devices which support unified
  -- * addressing (as may be queried using ::CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING).
  -- * The device pointer that may be used to access this host memory from those 
  -- * contexts is always equal to the returned host pointer \p *pp.
  -- * See \ref CUDA_UNIFIED for additional details.
  -- *
  -- * \param pp       - Returned host pointer to page-locked memory
  -- * \param bytesize - Requested allocation size in bytes
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE,
  -- * ::CUDA_ERROR_OUT_OF_MEMORY
  -- * \notefnerr
  -- *
  -- * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
  -- * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc,
  -- * ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
  -- * ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
  -- * ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
  -- * ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
  -- * ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
  -- * ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
  -- * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D16,
  -- * ::cuMemsetD2D32, ::cuMemsetD8, ::cuMemsetD16, ::cuMemsetD32
  --  

   function cuMemAllocHost_v2 (pp : System.Address; bytesize : stddef_h.size_t) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:4164
   pragma Import (C, cuMemAllocHost_v2, "cuMemAllocHost_v2");

  --*
  -- * \brief Frees page-locked host memory
  -- *
  -- * Frees the memory space pointed to by \p p, which must have been returned by
  -- * a previous call to ::cuMemAllocHost().
  -- *
  -- * \param p - Pointer to memory to free
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- * \notefnerr
  -- *
  -- * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
  -- * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
  -- * ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
  -- * ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
  -- * ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
  -- * ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
  -- * ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree,
  -- * ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
  -- * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D16,
  -- * ::cuMemsetD2D32, ::cuMemsetD8, ::cuMemsetD16, ::cuMemsetD32
  --  

   function cuMemFreeHost (p : System.Address) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:4194
   pragma Import (C, cuMemFreeHost, "cuMemFreeHost");

  --*
  -- * \brief Allocates page-locked host memory
  -- *
  -- * Allocates \p bytesize bytes of host memory that is page-locked and accessible
  -- * to the device. The driver tracks the virtual memory ranges allocated with
  -- * this function and automatically accelerates calls to functions such as
  -- * ::cuMemcpyHtoD(). Since the memory can be accessed directly by the device,
  -- * it can be read or written with much higher bandwidth than pageable memory
  -- * obtained with functions such as ::malloc(). Allocating excessive amounts of
  -- * pinned memory may degrade system performance, since it reduces the amount
  -- * of memory available to the system for paging. As a result, this function is
  -- * best used sparingly to allocate staging areas for data exchange between
  -- * host and device.
  -- *
  -- * The \p Flags parameter enables different options to be specified that
  -- * affect the allocation, as follows.
  -- *
  -- * - ::CU_MEMHOSTALLOC_PORTABLE: The memory returned by this call will be
  -- *   considered as pinned memory by all CUDA contexts, not just the one that
  -- *   performed the allocation.
  -- *
  -- * - ::CU_MEMHOSTALLOC_DEVICEMAP: Maps the allocation into the CUDA address
  -- *   space. The device pointer to the memory may be obtained by calling
  -- *   ::cuMemHostGetDevicePointer(). This feature is available only on GPUs
  -- *   with compute capability greater than or equal to 1.1.
  -- *
  -- * - ::CU_MEMHOSTALLOC_WRITECOMBINED: Allocates the memory as write-combined
  -- *   (WC). WC memory can be transferred across the PCI Express bus more
  -- *   quickly on some system configurations, but cannot be read efficiently by
  -- *   most CPUs. WC memory is a good option for buffers that will be written by
  -- *   the CPU and read by the GPU via mapped pinned memory or host->device
  -- *   transfers.
  -- *
  -- * All of these flags are orthogonal to one another: a developer may allocate
  -- * memory that is portable, mapped and/or write-combined with no restrictions.
  -- *
  -- * The CUDA context must have been created with the ::CU_CTX_MAP_HOST flag in
  -- * order for the ::CU_MEMHOSTALLOC_DEVICEMAP flag to have any effect.
  -- *
  -- * The ::CU_MEMHOSTALLOC_DEVICEMAP flag may be specified on CUDA contexts for
  -- * devices that do not support mapped pinned memory. The failure is deferred
  -- * to ::cuMemHostGetDevicePointer() because the memory may be mapped into
  -- * other CUDA contexts via the ::CU_MEMHOSTALLOC_PORTABLE flag.
  -- *
  -- * The memory allocated by this function must be freed with ::cuMemFreeHost().
  -- *
  -- * Note all host memory allocated using ::cuMemHostAlloc() will automatically
  -- * be immediately accessible to all contexts on all devices which support unified
  -- * addressing (as may be queried using ::CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING).
  -- * Unless the flag ::CU_MEMHOSTALLOC_WRITECOMBINED is specified, the device pointer 
  -- * that may be used to access this host memory from those contexts is always equal 
  -- * to the returned host pointer \p *pp.  If the flag ::CU_MEMHOSTALLOC_WRITECOMBINED
  -- * is specified, then the function ::cuMemHostGetDevicePointer() must be used
  -- * to query the device pointer, even if the context supports unified addressing.
  -- * See \ref CUDA_UNIFIED for additional details.
  -- *
  -- * \param pp       - Returned host pointer to page-locked memory
  -- * \param bytesize - Requested allocation size in bytes
  -- * \param Flags    - Flags for allocation request
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE,
  -- * ::CUDA_ERROR_OUT_OF_MEMORY
  -- * \notefnerr
  -- *
  -- * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
  -- * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
  -- * ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
  -- * ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
  -- * ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
  -- * ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
  -- * ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
  -- * ::cuMemGetAddressRange, ::cuMemGetInfo,
  -- * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D16,
  -- * ::cuMemsetD2D32, ::cuMemsetD8, ::cuMemsetD16, ::cuMemsetD32
  --  

   function cuMemHostAlloc
     (pp : System.Address;
      bytesize : stddef_h.size_t;
      Flags : unsigned) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:4276
   pragma Import (C, cuMemHostAlloc, "cuMemHostAlloc");

  --*
  -- * \brief Passes back device pointer of mapped pinned memory
  -- *
  -- * Passes back the device pointer \p pdptr corresponding to the mapped, pinned
  -- * host buffer \p p allocated by ::cuMemHostAlloc.
  -- *
  -- * ::cuMemHostGetDevicePointer() will fail if the ::CU_MEMHOSTALLOC_DEVICEMAP
  -- * flag was not specified at the time the memory was allocated, or if the
  -- * function is called on a GPU that does not support mapped pinned memory.
  -- *
  -- * For devices that have a non-zero value for the device attribute
  -- * ::CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM, the memory
  -- * can also be accessed from the device using the host pointer \p p.
  -- * The device pointer returned by ::cuMemHostGetDevicePointer() may or may not
  -- * match the original host pointer \p p and depends on the devices visible to the
  -- * application. If all devices visible to the application have a non-zero value for the
  -- * device attribute, the device pointer returned by ::cuMemHostGetDevicePointer()
  -- * will match the original pointer \p p. If any device visible to the application
  -- * has a zero value for the device attribute, the device pointer returned by
  -- * ::cuMemHostGetDevicePointer() will not match the original host pointer \p p,
  -- * but it will be suitable for use on all devices provided Unified Virtual Addressing
  -- * is enabled. In such systems, it is valid to access the memory using either pointer
  -- * on devices that have a non-zero value for the device attribute. Note however that
  -- * such devices should access the memory using only of the two pointers and not both.
  -- *
  -- * \p Flags provides for future releases. For now, it must be set to 0.
  -- *
  -- * \param pdptr - Returned device pointer
  -- * \param p     - Host pointer
  -- * \param Flags - Options (must be 0)
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- * \notefnerr
  -- *
  -- * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
  -- * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
  -- * ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
  -- * ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
  -- * ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
  -- * ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
  -- * ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
  -- * ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
  -- * ::cuMemsetD2D8, ::cuMemsetD2D16,
  -- * ::cuMemsetD2D32, ::cuMemsetD8, ::cuMemsetD16, ::cuMemsetD32
  --  

   function cuMemHostGetDevicePointer_v2
     (pdptr : access CUdeviceptr;
      p : System.Address;
      Flags : unsigned) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:4329
   pragma Import (C, cuMemHostGetDevicePointer_v2, "cuMemHostGetDevicePointer_v2");

  --*
  -- * \brief Passes back flags that were used for a pinned allocation
  -- *
  -- * Passes back the flags \p pFlags that were specified when allocating
  -- * the pinned host buffer \p p allocated by ::cuMemHostAlloc.
  -- *
  -- * ::cuMemHostGetFlags() will fail if the pointer does not reside in
  -- * an allocation performed by ::cuMemAllocHost() or ::cuMemHostAlloc().
  -- *
  -- * \param pFlags - Returned flags word
  -- * \param p     - Host pointer
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- * \notefnerr
  -- *
  -- * \sa ::cuMemAllocHost, ::cuMemHostAlloc
  --  

   function cuMemHostGetFlags (pFlags : access unsigned; p : System.Address) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:4354
   pragma Import (C, cuMemHostGetFlags, "cuMemHostGetFlags");

  --*
  -- * \brief Allocates memory that will be automatically managed by the Unified Memory system
  -- *
  -- * Allocates \p bytesize bytes of managed memory on the device and returns in
  -- * \p *dptr a pointer to the allocated memory. If the device doesn't support
  -- * allocating managed memory, ::CUDA_ERROR_NOT_SUPPORTED is returned. Support
  -- * for managed memory can be queried using the device attribute
  -- * ::CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY. The allocated memory is suitably
  -- * aligned for any kind of variable. The memory is not cleared. If \p bytesize
  -- * is 0, ::cuMemAllocManaged returns ::CUDA_ERROR_INVALID_VALUE. The pointer
  -- * is valid on the CPU and on all GPUs in the system that support managed memory.
  -- * All accesses to this pointer must obey the Unified Memory programming model.
  -- *
  -- * \p flags specifies the default stream association for this allocation.
  -- * \p flags must be one of ::CU_MEM_ATTACH_GLOBAL or ::CU_MEM_ATTACH_HOST. If
  -- * ::CU_MEM_ATTACH_GLOBAL is specified, then this memory is accessible from
  -- * any stream on any device. If ::CU_MEM_ATTACH_HOST is specified, then the
  -- * allocation should not be accessed from devices that have a zero value for the
  -- * device attribute ::CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS; an explicit call to
  -- * ::cuStreamAttachMemAsync will be required to enable access on such devices.
  -- *
  -- * If the association is later changed via ::cuStreamAttachMemAsync to
  -- * a single stream, the default association as specifed during ::cuMemAllocManaged
  -- * is restored when that stream is destroyed. For __managed__ variables, the
  -- * default association is always ::CU_MEM_ATTACH_GLOBAL. Note that destroying a
  -- * stream is an asynchronous operation, and as a result, the change to default
  -- * association won't happen until all work in the stream has completed.
  -- *
  -- * Memory allocated with ::cuMemAllocManaged should be released with ::cuMemFree.
  -- *
  -- * Device memory oversubscription is possible for GPUs that have a non-zero value for the
  -- * device attribute ::CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS. Managed memory on
  -- * such GPUs may be evicted from device memory to host memory at any time by the Unified
  -- * Memory driver in order to make room for other allocations.
  -- *
  -- * In a multi-GPU system where all GPUs have a non-zero value for the device attribute
  -- * ::CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS, managed memory may not be populated when this
  -- * API returns and instead may be populated on access. In such systems, managed memory can
  -- * migrate to any processor's memory at any time. The Unified Memory driver will employ heuristics to
  -- * maintain data locality and prevent excessive page faults to the extent possible. The application
  -- * can also guide the driver about memory usage patterns via ::cuMemAdvise. The application
  -- * can also explicitly migrate memory to a desired processor's memory via
  -- * ::cuMemPrefetchAsync.
  -- *
  -- * In a multi-GPU system where all of the GPUs have a zero value for the device attribute
  -- * ::CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS and all the GPUs have peer-to-peer support
  -- * with each other, the physical storage for managed memory is created on the GPU which is active
  -- * at the time ::cuMemAllocManaged is called. All other GPUs will reference the data at reduced
  -- * bandwidth via peer mappings over the PCIe bus. The Unified Memory driver does not migrate
  -- * memory among such GPUs.
  -- *
  -- * In a multi-GPU system where not all GPUs have peer-to-peer support with each other and
  -- * where the value of the device attribute ::CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS
  -- * is zero for at least one of those GPUs, the location chosen for physical storage of managed
  -- * memory is system-dependent.
  -- * - On Linux, the location chosen will be device memory as long as the current set of active
  -- * contexts are on devices that either have peer-to-peer support with each other or have a
  -- * non-zero value for the device attribute ::CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS.
  -- * If there is an active context on a GPU that does not have a non-zero value for that device
  -- * attribute and it does not have peer-to-peer support with the other devices that have active
  -- * contexts on them, then the location for physical storage will be 'zero-copy' or host memory.
  -- * Note that this means that managed memory that is located in device memory is migrated to
  -- * host memory if a new context is created on a GPU that doesn't have a non-zero value for
  -- * the device attribute and does not support peer-to-peer with at least one of the other devices
  -- * that has an active context. This in turn implies that context creation may fail if there is
  -- * insufficient host memory to migrate all managed allocations.
  -- * - On Windows, the physical storage is always created in 'zero-copy' or host memory.
  -- * All GPUs will reference the data at reduced bandwidth over the PCIe bus. In these
  -- * circumstances, use of the environment variable CUDA_VISIBLE_DEVICES is recommended to
  -- * restrict CUDA to only use those GPUs that have peer-to-peer support.
  -- * Alternatively, users can also set CUDA_MANAGED_FORCE_DEVICE_ALLOC to a
  -- * non-zero value to force the driver to always use device memory for physical storage.
  -- * When this environment variable is set to a non-zero value, all contexts created in
  -- * that process on devices that support managed memory have to be peer-to-peer compatible
  -- * with each other. Context creation will fail if a context is created on a device that
  -- * supports managed memory and is not peer-to-peer compatible with any of the other
  -- * managed memory supporting devices on which contexts were previously created, even if
  -- * those contexts have been destroyed. These environment variables are described
  -- * in the CUDA programming guide under the "CUDA environment variables" section.
  -- *
  -- * \param dptr     - Returned device pointer
  -- * \param bytesize - Requested allocation size in bytes
  -- * \param flags    - Must be one of ::CU_MEM_ATTACH_GLOBAL or ::CU_MEM_ATTACH_HOST
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_NOT_SUPPORTED,
  -- * ::CUDA_ERROR_INVALID_VALUE,
  -- * ::CUDA_ERROR_OUT_OF_MEMORY
  -- * \notefnerr
  -- *
  -- * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
  -- * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAllocHost,
  -- * ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
  -- * ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
  -- * ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
  -- * ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
  -- * ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
  -- * ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
  -- * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D16,
  -- * ::cuMemsetD2D32, ::cuMemsetD8, ::cuMemsetD16, ::cuMemsetD32,
  -- * ::cuDeviceGetAttribute, ::cuStreamAttachMemAsync
  --  

   function cuMemAllocManaged
     (dptr : access CUdeviceptr;
      bytesize : stddef_h.size_t;
      flags : unsigned) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:4464
   pragma Import (C, cuMemAllocManaged, "cuMemAllocManaged");

  --*
  -- * \brief Returns a handle to a compute device
  -- *
  -- * Returns in \p *device a device handle given a PCI bus ID string.
  -- *
  -- * \param dev      - Returned device handle
  -- *
  -- * \param pciBusId - String in one of the following forms: 
  -- * [domain]:[bus]:[device].[function]
  -- * [domain]:[bus]:[device]
  -- * [bus]:[device].[function]
  -- * where \p domain, \p bus, \p device, and \p function are all hexadecimal values
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_VALUE,
  -- * ::CUDA_ERROR_INVALID_DEVICE
  -- * \notefnerr
  -- *
  -- * \sa ::cuDeviceGet, ::cuDeviceGetAttribute, ::cuDeviceGetPCIBusId
  --  

   function cuDeviceGetByPCIBusId (dev : access CUdevice; pciBusId : Interfaces.C.Strings.chars_ptr) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:4493
   pragma Import (C, cuDeviceGetByPCIBusId, "cuDeviceGetByPCIBusId");

  --*
  -- * \brief Returns a PCI Bus Id string for the device
  -- *
  -- * Returns an ASCII string identifying the device \p dev in the NULL-terminated
  -- * string pointed to by \p pciBusId. \p len specifies the maximum length of the
  -- * string that may be returned.
  -- *
  -- * \param pciBusId - Returned identifier string for the device in the following format
  -- * [domain]:[bus]:[device].[function]
  -- * where \p domain, \p bus, \p device, and \p function are all hexadecimal values.
  -- * pciBusId should be large enough to store 13 characters including the NULL-terminator.
  -- *
  -- * \param len      - Maximum length of string to store in \p name
  -- *
  -- * \param dev      - Device to get identifier string for
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_VALUE,
  -- * ::CUDA_ERROR_INVALID_DEVICE
  -- * \notefnerr
  -- *
  -- * \sa ::cuDeviceGet, ::cuDeviceGetAttribute, ::cuDeviceGetByPCIBusId
  --  

   function cuDeviceGetPCIBusId
     (pciBusId : Interfaces.C.Strings.chars_ptr;
      len : int;
      dev : CUdevice) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:4521
   pragma Import (C, cuDeviceGetPCIBusId, "cuDeviceGetPCIBusId");

  --*
  -- * \brief Gets an interprocess handle for a previously allocated event
  -- *
  -- * Takes as input a previously allocated event. This event must have been 
  -- * created with the ::CU_EVENT_INTERPROCESS and ::CU_EVENT_DISABLE_TIMING 
  -- * flags set. This opaque handle may be copied into other processes and
  -- * opened with ::cuIpcOpenEventHandle to allow efficient hardware
  -- * synchronization between GPU work in different processes.
  -- *
  -- * After the event has been opened in the importing process, 
  -- * ::cuEventRecord, ::cuEventSynchronize, ::cuStreamWaitEvent and 
  -- * ::cuEventQuery may be used in either process. Performing operations 
  -- * on the imported event after the exported event has been freed 
  -- * with ::cuEventDestroy will result in undefined behavior.
  -- *
  -- * IPC functionality is restricted to devices with support for unified 
  -- * addressing on Linux operating systems.
  -- *
  -- * \param pHandle - Pointer to a user allocated CUipcEventHandle
  -- *                    in which to return the opaque event handle
  -- * \param event   - Event allocated with ::CU_EVENT_INTERPROCESS and 
  -- *                    ::CU_EVENT_DISABLE_TIMING flags.
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_INVALID_HANDLE,
  -- * ::CUDA_ERROR_OUT_OF_MEMORY,
  -- * ::CUDA_ERROR_MAP_FAILED
  -- *
  -- * \sa 
  -- * ::cuEventCreate, 
  -- * ::cuEventDestroy, 
  -- * ::cuEventSynchronize,
  -- * ::cuEventQuery,
  -- * ::cuStreamWaitEvent,
  -- * ::cuIpcOpenEventHandle,
  -- * ::cuIpcGetMemHandle,
  -- * ::cuIpcOpenMemHandle,
  -- * ::cuIpcCloseMemHandle
  --  

   function cuIpcGetEventHandle (pHandle : access CUipcEventHandle; event : CUevent) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:4563
   pragma Import (C, cuIpcGetEventHandle, "cuIpcGetEventHandle");

  --*
  -- * \brief Opens an interprocess event handle for use in the current process
  -- *
  -- * Opens an interprocess event handle exported from another process with 
  -- * ::cuIpcGetEventHandle. This function returns a ::CUevent that behaves like 
  -- * a locally created event with the ::CU_EVENT_DISABLE_TIMING flag specified. 
  -- * This event must be freed with ::cuEventDestroy.
  -- *
  -- * Performing operations on the imported event after the exported event has 
  -- * been freed with ::cuEventDestroy will result in undefined behavior.
  -- *
  -- * IPC functionality is restricted to devices with support for unified 
  -- * addressing on Linux operating systems.
  -- *
  -- * \param phEvent - Returns the imported event
  -- * \param handle  - Interprocess handle to open
  -- *
  -- * \returns
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_MAP_FAILED,
  -- * ::CUDA_ERROR_PEER_ACCESS_UNSUPPORTED,
  -- * ::CUDA_ERROR_INVALID_HANDLE
  -- *
  -- * \sa
  -- * ::cuEventCreate, 
  -- * ::cuEventDestroy, 
  -- * ::cuEventSynchronize,
  -- * ::cuEventQuery,
  -- * ::cuStreamWaitEvent,
  -- * ::cuIpcGetEventHandle,
  -- * ::cuIpcGetMemHandle,
  -- * ::cuIpcOpenMemHandle,
  -- * ::cuIpcCloseMemHandle
  --  

   function cuIpcOpenEventHandle (phEvent : System.Address; handle : CUipcEventHandle) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:4600
   pragma Import (C, cuIpcOpenEventHandle, "cuIpcOpenEventHandle");

  --*
  -- * \brief Gets an interprocess memory handle for an existing device memory
  -- * allocation
  -- *
  -- * Takes a pointer to the base of an existing device memory allocation created 
  -- * with ::cuMemAlloc and exports it for use in another process. This is a 
  -- * lightweight operation and may be called multiple times on an allocation
  -- * without adverse effects. 
  -- *
  -- * If a region of memory is freed with ::cuMemFree and a subsequent call
  -- * to ::cuMemAlloc returns memory with the same device address,
  -- * ::cuIpcGetMemHandle will return a unique handle for the
  -- * new memory. 
  -- *
  -- * IPC functionality is restricted to devices with support for unified 
  -- * addressing on Linux operating systems.
  -- *
  -- * \param pHandle - Pointer to user allocated ::CUipcMemHandle to return
  -- *                    the handle in.
  -- * \param dptr    - Base pointer to previously allocated device memory 
  -- *
  -- * \returns
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_INVALID_HANDLE,
  -- * ::CUDA_ERROR_OUT_OF_MEMORY,
  -- * ::CUDA_ERROR_MAP_FAILED,
  -- * 
  -- * \sa
  -- * ::cuMemAlloc,
  -- * ::cuMemFree,
  -- * ::cuIpcGetEventHandle,
  -- * ::cuIpcOpenEventHandle,
  -- * ::cuIpcOpenMemHandle,
  -- * ::cuIpcCloseMemHandle
  --  

   function cuIpcGetMemHandle (pHandle : access CUipcMemHandle; dptr : CUdeviceptr) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:4637
   pragma Import (C, cuIpcGetMemHandle, "cuIpcGetMemHandle");

  --*
  -- * \brief Opens an interprocess memory handle exported from another process
  -- * and returns a device pointer usable in the local process.
  -- *
  -- * Maps memory exported from another process with ::cuIpcGetMemHandle into
  -- * the current device address space. For contexts on different devices 
  -- * ::cuIpcOpenMemHandle can attempt to enable peer access between the
  -- * devices as if the user called ::cuCtxEnablePeerAccess. This behavior is 
  -- * controlled by the ::CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS flag. 
  -- * ::cuDeviceCanAccessPeer can determine if a mapping is possible.
  -- *
  -- * Contexts that may open ::CUipcMemHandles are restricted in the following way.
  -- * ::CUipcMemHandles from each ::CUdevice in a given process may only be opened 
  -- * by one ::CUcontext per ::CUdevice per other process.
  -- *
  -- * Memory returned from ::cuIpcOpenMemHandle must be freed with
  -- * ::cuIpcCloseMemHandle.
  -- *
  -- * Calling ::cuMemFree on an exported memory region before calling
  -- * ::cuIpcCloseMemHandle in the importing context will result in undefined
  -- * behavior.
  -- *
  -- * IPC functionality is restricted to devices with support for unified 
  -- * addressing on Linux operating systems.
  -- * 
  -- * \param pdptr  - Returned device pointer
  -- * \param handle - ::CUipcMemHandle to open
  -- * \param Flags  - Flags for this operation. Must be specified as ::CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS
  -- *
  -- * \returns
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_MAP_FAILED,
  -- * ::CUDA_ERROR_INVALID_HANDLE,
  -- * ::CUDA_ERROR_TOO_MANY_PEERS
  -- *
  -- * \note No guarantees are made about the address returned in \p *pdptr.  
  -- * In particular, multiple processes may not receive the same address for the same \p handle.
  -- *
  -- * \sa
  -- * ::cuMemAlloc,
  -- * ::cuMemFree,
  -- * ::cuIpcGetEventHandle,
  -- * ::cuIpcOpenEventHandle,
  -- * ::cuIpcGetMemHandle,
  -- * ::cuIpcCloseMemHandle,
  -- * ::cuCtxEnablePeerAccess,
  -- * ::cuDeviceCanAccessPeer,
  --  

   function cuIpcOpenMemHandle
     (pdptr : access CUdeviceptr;
      handle : CUipcMemHandle;
      Flags : unsigned) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:4688
   pragma Import (C, cuIpcOpenMemHandle, "cuIpcOpenMemHandle");

  --*
  -- * \brief Close memory mapped with ::cuIpcOpenMemHandle
  -- * 
  -- * Unmaps memory returnd by ::cuIpcOpenMemHandle. The original allocation
  -- * in the exporting process as well as imported mappings in other processes
  -- * will be unaffected.
  -- *
  -- * Any resources used to enable peer access will be freed if this is the
  -- * last mapping using them.
  -- *
  -- * IPC functionality is restricted to devices with support for unified 
  -- * addressing on Linux operating systems.
  -- *
  -- * \param dptr - Device pointer returned by ::cuIpcOpenMemHandle
  -- * 
  -- * \returns
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_MAP_FAILED,
  -- * ::CUDA_ERROR_INVALID_HANDLE,
  -- *
  -- * \sa
  -- * ::cuMemAlloc,
  -- * ::cuMemFree,
  -- * ::cuIpcGetEventHandle,
  -- * ::cuIpcOpenEventHandle,
  -- * ::cuIpcGetMemHandle,
  -- * ::cuIpcOpenMemHandle,
  --  

   function cuIpcCloseMemHandle (dptr : CUdeviceptr) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:4719
   pragma Import (C, cuIpcCloseMemHandle, "cuIpcCloseMemHandle");

  --*
  -- * \brief Registers an existing host memory range for use by CUDA
  -- *
  -- * Page-locks the memory range specified by \p p and \p bytesize and maps it
  -- * for the device(s) as specified by \p Flags. This memory range also is added
  -- * to the same tracking mechanism as ::cuMemHostAlloc to automatically accelerate
  -- * calls to functions such as ::cuMemcpyHtoD(). Since the memory can be accessed 
  -- * directly by the device, it can be read or written with much higher bandwidth 
  -- * than pageable memory that has not been registered.  Page-locking excessive
  -- * amounts of memory may degrade system performance, since it reduces the amount
  -- * of memory available to the system for paging. As a result, this function is
  -- * best used sparingly to register staging areas for data exchange between
  -- * host and device.
  -- *
  -- * This function has limited support on Mac OS X. OS 10.7 or higher is required.
  -- *
  -- * The \p Flags parameter enables different options to be specified that
  -- * affect the allocation, as follows.
  -- *
  -- * - ::CU_MEMHOSTREGISTER_PORTABLE: The memory returned by this call will be
  -- *   considered as pinned memory by all CUDA contexts, not just the one that
  -- *   performed the allocation.
  -- *
  -- * - ::CU_MEMHOSTREGISTER_DEVICEMAP: Maps the allocation into the CUDA address
  -- *   space. The device pointer to the memory may be obtained by calling
  -- *   ::cuMemHostGetDevicePointer(). This feature is available only on GPUs
  -- *   with compute capability greater than or equal to 1.1.
  -- *
  -- * - ::CU_MEMHOSTREGISTER_IOMEMORY: The pointer is treated as pointing to some
  -- *   I/O memory space, e.g. the PCI Express resource of a 3rd party device.
  -- *
  -- * All of these flags are orthogonal to one another: a developer may page-lock
  -- * memory that is portable or mapped with no restrictions.
  -- *
  -- * The CUDA context must have been created with the ::CU_CTX_MAP_HOST flag in
  -- * order for the ::CU_MEMHOSTREGISTER_DEVICEMAP flag to have any effect.
  -- *
  -- * The ::CU_MEMHOSTREGISTER_DEVICEMAP flag may be specified on CUDA contexts for
  -- * devices that do not support mapped pinned memory. The failure is deferred
  -- * to ::cuMemHostGetDevicePointer() because the memory may be mapped into
  -- * other CUDA contexts via the ::CU_MEMHOSTREGISTER_PORTABLE flag.
  -- *
  -- * For devices that have a non-zero value for the device attribute
  -- * ::CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM, the memory
  -- * can also be accessed from the device using the host pointer \p p.
  -- * The device pointer returned by ::cuMemHostGetDevicePointer() may or may not
  -- * match the original host pointer \p ptr and depends on the devices visible to the
  -- * application. If all devices visible to the application have a non-zero value for the
  -- * device attribute, the device pointer returned by ::cuMemHostGetDevicePointer()
  -- * will match the original pointer \p ptr. If any device visible to the application
  -- * has a zero value for the device attribute, the device pointer returned by
  -- * ::cuMemHostGetDevicePointer() will not match the original host pointer \p ptr,
  -- * but it will be suitable for use on all devices provided Unified Virtual Addressing
  -- * is enabled. In such systems, it is valid to access the memory using either pointer
  -- * on devices that have a non-zero value for the device attribute. Note however that
  -- * such devices should access the memory using only of the two pointers and not both.
  -- *
  -- * The memory page-locked by this function must be unregistered with 
  -- * ::cuMemHostUnregister().
  -- *
  -- * \param p        - Host pointer to memory to page-lock
  -- * \param bytesize - Size in bytes of the address range to page-lock
  -- * \param Flags    - Flags for allocation request
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE,
  -- * ::CUDA_ERROR_OUT_OF_MEMORY,
  -- * ::CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED,
  -- * ::CUDA_ERROR_NOT_PERMITTED,
  -- * ::CUDA_ERROR_NOT_SUPPORTED
  -- * \notefnerr
  -- *
  -- * \sa ::cuMemHostUnregister, ::cuMemHostGetFlags, ::cuMemHostGetDevicePointer
  --  

   function cuMemHostRegister_v2
     (p : System.Address;
      bytesize : stddef_h.size_t;
      Flags : unsigned) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:4802
   pragma Import (C, cuMemHostRegister_v2, "cuMemHostRegister_v2");

  --*
  -- * \brief Unregisters a memory range that was registered with cuMemHostRegister.
  -- *
  -- * Unmaps the memory range whose base address is specified by \p p, and makes
  -- * it pageable again.
  -- *
  -- * The base address must be the same one specified to ::cuMemHostRegister().
  -- *
  -- * \param p - Host pointer to memory to unregister
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE,
  -- * ::CUDA_ERROR_OUT_OF_MEMORY,
  -- * ::CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED,
  -- * \notefnerr
  -- *
  -- * \sa ::cuMemHostRegister
  --  

   function cuMemHostUnregister (p : System.Address) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:4826
   pragma Import (C, cuMemHostUnregister, "cuMemHostUnregister");

  --*
  -- * \brief Copies memory
  -- *
  -- * Copies data between two pointers. 
  -- * \p dst and \p src are base pointers of the destination and source, respectively.  
  -- * \p ByteCount specifies the number of bytes to copy.
  -- * Note that this function infers the type of the transfer (host to host, host to 
  -- *   device, device to device, or device to host) from the pointer values.  This
  -- *   function is only allowed in contexts which support unified addressing.
  -- *
  -- * \param dst - Destination unified virtual address space pointer
  -- * \param src - Source unified virtual address space pointer
  -- * \param ByteCount - Size of memory copy in bytes
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- * \notefnerr
  -- * \note_sync
  -- *
  -- * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
  -- * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
  -- * ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
  -- * ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
  -- * ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA,
  -- * ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
  -- * ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
  -- * ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
  -- * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D16,
  -- * ::cuMemsetD2D32, ::cuMemsetD8, ::cuMemsetD16, ::cuMemsetD32
  --  

   function cuMemcpy
     (dst : CUdeviceptr;
      src : CUdeviceptr;
      ByteCount : stddef_h.size_t) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:4862
   pragma Import (C, cuMemcpy, "cuMemcpy");

  --*
  -- * \brief Copies device memory between two contexts
  -- *
  -- * Copies from device memory in one context to device memory in another
  -- * context. \p dstDevice is the base device pointer of the destination memory 
  -- * and \p dstContext is the destination context.  \p srcDevice is the base 
  -- * device pointer of the source memory and \p srcContext is the source pointer.  
  -- * \p ByteCount specifies the number of bytes to copy.
  -- *
  -- * \param dstDevice  - Destination device pointer
  -- * \param dstContext - Destination context
  -- * \param srcDevice  - Source device pointer
  -- * \param srcContext - Source context
  -- * \param ByteCount  - Size of memory copy in bytes
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- * \notefnerr
  -- * \note_sync
  -- *
  -- * \sa ::cuMemcpyDtoD, ::cuMemcpy3DPeer, ::cuMemcpyDtoDAsync, ::cuMemcpyPeerAsync,
  -- * ::cuMemcpy3DPeerAsync
  --  

   function cuMemcpyPeer
     (dstDevice : CUdeviceptr;
      dstContext : CUcontext;
      srcDevice : CUdeviceptr;
      srcContext : CUcontext;
      ByteCount : stddef_h.size_t) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:4891
   pragma Import (C, cuMemcpyPeer, "cuMemcpyPeer");

  --*
  -- * \brief Copies memory from Host to Device
  -- *
  -- * Copies from host memory to device memory. \p dstDevice and \p srcHost are
  -- * the base addresses of the destination and source, respectively. \p ByteCount
  -- * specifies the number of bytes to copy.
  -- *
  -- * \param dstDevice - Destination device pointer
  -- * \param srcHost   - Source host pointer
  -- * \param ByteCount - Size of memory copy in bytes
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- * \notefnerr
  -- * \note_sync
  -- *
  -- * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
  -- * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
  -- * ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
  -- * ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
  -- * ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
  -- * ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
  -- * ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
  -- * ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
  -- * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D16,
  -- * ::cuMemsetD2D32, ::cuMemsetD8, ::cuMemsetD16, ::cuMemsetD32
  --  

   function cuMemcpyHtoD_v2
     (dstDevice : CUdeviceptr;
      srcHost : System.Address;
      ByteCount : stddef_h.size_t) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:4927
   pragma Import (C, cuMemcpyHtoD_v2, "cuMemcpyHtoD_v2");

  --*
  -- * \brief Copies memory from Device to Host
  -- *
  -- * Copies from device to host memory. \p dstHost and \p srcDevice specify the
  -- * base pointers of the destination and source, respectively. \p ByteCount
  -- * specifies the number of bytes to copy.
  -- *
  -- * \param dstHost   - Destination host pointer
  -- * \param srcDevice - Source device pointer
  -- * \param ByteCount - Size of memory copy in bytes
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- * \notefnerr
  -- * \note_sync
  -- *
  -- * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
  -- * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
  -- * ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
  -- * ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
  -- * ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
  -- * ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
  -- * ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
  -- * ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
  -- * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D16,
  -- * ::cuMemsetD2D32, ::cuMemsetD8, ::cuMemsetD16, ::cuMemsetD32
  --  

   function cuMemcpyDtoH_v2
     (dstHost : System.Address;
      srcDevice : CUdeviceptr;
      ByteCount : stddef_h.size_t) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:4960
   pragma Import (C, cuMemcpyDtoH_v2, "cuMemcpyDtoH_v2");

  --*
  -- * \brief Copies memory from Device to Device
  -- *
  -- * Copies from device memory to device memory. \p dstDevice and \p srcDevice
  -- * are the base pointers of the destination and source, respectively.
  -- * \p ByteCount specifies the number of bytes to copy.
  -- *
  -- * \param dstDevice - Destination device pointer
  -- * \param srcDevice - Source device pointer
  -- * \param ByteCount - Size of memory copy in bytes
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- * \notefnerr
  -- * \note_sync
  -- *
  -- * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
  -- * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
  -- * ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
  -- * ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
  -- * ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA,
  -- * ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
  -- * ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
  -- * ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
  -- * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D16,
  -- * ::cuMemsetD2D32, ::cuMemsetD8, ::cuMemsetD16, ::cuMemsetD32
  --  

   function cuMemcpyDtoD_v2
     (dstDevice : CUdeviceptr;
      srcDevice : CUdeviceptr;
      ByteCount : stddef_h.size_t) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:4993
   pragma Import (C, cuMemcpyDtoD_v2, "cuMemcpyDtoD_v2");

  --*
  -- * \brief Copies memory from Device to Array
  -- *
  -- * Copies from device memory to a 1D CUDA array. \p dstArray and \p dstOffset
  -- * specify the CUDA array handle and starting index of the destination data.
  -- * \p srcDevice specifies the base pointer of the source. \p ByteCount
  -- * specifies the number of bytes to copy.
  -- *
  -- * \param dstArray  - Destination array
  -- * \param dstOffset - Offset in bytes of destination array
  -- * \param srcDevice - Source device pointer
  -- * \param ByteCount - Size of memory copy in bytes
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- * \notefnerr
  -- * \note_sync
  -- *
  -- * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
  -- * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
  -- * ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
  -- * ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
  -- * ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
  -- * ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
  -- * ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
  -- * ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
  -- * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D16,
  -- * ::cuMemsetD2D32, ::cuMemsetD8, ::cuMemsetD16, ::cuMemsetD32
  --  

   function cuMemcpyDtoA_v2
     (dstArray : CUarray;
      dstOffset : stddef_h.size_t;
      srcDevice : CUdeviceptr;
      ByteCount : stddef_h.size_t) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:5028
   pragma Import (C, cuMemcpyDtoA_v2, "cuMemcpyDtoA_v2");

  --*
  -- * \brief Copies memory from Array to Device
  -- *
  -- * Copies from one 1D CUDA array to device memory. \p dstDevice specifies the
  -- * base pointer of the destination and must be naturally aligned with the CUDA
  -- * array elements. \p srcArray and \p srcOffset specify the CUDA array handle
  -- * and the offset in bytes into the array where the copy is to begin.
  -- * \p ByteCount specifies the number of bytes to copy and must be evenly
  -- * divisible by the array element size.
  -- *
  -- * \param dstDevice - Destination device pointer
  -- * \param srcArray  - Source array
  -- * \param srcOffset - Offset in bytes of source array
  -- * \param ByteCount - Size of memory copy in bytes
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- * \notefnerr
  -- * \note_sync
  -- *
  -- * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
  -- * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
  -- * ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
  -- * ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA,
  -- * ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
  -- * ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
  -- * ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
  -- * ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
  -- * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D16,
  -- * ::cuMemsetD2D32, ::cuMemsetD8, ::cuMemsetD16, ::cuMemsetD32
  --  

   function cuMemcpyAtoD_v2
     (dstDevice : CUdeviceptr;
      srcArray : CUarray;
      srcOffset : stddef_h.size_t;
      ByteCount : stddef_h.size_t) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:5065
   pragma Import (C, cuMemcpyAtoD_v2, "cuMemcpyAtoD_v2");

  --*
  -- * \brief Copies memory from Host to Array
  -- *
  -- * Copies from host memory to a 1D CUDA array. \p dstArray and \p dstOffset
  -- * specify the CUDA array handle and starting offset in bytes of the destination
  -- * data.  \p pSrc specifies the base address of the source. \p ByteCount specifies
  -- * the number of bytes to copy.
  -- *
  -- * \param dstArray  - Destination array
  -- * \param dstOffset - Offset in bytes of destination array
  -- * \param srcHost   - Source host pointer
  -- * \param ByteCount - Size of memory copy in bytes
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- * \notefnerr
  -- * \note_sync
  -- *
  -- * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
  -- * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
  -- * ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
  -- * ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
  -- * ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
  -- * ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoAAsync,
  -- * ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
  -- * ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
  -- * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D16,
  -- * ::cuMemsetD2D32, ::cuMemsetD8, ::cuMemsetD16, ::cuMemsetD32
  --  

   function cuMemcpyHtoA_v2
     (dstArray : CUarray;
      dstOffset : stddef_h.size_t;
      srcHost : System.Address;
      ByteCount : stddef_h.size_t) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:5100
   pragma Import (C, cuMemcpyHtoA_v2, "cuMemcpyHtoA_v2");

  --*
  -- * \brief Copies memory from Array to Host
  -- *
  -- * Copies from one 1D CUDA array to host memory. \p dstHost specifies the base
  -- * pointer of the destination. \p srcArray and \p srcOffset specify the CUDA
  -- * array handle and starting offset in bytes of the source data.
  -- * \p ByteCount specifies the number of bytes to copy.
  -- *
  -- * \param dstHost   - Destination device pointer
  -- * \param srcArray  - Source array
  -- * \param srcOffset - Offset in bytes of source array
  -- * \param ByteCount - Size of memory copy in bytes
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- * \notefnerr
  -- * \note_sync
  -- *
  -- * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
  -- * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
  -- * ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
  -- * ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
  -- * ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
  -- * ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
  -- * ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
  -- * ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
  -- * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D16,
  -- * ::cuMemsetD2D32, ::cuMemsetD8, ::cuMemsetD16, ::cuMemsetD32
  --  

   function cuMemcpyAtoH_v2
     (dstHost : System.Address;
      srcArray : CUarray;
      srcOffset : stddef_h.size_t;
      ByteCount : stddef_h.size_t) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:5135
   pragma Import (C, cuMemcpyAtoH_v2, "cuMemcpyAtoH_v2");

  --*
  -- * \brief Copies memory from Array to Array
  -- *
  -- * Copies from one 1D CUDA array to another. \p dstArray and \p srcArray
  -- * specify the handles of the destination and source CUDA arrays for the copy,
  -- * respectively. \p dstOffset and \p srcOffset specify the destination and
  -- * source offsets in bytes into the CUDA arrays. \p ByteCount is the number of
  -- * bytes to be copied. The size of the elements in the CUDA arrays need not be
  -- * the same format, but the elements must be the same size; and count must be
  -- * evenly divisible by that size.
  -- *
  -- * \param dstArray  - Destination array
  -- * \param dstOffset - Offset in bytes of destination array
  -- * \param srcArray  - Source array
  -- * \param srcOffset - Offset in bytes of source array
  -- * \param ByteCount - Size of memory copy in bytes
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- * \notefnerr
  -- * \note_sync
  -- *
  -- * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
  -- * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
  -- * ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
  -- * ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoD,
  -- * ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
  -- * ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
  -- * ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
  -- * ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
  -- * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D16,
  -- * ::cuMemsetD2D32, ::cuMemsetD8, ::cuMemsetD16, ::cuMemsetD32
  --  

   function cuMemcpyAtoA_v2
     (dstArray : CUarray;
      dstOffset : stddef_h.size_t;
      srcArray : CUarray;
      srcOffset : stddef_h.size_t;
      ByteCount : stddef_h.size_t) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:5174
   pragma Import (C, cuMemcpyAtoA_v2, "cuMemcpyAtoA_v2");

  --*
  -- * \brief Copies memory for 2D arrays
  -- *
  -- * Perform a 2D memory copy according to the parameters specified in \p pCopy.
  -- * The ::CUDA_MEMCPY2D structure is defined as:
  -- *
  -- * \code
  --   typedef struct CUDA_MEMCPY2D_st {
  --      unsigned int srcXInBytes, srcY;
  --      CUmemorytype srcMemoryType;
  --          const void *srcHost;
  --          CUdeviceptr srcDevice;
  --          CUarray srcArray;
  --          unsigned int srcPitch;
  --      unsigned int dstXInBytes, dstY;
  --      CUmemorytype dstMemoryType;
  --          void *dstHost;
  --          CUdeviceptr dstDevice;
  --          CUarray dstArray;
  --          unsigned int dstPitch;
  --      unsigned int WidthInBytes;
  --      unsigned int Height;
  --   } CUDA_MEMCPY2D;
  -- * \endcode
  -- * where:
  -- * - ::srcMemoryType and ::dstMemoryType specify the type of memory of the
  -- *   source and destination, respectively; ::CUmemorytype_enum is defined as:
  -- *
  -- * \code
  --   typedef enum CUmemorytype_enum {
  --      CU_MEMORYTYPE_HOST = 0x01,
  --      CU_MEMORYTYPE_DEVICE = 0x02,
  --      CU_MEMORYTYPE_ARRAY = 0x03,
  --      CU_MEMORYTYPE_UNIFIED = 0x04
  --   } CUmemorytype;
  -- * \endcode
  -- *
  -- * \par
  -- * If ::srcMemoryType is ::CU_MEMORYTYPE_UNIFIED, ::srcDevice and ::srcPitch
  -- *   specify the (unified virtual address space) base address of the source data 
  -- *   and the bytes per row to apply.  ::srcArray is ignored.  
  -- * This value may be used only if unified addressing is supported in the calling 
  -- *   context.
  -- *
  -- * \par
  -- * If ::srcMemoryType is ::CU_MEMORYTYPE_HOST, ::srcHost and ::srcPitch
  -- * specify the (host) base address of the source data and the bytes per row to
  -- * apply. ::srcArray is ignored.
  -- *
  -- * \par
  -- * If ::srcMemoryType is ::CU_MEMORYTYPE_DEVICE, ::srcDevice and ::srcPitch
  -- * specify the (device) base address of the source data and the bytes per row
  -- * to apply. ::srcArray is ignored.
  -- *
  -- * \par
  -- * If ::srcMemoryType is ::CU_MEMORYTYPE_ARRAY, ::srcArray specifies the
  -- * handle of the source data. ::srcHost, ::srcDevice and ::srcPitch are
  -- * ignored.
  -- *
  -- * \par
  -- * If ::dstMemoryType is ::CU_MEMORYTYPE_HOST, ::dstHost and ::dstPitch
  -- * specify the (host) base address of the destination data and the bytes per
  -- * row to apply. ::dstArray is ignored.
  -- *
  -- * \par
  -- * If ::dstMemoryType is ::CU_MEMORYTYPE_UNIFIED, ::dstDevice and ::dstPitch
  -- *   specify the (unified virtual address space) base address of the source data 
  -- *   and the bytes per row to apply.  ::dstArray is ignored.  
  -- * This value may be used only if unified addressing is supported in the calling 
  -- *   context.
  -- *
  -- * \par
  -- * If ::dstMemoryType is ::CU_MEMORYTYPE_DEVICE, ::dstDevice and ::dstPitch
  -- * specify the (device) base address of the destination data and the bytes per
  -- * row to apply. ::dstArray is ignored.
  -- *
  -- * \par
  -- * If ::dstMemoryType is ::CU_MEMORYTYPE_ARRAY, ::dstArray specifies the
  -- * handle of the destination data. ::dstHost, ::dstDevice and ::dstPitch are
  -- * ignored.
  -- *
  -- * - ::srcXInBytes and ::srcY specify the base address of the source data for
  -- *   the copy.
  -- *
  -- * \par
  -- * For host pointers, the starting address is
  -- * \code
  --  void* Start = (void*)((char*)srcHost+srcY*srcPitch + srcXInBytes);
  -- * \endcode
  -- *
  -- * \par
  -- * For device pointers, the starting address is
  -- * \code
  --  CUdeviceptr Start = srcDevice+srcY*srcPitch+srcXInBytes;
  -- * \endcode
  -- *
  -- * \par
  -- * For CUDA arrays, ::srcXInBytes must be evenly divisible by the array
  -- * element size.
  -- *
  -- * - ::dstXInBytes and ::dstY specify the base address of the destination data
  -- *   for the copy.
  -- *
  -- * \par
  -- * For host pointers, the base address is
  -- * \code
  --  void* dstStart = (void*)((char*)dstHost+dstY*dstPitch + dstXInBytes);
  -- * \endcode
  -- *
  -- * \par
  -- * For device pointers, the starting address is
  -- * \code
  --  CUdeviceptr dstStart = dstDevice+dstY*dstPitch+dstXInBytes;
  -- * \endcode
  -- *
  -- * \par
  -- * For CUDA arrays, ::dstXInBytes must be evenly divisible by the array
  -- * element size.
  -- *
  -- * - ::WidthInBytes and ::Height specify the width (in bytes) and height of
  -- *   the 2D copy being performed.
  -- * - If specified, ::srcPitch must be greater than or equal to ::WidthInBytes +
  -- *   ::srcXInBytes, and ::dstPitch must be greater than or equal to
  -- *   ::WidthInBytes + dstXInBytes.
  -- *
  -- * \par
  -- * ::cuMemcpy2D() returns an error if any pitch is greater than the maximum
  -- * allowed (::CU_DEVICE_ATTRIBUTE_MAX_PITCH). ::cuMemAllocPitch() passes back
  -- * pitches that always work with ::cuMemcpy2D(). On intra-device memory copies
  -- * (device to device, CUDA array to device, CUDA array to CUDA array),
  -- * ::cuMemcpy2D() may fail for pitches not computed by ::cuMemAllocPitch().
  -- * ::cuMemcpy2DUnaligned() does not have this restriction, but may run
  -- * significantly slower in the cases where ::cuMemcpy2D() would have returned
  -- * an error code.
  -- *
  -- * \param pCopy - Parameters for the memory copy
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- * \notefnerr
  -- * \note_sync
  -- *
  -- * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
  -- * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
  -- * ::cuMemAllocPitch, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
  -- * ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
  -- * ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
  -- * ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
  -- * ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
  -- * ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
  -- * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D16,
  -- * ::cuMemsetD2D32, ::cuMemsetD8, ::cuMemsetD16, ::cuMemsetD32
  --  

   function cuMemcpy2D_v2 (pCopy : access constant CUDA_MEMCPY2D) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:5335
   pragma Import (C, cuMemcpy2D_v2, "cuMemcpy2D_v2");

  --*
  -- * \brief Copies memory for 2D arrays
  -- *
  -- * Perform a 2D memory copy according to the parameters specified in \p pCopy.
  -- * The ::CUDA_MEMCPY2D structure is defined as:
  -- *
  -- * \code
  --   typedef struct CUDA_MEMCPY2D_st {
  --      unsigned int srcXInBytes, srcY;
  --      CUmemorytype srcMemoryType;
  --      const void *srcHost;
  --      CUdeviceptr srcDevice;
  --      CUarray srcArray;
  --      unsigned int srcPitch;
  --      unsigned int dstXInBytes, dstY;
  --      CUmemorytype dstMemoryType;
  --      void *dstHost;
  --      CUdeviceptr dstDevice;
  --      CUarray dstArray;
  --      unsigned int dstPitch;
  --      unsigned int WidthInBytes;
  --      unsigned int Height;
  --   } CUDA_MEMCPY2D;
  -- * \endcode
  -- * where:
  -- * - ::srcMemoryType and ::dstMemoryType specify the type of memory of the
  -- *   source and destination, respectively; ::CUmemorytype_enum is defined as:
  -- *
  -- * \code
  --   typedef enum CUmemorytype_enum {
  --      CU_MEMORYTYPE_HOST = 0x01,
  --      CU_MEMORYTYPE_DEVICE = 0x02,
  --      CU_MEMORYTYPE_ARRAY = 0x03,
  --      CU_MEMORYTYPE_UNIFIED = 0x04
  --   } CUmemorytype;
  -- * \endcode
  -- *
  -- * \par
  -- * If ::srcMemoryType is ::CU_MEMORYTYPE_UNIFIED, ::srcDevice and ::srcPitch
  -- *   specify the (unified virtual address space) base address of the source data 
  -- *   and the bytes per row to apply.  ::srcArray is ignored.  
  -- * This value may be used only if unified addressing is supported in the calling 
  -- *   context.
  -- *
  -- * \par
  -- * If ::srcMemoryType is ::CU_MEMORYTYPE_HOST, ::srcHost and ::srcPitch
  -- * specify the (host) base address of the source data and the bytes per row to
  -- * apply. ::srcArray is ignored.
  -- *
  -- * \par
  -- * If ::srcMemoryType is ::CU_MEMORYTYPE_DEVICE, ::srcDevice and ::srcPitch
  -- * specify the (device) base address of the source data and the bytes per row
  -- * to apply. ::srcArray is ignored.
  -- *
  -- * \par
  -- * If ::srcMemoryType is ::CU_MEMORYTYPE_ARRAY, ::srcArray specifies the
  -- * handle of the source data. ::srcHost, ::srcDevice and ::srcPitch are
  -- * ignored.
  -- *
  -- * \par
  -- * If ::dstMemoryType is ::CU_MEMORYTYPE_UNIFIED, ::dstDevice and ::dstPitch
  -- *   specify the (unified virtual address space) base address of the source data 
  -- *   and the bytes per row to apply.  ::dstArray is ignored.  
  -- * This value may be used only if unified addressing is supported in the calling 
  -- *   context.
  -- *
  -- * \par
  -- * If ::dstMemoryType is ::CU_MEMORYTYPE_HOST, ::dstHost and ::dstPitch
  -- * specify the (host) base address of the destination data and the bytes per
  -- * row to apply. ::dstArray is ignored.
  -- *
  -- * \par
  -- * If ::dstMemoryType is ::CU_MEMORYTYPE_DEVICE, ::dstDevice and ::dstPitch
  -- * specify the (device) base address of the destination data and the bytes per
  -- * row to apply. ::dstArray is ignored.
  -- *
  -- * \par
  -- * If ::dstMemoryType is ::CU_MEMORYTYPE_ARRAY, ::dstArray specifies the
  -- * handle of the destination data. ::dstHost, ::dstDevice and ::dstPitch are
  -- * ignored.
  -- *
  -- * - ::srcXInBytes and ::srcY specify the base address of the source data for
  -- *   the copy.
  -- *
  -- * \par
  -- * For host pointers, the starting address is
  -- * \code
  --  void* Start = (void*)((char*)srcHost+srcY*srcPitch + srcXInBytes);
  -- * \endcode
  -- *
  -- * \par
  -- * For device pointers, the starting address is
  -- * \code
  --  CUdeviceptr Start = srcDevice+srcY*srcPitch+srcXInBytes;
  -- * \endcode
  -- *
  -- * \par
  -- * For CUDA arrays, ::srcXInBytes must be evenly divisible by the array
  -- * element size.
  -- *
  -- * - ::dstXInBytes and ::dstY specify the base address of the destination data
  -- *   for the copy.
  -- *
  -- * \par
  -- * For host pointers, the base address is
  -- * \code
  --  void* dstStart = (void*)((char*)dstHost+dstY*dstPitch + dstXInBytes);
  -- * \endcode
  -- *
  -- * \par
  -- * For device pointers, the starting address is
  -- * \code
  --  CUdeviceptr dstStart = dstDevice+dstY*dstPitch+dstXInBytes;
  -- * \endcode
  -- *
  -- * \par
  -- * For CUDA arrays, ::dstXInBytes must be evenly divisible by the array
  -- * element size.
  -- *
  -- * - ::WidthInBytes and ::Height specify the width (in bytes) and height of
  -- *   the 2D copy being performed.
  -- * - If specified, ::srcPitch must be greater than or equal to ::WidthInBytes +
  -- *   ::srcXInBytes, and ::dstPitch must be greater than or equal to
  -- *   ::WidthInBytes + dstXInBytes.
  -- *
  -- * \par
  -- * ::cuMemcpy2D() returns an error if any pitch is greater than the maximum
  -- * allowed (::CU_DEVICE_ATTRIBUTE_MAX_PITCH). ::cuMemAllocPitch() passes back
  -- * pitches that always work with ::cuMemcpy2D(). On intra-device memory copies
  -- * (device to device, CUDA array to device, CUDA array to CUDA array),
  -- * ::cuMemcpy2D() may fail for pitches not computed by ::cuMemAllocPitch().
  -- * ::cuMemcpy2DUnaligned() does not have this restriction, but may run
  -- * significantly slower in the cases where ::cuMemcpy2D() would have returned
  -- * an error code.
  -- *
  -- * \param pCopy - Parameters for the memory copy
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- * \notefnerr
  -- * \note_sync
  -- *
  -- * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
  -- * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
  -- * ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync,
  -- * ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
  -- * ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
  -- * ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
  -- * ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
  -- * ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
  -- * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D16,
  -- * ::cuMemsetD2D32, ::cuMemsetD8, ::cuMemsetD16, ::cuMemsetD32
  --  

   function cuMemcpy2DUnaligned_v2 (pCopy : access constant CUDA_MEMCPY2D) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:5494
   pragma Import (C, cuMemcpy2DUnaligned_v2, "cuMemcpy2DUnaligned_v2");

  --*
  -- * \brief Copies memory for 3D arrays
  -- *
  -- * Perform a 3D memory copy according to the parameters specified in
  -- * \p pCopy. The ::CUDA_MEMCPY3D structure is defined as:
  -- *
  -- * \code
  --        typedef struct CUDA_MEMCPY3D_st {
  --            unsigned int srcXInBytes, srcY, srcZ;
  --            unsigned int srcLOD;
  --            CUmemorytype srcMemoryType;
  --                const void *srcHost;
  --                CUdeviceptr srcDevice;
  --                CUarray srcArray;
  --                unsigned int srcPitch;  // ignored when src is array
  --                unsigned int srcHeight; // ignored when src is array; may be 0 if Depth==1
  --            unsigned int dstXInBytes, dstY, dstZ;
  --            unsigned int dstLOD;
  --            CUmemorytype dstMemoryType;
  --                void *dstHost;
  --                CUdeviceptr dstDevice;
  --                CUarray dstArray;
  --                unsigned int dstPitch;  // ignored when dst is array
  --                unsigned int dstHeight; // ignored when dst is array; may be 0 if Depth==1
  --            unsigned int WidthInBytes;
  --            unsigned int Height;
  --            unsigned int Depth;
  --        } CUDA_MEMCPY3D;
  -- * \endcode
  -- * where:
  -- * - ::srcMemoryType and ::dstMemoryType specify the type of memory of the
  -- *   source and destination, respectively; ::CUmemorytype_enum is defined as:
  -- *
  -- * \code
  --   typedef enum CUmemorytype_enum {
  --      CU_MEMORYTYPE_HOST = 0x01,
  --      CU_MEMORYTYPE_DEVICE = 0x02,
  --      CU_MEMORYTYPE_ARRAY = 0x03,
  --      CU_MEMORYTYPE_UNIFIED = 0x04
  --   } CUmemorytype;
  -- * \endcode
  -- *
  -- * \par
  -- * If ::srcMemoryType is ::CU_MEMORYTYPE_UNIFIED, ::srcDevice and ::srcPitch
  -- *   specify the (unified virtual address space) base address of the source data 
  -- *   and the bytes per row to apply.  ::srcArray is ignored.  
  -- * This value may be used only if unified addressing is supported in the calling 
  -- *   context.
  -- *
  -- * \par
  -- * If ::srcMemoryType is ::CU_MEMORYTYPE_HOST, ::srcHost, ::srcPitch and
  -- * ::srcHeight specify the (host) base address of the source data, the bytes
  -- * per row, and the height of each 2D slice of the 3D array. ::srcArray is
  -- * ignored.
  -- *
  -- * \par
  -- * If ::srcMemoryType is ::CU_MEMORYTYPE_DEVICE, ::srcDevice, ::srcPitch and
  -- * ::srcHeight specify the (device) base address of the source data, the bytes
  -- * per row, and the height of each 2D slice of the 3D array. ::srcArray is
  -- * ignored.
  -- *
  -- * \par
  -- * If ::srcMemoryType is ::CU_MEMORYTYPE_ARRAY, ::srcArray specifies the
  -- * handle of the source data. ::srcHost, ::srcDevice, ::srcPitch and
  -- * ::srcHeight are ignored.
  -- *
  -- * \par
  -- * If ::dstMemoryType is ::CU_MEMORYTYPE_UNIFIED, ::dstDevice and ::dstPitch
  -- *   specify the (unified virtual address space) base address of the source data 
  -- *   and the bytes per row to apply.  ::dstArray is ignored.  
  -- * This value may be used only if unified addressing is supported in the calling 
  -- *   context.
  -- *
  -- * \par
  -- * If ::dstMemoryType is ::CU_MEMORYTYPE_HOST, ::dstHost and ::dstPitch
  -- * specify the (host) base address of the destination data, the bytes per row,
  -- * and the height of each 2D slice of the 3D array. ::dstArray is ignored.
  -- *
  -- * \par
  -- * If ::dstMemoryType is ::CU_MEMORYTYPE_DEVICE, ::dstDevice and ::dstPitch
  -- * specify the (device) base address of the destination data, the bytes per
  -- * row, and the height of each 2D slice of the 3D array. ::dstArray is ignored.
  -- *
  -- * \par
  -- * If ::dstMemoryType is ::CU_MEMORYTYPE_ARRAY, ::dstArray specifies the
  -- * handle of the destination data. ::dstHost, ::dstDevice, ::dstPitch and
  -- * ::dstHeight are ignored.
  -- *
  -- * - ::srcXInBytes, ::srcY and ::srcZ specify the base address of the source
  -- *   data for the copy.
  -- *
  -- * \par
  -- * For host pointers, the starting address is
  -- * \code
  --  void* Start = (void*)((char*)srcHost+(srcZ*srcHeight+srcY)*srcPitch + srcXInBytes);
  -- * \endcode
  -- *
  -- * \par
  -- * For device pointers, the starting address is
  -- * \code
  --  CUdeviceptr Start = srcDevice+(srcZ*srcHeight+srcY)*srcPitch+srcXInBytes;
  -- * \endcode
  -- *
  -- * \par
  -- * For CUDA arrays, ::srcXInBytes must be evenly divisible by the array
  -- * element size.
  -- *
  -- * - dstXInBytes, ::dstY and ::dstZ specify the base address of the
  -- *   destination data for the copy.
  -- *
  -- * \par
  -- * For host pointers, the base address is
  -- * \code
  --  void* dstStart = (void*)((char*)dstHost+(dstZ*dstHeight+dstY)*dstPitch + dstXInBytes);
  -- * \endcode
  -- *
  -- * \par
  -- * For device pointers, the starting address is
  -- * \code
  --  CUdeviceptr dstStart = dstDevice+(dstZ*dstHeight+dstY)*dstPitch+dstXInBytes;
  -- * \endcode
  -- *
  -- * \par
  -- * For CUDA arrays, ::dstXInBytes must be evenly divisible by the array
  -- * element size.
  -- *
  -- * - ::WidthInBytes, ::Height and ::Depth specify the width (in bytes), height
  -- *   and depth of the 3D copy being performed.
  -- * - If specified, ::srcPitch must be greater than or equal to ::WidthInBytes +
  -- *   ::srcXInBytes, and ::dstPitch must be greater than or equal to
  -- *   ::WidthInBytes + dstXInBytes.
  -- * - If specified, ::srcHeight must be greater than or equal to ::Height +
  -- *   ::srcY, and ::dstHeight must be greater than or equal to ::Height + ::dstY.
  -- *
  -- * \par
  -- * ::cuMemcpy3D() returns an error if any pitch is greater than the maximum
  -- * allowed (::CU_DEVICE_ATTRIBUTE_MAX_PITCH).
  -- *
  -- * The ::srcLOD and ::dstLOD members of the ::CUDA_MEMCPY3D structure must be
  -- * set to 0.
  -- *
  -- * \param pCopy - Parameters for the memory copy
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- * \notefnerr
  -- * \note_sync
  -- *
  -- * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
  -- * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
  -- * ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
  -- * ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
  -- * ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
  -- * ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
  -- * ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
  -- * ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
  -- * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D16,
  -- * ::cuMemsetD2D32, ::cuMemsetD8, ::cuMemsetD16, ::cuMemsetD32
  --  

   function cuMemcpy3D_v2 (pCopy : access constant CUDA_MEMCPY3D) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:5662
   pragma Import (C, cuMemcpy3D_v2, "cuMemcpy3D_v2");

  --*
  -- * \brief Copies memory between contexts
  -- *
  -- * Perform a 3D memory copy according to the parameters specified in
  -- * \p pCopy.  See the definition of the ::CUDA_MEMCPY3D_PEER structure
  -- * for documentation of its parameters.
  -- *
  -- * \param pCopy - Parameters for the memory copy
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- * \notefnerr
  -- * \note_sync
  -- *
  -- * \sa ::cuMemcpyDtoD, ::cuMemcpyPeer, ::cuMemcpyDtoDAsync, ::cuMemcpyPeerAsync,
  -- * ::cuMemcpy3DPeerAsync
  --  

   function cuMemcpy3DPeer (pCopy : access constant CUDA_MEMCPY3D_PEER) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:5687
   pragma Import (C, cuMemcpy3DPeer, "cuMemcpy3DPeer");

  --*
  -- * \brief Copies memory asynchronously
  -- *
  -- * Copies data between two pointers. 
  -- * \p dst and \p src are base pointers of the destination and source, respectively.  
  -- * \p ByteCount specifies the number of bytes to copy.
  -- * Note that this function infers the type of the transfer (host to host, host to 
  -- *   device, device to device, or device to host) from the pointer values.  This
  -- *   function is only allowed in contexts which support unified addressing.
  -- *
  -- * \param dst       - Destination unified virtual address space pointer
  -- * \param src       - Source unified virtual address space pointer
  -- * \param ByteCount - Size of memory copy in bytes
  -- * \param hStream   - Stream identifier
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- * \notefnerr
  -- * \note_async
  -- * \note_null_stream
  -- *
  -- * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
  -- * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
  -- * ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
  -- * ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
  -- * ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD,
  -- * ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
  -- * ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
  -- * ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
  -- * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D8Async,
  -- * ::cuMemsetD2D16, ::cuMemsetD2D16Async, ::cuMemsetD2D32, ::cuMemsetD2D32Async,
  -- * ::cuMemsetD8, ::cuMemsetD8Async, ::cuMemsetD16, ::cuMemsetD16Async,
  -- * ::cuMemsetD32, ::cuMemsetD32Async
  --  

   function cuMemcpyAsync
     (dst : CUdeviceptr;
      src : CUdeviceptr;
      ByteCount : stddef_h.size_t;
      hStream : CUstream) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:5727
   pragma Import (C, cuMemcpyAsync, "cuMemcpyAsync");

  --*
  -- * \brief Copies device memory between two contexts asynchronously.
  -- *
  -- * Copies from device memory in one context to device memory in another
  -- * context. \p dstDevice is the base device pointer of the destination memory 
  -- * and \p dstContext is the destination context.  \p srcDevice is the base 
  -- * device pointer of the source memory and \p srcContext is the source pointer.  
  -- * \p ByteCount specifies the number of bytes to copy.
  -- *
  -- * \param dstDevice  - Destination device pointer
  -- * \param dstContext - Destination context
  -- * \param srcDevice  - Source device pointer
  -- * \param srcContext - Source context
  -- * \param ByteCount  - Size of memory copy in bytes
  -- * \param hStream    - Stream identifier
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- * \notefnerr
  -- * \note_async
  -- * \note_null_stream
  -- *
  -- * \sa ::cuMemcpyDtoD, ::cuMemcpyPeer, ::cuMemcpy3DPeer, ::cuMemcpyDtoDAsync, 
  -- * ::cuMemcpy3DPeerAsync
  --  

   function cuMemcpyPeerAsync
     (dstDevice : CUdeviceptr;
      dstContext : CUcontext;
      srcDevice : CUdeviceptr;
      srcContext : CUcontext;
      ByteCount : stddef_h.size_t;
      hStream : CUstream) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:5758
   pragma Import (C, cuMemcpyPeerAsync, "cuMemcpyPeerAsync");

  --*
  -- * \brief Copies memory from Host to Device
  -- *
  -- * Copies from host memory to device memory. \p dstDevice and \p srcHost are
  -- * the base addresses of the destination and source, respectively. \p ByteCount
  -- * specifies the number of bytes to copy.
  -- *
  -- * \param dstDevice - Destination device pointer
  -- * \param srcHost   - Source host pointer
  -- * \param ByteCount - Size of memory copy in bytes
  -- * \param hStream   - Stream identifier
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- * \notefnerr
  -- * \note_async
  -- * \note_null_stream
  -- *
  -- * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
  -- * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
  -- * ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
  -- * ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
  -- * ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
  -- * ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
  -- * ::cuMemcpyHtoD, ::cuMemFree, ::cuMemFreeHost,
  -- * ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
  -- * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D8Async,
  -- * ::cuMemsetD2D16, ::cuMemsetD2D16Async, ::cuMemsetD2D32, ::cuMemsetD2D32Async,
  -- * ::cuMemsetD8, ::cuMemsetD8Async, ::cuMemsetD16, ::cuMemsetD16Async,
  -- * ::cuMemsetD32, ::cuMemsetD32Async
  --  

   function cuMemcpyHtoDAsync_v2
     (dstDevice : CUdeviceptr;
      srcHost : System.Address;
      ByteCount : stddef_h.size_t;
      hStream : CUstream) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:5797
   pragma Import (C, cuMemcpyHtoDAsync_v2, "cuMemcpyHtoDAsync_v2");

  --*
  -- * \brief Copies memory from Device to Host
  -- *
  -- * Copies from device to host memory. \p dstHost and \p srcDevice specify the
  -- * base pointers of the destination and source, respectively. \p ByteCount
  -- * specifies the number of bytes to copy.
  -- *
  -- * \param dstHost   - Destination host pointer
  -- * \param srcDevice - Source device pointer
  -- * \param ByteCount - Size of memory copy in bytes
  -- * \param hStream   - Stream identifier
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- * \notefnerr
  -- * \note_async
  -- * \note_null_stream
  -- *
  -- * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
  -- * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
  -- * ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
  -- * ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
  -- * ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
  -- * ::cuMemcpyDtoH, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
  -- * ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
  -- * ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
  -- * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D8Async,
  -- * ::cuMemsetD2D16, ::cuMemsetD2D16Async, ::cuMemsetD2D32, ::cuMemsetD2D32Async,
  -- * ::cuMemsetD8, ::cuMemsetD8Async, ::cuMemsetD16, ::cuMemsetD16Async,
  -- * ::cuMemsetD32, ::cuMemsetD32Async
  --  

   function cuMemcpyDtoHAsync_v2
     (dstHost : System.Address;
      srcDevice : CUdeviceptr;
      ByteCount : stddef_h.size_t;
      hStream : CUstream) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:5834
   pragma Import (C, cuMemcpyDtoHAsync_v2, "cuMemcpyDtoHAsync_v2");

  --*
  -- * \brief Copies memory from Device to Device
  -- *
  -- * Copies from device memory to device memory. \p dstDevice and \p srcDevice
  -- * are the base pointers of the destination and source, respectively.
  -- * \p ByteCount specifies the number of bytes to copy.
  -- *
  -- * \param dstDevice - Destination device pointer
  -- * \param srcDevice - Source device pointer
  -- * \param ByteCount - Size of memory copy in bytes
  -- * \param hStream   - Stream identifier
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- * \notefnerr
  -- * \note_async
  -- * \note_null_stream
  -- *
  -- * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
  -- * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
  -- * ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
  -- * ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
  -- * ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD,
  -- * ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
  -- * ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
  -- * ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
  -- * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D8Async,
  -- * ::cuMemsetD2D16, ::cuMemsetD2D16Async, ::cuMemsetD2D32, ::cuMemsetD2D32Async,
  -- * ::cuMemsetD8, ::cuMemsetD8Async, ::cuMemsetD16, ::cuMemsetD16Async,
  -- * ::cuMemsetD32, ::cuMemsetD32Async
  --  

   function cuMemcpyDtoDAsync_v2
     (dstDevice : CUdeviceptr;
      srcDevice : CUdeviceptr;
      ByteCount : stddef_h.size_t;
      hStream : CUstream) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:5871
   pragma Import (C, cuMemcpyDtoDAsync_v2, "cuMemcpyDtoDAsync_v2");

  --*
  -- * \brief Copies memory from Host to Array
  -- *
  -- * Copies from host memory to a 1D CUDA array. \p dstArray and \p dstOffset
  -- * specify the CUDA array handle and starting offset in bytes of the
  -- * destination data. \p srcHost specifies the base address of the source.
  -- * \p ByteCount specifies the number of bytes to copy.
  -- *
  -- * \param dstArray  - Destination array
  -- * \param dstOffset - Offset in bytes of destination array
  -- * \param srcHost   - Source host pointer
  -- * \param ByteCount - Size of memory copy in bytes
  -- * \param hStream   - Stream identifier
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- * \notefnerr
  -- * \note_async
  -- * \note_null_stream
  -- *
  -- * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
  -- * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
  -- * ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
  -- * ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
  -- * ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
  -- * ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA,
  -- * ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
  -- * ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
  -- * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D8Async,
  -- * ::cuMemsetD2D16, ::cuMemsetD2D16Async, ::cuMemsetD2D32, ::cuMemsetD2D32Async,
  -- * ::cuMemsetD8, ::cuMemsetD8Async, ::cuMemsetD16, ::cuMemsetD16Async,
  -- * ::cuMemsetD32, ::cuMemsetD32Async
  --  

   function cuMemcpyHtoAAsync_v2
     (dstArray : CUarray;
      dstOffset : stddef_h.size_t;
      srcHost : System.Address;
      ByteCount : stddef_h.size_t;
      hStream : CUstream) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:5910
   pragma Import (C, cuMemcpyHtoAAsync_v2, "cuMemcpyHtoAAsync_v2");

  --*
  -- * \brief Copies memory from Array to Host
  -- *
  -- * Copies from one 1D CUDA array to host memory. \p dstHost specifies the base
  -- * pointer of the destination. \p srcArray and \p srcOffset specify the CUDA
  -- * array handle and starting offset in bytes of the source data.
  -- * \p ByteCount specifies the number of bytes to copy.
  -- *
  -- * \param dstHost   - Destination pointer
  -- * \param srcArray  - Source array
  -- * \param srcOffset - Offset in bytes of source array
  -- * \param ByteCount - Size of memory copy in bytes
  -- * \param hStream   - Stream identifier
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- * \notefnerr
  -- * \note_async
  -- * \note_null_stream
  -- *
  -- * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
  -- * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
  -- * ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
  -- * ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
  -- * ::cuMemcpyAtoH, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
  -- * ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
  -- * ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
  -- * ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
  -- * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D8Async,
  -- * ::cuMemsetD2D16, ::cuMemsetD2D16Async, ::cuMemsetD2D32, ::cuMemsetD2D32Async,
  -- * ::cuMemsetD8, ::cuMemsetD8Async, ::cuMemsetD16, ::cuMemsetD16Async,
  -- * ::cuMemsetD32, ::cuMemsetD32Async
  --  

   function cuMemcpyAtoHAsync_v2
     (dstHost : System.Address;
      srcArray : CUarray;
      srcOffset : stddef_h.size_t;
      ByteCount : stddef_h.size_t;
      hStream : CUstream) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:5949
   pragma Import (C, cuMemcpyAtoHAsync_v2, "cuMemcpyAtoHAsync_v2");

  --*
  -- * \brief Copies memory for 2D arrays
  -- *
  -- * Perform a 2D memory copy according to the parameters specified in \p pCopy.
  -- * The ::CUDA_MEMCPY2D structure is defined as:
  -- *
  -- * \code
  --   typedef struct CUDA_MEMCPY2D_st {
  --      unsigned int srcXInBytes, srcY;
  --      CUmemorytype srcMemoryType;
  --      const void *srcHost;
  --      CUdeviceptr srcDevice;
  --      CUarray srcArray;
  --      unsigned int srcPitch;
  --      unsigned int dstXInBytes, dstY;
  --      CUmemorytype dstMemoryType;
  --      void *dstHost;
  --      CUdeviceptr dstDevice;
  --      CUarray dstArray;
  --      unsigned int dstPitch;
  --      unsigned int WidthInBytes;
  --      unsigned int Height;
  --   } CUDA_MEMCPY2D;
  -- * \endcode
  -- * where:
  -- * - ::srcMemoryType and ::dstMemoryType specify the type of memory of the
  -- *   source and destination, respectively; ::CUmemorytype_enum is defined as:
  -- *
  -- * \code
  --   typedef enum CUmemorytype_enum {
  --      CU_MEMORYTYPE_HOST = 0x01,
  --      CU_MEMORYTYPE_DEVICE = 0x02,
  --      CU_MEMORYTYPE_ARRAY = 0x03,
  --      CU_MEMORYTYPE_UNIFIED = 0x04
  --   } CUmemorytype;
  -- * \endcode
  -- *
  -- * \par
  -- * If ::srcMemoryType is ::CU_MEMORYTYPE_HOST, ::srcHost and ::srcPitch
  -- * specify the (host) base address of the source data and the bytes per row to
  -- * apply. ::srcArray is ignored.
  -- *
  -- * \par
  -- * If ::srcMemoryType is ::CU_MEMORYTYPE_UNIFIED, ::srcDevice and ::srcPitch
  -- *   specify the (unified virtual address space) base address of the source data 
  -- *   and the bytes per row to apply.  ::srcArray is ignored.  
  -- * This value may be used only if unified addressing is supported in the calling 
  -- *   context.
  -- *
  -- * \par
  -- * If ::srcMemoryType is ::CU_MEMORYTYPE_DEVICE, ::srcDevice and ::srcPitch
  -- * specify the (device) base address of the source data and the bytes per row
  -- * to apply. ::srcArray is ignored.
  -- *
  -- * \par
  -- * If ::srcMemoryType is ::CU_MEMORYTYPE_ARRAY, ::srcArray specifies the
  -- * handle of the source data. ::srcHost, ::srcDevice and ::srcPitch are
  -- * ignored.
  -- *
  -- * \par
  -- * If ::dstMemoryType is ::CU_MEMORYTYPE_UNIFIED, ::dstDevice and ::dstPitch
  -- *   specify the (unified virtual address space) base address of the source data 
  -- *   and the bytes per row to apply.  ::dstArray is ignored.  
  -- * This value may be used only if unified addressing is supported in the calling 
  -- *   context.
  -- *
  -- * \par
  -- * If ::dstMemoryType is ::CU_MEMORYTYPE_HOST, ::dstHost and ::dstPitch
  -- * specify the (host) base address of the destination data and the bytes per
  -- * row to apply. ::dstArray is ignored.
  -- *
  -- * \par
  -- * If ::dstMemoryType is ::CU_MEMORYTYPE_DEVICE, ::dstDevice and ::dstPitch
  -- * specify the (device) base address of the destination data and the bytes per
  -- * row to apply. ::dstArray is ignored.
  -- *
  -- * \par
  -- * If ::dstMemoryType is ::CU_MEMORYTYPE_ARRAY, ::dstArray specifies the
  -- * handle of the destination data. ::dstHost, ::dstDevice and ::dstPitch are
  -- * ignored.
  -- *
  -- * - ::srcXInBytes and ::srcY specify the base address of the source data for
  -- *   the copy.
  -- *
  -- * \par
  -- * For host pointers, the starting address is
  -- * \code
  --  void* Start = (void*)((char*)srcHost+srcY*srcPitch + srcXInBytes);
  -- * \endcode
  -- *
  -- * \par
  -- * For device pointers, the starting address is
  -- * \code
  --  CUdeviceptr Start = srcDevice+srcY*srcPitch+srcXInBytes;
  -- * \endcode
  -- *
  -- * \par
  -- * For CUDA arrays, ::srcXInBytes must be evenly divisible by the array
  -- * element size.
  -- *
  -- * - ::dstXInBytes and ::dstY specify the base address of the destination data
  -- *   for the copy.
  -- *
  -- * \par
  -- * For host pointers, the base address is
  -- * \code
  --  void* dstStart = (void*)((char*)dstHost+dstY*dstPitch + dstXInBytes);
  -- * \endcode
  -- *
  -- * \par
  -- * For device pointers, the starting address is
  -- * \code
  --  CUdeviceptr dstStart = dstDevice+dstY*dstPitch+dstXInBytes;
  -- * \endcode
  -- *
  -- * \par
  -- * For CUDA arrays, ::dstXInBytes must be evenly divisible by the array
  -- * element size.
  -- *
  -- * - ::WidthInBytes and ::Height specify the width (in bytes) and height of
  -- *   the 2D copy being performed.
  -- * - If specified, ::srcPitch must be greater than or equal to ::WidthInBytes +
  -- *   ::srcXInBytes, and ::dstPitch must be greater than or equal to
  -- *   ::WidthInBytes + dstXInBytes.
  -- * - If specified, ::srcPitch must be greater than or equal to ::WidthInBytes +
  -- *   ::srcXInBytes, and ::dstPitch must be greater than or equal to
  -- *   ::WidthInBytes + dstXInBytes.
  -- * - If specified, ::srcHeight must be greater than or equal to ::Height +
  -- *   ::srcY, and ::dstHeight must be greater than or equal to ::Height + ::dstY.
  -- *
  -- * \par
  -- * ::cuMemcpy2DAsync() returns an error if any pitch is greater than the maximum
  -- * allowed (::CU_DEVICE_ATTRIBUTE_MAX_PITCH). ::cuMemAllocPitch() passes back
  -- * pitches that always work with ::cuMemcpy2D(). On intra-device memory copies
  -- * (device to device, CUDA array to device, CUDA array to CUDA array),
  -- * ::cuMemcpy2DAsync() may fail for pitches not computed by ::cuMemAllocPitch().
  -- *
  -- * \param pCopy   - Parameters for the memory copy
  -- * \param hStream - Stream identifier
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- * \notefnerr
  -- * \note_async
  -- * \note_null_stream
  -- *
  -- * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
  -- * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
  -- * ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DUnaligned,
  -- * ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
  -- * ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
  -- * ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
  -- * ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
  -- * ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
  -- * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D8Async,
  -- * ::cuMemsetD2D16, ::cuMemsetD2D16Async, ::cuMemsetD2D32, ::cuMemsetD2D32Async,
  -- * ::cuMemsetD8, ::cuMemsetD8Async, ::cuMemsetD16, ::cuMemsetD16Async,
  -- * ::cuMemsetD32, ::cuMemsetD32Async
  --  

   function cuMemcpy2DAsync_v2 (pCopy : access constant CUDA_MEMCPY2D; hStream : CUstream) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:6114
   pragma Import (C, cuMemcpy2DAsync_v2, "cuMemcpy2DAsync_v2");

  --*
  -- * \brief Copies memory for 3D arrays
  -- *
  -- * Perform a 3D memory copy according to the parameters specified in
  -- * \p pCopy. The ::CUDA_MEMCPY3D structure is defined as:
  -- *
  -- * \code
  --        typedef struct CUDA_MEMCPY3D_st {
  --            unsigned int srcXInBytes, srcY, srcZ;
  --            unsigned int srcLOD;
  --            CUmemorytype srcMemoryType;
  --                const void *srcHost;
  --                CUdeviceptr srcDevice;
  --                CUarray srcArray;
  --                unsigned int srcPitch;  // ignored when src is array
  --                unsigned int srcHeight; // ignored when src is array; may be 0 if Depth==1
  --            unsigned int dstXInBytes, dstY, dstZ;
  --            unsigned int dstLOD;
  --            CUmemorytype dstMemoryType;
  --                void *dstHost;
  --                CUdeviceptr dstDevice;
  --                CUarray dstArray;
  --                unsigned int dstPitch;  // ignored when dst is array
  --                unsigned int dstHeight; // ignored when dst is array; may be 0 if Depth==1
  --            unsigned int WidthInBytes;
  --            unsigned int Height;
  --            unsigned int Depth;
  --        } CUDA_MEMCPY3D;
  -- * \endcode
  -- * where:
  -- * - ::srcMemoryType and ::dstMemoryType specify the type of memory of the
  -- *   source and destination, respectively; ::CUmemorytype_enum is defined as:
  -- *
  -- * \code
  --   typedef enum CUmemorytype_enum {
  --      CU_MEMORYTYPE_HOST = 0x01,
  --      CU_MEMORYTYPE_DEVICE = 0x02,
  --      CU_MEMORYTYPE_ARRAY = 0x03,
  --      CU_MEMORYTYPE_UNIFIED = 0x04
  --   } CUmemorytype;
  -- * \endcode
  -- *
  -- * \par
  -- * If ::srcMemoryType is ::CU_MEMORYTYPE_UNIFIED, ::srcDevice and ::srcPitch
  -- *   specify the (unified virtual address space) base address of the source data 
  -- *   and the bytes per row to apply.  ::srcArray is ignored.  
  -- * This value may be used only if unified addressing is supported in the calling 
  -- *   context.
  -- *
  -- * \par
  -- * If ::srcMemoryType is ::CU_MEMORYTYPE_HOST, ::srcHost, ::srcPitch and
  -- * ::srcHeight specify the (host) base address of the source data, the bytes
  -- * per row, and the height of each 2D slice of the 3D array. ::srcArray is
  -- * ignored.
  -- *
  -- * \par
  -- * If ::srcMemoryType is ::CU_MEMORYTYPE_DEVICE, ::srcDevice, ::srcPitch and
  -- * ::srcHeight specify the (device) base address of the source data, the bytes
  -- * per row, and the height of each 2D slice of the 3D array. ::srcArray is
  -- * ignored.
  -- *
  -- * \par
  -- * If ::srcMemoryType is ::CU_MEMORYTYPE_ARRAY, ::srcArray specifies the
  -- * handle of the source data. ::srcHost, ::srcDevice, ::srcPitch and
  -- * ::srcHeight are ignored.
  -- *
  -- * \par
  -- * If ::dstMemoryType is ::CU_MEMORYTYPE_UNIFIED, ::dstDevice and ::dstPitch
  -- *   specify the (unified virtual address space) base address of the source data 
  -- *   and the bytes per row to apply.  ::dstArray is ignored.  
  -- * This value may be used only if unified addressing is supported in the calling 
  -- *   context.
  -- *
  -- * \par
  -- * If ::dstMemoryType is ::CU_MEMORYTYPE_HOST, ::dstHost and ::dstPitch
  -- * specify the (host) base address of the destination data, the bytes per row,
  -- * and the height of each 2D slice of the 3D array. ::dstArray is ignored.
  -- *
  -- * \par
  -- * If ::dstMemoryType is ::CU_MEMORYTYPE_DEVICE, ::dstDevice and ::dstPitch
  -- * specify the (device) base address of the destination data, the bytes per
  -- * row, and the height of each 2D slice of the 3D array. ::dstArray is ignored.
  -- *
  -- * \par
  -- * If ::dstMemoryType is ::CU_MEMORYTYPE_ARRAY, ::dstArray specifies the
  -- * handle of the destination data. ::dstHost, ::dstDevice, ::dstPitch and
  -- * ::dstHeight are ignored.
  -- *
  -- * - ::srcXInBytes, ::srcY and ::srcZ specify the base address of the source
  -- *   data for the copy.
  -- *
  -- * \par
  -- * For host pointers, the starting address is
  -- * \code
  --  void* Start = (void*)((char*)srcHost+(srcZ*srcHeight+srcY)*srcPitch + srcXInBytes);
  -- * \endcode
  -- *
  -- * \par
  -- * For device pointers, the starting address is
  -- * \code
  --  CUdeviceptr Start = srcDevice+(srcZ*srcHeight+srcY)*srcPitch+srcXInBytes;
  -- * \endcode
  -- *
  -- * \par
  -- * For CUDA arrays, ::srcXInBytes must be evenly divisible by the array
  -- * element size.
  -- *
  -- * - dstXInBytes, ::dstY and ::dstZ specify the base address of the
  -- *   destination data for the copy.
  -- *
  -- * \par
  -- * For host pointers, the base address is
  -- * \code
  --  void* dstStart = (void*)((char*)dstHost+(dstZ*dstHeight+dstY)*dstPitch + dstXInBytes);
  -- * \endcode
  -- *
  -- * \par
  -- * For device pointers, the starting address is
  -- * \code
  --  CUdeviceptr dstStart = dstDevice+(dstZ*dstHeight+dstY)*dstPitch+dstXInBytes;
  -- * \endcode
  -- *
  -- * \par
  -- * For CUDA arrays, ::dstXInBytes must be evenly divisible by the array
  -- * element size.
  -- *
  -- * - ::WidthInBytes, ::Height and ::Depth specify the width (in bytes), height
  -- *   and depth of the 3D copy being performed.
  -- * - If specified, ::srcPitch must be greater than or equal to ::WidthInBytes +
  -- *   ::srcXInBytes, and ::dstPitch must be greater than or equal to
  -- *   ::WidthInBytes + dstXInBytes.
  -- * - If specified, ::srcHeight must be greater than or equal to ::Height +
  -- *   ::srcY, and ::dstHeight must be greater than or equal to ::Height + ::dstY.
  -- *
  -- * \par
  -- * ::cuMemcpy3DAsync() returns an error if any pitch is greater than the maximum
  -- * allowed (::CU_DEVICE_ATTRIBUTE_MAX_PITCH).
  -- *
  -- * The ::srcLOD and ::dstLOD members of the ::CUDA_MEMCPY3D structure must be
  -- * set to 0.
  -- *
  -- * \param pCopy - Parameters for the memory copy
  -- * \param hStream - Stream identifier
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- * \notefnerr
  -- * \note_async
  -- * \note_null_stream
  -- *
  -- * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
  -- * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
  -- * ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
  -- * ::cuMemcpy3D, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
  -- * ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
  -- * ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
  -- * ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
  -- * ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
  -- * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D8Async,
  -- * ::cuMemsetD2D16, ::cuMemsetD2D16Async, ::cuMemsetD2D32, ::cuMemsetD2D32Async,
  -- * ::cuMemsetD8, ::cuMemsetD8Async, ::cuMemsetD16, ::cuMemsetD16Async,
  -- * ::cuMemsetD32, ::cuMemsetD32Async
  --  

   function cuMemcpy3DAsync_v2 (pCopy : access constant CUDA_MEMCPY3D; hStream : CUstream) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:6286
   pragma Import (C, cuMemcpy3DAsync_v2, "cuMemcpy3DAsync_v2");

  --*
  -- * \brief Copies memory between contexts asynchronously.
  -- *
  -- * Perform a 3D memory copy according to the parameters specified in
  -- * \p pCopy.  See the definition of the ::CUDA_MEMCPY3D_PEER structure
  -- * for documentation of its parameters.
  -- *
  -- * \param pCopy - Parameters for the memory copy
  -- * \param hStream - Stream identifier
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- * \notefnerr
  -- * \note_async
  -- * \note_null_stream
  -- *
  -- * \sa ::cuMemcpyDtoD, ::cuMemcpyPeer, ::cuMemcpyDtoDAsync, ::cuMemcpyPeerAsync,
  -- * ::cuMemcpy3DPeerAsync
  --  

   function cuMemcpy3DPeerAsync (pCopy : access constant CUDA_MEMCPY3D_PEER; hStream : CUstream) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:6313
   pragma Import (C, cuMemcpy3DPeerAsync, "cuMemcpy3DPeerAsync");

  --*
  -- * \brief Initializes device memory
  -- *
  -- * Sets the memory range of \p N 8-bit values to the specified value
  -- * \p uc.
  -- *
  -- * \param dstDevice - Destination device pointer
  -- * \param uc        - Value to set
  -- * \param N         - Number of elements
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- * \notefnerr
  -- * \note_memset
  -- *
  -- * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
  -- * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
  -- * ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
  -- * ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
  -- * ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
  -- * ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
  -- * ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
  -- * ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
  -- * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D8Async,
  -- * ::cuMemsetD2D16, ::cuMemsetD2D16Async, ::cuMemsetD2D32, ::cuMemsetD2D32Async,
  -- * ::cuMemsetD8Async, ::cuMemsetD16, ::cuMemsetD16Async,
  -- * ::cuMemsetD32, ::cuMemsetD32Async
  --  

   function cuMemsetD8_v2
     (dstDevice : CUdeviceptr;
      uc : unsigned_char;
      N : stddef_h.size_t) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:6349
   pragma Import (C, cuMemsetD8_v2, "cuMemsetD8_v2");

  --*
  -- * \brief Initializes device memory
  -- *
  -- * Sets the memory range of \p N 16-bit values to the specified value
  -- * \p us. The \p dstDevice pointer must be two byte aligned.
  -- *
  -- * \param dstDevice - Destination device pointer
  -- * \param us        - Value to set
  -- * \param N         - Number of elements
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- * \notefnerr
  -- * \note_memset
  -- *
  -- * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
  -- * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
  -- * ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
  -- * ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
  -- * ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
  -- * ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
  -- * ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
  -- * ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
  -- * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D8Async,
  -- * ::cuMemsetD2D16, ::cuMemsetD2D16Async, ::cuMemsetD2D32, ::cuMemsetD2D32Async,
  -- * ::cuMemsetD8, ::cuMemsetD8Async, ::cuMemsetD16Async,
  -- * ::cuMemsetD32, ::cuMemsetD32Async
  --  

   function cuMemsetD16_v2
     (dstDevice : CUdeviceptr;
      us : unsigned_short;
      N : stddef_h.size_t) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:6383
   pragma Import (C, cuMemsetD16_v2, "cuMemsetD16_v2");

  --*
  -- * \brief Initializes device memory
  -- *
  -- * Sets the memory range of \p N 32-bit values to the specified value
  -- * \p ui. The \p dstDevice pointer must be four byte aligned.
  -- *
  -- * \param dstDevice - Destination device pointer
  -- * \param ui        - Value to set
  -- * \param N         - Number of elements
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- * \notefnerr
  -- * \note_memset
  -- *
  -- * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
  -- * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
  -- * ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
  -- * ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
  -- * ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
  -- * ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
  -- * ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
  -- * ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
  -- * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D8Async,
  -- * ::cuMemsetD2D16, ::cuMemsetD2D16Async, ::cuMemsetD2D32, ::cuMemsetD2D32Async,
  -- * ::cuMemsetD8, ::cuMemsetD8Async, ::cuMemsetD16, ::cuMemsetD16Async,
  -- * ::cuMemsetD32Async
  --  

   function cuMemsetD32_v2
     (dstDevice : CUdeviceptr;
      ui : unsigned;
      N : stddef_h.size_t) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:6417
   pragma Import (C, cuMemsetD32_v2, "cuMemsetD32_v2");

  --*
  -- * \brief Initializes device memory
  -- *
  -- * Sets the 2D memory range of \p Width 8-bit values to the specified value
  -- * \p uc. \p Height specifies the number of rows to set, and \p dstPitch
  -- * specifies the number of bytes between each row. This function performs
  -- * fastest when the pitch is one that has been passed back by
  -- * ::cuMemAllocPitch().
  -- *
  -- * \param dstDevice - Destination device pointer
  -- * \param dstPitch  - Pitch of destination device pointer
  -- * \param uc        - Value to set
  -- * \param Width     - Width of row
  -- * \param Height    - Number of rows
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- * \notefnerr
  -- * \note_memset
  -- *
  -- * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
  -- * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
  -- * ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
  -- * ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
  -- * ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
  -- * ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
  -- * ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
  -- * ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
  -- * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8Async,
  -- * ::cuMemsetD2D16, ::cuMemsetD2D16Async, ::cuMemsetD2D32, ::cuMemsetD2D32Async,
  -- * ::cuMemsetD8, ::cuMemsetD8Async, ::cuMemsetD16, ::cuMemsetD16Async,
  -- * ::cuMemsetD32, ::cuMemsetD32Async
  --  

   function cuMemsetD2D8_v2
     (dstDevice : CUdeviceptr;
      dstPitch : stddef_h.size_t;
      uc : unsigned_char;
      Width : stddef_h.size_t;
      Height : stddef_h.size_t) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:6456
   pragma Import (C, cuMemsetD2D8_v2, "cuMemsetD2D8_v2");

  --*
  -- * \brief Initializes device memory
  -- *
  -- * Sets the 2D memory range of \p Width 16-bit values to the specified value
  -- * \p us. \p Height specifies the number of rows to set, and \p dstPitch
  -- * specifies the number of bytes between each row. The \p dstDevice pointer
  -- * and \p dstPitch offset must be two byte aligned. This function performs
  -- * fastest when the pitch is one that has been passed back by
  -- * ::cuMemAllocPitch().
  -- *
  -- * \param dstDevice - Destination device pointer
  -- * \param dstPitch  - Pitch of destination device pointer
  -- * \param us        - Value to set
  -- * \param Width     - Width of row
  -- * \param Height    - Number of rows
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- * \notefnerr
  -- * \note_memset
  -- *
  -- * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
  -- * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
  -- * ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
  -- * ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
  -- * ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
  -- * ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
  -- * ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
  -- * ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
  -- * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D8Async,
  -- * ::cuMemsetD2D16Async, ::cuMemsetD2D32, ::cuMemsetD2D32Async,
  -- * ::cuMemsetD8, ::cuMemsetD8Async, ::cuMemsetD16, ::cuMemsetD16Async,
  -- * ::cuMemsetD32, ::cuMemsetD32Async
  --  

   function cuMemsetD2D16_v2
     (dstDevice : CUdeviceptr;
      dstPitch : stddef_h.size_t;
      us : unsigned_short;
      Width : stddef_h.size_t;
      Height : stddef_h.size_t) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:6496
   pragma Import (C, cuMemsetD2D16_v2, "cuMemsetD2D16_v2");

  --*
  -- * \brief Initializes device memory
  -- *
  -- * Sets the 2D memory range of \p Width 32-bit values to the specified value
  -- * \p ui. \p Height specifies the number of rows to set, and \p dstPitch
  -- * specifies the number of bytes between each row. The \p dstDevice pointer
  -- * and \p dstPitch offset must be four byte aligned. This function performs
  -- * fastest when the pitch is one that has been passed back by
  -- * ::cuMemAllocPitch().
  -- *
  -- * \param dstDevice - Destination device pointer
  -- * \param dstPitch  - Pitch of destination device pointer
  -- * \param ui        - Value to set
  -- * \param Width     - Width of row
  -- * \param Height    - Number of rows
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- * \notefnerr
  -- * \note_memset
  -- *
  -- * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
  -- * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
  -- * ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
  -- * ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
  -- * ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
  -- * ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
  -- * ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
  -- * ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
  -- * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D8Async,
  -- * ::cuMemsetD2D16, ::cuMemsetD2D16Async, ::cuMemsetD2D32Async,
  -- * ::cuMemsetD8, ::cuMemsetD8Async, ::cuMemsetD16, ::cuMemsetD16Async,
  -- * ::cuMemsetD32, ::cuMemsetD32Async
  --  

   function cuMemsetD2D32_v2
     (dstDevice : CUdeviceptr;
      dstPitch : stddef_h.size_t;
      ui : unsigned;
      Width : stddef_h.size_t;
      Height : stddef_h.size_t) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:6536
   pragma Import (C, cuMemsetD2D32_v2, "cuMemsetD2D32_v2");

  --*
  -- * \brief Sets device memory
  -- *
  -- * Sets the memory range of \p N 8-bit values to the specified value
  -- * \p uc.
  -- *
  -- * \param dstDevice - Destination device pointer
  -- * \param uc        - Value to set
  -- * \param N         - Number of elements
  -- * \param hStream   - Stream identifier
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- * \notefnerr
  -- * \note_memset
  -- * \note_null_stream
  -- *
  -- * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
  -- * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
  -- * ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
  -- * ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
  -- * ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
  -- * ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
  -- * ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
  -- * ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
  -- * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D8Async,
  -- * ::cuMemsetD2D16, ::cuMemsetD2D16Async, ::cuMemsetD2D32, ::cuMemsetD2D32Async,
  -- * ::cuMemsetD8, ::cuMemsetD16, ::cuMemsetD16Async,
  -- * ::cuMemsetD32, ::cuMemsetD32Async
  --  

   function cuMemsetD8Async
     (dstDevice : CUdeviceptr;
      uc : unsigned_char;
      N : stddef_h.size_t;
      hStream : CUstream) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:6572
   pragma Import (C, cuMemsetD8Async, "cuMemsetD8Async");

  --*
  -- * \brief Sets device memory
  -- *
  -- * Sets the memory range of \p N 16-bit values to the specified value
  -- * \p us. The \p dstDevice pointer must be two byte aligned.
  -- *
  -- * \param dstDevice - Destination device pointer
  -- * \param us        - Value to set
  -- * \param N         - Number of elements
  -- * \param hStream   - Stream identifier
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- * \notefnerr
  -- * \note_memset
  -- * \note_null_stream
  -- *
  -- * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
  -- * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
  -- * ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
  -- * ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
  -- * ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
  -- * ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
  -- * ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
  -- * ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
  -- * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D8Async,
  -- * ::cuMemsetD2D16, ::cuMemsetD2D16Async, ::cuMemsetD2D32, ::cuMemsetD2D32Async,
  -- * ::cuMemsetD8, ::cuMemsetD8Async, ::cuMemsetD16,
  -- * ::cuMemsetD32, ::cuMemsetD32Async
  --  

   function cuMemsetD16Async
     (dstDevice : CUdeviceptr;
      us : unsigned_short;
      N : stddef_h.size_t;
      hStream : CUstream) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:6608
   pragma Import (C, cuMemsetD16Async, "cuMemsetD16Async");

  --*
  -- * \brief Sets device memory
  -- *
  -- * Sets the memory range of \p N 32-bit values to the specified value
  -- * \p ui. The \p dstDevice pointer must be four byte aligned.
  -- *
  -- * \param dstDevice - Destination device pointer
  -- * \param ui        - Value to set
  -- * \param N         - Number of elements
  -- * \param hStream   - Stream identifier
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- * \notefnerr
  -- * \note_memset
  -- * \note_null_stream
  -- *
  -- * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
  -- * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
  -- * ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
  -- * ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
  -- * ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
  -- * ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
  -- * ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
  -- * ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
  -- * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D8Async,
  -- * ::cuMemsetD2D16, ::cuMemsetD2D16Async, ::cuMemsetD2D32, ::cuMemsetD2D32Async,
  -- * ::cuMemsetD8, ::cuMemsetD8Async, ::cuMemsetD16, ::cuMemsetD16Async, ::cuMemsetD32
  --  

   function cuMemsetD32Async
     (dstDevice : CUdeviceptr;
      ui : unsigned;
      N : stddef_h.size_t;
      hStream : CUstream) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:6643
   pragma Import (C, cuMemsetD32Async, "cuMemsetD32Async");

  --*
  -- * \brief Sets device memory
  -- *
  -- * Sets the 2D memory range of \p Width 8-bit values to the specified value
  -- * \p uc. \p Height specifies the number of rows to set, and \p dstPitch
  -- * specifies the number of bytes between each row. This function performs
  -- * fastest when the pitch is one that has been passed back by
  -- * ::cuMemAllocPitch().
  -- *
  -- * \param dstDevice - Destination device pointer
  -- * \param dstPitch  - Pitch of destination device pointer
  -- * \param uc        - Value to set
  -- * \param Width     - Width of row
  -- * \param Height    - Number of rows
  -- * \param hStream   - Stream identifier
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- * \notefnerr
  -- * \note_memset
  -- * \note_null_stream
  -- *
  -- * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
  -- * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
  -- * ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
  -- * ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
  -- * ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
  -- * ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
  -- * ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
  -- * ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
  -- * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8,
  -- * ::cuMemsetD2D16, ::cuMemsetD2D16Async, ::cuMemsetD2D32, ::cuMemsetD2D32Async,
  -- * ::cuMemsetD8, ::cuMemsetD8Async, ::cuMemsetD16, ::cuMemsetD16Async,
  -- * ::cuMemsetD32, ::cuMemsetD32Async
  --  

   function cuMemsetD2D8Async
     (dstDevice : CUdeviceptr;
      dstPitch : stddef_h.size_t;
      uc : unsigned_char;
      Width : stddef_h.size_t;
      Height : stddef_h.size_t;
      hStream : CUstream) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:6684
   pragma Import (C, cuMemsetD2D8Async, "cuMemsetD2D8Async");

  --*
  -- * \brief Sets device memory
  -- *
  -- * Sets the 2D memory range of \p Width 16-bit values to the specified value
  -- * \p us. \p Height specifies the number of rows to set, and \p dstPitch
  -- * specifies the number of bytes between each row. The \p dstDevice pointer 
  -- * and \p dstPitch offset must be two byte aligned. This function performs
  -- * fastest when the pitch is one that has been passed back by
  -- * ::cuMemAllocPitch().
  -- *
  -- * \param dstDevice - Destination device pointer
  -- * \param dstPitch  - Pitch of destination device pointer
  -- * \param us        - Value to set
  -- * \param Width     - Width of row
  -- * \param Height    - Number of rows
  -- * \param hStream   - Stream identifier
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- * \notefnerr
  -- * \note_memset
  -- * \note_null_stream
  -- *
  -- * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
  -- * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
  -- * ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
  -- * ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
  -- * ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
  -- * ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
  -- * ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
  -- * ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
  -- * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D8Async,
  -- * ::cuMemsetD2D16, ::cuMemsetD2D32, ::cuMemsetD2D32Async,
  -- * ::cuMemsetD8, ::cuMemsetD8Async, ::cuMemsetD16, ::cuMemsetD16Async,
  -- * ::cuMemsetD32, ::cuMemsetD32Async
  --  

   function cuMemsetD2D16Async
     (dstDevice : CUdeviceptr;
      dstPitch : stddef_h.size_t;
      us : unsigned_short;
      Width : stddef_h.size_t;
      Height : stddef_h.size_t;
      hStream : CUstream) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:6726
   pragma Import (C, cuMemsetD2D16Async, "cuMemsetD2D16Async");

  --*
  -- * \brief Sets device memory
  -- *
  -- * Sets the 2D memory range of \p Width 32-bit values to the specified value
  -- * \p ui. \p Height specifies the number of rows to set, and \p dstPitch
  -- * specifies the number of bytes between each row. The \p dstDevice pointer
  -- * and \p dstPitch offset must be four byte aligned. This function performs
  -- * fastest when the pitch is one that has been passed back by
  -- * ::cuMemAllocPitch().
  -- *
  -- * \param dstDevice - Destination device pointer
  -- * \param dstPitch  - Pitch of destination device pointer
  -- * \param ui        - Value to set
  -- * \param Width     - Width of row
  -- * \param Height    - Number of rows
  -- * \param hStream   - Stream identifier
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- * \notefnerr
  -- * \note_memset
  -- * \note_null_stream
  -- *
  -- * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
  -- * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
  -- * ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
  -- * ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
  -- * ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
  -- * ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
  -- * ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
  -- * ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
  -- * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D8Async,
  -- * ::cuMemsetD2D16, ::cuMemsetD2D16Async, ::cuMemsetD2D32,
  -- * ::cuMemsetD8, ::cuMemsetD8Async, ::cuMemsetD16, ::cuMemsetD16Async,
  -- * ::cuMemsetD32, ::cuMemsetD32Async
  --  

   function cuMemsetD2D32Async
     (dstDevice : CUdeviceptr;
      dstPitch : stddef_h.size_t;
      ui : unsigned;
      Width : stddef_h.size_t;
      Height : stddef_h.size_t;
      hStream : CUstream) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:6768
   pragma Import (C, cuMemsetD2D32Async, "cuMemsetD2D32Async");

  --*
  -- * \brief Creates a 1D or 2D CUDA array
  -- *
  -- * Creates a CUDA array according to the ::CUDA_ARRAY_DESCRIPTOR structure
  -- * \p pAllocateArray and returns a handle to the new CUDA array in \p *pHandle.
  -- * The ::CUDA_ARRAY_DESCRIPTOR is defined as:
  -- *
  -- * \code
  --    typedef struct {
  --        unsigned int Width;
  --        unsigned int Height;
  --        CUarray_format Format;
  --        unsigned int NumChannels;
  --    } CUDA_ARRAY_DESCRIPTOR;
  -- * \endcode
  -- * where:
  -- *
  -- * - \p Width, and \p Height are the width, and height of the CUDA array (in
  -- * elements); the CUDA array is one-dimensional if height is 0, two-dimensional
  -- * otherwise;
  -- * - ::Format specifies the format of the elements; ::CUarray_format is
  -- * defined as:
  -- * \code
  --    typedef enum CUarray_format_enum {
  --        CU_AD_FORMAT_UNSIGNED_INT8 = 0x01,
  --        CU_AD_FORMAT_UNSIGNED_INT16 = 0x02,
  --        CU_AD_FORMAT_UNSIGNED_INT32 = 0x03,
  --        CU_AD_FORMAT_SIGNED_INT8 = 0x08,
  --        CU_AD_FORMAT_SIGNED_INT16 = 0x09,
  --        CU_AD_FORMAT_SIGNED_INT32 = 0x0a,
  --        CU_AD_FORMAT_HALF = 0x10,
  --        CU_AD_FORMAT_FLOAT = 0x20
  --    } CUarray_format;
  -- *  \endcode
  -- * - \p NumChannels specifies the number of packed components per CUDA array
  -- * element; it may be 1, 2, or 4;
  -- *
  -- * Here are examples of CUDA array descriptions:
  -- *
  -- * Description for a CUDA array of 2048 floats:
  -- * \code
  --    CUDA_ARRAY_DESCRIPTOR desc;
  --    desc.Format = CU_AD_FORMAT_FLOAT;
  --    desc.NumChannels = 1;
  --    desc.Width = 2048;
  --    desc.Height = 1;
  -- * \endcode
  -- *
  -- * Description for a 64 x 64 CUDA array of floats:
  -- * \code
  --    CUDA_ARRAY_DESCRIPTOR desc;
  --    desc.Format = CU_AD_FORMAT_FLOAT;
  --    desc.NumChannels = 1;
  --    desc.Width = 64;
  --    desc.Height = 64;
  -- * \endcode
  -- *
  -- * Description for a \p width x \p height CUDA array of 64-bit, 4x16-bit
  -- * float16's:
  -- * \code
  --    CUDA_ARRAY_DESCRIPTOR desc;
  --    desc.FormatFlags = CU_AD_FORMAT_HALF;
  --    desc.NumChannels = 4;
  --    desc.Width = width;
  --    desc.Height = height;
  -- * \endcode
  -- *
  -- * Description for a \p width x \p height CUDA array of 16-bit elements, each
  -- * of which is two 8-bit unsigned chars:
  -- * \code
  --    CUDA_ARRAY_DESCRIPTOR arrayDesc;
  --    desc.FormatFlags = CU_AD_FORMAT_UNSIGNED_INT8;
  --    desc.NumChannels = 2;
  --    desc.Width = width;
  --    desc.Height = height;
  -- * \endcode
  -- *
  -- * \param pHandle        - Returned array
  -- * \param pAllocateArray - Array descriptor
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE,
  -- * ::CUDA_ERROR_OUT_OF_MEMORY,
  -- * ::CUDA_ERROR_UNKNOWN
  -- * \notefnerr
  -- *
  -- * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor,
  -- * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
  -- * ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
  -- * ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
  -- * ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
  -- * ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
  -- * ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
  -- * ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
  -- * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D16,
  -- * ::cuMemsetD2D32, ::cuMemsetD8, ::cuMemsetD16, ::cuMemsetD32
  --  

   function cuArrayCreate_v2 (pHandle : System.Address; pAllocateArray : access constant CUDA_ARRAY_DESCRIPTOR) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:6871
   pragma Import (C, cuArrayCreate_v2, "cuArrayCreate_v2");

  --*
  -- * \brief Get a 1D or 2D CUDA array descriptor
  -- *
  -- * Returns in \p *pArrayDescriptor a descriptor containing information on the
  -- * format and dimensions of the CUDA array \p hArray. It is useful for
  -- * subroutines that have been passed a CUDA array, but need to know the CUDA
  -- * array parameters for validation or other purposes.
  -- *
  -- * \param pArrayDescriptor - Returned array descriptor
  -- * \param hArray           - Array to get descriptor of
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE,
  -- * ::CUDA_ERROR_INVALID_HANDLE
  -- * \notefnerr
  -- *
  -- * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
  -- * ::cuArrayDestroy, ::cuMemAlloc, ::cuMemAllocHost,
  -- * ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
  -- * ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
  -- * ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
  -- * ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
  -- * ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
  -- * ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
  -- * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D16,
  -- * ::cuMemsetD2D32, ::cuMemsetD8, ::cuMemsetD16, ::cuMemsetD32
  --  

   function cuArrayGetDescriptor_v2 (pArrayDescriptor : access CUDA_ARRAY_DESCRIPTOR; hArray : CUarray) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:6904
   pragma Import (C, cuArrayGetDescriptor_v2, "cuArrayGetDescriptor_v2");

  --*
  -- * \brief Destroys a CUDA array
  -- *
  -- * Destroys the CUDA array \p hArray.
  -- *
  -- * \param hArray - Array to destroy
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_HANDLE,
  -- * ::CUDA_ERROR_ARRAY_IS_MAPPED
  -- * \notefnerr
  -- *
  -- * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
  -- * ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
  -- * ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
  -- * ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
  -- * ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
  -- * ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
  -- * ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
  -- * ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
  -- * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D16,
  -- * ::cuMemsetD2D32, ::cuMemsetD8, ::cuMemsetD16, ::cuMemsetD32
  --  

   function cuArrayDestroy (hArray : CUarray) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:6935
   pragma Import (C, cuArrayDestroy, "cuArrayDestroy");

  --*
  -- * \brief Creates a 3D CUDA array
  -- *
  -- * Creates a CUDA array according to the ::CUDA_ARRAY3D_DESCRIPTOR structure
  -- * \p pAllocateArray and returns a handle to the new CUDA array in \p *pHandle.
  -- * The ::CUDA_ARRAY3D_DESCRIPTOR is defined as:
  -- *
  -- * \code
  --    typedef struct {
  --        unsigned int Width;
  --        unsigned int Height;
  --        unsigned int Depth;
  --        CUarray_format Format;
  --        unsigned int NumChannels;
  --        unsigned int Flags;
  --    } CUDA_ARRAY3D_DESCRIPTOR;
  -- * \endcode
  -- * where:
  -- *
  -- * - \p Width, \p Height, and \p Depth are the width, height, and depth of the
  -- * CUDA array (in elements); the following types of CUDA arrays can be allocated:
  -- *     - A 1D array is allocated if \p Height and \p Depth extents are both zero.
  -- *     - A 2D array is allocated if only \p Depth extent is zero.
  -- *     - A 3D array is allocated if all three extents are non-zero.
  -- *     - A 1D layered CUDA array is allocated if only \p Height is zero and the 
  -- *       ::CUDA_ARRAY3D_LAYERED flag is set. Each layer is a 1D array. The number 
  -- *       of layers is determined by the depth extent.
  -- *     - A 2D layered CUDA array is allocated if all three extents are non-zero and 
  -- *       the ::CUDA_ARRAY3D_LAYERED flag is set. Each layer is a 2D array. The number 
  -- *       of layers is determined by the depth extent.
  -- *     - A cubemap CUDA array is allocated if all three extents are non-zero and the
  -- *       ::CUDA_ARRAY3D_CUBEMAP flag is set. \p Width must be equal to \p Height, and 
  -- *       \p Depth must be six. A cubemap is a special type of 2D layered CUDA array, 
  -- *       where the six layers represent the six faces of a cube. The order of the six 
  -- *       layers in memory is the same as that listed in ::CUarray_cubemap_face.
  -- *     - A cubemap layered CUDA array is allocated if all three extents are non-zero, 
  -- *       and both, ::CUDA_ARRAY3D_CUBEMAP and ::CUDA_ARRAY3D_LAYERED flags are set. 
  -- *       \p Width must be equal to \p Height, and \p Depth must be a multiple of six. 
  -- *       A cubemap layered CUDA array is a special type of 2D layered CUDA array that 
  -- *       consists of a collection of cubemaps. The first six layers represent the first 
  -- *       cubemap, the next six layers form the second cubemap, and so on.
  -- *
  -- * - ::Format specifies the format of the elements; ::CUarray_format is
  -- * defined as:
  -- * \code
  --    typedef enum CUarray_format_enum {
  --        CU_AD_FORMAT_UNSIGNED_INT8 = 0x01,
  --        CU_AD_FORMAT_UNSIGNED_INT16 = 0x02,
  --        CU_AD_FORMAT_UNSIGNED_INT32 = 0x03,
  --        CU_AD_FORMAT_SIGNED_INT8 = 0x08,
  --        CU_AD_FORMAT_SIGNED_INT16 = 0x09,
  --        CU_AD_FORMAT_SIGNED_INT32 = 0x0a,
  --        CU_AD_FORMAT_HALF = 0x10,
  --        CU_AD_FORMAT_FLOAT = 0x20
  --    } CUarray_format;
  -- *  \endcode
  -- *
  -- * - \p NumChannels specifies the number of packed components per CUDA array
  -- * element; it may be 1, 2, or 4;
  -- *
  -- * - ::Flags may be set to 
  -- *   - ::CUDA_ARRAY3D_LAYERED to enable creation of layered CUDA arrays. If this flag is set, 
  -- *     \p Depth specifies the number of layers, not the depth of a 3D array.
  -- *   - ::CUDA_ARRAY3D_SURFACE_LDST to enable surface references to be bound to the CUDA array.  
  -- *     If this flag is not set, ::cuSurfRefSetArray will fail when attempting to bind the CUDA array 
  -- *     to a surface reference.
  -- *   - ::CUDA_ARRAY3D_CUBEMAP to enable creation of cubemaps. If this flag is set, \p Width must be
  -- *     equal to \p Height, and \p Depth must be six. If the ::CUDA_ARRAY3D_LAYERED flag is also set,
  -- *     then \p Depth must be a multiple of six.
  -- *   - ::CUDA_ARRAY3D_TEXTURE_GATHER to indicate that the CUDA array will be used for texture gather.
  -- *     Texture gather can only be performed on 2D CUDA arrays.
  -- *
  -- * \p Width, \p Height and \p Depth must meet certain size requirements as listed in the following table. 
  -- * All values are specified in elements. Note that for brevity's sake, the full name of the device attribute 
  -- * is not specified. For ex., TEXTURE1D_WIDTH refers to the device attribute 
  -- * ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH.
  -- *
  -- * Note that 2D CUDA arrays have different size requirements if the ::CUDA_ARRAY3D_TEXTURE_GATHER flag 
  -- * is set. \p Width and \p Height must not be greater than ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH 
  -- * and ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT respectively, in that case.
  -- *
  -- * <table>
  -- * <tr><td><b>CUDA array type</b></td>
  -- * <td><b>Valid extents that must always be met<br>{(width range in elements), (height range), 
  -- * (depth range)}</b></td>
  -- * <td><b>Valid extents with CUDA_ARRAY3D_SURFACE_LDST set<br> 
  -- * {(width range in elements), (height range), (depth range)}</b></td></tr>
  -- * <tr><td>1D</td>
  -- * <td><small>{ (1,TEXTURE1D_WIDTH), 0, 0 }</small></td>
  -- * <td><small>{ (1,SURFACE1D_WIDTH), 0, 0 }</small></td></tr>
  -- * <tr><td>2D</td>
  -- * <td><small>{ (1,TEXTURE2D_WIDTH), (1,TEXTURE2D_HEIGHT), 0 }</small></td>
  -- * <td><small>{ (1,SURFACE2D_WIDTH), (1,SURFACE2D_HEIGHT), 0 }</small></td></tr>
  -- * <tr><td>3D</td>
  -- * <td><small>{ (1,TEXTURE3D_WIDTH), (1,TEXTURE3D_HEIGHT), (1,TEXTURE3D_DEPTH) }
  -- * <br>OR<br>{ (1,TEXTURE3D_WIDTH_ALTERNATE), (1,TEXTURE3D_HEIGHT_ALTERNATE), 
  -- * (1,TEXTURE3D_DEPTH_ALTERNATE) }</small></td>
  -- * <td><small>{ (1,SURFACE3D_WIDTH), (1,SURFACE3D_HEIGHT), 
  -- * (1,SURFACE3D_DEPTH) }</small></td></tr>
  -- * <tr><td>1D Layered</td>
  -- * <td><small>{ (1,TEXTURE1D_LAYERED_WIDTH), 0, 
  -- * (1,TEXTURE1D_LAYERED_LAYERS) }</small></td>
  -- * <td><small>{ (1,SURFACE1D_LAYERED_WIDTH), 0, 
  -- * (1,SURFACE1D_LAYERED_LAYERS) }</small></td></tr>
  -- * <tr><td>2D Layered</td>
  -- * <td><small>{ (1,TEXTURE2D_LAYERED_WIDTH), (1,TEXTURE2D_LAYERED_HEIGHT), 
  -- * (1,TEXTURE2D_LAYERED_LAYERS) }</small></td>
  -- * <td><small>{ (1,SURFACE2D_LAYERED_WIDTH), (1,SURFACE2D_LAYERED_HEIGHT), 
  -- * (1,SURFACE2D_LAYERED_LAYERS) }</small></td></tr>
  -- * <tr><td>Cubemap</td>
  -- * <td><small>{ (1,TEXTURECUBEMAP_WIDTH), (1,TEXTURECUBEMAP_WIDTH), 6 }</small></td>
  -- * <td><small>{ (1,SURFACECUBEMAP_WIDTH), 
  -- * (1,SURFACECUBEMAP_WIDTH), 6 }</small></td></tr>
  -- * <tr><td>Cubemap Layered</td>
  -- * <td><small>{ (1,TEXTURECUBEMAP_LAYERED_WIDTH), (1,TEXTURECUBEMAP_LAYERED_WIDTH), 
  -- * (1,TEXTURECUBEMAP_LAYERED_LAYERS) }</small></td>
  -- * <td><small>{ (1,SURFACECUBEMAP_LAYERED_WIDTH), (1,SURFACECUBEMAP_LAYERED_WIDTH), 
  -- * (1,SURFACECUBEMAP_LAYERED_LAYERS) }</small></td></tr>
  -- * </table>
  -- *
  -- * Here are examples of CUDA array descriptions:
  -- *
  -- * Description for a CUDA array of 2048 floats:
  -- * \code
  --    CUDA_ARRAY3D_DESCRIPTOR desc;
  --    desc.Format = CU_AD_FORMAT_FLOAT;
  --    desc.NumChannels = 1;
  --    desc.Width = 2048;
  --    desc.Height = 0;
  --    desc.Depth = 0;
  -- * \endcode
  -- *
  -- * Description for a 64 x 64 CUDA array of floats:
  -- * \code
  --    CUDA_ARRAY3D_DESCRIPTOR desc;
  --    desc.Format = CU_AD_FORMAT_FLOAT;
  --    desc.NumChannels = 1;
  --    desc.Width = 64;
  --    desc.Height = 64;
  --    desc.Depth = 0;
  -- * \endcode
  -- *
  -- * Description for a \p width x \p height x \p depth CUDA array of 64-bit,
  -- * 4x16-bit float16's:
  -- * \code
  --    CUDA_ARRAY3D_DESCRIPTOR desc;
  --    desc.FormatFlags = CU_AD_FORMAT_HALF;
  --    desc.NumChannels = 4;
  --    desc.Width = width;
  --    desc.Height = height;
  --    desc.Depth = depth;
  -- * \endcode
  -- *
  -- * \param pHandle        - Returned array
  -- * \param pAllocateArray - 3D array descriptor
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE,
  -- * ::CUDA_ERROR_OUT_OF_MEMORY,
  -- * ::CUDA_ERROR_UNKNOWN
  -- * \notefnerr
  -- *
  -- * \sa ::cuArray3DGetDescriptor, ::cuArrayCreate,
  -- * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
  -- * ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
  -- * ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
  -- * ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
  -- * ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
  -- * ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
  -- * ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
  -- * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D16,
  -- * ::cuMemsetD2D32, ::cuMemsetD8, ::cuMemsetD16, ::cuMemsetD32
  --  

   function cuArray3DCreate_v2 (pHandle : System.Address; pAllocateArray : access constant CUDA_ARRAY3D_DESCRIPTOR) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:7115
   pragma Import (C, cuArray3DCreate_v2, "cuArray3DCreate_v2");

  --*
  -- * \brief Get a 3D CUDA array descriptor
  -- *
  -- * Returns in \p *pArrayDescriptor a descriptor containing information on the
  -- * format and dimensions of the CUDA array \p hArray. It is useful for
  -- * subroutines that have been passed a CUDA array, but need to know the CUDA
  -- * array parameters for validation or other purposes.
  -- *
  -- * This function may be called on 1D and 2D arrays, in which case the \p Height
  -- * and/or \p Depth members of the descriptor struct will be set to 0.
  -- *
  -- * \param pArrayDescriptor - Returned 3D array descriptor
  -- * \param hArray           - 3D array to get descriptor of
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE,
  -- * ::CUDA_ERROR_INVALID_HANDLE
  -- * \notefnerr
  -- *
  -- * \sa ::cuArray3DCreate, ::cuArrayCreate,
  -- * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
  -- * ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
  -- * ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
  -- * ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
  -- * ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
  -- * ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
  -- * ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
  -- * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D16,
  -- * ::cuMemsetD2D32, ::cuMemsetD8, ::cuMemsetD16, ::cuMemsetD32
  --  

   function cuArray3DGetDescriptor_v2 (pArrayDescriptor : access CUDA_ARRAY3D_DESCRIPTOR; hArray : CUarray) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:7151
   pragma Import (C, cuArray3DGetDescriptor_v2, "cuArray3DGetDescriptor_v2");

  --*
  -- * \brief Creates a CUDA mipmapped array
  -- *
  -- * Creates a CUDA mipmapped array according to the ::CUDA_ARRAY3D_DESCRIPTOR structure
  -- * \p pMipmappedArrayDesc and returns a handle to the new CUDA mipmapped array in \p *pHandle.
  -- * \p numMipmapLevels specifies the number of mipmap levels to be allocated. This value is
  -- * clamped to the range [1, 1 + floor(log2(max(width, height, depth)))].
  -- *
  -- * The ::CUDA_ARRAY3D_DESCRIPTOR is defined as:
  -- *
  -- * \code
  --    typedef struct {
  --        unsigned int Width;
  --        unsigned int Height;
  --        unsigned int Depth;
  --        CUarray_format Format;
  --        unsigned int NumChannels;
  --        unsigned int Flags;
  --    } CUDA_ARRAY3D_DESCRIPTOR;
  -- * \endcode
  -- * where:
  -- *
  -- * - \p Width, \p Height, and \p Depth are the width, height, and depth of the
  -- * CUDA array (in elements); the following types of CUDA arrays can be allocated:
  -- *     - A 1D mipmapped array is allocated if \p Height and \p Depth extents are both zero.
  -- *     - A 2D mipmapped array is allocated if only \p Depth extent is zero.
  -- *     - A 3D mipmapped array is allocated if all three extents are non-zero.
  -- *     - A 1D layered CUDA mipmapped array is allocated if only \p Height is zero and the 
  -- *       ::CUDA_ARRAY3D_LAYERED flag is set. Each layer is a 1D array. The number 
  -- *       of layers is determined by the depth extent.
  -- *     - A 2D layered CUDA mipmapped array is allocated if all three extents are non-zero and 
  -- *       the ::CUDA_ARRAY3D_LAYERED flag is set. Each layer is a 2D array. The number 
  -- *       of layers is determined by the depth extent.
  -- *     - A cubemap CUDA mipmapped array is allocated if all three extents are non-zero and the
  -- *       ::CUDA_ARRAY3D_CUBEMAP flag is set. \p Width must be equal to \p Height, and 
  -- *       \p Depth must be six. A cubemap is a special type of 2D layered CUDA array, 
  -- *       where the six layers represent the six faces of a cube. The order of the six 
  -- *       layers in memory is the same as that listed in ::CUarray_cubemap_face.
  -- *     - A cubemap layered CUDA mipmapped array is allocated if all three extents are non-zero, 
  -- *       and both, ::CUDA_ARRAY3D_CUBEMAP and ::CUDA_ARRAY3D_LAYERED flags are set. 
  -- *       \p Width must be equal to \p Height, and \p Depth must be a multiple of six. 
  -- *       A cubemap layered CUDA array is a special type of 2D layered CUDA array that 
  -- *       consists of a collection of cubemaps. The first six layers represent the first 
  -- *       cubemap, the next six layers form the second cubemap, and so on.
  -- *
  -- * - ::Format specifies the format of the elements; ::CUarray_format is
  -- * defined as:
  -- * \code
  --    typedef enum CUarray_format_enum {
  --        CU_AD_FORMAT_UNSIGNED_INT8 = 0x01,
  --        CU_AD_FORMAT_UNSIGNED_INT16 = 0x02,
  --        CU_AD_FORMAT_UNSIGNED_INT32 = 0x03,
  --        CU_AD_FORMAT_SIGNED_INT8 = 0x08,
  --        CU_AD_FORMAT_SIGNED_INT16 = 0x09,
  --        CU_AD_FORMAT_SIGNED_INT32 = 0x0a,
  --        CU_AD_FORMAT_HALF = 0x10,
  --        CU_AD_FORMAT_FLOAT = 0x20
  --    } CUarray_format;
  -- *  \endcode
  -- *
  -- * - \p NumChannels specifies the number of packed components per CUDA array
  -- * element; it may be 1, 2, or 4;
  -- *
  -- * - ::Flags may be set to 
  -- *   - ::CUDA_ARRAY3D_LAYERED to enable creation of layered CUDA mipmapped arrays. If this flag is set, 
  -- *     \p Depth specifies the number of layers, not the depth of a 3D array.
  -- *   - ::CUDA_ARRAY3D_SURFACE_LDST to enable surface references to be bound to individual mipmap levels of
  -- *     the CUDA mipmapped array. If this flag is not set, ::cuSurfRefSetArray will fail when attempting to 
  -- *     bind a mipmap level of the CUDA mipmapped array to a surface reference.
  --  *   - ::CUDA_ARRAY3D_CUBEMAP to enable creation of mipmapped cubemaps. If this flag is set, \p Width must be
  -- *     equal to \p Height, and \p Depth must be six. If the ::CUDA_ARRAY3D_LAYERED flag is also set,
  -- *     then \p Depth must be a multiple of six.
  -- *   - ::CUDA_ARRAY3D_TEXTURE_GATHER to indicate that the CUDA mipmapped array will be used for texture gather.
  -- *     Texture gather can only be performed on 2D CUDA mipmapped arrays.
  -- *
  -- * \p Width, \p Height and \p Depth must meet certain size requirements as listed in the following table. 
  -- * All values are specified in elements. Note that for brevity's sake, the full name of the device attribute 
  -- * is not specified. For ex., TEXTURE1D_MIPMAPPED_WIDTH refers to the device attribute 
  -- * ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH.
  -- *
  -- * <table>
  -- * <tr><td><b>CUDA array type</b></td>
  -- * <td><b>Valid extents that must always be met<br>{(width range in elements), (height range), 
  -- * (depth range)}</b></td></tr>
  -- * <tr><td>1D</td>
  -- * <td><small>{ (1,TEXTURE1D_MIPMAPPED_WIDTH), 0, 0 }</small></td></tr>
  -- * <tr><td>2D</td>
  -- * <td><small>{ (1,TEXTURE2D_MIPMAPPED_WIDTH), (1,TEXTURE2D_MIPMAPPED_HEIGHT), 0 }</small></td></tr>
  -- * <tr><td>3D</td>
  -- * <td><small>{ (1,TEXTURE3D_WIDTH), (1,TEXTURE3D_HEIGHT), (1,TEXTURE3D_DEPTH) }
  -- * <br>OR<br>{ (1,TEXTURE3D_WIDTH_ALTERNATE), (1,TEXTURE3D_HEIGHT_ALTERNATE), 
  -- * (1,TEXTURE3D_DEPTH_ALTERNATE) }</small></td></tr>
  -- * <tr><td>1D Layered</td>
  -- * <td><small>{ (1,TEXTURE1D_LAYERED_WIDTH), 0, 
  -- * (1,TEXTURE1D_LAYERED_LAYERS) }</small></td></tr>
  -- * <tr><td>2D Layered</td>
  -- * <td><small>{ (1,TEXTURE2D_LAYERED_WIDTH), (1,TEXTURE2D_LAYERED_HEIGHT), 
  -- * (1,TEXTURE2D_LAYERED_LAYERS) }</small></td></tr>
  -- * <tr><td>Cubemap</td>
  -- * <td><small>{ (1,TEXTURECUBEMAP_WIDTH), (1,TEXTURECUBEMAP_WIDTH), 6 }</small></td></tr>
  -- * <tr><td>Cubemap Layered</td>
  -- * <td><small>{ (1,TEXTURECUBEMAP_LAYERED_WIDTH), (1,TEXTURECUBEMAP_LAYERED_WIDTH), 
  -- * (1,TEXTURECUBEMAP_LAYERED_LAYERS) }</small></td></tr>
  -- * </table>
  -- *
  -- *
  -- * \param pHandle             - Returned mipmapped array
  -- * \param pMipmappedArrayDesc - mipmapped array descriptor
  -- * \param numMipmapLevels     - Number of mipmap levels
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE,
  -- * ::CUDA_ERROR_OUT_OF_MEMORY,
  -- * ::CUDA_ERROR_UNKNOWN
  -- * \notefnerr
  -- *
  -- * \sa ::cuMipmappedArrayDestroy, ::cuMipmappedArrayGetLevel, ::cuArrayCreate,
  --  

   function cuMipmappedArrayCreate
     (pHandle : System.Address;
      pMipmappedArrayDesc : access constant CUDA_ARRAY3D_DESCRIPTOR;
      numMipmapLevels : unsigned) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:7278
   pragma Import (C, cuMipmappedArrayCreate, "cuMipmappedArrayCreate");

  --*
  -- * \brief Gets a mipmap level of a CUDA mipmapped array
  -- *
  -- * Returns in \p *pLevelArray a CUDA array that represents a single mipmap level
  -- * of the CUDA mipmapped array \p hMipmappedArray.
  -- *
  -- * If \p level is greater than the maximum number of levels in this mipmapped array,
  -- * ::CUDA_ERROR_INVALID_VALUE is returned.
  -- *
  -- * \param pLevelArray     - Returned mipmap level CUDA array
  -- * \param hMipmappedArray - CUDA mipmapped array
  -- * \param level           - Mipmap level
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE,
  -- * ::CUDA_ERROR_INVALID_HANDLE
  -- * \notefnerr
  -- *
  -- * \sa ::cuMipmappedArrayCreate, ::cuMipmappedArrayDestroy, ::cuArrayCreate,
  --  

   function cuMipmappedArrayGetLevel
     (pLevelArray : System.Address;
      hMipmappedArray : CUmipmappedArray;
      level : unsigned) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:7304
   pragma Import (C, cuMipmappedArrayGetLevel, "cuMipmappedArrayGetLevel");

  --*
  -- * \brief Destroys a CUDA mipmapped array
  -- *
  -- * Destroys the CUDA mipmapped array \p hMipmappedArray.
  -- *
  -- * \param hMipmappedArray - Mipmapped array to destroy
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_HANDLE,
  -- * ::CUDA_ERROR_ARRAY_IS_MAPPED
  -- * \notefnerr
  -- *
  -- * \sa ::cuMipmappedArrayCreate, ::cuMipmappedArrayGetLevel, ::cuArrayCreate,
  --  

   function cuMipmappedArrayDestroy (hMipmappedArray : CUmipmappedArray) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:7324
   pragma Import (C, cuMipmappedArrayDestroy, "cuMipmappedArrayDestroy");

  --* @}  
  -- END CUDA_MEM  
  --*
  -- * \defgroup CUDA_UNIFIED Unified Addressing
  -- *
  -- * ___MANBRIEF___ unified addressing functions of the low-level CUDA driver
  -- * API (___CURRENT_FILE___) ___ENDMANBRIEF___
  -- *
  -- * This section describes the unified addressing functions of the 
  -- * low-level CUDA driver application programming interface.
  -- *
  -- * @{
  -- *
  -- * \section CUDA_UNIFIED_overview Overview
  -- *
  -- * CUDA devices can share a unified address space with the host.  
  -- * For these devices there is no distinction between a device
  -- * pointer and a host pointer -- the same pointer value may be 
  -- * used to access memory from the host program and from a kernel 
  -- * running on the device (with exceptions enumerated below).
  -- *
  -- * \section CUDA_UNIFIED_support Supported Platforms
  -- * 
  -- * Whether or not a device supports unified addressing may be 
  -- * queried by calling ::cuDeviceGetAttribute() with the device 
  -- * attribute ::CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING.
  -- *
  -- * Unified addressing is automatically enabled in 64-bit processes 
  -- * on devices with compute capability greater than or equal to 2.0.
  -- *
  -- * \section CUDA_UNIFIED_lookup Looking Up Information from Pointer Values
  -- *
  -- * It is possible to look up information about the memory which backs a 
  -- * pointer value.  For instance, one may want to know if a pointer points
  -- * to host or device memory.  As another example, in the case of device 
  -- * memory, one may want to know on which CUDA device the memory 
  -- * resides.  These properties may be queried using the function 
  -- * ::cuPointerGetAttribute()
  -- *
  -- * Since pointers are unique, it is not necessary to specify information
  -- * about the pointers specified to the various copy functions in the 
  -- * CUDA API.  The function ::cuMemcpy() may be used to perform a copy
  -- * between two pointers, ignoring whether they point to host or device
  -- * memory (making ::cuMemcpyHtoD(), ::cuMemcpyDtoD(), and ::cuMemcpyDtoH()
  -- * unnecessary for devices supporting unified addressing).  For 
  -- * multidimensional copies, the memory type ::CU_MEMORYTYPE_UNIFIED may be
  -- * used to specify that the CUDA driver should infer the location of the
  -- * pointer from its value.
  -- *
  -- * \section CUDA_UNIFIED_automaphost Automatic Mapping of Host Allocated Host Memory
  -- *
  -- * All host memory allocated in all contexts using ::cuMemAllocHost() and
  -- * ::cuMemHostAlloc() is always directly accessible from all contexts on
  -- * all devices that support unified addressing.  This is the case regardless 
  -- * of whether or not the flags ::CU_MEMHOSTALLOC_PORTABLE and
  -- * ::CU_MEMHOSTALLOC_DEVICEMAP are specified.
  -- *
  -- * The pointer value through which allocated host memory may be accessed 
  -- * in kernels on all devices that support unified addressing is the same 
  -- * as the pointer value through which that memory is accessed on the host,
  -- * so it is not necessary to call ::cuMemHostGetDevicePointer() to get the device 
  -- * pointer for these allocations.
  -- * 
  -- * Note that this is not the case for memory allocated using the flag
  -- * ::CU_MEMHOSTALLOC_WRITECOMBINED, as discussed below.
  -- *
  -- * \section CUDA_UNIFIED_autopeerregister Automatic Registration of Peer Memory
  -- *
  -- * Upon enabling direct access from a context that supports unified addressing 
  -- * to another peer context that supports unified addressing using 
  -- * ::cuCtxEnablePeerAccess() all memory allocated in the peer context using 
  -- * ::cuMemAlloc() and ::cuMemAllocPitch() will immediately be accessible 
  -- * by the current context.  The device pointer value through
  -- * which any peer memory may be accessed in the current context
  -- * is the same pointer value through which that memory may be
  -- * accessed in the peer context.
  -- *
  -- * \section CUDA_UNIFIED_exceptions Exceptions, Disjoint Addressing
  -- * 
  -- * Not all memory may be accessed on devices through the same pointer
  -- * value through which they are accessed on the host.  These exceptions
  -- * are host memory registered using ::cuMemHostRegister() and host memory
  -- * allocated using the flag ::CU_MEMHOSTALLOC_WRITECOMBINED.  For these 
  -- * exceptions, there exists a distinct host and device address for the
  -- * memory.  The device address is guaranteed to not overlap any valid host
  -- * pointer range and is guaranteed to have the same value across all 
  -- * contexts that support unified addressing.  
  -- * 
  -- * This device address may be queried using ::cuMemHostGetDevicePointer() 
  -- * when a context using unified addressing is current.  Either the host 
  -- * or the unified device pointer value may be used to refer to this memory 
  -- * through ::cuMemcpy() and similar functions using the 
  -- * ::CU_MEMORYTYPE_UNIFIED memory type.
  -- *
  --  

  --*
  -- * \brief Returns information about a pointer
  -- * 
  -- * The supported attributes are:
  -- * 
  -- * - ::CU_POINTER_ATTRIBUTE_CONTEXT: 
  -- * 
  -- *      Returns in \p *data the ::CUcontext in which \p ptr was allocated or 
  -- *      registered.   
  -- *      The type of \p data must be ::CUcontext *.  
  -- *      
  -- *      If \p ptr was not allocated by, mapped by, or registered with
  -- *      a ::CUcontext which uses unified virtual addressing then 
  -- *      ::CUDA_ERROR_INVALID_VALUE is returned.
  -- * 
  -- * - ::CU_POINTER_ATTRIBUTE_MEMORY_TYPE: 
  -- *    
  -- *      Returns in \p *data the physical memory type of the memory that 
  -- *      \p ptr addresses as a ::CUmemorytype enumerated value.
  -- *      The type of \p data must be unsigned int.
  -- *      
  -- *      If \p ptr addresses device memory then \p *data is set to 
  -- *      ::CU_MEMORYTYPE_DEVICE.  The particular ::CUdevice on which the 
  -- *      memory resides is the ::CUdevice of the ::CUcontext returned by the 
  -- *      ::CU_POINTER_ATTRIBUTE_CONTEXT attribute of \p ptr.
  -- *      
  -- *      If \p ptr addresses host memory then \p *data is set to 
  -- *      ::CU_MEMORYTYPE_HOST.
  -- *      
  -- *      If \p ptr was not allocated by, mapped by, or registered with
  -- *      a ::CUcontext which uses unified virtual addressing then 
  -- *      ::CUDA_ERROR_INVALID_VALUE is returned.
  -- *
  -- *      If the current ::CUcontext does not support unified virtual 
  -- *      addressing then ::CUDA_ERROR_INVALID_CONTEXT is returned.
  -- *    
  -- * - ::CU_POINTER_ATTRIBUTE_DEVICE_POINTER:
  -- * 
  -- *      Returns in \p *data the device pointer value through which
  -- *      \p ptr may be accessed by kernels running in the current 
  -- *      ::CUcontext.
  -- *      The type of \p data must be CUdeviceptr *.
  -- * 
  -- *      If there exists no device pointer value through which
  -- *      kernels running in the current ::CUcontext may access
  -- *      \p ptr then ::CUDA_ERROR_INVALID_VALUE is returned.
  -- * 
  -- *      If there is no current ::CUcontext then 
  -- *      ::CUDA_ERROR_INVALID_CONTEXT is returned.
  -- *      
  -- *      Except in the exceptional disjoint addressing cases discussed 
  -- *      below, the value returned in \p *data will equal the input 
  -- *      value \p ptr.
  -- * 
  -- * - ::CU_POINTER_ATTRIBUTE_HOST_POINTER:
  -- * 
  -- *      Returns in \p *data the host pointer value through which 
  -- *      \p ptr may be accessed by by the host program.
  -- *      The type of \p data must be void **.
  -- *      If there exists no host pointer value through which
  -- *      the host program may directly access \p ptr then 
  -- *      ::CUDA_ERROR_INVALID_VALUE is returned.
  -- * 
  -- *      Except in the exceptional disjoint addressing cases discussed 
  -- *      below, the value returned in \p *data will equal the input 
  -- *      value \p ptr.
  -- *
  -- * - ::CU_POINTER_ATTRIBUTE_P2P_TOKENS:
  -- *
  -- *      Returns in \p *data two tokens for use with the nv-p2p.h Linux
  -- *      kernel interface. \p data must be a struct of type
  -- *      CUDA_POINTER_ATTRIBUTE_P2P_TOKENS.
  -- *
  -- *      \p ptr must be a pointer to memory obtained from :cuMemAlloc().
  -- *      Note that p2pToken and vaSpaceToken are only valid for the
  -- *      lifetime of the source allocation. A subsequent allocation at
  -- *      the same address may return completely different tokens.
  -- *      Querying this attribute has a side effect of setting the attribute
  -- *      ::CU_POINTER_ATTRIBUTE_SYNC_MEMOPS for the region of memory that
  -- *      \p ptr points to.
  -- * 
  -- * - ::CU_POINTER_ATTRIBUTE_SYNC_MEMOPS:
  -- *
  -- *      A boolean attribute which when set, ensures that synchronous memory operations
  -- *      initiated on the region of memory that \p ptr points to will always synchronize.
  -- *      See further documentation in the section titled "API synchronization behavior"
  -- *      to learn more about cases when synchronous memory operations can
  -- *      exhibit asynchronous behavior.
  -- *
  -- * - ::CU_POINTER_ATTRIBUTE_BUFFER_ID:
  -- *
  -- *      Returns in \p *data a buffer ID which is guaranteed to be unique within the process.
  -- *      \p data must point to an unsigned long long.
  -- *
  -- *      \p ptr must be a pointer to memory obtained from a CUDA memory allocation API.
  -- *      Every memory allocation from any of the CUDA memory allocation APIs will
  -- *      have a unique ID over a process lifetime. Subsequent allocations do not reuse IDs
  -- *      from previous freed allocations. IDs are only unique within a single process.
  -- *
  -- *
  -- * - ::CU_POINTER_ATTRIBUTE_IS_MANAGED:
  -- *
  -- *      Returns in \p *data a boolean that indicates whether the pointer points to
  -- *      managed memory or not.
  -- *
  -- * \par
  -- *
  -- * Note that for most allocations in the unified virtual address space
  -- * the host and device pointer for accessing the allocation will be the 
  -- * same.  The exceptions to this are
  -- *  - user memory registered using ::cuMemHostRegister 
  -- *  - host memory allocated using ::cuMemHostAlloc with the 
  -- *    ::CU_MEMHOSTALLOC_WRITECOMBINED flag
  -- * For these types of allocation there will exist separate, disjoint host 
  -- * and device addresses for accessing the allocation.  In particular 
  -- *  - The host address will correspond to an invalid unmapped device address 
  -- *    (which will result in an exception if accessed from the device) 
  -- *  - The device address will correspond to an invalid unmapped host address 
  -- *    (which will result in an exception if accessed from the host).
  -- * For these types of allocations, querying ::CU_POINTER_ATTRIBUTE_HOST_POINTER 
  -- * and ::CU_POINTER_ATTRIBUTE_DEVICE_POINTER may be used to retrieve the host 
  -- * and device addresses from either address.
  -- *
  -- * \param data      - Returned pointer attribute value
  -- * \param attribute - Pointer attribute to query
  -- * \param ptr       - Pointer
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE,
  -- * ::CUDA_ERROR_INVALID_DEVICE
  -- * \notefnerr
  -- *
  -- * \sa cuPointerSetAttribute,
  -- * ::cuMemAlloc,
  -- * ::cuMemFree,
  -- * ::cuMemAllocHost,
  -- * ::cuMemFreeHost,
  -- * ::cuMemHostAlloc,
  -- * ::cuMemHostRegister,
  -- * ::cuMemHostUnregister
  --  

   function cuPointerGetAttribute
     (data : System.Address;
      attribute : CUpointer_attribute;
      ptr : CUdeviceptr) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:7570
   pragma Import (C, cuPointerGetAttribute, "cuPointerGetAttribute");

  --*
  -- * \brief Prefetches memory to the specified destination device
  -- *
  -- * Prefetches memory to the specified destination device.  \p devPtr is the 
  -- * base device pointer of the memory to be prefetched and \p dstDevice is the 
  -- * destination device. \p count specifies the number of bytes to copy. \p hStream
  -- * is the stream in which the operation is enqueued. The memory range must refer
  -- * to managed memory allocated via ::cuMemAllocManaged or declared via __managed__ variables.
  -- *
  -- * Passing in CU_DEVICE_CPU for \p dstDevice will prefetch the data to host memory. If
  -- * \p dstDevice is a GPU, then the device attribute ::CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS
  -- * must be non-zero. Additionally, \p hStream must be associated with a device that has a
  -- * non-zero value for the device attribute ::CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS.
  -- *
  -- * The start address and end address of the memory range will be rounded down and rounded up
  -- * respectively to be aligned to CPU page size before the prefetch operation is enqueued
  -- * in the stream.
  -- *
  -- * If no physical memory has been allocated for this region, then this memory region
  -- * will be populated and mapped on the destination device. If there's insufficient
  -- * memory to prefetch the desired region, the Unified Memory driver may evict pages from other
  -- * ::cuMemAllocManaged allocations to host memory in order to make room. Device memory
  -- * allocated using ::cuMemAlloc or ::cuArrayCreate will not be evicted.
  -- *
  -- * By default, any mappings to the previous location of the migrated pages are removed and
  -- * mappings for the new location are only setup on \p dstDevice. The exact behavior however
  -- * also depends on the settings applied to this memory range via ::cuMemAdvise as described
  -- * below:
  -- *
  -- * If ::CU_MEM_ADVISE_SET_READ_MOSTLY was set on any subset of this memory range,
  -- * then that subset will create a read-only copy of the pages on \p dstDevice.
  -- *
  -- * If ::CU_MEM_ADVISE_SET_PREFERRED_LOCATION was called on any subset of this memory
  -- * range, then the pages will be migrated to \p dstDevice even if \p dstDevice is not the
  -- * preferred location of any pages in the memory range.
  -- *
  -- * If ::CU_MEM_ADVISE_SET_ACCESSED_BY was called on any subset of this memory range,
  -- * then mappings to those pages from all the appropriate processors are updated to
  -- * refer to the new location if establishing such a mapping is possible. Otherwise,
  -- * those mappings are cleared.
  -- *
  -- * Note that this API is not required for functionality and only serves to improve performance
  -- * by allowing the application to migrate data to a suitable location before it is accessed.
  -- * Memory accesses to this range are always coherent and are allowed even when the data is
  -- * actively being migrated.
  -- *
  -- * Note that this function is asynchronous with respect to the host and all work
  -- * on other devices.
  -- *
  -- * \param devPtr    - Pointer to be prefetched
  -- * \param count     - Size in bytes
  -- * \param dstDevice - Destination device to prefetch to
  -- * \param hStream    - Stream to enqueue prefetch operation
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_INVALID_VALUE,
  -- * ::CUDA_ERROR_INVALID_DEVICE
  -- * \notefnerr
  -- * \note_async
  -- * \note_null_stream
  -- *
  -- * \sa ::cuMemcpy, ::cuMemcpyPeer, ::cuMemcpyAsync,
  -- * ::cuMemcpy3DPeerAsync, ::cuMemAdvise
  --  

   function cuMemPrefetchAsync
     (devPtr : CUdeviceptr;
      count : stddef_h.size_t;
      dstDevice : CUdevice;
      hStream : CUstream) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:7639
   pragma Import (C, cuMemPrefetchAsync, "cuMemPrefetchAsync");

  --*
  -- * \brief Advise about the usage of a given memory range
  -- *
  -- * Advise the Unified Memory subsystem about the usage pattern for the memory range
  -- * starting at \p devPtr with a size of \p count bytes. The start address and end address of the memory
  -- * range will be rounded down and rounded up respectively to be aligned to CPU page size before the
  -- * advice is applied. The memory range must refer to managed memory allocated via ::cuMemAllocManaged
  -- * or declared via __managed__ variables.
  -- *
  -- * The \p advice parameter can take the following values:
  -- * - ::CU_MEM_ADVISE_SET_READ_MOSTLY: This implies that the data is mostly going to be read
  -- * from and only occasionally written to. Any read accesses from any processor to this region will create a
  -- * read-only copy of at least the accessed pages in that processor's memory. Additionally, if ::cuMemPrefetchAsync
  -- * is called on this region, it will create a read-only copy of the data on the destination processor.
  -- * If any processor writes to this region, all copies of the corresponding page will be invalidated
  -- * except for the one where the write occurred. The \p device argument is ignored for this advice.
  -- * Note that for a page to be read-duplicated, the accessing processor must either be the CPU or a GPU
  -- * that has a non-zero value for the device attribute ::CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS.
  -- * Also, if a context is created on a device that does not have the device attribute
  -- * ::CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS set, then read-duplication will not occur until
  -- * all such contexts are destroyed.
  -- * - ::CU_MEM_ADVISE_UNSET_READ_MOSTLY:  Undoes the effect of ::CU_MEM_ADVISE_SET_READ_MOSTLY and also prevents the
  -- * Unified Memory driver from attempting heuristic read-duplication on the memory range. Any read-duplicated
  -- * copies of the data will be collapsed into a single copy. The location for the collapsed
  -- * copy will be the preferred location if the page has a preferred location and one of the read-duplicated
  -- * copies was resident at that location. Otherwise, the location chosen is arbitrary.
  -- * - ::CU_MEM_ADVISE_SET_PREFERRED_LOCATION: This advice sets the preferred location for the
  -- * data to be the memory belonging to \p device. Passing in CU_DEVICE_CPU for \p device sets the
  -- * preferred location as host memory. If \p device is a GPU, then it must have a non-zero value for the
  -- * device attribute ::CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS. Setting the preferred location
  -- * does not cause data to migrate to that location immediately. Instead, it guides the migration policy
  -- * when a fault occurs on that memory region. If the data is already in its preferred location and the
  -- * faulting processor can establish a mapping without requiring the data to be migrated, then
  -- * data migration will be avoided. On the other hand, if the data is not in its preferred location
  -- * or if a direct mapping cannot be established, then it will be migrated to the processor accessing
  -- * it. It is important to note that setting the preferred location does not prevent data prefetching
  -- * done using ::cuMemPrefetchAsync.
  -- * Having a preferred location can override the page thrash detection and resolution logic in the Unified
  -- * Memory driver. Normally, if a page is detected to be constantly thrashing between for example host and device
  -- * memory, the page may eventually be pinned to host memory by the Unified Memory driver. But
  -- * if the preferred location is set as device memory, then the page will continue to thrash indefinitely.
  -- * If ::CU_MEM_ADVISE_SET_READ_MOSTLY is also set on this memory region or any subset of it, then the
  -- * policies associated with that advice will override the policies of this advice.
  -- * - ::CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION: Undoes the effect of ::CU_MEM_ADVISE_SET_PREFERRED_LOCATION
  -- * and changes the preferred location to none.
  -- * - ::CU_MEM_ADVISE_SET_ACCESSED_BY: This advice implies that the data will be accessed by \p device.
  -- * Passing in ::CU_DEVICE_CPU for \p device will set the advice for the CPU. If \p device is a GPU, then
  -- * the device attribute ::CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS must be non-zero.
  -- * This advice does not cause data migration and has no impact on the location of the data per se. Instead,
  -- * it causes the data to always be mapped in the specified processor's page tables, as long as the
  -- * location of the data permits a mapping to be established. If the data gets migrated for any reason,
  -- * the mappings are updated accordingly.
  -- * This advice is recommended in scenarios where data locality is not important, but avoiding faults is.
  -- * Consider for example a system containing multiple GPUs with peer-to-peer access enabled, where the
  -- * data located on one GPU is occasionally accessed by peer GPUs. In such scenarios, migrating data
  -- * over to the other GPUs is not as important because the accesses are infrequent and the overhead of
  -- * migration may be too high. But preventing faults can still help improve performance, and so having
  -- * a mapping set up in advance is useful. Note that on CPU access of this data, the data may be migrated
  -- * to host memory because the CPU typically cannot access device memory directly. Any GPU that had the
  -- * ::CU_MEM_ADVISE_SET_ACCESSED_BY flag set for this data will now have its mapping updated to point to the
  -- * page in host memory.
  -- * If ::CU_MEM_ADVISE_SET_READ_MOSTLY is also set on this memory region or any subset of it, then the
  -- * policies associated with that advice will override the policies of this advice. Additionally, if the
  -- * preferred location of this memory region or any subset of it is also \p device, then the policies
  -- * associated with ::CU_MEM_ADVISE_SET_PREFERRED_LOCATION will override the policies of this advice.
  -- * - ::CU_MEM_ADVISE_UNSET_ACCESSED_BY: Undoes the effect of ::CU_MEM_ADVISE_SET_ACCESSED_BY. Any mappings to
  -- * the data from \p device may be removed at any time causing accesses to result in non-fatal page faults.
  -- *
  -- * \param devPtr - Pointer to memory to set the advice for
  -- * \param count  - Size in bytes of the memory range
  -- * \param advice - Advice to be applied for the specified memory range
  -- * \param device - Device to apply the advice for
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_INVALID_VALUE,
  -- * ::CUDA_ERROR_INVALID_DEVICE
  -- * \notefnerr
  -- * \note_async
  -- * \note_null_stream
  -- *
  -- * \sa ::cuMemcpy, ::cuMemcpyPeer, ::cuMemcpyAsync,
  -- * ::cuMemcpy3DPeerAsync, ::cuMemPrefetchAsync
  --  

   function cuMemAdvise
     (devPtr : CUdeviceptr;
      count : stddef_h.size_t;
      advice : CUmem_advise;
      device : CUdevice) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:7725
   pragma Import (C, cuMemAdvise, "cuMemAdvise");

  --*
  -- * \brief Query an attribute of a given memory range
  -- * 
  -- * Query an attribute about the memory range starting at \p devPtr with a size of \p count bytes. The
  -- * memory range must refer to managed memory allocated via ::cuMemAllocManaged or declared via
  -- * __managed__ variables.
  -- *
  -- * The \p attribute parameter can take the following values:
  -- * - ::CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY: If this attribute is specified, \p data will be interpreted
  -- * as a 32-bit integer, and \p dataSize must be 4. The result returned will be 1 if all pages in the given
  -- * memory range have read-duplication enabled, or 0 otherwise.
  -- * - ::CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION: If this attribute is specified, \p data will be
  -- * interpreted as a 32-bit integer, and \p dataSize must be 4. The result returned will be a GPU device
  -- * id if all pages in the memory range have that GPU as their preferred location, or it will be CU_DEVICE_CPU
  -- * if all pages in the memory range have the CPU as their preferred location, or it will be CU_DEVICE_INVALID
  -- * if either all the pages don't have the same preferred location or some of the pages don't have a
  -- * preferred location at all. Note that the actual location of the pages in the memory range at the time of
  -- * the query may be different from the preferred location. 
  -- * - ::CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY: If this attribute is specified, \p data will be interpreted
  -- * as an array of 32-bit integers, and \p dataSize must be a non-zero multiple of 4. The result returned
  -- * will be a list of device ids that had ::CU_MEM_ADVISE_SET_ACCESSED_BY set for that entire memory range.
  -- * If any device does not have that advice set for the entire memory range, that device will not be included.
  -- * If \p data is larger than the number of devices that have that advice set for that memory range,
  -- * CU_DEVICE_INVALID will be returned in all the extra space provided. For ex., if \p dataSize is 12
  -- * (i.e. \p data has 3 elements) and only device 0 has the advice set, then the result returned will be
  -- * { 0, CU_DEVICE_INVALID, CU_DEVICE_INVALID }. If \p data is smaller than the number of devices that have
  -- * that advice set, then only as many devices will be returned as can fit in the array. There is no
  -- * guarantee on which specific devices will be returned, however.
  -- * - ::CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION: If this attribute is specified, \p data will be
  -- * interpreted as a 32-bit integer, and \p dataSize must be 4. The result returned will be the last location
  -- * to which all pages in the memory range were prefetched explicitly via ::cuMemPrefetchAsync. This will either be
  -- * a GPU id or CU_DEVICE_CPU depending on whether the last location for prefetch was a GPU or the CPU
  -- * respectively. If any page in the memory range was never explicitly prefetched or if all pages were not
  -- * prefetched to the same location, CU_DEVICE_INVALID will be returned. Note that this simply returns the
  -- * last location that the applicaton requested to prefetch the memory range to. It gives no indication as to
  -- * whether the prefetch operation to that location has completed or even begun.
  -- *
  -- * \param data      - A pointers to a memory location where the result
  -- *                    of each attribute query will be written to.
  -- * \param dataSize  - Array containing the size of data
  -- * \param attribute - The attribute to query
  -- * \param devPtr    - Start of the range to query
  -- * \param count     - Size of the range to query
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_INVALID_VALUE,
  -- * ::CUDA_ERROR_INVALID_DEVICE
  -- * \notefnerr
  -- * \note_async
  -- * \note_null_stream
  -- *
  -- * \sa ::cuMemRangeGetAttributes, ::cuMemPrefetchAsync,
  -- * ::cuMemAdvise
  --  

   function cuMemRangeGetAttribute
     (data : System.Address;
      dataSize : stddef_h.size_t;
      attribute : CUmem_range_attribute;
      devPtr : CUdeviceptr;
      count : stddef_h.size_t) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:7782
   pragma Import (C, cuMemRangeGetAttribute, "cuMemRangeGetAttribute");

  --*
  -- * \brief Query attributes of a given memory range.
  -- *
  -- * Query attributes of the memory range starting at \p devPtr with a size of \p count bytes. The
  -- * memory range must refer to managed memory allocated via ::cuMemAllocManaged or declared via
  -- * __managed__ variables. The \p attributes array will be interpreted to have \p numAttributes
  -- * entries. The \p dataSizes array will also be interpreted to have \p numAttributes entries.
  -- * The results of the query will be stored in \p data.
  -- *
  -- * The list of supported attributes are given below. Please refer to ::cuMemRangeGetAttribute for
  -- * attribute descriptions and restrictions.
  -- *
  -- * - ::CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY
  -- * - ::CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION
  -- * - ::CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY
  -- * - ::CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION
  -- *
  -- * \param data          - A two-dimensional array containing pointers to memory
  -- *                        locations where the result of each attribute query will be written to.
  -- * \param dataSizes     - Array containing the sizes of each result
  -- * \param attributes    - An array of attributes to query
  -- *                        (numAttributes and the number of attributes in this array should match)
  -- * \param numAttributes - Number of attributes to query
  -- * \param devPtr        - Start of the range to query
  -- * \param count         - Size of the range to query
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE,
  -- * ::CUDA_ERROR_INVALID_DEVICE
  -- * \notefnerr
  -- *
  -- * \sa ::cuMemRangeGetAttribute, ::cuMemAdvise
  -- * ::cuMemPrefetchAsync
  --  

   function cuMemRangeGetAttributes
     (data : System.Address;
      dataSizes : access stddef_h.size_t;
      attributes : access CUmem_range_attribute;
      numAttributes : stddef_h.size_t;
      devPtr : CUdeviceptr;
      count : stddef_h.size_t) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:7821
   pragma Import (C, cuMemRangeGetAttributes, "cuMemRangeGetAttributes");

  --*
  -- * \brief Set attributes on a previously allocated memory region
  -- *
  -- * The supported attributes are:
  -- *
  -- * - ::CU_POINTER_ATTRIBUTE_SYNC_MEMOPS:
  -- *
  -- *      A boolean attribute that can either be set (1) or unset (0). When set,
  -- *      the region of memory that \p ptr points to is guaranteed to always synchronize
  -- *      memory operations that are synchronous. If there are some previously initiated
  -- *      synchronous memory operations that are pending when this attribute is set, the
  -- *      function does not return until those memory operations are complete.
  -- *      See further documentation in the section titled "API synchronization behavior"
  -- *      to learn more about cases when synchronous memory operations can
  -- *      exhibit asynchronous behavior.
  -- *      \p value will be considered as a pointer to an unsigned integer to which this attribute is to be set.
  -- *
  -- * \param value     - Pointer to memory containing the value to be set
  -- * \param attribute - Pointer attribute to set
  -- * \param ptr       - Pointer to a memory region allocated using CUDA memory allocation APIs
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE,
  -- * ::CUDA_ERROR_INVALID_DEVICE
  -- * \notefnerr
  -- *
  -- * \sa ::cuPointerGetAttribute,
  -- * ::cuPointerGetAttributes,
  -- * ::cuMemAlloc,
  -- * ::cuMemFree,
  -- * ::cuMemAllocHost,
  -- * ::cuMemFreeHost,
  -- * ::cuMemHostAlloc,
  -- * ::cuMemHostRegister,
  -- * ::cuMemHostUnregister
  --  

   function cuPointerSetAttribute
     (value : System.Address;
      attribute : CUpointer_attribute;
      ptr : CUdeviceptr) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:7865
   pragma Import (C, cuPointerSetAttribute, "cuPointerSetAttribute");

  --*
  -- * \brief Returns information about a pointer.
  -- *
  -- * The supported attributes are (refer to ::cuPointerGetAttribute for attribute descriptions and restrictions):
  -- *
  -- * - ::CU_POINTER_ATTRIBUTE_CONTEXT
  -- * - ::CU_POINTER_ATTRIBUTE_MEMORY_TYPE
  -- * - ::CU_POINTER_ATTRIBUTE_DEVICE_POINTER
  -- * - ::CU_POINTER_ATTRIBUTE_HOST_POINTER
  -- * - ::CU_POINTER_ATTRIBUTE_SYNC_MEMOPS
  -- * - ::CU_POINTER_ATTRIBUTE_BUFFER_ID
  -- * - ::CU_POINTER_ATTRIBUTE_IS_MANAGED
  -- *
  -- * \param numAttributes - Number of attributes to query
  -- * \param attributes    - An array of attributes to query
  -- *                      (numAttributes and the number of attributes in this array should match)
  -- * \param data          - A two-dimensional array containing pointers to memory
  -- *                      locations where the result of each attribute query will be written to.
  -- * \param ptr           - Pointer to query
  -- *
  -- * Unlike ::cuPointerGetAttribute, this function will not return an error when the \p ptr
  -- * encountered is not a valid CUDA pointer. Instead, the attributes are assigned default NULL values
  -- * and CUDA_SUCCESS is returned.
  -- *
  -- * If \p ptr was not allocated by, mapped by, or registered with a ::CUcontext which uses UVA
  -- * (Unified Virtual Addressing), ::CUDA_ERROR_INVALID_CONTEXT is returned.
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE,
  -- * ::CUDA_ERROR_INVALID_DEVICE
  -- * \notefnerr
  -- *
  -- * \sa ::cuPointerGetAttribute,
  -- * ::cuPointerSetAttribute
  --  

   function cuPointerGetAttributes
     (numAttributes : unsigned;
      attributes : access CUpointer_attribute;
      data : System.Address;
      ptr : CUdeviceptr) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:7907
   pragma Import (C, cuPointerGetAttributes, "cuPointerGetAttributes");

  --* @}  
  -- END CUDA_UNIFIED  
  --*
  -- * \defgroup CUDA_STREAM Stream Management
  -- *
  -- * ___MANBRIEF___ stream management functions of the low-level CUDA driver API
  -- * (___CURRENT_FILE___) ___ENDMANBRIEF___
  -- *
  -- * This section describes the stream management functions of the low-level CUDA
  -- * driver application programming interface.
  -- *
  -- * @{
  --  

  --*
  -- * \brief Create a stream
  -- *
  -- * Creates a stream and returns a handle in \p phStream.  The \p Flags argument
  -- * determines behaviors of the stream.  Valid values for \p Flags are:
  -- * - ::CU_STREAM_DEFAULT: Default stream creation flag.
  -- * - ::CU_STREAM_NON_BLOCKING: Specifies that work running in the created 
  -- *   stream may run concurrently with work in stream 0 (the NULL stream), and that
  -- *   the created stream should perform no implicit synchronization with stream 0.
  -- *
  -- * \param phStream - Returned newly created stream
  -- * \param Flags    - Parameters for stream creation
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE,
  -- * ::CUDA_ERROR_OUT_OF_MEMORY
  -- * \notefnerr
  -- *
  -- * \sa ::cuStreamDestroy,
  -- * ::cuStreamCreateWithPriority,
  -- * ::cuStreamGetPriority,
  -- * ::cuStreamGetFlags,
  -- * ::cuStreamWaitEvent,
  -- * ::cuStreamQuery,
  -- * ::cuStreamSynchronize,
  -- * ::cuStreamAddCallback
  --  

   function cuStreamCreate (phStream : System.Address; Flags : unsigned) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:7955
   pragma Import (C, cuStreamCreate, "cuStreamCreate");

  --*
  -- * \brief Create a stream with the given priority
  -- *
  -- * Creates a stream with the specified priority and returns a handle in \p phStream.
  -- * This API alters the scheduler priority of work in the stream. Work in a higher
  -- * priority stream may preempt work already executing in a low priority stream.
  -- *
  -- * \p priority follows a convention where lower numbers represent higher priorities.
  -- * '0' represents default priority. The range of meaningful numerical priorities can
  -- * be queried using ::cuCtxGetStreamPriorityRange. If the specified priority is
  -- * outside the numerical range returned by ::cuCtxGetStreamPriorityRange,
  -- * it will automatically be clamped to the lowest or the highest number in the range.
  -- *
  -- * \param phStream    - Returned newly created stream
  -- * \param flags       - Flags for stream creation. See ::cuStreamCreate for a list of
  -- *                      valid flags
  -- * \param priority    - Stream priority. Lower numbers represent higher priorities.
  -- *                      See ::cuCtxGetStreamPriorityRange for more information about
  -- *                      meaningful stream priorities that can be passed.
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE,
  -- * ::CUDA_ERROR_OUT_OF_MEMORY
  -- * \notefnerr
  -- *
  -- * \note Stream priorities are supported only on Quadro and Tesla GPUs
  -- * with compute capability 3.5 or higher.
  -- *
  -- * \note In the current implementation, only compute kernels launched in
  -- * priority streams are affected by the stream's priority. Stream priorities have
  -- * no effect on host-to-device and device-to-host memory operations.
  -- *
  -- * \sa ::cuStreamDestroy,
  -- * ::cuStreamCreate,
  -- * ::cuStreamGetPriority,
  -- * ::cuCtxGetStreamPriorityRange,
  -- * ::cuStreamGetFlags,
  -- * ::cuStreamWaitEvent,
  -- * ::cuStreamQuery,
  -- * ::cuStreamSynchronize,
  -- * ::cuStreamAddCallback
  --  

   function cuStreamCreateWithPriority
     (phStream : System.Address;
      flags : unsigned;
      priority : int) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:8003
   pragma Import (C, cuStreamCreateWithPriority, "cuStreamCreateWithPriority");

  --*
  -- * \brief Query the priority of a given stream
  -- *
  -- * Query the priority of a stream created using ::cuStreamCreate or ::cuStreamCreateWithPriority
  -- * and return the priority in \p priority. Note that if the stream was created with a
  -- * priority outside the numerical range returned by ::cuCtxGetStreamPriorityRange,
  -- * this function returns the clamped priority.
  -- * See ::cuStreamCreateWithPriority for details about priority clamping.
  -- *
  -- * \param hStream    - Handle to the stream to be queried
  -- * \param priority   - Pointer to a signed integer in which the stream's priority is returned
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE,
  -- * ::CUDA_ERROR_INVALID_HANDLE,
  -- * ::CUDA_ERROR_OUT_OF_MEMORY
  -- * \notefnerr
  -- *
  -- * \sa ::cuStreamDestroy,
  -- * ::cuStreamCreate,
  -- * ::cuStreamCreateWithPriority,
  -- * ::cuCtxGetStreamPriorityRange,
  -- * ::cuStreamGetFlags
  --  

   function cuStreamGetPriority (hStream : CUstream; priority : access int) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:8033
   pragma Import (C, cuStreamGetPriority, "cuStreamGetPriority");

  --*
  -- * \brief Query the flags of a given stream
  -- *
  -- * Query the flags of a stream created using ::cuStreamCreate or ::cuStreamCreateWithPriority
  -- * and return the flags in \p flags.
  -- *
  -- * \param hStream    - Handle to the stream to be queried
  -- * \param flags      - Pointer to an unsigned integer in which the stream's flags are returned
  -- *                     The value returned in \p flags is a logical 'OR' of all flags that
  -- *                     were used while creating this stream. See ::cuStreamCreate for the list
  -- *                     of valid flags
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE,
  -- * ::CUDA_ERROR_INVALID_HANDLE,
  -- * ::CUDA_ERROR_OUT_OF_MEMORY
  -- * \notefnerr
  -- *
  -- * \sa ::cuStreamDestroy,
  -- * ::cuStreamCreate,
  -- * ::cuStreamGetPriority
  --  

   function cuStreamGetFlags (hStream : CUstream; flags : access unsigned) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:8060
   pragma Import (C, cuStreamGetFlags, "cuStreamGetFlags");

  --*
  -- * \brief Make a compute stream wait on an event
  -- *
  -- * Makes all future work submitted to \p hStream wait until \p hEvent
  -- * reports completion before beginning execution.  This synchronization
  -- * will be performed efficiently on the device.  The event \p hEvent may
  -- * be from a different context than \p hStream, in which case this function
  -- * will perform cross-device synchronization.
  -- *
  -- * The stream \p hStream will wait only for the completion of the most recent
  -- * host call to ::cuEventRecord() on \p hEvent.  Once this call has returned,
  -- * any functions (including ::cuEventRecord() and ::cuEventDestroy()) may be
  -- * called on \p hEvent again, and subsequent calls will not have any
  -- * effect on \p hStream.
  -- *
  -- * If ::cuEventRecord() has not been called on \p hEvent, this call acts as if
  -- * the record has already completed, and so is a functional no-op.
  -- *
  -- * \param hStream - Stream to wait
  -- * \param hEvent  - Event to wait on (may not be NULL)
  -- * \param Flags   - Parameters for the operation (must be 0)
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_HANDLE,
  -- * \note_null_stream
  -- * \notefnerr
  -- *
  -- * \sa ::cuStreamCreate,
  -- * ::cuEventRecord,
  -- * ::cuStreamQuery,
  -- * ::cuStreamSynchronize,
  -- * ::cuStreamAddCallback,
  -- * ::cuStreamDestroy
  --  

   function cuStreamWaitEvent
     (hStream : CUstream;
      hEvent : CUevent;
      Flags : unsigned) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:8101
   pragma Import (C, cuStreamWaitEvent, "cuStreamWaitEvent");

  --*
  -- * \brief Add a callback to a compute stream
  -- *
  -- * Adds a callback to be called on the host after all currently enqueued
  -- * items in the stream have completed.  For each 
  -- * cuStreamAddCallback call, the callback will be executed exactly once.
  -- * The callback will block later work in the stream until it is finished.
  -- *
  -- * The callback may be passed ::CUDA_SUCCESS or an error code.  In the event
  -- * of a device error, all subsequently executed callbacks will receive an
  -- * appropriate ::CUresult.
  -- *
  -- * Callbacks must not make any CUDA API calls.  Attempting to use a CUDA API
  -- * will result in ::CUDA_ERROR_NOT_PERMITTED.  Callbacks must not perform any
  -- * synchronization that may depend on outstanding device work or other callbacks
  -- * that are not mandated to run earlier.  Callbacks without a mandated order
  -- * (in independent streams) execute in undefined order and may be serialized.
  -- *
  -- * This API requires compute capability 1.1 or greater.  See
  -- * ::cuDeviceGetAttribute or ::cuDeviceGetProperties to query compute
  -- * capability.  Attempting to use this API with earlier compute versions will
  -- * return ::CUDA_ERROR_NOT_SUPPORTED.
  -- *
  -- * For the purposes of Unified Memory, callback execution makes a number of
  -- * guarantees:
  -- * <ul>
  -- *   <li>The callback stream is considered idle for the duration of the
  -- *   callback.  Thus, for example, a callback may always use memory attached
  -- *   to the callback stream.</li>
  -- *   <li>The start of execution of a callback has the same effect as
  -- *   synchronizing an event recorded in the same stream immediately prior to
  -- *   the callback.  It thus synchronizes streams which have been "joined"
  -- *   prior to the callback.</li>
  -- *   <li>Adding device work to any stream does not have the effect of making
  -- *   the stream active until all preceding callbacks have executed.  Thus, for
  -- *   example, a callback might use global attached memory even if work has
  -- *   been added to another stream, if it has been properly ordered with an
  -- *   event.</li>
  -- *   <li>Completion of a callback does not cause a stream to become
  -- *   active except as described above.  The callback stream will remain idle
  -- *   if no device work follows the callback, and will remain idle across
  -- *   consecutive callbacks without device work in between.  Thus, for example,
  -- *   stream synchronization can be done by signaling from a callback at the
  -- *   end of the stream.</li>
  -- * </ul>
  -- *
  -- * \param hStream  - Stream to add callback to
  -- * \param callback - The function to call once preceding stream operations are complete
  -- * \param userData - User specified data to be passed to the callback function
  -- * \param flags    - Reserved for future use, must be 0
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_HANDLE,
  -- * ::CUDA_ERROR_NOT_SUPPORTED
  -- * \note_null_stream
  -- * \notefnerr
  -- *
  -- * \sa ::cuStreamCreate,
  -- * ::cuStreamQuery,
  -- * ::cuStreamSynchronize,
  -- * ::cuStreamWaitEvent,
  -- * ::cuStreamDestroy,
  -- * ::cuMemAllocManaged,
  -- * ::cuStreamAttachMemAsync
  --  

   function cuStreamAddCallback
     (hStream : CUstream;
      callback : CUstreamCallback;
      userData : System.Address;
      flags : unsigned) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:8172
   pragma Import (C, cuStreamAddCallback, "cuStreamAddCallback");

  --*
  -- * \brief Attach memory to a stream asynchronously
  -- *
  -- * Enqueues an operation in \p hStream to specify stream association of
  -- * \p length bytes of memory starting from \p dptr. This function is a
  -- * stream-ordered operation, meaning that it is dependent on, and will
  -- * only take effect when, previous work in stream has completed. Any
  -- * previous association is automatically replaced.
  -- *
  -- * \p dptr must point to an address within managed memory space declared
  -- * using the __managed__ keyword or allocated with ::cuMemAllocManaged.
  -- *
  -- * \p length must be zero, to indicate that the entire allocation's
  -- * stream association is being changed. Currently, it's not possible
  -- * to change stream association for a portion of an allocation.
  -- *
  -- * The stream association is specified using \p flags which must be
  -- * one of ::CUmemAttach_flags.
  -- * If the ::CU_MEM_ATTACH_GLOBAL flag is specified, the memory can be accessed
  -- * by any stream on any device.
  -- * If the ::CU_MEM_ATTACH_HOST flag is specified, the program makes a guarantee
  -- * that it won't access the memory on the device from any stream on a device that
  -- * has a zero value for the device attribute ::CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS.
  -- * If the ::CU_MEM_ATTACH_SINGLE flag is specified and \p hStream is associated with
  -- * a device that has a zero value for the device attribute ::CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS,
  -- * the program makes a guarantee that it will only access the memory on the device
  -- * from \p hStream. It is illegal to attach singly to the NULL stream, because the
  -- * NULL stream is a virtual global stream and not a specific stream. An error will
  -- * be returned in this case.
  -- *
  -- * When memory is associated with a single stream, the Unified Memory system will
  -- * allow CPU access to this memory region so long as all operations in \p hStream
  -- * have completed, regardless of whether other streams are active. In effect,
  -- * this constrains exclusive ownership of the managed memory region by
  -- * an active GPU to per-stream activity instead of whole-GPU activity.
  -- *
  -- * Accessing memory on the device from streams that are not associated with
  -- * it will produce undefined results. No error checking is performed by the
  -- * Unified Memory system to ensure that kernels launched into other streams
  -- * do not access this region. 
  -- *
  -- * It is a program's responsibility to order calls to ::cuStreamAttachMemAsync
  -- * via events, synchronization or other means to ensure legal access to memory
  -- * at all times. Data visibility and coherency will be changed appropriately
  -- * for all kernels which follow a stream-association change.
  -- *
  -- * If \p hStream is destroyed while data is associated with it, the association is
  -- * removed and the association reverts to the default visibility of the allocation
  -- * as specified at ::cuMemAllocManaged. For __managed__ variables, the default
  -- * association is always ::CU_MEM_ATTACH_GLOBAL. Note that destroying a stream is an
  -- * asynchronous operation, and as a result, the change to default association won't
  -- * happen until all work in the stream has completed.
  -- *
  -- * \param hStream - Stream in which to enqueue the attach operation
  -- * \param dptr    - Pointer to memory (must be a pointer to managed memory)
  -- * \param length  - Length of memory (must be zero)
  -- * \param flags   - Must be one of ::CUmemAttach_flags
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_HANDLE,
  -- * ::CUDA_ERROR_NOT_SUPPORTED
  -- * \note_null_stream
  -- * \notefnerr
  -- *
  -- * \sa ::cuStreamCreate,
  -- * ::cuStreamQuery,
  -- * ::cuStreamSynchronize,
  -- * ::cuStreamWaitEvent,
  -- * ::cuStreamDestroy,
  -- * ::cuMemAllocManaged
  --  

   function cuStreamAttachMemAsync
     (hStream : CUstream;
      dptr : CUdeviceptr;
      length : stddef_h.size_t;
      flags : unsigned) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:8251
   pragma Import (C, cuStreamAttachMemAsync, "cuStreamAttachMemAsync");

  --*
  -- * \brief Determine status of a compute stream
  -- *
  -- * Returns ::CUDA_SUCCESS if all operations in the stream specified by
  -- * \p hStream have completed, or ::CUDA_ERROR_NOT_READY if not.
  -- *
  -- * For the purposes of Unified Memory, a return value of ::CUDA_SUCCESS
  -- * is equivalent to having called ::cuStreamSynchronize().
  -- *
  -- * \param hStream - Stream to query status of
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_HANDLE,
  -- * ::CUDA_ERROR_NOT_READY
  -- * \note_null_stream
  -- * \notefnerr
  -- *
  -- * \sa ::cuStreamCreate,
  -- * ::cuStreamWaitEvent,
  -- * ::cuStreamDestroy,
  -- * ::cuStreamSynchronize,
  -- * ::cuStreamAddCallback
  --  

   function cuStreamQuery (hStream : CUstream) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:8282
   pragma Import (C, cuStreamQuery, "cuStreamQuery");

  --*
  -- * \brief Wait until a stream's tasks are completed
  -- *
  -- * Waits until the device has completed all operations in the stream specified
  -- * by \p hStream. If the context was created with the 
  -- * ::CU_CTX_SCHED_BLOCKING_SYNC flag, the CPU thread will block until the
  -- * stream is finished with all of its tasks.
  -- *
  -- * \param hStream - Stream to wait for
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_HANDLE
  -- * \note_null_stream
  -- * \notefnerr
  -- *
  -- * \sa ::cuStreamCreate,
  -- * ::cuStreamDestroy,
  -- * ::cuStreamWaitEvent,
  -- * ::cuStreamQuery,
  -- * ::cuStreamAddCallback
  --  

   function cuStreamSynchronize (hStream : CUstream) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:8309
   pragma Import (C, cuStreamSynchronize, "cuStreamSynchronize");

  --*
  -- * \brief Destroys a stream
  -- *
  -- * Destroys the stream specified by \p hStream.  
  -- *
  -- * In case the device is still doing work in the stream \p hStream
  -- * when ::cuStreamDestroy() is called, the function will return immediately 
  -- * and the resources associated with \p hStream will be released automatically 
  -- * once the device has completed all work in \p hStream.
  -- *
  -- * \param hStream - Stream to destroy
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- * \notefnerr
  -- *
  -- * \sa ::cuStreamCreate,
  -- * ::cuStreamWaitEvent,
  -- * ::cuStreamQuery,
  -- * ::cuStreamSynchronize,
  -- * ::cuStreamAddCallback
  --  

   function cuStreamDestroy_v2 (hStream : CUstream) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:8338
   pragma Import (C, cuStreamDestroy_v2, "cuStreamDestroy_v2");

  --* @}  
  -- END CUDA_STREAM  
  --*
  -- * \defgroup CUDA_EVENT Event Management
  -- *
  -- * ___MANBRIEF___ event management functions of the low-level CUDA driver API
  -- * (___CURRENT_FILE___) ___ENDMANBRIEF___
  -- *
  -- * This section describes the event management functions of the low-level CUDA
  -- * driver application programming interface.
  -- *
  -- * @{
  --  

  --*
  -- * \brief Creates an event
  -- *
  -- * Creates an event *phEvent with the flags specified via \p Flags. Valid flags
  -- * include:
  -- * - ::CU_EVENT_DEFAULT: Default event creation flag.
  -- * - ::CU_EVENT_BLOCKING_SYNC: Specifies that the created event should use blocking
  -- *   synchronization.  A CPU thread that uses ::cuEventSynchronize() to wait on
  -- *   an event created with this flag will block until the event has actually
  -- *   been recorded.
  -- * - ::CU_EVENT_DISABLE_TIMING: Specifies that the created event does not need
  -- *   to record timing data.  Events created with this flag specified and
  -- *   the ::CU_EVENT_BLOCKING_SYNC flag not specified will provide the best
  -- *   performance when used with ::cuStreamWaitEvent() and ::cuEventQuery().
  -- * - ::CU_EVENT_INTERPROCESS: Specifies that the created event may be used as an
  -- *   interprocess event by ::cuIpcGetEventHandle(). ::CU_EVENT_INTERPROCESS must
  -- *   be specified along with ::CU_EVENT_DISABLE_TIMING.
  -- *
  -- * \param phEvent - Returns newly created event
  -- * \param Flags   - Event creation flags
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE,
  -- * ::CUDA_ERROR_OUT_OF_MEMORY
  -- * \notefnerr
  -- *
  -- * \sa
  -- * ::cuEventRecord,
  -- * ::cuEventQuery,
  -- * ::cuEventSynchronize,
  -- * ::cuEventDestroy,
  -- * ::cuEventElapsedTime
  --  

   function cuEventCreate (phEvent : System.Address; Flags : unsigned) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:8393
   pragma Import (C, cuEventCreate, "cuEventCreate");

  --*
  -- * \brief Records an event
  -- *
  -- * Records an event. See note on NULL stream behavior. Since operation is
  -- * asynchronous, ::cuEventQuery or ::cuEventSynchronize() must be used
  -- * to determine when the event has actually been recorded.
  -- *
  -- * If ::cuEventRecord() has previously been called on \p hEvent, then this
  -- * call will overwrite any existing state in \p hEvent.  Any subsequent calls
  -- * which examine the status of \p hEvent will only examine the completion of
  -- * this most recent call to ::cuEventRecord().
  -- *
  -- * It is necessary that \p hEvent and \p hStream be created on the same context.
  -- *
  -- * \param hEvent  - Event to record
  -- * \param hStream - Stream to record event for
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_HANDLE,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- * \note_null_stream
  -- * \notefnerr
  -- *
  -- * \sa ::cuEventCreate,
  -- * ::cuEventQuery,
  -- * ::cuEventSynchronize,
  -- * ::cuStreamWaitEvent,
  -- * ::cuEventDestroy,
  -- * ::cuEventElapsedTime
  --  

   function cuEventRecord (hEvent : CUevent; hStream : CUstream) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:8429
   pragma Import (C, cuEventRecord, "cuEventRecord");

  --*
  -- * \brief Queries an event's status
  -- *
  -- * Query the status of all device work preceding the most recent
  -- * call to ::cuEventRecord() (in the appropriate compute streams,
  -- * as specified by the arguments to ::cuEventRecord()).
  -- *
  -- * If this work has successfully been completed by the device, or if
  -- * ::cuEventRecord() has not been called on \p hEvent, then ::CUDA_SUCCESS is
  -- * returned. If this work has not yet been completed by the device then
  -- * ::CUDA_ERROR_NOT_READY is returned.
  -- *
  -- * For the purposes of Unified Memory, a return value of ::CUDA_SUCCESS
  -- * is equivalent to having called ::cuEventSynchronize().
  -- *
  -- * \param hEvent - Event to query
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_HANDLE,
  -- * ::CUDA_ERROR_INVALID_VALUE,
  -- * ::CUDA_ERROR_NOT_READY
  -- * \notefnerr
  -- *
  -- * \sa ::cuEventCreate,
  -- * ::cuEventRecord,
  -- * ::cuEventSynchronize,
  -- * ::cuEventDestroy,
  -- * ::cuEventElapsedTime
  --  

   function cuEventQuery (hEvent : CUevent) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:8463
   pragma Import (C, cuEventQuery, "cuEventQuery");

  --*
  -- * \brief Waits for an event to complete
  -- *
  -- * Wait until the completion of all device work preceding the most recent
  -- * call to ::cuEventRecord() (in the appropriate compute streams, as specified
  -- * by the arguments to ::cuEventRecord()).
  -- *
  -- * If ::cuEventRecord() has not been called on \p hEvent, ::CUDA_SUCCESS is
  -- * returned immediately.
  -- *
  -- * Waiting for an event that was created with the ::CU_EVENT_BLOCKING_SYNC
  -- * flag will cause the calling CPU thread to block until the event has
  -- * been completed by the device.  If the ::CU_EVENT_BLOCKING_SYNC flag has
  -- * not been set, then the CPU thread will busy-wait until the event has
  -- * been completed by the device.
  -- *
  -- * \param hEvent - Event to wait for
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_HANDLE
  -- * \notefnerr
  -- *
  -- * \sa ::cuEventCreate,
  -- * ::cuEventRecord,
  -- * ::cuEventQuery,
  -- * ::cuEventDestroy,
  -- * ::cuEventElapsedTime
  --  

   function cuEventSynchronize (hEvent : CUevent) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:8497
   pragma Import (C, cuEventSynchronize, "cuEventSynchronize");

  --*
  -- * \brief Destroys an event
  -- *
  -- * Destroys the event specified by \p hEvent.
  -- *
  -- * In case \p hEvent has been recorded but has not yet been completed
  -- * when ::cuEventDestroy() is called, the function will return immediately and 
  -- * the resources associated with \p hEvent will be released automatically once
  -- * the device has completed \p hEvent.
  -- *
  -- * \param hEvent - Event to destroy
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_HANDLE
  -- * \notefnerr
  -- *
  -- * \sa ::cuEventCreate,
  -- * ::cuEventRecord,
  -- * ::cuEventQuery,
  -- * ::cuEventSynchronize,
  -- * ::cuEventElapsedTime
  --  

   function cuEventDestroy_v2 (hEvent : CUevent) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:8526
   pragma Import (C, cuEventDestroy_v2, "cuEventDestroy_v2");

  --*
  -- * \brief Computes the elapsed time between two events
  -- *
  -- * Computes the elapsed time between two events (in milliseconds with a
  -- * resolution of around 0.5 microseconds).
  -- *
  -- * If either event was last recorded in a non-NULL stream, the resulting time
  -- * may be greater than expected (even if both used the same stream handle). This
  -- * happens because the ::cuEventRecord() operation takes place asynchronously
  -- * and there is no guarantee that the measured latency is actually just between
  -- * the two events. Any number of other different stream operations could execute
  -- * in between the two measured events, thus altering the timing in a significant
  -- * way.
  -- *
  -- * If ::cuEventRecord() has not been called on either event then
  -- * ::CUDA_ERROR_INVALID_HANDLE is returned. If ::cuEventRecord() has been called
  -- * on both events but one or both of them has not yet been completed (that is,
  -- * ::cuEventQuery() would return ::CUDA_ERROR_NOT_READY on at least one of the
  -- * events), ::CUDA_ERROR_NOT_READY is returned. If either event was created with
  -- * the ::CU_EVENT_DISABLE_TIMING flag, then this function will return
  -- * ::CUDA_ERROR_INVALID_HANDLE.
  -- *
  -- * \param pMilliseconds - Time between \p hStart and \p hEnd in ms
  -- * \param hStart        - Starting event
  -- * \param hEnd          - Ending event
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_HANDLE,
  -- * ::CUDA_ERROR_NOT_READY
  -- * \notefnerr
  -- *
  -- * \sa ::cuEventCreate,
  -- * ::cuEventRecord,
  -- * ::cuEventQuery,
  -- * ::cuEventSynchronize,
  -- * ::cuEventDestroy
  --  

   function cuEventElapsedTime
     (pMilliseconds : access float;
      hStart : CUevent;
      hEnd : CUevent) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:8570
   pragma Import (C, cuEventElapsedTime, "cuEventElapsedTime");

  --*
  -- * \brief Wait on a memory location
  -- *
  -- * Enqueues a synchronization of the stream on the given memory location. Work
  -- * ordered after the operation will block until the given condition on the
  -- * memory is satisfied. By default, the condition is to wait for
  -- * (int32_t)(*addr - value) >= 0, a cyclic greater-or-equal.
  -- * Other condition types can be specified via \p flags.
  -- *
  -- * If the memory was registered via ::cuMemHostRegister(), the device pointer
  -- * should be obtained with ::cuMemHostGetDevicePointer(). This function cannot
  -- * be used with managed memory (::cuMemAllocManaged).
  -- *
  -- * On Windows, the device must be using TCC, or the operation is not supported.
  -- * See ::cuDeviceGetAttributes().
  -- *
  -- * \param stream The stream to synchronize on the memory location.
  -- * \param addr The memory location to wait on.
  -- * \param value The value to compare with the memory location.
  -- * \param flags See ::CUstreamWaitValue_flags.
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_INVALID_VALUE,
  -- * ::CUDA_ERROR_NOT_SUPPORTED
  -- * \notefnerr
  -- *
  -- * \sa ::cuStreamWriteValue32,
  -- * ::cuStreamBatchMemOp,
  -- * ::cuMemHostRegister,
  -- * ::cuStreamWaitEvent
  --  

   function cuStreamWaitValue32
     (stream : CUstream;
      addr : CUdeviceptr;
      value : cuuint32_t;
      flags : unsigned) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:8605
   pragma Import (C, cuStreamWaitValue32, "cuStreamWaitValue32");

  --*
  -- * \brief Write a value to memory
  -- *
  -- * Write a value to memory. Unless the ::CU_STREAM_WRITE_VALUE_NO_MEMORY_BARRIER
  -- * flag is passed, the write is preceded by a system-wide memory fence,
  -- * equivalent to a __threadfence_system() but scoped to the stream
  -- * rather than a CUDA thread.
  -- *
  -- * If the memory was registered via ::cuMemHostRegister(), the device pointer
  -- * should be obtained with ::cuMemHostGetDevicePointer(). This function cannot
  -- * be used with managed memory (::cuMemAllocManaged).
  -- *
  -- * On Windows, the device must be using TCC, or the operation is not supported.
  -- * See ::cuDeviceGetAttribute().
  -- *
  -- * \param stream The stream to do the write in.
  -- * \param addr The device address to write to.
  -- * \param value The value to write.
  -- * \param flags See ::CUstreamWriteValue_flags.
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_INVALID_VALUE,
  -- * ::CUDA_ERROR_NOT_SUPPORTED
  -- * \notefnerr
  -- *
  -- * \sa ::cuStreamWaitValue32,
  -- * ::cuStreamBatchMemOp,
  -- * ::cuMemHostRegister,
  -- * ::cuEventRecord
  --  

   function cuStreamWriteValue32
     (stream : CUstream;
      addr : CUdeviceptr;
      value : cuuint32_t;
      flags : unsigned) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:8638
   pragma Import (C, cuStreamWriteValue32, "cuStreamWriteValue32");

  --*
  -- * \brief Batch operations to synchronize the stream via memory operations
  -- *
  -- * This is a batch version of ::cuStreamWaitValue32() and ::cuStreamWriteValue32().
  -- * Batching operations may avoid some performance overhead in both the API call
  -- * and the device execution versus adding them to the stream in separate API
  -- * calls. The operations are enqueued in the order they appear in the array.
  -- *
  -- * See ::CUstreamBatchMemOpType for the full set of supported operations, and
  -- * ::cuStreamWaitValue32() and ::cuStreamWriteValue32() for details of specific
  -- * operations.
  -- *
  -- * On Windows, the device must be using TCC, or this call is not supported. See
  -- * ::cuDeviceGetAttribute().
  -- *
  -- * \param stream The stream to enqueue the operations in.
  -- * \param count The number of operations in the array. Must be less than 256.
  -- * \param paramArray The types and parameters of the individual operations.
  -- * \param flags Reserved for future expansion; must be 0.
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_INVALID_VALUE,
  -- * ::CUDA_ERROR_NOT_SUPPORTED
  -- * \notefnerr
  -- *
  -- * \sa ::cuStreamWaitValue32,
  -- * ::cuStreamWriteValue32,
  -- * ::cuMemHostRegister
  --  

   function cuStreamBatchMemOp
     (stream : CUstream;
      count : unsigned;
      paramArray : access CUstreamBatchMemOpParams;
      flags : unsigned) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:8670
   pragma Import (C, cuStreamBatchMemOp, "cuStreamBatchMemOp");

  --* @}  
  -- END CUDA_EVENT  
  --*
  -- * \defgroup CUDA_EXEC Execution Control
  -- *
  -- * ___MANBRIEF___ execution control functions of the low-level CUDA driver API
  -- * (___CURRENT_FILE___) ___ENDMANBRIEF___
  -- *
  -- * This section describes the execution control functions of the low-level CUDA
  -- * driver application programming interface.
  -- *
  -- * @{
  --  

  --*
  -- * \brief Returns information about a function
  -- *
  -- * Returns in \p *pi the integer value of the attribute \p attrib on the kernel
  -- * given by \p hfunc. The supported attributes are:
  -- * - ::CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK: The maximum number of threads
  -- *   per block, beyond which a launch of the function would fail. This number
  -- *   depends on both the function and the device on which the function is
  -- *   currently loaded.
  -- * - ::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES: The size in bytes of
  -- *   statically-allocated shared memory per block required by this function.
  -- *   This does not include dynamically-allocated shared memory requested by
  -- *   the user at runtime.
  -- * - ::CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES: The size in bytes of user-allocated
  -- *   constant memory required by this function.
  -- * - ::CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES: The size in bytes of local memory
  -- *   used by each thread of this function.
  -- * - ::CU_FUNC_ATTRIBUTE_NUM_REGS: The number of registers used by each thread
  -- *   of this function.
  -- * - ::CU_FUNC_ATTRIBUTE_PTX_VERSION: The PTX virtual architecture version for
  -- *   which the function was compiled. This value is the major PTX version * 10
  -- *   + the minor PTX version, so a PTX version 1.3 function would return the
  -- *   value 13. Note that this may return the undefined value of 0 for cubins
  -- *   compiled prior to CUDA 3.0.
  -- * - ::CU_FUNC_ATTRIBUTE_BINARY_VERSION: The binary architecture version for
  -- *   which the function was compiled. This value is the major binary
  -- *   version * 10 + the minor binary version, so a binary version 1.3 function
  -- *   would return the value 13. Note that this will return a value of 10 for
  -- *   legacy cubins that do not have a properly-encoded binary architecture
  -- *   version.
  -- * - ::CU_FUNC_CACHE_MODE_CA: The attribute to indicate whether the function has  
  -- *   been compiled with user specified option "-Xptxas --dlcm=ca" set .
  -- *
  -- * \param pi     - Returned attribute value
  -- * \param attrib - Attribute requested
  -- * \param hfunc  - Function to query attribute of
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_HANDLE,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- * \notefnerr
  -- *
  -- * \sa ::cuCtxGetCacheConfig,
  -- * ::cuCtxSetCacheConfig,
  -- * ::cuFuncSetCacheConfig,
  -- * ::cuLaunchKernel
  --  

   function cuFuncGetAttribute
     (pi : access int;
      attrib : CUfunction_attribute;
      hfunc : CUfunction) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:8738
   pragma Import (C, cuFuncGetAttribute, "cuFuncGetAttribute");

  --*
  -- * \brief Sets the preferred cache configuration for a device function
  -- *
  -- * On devices where the L1 cache and shared memory use the same hardware
  -- * resources, this sets through \p config the preferred cache configuration for
  -- * the device function \p hfunc. This is only a preference. The driver will use
  -- * the requested configuration if possible, but it is free to choose a different
  -- * configuration if required to execute \p hfunc.  Any context-wide preference
  -- * set via ::cuCtxSetCacheConfig() will be overridden by this per-function
  -- * setting unless the per-function setting is ::CU_FUNC_CACHE_PREFER_NONE. In
  -- * that case, the current context-wide setting will be used.
  -- *
  -- * This setting does nothing on devices where the size of the L1 cache and
  -- * shared memory are fixed.
  -- *
  -- * Launching a kernel with a different preference than the most recent
  -- * preference setting may insert a device-side synchronization point.
  -- *
  -- *
  -- * The supported cache configurations are:
  -- * - ::CU_FUNC_CACHE_PREFER_NONE: no preference for shared memory or L1 (default)
  -- * - ::CU_FUNC_CACHE_PREFER_SHARED: prefer larger shared memory and smaller L1 cache
  -- * - ::CU_FUNC_CACHE_PREFER_L1: prefer larger L1 cache and smaller shared memory
  -- * - ::CU_FUNC_CACHE_PREFER_EQUAL: prefer equal sized L1 cache and shared memory
  -- *
  -- * \param hfunc  - Kernel to configure cache for
  -- * \param config - Requested cache configuration
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_INVALID_VALUE,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT
  -- * \notefnerr
  -- *
  -- * \sa ::cuCtxGetCacheConfig,
  -- * ::cuCtxSetCacheConfig,
  -- * ::cuFuncGetAttribute,
  -- * ::cuLaunchKernel
  --  

   function cuFuncSetCacheConfig (hfunc : CUfunction; config : CUfunc_cache) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:8781
   pragma Import (C, cuFuncSetCacheConfig, "cuFuncSetCacheConfig");

  --*
  -- * \brief Sets the shared memory configuration for a device function.
  -- *
  -- * On devices with configurable shared memory banks, this function will 
  -- * force all subsequent launches of the specified device function to have
  -- * the given shared memory bank size configuration. On any given launch of the
  -- * function, the shared memory configuration of the device will be temporarily
  -- * changed if needed to suit the function's preferred configuration. Changes in
  -- * shared memory configuration between subsequent launches of functions, 
  -- * may introduce a device side synchronization point.
  -- *
  -- * Any per-function setting of shared memory bank size set via 
  -- * ::cuFuncSetSharedMemConfig will override the context wide setting set with
  -- * ::cuCtxSetSharedMemConfig.
  -- *
  -- * Changing the shared memory bank size will not increase shared memory usage
  -- * or affect occupancy of kernels, but may have major effects on performance. 
  -- * Larger bank sizes will allow for greater potential bandwidth to shared memory,
  -- * but will change what kinds of accesses to shared memory will result in bank 
  -- * conflicts.
  -- *
  -- * This function will do nothing on devices with fixed shared memory bank size.
  -- *
  -- * The supported bank configurations are:
  -- * - ::CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE: use the context's shared memory 
  -- *   configuration when launching this function.
  -- * - ::CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE: set shared memory bank width to
  -- *   be natively four bytes when launching this function.
  -- * - ::CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE: set shared memory bank width to
  -- *   be natively eight bytes when launching this function.
  -- *
  -- * \param hfunc  - kernel to be given a shared memory config
  -- * \param config - requested shared memory configuration
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_INVALID_VALUE,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT
  -- * \notefnerr
  -- *
  -- * \sa ::cuCtxGetCacheConfig,
  -- * ::cuCtxSetCacheConfig,
  -- * ::cuCtxGetSharedMemConfig,
  -- * ::cuCtxSetSharedMemConfig,
  -- * ::cuFuncGetAttribute,
  -- * ::cuLaunchKernel
  --  

   function cuFuncSetSharedMemConfig (hfunc : CUfunction; config : CUsharedconfig) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:8833
   pragma Import (C, cuFuncSetSharedMemConfig, "cuFuncSetSharedMemConfig");

  --*
  -- * \brief Launches a CUDA function
  -- *
  -- * Invokes the kernel \p f on a \p gridDimX x \p gridDimY x \p gridDimZ
  -- * grid of blocks. Each block contains \p blockDimX x \p blockDimY x
  -- * \p blockDimZ threads.
  -- *
  -- * \p sharedMemBytes sets the amount of dynamic shared memory that will be
  -- * available to each thread block.
  -- *
  -- * Kernel parameters to \p f can be specified in one of two ways:
  -- *
  -- * 1) Kernel parameters can be specified via \p kernelParams.  If \p f
  -- * has N parameters, then \p kernelParams needs to be an array of N
  -- * pointers.  Each of \p kernelParams[0] through \p kernelParams[N-1]
  -- * must point to a region of memory from which the actual kernel
  -- * parameter will be copied.  The number of kernel parameters and their
  -- * offsets and sizes do not need to be specified as that information is
  -- * retrieved directly from the kernel's image.
  -- *
  -- * 2) Kernel parameters can also be packaged by the application into
  -- * a single buffer that is passed in via the \p extra parameter.
  -- * This places the burden on the application of knowing each kernel
  -- * parameter's size and alignment/padding within the buffer.  Here is
  -- * an example of using the \p extra parameter in this manner:
  -- * \code
  --    size_t argBufferSize;
  --    char argBuffer[256];
  --    // populate argBuffer and argBufferSize
  --    void *config[] = {
  --        CU_LAUNCH_PARAM_BUFFER_POINTER, argBuffer,
  --        CU_LAUNCH_PARAM_BUFFER_SIZE,    &argBufferSize,
  --        CU_LAUNCH_PARAM_END
  --    };
  --    status = cuLaunchKernel(f, gx, gy, gz, bx, by, bz, sh, s, NULL, config);
  -- * \endcode
  -- *
  -- * The \p extra parameter exists to allow ::cuLaunchKernel to take
  -- * additional less commonly used arguments.  \p extra specifies a list of
  -- * names of extra settings and their corresponding values.  Each extra
  -- * setting name is immediately followed by the corresponding value.  The
  -- * list must be terminated with either NULL or ::CU_LAUNCH_PARAM_END.
  -- *
  -- * - ::CU_LAUNCH_PARAM_END, which indicates the end of the \p extra
  -- *   array;
  -- * - ::CU_LAUNCH_PARAM_BUFFER_POINTER, which specifies that the next
  -- *   value in \p extra will be a pointer to a buffer containing all
  -- *   the kernel parameters for launching kernel \p f;
  -- * - ::CU_LAUNCH_PARAM_BUFFER_SIZE, which specifies that the next
  -- *   value in \p extra will be a pointer to a size_t containing the
  -- *   size of the buffer specified with ::CU_LAUNCH_PARAM_BUFFER_POINTER;
  -- *
  -- * The error ::CUDA_ERROR_INVALID_VALUE will be returned if kernel
  -- * parameters are specified with both \p kernelParams and \p extra
  -- * (i.e. both \p kernelParams and \p extra are non-NULL).
  -- *
  -- * Calling ::cuLaunchKernel() sets persistent function state that is
  -- * the same as function state set through the following deprecated APIs:
  -- *  ::cuFuncSetBlockShape(),
  -- *  ::cuFuncSetSharedSize(),
  -- *  ::cuParamSetSize(),
  -- *  ::cuParamSeti(),
  -- *  ::cuParamSetf(),
  -- *  ::cuParamSetv().
  -- *
  -- * When the kernel \p f is launched via ::cuLaunchKernel(), the previous
  -- * block shape, shared size and parameter info associated with \p f
  -- * is overwritten.
  -- *
  -- * Note that to use ::cuLaunchKernel(), the kernel \p f must either have
  -- * been compiled with toolchain version 3.2 or later so that it will
  -- * contain kernel parameter information, or have no kernel parameters.
  -- * If either of these conditions is not met, then ::cuLaunchKernel() will
  -- * return ::CUDA_ERROR_INVALID_IMAGE.
  -- *
  -- * \param f              - Kernel to launch
  -- * \param gridDimX       - Width of grid in blocks
  -- * \param gridDimY       - Height of grid in blocks
  -- * \param gridDimZ       - Depth of grid in blocks
  -- * \param blockDimX      - X dimension of each thread block
  -- * \param blockDimY      - Y dimension of each thread block
  -- * \param blockDimZ      - Z dimension of each thread block
  -- * \param sharedMemBytes - Dynamic shared-memory size per thread block in bytes
  -- * \param hStream        - Stream identifier
  -- * \param kernelParams   - Array of pointers to kernel parameters
  -- * \param extra          - Extra options
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_HANDLE,
  -- * ::CUDA_ERROR_INVALID_IMAGE,
  -- * ::CUDA_ERROR_INVALID_VALUE,
  -- * ::CUDA_ERROR_LAUNCH_FAILED,
  -- * ::CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES,
  -- * ::CUDA_ERROR_LAUNCH_TIMEOUT,
  -- * ::CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING,
  -- * ::CUDA_ERROR_SHARED_OBJECT_INIT_FAILED
  -- * \note_null_stream
  -- * \notefnerr
  -- *
  -- * \sa ::cuCtxGetCacheConfig,
  -- * ::cuCtxSetCacheConfig,
  -- * ::cuFuncSetCacheConfig,
  -- * ::cuFuncGetAttribute
  --  

   function cuLaunchKernel
     (f : CUfunction;
      gridDimX : unsigned;
      gridDimY : unsigned;
      gridDimZ : unsigned;
      blockDimX : unsigned;
      blockDimY : unsigned;
      blockDimZ : unsigned;
      sharedMemBytes : unsigned;
      hStream : CUstream;
      kernelParams : System.Address;
      extra : System.Address) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:8947
   pragma Import (C, cuLaunchKernel, "cuLaunchKernel");

  --* @}  
  -- END CUDA_EXEC  
  --*
  -- * \defgroup CUDA_EXEC_DEPRECATED Execution Control [DEPRECATED]
  -- *
  -- * ___MANBRIEF___ deprecated execution control functions of the low-level CUDA
  -- * driver API (___CURRENT_FILE___) ___ENDMANBRIEF___
  -- *
  -- * This section describes the deprecated execution control functions of the
  -- * low-level CUDA driver application programming interface.
  -- *
  -- * @{
  --  

  --*
  -- * \brief Sets the block-dimensions for the function
  -- *
  -- * \deprecated
  -- *
  -- * Specifies the \p x, \p y, and \p z dimensions of the thread blocks that are
  -- * created when the kernel given by \p hfunc is launched.
  -- *
  -- * \param hfunc - Kernel to specify dimensions of
  -- * \param x     - X dimension
  -- * \param y     - Y dimension
  -- * \param z     - Z dimension
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_HANDLE,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- * \notefnerr
  -- *
  -- * \sa ::cuFuncSetSharedSize,
  -- * ::cuFuncSetCacheConfig,
  -- * ::cuFuncGetAttribute,
  -- * ::cuParamSetSize,
  -- * ::cuParamSeti,
  -- * ::cuParamSetf,
  -- * ::cuParamSetv,
  -- * ::cuLaunch,
  -- * ::cuLaunchGrid,
  -- * ::cuLaunchGridAsync,
  -- * ::cuLaunchKernel
  --  

   function cuFuncSetBlockShape
     (hfunc : CUfunction;
      x : int;
      y : int;
      z : int) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:9008
   pragma Import (C, cuFuncSetBlockShape, "cuFuncSetBlockShape");

  --*
  -- * \brief Sets the dynamic shared-memory size for the function
  -- *
  -- * \deprecated
  -- *
  -- * Sets through \p bytes the amount of dynamic shared memory that will be
  -- * available to each thread block when the kernel given by \p hfunc is launched.
  -- *
  -- * \param hfunc - Kernel to specify dynamic shared-memory size for
  -- * \param bytes - Dynamic shared-memory size per thread in bytes
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_HANDLE,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- * \notefnerr
  -- *
  -- * \sa ::cuFuncSetBlockShape,
  -- * ::cuFuncSetCacheConfig,
  -- * ::cuFuncGetAttribute,
  -- * ::cuParamSetSize,
  -- * ::cuParamSeti,
  -- * ::cuParamSetf,
  -- * ::cuParamSetv,
  -- * ::cuLaunch,
  -- * ::cuLaunchGrid,
  -- * ::cuLaunchGridAsync,
  -- * ::cuLaunchKernel
  --  

   function cuFuncSetSharedSize (hfunc : CUfunction; bytes : unsigned) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:9042
   pragma Import (C, cuFuncSetSharedSize, "cuFuncSetSharedSize");

  --*
  -- * \brief Sets the parameter size for the function
  -- *
  -- * \deprecated
  -- *
  -- * Sets through \p numbytes the total size in bytes needed by the function
  -- * parameters of the kernel corresponding to \p hfunc.
  -- *
  -- * \param hfunc    - Kernel to set parameter size for
  -- * \param numbytes - Size of parameter list in bytes
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- * \notefnerr
  -- *
  -- * \sa ::cuFuncSetBlockShape,
  -- * ::cuFuncSetSharedSize,
  -- * ::cuFuncGetAttribute,
  -- * ::cuParamSetf,
  -- * ::cuParamSeti,
  -- * ::cuParamSetv,
  -- * ::cuLaunch,
  -- * ::cuLaunchGrid,
  -- * ::cuLaunchGridAsync,
  -- * ::cuLaunchKernel
  --  

   function cuParamSetSize (hfunc : CUfunction; numbytes : unsigned) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:9074
   pragma Import (C, cuParamSetSize, "cuParamSetSize");

  --*
  -- * \brief Adds an integer parameter to the function's argument list
  -- *
  -- * \deprecated
  -- *
  -- * Sets an integer parameter that will be specified the next time the
  -- * kernel corresponding to \p hfunc will be invoked. \p offset is a byte offset.
  -- *
  -- * \param hfunc  - Kernel to add parameter to
  -- * \param offset - Offset to add parameter to argument list
  -- * \param value  - Value of parameter
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- * \notefnerr
  -- *
  -- * \sa ::cuFuncSetBlockShape,
  -- * ::cuFuncSetSharedSize,
  -- * ::cuFuncGetAttribute,
  -- * ::cuParamSetSize,
  -- * ::cuParamSetf,
  -- * ::cuParamSetv,
  -- * ::cuLaunch,
  -- * ::cuLaunchGrid,
  -- * ::cuLaunchGridAsync,
  -- * ::cuLaunchKernel
  --  

   function cuParamSeti
     (hfunc : CUfunction;
      offset : int;
      value : unsigned) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:9107
   pragma Import (C, cuParamSeti, "cuParamSeti");

  --*
  -- * \brief Adds a floating-point parameter to the function's argument list
  -- *
  -- * \deprecated
  -- *
  -- * Sets a floating-point parameter that will be specified the next time the
  -- * kernel corresponding to \p hfunc will be invoked. \p offset is a byte offset.
  -- *
  -- * \param hfunc  - Kernel to add parameter to
  -- * \param offset - Offset to add parameter to argument list
  -- * \param value  - Value of parameter
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- * \notefnerr
  -- *
  -- * \sa ::cuFuncSetBlockShape,
  -- * ::cuFuncSetSharedSize,
  -- * ::cuFuncGetAttribute,
  -- * ::cuParamSetSize,
  -- * ::cuParamSeti,
  -- * ::cuParamSetv,
  -- * ::cuLaunch,
  -- * ::cuLaunchGrid,
  -- * ::cuLaunchGridAsync,
  -- * ::cuLaunchKernel
  --  

   function cuParamSetf
     (hfunc : CUfunction;
      offset : int;
      value : float) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:9140
   pragma Import (C, cuParamSetf, "cuParamSetf");

  --*
  -- * \brief Adds arbitrary data to the function's argument list
  -- *
  -- * \deprecated
  -- *
  -- * Copies an arbitrary amount of data (specified in \p numbytes) from \p ptr
  -- * into the parameter space of the kernel corresponding to \p hfunc. \p offset
  -- * is a byte offset.
  -- *
  -- * \param hfunc    - Kernel to add data to
  -- * \param offset   - Offset to add data to argument list
  -- * \param ptr      - Pointer to arbitrary data
  -- * \param numbytes - Size of data to copy in bytes
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- * \notefnerr
  -- *
  -- * \sa ::cuFuncSetBlockShape,
  -- * ::cuFuncSetSharedSize,
  -- * ::cuFuncGetAttribute,
  -- * ::cuParamSetSize,
  -- * ::cuParamSetf,
  -- * ::cuParamSeti,
  -- * ::cuLaunch,
  -- * ::cuLaunchGrid,
  -- * ::cuLaunchGridAsync,
  -- * ::cuLaunchKernel
  --  

   function cuParamSetv
     (hfunc : CUfunction;
      offset : int;
      ptr : System.Address;
      numbytes : unsigned) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:9175
   pragma Import (C, cuParamSetv, "cuParamSetv");

  --*
  -- * \brief Launches a CUDA function
  -- *
  -- * \deprecated
  -- *
  -- * Invokes the kernel \p f on a 1 x 1 x 1 grid of blocks. The block
  -- * contains the number of threads specified by a previous call to
  -- * ::cuFuncSetBlockShape().
  -- *
  -- * \param f - Kernel to launch
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE,
  -- * ::CUDA_ERROR_LAUNCH_FAILED,
  -- * ::CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES,
  -- * ::CUDA_ERROR_LAUNCH_TIMEOUT,
  -- * ::CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING,
  -- * ::CUDA_ERROR_SHARED_OBJECT_INIT_FAILED
  -- * \notefnerr
  -- *
  -- * \sa ::cuFuncSetBlockShape,
  -- * ::cuFuncSetSharedSize,
  -- * ::cuFuncGetAttribute,
  -- * ::cuParamSetSize,
  -- * ::cuParamSetf,
  -- * ::cuParamSeti,
  -- * ::cuParamSetv,
  -- * ::cuLaunchGrid,
  -- * ::cuLaunchGridAsync,
  -- * ::cuLaunchKernel
  --  

   function cuLaunch (f : CUfunction) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:9212
   pragma Import (C, cuLaunch, "cuLaunch");

  --*
  -- * \brief Launches a CUDA function
  -- *
  -- * \deprecated
  -- *
  -- * Invokes the kernel \p f on a \p grid_width x \p grid_height grid of
  -- * blocks. Each block contains the number of threads specified by a previous
  -- * call to ::cuFuncSetBlockShape().
  -- *
  -- * \param f           - Kernel to launch
  -- * \param grid_width  - Width of grid in blocks
  -- * \param grid_height - Height of grid in blocks
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE,
  -- * ::CUDA_ERROR_LAUNCH_FAILED,
  -- * ::CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES,
  -- * ::CUDA_ERROR_LAUNCH_TIMEOUT,
  -- * ::CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING,
  -- * ::CUDA_ERROR_SHARED_OBJECT_INIT_FAILED
  -- * \notefnerr
  -- *
  -- * \sa ::cuFuncSetBlockShape,
  -- * ::cuFuncSetSharedSize,
  -- * ::cuFuncGetAttribute,
  -- * ::cuParamSetSize,
  -- * ::cuParamSetf,
  -- * ::cuParamSeti,
  -- * ::cuParamSetv,
  -- * ::cuLaunch,
  -- * ::cuLaunchGridAsync,
  -- * ::cuLaunchKernel
  --  

   function cuLaunchGrid
     (f : CUfunction;
      grid_width : int;
      grid_height : int) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:9251
   pragma Import (C, cuLaunchGrid, "cuLaunchGrid");

  --*
  -- * \brief Launches a CUDA function
  -- *
  -- * \deprecated
  -- *
  -- * Invokes the kernel \p f on a \p grid_width x \p grid_height grid of
  -- * blocks. Each block contains the number of threads specified by a previous
  -- * call to ::cuFuncSetBlockShape().
  -- *
  -- * \param f           - Kernel to launch
  -- * \param grid_width  - Width of grid in blocks
  -- * \param grid_height - Height of grid in blocks
  -- * \param hStream     - Stream identifier
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_HANDLE,
  -- * ::CUDA_ERROR_INVALID_VALUE,
  -- * ::CUDA_ERROR_LAUNCH_FAILED,
  -- * ::CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES,
  -- * ::CUDA_ERROR_LAUNCH_TIMEOUT,
  -- * ::CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING,
  -- * ::CUDA_ERROR_SHARED_OBJECT_INIT_FAILED
  -- *
  -- * \note In certain cases where cubins are created with no ABI (i.e., using \p ptxas \p --abi-compile \p no), 
  -- *       this function may serialize kernel launches. In order to force the CUDA driver to retain 
  -- *		 asynchronous behavior, set the ::CU_CTX_LMEM_RESIZE_TO_MAX flag during context creation (see ::cuCtxCreate).
  -- *       
  -- * \note_null_stream
  -- * \notefnerr
  -- *
  -- * \sa ::cuFuncSetBlockShape,
  -- * ::cuFuncSetSharedSize,
  -- * ::cuFuncGetAttribute,
  -- * ::cuParamSetSize,
  -- * ::cuParamSetf,
  -- * ::cuParamSeti,
  -- * ::cuParamSetv,
  -- * ::cuLaunch,
  -- * ::cuLaunchGrid,
  -- * ::cuLaunchKernel
  --  

   function cuLaunchGridAsync
     (f : CUfunction;
      grid_width : int;
      grid_height : int;
      hStream : CUstream) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:9298
   pragma Import (C, cuLaunchGridAsync, "cuLaunchGridAsync");

  --*
  -- * \brief Adds a texture-reference to the function's argument list
  -- *
  -- * \deprecated
  -- *
  -- * Makes the CUDA array or linear memory bound to the texture reference
  -- * \p hTexRef available to a device program as a texture. In this version of
  -- * CUDA, the texture-reference must be obtained via ::cuModuleGetTexRef() and
  -- * the \p texunit parameter must be set to ::CU_PARAM_TR_DEFAULT.
  -- *
  -- * \param hfunc   - Kernel to add texture-reference to
  -- * \param texunit - Texture unit (must be ::CU_PARAM_TR_DEFAULT)
  -- * \param hTexRef - Texture-reference to add to argument list
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- * \notefnerr
  --  

   function cuParamSetTexRef
     (hfunc : CUfunction;
      texunit : int;
      hTexRef : CUtexref) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:9323
   pragma Import (C, cuParamSetTexRef, "cuParamSetTexRef");

  --* @}  
  -- END CUDA_EXEC_DEPRECATED  
  --*
  -- * \defgroup CUDA_OCCUPANCY Occupancy
  -- *
  -- * ___MANBRIEF___ occupancy calculation functions of the low-level CUDA driver
  -- * API (___CURRENT_FILE___) ___ENDMANBRIEF___
  -- *
  -- * This section describes the occupancy calculation functions of the low-level CUDA
  -- * driver application programming interface.
  -- *
  -- * @{
  --  

  --*
  -- * \brief Returns occupancy of a function
  -- *
  -- * Returns in \p *numBlocks the number of the maximum active blocks per
  -- * streaming multiprocessor.
  -- *
  -- * \param numBlocks       - Returned occupancy
  -- * \param func            - Kernel for which occupancy is calculated
  -- * \param blockSize       - Block size the kernel is intended to be launched with
  -- * \param dynamicSMemSize - Per-block dynamic shared memory usage intended, in bytes
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE,
  -- * ::CUDA_ERROR_UNKNOWN
  -- * \notefnerr
  -- *
  --  

   function cuOccupancyMaxActiveBlocksPerMultiprocessor
     (numBlocks : access int;
      func : CUfunction;
      blockSize : int;
      dynamicSMemSize : stddef_h.size_t) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:9361
   pragma Import (C, cuOccupancyMaxActiveBlocksPerMultiprocessor, "cuOccupancyMaxActiveBlocksPerMultiprocessor");

  --*
  -- * \brief Returns occupancy of a function
  -- *
  -- * Returns in \p *numBlocks the number of the maximum active blocks per
  -- * streaming multiprocessor.
  -- *
  -- * The \p Flags parameter controls how special cases are handled. The
  -- * valid flags are:
  -- *
  -- * - ::CU_OCCUPANCY_DEFAULT, which maintains the default behavior as
  -- *   ::cuOccupancyMaxActiveBlocksPerMultiprocessor;
  -- *
  -- * - ::CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE, which suppresses the
  -- *   default behavior on platform where global caching affects
  -- *   occupancy. On such platforms, if caching is enabled, but
  -- *   per-block SM resource usage would result in zero occupancy, the
  -- *   occupancy calculator will calculate the occupancy as if caching
  -- *   is disabled. Setting ::CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE makes
  -- *   the occupancy calculator to return 0 in such cases. More information
  -- *   can be found about this feature in the "Unified L1/Texture Cache"
  -- *   section of the Maxwell tuning guide.
  -- *
  -- * \param numBlocks       - Returned occupancy
  -- * \param func            - Kernel for which occupancy is calculated
  -- * \param blockSize       - Block size the kernel is intended to be launched with
  -- * \param dynamicSMemSize - Per-block dynamic shared memory usage intended, in bytes
  -- * \param flags           - Requested behavior for the occupancy calculator
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE,
  -- * ::CUDA_ERROR_UNKNOWN
  -- * \notefnerr
  -- *
  --  

   function cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
     (numBlocks : access int;
      func : CUfunction;
      blockSize : int;
      dynamicSMemSize : stddef_h.size_t;
      flags : unsigned) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:9401
   pragma Import (C, cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags, "cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags");

  --*
  -- * \brief Suggest a launch configuration with reasonable occupancy
  -- *
  -- * Returns in \p *blockSize a reasonable block size that can achieve
  -- * the maximum occupancy (or, the maximum number of active warps with
  -- * the fewest blocks per multiprocessor), and in \p *minGridSize the
  -- * minimum grid size to achieve the maximum occupancy.
  -- *
  -- * If \p blockSizeLimit is 0, the configurator will use the maximum
  -- * block size permitted by the device / function instead.
  -- *
  -- * If per-block dynamic shared memory allocation is not needed, the
  -- * user should leave both \p blockSizeToDynamicSMemSize and \p
  -- * dynamicSMemSize as 0.
  -- *
  -- * If per-block dynamic shared memory allocation is needed, then if
  -- * the dynamic shared memory size is constant regardless of block
  -- * size, the size should be passed through \p dynamicSMemSize, and \p
  -- * blockSizeToDynamicSMemSize should be NULL.
  -- *
  -- * Otherwise, if the per-block dynamic shared memory size varies with
  -- * different block sizes, the user needs to provide a unary function
  -- * through \p blockSizeToDynamicSMemSize that computes the dynamic
  -- * shared memory needed by \p func for any given block size. \p
  -- * dynamicSMemSize is ignored. An example signature is:
  -- *
  -- * \code
  -- *    // Take block size, returns dynamic shared memory needed
  -- *    size_t blockToSmem(int blockSize);
  -- * \endcode
  -- *
  -- * \param minGridSize - Returned minimum grid size needed to achieve the maximum occupancy
  -- * \param blockSize   - Returned maximum block size that can achieve the maximum occupancy
  -- * \param func        - Kernel for which launch configuration is calculated
  -- * \param blockSizeToDynamicSMemSize - A function that calculates how much per-block dynamic shared memory \p func uses based on the block size
  -- * \param dynamicSMemSize - Dynamic shared memory usage intended, in bytes
  -- * \param blockSizeLimit  - The maximum block size \p func is designed to handle
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE,
  -- * ::CUDA_ERROR_UNKNOWN
  -- * \notefnerr
  -- *
  --  

   function cuOccupancyMaxPotentialBlockSize
     (minGridSize : access int;
      blockSize : access int;
      func : CUfunction;
      blockSizeToDynamicSMemSize : CUoccupancyB2DSize;
      dynamicSMemSize : stddef_h.size_t;
      blockSizeLimit : int) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:9451
   pragma Import (C, cuOccupancyMaxPotentialBlockSize, "cuOccupancyMaxPotentialBlockSize");

  --*
  -- * \brief Suggest a launch configuration with reasonable occupancy
  -- *
  -- * An extended version of ::cuOccupancyMaxPotentialBlockSize. In
  -- * addition to arguments passed to ::cuOccupancyMaxPotentialBlockSize,
  -- * ::cuOccupancyMaxPotentialBlockSizeWithFlags also takes a \p Flags
  -- * parameter.
  -- *
  -- * The \p Flags parameter controls how special cases are handled. The
  -- * valid flags are:
  -- *
  -- * - ::CU_OCCUPANCY_DEFAULT, which maintains the default behavior as
  -- *   ::cuOccupancyMaxPotentialBlockSize;
  -- *
  -- * - ::CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE, which suppresses the
  -- *   default behavior on platform where global caching affects
  -- *   occupancy. On such platforms, the launch configurations that
  -- *   produces maximal occupancy might not support global
  -- *   caching. Setting ::CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE
  -- *   guarantees that the the produced launch configuration is global
  -- *   caching compatible at a potential cost of occupancy. More information
  -- *   can be found about this feature in the "Unified L1/Texture Cache"
  -- *   section of the Maxwell tuning guide.
  -- *
  -- * \param minGridSize - Returned minimum grid size needed to achieve the maximum occupancy
  -- * \param blockSize   - Returned maximum block size that can achieve the maximum occupancy
  -- * \param func        - Kernel for which launch configuration is calculated
  -- * \param blockSizeToDynamicSMemSize - A function that calculates how much per-block dynamic shared memory \p func uses based on the block size
  -- * \param dynamicSMemSize - Dynamic shared memory usage intended, in bytes
  -- * \param blockSizeLimit  - The maximum block size \p func is designed to handle
  -- * \param flags       - Options
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE,
  -- * ::CUDA_ERROR_UNKNOWN
  -- * \notefnerr
  -- *
  --  

   function cuOccupancyMaxPotentialBlockSizeWithFlags
     (minGridSize : access int;
      blockSize : access int;
      func : CUfunction;
      blockSizeToDynamicSMemSize : CUoccupancyB2DSize;
      dynamicSMemSize : stddef_h.size_t;
      blockSizeLimit : int;
      flags : unsigned) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:9495
   pragma Import (C, cuOccupancyMaxPotentialBlockSizeWithFlags, "cuOccupancyMaxPotentialBlockSizeWithFlags");

  --* @}  
  -- END CUDA_OCCUPANCY  
  --*
  -- * \defgroup CUDA_TEXREF Texture Reference Management
  -- *
  -- * ___MANBRIEF___ texture reference management functions of the low-level CUDA
  -- * driver API (___CURRENT_FILE___) ___ENDMANBRIEF___
  -- *
  -- * This section describes the texture reference management functions of the
  -- * low-level CUDA driver application programming interface.
  -- *
  -- * @{
  --  

  --*
  -- * \brief Binds an array as a texture reference
  -- *
  -- * Binds the CUDA array \p hArray to the texture reference \p hTexRef. Any
  -- * previous address or CUDA array state associated with the texture reference
  -- * is superseded by this function. \p Flags must be set to
  -- * ::CU_TRSA_OVERRIDE_FORMAT. Any CUDA array previously bound to \p hTexRef is
  -- * unbound.
  -- *
  -- * \param hTexRef - Texture reference to bind
  -- * \param hArray  - Array to bind
  -- * \param Flags   - Options (must be ::CU_TRSA_OVERRIDE_FORMAT)
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- *
  -- * \sa ::cuTexRefSetAddress,
  -- * ::cuTexRefSetAddress2D, ::cuTexRefSetAddressMode,
  -- * ::cuTexRefSetFilterMode, ::cuTexRefSetFlags, ::cuTexRefSetFormat,
  -- * ::cuTexRefGetAddress, ::cuTexRefGetAddressMode, ::cuTexRefGetArray,
  -- * ::cuTexRefGetFilterMode, ::cuTexRefGetFlags, ::cuTexRefGetFormat
  --  

   function cuTexRefSetArray
     (hTexRef : CUtexref;
      hArray : CUarray;
      Flags : unsigned) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:9538
   pragma Import (C, cuTexRefSetArray, "cuTexRefSetArray");

  --*
  -- * \brief Binds a mipmapped array to a texture reference
  -- *
  -- * Binds the CUDA mipmapped array \p hMipmappedArray to the texture reference \p hTexRef.
  -- * Any previous address or CUDA array state associated with the texture reference
  -- * is superseded by this function. \p Flags must be set to ::CU_TRSA_OVERRIDE_FORMAT. 
  -- * Any CUDA array previously bound to \p hTexRef is unbound.
  -- *
  -- * \param hTexRef         - Texture reference to bind
  -- * \param hMipmappedArray - Mipmapped array to bind
  -- * \param Flags           - Options (must be ::CU_TRSA_OVERRIDE_FORMAT)
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- *
  -- * \sa ::cuTexRefSetAddress,
  -- * ::cuTexRefSetAddress2D, ::cuTexRefSetAddressMode,
  -- * ::cuTexRefSetFilterMode, ::cuTexRefSetFlags, ::cuTexRefSetFormat,
  -- * ::cuTexRefGetAddress, ::cuTexRefGetAddressMode, ::cuTexRefGetArray,
  -- * ::cuTexRefGetFilterMode, ::cuTexRefGetFlags, ::cuTexRefGetFormat
  --  

   function cuTexRefSetMipmappedArray
     (hTexRef : CUtexref;
      hMipmappedArray : CUmipmappedArray;
      Flags : unsigned) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:9565
   pragma Import (C, cuTexRefSetMipmappedArray, "cuTexRefSetMipmappedArray");

  --*
  -- * \brief Binds an address as a texture reference
  -- *
  -- * Binds a linear address range to the texture reference \p hTexRef. Any
  -- * previous address or CUDA array state associated with the texture reference
  -- * is superseded by this function. Any memory previously bound to \p hTexRef
  -- * is unbound.
  -- *
  -- * Since the hardware enforces an alignment requirement on texture base
  -- * addresses, ::cuTexRefSetAddress() passes back a byte offset in
  -- * \p *ByteOffset that must be applied to texture fetches in order to read from
  -- * the desired memory. This offset must be divided by the texel size and
  -- * passed to kernels that read from the texture so they can be applied to the
  -- * ::tex1Dfetch() function.
  -- *
  -- * If the device memory pointer was returned from ::cuMemAlloc(), the offset
  -- * is guaranteed to be 0 and NULL may be passed as the \p ByteOffset parameter.
  -- *
  -- * The total number of elements (or texels) in the linear address range
  -- * cannot exceed ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH.
  -- * The number of elements is computed as (\p bytes / bytesPerElement),
  -- * where bytesPerElement is determined from the data format and number of 
  -- * components set using ::cuTexRefSetFormat().
  -- *
  -- * \param ByteOffset - Returned byte offset
  -- * \param hTexRef    - Texture reference to bind
  -- * \param dptr       - Device pointer to bind
  -- * \param bytes      - Size of memory to bind in bytes
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- *
  -- * \sa ::cuTexRefSetAddress2D, ::cuTexRefSetAddressMode, ::cuTexRefSetArray,
  -- * ::cuTexRefSetFilterMode, ::cuTexRefSetFlags, ::cuTexRefSetFormat,
  -- * ::cuTexRefGetAddress, ::cuTexRefGetAddressMode, ::cuTexRefGetArray,
  -- * ::cuTexRefGetFilterMode, ::cuTexRefGetFlags, ::cuTexRefGetFormat
  --  

   function cuTexRefSetAddress_v2
     (ByteOffset : access stddef_h.size_t;
      hTexRef : CUtexref;
      dptr : CUdeviceptr;
      bytes : stddef_h.size_t) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:9609
   pragma Import (C, cuTexRefSetAddress_v2, "cuTexRefSetAddress_v2");

  --*
  -- * \brief Binds an address as a 2D texture reference
  -- *
  -- * Binds a linear address range to the texture reference \p hTexRef. Any
  -- * previous address or CUDA array state associated with the texture reference
  -- * is superseded by this function. Any memory previously bound to \p hTexRef
  -- * is unbound.
  -- *
  -- * Using a ::tex2D() function inside a kernel requires a call to either
  -- * ::cuTexRefSetArray() to bind the corresponding texture reference to an
  -- * array, or ::cuTexRefSetAddress2D() to bind the texture reference to linear
  -- * memory.
  -- *
  -- * Function calls to ::cuTexRefSetFormat() cannot follow calls to
  -- * ::cuTexRefSetAddress2D() for the same texture reference.
  -- *
  -- * It is required that \p dptr be aligned to the appropriate hardware-specific
  -- * texture alignment. You can query this value using the device attribute
  -- * ::CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT. If an unaligned \p dptr is
  -- * supplied, ::CUDA_ERROR_INVALID_VALUE is returned.
  -- *
  -- * \p Pitch has to be aligned to the hardware-specific texture pitch alignment.
  -- * This value can be queried using the device attribute 
  -- * ::CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT. If an unaligned \p Pitch is 
  -- * supplied, ::CUDA_ERROR_INVALID_VALUE is returned.
  -- *
  -- * Width and Height, which are specified in elements (or texels), cannot exceed
  -- * ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH and
  -- * ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT respectively.
  -- * \p Pitch, which is specified in bytes, cannot exceed 
  -- * ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH.
  -- *
  -- * \param hTexRef - Texture reference to bind
  -- * \param desc    - Descriptor of CUDA array
  -- * \param dptr    - Device pointer to bind
  -- * \param Pitch   - Line pitch in bytes
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- *
  -- * \sa ::cuTexRefSetAddress,
  -- * ::cuTexRefSetAddressMode, ::cuTexRefSetArray,
  -- * ::cuTexRefSetFilterMode, ::cuTexRefSetFlags, ::cuTexRefSetFormat,
  -- * ::cuTexRefGetAddress, ::cuTexRefGetAddressMode, ::cuTexRefGetArray,
  -- * ::cuTexRefGetFilterMode, ::cuTexRefGetFlags, ::cuTexRefGetFormat
  --  

   function cuTexRefSetAddress2D_v3
     (hTexRef : CUtexref;
      desc : access constant CUDA_ARRAY_DESCRIPTOR;
      dptr : CUdeviceptr;
      Pitch : stddef_h.size_t) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:9661
   pragma Import (C, cuTexRefSetAddress2D_v3, "cuTexRefSetAddress2D_v3");

  --*
  -- * \brief Sets the format for a texture reference
  -- *
  -- * Specifies the format of the data to be read by the texture reference
  -- * \p hTexRef. \p fmt and \p NumPackedComponents are exactly analogous to the
  -- * ::Format and ::NumChannels members of the ::CUDA_ARRAY_DESCRIPTOR structure:
  -- * They specify the format of each component and the number of components per
  -- * array element.
  -- *
  -- * \param hTexRef             - Texture reference
  -- * \param fmt                 - Format to set
  -- * \param NumPackedComponents - Number of components per array element
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- *
  -- * \sa ::cuTexRefSetAddress,
  -- * ::cuTexRefSetAddress2D, ::cuTexRefSetAddressMode, ::cuTexRefSetArray,
  -- * ::cuTexRefSetFilterMode, ::cuTexRefSetFlags,
  -- * ::cuTexRefGetAddress, ::cuTexRefGetAddressMode, ::cuTexRefGetArray,
  -- * ::cuTexRefGetFilterMode, ::cuTexRefGetFlags, ::cuTexRefGetFormat
  --  

   function cuTexRefSetFormat
     (hTexRef : CUtexref;
      fmt : CUarray_format;
      NumPackedComponents : int) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:9690
   pragma Import (C, cuTexRefSetFormat, "cuTexRefSetFormat");

  --*
  -- * \brief Sets the addressing mode for a texture reference
  -- *
  -- * Specifies the addressing mode \p am for the given dimension \p dim of the
  -- * texture reference \p hTexRef. If \p dim is zero, the addressing mode is
  -- * applied to the first parameter of the functions used to fetch from the
  -- * texture; if \p dim is 1, the second, and so on. ::CUaddress_mode is defined
  -- * as:
  -- * \code
  --   typedef enum CUaddress_mode_enum {
  --      CU_TR_ADDRESS_MODE_WRAP = 0,
  --      CU_TR_ADDRESS_MODE_CLAMP = 1,
  --      CU_TR_ADDRESS_MODE_MIRROR = 2,
  --      CU_TR_ADDRESS_MODE_BORDER = 3
  --   } CUaddress_mode;
  -- * \endcode
  -- *
  -- * Note that this call has no effect if \p hTexRef is bound to linear memory.
  -- * Also, if the flag, ::CU_TRSF_NORMALIZED_COORDINATES, is not set, the only 
  -- * supported address mode is ::CU_TR_ADDRESS_MODE_CLAMP.
  -- *
  -- * \param hTexRef - Texture reference
  -- * \param dim     - Dimension
  -- * \param am      - Addressing mode to set
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- *
  -- * \sa ::cuTexRefSetAddress,
  -- * ::cuTexRefSetAddress2D, ::cuTexRefSetArray,
  -- * ::cuTexRefSetFilterMode, ::cuTexRefSetFlags, ::cuTexRefSetFormat,
  -- * ::cuTexRefGetAddress, ::cuTexRefGetAddressMode, ::cuTexRefGetArray,
  -- * ::cuTexRefGetFilterMode, ::cuTexRefGetFlags, ::cuTexRefGetFormat
  --  

   function cuTexRefSetAddressMode
     (hTexRef : CUtexref;
      dim : int;
      am : CUaddress_mode) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:9730
   pragma Import (C, cuTexRefSetAddressMode, "cuTexRefSetAddressMode");

  --*
  -- * \brief Sets the filtering mode for a texture reference
  -- *
  -- * Specifies the filtering mode \p fm to be used when reading memory through
  -- * the texture reference \p hTexRef. ::CUfilter_mode_enum is defined as:
  -- *
  -- * \code
  --   typedef enum CUfilter_mode_enum {
  --      CU_TR_FILTER_MODE_POINT = 0,
  --      CU_TR_FILTER_MODE_LINEAR = 1
  --   } CUfilter_mode;
  -- * \endcode
  -- *
  -- * Note that this call has no effect if \p hTexRef is bound to linear memory.
  -- *
  -- * \param hTexRef - Texture reference
  -- * \param fm      - Filtering mode to set
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- *
  -- * \sa ::cuTexRefSetAddress,
  -- * ::cuTexRefSetAddress2D, ::cuTexRefSetAddressMode, ::cuTexRefSetArray,
  -- * ::cuTexRefSetFlags, ::cuTexRefSetFormat,
  -- * ::cuTexRefGetAddress, ::cuTexRefGetAddressMode, ::cuTexRefGetArray,
  -- * ::cuTexRefGetFilterMode, ::cuTexRefGetFlags, ::cuTexRefGetFormat
  --  

   function cuTexRefSetFilterMode (hTexRef : CUtexref; fm : CUfilter_mode) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:9763
   pragma Import (C, cuTexRefSetFilterMode, "cuTexRefSetFilterMode");

  --*
  -- * \brief Sets the mipmap filtering mode for a texture reference
  -- *
  -- * Specifies the mipmap filtering mode \p fm to be used when reading memory through
  -- * the texture reference \p hTexRef. ::CUfilter_mode_enum is defined as:
  -- *
  -- * \code
  --   typedef enum CUfilter_mode_enum {
  --      CU_TR_FILTER_MODE_POINT = 0,
  --      CU_TR_FILTER_MODE_LINEAR = 1
  --   } CUfilter_mode;
  -- * \endcode
  -- *
  -- * Note that this call has no effect if \p hTexRef is not bound to a mipmapped array.
  -- *
  -- * \param hTexRef - Texture reference
  -- * \param fm      - Filtering mode to set
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- *
  -- * \sa ::cuTexRefSetAddress,
  -- * ::cuTexRefSetAddress2D, ::cuTexRefSetAddressMode, ::cuTexRefSetArray,
  -- * ::cuTexRefSetFlags, ::cuTexRefSetFormat,
  -- * ::cuTexRefGetAddress, ::cuTexRefGetAddressMode, ::cuTexRefGetArray,
  -- * ::cuTexRefGetFilterMode, ::cuTexRefGetFlags, ::cuTexRefGetFormat
  --  

   function cuTexRefSetMipmapFilterMode (hTexRef : CUtexref; fm : CUfilter_mode) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:9796
   pragma Import (C, cuTexRefSetMipmapFilterMode, "cuTexRefSetMipmapFilterMode");

  --*
  -- * \brief Sets the mipmap level bias for a texture reference
  -- *
  -- * Specifies the mipmap level bias \p bias to be added to the specified mipmap level when 
  -- * reading memory through the texture reference \p hTexRef.
  -- *
  -- * Note that this call has no effect if \p hTexRef is not bound to a mipmapped array.
  -- *
  -- * \param hTexRef - Texture reference
  -- * \param bias    - Mipmap level bias
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- *
  -- * \sa ::cuTexRefSetAddress,
  -- * ::cuTexRefSetAddress2D, ::cuTexRefSetAddressMode, ::cuTexRefSetArray,
  -- * ::cuTexRefSetFlags, ::cuTexRefSetFormat,
  -- * ::cuTexRefGetAddress, ::cuTexRefGetAddressMode, ::cuTexRefGetArray,
  -- * ::cuTexRefGetFilterMode, ::cuTexRefGetFlags, ::cuTexRefGetFormat
  --  

   function cuTexRefSetMipmapLevelBias (hTexRef : CUtexref; bias : float) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:9822
   pragma Import (C, cuTexRefSetMipmapLevelBias, "cuTexRefSetMipmapLevelBias");

  --*
  -- * \brief Sets the mipmap min/max mipmap level clamps for a texture reference
  -- *
  -- * Specifies the min/max mipmap level clamps, \p minMipmapLevelClamp and \p maxMipmapLevelClamp
  -- * respectively, to be used when reading memory through the texture reference 
  -- * \p hTexRef.
  -- *
  -- * Note that this call has no effect if \p hTexRef is not bound to a mipmapped array.
  -- *
  -- * \param hTexRef        - Texture reference
  -- * \param minMipmapLevelClamp - Mipmap min level clamp
  -- * \param maxMipmapLevelClamp - Mipmap max level clamp
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- *
  -- * \sa ::cuTexRefSetAddress,
  -- * ::cuTexRefSetAddress2D, ::cuTexRefSetAddressMode, ::cuTexRefSetArray,
  -- * ::cuTexRefSetFlags, ::cuTexRefSetFormat,
  -- * ::cuTexRefGetAddress, ::cuTexRefGetAddressMode, ::cuTexRefGetArray,
  -- * ::cuTexRefGetFilterMode, ::cuTexRefGetFlags, ::cuTexRefGetFormat
  --  

   function cuTexRefSetMipmapLevelClamp
     (hTexRef : CUtexref;
      minMipmapLevelClamp : float;
      maxMipmapLevelClamp : float) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:9850
   pragma Import (C, cuTexRefSetMipmapLevelClamp, "cuTexRefSetMipmapLevelClamp");

  --*
  -- * \brief Sets the maximum anisotropy for a texture reference
  -- *
  -- * Specifies the maximum anisotropy \p maxAniso to be used when reading memory through
  -- * the texture reference \p hTexRef. 
  -- *
  -- * Note that this call has no effect if \p hTexRef is bound to linear memory.
  -- *
  -- * \param hTexRef  - Texture reference
  -- * \param maxAniso - Maximum anisotropy
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- *
  -- * \sa ::cuTexRefSetAddress,
  -- * ::cuTexRefSetAddress2D, ::cuTexRefSetAddressMode, ::cuTexRefSetArray,
  -- * ::cuTexRefSetFlags, ::cuTexRefSetFormat,
  -- * ::cuTexRefGetAddress, ::cuTexRefGetAddressMode, ::cuTexRefGetArray,
  -- * ::cuTexRefGetFilterMode, ::cuTexRefGetFlags, ::cuTexRefGetFormat
  --  

   function cuTexRefSetMaxAnisotropy (hTexRef : CUtexref; maxAniso : unsigned) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:9876
   pragma Import (C, cuTexRefSetMaxAnisotropy, "cuTexRefSetMaxAnisotropy");

  --*
  -- * \brief Sets the border color for a texture reference
  -- *
  -- * Specifies the value of the RGBA color via the \p pBorderColor to the texture reference
  -- * \p hTexRef. The color value supports only float type and holds color components in
  -- * the following sequence:
  -- * pBorderColor[0] holds 'R' component
  -- * pBorderColor[1] holds 'G' component
  -- * pBorderColor[2] holds 'B' component
  -- * pBorderColor[3] holds 'A' component
  -- *
  -- * Note that the color values can be set only when the Address mode is set to
  -- * CU_TR_ADDRESS_MODE_BORDER using ::cuTexRefSetAddressMode.
  -- * Applications using integer border color values have to "reinterpret_cast" their values to float.
  -- *
  -- * \param hTexRef       - Texture reference
  -- * \param pBorderColor  - RGBA color
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- *
  -- * \sa ::cuTexRefSetAddressMode,
  -- * ::cuTexRefGetAddressMode, ::cuTexRefGetBorderColor
  --  

   function cuTexRefSetBorderColor (hTexRef : CUtexref; pBorderColor : access float) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:9906
   pragma Import (C, cuTexRefSetBorderColor, "cuTexRefSetBorderColor");

  --*
  -- * \brief Sets the flags for a texture reference
  -- *
  -- * Specifies optional flags via \p Flags to specify the behavior of data
  -- * returned through the texture reference \p hTexRef. The valid flags are:
  -- *
  -- * - ::CU_TRSF_READ_AS_INTEGER, which suppresses the default behavior of
  -- *   having the texture promote integer data to floating point data in the
  -- *   range [0, 1]. Note that texture with 32-bit integer format
  -- *   would not be promoted, regardless of whether or not this
  -- *   flag is specified;
  -- * - ::CU_TRSF_NORMALIZED_COORDINATES, which suppresses the 
  -- *   default behavior of having the texture coordinates range
  -- *   from [0, Dim) where Dim is the width or height of the CUDA
  -- *   array. Instead, the texture coordinates [0, 1.0) reference
  -- *   the entire breadth of the array dimension;
  -- *
  -- * \param hTexRef - Texture reference
  -- * \param Flags   - Optional flags to set
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- *
  -- * \sa ::cuTexRefSetAddress,
  -- * ::cuTexRefSetAddress2D, ::cuTexRefSetAddressMode, ::cuTexRefSetArray,
  -- * ::cuTexRefSetFilterMode, ::cuTexRefSetFormat,
  -- * ::cuTexRefGetAddress, ::cuTexRefGetAddressMode, ::cuTexRefGetArray,
  -- * ::cuTexRefGetFilterMode, ::cuTexRefGetFlags, ::cuTexRefGetFormat
  --  

   function cuTexRefSetFlags (hTexRef : CUtexref; Flags : unsigned) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:9941
   pragma Import (C, cuTexRefSetFlags, "cuTexRefSetFlags");

  --*
  -- * \brief Gets the address associated with a texture reference
  -- *
  -- * Returns in \p *pdptr the base address bound to the texture reference
  -- * \p hTexRef, or returns ::CUDA_ERROR_INVALID_VALUE if the texture reference
  -- * is not bound to any device memory range.
  -- *
  -- * \param pdptr   - Returned device address
  -- * \param hTexRef - Texture reference
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- *
  -- * \sa ::cuTexRefSetAddress,
  -- * ::cuTexRefSetAddress2D, ::cuTexRefSetAddressMode, ::cuTexRefSetArray,
  -- * ::cuTexRefSetFilterMode, ::cuTexRefSetFlags, ::cuTexRefSetFormat,
  -- * ::cuTexRefGetAddressMode, ::cuTexRefGetArray,
  -- * ::cuTexRefGetFilterMode, ::cuTexRefGetFlags, ::cuTexRefGetFormat
  --  

   function cuTexRefGetAddress_v2 (pdptr : access CUdeviceptr; hTexRef : CUtexref) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:9967
   pragma Import (C, cuTexRefGetAddress_v2, "cuTexRefGetAddress_v2");

  --*
  -- * \brief Gets the array bound to a texture reference
  -- *
  -- * Returns in \p *phArray the CUDA array bound to the texture reference
  -- * \p hTexRef, or returns ::CUDA_ERROR_INVALID_VALUE if the texture reference
  -- * is not bound to any CUDA array.
  -- *
  -- * \param phArray - Returned array
  -- * \param hTexRef - Texture reference
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- *
  -- * \sa ::cuTexRefSetAddress,
  -- * ::cuTexRefSetAddress2D, ::cuTexRefSetAddressMode, ::cuTexRefSetArray,
  -- * ::cuTexRefSetFilterMode, ::cuTexRefSetFlags, ::cuTexRefSetFormat,
  -- * ::cuTexRefGetAddress, ::cuTexRefGetAddressMode,
  -- * ::cuTexRefGetFilterMode, ::cuTexRefGetFlags, ::cuTexRefGetFormat
  --  

   function cuTexRefGetArray (phArray : System.Address; hTexRef : CUtexref) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:9993
   pragma Import (C, cuTexRefGetArray, "cuTexRefGetArray");

  --*
  -- * \brief Gets the mipmapped array bound to a texture reference
  -- *
  -- * Returns in \p *phMipmappedArray the CUDA mipmapped array bound to the texture 
  -- * reference \p hTexRef, or returns ::CUDA_ERROR_INVALID_VALUE if the texture reference
  -- * is not bound to any CUDA mipmapped array.
  -- *
  -- * \param phMipmappedArray - Returned mipmapped array
  -- * \param hTexRef          - Texture reference
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- *
  -- * \sa ::cuTexRefSetAddress,
  -- * ::cuTexRefSetAddress2D, ::cuTexRefSetAddressMode, ::cuTexRefSetArray,
  -- * ::cuTexRefSetFilterMode, ::cuTexRefSetFlags, ::cuTexRefSetFormat,
  -- * ::cuTexRefGetAddress, ::cuTexRefGetAddressMode,
  -- * ::cuTexRefGetFilterMode, ::cuTexRefGetFlags, ::cuTexRefGetFormat
  --  

   function cuTexRefGetMipmappedArray (phMipmappedArray : System.Address; hTexRef : CUtexref) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:10018
   pragma Import (C, cuTexRefGetMipmappedArray, "cuTexRefGetMipmappedArray");

  --*
  -- * \brief Gets the addressing mode used by a texture reference
  -- *
  -- * Returns in \p *pam the addressing mode corresponding to the
  -- * dimension \p dim of the texture reference \p hTexRef. Currently, the only
  -- * valid value for \p dim are 0 and 1.
  -- *
  -- * \param pam     - Returned addressing mode
  -- * \param hTexRef - Texture reference
  -- * \param dim     - Dimension
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- *
  -- * \sa ::cuTexRefSetAddress,
  -- * ::cuTexRefSetAddress2D, ::cuTexRefSetAddressMode, ::cuTexRefSetArray,
  -- * ::cuTexRefSetFilterMode, ::cuTexRefSetFlags, ::cuTexRefSetFormat,
  -- * ::cuTexRefGetAddress, ::cuTexRefGetArray,
  -- * ::cuTexRefGetFilterMode, ::cuTexRefGetFlags, ::cuTexRefGetFormat
  --  

   function cuTexRefGetAddressMode
     (pam : access CUaddress_mode;
      hTexRef : CUtexref;
      dim : int) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:10044
   pragma Import (C, cuTexRefGetAddressMode, "cuTexRefGetAddressMode");

  --*
  -- * \brief Gets the filter-mode used by a texture reference
  -- *
  -- * Returns in \p *pfm the filtering mode of the texture reference
  -- * \p hTexRef.
  -- *
  -- * \param pfm     - Returned filtering mode
  -- * \param hTexRef - Texture reference
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- *
  -- * \sa ::cuTexRefSetAddress,
  -- * ::cuTexRefSetAddress2D, ::cuTexRefSetAddressMode, ::cuTexRefSetArray,
  -- * ::cuTexRefSetFilterMode, ::cuTexRefSetFlags, ::cuTexRefSetFormat,
  -- * ::cuTexRefGetAddress, ::cuTexRefGetAddressMode, ::cuTexRefGetArray,
  -- * ::cuTexRefGetFlags, ::cuTexRefGetFormat
  --  

   function cuTexRefGetFilterMode (pfm : access CUfilter_mode; hTexRef : CUtexref) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:10068
   pragma Import (C, cuTexRefGetFilterMode, "cuTexRefGetFilterMode");

  --*
  -- * \brief Gets the format used by a texture reference
  -- *
  -- * Returns in \p *pFormat and \p *pNumChannels the format and number
  -- * of components of the CUDA array bound to the texture reference \p hTexRef.
  -- * If \p pFormat or \p pNumChannels is NULL, it will be ignored.
  -- *
  -- * \param pFormat      - Returned format
  -- * \param pNumChannels - Returned number of components
  -- * \param hTexRef      - Texture reference
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- *
  -- * \sa ::cuTexRefSetAddress,
  -- * ::cuTexRefSetAddress2D, ::cuTexRefSetAddressMode, ::cuTexRefSetArray,
  -- * ::cuTexRefSetFilterMode, ::cuTexRefSetFlags, ::cuTexRefSetFormat,
  -- * ::cuTexRefGetAddress, ::cuTexRefGetAddressMode, ::cuTexRefGetArray,
  -- * ::cuTexRefGetFilterMode, ::cuTexRefGetFlags
  --  

   function cuTexRefGetFormat
     (pFormat : access CUarray_format;
      pNumChannels : access int;
      hTexRef : CUtexref) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:10094
   pragma Import (C, cuTexRefGetFormat, "cuTexRefGetFormat");

  --*
  -- * \brief Gets the mipmap filtering mode for a texture reference
  -- *
  -- * Returns the mipmap filtering mode in \p pfm that's used when reading memory through
  -- * the texture reference \p hTexRef.
  -- *
  -- * \param pfm     - Returned mipmap filtering mode
  -- * \param hTexRef - Texture reference
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- *
  -- * \sa ::cuTexRefSetAddress,
  -- * ::cuTexRefSetAddress2D, ::cuTexRefSetAddressMode, ::cuTexRefSetArray,
  -- * ::cuTexRefSetFlags, ::cuTexRefSetFormat,
  -- * ::cuTexRefGetAddress, ::cuTexRefGetAddressMode, ::cuTexRefGetArray,
  -- * ::cuTexRefGetFilterMode, ::cuTexRefGetFlags, ::cuTexRefGetFormat
  --  

   function cuTexRefGetMipmapFilterMode (pfm : access CUfilter_mode; hTexRef : CUtexref) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:10118
   pragma Import (C, cuTexRefGetMipmapFilterMode, "cuTexRefGetMipmapFilterMode");

  --*
  -- * \brief Gets the mipmap level bias for a texture reference
  -- *
  -- * Returns the mipmap level bias in \p pBias that's added to the specified mipmap
  -- * level when reading memory through the texture reference \p hTexRef.
  -- *
  -- * \param pbias   - Returned mipmap level bias
  -- * \param hTexRef - Texture reference
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- *
  -- * \sa ::cuTexRefSetAddress,
  -- * ::cuTexRefSetAddress2D, ::cuTexRefSetAddressMode, ::cuTexRefSetArray,
  -- * ::cuTexRefSetFlags, ::cuTexRefSetFormat,
  -- * ::cuTexRefGetAddress, ::cuTexRefGetAddressMode, ::cuTexRefGetArray,
  -- * ::cuTexRefGetFilterMode, ::cuTexRefGetFlags, ::cuTexRefGetFormat
  --  

   function cuTexRefGetMipmapLevelBias (pbias : access float; hTexRef : CUtexref) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:10142
   pragma Import (C, cuTexRefGetMipmapLevelBias, "cuTexRefGetMipmapLevelBias");

  --*
  -- * \brief Gets the min/max mipmap level clamps for a texture reference
  -- *
  -- * Returns the min/max mipmap level clamps in \p pminMipmapLevelClamp and \p pmaxMipmapLevelClamp
  -- * that's used when reading memory through the texture reference \p hTexRef. 
  -- *
  -- * \param pminMipmapLevelClamp - Returned mipmap min level clamp
  -- * \param pmaxMipmapLevelClamp - Returned mipmap max level clamp
  -- * \param hTexRef              - Texture reference
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- *
  -- * \sa ::cuTexRefSetAddress,
  -- * ::cuTexRefSetAddress2D, ::cuTexRefSetAddressMode, ::cuTexRefSetArray,
  -- * ::cuTexRefSetFlags, ::cuTexRefSetFormat,
  -- * ::cuTexRefGetAddress, ::cuTexRefGetAddressMode, ::cuTexRefGetArray,
  -- * ::cuTexRefGetFilterMode, ::cuTexRefGetFlags, ::cuTexRefGetFormat
  --  

   function cuTexRefGetMipmapLevelClamp
     (pminMipmapLevelClamp : access float;
      pmaxMipmapLevelClamp : access float;
      hTexRef : CUtexref) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:10167
   pragma Import (C, cuTexRefGetMipmapLevelClamp, "cuTexRefGetMipmapLevelClamp");

  --*
  -- * \brief Gets the maximum anisotropy for a texture reference
  -- *
  -- * Returns the maximum anisotropy in \p pmaxAniso that's used when reading memory through
  -- * the texture reference \p hTexRef. 
  -- *
  -- * \param pmaxAniso - Returned maximum anisotropy
  -- * \param hTexRef   - Texture reference
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- *
  -- * \sa ::cuTexRefSetAddress,
  -- * ::cuTexRefSetAddress2D, ::cuTexRefSetAddressMode, ::cuTexRefSetArray,
  -- * ::cuTexRefSetFlags, ::cuTexRefSetFormat,
  -- * ::cuTexRefGetAddress, ::cuTexRefGetAddressMode, ::cuTexRefGetArray,
  -- * ::cuTexRefGetFilterMode, ::cuTexRefGetFlags, ::cuTexRefGetFormat
  --  

   function cuTexRefGetMaxAnisotropy (pmaxAniso : access int; hTexRef : CUtexref) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:10191
   pragma Import (C, cuTexRefGetMaxAnisotropy, "cuTexRefGetMaxAnisotropy");

  --*
  -- * \brief Gets the border color used by a texture reference
  -- *
  -- * Returns in \p pBorderColor, values of the RGBA color used by
  -- * the texture reference \p hTexRef.
  -- * The color value is of type float and holds color components in
  -- * the following sequence:
  -- * pBorderColor[0] holds 'R' component
  -- * pBorderColor[1] holds 'G' component
  -- * pBorderColor[2] holds 'B' component
  -- * pBorderColor[3] holds 'A' component
  -- *
  -- * \param hTexRef  - Texture reference
  -- * \param pBorderColor   - Returned Type and Value of RGBA color
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- *
  -- * \sa ::cuTexRefSetAddressMode,
  -- * ::cuTexRefSetAddressMode, ::cuTexRefSetBorderColor
  --  

   function cuTexRefGetBorderColor (pBorderColor : access float; hTexRef : CUtexref) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:10218
   pragma Import (C, cuTexRefGetBorderColor, "cuTexRefGetBorderColor");

  --*
  -- * \brief Gets the flags used by a texture reference
  -- *
  -- * Returns in \p *pFlags the flags of the texture reference \p hTexRef.
  -- *
  -- * \param pFlags  - Returned flags
  -- * \param hTexRef - Texture reference
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- *
  -- * \sa ::cuTexRefSetAddress,
  -- * ::cuTexRefSetAddress2D, ::cuTexRefSetAddressMode, ::cuTexRefSetArray,
  -- * ::cuTexRefSetFilterMode, ::cuTexRefSetFlags, ::cuTexRefSetFormat,
  -- * ::cuTexRefGetAddress, ::cuTexRefGetAddressMode, ::cuTexRefGetArray,
  -- * ::cuTexRefGetFilterMode, ::cuTexRefGetFormat
  --  

   function cuTexRefGetFlags (pFlags : access unsigned; hTexRef : CUtexref) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:10241
   pragma Import (C, cuTexRefGetFlags, "cuTexRefGetFlags");

  --* @}  
  -- END CUDA_TEXREF  
  --*
  -- * \defgroup CUDA_TEXREF_DEPRECATED Texture Reference Management [DEPRECATED]
  -- *
  -- * ___MANBRIEF___ deprecated texture reference management functions of the
  -- * low-level CUDA driver API (___CURRENT_FILE___) ___ENDMANBRIEF___
  -- *
  -- * This section describes the deprecated texture reference management
  -- * functions of the low-level CUDA driver application programming interface.
  -- *
  -- * @{
  --  

  --*
  -- * \brief Creates a texture reference
  -- *
  -- * \deprecated
  -- *
  -- * Creates a texture reference and returns its handle in \p *pTexRef. Once
  -- * created, the application must call ::cuTexRefSetArray() or
  -- * ::cuTexRefSetAddress() to associate the reference with allocated memory.
  -- * Other texture reference functions are used to specify the format and
  -- * interpretation (addressing, filtering, etc.) to be used when the memory is
  -- * read through this texture reference.
  -- *
  -- * \param pTexRef - Returned texture reference
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- *
  -- * \sa ::cuTexRefDestroy
  --  

   function cuTexRefCreate (pTexRef : System.Address) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:10280
   pragma Import (C, cuTexRefCreate, "cuTexRefCreate");

  --*
  -- * \brief Destroys a texture reference
  -- *
  -- * \deprecated
  -- *
  -- * Destroys the texture reference specified by \p hTexRef.
  -- *
  -- * \param hTexRef - Texture reference to destroy
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- *
  -- * \sa ::cuTexRefCreate
  --  

   function cuTexRefDestroy (hTexRef : CUtexref) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:10300
   pragma Import (C, cuTexRefDestroy, "cuTexRefDestroy");

  --* @}  
  -- END CUDA_TEXREF_DEPRECATED  
  --*
  -- * \defgroup CUDA_SURFREF Surface Reference Management
  -- *
  -- * ___MANBRIEF___ surface reference management functions of the low-level CUDA
  -- * driver API (___CURRENT_FILE___) ___ENDMANBRIEF___
  -- *
  -- * This section describes the surface reference management functions of the
  -- * low-level CUDA driver application programming interface.
  -- *
  -- * @{
  --  

  --*
  -- * \brief Sets the CUDA array for a surface reference.
  -- *
  -- * Sets the CUDA array \p hArray to be read and written by the surface reference
  -- * \p hSurfRef.  Any previous CUDA array state associated with the surface
  -- * reference is superseded by this function.  \p Flags must be set to 0.
  -- * The ::CUDA_ARRAY3D_SURFACE_LDST flag must have been set for the CUDA array.
  -- * Any CUDA array previously bound to \p hSurfRef is unbound.
  -- * \param hSurfRef - Surface reference handle
  -- * \param hArray - CUDA array handle
  -- * \param Flags - set to 0
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- *
  -- * \sa ::cuModuleGetSurfRef, ::cuSurfRefGetArray
  --  

   function cuSurfRefSetArray
     (hSurfRef : CUsurfref;
      hArray : CUarray;
      Flags : unsigned) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:10339
   pragma Import (C, cuSurfRefSetArray, "cuSurfRefSetArray");

  --*
  -- * \brief Passes back the CUDA array bound to a surface reference.
  -- *
  -- * Returns in \p *phArray the CUDA array bound to the surface reference
  -- * \p hSurfRef, or returns ::CUDA_ERROR_INVALID_VALUE if the surface reference
  -- * is not bound to any CUDA array.
  -- * \param phArray - Surface reference handle
  -- * \param hSurfRef - Surface reference handle
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- *
  -- * \sa ::cuModuleGetSurfRef, ::cuSurfRefSetArray
  --  

   function cuSurfRefGetArray (phArray : System.Address; hSurfRef : CUsurfref) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:10360
   pragma Import (C, cuSurfRefGetArray, "cuSurfRefGetArray");

  --* @}  
  -- END CUDA_SURFREF  
  --*
  -- * \defgroup CUDA_TEXOBJECT Texture Object Management
  -- *
  -- * ___MANBRIEF___ texture object management functions of the low-level CUDA
  -- * driver API (___CURRENT_FILE___) ___ENDMANBRIEF___
  -- *
  -- * This section describes the texture object management functions of the
  -- * low-level CUDA driver application programming interface. The texture
  -- * object API is only supported on devices of compute capability 3.0 or higher.
  -- *
  -- * @{
  --  

  --*
  -- * \brief Creates a texture object
  -- *
  -- * Creates a texture object and returns it in \p pTexObject. \p pResDesc describes
  -- * the data to texture from. \p pTexDesc describes how the data should be sampled.
  -- * \p pResViewDesc is an optional argument that specifies an alternate format for
  -- * the data described by \p pResDesc, and also describes the subresource region
  -- * to restrict access to when texturing. \p pResViewDesc can only be specified if
  -- * the type of resource is a CUDA array or a CUDA mipmapped array.
  -- *
  -- * Texture objects are only supported on devices of compute capability 3.0 or higher.
  -- * Additionally, a texture object is an opaque value, and, as such, should only be
  -- * accessed through CUDA API calls.
  -- *
  -- * The ::CUDA_RESOURCE_DESC structure is defined as:
  -- * \code
  --        typedef struct CUDA_RESOURCE_DESC_st
  --        {
  --            CUresourcetype resType;
  --            union {
  --                struct {
  --                    CUarray hArray;
  --                } array;
  --                struct {
  --                    CUmipmappedArray hMipmappedArray;
  --                } mipmap;
  --                struct {
  --                    CUdeviceptr devPtr;
  --                    CUarray_format format;
  --                    unsigned int numChannels;
  --                    size_t sizeInBytes;
  --                } linear;
  --                struct {
  --                    CUdeviceptr devPtr;
  --                    CUarray_format format;
  --                    unsigned int numChannels;
  --                    size_t width;
  --                    size_t height;
  --                    size_t pitchInBytes;
  --                } pitch2D;
  --            } res;
  --            unsigned int flags;
  --        } CUDA_RESOURCE_DESC;
  -- * \endcode
  -- * where:
  -- * - ::CUDA_RESOURCE_DESC::resType specifies the type of resource to texture from.
  -- * CUresourceType is defined as:
  -- * \code
  --        typedef enum CUresourcetype_enum {
  --            CU_RESOURCE_TYPE_ARRAY           = 0x00,
  --            CU_RESOURCE_TYPE_MIPMAPPED_ARRAY = 0x01,
  --            CU_RESOURCE_TYPE_LINEAR          = 0x02,
  --            CU_RESOURCE_TYPE_PITCH2D         = 0x03
  --        } CUresourcetype;
  -- * \endcode
  -- *
  -- * \par
  -- * If ::CUDA_RESOURCE_DESC::resType is set to ::CU_RESOURCE_TYPE_ARRAY, ::CUDA_RESOURCE_DESC::res::array::hArray
  -- * must be set to a valid CUDA array handle.
  -- *
  -- * \par
  -- * If ::CUDA_RESOURCE_DESC::resType is set to ::CU_RESOURCE_TYPE_MIPMAPPED_ARRAY, ::CUDA_RESOURCE_DESC::res::mipmap::hMipmappedArray
  -- * must be set to a valid CUDA mipmapped array handle.
  -- *
  -- * \par
  -- * If ::CUDA_RESOURCE_DESC::resType is set to ::CU_RESOURCE_TYPE_LINEAR, ::CUDA_RESOURCE_DESC::res::linear::devPtr
  -- * must be set to a valid device pointer, that is aligned to ::CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT.
  -- * ::CUDA_RESOURCE_DESC::res::linear::format and ::CUDA_RESOURCE_DESC::res::linear::numChannels
  -- * describe the format of each component and the number of components per array element. ::CUDA_RESOURCE_DESC::res::linear::sizeInBytes
  -- * specifies the size of the array in bytes. The total number of elements in the linear address range cannot exceed 
  -- * ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH. The number of elements is computed as (sizeInBytes / (sizeof(format) * numChannels)).
  -- *
  -- * \par
  -- * If ::CUDA_RESOURCE_DESC::resType is set to ::CU_RESOURCE_TYPE_PITCH2D, ::CUDA_RESOURCE_DESC::res::pitch2D::devPtr
  -- * must be set to a valid device pointer, that is aligned to ::CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT.
  -- * ::CUDA_RESOURCE_DESC::res::pitch2D::format and ::CUDA_RESOURCE_DESC::res::pitch2D::numChannels
  -- * describe the format of each component and the number of components per array element. ::CUDA_RESOURCE_DESC::res::pitch2D::width
  -- * and ::CUDA_RESOURCE_DESC::res::pitch2D::height specify the width and height of the array in elements, and cannot exceed
  -- * ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH and ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT respectively.
  -- * ::CUDA_RESOURCE_DESC::res::pitch2D::pitchInBytes specifies the pitch between two rows in bytes and has to be aligned to 
  -- * ::CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT. Pitch cannot exceed ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH.
  -- *
  -- * - ::flags must be set to zero.
  -- *
  -- *
  -- * The ::CUDA_TEXTURE_DESC struct is defined as
  -- * \code
  --        typedef struct CUDA_TEXTURE_DESC_st {
  --            CUaddress_mode addressMode[3];
  --            CUfilter_mode filterMode;
  --            unsigned int flags;
  --            unsigned int maxAnisotropy;
  --            CUfilter_mode mipmapFilterMode;
  --            float mipmapLevelBias;
  --            float minMipmapLevelClamp;
  --            float maxMipmapLevelClamp;
  --        } CUDA_TEXTURE_DESC;
  -- * \endcode
  -- * where
  -- * - ::CUDA_TEXTURE_DESC::addressMode specifies the addressing mode for each dimension of the texture data. ::CUaddress_mode is defined as:
  -- *   \code
  --        typedef enum CUaddress_mode_enum {
  --            CU_TR_ADDRESS_MODE_WRAP = 0,
  --            CU_TR_ADDRESS_MODE_CLAMP = 1,
  --            CU_TR_ADDRESS_MODE_MIRROR = 2,
  --            CU_TR_ADDRESS_MODE_BORDER = 3
  --        } CUaddress_mode;
  -- *   \endcode
  -- *   This is ignored if ::CUDA_RESOURCE_DESC::resType is ::CU_RESOURCE_TYPE_LINEAR. Also, if the flag, ::CU_TRSF_NORMALIZED_COORDINATES 
  -- *   is not set, the only supported address mode is ::CU_TR_ADDRESS_MODE_CLAMP.
  -- *
  -- * - ::CUDA_TEXTURE_DESC::filterMode specifies the filtering mode to be used when fetching from the texture. CUfilter_mode is defined as:
  -- *   \code
  --        typedef enum CUfilter_mode_enum {
  --            CU_TR_FILTER_MODE_POINT = 0,
  --            CU_TR_FILTER_MODE_LINEAR = 1
  --        } CUfilter_mode;
  -- *   \endcode
  -- *   This is ignored if ::CUDA_RESOURCE_DESC::resType is ::CU_RESOURCE_TYPE_LINEAR.
  -- *
  -- * - ::CUDA_TEXTURE_DESC::flags can be any combination of the following:
  -- *   - ::CU_TRSF_READ_AS_INTEGER, which suppresses the default behavior of having the texture promote integer data to floating point data in the
  -- *     range [0, 1]. Note that texture with 32-bit integer format would not be promoted, regardless of whether or not this flag is specified.
  -- *   - ::CU_TRSF_NORMALIZED_COORDINATES, which suppresses the default behavior of having the texture coordinates range from [0, Dim) where Dim is
  -- *     the width or height of the CUDA array. Instead, the texture coordinates [0, 1.0) reference the entire breadth of the array dimension; Note
  -- *     that for CUDA mipmapped arrays, this flag has to be set.
  -- *
  -- * - ::CUDA_TEXTURE_DESC::maxAnisotropy specifies the maximum anisotropy ratio to be used when doing anisotropic filtering. This value will be
  -- *   clamped to the range [1,16].
  -- *
  -- * - ::CUDA_TEXTURE_DESC::mipmapFilterMode specifies the filter mode when the calculated mipmap level lies between two defined mipmap levels.
  -- *
  -- * - ::CUDA_TEXTURE_DESC::mipmapLevelBias specifies the offset to be applied to the calculated mipmap level.
  -- *
  -- * - ::CUDA_TEXTURE_DESC::minMipmapLevelClamp specifies the lower end of the mipmap level range to clamp access to.
  -- *
  -- * - ::CUDA_TEXTURE_DESC::maxMipmapLevelClamp specifies the upper end of the mipmap level range to clamp access to.
  -- *
  -- *
  -- * The ::CUDA_RESOURCE_VIEW_DESC struct is defined as
  -- * \code
  --        typedef struct CUDA_RESOURCE_VIEW_DESC_st
  --        {
  --            CUresourceViewFormat format;
  --            size_t width;
  --            size_t height;
  --            size_t depth;
  --            unsigned int firstMipmapLevel;
  --            unsigned int lastMipmapLevel;
  --            unsigned int firstLayer;
  --            unsigned int lastLayer;
  --        } CUDA_RESOURCE_VIEW_DESC;
  -- * \endcode
  -- * where:
  -- * - ::CUDA_RESOURCE_VIEW_DESC::format specifies how the data contained in the CUDA array or CUDA mipmapped array should
  -- *   be interpreted. Note that this can incur a change in size of the texture data. If the resource view format is a block
  -- *   compressed format, then the underlying CUDA array or CUDA mipmapped array has to have a base of format ::CU_AD_FORMAT_UNSIGNED_INT32.
  -- *   with 2 or 4 channels, depending on the block compressed format. For ex., BC1 and BC4 require the underlying CUDA array to have
  -- *   a format of ::CU_AD_FORMAT_UNSIGNED_INT32 with 2 channels. The other BC formats require the underlying resource to have the same base
  -- *   format but with 4 channels.
  -- *
  -- * - ::CUDA_RESOURCE_VIEW_DESC::width specifies the new width of the texture data. If the resource view format is a block
  -- *   compressed format, this value has to be 4 times the original width of the resource. For non block compressed formats,
  -- *   this value has to be equal to that of the original resource.
  -- *
  -- * - ::CUDA_RESOURCE_VIEW_DESC::height specifies the new height of the texture data. If the resource view format is a block
  -- *   compressed format, this value has to be 4 times the original height of the resource. For non block compressed formats,
  -- *   this value has to be equal to that of the original resource.
  -- *
  -- * - ::CUDA_RESOURCE_VIEW_DESC::depth specifies the new depth of the texture data. This value has to be equal to that of the
  -- *   original resource.
  -- *
  -- * - ::CUDA_RESOURCE_VIEW_DESC::firstMipmapLevel specifies the most detailed mipmap level. This will be the new mipmap level zero.
  -- *   For non-mipmapped resources, this value has to be zero.::CUDA_TEXTURE_DESC::minMipmapLevelClamp and ::CUDA_TEXTURE_DESC::maxMipmapLevelClamp
  -- *   will be relative to this value. For ex., if the firstMipmapLevel is set to 2, and a minMipmapLevelClamp of 1.2 is specified,
  -- *   then the actual minimum mipmap level clamp will be 3.2.
  -- *
  -- * - ::CUDA_RESOURCE_VIEW_DESC::lastMipmapLevel specifies the least detailed mipmap level. For non-mipmapped resources, this value
  -- *   has to be zero.
  -- *
  -- * - ::CUDA_RESOURCE_VIEW_DESC::firstLayer specifies the first layer index for layered textures. This will be the new layer zero.
  -- *   For non-layered resources, this value has to be zero.
  -- *
  -- * - ::CUDA_RESOURCE_VIEW_DESC::lastLayer specifies the last layer index for layered textures. For non-layered resources, 
  -- *   this value has to be zero.
  -- *
  -- *
  -- * \param pTexObject   - Texture object to create
  -- * \param pResDesc     - Resource descriptor
  -- * \param pTexDesc     - Texture descriptor
  -- * \param pResViewDesc - Resource view descriptor 
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- *
  -- * \sa ::cuTexObjectDestroy
  --  

   function cuTexObjectCreate
     (pTexObject : access CUtexObject;
      pResDesc : access constant CUDA_RESOURCE_DESC;
      pTexDesc : access constant CUDA_TEXTURE_DESC;
      pResViewDesc : access constant CUDA_RESOURCE_VIEW_DESC) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:10582
   pragma Import (C, cuTexObjectCreate, "cuTexObjectCreate");

  --*
  -- * \brief Destroys a texture object
  -- *
  -- * Destroys the texture object specified by \p texObject.
  -- *
  -- * \param texObject - Texture object to destroy
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- *
  -- * \sa ::cuTexObjectCreate
  --  

   function cuTexObjectDestroy (texObject : CUtexObject) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:10600
   pragma Import (C, cuTexObjectDestroy, "cuTexObjectDestroy");

  --*
  -- * \brief Returns a texture object's resource descriptor
  -- *
  -- * Returns the resource descriptor for the texture object specified by \p texObject.
  -- *
  -- * \param pResDesc  - Resource descriptor
  -- * \param texObject - Texture object
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- *
  -- * \sa ::cuTexObjectCreate
  --  

   function cuTexObjectGetResourceDesc (pResDesc : access CUDA_RESOURCE_DESC; texObject : CUtexObject) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:10619
   pragma Import (C, cuTexObjectGetResourceDesc, "cuTexObjectGetResourceDesc");

  --*
  -- * \brief Returns a texture object's texture descriptor
  -- *
  -- * Returns the texture descriptor for the texture object specified by \p texObject.
  -- *
  -- * \param pTexDesc  - Texture descriptor
  -- * \param texObject - Texture object
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- *
  -- * \sa ::cuTexObjectCreate
  --  

   function cuTexObjectGetTextureDesc (pTexDesc : access CUDA_TEXTURE_DESC; texObject : CUtexObject) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:10638
   pragma Import (C, cuTexObjectGetTextureDesc, "cuTexObjectGetTextureDesc");

  --*
  -- * \brief Returns a texture object's resource view descriptor
  -- *
  -- * Returns the resource view descriptor for the texture object specified by \p texObject.
  -- * If no resource view was set for \p texObject, the ::CUDA_ERROR_INVALID_VALUE is returned.
  -- *
  -- * \param pResViewDesc - Resource view descriptor
  -- * \param texObject    - Texture object
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- *
  -- * \sa ::cuTexObjectCreate
  --  

   function cuTexObjectGetResourceViewDesc (pResViewDesc : access CUDA_RESOURCE_VIEW_DESC; texObject : CUtexObject) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:10658
   pragma Import (C, cuTexObjectGetResourceViewDesc, "cuTexObjectGetResourceViewDesc");

  --* @}  
  -- END CUDA_TEXOBJECT  
  --*
  -- * \defgroup CUDA_SURFOBJECT Surface Object Management
  -- *
  -- * ___MANBRIEF___ surface object management functions of the low-level CUDA
  -- * driver API (___CURRENT_FILE___) ___ENDMANBRIEF___
  -- *
  -- * This section describes the surface object management functions of the
  -- * low-level CUDA driver application programming interface. The surface
  -- * object API is only supported on devices of compute capability 3.0 or higher.
  -- *
  -- * @{
  --  

  --*
  -- * \brief Creates a surface object
  -- *
  -- * Creates a surface object and returns it in \p pSurfObject. \p pResDesc describes
  -- * the data to perform surface load/stores on. ::CUDA_RESOURCE_DESC::resType must be 
  -- * ::CU_RESOURCE_TYPE_ARRAY and  ::CUDA_RESOURCE_DESC::res::array::hArray
  -- * must be set to a valid CUDA array handle. ::CUDA_RESOURCE_DESC::flags must be set to zero.
  -- *
  -- * Surface objects are only supported on devices of compute capability 3.0 or higher.
  -- * Additionally, a surface object is an opaque value, and, as such, should only be
  -- * accessed through CUDA API calls.
  -- *
  -- * \param pSurfObject - Surface object to create
  -- * \param pResDesc    - Resource descriptor
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- *
  -- * \sa ::cuSurfObjectDestroy
  --  

   function cuSurfObjectCreate (pSurfObject : access CUsurfObject; pResDesc : access constant CUDA_RESOURCE_DESC) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:10699
   pragma Import (C, cuSurfObjectCreate, "cuSurfObjectCreate");

  --*
  -- * \brief Destroys a surface object
  -- *
  -- * Destroys the surface object specified by \p surfObject.
  -- *
  -- * \param surfObject - Surface object to destroy
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- *
  -- * \sa ::cuSurfObjectCreate
  --  

   function cuSurfObjectDestroy (surfObject : CUsurfObject) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:10717
   pragma Import (C, cuSurfObjectDestroy, "cuSurfObjectDestroy");

  --*
  -- * \brief Returns a surface object's resource descriptor
  -- *
  -- * Returns the resource descriptor for the surface object specified by \p surfObject.
  -- *
  -- * \param pResDesc   - Resource descriptor
  -- * \param surfObject - Surface object
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- *
  -- * \sa ::cuSurfObjectCreate
  --  

   function cuSurfObjectGetResourceDesc (pResDesc : access CUDA_RESOURCE_DESC; surfObject : CUsurfObject) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:10736
   pragma Import (C, cuSurfObjectGetResourceDesc, "cuSurfObjectGetResourceDesc");

  --* @}  
  -- END CUDA_SURFOBJECT  
  --*
  -- * \defgroup CUDA_PEER_ACCESS Peer Context Memory Access
  -- *
  -- * ___MANBRIEF___ direct peer context memory access functions of the low-level
  -- * CUDA driver API (___CURRENT_FILE___) ___ENDMANBRIEF___
  -- *
  -- * This section describes the direct peer context memory access functions 
  -- * of the low-level CUDA driver application programming interface.
  -- *
  -- * @{
  --  

  --*
  -- * \brief Queries if a device may directly access a peer device's memory.
  -- *
  -- * Returns in \p *canAccessPeer a value of 1 if contexts on \p dev are capable of
  -- * directly accessing memory from contexts on \p peerDev and 0 otherwise.
  -- * If direct access of \p peerDev from \p dev is possible, then access may be
  -- * enabled on two specific contexts by calling ::cuCtxEnablePeerAccess().
  -- *
  -- * \param canAccessPeer - Returned access capability
  -- * \param dev           - Device from which allocations on \p peerDev are to
  -- *                        be directly accessed.
  -- * \param peerDev       - Device on which the allocations to be directly accessed 
  -- *                        by \p dev reside.
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_DEVICE
  -- * \notefnerr
  -- *
  -- * \sa ::cuCtxEnablePeerAccess,
  -- * ::cuCtxDisablePeerAccess
  --  

   function cuDeviceCanAccessPeer
     (canAccessPeer : access int;
      dev : CUdevice;
      peerDev : CUdevice) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:10778
   pragma Import (C, cuDeviceCanAccessPeer, "cuDeviceCanAccessPeer");

  --*
  -- * \brief Queries attributes of the link between two devices.
  -- *
  -- * Returns in \p *value the value of the requested attribute \p attrib of the
  -- * link between \p srcDevice and \p dstDevice. The supported attributes are:
  -- * - ::CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK: A relative value indicating the
  -- *   performance of the link between two devices.
  -- * - ::CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED P2P: 1 if P2P Access is enable.
  -- * - ::CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED: 1 if Atomic operations over
  -- *   the link are supported.
  -- *
  -- * Returns ::CUDA_ERROR_INVALID_DEVICE if \p srcDevice or \p dstDevice are not valid
  -- * or if they represent the same device.
  -- *
  -- * Returns ::CUDA_ERROR_INVALID_VALUE if \p attrib is not valid or if \p value is
  -- * a null pointer.
  -- *
  -- * \param value         - Returned value of the requested attribute
  -- * \param attrib        - The requested attribute of the link between \p srcDevice and \p dstDevice.
  -- * \param srcDevice     - The source device of the target link.
  -- * \param dstDevice     - The destination device of the target link.
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_DEVICE,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- * \notefnerr
  -- *
  -- * \sa ::cuCtxEnablePeerAccess,
  -- * ::cuCtxDisablePeerAccess,
  -- * ::cuCtxCanAccessPeer
  --  

   function cuDeviceGetP2PAttribute
     (value : access int;
      attrib : CUdevice_P2PAttribute;
      srcDevice : CUdevice;
      dstDevice : CUdevice) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:10815
   pragma Import (C, cuDeviceGetP2PAttribute, "cuDeviceGetP2PAttribute");

  --*
  -- * \brief Enables direct access to memory allocations in a peer context.
  -- *
  -- * If both the current context and \p peerContext are on devices which support unified
  -- * addressing (as may be queried using ::CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING) and same
  -- * major compute capability, then on success all allocations from \p peerContext will
  -- * immediately be accessible by the current context.  See \ref CUDA_UNIFIED for additional
  -- * details.
  -- *
  -- * Note that access granted by this call is unidirectional and that in order to access
  -- * memory from the current context in \p peerContext, a separate symmetric call 
  -- * to ::cuCtxEnablePeerAccess() is required.
  -- *
  -- * There is a system-wide maximum of eight peer connections per device.
  -- *
  -- * Returns ::CUDA_ERROR_PEER_ACCESS_UNSUPPORTED if ::cuDeviceCanAccessPeer() indicates
  -- * that the ::CUdevice of the current context cannot directly access memory
  -- * from the ::CUdevice of \p peerContext.
  -- *
  -- * Returns ::CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED if direct access of
  -- * \p peerContext from the current context has already been enabled.
  -- *
  -- * Returns ::CUDA_ERROR_TOO_MANY_PEERS if direct peer access is not possible 
  -- * because hardware resources required for peer access have been exhausted.
  -- *
  -- * Returns ::CUDA_ERROR_INVALID_CONTEXT if there is no current context, \p peerContext
  -- * is not a valid context, or if the current context is \p peerContext.
  -- *
  -- * Returns ::CUDA_ERROR_INVALID_VALUE if \p Flags is not 0.
  -- *
  -- * \param peerContext - Peer context to enable direct access to from the current context
  -- * \param Flags       - Reserved for future use and must be set to 0
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED,
  -- * ::CUDA_ERROR_TOO_MANY_PEERS,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_PEER_ACCESS_UNSUPPORTED,
  -- * ::CUDA_ERROR_INVALID_VALUE
  -- * \notefnerr
  -- *
  -- * \sa ::cuDeviceCanAccessPeer,
  -- * ::cuCtxDisablePeerAccess
  --  

   function cuCtxEnablePeerAccess (peerContext : CUcontext; Flags : unsigned) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:10864
   pragma Import (C, cuCtxEnablePeerAccess, "cuCtxEnablePeerAccess");

  --*
  -- * \brief Disables direct access to memory allocations in a peer context and 
  -- * unregisters any registered allocations.
  -- *
  --  Returns ::CUDA_ERROR_PEER_ACCESS_NOT_ENABLED if direct peer access has 
  -- * not yet been enabled from \p peerContext to the current context.
  -- *
  -- * Returns ::CUDA_ERROR_INVALID_CONTEXT if there is no current context, or if
  -- * \p peerContext is not a valid context.
  -- *
  -- * \param peerContext - Peer context to disable direct access to
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_PEER_ACCESS_NOT_ENABLED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * \notefnerr
  -- *
  -- * \sa ::cuDeviceCanAccessPeer,
  -- * ::cuCtxEnablePeerAccess
  --  

   function cuCtxDisablePeerAccess (peerContext : CUcontext) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:10889
   pragma Import (C, cuCtxDisablePeerAccess, "cuCtxDisablePeerAccess");

  --* @}  
  -- END CUDA_PEER_ACCESS  
  --*
  -- * \defgroup CUDA_GRAPHICS Graphics Interoperability
  -- *
  -- * ___MANBRIEF___ graphics interoperability functions of the low-level CUDA
  -- * driver API (___CURRENT_FILE___) ___ENDMANBRIEF___
  -- *
  -- * This section describes the graphics interoperability functions of the
  -- * low-level CUDA driver application programming interface.
  -- *
  -- * @{
  --  

  --*
  -- * \brief Unregisters a graphics resource for access by CUDA
  -- *
  -- * Unregisters the graphics resource \p resource so it is not accessible by
  -- * CUDA unless registered again.
  -- *
  -- * If \p resource is invalid then ::CUDA_ERROR_INVALID_HANDLE is
  -- * returned.
  -- *
  -- * \param resource - Resource to unregister
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_HANDLE,
  -- * ::CUDA_ERROR_UNKNOWN
  -- * \notefnerr
  -- *
  -- * \sa
  -- * ::cuGraphicsD3D9RegisterResource,
  -- * ::cuGraphicsD3D10RegisterResource,
  -- * ::cuGraphicsD3D11RegisterResource,
  -- * ::cuGraphicsGLRegisterBuffer,
  -- * ::cuGraphicsGLRegisterImage
  --  

   function cuGraphicsUnregisterResource (resource : CUgraphicsResource) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:10933
   pragma Import (C, cuGraphicsUnregisterResource, "cuGraphicsUnregisterResource");

  --*
  -- * \brief Get an array through which to access a subresource of a mapped graphics resource.
  -- *
  -- * Returns in \p *pArray an array through which the subresource of the mapped
  -- * graphics resource \p resource which corresponds to array index \p arrayIndex
  -- * and mipmap level \p mipLevel may be accessed.  The value set in \p *pArray may
  -- * change every time that \p resource is mapped.
  -- *
  -- * If \p resource is not a texture then it cannot be accessed via an array and
  -- * ::CUDA_ERROR_NOT_MAPPED_AS_ARRAY is returned.
  -- * If \p arrayIndex is not a valid array index for \p resource then
  -- * ::CUDA_ERROR_INVALID_VALUE is returned.
  -- * If \p mipLevel is not a valid mipmap level for \p resource then
  -- * ::CUDA_ERROR_INVALID_VALUE is returned.
  -- * If \p resource is not mapped then ::CUDA_ERROR_NOT_MAPPED is returned.
  -- *
  -- * \param pArray      - Returned array through which a subresource of \p resource may be accessed
  -- * \param resource    - Mapped resource to access
  -- * \param arrayIndex  - Array index for array textures or cubemap face
  -- *                      index as defined by ::CUarray_cubemap_face for
  -- *                      cubemap textures for the subresource to access
  -- * \param mipLevel    - Mipmap level for the subresource to access
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE,
  -- * ::CUDA_ERROR_INVALID_HANDLE,
  -- * ::CUDA_ERROR_NOT_MAPPED,
  -- * ::CUDA_ERROR_NOT_MAPPED_AS_ARRAY
  -- * \notefnerr
  -- *
  -- * \sa ::cuGraphicsResourceGetMappedPointer
  --  

   function cuGraphicsSubResourceGetMappedArray
     (pArray : System.Address;
      resource : CUgraphicsResource;
      arrayIndex : unsigned;
      mipLevel : unsigned) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:10971
   pragma Import (C, cuGraphicsSubResourceGetMappedArray, "cuGraphicsSubResourceGetMappedArray");

  --*
  -- * \brief Get a mipmapped array through which to access a mapped graphics resource.
  -- *
  -- * Returns in \p *pMipmappedArray a mipmapped array through which the mapped graphics 
  -- * resource \p resource. The value set in \p *pMipmappedArray may change every time 
  -- * that \p resource is mapped.
  -- *
  -- * If \p resource is not a texture then it cannot be accessed via a mipmapped array and
  -- * ::CUDA_ERROR_NOT_MAPPED_AS_ARRAY is returned.
  -- * If \p resource is not mapped then ::CUDA_ERROR_NOT_MAPPED is returned.
  -- *
  -- * \param pMipmappedArray - Returned mipmapped array through which \p resource may be accessed
  -- * \param resource        - Mapped resource to access
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE,
  -- * ::CUDA_ERROR_INVALID_HANDLE,
  -- * ::CUDA_ERROR_NOT_MAPPED,
  -- * ::CUDA_ERROR_NOT_MAPPED_AS_ARRAY
  -- * \notefnerr
  -- *
  -- * \sa ::cuGraphicsResourceGetMappedPointer
  --  

   function cuGraphicsResourceGetMappedMipmappedArray (pMipmappedArray : System.Address; resource : CUgraphicsResource) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:11002
   pragma Import (C, cuGraphicsResourceGetMappedMipmappedArray, "cuGraphicsResourceGetMappedMipmappedArray");

  --*
  -- * \brief Get a device pointer through which to access a mapped graphics resource.
  -- *
  -- * Returns in \p *pDevPtr a pointer through which the mapped graphics resource
  -- * \p resource may be accessed.
  -- * Returns in \p pSize the size of the memory in bytes which may be accessed from that pointer.
  -- * The value set in \p pPointer may change every time that \p resource is mapped.
  -- *
  -- * If \p resource is not a buffer then it cannot be accessed via a pointer and
  -- * ::CUDA_ERROR_NOT_MAPPED_AS_POINTER is returned.
  -- * If \p resource is not mapped then ::CUDA_ERROR_NOT_MAPPED is returned.
  -- * *
  -- * \param pDevPtr    - Returned pointer through which \p resource may be accessed
  -- * \param pSize      - Returned size of the buffer accessible starting at \p *pPointer
  -- * \param resource   - Mapped resource to access
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE,
  -- * ::CUDA_ERROR_INVALID_HANDLE,
  -- * ::CUDA_ERROR_NOT_MAPPED,
  -- * ::CUDA_ERROR_NOT_MAPPED_AS_POINTER
  -- * \notefnerr
  -- *
  -- * \sa
  -- * ::cuGraphicsMapResources,
  -- * ::cuGraphicsSubResourceGetMappedArray
  --  

   function cuGraphicsResourceGetMappedPointer_v2
     (pDevPtr : access CUdeviceptr;
      pSize : access stddef_h.size_t;
      resource : CUgraphicsResource) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:11038
   pragma Import (C, cuGraphicsResourceGetMappedPointer_v2, "cuGraphicsResourceGetMappedPointer_v2");

  --*
  -- * \brief Set usage flags for mapping a graphics resource
  -- *
  -- * Set \p flags for mapping the graphics resource \p resource.
  -- *
  -- * Changes to \p flags will take effect the next time \p resource is mapped.
  -- * The \p flags argument may be any of the following:
  -- * - ::CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE: Specifies no hints about how this
  -- *   resource will be used. It is therefore assumed that this resource will be
  -- *   read from and written to by CUDA kernels.  This is the default value.
  -- * - ::CU_GRAPHICS_MAP_RESOURCE_FLAGS_READONLY: Specifies that CUDA kernels which
  -- *   access this resource will not write to this resource.
  -- * - ::CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITEDISCARD: Specifies that CUDA kernels
  -- *   which access this resource will not read from this resource and will
  -- *   write over the entire contents of the resource, so none of the data
  -- *   previously stored in the resource will be preserved.
  -- *
  -- * If \p resource is presently mapped for access by CUDA then
  -- * ::CUDA_ERROR_ALREADY_MAPPED is returned.
  -- * If \p flags is not one of the above values then ::CUDA_ERROR_INVALID_VALUE is returned.
  -- *
  -- * \param resource - Registered resource to set flags for
  -- * \param flags    - Parameters for resource mapping
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_VALUE,
  -- * ::CUDA_ERROR_INVALID_HANDLE,
  -- * ::CUDA_ERROR_ALREADY_MAPPED
  -- * \notefnerr
  -- *
  -- * \sa
  -- * ::cuGraphicsMapResources
  --  

   function cuGraphicsResourceSetMapFlags_v2 (resource : CUgraphicsResource; flags : unsigned) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:11079
   pragma Import (C, cuGraphicsResourceSetMapFlags_v2, "cuGraphicsResourceSetMapFlags_v2");

  --*
  -- * \brief Map graphics resources for access by CUDA
  -- *
  -- * Maps the \p count graphics resources in \p resources for access by CUDA.
  -- *
  -- * The resources in \p resources may be accessed by CUDA until they
  -- * are unmapped. The graphics API from which \p resources were registered
  -- * should not access any resources while they are mapped by CUDA. If an
  -- * application does so, the results are undefined.
  -- *
  -- * This function provides the synchronization guarantee that any graphics calls
  -- * issued before ::cuGraphicsMapResources() will complete before any subsequent CUDA
  -- * work issued in \p stream begins.
  -- *
  -- * If \p resources includes any duplicate entries then ::CUDA_ERROR_INVALID_HANDLE is returned.
  -- * If any of \p resources are presently mapped for access by CUDA then ::CUDA_ERROR_ALREADY_MAPPED is returned.
  -- *
  -- * \param count      - Number of resources to map
  -- * \param resources  - Resources to map for CUDA usage
  -- * \param hStream    - Stream with which to synchronize
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_HANDLE,
  -- * ::CUDA_ERROR_ALREADY_MAPPED,
  -- * ::CUDA_ERROR_UNKNOWN
  -- * \note_null_stream
  -- * \notefnerr
  -- *
  -- * \sa
  -- * ::cuGraphicsResourceGetMappedPointer,
  -- * ::cuGraphicsSubResourceGetMappedArray,
  -- * ::cuGraphicsUnmapResources
  --  

   function cuGraphicsMapResources
     (count : unsigned;
      resources : System.Address;
      hStream : CUstream) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:11118
   pragma Import (C, cuGraphicsMapResources, "cuGraphicsMapResources");

  --*
  -- * \brief Unmap graphics resources.
  -- *
  -- * Unmaps the \p count graphics resources in \p resources.
  -- *
  -- * Once unmapped, the resources in \p resources may not be accessed by CUDA
  -- * until they are mapped again.
  -- *
  -- * This function provides the synchronization guarantee that any CUDA work issued
  -- * in \p stream before ::cuGraphicsUnmapResources() will complete before any
  -- * subsequently issued graphics work begins.
  -- *
  -- *
  -- * If \p resources includes any duplicate entries then ::CUDA_ERROR_INVALID_HANDLE is returned.
  -- * If any of \p resources are not presently mapped for access by CUDA then ::CUDA_ERROR_NOT_MAPPED is returned.
  -- *
  -- * \param count      - Number of resources to unmap
  -- * \param resources  - Resources to unmap
  -- * \param hStream    - Stream with which to synchronize
  -- *
  -- * \return
  -- * ::CUDA_SUCCESS,
  -- * ::CUDA_ERROR_DEINITIALIZED,
  -- * ::CUDA_ERROR_NOT_INITIALIZED,
  -- * ::CUDA_ERROR_INVALID_CONTEXT,
  -- * ::CUDA_ERROR_INVALID_HANDLE,
  -- * ::CUDA_ERROR_NOT_MAPPED,
  -- * ::CUDA_ERROR_UNKNOWN
  -- * \note_null_stream
  -- * \notefnerr
  -- *
  -- * \sa
  -- * ::cuGraphicsMapResources
  --  

   function cuGraphicsUnmapResources
     (count : unsigned;
      resources : System.Address;
      hStream : CUstream) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:11154
   pragma Import (C, cuGraphicsUnmapResources, "cuGraphicsUnmapResources");

  --* @}  
  -- END CUDA_GRAPHICS  
   function cuGetExportTable (ppExportTable : System.Address; pExportTableId : access constant CUuuid) return CUresult;  -- /usr/local/cuda-8.0/include/cuda.h:11158
   pragma Import (C, cuGetExportTable, "cuGetExportTable");

  --*
  -- * CUDA API versioning support
  --  

  --*
  -- * CUDA API made obselete at API version 3020
  --  

  --*< Source X in bytes  
  --*< Source Y  
  --*< Source memory type (host, device, array)  
  --*< Source host pointer  
  --*< Source device pointer  
  --*< Source array reference  
  --*< Source pitch (ignored when src is array)  
  --*< Destination X in bytes  
  --*< Destination Y  
  --*< Destination memory type (host, device, array)  
  --*< Destination host pointer  
  --*< Destination device pointer  
  --*< Destination array reference  
  --*< Destination pitch (ignored when dst is array)  
  --*< Width of 2D memory copy in bytes  
  --*< Height of 2D memory copy  
  --*< Source X in bytes  
  --*< Source Y  
  --*< Source Z  
  --*< Source LOD  
  --*< Source memory type (host, device, array)  
  --*< Source host pointer  
  --*< Source device pointer  
  --*< Source array reference  
  --*< Must be NULL  
  --*< Source pitch (ignored when src is array)  
  --*< Source height (ignored when src is array; may be 0 if Depth==1)  
  --*< Destination X in bytes  
  --*< Destination Y  
  --*< Destination Z  
  --*< Destination LOD  
  --*< Destination memory type (host, device, array)  
  --*< Destination host pointer  
  --*< Destination device pointer  
  --*< Destination array reference  
  --*< Must be NULL  
  --*< Destination pitch (ignored when dst is array)  
  --*< Destination height (ignored when dst is array; may be 0 if Depth==1)  
  --*< Width of 3D memory copy in bytes  
  --*< Height of 3D memory copy  
  --*< Depth of 3D memory copy  
  --*< Width of array  
  --*< Height of array  
  --*< Array format  
  --*< Channels per array element  
  --*< Width of 3D array  
  --*< Height of 3D array  
  --*< Depth of 3D array  
  --*< Array format  
  --*< Channels per array element  
  --*< Flags  
end cuda_h;

pragma Ada_2005;
pragma Style_Checks (Off);

with Interfaces.C; use Interfaces.C;
with System;
with stddef_h;

package driver_types_h is

   cudaHostAllocDefault : constant := 16#00#;  --  /usr/local/cuda-8.0/include/driver_types.h:75
   cudaHostAllocPortable : constant := 16#01#;  --  /usr/local/cuda-8.0/include/driver_types.h:76
   cudaHostAllocMapped : constant := 16#02#;  --  /usr/local/cuda-8.0/include/driver_types.h:77
   cudaHostAllocWriteCombined : constant := 16#04#;  --  /usr/local/cuda-8.0/include/driver_types.h:78

   cudaHostRegisterDefault : constant := 16#00#;  --  /usr/local/cuda-8.0/include/driver_types.h:80
   cudaHostRegisterPortable : constant := 16#01#;  --  /usr/local/cuda-8.0/include/driver_types.h:81
   cudaHostRegisterMapped : constant := 16#02#;  --  /usr/local/cuda-8.0/include/driver_types.h:82
   cudaHostRegisterIoMemory : constant := 16#04#;  --  /usr/local/cuda-8.0/include/driver_types.h:83

   cudaPeerAccessDefault : constant := 16#00#;  --  /usr/local/cuda-8.0/include/driver_types.h:85

   cudaStreamDefault : constant := 16#00#;  --  /usr/local/cuda-8.0/include/driver_types.h:87
   cudaStreamNonBlocking : constant := 16#01#;  --  /usr/local/cuda-8.0/include/driver_types.h:88
   --  unsupported macro: cudaStreamLegacy ((cudaStream_t)0x1)
   --  unsupported macro: cudaStreamPerThread ((cudaStream_t)0x2)

   cudaEventDefault : constant := 16#00#;  --  /usr/local/cuda-8.0/include/driver_types.h:110
   cudaEventBlockingSync : constant := 16#01#;  --  /usr/local/cuda-8.0/include/driver_types.h:111
   cudaEventDisableTiming : constant := 16#02#;  --  /usr/local/cuda-8.0/include/driver_types.h:112
   cudaEventInterprocess : constant := 16#04#;  --  /usr/local/cuda-8.0/include/driver_types.h:113

   cudaDeviceScheduleAuto : constant := 16#00#;  --  /usr/local/cuda-8.0/include/driver_types.h:115
   cudaDeviceScheduleSpin : constant := 16#01#;  --  /usr/local/cuda-8.0/include/driver_types.h:116
   cudaDeviceScheduleYield : constant := 16#02#;  --  /usr/local/cuda-8.0/include/driver_types.h:117
   cudaDeviceScheduleBlockingSync : constant := 16#04#;  --  /usr/local/cuda-8.0/include/driver_types.h:118
   cudaDeviceBlockingSync : constant := 16#04#;  --  /usr/local/cuda-8.0/include/driver_types.h:119

   cudaDeviceScheduleMask : constant := 16#07#;  --  /usr/local/cuda-8.0/include/driver_types.h:122
   cudaDeviceMapHost : constant := 16#08#;  --  /usr/local/cuda-8.0/include/driver_types.h:123
   cudaDeviceLmemResizeToMax : constant := 16#10#;  --  /usr/local/cuda-8.0/include/driver_types.h:124
   cudaDeviceMask : constant := 16#1f#;  --  /usr/local/cuda-8.0/include/driver_types.h:125

   cudaArrayDefault : constant := 16#00#;  --  /usr/local/cuda-8.0/include/driver_types.h:127
   cudaArrayLayered : constant := 16#01#;  --  /usr/local/cuda-8.0/include/driver_types.h:128
   cudaArraySurfaceLoadStore : constant := 16#02#;  --  /usr/local/cuda-8.0/include/driver_types.h:129
   cudaArrayCubemap : constant := 16#04#;  --  /usr/local/cuda-8.0/include/driver_types.h:130
   cudaArrayTextureGather : constant := 16#08#;  --  /usr/local/cuda-8.0/include/driver_types.h:131

   cudaIpcMemLazyEnablePeerAccess : constant := 16#01#;  --  /usr/local/cuda-8.0/include/driver_types.h:133

   cudaMemAttachGlobal : constant := 16#01#;  --  /usr/local/cuda-8.0/include/driver_types.h:135
   cudaMemAttachHost : constant := 16#02#;  --  /usr/local/cuda-8.0/include/driver_types.h:136
   cudaMemAttachSingle : constant := 16#04#;  --  /usr/local/cuda-8.0/include/driver_types.h:137

   cudaOccupancyDefault : constant := 16#00#;  --  /usr/local/cuda-8.0/include/driver_types.h:139
   cudaOccupancyDisableCachingOverride : constant := 16#01#;  --  /usr/local/cuda-8.0/include/driver_types.h:140
   --  unsupported macro: cudaCpuDeviceId ((int)-1)
   --  unsupported macro: cudaInvalidDeviceId ((int)-2)
   --  unsupported macro: cudaDevicePropDontCare { {'\0'}, 0, 0, 0, 0, 0, 0, {0, 0, 0}, {0, 0, 0}, 0, 0, -1, -1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, {0, 0}, {0, 0}, {0, 0, 0}, {0, 0}, {0, 0, 0}, {0, 0, 0}, 0, {0, 0}, {0, 0, 0}, {0, 0}, 0, {0, 0}, {0, 0, 0}, {0, 0}, {0, 0, 0}, 0, {0, 0}, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, }

   CUDA_IPC_HANDLE_SIZE : constant := 64;  --  /usr/local/cuda-8.0/include/driver_types.h:1451

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

  --*
  -- * \defgroup CUDART_TYPES Data types used by CUDA Runtime
  -- * \ingroup CUDART
  -- *
  -- * @{
  --  

  --******************************************************************************
  --*                                                                              *
  --*  TYPE DEFINITIONS USED BY RUNTIME API                                        *
  --*                                                                              *
  --****************************************************************************** 

  --*
  -- * Legacy stream handle
  -- *
  -- * Stream handle that can be passed as a cudaStream_t to use an implicit stream
  -- * with legacy synchronization behavior.
  -- *
  -- * See details of the \link_sync_behavior
  --  

  --*
  -- * Per-thread stream handle
  -- *
  -- * Stream handle that can be passed as a cudaStream_t to use an implicit stream
  -- * with per-thread synchronization behavior.
  -- *
  -- * See details of the \link_sync_behavior
  --  

  --******************************************************************************
  --*                                                                              *
  --*                                                                              *
  --*                                                                              *
  --****************************************************************************** 

  --*
  -- * CUDA error types
  --  

   subtype cudaError is unsigned;
   cudaSuccess : constant cudaError := 0;
   cudaErrorMissingConfiguration : constant cudaError := 1;
   cudaErrorMemoryAllocation : constant cudaError := 2;
   cudaErrorInitializationError : constant cudaError := 3;
   cudaErrorLaunchFailure : constant cudaError := 4;
   cudaErrorPriorLaunchFailure : constant cudaError := 5;
   cudaErrorLaunchTimeout : constant cudaError := 6;
   cudaErrorLaunchOutOfResources : constant cudaError := 7;
   cudaErrorInvalidDeviceFunction : constant cudaError := 8;
   cudaErrorInvalidConfiguration : constant cudaError := 9;
   cudaErrorInvalidDevice : constant cudaError := 10;
   cudaErrorInvalidValue : constant cudaError := 11;
   cudaErrorInvalidPitchValue : constant cudaError := 12;
   cudaErrorInvalidSymbol : constant cudaError := 13;
   cudaErrorMapBufferObjectFailed : constant cudaError := 14;
   cudaErrorUnmapBufferObjectFailed : constant cudaError := 15;
   cudaErrorInvalidHostPointer : constant cudaError := 16;
   cudaErrorInvalidDevicePointer : constant cudaError := 17;
   cudaErrorInvalidTexture : constant cudaError := 18;
   cudaErrorInvalidTextureBinding : constant cudaError := 19;
   cudaErrorInvalidChannelDescriptor : constant cudaError := 20;
   cudaErrorInvalidMemcpyDirection : constant cudaError := 21;
   cudaErrorAddressOfConstant : constant cudaError := 22;
   cudaErrorTextureFetchFailed : constant cudaError := 23;
   cudaErrorTextureNotBound : constant cudaError := 24;
   cudaErrorSynchronizationError : constant cudaError := 25;
   cudaErrorInvalidFilterSetting : constant cudaError := 26;
   cudaErrorInvalidNormSetting : constant cudaError := 27;
   cudaErrorMixedDeviceExecution : constant cudaError := 28;
   cudaErrorCudartUnloading : constant cudaError := 29;
   cudaErrorUnknown : constant cudaError := 30;
   cudaErrorNotYetImplemented : constant cudaError := 31;
   cudaErrorMemoryValueTooLarge : constant cudaError := 32;
   cudaErrorInvalidResourceHandle : constant cudaError := 33;
   cudaErrorNotReady : constant cudaError := 34;
   cudaErrorInsufficientDriver : constant cudaError := 35;
   cudaErrorSetOnActiveProcess : constant cudaError := 36;
   cudaErrorInvalidSurface : constant cudaError := 37;
   cudaErrorNoDevice : constant cudaError := 38;
   cudaErrorECCUncorrectable : constant cudaError := 39;
   cudaErrorSharedObjectSymbolNotFound : constant cudaError := 40;
   cudaErrorSharedObjectInitFailed : constant cudaError := 41;
   cudaErrorUnsupportedLimit : constant cudaError := 42;
   cudaErrorDuplicateVariableName : constant cudaError := 43;
   cudaErrorDuplicateTextureName : constant cudaError := 44;
   cudaErrorDuplicateSurfaceName : constant cudaError := 45;
   cudaErrorDevicesUnavailable : constant cudaError := 46;
   cudaErrorInvalidKernelImage : constant cudaError := 47;
   cudaErrorNoKernelImageForDevice : constant cudaError := 48;
   cudaErrorIncompatibleDriverContext : constant cudaError := 49;
   cudaErrorPeerAccessAlreadyEnabled : constant cudaError := 50;
   cudaErrorPeerAccessNotEnabled : constant cudaError := 51;
   cudaErrorDeviceAlreadyInUse : constant cudaError := 54;
   cudaErrorProfilerDisabled : constant cudaError := 55;
   cudaErrorProfilerNotInitialized : constant cudaError := 56;
   cudaErrorProfilerAlreadyStarted : constant cudaError := 57;
   cudaErrorProfilerAlreadyStopped : constant cudaError := 58;
   cudaErrorAssert : constant cudaError := 59;
   cudaErrorTooManyPeers : constant cudaError := 60;
   cudaErrorHostMemoryAlreadyRegistered : constant cudaError := 61;
   cudaErrorHostMemoryNotRegistered : constant cudaError := 62;
   cudaErrorOperatingSystem : constant cudaError := 63;
   cudaErrorPeerAccessUnsupported : constant cudaError := 64;
   cudaErrorLaunchMaxDepthExceeded : constant cudaError := 65;
   cudaErrorLaunchFileScopedTex : constant cudaError := 66;
   cudaErrorLaunchFileScopedSurf : constant cudaError := 67;
   cudaErrorSyncDepthExceeded : constant cudaError := 68;
   cudaErrorLaunchPendingCountExceeded : constant cudaError := 69;
   cudaErrorNotPermitted : constant cudaError := 70;
   cudaErrorNotSupported : constant cudaError := 71;
   cudaErrorHardwareStackError : constant cudaError := 72;
   cudaErrorIllegalInstruction : constant cudaError := 73;
   cudaErrorMisalignedAddress : constant cudaError := 74;
   cudaErrorInvalidAddressSpace : constant cudaError := 75;
   cudaErrorInvalidPc : constant cudaError := 76;
   cudaErrorIllegalAddress : constant cudaError := 77;
   cudaErrorInvalidPtx : constant cudaError := 78;
   cudaErrorInvalidGraphicsContext : constant cudaError := 79;
   cudaErrorNvlinkUncorrectable : constant cudaError := 80;
   cudaErrorStartupFailure : constant cudaError := 127;
   cudaErrorApiFailureBase : constant cudaError := 10000;  -- /usr/local/cuda-8.0/include/driver_types.h:156

  --*
  --     * The API call returned with no errors. In the case of query calls, this
  --     * can also mean that the operation being queried is complete (see
  --     * ::cudaEventQuery() and ::cudaStreamQuery()).
  --      

  --*
  --     * The device function being invoked (usually via ::cudaLaunchKernel()) was not
  --     * previously configured via the ::cudaConfigureCall() function.
  --      

  --*
  --     * The API call failed because it was unable to allocate enough memory to
  --     * perform the requested operation.
  --      

  --*
  --     * The API call failed because the CUDA driver and runtime could not be
  --     * initialized.
  --      

  --*
  --     * An exception occurred on the device while executing a kernel. Common
  --     * causes include dereferencing an invalid device pointer and accessing
  --     * out of bounds shared memory. The device cannot be used until
  --     * ::cudaThreadExit() is called. All existing device memory allocations
  --     * are invalid and must be reconstructed if the program is to continue
  --     * using CUDA.
  --      

  --*
  --     * This indicated that a previous kernel launch failed. This was previously
  --     * used for device emulation of kernel launches.
  --     * \deprecated
  --     * This error return is deprecated as of CUDA 3.1. Device emulation mode was
  --     * removed with the CUDA 3.1 release.
  --      

  --*
  --     * This indicates that the device kernel took too long to execute. This can
  --     * only occur if timeouts are enabled - see the device property
  --     * \ref ::cudaDeviceProp::kernelExecTimeoutEnabled "kernelExecTimeoutEnabled"
  --     * for more information.
  --     * This leaves the process in an inconsistent state and any further CUDA work
  --     * will return the same error. To continue using CUDA, the process must be terminated
  --     * and relaunched.
  --      

  --*
  --     * This indicates that a launch did not occur because it did not have
  --     * appropriate resources. Although this error is similar to
  --     * ::cudaErrorInvalidConfiguration, this error usually indicates that the
  --     * user has attempted to pass too many arguments to the device kernel, or the
  --     * kernel launch specifies too many threads for the kernel's register count.
  --      

  --*
  --     * The requested device function does not exist or is not compiled for the
  --     * proper device architecture.
  --      

  --*
  --     * This indicates that a kernel launch is requesting resources that can
  --     * never be satisfied by the current device. Requesting more shared memory
  --     * per block than the device supports will trigger this error, as will
  --     * requesting too many threads or blocks. See ::cudaDeviceProp for more
  --     * device limitations.
  --      

  --*
  --     * This indicates that the device ordinal supplied by the user does not
  --     * correspond to a valid CUDA device.
  --      

  --*
  --     * This indicates that one or more of the parameters passed to the API call
  --     * is not within an acceptable range of values.
  --      

  --*
  --     * This indicates that one or more of the pitch-related parameters passed
  --     * to the API call is not within the acceptable range for pitch.
  --      

  --*
  --     * This indicates that the symbol name/identifier passed to the API call
  --     * is not a valid name or identifier.
  --      

  --*
  --     * This indicates that the buffer object could not be mapped.
  --      

  --*
  --     * This indicates that the buffer object could not be unmapped.
  --      

  --*
  --     * This indicates that at least one host pointer passed to the API call is
  --     * not a valid host pointer.
  --      

  --*
  --     * This indicates that at least one device pointer passed to the API call is
  --     * not a valid device pointer.
  --      

  --*
  --     * This indicates that the texture passed to the API call is not a valid
  --     * texture.
  --      

  --*
  --     * This indicates that the texture binding is not valid. This occurs if you
  --     * call ::cudaGetTextureAlignmentOffset() with an unbound texture.
  --      

  --*
  --     * This indicates that the channel descriptor passed to the API call is not
  --     * valid. This occurs if the format is not one of the formats specified by
  --     * ::cudaChannelFormatKind, or if one of the dimensions is invalid.
  --      

  --*
  --     * This indicates that the direction of the memcpy passed to the API call is
  --     * not one of the types specified by ::cudaMemcpyKind.
  --      

  --*
  --     * This indicated that the user has taken the address of a constant variable,
  --     * which was forbidden up until the CUDA 3.1 release.
  --     * \deprecated
  --     * This error return is deprecated as of CUDA 3.1. Variables in constant
  --     * memory may now have their address taken by the runtime via
  --     * ::cudaGetSymbolAddress().
  --      

  --*
  --     * This indicated that a texture fetch was not able to be performed.
  --     * This was previously used for device emulation of texture operations.
  --     * \deprecated
  --     * This error return is deprecated as of CUDA 3.1. Device emulation mode was
  --     * removed with the CUDA 3.1 release.
  --      

  --*
  --     * This indicated that a texture was not bound for access.
  --     * This was previously used for device emulation of texture operations.
  --     * \deprecated
  --     * This error return is deprecated as of CUDA 3.1. Device emulation mode was
  --     * removed with the CUDA 3.1 release.
  --      

  --*
  --     * This indicated that a synchronization operation had failed.
  --     * This was previously used for some device emulation functions.
  --     * \deprecated
  --     * This error return is deprecated as of CUDA 3.1. Device emulation mode was
  --     * removed with the CUDA 3.1 release.
  --      

  --*
  --     * This indicates that a non-float texture was being accessed with linear
  --     * filtering. This is not supported by CUDA.
  --      

  --*
  --     * This indicates that an attempt was made to read a non-float texture as a
  --     * normalized float. This is not supported by CUDA.
  --      

  --*
  --     * Mixing of device and device emulation code was not allowed.
  --     * \deprecated
  --     * This error return is deprecated as of CUDA 3.1. Device emulation mode was
  --     * removed with the CUDA 3.1 release.
  --      

  --*
  --     * This indicates that a CUDA Runtime API call cannot be executed because
  --     * it is being called during process shut down, at a point in time after
  --     * CUDA driver has been unloaded.
  --      

  --*
  --     * This indicates that an unknown internal error has occurred.
  --      

  --*
  --     * This indicates that the API call is not yet implemented. Production
  --     * releases of CUDA will never return this error.
  --     * \deprecated
  --     * This error return is deprecated as of CUDA 4.1.
  --      

  --*
  --     * This indicated that an emulated device pointer exceeded the 32-bit address
  --     * range.
  --     * \deprecated
  --     * This error return is deprecated as of CUDA 3.1. Device emulation mode was
  --     * removed with the CUDA 3.1 release.
  --      

  --*
  --     * This indicates that a resource handle passed to the API call was not
  --     * valid. Resource handles are opaque types like ::cudaStream_t and
  --     * ::cudaEvent_t.
  --      

  --*
  --     * This indicates that asynchronous operations issued previously have not
  --     * completed yet. This result is not actually an error, but must be indicated
  --     * differently than ::cudaSuccess (which indicates completion). Calls that
  --     * may return this value include ::cudaEventQuery() and ::cudaStreamQuery().
  --      

  --*
  --     * This indicates that the installed NVIDIA CUDA driver is older than the
  --     * CUDA runtime library. This is not a supported configuration. Users should
  --     * install an updated NVIDIA display driver to allow the application to run.
  --      

  --*
  --     * This indicates that the user has called ::cudaSetValidDevices(),
  --     * ::cudaSetDeviceFlags(), ::cudaD3D9SetDirect3DDevice(),
  --     * ::cudaD3D10SetDirect3DDevice, ::cudaD3D11SetDirect3DDevice(), or
  --     * ::cudaVDPAUSetVDPAUDevice() after initializing the CUDA runtime by
  --     * calling non-device management operations (allocating memory and
  --     * launching kernels are examples of non-device management operations).
  --     * This error can also be returned if using runtime/driver
  --     * interoperability and there is an existing ::CUcontext active on the
  --     * host thread.
  --      

  --*
  --     * This indicates that the surface passed to the API call is not a valid
  --     * surface.
  --      

  --*
  --     * This indicates that no CUDA-capable devices were detected by the installed
  --     * CUDA driver.
  --      

  --*
  --     * This indicates that an uncorrectable ECC error was detected during
  --     * execution.
  --      

  --*
  --     * This indicates that a link to a shared object failed to resolve.
  --      

  --*
  --     * This indicates that initialization of a shared object failed.
  --      

  --*
  --     * This indicates that the ::cudaLimit passed to the API call is not
  --     * supported by the active device.
  --      

  --*
  --     * This indicates that multiple global or constant variables (across separate
  --     * CUDA source files in the application) share the same string name.
  --      

  --*
  --     * This indicates that multiple textures (across separate CUDA source
  --     * files in the application) share the same string name.
  --      

  --*
  --     * This indicates that multiple surfaces (across separate CUDA source
  --     * files in the application) share the same string name.
  --      

  --*
  --     * This indicates that all CUDA devices are busy or unavailable at the current
  --     * time. Devices are often busy/unavailable due to use of
  --     * ::cudaComputeModeExclusive, ::cudaComputeModeProhibited or when long
  --     * running CUDA kernels have filled up the GPU and are blocking new work
  --     * from starting. They can also be unavailable due to memory constraints
  --     * on a device that already has active CUDA work being performed.
  --      

  --*
  --     * This indicates that the device kernel image is invalid.
  --      

  --*
  --     * This indicates that there is no kernel image available that is suitable
  --     * for the device. This can occur when a user specifies code generation
  --     * options for a particular CUDA source file that do not include the
  --     * corresponding device configuration.
  --      

  --*
  --     * This indicates that the current context is not compatible with this
  --     * the CUDA Runtime. This can only occur if you are using CUDA
  --     * Runtime/Driver interoperability and have created an existing Driver
  --     * context using the driver API. The Driver context may be incompatible
  --     * either because the Driver context was created using an older version 
  --     * of the API, because the Runtime API call expects a primary driver 
  --     * context and the Driver context is not primary, or because the Driver 
  --     * context has been destroyed. Please see \ref CUDART_DRIVER "Interactions 
  --     * with the CUDA Driver API" for more information.
  --      

  --*
  --     * This error indicates that a call to ::cudaDeviceEnablePeerAccess() is
  --     * trying to re-enable peer addressing on from a context which has already
  --     * had peer addressing enabled.
  --      

  --*
  --     * This error indicates that ::cudaDeviceDisablePeerAccess() is trying to 
  --     * disable peer addressing which has not been enabled yet via 
  --     * ::cudaDeviceEnablePeerAccess().
  --      

  --*
  --     * This indicates that a call tried to access an exclusive-thread device that 
  --     * is already in use by a different thread.
  --      

  --*
  --     * This indicates profiler is not initialized for this run. This can
  --     * happen when the application is running with external profiling tools
  --     * like visual profiler.
  --      

  --*
  --     * \deprecated
  --     * This error return is deprecated as of CUDA 5.0. It is no longer an error
  --     * to attempt to enable/disable the profiling via ::cudaProfilerStart or
  --     * ::cudaProfilerStop without initialization.
  --      

  --*
  --     * \deprecated
  --     * This error return is deprecated as of CUDA 5.0. It is no longer an error
  --     * to call cudaProfilerStart() when profiling is already enabled.
  --      

  --*
  --     * \deprecated
  --     * This error return is deprecated as of CUDA 5.0. It is no longer an error
  --     * to call cudaProfilerStop() when profiling is already disabled.
  --      

  --*
  --     * An assert triggered in device code during kernel execution. The device
  --     * cannot be used again until ::cudaThreadExit() is called. All existing 
  --     * allocations are invalid and must be reconstructed if the program is to
  --     * continue using CUDA. 
  --      

  --*
  --     * This error indicates that the hardware resources required to enable
  --     * peer access have been exhausted for one or more of the devices 
  --     * passed to ::cudaEnablePeerAccess().
  --      

  --*
  --     * This error indicates that the memory range passed to ::cudaHostRegister()
  --     * has already been registered.
  --      

  --*
  --     * This error indicates that the pointer passed to ::cudaHostUnregister()
  --     * does not correspond to any currently registered memory region.
  --      

  --*
  --     * This error indicates that an OS call failed.
  --      

  --*
  --     * This error indicates that P2P access is not supported across the given
  --     * devices.
  --      

  --*
  --     * This error indicates that a device runtime grid launch did not occur 
  --     * because the depth of the child grid would exceed the maximum supported
  --     * number of nested grid launches. 
  --      

  --*
  --     * This error indicates that a grid launch did not occur because the kernel 
  --     * uses file-scoped textures which are unsupported by the device runtime. 
  --     * Kernels launched via the device runtime only support textures created with 
  --     * the Texture Object API's.
  --      

  --*
  --     * This error indicates that a grid launch did not occur because the kernel 
  --     * uses file-scoped surfaces which are unsupported by the device runtime.
  --     * Kernels launched via the device runtime only support surfaces created with
  --     * the Surface Object API's.
  --      

  --*
  --     * This error indicates that a call to ::cudaDeviceSynchronize made from
  --     * the device runtime failed because the call was made at grid depth greater
  --     * than than either the default (2 levels of grids) or user specified device 
  --     * limit ::cudaLimitDevRuntimeSyncDepth. To be able to synchronize on 
  --     * launched grids at a greater depth successfully, the maximum nested 
  --     * depth at which ::cudaDeviceSynchronize will be called must be specified 
  --     * with the ::cudaLimitDevRuntimeSyncDepth limit to the ::cudaDeviceSetLimit
  --     * api before the host-side launch of a kernel using the device runtime. 
  --     * Keep in mind that additional levels of sync depth require the runtime 
  --     * to reserve large amounts of device memory that cannot be used for 
  --     * user allocations.
  --      

  --*
  --     * This error indicates that a device runtime grid launch failed because
  --     * the launch would exceed the limit ::cudaLimitDevRuntimePendingLaunchCount.
  --     * For this launch to proceed successfully, ::cudaDeviceSetLimit must be
  --     * called to set the ::cudaLimitDevRuntimePendingLaunchCount to be higher 
  --     * than the upper bound of outstanding launches that can be issued to the
  --     * device runtime. Keep in mind that raising the limit of pending device
  --     * runtime launches will require the runtime to reserve device memory that
  --     * cannot be used for user allocations.
  --      

  --*
  --     * This error indicates the attempted operation is not permitted.
  --      

  --*
  --     * This error indicates the attempted operation is not supported
  --     * on the current system or device.
  --      

  --*
  --     * Device encountered an error in the call stack during kernel execution,
  --     * possibly due to stack corruption or exceeding the stack size limit.
  --     * This leaves the process in an inconsistent state and any further CUDA work
  --     * will return the same error. To continue using CUDA, the process must be terminated
  --     * and relaunched.
  --      

  --*
  --     * The device encountered an illegal instruction during kernel execution
  --     * This leaves the process in an inconsistent state and any further CUDA work
  --     * will return the same error. To continue using CUDA, the process must be terminated
  --     * and relaunched.
  --      

  --*
  --     * The device encountered a load or store instruction
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
  --     * The device encountered an invalid program counter.
  --     * This leaves the process in an inconsistent state and any further CUDA work
  --     * will return the same error. To continue using CUDA, the process must be terminated
  --     * and relaunched.
  --      

  --*
  --     * The device encountered a load or store instruction on an invalid memory address.
  --     * This leaves the process in an inconsistent state and any further CUDA work
  --     * will return the same error. To continue using CUDA, the process must be terminated
  --     * and relaunched.
  --      

  --*
  --     * A PTX compilation failed. The runtime may fall back to compiling PTX if
  --     * an application does not contain a suitable binary for the current device.
  --      

  --*
  --     * This indicates an error with the OpenGL or DirectX context.
  --      

  --*
  --     * This indicates that an uncorrectable NVLink error was detected during the
  --     * execution.
  --      

  --*
  --     * This indicates an internal startup failure in the CUDA runtime.
  --      

  --*
  --     * Any unhandled CUDA driver error is added to this value and returned via
  --     * the runtime. Production releases of CUDA should not return such errors.
  --     * \deprecated
  --     * This error return is deprecated as of CUDA 4.1.
  --      

  --*
  -- * Channel format kind
  --  

   type cudaChannelFormatKind is 
     (cudaChannelFormatKindSigned,
      cudaChannelFormatKindUnsigned,
      cudaChannelFormatKindFloat,
      cudaChannelFormatKindNone);
   pragma Convention (C, cudaChannelFormatKind);  -- /usr/local/cuda-8.0/include/driver_types.h:752

  --*< Signed channel format  
  --*< Unsigned channel format  
  --*< Float channel format  
  --*< No channel format  
  --*
  -- * CUDA Channel format descriptor
  --  

  --*< x  
   type cudaChannelFormatDesc is record
      x : aliased int;  -- /usr/local/cuda-8.0/include/driver_types.h:765
      y : aliased int;  -- /usr/local/cuda-8.0/include/driver_types.h:766
      z : aliased int;  -- /usr/local/cuda-8.0/include/driver_types.h:767
      w : aliased int;  -- /usr/local/cuda-8.0/include/driver_types.h:768
      f : aliased cudaChannelFormatKind;  -- /usr/local/cuda-8.0/include/driver_types.h:769
   end record;
   pragma Convention (C_Pass_By_Copy, cudaChannelFormatDesc);  -- /usr/local/cuda-8.0/include/driver_types.h:763

  --*< y  
  --*< z  
  --*< w  
  --*< Channel format kind  
  --*
  -- * CUDA array
  --  

   --  skipped empty struct cudaArray

   type cudaArray_t is new System.Address;  -- /usr/local/cuda-8.0/include/driver_types.h:775

  --*
  -- * CUDA array (as source copy argument)
  --  

   type cudaArray_const_t is new System.Address;  -- /usr/local/cuda-8.0/include/driver_types.h:780

  --*
  -- * CUDA mipmapped array
  --  

   --  skipped empty struct cudaMipmappedArray

   type cudaMipmappedArray_t is new System.Address;  -- /usr/local/cuda-8.0/include/driver_types.h:787

  --*
  -- * CUDA mipmapped array (as source argument)
  --  

   type cudaMipmappedArray_const_t is new System.Address;  -- /usr/local/cuda-8.0/include/driver_types.h:792

  --*
  -- * CUDA memory types
  --  

   subtype cudaMemoryType is unsigned;
   cudaMemoryTypeHost : constant cudaMemoryType := 1;
   cudaMemoryTypeDevice : constant cudaMemoryType := 2;  -- /usr/local/cuda-8.0/include/driver_types.h:799

  --*< Host memory  
  --*< Device memory  
  --*
  -- * CUDA memory copy types
  --  

   type cudaMemcpyKind is 
     (cudaMemcpyHostToHost,
      cudaMemcpyHostToDevice,
      cudaMemcpyDeviceToHost,
      cudaMemcpyDeviceToDevice,
      cudaMemcpyDefault);
   pragma Convention (C, cudaMemcpyKind);  -- /usr/local/cuda-8.0/include/driver_types.h:808

  --*< Host   -> Host  
  --*< Host   -> Device  
  --*< Device -> Host  
  --*< Device -> Device  
  --*< Direction of the transfer is inferred from the pointer values. Requires unified virtual addressing  
  --*
  -- * CUDA Pitched memory pointer
  -- *
  -- * \sa ::make_cudaPitchedPtr
  --  

  --*< Pointer to allocated memory  
   type cudaPitchedPtr is record
      ptr : System.Address;  -- /usr/local/cuda-8.0/include/driver_types.h:824
      pitch : aliased stddef_h.size_t;  -- /usr/local/cuda-8.0/include/driver_types.h:825
      xsize : aliased stddef_h.size_t;  -- /usr/local/cuda-8.0/include/driver_types.h:826
      ysize : aliased stddef_h.size_t;  -- /usr/local/cuda-8.0/include/driver_types.h:827
   end record;
   pragma Convention (C_Pass_By_Copy, cudaPitchedPtr);  -- /usr/local/cuda-8.0/include/driver_types.h:822

  --*< Pitch of allocated memory in bytes  
  --*< Logical width of allocation in elements  
  --*< Logical height of allocation in elements  
  --*
  -- * CUDA extent
  -- *
  -- * \sa ::make_cudaExtent
  --  

  --*< Width in elements when referring to array memory, in bytes when referring to linear memory  
   type cudaExtent is record
      width : aliased stddef_h.size_t;  -- /usr/local/cuda-8.0/include/driver_types.h:837
      height : aliased stddef_h.size_t;  -- /usr/local/cuda-8.0/include/driver_types.h:838
      depth : aliased stddef_h.size_t;  -- /usr/local/cuda-8.0/include/driver_types.h:839
   end record;
   pragma Convention (C_Pass_By_Copy, cudaExtent);  -- /usr/local/cuda-8.0/include/driver_types.h:835

  --*< Height in elements  
  --*< Depth in elements  
  --*
  -- * CUDA 3D position
  -- *
  -- * \sa ::make_cudaPos
  --  

  --*< x  
   type cudaPos is record
      x : aliased stddef_h.size_t;  -- /usr/local/cuda-8.0/include/driver_types.h:849
      y : aliased stddef_h.size_t;  -- /usr/local/cuda-8.0/include/driver_types.h:850
      z : aliased stddef_h.size_t;  -- /usr/local/cuda-8.0/include/driver_types.h:851
   end record;
   pragma Convention (C_Pass_By_Copy, cudaPos);  -- /usr/local/cuda-8.0/include/driver_types.h:847

  --*< y  
  --*< z  
  --*
  -- * CUDA 3D memory copying parameters
  --  

  --*< Source memory address  
   type cudaMemcpy3DParms is record
      srcArray : cudaArray_t;  -- /usr/local/cuda-8.0/include/driver_types.h:859
      srcPos : aliased cudaPos;  -- /usr/local/cuda-8.0/include/driver_types.h:860
      srcPtr : aliased cudaPitchedPtr;  -- /usr/local/cuda-8.0/include/driver_types.h:861
      dstArray : cudaArray_t;  -- /usr/local/cuda-8.0/include/driver_types.h:863
      dstPos : aliased cudaPos;  -- /usr/local/cuda-8.0/include/driver_types.h:864
      dstPtr : aliased cudaPitchedPtr;  -- /usr/local/cuda-8.0/include/driver_types.h:865
      extent : aliased cudaExtent;  -- /usr/local/cuda-8.0/include/driver_types.h:867
      kind : aliased cudaMemcpyKind;  -- /usr/local/cuda-8.0/include/driver_types.h:868
   end record;
   pragma Convention (C_Pass_By_Copy, cudaMemcpy3DParms);  -- /usr/local/cuda-8.0/include/driver_types.h:857

  --*< Source position offset  
  --*< Pitched source memory address  
  --*< Destination memory address  
  --*< Destination position offset  
  --*< Pitched destination memory address  
  --*< Requested memory copy size  
  --*< Type of transfer  
  --*
  -- * CUDA 3D cross-device memory copying parameters
  --  

  --*< Source memory address  
   type cudaMemcpy3DPeerParms is record
      srcArray : cudaArray_t;  -- /usr/local/cuda-8.0/include/driver_types.h:876
      srcPos : aliased cudaPos;  -- /usr/local/cuda-8.0/include/driver_types.h:877
      srcPtr : aliased cudaPitchedPtr;  -- /usr/local/cuda-8.0/include/driver_types.h:878
      srcDevice : aliased int;  -- /usr/local/cuda-8.0/include/driver_types.h:879
      dstArray : cudaArray_t;  -- /usr/local/cuda-8.0/include/driver_types.h:881
      dstPos : aliased cudaPos;  -- /usr/local/cuda-8.0/include/driver_types.h:882
      dstPtr : aliased cudaPitchedPtr;  -- /usr/local/cuda-8.0/include/driver_types.h:883
      dstDevice : aliased int;  -- /usr/local/cuda-8.0/include/driver_types.h:884
      extent : aliased cudaExtent;  -- /usr/local/cuda-8.0/include/driver_types.h:886
   end record;
   pragma Convention (C_Pass_By_Copy, cudaMemcpy3DPeerParms);  -- /usr/local/cuda-8.0/include/driver_types.h:874

  --*< Source position offset  
  --*< Pitched source memory address  
  --*< Source device  
  --*< Destination memory address  
  --*< Destination position offset  
  --*< Pitched destination memory address  
  --*< Destination device  
  --*< Requested memory copy size  
  --*
  -- * CUDA graphics interop resource
  --  

   --  skipped empty struct cudaGraphicsResource

  --*
  -- * CUDA graphics interop register flags
  --  

   subtype cudaGraphicsRegisterFlags is unsigned;
   cudaGraphicsRegisterFlagsNone : constant cudaGraphicsRegisterFlags := 0;
   cudaGraphicsRegisterFlagsReadOnly : constant cudaGraphicsRegisterFlags := 1;
   cudaGraphicsRegisterFlagsWriteDiscard : constant cudaGraphicsRegisterFlags := 2;
   cudaGraphicsRegisterFlagsSurfaceLoadStore : constant cudaGraphicsRegisterFlags := 4;
   cudaGraphicsRegisterFlagsTextureGather : constant cudaGraphicsRegisterFlags := 8;  -- /usr/local/cuda-8.0/include/driver_types.h:897

  --*< Default  
  --*< CUDA will not write to this resource  
  --*< CUDA will only write to and will not read from this resource  
  --*< CUDA will bind this resource to a surface reference  
  --*< CUDA will perform texture gather operations on this resource  
  --*
  -- * CUDA graphics interop map flags
  --  

   type cudaGraphicsMapFlags is 
     (cudaGraphicsMapFlagsNone,
      cudaGraphicsMapFlagsReadOnly,
      cudaGraphicsMapFlagsWriteDiscard);
   pragma Convention (C, cudaGraphicsMapFlags);  -- /usr/local/cuda-8.0/include/driver_types.h:909

  --*< Default; Assume resource can be read/written  
  --*< CUDA will not write to this resource  
  --*< CUDA will only write to and will not read from this resource  
  --*
  -- * CUDA graphics interop array indices for cube maps
  --  

   type cudaGraphicsCubeFace is 
     (cudaGraphicsCubeFacePositiveX,
      cudaGraphicsCubeFaceNegativeX,
      cudaGraphicsCubeFacePositiveY,
      cudaGraphicsCubeFaceNegativeY,
      cudaGraphicsCubeFacePositiveZ,
      cudaGraphicsCubeFaceNegativeZ);
   pragma Convention (C, cudaGraphicsCubeFace);  -- /usr/local/cuda-8.0/include/driver_types.h:919

  --*< Positive X face of cubemap  
  --*< Negative X face of cubemap  
  --*< Positive Y face of cubemap  
  --*< Negative Y face of cubemap  
  --*< Positive Z face of cubemap  
  --*< Negative Z face of cubemap  
  --*
  -- * CUDA resource types
  --  

   type cudaResourceType is 
     (cudaResourceTypeArray,
      cudaResourceTypeMipmappedArray,
      cudaResourceTypeLinear,
      cudaResourceTypePitch2D);
   pragma Convention (C, cudaResourceType);  -- /usr/local/cuda-8.0/include/driver_types.h:932

  --*< Array resource  
  --*< Mipmapped array resource  
  --*< Linear resource  
  --*< Pitch 2D resource  
  --*
  -- * CUDA texture resource view formats
  --  

   type cudaResourceViewFormat is 
     (cudaResViewFormatNone,
      cudaResViewFormatUnsignedChar1,
      cudaResViewFormatUnsignedChar2,
      cudaResViewFormatUnsignedChar4,
      cudaResViewFormatSignedChar1,
      cudaResViewFormatSignedChar2,
      cudaResViewFormatSignedChar4,
      cudaResViewFormatUnsignedShort1,
      cudaResViewFormatUnsignedShort2,
      cudaResViewFormatUnsignedShort4,
      cudaResViewFormatSignedShort1,
      cudaResViewFormatSignedShort2,
      cudaResViewFormatSignedShort4,
      cudaResViewFormatUnsignedInt1,
      cudaResViewFormatUnsignedInt2,
      cudaResViewFormatUnsignedInt4,
      cudaResViewFormatSignedInt1,
      cudaResViewFormatSignedInt2,
      cudaResViewFormatSignedInt4,
      cudaResViewFormatHalf1,
      cudaResViewFormatHalf2,
      cudaResViewFormatHalf4,
      cudaResViewFormatFloat1,
      cudaResViewFormatFloat2,
      cudaResViewFormatFloat4,
      cudaResViewFormatUnsignedBlockCompressed1,
      cudaResViewFormatUnsignedBlockCompressed2,
      cudaResViewFormatUnsignedBlockCompressed3,
      cudaResViewFormatUnsignedBlockCompressed4,
      cudaResViewFormatSignedBlockCompressed4,
      cudaResViewFormatUnsignedBlockCompressed5,
      cudaResViewFormatSignedBlockCompressed5,
      cudaResViewFormatUnsignedBlockCompressed6H,
      cudaResViewFormatSignedBlockCompressed6H,
      cudaResViewFormatUnsignedBlockCompressed7);
   pragma Convention (C, cudaResourceViewFormat);  -- /usr/local/cuda-8.0/include/driver_types.h:943

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
  --*
  -- * CUDA resource descriptor
  --  

  --*< Resource type  
   type cudaResourceDesc;
   type anon_0;
   type anon_1 is record
      c_array : cudaArray_t;  -- /usr/local/cuda-8.0/include/driver_types.h:990
   end record;
   pragma Convention (C_Pass_By_Copy, anon_1);
   type anon_2 is record
      mipmap : cudaMipmappedArray_t;  -- /usr/local/cuda-8.0/include/driver_types.h:993
   end record;
   pragma Convention (C_Pass_By_Copy, anon_2);
   type anon_3 is record
      devPtr : System.Address;  -- /usr/local/cuda-8.0/include/driver_types.h:996
      desc : aliased cudaChannelFormatDesc;  -- /usr/local/cuda-8.0/include/driver_types.h:997
      sizeInBytes : aliased stddef_h.size_t;  -- /usr/local/cuda-8.0/include/driver_types.h:998
   end record;
   pragma Convention (C_Pass_By_Copy, anon_3);
   type anon_4 is record
      devPtr : System.Address;  -- /usr/local/cuda-8.0/include/driver_types.h:1001
      desc : aliased cudaChannelFormatDesc;  -- /usr/local/cuda-8.0/include/driver_types.h:1002
      width : aliased stddef_h.size_t;  -- /usr/local/cuda-8.0/include/driver_types.h:1003
      height : aliased stddef_h.size_t;  -- /usr/local/cuda-8.0/include/driver_types.h:1004
      pitchInBytes : aliased stddef_h.size_t;  -- /usr/local/cuda-8.0/include/driver_types.h:1005
   end record;
   pragma Convention (C_Pass_By_Copy, anon_4);
   type anon_0 (discr : unsigned := 0) is record
      case discr is
         when 0 =>
            c_array : aliased anon_1;  -- /usr/local/cuda-8.0/include/driver_types.h:991
         when 1 =>
            mipmap : aliased anon_2;  -- /usr/local/cuda-8.0/include/driver_types.h:994
         when 2 =>
            linear : aliased anon_3;  -- /usr/local/cuda-8.0/include/driver_types.h:999
         when others =>
            pitch2D : aliased anon_4;  -- /usr/local/cuda-8.0/include/driver_types.h:1006
      end case;
   end record;
   pragma Convention (C_Pass_By_Copy, anon_0);
   pragma Unchecked_Union (anon_0);type cudaResourceDesc is record
      resType : aliased cudaResourceType;  -- /usr/local/cuda-8.0/include/driver_types.h:986
      res : aliased anon_0;  -- /usr/local/cuda-8.0/include/driver_types.h:1007
   end record;
   pragma Convention (C_Pass_By_Copy, cudaResourceDesc);  -- /usr/local/cuda-8.0/include/driver_types.h:985

  --*< CUDA array  
  --*< CUDA mipmapped array  
  --*< Device pointer  
  --*< Channel descriptor  
  --*< Size in bytes  
  --*< Device pointer  
  --*< Channel descriptor  
  --*< Width of the array in elements  
  --*< Height of the array in elements  
  --*< Pitch between two rows in bytes  
  --*
  -- * CUDA resource view descriptor
  --  

  --*< Resource view format  
   type cudaResourceViewDesc is record
      format : aliased cudaResourceViewFormat;  -- /usr/local/cuda-8.0/include/driver_types.h:1015
      width : aliased stddef_h.size_t;  -- /usr/local/cuda-8.0/include/driver_types.h:1016
      height : aliased stddef_h.size_t;  -- /usr/local/cuda-8.0/include/driver_types.h:1017
      depth : aliased stddef_h.size_t;  -- /usr/local/cuda-8.0/include/driver_types.h:1018
      firstMipmapLevel : aliased unsigned;  -- /usr/local/cuda-8.0/include/driver_types.h:1019
      lastMipmapLevel : aliased unsigned;  -- /usr/local/cuda-8.0/include/driver_types.h:1020
      firstLayer : aliased unsigned;  -- /usr/local/cuda-8.0/include/driver_types.h:1021
      lastLayer : aliased unsigned;  -- /usr/local/cuda-8.0/include/driver_types.h:1022
   end record;
   pragma Convention (C_Pass_By_Copy, cudaResourceViewDesc);  -- /usr/local/cuda-8.0/include/driver_types.h:1013

  --*< Width of the resource view  
  --*< Height of the resource view  
  --*< Depth of the resource view  
  --*< First defined mipmap level  
  --*< Last defined mipmap level  
  --*< First layer index  
  --*< Last layer index  
  --*
  -- * CUDA pointer attributes
  --  

  --*
  --     * The physical location of the memory, ::cudaMemoryTypeHost or 
  --     * ::cudaMemoryTypeDevice.
  --      

   type cudaPointerAttributes is record
      memoryType : aliased cudaMemoryType;  -- /usr/local/cuda-8.0/include/driver_types.h:1034
      device : aliased int;  -- /usr/local/cuda-8.0/include/driver_types.h:1045
      devicePointer : System.Address;  -- /usr/local/cuda-8.0/include/driver_types.h:1051
      hostPointer : System.Address;  -- /usr/local/cuda-8.0/include/driver_types.h:1057
      isManaged : aliased int;  -- /usr/local/cuda-8.0/include/driver_types.h:1062
   end record;
   pragma Convention (C_Pass_By_Copy, cudaPointerAttributes);  -- /usr/local/cuda-8.0/include/driver_types.h:1028

  --* 
  --     * The device against which the memory was allocated or registered.
  --     * If the memory type is ::cudaMemoryTypeDevice then this identifies 
  --     * the device on which the memory referred physically resides.  If
  --     * the memory type is ::cudaMemoryTypeHost then this identifies the 
  --     * device which was current when the memory was allocated or registered
  --     * (and if that device is deinitialized then this allocation will vanish
  --     * with that device's state).
  --      

  --*
  --     * The address which may be dereferenced on the current device to access 
  --     * the memory or NULL if no such address exists.
  --      

  --*
  --     * The address which may be dereferenced on the host to access the
  --     * memory or NULL if no such address exists.
  --      

  --*
  --     * Indicates if this pointer points to managed memory
  --      

  --*
  -- * CUDA function attributes
  --  

  --*
  --    * The size in bytes of statically-allocated shared memory per block
  --    * required by this function. This does not include dynamically-allocated
  --    * shared memory requested by the user at runtime.
  --     

   type cudaFuncAttributes is record
      sharedSizeBytes : aliased stddef_h.size_t;  -- /usr/local/cuda-8.0/include/driver_types.h:1075
      constSizeBytes : aliased stddef_h.size_t;  -- /usr/local/cuda-8.0/include/driver_types.h:1081
      localSizeBytes : aliased stddef_h.size_t;  -- /usr/local/cuda-8.0/include/driver_types.h:1086
      maxThreadsPerBlock : aliased int;  -- /usr/local/cuda-8.0/include/driver_types.h:1093
      numRegs : aliased int;  -- /usr/local/cuda-8.0/include/driver_types.h:1098
      ptxVersion : aliased int;  -- /usr/local/cuda-8.0/include/driver_types.h:1105
      binaryVersion : aliased int;  -- /usr/local/cuda-8.0/include/driver_types.h:1112
      cacheModeCA : aliased int;  -- /usr/local/cuda-8.0/include/driver_types.h:1118
   end record;
   pragma Convention (C_Pass_By_Copy, cudaFuncAttributes);  -- /usr/local/cuda-8.0/include/driver_types.h:1068

  --*
  --    * The size in bytes of user-allocated constant memory required by this
  --    * function.
  --     

  --*
  --    * The size in bytes of local memory used by each thread of this function.
  --     

  --*
  --    * The maximum number of threads per block, beyond which a launch of the
  --    * function would fail. This number depends on both the function and the
  --    * device on which the function is currently loaded.
  --     

  --*
  --    * The number of registers used by each thread of this function.
  --     

  --*
  --    * The PTX virtual architecture version for which the function was
  --    * compiled. This value is the major PTX version * 10 + the minor PTX
  --    * version, so a PTX version 1.3 function would return the value 13.
  --     

  --*
  --    * The binary architecture version for which the function was compiled.
  --    * This value is the major binary version * 10 + the minor binary version,
  --    * so a binary version 1.3 function would return the value 13.
  --     

  --*
  --    * The attribute to indicate whether the function has been compiled with 
  --    * user specified option "-Xptxas --dlcm=ca" set.
  --     

  --*
  -- * CUDA function cache configurations
  --  

   type cudaFuncCache is 
     (cudaFuncCachePreferNone,
      cudaFuncCachePreferShared,
      cudaFuncCachePreferL1,
      cudaFuncCachePreferEqual);
   pragma Convention (C, cudaFuncCache);  -- /usr/local/cuda-8.0/include/driver_types.h:1124

  --*< Default function cache configuration, no preference  
  --*< Prefer larger shared memory and smaller L1 cache   
  --*< Prefer larger L1 cache and smaller shared memory  
  --*< Prefer equal size L1 cache and shared memory  
  --*
  -- * CUDA shared memory configuration
  --  

   type cudaSharedMemConfig is 
     (cudaSharedMemBankSizeDefault,
      cudaSharedMemBankSizeFourByte,
      cudaSharedMemBankSizeEightByte);
   pragma Convention (C, cudaSharedMemConfig);  -- /usr/local/cuda-8.0/include/driver_types.h:1136

  --*
  -- * CUDA device compute modes
  --  

   type cudaComputeMode is 
     (cudaComputeModeDefault,
      cudaComputeModeExclusive,
      cudaComputeModeProhibited,
      cudaComputeModeExclusiveProcess);
   pragma Convention (C, cudaComputeMode);  -- /usr/local/cuda-8.0/include/driver_types.h:1146

  --*< Default compute mode (Multiple threads can use ::cudaSetDevice() with this device)  
  --*< Compute-exclusive-thread mode (Only one thread in one process will be able to use ::cudaSetDevice() with this device)  
  --*< Compute-prohibited mode (No threads can use ::cudaSetDevice() with this device)  
  --*< Compute-exclusive-process mode (Many threads in one process will be able to use ::cudaSetDevice() with this device)  
  --*
  -- * CUDA Limits
  --  

   type cudaLimit is 
     (cudaLimitStackSize,
      cudaLimitPrintfFifoSize,
      cudaLimitMallocHeapSize,
      cudaLimitDevRuntimeSyncDepth,
      cudaLimitDevRuntimePendingLaunchCount);
   pragma Convention (C, cudaLimit);  -- /usr/local/cuda-8.0/include/driver_types.h:1157

  --*< GPU thread stack size  
  --*< GPU printf/fprintf FIFO size  
  --*< GPU malloc heap size  
  --*< GPU device runtime synchronize depth  
  --*< GPU device runtime pending launch count  
  --*
  -- * CUDA Memory Advise values
  --  

   subtype cudaMemoryAdvise is unsigned;
   cudaMemAdviseSetReadMostly : constant cudaMemoryAdvise := 1;
   cudaMemAdviseUnsetReadMostly : constant cudaMemoryAdvise := 2;
   cudaMemAdviseSetPreferredLocation : constant cudaMemoryAdvise := 3;
   cudaMemAdviseUnsetPreferredLocation : constant cudaMemoryAdvise := 4;
   cudaMemAdviseSetAccessedBy : constant cudaMemoryAdvise := 5;
   cudaMemAdviseUnsetAccessedBy : constant cudaMemoryAdvise := 6;  -- /usr/local/cuda-8.0/include/driver_types.h:1169

  --*< Data will mostly be read and only occassionally be written to  
  --*< Undo the effect of ::cudaMemAdviseSetReadMostly  
  --*< Set the preferred location for the data as the specified device  
  --*< Clear the preferred location for the data  
  --*< Data will be accessed by the specified device, so prevent page faults as much as possible  
  --*< Let the Unified Memory subsystem decide on the page faulting policy for the specified device  
  --*
  -- * CUDA range attributes
  --  

   subtype cudaMemRangeAttribute is unsigned;
   cudaMemRangeAttributeReadMostly : constant cudaMemRangeAttribute := 1;
   cudaMemRangeAttributePreferredLocation : constant cudaMemRangeAttribute := 2;
   cudaMemRangeAttributeAccessedBy : constant cudaMemRangeAttribute := 3;
   cudaMemRangeAttributeLastPrefetchLocation : constant cudaMemRangeAttribute := 4;  -- /usr/local/cuda-8.0/include/driver_types.h:1182

  --*< Whether the range will mostly be read and only occassionally be written to  
  --*< The preferred location of the range  
  --*< Memory range has ::cudaMemAdviseSetAccessedBy set for specified device  
  --*< The last location to which the range was prefetched  
  --*
  -- * CUDA Profiler Output modes
  --  

   type cudaOutputMode is 
     (cudaKeyValuePair,
      cudaCSV);
   pragma Convention (C, cudaOutputMode);  -- /usr/local/cuda-8.0/include/driver_types.h:1193

  --*< Output mode Key-Value pair format.  
  --*< Output mode Comma separated values format.  
  --*
  -- * CUDA device attributes
  --  

   subtype cudaDeviceAttr is unsigned;
   cudaDevAttrMaxThreadsPerBlock : constant cudaDeviceAttr := 1;
   cudaDevAttrMaxBlockDimX : constant cudaDeviceAttr := 2;
   cudaDevAttrMaxBlockDimY : constant cudaDeviceAttr := 3;
   cudaDevAttrMaxBlockDimZ : constant cudaDeviceAttr := 4;
   cudaDevAttrMaxGridDimX : constant cudaDeviceAttr := 5;
   cudaDevAttrMaxGridDimY : constant cudaDeviceAttr := 6;
   cudaDevAttrMaxGridDimZ : constant cudaDeviceAttr := 7;
   cudaDevAttrMaxSharedMemoryPerBlock : constant cudaDeviceAttr := 8;
   cudaDevAttrTotalConstantMemory : constant cudaDeviceAttr := 9;
   cudaDevAttrWarpSize : constant cudaDeviceAttr := 10;
   cudaDevAttrMaxPitch : constant cudaDeviceAttr := 11;
   cudaDevAttrMaxRegistersPerBlock : constant cudaDeviceAttr := 12;
   cudaDevAttrClockRate : constant cudaDeviceAttr := 13;
   cudaDevAttrTextureAlignment : constant cudaDeviceAttr := 14;
   cudaDevAttrGpuOverlap : constant cudaDeviceAttr := 15;
   cudaDevAttrMultiProcessorCount : constant cudaDeviceAttr := 16;
   cudaDevAttrKernelExecTimeout : constant cudaDeviceAttr := 17;
   cudaDevAttrIntegrated : constant cudaDeviceAttr := 18;
   cudaDevAttrCanMapHostMemory : constant cudaDeviceAttr := 19;
   cudaDevAttrComputeMode : constant cudaDeviceAttr := 20;
   cudaDevAttrMaxTexture1DWidth : constant cudaDeviceAttr := 21;
   cudaDevAttrMaxTexture2DWidth : constant cudaDeviceAttr := 22;
   cudaDevAttrMaxTexture2DHeight : constant cudaDeviceAttr := 23;
   cudaDevAttrMaxTexture3DWidth : constant cudaDeviceAttr := 24;
   cudaDevAttrMaxTexture3DHeight : constant cudaDeviceAttr := 25;
   cudaDevAttrMaxTexture3DDepth : constant cudaDeviceAttr := 26;
   cudaDevAttrMaxTexture2DLayeredWidth : constant cudaDeviceAttr := 27;
   cudaDevAttrMaxTexture2DLayeredHeight : constant cudaDeviceAttr := 28;
   cudaDevAttrMaxTexture2DLayeredLayers : constant cudaDeviceAttr := 29;
   cudaDevAttrSurfaceAlignment : constant cudaDeviceAttr := 30;
   cudaDevAttrConcurrentKernels : constant cudaDeviceAttr := 31;
   cudaDevAttrEccEnabled : constant cudaDeviceAttr := 32;
   cudaDevAttrPciBusId : constant cudaDeviceAttr := 33;
   cudaDevAttrPciDeviceId : constant cudaDeviceAttr := 34;
   cudaDevAttrTccDriver : constant cudaDeviceAttr := 35;
   cudaDevAttrMemoryClockRate : constant cudaDeviceAttr := 36;
   cudaDevAttrGlobalMemoryBusWidth : constant cudaDeviceAttr := 37;
   cudaDevAttrL2CacheSize : constant cudaDeviceAttr := 38;
   cudaDevAttrMaxThreadsPerMultiProcessor : constant cudaDeviceAttr := 39;
   cudaDevAttrAsyncEngineCount : constant cudaDeviceAttr := 40;
   cudaDevAttrUnifiedAddressing : constant cudaDeviceAttr := 41;
   cudaDevAttrMaxTexture1DLayeredWidth : constant cudaDeviceAttr := 42;
   cudaDevAttrMaxTexture1DLayeredLayers : constant cudaDeviceAttr := 43;
   cudaDevAttrMaxTexture2DGatherWidth : constant cudaDeviceAttr := 45;
   cudaDevAttrMaxTexture2DGatherHeight : constant cudaDeviceAttr := 46;
   cudaDevAttrMaxTexture3DWidthAlt : constant cudaDeviceAttr := 47;
   cudaDevAttrMaxTexture3DHeightAlt : constant cudaDeviceAttr := 48;
   cudaDevAttrMaxTexture3DDepthAlt : constant cudaDeviceAttr := 49;
   cudaDevAttrPciDomainId : constant cudaDeviceAttr := 50;
   cudaDevAttrTexturePitchAlignment : constant cudaDeviceAttr := 51;
   cudaDevAttrMaxTextureCubemapWidth : constant cudaDeviceAttr := 52;
   cudaDevAttrMaxTextureCubemapLayeredWidth : constant cudaDeviceAttr := 53;
   cudaDevAttrMaxTextureCubemapLayeredLayers : constant cudaDeviceAttr := 54;
   cudaDevAttrMaxSurface1DWidth : constant cudaDeviceAttr := 55;
   cudaDevAttrMaxSurface2DWidth : constant cudaDeviceAttr := 56;
   cudaDevAttrMaxSurface2DHeight : constant cudaDeviceAttr := 57;
   cudaDevAttrMaxSurface3DWidth : constant cudaDeviceAttr := 58;
   cudaDevAttrMaxSurface3DHeight : constant cudaDeviceAttr := 59;
   cudaDevAttrMaxSurface3DDepth : constant cudaDeviceAttr := 60;
   cudaDevAttrMaxSurface1DLayeredWidth : constant cudaDeviceAttr := 61;
   cudaDevAttrMaxSurface1DLayeredLayers : constant cudaDeviceAttr := 62;
   cudaDevAttrMaxSurface2DLayeredWidth : constant cudaDeviceAttr := 63;
   cudaDevAttrMaxSurface2DLayeredHeight : constant cudaDeviceAttr := 64;
   cudaDevAttrMaxSurface2DLayeredLayers : constant cudaDeviceAttr := 65;
   cudaDevAttrMaxSurfaceCubemapWidth : constant cudaDeviceAttr := 66;
   cudaDevAttrMaxSurfaceCubemapLayeredWidth : constant cudaDeviceAttr := 67;
   cudaDevAttrMaxSurfaceCubemapLayeredLayers : constant cudaDeviceAttr := 68;
   cudaDevAttrMaxTexture1DLinearWidth : constant cudaDeviceAttr := 69;
   cudaDevAttrMaxTexture2DLinearWidth : constant cudaDeviceAttr := 70;
   cudaDevAttrMaxTexture2DLinearHeight : constant cudaDeviceAttr := 71;
   cudaDevAttrMaxTexture2DLinearPitch : constant cudaDeviceAttr := 72;
   cudaDevAttrMaxTexture2DMipmappedWidth : constant cudaDeviceAttr := 73;
   cudaDevAttrMaxTexture2DMipmappedHeight : constant cudaDeviceAttr := 74;
   cudaDevAttrComputeCapabilityMajor : constant cudaDeviceAttr := 75;
   cudaDevAttrComputeCapabilityMinor : constant cudaDeviceAttr := 76;
   cudaDevAttrMaxTexture1DMipmappedWidth : constant cudaDeviceAttr := 77;
   cudaDevAttrStreamPrioritiesSupported : constant cudaDeviceAttr := 78;
   cudaDevAttrGlobalL1CacheSupported : constant cudaDeviceAttr := 79;
   cudaDevAttrLocalL1CacheSupported : constant cudaDeviceAttr := 80;
   cudaDevAttrMaxSharedMemoryPerMultiprocessor : constant cudaDeviceAttr := 81;
   cudaDevAttrMaxRegistersPerMultiprocessor : constant cudaDeviceAttr := 82;
   cudaDevAttrManagedMemory : constant cudaDeviceAttr := 83;
   cudaDevAttrIsMultiGpuBoard : constant cudaDeviceAttr := 84;
   cudaDevAttrMultiGpuBoardGroupID : constant cudaDeviceAttr := 85;
   cudaDevAttrHostNativeAtomicSupported : constant cudaDeviceAttr := 86;
   cudaDevAttrSingleToDoublePrecisionPerfRatio : constant cudaDeviceAttr := 87;
   cudaDevAttrPageableMemoryAccess : constant cudaDeviceAttr := 88;
   cudaDevAttrConcurrentManagedAccess : constant cudaDeviceAttr := 89;
   cudaDevAttrComputePreemptionSupported : constant cudaDeviceAttr := 90;
   cudaDevAttrCanUseHostPointerForRegisteredMem : constant cudaDeviceAttr := 91;  -- /usr/local/cuda-8.0/include/driver_types.h:1202

  --*< Maximum number of threads per block  
  --*< Maximum block dimension X  
  --*< Maximum block dimension Y  
  --*< Maximum block dimension Z  
  --*< Maximum grid dimension X  
  --*< Maximum grid dimension Y  
  --*< Maximum grid dimension Z  
  --*< Maximum shared memory available per block in bytes  
  --*< Memory available on device for __constant__ variables in a CUDA C kernel in bytes  
  --*< Warp size in threads  
  --*< Maximum pitch in bytes allowed by memory copies  
  --*< Maximum number of 32-bit registers available per block  
  --*< Peak clock frequency in kilohertz  
  --*< Alignment requirement for textures  
  --*< Device can possibly copy memory and execute a kernel concurrently  
  --*< Number of multiprocessors on device  
  --*< Specifies whether there is a run time limit on kernels  
  --*< Device is integrated with host memory  
  --*< Device can map host memory into CUDA address space  
  --*< Compute mode (See ::cudaComputeMode for details)  
  --*< Maximum 1D texture width  
  --*< Maximum 2D texture width  
  --*< Maximum 2D texture height  
  --*< Maximum 3D texture width  
  --*< Maximum 3D texture height  
  --*< Maximum 3D texture depth  
  --*< Maximum 2D layered texture width  
  --*< Maximum 2D layered texture height  
  --*< Maximum layers in a 2D layered texture  
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
  --*< Maximum 2D texture width if cudaArrayTextureGather is set  
  --*< Maximum 2D texture height if cudaArrayTextureGather is set  
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
  --*< Unique identifier for a group of devices on the same multi-GPU board  
  --*< Link between the device and the host supports native atomic operations  
  --*< Ratio of single precision performance (in floating-point operations per second) to double precision performance  
  --*< Device supports coherently accessing pageable memory without calling cudaHostRegister on it  
  --*< Device can coherently access managed memory concurrently with the CPU  
  --*< Device supports Compute Preemption  
  --*< Device can access host registered memory at the same virtual address as the CPU  
  --*
  -- * CUDA device P2P attributes
  --  

   subtype cudaDeviceP2PAttr is unsigned;
   cudaDevP2PAttrPerformanceRank : constant cudaDeviceP2PAttr := 1;
   cudaDevP2PAttrAccessSupported : constant cudaDeviceP2PAttr := 2;
   cudaDevP2PAttrNativeAtomicSupported : constant cudaDeviceP2PAttr := 3;  -- /usr/local/cuda-8.0/include/driver_types.h:1300

  --*< A relative value indicating the performance of the link between two devices  
  --*< Peer access is enabled  
  --*< Native atomic operation over the link supported  
  --*
  -- * CUDA device properties
  --  

  --*< ASCII string identifying device  
   subtype cudaDeviceProp_name_array is Interfaces.C.char_array (0 .. 255);
   type cudaDeviceProp_maxThreadsDim_array is array (0 .. 2) of aliased int;
   type cudaDeviceProp_maxGridSize_array is array (0 .. 2) of aliased int;
   type cudaDeviceProp_maxTexture2D_array is array (0 .. 1) of aliased int;
   type cudaDeviceProp_maxTexture2DMipmap_array is array (0 .. 1) of aliased int;
   type cudaDeviceProp_maxTexture2DLinear_array is array (0 .. 2) of aliased int;
   type cudaDeviceProp_maxTexture2DGather_array is array (0 .. 1) of aliased int;
   type cudaDeviceProp_maxTexture3D_array is array (0 .. 2) of aliased int;
   type cudaDeviceProp_maxTexture3DAlt_array is array (0 .. 2) of aliased int;
   type cudaDeviceProp_maxTexture1DLayered_array is array (0 .. 1) of aliased int;
   type cudaDeviceProp_maxTexture2DLayered_array is array (0 .. 2) of aliased int;
   type cudaDeviceProp_maxTextureCubemapLayered_array is array (0 .. 1) of aliased int;
   type cudaDeviceProp_maxSurface2D_array is array (0 .. 1) of aliased int;
   type cudaDeviceProp_maxSurface3D_array is array (0 .. 2) of aliased int;
   type cudaDeviceProp_maxSurface1DLayered_array is array (0 .. 1) of aliased int;
   type cudaDeviceProp_maxSurface2DLayered_array is array (0 .. 2) of aliased int;
   type cudaDeviceProp_maxSurfaceCubemapLayered_array is array (0 .. 1) of aliased int;
   type cudaDeviceProp is record
      name : aliased cudaDeviceProp_name_array;  -- /usr/local/cuda-8.0/include/driver_types.h:1310
      totalGlobalMem : aliased stddef_h.size_t;  -- /usr/local/cuda-8.0/include/driver_types.h:1311
      sharedMemPerBlock : aliased stddef_h.size_t;  -- /usr/local/cuda-8.0/include/driver_types.h:1312
      regsPerBlock : aliased int;  -- /usr/local/cuda-8.0/include/driver_types.h:1313
      warpSize : aliased int;  -- /usr/local/cuda-8.0/include/driver_types.h:1314
      memPitch : aliased stddef_h.size_t;  -- /usr/local/cuda-8.0/include/driver_types.h:1315
      maxThreadsPerBlock : aliased int;  -- /usr/local/cuda-8.0/include/driver_types.h:1316
      maxThreadsDim : aliased cudaDeviceProp_maxThreadsDim_array;  -- /usr/local/cuda-8.0/include/driver_types.h:1317
      maxGridSize : aliased cudaDeviceProp_maxGridSize_array;  -- /usr/local/cuda-8.0/include/driver_types.h:1318
      clockRate : aliased int;  -- /usr/local/cuda-8.0/include/driver_types.h:1319
      totalConstMem : aliased stddef_h.size_t;  -- /usr/local/cuda-8.0/include/driver_types.h:1320
      major : aliased int;  -- /usr/local/cuda-8.0/include/driver_types.h:1321
      minor : aliased int;  -- /usr/local/cuda-8.0/include/driver_types.h:1322
      textureAlignment : aliased stddef_h.size_t;  -- /usr/local/cuda-8.0/include/driver_types.h:1323
      texturePitchAlignment : aliased stddef_h.size_t;  -- /usr/local/cuda-8.0/include/driver_types.h:1324
      deviceOverlap : aliased int;  -- /usr/local/cuda-8.0/include/driver_types.h:1325
      multiProcessorCount : aliased int;  -- /usr/local/cuda-8.0/include/driver_types.h:1326
      kernelExecTimeoutEnabled : aliased int;  -- /usr/local/cuda-8.0/include/driver_types.h:1327
      integrated : aliased int;  -- /usr/local/cuda-8.0/include/driver_types.h:1328
      canMapHostMemory : aliased int;  -- /usr/local/cuda-8.0/include/driver_types.h:1329
      computeMode : aliased int;  -- /usr/local/cuda-8.0/include/driver_types.h:1330
      maxTexture1D : aliased int;  -- /usr/local/cuda-8.0/include/driver_types.h:1331
      maxTexture1DMipmap : aliased int;  -- /usr/local/cuda-8.0/include/driver_types.h:1332
      maxTexture1DLinear : aliased int;  -- /usr/local/cuda-8.0/include/driver_types.h:1333
      maxTexture2D : aliased cudaDeviceProp_maxTexture2D_array;  -- /usr/local/cuda-8.0/include/driver_types.h:1334
      maxTexture2DMipmap : aliased cudaDeviceProp_maxTexture2DMipmap_array;  -- /usr/local/cuda-8.0/include/driver_types.h:1335
      maxTexture2DLinear : aliased cudaDeviceProp_maxTexture2DLinear_array;  -- /usr/local/cuda-8.0/include/driver_types.h:1336
      maxTexture2DGather : aliased cudaDeviceProp_maxTexture2DGather_array;  -- /usr/local/cuda-8.0/include/driver_types.h:1337
      maxTexture3D : aliased cudaDeviceProp_maxTexture3D_array;  -- /usr/local/cuda-8.0/include/driver_types.h:1338
      maxTexture3DAlt : aliased cudaDeviceProp_maxTexture3DAlt_array;  -- /usr/local/cuda-8.0/include/driver_types.h:1339
      maxTextureCubemap : aliased int;  -- /usr/local/cuda-8.0/include/driver_types.h:1340
      maxTexture1DLayered : aliased cudaDeviceProp_maxTexture1DLayered_array;  -- /usr/local/cuda-8.0/include/driver_types.h:1341
      maxTexture2DLayered : aliased cudaDeviceProp_maxTexture2DLayered_array;  -- /usr/local/cuda-8.0/include/driver_types.h:1342
      maxTextureCubemapLayered : aliased cudaDeviceProp_maxTextureCubemapLayered_array;  -- /usr/local/cuda-8.0/include/driver_types.h:1343
      maxSurface1D : aliased int;  -- /usr/local/cuda-8.0/include/driver_types.h:1344
      maxSurface2D : aliased cudaDeviceProp_maxSurface2D_array;  -- /usr/local/cuda-8.0/include/driver_types.h:1345
      maxSurface3D : aliased cudaDeviceProp_maxSurface3D_array;  -- /usr/local/cuda-8.0/include/driver_types.h:1346
      maxSurface1DLayered : aliased cudaDeviceProp_maxSurface1DLayered_array;  -- /usr/local/cuda-8.0/include/driver_types.h:1347
      maxSurface2DLayered : aliased cudaDeviceProp_maxSurface2DLayered_array;  -- /usr/local/cuda-8.0/include/driver_types.h:1348
      maxSurfaceCubemap : aliased int;  -- /usr/local/cuda-8.0/include/driver_types.h:1349
      maxSurfaceCubemapLayered : aliased cudaDeviceProp_maxSurfaceCubemapLayered_array;  -- /usr/local/cuda-8.0/include/driver_types.h:1350
      surfaceAlignment : aliased stddef_h.size_t;  -- /usr/local/cuda-8.0/include/driver_types.h:1351
      concurrentKernels : aliased int;  -- /usr/local/cuda-8.0/include/driver_types.h:1352
      ECCEnabled : aliased int;  -- /usr/local/cuda-8.0/include/driver_types.h:1353
      pciBusID : aliased int;  -- /usr/local/cuda-8.0/include/driver_types.h:1354
      pciDeviceID : aliased int;  -- /usr/local/cuda-8.0/include/driver_types.h:1355
      pciDomainID : aliased int;  -- /usr/local/cuda-8.0/include/driver_types.h:1356
      tccDriver : aliased int;  -- /usr/local/cuda-8.0/include/driver_types.h:1357
      asyncEngineCount : aliased int;  -- /usr/local/cuda-8.0/include/driver_types.h:1358
      unifiedAddressing : aliased int;  -- /usr/local/cuda-8.0/include/driver_types.h:1359
      memoryClockRate : aliased int;  -- /usr/local/cuda-8.0/include/driver_types.h:1360
      memoryBusWidth : aliased int;  -- /usr/local/cuda-8.0/include/driver_types.h:1361
      l2CacheSize : aliased int;  -- /usr/local/cuda-8.0/include/driver_types.h:1362
      maxThreadsPerMultiProcessor : aliased int;  -- /usr/local/cuda-8.0/include/driver_types.h:1363
      streamPrioritiesSupported : aliased int;  -- /usr/local/cuda-8.0/include/driver_types.h:1364
      globalL1CacheSupported : aliased int;  -- /usr/local/cuda-8.0/include/driver_types.h:1365
      localL1CacheSupported : aliased int;  -- /usr/local/cuda-8.0/include/driver_types.h:1366
      sharedMemPerMultiprocessor : aliased stddef_h.size_t;  -- /usr/local/cuda-8.0/include/driver_types.h:1367
      regsPerMultiprocessor : aliased int;  -- /usr/local/cuda-8.0/include/driver_types.h:1368
      managedMemory : aliased int;  -- /usr/local/cuda-8.0/include/driver_types.h:1369
      isMultiGpuBoard : aliased int;  -- /usr/local/cuda-8.0/include/driver_types.h:1370
      multiGpuBoardGroupID : aliased int;  -- /usr/local/cuda-8.0/include/driver_types.h:1371
      hostNativeAtomicSupported : aliased int;  -- /usr/local/cuda-8.0/include/driver_types.h:1372
      singleToDoublePrecisionPerfRatio : aliased int;  -- /usr/local/cuda-8.0/include/driver_types.h:1373
      pageableMemoryAccess : aliased int;  -- /usr/local/cuda-8.0/include/driver_types.h:1374
      concurrentManagedAccess : aliased int;  -- /usr/local/cuda-8.0/include/driver_types.h:1375
   end record;
   pragma Convention (C_Pass_By_Copy, cudaDeviceProp);  -- /usr/local/cuda-8.0/include/driver_types.h:1308

  --*< Global memory available on device in bytes  
  --*< Shared memory available per block in bytes  
  --*< 32-bit registers available per block  
  --*< Warp size in threads  
  --*< Maximum pitch in bytes allowed by memory copies  
  --*< Maximum number of threads per block  
  --*< Maximum size of each dimension of a block  
  --*< Maximum size of each dimension of a grid  
  --*< Clock frequency in kilohertz  
  --*< Constant memory available on device in bytes  
  --*< Major compute capability  
  --*< Minor compute capability  
  --*< Alignment requirement for textures  
  --*< Pitch alignment requirement for texture references bound to pitched memory  
  --*< Device can concurrently copy memory and execute a kernel. Deprecated. Use instead asyncEngineCount.  
  --*< Number of multiprocessors on device  
  --*< Specified whether there is a run time limit on kernels  
  --*< Device is integrated as opposed to discrete  
  --*< Device can map host memory with cudaHostAlloc/cudaHostGetDevicePointer  
  --*< Compute mode (See ::cudaComputeMode)  
  --*< Maximum 1D texture size  
  --*< Maximum 1D mipmapped texture size  
  --*< Maximum size for 1D textures bound to linear memory  
  --*< Maximum 2D texture dimensions  
  --*< Maximum 2D mipmapped texture dimensions  
  --*< Maximum dimensions (width, height, pitch) for 2D textures bound to pitched memory  
  --*< Maximum 2D texture dimensions if texture gather operations have to be performed  
  --*< Maximum 3D texture dimensions  
  --*< Maximum alternate 3D texture dimensions  
  --*< Maximum Cubemap texture dimensions  
  --*< Maximum 1D layered texture dimensions  
  --*< Maximum 2D layered texture dimensions  
  --*< Maximum Cubemap layered texture dimensions  
  --*< Maximum 1D surface size  
  --*< Maximum 2D surface dimensions  
  --*< Maximum 3D surface dimensions  
  --*< Maximum 1D layered surface dimensions  
  --*< Maximum 2D layered surface dimensions  
  --*< Maximum Cubemap surface dimensions  
  --*< Maximum Cubemap layered surface dimensions  
  --*< Alignment requirements for surfaces  
  --*< Device can possibly execute multiple kernels concurrently  
  --*< Device has ECC support enabled  
  --*< PCI bus ID of the device  
  --*< PCI device ID of the device  
  --*< PCI domain ID of the device  
  --*< 1 if device is a Tesla device using TCC driver, 0 otherwise  
  --*< Number of asynchronous engines  
  --*< Device shares a unified address space with the host  
  --*< Peak memory clock frequency in kilohertz  
  --*< Global memory bus width in bits  
  --*< Size of L2 cache in bytes  
  --*< Maximum resident threads per multiprocessor  
  --*< Device supports stream priorities  
  --*< Device supports caching globals in L1  
  --*< Device supports caching locals in L1  
  --*< Shared memory available per multiprocessor in bytes  
  --*< 32-bit registers available per multiprocessor  
  --*< Device supports allocating managed memory on this system  
  --*< Device is on a multi-GPU board  
  --*< Unique identifier for a group of devices on the same multi-GPU board  
  --*< Link between the device and the host supports native atomic operations  
  --*< Ratio of single precision performance (in floating-point operations per second) to double precision performance  
  --*< Device supports coherently accessing pageable memory without calling cudaHostRegister on it  
  --*< Device can coherently access managed memory concurrently with the CPU  
  --*
  -- * CUDA IPC Handle Size
  --  

  --*
  -- * CUDA IPC event handle
  --  

   subtype cudaIpcEventHandle_st_reserved_array is Interfaces.C.char_array (0 .. 63);
   type cudaIpcEventHandle_st is record
      reserved : aliased cudaIpcEventHandle_st_reserved_array;  -- /usr/local/cuda-8.0/include/driver_types.h:1458
   end record;
   pragma Convention (C_Pass_By_Copy, cudaIpcEventHandle_st);  -- /usr/local/cuda-8.0/include/driver_types.h:1456

   subtype cudaIpcEventHandle_t is cudaIpcEventHandle_st;

  --*
  -- * CUDA IPC memory handle
  --  

   subtype cudaIpcMemHandle_st_reserved_array is Interfaces.C.char_array (0 .. 63);
   type cudaIpcMemHandle_st is record
      reserved : aliased cudaIpcMemHandle_st_reserved_array;  -- /usr/local/cuda-8.0/include/driver_types.h:1466
   end record;
   pragma Convention (C_Pass_By_Copy, cudaIpcMemHandle_st);  -- /usr/local/cuda-8.0/include/driver_types.h:1464

   subtype cudaIpcMemHandle_t is cudaIpcMemHandle_st;

  --******************************************************************************
  --*                                                                              *
  --*  SHORTHAND TYPE DEFINITION USED BY RUNTIME API                               *
  --*                                                                              *
  --****************************************************************************** 

  --*
  -- * CUDA Error types
  --  

   subtype cudaError_t is cudaError;

  --*
  -- * CUDA stream
  --  

   --  skipped empty struct CUstream_st

   type cudaStream_t is new System.Address;  -- /usr/local/cuda-8.0/include/driver_types.h:1483

  --*
  -- * CUDA event types
  --  

   --  skipped empty struct CUevent_st

   type cudaEvent_t is new System.Address;  -- /usr/local/cuda-8.0/include/driver_types.h:1488

  --*
  -- * CUDA graphics resource types
  --  

   type cudaGraphicsResource_t is new System.Address;  -- /usr/local/cuda-8.0/include/driver_types.h:1493

  --*
  -- * CUDA UUID types
  --  

   --  skipped empty struct CUuuid_st

   --  skipped empty struct cudaUUID_t

  --*
  -- * CUDA output file modes
  --  

   subtype cudaOutputMode_t is cudaOutputMode;

  --* @}  
  --* @}  
  -- END CUDART_TYPES  
end driver_types_h;

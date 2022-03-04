pragma Ada_2005;
pragma Style_Checks (Off);

with Interfaces.C; use Interfaces.C;
with stddef_h;

package cuda_occupancy_h is

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
  -- * CUDA Occupancy Calculator
  -- *
  -- * NAME
  -- *
  -- *   cudaOccMaxActiveBlocksPerMultiprocessor,
  -- *   cudaOccMaxPotentialOccupancyBlockSize,
  -- *   cudaOccMaxPotentialOccupancyBlockSizeVariableSMem
  -- *
  -- * DESCRIPTION
  -- *
  -- *   The CUDA occupancy calculator provides a standalone, programmatical
  -- *   interface to compute the occupancy of a function on a device. It can also
  -- *   provide occupancy-oriented launch configuration suggestions.
  -- *
  -- *   The function and device are defined by the user through
  -- *   cudaOccFuncAttributes, cudaOccDeviceProp, and cudaOccDeviceState
  -- *   structures. All APIs require all 3 of them.
  -- *
  -- *   See the structure definition for more details about the device / function
  -- *   descriptors.
  -- *
  -- *   See each API's prototype for API usage.
  -- *
  -- * COMPATIBILITY
  -- *
  -- *   The occupancy calculator will be updated on each major CUDA toolkit
  -- *   release. It does not provide forward compatibility, i.e. new hardwares
  -- *   released after this implementation's release will not be supported.
  -- *
  -- * NOTE
  -- *
  -- *   If there is access to CUDA runtime, and the sole intent is to calculate
  -- *   occupancy related values on one of the accessible CUDA devices, using CUDA
  -- *   runtime's occupancy calculation APIs is recommended.
  -- *
  --  

  -- __OCC_INLINE will be undefined at the end of this header
   type cudaOccError_enum is 
     (CUDA_OCC_SUCCESS,
      CUDA_OCC_ERROR_INVALID_INPUT,
      CUDA_OCC_ERROR_UNKNOWN_DEVICE);
   pragma Convention (C, cudaOccError_enum);  -- /usr/local/cuda-8.0/include/cuda_occupancy.h:104

  -- no error encountered
  -- input parameter is invalid
  -- requested device is not supported in
  -- current implementation or device is
  -- invalid
   subtype cudaOccError is cudaOccError_enum;

  --*
  -- * The CUDA occupancy calculator computes the occupancy of the function
  -- * described by attributes with the given block size (blockSize), static device
  -- * properties (properties), dynamic device states (states) and per-block dynamic
  -- * shared memory allocation (dynamicSMemSize) in bytes, and output it through
  -- * result along with other useful information. The occupancy is computed in
  -- * terms of the maximum number of active blocks per multiprocessor. The user can
  -- * then convert it to other metrics, such as number of active warps.
  -- *
  -- * RETURN VALUE
  -- *
  -- * The occupancy and related information is returned through result.
  -- *
  -- * If result->activeBlocksPerMultiprocessor is 0, then the given parameter
  -- * combination cannot run on the device.
  -- *
  -- * ERRORS
  -- *
  -- *     CUDA_OCC_ERROR_INVALID_INPUT   input parameter is invalid.
  -- *     CUDA_OCC_ERROR_UNKNOWN_DEVICE  requested device is not supported in
  -- *     current implementation or device is invalid
  --  

  -- out
  -- in
  -- in
  -- in
  -- in
  -- in
  --*
  -- * The CUDA launch configurator C API suggests a grid / block size pair (in
  -- * minGridSize and blockSize) that achieves the best potential occupancy
  -- * (i.e. maximum number of active warps with the smallest number of blocks) for
  -- * the given function described by attributes, on a device described by
  -- * properties with settings in state.
  -- *
  -- * If per-block dynamic shared memory allocation is not needed, the user should
  -- * leave both blockSizeToDynamicSMemSize and dynamicSMemSize as 0.
  -- *
  -- * If per-block dynamic shared memory allocation is needed, then if the dynamic
  -- * shared memory size is constant regardless of block size, the size should be
  -- * passed through dynamicSMemSize, and blockSizeToDynamicSMemSize should be
  -- * NULL.
  -- *
  -- * Otherwise, if the per-block dynamic shared memory size varies with different
  -- * block sizes, the user needs to provide a pointer to an unary function through
  -- * blockSizeToDynamicSMemSize that computes the dynamic shared memory needed by
  -- * a block of the function for any given block size. dynamicSMemSize is
  -- * ignored. An example signature is:
  -- *
  -- *    // Take block size, returns dynamic shared memory needed
  -- *    size_t blockToSmem(int blockSize);
  -- *
  -- * RETURN VALUE
  -- *
  -- * The suggested block size and the minimum number of blocks needed to achieve
  -- * the maximum occupancy are returned through blockSize and minGridSize.
  -- *
  -- * If *blockSize is 0, then the given combination cannot run on the device.
  -- *
  -- * ERRORS
  -- *
  -- *     CUDA_OCC_ERROR_INVALID_INPUT   input parameter is invalid.
  -- *     CUDA_OCC_ERROR_UNKNOWN_DEVICE  requested device is not supported in
  -- *     current implementation or device is invalid
  -- *
  --  

  -- out
  -- out
  -- in
  -- in
  -- in
  -- in
  -- in
  --*
  -- * The CUDA launch configurator C++ API suggests a grid / block size pair (in
  -- * minGridSize and blockSize) that achieves the best potential occupancy
  -- * (i.e. the maximum number of active warps with the smallest number of blocks)
  -- * for the given function described by attributes, on a device described by
  -- * properties with settings in state.
  -- *
  -- * If per-block dynamic shared memory allocation is 0 or constant regardless of
  -- * block size, the user can use cudaOccMaxPotentialOccupancyBlockSize to
  -- * configure the launch. A constant dynamic shared memory allocation size in
  -- * bytes can be passed through dynamicSMemSize.
  -- *
  -- * Otherwise, if the per-block dynamic shared memory size varies with different
  -- * block sizes, the user needs to use
  -- * cudaOccMaxPotentialOccupancyBlockSizeVariableSmem instead, and provide a
  -- * functor / pointer to an unary function (blockSizeToDynamicSMemSize) that
  -- * computes the dynamic shared memory needed by func for any given block
  -- * size. An example signature is:
  -- *
  -- *  // Take block size, returns per-block dynamic shared memory needed
  -- *  size_t blockToSmem(int blockSize);
  -- *
  -- * RETURN VALUE
  -- *
  -- * The suggested block size and the minimum number of blocks needed to achieve
  -- * the maximum occupancy are returned through blockSize and minGridSize.
  -- *
  -- * If *blockSize is 0, then the given combination cannot run on the device.
  -- *
  -- * ERRORS
  -- *
  -- *     CUDA_OCC_ERROR_INVALID_INPUT   input parameter is invalid.
  -- *     CUDA_OCC_ERROR_UNKNOWN_DEVICE  requested device is not supported in
  -- *     current implementation or device is invalid
  -- *
  --  

  -- out
  -- out
  -- in
  -- in
  -- in
  -- in
  -- out
  -- out
  -- in
  -- in
  -- in
  -- in
  -- namespace anonymous
  --*
  -- * Data structures
  -- *
  -- * These structures are subject to change for future architecture and CUDA
  -- * releases. C users should initialize the structure as {0}.
  -- *
  --  

  --*
  -- * Device descriptor
  -- *
  -- * This structure describes a device.
  --  

  -- Compute capability major version
   package Class_cudaOccDeviceProp is
      type cudaOccDeviceProp is limited record
         computeMajor : aliased int;  -- /usr/local/cuda-8.0/include/cuda_occupancy.h:272
         computeMinor : aliased int;  -- /usr/local/cuda-8.0/include/cuda_occupancy.h:273
         maxThreadsPerBlock : aliased int;  -- /usr/local/cuda-8.0/include/cuda_occupancy.h:276
         maxThreadsPerMultiprocessor : aliased int;  -- /usr/local/cuda-8.0/include/cuda_occupancy.h:277
         regsPerBlock : aliased int;  -- /usr/local/cuda-8.0/include/cuda_occupancy.h:280
         regsPerMultiprocessor : aliased int;  -- /usr/local/cuda-8.0/include/cuda_occupancy.h:281
         warpSize : aliased int;  -- /usr/local/cuda-8.0/include/cuda_occupancy.h:282
         sharedMemPerBlock : aliased stddef_h.size_t;  -- /usr/local/cuda-8.0/include/cuda_occupancy.h:283
         sharedMemPerMultiprocessor : aliased stddef_h.size_t;  -- /usr/local/cuda-8.0/include/cuda_occupancy.h:284
         numSms : aliased int;  -- /usr/local/cuda-8.0/include/cuda_occupancy.h:285
      end record;
      pragma Import (CPP, cudaOccDeviceProp);

      function New_cudaOccDeviceProp return cudaOccDeviceProp;  -- /usr/local/cuda-8.0/include/cuda_occupancy.h:327
      pragma CPP_Constructor (New_cudaOccDeviceProp, "_ZN17cudaOccDevicePropC1Ev");
   end;
   use Class_cudaOccDeviceProp;
  -- Compute capability minor
  -- version. None supported minor version
  -- may cause error
  -- Maximum number of threads per block
  -- Maximum number of threads per SM
  -- i.e. (Max. number of warps) x (warp
  -- size)
  -- Maximum number of registers per block
  -- Maximum number of registers per SM
  -- Warp size
  -- Maximum shared memory size per block
  -- Maximum shared memory size per SM
  -- Number of SMs available
  -- This structure can be converted from a cudaDeviceProp structure for users
  -- that use this header in their CUDA applications.
  -- If the application have access to the CUDA Runtime API, the application
  -- can obtain the device properties of a CUDA device through
  -- cudaGetDeviceProperties, and initialize a cudaOccDeviceProp with the
  -- cudaDeviceProp structure.
  -- Example:
  --     {
  --         cudaDeviceProp prop;
  --         cudaGetDeviceProperties(&prop, ...);
  --         cudaOccDeviceProp occProp = prop;
  --         ...
  --         cudaOccMaxPotentialOccupancyBlockSize(..., &occProp, ...);
  --     }
  --      

  --*
  -- * Partitioned global caching option
  --  

   type cudaOccPartitionedGCConfig_enum is 
     (PARTITIONED_GC_OFF,
      PARTITIONED_GC_ON,
      PARTITIONED_GC_ON_STRICT);
   pragma Convention (C, cudaOccPartitionedGCConfig_enum);  -- /usr/local/cuda-8.0/include/cuda_occupancy.h:345

  -- Disable partitioned global caching
  -- Prefer partitioned global caching
  -- Force partitioned global caching
   subtype cudaOccPartitionedGCConfig is cudaOccPartitionedGCConfig_enum;

  --*
  -- * Function descriptor
  -- *
  -- * This structure describes a CUDA function.
  --  

  -- Maximum block size the function can work with. If
   package Class_cudaOccFuncAttributes is
      type cudaOccFuncAttributes is limited record
         maxThreadsPerBlock : aliased int;  -- /usr/local/cuda-8.0/include/cuda_occupancy.h:357
         numRegs : aliased int;  -- /usr/local/cuda-8.0/include/cuda_occupancy.h:360
         sharedSizeBytes : aliased stddef_h.size_t;  -- /usr/local/cuda-8.0/include/cuda_occupancy.h:363
         partitionedGCConfig : aliased cudaOccPartitionedGCConfig;  -- /usr/local/cuda-8.0/include/cuda_occupancy.h:365
      end record;
      pragma Import (CPP, cudaOccFuncAttributes);

      function New_cudaOccFuncAttributes return cudaOccFuncAttributes;  -- /usr/local/cuda-8.0/include/cuda_occupancy.h:420
      pragma CPP_Constructor (New_cudaOccFuncAttributes, "_ZN21cudaOccFuncAttributesC1Ev");
   end;
   use Class_cudaOccFuncAttributes;
  -- unlimited, use INT_MAX or any value greater than
  -- or equal to maxThreadsPerBlock of the device
  -- Number of registers used. When the function is
  -- launched on device, the register count may change
  -- due to internal tools requirements.
  -- Number of static shared memory used
  -- Partitioned global caching is required to enable
  -- caching on certain chips, such as sm_52
  -- devices. Partitioned global caching can be
  -- automatically disabled if the occupancy
  -- requirement of the launch cannot support caching.
  -- To override this behavior with caching on and
  -- calculate occupancy strictly according to the
  -- preference, set partitionedGCConfig to
  -- PARTITIONED_GC_ON_STRICT. This is especially
  -- useful for experimenting and finding launch
  -- configurations (MaxPotentialOccupancyBlockSize)
  -- that allow global caching to take effect.
  -- This flag only affects the occupancy calculation.
  -- This structure can be converted from a cudaFuncAttributes structure for
  -- users that use this header in their CUDA applications.
  -- If the application have access to the CUDA Runtime API, the application
  -- can obtain the function attributes of a CUDA kernel function through
  -- cudaFuncGetAttributes, and initialize a cudaOccFuncAttributes with the
  -- cudaFuncAttributes structure.
  -- Example:
  --      __global__ void foo() {...}
  --      ...
  --      {
  --          cudaFuncAttributes attr;
  --          cudaFuncGetAttributes(&attr, foo);
  --          cudaOccFuncAttributes occAttr = attr;
  --          ...
  --          cudaOccMaxPotentialOccupancyBlockSize(..., &occAttr, ...);
  --      }
  --      

   type cudaOccCacheConfig_enum is 
     (CACHE_PREFER_NONE,
      CACHE_PREFER_SHARED,
      CACHE_PREFER_L1,
      CACHE_PREFER_EQUAL);
   pragma Convention (C, cudaOccCacheConfig_enum);  -- /usr/local/cuda-8.0/include/cuda_occupancy.h:429

  -- no preference for shared memory or L1 (default)
  -- prefer larger shared memory and smaller L1 cache
  -- prefer larger L1 cache and smaller shared memory
  -- prefer equal sized L1 cache and shared memory
   subtype cudaOccCacheConfig is cudaOccCacheConfig_enum;

  --*
  -- * Device state descriptor
  -- *
  -- * This structure describes device settings that affect occupancy calculation.
  --  

  -- Cache / L1 split preference. Ignored for
   package Class_cudaOccDeviceState is
      type cudaOccDeviceState is limited record
         cacheConfig : aliased cudaOccCacheConfig;  -- /usr/local/cuda-8.0/include/cuda_occupancy.h:443
      end record;
      pragma Import (CPP, cudaOccDeviceState);

      function New_cudaOccDeviceState return cudaOccDeviceState;  -- /usr/local/cuda-8.0/include/cuda_occupancy.h:447
      pragma CPP_Constructor (New_cudaOccDeviceState, "_ZN18cudaOccDeviceStateC1Ev");
   end;
   use Class_cudaOccDeviceState;
  -- sm1x/5x (Tesla/Maxwell)
   subtype cudaOccLimitingFactor_enum is unsigned;
   OCC_LIMIT_WARPS : constant cudaOccLimitingFactor_enum := 1;
   OCC_LIMIT_REGISTERS : constant cudaOccLimitingFactor_enum := 2;
   OCC_LIMIT_SHARED_MEMORY : constant cudaOccLimitingFactor_enum := 4;
   OCC_LIMIT_BLOCKS : constant cudaOccLimitingFactor_enum := 8;  -- /usr/local/cuda-8.0/include/cuda_occupancy.h:453

  -- Occupancy limited due to:
  -- - warps available
  -- - registers available
  -- - shared memory available
  -- - blocks available
   subtype cudaOccLimitingFactor is cudaOccLimitingFactor_enum;

  --*
  -- * Occupancy output
  -- *
  -- * This structure contains occupancy calculator's output.
  --  

  -- Occupancy
   type cudaOccResult is record
      activeBlocksPerMultiprocessor : aliased int;  -- /usr/local/cuda-8.0/include/cuda_occupancy.h:467
      limitingFactors : aliased unsigned;  -- /usr/local/cuda-8.0/include/cuda_occupancy.h:468
      blockLimitRegs : aliased int;  -- /usr/local/cuda-8.0/include/cuda_occupancy.h:471
      blockLimitSharedMem : aliased int;  -- /usr/local/cuda-8.0/include/cuda_occupancy.h:474
      blockLimitWarps : aliased int;  -- /usr/local/cuda-8.0/include/cuda_occupancy.h:477
      blockLimitBlocks : aliased int;  -- /usr/local/cuda-8.0/include/cuda_occupancy.h:478
      allocatedRegistersPerBlock : aliased int;  -- /usr/local/cuda-8.0/include/cuda_occupancy.h:480
      allocatedSharedMemPerBlock : aliased stddef_h.size_t;  -- /usr/local/cuda-8.0/include/cuda_occupancy.h:482
      partitionedGCConfig : aliased cudaOccPartitionedGCConfig;  -- /usr/local/cuda-8.0/include/cuda_occupancy.h:484
   end record;
   pragma Convention (C_Pass_By_Copy, cudaOccResult);  -- /usr/local/cuda-8.0/include/cuda_occupancy.h:466

  -- Factors that limited occupancy. A bit
  -- field that counts the limiting
  -- factors, see cudaOccLimitingFactor
  -- Occupancy due to register
  -- usage, INT_MAX if the kernel does not
  -- use any register.
  -- Occupancy due to shared memory
  -- usage, INT_MAX if the kernel does not
  -- use shared memory.
  -- Occupancy due to block size limit
  -- Occupancy due to maximum number of blocks
  -- managable per SM
  -- Actual number of registers allocated per
  -- block
  -- Actual size of shared memory allocated
  -- per block
  -- Report if partitioned global caching
  -- is actually enabled.
  --*
  -- * Partitioned global caching support
  -- *
  -- * See cudaOccPartitionedGlobalCachingModeSupport
  --  

   type cudaOccPartitionedGCSupport_enum is 
     (PARTITIONED_GC_NOT_SUPPORTED,
      PARTITIONED_GC_SUPPORTED,
      PARTITIONED_GC_ALWAYS_ON);
   pragma Convention (C, cudaOccPartitionedGCSupport_enum);  -- /usr/local/cuda-8.0/include/cuda_occupancy.h:494

  -- Partitioned global caching is not supported
  -- Partitioned global caching is supported
  -- This is only needed for Pascal. This, and
  -- all references / explanations for this,
  -- should be removed from the header before
  -- exporting to toolkit.
   subtype cudaOccPartitionedGCSupport is cudaOccPartitionedGCSupport_enum;

  --*
  -- * Implementation
  --  

  --*
  -- * Max compute capability supported
  --  

  --////////////////////////////////////////
  --    Mathematical Helper Functions     //
  --////////////////////////////////////////
   --  skipped func __occMin

   --  skipped func __occDivideRoundUp

   --  skipped func __occRoundUp

  --////////////////////////////////////////
  --      Architectural Properties        //
  --////////////////////////////////////////
  --*
  -- * Granularity of shared memory allocation
  --  

   function cudaOccSMemAllocationGranularity (limit : access int; properties : access constant cudaOccDeviceProp) return cudaOccError;  -- /usr/local/cuda-8.0/include/cuda_occupancy.h:539
   pragma Import (CPP, cudaOccSMemAllocationGranularity, "_ZL32cudaOccSMemAllocationGranularityPiPK17cudaOccDeviceProp");

  --*
  -- * Granularity of register allocation
  --  

   function cudaOccRegAllocationGranularity
     (limit : access int;
      properties : access constant cudaOccDeviceProp;
      regsPerThread : int) return cudaOccError;  -- /usr/local/cuda-8.0/include/cuda_occupancy.h:564
   pragma Import (CPP, cudaOccRegAllocationGranularity, "_ZL31cudaOccRegAllocationGranularityPiPK17cudaOccDevicePropi");

  -- Fermi+ allocates registers to warps
  --*
  -- * Number of sub-partitions
  --  

   function cudaOccSubPartitionsPerMultiprocessor (limit : access int; properties : access constant cudaOccDeviceProp) return cudaOccError;  -- /usr/local/cuda-8.0/include/cuda_occupancy.h:605
   pragma Import (CPP, cudaOccSubPartitionsPerMultiprocessor, "_ZL37cudaOccSubPartitionsPerMultiprocessorPiPK17cudaOccDeviceProp");

  --*
  -- * Maximum number of blocks that can run simultaneously on a multiprocessor
  --  

   function cudaOccMaxBlocksPerMultiprocessor (limit : access int; properties : access constant cudaOccDeviceProp) return cudaOccError;  -- /usr/local/cuda-8.0/include/cuda_occupancy.h:632
   pragma Import (CPP, cudaOccMaxBlocksPerMultiprocessor, "_ZL33cudaOccMaxBlocksPerMultiprocessorPiPK17cudaOccDeviceProp");

  --*
  -- * Shared memory based on config requested by User
  --  

   function cudaOccSMemPerMultiprocessor
     (limit : access stddef_h.size_t;
      properties : access constant cudaOccDeviceProp;
      cacheConfig : cudaOccCacheConfig) return cudaOccError;  -- /usr/local/cuda-8.0/include/cuda_occupancy.h:661
   pragma Import (CPP, cudaOccSMemPerMultiprocessor, "_ZL28cudaOccSMemPerMultiprocessorPmPK17cudaOccDeviceProp23cudaOccCacheConfig_enum");

  -- Fermi and Kepler has shared L1 cache / shared memory, and support cache
  -- configuration to trade one for the other. These values are needed to
  -- calculate the correct shared memory size for user requested cache
  -- configuration.
  -- Fermi supports 48KB / 16KB or 16KB / 48KB partitions for shared /
  -- L1.
  -- Kepler supports 16KB, 32KB, or 48KB partitions for L1. The rest
  -- is shared memory.
  -- Equal is the mid-point between high and low. It should be
  -- equivalent to low + 16KB.
  -- Maxwell and Pascal have dedicated shared memory.
  --*
  -- * Partitioned global caching mode support
  --  

   function cudaOccPartitionedGlobalCachingModeSupport (limit : access cudaOccPartitionedGCSupport; properties : access constant cudaOccDeviceProp) return cudaOccError;  -- /usr/local/cuda-8.0/include/cuda_occupancy.h:732
   pragma Import (CPP, cudaOccPartitionedGlobalCachingModeSupport, "_ZL42cudaOccPartitionedGlobalCachingModeSupportP32cudaOccPartitionedGCSupport_enumPK17cudaOccDeviceProp");

  --/////////////////////////////////////////////
  --            User Input Sanity              //
  --/////////////////////////////////////////////
   function cudaOccDevicePropCheck (properties : access constant cudaOccDeviceProp) return cudaOccError;  -- /usr/local/cuda-8.0/include/cuda_occupancy.h:752
   pragma Import (CPP, cudaOccDevicePropCheck, "_ZL22cudaOccDevicePropCheckPK17cudaOccDeviceProp");

  -- Verify device properties
  -- Each of these limits must be a positive number.
  -- Compute capacity is checked during the occupancy calculation
   function cudaOccFuncAttributesCheck (attributes : access constant cudaOccFuncAttributes) return cudaOccError;  -- /usr/local/cuda-8.0/include/cuda_occupancy.h:774
   pragma Import (CPP, cudaOccFuncAttributesCheck, "_ZL26cudaOccFuncAttributesCheckPK21cudaOccFuncAttributes");

  -- Verify function attributes
  -- Compiler may choose not to use
  -- any register (empty kernels,
  -- etc.)
   function cudaOccDeviceStateCheck (state : access constant cudaOccDeviceState) return cudaOccError;  -- /usr/local/cuda-8.0/include/cuda_occupancy.h:788
   pragma Import (CPP, cudaOccDeviceStateCheck, "_ZL23cudaOccDeviceStateCheckPK18cudaOccDeviceState");

  -- Placeholder
   function cudaOccInputCheck
     (properties : access constant cudaOccDeviceProp;
      attributes : access constant cudaOccFuncAttributes;
      state : access constant cudaOccDeviceState) return cudaOccError;  -- /usr/local/cuda-8.0/include/cuda_occupancy.h:796
   pragma Import (CPP, cudaOccInputCheck, "_ZL17cudaOccInputCheckPK17cudaOccDevicePropPK21cudaOccFuncAttributesPK18cudaOccDeviceState");

  --/////////////////////////////////////////////
  --    Occupancy calculation Functions        //
  --/////////////////////////////////////////////
   function cudaOccPartitionedGCForced (properties : access constant cudaOccDeviceProp) return int;  -- /usr/local/cuda-8.0/include/cuda_occupancy.h:825
   pragma Import (CPP, cudaOccPartitionedGCForced, "_ZL26cudaOccPartitionedGCForcedPK17cudaOccDeviceProp");

   function cudaOccPartitionedGCExpected (properties : access constant cudaOccDeviceProp; attributes : access constant cudaOccFuncAttributes) return cudaOccPartitionedGCConfig;  -- /usr/local/cuda-8.0/include/cuda_occupancy.h:835
   pragma Import (CPP, cudaOccPartitionedGCExpected, "_ZL28cudaOccPartitionedGCExpectedPK17cudaOccDevicePropPK21cudaOccFuncAttributes");

  -- Warp limit
   function cudaOccMaxBlocksPerSMWarpsLimit
     (limit : access int;
      gcConfig : cudaOccPartitionedGCConfig;
      properties : access constant cudaOccDeviceProp;
      attributes : access constant cudaOccFuncAttributes;
      blockSize : int) return cudaOccError;  -- /usr/local/cuda-8.0/include/cuda_occupancy.h:859
   pragma Import (CPP, cudaOccMaxBlocksPerSMWarpsLimit, "_ZL31cudaOccMaxBlocksPerSMWarpsLimitPi31cudaOccPartitionedGCConfig_enumPK17cudaOccDevicePropPK21cudaOccFuncAttributesi");

  -- If partitioned global caching is on, then a CTA can only use a SM
  -- partition (a half SM), and thus a half of the warp slots
  -- available per SM
  -- On hardware that supports partitioned global caching, each half SM is
  -- guaranteed to support at least 32 warps (maximum number of warps of a
  -- CTA), so caching will not cause 0 occupancy due to insufficient warp
  -- allocation slots.
  -- Shared memory limit
   function cudaOccMaxBlocksPerSMSmemLimit
     (limit : access int;
      result : access cudaOccResult;
      properties : access constant cudaOccDeviceProp;
      attributes : access constant cudaOccFuncAttributes;
      state : access constant cudaOccDeviceState;
      blockSize : int;
      dynamicSmemSize : stddef_h.size_t) return cudaOccError;  -- /usr/local/cuda-8.0/include/cuda_occupancy.h:908
   pragma Import (CPP, cudaOccMaxBlocksPerSMSmemLimit, "_ZL30cudaOccMaxBlocksPerSMSmemLimitPiP13cudaOccResultPK17cudaOccDevicePropPK21cudaOccFuncAttributesPK18cudaOccDeviceStateim");

  -- Obtain the user preferred shared memory size. This setting is ignored if
  -- user requests more shared memory than preferred.
  -- User requested shared memory limit is used as long as it is greater
  -- than the total shared memory used per CTA, i.e. as long as at least
  -- one CTA can be launched. Otherwise, the maximum shared memory limit
  -- is used instead.
   function cudaOccMaxBlocksPerSMRegsLimit
     (limit : access int;
      gcConfig : access cudaOccPartitionedGCConfig;
      result : access cudaOccResult;
      properties : access constant cudaOccDeviceProp;
      attributes : access constant cudaOccFuncAttributes;
      blockSize : int) return cudaOccError;  -- /usr/local/cuda-8.0/include/cuda_occupancy.h:973
   pragma Import (CPP, cudaOccMaxBlocksPerSMRegsLimit, "_ZL30cudaOccMaxBlocksPerSMRegsLimitPiP31cudaOccPartitionedGCConfig_enumP13cudaOccResultPK17cudaOccDevicePropPK21cudaOccFuncAttributesi");

  -- Fermi requires special handling of certain register usage
  -- GPUs of compute capability 2.x and higher allocate registers to warps
  -- Number of regs per warp is regs per thread x warp size, rounded up to
  -- register allocation granularity
  -- Hardware verifies if a launch fits the per-CTA register limit. For
  -- historical reasons, the verification logic assumes register
  -- allocations are made to all partitions simultaneously. Therefore, to
  -- simulate the hardware check, the warp allocation needs to be rounded
  -- up to the number of partitions.
  -- Hardware check
  -- Software check
  -- Registers are allocated in each sub-partition. The max number
  -- of warps that can fit on an SM is equal to the max number of
  -- warps per sub-partition x number of sub-partitions.
  -- If partitioned global caching is on, then a CTA can only
  -- use a half SM, and thus a half of the registers available
  -- per SM
  -- Try again if partitioned global caching is not enabled, or if
  -- the CTA cannot fit on the SM with caching on. In the latter
  -- case, the device will automatically turn off caching, except
  -- if the device forces it. The user can also override this
  -- assumption with PARTITIONED_GC_ON_STRICT to calculate
  -- occupancy and launch configuration.
  --/////////////////////////////////
  --      API Implementations      //
  --/////////////////////////////////
   function cudaOccMaxActiveBlocksPerMultiprocessor
     (result : access cudaOccResult;
      properties : access constant cudaOccDeviceProp;
      attributes : access constant cudaOccFuncAttributes;
      state : access constant cudaOccDeviceState;
      blockSize : int;
      dynamicSmemSize : stddef_h.size_t) return cudaOccError;  -- /usr/local/cuda-8.0/include/cuda_occupancy.h:1094
   pragma Import (CPP, cudaOccMaxActiveBlocksPerMultiprocessor, "_ZL39cudaOccMaxActiveBlocksPerMultiprocessorP13cudaOccResultPK17cudaOccDevicePropPK21cudaOccFuncAttributesPK18cudaOccDeviceStateim");

  --/////////////////////////
  -- Check user input
  --/////////////////////////
  --/////////////////////////
  -- Initialization
  --/////////////////////////
  --/////////////////////////
  -- Compute occupancy
  --/////////////////////////
  -- Limits due to registers/SM
  -- Also compute if partitioned global caching has to be turned off
  -- Limits due to warps/SM
  -- Limits due to blocks/SM
  -- Limits due to shared memory/SM
  --/////////////////////////
  -- Overall occupancy
  --/////////////////////////
  -- Overall limit is min() of limits due to above reasons
  -- Fill in the return values
  -- Determine occupancy limiting factors
  -- Final occupancy
   function cudaOccMaxPotentialOccupancyBlockSize
     (minGridSize : access int;
      blockSize : access int;
      properties : access constant cudaOccDeviceProp;
      attributes : access constant cudaOccFuncAttributes;
      state : access constant cudaOccDeviceState;
      blockSizeToDynamicSMemSize : access function (arg1 : int) return stddef_h.size_t;
      dynamicSMemSize : stddef_h.size_t) return cudaOccError;  -- /usr/local/cuda-8.0/include/cuda_occupancy.h:1203
   pragma Import (CPP, cudaOccMaxPotentialOccupancyBlockSize, "_ZL37cudaOccMaxPotentialOccupancyBlockSizePiS_PK17cudaOccDevicePropPK21cudaOccFuncAttributesPK18cudaOccDeviceStatePFmiEm");

  -- Limits
  -- Recorded maximum
  -- Temporary
  --/////////////////////////
  -- Check user input
  --/////////////////////////
  --///////////////////////////////////////////////////////////////////////////////
  -- Try each block size, and pick the block size with maximum occupancy
  --///////////////////////////////////////////////////////////////////////////////
  -- Ignore dynamicSMemSize if the user provides a mapping
  -- Early out if we have reached the maximum
  --/////////////////////////
  -- Return best available
  --/////////////////////////
  -- Suggested min grid size to achieve a full machine launch
   function cudaOccMaxPotentialOccupancyBlockSize
     (minGridSize : access int;
      blockSize : access int;
      properties : access constant cudaOccDeviceProp;
      attributes : access constant cudaOccFuncAttributes;
      state : access constant cudaOccDeviceState;
      dynamicSMemSize : stddef_h.size_t) return cudaOccError;  -- /usr/local/cuda-8.0/include/cuda_occupancy.h:1310
   pragma Import (CPP, cudaOccMaxPotentialOccupancyBlockSize, "_ZN12_GLOBAL__N_137cudaOccMaxPotentialOccupancyBlockSizeEPiS0_PK17cudaOccDevicePropPK21cudaOccFuncAttributesPK18cudaOccDeviceStatem");

  -- Limits
  -- Recorded maximum
  -- Temporary
  --/////////////////////////
  -- Check user input
  --/////////////////////////
  --///////////////////////////////////////////////////////////////////////////////
  -- Try each block size, and pick the block size with maximum occupancy
  --///////////////////////////////////////////////////////////////////////////////
  -- Early out if we have reached the maximum
  --/////////////////////////
  -- Return best available
  --/////////////////////////
  -- Suggested min grid size to achieve a full machine launch
  -- namespace anonymous
end cuda_occupancy_h;

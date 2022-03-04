pragma Ada_2005;
pragma Style_Checks (Off);

with Interfaces.C; use Interfaces.C;
with System;
with Interfaces.C.Extensions;
with Interfaces.C.Strings;

package nvml_h is

   NVML_API_VERSION : constant := 8;  --  /usr/local/cuda-8.0/include/nvml.h:98
   NVML_API_VERSION_STR : aliased constant String := "8" & ASCII.NUL;  --  /usr/local/cuda-8.0/include/nvml.h:99
   --  unsupported macro: nvmlInit nvmlInit_v2
   --  unsupported macro: nvmlDeviceGetPciInfo nvmlDeviceGetPciInfo_v2
   --  unsupported macro: nvmlDeviceGetCount nvmlDeviceGetCount_v2
   --  unsupported macro: nvmlDeviceGetHandleByIndex nvmlDeviceGetHandleByIndex_v2
   --  unsupported macro: nvmlDeviceGetHandleByPciBusId nvmlDeviceGetHandleByPciBusId_v2

   NVML_VALUE_NOT_AVAILABLE : constant := (-1);  --  /usr/local/cuda-8.0/include/nvml.h:118

   NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE : constant := 16;  --  /usr/local/cuda-8.0/include/nvml.h:125

   NVML_NVLINK_MAX_LINKS : constant := 4;  --  /usr/local/cuda-8.0/include/nvml.h:216

   NVML_MAX_PHYSICAL_BRIDGE : constant := (128);  --  /usr/local/cuda-8.0/include/nvml.h:332

   nvmlFlagDefault : constant := 16#00#;  --  /usr/local/cuda-8.0/include/nvml.h:457

   nvmlFlagForce : constant := 16#01#;  --  /usr/local/cuda-8.0/include/nvml.h:459
   --  unsupported macro: nvmlEccBitType_t nvmlMemoryErrorType_t
   --  unsupported macro: NVML_SINGLE_BIT_ECC NVML_MEMORY_ERROR_TYPE_CORRECTED
   --  unsupported macro: NVML_DOUBLE_BIT_ECC NVML_MEMORY_ERROR_TYPE_UNCORRECTED

   nvmlEventTypeSingleBitEccError : constant := 16#0000000000000001#;  --  /usr/local/cuda-8.0/include/nvml.h:877

   nvmlEventTypeDoubleBitEccError : constant := 16#0000000000000002#;  --  /usr/local/cuda-8.0/include/nvml.h:883

   nvmlEventTypePState : constant := 16#0000000000000004#;  --  /usr/local/cuda-8.0/include/nvml.h:891

   nvmlEventTypeXidCriticalError : constant := 16#0000000000000008#;  --  /usr/local/cuda-8.0/include/nvml.h:894

   nvmlEventTypeClock : constant := 16#0000000000000010#;  --  /usr/local/cuda-8.0/include/nvml.h:900

   nvmlEventTypeNone : constant := 16#0000000000000000#;  --  /usr/local/cuda-8.0/include/nvml.h:903
   --  unsupported macro: nvmlEventTypeAll (nvmlEventTypeNone | nvmlEventTypeSingleBitEccError | nvmlEventTypeDoubleBitEccError | nvmlEventTypePState | nvmlEventTypeClock | nvmlEventTypeXidCriticalError )

   nvmlClocksThrottleReasonGpuIdle : constant := 16#0000000000000001#;  --  /usr/local/cuda-8.0/include/nvml.h:936

   nvmlClocksThrottleReasonApplicationsClocksSetting : constant := 16#0000000000000002#;  --  /usr/local/cuda-8.0/include/nvml.h:943
   --  unsupported macro: nvmlClocksThrottleReasonUserDefinedClocks nvmlClocksThrottleReasonApplicationsClocksSetting

   nvmlClocksThrottleReasonSwPowerCap : constant := 16#0000000000000004#;  --  /usr/local/cuda-8.0/include/nvml.h:957

   nvmlClocksThrottleReasonHwSlowdown : constant := 16#0000000000000008#;  --  /usr/local/cuda-8.0/include/nvml.h:972

   nvmlClocksThrottleReasonSyncBoost : constant := 16#0000000000000010#;  --  /usr/local/cuda-8.0/include/nvml.h:983

   nvmlClocksThrottleReasonUnknown : constant := 16#8000000000000000#;  --  /usr/local/cuda-8.0/include/nvml.h:986

   nvmlClocksThrottleReasonNone : constant := 16#0000000000000000#;  --  /usr/local/cuda-8.0/include/nvml.h:992
   --  unsupported macro: nvmlClocksThrottleReasonAll (nvmlClocksThrottleReasonNone | nvmlClocksThrottleReasonGpuIdle | nvmlClocksThrottleReasonApplicationsClocksSetting | nvmlClocksThrottleReasonSwPowerCap | nvmlClocksThrottleReasonHwSlowdown | nvmlClocksThrottleReasonSyncBoost | nvmlClocksThrottleReasonUnknown )

   NVML_DEVICE_INFOROM_VERSION_BUFFER_SIZE : constant := 16;  --  /usr/local/cuda-8.0/include/nvml.h:1134

   NVML_DEVICE_UUID_BUFFER_SIZE : constant := 80;  --  /usr/local/cuda-8.0/include/nvml.h:1139

   NVML_DEVICE_PART_NUMBER_BUFFER_SIZE : constant := 80;  --  /usr/local/cuda-8.0/include/nvml.h:1144

   NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE : constant := 80;  --  /usr/local/cuda-8.0/include/nvml.h:1149

   NVML_SYSTEM_NVML_VERSION_BUFFER_SIZE : constant := 80;  --  /usr/local/cuda-8.0/include/nvml.h:1154

   NVML_DEVICE_NAME_BUFFER_SIZE : constant := 64;  --  /usr/local/cuda-8.0/include/nvml.h:1159

   NVML_DEVICE_SERIAL_BUFFER_SIZE : constant := 30;  --  /usr/local/cuda-8.0/include/nvml.h:1164

   NVML_DEVICE_VBIOS_VERSION_BUFFER_SIZE : constant := 32;  --  /usr/local/cuda-8.0/include/nvml.h:1169

  -- * Copyright 1993-2016 NVIDIA Corporation.  All rights reserved.
  -- *
  -- * NOTICE TO USER:   
  -- *
  -- * This source code is subject to NVIDIA ownership rights under U.S. and 
  -- * international Copyright laws.  Users and possessors of this source code 
  -- * are hereby granted a nonexclusive, royalty-free license to use this code 
  -- * in individual and commercial software.
  -- *
  -- * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
  -- * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
  -- * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
  -- * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
  -- * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
  -- * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
  -- * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
  -- * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
  -- * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE 
  -- * OR PERFORMANCE OF THIS SOURCE CODE.  
  -- *
  -- * U.S. Government End Users.   This source code is a "commercial item" as 
  -- * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of 
  -- * "commercial computer  software"  and "commercial computer software 
  -- * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995) 
  -- * and is provided to the U.S. Government only as a commercial end item.  
  -- * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
  -- * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
  -- * source code with only those rights set forth herein. 
  -- *
  -- * Any use of this source code in individual and commercial software must 
  -- * include, in the user documentation and internal comments to the code,
  -- * the above Disclaimer and U.S. Government End Users Notice.
  --  

  -- 
  --NVML API Reference
  --The NVIDIA Management Library (NVML) is a C-based programmatic interface for monitoring and 
  --managing various states within NVIDIA Tesla &tm; GPUs. It is intended to be a platform for building
  --3rd party applications, and is also the underlying library for the NVIDIA-supported nvidia-smi
  --tool. NVML is thread-safe so it is safe to make simultaneous NVML calls from multiple threads.
  --API Documentation
  --Supported platforms:
  --- Windows:     Windows Server 2008 R2 64bit, Windows Server 2012 R2 64bit, Windows 7 64bit, Windows 8 64bit, Windows 10 64bit
  --- Linux:       32-bit and 64-bit
  --- Hypervisors: Windows Server 2008R2/2012 Hyper-V 64bit, Citrix XenServer 6.2 SP1+, VMware ESX 5.1/5.5
  --Supported products:
  --- Full Support
  --    - All Tesla products, starting with the Fermi architecture
  --    - All Quadro products, starting with the Fermi architecture
  --    - All GRID products, starting with the Kepler architecture
  --    - Selected GeForce Titan products
  --- Limited Support
  --    - All Geforce products, starting with the Fermi architecture
  --The NVML library can be found at \%ProgramW6432\%\\"NVIDIA Corporation"\\NVSMI\\ on Windows. It is
  --not be added to the system path by default. To dynamically link to NVML, add this path to the PATH 
  --environmental variable. To dynamically load NVML, call LoadLibrary with this path.
  --On Linux the NVML library will be found on the standard library path. For 64 bit Linux, both the 32 bit
  --and 64 bit NVML libraries will be installed.
  --Online documentation for this library is available at http://docs.nvidia.com/deploy/nvml-api/index.html
  -- 

  -- * On Windows, set up methods for DLL export
  -- * define NVML_STATIC_IMPORT when using nvml_loader library
  --  

  --*
  -- * NVML API versioning support
  --  

  --************************************************************************************************* 
  --* @defgroup nvmlDeviceStructs Device Structs
  -- *  @{
  --  

  --************************************************************************************************* 
  --*
  -- * Special constant that some fields take when they are not available.
  -- * Used when only part of the struct is not available.
  -- *
  -- * Each structure explicitly states when to check for this value.
  --  

   --  skipped empty struct nvmlDevice_st

   type nvmlDevice_t is new System.Address;  -- /usr/local/cuda-8.0/include/nvml.h:120

  --*
  -- * Buffer size guaranteed to be large enough for pci bus id
  --  

  --*
  -- * PCI information about a GPU device.
  --  

  --!< The tuple domain:bus:device.function PCI identifier (&amp; NULL terminator)
   subtype nvmlPciInfo_st_busId_array is Interfaces.C.char_array (0 .. 15);
   type nvmlPciInfo_st is record
      busId : aliased nvmlPciInfo_st_busId_array;  -- /usr/local/cuda-8.0/include/nvml.h:132
      domain : aliased unsigned;  -- /usr/local/cuda-8.0/include/nvml.h:133
      bus : aliased unsigned;  -- /usr/local/cuda-8.0/include/nvml.h:134
      device : aliased unsigned;  -- /usr/local/cuda-8.0/include/nvml.h:135
      pciDeviceId : aliased unsigned;  -- /usr/local/cuda-8.0/include/nvml.h:136
      pciSubSystemId : aliased unsigned;  -- /usr/local/cuda-8.0/include/nvml.h:139
      reserved0 : aliased unsigned;  -- /usr/local/cuda-8.0/include/nvml.h:142
      reserved1 : aliased unsigned;  -- /usr/local/cuda-8.0/include/nvml.h:143
      reserved2 : aliased unsigned;  -- /usr/local/cuda-8.0/include/nvml.h:144
      reserved3 : aliased unsigned;  -- /usr/local/cuda-8.0/include/nvml.h:145
   end record;
   pragma Convention (C_Pass_By_Copy, nvmlPciInfo_st);  -- /usr/local/cuda-8.0/include/nvml.h:130

  --!< The PCI domain on which the device's bus resides, 0 to 0xffff
  --!< The bus on which the device resides, 0 to 0xff
  --!< The device's id on the bus, 0 to 31
  --!< The combined 16-bit device id and 16-bit vendor id
  -- Added in NVML 2.285 API
  --!< The 32-bit Sub System Device ID
  -- NVIDIA reserved for internal use only
   subtype nvmlPciInfo_t is nvmlPciInfo_st;

  --*
  -- * Detailed ECC error counts for a device.
  -- *
  -- * @deprecated  Different GPU families can have different memory error counters
  -- *              See \ref nvmlDeviceGetMemoryErrorCounter
  --  

  --!< L1 cache errors
   type nvmlEccErrorCounts_st is record
      l1Cache : aliased Extensions.unsigned_long_long;  -- /usr/local/cuda-8.0/include/nvml.h:156
      l2Cache : aliased Extensions.unsigned_long_long;  -- /usr/local/cuda-8.0/include/nvml.h:157
      deviceMemory : aliased Extensions.unsigned_long_long;  -- /usr/local/cuda-8.0/include/nvml.h:158
      registerFile : aliased Extensions.unsigned_long_long;  -- /usr/local/cuda-8.0/include/nvml.h:159
   end record;
   pragma Convention (C_Pass_By_Copy, nvmlEccErrorCounts_st);  -- /usr/local/cuda-8.0/include/nvml.h:154

  --!< L2 cache errors
  --!< Device memory errors
  --!< Register file errors
   subtype nvmlEccErrorCounts_t is nvmlEccErrorCounts_st;

  --* 
  -- * Utilization information for a device.
  -- * Each sample period may be between 1 second and 1/6 second, depending on the product being queried.
  --  

  --!< Percent of time over the past sample period during which one or more kernels was executing on the GPU
   type nvmlUtilization_st is record
      gpu : aliased unsigned;  -- /usr/local/cuda-8.0/include/nvml.h:168
      memory : aliased unsigned;  -- /usr/local/cuda-8.0/include/nvml.h:169
   end record;
   pragma Convention (C_Pass_By_Copy, nvmlUtilization_st);  -- /usr/local/cuda-8.0/include/nvml.h:166

  --!< Percent of time over the past sample period during which global (device) memory was being read or written
   subtype nvmlUtilization_t is nvmlUtilization_st;

  --* 
  -- * Memory allocation information for a device.
  --  

  --!< Total installed FB memory (in bytes)
   type nvmlMemory_st is record
      total : aliased Extensions.unsigned_long_long;  -- /usr/local/cuda-8.0/include/nvml.h:177
      free : aliased Extensions.unsigned_long_long;  -- /usr/local/cuda-8.0/include/nvml.h:178
      used : aliased Extensions.unsigned_long_long;  -- /usr/local/cuda-8.0/include/nvml.h:179
   end record;
   pragma Convention (C_Pass_By_Copy, nvmlMemory_st);  -- /usr/local/cuda-8.0/include/nvml.h:175

  --!< Unallocated FB memory (in bytes)
  --!< Allocated FB memory (in bytes). Note that the driver/GPU always sets aside a small amount of memory for bookkeeping
   subtype nvmlMemory_t is nvmlMemory_st;

  --*
  -- * BAR1 Memory allocation Information for a device
  --  

  --!< Total BAR1 Memory (in bytes)
   type nvmlBAR1Memory_st is record
      bar1Total : aliased Extensions.unsigned_long_long;  -- /usr/local/cuda-8.0/include/nvml.h:187
      bar1Free : aliased Extensions.unsigned_long_long;  -- /usr/local/cuda-8.0/include/nvml.h:188
      bar1Used : aliased Extensions.unsigned_long_long;  -- /usr/local/cuda-8.0/include/nvml.h:189
   end record;
   pragma Convention (C_Pass_By_Copy, nvmlBAR1Memory_st);  -- /usr/local/cuda-8.0/include/nvml.h:185

  --!< Unallocated BAR1 Memory (in bytes)
  --!< Allocated Used Memory (in bytes)
   subtype nvmlBAR1Memory_t is nvmlBAR1Memory_st;

  --*
  -- * Information about running compute processes on the GPU
  --  

  --!< Process ID
   type nvmlProcessInfo_st is record
      pid : aliased unsigned;  -- /usr/local/cuda-8.0/include/nvml.h:197
      usedGpuMemory : aliased Extensions.unsigned_long_long;  -- /usr/local/cuda-8.0/include/nvml.h:198
   end record;
   pragma Convention (C_Pass_By_Copy, nvmlProcessInfo_st);  -- /usr/local/cuda-8.0/include/nvml.h:195

  --!< Amount of used GPU memory in bytes.
  --! Under WDDM, \ref NVML_VALUE_NOT_AVAILABLE is always reported
  --! because Windows KMD manages all the memory and not the NVIDIA driver
   subtype nvmlProcessInfo_t is nvmlProcessInfo_st;

  --*
  -- * Enum to represent type of bridge chip
  --  

   type nvmlBridgeChipType_enum is 
     (NVML_BRIDGE_CHIP_PLX,
      NVML_BRIDGE_CHIP_BRO4);
   pragma Convention (C, nvmlBridgeChipType_enum);  -- /usr/local/cuda-8.0/include/nvml.h:207

   subtype nvmlBridgeChipType_t is nvmlBridgeChipType_enum;

  --*
  -- * Maximum number of NvLink links supported 
  --  

  --*
  -- * Enum to represent the NvLink utilization counter packet units
  --  

   type nvmlNvLinkUtilizationCountUnits_enum is 
     (NVML_NVLINK_COUNTER_UNIT_CYCLES,
      NVML_NVLINK_COUNTER_UNIT_PACKETS,
      NVML_NVLINK_COUNTER_UNIT_BYTES,
      NVML_NVLINK_COUNTER_UNIT_COUNT);
   pragma Convention (C, nvmlNvLinkUtilizationCountUnits_enum);  -- /usr/local/cuda-8.0/include/nvml.h:221

  -- count by cycles
  -- count by packets
  -- count by bytes
  -- this must be last
   subtype nvmlNvLinkUtilizationCountUnits_t is nvmlNvLinkUtilizationCountUnits_enum;

  --*
  -- * Enum to represent the NvLink utilization counter packet types to count
  -- *  ** this is ONLY applicable with the units as packets or bytes
  -- *  ** as specified in \a nvmlNvLinkUtilizationCountUnits_t
  -- *  ** all packet filter descriptions are target GPU centric
  -- *  ** these can be "OR'd" together 
  --  

   subtype nvmlNvLinkUtilizationCountPktTypes_enum is unsigned;
   NVML_NVLINK_COUNTER_PKTFILTER_NOP : constant nvmlNvLinkUtilizationCountPktTypes_enum := 1;
   NVML_NVLINK_COUNTER_PKTFILTER_READ : constant nvmlNvLinkUtilizationCountPktTypes_enum := 2;
   NVML_NVLINK_COUNTER_PKTFILTER_WRITE : constant nvmlNvLinkUtilizationCountPktTypes_enum := 4;
   NVML_NVLINK_COUNTER_PKTFILTER_RATOM : constant nvmlNvLinkUtilizationCountPktTypes_enum := 8;
   NVML_NVLINK_COUNTER_PKTFILTER_NRATOM : constant nvmlNvLinkUtilizationCountPktTypes_enum := 16;
   NVML_NVLINK_COUNTER_PKTFILTER_FLUSH : constant nvmlNvLinkUtilizationCountPktTypes_enum := 32;
   NVML_NVLINK_COUNTER_PKTFILTER_RESPDATA : constant nvmlNvLinkUtilizationCountPktTypes_enum := 64;
   NVML_NVLINK_COUNTER_PKTFILTER_RESPNODATA : constant nvmlNvLinkUtilizationCountPktTypes_enum := 128;
   NVML_NVLINK_COUNTER_PKTFILTER_ALL : constant nvmlNvLinkUtilizationCountPktTypes_enum := 255;  -- /usr/local/cuda-8.0/include/nvml.h:238

  -- no operation packets
  -- read packets
  -- write packets
  -- reduction atomic requests
  -- non-reduction atomic requests
  -- flush requests
  -- responses with data
  -- responses without data
  -- all packets
   subtype nvmlNvLinkUtilizationCountPktTypes_t is nvmlNvLinkUtilizationCountPktTypes_enum;

  --* 
  -- * Struct to define the NVLINK counter controls
  --  

   type nvmlNvLinkUtilizationControl_st is record
      units : aliased nvmlNvLinkUtilizationCountUnits_t;  -- /usr/local/cuda-8.0/include/nvml.h:256
      pktfilter : aliased nvmlNvLinkUtilizationCountPktTypes_t;  -- /usr/local/cuda-8.0/include/nvml.h:257
   end record;
   pragma Convention (C_Pass_By_Copy, nvmlNvLinkUtilizationControl_st);  -- /usr/local/cuda-8.0/include/nvml.h:254

   subtype nvmlNvLinkUtilizationControl_t is nvmlNvLinkUtilizationControl_st;

  --*
  -- * Enum to represent NvLink queryable capabilities
  --  

   type nvmlNvLinkCapability_enum is 
     (NVML_NVLINK_CAP_P2P_SUPPORTED,
      NVML_NVLINK_CAP_SYSMEM_ACCESS,
      NVML_NVLINK_CAP_P2P_ATOMICS,
      NVML_NVLINK_CAP_SYSMEM_ATOMICS,
      NVML_NVLINK_CAP_SLI_BRIDGE,
      NVML_NVLINK_CAP_VALID,
      NVML_NVLINK_CAP_COUNT);
   pragma Convention (C, nvmlNvLinkCapability_enum);  -- /usr/local/cuda-8.0/include/nvml.h:263

  -- P2P over NVLink is supported
  -- Access to system memory is supported
  -- P2P atomics are supported
  -- System memory atomics are supported
  -- SLI is supported over this link
  -- Link is supported on this device
  -- should be last
   subtype nvmlNvLinkCapability_t is nvmlNvLinkCapability_enum;

  --*
  -- * Enum to represent NvLink queryable error counters
  --  

   type nvmlNvLinkErrorCounter_enum is 
     (NVML_NVLINK_ERROR_DL_REPLAY,
      NVML_NVLINK_ERROR_DL_RECOVERY,
      NVML_NVLINK_ERROR_DL_CRC_FLIT,
      NVML_NVLINK_ERROR_DL_CRC_DATA,
      NVML_NVLINK_ERROR_COUNT);
   pragma Convention (C, nvmlNvLinkErrorCounter_enum);  -- /usr/local/cuda-8.0/include/nvml.h:278

  -- Data link transmit replay error counter
  -- Data link transmit recovery error counter
  -- Data link receive flow control digit CRC error counter
  -- Data link receive data CRC error counter
  -- this must be last
   subtype nvmlNvLinkErrorCounter_t is nvmlNvLinkErrorCounter_enum;

  --*
  -- * Represents level relationships within a system between two GPUs
  -- * The enums are spaced to allow for future relationships
  --  

   subtype nvmlGpuLevel_enum is unsigned;
   NVML_TOPOLOGY_INTERNAL : constant nvmlGpuLevel_enum := 0;
   NVML_TOPOLOGY_SINGLE : constant nvmlGpuLevel_enum := 10;
   NVML_TOPOLOGY_MULTIPLE : constant nvmlGpuLevel_enum := 20;
   NVML_TOPOLOGY_HOSTBRIDGE : constant nvmlGpuLevel_enum := 30;
   NVML_TOPOLOGY_CPU : constant nvmlGpuLevel_enum := 40;
   NVML_TOPOLOGY_SYSTEM : constant nvmlGpuLevel_enum := 50;  -- /usr/local/cuda-8.0/include/nvml.h:293

  -- e.g. Tesla K80
  -- all devices that only need traverse a single PCIe switch
  -- all devices that need not traverse a host bridge
  -- all devices that are connected to the same host bridge
  -- all devices that are connected to the same CPU but possibly multiple host bridges
  -- all devices in the system
  -- there is purposefully no COUNT here because of the need for spacing above
   subtype nvmlGpuTopologyLevel_t is nvmlGpuLevel_enum;

  -- P2P Capability Index Status 
   type nvmlGpuP2PStatus_enum is 
     (NVML_P2P_STATUS_OK,
      NVML_P2P_STATUS_CHIPSET_NOT_SUPPORED,
      NVML_P2P_STATUS_GPU_NOT_SUPPORTED,
      NVML_P2P_STATUS_IOH_TOPOLOGY_NOT_SUPPORTED,
      NVML_P2P_STATUS_DISABLED_BY_REGKEY,
      NVML_P2P_STATUS_NOT_SUPPORTED,
      NVML_P2P_STATUS_UNKNOWN);
   pragma Convention (C, nvmlGpuP2PStatus_enum);  -- /usr/local/cuda-8.0/include/nvml.h:306

   subtype nvmlGpuP2PStatus_t is nvmlGpuP2PStatus_enum;

  -- P2P Capability Index 
   type nvmlGpuP2PCapsIndex_enum is 
     (NVML_P2P_CAPS_INDEX_READ,
      NVML_P2P_CAPS_INDEX_WRITE,
      NVML_P2P_CAPS_INDEX_NVLINK,
      NVML_P2P_CAPS_INDEX_ATOMICS,
      NVML_P2P_CAPS_INDEX_PROP,
      NVML_P2P_CAPS_INDEX_UNKNOWN);
   pragma Convention (C, nvmlGpuP2PCapsIndex_enum);  -- /usr/local/cuda-8.0/include/nvml.h:319

   subtype nvmlGpuP2PCapsIndex_t is nvmlGpuP2PCapsIndex_enum;

  --*
  -- * Maximum limit on Physical Bridges per Board
  --  

  --*
  -- * Information about the Bridge Chip Firmware
  --  

  --!< Type of Bridge Chip 
   type nvmlBridgeChipInfo_st is record
      c_type : aliased nvmlBridgeChipType_t;  -- /usr/local/cuda-8.0/include/nvml.h:339
      fwVersion : aliased unsigned;  -- /usr/local/cuda-8.0/include/nvml.h:340
   end record;
   pragma Convention (C_Pass_By_Copy, nvmlBridgeChipInfo_st);  -- /usr/local/cuda-8.0/include/nvml.h:337

  --!< Firmware Version. 0=Version is unavailable
   subtype nvmlBridgeChipInfo_t is nvmlBridgeChipInfo_st;

  --*
  -- * This structure stores the complete Hierarchy of the Bridge Chip within the board. The immediate 
  -- * bridge is stored at index 0 of bridgeInfoList, parent to immediate bridge is at index 1 and so forth.
  --  

  --!< Number of Bridge Chips on the Board
   type nvmlBridgeChipHierarchy_st_bridgeChipInfo_array is array (0 .. 127) of aliased nvmlBridgeChipInfo_t;
   type nvmlBridgeChipHierarchy_st is record
      bridgeCount : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/nvml.h:349
      bridgeChipInfo : aliased nvmlBridgeChipHierarchy_st_bridgeChipInfo_array;  -- /usr/local/cuda-8.0/include/nvml.h:350
   end record;
   pragma Convention (C_Pass_By_Copy, nvmlBridgeChipHierarchy_st);  -- /usr/local/cuda-8.0/include/nvml.h:347

  --!< Hierarchy of Bridge Chips on the board
   subtype nvmlBridgeChipHierarchy_t is nvmlBridgeChipHierarchy_st;

  --*
  -- *  Represents Type of Sampling Event
  --  

   type nvmlSamplingType_enum is 
     (NVML_TOTAL_POWER_SAMPLES,
      NVML_GPU_UTILIZATION_SAMPLES,
      NVML_MEMORY_UTILIZATION_SAMPLES,
      NVML_ENC_UTILIZATION_SAMPLES,
      NVML_DEC_UTILIZATION_SAMPLES,
      NVML_PROCESSOR_CLK_SAMPLES,
      NVML_MEMORY_CLK_SAMPLES,
      NVML_SAMPLINGTYPE_COUNT);
   pragma Convention (C, nvmlSamplingType_enum);  -- /usr/local/cuda-8.0/include/nvml.h:356

  --!< To represent total power drawn by GPU
  --!< To represent percent of time during which one or more kernels was executing on the GPU
  --!< To represent percent of time during which global (device) memory was being read or written
  --!< To represent percent of time during which NVENC remains busy
  --!< To represent percent of time during which NVDEC remains busy            
  --!< To represent processor clock samples
  --!< To represent memory clock samples
  -- Keep this last
   subtype nvmlSamplingType_t is nvmlSamplingType_enum;

  --*
  -- * Represents the queryable PCIe utilization counters
  --  

   type nvmlPcieUtilCounter_enum is 
     (NVML_PCIE_UTIL_TX_BYTES,
      NVML_PCIE_UTIL_RX_BYTES,
      NVML_PCIE_UTIL_COUNT);
   pragma Convention (C, nvmlPcieUtilCounter_enum);  -- /usr/local/cuda-8.0/include/nvml.h:373

  -- 1KB granularity
  -- 1KB granularity
  -- Keep this last
   subtype nvmlPcieUtilCounter_t is nvmlPcieUtilCounter_enum;

  --*
  -- * Represents the type for sample value returned
  --  

   type nvmlValueType_enum is 
     (NVML_VALUE_TYPE_DOUBLE,
      NVML_VALUE_TYPE_UNSIGNED_INT,
      NVML_VALUE_TYPE_UNSIGNED_LONG,
      NVML_VALUE_TYPE_UNSIGNED_LONG_LONG,
      NVML_VALUE_TYPE_COUNT);
   pragma Convention (C, nvmlValueType_enum);  -- /usr/local/cuda-8.0/include/nvml.h:385

  -- Keep this last
   subtype nvmlValueType_t is nvmlValueType_enum;

  --*
  -- * Union to represent different types of Value
  --  

  --!< If the value is double
   type nvmlValue_st (discr : unsigned := 0) is record
      case discr is
         when 0 =>
            dVal : aliased double;  -- /usr/local/cuda-8.0/include/nvml.h:402
         when 1 =>
            uiVal : aliased unsigned;  -- /usr/local/cuda-8.0/include/nvml.h:403
         when 2 =>
            ulVal : aliased unsigned_long;  -- /usr/local/cuda-8.0/include/nvml.h:404
         when others =>
            ullVal : aliased Extensions.unsigned_long_long;  -- /usr/local/cuda-8.0/include/nvml.h:405
      end case;
   end record;
   pragma Convention (C_Pass_By_Copy, nvmlValue_st);
   pragma Unchecked_Union (nvmlValue_st);  -- /usr/local/cuda-8.0/include/nvml.h:400

  --!< If the value is unsigned int
  --!< If the value is unsigned long
  --!< If the value is unsigned long long
   subtype nvmlValue_t is nvmlValue_st;

  --*
  -- * Information for Sample
  --  

  --!< CPU Timestamp in microseconds
   type nvmlSample_st is record
      timeStamp : aliased Extensions.unsigned_long_long;  -- /usr/local/cuda-8.0/include/nvml.h:413
      sampleValue : aliased nvmlValue_t;  -- /usr/local/cuda-8.0/include/nvml.h:414
   end record;
   pragma Convention (C_Pass_By_Copy, nvmlSample_st);  -- /usr/local/cuda-8.0/include/nvml.h:411

  --!< Sample Value
   subtype nvmlSample_t is nvmlSample_st;

  --*
  -- * Represents type of perf policy for which violation times can be queried 
  --  

   type nvmlPerfPolicyType_enum is 
     (NVML_PERF_POLICY_POWER,
      NVML_PERF_POLICY_THERMAL,
      NVML_PERF_POLICY_SYNC_BOOST,
      NVML_PERF_POLICY_COUNT);
   pragma Convention (C, nvmlPerfPolicyType_enum);  -- /usr/local/cuda-8.0/include/nvml.h:420

  -- Keep this last
   subtype nvmlPerfPolicyType_t is nvmlPerfPolicyType_enum;

  --*
  -- * Struct to hold perf policy violation status data
  --  

  --!< referenceTime represents CPU timestamp in microseconds
   type nvmlViolationTime_st is record
      referenceTime : aliased Extensions.unsigned_long_long;  -- /usr/local/cuda-8.0/include/nvml.h:435
      violationTime : aliased Extensions.unsigned_long_long;  -- /usr/local/cuda-8.0/include/nvml.h:436
   end record;
   pragma Convention (C_Pass_By_Copy, nvmlViolationTime_st);  -- /usr/local/cuda-8.0/include/nvml.h:433

  --!< violationTime in Nanoseconds
   subtype nvmlViolationTime_t is nvmlViolationTime_st;

  --* @}  
  --************************************************************************************************* 
  --* @defgroup nvmlDeviceEnumvs Device Enums
  -- *  @{
  --  

  --************************************************************************************************* 
  --* 
  -- * Generic enable/disable enum. 
  --  

   type nvmlEnableState_enum is 
     (NVML_FEATURE_DISABLED,
      NVML_FEATURE_ENABLED);
   pragma Convention (C, nvmlEnableState_enum);  -- /usr/local/cuda-8.0/include/nvml.h:450

  --!< Feature disabled 
  --!< Feature enabled
   subtype nvmlEnableState_t is nvmlEnableState_enum;

  --! Generic flag used to specify the default behavior of some functions. See description of particular functions for details.
  --! Generic flag used to force some behavior. See description of particular functions for details.
  --*
  -- *  * The Brand of the GPU
  -- *    

   type nvmlBrandType_enum is 
     (NVML_BRAND_UNKNOWN,
      NVML_BRAND_QUADRO,
      NVML_BRAND_TESLA,
      NVML_BRAND_NVS,
      NVML_BRAND_GRID,
      NVML_BRAND_GEFORCE,
      NVML_BRAND_COUNT);
   pragma Convention (C, nvmlBrandType_enum);  -- /usr/local/cuda-8.0/include/nvml.h:464

  -- Keep this last
   subtype nvmlBrandType_t is nvmlBrandType_enum;

  --*
  -- * Temperature thresholds.
  --  

   type nvmlTemperatureThresholds_enum is 
     (NVML_TEMPERATURE_THRESHOLD_SHUTDOWN,
      NVML_TEMPERATURE_THRESHOLD_SLOWDOWN,
      NVML_TEMPERATURE_THRESHOLD_COUNT);
   pragma Convention (C, nvmlTemperatureThresholds_enum);  -- /usr/local/cuda-8.0/include/nvml.h:480

  -- Temperature at which the GPU will shut down
  -- for HW protection
  -- Temperature at which the GPU will begin slowdown
  -- Keep this last
   subtype nvmlTemperatureThresholds_t is nvmlTemperatureThresholds_enum;

  --* 
  -- * Temperature sensors. 
  --  

   type nvmlTemperatureSensors_enum is 
     (NVML_TEMPERATURE_GPU,
      NVML_TEMPERATURE_COUNT);
   pragma Convention (C, nvmlTemperatureSensors_enum);  -- /usr/local/cuda-8.0/include/nvml.h:492

  --!< Temperature sensor for the GPU die
  -- Keep this last
   subtype nvmlTemperatureSensors_t is nvmlTemperatureSensors_enum;

  --* 
  -- * Compute mode. 
  -- *
  -- * NVML_COMPUTEMODE_EXCLUSIVE_PROCESS was added in CUDA 4.0.
  -- * Earlier CUDA versions supported a single exclusive mode, 
  -- * which is equivalent to NVML_COMPUTEMODE_EXCLUSIVE_THREAD in CUDA 4.0 and beyond.
  --  

   type nvmlComputeMode_enum is 
     (NVML_COMPUTEMODE_DEFAULT,
      NVML_COMPUTEMODE_EXCLUSIVE_THREAD,
      NVML_COMPUTEMODE_PROHIBITED,
      NVML_COMPUTEMODE_EXCLUSIVE_PROCESS,
      NVML_COMPUTEMODE_COUNT);
   pragma Convention (C, nvmlComputeMode_enum);  -- /usr/local/cuda-8.0/include/nvml.h:507

  --!< Default compute mode -- multiple contexts per device
  --!< Support Removed
  --!< Compute-prohibited mode -- no contexts per device
  --!< Compute-exclusive-process mode -- only one context per device, usable from multiple threads at a time
  -- Keep this last
   subtype nvmlComputeMode_t is nvmlComputeMode_enum;

  --* 
  -- * ECC bit types.
  -- *
  -- * @deprecated See \ref nvmlMemoryErrorType_t for a more flexible type
  --  

  --*
  -- * Single bit ECC errors
  -- *
  -- * @deprecated Mapped to \ref NVML_MEMORY_ERROR_TYPE_CORRECTED
  --  

  --*
  -- * Double bit ECC errors
  -- *
  -- * @deprecated Mapped to \ref NVML_MEMORY_ERROR_TYPE_UNCORRECTED
  --  

  --*
  -- * Memory error types
  --  

   type nvmlMemoryErrorType_enum is 
     (NVML_MEMORY_ERROR_TYPE_CORRECTED,
      NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
      NVML_MEMORY_ERROR_TYPE_COUNT);
   pragma Convention (C, nvmlMemoryErrorType_enum);  -- /usr/local/cuda-8.0/include/nvml.h:542

  --*
  --     * A memory error that was corrected
  --     * 
  --     * For ECC errors, these are single bit errors
  --     * For Texture memory, these are errors fixed by resend
  --      

  --*
  --     * A memory error that was not corrected
  --     * 
  --     * For ECC errors, these are double bit errors
  --     * For Texture memory, these are errors where the resend fails
  --      

  -- Keep this last
  --!< Count of memory error types
   subtype nvmlMemoryErrorType_t is nvmlMemoryErrorType_enum;

  --* 
  -- * ECC counter types. 
  -- *
  -- * Note: Volatile counts are reset each time the driver loads. On Windows this is once per boot. On Linux this can be more frequent.
  -- *       On Linux the driver unloads when no active clients exist. If persistence mode is enabled or there is always a driver 
  -- *       client active (e.g. X11), then Linux also sees per-boot behavior. If not, volatile counts are reset each time a compute app
  -- *       is run.
  --  

   type nvmlEccCounterType_enum is 
     (NVML_VOLATILE_ECC,
      NVML_AGGREGATE_ECC,
      NVML_ECC_COUNTER_TYPE_COUNT);
   pragma Convention (C, nvmlEccCounterType_enum);  -- /usr/local/cuda-8.0/include/nvml.h:573

  --!< Volatile counts are reset each time the driver loads.
  --!< Aggregate counts persist across reboots (i.e. for the lifetime of the device)
  -- Keep this last
  --!< Count of memory counter types
   subtype nvmlEccCounterType_t is nvmlEccCounterType_enum;

  --* 
  -- * Clock types. 
  -- * 
  -- * All speeds are in Mhz.
  --  

   type nvmlClockType_enum is 
     (NVML_CLOCK_GRAPHICS,
      NVML_CLOCK_SM,
      NVML_CLOCK_MEM,
      NVML_CLOCK_VIDEO,
      NVML_CLOCK_COUNT);
   pragma Convention (C, nvmlClockType_enum);  -- /usr/local/cuda-8.0/include/nvml.h:587

  --!< Graphics clock domain
  --!< SM clock domain
  --!< Memory clock domain
  --!< Video encoder/decoder clock domain
  -- Keep this last
  --<! Count of clock types
   subtype nvmlClockType_t is nvmlClockType_enum;

  --*
  -- * Clock Ids.  These are used in combination with nvmlClockType_t
  -- * to specify a single clock value.
  --  

   type nvmlClockId_enum is 
     (NVML_CLOCK_ID_CURRENT,
      NVML_CLOCK_ID_APP_CLOCK_TARGET,
      NVML_CLOCK_ID_APP_CLOCK_DEFAULT,
      NVML_CLOCK_ID_CUSTOMER_BOOST_MAX,
      NVML_CLOCK_ID_COUNT);
   pragma Convention (C, nvmlClockId_enum);  -- /usr/local/cuda-8.0/include/nvml.h:602

  --!< Current actual clock value
  --!< Target application clock
  --!< Default application clock target
  --!< OEM-defined maximum clock rate
  --Keep this last
  --<! Count of Clock Ids.
   subtype nvmlClockId_t is nvmlClockId_enum;

  --* 
  -- * Driver models. 
  -- *
  -- * Windows only.
  --  

   type nvmlDriverModel_enum is 
     (NVML_DRIVER_WDDM,
      NVML_DRIVER_WDM);
   pragma Convention (C, nvmlDriverModel_enum);  -- /usr/local/cuda-8.0/include/nvml.h:618

  --!< WDDM driver model -- GPU treated as a display device
  --!< WDM (TCC) model (recommended) -- GPU treated as a generic device
   subtype nvmlDriverModel_t is nvmlDriverModel_enum;

  --*
  -- * Allowed PStates.
  --  

   subtype nvmlPStates_enum is unsigned;
   NVML_PSTATE_0 : constant nvmlPStates_enum := 0;
   NVML_PSTATE_1 : constant nvmlPStates_enum := 1;
   NVML_PSTATE_2 : constant nvmlPStates_enum := 2;
   NVML_PSTATE_3 : constant nvmlPStates_enum := 3;
   NVML_PSTATE_4 : constant nvmlPStates_enum := 4;
   NVML_PSTATE_5 : constant nvmlPStates_enum := 5;
   NVML_PSTATE_6 : constant nvmlPStates_enum := 6;
   NVML_PSTATE_7 : constant nvmlPStates_enum := 7;
   NVML_PSTATE_8 : constant nvmlPStates_enum := 8;
   NVML_PSTATE_9 : constant nvmlPStates_enum := 9;
   NVML_PSTATE_10 : constant nvmlPStates_enum := 10;
   NVML_PSTATE_11 : constant nvmlPStates_enum := 11;
   NVML_PSTATE_12 : constant nvmlPStates_enum := 12;
   NVML_PSTATE_13 : constant nvmlPStates_enum := 13;
   NVML_PSTATE_14 : constant nvmlPStates_enum := 14;
   NVML_PSTATE_15 : constant nvmlPStates_enum := 15;
   NVML_PSTATE_UNKNOWN : constant nvmlPStates_enum := 32;  -- /usr/local/cuda-8.0/include/nvml.h:627

  --!< Performance state 0 -- Maximum Performance
  --!< Performance state 1 
  --!< Performance state 2
  --!< Performance state 3
  --!< Performance state 4
  --!< Performance state 5
  --!< Performance state 6
  --!< Performance state 7
  --!< Performance state 8
  --!< Performance state 9
  --!< Performance state 10
  --!< Performance state 11
  --!< Performance state 12
  --!< Performance state 13
  --!< Performance state 14
  --!< Performance state 15 -- Minimum Performance 
  --!< Unknown performance state
   subtype nvmlPstates_t is nvmlPStates_enum;

  --*
  -- * GPU Operation Mode
  -- *
  -- * GOM allows to reduce power usage and optimize GPU throughput by disabling GPU features.
  -- *
  -- * Each GOM is designed to meet specific user needs.
  --  

   type nvmlGom_enum is 
     (NVML_GOM_ALL_ON,
      NVML_GOM_COMPUTE,
      NVML_GOM_LOW_DP);
   pragma Convention (C, nvmlGom_enum);  -- /usr/local/cuda-8.0/include/nvml.h:655

  --!< Everything is enabled and running at full speed
  --!< Designed for running only compute tasks. Graphics operations
  --!< are not allowed
  --!< Designed for running graphics applications that don't require
  --!< high bandwidth double precision
   subtype nvmlGpuOperationMode_t is nvmlGom_enum;

  --* 
  -- * Available infoROM objects.
  --  

   type nvmlInforomObject_enum is 
     (NVML_INFOROM_OEM,
      NVML_INFOROM_ECC,
      NVML_INFOROM_POWER,
      NVML_INFOROM_COUNT);
   pragma Convention (C, nvmlInforomObject_enum);  -- /usr/local/cuda-8.0/include/nvml.h:669

  --!< An object defined by OEM
  --!< The ECC object determining the level of ECC support
  --!< The power management object
  -- Keep this last
  --!< This counts the number of infoROM objects the driver knows about
   subtype nvmlInforomObject_t is nvmlInforomObject_enum;

  --* 
  -- * Return values for NVML API calls. 
  --  

   subtype nvmlReturn_enum is unsigned;
   NVML_SUCCESS : constant nvmlReturn_enum := 0;
   NVML_ERROR_UNINITIALIZED : constant nvmlReturn_enum := 1;
   NVML_ERROR_INVALID_ARGUMENT : constant nvmlReturn_enum := 2;
   NVML_ERROR_NOT_SUPPORTED : constant nvmlReturn_enum := 3;
   NVML_ERROR_NO_PERMISSION : constant nvmlReturn_enum := 4;
   NVML_ERROR_ALREADY_INITIALIZED : constant nvmlReturn_enum := 5;
   NVML_ERROR_NOT_FOUND : constant nvmlReturn_enum := 6;
   NVML_ERROR_INSUFFICIENT_SIZE : constant nvmlReturn_enum := 7;
   NVML_ERROR_INSUFFICIENT_POWER : constant nvmlReturn_enum := 8;
   NVML_ERROR_DRIVER_NOT_LOADED : constant nvmlReturn_enum := 9;
   NVML_ERROR_TIMEOUT : constant nvmlReturn_enum := 10;
   NVML_ERROR_IRQ_ISSUE : constant nvmlReturn_enum := 11;
   NVML_ERROR_LIBRARY_NOT_FOUND : constant nvmlReturn_enum := 12;
   NVML_ERROR_FUNCTION_NOT_FOUND : constant nvmlReturn_enum := 13;
   NVML_ERROR_CORRUPTED_INFOROM : constant nvmlReturn_enum := 14;
   NVML_ERROR_GPU_IS_LOST : constant nvmlReturn_enum := 15;
   NVML_ERROR_RESET_REQUIRED : constant nvmlReturn_enum := 16;
   NVML_ERROR_OPERATING_SYSTEM : constant nvmlReturn_enum := 17;
   NVML_ERROR_LIB_RM_VERSION_MISMATCH : constant nvmlReturn_enum := 18;
   NVML_ERROR_IN_USE : constant nvmlReturn_enum := 19;
   NVML_ERROR_NO_DATA : constant nvmlReturn_enum := 20;
   NVML_ERROR_UNKNOWN : constant nvmlReturn_enum := 999;  -- /usr/local/cuda-8.0/include/nvml.h:682

  --!< The operation was successful
  --!< NVML was not first initialized with nvmlInit()
  --!< A supplied argument is invalid
  --!< The requested operation is not available on target device
  --!< The current user does not have permission for operation
  --!< Deprecated: Multiple initializations are now allowed through ref counting
  --!< A query to find an object was unsuccessful
  --!< An input argument is not large enough
  --!< A device's external power cables are not properly attached
  --!< NVIDIA driver is not loaded
  --!< User provided timeout passed
  --!< NVIDIA Kernel detected an interrupt issue with a GPU
  --!< NVML Shared Library couldn't be found or loaded
  --!< Local version of NVML doesn't implement this function
  --!< infoROM is corrupted
  --!< The GPU has fallen off the bus or has otherwise become inaccessible
  --!< The GPU requires a reset before it can be used again
  --!< The GPU control device has been blocked by the operating system/cgroups
  --!< RM detects a driver/library version mismatch
  --!< An operation cannot be performed because the GPU is currently in use
  --!< No data
  --!< An internal driver error occurred
   subtype nvmlReturn_t is nvmlReturn_enum;

  --*
  -- * Memory locations
  -- *
  -- * See \ref nvmlDeviceGetMemoryErrorCounter
  --  

   type nvmlMemoryLocation_enum is 
     (NVML_MEMORY_LOCATION_L1_CACHE,
      NVML_MEMORY_LOCATION_L2_CACHE,
      NVML_MEMORY_LOCATION_DEVICE_MEMORY,
      NVML_MEMORY_LOCATION_REGISTER_FILE,
      NVML_MEMORY_LOCATION_TEXTURE_MEMORY,
      NVML_MEMORY_LOCATION_TEXTURE_SHM,
      NVML_MEMORY_LOCATION_COUNT);
   pragma Convention (C, nvmlMemoryLocation_enum);  -- /usr/local/cuda-8.0/include/nvml.h:713

  --!< GPU L1 Cache
  --!< GPU L2 Cache
  --!< GPU Device Memory
  --!< GPU Register File
  --!< GPU Texture Memory
  --!< Shared memory
  -- Keep this last
  --!< This counts the number of memory locations the driver knows about
   subtype nvmlMemoryLocation_t is nvmlMemoryLocation_enum;

  --*
  -- * Causes for page retirement
  --  

   type nvmlPageRetirementCause_enum is 
     (NVML_PAGE_RETIREMENT_CAUSE_MULTIPLE_SINGLE_BIT_ECC_ERRORS,
      NVML_PAGE_RETIREMENT_CAUSE_DOUBLE_BIT_ECC_ERROR,
      NVML_PAGE_RETIREMENT_CAUSE_COUNT);
   pragma Convention (C, nvmlPageRetirementCause_enum);  -- /usr/local/cuda-8.0/include/nvml.h:729

  --!< Page was retired due to multiple single bit ECC error
  --!< Page was retired due to double bit ECC error
  -- Keep this last
   subtype nvmlPageRetirementCause_t is nvmlPageRetirementCause_enum;

  --*
  -- * API types that allow changes to default permission restrictions
  --  

   type nvmlRestrictedAPI_enum is 
     (NVML_RESTRICTED_API_SET_APPLICATION_CLOCKS,
      NVML_RESTRICTED_API_SET_AUTO_BOOSTED_CLOCKS,
      NVML_RESTRICTED_API_COUNT);
   pragma Convention (C, nvmlRestrictedAPI_enum);  -- /usr/local/cuda-8.0/include/nvml.h:741

  --!< APIs that change application clocks, see nvmlDeviceSetApplicationsClocks 
  --!< and see nvmlDeviceResetApplicationsClocks
  --!< APIs that enable/disable Auto Boosted clocks
  --!< see nvmlDeviceSetAutoBoostedClocksEnabled
  -- Keep this last
   subtype nvmlRestrictedAPI_t is nvmlRestrictedAPI_enum;

  --* @}  
  --************************************************************************************************* 
  --* @defgroup nvmlUnitStructs Unit Structs
  -- *  @{
  --  

  --************************************************************************************************* 
   --  skipped empty struct nvmlUnit_st

   type nvmlUnit_t is new System.Address;  -- /usr/local/cuda-8.0/include/nvml.h:759

  --* 
  -- * Description of HWBC entry 
  --  

   subtype nvmlHwbcEntry_st_firmwareVersion_array is Interfaces.C.char_array (0 .. 31);
   type nvmlHwbcEntry_st is record
      hwbcId : aliased unsigned;  -- /usr/local/cuda-8.0/include/nvml.h:766
      firmwareVersion : aliased nvmlHwbcEntry_st_firmwareVersion_array;  -- /usr/local/cuda-8.0/include/nvml.h:767
   end record;
   pragma Convention (C_Pass_By_Copy, nvmlHwbcEntry_st);  -- /usr/local/cuda-8.0/include/nvml.h:764

   subtype nvmlHwbcEntry_t is nvmlHwbcEntry_st;

  --* 
  -- * Fan state enum. 
  --  

   type nvmlFanState_enum is 
     (NVML_FAN_NORMAL,
      NVML_FAN_FAILED);
   pragma Convention (C, nvmlFanState_enum);  -- /usr/local/cuda-8.0/include/nvml.h:773

  --!< Fan is working properly
  --!< Fan has failed
   subtype nvmlFanState_t is nvmlFanState_enum;

  --* 
  -- * Led color enum. 
  --  

   type nvmlLedColor_enum is 
     (NVML_LED_COLOR_GREEN,
      NVML_LED_COLOR_AMBER);
   pragma Convention (C, nvmlLedColor_enum);  -- /usr/local/cuda-8.0/include/nvml.h:782

  --!< GREEN, indicates good health
  --!< AMBER, indicates problem
   subtype nvmlLedColor_t is nvmlLedColor_enum;

  --* 
  -- * LED states for an S-class unit.
  --  

  --!< If amber, a text description of the cause
   subtype nvmlLedState_st_cause_array is Interfaces.C.char_array (0 .. 255);
   type nvmlLedState_st is record
      cause : aliased nvmlLedState_st_cause_array;  -- /usr/local/cuda-8.0/include/nvml.h:794
      color : aliased nvmlLedColor_t;  -- /usr/local/cuda-8.0/include/nvml.h:795
   end record;
   pragma Convention (C_Pass_By_Copy, nvmlLedState_st);  -- /usr/local/cuda-8.0/include/nvml.h:792

  --!< GREEN or AMBER
   subtype nvmlLedState_t is nvmlLedState_st;

  --* 
  -- * Static S-class unit info.
  --  

  --!< Product name
   subtype nvmlUnitInfo_st_name_array is Interfaces.C.char_array (0 .. 95);
   subtype nvmlUnitInfo_st_id_array is Interfaces.C.char_array (0 .. 95);
   subtype nvmlUnitInfo_st_serial_array is Interfaces.C.char_array (0 .. 95);
   subtype nvmlUnitInfo_st_firmwareVersion_array is Interfaces.C.char_array (0 .. 95);
   type nvmlUnitInfo_st is record
      name : aliased nvmlUnitInfo_st_name_array;  -- /usr/local/cuda-8.0/include/nvml.h:803
      id : aliased nvmlUnitInfo_st_id_array;  -- /usr/local/cuda-8.0/include/nvml.h:804
      serial : aliased nvmlUnitInfo_st_serial_array;  -- /usr/local/cuda-8.0/include/nvml.h:805
      firmwareVersion : aliased nvmlUnitInfo_st_firmwareVersion_array;  -- /usr/local/cuda-8.0/include/nvml.h:806
   end record;
   pragma Convention (C_Pass_By_Copy, nvmlUnitInfo_st);  -- /usr/local/cuda-8.0/include/nvml.h:801

  --!< Product identifier
  --!< Product serial number
  --!< Firmware version
   subtype nvmlUnitInfo_t is nvmlUnitInfo_st;

  --* 
  -- * Power usage information for an S-class unit.
  -- * The power supply state is a human readable string that equals "Normal" or contains
  -- * a combination of "Abnormal" plus one or more of the following:
  -- *    
  -- *    - High voltage
  -- *    - Fan failure
  -- *    - Heatsink temperature
  -- *    - Current limit
  -- *    - Voltage below UV alarm threshold
  -- *    - Low-voltage
  -- *    - SI2C remote off command
  -- *    - MOD_DISABLE input
  -- *    - Short pin transition 
  -- 

  --!< The power supply state
   subtype nvmlPSUInfo_st_state_array is Interfaces.C.char_array (0 .. 255);
   type nvmlPSUInfo_st is record
      state : aliased nvmlPSUInfo_st_state_array;  -- /usr/local/cuda-8.0/include/nvml.h:826
      current : aliased unsigned;  -- /usr/local/cuda-8.0/include/nvml.h:827
      voltage : aliased unsigned;  -- /usr/local/cuda-8.0/include/nvml.h:828
      power : aliased unsigned;  -- /usr/local/cuda-8.0/include/nvml.h:829
   end record;
   pragma Convention (C_Pass_By_Copy, nvmlPSUInfo_st);  -- /usr/local/cuda-8.0/include/nvml.h:824

  --!< PSU current (A)
  --!< PSU voltage (V)
  --!< PSU power draw (W)
   subtype nvmlPSUInfo_t is nvmlPSUInfo_st;

  --* 
  -- * Fan speed reading for a single fan in an S-class unit.
  --  

  --!< Fan speed (RPM)
   type nvmlUnitFanInfo_st is record
      speed : aliased unsigned;  -- /usr/local/cuda-8.0/include/nvml.h:837
      state : aliased nvmlFanState_t;  -- /usr/local/cuda-8.0/include/nvml.h:838
   end record;
   pragma Convention (C_Pass_By_Copy, nvmlUnitFanInfo_st);  -- /usr/local/cuda-8.0/include/nvml.h:835

  --!< Flag that indicates whether fan is working properly
   subtype nvmlUnitFanInfo_t is nvmlUnitFanInfo_st;

  --* 
  -- * Fan speed readings for an entire S-class unit.
  --  

  --!< Fan speed data for each fan
   type nvmlUnitFanSpeeds_st_fans_array is array (0 .. 23) of aliased nvmlUnitFanInfo_t;
   type nvmlUnitFanSpeeds_st is record
      fans : aliased nvmlUnitFanSpeeds_st_fans_array;  -- /usr/local/cuda-8.0/include/nvml.h:846
      count : aliased unsigned;  -- /usr/local/cuda-8.0/include/nvml.h:847
   end record;
   pragma Convention (C_Pass_By_Copy, nvmlUnitFanSpeeds_st);  -- /usr/local/cuda-8.0/include/nvml.h:844

  --!< Number of fans in unit
   subtype nvmlUnitFanSpeeds_t is nvmlUnitFanSpeeds_st;

  --* @}  
  --************************************************************************************************* 
  --* @addtogroup nvmlEvents 
  -- *  @{
  --  

  --************************************************************************************************* 
  --* 
  -- * Handle to an event set
  --  

   --  skipped empty struct nvmlEventSet_st

   type nvmlEventSet_t is new System.Address;  -- /usr/local/cuda-8.0/include/nvml.h:861

  --* @defgroup nvmlEventType Event Types
  -- * @{
  -- * Event Types which user can be notified about.
  -- * See description of particular functions for details.
  -- *
  -- * See \ref nvmlDeviceRegisterEvents and \ref nvmlDeviceGetSupportedEventTypes to check which devices 
  -- * support each event.
  -- *
  -- * Types can be combined with bitwise or operator '|' when passed to \ref nvmlDeviceRegisterEvents
  --  

  --! Event about single bit ECC errors
  --*
  -- * \note A corrected texture memory error is not an ECC error, so it does not generate a single bit event
  --  

  --! Event about double bit ECC errors
  --*
  -- * \note An uncorrected texture memory error is not an ECC error, so it does not generate a double bit event
  --  

  --! Event about PState changes
  --*
  -- *  \note On Fermi architecture PState changes are also an indicator that GPU is throttling down due to
  -- *  no work being executed on the GPU, power capping or thermal capping. In a typical situation,
  -- *  Fermi-based GPU should stay in P0 for the duration of the execution of the compute process.
  --  

  --! Event that Xid critical error occurred
  --! Event about clock changes
  --*
  -- * Kepler only
  --  

  --! Mask with no events
  --! Mask of all events
  --* @}  
  --* 
  -- * Information about occurred event
  --  

  --!< Specific device where the event occurred
   type nvmlEventData_st is record
      device : nvmlDevice_t;  -- /usr/local/cuda-8.0/include/nvml.h:919
      eventType : aliased Extensions.unsigned_long_long;  -- /usr/local/cuda-8.0/include/nvml.h:920
      eventData : aliased Extensions.unsigned_long_long;  -- /usr/local/cuda-8.0/include/nvml.h:921
   end record;
   pragma Convention (C_Pass_By_Copy, nvmlEventData_st);  -- /usr/local/cuda-8.0/include/nvml.h:917

  --!< Information about what specific event occurred
  --!< Stores last XID error for the device in the event of nvmlEventTypeXidCriticalError, 
  --  eventData is 0 for any other event. eventData is set as 999 for unknown xid error.
   subtype nvmlEventData_t is nvmlEventData_st;

  --* @}  
  --************************************************************************************************* 
  --* @addtogroup nvmlClocksThrottleReasons
  -- *  @{
  --  

  --************************************************************************************************* 
  --* Nothing is running on the GPU and the clocks are dropping to Idle state
  -- * \note This limiter may be removed in a later release
  --  

  --* GPU clocks are limited by current setting of applications clocks
  -- *
  -- * @see nvmlDeviceSetApplicationsClocks
  -- * @see nvmlDeviceGetApplicationsClock
  --  

  --* 
  -- * @deprecated Renamed to \ref nvmlClocksThrottleReasonApplicationsClocksSetting 
  -- *             as the name describes the situation more accurately.
  --  

  --* SW Power Scaling algorithm is reducing the clocks below requested clocks 
  -- *
  -- * @see nvmlDeviceGetPowerUsage
  -- * @see nvmlDeviceSetPowerManagementLimit
  -- * @see nvmlDeviceGetPowerManagementLimit
  --  

  --* HW Slowdown (reducing the core clocks by a factor of 2 or more) is engaged
  -- * 
  -- * This is an indicator of:
  -- *   - temperature being too high
  -- *   - External Power Brake Assertion is triggered (e.g. by the system power supply)
  -- *   - Power draw is too high and Fast Trigger protection is reducing the clocks
  -- *   - May be also reported during PState or clock change
  -- *      - This behavior may be removed in a later release.
  -- *
  -- * @see nvmlDeviceGetTemperature
  -- * @see nvmlDeviceGetTemperatureThreshold
  -- * @see nvmlDeviceGetPowerUsage
  --  

  --* Sync Boost
  -- *
  -- * This GPU has been added to a Sync boost group with nvidia-smi or DCGM in
  -- * order to maximize performance per watt. All GPUs in the sync boost group
  -- * will boost to the minimum possible clocks across the entire group. Look at
  -- * the throttle reasons for other GPUs in the system to see why those GPUs are
  -- * holding this one at lower clocks.
  -- *
  --  

  --* Some other unspecified factor is reducing the clocks  
  --* Bit mask representing no clocks throttling
  -- *
  -- * Clocks are as high as possible.
  -- *  

  --* Bit mask representing all supported clocks throttling reasons 
  -- * New reasons might be added to this list in the future
  --  

  --* @}  
  --************************************************************************************************* 
  --* @defgroup nvmlAccountingStats Accounting Statistics
  -- *  @{
  -- *
  -- *  Set of APIs designed to provide per process information about usage of GPU.
  -- *
  -- *  @note All accounting statistics and accounting mode live in nvidia driver and reset 
  -- *        to default (Disabled) when driver unloads.
  -- *        It is advised to run with persistence mode enabled.
  -- *
  -- *  @note Enabling accounting mode has no negative impact on the GPU performance.
  --  

  --************************************************************************************************* 
  --*
  -- * Describes accounting statistics of a process.
  --  

  --!< Percent of time over the process's lifetime during which one or more kernels was executing on the GPU.
   type nvmlAccountingStats_st_reserved_array is array (0 .. 4) of aliased unsigned;
   type nvmlAccountingStats_st is record
      gpuUtilization : aliased unsigned;  -- /usr/local/cuda-8.0/include/nvml.h:1025
      memoryUtilization : aliased unsigned;  -- /usr/local/cuda-8.0/include/nvml.h:1030
      maxMemoryUsage : aliased Extensions.unsigned_long_long;  -- /usr/local/cuda-8.0/include/nvml.h:1033
      time : aliased Extensions.unsigned_long_long;  -- /usr/local/cuda-8.0/include/nvml.h:1037
      startTime : aliased Extensions.unsigned_long_long;  -- /usr/local/cuda-8.0/include/nvml.h:1040
      isRunning : aliased unsigned;  -- /usr/local/cuda-8.0/include/nvml.h:1042
      reserved : aliased nvmlAccountingStats_st_reserved_array;  -- /usr/local/cuda-8.0/include/nvml.h:1044
   end record;
   pragma Convention (C_Pass_By_Copy, nvmlAccountingStats_st);  -- /usr/local/cuda-8.0/include/nvml.h:1024

  --! Utilization stats just like returned by \ref nvmlDeviceGetUtilizationRates but for the life time of a
  --! process (not just the last sample period).
  --! Set to NVML_VALUE_NOT_AVAILABLE if nvmlDeviceGetUtilizationRates is not supported
  --!< Percent of time over the process's lifetime during which global (device) memory was being read or written.
  --! Set to NVML_VALUE_NOT_AVAILABLE if nvmlDeviceGetUtilizationRates is not supported
  --!< Maximum total memory in bytes that was ever allocated by the process.
  --! Set to NVML_VALUE_NOT_AVAILABLE if nvmlProcessInfo_t->usedGpuMemory is not supported
  --!< Amount of time in ms during which the compute context was active. The time is reported as 0 if 
  --!< the process is not terminated
  --!< CPU Timestamp in usec representing start time for the process
  --!< Flag to represent if the process is running (1 for running, 0 for terminated)
  --!< Reserved for future use
   subtype nvmlAccountingStats_t is nvmlAccountingStats_st;

  --* @}  
  --************************************************************************************************* 
  --* @defgroup nvmlInitializationAndCleanup Initialization and Cleanup
  -- * This chapter describes the methods that handle NVML initialization and cleanup.
  -- * It is the user's responsibility to call \ref nvmlInit() before calling any other methods, and 
  -- * nvmlShutdown() once NVML is no longer being used.
  -- *  @{
  --  

  --************************************************************************************************* 
  --*
  -- * Initialize NVML, but don't initialize any GPUs yet.
  -- *
  -- * \note In NVML 5.319 new nvmlInit_v2 has replaced nvmlInit"_v1" (default in NVML 4.304 and older) that
  -- *       did initialize all GPU devices in the system.
  -- *       
  -- * This allows NVML to communicate with a GPU
  -- * when other GPUs in the system are unstable or in a bad state.  When using this API, GPUs are
  -- * discovered and initialized in nvmlDeviceGetHandleBy* functions instead.
  -- * 
  -- * \note To contrast nvmlInit_v2 with nvmlInit"_v1", NVML 4.304 nvmlInit"_v1" will fail when any detected GPU is in
  -- *       a bad or unstable state.
  -- * 
  -- * For all products.
  -- *
  -- * This method, should be called once before invoking any other methods in the library.
  -- * A reference count of the number of initializations is maintained.  Shutdown only occurs
  -- * when the reference count reaches zero.
  -- * 
  -- * @return 
  -- *         - \ref NVML_SUCCESS                   if NVML has been properly initialized
  -- *         - \ref NVML_ERROR_DRIVER_NOT_LOADED   if NVIDIA driver is not running
  -- *         - \ref NVML_ERROR_NO_PERMISSION       if NVML does not have permission to talk to the driver
  -- *         - \ref NVML_ERROR_UNKNOWN             on any unexpected error
  --  

   function nvmlInit_v2 return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:1083
   pragma Import (C, nvmlInit_v2, "nvmlInit_v2");

  --*
  -- * Shut down NVML by releasing all GPU resources previously allocated with \ref nvmlInit().
  -- * 
  -- * For all products.
  -- *
  -- * This method should be called after NVML work is done, once for each call to \ref nvmlInit()
  -- * A reference count of the number of initializations is maintained.  Shutdown only occurs
  -- * when the reference count reaches zero.  For backwards compatibility, no error is reported if
  -- * nvmlShutdown() is called more times than nvmlInit().
  -- * 
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if NVML has been properly shut down
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  --  

   function nvmlShutdown return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:1100
   pragma Import (C, nvmlShutdown, "nvmlShutdown");

  --* @}  
  --************************************************************************************************* 
  --* @defgroup nvmlErrorReporting Error reporting
  -- * This chapter describes helper functions for error reporting routines.
  -- *  @{
  --  

  --************************************************************************************************* 
  --*
  -- * Helper method for converting NVML error codes into readable strings.
  -- *
  -- * For all products.
  -- *
  -- * @param result                               NVML error code to convert
  -- *
  -- * @return String representation of the error.
  -- *
  --  

   function nvmlErrorString (result : nvmlReturn_t) return Interfaces.C.Strings.chars_ptr;  -- /usr/local/cuda-8.0/include/nvml.h:1121
   pragma Import (C, nvmlErrorString, "nvmlErrorString");

  --* @}  
  --************************************************************************************************* 
  --* @defgroup nvmlConstants Constants
  -- *  @{
  --  

  --************************************************************************************************* 
  --*
  -- * Buffer size guaranteed to be large enough for \ref nvmlDeviceGetInforomVersion and \ref nvmlDeviceGetInforomImageVersion
  --  

  --*
  -- * Buffer size guaranteed to be large enough for \ref nvmlDeviceGetUUID
  --  

  --*
  -- * Buffer size guaranteed to be large enough for \ref nvmlDeviceGetBoardPartNumber
  --  

  --*
  -- * Buffer size guaranteed to be large enough for \ref nvmlSystemGetDriverVersion
  --  

  --*
  -- * Buffer size guaranteed to be large enough for \ref nvmlSystemGetNVMLVersion
  --  

  --*
  -- * Buffer size guaranteed to be large enough for \ref nvmlDeviceGetName
  --  

  --*
  -- * Buffer size guaranteed to be large enough for \ref nvmlDeviceGetSerial
  --  

  --*
  -- * Buffer size guaranteed to be large enough for \ref nvmlDeviceGetVbiosVersion
  --  

  --* @}  
  --************************************************************************************************* 
  --* @defgroup nvmlSystemQueries System Queries
  -- * This chapter describes the queries that NVML can perform against the local system. These queries
  -- * are not device-specific.
  -- *  @{
  --  

  --************************************************************************************************* 
  --*
  -- * Retrieves the version of the system's graphics driver.
  -- * 
  -- * For all products.
  -- *
  -- * The version identifier is an alphanumeric string.  It will not exceed 80 characters in length
  -- * (including the NULL terminator).  See \ref nvmlConstants::NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE.
  -- *
  -- * @param version                              Reference in which to return the version identifier
  -- * @param length                               The maximum allowed length of the string returned in \a version
  -- *
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if \a version has been set
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a version is NULL
  -- *         - \ref NVML_ERROR_INSUFFICIENT_SIZE if \a length is too small 
  --  

   function nvmlSystemGetDriverVersion (version : Interfaces.C.Strings.chars_ptr; length : unsigned) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:1198
   pragma Import (C, nvmlSystemGetDriverVersion, "nvmlSystemGetDriverVersion");

  --*
  -- * Retrieves the version of the NVML library.
  -- * 
  -- * For all products.
  -- *
  -- * The version identifier is an alphanumeric string.  It will not exceed 80 characters in length
  -- * (including the NULL terminator).  See \ref nvmlConstants::NVML_SYSTEM_NVML_VERSION_BUFFER_SIZE.
  -- *
  -- * @param version                              Reference in which to return the version identifier
  -- * @param length                               The maximum allowed length of the string returned in \a version
  -- *
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if \a version has been set
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a version is NULL
  -- *         - \ref NVML_ERROR_INSUFFICIENT_SIZE if \a length is too small 
  --  

   function nvmlSystemGetNVMLVersion (version : Interfaces.C.Strings.chars_ptr; length : unsigned) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:1216
   pragma Import (C, nvmlSystemGetNVMLVersion, "nvmlSystemGetNVMLVersion");

  --*
  -- * Gets name of the process with provided process id
  -- *
  -- * For all products.
  -- *
  -- * Returned process name is cropped to provided length.
  -- * name string is encoded in ANSI.
  -- *
  -- * @param pid                                  The identifier of the process
  -- * @param name                                 Reference in which to return the process name
  -- * @param length                               The maximum allowed length of the string returned in \a name
  -- * 
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if \a name has been set
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a name is NULL or \a length is 0.
  -- *         - \ref NVML_ERROR_NOT_FOUND         if process doesn't exists
  -- *         - \ref NVML_ERROR_NO_PERMISSION     if the user doesn't have permission to perform this operation
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  --  

   function nvmlSystemGetProcessName
     (pid : unsigned;
      name : Interfaces.C.Strings.chars_ptr;
      length : unsigned) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:1238
   pragma Import (C, nvmlSystemGetProcessName, "nvmlSystemGetProcessName");

  --* @}  
  --************************************************************************************************* 
  --* @defgroup nvmlUnitQueries Unit Queries
  -- * This chapter describes that queries that NVML can perform against each unit. For S-class systems only.
  -- * In each case the device is identified with an nvmlUnit_t handle. This handle is obtained by 
  -- * calling \ref nvmlUnitGetHandleByIndex().
  -- *  @{
  --  

  --************************************************************************************************* 
  --*
  -- * Retrieves the number of units in the system.
  -- *
  -- * For S-class products.
  -- *
  -- * @param unitCount                            Reference in which to return the number of units
  -- * 
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if \a unitCount has been set
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a unitCount is NULL
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  --  

   function nvmlUnitGetCount (unitCount : access unsigned) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:1264
   pragma Import (C, nvmlUnitGetCount, "nvmlUnitGetCount");

  --*
  -- * Acquire the handle for a particular unit, based on its index.
  -- *
  -- * For S-class products.
  -- *
  -- * Valid indices are derived from the \a unitCount returned by \ref nvmlUnitGetCount(). 
  -- *   For example, if \a unitCount is 2 the valid indices are 0 and 1, corresponding to UNIT 0 and UNIT 1.
  -- *
  -- * The order in which NVML enumerates units has no guarantees of consistency between reboots.
  -- *
  -- * @param index                                The index of the target unit, >= 0 and < \a unitCount
  -- * @param unit                                 Reference in which to return the unit handle
  -- * 
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if \a unit has been set
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a index is invalid or \a unit is NULL
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  --  

   function nvmlUnitGetHandleByIndex (index : unsigned; unit : System.Address) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:1285
   pragma Import (C, nvmlUnitGetHandleByIndex, "nvmlUnitGetHandleByIndex");

  --*
  -- * Retrieves the static information associated with a unit.
  -- *
  -- * For S-class products.
  -- *
  -- * See \ref nvmlUnitInfo_t for details on available unit info.
  -- *
  -- * @param unit                                 The identifier of the target unit
  -- * @param info                                 Reference in which to return the unit information
  -- * 
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if \a info has been populated
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a unit is invalid or \a info is NULL
  --  

   function nvmlUnitGetUnitInfo (unit : nvmlUnit_t; info : access nvmlUnitInfo_t) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:1302
   pragma Import (C, nvmlUnitGetUnitInfo, "nvmlUnitGetUnitInfo");

  --*
  -- * Retrieves the LED state associated with this unit.
  -- *
  -- * For S-class products.
  -- *
  -- * See \ref nvmlLedState_t for details on allowed states.
  -- *
  -- * @param unit                                 The identifier of the target unit
  -- * @param state                                Reference in which to return the current LED state
  -- * 
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if \a state has been set
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a unit is invalid or \a state is NULL
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if this is not an S-class product
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  -- * 
  -- * @see nvmlUnitSetLedState()
  --  

   function nvmlUnitGetLedState (unit : nvmlUnit_t; state : access nvmlLedState_t) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:1323
   pragma Import (C, nvmlUnitGetLedState, "nvmlUnitGetLedState");

  --*
  -- * Retrieves the PSU stats for the unit.
  -- *
  -- * For S-class products.
  -- *
  -- * See \ref nvmlPSUInfo_t for details on available PSU info.
  -- *
  -- * @param unit                                 The identifier of the target unit
  -- * @param psu                                  Reference in which to return the PSU information
  -- * 
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if \a psu has been populated
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a unit is invalid or \a psu is NULL
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if this is not an S-class product
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  --  

   function nvmlUnitGetPsuInfo (unit : nvmlUnit_t; psu : access nvmlPSUInfo_t) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:1342
   pragma Import (C, nvmlUnitGetPsuInfo, "nvmlUnitGetPsuInfo");

  --*
  -- * Retrieves the temperature readings for the unit, in degrees C.
  -- *
  -- * For S-class products.
  -- *
  -- * Depending on the product, readings may be available for intake (type=0), 
  -- * exhaust (type=1) and board (type=2).
  -- *
  -- * @param unit                                 The identifier of the target unit
  -- * @param type                                 The type of reading to take
  -- * @param temp                                 Reference in which to return the intake temperature
  -- * 
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if \a temp has been populated
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a unit or \a type is invalid or \a temp is NULL
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if this is not an S-class product
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  --  

   function nvmlUnitGetTemperature
     (unit : nvmlUnit_t;
      c_type : unsigned;
      temp : access unsigned) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:1363
   pragma Import (C, nvmlUnitGetTemperature, "nvmlUnitGetTemperature");

  --*
  -- * Retrieves the fan speed readings for the unit.
  -- *
  -- * For S-class products.
  -- *
  -- * See \ref nvmlUnitFanSpeeds_t for details on available fan speed info.
  -- *
  -- * @param unit                                 The identifier of the target unit
  -- * @param fanSpeeds                            Reference in which to return the fan speed information
  -- * 
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if \a fanSpeeds has been populated
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a unit is invalid or \a fanSpeeds is NULL
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if this is not an S-class product
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  --  

   function nvmlUnitGetFanSpeedInfo (unit : nvmlUnit_t; fanSpeeds : access nvmlUnitFanSpeeds_t) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:1382
   pragma Import (C, nvmlUnitGetFanSpeedInfo, "nvmlUnitGetFanSpeedInfo");

  --*
  -- * Retrieves the set of GPU devices that are attached to the specified unit.
  -- *
  -- * For S-class products.
  -- *
  -- * The \a deviceCount argument is expected to be set to the size of the input \a devices array.
  -- *
  -- * @param unit                                 The identifier of the target unit
  -- * @param deviceCount                          Reference in which to provide the \a devices array size, and
  -- *                                             to return the number of attached GPU devices
  -- * @param devices                              Reference in which to return the references to the attached GPU devices
  -- * 
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if \a deviceCount and \a devices have been populated
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INSUFFICIENT_SIZE if \a deviceCount indicates that the \a devices array is too small
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a unit is invalid, either of \a deviceCount or \a devices is NULL
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  --  

   function nvmlUnitGetDevices
     (unit : nvmlUnit_t;
      deviceCount : access unsigned;
      devices : System.Address) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:1403
   pragma Import (C, nvmlUnitGetDevices, "nvmlUnitGetDevices");

  --*
  -- * Retrieves the IDs and firmware versions for any Host Interface Cards (HICs) in the system.
  -- * 
  -- * For S-class products.
  -- *
  -- * The \a hwbcCount argument is expected to be set to the size of the input \a hwbcEntries array.
  -- * The HIC must be connected to an S-class system for it to be reported by this function.
  -- *
  -- * @param hwbcCount                            Size of hwbcEntries array
  -- * @param hwbcEntries                          Array holding information about hwbc
  -- *
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if \a hwbcCount and \a hwbcEntries have been populated
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if either \a hwbcCount or \a hwbcEntries is NULL
  -- *         - \ref NVML_ERROR_INSUFFICIENT_SIZE if \a hwbcCount indicates that the \a hwbcEntries array is too small
  --  

   function nvmlSystemGetHicVersion (hwbcCount : access unsigned; hwbcEntries : access nvmlHwbcEntry_t) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:1422
   pragma Import (C, nvmlSystemGetHicVersion, "nvmlSystemGetHicVersion");

  --* @}  
  --************************************************************************************************* 
  --* @defgroup nvmlDeviceQueries Device Queries
  -- * This chapter describes that queries that NVML can perform against each device.
  -- * In each case the device is identified with an nvmlDevice_t handle. This handle is obtained by  
  -- * calling one of \ref nvmlDeviceGetHandleByIndex(), \ref nvmlDeviceGetHandleBySerial(),
  -- * \ref nvmlDeviceGetHandleByPciBusId(). or \ref nvmlDeviceGetHandleByUUID(). 
  -- *  @{
  --  

  --************************************************************************************************* 
  --*
  -- * Retrieves the number of compute devices in the system. A compute device is a single GPU.
  -- * 
  -- * For all products.
  -- *
  -- * Note: New nvmlDeviceGetCount_v2 (default in NVML 5.319) returns count of all devices in the system
  -- *       even if nvmlDeviceGetHandleByIndex_v2 returns NVML_ERROR_NO_PERMISSION for such device.
  -- *       Update your code to handle this error, or use NVML 4.304 or older nvml header file.
  -- *       For backward binary compatibility reasons _v1 version of the API is still present in the shared
  -- *       library.
  -- *       Old _v1 version of nvmlDeviceGetCount doesn't count devices that NVML has no permission to talk to.
  -- *
  -- * @param deviceCount                          Reference in which to return the number of accessible devices
  -- * 
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if \a deviceCount has been set
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a deviceCount is NULL
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  --  

   function nvmlDeviceGetCount_v2 (deviceCount : access unsigned) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:1455
   pragma Import (C, nvmlDeviceGetCount_v2, "nvmlDeviceGetCount_v2");

  --*
  -- * Acquire the handle for a particular device, based on its index.
  -- * 
  -- * For all products.
  -- *
  -- * Valid indices are derived from the \a accessibleDevices count returned by 
  -- *   \ref nvmlDeviceGetCount(). For example, if \a accessibleDevices is 2 the valid indices  
  -- *   are 0 and 1, corresponding to GPU 0 and GPU 1.
  -- *
  -- * The order in which NVML enumerates devices has no guarantees of consistency between reboots. For that reason it
  -- *   is recommended that devices be looked up by their PCI ids or UUID. See 
  -- *   \ref nvmlDeviceGetHandleByUUID() and \ref nvmlDeviceGetHandleByPciBusId().
  -- *
  -- * Note: The NVML index may not correlate with other APIs, such as the CUDA device index.
  -- *
  -- * Starting from NVML 5, this API causes NVML to initialize the target GPU
  -- * NVML may initialize additional GPUs if:
  -- *  - The target GPU is an SLI slave
  -- * 
  -- * Note: New nvmlDeviceGetCount_v2 (default in NVML 5.319) returns count of all devices in the system
  -- *       even if nvmlDeviceGetHandleByIndex_v2 returns NVML_ERROR_NO_PERMISSION for such device.
  -- *       Update your code to handle this error, or use NVML 4.304 or older nvml header file.
  -- *       For backward binary compatibility reasons _v1 version of the API is still present in the shared
  -- *       library.
  -- *       Old _v1 version of nvmlDeviceGetCount doesn't count devices that NVML has no permission to talk to.
  -- *
  -- *       This means that nvmlDeviceGetHandleByIndex_v2 and _v1 can return different devices for the same index.
  -- *       If you don't touch macros that map old (_v1) versions to _v2 versions at the top of the file you don't
  -- *       need to worry about that.
  -- *
  -- * @param index                                The index of the target GPU, >= 0 and < \a accessibleDevices
  -- * @param device                               Reference in which to return the device handle
  -- * 
  -- * @return 
  -- *         - \ref NVML_SUCCESS                  if \a device has been set
  -- *         - \ref NVML_ERROR_UNINITIALIZED      if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT   if \a index is invalid or \a device is NULL
  -- *         - \ref NVML_ERROR_INSUFFICIENT_POWER if any attached devices have improperly attached external power cables
  -- *         - \ref NVML_ERROR_NO_PERMISSION      if the user doesn't have permission to talk to this device
  -- *         - \ref NVML_ERROR_IRQ_ISSUE          if NVIDIA kernel detected an interrupt issue with the attached GPUs
  -- *         - \ref NVML_ERROR_GPU_IS_LOST        if the target GPU has fallen off the bus or is otherwise inaccessible
  -- *         - \ref NVML_ERROR_UNKNOWN            on any unexpected error
  -- *
  -- * @see nvmlDeviceGetIndex
  -- * @see nvmlDeviceGetCount
  --  

   function nvmlDeviceGetHandleByIndex_v2 (index : unsigned; device : System.Address) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:1503
   pragma Import (C, nvmlDeviceGetHandleByIndex_v2, "nvmlDeviceGetHandleByIndex_v2");

  --*
  -- * Acquire the handle for a particular device, based on its board serial number.
  -- *
  -- * For Fermi &tm; or newer fully supported devices.
  -- *
  -- * This number corresponds to the value printed directly on the board, and to the value returned by
  -- *   \ref nvmlDeviceGetSerial().
  -- *
  -- * @deprecated Since more than one GPU can exist on a single board this function is deprecated in favor 
  -- *             of \ref nvmlDeviceGetHandleByUUID.
  -- *             For dual GPU boards this function will return NVML_ERROR_INVALID_ARGUMENT.
  -- *
  -- * Starting from NVML 5, this API causes NVML to initialize the target GPU
  -- * NVML may initialize additional GPUs as it searches for the target GPU
  -- *
  -- * @param serial                               The board serial number of the target GPU
  -- * @param device                               Reference in which to return the device handle
  -- * 
  -- * @return 
  -- *         - \ref NVML_SUCCESS                  if \a device has been set
  -- *         - \ref NVML_ERROR_UNINITIALIZED      if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT   if \a serial is invalid, \a device is NULL or more than one
  -- *                                              device has the same serial (dual GPU boards)
  -- *         - \ref NVML_ERROR_NOT_FOUND          if \a serial does not match a valid device on the system
  -- *         - \ref NVML_ERROR_INSUFFICIENT_POWER if any attached devices have improperly attached external power cables
  -- *         - \ref NVML_ERROR_IRQ_ISSUE          if NVIDIA kernel detected an interrupt issue with the attached GPUs
  -- *         - \ref NVML_ERROR_GPU_IS_LOST        if any GPU has fallen off the bus or is otherwise inaccessible
  -- *         - \ref NVML_ERROR_UNKNOWN            on any unexpected error
  -- *
  -- * @see nvmlDeviceGetSerial
  -- * @see nvmlDeviceGetHandleByUUID
  --  

   function nvmlDeviceGetHandleBySerial (serial : Interfaces.C.Strings.chars_ptr; device : System.Address) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:1537
   pragma Import (C, nvmlDeviceGetHandleBySerial, "nvmlDeviceGetHandleBySerial");

  --*
  -- * Acquire the handle for a particular device, based on its globally unique immutable UUID associated with each device.
  -- *
  -- * For all products.
  -- *
  -- * @param uuid                                 The UUID of the target GPU
  -- * @param device                               Reference in which to return the device handle
  -- * 
  -- * Starting from NVML 5, this API causes NVML to initialize the target GPU
  -- * NVML may initialize additional GPUs as it searches for the target GPU
  -- *
  -- * @return 
  -- *         - \ref NVML_SUCCESS                  if \a device has been set
  -- *         - \ref NVML_ERROR_UNINITIALIZED      if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT   if \a uuid is invalid or \a device is null
  -- *         - \ref NVML_ERROR_NOT_FOUND          if \a uuid does not match a valid device on the system
  -- *         - \ref NVML_ERROR_INSUFFICIENT_POWER if any attached devices have improperly attached external power cables
  -- *         - \ref NVML_ERROR_IRQ_ISSUE          if NVIDIA kernel detected an interrupt issue with the attached GPUs
  -- *         - \ref NVML_ERROR_GPU_IS_LOST        if any GPU has fallen off the bus or is otherwise inaccessible
  -- *         - \ref NVML_ERROR_UNKNOWN            on any unexpected error
  -- *
  -- * @see nvmlDeviceGetUUID
  --  

   function nvmlDeviceGetHandleByUUID (uuid : Interfaces.C.Strings.chars_ptr; device : System.Address) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:1562
   pragma Import (C, nvmlDeviceGetHandleByUUID, "nvmlDeviceGetHandleByUUID");

  --*
  -- * Acquire the handle for a particular device, based on its PCI bus id.
  -- * 
  -- * For all products.
  -- *
  -- * This value corresponds to the nvmlPciInfo_t::busId returned by \ref nvmlDeviceGetPciInfo().
  -- *
  -- * Starting from NVML 5, this API causes NVML to initialize the target GPU
  -- * NVML may initialize additional GPUs if:
  -- *  - The target GPU is an SLI slave
  -- *
  -- * \note NVML 4.304 and older version of nvmlDeviceGetHandleByPciBusId"_v1" returns NVML_ERROR_NOT_FOUND 
  -- *       instead of NVML_ERROR_NO_PERMISSION.
  -- *
  -- * @param pciBusId                             The PCI bus id of the target GPU
  -- * @param device                               Reference in which to return the device handle
  -- * 
  -- * @return 
  -- *         - \ref NVML_SUCCESS                  if \a device has been set
  -- *         - \ref NVML_ERROR_UNINITIALIZED      if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT   if \a pciBusId is invalid or \a device is NULL
  -- *         - \ref NVML_ERROR_NOT_FOUND          if \a pciBusId does not match a valid device on the system
  -- *         - \ref NVML_ERROR_INSUFFICIENT_POWER if the attached device has improperly attached external power cables
  -- *         - \ref NVML_ERROR_NO_PERMISSION      if the user doesn't have permission to talk to this device
  -- *         - \ref NVML_ERROR_IRQ_ISSUE          if NVIDIA kernel detected an interrupt issue with the attached GPUs
  -- *         - \ref NVML_ERROR_GPU_IS_LOST        if the target GPU has fallen off the bus or is otherwise inaccessible
  -- *         - \ref NVML_ERROR_UNKNOWN            on any unexpected error
  --  

   function nvmlDeviceGetHandleByPciBusId_v2 (pciBusId : Interfaces.C.Strings.chars_ptr; device : System.Address) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:1592
   pragma Import (C, nvmlDeviceGetHandleByPciBusId_v2, "nvmlDeviceGetHandleByPciBusId_v2");

  --*
  -- * Retrieves the name of this device. 
  -- * 
  -- * For all products.
  -- *
  -- * The name is an alphanumeric string that denotes a particular product, e.g. Tesla &tm; C2070. It will not
  -- * exceed 64 characters in length (including the NULL terminator).  See \ref
  -- * nvmlConstants::NVML_DEVICE_NAME_BUFFER_SIZE.
  -- *
  -- * @param device                               The identifier of the target device
  -- * @param name                                 Reference in which to return the product name
  -- * @param length                               The maximum allowed length of the string returned in \a name
  -- * 
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if \a name has been set
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid, or \a name is NULL
  -- *         - \ref NVML_ERROR_INSUFFICIENT_SIZE if \a length is too small
  -- *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  --  

   function nvmlDeviceGetName
     (device : nvmlDevice_t;
      name : Interfaces.C.Strings.chars_ptr;
      length : unsigned) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:1615
   pragma Import (C, nvmlDeviceGetName, "nvmlDeviceGetName");

  --*
  -- * Retrieves the brand of this device.
  -- *
  -- * For all products.
  -- *
  -- * The type is a member of \ref nvmlBrandType_t defined above.
  -- *
  -- * @param device                               The identifier of the target device
  -- * @param type                                 Reference in which to return the product brand type
  -- *
  -- * @return
  -- *         - \ref NVML_SUCCESS                 if \a name has been set
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid, or \a type is NULL
  -- *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  --  

   function nvmlDeviceGetBrand (device : nvmlDevice_t; c_type : access nvmlBrandType_t) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:1634
   pragma Import (C, nvmlDeviceGetBrand, "nvmlDeviceGetBrand");

  --*
  -- * Retrieves the NVML index of this device.
  -- *
  -- * For all products.
  -- * 
  -- * Valid indices are derived from the \a accessibleDevices count returned by 
  -- *   \ref nvmlDeviceGetCount(). For example, if \a accessibleDevices is 2 the valid indices  
  -- *   are 0 and 1, corresponding to GPU 0 and GPU 1.
  -- *
  -- * The order in which NVML enumerates devices has no guarantees of consistency between reboots. For that reason it
  -- *   is recommended that devices be looked up by their PCI ids or GPU UUID. See 
  -- *   \ref nvmlDeviceGetHandleByPciBusId() and \ref nvmlDeviceGetHandleByUUID().
  -- *
  -- * Note: The NVML index may not correlate with other APIs, such as the CUDA device index.
  -- *
  -- * @param device                               The identifier of the target device
  -- * @param index                                Reference in which to return the NVML index of the device
  -- *
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if \a index has been set
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid, or \a index is NULL
  -- *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  -- *
  -- * @see nvmlDeviceGetHandleByIndex()
  -- * @see nvmlDeviceGetCount()
  --  

   function nvmlDeviceGetIndex (device : nvmlDevice_t; index : access unsigned) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:1664
   pragma Import (C, nvmlDeviceGetIndex, "nvmlDeviceGetIndex");

  --*
  -- * Retrieves the globally unique board serial number associated with this device's board.
  -- *
  -- * For all products with an inforom.
  -- *
  -- * The serial number is an alphanumeric string that will not exceed 30 characters (including the NULL terminator).
  -- * This number matches the serial number tag that is physically attached to the board.  See \ref
  -- * nvmlConstants::NVML_DEVICE_SERIAL_BUFFER_SIZE.
  -- *
  -- * @param device                               The identifier of the target device
  -- * @param serial                               Reference in which to return the board/module serial number
  -- * @param length                               The maximum allowed length of the string returned in \a serial
  -- * 
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if \a serial has been set
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid, or \a serial is NULL
  -- *         - \ref NVML_ERROR_INSUFFICIENT_SIZE if \a length is too small
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
  -- *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  --  

   function nvmlDeviceGetSerial
     (device : nvmlDevice_t;
      serial : Interfaces.C.Strings.chars_ptr;
      length : unsigned) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:1688
   pragma Import (C, nvmlDeviceGetSerial, "nvmlDeviceGetSerial");

  --*
  -- * Retrieves an array of unsigned ints (sized to cpuSetSize) of bitmasks with the ideal CPU affinity for the device
  -- * For example, if processors 0, 1, 32, and 33 are ideal for the device and cpuSetSize == 2,
  -- *     result[0] = 0x3, result[1] = 0x3
  -- *
  -- * For Kepler &tm; or newer fully supported devices.
  -- * Supported on Linux only.
  -- *
  -- * @param device                               The identifier of the target device
  -- * @param cpuSetSize                           The size of the cpuSet array that is safe to access
  -- * @param cpuSet                               Array reference in which to return a bitmask of CPUs, 64 CPUs per 
  -- *                                                 unsigned long on 64-bit machines, 32 on 32-bit machines
  -- *
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if \a cpuAffinity has been filled
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid, cpuSetSize == 0, or cpuSet is NULL
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
  -- *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  --  

   function nvmlDeviceGetCpuAffinity
     (device : nvmlDevice_t;
      cpuSetSize : unsigned;
      cpuSet : access unsigned_long) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:1711
   pragma Import (C, nvmlDeviceGetCpuAffinity, "nvmlDeviceGetCpuAffinity");

  --*
  -- * Sets the ideal affinity for the calling thread and device using the guidelines 
  -- * given in nvmlDeviceGetCpuAffinity().  Note, this is a change as of version 8.0.  
  -- * Older versions set the affinity for a calling process and all children.
  -- * Currently supports up to 64 processors.
  -- *
  -- * For Kepler &tm; or newer fully supported devices.
  -- * Supported on Linux only.
  -- *
  -- * @param device                               The identifier of the target device
  -- *
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if the calling process has been successfully bound
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
  -- *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  --  

   function nvmlDeviceSetCpuAffinity (device : nvmlDevice_t) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:1732
   pragma Import (C, nvmlDeviceSetCpuAffinity, "nvmlDeviceSetCpuAffinity");

  --*
  -- * Clear all affinity bindings for the calling thread.  Note, this is a change as of version
  -- * 8.0 as older versions cleared the affinity for a calling process and all children.
  -- *
  -- * For Kepler &tm; or newer fully supported devices.
  -- * Supported on Linux only.
  -- *
  -- * @param device                               The identifier of the target device
  -- *
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if the calling process has been successfully unbound
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  --  

   function nvmlDeviceClearCpuAffinity (device : nvmlDevice_t) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:1749
   pragma Import (C, nvmlDeviceClearCpuAffinity, "nvmlDeviceClearCpuAffinity");

  --*
  -- * Retrieve the common ancestor for two devices
  -- * For all products.
  -- * Supported on Linux only.
  -- *
  -- * @param device1                              The identifier of the first device
  -- * @param device2                              The identifier of the second device
  -- * @param pathInfo                             A \ref nvmlGpuTopologyLevel_t that gives the path type
  -- *
  -- * @return
  -- *         - \ref NVML_SUCCESS                 if \a pathInfo has been set
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device1, or \a device2 is invalid, or \a pathInfo is NULL
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device or OS does not support this feature
  -- *         - \ref NVML_ERROR_UNKNOWN           an error has occurred in underlying topology discovery
  --  

   function nvmlDeviceGetTopologyCommonAncestor
     (device1 : nvmlDevice_t;
      device2 : nvmlDevice_t;
      pathInfo : access nvmlGpuTopologyLevel_t) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:1766
   pragma Import (C, nvmlDeviceGetTopologyCommonAncestor, "nvmlDeviceGetTopologyCommonAncestor");

  --*
  -- * Retrieve the set of GPUs that are nearest to a given device at a specific interconnectivity level
  -- * For all products.
  -- * Supported on Linux only.
  -- *
  -- * @param device                               The identifier of the first device
  -- * @param level                                The \ref nvmlGpuTopologyLevel_t level to search for other GPUs
  -- * @param count                                When zero, is set to the number of matching GPUs such that \a deviceArray 
  -- *                                             can be malloc'd.  When non-zero, \a deviceArray will be filled with \a count
  -- *                                             number of device handles.
  -- * @param deviceArray                          An array of device handles for GPUs found at \a level
  -- *
  -- * @return
  -- *         - \ref NVML_SUCCESS                 if \a deviceArray or \a count (if initially zero) has been set
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device, \a level, or \a count is invalid, or \a deviceArray is NULL with a non-zero \a count
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device or OS does not support this feature
  -- *         - \ref NVML_ERROR_UNKNOWN           an error has occurred in underlying topology discovery
  --  

   function nvmlDeviceGetTopologyNearestGpus
     (device : nvmlDevice_t;
      level : nvmlGpuTopologyLevel_t;
      count : access unsigned;
      deviceArray : System.Address) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:1786
   pragma Import (C, nvmlDeviceGetTopologyNearestGpus, "nvmlDeviceGetTopologyNearestGpus");

  --*
  -- * Retrieve the set of GPUs that have a CPU affinity with the given CPU number
  -- * For all products.
  -- * Supported on Linux only.
  -- *
  -- * @param cpuNumber                            The CPU number
  -- * @param count                                When zero, is set to the number of matching GPUs such that \a deviceArray 
  -- *                                             can be malloc'd.  When non-zero, \a deviceArray will be filled with \a count
  -- *                                             number of device handles.
  -- * @param deviceArray                          An array of device handles for GPUs found with affinity to \a cpuNumber
  -- *
  -- * @return
  -- *         - \ref NVML_SUCCESS                 if \a deviceArray or \a count (if initially zero) has been set
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a cpuNumber, or \a count is invalid, or \a deviceArray is NULL with a non-zero \a count
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device or OS does not support this feature
  -- *         - \ref NVML_ERROR_UNKNOWN           an error has occurred in underlying topology discovery
  --  

   function nvmlSystemGetTopologyGpuSet
     (cpuNumber : unsigned;
      count : access unsigned;
      deviceArray : System.Address) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:1805
   pragma Import (C, nvmlSystemGetTopologyGpuSet, "nvmlSystemGetTopologyGpuSet");

  --*
  -- * Retrieve the status for a given p2p capability index between a given pair of GPU 
  -- * 
  -- * @param device1                              The first device 
  -- * @param device2                              The second device
  -- * @param p2pIndex                             p2p Capability Index being looked for between \a device1 and \a device2
  -- * @param p2pStatus                            Reference in which to return the status of the \a p2pIndex 
  -- *                                             between \a device1 and \a device2
  -- * @return 
  -- *         - \ref NVML_SUCCESS         if \a p2pStatus has been populated
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT     if \a device1 or \a device2 or \a p2pIndex is invalid or \a p2pStatus is NULL
  -- *         - \ref NVML_ERROR_UNKNOWN              on any unexpected error
  --  

   function nvmlDeviceGetP2PStatus
     (device1 : nvmlDevice_t;
      device2 : nvmlDevice_t;
      p2pIndex : nvmlGpuP2PCapsIndex_t;
      p2pStatus : access nvmlGpuP2PStatus_t) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:1820
   pragma Import (C, nvmlDeviceGetP2PStatus, "nvmlDeviceGetP2PStatus");

  --*
  -- * Retrieves the globally unique immutable UUID associated with this device, as a 5 part hexadecimal string,
  -- * that augments the immutable, board serial identifier.
  -- *
  -- * For all products.
  -- *
  -- * The UUID is a globally unique identifier. It is the only available identifier for pre-Fermi-architecture products.
  -- * It does NOT correspond to any identifier printed on the board.  It will not exceed 80 characters in length
  -- * (including the NULL terminator).  See \ref nvmlConstants::NVML_DEVICE_UUID_BUFFER_SIZE.
  -- *
  -- * @param device                               The identifier of the target device
  -- * @param uuid                                 Reference in which to return the GPU UUID
  -- * @param length                               The maximum allowed length of the string returned in \a uuid
  -- * 
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if \a uuid has been set
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid, or \a uuid is NULL
  -- *         - \ref NVML_ERROR_INSUFFICIENT_SIZE if \a length is too small 
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
  -- *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  --  

   function nvmlDeviceGetUUID
     (device : nvmlDevice_t;
      uuid : Interfaces.C.Strings.chars_ptr;
      length : unsigned) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:1846
   pragma Import (C, nvmlDeviceGetUUID, "nvmlDeviceGetUUID");

  --*
  -- * Retrieves minor number for the device. The minor number for the device is such that the Nvidia device node file for 
  -- * each GPU will have the form /dev/nvidia[minor number].
  -- *
  -- * For all products.
  -- * Supported only for Linux
  -- *
  -- * @param device                                The identifier of the target device
  -- * @param minorNumber                           Reference in which to return the minor number for the device
  -- * @return
  -- *         - \ref NVML_SUCCESS                 if the minor number is successfully retrieved
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a minorNumber is NULL
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if this query is not supported by the device
  -- *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  --  

   function nvmlDeviceGetMinorNumber (device : nvmlDevice_t; minorNumber : access unsigned) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:1865
   pragma Import (C, nvmlDeviceGetMinorNumber, "nvmlDeviceGetMinorNumber");

  --*
  -- * Retrieves the the device board part number which is programmed into the board's InfoROM
  -- *
  -- * For all products.
  -- *
  -- * @param device                                Identifier of the target device
  -- * @param partNumber                            Reference to the buffer to return
  -- * @param length                                Length of the buffer reference
  -- *
  -- * @return
  -- *         - \ref NVML_SUCCESS                  if \a partNumber has been set
  -- *         - \ref NVML_ERROR_UNINITIALIZED      if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED      if the needed VBIOS fields have not been filled
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT   if \a device is invalid or \a serial is NULL
  -- *         - \ref NVML_ERROR_GPU_IS_LOST        if the target GPU has fallen off the bus or is otherwise inaccessible
  -- *         - \ref NVML_ERROR_UNKNOWN            on any unexpected error
  --  

   function nvmlDeviceGetBoardPartNumber
     (device : nvmlDevice_t;
      partNumber : Interfaces.C.Strings.chars_ptr;
      length : unsigned) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:1884
   pragma Import (C, nvmlDeviceGetBoardPartNumber, "nvmlDeviceGetBoardPartNumber");

  --*
  -- * Retrieves the version information for the device's infoROM object.
  -- *
  -- * For all products with an inforom.
  -- *
  -- * Fermi and higher parts have non-volatile on-board memory for persisting device info, such as aggregate 
  -- * ECC counts. The version of the data structures in this memory may change from time to time. It will not
  -- * exceed 16 characters in length (including the NULL terminator).
  -- * See \ref nvmlConstants::NVML_DEVICE_INFOROM_VERSION_BUFFER_SIZE.
  -- *
  -- * See \ref nvmlInforomObject_t for details on the available infoROM objects.
  -- *
  -- * @param device                               The identifier of the target device
  -- * @param object                               The target infoROM object
  -- * @param version                              Reference in which to return the infoROM version
  -- * @param length                               The maximum allowed length of the string returned in \a version
  -- *
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if \a version has been set
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a version is NULL
  -- *         - \ref NVML_ERROR_INSUFFICIENT_SIZE if \a length is too small 
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not have an infoROM
  -- *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  -- *
  -- * @see nvmlDeviceGetInforomImageVersion
  --  

   function nvmlDeviceGetInforomVersion
     (device : nvmlDevice_t;
      object : nvmlInforomObject_t;
      version : Interfaces.C.Strings.chars_ptr;
      length : unsigned) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:1914
   pragma Import (C, nvmlDeviceGetInforomVersion, "nvmlDeviceGetInforomVersion");

  --*
  -- * Retrieves the global infoROM image version
  -- *
  -- * For all products with an inforom.
  -- *
  -- * Image version just like VBIOS version uniquely describes the exact version of the infoROM flashed on the board 
  -- * in contrast to infoROM object version which is only an indicator of supported features.
  -- * Version string will not exceed 16 characters in length (including the NULL terminator).
  -- * See \ref nvmlConstants::NVML_DEVICE_INFOROM_VERSION_BUFFER_SIZE.
  -- *
  -- * @param device                               The identifier of the target device
  -- * @param version                              Reference in which to return the infoROM image version
  -- * @param length                               The maximum allowed length of the string returned in \a version
  -- *
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if \a version has been set
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a version is NULL
  -- *         - \ref NVML_ERROR_INSUFFICIENT_SIZE if \a length is too small 
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not have an infoROM
  -- *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  -- *
  -- * @see nvmlDeviceGetInforomVersion
  --  

   function nvmlDeviceGetInforomImageVersion
     (device : nvmlDevice_t;
      version : Interfaces.C.Strings.chars_ptr;
      length : unsigned) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:1941
   pragma Import (C, nvmlDeviceGetInforomImageVersion, "nvmlDeviceGetInforomImageVersion");

  --*
  -- * Retrieves the checksum of the configuration stored in the device's infoROM.
  -- *
  -- * For all products with an inforom.
  -- *
  -- * Can be used to make sure that two GPUs have the exact same configuration.
  -- * Current checksum takes into account configuration stored in PWR and ECC infoROM objects.
  -- * Checksum can change between driver releases or when user changes configuration (e.g. disable/enable ECC)
  -- *
  -- * @param device                               The identifier of the target device
  -- * @param checksum                             Reference in which to return the infoROM configuration checksum
  -- *
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if \a checksum has been set
  -- *         - \ref NVML_ERROR_CORRUPTED_INFOROM if the device's checksum couldn't be retrieved due to infoROM corruption
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a checksum is NULL
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
  -- *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error 
  --  

   function nvmlDeviceGetInforomConfigurationChecksum (device : nvmlDevice_t; checksum : access unsigned) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:1964
   pragma Import (C, nvmlDeviceGetInforomConfigurationChecksum, "nvmlDeviceGetInforomConfigurationChecksum");

  --*
  -- * Reads the infoROM from the flash and verifies the checksums.
  -- *
  -- * For all products with an inforom.
  -- *
  -- * @param device                               The identifier of the target device
  -- *
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if infoROM is not corrupted
  -- *         - \ref NVML_ERROR_CORRUPTED_INFOROM if the device's infoROM is corrupted
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
  -- *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error 
  --  

   function nvmlDeviceValidateInforom (device : nvmlDevice_t) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:1981
   pragma Import (C, nvmlDeviceValidateInforom, "nvmlDeviceValidateInforom");

  --*
  -- * Retrieves the display mode for the device.
  -- *
  -- * For all products.
  -- *
  -- * This method indicates whether a physical display (e.g. monitor) is currently connected to
  -- * any of the device's connectors.
  -- *
  -- * See \ref nvmlEnableState_t for details on allowed modes.
  -- *
  -- * @param device                               The identifier of the target device
  -- * @param display                              Reference in which to return the display mode
  -- * 
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if \a display has been set
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a display is NULL
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
  -- *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  --  

   function nvmlDeviceGetDisplayMode (device : nvmlDevice_t; display : access nvmlEnableState_t) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:2004
   pragma Import (C, nvmlDeviceGetDisplayMode, "nvmlDeviceGetDisplayMode");

  --*
  -- * Retrieves the display active state for the device.
  -- *
  -- * For all products.
  -- *
  -- * This method indicates whether a display is initialized on the device.
  -- * For example whether X Server is attached to this device and has allocated memory for the screen.
  -- *
  -- * Display can be active even when no monitor is physically attached.
  -- *
  -- * See \ref nvmlEnableState_t for details on allowed modes.
  -- *
  -- * @param device                               The identifier of the target device
  -- * @param isActive                             Reference in which to return the display active state
  -- * 
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if \a isActive has been set
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a isActive is NULL
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
  -- *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  --  

   function nvmlDeviceGetDisplayActive (device : nvmlDevice_t; isActive : access nvmlEnableState_t) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:2029
   pragma Import (C, nvmlDeviceGetDisplayActive, "nvmlDeviceGetDisplayActive");

  --*
  -- * Retrieves the persistence mode associated with this device.
  -- *
  -- * For all products.
  -- * For Linux only.
  -- *
  -- * When driver persistence mode is enabled the driver software state is not torn down when the last 
  -- * client disconnects. By default this feature is disabled. 
  -- *
  -- * See \ref nvmlEnableState_t for details on allowed modes.
  -- *
  -- * @param device                               The identifier of the target device
  -- * @param mode                                 Reference in which to return the current driver persistence mode
  -- * 
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if \a mode has been set
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a mode is NULL
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
  -- *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  -- *
  -- * @see nvmlDeviceSetPersistenceMode()
  --  

   function nvmlDeviceGetPersistenceMode (device : nvmlDevice_t; mode : access nvmlEnableState_t) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:2055
   pragma Import (C, nvmlDeviceGetPersistenceMode, "nvmlDeviceGetPersistenceMode");

  --*
  -- * Retrieves the PCI attributes of this device.
  -- * 
  -- * For all products.
  -- *
  -- * See \ref nvmlPciInfo_t for details on the available PCI info.
  -- *
  -- * @param device                               The identifier of the target device
  -- * @param pci                                  Reference in which to return the PCI info
  -- * 
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if \a pci has been populated
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a pci is NULL
  -- *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  --  

   function nvmlDeviceGetPciInfo_v2 (device : nvmlDevice_t; pci : access nvmlPciInfo_t) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:2074
   pragma Import (C, nvmlDeviceGetPciInfo_v2, "nvmlDeviceGetPciInfo_v2");

  --*
  -- * Retrieves the maximum PCIe link generation possible with this device and system
  -- *
  -- * I.E. for a generation 2 PCIe device attached to a generation 1 PCIe bus the max link generation this function will
  -- * report is generation 1.
  -- * 
  -- * For Fermi &tm; or newer fully supported devices.
  -- * 
  -- * @param device                               The identifier of the target device
  -- * @param maxLinkGen                           Reference in which to return the max PCIe link generation
  -- * 
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if \a maxLinkGen has been populated
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a maxLinkGen is null
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if PCIe link information is not available
  -- *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  --  

   function nvmlDeviceGetMaxPcieLinkGeneration (device : nvmlDevice_t; maxLinkGen : access unsigned) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:2095
   pragma Import (C, nvmlDeviceGetMaxPcieLinkGeneration, "nvmlDeviceGetMaxPcieLinkGeneration");

  --*
  -- * Retrieves the maximum PCIe link width possible with this device and system
  -- *
  -- * I.E. for a device with a 16x PCIe bus width attached to a 8x PCIe system bus this function will report
  -- * a max link width of 8.
  -- * 
  -- * For Fermi &tm; or newer fully supported devices.
  -- * 
  -- * @param device                               The identifier of the target device
  -- * @param maxLinkWidth                         Reference in which to return the max PCIe link generation
  -- * 
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if \a maxLinkWidth has been populated
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a maxLinkWidth is null
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if PCIe link information is not available
  -- *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  --  

   function nvmlDeviceGetMaxPcieLinkWidth (device : nvmlDevice_t; maxLinkWidth : access unsigned) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:2116
   pragma Import (C, nvmlDeviceGetMaxPcieLinkWidth, "nvmlDeviceGetMaxPcieLinkWidth");

  --*
  -- * Retrieves the current PCIe link generation
  -- * 
  -- * For Fermi &tm; or newer fully supported devices.
  -- * 
  -- * @param device                               The identifier of the target device
  -- * @param currLinkGen                          Reference in which to return the current PCIe link generation
  -- * 
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if \a currLinkGen has been populated
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a currLinkGen is null
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if PCIe link information is not available
  -- *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  --  

   function nvmlDeviceGetCurrPcieLinkGeneration (device : nvmlDevice_t; currLinkGen : access unsigned) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:2134
   pragma Import (C, nvmlDeviceGetCurrPcieLinkGeneration, "nvmlDeviceGetCurrPcieLinkGeneration");

  --*
  -- * Retrieves the current PCIe link width
  -- * 
  -- * For Fermi &tm; or newer fully supported devices.
  -- * 
  -- * @param device                               The identifier of the target device
  -- * @param currLinkWidth                        Reference in which to return the current PCIe link generation
  -- * 
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if \a currLinkWidth has been populated
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a currLinkWidth is null
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if PCIe link information is not available
  -- *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  --  

   function nvmlDeviceGetCurrPcieLinkWidth (device : nvmlDevice_t; currLinkWidth : access unsigned) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:2152
   pragma Import (C, nvmlDeviceGetCurrPcieLinkWidth, "nvmlDeviceGetCurrPcieLinkWidth");

  --*
  -- * Retrieve PCIe utilization information.
  -- * This function is querying a byte counter over a 20ms interval and thus is the 
  -- *   PCIe throughput over that interval.
  -- *
  -- * For Maxwell &tm; or newer fully supported devices.
  -- *
  -- * This method is not supported on virtualized GPU environments.
  -- *
  -- * @param device                               The identifier of the target device
  -- * @param counter                              The specific counter that should be queried \ref nvmlPcieUtilCounter_t
  -- * @param value                                Reference in which to return throughput in KB/s
  -- *
  -- * @return
  -- *         - \ref NVML_SUCCESS                 if \a value has been set
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device or \a counter is invalid, or \a value is NULL
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
  -- *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  --  

   function nvmlDeviceGetPcieThroughput
     (device : nvmlDevice_t;
      counter : nvmlPcieUtilCounter_t;
      value : access unsigned) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:2175
   pragma Import (C, nvmlDeviceGetPcieThroughput, "nvmlDeviceGetPcieThroughput");

  --*  
  -- * Retrieve the PCIe replay counter.
  -- *
  -- * For Kepler &tm; or newer fully supported devices.
  -- *
  -- * @param device                               The identifier of the target device
  -- * @param value                                Reference in which to return the counter's value
  -- *
  -- * @return
  -- *         - \ref NVML_SUCCESS                 if \a value and \a rollover have been set
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid, or \a value or \a rollover are NULL
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
  -- *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  --  

   function nvmlDeviceGetPcieReplayCounter (device : nvmlDevice_t; value : access unsigned) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:2193
   pragma Import (C, nvmlDeviceGetPcieReplayCounter, "nvmlDeviceGetPcieReplayCounter");

  --*
  -- * Retrieves the current clock speeds for the device.
  -- *
  -- * For Fermi &tm; or newer fully supported devices.
  -- *
  -- * See \ref nvmlClockType_t for details on available clock information.
  -- *
  -- * @param device                               The identifier of the target device
  -- * @param type                                 Identify which clock domain to query
  -- * @param clock                                Reference in which to return the clock speed in MHz
  -- * 
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if \a clock has been set
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a clock is NULL
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device cannot report the specified clock
  -- *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  --  

   function nvmlDeviceGetClockInfo
     (device : nvmlDevice_t;
      c_type : nvmlClockType_t;
      clock : access unsigned) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:2214
   pragma Import (C, nvmlDeviceGetClockInfo, "nvmlDeviceGetClockInfo");

  --*
  -- * Retrieves the maximum clock speeds for the device.
  -- *
  -- * For Fermi &tm; or newer fully supported devices.
  -- *
  -- * See \ref nvmlClockType_t for details on available clock information.
  -- *
  -- * \note On GPUs from Fermi family current P0 clocks (reported by \ref nvmlDeviceGetClockInfo) can differ from max clocks
  -- *       by few MHz.
  -- *
  -- * @param device                               The identifier of the target device
  -- * @param type                                 Identify which clock domain to query
  -- * @param clock                                Reference in which to return the clock speed in MHz
  -- * 
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if \a clock has been set
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a clock is NULL
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device cannot report the specified clock
  -- *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  --  

   function nvmlDeviceGetMaxClockInfo
     (device : nvmlDevice_t;
      c_type : nvmlClockType_t;
      clock : access unsigned) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:2238
   pragma Import (C, nvmlDeviceGetMaxClockInfo, "nvmlDeviceGetMaxClockInfo");

  --*
  -- * Retrieves the current setting of a clock that applications will use unless an overspec situation occurs.
  -- * Can be changed using \ref nvmlDeviceSetApplicationsClocks.
  -- *
  -- * For Kepler &tm; or newer fully supported devices.
  -- *
  -- * @param device                               The identifier of the target device
  -- * @param clockType                            Identify which clock domain to query
  -- * @param clockMHz                             Reference in which to return the clock in MHz
  -- * 
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if \a clockMHz has been set
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a clockMHz is NULL or \a clockType is invalid
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
  -- *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  --  

   function nvmlDeviceGetApplicationsClock
     (device : nvmlDevice_t;
      clockType : nvmlClockType_t;
      clockMHz : access unsigned) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:2258
   pragma Import (C, nvmlDeviceGetApplicationsClock, "nvmlDeviceGetApplicationsClock");

  --*
  -- * Retrieves the default applications clock that GPU boots with or 
  -- * defaults to after \ref nvmlDeviceResetApplicationsClocks call.
  -- *
  -- * For Kepler &tm; or newer fully supported devices.
  -- *
  -- * @param device                               The identifier of the target device
  -- * @param clockType                            Identify which clock domain to query
  -- * @param clockMHz                             Reference in which to return the default clock in MHz
  -- * 
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if \a clockMHz has been set
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a clockMHz is NULL or \a clockType is invalid
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
  -- *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  -- *
  -- * \see nvmlDeviceGetApplicationsClock
  --  

   function nvmlDeviceGetDefaultApplicationsClock
     (device : nvmlDevice_t;
      clockType : nvmlClockType_t;
      clockMHz : access unsigned) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:2280
   pragma Import (C, nvmlDeviceGetDefaultApplicationsClock, "nvmlDeviceGetDefaultApplicationsClock");

  --*
  -- * Resets the application clock to the default value
  -- *
  -- * This is the applications clock that will be used after system reboot or driver reload.
  -- * Default value is constant, but the current value an be changed using \ref nvmlDeviceSetApplicationsClocks.
  -- *
  -- * On Pascal and newer hardware, if clocks were previously locked with \ref nvmlDeviceSetApplicationsClocks,
  -- * this call will unlock clocks. This returns clocks their default behavior ofautomatically boosting above
  -- * base clocks as thermal limits allow.
  -- *
  -- * @see nvmlDeviceGetApplicationsClock
  -- * @see nvmlDeviceSetApplicationsClocks
  -- *
  -- * For Fermi &tm; or newer non-GeForce fully supported devices and Maxwell or newer GeForce devices.
  -- *
  -- * @param device                               The identifier of the target device
  -- * 
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if new settings were successfully set
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
  -- *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  --  

   function nvmlDeviceResetApplicationsClocks (device : nvmlDevice_t) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:2307
   pragma Import (C, nvmlDeviceResetApplicationsClocks, "nvmlDeviceResetApplicationsClocks");

  --*
  -- * Retrieves the clock speed for the clock specified by the clock type and clock ID.
  -- *
  -- * For Kepler &tm; or newer fully supported devices.
  -- *
  -- * @param device                               The identifier of the target device
  -- * @param clockType                            Identify which clock domain to query
  -- * @param clockId                              Identify which clock in the domain to query
  -- * @param clockMHz                             Reference in which to return the clock in MHz
  -- *
  -- * @return
  -- *         - \ref NVML_SUCCESS                 if \a clockMHz has been set
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a clockMHz is NULL or \a clockType is invalid
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
  -- *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  --  

   function nvmlDeviceGetClock
     (device : nvmlDevice_t;
      clockType : nvmlClockType_t;
      clockId : nvmlClockId_t;
      clockMHz : access unsigned) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:2327
   pragma Import (C, nvmlDeviceGetClock, "nvmlDeviceGetClock");

  --*
  -- * Retrieves the customer defined maximum boost clock speed specified by the given clock type.
  -- *
  -- * For newer than Maxwell &tm; fully supported devices.
  -- *
  -- * @param device                               The identifier of the target device
  -- * @param clockType                            Identify which clock domain to query
  -- * @param clockMHz                             Reference in which to return the clock in MHz
  -- *
  -- * @return
  -- *         - \ref NVML_SUCCESS                 if \a clockMHz has been set
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a clockMHz is NULL or \a clockType is invalid
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device or the \a clockType on this device does not support this feature
  -- *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  --  

   function nvmlDeviceGetMaxCustomerBoostClock
     (device : nvmlDevice_t;
      clockType : nvmlClockType_t;
      clockMHz : access unsigned) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:2346
   pragma Import (C, nvmlDeviceGetMaxCustomerBoostClock, "nvmlDeviceGetMaxCustomerBoostClock");

  --*
  -- * Retrieves the list of possible memory clocks that can be used as an argument for \ref nvmlDeviceSetApplicationsClocks.
  -- *
  -- * For Kepler &tm; or newer fully supported devices.
  -- *
  -- * @param device                               The identifier of the target device
  -- * @param count                                Reference in which to provide the \a clocksMHz array size, and
  -- *                                             to return the number of elements
  -- * @param clocksMHz                            Reference in which to return the clock in MHz
  -- * 
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if \a count and \a clocksMHz have been populated 
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a count is NULL
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
  -- *         - \ref NVML_ERROR_INSUFFICIENT_SIZE if \a count is too small (\a count is set to the number of
  -- *                                                required elements)
  -- *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  -- *
  -- * @see nvmlDeviceSetApplicationsClocks
  -- * @see nvmlDeviceGetSupportedGraphicsClocks
  --  

   function nvmlDeviceGetSupportedMemoryClocks
     (device : nvmlDevice_t;
      count : access unsigned;
      clocksMHz : access unsigned) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:2371
   pragma Import (C, nvmlDeviceGetSupportedMemoryClocks, "nvmlDeviceGetSupportedMemoryClocks");

  --*
  -- * Retrieves the list of possible graphics clocks that can be used as an argument for \ref nvmlDeviceSetApplicationsClocks.
  -- *
  -- * For Kepler &tm; or newer fully supported devices.
  -- *
  -- * @param device                               The identifier of the target device
  -- * @param memoryClockMHz                       Memory clock for which to return possible graphics clocks
  -- * @param count                                Reference in which to provide the \a clocksMHz array size, and
  -- *                                             to return the number of elements
  -- * @param clocksMHz                            Reference in which to return the clocks in MHz
  -- * 
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if \a count and \a clocksMHz have been populated 
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_NOT_FOUND         if the specified \a memoryClockMHz is not a supported frequency
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a clock is NULL
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
  -- *         - \ref NVML_ERROR_INSUFFICIENT_SIZE if \a count is too small 
  -- *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  -- *
  -- * @see nvmlDeviceSetApplicationsClocks
  -- * @see nvmlDeviceGetSupportedMemoryClocks
  --  

   function nvmlDeviceGetSupportedGraphicsClocks
     (device : nvmlDevice_t;
      memoryClockMHz : unsigned;
      count : access unsigned;
      clocksMHz : access unsigned) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:2397
   pragma Import (C, nvmlDeviceGetSupportedGraphicsClocks, "nvmlDeviceGetSupportedGraphicsClocks");

  --*
  -- * Retrieve the current state of Auto Boosted clocks on a device and store it in \a isEnabled
  -- *
  -- * For Kepler &tm; or newer fully supported devices.
  -- *
  -- * Auto Boosted clocks are enabled by default on some hardware, allowing the GPU to run at higher clock rates
  -- * to maximize performance as thermal limits allow.
  -- *
  -- * On Pascal and newer hardware, Auto Aoosted clocks are controlled through application clocks.
  -- * Use \ref nvmlDeviceSetApplicationsClocks and \ref nvmlDeviceResetApplicationsClocks to control Auto Boost
  -- * behavior.
  -- *
  -- * @param device                               The identifier of the target device
  -- * @param isEnabled                            Where to store the current state of Auto Boosted clocks of the target device
  -- * @param defaultIsEnabled                     Where to store the default Auto Boosted clocks behavior of the target device that the device will
  -- *                                                 revert to when no applications are using the GPU
  -- *
  -- * @return
  -- *         - \ref NVML_SUCCESS                 If \a isEnabled has been been set with the Auto Boosted clocks state of \a device
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a isEnabled is NULL
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support Auto Boosted clocks
  -- *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  -- *
  --  

   function nvmlDeviceGetAutoBoostedClocksEnabled
     (device : nvmlDevice_t;
      isEnabled : access nvmlEnableState_t;
      defaultIsEnabled : access nvmlEnableState_t) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:2425
   pragma Import (C, nvmlDeviceGetAutoBoostedClocksEnabled, "nvmlDeviceGetAutoBoostedClocksEnabled");

  --*
  -- * Try to set the current state of Auto Boosted clocks on a device.
  -- *
  -- * For Kepler &tm; or newer fully supported devices.
  -- *
  -- * Auto Boosted clocks are enabled by default on some hardware, allowing the GPU to run at higher clock rates
  -- * to maximize performance as thermal limits allow. Auto Boosted clocks should be disabled if fixed clock
  -- * rates are desired.
  -- *
  -- * Non-root users may use this API by default but can be restricted by root from using this API by calling
  -- * \ref nvmlDeviceSetAPIRestriction with apiType=NVML_RESTRICTED_API_SET_AUTO_BOOSTED_CLOCKS.
  -- * Note: Persistence Mode is required to modify current Auto Boost settings, therefore, it must be enabled.
  -- *
  -- * On Pascal and newer hardware, Auto Boosted clocks are controlled through application clocks.
  -- * Use \ref nvmlDeviceSetApplicationsClocks and \ref nvmlDeviceResetApplicationsClocks to control Auto Boost
  -- * behavior.
  -- *
  -- * @param device                               The identifier of the target device
  -- * @param enabled                              What state to try to set Auto Boosted clocks of the target device to
  -- *
  -- * @return
  -- *         - \ref NVML_SUCCESS                 If the Auto Boosted clocks were successfully set to the state specified by \a enabled
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support Auto Boosted clocks
  -- *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  -- *
  --  

   function nvmlDeviceSetAutoBoostedClocksEnabled (device : nvmlDevice_t; enabled : nvmlEnableState_t) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:2456
   pragma Import (C, nvmlDeviceSetAutoBoostedClocksEnabled, "nvmlDeviceSetAutoBoostedClocksEnabled");

  --*
  -- * Try to set the default state of Auto Boosted clocks on a device. This is the default state that Auto Boosted clocks will
  -- * return to when no compute running processes (e.g. CUDA application which have an active context) are running
  -- *
  -- * For Kepler &tm; or newer non-GeForce fully supported devices and Maxwell or newer GeForce devices.
  -- * Requires root/admin permissions.
  -- *
  -- * Auto Boosted clocks are enabled by default on some hardware, allowing the GPU to run at higher clock rates
  -- * to maximize performance as thermal limits allow. Auto Boosted clocks should be disabled if fixed clock
  -- * rates are desired.
  -- *
  -- * On Pascal and newer hardware, Auto Boosted clocks are controlled through application clocks.
  -- * Use \ref nvmlDeviceSetApplicationsClocks and \ref nvmlDeviceResetApplicationsClocks to control Auto Boost
  -- * behavior.
  -- *
  -- * @param device                               The identifier of the target device
  -- * @param enabled                              What state to try to set default Auto Boosted clocks of the target device to
  -- * @param flags                                Flags that change the default behavior. Currently Unused.
  -- *
  -- * @return
  -- *         - \ref NVML_SUCCESS                 If the Auto Boosted clock's default state was successfully set to the state specified by \a enabled
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_NO_PERMISSION     If the calling user does not have permission to change Auto Boosted clock's default state.
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support Auto Boosted clocks
  -- *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  -- *
  --  

   function nvmlDeviceSetDefaultAutoBoostedClocksEnabled
     (device : nvmlDevice_t;
      enabled : nvmlEnableState_t;
      flags : unsigned) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:2487
   pragma Import (C, nvmlDeviceSetDefaultAutoBoostedClocksEnabled, "nvmlDeviceSetDefaultAutoBoostedClocksEnabled");

  --*
  -- * Retrieves the intended operating speed of the device's fan.
  -- *
  -- * Note: The reported speed is the intended fan speed.  If the fan is physically blocked and unable to spin, the
  -- * output will not match the actual fan speed.
  -- * 
  -- * For all discrete products with dedicated fans.
  -- *
  -- * The fan speed is expressed as a percent of the maximum, i.e. full speed is 100%.
  -- *
  -- * @param device                               The identifier of the target device
  -- * @param speed                                Reference in which to return the fan speed percentage
  -- * 
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if \a speed has been set
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a speed is NULL
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not have a fan
  -- *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  --  

   function nvmlDeviceGetFanSpeed (device : nvmlDevice_t; speed : access unsigned) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:2511
   pragma Import (C, nvmlDeviceGetFanSpeed, "nvmlDeviceGetFanSpeed");

  --*
  -- * Retrieves the current temperature readings for the device, in degrees C. 
  -- * 
  -- * For all products.
  -- *
  -- * See \ref nvmlTemperatureSensors_t for details on available temperature sensors.
  -- *
  -- * @param device                               The identifier of the target device
  -- * @param sensorType                           Flag that indicates which sensor reading to retrieve
  -- * @param temp                                 Reference in which to return the temperature reading
  -- * 
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if \a temp has been set
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid, \a sensorType is invalid or \a temp is NULL
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not have the specified sensor
  -- *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  --  

   function nvmlDeviceGetTemperature
     (device : nvmlDevice_t;
      sensorType : nvmlTemperatureSensors_t;
      temp : access unsigned) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:2532
   pragma Import (C, nvmlDeviceGetTemperature, "nvmlDeviceGetTemperature");

  --*
  -- * Retrieves the temperature threshold for the GPU with the specified threshold type in degrees C.
  -- *
  -- * For Kepler &tm; or newer fully supported devices.
  -- *
  -- * See \ref nvmlTemperatureThresholds_t for details on available temperature thresholds.
  -- *
  -- * @param device                               The identifier of the target device
  -- * @param thresholdType                        The type of threshold value queried
  -- * @param temp                                 Reference in which to return the temperature reading
  -- * @return
  -- *         - \ref NVML_SUCCESS                 if \a temp has been set
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid, \a thresholdType is invalid or \a temp is NULL
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not have a temperature sensor or is unsupported
  -- *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  --  

   function nvmlDeviceGetTemperatureThreshold
     (device : nvmlDevice_t;
      thresholdType : nvmlTemperatureThresholds_t;
      temp : access unsigned) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:2552
   pragma Import (C, nvmlDeviceGetTemperatureThreshold, "nvmlDeviceGetTemperatureThreshold");

  --*
  -- * Retrieves the current performance state for the device. 
  -- *
  -- * For Fermi &tm; or newer fully supported devices.
  -- *
  -- * See \ref nvmlPstates_t for details on allowed performance states.
  -- *
  -- * @param device                               The identifier of the target device
  -- * @param pState                               Reference in which to return the performance state reading
  -- * 
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if \a pState has been set
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a pState is NULL
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
  -- *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  --  

   function nvmlDeviceGetPerformanceState (device : nvmlDevice_t; pState : access nvmlPstates_t) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:2572
   pragma Import (C, nvmlDeviceGetPerformanceState, "nvmlDeviceGetPerformanceState");

  --*
  -- * Retrieves current clocks throttling reasons.
  -- *
  -- * For all fully supported products.
  -- *
  -- * \note More than one bit can be enabled at the same time. Multiple reasons can be affecting clocks at once.
  -- *
  -- * @param device                                The identifier of the target device
  -- * @param clocksThrottleReasons                 Reference in which to return bitmask of active clocks throttle
  -- *                                                  reasons
  -- *
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if \a clocksThrottleReasons has been set
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a clocksThrottleReasons is NULL
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
  -- *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  -- *
  -- * @see nvmlClocksThrottleReasons
  -- * @see nvmlDeviceGetSupportedClocksThrottleReasons
  --  

   function nvmlDeviceGetCurrentClocksThrottleReasons (device : nvmlDevice_t; clocksThrottleReasons : access Extensions.unsigned_long_long) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:2596
   pragma Import (C, nvmlDeviceGetCurrentClocksThrottleReasons, "nvmlDeviceGetCurrentClocksThrottleReasons");

  --*
  -- * Retrieves bitmask of supported clocks throttle reasons that can be returned by 
  -- * \ref nvmlDeviceGetCurrentClocksThrottleReasons
  -- *
  -- * For all fully supported products.
  -- *
  -- * This method is not supported on virtualized GPU environments.
  -- *
  -- * @param device                               The identifier of the target device
  -- * @param supportedClocksThrottleReasons       Reference in which to return bitmask of supported
  -- *                                              clocks throttle reasons
  -- *
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if \a supportedClocksThrottleReasons has been set
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a supportedClocksThrottleReasons is NULL
  -- *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  -- *
  -- * @see nvmlClocksThrottleReasons
  -- * @see nvmlDeviceGetCurrentClocksThrottleReasons
  --  

   function nvmlDeviceGetSupportedClocksThrottleReasons (device : nvmlDevice_t; supportedClocksThrottleReasons : access Extensions.unsigned_long_long) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:2620
   pragma Import (C, nvmlDeviceGetSupportedClocksThrottleReasons, "nvmlDeviceGetSupportedClocksThrottleReasons");

  --*
  -- * Deprecated: Use \ref nvmlDeviceGetPerformanceState. This function exposes an incorrect generalization.
  -- *
  -- * Retrieve the current performance state for the device. 
  -- *
  -- * For Fermi &tm; or newer fully supported devices.
  -- *
  -- * See \ref nvmlPstates_t for details on allowed performance states.
  -- *
  -- * @param device                               The identifier of the target device
  -- * @param pState                               Reference in which to return the performance state reading
  -- * 
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if \a pState has been set
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a pState is NULL
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
  -- *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  --  

   function nvmlDeviceGetPowerState (device : nvmlDevice_t; pState : access nvmlPstates_t) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:2642
   pragma Import (C, nvmlDeviceGetPowerState, "nvmlDeviceGetPowerState");

  --*
  -- * This API has been deprecated.
  -- *
  -- * Retrieves the power management mode associated with this device.
  -- *
  -- * For products from the Fermi family.
  -- *     - Requires \a NVML_INFOROM_POWER version 3.0 or higher.
  -- *
  -- * For from the Kepler or newer families.
  -- *     - Does not require \a NVML_INFOROM_POWER object.
  -- *
  -- * This flag indicates whether any power management algorithm is currently active on the device. An 
  -- * enabled state does not necessarily mean the device is being actively throttled -- only that 
  -- * that the driver will do so if the appropriate conditions are met.
  -- *
  -- * See \ref nvmlEnableState_t for details on allowed modes.
  -- *
  -- * @param device                               The identifier of the target device
  -- * @param mode                                 Reference in which to return the current power management mode
  -- * 
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if \a mode has been set
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a mode is NULL
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
  -- *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  --  

   function nvmlDeviceGetPowerManagementMode (device : nvmlDevice_t; mode : access nvmlEnableState_t) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:2672
   pragma Import (C, nvmlDeviceGetPowerManagementMode, "nvmlDeviceGetPowerManagementMode");

  --*
  -- * Retrieves the power management limit associated with this device.
  -- *
  -- * For Fermi &tm; or newer fully supported devices.
  -- *
  -- * The power limit defines the upper boundary for the card's power draw. If
  -- * the card's total power draw reaches this limit the power management algorithm kicks in.
  -- *
  -- * This reading is only available if power management mode is supported. 
  -- * See \ref nvmlDeviceGetPowerManagementMode.
  -- *
  -- * @param device                               The identifier of the target device
  -- * @param limit                                Reference in which to return the power management limit in milliwatts
  -- * 
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if \a limit has been set
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a limit is NULL
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
  -- *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  --  

   function nvmlDeviceGetPowerManagementLimit (device : nvmlDevice_t; limit : access unsigned) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:2696
   pragma Import (C, nvmlDeviceGetPowerManagementLimit, "nvmlDeviceGetPowerManagementLimit");

  --*
  -- * Retrieves information about possible values of power management limits on this device.
  -- *
  -- * For Kepler &tm; or newer fully supported devices.
  -- *
  -- * @param device                               The identifier of the target device
  -- * @param minLimit                             Reference in which to return the minimum power management limit in milliwatts
  -- * @param maxLimit                             Reference in which to return the maximum power management limit in milliwatts
  -- * 
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if \a minLimit and \a maxLimit have been set
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a minLimit or \a maxLimit is NULL
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
  -- *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  -- *
  -- * @see nvmlDeviceSetPowerManagementLimit
  --  

   function nvmlDeviceGetPowerManagementLimitConstraints
     (device : nvmlDevice_t;
      minLimit : access unsigned;
      maxLimit : access unsigned) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:2717
   pragma Import (C, nvmlDeviceGetPowerManagementLimitConstraints, "nvmlDeviceGetPowerManagementLimitConstraints");

  --*
  -- * Retrieves default power management limit on this device, in milliwatts.
  -- * Default power management limit is a power management limit that the device boots with.
  -- *
  -- * For Kepler &tm; or newer fully supported devices.
  -- *
  -- * @param device                               The identifier of the target device
  -- * @param defaultLimit                         Reference in which to return the default power management limit in milliwatts
  -- * 
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if \a defaultLimit has been set
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a defaultLimit is NULL
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
  -- *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  --  

   function nvmlDeviceGetPowerManagementDefaultLimit (device : nvmlDevice_t; defaultLimit : access unsigned) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:2736
   pragma Import (C, nvmlDeviceGetPowerManagementDefaultLimit, "nvmlDeviceGetPowerManagementDefaultLimit");

  --*
  -- * Retrieves power usage for this GPU in milliwatts and its associated circuitry (e.g. memory)
  -- *
  -- * For Fermi &tm; or newer fully supported devices.
  -- *
  -- * On Fermi and Kepler GPUs the reading is accurate to within +/- 5% of current power draw.
  -- *
  -- * It is only available if power management mode is supported. See \ref nvmlDeviceGetPowerManagementMode.
  -- *
  -- * @param device                               The identifier of the target device
  -- * @param power                                Reference in which to return the power usage information
  -- * 
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if \a power has been populated
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a power is NULL
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support power readings
  -- *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  --  

   function nvmlDeviceGetPowerUsage (device : nvmlDevice_t; power : access unsigned) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:2758
   pragma Import (C, nvmlDeviceGetPowerUsage, "nvmlDeviceGetPowerUsage");

  --*
  -- * Get the effective power limit that the driver enforces after taking into account all limiters
  -- *
  -- * Note: This can be different from the \ref nvmlDeviceGetPowerManagementLimit if other limits are set elsewhere
  -- * This includes the out of band power limit interface
  -- *
  -- * For Kepler &tm; or newer fully supported devices.
  -- *
  -- * @param device                           The device to communicate with
  -- * @param limit                            Reference in which to return the power management limit in milliwatts
  -- *
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if \a limit has been set
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a limit is NULL
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
  -- *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  --  

   function nvmlDeviceGetEnforcedPowerLimit (device : nvmlDevice_t; limit : access unsigned) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:2779
   pragma Import (C, nvmlDeviceGetEnforcedPowerLimit, "nvmlDeviceGetEnforcedPowerLimit");

  --*
  -- * Retrieves the current GOM and pending GOM (the one that GPU will switch to after reboot).
  -- *
  -- * For GK110 M-class and X-class Tesla &tm; products from the Kepler family.
  -- * Modes \ref NVML_GOM_LOW_DP and \ref NVML_GOM_ALL_ON are supported on fully supported GeForce products.
  -- * Not supported on Quadro &reg; and Tesla &tm; C-class products.
  -- *
  -- * @param device                               The identifier of the target device
  -- * @param current                              Reference in which to return the current GOM
  -- * @param pending                              Reference in which to return the pending GOM
  -- * 
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if \a mode has been populated
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a current or \a pending is NULL
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
  -- *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  -- *
  -- * @see nvmlGpuOperationMode_t
  -- * @see nvmlDeviceSetGpuOperationMode
  --  

   function nvmlDeviceGetGpuOperationMode
     (device : nvmlDevice_t;
      current : access nvmlGpuOperationMode_t;
      pending : access nvmlGpuOperationMode_t) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:2803
   pragma Import (C, nvmlDeviceGetGpuOperationMode, "nvmlDeviceGetGpuOperationMode");

  --*
  -- * Retrieves the amount of used, free and total memory available on the device, in bytes.
  -- * 
  -- * For all products.
  -- *
  -- * Enabling ECC reduces the amount of total available memory, due to the extra required parity bits.
  -- * Under WDDM most device memory is allocated and managed on startup by Windows.
  -- *
  -- * Under Linux and Windows TCC, the reported amount of used memory is equal to the sum of memory allocated 
  -- * by all active channels on the device.
  -- *
  -- * See \ref nvmlMemory_t for details on available memory info.
  -- *
  -- * @param device                               The identifier of the target device
  -- * @param memory                               Reference in which to return the memory information
  -- * 
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if \a memory has been populated
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a memory is NULL
  -- *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  --  

   function nvmlDeviceGetMemoryInfo (device : nvmlDevice_t; memory : access nvmlMemory_t) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:2828
   pragma Import (C, nvmlDeviceGetMemoryInfo, "nvmlDeviceGetMemoryInfo");

  --*
  -- * Retrieves the current compute mode for the device.
  -- *
  -- * For all products.
  -- *
  -- * See \ref nvmlComputeMode_t for details on allowed compute modes.
  -- *
  -- * @param device                               The identifier of the target device
  -- * @param mode                                 Reference in which to return the current compute mode
  -- * 
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if \a mode has been set
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a mode is NULL
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
  -- *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  -- *
  -- * @see nvmlDeviceSetComputeMode()
  --  

   function nvmlDeviceGetComputeMode (device : nvmlDevice_t; mode : access nvmlComputeMode_t) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:2850
   pragma Import (C, nvmlDeviceGetComputeMode, "nvmlDeviceGetComputeMode");

  --*
  -- * Retrieves the current and pending ECC modes for the device.
  -- *
  -- * For Fermi &tm; or newer fully supported devices.
  -- * Only applicable to devices with ECC.
  -- * Requires \a NVML_INFOROM_ECC version 1.0 or higher.
  -- *
  -- * Changing ECC modes requires a reboot. The "pending" ECC mode refers to the target mode following
  -- * the next reboot.
  -- *
  -- * See \ref nvmlEnableState_t for details on allowed modes.
  -- *
  -- * @param device                               The identifier of the target device
  -- * @param current                              Reference in which to return the current ECC mode
  -- * @param pending                              Reference in which to return the pending ECC mode
  -- * 
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if \a current and \a pending have been set
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or either \a current or \a pending is NULL
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
  -- *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  -- *
  -- * @see nvmlDeviceSetEccMode()
  --  

   function nvmlDeviceGetEccMode
     (device : nvmlDevice_t;
      current : access nvmlEnableState_t;
      pending : access nvmlEnableState_t) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:2878
   pragma Import (C, nvmlDeviceGetEccMode, "nvmlDeviceGetEccMode");

  --*
  -- * Retrieves the device boardId from 0-N.
  -- * Devices with the same boardId indicate GPUs connected to the same PLX.  Use in conjunction with 
  -- *  \ref nvmlDeviceGetMultiGpuBoard() to decide if they are on the same board as well.
  -- *  The boardId returned is a unique ID for the current configuration.  Uniqueness and ordering across 
  -- *  reboots and system configurations is not guaranteed (i.e. if a Tesla K40c returns 0x100 and
  -- *  the two GPUs on a Tesla K10 in the same system returns 0x200 it is not guaranteed they will 
  -- *  always return those values but they will always be different from each other).
  -- *  
  -- *
  -- * For Fermi &tm; or newer fully supported devices.
  -- *
  -- * @param device                               The identifier of the target device
  -- * @param boardId                              Reference in which to return the device's board ID
  -- *
  -- * @return
  -- *         - \ref NVML_SUCCESS                 if \a boardId has been set
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a boardId is NULL
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
  -- *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  --  

   function nvmlDeviceGetBoardId (device : nvmlDevice_t; boardId : access unsigned) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:2903
   pragma Import (C, nvmlDeviceGetBoardId, "nvmlDeviceGetBoardId");

  --*
  -- * Retrieves whether the device is on a Multi-GPU Board
  -- * Devices that are on multi-GPU boards will set \a multiGpuBool to a non-zero value.
  -- *
  -- * For Fermi &tm; or newer fully supported devices.
  -- *
  -- * @param device                               The identifier of the target device
  -- * @param multiGpuBool                         Reference in which to return a zero or non-zero value
  -- *                                                 to indicate whether the device is on a multi GPU board
  -- *
  -- * @return
  -- *         - \ref NVML_SUCCESS                 if \a multiGpuBool has been set
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a multiGpuBool is NULL
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
  -- *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  --  

   function nvmlDeviceGetMultiGpuBoard (device : nvmlDevice_t; multiGpuBool : access unsigned) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:2923
   pragma Import (C, nvmlDeviceGetMultiGpuBoard, "nvmlDeviceGetMultiGpuBoard");

  --*
  -- * Retrieves the total ECC error counts for the device.
  -- *
  -- * For Fermi &tm; or newer fully supported devices.
  -- * Only applicable to devices with ECC.
  -- * Requires \a NVML_INFOROM_ECC version 1.0 or higher.
  -- * Requires ECC Mode to be enabled.
  -- *
  -- * The total error count is the sum of errors across each of the separate memory systems, i.e. the total set of 
  -- * errors across the entire device.
  -- *
  -- * See \ref nvmlMemoryErrorType_t for a description of available error types.\n
  -- * See \ref nvmlEccCounterType_t for a description of available counter types.
  -- *
  -- * @param device                               The identifier of the target device
  -- * @param errorType                            Flag that specifies the type of the errors. 
  -- * @param counterType                          Flag that specifies the counter-type of the errors. 
  -- * @param eccCounts                            Reference in which to return the specified ECC errors
  -- * 
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if \a eccCounts has been set
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device, \a errorType or \a counterType is invalid, or \a eccCounts is NULL
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
  -- *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  -- *
  -- * @see nvmlDeviceClearEccErrorCounts()
  --  

   function nvmlDeviceGetTotalEccErrors
     (device : nvmlDevice_t;
      errorType : nvmlMemoryErrorType_t;
      counterType : nvmlEccCounterType_t;
      eccCounts : access Extensions.unsigned_long_long) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:2954
   pragma Import (C, nvmlDeviceGetTotalEccErrors, "nvmlDeviceGetTotalEccErrors");

  --*
  -- * Retrieves the detailed ECC error counts for the device.
  -- *
  -- * @deprecated   This API supports only a fixed set of ECC error locations
  -- *               On different GPU architectures different locations are supported
  -- *               See \ref nvmlDeviceGetMemoryErrorCounter
  -- *
  -- * For Fermi &tm; or newer fully supported devices.
  -- * Only applicable to devices with ECC.
  -- * Requires \a NVML_INFOROM_ECC version 2.0 or higher to report aggregate location-based ECC counts.
  -- * Requires \a NVML_INFOROM_ECC version 1.0 or higher to report all other ECC counts.
  -- * Requires ECC Mode to be enabled.
  -- *
  -- * Detailed errors provide separate ECC counts for specific parts of the memory system.
  -- *
  -- * Reports zero for unsupported ECC error counters when a subset of ECC error counters are supported.
  -- *
  -- * See \ref nvmlMemoryErrorType_t for a description of available bit types.\n
  -- * See \ref nvmlEccCounterType_t for a description of available counter types.\n
  -- * See \ref nvmlEccErrorCounts_t for a description of provided detailed ECC counts.
  -- *
  -- * @param device                               The identifier of the target device
  -- * @param errorType                            Flag that specifies the type of the errors. 
  -- * @param counterType                          Flag that specifies the counter-type of the errors. 
  -- * @param eccCounts                            Reference in which to return the specified ECC errors
  -- * 
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if \a eccCounts has been populated
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device, \a errorType or \a counterType is invalid, or \a eccCounts is NULL
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
  -- *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  -- *
  -- * @see nvmlDeviceClearEccErrorCounts()
  --  

   function nvmlDeviceGetDetailedEccErrors
     (device : nvmlDevice_t;
      errorType : nvmlMemoryErrorType_t;
      counterType : nvmlEccCounterType_t;
      eccCounts : access nvmlEccErrorCounts_t) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:2992
   pragma Import (C, nvmlDeviceGetDetailedEccErrors, "nvmlDeviceGetDetailedEccErrors");

  --*
  -- * Retrieves the requested memory error counter for the device.
  -- *
  -- * For Fermi &tm; or newer fully supported devices.
  -- * Requires \a NVML_INFOROM_ECC version 2.0 or higher to report aggregate location-based memory error counts.
  -- * Requires \a NVML_INFOROM_ECC version 1.0 or higher to report all other memory error counts.
  -- *
  -- * Only applicable to devices with ECC.
  -- *
  -- * Requires ECC Mode to be enabled.
  -- *
  -- * See \ref nvmlMemoryErrorType_t for a description of available memory error types.\n
  -- * See \ref nvmlEccCounterType_t for a description of available counter types.\n
  -- * See \ref nvmlMemoryLocation_t for a description of available counter locations.\n
  -- * 
  -- * @param device                               The identifier of the target device
  -- * @param errorType                            Flag that specifies the type of error.
  -- * @param counterType                          Flag that specifies the counter-type of the errors. 
  -- * @param locationType                         Specifies the location of the counter. 
  -- * @param count                                Reference in which to return the ECC counter
  -- * 
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if \a count has been populated
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device, \a bitTyp,e \a counterType or \a locationType is
  -- *                                             invalid, or \a count is NULL
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support ECC error reporting in the specified memory
  -- *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  --  

   function nvmlDeviceGetMemoryErrorCounter
     (device : nvmlDevice_t;
      errorType : nvmlMemoryErrorType_t;
      counterType : nvmlEccCounterType_t;
      locationType : nvmlMemoryLocation_t;
      count : access Extensions.unsigned_long_long) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:3024
   pragma Import (C, nvmlDeviceGetMemoryErrorCounter, "nvmlDeviceGetMemoryErrorCounter");

  --*
  -- * Retrieves the current utilization rates for the device's major subsystems.
  -- *
  -- * For Fermi &tm; or newer fully supported devices.
  -- *
  -- * See \ref nvmlUtilization_t for details on available utilization rates.
  -- *
  -- * \note During driver initialization when ECC is enabled one can see high GPU and Memory Utilization readings.
  -- *       This is caused by ECC Memory Scrubbing mechanism that is performed during driver initialization.
  -- *
  -- * @param device                               The identifier of the target device
  -- * @param utilization                          Reference in which to return the utilization information
  -- * 
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if \a utilization has been populated
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a utilization is NULL
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
  -- *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  --  

   function nvmlDeviceGetUtilizationRates (device : nvmlDevice_t; utilization : access nvmlUtilization_t) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:3049
   pragma Import (C, nvmlDeviceGetUtilizationRates, "nvmlDeviceGetUtilizationRates");

  --*
  -- * Retrieves the current utilization and sampling size in microseconds for the Encoder
  -- *
  -- * For Kepler &tm; or newer fully supported devices.
  -- *
  -- * @param device                               The identifier of the target device
  -- * @param utilization                          Reference to an unsigned int for encoder utilization info
  -- * @param samplingPeriodUs                     Reference to an unsigned int for the sampling period in US
  -- *
  -- * @return
  -- *         - \ref NVML_SUCCESS                 if \a utilization has been populated
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid, \a utilization is NULL, or \a samplingPeriodUs is NULL
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
  -- *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  --  

   function nvmlDeviceGetEncoderUtilization
     (device : nvmlDevice_t;
      utilization : access unsigned;
      samplingPeriodUs : access unsigned) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:3068
   pragma Import (C, nvmlDeviceGetEncoderUtilization, "nvmlDeviceGetEncoderUtilization");

  --*
  -- * Retrieves the current utilization and sampling size in microseconds for the Decoder
  -- *
  -- * For Kepler &tm; or newer fully supported devices.
  -- *
  -- * @param device                               The identifier of the target device
  -- * @param utilization                          Reference to an unsigned int for decoder utilization info
  -- * @param samplingPeriodUs                     Reference to an unsigned int for the sampling period in US
  -- *
  -- * @return
  -- *         - \ref NVML_SUCCESS                 if \a utilization has been populated
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid, \a utilization is NULL, or \a samplingPeriodUs is NULL
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
  -- *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  --  

   function nvmlDeviceGetDecoderUtilization
     (device : nvmlDevice_t;
      utilization : access unsigned;
      samplingPeriodUs : access unsigned) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:3087
   pragma Import (C, nvmlDeviceGetDecoderUtilization, "nvmlDeviceGetDecoderUtilization");

  --*
  -- * Retrieves the current and pending driver model for the device.
  -- *
  -- * For Fermi &tm; or newer fully supported devices.
  -- * For windows only.
  -- *
  -- * On Windows platforms the device driver can run in either WDDM or WDM (TCC) mode. If a display is attached
  -- * to the device it must run in WDDM mode. TCC mode is preferred if a display is not attached.
  -- *
  -- * See \ref nvmlDriverModel_t for details on available driver models.
  -- *
  -- * @param device                               The identifier of the target device
  -- * @param current                              Reference in which to return the current driver model
  -- * @param pending                              Reference in which to return the pending driver model
  -- * 
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if either \a current and/or \a pending have been set
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or both \a current and \a pending are NULL
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if the platform is not windows
  -- *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  -- * 
  -- * @see nvmlDeviceSetDriverModel()
  --  

   function nvmlDeviceGetDriverModel
     (device : nvmlDevice_t;
      current : access nvmlDriverModel_t;
      pending : access nvmlDriverModel_t) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:3114
   pragma Import (C, nvmlDeviceGetDriverModel, "nvmlDeviceGetDriverModel");

  --*
  -- * Get VBIOS version of the device.
  -- *
  -- * For all products.
  -- *
  -- * The VBIOS version may change from time to time. It will not exceed 32 characters in length 
  -- * (including the NULL terminator).  See \ref nvmlConstants::NVML_DEVICE_VBIOS_VERSION_BUFFER_SIZE.
  -- *
  -- * @param device                               The identifier of the target device
  -- * @param version                              Reference to which to return the VBIOS version
  -- * @param length                               The maximum allowed length of the string returned in \a version
  -- * 
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if \a version has been set
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid, or \a version is NULL
  -- *         - \ref NVML_ERROR_INSUFFICIENT_SIZE if \a length is too small 
  -- *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  --  

   function nvmlDeviceGetVbiosVersion
     (device : nvmlDevice_t;
      version : Interfaces.C.Strings.chars_ptr;
      length : unsigned) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:3136
   pragma Import (C, nvmlDeviceGetVbiosVersion, "nvmlDeviceGetVbiosVersion");

  --*
  -- * Get Bridge Chip Information for all the bridge chips on the board.
  -- * 
  -- * For all fully supported products.
  -- * Only applicable to multi-GPU products.
  -- * 
  -- * @param device                                The identifier of the target device
  -- * @param bridgeHierarchy                       Reference to the returned bridge chip Hierarchy
  -- * 
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if bridge chip exists
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid, or \a bridgeInfo is NULL
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if bridge chip not supported on the device
  -- *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  -- * 
  --  

   function nvmlDeviceGetBridgeChipInfo (device : nvmlDevice_t; bridgeHierarchy : access nvmlBridgeChipHierarchy_t) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:3156
   pragma Import (C, nvmlDeviceGetBridgeChipInfo, "nvmlDeviceGetBridgeChipInfo");

  --*
  -- * Get information about processes with a compute context on a device
  -- *
  -- * For Fermi &tm; or newer fully supported devices.
  -- *
  -- * This function returns information only about compute running processes (e.g. CUDA application which have
  -- * active context). Any graphics applications (e.g. using OpenGL, DirectX) won't be listed by this function.
  -- *
  -- * To query the current number of running compute processes, call this function with *infoCount = 0. The
  -- * return code will be NVML_ERROR_INSUFFICIENT_SIZE, or NVML_SUCCESS if none are running. For this call
  -- * \a infos is allowed to be NULL.
  -- *
  -- * The usedGpuMemory field returned is all of the memory used by the application.
  -- *
  -- * Keep in mind that information returned by this call is dynamic and the number of elements might change in
  -- * time. Allocate more space for \a infos table in case new compute processes are spawned.
  -- *
  -- * @param device                               The identifier of the target device
  -- * @param infoCount                            Reference in which to provide the \a infos array size, and
  -- *                                             to return the number of returned elements
  -- * @param infos                                Reference in which to return the process information
  -- * 
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if \a infoCount and \a infos have been populated
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INSUFFICIENT_SIZE if \a infoCount indicates that the \a infos array is too small
  -- *                                             \a infoCount will contain minimal amount of space necessary for
  -- *                                             the call to complete
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid, either of \a infoCount or \a infos is NULL
  -- *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  -- *
  -- * @see \ref nvmlSystemGetProcessName
  --  

   function nvmlDeviceGetComputeRunningProcesses
     (device : nvmlDevice_t;
      infoCount : access unsigned;
      infos : access nvmlProcessInfo_t) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:3192
   pragma Import (C, nvmlDeviceGetComputeRunningProcesses, "nvmlDeviceGetComputeRunningProcesses");

  --*
  -- * Get information about processes with a graphics context on a device
  -- *
  -- * For Kepler &tm; or newer fully supported devices.
  -- *
  -- * This function returns information only about graphics based processes 
  -- * (eg. applications using OpenGL, DirectX)
  -- *
  -- * To query the current number of running graphics processes, call this function with *infoCount = 0. The
  -- * return code will be NVML_ERROR_INSUFFICIENT_SIZE, or NVML_SUCCESS if none are running. For this call
  -- * \a infos is allowed to be NULL.
  -- *
  -- * The usedGpuMemory field returned is all of the memory used by the application.
  -- *
  -- * Keep in mind that information returned by this call is dynamic and the number of elements might change in
  -- * time. Allocate more space for \a infos table in case new graphics processes are spawned.
  -- *
  -- * @param device                               The identifier of the target device
  -- * @param infoCount                            Reference in which to provide the \a infos array size, and
  -- *                                             to return the number of returned elements
  -- * @param infos                                Reference in which to return the process information
  -- * 
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if \a infoCount and \a infos have been populated
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INSUFFICIENT_SIZE if \a infoCount indicates that the \a infos array is too small
  -- *                                             \a infoCount will contain minimal amount of space necessary for
  -- *                                             the call to complete
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid, either of \a infoCount or \a infos is NULL
  -- *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  -- *
  -- * @see \ref nvmlSystemGetProcessName
  --  

   function nvmlDeviceGetGraphicsRunningProcesses
     (device : nvmlDevice_t;
      infoCount : access unsigned;
      infos : access nvmlProcessInfo_t) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:3228
   pragma Import (C, nvmlDeviceGetGraphicsRunningProcesses, "nvmlDeviceGetGraphicsRunningProcesses");

  --*
  -- * Check if the GPU devices are on the same physical board.
  -- *
  -- * For all fully supported products.
  -- *
  -- * @param device1                               The first GPU device
  -- * @param device2                               The second GPU device
  -- * @param onSameBoard                           Reference in which to return the status.
  -- *                                              Non-zero indicates that the GPUs are on the same board.
  -- *
  -- * @return
  -- *         - \ref NVML_SUCCESS                 if \a onSameBoard has been set
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a dev1 or \a dev2 are invalid or \a onSameBoard is NULL
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if this check is not supported by the device
  -- *         - \ref NVML_ERROR_GPU_IS_LOST       if the either GPU has fallen off the bus or is otherwise inaccessible
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  --  

   function nvmlDeviceOnSameBoard
     (device1 : nvmlDevice_t;
      device2 : nvmlDevice_t;
      onSameBoard : access int) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:3248
   pragma Import (C, nvmlDeviceOnSameBoard, "nvmlDeviceOnSameBoard");

  --*
  -- * Retrieves the root/admin permissions on the target API. See \a nvmlRestrictedAPI_t for the list of supported APIs.
  -- * If an API is restricted only root users can call that API. See \a nvmlDeviceSetAPIRestriction to change current permissions.
  -- *
  -- * For all fully supported products.
  -- *
  -- * @param device                               The identifier of the target device
  -- * @param apiType                              Target API type for this operation
  -- * @param isRestricted                         Reference in which to return the current restriction 
  -- *                                             NVML_FEATURE_ENABLED indicates that the API is root-only
  -- *                                             NVML_FEATURE_DISABLED indicates that the API is accessible to all users
  -- *
  -- * @return
  -- *         - \ref NVML_SUCCESS                 if \a isRestricted has been set
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid, \a apiType incorrect or \a isRestricted is NULL
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if this query is not supported by the device or the device does not support
  -- *                                                 the feature that is being queried (E.G. Enabling/disabling Auto Boosted clocks is
  -- *                                                 not supported by the device)
  -- *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  -- *
  -- * @see nvmlRestrictedAPI_t
  --  

   function nvmlDeviceGetAPIRestriction
     (device : nvmlDevice_t;
      apiType : nvmlRestrictedAPI_t;
      isRestricted : access nvmlEnableState_t) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:3274
   pragma Import (C, nvmlDeviceGetAPIRestriction, "nvmlDeviceGetAPIRestriction");

  --*
  -- * Gets recent samples for the GPU.
  -- * 
  -- * For Kepler &tm; or newer fully supported devices.
  -- * 
  -- * Based on type, this method can be used to fetch the power, utilization or clock samples maintained in the buffer by 
  -- * the driver.
  -- * 
  -- * Power, Utilization and Clock samples are returned as type "unsigned int" for the union nvmlValue_t.
  -- * 
  -- * To get the size of samples that user needs to allocate, the method is invoked with samples set to NULL. 
  -- * The returned samplesCount will provide the number of samples that can be queried. The user needs to 
  -- * allocate the buffer with size as samplesCount * sizeof(nvmlSample_t).
  -- * 
  -- * lastSeenTimeStamp represents CPU timestamp in microseconds. Set it to 0 to fetch all the samples maintained by the 
  -- * underlying buffer. Set lastSeenTimeStamp to one of the timeStamps retrieved from the date of the previous query 
  -- * to get more recent samples.
  -- * 
  -- * This method fetches the number of entries which can be accommodated in the provided samples array, and the 
  -- * reference samplesCount is updated to indicate how many samples were actually retrieved. The advantage of using this 
  -- * method for samples in contrast to polling via existing methods is to get get higher frequency data at lower polling cost.
  -- * 
  -- * @param device                        The identifier for the target device
  -- * @param type                          Type of sampling event
  -- * @param lastSeenTimeStamp             Return only samples with timestamp greater than lastSeenTimeStamp. 
  -- * @param sampleValType                 Output parameter to represent the type of sample value as described in nvmlSampleVal_t
  -- * @param sampleCount                   Reference to provide the number of elements which can be queried in samples array
  -- * @param samples                       Reference in which samples are returned
  -- 
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if samples are successfully retrieved
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid, \a samplesCount is NULL or 
  -- *                                             reference to \a sampleCount is 0 for non null \a samples
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if this query is not supported by the device
  -- *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
  -- *         - \ref NVML_ERROR_NOT_FOUND         if sample entries are not found
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  --  

   function nvmlDeviceGetSamples
     (device : nvmlDevice_t;
      c_type : nvmlSamplingType_t;
      lastSeenTimeStamp : Extensions.unsigned_long_long;
      sampleValType : access nvmlValueType_t;
      sampleCount : access unsigned;
      samples : access nvmlSample_t) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:3315
   pragma Import (C, nvmlDeviceGetSamples, "nvmlDeviceGetSamples");

  --*
  -- * Gets Total, Available and Used size of BAR1 memory.
  -- * 
  -- * BAR1 is used to map the FB (device memory) so that it can be directly accessed by the CPU or by 3rd party 
  -- * devices (peer-to-peer on the PCIE bus). 
  -- * 
  -- * For Kepler &tm; or newer fully supported devices.
  -- *
  -- * @param device                               The identifier of the target device
  -- * @param bar1Memory                           Reference in which BAR1 memory
  -- *                                             information is returned.
  -- *
  -- * @return
  -- *         - \ref NVML_SUCCESS                 if BAR1 memory is successfully retrieved
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid, \a bar1Memory is NULL
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if this query is not supported by the device
  -- *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  -- *
  --  

   function nvmlDeviceGetBAR1MemoryInfo (device : nvmlDevice_t; bar1Memory : access nvmlBAR1Memory_t) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:3339
   pragma Import (C, nvmlDeviceGetBAR1MemoryInfo, "nvmlDeviceGetBAR1MemoryInfo");

  --*
  -- * Gets the duration of time during which the device was throttled (lower than requested clocks) due to power 
  -- * or thermal constraints.
  -- *
  -- * The method is important to users who are tying to understand if their GPUs throttle at any point during their applications. The
  -- * difference in violation times at two different reference times gives the indication of GPU throttling event. 
  -- *
  -- * Violation for thermal capping is not supported at this time.
  -- * 
  -- * For Kepler &tm; or newer fully supported devices.
  -- *
  -- * @param device                               The identifier of the target device
  -- * @param perfPolicyType                       Represents Performance policy which can trigger GPU throttling
  -- * @param violTime                             Reference to which violation time related information is returned 
  -- *                                         
  -- *
  -- * @return
  -- *         - \ref NVML_SUCCESS                 if violation time is successfully retrieved
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid, \a perfPolicyType is invalid, or \a violTime is NULL
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if this query is not supported by the device
  -- *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
  -- *
  --  

   function nvmlDeviceGetViolationStatus
     (device : nvmlDevice_t;
      perfPolicyType : nvmlPerfPolicyType_t;
      violTime : access nvmlViolationTime_t) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:3366
   pragma Import (C, nvmlDeviceGetViolationStatus, "nvmlDeviceGetViolationStatus");

  --*
  -- * @}
  --  

  --* @addtogroup nvmlAccountingStats
  -- *  @{
  --  

  --*
  -- * Queries the state of per process accounting mode.
  -- *
  -- * For Kepler &tm; or newer fully supported devices.
  -- *
  -- * See \ref nvmlDeviceGetAccountingStats for more details.
  -- * See \ref nvmlDeviceSetAccountingMode
  -- *
  -- * @param device                               The identifier of the target device
  -- * @param mode                                 Reference in which to return the current accounting mode
  -- *
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if the mode has been successfully retrieved 
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a mode are NULL
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  --  

   function nvmlDeviceGetAccountingMode (device : nvmlDevice_t; mode : access nvmlEnableState_t) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:3394
   pragma Import (C, nvmlDeviceGetAccountingMode, "nvmlDeviceGetAccountingMode");

  --*
  -- * Queries process's accounting stats.
  -- *
  -- * For Kepler &tm; or newer fully supported devices.
  -- * 
  -- * Accounting stats capture GPU utilization and other statistics across the lifetime of a process.
  -- * Accounting stats can be queried during life time of the process and after its termination.
  -- * The time field in \ref nvmlAccountingStats_t is reported as 0 during the lifetime of the process and 
  -- * updated to actual running time after its termination.
  -- * Accounting stats are kept in a circular buffer, newly created processes overwrite information about old
  -- * processes.
  -- *
  -- * See \ref nvmlAccountingStats_t for description of each returned metric.
  -- * List of processes that can be queried can be retrieved from \ref nvmlDeviceGetAccountingPids.
  -- *
  -- * @note Accounting Mode needs to be on. See \ref nvmlDeviceGetAccountingMode.
  -- * @note Only compute and graphics applications stats can be queried. Monitoring applications stats can't be
  -- *         queried since they don't contribute to GPU utilization.
  -- * @note In case of pid collision stats of only the latest process (that terminated last) will be reported
  -- *
  -- * @warning On Kepler devices per process statistics are accurate only if there's one process running on a GPU.
  -- * 
  -- * @param device                               The identifier of the target device
  -- * @param pid                                  Process Id of the target process to query stats for
  -- * @param stats                                Reference in which to return the process's accounting stats
  -- *
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if stats have been successfully retrieved
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a stats are NULL
  -- *         - \ref NVML_ERROR_NOT_FOUND         if process stats were not found
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature or accounting mode is disabled
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  -- *
  -- * @see nvmlDeviceGetAccountingBufferSize
  --  

   function nvmlDeviceGetAccountingStats
     (device : nvmlDevice_t;
      pid : unsigned;
      stats : access nvmlAccountingStats_t) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:3432
   pragma Import (C, nvmlDeviceGetAccountingStats, "nvmlDeviceGetAccountingStats");

  --*
  -- * Queries list of processes that can be queried for accounting stats. The list of processes returned 
  -- * can be in running or terminated state.
  -- *
  -- * For Kepler &tm; or newer fully supported devices.
  -- *
  -- * To just query the number of processes ready to be queried, call this function with *count = 0 and
  -- * pids=NULL. The return code will be NVML_ERROR_INSUFFICIENT_SIZE, or NVML_SUCCESS if list is empty.
  -- * 
  -- * For more details see \ref nvmlDeviceGetAccountingStats.
  -- *
  -- * @note In case of PID collision some processes might not be accessible before the circular buffer is full.
  -- *
  -- * @param device                               The identifier of the target device
  -- * @param count                                Reference in which to provide the \a pids array size, and
  -- *                                               to return the number of elements ready to be queried
  -- * @param pids                                 Reference in which to return list of process ids
  -- * 
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if pids were successfully retrieved
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a count is NULL
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature or accounting mode is disabled
  -- *         - \ref NVML_ERROR_INSUFFICIENT_SIZE if \a count is too small (\a count is set to
  -- *                                                 expected value)
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  -- *
  -- * @see nvmlDeviceGetAccountingBufferSize
  --  

   function nvmlDeviceGetAccountingPids
     (device : nvmlDevice_t;
      count : access unsigned;
      pids : access unsigned) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:3463
   pragma Import (C, nvmlDeviceGetAccountingPids, "nvmlDeviceGetAccountingPids");

  --*
  -- * Returns the number of processes that the circular buffer with accounting pids can hold.
  -- *
  -- * For Kepler &tm; or newer fully supported devices.
  -- *
  -- * This is the maximum number of processes that accounting information will be stored for before information
  -- * about oldest processes will get overwritten by information about new processes.
  -- *
  -- * @param device                               The identifier of the target device
  -- * @param bufferSize                           Reference in which to provide the size (in number of elements)
  -- *                                               of the circular buffer for accounting stats.
  -- * 
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if buffer size was successfully retrieved
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a bufferSize is NULL
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature or accounting mode is disabled
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  -- * 
  -- * @see nvmlDeviceGetAccountingStats
  -- * @see nvmlDeviceGetAccountingPids
  --  

   function nvmlDeviceGetAccountingBufferSize (device : nvmlDevice_t; bufferSize : access unsigned) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:3487
   pragma Import (C, nvmlDeviceGetAccountingBufferSize, "nvmlDeviceGetAccountingBufferSize");

  --* @}  
  --* @addtogroup nvmlDeviceQueries
  -- *  @{
  --  

  --*
  -- * Returns the list of retired pages by source, including pages that are pending retirement
  -- * The address information provided from this API is the hardware address of the page that was retired.  Note
  -- * that this does not match the virtual address used in CUDA, but will match the address information in XID 63
  -- * 
  -- * For Kepler &tm; or newer fully supported devices.
  -- *
  -- * @param device                            The identifier of the target device
  -- * @param cause                             Filter page addresses by cause of retirement
  -- * @param pageCount                         Reference in which to provide the \a addresses buffer size, and
  -- *                                          to return the number of retired pages that match \a cause
  -- *                                          Set to 0 to query the size without allocating an \a addresses buffer
  -- * @param addresses                         Buffer to write the page addresses into
  -- * 
  -- * @return
  -- *         - \ref NVML_SUCCESS                 if \a pageCount was populated and \a addresses was filled
  -- *         - \ref NVML_ERROR_INSUFFICIENT_SIZE if \a pageCount indicates the buffer is not large enough to store all the
  -- *                                             matching page addresses.  \a pageCount is set to the needed size.
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid, \a pageCount is NULL, \a cause is invalid, or 
  -- *                                             \a addresses is NULL
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
  -- *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  --  

   function nvmlDeviceGetRetiredPages
     (device : nvmlDevice_t;
      cause : nvmlPageRetirementCause_t;
      pageCount : access unsigned;
      addresses : access Extensions.unsigned_long_long) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:3520
   pragma Import (C, nvmlDeviceGetRetiredPages, "nvmlDeviceGetRetiredPages");

  --*
  -- * Check if any pages are pending retirement and need a reboot to fully retire.
  -- *
  -- * For Kepler &tm; or newer fully supported devices.
  -- *
  -- * @param device                            The identifier of the target device
  -- * @param isPending                         Reference in which to return the pending status
  -- * 
  -- * @return
  -- *         - \ref NVML_SUCCESS                 if \a isPending was populated
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a isPending is NULL
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
  -- *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  --  

   function nvmlDeviceGetRetiredPagesPendingStatus (device : nvmlDevice_t; isPending : access nvmlEnableState_t) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:3539
   pragma Import (C, nvmlDeviceGetRetiredPagesPendingStatus, "nvmlDeviceGetRetiredPagesPendingStatus");

  --* @}  
  --************************************************************************************************* 
  --* @defgroup nvmlUnitCommands Unit Commands
  -- *  This chapter describes NVML operations that change the state of the unit. For S-class products.
  -- *  Each of these requires root/admin access. Non-admin users will see an NVML_ERROR_NO_PERMISSION
  -- *  error code when invoking any of these methods.
  -- *  @{
  --  

  --************************************************************************************************* 
  --*
  -- * Set the LED state for the unit. The LED can be either green (0) or amber (1).
  -- *
  -- * For S-class products.
  -- * Requires root/admin permissions.
  -- *
  -- * This operation takes effect immediately.
  -- * 
  -- *
  -- * <b>Current S-Class products don't provide unique LEDs for each unit. As such, both front 
  -- * and back LEDs will be toggled in unison regardless of which unit is specified with this command.</b>
  -- *
  -- * See \ref nvmlLedColor_t for available colors.
  -- *
  -- * @param unit                                 The identifier of the target unit
  -- * @param color                                The target LED color
  -- * 
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if the LED color has been set
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a unit or \a color is invalid
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if this is not an S-class product
  -- *         - \ref NVML_ERROR_NO_PERMISSION     if the user doesn't have permission to perform this operation
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  -- * 
  -- * @see nvmlUnitGetLedState()
  --  

   function nvmlUnitSetLedState (unit : nvmlUnit_t; color : nvmlLedColor_t) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:3579
   pragma Import (C, nvmlUnitSetLedState, "nvmlUnitSetLedState");

  --* @}  
  --************************************************************************************************* 
  --* @defgroup nvmlDeviceCommands Device Commands
  -- *  This chapter describes NVML operations that change the state of the device.
  -- *  Each of these requires root/admin access. Non-admin users will see an NVML_ERROR_NO_PERMISSION
  -- *  error code when invoking any of these methods.
  -- *  @{
  --  

  --************************************************************************************************* 
  --*
  -- * Set the persistence mode for the device.
  -- *
  -- * For all products.
  -- * For Linux only.
  -- * Requires root/admin permissions.
  -- *
  -- * The persistence mode determines whether the GPU driver software is torn down after the last client
  -- * exits.
  -- *
  -- * This operation takes effect immediately. It is not persistent across reboots. After each reboot the
  -- * persistence mode is reset to "Disabled".
  -- *
  -- * See \ref nvmlEnableState_t for available modes.
  -- *
  -- * @param device                               The identifier of the target device
  -- * @param mode                                 The target persistence mode
  -- * 
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if the persistence mode was set
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a mode is invalid
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
  -- *         - \ref NVML_ERROR_NO_PERMISSION     if the user doesn't have permission to perform this operation
  -- *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  -- *
  -- * @see nvmlDeviceGetPersistenceMode()
  --  

   function nvmlDeviceSetPersistenceMode (device : nvmlDevice_t; mode : nvmlEnableState_t) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:3621
   pragma Import (C, nvmlDeviceSetPersistenceMode, "nvmlDeviceSetPersistenceMode");

  --*
  -- * Set the compute mode for the device.
  -- *
  -- * For all products.
  -- * Requires root/admin permissions.
  -- *
  -- * The compute mode determines whether a GPU can be used for compute operations and whether it can
  -- * be shared across contexts.
  -- *
  -- * This operation takes effect immediately. Under Linux it is not persistent across reboots and
  -- * always resets to "Default". Under windows it is persistent.
  -- *
  -- * Under windows compute mode may only be set to DEFAULT when running in WDDM
  -- *
  -- * See \ref nvmlComputeMode_t for details on available compute modes.
  -- *
  -- * @param device                               The identifier of the target device
  -- * @param mode                                 The target compute mode
  -- * 
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if the compute mode was set
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a mode is invalid
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
  -- *         - \ref NVML_ERROR_NO_PERMISSION     if the user doesn't have permission to perform this operation
  -- *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  -- *
  -- * @see nvmlDeviceGetComputeMode()
  --  

   function nvmlDeviceSetComputeMode (device : nvmlDevice_t; mode : nvmlComputeMode_t) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:3653
   pragma Import (C, nvmlDeviceSetComputeMode, "nvmlDeviceSetComputeMode");

  --*
  -- * Set the ECC mode for the device.
  -- *
  -- * For Kepler &tm; or newer fully supported devices.
  -- * Only applicable to devices with ECC.
  -- * Requires \a NVML_INFOROM_ECC version 1.0 or higher.
  -- * Requires root/admin permissions.
  -- *
  -- * The ECC mode determines whether the GPU enables its ECC support.
  -- *
  -- * This operation takes effect after the next reboot.
  -- *
  -- * See \ref nvmlEnableState_t for details on available modes.
  -- *
  -- * @param device                               The identifier of the target device
  -- * @param ecc                                  The target ECC mode
  -- * 
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if the ECC mode was set
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a ecc is invalid
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
  -- *         - \ref NVML_ERROR_NO_PERMISSION     if the user doesn't have permission to perform this operation
  -- *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  -- *
  -- * @see nvmlDeviceGetEccMode()
  --  

   function nvmlDeviceSetEccMode (device : nvmlDevice_t; ecc : nvmlEnableState_t) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:3683
   pragma Import (C, nvmlDeviceSetEccMode, "nvmlDeviceSetEccMode");

  --*
  -- * Clear the ECC error and other memory error counts for the device.
  -- *
  -- * For Kepler &tm; or newer fully supported devices.
  -- * Only applicable to devices with ECC.
  -- * Requires \a NVML_INFOROM_ECC version 2.0 or higher to clear aggregate location-based ECC counts.
  -- * Requires \a NVML_INFOROM_ECC version 1.0 or higher to clear all other ECC counts.
  -- * Requires root/admin permissions.
  -- * Requires ECC Mode to be enabled.
  -- *
  -- * Sets all of the specified ECC counters to 0, including both detailed and total counts.
  -- *
  -- * This operation takes effect immediately.
  -- *
  -- * See \ref nvmlMemoryErrorType_t for details on available counter types.
  -- *
  -- * @param device                               The identifier of the target device
  -- * @param counterType                          Flag that indicates which type of errors should be cleared.
  -- * 
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if the error counts were cleared
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a counterType is invalid
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
  -- *         - \ref NVML_ERROR_NO_PERMISSION     if the user doesn't have permission to perform this operation
  -- *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  -- *
  -- * @see 
  -- *      - nvmlDeviceGetDetailedEccErrors()
  -- *      - nvmlDeviceGetTotalEccErrors()
  --  

   function nvmlDeviceClearEccErrorCounts (device : nvmlDevice_t; counterType : nvmlEccCounterType_t) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:3717
   pragma Import (C, nvmlDeviceClearEccErrorCounts, "nvmlDeviceClearEccErrorCounts");

  --*
  -- * Set the driver model for the device.
  -- *
  -- * For Fermi &tm; or newer fully supported devices.
  -- * For windows only.
  -- * Requires root/admin permissions.
  -- *
  -- * On Windows platforms the device driver can run in either WDDM or WDM (TCC) mode. If a display is attached
  -- * to the device it must run in WDDM mode.  
  -- *
  -- * It is possible to force the change to WDM (TCC) while the display is still attached with a force flag (nvmlFlagForce).
  -- * This should only be done if the host is subsequently powered down and the display is detached from the device
  -- * before the next reboot. 
  -- *
  -- * This operation takes effect after the next reboot.
  -- * 
  -- * Windows driver model may only be set to WDDM when running in DEFAULT compute mode.
  -- *
  -- * Change driver model to WDDM is not supported when GPU doesn't support graphics acceleration or 
  -- * will not support it after reboot. See \ref nvmlDeviceSetGpuOperationMode.
  -- *
  -- * See \ref nvmlDriverModel_t for details on available driver models.
  -- * See \ref nvmlFlagDefault and \ref nvmlFlagForce
  -- *
  -- * @param device                               The identifier of the target device
  -- * @param driverModel                          The target driver model
  -- * @param flags                                Flags that change the default behavior
  -- * 
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if the driver model has been set
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a driverModel is invalid
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if the platform is not windows or the device does not support this feature
  -- *         - \ref NVML_ERROR_NO_PERMISSION     if the user doesn't have permission to perform this operation
  -- *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  -- * 
  -- * @see nvmlDeviceGetDriverModel()
  --  

   function nvmlDeviceSetDriverModel
     (device : nvmlDevice_t;
      driverModel : nvmlDriverModel_t;
      flags : unsigned) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:3758
   pragma Import (C, nvmlDeviceSetDriverModel, "nvmlDeviceSetDriverModel");

  --*
  -- * Set clocks that applications will lock to.
  -- *
  -- * Sets the clocks that compute and graphics applications will be running at.
  -- * e.g. CUDA driver requests these clocks during context creation which means this property 
  -- * defines clocks at which CUDA applications will be running unless some overspec event
  -- * occurs (e.g. over power, over thermal or external HW brake).
  -- *
  -- * Can be used as a setting to request constant performance.
  -- *
  -- * On Pascal and newer hardware, this will automatically disable automatic boosting of clocks.
  -- *
  -- * On K80 and newer Kepler and Maxwell GPUs, users desiring fixed performance should also call
  -- * \ref nvmlDeviceSetAutoBoostedClocksEnabled to prevent clocks from automatically boosting
  -- * above the clock value being set.
  -- *
  -- * For Kepler &tm; or newer non-GeForce fully supported devices and Maxwell or newer GeForce devices.
  -- * Requires root/admin permissions. 
  -- *
  -- * See \ref nvmlDeviceGetSupportedMemoryClocks and \ref nvmlDeviceGetSupportedGraphicsClocks 
  -- * for details on how to list available clocks combinations.
  -- *
  -- * After system reboot or driver reload applications clocks go back to their default value.
  -- * See \ref nvmlDeviceResetApplicationsClocks.
  -- *
  -- * @param device                               The identifier of the target device
  -- * @param memClockMHz                          Requested memory clock in MHz
  -- * @param graphicsClockMHz                     Requested graphics clock in MHz
  -- * 
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if new settings were successfully set
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a memClockMHz and \a graphicsClockMHz 
  -- *                                                 is not a valid clock combination
  -- *         - \ref NVML_ERROR_NO_PERMISSION     if the user doesn't have permission to perform this operation 
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
  -- *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  --  

   function nvmlDeviceSetApplicationsClocks
     (device : nvmlDevice_t;
      memClockMHz : unsigned;
      graphicsClockMHz : unsigned) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:3799
   pragma Import (C, nvmlDeviceSetApplicationsClocks, "nvmlDeviceSetApplicationsClocks");

  --*
  -- * Set new power limit of this device.
  -- * 
  -- * For Kepler &tm; or newer fully supported devices.
  -- * Requires root/admin permissions.
  -- *
  -- * See \ref nvmlDeviceGetPowerManagementLimitConstraints to check the allowed ranges of values.
  -- *
  -- * \note Limit is not persistent across reboots or driver unloads.
  -- * Enable persistent mode to prevent driver from unloading when no application is using the device.
  -- *
  -- * @param device                               The identifier of the target device
  -- * @param limit                                Power management limit in milliwatts to set
  -- * 
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if \a limit has been set
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a defaultLimit is out of range
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
  -- *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  -- *
  -- * @see nvmlDeviceGetPowerManagementLimitConstraints
  -- * @see nvmlDeviceGetPowerManagementDefaultLimit
  --  

   function nvmlDeviceSetPowerManagementLimit (device : nvmlDevice_t; limit : unsigned) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:3826
   pragma Import (C, nvmlDeviceSetPowerManagementLimit, "nvmlDeviceSetPowerManagementLimit");

  --*
  -- * Sets new GOM. See \a nvmlGpuOperationMode_t for details.
  -- *
  -- * For GK110 M-class and X-class Tesla &tm; products from the Kepler family.
  -- * Modes \ref NVML_GOM_LOW_DP and \ref NVML_GOM_ALL_ON are supported on fully supported GeForce products.
  -- * Not supported on Quadro &reg; and Tesla &tm; C-class products.
  -- * Requires root/admin permissions.
  -- * 
  -- * Changing GOMs requires a reboot. 
  -- * The reboot requirement might be removed in the future.
  -- *
  -- * Compute only GOMs don't support graphics acceleration. Under windows switching to these GOMs when
  -- * pending driver model is WDDM is not supported. See \ref nvmlDeviceSetDriverModel.
  -- * 
  -- * @param device                               The identifier of the target device
  -- * @param mode                                 Target GOM
  -- * 
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if \a mode has been set
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a mode incorrect
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support GOM or specific mode
  -- *         - \ref NVML_ERROR_NO_PERMISSION     if the user doesn't have permission to perform this operation
  -- *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  -- *
  -- * @see nvmlGpuOperationMode_t
  -- * @see nvmlDeviceGetGpuOperationMode
  --  

   function nvmlDeviceSetGpuOperationMode (device : nvmlDevice_t; mode : nvmlGpuOperationMode_t) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:3857
   pragma Import (C, nvmlDeviceSetGpuOperationMode, "nvmlDeviceSetGpuOperationMode");

  --*
  -- * Changes the root/admin restructions on certain APIs. See \a nvmlRestrictedAPI_t for the list of supported APIs.
  -- * This method can be used by a root/admin user to give non-root/admin access to certain otherwise-restricted APIs.
  -- * The new setting lasts for the lifetime of the NVIDIA driver; it is not persistent. See \a nvmlDeviceGetAPIRestriction
  -- * to query the current restriction settings.
  -- * 
  -- * For Kepler &tm; or newer fully supported devices.
  -- * Requires root/admin permissions.
  -- *
  -- * @param device                               The identifier of the target device
  -- * @param apiType                              Target API type for this operation
  -- * @param isRestricted                         The target restriction
  -- *
  -- * @return
  -- *         - \ref NVML_SUCCESS                 if \a isRestricted has been set
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a apiType incorrect
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support changing API restrictions or the device does not support
  -- *                                                 the feature that api restrictions are being set for (E.G. Enabling/disabling auto 
  -- *                                                 boosted clocks is not supported by the device)
  -- *         - \ref NVML_ERROR_NO_PERMISSION     if the user doesn't have permission to perform this operation
  -- *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  -- *
  -- * @see nvmlRestrictedAPI_t
  --  

   function nvmlDeviceSetAPIRestriction
     (device : nvmlDevice_t;
      apiType : nvmlRestrictedAPI_t;
      isRestricted : nvmlEnableState_t) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:3885
   pragma Import (C, nvmlDeviceSetAPIRestriction, "nvmlDeviceSetAPIRestriction");

  --*
  -- * @}
  --  

  --* @addtogroup nvmlAccountingStats
  -- *  @{
  --  

  --*
  -- * Enables or disables per process accounting.
  -- *
  -- * For Kepler &tm; or newer fully supported devices.
  -- * Requires root/admin permissions.
  -- *
  -- * @note This setting is not persistent and will default to disabled after driver unloads.
  -- *       Enable persistence mode to be sure the setting doesn't switch off to disabled.
  -- * 
  -- * @note Enabling accounting mode has no negative impact on the GPU performance.
  -- *
  -- * @note Disabling accounting clears all accounting pids information.
  -- *
  -- * See \ref nvmlDeviceGetAccountingMode
  -- * See \ref nvmlDeviceGetAccountingStats
  -- * See \ref nvmlDeviceClearAccountingPids
  -- *
  -- * @param device                               The identifier of the target device
  -- * @param mode                                 The target accounting mode
  -- *
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if the new mode has been set
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device or \a mode are invalid
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
  -- *         - \ref NVML_ERROR_NO_PERMISSION     if the user doesn't have permission to perform this operation
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  --  

   function nvmlDeviceSetAccountingMode (device : nvmlDevice_t; mode : nvmlEnableState_t) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:3923
   pragma Import (C, nvmlDeviceSetAccountingMode, "nvmlDeviceSetAccountingMode");

  --*
  -- * Clears accounting information about all processes that have already terminated.
  -- *
  -- * For Kepler &tm; or newer fully supported devices.
  -- * Requires root/admin permissions.
  -- *
  -- * See \ref nvmlDeviceGetAccountingMode
  -- * See \ref nvmlDeviceGetAccountingStats
  -- * See \ref nvmlDeviceSetAccountingMode
  -- *
  -- * @param device                               The identifier of the target device
  -- *
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if accounting information has been cleared 
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device are invalid
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
  -- *         - \ref NVML_ERROR_NO_PERMISSION     if the user doesn't have permission to perform this operation
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  --  

   function nvmlDeviceClearAccountingPids (device : nvmlDevice_t) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:3945
   pragma Import (C, nvmlDeviceClearAccountingPids, "nvmlDeviceClearAccountingPids");

  --* @}  
  --************************************************************************************************* 
  --* @defgroup NvLink NvLink Methods
  -- * This chapter describes methods that NVML can perform on NVLINK enabled devices.
  -- *  @{
  --  

  --************************************************************************************************* 
  --*
  -- * Retrieves the state of the device's NvLink for the link specified
  -- *
  -- * For newer than Maxwell &tm; fully supported devices.
  -- *
  -- * @param device                               The identifier of the target device
  -- * @param link                                 Specifies the NvLink link to be queried
  -- * @param isActive                             \a nvmlEnableState_t where NVML_FEATURE_ENABLED indicates that
  -- *                                             the link is active and NVML_FEATURE_DISABLED indicates it 
  -- *                                             is inactive
  -- *
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if \a isActive has been set
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device or \a link is invalid or \a isActive is NULL
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  --  

   function nvmlDeviceGetNvLinkState
     (device : nvmlDevice_t;
      link : unsigned;
      isActive : access nvmlEnableState_t) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:3974
   pragma Import (C, nvmlDeviceGetNvLinkState, "nvmlDeviceGetNvLinkState");

  --*
  -- * Retrieves the version of the device's NvLink for the link specified
  -- *
  -- * For newer than Maxwell &tm; fully supported devices.
  -- *
  -- * @param device                               The identifier of the target device
  -- * @param link                                 Specifies the NvLink link to be queried
  -- * @param version                              Requested NvLink version
  -- *
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if \a version has been set
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device or \a link is invalid or \a version is NULL
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  --  

   function nvmlDeviceGetNvLinkVersion
     (device : nvmlDevice_t;
      link : unsigned;
      version : access unsigned) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:3992
   pragma Import (C, nvmlDeviceGetNvLinkVersion, "nvmlDeviceGetNvLinkVersion");

  --*
  -- * Retrieves the requested capability from the device's NvLink for the link specified
  -- * Please refer to the \a nvmlNvLinkCapability_t structure for the specific caps that can be queried
  -- * The return value should be treated as a boolean.
  -- *
  -- * For newer than Maxwell &tm; fully supported devices.
  -- *
  -- * @param device                               The identifier of the target device
  -- * @param link                                 Specifies the NvLink link to be queried
  -- * @param capability                           Specifies the \a nvmlNvLinkCapability_t to be queried
  -- * @param capResult                            A boolean for the queried capability indicating that feature is available
  -- *
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if \a capResult has been set
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device, \a link, or \a capability is invalid or \a capResult is NULL
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  --  

   function nvmlDeviceGetNvLinkCapability
     (device : nvmlDevice_t;
      link : unsigned;
      capability : nvmlNvLinkCapability_t;
      capResult : access unsigned) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:4013
   pragma Import (C, nvmlDeviceGetNvLinkCapability, "nvmlDeviceGetNvLinkCapability");

  --*
  -- * Retrieves the PCI information for the remote node on a NvLink link 
  -- * Note: pciSubSystemId is not filled in this function and is indeterminate
  -- *
  -- * For newer than Maxwell &tm; fully supported devices.
  -- *
  -- * @param device                               The identifier of the target device
  -- * @param link                                 Specifies the NvLink link to be queried
  -- * @param pci                                  \a nvmlPciInfo_t of the remote node for the specified link                            
  -- *
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if \a pci has been set
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device or \a link is invalid or \a pci is NULL
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  --  

   function nvmlDeviceGetNvLinkRemotePciInfo
     (device : nvmlDevice_t;
      link : unsigned;
      pci : access nvmlPciInfo_t) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:4033
   pragma Import (C, nvmlDeviceGetNvLinkRemotePciInfo, "nvmlDeviceGetNvLinkRemotePciInfo");

  --*
  -- * Retrieves the specified error counter value
  -- * Please refer to \a nvmlNvLinkErrorCounter_t for error counters that are available
  -- *
  -- * For newer than Maxwell &tm; fully supported devices.
  -- *
  -- * @param device                               The identifier of the target device
  -- * @param link                                 Specifies the NvLink link to be queried
  -- * @param counter                              Specifies the NvLink counter to be queried
  -- * @param counterValue                         Returned counter value
  -- *
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if \a counter has been set
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device, \a link, or \a counter is invalid or \a counterValue is NULL
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  --  

   function nvmlDeviceGetNvLinkErrorCounter
     (device : nvmlDevice_t;
      link : unsigned;
      counter : nvmlNvLinkErrorCounter_t;
      counterValue : access Extensions.unsigned_long_long) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:4053
   pragma Import (C, nvmlDeviceGetNvLinkErrorCounter, "nvmlDeviceGetNvLinkErrorCounter");

  --*
  -- * Resets all error counters to zero
  -- * Please refer to \a nvmlNvLinkErrorCounter_t for the list of error counters that are reset
  -- *
  -- * For newer than Maxwell &tm; fully supported devices.
  -- *
  -- * @param device                               The identifier of the target device
  -- * @param link                                 Specifies the NvLink link to be queried
  -- *
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if the reset is successful
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device or \a link is invalid
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  --  

   function nvmlDeviceResetNvLinkErrorCounters (device : nvmlDevice_t; link : unsigned) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:4072
   pragma Import (C, nvmlDeviceResetNvLinkErrorCounters, "nvmlDeviceResetNvLinkErrorCounters");

  --*
  -- * Set the NVLINK utilization counter control information for the specified counter, 0 or 1.
  -- * Please refer to \a nvmlNvLinkUtilizationControl_t for the structure definition.  Performs a reset
  -- * of the counters if the reset parameter is non-zero.
  -- *
  -- * For newer than Maxwell &tm; fully supported devices.
  -- *
  -- * @param device                               The identifier of the target device
  -- * @param counter                              Specifies the counter that should be set (0 or 1).
  -- * @param link                                 Specifies the NvLink link to be queried
  -- * @param control                              A reference to the \a nvmlNvLinkUtilizationControl_t to set
  -- * @param reset                                Resets the counters on set if non-zero
  -- *
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if the control has been set successfully
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device, \a counter, \a link, or \a control is invalid 
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  --  

   function nvmlDeviceSetNvLinkUtilizationControl
     (device : nvmlDevice_t;
      link : unsigned;
      counter : unsigned;
      control : access nvmlNvLinkUtilizationControl_t;
      reset : unsigned) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:4094
   pragma Import (C, nvmlDeviceSetNvLinkUtilizationControl, "nvmlDeviceSetNvLinkUtilizationControl");

  --*
  -- * Get the NVLINK utilization counter control information for the specified counter, 0 or 1.
  -- * Please refer to \a nvmlNvLinkUtilizationControl_t for the structure definition
  -- *
  -- * For newer than Maxwell &tm; fully supported devices.
  -- *
  -- * @param device                               The identifier of the target device
  -- * @param counter                              Specifies the counter that should be set (0 or 1).
  -- * @param link                                 Specifies the NvLink link to be queried
  -- * @param control                              A reference to the \a nvmlNvLinkUtilizationControl_t to place information
  -- *
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if the control has been set successfully
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device, \a counter, \a link, or \a control is invalid 
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  --  

   function nvmlDeviceGetNvLinkUtilizationControl
     (device : nvmlDevice_t;
      link : unsigned;
      counter : unsigned;
      control : access nvmlNvLinkUtilizationControl_t) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:4115
   pragma Import (C, nvmlDeviceGetNvLinkUtilizationControl, "nvmlDeviceGetNvLinkUtilizationControl");

  --*
  -- * Retrieve the NVLINK utilization counter based on the current control for a specified counter.
  -- * In general it is good practice to use \a nvmlDeviceSetNvLinkUtilizationControl
  -- *  before reading the utilization counters as they have no default state
  -- *
  -- * For newer than Maxwell &tm; fully supported devices.
  -- *
  -- * @param device                               The identifier of the target device
  -- * @param link                                 Specifies the NvLink link to be queried
  -- * @param counter                              Specifies the counter that should be read (0 or 1).
  -- * @param rxcounter                            Receive counter return value
  -- * @param txcounter                            Transmit counter return value
  -- *
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if \a rxcounter and \a txcounter have been successfully set
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device, \a counter, or \a link is invalid or \a rxcounter or \a txcounter are NULL
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  --  

   function nvmlDeviceGetNvLinkUtilizationCounter
     (device : nvmlDevice_t;
      link : unsigned;
      counter : unsigned;
      rxcounter : access Extensions.unsigned_long_long;
      txcounter : access Extensions.unsigned_long_long) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:4139
   pragma Import (C, nvmlDeviceGetNvLinkUtilizationCounter, "nvmlDeviceGetNvLinkUtilizationCounter");

  --*
  -- * Freeze the NVLINK utilization counters 
  -- * Both the receive and transmit counters are operated on by this function
  -- *
  -- * For newer than Maxwell &tm; fully supported devices.
  -- *
  -- * @param device                               The identifier of the target device
  -- * @param link                                 Specifies the NvLink link to be queried
  -- * @param counter                              Specifies the counter that should be frozen (0 or 1).
  -- * @param freeze                               NVML_FEATURE_ENABLED = freeze the receive and transmit counters
  -- *                                             NVML_FEATURE_DISABLED = unfreeze the receive and transmit counters
  -- *
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if counters were successfully frozen or unfrozen
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device, \a link, \a counter, or \a freeze is invalid
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  --  

   function nvmlDeviceFreezeNvLinkUtilizationCounter
     (device : nvmlDevice_t;
      link : unsigned;
      counter : unsigned;
      freeze : nvmlEnableState_t) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:4161
   pragma Import (C, nvmlDeviceFreezeNvLinkUtilizationCounter, "nvmlDeviceFreezeNvLinkUtilizationCounter");

  --*
  -- * Reset the NVLINK utilization counters 
  -- * Both the receive and transmit counters are operated on by this function
  -- *
  -- * For newer than Maxwell &tm; fully supported devices.
  -- *
  -- * @param device                               The identifier of the target device
  -- * @param link                                 Specifies the NvLink link to be reset
  -- * @param counter                              Specifies the counter that should be reset (0 or 1)
  -- *
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if counters were successfully reset
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device, \a link, or \a counter is invalid
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  --  

   function nvmlDeviceResetNvLinkUtilizationCounter
     (device : nvmlDevice_t;
      link : unsigned;
      counter : unsigned) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:4181
   pragma Import (C, nvmlDeviceResetNvLinkUtilizationCounter, "nvmlDeviceResetNvLinkUtilizationCounter");

  --* @}  
  --************************************************************************************************* 
  --* @defgroup nvmlEvents Event Handling Methods
  -- * This chapter describes methods that NVML can perform against each device to register and wait for 
  -- * some event to occur.
  -- *  @{
  --  

  --************************************************************************************************* 
  --*
  -- * Create an empty set of events.
  -- * Event set should be freed by \ref nvmlEventSetFree
  -- *
  -- * For Fermi &tm; or newer fully supported devices.
  -- * @param set                                  Reference in which to return the event handle
  -- * 
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if the event has been set
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a set is NULL
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  -- * 
  -- * @see nvmlEventSetFree
  --  

   function nvmlEventSetCreate (set : System.Address) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:4208
   pragma Import (C, nvmlEventSetCreate, "nvmlEventSetCreate");

  --*
  -- * Starts recording of events on a specified devices and add the events to specified \ref nvmlEventSet_t
  -- *
  -- * For Fermi &tm; or newer fully supported devices.
  -- * Ecc events are available only on ECC enabled devices (see \ref nvmlDeviceGetTotalEccErrors)
  -- * Power capping events are available only on Power Management enabled devices (see \ref nvmlDeviceGetPowerManagementMode)
  -- *
  -- * For Linux only.
  -- *
  -- * \b IMPORTANT: Operations on \a set are not thread safe
  -- *
  -- * This call starts recording of events on specific device.
  -- * All events that occurred before this call are not recorded.
  -- * Checking if some event occurred can be done with \ref nvmlEventSetWait
  -- *
  -- * If function reports NVML_ERROR_UNKNOWN, event set is in undefined state and should be freed.
  -- * If function reports NVML_ERROR_NOT_SUPPORTED, event set can still be used. None of the requested eventTypes
  -- *     are registered in that case.
  -- *
  -- * @param device                               The identifier of the target device
  -- * @param eventTypes                           Bitmask of \ref nvmlEventType to record
  -- * @param set                                  Set to which add new event types
  -- * 
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if the event has been set
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a eventTypes is invalid or \a set is NULL
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if the platform does not support this feature or some of requested event types
  -- *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  -- * 
  -- * @see nvmlEventType
  -- * @see nvmlDeviceGetSupportedEventTypes
  -- * @see nvmlEventSetWait
  -- * @see nvmlEventSetFree
  --  

   function nvmlDeviceRegisterEvents
     (device : nvmlDevice_t;
      eventTypes : Extensions.unsigned_long_long;
      set : nvmlEventSet_t) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:4246
   pragma Import (C, nvmlDeviceRegisterEvents, "nvmlDeviceRegisterEvents");

  --*
  -- * Returns information about events supported on device
  -- *
  -- * For Fermi &tm; or newer fully supported devices.
  -- *
  -- * Events are not supported on Windows. So this function returns an empty mask in \a eventTypes on Windows.
  -- *
  -- * @param device                               The identifier of the target device
  -- * @param eventTypes                           Reference in which to return bitmask of supported events
  -- * 
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if the eventTypes has been set
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a eventType is NULL
  -- *         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  -- * 
  -- * @see nvmlEventType
  -- * @see nvmlDeviceRegisterEvents
  --  

   function nvmlDeviceGetSupportedEventTypes (device : nvmlDevice_t; eventTypes : access Extensions.unsigned_long_long) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:4268
   pragma Import (C, nvmlDeviceGetSupportedEventTypes, "nvmlDeviceGetSupportedEventTypes");

  --*
  -- * Waits on events and delivers events
  -- *
  -- * For Fermi &tm; or newer fully supported devices.
  -- *
  -- * If some events are ready to be delivered at the time of the call, function returns immediately.
  -- * If there are no events ready to be delivered, function sleeps till event arrives 
  -- * but not longer than specified timeout. This function in certain conditions can return before
  -- * specified timeout passes (e.g. when interrupt arrives)
  -- * 
  -- * In case of xid error, the function returns the most recent xid error type seen by the system. If there are multiple
  -- * xid errors generated before nvmlEventSetWait is invoked then the last seen xid error type is returned for all
  -- * xid error events.
  -- * 
  -- * @param set                                  Reference to set of events to wait on
  -- * @param data                                 Reference in which to return event data
  -- * @param timeoutms                            Maximum amount of wait time in milliseconds for registered event
  -- * 
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if the data has been set
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a data is NULL
  -- *         - \ref NVML_ERROR_TIMEOUT           if no event arrived in specified timeout or interrupt arrived
  -- *         - \ref NVML_ERROR_GPU_IS_LOST       if a GPU has fallen off the bus or is otherwise inaccessible
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  -- * 
  -- * @see nvmlEventType
  -- * @see nvmlDeviceRegisterEvents
  --  

   function nvmlEventSetWait
     (set : nvmlEventSet_t;
      data : access nvmlEventData_t;
      timeoutms : unsigned) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:4299
   pragma Import (C, nvmlEventSetWait, "nvmlEventSetWait");

  --*
  -- * Releases events in the set
  -- *
  -- * For Fermi &tm; or newer fully supported devices.
  -- *
  -- * @param set                                  Reference to events to be released 
  -- * 
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if the event has been successfully released
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  -- * 
  -- * @see nvmlDeviceRegisterEvents
  --  

   function nvmlEventSetFree (set : nvmlEventSet_t) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:4315
   pragma Import (C, nvmlEventSetFree, "nvmlEventSetFree");

  --* @}  
  --************************************************************************************************* 
  --* @defgroup nvmlZPI Drain states 
  -- * This chapter describes methods that NVML can perform against each device to control their drain state
  -- * and recognition by NVML and NVIDIA kernel driver. These methods can be used with out-of-band tools to
  -- * power on/off GPUs, enable robust reset scenarios, etc.
  -- *  @{
  --  

  --************************************************************************************************* 
  --*
  -- * Modify the drain state of a GPU.  This method forces a GPU to no longer accept new incoming requests.
  -- * Any new NVML process will no longer see this GPU.  Persistence mode for this GPU must be turned off before
  -- * this call is made.
  -- * Must be called as administrator.
  -- * For Linux only.
  -- * 
  -- * For newer than Maxwell &tm; fully supported devices.
  -- * Some Kepler devices supported.
  -- *
  -- * @param pciInfo                              The PCI address of the GPU drain state to be modified
  -- * @param newState                             The drain state that should be entered, see \ref nvmlEnableState_t
  -- *
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if counters were successfully reset
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a nvmlIndex or \a newState is invalid
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
  -- *         - \ref NVML_ERROR_NO_PERMISSION     if the calling process has insufficient permissions to perform operation
  -- *         - \ref NVML_ERROR_IN_USE            if the device has persistence mode turned on
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  --  

   function nvmlDeviceModifyDrainState (pciInfo : access nvmlPciInfo_t; newState : nvmlEnableState_t) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:4350
   pragma Import (C, nvmlDeviceModifyDrainState, "nvmlDeviceModifyDrainState");

  --*
  -- * Query the drain state of a GPU.  This method is used to check if a GPU is in a currently draining
  -- * state.
  -- * For Linux only.
  -- * 
  -- * For newer than Maxwell &tm; fully supported devices.
  -- * Some Kepler devices supported.
  -- *
  -- * @param pciInfo                              The PCI address of the GPU drain state to be queried
  -- * @param currentState                         The current drain state for this GPU, see \ref nvmlEnableState_t
  -- *
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if counters were successfully reset
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a nvmlIndex or \a currentState is invalid
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  --  

   function nvmlDeviceQueryDrainState (pciInfo : access nvmlPciInfo_t; currentState : access nvmlEnableState_t) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:4370
   pragma Import (C, nvmlDeviceQueryDrainState, "nvmlDeviceQueryDrainState");

  --*
  -- * This method will remove the specified GPU from the view of both NVML and the NVIDIA kernel driver
  -- * as long as no other processes are attached. If other processes are attached, this call will return
  -- * NVML_ERROR_IN_USE and the GPU will be returned to its original "draining" state. Note: the
  -- * only situation where a process can still be attached after nvmlDeviceModifyDrainState() is called
  -- * to initiate the draining state is if that process was using, and is still using, a GPU before the 
  -- * call was made. Also note, persistence mode counts as an attachment to the GPU thus it must be disabled
  -- * prior to this call.
  -- *
  -- * For long-running NVML processes please note that this will change the enumeration of current GPUs.
  -- * For example, if there are four GPUs present and GPU1 is removed, the new enumeration will be 0-2.
  -- * Also, device handles after the removed GPU will not be valid and must be re-established.
  -- * Must be run as administrator. 
  -- * For Linux only.
  -- *
  -- * For newer than Maxwell &tm; fully supported devices.
  -- * Some Kepler devices supported.
  -- *
  -- * @param pciInfo                              The PCI address of the GPU to be removed
  -- *
  -- * @return
  -- *         - \ref NVML_SUCCESS                 if counters were successfully reset
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a nvmlIndex is invalid
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
  -- *         - \ref NVML_ERROR_IN_USE            if the device is still in use and cannot be removed
  --  

   function nvmlDeviceRemoveGpu (pciInfo : access nvmlPciInfo_t) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:4399
   pragma Import (C, nvmlDeviceRemoveGpu, "nvmlDeviceRemoveGpu");

  --*
  -- * Request the OS and the NVIDIA kernel driver to rediscover a portion of the PCI subsystem looking for GPUs that
  -- * were previously removed. The portion of the PCI tree can be narrowed by specifying a domain, bus, and device.  
  -- * If all are zeroes then the entire PCI tree will be searched.  Please note that for long-running NVML processes
  -- * the enumeration will change based on how many GPUs are discovered and where they are inserted in bus order.
  -- *
  -- * In addition, all newly discovered GPUs will be initialized and their ECC scrubbed which may take several seconds
  -- * per GPU. Also, all device handles are no longer guaranteed to be valid post discovery.
  -- *
  -- * Must be run as administrator.
  -- * For Linux only.
  -- * 
  -- * For newer than Maxwell &tm; fully supported devices.
  -- * Some Kepler devices supported.
  -- *
  -- * @param pciInfo                              The PCI tree to be searched.  Only the domain, bus, and device
  -- *                                             fields are used in this call.
  -- *
  -- * @return 
  -- *         - \ref NVML_SUCCESS                 if counters were successfully reset
  -- *         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
  -- *         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a pciInfo is invalid
  -- *         - \ref NVML_ERROR_NOT_SUPPORTED     if the operating system does not support this feature
  -- *         - \ref NVML_ERROR_OPERATING_SYSTEM  if the operating system is denying this feature
  -- *         - \ref NVML_ERROR_NO_PERMISSION     if the calling process has insufficient permissions to perform operation
  -- *         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
  --  

   function nvmlDeviceDiscoverGpus (pciInfo : access nvmlPciInfo_t) return nvmlReturn_t;  -- /usr/local/cuda-8.0/include/nvml.h:4428
   pragma Import (C, nvmlDeviceDiscoverGpus, "nvmlDeviceDiscoverGpus");

  --* @}  
  --*
  -- * NVML API versioning support
  --  

end nvml_h;

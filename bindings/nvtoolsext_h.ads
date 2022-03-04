pragma Ada_2005;
pragma Style_Checks (Off);

with Interfaces.C; use Interfaces.C;
with stdint_h;
with System;
with Interfaces.C.Strings;

package nvToolsExt_h is

   --  unsupported macro: NVTX_INLINE_STATIC inline static
   NVTX_VERSION : constant := 2;  --  /usr/local/cuda-8.0/include/nvToolsExt.h:223
   --  unsupported macro: NVTX_EVENT_ATTRIB_STRUCT_SIZE ( (uint16_t)( sizeof(nvtxEventAttributes_t) ) )
   --  unsupported macro: NVTX_INITIALIZATION_ATTRIB_STRUCT_SIZE ( (uint16_t)( sizeof(nvtxInitializationAttributes_t) ) )
   --  unsupported macro: NVTX_NO_PUSH_POP_TRACKING ((int)-2)
   --  unsupported macro: NVTX_RESOURCE_MAKE_TYPE(CLASS,INDEX) ((((uint32_t)(NVTX_RESOURCE_CLASS_ ## CLASS))<<16)|((uint32_t)(INDEX)))

   NVTX_RESOURCE_CLASS_GENERIC : constant := 1;  --  /usr/local/cuda-8.0/include/nvToolsExt.h:1080
   --  unsupported macro: NVTX_RESOURCE_ATTRIB_STRUCT_SIZE ( (uint16_t)( sizeof(nvtxResourceAttributes_v0) ) )
   --  unsupported macro: nvtxMark nvtxMarkA
   --  unsupported macro: nvtxRangeStart nvtxRangeStartA
   --  unsupported macro: nvtxRangePush nvtxRangePushA
   --  unsupported macro: nvtxNameCategory nvtxNameCategoryA
   --  unsupported macro: nvtxNameOsThread nvtxNameOsThreadA
   --  unsupported macro: nvtxDomainCreate nvtxDomainCreateA
   --  unsupported macro: nvtxDomainRegisterString nvtxDomainRegisterStringA
   --  unsupported macro: nvtxDomainNameCategory nvtxDomainNameCategoryA

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

  --* \file nvToolsExt.h
  --  

  -- =========================================================================  
  --* \mainpage
  -- * \tableofcontents
  -- * \section INTRODUCTION Introduction
  -- *
  -- * The NVIDIA Tools Extension library is a set of functions that a
  -- * developer can use to provide additional information to tools.
  -- * The additional information is used by the tool to improve
  -- * analysis and visualization of data.
  -- *
  -- * The library introduces close to zero overhead if no tool is
  -- * attached to the application.  The overhead when a tool is
  -- * attached is specific to the tool.
  -- *
  -- * \section INITIALIZATION_SECTION Initialization
  -- *
  -- * Typically the tool's library that plugs into NVTX is indirectly 
  -- * loaded via enviromental properties that are platform specific. 
  -- * For some platform or special cases, the user may be required 
  -- * to instead explicity initialize instead though.   This can also
  -- * be helpful to control when the API loads a tool's library instead
  -- * of what would typically be the first function call to emit info.
  -- * For these rare case, see \ref INITIALIZATION for additional information.
  -- *
  -- * \section MARKERS_AND_RANGES Markers and Ranges
  -- *
  -- * Markers and ranges are used to describe events at a specific time (markers)
  -- * or over a time span (ranges) during the execution of the application
  -- * respectively. 
  -- *
  -- * \subsection MARKERS Markers
  -- * 
  -- * Markers denote specific moments in time.
  -- * 
  -- * 
  -- * See \ref DOMAINS and \ref EVENT_ATTRIBUTES for additional information on
  -- * how to specify the domain.
  -- * 
  -- * \subsection THREAD_RANGES Thread Ranges
  -- *
  -- * Thread ranges denote nested time ranges. Nesting is maintained per thread
  -- * per domain and does not require any additional correlation mechanism. The
  -- * duration of a thread range is defined by the corresponding pair of
  -- * nvtxRangePush* to nvtxRangePop API calls.
  -- *
  -- * See \ref DOMAINS and \ref EVENT_ATTRIBUTES for additional information on
  -- * how to specify the domain.
  -- *
  -- * \subsection PROCESS_RANGES Process Ranges
  -- *
  -- * Process ranges denote a time span that can expose arbitrary concurrency, as 
  -- * opposed to thread ranges that only support nesting. In addition the range
  -- * start event can happen on a different thread than the end marker. For the 
  -- * correlation of a start/end pair an unique correlation ID is used that is
  -- * returned from the start API call and needs to be passed into the end API
  -- * call.
  -- *
  -- * \subsection EVENT_ATTRIBUTES Event Attributes
  -- *
  -- * \ref MARKERS_AND_RANGES can be annotated with various attributes to provide
  -- * additional information for an event or to guide the tool's visualization of
  -- * the data. Each of the attributes is optional and if left unused the
  -- * attributes fall back to a default value. The attributes include:
  -- * - color
  -- * - category
  -- *
  -- * To specify any attribute other than the text message, the \ref
  -- * EVENT_ATTRIBUTE_STRUCTURE "Event Attribute Structure" must be used.
  -- *
  -- * \section DOMAINS Domains
  -- *
  -- * Domains enable developers to scope annotations. By default all events and
  -- * annotations are in the default domain. Additional domains can be registered.
  -- * This allows developers to scope markers, ranges, and resources names to
  -- * avoid conflicts.
  -- *
  -- * The function ::nvtxDomainCreateA or ::nvtxDomainCreateW is used to create
  -- * a named domain.
  -- * 
  -- * Each domain maintains its own
  -- * - categories
  -- * - thread range stacks
  -- * - registered strings
  -- *
  -- * The function ::nvtxDomainDestroy marks the end of the domain. Destroying 
  -- * a domain unregisters and destroys all objects associated with it such as 
  -- * registered strings, resource objects, named categories, and started ranges. 
  -- *
  -- * \section RESOURCE_NAMING Resource Naming
  -- *
  -- * This section covers calls that allow to annotate objects with user-provided
  -- * names in order to allow for a better analysis of complex trace data. All of
  -- * the functions take the handle or the ID of the object to name and the name.
  -- * The functions can be called multiple times during the execution of an
  -- * application, however, in that case it is implementation dependent which
  -- * name will be reported by the tool.
  -- * 
  -- * \subsection CATEGORY_NAMING Category Naming
  -- *
  -- * Some function in this library support associating an integer category 
  -- * to enable filtering and sorting.  The category naming functions allow 
  -- * the application to associate a user friendly name with the integer 
  -- * category.  Support for domains have been added in NVTX_VERSION_2 to 
  -- * avoid collisions when domains are developed independantly. 
  -- *
  -- * \subsection RESOURCE_OBJECTS Resource Objects
  -- *
  -- * Resource objects are a generic mechanism for attaching data to an application 
  -- * resource.  The identifier field makes the association to a pointer or handle, 
  -- * while the type field helps provide deeper understanding of the identifier as 
  -- * well as enabling differentiation in cases where handles generated by different
  -- * APIs may collide.  The resource object may also have an associated message to
  -- * associate with the application resource, enabling further annotation of this 
  -- * object and how it is used.
  -- * 
  -- * The resource object was introduced in NVTX_VERSION_2 to supersede existing naming
  -- * functions and allow the application resource identified by those functions to be
  -- * associated to a domain.  The other naming functions are still supported for backward
  -- * compatibility but will be associated only to the default domain.
  -- *
  -- * \subsection RESOURCE_NAMING_OS Resource Naming
  -- * 
  -- * Some operating system resources creation APIs do not support providing a user friendly 
  -- * name, such as some OS thread creation APIs.  This API support resource naming though 
  -- * both through resource objects and functions following the pattern 
  -- * nvtxName[RESOURCE_TYPE][A|W](identifier, name).  Resource objects introduced in NVTX_VERSION 2 
  -- * supersede the other functions with a a more general method of assigning names to OS resources,
  -- * along with associating them to domains too.  The older nvtxName* functions are only associated 
  -- * with the default domain.
  -- * \section EXTENSIONS Optional Extensions
  -- * Optional extensions will either appear within the existing sections the extend or appear 
  -- * in the "Related Pages" when they introduce new concepts.
  --  

  --*
  -- * The nvToolsExt library depends on stdint.h.  If the build tool chain in use
  -- * does not include stdint.h then define NVTX_STDINT_TYPES_ALREADY_DEFINED
  -- * and define the following types:
  -- * <ul>
  -- *   <li>uint8_t
  -- *   <li>int8_t
  -- *   <li>uint16_t
  -- *   <li>int16_t
  -- *   <li>uint32_t
  -- *   <li>int32_t
  -- *   <li>uint64_t
  -- *   <li>int64_t
  -- *   <li>uintptr_t
  -- *   <li>intptr_t
  -- * </ul>
  -- #define NVTX_STDINT_TYPES_ALREADY_DEFINED if you are using your own header file.
  --  

  --*
  -- * Tools Extension API version
  --  

  --*
  -- * Size of the nvtxEventAttributes_t structure.
  --  

  --*
  -- * Size of the nvtxInitializationAttributes_t structure.
  --  

   subtype nvtxRangeId_t is stdint_h.uint64_t;  -- /usr/local/cuda-8.0/include/nvToolsExt.h:237

  -- \brief String Handle Structure.
  --* \anchor STRING_HANDLE_STRUCTURE
  --*
  --* This structure is opaque to the user and is used as a handle to reference
  --* a string.  The tools will return a pointer through the API for the application
  --* to hold on it's behalf to reference the string in the future.
  --*
  -- 

   --  skipped empty struct nvtxStringHandle

   type nvtxStringHandle_t is new System.Address;  -- /usr/local/cuda-8.0/include/nvToolsExt.h:248

  -- \brief Domain Handle Structure.
  --* \anchor DOMAIN_HANDLE_STRUCTURE
  --*
  --* This structure is opaque to the user and is used as a handle to reference
  --* a domain.  The tools will return a pointer through the API for the application
  --* to hold on its behalf to reference the domain in the future.
  --*
  -- 

   --  skipped empty struct nvtxDomainHandle

   type nvtxDomainHandle_t is new System.Address;  -- /usr/local/cuda-8.0/include/nvToolsExt.h:258

  -- =========================================================================  
  --* \defgroup GENERAL General
  -- * @{
  --  

  --* ---------------------------------------------------------------------------
  -- * Color Types
  -- * -------------------------------------------------------------------------  

   type nvtxColorType_t is 
     (NVTX_COLOR_UNKNOWN,
      NVTX_COLOR_ARGB);
   pragma Convention (C, nvtxColorType_t);  -- /usr/local/cuda-8.0/include/nvToolsExt.h:273

  --*< Color attribute is unused.  
  --*< An ARGB color is provided.  
  --* ---------------------------------------------------------------------------
  -- * Message Types
  -- * -------------------------------------------------------------------------  

   type nvtxMessageType_t is 
     (NVTX_MESSAGE_UNKNOWN,
      NVTX_MESSAGE_TYPE_ASCII,
      NVTX_MESSAGE_TYPE_UNICODE,
      NVTX_MESSAGE_TYPE_REGISTERED);
   pragma Convention (C, nvtxMessageType_t);  -- /usr/local/cuda-8.0/include/nvToolsExt.h:282

  --*< Message payload is unused.  
  --*< A character sequence is used as payload.  
  --*< A wide character sequence is used as payload.  
  -- NVTX_VERSION_2  
  --*< A unique string handle that was registered
  --                                                with \ref nvtxDomainRegisterStringA() or 
  --                                                \ref nvtxDomainRegisterStringW().  

   type nvtxMessageValue_t (discr : unsigned := 0) is record
      case discr is
         when 0 =>
            ascii : Interfaces.C.Strings.chars_ptr;  -- /usr/local/cuda-8.0/include/nvToolsExt.h:295
         when 1 =>
            unicode : access wchar_t;  -- /usr/local/cuda-8.0/include/nvToolsExt.h:296
         when others =>
            registered : nvtxStringHandle_t;  -- /usr/local/cuda-8.0/include/nvToolsExt.h:298
      end case;
   end record;
   pragma Convention (C_Pass_By_Copy, nvtxMessageValue_t);
   pragma Unchecked_Union (nvtxMessageValue_t);  -- /usr/local/cuda-8.0/include/nvToolsExt.h:293

  -- NVTX_VERSION_2  
  --* @}  
  --END defgroup 
  -- =========================================================================  
  --* \defgroup INITIALIZATION Initialization
  --* @{
  --* Typically the tool's library that plugs into NVTX is indirectly
  --* loaded via enviromental properties that are platform specific.
  --* For some platform or special cases, the user may be required
  --* to instead explicity initialize instead though.  This can also
  --* be helpful to control when the API loads a tool's library instead
  --* of what would typically be the first function call to emit info.
  -- 

  --* ---------------------------------------------------------------------------
  --* Initialization Modes
  --* -------------------------------------------------------------------------  

   type nvtxInitializationMode_t is 
     (NVTX_INITIALIZATION_MODE_UNKNOWN,
      NVTX_INITIALIZATION_MODE_CALLBACK_V1,
      NVTX_INITIALIZATION_MODE_CALLBACK_V2,
      NVTX_INITIALIZATION_MODE_SIZE);
   pragma Convention (C, nvtxInitializationMode_t);  -- /usr/local/cuda-8.0/include/nvToolsExt.h:318

  --*< A platform that supports indirect initialization will attempt this style, otherwise expect failure.  
  --*< A function pointer conforming to NVTX_VERSION=1 will be used.  
  --*< A function pointer conforming to NVTX_VERSION=2 will be used.  
  --* \brief Initialization Attribute Structure.
  --* \anchor INITIALIZATION_ATTRIBUTE_STRUCTURE
  --*
  --* This structure is used to describe the attributes used for initialization
  --* of the NVTX API.
  --*
  --* \par Initializing the Attributes
  --*
  --* The caller should always perform the following three tasks when using
  --* attributes:
  --* <ul>
  --*    <li>Zero the structure
  --*    <li>Set the version field
  --*    <li>Set the size field
  --* </ul>
  --*
  --* Zeroing the structure sets all the event attributes types and values
  --* to the default value.
  --*
  --* The version and size field are used by the Tools Extension
  --* implementation to handle multiple versions of the attributes structure.
  --* NVTX_INITIALIZATION_ATTRIB_STRUCT_SIZE may be used for the size.
  --*
  --* It is recommended that the caller use one of the following to methods
  --* to initialize the event attributes structure:
  --*
  --* \par Method 1: Initializing nvtxInitializationAttributes_t for future compatibility
  --* \code
  --* nvtxInitializationAttributes_t initAttribs = {0};
  --* initAttribs.version = NVTX_VERSION;
  --* initAttribs.size = NVTX_INITIALIZATION_ATTRIB_STRUCT_SIZE;
  --* \endcode
  --*
  --* \par Method 2: Initializing nvtxInitializationAttributes_t for a specific version
  --* \code
  --* nvtxInitializationAttributes_t initAttribs = {0};
  --* initAttribs.version =2;
  --* initAttribs.size = (uint16_t)(sizeof(nvtxInitializationAttributes_v2));
  --* \endcode
  --*
  --* If the caller uses Method 1 it is critical that the entire binary
  --* layout of the structure be configured to 0 so that all fields
  --* are initialized to the default value.
  --*
  --* The caller should either use both NVTX_VERSION and
  --* NVTX_INITIALIZATION_ATTRIB_STRUCT_SIZE (Method 1) or use explicit values
  --* and a versioned type (Method 2).  Using a mix of the two methods
  --* will likely cause either source level incompatibility or binary
  --* incompatibility in the future.
  --*
  --* \par Settings Attribute Types and Values
  --*
  --*
  --* \par Example:
  --* \code
  --* // Initialize
  --* nvtxInitializationAttributes_t initAttribs = {0};
  --* initAttribs.version = NVTX_VERSION;
  --* initAttribs.size = NVTX_INITIALIZATION_ATTRIB_STRUCT_SIZE;
  --*
  --* // Configure the Attributes
  --* initAttribs.mode = NVTX_INITIALIZATION_MODE_CALLBACK_V2;
  --* initAttribs.fnptr = InitializeInjectionNvtx2;
  --* \endcode
  --* \sa
  --* ::nvtxInitializationMode_t
  --* ::nvtxInitialize
  -- 

  --*
  --    * \brief Version flag of the structure.
  --    *
  --    * Needs to be set to NVTX_VERSION to indicate the version of NVTX APIs
  --    * supported in this header file. This can optionally be overridden to
  --    * another version of the tools extension library.
  --     

   type nvtxInitializationAttributes_v2 is record
      version : aliased stdint_h.uint16_t;  -- /usr/local/cuda-8.0/include/nvToolsExt.h:405
      size : aliased stdint_h.uint16_t;  -- /usr/local/cuda-8.0/include/nvToolsExt.h:413
      mode : aliased stdint_h.uint32_t;  -- /usr/local/cuda-8.0/include/nvToolsExt.h:425
      fnptr : access procedure;  -- /usr/local/cuda-8.0/include/nvToolsExt.h:445
   end record;
   pragma Convention (C_Pass_By_Copy, nvtxInitializationAttributes_v2);  -- /usr/local/cuda-8.0/include/nvToolsExt.h:396

  --*
  --    * \brief Size of the structure.
  --    *
  --    * Needs to be set to the size in bytes of the event attribute
  --    * structure used to specify the event.
  --     

  --*
  --    * \brief Mode of initialization.
  --    *
  --    * The mode of initialization dictates the overall behavior and which
  --    * attributes in this struct will be used.
  --    *
  --    * Default Value is NVTX_INITIALIZATION_MODE_UNKNOWN = 0
  --    * \sa
  --    * ::nvtxInitializationMode_t
  --     

  --*
  --    * \brief Function pointer used for initialization if the mode requires
  --    *
  --    * The user has retrieved this function pointer from the tool library
  --    * and would like to use it to initialize.  The mode must be set to a
  --    * NVTX_INITIALIZATION_MODE_CALLBACK_V# for this to be used.  The mode
  --    * will dictate the expectations for this member.  The function signature
  --    * will be cast from void(*)() to the appropriate signature for the mode.
  --    * the expected behavior of the function will also depend on the mode
  --    * beyond the simple function signature.
  --    *
  --    * Default Value is NVTX_INITIALIZATION_MODE_UNKNOWN which will either
  --    * initialize based on external properties or fail if not supported on
  --    * the given platform.
  --    * \sa
  --    * ::nvtxInitializationMode_t
  --     

   subtype nvtxInitializationAttributes_t is nvtxInitializationAttributes_v2;

  -- -------------------------------------------------------------------------  
  --* \brief Force initialization (optional on most platforms)
  --*
  --* Force NVTX library to initialize.  On some platform NVTX will implicit initialize
  --* upon the first function call into an NVTX API.
  --*
  --* \return Result codes are simplest to assume NVTX_SUCCESS or !NVTX_SUCCESS
  --*
  --* \param initAttrib - The initialization attribute structure
  --*
  --* \sa
  --* ::nvtxInitializationAttributes_t
  --*
  --* \version \NVTX_VERSION_2
  --* @{  

   function nvtxInitialize (initAttrib : access constant nvtxInitializationAttributes_t) return int;  -- /usr/local/cuda-8.0/include/nvToolsExt.h:467
   pragma Import (C, nvtxInitialize, "nvtxInitialize");

  --* @}  
  --* @}  
  --END defgroup 
  -- =========================================================================  
  --* \defgroup EVENT_ATTRIBUTES Event Attributes
  --* @{
  -- 

  --* ---------------------------------------------------------------------------
  --* Payload Types
  --* -------------------------------------------------------------------------  

   type nvtxPayloadType_t is 
     (NVTX_PAYLOAD_UNKNOWN,
      NVTX_PAYLOAD_TYPE_UNSIGNED_INT64,
      NVTX_PAYLOAD_TYPE_INT64,
      NVTX_PAYLOAD_TYPE_DOUBLE,
      NVTX_PAYLOAD_TYPE_UNSIGNED_INT32,
      NVTX_PAYLOAD_TYPE_INT32,
      NVTX_PAYLOAD_TYPE_FLOAT);
   pragma Convention (C, nvtxPayloadType_t);  -- /usr/local/cuda-8.0/include/nvToolsExt.h:481

  --*< Color payload is unused.  
  --*< A 64 bit unsigned integer value is used as payload.  
  --*< A 64 bit signed integer value is used as payload.  
  --*< A 64 bit floating point value is used as payload.  
  -- NVTX_VERSION_2  
  --*< A 32 bit floating point value is used as payload.  
  --*< A 32 bit floating point value is used as payload.  
  --*< A 32 bit floating point value is used as payload.  
  --* \brief Event Attribute Structure.
  -- * \anchor EVENT_ATTRIBUTE_STRUCTURE
  -- *
  -- * This structure is used to describe the attributes of an event. The layout of
  -- * the structure is defined by a specific version of the tools extension
  -- * library and can change between different versions of the Tools Extension
  -- * library.
  -- *
  -- * \par Initializing the Attributes
  -- *
  -- * The caller should always perform the following three tasks when using
  -- * attributes:
  -- * <ul>
  -- *    <li>Zero the structure
  -- *    <li>Set the version field
  -- *    <li>Set the size field
  -- * </ul>
  -- *
  -- * Zeroing the structure sets all the event attributes types and values
  -- * to the default value.
  -- *
  -- * The version and size field are used by the Tools Extension
  -- * implementation to handle multiple versions of the attributes structure.
  -- *
  -- * It is recommended that the caller use one of the following to methods
  -- * to initialize the event attributes structure:
  -- *
  -- * \par Method 1: Initializing nvtxEventAttributes for future compatibility
  -- * \code
  -- * nvtxEventAttributes_t eventAttrib = {0};
  -- * eventAttrib.version = NVTX_VERSION;
  -- * eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  -- * \endcode
  -- *
  -- * \par Method 2: Initializing nvtxEventAttributes for a specific version
  -- * \code
  -- * nvtxEventAttributes_t eventAttrib = {0};
  -- * eventAttrib.version = 1;
  -- * eventAttrib.size = (uint16_t)(sizeof(nvtxEventAttributes_v1));
  -- * \endcode
  -- *
  -- * If the caller uses Method 1 it is critical that the entire binary
  -- * layout of the structure be configured to 0 so that all fields
  -- * are initialized to the default value.
  -- *
  -- * The caller should either use both NVTX_VERSION and
  -- * NVTX_EVENT_ATTRIB_STRUCT_SIZE (Method 1) or use explicit values
  -- * and a versioned type (Method 2).  Using a mix of the two methods
  -- * will likely cause either source level incompatibility or binary
  -- * incompatibility in the future.
  -- *
  -- * \par Settings Attribute Types and Values
  -- *
  -- *
  -- * \par Example:
  -- * \code
  -- * // Initialize
  -- * nvtxEventAttributes_t eventAttrib = {0};
  -- * eventAttrib.version = NVTX_VERSION;
  -- * eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  -- *
  -- * // Configure the Attributes
  -- * eventAttrib.colorType = NVTX_COLOR_ARGB;
  -- * eventAttrib.color = 0xFF880000;
  -- * eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  -- * eventAttrib.message.ascii = "Example";
  -- * \endcode
  -- *
  -- * In the example the caller does not have to set the value of
  -- * \ref ::nvtxEventAttributes_v2::category or
  -- * \ref ::nvtxEventAttributes_v2::payload as these fields were set to
  -- * the default value by {0}.
  -- * \sa
  -- * ::nvtxDomainMarkEx
  -- * ::nvtxDomainRangeStartEx
  -- * ::nvtxDomainRangePushEx
  --  

  --*
  --    * \brief Version flag of the structure.
  --    *
  --    * Needs to be set to NVTX_VERSION to indicate the version of NVTX APIs
  --    * supported in this header file. This can optionally be overridden to
  --    * another version of the tools extension library.
  --     

   type nvtxEventAttributes_v2;
   type payload_t (discr : unsigned := 0) is record
      case discr is
         when 0 =>
            ullValue : aliased stdint_h.uint64_t;  -- /usr/local/cuda-8.0/include/nvToolsExt.h:637
         when 1 =>
            llValue : aliased stdint_h.int64_t;  -- /usr/local/cuda-8.0/include/nvToolsExt.h:638
         when 2 =>
            dValue : aliased double;  -- /usr/local/cuda-8.0/include/nvToolsExt.h:639
         when 3 =>
            uiValue : aliased stdint_h.uint32_t;  -- /usr/local/cuda-8.0/include/nvToolsExt.h:641
         when 4 =>
            iValue : aliased stdint_h.int32_t;  -- /usr/local/cuda-8.0/include/nvToolsExt.h:642
         when others =>
            fValue : aliased float;  -- /usr/local/cuda-8.0/include/nvToolsExt.h:643
      end case;
   end record;
   pragma Convention (C_Pass_By_Copy, payload_t);
   pragma Unchecked_Union (payload_t);type nvtxEventAttributes_v2 is record
      version : aliased stdint_h.uint16_t;  -- /usr/local/cuda-8.0/include/nvToolsExt.h:579
      size : aliased stdint_h.uint16_t;  -- /usr/local/cuda-8.0/include/nvToolsExt.h:587
      category : aliased stdint_h.uint32_t;  -- /usr/local/cuda-8.0/include/nvToolsExt.h:600
      colorType : aliased stdint_h.int32_t;  -- /usr/local/cuda-8.0/include/nvToolsExt.h:609
      color : aliased stdint_h.uint32_t;  -- /usr/local/cuda-8.0/include/nvToolsExt.h:615
      payloadType : aliased stdint_h.int32_t;  -- /usr/local/cuda-8.0/include/nvToolsExt.h:625
      reserved0 : aliased stdint_h.int32_t;  -- /usr/local/cuda-8.0/include/nvToolsExt.h:627
      payload : aliased payload_t;  -- /usr/local/cuda-8.0/include/nvToolsExt.h:644
      messageType : aliased stdint_h.int32_t;  -- /usr/local/cuda-8.0/include/nvToolsExt.h:653
      message : aliased nvtxMessageValue_t;  -- /usr/local/cuda-8.0/include/nvToolsExt.h:659
   end record;
   pragma Convention (C_Pass_By_Copy, nvtxEventAttributes_v2);  -- /usr/local/cuda-8.0/include/nvToolsExt.h:570

  --*
  --    * \brief Size of the structure.
  --    *
  --    * Needs to be set to the size in bytes of the event attribute
  --    * structure used to specify the event.
  --     

  --*
  --     * \brief ID of the category the event is assigned to.
  --     *
  --     * A category is a user-controlled ID that can be used to group
  --     * events.  The tool may use category IDs to improve filtering or
  --     * enable grouping of events in the same category. The functions
  --     * \ref ::nvtxNameCategoryA or \ref ::nvtxNameCategoryW can be used
  --     * to name a category.
  --     *
  --     * Default Value is 0
  --      

  --* \brief Color type specified in this attribute structure.
  --     *
  --     * Defines the color format of the attribute structure's \ref COLOR_FIELD
  --     * "color" field.
  --     *
  --     * Default Value is NVTX_COLOR_UNKNOWN
  --      

  -- nvtxColorType_t  
  --* \brief Color assigned to this event. \anchor COLOR_FIELD
  --     *
  --     * The color that the tool should use to visualize the event.
  --      

  --*
  --     * \brief Payload type specified in this attribute structure.
  --     *
  --     * Defines the payload format of the attribute structure's \ref PAYLOAD_FIELD
  --     * "payload" field.
  --     *
  --     * Default Value is NVTX_PAYLOAD_UNKNOWN
  --      

  -- nvtxPayloadType_t  
  --*
  --     * \brief Payload assigned to this event. \anchor PAYLOAD_FIELD
  --     *
  --     * A numerical value that can be used to annotate an event. The tool could
  --     * use the payload data to reconstruct graphs and diagrams.
  --      

  -- NVTX_VERSION_2  
  --* \brief Message type specified in this attribute structure.
  --     *
  --     * Defines the message format of the attribute structure's \ref MESSAGE_FIELD
  --     * "message" field.
  --     *
  --     * Default Value is NVTX_MESSAGE_UNKNOWN
  --      

  -- nvtxMessageType_t  
  --* \brief Message assigned to this attribute structure. \anchor MESSAGE_FIELD
  --     *
  --     * The text message that is attached to an event.
  --      

   subtype nvtxEventAttributes_t is nvtxEventAttributes_v2;

  --* @}  
  --END defgroup 
  -- =========================================================================  
  --* \defgroup MARKERS_AND_RANGES Markers and Ranges
  -- *
  -- * See \ref MARKERS_AND_RANGES for more details
  -- *
  -- * @{
  --  

  --* \name Marker  
  -- -------------------------------------------------------------------------  
  --* \brief Marks an instantaneous event in the application.
  --*
  --* A marker can contain a text message or specify additional information
  --* using the event attributes structure.  These attributes include a text
  --* message, color, category, and a payload. Each of the attributes is optional
  --* and can only be sent out using the \ref nvtxDomainMarkEx function.
  --*
  --* nvtxDomainMarkEx(NULL, event) is equivalent to calling
  --* nvtxMarkEx(event).
  --*
  --* \param domain    - The domain of scoping the category.
  --* \param eventAttrib - The event attribute structure defining the marker's
  --* attribute types and attribute values.
  --*
  --* \sa
  --* ::nvtxMarkEx
  --*
  --* \version \NVTX_VERSION_2
  --* @{  

   procedure nvtxDomainMarkEx (domain : nvtxDomainHandle_t; eventAttrib : access constant nvtxEventAttributes_t);  -- /usr/local/cuda-8.0/include/nvToolsExt.h:696
   pragma Import (C, nvtxDomainMarkEx, "nvtxDomainMarkEx");

  --* @}  
  -- -------------------------------------------------------------------------  
  --* \brief Marks an instantaneous event in the application.
  -- *
  -- * A marker can contain a text message or specify additional information
  -- * using the event attributes structure.  These attributes include a text
  -- * message, color, category, and a payload. Each of the attributes is optional
  -- * and can only be sent out using the \ref nvtxMarkEx function.
  -- * If \ref nvtxMarkA or \ref nvtxMarkW are used to specify the marker
  -- * or if an attribute is unspecified then a default value will be used.
  -- *
  -- * \param eventAttrib - The event attribute structure defining the marker's
  -- * attribute types and attribute values.
  -- *
  -- * \par Example:
  -- * \code
  -- * // zero the structure
  -- * nvtxEventAttributes_t eventAttrib = {0};
  -- * // set the version and the size information
  -- * eventAttrib.version = NVTX_VERSION;
  -- * eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  -- * // configure the attributes.  0 is the default for all attributes.
  -- * eventAttrib.colorType = NVTX_COLOR_ARGB;
  -- * eventAttrib.color = 0xFF880000;
  -- * eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  -- * eventAttrib.message.ascii = "Example nvtxMarkEx";
  -- * nvtxMarkEx(&eventAttrib);
  -- * \endcode
  -- *
  -- * \sa
  -- * ::nvtxDomainMarkEx
  -- *
  -- * \version \NVTX_VERSION_1
  -- * @{  

   procedure nvtxMarkEx (eventAttrib : access constant nvtxEventAttributes_t);  -- /usr/local/cuda-8.0/include/nvToolsExt.h:732
   pragma Import (C, nvtxMarkEx, "nvtxMarkEx");

  --* @}  
  -- -------------------------------------------------------------------------  
  --* \brief Marks an instantaneous event in the application.
  -- *
  -- * A marker created using \ref nvtxMarkA or \ref nvtxMarkW contains only a
  -- * text message.
  -- *
  -- * \param message     - The message associated to this marker event.
  -- *
  -- * \par Example:
  -- * \code
  -- * nvtxMarkA("Example nvtxMarkA");
  -- * nvtxMarkW(L"Example nvtxMarkW");
  -- * \endcode
  -- *
  -- * \sa
  -- * ::nvtxDomainMarkEx
  -- * ::nvtxMarkEx
  -- *
  -- * \version \NVTX_VERSION_0
  -- * @{  

   procedure nvtxMarkA (message : Interfaces.C.Strings.chars_ptr);  -- /usr/local/cuda-8.0/include/nvToolsExt.h:755
   pragma Import (C, nvtxMarkA, "nvtxMarkA");

   procedure nvtxMarkW (message : access wchar_t);  -- /usr/local/cuda-8.0/include/nvToolsExt.h:756
   pragma Import (C, nvtxMarkW, "nvtxMarkW");

  --* @}  
  --* \name Process Ranges  
  -- -------------------------------------------------------------------------  
  --* \brief Starts a process range in a domain.
  --*
  --* \param domain    - The domain of scoping the category.
  --* \param eventAttrib - The event attribute structure defining the range's
  --* attribute types and attribute values.
  --*
  --* \return The unique ID used to correlate a pair of Start and End events.
  --*
  --* \remarks Ranges defined by Start/End can overlap.
  --*
  --* \par Example:
  --* \code
  --* nvtxDomainHandle_t domain = nvtxDomainCreateA("my domain");
  --* nvtxEventAttributes_t eventAttrib = {0};
  --* eventAttrib.version = NVTX_VERSION;
  --* eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  --* eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  --* eventAttrib.message.ascii = "my range";
  --* nvtxRangeId_t rangeId = nvtxDomainRangeStartEx(&eventAttrib);
  --* // ...
  --* nvtxDomainRangeEnd(rangeId);
  --* \endcode
  --*
  --* \sa
  --* ::nvtxDomainRangeEnd
  --*
  --* \version \NVTX_VERSION_2
  --* @{  

   function nvtxDomainRangeStartEx (domain : nvtxDomainHandle_t; eventAttrib : access constant nvtxEventAttributes_t) return nvtxRangeId_t;  -- /usr/local/cuda-8.0/include/nvToolsExt.h:791
   pragma Import (C, nvtxDomainRangeStartEx, "nvtxDomainRangeStartEx");

  --* @}  
  -- -------------------------------------------------------------------------  
  --* \brief Starts a process range.
  -- *
  -- * \param eventAttrib - The event attribute structure defining the range's
  -- * attribute types and attribute values.
  -- *
  -- * \return The unique ID used to correlate a pair of Start and End events.
  -- *
  -- * \remarks Ranges defined by Start/End can overlap.
  -- *
  -- * \par Example:
  -- * \code
  -- * nvtxEventAttributes_t eventAttrib = {0};
  -- * eventAttrib.version = NVTX_VERSION;
  -- * eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  -- * eventAttrib.category = 3;
  -- * eventAttrib.colorType = NVTX_COLOR_ARGB;
  -- * eventAttrib.color = 0xFF0088FF;
  -- * eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  -- * eventAttrib.message.ascii = "Example Range";
  -- * nvtxRangeId_t rangeId = nvtxRangeStartEx(&eventAttrib);
  -- * // ...
  -- * nvtxRangeEnd(rangeId);
  -- * \endcode
  -- *
  -- * \sa
  -- * ::nvtxRangeEnd
  -- * ::nvtxDomainRangeStartEx
  -- *
  -- * \version \NVTX_VERSION_1
  -- * @{  

   function nvtxRangeStartEx (eventAttrib : access constant nvtxEventAttributes_t) return nvtxRangeId_t;  -- /usr/local/cuda-8.0/include/nvToolsExt.h:825
   pragma Import (C, nvtxRangeStartEx, "nvtxRangeStartEx");

  --* @}  
  -- -------------------------------------------------------------------------  
  --* \brief Starts a process range.
  -- *
  -- * \param message     - The event message associated to this range event.
  -- *
  -- * \return The unique ID used to correlate a pair of Start and End events.
  -- *
  -- * \remarks Ranges defined by Start/End can overlap.
  -- *
  -- * \par Example:
  -- * \code
  -- * nvtxRangeId_t r1 = nvtxRangeStartA("Range 1");
  -- * nvtxRangeId_t r2 = nvtxRangeStartW(L"Range 2");
  -- * nvtxRangeEnd(r1);
  -- * nvtxRangeEnd(r2);
  -- * \endcode
  -- *
  -- * \sa
  -- * ::nvtxRangeEnd
  -- * ::nvtxRangeStartEx
  -- * ::nvtxDomainRangeStartEx
  -- *
  -- * \version \NVTX_VERSION_0
  -- * @{  

   function nvtxRangeStartA (message : Interfaces.C.Strings.chars_ptr) return nvtxRangeId_t;  -- /usr/local/cuda-8.0/include/nvToolsExt.h:852
   pragma Import (C, nvtxRangeStartA, "nvtxRangeStartA");

   function nvtxRangeStartW (message : access wchar_t) return nvtxRangeId_t;  -- /usr/local/cuda-8.0/include/nvToolsExt.h:853
   pragma Import (C, nvtxRangeStartW, "nvtxRangeStartW");

  --* @}  
  -- -------------------------------------------------------------------------  
  --* \brief Ends a process range.
  --*
  --* \param domain - The domain 
  --* \param id - The correlation ID returned from a nvtxRangeStart call.
  --*
  --* \remarks This function is offered completeness but is an alias for ::nvtxRangeEnd. 
  --* It does not need a domain param since that is associated iwth the range ID at ::nvtxDomainRangeStartEx
  --*
  --* \par Example:
  --* \code
  --* nvtxDomainHandle_t domain = nvtxDomainCreateA("my domain");
  --* nvtxEventAttributes_t eventAttrib = {0};
  --* eventAttrib.version = NVTX_VERSION;
  --* eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  --* eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  --* eventAttrib.message.ascii = "my range";
  --* nvtxRangeId_t rangeId = nvtxDomainRangeStartEx(&eventAttrib);
  --* // ...
  --* nvtxDomainRangeEnd(rangeId);
  --* \endcode
  --*
  --* \sa
  --* ::nvtxDomainRangeStartEx
  --*
  --* \version \NVTX_VERSION_2
  --* @{  

   procedure nvtxDomainRangeEnd (domain : nvtxDomainHandle_t; id : nvtxRangeId_t);  -- /usr/local/cuda-8.0/include/nvToolsExt.h:883
   pragma Import (C, nvtxDomainRangeEnd, "nvtxDomainRangeEnd");

  --* @}  
  -- -------------------------------------------------------------------------  
  --* \brief Ends a process range.
  -- *
  -- * \param id - The correlation ID returned from an nvtxRangeStart call.
  -- *
  -- * \sa
  -- * ::nvtxDomainRangeStartEx
  -- * ::nvtxRangeStartEx
  -- * ::nvtxRangeStartA
  -- * ::nvtxRangeStartW
  -- *
  -- * \version \NVTX_VERSION_0
  -- * @{  

   procedure nvtxRangeEnd (id : nvtxRangeId_t);  -- /usr/local/cuda-8.0/include/nvToolsExt.h:899
   pragma Import (C, nvtxRangeEnd, "nvtxRangeEnd");

  --* @}  
  --* \name Thread Ranges  
  -- -------------------------------------------------------------------------  
  --* \brief Starts a nested thread range.
  --*
  --* \param domain    - The domain of scoping.
  --* \param eventAttrib - The event attribute structure defining the range's
  --* attribute types and attribute values.
  --*
  --* \return The 0 based level of range being started. This value is scoped to the domain.
  --* If an error occurs, a negative value is returned.
  --*
  --* \par Example:
  --* \code
  --* nvtxDomainHandle_t domain = nvtxDomainCreateA("example domain");
  --* nvtxEventAttributes_t eventAttrib = {0};
  --* eventAttrib.version = NVTX_VERSION;
  --* eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  --* eventAttrib.colorType = NVTX_COLOR_ARGB;
  --* eventAttrib.color = 0xFFFF0000;
  --* eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  --* eventAttrib.message.ascii = "Level 0";
  --* nvtxDomainRangePushEx(domain, &eventAttrib);
  --*
  --* // Re-use eventAttrib
  --* eventAttrib.messageType = NVTX_MESSAGE_TYPE_UNICODE;
  --* eventAttrib.message.unicode = L"Level 1";
  --* nvtxDomainRangePushEx(domain, &eventAttrib);
  --*
  --* nvtxDomainRangePop(domain); //level 1
  --* nvtxDomainRangePop(domain); //level 0
  --* \endcode
  --*
  --* \sa
  --* ::nvtxDomainRangePop
  --*
  --* \version \NVTX_VERSION_2
  --* @{  

   function nvtxDomainRangePushEx (domain : nvtxDomainHandle_t; eventAttrib : access constant nvtxEventAttributes_t) return int;  -- /usr/local/cuda-8.0/include/nvToolsExt.h:940
   pragma Import (C, nvtxDomainRangePushEx, "nvtxDomainRangePushEx");

  --* @}  
  -- -------------------------------------------------------------------------  
  --* \brief Starts a nested thread range.
  -- *
  -- * \param eventAttrib - The event attribute structure defining the range's
  -- * attribute types and attribute values.
  -- *
  -- * \return The 0 based level of range being started. This level is per domain.
  -- * If an error occurs a negative value is returned.
  -- *
  -- * \par Example:
  -- * \code
  -- * nvtxEventAttributes_t eventAttrib = {0};
  -- * eventAttrib.version = NVTX_VERSION;
  -- * eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  -- * eventAttrib.colorType = NVTX_COLOR_ARGB;
  -- * eventAttrib.color = 0xFFFF0000;
  -- * eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  -- * eventAttrib.message.ascii = "Level 0";
  -- * nvtxRangePushEx(&eventAttrib);
  -- *
  -- * // Re-use eventAttrib
  -- * eventAttrib.messageType = NVTX_MESSAGE_TYPE_UNICODE;
  -- * eventAttrib.message.unicode = L"Level 1";
  -- * nvtxRangePushEx(&eventAttrib);
  -- *
  -- * nvtxRangePop();
  -- * nvtxRangePop();
  -- * \endcode
  -- *
  -- * \sa
  -- * ::nvtxDomainRangePushEx
  -- * ::nvtxRangePop
  -- *
  -- * \version \NVTX_VERSION_1
  -- * @{  

   function nvtxRangePushEx (eventAttrib : access constant nvtxEventAttributes_t) return int;  -- /usr/local/cuda-8.0/include/nvToolsExt.h:978
   pragma Import (C, nvtxRangePushEx, "nvtxRangePushEx");

  --* @}  
  -- -------------------------------------------------------------------------  
  --* \brief Starts a nested thread range.
  -- *
  -- * \param message     - The event message associated to this range event.
  -- *
  -- * \return The 0 based level of range being started.  If an error occurs a
  -- * negative value is returned.
  -- *
  -- * \par Example:
  -- * \code
  -- * nvtxRangePushA("Level 0");
  -- * nvtxRangePushW(L"Level 1");
  -- * nvtxRangePop();
  -- * nvtxRangePop();
  -- * \endcode
  -- *
  -- * \sa
  -- * ::nvtxDomainRangePushEx
  -- * ::nvtxRangePop
  -- *
  -- * \version \NVTX_VERSION_0
  -- * @{  

   function nvtxRangePushA (message : Interfaces.C.Strings.chars_ptr) return int;  -- /usr/local/cuda-8.0/include/nvToolsExt.h:1003
   pragma Import (C, nvtxRangePushA, "nvtxRangePushA");

   function nvtxRangePushW (message : access wchar_t) return int;  -- /usr/local/cuda-8.0/include/nvToolsExt.h:1004
   pragma Import (C, nvtxRangePushW, "nvtxRangePushW");

  --* @}  
  -- -------------------------------------------------------------------------  
  --* \brief Ends a nested thread range.
  --*
  --* \return The level of the range being ended. If an error occurs a negative
  --* value is returned on the current thread.
  --*
  --* \par Example:
  --* \code
  --* nvtxDomainHandle_t domain = nvtxDomainCreate("example library");
  --* nvtxDomainRangePushA(domain, "Level 0");
  --* nvtxDomainRangePushW(domain, L"Level 1");
  --* nvtxDomainRangePop(domain);
  --* nvtxDomainRangePop(domain);
  --* \endcode
  --*
  --* \sa
  --* ::nvtxRangePushEx
  --* ::nvtxRangePushA
  --* ::nvtxRangePushW
  --*
  --* \version \NVTX_VERSION_2
  --* @{  

   function nvtxDomainRangePop (domain : nvtxDomainHandle_t) return int;  -- /usr/local/cuda-8.0/include/nvToolsExt.h:1030
   pragma Import (C, nvtxDomainRangePop, "nvtxDomainRangePop");

  --* @}  
  -- -------------------------------------------------------------------------  
  --* \brief Ends a nested thread range.
  -- *
  -- * \return The level of the range being ended. If an error occurs a negative
  -- * value is returned on the current thread.
  -- *
  -- * \par Example:
  -- * \code
  -- * nvtxRangePushA("Level 0");
  -- * nvtxRangePushW(L"Level 1");
  -- * nvtxRangePop();
  -- * nvtxRangePop();
  -- * \endcode
  -- *
  -- * \sa
  -- * ::nvtxRangePushEx
  -- * ::nvtxRangePushA
  -- * ::nvtxRangePushW
  -- *
  -- * \version \NVTX_VERSION_0
  -- * @{  

   function nvtxRangePop return int;  -- /usr/local/cuda-8.0/include/nvToolsExt.h:1054
   pragma Import (C, nvtxRangePop, "nvtxRangePop");

  --* @}  
  --* @}  
  --END defgroup 
  -- =========================================================================  
  --* \defgroup RESOURCE_NAMING Resource Naming
  -- *
  -- * See \ref RESOURCE_NAMING for more details
  -- *
  -- * @{
  --  

  --  -------------------------------------------------------------------------  
  --* \name Functions for Generic Resource Naming 
  --  -------------------------------------------------------------------------  
  --  -------------------------------------------------------------------------  
  --* \cond SHOW_HIDDEN
  --* \brief Resource typing helpers.  
  --*
  --* Classes are used to make it easy to create a series of resource types 
  --* per API without collisions 
  -- 

  --* \endcond  
  -- -------------------------------------------------------------------------  
  --* \brief Generic resource type for when a resource class is not available.
  --*
  --* \sa
  --* ::nvtxDomainResourceCreate
  --*
  --* \version \NVTX_VERSION_2
  -- 

   subtype nvtxResourceGenericType_t is unsigned;
   NVTX_RESOURCE_TYPE_UNKNOWN : constant nvtxResourceGenericType_t := 0;
   NVTX_RESOURCE_TYPE_GENERIC_POINTER : constant nvtxResourceGenericType_t := 65537;
   NVTX_RESOURCE_TYPE_GENERIC_HANDLE : constant nvtxResourceGenericType_t := 65538;
   NVTX_RESOURCE_TYPE_GENERIC_THREAD_NATIVE : constant nvtxResourceGenericType_t := 65539;
   NVTX_RESOURCE_TYPE_GENERIC_THREAD_POSIX : constant nvtxResourceGenericType_t := 65540;  -- /usr/local/cuda-8.0/include/nvToolsExt.h:1091

  --*< Generic pointer assumed to have no collisions with other pointers.  
  --*< Generic handle assumed to have no collisions with other handles.  
  --*< OS native thread identifier.  
  --*< POSIX pthread identifier.  
  --* \brief Resource Attribute Structure.
  --* \anchor RESOURCE_ATTRIBUTE_STRUCTURE
  --*
  --* This structure is used to describe the attributes of a resource. The layout of
  --* the structure is defined by a specific version of the tools extension
  --* library and can change between different versions of the Tools Extension
  --* library.
  --*
  --* \par Initializing the Attributes
  --*
  --* The caller should always perform the following three tasks when using
  --* attributes:
  --* <ul>
  --*    <li>Zero the structure
  --*    <li>Set the version field
  --*    <li>Set the size field
  --* </ul>
  --*
  --* Zeroing the structure sets all the resource attributes types and values
  --* to the default value.
  --*
  --* The version and size field are used by the Tools Extension
  --* implementation to handle multiple versions of the attributes structure.
  --*
  --* It is recommended that the caller use one of the following to methods
  --* to initialize the event attributes structure:
  --*
  --* \par Method 1: Initializing nvtxEventAttributes for future compatibility
  --* \code
  --* nvtxResourceAttributes_t attribs = {0};
  --* attribs.version = NVTX_VERSION;
  --* attribs.size = NVTX_RESOURCE_ATTRIB_STRUCT_SIZE;
  --* \endcode
  --*
  --* \par Method 2: Initializing nvtxEventAttributes for a specific version
  --* \code
  --* nvtxResourceAttributes_v0 attribs = {0};
  --* attribs.version = 2;
  --* attribs.size = (uint16_t)(sizeof(nvtxResourceAttributes_v0));
  --* \endcode
  --*
  --* If the caller uses Method 1 it is critical that the entire binary
  --* layout of the structure be configured to 0 so that all fields
  --* are initialized to the default value.
  --*
  --* The caller should either use both NVTX_VERSION and
  --* NVTX_RESOURCE_ATTRIB_STRUCT_SIZE (Method 1) or use explicit values
  --* and a versioned type (Method 2).  Using a mix of the two methods
  --* will likely cause either source level incompatibility or binary
  --* incompatibility in the future.
  --*
  --* \par Settings Attribute Types and Values
  --*
  --*
  --* \par Example:
  --* \code
  --* nvtxDomainHandle_t domain = nvtxDomainCreateA("example domain");
  --*
  --* // Initialize
  --* nvtxResourceAttributes_t attribs = {0};
  --* attribs.version = NVTX_VERSION;
  --* attribs.size = NVTX_RESOURCE_ATTRIB_STRUCT_SIZE;
  --*
  --* // Configure the Attributes
  --* attribs.identifierType = NVTX_RESOURCE_TYPE_GENERIC_POINTER;
  --* attribs.identifier.pValue = (const void*)pMutex;
  --* attribs.messageType = NVTX_MESSAGE_TYPE_ASCII;
  --* attribs.message.ascii = "Single thread access to database.";
  --*
  --* nvtxResourceHandle_t handle = nvtxDomainResourceCreate(domain, attribs);
  --* \endcode
  --*
  --* \sa
  --* ::nvtxDomainResourceCreate
  -- 

  --*
  --    * \brief Version flag of the structure.
  --    *
  --    * Needs to be set to NVTX_VERSION to indicate the version of NVTX APIs
  --    * supported in this header file. This can optionally be overridden to
  --    * another version of the tools extension library.
  --     

   type nvtxResourceAttributes_v0;
   type identifier_t (discr : unsigned := 0) is record
      case discr is
         when 0 =>
            pValue : System.Address;  -- /usr/local/cuda-8.0/include/nvToolsExt.h:1215
         when others =>
            ullValue : aliased stdint_h.uint64_t;  -- /usr/local/cuda-8.0/include/nvToolsExt.h:1216
      end case;
   end record;
   pragma Convention (C_Pass_By_Copy, identifier_t);
   pragma Unchecked_Union (identifier_t);type nvtxResourceAttributes_v0 is record
      version : aliased stdint_h.uint16_t;  -- /usr/local/cuda-8.0/include/nvToolsExt.h:1186
      size : aliased stdint_h.uint16_t;  -- /usr/local/cuda-8.0/include/nvToolsExt.h:1194
      identifierType : aliased stdint_h.int32_t;  -- /usr/local/cuda-8.0/include/nvToolsExt.h:1204
      identifier : aliased identifier_t;  -- /usr/local/cuda-8.0/include/nvToolsExt.h:1217
      messageType : aliased stdint_h.int32_t;  -- /usr/local/cuda-8.0/include/nvToolsExt.h:1226
      message : aliased nvtxMessageValue_t;  -- /usr/local/cuda-8.0/include/nvToolsExt.h:1232
   end record;
   pragma Convention (C_Pass_By_Copy, nvtxResourceAttributes_v0);  -- /usr/local/cuda-8.0/include/nvToolsExt.h:1177

  --*
  --    * \brief Size of the structure.
  --    *
  --    * Needs to be set to the size in bytes of this attribute
  --    * structure.
  --     

  --*
  --    * \brief Identifier type specifies how to interpret the identifier field
  --    *
  --    * Defines the identifier format of the attribute structure's \ref RESOURCE_IDENTIFIER_FIELD
  --    * "identifier" field.
  --    *
  --    * Default Value is NVTX_RESOURCE_TYPE_UNKNOWN
  --     

  -- values from enums following the pattern nvtxResource[name]Type_t  
  --*
  --    * \brief Identifier for the resource. 
  --    * \anchor RESOURCE_IDENTIFIER_FIELD
  --    *
  --    * An identifier may be a pointer or a handle to an OS or middleware API object.
  --    * The resource type will assist in avoiding collisions where handles values may collide.
  --     

  --* \brief Message type specified in this attribute structure.
  --    *
  --    * Defines the message format of the attribute structure's \ref RESOURCE_MESSAGE_FIELD
  --    * "message" field.
  --    *
  --    * Default Value is NVTX_MESSAGE_UNKNOWN
  --     

  -- nvtxMessageType_t  
  --* \brief Message assigned to this attribute structure. \anchor RESOURCE_MESSAGE_FIELD
  --    *
  --    * The text message that is attached to a resource.
  --     

   subtype nvtxResourceAttributes_t is nvtxResourceAttributes_v0;

  -- \cond SHOW_HIDDEN 
  --* \version \NVTX_VERSION_2
  -- 

   --  skipped empty struct nvtxResourceHandle

   type nvtxResourceHandle_t is new System.Address;  -- /usr/local/cuda-8.0/include/nvToolsExt.h:1242

  --* \endcond  
  -- -------------------------------------------------------------------------  
  --* \brief Create a resource object to track and associate data with OS and middleware objects
  --*
  --* Allows users to associate an API handle or pointer with a user-provided name.
  --* 
  --*
  --* \param domain - Domain to own the resource object
  --* \param attribs - Attributes to be associated with the resource
  --*
  --* \return A handle that represents the newly created resource object.
  --*
  --* \par Example:
  --* \code
  --* nvtxDomainHandle_t domain = nvtxDomainCreateA("example domain");
  --* nvtxResourceAttributes_t attribs = {0};
  --* attribs.version = NVTX_VERSION;
  --* attribs.size = NVTX_RESOURCE_ATTRIB_STRUCT_SIZE;
  --* attribs.identifierType = NVTX_RESOURCE_TYPE_GENERIC_POINTER;
  --* attribs.identifier.pValue = (const void*)pMutex;
  --* attribs.messageType = NVTX_MESSAGE_TYPE_ASCII;
  --* attribs.message.ascii = "Single thread access to database.";
  --* nvtxResourceHandle_t handle = nvtxDomainResourceCreate(domain, attribs);
  --* \endcode
  --*
  --* \sa
  --* ::nvtxResourceAttributes_t
  --* ::nvtxDomainResourceDestroy
  --*
  --* \version \NVTX_VERSION_2
  --* @{  

   function nvtxDomainResourceCreate (domain : nvtxDomainHandle_t; attribs : access nvtxResourceAttributes_t) return nvtxResourceHandle_t;  -- /usr/local/cuda-8.0/include/nvToolsExt.h:1277
   pragma Import (C, nvtxDomainResourceCreate, "nvtxDomainResourceCreate");

  --* @}  
  -- -------------------------------------------------------------------------  
  --* \brief Destroy a resource object to track and associate data with OS and middleware objects
  --*
  --* Allows users to associate an API handle or pointer with a user-provided name.
  --*
  --* \param resource - Handle to the resource in which to operate.
  --*
  --* \par Example:
  --* \code
  --* nvtxDomainHandle_t domain = nvtxDomainCreateA("example domain");
  --* nvtxResourceAttributes_t attribs = {0};
  --* attribs.version = NVTX_VERSION;
  --* attribs.size = NVTX_RESOURCE_ATTRIB_STRUCT_SIZE;
  --* attribs.identifierType = NVTX_RESOURCE_TYPE_GENERIC_POINTER;
  --* attribs.identifier.pValue = (const void*)pMutex;
  --* attribs.messageType = NVTX_MESSAGE_TYPE_ASCII;
  --* attribs.message.ascii = "Single thread access to database.";
  --* nvtxResourceHandle_t handle = nvtxDomainResourceCreate(domain, attribs);
  --* nvtxDomainResourceDestroy(handle);
  --* \endcode
  --*
  --* \sa
  --* ::nvtxDomainResourceCreate
  --*
  --* \version \NVTX_VERSION_2
  --* @{  

   procedure nvtxDomainResourceDestroy (resource : nvtxResourceHandle_t);  -- /usr/local/cuda-8.0/include/nvToolsExt.h:1306
   pragma Import (C, nvtxDomainResourceDestroy, "nvtxDomainResourceDestroy");

  --* @}  
  --* \name Functions for NVTX Category Naming 
  -- -------------------------------------------------------------------------  
  --*
  --* \brief Annotate an NVTX category used within a domain.
  --*
  --* Categories are used to group sets of events. Each category is identified
  --* through a unique ID and that ID is passed into any of the marker/range
  --* events to assign that event to a specific category. The nvtxDomainNameCategory
  --* function calls allow the user to assign a name to a category ID that is
  --* specific to the domain.
  --*
  --* nvtxDomainNameCategory(NULL, category, name) is equivalent to calling
  --* nvtxNameCategory(category, name).
  --*
  --* \param domain    - The domain of scoping the category.
  --* \param category  - The category ID to name.
  --* \param name      - The name of the category.
  --*
  --* \remarks The category names are tracked per domain.
  --*
  --* \par Example:
  --* \code
  --* nvtxDomainHandle_t domain = nvtxDomainCreateA("example");
  --* nvtxDomainNameCategoryA(domain, 1, "Memory Allocation");
  --* nvtxDomainNameCategoryW(domain, 2, L"Memory Transfer");
  --* \endcode
  --*
  --* \version \NVTX_VERSION_2
  --* @{  

   procedure nvtxDomainNameCategoryA
     (domain : nvtxDomainHandle_t;
      category : stdint_h.uint32_t;
      name : Interfaces.C.Strings.chars_ptr);  -- /usr/local/cuda-8.0/include/nvToolsExt.h:1340
   pragma Import (C, nvtxDomainNameCategoryA, "nvtxDomainNameCategoryA");

   procedure nvtxDomainNameCategoryW
     (domain : nvtxDomainHandle_t;
      category : stdint_h.uint32_t;
      name : access wchar_t);  -- /usr/local/cuda-8.0/include/nvToolsExt.h:1341
   pragma Import (C, nvtxDomainNameCategoryW, "nvtxDomainNameCategoryW");

  --* @}  
  --* \brief Annotate an NVTX category.
  -- *
  -- * Categories are used to group sets of events. Each category is identified
  -- * through a unique ID and that ID is passed into any of the marker/range
  -- * events to assign that event to a specific category. The nvtxNameCategory
  -- * function calls allow the user to assign a name to a category ID.
  -- *
  -- * \param category - The category ID to name.
  -- * \param name     - The name of the category.
  -- *
  -- * \remarks The category names are tracked per process.
  -- *
  -- * \par Example:
  -- * \code
  -- * nvtxNameCategory(1, "Memory Allocation");
  -- * nvtxNameCategory(2, "Memory Transfer");
  -- * nvtxNameCategory(3, "Memory Object Lifetime");
  -- * \endcode
  -- *
  -- * \version \NVTX_VERSION_1
  -- * @{  

   procedure nvtxNameCategoryA (category : stdint_h.uint32_t; name : Interfaces.C.Strings.chars_ptr);  -- /usr/local/cuda-8.0/include/nvToolsExt.h:1365
   pragma Import (C, nvtxNameCategoryA, "nvtxNameCategoryA");

   procedure nvtxNameCategoryW (category : stdint_h.uint32_t; name : access wchar_t);  -- /usr/local/cuda-8.0/include/nvToolsExt.h:1366
   pragma Import (C, nvtxNameCategoryW, "nvtxNameCategoryW");

  --* @}  
  --* \name Functions for OS Threads Naming 
  -- -------------------------------------------------------------------------  
  --* \brief Annotate an OS thread.
  -- *
  -- * Allows the user to name an active thread of the current process. If an
  -- * invalid thread ID is provided or a thread ID from a different process is
  -- * used the behavior of the tool is implementation dependent.
  -- *
  -- * The thread name is associated to the default domain.  To support domains 
  -- * use resource objects via ::nvtxDomainResourceCreate.
  -- *
  -- * \param threadId - The ID of the thread to name.
  -- * \param name     - The name of the thread.
  -- *
  -- * \par Example:
  -- * \code
  -- * nvtxNameOsThread(GetCurrentThreadId(), "MAIN_THREAD");
  -- * \endcode
  -- *
  -- * \version \NVTX_VERSION_1
  -- * @{  

   procedure nvtxNameOsThreadA (threadId : stdint_h.uint32_t; name : Interfaces.C.Strings.chars_ptr);  -- /usr/local/cuda-8.0/include/nvToolsExt.h:1391
   pragma Import (C, nvtxNameOsThreadA, "nvtxNameOsThreadA");

   procedure nvtxNameOsThreadW (threadId : stdint_h.uint32_t; name : access wchar_t);  -- /usr/local/cuda-8.0/include/nvToolsExt.h:1392
   pragma Import (C, nvtxNameOsThreadW, "nvtxNameOsThreadW");

  --* @}  
  --* @}  
  --END defgroup 
  -- =========================================================================  
  --* \defgroup STRING_REGISTRATION String Registration
  --*
  --* Registered strings are intended to increase performance by lowering instrumentation
  --* overhead.  String may be registered once and the handle may be passed in place of
  --* a string where an the APIs may allow.
  --*
  --* See \ref STRING_REGISTRATION for more details
  --*
  --* @{
  -- 

  -- -------------------------------------------------------------------------  
  --* \brief Register a string.
  --* Registers an immutable string with NVTX. Once registered the pointer used
  --* to register the domain name can be used in nvtxEventAttributes_t
  --* \ref MESSAGE_FIELD. This allows NVTX implementation to skip copying the
  --* contents of the message on each event invocation.
  --*
  --* String registration is an optimization. It is recommended to use string
  --* registration if the string will be passed to an event many times.
  --*
  --* String are not unregistered, except that by unregistering the entire domain
  --*
  --* \param domain  - Domain handle. If NULL then the global domain is used.
  --* \param string    - A unique pointer to a sequence of characters.
  --*
  --* \return A handle representing the registered string.
  --*
  --* \par Example:
  --* \code
  --* nvtxDomainCreateA("com.nvidia.nvtx.example");
  --* nvtxStringHandle_t message = nvtxDomainRegisterStringA(domain, "registered string");
  --* nvtxEventAttributes_t eventAttrib = {0};
  --* eventAttrib.version = NVTX_VERSION;
  --* eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  --* eventAttrib.messageType = NVTX_MESSAGE_TYPE_REGISTERED;
  --* eventAttrib.message.registered = message;
  --* \endcode
  --*
  --* \version \NVTX_VERSION_2
  --* @{  

   function nvtxDomainRegisterStringA (domain : nvtxDomainHandle_t; string : Interfaces.C.Strings.chars_ptr) return nvtxStringHandle_t;  -- /usr/local/cuda-8.0/include/nvToolsExt.h:1440
   pragma Import (C, nvtxDomainRegisterStringA, "nvtxDomainRegisterStringA");

   function nvtxDomainRegisterStringW (domain : nvtxDomainHandle_t; string : access wchar_t) return nvtxStringHandle_t;  -- /usr/local/cuda-8.0/include/nvToolsExt.h:1441
   pragma Import (C, nvtxDomainRegisterStringW, "nvtxDomainRegisterStringW");

  --* @}  
  --* @}  
  --END defgroup 
  -- =========================================================================  
  --* \defgroup DOMAINS Domains
  --*
  --* Domains are used to group events to a developer defined scope. Middleware
  --* vendors may also scope their own events to avoid collisions with the
  --* the application developer's events, so that the application developer may
  --* inspect both parts and easily differentiate or filter them.  By default
  --* all events are scoped to a global domain where NULL is provided or when
  --* using APIs provided b versions of NVTX below v2
  --*
  --* Domains are intended to be typically long lived objects with the intention
  --* of logically separating events of large modules from each other such as
  --* middleware libraries from each other and the main application.
  --*
  --* See \ref DOMAINS for more details
  --*
  --* @{
  -- 

  -- -------------------------------------------------------------------------  
  --* \brief Register a NVTX domain.
  --*
  --* Domains are used to scope annotations. All NVTX_VERSION_0 and NVTX_VERSION_1
  --* annotations are scoped to the global domain. The function nvtxDomainCreate
  --* creates a new named domain.
  --*
  --* Each domain maintains its own nvtxRangePush and nvtxRangePop stack.
  --*
  --* \param name - A unique string representing the domain.
  --*
  --* \return A handle representing the domain.
  --*
  --* \par Example:
  --* \code
  --* nvtxDomainHandle_t domain = nvtxDomainCreateA("com.nvidia.nvtx.example");
  --*
  --* nvtxMarkA("nvtxMarkA to global domain");
  --*
  --* nvtxEventAttributes_t eventAttrib1 = {0};
  --* eventAttrib1.version = NVTX_VERSION;
  --* eventAttrib1.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  --* eventAttrib1.message.ascii = "nvtxDomainMarkEx to global domain";
  --* nvtxDomainMarkEx(NULL, &eventAttrib1);
  --*
  --* nvtxEventAttributes_t eventAttrib2 = {0};
  --* eventAttrib2.version = NVTX_VERSION;
  --* eventAttrib2.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  --* eventAttrib2.message.ascii = "nvtxDomainMarkEx to com.nvidia.nvtx.example";
  --* nvtxDomainMarkEx(domain, &eventAttrib2);
  --* nvtxDomainDestroy(domain);
  --* \endcode
  --*
  --* \sa
  --* ::nvtxDomainDestroy
  --*
  --* \version \NVTX_VERSION_2
  --* @{  

   function nvtxDomainCreateA (name : Interfaces.C.Strings.chars_ptr) return nvtxDomainHandle_t;  -- /usr/local/cuda-8.0/include/nvToolsExt.h:1502
   pragma Import (C, nvtxDomainCreateA, "nvtxDomainCreateA");

   function nvtxDomainCreateW (name : access wchar_t) return nvtxDomainHandle_t;  -- /usr/local/cuda-8.0/include/nvToolsExt.h:1503
   pragma Import (C, nvtxDomainCreateW, "nvtxDomainCreateW");

  --* @}  
  -- -------------------------------------------------------------------------  
  --* \brief Unregister a NVTX domain.
  --*
  --* Unregisters the domain handle and frees all domain specific resources.
  --*
  --* \param domain    - the domain handle
  --*
  --* \par Example:
  --* \code
  --* nvtxDomainHandle_t domain = nvtxDomainCreateA("com.nvidia.nvtx.example");
  --* nvtxDomainDestroy(domain);
  --* \endcode
  --*
  --* \sa
  --* ::nvtxDomainCreateA
  --* ::nvtxDomainCreateW
  --*
  --* \version \NVTX_VERSION_2
  --* @{  

   procedure nvtxDomainDestroy (domain : nvtxDomainHandle_t);  -- /usr/local/cuda-8.0/include/nvToolsExt.h:1525
   pragma Import (C, nvtxDomainDestroy, "nvtxDomainDestroy");

  --* @}  
  --* @}  
  --END defgroup 
  -- =========================================================================  
  --* \cond SHOW_HIDDEN  
  -- NVTX_VERSION_2  
  -- NVTX_VERSION_2  
  --* \endcond  
end nvToolsExt_h;

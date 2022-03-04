pragma Ada_2005;
pragma Style_Checks (Off);

with Interfaces.C; use Interfaces.C;
with Interfaces.C.Extensions;
with System;
with library_types_h;
with driver_types_h;
with stddef_h;

package curand_h is

  -- Copyright 2010-2014 NVIDIA Corporation.  All rights reserved.
  --  *
  --  * NOTICE TO LICENSEE:
  --  *
  --  * The source code and/or documentation ("Licensed Deliverables") are
  --  * subject to NVIDIA intellectual property rights under U.S. and
  --  * international Copyright laws.
  --  *
  --  * The Licensed Deliverables contained herein are PROPRIETARY and
  --  * CONFIDENTIAL to NVIDIA and are being provided under the terms and
  --  * conditions of a form of NVIDIA software license agreement by and
  --  * between NVIDIA and Licensee ("License Agreement") or electronically
  --  * accepted by Licensee.  Notwithstanding any terms or conditions to
  --  * the contrary in the License Agreement, reproduction or disclosure
  --  * of the Licensed Deliverables to any third party without the express
  --  * written consent of NVIDIA is prohibited.
  --  *
  --  * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
  --  * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
  --  * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  THEY ARE
  --  * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
  --  * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
  --  * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
  --  * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
  --  * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
  --  * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
  --  * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
  --  * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
  --  * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
  --  * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
  --  * OF THESE LICENSED DELIVERABLES.
  --  *
  --  * U.S. Government End Users.  These Licensed Deliverables are a
  --  * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
  --  * 1995), consisting of "commercial computer software" and "commercial
  --  * computer software documentation" as such terms are used in 48
  --  * C.F.R. 12.212 (SEPT 1995) and are provided to the U.S. Government
  --  * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
  --  * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
  --  * U.S. Government End Users acquire the Licensed Deliverables with
  --  * only those rights set forth herein.
  --  *
  --  * Any use of the Licensed Deliverables in individual and commercial
  --  * software must include, in the user documentation and internal
  --  * comments to the code, the above Disclaimer and U.S. Government End
  --  * Users Notice.
  --   

  --*
  -- * \defgroup HOST Host API
  -- *
  -- * @{
  --  

  -- CURAND Host API datatypes  
  --*  
  -- * @{
  --  

  --*
  -- * CURAND function call status types 
  --  

   subtype curandStatus is unsigned;
   CURAND_STATUS_SUCCESS : constant curandStatus := 0;
   CURAND_STATUS_VERSION_MISMATCH : constant curandStatus := 100;
   CURAND_STATUS_NOT_INITIALIZED : constant curandStatus := 101;
   CURAND_STATUS_ALLOCATION_FAILED : constant curandStatus := 102;
   CURAND_STATUS_TYPE_ERROR : constant curandStatus := 103;
   CURAND_STATUS_OUT_OF_RANGE : constant curandStatus := 104;
   CURAND_STATUS_LENGTH_NOT_MULTIPLE : constant curandStatus := 105;
   CURAND_STATUS_DOUBLE_PRECISION_REQUIRED : constant curandStatus := 106;
   CURAND_STATUS_LAUNCH_FAILURE : constant curandStatus := 201;
   CURAND_STATUS_PREEXISTING_FAILURE : constant curandStatus := 202;
   CURAND_STATUS_INITIALIZATION_FAILED : constant curandStatus := 203;
   CURAND_STATUS_ARCH_MISMATCH : constant curandStatus := 204;
   CURAND_STATUS_INTERNAL_ERROR : constant curandStatus := 999;  -- /usr/local/cuda-8.0/include/curand.h:82

  --/< No errors
  --/< Header file and linked library version do not match
  --/< Generator not initialized
  --/< Memory allocation failed
  --/< Generator is wrong type
  --/< Argument out of range
  --/< Length requested is not a multple of dimension
  --/< GPU does not have double precision required by MRG32k3a
  --/< Kernel launch failure
  --/< Preexisting failure on library entry
  --/< Initialization of CUDA failed
  --/< Architecture mismatch, GPU does not support requested feature
  --/< Internal library error
  -- * CURAND function call status types
  -- 

  --* \cond UNHIDE_TYPEDEFS  
   subtype curandStatus_t is curandStatus;

  --* \endcond  
  --*
  -- * CURAND generator types
  --  

   subtype curandRngType is unsigned;
   CURAND_RNG_TEST : constant curandRngType := 0;
   CURAND_RNG_PSEUDO_DEFAULT : constant curandRngType := 100;
   CURAND_RNG_PSEUDO_XORWOW : constant curandRngType := 101;
   CURAND_RNG_PSEUDO_MRG32K3A : constant curandRngType := 121;
   CURAND_RNG_PSEUDO_MTGP32 : constant curandRngType := 141;
   CURAND_RNG_PSEUDO_MT19937 : constant curandRngType := 142;
   CURAND_RNG_PSEUDO_PHILOX4_32_10 : constant curandRngType := 161;
   CURAND_RNG_QUASI_DEFAULT : constant curandRngType := 200;
   CURAND_RNG_QUASI_SOBOL32 : constant curandRngType := 201;
   CURAND_RNG_QUASI_SCRAMBLED_SOBOL32 : constant curandRngType := 202;
   CURAND_RNG_QUASI_SOBOL64 : constant curandRngType := 203;
   CURAND_RNG_QUASI_SCRAMBLED_SOBOL64 : constant curandRngType := 204;  -- /usr/local/cuda-8.0/include/curand.h:108

  --/< Default pseudorandom generator
  --/< XORWOW pseudorandom generator
  --/< MRG32k3a pseudorandom generator
  --/< Mersenne Twister MTGP32 pseudorandom generator
  --/< Mersenne Twister MT19937 pseudorandom generator
  --/< PHILOX-4x32-10 pseudorandom generator
  --/< Default quasirandom generator
  --/< Sobol32 quasirandom generator
  --/< Scrambled Sobol32 quasirandom generator
  --/< Sobol64 quasirandom generator
  --/< Scrambled Sobol64 quasirandom generator
  -- * CURAND generator types
  --  

  --* \cond UNHIDE_TYPEDEFS  
   subtype curandRngType_t is curandRngType;

  --* \endcond  
  --*
  -- * CURAND ordering of results in memory
  --  

   subtype curandOrdering is unsigned;
   CURAND_ORDERING_PSEUDO_BEST : constant curandOrdering := 100;
   CURAND_ORDERING_PSEUDO_DEFAULT : constant curandOrdering := 101;
   CURAND_ORDERING_PSEUDO_SEEDED : constant curandOrdering := 102;
   CURAND_ORDERING_QUASI_DEFAULT : constant curandOrdering := 201;  -- /usr/local/cuda-8.0/include/curand.h:133

  --/< Best ordering for pseudorandom results
  --/< Specific default 4096 thread sequence for pseudorandom results
  --/< Specific seeding pattern for fast lower quality pseudorandom results
  --/< Specific n-dimensional ordering for quasirandom results
  -- * CURAND ordering of results in memory
  --  

  --* \cond UNHIDE_TYPEDEFS  
   subtype curandOrdering_t is curandOrdering;

  --* \endcond  
  --*
  -- * CURAND choice of direction vector set
  --  

   subtype curandDirectionVectorSet is unsigned;
   CURAND_DIRECTION_VECTORS_32_JOEKUO6 : constant curandDirectionVectorSet := 101;
   CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6 : constant curandDirectionVectorSet := 102;
   CURAND_DIRECTION_VECTORS_64_JOEKUO6 : constant curandDirectionVectorSet := 103;
   CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6 : constant curandDirectionVectorSet := 104;  -- /usr/local/cuda-8.0/include/curand.h:150

  --/< Specific set of 32-bit direction vectors generated from polynomials recommended by S. Joe and F. Y. Kuo, for up to 20,000 dimensions
  --/< Specific set of 32-bit direction vectors generated from polynomials recommended by S. Joe and F. Y. Kuo, for up to 20,000 dimensions, and scrambled
  --/< Specific set of 64-bit direction vectors generated from polynomials recommended by S. Joe and F. Y. Kuo, for up to 20,000 dimensions
  --/< Specific set of 64-bit direction vectors generated from polynomials recommended by S. Joe and F. Y. Kuo, for up to 20,000 dimensions, and scrambled
  -- * CURAND choice of direction vector set
  --  

  --* \cond UNHIDE_TYPEDEFS  
   subtype curandDirectionVectorSet_t is curandDirectionVectorSet;

  --* \endcond  
  --*
  -- * CURAND array of 32-bit direction vectors
  --  

  --* \cond UNHIDE_TYPEDEFS  
   type curandDirectionVectors32_t is array (0 .. 31) of aliased unsigned;  -- /usr/local/cuda-8.0/include/curand.h:168

  --* \endcond  
  --*
  -- * CURAND array of 64-bit direction vectors
  --  

  --* \cond UNHIDE_TYPEDEFS  
   type curandDirectionVectors64_t is array (0 .. 63) of aliased Extensions.unsigned_long_long;  -- /usr/local/cuda-8.0/include/curand.h:175

  --* \endcond * 
  --*
  -- * CURAND generator (opaque)
  --  

   --  skipped empty struct curandGenerator_st

  --*
  -- * CURAND generator
  --  

  --* \cond UNHIDE_TYPEDEFS  
   type curandGenerator_t is new System.Address;  -- /usr/local/cuda-8.0/include/curand.h:187

  --* \endcond  
  --*
  -- * CURAND distribution
  --  

  --* \cond UNHIDE_TYPEDEFS  
   subtype curandDistribution_st is double;  -- /usr/local/cuda-8.0/include/curand.h:194

   type curandDistribution_t is access all curandDistribution_st;  -- /usr/local/cuda-8.0/include/curand.h:195

   --  skipped empty struct curandDistributionShift_st

   type curandDistributionShift_t is new System.Address;  -- /usr/local/cuda-8.0/include/curand.h:196

  --* \endcond  
  --*
  -- * CURAND distribution M2
  --  

  --* \cond UNHIDE_TYPEDEFS  
   --  skipped empty struct curandDistributionM2Shift_st

   type curandDistributionM2Shift_t is new System.Address;  -- /usr/local/cuda-8.0/include/curand.h:202

   --  skipped empty struct curandHistogramM2_st

   type curandHistogramM2_t is new System.Address;  -- /usr/local/cuda-8.0/include/curand.h:203

   subtype curandHistogramM2K_st is unsigned;  -- /usr/local/cuda-8.0/include/curand.h:204

   type curandHistogramM2K_t is access all curandHistogramM2K_st;  -- /usr/local/cuda-8.0/include/curand.h:205

   subtype curandHistogramM2V_st is curandDistribution_st;  -- /usr/local/cuda-8.0/include/curand.h:206

   type curandHistogramM2V_t is access all curandHistogramM2V_st;  -- /usr/local/cuda-8.0/include/curand.h:207

   --  skipped empty struct curandDiscreteDistribution_st

   type curandDiscreteDistribution_t is new System.Address;  -- /usr/local/cuda-8.0/include/curand.h:209

  --* \endcond  
  -- * CURAND METHOD
  --  

  --* \cond UNHIDE_ENUMS  
   type curandMethod is 
     (CURAND_CHOOSE_BEST,
      CURAND_ITR,
      CURAND_KNUTH,
      CURAND_HITR,
      CURAND_M1,
      CURAND_M2,
      CURAND_BINARY_SEARCH,
      CURAND_DISCRETE_GAUSS,
      CURAND_REJECTION,
      CURAND_DEVICE_API,
      CURAND_FAST_REJECTION,
      CURAND_3RD,
      CURAND_DEFINITION,
      CURAND_POISSON);
   pragma Convention (C, curandMethod);  -- /usr/local/cuda-8.0/include/curand.h:216

  -- choose best depends on args
   subtype curandMethod_t is curandMethod;

  --* \endcond  
  --*
  -- * @}
  --  

  --*
  -- * \brief Create new random number generator.
  -- *
  -- * Creates a new random number generator of type \p rng_type
  -- * and returns it in \p *generator.
  -- *
  -- * Legal values for \p rng_type are:
  -- * - CURAND_RNG_PSEUDO_DEFAULT
  -- * - CURAND_RNG_PSEUDO_XORWOW 
  -- * - CURAND_RNG_PSEUDO_MRG32K3A
  -- * - CURAND_RNG_PSEUDO_MTGP32
  -- * - CURAND_RNG_PSEUDO_MT19937 
  -- * - CURAND_RNG_PSEUDO_PHILOX4_32_10
  -- * - CURAND_RNG_QUASI_DEFAULT
  -- * - CURAND_RNG_QUASI_SOBOL32
  -- * - CURAND_RNG_QUASI_SCRAMBLED_SOBOL32
  -- * - CURAND_RNG_QUASI_SOBOL64
  -- * - CURAND_RNG_QUASI_SCRAMBLED_SOBOL64
  -- * 
  -- * When \p rng_type is CURAND_RNG_PSEUDO_DEFAULT, the type chosen
  -- * is CURAND_RNG_PSEUDO_XORWOW.  \n
  -- * When \p rng_type is CURAND_RNG_QUASI_DEFAULT,
  -- * the type chosen is CURAND_RNG_QUASI_SOBOL32.
  -- * 
  -- * The default values for \p rng_type = CURAND_RNG_PSEUDO_XORWOW are:
  -- * - \p seed = 0
  -- * - \p offset = 0
  -- * - \p ordering = CURAND_ORDERING_PSEUDO_DEFAULT
  -- *
  -- * The default values for \p rng_type = CURAND_RNG_PSEUDO_MRG32K3A are:
  -- * - \p seed = 0
  -- * - \p offset = 0
  -- * - \p ordering = CURAND_ORDERING_PSEUDO_DEFAULT
  -- *
  -- * The default values for \p rng_type = CURAND_RNG_PSEUDO_MTGP32 are:
  -- * - \p seed = 0
  -- * - \p offset = 0
  -- * - \p ordering = CURAND_ORDERING_PSEUDO_DEFAULT
  -- *
  -- * The default values for \p rng_type = CURAND_RNG_PSEUDO_MT19937 are:
  -- * - \p seed = 0
  -- * - \p offset = 0
  -- * - \p ordering = CURAND_ORDERING_PSEUDO_DEFAULT
  -- *
  -- * * The default values for \p rng_type = CURAND_RNG_PSEUDO_PHILOX4_32_10 are:
  -- * - \p seed = 0
  -- * - \p offset = 0
  -- * - \p ordering = CURAND_ORDERING_PSEUDO_DEFAULT
  -- *
  -- * The default values for \p rng_type = CURAND_RNG_QUASI_SOBOL32 are:
  -- * - \p dimensions = 1
  -- * - \p offset = 0
  -- * - \p ordering = CURAND_ORDERING_QUASI_DEFAULT
  -- *
  -- * The default values for \p rng_type = CURAND_RNG_QUASI_SOBOL64 are:
  -- * - \p dimensions = 1
  -- * - \p offset = 0
  -- * - \p ordering = CURAND_ORDERING_QUASI_DEFAULT
  -- *
  -- * The default values for \p rng_type = CURAND_RNG_QUASI_SCRAMBBLED_SOBOL32 are:
  -- * - \p dimensions = 1
  -- * - \p offset = 0
  -- * - \p ordering = CURAND_ORDERING_QUASI_DEFAULT
  -- *
  -- * The default values for \p rng_type = CURAND_RNG_QUASI_SCRAMBLED_SOBOL64 are:
  -- * - \p dimensions = 1
  -- * - \p offset = 0
  -- * - \p ordering = CURAND_ORDERING_QUASI_DEFAULT
  -- *
  -- * \param generator - Pointer to generator
  -- * \param rng_type - Type of generator to create
  -- *
  -- * \return 
  -- * - CURAND_STATUS_ALLOCATION_FAILED, if memory could not be allocated \n
  -- * - CURAND_STATUS_INITIALIZATION_FAILED if there was a problem setting up the GPU \n
  -- * - CURAND_STATUS_VERSION_MISMATCH if the header file version does not match the 
  -- *   dynamically linked library version \n
  -- * - CURAND_STATUS_TYPE_ERROR if the value for \p rng_type is invalid \n
  -- * - CURAND_STATUS_SUCCESS if generator was created successfully \n
  -- * 
  --  

   function curandCreateGenerator (generator : System.Address; rng_type : curandRngType_t) return curandStatus_t;  -- /usr/local/cuda-8.0/include/curand.h:322
   pragma Import (C, curandCreateGenerator, "curandCreateGenerator");

  --*
  -- * \brief Create new host CPU random number generator.
  -- *
  -- * Creates a new host CPU random number generator of type \p rng_type
  -- * and returns it in \p *generator.
  -- *
  -- * Legal values for \p rng_type are:
  -- * - CURAND_RNG_PSEUDO_DEFAULT
  -- * - CURAND_RNG_PSEUDO_XORWOW 
  -- * - CURAND_RNG_PSEUDO_MRG32K3A
  -- * - CURAND_RNG_PSEUDO_MTGP32
  -- * - CURAND_RNG_PSEUDO_MT19937
  -- * - CURAND_RNG_PSEUDO_PHILOX4_32_10
  -- * - CURAND_RNG_QUASI_DEFAULT
  -- * - CURAND_RNG_QUASI_SOBOL32
  -- * 
  -- * When \p rng_type is CURAND_RNG_PSEUDO_DEFAULT, the type chosen
  -- * is CURAND_RNG_PSEUDO_XORWOW.  \n
  -- * When \p rng_type is CURAND_RNG_QUASI_DEFAULT,
  -- * the type chosen is CURAND_RNG_QUASI_SOBOL32.
  -- * 
  -- * The default values for \p rng_type = CURAND_RNG_PSEUDO_XORWOW are:
  -- * - \p seed = 0
  -- * - \p offset = 0
  -- * - \p ordering = CURAND_ORDERING_PSEUDO_DEFAULT
  -- *
  -- * The default values for \p rng_type = CURAND_RNG_PSEUDO_MRG32K3A are:
  -- * - \p seed = 0
  -- * - \p offset = 0
  -- * - \p ordering = CURAND_ORDERING_PSEUDO_DEFAULT
  -- *
  -- * The default values for \p rng_type = CURAND_RNG_PSEUDO_MTGP32 are:
  -- * - \p seed = 0
  -- * - \p offset = 0
  -- * - \p ordering = CURAND_ORDERING_PSEUDO_DEFAULT
  -- *
  -- * The default values for \p rng_type = CURAND_RNG_PSEUDO_MT19937 are:
  -- * - \p seed = 0
  -- * - \p offset = 0
  -- * - \p ordering = CURAND_ORDERING_PSEUDO_DEFAULT
  -- *
  -- * * The default values for \p rng_type = CURAND_RNG_PSEUDO_PHILOX4_32_10 are:
  -- * - \p seed = 0
  -- * - \p offset = 0
  -- * - \p ordering = CURAND_ORDERING_PSEUDO_DEFAULT
  -- *
  -- * The default values for \p rng_type = CURAND_RNG_QUASI_SOBOL32 are:
  -- * - \p dimensions = 1
  -- * - \p offset = 0
  -- * - \p ordering = CURAND_ORDERING_QUASI_DEFAULT
  -- *
  -- * The default values for \p rng_type = CURAND_RNG_QUASI_SOBOL64 are:
  -- * - \p dimensions = 1
  -- * - \p offset = 0
  -- * - \p ordering = CURAND_ORDERING_QUASI_DEFAULT
  -- *
  -- * The default values for \p rng_type = CURAND_RNG_QUASI_SCRAMBLED_SOBOL32 are:
  -- * - \p dimensions = 1
  -- * - \p offset = 0
  -- * - \p ordering = CURAND_ORDERING_QUASI_DEFAULT
  -- *
  -- * The default values for \p rng_type = CURAND_RNG_QUASI_SCRAMBLED_SOBOL64 are:
  -- * - \p dimensions = 1
  -- * - \p offset = 0
  -- * - \p ordering = CURAND_ORDERING_QUASI_DEFAULT
  -- *
  -- * \param generator - Pointer to generator
  -- * \param rng_type - Type of generator to create
  -- *
  -- * \return
  -- * - CURAND_STATUS_ALLOCATION_FAILED if memory could not be allocated \n
  -- * - CURAND_STATUS_INITIALIZATION_FAILED if there was a problem setting up the GPU \n
  -- * - CURAND_STATUS_VERSION_MISMATCH if the header file version does not match the
  -- *   dynamically linked library version \n
  -- * - CURAND_STATUS_TYPE_ERROR if the value for \p rng_type is invalid \n
  -- * - CURAND_STATUS_SUCCESS if generator was created successfully \n
  --  

   function curandCreateGeneratorHost (generator : System.Address; rng_type : curandRngType_t) return curandStatus_t;  -- /usr/local/cuda-8.0/include/curand.h:402
   pragma Import (C, curandCreateGeneratorHost, "curandCreateGeneratorHost");

  --*
  -- * \brief Destroy an existing generator.
  -- *
  -- * Destroy an existing generator and free all memory associated with its state.
  -- *
  -- * \param generator - Generator to destroy
  -- *
  -- * \return
  -- * - CURAND_STATUS_NOT_INITIALIZED if the generator was never created \n
  -- * - CURAND_STATUS_SUCCESS if generator was destroyed successfully \n
  --  

   function curandDestroyGenerator (generator : curandGenerator_t) return curandStatus_t;  -- /usr/local/cuda-8.0/include/curand.h:416
   pragma Import (C, curandDestroyGenerator, "curandDestroyGenerator");

  --*
  -- * \brief Return the version number of the library.
  -- *
  -- * Return in \p *version the version number of the dynamically linked CURAND
  -- * library.  The format is the same as CUDART_VERSION from the CUDA Runtime.
  -- * The only supported configuration is CURAND version equal to CUDA Runtime
  -- * version.
  -- *
  -- * \param version - CURAND library version
  -- *
  -- * \return
  -- * - CURAND_STATUS_SUCCESS if the version number was successfully returned \n
  --  

   function curandGetVersion (version : access int) return curandStatus_t;  -- /usr/local/cuda-8.0/include/curand.h:432
   pragma Import (C, curandGetVersion, "curandGetVersion");

  --*
  --* \brief Return the value of the curand property.
  --*
  --* Return in \p *value the number for the property described by \p type of the
  --* dynamically linked CURAND library.
  --*
  --* \param type - CUDA library property
  --* \param value - integer value for the requested property
  --*
  --* \return
  --* - CURAND_STATUS_SUCCESS if the property value was successfully returned \n
  --* - CURAND_STATUS_OUT_OF_RANGE if the property type is not recognized \n
  -- 

   function curandGetProperty (c_type : library_types_h.libraryPropertyType; value : access int) return curandStatus_t;  -- /usr/local/cuda-8.0/include/curand.h:448
   pragma Import (C, curandGetProperty, "curandGetProperty");

  --*
  -- * \brief Set the current stream for CURAND kernel launches.
  -- *
  -- * Set the current stream for CURAND kernel launches.  All library functions
  -- * will use this stream until set again.
  -- *
  -- * \param generator - Generator to modify
  -- * \param stream - Stream to use or ::NULL for null stream
  -- *
  -- * \return
  -- * - CURAND_STATUS_NOT_INITIALIZED if the generator was never created \n
  -- * - CURAND_STATUS_SUCCESS if stream was set successfully \n
  --  

   function curandSetStream (generator : curandGenerator_t; stream : driver_types_h.cudaStream_t) return curandStatus_t;  -- /usr/local/cuda-8.0/include/curand.h:465
   pragma Import (C, curandSetStream, "curandSetStream");

  --*
  -- * \brief Set the seed value of the pseudo-random number generator.  
  -- * 
  -- * Set the seed value of the pseudorandom number generator.
  -- * All values of seed are valid.  Different seeds will produce different sequences.
  -- * Different seeds will often not be statistically correlated with each other,
  -- * but some pairs of seed values may generate sequences which are statistically correlated.
  -- *
  -- * \param generator - Generator to modify
  -- * \param seed - Seed value
  -- * 
  -- * \return
  -- * - CURAND_STATUS_NOT_INITIALIZED if the generator was never created \n
  -- * - CURAND_STATUS_TYPE_ERROR if the generator is not a pseudorandom number generator \n
  -- * - CURAND_STATUS_SUCCESS if generator seed was set successfully \n
  --  

   function curandSetPseudoRandomGeneratorSeed (generator : curandGenerator_t; seed : Extensions.unsigned_long_long) return curandStatus_t;  -- /usr/local/cuda-8.0/include/curand.h:484
   pragma Import (C, curandSetPseudoRandomGeneratorSeed, "curandSetPseudoRandomGeneratorSeed");

  --*
  -- * \brief Set the absolute offset of the pseudo or quasirandom number generator.
  -- *
  -- * Set the absolute offset of the pseudo or quasirandom number generator.
  -- *
  -- * All values of offset are valid.  The offset position is absolute, not 
  -- * relative to the current position in the sequence.
  -- *
  -- * \param generator - Generator to modify
  -- * \param offset - Absolute offset position
  -- *
  -- * \return
  -- * - CURAND_STATUS_NOT_INITIALIZED if the generator was never created \n
  -- * - CURAND_STATUS_SUCCESS if generator offset was set successfully \n
  --  

   function curandSetGeneratorOffset (generator : curandGenerator_t; offset : Extensions.unsigned_long_long) return curandStatus_t;  -- /usr/local/cuda-8.0/include/curand.h:502
   pragma Import (C, curandSetGeneratorOffset, "curandSetGeneratorOffset");

  --*
  -- * \brief Set the ordering of results of the pseudo or quasirandom number generator.
  -- *
  -- * Set the ordering of results of the pseudo or quasirandom number generator.
  -- *
  -- * Legal values of \p order for pseudorandom generators are:
  -- * - CURAND_ORDERING_PSEUDO_DEFAULT
  -- * - CURAND_ORDERING_PSEUDO_BEST
  -- * - CURAND_ORDERING_PSEUDO_SEEDED
  -- *
  -- * Legal values of \p order for quasirandom generators are:
  -- * - CURAND_ORDERING_QUASI_DEFAULT
  -- *
  -- * \param generator - Generator to modify
  -- * \param order - Ordering of results
  -- *
  -- * \return
  -- * - CURAND_STATUS_NOT_INITIALIZED if the generator was never created \n
  -- * - CURAND_STATUS_OUT_OF_RANGE if the ordering is not valid \n
  -- * - CURAND_STATUS_SUCCESS if generator ordering was set successfully \n
  --  

   function curandSetGeneratorOrdering (generator : curandGenerator_t; order : curandOrdering_t) return curandStatus_t;  -- /usr/local/cuda-8.0/include/curand.h:526
   pragma Import (C, curandSetGeneratorOrdering, "curandSetGeneratorOrdering");

  --*
  -- * \brief Set the number of dimensions.
  -- *
  -- * Set the number of dimensions to be generated by the quasirandom number
  -- * generator.
  -- * 
  -- * Legal values for \p num_dimensions are 1 to 20000.
  -- * 
  -- * \param generator - Generator to modify
  -- * \param num_dimensions - Number of dimensions
  -- *
  -- * \return
  -- * - CURAND_STATUS_NOT_INITIALIZED if the generator was never created \n
  -- * - CURAND_STATUS_OUT_OF_RANGE if num_dimensions is not valid \n
  -- * - CURAND_STATUS_TYPE_ERROR if the generator is not a quasirandom number generator \n
  -- * - CURAND_STATUS_SUCCESS if generator ordering was set successfully \n
  --  

   function curandSetQuasiRandomGeneratorDimensions (generator : curandGenerator_t; num_dimensions : unsigned) return curandStatus_t;  -- /usr/local/cuda-8.0/include/curand.h:546
   pragma Import (C, curandSetQuasiRandomGeneratorDimensions, "curandSetQuasiRandomGeneratorDimensions");

  --*
  -- * \brief Generate 32-bit pseudo or quasirandom numbers.
  -- *
  -- * Use \p generator to generate \p num 32-bit results into the device memory at
  -- * \p outputPtr.  The device memory must have been previously allocated and be
  -- * large enough to hold all the results.  Launches are done with the stream
  -- * set using ::curandSetStream(), or the null stream if no stream has been set.
  -- *
  -- * Results are 32-bit values with every bit random.
  -- *
  -- * \param generator - Generator to use
  -- * \param outputPtr - Pointer to device memory to store CUDA-generated results, or
  -- *                 Pointer to host memory to store CPU-generated results
  -- * \param num - Number of random 32-bit values to generate
  -- *
  -- * \return
  -- * - CURAND_STATUS_NOT_INITIALIZED if the generator was never created \n
  -- * - CURAND_STATUS_PREEXISTING_FAILURE if there was an existing error from 
  -- *     a previous kernel launch \n
  -- * - CURAND_STATUS_LENGTH_NOT_MULTIPLE if the number of output samples is
  -- *    not a multiple of the quasirandom dimension \n
  -- * - CURAND_STATUS_LAUNCH_FAILURE if the kernel launch failed for any reason \n
  -- * - CURAND_STATUS_TYPE_ERROR if the generator is a 64 bit quasirandom generator.
  -- * (use ::curandGenerateLongLong() with 64 bit quasirandom generators)
  -- * - CURAND_STATUS_SUCCESS if the results were generated successfully \n
  --  

   function curandGenerate
     (generator : curandGenerator_t;
      outputPtr : access unsigned;
      num : stddef_h.size_t) return curandStatus_t;  -- /usr/local/cuda-8.0/include/curand.h:575
   pragma Import (C, curandGenerate, "curandGenerate");

  --*
  -- * \brief Generate 64-bit quasirandom numbers.
  -- *
  -- * Use \p generator to generate \p num 64-bit results into the device memory at
  -- * \p outputPtr.  The device memory must have been previously allocated and be
  -- * large enough to hold all the results.  Launches are done with the stream
  -- * set using ::curandSetStream(), or the null stream if no stream has been set.
  -- *
  -- * Results are 64-bit values with every bit random.
  -- *
  -- * \param generator - Generator to use
  -- * \param outputPtr - Pointer to device memory to store CUDA-generated results, or
  -- *                 Pointer to host memory to store CPU-generated results
  -- * \param num - Number of random 64-bit values to generate
  -- *
  -- * \return
  -- * - CURAND_STATUS_NOT_INITIALIZED if the generator was never created \n
  -- * - CURAND_STATUS_PREEXISTING_FAILURE if there was an existing error from 
  -- *     a previous kernel launch \n
  -- * - CURAND_STATUS_LENGTH_NOT_MULTIPLE if the number of output samples is
  -- *    not a multiple of the quasirandom dimension \n
  -- * - CURAND_STATUS_LAUNCH_FAILURE if the kernel launch failed for any reason \n
  -- * - CURAND_STATUS_TYPE_ERROR if the generator is not a 64 bit quasirandom generator\n
  -- * - CURAND_STATUS_SUCCESS if the results were generated successfully \n
  --  

   function curandGenerateLongLong
     (generator : curandGenerator_t;
      outputPtr : access Extensions.unsigned_long_long;
      num : stddef_h.size_t) return curandStatus_t;  -- /usr/local/cuda-8.0/include/curand.h:603
   pragma Import (C, curandGenerateLongLong, "curandGenerateLongLong");

  --*
  -- * \brief Generate uniformly distributed floats.
  -- *
  -- * Use \p generator to generate \p num float results into the device memory at
  -- * \p outputPtr.  The device memory must have been previously allocated and be
  -- * large enough to hold all the results.  Launches are done with the stream
  -- * set using ::curandSetStream(), or the null stream if no stream has been set.
  -- *
  -- * Results are 32-bit floating point values between \p 0.0f and \p 1.0f,
  -- * excluding \p 0.0f and including \p 1.0f.
  -- *
  -- * \param generator - Generator to use
  -- * \param outputPtr - Pointer to device memory to store CUDA-generated results, or
  -- *                 Pointer to host memory to store CPU-generated results
  -- * \param num - Number of floats to generate
  -- *
  -- * \return
  -- * - CURAND_STATUS_NOT_INITIALIZED if the generator was never created \n
  -- * - CURAND_STATUS_PREEXISTING_FAILURE if there was an existing error from
  -- *    a previous kernel launch \n
  -- * - CURAND_STATUS_LAUNCH_FAILURE if the kernel launch failed for any reason \n
  -- * - CURAND_STATUS_LENGTH_NOT_MULTIPLE if the number of output samples is
  -- *    not a multiple of the quasirandom dimension \n
  -- * - CURAND_STATUS_SUCCESS if the results were generated successfully \n
  --  

   function curandGenerateUniform
     (generator : curandGenerator_t;
      outputPtr : access float;
      num : stddef_h.size_t) return curandStatus_t;  -- /usr/local/cuda-8.0/include/curand.h:631
   pragma Import (C, curandGenerateUniform, "curandGenerateUniform");

  --*
  -- * \brief Generate uniformly distributed doubles.
  -- *
  -- * Use \p generator to generate \p num double results into the device memory at
  -- * \p outputPtr.  The device memory must have been previously allocated and be
  -- * large enough to hold all the results.  Launches are done with the stream
  -- * set using ::curandSetStream(), or the null stream if no stream has been set.
  -- *
  -- * Results are 64-bit double precision floating point values between 
  -- * \p 0.0 and \p 1.0, excluding \p 0.0 and including \p 1.0.
  -- *
  -- * \param generator - Generator to use
  -- * \param outputPtr - Pointer to device memory to store CUDA-generated results, or
  -- *                 Pointer to host memory to store CPU-generated results
  -- * \param num - Number of doubles to generate
  -- *
  -- * \return
  -- * - CURAND_STATUS_NOT_INITIALIZED if the generator was never created \n
  -- * - CURAND_STATUS_PREEXISTING_FAILURE if there was an existing error from
  -- *    a previous kernel launch \n
  -- * - CURAND_STATUS_LAUNCH_FAILURE if the kernel launch failed for any reason \n
  -- * - CURAND_STATUS_LENGTH_NOT_MULTIPLE if the number of output samples is
  -- *    not a multiple of the quasirandom dimension \n
  -- * - CURAND_STATUS_DOUBLE_PRECISION_REQUIRED if the GPU does not support double precision \n
  -- * - CURAND_STATUS_SUCCESS if the results were generated successfully \n
  --  

   function curandGenerateUniformDouble
     (generator : curandGenerator_t;
      outputPtr : access double;
      num : stddef_h.size_t) return curandStatus_t;  -- /usr/local/cuda-8.0/include/curand.h:660
   pragma Import (C, curandGenerateUniformDouble, "curandGenerateUniformDouble");

  --*
  -- * \brief Generate normally distributed doubles.
  -- *
  -- * Use \p generator to generate \p n float results into the device memory at
  -- * \p outputPtr.  The device memory must have been previously allocated and be
  -- * large enough to hold all the results.  Launches are done with the stream
  -- * set using ::curandSetStream(), or the null stream if no stream has been set.
  -- *
  -- * Results are 32-bit floating point values with mean \p mean and standard
  -- * deviation \p stddev.
  -- *
  -- * Normally distributed results are generated from pseudorandom generators
  -- * with a Box-Muller transform, and so require \p n to be even.
  -- * Quasirandom generators use an inverse cumulative distribution 
  -- * function to preserve dimensionality.
  -- *
  -- * There may be slight numerical differences between results generated
  -- * on the GPU with generators created with ::curandCreateGenerator()
  -- * and results calculated on the CPU with generators created with
  -- * ::curandCreateGeneratorHost().  These differences arise because of
  -- * differences in results for transcendental functions.  In addition,
  -- * future versions of CURAND may use newer versions of the CUDA math
  -- * library, so different versions of CURAND may give slightly different
  -- * numerical values.
  -- *
  -- * \param generator - Generator to use
  -- * \param outputPtr - Pointer to device memory to store CUDA-generated results, or
  -- *                 Pointer to host memory to store CPU-generated results
  -- * \param n - Number of floats to generate
  -- * \param mean - Mean of normal distribution
  -- * \param stddev - Standard deviation of normal distribution
  -- *
  -- * \return
  -- * - CURAND_STATUS_NOT_INITIALIZED if the generator was never created \n
  -- * - CURAND_STATUS_PREEXISTING_FAILURE if there was an existing error from
  -- *    a previous kernel launch \n
  -- * - CURAND_STATUS_LAUNCH_FAILURE if the kernel launch failed for any reason \n
  -- * - CURAND_STATUS_LENGTH_NOT_MULTIPLE if the number of output samples is
  -- *    not a multiple of the quasirandom dimension, or is not a multiple
  -- *    of two for pseudorandom generators \n
  -- * - CURAND_STATUS_SUCCESS if the results were generated successfully \n
  --  

   function curandGenerateNormal
     (generator : curandGenerator_t;
      outputPtr : access float;
      n : stddef_h.size_t;
      mean : float;
      stddev : float) return curandStatus_t;  -- /usr/local/cuda-8.0/include/curand.h:705
   pragma Import (C, curandGenerateNormal, "curandGenerateNormal");

  --*
  -- * \brief Generate normally distributed doubles.
  -- *
  -- * Use \p generator to generate \p n double results into the device memory at
  -- * \p outputPtr.  The device memory must have been previously allocated and be
  -- * large enough to hold all the results.  Launches are done with the stream
  -- * set using ::curandSetStream(), or the null stream if no stream has been set.
  -- *
  -- * Results are 64-bit floating point values with mean \p mean and standard
  -- * deviation \p stddev.
  -- *
  -- * Normally distributed results are generated from pseudorandom generators
  -- * with a Box-Muller transform, and so require \p n to be even.
  -- * Quasirandom generators use an inverse cumulative distribution 
  -- * function to preserve dimensionality.
  -- *
  -- * There may be slight numerical differences between results generated
  -- * on the GPU with generators created with ::curandCreateGenerator()
  -- * and results calculated on the CPU with generators created with
  -- * ::curandCreateGeneratorHost().  These differences arise because of
  -- * differences in results for transcendental functions.  In addition,
  -- * future versions of CURAND may use newer versions of the CUDA math
  -- * library, so different versions of CURAND may give slightly different
  -- * numerical values.
  -- *
  -- * \param generator - Generator to use
  -- * \param outputPtr - Pointer to device memory to store CUDA-generated results, or
  -- *                 Pointer to host memory to store CPU-generated results
  -- * \param n - Number of doubles to generate
  -- * \param mean - Mean of normal distribution
  -- * \param stddev - Standard deviation of normal distribution
  -- *
  -- * \return
  -- * - CURAND_STATUS_NOT_INITIALIZED if the generator was never created \n
  -- * - CURAND_STATUS_PREEXISTING_FAILURE if there was an existing error from
  -- *    a previous kernel launch \n
  -- * - CURAND_STATUS_LAUNCH_FAILURE if the kernel launch failed for any reason \n
  -- * - CURAND_STATUS_LENGTH_NOT_MULTIPLE if the number of output samples is
  -- *    not a multiple of the quasirandom dimension, or is not a multiple
  -- *    of two for pseudorandom generators \n
  -- * - CURAND_STATUS_DOUBLE_PRECISION_REQUIRED if the GPU does not support double precision \n
  -- * - CURAND_STATUS_SUCCESS if the results were generated successfully \n
  --  

   function curandGenerateNormalDouble
     (generator : curandGenerator_t;
      outputPtr : access double;
      n : stddef_h.size_t;
      mean : double;
      stddev : double) return curandStatus_t;  -- /usr/local/cuda-8.0/include/curand.h:752
   pragma Import (C, curandGenerateNormalDouble, "curandGenerateNormalDouble");

  --*
  -- * \brief Generate log-normally distributed floats.
  -- *
  -- * Use \p generator to generate \p n float results into the device memory at
  -- * \p outputPtr.  The device memory must have been previously allocated and be
  -- * large enough to hold all the results.  Launches are done with the stream
  -- * set using ::curandSetStream(), or the null stream if no stream has been set.
  -- *
  -- * Results are 32-bit floating point values with log-normal distribution based on
  -- * an associated normal distribution with mean \p mean and standard deviation \p stddev.
  -- *
  -- * Normally distributed results are generated from pseudorandom generators
  -- * with a Box-Muller transform, and so require \p n to be even.
  -- * Quasirandom generators use an inverse cumulative distribution 
  -- * function to preserve dimensionality. 
  -- * The normally distributed results are transformed into log-normal distribution.
  -- *
  -- * There may be slight numerical differences between results generated
  -- * on the GPU with generators created with ::curandCreateGenerator()
  -- * and results calculated on the CPU with generators created with
  -- * ::curandCreateGeneratorHost().  These differences arise because of
  -- * differences in results for transcendental functions.  In addition,
  -- * future versions of CURAND may use newer versions of the CUDA math
  -- * library, so different versions of CURAND may give slightly different
  -- * numerical values.
  -- *
  -- * \param generator - Generator to use
  -- * \param outputPtr - Pointer to device memory to store CUDA-generated results, or
  -- *                 Pointer to host memory to store CPU-generated results
  -- * \param n - Number of floats to generate
  -- * \param mean - Mean of associated normal distribution
  -- * \param stddev - Standard deviation of associated normal distribution
  -- *
  -- * \return
  -- * - CURAND_STATUS_NOT_INITIALIZED if the generator was never created \n
  -- * - CURAND_STATUS_PREEXISTING_FAILURE if there was an existing error from
  -- *    a previous kernel launch \n
  -- * - CURAND_STATUS_LAUNCH_FAILURE if the kernel launch failed for any reason \n
  -- * - CURAND_STATUS_LENGTH_NOT_MULTIPLE if the number of output samples is
  -- *    not a multiple of the quasirandom dimension, or is not a multiple
  -- *    of two for pseudorandom generators \n
  -- * - CURAND_STATUS_SUCCESS if the results were generated successfully \n
  --  

   function curandGenerateLogNormal
     (generator : curandGenerator_t;
      outputPtr : access float;
      n : stddef_h.size_t;
      mean : float;
      stddev : float) return curandStatus_t;  -- /usr/local/cuda-8.0/include/curand.h:799
   pragma Import (C, curandGenerateLogNormal, "curandGenerateLogNormal");

  --*
  -- * \brief Generate log-normally distributed doubles.
  -- *
  -- * Use \p generator to generate \p n double results into the device memory at
  -- * \p outputPtr.  The device memory must have been previously allocated and be
  -- * large enough to hold all the results.  Launches are done with the stream
  -- * set using ::curandSetStream(), or the null stream if no stream has been set.
  -- *
  -- * Results are 64-bit floating point values with log-normal distribution based on
  -- * an associated normal distribution with mean \p mean and standard deviation \p stddev.
  -- *
  -- * Normally distributed results are generated from pseudorandom generators
  -- * with a Box-Muller transform, and so require \p n to be even.
  -- * Quasirandom generators use an inverse cumulative distribution 
  -- * function to preserve dimensionality.
  -- * The normally distributed results are transformed into log-normal distribution.
  -- *
  -- * There may be slight numerical differences between results generated
  -- * on the GPU with generators created with ::curandCreateGenerator()
  -- * and results calculated on the CPU with generators created with
  -- * ::curandCreateGeneratorHost().  These differences arise because of
  -- * differences in results for transcendental functions.  In addition,
  -- * future versions of CURAND may use newer versions of the CUDA math
  -- * library, so different versions of CURAND may give slightly different
  -- * numerical values.
  -- *
  -- * \param generator - Generator to use
  -- * \param outputPtr - Pointer to device memory to store CUDA-generated results, or
  -- *                 Pointer to host memory to store CPU-generated results
  -- * \param n - Number of doubles to generate
  -- * \param mean - Mean of normal distribution
  -- * \param stddev - Standard deviation of normal distribution
  -- *
  -- * \return
  -- * - CURAND_STATUS_NOT_INITIALIZED if the generator was never created \n
  -- * - CURAND_STATUS_PREEXISTING_FAILURE if there was an existing error from
  -- *    a previous kernel launch \n
  -- * - CURAND_STATUS_LAUNCH_FAILURE if the kernel launch failed for any reason \n
  -- * - CURAND_STATUS_LENGTH_NOT_MULTIPLE if the number of output samples is
  -- *    not a multiple of the quasirandom dimension, or is not a multiple
  -- *    of two for pseudorandom generators \n
  -- * - CURAND_STATUS_DOUBLE_PRECISION_REQUIRED if the GPU does not support double precision \n
  -- * - CURAND_STATUS_SUCCESS if the results were generated successfully \n
  --  

   function curandGenerateLogNormalDouble
     (generator : curandGenerator_t;
      outputPtr : access double;
      n : stddef_h.size_t;
      mean : double;
      stddev : double) return curandStatus_t;  -- /usr/local/cuda-8.0/include/curand.h:847
   pragma Import (C, curandGenerateLogNormalDouble, "curandGenerateLogNormalDouble");

  --*
  -- * \brief Construct the histogram array for a Poisson distribution.
  -- *
  -- * Construct the histogram array for the Poisson distribution with lambda \p lambda.
  -- * For lambda greater than 2000, an approximation with a normal distribution is used.
  -- *
  -- * \param lambda - lambda for the Poisson distribution
  -- *
  -- *
  -- * \param discrete_distribution - pointer to the histogram in device memory
  -- *
  -- * \return
  -- * - CURAND_STATUS_ALLOCATION_FAILED if memory could not be allocated \n
  -- * - CURAND_STATUS_DOUBLE_PRECISION_REQUIRED if the GPU does not support double precision \n
  -- * - CURAND_STATUS_INITIALIZATION_FAILED if there was a problem setting up the GPU \n
  -- * - CURAND_STATUS_NOT_INITIALIZED if the distribution pointer was null \n
  -- * - CURAND_STATUS_PREEXISTING_FAILURE if there was an existing error from
  -- *    a previous kernel launch \n
  -- * - CURAND_STATUS_OUT_OF_RANGE if lambda is non-positive or greater than 400,000 \n
  -- * - CURAND_STATUS_SUCCESS if the histogram was generated successfully \n
  --  

   function curandCreatePoissonDistribution (lambda : double; discrete_distribution : System.Address) return curandStatus_t;  -- /usr/local/cuda-8.0/include/curand.h:873
   pragma Import (C, curandCreatePoissonDistribution, "curandCreatePoissonDistribution");

  --*
  -- * \brief Destroy the histogram array for a discrete distribution (e.g. Poisson).
  -- *
  -- * Destroy the histogram array for a discrete distribution created by curandCreatePoissonDistribution.
  -- *
  -- * \param discrete_distribution - pointer to device memory where the histogram is stored
  -- *
  -- * \return
  -- * - CURAND_STATUS_NOT_INITIALIZED if the histogram was never created \n
  -- * - CURAND_STATUS_SUCCESS if the histogram was destroyed successfully \n
  --  

   function curandDestroyDistribution (discrete_distribution : curandDiscreteDistribution_t) return curandStatus_t;  -- /usr/local/cuda-8.0/include/curand.h:889
   pragma Import (C, curandDestroyDistribution, "curandDestroyDistribution");

  --*
  -- * \brief Generate Poisson-distributed unsigned ints.
  -- *
  -- * Use \p generator to generate \p n unsigned int results into device memory at
  -- * \p outputPtr.  The device memory must have been previously allocated and must be
  -- * large enough to hold all the results.  Launches are done with the stream
  -- * set using ::curandSetStream(), or the null stream if no stream has been set.
  -- *
  -- * Results are 32-bit unsigned int point values with Poisson distribution, with lambda \p lambda.
  -- *
  -- * \param generator - Generator to use
  -- * \param outputPtr - Pointer to device memory to store CUDA-generated results, or
  -- *                 Pointer to host memory to store CPU-generated results
  -- * \param n - Number of unsigned ints to generate
  -- * \param lambda - lambda for the Poisson distribution
  -- *
  -- * \return
  -- * - CURAND_STATUS_NOT_INITIALIZED if the generator was never created \n
  -- * - CURAND_STATUS_PREEXISTING_FAILURE if there was an existing error from
  -- *    a previous kernel launch \n
  -- * - CURAND_STATUS_LAUNCH_FAILURE if the kernel launch failed for any reason \n
  -- * - CURAND_STATUS_LENGTH_NOT_MULTIPLE if the number of output samples is
  -- *    not a multiple of the quasirandom dimension\n
  -- * - CURAND_STATUS_DOUBLE_PRECISION_REQUIRED if the GPU or sm does not support double precision \n
  -- * - CURAND_STATUS_OUT_OF_RANGE if lambda is non-positive or greater than 400,000 \n
  -- * - CURAND_STATUS_SUCCESS if the results were generated successfully \n
  --  

   function curandGeneratePoisson
     (generator : curandGenerator_t;
      outputPtr : access unsigned;
      n : stddef_h.size_t;
      lambda : double) return curandStatus_t;  -- /usr/local/cuda-8.0/include/curand.h:921
   pragma Import (C, curandGeneratePoisson, "curandGeneratePoisson");

  -- just for internal usage
   function curandGeneratePoissonMethod
     (generator : curandGenerator_t;
      outputPtr : access unsigned;
      n : stddef_h.size_t;
      lambda : double;
      method : curandMethod_t) return curandStatus_t;  -- /usr/local/cuda-8.0/include/curand.h:925
   pragma Import (C, curandGeneratePoissonMethod, "curandGeneratePoissonMethod");

   function curandGenerateBinomial
     (generator : curandGenerator_t;
      outputPtr : access unsigned;
      num : stddef_h.size_t;
      n : unsigned;
      p : double) return curandStatus_t;  -- /usr/local/cuda-8.0/include/curand.h:930
   pragma Import (C, curandGenerateBinomial, "curandGenerateBinomial");

  -- just for internal usage
   function curandGenerateBinomialMethod
     (generator : curandGenerator_t;
      outputPtr : access unsigned;
      num : stddef_h.size_t;
      n : unsigned;
      p : double;
      method : curandMethod_t) return curandStatus_t;  -- /usr/local/cuda-8.0/include/curand.h:934
   pragma Import (C, curandGenerateBinomialMethod, "curandGenerateBinomialMethod");

  --*
  -- * \brief Setup starting states.
  -- *
  -- * Generate the starting state of the generator.  This function is
  -- * automatically called by generation functions such as
  -- * ::curandGenerate() and ::curandGenerateUniform().
  -- * It can be called manually for performance testing reasons to separate
  -- * timings for starting state generation and random number generation.
  -- *
  -- * \param generator - Generator to update
  -- *
  -- * \return
  -- * - CURAND_STATUS_NOT_INITIALIZED if the generator was never created \n
  -- * - CURAND_STATUS_PREEXISTING_FAILURE if there was an existing error from 
  -- *     a previous kernel launch \n
  -- * - CURAND_STATUS_LAUNCH_FAILURE if the kernel launch failed for any reason \n
  -- * - CURAND_STATUS_SUCCESS if the seeds were generated successfully \n
  --  

   function curandGenerateSeeds (generator : curandGenerator_t) return curandStatus_t;  -- /usr/local/cuda-8.0/include/curand.h:959
   pragma Import (C, curandGenerateSeeds, "curandGenerateSeeds");

  --*
  -- * \brief Get direction vectors for 32-bit quasirandom number generation.
  -- *
  -- * Get a pointer to an array of direction vectors that can be used
  -- * for quasirandom number generation.  The resulting pointer will
  -- * reference an array of direction vectors in host memory.
  -- *
  -- * The array contains vectors for many dimensions.  Each dimension
  -- * has 32 vectors.  Each individual vector is an unsigned int.
  -- *
  -- * Legal values for \p set are:
  -- * - CURAND_DIRECTION_VECTORS_32_JOEKUO6 (20,000 dimensions)
  -- * - CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6 (20,000 dimensions)
  -- *
  -- * \param vectors - Address of pointer in which to return direction vectors
  -- * \param set - Which set of direction vectors to use
  -- *
  -- * \return
  -- * - CURAND_STATUS_OUT_OF_RANGE if the choice of set is invalid \n
  -- * - CURAND_STATUS_SUCCESS if the pointer was set successfully \n
  --  

   function curandGetDirectionVectors32 (vectors : System.Address; set : curandDirectionVectorSet_t) return curandStatus_t;  -- /usr/local/cuda-8.0/include/curand.h:983
   pragma Import (C, curandGetDirectionVectors32, "curandGetDirectionVectors32");

  --*
  -- * \brief Get scramble constants for 32-bit scrambled Sobol' .
  -- *
  -- * Get a pointer to an array of scramble constants that can be used
  -- * for quasirandom number generation.  The resulting pointer will
  -- * reference an array of unsinged ints in host memory.
  -- *
  -- * The array contains constants for many dimensions.  Each dimension
  -- * has a single unsigned int constant.
  -- *
  -- * \param constants - Address of pointer in which to return scramble constants
  -- *
  -- * \return
  -- * - CURAND_STATUS_SUCCESS if the pointer was set successfully \n
  --  

   function curandGetScrambleConstants32 (constants : System.Address) return curandStatus_t;  -- /usr/local/cuda-8.0/include/curand.h:1001
   pragma Import (C, curandGetScrambleConstants32, "curandGetScrambleConstants32");

  --*
  -- * \brief Get direction vectors for 64-bit quasirandom number generation.
  -- *
  -- * Get a pointer to an array of direction vectors that can be used
  -- * for quasirandom number generation.  The resulting pointer will
  -- * reference an array of direction vectors in host memory.
  -- *
  -- * The array contains vectors for many dimensions.  Each dimension
  -- * has 64 vectors.  Each individual vector is an unsigned long long.
  -- *
  -- * Legal values for \p set are:
  -- * - CURAND_DIRECTION_VECTORS_64_JOEKUO6 (20,000 dimensions)
  -- * - CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6 (20,000 dimensions)
  -- *
  -- * \param vectors - Address of pointer in which to return direction vectors
  -- * \param set - Which set of direction vectors to use
  -- *
  -- * \return
  -- * - CURAND_STATUS_OUT_OF_RANGE if the choice of set is invalid \n
  -- * - CURAND_STATUS_SUCCESS if the pointer was set successfully \n
  --  

   function curandGetDirectionVectors64 (vectors : System.Address; set : curandDirectionVectorSet_t) return curandStatus_t;  -- /usr/local/cuda-8.0/include/curand.h:1025
   pragma Import (C, curandGetDirectionVectors64, "curandGetDirectionVectors64");

  --*
  -- * \brief Get scramble constants for 64-bit scrambled Sobol' .
  -- *
  -- * Get a pointer to an array of scramble constants that can be used
  -- * for quasirandom number generation.  The resulting pointer will
  -- * reference an array of unsinged long longs in host memory.
  -- *
  -- * The array contains constants for many dimensions.  Each dimension
  -- * has a single unsigned long long constant.
  -- *
  -- * \param constants - Address of pointer in which to return scramble constants
  -- *
  -- * \return
  -- * - CURAND_STATUS_SUCCESS if the pointer was set successfully \n
  --  

   function curandGetScrambleConstants64 (constants : System.Address) return curandStatus_t;  -- /usr/local/cuda-8.0/include/curand.h:1043
   pragma Import (C, curandGetScrambleConstants64, "curandGetScrambleConstants64");

  --* @}  
end curand_h;

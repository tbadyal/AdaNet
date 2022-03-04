pragma Ada_2005;
pragma Style_Checks (Off);

with Interfaces.C; use Interfaces.C;
with Interfaces.C.Extensions;
with System;
limited with curand_philox4x32_x_h;
with vector_types_h;

package curand_kernel_h is

   --  unsupported macro: QUALIFIERS static __forceinline__ __device__
   EXTRA_FLAG_NORMAL : constant := 16#00000001#;  --  /usr/local/cuda-8.0/include/curand_kernel.h:123
   EXTRA_FLAG_LOG_NORMAL : constant := 16#00000002#;  --  /usr/local/cuda-8.0/include/curand_kernel.h:124

   MRG32K3A_MOD1 : constant := 4294967087.;  --  /usr/local/cuda-8.0/include/curand_kernel.h:141
   MRG32K3A_MOD2 : constant := 4294944443.;  --  /usr/local/cuda-8.0/include/curand_kernel.h:142

   MRG32K3A_A12 : constant := 1403580.;  --  /usr/local/cuda-8.0/include/curand_kernel.h:146
   MRG32K3A_A13N : constant := 810728.;  --  /usr/local/cuda-8.0/include/curand_kernel.h:147
   MRG32K3A_A21 : constant := 527612.;  --  /usr/local/cuda-8.0/include/curand_kernel.h:148
   MRG32K3A_A23N : constant := 1370589.;  --  /usr/local/cuda-8.0/include/curand_kernel.h:149
   MRG32K3A_NORM : constant := 2.328306549295728e-10;  --  /usr/local/cuda-8.0/include/curand_kernel.h:150

   MRG32K3A_BITS_NORM : constant := 1.000000048662;  --  /usr/local/cuda-8.0/include/curand_kernel.h:154
   --  unsupported macro: MRG32K3A_SKIPUNITS_DOUBLES (sizeof(struct sMRG32k3aSkipUnits)/sizeof(double))
   --  unsupported macro: MRG32K3A_SKIPSUBSEQ_DOUBLES (sizeof(struct sMRG32k3aSkipSubSeq)/sizeof(double))
   --  unsupported macro: MRG32K3A_SKIPSEQ_DOUBLES (sizeof(struct sMRG32k3aSkipSeq)/sizeof(double))

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
  -- * \defgroup DEVICE Device API
  -- *
  -- * @{
  --  

  -- Test RNG  
  -- This generator uses the formula:
  --   x_n = x_(n-1) + 1 mod 2^32
  --   x_0 = (unsigned int)seed * 3
  --   Subsequences are spaced 31337 steps apart.
  -- 

   type curandStateTest is record
      v : aliased unsigned;  -- /usr/local/cuda-8.0/include/curand_kernel.h:83
   end record;
   pragma Convention (C_Pass_By_Copy, curandStateTest);  -- /usr/local/cuda-8.0/include/curand_kernel.h:82

  --* \cond UNHIDE_TYPEDEFS  
   subtype curandStateTest_t is curandStateTest;

  --* \endcond  
  -- XORSHIFT FAMILY RNGs  
  -- These generators are a family proposed by Marsaglia.  They keep state
  --   in 32 bit chunks, then use repeated shift and xor operations to scramble
  --   the bits.  The following generators are a combination of a simple Weyl
  --   generator with an N variable XORSHIFT generator.
  -- 

  -- XORSHIFT RNG  
  -- This generator uses the xorwow formula of
  --www.jstatsoft.org/v08/i14/paper page 5
  --Has period 2^192 - 2^32.
  -- 

  --*
  -- * CURAND XORWOW state 
  --  

  -- * Implementation details not in reference documentation  
   type curandStateXORWOW_v_array is array (0 .. 4) of aliased unsigned;
   type curandStateXORWOW is record
      d : aliased unsigned;  -- /usr/local/cuda-8.0/include/curand_kernel.h:110
      v : aliased curandStateXORWOW_v_array;  -- /usr/local/cuda-8.0/include/curand_kernel.h:110
      boxmuller_flag : aliased int;  -- /usr/local/cuda-8.0/include/curand_kernel.h:111
      boxmuller_flag_double : aliased int;  -- /usr/local/cuda-8.0/include/curand_kernel.h:112
      boxmuller_extra : aliased float;  -- /usr/local/cuda-8.0/include/curand_kernel.h:113
      boxmuller_extra_double : aliased double;  -- /usr/local/cuda-8.0/include/curand_kernel.h:114
   end record;
   pragma Convention (C_Pass_By_Copy, curandStateXORWOW);  -- /usr/local/cuda-8.0/include/curand_kernel.h:109

  -- * CURAND XORWOW state 
  --  

  --* \cond UNHIDE_TYPEDEFS  
   subtype curandStateXORWOW_t is curandStateXORWOW;

  --* \endcond  
  -- Combined Multiple Recursive Generators  
  -- These generators are a family proposed by L'Ecuyer.  They keep state
  --   in sets of doubles, then use repeated modular arithmetic multiply operations 
  --   to scramble the bits in each set, and combine the result.
  -- 

  -- MRG32k3a RNG  
  -- This generator uses the MRG32k3A formula of
  --http://www.iro.umontreal.ca/~lecuyer/myftp/streams00/c++/streams4.pdf
  --Has period 2^191.
  -- 

  -- moduli for the recursions  
  --* \cond UNHIDE_DEFINES  
  -- Constants used in generation  
  -- #define MRG32K3A_BITS_NORM ((double)((POW32_DOUBLE-1.0)/MOD1))
  --  above constant, used verbatim, rounds differently on some host systems.
  -- Constants for address manipulation  
  --* \endcond  
  --*
  -- * CURAND MRG32K3A state 
  --  

  -- Implementation details not in reference documentation  
   type curandStateMRG32k3a_s1_array is array (0 .. 2) of aliased double;
   type curandStateMRG32k3a_s2_array is array (0 .. 2) of aliased double;
   type curandStateMRG32k3a is record
      s1 : aliased curandStateMRG32k3a_s1_array;  -- /usr/local/cuda-8.0/include/curand_kernel.h:174
      s2 : aliased curandStateMRG32k3a_s2_array;  -- /usr/local/cuda-8.0/include/curand_kernel.h:175
      boxmuller_flag : aliased int;  -- /usr/local/cuda-8.0/include/curand_kernel.h:176
      boxmuller_flag_double : aliased int;  -- /usr/local/cuda-8.0/include/curand_kernel.h:177
      boxmuller_extra : aliased float;  -- /usr/local/cuda-8.0/include/curand_kernel.h:178
      boxmuller_extra_double : aliased double;  -- /usr/local/cuda-8.0/include/curand_kernel.h:179
   end record;
   pragma Convention (C_Pass_By_Copy, curandStateMRG32k3a);  -- /usr/local/cuda-8.0/include/curand_kernel.h:173

  -- * CURAND MRG32K3A state 
  --  

  --* \cond UNHIDE_TYPEDEFS  
   subtype curandStateMRG32k3a_t is curandStateMRG32k3a;

  --* \endcond  
  -- SOBOL QRNG  
  --*
  -- * CURAND Sobol32 state 
  --  

  -- Implementation details not in reference documentation  
   type curandStateSobol32_direction_vectors_array is array (0 .. 31) of aliased unsigned;
   type curandStateSobol32 is record
      i : aliased unsigned;  -- /usr/local/cuda-8.0/include/curand_kernel.h:197
      x : aliased unsigned;  -- /usr/local/cuda-8.0/include/curand_kernel.h:197
      c : aliased unsigned;  -- /usr/local/cuda-8.0/include/curand_kernel.h:197
      direction_vectors : aliased curandStateSobol32_direction_vectors_array;  -- /usr/local/cuda-8.0/include/curand_kernel.h:198
   end record;
   pragma Convention (C_Pass_By_Copy, curandStateSobol32);  -- /usr/local/cuda-8.0/include/curand_kernel.h:196

  -- * CURAND Sobol32 state 
  --  

  --* \cond UNHIDE_TYPEDEFS  
   subtype curandStateSobol32_t is curandStateSobol32;

  --* \endcond  
  --*
  -- * CURAND Scrambled Sobol32 state 
  --  

  -- Implementation details not in reference documentation  
   type curandStateScrambledSobol32_direction_vectors_array is array (0 .. 31) of aliased unsigned;
   type curandStateScrambledSobol32 is record
      i : aliased unsigned;  -- /usr/local/cuda-8.0/include/curand_kernel.h:215
      x : aliased unsigned;  -- /usr/local/cuda-8.0/include/curand_kernel.h:215
      c : aliased unsigned;  -- /usr/local/cuda-8.0/include/curand_kernel.h:215
      direction_vectors : aliased curandStateScrambledSobol32_direction_vectors_array;  -- /usr/local/cuda-8.0/include/curand_kernel.h:216
   end record;
   pragma Convention (C_Pass_By_Copy, curandStateScrambledSobol32);  -- /usr/local/cuda-8.0/include/curand_kernel.h:214

  -- * CURAND Scrambled Sobol32 state 
  --  

  --* \cond UNHIDE_TYPEDEFS  
   subtype curandStateScrambledSobol32_t is curandStateScrambledSobol32;

  --* \endcond  
  --*
  -- * CURAND Sobol64 state 
  --  

  -- Implementation details not in reference documentation  
   type curandStateSobol64_direction_vectors_array is array (0 .. 63) of aliased Extensions.unsigned_long_long;
   type curandStateSobol64 is record
      i : aliased Extensions.unsigned_long_long;  -- /usr/local/cuda-8.0/include/curand_kernel.h:233
      x : aliased Extensions.unsigned_long_long;  -- /usr/local/cuda-8.0/include/curand_kernel.h:233
      c : aliased Extensions.unsigned_long_long;  -- /usr/local/cuda-8.0/include/curand_kernel.h:233
      direction_vectors : aliased curandStateSobol64_direction_vectors_array;  -- /usr/local/cuda-8.0/include/curand_kernel.h:234
   end record;
   pragma Convention (C_Pass_By_Copy, curandStateSobol64);  -- /usr/local/cuda-8.0/include/curand_kernel.h:232

  -- * CURAND Sobol64 state 
  --  

  --* \cond UNHIDE_TYPEDEFS  
   subtype curandStateSobol64_t is curandStateSobol64;

  --* \endcond  
  --*
  -- * CURAND Scrambled Sobol64 state 
  --  

  -- Implementation details not in reference documentation  
   type curandStateScrambledSobol64_direction_vectors_array is array (0 .. 63) of aliased Extensions.unsigned_long_long;
   type curandStateScrambledSobol64 is record
      i : aliased Extensions.unsigned_long_long;  -- /usr/local/cuda-8.0/include/curand_kernel.h:251
      x : aliased Extensions.unsigned_long_long;  -- /usr/local/cuda-8.0/include/curand_kernel.h:251
      c : aliased Extensions.unsigned_long_long;  -- /usr/local/cuda-8.0/include/curand_kernel.h:251
      direction_vectors : aliased curandStateScrambledSobol64_direction_vectors_array;  -- /usr/local/cuda-8.0/include/curand_kernel.h:252
   end record;
   pragma Convention (C_Pass_By_Copy, curandStateScrambledSobol64);  -- /usr/local/cuda-8.0/include/curand_kernel.h:250

  -- * CURAND Scrambled Sobol64 state 
  --  

  --* \cond UNHIDE_TYPEDEFS  
   subtype curandStateScrambledSobol64_t is curandStateScrambledSobol64;

  --* \endcond  
  -- * Default RNG
  --  

  --* \cond UNHIDE_TYPEDEFS  
   subtype curandState_t is curandStateXORWOW;

   subtype curandState is curandStateXORWOW;

  --* \endcond  
  --************************************************************************** 
  -- Utility functions needed by RNGs  
  --************************************************************************** 
  --* \cond UNHIDE_UTILITIES  
  -- 
  --   multiply vector by matrix, store in result
  --   matrix is n x n, measured in 32 bit units
  --   matrix is stored in row major order
  --   vector and result cannot be same pointer
  -- 

   --  skipped func __curand_matvec

  -- generate identity matrix  
   --  skipped func __curand_matidentity

  -- multiply matrixA by matrixB, store back in matrixA
  --   matrixA and matrixB must not be same matrix  

   --  skipped func __curand_matmat

  -- copy vectorA to vector  
   --  skipped func __curand_veccopy

  -- copy matrixA to matrix  
   --  skipped func __curand_matcopy

  -- compute matrixA to power p, store result in matrix  
   --  skipped func __curand_matpow

  -- Convert unsigned int to float, use no intrinsics  
   --  skipped func __curand_uint32AsFloat

  -- Convert two unsigned ints to double, use no intrinsics  
   --  skipped func __curand_hilouint32AsDouble

  -- Convert unsigned int to float, as efficiently as possible  
   --  skipped func __curand_uint32_as_float

  --QUALIFIERS double __curand_hilouint32_as_double(unsigned int hi, unsigned int lo)
  --{
  --#if __CUDA_ARCH__ > 0
  --    return __hiloint2double(hi, lo);
  --#elif !defined(__CUDA_ARCH__)
  --    return hilouint32AsDouble(hi, lo);
  --#endif
  --}
  -- 

  --************************************************************************** 
  -- Utility functions needed by MRG32k3a RNG                                  
  -- Matrix operations modulo some integer less than 2**32, done in            
  -- double precision floating point, with care not to overflow 53 bits        
  --************************************************************************** 
  -- return i mod m.                                                           
  -- assumes i and m are integers represented accurately in doubles            
   function curand_MRGmod (i : double; m : double) return double;  -- /usr/local/cuda-8.0/include/curand_kernel.h:414
   pragma Import (CPP, curand_MRGmod, "_ZL13curand_MRGmoddd");

  -- Multiplication modulo m. Inputs i and j less than 2**32                   
  -- Ensure intermediate results do not exceed 2**53                           
   function curand_MRGmodMul
     (i : double;
      j : double;
      m : double) return double;  -- /usr/local/cuda-8.0/include/curand_kernel.h:427
   pragma Import (CPP, curand_MRGmodMul, "_ZL16curand_MRGmodMulddd");

  -- multiply 3 by 3 matrices of doubles, modulo m                             
   procedure curand_MRGmatMul3x3
     (i1 : System.Address;
      i2 : System.Address;
      o : System.Address;
      m : double);  -- /usr/local/cuda-8.0/include/curand_kernel.h:442
   pragma Import (CPP, curand_MRGmatMul3x3, "_ZL19curand_MRGmatMul3x3PA3_dS0_S0_d");

  -- multiply 3 by 3 matrix times 3 by 1 vector of doubles, modulo m           
   procedure curand_MRGmatVecMul3x3
     (i : System.Address;
      v : access double;
      m : double);  -- /usr/local/cuda-8.0/include/curand_kernel.h:463
   pragma Import (CPP, curand_MRGmatVecMul3x3, "_ZL22curand_MRGmatVecMul3x3PA3_dPdd");

  -- raise a 3 by 3 matrix of doubles to a 64 bit integer power pow, modulo m  
  -- input is index zero of an array of 3 by 3 matrices m,                     
  -- each m = m[0]**(2**index)                                                 
   procedure curand_MRGmatPow3x3
     (c_in : System.Address;
      o : System.Address;
      m : double;
      pow : Extensions.unsigned_long_long);  -- /usr/local/cuda-8.0/include/curand_kernel.h:483
   pragma Import (CPP, curand_MRGmatPow3x3, "_ZL19curand_MRGmatPow3x3PA3_A3_dPS_dy");

  -- raise a 3 by 3 matrix of doubles to the power                             
  -- 2 to the power (pow modulo 191), modulo m                                 
   procedure curnand_MRGmatPow2Pow3x3
     (c_in : System.Address;
      o : System.Address;
      m : double;
      pow : unsigned_long);  -- /usr/local/cuda-8.0/include/curand_kernel.h:506
   pragma Import (CPP, curnand_MRGmatPow2Pow3x3, "_ZL24curnand_MRGmatPow2Pow3x3PA3_dS0_dm");

  --* \endcond  
  --************************************************************************** 
  -- Kernel implementations of RNGs                                            
  --************************************************************************** 
  -- Test RNG  
   procedure curand_init
     (seed : Extensions.unsigned_long_long;
      subsequence : Extensions.unsigned_long_long;
      offset : Extensions.unsigned_long_long;
      state : access curandStateTest_t);  -- /usr/local/cuda-8.0/include/curand_kernel.h:535
   pragma Import (CPP, curand_init, "_ZL11curand_inityyyP15curandStateTest");

   function curand (state : access curandStateTest_t) return unsigned;  -- /usr/local/cuda-8.0/include/curand_kernel.h:545
   pragma Import (CPP, curand, "_ZL6curandP15curandStateTest");

   procedure skipahead (n : Extensions.unsigned_long_long; state : access curandStateTest_t);  -- /usr/local/cuda-8.0/include/curand_kernel.h:551
   pragma Import (CPP, skipahead, "_ZL9skipaheadyP15curandStateTest");

  -- XORWOW RNG  
  -- Generate matrix that advances one step
  -- matrix has n * n * 32 32-bit elements
  -- solve for matrix by stepping single bit states
  -- unsigned int matrix[n * n * 32];
  -- unsigned int matrixA[n * n * 32];
  -- unsigned int vector[n];
  -- unsigned int result[n];
  -- unsigned int matrix[n * n * 32];
  -- unsigned int matrixA[n * n * 32];
  -- unsigned int vector[n];
  -- unsigned int result[n];
  -- No update of state->d needed, guaranteed to be a multiple of 2^32  
  --*
  -- * \brief Update XORWOW state to skip \p n elements.
  -- *
  -- * Update the XORWOW state in \p state to skip ahead \p n elements.
  -- *
  -- * All values of \p n are valid.  Large values require more computation and so
  -- * will take more time to complete.
  -- *
  -- * \param n - Number of elements to skip
  -- * \param state - Pointer to state to update
  --  

   procedure skipahead (n : Extensions.unsigned_long_long; state : access curandStateXORWOW_t);  -- /usr/local/cuda-8.0/include/curand_kernel.h:701
   pragma Import (CPP, skipahead, "_ZL9skipaheadyP17curandStateXORWOW");

  --*
  -- * \brief Update XORWOW state to skip ahead \p n subsequences.
  -- *
  -- * Update the XORWOW state in \p state to skip ahead \p n subsequences.  Each
  -- * subsequence is \xmlonly<ph outputclass="xmlonly">2<sup>67</sup></ph>\endxmlonly elements long, so this means the function will skip ahead
  -- * \xmlonly<ph outputclass="xmlonly">2<sup>67</sup></ph>\endxmlonly  * n elements.
  -- *
  -- * All values of \p n are valid.  Large values require more computation and so
  -- * will take more time to complete.
  -- *
  -- * \param n - Number of subsequences to skip
  -- * \param state - Pointer to state to update
  --  

   procedure skipahead_sequence (n : Extensions.unsigned_long_long; state : access curandStateXORWOW_t);  -- /usr/local/cuda-8.0/include/curand_kernel.h:720
   pragma Import (CPP, skipahead_sequence, "_ZL18skipahead_sequenceyP17curandStateXORWOW");

   --  skipped func _curand_init_scratch

  -- Break up seed, apply salt
  -- Constants are arbitrary nonzero values
  -- Simple multiplication to mix up bits
  -- Constants are arbitrary odd values
  --*
  -- * \brief Initialize XORWOW state.
  -- *
  -- * Initialize XORWOW state in \p state with the given \p seed, \p subsequence,
  -- * and \p offset.
  -- *
  -- * All input values of \p seed, \p subsequence, and \p offset are legal.  Large
  -- * values for \p subsequence and \p offset require more computation and so will
  -- * take more time to complete.
  -- *
  -- * A value of 0 for \p seed sets the state to the values of the original
  -- * published version of the \p xorwow algorithm.
  -- *
  -- * \param seed - Arbitrary bits to use as a seed
  -- * \param subsequence - Subsequence to start at
  -- * \param offset - Absolute offset into sequence
  -- * \param state - Pointer to state to initialize
  --  

   procedure curand_init
     (seed : Extensions.unsigned_long_long;
      subsequence : Extensions.unsigned_long_long;
      offset : Extensions.unsigned_long_long;
      state : access curandStateXORWOW_t);  -- /usr/local/cuda-8.0/include/curand_kernel.h:772
   pragma Import (CPP, curand_init, "_ZL11curand_inityyyP17curandStateXORWOW");

  --*
  -- * \brief Return 32-bits of pseudorandomness from an XORWOW generator.
  -- *
  -- * Return 32-bits of pseudorandomness from the XORWOW generator in \p state,
  -- * increment position of generator by one.
  -- *
  -- * \param state - Pointer to state to update
  -- *
  -- * \return 32-bits of pseudorandomness as an unsigned int, all bits valid to use.
  --  

   function curand (state : access curandStateXORWOW_t) return unsigned;  -- /usr/local/cuda-8.0/include/curand_kernel.h:791
   pragma Import (CPP, curand, "_ZL6curandP17curandStateXORWOW");

  --*
  -- * \brief Return 32-bits of pseudorandomness from an Philox4_32_10 generator.
  -- *
  -- * Return 32-bits of pseudorandomness from the Philox4_32_10 generator in \p state,
  -- * increment position of generator by one.
  -- *
  -- * \param state - Pointer to state to update
  -- *
  -- * \return 32-bits of pseudorandomness as an unsigned int, all bits valid to use.
  --  

   function curand (state : access curand_philox4x32_x_h.curandStatePhilox4_32_10_t) return unsigned;  -- /usr/local/cuda-8.0/include/curand_kernel.h:816
   pragma Import (CPP, curand, "_ZL6curandP24curandStatePhilox4_32_10");

  -- Maintain the invariant: output[STATE] is always "good" and
  --  is the next value to be returned by curand.
  --*
  -- * \brief Return tuple of 4 32-bit pseudorandoms from a Philox4_32_10 generator.
  -- *
  -- * Return 128 bits of pseudorandomness from the Philox4_32_10 generator in \p state,
  -- * increment position of generator by four.
  -- *
  -- * \param state - Pointer to state to update
  -- *
  -- * \return 128-bits of pseudorandomness as a uint4, all bits valid to use.
  --  

   function curand4 (state : access curand_philox4x32_x_h.curandStatePhilox4_32_10_t) return vector_types_h.uint4;  -- /usr/local/cuda-8.0/include/curand_kernel.h:840
   pragma Import (CPP, curand4, "_ZL7curand4P24curandStatePhilox4_32_10");

  -- NOT possible but needed to avoid compiler warnings
  --*
  -- * \brief Update Philox4_32_10 state to skip \p n elements.
  -- *
  -- * Update the Philox4_32_10 state in \p state to skip ahead \p n elements.
  -- *
  -- * All values of \p n are valid.
  -- *
  -- * \param n - Number of elements to skip
  -- * \param state - Pointer to state to update
  --  

   procedure skipahead (n : Extensions.unsigned_long_long; state : access curand_philox4x32_x_h.curandStatePhilox4_32_10_t);  -- /usr/local/cuda-8.0/include/curand_kernel.h:885
   pragma Import (CPP, skipahead, "_ZL9skipaheadyP24curandStatePhilox4_32_10");

  --*
  -- * \brief Update Philox4_32_10 state to skip ahead \p n subsequences.
  -- *
  -- * Update the Philox4_32_10 state in \p state to skip ahead \p n subsequences.  Each
  -- * subsequence is \xmlonly<ph outputclass="xmlonly">2<sup>66</sup></ph>\endxmlonly elements long, so this means the function will skip ahead
  -- * \xmlonly<ph outputclass="xmlonly">2<sup>66</sup></ph>\endxmlonly * n elements.
  -- *
  -- * All values of \p n are valid.
  -- *
  -- * \param n - Number of subsequences to skip
  -- * \param state - Pointer to state to update
  --  

   procedure skipahead_sequence (n : Extensions.unsigned_long_long; state : access curand_philox4x32_x_h.curandStatePhilox4_32_10_t);  -- /usr/local/cuda-8.0/include/curand_kernel.h:909
   pragma Import (CPP, skipahead_sequence, "_ZL18skipahead_sequenceyP24curandStatePhilox4_32_10");

  --*
  -- * \brief Initialize Philox4_32_10 state.
  -- *
  -- * Initialize Philox4_32_10 state in \p state with the given \p seed, p\ subsequence,
  -- * and \p offset.
  -- *
  -- * All input values for \p seed, \p subseqence and \p offset are legal.  Each of the
  -- * \xmlonly<ph outputclass="xmlonly">2<sup>64</sup></ph>\endxmlonly possible
  -- * values of seed selects an independent sequence of length 
  -- * \xmlonly<ph outputclass="xmlonly">2<sup>130</sup></ph>\endxmlonly.
  -- * The first 
  -- * \xmlonly<ph outputclass="xmlonly">2<sup>66</sup> * subsequence + offset</ph>\endxmlonly.
  -- * values of the sequence are skipped.
  -- * I.e., subsequences are of length
  -- * \xmlonly<ph outputclass="xmlonly">2<sup>66</sup></ph>\endxmlonly.
  -- *
  -- * \param seed - Arbitrary bits to use as a seed
  -- * \param subsequence - Subsequence to start at
  -- * \param offset - Absolute offset into subsequence
  -- * \param state - Pointer to state to initialize
  --  

   procedure curand_init
     (seed : Extensions.unsigned_long_long;
      subsequence : Extensions.unsigned_long_long;
      offset : Extensions.unsigned_long_long;
      state : access curand_philox4x32_x_h.curandStatePhilox4_32_10_t);  -- /usr/local/cuda-8.0/include/curand_kernel.h:936
   pragma Import (CPP, curand_init, "_ZL11curand_inityyyP24curandStatePhilox4_32_10");

  -- MRG32k3a RNG  
  -- Base generator for MRG32k3a                                               
  -- note that the parameters have been selected such that intermediate        
  -- results stay within 53 bits                                               
  --  nj's implementation  
  -- (1.0 / m1)__hi  
  -- (1.0 / m1)__lo  
  -- (1.0 / m2)__hi  
  -- (1.0 / m2)__lo  
  -- end nj's implementation  
   function curand_MRG32k3a (state : access curandStateMRG32k3a_t) return double;  -- /usr/local/cuda-8.0/include/curand_kernel.h:991
   pragma Import (CPP, curand_MRG32k3a, "_ZL15curand_MRG32k3aP19curandStateMRG32k3a");

  --*
  -- * \brief Return 32-bits of pseudorandomness from an MRG32k3a generator.
  -- *
  -- * Return 32-bits of pseudorandomness from the MRG32k3a generator in \p state,
  -- * increment position of generator by one.
  -- *
  -- * \param state - Pointer to state to update
  -- *
  -- * \return 32-bits of pseudorandomness as an unsigned int, all bits valid to use.
  --  

   function curand (state : access curandStateMRG32k3a_t) return unsigned;  -- /usr/local/cuda-8.0/include/curand_kernel.h:1023
   pragma Import (CPP, curand, "_ZL6curandP19curandStateMRG32k3a");

  --*
  -- * \brief Update MRG32k3a state to skip \p n elements.
  -- *
  -- * Update the MRG32k3a state in \p state to skip ahead \p n elements.
  -- *
  -- * All values of \p n are valid.  Large values require more computation and so
  -- * will take more time to complete.
  -- *
  -- * \param n - Number of elements to skip
  -- * \param state - Pointer to state to update
  --  

   procedure skipahead (n : Extensions.unsigned_long_long; state : access curandStateMRG32k3a_t);  -- /usr/local/cuda-8.0/include/curand_kernel.h:1043
   pragma Import (CPP, skipahead, "_ZL9skipaheadyP19curandStateMRG32k3a");

  --*
  -- * \brief Update MRG32k3a state to skip ahead \p n subsequences.
  -- *
  -- * Update the MRG32k3a state in \p state to skip ahead \p n subsequences.  Each
  -- * subsequence is \xmlonly<ph outputclass="xmlonly">2<sup>127</sup></ph>\endxmlonly
  -- *
  -- * \xmlonly<ph outputclass="xmlonly">2<sup>76</sup></ph>\endxmlonly elements long, so this means the function will skip ahead
  -- * \xmlonly<ph outputclass="xmlonly">2<sup>67</sup></ph>\endxmlonly * n elements.
  -- *
  -- * Valid values of \p n are 0 to \xmlonly<ph outputclass="xmlonly">2<sup>51</sup></ph>\endxmlonly.  Note \p n will be masked to 51 bits
  -- *
  -- * \param n - Number of subsequences to skip
  -- * \param state - Pointer to state to update 
  --  

   procedure skipahead_subsequence (n : Extensions.unsigned_long_long; state : access curandStateMRG32k3a_t);  -- /usr/local/cuda-8.0/include/curand_kernel.h:1073
   pragma Import (CPP, skipahead_subsequence, "_ZL21skipahead_subsequenceyP19curandStateMRG32k3a");

  --*
  -- * \brief Update MRG32k3a state to skip ahead \p n sequences.
  -- *
  -- * Update the MRG32k3a state in \p state to skip ahead \p n sequences.  Each
  -- * sequence is \xmlonly<ph outputclass="xmlonly">2<sup>127</sup></ph>\endxmlonly elements long, so this means the function will skip ahead
  -- * \xmlonly<ph outputclass="xmlonly">2<sup>127</sup></ph>\endxmlonly * n elements. 
  -- *
  -- * All values of \p n are valid.  Large values require more computation and so
  -- * will take more time to complete.
  -- *
  -- * \param n - Number of sequences to skip
  -- * \param state - Pointer to state to update
  --  

   procedure skipahead_sequence (n : Extensions.unsigned_long_long; state : access curandStateMRG32k3a_t);  -- /usr/local/cuda-8.0/include/curand_kernel.h:1102
   pragma Import (CPP, skipahead_sequence, "_ZL18skipahead_sequenceyP19curandStateMRG32k3a");

  --*
  -- * \brief Initialize MRG32k3a state.
  -- *
  -- * Initialize MRG32k3a state in \p state with the given \p seed, \p subsequence,
  -- * and \p offset.
  -- *
  -- * All input values of \p seed, \p subsequence, and \p offset are legal. 
  -- * \p subsequence will be truncated to 51 bits to avoid running into the next sequence
  -- *
  -- * A value of 0 for \p seed sets the state to the values of the original
  -- * published version of the \p MRG32k3a algorithm.
  -- *
  -- * \param seed - Arbitrary bits to use as a seed
  -- * \param subsequence - Subsequence to start at
  -- * \param offset - Absolute offset into sequence
  -- * \param state - Pointer to state to initialize
  --  

   procedure curand_init
     (seed : Extensions.unsigned_long_long;
      subsequence : Extensions.unsigned_long_long;
      offset : Extensions.unsigned_long_long;
      state : access curandStateMRG32k3a_t);  -- /usr/local/cuda-8.0/include/curand_kernel.h:1136
   pragma Import (CPP, curand_init, "_ZL11curand_inityyyP19curandStateMRG32k3a");

  --*
  -- * \brief Update Sobol32 state to skip \p n elements.
  -- *
  -- * Update the Sobol32 state in \p state to skip ahead \p n elements.
  -- *
  -- * All values of \p n are valid.
  -- *
  -- * \param n - Number of elements to skip
  -- * \param state - Pointer to state to update
  --  

  -- Convert state->i to gray code  
  --*
  -- * \brief Update Sobol64 state to skip \p n elements.
  -- *
  -- * Update the Sobol64 state in \p state to skip ahead \p n elements.
  -- *
  -- * All values of \p n are valid.
  -- *
  -- * \param n - Number of elements to skip
  -- * \param state - Pointer to state to update
  --  

  -- Convert state->i to gray code  
  --*
  -- * \brief Initialize Sobol32 state.
  -- *
  -- * Initialize Sobol32 state in \p state with the given \p direction \p vectors and 
  -- * \p offset.
  -- *
  -- * The direction vector is a device pointer to an array of 32 unsigned ints.
  -- * All input values of \p offset are legal.
  -- *
  -- * \param direction_vectors - Pointer to array of 32 unsigned ints representing the
  -- * direction vectors for the desired dimension
  -- * \param offset - Absolute offset into sequence
  -- * \param state - Pointer to state to initialize
  --  

   procedure curand_init
     (direction_vectors : access unsigned;
      offset : unsigned;
      state : access curandStateSobol32_t);  -- /usr/local/cuda-8.0/include/curand_kernel.h:1228
   pragma Import (CPP, curand_init, "_ZL11curand_initPjjP18curandStateSobol32");

  --*
  -- * \brief Initialize Scrambled Sobol32 state.
  -- *
  -- * Initialize Sobol32 state in \p state with the given \p direction \p vectors and 
  -- * \p offset.
  -- *
  -- * The direction vector is a device pointer to an array of 32 unsigned ints.
  -- * All input values of \p offset are legal.
  -- *
  -- * \param direction_vectors - Pointer to array of 32 unsigned ints representing the
  -- direction vectors for the desired dimension
  -- * \param scramble_c Scramble constant
  -- * \param offset - Absolute offset into sequence
  -- * \param state - Pointer to state to initialize
  --  

   procedure curand_init
     (direction_vectors : access unsigned;
      scramble_c : unsigned;
      offset : unsigned;
      state : access curandStateScrambledSobol32_t);  -- /usr/local/cuda-8.0/include/curand_kernel.h:1255
   pragma Import (CPP, curand_init, "_ZL11curand_initPjjjP27curandStateScrambledSobol32");

  --*
  -- * \brief Initialize Sobol64 state.
  -- *
  -- * Initialize Sobol64 state in \p state with the given \p direction \p vectors and 
  -- * \p offset.
  -- *
  -- * The direction vector is a device pointer to an array of 64 unsigned long longs.
  -- * All input values of \p offset are legal.
  -- *
  -- * \param direction_vectors - Pointer to array of 64 unsigned long longs representing the
  -- direction vectors for the desired dimension
  -- * \param offset - Absolute offset into sequence
  -- * \param state - Pointer to state to initialize
  --  

   procedure curand_init
     (direction_vectors : access Extensions.unsigned_long_long;
      offset : Extensions.unsigned_long_long;
      state : access curandStateSobol64_t);  -- /usr/local/cuda-8.0/include/curand_kernel.h:1302
   pragma Import (CPP, curand_init, "_ZL11curand_initPyyP18curandStateSobol64");

  -- Moving from i to i+2^n_log2 element in gray code is flipping two bits  
  --*
  -- * \brief Initialize Scrambled Sobol64 state.
  -- *
  -- * Initialize Sobol64 state in \p state with the given \p direction \p vectors and 
  -- * \p offset.
  -- *
  -- * The direction vector is a device pointer to an array of 64 unsigned long longs.
  -- * All input values of \p offset are legal.
  -- *
  -- * \param direction_vectors - Pointer to array of 64 unsigned long longs representing the
  -- direction vectors for the desired dimension
  -- * \param scramble_c Scramble constant
  -- * \param offset - Absolute offset into sequence
  -- * \param state - Pointer to state to initialize
  --  

   procedure curand_init
     (direction_vectors : access Extensions.unsigned_long_long;
      scramble_c : Extensions.unsigned_long_long;
      offset : Extensions.unsigned_long_long;
      state : access curandStateScrambledSobol64_t);  -- /usr/local/cuda-8.0/include/curand_kernel.h:1341
   pragma Import (CPP, curand_init, "_ZL11curand_initPyyyP27curandStateScrambledSobol64");

  --*
  -- * \brief Return 32-bits of quasirandomness from a Sobol32 generator.
  -- *
  -- * Return 32-bits of quasirandomness from the Sobol32 generator in \p state,
  -- * increment position of generator by one.
  -- *
  -- * \param state - Pointer to state to update
  -- *
  -- * \return 32-bits of quasirandomness as an unsigned int, all bits valid to use.
  --  

   function curand (state : access curandStateSobol32_t) return unsigned;  -- /usr/local/cuda-8.0/include/curand_kernel.h:1366
   pragma Import (CPP, curand, "_ZL6curandP18curandStateSobol32");

  -- Moving from i to i+1 element in gray code is flipping one bit,
  --       the trailing zero bit of i
  --     

  --*
  -- * \brief Return 32-bits of quasirandomness from a scrambled Sobol32 generator.
  -- *
  -- * Return 32-bits of quasirandomness from the scrambled Sobol32 generator in \p state,
  -- * increment position of generator by one.
  -- *
  -- * \param state - Pointer to state to update
  -- *
  -- * \return 32-bits of quasirandomness as an unsigned int, all bits valid to use.
  --  

   function curand (state : access curandStateScrambledSobol32_t) return unsigned;  -- /usr/local/cuda-8.0/include/curand_kernel.h:1388
   pragma Import (CPP, curand, "_ZL6curandP27curandStateScrambledSobol32");

  -- Moving from i to i+1 element in gray code is flipping one bit,
  --       the trailing zero bit of i
  --     

  --*
  -- * \brief Return 64-bits of quasirandomness from a Sobol64 generator.
  -- *
  -- * Return 64-bits of quasirandomness from the Sobol64 generator in \p state,
  -- * increment position of generator by one.
  -- *
  -- * \param state - Pointer to state to update
  -- *
  -- * \return 64-bits of quasirandomness as an unsigned long long, all bits valid to use.
  --  

   function curand (state : access curandStateSobol64_t) return Extensions.unsigned_long_long;  -- /usr/local/cuda-8.0/include/curand_kernel.h:1410
   pragma Import (CPP, curand, "_ZL6curandP18curandStateSobol64");

  -- Moving from i to i+1 element in gray code is flipping one bit,
  --       the trailing zero bit of i
  --     

  --*
  -- * \brief Return 64-bits of quasirandomness from a scrambled Sobol64 generator.
  -- *
  -- * Return 64-bits of quasirandomness from the scrambled Sobol32 generator in \p state,
  -- * increment position of generator by one.
  -- *
  -- * \param state - Pointer to state to update
  -- *
  -- * \return 64-bits of quasirandomness as an unsigned long long, all bits valid to use.
  --  

   function curand (state : access curandStateScrambledSobol64_t) return Extensions.unsigned_long_long;  -- /usr/local/cuda-8.0/include/curand_kernel.h:1432
   pragma Import (CPP, curand, "_ZL6curandP27curandStateScrambledSobol64");

  -- Moving from i to i+1 element in gray code is flipping one bit,
  --       the trailing zero bit of i
  --     

   --  skipped func __get_precalculated_matrix

   --  skipped func __get_precalculated_matrix_host

   --  skipped func __get_mrg32k3a_matrix

   --  skipped func __get_mrg32k3a_matrix_host

   --  skipped func __get__cr_lgamma_table_host

  --* @}  
end curand_kernel_h;

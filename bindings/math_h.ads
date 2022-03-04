pragma Ada_2005;
pragma Style_Checks (Off);

with Interfaces.C; use Interfaces.C;
with Interfaces.C.Strings;

package math_h is

   FP_NAN : constant := 0;  --  /usr/include/math.h:212

   FP_INFINITE : constant := 1;  --  /usr/include/math.h:215

   FP_ZERO : constant := 2;  --  /usr/include/math.h:218

   FP_SUBNORMAL : constant := 3;  --  /usr/include/math.h:221

   FP_NORMAL : constant := 4;  --  /usr/include/math.h:224
   --  arg-macro: procedure fpclassify __builtin_fpclassify (FP_NAN, FP_INFINITE, FP_NORMAL, FP_SUBNORMAL, FP_ZERO, x)
   --    __builtin_fpclassify (FP_NAN, FP_INFINITE, FP_NORMAL, FP_SUBNORMAL, FP_ZERO, x)
   --  arg-macro: function signbit (sizeof (x) = sizeof (float) ? __builtin_signbitf (x) : sizeof (x) = sizeof (double) ? __builtin_signbit (x) : __builtin_signbitl (x)
   --    return sizeof (x) = sizeof (float) ? __builtin_signbitf (x) : sizeof (x) = sizeof (double) ? __builtin_signbit (x) : __builtin_signbitl (x);
   --  arg-macro: procedure isfinite __builtin_isfinite (x)
   --    __builtin_isfinite (x)
   --  arg-macro: procedure isnormal __builtin_isnormal (x)
   --    __builtin_isnormal (x)
   --  arg-macro: procedure isnan __builtin_isnan (x)
   --    __builtin_isnan (x)
   --  arg-macro: procedure isinf __builtin_isinf_sign (x)
   --    __builtin_isinf_sign (x)

   MATH_ERRNO : constant := 1;  --  /usr/include/math.h:319
   MATH_ERREXCEPT : constant := 2;  --  /usr/include/math.h:320
   --  unsupported macro: math_errhandling (MATH_ERRNO | MATH_ERREXCEPT)
   --  arg-macro: function issignaling (sizeof (x) = sizeof (float) ? __issignalingf (x) : sizeof (x) = sizeof (double) ? __issignaling (x) : __issignalingl (x)
   --    return sizeof (x) = sizeof (float) ? __issignalingf (x) : sizeof (x) = sizeof (double) ? __issignaling (x) : __issignalingl (x);

   X_TLOSS : constant := 1.41484755040568800000e+16;  --  /usr/include/math.h:388

   DOMAIN : constant := 1;  --  /usr/include/math.h:391
   SING : constant := 2;  --  /usr/include/math.h:392
   OVERFLOW : constant := 3;  --  /usr/include/math.h:393
   UNDERFLOW : constant := 4;  --  /usr/include/math.h:394
   TLOSS : constant := 5;  --  /usr/include/math.h:395
   PLOSS : constant := 6;  --  /usr/include/math.h:396

   HUGE : constant := 3.40282347e+38F;  --  /usr/include/math.h:399

   M_E : constant := 2.7182818284590452354;  --  /usr/include/math.h:413
   M_LOG2E : constant := 1.4426950408889634074;  --  /usr/include/math.h:414
   M_LOG10E : constant := 8#.43429448190325182765#;  --  /usr/include/math.h:415
   M_LN2 : constant := 8#.69314718055994530942#;  --  /usr/include/math.h:416
   M_LN10 : constant := 2.30258509299404568402;  --  /usr/include/math.h:417
   M_PI : constant := 3.14159265358979323846;  --  /usr/include/math.h:418
   M_PI_2 : constant := 1.57079632679489661923;  --  /usr/include/math.h:419
   M_PI_4 : constant := 8#.78539816339744830962#;  --  /usr/include/math.h:420
   M_1_PI : constant := 8#.31830988618379067154#;  --  /usr/include/math.h:421
   M_2_PI : constant := 8#.63661977236758134308#;  --  /usr/include/math.h:422
   M_2_SQRTPI : constant := 1.12837916709551257390;  --  /usr/include/math.h:423
   M_SQRT2 : constant := 1.41421356237309504880;  --  /usr/include/math.h:424
   M_SQRT1_2 : constant := 8#.70710678118654752440#;  --  /usr/include/math.h:425

   M_El : constant := 2.718281828459045235360287471352662498;  --  /usr/include/math.h:432
   M_LOG2El : constant := 1.442695040888963407359924681001892137;  --  /usr/include/math.h:433
   M_LOG10El : constant := 8#.434294481903251827651128918916605082#;  --  /usr/include/math.h:434
   M_LN2l : constant := 8#.693147180559945309417232121458176568#;  --  /usr/include/math.h:435
   M_LN10l : constant := 2.302585092994045684017991454684364208;  --  /usr/include/math.h:436
   M_PIl : constant := 3.141592653589793238462643383279502884;  --  /usr/include/math.h:437
   M_PI_2l : constant := 1.570796326794896619231321691639751442;  --  /usr/include/math.h:438
   M_PI_4l : constant := 8#.785398163397448309615660845819875721#;  --  /usr/include/math.h:439
   M_1_PIl : constant := 8#.318309886183790671537767526745028724#;  --  /usr/include/math.h:440
   M_2_PIl : constant := 8#.636619772367581343075535053490057448#;  --  /usr/include/math.h:441
   M_2_SQRTPIl : constant := 1.128379167095512573896158903121545172;  --  /usr/include/math.h:442
   M_SQRT2l : constant := 1.414213562373095048801688724209698079;  --  /usr/include/math.h:443
   M_SQRT1_2l : constant := 8#.707106781186547524400844362104849039#;  --  /usr/include/math.h:444
   --  arg-macro: procedure isgreater __builtin_isgreater(x, y)
   --    __builtin_isgreater(x, y)
   --  arg-macro: procedure isgreaterequal __builtin_isgreaterequal(x, y)
   --    __builtin_isgreaterequal(x, y)
   --  arg-macro: procedure isless __builtin_isless(x, y)
   --    __builtin_isless(x, y)
   --  arg-macro: procedure islessequal __builtin_islessequal(x, y)
   --    __builtin_islessequal(x, y)
   --  arg-macro: procedure islessgreater __builtin_islessgreater(x, y)
   --    __builtin_islessgreater(x, y)
   --  arg-macro: procedure isunordered __builtin_isunordered(u, v)
   --    __builtin_isunordered(u, v)

  -- Declarations for math functions.
  --   Copyright (C) 1991-2016 Free Software Foundation, Inc.
  --   This file is part of the GNU C Library.
  --   The GNU C Library is free software; you can redistribute it and/or
  --   modify it under the terms of the GNU Lesser General Public
  --   License as published by the Free Software Foundation; either
  --   version 2.1 of the License, or (at your option) any later version.
  --   The GNU C Library is distributed in the hope that it will be useful,
  --   but WITHOUT ANY WARRANTY; without even the implied warranty of
  --   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  --   Lesser General Public License for more details.
  --   You should have received a copy of the GNU Lesser General Public
  --   License along with the GNU C Library; if not, see
  --   <http://www.gnu.org/licenses/>.   

  -- *	ISO C99 Standard: 7.12 Mathematics	<math.h>
  --  

  -- Get machine-dependent vector math functions declarations.   
  -- Get machine-dependent HUGE_VAL value (returned on overflow).
  --   On all IEEE754 machines, this is +Infinity.   

  -- Get machine-dependent INFINITY value.   
  -- Get machine-dependent NAN value (returned for some domain errors).   
  -- Get general and ISO C99 specific information.   
  -- The file <bits/mathcalls.h> contains the prototypes for all the
  --   actual math functions.  These macros are used for those prototypes,
  --   so we can easily declare each function as both `name' and `__name',
  --   and can declare the float versions `namef' and `__namef'.   

  -- Include the file of declarations again, this time using `float'
  --   instead of `double' and appending f to each function name.   

  -- Include the file of declarations again, this time using `long double'
  --   instead of `double' and appending l to each function name.   

  -- This variable is used by `gamma' and `lgamma'.   
   signgam : aliased int;  -- /usr/include/math.h:168
   pragma Import (C, signgam, "signgam");

  -- ISO C99 defines some generic macros which work on any data type.   
  -- Get the architecture specific values describing the floating-point
  --   evaluation.  The following symbols will get defined:
  --    float_t	floating-point type at least as wide as `float' used
  --		to evaluate `float' expressions
  --    double_t	floating-point type at least as wide as `double' used
  --		to evaluate `double' expressions
  --    FLT_EVAL_METHOD
  --		Defined to
  --		  0	if `float_t' is `float' and `double_t' is `double'
  --		  1	if `float_t' and `double_t' are `double'
  --		  2	if `float_t' and `double_t' are `long double'
  --		  else	`float_t' and `double_t' are unspecified
  --    INFINITY	representation of the infinity value of type `float'
  --    FP_FAST_FMA
  --    FP_FAST_FMAF
  --    FP_FAST_FMAL
  --		If defined it indicates that the `fma' function
  --		generally executes about as fast as a multiply and an add.
  --		This macro is defined only iff the `fma' function is
  --		implemented directly with a hardware multiply-add instructions.
  --    FP_ILOGB0	Expands to a value returned by `ilogb (0.0)'.
  --    FP_ILOGBNAN	Expands to a value returned by `ilogb (NAN)'.
  --    DECIMAL_DIG	Number of decimal digits supported by conversion between
  --		decimal and all internal floating-point formats.
  -- 

  -- All floating-point numbers can be put in one of these categories.   
  -- GCC bug 66462 means we cannot use the math builtins with -fsignaling-nan,
  --   so disable builtins if this is enabled.  When fixed in a newer GCC,
  --   the __SUPPORT_SNAN__ check may be skipped for those versions.   

  -- Return number of classification appropriate for X.   
  -- Return nonzero value if sign of X is negative.   
  -- Return nonzero value if X is not +-Inf or NaN.   
  -- Return nonzero value if X is neither zero, subnormal, Inf, nor NaN.   
  -- Return nonzero value if X is a NaN.  We could use `fpclassify' but
  --   we already have this functions `__isnan' and it is faster.   

  -- Return nonzero value if X is positive or negative infinity.   
  -- Bitmasks for the math_errhandling macro.   
  -- By default all functions support both errno and exception handling.
  --   In gcc's fast math mode and if inline functions are defined this
  --   might not be true.   

  -- Return nonzero value if X is a signaling NaN.   
  -- Support for various different standard error handling behaviors.   
  -- According to IEEE 754/IEEE 854.   
  -- According to System V, release 4.   
  -- Nowadays also Unix98.   
  -- Actually this is ISO C99.   
   subtype u_LIB_VERSION_TYPE is unsigned;
   u_IEEE_u : constant u_LIB_VERSION_TYPE := -1;
   u_SVID_u : constant u_LIB_VERSION_TYPE := 0;
   u_XOPEN_u : constant u_LIB_VERSION_TYPE := 1;
   u_POSIX_u : constant u_LIB_VERSION_TYPE := 2;
   u_ISOC_u : constant u_LIB_VERSION_TYPE := 3;  -- /usr/include/math.h:354

  -- This variable can be changed at run-time to any of the values above to
  --   affect floating point error handling behavior (it may also be necessary
  --   to change the hardware FPU exception settings).   

  -- In SVID error handling, `matherr' is called with this description
  --   of the exceptional condition.
  --   We have a problem when using C++ since `exception' is a reserved
  --   name in C++.   

   type uu_exception is record
      c_type : aliased int;  -- /usr/include/math.h:375
      name : Interfaces.C.Strings.chars_ptr;  -- /usr/include/math.h:376
      arg1 : aliased double;  -- /usr/include/math.h:377
      arg2 : aliased double;  -- /usr/include/math.h:378
      retval : aliased double;  -- /usr/include/math.h:379
   end record;
   pragma Convention (C_Pass_By_Copy, uu_exception);  -- /usr/include/math.h:370

   function matherr (uu_exc : access uu_exception) return int;  -- /usr/include/math.h:383
   pragma Import (C, matherr, "matherr");

  -- Types of exceptions in the `type' field.   
  -- SVID mode specifies returning this large value instead of infinity.   
  -- X/Open wants another strange constant.   
  -- Some useful constants.   
  -- The above constants are not adequate for computation using `long double's.
  --   Therefore we provide as an extension constants with similar names as a
  --   GNU extension.  Provide enough digits for the 128-bit IEEE quad.   

  -- When compiling in strict ISO C compatible mode we must not use the
  --   inline functions since they, among other things, do not set the
  --   `errno' variable correctly.   

  -- ISO C99 defines some macros to compare number while taking care for
  --   unordered numbers.  Many FPUs provide special instructions to support
  --   these operations.  Generic support in GCC for these as builtins went
  --   in before 3.0.0, but not all cpus added their patterns.  We define
  --   versions that use the builtins here, and <bits/mathinline.h> will
  --   undef/redefine as appropriate for the specific GCC version in use.   

  -- Get machine-dependent inline versions (if there are any).   
  -- Define special entry points to use when the compiler got told to
  --   only expect finite results.   

  -- If we've still got undefined comparison macros, provide defaults.   
  -- Return nonzero value if X is greater than Y.   
  -- Return nonzero value if X is greater than or equal to Y.   
  -- Return nonzero value if X is less than Y.   
  -- Return nonzero value if X is less than or equal to Y.   
  -- Return nonzero value if either X is less than Y or Y is less than X.   
  -- Return nonzero value if arguments are unordered.   
end math_h;

pragma Ada_2005;
pragma Style_Checks (Off);

with Interfaces.C; use Interfaces.C;

package math_functions_hpp is

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

  --******************************************************************************
  --*                                                                              *
  --*                                                                              *
  --*                                                                              *
  --****************************************************************************** 

  -- Suppress VS warning: warning C4127: conditional expression is constant  
  -- long can be of 32-bit type on some systems.  
  -- long can be of 32-bit type on some systems.  
  --******************************************************************************
  --*                                                                              *
  --*                                                                              *
  --*                                                                              *
  --****************************************************************************** 

  --******************************************************************************
  --*                                                                              *
  --* DEVICE                                                                       *
  --*                                                                              *
  --****************************************************************************** 

  --******************************************************************************
  --*                                                                              *
  --* DEVICE IMPLEMENTATIONS FOR FUNCTIONS WITH BUILTIN NVOPENCC OPERATIONS        *
  --*                                                                              *
  --****************************************************************************** 

  --******************************************************************************
  --*                                                                              *
  --* DEVICE IMPLEMENTATIONS FOR FUNCTIONS WITHOUT BUILTIN NVOPENCC OPERATIONS     *
  --*                                                                              *
  --****************************************************************************** 

  -- we do not support long double yet, hence double  
  -- we do not support long double yet, hence double  
  --******************************************************************************
  --*                                                                              *
  --*                                                                              *
  --*                                                                              *
  --****************************************************************************** 

  -- __THROW  
  -- __THROW  
  -- __THROW  
  --__THROW  
  -- __THROW  
  -- __THROW  
  -- __THROW  
  -- __THROW  
  -- __THROW  
  -- __THROW  
  -- __THROW  
  -- provide own versions of QNX builtins  
  --******************************************************************************
  --*                                                                              *
  --* ONLY FOR HOST CODE! NOT FOR DEVICE EXECUTION                                 *
  --*                                                                              *
  --****************************************************************************** 

  --******************************************************************************
  --*                                                                              *
  --* HOST IMPLEMENTATION FOR DOUBLE ROUTINES FOR WINDOWS & APPLE PLATFORMS        *
  --*                                                                              *
  --****************************************************************************** 

  --******************************************************************************
  --*                                                                              *
  --* HOST IMPLEMENTATION FOR DOUBLE ROUTINES FOR WINDOWS PLATFORM                 *
  --*                                                                              *
  --****************************************************************************** 

  --  
  -- * The following is based on: David Goldberg, "What every computer scientist 
  -- * should know about floating-point arithmetic", ACM Computing Surveys, Volume 
  -- * 23, Issue 1, March 1991.
  --  

  -- a very close to zero  
  -- a somewhat close to zero  
  -- * This code based on: http://www.cs.berkeley.edu/~wkahan/Math128/Sumnfp.pdf
  --  

  -- a very close zero  
  -- a somewhat close zero  
  -- initial approximation  
  -- refine approximation  
  -- do FPREM1, a.k.a IEEE remainder  
  -- initialize quotient  
  -- remainder has sign of dividend  
  -- default  
  -- handled NaNs and infinities  
  -- normalize denormals  
  -- clamp iterations if exponent difference negative  
  -- Shift dividend and divisor right by one bit to prevent overflow
  --     during the division algorithm.
  --    

  -- default exponent of result    
  -- Use binary longhand division (restoring)  
  -- Save current remainder  
  -- If remainder's mantissa is all zeroes, final result is zero.  
  -- Normalize result  
  -- For IEEE remainder (quotient rounded to nearest-even we might need to 
  --     do a final subtraction of the divisor from the remainder.
  --   

  -- round quotient to nearest even  
  -- Since the divisor is greater than the remainder, the result will
  --         have opposite sign of the dividend. To avoid a negative mantissa
  --         when subtracting the divisor from remainder, reverse subtraction
  --       

  -- normalize result  
  -- package up result  
  -- normal  
  -- denormal  
  -- mask quotient down to least significant three bits  
  -- expo(x) - 1  
  -- expo(y) - 1  
  -- expo(z) - 1  
  -- fma (nan, y, z) --> nan
  --       fma (x, nan, z) --> nan
  --       fma (x, y, nan) --> nan 
  --     

  -- fma (0, inf, z) --> INDEFINITE
  --       fma (inf, 0, z) --> INDEFINITE
  --       fma (-inf,+y,+inf) --> INDEFINITE
  --       fma (+x,-inf,+inf) --> INDEFINITE
  --       fma (+inf,-y,+inf) --> INDEFINITE
  --       fma (-x,+inf,+inf) --> INDEFINITE
  --       fma (-inf,-y,-inf) --> INDEFINITE
  --       fma (-x,-inf,-inf) --> INDEFINITE
  --       fma (+inf,+y,-inf) --> INDEFINITE
  --       fma (+x,+inf,-inf) --> INDEFINITE
  --     

  -- fma (inf, y, z) --> inf
  --       fma (x, inf, z) --> inf
  --       fma (x, y, inf) --> inf
  --     

  -- fma (+0, -y, -0) --> -0
  --       fma (-0, +y, -0) --> -0
  --       fma (+x, -0, -0) --> -0
  --       fma (-x, +0, -0) --> -0
  --     

  -- fma (0, y, 0) --> +0  (-0 if round down and signs of addend differ)
  --       fma (x, 0, 0) --> +0  (-0 if round down and signs of addend differ)
  --     

  -- fma (0, y, z) --> z
  --       fma (x, 0, z) --> z
  --     

  -- set mantissa hidden bit  
  -- set mantissa hidden bit  
  -- mantissa  
  -- mantissa  
  -- mantissa  
  -- mantissa  
  -- expo-1  
  -- sign  
  -- z is not zero  
  -- compare and swap. put augend into xx:yy  
  -- augend_sign = expo_y, augend_mant = xx:yy, augend_expo = expo_x  
  -- addend_sign = s, addend_mant = zz:ww, addend_expo = expo_z  
  -- denormalize addend  
  -- signs differ, effective subtraction  
  -- complete cancelation, return 0  
  -- Oops, augend had smaller mantissa. Negate mantissa and flip
  --           sign of result
  --         

  -- normalize mantissa, if necessary  
  -- signs are the same, effective addition  
  -- or in sign bit  
  -- normal  
  -- lop off integer bit  
  -- mantissa lsb  
  -- overflow  
  -- subnormal  
  -- save sign bit  
  -- NaN  
  -- crossover  
  -- Stirling approximation; coefficients from Hart et al, "Computer 
  --       * Approximations", Wiley 1968. Approximation 5404. 
  --        

  -- a is an integer: return infinity  
  -- INF  
  -- compute sin(pi*x) accurately  
  --******************************************************************************
  --*                                                                              *
  --* HOST IMPLEMENTATION FOR FLOAT AND LONG DOUBLE ROUTINES FOR WINDOWS PLATFORM  *
  --* MAP FLOAT AND LONG DOUBLE ROUTINES TO DOUBLE ROUTINES                        *
  --*                                                                              *
  --****************************************************************************** 

  --******************************************************************************
  --*                                                                              *
  --* HOST IMPLEMENTATION FOR FLOAT ROUTINES FOR WINDOWS PLATFORM                  *
  --*                                                                              *
  --****************************************************************************** 

  --NaN 
  --crossover 
  --******************************************************************************
  --*                                                                              *
  --* HOST IMPLEMENTATION FOR DOUBLE AND FLOAT ROUTINES. ALL PLATFORMS             *
  --*                                                                              *
  --****************************************************************************** 

   function rsqrt (a : double) return double;  -- /usr/local/cuda-8.0/include/math_functions.hpp:3420
   pragma Import (CPP, rsqrt, "_Z5rsqrtd");

   function rcbrt (a : double) return double;  -- /usr/local/cuda-8.0/include/math_functions.hpp:3425
   pragma Import (CPP, rcbrt, "_Z5rcbrtd");

  -- initial approximation  
  -- refine approximation  
   function sinpi (a : double) return double;  -- /usr/local/cuda-8.0/include/math_functions.hpp:3449
   pragma Import (CPP, sinpi, "_Z5sinpid");

   function cospi (a : double) return double;  -- /usr/local/cuda-8.0/include/math_functions.hpp:3475
   pragma Import (CPP, cospi, "_Z5cospid");

   procedure sincospi
     (a : double;
      sptr : access double;
      cptr : access double);  -- /usr/local/cuda-8.0/include/math_functions.hpp:3505
   pragma Import (CPP, sincospi, "_Z8sincospidPdS_");

   function erfinv (a : double) return double;  -- /usr/local/cuda-8.0/include/math_functions.hpp:3511
   pragma Import (CPP, erfinv, "_Z6erfinvd");

  -- INDEFINITE  
  -- Infinity  
  -- Based on: J.M. Blair, C.A. Edwards, J.H. Johnson: Rational Chebyshev
  --       Approximations for the Inverse of the Error Function. Mathematics of
  --       Computation, Vol. 30, No. 136 (Oct. 1976), pp. 827-830. Table 59
  --      

  -- Based on: J.M. Blair, C.A. Edwards, J.H. Johnson: Rational Chebyshev
  --       Approximations for the Inverse of the Error Function. Mathematics of
  --       Computation, Vol. 30, No. 136 (Oct. 1976), pp. 827-830. Table 39
  --     

  -- Based on: J.M. Blair, C.A. Edwards, J.H. Johnson: Rational Chebyshev
  --       Approximations for the Inverse of the Error Function. Mathematics of
  --       Computation, Vol. 30, No. 136 (Oct. 1976), pp. 827-830. Table 18
  --     

   function erfcinv (a : double) return double;  -- /usr/local/cuda-8.0/include/math_functions.hpp:3607
   pragma Import (CPP, erfcinv, "_Z7erfcinvd");

  -- INDEFINITE  
  -- Infinity  
  -- Based on: J.M. Blair, C.A. Edwards, J.H. Johnson: Rational Chebyshev
  --       Approximations for the Inverse of the Error Function. Mathematics of
  --       Computation, Vol. 30, No. 136 (Oct. 1976), pp. 827-830. Table 59
  --     

  -- Based on: J.M. Blair, C.A. Edwards, J.H. Johnson: Rational Chebyshev
  --       Approximations for the Inverse of the Error Function. Mathematics of
  --       Computation, Vol. 30, No. 136 (Oct. 1976), pp. 827-830. Table 82
  --     

   function normcdfinv (a : double) return double;  -- /usr/local/cuda-8.0/include/math_functions.hpp:3690
   pragma Import (CPP, normcdfinv, "_Z10normcdfinvd");

   function normcdf (a : double) return double;  -- /usr/local/cuda-8.0/include/math_functions.hpp:3695
   pragma Import (CPP, normcdf, "_Z7normcdfd");

   function erfcx (a : double) return double;  -- /usr/local/cuda-8.0/include/math_functions.hpp:3717
   pragma Import (CPP, erfcx, "_Z5erfcxd");

  --  
  --     * This implementation of erfcx() is based on the algorithm in: M. M. 
  --     * Shepherd and J. G. Laframboise, "Chebyshev Approximation of (1 + 2x)
  --     * exp(x^2)erfc x in 0 <= x < INF", Mathematics of Computation, Vol. 
  --     * 36, No. 153, January 1981, pp. 249-253. For the core approximation,
  --     * the input domain [0,INF] is transformed via (x-k) / (x+k) where k is
  --     * a precision-dependent constant. Here, we choose k = 4.0, so the input 
  --     * domain [0, 27.3] is transformed into the core approximation domain 
  --     * [-1, 0.744409].   
  --      

  -- (1+2*x)*exp(x*x)*erfc(x)  
  -- t2 = (x-4.0)/(x+4.0), transforming [0,INF] to [-1,+1]  
  -- approximate on [-1, 0.744409]  
  -- (1+2*x)*exp(x*x)*erfc(x) / (1+2*x) = exp(x*x)*erfc(x)  
  -- asymptotic expansion for large aguments  
  -- erfcx(x) = 2*exp(x^2) - erfcx(|x|)  
   function rsqrtf (a : float) return float;  -- /usr/local/cuda-8.0/include/math_functions.hpp:3792
   pragma Import (CPP, rsqrtf, "_Z6rsqrtff");

   function rcbrtf (a : float) return float;  -- /usr/local/cuda-8.0/include/math_functions.hpp:3797
   pragma Import (CPP, rcbrtf, "_Z6rcbrtff");

   function sinpif (a : float) return float;  -- /usr/local/cuda-8.0/include/math_functions.hpp:3802
   pragma Import (CPP, sinpif, "_Z6sinpiff");

   function cospif (a : float) return float;  -- /usr/local/cuda-8.0/include/math_functions.hpp:3807
   pragma Import (CPP, cospif, "_Z6cospiff");

   procedure sincospif
     (a : float;
      sptr : access float;
      cptr : access float);  -- /usr/local/cuda-8.0/include/math_functions.hpp:3812
   pragma Import (CPP, sincospif, "_Z9sincospiffPfS_");

   function erfinvf (a : float) return float;  -- /usr/local/cuda-8.0/include/math_functions.hpp:3821
   pragma Import (CPP, erfinvf, "_Z7erfinvff");

   function erfcinvf (a : float) return float;  -- /usr/local/cuda-8.0/include/math_functions.hpp:3826
   pragma Import (CPP, erfcinvf, "_Z8erfcinvff");

   function normcdfinvf (a : float) return float;  -- /usr/local/cuda-8.0/include/math_functions.hpp:3831
   pragma Import (CPP, normcdfinvf, "_Z11normcdfinvff");

   function normcdff (a : float) return float;  -- /usr/local/cuda-8.0/include/math_functions.hpp:3836
   pragma Import (CPP, normcdff, "_Z8normcdfff");

   function erfcxf (a : float) return float;  -- /usr/local/cuda-8.0/include/math_functions.hpp:3841
   pragma Import (CPP, erfcxf, "_Z6erfcxff");

end math_functions_hpp;

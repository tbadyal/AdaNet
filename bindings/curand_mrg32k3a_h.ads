pragma Ada_2005;
pragma Style_Checks (Off);

with Interfaces.C; use Interfaces.C;

package curand_mrg32k3a_h is

  -- * Copyright 2010-2014 NVIDIA Corporation.  All rights reserved.
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

  -- base matrices  
  -- these are not actually used in the runtime code. They are    
  -- used in computing the skipahead matrices, and are included   
  -- for reference                                                
  --    double M1[3][3] = {        0.,       1.,       0.,
  --                               0.,       0.,       1.,
  --                         -810728., 1403580.,       0. };
  --    double M2[3][3] = {        0.,       1.,       0.,
  --                               0.,       0.,       1.,
  --                        -1370589.,       0.,  527612. };
  --     

  -- Base matrices to power 2 to the power n, n the first array index, from 0..63  
   mrg32k3aM1Host : aliased array (0 .. 63, 0 .. 2, 0 .. 2) of aliased double;  -- /usr/local/cuda-8.0/include/curand_mrg32k3a.h:393
   pragma Import (CPP, mrg32k3aM1Host, "_ZL14mrg32k3aM1Host");

   mrg32k3aM2Host : aliased array (0 .. 63, 0 .. 2, 0 .. 2) of aliased double;  -- /usr/local/cuda-8.0/include/curand_mrg32k3a.h:1039
   pragma Import (CPP, mrg32k3aM2Host, "_ZL14mrg32k3aM2Host");

  --Base matrices to power (2 to the power 76) to power 2 to power n + 1, n the first array index, from 0..63 
   mrg32k3aM1SubSeqHost : aliased array (0 .. 55, 0 .. 2, 0 .. 2) of aliased double;  -- /usr/local/cuda-8.0/include/curand_mrg32k3a.h:1622
   pragma Import (CPP, mrg32k3aM1SubSeqHost, "_ZL20mrg32k3aM1SubSeqHost");

   mrg32k3aM2SubSeqHost : aliased array (0 .. 55, 0 .. 2, 0 .. 2) of aliased double;  -- /usr/local/cuda-8.0/include/curand_mrg32k3a.h:2138
   pragma Import (CPP, mrg32k3aM2SubSeqHost, "_ZL20mrg32k3aM2SubSeqHost");

  --Base matrices to power (2 to the power 127) to power 2 to power n+1, n the first array index, from 0..63 
   mrg32k3aM1SeqHost : aliased array (0 .. 63, 0 .. 2, 0 .. 2) of aliased double;  -- /usr/local/cuda-8.0/include/curand_mrg32k3a.h:2721
   pragma Import (CPP, mrg32k3aM1SeqHost, "_ZL17mrg32k3aM1SeqHost");

   mrg32k3aM2SeqHost : aliased array (0 .. 63, 0 .. 2, 0 .. 2) of aliased double;  -- /usr/local/cuda-8.0/include/curand_mrg32k3a.h:3367
   pragma Import (CPP, mrg32k3aM2SeqHost, "_ZL17mrg32k3aM2SeqHost");

end curand_mrg32k3a_h;

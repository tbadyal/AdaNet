pragma Ada_2005;
pragma Style_Checks (Off);

with Interfaces.C; use Interfaces.C;
with System;
with stddef_h;
limited with stdio_h;

package cufftw_h is

   FFTW_FORWARD : constant := -1;  --  /usr/local/cuda-8.0/include/cufftw.h:67
   FFTW_INVERSE : constant := 1;  --  /usr/local/cuda-8.0/include/cufftw.h:68
   FFTW_BACKWARD : constant := 1;  --  /usr/local/cuda-8.0/include/cufftw.h:69

   FFTW_ESTIMATE : constant := 16#01#;  --  /usr/local/cuda-8.0/include/cufftw.h:73
   FFTW_MEASURE : constant := 16#02#;  --  /usr/local/cuda-8.0/include/cufftw.h:74
   FFTW_PATIENT : constant := 16#03#;  --  /usr/local/cuda-8.0/include/cufftw.h:75
   FFTW_EXHAUSTIVE : constant := 16#04#;  --  /usr/local/cuda-8.0/include/cufftw.h:76
   FFTW_WISDOM_ONLY : constant := 16#05#;  --  /usr/local/cuda-8.0/include/cufftw.h:77

   FFTW_DESTROY_INPUT : constant := 16#08#;  --  /usr/local/cuda-8.0/include/cufftw.h:81
   FFTW_PRESERVE_INPUT : constant := 16#0C#;  --  /usr/local/cuda-8.0/include/cufftw.h:82
   FFTW_UNALIGNED : constant := 16#10#;  --  /usr/local/cuda-8.0/include/cufftw.h:83

  -- Copyright 2005-2014 NVIDIA Corporation.  All rights reserved.
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

  --!
  --* \file cufftw.h
  --* \brief Public header file for the NVIDIA CUDA FFTW library (CUFFTW)
  -- 

  -- transform direction
  -- Planner flags
  --Algorithm restriction flags
  -- CUFFTW defines and supports the following data types
  -- note if complex.h has been included we use the C99 complex types
   type fftw_complex is array (0 .. 1) of aliased double;  -- /usr/local/cuda-8.0/include/cufftw.h:92

   type fftwf_complex is array (0 .. 1) of aliased float;  -- /usr/local/cuda-8.0/include/cufftw.h:93

   type fftw_plan is new System.Address;  -- /usr/local/cuda-8.0/include/cufftw.h:96

   type fftwf_plan is new System.Address;  -- /usr/local/cuda-8.0/include/cufftw.h:98

   type fftw_iodim is record
      n : aliased int;  -- /usr/local/cuda-8.0/include/cufftw.h:101
      c_is : aliased int;  -- /usr/local/cuda-8.0/include/cufftw.h:102
      os : aliased int;  -- /usr/local/cuda-8.0/include/cufftw.h:103
   end record;
   pragma Convention (C_Pass_By_Copy, fftw_iodim);  -- /usr/local/cuda-8.0/include/cufftw.h:104

   --  skipped anonymous struct anon_13

   subtype fftwf_iodim is fftw_iodim;

   type fftw_iodim64 is record
      n : aliased int;  -- /usr/local/cuda-8.0/include/cufftw.h:109
      c_is : aliased stddef_h.size_t;  -- /usr/local/cuda-8.0/include/cufftw.h:110
      os : aliased stddef_h.size_t;  -- /usr/local/cuda-8.0/include/cufftw.h:111
   end record;
   pragma Convention (C_Pass_By_Copy, fftw_iodim64);  -- /usr/local/cuda-8.0/include/cufftw.h:112

   --  skipped anonymous struct anon_14

   subtype fftwf_iodim64 is fftw_iodim64;

  -- CUFFTW defines and supports the following double precision APIs
   function fftw_plan_dft_1d
     (n : int;
      c_in : access fftw_complex;
      c_out : access fftw_complex;
      sign : int;
      flags : unsigned) return fftw_plan;  -- /usr/local/cuda-8.0/include/cufftw.h:120
   pragma Import (C, fftw_plan_dft_1d, "fftw_plan_dft_1d");

   function fftw_plan_dft_2d
     (n0 : int;
      n1 : int;
      c_in : access fftw_complex;
      c_out : access fftw_complex;
      sign : int;
      flags : unsigned) return fftw_plan;  -- /usr/local/cuda-8.0/include/cufftw.h:126
   pragma Import (C, fftw_plan_dft_2d, "fftw_plan_dft_2d");

   function fftw_plan_dft_3d
     (n0 : int;
      n1 : int;
      n2 : int;
      c_in : access fftw_complex;
      c_out : access fftw_complex;
      sign : int;
      flags : unsigned) return fftw_plan;  -- /usr/local/cuda-8.0/include/cufftw.h:133
   pragma Import (C, fftw_plan_dft_3d, "fftw_plan_dft_3d");

   function fftw_plan_dft
     (rank : int;
      n : access int;
      c_in : access fftw_complex;
      c_out : access fftw_complex;
      sign : int;
      flags : unsigned) return fftw_plan;  -- /usr/local/cuda-8.0/include/cufftw.h:141
   pragma Import (C, fftw_plan_dft, "fftw_plan_dft");

   function fftw_plan_dft_r2c_1d
     (n : int;
      c_in : access double;
      c_out : access fftw_complex;
      flags : unsigned) return fftw_plan;  -- /usr/local/cuda-8.0/include/cufftw.h:148
   pragma Import (C, fftw_plan_dft_r2c_1d, "fftw_plan_dft_r2c_1d");

   function fftw_plan_dft_r2c_2d
     (n0 : int;
      n1 : int;
      c_in : access double;
      c_out : access fftw_complex;
      flags : unsigned) return fftw_plan;  -- /usr/local/cuda-8.0/include/cufftw.h:153
   pragma Import (C, fftw_plan_dft_r2c_2d, "fftw_plan_dft_r2c_2d");

   function fftw_plan_dft_r2c_3d
     (n0 : int;
      n1 : int;
      n2 : int;
      c_in : access double;
      c_out : access fftw_complex;
      flags : unsigned) return fftw_plan;  -- /usr/local/cuda-8.0/include/cufftw.h:159
   pragma Import (C, fftw_plan_dft_r2c_3d, "fftw_plan_dft_r2c_3d");

   function fftw_plan_dft_r2c
     (rank : int;
      n : access int;
      c_in : access double;
      c_out : access fftw_complex;
      flags : unsigned) return fftw_plan;  -- /usr/local/cuda-8.0/include/cufftw.h:166
   pragma Import (C, fftw_plan_dft_r2c, "fftw_plan_dft_r2c");

   function fftw_plan_dft_c2r_1d
     (n : int;
      c_in : access fftw_complex;
      c_out : access double;
      flags : unsigned) return fftw_plan;  -- /usr/local/cuda-8.0/include/cufftw.h:172
   pragma Import (C, fftw_plan_dft_c2r_1d, "fftw_plan_dft_c2r_1d");

   function fftw_plan_dft_c2r_2d
     (n0 : int;
      n1 : int;
      c_in : access fftw_complex;
      c_out : access double;
      flags : unsigned) return fftw_plan;  -- /usr/local/cuda-8.0/include/cufftw.h:177
   pragma Import (C, fftw_plan_dft_c2r_2d, "fftw_plan_dft_c2r_2d");

   function fftw_plan_dft_c2r_3d
     (n0 : int;
      n1 : int;
      n2 : int;
      c_in : access fftw_complex;
      c_out : access double;
      flags : unsigned) return fftw_plan;  -- /usr/local/cuda-8.0/include/cufftw.h:183
   pragma Import (C, fftw_plan_dft_c2r_3d, "fftw_plan_dft_c2r_3d");

   function fftw_plan_dft_c2r
     (rank : int;
      n : access int;
      c_in : access fftw_complex;
      c_out : access double;
      flags : unsigned) return fftw_plan;  -- /usr/local/cuda-8.0/include/cufftw.h:190
   pragma Import (C, fftw_plan_dft_c2r, "fftw_plan_dft_c2r");

   function fftw_plan_many_dft
     (rank : int;
      n : access int;
      batch : int;
      c_in : access fftw_complex;
      inembed : access int;
      istride : int;
      idist : int;
      c_out : access fftw_complex;
      onembed : access int;
      ostride : int;
      odist : int;
      sign : int;
      flags : unsigned) return fftw_plan;  -- /usr/local/cuda-8.0/include/cufftw.h:197
   pragma Import (C, fftw_plan_many_dft, "fftw_plan_many_dft");

   function fftw_plan_many_dft_r2c
     (rank : int;
      n : access int;
      batch : int;
      c_in : access double;
      inembed : access int;
      istride : int;
      idist : int;
      c_out : access fftw_complex;
      onembed : access int;
      ostride : int;
      odist : int;
      flags : unsigned) return fftw_plan;  -- /usr/local/cuda-8.0/include/cufftw.h:206
   pragma Import (C, fftw_plan_many_dft_r2c, "fftw_plan_many_dft_r2c");

   function fftw_plan_many_dft_c2r
     (rank : int;
      n : access int;
      batch : int;
      c_in : access fftw_complex;
      inembed : access int;
      istride : int;
      idist : int;
      c_out : access double;
      onembed : access int;
      ostride : int;
      odist : int;
      flags : unsigned) return fftw_plan;  -- /usr/local/cuda-8.0/include/cufftw.h:215
   pragma Import (C, fftw_plan_many_dft_c2r, "fftw_plan_many_dft_c2r");

   function fftw_plan_guru_dft
     (rank : int;
      dims : access constant fftw_iodim;
      batch_rank : int;
      batch_dims : access constant fftw_iodim;
      c_in : access fftw_complex;
      c_out : access fftw_complex;
      sign : int;
      flags : unsigned) return fftw_plan;  -- /usr/local/cuda-8.0/include/cufftw.h:224
   pragma Import (C, fftw_plan_guru_dft, "fftw_plan_guru_dft");

   function fftw_plan_guru_dft_r2c
     (rank : int;
      dims : access constant fftw_iodim;
      batch_rank : int;
      batch_dims : access constant fftw_iodim;
      c_in : access double;
      c_out : access fftw_complex;
      flags : unsigned) return fftw_plan;  -- /usr/local/cuda-8.0/include/cufftw.h:229
   pragma Import (C, fftw_plan_guru_dft_r2c, "fftw_plan_guru_dft_r2c");

   function fftw_plan_guru_dft_c2r
     (rank : int;
      dims : access constant fftw_iodim;
      batch_rank : int;
      batch_dims : access constant fftw_iodim;
      c_in : access fftw_complex;
      c_out : access double;
      flags : unsigned) return fftw_plan;  -- /usr/local/cuda-8.0/include/cufftw.h:234
   pragma Import (C, fftw_plan_guru_dft_c2r, "fftw_plan_guru_dft_c2r");

   procedure fftw_execute (plan : fftw_plan);  -- /usr/local/cuda-8.0/include/cufftw.h:239
   pragma Import (C, fftw_execute, "fftw_execute");

   procedure fftw_execute_dft
     (plan : fftw_plan;
      idata : access fftw_complex;
      odata : access fftw_complex);  -- /usr/local/cuda-8.0/include/cufftw.h:241
   pragma Import (C, fftw_execute_dft, "fftw_execute_dft");

   procedure fftw_execute_dft_r2c
     (plan : fftw_plan;
      idata : access double;
      odata : access fftw_complex);  -- /usr/local/cuda-8.0/include/cufftw.h:245
   pragma Import (C, fftw_execute_dft_r2c, "fftw_execute_dft_r2c");

   procedure fftw_execute_dft_c2r
     (plan : fftw_plan;
      idata : access fftw_complex;
      odata : access double);  -- /usr/local/cuda-8.0/include/cufftw.h:249
   pragma Import (C, fftw_execute_dft_c2r, "fftw_execute_dft_c2r");

  -- CUFFTW defines and supports the following single precision APIs
   function fftwf_plan_dft_1d
     (n : int;
      c_in : access fftwf_complex;
      c_out : access fftwf_complex;
      sign : int;
      flags : unsigned) return fftwf_plan;  -- /usr/local/cuda-8.0/include/cufftw.h:256
   pragma Import (C, fftwf_plan_dft_1d, "fftwf_plan_dft_1d");

   function fftwf_plan_dft_2d
     (n0 : int;
      n1 : int;
      c_in : access fftwf_complex;
      c_out : access fftwf_complex;
      sign : int;
      flags : unsigned) return fftwf_plan;  -- /usr/local/cuda-8.0/include/cufftw.h:262
   pragma Import (C, fftwf_plan_dft_2d, "fftwf_plan_dft_2d");

   function fftwf_plan_dft_3d
     (n0 : int;
      n1 : int;
      n2 : int;
      c_in : access fftwf_complex;
      c_out : access fftwf_complex;
      sign : int;
      flags : unsigned) return fftwf_plan;  -- /usr/local/cuda-8.0/include/cufftw.h:269
   pragma Import (C, fftwf_plan_dft_3d, "fftwf_plan_dft_3d");

   function fftwf_plan_dft
     (rank : int;
      n : access int;
      c_in : access fftwf_complex;
      c_out : access fftwf_complex;
      sign : int;
      flags : unsigned) return fftwf_plan;  -- /usr/local/cuda-8.0/include/cufftw.h:277
   pragma Import (C, fftwf_plan_dft, "fftwf_plan_dft");

   function fftwf_plan_dft_r2c_1d
     (n : int;
      c_in : access float;
      c_out : access fftwf_complex;
      flags : unsigned) return fftwf_plan;  -- /usr/local/cuda-8.0/include/cufftw.h:284
   pragma Import (C, fftwf_plan_dft_r2c_1d, "fftwf_plan_dft_r2c_1d");

   function fftwf_plan_dft_r2c_2d
     (n0 : int;
      n1 : int;
      c_in : access float;
      c_out : access fftwf_complex;
      flags : unsigned) return fftwf_plan;  -- /usr/local/cuda-8.0/include/cufftw.h:289
   pragma Import (C, fftwf_plan_dft_r2c_2d, "fftwf_plan_dft_r2c_2d");

   function fftwf_plan_dft_r2c_3d
     (n0 : int;
      n1 : int;
      n2 : int;
      c_in : access float;
      c_out : access fftwf_complex;
      flags : unsigned) return fftwf_plan;  -- /usr/local/cuda-8.0/include/cufftw.h:295
   pragma Import (C, fftwf_plan_dft_r2c_3d, "fftwf_plan_dft_r2c_3d");

   function fftwf_plan_dft_r2c
     (rank : int;
      n : access int;
      c_in : access float;
      c_out : access fftwf_complex;
      flags : unsigned) return fftwf_plan;  -- /usr/local/cuda-8.0/include/cufftw.h:302
   pragma Import (C, fftwf_plan_dft_r2c, "fftwf_plan_dft_r2c");

   function fftwf_plan_dft_c2r_1d
     (n : int;
      c_in : access fftwf_complex;
      c_out : access float;
      flags : unsigned) return fftwf_plan;  -- /usr/local/cuda-8.0/include/cufftw.h:308
   pragma Import (C, fftwf_plan_dft_c2r_1d, "fftwf_plan_dft_c2r_1d");

   function fftwf_plan_dft_c2r_2d
     (n0 : int;
      n1 : int;
      c_in : access fftwf_complex;
      c_out : access float;
      flags : unsigned) return fftwf_plan;  -- /usr/local/cuda-8.0/include/cufftw.h:313
   pragma Import (C, fftwf_plan_dft_c2r_2d, "fftwf_plan_dft_c2r_2d");

   function fftwf_plan_dft_c2r_3d
     (n0 : int;
      n1 : int;
      n2 : int;
      c_in : access fftwf_complex;
      c_out : access float;
      flags : unsigned) return fftwf_plan;  -- /usr/local/cuda-8.0/include/cufftw.h:319
   pragma Import (C, fftwf_plan_dft_c2r_3d, "fftwf_plan_dft_c2r_3d");

   function fftwf_plan_dft_c2r
     (rank : int;
      n : access int;
      c_in : access fftwf_complex;
      c_out : access float;
      flags : unsigned) return fftwf_plan;  -- /usr/local/cuda-8.0/include/cufftw.h:326
   pragma Import (C, fftwf_plan_dft_c2r, "fftwf_plan_dft_c2r");

   function fftwf_plan_many_dft
     (rank : int;
      n : access int;
      batch : int;
      c_in : access fftwf_complex;
      inembed : access int;
      istride : int;
      idist : int;
      c_out : access fftwf_complex;
      onembed : access int;
      ostride : int;
      odist : int;
      sign : int;
      flags : unsigned) return fftwf_plan;  -- /usr/local/cuda-8.0/include/cufftw.h:332
   pragma Import (C, fftwf_plan_many_dft, "fftwf_plan_many_dft");

   function fftwf_plan_many_dft_r2c
     (rank : int;
      n : access int;
      batch : int;
      c_in : access float;
      inembed : access int;
      istride : int;
      idist : int;
      c_out : access fftwf_complex;
      onembed : access int;
      ostride : int;
      odist : int;
      flags : unsigned) return fftwf_plan;  -- /usr/local/cuda-8.0/include/cufftw.h:341
   pragma Import (C, fftwf_plan_many_dft_r2c, "fftwf_plan_many_dft_r2c");

   function fftwf_plan_many_dft_c2r
     (rank : int;
      n : access int;
      batch : int;
      c_in : access fftwf_complex;
      inembed : access int;
      istride : int;
      idist : int;
      c_out : access float;
      onembed : access int;
      ostride : int;
      odist : int;
      flags : unsigned) return fftwf_plan;  -- /usr/local/cuda-8.0/include/cufftw.h:350
   pragma Import (C, fftwf_plan_many_dft_c2r, "fftwf_plan_many_dft_c2r");

   function fftwf_plan_guru_dft
     (rank : int;
      dims : access constant fftwf_iodim;
      batch_rank : int;
      batch_dims : access constant fftwf_iodim;
      c_in : access fftwf_complex;
      c_out : access fftwf_complex;
      sign : int;
      flags : unsigned) return fftwf_plan;  -- /usr/local/cuda-8.0/include/cufftw.h:359
   pragma Import (C, fftwf_plan_guru_dft, "fftwf_plan_guru_dft");

   function fftwf_plan_guru_dft_r2c
     (rank : int;
      dims : access constant fftwf_iodim;
      batch_rank : int;
      batch_dims : access constant fftwf_iodim;
      c_in : access float;
      c_out : access fftwf_complex;
      flags : unsigned) return fftwf_plan;  -- /usr/local/cuda-8.0/include/cufftw.h:364
   pragma Import (C, fftwf_plan_guru_dft_r2c, "fftwf_plan_guru_dft_r2c");

   function fftwf_plan_guru_dft_c2r
     (rank : int;
      dims : access constant fftwf_iodim;
      batch_rank : int;
      batch_dims : access constant fftwf_iodim;
      c_in : access fftwf_complex;
      c_out : access float;
      flags : unsigned) return fftwf_plan;  -- /usr/local/cuda-8.0/include/cufftw.h:369
   pragma Import (C, fftwf_plan_guru_dft_c2r, "fftwf_plan_guru_dft_c2r");

   procedure fftwf_execute (plan : fftw_plan);  -- /usr/local/cuda-8.0/include/cufftw.h:374
   pragma Import (C, fftwf_execute, "fftwf_execute");

   procedure fftwf_execute_dft
     (plan : fftwf_plan;
      idata : access fftwf_complex;
      odata : access fftwf_complex);  -- /usr/local/cuda-8.0/include/cufftw.h:376
   pragma Import (C, fftwf_execute_dft, "fftwf_execute_dft");

   procedure fftwf_execute_dft_r2c
     (plan : fftwf_plan;
      idata : access float;
      odata : access fftwf_complex);  -- /usr/local/cuda-8.0/include/cufftw.h:380
   pragma Import (C, fftwf_execute_dft_r2c, "fftwf_execute_dft_r2c");

   procedure fftwf_execute_dft_c2r
     (plan : fftwf_plan;
      idata : access fftwf_complex;
      odata : access float);  -- /usr/local/cuda-8.0/include/cufftw.h:384
   pragma Import (C, fftwf_execute_dft_c2r, "fftwf_execute_dft_c2r");

  -- CUFFTW defines and supports the following support APIs
   function fftw_malloc (n : stddef_h.size_t) return System.Address;  -- /usr/local/cuda-8.0/include/cufftw.h:395
   pragma Import (C, fftw_malloc, "fftw_malloc");

   function fftwf_malloc (n : stddef_h.size_t) return System.Address;  -- /usr/local/cuda-8.0/include/cufftw.h:397
   pragma Import (C, fftwf_malloc, "fftwf_malloc");

   procedure fftw_free (pointer : System.Address);  -- /usr/local/cuda-8.0/include/cufftw.h:399
   pragma Import (C, fftw_free, "fftw_free");

   procedure fftwf_free (pointer : System.Address);  -- /usr/local/cuda-8.0/include/cufftw.h:401
   pragma Import (C, fftwf_free, "fftwf_free");

   procedure fftw_export_wisdom_to_file (output_file : access stdio_h.FILE);  -- /usr/local/cuda-8.0/include/cufftw.h:403
   pragma Import (C, fftw_export_wisdom_to_file, "fftw_export_wisdom_to_file");

   procedure fftwf_export_wisdom_to_file (output_file : access stdio_h.FILE);  -- /usr/local/cuda-8.0/include/cufftw.h:405
   pragma Import (C, fftwf_export_wisdom_to_file, "fftwf_export_wisdom_to_file");

   procedure fftw_import_wisdom_from_file (input_file : access stdio_h.FILE);  -- /usr/local/cuda-8.0/include/cufftw.h:407
   pragma Import (C, fftw_import_wisdom_from_file, "fftw_import_wisdom_from_file");

   procedure fftwf_import_wisdom_from_file (input_file : access stdio_h.FILE);  -- /usr/local/cuda-8.0/include/cufftw.h:409
   pragma Import (C, fftwf_import_wisdom_from_file, "fftwf_import_wisdom_from_file");

   procedure fftw_print_plan (plan : fftw_plan);  -- /usr/local/cuda-8.0/include/cufftw.h:411
   pragma Import (C, fftw_print_plan, "fftw_print_plan");

   procedure fftwf_print_plan (plan : fftwf_plan);  -- /usr/local/cuda-8.0/include/cufftw.h:413
   pragma Import (C, fftwf_print_plan, "fftwf_print_plan");

   procedure fftw_set_timelimit (seconds : double);  -- /usr/local/cuda-8.0/include/cufftw.h:415
   pragma Import (C, fftw_set_timelimit, "fftw_set_timelimit");

   procedure fftwf_set_timelimit (seconds : double);  -- /usr/local/cuda-8.0/include/cufftw.h:417
   pragma Import (C, fftwf_set_timelimit, "fftwf_set_timelimit");

   function fftw_cost (plan : fftw_plan) return double;  -- /usr/local/cuda-8.0/include/cufftw.h:419
   pragma Import (C, fftw_cost, "fftw_cost");

   function fftwf_cost (plan : fftw_plan) return double;  -- /usr/local/cuda-8.0/include/cufftw.h:421
   pragma Import (C, fftwf_cost, "fftwf_cost");

   procedure fftw_flops
     (plan : fftw_plan;
      add : access double;
      mul : access double;
      fma : access double);  -- /usr/local/cuda-8.0/include/cufftw.h:423
   pragma Import (C, fftw_flops, "fftw_flops");

   procedure fftwf_flops
     (plan : fftw_plan;
      add : access double;
      mul : access double;
      fma : access double);  -- /usr/local/cuda-8.0/include/cufftw.h:425
   pragma Import (C, fftwf_flops, "fftwf_flops");

   procedure fftw_destroy_plan (plan : fftw_plan);  -- /usr/local/cuda-8.0/include/cufftw.h:427
   pragma Import (C, fftw_destroy_plan, "fftw_destroy_plan");

   procedure fftwf_destroy_plan (plan : fftwf_plan);  -- /usr/local/cuda-8.0/include/cufftw.h:429
   pragma Import (C, fftwf_destroy_plan, "fftwf_destroy_plan");

   procedure fftw_cleanup;  -- /usr/local/cuda-8.0/include/cufftw.h:431
   pragma Import (C, fftw_cleanup, "fftw_cleanup");

   procedure fftwf_cleanup;  -- /usr/local/cuda-8.0/include/cufftw.h:433
   pragma Import (C, fftwf_cleanup, "fftwf_cleanup");

end cufftw_h;

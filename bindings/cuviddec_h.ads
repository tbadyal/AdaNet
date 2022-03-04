pragma Ada_2005;
pragma Style_Checks (Off);

with Interfaces.C; use Interfaces.C;
with System;
with Interfaces.C.Extensions;
with cuda_h;

package cuviddec_h is

   I_VOP : constant := 0;  --  /usr/local/cuda-8.0/include/cuviddec.h:286
   P_VOP : constant := 1;  --  /usr/local/cuda-8.0/include/cuviddec.h:287
   B_VOP : constant := 2;  --  /usr/local/cuda-8.0/include/cuviddec.h:288
   S_VOP : constant := 3;  --  /usr/local/cuda-8.0/include/cuviddec.h:289
   --  unsupported macro: cuvidMapVideoFrame cuvidMapVideoFrame64
   --  unsupported macro: cuvidUnmapVideoFrame cuvidUnmapVideoFrame64

  -- * Copyright 1993-2008 NVIDIA Corporation.  All rights reserved.
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

   type CUvideodecoder is new System.Address;  -- /usr/local/cuda-8.0/include/cuviddec.h:53

   --  skipped empty struct u_CUcontextlock_st

   type CUvideoctxlock is new System.Address;  -- /usr/local/cuda-8.0/include/cuviddec.h:54

   subtype cudaVideoCodec_enum is unsigned;
   cudaVideoCodec_MPEG1 : constant cudaVideoCodec_enum := 0;
   cudaVideoCodec_MPEG2 : constant cudaVideoCodec_enum := 1;
   cudaVideoCodec_MPEG4 : constant cudaVideoCodec_enum := 2;
   cudaVideoCodec_VC1 : constant cudaVideoCodec_enum := 3;
   cudaVideoCodec_H264 : constant cudaVideoCodec_enum := 4;
   cudaVideoCodec_JPEG : constant cudaVideoCodec_enum := 5;
   cudaVideoCodec_H264_SVC : constant cudaVideoCodec_enum := 6;
   cudaVideoCodec_H264_MVC : constant cudaVideoCodec_enum := 7;
   cudaVideoCodec_HEVC : constant cudaVideoCodec_enum := 8;
   cudaVideoCodec_NumCodecs : constant cudaVideoCodec_enum := 9;
   cudaVideoCodec_YUV420 : constant cudaVideoCodec_enum := 1230591318;
   cudaVideoCodec_YV12 : constant cudaVideoCodec_enum := 1498820914;
   cudaVideoCodec_NV12 : constant cudaVideoCodec_enum := 1314271538;
   cudaVideoCodec_YUYV : constant cudaVideoCodec_enum := 1498765654;
   cudaVideoCodec_UYVY : constant cudaVideoCodec_enum := 1431918169;  -- /usr/local/cuda-8.0/include/cuviddec.h:56

  -- Uncompressed YUV
  -- Y,U,V (4:2:0)
  -- Y,V,U (4:2:0)
  -- Y,UV  (4:2:0)
  -- YUYV/YUY2 (4:2:2)
  -- UYVY (4:2:2)
   subtype cudaVideoCodec is cudaVideoCodec_enum;

   type cudaVideoSurfaceFormat_enum is 
     (cudaVideoSurfaceFormat_NV12);
   pragma Convention (C, cudaVideoSurfaceFormat_enum);  -- /usr/local/cuda-8.0/include/cuviddec.h:75

  -- NV12 (currently the only supported output format)
   subtype cudaVideoSurfaceFormat is cudaVideoSurfaceFormat_enum;

   type cudaVideoDeinterlaceMode_enum is 
     (cudaVideoDeinterlaceMode_Weave,
      cudaVideoDeinterlaceMode_Bob,
      cudaVideoDeinterlaceMode_Adaptive);
   pragma Convention (C, cudaVideoDeinterlaceMode_enum);  -- /usr/local/cuda-8.0/include/cuviddec.h:79

  -- Weave both fields (no deinterlacing)
  -- Drop one field
  -- Adaptive deinterlacing
   subtype cudaVideoDeinterlaceMode is cudaVideoDeinterlaceMode_enum;

   type cudaVideoChromaFormat_enum is 
     (cudaVideoChromaFormat_Monochrome,
      cudaVideoChromaFormat_420,
      cudaVideoChromaFormat_422,
      cudaVideoChromaFormat_444);
   pragma Convention (C, cudaVideoChromaFormat_enum);  -- /usr/local/cuda-8.0/include/cuviddec.h:85

   subtype cudaVideoChromaFormat is cudaVideoChromaFormat_enum;

   subtype cudaVideoCreateFlags_enum is unsigned;
   cudaVideoCreate_Default : constant cudaVideoCreateFlags_enum := 0;
   cudaVideoCreate_PreferCUDA : constant cudaVideoCreateFlags_enum := 1;
   cudaVideoCreate_PreferDXVA : constant cudaVideoCreateFlags_enum := 2;
   cudaVideoCreate_PreferCUVID : constant cudaVideoCreateFlags_enum := 4;  -- /usr/local/cuda-8.0/include/cuviddec.h:92

  -- Default operation mode: use dedicated video engines
  -- Use a CUDA-based decoder if faster than dedicated engines (requires a valid vidLock object for multi-threading)
  -- Go through DXVA internally if possible (requires D3D9 interop)
  -- Use dedicated video engines directly
   subtype cudaVideoCreateFlags is cudaVideoCreateFlags_enum;

  -- Decoding
  -- Coded Sequence Width
   type u_CUVIDDECODECREATEINFO;
   type anon_27 is record
      left : aliased short;  -- /usr/local/cuda-8.0/include/cuviddec.h:111
      top : aliased short;  -- /usr/local/cuda-8.0/include/cuviddec.h:112
      right : aliased short;  -- /usr/local/cuda-8.0/include/cuviddec.h:113
      bottom : aliased short;  -- /usr/local/cuda-8.0/include/cuviddec.h:114
   end record;
   pragma Convention (C_Pass_By_Copy, anon_27);
   type anon_28 is record
      left : aliased short;  -- /usr/local/cuda-8.0/include/cuviddec.h:124
      top : aliased short;  -- /usr/local/cuda-8.0/include/cuviddec.h:125
      right : aliased short;  -- /usr/local/cuda-8.0/include/cuviddec.h:126
      bottom : aliased short;  -- /usr/local/cuda-8.0/include/cuviddec.h:127
   end record;
   pragma Convention (C_Pass_By_Copy, anon_28);
   type u_CUVIDDECODECREATEINFO_Reserved1_array is array (0 .. 4) of aliased unsigned_long;
   type u_CUVIDDECODECREATEINFO_Reserved2_array is array (0 .. 4) of aliased unsigned_long;
   type u_CUVIDDECODECREATEINFO is record
      ulWidth : aliased unsigned_long;  -- /usr/local/cuda-8.0/include/cuviddec.h:103
      ulHeight : aliased unsigned_long;  -- /usr/local/cuda-8.0/include/cuviddec.h:104
      ulNumDecodeSurfaces : aliased unsigned_long;  -- /usr/local/cuda-8.0/include/cuviddec.h:105
      CodecType : aliased cudaVideoCodec;  -- /usr/local/cuda-8.0/include/cuviddec.h:106
      ChromaFormat : aliased cudaVideoChromaFormat;  -- /usr/local/cuda-8.0/include/cuviddec.h:107
      ulCreationFlags : aliased unsigned_long;  -- /usr/local/cuda-8.0/include/cuviddec.h:108
      Reserved1 : aliased u_CUVIDDECODECREATEINFO_Reserved1_array;  -- /usr/local/cuda-8.0/include/cuviddec.h:109
      display_area : aliased anon_27;  -- /usr/local/cuda-8.0/include/cuviddec.h:115
      OutputFormat : aliased cudaVideoSurfaceFormat;  -- /usr/local/cuda-8.0/include/cuviddec.h:117
      DeinterlaceMode : aliased cudaVideoDeinterlaceMode;  -- /usr/local/cuda-8.0/include/cuviddec.h:118
      ulTargetWidth : aliased unsigned_long;  -- /usr/local/cuda-8.0/include/cuviddec.h:119
      ulTargetHeight : aliased unsigned_long;  -- /usr/local/cuda-8.0/include/cuviddec.h:120
      ulNumOutputSurfaces : aliased unsigned_long;  -- /usr/local/cuda-8.0/include/cuviddec.h:121
      vidLock : CUvideoctxlock;  -- /usr/local/cuda-8.0/include/cuviddec.h:122
      target_rect : aliased anon_28;  -- /usr/local/cuda-8.0/include/cuviddec.h:128
      Reserved2 : aliased u_CUVIDDECODECREATEINFO_Reserved2_array;  -- /usr/local/cuda-8.0/include/cuviddec.h:129
   end record;
   pragma Convention (C_Pass_By_Copy, u_CUVIDDECODECREATEINFO);  -- /usr/local/cuda-8.0/include/cuviddec.h:100

  -- Coded Sequence Height
  -- Maximum number of internal decode surfaces
  -- cudaVideoCodec_XXX
  -- cudaVideoChromaFormat_XXX (only 4:2:0 is currently supported)
  -- Decoder creation flags (cudaVideoCreateFlags_XXX)
  -- Reserved for future use - set to zero
  -- area of the frame that should be displayed
  -- Output format
  -- cudaVideoSurfaceFormat_XXX
  -- cudaVideoDeinterlaceMode_XXX
  -- Post-processed Output Width 
  -- Post-processed Output Height
  -- Maximum number of output surfaces simultaneously mapped
  -- If non-NULL, context lock used for synchronizing ownership of the cuda context
  -- target rectangle in the output frame (for aspect ratio conversion)
  -- if a null rectangle is specified, {0,0,ulTargetWidth,ulTargetHeight} will be used
  -- Reserved for future use - set to zero
   subtype CUVIDDECODECREATEINFO is u_CUVIDDECODECREATEINFO;

  --//////////////////////////////////////////////////////////////////////////////////////////////
  -- H.264 Picture Parameters
  -- picture index of reference frame
   type u_CUVIDH264DPBENTRY_FieldOrderCnt_array is array (0 .. 1) of aliased int;
   type u_CUVIDH264DPBENTRY is record
      PicIdx : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:140
      FrameIdx : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:141
      is_long_term : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:142
      not_existing : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:143
      used_for_reference : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:144
      FieldOrderCnt : aliased u_CUVIDH264DPBENTRY_FieldOrderCnt_array;  -- /usr/local/cuda-8.0/include/cuviddec.h:145
   end record;
   pragma Convention (C_Pass_By_Copy, u_CUVIDH264DPBENTRY);  -- /usr/local/cuda-8.0/include/cuviddec.h:138

  -- frame_num(short-term) or LongTermFrameIdx(long-term)
  -- 0=short term reference, 1=long term reference
  -- non-existing reference frame (corresponding PicIdx should be set to -1)
  -- 0=unused, 1=top_field, 2=bottom_field, 3=both_fields
  -- field order count of top and bottom fields
   subtype CUVIDH264DPBENTRY is u_CUVIDH264DPBENTRY;

   type u_CUVIDH264MVCEXT_InterViewRefsL0_array is array (0 .. 15) of aliased int;
   type u_CUVIDH264MVCEXT_InterViewRefsL1_array is array (0 .. 15) of aliased int;
   type u_CUVIDH264MVCEXT is record
      num_views_minus1 : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:150
      view_id : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:151
      inter_view_flag : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:152
      num_inter_view_refs_l0 : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:153
      num_inter_view_refs_l1 : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:154
      MVCReserved8Bits : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:155
      InterViewRefsL0 : aliased u_CUVIDH264MVCEXT_InterViewRefsL0_array;  -- /usr/local/cuda-8.0/include/cuviddec.h:156
      InterViewRefsL1 : aliased u_CUVIDH264MVCEXT_InterViewRefsL1_array;  -- /usr/local/cuda-8.0/include/cuviddec.h:157
   end record;
   pragma Convention (C_Pass_By_Copy, u_CUVIDH264MVCEXT);  -- /usr/local/cuda-8.0/include/cuviddec.h:148

   subtype CUVIDH264MVCEXT is u_CUVIDH264MVCEXT;

   type u_CUVIDPICPARAMS;
   type u_CUVIDH264SVCEXT is record
      profile_idc : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:162
      level_idc : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:163
      DQId : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:164
      DQIdMax : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:165
      disable_inter_layer_deblocking_filter_idc : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:166
      ref_layer_chroma_phase_y_plus1 : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:167
      inter_layer_slice_alpha_c0_offset_div2 : aliased signed_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:168
      inter_layer_slice_beta_offset_div2 : aliased signed_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:169
      DPBEntryValidFlag : aliased unsigned_short;  -- /usr/local/cuda-8.0/include/cuviddec.h:171
      inter_layer_deblocking_filter_control_present_flag : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:172
      extended_spatial_scalability_idc : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:173
      adaptive_tcoeff_level_prediction_flag : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:174
      slice_header_restriction_flag : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:175
      chroma_phase_x_plus1_flag : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:176
      chroma_phase_y_plus1 : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:177
      tcoeff_level_prediction_flag : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:179
      constrained_intra_resampling_flag : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:180
      ref_layer_chroma_phase_x_plus1_flag : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:181
      store_ref_base_pic_flag : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:182
      Reserved8BitsA : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:183
      Reserved8BitsB : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:184
      scaled_ref_layer_left_offset : aliased short;  -- /usr/local/cuda-8.0/include/cuviddec.h:188
      scaled_ref_layer_top_offset : aliased short;  -- /usr/local/cuda-8.0/include/cuviddec.h:189
      scaled_ref_layer_right_offset : aliased short;  -- /usr/local/cuda-8.0/include/cuviddec.h:190
      scaled_ref_layer_bottom_offset : aliased short;  -- /usr/local/cuda-8.0/include/cuviddec.h:191
      Reserved16Bits : aliased unsigned_short;  -- /usr/local/cuda-8.0/include/cuviddec.h:192
      pNextLayer : access u_CUVIDPICPARAMS;  -- /usr/local/cuda-8.0/include/cuviddec.h:193
      bRefBaseLayer : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:194
   end record;
   pragma Convention (C_Pass_By_Copy, u_CUVIDH264SVCEXT);  -- /usr/local/cuda-8.0/include/cuviddec.h:160

  -- For the 4 scaled_ref_layer_XX fields below,
  -- if (extended_spatial_scalability_idc == 1), SPS field, G.7.3.2.1.4, add prefix "seq_"
  -- if (extended_spatial_scalability_idc == 2), SLH field, G.7.3.3.4, 
  -- Points to the picparams for the next layer to be decoded. Linked list ends at the target layer.
  -- whether to store ref base pic
   subtype CUVIDH264SVCEXT is u_CUVIDH264SVCEXT;

  -- SPS
   type u_CUVIDH264PICPARAMS;
   type anon_29 (discr : unsigned := 0) is record
      case discr is
         when 0 =>
            slice_group_map_addr : aliased Extensions.unsigned_long_long;  -- /usr/local/cuda-8.0/include/cuviddec.h:242
         when others =>
            pMb2SliceGroupMap : access unsigned_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:243
      end case;
   end record;
   pragma Convention (C_Pass_By_Copy, anon_29);
   pragma Unchecked_Union (anon_29);type anon_30 (discr : unsigned := 0) is record
      case discr is
         when 0 =>
            mvcext : aliased CUVIDH264MVCEXT;  -- /usr/local/cuda-8.0/include/cuviddec.h:249
         when others =>
            svcext : aliased CUVIDH264SVCEXT;  -- /usr/local/cuda-8.0/include/cuviddec.h:250
      end case;
   end record;
   pragma Convention (C_Pass_By_Copy, anon_30);
   pragma Unchecked_Union (anon_30);type u_CUVIDH264PICPARAMS_CurrFieldOrderCnt_array is array (0 .. 1) of aliased int;
   type u_CUVIDH264PICPARAMS_dpb_array is array (0 .. 15) of aliased CUVIDH264DPBENTRY;
   type u_CUVIDH264PICPARAMS_WeightScale4x4_array is array (0 .. 5, 0 .. 15) of aliased unsigned_char;
   type u_CUVIDH264PICPARAMS_WeightScale8x8_array is array (0 .. 1, 0 .. 63) of aliased unsigned_char;
   type u_CUVIDH264PICPARAMS_Reserved_array is array (0 .. 11) of aliased unsigned;
   type u_CUVIDH264PICPARAMS is record
      log2_max_frame_num_minus4 : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:200
      pic_order_cnt_type : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:201
      log2_max_pic_order_cnt_lsb_minus4 : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:202
      delta_pic_order_always_zero_flag : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:203
      frame_mbs_only_flag : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:204
      direct_8x8_inference_flag : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:205
      num_ref_frames : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:206
      residual_colour_transform_flag : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:207
      bit_depth_luma_minus8 : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:208
      bit_depth_chroma_minus8 : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:209
      qpprime_y_zero_transform_bypass_flag : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:210
      entropy_coding_mode_flag : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:212
      pic_order_present_flag : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:213
      num_ref_idx_l0_active_minus1 : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:214
      num_ref_idx_l1_active_minus1 : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:215
      weighted_pred_flag : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:216
      weighted_bipred_idc : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:217
      pic_init_qp_minus26 : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:218
      deblocking_filter_control_present_flag : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:219
      redundant_pic_cnt_present_flag : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:220
      transform_8x8_mode_flag : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:221
      MbaffFrameFlag : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:222
      constrained_intra_pred_flag : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:223
      chroma_qp_index_offset : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:224
      second_chroma_qp_index_offset : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:225
      ref_pic_flag : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:226
      frame_num : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:227
      CurrFieldOrderCnt : aliased u_CUVIDH264PICPARAMS_CurrFieldOrderCnt_array;  -- /usr/local/cuda-8.0/include/cuviddec.h:228
      dpb : aliased u_CUVIDH264PICPARAMS_dpb_array;  -- /usr/local/cuda-8.0/include/cuviddec.h:230
      WeightScale4x4 : aliased u_CUVIDH264PICPARAMS_WeightScale4x4_array;  -- /usr/local/cuda-8.0/include/cuviddec.h:232
      WeightScale8x8 : aliased u_CUVIDH264PICPARAMS_WeightScale8x8_array;  -- /usr/local/cuda-8.0/include/cuviddec.h:233
      fmo_aso_enable : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:235
      num_slice_groups_minus1 : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:236
      slice_group_map_type : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:237
      pic_init_qs_minus26 : aliased signed_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:238
      slice_group_change_rate_minus1 : aliased unsigned;  -- /usr/local/cuda-8.0/include/cuviddec.h:239
      fmo : aliased anon_29;  -- /usr/local/cuda-8.0/include/cuviddec.h:244
      Reserved : aliased u_CUVIDH264PICPARAMS_Reserved_array;  -- /usr/local/cuda-8.0/include/cuviddec.h:245
      field_39 : aliased anon_30;
   end record;
   pragma Convention (C_Pass_By_Copy, u_CUVIDH264PICPARAMS);  -- /usr/local/cuda-8.0/include/cuviddec.h:197

  -- NOTE: shall meet level 4.1 restrictions
  -- Must be 0 (only 8-bit supported)
  -- Must be 0 (only 8-bit supported)
  -- PPS
  -- DPB
  -- List of reference frames within the DPB
  -- Quantization Matrices (raster-order)
  -- FMO/ASO
  -- SVC/MVC
   subtype CUVIDH264PICPARAMS is u_CUVIDH264PICPARAMS;

  --//////////////////////////////////////////////////////////////////////////////////////////////
  -- MPEG-2 Picture Parameters
  -- Picture index of forward reference (P/B-frames)
   type u_CUVIDMPEG2PICPARAMS_f_code_array is array (0 .. 1, 0 .. 1) of aliased int;
   type u_CUVIDMPEG2PICPARAMS_QuantMatrixIntra_array is array (0 .. 63) of aliased unsigned_char;
   type u_CUVIDMPEG2PICPARAMS_QuantMatrixInter_array is array (0 .. 63) of aliased unsigned_char;
   type u_CUVIDMPEG2PICPARAMS is record
      ForwardRefIdx : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:262
      BackwardRefIdx : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:263
      picture_coding_type : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:264
      full_pel_forward_vector : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:265
      full_pel_backward_vector : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:266
      f_code : aliased u_CUVIDMPEG2PICPARAMS_f_code_array;  -- /usr/local/cuda-8.0/include/cuviddec.h:267
      intra_dc_precision : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:268
      frame_pred_frame_dct : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:269
      concealment_motion_vectors : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:270
      q_scale_type : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:271
      intra_vlc_format : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:272
      alternate_scan : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:273
      top_field_first : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:274
      QuantMatrixIntra : aliased u_CUVIDMPEG2PICPARAMS_QuantMatrixIntra_array;  -- /usr/local/cuda-8.0/include/cuviddec.h:276
      QuantMatrixInter : aliased u_CUVIDMPEG2PICPARAMS_QuantMatrixInter_array;  -- /usr/local/cuda-8.0/include/cuviddec.h:277
   end record;
   pragma Convention (C_Pass_By_Copy, u_CUVIDMPEG2PICPARAMS);  -- /usr/local/cuda-8.0/include/cuviddec.h:260

  -- Picture index of backward reference (B-frames)
  -- Quantization matrices (raster order)
   subtype CUVIDMPEG2PICPARAMS is u_CUVIDMPEG2PICPARAMS;

  --//////////////////////////////////////////////////////////////////////////////////////////////
  -- MPEG-4 Picture Parameters
  -- MPEG-4 has VOP types instead of Picture types
  -- Picture index of forward reference (P/B-frames)
   type u_CUVIDMPEG4PICPARAMS_trd_array is array (0 .. 1) of aliased int;
   type u_CUVIDMPEG4PICPARAMS_trb_array is array (0 .. 1) of aliased int;
   type u_CUVIDMPEG4PICPARAMS_QuantMatrixIntra_array is array (0 .. 63) of aliased unsigned_char;
   type u_CUVIDMPEG4PICPARAMS_QuantMatrixInter_array is array (0 .. 63) of aliased unsigned_char;
   type u_CUVIDMPEG4PICPARAMS is record
      ForwardRefIdx : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:293
      BackwardRefIdx : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:294
      video_object_layer_width : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:296
      video_object_layer_height : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:297
      vop_time_increment_bitcount : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:298
      top_field_first : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:299
      resync_marker_disable : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:300
      quant_type : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:301
      quarter_sample : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:302
      short_video_header : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:303
      divx_flags : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:304
      vop_coding_type : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:306
      vop_coded : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:307
      vop_rounding_type : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:308
      alternate_vertical_scan_flag : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:309
      interlaced : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:310
      vop_fcode_forward : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:311
      vop_fcode_backward : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:312
      trd : aliased u_CUVIDMPEG4PICPARAMS_trd_array;  -- /usr/local/cuda-8.0/include/cuviddec.h:313
      trb : aliased u_CUVIDMPEG4PICPARAMS_trb_array;  -- /usr/local/cuda-8.0/include/cuviddec.h:314
      QuantMatrixIntra : aliased u_CUVIDMPEG4PICPARAMS_QuantMatrixIntra_array;  -- /usr/local/cuda-8.0/include/cuviddec.h:316
      QuantMatrixInter : aliased u_CUVIDMPEG4PICPARAMS_QuantMatrixInter_array;  -- /usr/local/cuda-8.0/include/cuviddec.h:317
      gmc_enabled : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:318
   end record;
   pragma Convention (C_Pass_By_Copy, u_CUVIDMPEG4PICPARAMS);  -- /usr/local/cuda-8.0/include/cuviddec.h:291

  -- Picture index of backward reference (B-frames)
  -- VOL
  -- VOP
  -- Quantization matrices (raster order)
   subtype CUVIDMPEG4PICPARAMS is u_CUVIDMPEG4PICPARAMS;

  --//////////////////////////////////////////////////////////////////////////////////////////////
  -- VC1 Picture Parameters
  -- Picture index of forward reference (P/B-frames)
   type u_CUVIDVC1PICPARAMS is record
      ForwardRefIdx : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:328
      BackwardRefIdx : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:329
      FrameWidth : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:330
      FrameHeight : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:331
      intra_pic_flag : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:333
      ref_pic_flag : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:334
      progressive_fcm : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:335
      profile : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:337
      postprocflag : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:338
      pulldown : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:339
      interlace : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:340
      tfcntrflag : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:341
      finterpflag : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:342
      psf : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:343
      multires : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:344
      syncmarker : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:345
      rangered : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:346
      maxbframes : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:347
      panscan_flag : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:349
      refdist_flag : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:350
      extended_mv : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:351
      dquant : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:352
      vstransform : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:353
      loopfilter : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:354
      fastuvmc : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:355
      overlap : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:356
      quantizer : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:357
      extended_dmv : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:358
      range_mapy_flag : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:359
      range_mapy : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:360
      range_mapuv_flag : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:361
      range_mapuv : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:362
      rangeredfrm : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:363
   end record;
   pragma Convention (C_Pass_By_Copy, u_CUVIDVC1PICPARAMS);  -- /usr/local/cuda-8.0/include/cuviddec.h:326

  -- Picture index of backward reference (B-frames)
  -- Actual frame width
  -- Actual frame height
  -- PICTURE
  -- Set to 1 for I,BI frames
  -- Set to 1 for I,P frames
  -- Progressive frame
  -- SEQUENCE
  -- ENTRYPOINT
  -- range reduction state
   subtype CUVIDVC1PICPARAMS is u_CUVIDVC1PICPARAMS;

  --//////////////////////////////////////////////////////////////////////////////////////////////
  -- JPEG Picture Parameters
   type u_CUVIDJPEGPICPARAMS is record
      Reserved : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:373
   end record;
   pragma Convention (C_Pass_By_Copy, u_CUVIDJPEGPICPARAMS);  -- /usr/local/cuda-8.0/include/cuviddec.h:371

   subtype CUVIDJPEGPICPARAMS is u_CUVIDJPEGPICPARAMS;

  --//////////////////////////////////////////////////////////////////////////////////////////////
  -- HEVC Picture Parameters
  -- sps
   type u_CUVIDHEVCPICPARAMS_reserved1_array is array (0 .. 13) of aliased unsigned_char;
   type u_CUVIDHEVCPICPARAMS_column_width_minus1_array is array (0 .. 20) of aliased unsigned_short;
   type u_CUVIDHEVCPICPARAMS_row_height_minus1_array is array (0 .. 20) of aliased unsigned_short;
   type u_CUVIDHEVCPICPARAMS_reserved3_array is array (0 .. 14) of aliased unsigned;
   type u_CUVIDHEVCPICPARAMS_RefPicIdx_array is array (0 .. 15) of aliased int;
   type u_CUVIDHEVCPICPARAMS_PicOrderCntVal_array is array (0 .. 15) of aliased int;
   type u_CUVIDHEVCPICPARAMS_IsLongTerm_array is array (0 .. 15) of aliased unsigned_char;
   type u_CUVIDHEVCPICPARAMS_RefPicSetStCurrBefore_array is array (0 .. 7) of aliased unsigned_char;
   type u_CUVIDHEVCPICPARAMS_RefPicSetStCurrAfter_array is array (0 .. 7) of aliased unsigned_char;
   type u_CUVIDHEVCPICPARAMS_RefPicSetLtCurr_array is array (0 .. 7) of aliased unsigned_char;
   type u_CUVIDHEVCPICPARAMS_reserved4_array is array (0 .. 15) of aliased unsigned;
   type u_CUVIDHEVCPICPARAMS_ScalingList4x4_array is array (0 .. 5, 0 .. 15) of aliased unsigned_char;
   type u_CUVIDHEVCPICPARAMS_ScalingList8x8_array is array (0 .. 5, 0 .. 63) of aliased unsigned_char;
   type u_CUVIDHEVCPICPARAMS_ScalingList16x16_array is array (0 .. 5, 0 .. 63) of aliased unsigned_char;
   type u_CUVIDHEVCPICPARAMS_ScalingList32x32_array is array (0 .. 1, 0 .. 63) of aliased unsigned_char;
   type u_CUVIDHEVCPICPARAMS_ScalingListDCCoeff16x16_array is array (0 .. 5) of aliased unsigned_char;
   type u_CUVIDHEVCPICPARAMS_ScalingListDCCoeff32x32_array is array (0 .. 1) of aliased unsigned_char;
   type u_CUVIDHEVCPICPARAMS is record
      pic_width_in_luma_samples : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:385
      pic_height_in_luma_samples : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:386
      log2_min_luma_coding_block_size_minus3 : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:387
      log2_diff_max_min_luma_coding_block_size : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:388
      log2_min_transform_block_size_minus2 : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:389
      log2_diff_max_min_transform_block_size : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:390
      pcm_enabled_flag : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:391
      log2_min_pcm_luma_coding_block_size_minus3 : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:392
      log2_diff_max_min_pcm_luma_coding_block_size : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:393
      pcm_sample_bit_depth_luma_minus1 : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:394
      pcm_sample_bit_depth_chroma_minus1 : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:396
      pcm_loop_filter_disabled_flag : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:397
      strong_intra_smoothing_enabled_flag : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:398
      max_transform_hierarchy_depth_intra : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:399
      max_transform_hierarchy_depth_inter : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:400
      amp_enabled_flag : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:401
      separate_colour_plane_flag : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:402
      log2_max_pic_order_cnt_lsb_minus4 : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:403
      num_short_term_ref_pic_sets : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:405
      long_term_ref_pics_present_flag : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:406
      num_long_term_ref_pics_sps : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:407
      sps_temporal_mvp_enabled_flag : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:408
      sample_adaptive_offset_enabled_flag : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:409
      scaling_list_enable_flag : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:410
      IrapPicFlag : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:411
      IdrPicFlag : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:412
      bit_depth_luma_minus8 : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:414
      bit_depth_chroma_minus8 : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:415
      reserved1 : aliased u_CUVIDHEVCPICPARAMS_reserved1_array;  -- /usr/local/cuda-8.0/include/cuviddec.h:416
      dependent_slice_segments_enabled_flag : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:419
      slice_segment_header_extension_present_flag : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:420
      sign_data_hiding_enabled_flag : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:421
      cu_qp_delta_enabled_flag : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:422
      diff_cu_qp_delta_depth : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:423
      init_qp_minus26 : aliased signed_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:424
      pps_cb_qp_offset : aliased signed_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:425
      pps_cr_qp_offset : aliased signed_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:426
      constrained_intra_pred_flag : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:428
      weighted_pred_flag : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:429
      weighted_bipred_flag : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:430
      transform_skip_enabled_flag : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:431
      transquant_bypass_enabled_flag : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:432
      entropy_coding_sync_enabled_flag : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:433
      log2_parallel_merge_level_minus2 : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:434
      num_extra_slice_header_bits : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:435
      loop_filter_across_tiles_enabled_flag : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:437
      loop_filter_across_slices_enabled_flag : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:438
      output_flag_present_flag : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:439
      num_ref_idx_l0_default_active_minus1 : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:440
      num_ref_idx_l1_default_active_minus1 : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:441
      lists_modification_present_flag : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:442
      cabac_init_present_flag : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:443
      pps_slice_chroma_qp_offsets_present_flag : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:444
      deblocking_filter_override_enabled_flag : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:446
      pps_deblocking_filter_disabled_flag : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:447
      pps_beta_offset_div2 : aliased signed_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:448
      pps_tc_offset_div2 : aliased signed_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:449
      tiles_enabled_flag : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:450
      uniform_spacing_flag : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:451
      num_tile_columns_minus1 : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:452
      num_tile_rows_minus1 : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:453
      column_width_minus1 : aliased u_CUVIDHEVCPICPARAMS_column_width_minus1_array;  -- /usr/local/cuda-8.0/include/cuviddec.h:455
      row_height_minus1 : aliased u_CUVIDHEVCPICPARAMS_row_height_minus1_array;  -- /usr/local/cuda-8.0/include/cuviddec.h:456
      reserved3 : aliased u_CUVIDHEVCPICPARAMS_reserved3_array;  -- /usr/local/cuda-8.0/include/cuviddec.h:457
      NumBitsForShortTermRPSInSlice : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:460
      NumDeltaPocsOfRefRpsIdx : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:461
      NumPocTotalCurr : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:462
      NumPocStCurrBefore : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:463
      NumPocStCurrAfter : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:464
      NumPocLtCurr : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:465
      CurrPicOrderCntVal : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:466
      RefPicIdx : aliased u_CUVIDHEVCPICPARAMS_RefPicIdx_array;  -- /usr/local/cuda-8.0/include/cuviddec.h:467
      PicOrderCntVal : aliased u_CUVIDHEVCPICPARAMS_PicOrderCntVal_array;  -- /usr/local/cuda-8.0/include/cuviddec.h:468
      IsLongTerm : aliased u_CUVIDHEVCPICPARAMS_IsLongTerm_array;  -- /usr/local/cuda-8.0/include/cuviddec.h:469
      RefPicSetStCurrBefore : aliased u_CUVIDHEVCPICPARAMS_RefPicSetStCurrBefore_array;  -- /usr/local/cuda-8.0/include/cuviddec.h:470
      RefPicSetStCurrAfter : aliased u_CUVIDHEVCPICPARAMS_RefPicSetStCurrAfter_array;  -- /usr/local/cuda-8.0/include/cuviddec.h:471
      RefPicSetLtCurr : aliased u_CUVIDHEVCPICPARAMS_RefPicSetLtCurr_array;  -- /usr/local/cuda-8.0/include/cuviddec.h:472
      reserved4 : aliased u_CUVIDHEVCPICPARAMS_reserved4_array;  -- /usr/local/cuda-8.0/include/cuviddec.h:473
      ScalingList4x4 : aliased u_CUVIDHEVCPICPARAMS_ScalingList4x4_array;  -- /usr/local/cuda-8.0/include/cuviddec.h:476
      ScalingList8x8 : aliased u_CUVIDHEVCPICPARAMS_ScalingList8x8_array;  -- /usr/local/cuda-8.0/include/cuviddec.h:477
      ScalingList16x16 : aliased u_CUVIDHEVCPICPARAMS_ScalingList16x16_array;  -- /usr/local/cuda-8.0/include/cuviddec.h:478
      ScalingList32x32 : aliased u_CUVIDHEVCPICPARAMS_ScalingList32x32_array;  -- /usr/local/cuda-8.0/include/cuviddec.h:479
      ScalingListDCCoeff16x16 : aliased u_CUVIDHEVCPICPARAMS_ScalingListDCCoeff16x16_array;  -- /usr/local/cuda-8.0/include/cuviddec.h:480
      ScalingListDCCoeff32x32 : aliased u_CUVIDHEVCPICPARAMS_ScalingListDCCoeff32x32_array;  -- /usr/local/cuda-8.0/include/cuviddec.h:481
   end record;
   pragma Convention (C_Pass_By_Copy, u_CUVIDHEVCPICPARAMS);  -- /usr/local/cuda-8.0/include/cuviddec.h:382

  -- pps
  -- RefPicSets
  -- [refpic] Indices of valid reference pictures (-1 if unused for reference)
  -- [refpic]
  -- [refpic] 0=not a long-term reference, 1=long-term reference
  -- [0..NumPocStCurrBefore-1] -> refpic (0..15)
  -- [0..NumPocStCurrAfter-1] -> refpic (0..15)
  -- [0..NumPocLtCurr-1] -> refpic (0..15)
  -- scaling lists (diag order)
  -- [matrixId][i]
  -- [matrixId][i]
  -- [matrixId][i]
  -- [matrixId][i]
  -- [matrixId]
  -- [matrixId]
   subtype CUVIDHEVCPICPARAMS is u_CUVIDHEVCPICPARAMS;

  --//////////////////////////////////////////////////////////////////////////////////////////////
  -- Picture Parameters for Decoding 
  -- Coded Frame Size
   type u_CUVIDPICPARAMS is record
      PicWidthInMbs : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:492
      FrameHeightInMbs : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:493
      CurrPicIdx : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:494
      field_pic_flag : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:495
      bottom_field_flag : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:496
      second_field : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:497
      nBitstreamDataLen : aliased unsigned;  -- /usr/local/cuda-8.0/include/cuviddec.h:499
      pBitstreamData : access unsigned_char;  -- /usr/local/cuda-8.0/include/cuviddec.h:500
      nNumSlices : aliased unsigned;  -- /usr/local/cuda-8.0/include/cuviddec.h:501
      pSliceDataOffsets : access unsigned;  -- /usr/local/cuda-8.0/include/cuviddec.h:502
      ref_pic_flag : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:503
      intra_pic_flag : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:504
      Reserved : aliased u_CUVIDPICPARAMS_Reserved_array;  -- /usr/local/cuda-8.0/include/cuviddec.h:505
      CodecSpecific : aliased anon_31;  -- /usr/local/cuda-8.0/include/cuviddec.h:515
   end record;
   pragma Convention (C_Pass_By_Copy, u_CUVIDPICPARAMS);  -- /usr/local/cuda-8.0/include/cuviddec.h:490

  -- Coded Frame Height
  -- Output index of the current picture
  -- 0=frame picture, 1=field picture
  -- 0=top field, 1=bottom field (ignored if field_pic_flag=0)
  -- Second field of a complementary field pair
  -- Bitstream data
  -- Number of bytes in bitstream data buffer
  -- Ptr to bitstream data for this picture (slice-layer)
  -- Number of slices in this picture
  -- nNumSlices entries, contains offset of each slice within the bitstream data buffer
  -- This picture is a reference picture
  -- This picture is entirely intra coded
  -- Reserved for future use
  -- Codec-specific data
  -- Also used for MPEG-1
   subtype CUVIDPICPARAMS is u_CUVIDPICPARAMS;

  --//////////////////////////////////////////////////////////////////////////////////////////////
  -- Post-processing
  -- Input is progressive (deinterlace_mode will be ignored)
   type u_CUVIDPROCPARAMS_Reserved_array is array (0 .. 47) of aliased unsigned;
   type u_CUVIDPROCPARAMS_Reserved3_array is array (0 .. 2) of System.Address;
   type u_CUVIDPROCPARAMS is record
      progressive_frame : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:526
      second_field : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:527
      top_field_first : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:528
      unpaired_field : aliased int;  -- /usr/local/cuda-8.0/include/cuviddec.h:529
      reserved_flags : aliased unsigned;  -- /usr/local/cuda-8.0/include/cuviddec.h:531
      reserved_zero : aliased unsigned;  -- /usr/local/cuda-8.0/include/cuviddec.h:532
      raw_input_dptr : aliased Extensions.unsigned_long_long;  -- /usr/local/cuda-8.0/include/cuviddec.h:533
      raw_input_pitch : aliased unsigned;  -- /usr/local/cuda-8.0/include/cuviddec.h:534
      raw_input_format : aliased unsigned;  -- /usr/local/cuda-8.0/include/cuviddec.h:535
      raw_output_dptr : aliased Extensions.unsigned_long_long;  -- /usr/local/cuda-8.0/include/cuviddec.h:536
      raw_output_pitch : aliased unsigned;  -- /usr/local/cuda-8.0/include/cuviddec.h:537
      Reserved : aliased u_CUVIDPROCPARAMS_Reserved_array;  -- /usr/local/cuda-8.0/include/cuviddec.h:538
      Reserved3 : u_CUVIDPROCPARAMS_Reserved3_array;  -- /usr/local/cuda-8.0/include/cuviddec.h:539
   end record;
   pragma Convention (C_Pass_By_Copy, u_CUVIDPROCPARAMS);  -- /usr/local/cuda-8.0/include/cuviddec.h:524

  -- Output the second field (ignored if deinterlace mode is Weave)
  -- Input frame is top field first (1st field is top, 2nd field is bottom)
  -- Input only contains one field (2nd field is invalid)
  -- The fields below are used for raw YUV input
  -- Reserved for future use (set to zero)
  -- Reserved (set to zero)
  -- Input CUdeviceptr for raw YUV extensions
  -- pitch in bytes of raw YUV input (should be aligned appropriately)
  -- Reserved for future use (set to zero)
  -- Reserved for future use (set to zero)
  -- Reserved for future use (set to zero)
   subtype CUVIDPROCPARAMS is u_CUVIDPROCPARAMS;

  --//////////////////////////////////////////////////////////////////////////////////////////////
  -- In order to maximize decode latencies, there should be always at least 2 pictures in the decode
  -- queue at any time, in order to make sure that all decode engines are always busy.
  -- Overall data flow:
  --  - cuvidCreateDecoder(...)
  --  For each picture:
  --  - cuvidDecodePicture(N)
  --  - cuvidMapVideoFrame(N-4)
  --  - do some processing in cuda
  --  - cuvidUnmapVideoFrame(N-4)
  --  - cuvidDecodePicture(N+1)
  --  - cuvidMapVideoFrame(N-3)
  --    ...
  --  - cuvidDestroyDecoder(...)
  -- NOTE:
  -- - In the current version, the cuda context MUST be created from a D3D device, using cuD3D9CtxCreate function.
  --   For multi-threaded operation, the D3D device must also be created with the D3DCREATE_MULTITHREADED flag.
  -- - There is a limit to how many pictures can be mapped simultaneously (ulNumOutputSurfaces)
  -- - cuVidDecodePicture may block the calling thread if there are too many pictures pending 
  --   in the decode queue
  --//////////////////////////////////////////////////////////////////////////////////////////////
  -- Create/Destroy the decoder object
   function cuvidCreateDecoder (phDecoder : System.Address; pdci : access CUVIDDECODECREATEINFO) return cuda_h.CUresult;  -- /usr/local/cuda-8.0/include/cuviddec.h:569
   pragma Import (C, cuvidCreateDecoder, "cuvidCreateDecoder");

   function cuvidDestroyDecoder (hDecoder : CUvideodecoder) return cuda_h.CUresult;  -- /usr/local/cuda-8.0/include/cuviddec.h:570
   pragma Import (C, cuvidDestroyDecoder, "cuvidDestroyDecoder");

  -- Decode a single picture (field or frame)
   function cuvidDecodePicture (hDecoder : CUvideodecoder; pPicParams : access CUVIDPICPARAMS) return cuda_h.CUresult;  -- /usr/local/cuda-8.0/include/cuviddec.h:573
   pragma Import (C, cuvidDecodePicture, "cuvidDecodePicture");

  -- Post-process and map a video frame for use in cuda
  -- Unmap a previously mapped video frame
   function cuvidMapVideoFrame64
     (hDecoder : CUvideodecoder;
      nPicIdx : int;
      pDevPtr : access Extensions.unsigned_long_long;
      pPitch : access unsigned;
      pVPP : access CUVIDPROCPARAMS) return cuda_h.CUresult;  -- /usr/local/cuda-8.0/include/cuviddec.h:585
   pragma Import (C, cuvidMapVideoFrame64, "cuvidMapVideoFrame64");

   function cuvidUnmapVideoFrame64 (hDecoder : CUvideodecoder; DevPtr : Extensions.unsigned_long_long) return cuda_h.CUresult;  -- /usr/local/cuda-8.0/include/cuviddec.h:587
   pragma Import (C, cuvidUnmapVideoFrame64, "cuvidUnmapVideoFrame64");

  -- Get the pointer to the d3d9 surface that is the decode RT
   function cuvidGetVideoFrameSurface
     (hDecoder : CUvideodecoder;
      nPicIdx : int;
      pSrcSurface : System.Address) return cuda_h.CUresult;  -- /usr/local/cuda-8.0/include/cuviddec.h:595
   pragma Import (C, cuvidGetVideoFrameSurface, "cuvidGetVideoFrameSurface");

  --//////////////////////////////////////////////////////////////////////////////////////////////
  -- Context-locking: to facilitate multi-threaded implementations, the following 4 functions
  -- provide a simple mutex-style host synchronization. If a non-NULL context is specified
  -- in CUVIDDECODECREATEINFO, the codec library will acquire the mutex associated with the given 
  -- context before making any cuda calls.
  -- A multi-threaded application could create a lock associated with a context handle so that
  -- multiple threads can safely share the same cuda context:
  --  - use cuCtxPopCurrent immediately after context creation in order to create a 'floating' context
  --    that can be passed to cuvidCtxLockCreate.
  --  - When using a floating context, all cuda calls should only be made within a cuvidCtxLock/cuvidCtxUnlock section.
  -- NOTE: This is a safer alternative to cuCtxPushCurrent and cuCtxPopCurrent, and is not related to video
  -- decoder in any way (implemented as a critical section associated with cuCtx{Push|Pop}Current calls).
   function cuvidCtxLockCreate (pLock : System.Address; ctx : cuda_h.CUcontext) return cuda_h.CUresult;  -- /usr/local/cuda-8.0/include/cuviddec.h:612
   pragma Import (C, cuvidCtxLockCreate, "cuvidCtxLockCreate");

   function cuvidCtxLockDestroy (lck : CUvideoctxlock) return cuda_h.CUresult;  -- /usr/local/cuda-8.0/include/cuviddec.h:613
   pragma Import (C, cuvidCtxLockDestroy, "cuvidCtxLockDestroy");

   function cuvidCtxLock (lck : CUvideoctxlock; reserved_flags : unsigned) return cuda_h.CUresult;  -- /usr/local/cuda-8.0/include/cuviddec.h:614
   pragma Import (C, cuvidCtxLock, "cuvidCtxLock");

   function cuvidCtxUnlock (lck : CUvideoctxlock; reserved_flags : unsigned) return cuda_h.CUresult;  -- /usr/local/cuda-8.0/include/cuviddec.h:615
   pragma Import (C, cuvidCtxUnlock, "cuvidCtxUnlock");

  --//////////////////////////////////////////////////////////////////////////////////////////////
  -- Auto-lock helper for C++ applications
   package Class_CCtxAutoLock is
      type CCtxAutoLock is limited record
         m_ctx : CUvideoctxlock;  -- /usr/local/cuda-8.0/include/cuviddec.h:626
      end record;
      pragma Import (CPP, CCtxAutoLock);

      function New_CCtxAutoLock (ctx : CUvideoctxlock) return CCtxAutoLock;  -- /usr/local/cuda-8.0/include/cuviddec.h:628
      pragma CPP_Constructor (New_CCtxAutoLock, "_ZN12CCtxAutoLockC1EP17_CUcontextlock_st");

      procedure Delete_CCtxAutoLock (this : access CCtxAutoLock);  -- /usr/local/cuda-8.0/include/cuviddec.h:629
      pragma Import (CPP, Delete_CCtxAutoLock, "_ZN12CCtxAutoLockD1Ev");
   end;
   use Class_CCtxAutoLock;
end cuviddec_h;

pragma Ada_2005;
pragma Style_Checks (Off);

with Interfaces.C; use Interfaces.C;
with System;
with Interfaces.C.Extensions;
with cuviddec_h;
with Interfaces.C.Strings;
with cuda_h;

package nvcuvid_h is

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

  --//////////////////////////////////////////////////////////////////////////////////////////////
  -- High-level helper APIs for video sources
   type CUvideosource is new System.Address;  -- /usr/local/cuda-8.0/include/nvcuvid.h:50

   type CUvideoparser is new System.Address;  -- /usr/local/cuda-8.0/include/nvcuvid.h:51

   subtype CUvideotimestamp is Long_Long_Integer;  -- /usr/local/cuda-8.0/include/nvcuvid.h:52

  --//////////////////////////////////////////////////////////////////////////////////////////////
  -- video data structures
  -- Video Source State
  -- Error state (invalid source)
  -- Source is stopped (or reached end-of-stream)
  -- Source is running and delivering data
   subtype cudaVideoState is unsigned;
   cudaVideoState_Error : constant cudaVideoState := -1;
   cudaVideoState_Stopped : constant cudaVideoState := 0;
   cudaVideoState_Started : constant cudaVideoState := 1;  -- /usr/local/cuda-8.0/include/nvcuvid.h:64

  -- Audio compression
  -- MPEG-1 Audio
  -- MPEG-2 Audio
  -- MPEG-1 Layer III Audio
  -- Dolby Digital (AC3) Audio
  -- PCM Audio
   type cudaAudioCodec is 
     (cudaAudioCodec_MPEG1,
      cudaAudioCodec_MPEG2,
      cudaAudioCodec_MP3,
      cudaAudioCodec_AC3,
      cudaAudioCodec_LPCM);
   pragma Convention (C, cudaAudioCodec);  -- /usr/local/cuda-8.0/include/nvcuvid.h:73

  -- Video format
  -- Compression format
   type CUVIDEOFORMAT;
   type anon_35 is record
      numerator : aliased unsigned;  -- /usr/local/cuda-8.0/include/nvcuvid.h:81
      denominator : aliased unsigned;  -- /usr/local/cuda-8.0/include/nvcuvid.h:82
   end record;
   pragma Convention (C_Pass_By_Copy, anon_35);
   type anon_36 is record
      left : aliased int;  -- /usr/local/cuda-8.0/include/nvcuvid.h:91
      top : aliased int;  -- /usr/local/cuda-8.0/include/nvcuvid.h:92
      right : aliased int;  -- /usr/local/cuda-8.0/include/nvcuvid.h:93
      bottom : aliased int;  -- /usr/local/cuda-8.0/include/nvcuvid.h:94
   end record;
   pragma Convention (C_Pass_By_Copy, anon_36);
   type anon_37 is record
      x : aliased int;  -- /usr/local/cuda-8.0/include/nvcuvid.h:99
      y : aliased int;  -- /usr/local/cuda-8.0/include/nvcuvid.h:100
   end record;
   pragma Convention (C_Pass_By_Copy, anon_37);
   type anon_38 is record
      video_format : Extensions.Unsigned_3;  -- /usr/local/cuda-8.0/include/nvcuvid.h:103
      video_full_range_flag : Extensions.Unsigned_1;  -- /usr/local/cuda-8.0/include/nvcuvid.h:104
      reserved_zero_bits : Extensions.Unsigned_4;  -- /usr/local/cuda-8.0/include/nvcuvid.h:105
      color_primaries : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/nvcuvid.h:106
      transfer_characteristics : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/nvcuvid.h:107
      matrix_coefficients : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/nvcuvid.h:108
   end record;
   pragma Convention (C_Pass_By_Copy, anon_38);
   type CUVIDEOFORMAT is record
      codec : aliased cuviddec_h.cudaVideoCodec;  -- /usr/local/cuda-8.0/include/nvcuvid.h:79
      frame_rate : aliased anon_35;  -- /usr/local/cuda-8.0/include/nvcuvid.h:83
      progressive_sequence : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/nvcuvid.h:84
      bit_depth_luma_minus8 : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/nvcuvid.h:85
      bit_depth_chroma_minus8 : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/nvcuvid.h:86
      reserved1 : aliased unsigned_char;  -- /usr/local/cuda-8.0/include/nvcuvid.h:87
      coded_width : aliased unsigned;  -- /usr/local/cuda-8.0/include/nvcuvid.h:88
      coded_height : aliased unsigned;  -- /usr/local/cuda-8.0/include/nvcuvid.h:89
      display_area : aliased anon_36;  -- /usr/local/cuda-8.0/include/nvcuvid.h:95
      chroma_format : aliased cuviddec_h.cudaVideoChromaFormat;  -- /usr/local/cuda-8.0/include/nvcuvid.h:96
      bitrate : aliased unsigned;  -- /usr/local/cuda-8.0/include/nvcuvid.h:97
      display_aspect_ratio : aliased anon_37;  -- /usr/local/cuda-8.0/include/nvcuvid.h:101
      video_signal_description : aliased anon_38;  -- /usr/local/cuda-8.0/include/nvcuvid.h:109
      seqhdr_data_length : aliased unsigned;  -- /usr/local/cuda-8.0/include/nvcuvid.h:110
   end record;
   pragma Convention (C_Pass_By_Copy, CUVIDEOFORMAT);  -- /usr/local/cuda-8.0/include/nvcuvid.h:111

   --  skipped anonymous struct anon_34

  -- frame rate numerator   (0 = unspecified or variable frame rate)
  -- frame rate denominator (0 = unspecified or variable frame rate)
  -- frame rate = numerator / denominator (for example: 30000/1001)
  -- 0=interlaced, 1=progressive
  -- coded frame width
  -- coded frame height 
  -- area of the frame that should be displayed
  -- typical example:
  --   coded_width = 1920, coded_height = 1088
  --   display_area = { 0,0,1920,1080 }
  -- Chroma format
  -- video bitrate (bps, 0=unknown)
  -- Display Aspect Ratio = x:y (4:3, 16:9, etc)
  -- Additional bytes following (CUVIDEOFORMATEX)
  -- Video format including raw sequence header information
   type CUVIDEOFORMATEX_raw_seqhdr_data_array is array (0 .. 1023) of aliased unsigned_char;
   type CUVIDEOFORMATEX is record
      format : aliased CUVIDEOFORMAT;  -- /usr/local/cuda-8.0/include/nvcuvid.h:116
      raw_seqhdr_data : aliased CUVIDEOFORMATEX_raw_seqhdr_data_array;  -- /usr/local/cuda-8.0/include/nvcuvid.h:117
   end record;
   pragma Convention (C_Pass_By_Copy, CUVIDEOFORMATEX);  -- /usr/local/cuda-8.0/include/nvcuvid.h:118

   --  skipped anonymous struct anon_39

  -- Audio Format
  -- Compression format
   type CUAUDIOFORMAT is record
      codec : aliased cudaAudioCodec;  -- /usr/local/cuda-8.0/include/nvcuvid.h:124
      channels : aliased unsigned;  -- /usr/local/cuda-8.0/include/nvcuvid.h:125
      samplespersec : aliased unsigned;  -- /usr/local/cuda-8.0/include/nvcuvid.h:126
      bitrate : aliased unsigned;  -- /usr/local/cuda-8.0/include/nvcuvid.h:127
      reserved1 : aliased unsigned;  -- /usr/local/cuda-8.0/include/nvcuvid.h:128
      reserved2 : aliased unsigned;  -- /usr/local/cuda-8.0/include/nvcuvid.h:129
   end record;
   pragma Convention (C_Pass_By_Copy, CUAUDIOFORMAT);  -- /usr/local/cuda-8.0/include/nvcuvid.h:130

   --  skipped anonymous struct anon_40

  -- number of audio channels
  -- sampling frequency
  -- For uncompressed, can also be used to determine bits per sample
  --//////////////////////////////////////////////////////////////////////////////////////////////
  -- video source
  -- Data packet
  -- Set when this is the last packet for this stream
  -- Timestamp is valid
  -- Set when a discontinuity has to be signalled
   subtype CUvideopacketflags is unsigned;
   CUVID_PKT_ENDOFSTREAM : constant CUvideopacketflags := 1;
   CUVID_PKT_TIMESTAMP : constant CUvideopacketflags := 2;
   CUVID_PKT_DISCONTINUITY : constant CUvideopacketflags := 4;  -- /usr/local/cuda-8.0/include/nvcuvid.h:144

  -- Combination of CUVID_PKT_XXX flags
   type u_CUVIDSOURCEDATAPACKET is record
      flags : aliased unsigned_long;  -- /usr/local/cuda-8.0/include/nvcuvid.h:148
      payload_size : aliased unsigned_long;  -- /usr/local/cuda-8.0/include/nvcuvid.h:149
      payload : access unsigned_char;  -- /usr/local/cuda-8.0/include/nvcuvid.h:150
      timestamp : aliased CUvideotimestamp;  -- /usr/local/cuda-8.0/include/nvcuvid.h:151
   end record;
   pragma Convention (C_Pass_By_Copy, u_CUVIDSOURCEDATAPACKET);  -- /usr/local/cuda-8.0/include/nvcuvid.h:146

  -- number of bytes in the payload (may be zero if EOS flag is set)
  -- Pointer to packet payload data (may be NULL if EOS flag is set)
  -- Presentation timestamp (10MHz clock), only valid if CUVID_PKT_TIMESTAMP flag is set
   subtype CUVIDSOURCEDATAPACKET is u_CUVIDSOURCEDATAPACKET;

  -- Callback for packet delivery
   type PFNVIDSOURCECALLBACK is access function (arg1 : System.Address; arg2 : access CUVIDSOURCEDATAPACKET) return int;
   pragma Convention (C, PFNVIDSOURCECALLBACK);  -- /usr/local/cuda-8.0/include/nvcuvid.h:155

  -- Timestamp units in Hz (0=default=10000000Hz)
   type u_CUVIDSOURCEPARAMS_uReserved1_array is array (0 .. 6) of aliased unsigned;
   type u_CUVIDSOURCEPARAMS_pvReserved2_array is array (0 .. 7) of System.Address;
   type u_CUVIDSOURCEPARAMS is record
      ulClockRate : aliased unsigned;  -- /usr/local/cuda-8.0/include/nvcuvid.h:159
      uReserved1 : aliased u_CUVIDSOURCEPARAMS_uReserved1_array;  -- /usr/local/cuda-8.0/include/nvcuvid.h:160
      pUserData : System.Address;  -- /usr/local/cuda-8.0/include/nvcuvid.h:161
      pfnVideoDataHandler : PFNVIDSOURCECALLBACK;  -- /usr/local/cuda-8.0/include/nvcuvid.h:162
      pfnAudioDataHandler : PFNVIDSOURCECALLBACK;  -- /usr/local/cuda-8.0/include/nvcuvid.h:163
      pvReserved2 : u_CUVIDSOURCEPARAMS_pvReserved2_array;  -- /usr/local/cuda-8.0/include/nvcuvid.h:164
   end record;
   pragma Convention (C_Pass_By_Copy, u_CUVIDSOURCEPARAMS);  -- /usr/local/cuda-8.0/include/nvcuvid.h:157

  -- Reserved for future use - set to zero
  -- Parameter passed in to the data handlers
  -- Called to deliver audio packets
  -- Called to deliver video packets
  -- Reserved for future use - set to NULL
   subtype CUVIDSOURCEPARAMS is u_CUVIDSOURCEPARAMS;

  -- Return extended format structure (CUVIDEOFORMATEX)
   subtype CUvideosourceformat_flags is unsigned;
   CUVID_FMT_EXTFORMATINFO : constant CUvideosourceformat_flags := 256;  -- /usr/local/cuda-8.0/include/nvcuvid.h:169

  -- Video file source
   function cuvidCreateVideoSource
     (pObj : System.Address;
      pszFileName : Interfaces.C.Strings.chars_ptr;
      pParams : access CUVIDSOURCEPARAMS) return cuda_h.CUresult;  -- /usr/local/cuda-8.0/include/nvcuvid.h:173
   pragma Import (C, cuvidCreateVideoSource, "cuvidCreateVideoSource");

   function cuvidCreateVideoSourceW
     (pObj : System.Address;
      pwszFileName : access wchar_t;
      pParams : access CUVIDSOURCEPARAMS) return cuda_h.CUresult;  -- /usr/local/cuda-8.0/include/nvcuvid.h:174
   pragma Import (C, cuvidCreateVideoSourceW, "cuvidCreateVideoSourceW");

   function cuvidDestroyVideoSource (obj : CUvideosource) return cuda_h.CUresult;  -- /usr/local/cuda-8.0/include/nvcuvid.h:175
   pragma Import (C, cuvidDestroyVideoSource, "cuvidDestroyVideoSource");

   function cuvidSetVideoSourceState (obj : CUvideosource; state : cudaVideoState) return cuda_h.CUresult;  -- /usr/local/cuda-8.0/include/nvcuvid.h:176
   pragma Import (C, cuvidSetVideoSourceState, "cuvidSetVideoSourceState");

   function cuvidGetVideoSourceState (obj : CUvideosource) return cudaVideoState;  -- /usr/local/cuda-8.0/include/nvcuvid.h:177
   pragma Import (C, cuvidGetVideoSourceState, "cuvidGetVideoSourceState");

   function cuvidGetSourceVideoFormat
     (obj : CUvideosource;
      pvidfmt : access CUVIDEOFORMAT;
      flags : unsigned) return cuda_h.CUresult;  -- /usr/local/cuda-8.0/include/nvcuvid.h:178
   pragma Import (C, cuvidGetSourceVideoFormat, "cuvidGetSourceVideoFormat");

   function cuvidGetSourceAudioFormat
     (obj : CUvideosource;
      paudfmt : access CUAUDIOFORMAT;
      flags : unsigned) return cuda_h.CUresult;  -- /usr/local/cuda-8.0/include/nvcuvid.h:179
   pragma Import (C, cuvidGetSourceAudioFormat, "cuvidGetSourceAudioFormat");

  --//////////////////////////////////////////////////////////////////////////////////////////////
  -- Video parser
   type u_CUVIDPARSERDISPINFO is record
      picture_index : aliased int;  -- /usr/local/cuda-8.0/include/nvcuvid.h:189
      progressive_frame : aliased int;  -- /usr/local/cuda-8.0/include/nvcuvid.h:190
      top_field_first : aliased int;  -- /usr/local/cuda-8.0/include/nvcuvid.h:191
      repeat_first_field : aliased int;  -- /usr/local/cuda-8.0/include/nvcuvid.h:192
      timestamp : aliased CUvideotimestamp;  -- /usr/local/cuda-8.0/include/nvcuvid.h:193
   end record;
   pragma Convention (C_Pass_By_Copy, u_CUVIDPARSERDISPINFO);  -- /usr/local/cuda-8.0/include/nvcuvid.h:187

  -- Number of additional fields (1=ivtc, 2=frame doubling, 4=frame tripling, -1=unpaired field)
   subtype CUVIDPARSERDISPINFO is u_CUVIDPARSERDISPINFO;

  -- Parser callbacks
  -- The parser will call these synchronously from within cuvidParseVideoData(), whenever a picture is ready to
  -- be decoded and/or displayed.
   type PFNVIDSEQUENCECALLBACK is access function (arg1 : System.Address; arg2 : access CUVIDEOFORMAT) return int;
   pragma Convention (C, PFNVIDSEQUENCECALLBACK);  -- /usr/local/cuda-8.0/include/nvcuvid.h:201

   type PFNVIDDECODECALLBACK is access function (arg1 : System.Address; arg2 : access cuviddec_h.CUVIDPICPARAMS) return int;
   pragma Convention (C, PFNVIDDECODECALLBACK);  -- /usr/local/cuda-8.0/include/nvcuvid.h:202

   type PFNVIDDISPLAYCALLBACK is access function (arg1 : System.Address; arg2 : access CUVIDPARSERDISPINFO) return int;
   pragma Convention (C, PFNVIDDISPLAYCALLBACK);  -- /usr/local/cuda-8.0/include/nvcuvid.h:203

  -- cudaVideoCodec_XXX
   type u_CUVIDPARSERPARAMS_uReserved1_array is array (0 .. 4) of aliased unsigned;
   type u_CUVIDPARSERPARAMS_pvReserved2_array is array (0 .. 6) of System.Address;
   type u_CUVIDPARSERPARAMS is record
      CodecType : aliased cuviddec_h.cudaVideoCodec;  -- /usr/local/cuda-8.0/include/nvcuvid.h:207
      ulMaxNumDecodeSurfaces : aliased unsigned;  -- /usr/local/cuda-8.0/include/nvcuvid.h:208
      ulClockRate : aliased unsigned;  -- /usr/local/cuda-8.0/include/nvcuvid.h:209
      ulErrorThreshold : aliased unsigned;  -- /usr/local/cuda-8.0/include/nvcuvid.h:210
      ulMaxDisplayDelay : aliased unsigned;  -- /usr/local/cuda-8.0/include/nvcuvid.h:211
      uReserved1 : aliased u_CUVIDPARSERPARAMS_uReserved1_array;  -- /usr/local/cuda-8.0/include/nvcuvid.h:212
      pUserData : System.Address;  -- /usr/local/cuda-8.0/include/nvcuvid.h:213
      pfnSequenceCallback : PFNVIDSEQUENCECALLBACK;  -- /usr/local/cuda-8.0/include/nvcuvid.h:214
      pfnDecodePicture : PFNVIDDECODECALLBACK;  -- /usr/local/cuda-8.0/include/nvcuvid.h:215
      pfnDisplayPicture : PFNVIDDISPLAYCALLBACK;  -- /usr/local/cuda-8.0/include/nvcuvid.h:216
      pvReserved2 : u_CUVIDPARSERPARAMS_pvReserved2_array;  -- /usr/local/cuda-8.0/include/nvcuvid.h:217
      pExtVideoInfo : access CUVIDEOFORMATEX;  -- /usr/local/cuda-8.0/include/nvcuvid.h:218
   end record;
   pragma Convention (C_Pass_By_Copy, u_CUVIDPARSERPARAMS);  -- /usr/local/cuda-8.0/include/nvcuvid.h:205

  -- Max # of decode surfaces (parser will cycle through these)
  -- Timestamp units in Hz (0=default=10000000Hz)
  -- % Error threshold (0-100) for calling pfnDecodePicture (100=always call pfnDecodePicture even if picture bitstream is fully corrupted)
  -- Max display queue delay (improves pipelining of decode with display) - 0=no delay (recommended values: 2..4)
  -- Reserved for future use - set to 0
  -- User data for callbacks
  -- Called before decoding frames and/or whenever there is a format change
  -- Called when a picture is ready to be decoded (decode order)
  -- Called whenever a picture is ready to be displayed (display order)
  -- Reserved for future use - set to NULL
  -- [Optional] sequence header data from system layer
   subtype CUVIDPARSERPARAMS is u_CUVIDPARSERPARAMS;

   function cuvidCreateVideoParser (pObj : System.Address; pParams : access CUVIDPARSERPARAMS) return cuda_h.CUresult;  -- /usr/local/cuda-8.0/include/nvcuvid.h:222
   pragma Import (C, cuvidCreateVideoParser, "cuvidCreateVideoParser");

   function cuvidParseVideoData (obj : CUvideoparser; pPacket : access CUVIDSOURCEDATAPACKET) return cuda_h.CUresult;  -- /usr/local/cuda-8.0/include/nvcuvid.h:223
   pragma Import (C, cuvidParseVideoData, "cuvidParseVideoData");

   function cuvidDestroyVideoParser (obj : CUvideoparser) return cuda_h.CUresult;  -- /usr/local/cuda-8.0/include/nvcuvid.h:224
   pragma Import (C, cuvidDestroyVideoParser, "cuvidDestroyVideoParser");

  --//////////////////////////////////////////////////////////////////////////////////////////////
end nvcuvid_h;

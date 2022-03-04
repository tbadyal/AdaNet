pragma Ada_2005;
pragma Style_Checks (Off);

with Interfaces.C; use Interfaces.C;
with System;

package GL_gl_h is

   --  unsupported macro: GLAPI __attribute__((visibility("default")))
   --  unsupported macro: APIENTRY GLAPIENTRY
   --  unsupported macro: APIENTRYP APIENTRY *
   --  unsupported macro: GLAPIENTRYP GLAPIENTRY *
   GL_VERSION_1_1 : constant := 1;  --  /usr/include/GL/gl.h:112
   GL_VERSION_1_2 : constant := 1;  --  /usr/include/GL/gl.h:113
   GL_VERSION_1_3 : constant := 1;  --  /usr/include/GL/gl.h:114
   GL_ARB_imaging : constant := 1;  --  /usr/include/GL/gl.h:115

   GL_FALSE : constant := 0;  --  /usr/include/GL/gl.h:144
   GL_TRUE : constant := 1;  --  /usr/include/GL/gl.h:145

   GL_BYTE : constant := 16#1400#;  --  /usr/include/GL/gl.h:148
   GL_UNSIGNED_BYTE : constant := 16#1401#;  --  /usr/include/GL/gl.h:149
   GL_SHORT : constant := 16#1402#;  --  /usr/include/GL/gl.h:150
   GL_UNSIGNED_SHORT : constant := 16#1403#;  --  /usr/include/GL/gl.h:151
   GL_INT : constant := 16#1404#;  --  /usr/include/GL/gl.h:152
   GL_UNSIGNED_INT : constant := 16#1405#;  --  /usr/include/GL/gl.h:153
   GL_FLOAT : constant := 16#1406#;  --  /usr/include/GL/gl.h:154
   GL_2_BYTES : constant := 16#1407#;  --  /usr/include/GL/gl.h:155
   GL_3_BYTES : constant := 16#1408#;  --  /usr/include/GL/gl.h:156
   GL_4_BYTES : constant := 16#1409#;  --  /usr/include/GL/gl.h:157
   GL_DOUBLE : constant := 16#140A#;  --  /usr/include/GL/gl.h:158

   GL_POINTS : constant := 16#0000#;  --  /usr/include/GL/gl.h:161
   GL_LINES : constant := 16#0001#;  --  /usr/include/GL/gl.h:162
   GL_LINE_LOOP : constant := 16#0002#;  --  /usr/include/GL/gl.h:163
   GL_LINE_STRIP : constant := 16#0003#;  --  /usr/include/GL/gl.h:164
   GL_TRIANGLES : constant := 16#0004#;  --  /usr/include/GL/gl.h:165
   GL_TRIANGLE_STRIP : constant := 16#0005#;  --  /usr/include/GL/gl.h:166
   GL_TRIANGLE_FAN : constant := 16#0006#;  --  /usr/include/GL/gl.h:167
   GL_QUADS : constant := 16#0007#;  --  /usr/include/GL/gl.h:168
   GL_QUAD_STRIP : constant := 16#0008#;  --  /usr/include/GL/gl.h:169
   GL_POLYGON : constant := 16#0009#;  --  /usr/include/GL/gl.h:170

   GL_VERTEX_ARRAY : constant := 16#8074#;  --  /usr/include/GL/gl.h:173
   GL_NORMAL_ARRAY : constant := 16#8075#;  --  /usr/include/GL/gl.h:174
   GL_COLOR_ARRAY : constant := 16#8076#;  --  /usr/include/GL/gl.h:175
   GL_INDEX_ARRAY : constant := 16#8077#;  --  /usr/include/GL/gl.h:176
   GL_TEXTURE_COORD_ARRAY : constant := 16#8078#;  --  /usr/include/GL/gl.h:177
   GL_EDGE_FLAG_ARRAY : constant := 16#8079#;  --  /usr/include/GL/gl.h:178
   GL_VERTEX_ARRAY_SIZE : constant := 16#807A#;  --  /usr/include/GL/gl.h:179
   GL_VERTEX_ARRAY_TYPE : constant := 16#807B#;  --  /usr/include/GL/gl.h:180
   GL_VERTEX_ARRAY_STRIDE : constant := 16#807C#;  --  /usr/include/GL/gl.h:181
   GL_NORMAL_ARRAY_TYPE : constant := 16#807E#;  --  /usr/include/GL/gl.h:182
   GL_NORMAL_ARRAY_STRIDE : constant := 16#807F#;  --  /usr/include/GL/gl.h:183
   GL_COLOR_ARRAY_SIZE : constant := 16#8081#;  --  /usr/include/GL/gl.h:184
   GL_COLOR_ARRAY_TYPE : constant := 16#8082#;  --  /usr/include/GL/gl.h:185
   GL_COLOR_ARRAY_STRIDE : constant := 16#8083#;  --  /usr/include/GL/gl.h:186
   GL_INDEX_ARRAY_TYPE : constant := 16#8085#;  --  /usr/include/GL/gl.h:187
   GL_INDEX_ARRAY_STRIDE : constant := 16#8086#;  --  /usr/include/GL/gl.h:188
   GL_TEXTURE_COORD_ARRAY_SIZE : constant := 16#8088#;  --  /usr/include/GL/gl.h:189
   GL_TEXTURE_COORD_ARRAY_TYPE : constant := 16#8089#;  --  /usr/include/GL/gl.h:190
   GL_TEXTURE_COORD_ARRAY_STRIDE : constant := 16#808A#;  --  /usr/include/GL/gl.h:191
   GL_EDGE_FLAG_ARRAY_STRIDE : constant := 16#808C#;  --  /usr/include/GL/gl.h:192
   GL_VERTEX_ARRAY_POINTER : constant := 16#808E#;  --  /usr/include/GL/gl.h:193
   GL_NORMAL_ARRAY_POINTER : constant := 16#808F#;  --  /usr/include/GL/gl.h:194
   GL_COLOR_ARRAY_POINTER : constant := 16#8090#;  --  /usr/include/GL/gl.h:195
   GL_INDEX_ARRAY_POINTER : constant := 16#8091#;  --  /usr/include/GL/gl.h:196
   GL_TEXTURE_COORD_ARRAY_POINTER : constant := 16#8092#;  --  /usr/include/GL/gl.h:197
   GL_EDGE_FLAG_ARRAY_POINTER : constant := 16#8093#;  --  /usr/include/GL/gl.h:198
   GL_V2F : constant := 16#2A20#;  --  /usr/include/GL/gl.h:199
   GL_V3F : constant := 16#2A21#;  --  /usr/include/GL/gl.h:200
   GL_C4UB_V2F : constant := 16#2A22#;  --  /usr/include/GL/gl.h:201
   GL_C4UB_V3F : constant := 16#2A23#;  --  /usr/include/GL/gl.h:202
   GL_C3F_V3F : constant := 16#2A24#;  --  /usr/include/GL/gl.h:203
   GL_N3F_V3F : constant := 16#2A25#;  --  /usr/include/GL/gl.h:204
   GL_C4F_N3F_V3F : constant := 16#2A26#;  --  /usr/include/GL/gl.h:205
   GL_T2F_V3F : constant := 16#2A27#;  --  /usr/include/GL/gl.h:206
   GL_T4F_V4F : constant := 16#2A28#;  --  /usr/include/GL/gl.h:207
   GL_T2F_C4UB_V3F : constant := 16#2A29#;  --  /usr/include/GL/gl.h:208
   GL_T2F_C3F_V3F : constant := 16#2A2A#;  --  /usr/include/GL/gl.h:209
   GL_T2F_N3F_V3F : constant := 16#2A2B#;  --  /usr/include/GL/gl.h:210
   GL_T2F_C4F_N3F_V3F : constant := 16#2A2C#;  --  /usr/include/GL/gl.h:211
   GL_T4F_C4F_N3F_V4F : constant := 16#2A2D#;  --  /usr/include/GL/gl.h:212

   GL_MATRIX_MODE : constant := 16#0BA0#;  --  /usr/include/GL/gl.h:215
   GL_MODELVIEW : constant := 16#1700#;  --  /usr/include/GL/gl.h:216
   GL_PROJECTION : constant := 16#1701#;  --  /usr/include/GL/gl.h:217
   GL_TEXTURE : constant := 16#1702#;  --  /usr/include/GL/gl.h:218

   GL_POINT_SMOOTH : constant := 16#0B10#;  --  /usr/include/GL/gl.h:221
   GL_POINT_SIZE : constant := 16#0B11#;  --  /usr/include/GL/gl.h:222
   GL_POINT_SIZE_GRANULARITY : constant := 16#0B13#;  --  /usr/include/GL/gl.h:223
   GL_POINT_SIZE_RANGE : constant := 16#0B12#;  --  /usr/include/GL/gl.h:224

   GL_LINE_SMOOTH : constant := 16#0B20#;  --  /usr/include/GL/gl.h:227
   GL_LINE_STIPPLE : constant := 16#0B24#;  --  /usr/include/GL/gl.h:228
   GL_LINE_STIPPLE_PATTERN : constant := 16#0B25#;  --  /usr/include/GL/gl.h:229
   GL_LINE_STIPPLE_REPEAT : constant := 16#0B26#;  --  /usr/include/GL/gl.h:230
   GL_LINE_WIDTH : constant := 16#0B21#;  --  /usr/include/GL/gl.h:231
   GL_LINE_WIDTH_GRANULARITY : constant := 16#0B23#;  --  /usr/include/GL/gl.h:232
   GL_LINE_WIDTH_RANGE : constant := 16#0B22#;  --  /usr/include/GL/gl.h:233

   GL_POINT : constant := 16#1B00#;  --  /usr/include/GL/gl.h:236
   GL_LINE : constant := 16#1B01#;  --  /usr/include/GL/gl.h:237
   GL_FILL : constant := 16#1B02#;  --  /usr/include/GL/gl.h:238
   GL_CW : constant := 16#0900#;  --  /usr/include/GL/gl.h:239
   GL_CCW : constant := 16#0901#;  --  /usr/include/GL/gl.h:240
   GL_FRONT : constant := 16#0404#;  --  /usr/include/GL/gl.h:241
   GL_BACK : constant := 16#0405#;  --  /usr/include/GL/gl.h:242
   GL_POLYGON_MODE : constant := 16#0B40#;  --  /usr/include/GL/gl.h:243
   GL_POLYGON_SMOOTH : constant := 16#0B41#;  --  /usr/include/GL/gl.h:244
   GL_POLYGON_STIPPLE : constant := 16#0B42#;  --  /usr/include/GL/gl.h:245
   GL_EDGE_FLAG : constant := 16#0B43#;  --  /usr/include/GL/gl.h:246
   GL_CULL_FACE : constant := 16#0B44#;  --  /usr/include/GL/gl.h:247
   GL_CULL_FACE_MODE : constant := 16#0B45#;  --  /usr/include/GL/gl.h:248
   GL_FRONT_FACE : constant := 16#0B46#;  --  /usr/include/GL/gl.h:249
   GL_POLYGON_OFFSET_FACTOR : constant := 16#8038#;  --  /usr/include/GL/gl.h:250
   GL_POLYGON_OFFSET_UNITS : constant := 16#2A00#;  --  /usr/include/GL/gl.h:251
   GL_POLYGON_OFFSET_POINT : constant := 16#2A01#;  --  /usr/include/GL/gl.h:252
   GL_POLYGON_OFFSET_LINE : constant := 16#2A02#;  --  /usr/include/GL/gl.h:253
   GL_POLYGON_OFFSET_FILL : constant := 16#8037#;  --  /usr/include/GL/gl.h:254

   GL_COMPILE : constant := 16#1300#;  --  /usr/include/GL/gl.h:257
   GL_COMPILE_AND_EXECUTE : constant := 16#1301#;  --  /usr/include/GL/gl.h:258
   GL_LIST_BASE : constant := 16#0B32#;  --  /usr/include/GL/gl.h:259
   GL_LIST_INDEX : constant := 16#0B33#;  --  /usr/include/GL/gl.h:260
   GL_LIST_MODE : constant := 16#0B30#;  --  /usr/include/GL/gl.h:261

   GL_NEVER : constant := 16#0200#;  --  /usr/include/GL/gl.h:264
   GL_LESS : constant := 16#0201#;  --  /usr/include/GL/gl.h:265
   GL_EQUAL : constant := 16#0202#;  --  /usr/include/GL/gl.h:266
   GL_LEQUAL : constant := 16#0203#;  --  /usr/include/GL/gl.h:267
   GL_GREATER : constant := 16#0204#;  --  /usr/include/GL/gl.h:268
   GL_NOTEQUAL : constant := 16#0205#;  --  /usr/include/GL/gl.h:269
   GL_GEQUAL : constant := 16#0206#;  --  /usr/include/GL/gl.h:270
   GL_ALWAYS : constant := 16#0207#;  --  /usr/include/GL/gl.h:271
   GL_DEPTH_TEST : constant := 16#0B71#;  --  /usr/include/GL/gl.h:272
   GL_DEPTH_BITS : constant := 16#0D56#;  --  /usr/include/GL/gl.h:273
   GL_DEPTH_CLEAR_VALUE : constant := 16#0B73#;  --  /usr/include/GL/gl.h:274
   GL_DEPTH_FUNC : constant := 16#0B74#;  --  /usr/include/GL/gl.h:275
   GL_DEPTH_RANGE : constant := 16#0B70#;  --  /usr/include/GL/gl.h:276
   GL_DEPTH_WRITEMASK : constant := 16#0B72#;  --  /usr/include/GL/gl.h:277
   GL_DEPTH_COMPONENT : constant := 16#1902#;  --  /usr/include/GL/gl.h:278

   GL_LIGHTING : constant := 16#0B50#;  --  /usr/include/GL/gl.h:281
   GL_LIGHT0 : constant := 16#4000#;  --  /usr/include/GL/gl.h:282
   GL_LIGHT1 : constant := 16#4001#;  --  /usr/include/GL/gl.h:283
   GL_LIGHT2 : constant := 16#4002#;  --  /usr/include/GL/gl.h:284
   GL_LIGHT3 : constant := 16#4003#;  --  /usr/include/GL/gl.h:285
   GL_LIGHT4 : constant := 16#4004#;  --  /usr/include/GL/gl.h:286
   GL_LIGHT5 : constant := 16#4005#;  --  /usr/include/GL/gl.h:287
   GL_LIGHT6 : constant := 16#4006#;  --  /usr/include/GL/gl.h:288
   GL_LIGHT7 : constant := 16#4007#;  --  /usr/include/GL/gl.h:289
   GL_SPOT_EXPONENT : constant := 16#1205#;  --  /usr/include/GL/gl.h:290
   GL_SPOT_CUTOFF : constant := 16#1206#;  --  /usr/include/GL/gl.h:291
   GL_CONSTANT_ATTENUATION : constant := 16#1207#;  --  /usr/include/GL/gl.h:292
   GL_LINEAR_ATTENUATION : constant := 16#1208#;  --  /usr/include/GL/gl.h:293
   GL_QUADRATIC_ATTENUATION : constant := 16#1209#;  --  /usr/include/GL/gl.h:294
   GL_AMBIENT : constant := 16#1200#;  --  /usr/include/GL/gl.h:295
   GL_DIFFUSE : constant := 16#1201#;  --  /usr/include/GL/gl.h:296
   GL_SPECULAR : constant := 16#1202#;  --  /usr/include/GL/gl.h:297
   GL_SHININESS : constant := 16#1601#;  --  /usr/include/GL/gl.h:298
   GL_EMISSION : constant := 16#1600#;  --  /usr/include/GL/gl.h:299
   GL_POSITION : constant := 16#1203#;  --  /usr/include/GL/gl.h:300
   GL_SPOT_DIRECTION : constant := 16#1204#;  --  /usr/include/GL/gl.h:301
   GL_AMBIENT_AND_DIFFUSE : constant := 16#1602#;  --  /usr/include/GL/gl.h:302
   GL_COLOR_INDEXES : constant := 16#1603#;  --  /usr/include/GL/gl.h:303
   GL_LIGHT_MODEL_TWO_SIDE : constant := 16#0B52#;  --  /usr/include/GL/gl.h:304
   GL_LIGHT_MODEL_LOCAL_VIEWER : constant := 16#0B51#;  --  /usr/include/GL/gl.h:305
   GL_LIGHT_MODEL_AMBIENT : constant := 16#0B53#;  --  /usr/include/GL/gl.h:306
   GL_FRONT_AND_BACK : constant := 16#0408#;  --  /usr/include/GL/gl.h:307
   GL_SHADE_MODEL : constant := 16#0B54#;  --  /usr/include/GL/gl.h:308
   GL_FLAT : constant := 16#1D00#;  --  /usr/include/GL/gl.h:309
   GL_SMOOTH : constant := 16#1D01#;  --  /usr/include/GL/gl.h:310
   GL_COLOR_MATERIAL : constant := 16#0B57#;  --  /usr/include/GL/gl.h:311
   GL_COLOR_MATERIAL_FACE : constant := 16#0B55#;  --  /usr/include/GL/gl.h:312
   GL_COLOR_MATERIAL_PARAMETER : constant := 16#0B56#;  --  /usr/include/GL/gl.h:313
   GL_NORMALIZE : constant := 16#0BA1#;  --  /usr/include/GL/gl.h:314

   GL_CLIP_PLANE0 : constant := 16#3000#;  --  /usr/include/GL/gl.h:317
   GL_CLIP_PLANE1 : constant := 16#3001#;  --  /usr/include/GL/gl.h:318
   GL_CLIP_PLANE2 : constant := 16#3002#;  --  /usr/include/GL/gl.h:319
   GL_CLIP_PLANE3 : constant := 16#3003#;  --  /usr/include/GL/gl.h:320
   GL_CLIP_PLANE4 : constant := 16#3004#;  --  /usr/include/GL/gl.h:321
   GL_CLIP_PLANE5 : constant := 16#3005#;  --  /usr/include/GL/gl.h:322

   GL_ACCUM_RED_BITS : constant := 16#0D58#;  --  /usr/include/GL/gl.h:325
   GL_ACCUM_GREEN_BITS : constant := 16#0D59#;  --  /usr/include/GL/gl.h:326
   GL_ACCUM_BLUE_BITS : constant := 16#0D5A#;  --  /usr/include/GL/gl.h:327
   GL_ACCUM_ALPHA_BITS : constant := 16#0D5B#;  --  /usr/include/GL/gl.h:328
   GL_ACCUM_CLEAR_VALUE : constant := 16#0B80#;  --  /usr/include/GL/gl.h:329
   GL_ACCUM : constant := 16#0100#;  --  /usr/include/GL/gl.h:330
   GL_ADD : constant := 16#0104#;  --  /usr/include/GL/gl.h:331
   GL_LOAD : constant := 16#0101#;  --  /usr/include/GL/gl.h:332
   GL_MULT : constant := 16#0103#;  --  /usr/include/GL/gl.h:333
   GL_RETURN : constant := 16#0102#;  --  /usr/include/GL/gl.h:334

   GL_ALPHA_TEST : constant := 16#0BC0#;  --  /usr/include/GL/gl.h:337
   GL_ALPHA_TEST_REF : constant := 16#0BC2#;  --  /usr/include/GL/gl.h:338
   GL_ALPHA_TEST_FUNC : constant := 16#0BC1#;  --  /usr/include/GL/gl.h:339

   GL_BLEND : constant := 16#0BE2#;  --  /usr/include/GL/gl.h:342
   GL_BLEND_SRC : constant := 16#0BE1#;  --  /usr/include/GL/gl.h:343
   GL_BLEND_DST : constant := 16#0BE0#;  --  /usr/include/GL/gl.h:344
   GL_ZERO : constant := 0;  --  /usr/include/GL/gl.h:345
   GL_ONE : constant := 1;  --  /usr/include/GL/gl.h:346
   GL_SRC_COLOR : constant := 16#0300#;  --  /usr/include/GL/gl.h:347
   GL_ONE_MINUS_SRC_COLOR : constant := 16#0301#;  --  /usr/include/GL/gl.h:348
   GL_SRC_ALPHA : constant := 16#0302#;  --  /usr/include/GL/gl.h:349
   GL_ONE_MINUS_SRC_ALPHA : constant := 16#0303#;  --  /usr/include/GL/gl.h:350
   GL_DST_ALPHA : constant := 16#0304#;  --  /usr/include/GL/gl.h:351
   GL_ONE_MINUS_DST_ALPHA : constant := 16#0305#;  --  /usr/include/GL/gl.h:352
   GL_DST_COLOR : constant := 16#0306#;  --  /usr/include/GL/gl.h:353
   GL_ONE_MINUS_DST_COLOR : constant := 16#0307#;  --  /usr/include/GL/gl.h:354
   GL_SRC_ALPHA_SATURATE : constant := 16#0308#;  --  /usr/include/GL/gl.h:355

   GL_FEEDBACK : constant := 16#1C01#;  --  /usr/include/GL/gl.h:358
   GL_RENDER : constant := 16#1C00#;  --  /usr/include/GL/gl.h:359
   GL_SELECT : constant := 16#1C02#;  --  /usr/include/GL/gl.h:360

   GL_2D : constant := 16#0600#;  --  /usr/include/GL/gl.h:363
   GL_3D : constant := 16#0601#;  --  /usr/include/GL/gl.h:364
   GL_3D_COLOR : constant := 16#0602#;  --  /usr/include/GL/gl.h:365
   GL_3D_COLOR_TEXTURE : constant := 16#0603#;  --  /usr/include/GL/gl.h:366
   GL_4D_COLOR_TEXTURE : constant := 16#0604#;  --  /usr/include/GL/gl.h:367
   GL_POINT_TOKEN : constant := 16#0701#;  --  /usr/include/GL/gl.h:368
   GL_LINE_TOKEN : constant := 16#0702#;  --  /usr/include/GL/gl.h:369
   GL_LINE_RESET_TOKEN : constant := 16#0707#;  --  /usr/include/GL/gl.h:370
   GL_POLYGON_TOKEN : constant := 16#0703#;  --  /usr/include/GL/gl.h:371
   GL_BITMAP_TOKEN : constant := 16#0704#;  --  /usr/include/GL/gl.h:372
   GL_DRAW_PIXEL_TOKEN : constant := 16#0705#;  --  /usr/include/GL/gl.h:373
   GL_COPY_PIXEL_TOKEN : constant := 16#0706#;  --  /usr/include/GL/gl.h:374
   GL_PASS_THROUGH_TOKEN : constant := 16#0700#;  --  /usr/include/GL/gl.h:375
   GL_FEEDBACK_BUFFER_POINTER : constant := 16#0DF0#;  --  /usr/include/GL/gl.h:376
   GL_FEEDBACK_BUFFER_SIZE : constant := 16#0DF1#;  --  /usr/include/GL/gl.h:377
   GL_FEEDBACK_BUFFER_TYPE : constant := 16#0DF2#;  --  /usr/include/GL/gl.h:378

   GL_SELECTION_BUFFER_POINTER : constant := 16#0DF3#;  --  /usr/include/GL/gl.h:381
   GL_SELECTION_BUFFER_SIZE : constant := 16#0DF4#;  --  /usr/include/GL/gl.h:382

   GL_FOG : constant := 16#0B60#;  --  /usr/include/GL/gl.h:385
   GL_FOG_MODE : constant := 16#0B65#;  --  /usr/include/GL/gl.h:386
   GL_FOG_DENSITY : constant := 16#0B62#;  --  /usr/include/GL/gl.h:387
   GL_FOG_COLOR : constant := 16#0B66#;  --  /usr/include/GL/gl.h:388
   GL_FOG_INDEX : constant := 16#0B61#;  --  /usr/include/GL/gl.h:389
   GL_FOG_START : constant := 16#0B63#;  --  /usr/include/GL/gl.h:390
   GL_FOG_END : constant := 16#0B64#;  --  /usr/include/GL/gl.h:391
   GL_LINEAR : constant := 16#2601#;  --  /usr/include/GL/gl.h:392
   GL_EXP : constant := 16#0800#;  --  /usr/include/GL/gl.h:393
   GL_EXP2 : constant := 16#0801#;  --  /usr/include/GL/gl.h:394

   GL_LOGIC_OP : constant := 16#0BF1#;  --  /usr/include/GL/gl.h:397
   GL_INDEX_LOGIC_OP : constant := 16#0BF1#;  --  /usr/include/GL/gl.h:398
   GL_COLOR_LOGIC_OP : constant := 16#0BF2#;  --  /usr/include/GL/gl.h:399
   GL_LOGIC_OP_MODE : constant := 16#0BF0#;  --  /usr/include/GL/gl.h:400
   GL_CLEAR : constant := 16#1500#;  --  /usr/include/GL/gl.h:401
   GL_SET : constant := 16#150F#;  --  /usr/include/GL/gl.h:402
   GL_COPY : constant := 16#1503#;  --  /usr/include/GL/gl.h:403
   GL_COPY_INVERTED : constant := 16#150C#;  --  /usr/include/GL/gl.h:404
   GL_NOOP : constant := 16#1505#;  --  /usr/include/GL/gl.h:405
   GL_INVERT : constant := 16#150A#;  --  /usr/include/GL/gl.h:406
   GL_AND : constant := 16#1501#;  --  /usr/include/GL/gl.h:407
   GL_NAND : constant := 16#150E#;  --  /usr/include/GL/gl.h:408
   GL_OR : constant := 16#1507#;  --  /usr/include/GL/gl.h:409
   GL_NOR : constant := 16#1508#;  --  /usr/include/GL/gl.h:410
   GL_XOR : constant := 16#1506#;  --  /usr/include/GL/gl.h:411
   GL_EQUIV : constant := 16#1509#;  --  /usr/include/GL/gl.h:412
   GL_AND_REVERSE : constant := 16#1502#;  --  /usr/include/GL/gl.h:413
   GL_AND_INVERTED : constant := 16#1504#;  --  /usr/include/GL/gl.h:414
   GL_OR_REVERSE : constant := 16#150B#;  --  /usr/include/GL/gl.h:415
   GL_OR_INVERTED : constant := 16#150D#;  --  /usr/include/GL/gl.h:416

   GL_STENCIL_BITS : constant := 16#0D57#;  --  /usr/include/GL/gl.h:419
   GL_STENCIL_TEST : constant := 16#0B90#;  --  /usr/include/GL/gl.h:420
   GL_STENCIL_CLEAR_VALUE : constant := 16#0B91#;  --  /usr/include/GL/gl.h:421
   GL_STENCIL_FUNC : constant := 16#0B92#;  --  /usr/include/GL/gl.h:422
   GL_STENCIL_VALUE_MASK : constant := 16#0B93#;  --  /usr/include/GL/gl.h:423
   GL_STENCIL_FAIL : constant := 16#0B94#;  --  /usr/include/GL/gl.h:424
   GL_STENCIL_PASS_DEPTH_FAIL : constant := 16#0B95#;  --  /usr/include/GL/gl.h:425
   GL_STENCIL_PASS_DEPTH_PASS : constant := 16#0B96#;  --  /usr/include/GL/gl.h:426
   GL_STENCIL_REF : constant := 16#0B97#;  --  /usr/include/GL/gl.h:427
   GL_STENCIL_WRITEMASK : constant := 16#0B98#;  --  /usr/include/GL/gl.h:428
   GL_STENCIL_INDEX : constant := 16#1901#;  --  /usr/include/GL/gl.h:429
   GL_KEEP : constant := 16#1E00#;  --  /usr/include/GL/gl.h:430
   GL_REPLACE : constant := 16#1E01#;  --  /usr/include/GL/gl.h:431
   GL_INCR : constant := 16#1E02#;  --  /usr/include/GL/gl.h:432
   GL_DECR : constant := 16#1E03#;  --  /usr/include/GL/gl.h:433

   GL_NONE : constant := 0;  --  /usr/include/GL/gl.h:436
   GL_LEFT : constant := 16#0406#;  --  /usr/include/GL/gl.h:437
   GL_RIGHT : constant := 16#0407#;  --  /usr/include/GL/gl.h:438

   GL_FRONT_LEFT : constant := 16#0400#;  --  /usr/include/GL/gl.h:442
   GL_FRONT_RIGHT : constant := 16#0401#;  --  /usr/include/GL/gl.h:443
   GL_BACK_LEFT : constant := 16#0402#;  --  /usr/include/GL/gl.h:444
   GL_BACK_RIGHT : constant := 16#0403#;  --  /usr/include/GL/gl.h:445
   GL_AUX0 : constant := 16#0409#;  --  /usr/include/GL/gl.h:446
   GL_AUX1 : constant := 16#040A#;  --  /usr/include/GL/gl.h:447
   GL_AUX2 : constant := 16#040B#;  --  /usr/include/GL/gl.h:448
   GL_AUX3 : constant := 16#040C#;  --  /usr/include/GL/gl.h:449
   GL_COLOR_INDEX : constant := 16#1900#;  --  /usr/include/GL/gl.h:450
   GL_RED : constant := 16#1903#;  --  /usr/include/GL/gl.h:451
   GL_GREEN : constant := 16#1904#;  --  /usr/include/GL/gl.h:452
   GL_BLUE : constant := 16#1905#;  --  /usr/include/GL/gl.h:453
   GL_ALPHA : constant := 16#1906#;  --  /usr/include/GL/gl.h:454
   GL_LUMINANCE : constant := 16#1909#;  --  /usr/include/GL/gl.h:455
   GL_LUMINANCE_ALPHA : constant := 16#190A#;  --  /usr/include/GL/gl.h:456
   GL_ALPHA_BITS : constant := 16#0D55#;  --  /usr/include/GL/gl.h:457
   GL_RED_BITS : constant := 16#0D52#;  --  /usr/include/GL/gl.h:458
   GL_GREEN_BITS : constant := 16#0D53#;  --  /usr/include/GL/gl.h:459
   GL_BLUE_BITS : constant := 16#0D54#;  --  /usr/include/GL/gl.h:460
   GL_INDEX_BITS : constant := 16#0D51#;  --  /usr/include/GL/gl.h:461
   GL_SUBPIXEL_BITS : constant := 16#0D50#;  --  /usr/include/GL/gl.h:462
   GL_AUX_BUFFERS : constant := 16#0C00#;  --  /usr/include/GL/gl.h:463
   GL_READ_BUFFER : constant := 16#0C02#;  --  /usr/include/GL/gl.h:464
   GL_DRAW_BUFFER : constant := 16#0C01#;  --  /usr/include/GL/gl.h:465
   GL_DOUBLEBUFFER : constant := 16#0C32#;  --  /usr/include/GL/gl.h:466
   GL_STEREO : constant := 16#0C33#;  --  /usr/include/GL/gl.h:467
   GL_BITMAP : constant := 16#1A00#;  --  /usr/include/GL/gl.h:468
   GL_COLOR : constant := 16#1800#;  --  /usr/include/GL/gl.h:469
   GL_DEPTH : constant := 16#1801#;  --  /usr/include/GL/gl.h:470
   GL_STENCIL : constant := 16#1802#;  --  /usr/include/GL/gl.h:471
   GL_DITHER : constant := 16#0BD0#;  --  /usr/include/GL/gl.h:472
   GL_RGB : constant := 16#1907#;  --  /usr/include/GL/gl.h:473
   GL_RGBA : constant := 16#1908#;  --  /usr/include/GL/gl.h:474

   GL_MAX_LIST_NESTING : constant := 16#0B31#;  --  /usr/include/GL/gl.h:477
   GL_MAX_EVAL_ORDER : constant := 16#0D30#;  --  /usr/include/GL/gl.h:478
   GL_MAX_LIGHTS : constant := 16#0D31#;  --  /usr/include/GL/gl.h:479
   GL_MAX_CLIP_PLANES : constant := 16#0D32#;  --  /usr/include/GL/gl.h:480
   GL_MAX_TEXTURE_SIZE : constant := 16#0D33#;  --  /usr/include/GL/gl.h:481
   GL_MAX_PIXEL_MAP_TABLE : constant := 16#0D34#;  --  /usr/include/GL/gl.h:482
   GL_MAX_ATTRIB_STACK_DEPTH : constant := 16#0D35#;  --  /usr/include/GL/gl.h:483
   GL_MAX_MODELVIEW_STACK_DEPTH : constant := 16#0D36#;  --  /usr/include/GL/gl.h:484
   GL_MAX_NAME_STACK_DEPTH : constant := 16#0D37#;  --  /usr/include/GL/gl.h:485
   GL_MAX_PROJECTION_STACK_DEPTH : constant := 16#0D38#;  --  /usr/include/GL/gl.h:486
   GL_MAX_TEXTURE_STACK_DEPTH : constant := 16#0D39#;  --  /usr/include/GL/gl.h:487
   GL_MAX_VIEWPORT_DIMS : constant := 16#0D3A#;  --  /usr/include/GL/gl.h:488
   GL_MAX_CLIENT_ATTRIB_STACK_DEPTH : constant := 16#0D3B#;  --  /usr/include/GL/gl.h:489

   GL_ATTRIB_STACK_DEPTH : constant := 16#0BB0#;  --  /usr/include/GL/gl.h:492
   GL_CLIENT_ATTRIB_STACK_DEPTH : constant := 16#0BB1#;  --  /usr/include/GL/gl.h:493
   GL_COLOR_CLEAR_VALUE : constant := 16#0C22#;  --  /usr/include/GL/gl.h:494
   GL_COLOR_WRITEMASK : constant := 16#0C23#;  --  /usr/include/GL/gl.h:495
   GL_CURRENT_INDEX : constant := 16#0B01#;  --  /usr/include/GL/gl.h:496
   GL_CURRENT_COLOR : constant := 16#0B00#;  --  /usr/include/GL/gl.h:497
   GL_CURRENT_NORMAL : constant := 16#0B02#;  --  /usr/include/GL/gl.h:498
   GL_CURRENT_RASTER_COLOR : constant := 16#0B04#;  --  /usr/include/GL/gl.h:499
   GL_CURRENT_RASTER_DISTANCE : constant := 16#0B09#;  --  /usr/include/GL/gl.h:500
   GL_CURRENT_RASTER_INDEX : constant := 16#0B05#;  --  /usr/include/GL/gl.h:501
   GL_CURRENT_RASTER_POSITION : constant := 16#0B07#;  --  /usr/include/GL/gl.h:502
   GL_CURRENT_RASTER_TEXTURE_COORDS : constant := 16#0B06#;  --  /usr/include/GL/gl.h:503
   GL_CURRENT_RASTER_POSITION_VALID : constant := 16#0B08#;  --  /usr/include/GL/gl.h:504
   GL_CURRENT_TEXTURE_COORDS : constant := 16#0B03#;  --  /usr/include/GL/gl.h:505
   GL_INDEX_CLEAR_VALUE : constant := 16#0C20#;  --  /usr/include/GL/gl.h:506
   GL_INDEX_MODE : constant := 16#0C30#;  --  /usr/include/GL/gl.h:507
   GL_INDEX_WRITEMASK : constant := 16#0C21#;  --  /usr/include/GL/gl.h:508
   GL_MODELVIEW_MATRIX : constant := 16#0BA6#;  --  /usr/include/GL/gl.h:509
   GL_MODELVIEW_STACK_DEPTH : constant := 16#0BA3#;  --  /usr/include/GL/gl.h:510
   GL_NAME_STACK_DEPTH : constant := 16#0D70#;  --  /usr/include/GL/gl.h:511
   GL_PROJECTION_MATRIX : constant := 16#0BA7#;  --  /usr/include/GL/gl.h:512
   GL_PROJECTION_STACK_DEPTH : constant := 16#0BA4#;  --  /usr/include/GL/gl.h:513
   GL_RENDER_MODE : constant := 16#0C40#;  --  /usr/include/GL/gl.h:514
   GL_RGBA_MODE : constant := 16#0C31#;  --  /usr/include/GL/gl.h:515
   GL_TEXTURE_MATRIX : constant := 16#0BA8#;  --  /usr/include/GL/gl.h:516
   GL_TEXTURE_STACK_DEPTH : constant := 16#0BA5#;  --  /usr/include/GL/gl.h:517
   GL_VIEWPORT : constant := 16#0BA2#;  --  /usr/include/GL/gl.h:518

   GL_AUTO_NORMAL : constant := 16#0D80#;  --  /usr/include/GL/gl.h:521
   GL_MAP1_COLOR_4 : constant := 16#0D90#;  --  /usr/include/GL/gl.h:522
   GL_MAP1_INDEX : constant := 16#0D91#;  --  /usr/include/GL/gl.h:523
   GL_MAP1_NORMAL : constant := 16#0D92#;  --  /usr/include/GL/gl.h:524
   GL_MAP1_TEXTURE_COORD_1 : constant := 16#0D93#;  --  /usr/include/GL/gl.h:525
   GL_MAP1_TEXTURE_COORD_2 : constant := 16#0D94#;  --  /usr/include/GL/gl.h:526
   GL_MAP1_TEXTURE_COORD_3 : constant := 16#0D95#;  --  /usr/include/GL/gl.h:527
   GL_MAP1_TEXTURE_COORD_4 : constant := 16#0D96#;  --  /usr/include/GL/gl.h:528
   GL_MAP1_VERTEX_3 : constant := 16#0D97#;  --  /usr/include/GL/gl.h:529
   GL_MAP1_VERTEX_4 : constant := 16#0D98#;  --  /usr/include/GL/gl.h:530
   GL_MAP2_COLOR_4 : constant := 16#0DB0#;  --  /usr/include/GL/gl.h:531
   GL_MAP2_INDEX : constant := 16#0DB1#;  --  /usr/include/GL/gl.h:532
   GL_MAP2_NORMAL : constant := 16#0DB2#;  --  /usr/include/GL/gl.h:533
   GL_MAP2_TEXTURE_COORD_1 : constant := 16#0DB3#;  --  /usr/include/GL/gl.h:534
   GL_MAP2_TEXTURE_COORD_2 : constant := 16#0DB4#;  --  /usr/include/GL/gl.h:535
   GL_MAP2_TEXTURE_COORD_3 : constant := 16#0DB5#;  --  /usr/include/GL/gl.h:536
   GL_MAP2_TEXTURE_COORD_4 : constant := 16#0DB6#;  --  /usr/include/GL/gl.h:537
   GL_MAP2_VERTEX_3 : constant := 16#0DB7#;  --  /usr/include/GL/gl.h:538
   GL_MAP2_VERTEX_4 : constant := 16#0DB8#;  --  /usr/include/GL/gl.h:539
   GL_MAP1_GRID_DOMAIN : constant := 16#0DD0#;  --  /usr/include/GL/gl.h:540
   GL_MAP1_GRID_SEGMENTS : constant := 16#0DD1#;  --  /usr/include/GL/gl.h:541
   GL_MAP2_GRID_DOMAIN : constant := 16#0DD2#;  --  /usr/include/GL/gl.h:542
   GL_MAP2_GRID_SEGMENTS : constant := 16#0DD3#;  --  /usr/include/GL/gl.h:543
   GL_COEFF : constant := 16#0A00#;  --  /usr/include/GL/gl.h:544
   GL_ORDER : constant := 16#0A01#;  --  /usr/include/GL/gl.h:545
   GL_DOMAIN : constant := 16#0A02#;  --  /usr/include/GL/gl.h:546

   GL_PERSPECTIVE_CORRECTION_HINT : constant := 16#0C50#;  --  /usr/include/GL/gl.h:549
   GL_POINT_SMOOTH_HINT : constant := 16#0C51#;  --  /usr/include/GL/gl.h:550
   GL_LINE_SMOOTH_HINT : constant := 16#0C52#;  --  /usr/include/GL/gl.h:551
   GL_POLYGON_SMOOTH_HINT : constant := 16#0C53#;  --  /usr/include/GL/gl.h:552
   GL_FOG_HINT : constant := 16#0C54#;  --  /usr/include/GL/gl.h:553
   GL_DONT_CARE : constant := 16#1100#;  --  /usr/include/GL/gl.h:554
   GL_FASTEST : constant := 16#1101#;  --  /usr/include/GL/gl.h:555
   GL_NICEST : constant := 16#1102#;  --  /usr/include/GL/gl.h:556

   GL_SCISSOR_BOX : constant := 16#0C10#;  --  /usr/include/GL/gl.h:559
   GL_SCISSOR_TEST : constant := 16#0C11#;  --  /usr/include/GL/gl.h:560

   GL_MAP_COLOR : constant := 16#0D10#;  --  /usr/include/GL/gl.h:563
   GL_MAP_STENCIL : constant := 16#0D11#;  --  /usr/include/GL/gl.h:564
   GL_INDEX_SHIFT : constant := 16#0D12#;  --  /usr/include/GL/gl.h:565
   GL_INDEX_OFFSET : constant := 16#0D13#;  --  /usr/include/GL/gl.h:566
   GL_RED_SCALE : constant := 16#0D14#;  --  /usr/include/GL/gl.h:567
   GL_RED_BIAS : constant := 16#0D15#;  --  /usr/include/GL/gl.h:568
   GL_GREEN_SCALE : constant := 16#0D18#;  --  /usr/include/GL/gl.h:569
   GL_GREEN_BIAS : constant := 16#0D19#;  --  /usr/include/GL/gl.h:570
   GL_BLUE_SCALE : constant := 16#0D1A#;  --  /usr/include/GL/gl.h:571
   GL_BLUE_BIAS : constant := 16#0D1B#;  --  /usr/include/GL/gl.h:572
   GL_ALPHA_SCALE : constant := 16#0D1C#;  --  /usr/include/GL/gl.h:573
   GL_ALPHA_BIAS : constant := 16#0D1D#;  --  /usr/include/GL/gl.h:574
   GL_DEPTH_SCALE : constant := 16#0D1E#;  --  /usr/include/GL/gl.h:575
   GL_DEPTH_BIAS : constant := 16#0D1F#;  --  /usr/include/GL/gl.h:576
   GL_PIXEL_MAP_S_TO_S_SIZE : constant := 16#0CB1#;  --  /usr/include/GL/gl.h:577
   GL_PIXEL_MAP_I_TO_I_SIZE : constant := 16#0CB0#;  --  /usr/include/GL/gl.h:578
   GL_PIXEL_MAP_I_TO_R_SIZE : constant := 16#0CB2#;  --  /usr/include/GL/gl.h:579
   GL_PIXEL_MAP_I_TO_G_SIZE : constant := 16#0CB3#;  --  /usr/include/GL/gl.h:580
   GL_PIXEL_MAP_I_TO_B_SIZE : constant := 16#0CB4#;  --  /usr/include/GL/gl.h:581
   GL_PIXEL_MAP_I_TO_A_SIZE : constant := 16#0CB5#;  --  /usr/include/GL/gl.h:582
   GL_PIXEL_MAP_R_TO_R_SIZE : constant := 16#0CB6#;  --  /usr/include/GL/gl.h:583
   GL_PIXEL_MAP_G_TO_G_SIZE : constant := 16#0CB7#;  --  /usr/include/GL/gl.h:584
   GL_PIXEL_MAP_B_TO_B_SIZE : constant := 16#0CB8#;  --  /usr/include/GL/gl.h:585
   GL_PIXEL_MAP_A_TO_A_SIZE : constant := 16#0CB9#;  --  /usr/include/GL/gl.h:586
   GL_PIXEL_MAP_S_TO_S : constant := 16#0C71#;  --  /usr/include/GL/gl.h:587
   GL_PIXEL_MAP_I_TO_I : constant := 16#0C70#;  --  /usr/include/GL/gl.h:588
   GL_PIXEL_MAP_I_TO_R : constant := 16#0C72#;  --  /usr/include/GL/gl.h:589
   GL_PIXEL_MAP_I_TO_G : constant := 16#0C73#;  --  /usr/include/GL/gl.h:590
   GL_PIXEL_MAP_I_TO_B : constant := 16#0C74#;  --  /usr/include/GL/gl.h:591
   GL_PIXEL_MAP_I_TO_A : constant := 16#0C75#;  --  /usr/include/GL/gl.h:592
   GL_PIXEL_MAP_R_TO_R : constant := 16#0C76#;  --  /usr/include/GL/gl.h:593
   GL_PIXEL_MAP_G_TO_G : constant := 16#0C77#;  --  /usr/include/GL/gl.h:594
   GL_PIXEL_MAP_B_TO_B : constant := 16#0C78#;  --  /usr/include/GL/gl.h:595
   GL_PIXEL_MAP_A_TO_A : constant := 16#0C79#;  --  /usr/include/GL/gl.h:596
   GL_PACK_ALIGNMENT : constant := 16#0D05#;  --  /usr/include/GL/gl.h:597
   GL_PACK_LSB_FIRST : constant := 16#0D01#;  --  /usr/include/GL/gl.h:598
   GL_PACK_ROW_LENGTH : constant := 16#0D02#;  --  /usr/include/GL/gl.h:599
   GL_PACK_SKIP_PIXELS : constant := 16#0D04#;  --  /usr/include/GL/gl.h:600
   GL_PACK_SKIP_ROWS : constant := 16#0D03#;  --  /usr/include/GL/gl.h:601
   GL_PACK_SWAP_BYTES : constant := 16#0D00#;  --  /usr/include/GL/gl.h:602
   GL_UNPACK_ALIGNMENT : constant := 16#0CF5#;  --  /usr/include/GL/gl.h:603
   GL_UNPACK_LSB_FIRST : constant := 16#0CF1#;  --  /usr/include/GL/gl.h:604
   GL_UNPACK_ROW_LENGTH : constant := 16#0CF2#;  --  /usr/include/GL/gl.h:605
   GL_UNPACK_SKIP_PIXELS : constant := 16#0CF4#;  --  /usr/include/GL/gl.h:606
   GL_UNPACK_SKIP_ROWS : constant := 16#0CF3#;  --  /usr/include/GL/gl.h:607
   GL_UNPACK_SWAP_BYTES : constant := 16#0CF0#;  --  /usr/include/GL/gl.h:608
   GL_ZOOM_X : constant := 16#0D16#;  --  /usr/include/GL/gl.h:609
   GL_ZOOM_Y : constant := 16#0D17#;  --  /usr/include/GL/gl.h:610

   GL_TEXTURE_ENV : constant := 16#2300#;  --  /usr/include/GL/gl.h:613
   GL_TEXTURE_ENV_MODE : constant := 16#2200#;  --  /usr/include/GL/gl.h:614
   GL_TEXTURE_1D : constant := 16#0DE0#;  --  /usr/include/GL/gl.h:615
   GL_TEXTURE_2D : constant := 16#0DE1#;  --  /usr/include/GL/gl.h:616
   GL_TEXTURE_WRAP_S : constant := 16#2802#;  --  /usr/include/GL/gl.h:617
   GL_TEXTURE_WRAP_T : constant := 16#2803#;  --  /usr/include/GL/gl.h:618
   GL_TEXTURE_MAG_FILTER : constant := 16#2800#;  --  /usr/include/GL/gl.h:619
   GL_TEXTURE_MIN_FILTER : constant := 16#2801#;  --  /usr/include/GL/gl.h:620
   GL_TEXTURE_ENV_COLOR : constant := 16#2201#;  --  /usr/include/GL/gl.h:621
   GL_TEXTURE_GEN_S : constant := 16#0C60#;  --  /usr/include/GL/gl.h:622
   GL_TEXTURE_GEN_T : constant := 16#0C61#;  --  /usr/include/GL/gl.h:623
   GL_TEXTURE_GEN_R : constant := 16#0C62#;  --  /usr/include/GL/gl.h:624
   GL_TEXTURE_GEN_Q : constant := 16#0C63#;  --  /usr/include/GL/gl.h:625
   GL_TEXTURE_GEN_MODE : constant := 16#2500#;  --  /usr/include/GL/gl.h:626
   GL_TEXTURE_BORDER_COLOR : constant := 16#1004#;  --  /usr/include/GL/gl.h:627
   GL_TEXTURE_WIDTH : constant := 16#1000#;  --  /usr/include/GL/gl.h:628
   GL_TEXTURE_HEIGHT : constant := 16#1001#;  --  /usr/include/GL/gl.h:629
   GL_TEXTURE_BORDER : constant := 16#1005#;  --  /usr/include/GL/gl.h:630
   GL_TEXTURE_COMPONENTS : constant := 16#1003#;  --  /usr/include/GL/gl.h:631
   GL_TEXTURE_RED_SIZE : constant := 16#805C#;  --  /usr/include/GL/gl.h:632
   GL_TEXTURE_GREEN_SIZE : constant := 16#805D#;  --  /usr/include/GL/gl.h:633
   GL_TEXTURE_BLUE_SIZE : constant := 16#805E#;  --  /usr/include/GL/gl.h:634
   GL_TEXTURE_ALPHA_SIZE : constant := 16#805F#;  --  /usr/include/GL/gl.h:635
   GL_TEXTURE_LUMINANCE_SIZE : constant := 16#8060#;  --  /usr/include/GL/gl.h:636
   GL_TEXTURE_INTENSITY_SIZE : constant := 16#8061#;  --  /usr/include/GL/gl.h:637
   GL_NEAREST_MIPMAP_NEAREST : constant := 16#2700#;  --  /usr/include/GL/gl.h:638
   GL_NEAREST_MIPMAP_LINEAR : constant := 16#2702#;  --  /usr/include/GL/gl.h:639
   GL_LINEAR_MIPMAP_NEAREST : constant := 16#2701#;  --  /usr/include/GL/gl.h:640
   GL_LINEAR_MIPMAP_LINEAR : constant := 16#2703#;  --  /usr/include/GL/gl.h:641
   GL_OBJECT_LINEAR : constant := 16#2401#;  --  /usr/include/GL/gl.h:642
   GL_OBJECT_PLANE : constant := 16#2501#;  --  /usr/include/GL/gl.h:643
   GL_EYE_LINEAR : constant := 16#2400#;  --  /usr/include/GL/gl.h:644
   GL_EYE_PLANE : constant := 16#2502#;  --  /usr/include/GL/gl.h:645
   GL_SPHERE_MAP : constant := 16#2402#;  --  /usr/include/GL/gl.h:646
   GL_DECAL : constant := 16#2101#;  --  /usr/include/GL/gl.h:647
   GL_MODULATE : constant := 16#2100#;  --  /usr/include/GL/gl.h:648
   GL_NEAREST : constant := 16#2600#;  --  /usr/include/GL/gl.h:649
   GL_REPEAT : constant := 16#2901#;  --  /usr/include/GL/gl.h:650
   GL_CLAMP : constant := 16#2900#;  --  /usr/include/GL/gl.h:651
   GL_S : constant := 16#2000#;  --  /usr/include/GL/gl.h:652
   GL_T : constant := 16#2001#;  --  /usr/include/GL/gl.h:653
   GL_R : constant := 16#2002#;  --  /usr/include/GL/gl.h:654
   GL_Q : constant := 16#2003#;  --  /usr/include/GL/gl.h:655

   GL_VENDOR : constant := 16#1F00#;  --  /usr/include/GL/gl.h:658
   GL_RENDERER : constant := 16#1F01#;  --  /usr/include/GL/gl.h:659
   GL_VERSION : constant := 16#1F02#;  --  /usr/include/GL/gl.h:660
   GL_EXTENSIONS : constant := 16#1F03#;  --  /usr/include/GL/gl.h:661

   GL_NO_ERROR : constant := 0;  --  /usr/include/GL/gl.h:664
   GL_INVALID_ENUM : constant := 16#0500#;  --  /usr/include/GL/gl.h:665
   GL_INVALID_VALUE : constant := 16#0501#;  --  /usr/include/GL/gl.h:666
   GL_INVALID_OPERATION : constant := 16#0502#;  --  /usr/include/GL/gl.h:667
   GL_STACK_OVERFLOW : constant := 16#0503#;  --  /usr/include/GL/gl.h:668
   GL_STACK_UNDERFLOW : constant := 16#0504#;  --  /usr/include/GL/gl.h:669
   GL_OUT_OF_MEMORY : constant := 16#0505#;  --  /usr/include/GL/gl.h:670

   GL_CURRENT_BIT : constant := 16#00000001#;  --  /usr/include/GL/gl.h:673
   GL_POINT_BIT : constant := 16#00000002#;  --  /usr/include/GL/gl.h:674
   GL_LINE_BIT : constant := 16#00000004#;  --  /usr/include/GL/gl.h:675
   GL_POLYGON_BIT : constant := 16#00000008#;  --  /usr/include/GL/gl.h:676
   GL_POLYGON_STIPPLE_BIT : constant := 16#00000010#;  --  /usr/include/GL/gl.h:677
   GL_PIXEL_MODE_BIT : constant := 16#00000020#;  --  /usr/include/GL/gl.h:678
   GL_LIGHTING_BIT : constant := 16#00000040#;  --  /usr/include/GL/gl.h:679
   GL_FOG_BIT : constant := 16#00000080#;  --  /usr/include/GL/gl.h:680
   GL_DEPTH_BUFFER_BIT : constant := 16#00000100#;  --  /usr/include/GL/gl.h:681
   GL_ACCUM_BUFFER_BIT : constant := 16#00000200#;  --  /usr/include/GL/gl.h:682
   GL_STENCIL_BUFFER_BIT : constant := 16#00000400#;  --  /usr/include/GL/gl.h:683
   GL_VIEWPORT_BIT : constant := 16#00000800#;  --  /usr/include/GL/gl.h:684
   GL_TRANSFORM_BIT : constant := 16#00001000#;  --  /usr/include/GL/gl.h:685
   GL_ENABLE_BIT : constant := 16#00002000#;  --  /usr/include/GL/gl.h:686
   GL_COLOR_BUFFER_BIT : constant := 16#00004000#;  --  /usr/include/GL/gl.h:687
   GL_HINT_BIT : constant := 16#00008000#;  --  /usr/include/GL/gl.h:688
   GL_EVAL_BIT : constant := 16#00010000#;  --  /usr/include/GL/gl.h:689
   GL_LIST_BIT : constant := 16#00020000#;  --  /usr/include/GL/gl.h:690
   GL_TEXTURE_BIT : constant := 16#00040000#;  --  /usr/include/GL/gl.h:691
   GL_SCISSOR_BIT : constant := 16#00080000#;  --  /usr/include/GL/gl.h:692
   GL_ALL_ATTRIB_BITS : constant := 16#FFFFFFFF#;  --  /usr/include/GL/gl.h:693

   GL_PROXY_TEXTURE_1D : constant := 16#8063#;  --  /usr/include/GL/gl.h:697
   GL_PROXY_TEXTURE_2D : constant := 16#8064#;  --  /usr/include/GL/gl.h:698
   GL_TEXTURE_PRIORITY : constant := 16#8066#;  --  /usr/include/GL/gl.h:699
   GL_TEXTURE_RESIDENT : constant := 16#8067#;  --  /usr/include/GL/gl.h:700
   GL_TEXTURE_BINDING_1D : constant := 16#8068#;  --  /usr/include/GL/gl.h:701
   GL_TEXTURE_BINDING_2D : constant := 16#8069#;  --  /usr/include/GL/gl.h:702
   GL_TEXTURE_INTERNAL_FORMAT : constant := 16#1003#;  --  /usr/include/GL/gl.h:703
   GL_ALPHA4 : constant := 16#803B#;  --  /usr/include/GL/gl.h:704
   GL_ALPHA8 : constant := 16#803C#;  --  /usr/include/GL/gl.h:705
   GL_ALPHA12 : constant := 16#803D#;  --  /usr/include/GL/gl.h:706
   GL_ALPHA16 : constant := 16#803E#;  --  /usr/include/GL/gl.h:707
   GL_LUMINANCE4 : constant := 16#803F#;  --  /usr/include/GL/gl.h:708
   GL_LUMINANCE8 : constant := 16#8040#;  --  /usr/include/GL/gl.h:709
   GL_LUMINANCE12 : constant := 16#8041#;  --  /usr/include/GL/gl.h:710
   GL_LUMINANCE16 : constant := 16#8042#;  --  /usr/include/GL/gl.h:711
   GL_LUMINANCE4_ALPHA4 : constant := 16#8043#;  --  /usr/include/GL/gl.h:712
   GL_LUMINANCE6_ALPHA2 : constant := 16#8044#;  --  /usr/include/GL/gl.h:713
   GL_LUMINANCE8_ALPHA8 : constant := 16#8045#;  --  /usr/include/GL/gl.h:714
   GL_LUMINANCE12_ALPHA4 : constant := 16#8046#;  --  /usr/include/GL/gl.h:715
   GL_LUMINANCE12_ALPHA12 : constant := 16#8047#;  --  /usr/include/GL/gl.h:716
   GL_LUMINANCE16_ALPHA16 : constant := 16#8048#;  --  /usr/include/GL/gl.h:717
   GL_INTENSITY : constant := 16#8049#;  --  /usr/include/GL/gl.h:718
   GL_INTENSITY4 : constant := 16#804A#;  --  /usr/include/GL/gl.h:719
   GL_INTENSITY8 : constant := 16#804B#;  --  /usr/include/GL/gl.h:720
   GL_INTENSITY12 : constant := 16#804C#;  --  /usr/include/GL/gl.h:721
   GL_INTENSITY16 : constant := 16#804D#;  --  /usr/include/GL/gl.h:722
   GL_R3_G3_B2 : constant := 16#2A10#;  --  /usr/include/GL/gl.h:723
   GL_RGB4 : constant := 16#804F#;  --  /usr/include/GL/gl.h:724
   GL_RGB5 : constant := 16#8050#;  --  /usr/include/GL/gl.h:725
   GL_RGB8 : constant := 16#8051#;  --  /usr/include/GL/gl.h:726
   GL_RGB10 : constant := 16#8052#;  --  /usr/include/GL/gl.h:727
   GL_RGB12 : constant := 16#8053#;  --  /usr/include/GL/gl.h:728
   GL_RGB16 : constant := 16#8054#;  --  /usr/include/GL/gl.h:729
   GL_RGBA2 : constant := 16#8055#;  --  /usr/include/GL/gl.h:730
   GL_RGBA4 : constant := 16#8056#;  --  /usr/include/GL/gl.h:731
   GL_RGB5_A1 : constant := 16#8057#;  --  /usr/include/GL/gl.h:732
   GL_RGBA8 : constant := 16#8058#;  --  /usr/include/GL/gl.h:733
   GL_RGB10_A2 : constant := 16#8059#;  --  /usr/include/GL/gl.h:734
   GL_RGBA12 : constant := 16#805A#;  --  /usr/include/GL/gl.h:735
   GL_RGBA16 : constant := 16#805B#;  --  /usr/include/GL/gl.h:736
   GL_CLIENT_PIXEL_STORE_BIT : constant := 16#00000001#;  --  /usr/include/GL/gl.h:737
   GL_CLIENT_VERTEX_ARRAY_BIT : constant := 16#00000002#;  --  /usr/include/GL/gl.h:738
   GL_ALL_CLIENT_ATTRIB_BITS : constant := 16#FFFFFFFF#;  --  /usr/include/GL/gl.h:739
   GL_CLIENT_ALL_ATTRIB_BITS : constant := 16#FFFFFFFF#;  --  /usr/include/GL/gl.h:740

   GL_RESCALE_NORMAL : constant := 16#803A#;  --  /usr/include/GL/gl.h:1451
   GL_CLAMP_TO_EDGE : constant := 16#812F#;  --  /usr/include/GL/gl.h:1452
   GL_MAX_ELEMENTS_VERTICES : constant := 16#80E8#;  --  /usr/include/GL/gl.h:1453
   GL_MAX_ELEMENTS_INDICES : constant := 16#80E9#;  --  /usr/include/GL/gl.h:1454
   GL_BGR : constant := 16#80E0#;  --  /usr/include/GL/gl.h:1455
   GL_BGRA : constant := 16#80E1#;  --  /usr/include/GL/gl.h:1456
   GL_UNSIGNED_BYTE_3_3_2 : constant := 16#8032#;  --  /usr/include/GL/gl.h:1457
   GL_UNSIGNED_BYTE_2_3_3_REV : constant := 16#8362#;  --  /usr/include/GL/gl.h:1458
   GL_UNSIGNED_SHORT_5_6_5 : constant := 16#8363#;  --  /usr/include/GL/gl.h:1459
   GL_UNSIGNED_SHORT_5_6_5_REV : constant := 16#8364#;  --  /usr/include/GL/gl.h:1460
   GL_UNSIGNED_SHORT_4_4_4_4 : constant := 16#8033#;  --  /usr/include/GL/gl.h:1461
   GL_UNSIGNED_SHORT_4_4_4_4_REV : constant := 16#8365#;  --  /usr/include/GL/gl.h:1462
   GL_UNSIGNED_SHORT_5_5_5_1 : constant := 16#8034#;  --  /usr/include/GL/gl.h:1463
   GL_UNSIGNED_SHORT_1_5_5_5_REV : constant := 16#8366#;  --  /usr/include/GL/gl.h:1464
   GL_UNSIGNED_INT_8_8_8_8 : constant := 16#8035#;  --  /usr/include/GL/gl.h:1465
   GL_UNSIGNED_INT_8_8_8_8_REV : constant := 16#8367#;  --  /usr/include/GL/gl.h:1466
   GL_UNSIGNED_INT_10_10_10_2 : constant := 16#8036#;  --  /usr/include/GL/gl.h:1467
   GL_UNSIGNED_INT_2_10_10_10_REV : constant := 16#8368#;  --  /usr/include/GL/gl.h:1468
   GL_LIGHT_MODEL_COLOR_CONTROL : constant := 16#81F8#;  --  /usr/include/GL/gl.h:1469
   GL_SINGLE_COLOR : constant := 16#81F9#;  --  /usr/include/GL/gl.h:1470
   GL_SEPARATE_SPECULAR_COLOR : constant := 16#81FA#;  --  /usr/include/GL/gl.h:1471
   GL_TEXTURE_MIN_LOD : constant := 16#813A#;  --  /usr/include/GL/gl.h:1472
   GL_TEXTURE_MAX_LOD : constant := 16#813B#;  --  /usr/include/GL/gl.h:1473
   GL_TEXTURE_BASE_LEVEL : constant := 16#813C#;  --  /usr/include/GL/gl.h:1474
   GL_TEXTURE_MAX_LEVEL : constant := 16#813D#;  --  /usr/include/GL/gl.h:1475
   GL_SMOOTH_POINT_SIZE_RANGE : constant := 16#0B12#;  --  /usr/include/GL/gl.h:1476
   GL_SMOOTH_POINT_SIZE_GRANULARITY : constant := 16#0B13#;  --  /usr/include/GL/gl.h:1477
   GL_SMOOTH_LINE_WIDTH_RANGE : constant := 16#0B22#;  --  /usr/include/GL/gl.h:1478
   GL_SMOOTH_LINE_WIDTH_GRANULARITY : constant := 16#0B23#;  --  /usr/include/GL/gl.h:1479
   GL_ALIASED_POINT_SIZE_RANGE : constant := 16#846D#;  --  /usr/include/GL/gl.h:1480
   GL_ALIASED_LINE_WIDTH_RANGE : constant := 16#846E#;  --  /usr/include/GL/gl.h:1481
   GL_PACK_SKIP_IMAGES : constant := 16#806B#;  --  /usr/include/GL/gl.h:1482
   GL_PACK_IMAGE_HEIGHT : constant := 16#806C#;  --  /usr/include/GL/gl.h:1483
   GL_UNPACK_SKIP_IMAGES : constant := 16#806D#;  --  /usr/include/GL/gl.h:1484
   GL_UNPACK_IMAGE_HEIGHT : constant := 16#806E#;  --  /usr/include/GL/gl.h:1485
   GL_TEXTURE_3D : constant := 16#806F#;  --  /usr/include/GL/gl.h:1486
   GL_PROXY_TEXTURE_3D : constant := 16#8070#;  --  /usr/include/GL/gl.h:1487
   GL_TEXTURE_DEPTH : constant := 16#8071#;  --  /usr/include/GL/gl.h:1488
   GL_TEXTURE_WRAP_R : constant := 16#8072#;  --  /usr/include/GL/gl.h:1489
   GL_MAX_3D_TEXTURE_SIZE : constant := 16#8073#;  --  /usr/include/GL/gl.h:1490
   GL_TEXTURE_BINDING_3D : constant := 16#806A#;  --  /usr/include/GL/gl.h:1491

   GL_COLOR_TABLE : constant := 16#80D0#;  --  /usr/include/GL/gl.h:1530
   GL_POST_CONVOLUTION_COLOR_TABLE : constant := 16#80D1#;  --  /usr/include/GL/gl.h:1531
   GL_POST_COLOR_MATRIX_COLOR_TABLE : constant := 16#80D2#;  --  /usr/include/GL/gl.h:1532
   GL_PROXY_COLOR_TABLE : constant := 16#80D3#;  --  /usr/include/GL/gl.h:1533
   GL_PROXY_POST_CONVOLUTION_COLOR_TABLE : constant := 16#80D4#;  --  /usr/include/GL/gl.h:1534
   GL_PROXY_POST_COLOR_MATRIX_COLOR_TABLE : constant := 16#80D5#;  --  /usr/include/GL/gl.h:1535
   GL_COLOR_TABLE_SCALE : constant := 16#80D6#;  --  /usr/include/GL/gl.h:1536
   GL_COLOR_TABLE_BIAS : constant := 16#80D7#;  --  /usr/include/GL/gl.h:1537
   GL_COLOR_TABLE_FORMAT : constant := 16#80D8#;  --  /usr/include/GL/gl.h:1538
   GL_COLOR_TABLE_WIDTH : constant := 16#80D9#;  --  /usr/include/GL/gl.h:1539
   GL_COLOR_TABLE_RED_SIZE : constant := 16#80DA#;  --  /usr/include/GL/gl.h:1540
   GL_COLOR_TABLE_GREEN_SIZE : constant := 16#80DB#;  --  /usr/include/GL/gl.h:1541
   GL_COLOR_TABLE_BLUE_SIZE : constant := 16#80DC#;  --  /usr/include/GL/gl.h:1542
   GL_COLOR_TABLE_ALPHA_SIZE : constant := 16#80DD#;  --  /usr/include/GL/gl.h:1543
   GL_COLOR_TABLE_LUMINANCE_SIZE : constant := 16#80DE#;  --  /usr/include/GL/gl.h:1544
   GL_COLOR_TABLE_INTENSITY_SIZE : constant := 16#80DF#;  --  /usr/include/GL/gl.h:1545
   GL_CONVOLUTION_1D : constant := 16#8010#;  --  /usr/include/GL/gl.h:1546
   GL_CONVOLUTION_2D : constant := 16#8011#;  --  /usr/include/GL/gl.h:1547
   GL_SEPARABLE_2D : constant := 16#8012#;  --  /usr/include/GL/gl.h:1548
   GL_CONVOLUTION_BORDER_MODE : constant := 16#8013#;  --  /usr/include/GL/gl.h:1549
   GL_CONVOLUTION_FILTER_SCALE : constant := 16#8014#;  --  /usr/include/GL/gl.h:1550
   GL_CONVOLUTION_FILTER_BIAS : constant := 16#8015#;  --  /usr/include/GL/gl.h:1551
   GL_REDUCE : constant := 16#8016#;  --  /usr/include/GL/gl.h:1552
   GL_CONVOLUTION_FORMAT : constant := 16#8017#;  --  /usr/include/GL/gl.h:1553
   GL_CONVOLUTION_WIDTH : constant := 16#8018#;  --  /usr/include/GL/gl.h:1554
   GL_CONVOLUTION_HEIGHT : constant := 16#8019#;  --  /usr/include/GL/gl.h:1555
   GL_MAX_CONVOLUTION_WIDTH : constant := 16#801A#;  --  /usr/include/GL/gl.h:1556
   GL_MAX_CONVOLUTION_HEIGHT : constant := 16#801B#;  --  /usr/include/GL/gl.h:1557
   GL_POST_CONVOLUTION_RED_SCALE : constant := 16#801C#;  --  /usr/include/GL/gl.h:1558
   GL_POST_CONVOLUTION_GREEN_SCALE : constant := 16#801D#;  --  /usr/include/GL/gl.h:1559
   GL_POST_CONVOLUTION_BLUE_SCALE : constant := 16#801E#;  --  /usr/include/GL/gl.h:1560
   GL_POST_CONVOLUTION_ALPHA_SCALE : constant := 16#801F#;  --  /usr/include/GL/gl.h:1561
   GL_POST_CONVOLUTION_RED_BIAS : constant := 16#8020#;  --  /usr/include/GL/gl.h:1562
   GL_POST_CONVOLUTION_GREEN_BIAS : constant := 16#8021#;  --  /usr/include/GL/gl.h:1563
   GL_POST_CONVOLUTION_BLUE_BIAS : constant := 16#8022#;  --  /usr/include/GL/gl.h:1564
   GL_POST_CONVOLUTION_ALPHA_BIAS : constant := 16#8023#;  --  /usr/include/GL/gl.h:1565
   GL_CONSTANT_BORDER : constant := 16#8151#;  --  /usr/include/GL/gl.h:1566
   GL_REPLICATE_BORDER : constant := 16#8153#;  --  /usr/include/GL/gl.h:1567
   GL_CONVOLUTION_BORDER_COLOR : constant := 16#8154#;  --  /usr/include/GL/gl.h:1568
   GL_COLOR_MATRIX : constant := 16#80B1#;  --  /usr/include/GL/gl.h:1569
   GL_COLOR_MATRIX_STACK_DEPTH : constant := 16#80B2#;  --  /usr/include/GL/gl.h:1570
   GL_MAX_COLOR_MATRIX_STACK_DEPTH : constant := 16#80B3#;  --  /usr/include/GL/gl.h:1571
   GL_POST_COLOR_MATRIX_RED_SCALE : constant := 16#80B4#;  --  /usr/include/GL/gl.h:1572
   GL_POST_COLOR_MATRIX_GREEN_SCALE : constant := 16#80B5#;  --  /usr/include/GL/gl.h:1573
   GL_POST_COLOR_MATRIX_BLUE_SCALE : constant := 16#80B6#;  --  /usr/include/GL/gl.h:1574
   GL_POST_COLOR_MATRIX_ALPHA_SCALE : constant := 16#80B7#;  --  /usr/include/GL/gl.h:1575
   GL_POST_COLOR_MATRIX_RED_BIAS : constant := 16#80B8#;  --  /usr/include/GL/gl.h:1576
   GL_POST_COLOR_MATRIX_GREEN_BIAS : constant := 16#80B9#;  --  /usr/include/GL/gl.h:1577
   GL_POST_COLOR_MATRIX_BLUE_BIAS : constant := 16#80BA#;  --  /usr/include/GL/gl.h:1578
   GL_POST_COLOR_MATRIX_ALPHA_BIAS : constant := 16#80BB#;  --  /usr/include/GL/gl.h:1579
   GL_HISTOGRAM : constant := 16#8024#;  --  /usr/include/GL/gl.h:1580
   GL_PROXY_HISTOGRAM : constant := 16#8025#;  --  /usr/include/GL/gl.h:1581
   GL_HISTOGRAM_WIDTH : constant := 16#8026#;  --  /usr/include/GL/gl.h:1582
   GL_HISTOGRAM_FORMAT : constant := 16#8027#;  --  /usr/include/GL/gl.h:1583
   GL_HISTOGRAM_RED_SIZE : constant := 16#8028#;  --  /usr/include/GL/gl.h:1584
   GL_HISTOGRAM_GREEN_SIZE : constant := 16#8029#;  --  /usr/include/GL/gl.h:1585
   GL_HISTOGRAM_BLUE_SIZE : constant := 16#802A#;  --  /usr/include/GL/gl.h:1586
   GL_HISTOGRAM_ALPHA_SIZE : constant := 16#802B#;  --  /usr/include/GL/gl.h:1587
   GL_HISTOGRAM_LUMINANCE_SIZE : constant := 16#802C#;  --  /usr/include/GL/gl.h:1588
   GL_HISTOGRAM_SINK : constant := 16#802D#;  --  /usr/include/GL/gl.h:1589
   GL_MINMAX : constant := 16#802E#;  --  /usr/include/GL/gl.h:1590
   GL_MINMAX_FORMAT : constant := 16#802F#;  --  /usr/include/GL/gl.h:1591
   GL_MINMAX_SINK : constant := 16#8030#;  --  /usr/include/GL/gl.h:1592
   GL_TABLE_TOO_LARGE : constant := 16#8031#;  --  /usr/include/GL/gl.h:1593
   GL_BLEND_EQUATION : constant := 16#8009#;  --  /usr/include/GL/gl.h:1594

   GL_BLEND_COLOR : constant := 16#8005#;  --  /usr/include/GL/gl.h:1600

   GL_TEXTURE0 : constant := 16#84C0#;  --  /usr/include/GL/gl.h:1719
   GL_TEXTURE1 : constant := 16#84C1#;  --  /usr/include/GL/gl.h:1720
   GL_TEXTURE2 : constant := 16#84C2#;  --  /usr/include/GL/gl.h:1721
   GL_TEXTURE3 : constant := 16#84C3#;  --  /usr/include/GL/gl.h:1722
   GL_TEXTURE4 : constant := 16#84C4#;  --  /usr/include/GL/gl.h:1723
   GL_TEXTURE5 : constant := 16#84C5#;  --  /usr/include/GL/gl.h:1724
   GL_TEXTURE6 : constant := 16#84C6#;  --  /usr/include/GL/gl.h:1725
   GL_TEXTURE7 : constant := 16#84C7#;  --  /usr/include/GL/gl.h:1726
   GL_TEXTURE8 : constant := 16#84C8#;  --  /usr/include/GL/gl.h:1727
   GL_TEXTURE9 : constant := 16#84C9#;  --  /usr/include/GL/gl.h:1728
   GL_TEXTURE10 : constant := 16#84CA#;  --  /usr/include/GL/gl.h:1729
   GL_TEXTURE11 : constant := 16#84CB#;  --  /usr/include/GL/gl.h:1730
   GL_TEXTURE12 : constant := 16#84CC#;  --  /usr/include/GL/gl.h:1731
   GL_TEXTURE13 : constant := 16#84CD#;  --  /usr/include/GL/gl.h:1732
   GL_TEXTURE14 : constant := 16#84CE#;  --  /usr/include/GL/gl.h:1733
   GL_TEXTURE15 : constant := 16#84CF#;  --  /usr/include/GL/gl.h:1734
   GL_TEXTURE16 : constant := 16#84D0#;  --  /usr/include/GL/gl.h:1735
   GL_TEXTURE17 : constant := 16#84D1#;  --  /usr/include/GL/gl.h:1736
   GL_TEXTURE18 : constant := 16#84D2#;  --  /usr/include/GL/gl.h:1737
   GL_TEXTURE19 : constant := 16#84D3#;  --  /usr/include/GL/gl.h:1738
   GL_TEXTURE20 : constant := 16#84D4#;  --  /usr/include/GL/gl.h:1739
   GL_TEXTURE21 : constant := 16#84D5#;  --  /usr/include/GL/gl.h:1740
   GL_TEXTURE22 : constant := 16#84D6#;  --  /usr/include/GL/gl.h:1741
   GL_TEXTURE23 : constant := 16#84D7#;  --  /usr/include/GL/gl.h:1742
   GL_TEXTURE24 : constant := 16#84D8#;  --  /usr/include/GL/gl.h:1743
   GL_TEXTURE25 : constant := 16#84D9#;  --  /usr/include/GL/gl.h:1744
   GL_TEXTURE26 : constant := 16#84DA#;  --  /usr/include/GL/gl.h:1745
   GL_TEXTURE27 : constant := 16#84DB#;  --  /usr/include/GL/gl.h:1746
   GL_TEXTURE28 : constant := 16#84DC#;  --  /usr/include/GL/gl.h:1747
   GL_TEXTURE29 : constant := 16#84DD#;  --  /usr/include/GL/gl.h:1748
   GL_TEXTURE30 : constant := 16#84DE#;  --  /usr/include/GL/gl.h:1749
   GL_TEXTURE31 : constant := 16#84DF#;  --  /usr/include/GL/gl.h:1750
   GL_ACTIVE_TEXTURE : constant := 16#84E0#;  --  /usr/include/GL/gl.h:1751
   GL_CLIENT_ACTIVE_TEXTURE : constant := 16#84E1#;  --  /usr/include/GL/gl.h:1752
   GL_MAX_TEXTURE_UNITS : constant := 16#84E2#;  --  /usr/include/GL/gl.h:1753

   GL_NORMAL_MAP : constant := 16#8511#;  --  /usr/include/GL/gl.h:1755
   GL_REFLECTION_MAP : constant := 16#8512#;  --  /usr/include/GL/gl.h:1756
   GL_TEXTURE_CUBE_MAP : constant := 16#8513#;  --  /usr/include/GL/gl.h:1757
   GL_TEXTURE_BINDING_CUBE_MAP : constant := 16#8514#;  --  /usr/include/GL/gl.h:1758
   GL_TEXTURE_CUBE_MAP_POSITIVE_X : constant := 16#8515#;  --  /usr/include/GL/gl.h:1759
   GL_TEXTURE_CUBE_MAP_NEGATIVE_X : constant := 16#8516#;  --  /usr/include/GL/gl.h:1760
   GL_TEXTURE_CUBE_MAP_POSITIVE_Y : constant := 16#8517#;  --  /usr/include/GL/gl.h:1761
   GL_TEXTURE_CUBE_MAP_NEGATIVE_Y : constant := 16#8518#;  --  /usr/include/GL/gl.h:1762
   GL_TEXTURE_CUBE_MAP_POSITIVE_Z : constant := 16#8519#;  --  /usr/include/GL/gl.h:1763
   GL_TEXTURE_CUBE_MAP_NEGATIVE_Z : constant := 16#851A#;  --  /usr/include/GL/gl.h:1764
   GL_PROXY_TEXTURE_CUBE_MAP : constant := 16#851B#;  --  /usr/include/GL/gl.h:1765
   GL_MAX_CUBE_MAP_TEXTURE_SIZE : constant := 16#851C#;  --  /usr/include/GL/gl.h:1766

   GL_COMPRESSED_ALPHA : constant := 16#84E9#;  --  /usr/include/GL/gl.h:1768
   GL_COMPRESSED_LUMINANCE : constant := 16#84EA#;  --  /usr/include/GL/gl.h:1769
   GL_COMPRESSED_LUMINANCE_ALPHA : constant := 16#84EB#;  --  /usr/include/GL/gl.h:1770
   GL_COMPRESSED_INTENSITY : constant := 16#84EC#;  --  /usr/include/GL/gl.h:1771
   GL_COMPRESSED_RGB : constant := 16#84ED#;  --  /usr/include/GL/gl.h:1772
   GL_COMPRESSED_RGBA : constant := 16#84EE#;  --  /usr/include/GL/gl.h:1773
   GL_TEXTURE_COMPRESSION_HINT : constant := 16#84EF#;  --  /usr/include/GL/gl.h:1774
   GL_TEXTURE_COMPRESSED_IMAGE_SIZE : constant := 16#86A0#;  --  /usr/include/GL/gl.h:1775
   GL_TEXTURE_COMPRESSED : constant := 16#86A1#;  --  /usr/include/GL/gl.h:1776
   GL_NUM_COMPRESSED_TEXTURE_FORMATS : constant := 16#86A2#;  --  /usr/include/GL/gl.h:1777
   GL_COMPRESSED_TEXTURE_FORMATS : constant := 16#86A3#;  --  /usr/include/GL/gl.h:1778

   GL_MULTISAMPLE : constant := 16#809D#;  --  /usr/include/GL/gl.h:1780
   GL_SAMPLE_ALPHA_TO_COVERAGE : constant := 16#809E#;  --  /usr/include/GL/gl.h:1781
   GL_SAMPLE_ALPHA_TO_ONE : constant := 16#809F#;  --  /usr/include/GL/gl.h:1782
   GL_SAMPLE_COVERAGE : constant := 16#80A0#;  --  /usr/include/GL/gl.h:1783
   GL_SAMPLE_BUFFERS : constant := 16#80A8#;  --  /usr/include/GL/gl.h:1784
   GL_SAMPLES : constant := 16#80A9#;  --  /usr/include/GL/gl.h:1785
   GL_SAMPLE_COVERAGE_VALUE : constant := 16#80AA#;  --  /usr/include/GL/gl.h:1786
   GL_SAMPLE_COVERAGE_INVERT : constant := 16#80AB#;  --  /usr/include/GL/gl.h:1787
   GL_MULTISAMPLE_BIT : constant := 16#20000000#;  --  /usr/include/GL/gl.h:1788

   GL_TRANSPOSE_MODELVIEW_MATRIX : constant := 16#84E3#;  --  /usr/include/GL/gl.h:1790
   GL_TRANSPOSE_PROJECTION_MATRIX : constant := 16#84E4#;  --  /usr/include/GL/gl.h:1791
   GL_TRANSPOSE_TEXTURE_MATRIX : constant := 16#84E5#;  --  /usr/include/GL/gl.h:1792
   GL_TRANSPOSE_COLOR_MATRIX : constant := 16#84E6#;  --  /usr/include/GL/gl.h:1793

   GL_COMBINE : constant := 16#8570#;  --  /usr/include/GL/gl.h:1795
   GL_COMBINE_RGB : constant := 16#8571#;  --  /usr/include/GL/gl.h:1796
   GL_COMBINE_ALPHA : constant := 16#8572#;  --  /usr/include/GL/gl.h:1797
   GL_SOURCE0_RGB : constant := 16#8580#;  --  /usr/include/GL/gl.h:1798
   GL_SOURCE1_RGB : constant := 16#8581#;  --  /usr/include/GL/gl.h:1799
   GL_SOURCE2_RGB : constant := 16#8582#;  --  /usr/include/GL/gl.h:1800
   GL_SOURCE0_ALPHA : constant := 16#8588#;  --  /usr/include/GL/gl.h:1801
   GL_SOURCE1_ALPHA : constant := 16#8589#;  --  /usr/include/GL/gl.h:1802
   GL_SOURCE2_ALPHA : constant := 16#858A#;  --  /usr/include/GL/gl.h:1803
   GL_OPERAND0_RGB : constant := 16#8590#;  --  /usr/include/GL/gl.h:1804
   GL_OPERAND1_RGB : constant := 16#8591#;  --  /usr/include/GL/gl.h:1805
   GL_OPERAND2_RGB : constant := 16#8592#;  --  /usr/include/GL/gl.h:1806
   GL_OPERAND0_ALPHA : constant := 16#8598#;  --  /usr/include/GL/gl.h:1807
   GL_OPERAND1_ALPHA : constant := 16#8599#;  --  /usr/include/GL/gl.h:1808
   GL_OPERAND2_ALPHA : constant := 16#859A#;  --  /usr/include/GL/gl.h:1809
   GL_RGB_SCALE : constant := 16#8573#;  --  /usr/include/GL/gl.h:1810
   GL_ADD_SIGNED : constant := 16#8574#;  --  /usr/include/GL/gl.h:1811
   GL_INTERPOLATE : constant := 16#8575#;  --  /usr/include/GL/gl.h:1812
   GL_SUBTRACT : constant := 16#84E7#;  --  /usr/include/GL/gl.h:1813
   GL_CONSTANT : constant := 16#8576#;  --  /usr/include/GL/gl.h:1814
   GL_PRIMARY_COLOR : constant := 16#8577#;  --  /usr/include/GL/gl.h:1815
   GL_PREVIOUS : constant := 16#8578#;  --  /usr/include/GL/gl.h:1816

   GL_DOT3_RGB : constant := 16#86AE#;  --  /usr/include/GL/gl.h:1818
   GL_DOT3_RGBA : constant := 16#86AF#;  --  /usr/include/GL/gl.h:1819

   GL_CLAMP_TO_BORDER : constant := 16#812D#;  --  /usr/include/GL/gl.h:1821

   GL_ARB_multitexture : constant := 1;  --  /usr/include/GL/gl.h:1933

   GL_TEXTURE0_ARB : constant := 16#84C0#;  --  /usr/include/GL/gl.h:1935
   GL_TEXTURE1_ARB : constant := 16#84C1#;  --  /usr/include/GL/gl.h:1936
   GL_TEXTURE2_ARB : constant := 16#84C2#;  --  /usr/include/GL/gl.h:1937
   GL_TEXTURE3_ARB : constant := 16#84C3#;  --  /usr/include/GL/gl.h:1938
   GL_TEXTURE4_ARB : constant := 16#84C4#;  --  /usr/include/GL/gl.h:1939
   GL_TEXTURE5_ARB : constant := 16#84C5#;  --  /usr/include/GL/gl.h:1940
   GL_TEXTURE6_ARB : constant := 16#84C6#;  --  /usr/include/GL/gl.h:1941
   GL_TEXTURE7_ARB : constant := 16#84C7#;  --  /usr/include/GL/gl.h:1942
   GL_TEXTURE8_ARB : constant := 16#84C8#;  --  /usr/include/GL/gl.h:1943
   GL_TEXTURE9_ARB : constant := 16#84C9#;  --  /usr/include/GL/gl.h:1944
   GL_TEXTURE10_ARB : constant := 16#84CA#;  --  /usr/include/GL/gl.h:1945
   GL_TEXTURE11_ARB : constant := 16#84CB#;  --  /usr/include/GL/gl.h:1946
   GL_TEXTURE12_ARB : constant := 16#84CC#;  --  /usr/include/GL/gl.h:1947
   GL_TEXTURE13_ARB : constant := 16#84CD#;  --  /usr/include/GL/gl.h:1948
   GL_TEXTURE14_ARB : constant := 16#84CE#;  --  /usr/include/GL/gl.h:1949
   GL_TEXTURE15_ARB : constant := 16#84CF#;  --  /usr/include/GL/gl.h:1950
   GL_TEXTURE16_ARB : constant := 16#84D0#;  --  /usr/include/GL/gl.h:1951
   GL_TEXTURE17_ARB : constant := 16#84D1#;  --  /usr/include/GL/gl.h:1952
   GL_TEXTURE18_ARB : constant := 16#84D2#;  --  /usr/include/GL/gl.h:1953
   GL_TEXTURE19_ARB : constant := 16#84D3#;  --  /usr/include/GL/gl.h:1954
   GL_TEXTURE20_ARB : constant := 16#84D4#;  --  /usr/include/GL/gl.h:1955
   GL_TEXTURE21_ARB : constant := 16#84D5#;  --  /usr/include/GL/gl.h:1956
   GL_TEXTURE22_ARB : constant := 16#84D6#;  --  /usr/include/GL/gl.h:1957
   GL_TEXTURE23_ARB : constant := 16#84D7#;  --  /usr/include/GL/gl.h:1958
   GL_TEXTURE24_ARB : constant := 16#84D8#;  --  /usr/include/GL/gl.h:1959
   GL_TEXTURE25_ARB : constant := 16#84D9#;  --  /usr/include/GL/gl.h:1960
   GL_TEXTURE26_ARB : constant := 16#84DA#;  --  /usr/include/GL/gl.h:1961
   GL_TEXTURE27_ARB : constant := 16#84DB#;  --  /usr/include/GL/gl.h:1962
   GL_TEXTURE28_ARB : constant := 16#84DC#;  --  /usr/include/GL/gl.h:1963
   GL_TEXTURE29_ARB : constant := 16#84DD#;  --  /usr/include/GL/gl.h:1964
   GL_TEXTURE30_ARB : constant := 16#84DE#;  --  /usr/include/GL/gl.h:1965
   GL_TEXTURE31_ARB : constant := 16#84DF#;  --  /usr/include/GL/gl.h:1966
   GL_ACTIVE_TEXTURE_ARB : constant := 16#84E0#;  --  /usr/include/GL/gl.h:1967
   GL_CLIENT_ACTIVE_TEXTURE_ARB : constant := 16#84E1#;  --  /usr/include/GL/gl.h:1968
   GL_MAX_TEXTURE_UNITS_ARB : constant := 16#84E2#;  --  /usr/include/GL/gl.h:1969

   GL_MESA_packed_depth_stencil : constant := 1;  --  /usr/include/GL/gl.h:2066

   GL_DEPTH_STENCIL_MESA : constant := 16#8750#;  --  /usr/include/GL/gl.h:2068
   GL_UNSIGNED_INT_24_8_MESA : constant := 16#8751#;  --  /usr/include/GL/gl.h:2069
   GL_UNSIGNED_INT_8_24_REV_MESA : constant := 16#8752#;  --  /usr/include/GL/gl.h:2070
   GL_UNSIGNED_SHORT_15_1_MESA : constant := 16#8753#;  --  /usr/include/GL/gl.h:2071
   GL_UNSIGNED_SHORT_1_15_REV_MESA : constant := 16#8754#;  --  /usr/include/GL/gl.h:2072

   GL_ATI_blend_equation_separate : constant := 1;  --  /usr/include/GL/gl.h:2078

   GL_ALPHA_BLEND_EQUATION_ATI : constant := 16#883D#;  --  /usr/include/GL/gl.h:2080

   GL_OES_EGL_image : constant := 1;  --  /usr/include/GL/gl.h:2094

  -- * Mesa 3-D graphics library
  -- *
  -- * Copyright (C) 1999-2006  Brian Paul   All Rights Reserved.
  -- * Copyright (C) 2009  VMware, Inc.  All Rights Reserved.
  -- *
  -- * Permission is hereby granted, free of charge, to any person obtaining a
  -- * copy of this software and associated documentation files (the "Software"),
  -- * to deal in the Software without restriction, including without limitation
  -- * the rights to use, copy, modify, merge, publish, distribute, sublicense,
  -- * and/or sell copies of the Software, and to permit persons to whom the
  -- * Software is furnished to do so, subject to the following conditions:
  -- *
  -- * The above copyright notice and this permission notice shall be included
  -- * in all copies or substantial portions of the Software.
  -- *
  -- * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
  -- * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  -- * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
  -- * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
  -- * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
  -- * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
  -- * OTHER DEALINGS IN THE SOFTWARE.
  --  

  --*********************************************************************
  -- * Begin system-specific stuff.
  --  

  -- * WINDOWS: Include windows.h here to define APIENTRY.
  -- * It is also useful when applications include this file by
  -- * including only glut.h, since glut.h depends on windows.h.
  -- * Applications needing to include windows.h with parms other
  -- * than "WIN32_LEAN_AND_MEAN" may include windows.h before
  -- * glut.h or gl.h.
  --  

  -- "P" suffix to be used for a pointer to a function  
  -- * End system-specific stuff.
  -- ********************************************************************* 

  -- * Datatypes
  --  

   subtype GLenum is unsigned;  -- /usr/include/GL/gl.h:121

   subtype GLboolean is unsigned_char;  -- /usr/include/GL/gl.h:122

   subtype GLbitfield is unsigned;  -- /usr/include/GL/gl.h:123

   subtype GLvoid is System.Address;  -- /usr/include/GL/gl.h:124

  -- 1-byte signed  
   subtype GLbyte is signed_char;  -- /usr/include/GL/gl.h:125

  -- 2-byte signed  
   subtype GLshort is short;  -- /usr/include/GL/gl.h:126

  -- 4-byte signed  
   subtype GLint is int;  -- /usr/include/GL/gl.h:127

  -- 1-byte unsigned  
   subtype GLubyte is unsigned_char;  -- /usr/include/GL/gl.h:128

  -- 2-byte unsigned  
   subtype GLushort is unsigned_short;  -- /usr/include/GL/gl.h:129

  -- 4-byte unsigned  
   subtype GLuint is unsigned;  -- /usr/include/GL/gl.h:130

  -- 4-byte signed  
   subtype GLsizei is int;  -- /usr/include/GL/gl.h:131

  -- single precision float  
   subtype GLfloat is float;  -- /usr/include/GL/gl.h:132

  -- single precision float in [0,1]  
   subtype GLclampf is float;  -- /usr/include/GL/gl.h:133

  -- double precision float  
   subtype GLdouble is double;  -- /usr/include/GL/gl.h:134

  -- double precision float in [0,1]  
   subtype GLclampd is double;  -- /usr/include/GL/gl.h:135

  -- * Constants
  --  

  -- Boolean values  
  -- Data types  
  -- Primitives  
  -- Vertex Arrays  
  -- Matrix Mode  
  -- Points  
  -- Lines  
  -- Polygons  
  -- Display Lists  
  -- Depth buffer  
  -- Lighting  
  -- User clipping planes  
  -- Accumulation buffer  
  -- Alpha testing  
  -- Blending  
  -- Render Mode  
  -- Feedback  
  -- Selection  
  -- Fog  
  -- Logic Ops  
  -- Stencil  
  -- Buffers, Pixel Drawing/Reading  
  --GL_FRONT					0x0404  
  --GL_BACK					0x0405  
  --GL_FRONT_AND_BACK				0x0408  
  -- Implementation limits  
  -- Gets  
  -- Evaluators  
  -- Hints  
  -- Scissor box  
  -- Pixel Mode / Transfer  
  -- Texture mapping  
  -- Utility  
  -- Errors  
  -- glPush/PopAttrib bits  
  -- OpenGL 1.1  
  -- * Miscellaneous
  --  

   procedure glClearIndex (c : GLfloat);  -- /usr/include/GL/gl.h:748
   pragma Import (C, glClearIndex, "glClearIndex");

   procedure glClearColor
     (red : GLclampf;
      green : GLclampf;
      blue : GLclampf;
      alpha : GLclampf);  -- /usr/include/GL/gl.h:750
   pragma Import (C, glClearColor, "glClearColor");

   procedure glClear (mask : GLbitfield);  -- /usr/include/GL/gl.h:752
   pragma Import (C, glClear, "glClear");

   procedure glIndexMask (mask : GLuint);  -- /usr/include/GL/gl.h:754
   pragma Import (C, glIndexMask, "glIndexMask");

   procedure glColorMask
     (red : GLboolean;
      green : GLboolean;
      blue : GLboolean;
      alpha : GLboolean);  -- /usr/include/GL/gl.h:756
   pragma Import (C, glColorMask, "glColorMask");

   procedure glAlphaFunc (func : GLenum; ref : GLclampf);  -- /usr/include/GL/gl.h:758
   pragma Import (C, glAlphaFunc, "glAlphaFunc");

   procedure glBlendFunc (sfactor : GLenum; dfactor : GLenum);  -- /usr/include/GL/gl.h:760
   pragma Import (C, glBlendFunc, "glBlendFunc");

   procedure glLogicOp (opcode : GLenum);  -- /usr/include/GL/gl.h:762
   pragma Import (C, glLogicOp, "glLogicOp");

   procedure glCullFace (mode : GLenum);  -- /usr/include/GL/gl.h:764
   pragma Import (C, glCullFace, "glCullFace");

   procedure glFrontFace (mode : GLenum);  -- /usr/include/GL/gl.h:766
   pragma Import (C, glFrontFace, "glFrontFace");

   procedure glPointSize (size : GLfloat);  -- /usr/include/GL/gl.h:768
   pragma Import (C, glPointSize, "glPointSize");

   procedure glLineWidth (width : GLfloat);  -- /usr/include/GL/gl.h:770
   pragma Import (C, glLineWidth, "glLineWidth");

   procedure glLineStipple (factor : GLint; pattern : GLushort);  -- /usr/include/GL/gl.h:772
   pragma Import (C, glLineStipple, "glLineStipple");

   procedure glPolygonMode (face : GLenum; mode : GLenum);  -- /usr/include/GL/gl.h:774
   pragma Import (C, glPolygonMode, "glPolygonMode");

   procedure glPolygonOffset (factor : GLfloat; units : GLfloat);  -- /usr/include/GL/gl.h:776
   pragma Import (C, glPolygonOffset, "glPolygonOffset");

   procedure glPolygonStipple (mask : access GLubyte);  -- /usr/include/GL/gl.h:778
   pragma Import (C, glPolygonStipple, "glPolygonStipple");

   procedure glGetPolygonStipple (mask : access GLubyte);  -- /usr/include/GL/gl.h:780
   pragma Import (C, glGetPolygonStipple, "glGetPolygonStipple");

   procedure glEdgeFlag (flag : GLboolean);  -- /usr/include/GL/gl.h:782
   pragma Import (C, glEdgeFlag, "glEdgeFlag");

   procedure glEdgeFlagv (flag : access GLboolean);  -- /usr/include/GL/gl.h:784
   pragma Import (C, glEdgeFlagv, "glEdgeFlagv");

   procedure glScissor
     (x : GLint;
      y : GLint;
      width : GLsizei;
      height : GLsizei);  -- /usr/include/GL/gl.h:786
   pragma Import (C, glScissor, "glScissor");

   procedure glClipPlane (plane : GLenum; equation : access GLdouble);  -- /usr/include/GL/gl.h:788
   pragma Import (C, glClipPlane, "glClipPlane");

   procedure glGetClipPlane (plane : GLenum; equation : access GLdouble);  -- /usr/include/GL/gl.h:790
   pragma Import (C, glGetClipPlane, "glGetClipPlane");

   procedure glDrawBuffer (mode : GLenum);  -- /usr/include/GL/gl.h:792
   pragma Import (C, glDrawBuffer, "glDrawBuffer");

   procedure glReadBuffer (mode : GLenum);  -- /usr/include/GL/gl.h:794
   pragma Import (C, glReadBuffer, "glReadBuffer");

   procedure glEnable (cap : GLenum);  -- /usr/include/GL/gl.h:796
   pragma Import (C, glEnable, "glEnable");

   procedure glDisable (cap : GLenum);  -- /usr/include/GL/gl.h:798
   pragma Import (C, glDisable, "glDisable");

   function glIsEnabled (cap : GLenum) return GLboolean;  -- /usr/include/GL/gl.h:800
   pragma Import (C, glIsEnabled, "glIsEnabled");

  -- 1.1  
   procedure glEnableClientState (cap : GLenum);  -- /usr/include/GL/gl.h:803
   pragma Import (C, glEnableClientState, "glEnableClientState");

  -- 1.1  
   procedure glDisableClientState (cap : GLenum);  -- /usr/include/GL/gl.h:805
   pragma Import (C, glDisableClientState, "glDisableClientState");

   procedure glGetBooleanv (pname : GLenum; params : access GLboolean);  -- /usr/include/GL/gl.h:808
   pragma Import (C, glGetBooleanv, "glGetBooleanv");

   procedure glGetDoublev (pname : GLenum; params : access GLdouble);  -- /usr/include/GL/gl.h:810
   pragma Import (C, glGetDoublev, "glGetDoublev");

   procedure glGetFloatv (pname : GLenum; params : access GLfloat);  -- /usr/include/GL/gl.h:812
   pragma Import (C, glGetFloatv, "glGetFloatv");

   procedure glGetIntegerv (pname : GLenum; params : access GLint);  -- /usr/include/GL/gl.h:814
   pragma Import (C, glGetIntegerv, "glGetIntegerv");

   procedure glPushAttrib (mask : GLbitfield);  -- /usr/include/GL/gl.h:817
   pragma Import (C, glPushAttrib, "glPushAttrib");

   procedure glPopAttrib;  -- /usr/include/GL/gl.h:819
   pragma Import (C, glPopAttrib, "glPopAttrib");

  -- 1.1  
   procedure glPushClientAttrib (mask : GLbitfield);  -- /usr/include/GL/gl.h:822
   pragma Import (C, glPushClientAttrib, "glPushClientAttrib");

  -- 1.1  
   procedure glPopClientAttrib;  -- /usr/include/GL/gl.h:824
   pragma Import (C, glPopClientAttrib, "glPopClientAttrib");

   function glRenderMode (mode : GLenum) return GLint;  -- /usr/include/GL/gl.h:827
   pragma Import (C, glRenderMode, "glRenderMode");

   function glGetError return GLenum;  -- /usr/include/GL/gl.h:829
   pragma Import (C, glGetError, "glGetError");

   function glGetString (name : GLenum) return access GLubyte;  -- /usr/include/GL/gl.h:831
   pragma Import (C, glGetString, "glGetString");

   procedure glFinish;  -- /usr/include/GL/gl.h:833
   pragma Import (C, glFinish, "glFinish");

   procedure glFlush;  -- /usr/include/GL/gl.h:835
   pragma Import (C, glFlush, "glFlush");

   procedure glHint (target : GLenum; mode : GLenum);  -- /usr/include/GL/gl.h:837
   pragma Import (C, glHint, "glHint");

  -- * Depth Buffer
  --  

   procedure glClearDepth (depth : GLclampd);  -- /usr/include/GL/gl.h:844
   pragma Import (C, glClearDepth, "glClearDepth");

   procedure glDepthFunc (func : GLenum);  -- /usr/include/GL/gl.h:846
   pragma Import (C, glDepthFunc, "glDepthFunc");

   procedure glDepthMask (flag : GLboolean);  -- /usr/include/GL/gl.h:848
   pragma Import (C, glDepthMask, "glDepthMask");

   procedure glDepthRange (near_val : GLclampd; far_val : GLclampd);  -- /usr/include/GL/gl.h:850
   pragma Import (C, glDepthRange, "glDepthRange");

  -- * Accumulation Buffer
  --  

   procedure glClearAccum
     (red : GLfloat;
      green : GLfloat;
      blue : GLfloat;
      alpha : GLfloat);  -- /usr/include/GL/gl.h:857
   pragma Import (C, glClearAccum, "glClearAccum");

   procedure glAccum (op : GLenum; value : GLfloat);  -- /usr/include/GL/gl.h:859
   pragma Import (C, glAccum, "glAccum");

  -- * Transformation
  --  

   procedure glMatrixMode (mode : GLenum);  -- /usr/include/GL/gl.h:866
   pragma Import (C, glMatrixMode, "glMatrixMode");

   procedure glOrtho
     (left : GLdouble;
      right : GLdouble;
      bottom : GLdouble;
      top : GLdouble;
      near_val : GLdouble;
      far_val : GLdouble);  -- /usr/include/GL/gl.h:868
   pragma Import (C, glOrtho, "glOrtho");

   procedure glFrustum
     (left : GLdouble;
      right : GLdouble;
      bottom : GLdouble;
      top : GLdouble;
      near_val : GLdouble;
      far_val : GLdouble);  -- /usr/include/GL/gl.h:872
   pragma Import (C, glFrustum, "glFrustum");

   procedure glViewport
     (x : GLint;
      y : GLint;
      width : GLsizei;
      height : GLsizei);  -- /usr/include/GL/gl.h:876
   pragma Import (C, glViewport, "glViewport");

   procedure glPushMatrix;  -- /usr/include/GL/gl.h:879
   pragma Import (C, glPushMatrix, "glPushMatrix");

   procedure glPopMatrix;  -- /usr/include/GL/gl.h:881
   pragma Import (C, glPopMatrix, "glPopMatrix");

   procedure glLoadIdentity;  -- /usr/include/GL/gl.h:883
   pragma Import (C, glLoadIdentity, "glLoadIdentity");

   procedure glLoadMatrixd (m : access GLdouble);  -- /usr/include/GL/gl.h:885
   pragma Import (C, glLoadMatrixd, "glLoadMatrixd");

   procedure glLoadMatrixf (m : access GLfloat);  -- /usr/include/GL/gl.h:886
   pragma Import (C, glLoadMatrixf, "glLoadMatrixf");

   procedure glMultMatrixd (m : access GLdouble);  -- /usr/include/GL/gl.h:888
   pragma Import (C, glMultMatrixd, "glMultMatrixd");

   procedure glMultMatrixf (m : access GLfloat);  -- /usr/include/GL/gl.h:889
   pragma Import (C, glMultMatrixf, "glMultMatrixf");

   procedure glRotated
     (angle : GLdouble;
      x : GLdouble;
      y : GLdouble;
      z : GLdouble);  -- /usr/include/GL/gl.h:891
   pragma Import (C, glRotated, "glRotated");

   procedure glRotatef
     (angle : GLfloat;
      x : GLfloat;
      y : GLfloat;
      z : GLfloat);  -- /usr/include/GL/gl.h:893
   pragma Import (C, glRotatef, "glRotatef");

   procedure glScaled
     (x : GLdouble;
      y : GLdouble;
      z : GLdouble);  -- /usr/include/GL/gl.h:896
   pragma Import (C, glScaled, "glScaled");

   procedure glScalef
     (x : GLfloat;
      y : GLfloat;
      z : GLfloat);  -- /usr/include/GL/gl.h:897
   pragma Import (C, glScalef, "glScalef");

   procedure glTranslated
     (x : GLdouble;
      y : GLdouble;
      z : GLdouble);  -- /usr/include/GL/gl.h:899
   pragma Import (C, glTranslated, "glTranslated");

   procedure glTranslatef
     (x : GLfloat;
      y : GLfloat;
      z : GLfloat);  -- /usr/include/GL/gl.h:900
   pragma Import (C, glTranslatef, "glTranslatef");

  -- * Display Lists
  --  

   function glIsList (list : GLuint) return GLboolean;  -- /usr/include/GL/gl.h:907
   pragma Import (C, glIsList, "glIsList");

   procedure glDeleteLists (list : GLuint; c_range : GLsizei);  -- /usr/include/GL/gl.h:909
   pragma Import (C, glDeleteLists, "glDeleteLists");

   function glGenLists (c_range : GLsizei) return GLuint;  -- /usr/include/GL/gl.h:911
   pragma Import (C, glGenLists, "glGenLists");

   procedure glNewList (list : GLuint; mode : GLenum);  -- /usr/include/GL/gl.h:913
   pragma Import (C, glNewList, "glNewList");

   procedure glEndList;  -- /usr/include/GL/gl.h:915
   pragma Import (C, glEndList, "glEndList");

   procedure glCallList (list : GLuint);  -- /usr/include/GL/gl.h:917
   pragma Import (C, glCallList, "glCallList");

   procedure glCallLists
     (n : GLsizei;
      c_type : GLenum;
      lists : System.Address);  -- /usr/include/GL/gl.h:919
   pragma Import (C, glCallLists, "glCallLists");

   procedure glListBase (base : GLuint);  -- /usr/include/GL/gl.h:922
   pragma Import (C, glListBase, "glListBase");

  -- * Drawing Functions
  --  

   procedure glBegin (mode : GLenum);  -- /usr/include/GL/gl.h:929
   pragma Import (C, glBegin, "glBegin");

   procedure glEnd;  -- /usr/include/GL/gl.h:931
   pragma Import (C, glEnd, "glEnd");

   procedure glVertex2d (x : GLdouble; y : GLdouble);  -- /usr/include/GL/gl.h:934
   pragma Import (C, glVertex2d, "glVertex2d");

   procedure glVertex2f (x : GLfloat; y : GLfloat);  -- /usr/include/GL/gl.h:935
   pragma Import (C, glVertex2f, "glVertex2f");

   procedure glVertex2i (x : GLint; y : GLint);  -- /usr/include/GL/gl.h:936
   pragma Import (C, glVertex2i, "glVertex2i");

   procedure glVertex2s (x : GLshort; y : GLshort);  -- /usr/include/GL/gl.h:937
   pragma Import (C, glVertex2s, "glVertex2s");

   procedure glVertex3d
     (x : GLdouble;
      y : GLdouble;
      z : GLdouble);  -- /usr/include/GL/gl.h:939
   pragma Import (C, glVertex3d, "glVertex3d");

   procedure glVertex3f
     (x : GLfloat;
      y : GLfloat;
      z : GLfloat);  -- /usr/include/GL/gl.h:940
   pragma Import (C, glVertex3f, "glVertex3f");

   procedure glVertex3i
     (x : GLint;
      y : GLint;
      z : GLint);  -- /usr/include/GL/gl.h:941
   pragma Import (C, glVertex3i, "glVertex3i");

   procedure glVertex3s
     (x : GLshort;
      y : GLshort;
      z : GLshort);  -- /usr/include/GL/gl.h:942
   pragma Import (C, glVertex3s, "glVertex3s");

   procedure glVertex4d
     (x : GLdouble;
      y : GLdouble;
      z : GLdouble;
      w : GLdouble);  -- /usr/include/GL/gl.h:944
   pragma Import (C, glVertex4d, "glVertex4d");

   procedure glVertex4f
     (x : GLfloat;
      y : GLfloat;
      z : GLfloat;
      w : GLfloat);  -- /usr/include/GL/gl.h:945
   pragma Import (C, glVertex4f, "glVertex4f");

   procedure glVertex4i
     (x : GLint;
      y : GLint;
      z : GLint;
      w : GLint);  -- /usr/include/GL/gl.h:946
   pragma Import (C, glVertex4i, "glVertex4i");

   procedure glVertex4s
     (x : GLshort;
      y : GLshort;
      z : GLshort;
      w : GLshort);  -- /usr/include/GL/gl.h:947
   pragma Import (C, glVertex4s, "glVertex4s");

   procedure glVertex2dv (v : access GLdouble);  -- /usr/include/GL/gl.h:949
   pragma Import (C, glVertex2dv, "glVertex2dv");

   procedure glVertex2fv (v : access GLfloat);  -- /usr/include/GL/gl.h:950
   pragma Import (C, glVertex2fv, "glVertex2fv");

   procedure glVertex2iv (v : access GLint);  -- /usr/include/GL/gl.h:951
   pragma Import (C, glVertex2iv, "glVertex2iv");

   procedure glVertex2sv (v : access GLshort);  -- /usr/include/GL/gl.h:952
   pragma Import (C, glVertex2sv, "glVertex2sv");

   procedure glVertex3dv (v : access GLdouble);  -- /usr/include/GL/gl.h:954
   pragma Import (C, glVertex3dv, "glVertex3dv");

   procedure glVertex3fv (v : access GLfloat);  -- /usr/include/GL/gl.h:955
   pragma Import (C, glVertex3fv, "glVertex3fv");

   procedure glVertex3iv (v : access GLint);  -- /usr/include/GL/gl.h:956
   pragma Import (C, glVertex3iv, "glVertex3iv");

   procedure glVertex3sv (v : access GLshort);  -- /usr/include/GL/gl.h:957
   pragma Import (C, glVertex3sv, "glVertex3sv");

   procedure glVertex4dv (v : access GLdouble);  -- /usr/include/GL/gl.h:959
   pragma Import (C, glVertex4dv, "glVertex4dv");

   procedure glVertex4fv (v : access GLfloat);  -- /usr/include/GL/gl.h:960
   pragma Import (C, glVertex4fv, "glVertex4fv");

   procedure glVertex4iv (v : access GLint);  -- /usr/include/GL/gl.h:961
   pragma Import (C, glVertex4iv, "glVertex4iv");

   procedure glVertex4sv (v : access GLshort);  -- /usr/include/GL/gl.h:962
   pragma Import (C, glVertex4sv, "glVertex4sv");

   procedure glNormal3b
     (nx : GLbyte;
      ny : GLbyte;
      nz : GLbyte);  -- /usr/include/GL/gl.h:965
   pragma Import (C, glNormal3b, "glNormal3b");

   procedure glNormal3d
     (nx : GLdouble;
      ny : GLdouble;
      nz : GLdouble);  -- /usr/include/GL/gl.h:966
   pragma Import (C, glNormal3d, "glNormal3d");

   procedure glNormal3f
     (nx : GLfloat;
      ny : GLfloat;
      nz : GLfloat);  -- /usr/include/GL/gl.h:967
   pragma Import (C, glNormal3f, "glNormal3f");

   procedure glNormal3i
     (nx : GLint;
      ny : GLint;
      nz : GLint);  -- /usr/include/GL/gl.h:968
   pragma Import (C, glNormal3i, "glNormal3i");

   procedure glNormal3s
     (nx : GLshort;
      ny : GLshort;
      nz : GLshort);  -- /usr/include/GL/gl.h:969
   pragma Import (C, glNormal3s, "glNormal3s");

   procedure glNormal3bv (v : access GLbyte);  -- /usr/include/GL/gl.h:971
   pragma Import (C, glNormal3bv, "glNormal3bv");

   procedure glNormal3dv (v : access GLdouble);  -- /usr/include/GL/gl.h:972
   pragma Import (C, glNormal3dv, "glNormal3dv");

   procedure glNormal3fv (v : access GLfloat);  -- /usr/include/GL/gl.h:973
   pragma Import (C, glNormal3fv, "glNormal3fv");

   procedure glNormal3iv (v : access GLint);  -- /usr/include/GL/gl.h:974
   pragma Import (C, glNormal3iv, "glNormal3iv");

   procedure glNormal3sv (v : access GLshort);  -- /usr/include/GL/gl.h:975
   pragma Import (C, glNormal3sv, "glNormal3sv");

   procedure glIndexd (c : GLdouble);  -- /usr/include/GL/gl.h:978
   pragma Import (C, glIndexd, "glIndexd");

   procedure glIndexf (c : GLfloat);  -- /usr/include/GL/gl.h:979
   pragma Import (C, glIndexf, "glIndexf");

   procedure glIndexi (c : GLint);  -- /usr/include/GL/gl.h:980
   pragma Import (C, glIndexi, "glIndexi");

   procedure glIndexs (c : GLshort);  -- /usr/include/GL/gl.h:981
   pragma Import (C, glIndexs, "glIndexs");

  -- 1.1  
   procedure glIndexub (c : GLubyte);  -- /usr/include/GL/gl.h:982
   pragma Import (C, glIndexub, "glIndexub");

   procedure glIndexdv (c : access GLdouble);  -- /usr/include/GL/gl.h:984
   pragma Import (C, glIndexdv, "glIndexdv");

   procedure glIndexfv (c : access GLfloat);  -- /usr/include/GL/gl.h:985
   pragma Import (C, glIndexfv, "glIndexfv");

   procedure glIndexiv (c : access GLint);  -- /usr/include/GL/gl.h:986
   pragma Import (C, glIndexiv, "glIndexiv");

   procedure glIndexsv (c : access GLshort);  -- /usr/include/GL/gl.h:987
   pragma Import (C, glIndexsv, "glIndexsv");

  -- 1.1  
   procedure glIndexubv (c : access GLubyte);  -- /usr/include/GL/gl.h:988
   pragma Import (C, glIndexubv, "glIndexubv");

   procedure glColor3b
     (red : GLbyte;
      green : GLbyte;
      blue : GLbyte);  -- /usr/include/GL/gl.h:990
   pragma Import (C, glColor3b, "glColor3b");

   procedure glColor3d
     (red : GLdouble;
      green : GLdouble;
      blue : GLdouble);  -- /usr/include/GL/gl.h:991
   pragma Import (C, glColor3d, "glColor3d");

   procedure glColor3f
     (red : GLfloat;
      green : GLfloat;
      blue : GLfloat);  -- /usr/include/GL/gl.h:992
   pragma Import (C, glColor3f, "glColor3f");

   procedure glColor3i
     (red : GLint;
      green : GLint;
      blue : GLint);  -- /usr/include/GL/gl.h:993
   pragma Import (C, glColor3i, "glColor3i");

   procedure glColor3s
     (red : GLshort;
      green : GLshort;
      blue : GLshort);  -- /usr/include/GL/gl.h:994
   pragma Import (C, glColor3s, "glColor3s");

   procedure glColor3ub
     (red : GLubyte;
      green : GLubyte;
      blue : GLubyte);  -- /usr/include/GL/gl.h:995
   pragma Import (C, glColor3ub, "glColor3ub");

   procedure glColor3ui
     (red : GLuint;
      green : GLuint;
      blue : GLuint);  -- /usr/include/GL/gl.h:996
   pragma Import (C, glColor3ui, "glColor3ui");

   procedure glColor3us
     (red : GLushort;
      green : GLushort;
      blue : GLushort);  -- /usr/include/GL/gl.h:997
   pragma Import (C, glColor3us, "glColor3us");

   procedure glColor4b
     (red : GLbyte;
      green : GLbyte;
      blue : GLbyte;
      alpha : GLbyte);  -- /usr/include/GL/gl.h:999
   pragma Import (C, glColor4b, "glColor4b");

   procedure glColor4d
     (red : GLdouble;
      green : GLdouble;
      blue : GLdouble;
      alpha : GLdouble);  -- /usr/include/GL/gl.h:1001
   pragma Import (C, glColor4d, "glColor4d");

   procedure glColor4f
     (red : GLfloat;
      green : GLfloat;
      blue : GLfloat;
      alpha : GLfloat);  -- /usr/include/GL/gl.h:1003
   pragma Import (C, glColor4f, "glColor4f");

   procedure glColor4i
     (red : GLint;
      green : GLint;
      blue : GLint;
      alpha : GLint);  -- /usr/include/GL/gl.h:1005
   pragma Import (C, glColor4i, "glColor4i");

   procedure glColor4s
     (red : GLshort;
      green : GLshort;
      blue : GLshort;
      alpha : GLshort);  -- /usr/include/GL/gl.h:1007
   pragma Import (C, glColor4s, "glColor4s");

   procedure glColor4ub
     (red : GLubyte;
      green : GLubyte;
      blue : GLubyte;
      alpha : GLubyte);  -- /usr/include/GL/gl.h:1009
   pragma Import (C, glColor4ub, "glColor4ub");

   procedure glColor4ui
     (red : GLuint;
      green : GLuint;
      blue : GLuint;
      alpha : GLuint);  -- /usr/include/GL/gl.h:1011
   pragma Import (C, glColor4ui, "glColor4ui");

   procedure glColor4us
     (red : GLushort;
      green : GLushort;
      blue : GLushort;
      alpha : GLushort);  -- /usr/include/GL/gl.h:1013
   pragma Import (C, glColor4us, "glColor4us");

   procedure glColor3bv (v : access GLbyte);  -- /usr/include/GL/gl.h:1017
   pragma Import (C, glColor3bv, "glColor3bv");

   procedure glColor3dv (v : access GLdouble);  -- /usr/include/GL/gl.h:1018
   pragma Import (C, glColor3dv, "glColor3dv");

   procedure glColor3fv (v : access GLfloat);  -- /usr/include/GL/gl.h:1019
   pragma Import (C, glColor3fv, "glColor3fv");

   procedure glColor3iv (v : access GLint);  -- /usr/include/GL/gl.h:1020
   pragma Import (C, glColor3iv, "glColor3iv");

   procedure glColor3sv (v : access GLshort);  -- /usr/include/GL/gl.h:1021
   pragma Import (C, glColor3sv, "glColor3sv");

   procedure glColor3ubv (v : access GLubyte);  -- /usr/include/GL/gl.h:1022
   pragma Import (C, glColor3ubv, "glColor3ubv");

   procedure glColor3uiv (v : access GLuint);  -- /usr/include/GL/gl.h:1023
   pragma Import (C, glColor3uiv, "glColor3uiv");

   procedure glColor3usv (v : access GLushort);  -- /usr/include/GL/gl.h:1024
   pragma Import (C, glColor3usv, "glColor3usv");

   procedure glColor4bv (v : access GLbyte);  -- /usr/include/GL/gl.h:1026
   pragma Import (C, glColor4bv, "glColor4bv");

   procedure glColor4dv (v : access GLdouble);  -- /usr/include/GL/gl.h:1027
   pragma Import (C, glColor4dv, "glColor4dv");

   procedure glColor4fv (v : access GLfloat);  -- /usr/include/GL/gl.h:1028
   pragma Import (C, glColor4fv, "glColor4fv");

   procedure glColor4iv (v : access GLint);  -- /usr/include/GL/gl.h:1029
   pragma Import (C, glColor4iv, "glColor4iv");

   procedure glColor4sv (v : access GLshort);  -- /usr/include/GL/gl.h:1030
   pragma Import (C, glColor4sv, "glColor4sv");

   procedure glColor4ubv (v : access GLubyte);  -- /usr/include/GL/gl.h:1031
   pragma Import (C, glColor4ubv, "glColor4ubv");

   procedure glColor4uiv (v : access GLuint);  -- /usr/include/GL/gl.h:1032
   pragma Import (C, glColor4uiv, "glColor4uiv");

   procedure glColor4usv (v : access GLushort);  -- /usr/include/GL/gl.h:1033
   pragma Import (C, glColor4usv, "glColor4usv");

   procedure glTexCoord1d (s : GLdouble);  -- /usr/include/GL/gl.h:1036
   pragma Import (C, glTexCoord1d, "glTexCoord1d");

   procedure glTexCoord1f (s : GLfloat);  -- /usr/include/GL/gl.h:1037
   pragma Import (C, glTexCoord1f, "glTexCoord1f");

   procedure glTexCoord1i (s : GLint);  -- /usr/include/GL/gl.h:1038
   pragma Import (C, glTexCoord1i, "glTexCoord1i");

   procedure glTexCoord1s (s : GLshort);  -- /usr/include/GL/gl.h:1039
   pragma Import (C, glTexCoord1s, "glTexCoord1s");

   procedure glTexCoord2d (s : GLdouble; t : GLdouble);  -- /usr/include/GL/gl.h:1041
   pragma Import (C, glTexCoord2d, "glTexCoord2d");

   procedure glTexCoord2f (s : GLfloat; t : GLfloat);  -- /usr/include/GL/gl.h:1042
   pragma Import (C, glTexCoord2f, "glTexCoord2f");

   procedure glTexCoord2i (s : GLint; t : GLint);  -- /usr/include/GL/gl.h:1043
   pragma Import (C, glTexCoord2i, "glTexCoord2i");

   procedure glTexCoord2s (s : GLshort; t : GLshort);  -- /usr/include/GL/gl.h:1044
   pragma Import (C, glTexCoord2s, "glTexCoord2s");

   procedure glTexCoord3d
     (s : GLdouble;
      t : GLdouble;
      r : GLdouble);  -- /usr/include/GL/gl.h:1046
   pragma Import (C, glTexCoord3d, "glTexCoord3d");

   procedure glTexCoord3f
     (s : GLfloat;
      t : GLfloat;
      r : GLfloat);  -- /usr/include/GL/gl.h:1047
   pragma Import (C, glTexCoord3f, "glTexCoord3f");

   procedure glTexCoord3i
     (s : GLint;
      t : GLint;
      r : GLint);  -- /usr/include/GL/gl.h:1048
   pragma Import (C, glTexCoord3i, "glTexCoord3i");

   procedure glTexCoord3s
     (s : GLshort;
      t : GLshort;
      r : GLshort);  -- /usr/include/GL/gl.h:1049
   pragma Import (C, glTexCoord3s, "glTexCoord3s");

   procedure glTexCoord4d
     (s : GLdouble;
      t : GLdouble;
      r : GLdouble;
      q : GLdouble);  -- /usr/include/GL/gl.h:1051
   pragma Import (C, glTexCoord4d, "glTexCoord4d");

   procedure glTexCoord4f
     (s : GLfloat;
      t : GLfloat;
      r : GLfloat;
      q : GLfloat);  -- /usr/include/GL/gl.h:1052
   pragma Import (C, glTexCoord4f, "glTexCoord4f");

   procedure glTexCoord4i
     (s : GLint;
      t : GLint;
      r : GLint;
      q : GLint);  -- /usr/include/GL/gl.h:1053
   pragma Import (C, glTexCoord4i, "glTexCoord4i");

   procedure glTexCoord4s
     (s : GLshort;
      t : GLshort;
      r : GLshort;
      q : GLshort);  -- /usr/include/GL/gl.h:1054
   pragma Import (C, glTexCoord4s, "glTexCoord4s");

   procedure glTexCoord1dv (v : access GLdouble);  -- /usr/include/GL/gl.h:1056
   pragma Import (C, glTexCoord1dv, "glTexCoord1dv");

   procedure glTexCoord1fv (v : access GLfloat);  -- /usr/include/GL/gl.h:1057
   pragma Import (C, glTexCoord1fv, "glTexCoord1fv");

   procedure glTexCoord1iv (v : access GLint);  -- /usr/include/GL/gl.h:1058
   pragma Import (C, glTexCoord1iv, "glTexCoord1iv");

   procedure glTexCoord1sv (v : access GLshort);  -- /usr/include/GL/gl.h:1059
   pragma Import (C, glTexCoord1sv, "glTexCoord1sv");

   procedure glTexCoord2dv (v : access GLdouble);  -- /usr/include/GL/gl.h:1061
   pragma Import (C, glTexCoord2dv, "glTexCoord2dv");

   procedure glTexCoord2fv (v : access GLfloat);  -- /usr/include/GL/gl.h:1062
   pragma Import (C, glTexCoord2fv, "glTexCoord2fv");

   procedure glTexCoord2iv (v : access GLint);  -- /usr/include/GL/gl.h:1063
   pragma Import (C, glTexCoord2iv, "glTexCoord2iv");

   procedure glTexCoord2sv (v : access GLshort);  -- /usr/include/GL/gl.h:1064
   pragma Import (C, glTexCoord2sv, "glTexCoord2sv");

   procedure glTexCoord3dv (v : access GLdouble);  -- /usr/include/GL/gl.h:1066
   pragma Import (C, glTexCoord3dv, "glTexCoord3dv");

   procedure glTexCoord3fv (v : access GLfloat);  -- /usr/include/GL/gl.h:1067
   pragma Import (C, glTexCoord3fv, "glTexCoord3fv");

   procedure glTexCoord3iv (v : access GLint);  -- /usr/include/GL/gl.h:1068
   pragma Import (C, glTexCoord3iv, "glTexCoord3iv");

   procedure glTexCoord3sv (v : access GLshort);  -- /usr/include/GL/gl.h:1069
   pragma Import (C, glTexCoord3sv, "glTexCoord3sv");

   procedure glTexCoord4dv (v : access GLdouble);  -- /usr/include/GL/gl.h:1071
   pragma Import (C, glTexCoord4dv, "glTexCoord4dv");

   procedure glTexCoord4fv (v : access GLfloat);  -- /usr/include/GL/gl.h:1072
   pragma Import (C, glTexCoord4fv, "glTexCoord4fv");

   procedure glTexCoord4iv (v : access GLint);  -- /usr/include/GL/gl.h:1073
   pragma Import (C, glTexCoord4iv, "glTexCoord4iv");

   procedure glTexCoord4sv (v : access GLshort);  -- /usr/include/GL/gl.h:1074
   pragma Import (C, glTexCoord4sv, "glTexCoord4sv");

   procedure glRasterPos2d (x : GLdouble; y : GLdouble);  -- /usr/include/GL/gl.h:1077
   pragma Import (C, glRasterPos2d, "glRasterPos2d");

   procedure glRasterPos2f (x : GLfloat; y : GLfloat);  -- /usr/include/GL/gl.h:1078
   pragma Import (C, glRasterPos2f, "glRasterPos2f");

   procedure glRasterPos2i (x : GLint; y : GLint);  -- /usr/include/GL/gl.h:1079
   pragma Import (C, glRasterPos2i, "glRasterPos2i");

   procedure glRasterPos2s (x : GLshort; y : GLshort);  -- /usr/include/GL/gl.h:1080
   pragma Import (C, glRasterPos2s, "glRasterPos2s");

   procedure glRasterPos3d
     (x : GLdouble;
      y : GLdouble;
      z : GLdouble);  -- /usr/include/GL/gl.h:1082
   pragma Import (C, glRasterPos3d, "glRasterPos3d");

   procedure glRasterPos3f
     (x : GLfloat;
      y : GLfloat;
      z : GLfloat);  -- /usr/include/GL/gl.h:1083
   pragma Import (C, glRasterPos3f, "glRasterPos3f");

   procedure glRasterPos3i
     (x : GLint;
      y : GLint;
      z : GLint);  -- /usr/include/GL/gl.h:1084
   pragma Import (C, glRasterPos3i, "glRasterPos3i");

   procedure glRasterPos3s
     (x : GLshort;
      y : GLshort;
      z : GLshort);  -- /usr/include/GL/gl.h:1085
   pragma Import (C, glRasterPos3s, "glRasterPos3s");

   procedure glRasterPos4d
     (x : GLdouble;
      y : GLdouble;
      z : GLdouble;
      w : GLdouble);  -- /usr/include/GL/gl.h:1087
   pragma Import (C, glRasterPos4d, "glRasterPos4d");

   procedure glRasterPos4f
     (x : GLfloat;
      y : GLfloat;
      z : GLfloat;
      w : GLfloat);  -- /usr/include/GL/gl.h:1088
   pragma Import (C, glRasterPos4f, "glRasterPos4f");

   procedure glRasterPos4i
     (x : GLint;
      y : GLint;
      z : GLint;
      w : GLint);  -- /usr/include/GL/gl.h:1089
   pragma Import (C, glRasterPos4i, "glRasterPos4i");

   procedure glRasterPos4s
     (x : GLshort;
      y : GLshort;
      z : GLshort;
      w : GLshort);  -- /usr/include/GL/gl.h:1090
   pragma Import (C, glRasterPos4s, "glRasterPos4s");

   procedure glRasterPos2dv (v : access GLdouble);  -- /usr/include/GL/gl.h:1092
   pragma Import (C, glRasterPos2dv, "glRasterPos2dv");

   procedure glRasterPos2fv (v : access GLfloat);  -- /usr/include/GL/gl.h:1093
   pragma Import (C, glRasterPos2fv, "glRasterPos2fv");

   procedure glRasterPos2iv (v : access GLint);  -- /usr/include/GL/gl.h:1094
   pragma Import (C, glRasterPos2iv, "glRasterPos2iv");

   procedure glRasterPos2sv (v : access GLshort);  -- /usr/include/GL/gl.h:1095
   pragma Import (C, glRasterPos2sv, "glRasterPos2sv");

   procedure glRasterPos3dv (v : access GLdouble);  -- /usr/include/GL/gl.h:1097
   pragma Import (C, glRasterPos3dv, "glRasterPos3dv");

   procedure glRasterPos3fv (v : access GLfloat);  -- /usr/include/GL/gl.h:1098
   pragma Import (C, glRasterPos3fv, "glRasterPos3fv");

   procedure glRasterPos3iv (v : access GLint);  -- /usr/include/GL/gl.h:1099
   pragma Import (C, glRasterPos3iv, "glRasterPos3iv");

   procedure glRasterPos3sv (v : access GLshort);  -- /usr/include/GL/gl.h:1100
   pragma Import (C, glRasterPos3sv, "glRasterPos3sv");

   procedure glRasterPos4dv (v : access GLdouble);  -- /usr/include/GL/gl.h:1102
   pragma Import (C, glRasterPos4dv, "glRasterPos4dv");

   procedure glRasterPos4fv (v : access GLfloat);  -- /usr/include/GL/gl.h:1103
   pragma Import (C, glRasterPos4fv, "glRasterPos4fv");

   procedure glRasterPos4iv (v : access GLint);  -- /usr/include/GL/gl.h:1104
   pragma Import (C, glRasterPos4iv, "glRasterPos4iv");

   procedure glRasterPos4sv (v : access GLshort);  -- /usr/include/GL/gl.h:1105
   pragma Import (C, glRasterPos4sv, "glRasterPos4sv");

   procedure glRectd
     (x1 : GLdouble;
      y1 : GLdouble;
      x2 : GLdouble;
      y2 : GLdouble);  -- /usr/include/GL/gl.h:1108
   pragma Import (C, glRectd, "glRectd");

   procedure glRectf
     (x1 : GLfloat;
      y1 : GLfloat;
      x2 : GLfloat;
      y2 : GLfloat);  -- /usr/include/GL/gl.h:1109
   pragma Import (C, glRectf, "glRectf");

   procedure glRecti
     (x1 : GLint;
      y1 : GLint;
      x2 : GLint;
      y2 : GLint);  -- /usr/include/GL/gl.h:1110
   pragma Import (C, glRecti, "glRecti");

   procedure glRects
     (x1 : GLshort;
      y1 : GLshort;
      x2 : GLshort;
      y2 : GLshort);  -- /usr/include/GL/gl.h:1111
   pragma Import (C, glRects, "glRects");

   procedure glRectdv (v1 : access GLdouble; v2 : access GLdouble);  -- /usr/include/GL/gl.h:1114
   pragma Import (C, glRectdv, "glRectdv");

   procedure glRectfv (v1 : access GLfloat; v2 : access GLfloat);  -- /usr/include/GL/gl.h:1115
   pragma Import (C, glRectfv, "glRectfv");

   procedure glRectiv (v1 : access GLint; v2 : access GLint);  -- /usr/include/GL/gl.h:1116
   pragma Import (C, glRectiv, "glRectiv");

   procedure glRectsv (v1 : access GLshort; v2 : access GLshort);  -- /usr/include/GL/gl.h:1117
   pragma Import (C, glRectsv, "glRectsv");

  -- * Vertex Arrays  (1.1)
  --  

   procedure glVertexPointer
     (size : GLint;
      c_type : GLenum;
      stride : GLsizei;
      ptr : System.Address);  -- /usr/include/GL/gl.h:1124
   pragma Import (C, glVertexPointer, "glVertexPointer");

   procedure glNormalPointer
     (c_type : GLenum;
      stride : GLsizei;
      ptr : System.Address);  -- /usr/include/GL/gl.h:1127
   pragma Import (C, glNormalPointer, "glNormalPointer");

   procedure glColorPointer
     (size : GLint;
      c_type : GLenum;
      stride : GLsizei;
      ptr : System.Address);  -- /usr/include/GL/gl.h:1130
   pragma Import (C, glColorPointer, "glColorPointer");

   procedure glIndexPointer
     (c_type : GLenum;
      stride : GLsizei;
      ptr : System.Address);  -- /usr/include/GL/gl.h:1133
   pragma Import (C, glIndexPointer, "glIndexPointer");

   procedure glTexCoordPointer
     (size : GLint;
      c_type : GLenum;
      stride : GLsizei;
      ptr : System.Address);  -- /usr/include/GL/gl.h:1136
   pragma Import (C, glTexCoordPointer, "glTexCoordPointer");

   procedure glEdgeFlagPointer (stride : GLsizei; ptr : System.Address);  -- /usr/include/GL/gl.h:1139
   pragma Import (C, glEdgeFlagPointer, "glEdgeFlagPointer");

   procedure glGetPointerv (pname : GLenum; params : System.Address);  -- /usr/include/GL/gl.h:1141
   pragma Import (C, glGetPointerv, "glGetPointerv");

   procedure glArrayElement (i : GLint);  -- /usr/include/GL/gl.h:1143
   pragma Import (C, glArrayElement, "glArrayElement");

   procedure glDrawArrays
     (mode : GLenum;
      first : GLint;
      count : GLsizei);  -- /usr/include/GL/gl.h:1145
   pragma Import (C, glDrawArrays, "glDrawArrays");

   procedure glDrawElements
     (mode : GLenum;
      count : GLsizei;
      c_type : GLenum;
      indices : System.Address);  -- /usr/include/GL/gl.h:1147
   pragma Import (C, glDrawElements, "glDrawElements");

   procedure glInterleavedArrays
     (format : GLenum;
      stride : GLsizei;
      pointer : System.Address);  -- /usr/include/GL/gl.h:1150
   pragma Import (C, glInterleavedArrays, "glInterleavedArrays");

  -- * Lighting
  --  

   procedure glShadeModel (mode : GLenum);  -- /usr/include/GL/gl.h:1157
   pragma Import (C, glShadeModel, "glShadeModel");

   procedure glLightf
     (light : GLenum;
      pname : GLenum;
      param : GLfloat);  -- /usr/include/GL/gl.h:1159
   pragma Import (C, glLightf, "glLightf");

   procedure glLighti
     (light : GLenum;
      pname : GLenum;
      param : GLint);  -- /usr/include/GL/gl.h:1160
   pragma Import (C, glLighti, "glLighti");

   procedure glLightfv
     (light : GLenum;
      pname : GLenum;
      params : access GLfloat);  -- /usr/include/GL/gl.h:1161
   pragma Import (C, glLightfv, "glLightfv");

   procedure glLightiv
     (light : GLenum;
      pname : GLenum;
      params : access GLint);  -- /usr/include/GL/gl.h:1163
   pragma Import (C, glLightiv, "glLightiv");

   procedure glGetLightfv
     (light : GLenum;
      pname : GLenum;
      params : access GLfloat);  -- /usr/include/GL/gl.h:1166
   pragma Import (C, glGetLightfv, "glGetLightfv");

   procedure glGetLightiv
     (light : GLenum;
      pname : GLenum;
      params : access GLint);  -- /usr/include/GL/gl.h:1168
   pragma Import (C, glGetLightiv, "glGetLightiv");

   procedure glLightModelf (pname : GLenum; param : GLfloat);  -- /usr/include/GL/gl.h:1171
   pragma Import (C, glLightModelf, "glLightModelf");

   procedure glLightModeli (pname : GLenum; param : GLint);  -- /usr/include/GL/gl.h:1172
   pragma Import (C, glLightModeli, "glLightModeli");

   procedure glLightModelfv (pname : GLenum; params : access GLfloat);  -- /usr/include/GL/gl.h:1173
   pragma Import (C, glLightModelfv, "glLightModelfv");

   procedure glLightModeliv (pname : GLenum; params : access GLint);  -- /usr/include/GL/gl.h:1174
   pragma Import (C, glLightModeliv, "glLightModeliv");

   procedure glMaterialf
     (face : GLenum;
      pname : GLenum;
      param : GLfloat);  -- /usr/include/GL/gl.h:1176
   pragma Import (C, glMaterialf, "glMaterialf");

   procedure glMateriali
     (face : GLenum;
      pname : GLenum;
      param : GLint);  -- /usr/include/GL/gl.h:1177
   pragma Import (C, glMateriali, "glMateriali");

   procedure glMaterialfv
     (face : GLenum;
      pname : GLenum;
      params : access GLfloat);  -- /usr/include/GL/gl.h:1178
   pragma Import (C, glMaterialfv, "glMaterialfv");

   procedure glMaterialiv
     (face : GLenum;
      pname : GLenum;
      params : access GLint);  -- /usr/include/GL/gl.h:1179
   pragma Import (C, glMaterialiv, "glMaterialiv");

   procedure glGetMaterialfv
     (face : GLenum;
      pname : GLenum;
      params : access GLfloat);  -- /usr/include/GL/gl.h:1181
   pragma Import (C, glGetMaterialfv, "glGetMaterialfv");

   procedure glGetMaterialiv
     (face : GLenum;
      pname : GLenum;
      params : access GLint);  -- /usr/include/GL/gl.h:1182
   pragma Import (C, glGetMaterialiv, "glGetMaterialiv");

   procedure glColorMaterial (face : GLenum; mode : GLenum);  -- /usr/include/GL/gl.h:1184
   pragma Import (C, glColorMaterial, "glColorMaterial");

  -- * Raster functions
  --  

   procedure glPixelZoom (xfactor : GLfloat; yfactor : GLfloat);  -- /usr/include/GL/gl.h:1191
   pragma Import (C, glPixelZoom, "glPixelZoom");

   procedure glPixelStoref (pname : GLenum; param : GLfloat);  -- /usr/include/GL/gl.h:1193
   pragma Import (C, glPixelStoref, "glPixelStoref");

   procedure glPixelStorei (pname : GLenum; param : GLint);  -- /usr/include/GL/gl.h:1194
   pragma Import (C, glPixelStorei, "glPixelStorei");

   procedure glPixelTransferf (pname : GLenum; param : GLfloat);  -- /usr/include/GL/gl.h:1196
   pragma Import (C, glPixelTransferf, "glPixelTransferf");

   procedure glPixelTransferi (pname : GLenum; param : GLint);  -- /usr/include/GL/gl.h:1197
   pragma Import (C, glPixelTransferi, "glPixelTransferi");

   procedure glPixelMapfv
     (map : GLenum;
      mapsize : GLsizei;
      values : access GLfloat);  -- /usr/include/GL/gl.h:1199
   pragma Import (C, glPixelMapfv, "glPixelMapfv");

   procedure glPixelMapuiv
     (map : GLenum;
      mapsize : GLsizei;
      values : access GLuint);  -- /usr/include/GL/gl.h:1201
   pragma Import (C, glPixelMapuiv, "glPixelMapuiv");

   procedure glPixelMapusv
     (map : GLenum;
      mapsize : GLsizei;
      values : access GLushort);  -- /usr/include/GL/gl.h:1203
   pragma Import (C, glPixelMapusv, "glPixelMapusv");

   procedure glGetPixelMapfv (map : GLenum; values : access GLfloat);  -- /usr/include/GL/gl.h:1206
   pragma Import (C, glGetPixelMapfv, "glGetPixelMapfv");

   procedure glGetPixelMapuiv (map : GLenum; values : access GLuint);  -- /usr/include/GL/gl.h:1207
   pragma Import (C, glGetPixelMapuiv, "glGetPixelMapuiv");

   procedure glGetPixelMapusv (map : GLenum; values : access GLushort);  -- /usr/include/GL/gl.h:1208
   pragma Import (C, glGetPixelMapusv, "glGetPixelMapusv");

   procedure glBitmap
     (width : GLsizei;
      height : GLsizei;
      xorig : GLfloat;
      yorig : GLfloat;
      xmove : GLfloat;
      ymove : GLfloat;
      bitmap : access GLubyte);  -- /usr/include/GL/gl.h:1210
   pragma Import (C, glBitmap, "glBitmap");

   procedure glReadPixels
     (x : GLint;
      y : GLint;
      width : GLsizei;
      height : GLsizei;
      format : GLenum;
      c_type : GLenum;
      pixels : System.Address);  -- /usr/include/GL/gl.h:1215
   pragma Import (C, glReadPixels, "glReadPixels");

   procedure glDrawPixels
     (width : GLsizei;
      height : GLsizei;
      format : GLenum;
      c_type : GLenum;
      pixels : System.Address);  -- /usr/include/GL/gl.h:1220
   pragma Import (C, glDrawPixels, "glDrawPixels");

   procedure glCopyPixels
     (x : GLint;
      y : GLint;
      width : GLsizei;
      height : GLsizei;
      c_type : GLenum);  -- /usr/include/GL/gl.h:1224
   pragma Import (C, glCopyPixels, "glCopyPixels");

  -- * Stenciling
  --  

   procedure glStencilFunc
     (func : GLenum;
      ref : GLint;
      mask : GLuint);  -- /usr/include/GL/gl.h:1232
   pragma Import (C, glStencilFunc, "glStencilFunc");

   procedure glStencilMask (mask : GLuint);  -- /usr/include/GL/gl.h:1234
   pragma Import (C, glStencilMask, "glStencilMask");

   procedure glStencilOp
     (fail : GLenum;
      zfail : GLenum;
      zpass : GLenum);  -- /usr/include/GL/gl.h:1236
   pragma Import (C, glStencilOp, "glStencilOp");

   procedure glClearStencil (s : GLint);  -- /usr/include/GL/gl.h:1238
   pragma Import (C, glClearStencil, "glClearStencil");

  -- * Texture mapping
  --  

   procedure glTexGend
     (coord : GLenum;
      pname : GLenum;
      param : GLdouble);  -- /usr/include/GL/gl.h:1246
   pragma Import (C, glTexGend, "glTexGend");

   procedure glTexGenf
     (coord : GLenum;
      pname : GLenum;
      param : GLfloat);  -- /usr/include/GL/gl.h:1247
   pragma Import (C, glTexGenf, "glTexGenf");

   procedure glTexGeni
     (coord : GLenum;
      pname : GLenum;
      param : GLint);  -- /usr/include/GL/gl.h:1248
   pragma Import (C, glTexGeni, "glTexGeni");

   procedure glTexGendv
     (coord : GLenum;
      pname : GLenum;
      params : access GLdouble);  -- /usr/include/GL/gl.h:1250
   pragma Import (C, glTexGendv, "glTexGendv");

   procedure glTexGenfv
     (coord : GLenum;
      pname : GLenum;
      params : access GLfloat);  -- /usr/include/GL/gl.h:1251
   pragma Import (C, glTexGenfv, "glTexGenfv");

   procedure glTexGeniv
     (coord : GLenum;
      pname : GLenum;
      params : access GLint);  -- /usr/include/GL/gl.h:1252
   pragma Import (C, glTexGeniv, "glTexGeniv");

   procedure glGetTexGendv
     (coord : GLenum;
      pname : GLenum;
      params : access GLdouble);  -- /usr/include/GL/gl.h:1254
   pragma Import (C, glGetTexGendv, "glGetTexGendv");

   procedure glGetTexGenfv
     (coord : GLenum;
      pname : GLenum;
      params : access GLfloat);  -- /usr/include/GL/gl.h:1255
   pragma Import (C, glGetTexGenfv, "glGetTexGenfv");

   procedure glGetTexGeniv
     (coord : GLenum;
      pname : GLenum;
      params : access GLint);  -- /usr/include/GL/gl.h:1256
   pragma Import (C, glGetTexGeniv, "glGetTexGeniv");

   procedure glTexEnvf
     (target : GLenum;
      pname : GLenum;
      param : GLfloat);  -- /usr/include/GL/gl.h:1259
   pragma Import (C, glTexEnvf, "glTexEnvf");

   procedure glTexEnvi
     (target : GLenum;
      pname : GLenum;
      param : GLint);  -- /usr/include/GL/gl.h:1260
   pragma Import (C, glTexEnvi, "glTexEnvi");

   procedure glTexEnvfv
     (target : GLenum;
      pname : GLenum;
      params : access GLfloat);  -- /usr/include/GL/gl.h:1262
   pragma Import (C, glTexEnvfv, "glTexEnvfv");

   procedure glTexEnviv
     (target : GLenum;
      pname : GLenum;
      params : access GLint);  -- /usr/include/GL/gl.h:1263
   pragma Import (C, glTexEnviv, "glTexEnviv");

   procedure glGetTexEnvfv
     (target : GLenum;
      pname : GLenum;
      params : access GLfloat);  -- /usr/include/GL/gl.h:1265
   pragma Import (C, glGetTexEnvfv, "glGetTexEnvfv");

   procedure glGetTexEnviv
     (target : GLenum;
      pname : GLenum;
      params : access GLint);  -- /usr/include/GL/gl.h:1266
   pragma Import (C, glGetTexEnviv, "glGetTexEnviv");

   procedure glTexParameterf
     (target : GLenum;
      pname : GLenum;
      param : GLfloat);  -- /usr/include/GL/gl.h:1269
   pragma Import (C, glTexParameterf, "glTexParameterf");

   procedure glTexParameteri
     (target : GLenum;
      pname : GLenum;
      param : GLint);  -- /usr/include/GL/gl.h:1270
   pragma Import (C, glTexParameteri, "glTexParameteri");

   procedure glTexParameterfv
     (target : GLenum;
      pname : GLenum;
      params : access GLfloat);  -- /usr/include/GL/gl.h:1272
   pragma Import (C, glTexParameterfv, "glTexParameterfv");

   procedure glTexParameteriv
     (target : GLenum;
      pname : GLenum;
      params : access GLint);  -- /usr/include/GL/gl.h:1274
   pragma Import (C, glTexParameteriv, "glTexParameteriv");

   procedure glGetTexParameterfv
     (target : GLenum;
      pname : GLenum;
      params : access GLfloat);  -- /usr/include/GL/gl.h:1277
   pragma Import (C, glGetTexParameterfv, "glGetTexParameterfv");

   procedure glGetTexParameteriv
     (target : GLenum;
      pname : GLenum;
      params : access GLint);  -- /usr/include/GL/gl.h:1279
   pragma Import (C, glGetTexParameteriv, "glGetTexParameteriv");

   procedure glGetTexLevelParameterfv
     (target : GLenum;
      level : GLint;
      pname : GLenum;
      params : access GLfloat);  -- /usr/include/GL/gl.h:1282
   pragma Import (C, glGetTexLevelParameterfv, "glGetTexLevelParameterfv");

   procedure glGetTexLevelParameteriv
     (target : GLenum;
      level : GLint;
      pname : GLenum;
      params : access GLint);  -- /usr/include/GL/gl.h:1284
   pragma Import (C, glGetTexLevelParameteriv, "glGetTexLevelParameteriv");

   procedure glTexImage1D
     (target : GLenum;
      level : GLint;
      internalFormat : GLint;
      width : GLsizei;
      border : GLint;
      format : GLenum;
      c_type : GLenum;
      pixels : System.Address);  -- /usr/include/GL/gl.h:1288
   pragma Import (C, glTexImage1D, "glTexImage1D");

   procedure glTexImage2D
     (target : GLenum;
      level : GLint;
      internalFormat : GLint;
      width : GLsizei;
      height : GLsizei;
      border : GLint;
      format : GLenum;
      c_type : GLenum;
      pixels : System.Address);  -- /usr/include/GL/gl.h:1294
   pragma Import (C, glTexImage2D, "glTexImage2D");

   procedure glGetTexImage
     (target : GLenum;
      level : GLint;
      format : GLenum;
      c_type : GLenum;
      pixels : System.Address);  -- /usr/include/GL/gl.h:1300
   pragma Import (C, glGetTexImage, "glGetTexImage");

  -- 1.1 functions  
   procedure glGenTextures (n : GLsizei; textures : access GLuint);  -- /usr/include/GL/gl.h:1307
   pragma Import (C, glGenTextures, "glGenTextures");

   procedure glDeleteTextures (n : GLsizei; textures : access GLuint);  -- /usr/include/GL/gl.h:1309
   pragma Import (C, glDeleteTextures, "glDeleteTextures");

   procedure glBindTexture (target : GLenum; texture : GLuint);  -- /usr/include/GL/gl.h:1311
   pragma Import (C, glBindTexture, "glBindTexture");

   procedure glPrioritizeTextures
     (n : GLsizei;
      textures : access GLuint;
      priorities : access GLclampf);  -- /usr/include/GL/gl.h:1313
   pragma Import (C, glPrioritizeTextures, "glPrioritizeTextures");

   function glAreTexturesResident
     (n : GLsizei;
      textures : access GLuint;
      residences : access GLboolean) return GLboolean;  -- /usr/include/GL/gl.h:1317
   pragma Import (C, glAreTexturesResident, "glAreTexturesResident");

   function glIsTexture (texture : GLuint) return GLboolean;  -- /usr/include/GL/gl.h:1321
   pragma Import (C, glIsTexture, "glIsTexture");

   procedure glTexSubImage1D
     (target : GLenum;
      level : GLint;
      xoffset : GLint;
      width : GLsizei;
      format : GLenum;
      c_type : GLenum;
      pixels : System.Address);  -- /usr/include/GL/gl.h:1324
   pragma Import (C, glTexSubImage1D, "glTexSubImage1D");

   procedure glTexSubImage2D
     (target : GLenum;
      level : GLint;
      xoffset : GLint;
      yoffset : GLint;
      width : GLsizei;
      height : GLsizei;
      format : GLenum;
      c_type : GLenum;
      pixels : System.Address);  -- /usr/include/GL/gl.h:1330
   pragma Import (C, glTexSubImage2D, "glTexSubImage2D");

   procedure glCopyTexImage1D
     (target : GLenum;
      level : GLint;
      internalformat : GLenum;
      x : GLint;
      y : GLint;
      width : GLsizei;
      border : GLint);  -- /usr/include/GL/gl.h:1337
   pragma Import (C, glCopyTexImage1D, "glCopyTexImage1D");

   procedure glCopyTexImage2D
     (target : GLenum;
      level : GLint;
      internalformat : GLenum;
      x : GLint;
      y : GLint;
      width : GLsizei;
      height : GLsizei;
      border : GLint);  -- /usr/include/GL/gl.h:1343
   pragma Import (C, glCopyTexImage2D, "glCopyTexImage2D");

   procedure glCopyTexSubImage1D
     (target : GLenum;
      level : GLint;
      xoffset : GLint;
      x : GLint;
      y : GLint;
      width : GLsizei);  -- /usr/include/GL/gl.h:1350
   pragma Import (C, glCopyTexSubImage1D, "glCopyTexSubImage1D");

   procedure glCopyTexSubImage2D
     (target : GLenum;
      level : GLint;
      xoffset : GLint;
      yoffset : GLint;
      x : GLint;
      y : GLint;
      width : GLsizei;
      height : GLsizei);  -- /usr/include/GL/gl.h:1355
   pragma Import (C, glCopyTexSubImage2D, "glCopyTexSubImage2D");

  -- * Evaluators
  --  

   procedure glMap1d
     (target : GLenum;
      u1 : GLdouble;
      u2 : GLdouble;
      stride : GLint;
      order : GLint;
      points : access GLdouble);  -- /usr/include/GL/gl.h:1365
   pragma Import (C, glMap1d, "glMap1d");

   procedure glMap1f
     (target : GLenum;
      u1 : GLfloat;
      u2 : GLfloat;
      stride : GLint;
      order : GLint;
      points : access GLfloat);  -- /usr/include/GL/gl.h:1368
   pragma Import (C, glMap1f, "glMap1f");

   procedure glMap2d
     (target : GLenum;
      u1 : GLdouble;
      u2 : GLdouble;
      ustride : GLint;
      uorder : GLint;
      v1 : GLdouble;
      v2 : GLdouble;
      vstride : GLint;
      vorder : GLint;
      points : access GLdouble);  -- /usr/include/GL/gl.h:1372
   pragma Import (C, glMap2d, "glMap2d");

   procedure glMap2f
     (target : GLenum;
      u1 : GLfloat;
      u2 : GLfloat;
      ustride : GLint;
      uorder : GLint;
      v1 : GLfloat;
      v2 : GLfloat;
      vstride : GLint;
      vorder : GLint;
      points : access GLfloat);  -- /usr/include/GL/gl.h:1376
   pragma Import (C, glMap2f, "glMap2f");

   procedure glGetMapdv
     (target : GLenum;
      query : GLenum;
      v : access GLdouble);  -- /usr/include/GL/gl.h:1381
   pragma Import (C, glGetMapdv, "glGetMapdv");

   procedure glGetMapfv
     (target : GLenum;
      query : GLenum;
      v : access GLfloat);  -- /usr/include/GL/gl.h:1382
   pragma Import (C, glGetMapfv, "glGetMapfv");

   procedure glGetMapiv
     (target : GLenum;
      query : GLenum;
      v : access GLint);  -- /usr/include/GL/gl.h:1383
   pragma Import (C, glGetMapiv, "glGetMapiv");

   procedure glEvalCoord1d (u : GLdouble);  -- /usr/include/GL/gl.h:1385
   pragma Import (C, glEvalCoord1d, "glEvalCoord1d");

   procedure glEvalCoord1f (u : GLfloat);  -- /usr/include/GL/gl.h:1386
   pragma Import (C, glEvalCoord1f, "glEvalCoord1f");

   procedure glEvalCoord1dv (u : access GLdouble);  -- /usr/include/GL/gl.h:1388
   pragma Import (C, glEvalCoord1dv, "glEvalCoord1dv");

   procedure glEvalCoord1fv (u : access GLfloat);  -- /usr/include/GL/gl.h:1389
   pragma Import (C, glEvalCoord1fv, "glEvalCoord1fv");

   procedure glEvalCoord2d (u : GLdouble; v : GLdouble);  -- /usr/include/GL/gl.h:1391
   pragma Import (C, glEvalCoord2d, "glEvalCoord2d");

   procedure glEvalCoord2f (u : GLfloat; v : GLfloat);  -- /usr/include/GL/gl.h:1392
   pragma Import (C, glEvalCoord2f, "glEvalCoord2f");

   procedure glEvalCoord2dv (u : access GLdouble);  -- /usr/include/GL/gl.h:1394
   pragma Import (C, glEvalCoord2dv, "glEvalCoord2dv");

   procedure glEvalCoord2fv (u : access GLfloat);  -- /usr/include/GL/gl.h:1395
   pragma Import (C, glEvalCoord2fv, "glEvalCoord2fv");

   procedure glMapGrid1d
     (un : GLint;
      u1 : GLdouble;
      u2 : GLdouble);  -- /usr/include/GL/gl.h:1397
   pragma Import (C, glMapGrid1d, "glMapGrid1d");

   procedure glMapGrid1f
     (un : GLint;
      u1 : GLfloat;
      u2 : GLfloat);  -- /usr/include/GL/gl.h:1398
   pragma Import (C, glMapGrid1f, "glMapGrid1f");

   procedure glMapGrid2d
     (un : GLint;
      u1 : GLdouble;
      u2 : GLdouble;
      vn : GLint;
      v1 : GLdouble;
      v2 : GLdouble);  -- /usr/include/GL/gl.h:1400
   pragma Import (C, glMapGrid2d, "glMapGrid2d");

   procedure glMapGrid2f
     (un : GLint;
      u1 : GLfloat;
      u2 : GLfloat;
      vn : GLint;
      v1 : GLfloat;
      v2 : GLfloat);  -- /usr/include/GL/gl.h:1402
   pragma Import (C, glMapGrid2f, "glMapGrid2f");

   procedure glEvalPoint1 (i : GLint);  -- /usr/include/GL/gl.h:1405
   pragma Import (C, glEvalPoint1, "glEvalPoint1");

   procedure glEvalPoint2 (i : GLint; j : GLint);  -- /usr/include/GL/gl.h:1407
   pragma Import (C, glEvalPoint2, "glEvalPoint2");

   procedure glEvalMesh1
     (mode : GLenum;
      i1 : GLint;
      i2 : GLint);  -- /usr/include/GL/gl.h:1409
   pragma Import (C, glEvalMesh1, "glEvalMesh1");

   procedure glEvalMesh2
     (mode : GLenum;
      i1 : GLint;
      i2 : GLint;
      j1 : GLint;
      j2 : GLint);  -- /usr/include/GL/gl.h:1411
   pragma Import (C, glEvalMesh2, "glEvalMesh2");

  -- * Fog
  --  

   procedure glFogf (pname : GLenum; param : GLfloat);  -- /usr/include/GL/gl.h:1418
   pragma Import (C, glFogf, "glFogf");

   procedure glFogi (pname : GLenum; param : GLint);  -- /usr/include/GL/gl.h:1420
   pragma Import (C, glFogi, "glFogi");

   procedure glFogfv (pname : GLenum; params : access GLfloat);  -- /usr/include/GL/gl.h:1422
   pragma Import (C, glFogfv, "glFogfv");

   procedure glFogiv (pname : GLenum; params : access GLint);  -- /usr/include/GL/gl.h:1424
   pragma Import (C, glFogiv, "glFogiv");

  -- * Selection and Feedback
  --  

   procedure glFeedbackBuffer
     (size : GLsizei;
      c_type : GLenum;
      buffer : access GLfloat);  -- /usr/include/GL/gl.h:1431
   pragma Import (C, glFeedbackBuffer, "glFeedbackBuffer");

   procedure glPassThrough (token : GLfloat);  -- /usr/include/GL/gl.h:1433
   pragma Import (C, glPassThrough, "glPassThrough");

   procedure glSelectBuffer (size : GLsizei; buffer : access GLuint);  -- /usr/include/GL/gl.h:1435
   pragma Import (C, glSelectBuffer, "glSelectBuffer");

   procedure glInitNames;  -- /usr/include/GL/gl.h:1437
   pragma Import (C, glInitNames, "glInitNames");

   procedure glLoadName (name : GLuint);  -- /usr/include/GL/gl.h:1439
   pragma Import (C, glLoadName, "glLoadName");

   procedure glPushName (name : GLuint);  -- /usr/include/GL/gl.h:1441
   pragma Import (C, glPushName, "glPushName");

   procedure glPopName;  -- /usr/include/GL/gl.h:1443
   pragma Import (C, glPopName, "glPopName");

  -- * OpenGL 1.2
  --  

   procedure glDrawRangeElements
     (mode : GLenum;
      start : GLuint;
      c_end : GLuint;
      count : GLsizei;
      c_type : GLenum;
      indices : System.Address);  -- /usr/include/GL/gl.h:1493
   pragma Import (C, glDrawRangeElements, "glDrawRangeElements");

   procedure glTexImage3D
     (target : GLenum;
      level : GLint;
      internalFormat : GLint;
      width : GLsizei;
      height : GLsizei;
      depth : GLsizei;
      border : GLint;
      format : GLenum;
      c_type : GLenum;
      pixels : System.Address);  -- /usr/include/GL/gl.h:1496
   pragma Import (C, glTexImage3D, "glTexImage3D");

   procedure glTexSubImage3D
     (target : GLenum;
      level : GLint;
      xoffset : GLint;
      yoffset : GLint;
      zoffset : GLint;
      width : GLsizei;
      height : GLsizei;
      depth : GLsizei;
      format : GLenum;
      c_type : GLenum;
      pixels : System.Address);  -- /usr/include/GL/gl.h:1503
   pragma Import (C, glTexSubImage3D, "glTexSubImage3D");

   procedure glCopyTexSubImage3D
     (target : GLenum;
      level : GLint;
      xoffset : GLint;
      yoffset : GLint;
      zoffset : GLint;
      x : GLint;
      y : GLint;
      width : GLsizei;
      height : GLsizei);  -- /usr/include/GL/gl.h:1510
   pragma Import (C, glCopyTexSubImage3D, "glCopyTexSubImage3D");

   type PFNGLDRAWRANGEELEMENTSPROC is access procedure
        (arg1 : GLenum;
         arg2 : GLuint;
         arg3 : GLuint;
         arg4 : GLsizei;
         arg5 : GLenum;
         arg6 : System.Address);
   pragma Convention (C, PFNGLDRAWRANGEELEMENTSPROC);  -- /usr/include/GL/gl.h:1516

   type PFNGLTEXIMAGE3DPROC is access procedure
        (arg1 : GLenum;
         arg2 : GLint;
         arg3 : GLint;
         arg4 : GLsizei;
         arg5 : GLsizei;
         arg6 : GLsizei;
         arg7 : GLint;
         arg8 : GLenum;
         arg9 : GLenum;
         arg10 : System.Address);
   pragma Convention (C, PFNGLTEXIMAGE3DPROC);  -- /usr/include/GL/gl.h:1517

   type PFNGLTEXSUBIMAGE3DPROC is access procedure
        (arg1 : GLenum;
         arg2 : GLint;
         arg3 : GLint;
         arg4 : GLint;
         arg5 : GLint;
         arg6 : GLsizei;
         arg7 : GLsizei;
         arg8 : GLsizei;
         arg9 : GLenum;
         arg10 : GLenum;
         arg11 : System.Address);
   pragma Convention (C, PFNGLTEXSUBIMAGE3DPROC);  -- /usr/include/GL/gl.h:1518

   type PFNGLCOPYTEXSUBIMAGE3DPROC is access procedure
        (arg1 : GLenum;
         arg2 : GLint;
         arg3 : GLint;
         arg4 : GLint;
         arg5 : GLint;
         arg6 : GLint;
         arg7 : GLint;
         arg8 : GLsizei;
         arg9 : GLsizei);
   pragma Convention (C, PFNGLCOPYTEXSUBIMAGE3DPROC);  -- /usr/include/GL/gl.h:1519

  -- * GL_ARB_imaging
  --  

   procedure glColorTable
     (target : GLenum;
      internalformat : GLenum;
      width : GLsizei;
      format : GLenum;
      c_type : GLenum;
      table : System.Address);  -- /usr/include/GL/gl.h:1603
   pragma Import (C, glColorTable, "glColorTable");

   procedure glColorSubTable
     (target : GLenum;
      start : GLsizei;
      count : GLsizei;
      format : GLenum;
      c_type : GLenum;
      data : System.Address);  -- /usr/include/GL/gl.h:1607
   pragma Import (C, glColorSubTable, "glColorSubTable");

   procedure glColorTableParameteriv
     (target : GLenum;
      pname : GLenum;
      params : access GLint);  -- /usr/include/GL/gl.h:1612
   pragma Import (C, glColorTableParameteriv, "glColorTableParameteriv");

   procedure glColorTableParameterfv
     (target : GLenum;
      pname : GLenum;
      params : access GLfloat);  -- /usr/include/GL/gl.h:1615
   pragma Import (C, glColorTableParameterfv, "glColorTableParameterfv");

   procedure glCopyColorSubTable
     (target : GLenum;
      start : GLsizei;
      x : GLint;
      y : GLint;
      width : GLsizei);  -- /usr/include/GL/gl.h:1618
   pragma Import (C, glCopyColorSubTable, "glCopyColorSubTable");

   procedure glCopyColorTable
     (target : GLenum;
      internalformat : GLenum;
      x : GLint;
      y : GLint;
      width : GLsizei);  -- /usr/include/GL/gl.h:1621
   pragma Import (C, glCopyColorTable, "glCopyColorTable");

   procedure glGetColorTable
     (target : GLenum;
      format : GLenum;
      c_type : GLenum;
      table : System.Address);  -- /usr/include/GL/gl.h:1624
   pragma Import (C, glGetColorTable, "glGetColorTable");

   procedure glGetColorTableParameterfv
     (target : GLenum;
      pname : GLenum;
      params : access GLfloat);  -- /usr/include/GL/gl.h:1627
   pragma Import (C, glGetColorTableParameterfv, "glGetColorTableParameterfv");

   procedure glGetColorTableParameteriv
     (target : GLenum;
      pname : GLenum;
      params : access GLint);  -- /usr/include/GL/gl.h:1630
   pragma Import (C, glGetColorTableParameteriv, "glGetColorTableParameteriv");

   procedure glBlendEquation (mode : GLenum);  -- /usr/include/GL/gl.h:1633
   pragma Import (C, glBlendEquation, "glBlendEquation");

   procedure glBlendColor
     (red : GLclampf;
      green : GLclampf;
      blue : GLclampf;
      alpha : GLclampf);  -- /usr/include/GL/gl.h:1635
   pragma Import (C, glBlendColor, "glBlendColor");

   procedure glHistogram
     (target : GLenum;
      width : GLsizei;
      internalformat : GLenum;
      sink : GLboolean);  -- /usr/include/GL/gl.h:1638
   pragma Import (C, glHistogram, "glHistogram");

   procedure glResetHistogram (target : GLenum);  -- /usr/include/GL/gl.h:1641
   pragma Import (C, glResetHistogram, "glResetHistogram");

   procedure glGetHistogram
     (target : GLenum;
      reset : GLboolean;
      format : GLenum;
      c_type : GLenum;
      values : System.Address);  -- /usr/include/GL/gl.h:1643
   pragma Import (C, glGetHistogram, "glGetHistogram");

   procedure glGetHistogramParameterfv
     (target : GLenum;
      pname : GLenum;
      params : access GLfloat);  -- /usr/include/GL/gl.h:1647
   pragma Import (C, glGetHistogramParameterfv, "glGetHistogramParameterfv");

   procedure glGetHistogramParameteriv
     (target : GLenum;
      pname : GLenum;
      params : access GLint);  -- /usr/include/GL/gl.h:1650
   pragma Import (C, glGetHistogramParameteriv, "glGetHistogramParameteriv");

   procedure glMinmax
     (target : GLenum;
      internalformat : GLenum;
      sink : GLboolean);  -- /usr/include/GL/gl.h:1653
   pragma Import (C, glMinmax, "glMinmax");

   procedure glResetMinmax (target : GLenum);  -- /usr/include/GL/gl.h:1656
   pragma Import (C, glResetMinmax, "glResetMinmax");

   procedure glGetMinmax
     (target : GLenum;
      reset : GLboolean;
      format : GLenum;
      types : GLenum;
      values : System.Address);  -- /usr/include/GL/gl.h:1658
   pragma Import (C, glGetMinmax, "glGetMinmax");

   procedure glGetMinmaxParameterfv
     (target : GLenum;
      pname : GLenum;
      params : access GLfloat);  -- /usr/include/GL/gl.h:1662
   pragma Import (C, glGetMinmaxParameterfv, "glGetMinmaxParameterfv");

   procedure glGetMinmaxParameteriv
     (target : GLenum;
      pname : GLenum;
      params : access GLint);  -- /usr/include/GL/gl.h:1665
   pragma Import (C, glGetMinmaxParameteriv, "glGetMinmaxParameteriv");

   procedure glConvolutionFilter1D
     (target : GLenum;
      internalformat : GLenum;
      width : GLsizei;
      format : GLenum;
      c_type : GLenum;
      image : System.Address);  -- /usr/include/GL/gl.h:1668
   pragma Import (C, glConvolutionFilter1D, "glConvolutionFilter1D");

   procedure glConvolutionFilter2D
     (target : GLenum;
      internalformat : GLenum;
      width : GLsizei;
      height : GLsizei;
      format : GLenum;
      c_type : GLenum;
      image : System.Address);  -- /usr/include/GL/gl.h:1672
   pragma Import (C, glConvolutionFilter2D, "glConvolutionFilter2D");

   procedure glConvolutionParameterf
     (target : GLenum;
      pname : GLenum;
      params : GLfloat);  -- /usr/include/GL/gl.h:1676
   pragma Import (C, glConvolutionParameterf, "glConvolutionParameterf");

   procedure glConvolutionParameterfv
     (target : GLenum;
      pname : GLenum;
      params : access GLfloat);  -- /usr/include/GL/gl.h:1679
   pragma Import (C, glConvolutionParameterfv, "glConvolutionParameterfv");

   procedure glConvolutionParameteri
     (target : GLenum;
      pname : GLenum;
      params : GLint);  -- /usr/include/GL/gl.h:1682
   pragma Import (C, glConvolutionParameteri, "glConvolutionParameteri");

   procedure glConvolutionParameteriv
     (target : GLenum;
      pname : GLenum;
      params : access GLint);  -- /usr/include/GL/gl.h:1685
   pragma Import (C, glConvolutionParameteriv, "glConvolutionParameteriv");

   procedure glCopyConvolutionFilter1D
     (target : GLenum;
      internalformat : GLenum;
      x : GLint;
      y : GLint;
      width : GLsizei);  -- /usr/include/GL/gl.h:1688
   pragma Import (C, glCopyConvolutionFilter1D, "glCopyConvolutionFilter1D");

   procedure glCopyConvolutionFilter2D
     (target : GLenum;
      internalformat : GLenum;
      x : GLint;
      y : GLint;
      width : GLsizei;
      height : GLsizei);  -- /usr/include/GL/gl.h:1691
   pragma Import (C, glCopyConvolutionFilter2D, "glCopyConvolutionFilter2D");

   procedure glGetConvolutionFilter
     (target : GLenum;
      format : GLenum;
      c_type : GLenum;
      image : System.Address);  -- /usr/include/GL/gl.h:1695
   pragma Import (C, glGetConvolutionFilter, "glGetConvolutionFilter");

   procedure glGetConvolutionParameterfv
     (target : GLenum;
      pname : GLenum;
      params : access GLfloat);  -- /usr/include/GL/gl.h:1698
   pragma Import (C, glGetConvolutionParameterfv, "glGetConvolutionParameterfv");

   procedure glGetConvolutionParameteriv
     (target : GLenum;
      pname : GLenum;
      params : access GLint);  -- /usr/include/GL/gl.h:1701
   pragma Import (C, glGetConvolutionParameteriv, "glGetConvolutionParameteriv");

   procedure glSeparableFilter2D
     (target : GLenum;
      internalformat : GLenum;
      width : GLsizei;
      height : GLsizei;
      format : GLenum;
      c_type : GLenum;
      row : System.Address;
      column : System.Address);  -- /usr/include/GL/gl.h:1704
   pragma Import (C, glSeparableFilter2D, "glSeparableFilter2D");

   procedure glGetSeparableFilter
     (target : GLenum;
      format : GLenum;
      c_type : GLenum;
      row : System.Address;
      column : System.Address;
      span : System.Address);  -- /usr/include/GL/gl.h:1708
   pragma Import (C, glGetSeparableFilter, "glGetSeparableFilter");

  -- * OpenGL 1.3
  --  

  -- multitexture  
  -- texture_cube_map  
  -- texture_compression  
  -- multisample  
  -- transpose_matrix  
  -- texture_env_combine  
  -- texture_env_dot3  
  -- texture_border_clamp  
   procedure glActiveTexture (texture : GLenum);  -- /usr/include/GL/gl.h:1823
   pragma Import (C, glActiveTexture, "glActiveTexture");

   procedure glClientActiveTexture (texture : GLenum);  -- /usr/include/GL/gl.h:1825
   pragma Import (C, glClientActiveTexture, "glClientActiveTexture");

   procedure glCompressedTexImage1D
     (target : GLenum;
      level : GLint;
      internalformat : GLenum;
      width : GLsizei;
      border : GLint;
      imageSize : GLsizei;
      data : System.Address);  -- /usr/include/GL/gl.h:1827
   pragma Import (C, glCompressedTexImage1D, "glCompressedTexImage1D");

   procedure glCompressedTexImage2D
     (target : GLenum;
      level : GLint;
      internalformat : GLenum;
      width : GLsizei;
      height : GLsizei;
      border : GLint;
      imageSize : GLsizei;
      data : System.Address);  -- /usr/include/GL/gl.h:1829
   pragma Import (C, glCompressedTexImage2D, "glCompressedTexImage2D");

   procedure glCompressedTexImage3D
     (target : GLenum;
      level : GLint;
      internalformat : GLenum;
      width : GLsizei;
      height : GLsizei;
      depth : GLsizei;
      border : GLint;
      imageSize : GLsizei;
      data : System.Address);  -- /usr/include/GL/gl.h:1831
   pragma Import (C, glCompressedTexImage3D, "glCompressedTexImage3D");

   procedure glCompressedTexSubImage1D
     (target : GLenum;
      level : GLint;
      xoffset : GLint;
      width : GLsizei;
      format : GLenum;
      imageSize : GLsizei;
      data : System.Address);  -- /usr/include/GL/gl.h:1833
   pragma Import (C, glCompressedTexSubImage1D, "glCompressedTexSubImage1D");

   procedure glCompressedTexSubImage2D
     (target : GLenum;
      level : GLint;
      xoffset : GLint;
      yoffset : GLint;
      width : GLsizei;
      height : GLsizei;
      format : GLenum;
      imageSize : GLsizei;
      data : System.Address);  -- /usr/include/GL/gl.h:1835
   pragma Import (C, glCompressedTexSubImage2D, "glCompressedTexSubImage2D");

   procedure glCompressedTexSubImage3D
     (target : GLenum;
      level : GLint;
      xoffset : GLint;
      yoffset : GLint;
      zoffset : GLint;
      width : GLsizei;
      height : GLsizei;
      depth : GLsizei;
      format : GLenum;
      imageSize : GLsizei;
      data : System.Address);  -- /usr/include/GL/gl.h:1837
   pragma Import (C, glCompressedTexSubImage3D, "glCompressedTexSubImage3D");

   procedure glGetCompressedTexImage
     (target : GLenum;
      lod : GLint;
      img : System.Address);  -- /usr/include/GL/gl.h:1839
   pragma Import (C, glGetCompressedTexImage, "glGetCompressedTexImage");

   procedure glMultiTexCoord1d (target : GLenum; s : GLdouble);  -- /usr/include/GL/gl.h:1841
   pragma Import (C, glMultiTexCoord1d, "glMultiTexCoord1d");

   procedure glMultiTexCoord1dv (target : GLenum; v : access GLdouble);  -- /usr/include/GL/gl.h:1843
   pragma Import (C, glMultiTexCoord1dv, "glMultiTexCoord1dv");

   procedure glMultiTexCoord1f (target : GLenum; s : GLfloat);  -- /usr/include/GL/gl.h:1845
   pragma Import (C, glMultiTexCoord1f, "glMultiTexCoord1f");

   procedure glMultiTexCoord1fv (target : GLenum; v : access GLfloat);  -- /usr/include/GL/gl.h:1847
   pragma Import (C, glMultiTexCoord1fv, "glMultiTexCoord1fv");

   procedure glMultiTexCoord1i (target : GLenum; s : GLint);  -- /usr/include/GL/gl.h:1849
   pragma Import (C, glMultiTexCoord1i, "glMultiTexCoord1i");

   procedure glMultiTexCoord1iv (target : GLenum; v : access GLint);  -- /usr/include/GL/gl.h:1851
   pragma Import (C, glMultiTexCoord1iv, "glMultiTexCoord1iv");

   procedure glMultiTexCoord1s (target : GLenum; s : GLshort);  -- /usr/include/GL/gl.h:1853
   pragma Import (C, glMultiTexCoord1s, "glMultiTexCoord1s");

   procedure glMultiTexCoord1sv (target : GLenum; v : access GLshort);  -- /usr/include/GL/gl.h:1855
   pragma Import (C, glMultiTexCoord1sv, "glMultiTexCoord1sv");

   procedure glMultiTexCoord2d
     (target : GLenum;
      s : GLdouble;
      t : GLdouble);  -- /usr/include/GL/gl.h:1857
   pragma Import (C, glMultiTexCoord2d, "glMultiTexCoord2d");

   procedure glMultiTexCoord2dv (target : GLenum; v : access GLdouble);  -- /usr/include/GL/gl.h:1859
   pragma Import (C, glMultiTexCoord2dv, "glMultiTexCoord2dv");

   procedure glMultiTexCoord2f
     (target : GLenum;
      s : GLfloat;
      t : GLfloat);  -- /usr/include/GL/gl.h:1861
   pragma Import (C, glMultiTexCoord2f, "glMultiTexCoord2f");

   procedure glMultiTexCoord2fv (target : GLenum; v : access GLfloat);  -- /usr/include/GL/gl.h:1863
   pragma Import (C, glMultiTexCoord2fv, "glMultiTexCoord2fv");

   procedure glMultiTexCoord2i
     (target : GLenum;
      s : GLint;
      t : GLint);  -- /usr/include/GL/gl.h:1865
   pragma Import (C, glMultiTexCoord2i, "glMultiTexCoord2i");

   procedure glMultiTexCoord2iv (target : GLenum; v : access GLint);  -- /usr/include/GL/gl.h:1867
   pragma Import (C, glMultiTexCoord2iv, "glMultiTexCoord2iv");

   procedure glMultiTexCoord2s
     (target : GLenum;
      s : GLshort;
      t : GLshort);  -- /usr/include/GL/gl.h:1869
   pragma Import (C, glMultiTexCoord2s, "glMultiTexCoord2s");

   procedure glMultiTexCoord2sv (target : GLenum; v : access GLshort);  -- /usr/include/GL/gl.h:1871
   pragma Import (C, glMultiTexCoord2sv, "glMultiTexCoord2sv");

   procedure glMultiTexCoord3d
     (target : GLenum;
      s : GLdouble;
      t : GLdouble;
      r : GLdouble);  -- /usr/include/GL/gl.h:1873
   pragma Import (C, glMultiTexCoord3d, "glMultiTexCoord3d");

   procedure glMultiTexCoord3dv (target : GLenum; v : access GLdouble);  -- /usr/include/GL/gl.h:1875
   pragma Import (C, glMultiTexCoord3dv, "glMultiTexCoord3dv");

   procedure glMultiTexCoord3f
     (target : GLenum;
      s : GLfloat;
      t : GLfloat;
      r : GLfloat);  -- /usr/include/GL/gl.h:1877
   pragma Import (C, glMultiTexCoord3f, "glMultiTexCoord3f");

   procedure glMultiTexCoord3fv (target : GLenum; v : access GLfloat);  -- /usr/include/GL/gl.h:1879
   pragma Import (C, glMultiTexCoord3fv, "glMultiTexCoord3fv");

   procedure glMultiTexCoord3i
     (target : GLenum;
      s : GLint;
      t : GLint;
      r : GLint);  -- /usr/include/GL/gl.h:1881
   pragma Import (C, glMultiTexCoord3i, "glMultiTexCoord3i");

   procedure glMultiTexCoord3iv (target : GLenum; v : access GLint);  -- /usr/include/GL/gl.h:1883
   pragma Import (C, glMultiTexCoord3iv, "glMultiTexCoord3iv");

   procedure glMultiTexCoord3s
     (target : GLenum;
      s : GLshort;
      t : GLshort;
      r : GLshort);  -- /usr/include/GL/gl.h:1885
   pragma Import (C, glMultiTexCoord3s, "glMultiTexCoord3s");

   procedure glMultiTexCoord3sv (target : GLenum; v : access GLshort);  -- /usr/include/GL/gl.h:1887
   pragma Import (C, glMultiTexCoord3sv, "glMultiTexCoord3sv");

   procedure glMultiTexCoord4d
     (target : GLenum;
      s : GLdouble;
      t : GLdouble;
      r : GLdouble;
      q : GLdouble);  -- /usr/include/GL/gl.h:1889
   pragma Import (C, glMultiTexCoord4d, "glMultiTexCoord4d");

   procedure glMultiTexCoord4dv (target : GLenum; v : access GLdouble);  -- /usr/include/GL/gl.h:1891
   pragma Import (C, glMultiTexCoord4dv, "glMultiTexCoord4dv");

   procedure glMultiTexCoord4f
     (target : GLenum;
      s : GLfloat;
      t : GLfloat;
      r : GLfloat;
      q : GLfloat);  -- /usr/include/GL/gl.h:1893
   pragma Import (C, glMultiTexCoord4f, "glMultiTexCoord4f");

   procedure glMultiTexCoord4fv (target : GLenum; v : access GLfloat);  -- /usr/include/GL/gl.h:1895
   pragma Import (C, glMultiTexCoord4fv, "glMultiTexCoord4fv");

   procedure glMultiTexCoord4i
     (target : GLenum;
      s : GLint;
      t : GLint;
      r : GLint;
      q : GLint);  -- /usr/include/GL/gl.h:1897
   pragma Import (C, glMultiTexCoord4i, "glMultiTexCoord4i");

   procedure glMultiTexCoord4iv (target : GLenum; v : access GLint);  -- /usr/include/GL/gl.h:1899
   pragma Import (C, glMultiTexCoord4iv, "glMultiTexCoord4iv");

   procedure glMultiTexCoord4s
     (target : GLenum;
      s : GLshort;
      t : GLshort;
      r : GLshort;
      q : GLshort);  -- /usr/include/GL/gl.h:1901
   pragma Import (C, glMultiTexCoord4s, "glMultiTexCoord4s");

   procedure glMultiTexCoord4sv (target : GLenum; v : access GLshort);  -- /usr/include/GL/gl.h:1903
   pragma Import (C, glMultiTexCoord4sv, "glMultiTexCoord4sv");

   procedure glLoadTransposeMatrixd (m : access GLdouble);  -- /usr/include/GL/gl.h:1906
   pragma Import (C, glLoadTransposeMatrixd, "glLoadTransposeMatrixd");

   procedure glLoadTransposeMatrixf (m : access GLfloat);  -- /usr/include/GL/gl.h:1908
   pragma Import (C, glLoadTransposeMatrixf, "glLoadTransposeMatrixf");

   procedure glMultTransposeMatrixd (m : access GLdouble);  -- /usr/include/GL/gl.h:1910
   pragma Import (C, glMultTransposeMatrixd, "glMultTransposeMatrixd");

   procedure glMultTransposeMatrixf (m : access GLfloat);  -- /usr/include/GL/gl.h:1912
   pragma Import (C, glMultTransposeMatrixf, "glMultTransposeMatrixf");

   procedure glSampleCoverage (value : GLclampf; invert : GLboolean);  -- /usr/include/GL/gl.h:1914
   pragma Import (C, glSampleCoverage, "glSampleCoverage");

   type PFNGLACTIVETEXTUREPROC is access procedure (arg1 : GLenum);
   pragma Convention (C, PFNGLACTIVETEXTUREPROC);  -- /usr/include/GL/gl.h:1917

   type PFNGLSAMPLECOVERAGEPROC is access procedure (arg1 : GLclampf; arg2 : GLboolean);
   pragma Convention (C, PFNGLSAMPLECOVERAGEPROC);  -- /usr/include/GL/gl.h:1918

   type PFNGLCOMPRESSEDTEXIMAGE3DPROC is access procedure
        (arg1 : GLenum;
         arg2 : GLint;
         arg3 : GLenum;
         arg4 : GLsizei;
         arg5 : GLsizei;
         arg6 : GLsizei;
         arg7 : GLint;
         arg8 : GLsizei;
         arg9 : System.Address);
   pragma Convention (C, PFNGLCOMPRESSEDTEXIMAGE3DPROC);  -- /usr/include/GL/gl.h:1919

   type PFNGLCOMPRESSEDTEXIMAGE2DPROC is access procedure
        (arg1 : GLenum;
         arg2 : GLint;
         arg3 : GLenum;
         arg4 : GLsizei;
         arg5 : GLsizei;
         arg6 : GLint;
         arg7 : GLsizei;
         arg8 : System.Address);
   pragma Convention (C, PFNGLCOMPRESSEDTEXIMAGE2DPROC);  -- /usr/include/GL/gl.h:1920

   type PFNGLCOMPRESSEDTEXIMAGE1DPROC is access procedure
        (arg1 : GLenum;
         arg2 : GLint;
         arg3 : GLenum;
         arg4 : GLsizei;
         arg5 : GLint;
         arg6 : GLsizei;
         arg7 : System.Address);
   pragma Convention (C, PFNGLCOMPRESSEDTEXIMAGE1DPROC);  -- /usr/include/GL/gl.h:1921

   type PFNGLCOMPRESSEDTEXSUBIMAGE3DPROC is access procedure
        (arg1 : GLenum;
         arg2 : GLint;
         arg3 : GLint;
         arg4 : GLint;
         arg5 : GLint;
         arg6 : GLsizei;
         arg7 : GLsizei;
         arg8 : GLsizei;
         arg9 : GLenum;
         arg10 : GLsizei;
         arg11 : System.Address);
   pragma Convention (C, PFNGLCOMPRESSEDTEXSUBIMAGE3DPROC);  -- /usr/include/GL/gl.h:1922

   type PFNGLCOMPRESSEDTEXSUBIMAGE2DPROC is access procedure
        (arg1 : GLenum;
         arg2 : GLint;
         arg3 : GLint;
         arg4 : GLint;
         arg5 : GLsizei;
         arg6 : GLsizei;
         arg7 : GLenum;
         arg8 : GLsizei;
         arg9 : System.Address);
   pragma Convention (C, PFNGLCOMPRESSEDTEXSUBIMAGE2DPROC);  -- /usr/include/GL/gl.h:1923

   type PFNGLCOMPRESSEDTEXSUBIMAGE1DPROC is access procedure
        (arg1 : GLenum;
         arg2 : GLint;
         arg3 : GLint;
         arg4 : GLsizei;
         arg5 : GLenum;
         arg6 : GLsizei;
         arg7 : System.Address);
   pragma Convention (C, PFNGLCOMPRESSEDTEXSUBIMAGE1DPROC);  -- /usr/include/GL/gl.h:1924

   type PFNGLGETCOMPRESSEDTEXIMAGEPROC is access procedure
        (arg1 : GLenum;
         arg2 : GLint;
         arg3 : System.Address);
   pragma Convention (C, PFNGLGETCOMPRESSEDTEXIMAGEPROC);  -- /usr/include/GL/gl.h:1925

  -- * GL_ARB_multitexture (ARB extension 1 and OpenGL 1.2.1)
  --  

   procedure glActiveTextureARB (texture : GLenum);  -- /usr/include/GL/gl.h:1971
   pragma Import (C, glActiveTextureARB, "glActiveTextureARB");

   procedure glClientActiveTextureARB (texture : GLenum);  -- /usr/include/GL/gl.h:1972
   pragma Import (C, glClientActiveTextureARB, "glClientActiveTextureARB");

   procedure glMultiTexCoord1dARB (target : GLenum; s : GLdouble);  -- /usr/include/GL/gl.h:1973
   pragma Import (C, glMultiTexCoord1dARB, "glMultiTexCoord1dARB");

   procedure glMultiTexCoord1dvARB (target : GLenum; v : access GLdouble);  -- /usr/include/GL/gl.h:1974
   pragma Import (C, glMultiTexCoord1dvARB, "glMultiTexCoord1dvARB");

   procedure glMultiTexCoord1fARB (target : GLenum; s : GLfloat);  -- /usr/include/GL/gl.h:1975
   pragma Import (C, glMultiTexCoord1fARB, "glMultiTexCoord1fARB");

   procedure glMultiTexCoord1fvARB (target : GLenum; v : access GLfloat);  -- /usr/include/GL/gl.h:1976
   pragma Import (C, glMultiTexCoord1fvARB, "glMultiTexCoord1fvARB");

   procedure glMultiTexCoord1iARB (target : GLenum; s : GLint);  -- /usr/include/GL/gl.h:1977
   pragma Import (C, glMultiTexCoord1iARB, "glMultiTexCoord1iARB");

   procedure glMultiTexCoord1ivARB (target : GLenum; v : access GLint);  -- /usr/include/GL/gl.h:1978
   pragma Import (C, glMultiTexCoord1ivARB, "glMultiTexCoord1ivARB");

   procedure glMultiTexCoord1sARB (target : GLenum; s : GLshort);  -- /usr/include/GL/gl.h:1979
   pragma Import (C, glMultiTexCoord1sARB, "glMultiTexCoord1sARB");

   procedure glMultiTexCoord1svARB (target : GLenum; v : access GLshort);  -- /usr/include/GL/gl.h:1980
   pragma Import (C, glMultiTexCoord1svARB, "glMultiTexCoord1svARB");

   procedure glMultiTexCoord2dARB
     (target : GLenum;
      s : GLdouble;
      t : GLdouble);  -- /usr/include/GL/gl.h:1981
   pragma Import (C, glMultiTexCoord2dARB, "glMultiTexCoord2dARB");

   procedure glMultiTexCoord2dvARB (target : GLenum; v : access GLdouble);  -- /usr/include/GL/gl.h:1982
   pragma Import (C, glMultiTexCoord2dvARB, "glMultiTexCoord2dvARB");

   procedure glMultiTexCoord2fARB
     (target : GLenum;
      s : GLfloat;
      t : GLfloat);  -- /usr/include/GL/gl.h:1983
   pragma Import (C, glMultiTexCoord2fARB, "glMultiTexCoord2fARB");

   procedure glMultiTexCoord2fvARB (target : GLenum; v : access GLfloat);  -- /usr/include/GL/gl.h:1984
   pragma Import (C, glMultiTexCoord2fvARB, "glMultiTexCoord2fvARB");

   procedure glMultiTexCoord2iARB
     (target : GLenum;
      s : GLint;
      t : GLint);  -- /usr/include/GL/gl.h:1985
   pragma Import (C, glMultiTexCoord2iARB, "glMultiTexCoord2iARB");

   procedure glMultiTexCoord2ivARB (target : GLenum; v : access GLint);  -- /usr/include/GL/gl.h:1986
   pragma Import (C, glMultiTexCoord2ivARB, "glMultiTexCoord2ivARB");

   procedure glMultiTexCoord2sARB
     (target : GLenum;
      s : GLshort;
      t : GLshort);  -- /usr/include/GL/gl.h:1987
   pragma Import (C, glMultiTexCoord2sARB, "glMultiTexCoord2sARB");

   procedure glMultiTexCoord2svARB (target : GLenum; v : access GLshort);  -- /usr/include/GL/gl.h:1988
   pragma Import (C, glMultiTexCoord2svARB, "glMultiTexCoord2svARB");

   procedure glMultiTexCoord3dARB
     (target : GLenum;
      s : GLdouble;
      t : GLdouble;
      r : GLdouble);  -- /usr/include/GL/gl.h:1989
   pragma Import (C, glMultiTexCoord3dARB, "glMultiTexCoord3dARB");

   procedure glMultiTexCoord3dvARB (target : GLenum; v : access GLdouble);  -- /usr/include/GL/gl.h:1990
   pragma Import (C, glMultiTexCoord3dvARB, "glMultiTexCoord3dvARB");

   procedure glMultiTexCoord3fARB
     (target : GLenum;
      s : GLfloat;
      t : GLfloat;
      r : GLfloat);  -- /usr/include/GL/gl.h:1991
   pragma Import (C, glMultiTexCoord3fARB, "glMultiTexCoord3fARB");

   procedure glMultiTexCoord3fvARB (target : GLenum; v : access GLfloat);  -- /usr/include/GL/gl.h:1992
   pragma Import (C, glMultiTexCoord3fvARB, "glMultiTexCoord3fvARB");

   procedure glMultiTexCoord3iARB
     (target : GLenum;
      s : GLint;
      t : GLint;
      r : GLint);  -- /usr/include/GL/gl.h:1993
   pragma Import (C, glMultiTexCoord3iARB, "glMultiTexCoord3iARB");

   procedure glMultiTexCoord3ivARB (target : GLenum; v : access GLint);  -- /usr/include/GL/gl.h:1994
   pragma Import (C, glMultiTexCoord3ivARB, "glMultiTexCoord3ivARB");

   procedure glMultiTexCoord3sARB
     (target : GLenum;
      s : GLshort;
      t : GLshort;
      r : GLshort);  -- /usr/include/GL/gl.h:1995
   pragma Import (C, glMultiTexCoord3sARB, "glMultiTexCoord3sARB");

   procedure glMultiTexCoord3svARB (target : GLenum; v : access GLshort);  -- /usr/include/GL/gl.h:1996
   pragma Import (C, glMultiTexCoord3svARB, "glMultiTexCoord3svARB");

   procedure glMultiTexCoord4dARB
     (target : GLenum;
      s : GLdouble;
      t : GLdouble;
      r : GLdouble;
      q : GLdouble);  -- /usr/include/GL/gl.h:1997
   pragma Import (C, glMultiTexCoord4dARB, "glMultiTexCoord4dARB");

   procedure glMultiTexCoord4dvARB (target : GLenum; v : access GLdouble);  -- /usr/include/GL/gl.h:1998
   pragma Import (C, glMultiTexCoord4dvARB, "glMultiTexCoord4dvARB");

   procedure glMultiTexCoord4fARB
     (target : GLenum;
      s : GLfloat;
      t : GLfloat;
      r : GLfloat;
      q : GLfloat);  -- /usr/include/GL/gl.h:1999
   pragma Import (C, glMultiTexCoord4fARB, "glMultiTexCoord4fARB");

   procedure glMultiTexCoord4fvARB (target : GLenum; v : access GLfloat);  -- /usr/include/GL/gl.h:2000
   pragma Import (C, glMultiTexCoord4fvARB, "glMultiTexCoord4fvARB");

   procedure glMultiTexCoord4iARB
     (target : GLenum;
      s : GLint;
      t : GLint;
      r : GLint;
      q : GLint);  -- /usr/include/GL/gl.h:2001
   pragma Import (C, glMultiTexCoord4iARB, "glMultiTexCoord4iARB");

   procedure glMultiTexCoord4ivARB (target : GLenum; v : access GLint);  -- /usr/include/GL/gl.h:2002
   pragma Import (C, glMultiTexCoord4ivARB, "glMultiTexCoord4ivARB");

   procedure glMultiTexCoord4sARB
     (target : GLenum;
      s : GLshort;
      t : GLshort;
      r : GLshort;
      q : GLshort);  -- /usr/include/GL/gl.h:2003
   pragma Import (C, glMultiTexCoord4sARB, "glMultiTexCoord4sARB");

   procedure glMultiTexCoord4svARB (target : GLenum; v : access GLshort);  -- /usr/include/GL/gl.h:2004
   pragma Import (C, glMultiTexCoord4svARB, "glMultiTexCoord4svARB");

   type PFNGLACTIVETEXTUREARBPROC is access procedure (arg1 : GLenum);
   pragma Convention (C, PFNGLACTIVETEXTUREARBPROC);  -- /usr/include/GL/gl.h:2006

   type PFNGLCLIENTACTIVETEXTUREARBPROC is access procedure (arg1 : GLenum);
   pragma Convention (C, PFNGLCLIENTACTIVETEXTUREARBPROC);  -- /usr/include/GL/gl.h:2007

   type PFNGLMULTITEXCOORD1DARBPROC is access procedure (arg1 : GLenum; arg2 : GLdouble);
   pragma Convention (C, PFNGLMULTITEXCOORD1DARBPROC);  -- /usr/include/GL/gl.h:2008

   type PFNGLMULTITEXCOORD1DVARBPROC is access procedure (arg1 : GLenum; arg2 : access GLdouble);
   pragma Convention (C, PFNGLMULTITEXCOORD1DVARBPROC);  -- /usr/include/GL/gl.h:2009

   type PFNGLMULTITEXCOORD1FARBPROC is access procedure (arg1 : GLenum; arg2 : GLfloat);
   pragma Convention (C, PFNGLMULTITEXCOORD1FARBPROC);  -- /usr/include/GL/gl.h:2010

   type PFNGLMULTITEXCOORD1FVARBPROC is access procedure (arg1 : GLenum; arg2 : access GLfloat);
   pragma Convention (C, PFNGLMULTITEXCOORD1FVARBPROC);  -- /usr/include/GL/gl.h:2011

   type PFNGLMULTITEXCOORD1IARBPROC is access procedure (arg1 : GLenum; arg2 : GLint);
   pragma Convention (C, PFNGLMULTITEXCOORD1IARBPROC);  -- /usr/include/GL/gl.h:2012

   type PFNGLMULTITEXCOORD1IVARBPROC is access procedure (arg1 : GLenum; arg2 : access GLint);
   pragma Convention (C, PFNGLMULTITEXCOORD1IVARBPROC);  -- /usr/include/GL/gl.h:2013

   type PFNGLMULTITEXCOORD1SARBPROC is access procedure (arg1 : GLenum; arg2 : GLshort);
   pragma Convention (C, PFNGLMULTITEXCOORD1SARBPROC);  -- /usr/include/GL/gl.h:2014

   type PFNGLMULTITEXCOORD1SVARBPROC is access procedure (arg1 : GLenum; arg2 : access GLshort);
   pragma Convention (C, PFNGLMULTITEXCOORD1SVARBPROC);  -- /usr/include/GL/gl.h:2015

   type PFNGLMULTITEXCOORD2DARBPROC is access procedure
        (arg1 : GLenum;
         arg2 : GLdouble;
         arg3 : GLdouble);
   pragma Convention (C, PFNGLMULTITEXCOORD2DARBPROC);  -- /usr/include/GL/gl.h:2016

   type PFNGLMULTITEXCOORD2DVARBPROC is access procedure (arg1 : GLenum; arg2 : access GLdouble);
   pragma Convention (C, PFNGLMULTITEXCOORD2DVARBPROC);  -- /usr/include/GL/gl.h:2017

   type PFNGLMULTITEXCOORD2FARBPROC is access procedure
        (arg1 : GLenum;
         arg2 : GLfloat;
         arg3 : GLfloat);
   pragma Convention (C, PFNGLMULTITEXCOORD2FARBPROC);  -- /usr/include/GL/gl.h:2018

   type PFNGLMULTITEXCOORD2FVARBPROC is access procedure (arg1 : GLenum; arg2 : access GLfloat);
   pragma Convention (C, PFNGLMULTITEXCOORD2FVARBPROC);  -- /usr/include/GL/gl.h:2019

   type PFNGLMULTITEXCOORD2IARBPROC is access procedure
        (arg1 : GLenum;
         arg2 : GLint;
         arg3 : GLint);
   pragma Convention (C, PFNGLMULTITEXCOORD2IARBPROC);  -- /usr/include/GL/gl.h:2020

   type PFNGLMULTITEXCOORD2IVARBPROC is access procedure (arg1 : GLenum; arg2 : access GLint);
   pragma Convention (C, PFNGLMULTITEXCOORD2IVARBPROC);  -- /usr/include/GL/gl.h:2021

   type PFNGLMULTITEXCOORD2SARBPROC is access procedure
        (arg1 : GLenum;
         arg2 : GLshort;
         arg3 : GLshort);
   pragma Convention (C, PFNGLMULTITEXCOORD2SARBPROC);  -- /usr/include/GL/gl.h:2022

   type PFNGLMULTITEXCOORD2SVARBPROC is access procedure (arg1 : GLenum; arg2 : access GLshort);
   pragma Convention (C, PFNGLMULTITEXCOORD2SVARBPROC);  -- /usr/include/GL/gl.h:2023

   type PFNGLMULTITEXCOORD3DARBPROC is access procedure
        (arg1 : GLenum;
         arg2 : GLdouble;
         arg3 : GLdouble;
         arg4 : GLdouble);
   pragma Convention (C, PFNGLMULTITEXCOORD3DARBPROC);  -- /usr/include/GL/gl.h:2024

   type PFNGLMULTITEXCOORD3DVARBPROC is access procedure (arg1 : GLenum; arg2 : access GLdouble);
   pragma Convention (C, PFNGLMULTITEXCOORD3DVARBPROC);  -- /usr/include/GL/gl.h:2025

   type PFNGLMULTITEXCOORD3FARBPROC is access procedure
        (arg1 : GLenum;
         arg2 : GLfloat;
         arg3 : GLfloat;
         arg4 : GLfloat);
   pragma Convention (C, PFNGLMULTITEXCOORD3FARBPROC);  -- /usr/include/GL/gl.h:2026

   type PFNGLMULTITEXCOORD3FVARBPROC is access procedure (arg1 : GLenum; arg2 : access GLfloat);
   pragma Convention (C, PFNGLMULTITEXCOORD3FVARBPROC);  -- /usr/include/GL/gl.h:2027

   type PFNGLMULTITEXCOORD3IARBPROC is access procedure
        (arg1 : GLenum;
         arg2 : GLint;
         arg3 : GLint;
         arg4 : GLint);
   pragma Convention (C, PFNGLMULTITEXCOORD3IARBPROC);  -- /usr/include/GL/gl.h:2028

   type PFNGLMULTITEXCOORD3IVARBPROC is access procedure (arg1 : GLenum; arg2 : access GLint);
   pragma Convention (C, PFNGLMULTITEXCOORD3IVARBPROC);  -- /usr/include/GL/gl.h:2029

   type PFNGLMULTITEXCOORD3SARBPROC is access procedure
        (arg1 : GLenum;
         arg2 : GLshort;
         arg3 : GLshort;
         arg4 : GLshort);
   pragma Convention (C, PFNGLMULTITEXCOORD3SARBPROC);  -- /usr/include/GL/gl.h:2030

   type PFNGLMULTITEXCOORD3SVARBPROC is access procedure (arg1 : GLenum; arg2 : access GLshort);
   pragma Convention (C, PFNGLMULTITEXCOORD3SVARBPROC);  -- /usr/include/GL/gl.h:2031

   type PFNGLMULTITEXCOORD4DARBPROC is access procedure
        (arg1 : GLenum;
         arg2 : GLdouble;
         arg3 : GLdouble;
         arg4 : GLdouble;
         arg5 : GLdouble);
   pragma Convention (C, PFNGLMULTITEXCOORD4DARBPROC);  -- /usr/include/GL/gl.h:2032

   type PFNGLMULTITEXCOORD4DVARBPROC is access procedure (arg1 : GLenum; arg2 : access GLdouble);
   pragma Convention (C, PFNGLMULTITEXCOORD4DVARBPROC);  -- /usr/include/GL/gl.h:2033

   type PFNGLMULTITEXCOORD4FARBPROC is access procedure
        (arg1 : GLenum;
         arg2 : GLfloat;
         arg3 : GLfloat;
         arg4 : GLfloat;
         arg5 : GLfloat);
   pragma Convention (C, PFNGLMULTITEXCOORD4FARBPROC);  -- /usr/include/GL/gl.h:2034

   type PFNGLMULTITEXCOORD4FVARBPROC is access procedure (arg1 : GLenum; arg2 : access GLfloat);
   pragma Convention (C, PFNGLMULTITEXCOORD4FVARBPROC);  -- /usr/include/GL/gl.h:2035

   type PFNGLMULTITEXCOORD4IARBPROC is access procedure
        (arg1 : GLenum;
         arg2 : GLint;
         arg3 : GLint;
         arg4 : GLint;
         arg5 : GLint);
   pragma Convention (C, PFNGLMULTITEXCOORD4IARBPROC);  -- /usr/include/GL/gl.h:2036

   type PFNGLMULTITEXCOORD4IVARBPROC is access procedure (arg1 : GLenum; arg2 : access GLint);
   pragma Convention (C, PFNGLMULTITEXCOORD4IVARBPROC);  -- /usr/include/GL/gl.h:2037

   type PFNGLMULTITEXCOORD4SARBPROC is access procedure
        (arg1 : GLenum;
         arg2 : GLshort;
         arg3 : GLshort;
         arg4 : GLshort;
         arg5 : GLshort);
   pragma Convention (C, PFNGLMULTITEXCOORD4SARBPROC);  -- /usr/include/GL/gl.h:2038

   type PFNGLMULTITEXCOORD4SVARBPROC is access procedure (arg1 : GLenum; arg2 : access GLshort);
   pragma Convention (C, PFNGLMULTITEXCOORD4SVARBPROC);  -- /usr/include/GL/gl.h:2039

  -- * Define this token if you want "old-style" header file behaviour (extensions
  -- * defined in gl.h).  Otherwise, extensions will be included from glext.h.
  --  

  -- All extensions that used to be here are now found in glext.h  
  -- * ???. GL_MESA_packed_depth_stencil
  -- * XXX obsolete
  --  

   procedure glBlendEquationSeparateATI (modeRGB : GLenum; modeA : GLenum);  -- /usr/include/GL/gl.h:2082
   pragma Import (C, glBlendEquationSeparateATI, "glBlendEquationSeparateATI");

   type PFNGLBLENDEQUATIONSEPARATEATIPROC is access procedure (arg1 : GLenum; arg2 : GLenum);
   pragma Convention (C, PFNGLBLENDEQUATIONSEPARATEATIPROC);  -- /usr/include/GL/gl.h:2083

  -- GL_OES_EGL_image  
   type GLeglImageOES is new System.Address;  -- /usr/include/GL/gl.h:2090

   type PFNGLEGLIMAGETARGETTEXTURE2DOESPROC is access procedure (arg1 : GLenum; arg2 : GLeglImageOES);
   pragma Convention (C, PFNGLEGLIMAGETARGETTEXTURE2DOESPROC);  -- /usr/include/GL/gl.h:2099

   type PFNGLEGLIMAGETARGETRENDERBUFFERSTORAGEOESPROC is access procedure (arg1 : GLenum; arg2 : GLeglImageOES);
   pragma Convention (C, PFNGLEGLIMAGETARGETRENDERBUFFERSTORAGEOESPROC);  -- /usr/include/GL/gl.h:2100

  --*
  -- ** NOTE!!!!!  If you add new functions to this file, or update
  -- ** glext.h be sure to regenerate the gl_mangle.h file.  See comments
  -- ** in that file for details.
  -- * 

end GL_gl_h;

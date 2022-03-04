pragma Ada_2005;
pragma Style_Checks (Off);

with Interfaces.C; use Interfaces.C;
with System;
with stddef_h;
with Interfaces.C.Strings;
with library_types_h;
with driver_types_h;
with Interfaces.C.Extensions;

package cudnn_h is

   CUDNN_MAJOR : constant := 7;  --  /usr/local/cuda-8.0/include/cudnn.h:57
   CUDNN_MINOR : constant := 0;  --  /usr/local/cuda-8.0/include/cudnn.h:58
   CUDNN_PATCHLEVEL : constant := 3;  --  /usr/local/cuda-8.0/include/cudnn.h:59
   --  unsupported macro: CUDNN_VERSION (CUDNN_MAJOR * 1000 + CUDNN_MINOR * 100 + CUDNN_PATCHLEVEL)

   CUDNN_DIM_MAX : constant := 8;  --  /usr/local/cuda-8.0/include/cudnn.h:194

   CUDNN_LRN_MIN_N : constant := 1;  --  /usr/local/cuda-8.0/include/cudnn.h:1183
   CUDNN_LRN_MAX_N : constant := 16;  --  /usr/local/cuda-8.0/include/cudnn.h:1184
   CUDNN_LRN_MIN_K : constant := 1.0e-5;  --  /usr/local/cuda-8.0/include/cudnn.h:1185
   CUDNN_LRN_MIN_BETA : constant := 0.01;  --  /usr/local/cuda-8.0/include/cudnn.h:1186

   CUDNN_BN_MIN_EPSILON : constant := 1.0e-5;  --  /usr/local/cuda-8.0/include/cudnn.h:1300

  -- * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
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

  --   cudnn : Neural Networks Library
  --  

   --  skipped empty struct cudnnContext

   type cudnnHandle_t is new System.Address;  -- /usr/local/cuda-8.0/include/cudnn.h:79

   function cudnnGetVersion return stddef_h.size_t;  -- /usr/local/cuda-8.0/include/cudnn.h:81
   pragma Import (C, cudnnGetVersion, "cudnnGetVersion");

  -- Returns CUDA Runtime version statically linked against cudnn  
   function cudnnGetCudartVersion return stddef_h.size_t;  -- /usr/local/cuda-8.0/include/cudnn.h:84
   pragma Import (C, cudnnGetCudartVersion, "cudnnGetCudartVersion");

  -- * CUDNN return codes
  --  

   type cudnnStatus_t is 
     (CUDNN_STATUS_SUCCESS,
      CUDNN_STATUS_NOT_INITIALIZED,
      CUDNN_STATUS_ALLOC_FAILED,
      CUDNN_STATUS_BAD_PARAM,
      CUDNN_STATUS_INTERNAL_ERROR,
      CUDNN_STATUS_INVALID_VALUE,
      CUDNN_STATUS_ARCH_MISMATCH,
      CUDNN_STATUS_MAPPING_ERROR,
      CUDNN_STATUS_EXECUTION_FAILED,
      CUDNN_STATUS_NOT_SUPPORTED,
      CUDNN_STATUS_LICENSE_ERROR,
      CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING,
      CUDNN_STATUS_RUNTIME_IN_PROGRESS,
      CUDNN_STATUS_RUNTIME_FP_OVERFLOW);
   pragma Convention (C, cudnnStatus_t);  -- /usr/local/cuda-8.0/include/cudnn.h:105

  -- human-readable error messages  
   function cudnnGetErrorString (status : cudnnStatus_t) return Interfaces.C.Strings.chars_ptr;  -- /usr/local/cuda-8.0/include/cudnn.h:108
   pragma Import (C, cudnnGetErrorString, "cudnnGetErrorString");

  -- Forward definition in this version only  
   --  skipped empty struct cudnnRuntimeTag_t

   type cudnnErrQueryMode_t is 
     (CUDNN_ERRQUERY_RAWCODE,
      CUDNN_ERRQUERY_NONBLOCKING,
      CUDNN_ERRQUERY_BLOCKING);
   pragma Convention (C, cudnnErrQueryMode_t);  -- /usr/local/cuda-8.0/include/cudnn.h:118

   function cudnnQueryRuntimeError
     (handle : cudnnHandle_t;
      rstatus : access cudnnStatus_t;
      mode : cudnnErrQueryMode_t;
      tag : System.Address) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:120
   pragma Import (C, cudnnQueryRuntimeError, "cudnnQueryRuntimeError");

   function cudnnGetProperty (c_type : library_types_h.libraryPropertyType; value : access int) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:137
   pragma Import (C, cudnnGetProperty, "cudnnGetProperty");

   function cudnnCreate (handle : System.Address) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:139
   pragma Import (C, cudnnCreate, "cudnnCreate");

   function cudnnDestroy (handle : cudnnHandle_t) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:140
   pragma Import (C, cudnnDestroy, "cudnnDestroy");

   function cudnnSetStream (handle : cudnnHandle_t; streamId : driver_types_h.cudaStream_t) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:141
   pragma Import (C, cudnnSetStream, "cudnnSetStream");

   function cudnnGetStream (handle : cudnnHandle_t; streamId : System.Address) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:142
   pragma Import (C, cudnnGetStream, "cudnnGetStream");

  -- Data structures to represent Image/Filter and the Neural Network Layer  
   --  skipped empty struct cudnnTensorStruct

   type cudnnTensorDescriptor_t is new System.Address;  -- /usr/local/cuda-8.0/include/cudnn.h:145

   --  skipped empty struct cudnnConvolutionStruct

   type cudnnConvolutionDescriptor_t is new System.Address;  -- /usr/local/cuda-8.0/include/cudnn.h:146

   --  skipped empty struct cudnnPoolingStruct

   type cudnnPoolingDescriptor_t is new System.Address;  -- /usr/local/cuda-8.0/include/cudnn.h:147

   --  skipped empty struct cudnnFilterStruct

   type cudnnFilterDescriptor_t is new System.Address;  -- /usr/local/cuda-8.0/include/cudnn.h:148

   --  skipped empty struct cudnnLRNStruct

   type cudnnLRNDescriptor_t is new System.Address;  -- /usr/local/cuda-8.0/include/cudnn.h:149

   --  skipped empty struct cudnnActivationStruct

   type cudnnActivationDescriptor_t is new System.Address;  -- /usr/local/cuda-8.0/include/cudnn.h:150

   --  skipped empty struct cudnnSpatialTransformerStruct

   type cudnnSpatialTransformerDescriptor_t is new System.Address;  -- /usr/local/cuda-8.0/include/cudnn.h:151

   --  skipped empty struct cudnnOpTensorStruct

   type cudnnOpTensorDescriptor_t is new System.Address;  -- /usr/local/cuda-8.0/include/cudnn.h:152

   --  skipped empty struct cudnnReduceTensorStruct

   type cudnnReduceTensorDescriptor_t is new System.Address;  -- /usr/local/cuda-8.0/include/cudnn.h:153

   --  skipped empty struct cudnnCTCLossStruct

   type cudnnCTCLossDescriptor_t is new System.Address;  -- /usr/local/cuda-8.0/include/cudnn.h:154

  --* CUDNN data type
  -- 

   type cudnnDataType_t is 
     (CUDNN_DATA_FLOAT,
      CUDNN_DATA_DOUBLE,
      CUDNN_DATA_HALF,
      CUDNN_DATA_INT8,
      CUDNN_DATA_INT32,
      CUDNN_DATA_INT8x4);
   pragma Convention (C, cudnnDataType_t);  -- /usr/local/cuda-8.0/include/cudnn.h:166

  --* CUDNN math type
  -- 

   type cudnnMathType_t is 
     (CUDNN_DEFAULT_MATH,
      CUDNN_TENSOR_OP_MATH);
   pragma Convention (C, cudnnMathType_t);  -- /usr/local/cuda-8.0/include/cudnn.h:174

  -- * CUDNN propagate Nan
  --  

   type cudnnNanPropagation_t is 
     (CUDNN_NOT_PROPAGATE_NAN,
      CUDNN_PROPAGATE_NAN);
   pragma Convention (C, cudnnNanPropagation_t);  -- /usr/local/cuda-8.0/include/cudnn.h:182

  -- 
  -- * CUDNN Determinism
  --  

   type cudnnDeterminism_t is 
     (CUDNN_NON_DETERMINISTIC,
      CUDNN_DETERMINISTIC);
   pragma Convention (C, cudnnDeterminism_t);  -- /usr/local/cuda-8.0/include/cudnn.h:191

  -- Maximum supported number of tensor dimensions  
  -- Create an instance of a generic Tensor descriptor  
   function cudnnCreateTensorDescriptor (tensorDesc : System.Address) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:197
   pragma Import (C, cudnnCreateTensorDescriptor, "cudnnCreateTensorDescriptor");

  -- row major (wStride = 1, hStride = w)  
  -- feature maps interleaved ( cStride = 1 ) 
  -- each image point is vector of element of C : the length of the vector is carried by the data type 
   type cudnnTensorFormat_t is 
     (CUDNN_TENSOR_NCHW,
      CUDNN_TENSOR_NHWC,
      CUDNN_TENSOR_NCHW_VECT_C);
   pragma Convention (C, cudnnTensorFormat_t);  -- /usr/local/cuda-8.0/include/cudnn.h:205

   function cudnnSetTensor4dDescriptor
     (tensorDesc : cudnnTensorDescriptor_t;
      format : cudnnTensorFormat_t;
      dataType : cudnnDataType_t;
      n : int;
      c : int;
      h : int;
      w : int) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:207
   pragma Import (C, cudnnSetTensor4dDescriptor, "cudnnSetTensor4dDescriptor");

  -- image data type  
  -- number of inputs (batch size)  
  -- number of input feature maps  
  -- height of input section  
  -- width of input section  
   function cudnnSetTensor4dDescriptorEx
     (tensorDesc : cudnnTensorDescriptor_t;
      dataType : cudnnDataType_t;
      n : int;
      c : int;
      h : int;
      w : int;
      nStride : int;
      cStride : int;
      hStride : int;
      wStride : int) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:216
   pragma Import (C, cudnnSetTensor4dDescriptorEx, "cudnnSetTensor4dDescriptorEx");

  -- image data type  
  -- number of inputs (batch size)  
  -- number of input feature maps  
  -- height of input section  
  -- width of input section  
   function cudnnGetTensor4dDescriptor
     (tensorDesc : cudnnTensorDescriptor_t;
      dataType : access cudnnDataType_t;
      n : access int;
      c : access int;
      h : access int;
      w : access int;
      nStride : access int;
      cStride : access int;
      hStride : access int;
      wStride : access int) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:228
   pragma Import (C, cudnnGetTensor4dDescriptor, "cudnnGetTensor4dDescriptor");

  -- image data type  
  -- number of inputs (batch size)  
  -- number of input feature maps   
  -- height of input section  
  -- width of input section  
   function cudnnSetTensorNdDescriptor
     (tensorDesc : cudnnTensorDescriptor_t;
      dataType : cudnnDataType_t;
      nbDims : int;
      dimA : access int;
      strideA : access int) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:240
   pragma Import (C, cudnnSetTensorNdDescriptor, "cudnnSetTensorNdDescriptor");

   function cudnnSetTensorNdDescriptorEx
     (tensorDesc : cudnnTensorDescriptor_t;
      format : cudnnTensorFormat_t;
      dataType : cudnnDataType_t;
      nbDims : int;
      dimA : access int) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:247
   pragma Import (C, cudnnSetTensorNdDescriptorEx, "cudnnSetTensorNdDescriptorEx");

   function cudnnGetTensorNdDescriptor
     (tensorDesc : cudnnTensorDescriptor_t;
      nbDimsRequested : int;
      dataType : access cudnnDataType_t;
      nbDims : access int;
      dimA : access int;
      strideA : access int) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:254
   pragma Import (C, cudnnGetTensorNdDescriptor, "cudnnGetTensorNdDescriptor");

   function cudnnGetTensorSizeInBytes (tensorDesc : cudnnTensorDescriptor_t; size : access stddef_h.size_t) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:262
   pragma Import (C, cudnnGetTensorSizeInBytes, "cudnnGetTensorSizeInBytes");

  -- PixelOffset( n, c, h, w ) = n *input_stride + c * feature_stride + h * h_stride + w * w_stride
  --   1)Example of all images in row major order one batch of features after the other (with an optional padding on row)
  --   input_stride :  c x h x h_stride
  --   feature_stride : h x h_stride
  --   h_stride  :  >= w  ( h_stride = w if no padding)
  --   w_stride  : 1
  --   2)Example of all images in row major with features maps interleaved
  --   input_stride :  c x h x h_stride
  --   feature_stride : 1
  --   h_stride  :  w x c
  --   w_stride  : c
  --   3)Example of all images in column major order one batch of features after the other (with optional padding on column)
  --   input_stride :  c x w x w_stride
  --   feature_stride : w x w_stride
  --   h_stride  :  1
  --   w_stride  :  >= h
  -- 

  -- Destroy an instance of Tensor4d descriptor  
   function cudnnDestroyTensorDescriptor (tensorDesc : cudnnTensorDescriptor_t) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:290
   pragma Import (C, cudnnDestroyTensorDescriptor, "cudnnDestroyTensorDescriptor");

  -- Tensor layout conversion helper (y = alpha * x + beta * y)  
   function cudnnTransformTensor
     (handle : cudnnHandle_t;
      alpha : System.Address;
      xDesc : cudnnTensorDescriptor_t;
      x : System.Address;
      beta : System.Address;
      yDesc : cudnnTensorDescriptor_t;
      y : System.Address) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:295
   pragma Import (C, cudnnTransformTensor, "cudnnTransformTensor");

  -- Tensor Bias addition : C = alpha * A + beta * C   
   function cudnnAddTensor
     (handle : cudnnHandle_t;
      alpha : System.Address;
      aDesc : cudnnTensorDescriptor_t;
      A : System.Address;
      beta : System.Address;
      cDesc : cudnnTensorDescriptor_t;
      C : System.Address) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:306
   pragma Import (C, cudnnAddTensor, "cudnnAddTensor");

  --* CUDNN OpTensor op type
  -- 

   type cudnnOpTensorOp_t is 
     (CUDNN_OP_TENSOR_ADD,
      CUDNN_OP_TENSOR_MUL,
      CUDNN_OP_TENSOR_MIN,
      CUDNN_OP_TENSOR_MAX,
      CUDNN_OP_TENSOR_SQRT,
      CUDNN_OP_TENSOR_NOT);
   pragma Convention (C, cudnnOpTensorOp_t);  -- /usr/local/cuda-8.0/include/cudnn.h:326

   function cudnnCreateOpTensorDescriptor (opTensorDesc : System.Address) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:328
   pragma Import (C, cudnnCreateOpTensorDescriptor, "cudnnCreateOpTensorDescriptor");

   function cudnnSetOpTensorDescriptor
     (opTensorDesc : cudnnOpTensorDescriptor_t;
      opTensorOp : cudnnOpTensorOp_t;
      opTensorCompType : cudnnDataType_t;
      opTensorNanOpt : cudnnNanPropagation_t) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:331
   pragma Import (C, cudnnSetOpTensorDescriptor, "cudnnSetOpTensorDescriptor");

   function cudnnGetOpTensorDescriptor
     (opTensorDesc : cudnnOpTensorDescriptor_t;
      opTensorOp : access cudnnOpTensorOp_t;
      opTensorCompType : access cudnnDataType_t;
      opTensorNanOpt : access cudnnNanPropagation_t) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:337
   pragma Import (C, cudnnGetOpTensorDescriptor, "cudnnGetOpTensorDescriptor");

   function cudnnDestroyOpTensorDescriptor (opTensorDesc : cudnnOpTensorDescriptor_t) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:343
   pragma Import (C, cudnnDestroyOpTensorDescriptor, "cudnnDestroyOpTensorDescriptor");

  -- Tensor operation : C = op( alpha1 * A, alpha2 * B ) + beta * C  
  -- B tensor is ignored for CUDNN_OP_TENSOR_SQRT, CUDNN_OP_TENSOR_NOT.  
   function cudnnOpTensor
     (handle : cudnnHandle_t;
      opTensorDesc : cudnnOpTensorDescriptor_t;
      alpha1 : System.Address;
      aDesc : cudnnTensorDescriptor_t;
      A : System.Address;
      alpha2 : System.Address;
      bDesc : cudnnTensorDescriptor_t;
      B : System.Address;
      beta : System.Address;
      cDesc : cudnnTensorDescriptor_t;
      C : System.Address) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:348
   pragma Import (C, cudnnOpTensor, "cudnnOpTensor");

  --* CUDNN ReduceTensor op type
  -- 

   type cudnnReduceTensorOp_t is 
     (CUDNN_REDUCE_TENSOR_ADD,
      CUDNN_REDUCE_TENSOR_MUL,
      CUDNN_REDUCE_TENSOR_MIN,
      CUDNN_REDUCE_TENSOR_MAX,
      CUDNN_REDUCE_TENSOR_AMAX,
      CUDNN_REDUCE_TENSOR_AVG,
      CUDNN_REDUCE_TENSOR_NORM1,
      CUDNN_REDUCE_TENSOR_NORM2,
      CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS);
   pragma Convention (C, cudnnReduceTensorOp_t);  -- /usr/local/cuda-8.0/include/cudnn.h:375

  --* CUDNN ReduceTensor indices type
  -- 

   type cudnnReduceTensorIndices_t is 
     (CUDNN_REDUCE_TENSOR_NO_INDICES,
      CUDNN_REDUCE_TENSOR_FLATTENED_INDICES);
   pragma Convention (C, cudnnReduceTensorIndices_t);  -- /usr/local/cuda-8.0/include/cudnn.h:384

  --* CUDNN tensor indices type size (all unsigned)
  --* Currently not supported, default is 32 bit unsigned.
  -- 

   type cudnnIndicesType_t is 
     (CUDNN_32BIT_INDICES,
      CUDNN_64BIT_INDICES,
      CUDNN_16BIT_INDICES,
      CUDNN_8BIT_INDICES);
   pragma Convention (C, cudnnIndicesType_t);  -- /usr/local/cuda-8.0/include/cudnn.h:396

   function cudnnCreateReduceTensorDescriptor (reduceTensorDesc : System.Address) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:398
   pragma Import (C, cudnnCreateReduceTensorDescriptor, "cudnnCreateReduceTensorDescriptor");

   function cudnnSetReduceTensorDescriptor
     (reduceTensorDesc : cudnnReduceTensorDescriptor_t;
      reduceTensorOp : cudnnReduceTensorOp_t;
      reduceTensorCompType : cudnnDataType_t;
      reduceTensorNanOpt : cudnnNanPropagation_t;
      reduceTensorIndices : cudnnReduceTensorIndices_t;
      reduceTensorIndicesType : cudnnIndicesType_t) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:401
   pragma Import (C, cudnnSetReduceTensorDescriptor, "cudnnSetReduceTensorDescriptor");

   function cudnnGetReduceTensorDescriptor
     (reduceTensorDesc : cudnnReduceTensorDescriptor_t;
      reduceTensorOp : access cudnnReduceTensorOp_t;
      reduceTensorCompType : access cudnnDataType_t;
      reduceTensorNanOpt : access cudnnNanPropagation_t;
      reduceTensorIndices : access cudnnReduceTensorIndices_t;
      reduceTensorIndicesType : access cudnnIndicesType_t) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:409
   pragma Import (C, cudnnGetReduceTensorDescriptor, "cudnnGetReduceTensorDescriptor");

   function cudnnDestroyReduceTensorDescriptor (reduceTensorDesc : cudnnReduceTensorDescriptor_t) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:417
   pragma Import (C, cudnnDestroyReduceTensorDescriptor, "cudnnDestroyReduceTensorDescriptor");

  -- Helper function to return the minimum size of the index space to be passed to the reduction given the input and output tensors  
   function cudnnGetReductionIndicesSize
     (handle : cudnnHandle_t;
      reduceTensorDesc : cudnnReduceTensorDescriptor_t;
      aDesc : cudnnTensorDescriptor_t;
      cDesc : cudnnTensorDescriptor_t;
      sizeInBytes : access stddef_h.size_t) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:421
   pragma Import (C, cudnnGetReductionIndicesSize, "cudnnGetReductionIndicesSize");

  -- Helper function to return the minimum size of the workspace to be passed to the reduction given the input and output tensors  
   function cudnnGetReductionWorkspaceSize
     (handle : cudnnHandle_t;
      reduceTensorDesc : cudnnReduceTensorDescriptor_t;
      aDesc : cudnnTensorDescriptor_t;
      cDesc : cudnnTensorDescriptor_t;
      sizeInBytes : access stddef_h.size_t) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:429
   pragma Import (C, cudnnGetReductionWorkspaceSize, "cudnnGetReductionWorkspaceSize");

  -- Tensor operation : C = reduce op( alpha * A ) + beta * C  
  -- The NaN propagation enum applies to only the min and max reduce ops; the other reduce ops propagate NaN as usual.  
  -- The indices space is ignored for reduce ops other than min or max.  
   function cudnnReduceTensor
     (handle : cudnnHandle_t;
      reduceTensorDesc : cudnnReduceTensorDescriptor_t;
      indices : System.Address;
      indicesSizeInBytes : stddef_h.size_t;
      workspace : System.Address;
      workspaceSizeInBytes : stddef_h.size_t;
      alpha : System.Address;
      aDesc : cudnnTensorDescriptor_t;
      A : System.Address;
      beta : System.Address;
      cDesc : cudnnTensorDescriptor_t;
      C : System.Address) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:439
   pragma Import (C, cudnnReduceTensor, "cudnnReduceTensor");

  -- Set all values of a tensor to a given value : y[i] = value[0]  
   function cudnnSetTensor
     (handle : cudnnHandle_t;
      yDesc : cudnnTensorDescriptor_t;
      y : System.Address;
      valuePtr : System.Address) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:454
   pragma Import (C, cudnnSetTensor, "cudnnSetTensor");

  -- Scale all values of a tensor by a given factor : y[i] = alpha * y[i]  
   function cudnnScaleTensor
     (handle : cudnnHandle_t;
      yDesc : cudnnTensorDescriptor_t;
      y : System.Address;
      alpha : System.Address) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:461
   pragma Import (C, cudnnScaleTensor, "cudnnScaleTensor");

  -- *  convolution mode
  --  

   type cudnnConvolutionMode_t is 
     (CUDNN_CONVOLUTION,
      CUDNN_CROSS_CORRELATION);
   pragma Convention (C, cudnnConvolutionMode_t);  -- /usr/local/cuda-8.0/include/cudnn.h:474

  -- Create an instance of FilterStruct  
   function cudnnCreateFilterDescriptor (filterDesc : System.Address) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:478
   pragma Import (C, cudnnCreateFilterDescriptor, "cudnnCreateFilterDescriptor");

   function cudnnSetFilter4dDescriptor
     (filterDesc : cudnnFilterDescriptor_t;
      dataType : cudnnDataType_t;
      format : cudnnTensorFormat_t;
      k : int;
      c : int;
      h : int;
      w : int) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:482
   pragma Import (C, cudnnSetFilter4dDescriptor, "cudnnSetFilter4dDescriptor");

  -- image data type  
  -- number of output feature maps  
  -- number of input feature maps  
  -- height of each input filter  
  -- width of  each input filter  
   function cudnnGetFilter4dDescriptor
     (filterDesc : cudnnFilterDescriptor_t;
      dataType : access cudnnDataType_t;
      format : access cudnnTensorFormat_t;
      k : access int;
      c : access int;
      h : access int;
      w : access int) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:492
   pragma Import (C, cudnnGetFilter4dDescriptor, "cudnnGetFilter4dDescriptor");

  -- image data type  
  -- number of output feature maps  
  -- number of input feature maps  
  -- height of each input filter  
  -- width of  each input filter  
   function cudnnSetFilterNdDescriptor
     (filterDesc : cudnnFilterDescriptor_t;
      dataType : cudnnDataType_t;
      format : cudnnTensorFormat_t;
      nbDims : int;
      filterDimA : access int) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:502
   pragma Import (C, cudnnSetFilterNdDescriptor, "cudnnSetFilterNdDescriptor");

  -- image data type  
   function cudnnGetFilterNdDescriptor
     (filterDesc : cudnnFilterDescriptor_t;
      nbDimsRequested : int;
      dataType : access cudnnDataType_t;
      format : access cudnnTensorFormat_t;
      nbDims : access int;
      filterDimA : access int) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:509
   pragma Import (C, cudnnGetFilterNdDescriptor, "cudnnGetFilterNdDescriptor");

  -- image data type  
   function cudnnDestroyFilterDescriptor (filterDesc : cudnnFilterDescriptor_t) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:518
   pragma Import (C, cudnnDestroyFilterDescriptor, "cudnnDestroyFilterDescriptor");

  -- Create an instance of convolution descriptor  
   function cudnnCreateConvolutionDescriptor (convDesc : System.Address) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:522
   pragma Import (C, cudnnCreateConvolutionDescriptor, "cudnnCreateConvolutionDescriptor");

   function cudnnSetConvolutionMathType (convDesc : cudnnConvolutionDescriptor_t; mathType : cudnnMathType_t) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:525
   pragma Import (C, cudnnSetConvolutionMathType, "cudnnSetConvolutionMathType");

   function cudnnGetConvolutionMathType (convDesc : cudnnConvolutionDescriptor_t; mathType : access cudnnMathType_t) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:528
   pragma Import (C, cudnnGetConvolutionMathType, "cudnnGetConvolutionMathType");

   function cudnnSetConvolutionGroupCount (convDesc : cudnnConvolutionDescriptor_t; groupCount : int) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:531
   pragma Import (C, cudnnSetConvolutionGroupCount, "cudnnSetConvolutionGroupCount");

   function cudnnGetConvolutionGroupCount (convDesc : cudnnConvolutionDescriptor_t; groupCount : access int) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:534
   pragma Import (C, cudnnGetConvolutionGroupCount, "cudnnGetConvolutionGroupCount");

   function cudnnSetConvolution2dDescriptor
     (convDesc : cudnnConvolutionDescriptor_t;
      pad_h : int;
      pad_w : int;
      u : int;
      v : int;
      dilation_h : int;
      dilation_w : int;
      mode : cudnnConvolutionMode_t;
      computeType : cudnnDataType_t) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:537
   pragma Import (C, cudnnSetConvolution2dDescriptor, "cudnnSetConvolution2dDescriptor");

  -- zero-padding height  
  -- zero-padding width  
  -- vertical filter stride  
  -- horizontal filter stride  
  -- filter dilation in the vertical dimension  
  -- filter dilation in the horizontal dimension  
   function cudnnGetConvolution2dDescriptor
     (convDesc : cudnnConvolutionDescriptor_t;
      pad_h : access int;
      pad_w : access int;
      u : access int;
      v : access int;
      dilation_h : access int;
      dilation_w : access int;
      mode : access cudnnConvolutionMode_t;
      computeType : access cudnnDataType_t) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:548
   pragma Import (C, cudnnGetConvolution2dDescriptor, "cudnnGetConvolution2dDescriptor");

  -- zero-padding height  
  -- zero-padding width  
  -- vertical filter stride  
  -- horizontal filter stride  
  -- filter dilation in the vertical dimension  
  -- filter dilation in the horizontal dimension  
  -- Helper function to return the dimensions of the output tensor given a convolution descriptor  
   function cudnnGetConvolution2dForwardOutputDim
     (convDesc : cudnnConvolutionDescriptor_t;
      inputTensorDesc : cudnnTensorDescriptor_t;
      filterDesc : cudnnFilterDescriptor_t;
      n : access int;
      c : access int;
      h : access int;
      w : access int) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:560
   pragma Import (C, cudnnGetConvolution2dForwardOutputDim, "cudnnGetConvolution2dForwardOutputDim");

   function cudnnSetConvolutionNdDescriptor
     (convDesc : cudnnConvolutionDescriptor_t;
      arrayLength : int;
      padA : access int;
      filterStrideA : access int;
      dilationA : access int;
      mode : cudnnConvolutionMode_t;
      computeType : cudnnDataType_t) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:570
   pragma Import (C, cudnnSetConvolutionNdDescriptor, "cudnnSetConvolutionNdDescriptor");

  -- nbDims-2 size  
  -- convolution data type  
   function cudnnGetConvolutionNdDescriptor
     (convDesc : cudnnConvolutionDescriptor_t;
      arrayLengthRequested : int;
      arrayLength : access int;
      padA : access int;
      strideA : access int;
      dilationA : access int;
      mode : access cudnnConvolutionMode_t;
      computeType : access cudnnDataType_t) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:579
   pragma Import (C, cudnnGetConvolutionNdDescriptor, "cudnnGetConvolutionNdDescriptor");

  -- convolution data type  
  -- Helper function to return the dimensions of the output tensor given a convolution descriptor  
   function cudnnGetConvolutionNdForwardOutputDim
     (convDesc : cudnnConvolutionDescriptor_t;
      inputTensorDesc : cudnnTensorDescriptor_t;
      filterDesc : cudnnFilterDescriptor_t;
      nbDims : int;
      tensorOuputDimA : access int) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:591
   pragma Import (C, cudnnGetConvolutionNdForwardOutputDim, "cudnnGetConvolutionNdForwardOutputDim");

  -- Destroy an instance of convolution descriptor  
   function cudnnDestroyConvolutionDescriptor (convDesc : cudnnConvolutionDescriptor_t) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:599
   pragma Import (C, cudnnDestroyConvolutionDescriptor, "cudnnDestroyConvolutionDescriptor");

  -- helper function to provide the convolution algo that fit best the requirement  
   type cudnnConvolutionFwdPreference_t is 
     (CUDNN_CONVOLUTION_FWD_NO_WORKSPACE,
      CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
      CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT);
   pragma Convention (C, cudnnConvolutionFwdPreference_t);  -- /usr/local/cuda-8.0/include/cudnn.h:609

   type cudnnConvolutionFwdAlgo_t is 
     (CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
      CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
      CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
      CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,
      CUDNN_CONVOLUTION_FWD_ALGO_FFT,
      CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING,
      CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
      CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED,
      CUDNN_CONVOLUTION_FWD_ALGO_COUNT);
   pragma Convention (C, cudnnConvolutionFwdAlgo_t);  -- /usr/local/cuda-8.0/include/cudnn.h:623

   type cudnnConvolutionFwdAlgoPerf_t_reserved_array is array (0 .. 2) of aliased int;
   type cudnnConvolutionFwdAlgoPerf_t is record
      algo : aliased cudnnConvolutionFwdAlgo_t;  -- /usr/local/cuda-8.0/include/cudnn.h:626
      status : aliased cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:627
      time : aliased float;  -- /usr/local/cuda-8.0/include/cudnn.h:628
      memory : aliased stddef_h.size_t;  -- /usr/local/cuda-8.0/include/cudnn.h:629
      determinism : aliased cudnnDeterminism_t;  -- /usr/local/cuda-8.0/include/cudnn.h:630
      mathType : aliased cudnnMathType_t;  -- /usr/local/cuda-8.0/include/cudnn.h:631
      reserved : aliased cudnnConvolutionFwdAlgoPerf_t_reserved_array;  -- /usr/local/cuda-8.0/include/cudnn.h:632
   end record;
   pragma Convention (C_Pass_By_Copy, cudnnConvolutionFwdAlgoPerf_t);  -- /usr/local/cuda-8.0/include/cudnn.h:633

   --  skipped anonymous struct anon_19

   function cudnnGetConvolutionForwardAlgorithmMaxCount (handle : cudnnHandle_t; count : access int) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:635
   pragma Import (C, cudnnGetConvolutionForwardAlgorithmMaxCount, "cudnnGetConvolutionForwardAlgorithmMaxCount");

   function cudnnFindConvolutionForwardAlgorithm
     (handle : cudnnHandle_t;
      xDesc : cudnnTensorDescriptor_t;
      wDesc : cudnnFilterDescriptor_t;
      convDesc : cudnnConvolutionDescriptor_t;
      yDesc : cudnnTensorDescriptor_t;
      requestedAlgoCount : int;
      returnedAlgoCount : access int;
      perfResults : access cudnnConvolutionFwdAlgoPerf_t) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:638
   pragma Import (C, cudnnFindConvolutionForwardAlgorithm, "cudnnFindConvolutionForwardAlgorithm");

   function cudnnFindConvolutionForwardAlgorithmEx
     (handle : cudnnHandle_t;
      xDesc : cudnnTensorDescriptor_t;
      x : System.Address;
      wDesc : cudnnFilterDescriptor_t;
      w : System.Address;
      convDesc : cudnnConvolutionDescriptor_t;
      yDesc : cudnnTensorDescriptor_t;
      y : System.Address;
      requestedAlgoCount : int;
      returnedAlgoCount : access int;
      perfResults : access cudnnConvolutionFwdAlgoPerf_t;
      workSpace : System.Address;
      workSpaceSizeInBytes : stddef_h.size_t) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:648
   pragma Import (C, cudnnFindConvolutionForwardAlgorithmEx, "cudnnFindConvolutionForwardAlgorithmEx");

   function cudnnGetConvolutionForwardAlgorithm
     (handle : cudnnHandle_t;
      xDesc : cudnnTensorDescriptor_t;
      wDesc : cudnnFilterDescriptor_t;
      convDesc : cudnnConvolutionDescriptor_t;
      yDesc : cudnnTensorDescriptor_t;
      preference : cudnnConvolutionFwdPreference_t;
      memoryLimitInBytes : stddef_h.size_t;
      algo : access cudnnConvolutionFwdAlgo_t) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:664
   pragma Import (C, cudnnGetConvolutionForwardAlgorithm, "cudnnGetConvolutionForwardAlgorithm");

   function cudnnGetConvolutionForwardAlgorithm_v7
     (handle : cudnnHandle_t;
      srcDesc : cudnnTensorDescriptor_t;
      filterDesc : cudnnFilterDescriptor_t;
      convDesc : cudnnConvolutionDescriptor_t;
      destDesc : cudnnTensorDescriptor_t;
      requestedAlgoCount : int;
      returnedAlgoCount : access int;
      perfResults : access cudnnConvolutionFwdAlgoPerf_t) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:675
   pragma Import (C, cudnnGetConvolutionForwardAlgorithm_v7, "cudnnGetConvolutionForwardAlgorithm_v7");

  -- *  convolution algorithm (which requires potentially some workspace)
  --  

  -- Helper function to return the minimum size of the workspace to be passed to the convolution given an algo 
   function cudnnGetConvolutionForwardWorkspaceSize
     (handle : cudnnHandle_t;
      xDesc : cudnnTensorDescriptor_t;
      wDesc : cudnnFilterDescriptor_t;
      convDesc : cudnnConvolutionDescriptor_t;
      yDesc : cudnnTensorDescriptor_t;
      algo : cudnnConvolutionFwdAlgo_t;
      sizeInBytes : access stddef_h.size_t) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:690
   pragma Import (C, cudnnGetConvolutionForwardWorkspaceSize, "cudnnGetConvolutionForwardWorkspaceSize");

  -- Convolution functions: All of the form "output = alpha * Op(inputs) + beta * output"  
  -- Function to perform the forward pass for batch convolution  
   function cudnnConvolutionForward
     (handle : cudnnHandle_t;
      alpha : System.Address;
      xDesc : cudnnTensorDescriptor_t;
      x : System.Address;
      wDesc : cudnnFilterDescriptor_t;
      w : System.Address;
      convDesc : cudnnConvolutionDescriptor_t;
      algo : cudnnConvolutionFwdAlgo_t;
      workSpace : System.Address;
      workSpaceSizeInBytes : stddef_h.size_t;
      beta : System.Address;
      yDesc : cudnnTensorDescriptor_t;
      y : System.Address) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:703
   pragma Import (C, cudnnConvolutionForward, "cudnnConvolutionForward");

  -- Fused conv/bias/activation operation : y = Act( alpha1 * conv(x) + alpha2 * z + bias )  
   function cudnnConvolutionBiasActivationForward
     (handle : cudnnHandle_t;
      alpha1 : System.Address;
      xDesc : cudnnTensorDescriptor_t;
      x : System.Address;
      wDesc : cudnnFilterDescriptor_t;
      w : System.Address;
      convDesc : cudnnConvolutionDescriptor_t;
      algo : cudnnConvolutionFwdAlgo_t;
      workSpace : System.Address;
      workSpaceSizeInBytes : stddef_h.size_t;
      alpha2 : System.Address;
      zDesc : cudnnTensorDescriptor_t;
      z : System.Address;
      biasDesc : cudnnTensorDescriptor_t;
      bias : System.Address;
      activationDesc : cudnnActivationDescriptor_t;
      yDesc : cudnnTensorDescriptor_t;
      y : System.Address) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:719
   pragma Import (C, cudnnConvolutionBiasActivationForward, "cudnnConvolutionBiasActivationForward");

  -- Function to compute the bias gradient for batch convolution  
   function cudnnConvolutionBackwardBias
     (handle : cudnnHandle_t;
      alpha : System.Address;
      dyDesc : cudnnTensorDescriptor_t;
      dy : System.Address;
      beta : System.Address;
      dbDesc : cudnnTensorDescriptor_t;
      db : System.Address) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:740
   pragma Import (C, cudnnConvolutionBackwardBias, "cudnnConvolutionBackwardBias");

  -- helper function to provide the convolution algo that fit best the requirement  
   type cudnnConvolutionBwdFilterPreference_t is 
     (CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE,
      CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
      CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT);
   pragma Convention (C, cudnnConvolutionBwdFilterPreference_t);  -- /usr/local/cuda-8.0/include/cudnn.h:756

  -- non-deterministic  
  -- non-deterministic  
  -- not implemented  
   type cudnnConvolutionBwdFilterAlgo_t is 
     (CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
      CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1,
      CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT,
      CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3,
      CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD,
      CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED,
      CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING,
      CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT);
   pragma Convention (C, cudnnConvolutionBwdFilterAlgo_t);  -- /usr/local/cuda-8.0/include/cudnn.h:768

   type cudnnConvolutionBwdFilterAlgoPerf_t_reserved_array is array (0 .. 2) of aliased int;
   type cudnnConvolutionBwdFilterAlgoPerf_t is record
      algo : aliased cudnnConvolutionBwdFilterAlgo_t;  -- /usr/local/cuda-8.0/include/cudnn.h:772
      status : aliased cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:773
      time : aliased float;  -- /usr/local/cuda-8.0/include/cudnn.h:774
      memory : aliased stddef_h.size_t;  -- /usr/local/cuda-8.0/include/cudnn.h:775
      determinism : aliased cudnnDeterminism_t;  -- /usr/local/cuda-8.0/include/cudnn.h:776
      mathType : aliased cudnnMathType_t;  -- /usr/local/cuda-8.0/include/cudnn.h:777
      reserved : aliased cudnnConvolutionBwdFilterAlgoPerf_t_reserved_array;  -- /usr/local/cuda-8.0/include/cudnn.h:778
   end record;
   pragma Convention (C_Pass_By_Copy, cudnnConvolutionBwdFilterAlgoPerf_t);  -- /usr/local/cuda-8.0/include/cudnn.h:779

   --  skipped anonymous struct anon_22

   function cudnnGetConvolutionBackwardFilterAlgorithmMaxCount (handle : cudnnHandle_t; count : access int) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:781
   pragma Import (C, cudnnGetConvolutionBackwardFilterAlgorithmMaxCount, "cudnnGetConvolutionBackwardFilterAlgorithmMaxCount");

   function cudnnFindConvolutionBackwardFilterAlgorithm
     (handle : cudnnHandle_t;
      xDesc : cudnnTensorDescriptor_t;
      dyDesc : cudnnTensorDescriptor_t;
      convDesc : cudnnConvolutionDescriptor_t;
      dwDesc : cudnnFilterDescriptor_t;
      requestedAlgoCount : int;
      returnedAlgoCount : access int;
      perfResults : access cudnnConvolutionBwdFilterAlgoPerf_t) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:784
   pragma Import (C, cudnnFindConvolutionBackwardFilterAlgorithm, "cudnnFindConvolutionBackwardFilterAlgorithm");

   function cudnnFindConvolutionBackwardFilterAlgorithmEx
     (handle : cudnnHandle_t;
      xDesc : cudnnTensorDescriptor_t;
      x : System.Address;
      dyDesc : cudnnTensorDescriptor_t;
      y : System.Address;
      convDesc : cudnnConvolutionDescriptor_t;
      dwDesc : cudnnFilterDescriptor_t;
      dw : System.Address;
      requestedAlgoCount : int;
      returnedAlgoCount : access int;
      perfResults : access cudnnConvolutionBwdFilterAlgoPerf_t;
      workSpace : System.Address;
      workSpaceSizeInBytes : stddef_h.size_t) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:794
   pragma Import (C, cudnnFindConvolutionBackwardFilterAlgorithmEx, "cudnnFindConvolutionBackwardFilterAlgorithmEx");

   function cudnnGetConvolutionBackwardFilterAlgorithm
     (handle : cudnnHandle_t;
      xDesc : cudnnTensorDescriptor_t;
      dyDesc : cudnnTensorDescriptor_t;
      convDesc : cudnnConvolutionDescriptor_t;
      dwDesc : cudnnFilterDescriptor_t;
      preference : cudnnConvolutionBwdFilterPreference_t;
      memoryLimitInBytes : stddef_h.size_t;
      algo : access cudnnConvolutionBwdFilterAlgo_t) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:809
   pragma Import (C, cudnnGetConvolutionBackwardFilterAlgorithm, "cudnnGetConvolutionBackwardFilterAlgorithm");

   function cudnnGetConvolutionBackwardFilterAlgorithm_v7
     (handle : cudnnHandle_t;
      srcDesc : cudnnTensorDescriptor_t;
      diffDesc : cudnnTensorDescriptor_t;
      convDesc : cudnnConvolutionDescriptor_t;
      gradDesc : cudnnFilterDescriptor_t;
      requestedAlgoCount : int;
      returnedAlgoCount : access int;
      perfResults : access cudnnConvolutionBwdFilterAlgoPerf_t) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:819
   pragma Import (C, cudnnGetConvolutionBackwardFilterAlgorithm_v7, "cudnnGetConvolutionBackwardFilterAlgorithm_v7");

  -- *  convolution algorithm (which requires potentially some workspace)
  --  

  -- Helper function to return the minimum size of the workspace to be passed to the convolution given an algo 
   function cudnnGetConvolutionBackwardFilterWorkspaceSize
     (handle : cudnnHandle_t;
      xDesc : cudnnTensorDescriptor_t;
      dyDesc : cudnnTensorDescriptor_t;
      convDesc : cudnnConvolutionDescriptor_t;
      gradDesc : cudnnFilterDescriptor_t;
      algo : cudnnConvolutionBwdFilterAlgo_t;
      sizeInBytes : access stddef_h.size_t) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:834
   pragma Import (C, cudnnGetConvolutionBackwardFilterWorkspaceSize, "cudnnGetConvolutionBackwardFilterWorkspaceSize");

   function cudnnConvolutionBackwardFilter
     (handle : cudnnHandle_t;
      alpha : System.Address;
      xDesc : cudnnTensorDescriptor_t;
      x : System.Address;
      dyDesc : cudnnTensorDescriptor_t;
      dy : System.Address;
      convDesc : cudnnConvolutionDescriptor_t;
      algo : cudnnConvolutionBwdFilterAlgo_t;
      workSpace : System.Address;
      workSpaceSizeInBytes : stddef_h.size_t;
      beta : System.Address;
      dwDesc : cudnnFilterDescriptor_t;
      dw : System.Address) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:843
   pragma Import (C, cudnnConvolutionBackwardFilter, "cudnnConvolutionBackwardFilter");

  --******************************************************* 
  -- helper function to provide the convolution algo that fit best the requirement  
   type cudnnConvolutionBwdDataPreference_t is 
     (CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE,
      CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
      CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT);
   pragma Convention (C, cudnnConvolutionBwdDataPreference_t);  -- /usr/local/cuda-8.0/include/cudnn.h:865

  -- non-deterministic  
   type cudnnConvolutionBwdDataAlgo_t is 
     (CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
      CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,
      CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT,
      CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING,
      CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD,
      CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED,
      CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT);
   pragma Convention (C, cudnnConvolutionBwdDataAlgo_t);  -- /usr/local/cuda-8.0/include/cudnn.h:876

   type cudnnConvolutionBwdDataAlgoPerf_t_reserved_array is array (0 .. 2) of aliased int;
   type cudnnConvolutionBwdDataAlgoPerf_t is record
      algo : aliased cudnnConvolutionBwdDataAlgo_t;  -- /usr/local/cuda-8.0/include/cudnn.h:879
      status : aliased cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:880
      time : aliased float;  -- /usr/local/cuda-8.0/include/cudnn.h:881
      memory : aliased stddef_h.size_t;  -- /usr/local/cuda-8.0/include/cudnn.h:882
      determinism : aliased cudnnDeterminism_t;  -- /usr/local/cuda-8.0/include/cudnn.h:883
      mathType : aliased cudnnMathType_t;  -- /usr/local/cuda-8.0/include/cudnn.h:884
      reserved : aliased cudnnConvolutionBwdDataAlgoPerf_t_reserved_array;  -- /usr/local/cuda-8.0/include/cudnn.h:885
   end record;
   pragma Convention (C_Pass_By_Copy, cudnnConvolutionBwdDataAlgoPerf_t);  -- /usr/local/cuda-8.0/include/cudnn.h:886

   --  skipped anonymous struct anon_25

   function cudnnGetConvolutionBackwardDataAlgorithmMaxCount (handle : cudnnHandle_t; count : access int) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:888
   pragma Import (C, cudnnGetConvolutionBackwardDataAlgorithmMaxCount, "cudnnGetConvolutionBackwardDataAlgorithmMaxCount");

   function cudnnFindConvolutionBackwardDataAlgorithm
     (handle : cudnnHandle_t;
      wDesc : cudnnFilterDescriptor_t;
      dyDesc : cudnnTensorDescriptor_t;
      convDesc : cudnnConvolutionDescriptor_t;
      dxDesc : cudnnTensorDescriptor_t;
      requestedAlgoCount : int;
      returnedAlgoCount : access int;
      perfResults : access cudnnConvolutionBwdDataAlgoPerf_t) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:891
   pragma Import (C, cudnnFindConvolutionBackwardDataAlgorithm, "cudnnFindConvolutionBackwardDataAlgorithm");

   function cudnnFindConvolutionBackwardDataAlgorithmEx
     (handle : cudnnHandle_t;
      wDesc : cudnnFilterDescriptor_t;
      w : System.Address;
      dyDesc : cudnnTensorDescriptor_t;
      dy : System.Address;
      convDesc : cudnnConvolutionDescriptor_t;
      dxDesc : cudnnTensorDescriptor_t;
      dx : System.Address;
      requestedAlgoCount : int;
      returnedAlgoCount : access int;
      perfResults : access cudnnConvolutionBwdDataAlgoPerf_t;
      workSpace : System.Address;
      workSpaceSizeInBytes : stddef_h.size_t) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:901
   pragma Import (C, cudnnFindConvolutionBackwardDataAlgorithmEx, "cudnnFindConvolutionBackwardDataAlgorithmEx");

   function cudnnGetConvolutionBackwardDataAlgorithm
     (handle : cudnnHandle_t;
      wDesc : cudnnFilterDescriptor_t;
      dyDesc : cudnnTensorDescriptor_t;
      convDesc : cudnnConvolutionDescriptor_t;
      dxDesc : cudnnTensorDescriptor_t;
      preference : cudnnConvolutionBwdDataPreference_t;
      memoryLimitInBytes : stddef_h.size_t;
      algo : access cudnnConvolutionBwdDataAlgo_t) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:916
   pragma Import (C, cudnnGetConvolutionBackwardDataAlgorithm, "cudnnGetConvolutionBackwardDataAlgorithm");

   function cudnnGetConvolutionBackwardDataAlgorithm_v7
     (handle : cudnnHandle_t;
      filterDesc : cudnnFilterDescriptor_t;
      diffDesc : cudnnTensorDescriptor_t;
      convDesc : cudnnConvolutionDescriptor_t;
      gradDesc : cudnnTensorDescriptor_t;
      requestedAlgoCount : int;
      returnedAlgoCount : access int;
      perfResults : access cudnnConvolutionBwdDataAlgoPerf_t) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:926
   pragma Import (C, cudnnGetConvolutionBackwardDataAlgorithm_v7, "cudnnGetConvolutionBackwardDataAlgorithm_v7");

  -- Helper function to return the minimum size of the workspace to be passed to the convolution given an algo 
   function cudnnGetConvolutionBackwardDataWorkspaceSize
     (handle : cudnnHandle_t;
      wDesc : cudnnFilterDescriptor_t;
      dyDesc : cudnnTensorDescriptor_t;
      convDesc : cudnnConvolutionDescriptor_t;
      dxDesc : cudnnTensorDescriptor_t;
      algo : cudnnConvolutionBwdDataAlgo_t;
      sizeInBytes : access stddef_h.size_t) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:937
   pragma Import (C, cudnnGetConvolutionBackwardDataWorkspaceSize, "cudnnGetConvolutionBackwardDataWorkspaceSize");

   function cudnnConvolutionBackwardData
     (handle : cudnnHandle_t;
      alpha : System.Address;
      wDesc : cudnnFilterDescriptor_t;
      w : System.Address;
      dyDesc : cudnnTensorDescriptor_t;
      dy : System.Address;
      convDesc : cudnnConvolutionDescriptor_t;
      algo : cudnnConvolutionBwdDataAlgo_t;
      workSpace : System.Address;
      workSpaceSizeInBytes : stddef_h.size_t;
      beta : System.Address;
      dxDesc : cudnnTensorDescriptor_t;
      dx : System.Address) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:947
   pragma Import (C, cudnnConvolutionBackwardData, "cudnnConvolutionBackwardData");

   function cudnnIm2Col
     (handle : cudnnHandle_t;
      xDesc : cudnnTensorDescriptor_t;
      x : System.Address;
      wDesc : cudnnFilterDescriptor_t;
      convDesc : cudnnConvolutionDescriptor_t;
      colBuffer : System.Address) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:963
   pragma Import (C, cudnnIm2Col, "cudnnIm2Col");

  -- *  softmax algorithm
  --  

  -- straightforward implementation  
  -- subtract max from every point to avoid overflow  
   type cudnnSoftmaxAlgorithm_t is 
     (CUDNN_SOFTMAX_FAST,
      CUDNN_SOFTMAX_ACCURATE,
      CUDNN_SOFTMAX_LOG);
   pragma Convention (C, cudnnSoftmaxAlgorithm_t);  -- /usr/local/cuda-8.0/include/cudnn.h:980

  -- compute the softmax over all C, H, W for each N  
  -- compute the softmax over all C for each H, W, N  
   type cudnnSoftmaxMode_t is 
     (CUDNN_SOFTMAX_MODE_INSTANCE,
      CUDNN_SOFTMAX_MODE_CHANNEL);
   pragma Convention (C, cudnnSoftmaxMode_t);  -- /usr/local/cuda-8.0/include/cudnn.h:986

  -- Softmax functions: All of the form "output = alpha * Op(inputs) + beta * output"  
  -- Function to perform forward softmax  
   function cudnnSoftmaxForward
     (handle : cudnnHandle_t;
      algo : cudnnSoftmaxAlgorithm_t;
      mode : cudnnSoftmaxMode_t;
      alpha : System.Address;
      xDesc : cudnnTensorDescriptor_t;
      x : System.Address;
      beta : System.Address;
      yDesc : cudnnTensorDescriptor_t;
      y : System.Address) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:991
   pragma Import (C, cudnnSoftmaxForward, "cudnnSoftmaxForward");

  -- Function to perform backward softmax  
   function cudnnSoftmaxBackward
     (handle : cudnnHandle_t;
      algo : cudnnSoftmaxAlgorithm_t;
      mode : cudnnSoftmaxMode_t;
      alpha : System.Address;
      yDesc : cudnnTensorDescriptor_t;
      y : System.Address;
      dyDesc : cudnnTensorDescriptor_t;
      dy : System.Address;
      beta : System.Address;
      dxDesc : cudnnTensorDescriptor_t;
      dx : System.Address) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:1003
   pragma Import (C, cudnnSoftmaxBackward, "cudnnSoftmaxBackward");

  -- *  pooling mode
  --  

  -- count for average includes padded values  
  -- count for average does not include padded values  
   type cudnnPoolingMode_t is 
     (CUDNN_POOLING_MAX,
      CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING,
      CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING,
      CUDNN_POOLING_MAX_DETERMINISTIC);
   pragma Convention (C, cudnnPoolingMode_t);  -- /usr/local/cuda-8.0/include/cudnn.h:1025

  -- Create an instance of pooling descriptor  
   function cudnnCreatePoolingDescriptor (poolingDesc : System.Address) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:1028
   pragma Import (C, cudnnCreatePoolingDescriptor, "cudnnCreatePoolingDescriptor");

   function cudnnSetPooling2dDescriptor
     (poolingDesc : cudnnPoolingDescriptor_t;
      mode : cudnnPoolingMode_t;
      maxpoolingNanOpt : cudnnNanPropagation_t;
      windowHeight : int;
      windowWidth : int;
      verticalPadding : int;
      horizontalPadding : int;
      verticalStride : int;
      horizontalStride : int) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:1031
   pragma Import (C, cudnnSetPooling2dDescriptor, "cudnnSetPooling2dDescriptor");

   function cudnnGetPooling2dDescriptor
     (poolingDesc : cudnnPoolingDescriptor_t;
      mode : access cudnnPoolingMode_t;
      maxpoolingNanOpt : access cudnnNanPropagation_t;
      windowHeight : access int;
      windowWidth : access int;
      verticalPadding : access int;
      horizontalPadding : access int;
      verticalStride : access int;
      horizontalStride : access int) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:1042
   pragma Import (C, cudnnGetPooling2dDescriptor, "cudnnGetPooling2dDescriptor");

   function cudnnSetPoolingNdDescriptor
     (poolingDesc : cudnnPoolingDescriptor_t;
      mode : cudnnPoolingMode_t;
      maxpoolingNanOpt : cudnnNanPropagation_t;
      nbDims : int;
      windowDimA : access int;
      paddingA : access int;
      strideA : access int) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:1053
   pragma Import (C, cudnnSetPoolingNdDescriptor, "cudnnSetPoolingNdDescriptor");

   function cudnnGetPoolingNdDescriptor
     (poolingDesc : cudnnPoolingDescriptor_t;
      nbDimsRequested : int;
      mode : access cudnnPoolingMode_t;
      maxpoolingNanOpt : access cudnnNanPropagation_t;
      nbDims : access int;
      windowDimA : access int;
      paddingA : access int;
      strideA : access int) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:1062
   pragma Import (C, cudnnGetPoolingNdDescriptor, "cudnnGetPoolingNdDescriptor");

   function cudnnGetPoolingNdForwardOutputDim
     (poolingDesc : cudnnPoolingDescriptor_t;
      inputTensorDesc : cudnnTensorDescriptor_t;
      nbDims : int;
      outputTensorDimA : access int) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:1072
   pragma Import (C, cudnnGetPoolingNdForwardOutputDim, "cudnnGetPoolingNdForwardOutputDim");

   function cudnnGetPooling2dForwardOutputDim
     (poolingDesc : cudnnPoolingDescriptor_t;
      inputTensorDesc : cudnnTensorDescriptor_t;
      n : access int;
      c : access int;
      h : access int;
      w : access int) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:1078
   pragma Import (C, cudnnGetPooling2dForwardOutputDim, "cudnnGetPooling2dForwardOutputDim");

  -- Destroy an instance of pooling descriptor  
   function cudnnDestroyPoolingDescriptor (poolingDesc : cudnnPoolingDescriptor_t) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:1088
   pragma Import (C, cudnnDestroyPoolingDescriptor, "cudnnDestroyPoolingDescriptor");

  -- Pooling functions: All of the form "output = alpha * Op(inputs) + beta * output"  
  -- Function to perform forward pooling  
   function cudnnPoolingForward
     (handle : cudnnHandle_t;
      poolingDesc : cudnnPoolingDescriptor_t;
      alpha : System.Address;
      xDesc : cudnnTensorDescriptor_t;
      x : System.Address;
      beta : System.Address;
      yDesc : cudnnTensorDescriptor_t;
      y : System.Address) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:1094
   pragma Import (C, cudnnPoolingForward, "cudnnPoolingForward");

  -- Function to perform backward pooling  
   function cudnnPoolingBackward
     (handle : cudnnHandle_t;
      poolingDesc : cudnnPoolingDescriptor_t;
      alpha : System.Address;
      yDesc : cudnnTensorDescriptor_t;
      y : System.Address;
      dyDesc : cudnnTensorDescriptor_t;
      dy : System.Address;
      xDesc : cudnnTensorDescriptor_t;
      x : System.Address;
      beta : System.Address;
      dxDesc : cudnnTensorDescriptor_t;
      dx : System.Address) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:1105
   pragma Import (C, cudnnPoolingBackward, "cudnnPoolingBackward");

  -- * activation mode
  --  

   type cudnnActivationMode_t is 
     (CUDNN_ACTIVATION_SIGMOID,
      CUDNN_ACTIVATION_RELU,
      CUDNN_ACTIVATION_TANH,
      CUDNN_ACTIVATION_CLIPPED_RELU,
      CUDNN_ACTIVATION_ELU);
   pragma Convention (C, cudnnActivationMode_t);  -- /usr/local/cuda-8.0/include/cudnn.h:1129

  -- Activation functions: All of the form "output = alpha * Op(inputs) + beta * output"  
   function cudnnCreateActivationDescriptor (activationDesc : System.Address) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:1132
   pragma Import (C, cudnnCreateActivationDescriptor, "cudnnCreateActivationDescriptor");

   function cudnnSetActivationDescriptor
     (activationDesc : cudnnActivationDescriptor_t;
      mode : cudnnActivationMode_t;
      reluNanOpt : cudnnNanPropagation_t;
      coef : double) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:1135
   pragma Import (C, cudnnSetActivationDescriptor, "cudnnSetActivationDescriptor");

  -- ceiling for clipped RELU, alpha for ELU  
   function cudnnGetActivationDescriptor
     (activationDesc : cudnnActivationDescriptor_t;
      mode : access cudnnActivationMode_t;
      reluNanOpt : access cudnnNanPropagation_t;
      coef : access double) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:1141
   pragma Import (C, cudnnGetActivationDescriptor, "cudnnGetActivationDescriptor");

  -- ceiling for clipped RELU, alpha for ELU  
   function cudnnDestroyActivationDescriptor (activationDesc : cudnnActivationDescriptor_t) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:1147
   pragma Import (C, cudnnDestroyActivationDescriptor, "cudnnDestroyActivationDescriptor");

  -- Function to perform forward activation   
   function cudnnActivationForward
     (handle : cudnnHandle_t;
      activationDesc : cudnnActivationDescriptor_t;
      alpha : System.Address;
      xDesc : cudnnTensorDescriptor_t;
      x : System.Address;
      beta : System.Address;
      yDesc : cudnnTensorDescriptor_t;
      y : System.Address) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:1151
   pragma Import (C, cudnnActivationForward, "cudnnActivationForward");

  -- Function to perform backward activation   
   function cudnnActivationBackward
     (handle : cudnnHandle_t;
      activationDesc : cudnnActivationDescriptor_t;
      alpha : System.Address;
      yDesc : cudnnTensorDescriptor_t;
      y : System.Address;
      dyDesc : cudnnTensorDescriptor_t;
      dy : System.Address;
      xDesc : cudnnTensorDescriptor_t;
      x : System.Address;
      beta : System.Address;
      dxDesc : cudnnTensorDescriptor_t;
      dx : System.Address) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:1162
   pragma Import (C, cudnnActivationBackward, "cudnnActivationBackward");

  -- 
  --* Create an instance of LRN (Local Response Normalization) descriptor
  --* Uses lrnN=5, lrnAlpha=1e-4, lrnBeta=0.75, lrnK=2.0 as defaults from Krizhevsky'12 ImageNet paper
  -- 

   function cudnnCreateLRNDescriptor (normDesc : System.Address) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:1180
   pragma Import (C, cudnnCreateLRNDescriptor, "cudnnCreateLRNDescriptor");

  -- LRN layer mode  
  -- Normalize across tensor's dimA[1] dimension  
   type cudnnLRNMode_t is 
     (CUDNN_LRN_CROSS_CHANNEL_DIM1);
   pragma Convention (C, cudnnLRNMode_t);  -- /usr/local/cuda-8.0/include/cudnn.h:1192

  --* Uses a window [center-lookBehind, center+lookAhead], where
  --* lookBehind = floor( (lrnN-1)/2 ), lookAhead = lrnN-lookBehind-1.
  --* Values of double parameters cast to tensor data type.
  -- 

   function cudnnSetLRNDescriptor
     (normDesc : cudnnLRNDescriptor_t;
      lrnN : unsigned;
      lrnAlpha : double;
      lrnBeta : double;
      lrnK : double) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:1199
   pragma Import (C, cudnnSetLRNDescriptor, "cudnnSetLRNDescriptor");

  --* Retrieve the settings currently stored in an LRN layer descriptor
  --* Any of the provided pointers can be NULL (no corresponding value will be returned)
  -- 

   function cudnnGetLRNDescriptor
     (normDesc : cudnnLRNDescriptor_t;
      lrnN : access unsigned;
      lrnAlpha : access double;
      lrnBeta : access double;
      lrnK : access double) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:1209
   pragma Import (C, cudnnGetLRNDescriptor, "cudnnGetLRNDescriptor");

  -- Destroy an instance of LRN descriptor  
   function cudnnDestroyLRNDescriptor (lrnDesc : cudnnLRNDescriptor_t) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:1217
   pragma Import (C, cudnnDestroyLRNDescriptor, "cudnnDestroyLRNDescriptor");

  -- LRN functions: output = alpha * normalize(x) + beta * old_y  
  -- LRN cross-channel forward computation. Double parameters cast to tensor data type  
   function cudnnLRNCrossChannelForward
     (handle : cudnnHandle_t;
      normDesc : cudnnLRNDescriptor_t;
      lrnMode : cudnnLRNMode_t;
      alpha : System.Address;
      xDesc : cudnnTensorDescriptor_t;
      x : System.Address;
      beta : System.Address;
      yDesc : cudnnTensorDescriptor_t;
      y : System.Address) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:1222
   pragma Import (C, cudnnLRNCrossChannelForward, "cudnnLRNCrossChannelForward");

  -- LRN cross-channel backward computation. Double parameters cast to tensor data type  
   function cudnnLRNCrossChannelBackward
     (handle : cudnnHandle_t;
      normDesc : cudnnLRNDescriptor_t;
      lrnMode : cudnnLRNMode_t;
      alpha : System.Address;
      yDesc : cudnnTensorDescriptor_t;
      y : System.Address;
      dyDesc : cudnnTensorDescriptor_t;
      dy : System.Address;
      xDesc : cudnnTensorDescriptor_t;
      x : System.Address;
      beta : System.Address;
      dxDesc : cudnnTensorDescriptor_t;
      dx : System.Address) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:1234
   pragma Import (C, cudnnLRNCrossChannelBackward, "cudnnLRNCrossChannelBackward");

   type cudnnDivNormMode_t is 
     (CUDNN_DIVNORM_PRECOMPUTED_MEANS);
   pragma Convention (C, cudnnDivNormMode_t);  -- /usr/local/cuda-8.0/include/cudnn.h:1252

  -- LCN/divisive normalization functions: y = alpha * normalize(x) + beta * y  
   function cudnnDivisiveNormalizationForward
     (handle : cudnnHandle_t;
      normDesc : cudnnLRNDescriptor_t;
      mode : cudnnDivNormMode_t;
      alpha : System.Address;
      xDesc : cudnnTensorDescriptor_t;
      x : System.Address;
      means : System.Address;
      temp : System.Address;
      temp2 : System.Address;
      beta : System.Address;
      yDesc : cudnnTensorDescriptor_t;
      y : System.Address) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:1255
   pragma Import (C, cudnnDivisiveNormalizationForward, "cudnnDivisiveNormalizationForward");

  -- same desc for means, temp, temp2  
  -- if NULL, means are assumed to be zero  
   function cudnnDivisiveNormalizationBackward
     (handle : cudnnHandle_t;
      normDesc : cudnnLRNDescriptor_t;
      mode : cudnnDivNormMode_t;
      alpha : System.Address;
      xDesc : cudnnTensorDescriptor_t;
      x : System.Address;
      means : System.Address;
      dy : System.Address;
      temp : System.Address;
      temp2 : System.Address;
      beta : System.Address;
      dXdMeansDesc : cudnnTensorDescriptor_t;
      dx : System.Address;
      dMeans : System.Address) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:1269
   pragma Import (C, cudnnDivisiveNormalizationBackward, "cudnnDivisiveNormalizationBackward");

  -- same desc for x, means, dy, temp, temp2  
  -- if NULL, means are assumed to be zero  
  -- same desc for dx, dMeans  
  -- output x differential  
  -- output means differential, can be NULL  
  -- bnScale, bnBias tensor dims are 1xCxHxWx.. (one value per CHW...-slice, normalized over N slice)  
  -- bnScale, bnBias tensor dims are 1xCx1x1 (one value per C-dim normalized over Nx1xHxW subtensors)  
  -- 
  --     * bnScale, bnBias tensor dims are 1xCx1x1 (one value per C-dim normalized over Nx1xHxW subtensors). 
  --     * May be faster than CUDNN_BATCHNORM_SPATIAL but imposes some limits on the range of values 
  --      

   type cudnnBatchNormMode_t is 
     (CUDNN_BATCHNORM_PER_ACTIVATION,
      CUDNN_BATCHNORM_SPATIAL,
      CUDNN_BATCHNORM_SPATIAL_PERSISTENT);
   pragma Convention (C, cudnnBatchNormMode_t);  -- /usr/local/cuda-8.0/include/cudnn.h:1298

  --* Derives a tensor descriptor from layer data descriptor for BatchNormalization 
  --* scale, invVariance, bnBias, bnScale tensors. Use this tensor desc for 
  --* bnScaleBiasMeanVarDesc and bnScaleBiasDiffDesc in Batch Normalization forward and backward functions.
  -- 

   function cudnnDeriveBNTensorDescriptor
     (derivedBnDesc : cudnnTensorDescriptor_t;
      xDesc : cudnnTensorDescriptor_t;
      mode : cudnnBatchNormMode_t) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:1307
   pragma Import (C, cudnnDeriveBNTensorDescriptor, "cudnnDeriveBNTensorDescriptor");

  -- Computes y = BN(x). Also accumulates moving averages of mean and inverse variances  
   function cudnnBatchNormalizationForwardTraining
     (handle : cudnnHandle_t;
      mode : cudnnBatchNormMode_t;
      alpha : System.Address;
      beta : System.Address;
      xDesc : cudnnTensorDescriptor_t;
      x : System.Address;
      yDesc : cudnnTensorDescriptor_t;
      y : System.Address;
      bnScaleBiasMeanVarDesc : cudnnTensorDescriptor_t;
      bnScale : System.Address;
      bnBias : System.Address;
      exponentialAverageFactor : double;
      resultRunningMean : System.Address;
      resultRunningVariance : System.Address;
      epsilon : double;
      resultSaveMean : System.Address;
      resultSaveInvVariance : System.Address) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:1313
   pragma Import (C, cudnnBatchNormalizationForwardTraining, "cudnnBatchNormalizationForwardTraining");

  -- alpha[0] = result blend factor  
  -- beta[0] = dest layer blend factor  
  -- NxCxHxW  
  -- NxCxHxW  
  -- Shared desc for the next 6 tensors in the argument list.
  --                                   Data type to be set as follows:
  --                                   type = (typeOf(x) == double) ? double : float
  --                                   Dimensions for this descriptor depend on normalization mode
  --                                   - Spatial Normalization : tensors are expected to have dims 1xCx1x1
  --                                    (normalization is performed across NxHxW)
  --                                   - Per-Activation Normalization : tensors are expected to have dims of 1xCxHxW 
  --                                    (normalization is performed across N)  

  -- 'Gamma' and 'Beta' respectively in Ioffe and Szegedy's paper's notation  
  -- MUST use factor=1 in the very first call of a complete training cycle.
  --                                   Use a factor=1/(1+n) at N-th call to the function to get
  --                                   Cumulative Moving Average (CMA) behavior
  --                                   CMA[n] = (x[1]+...+x[n])/n
  --                                   Since CMA[n+1] = (n*CMA[n]+x[n+1])/(n+1) =
  --                                   ((n+1)*CMA[n]-CMA[n])/(n+1) + x[n+1]/(n+1) =
  --                                   CMA[n]*(1-1/(n+1)) + x[n+1]*1/(n+1)  

  -- Used in Training phase only. 
  --                                   runningMean = newMean*factor + runningMean*(1-factor)  

  -- Output in training mode, input in inference. Is the moving average
  --                                   of  variance[x] (factor is applied in the same way as for runningMean)  

  -- Has to be >= CUDNN_BN_MIN_EPSILON. Should be the same in forward and backward functions.  
  -- Optionally save intermediate results from the forward pass here
  --                                   - can be reused to speed up backward pass. NULL if unused  

  --* Performs Batch Normalization during Inference: 
  --* y[i] = bnScale[k]*(x[i]-estimatedMean[k])/sqrt(epsilon+estimatedVariance[k]) + bnBias[k]
  --* with bnScale, bnBias, runningMean, runningInvVariance tensors indexed
  --* according to spatial or per-activation mode. Refer to cudnnBatchNormalizationForwardTraining
  --* above for notes on function arguments.
  -- 

   function cudnnBatchNormalizationForwardInference
     (handle : cudnnHandle_t;
      mode : cudnnBatchNormMode_t;
      alpha : System.Address;
      beta : System.Address;
      xDesc : cudnnTensorDescriptor_t;
      x : System.Address;
      yDesc : cudnnTensorDescriptor_t;
      y : System.Address;
      bnScaleBiasMeanVarDesc : cudnnTensorDescriptor_t;
      bnScale : System.Address;
      bnBias : System.Address;
      estimatedMean : System.Address;
      estimatedVariance : System.Address;
      epsilon : double) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:1370
   pragma Import (C, cudnnBatchNormalizationForwardInference, "cudnnBatchNormalizationForwardInference");

  -- alpha[0] = result blend factor  
  -- beta[0] = dest layer blend factor  
  -- NxCxHxW  
  -- NxCxHxW  
  -- Performs backward pass of Batch Normalization layer. Returns x gradient,
  --* bnScale gradient and bnBias gradient  

   function cudnnBatchNormalizationBackward
     (handle : cudnnHandle_t;
      mode : cudnnBatchNormMode_t;
      alphaDataDiff : System.Address;
      betaDataDiff : System.Address;
      alphaParamDiff : System.Address;
      betaParamDiff : System.Address;
      xDesc : cudnnTensorDescriptor_t;
      x : System.Address;
      dyDesc : cudnnTensorDescriptor_t;
      dy : System.Address;
      dxDesc : cudnnTensorDescriptor_t;
      dx : System.Address;
      dBnScaleBiasDesc : cudnnTensorDescriptor_t;
      bnScale : System.Address;
      dBnScaleResult : System.Address;
      dBnBiasResult : System.Address;
      epsilon : double;
      savedMean : System.Address;
      savedInvVariance : System.Address) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:1388
   pragma Import (C, cudnnBatchNormalizationBackward, "cudnnBatchNormalizationBackward");

  -- same desc for x, dx, dy  
  -- Shared tensor desc for the 4 tensors below  
  -- bnBias doesn't affect backpropagation  
  -- scale and bias diff are not backpropagated below this layer  
  -- Same epsilon as forward pass  
  -- Optionally cached intermediate results from
  --                                   forward pass  

  -- APIs for spatial transformer network 
   type cudnnSamplerType_t is 
     (CUDNN_SAMPLER_BILINEAR);
   pragma Convention (C, cudnnSamplerType_t);  -- /usr/local/cuda-8.0/include/cudnn.h:1419

   function cudnnCreateSpatialTransformerDescriptor (stDesc : System.Address) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:1421
   pragma Import (C, cudnnCreateSpatialTransformerDescriptor, "cudnnCreateSpatialTransformerDescriptor");

   function cudnnSetSpatialTransformerNdDescriptor
     (stDesc : cudnnSpatialTransformerDescriptor_t;
      samplerType : cudnnSamplerType_t;
      dataType : cudnnDataType_t;
      nbDims : int;
      dimA : access int) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:1424
   pragma Import (C, cudnnSetSpatialTransformerNdDescriptor, "cudnnSetSpatialTransformerNdDescriptor");

   function cudnnDestroySpatialTransformerDescriptor (stDesc : cudnnSpatialTransformerDescriptor_t) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:1431
   pragma Import (C, cudnnDestroySpatialTransformerDescriptor, "cudnnDestroySpatialTransformerDescriptor");

   function cudnnSpatialTfGridGeneratorForward
     (handle : cudnnHandle_t;
      stDesc : cudnnSpatialTransformerDescriptor_t;
      theta : System.Address;
      grid : System.Address) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:1434
   pragma Import (C, cudnnSpatialTfGridGeneratorForward, "cudnnSpatialTfGridGeneratorForward");

   function cudnnSpatialTfGridGeneratorBackward
     (handle : cudnnHandle_t;
      stDesc : cudnnSpatialTransformerDescriptor_t;
      dgrid : System.Address;
      dtheta : System.Address) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:1440
   pragma Import (C, cudnnSpatialTfGridGeneratorBackward, "cudnnSpatialTfGridGeneratorBackward");

   function cudnnSpatialTfSamplerForward
     (handle : cudnnHandle_t;
      stDesc : cudnnSpatialTransformerDescriptor_t;
      alpha : System.Address;
      xDesc : cudnnTensorDescriptor_t;
      x : System.Address;
      grid : System.Address;
      beta : System.Address;
      yDesc : cudnnTensorDescriptor_t;
      y : System.Address) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:1446
   pragma Import (C, cudnnSpatialTfSamplerForward, "cudnnSpatialTfSamplerForward");

   function cudnnSpatialTfSamplerBackward
     (handle : cudnnHandle_t;
      stDesc : cudnnSpatialTransformerDescriptor_t;
      alpha : System.Address;
      xDesc : cudnnTensorDescriptor_t;
      x : System.Address;
      beta : System.Address;
      dxDesc : cudnnTensorDescriptor_t;
      dx : System.Address;
      alphaDgrid : System.Address;
      dyDesc : cudnnTensorDescriptor_t;
      dy : System.Address;
      grid : System.Address;
      betaDgrid : System.Address;
      dgrid : System.Address) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:1457
   pragma Import (C, cudnnSpatialTfSamplerBackward, "cudnnSpatialTfSamplerBackward");

   --  skipped empty struct cudnnDropoutStruct

   type cudnnDropoutDescriptor_t is new System.Address;  -- /usr/local/cuda-8.0/include/cudnn.h:1473

   function cudnnCreateDropoutDescriptor (dropoutDesc : System.Address) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:1475
   pragma Import (C, cudnnCreateDropoutDescriptor, "cudnnCreateDropoutDescriptor");

   function cudnnDestroyDropoutDescriptor (dropoutDesc : cudnnDropoutDescriptor_t) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:1477
   pragma Import (C, cudnnDestroyDropoutDescriptor, "cudnnDestroyDropoutDescriptor");

  --helper function to determine size of the states to be passed to cudnnSetDropoutDescriptor  
   function cudnnDropoutGetStatesSize (handle : cudnnHandle_t; sizeInBytes : access stddef_h.size_t) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:1480
   pragma Import (C, cudnnDropoutGetStatesSize, "cudnnDropoutGetStatesSize");

  --helper function to determine size of the reserve space to be passed to dropout forward/backward calls  
   function cudnnDropoutGetReserveSpaceSize (xdesc : cudnnTensorDescriptor_t; sizeInBytes : access stddef_h.size_t) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:1483
   pragma Import (C, cudnnDropoutGetReserveSpaceSize, "cudnnDropoutGetReserveSpaceSize");

   function cudnnSetDropoutDescriptor
     (dropoutDesc : cudnnDropoutDescriptor_t;
      handle : cudnnHandle_t;
      dropout : float;
      states : System.Address;
      stateSizeInBytes : stddef_h.size_t;
      seed : Extensions.unsigned_long_long) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:1485
   pragma Import (C, cudnnSetDropoutDescriptor, "cudnnSetDropoutDescriptor");

  -- Restores the dropout descriptor to a previously saved-off state
   function cudnnRestoreDropoutDescriptor
     (dropoutDesc : cudnnDropoutDescriptor_t;
      handle : cudnnHandle_t;
      dropout : float;
      states : System.Address;
      stateSizeInBytes : stddef_h.size_t;
      seed : Extensions.unsigned_long_long) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:1493
   pragma Import (C, cudnnRestoreDropoutDescriptor, "cudnnRestoreDropoutDescriptor");

   function cudnnGetDropoutDescriptor
     (dropoutDesc : cudnnDropoutDescriptor_t;
      handle : cudnnHandle_t;
      dropout : access float;
      states : System.Address;
      seed : access Extensions.unsigned_long_long) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:1500
   pragma Import (C, cudnnGetDropoutDescriptor, "cudnnGetDropoutDescriptor");

   function cudnnDropoutForward
     (handle : cudnnHandle_t;
      dropoutDesc : cudnnDropoutDescriptor_t;
      xdesc : cudnnTensorDescriptor_t;
      x : System.Address;
      ydesc : cudnnTensorDescriptor_t;
      y : System.Address;
      reserveSpace : System.Address;
      reserveSpaceSizeInBytes : stddef_h.size_t) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:1506
   pragma Import (C, cudnnDropoutForward, "cudnnDropoutForward");

   function cudnnDropoutBackward
     (handle : cudnnHandle_t;
      dropoutDesc : cudnnDropoutDescriptor_t;
      dydesc : cudnnTensorDescriptor_t;
      dy : System.Address;
      dxdesc : cudnnTensorDescriptor_t;
      dx : System.Address;
      reserveSpace : System.Address;
      reserveSpaceSizeInBytes : stddef_h.size_t) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:1515
   pragma Import (C, cudnnDropoutBackward, "cudnnDropoutBackward");

  -- RNN API  
  -- Stock RNN with ReLu activation  
  -- Stock RNN with tanh activation  
  -- LSTM with no peephole connections  
  -- Using h' = tanh(r * Uh(t-1) + Wx) and h = (1 - z) * h' + z * h(t-1);  
   type cudnnRNNMode_t is 
     (CUDNN_RNN_RELU,
      CUDNN_RNN_TANH,
      CUDNN_LSTM,
      CUDNN_GRU);
   pragma Convention (C, cudnnRNNMode_t);  -- /usr/local/cuda-8.0/include/cudnn.h:1531

  -- Using output concatination at each step. Do we also want to support output sum?  
   type cudnnDirectionMode_t is 
     (CUDNN_UNIDIRECTIONAL,
      CUDNN_BIDIRECTIONAL);
   pragma Convention (C, cudnnDirectionMode_t);  -- /usr/local/cuda-8.0/include/cudnn.h:1537

   type cudnnRNNInputMode_t is 
     (CUDNN_LINEAR_INPUT,
      CUDNN_SKIP_INPUT);
   pragma Convention (C, cudnnRNNInputMode_t);  -- /usr/local/cuda-8.0/include/cudnn.h:1543

   type cudnnRNNAlgo_t is 
     (CUDNN_RNN_ALGO_STANDARD,
      CUDNN_RNN_ALGO_PERSIST_STATIC,
      CUDNN_RNN_ALGO_PERSIST_DYNAMIC);
   pragma Convention (C, cudnnRNNAlgo_t);  -- /usr/local/cuda-8.0/include/cudnn.h:1551

   --  skipped empty struct cudnnRNNStruct

   type cudnnRNNDescriptor_t is new System.Address;  -- /usr/local/cuda-8.0/include/cudnn.h:1554

   function cudnnCreateRNNDescriptor (rnnDesc : System.Address) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:1556
   pragma Import (C, cudnnCreateRNNDescriptor, "cudnnCreateRNNDescriptor");

   function cudnnDestroyRNNDescriptor (rnnDesc : cudnnRNNDescriptor_t) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:1557
   pragma Import (C, cudnnDestroyRNNDescriptor, "cudnnDestroyRNNDescriptor");

   --  skipped empty struct cudnnPersistentRNNPlan

   type cudnnPersistentRNNPlan_t is new System.Address;  -- /usr/local/cuda-8.0/include/cudnn.h:1560

  -- Expensive. Creates the plan for the specific settings.  
   function cudnnCreatePersistentRNNPlan
     (rnnDesc : cudnnRNNDescriptor_t;
      minibatch : int;
      dataType : cudnnDataType_t;
      plan : System.Address) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:1564
   pragma Import (C, cudnnCreatePersistentRNNPlan, "cudnnCreatePersistentRNNPlan");

  -- Attaches the plan to the descriptor.  
   function cudnnSetPersistentRNNPlan (rnnDesc : cudnnRNNDescriptor_t; plan : cudnnPersistentRNNPlan_t) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:1570
   pragma Import (C, cudnnSetPersistentRNNPlan, "cudnnSetPersistentRNNPlan");

   function cudnnDestroyPersistentRNNPlan (plan : cudnnPersistentRNNPlan_t) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:1573
   pragma Import (C, cudnnDestroyPersistentRNNPlan, "cudnnDestroyPersistentRNNPlan");

   function cudnnSetRNNDescriptor
     (handle : cudnnHandle_t;
      rnnDesc : cudnnRNNDescriptor_t;
      hiddenSize : int;
      numLayers : int;
      dropoutDesc : cudnnDropoutDescriptor_t;
      inputMode : cudnnRNNInputMode_t;
      direction : cudnnDirectionMode_t;
      mode : cudnnRNNMode_t;
      algo : cudnnRNNAlgo_t;
      dataType : cudnnDataType_t) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:1575
   pragma Import (C, cudnnSetRNNDescriptor, "cudnnSetRNNDescriptor");

  -- Between layers, not between recurrent steps.  
   function cudnnGetRNNDescriptor
     (cudnnHandle : cudnnHandle_t;
      rnnDesc : cudnnRNNDescriptor_t;
      hiddenSize : access int;
      numLayers : access int;
      dropoutDesc : System.Address;
      inputMode : access cudnnRNNInputMode_t;
      direction : access cudnnDirectionMode_t;
      mode : access cudnnRNNMode_t;
      algo : access cudnnRNNAlgo_t;
      dataType : access cudnnDataType_t) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:1586
   pragma Import (C, cudnnGetRNNDescriptor, "cudnnGetRNNDescriptor");

   function cudnnSetRNNMatrixMathType (desc : cudnnRNNDescriptor_t; math : cudnnMathType_t) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:1597
   pragma Import (C, cudnnSetRNNMatrixMathType, "cudnnSetRNNMatrixMathType");

  -- dataType in the RNN descriptor is used to determine math precision  
  -- dataType in weight descriptors and input descriptors is used to describe storage  
   function cudnnGetRNNWorkspaceSize
     (handle : cudnnHandle_t;
      rnnDesc : cudnnRNNDescriptor_t;
      seqLength : int;
      xDesc : System.Address;
      sizeInBytes : access stddef_h.size_t) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:1601
   pragma Import (C, cudnnGetRNNWorkspaceSize, "cudnnGetRNNWorkspaceSize");

   function cudnnGetRNNTrainingReserveSize
     (handle : cudnnHandle_t;
      rnnDesc : cudnnRNNDescriptor_t;
      seqLength : int;
      xDesc : System.Address;
      sizeInBytes : access stddef_h.size_t) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:1607
   pragma Import (C, cudnnGetRNNTrainingReserveSize, "cudnnGetRNNTrainingReserveSize");

   function cudnnGetRNNParamsSize
     (handle : cudnnHandle_t;
      rnnDesc : cudnnRNNDescriptor_t;
      xDesc : cudnnTensorDescriptor_t;
      sizeInBytes : access stddef_h.size_t;
      dataType : cudnnDataType_t) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:1614
   pragma Import (C, cudnnGetRNNParamsSize, "cudnnGetRNNParamsSize");

   function cudnnGetRNNLinLayerMatrixParams
     (handle : cudnnHandle_t;
      rnnDesc : cudnnRNNDescriptor_t;
      layer : int;
      xDesc : cudnnTensorDescriptor_t;
      wDesc : cudnnFilterDescriptor_t;
      w : System.Address;
      linLayerID : int;
      linLayerMatDesc : cudnnFilterDescriptor_t;
      linLayerMat : System.Address) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:1620
   pragma Import (C, cudnnGetRNNLinLayerMatrixParams, "cudnnGetRNNLinLayerMatrixParams");

   function cudnnGetRNNLinLayerBiasParams
     (handle : cudnnHandle_t;
      rnnDesc : cudnnRNNDescriptor_t;
      layer : int;
      xDesc : cudnnTensorDescriptor_t;
      wDesc : cudnnFilterDescriptor_t;
      w : System.Address;
      linLayerID : int;
      linLayerBiasDesc : cudnnFilterDescriptor_t;
      linLayerBias : System.Address) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:1630
   pragma Import (C, cudnnGetRNNLinLayerBiasParams, "cudnnGetRNNLinLayerBiasParams");

   function cudnnRNNForwardInference
     (handle : cudnnHandle_t;
      rnnDesc : cudnnRNNDescriptor_t;
      seqLength : int;
      xDesc : System.Address;
      x : System.Address;
      hxDesc : cudnnTensorDescriptor_t;
      hx : System.Address;
      cxDesc : cudnnTensorDescriptor_t;
      cx : System.Address;
      wDesc : cudnnFilterDescriptor_t;
      w : System.Address;
      yDesc : System.Address;
      y : System.Address;
      hyDesc : cudnnTensorDescriptor_t;
      hy : System.Address;
      cyDesc : cudnnTensorDescriptor_t;
      cy : System.Address;
      workspace : System.Address;
      workSpaceSizeInBytes : stddef_h.size_t) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:1640
   pragma Import (C, cudnnRNNForwardInference, "cudnnRNNForwardInference");

   function cudnnRNNForwardTraining
     (handle : cudnnHandle_t;
      rnnDesc : cudnnRNNDescriptor_t;
      seqLength : int;
      xDesc : System.Address;
      x : System.Address;
      hxDesc : cudnnTensorDescriptor_t;
      hx : System.Address;
      cxDesc : cudnnTensorDescriptor_t;
      cx : System.Address;
      wDesc : cudnnFilterDescriptor_t;
      w : System.Address;
      yDesc : System.Address;
      y : System.Address;
      hyDesc : cudnnTensorDescriptor_t;
      hy : System.Address;
      cyDesc : cudnnTensorDescriptor_t;
      cy : System.Address;
      workspace : System.Address;
      workSpaceSizeInBytes : stddef_h.size_t;
      reserveSpace : System.Address;
      reserveSpaceSizeInBytes : stddef_h.size_t) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:1660
   pragma Import (C, cudnnRNNForwardTraining, "cudnnRNNForwardTraining");

   function cudnnRNNBackwardData
     (handle : cudnnHandle_t;
      rnnDesc : cudnnRNNDescriptor_t;
      seqLength : int;
      yDesc : System.Address;
      y : System.Address;
      dyDesc : System.Address;
      dy : System.Address;
      dhyDesc : cudnnTensorDescriptor_t;
      dhy : System.Address;
      dcyDesc : cudnnTensorDescriptor_t;
      dcy : System.Address;
      wDesc : cudnnFilterDescriptor_t;
      w : System.Address;
      hxDesc : cudnnTensorDescriptor_t;
      hx : System.Address;
      cxDesc : cudnnTensorDescriptor_t;
      cx : System.Address;
      dxDesc : System.Address;
      dx : System.Address;
      dhxDesc : cudnnTensorDescriptor_t;
      dhx : System.Address;
      dcxDesc : cudnnTensorDescriptor_t;
      dcx : System.Address;
      workspace : System.Address;
      workSpaceSizeInBytes : stddef_h.size_t;
      reserveSpace : System.Address;
      reserveSpaceSizeInBytes : stddef_h.size_t) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:1682
   pragma Import (C, cudnnRNNBackwardData, "cudnnRNNBackwardData");

   function cudnnRNNBackwardWeights
     (handle : cudnnHandle_t;
      rnnDesc : cudnnRNNDescriptor_t;
      seqLength : int;
      xDesc : System.Address;
      x : System.Address;
      hxDesc : cudnnTensorDescriptor_t;
      hx : System.Address;
      yDesc : System.Address;
      y : System.Address;
      workspace : System.Address;
      workSpaceSizeInBytes : stddef_h.size_t;
      dwDesc : cudnnFilterDescriptor_t;
      dw : System.Address;
      reserveSpace : System.Address;
      reserveSpaceSizeInBytes : stddef_h.size_t) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:1711
   pragma Import (C, cudnnRNNBackwardWeights, "cudnnRNNBackwardWeights");

   type cudnnCTCLossAlgo_t is 
     (CUDNN_CTC_LOSS_ALGO_DETERMINISTIC,
      CUDNN_CTC_LOSS_ALGO_NON_DETERMINISTIC);
   pragma Convention (C, cudnnCTCLossAlgo_t);  -- /usr/local/cuda-8.0/include/cudnn.h:1731

  -- 
  --* Create an instance of a CTC (Connectionist Temporal Classification) loss descriptor
  -- 

   function cudnnCreateCTCLossDescriptor (ctcLossDesc : System.Address) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:1736
   pragma Import (C, cudnnCreateCTCLossDescriptor, "cudnnCreateCTCLossDescriptor");

   function cudnnSetCTCLossDescriptor (ctcLossDesc : cudnnCTCLossDescriptor_t; compType : cudnnDataType_t) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:1738
   pragma Import (C, cudnnSetCTCLossDescriptor, "cudnnSetCTCLossDescriptor");

   function cudnnGetCTCLossDescriptor (ctcLossDesc : cudnnCTCLossDescriptor_t; compType : access cudnnDataType_t) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:1742
   pragma Import (C, cudnnGetCTCLossDescriptor, "cudnnGetCTCLossDescriptor");

   function cudnnDestroyCTCLossDescriptor (ctcLossDesc : cudnnCTCLossDescriptor_t) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:1746
   pragma Import (C, cudnnDestroyCTCLossDescriptor, "cudnnDestroyCTCLossDescriptor");

  -- return the ctc costs and gradients, given the probabilities and labels  
   function cudnnCTCLoss
     (handle : cudnnHandle_t;
      probsDesc : cudnnTensorDescriptor_t;
      probs : System.Address;
      labels : access int;
      labelLengths : access int;
      inputLengths : access int;
      costs : System.Address;
      gradientsDesc : cudnnTensorDescriptor_t;
      gradients : System.Address;
      algo : cudnnCTCLossAlgo_t;
      ctcLossDesc : cudnnCTCLossDescriptor_t;
      workspace : System.Address;
      workSpaceSizeInBytes : stddef_h.size_t) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:1749
   pragma Import (C, cudnnCTCLoss, "cudnnCTCLoss");

  -- Tensor descriptor for probabilities, the dimensions are T,N,A (T is the timing steps, N is the mini batch size, A is the alphabet size)   
  -- probabilities after softmax, in GPU memory  
  -- labels, in CPU memory  
  -- the length of each label, in CPU memory  
  -- the lengths of timing steps in each batch, in CPU memory  
  -- the returned costs of CTC, in GPU memory  
  -- Tensor descriptor for gradients, the dimensions are T,N,A  
  -- the returned CTC gradients, in GPU memory, to compute costs only, set it to NULL  
  -- algorithm selected, supported now 0 and 1  
  -- pointer to the workspace, in GPU memory  
  -- the workspace size needed  
  -- return the workspace size needed for ctc  
   function cudnnGetCTCLossWorkspaceSize
     (handle : cudnnHandle_t;
      probsDesc : cudnnTensorDescriptor_t;
      gradientsDesc : cudnnTensorDescriptor_t;
      labels : access int;
      labelLengths : access int;
      inputLengths : access int;
      algo : cudnnCTCLossAlgo_t;
      ctcLossDesc : cudnnCTCLossDescriptor_t;
      sizeInBytes : access stddef_h.size_t) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:1764
   pragma Import (C, cudnnGetCTCLossWorkspaceSize, "cudnnGetCTCLossWorkspaceSize");

  -- Tensor descriptor for probabilities, the dimensions are T,N,A (T is the timing steps, N is the mini batch size, A is the alphabet size)  
  -- Tensor descriptor for gradients, the dimensions are T,N,A. To compute costs only, set it to NULL  
  -- labels, in CPU memory  
  -- the length of each label, in CPU memory  
  -- the lengths of timing steps in each batch, in CPU memory  
  -- algorithm selected, supported now 0 and 1  
  -- pointer to the returned workspace size  
  -- DEPRECATED routines to be removed next release : 
  --   User should use the non-suffixed version (which has the API and functionality of _v6 version)
  --   Routines with _v5 suffix has the functionality of the non-suffixed routines in the CUDNN V6
  --  

   function cudnnSetRNNDescriptor_v6
     (handle : cudnnHandle_t;
      rnnDesc : cudnnRNNDescriptor_t;
      hiddenSize : int;
      numLayers : int;
      dropoutDesc : cudnnDropoutDescriptor_t;
      inputMode : cudnnRNNInputMode_t;
      direction : cudnnDirectionMode_t;
      mode : cudnnRNNMode_t;
      algo : cudnnRNNAlgo_t;
      dataType : cudnnDataType_t) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:1781
   pragma Import (C, cudnnSetRNNDescriptor_v6, "cudnnSetRNNDescriptor_v6");

  -- Between layers, not between recurrent steps.  
   function cudnnSetRNNDescriptor_v5
     (rnnDesc : cudnnRNNDescriptor_t;
      hiddenSize : int;
      numLayers : int;
      dropoutDesc : cudnnDropoutDescriptor_t;
      inputMode : cudnnRNNInputMode_t;
      direction : cudnnDirectionMode_t;
      mode : cudnnRNNMode_t;
      dataType : cudnnDataType_t) return cudnnStatus_t;  -- /usr/local/cuda-8.0/include/cudnn.h:1792
   pragma Import (C, cudnnSetRNNDescriptor_v5, "cudnnSetRNNDescriptor_v5");

  -- Between layers, not between recurrent steps.  
end cudnn_h;

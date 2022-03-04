with tensors; use tensors;
with System;
with Ada.Containers.Indefinite_Vectors;
with solvers; use solvers;
with Ada.Streams.Stream_IO; use Ada.Streams.Stream_IO;

package cuda.cudnn is

   type execMode_t is (TRAIN, RUN);

   execMode : execMode_t := RUN;

   cudnnHandle : cudnn_h.cudnnHandle_t;
   cublasHandle : cublas_api_h.cublasHandle_t;

   type padding_t is (SAME,VALID);

   procedure createhandles;
   procedure destroyhandles;

    -----------------------------------------------------------------------------

   type layer_t is abstract tagged record
      x,y,dx,dy : tensor_t;
   end record;
   procedure init(self : in out layer_t; x : tensor_t) is abstract;
   procedure ibwd(self : in out layer_t; dy : tensor_t) is abstract;
   procedure free(self : in out layer_t) is abstract;
   procedure fwd(self : in out layer_t) is abstract;
   procedure bwd(self : in out layer_t) is abstract;
   procedure upd(self : in out layer_t) is abstract;
   procedure save(self : in out layer_t; stream : Stream_Access) is abstract;
   procedure load(self : in out layer_t; stream : Stream_Access) is abstract;
   procedure dsc(self : layer_t);

   package layer_pkg is new Ada.Containers.Indefinite_Vectors(Positive, layer_t'class);

   -----------------------------------------------------------------------------

   type sequential_t is tagged record
      layers : layer_pkg.Vector;
      l : Float := 0.0;
      a : Float := 0.0;
      t : Float := 0.0;
   end record;
   procedure add(self : in out sequential_t; lyr : layer_t'class);
   procedure init(self : in out sequential_t; x,dy : tensor_t);
   procedure free(self : in out sequential_t);
   procedure fwd(self : in out sequential_t);
   procedure bwd(self : in out sequential_t);
   procedure upd(self : in out sequential_t);
   procedure save (self : in out sequential_t; file : in out File_Type);
   procedure load(self : in out sequential_t; file : in out File_Type);
   procedure dsc(self : sequential_t);
   function loss(self : sequential_t) return float;
   function accuracy(self : sequential_t) return float;

   -----------------------------------------------------------------------------
   subtype ConvolutionMode_t is cudnn_h.cudnnConvolutionMode_t;
   CONVOLUTION : ConvolutionMode_t renames cudnn_h.CUDNN_CONVOLUTION;
   CROSS_CORRELATION : ConvolutionMode_t renames cudnn_h.CUDNN_CROSS_CORRELATION;

   type convolution_t is new layer_t with record
      mode : cudnn_h.cudnnConvolutionMode_t;
      desc : cudnn_h.cudnnConvolutionDescriptor_t;
      k,h,w,stride_h,stride_w : Positive;
      dilation_h,dilation_w : Positive;
      pad_h,pad_w : Integer;
      f,df: tensor_t;
      b,db : tensor_t;
      algo : aliased cudnn_h.cudnnConvolutionFwdAlgo_t;
      bwd_filter_algo : aliased cudnn_h.cudnnConvolutionBwdFilterAlgo_t;
      bwd_data_algo : aliased cudnn_h.cudnnConvolutionBwdDataAlgo_t;
      ws_size, ws_bwd_filter_size, ws_bwd_data_size : aliased stddef_h.size_t := 0;
      d_ws_address, d_ws_bwd_filter_address, d_ws_bwd_data_address : System.Address;
      sf,sb : adam_t;
   end record;
   procedure init(self : in out convolution_t; x : tensor_t);
   procedure ibwd(self : in out convolution_t; dy : tensor_t);
   procedure free(self : in out convolution_t);
   procedure fwd(self : in out convolution_t);
   procedure bwd(self : in out convolution_t);
   procedure upd(self : in out convolution_t);
   procedure save(self : in out convolution_t; stream : Stream_Access);
   procedure load(self : in out convolution_t; stream : Stream_Access);

   -----------------------------------------------------------------------------
   subtype PoolingMode_t is cudnn_h.cudnnPoolingMode_t;
   MAX_POOL : PoolingMode_t renames cudnn_h.CUDNN_POOLING_MAX;
   AVG_INC_PAD : PoolingMode_t renames cudnn_h.CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
   AVG_EXC_PAD : PoolingMode_t renames cudnn_h.CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
   MAX_DTRM : PoolingMode_t renames cudnn_h.CUDNN_POOLING_MAX_DETERMINISTIC;

   type pooling_t is new layer_t with record
      mode : cudnn_h.cudnnPoolingMode_t;
      desc : cudnn_h.cudnnPoolingDescriptor_t;
      h,w,stride_h,stride_w : Positive;
      pad_h,pad_w : Integer;
   end record;
   procedure init(self : in out pooling_t; x : tensor_t);
   procedure ibwd(self : in out pooling_t; dy : tensor_t);
   procedure free(self : in out pooling_t);
   procedure fwd(self : in out pooling_t);
   procedure bwd(self : in out pooling_t);
   procedure upd(self : in out pooling_t) is null;
   procedure save(self : in out pooling_t; stream : Stream_Access) is null;
   procedure load(self : in out pooling_t; stream : Stream_Access) is null;

   -----------------------------------------------------------------------------

   subtype activationMode_t is cudnn_h.cudnnActivationMode_t;
   SIGMOID : activationMode_t renames cudnn_h.CUDNN_ACTIVATION_SIGMOID;
   RELU : activationMode_t renames cudnn_h.CUDNN_ACTIVATION_RELU;
   TANH : activationMode_t renames cudnn_h.CUDNN_ACTIVATION_TANH;
   CLIPPED_RELU : activationMode_t renames cudnn_h.CUDNN_ACTIVATION_CLIPPED_RELU;
   ELU : activationMode_t renames cudnn_h.CUDNN_ACTIVATION_ELU;

   type activation_t is new layer_t with record
      mode : cudnn_h.cudnnActivationMode_t;
      desc : cudnn_h.cudnnActivationDescriptor_t;
      coef : float;
   end record;
   procedure init(self : in out activation_t; x : tensor_t);
   procedure ibwd(self : in out activation_t; dy : tensor_t);
   procedure free(self : in out activation_t);
   procedure fwd(self : in out activation_t);
   procedure bwd(self : in out activation_t);
   procedure upd(self : in out activation_t) is null;
   procedure save(self : in out activation_t; stream : Stream_Access) is null;
   procedure load(self : in out activation_t; stream : Stream_Access) is null;

   -----------------------------------------------------------------------------

   subtype batchNormMode_t is cudnn_h.cudnnBatchNormMode_t;
   PER_ACTIVATION : batchNormMode_t renames cudnn_h.CUDNN_BATCHNORM_PER_ACTIVATION;
   SPATIAL : batchNormMode_t renames cudnn_h.CUDNN_BATCHNORM_SPATIAL;
   SPATIAL_PERSISTENT : batchNormMode_t renames cudnn_h.CUDNN_BATCHNORM_SPATIAL_PERSISTENT;

   type batchnorm_t is new layer_t with record
      mode : cudnn_h.cudnnBatchNormMode_t;
      desc : cudnn_h.cudnnTensorDescriptor_t;
      f,df : tensor_t;
      b,db : tensor_t;
      rolling_mean,rolling_variance,saved_mean,saved_variance : tensor_t;
      exp_avg_factor : float;
      sf,sb : adam_t;
   end record;
   procedure init(self : in out batchnorm_t; x : tensor_t);
   procedure ibwd(self : in out batchnorm_t; dy : tensor_t);
   procedure free(self : in out batchnorm_t);
   procedure fwd(self : in out batchnorm_t);
   procedure bwd(self : in out batchnorm_t);
   procedure upd(self : in out batchnorm_t);
   procedure save(self : in out batchnorm_t; stream : Stream_Access);
   procedure load(self : in out batchnorm_t; stream : Stream_Access);

   -----------------------------------------------------------------------------

   type dropout_t is new layer_t with record
      desc : cudnn_h.cudnnDropoutDescriptor_t;
      value : Float;
      state_size, reserve_size : aliased stddef_h.size_t;
      d_state, d_reserve : System.Address;
   end record;
   procedure init(self : in out dropout_t; x : tensor_t);
   procedure ibwd(self : in out dropout_t; dy : tensor_t);
   procedure free(self : in out dropout_t);
   procedure fwd(self : in out dropout_t);
   procedure bwd(self : in out dropout_t);
   procedure upd(self : in out dropout_t) is null;
   procedure save(self : in out dropout_t; stream : Stream_Access) is null;
   procedure load(self : in out dropout_t; stream : Stream_Access) is null;

   -----------------------------------------------------------------------------

   type flatten_t is new layer_t with null record;

   procedure init(self : in out flatten_t; x : tensor_t);
   procedure ibwd(self : in out flatten_t; dy : tensor_t);
   procedure free(self : in out flatten_t);
   procedure fwd(self : in out flatten_t);
   procedure bwd(self : in out flatten_t);
   procedure upd(self : in out flatten_t) is null;
   procedure save(self : in out flatten_t; stream : Stream_Access) is null;
   procedure load(self : in out flatten_t; stream : Stream_Access) is null;

   -----------------------------------------------------------------------------

   type fullyconnected_t is new layer_t with record
      k : Positive;
      f,df : tensor_t;
      b,db : tensor_t;
      o : tensor_t;
      sf,sb : adam_t;
   end record;
   procedure init(self : in out fullyconnected_t; x : tensor_t);
   procedure ibwd(self : in out fullyconnected_t; dy : tensor_t);
   procedure free(self : in out fullyconnected_t);
   procedure fwd(self : in out fullyconnected_t);
   procedure bwd(self : in out fullyconnected_t);
   procedure upd(self : in out fullyconnected_t);
   procedure save(self : in out fullyconnected_t; stream : Stream_Access);
   procedure load(self : in out fullyconnected_t; stream : Stream_Access);

   -----------------------------------------------------------------------------
   subtype SoftmaxMode_t is cudnn_h.cudnnSoftmaxMode_t;
   INSTANCE : SoftmaxMode_t renames cudnn_h.CUDNN_SOFTMAX_MODE_INSTANCE;
   CHANNEL : SoftmaxMode_t renames cudnn_h.CUDNN_SOFTMAX_MODE_CHANNEL;

   type softmax_t is new layer_t with record
      mode : cudnn_h.cudnnSoftmaxMode_t;
   end record;
   procedure init(self : in out softmax_t; x : tensor_t);
   procedure ibwd(self : in out softmax_t; dy : tensor_t);
   procedure free(self : in out softmax_t);
   procedure fwd(self : in out softmax_t);
   procedure bwd(self : in out softmax_t);
   procedure upd(self : in out softmax_t) is null;
   procedure save(self : in out softmax_t; stream : Stream_Access) is null;
   procedure load(self : in out softmax_t; stream : Stream_Access) is null;


   -----------------------------------------------------------------------------

   type crossentropyloss_t is new layer_t with null record;
   procedure init(self : in out crossentropyloss_t; x : tensor_t);
   procedure ibwd(self : in out crossentropyloss_t; dy : tensor_t);
   procedure free(self : in out crossentropyloss_t);
   procedure fwd(self : in out crossentropyloss_t);
   procedure bwd(self : in out crossentropyloss_t);
   procedure upd(self : in out crossentropyloss_t) is null;
   function loss(self : crossentropyloss_t) return Float;
   function accuracy(self : crossentropyloss_t) return Float;
   procedure save(self : in out crossentropyloss_t; stream : Stream_Access) is null;
   procedure load(self : in out crossentropyloss_t; stream : Stream_Access) is null;


   -----------------------------------------------------------------------------

   type identity_block_t is new layer_t with record
      conv1 : convolution_t;
      bn1 : batchnorm_t;
      act1 : activation_t;

      conv2 : convolution_t;
      bn2 : batchnorm_t;
      act2 : activation_t;

      conv3 : convolution_t;
      bn3 : batchnorm_t;

      act_l : activation_t;

   end record;
   procedure init(self : in out identity_block_t; x : tensor_t);
   procedure ibwd(self : in out identity_block_t; dy : tensor_t);
   procedure free(self : in out identity_block_t);
   procedure fwd(self : in out identity_block_t);
   procedure bwd(self : in out identity_block_t);
   procedure upd(self : in out identity_block_t);
   procedure save(self : in out identity_block_t; stream : Stream_Access);
   procedure load(self : in out identity_block_t; stream : Stream_Access);
   procedure dsc(self : identity_block_t);



   -----------------------------------------------------------------------------

   type convolutional_block_t is new layer_t with record
      conv1 : convolution_t;
      bn1 : batchnorm_t;
      act1 : activation_t;

      conv2 : convolution_t;
      bn2 : batchnorm_t;
      act2 : activation_t;

      conv3 : convolution_t;
      bn3 : batchnorm_t;

      conv_sc : convolution_t;
      bn_sc : batchnorm_t;

      act_l : activation_t;

   end record;
   procedure init(self : in out convolutional_block_t; x : tensor_t);
   procedure ibwd(self : in out convolutional_block_t; dy : tensor_t);
   procedure free(self : in out convolutional_block_t);
   procedure fwd(self : in out convolutional_block_t);
   procedure bwd(self : in out convolutional_block_t);
   procedure upd(self : in out convolutional_block_t);
   procedure save(self : in out convolutional_block_t; stream : Stream_Access);
   procedure load(self : in out convolutional_block_t; stream : Stream_Access);
   procedure dsc(self : convolutional_block_t);

end cuda.cudnn;

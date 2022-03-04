with Interfaces.c.Extensions;
with cuda.io;
with ada.Calendar.Conversions;
with ada.Strings.Unbounded; use Ada.Strings.Unbounded;

package body cuda.cudnn.make is

   function convolution(mode : ConvolutionMode_t;
                        k,h,w : Positive;
                        stride : Positive := 1;
                        dilation : Positive := 1;
                        padding : padding_t := VALID) return convolution_t is

   begin

      return self : convolution_t do
         self.mode := mode;
         self.k := k;
         self.h := h;
         self.w := w;
         self.stride_h := stride;
         self.stride_w := stride;
         case padding is
            when VALID =>
               self.pad_h := 0;
               self.pad_w := 0;
            when SAME =>
               self.pad_h := Integer(Float'Floor((Float(h) - 1.0) / 2.0));
               self.pad_w := Integer(Float'Floor((Float(w) - 1.0) / 2.0));
         end case;
         self.dilation_h := dilation;
         self.dilation_w := dilation;
      end return;
   end convolution;

   function pooling(mode : PoolingMode_t;
                    h,w : Positive := 2;
                    stride : Positive := 2;
                    padding : padding_t := VALID) return pooling_t is
   begin
      return self : pooling_t do
         self.mode := mode;
         self.h := h;
         self.w := w;
         self.stride_h := stride;
         self.stride_w := stride;
         case padding is
            when VALID =>
               self.pad_h := 0;
               self.pad_w := 0;
            when SAME =>
               self.pad_h := Integer(Float'Ceiling((Float(h) - 1.0) / 2.0));
               self.pad_w := Integer(Float'Ceiling((Float(w) - 1.0) / 2.0));
         end case;
      end return;
   end pooling;

   function activation(mode : activationMode_t;
                       coef : Float := 0.0) return activation_t is
   begin
      return self : activation_t do
         self.mode := mode;
         self.coef := coef;
      end return;
   end activation;

   function batchnorm(mode : batchNormMode_t;
                      exp_avg_factor : Float := 0.01) return batchnorm_t is
   begin
      return self : batchnorm_t do
         self.mode := mode;
         self.exp_avg_factor := exp_avg_factor;
      end return;
   end batchnorm;

   function dropout(value : Float) return dropout_t is
   begin
      return self : dropout_t do
         self.value := value;
      end return;
   end dropout;

   function flatten return flatten_t is
      self : flatten_t;
      pragma Warnings(off,self);
   begin
      return self;
   end flatten;

   function fullyconnected(k : Positive) return fullyconnected_t is
   begin
      return self : fullyconnected_t do
         self.k := k;
      end return;
   end fullyconnected;

   function softmax(mode : SoftmaxMode_t) return softmax_t is
   begin
      return self : softmax_t do
         self.mode := mode;
      end return;
   end softmax;

   function crossentropyloss return crossentropyloss_t is
      self : crossentropyloss_t;
      pragma Warnings(off,self);
   begin
      return self;
   end crossentropyloss;

   function identity_block(k1,k2,k3 : Positive) return identity_block_t is
   begin
      return self : identity_block_t do

         self.conv1 := convolution(CROSS_CORRELATION,k1,1,1);
         self.bn1 := batchnorm(PER_ACTIVATION);
         self.act1 := activation(RELU);

         self.conv2 := convolution(CROSS_CORRELATION,k2,3,3,padding=>SAME);
         self.bn2 := batchnorm(PER_ACTIVATION);
         self.act2 := activation(RELU);

         self.conv3 := convolution(CROSS_CORRELATION,k3,1,1);
         self.bn3 := batchnorm(PER_ACTIVATION);

         self.act_l := activation(RELU);

      end return;
   end identity_block;

   function convolutional_block(k1,k2,k3: Positive; stride : Positive := 1)  return convolutional_block_t is
   begin
      return self : convolutional_block_t do

         self.conv1 := convolution(CROSS_CORRELATION,k1,1,1,stride=>stride);
         self.bn1 := batchnorm(PER_ACTIVATION);
         self.act1 := activation(RELU);

         self.conv2 := convolution(CROSS_CORRELATION,k2,3,3,padding=>SAME);
         self.bn2 := batchnorm(PER_ACTIVATION);
         self.act2 := activation(RELU);

         self.conv3 := convolution(CROSS_CORRELATION,k3,1,1);
         self.bn3 := batchnorm(PER_ACTIVATION);

         self.conv_sc := convolution(CROSS_CORRELATION,k3,1,1,stride=>stride);
         self.bn_sc := batchnorm(PER_ACTIVATION);

         self.act_l := activation(RELU);

      end return;
   end convolutional_block;


end cuda.cudnn.make;

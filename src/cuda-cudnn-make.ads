package cuda.cudnn.make is

   function convolution(mode : ConvolutionMode_t;
                        k,h,w : Positive;
                        stride : Positive := 1;
                        dilation : Positive := 1;
                        padding : padding_t := VALID) return convolution_t;

   function pooling(mode : PoolingMode_t;
                    h,w : Positive := 2;
                    stride : Positive := 2;
                    padding : padding_t := VALID) return pooling_t;

   function activation(mode : activationMode_t;
                       coef : Float := 0.0) return activation_t;

   function batchnorm(mode : batchNormMode_t;
                      exp_avg_factor : Float := 0.01) return batchnorm_t;

   function dropout(value : Float) return dropout_t;

   function flatten return flatten_t;

   function fullyconnected(k : Positive) return fullyconnected_t;

   function softmax(mode : SoftmaxMode_t) return softmax_t;

   function crossentropyloss return crossentropyloss_t;

   function identity_block(k1,k2,k3 : Positive)  return identity_block_t;

   function convolutional_block(k1,k2,k3 : Positive)  return convolutional_block_t;

end cuda.cudnn.make;

with system.Address_To_Access_Conversions;
with ada.Numerics.Elementary_Functions; use Ada.Numerics.Elementary_Functions;
with cuda.io;
with ada.Calendar.Conversions;
with cuda.driver.kernels;
with Interfaces.c.Extensions;
with Ada.Assertions;
with Ada.Strings.Unbounded.Text_IO; use Ada.Strings.Unbounded.Text_IO;
with Ada.Text_IO;

package body cuda.cudnn is


   alpha : aliased float := 1.0;
   beta : aliased float := 0.0;

   package Address_To_Access is new System.Address_To_Access_Conversions(Object => float);

   procedure createhandles is
   begin
      checkCUDNN(cudnn_h.cudnnCreate(cudnnHandle'Address));
      checkCUBLAS(cublas_api_h.cublasCreate_v2(cublasHandle'Address));
   end createhandles;

   procedure destroyhandles is
   begin
      checkCUBLAS(cublas_api_h.cublasDestroy_v2(cublasHandle));
      checkCUDNN(cudnn_h.cudnnDestroy(cudnnHandle));
   end destroyhandles;

   procedure dsc(self : layer_t) is
   begin
      ada.Text_IO.Put("x:"); self.x.dsc;
      ada.Text_IO.Put("y:"); self.y.dsc;
      ada.Text_IO.Put("dy:"); self.dy.dsc;
      ada.Text_IO.Put("dx:"); self.dx.dsc;
      ada.Text_IO.Put_Line("-------------------------");
   end dsc;



   -----------------------------------------------------------------------------

   procedure add(self : in out sequential_t; lyr : layer_t'class) is
   begin
      self.layers.Append(lyr);
   end add;

   procedure init(self : in out sequential_t; x,dy : tensor_t) is
      use type layer_pkg.Cursor;
   begin

      for i in self.layers.Iterate loop
         if i = self.layers.First then
            self.layers(i).init(x);
         else
            self.layers(i).init(self.layers(layer_pkg.Previous(i)).y);
         end if;
      end loop;

      for i in reverse self.layers.Iterate loop
         if i = self.layers.Last then
            self.layers(i).ibwd(dy);
         else
            self.layers(i).ibwd(self.layers(layer_pkg.Next(i)).dx);
         end if;
      end loop;

   end init;

   procedure fwd(self : in out sequential_t) is
   begin
      for i of self.layers loop
         i.fwd;
      end loop;
   end fwd;

   procedure bwd(self : in out sequential_t) is
   begin
      for i of reverse self.layers loop
         i.bwd;
      end loop;
   end bwd;

   procedure upd(self : in out sequential_t) is
   begin
      for i of self.layers loop
         i.upd;
      end loop;
   end upd;

   procedure free(self : in out sequential_t) is
   begin
      for i of self.layers loop
         i.free;
      end loop;
   end free;

   procedure save(self : in out sequential_t; file : in out File_Type) is
   begin
      Reset(file,Out_File);
      Float'Write(Stream(file), self.a);
      Float'Write(Stream(file), self.l);
      for i of self.layers loop
         i.save(Stream(file));
      end loop;
   end save;

   procedure load(self : in out sequential_t; file : in out File_Type) is
   begin
      Reset(file,In_File);
      Float'Read(Stream(file), self.a);
      Float'Read(Stream(file), self.l);
      for i of self.layers loop
         i.load(Stream(file));
      end loop;
   end load;

   procedure dsc(self : sequential_t) is
   begin
      ada.Text_IO.Put_Line("-------------------------");
      for i of self.layers loop
         i.dsc;
      end loop;
   end dsc;

   function loss(self : sequential_t) return float is
   begin
      return crossentropyloss_t(self.layers(self.layers.Last_Index).Element.all).loss;
   end loss;

   function accuracy(self : sequential_t) return float is
   begin
      return crossentropyloss_t(self.layers(self.layers.Last_Index).Element.all).accuracy;
   end accuracy;
   -----------------------------------------------------------------------------

   procedure init(self : in out convolution_t; x : tensor_t) is
   begin

      self.x := x;

      self.f := rand(self.k, self.x.c, self.h, self.w);
      self.b := zeros(1, self.k);
      self.df := init(self.f.n, self.f.c, self.f.h, self.f.w);
      self.db := init(self.b.n, self.b.c);

      checkCUDNN(cudnn_h.cudnnCreateConvolutionDescriptor(self.desc'Address));

      checkCUDNN(cudnn_h.cudnnSetConvolution2dDescriptor(convDesc    => self.desc,
                                                         pad_h       => Interfaces.C.int(self.pad_h),
                                                         pad_w       => Interfaces.C.int(self.pad_w),
                                                         u           => Interfaces.C.int(self.stride_h),
                                                         v           => Interfaces.C.int(self.stride_w),
                                                         dilation_h  => Interfaces.C.int(self.dilation_h),
                                                         dilation_w  => Interfaces.C.int(self.dilation_w),
                                                         mode        => self.mode,
                                                         computeType => cudnn_h.CUDNN_DATA_FLOAT));

      declare
         n, c, h, w : aliased Interfaces.c.int;
      begin
         checkCUDNN(cudnn_h.cudnnGetConvolution2dForwardOutputDim(convDesc        => self.desc,
                                                                  inputTensorDesc => self.x.desc,
                                                                  filterDesc      => self.f.fdesc,
                                                                  n               => n'access,
                                                                  c               => c'access,
                                                                  h               => h'access,
                                                                  w               => w'access));

         self.y := init(Positive(n), Positive(c), Positive(h), Positive(w));
      end;



      checkCUDNN(cudnn_h.cudnnGetConvolutionForwardAlgorithm(handle             => cudnnHandle,
                                                             xDesc              => self.x.desc,
                                                             wDesc              => self.f.fdesc,
                                                             convDesc           => self.desc,
                                                             yDesc              => self.y.desc,
                                                             preference         => cudnn_h.CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                             memoryLimitInBytes => 0,
                                                             algo               => self.algo'Access));


      checkCUDNN(cudnn_h.cudnnGetConvolutionForwardWorkspaceSize(handle => cudnnHandle,
                                                                 xDesc       => self.x.desc,
                                                                 wDesc       => self.f.fdesc,
                                                                 convDesc    => self.desc,
                                                                 yDesc       => self.y.desc,
                                                                 algo        => self.algo,
                                                                 sizeInBytes => self.ws_size'Access));



      self.d_ws_address := cuda.io.malloc(self.ws_size);

      self.sf.Init(self.f, self.df);
      self.sb.Init(self.b, self.db);


   end init;

   procedure ibwd(self : in out convolution_t; dy : tensor_t) is
   begin
      self.dy := dy;
      self.dx := init(self.x.n, self.x.c, self.x.h, self.x.w);


      checkCUDNN(cudnn_h.cudnnGetConvolutionBackwardFilterAlgorithm(handle             => cudnnHandle,
                                                                    xDesc              => self.x.desc,
                                                                    dyDesc             => self.dy.desc,
                                                                    convDesc           => self.desc,
                                                                    dwDesc             => self.f.fdesc,
                                                                    preference         => cudnn_h.CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
                                                                    memoryLimitInBytes => 0,
                                                                    algo               => self.bwd_filter_algo'Access));

      checkCUDNN(cudnn_h.cudnnGetConvolutionBackwardDataAlgorithm(handle             => cudnnHandle,
                                                                  wDesc              => self.f.fdesc,
                                                                  dyDesc             => self.dy.desc,
                                                                  convDesc           => self.desc,
                                                                  dxDesc             => self.dx.desc,
                                                                  preference         => cudnn_h.CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
                                                                  memoryLimitInBytes => 0,
                                                                  algo               => self.bwd_data_algo'Access));

      checkCUDNN(cudnn_h.cudnnGetConvolutionBackwardFilterWorkspaceSize(handle      => cudnnHandle,
                                                                        xDesc       => self.x.desc,
                                                                        dyDesc      => self.dy.desc,
                                                                        convDesc    => self.desc,
                                                                        gradDesc    => self.df.fdesc,
                                                                        algo        => self.bwd_filter_algo,
                                                                        sizeInBytes => self.ws_bwd_filter_size'access));

      checkCUDNN(cudnn_h.cudnnGetConvolutionBackwardDataWorkspaceSize(handle      => cudnnHandle,
                                                                      wDesc       => self.f.fdesc,
                                                                      dyDesc      => self.dy.desc,
                                                                      convDesc    => self.desc,
                                                                      dxDesc      => self.dx.desc,
                                                                      algo        => self.bwd_data_algo,
                                                                      sizeInBytes => self.ws_bwd_data_size'access));


      self.d_ws_bwd_filter_address := cuda.io.malloc(self.ws_bwd_filter_size);
      self.d_ws_bwd_data_address := cuda.io.malloc(self.ws_bwd_data_size);



   end ibwd;

   procedure free(self : in out convolution_t) is
   begin
      self.y.free;
      self.f.free;
      self.b.free;
      self.dx.free;
      self.df.free;
      self.db.free;

      checkCUDART(cuda_runtime_api_h.cudaFree(self.d_ws_address));
      checkCUDART(cuda_runtime_api_h.cudaFree(self.d_ws_bwd_filter_address));
      checkCUDART(cuda_runtime_api_h.cudaFree(self.d_ws_bwd_data_address));

      checkCUDNN(cudnn_h.cudnnDestroyConvolutionDescriptor(self.desc));
   end free;

   procedure fwd(self : in out convolution_t) is
   begin

      checkCUDNN(cudnn_h.cudnnConvolutionForward(handle               => cudnnHandle,
                                                 alpha                => alpha'Address,
                                                 xDesc                => self.x.desc,
                                                 x                    => self.x.d_address,
                                                 wDesc                => self.f.fdesc,
                                                 w                    => self.f.d_address,
                                                 convDesc             => self.desc,
                                                 algo                 => self.algo,
                                                 workSpace            => self.d_ws_address,
                                                 workSpaceSizeInBytes => self.ws_size,
                                                 beta                 => beta'Address,
                                                 yDesc                => self.y.desc,
                                                 y                    => self.y.d_address));

      checkCUDNN(cudnn_h.cudnnAddTensor(handle => cudnnHandle,
                                        alpha  => alpha'Address,
                                        aDesc  => self.b.desc,
                                        A      => self.b.d_address,
                                        beta   => alpha'Address,
                                        cDesc  => self.y.desc,
                                        C      => self.y.d_address));

   end fwd;

   procedure bwd(self : in out convolution_t) is
   begin

      checkCUDNN(cudnn_h.cudnnConvolutionBackwardBias(handle => cudnnHandle,
                                                      alpha  => alpha'Address,
                                                      dyDesc => self.dy.desc,
                                                      dy     => self.dy.d_address,
                                                      beta   => alpha'Address,
                                                      dbDesc => self.db.desc,
                                                      db     => self.db.d_address));

      checkCUDNN(cudnn_h.cudnnConvolutionBackwardFilter(handle               => cudnnHandle,
                                                        alpha                => alpha'address,
                                                        xDesc                => self.x.desc,
                                                        x                    => self.x.d_address,
                                                        dyDesc               => self.dy.desc,
                                                        dy                   => self.dy.d_address,
                                                        convDesc             => self.desc,
                                                        algo                 => self.bwd_filter_algo,
                                                        workSpace            => self.d_ws_bwd_filter_address,
                                                        workSpaceSizeInBytes => self.ws_bwd_filter_size,
                                                        beta                 => alpha'Address,
                                                        dwDesc               => self.df.fdesc,
                                                        dw                   => self.df.d_address));

      checkCUDNN(cudnn_h.cudnnConvolutionBackwardData(handle               => cudnnHandle,
                                                      alpha                => alpha'address,
                                                      wDesc                => self.f.fdesc,
                                                      w                    => self.f.d_address,
                                                      dyDesc               => self.dy.desc,
                                                      dy                   => self.dy.d_address,
                                                      convDesc             => self.desc,
                                                      algo                 => self.bwd_data_algo,
                                                      workSpace            => self.d_ws_bwd_data_address,
                                                      workSpaceSizeInBytes => self.ws_bwd_data_size,
                                                      beta                 => beta'Address,
                                                      dxDesc               => self.dx.desc,
                                                      dx                   => self.dx.d_address));

   end bwd;

   procedure upd(self : in out convolution_t) is
   begin
      self.sf.step;
      self.sb.step;
   end upd;

   procedure save(self : in out convolution_t; stream : Stream_Access) is
   begin
      self.f.save(stream);
      self.b.save(stream);
   end save;

   procedure load(self : in out convolution_t; stream : Stream_Access) is
   begin
      self.f.load(stream);
      self.b.load(stream);
   end load;

   -----------------------------------------------------------------------------

   procedure init(self : in out pooling_t; x : tensor_t) is
   begin
      self.x := x;

      checkCUDNN(cudnn_h.cudnnCreatePoolingDescriptor(self.desc'Address));

      checkCUDNN(cudnn_h.cudnnSetPooling2dDescriptor(poolingDesc       => self.desc,
                                                     mode              => cudnn_h.CUDNN_POOLING_MAX,
                                                     maxpoolingNanOpt  => cudnn_h.CUDNN_PROPAGATE_NAN,
                                                     windowHeight      => Interfaces.C.int(self.h),
                                                     windowWidth       => Interfaces.C.int(self.w),
                                                     verticalPadding   => Interfaces.C.int(self.pad_h),
                                                     horizontalPadding => Interfaces.C.int(self.pad_w),
                                                     verticalStride    => Interfaces.C.int(self.stride_h),
                                                     horizontalStride  => Interfaces.C.int(self.stride_w)));

      declare
         n, c, h, w : aliased Interfaces.c.int;
      begin

         checkCUDNN(cudnn_h.cudnnGetPooling2dForwardOutputDim(poolingDesc     => self.desc,
                                                              inputTensorDesc => self.x.desc,
                                                              n               => n'Access,
                                                              c               => c'Access,
                                                              h               => h'Access,
                                                              w               => w'Access));

         self.y := init(Positive(n), Positive(c), Positive(h), Positive(w));
      end;

   end init;

   procedure ibwd(self : in out pooling_t; dy : tensor_t) is
   begin
      self.dy := dy;
      self.dx := init(self.x.n, self.x.c, self.x.h, self.x.w);

   end ibwd;

   procedure free(self : in out pooling_t) is
   begin
      self.y.free;
      self.dx.free;

      checkCUDNN(cudnn_h.cudnnDestroyPoolingDescriptor(self.desc));
   end free;

   procedure fwd(self : in out pooling_t) is
   begin

      checkCUDNN(cudnn_h.cudnnPoolingForward(handle      => cudnnHandle,
                                             poolingDesc => self.desc,
                                             alpha       => alpha'Address,
                                             xDesc       => self.x.desc,
                                             x           => self.x.d_address,
                                             beta        => beta'Address,
                                             yDesc       => self.y.desc,
                                             y           => self.y.d_address));

   end fwd;

   procedure bwd(self : in out pooling_t) is
   begin

      checkCUDNN(cudnn_h.cudnnPoolingBackward(handle      => cudnnHandle,
                                              poolingDesc => self.desc,
                                              alpha       => alpha'Address,
                                              yDesc       => self.y.desc,
                                              y           => self.y.d_address,
                                              dyDesc      => self.y.desc,
                                              dy          => self.dy.d_address,
                                              xDesc       => self.x.desc,
                                              x           => self.x.d_address,
                                              beta        => beta'Address,
                                              dxDesc      => self.x.desc,
                                              dx          => self.dx.d_address));

   end bwd;

   -----------------------------------------------------------------------------

   procedure init(self : in out activation_t; x : tensor_t) is
   begin

      self.x := x;

      self.y := init(self.x.n, self.x.c ,self.x.h ,self.x.w);

      checkCUDNN(cudnn_h.cudnnCreateActivationDescriptor(self.desc'Address));

      checkCUDNN(cudnn_h.cudnnSetActivationDescriptor(activationDesc => self.desc,
                                                      mode           => self.mode,
                                                      reluNanOpt     => cudnn_h.CUDNN_PROPAGATE_NAN,
                                                      coef           => Interfaces.c.double(self.coef)));

   end init;

   procedure ibwd(self : in out activation_t; dy : tensor_t) is
   begin
      self.dy := dy;
      self.dx := init(self.x.n, self.x.c, self.x.h, self.x.w);
   end ibwd;

   procedure free(self : in out activation_t) is
   begin
      self.y.free;
      self.dx.free;

      checkCUDNN(cudnn_h.cudnnDestroyActivationDescriptor(self.desc));
   end free;

   procedure fwd(self : in out activation_t) is
   begin

      checkCUDNN(cudnn_h.cudnnActivationForward(handle         => cudnnHandle,
                                                activationDesc => self.desc,
                                                alpha          => alpha'Address,
                                                xDesc          => self.x.desc,
                                                x              => self.x.d_address,
                                                beta           => beta'Address,
                                                yDesc          => self.y.desc,
                                                y              => self.y.d_address));

   end fwd;

   procedure bwd(self : in out activation_t) is
   begin

      checkCUDNN(cudnn_h.cudnnActivationBackward(handle         => cudnnHandle,
                                                 activationDesc => self.desc,
                                                 alpha          => alpha'address,
                                                 yDesc          => self.y.desc,
                                                 y              => self.y.d_address,
                                                 dyDesc         => self.y.desc,
                                                 dy             => self.dy.d_address,
                                                 xDesc          => self.x.desc,
                                                 x              => self.x.d_address,
                                                 beta           => beta'address,
                                                 dxDesc         => self.x.desc,
                                                 dx             => self.dx.d_address));
   end bwd;

   -----------------------------------------------------------------------------

   procedure init(self : in out batchnorm_t; x : tensor_t) is
   begin
      self.x := x;

      self.y := init(self.x.n, self.x.c, self.x.h, self.x.w);

      checkCUDNN(cudnn_h.cudnnCreateTensorDescriptor(self.desc'Address));

      checkCUDNN(cudnn_h.cudnnDeriveBNTensorDescriptor(derivedBnDesc => self.desc,
                                                       xDesc         => self.x.desc,
                                                       mode          => self.mode));


      declare
         n, c, h, w, ns, cs, hs, ws : aliased Interfaces.c.int;
         datatype : aliased cudnn_h.cudnnDataType_t := cudnn_h.CUDNN_DATA_FLOAT;
      begin


         checkCUDNN(cudnn_h.cudnnGetTensor4dDescriptor(tensorDesc => self.desc,
                                                       dataType   => datatype'Access,
                                                       n          => n'Access,
                                                       c          => c'Access,
                                                       h          => h'Access,
                                                       w          => w'Access,
                                                       nStride    => ns'Access,
                                                       cStride    => cs'Access,
                                                       hStride    => hs'Access,
                                                       wStride    => ws'Access));


         self.f := rand(Positive(n), Positive(c), Positive(h), Positive(w));
         self.b := zeros(Positive(n), Positive(c), Positive(h), Positive(w));

         self.df := init(Positive(n), Positive(c), Positive(h), Positive(w));
         self.db := init(Positive(n), Positive(c), Positive(h), Positive(w));

         self.rolling_mean := zeros(Positive(n), Positive(c), Positive(h), Positive(w));
         self.rolling_variance := ones(Positive(n), Positive(c), Positive(h), Positive(w));
         self.saved_mean := zeros(Positive(n), Positive(c), Positive(h), Positive(w));
         self.saved_variance := ones(Positive(n), Positive(c), Positive(h), Positive(w));

      end;


      self.sf.Init(self.f, self.df);
      self.sb.Init(self.b, self.db);

   end init;

   procedure ibwd(self : in out batchnorm_t; dy : tensor_t) is
   begin
      self.dy := dy;
      self.dx := init(self.x.n, self.x.c, self.x.h, self.x.w);
   end ibwd;

   procedure free(self : in out batchnorm_t) is
   begin
      self.y.free;
      self.dx.free;
      self.f.free;
      self.b.free;
      self.df.free;
      self.db.free;
      self.rolling_mean.free;
      self.rolling_variance.free;
      self.saved_mean.free;
      self.saved_variance.free;

      checkCUDNN(cudnn_h.cudnnDestroyTensorDescriptor(self.desc));
   end free;

   procedure fwd(self : in out batchnorm_t) is
   begin

      case execMode is
         when TRAIN =>

            checkCUDNN(cudnn_h.cudnnBatchNormalizationForwardTraining(handle                   => cudnnHandle,
                                                                      mode                     => self.mode,
                                                                      alpha                    => alpha'Address,
                                                                      beta                     => beta'Address,
                                                                      xDesc                    => self.x.desc,
                                                                      x                        => self.x.d_address,
                                                                      yDesc                    => self.y.desc,
                                                                      y                        => self.y.d_address,
                                                                      bnScaleBiasMeanVarDesc   => self.desc,
                                                                      bnScale                  => self.f.d_address,
                                                                      bnBias                   => self.b.d_address,
                                                                      exponentialAverageFactor => Interfaces.c.double(self.exp_avg_factor),
                                                                      resultRunningMean        => self.rolling_mean.d_address,
                                                                      resultRunningVariance    => self.rolling_variance.d_address,
                                                                      epsilon                  => cudnn_h.CUDNN_BN_MIN_EPSILON,
                                                                      resultSaveMean           => self.saved_mean.d_address,
                                                                      resultSaveInvVariance    => self.saved_variance.d_address));
         when RUN =>

            checkCUDNN(cudnn_h.cudnnBatchNormalizationForwardInference(handle                 => cudnnHandle,
                                                                       mode                   => self.mode,
                                                                       alpha                  => alpha'Address,
                                                                       beta                   => beta'Address,
                                                                       xDesc                  => self.x.desc,
                                                                       x                      => self.x.d_address,
                                                                       yDesc                  => self.y.desc,
                                                                       y                      => self.y.d_address,
                                                                       bnScaleBiasMeanVarDesc => self.desc,
                                                                       bnScale                => self.f.d_address,
                                                                       bnBias                 => self.b.d_address,
                                                                       estimatedMean          => self.rolling_mean.d_address,
                                                                       estimatedVariance      => self.rolling_variance.d_address,
                                                                       epsilon                => cudnn_h.CUDNN_BN_MIN_EPSILON));
      end case;
   end fwd;

   procedure bwd(self : in out batchnorm_t) is
   begin


      checkCUDNN(cudnn_h.cudnnBatchNormalizationBackward(handle           => cudnnHandle,
                                                         mode             => self.mode,
                                                         alphaDataDiff    => alpha'Address,
                                                         betaDataDiff     => beta'Address,
                                                         alphaParamDiff   => alpha'Address,
                                                         betaParamDiff    => beta'Address,
                                                         xDesc            => self.x.desc,
                                                         x                => self.x.d_address,
                                                         dyDesc           => self.y.desc,
                                                         dy               => self.dy.d_address,
                                                         dxDesc           => self.x.desc,
                                                         dx               => self.dx.d_address,
                                                         dBnScaleBiasDesc => self.desc,
                                                         bnScale          => self.f.d_address,
                                                         dBnScaleResult   => self.df.d_address,
                                                         dBnBiasResult    => self.db.d_address,
                                                         epsilon          => cudnn_h.CUDNN_BN_MIN_EPSILON,
                                                         savedMean        => self.saved_mean.d_address,
                                                         savedInvVariance => self.saved_variance.d_address));

   end bwd;

   procedure upd(self : in out batchnorm_t) is
   begin
      self.sf.step;
      self.sb.step;
   end upd;

   procedure save(self : in out batchnorm_t; stream : Stream_Access) is
   begin
      self.f.save(stream);
      self.b.save(stream);
      self.rolling_mean.save(stream);
      self.rolling_variance.save(stream);
   end save;

   procedure load(self : in out batchnorm_t; stream : Stream_Access) is
   begin
      self.f.load(stream);
      self.b.load(stream);
      self.rolling_mean.load(stream);
      self.rolling_variance.load(stream);
   end load;

   -----------------------------------------------------------------------------

   procedure init(self : in out dropout_t; x : tensor_t) is
   begin
      self.x := x;

      self.y := init(self.x.n, self.x.c, self.x.h, self.x.w);

      checkCUDNN(cudnn_h.cudnnCreateDropoutDescriptor(self.desc'Address));

      checkCUDNN(cudnn_h.cudnnDropoutGetStatesSize(handle      => cudnnHandle,
                                                   sizeInBytes => self.state_size'Access));

      checkCUDNN(cudnn_h.cudnnDropoutGetReserveSpaceSize(xdesc       => self.x.desc,
                                                         sizeInBytes => self.reserve_size'Access));

      self.d_state := cuda.io.malloc(self.state_size);
      self.d_reserve := cuda.io.malloc(self.reserve_size);

      checkCUDNN(cudnn_h.cudnnSetDropoutDescriptor(dropoutDesc      => self.desc,
                                                   handle           => cudnnHandle,
                                                   dropout          => self.value,
                                                   states           => self.d_state,
                                                   stateSizeInBytes => self.state_size,
                                                   seed             => Interfaces.c.Extensions.unsigned_long_long(
                                                     Ada.Calendar.Conversions.To_Unix_Time(
                                                       Ada.Calendar.Clock))));


   end init;

   procedure ibwd(self : in out dropout_t; dy : tensor_t) is
   begin
      self.dy := dy;
      self.dx := init(self.x.n, self.x.c, self.x.h, self.x.w);
   end ibwd;

   procedure free(self : in out dropout_t) is
   begin
      self.y.free;
      self.dx.free;

      checkCUDART(cuda_runtime_api_h.cudaFree(self.d_state));
      checkCUDART(cuda_runtime_api_h.cudaFree(self.d_reserve));

      checkCUDNN(cudnn_h.cudnnDestroyDropoutDescriptor(self.desc));
   end free;

   procedure fwd(self : in out dropout_t) is
   begin

      if execMode = TRAIN then

         checkCUDNN(cudnn_h.cudnnDropoutForward(handle                  => cudnnHandle,
                                                dropoutDesc             => self.desc,
                                                xdesc                   => self.x.desc,
                                                x                       => self.x.d_address,
                                                ydesc                   => self.y.desc,
                                                y                       => self.y.d_address,
                                                reserveSpace            => self.d_reserve,
                                                reserveSpaceSizeInBytes => self.reserve_size));
      elsif execMode = Run then

         self.y.copy(self.x);

      end if;

   end fwd;

   procedure bwd(self : in out dropout_t) is
   begin

      checkCUDNN(cudnn_h.cudnnDropoutBackward(handle                  => cudnnHandle,
                                              dropoutDesc             => self.desc,
                                              dydesc                  => self.dy.desc,
                                              dy                      => self.dy.d_address,
                                              dxdesc                  => self.dx.desc,
                                              dx                      => self.dx.d_address,
                                              reserveSpace            => self.d_reserve,
                                              reserveSpaceSizeInBytes => self.reserve_size));
   end bwd;

   -----------------------------------------------------------------------------

   procedure init(self : in out flatten_t; x : tensor_t) is
   begin
      self.x := x;
      self.y := init(self.x.n, self.x.c * self.x.h * self.x.w);
   end init;

   procedure ibwd(self : in out flatten_t; dy : tensor_t) is
   begin
      self.dy := dy;
      self.dx := init(self.x.n, self.x.c, self.x.h, self.x.w);
   end ibwd;

   procedure free(self : in out flatten_t) is
   begin
      self.y.free;
      self.dx.free;
   end free;

   procedure fwd(self : in out flatten_t) is
   begin
      self.x.memcopy(self.y);
   end fwd;

   procedure bwd(self : in out flatten_t) is
   begin
      self.dy.memcopy(self.dx);
   end bwd;

   -----------------------------------------------------------------------------

   procedure init(self : in out fullyconnected_t; x : tensor_t) is
   begin
      self.x := x;

      self.y := init(self.x.n, self.k);
      self.o := ones(1, self.x.n);

      self.f := rand(self.k, self.x.c);
      self.b := zeros(1, self.k);

      self.df := init(self.k, self.x.c);
      self.db := init(1, self.k);

      self.sf.Init(self.f, self.df);
      self.sb.Init(self.b, self.db);

   end init;

   procedure ibwd(self : in out fullyconnected_t; dy : tensor_t) is
   begin
      self.dy := dy;
      self.dx := init(self.x.n, self.x.c);
   end ibwd;

   procedure free(self : in out fullyconnected_t) is
   begin
      self.y.free;
      self.f.free;
      self.b.free;
      self.o.free;
      self.dx.free;
      self.df.free;
      self.db.free;
   end free;

   procedure fwd(self : in out fullyconnected_t) is
   begin


      checkCUBLAS(cublas_api_h.cublasSgemm_v2(handle => cublasHandle,
                                             transa => cublas_api_h.CUBLAS_OP_T,
                                             transb => cublas_api_h.CUBLAS_OP_N,
                                             m      => Interfaces.c.int(self.y.c), --5  outputs
                                             n      => Interfaces.c.int(self.x.n),  --2  batchSize
                                             k      => Interfaces.c.int(self.x.c), --3  inputs
                                             alpha  => alpha'Access,
                                             A      => Address_To_Access.To_Pointer(self.f.d_address),
                                             lda    => Interfaces.c.int(self.x.c),  --3 inputs
                                             B      => Address_To_Access.To_Pointer(self.x.d_address),
                                             ldb    => Interfaces.c.int(self.x.c),  --3  inputs
                                             beta   => beta'Access,
                                             C      => Address_To_Access.To_Pointer(self.y.d_address),
                                             ldc    => Interfaces.c.int(self.y.c))); --5 outputs

      checkCUBLAS(cublas_api_h.cublasSgemm_v2(handle => cublasHandle,
                                            transa => cublas_api_h.CUBLAS_OP_N,
                                            transb => cublas_api_h.CUBLAS_OP_N,
                                            m      => Interfaces.c.int(self.y.c), --5 outputs
                                            n      => Interfaces.c.int(self.x.n),  --2 batchSize
                                            k      => Interfaces.c.int(self.o.n), --1
                                            alpha  => alpha'Access,
                                            A      => Address_To_Access.To_Pointer(self.b.d_address),
                                            lda    => Interfaces.c.int(self.y.c), --5 outputs
                                            B      => Address_To_Access.To_Pointer(self.o.d_address),
                                            ldb    => Interfaces.c.int(self.o.n), --1
                                            beta   => alpha'Access,
                                            C      => Address_To_Access.To_Pointer(self.y.d_address),
                                              ldc    => Interfaces.c.int(self.y.c))); --5 outputs

   end fwd;

   procedure bwd(self : in out fullyconnected_t) is
   begin

      checkCUBLAS(cublas_api_h.cublasSgemm_v2(handle => cublasHandle,
                                              transa => cublas_api_h.CUBLAS_OP_N,
                                              transb => cublas_api_h.CUBLAS_OP_T,
                                              m      => Interfaces.c.int(self.x.c), --inputs
                                              n      => Interfaces.c.int(self.y.c), --outputs
                                              k      => Interfaces.c.int(self.x.n), --batchSize
                                              alpha  => alpha'Access,
                                              A      => Address_To_Access.To_Pointer(self.x.d_address),
                                              lda    => Interfaces.c.int(self.x.c), --input
                                              B      => Address_To_Access.To_Pointer(self.dy.d_address),
                                              ldb    => Interfaces.c.int(self.y.c), --outputs
                                              beta   => beta'Access,
                                              C      => Address_To_Access.To_Pointer(self.df.d_address),
                                              ldc    => Interfaces.c.int(self.x.c))); --inputs


      checkCUBLAS(cublas_api_h.cublasSgemv_v2(handle => cublasHandle,
                                              trans  => cublas_api_h.CUBLAS_OP_N,
                                              m      => Interfaces.c.int(self.y.c), --outputs
                                              n      => Interfaces.c.int(self.x.n), --batchSize
                                              alpha  => alpha'Access,
                                              A      => Address_To_Access.To_Pointer(self.dy.d_address),
                                              lda    => Interfaces.c.int(self.y.c), --outputs
                                              x      => Address_To_Access.To_Pointer(self.o.d_address),
                                              incx   => Interfaces.c.int(self.o.n), --1
                                              beta   => beta'Access,
                                              y      => Address_To_Access.To_Pointer(self.db.d_address),
                                              incy   => Interfaces.c.int(self.o.n))); --1

         checkCUBLAS(cublas_api_h.cublasSgemm_v2(handle => cublasHandle,
                                                transa => cublas_api_h.CUBLAS_OP_N,
                                                transb => cublas_api_h.CUBLAS_OP_N,
                                                m      => Interfaces.c.int(self.x.c), --inputs
                                                n      => Interfaces.c.int(self.x.n), --batchSize
                                                k      => Interfaces.c.int(self.y.c), --outputs
                                                alpha  => alpha'Access,
                                                A      => Address_To_Access.To_Pointer(self.f.d_address),
                                                lda    => Interfaces.c.int(self.x.c), --inputs
                                                B      => Address_To_Access.To_Pointer(self.dy.d_address),
                                                ldb    => Interfaces.c.int(self.y.c), --outputs
                                                beta   => beta'Access,
                                                C      => Address_To_Access.To_Pointer(self.dx.d_address),
                                                ldc    => Interfaces.c.int(self.x.c))); --inputs


   end bwd;

   procedure upd(self : in out fullyconnected_t) is
   begin
      self.sf.step;
      self.sb.step;
   end upd;

   procedure save(self : in out fullyconnected_t; stream : Stream_Access) is
   begin
      self.f.save(stream);
      self.b.save(stream);
   end save;

   procedure load(self : in out fullyconnected_t; stream : Stream_Access) is
   begin
      self.f.load(stream);
      self.b.load(stream);
   end load;

   -----------------------------------------------------------------------------

   procedure init(self : in out softmax_t; x : tensor_t) is
   begin
      self.x := x;
      self.y := init(self.x.n, self.x.c);

   end init;

   procedure ibwd(self : in out softmax_t; dy : tensor_t) is
   begin
      self.dy := dy;
      self.dx := init(self.x.n, self.x.c);

   end ibwd;

   procedure free(self : in out softmax_t) is
   begin
      self.y.free;
      self.dx.free;
   end free;

   procedure fwd(self : in out softmax_t) is
   begin

      checkCUDNN(cudnn_h.cudnnSoftmaxForward(handle => cudnnHandle,
                                             algo   => cudnn_h.CUDNN_SOFTMAX_ACCURATE,
                                             mode   => self.mode,
                                             alpha  => alpha'Address,
                                             xDesc  => self.x.desc,
                                             x      => self.x.d_address,
                                             beta   => beta'Address,
                                             yDesc  => self.y.desc,
                                             y      => self.y.d_address));

   end fwd;

   procedure bwd(self : in out softmax_t) is

   begin

      checkCUDNN(cudnn_h.cudnnSoftmaxBackward(handle => cudnnHandle,
                                              algo   => cudnn_h.CUDNN_SOFTMAX_ACCURATE,
                                              mode   => self.mode,
                                              alpha  => alpha'Address,
                                              yDesc  => self.y.desc,
                                              y      => self.y.d_address,
                                              dyDesc => self.dy.desc,
                                              dy     => self.dy.d_address,
                                              beta   => beta'Address,
                                              dxDesc => self.dx.desc,
                                              dx     => self.dx.d_address));
   end bwd;

   -----------------------------------------------------------------------------

   procedure init(self : in out crossentropyloss_t; x : tensor_t) is
   begin
      self.x := x;
      self.y := init(self.x.n, self.x.c, self.x.h, self.x.w);
   end init;

   procedure ibwd(self : in out crossentropyloss_t; dy : tensor_t) is
   begin
      self.dy := dy;
      self.dx := init(self.x.n, self.x.c, self.x.h, self.x.w);
   end ibwd;

   procedure free(self : in out crossentropyloss_t) is
   begin
      self.y.free;
      self.dx.free;
   end free;

   procedure fwd(self : in out crossentropyloss_t) is
      args : cuda.driver.arguments := (self.x.num'Address,
                                       self.x.d_address'Address,
                                       self.dy.d_address'Address,
                                       self.y.d_address'Address);

   begin

      cuda.driver.kernels.calc_loss.exec(self.x.num, args);
   end fwd;

   procedure bwd(self : in out crossentropyloss_t) is

      args : cuda.driver.arguments := (self.x.num'Address,
                                       self.x.d_address'Address,
                                       self.dy.d_address'Address,
                                       self.dx.d_address'Address);

   begin

      cuda.driver.kernels.calc_diff.exec(self.x.num, args);

   end bwd;

   function loss(self : crossentropyloss_t) return Float is
   begin
      return return_value : Float := 0.0 do
         self.y.to_host;
         for i of self.y.data.all loop
            Return_Value := Return_Value + i;
         end loop;
        Return_Value := Return_Value / Float(self.y.n);
      end return;
   end loss;

   function accuracy(self : crossentropyloss_t) return Float is
      matchcnt : Integer := 0;
   begin
      self.x.to_host;
      self.dy.to_host;
      for n in 1..self.x.n loop

         declare
            match : Boolean := False;
         begin


            for c in 1..self.x.c loop
               if self.dy.idx(n,c) = 1.0 and float'Rounding(self.x.idx(n,c)) = 1.0 then
                  match := True;
               end if;
            end loop;

            if match then
               matchcnt := matchcnt + 1;
            end if;

         end;

      end loop;

      return float(matchcnt)/float(self.x.n);

   end accuracy;

   -----------------------------------------------------------------------------

  procedure init(self : in out identity_block_t; x : tensor_t) is
   begin
      self.x := x;

      self.conv1.init(self.x);
      self.bn1.init(self.conv1.y);
      self.act1.init(self.bn1.y);

      self.conv2.init(self.act1.y);
      self.bn2.init(self.conv2.y);
      self.act2.init(self.bn2.y);

      self.conv3.init(self.act2.y);
      self.bn3.init(self.conv3.y);

      self.act_l.init(self.bn3.y);

      self.y := self.act_l.y;

   end init;

   procedure ibwd(self : in out identity_block_t; dy : tensor_t) is
   begin
      self.dy := dy;

      self.act_l.ibwd(self.dy);

      self.bn3.ibwd(self.act_l.dx);
      self.conv3.ibwd(self.bn3.dx);

      self.act2.ibwd(self.conv3.dx);
      self.bn2.ibwd(self.act2.dx);
      self.conv2.ibwd(self.bn2.dx);

      self.act1.ibwd(self.conv2.dx);
      self.bn1.ibwd(self.act1.dx);
      self.conv1.ibwd(self.bn1.dx);

      --self.dx := self.conv1.dx;
      self.dx := self.act_l.dx;

   end ibwd;

   procedure free(self : in out identity_block_t) is
   begin
      self.y.free;
      self.dx.free;
      self.conv1.free;
      self.bn1.free;
      self.act1.free;
      self.conv2.free;
      self.bn2.free;
      self.act2.free;
      self.conv3.free;
      self.bn3.free;
      self.act_l.free;
   end free;

   procedure fwd(self : in out identity_block_t) is
   begin

      self.conv1.fwd;
      self.bn1.fwd;
      self.act1.fwd;

      self.conv2.fwd;
      self.bn2.fwd;
      self.act2.fwd;

      self.conv3.fwd;
      self.bn3.fwd;

      self.bn3.y.add(self.x);

      self.act_l.fwd;
   end fwd;

   procedure bwd(self : in out identity_block_t) is
   begin
      self.act_l.bwd;

      self.bn3.bwd;
      self.conv3.bwd;

      self.act2.bwd;
      self.bn2.bwd;
      self.conv2.bwd;

      self.act1.bwd;
      self.bn1.bwd;
      self.conv1.bwd;

      --self.dx.add(self.act_l.dx);

   end bwd;

   procedure upd(self : in out identity_block_t) is
   begin
      self.act_l.upd;

      self.bn3.upd;
      self.conv3.upd;

      self.act2.upd;
      self.bn2.upd;
      self.conv2.upd;

      self.act1.upd;
      self.bn1.upd;
      self.conv1.upd;

   end upd;

   procedure save(self : in out identity_block_t; stream : Stream_Access) is
   begin
      self.conv1.save(stream);
      self.bn1.save(stream);
      self.act1.save(stream);

      self.conv2.save(stream);
      self.bn2.save(stream);
      self.act2.save(stream);

      self.conv3.save(stream);
      self.bn3.save(stream);

      self.act_l.save(stream);
   end save;

   procedure load(self : in out identity_block_t; stream : Stream_Access) is
   begin
      self.conv1.load(stream);
      self.bn1.load(stream);
      self.act1.load(stream);

      self.conv2.load(stream);
      self.bn2.load(stream);
      self.act2.load(stream);

      self.conv3.load(stream);
      self.bn3.load(stream);

      self.act_l.load(stream);
   end load;

   procedure dsc(self : identity_block_t) is
   begin
      self.conv1.dsc;
      self.bn1.dsc;
      self.act1.dsc;
      self.conv2.dsc;
      self.bn2.dsc;
      self.act2.dsc;
      self.conv3.dsc;
      self.bn3.dsc;
      self.act_l.dsc;
   end dsc;

 -----------------------------------------------------------------------------

   procedure init(self : in out convolutional_block_t; x : tensor_t) is
   begin
      self.x := x;

      self.conv1.init(self.x);
      self.bn1.init(self.conv1.y);
      self.act1.init(self.bn1.y);

      self.conv2.init(self.act1.y);
      self.bn2.init(self.conv2.y);
      self.act2.init(self.bn2.y);

      self.conv3.init(self.act2.y);
      self.bn3.init(self.conv3.y);

      self.conv_sc.init(self.x);
      self.bn_sc.init(self.conv_sc.y);

      self.act_l.init(self.bn3.y);

      self.y := self.act_l.y;

   end init;

   procedure ibwd(self : in out convolutional_block_t; dy : tensor_t) is
   begin
      self.dy := dy;

      self.act_l.ibwd(self.dy);

      self.bn_sc.ibwd(self.act_l.dx);
      self.conv_sc.ibwd(self.bn_sc.dx);

      self.bn3.ibwd(self.act_l.dx);
      self.conv3.ibwd(self.bn3.dx);

      self.act2.ibwd(self.conv3.dx);
      self.bn2.ibwd(self.act2.dx);
      self.conv2.ibwd(self.bn2.dx);

      self.act1.ibwd(self.conv2.dx);
      self.bn1.ibwd(self.act1.dx);
      self.conv1.ibwd(self.bn1.dx);

      --self.dx := self.conv1.dx;
      self.dx := self.conv_sc.dx;

   end ibwd;

   procedure free(self : in out convolutional_block_t) is
   begin
      self.y.free;
      self.dx.free;
      self.conv1.free;
      self.bn1.free;
      self.act1.free;
      self.conv2.free;
      self.bn2.free;
      self.act2.free;
      self.conv3.free;
      self.bn3.free;
      self.conv_sc.free;
      self.bn_sc.free;
      self.act_l.free;
   end free;

   procedure fwd(self : in out convolutional_block_t) is
   begin

      self.conv1.fwd;
      self.bn1.fwd;
      self.act1.fwd;

      self.conv2.fwd;
      self.bn2.fwd;
      self.act2.fwd;

      self.conv3.fwd;
      self.bn3.fwd;

      self.conv_sc.fwd;
      self.bn_sc.fwd;

      self.bn3.y.add(self.bn_sc.y);

      self.act_l.fwd;
   end fwd;

   procedure bwd(self : in out convolutional_block_t) is
   begin
      self.act_l.bwd;

      self.bn_sc.bwd;
      self.conv_sc.bwd;

      self.bn3.bwd;
      self.conv3.bwd;

      self.act2.bwd;
      self.bn2.bwd;
      self.conv2.bwd;

      self.act1.bwd;
      self.bn1.bwd;
      self.conv1.bwd;

      --self.dx.add(self.conv_sc.dx);

   end bwd;

   procedure upd(self : in out convolutional_block_t) is
   begin
      self.act_l.upd;

      self.bn_sc.upd;
      self.conv_sc.upd;

      self.bn3.upd;
      self.conv3.upd;

      self.act2.upd;
      self.bn2.upd;
      self.conv2.upd;

      self.act1.upd;
      self.bn1.upd;
      self.conv1.upd;

   end upd;

   procedure save(self : in out convolutional_block_t; stream : Stream_Access) is
   begin
      self.conv1.save(stream);
      self.bn1.save(stream);
      self.act1.save(stream);

      self.conv2.save(stream);
      self.bn2.save(stream);
      self.act2.save(stream);

      self.conv3.save(stream);
      self.bn3.save(stream);

      self.conv_sc.save(stream);
      self.bn_sc.save(stream);

      self.act_l.save(stream);
   end save;

   procedure load(self : in out convolutional_block_t; stream : Stream_Access) is
   begin
      self.conv1.load(stream);
      self.bn1.load(stream);
      self.act1.load(stream);

      self.conv2.load(stream);
      self.bn2.load(stream);
      self.act2.load(stream);

      self.conv3.load(stream);
      self.bn3.load(stream);

      self.conv_sc.save(stream);
      self.bn_sc.save(stream);

      self.act_l.load(stream);
   end load;

   procedure dsc(self : convolutional_block_t) is
   begin
      self.conv1.dsc;
      self.bn1.dsc;
      self.act1.dsc;
      self.conv2.dsc;
      self.bn2.dsc;
      self.act2.dsc;
      self.conv3.dsc;
      self.bn3.dsc;
      self.conv_sc.dsc;
      self.bn_sc.dsc;
      self.act_l.dsc;
   end dsc;

begin

   createhandles;

end cuda.cudnn;

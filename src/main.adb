with argparse;
with cuda.cudnn.make;
with gnat.Traceback.Symbolic;
with ada.Text_IO;
with ada.Float_Text_IO;
with data.cifar10.test;
with tensors.io;
with ada.Streams.Stream_IO;
with ADA.IO_EXCEPTIONS;
with Utils;

procedure Main is

   type categories is(airplane,automobile,bird,cat,deer,dog,frog,horse,ship,truck);
   for categories use (1,2,3,4,5,6,7,8,9,10);

   file : Ada.Streams.Stream_IO.File_Type;
   args : argparse.parser_t;

   net : cuda.cudnn.sequential_t;

   package cifar10 is new data.cifar10(64,64,3,10);
   package cifar10_test is new cifar10.test;
   training_set : cifar10.vector_t;
   validation_set : cifar10_test.test_vector_t;

   loss, accuracy : utils.vector_t;


begin

   args.parse;
   loss.init(args.epoches);
   accuracy.init(args.epoches);

  begin
      ada.Streams.Stream_IO.Open(File => File,
                                 Mode => ada.Streams.Stream_IO.Out_File,
                                 Name => "cache.bin");
   exception
      when ADA.IO_EXCEPTIONS.NAME_ERROR =>
         ada.Streams.Stream_IO.Create(File => File,
                                      Mode => ada.Streams.Stream_IO.Out_File,
                                      Name => "cache.bin");
         args.resume := False;
   end;


    net.add(cuda.cudnn.make.input(3,64,64));

   net.add(cuda.cudnn.make.convolution(cuda.cudnn.CROSS_CORRELATION,64,7,7,stride => 2));
   net.add(cuda.cudnn.make.batchnorm(cuda.cudnn.PER_ACTIVATION));
   net.add(cuda.cudnn.make.activation(cuda.cudnn.RELU));
   net.add(cuda.cudnn.make.pooling(cuda.cudnn.MAX_POOL,3,3,2));
   --net.add(cuda.cudnn.make.dropout("drop1",0.5));

   net.add(cuda.cudnn.make.convolutional_block(64,64,256,stride => 1));
   net.add(cuda.cudnn.make.identity_block(64,64,256));
   net.add(cuda.cudnn.make.identity_block(64,64,256));

   net.add(cuda.cudnn.make.convolutional_block(128,128,512));
   net.add(cuda.cudnn.make.identity_block(128,128,512));
   net.add(cuda.cudnn.make.identity_block(128,128,512));
   net.add(cuda.cudnn.make.identity_block(128,128,512));

   net.add(cuda.cudnn.make.convolutional_block(256,256,1024));
   net.add(cuda.cudnn.make.identity_block(256,256,1024));
   net.add(cuda.cudnn.make.identity_block(256,256,1024));
   net.add(cuda.cudnn.make.identity_block(256,256,1024));
   net.add(cuda.cudnn.make.identity_block(256,256,1024));
   net.add(cuda.cudnn.make.identity_block(256,256,1024));

   net.add(cuda.cudnn.make.convolutional_block(512,512,2048));
   net.add(cuda.cudnn.make.identity_block(512,512,2048));
   net.add(cuda.cudnn.make.identity_block(512,512,2048));


   net.add(cuda.cudnn.make.pooling(cuda.cudnn.AVG_INC_PAD,2,2,1));

   net.add(cuda.cudnn.make.flatten);
   net.add(cuda.cudnn.make.fullyconnected(10));
   net.add(cuda.cudnn.make.softmax(cuda.cudnn.INSTANCE));

   net.add(cuda.cudnn.make.crossentropyloss);
   net.init(args.batch_size);

   --net.dsc;


   if args.exec_mode = 'T' then

      training_set.init;
      cifar10_test.init(validation_set);

      training_set.mean_std(net.mean,net.std);


      if args.resume then
         net.load(file);
      end if;

      for i in 1..args.epoches loop

         begin
            training_set.shuffle;
            cuda.cudnn.execMode := cuda.cudnn.TRAIN;


            --while not training_set.eof loop
            for x in 1..100 loop
               training_set.read(net.input,net.label);


               net.fwd;
               net.bwd;
               net.upd;

               tensors.io.show(net.layers(2).y);

               ada.Text_IO.Put(('='));


            end loop;

            ada.Text_IO.put(i'img&" -> ");
            ada.Float_Text_IO.put(net.loss, exp=>0);
            ada.Text_IO.put(" - ");
            ada.Float_Text_IO.put(net.accuracy, exp=>0);

         end;

         ada.Text_IO.put(" -- ");

         begin
            validation_set.shuffle;
            cuda.cudnn.execMode := cuda.cudnn.RUN;
            --while not validation_set.eof loop
            for x in 1..100 loop

               validation_set.read(net.input,net.label);

               net.fwd;


            end loop;
            accuracy.add(net.accuracy);
            loss.add(net.loss);

            ada.Float_Text_IO.put(net.loss, exp=>0);
            ada.Text_IO.put(" - ");
            ada.Float_Text_IO.put(net.accuracy, exp=>0);
         end;



         if accuracy.idx(i) > net.max_accuracy then
            net.max_accuracy := accuracy.idx(i);
            ada.Text_IO.Put(" -- Savepoint");
            net.save(file);
         end if;

         ada.Text_IO.New_Line;

      end loop;

      ada.Streams.Stream_IO.Close(File => file);

      loss.plot(accuracy);
   else
      cuda.cudnn.execMode := cuda.cudnn.RUN;
      declare
         imgs : tensors.io.files_t := (1=> args.img_file);
      begin
         net.load(file);
         tensors.io.load(net.input.all, imgs);
         net.fwd;


         cuda.cudnn.softmax_t(net.layers(net.layers.Last_Index-1).Element.all).y.print;
         begin
            ada.Text_IO.Put_Line(categories'Enum_Val(cuda.cudnn.softmax_t(net.layers(net.layers.Last_Index-1).Element.all).y.one_hot_decode(1))'img);
         exception when Constraint_Error=>
               ada.Text_IO.Put_Line("UNKNOWN");
         end;

         tensors.io.show(cuda.cudnn.input_t(net.layers(net.layers.First_Index).Element.all).x);
         ada.Streams.Stream_IO.Close(File => file);
      end;
   end if;

exception
   when e : argparse.EXIT_FROM_COMMAND_LINE =>
      null;
   when e : others =>
      ada.Text_IO.Put_Line(gnat.Traceback.Symbolic.Symbolic_Traceback(e));
      raise;
end Main;

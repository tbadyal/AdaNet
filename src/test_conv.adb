with Ada.Text_IO;
with ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with gnat.Traceback.Symbolic;
with cuda.cudnn.make;
with tensors;
with storage;
with Ada.Integer_Text_IO;

procedure test_conv is
   img : storage.files_t := (To_Unbounded_String("../img/test1.png"),
                             To_Unbounded_String("../img/ironman.jpg"));
   x : tensors.tensor_t := tensors.init(2,3,64,64);
   conv : cuda.cudnn.convolution_t :=
     cuda.cudnn.make.convolution(cuda.cudnn.CROSS_CORRELATION,64,7,7,2, padding => cuda.cudnn.VALID);
begin
   conv.init(x);


   x.read(img);

   conv.fwd;

   conv.y.show;
exception
   when e : others =>
      ada.Text_IO.Put_Line(gnat.Traceback.Symbolic.Symbolic_Traceback(e));
      raise;

end test_conv;

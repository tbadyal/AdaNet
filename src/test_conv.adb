with Ada.Text_IO;
with ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with gnat.Traceback.Symbolic;
with cuda.cudnn.make;
with tensors;
with storage;
with Ada.Integer_Text_IO;

procedure test_conv is
   img : storage.files_t := (To_Unbounded_String("/home/tushar/gprprojects/adanet2/img/test1.png"),
                             To_Unbounded_String("/home/tushar/gprprojects/adanet2/img/ironman.jpg"));
   x : tensors.tensor_t := tensors.init(2,3,32,32);
   dy : tensors.tensor_t := tensors.rand(2,2352);
   conv : cuda.cudnn.convolution_t :=
     cuda.cudnn.make.convolution(cuda.cudnn.CROSS_CORRELATION,3,5,5,1, padding => cuda.cudnn.VALID);
   flat : cuda.cudnn.flatten_t := cuda.cudnn.make.flatten;
begin
   conv.init(x);
   flat.init(conv.y);
   flat.ibwd(dy);
   conv.ibwd(flat.dx);

   x.read(img);

   conv.fwd;

   flat.fwd;


   flat.bwd;

   flat.dx.show;
   conv.bwd;
   conv.dx.print;
exception
   when e : others =>
      ada.Text_IO.Put_Line(gnat.Traceback.Symbolic.Symbolic_Traceback(e));
      raise;

end test_conv;

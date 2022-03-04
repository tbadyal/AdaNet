with Interfaces.C;
with Core.Operations,
     Imgproc.Operations,
     Highgui;
with ada.Text_IO; use Ada.Text_IO;
with ada.Float_Text_IO; use ada.Float_Text_IO;


package body storage.io is

   procedure read(self : in out storage_t; files : files_t) is
      ln : Positive := self'Length(1);
      lc : Positive := self'Length(2);
      lh : Positive := self'Length(3);
      lw : Positive := self'Length(4);

      img, img1, img2 : aliased core.Cv_Mat_Ptr;
      arr : core.Cv_32f_Array(1..lc*lh*lw);

   begin

      for n in 1..ln loop

         img := Highgui.Cv_Load_Image_M(Filename => To_String(files(n)),
                                        Iscolor  => Highgui.Cv_Load_Image_Color);

         img1 := Core.Operations.Cv_Create_Mat(Rows     => lh,
                                               Cols     => lw,
                                               Mat_Type => core.Cv_Make_Type(Depth => core.Cv_8u,
                                                                             Cn    => lc));

         Imgproc.Operations.Cv_Resize(Src          => img,
                                      Dst          => img1,
                                      Interplation => Imgproc.Cv_Inter_Linear);


         img2 :=  core.Operations.Cv_Create_Mat(Rows     => lh,
                                                Cols     => lw,
                                                Mat_Type => core.Cv_Make_Type(Depth => core.Cv_32f,
                                                                              Cn    => lc));

         core.Operations.Cv_Normalize(Src      => img1,
                                      Dst      => img2,
                                      Normtype => integer(core.Cv_C));

         arr := core.Cv_32f_Pointer_Pkg.Value(img2.Data.Cv_32f, Interfaces.C.ptrdiff_t(arr'Last));


         for c in 1..lc loop
            for h in 1..lh loop
               for w in 1..lw loop
                  self(n,c,h,w) := arr( (h-1)*lw*lc + (w-1)*lc + c );
               end loop;
            end loop;
         end loop;

      end loop;

      core.Operations.Cv_Release_Mat(img'access);
      core.Operations.Cv_Release_Mat(img1'access);
      core.Operations.Cv_Release_Mat(img2'access);

   end read;


   procedure show(self : storage_t; msec : Integer := 0; winName : String := "AdaNet") is
      ln : Positive := self'Length(1);
      lc : Positive := self'Length(2);
      lh : Positive := self'Length(3);
      lw : Positive := self'Length(4);

      img : aliased core.Cv_Mat_Ptr;
      arr : aliased core.Cv_32f_Array(1..ln*lc*lh*lw);
      last_idx : Positive := 1;
   begin

      for n in 1..ln loop
         for h in 1..lh loop
            for w in 1..lw loop
               for c in 1..lc loop
                  arr(last_idx) := self(n,c,h,w);
                  last_idx := last_idx + 1;
               end loop;
            end loop;
         end loop;
      end loop;

      img :=  core.Operations.Cv_Create_Mat(Rows     => lh * ln,
                                            Cols     => lw,
                                            Mat_Type => core.Cv_Make_Type(Depth => core.Cv_32f,
                                                                          Cn    => lc));

      core.Cv_32f_Pointer_Pkg.Copy_Array(Source => arr(arr'First)'unrestricted_access,
                                         Target => img.Data.Cv_32f,
                                         Length => Interfaces.C.ptrdiff_t(arr'Length));

      Highgui.Cv_Named_Window(Windowname => winName,
                              Flags      => Highgui.Cv_Window_Autosize);

      Highgui.Cv_Show_Image(Windowname => winName,
                            Image      => img);

      if Highgui.Cv_Wait_Key(Ms_Delay => msec)=ASCII.esc then
         Highgui.Cv_Destroy_Window (winName);
      end if;


      core.Operations.Cv_Release_Mat(img'access);

   end show;


   procedure resize(self, oth : in out storage_t) is

      ln : Positive := self'Length(1);
      lc : Positive := self'Length(2);
      lh : Positive := self'Length(3);
      lw : Positive := self'Length(4);

      lno : Positive := oth'Length(1);
      lco : Positive := oth'Length(2);
      lho : Positive := oth'Length(3);
      lwo : Positive := oth'Length(4);

      img , imgo: aliased core.Cv_Mat_Ptr;
      arr : aliased core.Cv_32f_Array(1..lc*lh*lw);
      arro : aliased core.Cv_32f_Array(1..lco*lho*lwo);
      last_idx,last_idxo : Positive := 1;


   begin

      for n in 1..ln loop

         for h in 1..lh loop
            for w in 1..lw loop
               for c in 1..lc loop
                  arr(last_idx) := self(n,c,h,w);
                  last_idx := last_idx + 1;
               end loop;
            end loop;
         end loop;


         img :=  core.Operations.Cv_Create_Mat(Rows     => lh,
                                               Cols     => lw,
                                               Mat_Type => core.Cv_Make_Type(Depth => core.Cv_32f,
                                                                             Cn    => lc));

         core.Cv_32f_Pointer_Pkg.Copy_Array(Source => arr(arr'First)'unrestricted_access,
                                            Target => img.Data.Cv_32f,
                                            Length => Interfaces.C.ptrdiff_t(arr'Length));

         imgo := Core.Operations.Cv_Create_Mat(Rows     => lho,
                                               Cols     => lwo,
                                               Mat_Type => core.Cv_Make_Type(Depth => core.Cv_32f,
                                                                             Cn    => lco));

         Imgproc.Operations.Cv_Resize(Src          => img,
                                      Dst          => imgo,
                                      Interplation => Imgproc.Cv_Inter_Linear);

         arro := core.Cv_32f_Pointer_Pkg.Value(imgo.Data.Cv_32f, Interfaces.C.ptrdiff_t(arro'Last));

         for h in 1..lho loop
            for w in 1..lwo loop
               for c in 1..lco loop
                  oth(n,c,h,w) := arro(last_idxo);
                  last_idxo := last_idxo+1;
               end loop;
            end loop;
         end loop;

      end loop;


      core.Operations.Cv_Release_Mat(img'access);
      core.Operations.Cv_Release_Mat(imgo'access);


   end resize;

   procedure dsc(self : storage_t) is
   begin
      Put_Line(" n:" & self'Length(1)'img & " c:" & self'Length(2)'img & " h:" & self'Length(3)'img & " w:" & self'Length(4)'img);
   end dsc;

   procedure print(self : storage_t) is
      ln : Positive := self'Length(1);
      lc : Positive := self'Length(2);
      lh : Positive := self'Length(3);
      lw : Positive := self'Length(4);
   begin
      dsc(self);
      for n in 1..ln loop
         for c in 1..lc loop
            for h in 1..lh loop
               for w in 1..lw loop
                  Put(self(n,c,h,w), exp=>0);
                  Put(" ");
               end loop;
               Put(" ");
            end loop;
            Put(" ");
         end loop;
         New_Line;
      end loop;

   end print;


end storage.io;

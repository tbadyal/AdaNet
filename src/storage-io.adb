with Interfaces.C;
with Core.Operations,
     Imgproc.Operations,
     Highgui;
with ada.Text_IO; use Ada.Text_IO;
with ada.Float_Text_IO; use ada.Float_Text_IO;


package body storage.io is

   procedure read(self : in out storage_t; files : files_t) is

      img, img1, img2 : aliased core.Cv_Mat_Ptr;
      arr : core.Cv_32f_Array(1..self.c*self.h*self.w);

   begin

      for n in 1..self.n loop

         img := Highgui.Cv_Load_Image_M(Filename => To_String(files(n)),
                                        Iscolor  => Highgui.Cv_Load_Image_Color);

         img1 := Core.Operations.Cv_Create_Mat(Rows     => self.h,
                                               Cols     => self.w,
                                               Mat_Type => core.Cv_Make_Type(Depth => core.Cv_8u,
                                                                             Cn    => self.c));

         Imgproc.Operations.Cv_Resize(Src          => img,
                                      Dst          => img1,
                                      Interplation => Imgproc.Cv_Inter_Linear);


         img2 :=  core.Operations.Cv_Create_Mat(Rows     => self.h,
                                                Cols     => self.w,
                                                Mat_Type => core.Cv_Make_Type(Depth => core.Cv_32f,
                                                                              Cn    => self.c));

         core.Operations.Cv_Normalize(Src      => img1,
                                      Dst      => img2,
                                      Normtype => integer(core.Cv_C));

         arr := core.Cv_32f_Pointer_Pkg.Value(img2.Data.Cv_32f, Interfaces.C.ptrdiff_t(arr'Last));


         for c in 1..self.c loop
            for h in 1..self.h loop
               for w in 1..self.w loop
                  self.idx(n,c,h,w) := arr( (h-1)*self.w*self.c + (w-1)*self.c + c );
               end loop;
            end loop;
         end loop;

      end loop;

      core.Operations.Cv_Release_Mat(img'access);
      core.Operations.Cv_Release_Mat(img1'access);
      core.Operations.Cv_Release_Mat(img2'access);

   end read;


   procedure show(self : storage_t; msec : Integer := 0; winName : String := "AdaNet") is

      img : aliased core.Cv_Mat_Ptr;
      arr : aliased core.Cv_32f_Array(1..self.num);
      last_idx : Positive := 1;
   begin

      for n in 1..self.n loop
         for h in 1..self.h loop
            for w in 1..self.w loop
               for c in 1..self.c loop
                  arr(last_idx) := self.idx(n,c,h,w);
                  last_idx := last_idx + 1;
               end loop;
            end loop;
         end loop;
      end loop;

      img :=  core.Operations.Cv_Create_Mat(Rows     => self.h * self.n,
                                            Cols     => self.w,
                                            Mat_Type => core.Cv_Make_Type(Depth => core.Cv_32f,
                                                                          Cn    => self.c));

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

      img , imgo: aliased core.Cv_Mat_Ptr;
      arr : aliased core.Cv_32f_Array(1..self.c*self.h*self.w);
      arro : aliased core.Cv_32f_Array(1..oth.c*oth.h*oth.w);
      last_idx,last_idxo : Positive := 1;


   begin

      for n in 1..self.n loop

         for h in 1..self.h loop
            for w in 1..self.w loop
               for c in 1..self.c loop
                  arr(last_idx) := self.idx(n,c,h,w);
                  last_idx := last_idx + 1;
               end loop;
            end loop;
         end loop;


         img :=  core.Operations.Cv_Create_Mat(Rows     => self.h,
                                               Cols     => self.w,
                                               Mat_Type => core.Cv_Make_Type(Depth => core.Cv_32f,
                                                                             Cn    => self.c));

         core.Cv_32f_Pointer_Pkg.Copy_Array(Source => arr(arr'First)'unrestricted_access,
                                            Target => img.Data.Cv_32f,
                                            Length => Interfaces.C.ptrdiff_t(arr'Length));

         imgo := Core.Operations.Cv_Create_Mat(Rows     => oth.h,
                                               Cols     => oth.w,
                                               Mat_Type => core.Cv_Make_Type(Depth => core.Cv_32f,
                                                                             Cn    => oth.c));

         Imgproc.Operations.Cv_Resize(Src          => img,
                                      Dst          => imgo,
                                      Interplation => Imgproc.Cv_Inter_Linear);

         arro := core.Cv_32f_Pointer_Pkg.Value(imgo.Data.Cv_32f, Interfaces.C.ptrdiff_t(arro'Last));

         for h in 1..oth.h loop
            for w in 1..oth.w loop
               for c in 1..oth.c loop
                  oth.idx(n,c,h,w) := arro(last_idxo);
                  last_idxo := last_idxo+1;
               end loop;
            end loop;
         end loop;

      end loop;


      core.Operations.Cv_Release_Mat(img'access);
      core.Operations.Cv_Release_Mat(imgo'access);


   end resize;


end storage.io;

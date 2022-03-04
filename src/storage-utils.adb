with PLplot; use PLplot;
with PLplot_Auxiliary; use PLplot_Auxiliary;

package body storage.utils is

   procedure plot(x : storage_t) is
      x_l : Real_Vector(1..x.n*x.c*x.h*x.w);
   begin
      Initialize_PLplot;

      for i in 1..x.n*x.c*x.h*x.w loop
         x_l(i) := Long_Float(x.data(i));
      end loop;

      Quick_Plot(x_l,
                 X_Label => "x",
                 Y_Label => "y",
                 Title_Label => "AdaNet");

      End_PLplot;
   end plot;

   procedure plot(x,y : storage_t) is
      x_l : Real_Vector(1..x.n*x.c*x.h*x.w);
      y_l : Real_Vector(1..y.n*y.c*y.h*y.w);
   begin
      Initialize_PLplot;

      for i in 1..x.n*x.c*x.h*x.w loop
         x_l(i) := Long_Float(x.data(i));
      end loop;

      for i in 1..y.n*y.c*y.h*y.w loop
         y_l(i) := Long_Float(y.data(i));
      end loop;

      Quick_Plot(x_l,
                 y_l,
                 X_Label => "x",
                 Y_Label => "y",
                 Title_Label => "AdaNet");

      End_PLplot;
   end plot;


end storage.utils;

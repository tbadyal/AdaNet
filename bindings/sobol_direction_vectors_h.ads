pragma Ada_2005;
pragma Style_Checks (Off);

with Interfaces.C; use Interfaces.C;

package sobol_direction_vectors_h is

   SOBOL_D : constant := (20000);  --  /usr/local/cuda-8.0/include/sobol_direction_vectors.h:4
   SOBOL_L : constant := (64);  --  /usr/local/cuda-8.0/include/sobol_direction_vectors.h:5

   sobol_v32 : aliased array (0 .. 19999, 0 .. 31) of aliased unsigned;  -- /usr/local/cuda-8.0/include/sobol_direction_vectors.h:6
   pragma Import (CPP, sobol_v32, "_ZL9sobol_v32");

   scrambled_sobol_v32 : aliased array (0 .. 19999, 0 .. 31) of aliased unsigned;  -- /usr/local/cuda-8.0/include/sobol_direction_vectors.h:7
   pragma Import (CPP, scrambled_sobol_v32, "_ZL19scrambled_sobol_v32");

   scrambled_sobol_c32 : aliased array (0 .. 19999) of aliased unsigned;  -- /usr/local/cuda-8.0/include/sobol_direction_vectors.h:8
   pragma Import (CPP, scrambled_sobol_c32, "_ZL19scrambled_sobol_c32");

   sobol_v_host : aliased array (0 .. 19999, 0 .. 127) of aliased unsigned;  -- /usr/local/cuda-8.0/include/sobol_direction_vectors.h:9
   pragma Import (CPP, sobol_v_host, "_ZL12sobol_v_host");

   scrambled_sobol_v_host : aliased array (0 .. 19999, 0 .. 127) of aliased unsigned;  -- /usr/local/cuda-8.0/include/sobol_direction_vectors.h:20012
   pragma Import (CPP, scrambled_sobol_v_host, "_ZL22scrambled_sobol_v_host");

   scrambled_sobol_c_host : aliased array (0 .. 39999) of aliased unsigned;  -- /usr/local/cuda-8.0/include/sobol_direction_vectors.h:40015
   pragma Import (CPP, scrambled_sobol_c_host, "_ZL22scrambled_sobol_c_host");

end sobol_direction_vectors_h;

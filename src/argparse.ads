with Ada.Strings.Unbounded;
with GNAT.Command_Line;

package argparse is

   EXIT_FROM_COMMAND_LINE : exception renames GNAT.Command_Line.EXIT_FROM_COMMAND_LINE;

   type parser_t is tagged record
      config : GNAT.Command_Line.Command_Line_Configuration;
      exec_mode : Character := ASCII.NUL;
      img_file : ada.Strings.Unbounded.Unbounded_String;
      epoches : Positive := 1;
      resume : Boolean := False;
      batch_size : Positive := 1;
   end record;

   procedure parse(self : in out parser_t);

end argparse;

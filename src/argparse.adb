with Ada.Directories;
with GNAT.OS_Lib;

package body argparse is



   procedure parse(self : in out parser_t) is

      procedure call_back(Switch    : String;
                          Parameter : String;
                          Section   : String) is
         use type ada.Directories.File_Kind;
         l_exec_mode : Character := (if Section="run" then 'R' else 'T');
      begin

         if self.exec_mode = ASCII.NUL or self.exec_mode = l_exec_mode then
            self.exec_mode := l_exec_mode;

            if switch = "-i" then

               declare
                  l_img_file : String := GNAT.OS_Lib.Normalize_Pathname(Parameter);
               begin
                  if ada.Directories.Exists(l_img_file) then
                     if ada.Directories.Kind(l_img_file) = ada.Directories.Ordinary_File then
                        self.img_file := ada.Strings.Unbounded.To_Unbounded_String(l_img_file);
                     else
                        raise gnat.Command_Line.Invalid_Parameter with "Input file path is a directory";
                     end if;
                  else
                     raise gnat.Command_Line.Invalid_Parameter with "Input file path does not exists";
                  end if;
               end;
            elsif switch = "-n" then
               self.epoches := Integer'Value(Parameter);
            elsif switch = "-r" then
               self.resume := True;
            elsif switch = "-b" then
               self.batch_size := Integer'Value(Parameter);
            end if;

         else

            raise gnat.Command_Line.Invalid_Section with "Mutually exclusive options";
         end if;

      end call_back;

   begin
      GNAT.Command_Line.Set_Usage(self.config,
                                  "-run -i<img_file> / -train -n<no_epoches> [-r]",
                                  "Use with either -run or -train switches",
                                  "-run          : Inference mode"&ASCII.LF&
                                    "-train        : Training mode"&ASCII.LF&
                                    "-i img_file   : Image file Path"&ASCII.LF&
                                    "-n no_epoches : No. of epoches"&ASCII.LF&
                                    "-b batch_size : Batch size"&ASCII.LF&
                                    "-r            : Resume from last savepoint");

      GNAT.Command_Line.Define_Section(self.config,"run");
      GNAT.Command_Line.Define_Section(self.config,"train");

      GNAT.Command_Line.Define_Switch(self.config, "-i:", Help => "Image file", Section => "run", Argument=>"String");
      GNAT.Command_Line.Define_Switch(self.config, "-n:", Help => "no. of epoches", Section => "train",Argument=>"Integer");
      GNAT.Command_Line.Define_Switch(self.config, "-b:", Help => "Batch Size", Section => "train", Argument=>"Integer");
      GNAT.Command_Line.Define_Switch(self.config, "-r", Help => "resume savepoint", Section => "train");

      GNAT.Command_Line.Getopt(self.config, call_back'Unrestricted_Access);

      if self.exec_mode = ASCII.NUL then
         GNAT.Command_Line.Display_Help(self.config);
         raise GNAT.Command_Line.Exit_From_Command_Line;
      end if;

   end parse;
end argparse;

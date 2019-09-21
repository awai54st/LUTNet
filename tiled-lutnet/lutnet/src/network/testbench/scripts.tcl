open_project -reset dummy_c6_LUTNet
set_top conv_wrapper_tp
add_files {hw/top_tb.cpp hw/hw_config.h hw/components.h}
add_files -tb {tb/main.cpp tb/cat1.png tb/airplane1.png}
open_solution -reset "soluion1"
#set_part {xcvu9p-flga2104-3-e-EVAL}
#set_part {xcvu9p-flga2104-3-e}
set_part {xcku115-flva1517-3-e}
#set_part {xc7z020clg484-1}
create_clock -period 10 -name default

# Uncomment one of the three lines below to run each of the build stages in turn.
# Then use the 'vivado_hls -f script.tcl' command.
# csim_design -clean -compiler gcc
#csim_design
csynth_design
#cosim_design -O -rtl vhdl

#export_design -rtl verilog -format ip_catalog
exit

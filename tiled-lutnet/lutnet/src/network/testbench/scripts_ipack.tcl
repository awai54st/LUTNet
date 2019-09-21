open_project dummy_c6_LUTNet
set_top conv_wrapper_tp
add_files {hw/top_tb.cpp hw/hw_config.h hw/components.h}
add_files -tb {tb/main.cpp tb/cat1.png tb/airplane1.png}
open_solution "soluion1"
set_part {xcku035-sfva784-3-e}
#set_part {xc7z020clg484-1}
create_clock -period 10 -name default

# Uncomment one of the three lines below to run each of the build stages in turn.
# Then use the 'vivado_hls -f script.tcl' command.
# csim_design -clean -compiler gcc
#csim_design
#csynth_design
#cosim_design -O -rtl verilog

export_design -rtl verilog -format ip_catalog
exit

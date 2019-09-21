###############################################################################
 #  Copyright (c) 2016, Xilinx, Inc.
 #  All rights reserved.
 #
 #  Redistribution and use in source and binary forms, with or without
 #  modification, are permitted provided that the following conditions are met:
 #
 #  1.  Redistributions of source code must retain the above copyright notice,
 #     this list of conditions and the following disclaimer.
 #
 #  2.  Redistributions in binary form must reproduce the above copyright
 #      notice, this list of conditions and the following disclaimer in the
 #      documentation and/or other materials provided with the distribution.
 #
 #  3.  Neither the name of the copyright holder nor the names of its
 #      contributors may be used to endorse or promote products derived from
 #      this software without specific prior written permission.
 #
 #  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 #  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 #  THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 #  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 #  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 #  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 #  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 #  OR BUSINESS INTERRUPTION). HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 #  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 #  OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 #  ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 #
###############################################################################

###############################################################################
 #
 #
 # @file make-vivado-proj.tcl
 #
 # tcl script for block design and bitstream generation. Automatically 
 # launched by make-hw.sh. Tested with Vivado 2018.2
 #
 #
###############################################################################

# Creates a Vivado project ready for synthesis and launches bitstream generation
if {$argc != 4} {
  puts "Expected: <jam repo> <proj name> <proj dir> <xdc_dir>"
  exit
}

# paths to donut and jam IP folders
set config_jam_repo [lindex $argv 0]

# project name, target dir and FPGA part to use
set config_proj_name [lindex $argv 1]
set config_proj_dir [lindex $argv 2]
#set config_proj_part "xc7z020clg400-1"
#set config_proj_part "xcvu9p-flga2104-3-e-EVAL"
set config_proj_part "xcku115-flva1517-3-e"

# other project config

set xdc_dir [lindex $argv 3]

# set up project
create_project -force $config_proj_name $config_proj_dir -part $config_proj_part
set_property ip_repo_paths [list $config_jam_repo] [current_project]
update_ip_catalog

#Add PYNQ XDC
#add_files -fileset constrs_1 -norecurse "${xdc_dir}/pynqZ1-Z2.xdc"
add_files -fileset constrs_1 -norecurse "${xdc_dir}/clk_constr.xdc"

# create block design
create_bd_design "procsys"
#create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 ps7
#set ps7 [get_bd_cells ps7]
#apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 -config {make_external "FIXED_IO, DDR" Master "Disable" Slave "Disable" } $ps7
#create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0
#apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 -config {make_external "FIXED_IO, DDR" apply_board_preset "1" Master "Disable" Slave "Disable" }  [get_bd_cells processing_system7_0]
#set_property -dict [list CONFIG.PCW_USE_S_AXI_HP0 {1}] [get_bd_cells processing_system7_0]
#source "${xdc_dir}/pynqZ1-Z2.tcl"

#set_property -dict [apply_preset $ps7] $ps7
#set_property -dict [list CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {100} CONFIG.PCW_FPGA1_PERIPHERAL_FREQMHZ {142.86} CONFIG.PCW_FPGA2_PERIPHERAL_FREQMHZ {200} CONFIG.PCW_FPGA3_PERIPHERAL_FREQMHZ {166.67} CONFIG.PCW_EN_CLK1_PORT {1} CONFIG.PCW_EN_CLK2_PORT {1} CONFIG.PCW_EN_CLK3_PORT {1} CONFIG.PCW_USE_M_AXI_GP0 {1} CONFIG.PCW_USE_S_AXI_HP0 {1}] $ps7

# instantiate jam
#create_bd_cell -type ip -vlnv xilinx.com:hls:BlackBoxJam:1.0 BlackBoxJam_0
#create_bd_cell -type ip -vlnv xilinx.com:hls:conv_wrapper_tp:1.0 BlackBoxJam_0
#create_bd_cell -type ip -vlnv xilinx.com:hls:conv_wrapper_tp:1.0 conv_wrapper_tp_0

# connect jam to ps7
#apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config {Master "/BlackBoxJam_0/m_axi_hostmem" Clk "Auto" }  [get_bd_intf_pins ps7/S_AXI_HP0]
#delete_bd_objs [get_bd_nets ps7_FCLK_CLK0]
#connect_bd_net [get_bd_pins ps7/FCLK_CLK0] [get_bd_pins rst_ps7_100M/slowest_sync_clk]
#connect_bd_net [get_bd_pins ps7/FCLK_CLK0] [get_bd_pins axi_mem_intercon/ACLK]
#connect_bd_net [get_bd_pins ps7/FCLK_CLK0] [get_bd_pins axi_mem_intercon/S00_ACLK]
#connect_bd_net [get_bd_pins ps7/FCLK_CLK0] [get_bd_pins axi_mem_intercon/M00_ACLK]
#connect_bd_net [get_bd_pins BlackBoxJam_0/ap_clk] [get_bd_pins ps7/FCLK_CLK0]
#connect_bd_net [get_bd_pins ps7/S_AXI_HP0_ACLK] [get_bd_pins ps7/FCLK_CLK0]
#apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config {Master "/ps7/M_AXI_GP0" Clk "/ps7/FCLK_CLK0 (100 MHz)" }  [get_bd_intf_pins BlackBoxJam_0/s_axi_control]
#apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config {Master "/conv_wrapper_tp_0/m_axi_MASTER_BUS" Clk "/processing_system7_0/FCLK_CLK0 (100 MHz)" }  [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
###################apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config {Master "/processing_system7_0/M_AXI_GP0" Clk "/processing_system7_0/FCLK_CLK0 (100 MHz)" }  [get_bd_intf_pins conv_wrapper_tp_0/s_axi_CTRL_BUS]

#create_bd_cell -type ip -vlnv xilinx.com:hls:conv_wrapper_tp:1.0 conv_wrapper_tp_0
#create_bd_port -dir I -type clk CLK_IN
#set_property CONFIG.FREQ_HZ 100000000 [get_bd_ports CLK_IN]
#create_bd_port -dir I -type rst RST
#set_property CONFIG.ASSOCIATED_RESET {RST} [get_bd_ports /CLK_IN]
#create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 AXI_IN
#set_property CONFIG.ASSOCIATED_BUSIF {AXI_IN} [get_bd_ports /CLK_IN]
#set_property -dict [list CONFIG.TDATA_NUM_BYTES {64}] [get_bd_intf_ports AXI_IN]
#create_bd_intf_port -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 AXI_OUT
#set_property CONFIG.ASSOCIATED_BUSIF {AXI_IN:AXI_OUT} [get_bd_ports /CLK_IN]
#connect_bd_net [get_bd_ports CLK_IN] [get_bd_pins conv_wrapper_tp_0/ap_clk]
#connect_bd_net [get_bd_ports RST] [get_bd_pins conv_wrapper_tp_0/ap_rst_n]
#connect_bd_intf_net [get_bd_intf_ports AXI_IN] [get_bd_intf_pins conv_wrapper_tp_0/frame_in_V_V]
#connect_bd_intf_net [get_bd_intf_ports AXI_OUT] [get_bd_intf_pins conv_wrapper_tp_0/frame_out_V_V]

create_bd_cell -type ip -vlnv xilinx.com:hls:BlackBoxJam:1.0 BlackBoxJam_0
create_bd_port -dir I -type clk CLK_IN
set_property CONFIG.FREQ_HZ 200000000 [get_bd_ports CLK_IN]
create_bd_port -dir I -type rst RST
set_property CONFIG.ASSOCIATED_RESET {RST} [get_bd_ports /CLK_IN]
create_bd_intf_port -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 AXI_OUT
create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 AXI_IN
set_property -dict [list CONFIG.ADDR_WIDTH {7} CONFIG.PROTOCOL {AXI4LITE} CONFIG.MAX_BURST_LENGTH {1} CONFIG.SUPPORTS_NARROW_BURST {0}] [get_bd_intf_ports AXI_IN]
set_property CONFIG.ASSOCIATED_BUSIF {AXI_IN} [get_bd_ports /CLK_IN]
set_property CONFIG.ASSOCIATED_BUSIF {AXI_IN:AXI_OUT} [get_bd_ports /CLK_IN]
set_property -dict [list CONFIG.NUM_WRITE_OUTSTANDING {16} CONFIG.NUM_READ_OUTSTANDING {16} CONFIG.ADDR_WIDTH {64} CONFIG.DATA_WIDTH {64}] [get_bd_intf_ports AXI_OUT]
#apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config {Master "/BlackBoxJam_0/m_axi_hostmem" Clk "/CLK_IN (100 MHz)" }  [get_bd_intf_ports AXI_OUT]
#apply_bd_automation -rule xilinx.com:bd_rule:board -config {rst_polarity "ACTIVE_LOW" }  [get_bd_pins rst_CLK_IN_100M/ext_reset_in]
connect_bd_intf_net [get_bd_intf_ports AXI_OUT] [get_bd_intf_pins BlackBoxJam_0/m_axi_hostmem]
connect_bd_net [get_bd_ports CLK_IN] [get_bd_pins BlackBoxJam_0/ap_clk]
connect_bd_intf_net [get_bd_intf_ports AXI_IN] [get_bd_intf_pins BlackBoxJam_0/s_axi_control]
connect_bd_net [get_bd_ports RST] [get_bd_pins BlackBoxJam_0/ap_rst_n]
assign_bd_address

# create HDL wrapper for the block design
save_bd_design
make_wrapper -files [get_files $config_proj_dir/$config_proj_name.srcs/sources_1/bd/procsys/procsys.bd] -top
add_files -norecurse $config_proj_dir/$config_proj_name.srcs/sources_1/bd/procsys/hdl/procsys_wrapper.v
update_compile_order -fileset sources_1
update_compile_order -fileset sim_1

#set_property strategy Flow_PerfOptimized_high [get_runs synth_1]
#
#set_property STEPS.SYNTH_DESIGN.ARGS.DIRECTIVE AlternateRoutability [get_runs synth_1]
#set_property STEPS.SYNTH_DESIGN.ARGS.RETIMING true [get_runs synth_1]
#
#set_property strategy Performance_ExtraTimingOpt [get_runs impl_1]
#set_property STEPS.OPT_DESIGN.ARGS.DIRECTIVE Explore [get_runs impl_1]
#set_property STEPS.POST_ROUTE_PHYS_OPT_DESIGN.ARGS.DIRECTIVE AggressiveExplore [get_runs impl_1]
#set_property STEPS.PHYS_OPT_DESIGN.ARGS.DIRECTIVE AggressiveExplore [get_runs impl_1]
#set_property STEPS.POST_ROUTE_PHYS_OPT_DESIGN.IS_ENABLED true [get_runs impl_1]
#
#write_bd_tcl $config_proj_dir/procsys.tcl
#
# launch bitstream generation
launch_runs impl_1 -to_step write_bitstream -jobs 2
wait_on_run impl_1



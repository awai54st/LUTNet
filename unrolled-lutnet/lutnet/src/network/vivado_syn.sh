HLS_IP_REPO="ip_catalog"
TARGET_NAME="CIFAR_10"
VIVADO_OUT_DIR="vivado_out"
VIVADO_SCRIPT_DIR=vivado_script
VIVADO_SCRIPT=$VIVADO_SCRIPT_DIR/make-vivado-proj.tcl

vivado -mode batch -notrace -source $VIVADO_SCRIPT -tclargs $HLS_IP_REPO $TARGET_NAME $VIVADO_OUT_DIR $VIVADO_SCRIPT_DIR

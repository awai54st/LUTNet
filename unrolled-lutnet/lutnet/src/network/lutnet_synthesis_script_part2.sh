#!/bin/bash
vivado_hls -f hls-export.tcl
rm -r ip_catalog/ip/
cp -r LUTNET_c6/sol1/impl/ip ip_catalog/
bash vivado_syn.sh

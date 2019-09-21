#!/bin/bash
vivado_hls -f hls-reb-export.tcl
rm -r ip_catalog/ip/
cp -r REBNET/sol1/impl/ip ip_catalog/
bash vivado_syn.sh

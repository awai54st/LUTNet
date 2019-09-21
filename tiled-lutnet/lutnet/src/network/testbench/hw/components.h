#define AP_INT_MAX_W 9216
#include <ap_int.h>
#include <hls_stream.h>
#include <ap_axi_sdata.h>
#include "hw_config.h"
#include "../../../library/hls/LUTNET_sliding_window.h"
#include "../../../library/hls/LUTNET_lut22_tm_matrixvector.h"
#include "../../../library/hls/LUTNET_lut22_tm_matrixvector_noth.h"
#include "../../../library/hls/LUTNET_maxpool.h"
#include "../../../library/hls/LUTNET_fxp_matrixvector.h"

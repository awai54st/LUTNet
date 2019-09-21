#define AP_INT_MAX_W 9216
#include <ap_int.h>
#include"weights.h"
#include "bnn-library.h"
#include "config.h"

// Allocate memory for the first (fixed point) fc layer
static ap_uint<64> weightMem0[16][208];
static ap_fixed<24, 16> thresMem0[16][16];
static ap_fixed<24, 16> alphaMem0[16][16];
static ap_fixed<24,16> means_in0[2];
static ap_fixed<24,16> means_out0[2];

unsigned int paddedSizeHW(unsigned int in, unsigned int padTo) {
  if(in % padTo == 0)
    return in;
  else
    return in + padTo - (in % padTo);
}


// My playground network
void conv_wrapper_tp(
  hls::stream<data_t_L2> &frame_in,
//  hls::stream<data_t_L8> &frame_in,
  hls::stream<data_t_out> &frame_out
//  hls::stream<data_t_L11> &frame_out
)
{

  /***********************************
   *  Initialise
   ***********************************/

  #pragma HLS DATAFLOW

//  #pragma HLS INTERFACE ap_ctrl_none port=return
//  #pragma HLS INTERFACE axis port=frame_in
//  #pragma HLS INTERFACE axis port=frame_out

  // AXI Streaming Connections
  // layer 1 conv
  hls::stream<data_t_L2> fcoutput_1;
  #pragma HLS STREAM variable=fcoutput_1 depth=1
  hls::stream<data_t_L3> fcoutput_2;
  #pragma HLS STREAM variable=fcoutput_2 depth=1

  hls::stream<data_t_L4> fcoutput_3;
  #pragma HLS STREAM variable=fcoutput_3 depth=1
  hls::stream<data_t_L5> fcoutput_4;
  #pragma HLS STREAM variable=fcoutput_4 depth=1

  /***********************************
   *  Weight memory partition
   ***********************************/

#pragma HLS ARRAY_PARTITION variable=weights_w_fc1_1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=weights_w_fc1_1 complete dim=4
#pragma HLS ARRAY_PARTITION variable=weights_w_fc2_1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=weights_w_fc2_1 complete dim=4
#pragma HLS ARRAY_PARTITION variable=weights_w_fc3_1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=weights_w_fc3_1 complete dim=4
#pragma HLS ARRAY_PARTITION variable=weights_w_fc4_1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=weights_w_fc4_1 complete dim=4
#pragma HLS ARRAY_PARTITION variable=weights_w_fc5_1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=weights_w_fc5_1 complete dim=4

  /***********************************
   *  Network
   ***********************************/

//  // First layer: fc layer
//  FXPMV<
//    L1_iR, L1_iC, L1_N, L1_oR, L1_oC, L1_M, L1_K, L1_ST, L1_BW, L2_BW, 24, 16, L1_TN, L1_TM,
//    hls::stream<data_t_L1>,
//    hls::stream<data_t_L2>
//    >(
//    frame_in,
//    fcoutput_1,
//    //frame_out,
//    weights_w_fc1_1,
//    //pruning_mask_conv1_1,
//    thresh_fc1,
//    alpha_fc1,
//    next_layer_means_fc1
//    );

  // Second layer: fc layer
  LUTNET_LUT51MV_TM<
    1, L2_iR, L2_iC, L2_N, L2_oR, L2_oC, L2_M, L2_K, L2_ST, L2_BW, 24, 16, L2_TN, L2_TM,
    hls::stream<data_t_L2>,
    hls::stream<data_t_L3>
    >(
    //fcoutput_1,
    frame_in,
    fcoutput_2,
    //frame_out,
    weights_w_fc2_1,
    thresh_fc2,
    alpha_fc2,
    next_layer_means_fc2,
    rand_map_fc2_1,
    rand_map_fc2_2,
    rand_map_fc2_3
    );

  // Third layer: fc layer
  LUTNET_LUT51MV_TM<
    2, L3_iR, L3_iC, L3_N, L3_oR, L3_oC, L3_M, L3_K, L3_ST, L3_BW, 24, 16, L3_TN, L3_TM,
    hls::stream<data_t_L3>,
    hls::stream<data_t_L4>
    >(
    fcoutput_2,
    fcoutput_3,
    weights_w_fc3_1,
    thresh_fc3,
    alpha_fc3,
    next_layer_means_fc3,
    rand_map_fc3_1,
    rand_map_fc3_2,
    rand_map_fc3_3
    );
  // Fourth layer: fc layer
  LUTNET_LUT51MV_TM<
    3, L4_iR, L4_iC, L4_N, L4_oR, L4_oC, L4_M, L4_K, L4_ST, L4_BW, 24, 16, L4_TN, L4_TM,
    hls::stream<data_t_L4>,
    hls::stream<data_t_L5>
    >(
    fcoutput_3,
    fcoutput_4,
    weights_w_fc4_1,
    thresh_fc4,
    alpha_fc4,
    next_layer_means_fc4,
    rand_map_fc4_1,
    rand_map_fc4_2,
    rand_map_fc4_3
    );
  // Fifth layer: fc layer
  LUTNET_LUT51MV_TM_NOTH<
    4, L5_iR, L5_iC, L5_N, L5_oR, L5_oC, L5_M, L5_K, L5_ST, L5_BW, 24, 16, OutputWidth_val, L5_TN, L5_TM,
    hls::stream<data_t_L5>,
    hls::stream<data_t_out>
    >(
    fcoutput_4,
    frame_out,
    weights_w_fc5_1,
    alpha_fc5,
    rand_map_fc5_1,
    rand_map_fc5_2,
    rand_map_fc5_3
    );



}

void DoCompute(ap_uint<64> * in, ap_uint<64> * out, const unsigned int numReps) {
#pragma HLS DATAFLOW

        stream<ap_uint<64> > memInStrm("DoCompute.memInStrm");
        #pragma HLS STREAM variable=memInStrm depth=256
        hls::stream<ap_uint<16>>  inter0("DoCompute.inter0");
        #pragma HLS stream depth=8 variable=inter0
        stream<ap_uint<256*2> > inter0_1("DoCompute.inter0_1");
        #pragma HLS STREAM variable=inter0_1 depth=1
        stream<ap_uint<L5_M*OutputWidth_val> > inter1("DoCompute.inter1");
        #pragma HLS STREAM variable=inter1 depth=128
        stream<ap_uint<16*OutputWidth_val> > inter1_1("DoCompute.inter1_1");
        #pragma HLS STREAM variable=inter1_1 depth=128
        stream<ap_uint<64> > memOutStrm("DoCompute.memOutStrm");


//unsigned const  L0_DEPTH = 128 / L0_PE;
//#define L0_SIMD 64
//#define L0_PE 16
//#define L0_WMEM 208
//#define L0_TMEM 16
//#define L0_MW 832
//#define L0_MH 256



        



        const unsigned int inBits = 28*28;
        const unsigned int inBitsPadded = 832*L1_BW; // paddedSizeHW(inBits, 64)
        const unsigned int inBytesPadded = inBitsPadded/8;



        Mem2Stream_Batch<64, inBytesPadded>(in, memInStrm, numReps);

        StreamingFCLayer_Batch<64,    16, 64, 16, 16, 24, 16,832, 256, 208, 16, 2>(memInStrm, inter0, weightMem0, thresMem0, alphaMem0, means_in0, means_out0, numReps);

        LUTNET_StreamingNumResConverter<16, 256*2, 1*1, 2>(inter0, inter0_1);



        const unsigned int outBits = 64*OutputWidth_val;
        const unsigned int outBytesPadded = outBits/8;

        conv_wrapper_tp(inter0_1, inter1);

        StreamingCast<ap_uint<L5_M*OutputWidth_val>, ap_uint<16*OutputWidth_val>>(inter1, inter1_1, numReps);
        StreamingDataWidthConverter_Batch<16*OutputWidth_val, 64, 1>(inter1_1, memOutStrm, numReps);
        //Stream2Mem_Batch<64, 512*8*3>(inter9, out, numReps);
        Stream2Mem_Batch<64, outBytesPadded>(memOutStrm, out, numReps);
}


void BlackBoxJam(ap_uint<64> * in, ap_uint<64> * out, bool doInit,
	unsigned int targetLayer, unsigned int targetMem,
	unsigned int targetInd, ap_uint<64> val, ap_fixed<24,16> fix_val) {

unsigned int numReps=1;

// pragmas for MLBP jam interface
// signals to be mapped to the AXI Lite slave port
#pragma HLS INTERFACE s_axilite port=return bundle=control
#pragma HLS INTERFACE s_axilite port=doInit bundle=control
#pragma HLS INTERFACE s_axilite port=targetLayer bundle=control
#pragma HLS INTERFACE s_axilite port=targetMem bundle=control
#pragma HLS INTERFACE s_axilite port=targetInd bundle=control
#pragma HLS INTERFACE s_axilite port=val bundle=control
#pragma HLS INTERFACE s_axilite port=fix_val bundle=control
#pragma HLS INTERFACE s_axilite port=numReps bundle=control
// signals to be mapped to the AXI master port (hostmem)
#pragma HLS INTERFACE m_axi offset=slave port=in bundle=hostmem depth=256
#pragma HLS INTERFACE s_axilite port=in bundle=control
#pragma HLS INTERFACE m_axi offset=slave port=out bundle=hostmem depth=256
#pragma HLS INTERFACE s_axilite port=out bundle=control

	if (doInit) {
		//DoMemInit(targetLayer, targetMem, targetInd, val,fix_val);
	} else {
		DoCompute(in, out, numReps);
	
	}
}

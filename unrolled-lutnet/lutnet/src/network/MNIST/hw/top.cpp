/******************************************************************************
 *  Copyright (c) 2018, ACES Lab, Univesity of California San Diego, CA, US.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *  1.  Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2.  Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *
 *  3.  Neither the name of the copyright holder nor the names of its
 *      contributors may be used to endorse or promote products derived from
 *      this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 *  THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 *  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 *  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 *  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 *  OR BUSINESS INTERRUPTION). HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 *  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 *  OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 *  ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *
 *  IMPORTANT NOTE:
 *  This work builds upon the binary CNN libary (BNN-PYNQ) provided by the following:
 *	Copyright (c) 2016, Xilinx, Inc.
 *	link to the original library (BNN-PYNQ) : https://github.com/Xilinx/BNN-PYNQ
 *
 *****************************************************************************/
/******************************************************************************
 *
 *
 * @file top.cpp
 *
 * HLS Description of the BNN, with axi-lite based parameter loading (DoMemInit)
 * and  dataflow architecture of the image inference (DoCompute)
 * 
 *
 *****************************************************************************/
#define AP_INT_MAX_W 9216
#include <ap_int.h>

#include "bnn-library.h"
#include "config.h"
#include"mnist_4lut_weights.h"


static ap_uint<L0_SIMD> weightMem0[L0_PE][L0_WMEM];
static ap_fixed<24, 16> thresMem0[L0_PE][L0_TMEM];
static ap_fixed<24, 16> alphaMem0[L0_PE][L0_TMEM];
static ap_fixed<24,16> means_in0[numRes];
static ap_fixed<24,16> means_out0[numRes];


unsigned int paddedSizeHW(unsigned int in, unsigned int padTo) {
  if(in % padTo == 0)
    return in;
  else
    return in + padTo - (in % padTo);
}

void DoMemInit(unsigned int targetLayer, unsigned int targetMem, unsigned int targetInd, ap_uint<64> val, ap_fixed<24,16> fix_val) {
	switch (targetLayer) {
		case 0:
			weightMem0[targetMem][targetInd] = val;
			break;
		case 1:
			thresMem0[targetMem][targetInd] = val;
			break;
		case 2:
			//weightMem1[targetMem][targetInd] = val;
			break;
		case 3:
			//thresMem1[targetMem][targetInd] = val;
			break;
		case 4:
			//weightMem2[targetMem][targetInd] = val;
			break;
		case 5:
			//thresMem2[targetMem][targetInd] = val;
			break;
		case 6:
			//weightMem3[targetMem][targetInd] = val;
			break;
		case 7:
			//thresMem3[targetMem][targetInd] = val;
			break;
		case 8:
			alphaMem0[targetMem][targetInd] = val;
			break;
		case 9:
			//alphaMem1[targetMem][targetInd] = val;
			break;
		case 10:
			//alphaMem2[targetMem][targetInd] = val;
			break;
		case 11:
			//alphaMem3[targetMem][targetInd] = val;
			break;
		case 12:
			//means_in1[targetMem][targetInd] = val;
			break;
		case 13:
			//means_in2[targetMem][targetInd] = val;
			break;
		case 14:
			//means_in3[targetMem][targetInd] = val;
			break;
		case 15:
			means_out0[targetMem][targetInd] = val;
			break;
		case 16:
			//means_out1[targetMem][targetInd] = val;
			break;
		case 17:
			//means_out2[targetMem][targetInd] = val;
			break;
		case 18:
			//means_out3[targetMem][targetInd] = val;
			break;
		case 19:
			means_in0[targetMem][targetInd] = val;
			break;
		default:
			break;
	}
}

void DoCompute(ap_uint<64> * in, ap_uint<64> * out, const unsigned int numReps) {
#pragma HLS DATAFLOW


        hls::stream<ap_uint<64>>     memInStrm("DoCompute.memInStrm");
        hls::stream<ap_uint<L0_PE>>  inter0("DoCompute.inter0");
        // This is where we implement LUTNet
	stream<ap_uint<256*2> > inter0_1("DoCompute.inter0_1");
#pragma HLS STREAM variable=inter0_1 depth=1
	stream<ap_uint<256*2> > inter0_2("DoCompute.inter0_2");
#pragma HLS STREAM variable=inter0_2 depth=1
	stream<ap_uint<256*2> > inter0_3("DoCompute.inter0_3");
#pragma HLS STREAM variable=inter0_3 depth=1
	stream<ap_uint<256*2> > inter0_4("DoCompute.inter0_4");
#pragma HLS STREAM variable=inter0_4 depth=1
	stream<ap_uint<10*24> > inter0_5("DoCompute.inter0_5");
#pragma HLS STREAM variable=inter0_5 depth=1
	stream<ap_uint<960> > inter0_6("DoCompute.inter0_6");
#pragma HLS STREAM variable=inter0_6 depth=1


// Back to REBNet 

          hls::stream<ap_uint<L1_PE>>  inter1("DoCompute.inter1");
          hls::stream<ap_uint<L2_PE>>  inter2("DoCompute.inter2");
          hls::stream<ap_uint<64>>     memOutStrm("DoCompute.memOutStrm");


          unsigned const  L0_DEPTH = 128 / L0_PE;

#pragma HLS DATAFLOW
#pragma HLS stream depth=256 variable=memInStrm         // mask memory latency
#pragma HLS stream depth=L0_DEPTH variable=inter0
#pragma HLS stream depth=256 variable=memOutStrm                // mask memory latency

  const unsigned int inBits = 28*28;
  const unsigned int inBitsPadded = 832; // paddedSizeHW(inBits, 64)
  const unsigned int inBytesPadded = inBitsPadded/8;
  const unsigned int outBits = 64;
  const unsigned int outBitsPadded = 64; // paddedSizeHW(outBits, 64)
  const unsigned int outBytesPadded = outBitsPadded/8;
  const unsigned int inWordsPerImg = inBitsPadded / 64;
  const unsigned int outWordsPerImg = outBitsPadded / 64;



        Mem2Stream_Batch<64, inBytesPadded>(in, memInStrm, numReps);

        StreamingFCLayer_Batch<64,    L0_PE, L0_SIMD, L0_PE, 16, 24, 16,L0_MW, L0_MH, L0_WMEM, L0_TMEM,numRes>(memInStrm, inter0, weightMem0, thresMem0, alphaMem0, means_in0, means_out0, numReps);

	LUTNET_StreamingNumResConverter<L0_PE, L0_MH*2, 1*1, 2>(inter0, inter0_1);

	LUTNET_LUT4MV<1, 1, 256, 1, 1, 256, 1, 1, numRes, 24, 16, hls::stream<ap_uint<256*2>>, hls::stream<ap_uint<256*2>>>(inter0_1, inter0_2, thresh_fc2, alpha_fc2, next_layer_means_fc2, rand_map_0_fc2, rand_map_1_fc2, rand_map_2_fc2);
	LUTNET_LUT4MV<1, 1, 256, 1, 1, 256, 1, 1, numRes, 24, 16, hls::stream<ap_uint<256*2>>, hls::stream<ap_uint<256*2>>>(inter0_2, inter0_3, thresh_fc3, alpha_fc3, next_layer_means_fc3, rand_map_0_fc3, rand_map_1_fc3, rand_map_2_fc3);
	LUTNET_LUT4MV<1, 1, 256, 1, 1, 256, 1, 1, numRes, 24, 16, hls::stream<ap_uint<256*2>>, hls::stream<ap_uint<256*2>>>(inter0_3, inter0_4, thresh_fc4, alpha_fc4, next_layer_means_fc4, rand_map_0_fc4, rand_map_1_fc4, rand_map_2_fc4);
	LUTNET_LUT4MV_NOTH<1, 1, 256, 1, 1, 10, 1, 1, numRes, 24, 16, hls::stream<ap_uint<256*2>>, hls::stream<ap_uint<10*24>>>(inter0_4, inter0_5, alpha_fc5, rand_map_0_fc5, rand_map_1_fc5, rand_map_2_fc5);

	LUTNET_StreamingNumResConverter<10*24, 960, 1*1, 2>(inter0_5, inter0_6);
	LUTNET_StreamingNumResConverter<960, 64, 1*1, 2>(inter0_6, memOutStrm);

        Stream2Mem_Batch<64, outBytesPadded>(memOutStrm, out, numReps);

}

void BlackBoxJam(ap_uint<64> * in, ap_uint<64> * out, bool doInit,
		unsigned int targetLayer, unsigned int targetMem,
		unsigned int targetInd, ap_uint<64> val, ap_fixed<24,16> fix_val) {

unsigned int numReps=2;
//#pragma HLS RESOURCE variable=thresMem4 core=RAM_S2P_LUTRAM
//#pragma HLS RESOURCE variable=thresMem5 core=RAM_S2P_LUTRAM
//#pragma HLS RESOURCE variable=thresMem6 core=RAM_S2P_LUTRAM
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

// partition PE arrays
#pragma HLS ARRAY_PARTITION variable=weightMem0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=thresMem0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=alphaMem0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=means_in0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=means_out0 complete dim=1

	if (doInit) {
		DoMemInit(targetLayer, targetMem, targetInd, val,fix_val);
	} else {
		DoCompute(in, out, numReps);

	}
}

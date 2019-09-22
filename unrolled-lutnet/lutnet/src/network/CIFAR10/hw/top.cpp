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
#include"c6_4lut_weights.h"


static ap_uint<L0_SIMD> weightMem0[L0_PE][L0_WMEM];
static ap_fixed<24, 16> thresMem0[L0_PE][L0_TMEM];
static ap_fixed<24, 16> alphaMem0[L0_PE][L0_TMEM];
static ap_fixed<24,16> means_out0[numRes];
static ap_uint<L1_SIMD> weightMem1[L1_PE][L1_WMEM];
static ap_fixed<24, 16> thresMem1[L1_PE][L1_TMEM];
static ap_fixed<24, 16> alphaMem1[L1_PE][L1_TMEM];
static ap_fixed<24,16> means_in1[numRes];
static ap_fixed<24,16> means_out1[numRes];
static ap_uint<L2_SIMD> weightMem2[L2_PE][L2_WMEM];
static ap_fixed<24, 16> thresMem2[L2_PE][L2_TMEM];
static ap_fixed<24, 16> alphaMem2[L2_PE][L2_TMEM];
static ap_fixed<24,16> means_in2[numRes];
static ap_fixed<24,16> means_out2[numRes];
static ap_uint<L3_SIMD> weightMem3[L3_PE][L3_WMEM];
static ap_fixed<24, 16> thresMem3[L3_PE][L3_TMEM];
static ap_fixed<24, 16> alphaMem3[L3_PE][L3_TMEM];
static ap_fixed<24,16> means_in3[numRes];
static ap_fixed<24,16> means_out3[numRes];
static ap_uint<L4_SIMD> weightMem4[L4_PE][L4_WMEM];
static ap_fixed<24, 16> thresMem4[L4_PE][L4_TMEM];
static ap_fixed<24, 16> alphaMem4[L4_PE][L4_TMEM];
static ap_fixed<24,16> means_in4[numRes];
static ap_fixed<24,16> means_out4[numRes];
//static ap_uint<L5_SIMD> weightMem5[L5_PE][L5_WMEM];
//static ap_fixed<24, 16> thresMem5[L5_PE][L5_TMEM];
//static ap_fixed<24, 16> alphaMem5[L5_PE][L5_TMEM];
//static ap_fixed<24,16> means_in5[numRes];
//static ap_fixed<24,16> means_out5[numRes];
static ap_uint<L6_SIMD> weightMem6[L6_PE][L6_WMEM];
static ap_fixed<24, 16> thresMem6[L6_PE][L6_TMEM];
static ap_fixed<24, 16> alphaMem6[L6_PE][L6_TMEM];
static ap_fixed<24,16> means_in6[numRes];
static ap_fixed<24,16> means_out6[numRes];
static ap_uint<L7_SIMD> weightMem7[L7_PE][L7_WMEM];
static ap_fixed<24, 16> thresMem7[L7_PE][L7_TMEM];
static ap_fixed<24, 16> alphaMem7[L7_PE][L7_TMEM];
static ap_fixed<24,16> means_in7[numRes];
static ap_fixed<24,16> means_out7[numRes];
static ap_uint<L8_SIMD> weightMem8[L8_PE][L8_WMEM];
static ap_fixed<24,16> means_in8[numRes];


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
		thresMem0[targetMem][targetInd] = *reinterpret_cast<ap_fixed<64,56> *>(&val);
		break;
	case 2:
		weightMem1[targetMem][targetInd] = val;
		break;
	case 3:
		thresMem1[targetMem][targetInd] = val;
		break;
	case 4:
		weightMem2[targetMem][targetInd] = val;
		break;
	case 5:
		thresMem2[targetMem][targetInd] = val;
		break;
	case 6:
		weightMem3[targetMem][targetInd] = val;
		break;
	case 7:
		thresMem3[targetMem][targetInd] = val;
		break;
	case 8:
		weightMem4[targetMem][targetInd] = val;
		break;
	case 9:
		thresMem4[targetMem][targetInd] = val;
		break;
	case 10:
		//weightMem5[targetMem][targetInd] = val;
		break;
	case 11:
		//thresMem5[targetMem][targetInd] = val;
		break;
	case 12:
		weightMem6[targetMem][targetInd] = val;
		break;
	case 13:
		thresMem6[targetMem][targetInd] = val;
		break;
	case 14:
		weightMem7[targetMem][targetInd] = val;
		break;
	case 15:
		thresMem7[targetMem][targetInd] = val;
		break;
	case 16:
		weightMem8[targetMem][targetInd] = val;
		break;
	case 17:
		alphaMem0[targetMem][targetInd] = val;
		break;
	case 18:
		alphaMem1[targetMem][targetInd] = val;
		break;
	case 19:
		alphaMem2[targetMem][targetInd] = val;
		break;
	case 20:
		alphaMem3[targetMem][targetInd] = val;
		break;
	case 21:
		alphaMem4[targetMem][targetInd] = val;
		break;
	case 22:
		//alphaMem5[targetMem][targetInd] = val;
		break;
	case 23:
		alphaMem6[targetMem][targetInd] = val;
		break;
	case 24:
		alphaMem7[targetMem][targetInd] = val;
		break;
	case 25:
		means_in1[targetMem][targetInd] = val;
		break;
	case 26:
		means_in2[targetMem][targetInd] = val;
		break;
	case 27:
		means_in3[targetMem][targetInd] = val;
		break;
	case 28:
		means_in4[targetMem][targetInd] = val;
		break;
	case 29:
		//means_in5[targetMem][targetInd] = val;
		break;
	case 30:
		means_in6[targetMem][targetInd] = val;
		break;
	case 31:
		means_in7[targetMem][targetInd] = val;
		break;
	case 32:
		means_out1[targetMem][targetInd] = val;
		break;
	case 33:
		means_out2[targetMem][targetInd] = val;
		break;
	case 34:
		means_out3[targetMem][targetInd] = val;
		break;
	case 35:
		means_out4[targetMem][targetInd] = val;
		break;
	case 36:
		//means_out5[targetMem][targetInd] = val;
		break;
	case 37:
		means_out6[targetMem][targetInd] = val;
		break;
	case 38:
		means_out7[targetMem][targetInd] = val;
		break;
	case 39:
		means_in8[targetMem][targetInd] = val;
		break;
	default:
		break;
	}
}

void DoCompute(ap_uint<64> * in, ap_uint<64> * out, const unsigned int numReps) {
#pragma HLS DATAFLOW

	stream<ap_uint<64> > inter0("DoCompute.inter0");
	stream<ap_uint<192> > inter0_1("DoCompute.inter0_1");
	stream<ap_uint<24> > inter0_2("DoCompute.inter0_2");
#pragma HLS STREAM variable=inter0_2 depth=128
	stream<ap_uint<64> > inter1("DoCompute.inter1");

#pragma HLS STREAM variable=inter1 depth=128
	stream<ap_uint<64> > inter2("DoCompute.inter2");
	stream<ap_uint<64> > inter3("DoCompute.inter3");
	#pragma HLS STREAM variable=inter3 depth=128
	stream<ap_uint<128> > inter4("DoCompute.inter4");
#pragma HLS STREAM variable=inter4 depth=128
	stream<ap_uint<128> > inter5("DoCompute.inter5");
	stream<ap_uint<128> > inter6("DoCompute.inter6");
#pragma HLS STREAM variable=inter6 depth=81
	stream<ap_uint<256> > inter7("DoCompute.inter7");
#pragma HLS STREAM variable=inter7 depth=1
// This is where we implement LUTNet
	stream<ap_uint<256*2> > inter7_1("DoCompute.inter7_1");
#pragma HLS STREAM variable=inter7_1 depth=1
	stream<ap_uint<256*2> > inter7_2("DoCompute.inter7_2");
#pragma HLS STREAM variable=inter7_2 depth=1
	stream<ap_uint<256*2> > inter7_3("DoCompute.inter7_3");
#pragma HLS STREAM variable=inter7_3 depth=1
// Back to REBNet 
	stream<ap_uint<256> > inter8("DoCompute.inter8");
#pragma HLS STREAM variable=inter8 depth=1
	stream<ap_uint<64> > inter9("DoCompute.inter9");
#pragma HLS STREAM variable=inter9 depth=128
	stream<ap_uint<64> > inter10("DoCompute.inter10");
#pragma HLS STREAM variable=inter10 depth=3
	stream<ap_uint<64> > memOutStrm("DoCompute.memOutStrm");



	const unsigned int inBits = 32*32*3*8;
	//const unsigned int inBitsPadded = paddedSize(inBits, 64);
	const unsigned int outBits = L8_MH*16;

	Mem2Stream_Batch<64, inBits/8>(in, inter0, numReps);
	StreamingDataWidthConverter_Batch<64, 192, (32*32*3*8) / 64>(inter0, inter0_1, numReps);
	StreamingDataWidthConverter_Batch<192, 24, (32*32*3*8) / 192>(inter0_1, inter0_2, numReps);

	StreamingFxdConvLayer_Batch<L0_K, L0_IFM_CH, L0_IFM_DIM, L0_OFM_CH, L0_OFM_DIM, 8, 1, L0_SIMD, L0_PE, 24, 16, L0_WMEM,	L0_TMEM,numRes>(inter0_2, inter1, weightMem0, thresMem0, alphaMem0, means_out0, numReps);
	StreamingConvLayer_Batch<L1_K, L1_IFM_CH, L1_IFM_DIM, L1_OFM_CH, L1_OFM_DIM, L1_SIMD, L1_PE, 16, 24,16,L1_WMEM, L1_TMEM,numRes>(inter1, inter2, weightMem1, thresMem1, alphaMem1, means_in1, means_out1, numReps);


	StreamingMaxPool_Batch<L1_OFM_DIM, 2, L1_OFM_CH,numRes>(inter2, inter3, numReps);
	StreamingConvLayer_Batch<L2_K, L2_IFM_CH, L2_IFM_DIM, L2_OFM_CH, L2_OFM_DIM, L2_SIMD, L2_PE, 16, 24, 16, L2_WMEM, L2_TMEM,numRes>(inter3, inter4, weightMem2, thresMem2, alphaMem2, means_in2, means_out2, numReps);
	StreamingConvLayer_Batch<L3_K, L3_IFM_CH, L3_IFM_DIM, L3_OFM_CH, L3_OFM_DIM, L3_SIMD, L3_PE, 16, 24, 16, L3_WMEM, L3_TMEM,numRes>(inter4, inter5, weightMem3, thresMem3, alphaMem3, means_in3, means_out3, numReps);
	StreamingMaxPool_Batch<L3_OFM_DIM, 2, L3_OFM_CH,numRes>(inter5, inter6, numReps);
	StreamingConvLayer_Batch<L4_K, L4_IFM_CH, L4_IFM_DIM, L4_OFM_CH, L4_OFM_DIM, L4_SIMD, L4_PE, 16, 24, 16, L4_WMEM, L4_TMEM,numRes>(inter6, inter7, weightMem4, thresMem4, alphaMem4, means_in4, means_out4, numReps);



	LUTNET_StreamingNumResConverter<256, 256*2, 3*3, 2>(inter7, inter7_1);

	LUTNET_SlidingWindow<L5_K, L5_IFM_CH, L5_IFM_DIM, L5_OFM_DIM, numRes, 1>(inter7_1, inter7_2, L5_K, L5_IFM_DIM, L5_OFM_DIM, 0);

	LUTNET_LUT4MV<L5_IFM_DIM, L5_IFM_DIM, L5_IFM_CH, L5_OFM_DIM, L5_OFM_DIM, L5_OFM_CH, L5_K, 1, numRes, 24, 16, hls::stream<ap_uint<256*2>>, hls::stream<ap_uint<256*2>>>(inter7_2, inter7_3, thresh_conv6, alpha_conv6, next_layer_means_conv6, rand_map_0_conv6, rand_map_1_conv6, rand_map_2_conv6);

	LUTNET_StreamingNumResConverter<256*2, 256, 1*1, 2>(inter7_3, inter8);



	// fully connected layers
	StreamingFCLayer_Batch<256, 64, L6_SIMD, L6_PE, 16, 24, 16 , L6_MW, L6_MH, L6_WMEM, L6_TMEM,numRes>(inter8, inter9, weightMem6, thresMem6, alphaMem6, means_in6, means_out6, numReps);
	StreamingFCLayer_Batch<64, 64, L7_SIMD, L7_PE, 16, 24,16,  L7_MW, L7_MH, L7_WMEM, L7_TMEM,numRes>(inter9, inter10, weightMem7, thresMem7, alphaMem7, means_in7, means_out7, numReps);

	StreamingFCLayer_NoActivation_Batch<64, 64, L8_SIMD, L8_PE, 16, 24, 16, L8_MW, L8_MH, L8_WMEM,numRes>(inter10, memOutStrm, weightMem8, means_in8, numReps);

	//Stream2Mem_Batch<64, 512*8*3>(inter9, out, numReps);
	Stream2Mem_Batch<64, outBits/8>(memOutStrm, out, numReps);
}

void BlackBoxJam(ap_uint<64> * in, ap_uint<64> * out, bool doInit,
		unsigned int targetLayer, unsigned int targetMem,
		unsigned int targetInd, ap_uint<64> val, ap_fixed<24,16> fix_val) {

unsigned int numReps=1;
#pragma HLS RESOURCE variable=thresMem4 core=RAM_S2P_LUTRAM
//#pragma HLS RESOURCE variable=thresMem5 core=RAM_S2P_LUTRAM
#pragma HLS RESOURCE variable=thresMem6 core=RAM_S2P_LUTRAM
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
#pragma HLS ARRAY_PARTITION variable=weightMem1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=thresMem1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=weightMem2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=thresMem2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=weightMem3 complete dim=1
#pragma HLS ARRAY_PARTITION variable=thresMem3 complete dim=1
#pragma HLS ARRAY_PARTITION variable=weightMem4 complete dim=1
#pragma HLS ARRAY_PARTITION variable=thresMem4 complete dim=1
//#pragma HLS ARRAY_PARTITION variable=weightMem5 complete dim=1
//#pragma HLS ARRAY_PARTITION variable=thresMem5 complete dim=1
#pragma HLS ARRAY_PARTITION variable=weightMem6 complete dim=1
#pragma HLS ARRAY_PARTITION variable=thresMem6 complete dim=1
#pragma HLS ARRAY_PARTITION variable=weightMem7 complete dim=1
#pragma HLS ARRAY_PARTITION variable=thresMem7 complete dim=1
#pragma HLS ARRAY_PARTITION variable=weightMem8 complete dim=1

#pragma HLS ARRAY_PARTITION variable=alphaMem0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=alphaMem1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=alphaMem2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=alphaMem3 complete dim=1
#pragma HLS ARRAY_PARTITION variable=alphaMem4 complete dim=1
//#pragma HLS ARRAY_PARTITION variable=alphaMem5 complete dim=1
#pragma HLS ARRAY_PARTITION variable=alphaMem6 complete dim=1
#pragma HLS ARRAY_PARTITION variable=alphaMem7 complete dim=1

#pragma HLS ARRAY_PARTITION variable=means_in1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=means_in2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=means_in3 complete dim=1
#pragma HLS ARRAY_PARTITION variable=means_in4 complete dim=1
//#pragma HLS ARRAY_PARTITION variable=means_in5 complete dim=1
#pragma HLS ARRAY_PARTITION variable=means_in6 complete dim=1
#pragma HLS ARRAY_PARTITION variable=means_in7 complete dim=1
#pragma HLS ARRAY_PARTITION variable=means_in8 complete dim=1

#pragma HLS ARRAY_PARTITION variable=means_out1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=means_out2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=means_out3 complete dim=1
#pragma HLS ARRAY_PARTITION variable=means_out4 complete dim=1
//#pragma HLS ARRAY_PARTITION variable=means_out5 complete dim=1
#pragma HLS ARRAY_PARTITION variable=means_out6 complete dim=1
#pragma HLS ARRAY_PARTITION variable=means_out7 complete dim=1
#pragma HLS ARRAY_PARTITION variable=means_out0 complete dim=1

	if (doInit) {
		DoMemInit(targetLayer, targetMem, targetInd, val,fix_val);
	} else {
		DoCompute(in, out, numReps);

	}
}

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
 * HLS Description of the CNV BNN, with axi-lite based parameter loading (DoMemInit)
 * and  dataflow architecture of the image inference (DoCompute)
 *
 *
 *****************************************************************************/
#include "bnn-library.h"
#include "config.h"


static ap_uint<L0_SIMD> weightMem0[L0_PE][L0_WMEM];
static ap_fixed<24, 16> thresMem0[L0_PE][L0_TMEM];
static ap_fixed<24, 16> alphaMem0[L0_PE][L0_TMEM];
static ap_fixed<24,16> means_in0[numRes];
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

unsigned int paddedSizeHW(unsigned int in, unsigned int padTo) {
  if(in % padTo == 0)
    return in;
  else
    return in + padTo - (in % padTo);
}

void DoMemInit(unsigned int targetLayer, unsigned int targetMem, unsigned int targetInd, ap_uint<64> val,  ap_fixed<24,16> fix_val) {
  switch(targetLayer) {
		case 0:
			weightMem0[targetMem][targetInd] = val;
			break;
		case 1:
			thresMem0[targetMem][targetInd] = val;
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
			alphaMem0[targetMem][targetInd] = val;
		    break;
		case 9:
			alphaMem1[targetMem][targetInd] = val;
			break;
		case 10:
			alphaMem2[targetMem][targetInd] = val;
			break;
		case 11:
			alphaMem3[targetMem][targetInd] = val;
			break;
		case 12:
			means_in1[targetMem][targetInd] = val;
			break;
		case 13:
			means_in2[targetMem][targetInd] = val;
			break;
		case 14:
			means_in3[targetMem][targetInd] = val;
			break;
		case 15:
			means_out0[targetMem][targetInd] = val;
			break;
		case 16:
			means_out1[targetMem][targetInd] = val;
			break;
		case 17:
			means_out2[targetMem][targetInd] = val;
			break;
		case 18:
			means_out3[targetMem][targetInd] = val;
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
	  hls::stream<ap_uint<L1_PE>>  inter1("DoCompute.inter1");
	  hls::stream<ap_uint<L2_PE>>  inter2("DoCompute.inter2");
	  hls::stream<ap_uint<64>>     memOutStrm("DoCompute.memOutStrm");


	  unsigned const  L0_DEPTH = 128 / L0_PE;
	  unsigned const  L1_DEPTH = 128 / L1_PE;
	  unsigned const  L2_DEPTH = 128 / L2_PE;

#pragma HLS DATAFLOW
#pragma HLS stream depth=256 variable=memInStrm     	// mask memory latency
#pragma HLS stream depth=L0_DEPTH variable=inter0
#pragma HLS stream depth=L1_DEPTH variable=inter1
#pragma HLS stream depth=L2_DEPTH variable=inter2
#pragma HLS stream depth=256 variable=memOutStrm		// mask memory latency

  const unsigned int inBits = 28*28;
  const unsigned int inBitsPadded = 832; // paddedSizeHW(inBits, 64)
  const unsigned int inBytesPadded = inBitsPadded/8;
  const unsigned int outBits = 64;
  const unsigned int outBitsPadded = 64; // paddedSizeHW(outBits, 64)
  const unsigned int outBytesPadded = outBitsPadded/8;
  const unsigned int inWordsPerImg = inBitsPadded / 64;
  const unsigned int outWordsPerImg = outBitsPadded / 64;



	Mem2Stream_Batch<64, inBytesPadded>(in, memInStrm, numReps);

	 StreamingFCLayer_Batch<64,    L0_PE, L0_SIMD, L0_PE, 16, 24, 16,L0_MW, L0_MH, L0_WMEM, L0_TMEM,numRes>
	    (memInStrm, inter0, weightMem0, thresMem0, alphaMem0, means_in0, means_out0, numReps);
	  StreamingFCLayer_Batch<L0_PE, L1_PE, L1_SIMD, L1_PE, 16, 24, 16,L1_MW, L1_MH, L1_WMEM, L1_TMEM,numRes>
	    (inter0, inter1, weightMem1, thresMem1, alphaMem1, means_in1, means_out1,numReps);
	  StreamingFCLayer_Batch<L1_PE, L2_PE, L2_SIMD, L2_PE, 16, 24, 16,L2_MW, L2_MH, L2_WMEM, L2_TMEM,numRes>
	    (inter1, inter2, weightMem2, thresMem2, alphaMem2, means_in2, means_out2,numReps);
	  StreamingFCLayer_Batch<L2_PE,    64, L3_SIMD, L3_PE, 16, 24, 16,L3_MW, L3_MH, L3_WMEM, L3_TMEM,numRes>
	    (inter2, memOutStrm, weightMem3, thresMem3, alphaMem3, means_in3, means_out3,numReps);
	  Stream2Mem_Batch<64, outBytesPadded>(memOutStrm, out, numReps);
}


void BlackBoxJam(ap_uint<64> * in, ap_uint<64> * out, bool doInit,
		unsigned int targetLayer, unsigned int targetMem,
		unsigned int targetInd, ap_uint<64> val, unsigned int numReps, ap_fixed<24,16> fix_val) {
// pragmas for MLBP jam interface
// signals to be mapped to the AXI Lite slave port
	numReps=1;
#pragma HLS INTERFACE s_axilite port=return bundle=control
#pragma HLS INTERFACE s_axilite port=doInit bundle=control
#pragma HLS INTERFACE s_axilite port=targetLayer bundle=control
#pragma HLS INTERFACE s_axilite port=targetMem bundle=control
#pragma HLS INTERFACE s_axilite port=targetInd bundle=control
#pragma HLS INTERFACE s_axilite port=val bundle=control
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


#pragma HLS ARRAY_PARTITION variable=alphaMem0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=alphaMem1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=alphaMem2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=alphaMem3 complete dim=1


#pragma HLS ARRAY_PARTITION variable=means_in0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=means_in1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=means_in2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=means_in3 complete dim=1

#pragma HLS ARRAY_PARTITION variable=means_out1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=means_out2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=means_out3 complete dim=1
#pragma HLS ARRAY_PARTITION variable=means_out0 complete dim=1

	if (doInit) {
		DoMemInit(targetLayer, targetMem, targetInd, val,fix_val);
	} else {
		DoCompute(in, out, numReps);
	}
}

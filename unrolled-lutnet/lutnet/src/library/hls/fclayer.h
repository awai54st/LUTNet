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
 * @file fclayer.h
 *
 * Library of templated HLS functions for BNN deployment. 
 * This file lists a set of convenience funtions used to implement fully 
 * connected layers
 * 
 *
 *****************************************************************************/

#define AP_INT_MAX_W 9216
#include <ap_int.h>

// helper function for fully connected layers
// instantiates matrix vector unit plus data width converters
template<unsigned int InStreamW, unsigned int OutStreamW,
		unsigned int SIMDWidth, unsigned int PECount,
		unsigned int PopCountWidth,
		unsigned int AccWidth, 	        // number of bits for accumulation
		unsigned int AccIntWidth,     // number of integer bits for accumulation
		unsigned int MatrixW, unsigned int MatrixH,
		unsigned int WMemCount, unsigned int TMemCount, unsigned int numRes>
void StreamingFCLayer_Batch(stream<ap_uint<InStreamW> > & in,
		stream<ap_uint<OutStreamW> > & out,
		const ap_uint<SIMDWidth> weightMem[PECount][WMemCount],
		const ap_fixed<AccWidth,AccIntWidth> thresMem[PECount][TMemCount],
		const ap_fixed<AccWidth,AccIntWidth> alphaMem[PECount][TMemCount],
		const ap_fixed<AccWidth,AccIntWidth> this_layer_means[numRes], // We added!
		const ap_fixed<AccWidth,AccIntWidth> next_layer_means[numRes], // We added!
		const unsigned int numReps) {
#pragma HLS INLINE
  unsigned const  InpPerImage = MatrixW / InStreamW;
  unsigned const  OutPerImage = MatrixH / PECount;

  Resid_WidthAdjustedInputStream <InStreamW, SIMDWidth, InpPerImage,numRes>  wa_in (in,  numReps);
  Resid_WidthAdjustedOutputStream<PECount,  OutStreamW, OutPerImage,numRes>  wa_out(out, numReps);

  StreamingMatrixVector_Batch<SIMDWidth, PECount, PopCountWidth, MatrixW, MatrixH, AccWidth, AccIntWidth, WMemCount, TMemCount,numRes>
    (wa_in, wa_out, weightMem, thresMem,  alphaMem, this_layer_means, next_layer_means, numReps);




}

// helper function for fully connected layers with no activation
// instantiates matrix vector unit plus data width converters
template<unsigned int InStreamW, unsigned int OutStreamW,
		unsigned int SIMDWidth, unsigned int PECount,
		unsigned int PopCountWidth,
		unsigned int AccWidth, 	        // number of bits for accumulation
		unsigned int AccIntWidth,     // number of integer bits for accumulation
		unsigned int MatrixW, unsigned int MatrixH,
		unsigned int WMemCount, unsigned int numRes>
void StreamingFCLayer_NoActivation_Batch(stream<ap_uint<InStreamW> > & in,
		stream<ap_uint<OutStreamW> > & out,
		const ap_uint<SIMDWidth> weightMem[PECount][WMemCount],
		const ap_fixed<AccWidth,AccIntWidth> this_layer_means[numRes], // We added!
		const unsigned int numReps) {
#pragma HLS INLINE

	stream<ap_uint<SIMDWidth> > in2mvu("StreamingFCLayer_NoAct_Batch.in2mvu");
	stream<ap_uint<PECount * PopCountWidth> > mvu2out("StreamingFCLayer_NoAct_Batch.mvu2out");
	const unsigned int InpPerImage = MatrixW / InStreamW;
	const unsigned int OutPerImage = MatrixH / PECount;




	Resid_StreamingDataWidthConverter_Batch<InStreamW, SIMDWidth, InpPerImage,numRes>(in,
			in2mvu, numReps);

	StreamingMatrixVector_NoActivation_Batch<SIMDWidth, PECount, PopCountWidth,
			MatrixW, MatrixH, AccWidth, AccIntWidth, WMemCount,numRes>(in2mvu, mvu2out, weightMem, this_layer_means, numReps);

	StreamingDataWidthConverter_Batch<PECount * PopCountWidth, OutStreamW,
			OutPerImage>(mvu2out, out, numReps);


 

}

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
 * @file convlayer.h
 *
 * Library of templated HLS functions for BNN deployment. 
 * This file lists a set of convenience funtions used to implement  
 * convolutional layers
 * 
 *
 *****************************************************************************/

#define AP_INT_MAX_W 9216
#include <ap_int.h>

template<
// convolution parameters
		unsigned int ConvKernelDim,	// e.g 3 for a 3x3 conv kernel (assumed square)
		unsigned int IFMChannels,		// number of input feature maps
		unsigned int IFMDim,	// width of input feature map (assumed square)
		unsigned int OFMChannels,		// number of output feature maps
		unsigned int OFMDim,			// IFMDim-ConvKernelDim+1 or less

		// matrix-vector unit parameters
		unsigned int SIMDWidth, 		// number of SIMD lanes
		unsigned int PECount,			// number of PEs
		unsigned int PopCountWidth, 	// number of bits for popcount
		unsigned int AccWidth, 	        // number of bits for accumulation
		unsigned int AccIntWidth,     // number of integer bits for accumulation
		unsigned int WMemCount,			// entries in each PEs weight memory
		unsigned int TMemCount,		// entries in each PEs threshold memory
		unsigned int numRes		// number of residual levels
>

void StreamingConvLayer_Batch(stream<ap_uint<IFMChannels> > & in,
		stream<ap_uint<OFMChannels> > & out,
		const ap_uint<SIMDWidth> weightMem[PECount][WMemCount],
		const ap_fixed<AccWidth,AccIntWidth> thresMem[PECount][TMemCount],
		const ap_fixed<AccWidth,AccIntWidth> alphaMem[PECount][TMemCount],
		const ap_fixed<AccWidth,AccIntWidth> this_layer_means[numRes], // We added!
		const ap_fixed<AccWidth,AccIntWidth> next_layer_means[numRes], // We added!
		const unsigned int numReps) {
	// compute weight matrix dimension from conv params
	const unsigned int MatrixW = ConvKernelDim * ConvKernelDim * IFMChannels;
	const unsigned int MatrixH = OFMChannels;

	#pragma HLS INLINE


	stream<ap_uint<IFMChannels> > convInp("StreamingConvLayer_Batch.convInp");

	Resid_WidthAdjustedOutputStream <PECount, OFMChannels, OFMDim * OFMDim * (OFMChannels / PECount),numRes>  mvOut (out,  numReps);

	//WidthAdjustedInputStream <IFMChannels, IFMChannels*2, IFMDim * IFMDim * 2>  in_1 (in,  numReps);

	StreamingConvolutionInputGenerator_Batch<ConvKernelDim, IFMChannels, IFMDim,
			OFMDim, 1, 1, numRes>(in, convInp, numReps);

	//WidthAdjustedInputStream <IFMChannels*2, IFMChannels, OFMDim * OFMDim * ConvKernelDim * ConvKernelDim>  in_2 (convInp,  numReps);

	Resid_WidthAdjustedInputStream <IFMChannels, SIMDWidth, OFMDim * OFMDim * ConvKernelDim * ConvKernelDim,numRes>  mvIn (convInp,  numReps);

	StreamingMatrixVector_Batch<SIMDWidth, PECount, PopCountWidth, MatrixW,
			MatrixH, AccWidth,AccIntWidth, WMemCount, TMemCount,numRes>(mvIn, mvOut, weightMem, thresMem, alphaMem, this_layer_means, next_layer_means,
			numReps * OFMDim * OFMDim);



}


template<
// convolution parameters
		unsigned int ConvKernelDim,	// e.g 3 for a 3x3 conv kernel (assumed square)
		unsigned int IFMChannels,		// number of input feature maps
		unsigned int IFMDim,	// width of input feature map (assumed square)
		unsigned int OFMChannels,		// number of output feature maps
		unsigned int OFMDim,			// IFMDim-ConvKernelDim+1 or less

		// matrix-vector unit parameters
		unsigned int InpWidth,          // size of the fixed point input
		unsigned int InpIntWidth, // number of integer bits for the fixed point input
		unsigned int SIMDWidth, 		// number of SIMD lanes
		unsigned int PECount,			// number of PEs
		unsigned int AccWidth, 	        // number of bits for accumulation
		unsigned int AccIntWidth,     // number of integer bits for accumulation
		unsigned int WMemCount,			// entries in each PEs weight memory
		unsigned int TMemCount,			// entries in each PEs threshold memory
		unsigned int numRes		// number of residual levels
>
void StreamingFxdConvLayer_Batch(stream<ap_uint<IFMChannels * InpWidth> > & in,
		stream<ap_uint<OFMChannels> > & out,
		const ap_uint<SIMDWidth> weightMem[PECount][WMemCount],
		const ap_fixed<AccWidth, AccIntWidth> thresMem[PECount][TMemCount],
		const ap_fixed<AccWidth, AccIntWidth> alphaMem[PECount][TMemCount],
		const ap_fixed<AccWidth,AccIntWidth> next_layer_means[numRes], // We added!
		const unsigned int numReps) {
	// compute weight matrix dimension from conv params
	const unsigned int MatrixW = ConvKernelDim * ConvKernelDim * IFMChannels;
	const unsigned int MatrixH = OFMChannels;
	#pragma HLS INLINE


	stream<ap_uint<IFMChannels * InpWidth> > convInp("StreamingFxdConvLayer_Batch.convInp");
			
	Resid_WidthAdjustedOutputStream <PECount, OFMChannels, OFMDim * OFMDim * (OFMChannels / PECount),numRes>  mvOut (out,  numReps);
			
	StreamingConvolutionInputGenerator_Batch<ConvKernelDim,
			IFMChannels, IFMDim, OFMDim, InpWidth, 1, numRes>(in, convInp, numReps);
			
	WidthAdjustedInputStream <IFMChannels * InpWidth, SIMDWidth * InpWidth, OFMDim * OFMDim * ConvKernelDim * ConvKernelDim>  mvIn (convInp,  numReps);

	StreamingFxdMatrixVector_Batch<InpWidth, InpIntWidth, SIMDWidth, PECount,
			AccWidth, AccIntWidth, MatrixW, MatrixH, WMemCount, TMemCount,numRes>(mvIn,
			mvOut, weightMem, thresMem, alphaMem, next_layer_means, numReps * OFMDim * OFMDim);



}

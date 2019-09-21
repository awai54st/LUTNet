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
 * @file dma.h
 *
 * Library of templated HLS functions for BNN deployment. 
 * This file lists a set of functions to access memory mapped values into 
 * streams 
 *
 *****************************************************************************/

#define AP_INT_MAX_W 9216
#include <ap_int.h>

// essentially small DMA generators, moving data between mem-mapped arrays and
// streams
template<unsigned int DataWidth, unsigned int numBytes>
void Mem2Stream(ap_uint<DataWidth> * in, stream<ap_uint<DataWidth> > & out) {
	CASSERT_DATAFLOW(DataWidth % 8 == 0);
	const unsigned int numWords = numBytes / (DataWidth / 8);
	CASSERT_DATAFLOW(numWords != 0);
	for (unsigned int i = 0; i < numWords; i++) {
#pragma HLS PIPELINE II=1
		ap_uint<DataWidth> e = in[i];
		out.write(e);
	}
}

template<unsigned int DataWidth, unsigned int numBytes>
void Stream2Mem(stream<ap_uint<DataWidth> > & in, ap_uint<DataWidth> * out) {
	CASSERT_DATAFLOW(DataWidth % 8 == 0);

	const unsigned int numWords = numBytes / (DataWidth / 8);
	CASSERT_DATAFLOW(numWords != 0);
	for (unsigned int i = 0; i < numWords; i++) {
#pragma HLS PIPELINE II=1
		ap_uint<DataWidth> e = in.read();
		out[i] = e;
	}
}

// call different statically-sized variants of Mem2Stream and Stream2Mem to
// generate larger bursts when possible. otherwise, reading single images all
// the time limits the memory throughput.
// the 16 here can be any power of two (has to be power of two, otherwise
// checking the modulo takes a lot more resources)
template<unsigned int DataWidth, unsigned int numBytes>
void Mem2Stream_Batch(ap_uint<DataWidth> * in,
		stream<ap_uint<DataWidth> > & out, const unsigned int numReps) {
	const unsigned int indsPerRep = numBytes / (DataWidth / 8);
	unsigned int rep = 0;
	// make sure Mem2Stream does not get inlined here
	// we lose burst inference otherwise

	const int iters_divisable_by_16=(int)(numReps/16);
	const int iters_remainder=numReps-iters_divisable_by_16*16;
	for (int i=0;i<iters_divisable_by_16;i++)
	{
		Mem2Stream<DataWidth, numBytes * 16>(&in[rep * indsPerRep], out);
		rep += 16;
	}
	for (int i=0;i<iters_remainder;i++)
	{
		Mem2Stream<DataWidth, numBytes>(&in[rep * indsPerRep], out);
					rep += 1;
	}
	/*while (rep != numReps) {
		unsigned int repsLeft = numReps - rep;
		if ((repsLeft & 0xF) == 0) {
			// repsLeft divisable by 16, read 16 images
			Mem2Stream<DataWidth, numBytes * 16>(&in[rep * indsPerRep], out);
			rep += 16;
		} else {
			// fallback, read single image
			Mem2Stream<DataWidth, numBytes>(&in[rep * indsPerRep], out);
			rep += 1;
		}
	}*/
}
template<unsigned int DataWidth, unsigned int numBytes>
void Stream2Mem_Batch(stream<ap_uint<DataWidth> > & in,
		ap_uint<DataWidth> * out, const unsigned int numReps) {
	const unsigned int indsPerRep = numBytes / (DataWidth / 8);
	unsigned int rep = 0;
	// make sure Stream2Mem does not get inlined here
	// we lose burst inference otherwise
	const int iters_divisable_by_16=(int)(numReps/16);
	const int iters_remainder=numReps-iters_divisable_by_16*16;
	for (int i=0;i<iters_divisable_by_16;i++)
	{
		Stream2Mem<DataWidth, numBytes * 16>(in, &out[rep * indsPerRep]);
		rep += 16;
	}
	for (int i=0;i<iters_remainder;i++)
	{
		Stream2Mem<DataWidth, numBytes>(in, &out[rep * indsPerRep]);
		rep += 1;
	}
	/*while (rep != numReps) {
		unsigned int repsLeft = numReps - rep;
		if ((repsLeft & 0xF) == 0) {
			// repsLeft divisable by 16, write 16 images
			Stream2Mem<DataWidth, numBytes * 16>(in, &out[rep * indsPerRep]);
			rep += 16;
		} else {
			// fallback, write single image
			Stream2Mem<DataWidth, numBytes>(in, &out[rep * indsPerRep]);
			rep += 1;
		}
	}*/
}

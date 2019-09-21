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
 * @file maxpool.h
 *
 * Library of templated HLS functions for BNN deployment. 
 * This file implement the BNN maxpool layer. For ReBNet, simple OR operations are
 * not possible and are replaced with fixed-point comparisons.
 *
 *****************************************************************************/

#define AP_INT_MAX_W 9216
#include <ap_int.h>

template<unsigned int ImgDim, unsigned int PoolDim, unsigned int NumChannels, unsigned int numRes>
void StreamingMaxPool(stream<ap_uint<NumChannels> > & in,
		stream<ap_uint<NumChannels> > & out) {
	CASSERT_DATAFLOW(ImgDim % PoolDim == 0);
	// need buffer space for a single maxpooled row of the image
	ap_uint<numRes> buf[ImgDim / PoolDim][NumChannels];
//#pragma HLS RESOURCE variable=buf core=RAM_2P_BRAM
	ap_uint<numRes> acc[NumChannels];
	ap_uint<numRes> vals[NumChannels];
	ap_uint<NumChannels> bits;
#pragma HLS ARRAY_PARTITION variable=acc complete dim=1
#pragma HLS ARRAY_PARTITION variable=vals complete dim=1
#pragma HLS ARRAY_PARTITION variable=buf complete dim=2
	for(unsigned int i = 0; i < ImgDim / PoolDim; i++) {
		for(int chan=0;chan<NumChannels;chan++)
		{
#pragma HLS UNROLL
			 buf[i][chan] = 0;
		}
	}


	for (unsigned int yp = 0; yp < ImgDim / PoolDim; yp++) {
			for (unsigned int ky = 0; ky < PoolDim; ky++) {
				for (unsigned int xp = 0; xp < ImgDim / PoolDim; xp++) {
					#pragma HLS PIPELINE II=1
					for (int chan=0;chan<NumChannels;chan++)
					{
						#pragma HLS UNROLL
						acc[chan]=0;
					}
					for (unsigned int kx = 0; kx < PoolDim; kx++) {
						//read and reformat the residual bits:
						for (int in_indx=0;in_indx<numRes;in_indx++)
						{
							bits=in.read();
							//if((xp==0)&&(yp==0)) printf("%llx\n",(uint64_t)bits);
							for (int chan=0;chan<NumChannels;chan++)
							{
								#pragma HLS UNROLL
								vals[chan](in_indx,in_indx)=bits(chan,chan);
							}
						}
						for (int chan=0;chan<NumChannels;chan++)
						{
							#pragma HLS UNROLL
							acc[chan]=(acc[chan]>vals[chan])?acc[chan]:vals[chan];
						}
					}
					// pool with old value in row buffer
					for (int chan=0;chan<NumChannels;chan++){
						#pragma HLS UNROLL
						buf[xp][chan]=(buf[xp][chan]>acc[chan])?buf[xp][chan]:acc[chan];
					}
				}
			}

			cout << endl;

			for (unsigned int outpix = 0; outpix < ImgDim / PoolDim; outpix++) {
				#pragma HLS PIPELINE II=1
				for (int in_indx=0;in_indx<numRes;in_indx++)
				{
					
					for (int chan=0;chan<NumChannels;chan++)
					{
						#pragma HLS UNROLL
						bits(chan,chan)=buf[outpix][chan](in_indx,in_indx);
					}
					//if((outpix ==0)&&(yp==0)) printf("%llx\n",(uint64_t)bits);
					out.write(bits);
				}
				
				// get buffer ready for next use
				for (int chan=0;chan<NumChannels;chan++)
				{
					#pragma HLS UNROLL
					buf[outpix][chan] = 0;
				}
				
			}

	}


}




// calling 1-image maxpool in a loop works well enough for now
template<unsigned int ImgDim, unsigned int PoolDim, unsigned int NumChannels, unsigned int numRes>
void StreamingMaxPool_Batch(stream<ap_uint<NumChannels> > & in,
		stream<ap_uint<NumChannels> > & out, unsigned int numReps) {
	for (unsigned int rep = 0; rep < numReps; rep++) {
		StreamingMaxPool<ImgDim, PoolDim, NumChannels,numRes>(in, out);
	}

}

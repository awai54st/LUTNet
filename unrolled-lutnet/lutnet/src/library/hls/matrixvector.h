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
 * @file matrix-vector.h
 *
 * Library of templated HLS functions for BNN deployment. 
 * This file implements the matrix-vector-threshold unit (MVTU) as described 
 * in the ReBNet paper. 
 *
 *****************************************************************************/


#define AP_INT_MAX_W 9216
#include <ap_int.h>

// popcount implemented as unsigned 1-bit add
// HLS automatically balances this into an adder tree
template<unsigned int SIMDWidth, unsigned int PopCountWidth>
	ap_int<PopCountWidth> NaivePopCount(ap_uint<SIMDWidth> & in) {
		ap_int<PopCountWidth> pct = 0;
		for (unsigned int i = 0; i < SIMDWidth; i++) {
			pct += in(i, i);
		}
		return pct;
	}

// streaming matrix-vector multiply component with binarized activation:
// binarized inputs, binarized weights, binarized outputs
template<unsigned int SIMDWidth, 		// number of SIMD lanes per PE
		unsigned int PECount,			// number of PEs
		unsigned int PopCountWidth, // number of bits in popcount accumulator (>=log2(fanin))
		unsigned int MatrixW,		// width of matrix, multiple of SIMDWidth
		unsigned int MatrixH,			// height of matrix, multiple of PECount
		unsigned int AccWidth, 	        // number of bits for accumulation
		unsigned int AccIntWidth,     // number of integer bits for accumulation
		unsigned int WMemCount,			// entries in weight memory
		unsigned int TMemCount,			// entries in threshold memory
		unsigned int numRes			// number of residual levels
		>
void StreamingMatrixVector_Batch(stream<ap_uint<SIMDWidth> > & in,
		stream<ap_uint<PECount> > & out,
		const ap_uint<SIMDWidth> weightMem[PECount][WMemCount],
		const ap_fixed<AccWidth,AccIntWidth> thresMem[PECount][TMemCount],
		const ap_fixed<AccWidth,AccIntWidth> alphaMem[PECount][TMemCount],
		const ap_fixed<AccWidth,AccIntWidth> this_layer_means[numRes], // We added!
		const ap_fixed<AccWidth,AccIntWidth> next_layer_means[numRes], // We added!
		const unsigned int numReps) {
		// how many different rows each neuron will compute
		// alternatively: number of vertical matrix chunks
		const unsigned int neuronFold = MatrixH / PECount;
		// how many synapse groups each row is split into
		// alternatively: number of horizontal matrix chunks
		const unsigned int synapseFold = MatrixW / SIMDWidth;
		// input vector buffer
		ap_uint<SIMDWidth> inputBuf[numRes][synapseFold];  // we extended this!
		// PE accumulator registers, initialized to zero on first call to function
		// why not defined as static? then different calls to StreamingMatrixVector
		// with the same template parameters would share these accumulator registers
		ap_int<PopCountWidth> accPopCount[numRes][PECount];
		ap_fixed<AccWidth,AccIntWidth> accResidual[PECount];
		#pragma HLS ARRAY_PARTITION variable=accPopCount complete dim=2
		#pragma HLS ARRAY_PARTITION variable=accPopCount complete dim=1
		#pragma HLS ARRAY_PARTITION variable=accResidual complete dim=1


		static int row=0;
		static int col=0;
		static int depth=0;
		//static unsigned long long enc[2]={0,0};
		static int chunk=0;


		for (int in_idx=0; in_idx<numRes; in_idx++)
		{
			for (unsigned int i = 0; i < PECount; i++)
			{
				#pragma HLS UNROLL
				accPopCount[in_idx][i] = 0;
			}
		}
		for (unsigned int i = 0; i < PECount; i++)
		{
			#pragma HLS UNROLL
			accResidual[i] = 0;
		}
		unsigned int nf = 0;
		unsigned int sf = 0;
		const unsigned int totalFold = neuronFold * synapseFold;


		
		// everything merged into a common iteration space (one "big" loop instead
		// of smaller nested loops) to get the pipelinening the way we want
		for (unsigned int i = 0; i < totalFold * numReps; i++)
		{
			#pragma HLS PIPELINE II=1
			for(int in_idx=0; in_idx<1; in_idx++)
			{   // we added!!
				#pragma HLS UNROLL
				ap_uint<SIMDWidth> inElem;
				if (nf == 0) {
					// read input from stream
					inElem = in.read();
					// buffer for reuse
					inputBuf[in_idx][sf] = inElem;  // we extended this!
				} else {
					// reuse buffered input
					inElem = inputBuf[in_idx][sf];
				}

				// compute matrix-vector product for each processing element
				for (unsigned int pe = 0; pe < PECount; pe++) {
					#pragma HLS UNROLL
					ap_uint<SIMDWidth> weight = weightMem[pe][nf * synapseFold + sf];
					ap_uint<SIMDWidth> masked = ~(weight ^ inElem);

					accPopCount[in_idx][pe] += NaivePopCount<SIMDWidth, PopCountWidth>(masked);


				}
			}
			// keep track of which folded synapse/neuron we are processing
			sf++;

			if(sf == synapseFold)
			{ // we added!!
				for(int in_idx=0; in_idx<numRes; in_idx++)
				{   // we added!!
					for (unsigned int pe = 0; pe < PECount; pe++)
					{
						#pragma HLS UNROLL
						accResidual[pe] = accResidual[pe]+accPopCount[in_idx][pe]*this_layer_means[in_idx];
						accPopCount[in_idx][pe] = 0;	// clear the accumulator
					}
				}


				for (unsigned int pe = 0; pe < PECount; pe++)
				{ // we added!!
					#pragma HLS UNROLL

					accResidual[pe] = alphaMem[pe][nf]*accResidual[pe]+thresMem[pe][nf];

				}

				for(int out_indx=0; out_indx<numRes; out_indx++)
				{

					ap_uint<PECount> outElem = 0;
					for (unsigned int pe = 0; pe < PECount; pe++)
					{ // we added!!
						#pragma HLS UNROLL
						outElem(pe, pe) = ((accResidual[pe]) > 0) ? 1 : 0;
						accResidual[pe] = outElem(pe, pe) ? (accResidual[pe] - next_layer_means[out_indx]) : (accResidual[pe] + next_layer_means[out_indx]);
					}
					out.write(outElem);
				}
				for (unsigned int pe = 0; pe < PECount; pe++)
				{
					#pragma HLS UNROLL
					accResidual[pe] = 0;
				}
				sf = 0;
				nf++;
			}
			if (nf == neuronFold)
			{
				// next image
				nf = 0;
			}

		}
	}


// TODO should be possible to integrate this into the baseline MVTU using a
// template parameter
// streaming matrix-vector multiply component with no activation:
// binarized inputs, binarized weights, PopCountWidth-bit outputs
template<unsigned int SIMDWidth, 		// number of SIMD lanes per PE
		unsigned int PECount,			// number of PEs
		unsigned int PopCountWidth, // number of bits in popcount accumulator (>=log2(fanin))
		unsigned int MatrixW,		// width of matrix, multiple of SIMDWidth
		unsigned int MatrixH,			// height of matrix, multiple of PECount
		unsigned int AccWidth, 	        // number of bits for accumulation
		unsigned int AccIntWidth,     // number of integer bits for accumulation
		unsigned int WMemCount,			// entries in weight memory
		unsigned int numRes			// number of residual levels
		>
void StreamingMatrixVector_NoActivation_Batch(stream<ap_uint<SIMDWidth> > & in,
		stream<ap_uint<PECount * PopCountWidth> > & out,
		const ap_uint<SIMDWidth> weightMem[PECount][WMemCount],
		const ap_fixed<AccWidth,AccIntWidth> this_layer_means[numRes], // We added!
		const unsigned int numReps) {
		// how many different rows each neuron will compute
		// alternatively: number of vertical matrix chunks
		const unsigned int neuronFold = MatrixH / PECount;
		// how many synapse groups each row is split into
		// alternatively: number of horizontal matrix chunks
		const unsigned int synapseFold = MatrixW / SIMDWidth;
		// input vector buffer
		ap_uint<SIMDWidth> inputBuf[numRes][synapseFold];
		// PE accumulator registers, initialized to zero on first call to function
		// why not defined as static? then different calls to StreamingMatrixVector
		// with the same template parameters would share these accumulator registers
		ap_int<PopCountWidth> accPopCount[numRes][PECount];
		ap_int<PopCountWidth> accResidual[PECount];
		#pragma HLS ARRAY_PARTITION variable=accPopCount complete dim=2
		#pragma HLS ARRAY_PARTITION variable=accResidual complete dim=1
		for (int in_idx=0; in_idx<numRes; in_idx++)
		{
			for (unsigned int i = 0; i < PECount; i++)
			{
				#pragma HLS UNROLL
				accPopCount[in_idx][i] = 0;
			}
		}
		for (unsigned int i = 0; i < PECount; i++)
		{
			#pragma HLS UNROLL
			accResidual[i] = 0;
		}


		unsigned int nf = 0;
		unsigned int sf = 0;
		const unsigned int totalFold = neuronFold * synapseFold;
		// everything merged into a common iteration space (one "big" loop instead
		// of smaller nested loops) to get the pipelinening the way we want
		for (unsigned int i = 0; i < totalFold * numReps; i++)
		{
			#pragma HLS PIPELINE II=1
			for(int in_idx=0; in_idx<numRes; in_idx++)
			{   // we added!!
				ap_uint<SIMDWidth> inElem;
				if (nf == 0) {
					// read input from stream
					inElem = in.read();
					// buffer for reuse
					inputBuf[in_idx][sf] = inElem;  // we extended this!
				} else {
					// reuse buffered input
					inElem = inputBuf[in_idx][sf];
				}
				// compute matrix-vector product for each processing element
				for (unsigned int pe = 0; pe < PECount; pe++) {
					#pragma HLS UNROLL
					ap_uint<SIMDWidth> weight = weightMem[pe][nf * synapseFold + sf];
					ap_uint<SIMDWidth> masked = ~(weight ^ inElem);
					accPopCount[in_idx][pe] += NaivePopCount<SIMDWidth, PopCountWidth>(masked);
				}
			}
			// keep track of which folded synapse/neuron we are processing
			sf++;

			if(sf == synapseFold)
			{
				for(int in_idx=0; in_idx<numRes; in_idx++)
				{   // we added!!
					for (unsigned int pe = 0; pe < PECount; pe++)
					{
						#pragma HLS UNROLL
						accResidual[pe] = accResidual[pe]+accPopCount[in_idx][pe]*this_layer_means[in_idx];
						accPopCount[in_idx][pe] = 0;	// clear the accumulator
					}
				}

				ap_uint<PECount * PopCountWidth> outElem = 0;
				for (unsigned int pe = 0; pe < PECount; pe++)
				{
					#pragma HLS UNROLL
					outElem((pe + 1) * PopCountWidth - 1, pe * PopCountWidth) = accResidual[pe];
					accResidual[pe] = 0;	// clear the accumulator
				}
				out.write(outElem);
				// next folded neuron
				sf = 0;
				nf++;
			}
			if (nf == neuronFold)
			{
				// next image
				nf = 0;
			}	
		}
	}

// streaming matrix-vector multiply component with binarized activation:
// fixed-point inputs, binarized weights, binarized outputs
template<unsigned int InpWidth,          // number of bits to use as the inputs.
		unsigned int InpIntWidth, // number of integer bits to use in the input.
		unsigned int SIMDWidth, 		// number of SIMD lanes per PE
		unsigned int PECount,			// number of PEs
		unsigned int AccWidth,          // number of bits in the accumulator
		unsigned int AccIntWidth, // number of integer bits to use in the accumulator.
		unsigned int MatrixW,		   // width of matrix, multiple of SIMDWidth
		unsigned int MatrixH,			// height of matrix, multiple of PECount
		unsigned int WMemCount,			// entries in weight memory
		unsigned int TMemCount,			// entries in threshold memory
		unsigned int numRes			// number of residual levels
		>
void StreamingFxdMatrixVector_Batch(stream<ap_uint<SIMDWidth * InpWidth> > & in,
		stream<ap_uint<PECount> > & out,
		const ap_uint<SIMDWidth> weightMem[PECount][WMemCount],
		const ap_fixed<AccWidth, AccIntWidth> thresMem[PECount][TMemCount],
		const ap_fixed<AccWidth,AccIntWidth> alphaMem[PECount][TMemCount],
		const ap_fixed<AccWidth, AccIntWidth> next_layer_means[numRes], // We added!
		const unsigned int numReps) {
		CASSERT_DATAFLOW(MatrixW % SIMDWidth == 0);CASSERT_DATAFLOW(
				MatrixH % PECount == 0);
		// how many different rows each neuron will compute
		// alternatively: number of vertical matrix chunks
		const unsigned int neuronFold = MatrixH / PECount;
		// how many synapse groups each row is split into
		// alternatively: number of horizontal matrix chunks
		const unsigned int synapseFold = MatrixW / SIMDWidth;
		// input vector buffer
		ap_uint<SIMDWidth * InpWidth> inputBuf[synapseFold];
		// PE accumulator registers, initialized to zero on first call to function
		// why not defined as static? then different calls to StreamingMatrixVector
		// with the same template parameters would share these accumulator registers
		ap_fixed<AccWidth, AccIntWidth, AP_TRN, AP_SAT> accReg[PECount];
		ap_fixed<AccWidth, AccIntWidth, AP_TRN, AP_SAT> intReg[PECount];
		for (unsigned int i = 0; i < PECount; i++) {
			#pragma HLS UNROLL
			accReg[i] = 0;
		}

		#pragma HLS ARRAY_PARTITION variable=accReg complete dim=1
		#pragma HLS ARRAY_PARTITION variable=intReg complete dim=1

		unsigned int nf = 0;
		unsigned int sf = 0;


		const unsigned int totalFold = neuronFold * synapseFold;
		// everything merged into a common iteration space (one "big" loop instead
		// of smaller nested loops) to get the pipelinening the way we want
		//int w_count=0;
		//FILE* fp = fopen("../../../../../../cnv-pynq/hw/test_im.bin","r");
		for (unsigned int i = 0; i < totalFold * numReps; i++) {
			#pragma HLS PIPELINE II=1
			ap_uint<SIMDWidth * InpWidth> inElem;
			if (nf == 0) {
				// read input from stream
				inElem = in.read();
				// buffer for reuse
				inputBuf[sf] = inElem;
			} else {
				// reuse buffered input
				inElem = inputBuf[sf];
			}
			// compute matrix-vector product for each processing element



			for (unsigned int pe = 0; pe < PECount; pe++) {
				#pragma HLS UNROLL
				ap_uint<SIMDWidth> weight = weightMem[pe][nf * synapseFold + sf];

				//ap_uint<SIMDWidth> masked = ~(weight ^ inElem);
				//accPopCount[pe] += NaivePopCount<SIMDWidth, PopCountWidth>(
				//		masked);
				//ap_fixed<InpWidth,InpIntWidth,AP_TRN,AP_SAT> * inVec = reinterpret_cast<ap_fixed<InpWidth,InpIntWidth,AP_TRN,AP_SAT> *>(&inElem);
				intReg[pe] = 0;
				for (unsigned int s = 0; s < SIMDWidth; s++) {
					#pragma HLS UNROLL
					ap_uint<InpWidth> tmp = inElem.range((s + 1) * InpWidth - 1,
							s * InpWidth);
					ap_fixed<InpWidth, InpIntWidth, AP_TRN, AP_SAT> val =
							*reinterpret_cast<ap_fixed<InpWidth, InpIntWidth,
									AP_TRN, AP_SAT> *>(&tmp);
					ap_int<2> w = (weight.range(s, s)) ? 1 : -1;
					intReg[pe] += w * val;
					// intReg[pe] += (ap_fixed<AccWidth, AccIntWidth, AP_TRN, AP_SAT>)(w * tmp)/256; // The working one!


					//if (weight.range(s,s)) accReg[pe] += val; // This is slower than the two lines above.
					//else accReg[pe] -= val;
				}
				accReg[pe] += intReg[pe];
				//cout<<accReg[pe]<<" ";
			}

			// keep track of which folded synapse/neuron we are processing
			sf++;
			if (sf == synapseFold) {
				for (unsigned int pe = 0; pe < PECount; pe++) {
					#pragma HLS UNROLL
					accReg[pe] = alphaMem[pe][nf]*accReg[pe]+thresMem[pe][nf];

				}

				for(int out_indx=0; out_indx<numRes; out_indx++)
				{
					ap_uint<PECount> outElem = 0;
					for (unsigned int pe = 0; pe < PECount; pe++)
					{
						#pragma HLS UNROLL
						outElem(pe, pe) = ((accReg[pe]) > 0) ? 1 : 0;
						accReg[pe] = ((accReg[pe]) > 0) ?  (accReg[pe] - next_layer_means[out_indx]) : (accReg[pe] + next_layer_means[out_indx]);
					}
					out.write(outElem);

				}



				for (unsigned int pe = 0; pe < PECount; pe++) {
					#pragma HLS UNROLL
					accReg[pe] = 0;
				}
				sf = 0;
				nf++;
				if (nf == neuronFold) {
				// next image
				nf = 0;
				}
			}

		}
	}

/*
    Copyright (c) 2018, Xilinx, Inc.
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    1.  Redistributions of source code must retain the above copyright notice,
        this list of conditions and the following disclaimer.

    2.  Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.

    3.  Neither the name of the copyright holder nor the names of its
        contributors may be used to endorse or promote products derived from
        this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
    THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
    PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
    CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
    EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
    PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
    OR BUSINESS INTERRUPTION). HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
    WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
    OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
    ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#define AP_INT_MAX_W 9216
#include <ap_int.h>

#ifndef MAX
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#endif


template<
          unsigned int MaxConvKernelDim,
          unsigned int MaxIFMChannels,
          unsigned int MaxIFMDim,
          unsigned int MaxOFMDim,
          unsigned int InpWidth, // bit per pixel
          unsigned int MaxStride
        >
void LUTNET_SlidingWindow(
        hls::stream<ap_uint<MaxIFMChannels*InpWidth> > & in,
        hls::stream<ap_uint<MaxIFMChannels*InpWidth> > & out,
        const unsigned int ConvKernelDim,
        const unsigned int IFMDim,
        const unsigned int OFMDim,
        const unsigned int Stride) {

    const unsigned int number_blocks = (ConvKernelDim >> Stride) + 1;
    const unsigned int cycles_write_block = (OFMDim * ConvKernelDim * ConvKernelDim);
    const unsigned int cycles_read_block = IFMDim << Stride;
    const unsigned int max_cycles = MAX(cycles_write_block,cycles_read_block);
    const unsigned int baseIter = IFMDim * ConvKernelDim + OFMDim * MAX(cycles_write_block, cycles_read_block);
    unsigned int counter_internal_block = 0;
    unsigned int current_block_write = 0;
#pragma HLS DEPENDENCE variable=current_block_write intra false
    unsigned int next_block_write = 0;
    unsigned int current_line = 0;
    unsigned int read_block = 0;
#pragma HLS RESET variable=read_block
    unsigned int inp = 0, ofm_y = 0, ofm_x = 0, k_y = 0, k_x = 0, current_k_y = 0;
#pragma HLS RESET variable=inp

    ap_uint<MaxIFMChannels*InpWidth> inputBuf[MaxConvKernelDim/MaxStride + 1][MaxStride * MaxIFMDim];
#pragma HLS DEPENDENCE variable=inputBuf inter false
#pragma HLS DEPENDENCE variable=inputBuf intra false
#pragma HLS ARRAY_PARTITION variable=inputBuf complete dim=1

//#pragma HLS RESOURCE variable inputBuf core=RAM_2P_LUTRAM
#pragma HLS RESOURCE variable inputBuf core=RAM_2P_BRAM

    for (unsigned int i = 0; i < baseIter; i++) {
#pragma HLS PIPELINE II=1
        if (inp < IFMDim * ConvKernelDim) // Initial buffer
        {
            ap_uint<MaxIFMChannels*InpWidth> inElem;
            inElem = in.read();
            inputBuf[current_block_write][current_line] = inElem;

            current_line++;
            inp++;

            if (current_line == (IFMDim << Stride))
            {
                current_line = 0;
                current_block_write++;
                read_block++;
                counter_internal_block = 0;
                if (current_block_write == number_blocks)
                    current_block_write = 0;
            }
        }
        else
        {
            if (counter_internal_block < cycles_write_block - 1) // We are writing output
            {
                unsigned int current_block_read = (current_block_write + 1 + (k_y >> Stride));
                if (current_block_read >= number_blocks)
                    current_block_read -= number_blocks;

                if (current_k_y >= (1 << Stride))
                    current_k_y = 0;
                unsigned int current_line_in_block = ((current_k_y) * IFMDim) + (ofm_x << Stride) + k_x;
                ap_uint<MaxIFMChannels*InpWidth> outElem = inputBuf[current_block_read][current_line_in_block];
                out.write(outElem);

                k_x++;
                if (k_x == ConvKernelDim) {
                    k_x = 0;
                    k_y++;
                    current_k_y++;

                    if (k_y == ConvKernelDim) {
                        k_y = 0;
                        current_k_y = 0;
                        ofm_x++;

                        if (ofm_x == OFMDim) {
                            ofm_x = 0;
                            ofm_y++;

                            if (ofm_y == OFMDim) {
                                ofm_y = 0;
                                inp = 0;
                            }
                        }
                    }
                }
            }

            if ((counter_internal_block < cycles_read_block-1) && (read_block < (IFMDim >> Stride))) // In parallel we write in the buffer, in the current block write if we still need to
            {
                ap_uint<MaxIFMChannels*InpWidth> inElem;
                inElem = in.read();
                inputBuf[current_block_write][current_line] = inElem;
                current_line++;
                if (current_line == (IFMDim << Stride)) // We read the whole block, we change the next block in which we want to we
                {
                    current_line = 0;
                    read_block++;
                    current_block_write++;
                    if (current_block_write == number_blocks)
                        current_block_write = 0;
                }
            }

            counter_internal_block++;
            if (counter_internal_block == (max_cycles - 1))
                counter_internal_block = 0;
        }
    } // End base_iter
}

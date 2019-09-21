/******************************************************************************
 * Copyright (c) 2016, Xilinx, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1.  Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2.  Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *
 * 3.  Neither the name of the copyright holder nor the names of its
 *     contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION). HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 * ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *****************************************************************************/

#define AP_INT_MAX_W 9216
#include <ap_int.h>

template<
// layer size
const unsigned int inRow,
const unsigned int inCol,
const unsigned int N,
const unsigned int outRow,
const unsigned int outCol,
const unsigned int K,
const unsigned int PRECISION_REB,
class frame_in_type,
class frame_out_type
>
void MaxPool(
  frame_in_type &frame_in,
  frame_out_type &frame_out
)
{

  // need buffer space for a single maxpooled row of the image
  ap_uint<PRECISION_REB*N> buf[outRow];
  for(unsigned int i = 0; i < outRow; i++) {
    #pragma HLS UNROLL
    buf[i] = 0;
  }

  
  //printf("inRow: %d, inCol: %d\n", inRow, inCol);
  //printf("outRow: %d, outCol: %d\n", outRow, outCol);
  //printf("K: %d\n", K);

  // control loop
  for (int tr=0; tr<outRow; tr++){
    for (int ti=0; ti<K; ti++){
      for (int tc=0; tc<outCol; tc++){
        #pragma HLS PIPELINE
        ap_uint<PRECISION_REB*N> acc = 0;
        for (int tj=0; tj<K; tj++){
          ap_uint<PRECISION_REB*N> input_buf = frame_in.read();
          for (int tn=0; tn<N; tn++){
            ap_uint<PRECISION_REB> acc_tmp = acc.range(PRECISION_REB*(tn+1)-1, PRECISION_REB*tn);
            ap_uint<PRECISION_REB> inp_tmp = input_buf.range(PRECISION_REB*(tn+1)-1, PRECISION_REB*tn);
            acc.range(PRECISION_REB*(tn+1)-1, PRECISION_REB*tn) = MAX(acc_tmp, inp_tmp);
            //if (tn==0) printf("%d ", (unsigned int)inp_tmp);
          }
        }
        // pool with old value in row buffer
        for (int tn=0; tn<N; tn++){
          ap_uint<PRECISION_REB> acc_tmp = acc.range(PRECISION_REB*(tn+1)-1, PRECISION_REB*tn);
          ap_uint<PRECISION_REB> buf_tmp = buf[tc].range(PRECISION_REB*(tn+1)-1, PRECISION_REB*tn);
          buf[tc].range(PRECISION_REB*(tn+1)-1, PRECISION_REB*tn) = MAX(acc_tmp, buf_tmp);
        }
      }
      //printf("\n ");
    }
    for (int tc_out = 0; tc_out < outCol; tc_out++) {
      #pragma HLS PIPELINE II=1
      frame_out.write(buf[tc_out]);
      // get buffer ready for next use
      buf[tc_out] = 0;
    }
  }
}






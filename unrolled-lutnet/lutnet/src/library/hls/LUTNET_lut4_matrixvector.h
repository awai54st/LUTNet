#define AP_INT_MAX_W 9216
#include <ap_int.h>

template<unsigned int MACS, unsigned int M>
void LUTARRAY_b0(ap_uint<MACS> in, ap_uint<MACS> in_1, ap_uint<MACS> in_2, ap_uint<MACS> in_3, ap_uint<MACS> *lut_out){
#pragma HLS inline off
        for (int tm = 0; tm < M; tm++) {
        #pragma HLS UNROLL
            lut_out[tm] = in ^ in_1 ^ in_2 ^ in_3;
        }
}

template<unsigned int MACS, unsigned int M>
void LUTARRAY_b1(ap_uint<MACS> in, ap_uint<MACS> in_1, ap_uint<MACS> in_2, ap_uint<MACS> in_3, ap_uint<MACS> *lut_out){
#pragma HLS inline off
        for (int tm = 0; tm < M; tm++) {
        #pragma HLS UNROLL
            lut_out[tm] = in ^ in_1 ^ in_2 ^ in_3;
        }
}

template<unsigned int PRECISION_REB, unsigned int MACS, unsigned int PopCountWidth, unsigned int PopCountIntWidth, unsigned int M>
ap_uint<PRECISION_REB*M> Naive4LUT(ap_uint<MACS> in[PRECISION_REB], ap_uint<MACS> in_1[PRECISION_REB], ap_uint<MACS> in_2[PRECISION_REB], ap_uint<MACS> in_3[PRECISION_REB], const ap_fixed<PopCountWidth, PopCountIntWidth> *thresh, const ap_fixed<PopCountWidth, PopCountIntWidth> *alpha, const ap_fixed<PopCountWidth, PopCountIntWidth> *next_layer_means, bool pt) {
        ap_uint<PRECISION_REB*M> tmp_out = 0;
        
        ap_fixed<PopCountWidth, PopCountIntWidth, AP_TRN, AP_SAT> accReg[M];
        #pragma HLS ARRAY_PARTITION variable=accReg complete dim=1
        const unsigned int max_fold = (MACS-1)/32+1;

        ap_uint<MACS> lut_out_b0[M];
        #pragma HLS ARRAY_PARTITION variable=lut_out_b0 complete dim=1
        LUTARRAY_b0<MACS, M>(in[0], in_1[0], in_2[0], in_3[0], lut_out_b0);

        ap_uint<MACS> lut_out_b1[M];
        #pragma HLS ARRAY_PARTITION variable=lut_out_b1 complete dim=1
        LUTARRAY_b1<MACS, M>(in[1], in_1[1], in_2[1], in_3[1], lut_out_b1);

        for (int tm = 0; tm < M; tm++) {
        #pragma HLS UNROLL
            accReg[tm] = 0;
            //for (int tb = 0; tb < PRECISION_REB; tb++){
            //    #pragma HLS UNROLL
                accReg[tm] += NaivePopCount<MACS, PopCountIntWidth>(lut_out_b1[tm]) * alpha[0];
                //if (tm==0){ 
                //    printf("%u ", (unsigned int)in.range(32*(fold+1)-1, 32*fold));
                //    printf("%u ", (unsigned int)weights[tm*max_fold+fold]);
                //    printf("%u ", (unsigned int)tmp_fold);
                //}
                
                accReg[tm] += NaivePopCount<MACS, PopCountIntWidth>(lut_out_b0[tm]) * alpha[1];


            //}

            //if(pt){
            //    printf("conv: ");
            //    printf("%.2f ", (float)accReg[tm]);
            //    printf("thres: ");
            //    printf("%.2f ", (float)thresh[tm]);
            //}

            ap_uint<PRECISION_REB> tmp_thresh;
            accReg[tm] -= thresh[tm];

            //if(tm==0){ 
            //    printf("%.2f ", (float)accReg[tm]);
            //    printf("%.2f ", (float)thresh[tm]);
            //}

            for (int tb = PRECISION_REB-1; tb >= 0; tb--){ // TODO: this loop only works for 2b at the moment
                tmp_thresh.range(tb, tb) = accReg[tm] > 0 ? 1 : 0; // MSB-first
                accReg[tm] = (accReg[tm] > 0) ? (accReg[tm] - next_layer_means[tm]) : (accReg[tm] + next_layer_means[tm]);
            }

            //if(pt){
            //    printf("%d", (int)tmp_thresh.range(0,0));
            //    printf("%d ", (int)tmp_thresh.range(1,1));
            //}
            //if(tm==0){ 
            //    printf("%d ", (unsigned int)tmp_thresh);
            //}

            tmp_out.range(PRECISION_REB*(tm+1)-1,PRECISION_REB*tm) = tmp_thresh;


        }
        //if(pt) printf("\n ");
	return tmp_out;
}


template<
// layer size
const unsigned int inRow,
const unsigned int inCol,
const unsigned int N,
const unsigned int outRow,
const unsigned int outCol,
const unsigned int M,
const unsigned int K,
const unsigned int ST,
const unsigned int PRECISION_REB,
const unsigned int PopCountWidth, // number of bits in popcount accumulator (>=log2(fanin))
const unsigned int PopCountIntWidth, // number of bits in popcount accumulator (>=log2(fanin))
class frame_in_type,
class frame_out_type
>
void LUTNET_LUT4MV(
  frame_in_type &frame_in,
  frame_out_type &frame_out,
  const ap_fixed<PopCountWidth, PopCountIntWidth> *thresh,
  const ap_fixed<PopCountWidth, PopCountIntWidth> *alpha,
  const ap_fixed<PopCountWidth, PopCountIntWidth> *next_layer_means,
  const unsigned int *rand_map_0,
  const unsigned int *rand_map_1,
  const unsigned int *rand_map_2
)
{

//#pragma HLS DATAFLOW

  ap_uint<PRECISION_REB*M> tmp_out = 0;
  ap_uint<PRECISION_REB*N*K*K> tmp_in = 0;

  ap_uint<N*K*K> tmp_in_reb [PRECISION_REB];
  #pragma HLS ARRAY_PARTITION variable=tmp_in_reb complete dim=1
  ap_uint<N*K*K> tmp_in_1_reb [PRECISION_REB];
  #pragma HLS ARRAY_PARTITION variable=tmp_in_1_reb complete dim=1
  ap_uint<N*K*K> tmp_in_2_reb [PRECISION_REB];
  #pragma HLS ARRAY_PARTITION variable=tmp_in_2_reb complete dim=1
  ap_uint<N*K*K> tmp_in_3_reb [PRECISION_REB];
  #pragma HLS ARRAY_PARTITION variable=tmp_in_3_reb complete dim=1

  bool pt = 1;

  // control loop
  for (int tr=0; tr<outRow; tr++){
    for (int tc=0; tc<outCol; tc++){
      for (int ti=0; ti<K; ti++){
        for (int tj=0; tj<K; tj++){
          #pragma HLS PIPELINE
          ap_uint<PRECISION_REB*N> input_buf = frame_in.read();
          //if (tr==0 & tc==0) printf("%lu ", (unsigned long int)input_buf);
          tmp_in.range(PRECISION_REB*N*(ti*K+tj+1)-1,PRECISION_REB*N*(ti*K+tj)) = input_buf;
          if (ti==K-1 && tj==K-1){
            for (int tb = 0; tb < PRECISION_REB; tb++){
                #pragma HLS UNROLL
                for (int t_mac = 0; t_mac < N*K*K; t_mac++){
                    #pragma HLS UNROLL
                    tmp_in_reb[tb].range(t_mac, t_mac) = tmp_in.range(t_mac*PRECISION_REB+tb, t_mac*PRECISION_REB+tb); // tmp_in_reb[0] is LSB
                }
                // perform randomisation on subsequent inputs
                for (int t_mac = 0; t_mac < N*K*K; t_mac++){
                    #pragma HLS UNROLL
                    tmp_in_1_reb[tb].range(t_mac, t_mac) = tmp_in_reb[tb].range(rand_map_0[t_mac], rand_map_0[t_mac]);
                    tmp_in_2_reb[tb].range(t_mac, t_mac) = tmp_in_reb[tb].range(rand_map_1[t_mac], rand_map_1[t_mac]);
                    tmp_in_3_reb[tb].range(t_mac, t_mac) = tmp_in_reb[tb].range(rand_map_2[t_mac], rand_map_2[t_mac]);
                }
            }
            if (tc==outCol-1) pt = 1;
            else pt = 0;
            tmp_out = Naive4LUT<PRECISION_REB, N*K*K, PopCountWidth, PopCountIntWidth, M> (tmp_in_reb, tmp_in_1_reb, tmp_in_2_reb, tmp_in_3_reb, thresh, alpha, next_layer_means, pt);
            frame_out.write(tmp_out);
          }
        }
      }
    }
  }
  //printf("\n ");
}

// Note: trying to hint the hls to generate parameters per channel (each channel is a new module) without going through extra codegen.






#define AP_INT_MAX_W 9216
#include <ap_int.h>

template<unsigned int PRECISION_IN, unsigned int PRECISION_REB, unsigned int PopCountWidth, unsigned int PopCountIntWidth, unsigned int MACS, unsigned int M, unsigned int TN, unsigned int TM>
ap_uint<PRECISION_REB*M> NaiveFXP(ap_uint<PRECISION_IN*MACS> in[TN], const ap_uint<32> weightMem[M/TM][TN][TM][(MACS-1)/32+1], const ap_fixed<PopCountWidth, PopCountIntWidth> *thresh, const ap_fixed<PopCountWidth, PopCountIntWidth> *alpha, const ap_fixed<PopCountWidth, PopCountIntWidth> *next_layer_means, bool pt) {
        ap_uint<PRECISION_REB*M> tmp_out = 0;
        ap_fixed<PopCountWidth, PopCountIntWidth, AP_TRN, AP_SAT> accReg[TM][M/TM];
        ap_fixed<PopCountWidth, PopCountIntWidth, AP_TRN, AP_SAT> accReg_tmp;
        const ap_fixed<PopCountWidth, PopCountIntWidth, AP_TRN, AP_SAT> accReg_zero = 0;
        #pragma HLS ARRAY_PARTITION variable=accReg complete dim=1
        #pragma HLS ARRAY_PARTITION variable=accReg complete dim=2
        #pragma HLS DEPENDENCE variable=accReg inter false
        #pragma HLS DEPENDENCE variable=accReg intra false
//        ap_uint<PRECISION_REB> tmp_thresh[TM][M/TM];
//        #pragma HLS ARRAY_PARTITION variable=tmp_thresh complete dim=1
//        #pragma HLS ARRAY_PARTITION variable=tmp_thresh complete dim=2

        const unsigned int max_fold = (MACS-1)/32+1;

        for (int tile_m = 0; tile_m < TM; tile_m++){
            #pragma HLS UNROLL
            for (int tm = 0; tm < M/TM; tm++){
                #pragma HLS UNROLL
                accReg[tile_m][tm] = 0;
            }
        }

        for (int tile_m = 0; tile_m < TM; tile_m++){
            for (int tile_n = 0; tile_n < TN; tile_n++){
            #pragma HLS PIPELINE II=1

                for (int tm = 0; tm < M/TM; tm++) {
                #pragma HLS UNROLL
                    for (int fold = 0; fold < max_fold; fold++){
                        #pragma HLS UNROLL
                        for (int i = 0; i < MIN(32, MACS); i++){
                            #pragma HLS UNROLL
                            ap_int<2> w = (weightMem[tm][tile_n][tile_m][fold].range(i, i)) ? 1 : -1;
                            ap_uint<PRECISION_IN> tmp = in[tile_n].range(PRECISION_IN*(32*fold+i+1)-1, PRECISION_IN*(32*fold+i));
                            ap_fixed<PRECISION_IN, 1, AP_TRN, AP_SAT> val = *reinterpret_cast<ap_fixed<PRECISION_IN, 1, AP_TRN, AP_SAT> *>(&tmp);
                            accReg_tmp = w * val * alpha[0];
                            //accReg[tm] += pruning_mask[tm*max_fold+fold].range(i, i) ? (accReg_tmp) : (accReg_zero);
                            //accReg[tile_m*M/TM+tm] += accReg_tmp;
                            accReg[tile_m][tm] += accReg_tmp;
                            //if (tile_m==0/*&&tile_n==0*/&&tm==0&&pt==1){
                            //    //printf("in: %.4f, w: %d  ", (float)val, (int)w);
                            //    //printf("in: %d, w: %d  ", (int)tmp, (int)w);
                            //}
                        }
                        //ap_uint<32> tmp_fold = (weights[tm*max_fold+fold] & in.range(8*32*(fold+1)-8, 8*32*fold)) | (~weights[tm*max_fold+fold] & ~in.range(8*32*(fold+1)-8, 8*32*fold));
                        //lut_out.range(MIN(32*(fold+1)-1, BITWIDTH-1), 32*fold) = tmp_fold;
                    }

                }

                if (tile_m==TM-1 && tile_n==TN-1){
                    for (int tile_mm = 0; tile_mm < TM; tile_mm++){
                    #pragma HLS UNROLL

                        for (int tm = 0; tm < M/TM; tm++){
                        #pragma HLS UNROLL
    
                            //if(tm==0){ 
                            //    printf("%.2f ", (float)accReg[tm]);
                            //}
    
                            ap_uint<PRECISION_REB> tmp_thresh;
                            accReg[tile_mm][tm] -= thresh[tile_mm*M/TM+tm];
    
                            for (int tb = PRECISION_REB-1; tb >= 0; tb--){ // TODO: this loop only works for 2b at the moment
                                tmp_thresh.range(tb, tb) = accReg[tile_mm][tm] > 0 ? 1 : 0; // MSB-first
                                accReg[tile_mm][tm] = (accReg[tile_mm][tm] > 0) ? (accReg[tile_mm][tm] - next_layer_means[tile_m*M/TM+tm]) : (accReg[tile_mm][tm] + next_layer_means[tile_m*M/TM+tm]);
                            }
    
                            tmp_out.range(PRECISION_REB*(tile_mm*M/TM+tm+1)-1,PRECISION_REB*(tile_mm*M/TM+tm)) = tmp_thresh;
                        }
                    }
                }
            }
        }

//        ap_uint<PRECISION_REB> tmp_thresh[M];
//        #pragma HLS ARRAY_PARTITION variable=tmp_thresh complete dim=1
//        for (int tile_m = 0; tile_m < TM; tile_m++){
//            #pragma HLS UNROLL
//
//            for (int tm = 0; tm < M/TM; tm++){
//            #pragma HLS UNROLL
////            #pragma HLS DEPENDENCE variable=tmp_out inter false
////            #pragma HLS DEPENDENCE variable=tmp_out intra false
//
//                //if(tm==0){ 
//                //    printf("%.2f ", (float)accReg[tm]);
//                //}
//
//                accReg[tile_m][tm] -= thresh[tile_m*M/TM+tm];
//
//                for (int tb = PRECISION_REB-1; tb >= 0; tb--){ // TODO: this loop only works for 2b at the moment
//                    tmp_thresh[tile_m*M/TM+tm].range(tb, tb) = accReg[tile_m][tm] > 0 ? 1 : 0; // MSB-first
//                    accReg[tile_m][tm] = (accReg[tile_m][tm] > 0) ? (accReg[tile_m][tm] - next_layer_means[tile_m*M/TM+tm]) : (accReg[tile_m][tm] + next_layer_means[tile_m*M/TM+tm]);
//                }
//
//            }
//        }
//        for (int tm = 0; tm < M; tm++){
//        #pragma HLS UNROLL
//            tmp_out.range(PRECISION_REB*(tm+1)-1,PRECISION_REB*tm) = tmp_thresh[tm];
//        }


        //if(pt) printf("\n ");
        //printf("\n");
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
const unsigned int PRECISION, // number of bits each input
const unsigned int PRECISION_REB, // number of bits for each binary input
const unsigned int PopCountWidth, // number of bits in popcount accumulator
const unsigned int PopCountIntWidth, // number of integer bits in popcount accumulator
const unsigned int TN,// number of tiles across ch_in dimension
const unsigned int TM,// number of tiles across ch_out dimension
class frame_in_type,
class frame_out_type
>
void FXPMV(
  frame_in_type &frame_in,
  frame_out_type &frame_out,
  //const ap_uint<32> *weights,
  const ap_uint<32> weightMem[M/TM][TN][TM][(N*K*K/TN-1)/32+1],
  //const ap_uint<32> *pruning_mask,
  const ap_fixed<PopCountWidth, PopCountIntWidth> *thresh,
  const ap_fixed<PopCountWidth, PopCountIntWidth> *alpha,
  const ap_fixed<PopCountWidth, PopCountIntWidth> *next_layer_means
)
{

//#pragma HLS DATAFLOW

  ap_uint<PRECISION_REB*M> tmp_out = 0;
  ap_uint<PRECISION*N*K*K> tmp_in = 0;

  ap_uint<N*K*K/TN*PRECISION> tmp_in_reb [TN];
  #pragma HLS ARRAY_PARTITION variable=tmp_in_reb complete dim=1

  bool pt = 1;

  // control loop
  for (int tr=0; tr<outRow; tr++){
    for (int tc=0; tc<outCol; tc++){
      for (int ti=0; ti<K; ti++){
        for (int tj=0; tj<K; tj++){
          #pragma HLS PIPELINE
          ap_uint<PRECISION*N> input_buf = frame_in.read();
          //printf("in: %d", (char)input_buf.range(PRECISION-1,0));
          tmp_in.range(PRECISION*N*(ti*K+tj+1)-1,PRECISION*N*(ti*K+tj)) = input_buf;
          if (ti==K-1 && tj==K-1){
//            for (int tb = 0; tb < PRECISION; tb++){
//                #pragma HLS UNROLL
                for (int t_ch_in = 0; t_ch_in < N/TN*PRECISION; t_ch_in++){
                    #pragma HLS UNROLL
                    for (int t_tn = 0; t_tn < TN; t_tn++){
                        #pragma HLS UNROLL
                        for (int t_kk = 0; t_kk < K*K; t_kk++){
                            #pragma HLS UNROLL
                            tmp_in_reb[t_tn].range(t_kk*N/TN*PRECISION+t_ch_in, t_kk*N/TN*PRECISION+t_ch_in) = tmp_in.range(t_kk*N*PRECISION+t_tn*N/TN*PRECISION+t_ch_in, t_kk*N*PRECISION+t_tn*N/TN*PRECISION+t_ch_in); // tmp_in_reb[0] is LSB
                        }
                    }
                }
                if (tr==0 && tc==0) {
                    pt = 1;
                    //printf("in1: %d", (unsigned int)tmp_in);
                    //printf("in2: %d", (unsigned int)tmp_in_reb[0]);
                }
                else pt = 0;
//            }
          }
        }
      }
      tmp_out = NaiveFXP<PRECISION, PRECISION_REB, PopCountWidth, PopCountIntWidth, N*K*K/TN, M> (tmp_in_reb, weightMem, thresh, alpha, next_layer_means, pt);
      frame_out.write(tmp_out);
    }
  }
  //printf("\n ");
}

// Note: trying to hint the hls to generate parameters per channel (each channel is a new module) without going through extra codegen.






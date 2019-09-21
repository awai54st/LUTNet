#pragma once


/***************************************
 **     Data type
 ***************************************/
#define AP_INT_MAX_W 9216
#include <ap_int.h>
#include <hls_stream.h>
#include <ap_axi_sdata.h>
#ifndef bitwidth
#define bitwidth 1
#define PopCountWidth_val 24
#define PopCountIntWidth_val 16
#endif

#ifndef data_t
#define data_t ap_uint<bitwidth>
#define data_t_L1 ap_uint<L1_BW*L1_N>
#define data_t_L2 ap_uint<L2_BW*L2_N>
#define data_t_L3 ap_uint<L3_BW*L3_N>
#define data_t_L4 ap_uint<L4_BW*L4_N>
#define data_t_L5 ap_uint<L5_BW*L5_N>
#define data_t_L6 ap_uint<L6_BW*L6_N>
#define data_t_L7 ap_uint<L7_BW*L7_N>
#define data_t_L8 ap_uint<L8_BW*L8_N>
#define data_t_L9 ap_uint<L9_BW*L9_N>
#define data_t_L10 ap_uint<L10_BW*L10_N>
#define data_t_L11 ap_uint<L11_BW*L11_N>
#define data_t_out ap_uint<OutputWidth_val*L11_M>
#endif

/***************************************
 * Basic operations
 ***************************************/
// inlined
//inline int min(int a, int b) { return (a<b) ? a : b;}
//inline int max(int a, int b) { return (a>b) ? a : b;}
//inline data_t max_bn(data_t a, data_t b) { return (a>b) ? a : b;}

/********************************************
 **	Layer 1: FullConv
 ********************************************/
// convolution parameters
#define L1_BW 8
#define L1_K 3 // kernel window size
#define L1_ST 1 //stride size
// input frame
#define L1_iR 32
#define L1_iC 32
#define L1_N 3
// output frame
#define L1_oR (L1_iR-2)
#define L1_oC (L1_iC-2)
#define L1_M 64
// tiling info
#define L1_TN 3
#define L1_TM 16

/********************************************
 **	Layer 2: FullConv
 ********************************************/
//// convolution parameters
//#define L2_BW 2
//#define L2_K 3 // kernel window size
//#define L2_ST 1 //stride size
//// input frame
//#define L2_iR L1_oR
//#define L2_iC L1_oC
//#define L2_N 16
//// output frame
//#define L2_oR (L2_iR-2)
//#define L2_oC (L2_iC-2)
//#define L2_M 16

// convolution parameters
#define L2_BW 2
#define L2_K 3 // kernel window size
#define L2_ST 1 //stride size
// input frame
#define L2_iR L1_oR
#define L2_iC L1_oC
#define L2_N L1_M
// output frame
#define L2_oR (L2_iR-2)
#define L2_oC (L2_iC-2)
#define L2_M 64
// tiling info
#define L2_TN 16
#define L2_TM 16


/********************************************
 **	Layer 3: MaxPool
 ********************************************/
// pooling parameters
#define L3_BW 2
#define L3_K 2 // kernel window size
#define L3_ST L3_K //stride size
// input frame
#define L3_iR L2_oR
#define L3_iC L2_oC
#define L3_N L2_M
// output frame
#define L3_oR (L3_iR / L3_K)
#define L3_oC (L3_iC / L3_K)
#define L3_M L3_N
// tiling info
#define L3_TN 16
#define L3_TM 16


/********************************************
 **	Layer 4: FullConv
 ********************************************/
// convolution parameters
#define L4_BW 2
#define L4_K 3 // kernel window size
#define L4_ST 1 //stride size
// input frame
#define L4_iR L3_oR
#define L4_iC L3_oC
#define L4_N L3_M
// output frame
#define L4_oR (L4_iR-2)
#define L4_oC (L4_iC-2)
#define L4_M 128
// tiling info
#define L4_TN 16
#define L4_TM 16


/********************************************
 **	Layer 5: FullConv
 ********************************************/
// convolution parameters
#define L5_BW 2
#define L5_K 3 // kernel window size
#define L5_ST 1 //stride size
// input frame
#define L5_iR L4_oR
#define L5_iC L4_oC
#define L5_N L4_M
// output frame
#define L5_oR (L5_iR-2)
#define L5_oC (L5_iC-2)
#define L5_M 128
// tiling info
#define L5_TN 16
#define L5_TM 16


/********************************************
 **	Layer 6: MaxPool
 ********************************************/
// pooling parameters
#define L6_BW 2
#define L6_K 2 // kernel window size
#define L6_ST L6_K //stride size
// input frame
#define L6_iR L5_oR
#define L6_iC L5_oC
#define L6_N L5_M
// output frame
#define L6_oR (L6_iR / L6_K)
#define L6_oC (L6_iC / L6_K)
#define L6_M L6_N
// tiling info
#define L6_TN 16
#define L6_TM 16


/********************************************
 **	Layer 7: FullConv
 ********************************************/
// convolution parameters
#define L7_BW 2
#define L7_K 3 // kernel window size
#define L7_ST 1 //stride size
// input frame
#define L7_iR L6_oR
#define L7_iC L6_oC
#define L7_N L6_M
// output frame
#define L7_oR (L7_iR-2)
#define L7_oC (L7_iC-2)
#define L7_M 256
// tiling info
#define L7_TN 16
#define L7_TM 16


/********************************************
 **	Layer 8: FullConv
 ********************************************/
// convolution parameters
#define L8_BW 2
#define L8_K 3 // kernel window size
#define L8_ST 1 //stride size
// input frame
#define L8_iR L7_oR
#define L8_iC L7_oC
#define L8_N L7_M
// output frame
#define L8_oR (L8_iR-2)
#define L8_oC (L8_iC-2)
#define L8_M 256
// tiling info
#define L8_TN 16
#define L8_TM 16


/********************************************
 **	Layer 9: FullyConnected
 ********************************************/
// fc parameters
#define L9_BW 2
#define L9_K 1 // kernel window size
#define L9_ST 1 //stride size
// input frame
#define L9_iR 1
#define L9_iC 1
#define L9_N (L8_M*L8_oR*L8_oC)
// output frame
#define L9_oR 1
#define L9_oC 1
#define L9_M 512
// tiling info
#define L9_TN 16
#define L9_TM 16


/********************************************
 **	Layer 10: FullyConnected
 ********************************************/
// fc parameters
#define L10_BW 2
#define L10_K 1 // kernel window size
#define L10_ST 1 //stride size
// input frame
#define L10_iR 1
#define L10_iC 1
#define L10_N (L9_M*L9_oR*L9_oC)
// output frame
#define L10_oR 1
#define L10_oC 1
#define L10_M 512
// tiling info
#define L10_TN 16
#define L10_TM 16


/********************************************
 **	Layer 11: FullyConnected
 ********************************************/
// fc parameters
#define L11_BW 2
#define L11_K 1 // kernel window size
#define L11_ST 1 //stride size
// input frame
#define L11_iR 1
#define L11_iC 1
#define L11_N (L10_M*L10_oR*L10_oC)
// output frame
#define L11_oR 1
#define L11_oC 1
#define L11_M 10
// tiling info
#define L11_TN 16
#define L11_TM 10

#define OutputWidth_val 24

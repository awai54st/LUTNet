
#define AP_INT_MAX_W 9216
#include <ap_int.h>

template<unsigned int InWidth,		// width of input stream
		unsigned int OutWidth,		// width of output stream
		unsigned int NumInWords,	// number of input words to process
		unsigned int NumRes         // number of residual levels used
>
void LUTNET_StreamingNumResConverter(stream<ap_uint<InWidth> > & in, stream<ap_uint<OutWidth> > & out) {
	if (InWidth > OutWidth) {
		CASSERT_DATAFLOW(InWidth % OutWidth == 0);
		ap_uint<InWidth> ei;
		ap_uint<OutWidth> eo[NumRes];
		for (unsigned int t = 0; t < NumInWords; t++) {
		#pragma HLS PIPELINE II=1
			ei = in.read();
			for(int res_ind=0;res_ind<NumRes;res_ind++)
			{
			#pragma HLS UNROLL
				for(int i=0; i<OutWidth; i++)
				{
					eo[res_ind].range(i,i) = ei.range(NumRes*i+res_ind, NumRes*i+res_ind);
				}
			}
			for(int res_ind=NumRes-1; res_ind>=0; res_ind--)
			{
				out.write(eo[res_ind]);
			}
		}
	}
	else if (InWidth == OutWidth)
	{
		{
			#pragma HLS PIPELINE II=1
			ap_uint<InWidth> e = in.read();
			out.write(e);
		}

	}
	else
	{
		CASSERT_DATAFLOW(OutWidth % InWidth == 0);
		ap_uint<OutWidth> eo;
		for (unsigned int t = 0; t < NumInWords; t++)
		{
		#pragma HLS PIPELINE II=1
			for(int res_ind=NumRes-1;res_ind>=0;res_ind--)
			{
				ap_uint<InWidth> ei = in.read();
				for(int i=0; i<InWidth; i++)
				{
					eo.range(NumRes*i+res_ind, NumRes*i+res_ind) = ei.range(i,i);
				}
				if (res_ind==0) out.write(eo);
			}
		}
	}
}


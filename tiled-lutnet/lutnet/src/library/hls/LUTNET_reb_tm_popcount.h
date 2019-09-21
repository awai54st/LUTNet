#define AP_INT_MAX_W 9216
#include <ap_int.h>

template<unsigned int MACS, unsigned int PopCountIntWidth>
ap_uint<PopCountIntWidth> NaivePopCount(ap_uint<MACS> in) {
        ap_uint<PopCountIntWidth> pct = 0;
        for (unsigned int i = 0; i < MACS; i++) {
                pct += in(i, i);
        }
        return pct;
}

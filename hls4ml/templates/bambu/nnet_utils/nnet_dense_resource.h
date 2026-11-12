#ifndef NNET_DENSE_RESOURCE_H_
#define NNET_DENSE_RESOURCE_H_

#include "hls_stream.h"
#include "nnet_common.h"
#include "nnet_mult.h"
#include <assert.h>
#include <math.h>

namespace nnet {

template <class data_T, class res_T, typename CONFIG_T>
void dense_resource_rf_leq_nin(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_out],
                               typename CONFIG_T::weight_t weights[CONFIG_T::n_in * CONFIG_T::n_out],
                               typename CONFIG_T::bias_t biases[CONFIG_T::n_out]) {

    const int rufactor = CONFIG_T::reuse_factor;
    const int multfactor = MIN(CONFIG_T::n_in, CONFIG_T::reuse_factor);
    const int multiplier_limit = DIV_ROUNDUP(CONFIG_T::n_in * CONFIG_T::n_out, multfactor);
    const int block_factor = DIV_ROUNDUP(CONFIG_T::n_in * CONFIG_T::n_out, CONFIG_T::reuse_factor);
    const int multscale = multiplier_limit / CONFIG_T::n_out;
    const int nin = CONFIG_T::n_in;
    const int nout = CONFIG_T::n_out;

    assert((multiplier_limit % nout == 0 || rufactor >= nin) && "The current Reuse Factor is not allowed");
    assert((multiplier_limit == block_factor) && "This function is correct only for RF <= N_IN");

    //#pragma HLS function_instantiate variable=weights,biases
    //#pragma HLS ARRAY_RESHAPE   variable=weights block factor=block_factor
    #pragma HLS ARRAY_PARTITION variable=biases complete

    if (CONFIG_T::reuse_factor > 1) {
        //#pragma HLS RESOURCE variable=weights core=ROM_nP_BRAM
    }

    typename CONFIG_T::accum_t acc[CONFIG_T::n_out];
    #pragma HLS ARRAY_PARTITION variable=acc complete

InitAccum:
    #pragma clang loop unroll(full)
    for (int iacc = 0; iacc < nout; iacc++) {
        //#pragma HLS UNROLL
        acc[iacc] = (typename CONFIG_T::accum_t)biases[iacc];
    }

    // An iteration ir reads data[ir + rufactor * k], k = 0 .. nin / rufactor - 1 (a valid RF <= n_in divides n_in).
    // With data partitioned, bambu reads data[ir + rufactor * k] through a pointer that can point to several
    // partitions, and so through the memory bus, which the dataflow modules share without arbitration: two convs
    // reading in the same cycle get each other's data. data is copied with constant indices into nin / rufactor
    // banks of rufactor elements, and data_b[k][ir], with k a constant once MultLoop is unrolled, is a direct read.
    const int nbanks = nin / rufactor;
    data_T data_b[CONFIG_T::n_in / CONFIG_T::reuse_factor][CONFIG_T::reuse_factor];
    #pragma HLS ARRAY_PARTITION variable=data_b complete dim=1
BankData:
    #pragma clang loop unroll(full)
    for (int i = 0; i < nin; i++) {
        data_b[i / rufactor][i % rufactor] = data[i];
    }

ReuseLoop:
    for (int ir = 0; ir < rufactor; ir++) {
        //#pragma HLS PIPELINE II=1 rewind

        int w_index = ir;
        int in_bank = 0;
        int out_index = 0;
        int acc_step = 0;

    MultLoop:
        #pragma clang loop unroll(full)
        for (int im = 0; im < block_factor; im++) {
            //#pragma HLS UNROLL

            // weights[w_index], w_index = im * rufactor + ir, is in bank im of the block partition of the weights.
            // CSROA proves it only while the reuse loop has no more iterations than there are banks; past that the
            // bank is picked at run time and the read goes through the memory bus too, so the weight is chosen among
            // reads with constant indices instead.
            typename CONFIG_T::weight_t weight;
            if (rufactor <= block_factor) {
                weight = weights[w_index];
            } else {
                weight = weights[im * rufactor];
            WeightSelect:
                #pragma clang loop unroll(full)
                for (int r = 1; r < rufactor; r++) {
                    if (r == ir) {
                        weight = weights[im * rufactor + r];
                    }
                }
            }

            acc[out_index] += static_cast<typename CONFIG_T::accum_t>(
                CONFIG_T::template product<data_T, typename CONFIG_T::weight_t>::product(data_b[in_bank][ir], weight));

            // Increment w_index
            w_index += rufactor;
            // Next bank (in_index += rufactor, back to ir past n_in)
            if (++in_bank == nbanks) {
                in_bank = 0;
            }
            // Increment out_index
            if (acc_step + 1 >= multscale) {
                acc_step = 0;
                out_index++;
            } else {
                acc_step++;
            }
        }
    }

// Cast to "res_t" type
Result:
    #pragma clang loop unroll(full)
    for (int ires = 0; ires < CONFIG_T::n_out; ires++) {
        //#pragma HLS UNROLL
        res[ires] = cast<data_T, res_T, CONFIG_T>(acc[ires]);
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void dense_resource_rf_gt_nin_rem0(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_out],
                                   typename CONFIG_T::weight_t weights[CONFIG_T::n_in * CONFIG_T::n_out],
                                   typename CONFIG_T::bias_t biases[CONFIG_T::n_out]) {

    const int rufactor = MIN(CONFIG_T::reuse_factor, CONFIG_T::n_in * CONFIG_T::n_out);
    const int multfactor = MIN(CONFIG_T::n_in, CONFIG_T::reuse_factor);
    const int multiplier_limit = DIV_ROUNDUP(CONFIG_T::n_in * CONFIG_T::n_out, multfactor);
    const int block_factor = DIV_ROUNDUP(CONFIG_T::n_in * CONFIG_T::n_out, CONFIG_T::reuse_factor);
    const int multscale = multiplier_limit / CONFIG_T::n_out;
    const int nin = CONFIG_T::n_in;
    const int nout = CONFIG_T::n_out;

    assert((multiplier_limit % nout == 0 || rufactor >= nin) && "The current Reuse Factor is not allowed");
    assert((rufactor > nin && rufactor % nin == 0) && "This function is correct only for RF > N_IN && RF % N_IN == 0");

    //#pragma HLS function_instantiate variable=weights,biases
    //#pragma HLS ARRAY_RESHAPE   variable=weights block factor=block_factor
    #pragma HLS ARRAY_PARTITION variable=biases complete

    if (CONFIG_T::reuse_factor > 1) {
        //#pragma HLS RESOURCE variable=weights core=ROM_nP_BRAM
    }

    typename CONFIG_T::accum_t acc[CONFIG_T::n_out];
    #pragma HLS ARRAY_PARTITION variable=acc complete

InitAccum:
    #pragma clang loop unroll(full)
    for (int iacc = 0; iacc < nout; iacc++) {
        //#pragma HLS UNROLL
        acc[iacc] = (typename CONFIG_T::accum_t)biases[iacc];
    }

    int w_index;
    int in_index = 0;
    int out_index;
    int outstep = 0;
    const int outscale = rufactor / nin;

    int outidx[rufactor];
IndexLoop:
    for (int ir = 0; ir < rufactor; ir++) {
        outidx[ir] = outstep;
        if ((ir + 1) % nin == 0) {
            outstep++;
        }
    }

ReuseLoop:
    for (int ir = 0; ir < rufactor; ir++) {
        //#pragma HLS PIPELINE II=1 rewind

        w_index = ir;
        out_index = outidx[ir] /*outstep*/;

    MultLoop:
        #pragma clang loop unroll(full)
        for (int im = 0; im < block_factor; im++) {
            //#pragma HLS UNROLL
            acc[out_index] += static_cast<typename CONFIG_T::accum_t>(
                CONFIG_T::template product<data_T, typename CONFIG_T::weight_t>::product(data[in_index], weights[w_index]));

            w_index += rufactor;
            if (w_index >= CONFIG_T::n_in * CONFIG_T::n_out)
                break; // check out of bounds
            out_index += outscale;
        }

        in_index++;
        if (in_index >= nin) {
            in_index = 0;
            // outstep++; // This causes a huge increase in scheduling and RTL generation times, hence the above workaround.
        }
    }

// Cast to "res_t" type
Result:
    #pragma clang loop unroll(full)
    for (int ires = 0; ires < CONFIG_T::n_out; ires++) {
        //#pragma HLS UNROLL
        res[ires] = cast<data_T, res_T, CONFIG_T>(acc[ires]);
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void dense_resource_rf_gt_nin(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_out],
                              typename CONFIG_T::weight_t weights[CONFIG_T::n_in * CONFIG_T::n_out],
                              typename CONFIG_T::bias_t biases[CONFIG_T::n_out]) {

    const int rufactor = CONFIG_T::reuse_factor;
    const int multfactor = MIN(CONFIG_T::n_in, CONFIG_T::reuse_factor);
    const int multiplier_limit = DIV_ROUNDUP(CONFIG_T::n_in * CONFIG_T::n_out, multfactor);
    const int block_factor = DIV_ROUNDUP(CONFIG_T::n_in * CONFIG_T::n_out, CONFIG_T::reuse_factor);
    const int multscale = multiplier_limit / CONFIG_T::n_out;
    const int nin = CONFIG_T::n_in;
    const int nout = CONFIG_T::n_out;

    assert((multiplier_limit % nout == 0 || rufactor >= nin) && "The current Reuse Factor is not allowed");
    assert((rufactor > nin) && "This function is correct only for RF > N_IN");

    //#pragma HLS function_instantiate variable=weights,biases
    //#pragma HLS ARRAY_RESHAPE   variable=weights block factor=block_factor
    #pragma HLS ARRAY_PARTITION variable=biases complete

    if (CONFIG_T::reuse_factor > 1) {
        //#pragma HLS RESOURCE variable=weights core=ROM_nP_BRAM
    }

    typename CONFIG_T::accum_t acc[CONFIG_T::n_out];
    #pragma HLS ARRAY_PARTITION variable=acc complete

InitAccum:
    #pragma clang loop unroll(full)
    for (int iacc = 0; iacc < nout; iacc++) {
        //#pragma HLS UNROLL
        acc[iacc] = (typename CONFIG_T::accum_t)biases[iacc];
    }

ReuseLoop:
    for (int ir = 0; ir < rufactor; ir++) {
        //#pragma HLS PIPELINE II=1 rewind
        typename CONFIG_T::accum_t tmpmult[block_factor];
        #pragma HLS ARRAY_PARTITION variable=tmpmult complete

    MultLoop:
        #pragma clang loop unroll(full)
        for (int im = 0; im < block_factor; im++) {
            //#pragma HLS UNROLL
            int w_index = ir + rufactor * im;
            int in_index = w_index % nin;
            if (w_index >= CONFIG_T::n_in * CONFIG_T::n_out)
                continue; // check out of bounds
            tmpmult[im] =
                CONFIG_T::template product<data_T, typename CONFIG_T::weight_t>::product(data[in_index], weights[w_index]);
        }

        typename CONFIG_T::accum_t mult[multiplier_limit];
        #pragma HLS ARRAY_PARTITION variable=mult complete

    ResetMult:
        #pragma clang loop unroll(full)
        for (int imult = 0; imult < multiplier_limit; imult++) {
            //#pragma HLS UNROLL
            mult[imult] = 0;
        }

    AccumLoop1:
        #pragma clang loop unroll(full)
        for (int im = 0; im < block_factor; im++) {
            //#pragma HLS UNROLL
            int w_index = ir + rufactor * im;
            int out_index = w_index / multfactor;
            if (out_index >= multiplier_limit)
                continue; // check out of bounds
            mult[out_index] += tmpmult[im];
        }

    AccumLoop2:
        #pragma clang loop unroll(full)
        for (int im = 0; im < multiplier_limit; im++) {
            //#pragma HLS UNROLL
            // int out_index = im/multscale; // This is the general case
            // acc[out_index] += mult[im];
            acc[im] += mult[im]; // If RF > N_IN then multiplier_limit == n_out
        }
    }

// Cast to "res_t" type
Result:
    #pragma clang loop unroll(full)
    for (int ires = 0; ires < CONFIG_T::n_out; ires++) {
        //#pragma HLS UNROLL
        res[ires] = cast<data_T, res_T, CONFIG_T>(acc[ires]);
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void dense_resource(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_out],
                    typename CONFIG_T::weight_t weights[CONFIG_T::n_in * CONFIG_T::n_out],
                    typename CONFIG_T::bias_t biases[CONFIG_T::n_out]) {

    #pragma HLS inline recursive

    if (CONFIG_T::reuse_factor <= CONFIG_T::n_in) {
        dense_resource_rf_leq_nin<data_T, res_T, CONFIG_T>(data, res, weights, biases);
    } else if (CONFIG_T::reuse_factor % CONFIG_T::n_in == 0) {
        dense_resource_rf_gt_nin_rem0<data_T, res_T, CONFIG_T>(data, res, weights, biases);
    } else {
        dense_resource_rf_gt_nin<data_T, res_T, CONFIG_T>(data, res, weights, biases);
    }
}

} // namespace nnet

#endif

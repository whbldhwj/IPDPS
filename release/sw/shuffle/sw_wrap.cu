#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <string.h>
#include <sys/time.h>
#include <ctime>
#include "params.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

extern __global__ void calMatrix(
    int refNum,
    int p_w_match, int p_w_mismatch, int p_w_open, int p_w_extend,
    char *ref, char *alt,
    short *refLen, short *altLen, short *strategy,
    btrack_t *btrack,
    int *sw_lastrow,
    int *sw_lastcol,
    int *best_gap_v,
    char *gap_size_v
);

extern __global__ void calCigar(
    int refNum,
    btrack_t *btrack,
    int *sw_lastrow,
    int *sw_lastcol,
    short *refLen, short *altLen, short *strategy,
    short *state, short *segmentLen, short *offset, short *num
);

float sw_wrap(
    int refNum,
    char *ref, char *alt,
    short *refLen, short *altLen,
    short *strategy,
    short *state, short *segmentLen, short *offset, short *num
){
//    int *sw_lastrow = (int*)malloc(sizeof(int)*PADDED_ALT_LEN*refNum);
//    int *sw_lastcol = (int*)malloc(sizeof(int)*PADDED_REF_LEN*refNum);

//    btrack_t *btrack = (btrack_t*)malloc(sizeof(btrack_t)*PADDED_REF_LEN*PADDED_ALT_LEN*refNum);
    int *best_gap_v = (int*)malloc(sizeof(int)*PADDED_ALT_LEN*refNum);
    for(int i=0; i<refNum; i++){
        for(int j=0; j<PADDED_ALT_LEN; j++){
            best_gap_v[i*PADDED_ALT_LEN + j] = lowInitValue;
        }
    }

//    short refMaxLen = 0;
//    short altMaxLen = 0;
//    for (int i=0; i<refNum; i++){
//        if (refLen[i] > refMaxLen)
//            refMaxLen = refLen[i];
//        if (altLen[i] > altMaxLen)
//            altMaxLen = altLen[i];
//    }
    int *sw_lastrow_cuda;
    int *sw_lastcol_cuda;
    btrack_t *btrack_cuda;
    char *ref_cuda;
    char *alt_cuda;
    short *refLen_cuda;
    short *altLen_cuda;
    short *strategy_cuda;

    int *best_gap_v_cuda;
    char *gap_size_v_cuda;
    
    cudaMalloc((void**)& sw_lastrow_cuda, PADDED_ALT_LEN*refNum*sizeof(int));
    cudaMalloc((void**)& sw_lastcol_cuda, PADDED_REF_LEN*refNum*sizeof(int));
    cudaMalloc((void**)& btrack_cuda, PADDED_REF_LEN*PADDED_ALT_LEN*refNum*sizeof(btrack_t));
    cudaMalloc((void**)& ref_cuda, MAX_REF_LEN*refNum*sizeof(char));
    cudaMalloc((void**)& alt_cuda, MAX_ALT_LEN*refNum*sizeof(char));
    cudaMalloc((void**)& refLen_cuda, refNum*sizeof(short));
    cudaMalloc((void**)& altLen_cuda, refNum*sizeof(short));
    cudaMalloc((void**)& strategy_cuda, refNum*sizeof(short));

    cudaMalloc((void**)& best_gap_v_cuda, refNum*PADDED_ALT_LEN*sizeof(int));
    cudaMalloc((void**)& gap_size_v_cuda, refNum*PADDED_ALT_LEN*sizeof(char));

    cudaEvent_t start, stop;
    float elapsedTime1;
    float elapsedTime2;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    cudaMemset(sw_lastrow_cuda, 0, PADDED_ALT_LEN*refNum*sizeof(int));
    cudaMemset(sw_lastcol_cuda, 0, PADDED_REF_LEN*refNum*sizeof(int));
    cudaMemcpy(ref_cuda, ref, MAX_REF_LEN*refNum*sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(alt_cuda, alt, MAX_ALT_LEN*refNum*sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(refLen_cuda, refLen, refNum*sizeof(short), cudaMemcpyHostToDevice);
    cudaMemcpy(altLen_cuda, altLen, refNum*sizeof(short), cudaMemcpyHostToDevice);
    cudaMemcpy(strategy_cuda, strategy, refNum*sizeof(short), cudaMemcpyHostToDevice);
    
    cudaMemcpy(best_gap_v_cuda, best_gap_v, refNum*PADDED_ALT_LEN*sizeof(int), cudaMemcpyHostToDevice);

    dim3 dimGrid(refNum, 1);
    dim3 dimBlock(BLOCK_SIZE,1);

//    cudaEventRecord(start, 0);

    calMatrix<<<dimGrid, dimBlock>>>(
        refNum,
        200, -150, -260, -11,
        ref_cuda, alt_cuda,
        refLen_cuda, altLen_cuda, strategy_cuda,
        btrack_cuda,
        sw_lastrow_cuda,
        sw_lastcol_cuda,
        best_gap_v_cuda, 
        gap_size_v_cuda
    );
//    cudaEventRecord(stop, 0);
//    cudaEventSynchronize(stop);

//    cudaMemcpy(sw_lastrow, sw_lastrow_cuda, sizeof(int)*PADDED_ALT_LEN*refNum, cudaMemcpyDeviceToHost);
//    cudaMemcpy(sw_lastcol, sw_lastcol_cuda, sizeof(int)*PADDED_REF_LEN*refNum, cudaMemcpyDeviceToHost);
//    cudaMemcpy(btrack, btrack_cuda, sizeof(btrack_t)*PADDED_REF_LEN*PADDED_ALT_LEN*refNum, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime1, start, stop);
   
    short *state_cuda, *segmentLen_cuda, *offset_cuda, *num_cuda;
    cudaMalloc((void**)& state_cuda, refNum*MAX_STATE_NUM*sizeof(short));
    cudaMalloc((void**)& segmentLen_cuda, refNum*MAX_STATE_NUM*sizeof(short));
    cudaMalloc((void**)& offset_cuda, refNum*sizeof(short));
    cudaMalloc((void**)& num_cuda, refNum*sizeof(short));

    cudaEventRecord(start, 0);

//    dim3 dimGrid(refNum, 1);
//    dim3 dimBlock(1, 1);
    dimBlock.x = 1;
    calCigar<<<dimGrid, dimBlock>>>(  
        refNum,
        btrack_cuda,
        sw_lastrow_cuda,
        sw_lastcol_cuda,
        refLen_cuda, altLen_cuda, strategy_cuda,
        state_cuda, segmentLen_cuda, offset_cuda, num_cuda
    );

    cudaMemcpy(state, state_cuda, sizeof(short)*MAX_STATE_NUM*refNum, cudaMemcpyDeviceToHost);
    cudaMemcpy(segmentLen, segmentLen_cuda, sizeof(short)*MAX_STATE_NUM*refNum, cudaMemcpyDeviceToHost);
    cudaMemcpy(offset, offset_cuda, sizeof(short)*refNum, cudaMemcpyDeviceToHost);
    cudaMemcpy(num, num_cuda, sizeof(short)*refNum, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
   
    cudaEventElapsedTime(&elapsedTime2, start, stop);

    cudaFree(sw_lastrow_cuda);
    cudaFree(sw_lastcol_cuda);
    cudaFree(btrack_cuda);
    cudaFree(ref_cuda);
    cudaFree(alt_cuda);
    cudaFree(refLen_cuda);
    cudaFree(altLen_cuda);
    cudaFree(strategy_cuda);   

    cudaFree(best_gap_v_cuda);
    cudaFree(gap_size_v_cuda);

    cudaFree(state_cuda);
    cudaFree(segmentLen_cuda);
    cudaFree(offset_cuda);
    cudaFree(num_cuda);

//    return elapsedTime1+elapsedTime2;
    return elapsedTime1;
}

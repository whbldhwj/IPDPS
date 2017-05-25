#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <string.h>
#include "params.h"
#include <ctime>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

__global__ void calMatrix(
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

__global__ void calCigar(
    int refNum,
//    int *sw, 
    btrack_t *btrack,
    int *sw_lastrow,
    int *sw_lastcol,
    short *refLen, short *altLen, short *strategy,
    short *state, short *segmentLen, short *offset, short *num
);

__global__ void calMatrix(
    int refNum,
    int p_w_match, int p_w_mismatch, int p_w_open, int p_w_extend,
    char *ref, char *alt,
    short *refLen, short *altLen, short *strategy,
    btrack_t *btrack,
    int *sw_lastrow,
    int *sw_lastcol,
    int *best_gap_v,
    char *gap_size_v
){
    int pairIter = blockIdx.x;
    int tx = threadIdx.x;

    __shared__ char alt_seq[BLOCK_SIZE+BLOCK_SIZE];   
    
    int sw1, sw2, sw3;

    __shared__ btrack_t btrack_tile[BLOCK_SIZE][BLOCK_SIZE+2];

    __shared__ int best_gap_v_tile[BLOCK_SIZE+BLOCK_SIZE];
    __shared__ char gap_size_v_tile[BLOCK_SIZE+BLOCK_SIZE];

//    int best_gap_v_local;
//    char gap_size_v_local;

    int best_gap_h_local = lowInitValue;
    char gap_size_h_local;
    
    short alt_len = altLen[pairIter];
    short ref_len = refLen[pairIter];

    short nrow_block = (ref_len - 1 + BLOCK_SIZE) / BLOCK_SIZE;
    short ncol_block = (alt_len - 1 + BLOCK_SIZE) / BLOCK_SIZE;

    char ref_base;
    int swLeft, swUp, swLeftUp;

    btrack += (PADDED_ALT_LEN + 1);
    btrack += pairIter*PADDED_ALT_LEN*PADDED_REF_LEN;
    best_gap_v += pairIter*PADDED_ALT_LEN;
    gap_size_v += pairIter*PADDED_ALT_LEN;
    sw_lastrow += pairIter*PADDED_ALT_LEN;
    sw_lastcol += pairIter*PADDED_REF_LEN;
    alt += pairIter*MAX_ALT_LEN;
    ref += pairIter*MAX_REF_LEN;

    int tx_prev = (tx+31) & 0x1f;

    for (short row_tile = 0; row_tile < nrow_block; row_tile++){
//        int sw_offset = pairIter*PADDED_ALT_LEN*PADDED_REF_LEN;
//        int ref_offset = pairIter*MAX_REF_LEN + row_tile*BLOCK_SIZE;
        ref_base = ref[tx];
//        int alt_offset = pairIter*MAX_ALT_LEN;
//        int gap_v_offset = pairIter*PADDED_ALT_LEN;
        // start
        alt_seq[tx] = alt[tx];        
        best_gap_v_tile[tx] = best_gap_v[tx];

        int sw_tmp;
        best_gap_h_local = lowInitValue;

        sw1 = 0;
        sw2 = 0;
        sw3 = 0;
        for (int diag = 0; diag < BLOCK_SIZE; diag++)
        {
            int t_index_x = tx;
            int t_index_y = diag - tx;
                
            swLeft = sw2;
//            swUp = __shfl(sw2, (tx+31) & 0x1f);
//            swLeftUp = __shfl(sw3, (tx+31) & 0x1f);
            swUp = __shfl(sw2, tx_prev);
            swLeftUp = __shfl(sw3, tx_prev);

//            best_gap_v_local = __shfl(best_gap_v_local, (tx+31) & 0x1f);
//            best_gap_v_local = __shfl(best_gap_v_local, tx_prev);
//            if (tx == 0){
//                best_gap_v_local = best_gap_v_tile[t_index_y];
//            }
//            gap_size_v_local = __shfl(gap_size_v_local, (tx+31) & 0x1f);
//            gap_size_v_local = __shfl(gap_size_v_local, tx_prev);
//            if (tx == 0){
//                gap_size_v_local = gap_size_v_tile[t_index_y];
//            }
          
            if (tx <= diag){
                char alt_base = alt_seq[t_index_y];
                int temp = (ref_base == alt_base)? p_w_match: p_w_mismatch;

                if (row_tile > 0 && tx == 0){
                    swUp = sw_lastrow[t_index_y + 1];
                    swLeftUp = sw_lastrow[t_index_y];
                }

                int step_diag = swLeftUp + temp;
                step_diag = (MATRIX_MIN_CUTOFF > step_diag)? MATRIX_MIN_CUTOFF: step_diag;
                int prev_gap = swUp + p_w_open;
//                best_gap_v_local += p_w_extend;
                best_gap_v_tile[t_index_y] += p_w_extend;
                if (prev_gap > best_gap_v_tile[t_index_y]){
                    best_gap_v_tile[t_index_y] = prev_gap;
                    gap_size_v_tile[t_index_y] = 1;
                } else {
                    gap_size_v_tile[t_index_y]++;
                }

                int step_down = best_gap_v_tile[t_index_y];

                step_down = (MATRIX_MIN_CUTOFF > step_down)? MATRIX_MIN_CUTOFF: step_down;
                char kd = gap_size_v_tile[t_index_y];

                prev_gap = swLeft + p_w_open;
                best_gap_h_local += p_w_extend;
                if (prev_gap >= best_gap_h_local){
                    best_gap_h_local = prev_gap;
                    gap_size_h_local = 1;
                } else {
                    gap_size_h_local++;
                }
                
                int step_right = best_gap_h_local;
                step_right = (MATRIX_MIN_CUTOFF > step_right)? MATRIX_MIN_CUTOFF: step_right;
                char ki = gap_size_h_local;

                btrack_t b_tmp;
                sw_tmp = max(step_diag, step_down);
                sw_tmp = max(sw_tmp, step_right);
                b_tmp = (short)kd;
                b_tmp = (sw_tmp == step_right)? -(short)ki: b_tmp;
                b_tmp = (sw_tmp == step_diag)? 0: b_tmp;
               
                sw1 = sw_tmp;
                btrack_tile[t_index_x][t_index_y] = b_tmp;

                if (row_tile < nrow_block-1){
                    if (tx == BLOCK_SIZE-1){
                        sw_lastrow[t_index_y + 1] = sw_tmp;
                    }
                } else {
                    if (t_index_x + row_tile*BLOCK_SIZE + 1 == ref_len){
                        sw_lastrow[t_index_y + 1] = sw_tmp;
                    }
                }
            }

            sw3 = sw2;
            sw2 = sw1;
//            if (tx == BLOCK_SIZE - 1){
//                best_gap_v_tile[t_index_y] = best_gap_v_local;
//                gap_size_v_tile[t_index_y] = gap_size_v_local;
//            }

        }
        for (int ty=0; ty<BLOCK_SIZE; ty++){
            btrack[ty*PADDED_ALT_LEN + tx] = btrack_tile[ty][tx];
        }

        // middle
        for (int block = 0; block < ncol_block-1; block++){
            if (block > 0){
                best_gap_v_tile[tx] = best_gap_v_tile[tx+BLOCK_SIZE];
            }
            best_gap_v_tile[tx + BLOCK_SIZE] = best_gap_v[tx + (block+1)*BLOCK_SIZE];
            if (block > 0){
                gap_size_v_tile[tx] = gap_size_v_tile[tx+BLOCK_SIZE];
            }
            gap_size_v_tile[tx + BLOCK_SIZE] = gap_size_v[tx + (block+1)*BLOCK_SIZE];
            if (block > 0){
                alt_seq[tx] = alt_seq[tx+BLOCK_SIZE];
            }
            alt_seq[tx + BLOCK_SIZE] = alt[BLOCK_SIZE + block*BLOCK_SIZE + tx];

            int diagBound = min(BLOCK_SIZE*2 + block*BLOCK_SIZE, alt_len);
            for (int diag = BLOCK_SIZE + block*BLOCK_SIZE; diag < diagBound; diag++){
                int t_index_x = tx;
                int t_index_y = diag - tx;

                char alt_base = alt_seq[t_index_y - block*BLOCK_SIZE];
                int temp = (ref_base == alt_base)? p_w_match: p_w_mismatch;
                swLeft = sw2;
                swUp = __shfl(sw2, tx_prev);
                swLeftUp = __shfl(sw3, tx_prev);

//                best_gap_v_local = __shfl(best_gap_v_local, tx_prev);
//                if (tx == 0){
//                    best_gap_v_local = best_gap_v_tile[t_index_y - block*BLOCK_SIZE];
//                }
//                gap_size_v_local = __shfl(gap_size_v_local, tx_prev);
//                if (tx == 0){
//                    gap_size_v_local = gap_size_v_tile[t_index_y - block*BLOCK_SIZE];
//                }              

                if (row_tile > 0 && tx == 0){
                    swUp = sw_lastrow[t_index_y + 1];
                    swLeftUp = sw_lastrow[t_index_y];
                }
                int step_diag = swLeftUp + temp;
                step_diag = (MATRIX_MIN_CUTOFF > step_diag)? MATRIX_MIN_CUTOFF: step_diag;
                int prev_gap = swUp + p_w_open;
//                best_gap_v_local += p_w_extend;
                int v_ind = t_index_y - block*BLOCK_SIZE;
                best_gap_v_tile[v_ind] += p_w_extend;

                if (prev_gap > best_gap_v_tile[v_ind]){
                    best_gap_v_tile[v_ind] = prev_gap;
                    gap_size_v_tile[v_ind] = 1;

                } else {
                    gap_size_v_tile[v_ind]++;
                }

                int step_down = best_gap_v_tile[v_ind];
                step_down = (MATRIX_MIN_CUTOFF > step_down)? MATRIX_MIN_CUTOFF: step_down;
                char kd = gap_size_v_tile[v_ind];

                prev_gap = swLeft + p_w_open;
                best_gap_h_local += p_w_extend;
                if (prev_gap >= best_gap_h_local){
                    best_gap_h_local = prev_gap;
                    gap_size_h_local = 1;
                } else {
                    gap_size_h_local++;
                }
                
                int step_right = best_gap_h_local;
                step_right = (MATRIX_MIN_CUTOFF > step_right)? MATRIX_MIN_CUTOFF: step_right;
                char ki = gap_size_h_local;

                btrack_t b_tmp;
                sw_tmp = max(step_diag, step_down);
                sw_tmp = max(sw_tmp, step_right);
                b_tmp = (short)kd;
                b_tmp = (sw_tmp == step_right)? -(short)ki: b_tmp;
                b_tmp = (sw_tmp == step_diag)? 0: b_tmp;


                sw1 = sw_tmp;
                btrack_tile[t_index_x][t_index_y - block*BLOCK_SIZE - (BLOCK_SIZE - tx)] = b_tmp;
                
                if (row_tile < nrow_block-1){
                    if (tx == BLOCK_SIZE-1){
                        sw_lastrow[t_index_y + 1] = sw_tmp;
                    }

                } else {
                    if (t_index_x + row_tile*BLOCK_SIZE + 1 == ref_len){
                        sw_lastrow[t_index_y + 1] = sw_tmp;
                    }
                }

                if (block == ncol_block-1-1){
                    if (t_index_y == alt_len-1){
                        sw_lastcol[t_index_x + 1] = sw_tmp;
                    }
                }

                sw3 = sw2;
                sw2 = sw1;

//                if (tx == BLOCK_SIZE - 1){
//                    best_gap_v_tile[t_index_y - block*BLOCK_SIZE] = best_gap_v_local;
//                    gap_size_v_tile[t_index_y - block*BLOCK_SIZE] = gap_size_v_local;
//                }
            }

            __syncthreads();
            best_gap_v[block*BLOCK_SIZE + tx] = best_gap_v_tile[tx];
            gap_size_v[block*BLOCK_SIZE + tx] = gap_size_v_tile[tx];
           
            for (int ty = 0; ty < BLOCK_SIZE; ty++){
                btrack[ty*PADDED_ALT_LEN + block*BLOCK_SIZE + BLOCK_SIZE - ty + tx] = btrack_tile[ty][tx];
            }
        }

        // end
        for (int diag = alt_len; diag < alt_len + BLOCK_SIZE - 1; diag++)
        {
            int diagStart = (diag - alt_len) + 1;
            swLeft = sw2;
            swUp = __shfl(sw2, tx_prev);
            swLeftUp = __shfl(sw3, tx_prev);

//            best_gap_v_local = __shfl(best_gap_v_local, tx_prev);
//            gap_size_v_local = __shfl(gap_size_v_local, tx_prev);

            if (tx >= diagStart){
                int t_index_x = tx;
                int t_index_y = diag - tx;

                char alt_base = alt_seq[t_index_y - (ncol_block-2)*BLOCK_SIZE];
                int temp = (ref_base == alt_base)? p_w_match: p_w_mismatch;

                if (row_tile > 0 && tx == 0){
                    swUp = sw_lastrow[t_index_y + 1];
                    swLeftUp = sw_lastrow[t_index_y];
                }
                int step_diag = swLeftUp + temp;
                step_diag = (MATRIX_MIN_CUTOFF > step_diag)? MATRIX_MIN_CUTOFF: step_diag;
                int prev_gap = swUp + p_w_open;
                int v_tile_offset = t_index_y - (ncol_block-2)*BLOCK_SIZE;
                best_gap_v_tile[v_tile_offset] += p_w_extend;
                if (prev_gap > best_gap_v_tile[v_tile_offset]){
                    best_gap_v_tile[v_tile_offset] = prev_gap;
                    gap_size_v_tile[v_tile_offset] = 1;
                } else {
                    gap_size_v_tile[v_tile_offset]++;
                }

                int step_down = best_gap_v_tile[v_tile_offset];
                step_down = (MATRIX_MIN_CUTOFF > step_down)? MATRIX_MIN_CUTOFF: step_down;
                char kd = gap_size_v_tile[v_tile_offset];

                prev_gap = swLeft + p_w_open;
                best_gap_h_local += p_w_extend;
                if (prev_gap >= best_gap_h_local){
                    best_gap_h_local = prev_gap;
                    gap_size_h_local = 1;
                } else {
                    gap_size_h_local++;
                }
                
                int step_right = best_gap_h_local;
                step_right = (MATRIX_MIN_CUTOFF > step_right)? MATRIX_MIN_CUTOFF: step_right;
                char ki = gap_size_h_local;

                btrack_t b_tmp;
                sw_tmp = max(step_diag, step_down);
                sw_tmp = max(sw_tmp, step_right);
                b_tmp = (short)kd;
                b_tmp = (sw_tmp == step_right)? -(short)ki: b_tmp;
                b_tmp = (sw_tmp == step_diag)? 0: b_tmp;               

                sw1 = sw_tmp;
                btrack_tile[t_index_x][t_index_y - (alt_len-BLOCK_SIZE) - (BLOCK_SIZE-tx)] = b_tmp;
                if (row_tile < nrow_block-1){
                    if (tx == BLOCK_SIZE-1){
                        sw_lastrow[t_index_y + 1] = sw_tmp;
                    }
                } else {
                    if (t_index_x + row_tile*BLOCK_SIZE + 1 == ref_len){
                        sw_lastrow[t_index_y + 1] = sw_tmp;
                    }
                }

                if (t_index_y == alt_len-1){
                    sw_lastcol[t_index_x + 1] = sw_tmp;
                }

                sw3 = sw2;
                sw2 = sw1;

//                if (tx == BLOCK_SIZE - 1){
//                    best_gap_v_tile[t_index_y - (ncol_block-2)*BLOCK_SIZE] = best_gap_v_local;
//                    gap_size_v_tile[t_index_y - (ncol_block-2)*BLOCK_SIZE] = gap_size_v_local;
//                }
            }
        }

        __syncthreads();
        for (int i = 0; i < BLOCK_SIZE; i++){
            for (int j=tx; j<i; j+=blockDim.x){
                btrack[i*PADDED_ALT_LEN + alt_len - i + j] = btrack_tile[i][j];
            }
        }

        for (int i = tx; i < BLOCK_SIZE*2; i+=blockDim.x){
            best_gap_v[(ncol_block-2)*BLOCK_SIZE + i] = best_gap_v_tile[i];
            gap_size_v[(ncol_block-2)*BLOCK_SIZE + i] = gap_size_v_tile[i];
        }

        ref += BLOCK_SIZE;
        btrack += BLOCK_SIZE*PADDED_ALT_LEN;
        sw_lastcol += BLOCK_SIZE;
        __syncthreads();
    }
}

__global__ void calCigar(
    int refNum,
//    int *sw, 
    btrack_t *btrack,
    int *sw_lastrow,
    int *sw_lastcol,
    short *refLen, short *altLen, short *strategy,
    short *state, short *segmentLen, short *offset, short *num
){
    int pairIter = blockIdx.x;
//    int tx = threadIdx.x;

//    short state_tmp[MAX_STATE_NUM];

//    for(int pairIter = 0; pairIter < refNum; pairIter++)
    {  
        short altLength = altLen[pairIter];
        short refLength = refLen[pairIter];
       
//        int j;
        short nrow = refLength + 1;    
       
        short alignment_offset = 0;
        short p1 = 0, p2 = 0;
        int state_cur = 0;
        short count_cigar = 0;
        int maxscore = MAXCORE_DATA;
        short segment_length = 0;
    
        if (strategy[pairIter] == 1) {
            p1 = refLength;
            p2 = altLength;
        } else {
            p2 = altLength;
            for(int i=1; i<nrow; i++) {
                int curScore = sw_lastcol[pairIter*PADDED_REF_LEN + i];
                if (curScore >= maxscore) {
                    p1 = i;
                    maxscore = curScore;
                }
            }
    
            if(strategy[pairIter] != 2) {
                for(int j=1; j<altLength+1; j++) {
                    int curScore = sw_lastrow[pairIter*PADDED_ALT_LEN + j];
                    int abs1, abs2;
                    if(refLength > j) {
                        abs1 = refLength - j;
                    } else {
                        abs1 = j - refLength;
                    }
                    if(p1 > p2) {
                        abs2 = p1 - p2;
                    } else {
                        abs2 = p2-  p1;
                    }
                    if(curScore > maxscore || (curScore == maxscore && abs1 < abs2)) {
                        p1 = refLength;
                        p2 = j;
                        maxscore = curScore;
                        segment_length = altLength - j;
                    }
                }
            }
        }
    
        if (segment_length > 0 && strategy[pairIter] == 0) {
            state[pairIter*MAX_STATE_NUM + count_cigar] = 3;
            segmentLen[pairIter*MAX_STATE_NUM + count_cigar] = segment_length;
            count_cigar++;
            segment_length = 0;
        }
    
        do {
            btrack_t btr = btrack[pairIter*PADDED_ALT_LEN*PADDED_REF_LEN+p1*PADDED_ALT_LEN+p2];
            int new_state;
            int step_length = 1;
            if (btr > 0) {
                new_state = 2;
                step_length = btr;
            } else if (btr < 0) {
                new_state = 1;
                step_length = (-btr);
            } else {    
                new_state = 0;
            }
    
            switch(new_state) {
                case 0: p1--; p2--; break; 
                case 1: p2 -= step_length; break; 
                case 2: p1 -= step_length; break; 
            }
    
            if (new_state == state_cur) {
                segment_length += step_length;
            } else {
                state[pairIter*MAX_STATE_NUM + count_cigar] = state_cur;
                segmentLen[pairIter*MAX_STATE_NUM + count_cigar] = segment_length;
                count_cigar++;
                segment_length = step_length;
                state_cur = new_state;
            }
        } while (p1 > 0 && p2 > 0);
    
        if(strategy[pairIter] == 0) {
            state[pairIter*MAX_STATE_NUM + count_cigar] = state_cur;
            segmentLen[pairIter*MAX_STATE_NUM + count_cigar] = segment_length;
            count_cigar++;
            if(p2 > 0) {
                state[pairIter*MAX_STATE_NUM + count_cigar] = 3;
                segmentLen[pairIter*MAX_STATE_NUM + count_cigar] = p2;
                count_cigar++;
            }
            alignment_offset = p1;
        } else if(strategy[pairIter] == 3) {
            state[pairIter*MAX_STATE_NUM + count_cigar] = state_cur;
            segmentLen[pairIter*MAX_STATE_NUM + count_cigar] = segment_length + p2;
            count_cigar++;
            alignment_offset = p1 - p2;
        } else {
            state[pairIter*MAX_STATE_NUM + count_cigar] = state_cur;
            segmentLen[pairIter*MAX_STATE_NUM + count_cigar] = segment_length;
            count_cigar++;
            if(p1 > 0) {
                state[pairIter*MAX_STATE_NUM + count_cigar] = 2;
                segmentLen[pairIter*MAX_STATE_NUM + count_cigar] = p1;
                count_cigar++;
            } else if (p2 > 0) {
                state[pairIter*MAX_STATE_NUM + count_cigar] = 1;
                segmentLen[pairIter*MAX_STATE_NUM + count_cigar] = p2;
                count_cigar++;
            }
            alignment_offset = 0;
        }
        
        offset[pairIter] = alignment_offset;
        num[pairIter] = count_cigar;  
    }
}

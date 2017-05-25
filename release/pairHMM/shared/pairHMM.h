#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include <omp.h>

typedef char uint8;

#define STREAM_NUM 16
#define STREAM_MASK 15

#define MAX_HAPLOTYPE_LENGTH 789  // bases
#define PADDED_MAX_HAPLOTYPE_LENGTH 790 // bases

#define MAX_READ_LENGTH 102 // bases
#define PADDED_MAX_READ_LENGTH 103 // bases

#define MAX_READ_LENGTH1 64 // bases
#define PADDED_MAX_READ_LENGTH1 65 // bases
#define MAX_READ_LENGTH2 96
#define PADDED_MAX_READ_LENGTH2 97 // bases
#define MAX_READ_LENGTH3 102
#define PADDED_MAX_READ_LENGTH3 103 // base

#define HAPLOTYPE_NUM 200
#define READ_NUM 9500

// -----------------------
// PairHMMmodel.java -> PairHMMModel
// -----------------------
#define TRANS_PROB_ARRAY_LENGTH 4 // length of the standard transition probability array
#define MATCH_TO_MATCH 0 // position in the transition probability array for the Match-to-Match transition
#define INDEL_TO_MATCH 1 // position in the transition probability array for the Indel-to-Match transition
#define MATCH_TO_INSERTION 2
#define INSERTION_TO_INSERTION 4
#define MATCH_TO_DELETION 3
#define DELETION_TO_DELETION 5

// -----------------------
// N2MemoryPairHMM.java
// -----------------------
#define doNotUseTristateCorrection false
#define TRISTATE_CORRECTION 3.0

// -----------------------
// Log10PairHMM.java
// -----------------------
//const prob_t NEGATIVE_INFINITY = -INFINITY;
//const float NEGATIVE_INFINITY = 0.0;

// -----------------------
// PairHMMModel.java
// -----------------------
#define MAX_QUAL 254

#define MAX_JACOBIAN_TOLERANCE 8.0
#define JACOBIAN_LOG_TABLE_STEP 0.0001
#define JACOBIAN_LOG_TABLE_INV_STEP (1.0 / JACOBIAN_LOG_TABLE_STEP)
#define MAXN 70000
#define LOG10_CACHE_SIZE (4*MAXN)
#define JACOBIAN_LOG_TABLE_SIZE ((int)(MAX_JACOBIAN_TOLERANCE / JACOBIAN_LOG_TABLE_STEP)+1)

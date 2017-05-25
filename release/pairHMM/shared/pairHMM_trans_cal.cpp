#include "pairHMM.h"

extern void initializeProbabilities(uint8 insertionGOP[MAX_READ_LENGTH],
                             uint8 deletionGOP[MAX_READ_LENGTH],
                             uint8 overallGCP[MAX_READ_LENGTH],
                             float transition1[PADDED_MAX_READ_LENGTH][TRANS_PROB_ARRAY_LENGTH],
							 float qualToErrorProbCache[MAX_QUAL+1],
							 float matchToMatch[(MAX_QUAL + 1) * (MAX_QUAL + 2) >> 1],
							 int readLength,
                             float jacobianLogTable[JACOBIAN_LOG_TABLE_SIZE]
);

void trans_compute(
    int readNum,
    uint8 *read_in,
    int *readLength_in,
    float  transition1[READ_NUM][PADDED_MAX_READ_LENGTH][TRANS_PROB_ARRAY_LENGTH],
    float qualToErrorProbCache1[MAX_QUAL+1],
    float matchToMatch1[(MAX_QUAL + 1) * (MAX_QUAL + 2) >> 1],
    float jacobianLogTable[JACOBIAN_LOG_TABLE_SIZE]
){
    uint8 readBaseInsertionQualitiesCur1[MAX_READ_LENGTH];
	uint8 readBaseDeletionQualitiesCur1[MAX_READ_LENGTH];
	uint8 readGCPCur1[MAX_READ_LENGTH];
	int readLengthCur1;
//#pragma omp parallel for schedule(dynamic) num_threads(6)
    for (int readIter = 0; readIter < readNum; readIter++){
        //int readLengthCur1;
        //uint8 readBaseInsertionQualitiesCur1[MAX_READ_LENGTH];
	    //uint8 readBaseDeletionQualitiesCur1[MAX_READ_LENGTH];
    	//uint8 readGCPCur1[MAX_READ_LENGTH];
        //int tid = omp_get_thread_num();
        //printf("hey I'm #:%d\n", tid);
        for (int i = 0; i < MAX_READ_LENGTH; i++){
 			readBaseInsertionQualitiesCur1[i] = read_in[readIter*MAX_READ_LENGTH*5+MAX_READ_LENGTH*2+i];
			readBaseDeletionQualitiesCur1[i] = read_in[readIter*MAX_READ_LENGTH*5+MAX_READ_LENGTH*3+i];
			readGCPCur1[i] = read_in[readIter*MAX_READ_LENGTH*5+MAX_READ_LENGTH*4+i];        
        }
        readLengthCur1 = readLength_in[readIter]; 		        
        initializeProbabilities(
				readBaseInsertionQualitiesCur1, readBaseDeletionQualitiesCur1, readGCPCur1,
				transition1[readIter],
				qualToErrorProbCache1, matchToMatch1, readLengthCur1,
                jacobianLogTable
		);    
    }
}

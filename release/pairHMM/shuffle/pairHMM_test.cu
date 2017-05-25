#include "pairHMM.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <sys/time.h>

__constant__ float qualToErrorProbCache_cuda[MAX_QUAL+1];
static float transition1[READ_NUM][PADDED_MAX_READ_LENGTH][TRANS_PROB_ARRAY_LENGTH];
static float jacobianLogTable[JACOBIAN_LOG_TABLE_SIZE];
static float matchToMatch1[(MAX_QUAL + 1) * (MAX_QUAL + 2) >> 1];

double gettime(){
    struct timeval t;
    gettimeofday(&t,NULL);
    return t.tv_sec+t.tv_usec*1e-6;
}

extern void initializeProbabilities(uint8 insertionGOP[MAX_READ_LENGTH],
                             uint8 deletionGOP[MAX_READ_LENGTH],
                             uint8 overallGCP[MAX_READ_LENGTH],
                             float transition1[PADDED_MAX_READ_LENGTH][TRANS_PROB_ARRAY_LENGTH],
							 float qualToErrorProbCache[MAX_QUAL+1],
							 float matchToMatch[(MAX_QUAL + 1) * (MAX_QUAL + 2) >> 1],
							 int readLength,
                             float jacobianLogTable[JACOBIAN_LOG_TABLE_SIZE]
                             );
extern void init_compute(
    float qualToErrorProbCache1[MAX_QUAL+1],
	float matchToMatch1[(MAX_QUAL + 1) * (MAX_QUAL + 2) >> 1]);

extern void initializeJacobianLogTable(
    float jacobianLogTable[JACOBIAN_LOG_TABLE_SIZE]       
);

extern void trans_compute(
    int readNum,
    uint8 *read_in,
    int *readLength_in,
    float  transition1[READ_NUM][PADDED_MAX_READ_LENGTH][TRANS_PROB_ARRAY_LENGTH],
    float qualToErrorProbCache1[MAX_QUAL+1],
    float matchToMatch1[(MAX_QUAL + 1) * (MAX_QUAL + 2) >> 1],
    float jacobianLogTable[JACOBIAN_LOG_TABLE_SIZE]
);

extern __global__ void kernel1(
		uint8 *read_in,		
        int *readLength_in,
		uint8 *hap_in,				
        int *hapLength_in,
		float *mLikelihoodArray,		
        int readNum,
		int haplotypeNum,		
		float *transition1
);

extern __global__ void kernel2(
		uint8 *read_in,		
        int *readLength_in,
		uint8 *hap_in,				
        int *hapLength_in,
		float *mLikelihoodArray,		
        int readNum,
		int haplotypeNum,		
		float *transition1
);

extern __global__ void kernel3(
		uint8 *read_in,		
        int *readLength_in,
		uint8 *hap_in,				
        int *hapLength_in,
		float *mLikelihoodArray,		
        int readNum,
		int haplotypeNum,		
		float *transition1
);

float cuda_kernel(
    uint8* read_in,
    uint8* hap_in,
    int* readLength_in,
    int* hapLength_in,
	int readNum,
	int hapNum,
    float* mLikelihoodArray_hw
){
    float qualToErrorProbCache1[MAX_QUAL+1];
	//float matchToMatch1[(MAX_QUAL + 1) * (MAX_QUAL + 2) >> 1];
    //float jacobianLogTable[JACOBIAN_LOG_TABLE_SIZE];

    init_compute(
        qualToErrorProbCache1,
		matchToMatch1
    );
    initializeJacobianLogTable(
        jacobianLogTable
    );
	
	//uint8 readBaseInsertionQualitiesCur1[MAX_READ_LENGTH];
	//uint8 readBaseDeletionQualitiesCur1[MAX_READ_LENGTH];
	//uint8 readGCPCur1[MAX_READ_LENGTH];
	//int readLengthCur1;

	uint8 *read_cuda;
	uint8 *hap_cuda;
    int *readLen_cuda;
    int *hapLen_cuda;
	float *mLikelihoodArray_cuda;
	float *transition_cuda;

    cudaEvent_t start, stop;
	float elapsed_time;
	
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//cudaEventRecord(start, 0);

    cudaMalloc((void**)& read_cuda, readNum * MAX_READ_LENGTH * 5 * sizeof(uint8));
    cudaMalloc((void**)& hap_cuda, hapNum * MAX_HAPLOTYPE_LENGTH * sizeof(uint8));
    cudaMalloc((void**)& readLen_cuda, readNum * sizeof(int));
    cudaMalloc((void**)& hapLen_cuda, hapNum * sizeof(int));
    cudaMalloc((void**)& mLikelihoodArray_cuda, sizeof(float) * readNum * hapNum);
	cudaMemset(mLikelihoodArray_cuda, 0, sizeof(float) * readNum * hapNum);
	cudaMalloc((void**)& transition_cuda, sizeof(float) * PADDED_MAX_READ_LENGTH * TRANS_PROB_ARRAY_LENGTH * readNum);

	cudaEventRecord(start, 0);
	
    cudaMemcpy(read_cuda, read_in, readNum * MAX_READ_LENGTH * 5 * sizeof(uint8), cudaMemcpyHostToDevice);
    cudaMemcpy(hap_cuda, hap_in, hapNum * MAX_HAPLOTYPE_LENGTH * sizeof(uint8), cudaMemcpyHostToDevice);
    cudaMemcpy(readLen_cuda, readLength_in, readNum * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(hapLen_cuda, hapLength_in, hapNum * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(qualToErrorProbCache_cuda, qualToErrorProbCache1, sizeof(float) * (MAX_QUAL+1));
    
    dim3 dimGrid(readNum,hapNum);
    //dim3 dimGrid(1,1);
    dim3 dimBlock(32,1);		

//	cudaEventRecord(start, 0);

	//float transition1[READ_NUM][PADDED_MAX_READ_LENGTH][TRANS_PROB_ARRAY_LENGTH];

    trans_compute(
        readNum,
        read_in,
        readLength_in,
        transition1,
        qualToErrorProbCache1,
        matchToMatch1,
        jacobianLogTable);

//#pragma omp parallel for num_threads(12)
//    for (int readIter = 0; readIter < readNum; readIter++){
//        int tid = omp_get_thread_num();
//        printf("hey I'm #:%d\n", tid);
//        for (int i = 0; i < MAX_READ_LENGTH; i++){
// 			readBaseInsertionQualitiesCur1[i] = read_in[readIter*MAX_READ_LENGTH*5+MAX_READ_LENGTH*2+i];
//			readBaseDeletionQualitiesCur1[i] = read_in[readIter*MAX_READ_LENGTH*5+MAX_READ_LENGTH*3+i];
//			readGCPCur1[i] = read_in[readIter*MAX_READ_LENGTH*5+MAX_READ_LENGTH*4+i];        
//        }
//        readLengthCur1 = readLength_in[readIter]; 		        
//        initializeProbabilities(
//				readBaseInsertionQualitiesCur1, readBaseDeletionQualitiesCur1, readGCPCur1,
//				transition1[readIter],
//				qualToErrorProbCache1, matchToMatch1, readLengthCur1,
//                jacobianLogTable
//		);    
//    }

	cudaMemcpy(transition_cuda, transition1, sizeof(float)*PADDED_MAX_READ_LENGTH*TRANS_PROB_ARRAY_LENGTH*readNum, cudaMemcpyHostToDevice);

    int readLenMax = 0;
    for (int i = 0; i < readNum; i++){
        if (readLength_in[i] > readLenMax)
            readLenMax = readLength_in[i];
    }

//	cudaEventRecord(start, 0);

    if (readLenMax <= MAX_READ_LENGTH1){
		kernel1<<<dimGrid, dimBlock>>>(
			read_cuda,
			readLen_cuda,
			hap_cuda,        
			hapLen_cuda,
			mLikelihoodArray_cuda,
			readNum,
			hapNum,        
			transition_cuda
		);
    } else if (readLenMax <= MAX_READ_LENGTH2){
		kernel2<<<dimGrid, dimBlock>>>(
			read_cuda,
			readLen_cuda,
			hap_cuda,        
			hapLen_cuda,
			mLikelihoodArray_cuda,
			readNum,
			hapNum,        
			transition_cuda
		);
    } else {
		kernel3<<<dimGrid, dimBlock>>>(
			read_cuda,
			readLen_cuda,
			hap_cuda,        
			hapLen_cuda,
			mLikelihoodArray_cuda,
			readNum,
			hapNum,        
			transition_cuda
		);
    }
//    cudaEventRecord(stop, 0);
//	cudaEventSynchronize(stop);   

    cudaMemcpy(mLikelihoodArray_hw, mLikelihoodArray_cuda, sizeof(float)*readNum*hapNum, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);   

    cudaFree(read_cuda);
    cudaFree(hap_cuda);
    cudaFree(readLen_cuda);
    cudaFree(hapLen_cuda);
    cudaFree(transition_cuda);
    cudaFree(qualToErrorProbCache_cuda);
    cudaFree(mLikelihoodArray_cuda);
   
	cudaEventElapsedTime(&elapsed_time, start, stop);

	return elapsed_time;
}

__device__ void wrapper1(			
			uint8 *haplotypeBases,
            uint8 *readBases,
			uint8 *readBaseQualities,
            int haplotypeLength,
            int readLength,
            float *transition,
            float *likelihood,
            float initialValue
)
{
	int nBound = 0;
	int haplotypeLengthCur = 0;
	haplotypeLengthCur = haplotypeLength;

	int minLength = (haplotypeLengthCur < readLength)? haplotypeLengthCur : readLength;
	int maxLength = (haplotypeLengthCur > readLength)? haplotypeLengthCur : readLength;

	int mBound = haplotypeLengthCur + readLength;
	
	int indI = 0;
	int indJ = 0;
	
	int tx = threadIdx.x;
    float likelihoodTmp = 0.0;

	float matchLine1 = 0.0, matchLine2 = 0.0, matchLine3 = 0.0;
	float insertionLine1 = 0.0, insertionLine2 = 0.0, insertionLine3 = 0.0;
	float deletionLine1 = 0.0, deletionLine2 = 0.0, deletionLine3 = 0.0;

	float deletionMatrixLeftUp;
	float deletionMatrixLeft;

	float matchMatrixLeftUp;
	float matchMatrixUp;
	float matchMatrixLeft;
		
	float insertionMatrixUp;
	float insertionMatrixLeftUp;
	float insertionMatrixLeft;
	
	float cache_tmp;
    uint8 x;
//  uint8 qual;
    float transition_local[6];
    float c1, c2;

    for (int layerIndex = 0; layerIndex < readLength; layerIndex ++){
		indI = layerIndex + 1 - tx;
        indJ = tx + 1;
		
		if (layerIndex < minLength) {
			nBound = layerIndex + 1;
		} else {
			if (layerIndex >= maxLength) {
				nBound = -layerIndex + haplotypeLengthCur + readLength - 1;
			} else {
				nBound = minLength;
			}
		}

    	matchMatrixLeft = __shfl(matchLine2, (tx+31) & 0x1f);
    	matchMatrixLeftUp = __shfl(matchLine3, (tx+31) & 0x1f);
    	matchMatrixUp = matchLine2;
    			
    	insertionMatrixLeft = __shfl(insertionLine2, (tx+31) & 0x1f);
    	insertionMatrixLeftUp = __shfl(insertionLine3, (tx+31) & 0x1f);
    	insertionMatrixUp = insertionLine2;
    	
    	deletionMatrixLeft = __shfl(deletionLine2, (tx+31) & 0x1f);
    	deletionMatrixLeftUp = (indI == 1)? initialValue: __shfl(deletionLine3, (tx+31) & 0x1f);

        if (tx < nBound)
        {
            x = __shfl(x, (tx+31) & 0x1f);
            if (tx == 0) x = readBases[indI-1];
            //qual = __shfl(qual, (tx+31) & 0x1f);
            //if (tx == 0) qual = readBaseQualities[indI-1];
			cache_tmp = __shfl(cache_tmp, (tx+31) & 0x1f);
            if (tx == 0) cache_tmp = qualToErrorProbCache_cuda[(int)readBaseQualities[indI-1] & 0xFF];
             
			c1 = 1 - cache_tmp;
			c2 = cache_tmp / TRISTATE_CORRECTION;

			uint8 y = haplotypeBases[indJ-1];
			float prior = (x == y || x == (uint8)'N' || y == (uint8)'N' ? c1 : c2);
          
//            transition_local[MATCH_TO_MATCH] = __shfl(transition_local[MATCH_TO_MATCH], (tx+31) & 0x1f);
//            transition_local[MATCH_TO_INSERTION] = __shfl(transition_local[MATCH_TO_INSERTION], (tx+31) & 0x1f);
//            transition_local[MATCH_TO_DELETION] = __shfl(transition_local[MATCH_TO_DELETION], (tx+31) & 0x1f);
//            transition_local[INDEL_TO_MATCH] = __shfl(transition_local[INDEL_TO_MATCH], (tx+31) & 0x1f); 
//            if (tx == 0) {
                transition_local[MATCH_TO_MATCH] = transition[indI*TRANS_PROB_ARRAY_LENGTH+MATCH_TO_MATCH];
                transition_local[MATCH_TO_INSERTION] = transition[indI*TRANS_PROB_ARRAY_LENGTH+MATCH_TO_INSERTION];
                transition_local[MATCH_TO_DELETION] = transition[indI*TRANS_PROB_ARRAY_LENGTH+MATCH_TO_DELETION];
                transition_local[INDEL_TO_MATCH] = transition[indI*TRANS_PROB_ARRAY_LENGTH+INDEL_TO_MATCH];
//            }
            transition_local[INSERTION_TO_INSERTION] = 1 - transition_local[INDEL_TO_MATCH];
            transition_local[DELETION_TO_DELETION] = transition_local[INSERTION_TO_INSERTION];
	    	
            matchLine1 = prior*(matchMatrixLeftUp*transition_local[MATCH_TO_MATCH]
                    +insertionMatrixLeftUp*transition_local[INDEL_TO_MATCH]
                    +deletionMatrixLeftUp*transition_local[INDEL_TO_MATCH]);
            insertionLine1 = matchMatrixUp*transition_local[MATCH_TO_INSERTION]
                    +insertionMatrixUp*transition_local[INSERTION_TO_INSERTION];
	    	deletionLine1 = matchMatrixLeft*transition_local[MATCH_TO_DELETION] 
                    +deletionMatrixLeft*transition_local[DELETION_TO_DELETION];
		
            if (tx == 0){
                if (layerIndex >= readLength) {
				    likelihoodTmp += matchMatrixLeft + insertionMatrixLeft;
			    }
            }

			matchLine3 = matchLine2; matchLine2 = matchLine1;
			insertionLine3 = insertionLine2; insertionLine2 = insertionLine1;
			deletionLine3 = deletionLine2; deletionLine2 = deletionLine1;	
		}
	} // for layerIndex

    if (tx < 31) {
        matchLine3 = __shfl(matchLine3, (tx+31) & 0x1f);
        insertionLine3 = __shfl(insertionLine3, (tx+31) & 0x1f);
        deletionLine3 = __shfl(deletionLine3, (tx+31) & 0x1f);
    }

	for (int layerIndex = readLength; layerIndex < mBound - 1; ++layerIndex){
		if (layerIndex < minLength) {
			nBound = layerIndex + 1;
		} else {
			if (layerIndex >= maxLength) {
				nBound = -layerIndex + haplotypeLengthCur + readLength - 1;
			} else {
				nBound = minLength;
			}
		}

		indI = readLength - tx;
        indJ = layerIndex + 1 - readLength + tx + 1;
	
        matchMatrixLeft = matchLine2;
        matchMatrixLeftUp = __shfl(matchLine3, tx+1);
        matchMatrixUp = __shfl(matchLine2, tx+1);
        		
        insertionMatrixLeft = insertionLine2;
        insertionMatrixLeftUp = __shfl(insertionLine3, tx+1);
        insertionMatrixUp = __shfl(insertionLine2, tx+1);
        
        deletionMatrixLeft = deletionLine2;
        deletionMatrixLeftUp = __shfl(deletionLine3, tx+1);
        if (indI == 1) deletionMatrixLeftUp = initialValue;
 
        if (tx < nBound) {
			uint8 y = haplotypeBases[indJ-1];
			float prior = (x == y || x == (uint8)'N' || y == (uint8)'N' ? c1 : c2);
	    		
            matchLine1 = prior*(matchMatrixLeftUp*transition_local[MATCH_TO_MATCH]
                    +insertionMatrixLeftUp*transition_local[INDEL_TO_MATCH]
                    +deletionMatrixLeftUp*transition_local[INDEL_TO_MATCH]);
            insertionLine1 = matchMatrixUp*transition_local[MATCH_TO_INSERTION]
                    +insertionMatrixUp*transition_local[INSERTION_TO_INSERTION];
	    	deletionLine1 = matchMatrixLeft*transition_local[MATCH_TO_DELETION] 
                    +deletionMatrixLeft*transition_local[DELETION_TO_DELETION];
	        
            if (tx == 0){
			    if (layerIndex >= readLength) {
				    likelihoodTmp += matchMatrixLeft + insertionMatrixLeft;
			    }
            }
			
			matchLine3 = matchLine2; matchLine2 = matchLine1;
			insertionLine3 = insertionLine2; insertionLine2 = insertionLine1;
			deletionLine3 = deletionLine2; deletionLine2 = deletionLine1;	
		}	
	} // for layerIndex

	likelihoodTmp += matchLine1 + insertionLine1;
    *likelihood = likelihoodTmp;
}

__device__ void wrapper2(			
			uint8 *haplotypeBases,
            uint8 *readBases,
			uint8 *readBaseQualities,
            int haplotypeLength,
            int readLength,
            float *transition,
            float *likelihood,
            float initialValue
)
{
//	float MAX = powf(2, 120); // float
//	float initialValue;
//	initialValue = MAX / haplotypeLength;

	int nBound = 0;
	int haplotypeLengthCur = 0;
	haplotypeLengthCur = haplotypeLength;

	int minLength = (haplotypeLengthCur < readLength)? haplotypeLengthCur : readLength;
	int maxLength = (haplotypeLengthCur > readLength)? haplotypeLengthCur : readLength;

	int mBound = haplotypeLengthCur + readLength;
	
	int indI = 0;
	int indJ = 0;
	
	int tx = threadIdx.x;
    float likelihoodTmp = 0.0;

    float matchLine1[2], matchLine2[2], matchLine3[2];
    float insertionLine1[2], insertionLine2[2], insertionLine3[2];
    float deletionLine1[2], deletionLine2[2], deletionLine3[2];

    #pragma unroll
    for (int i = 0; i < 2; i++){
        matchLine1[i] = 0.0;
        matchLine2[i] = 0.0;
        matchLine3[i] = 0.0;
        insertionLine1[i] = 0.0;
        insertionLine2[i] = 0.0;
        insertionLine3[i] = 0.0;
        deletionLine1[i] = 0.0;
        deletionLine2[i] = 0.0;
        deletionLine3[i] = 0.0;
    }

    float cache_tmp[2] = {0.0};
    uint8 x[2] = {0.0};
    //uint8 qual[2] = {0.0};
    float c1[2], c2[2];

	float deletionMatrixLeftUp;
	float deletionMatrixLeft;

	float matchMatrixLeftUp;
	float matchMatrixUp;
	float matchMatrixLeft;
		
	float insertionMatrixUp;
	float insertionMatrixLeftUp;
	float insertionMatrixLeft;
	
    float transition_tmp[2][6];
    
    for (int layerIndex = 0; layerIndex < readLength; layerIndex ++){
		indI = layerIndex + 1 - tx*2;
        indJ = tx*2 + 1;
		
		if (layerIndex < minLength) {
			nBound = layerIndex + 1;
		} else {
			if (layerIndex >= maxLength) {
				nBound = -layerIndex + haplotypeLengthCur + readLength - 1;
			} else {
				nBound = minLength;
			}
		}

        uint8 x_back = x[1];
        for (int i = 1; i > 0; i--)
            x[i] = x[i-1];
        x[0] = __shfl(x_back, (tx+31) & 0x1f, 32);
        if (tx == 0) x[0] = readBases[indI-1];

        //uint8 qual_back = qual[1];
        //for (int i = 1; i > 0; i--)
        //    qual[i] = qual[i-1];
        //qual[0] = __shfl(qual_back, (tx+31) & 0x1f);
        //if (tx == 0) qual[0] = readBaseQualities[indI-1];
 
        float cache_tmp_back = cache_tmp[1];
        for (int i = 1; i > 0; i--)
            cache_tmp[i] = cache_tmp[i-1];
        cache_tmp[0] = __shfl(cache_tmp_back, (tx+31) & 0x1f, 32);
        if (tx == 0) cache_tmp[0] = qualToErrorProbCache_cuda[(int)(readBaseQualities[indI-1]) & 0xFF];

//        float transition_tmp_back[4];
//        transition_tmp_back[0] = transition_tmp[1][MATCH_TO_MATCH];
//        transition_tmp_back[1] = transition_tmp[1][MATCH_TO_INSERTION];
//        transition_tmp_back[2] = transition_tmp[1][MATCH_TO_DELETION];
//        transition_tmp_back[3] = transition_tmp[1][INDEL_TO_MATCH];
//        for (int i = 1; i > 0; i--)
//        {
//            transition_tmp[i][MATCH_TO_MATCH] = transition_tmp[i-1][MATCH_TO_MATCH];
//            transition_tmp[i][MATCH_TO_INSERTION] = transition_tmp[i-1][MATCH_TO_INSERTION];
//            transition_tmp[i][MATCH_TO_DELETION] = transition_tmp[i-1][MATCH_TO_DELETION];
//            transition_tmp[i][INDEL_TO_MATCH] = transition_tmp[i-1][INDEL_TO_MATCH];
//        }
//        transition_tmp[0][MATCH_TO_MATCH] = __shfl(transition_tmp_back[0], (tx+31) & 0x1f, 32);
//        transition_tmp[0][MATCH_TO_INSERTION] = __shfl(transition_tmp_back[1], (tx+31) & 0x1f, 32);
//        transition_tmp[0][MATCH_TO_DELETION] = __shfl(transition_tmp_back[2], (tx+31) & 0x1f, 32);
//        transition_tmp[0][INDEL_TO_MATCH] = __shfl(transition_tmp_back[3], (tx+31) & 0x1f, 32);
//        if (tx == 0)
//        {
//            transition_tmp[0][MATCH_TO_MATCH] = transition[(indI-0)*TRANS_PROB_ARRAY_LENGTH+MATCH_TO_MATCH];
//            transition_tmp[0][MATCH_TO_INSERTION] = transition[(indI-0)*TRANS_PROB_ARRAY_LENGTH+MATCH_TO_INSERTION];
//            transition_tmp[0][MATCH_TO_DELETION] = transition[(indI-0)*TRANS_PROB_ARRAY_LENGTH+MATCH_TO_DELETION];
//            transition_tmp[0][INDEL_TO_MATCH] = transition[(indI-0)*TRANS_PROB_ARRAY_LENGTH+INDEL_TO_MATCH];
//        }

#pragma unroll
        for (int i = 0; i < 2; i++){
            if (i == 0){
                matchMatrixLeft = __shfl(matchLine2[1], (tx+31) & 0x1f);
                matchMatrixLeftUp = __shfl(matchLine3[1], (tx+31) & 0x1f);
                matchMatrixUp = matchLine2[i];

                insertionMatrixLeft = __shfl(insertionLine2[1], (tx+31) & 0x1f);
                insertionMatrixLeftUp = __shfl(insertionLine3[1], (tx+31) & 0x1f);
                insertionMatrixUp = insertionLine2[i];

                deletionMatrixLeft = __shfl(deletionLine2[1], (tx+31) & 0x1f);
                deletionMatrixLeftUp = (indI-i == 1)? initialValue: __shfl(deletionLine3[1], (tx+31) & 0x1f);
            } else {
                matchMatrixLeft = matchLine2[i-1];
                matchMatrixLeftUp = matchLine3[i-1];
                matchMatrixUp = matchLine2[i];

                insertionMatrixLeft = insertionLine2[i-1];
                insertionMatrixLeftUp = insertionLine3[i-1];
                insertionMatrixUp = insertionLine2[i];

                deletionMatrixLeft = deletionLine2[i-1];
                deletionMatrixLeftUp = (indI-i == 1)? initialValue: deletionLine3[i-1];                
            }
            if (tx*2+i < nBound)
            {
	    		c1[i] = 1 - cache_tmp[i];
	    		c2[i] = cache_tmp[i] / TRISTATE_CORRECTION;

	    		uint8 y = haplotypeBases[indJ+i-1];
	    		float prior = (x[i] == y || x[i] == (uint8)'N' || y == (uint8)'N' ? c1[i] : c2[i]);

                transition_tmp[i][MATCH_TO_MATCH] = transition[(indI-i)*TRANS_PROB_ARRAY_LENGTH+MATCH_TO_MATCH];
                transition_tmp[i][MATCH_TO_INSERTION] = transition[(indI-i)*TRANS_PROB_ARRAY_LENGTH+MATCH_TO_INSERTION];
                transition_tmp[i][MATCH_TO_DELETION] = transition[(indI-i)*TRANS_PROB_ARRAY_LENGTH+MATCH_TO_DELETION];
                transition_tmp[i][INDEL_TO_MATCH] = transition[(indI-i)*TRANS_PROB_ARRAY_LENGTH+INDEL_TO_MATCH];
                transition_tmp[i][INSERTION_TO_INSERTION] = 1 - transition_tmp[i][INDEL_TO_MATCH];
                transition_tmp[i][DELETION_TO_DELETION] = transition_tmp[i][INSERTION_TO_INSERTION];
	    		
                matchLine1[i] = prior*(matchMatrixLeftUp*transition_tmp[i][MATCH_TO_MATCH]
                        +insertionMatrixLeftUp*transition_tmp[i][INDEL_TO_MATCH]
                        +deletionMatrixLeftUp*transition_tmp[i][INDEL_TO_MATCH]);
               	insertionLine1[i] = matchMatrixUp*transition_tmp[i][MATCH_TO_INSERTION]
                        +insertionMatrixUp*transition_tmp[i][INSERTION_TO_INSERTION];
	    		deletionLine1[i] = matchMatrixLeft*transition_tmp[i][MATCH_TO_DELETION] 
                        +deletionMatrixLeft*transition_tmp[i][DELETION_TO_DELETION];

                if (i == 0){
                    if (tx == 0){
                        if (layerIndex >= readLength) {
	        			    likelihoodTmp += matchMatrixLeft + insertionMatrixLeft;
	    	    	    }
                    }
                }
	        }
        }
#pragma unroll
        for (int i = 0; i < 2; ++i){
            matchLine3[i] = matchLine2[i]; matchLine2[i] = matchLine1[i];
	    	insertionLine3[i] = insertionLine2[i]; insertionLine2[i] = insertionLine1[i];
	    	deletionLine3[i] = deletionLine2[i]; deletionLine2[i] = deletionLine1[i];	
        }
	} // for layerIndex

//    if (tx < 31) {
        float matchTmp = matchLine3[1];
        float insertionTmp = insertionLine3[1];
        float deletionTmp = deletionLine3[1];
    if (tx < 31) {
#pragma unroll
        for (int i = 2-1; i >= 0; i--){
            if (i == 0){
                matchLine3[i] = __shfl(matchTmp, (tx+31) & 0x1f);
                insertionLine3[i] = __shfl(insertionTmp, (tx+31) & 0x1f);
                deletionLine3[i] = __shfl(deletionTmp, (tx+31) & 0x1f);
            }else{
                matchLine3[i] = matchLine3[i-1];
                insertionLine3[i] = insertionLine3[i-1];
                deletionLine3[i] = deletionLine3[i-1];
            }
        }
    }

	for (int layerIndex = readLength; layerIndex < mBound - 1; ++layerIndex){
		if (layerIndex < minLength) {
			nBound = layerIndex + 1;
		} else {
			if (layerIndex >= maxLength) {
				nBound = -layerIndex + haplotypeLengthCur + readLength - 1;
			} else {
				nBound = minLength;
			}
		}

		indI = readLength - tx*2;
        indJ = layerIndex + 1 - readLength + tx*2 + 1;
#pragma unroll
        for (int i = 0; i < 2; i++){
            if (i == 1){
                matchMatrixLeft = matchLine2[i];
                matchMatrixLeftUp = __shfl(matchLine3[0], tx+1);
                matchMatrixUp = __shfl(matchLine2[0], tx+1);
            		
                insertionMatrixLeft = insertionLine2[i];
                insertionMatrixLeftUp = __shfl(insertionLine3[0], tx+1);
                insertionMatrixUp = __shfl(insertionLine2[0], tx+1);
            
                deletionMatrixLeft = deletionLine2[i];
                deletionMatrixLeftUp = __shfl(deletionLine3[0], tx+1);
                if (indI-i == 1) deletionMatrixLeftUp = initialValue;
            }else{
                matchMatrixLeft = matchLine2[i];
                matchMatrixLeftUp = matchLine3[i+1];
                matchMatrixUp = matchLine2[i+1];
            		
                insertionMatrixLeft = insertionLine2[i];
                insertionMatrixLeftUp = insertionLine3[i+1];
                insertionMatrixUp = insertionLine2[i+1];
            
                deletionMatrixLeft = deletionLine2[i];
                deletionMatrixLeftUp = deletionLine3[i+1];
                if (indI-i == 1) deletionMatrixLeftUp = initialValue;
            }
 
            if (tx*2+i < nBound) {
	    		uint8 y = haplotypeBases[indJ+i-1];
	    		float prior = (x[i] == y || x[i] == (uint8)'N' || y == (uint8)'N' ? c1[i] : c2[i]);
	 
                matchLine1[i] = prior*(matchMatrixLeftUp*transition_tmp[i][MATCH_TO_MATCH]
                        +insertionMatrixLeftUp*transition_tmp[i][INDEL_TO_MATCH]
                        +deletionMatrixLeftUp*transition_tmp[i][INDEL_TO_MATCH]);
               	insertionLine1[i] = matchMatrixUp*transition_tmp[i][MATCH_TO_INSERTION]
                        +insertionMatrixUp*transition_tmp[i][INSERTION_TO_INSERTION];
	    		deletionLine1[i] = matchMatrixLeft*transition_tmp[i][MATCH_TO_DELETION] 
                        +deletionMatrixLeft*transition_tmp[i][DELETION_TO_DELETION];

                if (i == 0){
                    if (tx == 0){
	    		        if (layerIndex >= readLength) {
	    			        likelihoodTmp += matchMatrixLeft + insertionMatrixLeft;
    	    		    }
                    }
                }	
	    	}	
        }
#pragma unroll   
        for (int i = 0; i < 2; ++i){
            matchLine3[i] = matchLine2[i]; matchLine2[i] = matchLine1[i];
	    	insertionLine3[i] = insertionLine2[i]; insertionLine2[i] = insertionLine1[i];
	    	deletionLine3[i] = deletionLine2[i]; deletionLine2[i] = deletionLine1[i];	
        }
	} // for layerIndex

	likelihoodTmp += matchLine1[0] + insertionLine1[0];
    *likelihood = likelihoodTmp;
}


__device__ void wrapper3(			
			uint8 *haplotypeBases,
            uint8 *readBases,
			uint8 *readBaseQualities,
            int haplotypeLength,
            int readLength,
            float *transition,
            float *likelihood,
            float initialValue
)
{
//	float MAX = powf(2, 120); // float
//	float initialValue;
//	initialValue = MAX / haplotypeLength;

	int nBound = 0;
	int haplotypeLengthCur = 0;
	haplotypeLengthCur = haplotypeLength;

	int minLength = (haplotypeLengthCur < readLength)? haplotypeLengthCur : readLength;
	int maxLength = (haplotypeLengthCur > readLength)? haplotypeLengthCur : readLength;

	int mBound = haplotypeLengthCur + readLength;
	
	int indI = 0;
	int indJ = 0;
	
	int tx = threadIdx.x;
    float likelihoodTmp = 0.0;

    float matchLine1[3], matchLine2[3], matchLine3[3];
    float insertionLine1[3], insertionLine2[3], insertionLine3[3];
    float deletionLine1[3], deletionLine2[3], deletionLine3[3];

    #pragma unroll
    for (int i = 0; i < 3; i++){
        matchLine1[i] = 0.0;
        matchLine2[i] = 0.0;
        matchLine3[i] = 0.0;
        insertionLine1[i] = 0.0;
        insertionLine2[i] = 0.0;
        insertionLine3[i] = 0.0;
        deletionLine1[i] = 0.0;
        deletionLine2[i] = 0.0;
        deletionLine3[i] = 0.0;
    }

    float cache_tmp[3] = {0.0};
    uint8 x[3] = {0.0};
    //uint8 qual[3] = {0.0};
    float c1[3], c2[3];

	float deletionMatrixLeftUp;
	float deletionMatrixLeft;

	float matchMatrixLeftUp;
	float matchMatrixUp;
	float matchMatrixLeft;
		
	float insertionMatrixUp;
	float insertionMatrixLeftUp;
	float insertionMatrixLeft;
	
    float transition_tmp[3][6];

    for (int layerIndex = 0; layerIndex < readLength; layerIndex ++){
		indI = layerIndex + 1 - tx*3;
        indJ = tx*3 + 1;
		
		if (layerIndex < minLength) {
			nBound = layerIndex + 1;
		} else {
			if (layerIndex >= maxLength) {
				nBound = -layerIndex + haplotypeLengthCur + readLength - 1;
			} else {
				nBound = minLength;
			}
		}

        uint8 x_back = x[2];
        for (int i = 2; i > 0; i--)
            x[i] = x[i-1];
        x[0] = __shfl(x_back, (tx+31) & 0x1f, 32);
        if (tx == 0) x[0] = readBases[indI-1];

        //uint8 qual_back = qual[2];
        //for (int i = 2; i > 0; i--)
        //    qual[i] = qual[i-1];
        //qual[0] = __shfl(qual_back, (tx+31) & 0x1f);
        //if (tx == 0) qual[0] = readBaseQualities[indI-1];
 
        float cache_tmp_back = cache_tmp[2];
        for (int i = 2; i > 0; i--)
            cache_tmp[i] = cache_tmp[i-1];
        cache_tmp[0] = __shfl(cache_tmp_back, (tx + 31) & 0x1f, 32);
        if (tx == 0) cache_tmp[0] = qualToErrorProbCache_cuda[(int)(readBaseQualities[indI-1]) & 0xFF];

//        float transition_tmp_back[4];
//        transition_tmp_back[0] = transition_tmp[2][MATCH_TO_MATCH];
//        transition_tmp_back[1] = transition_tmp[2][MATCH_TO_INSERTION];
//        transition_tmp_back[2] = transition_tmp[2][MATCH_TO_DELETION];
//        transition_tmp_back[3] = transition_tmp[2][INDEL_TO_MATCH];
//        for (int i = 2; i > 0; i--)
//        {
//            transition_tmp[i][MATCH_TO_MATCH] = transition_tmp[i-1][MATCH_TO_MATCH];
//            transition_tmp[i][MATCH_TO_INSERTION] = transition_tmp[i-1][MATCH_TO_INSERTION];
//            transition_tmp[i][MATCH_TO_DELETION] = transition_tmp[i-1][MATCH_TO_DELETION];
//            transition_tmp[i][INDEL_TO_MATCH] = transition_tmp[i-1][INDEL_TO_MATCH];
//        }
//        transition_tmp[0][MATCH_TO_MATCH] = __shfl(transition_tmp_back[0], (tx+31) & 0x1f, 32);
//        transition_tmp[0][MATCH_TO_INSERTION] = __shfl(transition_tmp_back[1], (tx+31) & 0x1f, 32);
//        transition_tmp[0][MATCH_TO_DELETION] = __shfl(transition_tmp_back[2], (tx+31) & 0x1f, 32);
//        transition_tmp[0][INDEL_TO_MATCH] = __shfl(transition_tmp_back[3], (tx+31) & 0x1f, 32);
//        if (tx == 0)
//        {
//            transition_tmp[0][MATCH_TO_MATCH] = transition[(indI-0)*TRANS_PROB_ARRAY_LENGTH+MATCH_TO_MATCH];
//            transition_tmp[0][MATCH_TO_INSERTION] = transition[(indI-0)*TRANS_PROB_ARRAY_LENGTH+MATCH_TO_INSERTION];
//            transition_tmp[0][MATCH_TO_DELETION] = transition[(indI-0)*TRANS_PROB_ARRAY_LENGTH+MATCH_TO_DELETION];
//            transition_tmp[0][INDEL_TO_MATCH] = transition[(indI-0)*TRANS_PROB_ARRAY_LENGTH+INDEL_TO_MATCH];
//        }

#pragma unroll
        for (int i = 0; i < 3; i++){
            if (i == 0){
                matchMatrixLeft = __shfl(matchLine2[2], (tx+31) & 0x1f);
                matchMatrixLeftUp = __shfl(matchLine3[2], (tx+31) & 0x1f);
                matchMatrixUp = matchLine2[i];

                insertionMatrixLeft = __shfl(insertionLine2[2], (tx+31) & 0x1f);
                insertionMatrixLeftUp = __shfl(insertionLine3[2], (tx+31) & 0x1f);
                insertionMatrixUp = insertionLine2[i];

                deletionMatrixLeft = __shfl(deletionLine2[2], (tx+31) & 0x1f);
                deletionMatrixLeftUp = (indI-i == 1)? initialValue: __shfl(deletionLine3[2], (tx+31) & 0x1f);
            } else {
                matchMatrixLeft = matchLine2[i-1];
                matchMatrixLeftUp = matchLine3[i-1];
                matchMatrixUp = matchLine2[i];

                insertionMatrixLeft = insertionLine2[i-1];
                insertionMatrixLeftUp = insertionLine3[i-1];
                insertionMatrixUp = insertionLine2[i];

                deletionMatrixLeft = deletionLine2[i-1];
                deletionMatrixLeftUp = (indI-i == 1)? initialValue: deletionLine3[i-1];                
            }
            if (tx*3+i < nBound)
            {
	    		c1[i] = 1 - cache_tmp[i];
	    		c2[i] = cache_tmp[i] / TRISTATE_CORRECTION;

	    		uint8 y = haplotypeBases[indJ+i-1];
	    		float prior = (x[i] == y || x[i] == (uint8)'N' || y == (uint8)'N' ? c1[i] : c2[i]);

                transition_tmp[i][MATCH_TO_MATCH] = transition[(indI-i)*TRANS_PROB_ARRAY_LENGTH+MATCH_TO_MATCH];
                transition_tmp[i][MATCH_TO_INSERTION] = transition[(indI-i)*TRANS_PROB_ARRAY_LENGTH+MATCH_TO_INSERTION];
                transition_tmp[i][MATCH_TO_DELETION] = transition[(indI-i)*TRANS_PROB_ARRAY_LENGTH+MATCH_TO_DELETION];
                transition_tmp[i][INDEL_TO_MATCH] = transition[(indI-i)*TRANS_PROB_ARRAY_LENGTH+INDEL_TO_MATCH];
                transition_tmp[i][INSERTION_TO_INSERTION] = 1 - transition_tmp[i][INDEL_TO_MATCH];
                transition_tmp[i][DELETION_TO_DELETION] = transition_tmp[i][INSERTION_TO_INSERTION];
               
                matchLine1[i] = prior*(matchMatrixLeftUp*transition_tmp[i][MATCH_TO_MATCH]
                        +insertionMatrixLeftUp*transition_tmp[i][INDEL_TO_MATCH]
                        +deletionMatrixLeftUp*transition_tmp[i][INDEL_TO_MATCH]);
               	insertionLine1[i] = matchMatrixUp*transition_tmp[i][MATCH_TO_INSERTION]
                        +insertionMatrixUp*transition_tmp[i][INSERTION_TO_INSERTION];
	    		deletionLine1[i] = matchMatrixLeft*transition_tmp[i][MATCH_TO_DELETION] 
                        +deletionMatrixLeft*transition_tmp[i][DELETION_TO_DELETION];
    
                if (i == 0){
                    if (tx == 0){
                        if (layerIndex >= readLength) {
	        			    likelihoodTmp += matchMatrixLeft + insertionMatrixLeft;
	    	    	    }
                    }
                }
	        }
        }
#pragma unroll
        for (int i = 0; i < 3; ++i){
            matchLine3[i] = matchLine2[i]; matchLine2[i] = matchLine1[i];
	    	insertionLine3[i] = insertionLine2[i]; insertionLine2[i] = insertionLine1[i];
	    	deletionLine3[i] = deletionLine2[i]; deletionLine2[i] = deletionLine1[i];	
        }
	} // for layerIndex

//    if (tx < 31) {
        float matchTmp = matchLine3[2];
        float insertionTmp = insertionLine3[2];
        float deletionTmp = deletionLine3[2];
    if (tx < 31) {
#pragma unroll
        for (int i = 3-1; i >= 0; i--){
            if (i == 0){
                matchLine3[i] = __shfl(matchTmp, (tx+31) & 0x1f);
                insertionLine3[i] = __shfl(insertionTmp, (tx+31) & 0x1f);
                deletionLine3[i] = __shfl(deletionTmp, (tx+31) & 0x1f);
            }else{
                matchLine3[i] = matchLine3[i-1];
                insertionLine3[i] = insertionLine3[i-1];
                deletionLine3[i] = deletionLine3[i-1];
            }
        }
    }

	for (int layerIndex = readLength; layerIndex < mBound - 1; ++layerIndex){
		if (layerIndex < minLength) {
			nBound = layerIndex + 1;
		} else {
			if (layerIndex >= maxLength) {
				nBound = -layerIndex + haplotypeLengthCur + readLength - 1;
			} else {
				nBound = minLength;
			}
		}

		indI = readLength - tx*3;
        indJ = layerIndex + 1 - readLength + tx*3 + 1;
#pragma unroll
        for (int i = 0; i < 3; i++){
            if (i == 2){
                matchMatrixLeft = matchLine2[i];
                matchMatrixLeftUp = __shfl(matchLine3[0], tx+1);
                matchMatrixUp = __shfl(matchLine2[0], tx+1);
            		
                insertionMatrixLeft = insertionLine2[i];
                insertionMatrixLeftUp = __shfl(insertionLine3[0], tx+1);
                insertionMatrixUp = __shfl(insertionLine2[0], tx+1);
            
                deletionMatrixLeft = deletionLine2[i];
                deletionMatrixLeftUp = __shfl(deletionLine3[0], tx+1);
                if (indI-i == 1) deletionMatrixLeftUp = initialValue;
            }else{
                matchMatrixLeft = matchLine2[i];
                matchMatrixLeftUp = matchLine3[i+1];
                matchMatrixUp = matchLine2[i+1];
            		
                insertionMatrixLeft = insertionLine2[i];
                insertionMatrixLeftUp = insertionLine3[i+1];
                insertionMatrixUp = insertionLine2[i+1];
            
                deletionMatrixLeft = deletionLine2[i];
                deletionMatrixLeftUp = deletionLine3[i+1];
                if (indI-i == 1) deletionMatrixLeftUp = initialValue;
            }
 
            if (tx*3+i < nBound) {
	    		uint8 y = haplotypeBases[indJ+i-1];
	    		float prior = (x[i] == y || x[i] == (uint8)'N' || y == (uint8)'N' ? c1[i] : c2[i]);
	 
                matchLine1[i] = prior*(matchMatrixLeftUp*transition_tmp[i][MATCH_TO_MATCH]
                        +insertionMatrixLeftUp*transition_tmp[i][INDEL_TO_MATCH]
                        +deletionMatrixLeftUp*transition_tmp[i][INDEL_TO_MATCH]);
               	insertionLine1[i] = matchMatrixUp*transition_tmp[i][MATCH_TO_INSERTION]
                        +insertionMatrixUp*transition_tmp[i][INSERTION_TO_INSERTION];
	    		deletionLine1[i] = matchMatrixLeft*transition_tmp[i][MATCH_TO_DELETION] 
                        +deletionMatrixLeft*transition_tmp[i][DELETION_TO_DELETION];

                if (i == 0){
                    if (tx == 0){
	    		        if (layerIndex >= readLength) {
	    			        likelihoodTmp += matchMatrixLeft + insertionMatrixLeft;
    	    		    }
                    }
                }	
	    	}	
        }
#pragma unroll   
        for (int i = 0; i < 3; ++i){
            matchLine3[i] = matchLine2[i]; matchLine2[i] = matchLine1[i];
	    	insertionLine3[i] = insertionLine2[i]; insertionLine2[i] = insertionLine1[i];
	    	deletionLine3[i] = deletionLine2[i]; deletionLine2[i] = deletionLine1[i];	
        }
	} // for layerIndex

	likelihoodTmp += matchLine1[0] + insertionLine1[0];
    *likelihood = likelihoodTmp;
}

__device__ void wrapper4(			
			uint8 *haplotypeBases,
            uint8 *readBases,
			uint8 *readBaseQualities,
            int haplotypeLength,
            int readLength,
            float *transition,
            float *likelihood,
            float initialValue
)
{
//	float MAX = powf(2, 120); // float
//	float initialValue;
//	initialValue = MAX / haplotypeLength;

	int nBound = 0;
	int haplotypeLengthCur = 0;
	haplotypeLengthCur = haplotypeLength;

	int minLength = (haplotypeLengthCur < readLength)? haplotypeLengthCur : readLength;
	int maxLength = (haplotypeLengthCur > readLength)? haplotypeLengthCur : readLength;

	int mBound = haplotypeLengthCur + readLength;
	
	int indI = 0;
	int indJ = 0;
	
	int tx = threadIdx.x;
    float likelihoodTmp = 0.0;

    float matchLine1[4], matchLine2[4], matchLine3[4];
    float insertionLine1[4], insertionLine2[4], insertionLine3[4];
    float deletionLine1[4], deletionLine2[4], deletionLine3[4];

    #pragma unroll
    for (int i = 0; i < 4; i++){
        matchLine1[i] = 0.0;
        matchLine2[i] = 0.0;
        matchLine3[i] = 0.0;
        insertionLine1[i] = 0.0;
        insertionLine2[i] = 0.0;
        insertionLine3[i] = 0.0;
        deletionLine1[i] = 0.0;
        deletionLine2[i] = 0.0;
        deletionLine3[i] = 0.0;
    }
    float cache_tmp[4] = {0.0};
    uint8 x[4] = {0.0};
    //uint8 qual[4] = {0.0};
    float transition_tmp[4][6];
    float c1[4], c2[4];

	float deletionMatrixLeftUp;
	float deletionMatrixLeft;

	float matchMatrixLeftUp;
	float matchMatrixUp;
	float matchMatrixLeft;
		
	float insertionMatrixUp;
	float insertionMatrixLeftUp;
	float insertionMatrixLeft;
	
    for (int layerIndex = 0; layerIndex < readLength; layerIndex ++){
		indI = layerIndex + 1 - tx*4;
        indJ = tx*4 + 1;
		
		if (layerIndex < minLength) {
			nBound = layerIndex + 1;
		} else {
			if (layerIndex >= maxLength) {
				nBound = -layerIndex + haplotypeLengthCur + readLength - 1;
			} else {
				nBound = minLength;
			}
		}

        uint8 x_back = x[3];
        for (int i = 3; i > 0; i--)
            x[i] = x[i-1];
        x[0] = __shfl(x_back, (tx+31) & 0x1f, 32);
        if (tx == 0) x[0] = readBases[indI-1];

        //uint8 qual_back = qual[3];
        //for (int i = 3; i > 0; i--)
        //    qual[i] = qual[i-1];
        //qual[0] = __shfl(qual_back, (tx+31) & 0x1f);
        //if (tx == 0) qual[0] = readBaseQualities[indI-1];

        float cache_tmp_back = cache_tmp[3];
        for (int i = 3; i > 0; i--)
            cache_tmp[i] = cache_tmp[i-1];
        cache_tmp[0] = __shfl(cache_tmp_back, (tx+31) & 0x1f, 32);
        if (tx == 0) cache_tmp[0] = qualToErrorProbCache_cuda[(int)(readBaseQualities[indI-1]) & 0xFF];

//        float transition_tmp_back[4];
//        transition_tmp_back[0] = transition_tmp[3][MATCH_TO_MATCH];
//        transition_tmp_back[1] = transition_tmp[3][MATCH_TO_INSERTION];
//        transition_tmp_back[2] = transition_tmp[3][MATCH_TO_DELETION];
//        transition_tmp_back[3] = transition_tmp[3][INDEL_TO_MATCH];
//        for (int i = 3; i > 0; i--)
//        {
//            transition_tmp[i][MATCH_TO_MATCH] = transition_tmp[i-1][MATCH_TO_MATCH];
//            transition_tmp[i][MATCH_TO_INSERTION] = transition_tmp[i-1][MATCH_TO_INSERTION];
//            transition_tmp[i][MATCH_TO_DELETION] = transition_tmp[i-1][MATCH_TO_DELETION];
//            transition_tmp[i][INDEL_TO_MATCH] = transition_tmp[i-1][INDEL_TO_MATCH];
//        }
//        transition_tmp[0][MATCH_TO_MATCH] = __shfl(transition_tmp_back[0], (tx+31) & 0x1f, 32);
//        transition_tmp[0][MATCH_TO_INSERTION] = __shfl(transition_tmp_back[1], (tx+31) & 0x1f, 32);
//        transition_tmp[0][MATCH_TO_DELETION] = __shfl(transition_tmp_back[2], (tx+31) & 0x1f, 32);
//        transition_tmp[0][INDEL_TO_MATCH] = __shfl(transition_tmp_back[3], (tx+31) & 0x1f, 32);
//        if (tx == 0)
//        {
//            transition_tmp[0][MATCH_TO_MATCH] = transition[(indI-0)*TRANS_PROB_ARRAY_LENGTH+MATCH_TO_MATCH];
//            transition_tmp[0][MATCH_TO_INSERTION] = transition[(indI-0)*TRANS_PROB_ARRAY_LENGTH+MATCH_TO_INSERTION];
//            transition_tmp[0][MATCH_TO_DELETION] = transition[(indI-0)*TRANS_PROB_ARRAY_LENGTH+MATCH_TO_DELETION];
//            transition_tmp[0][INDEL_TO_MATCH] = transition[(indI-0)*TRANS_PROB_ARRAY_LENGTH+INDEL_TO_MATCH];
//        }

#pragma unroll
        for (int i = 0; i < 4; i++){
            if (i == 0){
                matchMatrixLeft = __shfl(matchLine2[3], (tx+31) & 0x1f);
                matchMatrixLeftUp = __shfl(matchLine3[3], (tx+31) & 0x1f);
                matchMatrixUp = matchLine2[i];

                insertionMatrixLeft = __shfl(insertionLine2[3], (tx+31) & 0x1f);
                insertionMatrixLeftUp = __shfl(insertionLine3[3], (tx+31) & 0x1f);
                insertionMatrixUp = insertionLine2[i];

                deletionMatrixLeft = __shfl(deletionLine2[3], (tx+31) & 0x1f);
                deletionMatrixLeftUp = (indI-i == 1)? initialValue: __shfl(deletionLine3[3], (tx+31) & 0x1f);
            } else {
                matchMatrixLeft = matchLine2[i-1];
                matchMatrixLeftUp = matchLine3[i-1];
                matchMatrixUp = matchLine2[i];

                insertionMatrixLeft = insertionLine2[i-1];
                insertionMatrixLeftUp = insertionLine3[i-1];
                insertionMatrixUp = insertionLine2[i];

                deletionMatrixLeft = deletionLine2[i-1];
                deletionMatrixLeftUp = (indI-i == 1)? initialValue: deletionLine3[i-1];                
            }
            if (tx*4+i < nBound)
            {
	    		c1[i] = 1 - cache_tmp[i];
	    		c2[i] = cache_tmp[i] / TRISTATE_CORRECTION;

	    		uint8 y = haplotypeBases[indJ+i-1];
	    		float prior = (x[i] == y || x[i] == (uint8)'N' || y == (uint8)'N' ? c1[i] : c2[i]);

                transition_tmp[i][MATCH_TO_MATCH] = transition[(indI-i)*TRANS_PROB_ARRAY_LENGTH+MATCH_TO_MATCH];
                transition_tmp[i][MATCH_TO_INSERTION] = transition[(indI-i)*TRANS_PROB_ARRAY_LENGTH+MATCH_TO_INSERTION];
                transition_tmp[i][MATCH_TO_DELETION] = transition[(indI-i)*TRANS_PROB_ARRAY_LENGTH+MATCH_TO_DELETION];
                transition_tmp[i][INDEL_TO_MATCH] = transition[(indI-i)*TRANS_PROB_ARRAY_LENGTH+INDEL_TO_MATCH];
                transition_tmp[i][INSERTION_TO_INSERTION] = 1 - transition_tmp[i][INDEL_TO_MATCH];
                transition_tmp[i][DELETION_TO_DELETION] = transition_tmp[i][INSERTION_TO_INSERTION];
               
                matchLine1[i] = prior*(matchMatrixLeftUp*transition_tmp[i][MATCH_TO_MATCH]
                        +insertionMatrixLeftUp*transition_tmp[i][INDEL_TO_MATCH]
                        +deletionMatrixLeftUp*transition_tmp[i][INDEL_TO_MATCH]);
               	insertionLine1[i] = matchMatrixUp*transition_tmp[i][MATCH_TO_INSERTION]
                        +insertionMatrixUp*transition_tmp[i][INSERTION_TO_INSERTION];
	    		deletionLine1[i] = matchMatrixLeft*transition_tmp[i][MATCH_TO_DELETION] 
                        +deletionMatrixLeft*transition_tmp[i][DELETION_TO_DELETION];

                if (i == 0){
                    if (tx == 0){
                        if (layerIndex >= readLength) {
	        			    likelihoodTmp += matchMatrixLeft + insertionMatrixLeft;
	    	    	    }
                    }
                }
	        }
        }
#pragma unroll
        for (int i = 0; i < 4; ++i){
            matchLine3[i] = matchLine2[i]; matchLine2[i] = matchLine1[i];
	    	insertionLine3[i] = insertionLine2[i]; insertionLine2[i] = insertionLine1[i];
	    	deletionLine3[i] = deletionLine2[i]; deletionLine2[i] = deletionLine1[i];	
        }
	} // for layerIndex

//    if (tx < 31) {
        float matchTmp = matchLine3[3];
        float insertionTmp = insertionLine3[3];
        float deletionTmp = deletionLine3[3];
    if (tx < 31) {
#pragma unroll
        for (int i = 4-1; i >= 0; i--){
            if (i == 0){
                matchLine3[i] = __shfl(matchTmp, (tx+31) & 0x1f);
                insertionLine3[i] = __shfl(insertionTmp, (tx+31) & 0x1f);
                deletionLine3[i] = __shfl(deletionTmp, (tx+31) & 0x1f);
            }else{
                matchLine3[i] = matchLine3[i-1];
                insertionLine3[i] = insertionLine3[i-1];
                deletionLine3[i] = deletionLine3[i-1];
            }
        }
    }

	for (int layerIndex = readLength; layerIndex < mBound - 1; ++layerIndex){
		if (layerIndex < minLength) {
			nBound = layerIndex + 1;
		} else {
			if (layerIndex >= maxLength) {
				nBound = -layerIndex + haplotypeLengthCur + readLength - 1;
			} else {
				nBound = minLength;
			}
		}

		indI = readLength - tx*4;
        indJ = layerIndex + 1 - readLength + tx*4 + 1;
#pragma unroll
        for (int i = 0; i < 4; i++){
            if (i == 3){
                matchMatrixLeft = matchLine2[i];
                matchMatrixLeftUp = __shfl(matchLine3[0], tx+1);
                matchMatrixUp = __shfl(matchLine2[0], tx+1);
            		
                insertionMatrixLeft = insertionLine2[i];
                insertionMatrixLeftUp = __shfl(insertionLine3[0], tx+1);
                insertionMatrixUp = __shfl(insertionLine2[0], tx+1);
            
                deletionMatrixLeft = deletionLine2[i];
                deletionMatrixLeftUp = __shfl(deletionLine3[0], tx+1);
                if (indI-i == 1) deletionMatrixLeftUp = initialValue;
            }else{
                matchMatrixLeft = matchLine2[i];
                matchMatrixLeftUp = matchLine3[i+1];
                matchMatrixUp = matchLine2[i+1];
            		
                insertionMatrixLeft = insertionLine2[i];
                insertionMatrixLeftUp = insertionLine3[i+1];
                insertionMatrixUp = insertionLine2[i+1];
            
                deletionMatrixLeft = deletionLine2[i];
                deletionMatrixLeftUp = deletionLine3[i+1];
                if (indI-i == 1) deletionMatrixLeftUp = initialValue;
            }
 
            if (tx*4+i < nBound) {
	    		uint8 y = haplotypeBases[indJ+i-1];
	    		float prior = (x[i] == y || x[i] == (uint8)'N' || y == (uint8)'N' ? c1[i] : c2[i]);
	 
                matchLine1[i] = prior*(matchMatrixLeftUp*transition_tmp[i][MATCH_TO_MATCH]
                        +insertionMatrixLeftUp*transition_tmp[i][INDEL_TO_MATCH]
                        +deletionMatrixLeftUp*transition_tmp[i][INDEL_TO_MATCH]);
               	insertionLine1[i] = matchMatrixUp*transition_tmp[i][MATCH_TO_INSERTION]
                        +insertionMatrixUp*transition_tmp[i][INSERTION_TO_INSERTION];
	    		deletionLine1[i] = matchMatrixLeft*transition_tmp[i][MATCH_TO_DELETION] 
                        +deletionMatrixLeft*transition_tmp[i][DELETION_TO_DELETION];

                if (i == 0){
                    if (tx == 0){
	    		        if (layerIndex >= readLength) {
	    			        likelihoodTmp += matchMatrixLeft + insertionMatrixLeft;
    	    		    }
                    }
                }	
	    	}	
        }
#pragma unroll   
        for (int i = 0; i < 4; ++i){
            matchLine3[i] = matchLine2[i]; matchLine2[i] = matchLine1[i];
	    	insertionLine3[i] = insertionLine2[i]; insertionLine2[i] = insertionLine1[i];
	    	deletionLine3[i] = deletionLine2[i]; deletionLine2[i] = deletionLine1[i];	
        }
	} // for layerIndex

	likelihoodTmp += matchLine1[0] + insertionLine1[0];
    *likelihood = likelihoodTmp;
}

__global__ void kernel1(
		uint8 *read_in,
        int *readLength_in,
		uint8 *hap_in,
        int *hapLength_in,
		float* mLikelihoodArray,
		int readNum,
		int haplotypeNum,
		float *transition_in
){
    int readIter = blockIdx.x;
    int haplotypeIndex = blockIdx.y;

    int readLengthCur1 = readLength_in[readIter];
    int haplotypeLengthCur1 = hapLength_in[haplotypeIndex];
    int tx = threadIdx.x;
	
	__shared__ uint8 readCur1[MAX_READ_LENGTH1];
	__shared__ uint8 readBaseQualitiesCur1[MAX_READ_LENGTH1];
	__shared__ uint8 haplotypeCur1[MAX_HAPLOTYPE_LENGTH];
    __shared__ float transition1[PADDED_MAX_READ_LENGTH1*TRANS_PROB_ARRAY_LENGTH];

    for (int i = tx; i < readLengthCur1; i+=blockDim.x){
		readCur1[i] = read_in[readIter*MAX_READ_LENGTH*5+i];        
		readBaseQualitiesCur1[i] = read_in[readIter*MAX_READ_LENGTH*5+MAX_READ_LENGTH+i];
    }
    
	for (int i = tx; i < haplotypeLengthCur1; i+=blockDim.x){
		haplotypeCur1[i] = hap_in[haplotypeIndex*MAX_HAPLOTYPE_LENGTH+i];
	}

    for (int i = tx; i < (readLengthCur1+1)*TRANS_PROB_ARRAY_LENGTH; i+=blockDim.x){
        transition1[i] = transition_in[readIter*PADDED_MAX_READ_LENGTH*TRANS_PROB_ARRAY_LENGTH+i];
    }
	
	__syncthreads();
	
    float likelihood1;
    int minLength = (haplotypeLengthCur1 < readLengthCur1)? haplotypeLengthCur1 : readLengthCur1;

	float MAX = powf(2, 120); // float
	float initialValue;
	initialValue = MAX / haplotypeLengthCur1;

    if (minLength < 32){
        wrapper1(
            haplotypeCur1,readCur1,
         	readBaseQualitiesCur1,
         	haplotypeLengthCur1,readLengthCur1,
         	transition1,
            &likelihood1,
            initialValue
        ); 
    } else if (minLength < 63){
         wrapper2(
            haplotypeCur1,readCur1,
         	readBaseQualitiesCur1,
         	haplotypeLengthCur1,readLengthCur1,
         	transition1,
            &likelihood1,
            initialValue
        );
    }

    if (tx == 0){
    	//float INITIAL_CONDITION = powf(2, 120); // float
	    float INITIAL_CONDITION_LOG10 = log10f(MAX);
		mLikelihoodArray[readIter*haplotypeNum+haplotypeIndex] = log10f(likelihood1) - INITIAL_CONDITION_LOG10;
	}
}

__global__ void kernel2(
		uint8 *read_in,
        int *readLength_in,
		uint8 *hap_in,
        int *hapLength_in,
		float* mLikelihoodArray,
		int readNum,
		int haplotypeNum,
		float *transition_in
){
    int readIter = blockIdx.x;
    int haplotypeIndex = blockIdx.y;

    int readLengthCur1 = readLength_in[readIter];
    int haplotypeLengthCur1 = hapLength_in[haplotypeIndex];
    int tx = threadIdx.x;
	
	__shared__ uint8 readCur1[MAX_READ_LENGTH2];
	__shared__ uint8 readBaseQualitiesCur1[MAX_READ_LENGTH2];
	__shared__ uint8 haplotypeCur1[MAX_HAPLOTYPE_LENGTH];
    __shared__ float transition1[PADDED_MAX_READ_LENGTH2*TRANS_PROB_ARRAY_LENGTH];

    for (int i = tx; i < readLengthCur1; i+=blockDim.x){
		readCur1[i] = read_in[readIter*MAX_READ_LENGTH*5+i];        
		readBaseQualitiesCur1[i] = read_in[readIter*MAX_READ_LENGTH*5+MAX_READ_LENGTH+i];
    }
    
	for (int i = tx; i < haplotypeLengthCur1; i+=blockDim.x){
		haplotypeCur1[i] = hap_in[haplotypeIndex*MAX_HAPLOTYPE_LENGTH+i];
	}

    for (int i = tx; i < (readLengthCur1+1)*TRANS_PROB_ARRAY_LENGTH; i+=blockDim.x){
        transition1[i] = transition_in[readIter*PADDED_MAX_READ_LENGTH*TRANS_PROB_ARRAY_LENGTH+i];
    }
	
	__syncthreads();
	
    float likelihood1;
    int minLength = (haplotypeLengthCur1 < readLengthCur1)? haplotypeLengthCur1 : readLengthCur1;

	float MAX = powf(2, 120); // float
	float initialValue;
	initialValue = MAX / haplotypeLengthCur1;

    if (minLength < 32){
        wrapper1(
            haplotypeCur1,readCur1,
         	readBaseQualitiesCur1,
         	haplotypeLengthCur1,readLengthCur1,
         	transition1,
            &likelihood1,
            initialValue
        ); 
    } else if (minLength < 63){
         wrapper2(
            haplotypeCur1,readCur1,
         	readBaseQualitiesCur1,
         	haplotypeLengthCur1,readLengthCur1,
         	transition1,
            &likelihood1,
            initialValue
        );
    } else if (minLength < 94){
         wrapper3(
            haplotypeCur1,readCur1,
         	readBaseQualitiesCur1,
         	haplotypeLengthCur1,readLengthCur1,
         	transition1,
            &likelihood1,
            initialValue
        );
    }

    if (tx == 0){
    	//float INITIAL_CONDITION = powf(2, 120); // float
	    float INITIAL_CONDITION_LOG10 = log10f(MAX);
		mLikelihoodArray[readIter*haplotypeNum+haplotypeIndex] = log10f(likelihood1) - INITIAL_CONDITION_LOG10;
	}
}

__global__ void kernel3(
		uint8 *read_in,
        int *readLength_in,
		uint8 *hap_in,
        int *hapLength_in,
		float* mLikelihoodArray,
		int readNum,
		int haplotypeNum,
		float *transition_in
){
    int readIter = blockIdx.x;
    int haplotypeIndex = blockIdx.y;

    int readLengthCur1 = readLength_in[readIter];
    int haplotypeLengthCur1 = hapLength_in[haplotypeIndex];
    int tx = threadIdx.x;
	
	__shared__ uint8 readCur1[MAX_READ_LENGTH3];
	__shared__ uint8 readBaseQualitiesCur1[MAX_READ_LENGTH3];
	__shared__ uint8 haplotypeCur1[MAX_HAPLOTYPE_LENGTH];
    __shared__ float transition1[PADDED_MAX_READ_LENGTH3*TRANS_PROB_ARRAY_LENGTH];

    for (int i = tx; i < readLengthCur1; i+=blockDim.x){
		readCur1[i] = read_in[readIter*MAX_READ_LENGTH*5+i];        
		readBaseQualitiesCur1[i] = read_in[readIter*MAX_READ_LENGTH*5+MAX_READ_LENGTH+i];
    }
    
	for (int i = tx; i < haplotypeLengthCur1; i+=blockDim.x){
		haplotypeCur1[i] = hap_in[haplotypeIndex*MAX_HAPLOTYPE_LENGTH+i];
	}

    for (int i = tx; i < (readLengthCur1+1)*TRANS_PROB_ARRAY_LENGTH; i+=blockDim.x){
        transition1[i] = transition_in[readIter*PADDED_MAX_READ_LENGTH*TRANS_PROB_ARRAY_LENGTH+i];
    }
	
	__syncthreads();
	
    float likelihood1;
    int minLength = (haplotypeLengthCur1 < readLengthCur1)? haplotypeLengthCur1 : readLengthCur1;

	float MAX = powf(2, 120); // float
	float initialValue;
	initialValue = MAX / haplotypeLengthCur1;

    if (minLength < 32){
        wrapper1(
            haplotypeCur1,readCur1,
         	readBaseQualitiesCur1,
         	haplotypeLengthCur1,readLengthCur1,
         	transition1,
            &likelihood1,
            initialValue
        ); 
    } else if (minLength < 63){
         wrapper2(
            haplotypeCur1,readCur1,
         	readBaseQualitiesCur1,
         	haplotypeLengthCur1,readLengthCur1,
         	transition1,
            &likelihood1,
            initialValue
        );
    } else if (minLength < 94){
         wrapper3(
            haplotypeCur1,readCur1,
         	readBaseQualitiesCur1,
         	haplotypeLengthCur1,readLengthCur1,
         	transition1,
            &likelihood1,
            initialValue
        );
    } else if (minLength < 125){
         wrapper4(
            haplotypeCur1,readCur1,
         	readBaseQualitiesCur1,
         	haplotypeLengthCur1,readLengthCur1,
         	transition1,
            &likelihood1,
            initialValue
        );    
    }

    if (tx == 0){
    	//float INITIAL_CONDITION = powf(2, 120); // float
	    float INITIAL_CONDITION_LOG10 = log10f(MAX);
		mLikelihoodArray[readIter*haplotypeNum+haplotypeIndex] = log10f(likelihood1) - INITIAL_CONDITION_LOG10;
	}
}

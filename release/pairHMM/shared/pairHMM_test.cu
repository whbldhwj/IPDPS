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

__global__ void kernel1(
		uint8 *read_in,		
        int *readLength_in,
		uint8 *hap_in,				
        int *hapLength_in,
		float *mLikelihoodArray,		
        int readNum,
		int haplotypeNum,		
		//float qualToErrorProbCache1[MAX_QUAL+1],		
		float *transition1
);

__global__ void kernel2(
		uint8 *read_in,		
        int *readLength_in,
		uint8 *hap_in,				
        int *hapLength_in,
		float *mLikelihoodArray,		
        int readNum,
		int haplotypeNum,		
		//float qualToErrorProbCache1[MAX_QUAL+1],		
		float *transition1
);

__global__ void kernel3(
		uint8 *read_in,		
        int *readLength_in,
		uint8 *hap_in,				
        int *hapLength_in,
		float *mLikelihoodArray,		
        int readNum,
		int haplotypeNum,		
		//float qualToErrorProbCache1[MAX_QUAL+1],		
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
	//float *qualToErrorProbCache_cuda;

    cudaEvent_t start, stop;
	float elapsed_time;
	
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

    cudaMalloc((void**)& read_cuda, readNum * MAX_READ_LENGTH * 5 * sizeof(uint8));
    cudaMalloc((void**)& hap_cuda, hapNum * MAX_HAPLOTYPE_LENGTH * sizeof(uint8));
    cudaMalloc((void**)& readLen_cuda, readNum * sizeof(int));
    cudaMalloc((void**)& hapLen_cuda, hapNum * sizeof(int));
    cudaMalloc((void**)& mLikelihoodArray_cuda, sizeof(float) * readNum * hapNum);
	cudaMemset(mLikelihoodArray_cuda, 0, sizeof(float) * readNum * hapNum);
    //cudaMalloc((void**)& qualToErrorProbCache_cuda, sizeof(float) * (MAX_QUAL+1));
	cudaMalloc((void**)& transition_cuda, sizeof(float) * PADDED_MAX_READ_LENGTH * TRANS_PROB_ARRAY_LENGTH * readNum);

    cudaEventRecord(start, 0);

	cudaMemcpy(read_cuda, read_in, readNum * MAX_READ_LENGTH * 5 * sizeof(uint8), cudaMemcpyHostToDevice);
    cudaMemcpy(hap_cuda, hap_in, hapNum * MAX_HAPLOTYPE_LENGTH * sizeof(uint8), cudaMemcpyHostToDevice);
    cudaMemcpy(readLen_cuda, readLength_in, readNum * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(hapLen_cuda, hapLength_in, hapNum * sizeof(int), cudaMemcpyHostToDevice);
	//cudaMemcpy(qualToErrorProbCache_cuda, qualToErrorProbCache1, sizeof(float) * (MAX_QUAL+1), cudaMemcpyHostToDevice);	
    cudaMemcpyToSymbol(qualToErrorProbCache_cuda, qualToErrorProbCache1, sizeof(float) * (MAX_QUAL+1));

    dim3 dimGrid(readNum,hapNum);
	dim3 dimBlock(128,1);		
	
	//float transition1[READ_NUM][PADDED_MAX_READ_LENGTH][TRANS_PROB_ARRAY_LENGTH];

    trans_compute(
        readNum,
        read_in,
        readLength_in,
        transition1,
        qualToErrorProbCache1,
        matchToMatch1,
        jacobianLogTable);

    //for (int readIter = 0; readIter < readNum; readIter++){
    //    for (int i = 0; i < MAX_READ_LENGTH; i++){
 	//		readBaseInsertionQualitiesCur1[i] = read_in[readIter*MAX_READ_LENGTH*5+MAX_READ_LENGTH*2+i];
	//		readBaseDeletionQualitiesCur1[i] = read_in[readIter*MAX_READ_LENGTH*5+MAX_READ_LENGTH*3+i];
	//		readGCPCur1[i] = read_in[readIter*MAX_READ_LENGTH*5+MAX_READ_LENGTH*4+i];        
    //    }
    //    readLengthCur1 = readLength_in[readIter]; 		        
    //    initializeProbabilities(
	//			readBaseInsertionQualitiesCur1, readBaseDeletionQualitiesCur1, readGCPCur1,
	//			transition1[readIter],
	//			qualToErrorProbCache1, matchToMatch1, readLengthCur1
	//	);    
    //}

	cudaMemcpy(transition_cuda, transition1, sizeof(float)*PADDED_MAX_READ_LENGTH*TRANS_PROB_ARRAY_LENGTH*readNum, cudaMemcpyHostToDevice);	

    int readLenMax = 0;
    for (int i = 0; i < readNum; i++){
        if (readLength_in[i] > readLenMax)
            readLenMax = readLength_in[i];
    }

	//cudaEventRecord(start, 0);
    if (readLenMax <= MAX_READ_LENGTH1){
		kernel1<<<dimGrid, dimBlock>>>(
			read_cuda,
			readLen_cuda,
			hap_cuda,        
			hapLen_cuda,
			mLikelihoodArray_cuda,
			readNum,
			hapNum,        
			//qualToErrorProbCache_cuda,
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
			//qualToErrorProbCache_cuda,
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
			//qualToErrorProbCache_cuda,
			transition_cuda
    		);
    }

    //cudaEventRecord(stop, 0);
	//cudaEventSynchronize(stop);   
 
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
			//float qualToErrorProbCache[MAX_QUAL+1],
            float *likelihood
)
{
	float MAX = powf(2, 120); // float
	float initialValue;
	initialValue = MAX / haplotypeLength;

	int nBound = 0;
	int haplotypeLengthCur = 0;
	haplotypeLengthCur = haplotypeLength;

	int minLength = (haplotypeLengthCur < readLength)? haplotypeLengthCur : readLength;
	int maxLength = (haplotypeLengthCur > readLength)? haplotypeLengthCur : readLength;

	int mBound = haplotypeLengthCur + readLength;
	
	int indI = 0;
	int indJ = 0;
	
	float deletionResTmp1;
	float matchResTmp1;
	float insertionResTmp1;

//	int round_flag = 0;

	int tx = threadIdx.x;
    float likelihoodTmp = 0.0;

	__shared__ float matchLine1[PADDED_MAX_READ_LENGTH1];
	__shared__ float matchLine2[PADDED_MAX_READ_LENGTH1];
	__shared__ float matchLine3[PADDED_MAX_READ_LENGTH1];

	__shared__ float insertionLine1[PADDED_MAX_READ_LENGTH1];
	__shared__ float insertionLine2[PADDED_MAX_READ_LENGTH1];
	__shared__ float insertionLine3[PADDED_MAX_READ_LENGTH1];

	__shared__ float deletionLine1[PADDED_MAX_READ_LENGTH1];
	__shared__ float deletionLine2[PADDED_MAX_READ_LENGTH1];
	__shared__ float deletionLine3[PADDED_MAX_READ_LENGTH1];

    float *ptrMatch1 = &matchLine1[0];
    float *ptrMatch2 = &matchLine2[0];
    float *ptrMatch3 = &matchLine3[0];
    float *ptrInsert1 = &insertionLine1[0];
    float *ptrInsert2 = &insertionLine2[0];
    float *ptrInsert3 = &insertionLine3[0];
    float *ptrDelete1 = &deletionLine1[0];
    float *ptrDelete2 = &deletionLine2[0];
    float *ptrDelete3 = &deletionLine3[0];

    __shared__ float lineBuffer[PADDED_MAX_READ_LENGTH1];
    
	for (int i = tx; i < PADDED_MAX_READ_LENGTH1; i+=blockDim.x){
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

	float deletionMatrixLeftUp;
	float deletionMatrixLeft;

	float matchMatrixLeftUp;
	float matchMatrixUp;
	float matchMatrixLeft;
		
	float insertionMatrixUp;
	float insertionMatrixLeftUp;
	float insertionMatrixLeft;

    uint8 x;
//    uint8 qual;
//    float cache_tmp;
    float c1, c2;
    float transition_local[6];

    for (int layerIndex = 0; layerIndex < readLength; layerIndex++){
	//for (int layerIndex = 0; layerIndex < mBound-1; layerIndex++){
		if (layerIndex < minLength) {
			nBound = layerIndex + 1;
		} else {
			if (layerIndex >= maxLength) {
				nBound = -layerIndex + haplotypeLengthCur + readLength - 1;
			} else {
				nBound = minLength;
			}
		}
		//intra_layer_loop: for (int layerLocus = 0; layerLocus < nBound; layerLocus++){
		for (int layerLocus = tx; layerLocus < nBound; layerLocus+=blockDim.x)
		{
			//if (layerIndex < readLength) {
				indI = layerIndex + 1 + 1 - 0 - layerLocus - 1;
				indJ = layerLocus + 1;
			//} else {
			//	indI = readLength - layerLocus;
			//	indJ = layerIndex + 1 - readLength + layerLocus + 1;
			//}	

            matchMatrixLeft = ptrMatch2[indJ-1];
            matchMatrixLeftUp = ptrMatch3[indJ-1];
            matchMatrixUp = ptrMatch2[indJ];

            insertionMatrixLeft = ptrInsert2[indJ-1];
            insertionMatrixLeftUp = ptrInsert3[indJ-1];
            insertionMatrixUp = ptrInsert2[indJ];

            deletionMatrixLeft = ptrDelete2[indJ-1];
            deletionMatrixLeftUp = (indI==1)? initialValue: ptrDelete3[indJ-1];


			x = readBases[indI-1];
			uint8 qual = readBaseQualities[indI-1];
			float cache_tmp = qualToErrorProbCache_cuda[(int)qual & 0xFF];
			c1 = 1 - cache_tmp;
			c2 = cache_tmp / TRISTATE_CORRECTION;

			uint8 y = haplotypeBases[indJ-1];
			float prior = (x == y || x == (uint8)'N' || y == (uint8)'N' ? c1 : c2);

            transition_local[MATCH_TO_MATCH] = transition[(indI)*TRANS_PROB_ARRAY_LENGTH+MATCH_TO_MATCH];
            transition_local[MATCH_TO_INSERTION] = transition[(indI)*TRANS_PROB_ARRAY_LENGTH+MATCH_TO_INSERTION];
            transition_local[MATCH_TO_DELETION] = transition[(indI)*TRANS_PROB_ARRAY_LENGTH+MATCH_TO_DELETION];
            transition_local[INDEL_TO_MATCH] = transition[(indI)*TRANS_PROB_ARRAY_LENGTH+INDEL_TO_MATCH];
            transition_local[INSERTION_TO_INSERTION] = 1 - transition_local[INDEL_TO_MATCH];
            transition_local[DELETION_TO_DELETION] = transition_local[INSERTION_TO_INSERTION];

            matchResTmp1 = prior*(matchMatrixLeftUp*transition_local[MATCH_TO_MATCH]
                    +insertionMatrixLeftUp*transition_local[INDEL_TO_MATCH]
                    +deletionMatrixLeftUp*transition_local[INDEL_TO_MATCH]);
            insertionResTmp1 = matchMatrixUp*transition_local[MATCH_TO_INSERTION]
                    +insertionMatrixUp*transition_local[INSERTION_TO_INSERTION];
	    	deletionResTmp1 = matchMatrixLeft*transition_local[MATCH_TO_DELETION] 
                    +deletionMatrixLeft*transition_local[DELETION_TO_DELETION];

            ptrMatch1[indJ] = matchResTmp1;
            ptrInsert1[indJ] = insertionResTmp1;
            ptrDelete1[indJ] = deletionResTmp1;

            if (tx == 0){
                if (layerIndex >= readLength && layerLocus == 0){
                    likelihoodTmp += matchMatrixLeft + insertionMatrixLeft;   
                }
            }
		}
		__syncthreads();

        float *ptrTmp;
        ptrTmp = ptrMatch3;
        ptrMatch3 = ptrMatch2;
        ptrMatch2 = ptrMatch1;
        ptrMatch1 = ptrTmp;

        ptrTmp = ptrInsert3;
        ptrInsert3 = ptrInsert2;
        ptrInsert2 = ptrInsert1;
        ptrInsert1 = ptrTmp;

        ptrTmp = ptrDelete3;
        ptrDelete3 = ptrDelete2;
        ptrDelete2 = ptrDelete1;
        ptrDelete1 = ptrTmp;

	}

    for (int i = tx+1; i < PADDED_MAX_READ_LENGTH1; i+=blockDim.x){
        lineBuffer[i] = ptrMatch2[i];
    }
    __syncthreads();
    for (int i = tx; i < MAX_READ_LENGTH1; i+=blockDim.x){
        ptrMatch2[i] = lineBuffer[i+1];
    }
    __syncthreads();
    for (int i = tx+1; i < PADDED_MAX_READ_LENGTH1; i+=blockDim.x){
        lineBuffer[i] = ptrInsert2[i];
    }
    __syncthreads();
    for (int i = tx; i < MAX_READ_LENGTH1; i+=blockDim.x){
        ptrInsert2[i] = lineBuffer[i+1];
    }
    __syncthreads();
    for (int i = tx+1; i < PADDED_MAX_READ_LENGTH1; i+=blockDim.x){
        lineBuffer[i] = ptrDelete2[i];
    }
    __syncthreads();
    for (int i = tx; i < MAX_READ_LENGTH1; i+=blockDim.x){
        ptrDelete2[i] = lineBuffer[i+1];
    }
    __syncthreads();
        
    ptrMatch1[MAX_READ_LENGTH1] = 0.0;
    ptrMatch2[MAX_READ_LENGTH1] = 0.0;
    ptrMatch3[MAX_READ_LENGTH1] = 0.0;
    ptrInsert1[MAX_READ_LENGTH1] = 0.0;
    ptrInsert2[MAX_READ_LENGTH1] = 0.0;
    ptrInsert3[MAX_READ_LENGTH1] = 0.0;
    ptrDelete1[MAX_READ_LENGTH1] = 0.0;
    ptrDelete2[MAX_READ_LENGTH1] = 0.0;
    ptrDelete3[MAX_READ_LENGTH1] = 0.0;
    __syncthreads();

	//for (int layerIndex = 0; layerIndex < readLength; layerIndex++){
	for (int layerIndex = readLength; layerIndex < mBound-1; layerIndex++){
		if (layerIndex < minLength) {
			nBound = layerIndex + 1;
		} else {
			if (layerIndex >= maxLength) {
				nBound = -layerIndex + haplotypeLengthCur + readLength - 1;
			} else {
				nBound = minLength;
			}
		}
		//intra_layer_loop: for (int layerLocus = 0; layerLocus < nBound; layerLocus++){
		for (int layerLocus = tx; layerLocus < nBound; layerLocus+=blockDim.x)
		{
			//if (layerIndex < readLength) {
			//	indI = layerIndex + 1 + 1 - 0 - layerLocus - 1;
			//	indJ = layerLocus + 1;
			//} else {
				//indI = readLength - layerLocus;
                indI = layerLocus;
				indJ = layerIndex + 1 - readLength + layerLocus + 1;
			//}	

            matchMatrixLeft = ptrMatch2[indI];
            matchMatrixLeftUp = ptrMatch3[indI+1];
            matchMatrixUp = ptrMatch2[indI+1];

            insertionMatrixLeft = ptrInsert2[indI];
            insertionMatrixLeftUp = ptrInsert3[indI+1];
            insertionMatrixUp = ptrInsert2[indI+1];

            deletionMatrixLeft = ptrDelete2[indI];
            deletionMatrixLeftUp = (indI==readLength-1)? initialValue: ptrDelete3[indI+1];


			uint8 y = haplotypeBases[indJ-1];
			float prior = (x == y || x == (uint8)'N' || y == (uint8)'N' ? c1 : c2);


            matchResTmp1 = prior*(matchMatrixLeftUp*transition_local[MATCH_TO_MATCH]
                    +insertionMatrixLeftUp*transition_local[INDEL_TO_MATCH]
                    +deletionMatrixLeftUp*transition_local[INDEL_TO_MATCH]);
            insertionResTmp1 = matchMatrixUp*transition_local[MATCH_TO_INSERTION]
                    +insertionMatrixUp*transition_local[INSERTION_TO_INSERTION];
	    	deletionResTmp1 = matchMatrixLeft*transition_local[MATCH_TO_DELETION] 
                    +deletionMatrixLeft*transition_local[DELETION_TO_DELETION];

            ptrMatch1[indI] = matchResTmp1;
            ptrInsert1[indI] = insertionResTmp1;
            ptrDelete1[indI] = deletionResTmp1;


            if (tx == 0){
                if (layerIndex >= readLength && layerLocus == 0){
                    likelihoodTmp += matchMatrixLeft + insertionMatrixLeft;   
                }
            }
		}
		__syncthreads();
		

        float *ptrTmp;
        ptrTmp = ptrMatch3;
        ptrMatch3 = ptrMatch2;
        ptrMatch2 = ptrMatch1;
        ptrMatch1 = ptrTmp;

        ptrTmp = ptrInsert3;
        ptrInsert3 = ptrInsert2;
        ptrInsert2 = ptrInsert1;
        ptrInsert1 = ptrTmp;

        ptrTmp = ptrDelete3;
        ptrDelete3 = ptrDelete2;
        ptrDelete2 = ptrDelete1;
        ptrDelete1 = ptrTmp;
	}

    if (tx == 0){
        likelihoodTmp += ptrMatch2[0] + ptrInsert2[0];
    }
    
    *likelihood = likelihoodTmp;
}

__device__ void wrapper2(			
			uint8 *haplotypeBases,
            uint8 *readBases,
			uint8 *readBaseQualities,
            int haplotypeLength,
            int readLength,
            float *transition,
			//float qualToErrorProbCache[MAX_QUAL+1],
            float *likelihood
)
{
 	float MAX = powf(2, 120); // float
	float initialValue;
	initialValue = MAX / haplotypeLength;

	int nBound = 0;
	int haplotypeLengthCur = 0;
	haplotypeLengthCur = haplotypeLength;

	int minLength = (haplotypeLengthCur < readLength)? haplotypeLengthCur : readLength;
	int maxLength = (haplotypeLengthCur > readLength)? haplotypeLengthCur : readLength;

	int mBound = haplotypeLengthCur + readLength;
	
	int indI = 0;
	int indJ = 0;
	
	float deletionResTmp1;
	float matchResTmp1;
	float insertionResTmp1;

//	int round_flag = 0;

	int tx = threadIdx.x;
    float likelihoodTmp = 0.0;

	__shared__ float matchLine1[PADDED_MAX_READ_LENGTH2];
	__shared__ float matchLine2[PADDED_MAX_READ_LENGTH2];
	__shared__ float matchLine3[PADDED_MAX_READ_LENGTH2];

	__shared__ float insertionLine1[PADDED_MAX_READ_LENGTH2];
	__shared__ float insertionLine2[PADDED_MAX_READ_LENGTH2];
	__shared__ float insertionLine3[PADDED_MAX_READ_LENGTH2];

	__shared__ float deletionLine1[PADDED_MAX_READ_LENGTH2];
	__shared__ float deletionLine2[PADDED_MAX_READ_LENGTH2];
	__shared__ float deletionLine3[PADDED_MAX_READ_LENGTH2];

    float *ptrMatch1 = &matchLine1[0];
    float *ptrMatch2 = &matchLine2[0];
    float *ptrMatch3 = &matchLine3[0];
    float *ptrInsert1 = &insertionLine1[0];
    float *ptrInsert2 = &insertionLine2[0];
    float *ptrInsert3 = &insertionLine3[0];
    float *ptrDelete1 = &deletionLine1[0];
    float *ptrDelete2 = &deletionLine2[0];
    float *ptrDelete3 = &deletionLine3[0];

    __shared__ float lineBuffer[PADDED_MAX_READ_LENGTH2];
    
	for (int i = tx; i < PADDED_MAX_READ_LENGTH2; i+=blockDim.x){
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

	float deletionMatrixLeftUp;
	float deletionMatrixLeft;

	float matchMatrixLeftUp;
	float matchMatrixUp;
	float matchMatrixLeft;
		
	float insertionMatrixUp;
	float insertionMatrixLeftUp;
	float insertionMatrixLeft;

    uint8 x;
//    uint8 qual;
//    float cache_tmp;
    float c1, c2;
    float transition_local[6];

    for (int layerIndex = 0; layerIndex < readLength; layerIndex++){
	//for (int layerIndex = 0; layerIndex < mBound-1; layerIndex++){
		if (layerIndex < minLength) {
			nBound = layerIndex + 1;
		} else {
			if (layerIndex >= maxLength) {
				nBound = -layerIndex + haplotypeLengthCur + readLength - 1;
			} else {
				nBound = minLength;
			}
		}
		//intra_layer_loop: for (int layerLocus = 0; layerLocus < nBound; layerLocus++){
		for (int layerLocus = tx; layerLocus < nBound; layerLocus+=blockDim.x)
		{
			//if (layerIndex < readLength) {
				indI = layerIndex + 1 + 1 - 0 - layerLocus - 1;
				indJ = layerLocus + 1;
			//} else {
			//	indI = readLength - layerLocus;
			//	indJ = layerIndex + 1 - readLength + layerLocus + 1;
			//}	

            matchMatrixLeft = ptrMatch2[indJ-1];
            matchMatrixLeftUp = ptrMatch3[indJ-1];
            matchMatrixUp = ptrMatch2[indJ];

            insertionMatrixLeft = ptrInsert2[indJ-1];
            insertionMatrixLeftUp = ptrInsert3[indJ-1];
            insertionMatrixUp = ptrInsert2[indJ];

            deletionMatrixLeft = ptrDelete2[indJ-1];
            deletionMatrixLeftUp = (indI==1)? initialValue: ptrDelete3[indJ-1];


			x = readBases[indI-1];
			uint8 qual = readBaseQualities[indI-1];
			float cache_tmp = qualToErrorProbCache_cuda[(int)qual & 0xFF];
			c1 = 1 - cache_tmp;
			c2 = cache_tmp / TRISTATE_CORRECTION;

			uint8 y = haplotypeBases[indJ-1];
			float prior = (x == y || x == (uint8)'N' || y == (uint8)'N' ? c1 : c2);

            transition_local[MATCH_TO_MATCH] = transition[(indI)*TRANS_PROB_ARRAY_LENGTH+MATCH_TO_MATCH];
            transition_local[MATCH_TO_INSERTION] = transition[(indI)*TRANS_PROB_ARRAY_LENGTH+MATCH_TO_INSERTION];
            transition_local[MATCH_TO_DELETION] = transition[(indI)*TRANS_PROB_ARRAY_LENGTH+MATCH_TO_DELETION];
            transition_local[INDEL_TO_MATCH] = transition[(indI)*TRANS_PROB_ARRAY_LENGTH+INDEL_TO_MATCH];
            transition_local[INSERTION_TO_INSERTION] = 1 - transition_local[INDEL_TO_MATCH];
            transition_local[DELETION_TO_DELETION] = transition_local[INSERTION_TO_INSERTION];

            matchResTmp1 = prior*(matchMatrixLeftUp*transition_local[MATCH_TO_MATCH]
                    +insertionMatrixLeftUp*transition_local[INDEL_TO_MATCH]
                    +deletionMatrixLeftUp*transition_local[INDEL_TO_MATCH]);
            insertionResTmp1 = matchMatrixUp*transition_local[MATCH_TO_INSERTION]
                    +insertionMatrixUp*transition_local[INSERTION_TO_INSERTION];
	    	deletionResTmp1 = matchMatrixLeft*transition_local[MATCH_TO_DELETION] 
                    +deletionMatrixLeft*transition_local[DELETION_TO_DELETION];

            ptrMatch1[indJ] = matchResTmp1;
            ptrInsert1[indJ] = insertionResTmp1;
            ptrDelete1[indJ] = deletionResTmp1;

            if (tx == 0){
                if (layerIndex >= readLength && layerLocus == 0){
                    likelihoodTmp += matchMatrixLeft + insertionMatrixLeft;   
                }
            }
		}
		__syncthreads();

        float *ptrTmp;
        ptrTmp = ptrMatch3;
        ptrMatch3 = ptrMatch2;
        ptrMatch2 = ptrMatch1;
        ptrMatch1 = ptrTmp;

        ptrTmp = ptrInsert3;
        ptrInsert3 = ptrInsert2;
        ptrInsert2 = ptrInsert1;
        ptrInsert1 = ptrTmp;

        ptrTmp = ptrDelete3;
        ptrDelete3 = ptrDelete2;
        ptrDelete2 = ptrDelete1;
        ptrDelete1 = ptrTmp;

	}

    for (int i = tx+1; i < PADDED_MAX_READ_LENGTH2; i+=blockDim.x){
        lineBuffer[i] = ptrMatch2[i];
    }
    __syncthreads();
    for (int i = tx; i < MAX_READ_LENGTH2; i+=blockDim.x){
        ptrMatch2[i] = lineBuffer[i+1];
    }
    __syncthreads();
    for (int i = tx+1; i < PADDED_MAX_READ_LENGTH2; i+=blockDim.x){
        lineBuffer[i] = ptrInsert2[i];
    }
    __syncthreads();
    for (int i = tx; i < MAX_READ_LENGTH2; i+=blockDim.x){
        ptrInsert2[i] = lineBuffer[i+1];
    }
    __syncthreads();
    for (int i = tx+1; i < PADDED_MAX_READ_LENGTH2; i+=blockDim.x){
        lineBuffer[i] = ptrDelete2[i];
    }
    __syncthreads();
    for (int i = tx; i < MAX_READ_LENGTH2; i+=blockDim.x){
        ptrDelete2[i] = lineBuffer[i+1];
    }
    __syncthreads();
        
    ptrMatch1[MAX_READ_LENGTH2] = 0.0;
    ptrMatch2[MAX_READ_LENGTH2] = 0.0;
    ptrMatch3[MAX_READ_LENGTH2] = 0.0;
    ptrInsert1[MAX_READ_LENGTH2] = 0.0;
    ptrInsert2[MAX_READ_LENGTH2] = 0.0;
    ptrInsert3[MAX_READ_LENGTH2] = 0.0;
    ptrDelete1[MAX_READ_LENGTH2] = 0.0;
    ptrDelete2[MAX_READ_LENGTH2] = 0.0;
    ptrDelete3[MAX_READ_LENGTH2] = 0.0;
    __syncthreads();

	//for (int layerIndex = 0; layerIndex < readLength; layerIndex++){
	for (int layerIndex = readLength; layerIndex < mBound-1; layerIndex++){
		if (layerIndex < minLength) {
			nBound = layerIndex + 1;
		} else {
			if (layerIndex >= maxLength) {
				nBound = -layerIndex + haplotypeLengthCur + readLength - 1;
			} else {
				nBound = minLength;
			}
		}
		//intra_layer_loop: for (int layerLocus = 0; layerLocus < nBound; layerLocus++){
		for (int layerLocus = tx; layerLocus < nBound; layerLocus+=blockDim.x)
		{
			//if (layerIndex < readLength) {
			//	indI = layerIndex + 1 + 1 - 0 - layerLocus - 1;
			//	indJ = layerLocus + 1;
			//} else {
				//indI = readLength - layerLocus;
                indI = layerLocus;
				indJ = layerIndex + 1 - readLength + layerLocus + 1;
			//}	

            matchMatrixLeft = ptrMatch2[indI];
            matchMatrixLeftUp = ptrMatch3[indI+1];
            matchMatrixUp = ptrMatch2[indI+1];

            insertionMatrixLeft = ptrInsert2[indI];
            insertionMatrixLeftUp = ptrInsert3[indI+1];
            insertionMatrixUp = ptrInsert2[indI+1];

            deletionMatrixLeft = ptrDelete2[indI];
            deletionMatrixLeftUp = (indI==readLength-1)? initialValue: ptrDelete3[indI+1];


			uint8 y = haplotypeBases[indJ-1];
			float prior = (x == y || x == (uint8)'N' || y == (uint8)'N' ? c1 : c2);


            matchResTmp1 = prior*(matchMatrixLeftUp*transition_local[MATCH_TO_MATCH]
                    +insertionMatrixLeftUp*transition_local[INDEL_TO_MATCH]
                    +deletionMatrixLeftUp*transition_local[INDEL_TO_MATCH]);
            insertionResTmp1 = matchMatrixUp*transition_local[MATCH_TO_INSERTION]
                    +insertionMatrixUp*transition_local[INSERTION_TO_INSERTION];
	    	deletionResTmp1 = matchMatrixLeft*transition_local[MATCH_TO_DELETION] 
                    +deletionMatrixLeft*transition_local[DELETION_TO_DELETION];

            ptrMatch1[indI] = matchResTmp1;
            ptrInsert1[indI] = insertionResTmp1;
            ptrDelete1[indI] = deletionResTmp1;


            if (tx == 0){
                if (layerIndex >= readLength && layerLocus == 0){
                    likelihoodTmp += matchMatrixLeft + insertionMatrixLeft;   
                }
            }
		}
		__syncthreads();
		

        float *ptrTmp;
        ptrTmp = ptrMatch3;
        ptrMatch3 = ptrMatch2;
        ptrMatch2 = ptrMatch1;
        ptrMatch1 = ptrTmp;

        ptrTmp = ptrInsert3;
        ptrInsert3 = ptrInsert2;
        ptrInsert2 = ptrInsert1;
        ptrInsert1 = ptrTmp;

        ptrTmp = ptrDelete3;
        ptrDelete3 = ptrDelete2;
        ptrDelete2 = ptrDelete1;
        ptrDelete1 = ptrTmp;
	}

    if (tx == 0){
        likelihoodTmp += ptrMatch2[0] + ptrInsert2[0];
    }
    
    *likelihood = likelihoodTmp;
   
}

__device__ void wrapper3(			
			uint8 *haplotypeBases,
            uint8 *readBases,
			uint8 *readBaseQualities,
            int haplotypeLength,
            int readLength,
            float *transition,
			//float qualToErrorProbCache[MAX_QUAL+1],
            float *likelihood
)
{
  	float MAX = powf(2, 120); // float
	float initialValue;
	initialValue = MAX / haplotypeLength;

	int nBound = 0;
	int haplotypeLengthCur = 0;
	haplotypeLengthCur = haplotypeLength;

	int minLength = (haplotypeLengthCur < readLength)? haplotypeLengthCur : readLength;
	int maxLength = (haplotypeLengthCur > readLength)? haplotypeLengthCur : readLength;

	int mBound = haplotypeLengthCur + readLength;
	
	int indI = 0;
	int indJ = 0;
	
	float deletionResTmp1;
	float matchResTmp1;
	float insertionResTmp1;

//	int round_flag = 0;

	int tx = threadIdx.x;
    float likelihoodTmp = 0.0;

	__shared__ float matchLine1[PADDED_MAX_READ_LENGTH3];
	__shared__ float matchLine2[PADDED_MAX_READ_LENGTH3];
	__shared__ float matchLine3[PADDED_MAX_READ_LENGTH3];

	__shared__ float insertionLine1[PADDED_MAX_READ_LENGTH3];
	__shared__ float insertionLine2[PADDED_MAX_READ_LENGTH3];
	__shared__ float insertionLine3[PADDED_MAX_READ_LENGTH3];

	__shared__ float deletionLine1[PADDED_MAX_READ_LENGTH3];
	__shared__ float deletionLine2[PADDED_MAX_READ_LENGTH3];
	__shared__ float deletionLine3[PADDED_MAX_READ_LENGTH3];

    float *ptrMatch1 = &matchLine1[0];
    float *ptrMatch2 = &matchLine2[0];
    float *ptrMatch3 = &matchLine3[0];
    float *ptrInsert1 = &insertionLine1[0];
    float *ptrInsert2 = &insertionLine2[0];
    float *ptrInsert3 = &insertionLine3[0];
    float *ptrDelete1 = &deletionLine1[0];
    float *ptrDelete2 = &deletionLine2[0];
    float *ptrDelete3 = &deletionLine3[0];

    __shared__ float lineBuffer[PADDED_MAX_READ_LENGTH3];
    
	for (int i = tx; i < PADDED_MAX_READ_LENGTH3; i+=blockDim.x){
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

	float deletionMatrixLeftUp;
	float deletionMatrixLeft;

	float matchMatrixLeftUp;
	float matchMatrixUp;
	float matchMatrixLeft;
		
	float insertionMatrixUp;
	float insertionMatrixLeftUp;
	float insertionMatrixLeft;

    uint8 x;
//    uint8 qual;
//    float cache_tmp;
    float c1, c2;
    float transition_local[6];

    for (int layerIndex = 0; layerIndex < readLength; layerIndex++){
	//for (int layerIndex = 0; layerIndex < mBound-1; layerIndex++){
		if (layerIndex < minLength) {
			nBound = layerIndex + 1;
		} else {
			if (layerIndex >= maxLength) {
				nBound = -layerIndex + haplotypeLengthCur + readLength - 1;
			} else {
				nBound = minLength;
			}
		}
		//intra_layer_loop: for (int layerLocus = 0; layerLocus < nBound; layerLocus++){
		for (int layerLocus = tx; layerLocus < nBound; layerLocus+=blockDim.x)
		{
			//if (layerIndex < readLength) {
				indI = layerIndex + 1 + 1 - 0 - layerLocus - 1;
				indJ = layerLocus + 1;
			//} else {
			//	indI = readLength - layerLocus;
			//	indJ = layerIndex + 1 - readLength + layerLocus + 1;
			//}	

            matchMatrixLeft = ptrMatch2[indJ-1];
            matchMatrixLeftUp = ptrMatch3[indJ-1];
            matchMatrixUp = ptrMatch2[indJ];

            insertionMatrixLeft = ptrInsert2[indJ-1];
            insertionMatrixLeftUp = ptrInsert3[indJ-1];
            insertionMatrixUp = ptrInsert2[indJ];

            deletionMatrixLeft = ptrDelete2[indJ-1];
            deletionMatrixLeftUp = (indI==1)? initialValue: ptrDelete3[indJ-1];


			x = readBases[indI-1];
			uint8 qual = readBaseQualities[indI-1];
			float cache_tmp = qualToErrorProbCache_cuda[(int)qual & 0xFF];
			c1 = 1 - cache_tmp;
			c2 = cache_tmp / TRISTATE_CORRECTION;

			uint8 y = haplotypeBases[indJ-1];
			float prior = (x == y || x == (uint8)'N' || y == (uint8)'N' ? c1 : c2);

            transition_local[MATCH_TO_MATCH] = transition[(indI)*TRANS_PROB_ARRAY_LENGTH+MATCH_TO_MATCH];
            transition_local[MATCH_TO_INSERTION] = transition[(indI)*TRANS_PROB_ARRAY_LENGTH+MATCH_TO_INSERTION];
            transition_local[MATCH_TO_DELETION] = transition[(indI)*TRANS_PROB_ARRAY_LENGTH+MATCH_TO_DELETION];
            transition_local[INDEL_TO_MATCH] = transition[(indI)*TRANS_PROB_ARRAY_LENGTH+INDEL_TO_MATCH];
            transition_local[INSERTION_TO_INSERTION] = 1 - transition_local[INDEL_TO_MATCH];
            transition_local[DELETION_TO_DELETION] = transition_local[INSERTION_TO_INSERTION];

            matchResTmp1 = prior*(matchMatrixLeftUp*transition_local[MATCH_TO_MATCH]
                    +insertionMatrixLeftUp*transition_local[INDEL_TO_MATCH]
                    +deletionMatrixLeftUp*transition_local[INDEL_TO_MATCH]);
            insertionResTmp1 = matchMatrixUp*transition_local[MATCH_TO_INSERTION]
                    +insertionMatrixUp*transition_local[INSERTION_TO_INSERTION];
	    	deletionResTmp1 = matchMatrixLeft*transition_local[MATCH_TO_DELETION] 
                    +deletionMatrixLeft*transition_local[DELETION_TO_DELETION];

            ptrMatch1[indJ] = matchResTmp1;
            ptrInsert1[indJ] = insertionResTmp1;
            ptrDelete1[indJ] = deletionResTmp1;

            if (tx == 0){
                if (layerIndex >= readLength && layerLocus == 0){
                    likelihoodTmp += matchMatrixLeft + insertionMatrixLeft;   
                }
            }
		}
		__syncthreads();

        float *ptrTmp;
        ptrTmp = ptrMatch3;
        ptrMatch3 = ptrMatch2;
        ptrMatch2 = ptrMatch1;
        ptrMatch1 = ptrTmp;

        ptrTmp = ptrInsert3;
        ptrInsert3 = ptrInsert2;
        ptrInsert2 = ptrInsert1;
        ptrInsert1 = ptrTmp;

        ptrTmp = ptrDelete3;
        ptrDelete3 = ptrDelete2;
        ptrDelete2 = ptrDelete1;
        ptrDelete1 = ptrTmp;

	}

    for (int i = tx+1; i < PADDED_MAX_READ_LENGTH3; i+=blockDim.x){
        lineBuffer[i] = ptrMatch2[i];
    }
    __syncthreads();
    for (int i = tx; i < MAX_READ_LENGTH3; i+=blockDim.x){
        ptrMatch2[i] = lineBuffer[i+1];
    }
    __syncthreads();
    for (int i = tx+1; i < PADDED_MAX_READ_LENGTH3; i+=blockDim.x){
        lineBuffer[i] = ptrInsert2[i];
    }
    __syncthreads();
    for (int i = tx; i < MAX_READ_LENGTH3; i+=blockDim.x){
        ptrInsert2[i] = lineBuffer[i+1];
    }
    __syncthreads();
    for (int i = tx+1; i < PADDED_MAX_READ_LENGTH3; i+=blockDim.x){
        lineBuffer[i] = ptrDelete2[i];
    }
    __syncthreads();
    for (int i = tx; i < MAX_READ_LENGTH3; i+=blockDim.x){
        ptrDelete2[i] = lineBuffer[i+1];
    }
    __syncthreads();
        
    ptrMatch1[MAX_READ_LENGTH3] = 0.0;
    ptrMatch2[MAX_READ_LENGTH3] = 0.0;
    ptrMatch3[MAX_READ_LENGTH3] = 0.0;
    ptrInsert1[MAX_READ_LENGTH3] = 0.0;
    ptrInsert2[MAX_READ_LENGTH3] = 0.0;
    ptrInsert3[MAX_READ_LENGTH3] = 0.0;
    ptrDelete1[MAX_READ_LENGTH3] = 0.0;
    ptrDelete2[MAX_READ_LENGTH3] = 0.0;
    ptrDelete3[MAX_READ_LENGTH3] = 0.0;
    __syncthreads();

	//for (int layerIndex = 0; layerIndex < readLength; layerIndex++){
	for (int layerIndex = readLength; layerIndex < mBound-1; layerIndex++){
		if (layerIndex < minLength) {
			nBound = layerIndex + 1;
		} else {
			if (layerIndex >= maxLength) {
				nBound = -layerIndex + haplotypeLengthCur + readLength - 1;
			} else {
				nBound = minLength;
			}
		}
		//intra_layer_loop: for (int layerLocus = 0; layerLocus < nBound; layerLocus++){
		for (int layerLocus = tx; layerLocus < nBound; layerLocus+=blockDim.x)
		{
			//if (layerIndex < readLength) {
			//	indI = layerIndex + 1 + 1 - 0 - layerLocus - 1;
			//	indJ = layerLocus + 1;
			//} else {
				//indI = readLength - layerLocus;
                indI = layerLocus;
				indJ = layerIndex + 1 - readLength + layerLocus + 1;
			//}	

            matchMatrixLeft = ptrMatch2[indI];
            matchMatrixLeftUp = ptrMatch3[indI+1];
            matchMatrixUp = ptrMatch2[indI+1];

            insertionMatrixLeft = ptrInsert2[indI];
            insertionMatrixLeftUp = ptrInsert3[indI+1];
            insertionMatrixUp = ptrInsert2[indI+1];

            deletionMatrixLeft = ptrDelete2[indI];
            deletionMatrixLeftUp = (indI==readLength-1)? initialValue: ptrDelete3[indI+1];


			uint8 y = haplotypeBases[indJ-1];
			float prior = (x == y || x == (uint8)'N' || y == (uint8)'N' ? c1 : c2);


            matchResTmp1 = prior*(matchMatrixLeftUp*transition_local[MATCH_TO_MATCH]
                    +insertionMatrixLeftUp*transition_local[INDEL_TO_MATCH]
                    +deletionMatrixLeftUp*transition_local[INDEL_TO_MATCH]);
            insertionResTmp1 = matchMatrixUp*transition_local[MATCH_TO_INSERTION]
                    +insertionMatrixUp*transition_local[INSERTION_TO_INSERTION];
	    	deletionResTmp1 = matchMatrixLeft*transition_local[MATCH_TO_DELETION] 
                    +deletionMatrixLeft*transition_local[DELETION_TO_DELETION];

            ptrMatch1[indI] = matchResTmp1;
            ptrInsert1[indI] = insertionResTmp1;
            ptrDelete1[indI] = deletionResTmp1;


            if (tx == 0){
                if (layerIndex >= readLength && layerLocus == 0){
                    likelihoodTmp += matchMatrixLeft + insertionMatrixLeft;   
                }
            }
		}
		__syncthreads();
		

        float *ptrTmp;
        ptrTmp = ptrMatch3;
        ptrMatch3 = ptrMatch2;
        ptrMatch2 = ptrMatch1;
        ptrMatch1 = ptrTmp;

        ptrTmp = ptrInsert3;
        ptrInsert3 = ptrInsert2;
        ptrInsert2 = ptrInsert1;
        ptrInsert1 = ptrTmp;

        ptrTmp = ptrDelete3;
        ptrDelete3 = ptrDelete2;
        ptrDelete2 = ptrDelete1;
        ptrDelete1 = ptrTmp;
	}

    if (tx == 0){
        likelihoodTmp += ptrMatch2[0] + ptrInsert2[0];
    }
    
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
		//float qualToErrorProbCache1[MAX_QUAL+1],
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
    wrapper1(
        haplotypeCur1,readCur1,
		readBaseQualitiesCur1,
		haplotypeLengthCur1,readLengthCur1,
		transition1,
		//qualToErrorProbCache1,
        &likelihood1
    );

    if (tx == 0){
	float INITIAL_CONDITION = powf(2, 120); // float
	float INITIAL_CONDITION_LOG10 = log10f(INITIAL_CONDITION);
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
		//float qualToErrorProbCache1[MAX_QUAL+1],
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
    wrapper2(
        haplotypeCur1,readCur1,
		readBaseQualitiesCur1,
		haplotypeLengthCur1,readLengthCur1,
		transition1,
		//qualToErrorProbCache1,
        &likelihood1
    );

    if (tx == 0){
	float INITIAL_CONDITION = powf(2, 120); // float
	float INITIAL_CONDITION_LOG10 = log10f(INITIAL_CONDITION);
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
		//float qualToErrorProbCache1[MAX_QUAL+1],
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
    wrapper3(
        haplotypeCur1,readCur1,
		readBaseQualitiesCur1,
		haplotypeLengthCur1,readLengthCur1,
		transition1,
		//qualToErrorProbCache1,
        &likelihood1
    );

    if (tx == 0){
	float INITIAL_CONDITION = powf(2, 120); // float
	float INITIAL_CONDITION_LOG10 = log10f(INITIAL_CONDITION);
		mLikelihoodArray[readIter*haplotypeNum+haplotypeIndex] = log10f(likelihood1) - INITIAL_CONDITION_LOG10;
	}
}

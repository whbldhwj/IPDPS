#include "pairHMM.h"

#define MAX_JACOBIAN_TOLERANCE 8.0
#define JACOBIAN_LOG_TABLE_STEP 0.0001
#define JACOBIAN_LOG_TABLE_INV_STEP (1.0 / JACOBIAN_LOG_TABLE_STEP)

float myLog10SumLog10_2op(float var1, float var2)
{
	if (var1 == 0.0)
		return var2;
	else if (var2 == 0.0)
		return var1;
	else {
		float res1 = powf(10, var1);
		float res2 = powf(10, var2);
		float value = log10f(res1 + res2);
		return value;
	}
}

int fastRound(float d){
    return (d > 0.0)? (int)(d+0.5): (int)(d-0.5);
}

void initializeJacobianLogTable(
    float jacobianLogTable[JACOBIAN_LOG_TABLE_SIZE]       
)
{
    for (int k = 0; k < JACOBIAN_LOG_TABLE_SIZE; k++){
        jacobianLogTable[k] = (float)(log10(1.0 + pow(10.0, -(double)k)*JACOBIAN_LOG_TABLE_STEP));
    }
}

float approximateLog10SumLog10(
        float small, float big, 
        float jacobianLogTable[JACOBIAN_LOG_TABLE_SIZE])
{
    if (small > big){
        float t = big;
        big = small;
        small = t;
    }
    if (isinf(small) == -1 || isinf(big) == -1)
        return big;

    float diff = big - small;
    if (diff >= MAX_JACOBIAN_TOLERANCE)
        return big;

    int ind = fastRound(diff * JACOBIAN_LOG_TABLE_INV_STEP);
    return big + jacobianLogTable[ind];
}

float matchToMatchProb(
        uint8 insQual, uint8 delQual, 
        float matchToMatch[(MAX_QUAL + 1) * (MAX_QUAL + 2) >> 1],
        float jacobianLogTable[JACOBIAN_LOG_TABLE_SIZE])
{
    int insQual_int = (insQual & 0xFF);
    int delQual_int = (delQual & 0xFF);
    int minQual;
    int maxQual;
    if (insQual_int <= delQual_int)
    {
        minQual = insQual;
        maxQual = delQual;
    } else
    {
        minQual = delQual;
        maxQual = insQual;
    }
	//float value = (MAX_QUAL < maxQual) ? (float)1 - (float)powf(10, myLog10SumLog10_2op(-0.1*minQual, -0.1*maxQual)) : matchToMatch[((maxQual * (maxQual + 1)) >> 1) + minQual];
	float value = (MAX_QUAL < maxQual) ? (float)1 - (float)powf(10, approximateLog10SumLog10(-0.1*minQual, -0.1*maxQual, jacobianLogTable)) : matchToMatch[((maxQual * (maxQual + 1)) >> 1) + minQual];

    return value;
}

float qualToProb(uint8 qual, float qualToErrorProbCache[MAX_QUAL+1])
{
	return 1.0 - qualToErrorProbCache[(int)qual & 0xFF];
}

float qualToErrorProb(uint8 qual, float qualToErrorProbCache[MAX_QUAL+1])
{
    return qualToErrorProbCache[(int)qual & 0xFF];
}

float qualToErrorProbDouble(float qual)
{
    return powf(10, qual/-10.0);
}

void qualToTransProbs(
						float dest1[PADDED_MAX_READ_LENGTH][TRANS_PROB_ARRAY_LENGTH],
                        uint8 insQual[MAX_READ_LENGTH],
                        uint8 delQual[MAX_READ_LENGTH],
                        uint8 gcp[MAX_READ_LENGTH],
						float qualToErrorProbCache[MAX_QUAL+1],
						float matchToMatch[(MAX_QUAL + 1) * (MAX_QUAL + 2) >> 1],
						int readLength,
                        float jacobianLogTable[JACOBIAN_LOG_TABLE_SIZE]
						   )
{    
    for (int i = 0; i < MAX_READ_LENGTH; i++)
    {
		if (i < readLength) {
			dest1[i+1][MATCH_TO_MATCH] = matchToMatchProb(insQual[i], delQual[i], matchToMatch, jacobianLogTable);
			dest1[i+1][MATCH_TO_INSERTION] = qualToErrorProb(insQual[i], qualToErrorProbCache);
			dest1[i+1][MATCH_TO_DELETION] = qualToErrorProb(delQual[i], qualToErrorProbCache);
			dest1[i+1][INDEL_TO_MATCH] = qualToProb(gcp[i], qualToErrorProbCache);
			//float tmp = qualToErrorProb(gcp[i], qualToErrorProbCache);
			//dest1[i+1][INSERTION_TO_INSERTION] = tmp;
			//dest1[i+1][DELETION_TO_DELETION] = tmp;
		} else {
			break;
		}
    }
}

void initializeProbabilities(uint8 insertionGOP[MAX_READ_LENGTH],
                             uint8 deletionGOP[MAX_READ_LENGTH],
                             uint8 overallGCP[MAX_READ_LENGTH],
                             float transition1[PADDED_MAX_READ_LENGTH][TRANS_PROB_ARRAY_LENGTH],
							 float qualToErrorProbCache[MAX_QUAL+1],
							 float matchToMatch[(MAX_QUAL + 1) * (MAX_QUAL + 2) >> 1],
							 int readLength,
                             float jacobianLogTable[JACOBIAN_LOG_TABLE_SIZE]
)
{
	qualToTransProbs(
	transition1,
	insertionGOP, deletionGOP, overallGCP, qualToErrorProbCache, matchToMatch, readLength,
    jacobianLogTable
    );
}

void init_compute(
    float qualToErrorProbCache1[MAX_QUAL+1],
	float matchToMatch1[(MAX_QUAL + 1) * (MAX_QUAL + 2) >> 1]
)
{
	for (int i = 0; i <= MAX_QUAL; i++) {
		float tmp = qualToErrorProbDouble((float)i);
        qualToErrorProbCache1[i] = tmp;
	}

	float LN10 = logf(10);
	float INV_LN10 = 1.0 / LN10;

	int iInd = 0;
	int jInd = 0;
	for (int i = 0; i < (MAX_QUAL + 1) * (MAX_QUAL + 2) >> 1; i++){
		int iIndLocal = iInd;
		int jIndLocal = jInd;
		if (iInd == jInd){
			iInd = 0;
			jInd += 1;
		} else {
			iInd += 1;
		}
		float log10Sum = myLog10SumLog10_2op(-.1 * iIndLocal, -.1 * jIndLocal);
		float minVal = (1 < powf(10, log10Sum))? 1:powf(10,log10Sum);
		float log10Tmp = log1pf(-minVal) * INV_LN10;
				//min_cal((float)1, (float)(pow_cal(10, log10Sum))))) * INV_LN10;
		float tmp = powf(10, log10Tmp);
        matchToMatch1[i] = tmp;
	}
}


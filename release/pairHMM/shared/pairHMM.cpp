#include "pairHMM.h"

//#include "pairHMM_kernel.cu"
float cuda_kernel(
    uint8* read_in,
    uint8* hap_in,
    int* readLength_in,
    int* hapLength_in,
	int readNum,
	int hapNum,
    float* mLikelihoodArray_hw
);
					
int main(int argc, char** argv)
{
    int regionToRun = atoi(argv[1]);
    printf("Region to run: %d\n", regionToRun);

    FILE *fDataByte, *fOutByte;
    fDataByte = fopen("../data/inputByte.dat", "rb");
    if (fDataByte == NULL)
        printf("failed open inputByte!\n");
    fOutByte = fopen("../data/outputByte.dat", "rb");
    if (fOutByte == NULL)
        printf("failed open outputByte!\n");

	int err = 0;
	float thres = 0.001;
	char ch;
    int readNum = 0;
	int hapNum = 0;

    long int errTotal = 0;
//	long int cupsTotal = 0;
	float cupsTotal = 0;
	float timeTotal = 0;

    float cupsMax = 0;
	
    uint8 *haplotype = (uint8*)malloc(HAPLOTYPE_NUM * MAX_HAPLOTYPE_LENGTH * sizeof(uint8));
    uint8 *read = (uint8*)malloc(READ_NUM * MAX_READ_LENGTH * 5 * sizeof(uint8));
    int *haplotypeLength = (int*)malloc(HAPLOTYPE_NUM * sizeof(int));
    int *readLength = (int*)malloc(READ_NUM * sizeof(int));
    float *mLikelihoodArray_sw = (float*)malloc(READ_NUM * HAPLOTYPE_NUM * sizeof(float));
	float* mLikelihoodArray_hw = (float*)malloc(READ_NUM * HAPLOTYPE_NUM * sizeof(float));

    int regionNum = 0;

    int readOffset = 0;
    int hapOffset = 0;
    int readLenOffset = 0;
    int hapLenOffset = 0;
    int mLikelihoodArrayOffset = 0;

    for (int regionNum = 0; regionNum < regionToRun; regionNum++) {
        printf("Region #%d\n", regionNum);
        fread(&readNum, sizeof(int), 1, fDataByte);
        fread(&hapNum, sizeof(int), 1, fDataByte);

        for (int i = 0; i < readNum; i++){
            fread(&readLength[i+readLenOffset], sizeof(int), 1, fDataByte);
            fread(&read[i*MAX_READ_LENGTH*5+readOffset], sizeof(char), readLength[i+readLenOffset], fDataByte);
            fread(&read[i*MAX_READ_LENGTH*5+MAX_READ_LENGTH+readOffset], sizeof(char), readLength[i+readLenOffset], fDataByte);
            fread(&read[i*MAX_READ_LENGTH*5+MAX_READ_LENGTH*2+readOffset], sizeof(char), readLength[i+readLenOffset], fDataByte);
            fread(&read[i*MAX_READ_LENGTH*5+MAX_READ_LENGTH*3+readOffset], sizeof(char), readLength[i+readLenOffset], fDataByte);
            fread(&read[i*MAX_READ_LENGTH*5+MAX_READ_LENGTH*4+readOffset], sizeof(char), readLength[i+readLenOffset], fDataByte);
        }
		
        for (int i = 0; i < hapNum; i++){
            fread(&haplotypeLength[i+hapLenOffset], sizeof(int), 1, fDataByte);
            fread(&haplotype[i*MAX_HAPLOTYPE_LENGTH+hapOffset], sizeof(char), haplotypeLength[i+hapLenOffset], fDataByte);
        }

        int resCount = readNum * hapNum;
        fread(&resCount, sizeof(int), 1, fOutByte);
        //for (int i = 0; i < readNum * hapNum; i++){
            fread(&mLikelihoodArray_sw[mLikelihoodArrayOffset], sizeof(float), resCount, fOutByte);
        //}

        // cal cups
//		long int cups = 0;
		float cups = 0;
		for (int i = 0; i < readNum; i++){
			int readLen = readLength[i+readLenOffset];
			for (int j = 0; j < hapNum; j++){
				int hapLen = haplotypeLength[j+hapLenOffset];
				cups += readLen*hapLen;
			}
		}	

		cupsTotal += cups;
		
		float elapsed_time;	
		elapsed_time = cuda_kernel(
			read,
			haplotype,
			readLength,
			haplotypeLength,
			readNum,
			hapNum,
			mLikelihoodArray_hw);
		
		timeTotal += elapsed_time;
		
		for (int i = 0; i < readNum; i++) {
			for (int j = 0; j < hapNum; j++) {
				if (abs(mLikelihoodArray_sw[i * hapNum + j] - mLikelihoodArray_hw[i * hapNum + j]) > thres) {
					err++;
					printf("read: %d hap: %d - sw: %lf hw: %lf\n", i, j, mLikelihoodArray_sw[i * hapNum + j], mLikelihoodArray_hw[i * hapNum + j]);
				}
				else {
					//printf("read: %d hap: %d - sw: %lf hw: %lf\n", i, j, mLikelihoodArray_sw[i * hapNum + j], mLikelihoodArray_hw[i * hapNum + j]);
				}
			}
		}
	
        errTotal += err;
		if (err > 0)
		{
			printf("Error numbers = %d\n", err);
		}
		else
		{
			//printf("The test passed successfully.\n");
		}
		
		//printf("Elapsed Time: %f(ms)\n", elapsed_time);
		printf("Performance(MCUPs): %f\n", cups/elapsed_time/1e3);
        if (cups/elapsed_time/1e3 > cupsMax)
            cupsMax = cups/elapsed_time/1e3;

        //readOffset += readNum * MAX_READ_LENGTH * 5;
        //readLenOffset += readNum;
        //hapOffset += hapNum * MAX_HAPLOTYPE_LENGTH;
        //hapLenOffset += hapNum;
        //mLikelihoodArrayOffset += readNum * hapNum;

        //regionNum += 1;
        if (regionNum == regionToRun)
    		break;
    }

    if (errTotal == 0)
        printf("The test passed successfully.\n");
    printf("Elapsed Time(Total): %f(ms)\n", timeTotal);
    printf("Average Performance(MCUPs): %f\n", cupsTotal/timeTotal/1e3);
    printf("Max Performance(MCUPs): %f\n", cupsMax);

    free(haplotype);
    free(read);
    free(haplotypeLength);
    free(readLength);
    free(mLikelihoodArray_sw);
	free(mLikelihoodArray_hw);

    fclose(fDataByte);
    fclose(fOutByte);
	
    return 0;    
}

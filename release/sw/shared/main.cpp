#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <sys/time.h>
#include <ctime>
#include "params.h"

#define COMPARE
//#define PRINT

extern float sw_wrap(
    int refNum,
    char *ref, char *alt,
    short *refLen, short *altLen,
    short *strategy,
    short *state, short *segmentLen, short *offset, short *num
);

int main(int argc, char** argv)
{
    int regionToRun = atoi(argv[1]);
    printf("Region to run: %d\n", regionToRun);

    FILE *fAlt, *fRef;
    FILE *fAltLen, *fRefLen;
    FILE *fStrategy;

    fAlt = fopen("./data/alt.dat", "r");
    fAltLen = fopen("./data/altlen.dat", "r");
    fRef = fopen("./data/ref.dat", "r");
    fRefLen = fopen("./data/reflen.dat", "r");
    fStrategy = fopen("./data/strategy.dat", "r");

    FILE *fState, *fSegmentLen;
    FILE *fOffset, *fNum;

#ifdef PRINT
    fState = fopen("./data/state.dat", "w");
    fSegmentLen = fopen("./data/seglen.dat", "w");
    fOffset = fopen("./data/offset.dat", "w");
    fNum = fopen("./data/num.dat", "w");
#endif
#ifdef COMPARE    
    fState = fopen("./data/state.dat", "r");
    fSegmentLen = fopen("./data/seglen.dat", "r");
    fOffset = fopen("./data/offset.dat", "r");
    fNum = fopen("./data/num.dat", "r");
#endif    

    char *alt, *ref;
    short *altLen, *refLen;
    short *strategy;

    short *state;
    short *segmentLen;
    short *offset;
    short *num;
   
    alt = (char*)malloc(MAX_PAIR_NUM*MAX_ALT_LEN*sizeof(char));
    ref = (char*)malloc(MAX_PAIR_NUM*MAX_REF_LEN*sizeof(char));
    altLen = (short*)malloc(MAX_PAIR_NUM*sizeof(short));
    refLen = (short*)malloc(MAX_PAIR_NUM*sizeof(short));
    strategy = (short*)malloc(MAX_PAIR_NUM*sizeof(short));

    state = (short*)malloc(MAX_PAIR_NUM*MAX_STATE_NUM*sizeof(short));
    segmentLen = (short*)malloc(MAX_PAIR_NUM*MAX_STATE_NUM*sizeof(short));
    offset = (short*)malloc(MAX_PAIR_NUM*sizeof(short));
    num = (short*)malloc(MAX_PAIR_NUM*sizeof(short));

#ifdef COMPARE
    short *stateGolden;
    short *segmentLenGolden;
//    short *offsetGolden;
//    short *numGolden;

    stateGolden = (short*)malloc(MAX_PAIR_NUM*MAX_STATE_NUM*sizeof(short));
    segmentLenGolden = (short*)malloc(MAX_PAIR_NUM*MAX_STATE_NUM*sizeof(short));
//    offsetGolden = (short*)malloc(MAX_PAIR_NUM*sizeof(short));
//    numGolden = (short*)malloc(MAX_PAIR_NUM*sizeof(short));   
#endif    

//    int regionNum = 0;
    int refNum = 0;
    int altNum = 0;

//    long cupsTotal = 0;
    float cupsTotal = 0;
    float timeTotal = 0;
    struct timeval start, end;

    long errTotal = 0;
    float gcupsMax = 0;

    for (int regionIter = 0; regionIter < regionToRun; regionIter++)
    {
        printf("Region #%d\n", regionIter);
        fscanf(fAlt, "%d\n", &altNum);
        fscanf(fRef, "%d\n", &refNum);
        for (int altIter = 0; altIter < altNum; altIter++){
            fscanf(fAltLen, "%d", &altLen[altIter]);
        }
        for (int refIter = 0; refIter < refNum; refIter++){
            fscanf(fRefLen, "%d", &refLen[refIter]);
        }
        for (int altIter = 0; altIter < altNum; altIter++){
            int altLenCur = altLen[altIter];
            for (int altInd = 0; altInd < altLenCur; altInd++){
                char baseCur;
                fscanf(fAlt, "%c", &baseCur);
                alt[altIter*MAX_ALT_LEN+altInd] = baseCur;
            }
        }
        for (int refIter = 0; refIter < refNum; refIter++){
            int refLenCur = refLen[refIter];
            for (int refInd = 0; refInd < refLenCur; refInd++){
                char baseCur;
                fscanf(fRef, "%c", &baseCur);
                ref[refIter*MAX_REF_LEN+refInd] = baseCur;
            }
        }
        for (int pairIter = 0; pairIter < altNum; pairIter++){
            fscanf(fStrategy, "%d", &strategy[pairIter]);
        }
        float cupsCur = 0;
        for (int pairIter = 0; pairIter < altNum; pairIter++){
            cupsCur += altLen[pairIter]*refLen[pairIter];
        }
        cupsTotal += cupsCur;

//        gettimeofday(&start, NULL);
        float timeCur = sw_wrap(
                refNum,
                ref, alt,
                refLen, altLen,
                strategy,
                state, segmentLen, offset, num);
//        gettimeofday(&end, NULL);
//        long usec = end.tv_usec - start.tv_usec;
//        long sec = end.tv_sec - start.tv_sec;
//        float timeCur = sec*1000 + usec/1000;
        timeTotal += timeCur;
        
#ifdef PRINT        
        // print cigar
        for (int pairIter = 0; pairIter < refNum; pairIter++){
            short numCur = num[pairIter];
            short offsetCur = offset[pairIter];
            fprintf(fOffset, "%d\n", offsetCur);
            fprintf(fNum, "%d\n", numCur);
            for (int stateIter = 0; stateIter < numCur; stateIter++){
                fprintf(fState, "%d ", state[pairIter*MAX_STATE_NUM+stateIter]);
            }
//            fprintf(fState, "\n");
            for (int segIter = 0; segIter < numCur; segIter++){
                fprintf(fSegmentLen, "%d ", segmentLen[pairIter*MAX_STATE_NUM+segIter]);
            }
//            fprintf(fSegmentLen, "\n");
        }
#endif
        int errCur = 0;      
#ifdef COMPARE
        for (int pairIter = 0; pairIter < refNum; pairIter++){
            short numCur = num[pairIter];
            short offsetCur = offset[pairIter];

            short numGolden;
            short offsetGolden;            
            fscanf(fOffset, "%d\n", &offsetGolden);
            fscanf(fNum, "%d\n", &numGolden);

            bool numEqual, offsetEqual;
            if (offsetCur != offsetGolden)
                offsetEqual = false;
            else
                offsetEqual = true;
            if (numCur != numGolden)
                numEqual = false;
            else
                numEqual = true;

            bool stateEqual = true;
            bool segmentLenEqual = true;

            for (int stateIter = 0; stateIter < numCur; stateIter++){
                fscanf(fState, "%d", &stateGolden[pairIter*MAX_STATE_NUM+stateIter]);
//                fprintf(fState, "%d ", state[pairIter*MAX_STATE_NUM+stateIter]);
                if (stateGolden[pairIter*MAX_STATE_NUM+stateIter] != state[pairIter*MAX_STATE_NUM+stateIter])
                    stateEqual = false;
            }
//            fprintf(fState, "\n");
            for (int segIter = 0; segIter < numCur; segIter++){
                fscanf(fSegmentLen, "%d", &segmentLenGolden[pairIter*MAX_STATE_NUM+segIter]);
//                fprintf(fSegmentLen, "%d ", segmentLen[pairIter*MAX_STATE_NUM+segIter]);
                if (segmentLenGolden[pairIter*MAX_STATE_NUM+segIter] != segmentLen[pairIter*MAX_STATE_NUM+segIter])
                    segmentLenEqual = false;
            }
//            fprintf(fSegmentLen, "\n");
            if (offsetEqual == false || numEqual == false || stateEqual == false || segmentLenEqual == false)
                errCur += 1;
        }
        if (errCur == 0)
            printf("test passed!\n");
        else
            printf("test failed!\n");
        errTotal += errCur;
#endif        
        float gcupsCur = cupsCur / timeCur / 1e6;
        printf("gcups: %f\n", gcupsCur);
        if (gcupsCur > gcupsMax)
            gcupsMax = gcupsCur;
    }
#ifdef COMPARE
    if (errTotal == 0)
        printf("test passed!\n");
    else
        printf("test failed!\n");
#endif    
    float gcups = cupsTotal / timeTotal / 1e6;
    printf("average gcups: %f\n", gcups);
    printf("max gcups: %f\n", gcupsMax);
    printf("elapsed time(ms): %f\n", timeTotal);

    free(alt);
    free(ref);
    free(altLen);
    free(refLen);
    free(strategy);

    free(state);
    free(segmentLen);
    free(offset);
    free(num);
#ifdef COMPARE
    free(stateGolden);
    free(segmentLenGolden);
//    free(offsetGolden);
//    free(numGolden);   
#endif    

    fclose(fAlt);
    fclose(fAltLen);
    fclose(fRef);
    fclose(fRefLen);
    fclose(fStrategy);
    fclose(fState);
    fclose(fSegmentLen);
    fclose(fOffset);
    fclose(fNum);

    return 0;
}

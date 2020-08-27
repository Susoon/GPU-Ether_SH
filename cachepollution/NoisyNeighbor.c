#include <stdio.h>
#include <stdlib.h>
#include <time.h>

long timediff(clock_t t1, clock_t t2){
	return (t2 - t1)*1000/CLOCKS_PER_SEC;
}

int main(void){
	long **src1;
	long **src2;
	long **res;

	srand(time(NULL));

	src1 = (long**)malloc(sizeof(long*)*1140);
	src2 = (long**)malloc(sizeof(long*)*1140); 
	res = (long**)malloc(sizeof(long*)*1140); 

	for(int i = 0; i < 1140; i++){
		src1[i] = (long*)malloc(sizeof(long)*1140); 
		src2[i] = (long*)malloc(sizeof(long)*1140); 
		res[i] = (long*)malloc(sizeof(long)*1140); 
	}

	for(int i = 0; i < 1140; i++){
		for(int j = 0; j < 1140; j++){
			src1[i][j] = rand();
			src2[i][j] = rand();
			res[i][j] = rand();
		}
	}

	clock_t start = clock();

	for(int i = 0; i < 1140; i++)
		for(int j = 0; j < 1140; j++)
			for(int k = 0; k < 1140; k++)
				res[i][j] = src1[i][k] * src2[k][j];

	clock_t end = clock();
	
	printf("Time : %lums\n", timediff(start, end));

	return 0;
}

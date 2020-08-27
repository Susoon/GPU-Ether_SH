#include "master.h"
#include <stdio.h>
#include <time.h>

extern void gettQ(struct tunnelQ **tmp);

void master(void *data)
{
	srand(time(NULL));
#if 1
	struct tunnelQ *tQ = NULL;
	uint64_t start = monotonic_time();
	uint64_t end;
	int nb;

	while(tQ == NULL){
		gettQ(&tQ);
	}
	while(tQ->pktRXQ == NULL || tQ->pktTXQ == NULL);
	printf("GOT tQ!!!!\n");

	while(1){
		if(tQ->rhead >= PKT_BATCH_NUM - RX_NB){
			copy_to_gpu(tQ->pktRXQ, tQ->rhead);
			tQ->rhead = 0;
		}
#if 1 
		if(tQ->thead == 0 && (nb = copy_from_gpu(tQ->pktTXQ)) > 0){
			tQ->thead = nb;
		}
#endif
	}
#endif
}


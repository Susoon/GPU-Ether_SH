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

	//tQ를 받아올때까지 시도
	while(tQ == NULL){
		gettQ(&tQ);
	}
	//tQ의 field가 할당될 때까지 대기
	while(tQ->pktRXQ == NULL || tQ->pktTXQ == NULL);
	printf("GOT tQ!!!!\n");

	while(1){
		//rhead가 512 - 32개 이상이면
		if(tQ->rhead >= PKT_BATCH_NUM - RX_NB){
			//GPU로 전송
			copy_to_gpu(tQ->pktRXQ, tQ->rhead);
			tQ->rhead = 0;
		}
#if 1 
		//GPU로부터 패킷을 받아오면
		if(tQ->thead == 0 && (nb = copy_from_gpu(tQ->pktTXQ)) > 0){
			//thead값을 변경해 worker thread에게 패킷을 받음을 알림
			tQ->thead = nb;
		}
#endif
	}
#endif
}


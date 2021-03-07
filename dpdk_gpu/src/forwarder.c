#include "forwarder.h"

//extern struct tunnelQ *tQ;

void forwarder(void *data)
{
	struct tunnelQ *tQ = (struct tunnelQ *)data;
#if 1
	uint16_t nr, nt;
	uint8_t fc, wc;
	while(tQ == NULL) {sleep(1);}
	printf("TEST\n");
	unsigned char **pbuf = tQ->ptrBuf;

	while(1){
#if 1
		if(tQ->Wchapter != tQ->Fchapter){
			//printf("In IF\n");
			tQ->thead = tQ->rhead;
			swp_hdr_buf(tQ->ptrBuf, tQ->rhead);
		}
#endif
		sleep(1);
		nr = tQ->nb_rx; nt = tQ->nb_tx; wc = tQ->Wchapter;
		printf("nb_rx : %d, nb_tx : %d, Fchapter : %d, Wchapter : %d\n", nr, nt, tQ->Fchapter, wc);
		if(pbuf != NULL)
			printf("ptrBuf : %c%c\n", pbuf[0], pbuf[1]);
	}
#endif
}

#include "util.h"

#define GPU 1

//패킷 수가 카운트되어 저장된 변수들
double gpu_recv;
extern double cpu_recv;
extern double cpu_tran;

int unit = 0;
const char *u_str[] = {"", "K", "M", "G"};

uint64_t monotonic_time(){
	struct timespec timespec;
	clock_gettime(CLOCK_MONOTONIC, &timespec);
	return (uint64_t)timespec.tv_sec * ONE_SEC + timespec.tv_nsec;
}

//각 패킷들의 헤더를 변경
void swp_hdr(unsigned char* ptr){
    unsigned char tmp_mac[6];
    unsigned char tmp_ip[4];
    unsigned char tmp_port[2];
	int i = 0;
	// Swap mac
	for(i = 0; i < 6; i++){
		tmp_mac[i] = ptr[i];
		ptr[i] = ptr[i + 6];
		ptr[i + 6] = tmp_mac[i];
	}
	// Swap ip
	for(i = 26; i < 30; i++){
		tmp_ip[i-26] = ptr[i];
		ptr[i] = ptr[i + 4];
		ptr[i + 4] = tmp_ip[i-26];
	}
	// Swap port
	for(i = 34; i < 36; i++){
		tmp_port[i-34] = ptr[i];
		ptr[i] = ptr[i + 2];
		ptr[i + 2] = tmp_port[i-34];
	}
}

//rte_mbuf의 패킷들을 캐릭터배열에 복사
//기존에는 rte_mbuf의 패킷의 포인터를 캐릭터포인터배열에 저장하는 형식이었으나
//batch를 활용한 GPU와의 결합을 위해 변경했다.
void make_char_buf(struct rte_mbuf *buf[], unsigned char *pktQ, int head, int nb){
	for(int i = 0; i < nb; i++){
		memcpy(pktQ + head * PKT_SIZE + i * PKT_SIZE + HEADROOM_SIZE, (rte_ctrlmbuf_data(buf[i])), PKT_DATA_SIZE + 4);
	}
}

//캐릭터배열의 패킷들을 rte_mbuf에 복사
//각 패킷들의 길이를 rte_mbuf에 저장 -> 저장안 할 경우 NIC이 TX하지 않음
void make_rte_buf(unsigned char * pktQ, struct rte_mbuf **buf, int head, int nb){
	for(int i = 0; i < nb; i++){
		memcpy((rte_ctrlmbuf_data(buf[i])), pktQ + head*PKT_SIZE + i*PKT_SIZE + HEADROOM_SIZE - IPSECHEAD, PKT_DATA_SIZE + IPSECHEAD + IPSECTAIL + 4);
		buf[i]->pkt_len = PKT_DATA_SIZE + IPSECHEAD + IPSECTAIL;
		buf[i]->data_len = PKT_DATA_SIZE + IPSECHEAD + IPSECTAIL;
	}
}

//swp_hdr함수를 패킷 버퍼의 각 패마다 적용해주는 함수
void swp_hdr_buf(struct rte_mbuf **buf, int head, int nb){
	for(int i = head; i < nb; i++){
		swp_hdr(rte_ctrlmbuf_data(buf[i]));
	}
}

//1초마다 한번씩 카운트된 패킷을 출력해주는 함수
void *cpu_monitoring_loop(void *data){
	uint64_t start;
	uint64_t end;
	printf("START\n");

	start = monotonic_time();
	
	while(1){
		end = monotonic_time();
		if(end - start > ONE_SEC){
			//printf("start : %llu, end : %llu, end - start : %llu\n", start, end, end - start);
			start = end;
			print_cur_stat(u_str);
		}
	}
}

//패킷의 수를 출력해주는 함수
//fancy의 함수를 참고
//get_gpu_cnt함수를 통해 GPU가 카운트한 패킷 수를 가져오기 때문에 GPU를 사용하지 않는
//DPDK버전에서는 GPU를 0으로 만들어줘야한다.
void print_cur_stat(const char ** u_str)
{
	static uint32_t elapsed_time = 0;
	elapsed_time++;
	system("clear");
	double rpps = cpu_recv;
#if GPU
	double gpps = get_gpu_cnt();
#else
	double gpps = 0;
#endif
	double tpps = cpu_tran;

	cpu_recv = 0;
	cpu_tran = 0;

	uint64_t rtmp = rpps;
	uint64_t gtmp = gpps;
	uint64_t ttmp = tpps;

	double rbps = rpps * (PKT_DATA_SIZE + 4) * 8 + rpps * 20 * 8;
	double gbps = gpps * (PKT_DATA_SIZE + 4) * 8 + gpps * 20 * 8;
	double tbps = tpps * (PKT_DATA_SIZE + 4) * 8 + tpps * 20 * 8;

	double rpercent;
	double gpercent;
	double tpercent;

	int rpunit = 0;
	int gpunit = 0;
	int tpunit = 0;

	int rbunit = 0;
	int gbunit = 0;
	int tbunit = 0;

	while(rpps >= 1000){
		rpps /= 1000;
		rpunit++;
	}

	while(gpps >= 1000){
		gpps /= 1000;
		gpunit++;
	}

	while(tpps >= 1000){
		tpps /= 1000;
		tpunit++;
	}

	while(rbps >= 1000){
		rbps /= 1000;
		rbunit++;
	}
	rpercent = rbps;
	for(int i = rbunit; i < 3; i++)
		rpercent /= 1000;
	rpercent *= 100/10;

	while(gbps >= 1000){
		gbps /= 1000;
		gbunit++;
	}
	gpercent = gbps;
	for(int i = gbunit; i < 3; i++)
		gpercent /= 1000;
	gpercent *= 100/10;

	while(tbps >= 1000){
		tbps /= 1000;
		tbunit++;
	}
	tpercent = tbps;
	for(int i = tbunit; i < 3; i++)
		tpercent /= 1000;
	tpercent *= 100/10;

	printf("DPDK with GPU\n");
	printf("BATCH : ");
	START_RED
	printf("%d(%d + %d)", PKT_BATCH_NUM, PKT_BATCH_NUM - RX_NB, RX_NB);
	END
	printf("\tPKT DATA SIZE : ");
	START_RED
	printf("%d(%d + %d + %d)", IPSECHEAD + PKT_DATA_SIZE + IPSECTAIL, IPSECHEAD, PKT_DATA_SIZE, IPSECTAIL);
	END
	printf("\tPKT BUF SIZE : ");
	END
	START_RED
	printf("%d(%d + %d + %d)\n", PKT_SIZE, HEADROOM_SIZE, PKT_DATA_SIZE, TAILROOM_SIZE);
	END

	printf("RX\t: %.4lf%spps(%lld)\t%.4lf%sbps(", rpps, u_str[rpunit], rtmp, rbps, u_str[rbunit]);
	if(rpercent >= 99){
		START_GRN
		printf("%.2lf%%", rpercent);
		END
	}
	else if(rpercent >= 33){
		START_YLW
		printf("%.2lf%%", rpercent);
		END
	}
	else{
		START_RED
		printf("%.2lf%%", rpercent);
		END
	}
	printf(")\n");

	printf("GPU\t: %.4lf%spps(%lld)\t%.4lf%sbps(", gpps, u_str[gpunit], gtmp, gbps, u_str[gbunit]);
	if(gpercent >= 99){
		START_GRN
		printf("%.2lf%%", gpercent);
		END
	}
	else if(gpercent >= 33){
		START_YLW
		printf("%.2lf%%", gpercent);
		END
	}
	else{
		START_RED
		printf("%.2lf%%", gpercent);
		END
	}
	printf(")\n");

	printf("TX\t: %.4lf%spps(%lld)\t%.4lf%sbps(", tpps, u_str[tpunit], ttmp, tbps, u_str[tbunit]);
	if(tpercent >= 99){
		START_GRN
		printf("%.2lf%%", tpercent);
		END
	}
	else if(tpercent >= 33){
		START_YLW
		printf("%.2lf%%", tpercent);
		END
	}
	else{
		START_RED
		printf("%.2lf%%", tpercent);
		END
	}
	printf(")\n");

	printf("Elapsed time : %um %us\n", elapsed_time / 60, elapsed_time % 60);
	
}

//하나의 패킷을 dump하는 함수
void print_pkt(unsigned char * ptr, int idx)
{
	START_GRN
	printf("[CPU] %dth pkt dump: \n", idx);
	for(int i = 0; i < IPSECHEAD + PKT_DATA_SIZE + IPSECTAIL; i++){
		if(i != 0 && i % ONELINE == 0)
			printf("\n");
		printf("%02x ", ptr[i]);
	}
	printf("\n\n");
	END
}


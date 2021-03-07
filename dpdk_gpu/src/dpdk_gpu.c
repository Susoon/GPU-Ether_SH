#include "dpdk_gpu.h"

#define THREAD 1
#ifdef THREAD
#define CORE 0
#else
#define CORE 1
#endif

#define TX 1

//ethernet interface에 할당된 mempool
struct rte_mempool *mbuf_pool;
//cpu_monitoring함수가 카운트된 패킷수를 확인하기 위해 사용되는 변수들
double cpu_recv = 0;
double cpu_tran = 0;

//master thread함수
extern void master(void *data);

//패킷 RX/TX Queue와 각 Queue의 head를 담고 있는 구조체
static struct tunnelQ *tQ;

//static 변수이 tQ를 다른 파일에서 호출하기 위한 함수
void gettQ(struct tunnelQ **tmp){
	*tmp = tQ;
}

static void IOloop(uint8_t lid){
	//rx와 tx, 그리고 master thread가 헤더 swap을 마친 패킷 수
	uint16_t nb_rx;
	uint16_t nb_tx;
	uint16_t nb_swp;
	//DPDK가 사용하는 패킷 버퍼
	//PKT_BATCH_NUM는 512이다.
	struct rte_mbuf *buf[PKT_BATCH_NUM];

	start_lcore(l2p, lid);

	while(lcore_is_running(l2p, lid)){
		nb_rx = rte_eth_rx_burst(0, 0, buf, DEFAULT_PKT_BURST);

		//nb_rx가 0보다 크면 -> rx에 성공하면
		if(nb_rx > 0){
			cpu_recv += nb_rx;
			//rhead(rx queue의 head)가 PKT_BATCH_NUM - RX_NB(512 - 32)보다 작을때
			// -> master thread가 worker thread보다 tQ 변수에 접근이 느려 512개를 넘지 않게 제한을 걸어둠
			if(tQ->rhead < PKT_BATCH_NUM - RX_NB){
				//buf에 있는 DPDK패킷 버퍼를 RXQueue에 rhead부터 nb_rx개만큼  캐릭터 배열로 저장
				//cpu_recv += nb_rx;
				make_char_buf(buf, tQ->pktRXQ, tQ->rhead, nb_rx);
				tQ->rhead += nb_rx;
			}

			for(int i = 0; i < nb_rx; i++)
				rte_pktmbuf_free(buf[i]);
		}

#if TX
		//thead는 master thread가 변경
		//master thread가 GPU에서 패킷을 받아와 thead값을 변경시키면
		if(tQ->thead > 0){
			nb_swp = tQ->thead;

			//nb_swp만큼-> 480~512개의 개수만큼 버퍼를 할당받아옴
			for(int i = 0; i < nb_swp; i++)
				buf[i] = rte_pktmbuf_alloc(mbuf_pool);

			//TXQueue에 있는 패킷들을 DPDK버퍼인 buf에 저장
			make_rte_buf(tQ->pktTXQ, buf, 0, nb_swp);
			tQ->thead = 0;
			//TXQueue에 있는 패킷들을 0번부터 nb_swp개만큼 헤더를 변경
			swp_hdr_buf(buf, 0, nb_swp);

		//	for(int i = 0; i < nb_swp; i++)
		//		print_pkt(rte_ctrlmbuf_data(buf[i]), i);

			nb_tx = rte_eth_tx_burst(0, 0, buf, nb_swp);
			cpu_tran += nb_tx;

			for(int i = nb_tx; i < nb_swp; i++)
				rte_pktmbuf_free(buf[i]);
			nb_tx = 0; nb_swp = 0;
		}
#endif
	}
	printf("End of IOloop!! from lid: %d\n", lid);
}


int launch_one_lcore(void *arg __rte_unused)
{
	uint8_t lid = rte_lcore_id();
	IOloop(lid);
	return 0;
}

//dpdk를 위한 환경을 세팅하는 함수
//tunnelQ와 field들을 모두 할당해줌
//CORE와 THREAD 매크로 변경을 통해 master therad를 lcore로 실행시킬지 pthread로 실행시킬지 선택 가능
void dpdk_handler(int argc, char **argv)
{
	int ret;
	uint32_t sid; // Socket id
	int i;

	tQ = (struct tunnelQ *)malloc(sizeof(struct tunnelQ));
	tQ->pktRXQ = (unsigned char*)malloc(sizeof(unsigned char) * PKT_BATCH_SIZE);
	tQ->pktTXQ = (unsigned char*)malloc(sizeof(unsigned char) * PKT_BATCH_SIZE);
	tQ->rhead = 0;
	tQ->thead = 0;


#if PTHREAD_CNT	
	pthread_t thread;

	pthread_create(&thread, NULL, cpu_monitoring_loop, NULL); 
#endif

#if THREAD 
	pthread_t thread1;

	pthread_create(&thread1, NULL, master, NULL); 
#endif

	if((l2p = l2p_create()) == NULL)
		printf("Unable to create l2p\n");

	/* Initialize the Environment Abstraction Layer (EAL). */
	// This function is to be executed on the MASTER lcore only.
	ret = rte_eal_init(argc, argv);
	if(ret < 0)
		rte_exit(EXIT_FAILURE, "Error with EAL initialization.\n");

	/* Check if at least one port is available. */
	if(rte_eth_dev_count_total() == 0)
	 	rte_exit(EXIT_FAILURE, "Error: No port available.\n");
	/* Configure the Ethernet device */
	/* Params,
	 * (1) port id
	 * (2) nb_rx_queue
	 * (3) nb_tx_queue
	 * (4) eth_conf (The pointer to the configuration data to be used)
	 */
	ret = rte_eth_dev_configure(0, 1, 1, &default_port_conf);

	if(ret < 0)
		rte_exit(EXIT_FAILURE, "Cannot configure device: port %d.\n", 0);

	sid = rte_lcore_to_socket_id(1); // lcore 1

	/* Create a new mempool in memory to hold the mbufs. */
	/* Params,
	 * (1) The name of the mbuf pool.
	 * (2) The number of elements in the mbuf pool.
	 * (3) Size of per-core object cache. 
	 * (4) Size of the application private are between the rte_mbuf structure and the data buffer.
	 * (5) Size of data buffer in each mbuf.
	 * (6) The socket identifier where the memory should be allocated.
	*/
	mbuf_pool = rte_pktmbuf_pool_create("MBUF_POOL", NUM_MBUFS_DEFAULT, MBUF_CACHE_SIZE, 0, RTE_MBUF_DEFAULT_BUF_SIZE, sid);
	if(mbuf_pool == NULL)
		rte_exit(EXIT_FAILURE, "Cannot create mbuf pool\n");

	/* Allocate and set up RX queues. */
	/* Params,
	 * (1) port_id
	 * (2) rx_queue_id
	 * (3) nb_rx_desc
	 * (4) socket _id
	 * (5) rx_conf (The pointer to the configuration data to be used)
	 * (6) mb_pool (The pointer to the memory pool)
	*/
	ret = rte_eth_rx_queue_setup(0, 0, RX_DESC_DEFAULT, sid, NULL, mbuf_pool);
	if(ret)
		rte_exit(EXIT_FAILURE, "RX : Cannot init port %"PRIu8 "\n", 0);

	ret = rte_eth_tx_queue_setup(0, 0, RX_DESC_DEFAULT, sid, NULL);
	if(ret)
		rte_exit(EXIT_FAILURE, "TX : Cannot init port %"PRIu8 "\n", 0);


	/* Stats bindings (if more than one queue) */
	/* Params, 
	 * (1) port_id
	 * (2) rx_queue_id
	 * (3) stat_idx
	*/
	rte_eth_dev_set_rx_queue_stats_mapping(0, 0, 1);

	/* Display the port MAC address. */
	struct ether_addr addr;
	rte_eth_macaddr_get(0, &addr);
	printf("\n[CKJUNG]  Port %u: MAC=%02" PRIx8 ":%02" PRIx8 ":%02" PRIx8":%02" PRIx8 ":%02" PRIx8 ":%02" PRIx8 ", RXdesc/queue=%d\n", 0, addr.addr_bytes[0], addr.addr_bytes[1], addr.addr_bytes[2],addr.addr_bytes[3], addr.addr_bytes[4], addr.addr_bytes[5],RX_DESC_DEFAULT);

	/* Launch core job (Receiving pkt infinitely */
	/* Params, 
	 * (1) The function to be called
	 * (2) arg (arg for the function)
	 * (3) slave_id (The identifier of the lcore on which the function should be executed)
	*/
	if(rte_eal_remote_launch(launch_one_lcore, NULL, 1) < 0)
		rte_exit(EXIT_FAILURE, "Could not launch capture process on lcore %d.\n", 0);
#if CORE
	if(rte_eal_remote_launch(master, NULL, 2) < 0)
		rte_exit(EXIT_FAILURE, "Could not launch capture process on lcore %d.\n", 0);
#endif
	/* Start the port once everything is ready. */
	ret = rte_eth_dev_start(0);
	if(ret)
		rte_exit(EXIT_FAILURE, "Cannot start port %"PRIu8 "\n", 0);

	/* Enable RX in promiscuous mode for the Ethernet device. */
	 //rte_eth_promiscuous_enable(0);
	 //printf("[CKJUNG] <Enable promiscuous mode> \n");

	/* Write down previous stats */
	rte_eth_stats_get(0, &(ck_dpdk.info[0].prev_stats));

	//rte_timer_setup();

	/* Wait for all of the cores to stop running and exit. */
	ret = rte_eal_wait_lcore(1);
	if(ret < 0)
		rte_exit(EXIT_FAILURE, "Core %d did not stop correctly. \n", 1);

	RTE_ETH_FOREACH_DEV(i) {
		rte_eth_dev_stop(i);
		//rte_delay_us_sleep(100 * 1000);
		rte_eth_dev_close(i);
	}
}


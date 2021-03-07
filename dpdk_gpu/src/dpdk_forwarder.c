#include "dpdk.h"

struct rte_mempool *mbuf_pool;
double cpu_recv = 0;
int copy_cnt = 0;

static struct tunnelQ *tQ;

unsigned char** pktRXQ;

void gettQ(struct tunnelQ **tmp){
	*tmp = tQ;
}

static void IOloop(uint8_t lid)
{
	//rx와 tx, 그리고 forwarder thread가 헤더 swap을 마친 패킷 수
	uint16_t nb_rx;
	uint16_t nb_tx;
	uint16_t nb_swp;
	//dpdk에 사용되는 패킷 버퍼
	struct rte_mbuf *buf[DEFAULT_PKT_BURST];

	tQ = (struct tunnelQ *)malloc(sizeof(struct tunnelQ));
	pktRXQ = (unsigned char**)malloc(sizeof(unsigned char*) * 64);
	tQ->pktRXQ = pktRXQ;

	start_lcore(l2p, lid);

	while(lcore_is_running(l2p, lid)){
		nb_rx = rte_eth_rx_burst(0, 0, buf, DEFAULT_PKT_BURST);
		if(nb_rx > 0){
		//printf("nb_rx: %d\n", nb_rx);
			tQ->rhead = nb_rx;
			cpu_recv += nb_rx;

			make_char_buf(buf, pktRXQ, nb_rx);

			for(int i = 0; i < nb_rx; i++)
				rte_pktmbuf_free(buf[i]);
		}
		if(tQ->thead > 0){
			nb_swp = tQ->thead;
			tQ->thead = 0;
			for(int i = 0; i < nb_swp; i++)
				buf[i] = rte_pktmbuf_alloc(mbuf_pool);

			make_rte_buf(pktRXQ + 32, buf, nb_swp);
			//print_pkt(rte_ctrlmbuf_data(buf[0]));
		
		
			nb_tx = rte_eth_tx_burst(0, 0, buf, nb_swp);

			for(int i = nb_tx; i < nb_swp; i++)
				rte_pktmbuf_free(buf[i]);
			nb_tx = 0; nb_swp = 0;
		}
	}
	printf("End of IOloop!! from lid: %d\n", lid);
}


int launch_one_lcore(void *arg __rte_unused)
{
	uint8_t lid = rte_lcore_id();
	IOloop(lid);
	return 0;
}

void dpdk_handler(int argc, char **argv)
{
	int ret;
	uint32_t sid; // Socket id
	int i;


#if PTHREAD_CNT	
	pthread_t thread;

	pthread_create(&thread, NULL, cpu_monitoring_loop, NULL); 
#endif

#if 1
	pthread_t thread1;

	pthread_create(&thread1, NULL, forwarder, NULL); 
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


#include "dpdk_gpuone.h"

#define TX 1

struct rte_mempool *mbuf_pool;
double cpu_recv = 0;
double cpu_tran = 0;

static struct tunnelQ *tQ;

void gettQ(struct tunnelQ **tmp){
	*tmp = tQ;
}

static void IOloop(uint8_t lid){
	uint16_t nb_rx;
	uint16_t nb_tx;
	uint16_t nb_swp;
	struct rte_mbuf *buf[PKT_BATCH_NUM];

	start_lcore(l2p, lid);

	while(lcore_is_running(l2p, lid)){
		nb_rx = rte_eth_rx_burst(0, 0, buf, DEFAULT_PKT_BURST);

		if(nb_rx > 0){
			cpu_recv += nb_rx;
			if(tQ->rhead < PKT_BATCH_NUM - RX_NB){
				make_char_buf(buf, tQ->pktRXQ, tQ->rhead, nb_rx);
				tQ->rhead += nb_rx;
			}
			if(tQ->rhead >= PKT_BATCH_NUM - RX_NB){
				copy_to_gpu(tQ->pktRXQ, tQ->rhead);
				tQ->rhead = 0;
			}

			for(int i = 0; i < nb_rx; i++)
				rte_pktmbuf_free(buf[i]);
		}

#if TX
		if((tQ->thead = copy_from_gpu(tQ->pktTXQ)) > 0){
			nb_swp = tQ->thead;

			for(int i = 0; i < nb_swp; i++)
				buf[i] = rte_pktmbuf_alloc(mbuf_pool);
			make_rte_buf(tQ->pktTXQ, buf, 0, nb_swp);
			tQ->thead = 0;
		
			swp_hdr_buf(buf, 0, nb_swp);
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

#if PKT_MONITORING
	pthread_t thread;

	pthread_create(&thread, NULL, cpu_monitoring_loop, NULL); 
#endif

	if((l2p = l2p_create()) == NULL)
		printf("Unable to create l2p\n");

	ret = rte_eal_init(argc, argv);
	if(ret < 0)
		rte_exit(EXIT_FAILURE, "Error with EAL initialization.\n");

	if(rte_eth_dev_count_total() == 0)
	 	rte_exit(EXIT_FAILURE, "Error: No port available.\n");
	ret = rte_eth_dev_configure(0, 1, 1, &default_port_conf);

	if(ret < 0)
		rte_exit(EXIT_FAILURE, "Cannot configure device: port %d.\n", 0);

	sid = rte_lcore_to_socket_id(1); // lcore 1

	mbuf_pool = rte_pktmbuf_pool_create("MBUF_POOL", NUM_MBUFS_DEFAULT, MBUF_CACHE_SIZE, 0, RTE_MBUF_DEFAULT_BUF_SIZE, sid);
	if(mbuf_pool == NULL)
		rte_exit(EXIT_FAILURE, "Cannot create mbuf pool\n");

	ret = rte_eth_rx_queue_setup(0, 0, RX_DESC_DEFAULT, sid, NULL, mbuf_pool);
	if(ret)
		rte_exit(EXIT_FAILURE, "RX : Cannot init port %"PRIu8 "\n", 0);

	ret = rte_eth_tx_queue_setup(0, 0, RX_DESC_DEFAULT, sid, NULL);
	if(ret)
		rte_exit(EXIT_FAILURE, "TX : Cannot init port %"PRIu8 "\n", 0);


	rte_eth_dev_set_rx_queue_stats_mapping(0, 0, 1);

	struct ether_addr addr;
	rte_eth_macaddr_get(0, &addr);
	printf("\n[CKJUNG]  Port %u: MAC=%02" PRIx8 ":%02" PRIx8 ":%02" PRIx8":%02" PRIx8 ":%02" PRIx8 ":%02" PRIx8 ", RXdesc/queue=%d\n", 0, addr.addr_bytes[0], addr.addr_bytes[1], addr.addr_bytes[2],addr.addr_bytes[3], addr.addr_bytes[4], addr.addr_bytes[5],RX_DESC_DEFAULT);

	if(rte_eal_remote_launch(launch_one_lcore, NULL, 1) < 0)
		rte_exit(EXIT_FAILURE, "Could not launch capture process on lcore %d.\n", 0);

	ret = rte_eth_dev_start(0);
	if(ret)
		rte_exit(EXIT_FAILURE, "Cannot start port %"PRIu8 "\n", 0);

	rte_eth_stats_get(0, &(ck_dpdk.info[0].prev_stats));

	ret = rte_eal_wait_lcore(1);
	if(ret < 0)
		rte_exit(EXIT_FAILURE, "Core %d did not stop correctly. \n", 1);

	RTE_ETH_FOREACH_DEV(i) {
		rte_eth_dev_stop(i);
		rte_eth_dev_close(i);
	}
}


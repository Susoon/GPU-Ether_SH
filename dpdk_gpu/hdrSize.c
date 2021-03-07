#include <stdio.h>
#include <net/ethernet.h>
#include <netinet/ip.h>
#include <netinet/udp.h>
#include <netinet/tcp.h>
#include <linux/ipv6.h>

int main(void){

    printf("Size of Ether Header : %ld, IPv4 Header : %ld, IPv6 Header : %ld, UDP Header : %ld, TCP Header : %ld\n", sizeof(struct ether_header), sizeof(struct iphdr), sizeof(struct ipv6hdr), sizeof(struct udphdr), sizeof(struct tcphdr));

}

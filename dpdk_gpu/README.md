# DPDK Evaluation

* Evaluation을 위한 5가지 버전의 DPDK
* src : 소스코드 파일
* include : 헤더파일
* main : 각 dpdk버전에 해당하는 main 함수가 구현된 코드 파일
* apps : NF 코드 파일

* bashrc파일에 아래의 내용을 추가할것
```
export PKG_CONFIG_PATH=/usr/lib/pkgconfig
```

---
## 파일 구성
1. src 폴더
   * 각 DPDK 버전에 필요한 소스코드 파일이 내장되어 있다.
   * dpdk로 시작하는 모든 파일들이 실제 DPDK의 코드를 내장한 파일이다.
   * dpdk\_orig.c파일은 모든 dpdk의 원본이되는 파일이다.
   * util.c 파일은 모든 dpdk 버전에 공용으로 사용되는 함수와 매크로들을 담은 파일이다.
      * 매크로인 GPU를 변경하는 경우를 제외하고는 모두 동일한 함수가 사용된다.
      * GPU 값을 변경할 경우 print\_cur\_stat이라는 monitoring한 값을 출력하는 함수가 변경된다.
   * backup\_files는 초기 변경시에 만들었던 백업 파일들을 모아둔 것이다.
      * 현재는 필요없다.
   * forwarder.c와 master.c, gpu\_forwarder.c 파일은 각 버전에 필요한 코드를 담아둔 파일이다.

2. include 폴더
   * 각 DPDK버전에 사용된 소스코드 파일들의 헤더파일이 내장되어 있다.
   * src폴더의 소스코드와 구성이 동일하다.
   * dpdk.h파일은 모든 dpdk버전에 사용되는 매크로들이 내장된 파일이다.

3. main 폴더
   * 각 DPDK버전에 사용되는 main함수들이 구현된 파일이 내장되어 있다.
   * mainio.c는 dpdkIO에, mainforwarder.c는 dpdk\_forwarder에, maingpu.c는 dpdkGPU와 dpdkGPUONE에 사용된다.
   * mainio.c와 mainforwarder.c는 dpdk\_handler함수만을 호출한다.
      * dpdk\_handler함수는 DPDK에 필요한 변수와 메모리를 세팅한 뒤 DPDK를 실행시키는 함수이다.
      * mainio.c와 mainforwarder.c의 차이점은 헤더파일이다.
         * mainio.c는 dpdkio.h를, mainforwarder.c는 dpdk\_forwarder.h를 호출한다.
   * maingpu.c는 set\_gpu\_mem\_for\_dpdk함수와 monitor함수, dpdk\_handler함수를 호출한다.
      * set\_gpu\_mem\_for\_dpdk함수는 DPDK에 사용되는 GPU메모리를 세팅해주는 함수이다.
      * monitor함수는 GPU상에 패킷을 감지한 후 카운트하는 함수이며 pthread로 실행된다.

4. apps 폴더
   * NF 코드들이 내장되어있다.
   * 현재 모두 single thread로 구현된 코드이며 IPSec을 제외한 나머지 코드들은 정상작동한다.
   * IPSec 코드는 pktBuf에서 에러가 발생한다.
      * GPU의 IPSec 처리 속도에 의한 에러로 추정중

---
## tunnelQ

* Worker\-Master 형식으로 구현된 버전에 사용된다.
* Worker와 Master thread가 서로 정보를 공유하기위해 사용된다.
* RXQueue와 TXQueue, rhead, thead를 담고 있는 구조체이다.
   * rhead는 RXQueue의 head, thead는 TXQueue의 head를 의미한다.

---
## 각 DPDK버전 세부 설명 

---
### 1. dpdkIO

* dpdkio.sh 파일을 통해 컴파일 및 실행
* **util.c dpdkio.c mainio.c**을 사용
   * 위의 파일을 사용하기위해 **util.h dpdkio.h**파일을 사용
* GPU와 결합되지 않고 CPU만 사용하는 DPDK I/O 기능만을 확인하기 위한 파일
* util.c 파일에 정의된 매크로인 GPU를 0으로 설정 필요

---
### 2. dpdk\_forwarder

* forwarder.sh 파일을 통해 컴파일 및 실행
* **util.c dpdk_forwarder.c forwarder.c mainforwarder.c**파일을 사용
   * 위의 파일을 사용하기위해 **util.h dpdk_forwarder.h forwarder.h**파일을 사용
* GPU와 결합되지 않고 CPU만 사용하는 DPDK I/O기능을 Worker\-Master 형식으로 구현한 파일
* util.c 파일에 정의된 매크로인 GPU를 0으로 설정 필요
* **사용되지 않는 파일이기때문에 구현이 완료되지 않음**

---
### 3. dpdkGPU

* dpdkgpu.sh 파일을 통해 컴파일 및 실행
* **master.c gpu_forwarder.c dpdk_gpu.c util.c maingpu.c**파일을 사용
   * 위의 파일을 사용하기위해 **master.h gpu_forwarder.h dpdk_gpu.h util.h**파일을 사용
* GPU와 결합되어 NIC \-> CPU\(DPDK\) \-> GPU로의 패킷 전송이 진행되는 파일
* Worker\-Master 형식으로 구현된 파일
* util.c 파일에 정의된 매크로인 GPU를 1로 설정 필요

---
### 4. dpdkGPUONE

* dpdkgpuone.sh 파일을 통해 컴파일 및 실행
* **gpu_forwarder.c dpdk_gpuone.c util.c maingpu.c**파일을 사용
   * 위의 파일을 사용하기위해 **gpu_forwarder.h dpdk_gpuone.h util.h**파일을 사용
* GPU와 결합되어 NIC \-> CPU\(DPDK\) \-> GPU로의 패킷 전송이 진행되는 파일
* dpdkGPU와의 차이점은 dpdkGPUONE은 Worker\-Master 형식이 아닌 하나의 thread만을 사용하는 형식
* util,c 파일에 정의된 매크로인 GPU를 1로 설정 필요

---
### 5. dpdkGPUNF

* dpdkgpunf.sh 파일을 통해 컴파일 및 실행
* **gpu_forwarder.c dpdk_gpu.c util.c maingpunf.c router.cu nids.cu ipsec.cu**파일을 사용
   * 위의 파일을 사용하기위해 **gpu_forwarder.h dpdk_gpu.h util.h router.h nids.h ipsec.h gf\_tables.h sbox.h**파일을 사용
* GPU와 결합되어 NIC \-> CPU\(DPDK\) \-> GPU로의 패킷 전송이 진행되는 파일
* Worker\-Master 형식으로 구현된 DPDK와 NF가 결합된 파일
* maingpunf.c파일에서 initialize\_(NF)함수를 주석해제하면 해당 NF가 실행된다.
   * e.g.) initialize\_router(1);
* util,c 파일에 정의된 매크로인 GPU를 1로 설정 필요


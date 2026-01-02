# 设置平台
PLATFORM := xilinx_u280_gen3x16_xdma_1_202211_1
TARGET := hw
TEMP_DIR := ./_x.$(TARGET)
BUILD_DIR := ./build

# 文件定义
KERNEL_XO := forward.xo
XCLBIN := $(BUILD_DIR)/forward.xclbin
HOST_EXE := $(BUILD_DIR)/host_app

# 编译器设置
VPP := v++
GCC := g++

# 默认目标
all: $(XCLBIN) $(HOST_EXE)

# 1. 调用 Vitis HLS 生成 .xo 文件
$(KERNEL_XO): src/forward.cpp src/forward.h
	vitis_hls -f run_hls.tcl

# 2. 调用 v++ 链接生成 .xclbin 文件
$(XCLBIN): $(KERNEL_XO)
	mkdir -p $(BUILD_DIR)
	$(VPP) -t $(TARGET) --platform $(PLATFORM) --link $(KERNEL_XO) \
		--config link.cfg --kernel_frequency 250 \
		-o $(XCLBIN) --temp_dir $(TEMP_DIR) --vivado.synth.jobs 16 --vivado.impl.jobs 16

# 3. 编译主机端程序 (需要安装 XRT)
$(HOST_EXE): src/host.cpp
	mkdir -p $(BUILD_DIR)
	$(GCC) -Wall -g -I$(XILINX_XRT)/include -I./src -L$(XILINX_XRT)/lib \
		src/host.cpp -lxilinxopencl -lpthread -lrt -o $(HOST_EXE) -fuse-ld=bfd -std=c++17 -lxrt_coreutil

clean:
	rm -rf prj_hls $(KERNEL_XO) $(BUILD_DIR) $(TEMP_DIR) *.log *.jou .ipcache .Xil
# 创建项目
open_project -reset prj_hls
set_top forward

# 添加源文件
add_files src/forward.cpp -cflags "-I./src"
add_files src/forward.h
add_files src/typedefs.h
add_files src/config.h

# 设置解决方案和 U280 芯片型号
open_solution -reset "solution1" -flow_target vitis
set_part xcu280-fsvh2892-2L-e
create_clock -period 10.00 -name default

# 运行综合并导出 XO
csynth_design
export_design -format xo -output forward.xo
exit
DATA_PATH = objs
TARGET = $(DATA_PATH)/main
OBJ_PATH = $(DATA_PATH)/build
PREFIX_BIN = 

# 指定编译器
CC = g++
# include文件路径
INCLUDES = 
# lib文件路径
LIBS = 
# 编译选项
CFLAGS = -pg
# 链接选项
LINKFLAGS =
# 本地路径（当前目录）
LOCAL_PATH = .
# 编译文件的路径
SRCDIR = . lenet/sw mnist parameters lenet lenet/hw lenet/sw_golden

# 增加include文件路径
INCLUDES += -I$(LOCAL_PATH)/include
INCLUDES += -I/tools/Xilinx/Vivado/2019.1/include

# C程序
C_SRCDIR = $(SRCDIR)
# 所有的.c源文件
# foreach从C_SRCDIR中取出每一个路径放到一个叫d的变量中
# 在第三部分引用d所存储的变量并查找其中所有的.c文件，放入C_SOURCES中
C_SOURCES = $(foreach d,$(C_SRCDIR), $(wildcard $(d)/*.c))
# .c文件生成的中间文件
# patsubst将C_SOURCES中的所有.c文件替换成$(OBJ_PAT)目录中的.o文件
# 并将.o文件返回存储到C_OBJS中
C_OBJS = $(patsubst %.c, $(OBJ_PATH)/%.o, $(C_SOURCES))

# C++程序
CPP_SRCDIR = $(SRCDIR)
# 所有.cpp源文件
CPP_SOURCES = $(foreach d, $(C_SRCDIR), $(wildcard $(d)/*.cpp))
# .cpp文件生成的中间文件
CPP_OBJS = $(patsubst %.cpp, $(OBJ_PATH)/%.o, $(CPP_SOURCES))

# make默认执行的命令，依赖于init	compile
default:init compile

# 对于每一个要生成的目标文件echo "test"	依赖文件即为它的自身.c或.cpp文件
# 利用静态模式生成目标文件
# 静态模式的格式: <targets...>:	<target-parttern>:<prerequest-pattern>
# 					<command>	
$(C_OBJS):$(OBJ_PATH)/%.o:%.c
	$(CC) -c $(CFLAGS) $(INCLUDES) $< -o $@

$(CPP_OBJS):$(OBJ_PATH)/%.o:%.cpp
	$(CC) -c $(CFLAGS) $(INCLUDES) $< -o $@	

# 为每一个源文件路径在OBJ_PATH下生成一个对应的目标文件的路径
init:
	mkdir -p $(foreach d, $(SRCDIR), $(OBJ_PATH)/$(d))

# 输出测试信息
test:
	@echo "C_SOURCES: $(C_SOURCES)"
	@echo "C_OBJS: $(C_OBJS)"
	@echo "CPP_SOURCES: $(CPP_SOURCES)"
	@echo "CPP_OBJS: $(CPP_OBJS)"

# 对.c文件和.cpp文件进行链接，目标文件放置于TARGET
compile:$(C_OBJS) $(CPP_OBJS)
	$(CC) $(CFLAGS) $^ -o $(TARGET) $(LINKFLAGS) $(LIBS)

# 清除所有生成的目标文件和中间文件
clean:
	rm -rf $(DATA_PATH)
	rm -f $(TARGET)

# 安装：将所有的目标文件移动到bin目录中
install:${TARGET}
	cp $(TARGET) $(PREFIX_BIN)

# 卸载：将bin目录中的所有TARGET文件移除
uninstall:
	rm -f $(PREFIX_BIN)/$(TARGET)

# 重新生成：执行clean命令后执行compile
rebuild: clean default
# 定义编译器和编译选项
CC = gcc
#编译选项
CFLAGS = -g -O2 -Wall -Werror
CFLAGS = -Iinclude

# 定义源文件和目标文件
SRCS = lib/hello.c lib/add.c lib/mul.c
OBJS = $(SRCS:.c=.o)
TARGET = my_program.out

# 默认目标
all: $(TARGET)

# 链接目标文件生成可执行文件
$(TARGET): $(OBJS)
	$(CC) -o $@ $^

# 编译源文件为目标文件
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# 清理生成的文件
clean:
	del /Q lib\hello.o lib\add.o lib\mul.o $(TARGET)

.PHONY: all clean

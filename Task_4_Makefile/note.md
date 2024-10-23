
一个C文件到一个可执行的文件的全部流程。

### 1. 源代码编写
编写 C 语言源代码，以 `.c` 为后缀。例如，创建一个名为 `hello.c` 的文件

```c
#include <stdio.h>

int main() {
    printf("Hello, World!\n");
    return 0;
}
```

### 2. 预处理
在编译之前，C 编译器会首先对源代码进行预处理。这个过程包括：
- 处理以 `#` 开头的指令，如 `#include` 和 `#define`。
- 展开宏和文件包含。

运行预处理可以使用以下命令：
```bash
gcc -E hello.c -o hello.i
```
这将生成一个中间文件 `hello.i`，其中包含了预处理后的代码。

### 3. 编译
预处理完成后，编译器将 `.i` 文件转换为汇编语言代码。这个过程会生成一个 `.s` 文件。可以使用以下命令进行编译：
```bash
gcc -S hello.i -o hello.s
```
这将生成一个汇编语言文件 `hello.s`。

### 4. 汇编
接下来，汇编器将汇编语言代码转换为机器代码，并生成一个目标文件（`.o` 文件）。可以使用以下命令进行汇编：
```bash
gcc -c hello.s -o hello.o
```
这将生成一个目标文件 `hello.o`，其中包含了机器代码。

### 5. 链接
最后，链接器将一个或多个目标文件（`.o` 文件）和所需的库文件链接在一起，生成最终的可执行文件。可以使用以下命令进行链接：
```bash
gcc hello.o -o hello.out
```
这将生成一个可执行文件 `hello.out`。

### 6. 运行可执行文件
生成可执行文件后，可以通过终端运行它：
```bash
./hello.out
```
输出将会是：
```
Hello, World!
```

### 整个流程总结
1. **源代码**：编写 C 代码（`hello.c`）。
2. **预处理**：`gcc -E hello.c -o hello.i`。
3. **编译**：`gcc -S hello.i -o hello.s`。
4. **汇编**：`gcc -c hello.s -o hello.o`。
5. **链接**：`gcc hello.o -o hello.out`。
6. **运行**：`./hello.out`。

### 使用 Makefile
为了简化这个过程，通常会使用 Makefile。可以创建一个简单的 Makefile，如下所示：

```makefile
CC = gcc
CFLAGS = -Wall

all: hello.out

hello.out: hello.o
	$(CC) hello.o -o hello.out

hello.o: hello.c
	$(CC) $(CFLAGS) -c hello.c

clean:
	del hello.o hello.out
```

### 运行 Makefile
在终端中执行 `make` 命令，编译过程将自动进行，生成可执行文件。执行 `make clean` 可以清理生成的文件。

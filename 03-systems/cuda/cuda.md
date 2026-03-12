# Custom C++ and CUDA Operators

### **CUDA的关键概念**

1. **Kernel（核函数）** ：
   - 核函数是在GPU上运行的函数。
   - 它会被多个线程并行调用。
   - 使用 `__global__` 关键字声明。
2. **线程与块（Threads and Blocks）** ：
   - 线程是最小的执行单元。
   - 线程被组织成“块”（Block），块又被组织成“网格”（Grid）。
   - 每个线程可以通过 `blockIdx` 和 `threadIdx` 获取自己的位置。
3. **共享内存（Shared Memory）** ：
   - 共享内存是GPU上的一种高速缓存，供同一个块内的线程共享。
   - 使用 `__shared__` 关键字声明。
4. **同步（Synchronization）** ：
   - 使用 `__syncthreads()` 确保块内的所有线程都完成某些操作后再继续。
5. **原子操作（Atomic Operations）** ：
   - 当多个线程同时写入同一内存地址时，使用原子操作（如 `atomicAdd`）避免冲突。





参考资料：

- [Custom C++ and CUDA Operators](https://pytorch.org/tutorials/advanced/cpp_custom_ops.html#cpp-custom-ops-tutorial)
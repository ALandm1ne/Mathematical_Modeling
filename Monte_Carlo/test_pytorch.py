import torch
import time

def test_cuda_power():
    print("-" * 50)
    # 1. 基础硬件检查
    if not torch.cuda.is_available():
        print("❌ 错误：未检测到 CUDA，请检查驱动或安装版本！")
        return
    
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    print(f"✅ 检测到 GPU: {gpu_name}")
    print(f"📊 显存容量: {gpu_mem:.2f} GB")
    print(f"📦 PyTorch 版本: {torch.__version__}")
    print(f"🛠️ CUDA 编译版本: {torch.version.cuda}")
    print("-" * 50)

    # 2. 算力压测：创建一个 5000x5000 的大矩阵（模拟大规模蒙特卡洛计算）
    size = 5000
    print(f"🚀 正在进行 {size}x{size} 矩阵运算测试...")
    
    # 预热 (Warm-up)
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    _ = torch.matmul(a, b)
    torch.cuda.synchronize()

    # 正式测速
    start_time = time.time()
    for _ in range(100):
      c = torch.matmul(a, b)
    torch.cuda.synchronize()
    avg_time = (time.time() - start_time) * 1000 / 100
    print(f"🔥 预热后的平均运算耗时: {avg_time:.2f} ms")

if __name__ == "__main__":
    test_cuda_power()
import numpy as np
import os

def generate_large_csv(filename, rows=1001, cols=1001):
    print(f"正在生成 {rows}x{cols} 的数据文件: {filename}...")
    
    # 检查文件是否已存在
    if os.path.exists(filename):
        os.remove(filename)

    with open(filename, 'w') as f:
        for i in range(rows):
            # 每次生成一行随机数，减少内存占用
            # 使用 float32 足够满足 0-1 随机值需求
            row_data = np.random.rand(cols)
            
            # 将数组转换为逗号分隔的字符串
            # map(str, ...) 配合 join 是处理大量数据的常用方法
            line = ",".join(map(lambda x: f"{x:.6f}", row_data))
            
            # 按照你的要求：数据点之间用逗号，且每行行末也有逗号
            f.write(line + ",\n")
            
            # 打印进度
            if (i + 1) % 1000 == 0:
                print(f"已完成: {i + 1} / {rows} 行")

    print(f"生成完毕！文件大小约: {os.path.getsize(filename) / (1024**2):.2f} MB")

if __name__ == "__main__":
    generate_large_csv("test_data.csv")
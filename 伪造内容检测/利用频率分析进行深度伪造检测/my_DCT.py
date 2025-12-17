import numpy as np
import matplotlib.pyplot as plt
# 设置中文字体（避免中文乱码）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 1. 创建一个简单的8×8测试图像（棋盘格）
np.random.seed(42)
image = np.array([
    [20*i+5*j for j in range(8)] for i in range(8)
], dtype=np.float32)


# 2. 最简DCT函数（一行核心计算）
def simple_dct(block):
    N = 8
    # 减去128（JPEG标准）
    block = block - 128
    dct_result = np.zeros((N, N))

    for u in range(N):
        for v in range(N):
            sum_val = 0
            for x in range(N):
                for y in range(N):
                    # DCT核心公式
                    cos1 = np.cos(np.pi * u * (2 * x + 1) / (2 * N))
                    cos2 = np.cos(np.pi * v * (2 * y + 1) / (2 * N))
                    sum_val += block[x, y] * cos1 * cos2

            # 归一化系数
            c_u = np.sqrt(1 / N) if u == 0 else np.sqrt(2 / N)
            c_v = np.sqrt(1 / N) if v == 0 else np.sqrt(2 / N)
            dct_result[u, v] = c_u * c_v * sum_val

    return dct_result


# 3. 计算DCT
dct_coeff = simple_dct(image)

# 4. 绘制结果
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

# 子图1: 原始图像
axes[0, 0].imshow(image, cmap='gray', vmin=0, vmax=255)
axes[0, 0].set_title('原始8×8图像')
axes[0, 0].axis('off')

# 在图像上添加网格
for i in range(9):
    axes[0, 0].axhline(y=i - 0.5, color='red', alpha=0.3, linewidth=0.5)
    axes[0, 0].axvline(x=i - 0.5, color='red', alpha=0.3, linewidth=0.5)

# 子图2: DCT系数（原始值）
im = axes[0, 1].imshow(dct_coeff, cmap='RdBu_r', vmin=-500, vmax=500)
axes[0, 1].set_title('DCT系数矩阵')
axes[0, 1].set_xlabel('水平频率 u')
axes[0, 1].set_ylabel('垂直频率 v')
plt.colorbar(im, ax=axes[0, 1])

# 标记DC系数
axes[0, 1].scatter(0, 0, c='red', s=100, marker='o', edgecolors='white')
axes[0, 1].text(0, 0, ' DC', color='red', fontweight='bold')

# 子图3: DCT系数（绝对值，对数尺度）
im2 = axes[0, 2].imshow(np.log1p(np.abs(dct_coeff)), cmap='hot')
axes[0, 2].set_title('DCT频谱（对数尺度）')
axes[0, 2].set_xlabel('水平频率 u')
axes[0, 2].set_ylabel('垂直频率 v')
plt.colorbar(im2, ax=axes[0, 2])

# 标记高频区域
rect = plt.Rectangle((3.5, 3.5), 4, 4, fill=False, edgecolor='cyan', linewidth=2)
axes[0, 2].add_patch(rect)
axes[0, 2].text(5.5, 5.5, '高频区域', color='cyan', ha='center', fontweight='bold')

# 子图4: 系数值分布
axes[1, 0].bar(range(64), np.abs(dct_coeff.flatten()), edgecolor='black')
axes[1, 0].set_title('DCT系数绝对值分布')
axes[1, 0].set_xlabel('系数索引（按行展开）')
axes[1, 0].set_ylabel('系数绝对值')
axes[1, 0].grid(True, alpha=0.3)

# 标记DC系数
axes[1, 0].bar(0, np.abs(dct_coeff.flatten()[0]), color='red', edgecolor='black')

# 子图5: 能量分布饼图
# 划分低频、中频、高频
low_freq = dct_coeff[:2, :2].flatten()
mid_freq = np.concatenate([dct_coeff[:2, 2:].flatten(), dct_coeff[2:, :2].flatten()])
high_freq = dct_coeff[2:, 2:].flatten()

low_energy = np.sum(low_freq ** 2)
mid_energy = np.sum(mid_freq ** 2)
high_energy = np.sum(high_freq ** 2)
total_energy = low_energy + mid_energy + high_energy

labels = ['低频 (u,v<2)', '中频', '高频']
sizes = [low_energy / total_energy, mid_energy / total_energy, high_energy / total_energy]
colors = ['lightgreen', 'lightblue', 'lightcoral']

axes[1, 1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
               startangle=90, explode=(0.1, 0, 0))
axes[1, 1].set_title('能量按频率分布')

# 子图6: 重建对比
from scipy.fftpack import idctn


def reconstruct_dct(dct_coeff):
    """从DCT系数重建图像"""
    reconstructed = idctn(dct_coeff, norm='ortho') + 128
    return np.clip(reconstructed, 0, 255)


# 用不同数量的系数重建
def reconstruct_with_n_coeffs(dct_coeff, n):
    """只用前n个最大系数重建"""
    coeff_flat = dct_coeff.flatten()
    abs_coeff = np.abs(coeff_flat)
    threshold = np.sort(abs_coeff)[-n]
    mask = abs_coeff >= threshold
    coeff_filtered = coeff_flat * mask
    return reconstruct_dct(coeff_filtered.reshape(8, 8))


# 用前10个系数重建
recon_10 = reconstruct_with_n_coeffs(dct_coeff, 10)
axes[1, 2].imshow(recon_10, cmap='gray', vmin=0, vmax=255)
axes[1, 2].set_title('用前10个系数重建')
axes[1, 2].axis('off')
axes[1, 2].text(0.5, -0.1, f'保留能量: {np.sum(np.sort(np.abs(dct_coeff.flatten()))[-10:] ** 2) / total_energy:.1%}',
                transform=axes[1, 2].transAxes, ha='center')

plt.suptitle('8×8 DCT变换可视化', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# 5. 打印关键信息
print("=" * 60)
print("DCT分析结果")
print("=" * 60)
print(f"原始图像范围: {np.min(image):.0f} ~ {np.max(image):.0f}")
print(f"DC系数 (0,0): {dct_coeff[0, 0]:.2f}")
print(f"最大AC系数: {np.max(np.abs(dct_coeff[1:])):.2f}")
print(f"高频能量占比: {high_energy / total_energy:.1%}")
print(f"低频能量占比: {low_energy / total_energy:.1%}")
print(f"棋盘格图像的DCT特点: 主要在特定频率有能量")
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import dctn, idctn
# 设置中文字体（避免中文乱码）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建更直观的示例
np.random.seed(42)
test_image = np.random.randint(0, 256, (8, 8), dtype=np.uint8)
test_image_float = test_image.astype(np.float32) - 128

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# 1. 原始图像
axes[0, 0].imshow(test_image, cmap='gray', vmin=0, vmax=255)
axes[0, 0].set_title('1. 原始像素\n(0-255)')
axes[0, 0].axis('off')

# 2. 减去128后
im1 = axes[0, 1].imshow(test_image_float, cmap='RdBu_r', vmin=-128, vmax=127)
axes[0, 1].set_title('2. 减128后\n(-128到127)')
axes[0, 1].axis('off')
plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

# 3. DCT系数
dct_full = dctn(test_image_float, norm='ortho')
im2 = axes[0, 2].imshow(dct_full, cmap='RdBu_r', vmin=-500, vmax=500)
axes[0, 2].set_title('3. DCT系数\n(各种频率的强度)')
axes[0, 2].axis('off')
plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

# 4. 量化后的系数（模拟JPEG压缩）
quant_table = np.array([
    [16, 11, 10, 16,  24,  40,  51,  61],
    [12, 12, 14, 19,  26,  58,  60,  55],
    [14, 13, 16, 24,  40,  57,  69,  56],
    [14, 17, 22, 29,  51,  87,  80,  62],
    [18, 22, 37, 56,  68, 109, 103,  77],
    [24, 35, 55, 64,  81, 104, 113,  92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103,  99]
])

quantized_jpeg = np.round(dct_full / quant_table)
im3 = axes[0, 3].imshow(quantized_jpeg, cmap='RdBu_r', vmin=-30, vmax=30)
axes[0, 3].set_title('4. 量化后系数\n(很多高频变成0)')
axes[0, 3].axis('off')
plt.colorbar(im3, ax=axes[0, 3], fraction=0.046, pad=0.04)

# 在量化图上标记0的位置
zeros_mask = quantized_jpeg == 0
zero_positions = np.argwhere(zeros_mask)
for pos in zero_positions[:10]:  # 只标前10个
    axes[0, 3].scatter(pos[1], pos[0], s=20, c='green', marker='x')

# 5. 反量化
dequantized_jpeg = quantized_jpeg * quant_table
im4 = axes[1, 0].imshow(dequantized_jpeg, cmap='RdBu_r', vmin=-500, vmax=500)
axes[1, 0].set_title('5. 反量化\n(0还是0，其他恢复)')
axes[1, 0].axis('off')
plt.colorbar(im4, ax=axes[1, 0], fraction=0.046, pad=0.04)

# 6. 逆DCT
reconstructed = idctn(dequantized_jpeg, norm='ortho')
im5 = axes[1, 1].imshow(reconstructed, cmap='RdBu_r', vmin=-128, vmax=127)
axes[1, 1].set_title('6. 逆DCT\n(变回像素域，但有误差)')
axes[1, 1].axis('off')
plt.colorbar(im5, ax=axes[1, 1], fraction=0.046, pad=0.04)

# 7. 加回128
final_reconstructed = np.clip(reconstructed + 128, 0, 255)
im6 = axes[1, 2].imshow(final_reconstructed, cmap='gray', vmin=0, vmax=255)
axes[1, 2].set_title('7. 加回128\n(最终JPEG图像)')
axes[1, 2].axis('off')
plt.colorbar(im6, ax=axes[1, 2], fraction=0.046, pad=0.04)

# 8. 误差分析
error_map = np.abs(final_reconstructed - test_image)
im7 = axes[1, 3].imshow(error_map, cmap='hot', vmin=0, vmax=50)
axes[1, 3].set_title('8. 压缩误差\n(丢失高频的代价)')
axes[1, 3].axis('off')
plt.colorbar(im7, ax=axes[1, 3], fraction=0.046, pad=0.04)

plt.suptitle('JPEG压缩全过程：DCT → 量化舍弃高频 → 逆DCT恢复像素',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# 打印关键数据
print("="*60)
print("JPEG压缩效果统计")
print("="*60)
print(f"原始非零DCT系数：{np.sum(dct_full != 0)}")
print(f"量化后非零系数：{np.sum(quantized_jpeg != 0)}")
print(f"压缩率：{np.sum(quantized_jpeg != 0)/64:.1%}")
print(f"最大像素误差：{np.max(error_map):.1f}")
print(f"平均像素误差：{np.mean(error_map):.1f}")
print(f"PSNR（质量指标）：{10*np.log10(255**2/np.mean(error_map**2)):.1f} dB")
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 创建一个简单的测试图像，包含多种频率
image = np.zeros((128, 128))

# 添加不同频率的内容
# 1. 低频：大面积渐变（从黑到白）
for i in range(128):
    image[:, i] = i / 128  # 水平渐变

# 2. 中频：垂直条纹
for i in range(128):
    if i % 16 < 8:  # 每16像素一个周期，宽度8
        image[i, :] += 0.3

# 3. 高频：噪声
image += np.random.randn(128, 128) * 0.05

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('原始图像（包含多种频率）')
plt.colorbar()

# 对图像进行8x8块DCT
from scipy.fftpack import dct


def block_dct_2d(image, block_size=8):
    h, w = image.shape
    h_blocks = h // block_size
    w_blocks = w // block_size

    # 初始化DCT系数矩阵
    dct_coeffs = np.zeros((h_blocks, w_blocks, block_size, block_size))

    for i in range(h_blocks):
        for j in range(w_blocks):
            # 提取8x8块
            block = image[i * block_size:(i + 1) * block_size,
                    j * block_size:(j + 1) * block_size]
            # 对块进行2D DCT（先对行，再对列）
            dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
            dct_coeffs[i, j] = dct_block

    return dct_coeffs


dct_blocks = block_dct_2d(image)

# 取第一个块的DCT系数
first_block_dct = dct_blocks[0, 0]

plt.subplot(1, 2, 2)
plt.imshow(np.log1p(np.abs(first_block_dct)), cmap='hot')
plt.colorbar()
plt.title('单个8x8块的DCT系数（对数尺度）')
plt.xlabel('水平频率索引 u')
plt.ylabel('垂直频率索引 v')

plt.tight_layout()
plt.show()
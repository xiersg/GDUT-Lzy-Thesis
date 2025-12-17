import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体（避免中文乱码）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# 1. 生成8×8 DCT基函数
def generate_dct_basis(N=8):
    """生成N×N DCT的所有基函数"""
    basis_images = np.zeros((N, N, N, N))

    for u in range(N):  # 垂直频率索引
        for v in range(N):  # 水平频率索引
            # 创建一个空图像
            basis = np.zeros((N, N))

            # DCT基函数公式
            for x in range(N):
                for y in range(N):
                    # 归一化常数
                    c_u = np.sqrt(1 / N) if u == 0 else np.sqrt(2 / N)
                    c_v = np.sqrt(1 / N) if v == 0 else np.sqrt(2 / N)

                    # DCT基函数值
                    basis[x, y] = c_u * c_v * \
                                  np.cos(np.pi * u * (2 * x + 1) / (2 * N)) * \
                                  np.cos(np.pi * v * (2 * y + 1) / (2 * N))

            basis_images[u, v] = basis

    return basis_images


# 2. 生成所有64个基函数
N = 8
basis = generate_dct_basis(N)

# 3. 创建可视化图像
fig, axes = plt.subplots(N, N, figsize=(15, 15))

# 设置全局的数值范围，保证颜色一致
vmin, vmax = -0.5, 0.5  # DCT基函数值范围大致在[-0.5, 0.5]

for u in range(N):
    for v in range(N):
        ax = axes[u, v]
        im = ax.imshow(basis[u, v], cmap='RdBu', vmin=vmin, vmax=vmax)
        ax.axis('off')  # 关闭坐标轴

        # 在每个格子左上角添加(u,v)标签
        ax.text(0.05, 0.95, f'({v},{u})',
                transform=ax.transAxes,
                fontsize=10,
                fontweight='bold',
                color='black',
                verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.2',
                          facecolor='white',
                          alpha=0.7,
                          edgecolor='gray'))

        # 特殊标记几个重要的基函数
        if u == 0 and v == 0:
            ax.set_title('DC\n(平均值)', fontsize=9, color='red', pad=5)
        elif u == 0 and v == 1:
            ax.set_title('水平低频', fontsize=8, color='blue', pad=5)
        elif u == 1 and v == 0:
            ax.set_title('垂直低频', fontsize=8, color='green', pad=5)
        elif u == 7 and v == 7:
            ax.set_title('最高频', fontsize=8, color='purple', pad=5)

# 添加行列标签
for i in range(N):
    # 左侧：垂直频率索引
    axes[i, 0].text(-0.2, 0.5, f'v={i}',
                    transform=axes[i, 0].transAxes,
                    fontsize=12, fontweight='bold',
                    color='darkgreen',
                    ha='right', va='center')

    # 底部：水平频率索引
    axes[N - 1, i].text(0.5, -0.1, f'u={i}',
                        transform=axes[N - 1, i].transAxes,
                        fontsize=12, fontweight='bold',
                        color='darkblue',
                        ha='center', va='top')

# 4. 添加整体标题和说明
plt.suptitle('8×8 DCT基函数图像可视化', fontsize=20, fontweight='bold', y=0.95)

# 添加解释文本
fig.text(0.5, 0.02,
         '每个小格子显示一个DCT基函数图像\n'
         'u: 水平频率索引（从左到右频率增加）\n'
         'v: 垂直频率索引（从上到下频率增加）\n'
         '颜色: 红色=正值，蓝色=负值，白色=零值',
         ha='center', fontsize=12, style='italic')

# 调整布局
plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # 为标题留出空间
plt.show()

# 5. 额外：显示几个重要的基函数放大图
print("=" * 60)
print("重要基函数解释：")
print("=" * 60)

important_bases = [
    (0, 0, "DC分量 (u=0, v=0)", "常数图像，代表平均亮度"),
    (0, 1, "水平低频 (u=1, v=0)", "从左到右渐变，水平方向一次变化"),
    (1, 0, "垂直低频 (u=0, v=1)", "从上到下渐变，垂直方向一次变化"),
    (1, 1, "对角低频 (u=1, v=1)", "对角线方向渐变"),
    (0, 7, "水平高频 (u=7, v=0)", "密集水平条纹，水平方向七次变化"),
    (7, 0, "垂直高频 (u=0, v=7)", "密集垂直条纹，垂直方向七次变化"),
    (7, 7, "最高频 (u=7, v=7)", "棋盘格模式，两个方向都七次变化")
]

fig2, axes2 = plt.subplots(2, 4, figsize=(16, 8))

for idx, (v, u, title, desc) in enumerate(important_bases):
    ax = axes2[idx // 4, idx % 4]
    im = ax.imshow(basis[u, v], cmap='RdBu', vmin=vmin, vmax=vmax)
    ax.set_title(f'{title}\n{desc}', fontsize=11, pad=10)
    ax.axis('off')

    # 添加(u,v)标签
    ax.text(0.05, 0.95, f'u={u}, v={v}',
            transform=ax.transAxes,
            fontsize=10,
            fontweight='bold',
            color='black',
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.2',
                      facecolor='white',
                      alpha=0.7))

    # 添加颜色条
    plt.colorbar(im, ax=ax, shrink=0.8)

# 调整布局
plt.suptitle('重要DCT基函数详解', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
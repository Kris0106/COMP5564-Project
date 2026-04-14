import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

# 定义要合并的图片文件列表
files = [
    '/Users/kris/Desktop/COMP5564/Project2/diagnosis_4_signals_AAPL.png',
    '/Users/kris/Desktop/COMP5564/Project2/diagnosis_4_signals_AMZN.png',
    '/Users/kris/Desktop/COMP5564/Project2/diagnosis_4_signals_GOOGL.png',
    '/Users/kris/Desktop/COMP5564/Project2/diagnosis_4_signals_MSFT.png',
    '/Users/kris/Desktop/COMP5564/Project2/diagnosis_4_signals_NFLX.png'
]

# 创建一个 3x2 的网格（共6个位置，最后一个留空或用于其他）
fig, axes = plt.subplots(3, 2, figsize=(20, 15))
axes = axes.flatten()

# 遍历文件并加载图片
for i, file_path in enumerate(files):
    if os.path.exists(file_path):
        try:
            img = mpimg.imread(file_path)
            axes[i].imshow(img)
            axes[i].axis('off')  # 隐藏坐标轴
            axes[i].set_title(os.path.basename(file_path), fontsize=14)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            axes[i].text(0.5, 0.5, f"Error: {e}", ha='center')
    else:
        print(f"File not found: {file_path}")
        axes[i].text(0.5, 0.5, "File Not Found", ha='center')

# 隐藏多余的子图（这里第6个位置是空的）
for j in range(len(files), len(axes)):
    axes[j].axis('off')

plt.tight_layout()
output_path = '/Users/kris/Desktop/COMP5564/Project2/merged_signals_all.png'
plt.savefig(output_path, dpi=150)
print(f"Merged image saved to {output_path}")

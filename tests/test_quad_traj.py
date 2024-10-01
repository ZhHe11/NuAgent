import matplotlib.pyplot as plt
import matplotlib.image as mpimg



path_baseline = "/mnt/nfs2/zhanghe/project001/METRA/exp/Quadruped/baseline-path50sd000_1727681266_dmc_quadruped_SZN"
path_ours = "/mnt/nfs2/zhanghe/project001/METRA/exp/Quadruped/R-dist_sample_z-wo_norm-path50sd000_1727588796_dmc_quadruped_SZN"

for i in range(10000):
    id = i * 20
    img_path_baseline = path_baseline + '/plots/TrajPlot_RandomZ_' + str(id) +  '.png'
    img_path_ours = path_ours + '/plots/TrajPlot_RandomZ_' + str(id) +  '.png'
    img_baseline = mpimg.imread(img_path_baseline)
    img_ours = mpimg.imread(img_path_ours)

    plt.figure(figsize=(6, 3), dpi=300)
    # 显示第一张图像
    plt.subplot(1, 2, 1)  # 1行2列的第1个位置
    plt.imshow(img_baseline)
    plt.title('Baseline')
    plt.axis('off')  # 不显示坐标轴

    # 显示第二张图像
    plt.subplot(1, 2, 2)  # 1行2列的第2个位置
    plt.imshow(img_ours)
    plt.title('Ours')
    plt.axis('off')  # 不显示坐标轴

    # 显示图形
    plt.savefig('tmp.png', dpi=300)
    plt.close()
    a = input()
    
    



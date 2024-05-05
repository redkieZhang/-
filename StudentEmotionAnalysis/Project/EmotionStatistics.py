import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np

# 设置中文显示
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 设置微软雅黑字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

def getEmotionPieImage(allLabels):
    if len(allLabels)==0:
        return None
    enLabels=['angry','disgust','fear','happy','sad','surprise','neutral']
    labels = ['生气', '厌恶','害怕','快乐', '悲伤', '惊讶', '中性']
    allLabels=np.array(allLabels)
    enLabels=np.array(enLabels)
    labels = np.array(labels)
    unique,values=np.unique(allLabels,return_counts=True)
    indexs=np.array([np.where(enLabels==ui)[0][0] for ui in unique])
    labels=labels[indexs]
    colors = sns.color_palette("hsv", n_colors=len(labels))  # 使用Seaborn的bright颜色方案
    fig, ax = plt.subplots(figsize=(9, 9))
    wedges, texts, autotexts = ax.pie(values, labels=labels, colors=colors, autopct=lambda p: '{:.0f}人\n{:.1f}%'.format(p * sum(values) / 100, p), startangle=90,wedgeprops=dict(edgecolor='black', linewidth=1))
    plt.setp(texts, size=15)
    plt.setp(autotexts, size=15, color="black")
    fig.canvas.draw()
    rgb_string = fig.canvas.tostring_rgb()
    width, height = fig.canvas.get_width_height()
    img = np.frombuffer(rgb_string, dtype=np.uint8).reshape((height, width, 3))
    return img
    
if '__name__'=='__main__':
    plt.imshow(getEmotionPieImage(['angry','angry','angry','neutral','neutral','fear','surprise','surprise','sad']))
    plt.show()
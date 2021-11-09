# Part1 内容介绍

在给大家分享知识的过程中，发现很多同学在学习竞赛都存在较多的问题：

* Pandas、Numpy 处理数据不熟悉
* Sklearn、LightGBM 模型使用不熟悉
* 不知道如何构建特征工程
* 不知道如何筛选特征

而上述问题都是一个竞赛选手、一个算法工程师所必备的。因此我们将从本月组织一次竞赛训练营活动，希望能够帮助大家入门数据竞赛。在活动中我们将布置具体竞赛任务，然后参与的同学们不断闯关完成，竟可能的帮助大家入门。

10 月份的竞赛活动将以以下三个比赛展开：

* [结构化赛题：天池新人赛二手车交易价格预测](https://tianchi.aliyun.com/competition/entrance/231784/introduction)
* [CV赛题：科大讯飞新人赛人脸关键点检测挑战赛](https://challenge.xfyun.cn/topic/info?type=key-points-of-human-face&ch=dw-sq-1)
* [NLP赛题：天池新人赛新闻文本分类](https://tianchi.aliyun.com/competition/entrance/531810/introduction)

---


# Part2 活动安排

* 活动是免费学习活动，不会收取任何费用。
* **请各位同学添加下面微信，并回复【结构化、CV、NLP】，即可参与。**

![](https://cdn.coggle.club/coggle666_qrcode.png)


---
# Part3 积分说明和奖励

为了激励各位同学完成的学习任务，将学习任务根据难度进行划分，并根据是否完成进行评分难度高中低的任务分别分数为 3、2 和 1。在完成 10 月学习后，将按照积分顺序进行评选 Top3 的学习者。

|微信昵称|结构化|NLP赛题|CV赛题|总得分|
|:----|:----|:----|:----|:----|
|**Nagaloop**|**23**|**19**|**9**|**51**|
|**云中鹤**|**0**|**25**|**0**|**25**|
|**来来**|**0**|**25**|**0**|**25**|
|**·L·**|**0**|**25**|**0**|**25**|
|TNT|23|0|0|23|
|梳碧湖的砍菜猫|7|7|3|17|
|xX_Albert|14|0|0|14|
|iCLAUDE|0|0|11|11|
|orient|0|0|10|10|
|Murasame|10|0|0|10|
|宏颖.HongyingYUE|10|0|0|10|
|iikuan|0|9|0|9|
|Chenin|0|9|0|9|
|张红旭|0|9|0|9|
|王楠|0|9|0|9|
|邱泽凯|0|9|0|9|
|Rebecca|7|0|0|7|
|cici|7|0|0|7|
|风1886|7|0|0|7|
|曺愨|7|0|0|7|
|不想做咸鱼想努力向上的cpp|7|0|0|7|
|rainnielyt|4|0|0|4|
|MurasameLoryyse|4|0|0|4|
|Jeremy|4|0|0|4|
|莹和|4|0|0|4|
|闲云|4|0|0|4|
|李🥑Yuting|4|0|0|4|
|Pixel doooog|0|3|0|3|
|lmyouxiu|1|1|1|3|
|吴定俊|3|0|0|3|
|202xxx|0|3|0|3|
|Trouvaille|0|0|1|1|
|ly|1|0|0|1|
|karma_corr_hc|1|0|0|1|
|JYH_113|0|0|1|1|
|谁伴我闯荡|0|0|1|1|
|款子|1|0|0|1|
|xJun|0|0|0|0|

Top3 的学习者将获得以下**奖励**：

* Coggle & Datawhale 定制周边
* Coggle 竞赛专访机会
* 《机器学习算法竞赛实战》，鱼佬签名版
* 加入 Datawhale 优秀竞赛选手社群

注：

* Coggle 数据科学保留活动期间和结束后修改奖励和规则的权利。
* 如果有违反竞赛规则的情况，Coggle 数据科学保留取消相关参赛者的参与排名的权利。

---


# Part4  结构化赛题学习

## 赛题介绍

[https://tianchi.aliyun.com/competition/entrance/231784/information](https://tianchi.aliyun.com/competition/entrance/231784/information)

赛题以预测二手车的交易价格为任务，数据集报名后可见并可下载，该数据来自某交易平台的二手车交易记录，总数据量超过 40w，包含 31 列变量信息，其中 15 列为匿名变量。为了保证比赛的公平性，将会从中抽取 15 万条作为训练集，5 万条作为测试集 A，5 万条作为测试集 B，同时会对 name、model、brand 和 regionCode 等信息进行脱敏。

|**Field**|**Description**|
|:----|:----|
|SaleID|交易 ID，唯一编码|
|name|汽车交易名称，已脱敏|
|regDate|汽车注册日期，例如 20160101，2016 年 01 月 01 日|
|model|车型编码，已脱敏|
|brand|汽车品牌，已脱敏|
|bodyType|车身类型：豪华轿车：0，微型车：1，厢型车：2，大巴车：3，敞篷车：4，双门汽车：5，商务车：6，搅拌车：7|
|fuelType|燃油类型：汽油：0，柴油：1，液化石油气：2，天然气：3，混合动力：4，其他：5，电动：6|
|gearbox|变速箱：手动：0，自动：1|
|power|发动机功率：范围 [ 0, 600 ]|
|kilometer|汽车已行驶公里，单位万 km|
|notRepairedDamage|汽车有尚未修复的损坏：是：0，否：1|
|regionCode|地区编码，已脱敏|
|seller|销售方：个体：0，非个体：1|
|offerType|报价类型：提供：0，请求：1|
|creatDate|汽车上线时间，即开始售卖时间|
|price|二手车交易价格（预测目标）|
|v 系列特征|匿名特征，包含 v0-14 在内 15 个匿名特征|

评价标准为 MAE(Mean Absolute Error)。


MAE 越小，说明模型预测得越准确。

## 打卡汇总（满分23）

|任务名称|难度|所需技能|
|:----|:----|:----|
|报名比赛，下载比赛数据集并完成读取|低、1|Pandas|
|对数据字段进行理解，并对特征字段依次进行数据分析|中、2|Matplotlib、Seaborn|
|对标签进行数据分析，并使用 log 进行转换|低、1|Pandas|
|使用特征工程对比赛字段进行编码|高、3|Pandas、Sklearn|
|使用 Sklearn 中基础树模型完成训练和预测|中、2|Sklearn|
|成功将树模型的预测结果文件提交到天池|低、1|Pandas|
|使用 XGBoost 模型完成训练和预测|高、3|XGBoost|
|成功将 XGBoost 的预测结果文件提交到天池|低、1|Pandas|
|使用 LightGBM 中基础树模型完成训练和预测|高、3|LightGBM|
|成功将 LightGBM 的预测结果文件提交到天池|低、1|Pandas|
|对 XGBoost、LightGBM 模型进行调参|中、2|Sklearn、XGBoost、LightGBM|
|使用交叉验证 + Stacking 过程完成模型集成|高、3|模型集成|

## 打卡要求

**注：**

* **需要使用Python环境下Notebook完成以下任务**
* **需要完成所有的任务细节才算完成一个任务**
* **所有的任务可以写在一个Notebook内**

**（10月5日）任务1：报名比赛，下载比赛数据集并完成读取**

- [ ] 登录天池，可使用淘宝 & 支付宝登录，[https://account.aliyun.com/login/login.htm](https://account.aliyun.com/login/login.htm)
- [ ] 下载比赛数据集，[https://tianchi.aliyun.com/competition/entrance/231784/information](https://tianchi.aliyun.com/competition/entrance/231784/information)
- [ ] 配置本地Notebook环境，或使用天池DSW：[https://dsw-dev.data.aliyun.com/#/](https://dsw-dev.data.aliyun.com/#/)
- [ ] 使用Pandas完成数据集读取（其中zip文件需要解压后读取）
    - [ ] used_car_sample_submit.csv
    - [ ] used_car_testB_20200421.zip
    - [ ] used_car_train_20200313.zip
```python
import pandas as pd
Train_data = pd.read_csv('used_car_train_20200313.csv', sep=' ')
Test_data = pd.read_csv('used_car_testA_20200313.csv', sep=' ')
```

**（10月5日）****任务2：****对数据字段进行理解，并对特征字段依次进行数据分析**

- [ ] 使用Pandas对比赛数据集进行分析
    - [ ] 分析每个字段的取值、范围（unique）和类型（dtypes）
    - [ ] 结合[比赛页面](https://tianchi.aliyun.com/competition/entrance/231784/information)中具体字段的含义，对字段的取值分布进行分析
- [ ] 计算特征字段（30个）与标签的相关性
- [ ] 选择特征字段中与标签强相关的3个字段，绘制其余标签的分布关系图

**（10月5日）****任务3：****对标签进行数据分析，并使用 log 进行转换**

- [ ] 使用Pandas对标签字段进行数据分析
- [ ] 使用 log 对标签字段进行转换

**（10月11日）任务4：使用特征工程对比赛字段进行编码**

[https://tianchi.aliyun.com/notebook-ai/detail?postId=95501](https://tianchi.aliyun.com/notebook-ai/detail?postId=95501)

[Feature+Engineering.pdf](https://uploader.shimo.im/f/eYF7qexQskJnREuH.pdf?fileGuid=XkpH6d8pHRgCtgV8)


- [ ] 对数据集中类别字段（取值空间大于2）的进行字段进行onehot操作
- [ ] 对日期特征提取年月日等信息
```python
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')
X = [['Male', 1], ['Female', 3], ['Female', 2]]

enc.fit(X)
enc.transform([['Female', 1], ['Male', 4]]).toarray()
```

```
import pandas as pd
df = pd.DataFrame({'A': ['a', 'b', 'a'], 'B': ['b', 'a', 'c'],
                   'C': [1, 2, 3]})
                   
pd.get_dummies(df, prefix=['col1', 'col2'])
```

**（10月11日）任务5：使用 Sklearn 中基础树模型完成训练和预测**

- [ ] 学会五折交叉验证的数据划分方法（KFold）
```python
import numpy as np
from sklearn.model_selection import KFold
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([1, 2, 3, 4])
kf = KFold(n_splits=2)

for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
```
- [ ] 对标签price按照大小划分成10等分，然后使用StratifiedKFold进行划分
```python
import numpy as np
from sklearn.model_selection import StratifiedKFold
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([0, 0, 1, 1])
skf = StratifiedKFold(n_splits=2)

for train_index, test_index in skf.split(X, y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
```
- [ ] 学会使用sklearn中随机森林模型

**（10月11日）任务6：成功将树模型的预测结果文件提交到天池**

- [ ] 使用StratifiedKFold配合随机森林完成模型的训练和预测
- [ ] 在每折记录下模型对验证集和测试集的预测结果
- [ ] 将多折测试集结果进行求均值，并写入csv提交到天池

**（10月25日）任务7：使用 XGBoost 模型完成训练和预测**

- [ ] 学会XGBoost模型的基础使用
- [ ] 学会XGBoost模型的基础参数理解
- [ ] 学会XGBoost模型的保存和加载

**（10月25日）任务8：成功将 XGBoost 的预测结果文件提交到天池**

- [ ] 使用StratifiedKFold配合XGBoost完成模型的训练和预测
- [ ] 在每折记录下模型对验证集和测试集的预测结果
- [ ] 将多折测试集结果进行求均值，并写入csv提交到天池

**（10月25日）任务9：使用 LightGBM 模型完成训练和预测**

- [ ] 学会LightGBM模型的基础使用
- [ ] 学会LightGBM模型的基础参数理解
- [ ] 学会LightGBM模型的保存和加载

**（10月25日）任务10：成功将 LightGBM 的预测结果文件提交到天池**

- [ ] 使用StratifiedKFold配合LightGBM完成模型的训练和预测
- [ ] 在每折记录下模型对验证集和测试集的预测结果
- [ ] 将多折测试集结果进行求均值，并写入csv提交到天池

**（10月25日）任务11：成功将 LightGBM 的预测结果文件提交到天池**

- [ ] 使用StratifiedKFold配合LightGBM完成模型的训练和预测
- [ ] 在每折记录下模型对验证集和测试集的预测结果
- [ ] 将多折测试集结果进行求均值，并写入csv提交到天池

**（10月25日）任务11：对 XGBoost、LightGBM 模型进行调参**

- [ ] 网格参数搜索、随机搜索参数、贝叶斯搜索参数
- [ ] 使用Optuna完成模型调参

**（10月25日）任务12：使用交叉验证 + Stacking 过程完成模型集成**

- [ ] 使用Stacking完成模型集成
- [ ] 将多折测试集结果进行求均值，并写入csv提交到天池

---


# Part5  CV 赛题学习

## 赛题介绍

[https://challenge.xfyun.cn/topic/info?type=key-points-of-human-face&ch=dw-sq-1](https://challenge.xfyun.cn/topic/info?type=key-points-of-human-face&ch=dw-sq-1)

人脸识别是基于人的面部特征信息进行身份识别的一种生物识别技术，金融和安防是目前人脸识别应用最广泛的两个领域。人脸关键点是人脸识别中的关键技术。人脸关键点检测需要识别出人脸的指定位置坐标，例如眉毛、眼睛、鼻子、嘴巴和脸部轮廓等位置坐标等。

### 
给定人脸图像，找到 4 个人脸关键点，赛题任务可以视为一个关键点检测问题。

* 训练集：5 千张人脸图像，并且给定了具体的人脸关键点标注。
* 测试集：约 2 千张人脸图像，需要选手识别出具体的关键点位置。

本次竞赛的评价标准回归 MAE 进行评价，数值越小性能更优，最高分为 0。评估代码参考：

```plain
from sklearn.metrics import mean_absolute_error
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
mean_absolute_error(y_true, y_pred)
```
 
## 打卡汇总

|任务名称|难度|所需技能|
|:----|:----|:----|
|报名比赛，下载比赛数据集并完成读取|低、1|OpenCV|
|对数据集样本进行可视化，统计关键点坐标分布|中、2|OpenCV、Pandas|
|使用 Sklearn 中全连接网络构建简单的关键点回归模型|中、2|Sklearn|
|使用 Keras 搭建基础的全连接网络，完成关键点回归|中、2|Keras|
|使用 Keras 搭建卷积神经网络，完成关键点回归|高、3|Keras|
|使用 Keras 预训练模型，修改网络结构完成训练|高、3|Keras|
|使用 Pytorch 搭建基础的全连接网络，完成关键点回归|中|Pytorch|
|使用 Pytorch 搭建卷积神经网络，完成关键点回归|高|Pytorch|
|使用 Pytorch 预训练模型，修改网络结构完成训练|高|Pytorch|
|训练多个深度学习模型，完成模型集成|高、3|Pytorch、Keras|
|使用伪标签 或 模型蒸馏完成训练和预测|高、3|Pytorch、Keras|

## 打卡要求

**注：**

* **需要使用Python环境下Notebook完成以下任务**
* **需要完成所有的任务细节才算完成一个任务**
* **所有的任务可以写在一个Notebook内**

**（10月5日）任务1：报名比赛，下载比赛数据集并完成读取**

- [ ] 注册并登陆科大讯飞比赛，[https://challenge.xfyun.cn/topic/info?type=key-points-of-human-face](https://challenge.xfyun.cn/topic/info?type=key-points-of-human-face)
- [ ] 配置本地Notebook环境，或使用天池DSW：[https://dsw-dev.data.aliyun.com/#/](https://dsw-dev.data.aliyun.com/#/)
- [ ] 读取比赛数据集，读取代码参考如下
```python
import pandas as pd
import numpy as np
train_df = pd.read_csv('人脸关键点检测挑战赛_数据集/train.csv')
train_img = np.load('人脸关键点检测挑战赛_数据集/train.npy')
test_img = np.load('人脸关键点检测挑战赛_数据集/test.npy')
```

**（10月5日）任务2：对数据集样本进行可视化，统计关键点坐标分布**

- [ ] 使用opencv 或 matplotlib对人脸进行可视化
- [ ] 统计关键点具体的位置分布规律

**（10月11日）任务3：使用 Sklearn 中全连接网络构建简单的关键点回归模型**

- [ ] 对训练集关键点标注的缺失值进行填充
- [ ] 使用Sklearn中全连接网络构建简单的回归模型，[https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html)

**（10月11日，Keras/Pytorch二选一）任务4：使用 Keras 搭建基础的全连接网络，完成关键点回归**

网络结构（输出层是8，且激活函数用relu）

[人脸关键点检测-TF.ipynb](https://uploader.shimo.im/f/nxa904QDObRYPUq8.ipynb?fileGuid=XkpH6d8pHRgCtgV8)

[人脸关键点检测-Pytorch.ipynb](https://uploader.shimo.im/f/9QoAKTZ8bNvHzXgi.ipynb?fileGuid=XkpH6d8pHRgCtgV8)


- [ ] 导入keras模块，完成基础的MNIST模型训练
```python
import tensorflow as tf
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)
```
- [ ] 自定义模型，模型可以完成关键点回归任务
- [ ] 修改模型的损失函数和评价指标，完成模型训练

**（10月11日，Keras和Pytorch二选一）任务5：使用 Keras 搭建卷积神经网络，完成关键点回归**

- [ ] 搭建3个卷积层，1个全连接层的网络结构
- [ ] 计算一下卷积层和全连接的参数量

**（10月11日，Keras和Pytorch二选一）任务6：使用 Keras 预训练模型，修改网络结构完成训练**

- [ ] 加载ImageNet预训练模型，建议选择ResNet18
- [ ] 修改预训练模型的最终输出全连接层个数，进行训练
- [ ] 在训练过程中，每个epoch结尾加入模型存储的过程

**（10月11日，Keras和Pytorch二选一）任务7：使用 Pytorch 搭建基础的全连接网络，完成关键点回归**

- [ ] 导入Pytorch模块，完成基础的MNIST模型训练
- [ ] 自定义模型，模型可以完成关键点回归任务
- [ ] 修改模型的损失函数和评价指标，完成模型训练

**（10月11日，Keras和Pytorch二选一）任务8：使用 Pytorch 搭建卷积神经网络，完成关键点回归**

- [ ] 搭建3个卷积层，1个全连接层的网络结构
- [ ] 计算一下卷积层和全连接的参数量

**（10月11日，Keras和Pytorch二选一）任务9：使用 Pytorch 预训练模型，修改网络结构完成训练**

- [ ] 加载ImageNet预训练模型，建议选择ResNet18
- [ ] 修改预训练模型的最终输出全连接层个数，进行训练
- [ ] 在训练过程中，每个epoch结尾加入模型存储的过程

**（10月25日）任务10：训练多个深度学习模型，完成模型集成**

- [ ] 训练多个深度学习模型
- [ ] 然后将多个模型的结果进行集成

**（10月25日）任务11：使用伪标签 或 模型蒸馏完成训练和预测**

- [ ] 使用伪标签完成模型的训练和预测
- [ ] 使用模型蒸馏完成模型的训练和预测

---
# Part6 NLP 赛题学习

## 赛题介绍

[https://tianchi.aliyun.com/competition/entrance/531810/introduction](https://tianchi.aliyun.com/competition/entrance/531810/introduction)

赛题以匿名处理后的新闻数据为赛题数据，数据集报名后可见并可下载。赛题数据为新闻文本，并按照字符级别进行匿名处理。整合划分出 14 个候选分类类别：财经、彩票、房产、股票、家居、教育、科技、社会、时尚、时政、体育、星座、游戏、娱乐的文本数据。

赛题数据由以下几个部分构成：训练集 20w 条样本，测试集 A 包括 5w 条样本，测试集 B 包括 5w 条样本。为了预防选手人工标注测试集的情况，我们将比赛数据的文本按照字符级别进行了匿名处理。评价标准为类别 f1_score 的均值，选手提交结果与实际测试集的类别进行对比，结果越大越好。 

处理后的赛题训练数据如下：

|label|text|
|:----|:----|
|6|57 44 66 56 2 3 3 37 5 41 9 57 44 47 45 33 13 63 58 31 17 47 0 1 1 69 26 60 62 15 21 12 49 18 38 20 50 23 57 44 45 33 25 28 47 22 52 35 30 14 24 69 54 7 48 19 11 51 16 43 26 34 53 27 64 8 4 42 36 46 65 69 29 39 15 37 57 44 45 33 69 54 7 25 40 35 30 66 56 47 55 69 61 10 60 42 36 46 65 37 5 41 32 67 6 59 47 0 1 1 68|

在数据集中标签的对应的关系如下：{'科技': 0, '股票': 1, '体育': 2, '娱乐': 3, '时政': 4, '社会': 5, '教育': 6, '财经': 7, '家居': 8, '游戏': 9, '房产': 10, '时尚': 11, '彩票': 12, '星座': 13}

## 打卡汇总

|任务名称|难度|所需技能|
|:----|:----|:----|
|报名比赛，下载比赛数据集并完成读取|低、1|Pandas|
|对数据集字符进行可视化，统计标签和字符分布|中、2|Pandas|
|使用 TFIDF 提取文本特征|中、2|Sklearn|
|使用 TFIDF 特征 和 线性模型完成训练和预测|中、2|Sklearn|
|使用 TFIDF 特征 和 XGBoost 完成训练和预测|中、2|Sklearn、XGBoost|
|学会训练 FastText、Word2Vec 词向量|中、2|FastText、gensim|
|使用 Word2Vec 词向量，搭建 TextCNN 模型训练预测|高、3|Pytorch、Keras|
|使用 Word2Vec 词向量，搭建 BILSTM 模型训练预测|高、3|Pytorch、Keras|
|学会 Bert 基础，transformer 库基础使用|中、2|Pytorch、transformer|
|使用 Bert 在比赛数据集中完成预训练|高、3|Pytorch、transformer|
|使用 Bert 在比赛数据集上完成微调|高、3|Pytorch、transformer|

## 打卡要求

**注：**

* **需要使用Python环境下Notebook完成以下任务**
* **需要完成所有的任务细节才算完成一个任务**
* **所有的任务可以写在一个Notebook内**

**（10月5日）任务1：报名比赛，下载比赛数据集并完成读取**

- [ ] 登录天池，可使用淘宝 & 支付宝登录，[https://account.aliyun.com/login/login.htm](https://account.aliyun.com/login/login.htm)
- [ ] 下载比赛数据集，[https://tianchi.aliyun.com/competition/entrance/531810/introduction](https://tianchi.aliyun.com/competition/entrance/531810/introduction)
- [ ] 读取比赛数据集，读取代码参考如下
```python
import pandas as pd
train_df = pd.read_csv('train_set.csv', sep='\t', nrows=100)
train_df['word'] = train_df['text'].apply(lambda x: len(x.split(' ')))
```

**（10月5日）任务2：对数据集字符进行可视化，统计标签和字符分布**

- [ ] 统计数据集中所有句子所包含字符的平均个数
- [ ] 统计数据集中不同类别下句子平均字符的个数
- [ ] 统计数据集中类别分布的规律
- [ ] 统计数据集中不同类别下句子中最常见的5个字符

**（10月11日）任务3：使用 TFIDF 提取文本特征**

- [ ] 学习TFIDF的原理
- [ ] 学会使用CountVectorizer
- [ ] 学会使用TfidfVectorizer


**（10月11日）任务4：使用 TFIDF 特征 和 线性模型完成训练和预测**

- [ ] 使用TFIDF提取训练集和测试集特征
- [ ] 使用线性模型（LR等）完成模型的训练和预测

**（10月11日）任务5：使用 TFIDF 特征 和 XGBoost完成训练和预测**

- [ ] 使用TFIDF提取训练集和测试集特征
- [ ] 使用XGBoost完成模型的训练和预测

**（10月25日）任务6：学会训练 FastText、Word2Vec 词向量**

- [ ] 学习词向量的基础原理和优点
- [ ] 学会训练FastText词向量
- [ ] 学会训练Word2Vec词向量

**（10月25日）任务7：使用 Word2Vec 词向量，搭建 TextCNN 模型训练预测**

- [ ] 学习TextCNN网络模型结构
- [ ] 学习深度学习中Embeeding层使用
- [ ] 使用深度学习框架（推荐Pytorch）搭建TextCNN，完成训练和预测

**（10月25日）任务8：使用 Word2Vec 词向量，搭建 BILSTM 模型训练预测**

- [ ] 学习BILSTM网络模型结构
- [ ] 使用深度学习框架（推荐Pytorch）搭建BILSTM，完成训练和预测

**（10月25日）任务9：学会 Bert 基础，transformer 库基础使用**

- [ ] 学习Bert基础和使用
- [ ] 学习transformer库的基础使用

**（10月25日）任务10：使用 Bert 在比赛数据集中完成预训练**

- [ ] 学习Bert的pretrain任务
- [ ] 使用Bert在比赛数据集中完成预训练

**（10月25日）任务11：使用 Bert 在比赛数据集上完成微调**

- [ ] 学会Bert的finetune任务
- [ ] 学习 Bert 在比赛数据集上完成微调

---


# Part7 提问&回答

问：具体的活动是怎么安排的？

>有任务，自己先尝试，然后之后会视频演示和讨论。

问：本次活动是收费的吗，最终奖品如何发放？

>活动是免费的，最终奖品按照积分排行Top3进行发放，如果排名有并列都发送奖励。

问：环境和配置是什么？

linux上进行学习，python3


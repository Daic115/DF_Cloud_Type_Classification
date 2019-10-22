# DF分类
模型调用基于[pretrainedmodels](https://github.com/Cadene/pretrained-models.pytorch) 和 [effcientnet](https://github.com/lukemelas/EfficientNet-PyTorch)，模型与其详细信息可在链接中看到；

**单模表现(A榜)：**
**effcientnet-b4**:_0.547_
**senet154**: _0.551_(这个模型比较大，量力而行)
**inceptionresnetv2**: _0.522_

(TTA为左右翻转输出取平均)

se154+effb4 ensemble *0.571*

### Requirements

```angular2html
opencv-python
torchsummary
scikit-learn==0.21.2
albumentations==0.3.3
pytorch>=1.0.0
pretrainedmodels==0.7.4
efficientnet-pytorch==0.4.0
```
### 数据
单标签分类，数据集经过处理，对于多个类别的图片，**使用总体数量最多的作为其类别**。

数据集索引为Json格式，位于`./data/dataset_k1.json`。其中4/5用于训练，1/5用于验证。可以根据json自己划分一下。

测试数据集索引为`./data/dataset_test.json`

### 训练
训练参数在`main.py`中查看，都有相关解释；

部分重要的参数在`untils.py`中，可以检查一下，写得比较粗糙难免有一些错误：**创建优化器**`build_optimizer()`、**初始化模型**`build_cls_model()`*(支持修改relu激活函数为mish,但不建议使用，显存会炸)*、**损失函数**`build_loss_function()`(*单标签就用ce和smooth感觉就足够了*)

### 测试
测试部分`main.py`的`test(option_path,test_image_folder, save_path,model_path,test_info_path)`
**option_path**：训练过程保存了训练参数的json文件，就在模型保存的目录下。
**test_image_folder**：测试图片文件夹。
**save_path**：保存predictions的位置，以dict形式保存，`{'IMAGE_NAME':[OUTPUT_LIST]}`,输出的list是最后fc的输出，长度为类别数，这样保存是为了方便融模型，至于怎么融可自由发挥。
**model_path**：模型位置。多gpu保存的在`resum_load()`里面改参数。


### 可能有用的

**初始学习率寻找**：参考fastai的[lr_finder](https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html)
```
import matplotlib.pyplot as plt
trainer = MultiClsTrainer(opt)
logs = trainer.lr_finder()
plt.plot(logs['lr'][10:-5],logs['loss'][10:-5])
```

**预裁剪**：对于过长的图片，裁剪掉下方的一部分，再进行训练。

**split_weights**：权重衰减不优化再bias上，原理实验等谷歌。

**数据简单EDA**:位于tools文件夹

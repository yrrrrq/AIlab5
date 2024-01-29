# AIlab5
> 本次实验我实现两个模型，结果均为模型一所产生。模型一图像部分采用ResNet生成embedding，文本部分采用LSTM，最后将二者合并起来。
## 实验准备
* 所需库导入
```
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Dense, Flatten, concatenate, LSTM, Dropout, Embedding
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.utils import to_categorical
import os
```
可以通过以下命令行
```
pip install -r requirements.txt
```
将所需库导入。

* 得到训练集格式如下
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>guid</th>
      <th>tag</th>
      <th>text</th>
      <th>img</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2595</th>
      <td>2845</td>
      <td>2</td>
      <td>RT @orrie_yes: Need to calm myself so here's o...</td>
      <td>[[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0,...</td>
    </tr>
    <tr>
      <th>2982</th>
      <td>430</td>
      <td>0</td>
      <td>#ANIMALABUSE #TORONTO #PUPPY #TORTURE WE OFFER...</td>
      <td>[[[0.96862745, 0.96862745, 0.96862745], [0.984...</td>
    </tr>
    <tr>
      <th>246</th>
      <td>4392</td>
      <td>2</td>
      <td>RT @WorIdStarComedy: #TodaysKidsWillNeverKnow ...</td>
      <td>[[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0,...</td>
    </tr>
    <tr>
      <th>862</th>
      <td>4012</td>
      <td>2</td>
      <td>Thank u for your understanding heart and shini...</td>
      <td>[[[0.8156863, 0.67058825, 0.49411765], [0.8039...</td>
    </tr>
    <tr>
      <th>1941</th>
      <td>2379</td>
      <td>1</td>
      <td>RT @theIeansquad: When you rob a black persons...</td>
      <td>[[[0.07058824, 0.07058824, 0.07058824], [0.070...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1557</th>
      <td>4193</td>
      <td>2</td>
      <td>Euro buoyant ahead of Greek vote http://t.co/q...</td>
      <td>[[[0.72156864, 0.6784314, 0.43529412], [0.7254...</td>
    </tr>
    <tr>
      <th>2997</th>
      <td>333</td>
      <td>0</td>
      <td>#trashcomics lmao dick, so incensed. HOW COULD...</td>
      <td>[[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0,...</td>
    </tr>
    <tr>
      <th>3004</th>
      <td>1652</td>
      <td>2</td>
      <td>#February #Winter #Rainy #Stormy #Windy #Tuesd...</td>
      <td>[[[0.4, 0.38039216, 0.30588236], [0.3764706, 0...</td>
    </tr>
    <tr>
      <th>3687</th>
      <td>2609</td>
      <td>2</td>
      <td>RT @neiltyson: I once showed Pluto to Pluto. H...</td>
      <td>[[[0.39215687, 0.4392157, 0.29803923], [0.2431...</td>
    </tr>
    <tr>
      <th>3804</th>
      <td>4146</td>
      <td>0</td>
      <td>1100/45R46 BLEMISHED R1-W LSW 177A/8 TIRE http...</td>
      <td>[[[0.23529412, 0.23137255, 0.22352941], [0.219...</td>
    </tr>
  </tbody>
</table>
<p>3200 rows × 4 columns</p>
</div>

## 代码文件结构
```
|-- requirements.txt # 存放所需库信息
|-- result.ipynb # 代码实现
|-- test_with_label.txt  # 测试集标签预测结果
```

## 结果
> 结果均被写入test_with_label.txt文件中。

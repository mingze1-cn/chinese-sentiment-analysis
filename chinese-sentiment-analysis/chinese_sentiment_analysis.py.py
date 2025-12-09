"""导入库"""
import matplotlib
import matplotlib.pyplot as plt
import warnings
import re
import jieba
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from gensim.models import KeyedVectors

# 强制 TkAgg 后端
matplotlib.use('TkAgg')
# 防止中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
# 忽略警告
warnings.filterwarnings('ignore')

# 导入中文词向量库
cn_model = KeyedVectors.load_word2vec_format('data/embeddings/sgns.zhihu.bigram.bz2', binary=False, unicode_errors='ignore')

# 打印人工智能词向量长度
embedding_dim = len(cn_model['人工智能'])
print(f'人工智能词向量长度：{embedding_dim}')

# 打印饮料和果汁的相似度
print(f'饮料和果汁的相似度：{cn_model.similarity("饮料", "果汁"):.2%}')
# 打印与美丽有关的5个词
print(f'与美丽有关的5个词：{cn_model.most_similar("美丽", topn=5)}')

# 存储文本内容
train_texts_orig = []
# 存储标签内容
train_target = []

# 读取好评样本
with open('data/positive_samples.md', 'r', encoding='utf-8') as f:
    # 获取所有内容
    lines = f.readlines()

    # 遍历所有内容
    for line in lines:
        # 把字符串当 Python 表达式执行
        dic = eval(line)
        # 添加文本内容
        train_texts_orig.append(dic['text'])
        # 添加标签内容
        train_target.append(dic['label'])

# 读取差评样本
with open('data/negative_samples.md', 'r', encoding='utf-8') as f:
    # 获取所有内容
    lines = f.readlines()

    # 遍历所有内容
    for line in lines:
        # 把字符串当 Python 表达式执行
        dic = eval(line)
        # 添加文本内容
        train_texts_orig.append(dic['text'])
        # 添加标签内容
        train_target.append(dic['label'])

# 创建句子列表
train_tokens = []

# 去标点
for text in train_texts_orig:
    text = re.sub('[\s\/\"\'\.\!~@#$%^&*()_+,?]+|[·！￥……（）——+，。？、]+', '', text)

    # 结巴分词
    cut = jieba.cut(text)
    cut_list = [i for i in cut]

    # 遍历分词列表
    for i, word in enumerate(cut_list):
        try:
            # 通过键找到索引
            cut_list[i] = cn_model.key_to_index[word]
        except Exception as e:
            # 不在词向量内设为零
            cut_list[i] = 0
    
    # 结果添加至句子列表
    train_tokens.append(cut_list)

# 获取所有句子的长度
num_tokens = [len(tokens) for tokens in train_tokens]
# 将所有句子长度列表转为数组
num_tokens = np.array(num_tokens)
# 打印所有句子的长度
print(f'所有句子的长度：{len(num_tokens)}')

# 打印所有句子的的最大值
print(f'所有句子的最大值：{np.max(num_tokens)}')
# 打印所有句子的平均值
print(f'所有句子的平均值：{np.mean(num_tokens):.2f}')

# 绘制所有句子的对数分布图
plt.hist(np.log(num_tokens), bins=100)
# 设置标题和横纵标签
plt.title('所有句子的对数分布图')
plt.xlabel('所有句子的数量')
plt.ylabel('所有句子的长度')
# 设置x轴范围
plt.xlim(0, 10)
# 显示图像
plt.show(block=False)
plt.pause(3)
plt.close()

# 计算合理最大值
max_tokens = int(np.mean(num_tokens) + 2 * np.std(num_tokens))
# 打印填充或截断后的保留率
print(f'填充或截断后的保留率：{np.sum(num_tokens < max_tokens) / len(num_tokens):.2%}')

# 创建由索引返回文本的函数
def reverse_tokens(tokens):
    # 初始化字符串
    text = ''

    # 遍历所有索引
    for i in tokens:
        # 判断非零索引
        if i != 0:
            # 通过索引找到键
            text = text + cn_model.index_to_key[i]
        
        else:
            # 将零表示为空格
            text = text + ' '
    
    return text

# 获取去标点后的文本
reverse = reverse_tokens(train_tokens[0])
# 打印原文本和去标点后的文本
print(f'原文本：{train_texts_orig[0]}\n去标点后的文本：{reverse}')

# 设置词汇量
num_word = 50000
# 创建全零矩阵
embedding_matrix = np.zeros((num_word, embedding_dim))

# 遍历词汇量和词向量索引长度的最小值
for i in range(min(num_word, len(cn_model.index_to_key))):
    # 将训练词向量添加至矩阵内
    embedding_matrix[i,:] = cn_model[cn_model.index_to_key[i]]
# 设置矩阵类型
embedding_matrix = embedding_matrix.astype('float32')

# 打印词向量和矩阵索引333的对应总数
print(f'词向量和矩阵索引333的对应总数：{np.sum(cn_model[cn_model.index_to_key[333]] == embedding_matrix[333])}')
# 打印矩阵维度
print(f'矩阵维度：{embedding_matrix.shape}')

# 填充或截断
train_pad = tf.keras.preprocessing.sequence.pad_sequences(train_tokens, maxlen=max_tokens, padding='pre', truncating='pre')
# 超出词汇量部分设为0
train_pad[train_pad >= num_word] = 0
# 打印索引33填充或截断后的维度
print(f'索引33填充或截断后的维度：{train_pad[33].shape}')

# 将标签列表转为数组
train_target = np.array(train_target)
# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(train_pad, train_target, test_size=0.2, random_state=42)

# 构建神经网络模型
model = tf.keras.Sequential()
# 添加嵌入层
model.add(tf.keras.layers.Embedding(num_word, embedding_dim, weights=[embedding_matrix], trainable=False))
# 添加两层LSTM层
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)))
# 添加全连接层
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# 编译模型
model.compile(optimizer='adam', loss='binary_crossentropy', metrics='accuracy')
# 打印模型信息
model.summary()
# 创建检查点回调
path_checkpoint = 'data/sentiment_checkpoint.keras'

# 创建回调函数
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=path_checkpoint, monitor='val_loss', verbose=1, save_weights_only=True, save_best_only=True)

# 尝试加载权重文件
try:
    model.load_weights(path_checkpoint)
except Exception as e:
    print('错误：', e)

# 防止过拟合
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1)
# 调整学习率
lr_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=0, verbose=1, factor=0.1, min_lr=1e-8)

# 创建回调函数列表
callbacks = [
    checkpoint,
    early_stopping,
    lr_reduction
]

# 训练模型
model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_test, y_test), callbacks=callbacks)
# 评估模型
_, test_acc = model.evaluate(x_test, y_test)
print(f'准确率：{test_acc:.2%}')

# 创建预测函数
def predict_sentiment(text):
    # 打印原文本
    print(text)

    # 去标点
    text = re.sub('[\s\/\"\'\.\!~@#$%^&*()_+,?]+|[·！￥……（）——+，。？、]+', '', text)

    # 结巴分词
    cut = jieba.cut(text)
    cut_list = [i for i in cut]

    # 遍历分词列表
    for i, word in enumerate(cut_list):
        try:
            # 通过键找到索引
            cut_list[i] = cn_model.key_to_index[word]

            # 超出词汇量部分设为0
            if cut_list[i] >= 50000:
                cut_list[i] = 0

        except Exception as e:
            # 不在词向量内设为零
            cut_list[i] = 0
    
    # 填充或截断
    tokens_pad = tf.keras.preprocessing.sequence.pad_sequences([cut_list], maxlen=max_tokens, padding='pre', truncating='pre')

    # 预测
    result = model.predict(tokens_pad)
    coef = result[0][0]

    # 打印结果
    label = '正面' if coef > 0.5 else '负面'
    print(f'这是一例{label}评价，输出={coef:.2f}%')

# 测试数据：
test_list = [
    '酒店设施不是新的，服务态度很不好',
    '酒店卫生条件非常不好',
    '床铺非常舒适',
    '房间很凉，不给开暖气',
    '房间很凉爽，空调冷气很足',
    '酒店环境不好，住宿体验很不好',
    '房间隔音不到位',
    '晚上回来发现没有打扫卫生',
    '因为过节所以要我临时加钱，比团购的价格贵'
]

# 遍历测试数据
for text in test_list:
    # 调用预测函数
    predict_sentiment(text)
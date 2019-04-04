# jinntaoor@gmail.com


import numpy as np
from utils import fashion_loader


def load_data(data_path, selected_classes, kind='train'):
    """
    将fashion-MNIST的第selected_classes[0]类和selected_classes[1]类作为分类对象
    :param data_path: of fashion-MNIST的路径
    :param selected_classes: 输入的两个类别(0-9)
    :param kind:
    :return: 训练和测试数据的样本及其标签
    """
    fashion = fashion_loader.load_mnist(data_path, kind=kind)
    print('gdg',fashion[1][0])
    # dic = dict.fromkeys(range(10), list()) 坑
    dic = dict()
    for i in range(10):
        dic[i] = list()
    for (d, l) in zip(fashion[0], fashion[1]):
        dic[l].append(d)
    f_data = np.array(dic[selected_classes[0]]+dic[selected_classes[1]])
    f_label = [1]*len(dic[selected_classes[0]])+[0]*len(dic[selected_classes[1]])

    f_label = np.asarray(f_label)/255.0
    f_label = f_label.reshape((len(f_data), 1))
    return f_data, f_label


def loadData(fileName):
    '''
    加载Mnist数据集
    :param fileName:要加载的数据集路径
    :return: list形式的数据集及标记
    '''
    # 存放数据及标记的list
    dataList = []
    labelList = []
    # 打开文件
    fr = open(fileName, 'r')
    # 将文件按行读取
    for line in fr.readlines():
        # 对每一行数据按切割福','进行切割，返回字段列表
        curLine = line.strip().split(',')

        # Mnsit有0-9是个标记，由于是二分类任务，所以将标记0的作为1，其余为0
        # 验证过<5为1 >5为0时正确率在90%左右，猜测是因为数多了以后，可能不同数的特征较乱，不能有效地计算出一个合理的超平面
        # 查看了一下之前感知机的结果，以5为分界时正确率81，重新修改为0和其余数时正确率98.91%
        # 看来如果样本标签比较杂的话，对于是否能有效地划分超平面确实存在很大影响
        if int(curLine[0]) == 0:
            labelList.append(1)
            dataList.append([int(num) / 255. for num in curLine[1:]])
        elif int(curLine[0]) == 1:
            labelList.append(0)
            dataList.append([int(num) / 255. for num in curLine[1:]])
        # 存放标记
        # [int(num) for num in curLine[1:]] -> 遍历每一行中除了以第一哥元素（标记）外将所有元素转换成int类型
        # [int(num)/255 for num in curLine[1:]] -> 将所有数据除255归一化(非必须步骤，可以不归一化)
        # dataList.append([int(num) / 255 for num in curLine[1:]])
        # dataList.append([int(num) for num in curLine[1:]])

    # 返回data和label
    return dataList, labelList


def sigmoid(z):
    return 1.0/(1+np.exp(-z))


def grad_descent(samples, labels, steps, alpha):
    """
    :param samples: 数据,(12000, 784)
    :param labels: 标签,(12000, 1)
    :param steps: 最大学习步数
    :param alpha: 学习率
    :return: 模型w
    """
    # samples = np.array(samples)
    # labels = np.array(labels)
    # labels = labels.reshape((len(samples), 1))
    m, n = samples.shape
    weights = np.zeros((n, 1))
    for step in range(steps):
        A = np.dot(samples, weights)  # (12000, 1)
        error = sigmoid(A) - labels
        weights -= alpha * np.dot(samples.T, error)
    return weights


def predict(weights, a_sample):
    z = np.dot(a_sample, weights)
    p1 = np.exp(z)/(1+np.exp(z))
    if p1 > 0.5:
        return 1
    return 0


def test(samples, labels, weights):
    # samples = np.array(samples)
    # labels = np.array(labels)
    # labels = labels.reshape((len(samples), 1))
    n, m = samples.shape
    num_error = 0
    ans = []
    for sample, label in zip(samples, labels):
        pred = predict(weights, sample)
        ans.append(pred)
        if label != pred:
            num_error += 1
    print(ans)
    return 1-(num_error/n)


if __name__ == "__main__":
    from sklearn import datasets
    iris = datasets.load_iris()
    import time
    print('loading train samples')
    train_samples, train_labels = load_data('../fashion', [0, 5], kind='train')
    print('loading test samples')
    test_samples, test_labels = load_data('../fashion', [0, 5], kind='t10k')
    # train_samples, train_labels = loadData("../Mnist/mnist_train.csv")
    # test_samples, test_labels = loadData("../Mnist/mnist_test.csv")
    print('traing---')
    start = time.time()
    weights = grad_descent(train_samples, train_labels, steps=500, alpha=0.01)
    print('done training, use time: ', int(time.time()-start))
    print('test')
    acc = test(test_samples, test_labels, weights)
    print('test acc: ', acc)

# IntroML Midterm Review

## **机器学习期中复习 知识点整理**

## 机器学习是什么？

从数据中学习算法。

![传统学习和机器学习](https://christophm.github.io/interpretable-ml-book/images/programing-ml.png)

机器学习涉及到很多行业。

## 机器学习算法的分类？

### 监督式学习（Supervised Learning）

1. 分类（Classification）

    * 目标变量是离散的，是有限多个数中的一个。例：二进制 $y \in \{ 0, 1 \}$
    * 需要**过去的数据**作为训练集

2. 回归（Regression）

    * 预测的目标变量是连续的
    * 可能会有多个预测变量（Predictor）
    * 需要假设某种映射关系，例如线性关系 $y = \beta_0 + \beta_1 x$ 并通过数据来学习获得系数 $\beta_0$ 和 $\beta_1$

### 非监督式学习

* 学习“一般情况下会发生什么”
* 没有输出
* 例：将一些文档分类成三堆文件

### 加强式学习

* 机器需要学习如何与环境互动并采取不同操作来最大化奖励，环境一般是一个闭环系统
* 需要权衡：exploitation（学习过去的操作） vs. exploration（探索新的操作）

### 机器学习种类的导图

![机器学习的种类](https://7wdata.be/wp-content/uploads/2020/06/1FUZS9K4JPqzfXDcC83BQTw.png)

## 机器学习的不同模型

### 简单线性回归（Simple Linear Regression）

#### 一些概念

* $y$：所预测的变量

  * 一些别名：Dependent variable, response variable, target, regressand...

* $x$：拿来做预测的数据

  * 一些别名：predictor, attribute, covariate, regressor...

* 数据：一组点， $(x_i, y_i), \; i = 1, \dots, n$

#### 模型建立

* 假设一个线性关系

$$
y \approx \beta_0 + \beta_1 x
$$

* $\beta_0$ 是截距
* $\beta_1$ 是斜率
* $\beta = (\beta_0, \beta_1)$ 是这个模型的参数

如何判断这个模型是否合适？

#### 线性模型残差

有时 $x$ 并不能特别准确地预测 $y$ 的值，会有跟两个量都无关的因素导致偏差。需要加上一个量 $\epsilon$ ：

$$
y = \beta_0 + \beta_1 x + \epsilon
$$

用 $\hat y$ 表示预测值，$y$ 表示实际值。那么有

$$
\epsilon_i = y_i - \hat{y}_i
$$

表示第i个数据的实际值与预测值的垂直偏差。

#### 如何选择参数 $(\beta_0, \beta_1)$ ？

定义**残差平方和（Residual Sum of Squares）**：

$$
RSS(\beta_0, \beta_1) = \sum_{i=1}^N (y_i - \hat{y}_i)^2
$$

也叫平方残差（Squared Residuals，SSR）或者平方误差和（Sum of Squared Errors, SSE）。

#### **最小二乘解（Lease Squares Solution）**

定义：找到使RSS最小的参数 $(\beta_0, \beta_1)$。

设数据集为 $(x_i, y_i), \ i = 1, \dots, N$

涉及到的计算量：

1. 样本均值（mean）：

$$
\bar{x} = \frac{1}{N} \sum_{i=1}^N x_i, \quad \bar{y} = \frac{1}{N} \sum_{i=1}^N y_i
$$

2. 样本方差（variance）：

$$
s_x^2 = \frac{1}{N} \sum_{i=1}^N (x_i - \bar{x})^2, \quad s_y^2 = \frac{1}{N} \sum_{i=1}^N (y_i - \bar{y})^2
$$

3. 样本标准差（standard deviation）：

$$
s_x = \sqrt{s_x^2}, \quad s_y  = \sqrt{s_y^2}
$$

4. 样本协方差（covariance）：

$$
s_{xy} = \frac{1}{N} \sum_{i=1}^N (x_i - \bar{x})(y_i - \bar{y})
$$

5. 相关系数（correlation coefficient）

$$
r_{xy} = \frac{s_{xy}}{s_x s_y} \in [-1, 1]
$$

最小二乘解：

$$
\beta_1 = \frac{s_{xy}}{s_x^2} = \frac{r_{xy} s_y}{s_x}, \quad \beta_0 = \bar{y} - \beta_1 \bar{x}
$$

#### 
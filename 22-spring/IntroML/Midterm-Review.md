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

#### 决定系数 $R^2$ —— 另一个评判标准

1. 定义

$$
R^2 = r_{xy}^2
$$

2. 评判标准

    * $R^2 = r_{xy}^2 \approx 1$: 线性模型解释力较好，自变量能够较好地决定因变量。
    * $R^2 = r_{xy}^2 \approx 0$: 模型解释力较差，自变量不能很好决定因变量。

3. 符号关系

$$
\beta_1 = \frac{r_{xy} s_y}{s_x} \Rightarrow Sign(\beta_1) = Sign(r_{xy})
$$

也就是说，系数 $\beta_1$ 的符号和 $r_{xy}$ 的符号是相同的。

### 多重线性回归（Multiple Linear Regression）

多重线性回归是简单线性回归的推广，研究一个因变量与多个自变量之间的数量依存关系。 多重线性回归用回归方程描述一个因变量与多个自变量的依存关系，简称多重回归。

#### 和简单线性回归有什么区别？

* $y$：所预测的变量变成了多项式：

$$
y \approx \hat{y} = \beta_0 + \beta_1 x_1 + \cdots + \beta_k x_k
$$

* $x$：由标量变为向量，称为vector of attributes

$$
\boldsymbol{x} = [x_1, x_2, \dots, x_k]
$$

* 数据变为每个样本对应的属性矢量 $\boldsymbol{x}_i$ 和 对应的预测变量的标量值 $y_i$ 的组合 $(\boldsymbol{x}_i, y_i), \ i = 1, 2, \dots, n$

* 模型训练目标：找到最好的参数向量 $\boldsymbol{\beta} = [\beta_0, \beta_1, \cdots, \beta_k]$

#### 模型

$$
\hat{y} = \beta_0 + \beta_1 x_1 + \cdots + \beta_k x_k\
$$

有时也会写作权重-偏差形式（weight-bias version）

$$
\hat{y} = b + w_1 x_1 + \cdots + w_k x_k
$$

两种形式都可以转化为内积形式：

$$
\hat{y} = \beta_0 + \boldsymbol{\beta}_{1:k} \cdot \boldsymbol{x}
$$

$$
\hat{y} = b + \boldsymbol{w} \cdot \boldsymbol{x}
$$

#### 线性回归的矩阵形式

$$
\left[
\begin{matrix}
\hat{y}_1 \\
\hat{y}_2 \\
\vdots \\
\hat{y}_n
\end{matrix}
\right]
=
\left[
\begin{matrix}
1 & x_{11} & \cdots & x_{1k} \\
1 & x_{21} & \cdots & x_{2k} \\
\vdots & \vdots & & \vdots \\
1 & x_{n1} & \cdots & x_{nk}
\end{matrix}
\right]
\left[
\begin{matrix}
\beta_0 \\
\beta_1 \\
\vdots \\
\beta_k
\end{matrix}
\right]
$$

将矩阵 $\boldsymbol{A}$ 记为一个$n \times (k+1)$ 的矩阵，那么线性回归的矩阵形式可以简记为

$$
\hat{\boldsymbol{y}} = \boldsymbol{A \beta}
$$

其中

$$
\boldsymbol{A} = \left[
\begin{matrix}
1 & x_{11} & \cdots & x_{1k} \\
1 & x_{21} & \cdots & x_{2k} \\
\vdots & \vdots & & \vdots \\
1 & x_{n1} & \cdots & x_{nk}
\end{matrix}
\right]
$$

#### 多重线性回归的残差平方和

$$
RSS(\boldsymbol{\beta}) = \sum_{i=1}^N (y_i - \hat{y}_i)^2
$$

注意 $\hat{y}_i$ 是 $\boldsymbol{\beta} = [\beta_0, \beta_1, \cdots, \beta_k]$ 的函数。

我们要找到使 $RSS(\boldsymbol{\beta})$ 最小的 $\boldsymbol{\beta}$。

#### 多重线性回归的最小二乘解

$$
\widehat{\boldsymbol{\beta}} = (\boldsymbol{A}^\intercal \boldsymbol{A})^{-1} \boldsymbol{A}^\intercal \boldsymbol{y}
$$

如何证明：计算 $RSS(\boldsymbol{\beta})$ 的梯度并令其取值为0.

#### 模型评价标准: $R^2$

$$
R^2 = \frac{s_y^2 - MSE}{s_y^2} = 1 - \frac{MSE}{s_y^2}
$$
$$
MSE = \frac{RSS}{N} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2
$$

#### 拓展：独热编码

如果有一个“特征”（Feature）是离散的（或“分类的”， categorical），那么没有办法直接放入方程进行计算，但是可以通过编码将其转换成数字。

例如：预测量 $y$ 为汽车马力，对于汽车品牌特征 $x_1$ 有三种取值：Ford、BMW、GM。假设对于三种品牌，截距分别为 $\beta_0 + \beta_1$, $\beta_0 + \beta_2$, $\beta_0 + \beta_3$。模型可以是：

$$
y = \beta_0 + \beta_1 \phi_1 + \beta_2 \phi_2 + \beta_3 \phi_3 + \beta_4 x_2 + \cdots
$$

其中

$$
\boldsymbol{\phi} = [\phi_0, \phi_1, \phi_2] = \begin{cases}
[1, 0, 0], & \text{Brand = Ford} \\
[0, 1, 0], & \text{Brand = BMW} \\
[0, 0, 1], & \text{Brand = GM}
\end{cases}
$$

## 多项式模型以及模型阶数选择

### 多项式模型

$$
f(x) = \beta_0 + \beta_1 x + \beta_2 x^2 + \cdots + \beta_d x^d
$$

### 如何选择多项式的阶数？

**不能直接计算**不同阶数多项式模型的 $RSS$ 再选其中 $RSS$ 最小的阶数。因为随着阶数增加，模型对训练集的匹配程度变高，$RSS$ 是单调递减的，会导致**过拟合**，对新数据的预测效果会变差。

输出 $MSE$：

$$
MSE_y(\boldsymbol{x}_{test}) = MSE_f(\boldsymbol{x}_{test}) + \sigma_{\epsilon}^2
$$



引进两个变量：

* 偏差：$Bias(\boldsymbol{x}_{test}) = E[f(\boldsymbol{x}_{test}, \widehat{\boldsymbol{\beta}})] - f_0(\boldsymbol{x}_{test})$

* 方差：$Var(\boldsymbol{x}_{test}) = E[f(\boldsymbol{x}_{test}, \widehat{\boldsymbol{\beta}}) - E[f(\boldsymbol{x}_{test}, \widehat{\boldsymbol{\beta}})]]^2$

对于两种参数：

* 阶数较低时：偏差较高，方差较低
* 阶数中等（接近或等于实际阶数）时：没有偏差，方差中等
* 阶数较高时：没有偏差，方差较高

### 模型的 $MSE$

$$
MSE_f(\boldsymbol{x}_{test}) = Bias(\boldsymbol{x}_{test})^2 + Var(\boldsymbol{x}_{test})
$$

### WIP

### 交叉验证-K折交叉验证

将数据集作为一个整体，分为 $K$ 份，每次选取其中 $K-1$ 份训练模型，剩下1份用于验证。$K$ 的典型值是5或10。

验证过程：

1. 对所有阶数的K折，计算平均 $RSS$
2. 计算 $RSS$ 的标准误差：
$$
SE = \frac{RSS_{std}}{\sqrt{K-1}}
$$
3. 选择最优模型阶数

### One SE Rule - 单标准误差法

对于K折中的 $RSS$，设其最低值对应的阶数为 $d_0$, 选择 $d_{opt}$ 作为最终选择阶数，其中 $RSS(d_{opt}) < RSS(d_0) + SE(RSS(d_0))$。也就是说，不选择 $RSS$ 最低的阶数作为最优，而是选择 $RSS$ 距离最小值不超过对应 $SE$ 的最简模型作为最优。

## 特征选择



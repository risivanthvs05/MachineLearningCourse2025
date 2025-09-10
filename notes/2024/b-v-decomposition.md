---
title: 24秋机器学习笔记-03-偏差/方差分解
date: 2024-09-25 15:08:55
tags:
  - 本科课程
  - 机器学习
categories: [笔记, 本科课程, 机器学习]
---

> 本文主要涉及机器学习中的模型选择，以及偏差-方差分解。
> [一键回城](/note-ml2024fall/)。

## 模型选择（Model Selection）

在利用机器学习相关技术解决实际问题的时候，模型的选择是尤为重要的。

一般而言，会将全部数据的 $80\%$ 用于训练（训练集），$10\%$ 用于验证（验证集，validation set），最后 $10\%$ 用于测试（测试集，test set）。

> 如果数据包含时间戳，则应当按照时间顺序划分 Trn Val 和 Test，防止数据泄露问题（前面不应该看到后面）。否则，随机分配即可。

模型选择的原则：

- 如果有现成可用的（hand-out 的）验证集，则应当选择在验证集上表现更好的模型；
- 如果没有，则应当选择**更简单**的模型。
  *奥卡姆剃刀准则*，简单有效原理。


## Bias-Variance Decomposition

考虑在某训练集 $D$ 上训练后，在测试集上的误差是怎么来的。

记号约定；$D$ 为训练集 $\sim P(D)$，$x$ 为样本，$y$ 为标签。

$f(x;D)$ 为训练后，对 $x$ 的 prediction。

Then the error of $x$ (averaged over $P(D)$)：

$$
\mathbb{E}_D[(f(x;D) - y)^2] = \mathbb{E}_D\left[ \left( f(x;D) - \bar{f}(x) + \bar{f}(x) - y \right) ^{2} \right] 
$$

其中，$\bar f(x)$ 是指 $\mathbb{E}_D[f(x;D)]$，即在 $P(D)$ 上 sample 训练集 $D$。

$$
\mathbb{E}_D[(f(x;D) - y)^{2}] = \mathbb{E}_D[(f(x;D) - \bar f(x))^{2}] + \mathbb{E}_D[(\bar f(x) - y)^2] + 2\mathbb{E}_D[(f(x;D) - \bar f(x))(\bar f(x) - y)]
$$

注意到第二项与 $D$ 无关，其即为 $(\bar f(x) - y)^2$。

最后一项也等价于 $2(\bar f(x) - y)^2 \mathbb{E}_D[f(x;D) - \bar f(x)]$。进一步观察到其实际上就是 $0$。（前半截就是 $\bar f(x)$ 的定义）

最终形式为

$$
\mathbb{E}_D[(f(x;D) - \bar f(x))^{2}] + (\bar f(x) - y)^2
$$

考虑其意义。第一项为**方差**，第二项为**偏差**。

- variance：由于**过拟合**某个特定的 $D$ 产生的误差。
  解决方案：增加 $D$ 的大小。
- bias：由于**模型能力不足**以拟合数据产生的误差（见了很多 $D$ 但还是偏离实际值）。
  比如说尝试用线性模型拟合一个二次函数型分布的数据，且每次 sample $D$ 的时候都只能 sample 两个点。
  解决方案：提升模型容量（capability 或者说 expressivity，粗略估计：参数数量，记作 \#parameters）

这二者通常有一个 trade-off：简单模型->underfit->高 bias，而复杂模型->overfit->高 variance。当然，同时增大 $D$ 并增大模型，也可以二者得兼之，还可以加上一些其他的技巧如正则化等。

> 对 Bias-Variance Decom. 的小质疑
> 
> 定义 $\hat{f}(x):= f(x,|D|\to \infty)$，那么 $\bar f$ 是否等价于 $\hat{f}$？
> 
> 考虑极端情况：$\bar f$ 是在重复的小 $D \sim P(D)$  上取平均；$\hat{f}$ 在一个无限大的训练集上训练。这两种模型是截然不同的。
>
> 考虑 $|D| = 2$，$(x,y)\in \{(-1,1),(0,0),(1,1)\}$，且 $P(D)$ 各有 $\frac{1}{3}$ 的概率在集合中取两个元素。
>
> $\bar f(x)=\frac{1}{3}(-x + x + 1) = \frac{1}{3}$，而 $\hat{f}(x) = \arg\min_h 2h^2 + (1-h)^2 = \frac{2}{3}$，后者明显好于前者。
> 
> 所以，$\hat{f}$ 和 $\bar{f}$ 是不太一样的。而 Bias-Variance Decom. 并没有考虑 $\hat{f}$ 的情况。在这个例子里，$\hat{f}$ 比 $\bar{f}$，在某些其他情况下如随机森林，$\bar{f}$ 会更优。
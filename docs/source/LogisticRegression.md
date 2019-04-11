# Chapter. Logistic Regression

### What is logistic regression?

Logistic regression is a statistical analysis method used to predict a data value based on prior observations of a data set. A logistic regression model predicts a dependent data variable by analyzing the relationship between one or more existing independent variables.

逻辑回归是一种统计分析方法，用于根据已有得观察数据来预测未知数据。逻辑回归模型通过分析一个或多个现有自变量之间的关系来预测从属数据变量。

The dependent variable of logistics regression can be two-category or multi-category, but the two-category is more common and easier to explain. So the most common use in practice is the logistics of the two classifications.

逻辑回归的因变量可以是二分类的，也可以是多分类的，但是二分类的更为常用，也更加容易解释。


Logistics regression corresponds to a hidden status p through the function trumpetp = S(ax+b), then determine the value of the dependent
variable according to the size of p and 1-p.The function S here is the Sigmoid function:
                                         S(t)=1/(1+e^(-t)
By changing t to ax+b, you can get the parameter form of the logistic regression model:
                                          P(x;a,b) = 1 / (1 + e^(-ax+b))

logistic回归通过函数S将ax+b对应到一个隐状态p，p = S(ax+b)，然后根据p与1-p的大小决定因变量的值。这里的函数S就是Sigmoid函数:
                                          S(t)=1/(1+e^(-t)
                                      
将t换成ax+b，可以得到逻辑回归模型的参数形式：
                                          P(x;a,b) = 1 / (1 + e^(-ax+b))

![image](https://github.com/SciEvan/TensorFlow.NET/blob/master/docs/source/sigmoid.png)
                                  ###sigmoid函数的图像

By the function of the function S, we can limit the output value to the interval [0, 1],
p(x) can then be used to represent the probability p(y=1|x), the probability that y is divided into 1 group when an x occurs.

通过函数S的作用，我们可以将输出的值限制在区间[0， 1]上，p(x)则可以用来表示概率p(y=1|x)，即当一个x发生时，y被分到1那一组的概率


The full example is [here](https://github.com/SciSharp/TensorFlow.NET/blob/master/test/TensorFlowNET.Examples/LogisticRegression.cs).

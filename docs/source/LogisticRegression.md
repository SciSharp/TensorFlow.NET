# Chapter. Logistic Regression

### What is logistic regression?

Logistic regression is a statistical analysis method used to predict a data value based on prior observations of a data set. A logistic regression model predicts a dependent data variable by analyzing the relationship between one or more existing independent variables.

逻辑回归是一种统计分析方法，用于根据已有得观察数据来预测未知数据。逻辑回归模型通过分析一个或多个现有自变量之间的关系来预测从属数据变量。

The dependent variable of logistics regression can be two-category or multi-category, but the two-category is more common and easier to explain. So the most common use in practice is the logistics of the two classifications. An example used by TensorFlow.NET is a hand-written digit recognition, which is a multi-category.

逻辑回归的因变量可以是二分类的，也可以是多分类的，但是二分类的更为常用，也更加容易解释。 TensorFlow.NET用的例子是一个手写数字识别，它是一个多分类的问题。

Softmax regression allows us to handle ![1557035393445](_static\logistic-regression\1557035393445.png) where K is the number of classes.


The full example is [here](https://github.com/SciSharp/TensorFlow.NET/blob/master/test/TensorFlowNET.Examples/BasicModels/LogisticRegression.cs).

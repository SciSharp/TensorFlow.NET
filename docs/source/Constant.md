# Chapter. Constant

In TensorFlow, a constant is a special Tensor that cannot be modified while the graph is running. Like in a linear model $\tilde{y_i}=\boldsymbol{w}x_i+b$, constant $b$ can be represented as a Constant Tensor. Since the constant is a Tensor, it also has all the data characteristics of Tensor, including:

* value: scalar value or constant list matching the data type defined in TensorFlow;
* dtype: data type;
* shape: dimensions;
* name: constant's name;

在TensorFlow中，常量是一种特殊的Tensor，它在计算图运行的时候，不能被修改。比如在线性模型里$\tilde{y_i}=\boldsymbol{w}x_i+b$, 常数$b$就可以用一个常量来表示。既然常量是一种Tensor，那么它也就具有Tensor的所有数据特性，它包括：

* value: 符合TensorFlow中定义的数据类型的常数值或者常数列表;
* dtype：数据类型;
* shape：常量的形状;
* name：常量的名字;



##### How to create a Constant

TensorFlow provides a handy function to create a Constant. In TF.NET, you can use the same function name `tf.constant` to create it. TF.NET takes the same name as python binding to the API. Naming, although this will make developers who are used to C# naming habits feel uncomfortable, but after careful consideration, I decided to give up the C# convention naming method.

TensorFlow提供了一个很方便的函数用来创建一个Constant, 在TF.NET，可以使用同样的函数名`tf.constant`来创建，TF.NET采取尽可能使用和python binding一样的命名方式来对API命名，虽然这样会让习惯C#命名习惯的开发者感到不舒服，但我经过深思熟虑之后还是决定放弃C#的约定命名方式。

Initialize a scalar constant:

```csharp
var c1 = tf.constant(3); // int
var c2 = tf.constant(1.0f); // float
var c3 = tf.constant(2.0); // double
var c4 = tf.constant("Big Tree"); // string
```

Initialize a constant through ndarray:

```csharp
// dtype=int, shape=(2, 3)
var nd = np.array(new int[][]
{
	new int[]{3, 1, 1},
    new int[]{2, 3, 1}
});
var tensor = tf.constant(nd);
```

##### Dive in Constant

Now let's explore how constant works.

现在让我探究一下`tf.constant`是怎么工作的。



##### Other functions to create a Constant

* tf.zeros
* tf.zeros_like
* tf.ones
* tf.ones_like
* tf.fill
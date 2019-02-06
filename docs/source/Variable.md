# Chapter. Variable

The variables in TensorFlow are mainly used to represent variable parameter values in the machine learning model. Variables can be initialized by the `tf.Variable` function. During the graph computation the variables are modified by other operations. Variables exist in the session, as long as they are in the same session, other computing nodes on the network can access the same variable value. Variables use lazy loading and will only request memory space when they are used.

TensorFlow中变量主要用来表示机器学习模型中的可变参数值，变量通过可以通过`tf.Variable` 类进行初始化。在图运行过程中，通过各种操作对变量进行修改。变量存在于会话当中，只要是在同一个会话里，网络上的其它计算结节都可以访问到相同的变量值。变量采用延迟加载的方式，只有使用的时候才会申请内存空间。

```csharp
var x = tf.Variable(10, name: "x");
using (var session = tf.Session())
{
    session.run(x.initializer);
    var result = session.run(x);
    Console.Write(result); // should be 10
}
```

The above code first creates a variable operation, initializes the variable, then runs the session, and finally gets the result. This code is very simple, but it shows the complete process how TensorFlow operates on variables. When creating a variable, you pass a `tensor` as the initial value to the function `Variable()`. TensorFlow provides a series of operators to initialize the tensor, the initial value is a constant or a random value.

以上代码先创建变量操作，初始化变量，再运行会话，最后得到结果。这段代码非常简单，但是它体现了整个TensorFlow对变量操作的完整流程。当创建一个变量时，你将一个`张量`作为初始值传入函数`Variable()`。TensorFlow提供了一系列操作符来初始化张量，初始值是常量或是随机值。
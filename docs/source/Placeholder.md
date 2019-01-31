# Chapter. Placeholder

In this chapter we will talk about another common data type in TensorFlow: Placeholder. It is a simplified variable that can be passed to the required value by the session when the graph is run, that is, when you build the graph, you don't need to specify the value of that variable, but delay the session to the beginning. In TensorFlow terminology, we then feed data into the graph through these placeholders. The difference between placeholders and constants is that placeholders can specify coefficient values more flexibly without modifying the code that builds the graph. For example, mathematical constants are suitable for Constant, and some model smoothing values can be specified with Placeholder.

这章我们讲一下TensorFlow里的另一种常用数据类型：占位符。它是一种简化的变量，可以在图运行的时候由会话传入所需要的值，就是说你在构建图的时候，不需要具体指定那个变量的值，而是延迟到会话开始的时候以参数的方式从外部传入初始值。占位符和常量的区别是占位符可以更灵活的指定系数值，而不需要修改构建图的代码。比如数学常量就适合用Constant, 有些模型平滑值可以用Placeholder来指定。

```csharp
var x = tf.placeholder(tf.int32);
var y = x * 3;

Python.with<Session>(tf.Session(), sess =>
{
    var result = sess.run(y, feed_dict: new FeedItem[]
    {
        new FeedItem(x, 2)
    });
    // (int)result should be 6;
});
```


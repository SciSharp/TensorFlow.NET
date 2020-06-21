# Chapter. Placeholder

In this chapter we will talk about another common data type in TensorFlow: Placeholder. It is a simplified variable that can be passed to the required value by the session when the graph is run, that is, when you build the graph, you don't need to specify the value of that variable, but delay the session to the beginning. In TensorFlow terminology, we then feed data into the graph through these placeholders. The difference between placeholders and constants is that placeholders can specify coefficient values more flexibly without modifying the code that builds the graph. For example, mathematical constants are suitable for Constant, and some model smoothing values can be specified with Placeholder.



```csharp
var x = tf.placeholder(tf.int32);
var y = x * 3;

using (var sess = tf.Session())
{
    var result = sess.run(y, feed_dict: new FeedItem[]
    {
        new FeedItem(x, 2)
    });
    // (int)result should be 6;
}
```


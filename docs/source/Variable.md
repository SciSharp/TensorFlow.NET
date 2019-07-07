# Chapter. Variable

The variables in TensorFlow are mainly used to represent variable parameter values in the machine learning model. Variables can be initialized by the `tf.Variable` function. During the graph computation the variables are modified by other operations. Variables exist in the session, as long as they are in the same session, other computing nodes on the network can access the same variable value. Variables use lazy loading and will only request memory space when they are used.



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


# Chapter. Graph

TensorFlow uses a **dataflow graph** to represent your computation in terms of the dependencies between individual operations. A graph defines the computation. It doesn't compute anything, it doesn't hold any values, it just defines the operations that you specified in your code.

### Defining the Graph

We define a graph with a variable and three operations: `variable` returns the current value of our variable.   `initialize` assigns the initial value of 31 to that variable. `assign` assigns the new value of 12 to that variable.

```csharp
with<Graph>(tf.Graph().as_default(), graph =>
{
	var variable = tf.Variable(31, name: "tree");
	tf.global_variables_initializer();
	variable.assign(12);
});
```

TF.NET simulate a `with` syntax to manage the Graph lifecycle which will be disposed when the graph instance is no long need. The graph is also what the sessions in the next chapter use when not manually specifying a graph because use invoked the `as_default()`.

A typical graph is looks like below:

![image](../assets/graph_vis_animation.gif)


# Chapter. Session

TensorFlow **session** runs parts of the graph across a set of local and remote devices. A session allows to execute graphs or part of graphs. It allocates resources (on one or more machines) for that and holds the actual values of intermediate results and variables.

### Running Computations in a Session

Let's complete the example in last chapter.

```csharp
with<Graph>(tf.Graph(), graph =>
{
    var variable = tf.Variable(31, name: "tree");
    var init = tf.global_variables_initializer();

    var sess = tf.Session(graph);
    sess.run(init);

    var result = sess.run(variable); // 31

    var assign = variable.assign(12);
    result = sess.run(assign); // 12
});
```


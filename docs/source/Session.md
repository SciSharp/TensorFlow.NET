# Chapter. Session

TensorFlow **session** runs parts of the graph across a set of local and remote devices. A session allows to execute graphs or part of graphs. It allocates resources (on one or more machines) for that and holds the actual values of intermediate results and variables.



### Running Computations in a Session

Let's complete the example in last chapter.  To run any of the operations, we need to create a session for that graph. The session will also allocate memory to store the current value of the variable.



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

The value of our variables is only valid within one session. If we try to get the value in another session. TensorFlow will raise an error of `Attempting to use uninitialized value foo`. Of course, we can use the graph in more than one session, because session copies graph definition to new memory area. We just have to initialize the variables again. The values in the new session will be completely independent from the  previous one.

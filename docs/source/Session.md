# Chapter. Session

TensorFlow **session** runs parts of the graph across a set of local and remote devices. A session allows to execute graphs or part of graphs. It allocates resources (on one or more machines) for that and holds the actual values of intermediate results and variables.

TensorFlow **Session** 运行预定义的计算图，并且支持跨设备运行和分配GPU。Session可以运行整个计算图或者图的一部分，这样做的好处是对开发模型来话很方便，不需要每次都执行整个图。会话还负责当前计算图的内存分配，保留和传递中间结果。

### Running Computations in a Session

Let's complete the example in last chapter.  To run any of the operations, we need to create a session for that graph. The session will also allocate memory to store the current value of the variable.

让我们完成上一章的例子，在那个例子里我们只是定义了一个图的结构。为了运行这个图，我们需要创建一个Session来根据图定义来分配资源运行它。

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

变量值只会在一个Session里有效。如果我们试图从本Session来访问另一个Session创建的变量和值，就会得到一个`变量未初始化`的错误提示。当然，我们能从多个Session运行同一个计算图，因为计算图只是一个定义，Session初始化的时候会复制整图的定义到新的内存空间里。所以每个Session里的变量值是互相隔离的。
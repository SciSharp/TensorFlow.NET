# Chapter. Tensor

### Represents one of the outputs of an Operation



##### What is Tensor?

Tensor holds a multi-dimensional array of elements of a single data type which is very similar with numpy's ndarray. When the dimension is zero, it can be called a scalar. When the dimension is 2, it can be called a matrix. When the dimension is greater than 2, it is usually called a tensor. If you are very familiar with numpy, then understanding Tensor will be quite easy.

Tensor是一个具有单一数据类型的多维数组容器，当维度为零时，可以称之为标量，当维度为2时，可以称之为矩阵，当维度大于2时，通常称之为张量。Tensor的数据结构非常类似于numpy里的ndarray。如果你对numpy非常熟悉的话，那么对Tensor的理解会相当容易。



##### How to create a Tensor?

There are many ways to initialize a Tensor object in TF.NET. It can be initialized from a scalar, string, matrix or tensor.

在TF.NET中有很多种方式可以初始化一个Tensor对象。它可以从一个标量，字符串，矩阵或张量来初始化。

```csharp
// Create a tensor holds a scalar value
var t1 = new Tensor(3);

// Init from a string
var t2 = new Tensor("Hello! TensorFlow.NET");

// Tensor holds a ndarray
var nd = new NDArray(new int[]{3, 1, 1, 2});
var t3 = new Tensor(nd);

Console.WriteLine($"t1: {t1}, t2: {t2}, t3: {t3}");
```



##### Data Structure of Tensor





TF uses column major order. If we use NumSharp to generate a 2 x 3 matrix, if we access the data from 0 to 5 in order, we won't get a number of 1-6, but we get the order of 1, 4, 2, 5, 3, 6. a set of numbers.

TF 采用的是按列存储模式，如果我们用NumSharp产生一个2 X 3的矩阵，如果按顺序从0到5访问数据的话，是不会得到1-6的数字的，而是得到1，4， 2， 5， 3， 6这个顺序的一组数字。

```cs
// Generate a matrix:[[1, 2, 3], [4, 5, 6]]
var nd = np.array(1f, 2f, 3f, 4f, 5f, 6f).reshape(2, 3);
// The index will be   0  2  4    1  3  5, it's column-major order.
```



![column-major order](_static/column-major-order.png)

![row-major order](_static/row-major-order.png)

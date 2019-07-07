# Chapter. Constant

In TensorFlow, a constant is a special Tensor that cannot be modified while the graph is running. Like in a linear model $\tilde{y_i}=\boldsymbol{w}x_i+b$, constant $b$ can be represented as a Constant Tensor. Since the constant is a Tensor, it also has all the data characteristics of Tensor, including:

* value: scalar value or constant list matching the data type defined in TensorFlow;
* dtype: data type;
* shape: dimensions;
* name: constant's name;



##### How to create a Constant

TensorFlow provides a handy function to create a Constant. In TF.NET, you can use the same function name `tf.constant` to create it. TF.NET takes the same name as python binding to the API. Naming, although this will make developers who are used to C# naming habits feel uncomfortable, but after careful consideration, I decided to give up the C# convention naming method.

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

Now let's explore how `constant` works.



##### Other functions to create a Constant

* tf.zeros
* tf.zeros_like
* tf.ones
* tf.ones_like
* tf.fill
using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using Tensorflow;
using static Keras.Keras;
using Keras;
using NumSharp;
using Tensorflow.Operations.Activation;

namespace Keras.Layers
{
    public class Dense : ILayer
    {
        RefVariable W;
        int units;
        TensorShape WShape;
        string name;
        IActivation activation;

        public Dense(int units, string name = null, IActivation activation = null)
        {
            this.activation = activation;
            this.units = units;
            this.name = (string.IsNullOrEmpty(name) || string.IsNullOrWhiteSpace(name))?this.GetType().Name + "_" + this.GetType().GUID:name;
        }
        public ILayer __build__(TensorShape input_shape, int seed = 1, float stddev = -1f)
        {
            Console.WriteLine("Building Layer \"" + name + "\" ...");
            if (stddev == -1)
                stddev = (float)(1 / Math.Sqrt(2));
            var dim = input_shape.Dimensions;
            var input_dim = dim[dim.Length - 1];
            W = tf.Variable(create_tensor(new int[] { input_dim, units }, seed: seed, stddev: (float)stddev));
            WShape = new TensorShape(W.shape);
            return this;
        }
        public Tensor __call__(Tensor x)
        {
            var dot = tf.matmul(x, W);
            if (this.activation != null)
                dot = activation.Activate(dot);
            Console.WriteLine("Calling Layer \"" + name + "(" + np.array(dot.GetShape().Dimensions).ToString() + ")\" ...");
            return dot;
        }
        public TensorShape __shape__()
        {
            return WShape;
        }
        public TensorShape output_shape(TensorShape input_shape)
        {
            var output_shape = input_shape.Dimensions;
            output_shape[output_shape.Length - 1] = units;
            return new TensorShape(output_shape);
        }
    }
}

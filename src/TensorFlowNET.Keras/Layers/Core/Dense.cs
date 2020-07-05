/*****************************************************************************
   Copyright 2018 The TensorFlow.NET Authors. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
******************************************************************************/

using System;
using Tensorflow;
using static Keras.Keras;
using NumSharp;
using Tensorflow.Operations.Activation;
using static Tensorflow.Binding;

namespace Keras.Layers
{
    public class Dense : Layer
    {
        RefVariable W;
        int units;
        TensorShape WShape;
#pragma warning disable CS0108 // Member hides inherited member; missing new keyword
        string name;
#pragma warning restore CS0108 // Member hides inherited member; missing new keyword
        IActivation activation;

        public Dense(int units, string name = null, IActivation activation = null)
        {
            this.activation = activation;
            this.units = units;
            this.name = (string.IsNullOrEmpty(name) || string.IsNullOrWhiteSpace(name))?this.GetType().Name + "_" + this.GetType().GUID:name;
        }
        public Layer __build__(TensorShape input_shape, int seed = 1, float stddev = -1f)
        {
            Console.WriteLine("Building Layer \"" + name + "\" ...");
            if (stddev == -1)
                stddev = (float)(1 / Math.Sqrt(2));
            var dim = input_shape.dims;
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
            Console.WriteLine("Calling Layer \"" + name + "(" + np.array(dot.TensorShape.dims).ToString() + ")\" ...");
            return dot;
        }
        public TensorShape __shape__()
        {
            return WShape;
        }
#pragma warning disable CS0108 // Member hides inherited member; missing new keyword
        public TensorShape output_shape(TensorShape input_shape)
#pragma warning restore CS0108 // Member hides inherited member; missing new keyword
        {
            var output_shape = input_shape.dims;
            output_shape[output_shape.Length - 1] = units;
            return new TensorShape(output_shape);
        }
    }
}

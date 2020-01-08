using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.Applications
{
    public class Densenet
    {
        public static Tensor dense_block(Tensor x, int blocks, string name) => throw new NotImplementedException();

        public static Tensor transition_block(Tensor x, float reduction, string name) => throw new NotImplementedException();

        public static Tensor conv_block(Tensor x, float growth_rate, string name) => throw new NotImplementedException();

        public static Model DenseNet(int blocks, bool include_top=true, string weights = "imagenet",
                                    Tensor input_tensor = null, TensorShape input_shape = null, 
                                    string pooling = null, int classes = 1000) => throw new NotImplementedException();
    }
}

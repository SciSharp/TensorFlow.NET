using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.Applications
{
    public class Vgg19
    {
        public static Model VGG19(bool include_top = true, string weights = "imagenet",
                                Tensor input_tensor = null, TensorShape input_shape = null,
                                string pooling = null, int classes = 1000) => throw new NotImplementedException();

        public static Tensor preprocess_input(Tensor x, string data_format = null) => throw new NotImplementedException();

        public static Tensor decode_predictions(Tensor preds, int top = 5) => throw new NotImplementedException();
    }
}

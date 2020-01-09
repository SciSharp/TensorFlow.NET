using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.Applications
{
    public class InceptionV3
    {
        public static Model Inceptionv3(bool include_top = true, string weights = "imagenet",
                                  Tensor input_tensor = null, TensorShape input_shape = null,
                                  string pooling = null, int classes = 1000) => throw new NotImplementedException();

        public static Tensor conv2d_bn(Tensor x, int filters, int num_row, int num_col, string padding = "same", (int, int)? strides = null, string name = null) => throw new NotImplementedException();

        public static Tensor preprocess_input(Tensor x, string data_format = null) => throw new NotImplementedException();

        public static Tensor decode_predictions(Tensor preds, int top = 5) => throw new NotImplementedException();
    }
}

using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.Applications
{
    public class Mobilenet
    {
        public static Model MobileNet(TensorShape input_shape= null, float alpha= 1.0f, int depth_multiplier= 1, float dropout= 1e-3f,
                                    bool include_top= true, string weights= "imagenet", Tensor input_tensor= null, string pooling= null, int classes= 1000) => throw new NotImplementedException();

        public static Tensor conv2d_bn(Tensor x, int filters, float alpha, (int, int)? kernel = null, (int, int)? strides = null) => throw new NotImplementedException();

        public static Tensor preprocess_input(Tensor x, string data_format = null) => throw new NotImplementedException();

        public static Tensor decode_predictions(Tensor preds, int top = 5) => throw new NotImplementedException();
    }
}

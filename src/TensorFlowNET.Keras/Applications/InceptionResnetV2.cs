using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.Applications
{
    public class InceptionResnetV2
    {
        public static Model InceptionResNetV2(bool include_top = true, string weights = "imagenet",
                                   Tensor input_tensor = null, TensorShape input_shape = null,
                                   string pooling = null, int classes = 1000) => throw new NotImplementedException();

        public static Tensor conv2d_bn(Tensor x, int filters, (int, int) kernel_size, (int, int) strides, string padding= "same",
                                        string activation= "relu", bool use_bias= false, string name= null) => throw new NotImplementedException();

        public static Tensor inception_resnet_block(Tensor x, float scale, string block_type, int block_idx, string activation= "relu") => throw new NotImplementedException();

        public static Tensor preprocess_input(Tensor x, string data_format = null) => throw new NotImplementedException();

        public static Tensor decode_predictions(Tensor preds, int top = 5) => throw new NotImplementedException();
    }
}

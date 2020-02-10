using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.Applications
{
    public class BlockArg
    {

    }

    public class Efficientnet
    {
        public static Model EfficientNet(float width_coefficient, float depth_coefficient, int default_size, float dropout_rate = 0.2f,
                                        float drop_connect_rate = 0.2f, int depth_divisor = 8, string activation = "swish",
                                        BlockArg[] blocks_args = null, string model_name = "efficientnet", bool include_top = true,
                                        string weights = "imagenet", Tensor input_tensor = null, TensorShape input_shape = null, 
                                        string pooling = null, int classes = 1000) => throw new NotImplementedException();

        public static Tensor block(Tensor inputs, string activation= "swish", float drop_rate= 0f,string name= "",
                                    int filters_in= 32, int filters_out= 16, int kernel_size= 3, int strides= 1,
                                    int expand_ratio= 1, float se_ratio= 0, bool  id_skip= true) => throw new NotImplementedException();

        public static Model EfficientNetB0(bool include_top = true, string weights = "imagenet",
                                    Tensor input_tensor = null, TensorShape input_shape = null,
                                    string pooling = null, int classes = 1000) => throw new NotImplementedException();

        public static Model EfficientNetB1(bool include_top = true, string weights = "imagenet",
                                  Tensor input_tensor = null, TensorShape input_shape = null,
                                  string pooling = null, int classes = 1000) => throw new NotImplementedException();

        public static Model EfficientNetB2(bool include_top = true, string weights = "imagenet",
                                  Tensor input_tensor = null, TensorShape input_shape = null,
                                  string pooling = null, int classes = 1000) => throw new NotImplementedException();

        public static Model EfficientNetB3(bool include_top = true, string weights = "imagenet",
                                  Tensor input_tensor = null, TensorShape input_shape = null,
                                  string pooling = null, int classes = 1000) => throw new NotImplementedException();

        public static Model EfficientNetB4(bool include_top = true, string weights = "imagenet",
                                 Tensor input_tensor = null, TensorShape input_shape = null,
                                 string pooling = null, int classes = 1000) => throw new NotImplementedException();

        public static Model EfficientNetB5(bool include_top = true, string weights = "imagenet",
                                 Tensor input_tensor = null, TensorShape input_shape = null,
                                 string pooling = null, int classes = 1000) => throw new NotImplementedException();

        public static Model EfficientNetB6(bool include_top = true, string weights = "imagenet",
                                 Tensor input_tensor = null, TensorShape input_shape = null,
                                 string pooling = null, int classes = 1000) => throw new NotImplementedException();

        public static Model EfficientNetB7(bool include_top = true, string weights = "imagenet",
                                 Tensor input_tensor = null, TensorShape input_shape = null,
                                 string pooling = null, int classes = 1000) => throw new NotImplementedException();

        public static Tensor preprocess_input(Tensor x, string data_format = null) => throw new NotImplementedException();

        public static Tensor decode_predictions(Tensor preds, int top = 5) => throw new NotImplementedException();
    }
}

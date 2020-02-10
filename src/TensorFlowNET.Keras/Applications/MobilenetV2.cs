using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.Applications
{
    public class MobilenetV2
    {
        public static Model MobileNetV2(TensorShape input_shape = null, float alpha = 1.0f, bool include_top = true, 
                                        string weights = "imagenet", Tensor input_tensor = null, string pooling = null,
                                        int classes = 1000) => throw new NotImplementedException();

        public static Tensor _inverted_res_block(Tensor inputs, int expansion, (int, int) stride, float alpha, int filters, string block_id) => throw new NotImplementedException();

        public static Tensor _make_divisible(Tensor v, Tensor divisor, Tensor min_value= null) => throw new NotImplementedException();

        public static Tensor preprocess_input(Tensor x, string data_format = null) => throw new NotImplementedException();

        public static Tensor decode_predictions(Tensor preds, int top = 5) => throw new NotImplementedException();
    }
}

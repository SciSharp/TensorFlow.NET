using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.Applications
{
    public class Nasnet
    {
        public static Model NASNet(TensorShape input_shape = null, int penultimate_filters = 4032, int num_blocks = 6, int stem_block_filters = 96,
                                bool skip_reduction = true, int filter_multiplier = 2, bool include_top = true, string weights = null,
                                Tensor input_tensor = null, string pooling = null, int classes = 1000, int? default_size = null) => throw new NotImplementedException();

        public static Model NASNetMobile(TensorShape input_shape = null, bool include_top = true, string weights = "imagenet",
                                    Tensor input_tensor = null, string pooling = null, int classes = 1000) => throw new NotImplementedException();

        public static Model NASNetLarge(TensorShape input_shape = null, bool include_top = true, string weights = "imagenet",
                                    Tensor input_tensor = null, string pooling = null, int classes = 1000) => throw new NotImplementedException();

        public static Tensor _separable_conv_block(Tensor ip, int filters, (int, int)? kernel_size= null, (int, int)? strides= null, string block_id= null) => throw new NotImplementedException();

        public static Tensor _adjust_block(Tensor p, Tensor ip, int filters, string block_id= null) => throw new NotImplementedException();

        public static Tensor _normal_a_cell(Tensor p, Tensor ip, int filters, string block_id = null) => throw new NotImplementedException();

        public static Tensor _reduction_a_cell(Tensor p, Tensor ip, int filters, string block_id = null) => throw new NotImplementedException();

        public static Tensor preprocess_input(Tensor x, string data_format = null) => throw new NotImplementedException();

        public static Tensor decode_predictions(Tensor preds, int top = 5) => throw new NotImplementedException();
    }
}

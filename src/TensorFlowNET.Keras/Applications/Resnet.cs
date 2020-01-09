using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.Applications
{
    public class Resnet
    {
        public static Model ResNet(Func<Tensor, Tensor> stack_fn, bool preact, bool use_bias, string model_name= "resnet", bool include_top= true,
                                    string weights= "imagenet", Tensor input_tensor= null, TensorShape input_shape= null, string pooling= null,
                                    int classes= 1000) => throw new NotImplementedException();

        public static Tensor block1(Tensor x, int filters, int kernel_size= 3, int stride= 1, bool conv_shortcut= true, string name= null) => throw new NotImplementedException();

        public static Tensor stack1(Tensor x, int filters, int blocks, int stride1 = 2, string name = null) => throw new NotImplementedException();

        public static Tensor block2(Tensor x, int filters, int kernel_size = 3, int stride = 1, bool conv_shortcut = true, string name = null) => throw new NotImplementedException();

        public static Tensor stack2(Tensor x, int filters, int blocks, int stride1 = 2, string name = null) => throw new NotImplementedException();

        public static Tensor block3(Tensor x, int filters, int kernel_size = 3, int stride = 1, int groups = 32, bool conv_shortcut = true, string name = null) => throw new NotImplementedException();

        public static Tensor stack3(Tensor x, int filters, int blocks, int stride1 = 2, int groups = 32, string name = null) => throw new NotImplementedException();

        public static Model ResNet50(bool include_top = true, string weights = "imagenet",
                                  Tensor input_tensor = null, TensorShape input_shape = null,
                                  string pooling = null, int classes = 1000) => throw new NotImplementedException();

        public static Model ResNet101(bool include_top = true, string weights = "imagenet",
                                  Tensor input_tensor = null, TensorShape input_shape = null,
                                  string pooling = null, int classes = 1000) => throw new NotImplementedException();

        public static Model ResNet152(bool include_top = true, string weights = "imagenet",
                                  Tensor input_tensor = null, TensorShape input_shape = null,
                                  string pooling = null, int classes = 1000) => throw new NotImplementedException();

        public static Tensor preprocess_input(Tensor x, string data_format = null) => throw new NotImplementedException();

        public static Tensor decode_predictions(Tensor preds, int top = 5) => throw new NotImplementedException();
    }
}

using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.Applications
{
    public class ImagenetUtils
    {
        public static Tensor preprocess_input(Tensor x, string data_format= null, string mode= "caffe") => throw new NotImplementedException();
        
        public static Tensor decode_predictions(Tensor preds, int top= 5) => throw new NotImplementedException();

        public static Tensor _preprocess_numpy_input(Tensor x, string data_format, string mode) => throw new NotImplementedException();

        public static Tensor _preprocess_symbolic_input(Tensor x, string data_format, string mode) => throw new NotImplementedException();

        public static TensorShape obtain_input_shape(TensorShape input_shape, int default_size, int min_size,
                       string data_format, bool require_flatten, string weights= null) => throw new NotImplementedException();

        public static ((int, int), (int, int)) correct_pad(Tensor inputs, (int, int) kernel_size) => throw new NotImplementedException();
    }
}

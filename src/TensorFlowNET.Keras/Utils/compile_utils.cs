using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Framework.Models;
using Tensorflow.Util;

namespace Tensorflow.Keras.Utils
{
    internal static class compile_utils
    {
        public static List<string> create_pseudo_input_names(TensorSpec inputs)
        {
            return _create_pseudo_names(inputs, "input_");
        }

        private static List<string> _create_pseudo_names(TensorSpec tensors, string prefix)
        {
            // TODO(Rinne): align with tensorflow
            return new List<string>() { $"{prefix}1" };
        }
    }
}

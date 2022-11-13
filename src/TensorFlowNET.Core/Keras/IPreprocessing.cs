using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras
{
    public interface IPreprocessing
    {
        public ILayer Resizing(int height, int width, string interpolation = "bilinear");
        public ILayer TextVectorization(Func<Tensor, Tensor> standardize = null,
            string split = "whitespace",
            int max_tokens = -1,
            string output_mode = "int",
            int output_sequence_length = -1);
    }
}

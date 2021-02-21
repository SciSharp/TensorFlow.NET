using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.Layers
{
    /// <summary>
    /// Maps strings from a vocabulary to integer indices.
    /// </summary>
    class StringLookup : IndexLookup
    {
        public StringLookup(int max_tokens = -1,
            int num_oov_indices = 1,
            string mask_token = "",
            string[] vocabulary = null,
            string oov_token = "[UNK]",
            string encoding = "utf-8",
            bool invert = false)
        {

        }
    }
}

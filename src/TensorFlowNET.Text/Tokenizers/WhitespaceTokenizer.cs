using NumSharp;
using System;
using System.Collections.Generic;
using System.Text;
using static Tensorflow.Binding;

namespace Tensorflow.Text.Tokenizers
{
    public class WhitespaceTokenizer : ITokenizer
    {
        /// <summary>
        /// Tokenizes a tensor of UTF-8 strings on whitespaces.
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public Tensor tokenize(Tensor input)
        {
            tokenize_with_offsets(input);
            throw new NotImplementedException("");
        }

        Tensor[] tokenize_with_offsets(Tensor input)
        {
            tf_with(ops.name_scope(null, "WhitespaceTokenize"), scope =>
            {
                _whitespace_tokenize_with_offsets_encode_decode_wrapper(input);
            });
            throw new NotImplementedException("");
        }

        Tensor _whitespace_tokenize_with_offsets_encode_decode_wrapper(Tensor input_tensor)
        {
            // Decode the strings and get byte offsets
            var (codepoints, byte_start_offsets) = tf.strings.unicode_decode_with_offsets(input_tensor, "UTF-8");
            var byte_end_offsets = array_ops.concat(new Tensor[] 
            {
                byte_start_offsets[Slice.All, new Slice(1)],
                math_ops.cast(
                    array_ops.expand_dims(tf.strings.string_length(input_tensor), 1),
                    dtypes.int64)
            }, 1);
            return input_tensor;
        }
    }
}

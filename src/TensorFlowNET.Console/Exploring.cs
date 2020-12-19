using System;
using System.Collections.Generic;
using System.Text;
using static Tensorflow.Binding;
using static Tensorflow.TextApi;

namespace Tensorflow
{
    public class Exploring
    {
        public void Run()
        {
            var docs = tf.constant(new[] { "Everything not saved will be lost." });
            var tokenizer = text.WhitespaceTokenizer();
            text.wordshape(docs, Text.WordShape.HAS_TITLE_CASE);

            throw new NotImplementedException("");
        }
    }
}

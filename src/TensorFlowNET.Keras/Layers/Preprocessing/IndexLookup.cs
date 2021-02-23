using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;

namespace Tensorflow.Keras.Layers
{
    public class IndexLookup : CombinerPreprocessingLayer
    {
        public IndexLookup(int max_tokens = -1,
            int num_oov_indices = 1,
            string mask_token = "",
            string oov_token = "[UNK]",
            string encoding = "utf-8",
            bool invert = false) : base(new PreprocessingLayerArgs())
        {
            var num_mask_tokens = mask_token == null ? 0 : 1;
            var vocab_size = max_tokens - (num_oov_indices + num_mask_tokens);
            combiner = new IndexLookupCombiner(vocab_size, mask_token);
        }

        public override void adapt(IDatasetV2 data, bool reset_state = true)
        {
            if (!reset_state)
                throw new ValueError("IndexLookup does not support streaming adapts.");
            base.adapt(data, reset_state);
        }
    }
}

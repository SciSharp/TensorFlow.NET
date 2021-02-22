using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using static Tensorflow.Binding;

namespace Tensorflow.Keras.Layers
{
    public class TextVectorization : CombinerPreprocessingLayer
    {
        TextVectorizationArgs args;
        IndexLookup _index_lookup_layer;

        public TextVectorization(TextVectorizationArgs args)
            : base(args)
        {
            this.args = args;
            args.DType = TF_DataType.TF_STRING;
            // string standardize = "lower_and_strip_punctuation",

            var mask_token = args.OutputMode == "int" ? "" : null;
            _index_lookup_layer = new StringLookup(max_tokens: args.MaxTokens,
                mask_token: mask_token,
                vocabulary: args.Vocabulary);
        }

        /// <summary>
        /// Fits the state of the preprocessing layer to the dataset.
        /// </summary>
        /// <param name="data"></param>
        /// <param name="reset_state"></param>
        public override void adapt(IDatasetV2 data, bool reset_state = true)
        {
            var shape = data.output_shapes[0];
            if (shape.rank == 1)
                data = data.map(tensor => array_ops.expand_dims(tensor, -1));
            build(data.variant_tensor);
            var preprocessed_inputs = data.map(_preprocess);
            _index_lookup_layer.adapt(preprocessed_inputs);
        }

        protected override void build(Tensors inputs)
        {
            base.build(inputs);
        }

        Tensors _preprocess(Tensors inputs)
        {
            Tensor input_tensor = null;
            if (args.Standardize != null)
                input_tensor = args.Standardize(inputs);
            if (!string.IsNullOrEmpty(args.Split))
            {
                if (inputs.shape.ndim > 1)
                    input_tensor = array_ops.squeeze(inputs, axis: new[] { -1 });
                if (args.Split == "whitespace")
                    input_tensor = tf.strings.split(input_tensor);
            }
            return input_tensor;
        }
    }
}

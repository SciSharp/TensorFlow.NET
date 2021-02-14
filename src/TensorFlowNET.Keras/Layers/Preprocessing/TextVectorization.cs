using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;

namespace Tensorflow.Keras.Layers
{
    public class TextVectorization : CombinerPreprocessingLayer
    {
        TextVectorizationArgs args;

        public TextVectorization(TextVectorizationArgs args)
            : base(args)
        {
            this.args = args;
            args.DType = TF_DataType.TF_STRING;
            // string standardize = "lower_and_strip_punctuation",
        }

        /// <summary>
        /// Fits the state of the preprocessing layer to the dataset.
        /// </summary>
        /// <param name="data"></param>
        /// <param name="reset_state"></param>
        public void adapt(IDatasetV2 data, bool reset_state = true)
        {
            var shape = data.output_shapes[0];
            if (shape.rank == 1)
                data = data.map(tensor => array_ops.expand_dims(tensor, -1));
            build(data.variant_tensor);
            var preprocessed_inputs = data.map(_preprocess);
        }

        protected override void build(Tensors inputs)
        {
            base.build(inputs);
        }

        Tensors _preprocess(Tensors inputs)
        {
            if (args.Standardize != null)
                inputs = args.Standardize(inputs);
            if (!string.IsNullOrEmpty(args.Split))
            {
                if (inputs.shape.ndim > 1)
                    inputs = array_ops.squeeze(inputs, axis: new[] { -1 });
            }
            return inputs;
        }
    }
}

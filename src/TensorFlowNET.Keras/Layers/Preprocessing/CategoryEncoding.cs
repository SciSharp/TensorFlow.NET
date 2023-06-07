using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using Tensorflow.Common.Types;
namespace Tensorflow.Keras.Layers
{
    /// <summary>
    /// This layer provides options for condensing data into a categorical encoding when the total number of tokens are known in advance.
    /// </summary>
    public class CategoryEncoding : Layer
    {
        CategoryEncodingArgs args;

        public CategoryEncoding(CategoryEncodingArgs args) : base(args)
        {
            this.args = args;
        }

        protected override Tensors Call(Tensors inputs, Tensors state = null, bool? training = null, IOptionalArgs? optional_args = null)
        {
            var depth = args.NumTokens;
            var max_value = tf.reduce_max(inputs);
            var min_value = tf.reduce_min(inputs);

            /*var condition = tf.logical_and(tf.greater(tf.cast(constant_op.constant(depth), max_value.dtype), max_value),
                tf.greater_equal(min_value, tf.cast(constant_op.constant(0), min_value.dtype)));*/

            var bincounts = encode_categorical_inputs(inputs, args.OutputMode, depth, args.DType, 
                sparse: args.Sparse, 
                count_weights: args.CountWeights);

            if(args.OutputMode != "tf_idf")
            {
                return bincounts;
            }

            return inputs;
        }

        public override Shape ComputeOutputShape(Shape input_shape)
        {
            return input_shape;
        }

        Tensors encode_categorical_inputs(Tensor inputs, string output_mode, int depth, 
            TF_DataType dtype = TF_DataType.TF_FLOAT, 
            bool sparse = false, 
            Tensor count_weights = null)
        {
            bool binary_output = false;
            if (output_mode == "one_hot")
            {
                binary_output = true;
                if (inputs.shape[-1] != 1)
                {
                    inputs = tf.expand_dims(inputs, -1);
                }
            }
            else if (output_mode == "multi_hot")
            {
                binary_output = true;
            }

            var depth_tensor = constant_op.constant(depth);
            var result = tf.math.bincount(inputs, 
                weights: count_weights,
                minlength: depth_tensor,
                maxlength: depth_tensor,
                dtype: dtype,
                axis: -1,
                binary_output: binary_output);

            return result;
        }
    }
}

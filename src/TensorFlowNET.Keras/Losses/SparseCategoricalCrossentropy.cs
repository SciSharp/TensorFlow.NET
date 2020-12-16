using static Tensorflow.Binding;

namespace Tensorflow.Keras.Losses
{
    public class SparseCategoricalCrossentropy : LossFunctionWrapper, ILossFunc
    {
        public SparseCategoricalCrossentropy(
            bool from_logits = false,
            string reduction = null,
            string name = null) :
            base(reduction: reduction, name: name == null ? "sparse_categorical_crossentropy" : name){ }

        public override Tensor Apply(Tensor target, Tensor output, bool from_logits = false, int axis = -1)
        {
            target = tf.cast(target, dtype: TF_DataType.TF_INT64);

            // Try to adjust the shape so that rank of labels = rank of logits - 1.
            var output_shape = array_ops.shape_v2(output);
            var output_rank = output.TensorShape.ndim;
            var target_rank = target.TensorShape.ndim;
            var update_shape = target_rank != output_rank - 1;
            if (update_shape)
            {
                target = array_ops.reshape(target, new int[] { -1 });
                output = array_ops.reshape(output, new int[] { -1, output_shape[-1].numpy() });
            }
            return tf.nn.sparse_softmax_cross_entropy_with_logits(target, output);
        }
    }
}

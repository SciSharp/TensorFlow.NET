using static Tensorflow.Binding;

namespace Tensorflow.Keras.Losses;

public class SparseCategoricalCrossentropy : LossFunctionWrapper
{
    private bool _from_logits = false;

    public SparseCategoricalCrossentropy(
        bool from_logits = false,
        string reduction = null,
        string name = null) :
        base(reduction: reduction, name: name == null ? "sparse_categorical_crossentropy" : name)
    {
        _from_logits = from_logits;
    }

    public override Tensor Apply(Tensor target, Tensor output, bool from_logits = false, int axis = -1)
    {
        target = tf.cast(target, dtype: TF_DataType.TF_INT64);

        if (!_from_logits)
        {
            var epsilon = tf.constant(KerasApi.keras.backend.epsilon(), output.dtype);
            output = tf.clip_by_value(output, epsilon, 1 - epsilon);
            output = tf.log(output);
        }

        // Try to adjust the shape so that rank of labels = rank of logits - 1.
        var output_shape = array_ops.shape_v2(output);
        var output_rank = output.shape.ndim;
        var target_rank = target.shape.ndim;
        var update_shape = target_rank != output_rank - 1;
        if (update_shape)
        {
            target = array_ops.reshape(target, new int[] { -1 });
            output = array_ops.reshape(output, new int[] { -1, output_shape[-1].numpy() });
        }
        return tf.nn.sparse_softmax_cross_entropy_with_logits(target, output);
    }
}
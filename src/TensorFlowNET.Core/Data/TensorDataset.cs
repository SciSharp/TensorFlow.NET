using NumSharp;
using System.Linq;
using static Tensorflow.Binding;

namespace Tensorflow
{
    /// <summary>
    /// A `Dataset` with a single element.
    /// </summary>
    public class TensorDataset : DatasetSource
    {
        public TensorDataset(Tensor feature, Tensor label)
        {
            _tensors = new[] { feature, label };
            structure = _tensors.Select(x => x.ToTensorSpec()).ToArray();

            variant_tensor = ops.tensor_dataset(_tensors, output_shapes);
        }
        public TensorDataset(Tensor element)
        {
            _tensors = new[] { element };
            var batched_spec = _tensors.Select(x => x.ToTensorSpec()).ToArray();
            structure = batched_spec.Select(x => x._unbatch()).ToArray();

            variant_tensor = ops.tensor_dataset(_tensors, output_shapes);
        }

        public TensorDataset(NDArray element)
        {
            _tensors = new[] { tf.convert_to_tensor(element) };
            var batched_spec = _tensors.Select(x => x.ToTensorSpec()).ToArray();
            structure = batched_spec.ToArray();

            variant_tensor = ops.tensor_dataset(_tensors, output_shapes);
        }
    }
}

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
        public TensorDataset(Tensors elements)
        {
            _tensors = elements;
            structure = _tensors.Select(x => x.ToTensorSpec()).ToArray();

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

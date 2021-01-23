using System;
using System.Linq;
using Tensorflow.Framework.Models;
using static Tensorflow.Binding;

namespace Tensorflow
{
    /// <summary>
    /// An iterator producing tf.Tensor objects from a tf.data.Dataset.
    /// </summary>
    public class OwnedIterator : IDisposable
    {
        IDatasetV2 _dataset;
        TensorSpec[] _element_spec;
        dataset_ops ops = new dataset_ops();
        Tensor _deleter;
        Tensor _iterator_resource;

        public OwnedIterator(IDatasetV2 dataset)
        {
            _create_iterator(dataset);
        }

        void _create_iterator(IDatasetV2 dataset)
        {
            dataset = dataset.apply_options();
            _dataset = dataset;
            _element_spec = dataset.element_spec;
            (_iterator_resource, _deleter) = ops.anonymous_iterator_v2(_dataset.output_types, _dataset.output_shapes);
            ops.make_iterator(dataset.variant_tensor, _iterator_resource);
        }

        public Tensor[] next()
        {
            try
            {
                var results = ops.iterator_get_next(_iterator_resource, _dataset.output_types, _dataset.output_shapes);
                foreach(var (i, tensor) in enumerate(results))
                    tensor.set_shape(_element_spec[i].shape);
                return results;
            }
            catch (OutOfRangeError ex)
            {
                throw new StopIteration(ex.Message);
            }
        }

        public void Dispose()
        {
            tf.Runner.Execute(tf.Context, "DeleteIterator", 0, new[] { _iterator_resource, _deleter }, null);
        }
    }
}

using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Framework
{
    /// <summary>
    /// A sparse representation of a set of tensor slices at given indices.
    /// </summary>
    public class IndexedSlices : CompositeTensor
    {
        Tensor _values;
        public Tensor values => _values;

        public IndexedSlices(Tensor values, Tensor indices, Tensor dense_shape = null)
        {

        }

        public static implicit operator Tensor(IndexedSlices indexedSlices)
        {
            return indexedSlices.values;
        }
    }
}

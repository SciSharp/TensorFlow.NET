using NumSharp.Core;
using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    /// <summary>
    /// Feed dictionary item
    /// </summary>
    public class FeedItem
    {
        public Tensor Key { get; }
        public NDArray Value { get; }

        public FeedItem(Tensor tensor, NDArray nd)
        {
            Key = tensor;
            Value = nd;
        }
    }
}

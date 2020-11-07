using System;

namespace Tensorflow
{
    /// <summary>
    /// An object which cleans up an iterator resource handle.
    /// </summary>
    public class IteratorResourceDeleter : IDisposable
    {
        Tensor _handle;
        Tensor _deleter;
        dataset_ops ops;

        public IteratorResourceDeleter(Tensor handle, Tensor deleter)
        {
            _handle = handle;
            _deleter = deleter;
            ops = new dataset_ops();
        }

        public void Dispose()
        {
            ops.delete_iterator(_handle, _deleter);
        }
    }
}

namespace Tensorflow.Keras.Engine.DataAdapters
{
    /// <summary>
    /// In TF 2.0, tf.data is the preferred API for user to feed in data. In order
    /// to simplify the training code path, all the input data object will be
    /// converted to `tf.data.Dataset` if possible.
    /// </summary>
    public interface IDataAdapter
    {
        /// <summary>
        /// Whether the current DataAdapter could handle the input x and y.
        /// </summary>
        /// <param name="x">input features</param>
        /// <param name="y">target labels</param>
        /// <returns></returns>
        bool CanHandle(Tensors x, Tensors y = null);
        IDatasetV2 GetDataset();
        int GetSize();
        (Tensors, Tensors) Expand1d(Tensors x, Tensors y);
        (Tensors, Tensors, Tensors) Expand1d(Tensors x, Tensors y, Tensors sample_weight);

        bool ShouldRecreateIterator();
    }
}

using NumSharp;

namespace TensorFlowNET.Examples.Utility
{
    public class Datasets<T> where T : IDataSet
    {
        private T _train;
        public T train => _train;

        private T _validation;
        public T validation => _validation;

        private T _test;
        public T test => _test;

        public Datasets(T train, T validation, T test)
        {
            _train = train;
            _validation = validation;
            _test = test;
        }

        public (NDArray, NDArray) Randomize(NDArray x, NDArray y)
        {
            var perm = np.random.permutation(y.shape[0]);

            np.random.shuffle(perm);
            return (x[perm], y[perm]);
        }

        /// <summary>
        /// selects a few number of images determined by the batch_size variable (if you don't know why, read about Stochastic Gradient Method)
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="start"></param>
        /// <param name="end"></param>
        /// <returns></returns>
        public (NDArray, NDArray) GetNextBatch(NDArray x, NDArray y, int start, int end)
        {
            var x_batch = x[$"{start}:{end}"];
            var y_batch = y[$"{start}:{end}"];
            return (x_batch, y_batch);
        }
    }
}

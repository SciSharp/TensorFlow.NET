using System;
using System.Collections.Generic;
using System.Text;
using NumSharp;

namespace Tensorflow
{
    public class Datasets<TDataSet> where TDataSet : IDataSet
    {
        public TDataSet Train { get; private set; }

        public TDataSet Validation { get; private set; }

        public TDataSet Test { get; private set; }

        public Datasets(TDataSet train, TDataSet validation, TDataSet test)
        {
            Train = train;
            Validation = validation;
            Test = test;
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
            var slice = new Slice(start, end);
            var x_batch = x[slice];
            var y_batch = y[slice];
            return (x_batch, y_batch);
        }
    }
}

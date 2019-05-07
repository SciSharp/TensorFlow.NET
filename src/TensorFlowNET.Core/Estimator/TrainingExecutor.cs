using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Estimator
{
    /// <summary>
    /// The executor to run `Estimator` training and evaluation.
    /// <see cref="tensorflow_estimator\python\estimator\training.py"/>
    /// </summary>
    public class TrainingExecutor : Python
    {
        private IEstimator _estimator;
        public TrainingExecutor(IEstimator estimator)
        {
            _estimator = estimator;
        }
    }
}

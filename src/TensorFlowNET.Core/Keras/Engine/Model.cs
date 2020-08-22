using NumSharp;
using System;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Optimizers;

namespace Tensorflow.Keras.Engine
{
    /// <summary>
    /// `Model` groups layers into an object with training and inference features.
    /// </summary>
    public class Model : Layer
    {
#pragma warning disable CS0169 // The field 'Model._cloning' is never used
        bool _cloning;
#pragma warning restore CS0169 // The field 'Model._cloning' is never used
#pragma warning disable CS0108 // Member hides inherited member; missing new keyword
#pragma warning disable CS0414 // The field 'Model._is_compiled' is assigned but its value is never used
        bool _is_compiled;
#pragma warning restore CS0414 // The field 'Model._is_compiled' is assigned but its value is never used
#pragma warning restore CS0108 // Member hides inherited member; missing new keyword
        string loss;
        IOptimizer optimizer;

        public Model(ModelArgs args) 
            : base(args)
        {

        }

        public void compile(string optimizerName, string lossName)
        {
            switch (optimizerName)
            {
                case "rmsprop":
                    optimizer = new RMSprop();
                    break;
            }

            loss = lossName;
            _is_compiled = true;

            // Prepare list of loss functions, same size of model outputs.
        }

        /// <summary>
        /// Generates output predictions for the input samples.
        /// </summary>
        /// <param name="x">Input samples</param>
        /// <param name="batch_size">Number of samples per batch</param>
        /// <param name="verbose">Verbosity mode</param>
        /// <param name="steps">
        /// Total number of steps (batches of samples)
        /// before declaring the prediction round finished.
        /// </param>
        /// <param name="max_queue_size"></param>
        /// <param name="workers"></param>
        /// <param name="use_multiprocessing"></param>
        /// <returns></returns>
        public Tensor predict(Tensor x,
            int batch_size = 32,
            int verbose = 0,
            int steps = -1,
            int max_queue_size = 10,
            int workers = 1,
            bool use_multiprocessing = false)
        {
            throw new NotImplementedException("");
        }
    }
}

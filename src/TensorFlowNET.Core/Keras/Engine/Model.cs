using static Tensorflow.Binding;
using System;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine.DataAdapters;
using Tensorflow.Keras.Losses;
using Tensorflow.Keras.Optimizers;

namespace Tensorflow.Keras.Engine
{
    /// <summary>
    /// `Model` groups layers into an object with training and inference features.
    /// </summary>
    public partial class Model : Layer
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
        IVariableV1 _steps_per_execution;
        protected bool _is_graph_network;
        protected Tensors inputs;
        protected Tensors outputs;

        public Model(ModelArgs args) 
            : base(args)
        {
            
        }

        public void compile(ILossFunc loss, OptimizerV2 optimizer, string[] metrics)
        {

        }

        public void compile(string optimizerName, string lossName)
        {
            switch (optimizerName)
            {
                case "rmsprop":
                    optimizer = new RMSprop(new RMSpropArgs
                    {

                    });
                    break;
            }

            int experimental_steps_per_execution = 1;
            _configure_steps_per_execution(experimental_steps_per_execution);

            _reset_compile_cache();

            loss = lossName;
            _is_compiled = true;
        }

        void _configure_steps_per_execution(int steps_per_execution)
        {
            _steps_per_execution = tf.Variable(steps_per_execution,
                dtype: TF_DataType.TF_INT64,
                aggregation: VariableAggregation.OnlyFirstReplica);
        }

        void _reset_compile_cache()
        {

        }

        public void compile(string optimizerName, ILossFunc lossName)
        {
            throw new NotImplementedException("");
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
            var data_handler = new DataHandler(new DataHandlerArgs
            {
                X = x,
                BatchSize = batch_size,
                StepsPerEpoch = steps,
                InitialEpoch = 0,
                Epochs = 1,
                MaxQueueSize = max_queue_size,
                Workers = workers,
                UseMultiprocessing = use_multiprocessing,
                Model = this,
                StepsPerExecution = _steps_per_execution
            });

            throw new NotImplementedException("");
        }
    }
}

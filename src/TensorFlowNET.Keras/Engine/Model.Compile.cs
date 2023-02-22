using System;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Losses;
using Tensorflow.Keras.Metrics;
using Tensorflow.Keras.Optimizers;

namespace Tensorflow.Keras.Engine
{
    public partial class Model
    {
        LossesContainer compiled_loss;
        MetricsContainer compiled_metrics;

        public void compile(OptimizerV2 optimizer = null, 
            ILossFunc loss = null, 
            string[] metrics = null)
        {
            this.optimizer = optimizer ?? new RMSprop(new RMSpropArgs
            {
            });

            this.loss = loss ?? new MeanSquaredError();

            compiled_loss = new LossesContainer(loss, output_names: output_names);
            compiled_metrics = new MetricsContainer(metrics, output_names: output_names);

            int experimental_steps_per_execution = 1;
            _configure_steps_per_execution(experimental_steps_per_execution);

            // Initialize cache attrs.
            _reset_compile_cache();
            _is_compiled = true;
        }

        public void compile(OptimizerV2 optimizer = null,
            ILossFunc loss = null,
            IMetricFunc[] metrics = null)
        {
            this.optimizer = optimizer ?? new RMSprop(new RMSpropArgs
            {
            });

            this.loss = loss ?? new MeanSquaredError();

            compiled_loss = new LossesContainer(loss, output_names: output_names);
            compiled_metrics = new MetricsContainer(metrics, output_names: output_names);

            int experimental_steps_per_execution = 1;
            _configure_steps_per_execution(experimental_steps_per_execution);

            // Initialize cache attrs.
            _reset_compile_cache();
            _is_compiled = true;
        }

        public void compile(string optimizer, string loss, string[] metrics)
        {
            var _optimizer = optimizer switch
            {
                "rmsprop" => new RMSprop(new RMSpropArgs
                {

                }),
                _ => throw new NotImplementedException("")
            };

            ILossFunc _loss = loss switch
            {
                "mse" => new MeanSquaredError(),
                "mae" => new MeanAbsoluteError(),
                _ => throw new NotImplementedException("")
            };

            compile(optimizer: _optimizer, loss: _loss, metrics: metrics);
        }
    }
}

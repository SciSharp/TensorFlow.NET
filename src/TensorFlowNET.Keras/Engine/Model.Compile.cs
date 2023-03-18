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

        public void compile(IOptimizer optimizer,
            ILossFunc loss)
        {
            this.optimizer = optimizer ?? new RMSprop(new RMSpropArgs
            {
            });

            this.loss = loss ?? new MeanSquaredError();

            compiled_loss = new LossesContainer(this.loss, output_names: output_names);
            compiled_metrics = new MetricsContainer(new string[0], output_names: output_names);

            int experimental_steps_per_execution = 1;
            _configure_steps_per_execution(experimental_steps_per_execution);

            // Initialize cache attrs.
            _reset_compile_cache();
            _is_compiled = true;
        }

        public void compile(IOptimizer optimizer,
            ILossFunc loss,
            string[] metrics)
        {
            this.optimizer = optimizer ?? new RMSprop(new RMSpropArgs
            {
            });

            this.loss = loss ?? new MeanSquaredError();

            compiled_loss = new LossesContainer(this.loss, output_names: output_names);
            compiled_metrics = new MetricsContainer(metrics, output_names: output_names);

            int experimental_steps_per_execution = 1;
            _configure_steps_per_execution(experimental_steps_per_execution);

            // Initialize cache attrs.
            _reset_compile_cache();
            _is_compiled = true;
        }

        public void compile(string optimizer, 
            string loss, 
            string[] metrics)
        {
            this.optimizer = optimizer switch
            {
                "rmsprop" => new RMSprop(new RMSpropArgs
                {

                }),
                _ => new RMSprop(new RMSpropArgs
                {
                })
            };

            this.loss = loss switch
            {
                "mse" => new MeanSquaredError(),
                "mae" => new MeanAbsoluteError(),
                _ => new MeanSquaredError()
            };

            compiled_loss = new LossesContainer(this.loss, output_names: output_names);
            compiled_metrics = new MetricsContainer(metrics, output_names: output_names);

            int experimental_steps_per_execution = 1;
            _configure_steps_per_execution(experimental_steps_per_execution);

            // Initialize cache attrs.
            _reset_compile_cache();
            _is_compiled = true;
        }

        public void compile(IOptimizer optimizer,
            ILossFunc loss,
            IMetricFunc[] metrics)
        {
            this.optimizer = optimizer ?? new RMSprop(new RMSpropArgs
            {
            });

            this.loss = loss ?? new MeanSquaredError();

            compiled_loss = new LossesContainer(this.loss, output_names: output_names);
            compiled_metrics = new MetricsContainer(metrics, output_names: output_names);

            int experimental_steps_per_execution = 1;
            _configure_steps_per_execution(experimental_steps_per_execution);

            // Initialize cache attrs.
            _reset_compile_cache();
            _is_compiled = true;
        }
    }
}

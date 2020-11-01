using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Losses;
using Tensorflow.Keras.Optimizers;
using Tensorflow.Keras.Utils;

namespace Tensorflow.Keras.Engine
{
    public partial class Model
    {
        LossesContainer compiled_loss;
        MetricsContainer compiled_metrics;
        public void compile(string optimizerName, ILossFunc lossName)
        {
            throw new NotImplementedException("");
        }

        public void compile(ILossFunc loss, OptimizerV2 optimizer, string[] metrics)
        {
            this.optimizer = optimizer;
            compiled_loss = new LossesContainer(loss, output_names: output_names);
            compiled_metrics = new MetricsContainer(metrics, output_names: output_names);

            int experimental_steps_per_execution = 1;
            _configure_steps_per_execution(experimental_steps_per_execution);

            // Initialize cache attrs.
            _reset_compile_cache();
            _is_compiled = true;
            this.loss = loss;
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

            _is_compiled = true;
        }
    }
}

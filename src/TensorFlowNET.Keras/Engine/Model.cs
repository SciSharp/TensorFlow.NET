using System.Collections.Generic;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine.DataAdapters;
using Tensorflow.Keras.Losses;
using Tensorflow.Keras.Optimizers;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace Tensorflow.Keras.Engine
{
    /// <summary>
    /// `Model` groups layers into an object with training and inference features.
    /// </summary>
    public partial class Model : Layer, IModel
    {
#pragma warning disable CS0169 // The field 'Model._cloning' is never used
        bool _cloning;
#pragma warning restore CS0169 // The field 'Model._cloning' is never used
#pragma warning disable CS0108 // Member hides inherited member; missing new keyword
#pragma warning disable CS0414 // The field 'Model._is_compiled' is assigned but its value is never used
        bool _is_compiled;
#pragma warning restore CS0414 // The field 'Model._is_compiled' is assigned but its value is never used
#pragma warning restore CS0108 // Member hides inherited member; missing new keyword
        ILossFunc loss;
        OptimizerV2 optimizer;
        IVariableV1 _steps_per_execution;
        protected bool _is_graph_network;
        protected Tensors inputs;
        protected Tensors outputs;
        public string[] output_names;
        IVariableV1 _train_counter;
        IVariableV1 _test_counter;
        IVariableV1 _predict_counter;
        bool _base_model_initialized;
        bool stop_training;
        DataHandler data_handler;

        public Model(ModelArgs args)
            : base(args)
        {
            _init_batch_counters();
        }

        void _configure_steps_per_execution(int steps_per_execution)
        {
            _steps_per_execution = tf.Variable(steps_per_execution,
                dtype: TF_DataType.TF_INT64,
                aggregation: VariableAggregation.OnlyFirstReplica);
        }

        void _reset_compile_cache()
        {
            // Used to cache `trainable` attr of `Layer`s for `fit`.
            _compiled_trainable_state = _get_trainable_state();
            keras.backend._GRAPH = null;
        }

        void _init_batch_counters()
        {
            _train_counter = tf.Variable(0,
                dtype: TF_DataType.TF_INT64,
                aggregation: VariableAggregation.OnlyFirstReplica);

            _test_counter = tf.Variable(0,
                dtype: TF_DataType.TF_INT64,
                aggregation: VariableAggregation.OnlyFirstReplica);

            _predict_counter = tf.Variable(0,
                dtype: TF_DataType.TF_INT64,
                aggregation: VariableAggregation.OnlyFirstReplica);
        }

        public override List<IVariableV1> trainable_variables
        {
            get
            {
                var variables = new List<IVariableV1>();
                foreach (var layer in _layers)
                {
                    if (layer.Trainable)
                        variables.AddRange(layer.trainable_variables);
                }
                return variables;
            }
        }
    }
}

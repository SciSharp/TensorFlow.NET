using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Engine.DataAdapters;

namespace Tensorflow.Keras.Layers
{
    public class PreprocessingLayer : Layer
    {
        bool _is_compiled;
        bool _is_adapted;
        IVariableV1 _steps_per_execution;
        PreprocessingLayerArgs _args;
        public PreprocessingLayer(PreprocessingLayerArgs args) : base(args)
        {
            _args = args;
        }

        public override void adapt(Tensor data, int? batch_size = null, int? steps = null)
        {
            if (!_is_compiled)
            {
                compile();
            }

            if (built)
            {
                reset_state();
            }

            var data_handler = new DataHandler(new DataHandlerArgs
            {
                X = new Tensors(data),
                BatchSize = _args.BatchSize,
                Epochs = 1,
                StepsPerExecution = _steps_per_execution
            });

            foreach (var (epoch, iterator) in data_handler.enumerate_epochs())
            {
                foreach (var _ in data_handler.steps())
                {
                    run_step(iterator);
                }
            }
            finalize_state();
            _is_adapted = true;
        }

        private void run_step(OwnedIterator iterator)
        {
            var data = iterator.next();
            _adapt_maybe_build(data[0]);
            update_state(data[0]);
        }

        public virtual void reset_state()
        {

        }

        public virtual void finalize_state()
        {

        }

        public virtual void update_state(Tensor data)
        {

        }

        private void _adapt_maybe_build(Tensor data)
        {
            if (!built)
            {
                var data_shape = data.shape;
                var data_shape_nones = Enumerable.Range(0, data.ndim).Select(x => -1).ToArray();
                _args.BatchInputShape = BatchInputShape ?? new Saving.KerasShapesWrapper(new Shape(data_shape_nones));
                build(new Saving.KerasShapesWrapper(data_shape));
                built = true;
            }
        }

        public void compile(bool run_eagerly = false, int steps_per_execution = 1)
        {
            _steps_per_execution = tf.Variable(
                steps_per_execution,
                dtype: tf.int64,
                aggregation: VariableAggregation.OnlyFirstReplica
            );

            _is_compiled = true;
        }
    }
}

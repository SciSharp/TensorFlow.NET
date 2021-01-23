using System;
using System.Collections.Generic;
using Tensorflow.Keras.ArgsDefinition;
using static Tensorflow.Binding;

namespace Tensorflow.Keras.Engine.DataAdapters
{
    /// <summary>
    /// Handles iterating over epoch-level `tf.data.Iterator` objects.
    /// </summary>
    public class DataHandler
    {
        DataHandlerArgs args;
        IDataAdapter _adapter;
        public IDataAdapter DataAdapter => _adapter;
        IDatasetV2 _dataset;
        int _inferred_steps;
        public int Inferredsteps => _inferred_steps;
        int _current_step;
        int _step_increment;
        public int StepIncrement => _step_increment;
        bool _insufficient_data;
        int _steps_per_execution_value;
        int _initial_epoch => args.InitialEpoch;
        int _epochs => args.Epochs;
        IVariableV1 _steps_per_execution;

        public DataHandler(DataHandlerArgs args)
        {
            this.args = args;
            if (args.StepsPerExecution == null)
            {
                _steps_per_execution = tf.Variable(1);
                _steps_per_execution_value = 1;
            }
            else
            {
                _steps_per_execution = args.StepsPerExecution;
                _steps_per_execution_value = args.StepsPerExecution.numpy();
            }

            if(args.Dataset == null)
            {
                _adapter = new TensorLikeDataAdapter(new DataAdapterArgs
                {
                    X = args.X,
                    Y = args.Y,
                    BatchSize = args.BatchSize,
                    Steps = args.StepsPerEpoch,
                    Epochs = args.Epochs - args.InitialEpoch,
                    Shuffle = args.Shuffle,
                    MaxQueueSize = args.MaxQueueSize,
                    Worker = args.Workers,
                    UseMultiprocessing = args.UseMultiprocessing,
                    Model = args.Model
                });
            }
            else
            {
                _adapter = new DatasetAdapter(new DataAdapterArgs
                {
                    Dataset = args.Dataset,
                    BatchSize = args.BatchSize,
                    Steps = args.StepsPerEpoch,
                    Epochs = args.Epochs - args.InitialEpoch,
                    Shuffle = args.Shuffle,
                    MaxQueueSize = args.MaxQueueSize,
                    Worker = args.Workers,
                    UseMultiprocessing = args.UseMultiprocessing,
                    Model = args.Model
                });
            }
            
            _dataset = _adapter.GetDataset();
            _inferred_steps = _infer_steps(args.StepsPerEpoch, _dataset);
            _current_step = 0;
            _step_increment = _steps_per_execution_value - 1;
            _insufficient_data = false;
        }

        int _infer_steps(int steps_per_epoch, IDatasetV2 dataset)
        {
            if (steps_per_epoch > -1)
                return steps_per_epoch;

            var adapter_steps = _adapter.GetSize();
            if (adapter_steps > -1)
                return adapter_steps;

            var size = dataset.dataset_cardinality();
            return size.numpy();
        }

        public IEnumerable<(int, OwnedIterator)> enumerate_epochs()
        {
            foreach (var epoch in range(_initial_epoch, _epochs))
            {
                if (_insufficient_data)
                    break;
                using var data_iterator = new OwnedIterator(_dataset);
                yield return (epoch, data_iterator);
            }
        }

        public IEnumerable<int> steps()
        {
            _current_step = 0;
            while (_current_step < _inferred_steps)
            {
                if (_insufficient_data)
                    break;

                bool can_run_full_execution = _steps_per_execution_value == 1
                    || _inferred_steps < 0
                    || _inferred_steps - _current_step >= _steps_per_execution_value;

                if (can_run_full_execution)
                {
                    _step_increment = _steps_per_execution_value - 1;
                    yield return _current_step;
                    _current_step += _steps_per_execution_value;
                }
                else
                {
                    var steps_remaining = _inferred_steps - _current_step;
                    _steps_per_execution.assign(steps_remaining);
                    _step_increment = steps_remaining - 1;
                    yield return _current_step;
                    _current_step += steps_remaining;
                    _steps_per_execution.assign(_steps_per_execution_value);
                }
            }
        }
    }
}

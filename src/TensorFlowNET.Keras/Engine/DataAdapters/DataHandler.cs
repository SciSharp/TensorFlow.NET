using System;
using System.Collections.Generic;
using Tensorflow.Keras.ArgsDefinition;
using static Tensorflow.Binding;
using Tensorflow.Keras.Utils;
using Tensorflow.Util;
using Tensorflow.Framework;

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
        long _inferred_steps;
        public long Inferredsteps => _inferred_steps;
        long _current_step;
        long _step_increment;
        public long StepIncrement => _step_increment;
        bool _insufficient_data;
        long _steps_per_execution_value;
        int _initial_epoch => args.InitialEpoch;
        int _epochs => args.Epochs;
        NDArray _sample_weight => args.SampleWeight;
        IVariableV1 _steps_per_execution;

        public DataHandler(DataHandlerArgs args)
        {
            this.args = args;
            
            if (args.StepsPerExecution == null)
            {
                _steps_per_execution = tf.Variable(1L);
                _steps_per_execution_value = 1L;
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
                    SampleWeight = args.SampleWeight,
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
            _current_step = 0;
            _step_increment = _steps_per_execution_value - 1;
            _insufficient_data = false;
            _configure_dataset_and_inferred_steps(args.X, args.ClassWeight);
        }

        void _configure_dataset_and_inferred_steps(Tensors x, Dictionary<int, float> class_weight)
        {
            if (_dataset == null)
            {
                _dataset = _adapter.GetDataset();
                _inferred_steps = _infer_steps(args.StepsPerEpoch, _dataset);
            }

            if (class_weight != null)
            {
                _dataset = _dataset.map(_make_class_weight_map_fn(class_weight));
            }
            _inferred_steps = _infer_steps(args.StepsPerEpoch, _dataset);
        }


        Func<Tensors, Tensors> _make_class_weight_map_fn(Dictionary<int, float> class_weight)
        {
            var class_ids = class_weight.Keys.OrderBy(key => key).ToList();
            var expected_class_ids = range(class_ids[0], class_ids[class_ids.Count - 1] + 1);
            if (!class_ids.SequenceEqual(expected_class_ids))
            {
                throw new ValueError("Expected `class_weight` to be a dict with keys from 0 to one less "+
                    $"than the number of classes, found {class_weight}");
            }
            
            var class_weight_list = new List<float>();
            foreach (var class_id in class_ids)
            {
                class_weight_list.Add(class_weight[class_id]);
            }
            var class_weight_tensor = tf.convert_to_tensor(class_weight_list.ToArray());

            Func<Tensors, Tensors> _class_weight_map_fn = (Tensors data) =>
            {
                var x = data[0];
                var y = data[1];
                var sw = _sample_weight == null ? null : ops.convert_to_tensor(_sample_weight);

                if (y.shape.rank > 2)
                {
                    throw new ValueError("`class_weight` not supported for 3+ dimensional targets.");
                }

                var y_classes = smart_module.smart_cond(
                    y.shape.rank == 2 && y.shape[1] > 1,
                    () => math_ops.argmax(y, dimension: 1),
                    () => math_ops.cast(tf.reshape(y, (-1)), TF_DataType.TF_INT64));

                var cw = array_ops.gather(class_weight_tensor, y_classes);
                if (sw != null)
                {
                    cw = tf.cast(cw, sw.dtype);
                    cw *= sw;
                }
                else
                {
                    sw = cw;
                }
                return new Tensors { x, y, sw };
            };

            return _class_weight_map_fn;
        }

        long _infer_steps(int steps_per_epoch, IDatasetV2 dataset)
        {
            if (steps_per_epoch > -1)
                return steps_per_epoch;

            var adapter_steps = _adapter.GetSize();
            if (adapter_steps > -1)
                return adapter_steps;

            var size = dataset.cardinality();
            return size.numpy();
        }

        public IEnumerable<(int, OwnedIterator)> enumerate_epochs()
        {
            var data_iterator = new OwnedIterator(_dataset);
            foreach (var epoch in range(_initial_epoch, _epochs))
            {
                if (_insufficient_data)
                    break;
                if (_adapter.ShouldRecreateIterator())
                {
                    data_iterator = new OwnedIterator(_dataset);
                }
                yield return (epoch, data_iterator);
            }
            // _adapter.on_epoch_end()
        }

        public IEnumerable<long> steps()
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

using Google.Protobuf;
using System;
using System.Collections;
using System.Collections.Generic;
using System.IO.Compression;
using System.Linq;
using System.Linq.Expressions;
using System.Runtime.InteropServices;
using System.Text;
using Tensorflow.Functions;
using Tensorflow.Keras;
using Tensorflow.Operations.Activation;
using Tensorflow.Train;
using static Tensorflow.ApiDef.Types;

namespace Tensorflow.Training
{
    public class NoDependency
    {
        public Trackable Value { get; set; }
        public NoDependency(Trackable value)
        {
            Value = value;
        }
    }

    public abstract class TrackableDataStructure : Trackable
    {
        private bool _self_trainable;
        private List<IVariableV1> _self_extra_variables;

        public TrackableDataStructure()
        {
            _self_trainable = true;
            _self_extra_variables = new List<IVariableV1>();
        }

        public abstract IEnumerable<Trackable> Values { get; }
        public bool Trainable { get => _self_trainable; set => _self_trainable = value; }
        public IEnumerable<ILayer> Layers
        {
            get
            {
                List<ILayer> collected = new();
                foreach(var obj in Values)
                {
                    if(obj is ILayer)
                    {
                        collected.Add((ILayer)obj);
                    }
                    else if(obj is TrackableDataStructure)
                    {
                        collected.AddRange((obj as TrackableDataStructure).Layers);
                    }
                }
                return collected;
            }
        }
        public IEnumerable<IVariableV1> TrainableWeights
        {
            get
            {
                if (!_self_trainable)
                {
                    return new List<IVariableV1>();
                }
                List<IVariableV1> trainable_variables = new();
                foreach (var obj in Values)
                {
                    // skip the process of `module.Module`.
                    if (obj is TrackableDataStructure)
                    {
                        trainable_variables.AddRange((obj as TrackableDataStructure).TrainableVariables);
                    }
                }
                foreach(var v in _self_extra_variables)
                {
                    if (v.Trainable)
                    {
                        trainable_variables.Add(v);
                    }
                }
                return trainable_variables;
            }
        }
        public IEnumerable<IVariableV1> NonTrainableWeights
        {
            get
            {
                var trainable_extra_variables = _self_extra_variables.TakeWhile(x => x.Trainable).ToList();
                var non_trainable_extra_variables = _self_extra_variables.TakeWhile(x => !x.Trainable).ToList();
                List<IVariableV1> non_trainable_variables = new();
                foreach(var obj in Values)
                {
                    // skip the process of `module.Module`.
                    if (obj is TrackableDataStructure)
                    {
                        non_trainable_variables.AddRange((obj as TrackableDataStructure).NonTrainableVariables);
                    }
                }

                if (!_self_trainable)
                {
                    // Return order is all trainable vars, then all non-trainable vars.
                    List<IVariableV1> trainable_variables = new();
                    foreach(var obj in Values)
                    {
                        // skip the process of `module.Module`.
                        if (obj is TrackableDataStructure)
                        {
                            trainable_variables.AddRange((obj as TrackableDataStructure).TrainableVariables);
                        }
                    }
                    return trainable_variables.concat(trainable_extra_variables).concat(non_trainable_variables).concat(non_trainable_extra_variables);
                }
                else
                {
                    return non_trainable_variables.concat(non_trainable_extra_variables);
                }
            }
        }
        public IEnumerable<IVariableV1> Weights => TrainableWeights.Concat(NonTrainableWeights);
        public IEnumerable<IVariableV1> TrainableVariables => TrainableWeights;
        public IEnumerable<IVariableV1> NonTrainableVariables => NonTrainableWeights;
        public IEnumerable<IVariableV1> Variables => Weights;

        // TODO: `losses` property.

        /// <summary>
        /// Add a dependency on `value`.
        /// </summary>
        /// <param name="value"></param>
        /// <param name="name"></param>
        protected virtual Trackable _track_value(Trackable value, string name)
        {
            value = sticky_attribute_assignment(this, name, value);
            if(value is IVariableV1)
            {
                _self_extra_variables.Add(value as IVariableV1);
            }
            // skip the left process (need to be done in the future).
            return value;
        }

        public static Trackable wrap_or_unwrap(NoDependency value)
        {
            return value.Value;
        }

        public static Trackable wrap_or_unwrap(Trackable value)
        {
            return value;
        }

        public static Trackable wrap_or_unwrap(IList<Trackable> value)
        {
            return new ListWrapper(value);
        }

        public static Trackable wrap_or_unwrap(IEnumerable<Trackable> value)
        {
            return new ListWrapper(value.ToList());
        }

        protected static Trackable sticky_attribute_assignment(Trackable trackable, string name, Trackable value)
        {
            value = wrap_or_unwrap(value);
            trackable._track_trackable(value, name, true);
            return value;
        }

        protected static Trackable sticky_attribute_assignment(Trackable trackable, string name, NoDependency value)
        {
            var wrapped_value = wrap_or_unwrap(value);
            trackable._track_trackable(wrapped_value, name, true);
            return wrapped_value;
        }

        protected static Trackable sticky_attribute_assignment(Trackable trackable, string name, IList<Trackable> value)
        {
            var wrapped_value = wrap_or_unwrap(value);
            trackable._track_trackable(wrapped_value, name, true);
            return wrapped_value;
        }
    }

    public class ListWrapper : TrackableDataStructure, IList<Trackable>, ICloneable
    {
        private IList<Trackable> _storage;
        private bool _non_append_mutation_value;
        private bool _external_modification_value;
        private IList<Trackable> _last_wrapped_list_snapshot;
        /// <summary>
        /// 
        /// </summary>
        /// <param name="wrapped_list">The initial value of the data structure. A shallow copy may be maintained for error checking. `wrapped_list` itself should not be
        /// modified directly after constructing the `ListWrapper`, and if changes are detected the `ListWrapper` will throw an exception on save.</param>
        public ListWrapper(IList<Trackable> wrapped_list)
        {
            _storage = wrapped_list;
            _non_append_mutation_value = _external_modification_value = false;
            _last_wrapped_list_snapshot = new List<Trackable>(_storage);
        }

        protected bool NonAppendMuation { 
            get => _non_append_mutation_value;
            set
            {
                // TODO: deal with `attribute_sentinel`.
                _non_append_mutation_value = value;
            }
        }

        protected bool ExternalModification
        {
            get => _external_modification_value;
            set
            {
                // TODO: deal with `attribute_sentinel`.
                _external_modification_value = value;
            }
        }

        public override IEnumerable<Trackable> Values => this;
        public bool IsReadOnly { get => _storage.IsReadOnly; }

        /// <summary>
        /// Checks for any changes to the wrapped list not through the wrapper.
        /// </summary>
        private void check_external_modification()
        {
            if (_external_modification_value || _non_append_mutation_value) return;
            if (!_storage.SequenceEqual(_last_wrapped_list_snapshot))
            {
                _external_modification_value = true;
            }
        }

        private void update_snapshot()
        {
            // TODO: deal with `attribute_sentinel`.
            if (_external_modification_value || _non_append_mutation_value) return;
            _last_wrapped_list_snapshot = new List<Trackable>(_storage);
        }

        public override IDictionary<string, Trackable> _trackable_children(SaveType save_type, IDictionary<string, object>? cache = null)
        {
            check_external_modification();
            if (_non_append_mutation_value)
            {
                throw new ValueError($"Unable to save the object {this} (a list wrapper constructed to track trackable TensorFlow objects). A list element was replaced" +
                    $", deleted or moved (sort). In order to support restoration on object creation, tracking is exclusively for append-only data structures." +
                    $"\n\nIf you don't need this list checkpointed, wrap it in a non-trackable object; it will be subsequently ignored.");
            }
            if (_external_modification_value)
            {
                throw new ValueError($"Unable to save the object {this} (a list wrapper constructed to track trackable TensorFlow objects). The wrapped list was modified " +
                    $"outside the wrapper (its final value was {_storage}, its value when a checkpoint dependency was added was {_last_wrapped_list_snapshot}), which breaks " +
                    $"restoration on object creation.\n\nIf you don't need this list checkpointed, wrap it in a NoDependency object; it will be subsequently ignored.");
            }
            var children = base._trackable_children(save_type, cache);

            if(save_type == SaveType.SAVEDMODEL)
            {
                children = children.Concat(this.TakeWhile(x => x is Function or ConcreteFunction).Select((x, idx) => new KeyValuePair<string, Trackable>(idx.ToString(), x))).ToDictionary(x => x.Key, x => x.Value);
            }

            return children;
        }

        private bool has_mutation_or_trackable()
        {
            return _non_append_mutation_value;
        }

        /// <summary>
        /// Allows storage of non-trackable objects.
        /// </summary>
        /// <param name="value"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        protected override Trackable _track_value(Trackable value, string name)
        {
            try
            {
                base._track_value(value, name);
            }
            catch(ValueError ex)
            {
                value = sticky_attribute_assignment(this, name, value);
            }
            return value;
        }

        public object Clone()
        {
            var res = new ListWrapper(_storage.Select(x => x).ToList());
            res.NonAppendMuation= _non_append_mutation_value;
            res.ExternalModification = _external_modification_value;
            return res;
        }

        public Trackable this[int index] { 
            get => _storage[index];
            set
            {
                // skip the process of `Slice`, maybe support it in the future.
                _non_append_mutation_value = true;
                _storage[index] = _track_value(value, _name_element(index));

                update_snapshot();
            }
        }

        public int IndexOf(Trackable item) => _storage.IndexOf(item);

        public void Insert(int index, Trackable item)
        {
            check_external_modification();
            _non_append_mutation_value = true;
            _storage.Insert(index, item);
            update_snapshot();
        }

        public void RemoveAt(int index)
        {
            check_external_modification();
            if (has_mutation_or_trackable())
            {
                _non_append_mutation_value = true;
            }
            _storage.RemoveAt(index);
            update_snapshot();
        }

        public int Count { get => _storage.Count; }

        public void Add(Trackable item)
        {
            check_external_modification();
            _storage.Add(item);
            update_snapshot();
        }

        public void Clear() => _storage.Clear();

        public bool Contains(Trackable item) => _storage.Contains(item);

        public void CopyTo(Trackable[] array, int arrayIndex) => _storage.CopyTo(array, arrayIndex);

        public bool Remove(Trackable item)
        {
            check_external_modification();
            if (has_mutation_or_trackable())
            {
                _non_append_mutation_value = true;
            }
            var res = _storage.Remove(item);
            update_snapshot();
            return res;
        }

        public IEnumerator<Trackable> GetEnumerator() => _storage.GetEnumerator();

        IEnumerator IEnumerable.GetEnumerator() => _storage.GetEnumerator();

        protected string _name_element(int index) => $"{index}";
    }
}

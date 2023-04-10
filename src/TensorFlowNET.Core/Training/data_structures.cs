using Google.Protobuf;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.IO.Compression;
using System.Linq;
using System.Linq.Expressions;
using System.Runtime.InteropServices;
using System.Text;
using Tensorflow.Functions;
using Tensorflow.Keras;
using Tensorflow.Keras.Saving.SavedModel;
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

    static class TrackableWrapperUtils
    {
        internal static bool ShouldLoad(ITrackableWrapper wrapper, SavedUserObject proto)
        {
            if (proto.Identifier != wrapper.Identifier)
            {
                return false;
            }
            if (wrapper.Version < proto.Version.MinConsumer)
            {
                return false;
            }
            if (proto.Version.Producer < wrapper.MinProducerVersion)
            {
                return false;
            }
            foreach (var bad_version in proto.Version.BadConsumers)
            {
                if (bad_version == wrapper.Version)
                {
                    return false;
                }
            }
            return true;
        }

        internal static bool is_function(Trackable x)
        {
            return x is Function or ConcreteFunction;
        }
    }

    public interface ITrackableWrapper
    {
        void SetValue(object name, object value);
        String Identifier { get; }
        int Version { get; }
        int MinConsumerVersion { get; }
        int MinProducerVersion { get; }
        Trackable FromProto(SavedUserObject proto);
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

        public abstract ICollection<Trackable> Values { get; }
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
                var trainable_extra_variables = _self_extra_variables.Where(x => x.Trainable).ToList();
                var non_trainable_extra_variables = _self_extra_variables.Where(x => !x.Trainable).ToList();
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
            value = (Trackable)sticky_attribute_assignment(this, name, value);
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

        public static object wrap_or_unwrap(object value)
        {
            if(value is NoDependency dependency)
            {
                return dependency.Value;
            }
            if(value is Trackable trackable)
            {
                return trackable;
            }
            else if(value is IDictionary<object, Trackable> obj_dict)
            {
                return new DictWrapper(obj_dict);
            }
            else if(value is IList<Trackable> list)
            {
                return new ListWrapper(list);
            }
            else
            {
                return value;
            }
        }

        public static object sticky_attribute_assignment(Trackable trackable, string name, object value)
        {
            bool add_dependency = value is not NoDependency;
            value = wrap_or_unwrap(value);
            if (!add_dependency)
            {
                return value;
            }
            if(value is Trackable trackable_obj)
            {
                trackable._track_trackable(trackable_obj, name, true);
            }
            return value;
        }
    }
    // TODO(Rinne): Add Dict wrapper and Tuple wrapper

    public class DictWrapper : TrackableDataStructure, IDictionary<object, Trackable>, ICloneable, ITrackableWrapper
    {
        private IDictionary<object, Trackable> _storage;
        private bool _non_string_key;
        private bool _external_modification;
        private IDictionary<object, Trackable> _last_wrapped_dict_snapshot;

        public DictWrapper(IDictionary<object, Trackable> wrapped_dict = null)
        {
            if(wrapped_dict is not null)
            {
                _storage = new Dictionary<object, Trackable>(wrapped_dict);
            }
            else
            {
                _storage = new Dictionary<object, Trackable>();
            }
            _update_snapshot();
        }

        public void SetValue(object name, object value)
        {
            Debug.Assert(value is Trackable);
            this[name] = value as Trackable;
        }
        public String Identifier => "trackable_dict_wrapper";
        public int Version => 1;
        public int MinConsumerVersion => 1;
        public int MinProducerVersion => 1;
        public Trackable FromProto(SavedUserObject proto)
        {
            return new DictWrapper(new Dictionary<object, Trackable>());
        }

        public Trackable this[object key]
        {
            get
            {
                return _storage[key];
            }
            set
            {
                _check_self_external_modification();
                _maybe_initialize_trackable();
                bool no_dep = value is NoDependency;
                if(key is string)
                {
                    value = _track_value(value, key);
                }
                else
                {
                    value = (Trackable)wrap_or_unwrap(value);
                    if(!no_dep && value is Trackable)
                    {
                        _non_string_key = true;
                    }
                }
                _storage[key] = value;
                _update_snapshot();
            }
        }

        public ICollection<object> Keys => _storage.Keys;

        public override ICollection<Trackable> Values => _storage.OrderBy(x => x.Key).Select(x => x.Value).ToArray();

        public void Add(object key, Trackable value)
        {
            _storage[key] = value;
        }

        public bool ContainsKey(object key)
        {
            return _storage.ContainsKey(key);
        }

        public bool Remove(object key)
        {
            _check_self_external_modification();
            var res = _storage.Remove(key);
            _update_snapshot();
            return res;
        }

        public bool TryGetValue(object key, out Trackable value)
        {
            return _storage.TryGetValue(key, out value);
        }

        public int Count => _storage.Count;

        public bool IsReadOnly => _storage.IsReadOnly;

        public void Add(KeyValuePair<object, Trackable> item)
        {
            Add(item.Key, item.Value);
        }

        public void Clear()
        {
            _storage.Clear();
            _update_snapshot();
        }

        public bool Contains(KeyValuePair<object, Trackable> item)
        {
            return _storage.Contains(item);
        }

        public void CopyTo(KeyValuePair<object, Trackable>[] array, int arrayIndex)
        {
            _storage.CopyTo(array, arrayIndex);
        }

        public bool Remove(KeyValuePair<object, Trackable> item)
        {
            _check_self_external_modification();
            var res = Remove(item);
            _update_snapshot();
            return res;
        }

        public IEnumerator<KeyValuePair<object, Trackable>> GetEnumerator()
        {
            return _storage.GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator() => _storage.GetEnumerator();

        public object Clone()
        {
            var copied = new DictWrapper(_storage);
            copied._external_modification = _external_modification;
            copied._non_string_key = _non_string_key;
            return copied;
        }

        public override IDictionary<string, Trackable> _trackable_children(SaveType save_type = SaveType.CHECKPOINT, IDictionary<string, IDictionary<Trackable, ISerializedAttributes>>? cache = null)
        {
            _check_self_external_modification();
            if (_non_string_key)
            {
                throw new ValueError($"Unable to save the object {this} (a dictionary wrapper constructed \"" +
                    $"automatically on attribute assignment). The wrapped dictionary " +
                    $"contains a non-string key which maps to a trackable object or " +
                    $"mutable data structure.\n\nIf you don't need this dictionary " +
                    $"checkpointed, wrap it in a non-trackable " +
                    $"object; it will be subsequently ignored.");
            }
            if (_external_modification)
            {
                throw new ValueError($"Unable to save the object {this} (a dictionary wrapper constructed " +
                    $"automatically on attribute assignment). The wrapped dictionary was " +
                    $"modified outside the wrapper (its final value was {this}, its value" +
                    $" when a checkpoint dependency was added was " +
                    $"{this._last_wrapped_dict_snapshot}), which breaks " +
                    $"restoration on object creation.\n\nIf you don't need this " +
                    $"dictionary checkpointed, wrap it in a " +
                    $"non-trackable object; it will be subsequently ignored.");
            }
            Debug.Assert(!Dirty);
            var children = base._trackable_children(save_type, cache);

            if(save_type == SaveType.SAVEDMODEL)
            {
                foreach(var item in _storage)
                {
                    var key = item.Key;
                    var value = item.Value;
                    if (TrackableWrapperUtils.is_function(value))
                    {
                        Debug.Assert(key is string);
                        children[key as string] = value;
                    }
                }
            }

            return children;
        }

        protected Trackable _track_value(Trackable value, object name)
        {
            bool string_key = name is string;
            if (!string_key)
            {
                name = "-non_string_key";
            }
            try
            {
                bool no_dependency = value is NoDependency;
                value = base._track_value(value, name as string);
                if(!(string_key || no_dependency))
                {
                    _non_string_key = true;
                }
                return value;
            }
            catch (ValueError)
            {
                return (Trackable)sticky_attribute_assignment(this, name as string, value);
            }
        }

        private bool Dirty => _external_modification || _non_string_key;

        private void _check_self_external_modification()
        {
            if (Dirty)
            {
                return;
            }
            if(!this._storage.SequenceEqual(_last_wrapped_dict_snapshot))
            {
                _external_modification = true;
                _last_wrapped_dict_snapshot = null;
            }
        }

        private void _update_snapshot()
        {
            // TODO(Rinne): deal with attribute_sentinel.
            if (Dirty) return;
            _last_wrapped_dict_snapshot = new Dictionary<object, Trackable>(_storage);
        }
    }
    public class ListWrapper : TrackableDataStructure, IList<Trackable>, ICloneable, ITrackableWrapper
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
            _storage = new List<Trackable>(wrapped_list);
            _non_append_mutation_value = _external_modification_value = false;
            _last_wrapped_list_snapshot = new List<Trackable>(_storage);
        }

        public string Identifier => "trackable_list_wrapper";
        public int Version => 1;
        public int MinConsumerVersion => 1;
        public int MinProducerVersion => 1;
        public Trackable FromProto(SavedUserObject proto)
        {
            if(TrackableWrapperUtils.ShouldLoad(this, proto))
            {
                return new ListWrapper(new Trackable[] { });
            }
            else
            {
                return null;
            }
        }
        public void SetValue(object name, object value)
        {
            Debug.Assert(name is string);
            if(int.TryParse(name as string, out var index))
            {
                if(value is not Trackable trackable)
                {
                    throw new TypeError("Cannot set an object which is not trackable to ListWrapper.");
                }
                if(Count <= index)
                {
                    Add(trackable);
                }
                else
                {
                    this[index] = trackable;
                }
            }
            else
            {
                throw new NotImplementedException("Encounter an unexpected behavior in <ListWrapper.SetAttr>, please " +
                    "submit an issue to https://github.com/SciSharp/TensorFlow.NET/issues");
            }
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

        public override ICollection<Trackable> Values => this;
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
            // TODO(Rinne): deal with `attribute_sentinel`.
            if (_external_modification_value || _non_append_mutation_value) return;
            _last_wrapped_list_snapshot = new List<Trackable>(_storage);
        }

        public override IDictionary<string, Trackable> _trackable_children(SaveType save_type, IDictionary<string, IDictionary<Trackable, ISerializedAttributes>>? cache = null)
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
                children = children.Concat(this.Where(x => x is Function or ConcreteFunction).Select((x, idx) => new KeyValuePair<string, Trackable>(idx.ToString(), x))).ToDictionary(x => x.Key, x => x.Value);
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
            catch(ValueError)
            {
                value = (Trackable)sticky_attribute_assignment(this, name, value);
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

        public void Clear()
        {
            _storage.Clear();
            update_snapshot();
        }

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

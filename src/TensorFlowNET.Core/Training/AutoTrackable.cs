using System.Collections.Generic;
using System.Linq;
using Tensorflow.Functions;
using Tensorflow.Keras.Saving.SavedModel;
using Tensorflow.Operations.Activation;
using Tensorflow.Training;
using static Tensorflow.Binding;

namespace Tensorflow.Train
{
    public class AutoTrackable : Trackable
    {
        public void _delete_tracking(string name)
        {
            _maybe_initialize_trackable();
            if (_unconditional_dependency_names.ContainsKey(name))
            {
                _unconditional_dependency_names.Remove(name);
                for (int i = _unconditional_checkpoint_dependencies.Count - 1; i >= 0; i--)
                {
                    if (_unconditional_checkpoint_dependencies[i].Name == name)
                    {
                        _unconditional_checkpoint_dependencies.RemoveAt(i);
                    }
                }
            }
        }

        public override void SetAttr(string name, object value)
        {
            // TODO(Rinne): deal with `self_setattr_tracking`.
            value = TrackableDataStructure.sticky_attribute_assignment(this, name, value);
            base.SetAttr(name, value);
        }

        public override IDictionary<string, Trackable> _trackable_children(SaveType save_type, IDictionary<string, IDictionary<Trackable, ISerializedAttributes>>? cache = null)
        {
            if(save_type != SaveType.SAVEDMODEL)
            {
                return base._trackable_children(save_type, cache);
            }

            Dictionary<string, Trackable> functions = new();
            // TODO: process of logs.
            // TODO(Rinne): deal with members.
            var properties = this.GetType().GetProperties();
            foreach ( var property in properties )
            {
                if(property.PropertyType == typeof(Function) || property.PropertyType == typeof(ConcreteFunction))
                {
                    string name = property.Name;
                    object value = property.GetValue(this, null);
                    functions[name] = (Trackable)value;
                }
            }

            foreach(var item in CustomizedFields)
            {
                var name = item.Key;
                var value = item.Value;
                if (value is Function or ConcreteFunction)
                {
                    functions[name] = (Trackable)value;
                }
            }

            // TODO: process the type `core_types.GenericFunction`.

            Dictionary<string, Trackable> children = new();
            foreach(var pair in CheckpointDependencies)
            {
                var name = pair.Name;
                var child = pair.Refer;
                if(child is ConcreteFunction) // or Generic function
                {
                    continue;
                }
                if(functions.ContainsKey(name) && functions[name] != child)
                {
                    throw new ValueError($"Can't save object because it has multiple children with the same " +
                        $"name. Object: {this}, attribute name: {name}, child 1: " +
                        $"{child}, child 2: {functions[name]}");
                }
                children[name] = child;
            }

            return children.Concat(functions).ToDictionary(x => x.Key, x => x.Value);
        }
    }
}

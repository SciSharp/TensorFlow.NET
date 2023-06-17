/*****************************************************************************
   Copyright 2018 The TensorFlow.NET Authors. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
******************************************************************************/

using OneOf;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Tensorflow.Checkpoint;
using Tensorflow.Keras.Saving.SavedModel;
using Tensorflow.Training;
using Tensorflow.Training.Saving.SavedModel;
using static Tensorflow.Binding;

namespace Tensorflow.Train
{
    public abstract class Trackable: IWithTrackable
    {
        /// <summary>
        /// Corresponding to tensorflow/python/trackable/constants.py
        /// </summary>
        public static class Constants
        {
            public static readonly string OBJECT_GRAPH_PROTO_KEY = "_CHECKPOINTABLE_OBJECT_GRAPH";
            public static readonly string VARIABLE_VALUE_KEY = "VARIABLE_VALUE";
            public static readonly string OBJECT_CONFIG_JSON_KEY = "OBJECT_CONFIG_JSON";
        }
        protected int _self_update_uid;
        protected IDictionary<string, Trackable> _unconditional_dependency_names;

        protected IList<TrackableReference> _unconditional_checkpoint_dependencies;
        protected Dictionary<string, IList<CheckpointPosition>> _unconditional_deferred_dependencies;

        protected IDictionary<string, Func<string, OneOf<BaseResourceVariable, MySaveableObject>>> _self_saveable_object_factories =
            new Dictionary<string, Func<string, OneOf<BaseResourceVariable, MySaveableObject>>>();
        private bool _manual_tracking = true;

        private static Trackable _none = new AutoTrackable();
        /// <summary>
        /// This is a trick for that CSharp does not allow the key of `Dictionary` to be null.
        /// The `None` can be any object that inherits `Trackable`.
        /// This Property is supposed to be used only internal.
        /// </summary>
        public static Trackable None
        {
            get
            {
                return _none;
            }
        }
        public Trackable GetTrackable()
        {
            return this;
        }
        public virtual string ObjectIdentifier
        {
            get => "_generic_user_object";
        }
        public int UpdateUid { get => _self_update_uid; set => _self_update_uid = value; }
        public IList<TrackableReference> UnconditionalCheckpointDependencies { get => _unconditional_checkpoint_dependencies; }
        public IDictionary<string, Trackable> UnconditionalDependencyNames { get => _unconditional_dependency_names; }
        public IList<TrackableReference> CheckpointDependencies { get => UnconditionalCheckpointDependencies; }
        public Dictionary<string, IList<CheckpointPosition>> DeferredDependencies => _unconditional_deferred_dependencies;
        public IDictionary<string, Func<string, OneOf<BaseResourceVariable, MySaveableObject>>> SelfSaveableObjectFactories
        {
            get
            {
                return _self_saveable_object_factories;
            }
            set
            {
                _self_saveable_object_factories = value;
            }
        }
        public Dictionary<string, object> CustomizedFields { get; set; } = new Dictionary<string, object>();

        public virtual void SetAttr(string name, object value)
        {
            var t = this.GetType();
            var field_info = t.GetField(name);
            if(field_info is not null)
            {
                field_info.SetValue(this, value);
            }
            else
            {
                CustomizedFields[name] = value;
            }

            // On account of performance, we don't use reflection to set the attribute if it exists in `Trackable`.
            // When adding new members or properties to this class, please add corresponding process to this method.
            //switch (name)
            //{
            //    case "_manual_tracking":
            //        {
            //            _manual_tracking = (bool)value;
            //            break;
            //        }
            //    case "_self_saveable_object_factories":
            //        {
            //            _self_saveable_object_factories = (IDictionary<string, Func<string, OneOf<BaseResourceVariable, MySaveableObject>>>)value;
            //            break;
            //        }
            //    case "_self_update_uid":
            //        {
            //            _self_update_uid = (int)value;
            //            break;
            //        }
            //    case "_unconditional_checkpoint_dependencies":
            //        {
            //            _unconditional_checkpoint_dependencies = (IList<TrackableReference>)value;
            //            break;
            //        }
            //    case "_unconditional_deferred_dependencies":
            //        {
            //            _unconditional_deferred_dependencies = (Dictionary<string, IList<CheckpointPosition>>)value;
            //            break;
            //        }
            //    case "_unconditional_dependency_names":
            //        {
            //            _unconditional_dependency_names = (IDictionary<string, Trackable>)value;
            //            break;
            //        }
            //    case "SelfSaveableObjectFactories":
            //        {
            //            SelfSaveableObjectFactories = (IDictionary<string, Func<string, OneOf<BaseResourceVariable, MySaveableObject>>>)value;
            //            break;
            //        }
            //    case "UpdateUid":
            //        {
            //            UpdateUid = (int)value;
            //            break;
            //        }
            //    default:
            //        {
            //            CustomizedAttributes[name] = value;
            //            break;
            //        }
            // }
        }

        /// <summary>
        /// Restore-on-create for a variable be saved with this `Checkpointable`.
        /// </summary>
        /// <returns></returns>
        protected virtual IVariableV1 _add_variable_with_custom_getter(VariableArgs args)
        {
            tf_with(ops.init_scope(), delegate
            {
#pragma warning disable CS0219 // Variable is assigned but its value is never used
                IInitializer checkpoint_initializer = null;
#pragma warning restore CS0219 // Variable is assigned but its value is never used
                if (tf.Context.executing_eagerly())
#pragma warning disable CS0642 // Possible mistaken empty statement
                    ;
#pragma warning restore CS0642 // Possible mistaken empty statement
                else
                    checkpoint_initializer = null;
            });

            var new_variable = args.Getter(args);

            // If we set an initializer and the variable processed it, tracking will not
            // assign again. It will add this variable to our dependencies, and if there
            // is a non-trivial restoration queued, it will handle that. This also
            // handles slot variables.
            if (!args.Overwrite || new_variable is RefVariable || new_variable is Trackable)
            {
                var res = _track_trackable(new_variable as Trackable, args.Name, args.Overwrite);
                Debug.Assert(res is IVariableV1);
                return res as IVariableV1;
            }
            else
                return new_variable;
        }

        /// <summary>
        /// Pop and load any deferred checkpoint restores into `trackable`.
        /// </summary>
        /// <param name="name"></param>
        /// <param name="trackable"></param>
        protected void _handle_deferred_dependencies(string name, IVariableV1 trackable)
        {
            _maybe_initialize_trackable();
            // TODO
        }

        protected IVariableV1 _track_checkpointable(IVariableV1 checkpointable, string name, bool overwrite = false)
        {
            return checkpointable;
        }

        /// <summary>
        /// Initialize dependency management.
        /// </summary>
        public void _maybe_initialize_trackable()
        {
            if(_unconditional_checkpoint_dependencies is not null)
            {
                return;
            }
            _self_update_uid = -1;
            _unconditional_checkpoint_dependencies = new List<TrackableReference>();
            _unconditional_dependency_names = new Dictionary<string, Trackable>();
            _unconditional_deferred_dependencies = new Dictionary<string, IList<CheckpointPosition>>();
        }

        public virtual IDictionary<string, Trackable> _trackable_children(SaveType save_type = SaveType.CHECKPOINT, 
            IDictionary<string, IDictionary<Trackable, ISerializedAttributes>>? cache = null)
        {
            _maybe_initialize_trackable();
            return _unconditional_checkpoint_dependencies.ToDictionary(x => x.Name, x => x.Refer);
        }

        public virtual Trackable _track_trackable(Trackable trackable, string name, bool overwrite = false)
        {
            _maybe_initialize_trackable();
            if (!_manual_tracking) return trackable;
            var new_reference = new TrackableReference(name, trackable);
            var current_object = _lookup_dependency(name);

            if(current_object is null)
            {
                _unconditional_checkpoint_dependencies.Add(new_reference);
                _handle_deferred_dependencies(name, trackable);
            }
            _unconditional_dependency_names[name] = trackable;
            return trackable;
        }

        /// <summary>
        /// Pop and load any deferred checkpoint restores into `trackable`.
        /// This method does not add a new dependency on `trackable`, but it does check if any outstanding/deferred dependencies have been queued waiting for
        /// this dependency to be added (matched based on `name`). If so, `trackable` and its dependencies are restored. The restorations are 
        /// considered fulfilled and so are deleted.
        /// `_track_trackable` is more appropriate for adding a normal/unconditional dependency, and includes handling for deferred restorations. 
        /// This method allows objects such as `Optimizer` to use the same restoration logic while managing conditional dependencies themselves,
        /// by overriding `_checkpoint_dependencies` and `_lookup_dependency` to change the object's dependencies based on the context
        /// it is saved/restored in (a single optimizer instance can have state associated with multiple graphs).
        /// </summary>
        /// <param name="name"></param>
        /// <param name="trackable"></param>
        public virtual void _handle_deferred_dependencies(string name, Trackable trackable)
        {
            _maybe_initialize_trackable();
            trackable._maybe_initialize_trackable();

            if(_unconditional_deferred_dependencies.TryGetValue(name, out var dependencies))
            {
                _unconditional_deferred_dependencies.Remove(name);
                foreach(var checkpoint_position in dependencies.OrderByDescending(x => x.Checkpoint.RestoreUid))
                {
                    checkpoint_position.restore(trackable);
                }
            }

            // TODO(Rinne): deal with `_self_name_based_restores`
        }

        public virtual Trackable? _lookup_dependency(string name)
        {
            if (_unconditional_dependency_names.TryGetValue(name, out var dependency)) return dependency;
            else return null;
        }

        public static Trackable convert_to_trackable(object obj, object? parent = null)
        {
            if (obj is Trackable)
            {
                return (Trackable)obj;
            }
            else
            {
                throw new NotImplementedException();
            }
        }

        public virtual IDictionary<string, Trackable> deserialization_dependencies(IDictionary<string, Trackable> children)
        {
            return new Dictionary<string, Trackable>();
        }

        public virtual (IDictionary<Trackable, Trackable>, IDictionary<Tensor, Tensor>) map_resources(
            SaveOptions? save_options)
        {
            return (new Dictionary<Trackable, Trackable>(), new Dictionary<Tensor, Tensor>());
        }

        public virtual List<Tensor> export_to_saved_model_graph(IDictionary<Trackable, Trackable> object_map,
            IDictionary<Tensor, Tensor> tensor_map, SaveOptions? options = null)
        {
            var (self_object_map, self_tensor_map) = map_resources(options);
            foreach (var pair in self_object_map)
            {
                object_map.Add(pair);
            }
            foreach (var pair in self_tensor_map)
            {
                tensor_map.Add(pair);
            }

            return self_tensor_map.Keys.ToList();
        }

        public virtual IDictionary<string, Func<string, OneOf<BaseResourceVariable, MySaveableObject>>> gather_saveables_for_checkpoint()
        {
            OneOf<BaseResourceVariable, MySaveableObject> create_saveable(string name = "")
            {
                throw new NotImplementedException();
                //return new TrackableSaveable(this, null, name, null, null);
            }
            if (saveable_object_util.trackable_has_serialize_to_tensor(this))
            {
                // TODO: complete the implementation (need to complete the class `saveable_object_util.TrackableSaveable`).
                Dictionary<string, Func<string, OneOf<BaseResourceVariable, MySaveableObject>>> res = new();
                res[""] = create_saveable;
                return res;
            }
            else
            {
                return _self_saveable_object_factories;
            }
        }

        /// <summary>
        /// Gathers tensors to save to the checkpoint. You should only override `serialize_to_tensors` and `restore_from_tensors`
        /// if you are defining a custom resource or variable with custom ops.
        /// Otherwise, please store the state of your trackable in `tf.Variable` objects
        /// and add them to Trackable object hierarchy using `setattr` (for subclasses
        /// of `AutoTrackable`) or overriding the `_trackable_children` method.
        /// </summary>
        /// <returns></returns>
        /// <exception cref="NotImplementedException"></exception>
        public virtual IDictionary<string, IDictionary<string, OneOf<Tensor, SaveSpec>>> serialize_to_tensors()
        {
            throw new NotImplementedException();
        }

        public virtual IDictionary<string, Operation> _restore_from_tensors(IDictionary<string, OneOf<Tensor, IDictionary<string, Tensor>>> restored_tensors)
        {
            throw new NotImplementedException();
        }
    }

    public record class TrackableReference(string Name, Trackable Refer);

    public record class SlotVariableRestoration(int OptimizerId, int SlotVariableId, string SlotName);
}

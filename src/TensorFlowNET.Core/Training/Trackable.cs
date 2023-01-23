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

using System;
using System.Collections.Generic;
using System.Linq;
using Tensorflow.ModelSaving;
using static Tensorflow.Binding;

namespace Tensorflow.Train
{
    public abstract class Trackable
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

        protected IDictionary<string, ResourceVariable> _self_saveable_object_factories =
            new Dictionary<string, ResourceVariable>();

        private static Trackable _none = new Function();
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
        public virtual string ObjectIdentifier
        {
            get => "_generic_user_object";
        }
        public int UpdateUid { get => _self_update_uid; set => _self_update_uid = value; }
        public IList<TrackableReference> UnconditionalCheckpointDependencies { get => _unconditional_checkpoint_dependencies; }
        public IDictionary<string, Trackable> UnconditionalDependencyNames { get => _unconditional_dependency_names; }
        public IList<TrackableReference> CheckpointDependencies { get => UnconditionalCheckpointDependencies; }

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
            if (!args.Overwrite || new_variable is RefVariable)
                return _track_checkpointable(new_variable, name: args.Name,
                                        overwrite: args.Overwrite);
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
            _self_update_uid = -1;
            _unconditional_checkpoint_dependencies = new List<TrackableReference>();
            _unconditional_dependency_names = new Dictionary<string, Trackable>();
        }

        // TODO: cache
        public virtual IDictionary<string, Trackable> _trackable_children(SaveType save_type, IDictionary<string, object>? cache = null)
        {
            _maybe_initialize_trackable();
            return _unconditional_checkpoint_dependencies.ToDictionary(x => x.Name, x => x.Refer);
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

        public virtual List<Tensor> export_to_saved_model_graph(IDictionary<Trackable, Trackable>? object_map = null,
            IDictionary<Tensor, Tensor>? tensor_map = null, SaveOptions? options = null)
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

        public virtual IDictionary<string, ResourceVariable> gather_saveables_for_checkpoint()
        {
            return _self_saveable_object_factories;
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
        public virtual IDictionary<string, object> serialize_to_tensors()
        {
            throw new NotImplementedException();
        }
    }

    public record class TrackableReference(string Name, Trackable Refer);
}

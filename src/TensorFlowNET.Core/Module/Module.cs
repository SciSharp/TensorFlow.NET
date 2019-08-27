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
using System.Text.RegularExpressions;
using Tensorflow;
using Tensorflow.Train;
using static Tensorflow.Binding;

namespace Tensorflow.Module
{
    /// <summary>
    /// Base neural network module class.
    /// A module is a named container for `tf.Variable`s, other `tf.Module`s and
    /// functions which apply to user input. For example a dense layer in a neural
    /// network might be implemented as a `tf.Module`:
    /// 
    /// tensorflow/python/module/module.py
    /// </summary>
    public class Module : AutoTrackable
    {
        SortedSet<string> _TF_MODULE_IGNORED_PROPERTIES = new SortedSet<string>(){ "_self_unconditional_checkpoint_dependencies", "_self_unconditional_dependency_names"};

        protected string _name;
        /// <summary>
        /// Returns the name of this module as passed or determined in the ctor.
        /// NOTE: This is not the same as the `self.name_scope.name` which includes
        /// parent module names.
        /// </summary>
        public string name => _name;

        protected ops.NameScope _name_scope;
        protected ops.NameScope _scope_name;
        /// <summary>
        /// Returns a `tf.name_scope` instance for this class.
        /// </summary>
        public ops.NameScope name_scope
        {
            get{
                if(tf2.enabled())
                    return this._name_scope;
                return ops.name_scope(_scope_name);
            }
        }
        /// <summary>
        /// Sequence of variables owned by this module and it's submodules.
        /// Note: this method uses reflection to find variables on the current instance
        /// and submodules. For performance reasons you may wish to cache the result
        /// of calling this method if you don't expect the return value to change.
        /// </summary>
        /// <returns>
        ///  A sequence of variables for the current module (sorted by attribute
        ///  name) followed by variables from all submodules recursively (breadth
        ///  first).
        /// </returns>
        public ValueTuple variables => throw new NotImplementedException();

        /// <summary>
        /// Sequence of variables owned by this module and it's submodules.
        /// Note: this method uses reflection to find variables on the current instance
        /// and submodules. For performance reasons you may wish to cache the result
        /// of calling this method if you don't expect the return value to change.
        /// </summary>
        /// <returns>
        /// A sequence of variables for the current module (sorted by attribute
        /// name) followed by variables from all submodules recursively (breadth
        /// first).
        /// </returns>
        public ValueTuple trainable_variables => throw new NotImplementedException();

        /// <summary>
        /// Sequence of all sub-modules.
        /// Submodules are modules which are properties of this module, or found as
        /// properties of modules which are properties of this module (and so on).
        /// </summary>
        /// <returns>
        /// A sequence of all submodules.
        /// </returns>
        public ValueTuple submodules => throw new NotImplementedException();

        public Module(string name = null)
        {
            if(name == null)
                name = Module.camel_to_snake(this.GetType().Name);
            else
                if (!valid_identifier(name))
                    new ValueError(name +
                        " is not a valid module name. Module names must be valid Python " +
                        "identifiers (e.g. a valid class name).");
            this._name = name;
            //if (tf2.enabled())
            //    using( var scope_name = ops.name_scope_v2(name) )
            //        this._name_scope = ops.name_scope_v2(scope_name);
            //else
                using(var scope_name = ops.name_scope(name))
                    this._scope_name = scope_name;
        }

        public object with_name_scope(Func<object> method)
        {
            throw new NotImplementedException();
        }

        // NOTE : Below are all the static functions
        static string _CAMEL_TO_SNAKE_R = "((?<=[a-z0-9])[A-Z]|(?!^)[A-Z](?=[a-z]))";
        public static string camel_to_snake(string value)
        {
            return Regex.Match(value, _CAMEL_TO_SNAKE_R).Result("_${1}").ToLower();
        }
        static string _VALID_IDENTIFIER = "^[a-zA-Z_]([a-zA-Z0-9_])*$";
        public static bool valid_identifier(string name)
        {
            return Regex.Match(name, _VALID_IDENTIFIER).Success;
        }
        public static Array _flatten_module(Module module,
                    bool recursive,
                    object predicate,
                    object attribute_traversal_key,
                    object attributes_to_ignore,
                    bool with_path,
                    ValueTuple module_path = default(ValueTuple),
                    bool seen=false)
                    {
                        throw new System.NotImplementedException();
                    }
    }
}

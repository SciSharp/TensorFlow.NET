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

using System.Collections.Generic;
using static Tensorflow.Binding;

namespace Tensorflow
{
    /// <summary>
    /// Variable scope object to carry defaults to provide to `get_variable`
    /// </summary>
    public class VariableScope
    {
        public bool use_resource { get; set; }
#pragma warning disable CS0414 // The field 'VariableScope._reuse' is assigned but its value is never used
        private _ReuseMode _reuse;
#pragma warning restore CS0414 // The field 'VariableScope._reuse' is assigned but its value is never used
        public bool resue;

        private TF_DataType _dtype;
        string _name;
        public string name => _name;
        public string _name_scope { get; set; }
        public string original_name_scope => _name_scope;

        public VariableScope(bool reuse,
            string name = "",
            string name_scope = "",
            TF_DataType dtype = TF_DataType.TF_FLOAT)
        {
            _name = name;
            _name_scope = name_scope;
            _reuse = _ReuseMode.AUTO_REUSE;
            _dtype = dtype;
        }

        public IVariableV1 get_variable(_VariableStore var_store,
            string name,
            Shape shape = null,
            TF_DataType dtype = TF_DataType.DtInvalid,
            object initializer = null, // IInitializer or Tensor
            bool? trainable = null,
            List<string> collections = null,
            bool? use_resource = null,
            bool validate_shape = true,
            VariableSynchronization synchronization = VariableSynchronization.Auto,
            VariableAggregation aggregation = VariableAggregation.None)
        {
            string full_name = !string.IsNullOrEmpty(this.name) ? this.name + "/" + name : name;
            return tf_with(ops.name_scope(null), scope =>
            {
                if (dtype == TF_DataType.DtInvalid)
                    dtype = _dtype;

                return var_store.get_variable(full_name,
                    shape: shape,
                    dtype: dtype,
                    initializer: initializer,
                    reuse: resue,
                    trainable: trainable,
                    collections: collections,
                    synchronization: synchronization,
                    aggregation: aggregation);
            });
        }

        public void reuse_variables()
        {
            _reuse = _ReuseMode.AUTO_REUSE;
        }
    }
}

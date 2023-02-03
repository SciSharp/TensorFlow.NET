/*****************************************************************************
   Copyright 2020 The TensorFlow.NET Authors. All Rights Reserved.

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

using Tensorflow.NumPy;

namespace Tensorflow
{
    /// <summary>
    /// A variable maintains state in the graph across calls to `run()`. You add a
    /// variable to the graph by constructing an instance of the class `Variable`.
    /// 
    /// The `Variable()` constructor requires an initial value for the variable,
    /// which can be a `Tensor` of any type and shape. The initial value defines the
    /// type and shape of the variable. After construction, the type and shape of
    /// the variable are fixed. The value can be changed using one of the assign methods.
    /// https://tensorflow.org/guide/variables
    /// </summary>
    public interface IVariableV1
    {
        string UniqueId { get; }
        string Name { get; }
        /// <summary>
        /// Handle is ref type
        /// </summary>
        Tensor Handle { get; }
        string Device { get; }
        Operation Initializer { get; }
        Operation Op { get; }
        /// <summary>
        /// GraphElement is a copy of Handle
        /// </summary>
        Tensor GraphElement { get; }
        Graph Graph { get; }
        TF_DataType dtype { get; }
        Shape shape { get; }
        bool Trainable { get; }
        Tensor assign_add<T>(T delta, bool use_locking = false, string name = null, bool read_value = true);
        Tensor assign_sub<T>(T delta, bool use_locking = false, string name = null, bool read_value = true);
        IVariableV1 assign_sub_lazy_load(Tensor delta, string name = null);
        Tensor assign<T>(T value, bool use_locking = false, string name = null, bool read_value = true);
        IVariableV1 assign_lazy_load(Tensor value, string name = null);
        Tensor AsTensor(TF_DataType dtype = TF_DataType.DtInvalid, string name = null, bool as_ref = false);
        NDArray numpy();
    }
}

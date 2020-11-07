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

using Tensorflow.Contexts;
using static Tensorflow.Binding;

namespace Tensorflow
{
    public class io_ops
    {
        public Operation save_v2(Tensor prefix, string[] tensor_names, string[] shape_and_slices, Tensor[] tensors, string name = null)
        {
            var _op = tf.OpDefLib._apply_op_helper("SaveV2", name: name, args: new { prefix, tensor_names, shape_and_slices, tensors });

            return _op;
        }

        public Tensor[] restore_v2(Tensor prefix, string[] tensor_names, string[] shape_and_slices, TF_DataType[] dtypes, string name = null)
        {
            var _op = tf.OpDefLib._apply_op_helper("RestoreV2", name: name, args: new { prefix, tensor_names, shape_and_slices, dtypes });

            return _op.outputs;
        }

        public Tensor read_file<T>(T filename, string name = null)
        {
            if (tf.Context.executing_eagerly())
            {
                return read_file_eager_fallback(filename, name: name, tf.Context);
            }

            var _op = tf.OpDefLib._apply_op_helper("ReadFile", name: name, args: new { filename });

            return _op.outputs[0];
        }

        private Tensor read_file_eager_fallback<T>(T filename, string name = null, Context ctx = null)
        {
            var filename_tensor = ops.convert_to_tensor(filename, TF_DataType.TF_STRING);
            var _inputs_flat = new[] { filename_tensor };

            return tf.Runner.Execute(ctx, "ReadFile", 1, _inputs_flat, null, name: name)[0];
        }
    }
}

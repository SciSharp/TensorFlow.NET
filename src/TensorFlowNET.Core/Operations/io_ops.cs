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

using System.Linq;
using Tensorflow.Contexts;
using Tensorflow.Eager;
using static Tensorflow.Binding;

namespace Tensorflow
{
    public class io_ops
    {
        public Operation save_v2(Tensor prefix, string[] tensor_names, string[] shape_and_slices, Tensor[] tensors, string name = null)
        {
            var ctx = tf.Context;
            if (ctx.executing_eagerly())
            {
                try
                {
                    var result = tf.Runner.TFE_FastPathExecute(
                        new FastPathOpExecInfo(tf.Context, "SaveV2", name, new object[] { prefix, tensor_names, shape_and_slices, tensors }));
                    result = null;
                    return null;
                }
                catch (System.Exception)
                {
                    return save_v2_eager_fallback(prefix, tensor_names, shape_and_slices, tensors, name, ctx);
                }
            }
            var _op = tf.OpDefLib._apply_op_helper("SaveV2", name: name, args: new { prefix, tensor_names, shape_and_slices, tensors });

            return _op;
        }

        public Operation save_v2_eager_fallback(Tensor prefix, string[] tensor_names, string[] shape_and_slices, Tensor[] tensors, string name, Context ctx)
        {
            DataType[] attr_dtypes;
            (attr_dtypes, tensors) = _execute.onvert_to_mixed_eager_tensors(tensors, ctx);
            prefix = ops.convert_to_tensor(prefix, TF_DataType.TF_STRING);
            var tensor_names_tensor = ops.convert_to_tensor(tensor_names, TF_DataType.TF_STRING);
            var shape_and_slices_tensor = ops.convert_to_tensor(shape_and_slices, TF_DataType.TF_STRING);
            var inputs_flat = tensors.Concat(new Tensor[] { prefix, tensor_names_tensor, shape_and_slices_tensor }).ToArray();
            var attrs = new object[] { "dtypes", attr_dtypes };

            var result = _execute.quick_execute("SaveV2", 0, inputs_flat, attrs, ctx, name);
            result = null;
            return null;
        }

        public Tensor[] restore_v2(Tensor prefix, string[] tensor_names, string[] shape_and_slices, TF_DataType[] dtypes, string name = null)
        {
            // Note: this implementation is not correct in many cases, please consider using `gen_ops.restore_v2`.
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

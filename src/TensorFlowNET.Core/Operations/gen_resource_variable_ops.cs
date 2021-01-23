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

using static Tensorflow.Binding;

namespace Tensorflow
{
    public static class gen_resource_variable_ops
    {
        public static Operation assign_sub_variable_op(Tensor resource, Tensor value, string name = null)
        {
            if (tf.Context.executing_eagerly())
            {
                var results = tf.Runner.TFE_FastPathExecute(tf.Context, tf.Context.DeviceName,
                    "AssignSubVariableOp", name,
                    null,
                    resource, value);

                return null;
            }

            return null;
        }

        /// <summary>
        /// Adds a value to the current value of a variable.
        /// </summary>
        /// <param name="resource"></param>
        /// <param name="value"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Operation assign_add_variable_op(Tensor resource, Tensor value, string name = null)
        {
            if (tf.Context.executing_eagerly())
            {
                var results = tf.Runner.TFE_FastPathExecute(tf.Context, tf.Context.DeviceName,
                    "AssignAddVariableOp", name,
                    null,
                    resource, value);

                return null;
            }

            var _op = tf.OpDefLib._apply_op_helper("AssignAddVariableOp", name, new { resource, value });

            return _op;
        }

        public static Operation assign_variable_op(Tensor resource, Tensor value, string name = null)
        {
            if (tf.Context.executing_eagerly())
            {
                tf.Runner.TFE_FastPathExecute(tf.Context, tf.Context.DeviceName,
                    "AssignVariableOp", name,
                    null,
                    resource, value);

                return null;
            }

            var _op = tf.OpDefLib._apply_op_helper("AssignVariableOp", name, new { resource, value });

            return _op;
        }

        public static Tensor var_is_initialized_op(Tensor resource, string name = null)
        {
            if (tf.Context.executing_eagerly())
            {
                var results = tf.Runner.TFE_FastPathExecute(tf.Context, tf.Context.DeviceName,
                    "VarIsInitializedOp", name,
                    null,
                    resource);

                return results[0];
            }

            var _op = tf.OpDefLib._apply_op_helper("VarIsInitializedOp", name, new { resource });

            return _op.output;
        }

        /// <summary>
        /// Creates a handle to a Variable resource.
        /// </summary>
        /// <param name="dtype"></param>
        /// <param name="shape"></param>
        /// <param name="container"></param>
        /// <param name="shared_name"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor var_handle_op(TF_DataType dtype, TensorShape shape,
            string container = "", string shared_name = "", string name = null)
        {
            if (tf.Context.executing_eagerly())
            {
                var results = tf.Runner.TFE_FastPathExecute(tf.Context, tf.Context.DeviceName,
                    "VarHandleOp", name,
                    null,
                    "container", container,
                    "shared_name", shared_name,
                    "dtype", dtype,
                    "shape", shape.dims,
                    "allowed_devices", new string[0]);

                return results[0];
            }

            var _op = tf.OpDefLib._apply_op_helper("VarHandleOp", name, new
            {
                dtype,
                shape,
                container,
                shared_name
            });

            return _op.output;
        }

        public static Tensor destroy_resource_op(Tensor resource, bool ignore_lookup_error = true, string name = null)
        {
            if (tf.Context.executing_eagerly())
            {
                var results = tf.Runner.TFE_FastPathExecute(tf.Context, tf.Context.DeviceName,
                    "DestroyResourceOp", name,
                    null,
                    resource,
                    "ignore_lookup_error", ignore_lookup_error);

                return results.Length == 0 ? null : results[0];
            }

            var _op = tf.OpDefLib._apply_op_helper("DestroyResourceOp", name, new
            {
                resource,
                ignore_lookup_error
            });

            return _op.output;
        }

        /// <summary>
        /// Reads the value of a variable.
        /// </summary>
        /// <param name="resource"></param>
        /// <param name="dtype"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor read_variable_op(Tensor resource, TF_DataType dtype, string name = null)
        {
            if (tf.executing_eagerly())
            {
                var results = tf.Runner.TFE_FastPathExecute(tf.Context, tf.Context.DeviceName,
                    "ReadVariableOp", name,
                    null,
                    resource,
                    "dtype", dtype);

                return results[0];
            }

            var _op = tf.OpDefLib._apply_op_helper("ReadVariableOp", name, new
            {
                resource,
                dtype
            });

            return _op.output;
        }

        public static Tensor resource_gather(Tensor resource, Tensor indices, TF_DataType dtype,
            int batch_dims = 0, bool validate_indices = true, string name = null)
        {
            var _op = tf.OpDefLib._apply_op_helper("ResourceGather", name, new
            {
                resource,
                indices,
                dtype,
                batch_dims,
                validate_indices
            });

            return _op.output;
        }
    }
}

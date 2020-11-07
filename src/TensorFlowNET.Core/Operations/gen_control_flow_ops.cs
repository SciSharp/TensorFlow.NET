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

using Tensorflow.Operations;
using static Tensorflow.Binding;

namespace Tensorflow
{
    public class gen_control_flow_ops
    {
        public static Operation control_trigger(string name = null)
        {
            var _op = tf.OpDefLib._apply_op_helper("ControlTrigger", name, new
            {
            });

            return _op;
        }

        /// <summary>
        /// Creates or finds a child frame, and makes `data` available to the child frame.
        /// </summary>
        /// <param name="data"></param>
        /// <param name="frame_name"></param>
        /// <param name="is_constant"></param>
        /// <param name="parallel_iterations"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor enter(Tensor data, string frame_name = "frame_name", bool is_constant = false, int parallel_iterations = 10, string name = null)
        {
            var _op = tf.OpDefLib._apply_op_helper("Enter", name, new
            {
                data,
                frame_name,
                is_constant,
                parallel_iterations
            });

            return _op.output;
        }

        /// <summary>
        /// Forwards the input to the output.
        /// </summary>
        /// <param name="input"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor loop_cond(Tensor input, string name = null)
        {
            var _op = tf.OpDefLib._apply_op_helper("LoopCond", name, new { input });

            return _op.output;
        }

        /// <summary>
        /// Makes its input available to the next iteration.
        /// </summary>
        /// <param name="data"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor ref_next_iteration(Tensor data, string name = null)
        {
            var _op = tf.OpDefLib._apply_op_helper("RefNextIteration", name, new { data });

            return _op;
        }

        /// <summary>
        /// Makes its input available to the next iteration.
        /// </summary>
        /// <param name="data"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor next_iteration(Tensor data, string name = null)
        {
            var _op = tf.OpDefLib._apply_op_helper("NextIteration", name, new { data });

            return _op;
        }

        /// <summary>
        /// Exits the current frame to its parent frame.
        /// </summary>
        /// <param name="data"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor ref_exit(Tensor data, string name = null)
        {
            var _op = tf.OpDefLib._apply_op_helper("RefExit", name, new { data });

            return _op;
        }

        /// <summary>
        /// Exits the current frame to its parent frame.
        /// </summary>
        /// <param name="data"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor _exit(Tensor data, string name = null)
        {
            var _op = tf.OpDefLib._apply_op_helper("Exit", name, new { data });

            return _op;
        }

        public static Operation no_op(string name = null)
        {
            var _op = tf.OpDefLib._apply_op_helper("NoOp", name, null);

            return _op;
        }

        public static Tensor[] ref_switch(Tensor data, Tensor pred, string name = null)
        {
            var _op = tf.OpDefLib._apply_op_helper("RefSwitch", name, new { data, pred });
            return _op.outputs;
        }

        /// <summary>
        /// Forwards `data` to the output port determined by `pred`.
        /// 
        /// If `pred` is true, the `data` input is forwarded to `output_true`. Otherwise,
        /// the data goes to `output_false`.
        /// 
        /// See also `RefSwitch` and `Merge`.
        /// </summary>
        /// <param name="data">A `Tensor`. The tensor to be forwarded to the appropriate output.</param>
        /// <param name="pred">A `Tensor` of type `bool`.
        /// A scalar that specifies which output port will receive data.
        /// </param>
        /// <param name="name"> A name for the operation (optional).</param>
        /// <returns>A tuple of `Tensor` objects (output_false, output_true).
        /// 
        /// output_false: A `Tensor`. Has the same type as `data`.
        /// output_true: A `Tensor`. Has the same type as `data`.
        /// </returns>
        public static Tensor[] @switch(Tensor data, Tensor pred, string name = null)
        {
            var _op = tf.OpDefLib._apply_op_helper("Switch", name, new { data, pred });
            var _inputs_flat = _op.inputs;
#pragma warning disable CS0219 // Variable is assigned but its value is never used
            var _attrs = ("T", _op.get_attr("T"));
#pragma warning restore CS0219 // Variable is assigned but its value is never used
            // TODO: missing original code
            //_execute.record_gradient("Switch", _inputs_flat, _attrs, _result, name);
            return new[] { _op.outputs[0], _op.outputs[1] };
        }

        public static MergeOutput ref_merge(Tensor[] inputs, string name = null)
        {
            var _op = tf.OpDefLib._apply_op_helper("RefMerge", name, new { inputs });

            return new MergeOutput(_op.outputs);
        }

        public static MergeOutput merge(Tensor[] inputs, string name = null)
        {
            var _op = tf.OpDefLib._apply_op_helper("Merge", name, new { inputs });

            return new MergeOutput(_op.outputs);
        }
    }
}

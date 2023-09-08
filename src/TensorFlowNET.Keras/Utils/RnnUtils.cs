using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using Tensorflow.Common.Types;
using Tensorflow.Keras.Layers;
using Tensorflow.Common.Extensions;

namespace Tensorflow.Keras.Utils
{
    internal static class RnnUtils
    {
        internal static Tensors generate_zero_filled_state(Tensor batch_size_tensor, INestStructure<long> state_size, TF_DataType dtype)
        {
            Func<long, Tensor> create_zeros = (unnested_state_size) =>
            {
                var flat_dims = new Shape(unnested_state_size).dims;
                var init_state_size = new Tensor[] { batch_size_tensor }.
                    Concat(flat_dims.Select(x => tf.constant(x, dtypes.int32))).ToArray();
                return array_ops.zeros(init_state_size, dtype: dtype);
            };

            // TODO(Rinne): map structure with nested tensors.
            if(state_size.TotalNestedCount > 1)
            {
                return new Tensors(state_size.Flatten().Select(s => create_zeros(s)).ToArray());
            }
            else
            {
                return create_zeros(state_size.Flatten().First());
            }

        }

        internal static Tensors generate_zero_filled_state_for_cell(IRnnCell cell, Tensors inputs, Tensor batch_size, TF_DataType dtype)
        {
            if (inputs is not null)
            {
                batch_size = array_ops.shape(inputs)[0];
                dtype = inputs.dtype;
            }
            return generate_zero_filled_state(batch_size, cell.StateSize, dtype);
        }

        /// <summary>
        /// Standardizes `__call__` to a single list of tensor inputs.
        /// 
        /// When running a model loaded from a file, the input tensors
        /// `initial_state` and `constants` can be passed to `RNN.__call__()` as part
        /// of `inputs` instead of by the dedicated keyword arguments.This method
        /// makes sure the arguments are separated and that `initial_state` and
        /// `constants` are lists of tensors(or None).
        /// </summary>
        /// <param name="inputs">Tensor or list/tuple of tensors. which may include constants
        /// and initial states.In that case `num_constant` must be specified.</param>
        /// <param name="initial_state">Tensor or list of tensors or None, initial states.</param>
        /// <param name="constants">Tensor or list of tensors or None, constant tensors.</param>
        /// <param name="num_constants">Expected number of constants (if constants are passed as
        /// part of the `inputs` list.</param>
        /// <returns></returns>
        internal static (Tensors, Tensors, Tensors) standardize_args(Tensors inputs, Tensors initial_state, Tensors constants, int num_constants)
        {
            if(inputs.Length > 1)
            {
                // There are several situations here:
                // In the graph mode, __call__ will be only called once. The initial_state
                // and constants could be in inputs (from file loading).
                // In the eager mode, __call__ will be called twice, once during
                // rnn_layer(inputs=input_t, constants=c_t, ...), and second time will be
                // model.fit/train_on_batch/predict with real np data. In the second case,
                // the inputs will contain initial_state and constants as eager tensor.
                //
                // For either case, the real input is the first item in the list, which
                // could be a nested structure itself. Then followed by initial_states, which
                // could be a list of items, or list of list if the initial_state is complex
                // structure, and finally followed by constants which is a flat list.
                Debug.Assert(initial_state is null && constants is null);
                if(num_constants > 0)
                {
                    constants = inputs.TakeLast(num_constants).ToArray().ToTensors();
                    inputs = inputs.SkipLast(num_constants).ToArray().ToTensors();
                }
                if(inputs.Length > 1)
                {
                    initial_state = inputs.Skip(1).ToArray().ToTensors();
                    inputs = inputs.Take(1).ToArray().ToTensors();
                }
            }

            return (inputs, initial_state, constants);
        }

        /// <summary>
        /// Check whether the state_size contains multiple states.
        /// </summary>
        /// <param name="state_size"></param>
        /// <returns></returns>
        public static bool is_multiple_state(INestStructure<long> state_size)
        {
            return state_size.TotalNestedCount > 1;
        }
    }
}

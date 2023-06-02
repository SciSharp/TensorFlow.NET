using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Util;
using OneOf;
using Tensorflow.NumPy;

namespace Tensorflow.Keras.Layers.Rnn
{
    public class RNNUtils
    {
        public static Tensor generate_zero_filled_state(Tensor batch_size_tensor, StateSizeWrapper state_size, TF_DataType dtype = TF_DataType.TF_FLOAT)
        {
            if (batch_size_tensor == null || dtype == null)
            {
                throw new ValueError(
                    "batch_size and dtype cannot be None while constructing initial " +
                    $"state. Received: batch_size={batch_size_tensor}, dtype={dtype}");
            }

            Func<StateSizeWrapper, Tensor> create_zeros;
            create_zeros = (StateSizeWrapper unnested_state_size) =>
            {
                var flat_dims = unnested_state_size.state_size;
                //if (unnested_state_size is int[])
                //{
                //    flat_dims = new Shape(unnested_state_size.AsT0).as_int_list();
                //}
                //else if (unnested_state_size.IsT1)
                //{
                //    flat_dims = new Shape(unnested_state_size.AsT1).as_int_list();
                //}
                var init_state_size = batch_size_tensor.ToArray<int>().concat(flat_dims);
                return tf.zeros(init_state_size, dtype: dtype);
            };
            
            //if (nest.is_nested(state_size))
            //{
            //    return nest.map_structure(create_zeros, state_size);
            //}
            //else
            //{
            //    return create_zeros(state_size);
            //}
            return create_zeros(state_size);
            
        }

        public static Tensor generate_zero_filled_state_for_cell(SimpleRNNCell cell, Tensors inputs, Tensor batch_size, TF_DataType dtype)
        {
            if (inputs != null)
            {
                batch_size = tf.shape(inputs)[0];
                dtype = inputs.dtype;
            }
            return generate_zero_filled_state(batch_size, cell.state_size, dtype);
        }
    }
}

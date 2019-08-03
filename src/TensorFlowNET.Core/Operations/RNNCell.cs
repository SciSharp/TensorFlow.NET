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
using Tensorflow.Operations;
using Tensorflow.Util;
using static Tensorflow.Python;

namespace Tensorflow
{
    /// <summary>
    /// Abstract object representing an RNN cell.
    /// 
    /// Every `RNNCell` must have the properties below and implement `call` with
    /// the signature `(output, next_state) = call(input, state)`.  The optional
    /// third input argument, `scope`, is allowed for backwards compatibility
    /// purposes; but should be left off for new subclasses.
    /// 
    /// This definition of cell differs from the definition used in the literature.
    /// In the literature, 'cell' refers to an object with a single scalar output.
    /// This definition refers to a horizontal array of such units.
    /// 
    /// An RNN cell, in the most abstract setting, is anything that has
    /// a state and performs some operation that takes a matrix of inputs.
    /// This operation results in an output matrix with `self.output_size` columns.
    /// If `self.state_size` is an integer, this operation also results in a new
    /// state matrix with `self.state_size` columns.  If `self.state_size` is a
    /// (possibly nested tuple of) TensorShape object(s), then it should return a
    /// matching structure of Tensors having shape `[batch_size].concatenate(s)`
    /// for each `s` in `self.batch_size`.
    /// </summary>
    public abstract class RNNCell : Layers.Layer
    {
        /// <summary>
        /// Attribute that indicates whether the cell is a TF RNN cell, due the slight
        /// difference between TF and Keras RNN cell.
        /// </summary>
        protected bool _is_tf_rnn_cell = false;
        public virtual int state_size { get; }

        public virtual int output_size { get; }

        public RNNCell(bool trainable = true,
            string name = null,
            TF_DataType dtype = TF_DataType.DtInvalid,
            bool? _reuse = null) : base(trainable: trainable, 
                    name: name, 
                    dtype: dtype,
                    _reuse: _reuse)
        {
            _is_tf_rnn_cell = true;
        }

        public virtual Tensor get_initial_state(Tensor inputs = null, Tensor batch_size = null, TF_DataType dtype = TF_DataType.DtInvalid)
        {
            if (inputs != null)
                throw new NotImplementedException("get_initial_state input is not null");

            return zero_state(batch_size, dtype);
        }

        /// <summary>
        /// Return zero-filled state tensor(s).
        /// </summary>
        /// <param name="batch_size"></param>
        /// <param name="dtype"></param>
        /// <returns></returns>
        public Tensor zero_state(Tensor batch_size, TF_DataType dtype)
        {
            Tensor output = null;
            var state_size = this.state_size;
            tf_with(ops.name_scope($"{this.GetType().Name}ZeroState", values: new { batch_size }), delegate
            {
                output = _zero_state_tensors(state_size, batch_size, dtype);
            });

            return output;
        }

        private Tensor _zero_state_tensors(int state_size, Tensor batch_size, TF_DataType dtype)
        {
            var output = nest.map_structure(s =>
            {
                var c = rnn_cell_impl._concat(batch_size, s);
                var size = array_ops.zeros(c, dtype: dtype);

                var c_static = rnn_cell_impl._concat(batch_size, s, @static: true);
                size.set_shape(c_static);

                return size;
            }, state_size);

            return output;
        }
    }
}

using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Common.Types;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.ArgsDefinition.Rnn;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Utils;

namespace Tensorflow.Keras.Layers.Rnn
{
    public abstract class RnnCellBase: Layer, IRnnCell
    {
        public RnnCellBase(LayerArgs args) : base(args) { }
        public abstract GeneralizedTensorShape StateSize { get; }
        public abstract GeneralizedTensorShape OutputSize { get; }
        public abstract bool SupportOptionalArgs { get; }
        public abstract (Tensor, Tensors) Call(Tensors inputs, Tensors states, bool? training = null);
        public virtual Tensors GetInitialState(Tensors inputs, long batch_size, TF_DataType dtype)
        {
            return RnnUtils.generate_zero_filled_state_for_cell(this, inputs, batch_size, dtype);
        }
    }
}

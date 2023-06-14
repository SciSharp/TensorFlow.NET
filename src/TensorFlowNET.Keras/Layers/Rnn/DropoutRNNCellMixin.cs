﻿using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Common.Types;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;

namespace Tensorflow.Keras.Layers.Rnn
{
    public abstract class DropoutRNNCellMixin: RnnCellBase
    {
        public float dropout;
        public float recurrent_dropout;
        // TODO(Rinne): deal with cache.
        public DropoutRNNCellMixin(LayerArgs args): base(args)
        {

        }

        protected void _create_non_trackable_mask_cache()
        {
            
        }

        public void reset_dropout_mask()
        {

        }

        public void reset_recurrent_dropout_mask()
        {

        }

        public Tensors? get_dropout_maskcell_for_cell(Tensors input, bool training, int count = 1)
        {
            if (dropout == 0f)
                return null;
            return _generate_dropout_mask(
                tf.ones_like(input),
                dropout,
                training,
                count);
        }

        // Get the recurrent dropout mask for RNN cell.
        public Tensors? get_recurrent_dropout_maskcell_for_cell(Tensors input, bool training, int count = 1)
        {
            if (dropout == 0f)
                return null;
            return _generate_dropout_mask(
                tf.ones_like(input),
                recurrent_dropout,
                training,
                count);
        }

        public Tensors _create_dropout_mask(Tensors input, bool training, int count = 1)
        {
            return _generate_dropout_mask(
                tf.ones_like(input),
                dropout,
                training,
                count);
        }

        public Tensors _create_recurrent_dropout_mask(Tensors input, bool training, int count = 1)
        {
            return _generate_dropout_mask(
                tf.ones_like(input),
                recurrent_dropout,
                training,
                count);
        }

        public Tensors _generate_dropout_mask(Tensor ones, float rate, bool training, int count = 1)
        {
            Tensors dropped_inputs()
            {
                DropoutArgs args = new DropoutArgs();
                args.Rate = rate;
                var DropoutLayer = new Dropout(args);
                var mask = DropoutLayer.Apply(ones, training: training);
                return mask;
            }

            if (count > 1)
            {
                Tensors results = new Tensors();
                for (int i = 0; i < count; i++)
                {
                    results.Add(dropped_inputs());
                }
                return results;
            }

            return dropped_inputs();
        }
    }
}

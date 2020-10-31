using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Losses;
using Tensorflow.Keras.Metrics;

namespace Tensorflow.Keras.Engine
{
    public class LossesContainer : Container
    {
        ILossFunc _user_losses;
        ILossFunc _losses;
        Mean _loss_metric;

        public LossesContainer(ILossFunc losses, string[] output_names = null) 
            : base(output_names)
        {
            _user_losses = losses;
            _losses = losses;
            _loss_metric = new Mean(name: "loss");
            _built = false;
        }

        /// <summary>
        /// Computes the overall loss.
        /// </summary>
        /// <param name="y_true"></param>
        /// <param name="y_pred"></param>
        public void Apply(Tensor y_true, Tensor y_pred)
        {

        }

        public void Build()
        {

        }
    }
}

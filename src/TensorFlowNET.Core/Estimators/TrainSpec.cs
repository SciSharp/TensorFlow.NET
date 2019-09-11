using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Estimators
{
    public class TrainSpec
    {
        public int max_steps { get; set; }

        public TrainSpec(Action input_fn, int max_steps)
        {
            this.max_steps = max_steps;
        }
    }
}

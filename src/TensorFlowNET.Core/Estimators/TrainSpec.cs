using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Estimators
{
    public class TrainSpec
    {
        int _max_steps;
        public int max_steps => _max_steps;

        Action _input_fn;
        public Action input_fn => _input_fn;

        public TrainSpec(Action input_fn, int max_steps)
        {
            _max_steps = max_steps;
            _input_fn = input_fn;
        }
    }
}

using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Data;

namespace Tensorflow.Estimators
{
    public class TrainSpec
    {
        int _max_steps;
        public int max_steps => _max_steps;

        Func<DatasetV1Adapter> _input_fn;
        public Func<DatasetV1Adapter> input_fn => _input_fn;

        public TrainSpec(Func<DatasetV1Adapter> input_fn, int max_steps)
        {
            _max_steps = max_steps;
            _input_fn = input_fn;
        }
    }
}

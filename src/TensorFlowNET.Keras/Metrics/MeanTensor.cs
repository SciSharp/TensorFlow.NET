using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.Metrics
{
    public class MeanTensor : Metric
    {
        public int total
        {
            get
            {
                throw new NotImplementedException();
            }
        }

        public int count
        {
            get
            {
                throw new NotImplementedException();
            }
        }

        public MeanTensor(int num_classes, string name = "mean_tensor", string dtype = null) : base(name, dtype)
        {
        }


        private void _build(TensorShape shape) => throw new NotImplementedException();

        public override void reset_states()
        {
            throw new NotImplementedException();
        }

        public override Tensor result()
        {
            throw new NotImplementedException();
        }

        public override void update_state(Args args, KwArgs kwargs)
        {
            throw new NotImplementedException();
        }
    }
}

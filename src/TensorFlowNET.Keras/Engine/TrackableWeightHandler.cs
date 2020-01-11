using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.Engine
{
    public class TrackableWeightHandler
    {
        public int num_tensors
        {
            get
            {
                throw new NotImplementedException();
            }
        }

        public TrackableWeightHandler(bool trackable)
        {
            throw new NotImplementedException();
        }

        public void set_weights(Tensor[] weights) => throw new NotImplementedException();

        public void _set_weights_v1(Tensor[] weights) => throw new NotImplementedException();
    }
}

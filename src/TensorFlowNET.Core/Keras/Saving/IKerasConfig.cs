using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.Saving
{
    public interface IKerasConfig
    {
    }

    public interface IKerasConfigable
    {
        IKerasConfig get_config();
    }
}

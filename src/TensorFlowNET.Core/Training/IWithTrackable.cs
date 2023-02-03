using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Train;

namespace Tensorflow.Training
{
    public interface IWithTrackable
    {
        Trackable GetTrackable();
    }
}

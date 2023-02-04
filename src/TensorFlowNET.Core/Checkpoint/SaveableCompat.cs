using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Train;

namespace Tensorflow.Checkpoint
{
    internal static class SaveableCompat
    {
        public static string? get_saveable_name(Trackable cls_or_obj)
        {
            // TODO: implement it with Attribute.
            return null;
        }
    }
}

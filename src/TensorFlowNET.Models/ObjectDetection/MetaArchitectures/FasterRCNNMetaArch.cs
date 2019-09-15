using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Models.ObjectDetection
{
    public class FasterRCNNMetaArch : Core.DetectionModel
    {
        FasterRCNNInitArgs _args;

        public FasterRCNNMetaArch(FasterRCNNInitArgs args)
        {
            _args = args;
        }

        public (Tensor, Tensor) preprocess(Tensor tensor)
        {
            throw new NotImplementedException("");
        }
    }
}

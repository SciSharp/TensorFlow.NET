using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Models.ObjectDetection
{
    public class GridAnchorGenerator : Core.AnchorGenerator
    {
        public GridAnchorGenerator(float[] scales = null)
        {
            if (scales == null)
                scales = new[] { 0.5f, 1.0f, 2.0f };
        }
    }
}

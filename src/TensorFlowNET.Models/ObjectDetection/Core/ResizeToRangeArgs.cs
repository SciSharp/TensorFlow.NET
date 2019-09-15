using System;
using System.Collections.Generic;
using System.Text;
using static Tensorflow.tensorflow.image_internal;

namespace Tensorflow.Models.ObjectDetection.Core
{
    public class ResizeToRangeArgs
    {
        public Tensor image { get; set; }
        public int[] masks { get; set; }
        public int min_dimension { get; set; }
        public int max_dimension { get; set; }
        public ResizeMethod method {get;set;}
        public bool align_corners { get; set; }
        public bool pad_to_max_dimension { get; set; }
        public int[] per_channel_pad_value { get; set; }
    }
}

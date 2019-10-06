using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Models.ObjectDetection.Core;

namespace Tensorflow.Models.ObjectDetection
{
    public class FasterRCNNInitArgs
    {
        public bool is_training { get; set; }
        public int num_classes { get; set; }
        public Func<ResizeToRangeArgs, Tensor[]> image_resizer_fn { get; set; }
        public FasterRCNNFeatureExtractor feature_extractor { get; set; }
        public int number_of_stage { get; set; }
        public object first_stage_anchor_generator { get; set; }
        public object first_stage_target_assigner { get; set; }
        public int first_stage_atrous_rate { get; set; }
        public int parallel_iterations { get; set; } = 16;
    }
}

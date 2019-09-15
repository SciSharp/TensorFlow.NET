using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Models.ObjectDetection
{
    public class FasterRCNNInitArgs
    {
        public bool is_training { get; set; }
        public int num_classes { get; set; }
        public Action image_resizer_fn { get; set; }
        public Action feature_extractor { get; set; }
        public int number_of_stage { get; set; }
        public object first_stage_anchor_generator { get; set; }
        public object first_stage_target_assigner { get; set; }
        public int first_stage_atrous_rate { get; set; }
    }
}

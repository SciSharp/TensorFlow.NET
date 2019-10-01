using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Models.ObjectDetection
{
    /// <summary>
    /// Faster R-CNN Feature Extractor definition.
    /// </summary>
    public class FasterRCNNFeatureExtractor
    {
        bool _is_training;
        int _first_stage_features_stride;
        bool _reuse_weights = false;
        float _weight_decay = 0.0f;
        bool _train_batch_norm;

        public FasterRCNNFeatureExtractor(bool is_training, 
            int first_stage_features_stride,
            bool batch_norm_trainable = false,
            bool reuse_weights = false,
            float weight_decay = 0.0f)
        {
            _is_training = is_training;
            _first_stage_features_stride = first_stage_features_stride;
            _train_batch_norm = (batch_norm_trainable && is_training);
            _reuse_weights = reuse_weights;
            _weight_decay = weight_decay;
        }
    }
}

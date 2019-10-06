using System;
using System.Collections.Generic;
using System.Text;
using static Tensorflow.Binding;
using Tensorflow.Operations.Activation;
using Tensorflow.Models.Slim.Nets;

namespace Tensorflow.Models.ObjectDetection
{
    /// <summary>
    /// Faster R-CNN Resnet 101 feature extractor implementation.
    /// </summary>
    public class FasterRCNNResnet101FeatureExtractor : FasterRCNNResnetV1FeatureExtractor
    {
        public FasterRCNNResnet101FeatureExtractor(bool is_training, 
            int first_stage_features_stride,
            bool batch_norm_trainable = false, 
            bool reuse_weights = false, 
            float weight_decay = 0.0f,
            IActivation activation_fn = null) : base("resnet_v1_101", 
                ResNetV1.resnet_v1_101,
                is_training,
                first_stage_features_stride,
                batch_norm_trainable: batch_norm_trainable,
                reuse_weights: reuse_weights,
                weight_decay: weight_decay,
                activation_fn: activation_fn)
        {

        }
    }
}

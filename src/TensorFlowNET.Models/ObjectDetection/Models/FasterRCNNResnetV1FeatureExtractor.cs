using System;
using System.Collections.Generic;
using System.Text;
using static Tensorflow.Binding;
using Tensorflow.Operations.Activation;

namespace Tensorflow.Models.ObjectDetection
{
    public class FasterRCNNResnetV1FeatureExtractor : FasterRCNNFeatureExtractor
    {
        public FasterRCNNResnetV1FeatureExtractor(string architecture, 
            Action resnet_model, 
            bool is_training, 
            int first_stage_features_stride, 
            bool batch_norm_trainable = false,
            bool reuse_weights = false, 
            float weight_decay = 0.0f, 
            IActivation activation_fn = null) : base(is_training, 
                first_stage_features_stride,
                batch_norm_trainable: batch_norm_trainable,
                reuse_weights: reuse_weights,
                weight_decay: weight_decay)
        {
            if (activation_fn == null)
                activation_fn = tf.nn.relu();
        }
    }
}

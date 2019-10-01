using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Models.ObjectDetection.Protos;
using static Tensorflow.Models.ObjectDetection.Protos.DetectionModel;

namespace Tensorflow.Models.ObjectDetection
{
    public class ModelBuilder
    {
        ImageResizerBuilder _image_resizer_builder;
        FasterRCNNFeatureExtractor _feature_extractor;
        AnchorGeneratorBuilder _anchor_generator_builder;

        public ModelBuilder()
        {
            _image_resizer_builder = new ImageResizerBuilder();
            _anchor_generator_builder = new AnchorGeneratorBuilder();
        }

        /// <summary>
        /// Builds a DetectionModel based on the model config.
        /// </summary>
        /// <param name="model_config">A model.proto object containing the config for the desired DetectionModel.</param>
        /// <param name="is_training">True if this model is being built for training purposes.</param>
        /// <param name="add_summaries">Whether to add tensorflow summaries in the model graph.</param>
        /// <returns>DetectionModel based on the config.</returns>
        public FasterRCNNMetaArch build(DetectionModel model_config, bool is_training, bool add_summaries = true)
        {
            var meta_architecture = model_config.ModelCase;
            if (meta_architecture == ModelOneofCase.Ssd)
                throw new NotImplementedException("");
            else if (meta_architecture == ModelOneofCase.FasterRcnn)
                return _build_faster_rcnn_model(model_config.FasterRcnn, is_training, add_summaries);

            throw new ValueError($"Unknown meta architecture: {meta_architecture}");
        }

        /// <summary>
        /// Builds a Faster R-CNN or R-FCN detection model based on the model config.
        /// </summary>
        /// <param name="frcnn_config"></param>
        /// <param name="is_training"></param>
        /// <param name="add_summaries"></param>
        /// <returns>FasterRCNNMetaArch based on the config.</returns>
        private FasterRCNNMetaArch _build_faster_rcnn_model(FasterRcnn frcnn_config, bool is_training, bool add_summaries)
        {
            var num_classes = frcnn_config.NumClasses;
            var image_resizer_fn = _image_resizer_builder.build(frcnn_config.ImageResizer);

            var feature_extractor = _build_faster_rcnn_feature_extractor(frcnn_config.FeatureExtractor, is_training,
                inplace_batchnorm_update: frcnn_config.InplaceBatchnormUpdate);

            var number_of_stages = frcnn_config.NumberOfStages;
            var first_stage_anchor_generator = _anchor_generator_builder.build(frcnn_config.FirstStageAnchorGenerator);
            var first_stage_atrous_rate = frcnn_config.FirstStageAtrousRate;

            return new FasterRCNNMetaArch(new FasterRCNNInitArgs
            {
                is_training = is_training,
                num_classes = num_classes,
                image_resizer_fn = image_resizer_fn,
                feature_extractor = _feature_extractor,
                number_of_stage = number_of_stages,
                first_stage_anchor_generator = null,
                first_stage_atrous_rate = first_stage_atrous_rate
            });
        }

        public Action preprocess()
        {
            throw new NotImplementedException("");
        }

        private FasterRCNNFeatureExtractor _build_faster_rcnn_feature_extractor(FasterRcnnFeatureExtractor feature_extractor_config,
            bool is_training, bool reuse_weights = false, bool inplace_batchnorm_update = false)
        {
            if (inplace_batchnorm_update)
                throw new ValueError("inplace batchnorm updates not supported.");
            var feature_type = feature_extractor_config.Type;
            var first_stage_features_stride = feature_extractor_config.FirstStageFeaturesStride;
            var batch_norm_trainable = feature_extractor_config.BatchNormTrainable;

            return new FasterRCNNResnet101FeatureExtractor(is_training, first_stage_features_stride,
                batch_norm_trainable: batch_norm_trainable,
                reuse_weights: reuse_weights);
        }
    }
}

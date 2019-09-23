using Protobuf.Text;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Tensorflow.Models.ObjectDetection.Protos;

namespace Tensorflow.Models.ObjectDetection.Utils
{
    public class ConfigUtil
    {
        public static TrainEvalPipelineConfig get_configs_from_pipeline_file(string pipeline_config_path)
        {
            var config = File.ReadAllText(pipeline_config_path);
            var pipeline_config = TrainEvalPipelineConfig.Parser.ParseText(config);

            return pipeline_config;
        }

        public static ImageResizer get_image_resizer_config(DetectionModel model_config)
        {
            var meta_architecture = model_config.ModelCase;

            if (meta_architecture == DetectionModel.ModelOneofCase.FasterRcnn)
                return model_config.FasterRcnn.ImageResizer;
            else if (meta_architecture == DetectionModel.ModelOneofCase.Ssd)
                return model_config.Ssd.ImageResizer;
            
            throw new Exception($"Unknown model type: {meta_architecture}");
        }

        public static (int, int) get_spatial_image_size(ImageResizer image_resizer_config)
        {
            if (image_resizer_config.ImageResizerOneofCase == ImageResizer.ImageResizerOneofOneofCase.FixedShapeResizer)
                return (image_resizer_config.FixedShapeResizer.Height, image_resizer_config.FixedShapeResizer.Width);
            else if (image_resizer_config.ImageResizerOneofCase == ImageResizer.ImageResizerOneofOneofCase.KeepAspectRatioResizer)
            {
                if (image_resizer_config.KeepAspectRatioResizer.PadToMaxDimension)
                    return (image_resizer_config.KeepAspectRatioResizer.MaxDimension, image_resizer_config.KeepAspectRatioResizer.MaxDimension);
                else
                    return (-1, -1);
            }
            else if (image_resizer_config.ImageResizerOneofCase == ImageResizer.ImageResizerOneofOneofCase.IdentityResizer
                    || image_resizer_config.ImageResizerOneofCase == ImageResizer.ImageResizerOneofOneofCase.ConditionalShapeResizer)
            {
                return (-1, -1);
            }
            
            throw new Exception("Unknown image resizer type.");
        }

        public static Dictionary<string, object> create_configs_from_pipeline_proto(TrainEvalPipelineConfig pipeline_config)
        {
            var configs = new Dictionary<string, object>(StringComparer.OrdinalIgnoreCase);

            configs["model"] = pipeline_config.Model;
            configs["train_config"] = pipeline_config.TrainConfig;
            configs["train_input_config"] = pipeline_config.TrainInputReader;
            configs["eval_config"] = pipeline_config.EvalConfig;
            configs["eval_input_configs"] = pipeline_config.EvalInputReader;
            
            //# Keeps eval_input_config only for backwards compatibility. All clients should
            //# read eval_input_configs instead.
            if (pipeline_config.EvalInputReader != null && pipeline_config.EvalInputReader.Count > 0)
                configs["eval_input_config"] = pipeline_config.EvalInputReader[0];

            if (pipeline_config.GraphRewriter != null)
                configs["graph_rewriter_config"] = pipeline_config.GraphRewriter;

            return configs;
        }

        public static GraphRewriter get_graph_rewriter_config_from_file(string graph_rewriter_config_file)
        {
            throw new NotImplementedException();
        }

        public static int get_number_of_classes(DetectionModel model_config)
        {
            var meta_architecture = model_config.ModelCase;

            if (meta_architecture == DetectionModel.ModelOneofCase.FasterRcnn)
                return model_config.FasterRcnn.NumClasses;
            
            if (meta_architecture == DetectionModel.ModelOneofCase.Ssd)
                return model_config.Ssd.NumClasses;

            throw new Exception("Expected the model to be one of 'faster_rcnn' or 'ssd'.");
        }

        public static Protos.Optimizer.OptimizerOneofCase get_optimizer_type(TrainConfig train_config)
        {
            return train_config.Optimizer.OptimizerCase;
        }

        public static string get_learning_rate_type(Optimizer optimizer_config)
        {
            throw new NotImplementedException();
        }
    }
}

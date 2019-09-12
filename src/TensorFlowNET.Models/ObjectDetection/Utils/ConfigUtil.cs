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
            var json = File.ReadAllText(pipeline_config_path);
            var pipeline_config = TrainEvalPipelineConfig.Parser.ParseJson(json);

            return pipeline_config;
        }
    }
}

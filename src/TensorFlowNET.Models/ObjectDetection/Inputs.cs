using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Data;
using Tensorflow.Models.ObjectDetection.Protos;

namespace Tensorflow.Models.ObjectDetection
{
    public class Inputs
    {
        ModelBuilder modelBuilder;
        Dictionary<string, Func<DetectionModel, bool, bool, FasterRCNNMetaArch>> INPUT_BUILDER_UTIL_MAP;

        public Inputs()
        {
            modelBuilder = new ModelBuilder();
            INPUT_BUILDER_UTIL_MAP = new Dictionary<string, Func<DetectionModel, bool, bool, FasterRCNNMetaArch>>();
            INPUT_BUILDER_UTIL_MAP["model_build"] = modelBuilder.build;
        }

        public Func<DatasetV1Adapter> create_train_input_fn(TrainConfig train_config, InputReader train_input_config, DetectionModel model_config)
        {
            Func<DatasetV1Adapter> _train_input_fn = () =>
                train_input(train_config, train_input_config, model_config);

            return _train_input_fn;
        }

        /// <summary>
        /// Returns `features` and `labels` tensor dictionaries for training.
        /// </summary>
        /// <param name="train_config"></param>
        /// <param name="train_input_config"></param>
        /// <param name="model_config"></param>
        /// <returns></returns>
        public DatasetV1Adapter train_input(TrainConfig train_config, InputReader train_input_config, DetectionModel model_config)
        {
            var arch = INPUT_BUILDER_UTIL_MAP["model_build"](model_config, true, true);
            Func<Tensor, (Tensor, Tensor)> model_preprocess_fn = arch.preprocess;

            var dataset = DatasetBuilder.build(train_input_config);

            return dataset;
        }
    }
}

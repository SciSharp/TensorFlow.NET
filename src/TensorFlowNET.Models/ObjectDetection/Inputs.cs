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
        DatasetBuilder datasetBuilder;

        public Inputs()
        {
            modelBuilder = new ModelBuilder();
            datasetBuilder = new DatasetBuilder();
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
            var arch = modelBuilder.build(model_config, true, true);
            Func<Tensor, (Tensor, Tensor)> model_preprocess_fn = arch.preprocess;

            Func<Dictionary<string, Tensor>, (Dictionary<string, Tensor>, Dictionary<string, Tensor>) > transform_and_pad_input_data_fn = (tensor_dict) =>
            {
                return (_get_features_dict(tensor_dict), _get_labels_dict(tensor_dict));
            };

            var dataset = datasetBuilder.build(train_input_config);

            return dataset;
        }

        private Dictionary<string, Tensor> _get_features_dict(Dictionary<string, Tensor> input_dict)
        {
            throw new NotImplementedException("_get_features_dict");
        }

        private Dictionary<string, Tensor> _get_labels_dict(Dictionary<string, Tensor> input_dict)
        {
            throw new NotImplementedException("_get_labels_dict");
        }
    }
}

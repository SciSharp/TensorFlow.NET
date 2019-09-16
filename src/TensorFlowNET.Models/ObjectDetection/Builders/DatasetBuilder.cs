using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Data;
using Tensorflow.Models.ObjectDetection.Protos;

namespace Tensorflow.Models.ObjectDetection
{
    public class DatasetBuilder
    {
        public DatasetV1Adapter build(InputReader input_reader_config, 
            int batch_size = 0, 
            Action transform_input_data_fn = null)
        {
            Func<Dictionary<string, Tensor>, (Dictionary<string, Tensor>, Dictionary<string, Tensor>)> transform_and_pad_input_data_fn = (tensor_dict) =>
            {
                return (null, null);
            };

            var config = input_reader_config.TfRecordInputReader;

            throw new NotImplementedException("");
        }

        public Dictionary<string, Tensor> process_fn(Tensor value)
        {
            throw new NotImplementedException("");
        }
    }
}

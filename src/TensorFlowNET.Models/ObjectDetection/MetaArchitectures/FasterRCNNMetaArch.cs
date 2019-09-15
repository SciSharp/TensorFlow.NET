using System;
using System.Collections.Generic;
using System.Text;
using static Tensorflow.Binding;

namespace Tensorflow.Models.ObjectDetection
{
    public class FasterRCNNMetaArch : Core.DetectionModel
    {
        FasterRCNNInitArgs _args;

        public FasterRCNNMetaArch(FasterRCNNInitArgs args)
        {
            _args = args;
        }

        /// <summary>
        /// Feature-extractor specific preprocessing.
        /// </summary>
        /// <param name="inputs"></param>
        /// <returns></returns>
        public (Tensor, Tensor) preprocess(Tensor inputs)
        {
            tf_with(tf.name_scope("Preprocessor"), delegate
            {
                /*var outputs = shape_utils.static_or_dynamic_map_fn(
                  _image_resizer_fn,
                  elems: inputs,
                  dtype: new[] { tf.float32, tf.int32 },
                  parallel_iterations: _parallel_iterations);*/

            });

            throw new NotImplementedException("");
        }
    }
}

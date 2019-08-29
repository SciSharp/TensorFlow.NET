using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow;
using static Tensorflow.Binding;

namespace TensorFlowNET.Examples.ImageProcessing.YOLO
{
    class backbone
    {
        public static (Tensor, Tensor, Tensor) darknet53(Tensor input_data, Tensor trainable)
        {
            return tf_with(tf.variable_scope("darknet"), scope =>
            {
                input_data = common.convolutional(input_data, filters_shape: new int[] { 3, 3, 3, 32 }, trainable: trainable, name: "conv0");
                input_data = common.convolutional(input_data, filters_shape: new int[] { 3, 3, 32, 64 }, trainable: trainable, name: "conv1", downsample: true);

                foreach (var i in range(1))
                    input_data = common.residual_block(input_data, 64, 32, 64, trainable: trainable, name: $"residual{i + 0}");

                var route_1 = input_data;
                var route_2 = input_data;

                return (route_1, route_2, input_data);
            });
        }
    }
}

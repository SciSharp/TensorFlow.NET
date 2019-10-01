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

                input_data = common.convolutional(input_data, filters_shape: new[] { 3, 3, 64, 128 },
                                          trainable: trainable, name: "conv4", downsample: true);

                foreach (var i in range(2))
                    input_data = common.residual_block(input_data, 128, 64, 128, trainable: trainable, name: $"residual{i + 1}");

                input_data = common.convolutional(input_data, filters_shape: new[] { 3, 3, 128, 256 },
                                          trainable: trainable, name: "conv9", downsample: true);

                foreach (var i in range(8))
                    input_data = common.residual_block(input_data, 256, 128, 256, trainable: trainable, name: $"residual{i + 3}");

                var route_1 = input_data;
                input_data = common.convolutional(input_data, filters_shape: new int[] { 3, 3, 256, 512 },
                                          trainable: trainable, name: "conv26", downsample: true);

                foreach (var i in range(8))
                    input_data = common.residual_block(input_data, 512, 256, 512, trainable: trainable, name: $"residual{i + 11}");

                var route_2 = input_data;
                input_data = common.convolutional(input_data, filters_shape: new[] { 3, 3, 512, 1024 },
                                          trainable: trainable, name: "conv43", downsample: true);

                foreach (var i in range(4))
                    input_data = common.residual_block(input_data, 1024, 512, 1024, trainable: trainable, name: $"residual{i + 19}");

                return (route_1, route_2, input_data);
            });
        }
    }
}

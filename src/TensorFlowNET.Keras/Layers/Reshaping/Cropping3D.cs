using Tensorflow.Keras.ArgsDefinition.Reshaping;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Saving;
using Tensorflow.Common.Types;

namespace Tensorflow.Keras.Layers.Reshaping
{
    /// <summary>
    /// Similar to copping 2D
    /// </summary>
    public class Cropping3D : Layer
    {
        Cropping3DArgs args;
        public Cropping3D(Cropping3DArgs args) : base(args)
        {
            this.args = args;
        }

        public override void build(KerasShapesWrapper input_shape)
        {
            built = true;
            _buildInputShape = input_shape;
        }

        protected override Tensors Call(Tensors inputs, Tensors state = null, bool? training = null, IOptionalArgs? optional_args = null)
        {
            Tensor output = inputs;
            if (output.rank != 5)
            {
                // throw an ValueError exception
                throw new ValueError("Expected dim=5, found dim=" + output.rank);
            }

            if (args.cropping.shape == new Shape(1))
            {
                int crop = args.cropping[0];
                if (args.data_format == Cropping3DArgs.DataFormat.channels_last)
                {
                    output = output[new Slice(),
                                                  new Slice(crop, (int)output.shape[1] - crop),
                                                  new Slice(crop, (int)output.shape[2] - crop),
                                                  new Slice(crop, (int)output.shape[3] - crop),
                                                  new Slice()];
                }
                else
                {
                    output = output[new Slice(),
                                                  new Slice(),
                                                  new Slice(crop, (int)output.shape[2] - crop),
                                                  new Slice(crop, (int)output.shape[3] - crop),
                                                  new Slice(crop, (int)output.shape[4] - crop)];
                }

            }
            // int[1][3] equivalent to a tuple of 3 integers
            else if (args.cropping.shape == new Shape(3))
            {
                var crop_1 = args.cropping[0];
                var crop_2 = args.cropping[1];
                var crop_3 = args.cropping[2];
                if (args.data_format == Cropping3DArgs.DataFormat.channels_last)
                {
                    output = output[new Slice(),
                                                  new Slice(crop_1, (int)output.shape[1] - crop_1),
                                                  new Slice(crop_2, (int)output.shape[2] - crop_2),
                                                  new Slice(crop_3, (int)output.shape[3] - crop_3),
                                                  new Slice()];
                }
                else
                {
                    output = output[new Slice(),
                                                  new Slice(),
                                                  new Slice(crop_1, (int)output.shape[2] - crop_1),
                                                  new Slice(crop_2, (int)output.shape[3] - crop_2),
                                                   new Slice(crop_3, (int)output.shape[4] - crop_3)];
                }
            }
            else if (args.cropping.shape[0] == 3 && args.cropping.shape[1] == 2)
            {
                int x = args.cropping[0, 0], x_end = args.cropping[0, 1];
                int y = args.cropping[1, 0], y_end = args.cropping[1, 1];
                int z = args.cropping[2, 0], z_end = args.cropping[2, 1];
                if (args.data_format == Cropping3DArgs.DataFormat.channels_last)
                {
                    output = output[new Slice(),
                                                  new Slice(x, (int)output.shape[1] - x_end),
                                                  new Slice(y, (int)output.shape[2] - y_end),
                                                  new Slice(z, (int)output.shape[3] - z_end),
                                                  new Slice()];
                }
                else
                {
                    output = output[new Slice(),
                                                  new Slice(),
                                                  new Slice(x, (int)output.shape[2] - x_end),
                                                  new Slice(y, (int)output.shape[3] - y_end),
                                                   new Slice(z, (int)output.shape[4] - z_end)
                                                  ];
                }
            }
            return output;
        }
        public override Shape ComputeOutputShape(Shape input_shape)
        {
            if (args.cropping.shape == new Shape(1))
            {
                int crop = args.cropping[0];
                if (args.data_format == Cropping3DArgs.DataFormat.channels_last)
                {
                    return new Shape((int)input_shape[0], (int)input_shape[1] - crop * 2, (int)input_shape[2] - crop * 2, (int)input_shape[3] - crop * 2, (int)input_shape[4]);
                }
                else
                {
                    return new Shape((int)input_shape[0], (int)input_shape[1], (int)input_shape[2] - crop * 2, (int)input_shape[3] - crop * 2, (int)input_shape[4] - crop * 2);
                }
            }
            // int[1][3] equivalent to a tuple of 3 integers
            else if (args.cropping.shape == new Shape(3))
            {
                var crop_start_1 = args.cropping[0];
                var crop_start_2 = args.cropping[1];
                var crop_start_3 = args.cropping[2];
                if (args.data_format == Cropping3DArgs.DataFormat.channels_last)
                {
                    return new Shape((int)input_shape[0], (int)input_shape[1] - crop_start_1 * 2, (int)input_shape[2] - crop_start_2 * 2, (int)input_shape[3] - crop_start_3 * 2, (int)input_shape[4]);
                }
                else
                {
                    return new Shape((int)input_shape[0], (int)input_shape[1], (int)input_shape[2] - crop_start_1 * 2, (int)input_shape[3] - crop_start_2 * 2, (int)input_shape[4] - crop_start_3 * 2);
                }
            }
            else if (args.cropping.shape == new Shape(3, 2))
            {
                int x = args.cropping[0, 0], x_end = args.cropping[0, 1];
                int y = args.cropping[1, 0], y_end = args.cropping[1, 1];
                int z = args.cropping[2, 0], z_end = args.cropping[2, 1];
                if (args.data_format == Cropping3DArgs.DataFormat.channels_last)
                {
                    return new Shape((int)input_shape[0], (int)input_shape[1] - x - x_end, (int)input_shape[2] - y - y_end, (int)input_shape[3] - z - z_end, (int)input_shape[4]);
                }
                else
                {
                    return new Shape((int)input_shape[0], (int)input_shape[1], (int)input_shape[2] - x - x_end, (int)input_shape[3] - y - y_end, (int)input_shape[4] - z - z_end);
                }
            }
            else
            {
                throw new ValueError();
            }
        }
    }
}

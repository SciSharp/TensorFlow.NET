using Tensorflow.Keras.ArgsDefinition.Reshaping;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Saving;
using Tensorflow.Common.Types;

namespace Tensorflow.Keras.Layers.Reshaping
{
    /// <summary>
    /// Crop the input along axis 1 and 2.
    /// <para> For example: </para>
    /// <para> shape (1, 5, 5, 5) -- crop2D ((1, 2), (1, 3)) --> shape (1, 2, 1, 5) </para>
    /// </summary>
    public class Cropping2D : Layer
    {
        Cropping2DArgs args;
        public Cropping2D(Cropping2DArgs args) : base(args)
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
            if (output.rank != 4)
            {
                // throw an ValueError exception
                throw new ValueError("Expected dim=4, found dim=" + output.rank);
            }
            if (args.cropping.shape == new Shape(1))
            {
                int crop = args.cropping[0];
                if (args.data_format == Cropping2DArgs.DataFormat.channels_last)
                {
                    output = output[new Slice(),
                                                  new Slice(crop, (int)output.shape[1] - crop),
                                                  new Slice(crop, (int)output.shape[2] - crop),
                                                  new Slice()];
                }
                else
                {
                    output = output[new Slice(),
                                                  new Slice(),
                                                  new Slice(crop, (int)output.shape[2] - crop),
                                                  new Slice(crop, (int)output.shape[3] - crop)];
                }
            }
            // a tuple of 2 integers
            else if (args.cropping.shape == new Shape(2))
            {
                int crop_1 = args.cropping[0];
                int crop_2 = args.cropping[1];
                if (args.data_format == Cropping2DArgs.DataFormat.channels_last)
                {
                    output = output[new Slice(),
                                                  new Slice(crop_1, (int)output.shape[1] - crop_1),
                                                  new Slice(crop_2, (int)output.shape[2] - crop_2),
                                                  new Slice()];
                }
                else
                {
                    output = output[new Slice(),
                                                  new Slice(),
                                                  new Slice(crop_1, (int)output.shape[2] - crop_1),
                                                  new Slice(crop_2, (int)output.shape[3] - crop_2)];
                }
            }
            else if (args.cropping.shape[0] == 2 && args.cropping.shape[1] == 2)
            {
                int x_start = args.cropping[0, 0], x_end = args.cropping[0, 1];
                int y_start = args.cropping[1, 0], y_end = args.cropping[1, 1];
                if (args.data_format == Cropping2DArgs.DataFormat.channels_last)
                {
                    output = output[new Slice(),
                                                  new Slice(x_start, (int)output.shape[1] - x_end),
                                                  new Slice(y_start, (int)output.shape[2] - y_end),
                                                  new Slice()];
                }
                else
                {
                    output = output[new Slice(),
                                                  new Slice(),
                                                  new Slice(x_start, (int)output.shape[2] - x_end),
                                                  new Slice(y_start, (int)output.shape[3] - y_end)
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
                if (args.data_format == Cropping2DArgs.DataFormat.channels_last)
                {
                    return new Shape((int)input_shape[0], (int)input_shape[1] - crop * 2, (int)input_shape[2] - crop * 2, (int)input_shape[3]);
                }
                else
                {
                    return new Shape((int)input_shape[0], (int)input_shape[1], (int)input_shape[2] - crop * 2, (int)input_shape[3] - crop * 2);
                }
            }
            // a tuple of 2 integers
            else if (args.cropping.shape == new Shape(2))
            {
                int crop_1 = args.cropping[0], crop_2 = args.cropping[1];
                if (args.data_format == Cropping2DArgs.DataFormat.channels_last)
                {
                    return new Shape((int)input_shape[0], (int)input_shape[1] - crop_1 * 2, (int)input_shape[2] - crop_2 * 2, (int)input_shape[3]);
                }
                else
                {
                    return new Shape((int)input_shape[0], (int)input_shape[1], (int)input_shape[2] - crop_1 * 2, (int)input_shape[3] - crop_2 * 2);
                }
            }
            else if (args.cropping.shape == new Shape(2, 2))
            {
                int crop_1_start = args.cropping[0, 0], crop_1_end = args.cropping[0, 1];
                int crop_2_start = args.cropping[1, 0], crop_2_end = args.cropping[1, 1];
                if (args.data_format == Cropping2DArgs.DataFormat.channels_last)
                {
                    return new Shape((int)input_shape[0], (int)input_shape[1] - crop_1_start - crop_1_end,
                          (int)input_shape[2] - crop_2_start - crop_2_end, (int)input_shape[3]);
                }
                else
                {
                    return new Shape((int)input_shape[0], (int)input_shape[1],
                          (int)input_shape[2] - crop_1_start - crop_1_end, (int)input_shape[3] - crop_2_start - crop_2_end);
                }
            }
            else
            {
                throw new ValueError();
            }
        }
    }
}

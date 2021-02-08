using NumSharp;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Utils;
using static Tensorflow.KerasApi;

namespace Tensorflow.Keras.Layers
{
    /// <summary>
    /// Zero-padding layer for 2D input (e.g. picture).
    /// 
    /// This layer can add rows and columns of zeros
    /// at the top, bottom, left and right side of an image tensor.
    /// </summary>
    public class ZeroPadding2D : Layer
    {
        string data_format;
        NDArray padding;
        InputSpec input_spec;

        public ZeroPadding2D(ZeroPadding2DArgs args, string data_format = null)
            : base(args)
        {
            this.data_format = conv_utils.normalize_data_format(data_format);
            this.padding = args.Padding;
            this.input_spec = new InputSpec(ndim: 4);
        }

        protected override Tensors Call(Tensors inputs, Tensor state = null, bool? training = null)
        {
            return keras.backend.spatial_2d_padding(inputs,
                padding: padding,
                data_format: data_format);
        }
    }
}

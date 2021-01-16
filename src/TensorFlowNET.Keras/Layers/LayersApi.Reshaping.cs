using NumSharp;
using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.ArgsDefinition;

namespace Tensorflow.Keras.Layers
{
    public partial class LayersApi
    {
        /// <summary>
        /// Zero-padding layer for 2D input (e.g. picture).
        /// </summary>
        /// <param name="padding"></param>
        /// <returns></returns>
        public ZeroPadding2D ZeroPadding2D(NDArray padding)
            => new ZeroPadding2D(new ZeroPadding2DArgs
            {
                Padding = padding
            });

        /// <summary>
        /// Upsampling layer for 2D inputs.<br/>
        /// Repeats the rows and columns of the data by size[0] and size[1] respectively.
        /// </summary>
        /// <param name="size"></param>
        /// <param name="data_format"></param>
        /// <param name="interpolation"></param>
        /// <returns></returns>
        public UpSampling2D UpSampling2D(TensorShape size = null,
            string data_format = null,
            string interpolation = "nearest")
            => new UpSampling2D(new UpSampling2DArgs
            {
                Size = size ?? (2, 2)
            });

        /// <summary>
        /// Layer that reshapes inputs into the given shape.
        /// </summary>
        /// <param name="target_shape"></param>
        /// <returns></returns>
        public Reshape Reshape(TensorShape target_shape)
            => new Reshape(new ReshapeArgs
            {
                TargetShape = target_shape
            });

        public Reshape Reshape(object[] target_shape)
            => new Reshape(new ReshapeArgs
            {
                TargetShapeObjects = target_shape
            });
    }
}

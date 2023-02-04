using Tensorflow.NumPy;
using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.Layers.Reshaping;
using Tensorflow.Keras.ArgsDefinition.Reshaping;

namespace Tensorflow.Keras.Layers
{
    public partial class LayersApi {
            /// <summary>
            /// Cropping layer for 1D input
            /// </summary>
            /// <param name="cropping">cropping size</param>
            public ILayer Cropping1D ( NDArray cropping )
                => new Cropping1D(new Cropping1DArgs {
                      cropping = cropping
                });

            /// <summary>
            /// Cropping layer for 2D input <br/>
            /// </summary>
            public ILayer Cropping2D ( NDArray cropping, Cropping2DArgs.DataFormat data_format = Cropping2DArgs.DataFormat.channels_last )
                => new Cropping2D(new Cropping2DArgs {
                      cropping = cropping,
                      data_format = data_format
                });

            /// <summary>
            /// Cropping layer for 3D input <br/>
            /// </summary>
            public ILayer Cropping3D ( NDArray cropping, Cropping3DArgs.DataFormat data_format = Cropping3DArgs.DataFormat.channels_last )
                => new Cropping3D(new Cropping3DArgs {
                      cropping = cropping,
                      data_format = data_format
                });
      }
}

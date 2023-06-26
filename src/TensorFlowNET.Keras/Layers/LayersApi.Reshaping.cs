using Tensorflow.NumPy;
using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.ArgsDefinition;

namespace Tensorflow.Keras.Layers {
      public partial class LayersApi {

        /// <summary>
        /// Upsampling layer for 1D inputs. Repeats each temporal step `size` times along the time axis.
        /// </summary>
        /// <param name="size"></param>
        /// <returns></returns>
        public ILayer UpSampling1D(int size)
          => new UpSampling1D(new UpSampling1DArgs
          {
              Size = size
          });

        /// <summary>
        /// Zero-padding layer for 2D input (e.g. picture).
        /// </summary>
        /// <param name="padding"></param>
        /// <returns></returns>
        public ILayer ZeroPadding2D ( NDArray padding )
                => new ZeroPadding2D(new ZeroPadding2DArgs {
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
        public ILayer UpSampling2D(Shape size, string data_format, string interpolation)
            => new UpSampling2D(new UpSampling2DArgs
            {
                Size = size,
                DataFormat = data_format,
                Interpolation = interpolation
            });

        /// <summary>
        /// Permutes the dimensions of the input according to a given pattern.
        /// </summary>
        public ILayer Permute ( int[] dims )
                  => new Permute(new PermuteArgs {
                        dims = dims
                  });

            /// <summary>
            /// Layer that reshapes inputs into the given shape.
            /// </summary>
            /// <param name="target_shape"></param>
            /// <returns></returns>
            public ILayer Reshape ( Shape target_shape )
                => new Reshape(new ReshapeArgs {
                      TargetShape = target_shape
                });

            public ILayer Reshape ( object[] target_shape )
                => new Reshape(new ReshapeArgs {
                      TargetShapeObjects = target_shape
                });
      }
}

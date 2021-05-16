using System;
using System.IO;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Layers;
using static Tensorflow.KerasApi;

namespace Tensorflow.Keras
{
    public partial class Preprocessing
    {
        /// <summary>
        /// Image resizing layer
        /// </summary>
        /// <param name="height"></param>
        /// <param name="width"></param>
        /// <param name="interpolation"></param>
        /// <returns></returns>
        public Resizing Resizing(int height, int width, string interpolation = "bilinear")
            => new Resizing(new ResizingArgs
            {
                Height = height,
                Width = width,
                Interpolation = interpolation
            });
    }
}

﻿using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System;
using System.Text;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Saving;
using Tensorflow.Common.Types;

namespace Tensorflow.Keras.Layers
{
    /// <summary>
    /// Resize the batched image input to target height and width. 
    /// The input should be a 4-D tensor in the format of NHWC.
    /// </summary>
    public class Resizing : PreprocessingLayer
    {
        ResizingArgs args;
        public Resizing(ResizingArgs args) : base(args)
        {
            this.args = args;
        }

        protected override Tensors Call(Tensors inputs, Tensors state = null, bool? training = null, IOptionalArgs? optional_args = null)
        {
            return image_ops_impl.resize_images_v2(inputs, new[] { args.Height, args.Width }, method: args.Interpolation);
        }

        public override Shape ComputeOutputShape(Shape input_shape)
        {
            return new Shape(input_shape.dims[0], args.Height, args.Width, input_shape.dims[3]);
        }

        public static Resizing from_config(JObject config)
        {
            var args = JsonConvert.DeserializeObject<ResizingArgs>(config.ToString());
            args.IsFromConfig = true;
            return new Resizing(args);
        }
    }
}

/*****************************************************************************
   Copyright 2018 The TensorFlow.NET Authors. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
******************************************************************************/

using Tensorflow.Keras.ArgsDefinition;

namespace Tensorflow.Keras.Layers
{
    public class Conv2D : Convolutional
    {
        public Conv2D(Conv2DArgs args) : base(InitializeUndefinedArgs(args))
        {

        }

        private static Conv2DArgs InitializeUndefinedArgs(Conv2DArgs args)
        {
            if(args.Rank == 0)
            {
                args.Rank = 2;
            }
            if (args.Strides is null)
            {
                args.Strides = (1, 1);
            }
            if (string.IsNullOrEmpty(args.Padding))
            {
                args.Padding = "valid";
            }
            if (args.DilationRate == 0)
            {
                args.DilationRate = (1, 1);
            }
            if (args.Groups == 0)
            {
                args.Groups = 1;
            }
            if (args.KernelInitializer is null)
            {
                args.KernelInitializer = tf.glorot_uniform_initializer;
            }
            if (args.BiasInitializer is null)
            {
                args.BiasInitializer = tf.zeros_initializer;
            }
            return args;
        }
    }
}

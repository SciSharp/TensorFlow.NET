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

using Tensorflow.Operations.Activation;

namespace Tensorflow.Keras.Layers
{
    public class Conv2D : Conv
    {
        public Conv2D(int filters,
            int[] kernel_size,
            int[] strides = null,
            string padding = "valid",
            string data_format = "channels_last",
            int[] dilation_rate = null,
            IActivation activation = null,
            bool use_bias = true,
            IInitializer kernel_initializer = null,
            IInitializer bias_initializer = null,
            bool trainable = true,
            string name = null) : base(2, 
                filters,
                kernel_size,
                strides: strides,
                padding: padding,
                data_format: data_format,
                dilation_rate: dilation_rate,
                activation: activation,
                use_bias: use_bias,
                kernel_initializer: kernel_initializer,
                bias_initializer: bias_initializer,
                trainable: trainable, 
                name: name)
        {

        }
    }
}

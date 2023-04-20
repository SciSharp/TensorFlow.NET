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

using Tensorflow.Operations.Initializers;

namespace Tensorflow.Keras;

public partial class InitializersApi : IInitializersApi
{
    /// <summary>
    /// He normal initializer.
    /// </summary>
    /// <param name="seed"></param>
    /// <returns></returns>
    public IInitializer HeNormal(int? seed = null)
    {
        return new VarianceScaling(scale: 2.0f, mode: "fan_in", seed: seed);
    }

    public IInitializer Orthogonal(float gain = 1.0f, int? seed = null)
        => new Orthogonal(gain: gain, seed: seed);
}

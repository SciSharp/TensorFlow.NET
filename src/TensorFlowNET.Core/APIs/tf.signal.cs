/*****************************************************************************
   Copyright 2023 Konstantin Balashov All Rights Reserved.

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

using Tensorflow.Operations;

namespace Tensorflow
{
    public partial class tensorflow
    {
        public SignalApi signal { get; } = new SignalApi();
        public class SignalApi
        {
            public Tensor fft(Tensor input, string name = null)
                => gen_ops.f_f_t(input, name: name);
            public Tensor ifft(Tensor input, string name = null)
                => gen_ops.i_f_f_t(input,  name: name);
            public Tensor fft2d(Tensor input, string name = null)
                => gen_ops.f_f_t2d(input, name: name);
            public Tensor ifft2d(Tensor input, string name = null)
                => gen_ops.i_f_f_t2d(input, name: name);
            public Tensor fft3d(Tensor input, string name = null)
                => gen_ops.f_f_t3d(input, name: name);
            public Tensor ifft3d(Tensor input, string name = null)
                => gen_ops.i_f_f_t3d(input, name: name);
        }
    }
}

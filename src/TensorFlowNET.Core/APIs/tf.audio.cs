/*****************************************************************************
   Copyright 2021 Haiping Chen. All Rights Reserved.

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

using System.Collections.Generic;
using Tensorflow.IO;

namespace Tensorflow
{
    public partial class tensorflow
    {
        public AudioAPI audio { get; } = new AudioAPI();

        public class AudioAPI
        {
            audio_ops audio_ops = new audio_ops();

            public Tensors decode_wav(Tensor contents, int desired_channels = -1, int desired_samples = -1, string name = null)
                => audio_ops.decode_wav(contents,
                    desired_channels: desired_channels,
                    desired_samples: desired_samples,
                    name: name);
        }
    }
}

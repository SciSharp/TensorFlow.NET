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

using System;
using static Tensorflow.Binding;
using Tensorflow.Estimators;

namespace Tensorflow
{
    public partial class tensorflow
    {
        public Estimator_Internal estimator { get; } = new Estimator_Internal();

        public class Estimator_Internal
        {
            public Estimator Estimator(RunConfig config) 
                => new Estimator(config: config);

            public RunConfig RunConfig(string model_dir)
                => new RunConfig(model_dir: model_dir);

            public void train_and_evaluate()
                => Training.train_and_evaluate();
        }
    }
}

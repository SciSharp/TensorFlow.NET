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

            public void train_and_evaluate(Estimator estimator, TrainSpec train_spec, EvalSpec eval_spec)
                => Training.train_and_evaluate(estimator: estimator, train_spec: train_spec, eval_spec: eval_spec);

            public TrainSpec TrainSpec(Action input_fn, int max_steps)
                => new TrainSpec(input_fn: input_fn, max_steps: max_steps);

            /// <summary>
            /// Create an `Exporter` to use with `tf.estimator.EvalSpec`.
            /// </summary>
            /// <param name="name"></param>
            /// <param name="serving_input_receiver_fn"></param>
            /// <param name="as_text"></param>
            /// <returns></returns>
            public FinalExporter FinalExporter(string name, Action serving_input_receiver_fn, bool as_text = false)
                => new FinalExporter(name: name, serving_input_receiver_fn: serving_input_receiver_fn, 
                    as_text: as_text);

            public EvalSpec EvalSpec(string name, Action input_fn, FinalExporter exporters)
                => new EvalSpec(name: name, input_fn: input_fn, exporters: exporters);
        }
    }
}

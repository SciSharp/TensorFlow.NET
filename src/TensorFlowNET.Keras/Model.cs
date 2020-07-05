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

using Keras.Layers;
using NumSharp;
using System;
using System.Collections.Generic;
using Tensorflow;
using static Tensorflow.Binding;

namespace Tensorflow.Keras
{
    public class Model
    {
        public Tensor Flow;
        List<Layer> layer_stack;

        public TensorShape InputShape;

        public Model()
        {
            layer_stack = new List<Layer>();
        }
        public Model Add(Layer layer)
        {
            layer_stack.Add(layer);
            return this;
        }
        public Model Add(IEnumerable<Layer> layers)
        {
            layer_stack.AddRange(layers);
            return this;
        }
        public Tensor getFlow()
        {
            try
            {
                return Flow;
            }
#pragma warning disable CS0168 // Variable is declared but never used
            catch (Exception ex)
#pragma warning restore CS0168 // Variable is declared but never used
            {
                return null;
            }
        }
        public (Operation, Tensor, Tensor) make_graph(Tensor features, Tensor labels)
        {

            // TODO : Creating Loss Functions And Optimizers.....

            #region Model Layers Graph
            /*
            var stddev = 1 / Math.Sqrt(2);

            var d1 = new Dense(num_hidden);
            d1.__build__(features.getShape());
            var hidden_activations = tf.nn.relu(d1.__call__(features));

            var d1_output = d1.output_shape(features.getShape());
            

            var d2 = new Dense(1);
            d2.__build__(d1.output_shape(features.getShape()), seed: 17, stddev: (float)(1/ Math.Sqrt(num_hidden)));
            var logits = d2.__call__(hidden_activations);
            var predictions = tf.sigmoid(tf.squeeze(logits));
            */
            #endregion

            #region Model Graph Form Layer Stack
            var flow_shape = features.TensorShape;
            Flow = features;
            for (int i = 0; i < layer_stack.Count; i++)
            {
                //layer_stack[i].build(flow_shape);
                //flow_shape = layer_stack[i].output_shape(flow_shape);
                //Flow = layer_stack[i].__call__(Flow);
            }
            var predictions = tf.sigmoid(tf.squeeze(Flow));
            
            #endregion

            #region loss and optimizer
            var loss = tf.reduce_mean(tf.square(predictions - tf.cast(labels, tf.float32)), name: "loss");

            var gs = tf.Variable(0, trainable: false, name: "global_step");
            var train_op = tf.train.GradientDescentOptimizer(0.2f).minimize(loss, global_step: gs);
            #endregion

            return (train_op, loss, gs);
        }
        public float train(int num_steps, (NDArray, NDArray) training_dataset)
        {
            var (X, Y) = training_dataset;
            var x_shape = X.shape;
            var batch_size = x_shape[0];
            var graph = tf.Graph().as_default();

            var features = tf.placeholder(tf.float32, new TensorShape(batch_size, 2));
            var labels = tf.placeholder(tf.float32, new TensorShape(batch_size));

            var (train_op, loss, gs) = this.make_graph(features, labels);

            var init = tf.global_variables_initializer();

            float loss_value = 0;
            using (var sess = tf.Session(graph))
            {
                sess.run(init);
                var step = 0;


                while (step < num_steps)
                {
                    var result = sess.run(
                        new ITensorOrOperation[] { train_op, gs, loss },
                        new FeedItem(features, X),
                        new FeedItem(labels, Y));
                    loss_value = result[2];
                    step = result[1];
                    if (step % 1000 == 0)
                        Console.WriteLine($"Step {step} loss: {loss_value}");
                }
                Console.WriteLine($"Final loss: {loss_value}");
            }

            return loss_value;
        }
    }
}

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

using static Tensorflow.ops;

namespace Tensorflow
{
    public partial class tensorflow
    {
        public graph_util_impl graph_util => new graph_util_impl();
        public GraphTransformer graph_transforms => new GraphTransformer();
        public GraphKeys GraphKeys { get; } = new GraphKeys();

        public void reset_default_graph()
            => ops.reset_default_graph();

        public Graph get_default_graph()
            => ops.get_default_graph();

        public Graph peak_default_graph()
            => ops.peak_default_graph();

        /// <summary>
        ///     Creates a new graph.
        /// </summary>
        ///<remarks>Has no interaction with graph defaulting. Equivalent to new Graph();</remarks>
        public Graph Graph()
            => new Graph();
    }
}
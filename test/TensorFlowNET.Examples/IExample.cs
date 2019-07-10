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
using System.Collections.Generic;
using System.Text;
using Tensorflow;

namespace TensorFlowNET.Examples
{
    /// <summary>
    /// Interface of Example project
    /// All example should implement IExample so the entry program will find it.
    /// </summary>
    public interface IExample
    {
        /// <summary>
        /// True to run example
        /// </summary>
        bool Enabled { get; set; }

        /// <summary>
        /// Set true to import the computation graph instead of building it.
        /// </summary>
        bool IsImportingGraph { get; set; }

        string Name { get; }

        bool Run();

        /// <summary>
        /// Build dataflow graph, train and predict
        /// </summary>
        /// <returns></returns>
        void Train(Session sess);
        void Test(Session sess);

        void Predict(Session sess);

        Graph ImportGraph();

        Graph BuildGraph();

        /// <summary>
        /// Prepare dataset
        /// </summary>
        void PrepareData();
    }
}

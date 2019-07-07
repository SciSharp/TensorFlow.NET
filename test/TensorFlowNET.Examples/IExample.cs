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

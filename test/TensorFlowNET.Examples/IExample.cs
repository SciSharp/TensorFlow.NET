using System;
using System.Collections.Generic;
using System.Text;

namespace TensorFlowNET.Examples
{
    /// <summary>
    /// Interface of Example project
    /// All example should implement IExample so the entry program will find it.
    /// </summary>
    public interface IExample
    {
        /// <summary>
        /// running order
        /// </summary>
        int Priority { get; }

        /// <summary>
        /// True to run example
        /// </summary>
        bool Enabled { get; set; }

        /// <summary>
        /// Set true to import the computation graph instead of building it.
        /// </summary>
        bool ImportGraph { get; set; }

        string Name { get; }

        /// <summary>
        /// Build dataflow graph, train and predict
        /// </summary>
        /// <returns></returns>
        bool Run();
        /// <summary>
        /// Prepare dataset
        /// </summary>
        void PrepareData();
    }
}

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
        void Run();
        void PrepareData();
    }
}

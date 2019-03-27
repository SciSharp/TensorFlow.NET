using System;
using System.Collections.Generic;
using System.Text;

namespace TensorFlowNET.Examples.Utility
{
    public class Datasets
    {
        private DataSet _train;
        public DataSet train => _train;

        private DataSet _validation;
        public DataSet validation => _validation;

        private DataSet _test;
        public DataSet test => _test;

        public Datasets(DataSet train, DataSet validation, DataSet test)
        {
            _train = train;
            _validation = validation;
            _test = test;
        }
    }
}

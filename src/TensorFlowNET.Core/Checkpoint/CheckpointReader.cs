using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;

namespace Tensorflow.Checkpoint
{
    internal class CheckpointReader : IDisposable
    {
        private IntPtr _reader;
        public Dictionary<string, TF_DataType> VariableToDataTypeMap { get; set; }
        public Dictionary<string, Shape> VariableToShapeMap { get; set; }

        public CheckpointReader(string filename)
        {
            Status status = new Status();
            _reader = c_api.TF_NewCheckpointReader(filename, status.Handle);
            status.Check(true);
            ReadAllShapeAndType();
        }

        public int HasTensor(string name)
        {
            return c_api.TF_CheckpointReaderHasTensor(_reader, name);
        }

        /// <summary>
        /// Get the variable name.
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        public string GetVariable(int index)
        {
            return c_api.TF_CheckpointReaderGetVariable(_reader, index);
        }

        public int Size()
        {
            return c_api.TF_CheckpointReaderSize(_reader);
        }

        public TF_DataType GetVariableDataType(string name)
        {
            return c_api.TF_CheckpointReaderGetVariableDataType(_reader, name);
        }

        public Shape GetVariableShape(string name)
        {
            // TODO(Rinne): Change it to a constant.
            int num_dims = GetVariableNumDims(name);
            long[] dims = new long[num_dims];
            Status status = new Status();
            c_api.TF_CheckpointReaderGetVariableShape(_reader, name, dims, num_dims, status.Handle);
            status.Check(true);
            return new Shape(dims);
        }

        public int GetVariableNumDims(string name)
        {
            return c_api.TF_CheckpointReaderGetVariableNumDims(_reader, name);
        }

        public Tensor GetTensor(string name)
        {
            Status status = new Status();
            var tensor = c_api.TF_CheckpointReaderGetTensor(_reader, name, status.Handle);
            status.Check(true);
            var shape = GetVariableShape(name);
            var dtype = GetVariableDataType(name);
            return new Tensor(tensor, shape, dtype);
        }

        private void ReadAllShapeAndType()
        {
            VariableToDataTypeMap = new Dictionary<string, TF_DataType>();
            VariableToShapeMap = new Dictionary<string, Shape>();
            int size = Size();
            for(int i = 0; i < size; i++)
            {
                var name = GetVariable(i);
                var shape = GetVariableShape(name);
                var dtype = GetVariableDataType(name);
                VariableToDataTypeMap[name] = dtype;
                VariableToShapeMap[name] = shape;
            }
        }

        public void Dispose()
        {
            c_api.TF_DeleteCheckpointReader(_reader);
        }
    }
}

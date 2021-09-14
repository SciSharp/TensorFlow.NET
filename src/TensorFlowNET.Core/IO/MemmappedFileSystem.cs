/*****************************************************************************
   Copyright 2021 The TensorFlow.NET Authors. All Rights Reserved.

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
using System.IO;
using System.IO.MemoryMappedFiles;
using System.Linq;
using Tensorflow;

namespace Tensorflow.IO
{
    public class MemmappedFileSystem
    {
        public const string MEMMAPPED_PACKAGE_DEFAULT_NAME = "memmapped_package://.";

        private MemoryMappedFile _mmapFile;
        private MemmappedFileSystemDirectory _directory;

        public MemmappedFileSystem(string path)
        {
            using (var stream = File.OpenRead(path))
            {
                // Read the offset for the directory
                var offsetData = new byte[sizeof(ulong)];
                stream.Seek(-sizeof(ulong), SeekOrigin.End);
                stream.Read(offsetData, 0, sizeof(ulong));
                var offset = BitConverter.ToUInt64(offsetData, 0);

                var dirLength = stream.Length - (long) offset - sizeof(ulong);
                if (dirLength < 0)
                {
                    throw new InvalidDataException("Malformed mmapped filesystem!");
                }

                var dirData = new byte[dirLength];

                stream.Seek((long) offset, SeekOrigin.Begin);
                stream.Read(dirData, 0, (int) dirLength);

                _directory = MemmappedFileSystemDirectory.Parser.ParseFrom(dirData);
            }

            _mmapFile = MemoryMappedFile.CreateFromFile(path, FileMode.Open);
        }

        public Stream OpenMemmapped(string filename)
        {
            var entry = _directory.Element.FirstOrDefault(x => x.Name == filename);
            if (entry == null)
            {
                throw new FileNotFoundException($"Missing memmaped file entry: {filename}");
            }

            return _mmapFile.CreateViewStream((long) entry.Offset, (long) entry.Length);
        }
    }
}

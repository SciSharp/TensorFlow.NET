﻿/*****************************************************************************
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

using System.Collections.Generic;
using System.IO;

namespace Tensorflow.IO
{
    public class GFile
    {
        /// <summary>
        /// Recursive directory tree generator for directories.
        /// </summary>
        /// <param name="top">a Directory name</param>
        /// <param name="in_order">Traverse in order if True, post order if False.</param>
        public IEnumerable<(string, string[], string[])> Walk(string top, bool in_order = true)
        {
            return walk_v2(top, in_order);
        }

        private IEnumerable<(string, string[], string[])> walk_v2(string top, bool topdown)
        {
            var subdirs = Directory.GetDirectories(top);
            var files = Directory.GetFiles(top);

            var here = (top, subdirs, files);

            if (subdirs.Length == 0)
                yield return here;
            else
                foreach (var dir in subdirs)
                    foreach (var f in walk_v2(dir, topdown))
                        yield return f;
        }
    }
}

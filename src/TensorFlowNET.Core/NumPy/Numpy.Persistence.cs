/*****************************************************************************
   Copyright 2023 Haiping Chen. All Rights Reserved.

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

using System.IO;
using System.IO.Compression;

namespace Tensorflow.NumPy;

public partial class np
{
    [AutoNumPy]
    public static NpzDictionary loadz(string file)
    {
        using var stream = new FileStream(file, FileMode.Open);
        return new NpzDictionary(stream);
    }

    public static void save(string file, NDArray nd)
    {
        using var stream = new FileStream(file, FileMode.Create);
        NpyFormat.Save(nd, stream);
    }

    public static void savez(string file, params NDArray[] nds)
    {
        using var stream = new FileStream(file, FileMode.Create);
        NpzFormat.Save(nds, stream);
    }

    public static void savez(string file, object nds)
    {
        using var stream = new FileStream(file, FileMode.Create);
        NpzFormat.Save(nds, stream);
    }

    public static void savez_compressed(string file, params NDArray[] nds)
    {
        using var stream = new FileStream(file, FileMode.Create);
        NpzFormat.Save(nds, stream, CompressionLevel.Fastest);
    }

    public static void savez_compressed(string file, object nds)
    {
        using var stream = new FileStream(file, FileMode.Create);
        NpzFormat.Save(nds, stream, CompressionLevel.Fastest);
    }
}

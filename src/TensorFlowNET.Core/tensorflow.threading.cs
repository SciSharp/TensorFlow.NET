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

using System.Runtime.CompilerServices;
using System.Threading;

namespace Tensorflow
{
    public partial class tensorflow
    {
        protected ThreadLocal<Session> defaultSessionFactory;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void ConstructThreadingObjects()
        {
            defaultSessionFactory = new ThreadLocal<Session>(() => new Session());
        }

        public Session defaultSession
        {
            get
            {
                if (!ops.IsSingleThreaded)
                    return defaultSessionFactory.Value;

                return ops.get_default_session();
            }
            internal set
            {
                if (!ops.IsSingleThreaded)
                {
                    defaultSessionFactory.Value = value;
                    return;
                }

                ops.set_default_session(value);
            }
        }
    }
}
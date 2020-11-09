// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Microsoft.VisualStudio.TestTools.UnitTesting
{
    using System;
    //using System.Diagnostics;
    using System.Globalization;
    using System.Reflection;

    /// <summary>
    /// This class represents a private class for the Private Accessor functionality.
    /// </summary>
    internal class PrivateType
    {
        /// <summary>
        /// Binds to everything
        /// </summary>
        private const BindingFlags BindToEveryThing = BindingFlags.Default
                                                      | BindingFlags.NonPublic | BindingFlags.Instance
                                                      | BindingFlags.Public | BindingFlags.Static | BindingFlags.FlattenHierarchy;

        /// <summary>
        /// The wrapped type.
        /// </summary>
        private Type type;

        ///// <summary>
        ///// Initializes a new instance of the <see cref="PrivateType"/> class that contains the private type.
        ///// </summary>
        ///// <param name="assemblyName">Assembly name</param>
        ///// <param name="typeName">fully qualified name of the </param>
        //public PrivateType(string assemblyName, string typeName)
        //{
        //    Helper.CheckParameterNotNullOrEmpty(assemblyName, "assemblyName", string.Empty);
        //    Helper.CheckParameterNotNullOrEmpty(typeName, "typeName", string.Empty);
        //    Assembly asm = Assembly.Load(assemblyName);

        //    this.type = asm.GetType(typeName, true);
        //}

        /// <summary>
        /// Initializes a new instance of the <see cref="PrivateType"/> class that contains
        /// the private type from the type object
        /// </summary>
        /// <param name="type">The wrapped Type to create.</param>
        public PrivateType(Type type)
        {
            if (type == null)
            {
                throw new ArgumentNullException("type");
            }

            this.type = type;
        }

        /// <summary>
        /// Gets the referenced type
        /// </summary>
        public Type ReferencedType => this.type;

        ///// <summary>
        ///// Invokes static member
        ///// </summary>
        ///// <param name="name">Name of the member to InvokeHelper</param>
        ///// <param name="args">Arguements to the invoction</param>
        ///// <returns>Result of invocation</returns>
        //public object InvokeStatic(string name, params object[] args)
        //{
        //    return this.InvokeStatic(name, null, args, CultureInfo.InvariantCulture);
        //}

        ///// <summary>
        ///// Invokes static member
        ///// </summary>
        ///// <param name="name">Name of the member to InvokeHelper</param>
        ///// <param name="parameterTypes">An array of <see cref="T:System.Type"/> objects representing the number, order, and type of the parameters for the method to invoke</param>
        ///// <param name="args">Arguements to the invoction</param>
        ///// <returns>Result of invocation</returns>
        //public object InvokeStatic(string name, Type[] parameterTypes, object[] args)
        //{
        //    return this.InvokeStatic(name, parameterTypes, args, CultureInfo.InvariantCulture);
        //}

        ///// <summary>
        ///// Invokes static member
        ///// </summary>
        ///// <param name="name">Name of the member to InvokeHelper</param>
        ///// <param name="parameterTypes">An array of <see cref="T:System.Type"/> objects representing the number, order, and type of the parameters for the method to invoke</param>
        ///// <param name="args">Arguements to the invoction</param>
        ///// <param name="typeArguments">An array of types corresponding to the types of the generic arguments.</param>
        ///// <returns>Result of invocation</returns>
        //public object InvokeStatic(string name, Type[] parameterTypes, object[] args, Type[] typeArguments)
        //{
        //    return this.InvokeStatic(name, BindToEveryThing, parameterTypes, args, CultureInfo.InvariantCulture, typeArguments);
        //}

        ///// <summary>
        ///// Invokes the static method
        ///// </summary>
        ///// <param name="name">Name of the member</param>
        ///// <param name="args">Arguements to the invocation</param>
        ///// <param name="culture">Culture</param>
        ///// <returns>Result of invocation</returns>
        //public object InvokeStatic(string name, object[] args, CultureInfo culture)
        //{
        //    return this.InvokeStatic(name, null, args, culture);
        //}

        ///// <summary>
        ///// Invokes the static method
        ///// </summary>
        ///// <param name="name">Name of the member</param>
        ///// <param name="parameterTypes">An array of <see cref="T:System.Type"/> objects representing the number, order, and type of the parameters for the method to invoke</param>
        ///// <param name="args">Arguements to the invocation</param>
        ///// <param name="culture">Culture info</param>
        ///// <returns>Result of invocation</returns>
        //public object InvokeStatic(string name, Type[] parameterTypes, object[] args, CultureInfo culture)
        //{
        //    return this.InvokeStatic(name, BindingFlags.InvokeMethod, parameterTypes, args, culture);
        //}

        ///// <summary>
        ///// Invokes the static method
        ///// </summary>
        ///// <param name="name">Name of the member</param>
        ///// <param name="bindingFlags">Additional invocation attributes</param>
        ///// <param name="args">Arguements to the invocation</param>
        ///// <returns>Result of invocation</returns>
        //public object InvokeStatic(string name, BindingFlags bindingFlags, params object[] args)
        //{
        //    return this.InvokeStatic(name, bindingFlags, null, args, CultureInfo.InvariantCulture);
        //}

        ///// <summary>
        ///// Invokes the static method
        ///// </summary>
        ///// <param name="name">Name of the member</param>
        ///// <param name="bindingFlags">Additional invocation attributes</param>
        ///// <param name="parameterTypes">An array of <see cref="T:System.Type"/> objects representing the number, order, and type of the parameters for the method to invoke</param>
        ///// <param name="args">Arguements to the invocation</param>
        ///// <returns>Result of invocation</returns>
        //public object InvokeStatic(string name, BindingFlags bindingFlags, Type[] parameterTypes, object[] args)
        //{
        //    return this.InvokeStatic(name, bindingFlags, parameterTypes, args, CultureInfo.InvariantCulture);
        //}

        ///// <summary>
        ///// Invokes the static method
        ///// </summary>
        ///// <param name="name">Name of the member</param>
        ///// <param name="bindingFlags">Additional invocation attributes</param>
        ///// <param name="args">Arguements to the invocation</param>
        ///// <param name="culture">Culture</param>
        ///// <returns>Result of invocation</returns>
        //public object InvokeStatic(string name, BindingFlags bindingFlags, object[] args, CultureInfo culture)
        //{
        //    return this.InvokeStatic(name, bindingFlags, null, args, culture);
        //}

        ///// <summary>
        ///// Invokes the static method
        ///// </summary>
        ///// <param name="name">Name of the member</param>
        ///// <param name="bindingFlags">Additional invocation attributes</param>
        ///// /// <param name="parameterTypes">An array of <see cref="T:System.Type"/> objects representing the number, order, and type of the parameters for the method to invoke</param>
        ///// <param name="args">Arguements to the invocation</param>
        ///// <param name="culture">Culture</param>
        ///// <returns>Result of invocation</returns>
        //public object InvokeStatic(string name, BindingFlags bindingFlags, Type[] parameterTypes, object[] args, CultureInfo culture)
        //{
        //    return this.InvokeStatic(name, bindingFlags, parameterTypes, args, culture, null);
        //}

        ///// <summary>
        ///// Invokes the static method
        ///// </summary>
        ///// <param name="name">Name of the member</param>
        ///// <param name="bindingFlags">Additional invocation attributes</param>
        ///// /// <param name="parameterTypes">An array of <see cref="T:System.Type"/> objects representing the number, order, and type of the parameters for the method to invoke</param>
        ///// <param name="args">Arguements to the invocation</param>
        ///// <param name="culture">Culture</param>
        ///// <param name="typeArguments">An array of types corresponding to the types of the generic arguments.</param>
        ///// <returns>Result of invocation</returns>
        //public object InvokeStatic(string name, BindingFlags bindingFlags, Type[] parameterTypes, object[] args, CultureInfo culture, Type[] typeArguments)
        //{
        //    Helper.CheckParameterNotNull(name, "name", string.Empty);
        //    if (parameterTypes != null)
        //    {
        //        MethodInfo member = this.type.GetMethod(name, bindingFlags | BindToEveryThing | BindingFlags.Static, null, parameterTypes, null);
        //        if (member == null)
        //        {
        //            throw new ArgumentException(string.Format(CultureInfo.CurrentCulture, FrameworkMessages.PrivateAccessorMemberNotFound, name));
        //        }

        //        try
        //        {
        //            if (member.IsGenericMethodDefinition)
        //            {
        //                MethodInfo constructed = member.MakeGenericMethod(typeArguments);
        //                return constructed.Invoke(null, bindingFlags, null, args, culture);
        //            }
        //            else
        //            {
        //                return member.Invoke(null, bindingFlags, null, args, culture);
        //            }
        //        }
        //        catch (TargetInvocationException e)
        //        {
        //            Debug.Assert(e.InnerException != null, "Inner Exception should not be null.");
        //            if (e.InnerException != null)
        //            {
        //                throw e.InnerException;
        //            }

        //            throw;
        //        }
        //    }
        //    else
        //    {
        //        return this.InvokeHelperStatic(name, bindingFlags | BindingFlags.InvokeMethod, args, culture);
        //    }
        //}

        ///// <summary>
        ///// Gets the element in static array
        ///// </summary>
        ///// <param name="name">Name of the array</param>
        ///// <param name="indices">
        ///// A one-dimensional array of 32-bit integers that represent the indexes specifying
        ///// the position of the element to get. For instance, to access a[10][11] the indices would be {10,11}
        ///// </param>
        ///// <returns>element at the specified location</returns>
        //public object GetStaticArrayElement(string name, params int[] indices)
        //{
        //    Helper.CheckParameterNotNull(name, "name", string.Empty);
        //    return this.GetStaticArrayElement(name, BindToEveryThing, indices);
        //}

        ///// <summary>
        ///// Sets the memeber of the static array
        ///// </summary>
        ///// <param name="name">Name of the array</param>
        ///// <param name="value">value to set</param>
        ///// <param name="indices">
        ///// A one-dimensional array of 32-bit integers that represent the indexes specifying
        ///// the position of the element to set. For instance, to access a[10][11] the array would be {10,11}
        ///// </param>
        //public void SetStaticArrayElement(string name, object value, params int[] indices)
        //{
        //    Helper.CheckParameterNotNull(name, "name", string.Empty);
        //    this.SetStaticArrayElement(name, BindToEveryThing, value, indices);
        //}

        ///// <summary>
        ///// Gets the element in satatic array
        ///// </summary>
        ///// <param name="name">Name of the array</param>
        ///// <param name="bindingFlags">Additional InvokeHelper attributes</param>
        ///// <param name="indices">
        ///// A one-dimensional array of 32-bit integers that represent the indexes specifying
        ///// the position of the element to get. For instance, to access a[10][11] the array would be {10,11}
        ///// </param>
        ///// <returns>element at the spcified location</returns>
        //public object GetStaticArrayElement(string name, BindingFlags bindingFlags, params int[] indices)
        //{
        //    Helper.CheckParameterNotNull(name, "name", string.Empty);
        //    Array arr = (Array)this.InvokeHelperStatic(name, BindingFlags.GetField | BindingFlags.GetProperty | bindingFlags, null, CultureInfo.InvariantCulture);
        //    return arr.GetValue(indices);
        //}

        ///// <summary>
        ///// Sets the memeber of the static array
        ///// </summary>
        ///// <param name="name">Name of the array</param>
        ///// <param name="bindingFlags">Additional InvokeHelper attributes</param>
        ///// <param name="value">value to set</param>
        ///// <param name="indices">
        ///// A one-dimensional array of 32-bit integers that represent the indexes specifying
        ///// the position of the element to set. For instance, to access a[10][11] the array would be {10,11}
        ///// </param>
        //public void SetStaticArrayElement(string name, BindingFlags bindingFlags, object value, params int[] indices)
        //{
        //    Helper.CheckParameterNotNull(name, "name", string.Empty);
        //    Array arr = (Array)this.InvokeHelperStatic(name, BindingFlags.GetField | BindingFlags.GetProperty | BindingFlags.Static | bindingFlags, null, CultureInfo.InvariantCulture);
        //    arr.SetValue(value, indices);
        //}

        ///// <summary>
        ///// Gets the static field
        ///// </summary>
        ///// <param name="name">Name of the field</param>
        ///// <returns>The static field.</returns>
        //public object GetStaticField(string name)
        //{
        //    Helper.CheckParameterNotNull(name, "name", string.Empty);
        //    return this.GetStaticField(name, BindToEveryThing);
        //}

        ///// <summary>
        ///// Sets the static field
        ///// </summary>
        ///// <param name="name">Name of the field</param>
        ///// <param name="value">Arguement to the invocation</param>
        //public void SetStaticField(string name, object value)
        //{
        //    Helper.CheckParameterNotNull(name, "name", string.Empty);
        //    this.SetStaticField(name, BindToEveryThing, value);
        //}

        ///// <summary>
        ///// Gets the static field using specified InvokeHelper attributes
        ///// </summary>
        ///// <param name="name">Name of the field</param>
        ///// <param name="bindingFlags">Additional invocation attributes</param>
        ///// <returns>The static field.</returns>
        //public object GetStaticField(string name, BindingFlags bindingFlags)
        //{
        //    Helper.CheckParameterNotNull(name, "name", string.Empty);
        //    return this.InvokeHelperStatic(name, BindingFlags.GetField | BindingFlags.Static | bindingFlags, null, CultureInfo.InvariantCulture);
        //}

        ///// <summary>
        ///// Sets the static field using binding attributes
        ///// </summary>
        ///// <param name="name">Name of the field</param>
        ///// <param name="bindingFlags">Additional InvokeHelper attributes</param>
        ///// <param name="value">Arguement to the invocation</param>
        //public void SetStaticField(string name, BindingFlags bindingFlags, object value)
        //{
        //    Helper.CheckParameterNotNull(name, "name", string.Empty);
        //    this.InvokeHelperStatic(name, BindingFlags.SetField | bindingFlags | BindingFlags.Static, new[] { value }, CultureInfo.InvariantCulture);
        //}

        /// <summary>
        /// Gets the static field or property
        /// </summary>
        /// <param name="name">Name of the field or property</param>
        /// <returns>The static field or property.</returns>
        public object GetStaticFieldOrProperty(string name)
        {
            Helper.CheckParameterNotNull(name, "name", string.Empty);
            return this.GetStaticFieldOrProperty(name, BindToEveryThing);
        }

        /// <summary>
        /// Sets the static field or property
        /// </summary>
        /// <param name="name">Name of the field or property</param>
        /// <param name="value">Value to be set to field or property</param>
        public void SetStaticFieldOrProperty(string name, object value)
        {
            Helper.CheckParameterNotNull(name, "name", string.Empty);
            this.SetStaticFieldOrProperty(name, BindToEveryThing, value);
        }

        /// <summary>
        /// Gets the static field or property using specified InvokeHelper attributes
        /// </summary>
        /// <param name="name">Name of the field or property</param>
        /// <param name="bindingFlags">Additional invocation attributes</param>
        /// <returns>The static field or property.</returns>
        public object GetStaticFieldOrProperty(string name, BindingFlags bindingFlags)
        {
            Helper.CheckParameterNotNull(name, "name", string.Empty);
            return this.InvokeHelperStatic(name, BindingFlags.GetField | BindingFlags.GetProperty | BindingFlags.Static | bindingFlags, null, CultureInfo.InvariantCulture);
        }

        /// <summary>
        /// Sets the static field or property using binding attributes
        /// </summary>
        /// <param name="name">Name of the field or property</param>
        /// <param name="bindingFlags">Additional invocation attributes</param>
        /// <param name="value">Value to be set to field or property</param>
        public void SetStaticFieldOrProperty(string name, BindingFlags bindingFlags, object value)
        {
            Helper.CheckParameterNotNull(name, "name", string.Empty);
            this.InvokeHelperStatic(name, BindingFlags.SetField | BindingFlags.SetProperty | bindingFlags | BindingFlags.Static, new[] { value }, CultureInfo.InvariantCulture);
        }

        ///// <summary>
        ///// Gets the static property
        ///// </summary>
        ///// <param name="name">Name of the field or property</param>
        ///// <param name="args">Arguements to the invocation</param>
        ///// <returns>The static property.</returns>
        //public object GetStaticProperty(string name, params object[] args)
        //{
        //    return this.GetStaticProperty(name, BindToEveryThing, args);
        //}

        ///// <summary>
        ///// Sets the static property
        ///// </summary>
        ///// <param name="name">Name of the property</param>
        ///// <param name="value">Value to be set to field or property</param>
        ///// <param name="args">Arguments to pass to the member to invoke.</param>
        //public void SetStaticProperty(string name, object value, params object[] args)
        //{
        //    this.SetStaticProperty(name, BindToEveryThing, value, null, args);
        //}

        ///// <summary>
        ///// Sets the static property
        ///// </summary>
        ///// <param name="name">Name of the property</param>
        ///// <param name="value">Value to be set to field or property</param>
        ///// <param name="parameterTypes">An array of <see cref="T:System.Type"/> objects representing the number, order, and type of the parameters for the indexed property.</param>
        ///// <param name="args">Arguments to pass to the member to invoke.</param>
        //public void SetStaticProperty(string name, object value, Type[] parameterTypes, object[] args)
        //{
        //    this.SetStaticProperty(name, BindingFlags.SetProperty, value, parameterTypes, args);
        //}

        ///// <summary>
        ///// Gets the static property
        ///// </summary>
        ///// <param name="name">Name of the property</param>
        ///// <param name="bindingFlags">Additional invocation attributes.</param>
        ///// <param name="args">Arguments to pass to the member to invoke.</param>
        ///// <returns>The static property.</returns>
        //public object GetStaticProperty(string name, BindingFlags bindingFlags, params object[] args)
        //{
        //    return this.GetStaticProperty(name, BindingFlags.GetProperty | BindingFlags.Static | bindingFlags, null, args);
        //}

        ///// <summary>
        ///// Gets the static property
        ///// </summary>
        ///// <param name="name">Name of the property</param>
        ///// <param name="bindingFlags">Additional invocation attributes.</param>
        ///// <param name="parameterTypes">An array of <see cref="T:System.Type"/> objects representing the number, order, and type of the parameters for the indexed property.</param>
        ///// <param name="args">Arguments to pass to the member to invoke.</param>
        ///// <returns>The static property.</returns>
        //public object GetStaticProperty(string name, BindingFlags bindingFlags, Type[] parameterTypes, object[] args)
        //{
        //    Helper.CheckParameterNotNull(name, "name", string.Empty);
        //    if (parameterTypes != null)
        //    {
        //        PropertyInfo pi = this.type.GetProperty(name, bindingFlags | BindingFlags.Static, null, null, parameterTypes, null);
        //        if (pi == null)
        //        {
        //            throw new ArgumentException(string.Format(CultureInfo.CurrentCulture, FrameworkMessages.PrivateAccessorMemberNotFound, name));
        //        }

        //        return pi.GetValue(null, args);
        //    }
        //    else
        //    {
        //        return this.InvokeHelperStatic(name, bindingFlags | BindingFlags.GetProperty, args, null);
        //    }
        //}

        ///// <summary>
        ///// Sets the static property
        ///// </summary>
        ///// <param name="name">Name of the property</param>
        ///// <param name="bindingFlags">Additional invocation attributes.</param>
        ///// <param name="value">Value to be set to field or property</param>
        ///// <param name="args">Optional index values for indexed properties. The indexes of indexed properties are zero-based. This value should be null for non-indexed properties. </param>
        //public void SetStaticProperty(string name, BindingFlags bindingFlags, object value, params object[] args)
        //{
        //    this.SetStaticProperty(name, bindingFlags, value, null, args);
        //}

        ///// <summary>
        ///// Sets the static property
        ///// </summary>
        ///// <param name="name">Name of the property</param>
        ///// <param name="bindingFlags">Additional invocation attributes.</param>
        ///// <param name="value">Value to be set to field or property</param>
        ///// <param name="parameterTypes">An array of <see cref="T:System.Type"/> objects representing the number, order, and type of the parameters for the indexed property.</param>
        ///// <param name="args">Arguments to pass to the member to invoke.</param>
        //public void SetStaticProperty(string name, BindingFlags bindingFlags, object value, Type[] parameterTypes, object[] args)
        //{
        //    Helper.CheckParameterNotNull(name, "name", string.Empty);

        //    if (parameterTypes != null)
        //    {
        //        PropertyInfo pi = this.type.GetProperty(name, bindingFlags | BindingFlags.Static, null, null, parameterTypes, null);
        //        if (pi == null)
        //        {
        //            throw new ArgumentException(
        //                string.Format(CultureInfo.CurrentCulture, FrameworkMessages.PrivateAccessorMemberNotFound, name));
        //        }

        //        pi.SetValue(null, value, args);
        //    }
        //    else
        //    {
        //        object[] pass = new object[(args?.Length ?? 0) + 1];
        //        pass[0] = value;
        //        args?.CopyTo(pass, 1);
        //        this.InvokeHelperStatic(name, bindingFlags | BindingFlags.SetProperty, pass, null);
        //    }
        //}

        /// <summary>
        /// Invokes the static method
        /// </summary>
        /// <param name="name">Name of the member</param>
        /// <param name="bindingFlags">Additional invocation attributes</param>
        /// <param name="args">Arguements to the invocation</param>
        /// <param name="culture">Culture</param>
        /// <returns>Result of invocation</returns>
        private object InvokeHelperStatic(string name, BindingFlags bindingFlags, object[] args, CultureInfo culture)
        {
            Helper.CheckParameterNotNull(name, "name", string.Empty);
            try
            {
                return this.type.InvokeMember(name, bindingFlags | BindToEveryThing | BindingFlags.Static, null, null, args, culture);
            }
            catch (TargetInvocationException e)
            {
                //Debug.Assert(e.InnerException != null, "Inner Exception should not be null.");
                if (e.InnerException != null)
                {
                    throw e.InnerException;
                }

                throw;
            }
        }
    }

    /// <summary>
    /// The helper.
    /// </summary>
    internal static class Helper
    {
        /// <summary>
        /// The check parameter not null.
        /// </summary>
        /// <param name="param">
        /// The parameter.
        /// </param>
        /// <param name="parameterName">
        /// The parameter name.
        /// </param>
        /// <param name="message">
        /// The message.
        /// </param>
        /// <exception cref="ArgumentNullException"> Throws argument null exception when parameter is null. </exception>
        internal static void CheckParameterNotNull(object param, string parameterName, string message)
        {
            if (param == null)
            {
                throw new ArgumentNullException(parameterName, message);
            }
        }

        ///// <summary>
        ///// The check parameter not null or empty.
        ///// </summary>
        ///// <param name="param">
        ///// The parameter.
        ///// </param>
        ///// <param name="parameterName">
        ///// The parameter name.
        ///// </param>
        ///// <param name="message">
        ///// The message.
        ///// </param>
        ///// <exception cref="ArgumentException"> Throws ArgumentException when parameter is null. </exception>
        //internal static void CheckParameterNotNullOrEmpty(string param, string parameterName, string message)
        //{
        //    if (string.IsNullOrEmpty(param))
        //    {
        //        throw new ArgumentException(message, parameterName);
        //    }
        //}
    }
}
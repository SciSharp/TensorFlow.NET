// <copyright file="PrivateObjectExtensions.cs">
// Copyright (c) 2019 cactuaroid All Rights Reserved
// </copyright>
// <summary>
// Released under the MIT license
// https://github.com/cactuaroid/PrivateObjectExtensions
// </summary>

using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Linq;
using System.Reflection;

namespace System
{
    /// <summary>
    /// Extension methods for PrivateObject
    /// </summary>
    public static class PrivateObjectExtensions
    {
        private static readonly BindingFlags Static = BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly | BindingFlags.Static;
        private static readonly BindingFlags Instance = BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly | BindingFlags.Instance;

        /// <summary>
        /// Get from private (and any other) field/property.
        /// If the real type of specified object doesn't contain the specified field/property,
        /// base types are searched automatically.
        /// </summary>
        /// <param name="obj">The object to get from</param>
        /// <param name="name">The name of the field/property</param>
        /// <returns>The object got from the field/property</returns>
        /// <exception cref="ArgumentException">'name' is not found.</exception>
        /// <exception cref="ArgumentNullException">Arguments contain null.</exception>
        public static object GetPrivate(this object obj, string name)
        {
            if (obj == null) { throw new ArgumentNullException("obj"); }

            return GetPrivate(obj, name, obj.GetType(), null);
        }

        /// <summary>
        /// Get from private (and any other) field/property.
        /// If the real type of specified object doesn't contain the specified field/property,
        /// base types are searched automatically.
        /// </summary>
        /// <typeparam name="T">The type of the field/property</typeparam>
        /// <param name="obj">The object to get from</param>
        /// <param name="name">The name of the field/property</param>
        /// <returns>The object got from the field/property</returns>
        /// <exception cref="ArgumentException">'name' is not found.</exception>
        /// <exception cref="ArgumentNullException">Arguments contain null.</exception>
        public static T GetPrivate<T>(this object obj, string name)
        {
            if (obj == null) { throw new ArgumentNullException("obj"); }

            return (T)GetPrivate(obj, name, obj.GetType(), typeof(T));
        }

        /// <summary>
        /// Get from private (and any other) field/property with assuming the specified object as specified type.
        /// If the specified type doesn't contain the specified field/property,
        /// base types are searched automatically.
        /// </summary>
        /// <param name="obj">The object to get from</param>
        /// <param name="name">The name of the field/property</param>
        /// <param name="objType">The type of 'obj' for seaching member starting from. Real type of 'obj' is ignored.</param>
        /// <returns>The object got from the field/property</returns>
        /// <exception cref="ArgumentException">'name' is not found.</exception>
        /// <exception cref="ArgumentException">'objType' is not assignable from 'obj'.</exception>
        /// <exception cref="ArgumentNullException">Arguments contain null.</exception>
        public static object GetPrivate(this object obj, string name, Type objType)
        {
            return GetPrivate(obj, name, objType, null);
        }

        /// <summary>
        /// Get from private (and any other) field/property with assuming the specified object as specified type.
        /// If the specified type doesn't contain the specified field/property,
        /// base types are searched automatically.
        /// </summary>
        /// <typeparam name="T">The type of the field/property</typeparam>
        /// <param name="obj">The object to get from</param>
        /// <param name="name">The name of the field/property</param>
        /// <param name="objType">The type of 'obj' for seaching member starting from. Real type of 'obj' is ignored.</param>
        /// <returns>The object got from the field/property</returns>
        /// <exception cref="ArgumentException">'name' is not found.</exception>
        /// <exception cref="ArgumentException">'objType' is not assignable from 'obj'.</exception>
        /// <exception cref="ArgumentNullException">Arguments contain null.</exception>
        public static T GetPrivate<T>(this object obj, string name, Type objType)
        {
            return (T)GetPrivate(obj, name, objType, typeof(T));
        }

        private static object GetPrivate(object obj, string name, Type objType, Type memberType)
        {
            if (obj == null) { throw new ArgumentNullException("obj"); }
            if (name == null) { throw new ArgumentNullException("name"); }
            if (string.IsNullOrWhiteSpace(name)) { throw new ArgumentException("name is empty or white-space.", "name"); }
            if (objType == null) { throw new ArgumentNullException("objType"); }
            if (!objType.IsAssignableFrom(obj.GetType())) { throw new ArgumentException($"{objType} is not assignable from {obj.GetType()}.", "objType"); }

            bool memberTypeMatching(Type actualType) => actualType == memberType;

            if (TryFindFieldOrPropertyOwnerType(objType, name, memberType, memberTypeMatching, Instance, out var ownerType))
            {
                return new PrivateObject(obj, new PrivateType(ownerType)).GetFieldOrProperty(name);
            }
            else if (TryFindFieldOrPropertyOwnerType(objType, name, memberType, memberTypeMatching, Static, out ownerType))
            {
                return new PrivateType(ownerType).GetStaticFieldOrProperty(name);
            }

            throw new ArgumentException(((memberType != null) ? memberType + " " : "") + name + " is not found.");
        }

        /// <summary>
        /// Get from private (and any other) static field/property.
        /// </summary>
        /// <param name="type">The type to get from</param>
        /// <param name="name">The name of the static field/property</param>
        /// <returns>The object got from the static field/property</returns>
        /// <exception cref="ArgumentException">'name' is not found.</exception>
        /// <exception cref="ArgumentNullException">Arguments contain null.</exception>
        public static object GetPrivate(this Type type, string name)
        {
            return GetPrivate(type, name, null);
        }

        /// <summary>
        /// Get from private (and any other) static field/property.
        /// </summary>
        /// <typeparam name="T">The type of the field/property</typeparam>
        /// <param name="type">The type to get from</param>
        /// <param name="name">The name of the static field/property</param>
        /// <returns>The object got from the static field/property</returns>
        /// <exception cref="ArgumentException">'name' is not found.</exception>
        /// <exception cref="ArgumentNullException">Arguments contain null.</exception>
        public static T GetPrivate<T>(this Type type, string name)
        {
            return (T)GetPrivate(type, name, typeof(T));
        }

        private static object GetPrivate(this Type type, string name, Type memberType)
        {
            if (type == null) { throw new ArgumentNullException("type"); }
            if (name == null) { throw new ArgumentNullException("name"); }
            if (string.IsNullOrWhiteSpace(name)) { throw new ArgumentException("name is empty or white-space.", "name"); }

            bool memberTypeMatching(Type actualType) => actualType == memberType;

            if (type.ContainsFieldOrProperty(name, memberType, memberTypeMatching, Static))
            {
                return new PrivateType(type).GetStaticFieldOrProperty(name);
            }

            throw new ArgumentException(((memberType != null) ? memberType + " " : "") + name + " is not found.");
        }

        /// <summary>
        /// Set to private (and any other) field/property.
        /// If the real type of specified object doesn't contain the specified field/property,
        /// base types are searched automatically.
        /// </summary>
        /// <param name="obj">The object to set to</param>
        /// <param name="name">The name of the field/property</param>
        /// <param name="value">The value to set for 'name'</param>
        /// <exception cref="ArgumentException">'name' is not found.</exception>
        /// <exception cref="ArgumentNullException">Arguments contain null.</exception>
        public static void SetPrivate<T>(this object obj, string name, T value)
        {
            if (obj == null) { throw new ArgumentNullException("obj"); }

            SetPrivate(obj, name, value, obj.GetType());
        }

        /// <summary>
        /// Set to private (and any other) field/property with assuming the specified object as specified type.
        /// If the specified type doesn't contain the specified field/property,
        /// base types are searched automatically.
        /// </summary>
        /// <param name="obj">The object to set to</param>
        /// <param name="name">The name of the field/property</param>
        /// <param name="value">The value to set for 'name'</param>
        /// <param name="objType">The type of 'obj' for seaching member starting from. Real type of 'obj' is ignored.</param>
        /// <exception cref="ArgumentException">'name' is not found.</exception>
        /// <exception cref="ArgumentException">'objType' is not assignable from 'obj'.</exception>
        /// <exception cref="ArgumentNullException">Arguments contain null.</exception>
        public static void SetPrivate<T>(this object obj, string name, T value, Type objType)
        {
            if (obj == null) { throw new ArgumentNullException("obj"); }
            if (name == null) { throw new ArgumentNullException("name"); }
            if (string.IsNullOrWhiteSpace(name)) { throw new ArgumentException("name is empty or white-space.", "name"); }
            if (value == null) { throw new ArgumentNullException("value"); }
            if (objType == null) { throw new ArgumentNullException("objType"); }
            if (!objType.IsAssignableFrom(obj.GetType())) { throw new ArgumentException($"{objType} is not assignable from {obj.GetType()}.", "objType"); }

            if (TrySetPrivate(obj, name, value, objType)) { return; }

            // retry for the case of getter only property
            if (TrySetPrivate(obj, GetBackingFieldName(name), value, objType)) { return; }

            throw new ArgumentException($"{typeof(T)} {name} is not found.");
        }

        private static bool TrySetPrivate<T>(object obj, string name, T value, Type objType)
        {
            var memberType = typeof(T);
            bool memberTypeMatching(Type actualType) => actualType.IsAssignableFrom(memberType);

            try
            {
                if (TryFindFieldOrPropertyOwnerType(objType, name, memberType, memberTypeMatching, Instance, out var ownerType))
                {
                    new PrivateObject(obj, new PrivateType(ownerType)).SetFieldOrProperty(name, value);
                    return true;
                }
                else if (TryFindFieldOrPropertyOwnerType(objType, name, memberType, memberTypeMatching, Static, out ownerType))
                {
                    new PrivateType(ownerType).SetStaticFieldOrProperty(name, value);
                    return true;
                }
            }
            catch (MissingMethodException)
            {
                // When getter only property name is given, the property is found but fails to set.
                return false;
            }

            return false;
        }

        /// <summary>
        /// Set to private (and any other) static field/property.
        /// </summary>
        /// <param name="type">The type to set to</param>
        /// <param name="name">The name of the field/property</param>
        /// <param name="value">The value to set for 'name'</param>
        /// <exception cref="ArgumentException">'name' is not found.</exception>
        /// <exception cref="ArgumentNullException">Arguments contain null.</exception>
        public static void SetPrivate<T>(this Type type, string name, T value)
        {
            if (type == null) { throw new ArgumentNullException("type"); }
            if (name == null) { throw new ArgumentNullException("name"); }
            if (string.IsNullOrWhiteSpace(name)) { throw new ArgumentException("name is empty or white-space.", "name"); }

            if (TrySetPrivate(type, name, value)) { return; }

            // retry for the case of getter only property
            if (TrySetPrivate(type, GetBackingFieldName(name), value)) { return; }

            throw new ArgumentException($"{typeof(T)} {name} is not found.");
        }

        private static bool TrySetPrivate<T>(this Type type, string name, T value)
        {
            var memberType = typeof(T);
            bool memberTypeMatching(Type actualType) => actualType.IsAssignableFrom(memberType);

            try
            {
                if (type.ContainsFieldOrProperty(name, memberType, memberTypeMatching, Static))
                {
                    new PrivateType(type).SetStaticFieldOrProperty(name, value);
                    return true;
                }
            }
            catch (MissingMethodException)
            {
                // When getter only property name is given, the property is found but fails to set.
                return false;
            }

            return false;
        }

        private static string GetBackingFieldName(string propertyName)
            => $"<{propertyName}>k__BackingField"; // generated backing field name

        private static bool TryFindFieldOrPropertyOwnerType(Type objType, string name, Type memberType, Func<Type, bool> memberTypeMatching, BindingFlags bindingFlag, out Type ownerType)
        {
            ownerType = FindFieldOrPropertyOwnerType(objType, name, memberType, memberTypeMatching, bindingFlag);

            return (ownerType != null);
        }

        private static Type FindFieldOrPropertyOwnerType(Type objectType, string name, Type memberType, Func<Type, bool> memberTypeMatching, BindingFlags bindingFlags)
        {
            if (objectType == null) { return null; }

            if (objectType.ContainsFieldOrProperty(name, memberType, memberTypeMatching, bindingFlags))
            {
                return objectType;
            }

            return FindFieldOrPropertyOwnerType(objectType.BaseType, name, memberType, memberTypeMatching, bindingFlags);
        }

        private static bool ContainsFieldOrProperty(this Type objectType, string name, Type memberType, Func<Type, bool> memberTypeMatching, BindingFlags bindingFlags)
        {
            var fields = objectType
                .GetFields(bindingFlags)
                .Select((x) => new { Type = x.FieldType, Member = x as MemberInfo });

            var properties = objectType
                .GetProperties(bindingFlags)
                .Select((x) => new { Type = x.PropertyType, Member = x as MemberInfo });

            var members = fields.Concat(properties);

            return members.Any((actual) =>
                (memberType == null || memberTypeMatching.Invoke(actual.Type))
                && actual.Member.Name == name);
        }
    }
}
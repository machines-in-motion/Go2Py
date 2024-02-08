// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from unitree_go:msg/TimeSpec.idl
// generated code does not contain a copyright notice

#ifndef UNITREE_GO__MSG__DETAIL__TIME_SPEC__TRAITS_HPP_
#define UNITREE_GO__MSG__DETAIL__TIME_SPEC__TRAITS_HPP_

#include "unitree_go/msg/detail/time_spec__struct.hpp"
#include <rosidl_runtime_cpp/traits.hpp>
#include <stdint.h>
#include <type_traits>

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<unitree_go::msg::TimeSpec>()
{
  return "unitree_go::msg::TimeSpec";
}

template<>
inline const char * name<unitree_go::msg::TimeSpec>()
{
  return "unitree_go/msg/TimeSpec";
}

template<>
struct has_fixed_size<unitree_go::msg::TimeSpec>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<unitree_go::msg::TimeSpec>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<unitree_go::msg::TimeSpec>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // UNITREE_GO__MSG__DETAIL__TIME_SPEC__TRAITS_HPP_
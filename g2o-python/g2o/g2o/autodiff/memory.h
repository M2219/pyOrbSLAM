// Copyright 2017 The Abseil Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// -----------------------------------------------------------------------------
// File: memory.h
// -----------------------------------------------------------------------------
//
// This header file contains utility functions for managing the creation and
// conversion of smart pointers. This file is an extension to the C++
// standard <memory> library header file.

#ifndef G2O_CERES_PUBLIC_INTERNAL_MEMORY_H_
#define G2O_CERES_PUBLIC_INTERNAL_MEMORY_H_

#include <memory>

#ifdef G2O_CERES_HAVE_EXCEPTIONS
#define G2O_CERES_INTERNAL_TRY try
#define G2O_CERES_INTERNAL_CATCH_ANY catch (...)
#define G2O_CERES_INTERNAL_RETHROW \
  do {                             \
    throw;                         \
  } while (false)
#else  // G2O_CERES_HAVE_EXCEPTIONS
#define G2O_CERES_INTERNAL_TRY if (true)
#define G2O_CERES_INTERNAL_CATCH_ANY else if (false)
#define G2O_CERES_INTERNAL_RETHROW \
  do {                             \
  } while (false)
#endif  // G2O_CERES_HAVE_EXCEPTIONS

namespace g2o::ceres::internal {

template <typename Allocator, typename Iterator, typename... Args>
void ConstructRange(Allocator& alloc, Iterator first, Iterator last,
                    const Args&... args) {
  for (Iterator cur = first; cur != last; ++cur) {
    G2O_CERES_INTERNAL_TRY {
      std::allocator_traits<Allocator>::construct(alloc, std::addressof(*cur),
                                                  args...);
    }
    G2O_CERES_INTERNAL_CATCH_ANY {
      while (cur != first) {
        --cur;
        std::allocator_traits<Allocator>::destroy(alloc, std::addressof(*cur));
      }
      G2O_CERES_INTERNAL_RETHROW;
    }
  }
}

template <typename Allocator, typename Iterator, typename InputIterator>
void CopyRange(Allocator& alloc, Iterator destination, InputIterator first,
               InputIterator last) {
  for (Iterator cur = destination; first != last;
       static_cast<void>(++cur), static_cast<void>(++first)) {
    G2O_CERES_INTERNAL_TRY {
      std::allocator_traits<Allocator>::construct(alloc, std::addressof(*cur),
                                                  *first);
    }
    G2O_CERES_INTERNAL_CATCH_ANY {
      while (cur != destination) {
        --cur;
        std::allocator_traits<Allocator>::destroy(alloc, std::addressof(*cur));
      }
      G2O_CERES_INTERNAL_RETHROW;
    }
  }
}

}  // namespace g2o::ceres::internal

#endif  // G2O_CERES_PUBLIC_INTERNAL_MEMORY_H_

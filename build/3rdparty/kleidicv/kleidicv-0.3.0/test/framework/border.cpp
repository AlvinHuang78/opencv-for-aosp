// SPDX-FileCopyrightText: 2023 - 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "framework/border.h"

#include <gtest/gtest.h>

#include <algorithm>

namespace test {

// Creates replicated border elements.
//
// Replicating means that the elements at the edges are copied to bordering
// element positions. For example:
// | left border | elements  | right border |
// |         A A | A B C D E | E E          |
template <typename ElementType>
static void replicate(const Bordered *bordered,
                      TwoDimensional<ElementType> *elements) {
  ASSERT_LE((bordered->left() + bordered->right()) * elements->channels(),
            elements->width());
  ASSERT_LE(bordered->top() + bordered->bottom(), elements->height());

  // Replicate left and right border columns.
  for (size_t row = 0; row < elements->height(); ++row) {
    for (size_t channel = 0; channel < elements->channels(); ++channel) {
      // Prepare left border columns.
      for (size_t column = 0; column < bordered->left(); ++column) {
        size_t src_column = bordered->left() * elements->channels() + channel;
        size_t dst_column = column * elements->channels() + channel;
        elements->at(row, dst_column)[0] = elements->at(row, src_column)[0];
      }

      // Prepare right border columns.
      for (size_t column = 0; column < bordered->right(); ++column) {
        size_t src_column = elements->width() -
                            (bordered->right() + 1) * elements->channels() +
                            channel;
        size_t dst_column = src_column + (1 + column) * elements->channels();
        elements->at(row, dst_column)[0] = elements->at(row, src_column)[0];
      }
    }
  }

  // Replicate top border rows.
  size_t replicated_top_row = bordered->top();
  for (size_t row = 0; row < bordered->top(); ++row) {
    auto row_ptr = elements->at(replicated_top_row, 0);
    std::copy(row_ptr, row_ptr + elements->width(), elements->at(row, 0));
  }

  // Replicate bottom border rows.
  size_t replicated_bottom_row = elements->height() - bordered->bottom() - 1;
  for (size_t row = 0; row < bordered->bottom(); ++row) {
    auto row_ptr = elements->at(replicated_bottom_row, 0);
    std::copy(row_ptr, row_ptr + elements->width(),
              elements->at(replicated_bottom_row + row + 1, 0));
  }
}

// Creates constant border elements.
//
// Constant borders use the given values for top, left, right and bottom border.
// For example:
// | left border | elements  | right border |
// |         X X | A B C D E | Y Y          |
template <typename ElementType>
static void constant(const Bordered *bordered,
                     kleidicv_border_values_t border_values,
                     TwoDimensional<ElementType> *elements) {
  ASSERT_LE((bordered->left() + bordered->right()) * elements->channels(),
            elements->width());
  ASSERT_LE(bordered->top() + bordered->bottom(), elements->height());

  // Constant left and right border columns.
  for (size_t row = 0; row < elements->height(); ++row) {
    for (size_t channel = 0; channel < elements->channels(); ++channel) {
      // Prepare left border columns.
      for (size_t column = 0; column < bordered->left(); ++column) {
        size_t dst_column = column * elements->channels() + channel;
        elements->at(row, dst_column)[0] = border_values.left;
      }

      // Prepare right border columns.
      for (size_t column = 0; column < bordered->right(); ++column) {
        size_t dst_column =
            elements->width() +
            (column - bordered->right()) * elements->channels() + channel;
        elements->at(row, dst_column)[0] = border_values.right;
      }
    }
  }

  // Constant top border rows.
  for (size_t row = 0; row < bordered->top(); ++row) {
    for (size_t column = 0; column < elements->width(); ++column) {
      elements->at(row, column)[0] = border_values.top;
    }
  }

  // Constant bottom border rows.
  for (size_t row = elements->height() - bordered->bottom();
       row < elements->height(); ++row) {
    for (size_t column = 0; column < elements->width(); ++column) {
      elements->at(row, column)[0] = border_values.bottom;
    }
  }
}

// Creates reflected border elements.
//
// Reflecting means that the border mirrors the elements of the array.
// For example:
// | left border | elements  | right border |
// |       C B A | A B C D E | E D C        |
template <typename ElementType>
static void reflect(const Bordered *bordered,
                    TwoDimensional<ElementType> *elements) {
  ASSERT_LE((bordered->left() + bordered->right()) * elements->channels(),
            elements->width());
  ASSERT_LE(bordered->top() + bordered->bottom(), elements->height());

  // Reflect left and right border columns.
  for (size_t row = 0; row < elements->height(); ++row) {
    for (size_t channel = 0; channel < elements->channels(); ++channel) {
      // Prepare left border columns.
      for (size_t column = 0; column < bordered->left(); ++column) {
        size_t src_column =
            (bordered->left() + column) * elements->channels() + channel;
        size_t dst_column =
            (bordered->left() - 1 - column) * elements->channels() + channel;
        elements->at(row, dst_column)[0] = elements->at(row, src_column)[0];
      }

      // Prepare right border columns.
      for (size_t column = 0; column < bordered->right(); ++column) {
        size_t src_column =
            elements->width() -
            (bordered->right() + 1 + column) * elements->channels() + channel;
        size_t dst_column =
            elements->width() -
            (bordered->right() - column) * elements->channels() + channel;
        elements->at(row, dst_column)[0] = elements->at(row, src_column)[0];
      }
    }
  }

  // Reflect top border rows.
  size_t reflected_top_row = bordered->top();
  for (size_t row = 0; row < bordered->top(); ++row) {
    auto row_ptr = elements->at(reflected_top_row + row, 0);
    std::copy(row_ptr, row_ptr + elements->width(),
              elements->at(bordered->top() - 1 - row, 0));
  }

  // Reflect bottom border rows.
  size_t reflected_bottom_row = elements->height() - bordered->bottom() - 1;
  for (size_t row = 0; row < bordered->bottom(); ++row) {
    auto row_ptr = elements->at(reflected_bottom_row - row, 0);
    std::copy(row_ptr, row_ptr + elements->width(),
              elements->at(reflected_bottom_row + row + 1, 0));
  }
}

// Creates wrapping border elements.
//
// Wrapping means that the border 'continues' the elements of the array.
// For example:
// | left border | elements  | right border |
// |       C D E | A B C D E | A B C        |
template <typename ElementType>
static void wrap(const Bordered *bordered,
                 TwoDimensional<ElementType> *elements) {
  ASSERT_LE((bordered->left() + bordered->right()) * elements->channels(),
            elements->width());
  ASSERT_LE(bordered->top() + bordered->bottom(), elements->height());

  // Wrap left and right border columns.
  for (size_t row = 0; row < elements->height(); ++row) {
    for (size_t channel = 0; channel < elements->channels(); ++channel) {
      // Prepare left border columns.
      for (size_t column = 0; column < bordered->left(); ++column) {
        size_t src_column =
            elements->width() -
            (bordered->right() + 1 + column) * elements->channels() + channel;
        size_t dst_column =
            (bordered->left() - 1 - column) * elements->channels() + channel;
        elements->at(row, dst_column)[0] = elements->at(row, src_column)[0];
      }

      // Prepare right border columns.
      for (size_t column = 0; column < bordered->right(); ++column) {
        size_t src_column =
            (bordered->left() + column) * elements->channels() + channel;
        size_t dst_column =
            elements->width() -
            (bordered->right() - column) * elements->channels() + channel;
        elements->at(row, dst_column)[0] = elements->at(row, src_column)[0];
      }
    }
  }

  // Wrap top border rows.
  size_t wrapped_top_row = elements->height() - bordered->bottom() - 1;
  for (size_t row = 0; row < bordered->top(); ++row) {
    auto row_ptr = elements->at(wrapped_top_row - row, 0);
    std::copy(row_ptr, row_ptr + elements->width(),
              elements->at(bordered->top() - 1 - row, 0));
  }

  // Wrap bottom border rows.
  size_t wrapped_bottom_row = bordered->top();
  for (size_t row = 0; row < bordered->bottom(); ++row) {
    auto row_ptr = elements->at(wrapped_bottom_row + row, 0);
    std::copy(row_ptr, row_ptr + elements->width(),
              elements->at(wrapped_top_row + row + 1, 0));
  }
}

// Creates reversed border elements.

// Reversing means that the border mirrors the elements of the array,
// skipping the elements on the edge.
// For example:
// | left border | elements  | right border |
// |       D C B | A B C D E | D C B        |
template <typename ElementType>
static void reverse(const Bordered *bordered,
                    TwoDimensional<ElementType> *elements) {
  ASSERT_LE((bordered->left() + bordered->right()) * elements->channels(),
            elements->width());
  ASSERT_LE((bordered->left() + 1) * elements->channels(), elements->width());
  ASSERT_LE((bordered->right() + 1) * elements->channels(), elements->width());
  ASSERT_LE(bordered->top() + bordered->bottom(), elements->height());
  ASSERT_LE(bordered->top() + 1, elements->height());
  ASSERT_LE(bordered->bottom() + 1, elements->height());

  // Reverse left and right border columns.
  for (size_t row = 0; row < elements->height(); ++row) {
    for (size_t channel = 0; channel < elements->channels(); ++channel) {
      // Prepare left border columns.
      for (size_t column = 0; column < bordered->left(); ++column) {
        size_t src_column =
            (bordered->left() + column + 1) * elements->channels() + channel;
        size_t dst_column =
            (bordered->left() - 1 - column) * elements->channels() + channel;
        elements->at(row, dst_column)[0] = elements->at(row, src_column)[0];
      }

      // Prepare right border columns.
      for (size_t column = 0; column < bordered->right(); ++column) {
        size_t src_column =
            elements->width() -
            (bordered->right() + 2 + column) * elements->channels() + channel;
        size_t dst_column =
            elements->width() -
            (bordered->right() - column) * elements->channels() + channel;
        elements->at(row, dst_column)[0] = elements->at(row, src_column)[0];
      }
    }
  }

  // Reverse top border rows.
  size_t reversed_top_row = bordered->top() + 1;
  for (size_t row = 0; row < bordered->top(); ++row) {
    auto row_ptr = elements->at(reversed_top_row + row, 0);
    std::copy(row_ptr, row_ptr + elements->width(),
              elements->at(bordered->top() - 1 - row, 0));
  }

  // Reverse bottom border rows.
  size_t reversed_bottom_row = elements->height() - bordered->bottom() - 2;
  for (size_t row = 0; row < bordered->bottom(); ++row) {
    auto row_ptr = elements->at(reversed_bottom_row - row, 0);
    std::copy(row_ptr, row_ptr + elements->width(),
              elements->at(reversed_bottom_row + row + 2, 0));
  }
}

template <typename ElementType>
void prepare_borders(kleidicv_border_type_t border_type,
                     kleidicv_border_values_t border_values,
                     const Bordered *bordered,
                     TwoDimensional<ElementType> *elements) {
  ASSERT_NE(bordered, nullptr);
  ASSERT_NE(elements, nullptr);

  switch (border_type) {
    default:
      GTEST_FAIL() << "Border type is not implemented.";

    case KLEIDICV_BORDER_TYPE_REPLICATE:
      return replicate(bordered, elements);

    case KLEIDICV_BORDER_TYPE_CONSTANT:
      return constant(bordered, border_values, elements);

    case KLEIDICV_BORDER_TYPE_REFLECT:
      return reflect(bordered, elements);

    case KLEIDICV_BORDER_TYPE_WRAP:
      return wrap(bordered, elements);

    case KLEIDICV_BORDER_TYPE_REVERSE:
      return reverse(bordered, elements);
  }
}

template void prepare_borders<uint8_t>(kleidicv_border_type_t,
                                       kleidicv_border_values_t,
                                       const Bordered *,
                                       TwoDimensional<uint8_t> *);

template void prepare_borders<uint16_t>(kleidicv_border_type_t,
                                        kleidicv_border_values_t,
                                        const Bordered *,
                                        TwoDimensional<uint16_t> *);

template void prepare_borders<int16_t>(kleidicv_border_type_t,
                                       kleidicv_border_values_t,
                                       const Bordered *,
                                       TwoDimensional<int16_t> *);

}  // namespace test

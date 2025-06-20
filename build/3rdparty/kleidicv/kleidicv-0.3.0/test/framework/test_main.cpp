// SPDX-FileCopyrightText: 2023 - 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <getopt.h>
#include <gtest/gtest.h>

#include <iostream>
#include <random>
#include <string>

#include "framework/utils.h"

namespace test {

size_t Options::vector_length_{16};
uint64_t Options::seed_{0};

// Parses command line arguments.
static void parse_arguments(int argc, char **argv) {
  // clang-format off
  static struct option long_options[] = {
    {"vector-length", required_argument, nullptr, 'v'},
    {"seed", required_argument, nullptr, 's'},
    {"long-running-tests", no_argument, nullptr, 'l'},
    {nullptr, 0, nullptr, 0}
  };
  // clang-format on

  bool is_seed_set = false;

  for (;;) {
    int option_index = 0;
    int opt = getopt_long(argc, argv, "", long_options, &option_index);
    if (opt == -1) {
      break;
    }

    switch (opt) {
      default:
        std::cerr << "Unknown command line argument." << std::endl;
        break;

      case 'v':
        Options::set_vector_length(std::stoi(optarg));
        break;

      case 's':
        Options::set_seed(std::stoull(optarg));
        is_seed_set = true;
        break;

      case 'l':
        Options::turn_on_long_running_tests();
        break;
    }
  }

  // Set a random seed if it is not explicitly specified.
  if (!is_seed_set) {
    std::random_device rd;
    Options::set_seed(static_cast<uint64_t>(rd()));
  }

  std::cout << "Vector length is set to " << Options::vector_length()
            << " bytes." << std::endl;
  std::cout << "Seed is set to " << Options::seed() << "." << std::endl;
}

}  // namespace test

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  test::parse_arguments(argc, argv);
  return RUN_ALL_TESTS();
}

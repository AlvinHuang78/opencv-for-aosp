// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

/*
To learn how to use this program see the Google Benchmark user guide:
https://github.com/google/benchmark/blob/main/docs/user_guide.md

In addition to the regular Google Benchmark command line options, you can also
set the image size on which the tests will be run with the command line options
--image_width=123 and --image_height=123.
*/

#include <benchmark/benchmark.h>

#include <charconv>
#include <iostream>
#include <string_view>

size_t image_width = 1280, image_height = 720;

bool get_argument(std::string_view arg, std::string_view prefix,
                  size_t& value) {
  if (arg.substr(0, prefix.size()) != prefix) {
    return false;
  }

  auto value_str = arg.substr(prefix.size());

  auto result = std::from_chars(value_str.data(),
                                value_str.data() + value_str.size(), value);

  if (result.ec == std::errc::invalid_argument) {
    std::cerr << "Invalid argument: " << arg << std::endl;
    exit(1);
  }

  return true;
}

int main(int argc, char** argv) {
  for (int i = 1; i < argc;) {
    std::string_view arg{argv[i]};
    if (get_argument(arg, "--image_width=", image_width) ||
        get_argument(arg, "--image_height=", image_height)) {
      // Avoid passing the argument to Google Benchmark.
      --argc;
      for (int j = i; j < argc; ++j) {
        argv[j] = argv[j + 1];
      }
    } else {
      ++i;
    }
  }

  ::benchmark::Initialize(&argc, argv);
  if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;

  std::cout << "KleidiCV: Running benchmarks with:\n";
  std::cout << "  image_width: " << image_width << "\n";
  std::cout << "  image_height: " << image_height << std::endl;

  ::benchmark::RunSpecifiedBenchmarks();
  ::benchmark::Shutdown();
  return 0;
}

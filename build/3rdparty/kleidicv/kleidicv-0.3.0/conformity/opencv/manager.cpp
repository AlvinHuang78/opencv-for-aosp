// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <csignal>
#include <iostream>

#include "common.h"
#include "utils.h"

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Error! Subordinate task is not defined as the first argument!"
              << std::endl;
    return -1;
  }

  // Block USR1 signal as it terminates the process by default
  sigset_t usr1_sigset;
  sigemptyset(&usr1_sigset);
  sigaddset(&usr1_sigset, SIGUSR1);
  sigprocmask(SIG_BLOCK, &usr1_sigset, NULL);

  pid_t child_pid = fork();
  if (child_pid == 0) {
    // Waiting for the initialization of manager task
    timespec timeout = {3, 0};
    if (sigtimedwait(&usr1_sigset, NULL, &timeout) != SIGUSR1) {
      std::cerr
          << "Error! Wrong signal received or timeout reached in subordinate!"
          << std::endl;
      return -2;
    }
    // Starting subordinate task
    execl(argv[1], argv[1], static_cast<char*>(NULL));
    throw ExceptionWithErrno("Cannot start subordinate executable");
  }

  RecreatedSharedMemory sm{KLEIDICV_CONFORMITY_SHM_ID,
                           KLEIDICV_CONFORMITY_SHM_SIZE};
  RecreatedMessageQueue request_queue{KLEIDICV_CONFORMITY_REQUEST_MQ_ID, sm};
  RecreatedMessageQueue reply_queue{KLEIDICV_CONFORMITY_REPLY_MQ_ID, sm};

  // Let subordinate know that init is done
  kill(child_pid, SIGUSR1);

  int test_results = run_tests(request_queue, reply_queue);

  // Wait for subordinate to exit
  wait(NULL);

  std::cout << "Manager exits normally" << std::endl;

  return test_results;
}

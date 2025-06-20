// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_OPENCV_CONFORMITY_COMMON_H_
#define KLEIDICV_OPENCV_CONFORMITY_COMMON_H_

#include <fcntl.h>
#include <mqueue.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <ctime>
#include <exception>
#include <limits>
#include <string>
#include <type_traits>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/video.hpp"

#define KLEIDICV_CONFORMITY_SHM_ID "/opencv_kleidicv_conformity_check_shm"
#define KLEIDICV_CONFORMITY_SHM_SIZE (1024 * 1024)

#define KLEIDICV_CONFORMITY_REQUEST_MQ_ID \
  "/opencv_kleidicv_conformity_request_queue"
#define KLEIDICV_CONFORMITY_REPLY_MQ_ID \
  "/opencv_kleidicv_conformity_reply_queue"

#define KLEIDICV_CONFORMITY_MAX_MAT_DIMENSIONS 4

static float floatval(uint32_t v) {
  float result;  // Avoid cppcoreguidelines-init-variables. NOLINT
  static_assert(sizeof(result) == sizeof(v));
  memcpy(&result, &v, sizeof(result));
  return result;
}

const float quietNaN = std::numeric_limits<float>::quiet_NaN();
const float signalingNaN = std::numeric_limits<float>::signaling_NaN();
const float posInfinity = std::numeric_limits<float>::infinity();
const float negInfinity = -std::numeric_limits<float>::infinity();

const float minusNaN = floatval(0xFF800001);
const float plusNaN = floatval(0x7F800001);
const float plusZero = 0.0F;
const float minusZero = -0.0F;

const float oneNaN = floatval(0x7FC00001);
const float zeroDivZero = -std::numeric_limits<float>::quiet_NaN();
const float floatMin = std::numeric_limits<float>::lowest();
const float floatMax = std::numeric_limits<float>::max();

const float posSubnormalMin = std::numeric_limits<float>::denorm_min();
const float posSubnormalMax = floatval(0x007FFFFF);
const float negSubnormalMin = -std::numeric_limits<float>::denorm_min();
const float negSubnormalMax = floatval(0x807FFFFF);

static constexpr int custom_data_float_height = 8;
static constexpr int custom_data_float_width = 4;

static float
    custom_data_float[custom_data_float_height * custom_data_float_width] = {
        // clang-format off
        quietNaN, signalingNaN, posInfinity, negInfinity,
        minusNaN, plusNaN, plusZero, minusZero,
        oneNaN, zeroDivZero, floatMin, floatMax,
        posSubnormalMin, posSubnormalMax, negSubnormalMin, negSubnormalMax,
        1111.11, -1112.22, 113.33, 114.44,
        111.51, 112.62, 113.73, 114.84,
        126.66, 127.11, 128.66, 129.11,
        11.5, 12.5, -11.5, -12.5,
        // clang-format on
};

template <typename T>
static constexpr T lowest() {
  return std::numeric_limits<T>::lowest();
}

template <typename T>
static constexpr T max() {
  return std::numeric_limits<T>::max();
}

class ExceptionWithErrno : public std::exception {
 public:
  explicit ExceptionWithErrno(const std::string& msg)
      : msg_with_errno_{add_errno_details(msg)} {}
  virtual const char* what() const noexcept { return msg_with_errno_.c_str(); }

 private:
  std::string add_errno_details(const std::string& msg) {
    std::string errno_string(strerror(errno));
    return msg + ": " + errno_string;
  }

  std::string msg_with_errno_;
};  // end of class ExceptionWithErrno

// Class to provide a file descriptor created with shm_open()
template <bool Recreated>
class ShmFD {
 public:
  template <bool check = Recreated>
  explicit ShmFD(std::enable_if_t<!check, const std::string&> id)
      : id_{}, fd_{open(id)} {}

  template <bool check = Recreated>
  explicit ShmFD(std::enable_if_t<check, const std::string&> id)
      : id_{id}, fd_{unlink_and_open(id)} {}

  virtual ~ShmFD() {
    close(fd_);
    if (Recreated) {
      shm_unlink(id_.c_str());
    }
  }

  // Disable copying
  ShmFD(ShmFD const&) = delete;
  ShmFD& operator=(ShmFD) = delete;

  int fd() const { return fd_; }

 private:
  static int open(const std::string& id) {
    int fd = shm_open(id.c_str(), O_RDWR, 0666);
    if (fd < 0) {
      throw ExceptionWithErrno("Cannot open shared memory, id: " + id);
    }
    return fd;
  }

  static int unlink_and_open(const std::string& id) {
    if (shm_unlink(id.c_str())) {
      if (errno != ENOENT) {
        throw ExceptionWithErrno("Cannot delete shared memory, id: " + id);
      }
    }
    int fd = shm_open(id.c_str(), O_RDWR | O_CREAT | O_EXCL, 0666);
    if (fd < 0) {
      throw ExceptionWithErrno("Cannot open shared memory, id: " + id);
    }
    return fd;
  }

  const std::string id_;
  int fd_;
};  // end of class ShmFD<Recreated>

// Class to provide mapped shared memory
template <bool Recreated>
class SharedMemory {
 public:
  explicit SharedMemory(const std::string& id, size_t size)
      : mem_{nullptr}, size_{size}, shm_fd_{id} {
    if (ftruncate(shm_fd_.fd(), size)) {
      throw ExceptionWithErrno("Failed to set the size of shared memory, id: " +
                               id);
    }

    mem_ =
        mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd_.fd(), 0);
    if (mem_ == MAP_FAILED) {
      throw ExceptionWithErrno("Failed to map shared memory, id: " + id);
    }
  }

  virtual ~SharedMemory() { munmap(mem_, size_); }

  // Disable copying
  SharedMemory(SharedMemory const&) = delete;
  SharedMemory& operator=(SharedMemory) = delete;

  cv::Mat cv_mat(int ndims, const int* sizes, int mat_type) {
    size_t requested_size = cv::Mat(1, 1, mat_type).elemSize();
    for (int i = 0; i < ndims; ++i) requested_size *= sizes[i];
    if (requested_size > size_) {
      throw std::runtime_error(
          "Requested matrix is bigger than the shared memory size");
    }
    return cv::Mat(ndims, sizes, mat_type, mem_);
  }

  void store_mat(const cv::Mat& mat) {
    size_t matrix_size = mat.total() * mat.elemSize();
    if (matrix_size > size_) {
      throw std::runtime_error(
          "Input matrix is bigger than the shared memory size");
    }
    memcpy(mem_, reinterpret_cast<const void*>(mat.ptr()), matrix_size);
  }

 private:
  void* mem_;
  size_t size_;
  ShmFD<Recreated> shm_fd_;
};  // end of class SharedMemory<Recreated>

using OpenedSharedMemory = SharedMemory<false>;
using RecreatedSharedMemory = SharedMemory<true>;

// Class to provide a message queue
template <bool Recreated>
class MessageQueue {
 public:
  template <bool check = Recreated>
  explicit MessageQueue(std::enable_if_t<!check, const std::string&> id,
                        SharedMemory<false>& sm)
      : id_{}, queue_desc_{open(id)}, sm_{sm} {}

  template <bool check = Recreated>
  explicit MessageQueue(std::enable_if_t<check, const std::string&> id,
                        SharedMemory<true>& sm)
      : id_{id}, queue_desc_{unlink_and_open(id)}, sm_{sm} {}

  virtual ~MessageQueue() {
    mq_close(queue_desc_);
    if (Recreated) {
      mq_unlink(id_.c_str());
    }
  }

  // Disable copying
  MessageQueue(MessageQueue const&) = delete;
  MessageQueue& operator=(MessageQueue) = delete;

  void request_exit() {
    message m = {};
    m.cmd = -1;
    send(m);
  }

  void request_operation(int cmd, const cv::Mat& mat) {
    sm_.store_mat(mat);
    message m;
    m.cmd = cmd;
    m.type = mat.type();
    m.ndims = mat.dims;
    memcpy(m.sizes, mat.size.p, sizeof(int) * mat.dims);
    send(m);
  }

  void reply_operation(int cmd, const cv::Mat& mat) {
    request_operation(cmd, mat);
  }

  void wait() {
    timespec abs_timeout;
    clock_gettime(CLOCK_REALTIME, &abs_timeout);
    abs_timeout.tv_sec += 3;
    ssize_t read_bytes =
        mq_timedreceive(queue_desc_, reinterpret_cast<char*>(&last_message_),
                        sizeof(last_message_), nullptr, &abs_timeout);
    if (read_bytes != sizeof(last_message_)) {
      if (read_bytes == -1) {
        throw ExceptionWithErrno("Could not receive message");
      } else {
        throw std::runtime_error("Less bytes received than expected");
      }
    }
  }

  int last_cmd() const { return last_message_.cmd; }

  cv::Mat cv_mat_from_last_msg() const {
    return sm_.cv_mat(last_message_.ndims, last_message_.sizes,
                      last_message_.type);
  }

 private:
  struct message {
    int cmd;
    int type;
    int ndims;
    int sizes[KLEIDICV_CONFORMITY_MAX_MAT_DIMENSIONS];
  };

  static mqd_t open(const std::string& id) {
    mqd_t qd = mq_open(id.c_str(), O_RDWR);
    if (qd == static_cast<mqd_t>(-1)) {
      throw ExceptionWithErrno("Failed to open message queue, id:" + id);
    }

    return qd;
  }
  static mqd_t unlink_and_open(const std::string& id) {
    if (mq_unlink(id.c_str())) {
      if (errno != ENOENT) {
        throw ExceptionWithErrno("Cannot delete message queue, id: " + id);
      }
    }

    mq_attr attr = queue_attributes();
    mqd_t qd = mq_open(id.c_str(), O_RDWR | O_CREAT | O_EXCL, 0666, &attr);
    if (qd == static_cast<mqd_t>(-1)) {
      throw ExceptionWithErrno("Failed to open message queue, id:" + id);
    }

    return qd;
  }

  void send(message& m) const {
    if (mq_send(queue_desc_, reinterpret_cast<const char*>(&m), sizeof(m), 0)) {
      throw ExceptionWithErrno("Failed to send message on queue");
    }
  }

  static mq_attr queue_attributes() {
    mq_attr attr;
    attr.mq_maxmsg = 1;
    attr.mq_msgsize = sizeof(message);
    return attr;
  }

  const std::string id_;
  mqd_t queue_desc_;
  message last_message_;
  SharedMemory<Recreated>& sm_;
};  // end of class MessageQueue<Recreated>

class OpenedMessageQueue : public MessageQueue<false> {
 public:
  explicit OpenedMessageQueue(const std::string& id, SharedMemory<false>& sm)
      : MessageQueue{id, sm} {}
};  // end of class OpenedMessageQueue

class RecreatedMessageQueue : public MessageQueue<true> {
 public:
  explicit RecreatedMessageQueue(const std::string& id, SharedMemory<true>& sm)
      : MessageQueue{id, sm} {}
};  // end of class RecreatedMessageQueue

#endif  // KLEIDICV_OPENCV_CONFORMITY_COMMON_H_

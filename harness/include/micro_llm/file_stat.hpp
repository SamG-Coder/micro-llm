#pragma once

// Host file probe. MSVC struct stat is 32-bit and rejects a 15.3 GiB GGUF
// (_stat32 st_size overflow). Use _stat64.

#include <string>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <sys/stat.h>
#ifndef S_ISREG
#define S_ISREG(m) (((m) & _S_IFMT) == _S_IFREG)
#endif
#else
#include <sys/stat.h>
#endif

namespace micro_llm {

inline bool host_file_exists(const std::string& path) {
#ifdef _WIN32
#if defined(_MSC_VER)
    struct _stat64 st {};
    return _stat64(path.c_str(), &st) == 0 && ((st.st_mode & _S_IFREG) != 0);
#else
    struct __stat64 st {};
    return _stat64(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
#endif
#else
    struct stat st {};
    return stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
#endif
}

}  // namespace micro_llm

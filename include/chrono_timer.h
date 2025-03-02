#ifndef CHRONO_TIMER_H
#define CHRONO_TIMER_H

#include <chrono>
#include <iostream>

inline std::chrono::time_point<std::chrono::high_resolution_clock> begin() {
  return std::chrono::high_resolution_clock::now();
}

inline void
end(const std::chrono::time_point<std::chrono::high_resolution_clock> &start) {
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
  std::cout << duration.count() << std::endl;
}

#endif // CHRONO_TIMER_H

#include <iostream>
#include <fstream>
#include <string>
#include <omp.h>
#include <unistd.h>
#include <thread>

// helper to read cache info from Linux sysfs
long read_cache_size(int level) {
    std::string path = "/sys/devices/system/cpu/cpu0/cache/index" + std::to_string(level) + "/size";
    std::ifstream file(path);
    if (!file.is_open()) return -1;
    std::string value;
    file >> value;
    long size = std::stol(value);
    if (value.find('K') != std::string::npos) size *= 1024;
    if (value.find('M') != std::string::npos) size *= 1024 * 1024;
    return size;
}

int main() {
  // CPU and threading info
  int omp_procs = omp_get_num_procs();
  int omp_max_threads = omp_get_max_threads();
  unsigned int hw_threads = std::thread::hardware_concurrency();

  std::cout << "=== System Info (OpenMP + Hardware) ===\n";
  std::cout << "Logical processors available (OpenMP): " << omp_procs << "\n";
  std::cout << "Max OpenMP threads: " << omp_max_threads << "\n";
  std::cout << "Hardware concurrency (std::thread): " << hw_threads << "\n";

  // hyperthreading available if hardware threads > physical cores
  if (hw_threads > omp_procs / 2)
    std::cout << "Hyperthreading likely enabled.\n";
  else
    std::cout << "No hyperthreading detected (or not applicable).\n";

  // cache info
  for (int i = 0; i < 3; ++i) {
    long size = read_cache_size(i);
    if (size > 0)
      std::cout << "L" << (i + 1) << " cache size: " << size / 1024 << " KB\n";
  }

  // RAM info
  long pages = sysconf(_SC_PHYS_PAGES);
  long page_size = sysconf(_SC_PAGE_SIZE);
  double total_ram_gb = (double)pages * page_size / (1024.0 * 1024.0 * 1024.0);
  std::cout << "Total physical memory: " << total_ram_gb << " GB\n";

  // OpenMP runtime confirmation
  #pragma omp parallel
  {
    #pragma omp single
    std::cout << "Actual threads used by default by OpenMP: " << omp_get_num_threads() << "\n";
  }

  return 0;
}
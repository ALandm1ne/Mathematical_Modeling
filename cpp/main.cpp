#include <algorithm>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

#include "grid.hpp"
#include "utils.hpp"

using namespace std;

int main() {
	std::cout << "------------------------------------------" << std::endl;
	std::cout << "[INFO] main:" << std::endl;
	std::cout << "------------------------------------------" << std::endl;

	// 从配置文件中加载参数
	file::Config config = file::load_config("config.yml");

	// 从网格文件中加载网格数据
	GRID grid;
	grid.load_grid(config.grid_file_path, ",");
	grid.check_grid(config.grid_length, config.grid_height);
	grid.output_grid();
	grid.save_grid("test_grid_output.txt", ",");

	return 0;
}

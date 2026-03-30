#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <string>
#include <vector>

#include "grid.cpp"
#include "utils.hpp"

using namespace std;

int main() {
	std::cout << std::endl
	          << std::endl;
	std::cout << "------------------------------------------" << std::endl;
	std::cout << "[INFO] main:" << std::endl;
	std::cout << "------------------------------------------" << std::endl;



	// 从配置文件中加载参数
	file::Config config = file::load_config("config.yml");

	// 从网格文件中加载网格数据
	GRID grid;
	grid.load_grid(config.grid_file_path, ",");
	grid.check_grid(config.grid_length, config.grid_height);
	// grid.output_grid();
	// grid.save_grid("test_grid_output.txt", ",");

	control::wait_for_enter();

	int total_steps = static_cast<int>(config.total_time / config.time_step);
	std::cout << "------------------------------------------" << std::endl;
	std::cout << "[INFO] main: Starting simulation for " << total_steps << " steps (" << config.total_time << " hours)" << std::endl;
	std::cout << "------------------------------------------" << std::endl;

	int temp_n = 0;
	for (size_t step = 0; step < total_steps; step++) {
		std::cout << "------------------------------------------" << std::endl;
		std::cout << "[INFO] main: Simulation step " << step + 1 << "/" << total_steps << std::endl;
		std::cout << "------------------------------------------" << std::endl;

		grid.update_grid([&](double src_value, double distance) {
			return math::gaussian_pdf(std::abs(distance), 0.0, config.grid_step);
		});

		temp_n++;
		if (temp_n % 10 == 0) {
			std::cout << "------------------------------------------" << std::endl;
			std::cout << "[INFO] main: Outputting grid at step " << step + 1 << std::endl;
			std::cout << "------------------------------------------" << std::endl;
			// grid.output_grid();
			grid.save_grid("test_grid_output_step_" + std::to_string(step + 1) + ".txt", ",");
		}
	}



	return 0;
}

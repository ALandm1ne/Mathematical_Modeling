#include "grid.hpp"

void GRID::load_grid(const std::string & filename, const std::string & separator) {
	MAP_DOUBLE grid;
	std::ifstream infile(filename);

	try {
		if (!infile.is_open()) {
			throw std::runtime_error("Cannot open file: " + filename);
		}

		std::string line;
		while (std::getline(infile, line)) {
			std::vector<double> row;
			size_t start = 0, end = 0;
			while ((end = line.find(separator, start)) != std::string::npos) {
				std::string token = line.substr(start, end - start);
				if (!token.empty()) {
					row.push_back(std::stod(token));
				}
				start = end + separator.length();
			}
			std::string token = line.substr(start);
			if (!token.empty()) {
				row.push_back(std::stod(token));
			}
			if (!row.empty()) {
				grid.push_back(row);
			}
		}
		grid_ = grid;
	} catch (const std::ifstream::failure & e) {
		std::cerr << "[ERROR] File operation error: " << e.what() << std::endl;
		exit(EXIT_FAILURE);
	} catch (const std::invalid_argument & e) {
		std::cerr << "[ERROR] Invalid data format: " << e.what() << std::endl;
		exit(EXIT_FAILURE);
	} catch (const std::exception & e) {
		std::cerr << "[ERROR] Error loading grid: " << e.what() << std::endl;
		exit(EXIT_FAILURE);
	}
	infile.close();

	grid_height_ = grid_.size();
	grid_length_ = grid_.empty() ? 0 : grid_[0].size();

	std::cout << "------------------------------------------" << std::endl;
	std::cout << "[INFO] GRID::load_grid: loaded from: '" << filename << "'" << std::endl;
	std::cout << "------------------------------------------" << std::endl;
}



void GRID::check_grid(const int should_length, const int should_width) {
	if (grid_length_ != should_length) {
		std::cerr << "[WARNING] GRID::check_grid: Grid length mismatch: expected " << should_length << ", got " << grid_length_ << std::endl;
		return;
	}
	for (const auto & row : grid_) {
		if (grid_height_ != should_width) {
			std::cerr << "[WARNING] GRID::check_grid: Grid width mismatch: expected " << should_width << ", got " << grid_height_ << std::endl;
			return;
		}
	}
	std::cout << "------------------------------------------" << std::endl;
	std::cout << "[INFO] GRID::check_grid: check passed: length = " << should_length << ", width = " << should_width << std::endl;
	std::cout << "------------------------------------------" << std::endl;
}



void GRID::output_grid(std::string separator) {
	std::cout << "------------------------------------------" << std::endl;
	std::cout << "[INFO] GRID::output_grid: Grid data:" << std::endl;
	std::cout << std::fixed << std::setprecision(10);
	for (const auto & row : grid_) {
		for (const auto & value : row) {
			std::cout << value << separator;
		}
		std::cout << std::endl;
	}
	std::cout << "------------------------------------------" << std::endl;
}



void GRID::save_grid(const std::string & filename, const std::string & separator) {
	std::ofstream outfile(filename);
	outfile << std::fixed << std::setprecision(10);

	try {
		for (const auto & row : grid_) {
			for (const auto & value : row) {
				outfile << value << separator;
			}
			outfile << std::endl;
		}
	} catch (const std::ofstream::failure & e) {
		std::cerr << "[ERROR] GRID::save_grid: Error writing to file: " << e.what() << std::endl;
		exit(EXIT_FAILURE);
	}
	std::cout << "------------------------------------------" << std::endl;
	std::cout << "[INFO] GRID::save_grid: Grid data saved to: '" << filename << "'" << std::endl;
	std::cout << "------------------------------------------" << std::endl;
	outfile.close();
}



template <typename Func>
void GRID::spread_grid_from_point(const MAP_DOUBLE & src_grid, MAP_DOUBLE & dst_grid, const int src_i, const int src_j, const double influence_radius, Func & influence_func) {
	// clang-format off
		int start_i = std::max(0                , src_i - math::double_to_int(influence_radius));
		int end_i   = std::min(grid_height_ - 1 , src_i + math::double_to_int(influence_radius));
		int start_j = std::max(0                , src_j - math::double_to_int(influence_radius));
		int end_j   = std::min(grid_length_ - 1 , src_j + math::double_to_int(influence_radius));
	// clang-format on

	double influence_radius_pow_2 = std::pow(influence_radius, 2);
	for (int dst_i = start_i; dst_i <= end_i; dst_i++) {
		for (int dst_j = start_j; dst_j <= end_j; dst_j++) {

			double distance_pow_2 = std::pow(dst_i - src_i, 2) + std::pow(dst_j - src_j, 2);
			if (distance_pow_2 <= influence_radius_pow_2) {
				double influence = influence_func(src_grid[src_i][src_j], std::sqrt(distance_pow_2));
				dst_grid[dst_i][dst_j] += influence;
			}
		}
	}
}



template <typename Func>
void GRID::update_grid(Func influence_func) {
	MAP_DOUBLE new_grid = grid_;

	for (size_t from_i = 0; from_i < grid_.size(); from_i++) {
		for (size_t from_j = 0; from_j < grid_[from_i].size(); from_j++) {
			spread_grid_from_point(grid_, new_grid, from_i, from_j, influence_radius_, influence_func);
		}
	}
	grid_ = new_grid;
}
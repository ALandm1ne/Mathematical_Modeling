#pragma once

#include "utils.hpp"

// 网格类
class GRID {

  public:
	MAP_DOUBLE grid_;
	int grid_length_;
	int grid_height_;

	/*
	* @brief 从指定的文本文件中加载网格数据，文件格式为每行一个网格行，每行包含多个以指定分隔符分隔的数值
	* @param filename 网格数据文件的路径
	* @param separator 数值之间的分隔符，默认为一个空格
	* @to grid_
	*/
	void load_grid(const std::string & filename, const std::string & separator = " ") {
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
		} catch (const std::invalid_argument & e) {
			std::cerr << "[ERROR] Invalid data format: " << e.what() << std::endl;
		} catch (const std::exception & e) {
			std::cerr << "[ERROR] Error loading grid: " << e.what() << std::endl;
		}
		infile.close();

		grid_height_ = grid_.size();
		grid_length_ = grid_.empty() ? 0 : grid_[0].size();
	}

	/*
	* @brief 检查网格的长度和宽度是否符合预期的值
	* @param should_length 预期的网格长度
	* @param should_width 预期的网格宽度
	*/
	void check_grid(const int should_length, const int should_width) {
		if (grid_length_ != should_length) {
			std::cerr << "[WARNING] Grid length mismatch: expected " << should_length << ", got " << grid_length_ << std::endl;
			return;
		}
		for (const auto & row : grid_) {
			if (grid_height_ != should_width) {
				std::cerr << "[WARNING] Grid width mismatch: expected " << should_width << ", got " << grid_height_ << std::endl;
				return;
			}
		}
		std::cout << "------------------------------------------" << std::endl;
		std::cout << "[INFO] Grid check passed: length = " << should_length << ", width = " << should_width << std::endl;
		std::cout << "------------------------------------------" << std::endl;
	}

	/*
	* @brief 输出网格数据到控制台，格式为每行一个网格行，每行包含多个以指定分隔符分隔的数值
	* @param separator 数值之间的分隔符，默认为一个空格
	*/
	void output_grid(std::string separator = " ") {
		std::cout << "------------------------------------------" << std::endl;
		std::cout << "[INFO] Grid data:" << std::endl;
		for (auto i : grid_) {
			for (auto j : i) {
				std::cout << j << separator;
			}
			std::cout << std::endl;
		}
		std::cout << "------------------------------------------" << std::endl;
	}

	/*
	* @brief 将网格数据保存到指定的文本文件中，文件格式为每行一个网格行，每行包含多个以指定分隔符分隔的数值
	* @param filename 网格数据文件的路径
	* @param separator 数值之间的分隔符，默认为一个空格
	*/
	void save_grid(const std::string & filename, const std::string & separator = " ") {
		std::ofstream outfile(filename);

		try {
			for (const auto & row : grid_) {
				for (double value : row) {
					outfile << value << separator;
				}
				outfile << std::endl;
			}
		} catch (const std::ofstream::failure & e) {
			std::cerr << "[ERROR] Error writing to file: " << e.what() << std::endl;
		}
		outfile.close();
	}
};
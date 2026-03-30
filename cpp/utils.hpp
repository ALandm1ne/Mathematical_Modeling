#pragma once

#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>

using MAP_DOUBLE = std::vector<std::vector<double>>;

namespace file {

	// 配置结构体，包含时间步长、网格步长和目标速度
	struct Config {
		std::string grid_file_path; // 网格数据文件路径
		int grid_length;            // 网格长度，单位：模拟点
		int grid_height;            // 网格高度，单位：模拟点
		double time_step;           // 时间步长，单位：小时
		double grid_step;           // 网格步长，单位：公里
		double total_time;          // 总的模拟时长，单位：小时
		double velocity_target;     // 目标速度，单位：公里/小时
	};

	// 从指定的YAML文件中加载配置
	inline Config load_config(const std::string & filename) {
		Config config;
		YAML::Node yaml = YAML::LoadFile(filename);
		config.grid_file_path = yaml["Grid_File_Path"].as<std::string>();
		config.time_step = yaml["Time_Step"].as<double>();
		config.grid_step = yaml["Grid_Step"].as<double>();
		config.total_time = yaml["Total_time"].as<double>();
		config.velocity_target = yaml["Velocity_Target"].as<double>();
		config.grid_length = yaml["Grid_Length"].as<int>();
		config.grid_height = yaml["Grid_Height"].as<int>();

		// 输出加载的配置参数
		std::cout << "------------------------------------------" << std::endl;
		std::cout << "[INFO] 参数已加载:   " << std::endl;
		std::cout << "Grid File Path:   " << config.grid_file_path << std::endl;
		std::cout << "Time Step:        " << config.time_step << " hours" << std::endl;
		std::cout << "Grid Step:        " << config.grid_step << " km" << std::endl;
		std::cout << "Velocity Target:  " << config.velocity_target << " km/h" << std::endl;
		std::cout << "Grid Length:      " << config.grid_length << " simulation points" << std::endl;
		std::cout << "Grid Height:      " << config.grid_height << " simulation points" << std::endl;
		std::cout << "------------------------------------------" << std::endl;

		return config;
	}



} // namespace file

namespace math {

	constexpr double kPi = 3.14159265358979323846;
}
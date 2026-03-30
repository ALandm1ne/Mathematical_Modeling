#pragma once

#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include <yaml-cpp/yaml.h>

using MAP_DOUBLE = std::vector<std::vector<double>>;

namespace file {

	/**
	 * @brief  配置结构体
	 * @param grid_file_path 网格数据文件路径
	 * @param grid_length 网格长度，单位：模拟点
	 * @param grid_height 网格高度，单位：模拟点
	 * @param time_step 时间步长，单位：小时
	 * @param grid_step 网格步长，单位：公里
	 * @param total_time 总的模拟时长，单位：小时
	 * @param velocity_target 目标速度，单位：公里/小时
	 */
	struct Config {
		std::string grid_file_path; // 网格数据文件路径
		int grid_length;            // 网格长度，单位：模拟点
		int grid_height;            // 网格高度，单位：模拟点
		double time_step;           // 时间步长，单位：小时
		double grid_step;           // 网格步长，单位：公里
		double total_time;          // 总的模拟时长，单位：小时
		double velocity_target;     // 目标速度，单位：公里/小时
	};

	/**
	 * @brief  从指定的YAML文件中加载配置
	 * @param  filename YAML文件路径
	 * @return 加载的配置参数
	 */
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
		std::cout << "[INFO] file::load_config: 参数已加载:   " << std::endl;
		std::cout << "Grid File Path:   " << config.grid_file_path << std::endl;
		std::cout << "Time Step:        " << config.time_step << " hours" << std::endl;
		std::cout << "Grid Step:        " << config.grid_step << " km" << std::endl;
		std::cout << "Total Time:       " << config.total_time << " hours" << std::endl;
		std::cout << "Velocity Target:  " << config.velocity_target << " km/h" << std::endl;
		std::cout << "Grid Length:      " << config.grid_length << " simulation points" << std::endl;
		std::cout << "Grid Height:      " << config.grid_height << " simulation points" << std::endl;
		std::cout << "------------------------------------------" << std::endl;

		return config;
	}



} // namespace file

namespace math {

	constexpr double kPi = 3.14159265358979323846;

	/*
	* @brief 四舍五入 double -> int 
	*/
	inline int double_to_int(const double & value) {
		return static_cast<int>(std::round(value));
	}
} // namespace math

namespace control {

	/*
	* @brief 等待用户输入以继续
	*/
	inline void wait_for_enter() {
		std::cout << std::endl;
		std::cout << "*** Press Enter to continue... ***" << std::endl;
		std::cout << std::endl;
		std::cin.get();
	}

	/*
	* @brief 等待n毫秒
	*/
	inline void wait_for(int ms) {
		std::this_thread::sleep_for(std::chrono::milliseconds(ms));
	}

} // namespace control
#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>

namespace config {

	// 配置结构体，包含时间步长、网格步长和目标速度
	struct Config {
		double time_step;       // 时间步长，单位：小时
		double grid_step;       // 网格步长，单位：公里
		double velocity_target; // 目标速度，单位：公里/小时
	};

	// 从指定的YAML文件中加载配置
	inline Config load_config(const std::string & filename) {
		Config config;
		YAML::Node yaml = YAML::LoadFile(filename);
		config.time_step = yaml["Time_Step"].as<double>();
		config.grid_step = yaml["Grid_Step"].as<double>();
		config.velocity_target = yaml["Velocity_Target"].as<double>();
		return config;
	}

} // namespace config
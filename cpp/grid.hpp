#pragma once

#include <iomanip>

#include "utils.hpp"

// 网格类
class GRID {

  public:
	MAP_DOUBLE grid_;
	int grid_length_;
	int grid_height_;
	double influence_radius_;



	/*
	* @brief 从指定的文本文件中加载网格数据，文件格式为每行一个网格行，每行包含多个以指定分隔符分隔的数值
	* @param filename 网格数据文件的路径
	* @param separator 数值之间的分隔符，默认为一个空格
	* @to    grid_
	*/
	void load_grid(const std::string & filename, const std::string & separator = " ");



	/*
	* @brief 检查网格的长度和宽度是否符合预期的值
	* @param should_length 预期的网格长度
	* @param should_width 预期的网格宽度
	*/
	void check_grid(const int should_length, const int should_width);



	/*
	* @brief 输出网格数据到控制台，格式为每行一个网格行，每行包含多个以指定分隔符分隔的数值
	* @param separator 数值之间的分隔符，默认为一个空格
	*/
	void output_grid(std::string separator = " ");



	/*
	* @brief 将网格数据保存到指定的文本文件中，文件格式为每行一个网格行，每行包含多个以指定分隔符分隔的数值
	* @param filename 网格数据文件的路径
	* @param separator 数值之间的分隔符，默认为一个空格
	*/
	void save_grid(const std::string & filename, const std::string & separator = " ");



	/*
	* @brief 从指定点向周围传播网格影响
	* @param src_grid 源网格数据
	* @param dst_grid 目标网格数据
	* @param src_i 指定点的行索引
	* @param src_j 指定点的列索引
	* @param influence_radius 影响半径
	* @param influence_func(src_value, distance) 影响函数--根据源点的值和到目标点的距离计算对目标点的影响
	* @to    dst_grid
	*/
	template <typename Func>
	void spread_grid_from_point(
	    const MAP_DOUBLE & src_grid,
	    MAP_DOUBLE & dst_grid,
	    const int src_i,
	    const int src_j,
	    const double influence_radius,
	    Func & influence_func);



	/*
	* @brief 更新网格数据
	* @param influence_func(src_value, distance) 影响函数--根据源点的值和到目标点的距离计算对目标点的影响
	* @idea  先将grid_复制到new_grid中，然后遍历grid_中的每一个点,计算该点对网格中其它点的影响，并将结果存储在new_grid中，最后将new_grid赋值回grid_。
	*/
	template <typename Func>
	void update_grid(Func influence_func);
};
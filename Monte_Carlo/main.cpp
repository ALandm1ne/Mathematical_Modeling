#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <random>
#include <vector>

// 使用 M_PI 常量
#define _USE_MATH_DEFINES
#include <cmath>

// 参数设置（单位：km 和小时）
constexpr double AREA_WIDTH_KM = 306.0;  // 区域宽度 (km)
constexpr double AREA_HEIGHT_KM = 444.0; // 区域高度 (km)
constexpr int N_PARTICLES = 1000000;     // 初始粒子数
constexpr double TARGET_SPEED = 30.0;    // 目标速度 (km/h)
constexpr double UAV_SPEED = 150.0;      // 无人机速度 (km/h)
constexpr double UAV_SCAN_RADIUS = 15.0; // 扫描半径 (km)
constexpr double DT = 0.01;              // 仿真步长 (h)

// 缩放因子：1 km -> 1000 单位 (1 单位 = 1 m)
constexpr int SCALE = 1000;
// clang-format off
int64_t AREA_WIDTH_U      = static_cast<int64_t>(std::round(AREA_WIDTH_KM   * SCALE));
int64_t AREA_HEIGHT_U     = static_cast<int64_t>(std::round(AREA_HEIGHT_KM  * SCALE));
int64_t UAV_SCAN_RADIUS_U = static_cast<int64_t>(std::round(UAV_SCAN_RADIUS * SCALE));
// clang-format on

// 粒子结构体
struct Particle {
	int64_t x, y;   // 位置（整数单位）
	double v_angle; // 运动方向（弧度）
};

// 更新所有粒子的位置（边界反弹）
void update_particles(std::vector<Particle> & particles, double radius_U) {
	for (auto & p : particles) {
		// 计算位移分量（四舍五入到整数）

		// // 添加随机扰动（如果提供了）
		// p.v_angle += random() % 1000 / 1000.0 * 2.0 * M_PI * random_angle; // 随机扰动范围 [-random_angle, random_angle]

		int64_t dx = static_cast<int64_t>(std::round(radius_U * std::cos(p.v_angle)));
		int64_t dy = static_cast<int64_t>(std::round(radius_U * std::sin(p.v_angle)));
		p.x += dx;
		p.y += dy;

		// X 边界处理
		if (p.x < 0 || p.x > AREA_WIDTH_U) {
			p.v_angle = M_PI - p.v_angle; // 反射
			p.x = std::clamp(p.x, static_cast<int64_t>(0), AREA_WIDTH_U);
		}
		// Y 边界处理
		if (p.y < 0 || p.y > AREA_HEIGHT_U) {
			p.v_angle = -p.v_angle; // 反射
			p.y = std::clamp(p.y, static_cast<int64_t>(0), AREA_HEIGHT_U);
		}
	}
}

int main() {
	// ---------- 1. 初始化粒子 ----------
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<int64_t> distX(0, AREA_WIDTH_U);
	std::uniform_int_distribution<int64_t> distY(0, AREA_HEIGHT_U);
	std::uniform_real_distribution<double> distAngle(0.0, 2.0 * M_PI);

	std::vector<Particle> particles;
	particles.reserve(N_PARTICLES);
	for (int i = 0; i < N_PARTICLES; ++i) {
		particles.push_back({distX(gen),
		                     distY(gen),
		                     distAngle(gen)});
	}

	// ---------- 2. 初始化无人机 ----------
	int64_t uav_x = 0, uav_y = 0;

	// ---------- 3. 模拟循环 ----------
	std::vector<int> history_count; // 记录每一步剩余粒子数
	double time_elapsed = 0.0;

	std::cout << "开始搜索模拟...\n";
	const int MAX_STEPS = 100; // 最大步数
	for (int step = 0; step < MAX_STEPS; ++step) {
		// A. 目标移动
		update_particles(particles, TARGET_SPEED * DT * SCALE);

		// B. 无人机沿 Z 字形路径移动（简单策略）
		uav_y += static_cast<int64_t>(std::round(UAV_SPEED * DT * SCALE));
		if (uav_y > AREA_HEIGHT_U) {
			uav_y = 0;
			uav_x += 2 * UAV_SCAN_RADIUS_U; // 向右移动一个扫描直径
		}

		// C. 捕获检测（删除半径内的粒子）
		int64_t r2 = UAV_SCAN_RADIUS_U * UAV_SCAN_RADIUS_U;
		auto new_end = std::remove_if(particles.begin(), particles.end(),
		                              [&](const Particle & p) {
			                              int64_t dx = p.x - uav_x;
			                              int64_t dy = p.y - uav_y;
			                              return (dx * dx + dy * dy) <= r2;
		                              });
		particles.erase(new_end, particles.end());

		// 记录剩余粒子数
		history_count.push_back(static_cast<int>(particles.size()));
		time_elapsed += DT;

		// 若粒子数为零，提前结束
		if (particles.empty()) {
			std::cout << "目标已全部锁定！耗时: " << time_elapsed << " 小时\n";
			break;
		}
	}

	// ---------- 4. 使用 OpenCV 绘制曲线 ----------
	// 创建空白图像 (500x500, 3通道彩色)
	const int IMG_W = 800, IMG_H = 600;
	cv::Mat img(IMG_H, IMG_W, CV_8UC3, cv::Scalar(255, 255, 255));

	// 确定绘图范围
	int steps = static_cast<int>(history_count.size());
	if (steps == 0) {
		std::cout << "无数据可绘制。" << std::endl;
		return 0;
	}

	// 最大粒子数（初始值）
	int max_count = N_PARTICLES;
	// 缩放因子：将时间步映射到 [margin, IMG_W-margin]，粒子数映射到 [IMG_H-margin, margin]
	const int margin = 50;
	double scale_x = static_cast<double>(IMG_W - 2 * margin) / (steps - 1);
	double scale_y = static_cast<double>(IMG_H - 2 * margin) / max_count; // 注意 y 轴方向反转

	// 绘制坐标轴
	cv::line(img, cv::Point(margin, margin), cv::Point(margin, IMG_H - margin), cv::Scalar(0, 0, 0), 2);
	cv::line(img, cv::Point(margin, IMG_H - margin), cv::Point(IMG_W - margin, IMG_H - margin), cv::Scalar(0, 0, 0), 2);

	// 添加标签文字（简单）
	cv::putText(img, "Time Steps", cv::Point(IMG_W / 2 - 40, IMG_H - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
	cv::putText(img, "Particle Count", cv::Point(10, margin / 2), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
	cv::putText(img, "0", cv::Point(margin - 15, IMG_H - margin), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0));
	cv::putText(img, std::to_string(max_count), cv::Point(margin - 30, margin + 5), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0));
	cv::putText(img, "0", cv::Point(margin, IMG_H - margin + 5), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0));
	cv::putText(img, std::to_string(steps - 1), cv::Point(IMG_W - margin - 10, IMG_H - margin + 5), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0));

	// 绘制曲线点
	std::vector<cv::Point> points;
	for (int i = 0; i < steps; ++i) {
		int x = static_cast<int>(margin + i * scale_x);
		int y = static_cast<int>(IMG_H - margin - history_count[i] * scale_y);
		points.emplace_back(x, y);
	}

	// 连线
	for (size_t i = 1; i < points.size(); ++i) {
		cv::line(img, points[i - 1], points[i], cv::Scalar(0, 0, 255), 2);
	}

	// 显示图像
	cv::imshow("Remaining Potential Target Particles Over Time", img);
	cv::waitKey(0);

	return 0;
}
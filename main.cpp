#include <algorithm>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

#include "utils.hpp"

using namespace std;

int main() {
	// 从配置文件中加载参数
	config::Config config = config::load_config("config.yml");

	// 输出加载的配置参数
	cout << "Time Step: " << config.time_step << " hours" << endl;
	cout << "Grid Step: " << config.grid_step << " km" << endl;
	cout << "Velocity Target: " << config.velocity_target << " km/h" << endl;

	return 0;
}
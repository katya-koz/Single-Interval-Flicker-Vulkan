#pragma once
#include <Windows.h>
#include <string>
#include <algorithm>
#include <random>
#include "app.h"
#include <filesystem>
#include "tobii_research_streams.h"
#include <cmath>
#include <vector>
#include <math.h>
#include <numbers>
#include <limits>  
#include "DisplayInfo.h"
#include <sstream>
#include <iostream>
#include <cctype>
namespace Utils
{

	struct GazeStatistics
	{
		float leftMean_X;
		float leftMean_Y;
		float leftStdDev_X;
		float leftStdDev_Y;

		float rightMean_X;
		float rightMean_Y;
		float rightStdDev_X;
		float rightStdDev_Y;
	};


	static std::string ReadFile(const std::string& path) {
		std::ifstream f(path);
		return std::string(std::istreambuf_iterator<char>(f), {});
	}
	static void FatalError(const std::string& message)
	{
		const auto result = MessageBoxA(
			nullptr,
			message.c_str(),
			"Fatal Error",
			MB_OK | MB_ICONERROR | MB_TOPMOST
		);

		if (result == IDOK)
		{
			exit(1);
		}
	}
	static bool containsControl(std::string name) {
		// returns true if the given input contains 'control' in any case
		std::transform(name.begin(), name.end(), name.begin(),
			[](unsigned char c) {
				return std::tolower(c);
			});

		return name.find("control") != std::string::npos;
	}

	static GazeStatistics calculateGazeStatistics(const std::vector<TobiiResearchGazeData>& gazeData, DisplayInfo displayInfo)
	{
		const float nan = (std::numeric_limits<float>::quiet_NaN)();
		GazeStatistics stats{};
		stats.leftMean_X = stats.leftMean_Y = stats.leftStdDev_X = stats.leftStdDev_Y = nan;
		stats.rightMean_X = stats.rightMean_Y = stats.rightStdDev_X = stats.rightStdDev_Y = nan;

		//const float originX = (static_cast<float>(monitorWidth) - imageWidth) / 2.0f;
		//const float originY = (static_cast<float>(monitorHeight) - imageHeight) / 2.0f;

		// Only valid, finite samples are pushed into these.
		std::vector<double> leftX, leftY, rightX, rightY;
		leftX.reserve(gazeData.size());  leftY.reserve(gazeData.size());
		rightX.reserve(gazeData.size()); rightY.reserve(gazeData.size());

		auto toImage = [&](float normX, float normY, double& outX, double& outY) { outX = normX * displayInfo.pixelWidth;   outY = normY * displayInfo.pixelHeight;};

		for (const auto& g : gazeData)
		{
			const auto& L = g.left_eye.gaze_point;
			const auto& R = g.right_eye.gaze_point;

			if (L.validity == TOBII_RESEARCH_VALIDITY_VALID &&
				std::isfinite(L.position_on_display_area.x) &&
				std::isfinite(L.position_on_display_area.y))
			{
				double x, y;
				toImage(L.position_on_display_area.x, L.position_on_display_area.y, x, y);
				leftX.push_back(x); leftY.push_back(y);
			}

			if (R.validity == TOBII_RESEARCH_VALIDITY_VALID &&
				std::isfinite(R.position_on_display_area.x) &&
				std::isfinite(R.position_on_display_area.y))
			{
				double x, y;
				toImage(R.position_on_display_area.x, R.position_on_display_area.y, x, y);
				rightX.push_back(x); rightY.push_back(y);
			}
		}

		// mean + sample std dev (n-1). Mean needs n>=1, std dev needs n>=2.
		auto meanStd = [](const std::vector<double>& v, float& mean, float& sd) {
			if (v.empty()) return; // leaves NaN
			double sum = 0.0;
			for (double d : v) sum += d;
			const double m = sum / v.size();
			mean = static_cast<float>(m);
			if (v.size() < 2) return;  // sd stays NaN
			double sq = 0.0;
			for (double d : v) sq += (d - m) * (d - m);
			sd = static_cast<float>(std::sqrt(sq / (v.size() - 1)));
			};

		meanStd(leftX, stats.leftMean_X, stats.leftStdDev_X);
		meanStd(leftY, stats.leftMean_Y, stats.leftStdDev_Y);
		meanStd(rightX, stats.rightMean_X, stats.rightStdDev_X);
		meanStd(rightY, stats.rightMean_Y, stats.rightStdDev_Y);

		return stats;
	}

	static std::filesystem::path getExecutableDirectory()
	{
		char buffer[MAX_PATH];

		DWORD length = GetModuleFileNameA(
			nullptr,
			buffer,
			MAX_PATH
		);

		if (length == 0) {
			throw std::runtime_error("Failed to get executable path.");
		}

		return std::filesystem::path(buffer).parent_path();
	}

	/// <summary>
/// Converts a visual angle in degrees to a pixel radius.
/// </summary>
	static double degreesToPixel(double degrees, const DisplayInfo& displayInfo)
	{
		// Viewing distance in mm.
		const double viewingDistanceMm = static_cast<double>(displayInfo.viewingDistanceCm) * 10.0;

		// Physical radius corresponding to half the visual angle.
		const double radiusMm = viewingDistanceMm * std::tan(degrees * 0.5 * std::numbers::pi / 180.0);

		// Pixels per mm, using the horizontal scale.
		const double pixelsPerMmX = static_cast<double>(displayInfo.pixelWidth) / static_cast<double>(displayInfo.physicalWidthMm);

		return radiusMm * pixelsPerMmX;
	}

	/// <summary>
	/// Calculates the pixel radius of an error boundary around a fixation point.
	/// Angular error is interpolated linearly from the screen center to its corners.
	/// </summary>
	static double degreesToPixelError(double degreesCenter, double degreesEdge,
		const FixationCoordinates& fixationCoords,
		const Coords& centerCoords,
		const DisplayInfo& displayInfo)
	{
		//std::cout << displayInfo.toString() << std::endl;

		// Fixation point's distance from screen center.
		const double dx = static_cast<double>(fixationCoords.Left.X - centerCoords.X);
		const double dy = static_cast<double>(fixationCoords.Left.Y - centerCoords.Y);
		const double distanceFromCenter = std::hypot(dx, dy);

		// Maximum distance from center to a screen corner.
		const double halfWidth = static_cast<double>(displayInfo.pixelWidth) / 2.0;
		const double halfHeight = static_cast<double>(displayInfo.pixelHeight) / 2.0;
		const double maxDistanceFromCenter = std::hypot(halfWidth, halfHeight);

		// Normalize and clamp to [0, 1].
		const double t = maxDistanceFromCenter > 0.0 ? std::clamp(distanceFromCenter / maxDistanceFromCenter, 0.0, 1.0) : 0.0;

		// Linearly interpolate angular error from center to corner.
		const double degrees = degreesCenter + t * (degreesEdge - degreesCenter);

		return degreesToPixel(degrees, displayInfo);
	}

   
	///// <summary>
	///// Returns if the gaze is within the error boundary (true) or not (false)
	///// uses euclidean geometry
	///// </summary>
	///// <param name="leftX"></param>
	///// <param name="leftY"></param>
	///// <param name="rightX"></param>
	///// <param name="rightY"></param>
	///// <param name="pixelError"></param>
	//static bool isGazeValid(
	//	float leftX, float leftY, bool leftValid,
	//	float rightX, float rightY, bool rightValid,
	//	float pixelRadius,
	//	float leftFixX, float leftFixY,
	//	float rightFixX, float rightFixY)
	//{

	//	// mean per-eye offset from its own fixation == gaze midpoint - fixation midpoint
	//	float sumX = 0.0f, sumY = 0.0f;
	//	int count = 0;
	//	if (leftValid) { sumX += leftX - leftFixX;  sumY += leftY - leftFixY;  ++count; }
	//	if (rightValid) { sumX += rightX - rightFixX; sumY += rightY - rightFixY; ++count; }
	//	if (count == 0) return false;

	//	const float mx = sumX / count, my = sumY / count;
	//	return (mx * mx + my * my) <= pixelRadius * pixelRadius;
	//}

}
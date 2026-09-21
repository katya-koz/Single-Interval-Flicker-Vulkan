#pragma once
#include <Windows.h>
#include <string>
#include <algorithm>
#include <random>
#include "app.h"
#include <filesystem>
#include "tobii_research_streams.h"

#include <vector>
#include <cmath>


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

	static GazeStatistics calculateGazeStatistics(
		const std::vector<TobiiResearchGazeData>& gazeData,
		const int monitorWidth, int monitorHeight,   // physical/virtual display — undoes normalization
		const int imageWidth, int imageHeight,      // real stimulus resolution — undoes crop + mirror
		float leftFixationX, float leftFixationY,
		float rightFixationX, float rightFixationY)
	{
		GazeStatistics stats{};
		if (gazeData.empty()) return stats;

		// same centering math as renderFixationPoint's imageOriginX/Y
		const float originX = (static_cast<float>(monitorWidth) - imageWidth) / 2.0f;
		const float originY = (static_cast<float>(monitorHeight) - imageHeight) / 2.0f;

		std::vector<float> leftX, leftY, rightX, rightY;
		leftX.reserve(gazeData.size()); leftY.reserve(gazeData.size());
		rightX.reserve(gazeData.size()); rightY.reserve(gazeData.size());

		for (const auto& gaze : gazeData)
		{
			// normalized [0,1] -> monitor pixel space (matches how the raw gaze/mouse was normalized)
			float leftMonX = gaze.left_eye.gaze_point.position_on_display_area.x * monitorWidth;
			float leftMonY = gaze.left_eye.gaze_point.position_on_display_area.y * monitorHeight;
			float rightMonX = gaze.right_eye.gaze_point.position_on_display_area.x * monitorWidth;
			float rightMonY = gaze.right_eye.gaze_point.position_on_display_area.y * monitorHeight;

			// monitor pixel space -> image pixel space (undo the centering offset)
			// then undo the render-time left-right mirror -> back to original stimulus coords
			float leftGazeX = imageWidth - 1 - (leftMonX - originX);
			float leftGazeY = imageHeight - 1 - (leftMonY - originY);
			float rightGazeX = imageWidth - 1 - (rightMonX - originX);
			float rightGazeY = imageHeight - 1 - (rightMonY - originY);

			leftX.push_back(leftGazeX);  leftY.push_back(leftGazeY);
			rightX.push_back(rightGazeX); rightY.push_back(rightGazeY);
		}

        // Calculate means
        float leftSumX = 0.0f;
        float leftSumY = 0.0f;
        float rightSumX = 0.0f;
        float rightSumY = 0.0f;

        for (size_t i = 0; i < gazeData.size(); ++i)
        {
            leftSumX += leftX[i];
            leftSumY += leftY[i];

            rightSumX += rightX[i];
            rightSumY += rightY[i];
        }

        stats.leftMean_X = leftSumX / leftX.size();
        stats.leftMean_Y = leftSumY / leftY.size();

        stats.rightMean_X = rightSumX / rightX.size();
        stats.rightMean_Y = rightSumY / rightY.size();


        // Calculate standard deviations
        if (gazeData.size() > 1)
        {
            float leftSquaredDifferenceX = 0.0f;
            float leftSquaredDifferenceY = 0.0f;

            float rightSquaredDifferenceX = 0.0f;
            float rightSquaredDifferenceY = 0.0f;

            for (size_t i = 0; i < gazeData.size(); ++i)
            {
                float leftDifferenceX = leftX[i] - stats.leftMean_X;
                float leftDifferenceY = leftY[i] - stats.leftMean_Y;
                float rightDifferenceX = rightX[i] - stats.rightMean_X;
                float rightDifferenceY = rightY[i] - stats.rightMean_Y;
                leftSquaredDifferenceX += leftDifferenceX * leftDifferenceX;
                leftSquaredDifferenceY += leftDifferenceY * leftDifferenceY;
                rightSquaredDifferenceX += rightDifferenceX * rightDifferenceX;
                rightSquaredDifferenceY += rightDifferenceY * rightDifferenceY;
            }
			stats.leftStdDev_X = std::sqrt(leftSquaredDifferenceX / (leftX.size() - 1));
			stats.leftStdDev_Y = std::sqrt(leftSquaredDifferenceY / (leftY.size() - 1));
			stats.rightStdDev_X = std::sqrt(rightSquaredDifferenceX / (rightX.size() - 1));
			stats.rightStdDev_Y = std::sqrt(rightSquaredDifferenceY / (rightY.size() - 1));
		}

        return stats;
    }

	// commented out - trial and flicker order determined by trials.csv

	//static void ShuffleTrials(std::vector<ImagePaths>& trials)
	//{
	//	// shuffle the order of the flickers
	//	std::random_device rd;
	//	std::mt19937 gen(rd());

	//	std::shuffle(trials.begin(), trials.end(), gen);
	//}

	//static void ShuffleFlickers(std::vector<ImagePaths>& trials)
	//{
	//	// iterate through the trials, randomize the flicker to be shown either first or second
	//	std::random_device rd;
	//	static std::mt19937 gen(rd());
	//	static std::uniform_int_distribution<int> dist(0, 1);

	//	for (auto& n : trials) {
	//		n.flickerIndex = dist(gen);
	//	}
	//	return;
	//}


	/// <summary>
	/// Helper to get current directory of executeable. to construct path for assets folder.
	/// </summary>
	/// <returns></returns>
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

	// calculate the radius of the foveal view based off screen size, viewing distance, and given foveal width (degrees)
	static float degreesToRadiusPx(float degrees, float viewingDistanceMeters, float screenWidthMeters, float screenWidthPixels)
	{
		float radians = degrees * (3.14159265f / 180.0f);

		float radiusMeters = viewingDistanceMeters * tan(radians * 0.5f);

		float radiusPixels = (radiusMeters / screenWidthMeters) * screenWidthPixels;

		return radiusPixels;
	}

	static float fovealRadiusFromPixelsPerDegree(float pixPerDeg, float fovalWidthDeg) {
		return pixPerDeg * fovalWidthDeg; 
	}
	// randomize a local quad location and size for local flicker
	static std::tuple<float, float, float, float> randomizeQuad(int screenWidth, int screenHeight)
	{
		static std::mt19937 rng(std::random_device{}());

		float minSize = 100.0f;
		float maxSize = 400.0f;

		std::uniform_real_distribution<float> sizeDist(minSize, maxSize);
		float w = sizeDist(rng);
		float h = sizeDist(rng);

		std::uniform_real_distribution<float> xDist(0.0f, screenWidth - w);
		std::uniform_real_distribution<float> yDist(0.0f, screenHeight - h);
		float x = xDist(rng);
		float y = yDist(rng);

		return { x, y, w, h };
	}

   

}
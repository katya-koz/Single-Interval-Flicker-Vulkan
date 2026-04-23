#pragma once
#include <Windows.h>
#include <string>
#include <algorithm>
#include <random>
#include "app.h"

namespace Utils
{
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

	static void ShuffleTrials(std::vector<ImagePaths>& trials)
	{
		// shuffle the order of the flickers
		std::random_device rd;
		std::mt19937 gen(rd());

		std::shuffle(trials.begin(), trials.end(), gen);
	}

	static void ShuffleFlickers(std::vector<ImagePaths>& trials)
	{
		// iterate through the trials, randomize the flicker to be shown either first or second
		std::random_device rd;
		static std::mt19937 gen(rd());
		static std::uniform_int_distribution<int> dist(0, 1);

		for (auto& n : trials) {
			n.flickerIndex = dist(gen);
		}
		return;
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
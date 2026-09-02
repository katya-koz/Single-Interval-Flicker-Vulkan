#include "App.h"
#include <Windows.h>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <string>
#include <shlobj.h>
#include <sstream>


int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::cerr << "Usage: App.exe <config.json> <input.csv>\n";
        return -1;
    }

    std::string configPath = argv[1];
    std::string inputPath = argv[2];

    App app;

    if (!app.init(configPath, inputPath))
    {
        std::cerr << "Failed to initialize app with config: " << configPath << "\n";
        return -1;
    }

    app.run();
    return 0;
}
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
    if (argc < 2)
    {
        std::cerr << "Usage: App.exe <input.csv> [--config <config.json>]\n";
        return -1;
    }

    std::string inputPath = argv[1];
    std::string configPath = "config.json";

    for (int i = 2; i < argc; ++i)
    {
        std::string arg = argv[i];

        if (arg == "--config")
        {
            if (i + 1 >= argc)
            {
                std::cerr << "Error: --config requires a file path\n";
                return -1;
            }

            configPath = argv[++i];
        }
        else
        {
            std::cerr << "Error: Unknown argument: " << arg << "\n";
            return -1;
        }
    }

    App app;

    if (!app.init(configPath, inputPath))
    {
        std::cerr << "Failed to initialize app with config: "
            << configPath << "\n";
        return -1;
    }

    app.run();

    return 0;
}
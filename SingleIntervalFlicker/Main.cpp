#include "App.h"
#include <Windows.h>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <string>
#include <shlobj.h>
#include <sstream>
#include "selectVariantDialog.h"


int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: App.exe <config.json>\n";
        return -1;
    }

    std::string configPath = argv[1];

    App app;

    if (!app.init(configPath))
    {
        std::cerr << "Failed to initialize app with config: " << configPath << "\n";
        return -1;
    }

    app.run();
    return 0;
}
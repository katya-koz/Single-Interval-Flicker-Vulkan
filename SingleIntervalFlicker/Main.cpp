#include "App.h"
#include <Windows.h>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <string>
#include <shlobj.h>
#include <sstream>
#include "selectVariantDialog.h"


int selectedVariant = -1;



int WINAPI WinMain(
    HINSTANCE hInstance,
    HINSTANCE hPrevInstance,
    LPSTR lpCmdLine,
    int nCmdShow
) {

    
    App app;



    if (!app.init("config.json")) { // path to the config here !!
        return -1;
    }



    app.run();

    return 0;
}
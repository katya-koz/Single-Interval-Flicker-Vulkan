#pragma once


struct Coords {
    int X = 0;
    int Y = 0;
};

struct FixationCoordinates {
    Coords Left = Coords();
    Coords Right = Coords();
};

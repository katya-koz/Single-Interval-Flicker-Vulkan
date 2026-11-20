#pragma once


struct DisplayInfo
{
    int pixelWidth;
    int pixelHeight;

    int physicalWidthMm;
    int physicalHeightMm;

    double viewingDistanceCm;
    std::string toString() const
    {
        std::ostringstream out;

        out << "========== DisplayInfo ==========\n";
        out << "pixelWidth: " << pixelWidth << " px\n";
        out << "pixelHeight: " << pixelHeight << " px\n";
        out << "physicalWidthMm: " << physicalWidthMm << " mm\n";
        out << "physicalHeightMm: " << physicalHeightMm << " mm\n";
        out << "viewingDistanceCm: " << viewingDistanceCm << " cm\n";
        out << "=================================";

        return out.str();
    }
};
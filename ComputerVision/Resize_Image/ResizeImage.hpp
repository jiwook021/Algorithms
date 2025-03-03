/**
 * @file ResizeImage.hpp
 * @brief Seam carving image resizer
 * @details Content-aware image resizing using energy-based seam carving.
 *          Includes Matrix, Image (PPM format), and processing modules.
 */

#pragma once

#include "Matrix.h"
#include "Image.h"
#include "processing.h"

namespace ResizeImage {

/**
 * @brief Resize a PPM image using seam carving.
 * @param inputPath   Path to input PPM file.
 * @param outputPath  Path to write output PPM file.
 * @param newWidth    Target width (<= original).
 * @param newHeight   Target height (<= original).
 * @return 0 on success, non-zero on failure.
 */
int Run(const std::string& inputPath,
        const std::string& outputPath,
        int newWidth, int newHeight);

}  // namespace ResizeImage

#pragma once

#include "model_library.hpp"

#include <string>
#include <vector>

// Fetches Pokemon-3D-api GLBs (GitHub raw) and prepares them for Assimp.
//
// Raw assets use Draco + WebP + skins; Assimp cannot load them as-is. Each
// download is written to a RAM scratch dir, run through tools/poke3d_prep, and
// deleted when this object goes out of scope — same lifetime as ModelDownloader.
class Poke3dDownloader {
public:
    ~Poke3dDownloader();

    // Resolve NAME via PokeAPI (national dex), then downloadByDex. Form tokens
    // in the name (Heat Rotom, Alolan Raichu) select alternate asset folders.
    bool download(
        const std::string& pokemon_name,
        std::vector<ModelEntry>& out_models,
        std::string& error);

    // Download the best poke-3D asset for this dex. Optional card_name selects
    // regional / multiform variants when the assets repo has them.
    bool downloadByDex(
        int dex,
        std::vector<ModelEntry>& out_models,
        std::string& error,
        const std::string& card_name = "");

private:
    std::vector<std::string> temp_dirs_;
};

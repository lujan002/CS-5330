#pragma once

#include <string>

// Offline Pokédex heights (decimetres) keyed by national dex / species slug.
// Loaded from data/pokemon_heights.csv (PokeAPI / veekun dump).

struct PokemonHeights {
    // Load CSV once. Safe to call repeatedly; returns false only if no file found
    // and the table is still empty.
    static bool ensureLoaded();

    // Height in metres, or < 0 if unknown.
    static float heightMetresByDex(int dex);
    static float heightMetresByName(const std::string& name);

    // National dex for a display/species name, or -1 if unknown.
    static int dexByName(const std::string& name);
};

// Slugify for CSV identifier match: "Mr. Mime" -> "mr-mime", "Abomasnow" -> "abomasnow".
std::string pokemonSlug(const std::string& name);

// Find pokemon_heights.csv relative to common working directories.
std::string findPokemonHeightsCsv();

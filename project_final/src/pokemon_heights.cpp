#include "pokemon_heights.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <fstream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>

namespace {

std::mutex g_mutex;
bool g_attempted = false;
bool g_ok = false;
std::unordered_map<int, float> g_dex_to_m;           // dex -> metres
std::unordered_map<std::string, int> g_slug_to_dex;  // identifier -> dex

std::string trim(std::string s) {
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) {
        s.erase(s.begin());
    }
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) {
        s.pop_back();
    }
    return s;
}

bool loadCsv(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        return false;
    }

    std::string line;
    if (!std::getline(in, line)) {
        return false;
    }
    // Expect header dex,identifier,height_dm — tolerate BOM / extra columns.
    int loaded = 0;
    while (std::getline(in, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') {
            continue;
        }
        const std::size_t c1 = line.find(',');
        if (c1 == std::string::npos) {
            continue;
        }
        const std::size_t c2 = line.find(',', c1 + 1);
        if (c2 == std::string::npos) {
            continue;
        }
        const std::string dex_s = trim(line.substr(0, c1));
        const std::string ident = trim(line.substr(c1 + 1, c2 - c1 - 1));
        // height_dm may be followed by more columns; take the next field only.
        std::string dm_s = line.substr(c2 + 1);
        const std::size_t c3 = dm_s.find(',');
        if (c3 != std::string::npos) {
            dm_s = dm_s.substr(0, c3);
        }
        dm_s = trim(dm_s);
        if (dex_s.empty() || ident.empty() || dm_s.empty()) {
            continue;
        }
        int dex = 0;
        int height_dm = 0;
        try {
            dex = std::stoi(dex_s);
            height_dm = std::stoi(dm_s);
        } catch (...) {
            continue;
        }
        if (dex <= 0 || height_dm <= 0) {
            continue;
        }
        g_dex_to_m[dex] = static_cast<float>(height_dm) / 10.f;
        g_slug_to_dex[ident] = dex;
        ++loaded;
    }
    return loaded > 0;
}

}  // namespace

std::string pokemonSlug(const std::string& name) {
    std::string out;
    for (unsigned char c : name) {
        if (std::isalnum(c) != 0) {
            out.push_back(static_cast<char>(std::tolower(c)));
        } else if (c == ' ' || c == '_' || c == '-' || c == '.') {
            if (!out.empty() && out.back() != '-') {
                out.push_back('-');
            }
        }
    }
    while (!out.empty() && out.back() == '-') {
        out.pop_back();
    }
    return out;
}

std::string findPokemonHeightsCsv() {
    const char* candidates[] = {
        "data/pokemon_heights.csv",
        "../data/pokemon_heights.csv",
        "../../data/pokemon_heights.csv",
    };
    for (const char* path : candidates) {
        std::ifstream probe(path);
        if (probe.is_open()) {
            return path;
        }
    }
    return "";
}

bool PokemonHeights::ensureLoaded() {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_attempted) {
        return g_ok;
    }
    g_attempted = true;
    const std::string path = findPokemonHeightsCsv();
    if (path.empty()) {
        std::fprintf(stderr,
                     "warning: pokemon_heights.csv not found "
                     "(tried data/, ../data/, ../../data/)\n");
        g_ok = false;
        return false;
    }
    g_ok = loadCsv(path);
    if (!g_ok) {
        std::fprintf(stderr, "warning: failed to parse %s\n", path.c_str());
    } else {
        std::printf("Loaded %zu Pokédex heights from %s\n", g_dex_to_m.size(), path.c_str());
    }
    return g_ok;
}

float PokemonHeights::heightMetresByDex(int dex) {
    ensureLoaded();
    std::lock_guard<std::mutex> lock(g_mutex);
    const auto it = g_dex_to_m.find(dex);
    return it == g_dex_to_m.end() ? -1.f : it->second;
}

int PokemonHeights::dexByName(const std::string& name) {
    ensureLoaded();
    const std::string slug = pokemonSlug(name);
    if (slug.empty()) {
        return -1;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    const auto it = g_slug_to_dex.find(slug);
    return it == g_slug_to_dex.end() ? -1 : it->second;
}

float PokemonHeights::heightMetresByName(const std::string& name) {
    const int dex = dexByName(name);
    if (dex <= 0) {
        return -1.f;
    }
    return heightMetresByDex(dex);
}

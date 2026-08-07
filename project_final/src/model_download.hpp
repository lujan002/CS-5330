#pragma once

#include "model_library.hpp"

#include <map>
#include <string>
#include <vector>

// Fetches Pokemon XY models from The Models Resource at runtime.
//
// Archives are extracted into RAM-backed scratch space (/dev/shm) and deleted
// when the downloader goes out of scope, so nothing is written to persistent
// storage and downloaded models only exist for the life of the process.
class ModelDownloader {
public:
    ~ModelDownloader();

    // Downloads the asset index on first use. Returns false and sets error on failure.
    bool fetchIndex(std::string& error);

    // Resolves name against the index, downloads and unpacks the archive, and
    // returns every model it contains (a Venusaur archive yields both genders).
    bool download(
        const std::string& pokemon_name,
        std::vector<ModelEntry>& out_models,
        std::string& error);

    // Index entries whose name contains the query, for "did you mean" messages.
    std::vector<std::string> suggest(const std::string& query, std::size_t limit = 8) const;

    bool indexReady() const { return !entries_.empty(); }
    std::size_t indexSize() const { return entries_.size(); }

private:
    struct IndexEntry {
        std::string display;  // "#0003 Venusaur"
        std::string clean;    // "Venusaur"
        long asset_id = 0;
    };

    std::vector<IndexEntry> entries_;
    std::vector<std::string> temp_dirs_;
};

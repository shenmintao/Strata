// --serve's parked conversations on disk: the RAM conversation cache's second tier (--conversation-disk-dir).
//
// A conversation the RAM cache evicts (or one too large for its budget) is written as a session file
// (conversation_file.hpp: identity-bound, hashed, written to a temporary name and renamed) when it is at least
// --conversation-disk-min-tokens long; a request whose prompt one of them continues further than anything in RAM reads it
// back instead of reading the prompt again.  Beside each file a small index file (".meta") keeps what matching needs - the
// token ids and picture keys of the live state and of the checkpoint the file holds - so the tier survives a restart
// without opening the session files.  Writes are bounded: only long conversations, an identical one is not written again,
// an older turn of the same conversation is replaced, and the folder is kept under --conversation-disk-gib (least recently
// used first).
#pragma once

#include "strata/core/conversation_cache.hpp"
#include "strata/core/conversation_file.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

namespace strata::core {

struct ConversationDiskEntry {
    std::string base;                              // the files' path without the extension (.session / .meta)
    ConversationCheckpoint live;                   // ids and picture keys only
    std::vector<ConversationCheckpoint> checks;    // ids and picture keys only
    bool cvec = true;
    uint64_t bytes = 0;                            // the session file's size
    uint64_t used = 0;                             // unix seconds of the last write or read
};

class ConversationDisk {
public:
    ConversationDisk(std::string dir, uint64_t cap_bytes, int64_t min_tokens)
        : dir_(std::move(dir)), cap_(cap_bytes), min_tokens_(min_tokens) {}

    bool enabled() const { return !dir_.empty() && cap_ > 0; }
    bool wants(int64_t tokens) const { return enabled() && tokens >= std::max<int64_t>(min_tokens_, 1); }
    size_t size() const { return entries_.size(); }
    uint64_t bytes() const {
        uint64_t n = 0;
        for (const auto& e : entries_) n += e.bytes;
        return n;
    }
    const std::string& dir() const { return dir_; }

    // The index from the folder: every .meta with its .session; a file without its partner goes.
    bool load(std::string& error) {
        namespace fs = std::filesystem;
        std::error_code ec;
        fs::create_directories(dir_, ec);
        if (ec) { error = "cannot create " + dir_ + ": " + ec.message(); return false; }
        entries_.clear();
        for (const auto& it : fs::directory_iterator(dir_, ec)) {
            const fs::path p = it.path();
            if (p.extension() != ".meta") continue;
            const std::string base = (p.parent_path() / p.stem()).string();
            ConversationDiskEntry e;
            std::error_code ec2;
            const uint64_t size = (uint64_t) fs::file_size(base + ".session", ec2);
            if (ec2 || !read_meta(p.string(), e)) { remove_files(base); continue; }
            e.base = base;
            e.bytes = size;
            entries_.push_back(std::move(e));
        }
        for (const auto& it : fs::directory_iterator(dir_, ec)) {   // sessions whose index is gone
            const fs::path p = it.path();
            if (p.extension() != ".session") continue;
            std::error_code ec2;
            if (!fs::exists(p.parent_path() / (p.stem().string() + ".meta"), ec2)) fs::remove(p, ec2);
        }
        return true;
    }

    template<class Token>
    ConversationCache::Match best(const std::vector<Token>& prompt, const std::vector<ConversationImageKey>& images,
                                  bool cvec) const {
        ConversationCache::Match best;
        for (size_t i = 0; i < entries_.size(); ++i) {
            const auto& e = entries_[i];
            if (e.cvec != cvec) continue;
            auto consider = [&](const ConversationCheckpoint& c, bool live) {
                const int64_t n = conversation_prefix(c, prompt, images);
                if (n > best.tokens) best = {i, n, live};
            };
            consider(e.live, true);
            for (const auto& c : e.checks) consider(c, false);
        }
        return best;
    }

    // Writes `image` (keeping only its deepest checkpoint, the next turn's resume point).  `held`: an identical
    // conversation was already on disk (nothing written).  An older turn of the same conversation - one whose live
    // tokens this one starts with - is replaced, and the folder is trimmed to the cap afterwards.
    bool write(SavedConversation&& image, const SessionFileIdentity& id, std::string& error, size_t& written,
               bool& held) {
        written = 0;
        held = false;
        for (auto& e : entries_)
            if (e.cvec == image.cvec && e.live.ids == image.live.ids && e.live.imgs == image.live.imgs) {
                e.used = now();
                write_meta(e);
                held = true;
                return true;
            }
        image.checkpoints = session_checkpoints_to_save(image.checkpoints);
        ConversationDiskEntry e;
        e.live.ids = image.live.ids;
        e.live.imgs = image.live.imgs;
        for (const auto& c : image.checkpoints) {
            ConversationCheckpoint k;
            k.ids = c.ids;
            k.imgs = c.imgs;
            e.checks.push_back(std::move(k));
        }
        e.cvec = image.cvec;
        e.used = now();
        char name[64];
        std::snprintf(name, sizeof name, "conv-%016llx-%llu",
                      (unsigned long long) session_hash64(image.live.ids.data(), image.live.ids.size() * sizeof(int32_t), 7),
                      (unsigned long long) std::chrono::steady_clock::now().time_since_epoch().count());
        e.base = (std::filesystem::path(dir_) / name).string();
        SessionWriteOptions opt;
        opt.min_free_bytes = uint64_t(32) << 30;   // never fill the disk: refused below 32 GiB free
        opt.durable = false;                       // a cache: a file lost to a power cut only costs a prompt read
        size_t bytes = 0;
        if (!session_file_write(e.base + ".session", image, id, bytes, error, opt)) return false;
        e.bytes = bytes;
        if (!write_meta(e)) {
            remove_files(e.base);
            error = "cannot write the index file " + e.base + ".meta";
            return false;
        }
        written = bytes;
        // the same conversation a turn or more back: the new file continues it
        for (size_t i = 0; i < entries_.size();) {
            const auto& o = entries_[i];
            if (o.cvec == e.cvec && o.live.ids.size() < e.live.ids.size() &&
                std::equal(o.live.ids.begin(), o.live.ids.end(), e.live.ids.begin()) &&
                o.live.imgs.size() <= e.live.imgs.size() &&
                std::equal(o.live.imgs.begin(), o.live.imgs.end(), e.live.imgs.begin()))
                remove(i);
            else
                ++i;
        }
        entries_.push_back(std::move(e));
        trim();
        return true;
    }

    // The session file of entry `index` (fully validated by session_file_read).  A file that cannot be read or does not
    // belong to this engine any more (another model or engine version) goes.
    bool read(size_t index, const SessionFileIdentity& id, SavedConversation& image, const SessionReadLimits& limits,
              std::string& error, size_t& bytes, SessionStatus* status = nullptr) {
        auto& e = entries_.at(index);
        if (!session_file_read(e.base + ".session", id, image, bytes, error, limits, status)) {
            if (!status || status->error != SessionError::memory) remove(index);   // short of RAM: try another time
            return false;
        }
        e.used = now();
        write_meta(e);
        return true;
    }

    void remove(size_t index) {
        remove_files(entries_.at(index).base);
        entries_.erase(entries_.begin() + (std::ptrdiff_t) index);
    }

private:
    static uint64_t now() {
        return (uint64_t) std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
    }
    static void remove_files(const std::string& base) {
        std::error_code ec;
        std::filesystem::remove(base + ".session", ec);
        std::filesystem::remove(base + ".meta", ec);
    }
    void trim() {
        while (entries_.size() > 1 && bytes() > cap_) {
            size_t lru = 0;
            for (size_t i = 1; i + 1 < entries_.size(); ++i)   // never the newest (the last one)
                if (entries_[i].used < entries_[lru].used) lru = i;
            remove(lru);
        }
    }

    // .meta: "SCD1", cvec, used, the live ids/pictures, the checkpoints' ids/pictures; written to a temporary name
    // and renamed, so a reader never sees half of one
    static void put_list(std::ofstream& f, const ConversationCheckpoint& c) {
        const uint64_t n = c.ids.size(), m = c.imgs.size();
        f.write((const char*) &n, 8);
        f.write((const char*) c.ids.data(), (std::streamsize) (n * sizeof(int32_t)));
        f.write((const char*) &m, 8);
        for (const auto& k : c.imgs) {
            f.write((const char*) &k.start, 8);
            f.write((const char*) &k.hash, 8);
        }
    }
    static bool get_list(std::ifstream& f, ConversationCheckpoint& c, uint64_t left) {
        uint64_t n = 0, m = 0;
        if (!f.read((char*) &n, 8) || n > left / sizeof(int32_t)) return false;
        c.ids.resize(n);
        if (!f.read((char*) c.ids.data(), (std::streamsize) (n * sizeof(int32_t)))) return false;
        if (!f.read((char*) &m, 8) || m > left / 16) return false;
        c.imgs.resize(m);
        for (auto& k : c.imgs)
            if (!f.read((char*) &k.start, 8) || !f.read((char*) &k.hash, 8)) return false;
        return true;
    }
    static bool write_meta(const ConversationDiskEntry& e) {
        const std::string tmp = e.base + ".meta.tmp", path = e.base + ".meta";
        {
            std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
            if (!f) return false;
            f.write("SCD1", 4);
            const uint8_t cvec = e.cvec ? 1 : 0;
            f.write((const char*) &cvec, 1);
            f.write((const char*) &e.used, 8);
            put_list(f, e.live);
            const uint64_t k = e.checks.size();
            f.write((const char*) &k, 8);
            for (const auto& c : e.checks) put_list(f, c);
            if (!f.flush()) return false;
        }
        std::error_code ec;
        std::filesystem::rename(tmp, path, ec);
        return !ec;
    }
    static bool read_meta(const std::string& path, ConversationDiskEntry& e) {
        std::error_code ec;
        const uint64_t size = (uint64_t) std::filesystem::file_size(path, ec);
        if (ec || size < 13) return false;
        std::ifstream f(path, std::ios::binary);
        char magic[4];
        uint8_t cvec = 0;
        if (!f.read(magic, 4) || std::string(magic, 4) != "SCD1" || !f.read((char*) &cvec, 1) ||
            !f.read((char*) &e.used, 8))
            return false;
        e.cvec = cvec != 0;
        if (!get_list(f, e.live, size)) return false;
        uint64_t k = 0;
        if (!f.read((char*) &k, 8) || k > 64) return false;
        e.checks.resize(k);
        for (auto& c : e.checks)
            if (!get_list(f, c, size)) return false;
        return true;
    }

    std::string dir_;
    uint64_t cap_ = 0;
    int64_t min_tokens_ = 0;
    std::vector<ConversationDiskEntry> entries_;
};

} // namespace strata::core

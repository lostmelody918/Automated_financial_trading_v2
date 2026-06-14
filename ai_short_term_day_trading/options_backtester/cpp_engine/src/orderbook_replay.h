#pragma once

#include <vector>
#include <string>
#include <map>
#include <stdexcept>
#include <cmath>

namespace backtester {

struct Tick {
    int64_t timestamp_ms;
    double price;
    int volume;
    double bid_price;
    int bid_volume;
    double ask_price;
    int ask_volume;
};

class OrderBookReplay {
public:
    OrderBookReplay(const std::string& symbol);
    
    // Process a new tick and update the order book state
    void process_tick(const Tick& tick);
    
    // Calculate Mark-to-Market (MTM) value for a given position
    // position > 0 means long, < 0 means short
    double calculate_mtm(int position) const;
    
    // Getters
    double get_best_bid() const { return best_bid; }
    double get_best_ask() const { return best_ask; }
    double get_last_price() const { return last_price; }

private:
    std::string symbol_;
    double last_price;
    double best_bid;
    int best_bid_vol;
    double best_ask;
    int best_ask_vol;
    int64_t current_time_ms;
};

class SimulationEngine {
public:
    SimulationEngine();
    
    void add_contract(const std::string& symbol);
    void feed_ticks(const std::string& symbol, const std::vector<Tick>& ticks);
    
    // Step the simulation to a specific time
    void advance_to(int64_t target_time_ms);
    
    double get_contract_mtm(const std::string& symbol, int position) const;
    
    // Expose individual contract state queries for strategy backtesting
    double get_best_bid(const std::string& symbol) const;
    double get_best_ask(const std::string& symbol) const;
    double get_last_price(const std::string& symbol) const;

private:
    std::map<std::string, OrderBookReplay> orderbooks_;
    std::map<std::string, std::vector<Tick>> tick_queues_;
    std::map<std::string, size_t> queue_indices_;
    int64_t current_sim_time_ms_;
};

} // namespace backtester

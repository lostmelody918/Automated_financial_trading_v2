#include "orderbook_replay.h"
#include <algorithm>
#include <iostream>

namespace backtester {

OrderBookReplay::OrderBookReplay(const std::string& symbol) 
    : symbol_(symbol), last_price(0.0), best_bid(0.0), best_bid_vol(0), 
      best_ask(0.0), best_ask_vol(0), current_time_ms(0) {}

void OrderBookReplay::process_tick(const Tick& tick) {
    current_time_ms = tick.timestamp_ms;
    last_price = tick.price;
    best_bid = tick.bid_price;
    best_bid_vol = tick.bid_volume;
    best_ask = tick.ask_price;
    best_ask_vol = tick.ask_volume;
}

double OrderBookReplay::calculate_mtm(int position) const {
    if (position == 0) return 0.0;
    
    // If long, we would sell at the bid. If short, we would buy at the ask.
    if (position > 0) {
        return (best_bid > 0) ? (position * best_bid) : (position * last_price);
    } else {
        return (best_ask > 0) ? (position * best_ask) : (position * last_price);
    }
}

SimulationEngine::SimulationEngine() : current_sim_time_ms_(0) {}

void SimulationEngine::add_contract(const std::string& symbol) {
    if (orderbooks_.find(symbol) == orderbooks_.end()) {
        orderbooks_.emplace(symbol, OrderBookReplay(symbol));
        queue_indices_[symbol] = 0;
    }
}

void SimulationEngine::feed_ticks(const std::string& symbol, const std::vector<Tick>& ticks) {
    if (orderbooks_.find(symbol) == orderbooks_.end()) {
        add_contract(symbol);
    }
    tick_queues_[symbol] = ticks;
    queue_indices_[symbol] = 0;
}

void SimulationEngine::advance_to(int64_t target_time_ms) {
    current_sim_time_ms_ = target_time_ms;
    
    for (auto& pair : tick_queues_) {
        const std::string& symbol = pair.first;
        const std::vector<Tick>& queue = pair.second;
        size_t& idx = queue_indices_[symbol];
        OrderBookReplay& ob = orderbooks_.at(symbol);
        
        while (idx < queue.size() && queue[idx].timestamp_ms <= target_time_ms) {
            ob.process_tick(queue[idx]);
            idx++;
        }
    }
}

double SimulationEngine::get_contract_mtm(const std::string& symbol, int position) const {
    auto it = orderbooks_.find(symbol);
    if (it != orderbooks_.end()) {
        return it->second.calculate_mtm(position);
    }
    return 0.0;
}

double SimulationEngine::get_best_bid(const std::string& symbol) const {
    auto it = orderbooks_.find(symbol);
    if (it != orderbooks_.end()) return it->second.get_best_bid();
    return 0.0;
}

double SimulationEngine::get_best_ask(const std::string& symbol) const {
    auto it = orderbooks_.find(symbol);
    if (it != orderbooks_.end()) return it->second.get_best_ask();
    return 0.0;
}

double SimulationEngine::get_last_price(const std::string& symbol) const {
    auto it = orderbooks_.find(symbol);
    if (it != orderbooks_.end()) return it->second.get_last_price();
    return 0.0;
}

} // namespace backtester

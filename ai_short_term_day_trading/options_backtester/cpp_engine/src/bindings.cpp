#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "orderbook_replay.h"

namespace py = pybind11;
using namespace backtester;

PYBIND11_MODULE(options_replay, m) {
    m.doc() = "C++ High-Performance Options Order Book Replay Engine";

    py::class_<Tick>(m, "Tick")
        .def(py::init<>())
        .def_readwrite("timestamp_ms", &Tick::timestamp_ms)
        .def_readwrite("price", &Tick::price)
        .def_readwrite("volume", &Tick::volume)
        .def_readwrite("bid_price", &Tick::bid_price)
        .def_readwrite("bid_volume", &Tick::bid_volume)
        .def_readwrite("ask_price", &Tick::ask_price)
        .def_readwrite("ask_volume", &Tick::ask_volume);

    py::class_<OrderBookReplay>(m, "OrderBookReplay")
        .def(py::init<const std::string&>())
        .def("process_tick", &OrderBookReplay::process_tick)
        .def("calculate_mtm", &OrderBookReplay::calculate_mtm)
        .def_property_readonly("best_bid", &OrderBookReplay::get_best_bid)
        .def_property_readonly("best_ask", &OrderBookReplay::get_best_ask)
        .def_property_readonly("last_price", &OrderBookReplay::get_last_price);

    py::class_<SimulationEngine>(m, "SimulationEngine")
        .def(py::init<>())
        .def("add_contract", &SimulationEngine::add_contract)
        .def("feed_ticks", &SimulationEngine::feed_ticks)
        .def("advance_to", &SimulationEngine::advance_to)
        .def("get_contract_mtm", &SimulationEngine::get_contract_mtm)
        .def("get_best_bid", &SimulationEngine::get_best_bid)
        .def("get_best_ask", &SimulationEngine::get_best_ask)
        .def("get_last_price", &SimulationEngine::get_last_price);
}

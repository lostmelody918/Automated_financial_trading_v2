class Tick:
    def __init__(self):
        self.timestamp_ms = 0
        self.price = 0.0
        self.volume = 0
        self.bid_price = 0.0
        self.bid_volume = 0
        self.ask_price = 0.0
        self.ask_volume = 0

class OrderBookReplay:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.last_price = 0.0
        self.best_bid = 0.0
        self.best_bid_vol = 0
        self.best_ask = 0.0
        self.best_ask_vol = 0
        self.current_time_ms = 0

    def process_tick(self, tick: Tick):
        self.current_time_ms = tick.timestamp_ms
        self.last_price = tick.price
        self.best_bid = tick.bid_price
        self.best_bid_vol = tick.bid_volume
        self.best_ask = tick.ask_price
        self.best_ask_vol = tick.ask_volume

    def calculate_mtm(self, position: int) -> float:
        if position == 0:
            return 0.0
        if position > 0:
            return float(position * self.best_bid) if self.best_bid > 0 else float(position * self.last_price)
        else:
            return float(position * self.best_ask) if self.best_ask > 0 else float(position * self.last_price)

class SimulationEngine:
    def __init__(self):
        self.orderbooks = {}
        self.tick_queues = {}
        self.queue_indices = {}
        self.current_sim_time_ms = 0

    def add_contract(self, symbol: str):
        if symbol not in self.orderbooks:
            self.orderbooks[symbol] = OrderBookReplay(symbol)
            self.queue_indices[symbol] = 0

    def feed_ticks(self, symbol: str, ticks: list):
        if symbol not in self.orderbooks:
            self.add_contract(symbol)
        self.tick_queues[symbol] = ticks
        self.queue_indices[symbol] = 0

    def advance_to(self, target_time_ms: int):
        self.current_sim_time_ms = target_time_ms
        for symbol, queue in self.tick_queues.items():
            idx = self.queue_indices[symbol]
            ob = self.orderbooks[symbol]
            while idx < len(queue) and queue[idx].timestamp_ms <= target_time_ms:
                ob.process_tick(queue[idx])
                idx += 1
            self.queue_indices[symbol] = idx

    def get_contract_mtm(self, symbol: str, position: int) -> float:
        if symbol in self.orderbooks:
            return self.orderbooks[symbol].calculate_mtm(position)
        return 0.0

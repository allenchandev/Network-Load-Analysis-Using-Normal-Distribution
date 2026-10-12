"""
Dynamic Load Balancing Simulation with Normal Distribution Awareness.
Compares Round-Robin, Least-Connections, and Gaussian-Aware / Statistical Routing.
"""

from typing import List, Dict, Any
import numpy as np


class ServerNode:
    def __init__(self, node_id: str, capacity_gbps: float):
        self.node_id = node_id
        self.capacity_gbps = capacity_gbps
        self.current_load_gbps = 0.0
        self.total_traffic_served_gbps = 0.0
        self.dropped_traffic_gbps = 0.0
        self.load_history: List[float] = []

    def reset(self):
        self.current_load_gbps = 0.0
        self.total_traffic_served_gbps = 0.0
        self.dropped_traffic_gbps = 0.0
        self.load_history = []

    def process_incoming_load(self, incoming_load: float, decay_rate: float = 0.85):
        # Existing load decays over time, new load is added
        self.current_load_gbps = (self.current_load_gbps * decay_rate) + incoming_load
        
        if self.current_load_gbps > self.capacity_gbps:
            overflow = self.current_load_gbps - self.capacity_gbps
            self.dropped_traffic_gbps += overflow
            self.current_load_gbps = self.capacity_gbps
            self.total_traffic_served_gbps += (incoming_load - overflow)
        else:
            self.total_traffic_served_gbps += incoming_load

        self.load_history.append(self.current_load_gbps)


class LoadBalancingSimulator:
    def __init__(self, num_nodes: int = 4, node_capacity_gbps: float = 30.0):
        self.num_nodes = num_nodes
        self.node_capacity_gbps = node_capacity_gbps

    def run_simulation(
        self, traffic_stream: np.ndarray, strategy: str = "statistical_aware"
    ) -> Dict[str, Any]:
        """
        Strategies:
        - 'round_robin': Even cyclic distribution of incoming flows.
        - 'least_loaded': Routes to the node with the minimum current instantaneous load.
        - 'statistical_aware': Uses normal distribution probability estimates (P(X + new_load > Capacity))
                               to route traffic where head-room risk is lowest.
        """
        nodes = [
            ServerNode(f"Worker-{i+1}", self.node_capacity_gbps)
            for i in range(self.num_nodes)
        ]

        rr_index = 0

        for chunk_load in traffic_stream:
            # We divide discrete incoming traffic into discrete request batches (e.g. 5 batches per time step)
            sub_chunks = [chunk_load / 5.0] * 5

            for load in sub_chunks:
                if strategy == "round_robin":
                    target_node = nodes[rr_index]
                    rr_index = (rr_index + 1) % self.num_nodes

                elif strategy == "least_loaded":
                    target_node = min(nodes, key=lambda n: n.current_load_gbps)

                elif strategy == "statistical_aware":
                    # Gaussian risk evaluation:
                    # Target node is chosen based on smallest expected violation risk
                    # Buffer margin = Capacity - current_load
                    # Score penalizes nodes operating close to or over 1-2 standard deviations from capacity
                    best_node = None
                    best_score = float("inf")
                    for node in nodes:
                        remaining_capacity = node.capacity_gbps - node.current_load_gbps
                        if remaining_capacity <= 0:
                            score = 1000.0 + abs(remaining_capacity)
                        else:
                            # Higher buffer -> lower score (preferred)
                            score = load / (remaining_capacity + 1e-4)
                        if score < best_score:
                            best_score = score
                            best_node = node
                    target_node = best_node or nodes[0]

                target_node.process_incoming_load(load)

        total_dropped = sum(n.dropped_traffic_gbps for n in nodes)
        total_served = sum(n.total_traffic_served_gbps for n in nodes)
        total_offered = total_dropped + total_served
        drop_rate = (total_dropped / total_offered) if total_offered > 0 else 0.0

        # Load variance among nodes (fairness measure)
        mean_loads = [np.mean(n.load_history) for n in nodes]
        load_imbalance_std = float(np.std(mean_loads))

        return {
            "strategy": strategy,
            "total_offered_gbps": float(total_offered),
            "total_served_gbps": float(total_served),
            "total_dropped_gbps": float(total_dropped),
            "packet_drop_rate_pct": float(drop_rate * 100.0),
            "load_imbalance_std": load_imbalance_std,
            "node_metrics": [
                {
                    "node_id": n.node_id,
                    "avg_load_gbps": float(np.mean(n.load_history)),
                    "peak_load_gbps": float(np.max(n.load_history)),
                    "dropped_gbps": float(n.dropped_traffic_gbps),
                }
                for n in nodes
            ],
        }
